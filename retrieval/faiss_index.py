"""
retrieval/faiss_index.py — FAISS Vector Index for Fashion Retrieval.

Responsibilities:
  • Build a FAISS IVFFlat (or Flat fallback) index from product embeddings
  • Persist / reload the index and associated metadata
  • Expose search_similar_items() for per-category nearest-neighbour retrieval

All embeddings stored in the index are L2-normalised so inner-product search
equals cosine similarity.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COMBINED_EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    FAISS_N_RESULTS,
    FAISS_NPROBE,
    PRODUCT_IDS_FILE,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# INDEX BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_index(
    embeddings: np.ndarray,
    product_ids: list[str],
    products_df: pd.DataFrame,
    n_list: int = 50,
    force_rebuild: bool = False,
) -> Any:
    """
    Build and return a FAISS index over the provided embedding matrix.

    An IVFFlat index is preferred for speed; falls back to IndexFlatIP when
    the dataset is too small to train IVF clusters.

    Parameters
    ----------
    embeddings    : float32 array of shape (N, D) — L2-normalised
    product_ids   : list of N product ID strings
    products_df   : full product DataFrame (persisted to metadata pickle)
    n_list        : number of IVF voronoi cells
    force_rebuild : if False and a saved index exists, load from disk

    Returns
    -------
    faiss.Index
    """
    import faiss  # type: ignore

    if not force_rebuild and FAISS_INDEX_FILE.exists() and FAISS_META_FILE.exists():
        logger.info("FAISS index already exists — loading from disk.")
        return load_index()

    N, D = embeddings.shape
    logger.info("Building FAISS index: N=%d  D=%d …", N, D)

    # Choose index type based on dataset size
    use_ivf = N >= max(n_list * 10, 256)

    if use_ivf:
        quantiser = faiss.IndexFlatIP(D)
        index = faiss.IndexIVFFlat(quantiser, D, min(n_list, N // 10),
                                    faiss.METRIC_INNER_PRODUCT)
        logger.info("Training IVFFlat index with %d lists …", min(n_list, N // 10))
        index.train(embeddings)
        index.nprobe = FAISS_NPROBE
    else:
        logger.info("Dataset small (%d) — using IndexFlatIP.", N)
        index = faiss.IndexFlatIP(D)

    index.add(embeddings)
    logger.info("FAISS index built with %d vectors ✓", index.ntotal)

    save_index(index, product_ids, products_df)
    return index


def save_index(
    index: Any,
    product_ids: list[str],
    products_df: pd.DataFrame,
) -> None:
    """Persist the FAISS index and associated metadata to disk."""
    import faiss  # type: ignore

    FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    meta = {
        "product_ids": product_ids,
        "products_df": products_df.to_dict(orient="records"),
    }
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(meta, f)

    logger.info("FAISS index saved → %s", FAISS_INDEX_FILE)
    logger.info("Metadata saved → %s", FAISS_META_FILE)


def load_index() -> tuple[Any, list[str], pd.DataFrame]:
    """
    Load the FAISS index and metadata from disk.

    Returns
    -------
    (index, product_ids, products_df)
    """
    import faiss  # type: ignore

    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run `python retrieval/faiss_index.py` first."
        )

    index = faiss.read_index(str(FAISS_INDEX_FILE))
    # Re-set nprobe after loading (not persisted by faiss.write_index for IVF)
    if hasattr(index, "nprobe"):
        index.nprobe = FAISS_NPROBE

    with open(FAISS_META_FILE, "rb") as f:
        meta = pickle.load(f)

    product_ids: list[str] = meta["product_ids"]
    products_df = pd.DataFrame(meta["products_df"])

    logger.info(
        "FAISS index loaded: %d vectors, dim=%d", index.ntotal, index.d
    )
    return index, product_ids, products_df


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def search_similar_items(
    query_embedding: np.ndarray,
    index: Any,
    product_ids: list[str],
    products_df: pd.DataFrame,
    category: Optional[str] = None,
    gender_filter: Optional[str] = None,
    k: int = FAISS_N_RESULTS,
    over_fetch_factor: int = 5,
) -> pd.DataFrame:
    """
    Retrieve the top-k most similar products for a query embedding.

    Post-retrieval filters are applied for category and gender so the caller
    receives exactly what it needs.

    Parameters
    ----------
    query_embedding    : 1-D float32 array (L2-normalised)
    index              : loaded FAISS index
    product_ids        : ordered list of IDs matching each FAISS row
    products_df        : full product metadata DataFrame
    category           : 'Topwear' | 'Bottomwear' | 'Footwear' | 'Accessories' | None
    gender_filter      : 'men' | 'women' | 'unisex' | None
    k                  : number of results to return after filtering
    over_fetch_factor  : multiplier to fetch before filtering (ensures k results survive)

    Returns
    -------
    pd.DataFrame  — up to k rows from products_df, sorted by similarity score descending
    """
    query = query_embedding.reshape(1, -1).astype(np.float32)

    # Fetch more candidates than k to absorb filtering losses
    fetch_k = min(k * over_fetch_factor, index.ntotal)
    scores, indices = index.search(query, fetch_k)

    rows: list[dict] = []
    id_to_row: dict[str, dict] = {
        row["id"]: row for row in products_df.to_dict(orient="records")
    }

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(product_ids):
            continue
        pid = product_ids[idx]
        product = id_to_row.get(pid)
        if product is None:
            continue
        rows.append({**product, "similarity_score": float(score)})

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    # ── Category filter ────────────────────────────────────────────────────
    if category and "norm_category" in results.columns:
        cat_mask = results["norm_category"].str.lower() == category.lower()
        results = results[cat_mask]

    # ── Gender filter ──────────────────────────────────────────────────────
    if gender_filter and "gender" in results.columns:
        g = gender_filter.lower()
        gender_mask = (
            results["gender"].str.lower().str.contains(g, na=False)
            | results["gender"].str.lower().str.contains("unisex", na=False)
        )
        results = results[gender_mask]

    # Return top-k by similarity
    results = (
        results
        .sort_values("similarity_score", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER  (used by chat_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

class FashionRetriever:
    """
    Stateful wrapper that holds the FAISS index + metadata in memory and
    exposes a clean interface for the chat engine.
    """

    def __init__(self) -> None:
        self.index = None
        self.product_ids: list[str] = []
        self.products_df = pd.DataFrame()
        self._loaded = False

    # ------------------------------------------------------------------
    def load(self) -> "FashionRetriever":
        """Load index from disk (idempotent)."""
        if not self._loaded:
            self.index, self.product_ids, self.products_df = load_index()
            self._loaded = True
        return self

    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: np.ndarray,
        category: Optional[str] = None,
        gender: Optional[str] = None,
        k: int = FAISS_N_RESULTS,
    ) -> pd.DataFrame:
        """Retrieve products matching the query embedding."""
        if not self._loaded:
            self.load()
        return search_similar_items(
            query_embedding,
            self.index,
            self.product_ids,
            self.products_df,
            category=category,
            gender_filter=gender,
            k=k,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    from dataset_analysis import build_fashion_catalog
    from embedding_generator import build_and_save_embeddings, load_embeddings

    products, _, _ = build_fashion_catalog()

    # Build or load embeddings
    if COMBINED_EMBEDDINGS_FILE.exists():
        embeddings, product_ids = load_embeddings()
    else:
        embeddings = build_and_save_embeddings(products, use_clip=False)
        product_ids = products["id"].tolist()

    index = build_index(
        embeddings,
        product_ids,
        products,
        force_rebuild=True,
    )
    logger.info("FAISS index ready with %d vectors.", index.ntotal)

    # Quick sanity-check search
    from embedding_generator import generate_text_embedding

    q_emb = generate_text_embedding("casual summer shirt for men")
    results = search_similar_items(q_emb, index, product_ids, products,
                                    category="Topwear", gender_filter="men", k=5)
    print("\nTop-5 similar Topwear items:")
    print(results[["id", "name", "norm_category", "gender", "similarity_score"]].to_string())
