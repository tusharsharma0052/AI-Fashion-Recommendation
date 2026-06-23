"""
embedding_generator.py — Multi-modal Embedding Generation.

Generates:
  • Text embeddings via sentence-transformers (all-MiniLM-L6-v2)
  • Image embeddings via CLIP (openai/clip-vit-base-patch32)  [optional]
  • Combined / fused embeddings saved as .npy arrays

All embeddings are L2-normalised before saving so cosine similarity equals
dot-product similarity — required by FAISS IndexFlatIP.

Usage (standalone):
    python embedding_generator.py
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLIP_EMBED_DIM,
    CLIP_MODEL,
    COMBINED_EMBEDDINGS_FILE,
    EMBEDDINGS_DIR,
    IMAGE_EMBEDDINGS_FILE,
    PRODUCT_IDS_FILE,
    SENTENCE_MODEL,
    TEXT_EMBED_DIM,
    TEXT_EMBEDDINGS_FILE,
)
from dataset_analysis import build_fashion_catalog

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# LAZY MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

_sentence_model = None
_clip_processor = None
_clip_model = None


def _get_sentence_model():
    """Lazily load the sentence-transformers model."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers model: %s …", SENTENCE_MODEL)
        _sentence_model = SentenceTransformer(SENTENCE_MODEL)
        logger.info("Sentence model loaded ✓")
    return _sentence_model


def _get_clip_model():
    """Lazily load CLIP processor and model."""
    global _clip_processor, _clip_model
    if _clip_model is None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            logger.info("Loading CLIP model: %s …", CLIP_MODEL)
            _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            _clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
            _clip_model.eval()
            logger.info("CLIP model loaded ✓")
        except Exception as exc:
            logger.warning("Could not load CLIP model (%s). Image embeddings disabled.", exc)
            return None, None
    return _clip_processor, _clip_model


# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row of a 2-D float32 array in place.
    Rows with zero norm are left as zero vectors.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
    return (matrix / norms).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# TEXT EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def generate_text_embedding(text: str) -> np.ndarray:
    """
    Generate an L2-normalised text embedding for a single string.

    Parameters
    ----------
    text : str
        Any fashion-related query or product description.

    Returns
    -------
    np.ndarray of shape (TEXT_EMBED_DIM,), float32
    """
    model = _get_sentence_model()
    vec = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)


def generate_text_embeddings_batch(
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    """
    Batch-generate L2-normalised text embeddings.

    Parameters
    ----------
    texts      : list of N strings
    batch_size : sentences processed at once

    Returns
    -------
    np.ndarray of shape (N, TEXT_EMBED_DIM), float32
    """
    model = _get_sentence_model()
    logger.info("Generating text embeddings for %d items …", len(texts))
    matrix = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return matrix.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def generate_image_embedding(image_path: str | Path) -> Optional[np.ndarray]:
    """
    Generate an L2-normalised CLIP image embedding from a file path.

    Returns None if CLIP is unavailable or the image cannot be opened.
    """
    import torch
    from PIL import Image

    processor, model = _get_clip_model()
    if model is None:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feat = model.get_image_features(**inputs)
        vec = feat.squeeze().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec
    except Exception as exc:
        logger.debug("Image embedding failed for %s: %s", image_path, exc)
        return None


def generate_image_embeddings_batch(
    image_paths: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Batch-generate CLIP image embeddings.  Returns zero-filled rows for
    missing / unreadable images.

    Returns
    -------
    np.ndarray of shape (N, CLIP_EMBED_DIM), float32
    """
    import torch
    from PIL import Image

    processor, model = _get_clip_model()
    n = len(image_paths)
    matrix = np.zeros((n, CLIP_EMBED_DIM), dtype=np.float32)

    if model is None:
        logger.warning("CLIP unavailable — returning zero image embeddings.")
        return matrix

    logger.info("Generating image embeddings for %d items …", n)
    for start in tqdm(range(0, n, batch_size), desc="Image batches"):
        batch_paths = image_paths[start : start + batch_size]
        imgs, valid_indices = [], []
        for j, p in enumerate(batch_paths):
            try:
                imgs.append(Image.open(p).convert("RGB"))
                valid_indices.append(start + j)
            except Exception:
                pass  # leave row as zeros

        if not imgs:
            continue

        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = model.get_image_features(**inputs).numpy().astype(np.float32)

        for k, global_idx in enumerate(valid_indices):
            v = feats[k]
            norm = np.linalg.norm(v)
            matrix[global_idx] = v / norm if norm > 0 else v

    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# MULTIMODAL (FUSED) EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def generate_multimodal_embedding(
    text: str,
    image_path: Optional[str] = None,
    text_weight: float = 0.7,
    image_weight: float = 0.3,
) -> np.ndarray:
    """
    Fuse a text embedding and an image embedding into a single vector.

    When no image is available (or CLIP is disabled), the text embedding
    is returned unchanged.

    Both modalities are L2-normalised individually before fusion.
    The final fused vector is also L2-normalised.

    Parameters
    ----------
    text         : product search text
    image_path   : path to the product image (optional)
    text_weight  : weight for text modality (default 0.7)
    image_weight : weight for image modality (default 0.3)

    Returns
    -------
    np.ndarray of shape (TEXT_EMBED_DIM,) or fused shape, float32
    """
    text_vec = generate_text_embedding(text)

    if image_path:
        img_vec = generate_image_embedding(image_path)
    else:
        img_vec = None

    if img_vec is None:
        return text_vec

    # Pad or truncate so both vectors share the same dim for weighted sum
    # Simple approach: use text dim only (project image via mean or truncation)
    if len(img_vec) != len(text_vec):
        # Pad image vec to text dim with zeros
        padded = np.zeros(len(text_vec), dtype=np.float32)
        padded[: min(len(img_vec), len(text_vec))] = img_vec[: len(text_vec)]
        img_vec = padded

    fused = text_weight * text_vec + image_weight * img_vec
    norm = np.linalg.norm(fused)
    return (fused / norm).astype(np.float32) if norm > 0 else fused


# ══════════════════════════════════════════════════════════════════════════════
# FULL CATALOG EMBEDDING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_and_save_embeddings(
    products: pd.DataFrame,
    use_clip: bool = False,
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Build embeddings for all products and persist them to disk.

    Steps:
      1. Generate text embeddings from `search_text` column
      2. Optionally generate CLIP image embeddings
      3. Fuse if both are available; otherwise use text only
      4. Save embeddings + product IDs as .npy files

    Parameters
    ----------
    products      : enriched product DataFrame (from dataset_analysis)
    use_clip      : whether to include image embeddings via CLIP
    force_rebuild : ignore cached embeddings and recompute

    Returns
    -------
    np.ndarray of shape (N_products, embed_dim), float32 — the combined matrix
    """
    if not force_rebuild and COMBINED_EMBEDDINGS_FILE.exists():
        logger.info("Loading cached embeddings from %s …", COMBINED_EMBEDDINGS_FILE)
        return np.load(str(COMBINED_EMBEDDINGS_FILE))

    texts = products["search_text"].tolist()
    product_ids = products["id"].tolist()

    # ── Text embeddings ────────────────────────────────────────────────────
    text_matrix = generate_text_embeddings_batch(texts)
    np.save(str(TEXT_EMBEDDINGS_FILE), text_matrix)
    logger.info("Saved text embeddings → %s", TEXT_EMBEDDINGS_FILE)

    combined_matrix = text_matrix

    # ── Image embeddings (optional) ────────────────────────────────────────
    if use_clip:
        image_paths = products["image_path"].tolist()
        img_matrix = generate_image_embeddings_batch(image_paths)
        np.save(str(IMAGE_EMBEDDINGS_FILE), img_matrix)
        logger.info("Saved image embeddings → %s", IMAGE_EMBEDDINGS_FILE)

        # Fuse: weighted average where image dim != text dim we pad image matrix
        if img_matrix.shape[1] != text_matrix.shape[1]:
            logger.info(
                "Image dim %d ≠ text dim %d — using text-only embeddings.",
                img_matrix.shape[1],
                text_matrix.shape[1],
            )
        else:
            # Check which rows have non-zero image embeddings
            has_image = np.linalg.norm(img_matrix, axis=1) > 0
            combined_matrix = np.where(
                has_image[:, np.newaxis],
                0.7 * text_matrix + 0.3 * img_matrix,
                text_matrix,
            ).astype(np.float32)
            combined_matrix = _l2_normalise(combined_matrix)
            logger.info(
                "Fused image+text for %d/%d products with images.",
                has_image.sum(), len(products),
            )

    # ── Save final combined embeddings + product ID index ──────────────────
    np.save(str(COMBINED_EMBEDDINGS_FILE), combined_matrix)
    np.save(str(PRODUCT_IDS_FILE), np.array(product_ids, dtype=str))
    logger.info(
        "Combined embeddings saved → %s  shape=%s",
        COMBINED_EMBEDDINGS_FILE, combined_matrix.shape,
    )
    return combined_matrix


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    """
    Load pre-built embeddings from disk.

    Returns
    -------
    (matrix, product_ids)
        matrix      : np.ndarray of shape (N, D)
        product_ids : list of product ID strings matching each row
    """
    if not COMBINED_EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            "Embeddings not found. Run `python embedding_generator.py` first."
        )
    matrix = np.load(str(COMBINED_EMBEDDINGS_FILE))
    ids = np.load(str(PRODUCT_IDS_FILE), allow_pickle=True).tolist()
    return matrix.astype(np.float32), ids


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    products, _, _ = build_fashion_catalog()
    logger.info("Building embeddings for %d products …", len(products))

    # Try with CLIP; fall back gracefully
    matrix = build_and_save_embeddings(products, use_clip=True, force_rebuild=True)
    logger.info("Done. Embedding matrix shape: %s", matrix.shape)
