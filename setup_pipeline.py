"""
setup_pipeline.py — One-shot pipeline to prepare all data artefacts.

Run this ONCE after cloning the repo and placing the dataset files:

    python setup_pipeline.py [--clip] [--force]

Steps:
  1. Dataset analysis & validation
  2. Embedding generation (text; optionally CLIP image)
  3. FAISS index building

After this script completes, run the app:
    streamlit run app.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Fashion Assistant — Data Pipeline Setup"
    )
    parser.add_argument(
        "--clip", action="store_true",
        help="Include CLIP image embeddings (requires transformers + torch)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force rebuild even if artefacts already exist",
    )
    args = parser.parse_args()

    # ── Step 1: Dataset analysis ───────────────────────────────────────────
    logger.info("=" * 55)
    logger.info("STEP 1 — Dataset Analysis")
    logger.info("=" * 55)
    from dataset_analysis import build_fashion_catalog, validate_catalog, print_statistics
    products, outfits, stats = build_fashion_catalog()
    warnings = validate_catalog(products)
    print_statistics(stats, warnings)

    if products.empty:
        logger.error("No products loaded. Check that data/products.csv exists.")
        sys.exit(1)

    # ── Step 2: Embedding generation ───────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("STEP 2 — Embedding Generation (use_clip=%s)", args.clip)
    logger.info("=" * 55)
    from embedding_generator import build_and_save_embeddings
    embeddings = build_and_save_embeddings(
        products, use_clip=args.clip, force_rebuild=args.force
    )
    logger.info("Embeddings shape: %s", embeddings.shape)

    # ── Step 3: FAISS index ────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("STEP 3 — FAISS Index Building")
    logger.info("=" * 55)
    from retrieval.faiss_index import build_index
    product_ids = products["id"].tolist()
    index = build_index(
        embeddings, product_ids, products, force_rebuild=args.force
    )
    logger.info("FAISS index size: %d vectors", index.ntotal)

    # ── Done ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("✓ Setup complete! Start the app:")
    logger.info("  streamlit run app.py")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
