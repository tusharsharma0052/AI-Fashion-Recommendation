"""
dataset_analysis.py — Dataset Analysis & Catalog Builder.

Responsibilities:
  • Load and validate products.csv + outfits.csv
  • Clean and normalise metadata fields
  • Map raw category_label → normalised super-category (Topwear / Bottomwear /
    Footwear / Accessories)
  • Validate image paths
  • Generate category, gender, occasion, and colour statistics
  • Produce a clean, enriched fashion catalog DataFrame ready for downstream use

Usage (standalone):
    python dataset_analysis.py
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Make sure the project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CATEGORY_MAP,
    COLOR_HARMONY,
    DATA_DIR,
    IMAGES_DIR,
    OCCASION_MAP,
    PRODUCTS_CSV,
    OUTFITS_CSV,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_str(value: object) -> str:
    """Lowercase, strip, and collapse whitespace."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _extract_dominant_color(text: str) -> str:
    """Heuristically extract the first recognised colour mention from text."""
    known_colors = list(COLOR_HARMONY.keys()) + [
        "denim", "khaki", "tan", "camel", "rust", "cream", "lavender",
        "teal", "mustard", "coral", "mint", "burgundy", "charcoal",
    ]
    text_lower = text.lower()
    for color in known_colors:
        if color in text_lower:
            return color
    return "unknown"


def _normalise_category(raw_category: str, raw_name: str) -> str:
    """
    Map a raw dataset category_label → normalised super-category.
    Falls back to heuristic keyword matching on the product name if the
    category_label isn't in the map.
    """
    cleaned = _normalise_str(raw_category).replace(" ", "-")
    if cleaned in CATEGORY_MAP:
        return CATEGORY_MAP[cleaned]

    # Try partial matching
    for key, value in CATEGORY_MAP.items():
        if key in cleaned or cleaned in key:
            return value

    # Keyword fallback on product name
    name_lower = _normalise_str(raw_name)
    topwear_kw  = ["shirt", "top", "blouse", "kurta", "dress", "jacket",
                   "blazer", "sweatshirt", "hoodie", "sweater", "coat", "tee"]
    bottom_kw   = ["trouser", "jean", "pant", "chino", "skirt", "short",
                   "palazzo", "legging"]
    footwear_kw = ["shoe", "heel", "flat", "sneaker", "loafer", "boot",
                   "sandal", "slipper", "oxford", "mule"]
    access_kw   = ["bag", "belt", "watch", "jewel", "glass", "scarf",
                   "hat", "cap", "clutch", "dupatta"]

    for kw in topwear_kw:
        if kw in name_lower:
            return "Topwear"
    for kw in bottom_kw:
        if kw in name_lower:
            return "Bottomwear"
    for kw in footwear_kw:
        if kw in name_lower:
            return "Footwear"
    for kw in access_kw:
        if kw in name_lower:
            return "Accessories"

    logger.warning("Could not normalise category '%s' for product '%s'",
                   raw_category, raw_name[:50])
    return "Other"


def _resolve_image_path(image_field: str) -> Optional[Path]:
    """
    Resolve the relative image path from products.csv to an absolute Path.
    Returns None if the file does not exist on disk.
    """
    if not image_field:
        return None
    # Strip leading 'images/' or 'data/' prefix variations
    rel = image_field.lstrip("/").replace("images/", "", 1)
    candidate = IMAGES_DIR / rel
    if candidate.exists():
        return candidate
    # Try without sub-folder
    flat = IMAGES_DIR / Path(rel).name
    if flat.exists():
        return flat
    return None


# ══════════════════════════════════════════════════════════════════════════════
# CORE LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def load_products(csv_path: Path = PRODUCTS_CSV) -> pd.DataFrame:
    """
    Load and clean products.csv.

    Returns
    -------
    pd.DataFrame
        Enriched product catalog with columns:
        id, name, brand, price_inr, rating, rating_count, gender,
        wear_type, category, category_label, occasion, tags,
        description, image, norm_category, dominant_color,
        image_path, image_exists, search_text
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"products.csv not found at {csv_path}.\n"
            "Download from: https://github.com/DarexAI-AI-Startup/ML-TASK\n"
            f"and place it in: {DATA_DIR}/"
        )

    logger.info("Loading products from %s …", csv_path)
    df = pd.read_csv(csv_path, dtype=str)

    # ── Basic normalisation ────────────────────────────────────────────────
    str_cols = ["id", "name", "brand", "gender", "wear_type",
                "category", "category_label", "occasion", "tags", "description", "image"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    numeric_cols = {"price_inr": 0.0, "rating": 0.0, "rating_count": 0}
    for col, default in numeric_cols.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # ── Normalised super-category ──────────────────────────────────────────
    df["norm_category"] = df.apply(
        lambda r: _normalise_category(r.get("category_label", ""), r.get("name", "")),
        axis=1,
    )

    # ── Dominant colour extraction ─────────────────────────────────────────
    df["dominant_color"] = df.apply(
        lambda r: _extract_dominant_color(
            f"{r.get('name','')} {r.get('tags','')} {r.get('description','')}"
        ),
        axis=1,
    )

    # ── Image path validation ──────────────────────────────────────────────
    df["image_path"] = df["image"].apply(
    lambda p: f"images/{Path(p).name}" if p else ""
    )
    df["image_exists"] = df["image_path"].apply(lambda p: bool(p) and Path(p).exists())

    # ── Consolidated search text (used for embedding generation) ───────────
    def _build_search_text(row: pd.Series) -> str:
        parts = [
            row.get("name", ""),
            row.get("brand", ""),
            row.get("category_label", ""),
            row.get("norm_category", ""),
            row.get("occasion", ""),
            row.get("wear_type", ""),
            row.get("gender", ""),
            row.get("dominant_color", ""),
            row.get("tags", "").replace(";", " "),
            row.get("description", "")[:200],   # cap description length
        ]
        return " | ".join(p for p in parts if p)

    df["search_text"] = df.apply(_build_search_text, axis=1)

    logger.info("Loaded %d products ✓", len(df))
    return df


def load_outfits(csv_path: Path = OUTFITS_CSV) -> pd.DataFrame:
    """
    Load and lightly clean outfits.csv.

    Returns
    -------
    pd.DataFrame
        outfit_id, gender, wear_type, occasion, theme,
        hero, hero_id, second, second_id, layer, layer_id,
        footwear, footwear_id, accessory_1, accessory_1_id,
        accessory_2, accessory_2_id, palette, stylist_rationale
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"outfits.csv not found at {csv_path}.\n"
            "Download from: https://github.com/DarexAI-AI-Startup/ML-TASK"
        )

    logger.info("Loading outfits from %s …", csv_path)
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    for col in df.columns:
        df[col] = df[col].str.strip()
    logger.info("Loaded %d curated outfits ✓", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS & VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_statistics(products: pd.DataFrame, outfits: pd.DataFrame) -> dict:
    """
    Return a summary statistics dictionary for the loaded catalog.

    Keys:
        total_products, total_outfits, category_counts, gender_counts,
        occasion_counts, wear_type_counts, image_coverage_pct,
        top_brands, avg_rating, price_stats
    """
    stats: dict = {}

    stats["total_products"] = len(products)
    stats["total_outfits"] = len(outfits)

    # Category distribution
    stats["category_counts"] = products["norm_category"].value_counts().to_dict()

    # Gender
    stats["gender_counts"] = products["gender"].value_counts().to_dict()

    # Occasion
    stats["occasion_counts"] = (
        products["occasion"]
        .str.lower()
        .value_counts()
        .head(10)
        .to_dict()
    )

    # Wear type
    stats["wear_type_counts"] = products["wear_type"].value_counts().to_dict()

    # Image coverage
    total = len(products)
    with_image = products["image_exists"].sum()
    stats["image_coverage_pct"] = round(100 * with_image / total, 1) if total else 0.0
    stats["products_with_images"] = int(with_image)
    stats["products_missing_images"] = int(total - with_image)

    # Top brands
    stats["top_brands"] = (
        products["brand"]
        .str.lower()
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .head(10)
        .to_dict()
    )

    # Rating
    valid_ratings = products["rating"].replace(0, np.nan).dropna()
    stats["avg_rating"] = round(float(valid_ratings.mean()), 2) if len(valid_ratings) else 0.0

    # Price
    valid_prices = products["price_inr"].replace(0, np.nan).dropna()
    if len(valid_prices):
        stats["price_stats"] = {
            "min": float(valid_prices.min()),
            "max": float(valid_prices.max()),
            "mean": round(float(valid_prices.mean()), 2),
            "median": round(float(valid_prices.median()), 2),
        }
    else:
        stats["price_stats"] = {}

    return stats


def validate_catalog(products: pd.DataFrame) -> list[str]:
    """
    Run data-quality checks on the catalog and return a list of warning strings.
    An empty list means all checks passed.
    """
    warnings: list[str] = []

    # Missing IDs
    missing_id = products["id"].isna() | (products["id"] == "")
    if missing_id.any():
        warnings.append(f"{missing_id.sum()} products have no ID.")

    # Duplicate IDs
    dups = products["id"].duplicated(keep=False)
    if dups.any():
        warnings.append(f"{dups.sum()} duplicate product IDs detected.")

    # Unknown categories
    unknown = products[products["norm_category"] == "Other"]
    if not unknown.empty:
        warnings.append(
            f"{len(unknown)} products mapped to 'Other' category: "
            + ", ".join(unknown["id"].head(5).tolist())
        )

    # Missing images
    no_img = products[~products["image_exists"]]
    if not no_img.empty:
        warnings.append(
            f"{len(no_img)} products have missing images."
        )

    # Missing descriptions
    no_desc = products[products["description"].str.len() < 5]
    if not no_desc.empty:
        warnings.append(f"{len(no_desc)} products have very short/missing descriptions.")

    return warnings


def print_statistics(stats: dict, warnings: list[str]) -> None:
    """Pretty-print dataset statistics and validation warnings."""
    line = "─" * 60
    print(f"\n{line}")
    print("  AI Fashion Assistant — Dataset Analysis Report")
    print(line)
    print(f"  Total products      : {stats['total_products']}")
    print(f"  Total outfits       : {stats['total_outfits']}")
    print(f"  Image coverage      : {stats['image_coverage_pct']}%")
    print(f"  Average product rating: {stats['avg_rating']}")
    print()
    print("  Category Distribution:")
    for cat, count in stats["category_counts"].items():
        print(f"    {cat:<15} : {count}")
    print()
    print("  Gender Distribution:")
    for g, count in stats["gender_counts"].items():
        print(f"    {g:<15} : {count}")
    print()
    print("  Top Occasions:")
    for occ, count in list(stats["occasion_counts"].items())[:6]:
        print(f"    {occ:<20} : {count}")
    print()
    if stats.get("price_stats"):
        ps = stats["price_stats"]
        print(f"  Price Range (INR)   : ₹{ps['min']:.0f} – ₹{ps['max']:.0f}")
        print(f"  Mean / Median Price : ₹{ps['mean']:.0f} / ₹{ps['median']:.0f}")
    print()
    if warnings:
        print("  ⚠  Validation Warnings:")
        for w in warnings:
            print(f"    • {w}")
    else:
        print("  ✓  No validation warnings.")
    print(line)


# ══════════════════════════════════════════════════════════════════════════════
# CATALOG BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_fashion_catalog() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    High-level convenience function that loads, cleans, validates, and
    returns the full fashion catalog.

    Returns
    -------
    products : pd.DataFrame   — enriched product catalog
    outfits  : pd.DataFrame   — curated outfit definitions
    stats    : dict           — summary statistics
    """
    products = load_products()
    outfits = load_outfits()
    stats = generate_statistics(products, outfits)
    warnings = validate_catalog(products)
    return products, outfits, stats


# ══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    products, outfits, stats = build_fashion_catalog()
    warnings = validate_catalog(products)
    print_statistics(stats, warnings)

    # Save cleaned catalog for reference
    out_path = DATA_DIR / "products_clean.csv"
    products.to_csv(out_path, index=False)
    logger.info("Cleaned catalog saved to %s", out_path)
