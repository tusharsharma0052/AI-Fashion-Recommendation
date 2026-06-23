"""
utils/helpers.py — Shared utility functions used across the project.

Includes:
  • Image loading and resizing (Pillow)
  • Price and rating formatters
  • Logger factory
  • Colour badge HTML generator for Streamlit
  • Outfit dict → display dict converter
"""

from __future__ import annotations

import base64
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PLACEHOLDER_IMAGE


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_image(
    image_path: str | Path,
    max_width: int = 300,
    max_height: int = 400,
):
    """
    Load and resize a product image using Pillow.

    Returns
    -------
    PIL.Image or None if the path is missing / unreadable.
    """
    try:
        from PIL import Image  # type: ignore

        path = Path(image_path)
        if not path.exists():
            return None
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_width, max_height), Image.LANCZOS)
        return img
    except Exception:
        return None


def image_to_base64(image_path: str | Path) -> Optional[str]:
    """
    Convert a local image file to a base64-encoded data URI string,
    suitable for embedding in HTML / Streamlit markdown.

    Returns None if the image cannot be loaded.
    """
    img = load_image(image_path)
    if img is None:
        return None
    try:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def get_image_src(image_path: str | Path) -> str:
    """
    Return an image src suitable for an HTML <img> tag.
    Falls back to the placeholder URL if the local file is missing.
    """
    b64 = image_to_base64(image_path)
    return b64 if b64 else PLACEHOLDER_IMAGE


# ══════════════════════════════════════════════════════════════════════════════
# FORMATTERS
# ══════════════════════════════════════════════════════════════════════════════

def format_price(price: float | str) -> str:
    """Format a price value to a rupee string."""
    try:
        p = float(price)
        if p <= 0:
            return "Price N/A"
        return f"₹{p:,.0f}"
    except (ValueError, TypeError):
        return "Price N/A"


def format_rating(rating: float | str, count: float | str = 0) -> str:
    """Format a rating + review count into a readable string."""
    try:
        r = float(rating)
        c = int(float(count))
        if r <= 0:
            return "No rating"
        stars = "★" * round(r) + "☆" * (5 - round(r))
        return f"{stars} {r:.1f}" + (f" ({c:,} reviews)" if c > 0 else "")
    except (ValueError, TypeError):
        return "No rating"


def truncate(text: str, max_len: int = 120) -> str:
    """Truncate a string with an ellipsis."""
    text = text.strip()
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR BADGE
# ══════════════════════════════════════════════════════════════════════════════

_CSS_COLOR_MAP: dict[str, str] = {
    "white":   "#f5f5f5",
    "black":   "#1a1a1a",
    "navy":    "#001F5B",
    "grey":    "#808080",
    "gray":    "#808080",
    "beige":   "#F5F5DC",
    "brown":   "#8B4513",
    "blue":    "#4169E1",
    "green":   "#2E8B57",
    "red":     "#DC143C",
    "pink":    "#FF69B4",
    "yellow":  "#FFD700",
    "maroon":  "#800000",
    "orange":  "#FF8C00",
    "purple":  "#800080",
    "olive":   "#808000",
    "cream":   "#FFFDD0",
    "teal":    "#008080",
    "khaki":   "#C3B091",
    "denim":   "#1560BD",
    "tan":     "#D2B48C",
    "camel":   "#C19A6B",
    "rust":    "#B7410E",
    "mustard": "#FFDB58",
    "coral":   "#FF7F50",
    "mint":    "#98FF98",
    "lavender":"#E6E6FA",
    "burgundy":"#800020",
    "charcoal":"#36454F",
    "gold":    "#FFD700",
    "silver":  "#C0C0C0",
}


def color_badge_html(color_name: str) -> str:
    """
    Return an HTML string with a small coloured dot and the colour name.
    Used in Streamlit product cards.
    """
    css_color = _CSS_COLOR_MAP.get(color_name.lower(), "#cccccc")
    border = "border: 1px solid #999;" if color_name.lower() in ("white", "cream", "beige") else ""
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;">'
        f'<span style="width:14px;height:14px;border-radius:50%;'
        f'background:{css_color};{border}display:inline-block;"></span>'
        f'<span style="font-size:0.85rem;">{color_name.title()}</span>'
        f"</span>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# OUTFIT DISPLAY CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

def outfit_to_display(outfit: dict, outfit_number: int = 1) -> dict:
    """
    Convert a raw outfit dict from ChatEngine into a display-ready dict
    for the Streamlit UI.

    Returns
    -------
    dict with keys:
        number, total_score_pct, explanation,
        topwear, bottomwear, footwear, accessories (each a display_item dict)
    """
    def _display_item(item: dict) -> dict:
        if not item:
            return {}
        return {
            "id": item.get("id", ""),
            "name": item.get("name", "Unknown"),
            "brand": item.get("brand", ""),
            "price": format_price(item.get("price_inr", 0)),
            "rating": format_rating(item.get("rating", 0), item.get("rating_count", 0)),
            "color": item.get("dominant_color", "unknown"),
            "color_badge": color_badge_html(item.get("dominant_color", "unknown")),
            "category": item.get("norm_category", ""),
            "image_src": get_image_src(item.get("image_path", "")),
            "occasion": item.get("occasion", ""),
            "compat_score": item.get("compatibility_score", item.get("similarity_score", 0)),
            "description": truncate(item.get("description", ""), 150),
        }

    return {
        "number":          outfit_number,
        "total_score_pct": int(outfit.get("total_score", 0) * 100),
        "explanation":     outfit.get("explanation", ""),
        "topwear":         _display_item(outfit.get("topwear", {})),
        "bottomwear":      _display_item(outfit.get("bottomwear", {})),
        "footwear":        _display_item(outfit.get("footwear", {})),
        "accessories":     [_display_item(a) for a in outfit.get("accessories", [])],
    }
