"""
config.py — Centralised configuration for the AI Fashion Outfit Recommendation System.

All paths, model names, hyper-parameters, and API settings live here.
Import this module anywhere in the project instead of hard-coding constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────────
load_dotenv()

# ── Root paths ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = ROOT_DIR / "images"
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"
DOCS_DIR = ROOT_DIR / "docs"

# ── Data files ─────────────────────────────────────────────────────────────────
PRODUCTS_CSV = DATA_DIR / "products.csv"
OUTFITS_CSV = DATA_DIR / "outfits.csv"

# ── Embedding files ─────────────────────────────────────────────────────────────
TEXT_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "text_embeddings.npy"
IMAGE_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "image_embeddings.npy"
COMBINED_EMBEDDINGS_FILE = EMBEDDINGS_DIR / "combined_embeddings.npy"
PRODUCT_IDS_FILE = EMBEDDINGS_DIR / "product_ids.npy"
FAISS_INDEX_FILE = EMBEDDINGS_DIR / "faiss_index.bin"
FAISS_META_FILE = EMBEDDINGS_DIR / "faiss_meta.pkl"

# ── ML Model names ─────────────────────────────────────────────────────────────
# Sentence-transformers model for text embeddings
SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# CLIP model for image + text multimodal embeddings
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Embedding dimension produced by the sentence model
TEXT_EMBED_DIM = 384

# CLIP produces 512-d embeddings for both image and text
CLIP_EMBED_DIM = 512

# Final combined embedding dimension (text + optional CLIP)
COMBINED_EMBED_DIM = TEXT_EMBED_DIM  # change to TEXT_EMBED_DIM + CLIP_EMBED_DIM if CLIP is active

# ── Gemini API ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"          # free-tier compatible
GEMINI_MAX_OUTPUT_TOKENS = 1024
GEMINI_TEMPERATURE = 0.4

# ── FAISS retrieval ────────────────────────────────────────────────────────────
FAISS_N_RESULTS = 15          # candidates fetched per category before re-ranking
FAISS_NPROBE = 10             # IVF probe count (accuracy vs speed)

# ── Compatibility engine weights ────────────────────────────────────────────────
COMPAT_WEIGHT_EMBED = 0.40
COMPAT_WEIGHT_OCCASION = 0.20
COMPAT_WEIGHT_STYLE = 0.15
COMPAT_WEIGHT_COLOR = 0.15
COMPAT_WEIGHT_SEASON = 0.10

# ── Outfit building ────────────────────────────────────────────────────────────
TOP_K_OUTFITS = 3             # how many full outfit options to return
MIN_COMPATIBILITY_SCORE = 0.3 # discard items below this threshold

# ── Category normalisation map ─────────────────────────────────────────────────
# Maps dataset category_label values → normalised super-category
CATEGORY_MAP: dict[str, str] = {
    # ── Topwear ──
    "formal-shirts": "Topwear",
    "shirts": "Topwear",
    "t-shirts": "Topwear",
    "tops": "Topwear",
    "blouses": "Topwear",
    "kurtas": "Topwear",
    "kurtis": "Topwear",
    "sarees": "Topwear",
    "lehengas": "Topwear",
    "dresses": "Topwear",
    "co-ord-sets": "Topwear",
    "jumpsuits": "Topwear",
    "sweatshirts": "Topwear",
    "hoodies": "Topwear",
    "sweaters": "Topwear",
    "coats": "Topwear",
    "blazers": "Topwear",
    "jackets": "Topwear",
    "indo-western": "Topwear",
    # ── Bottomwear ──
    "trousers": "Bottomwear",
    "formal-trousers": "Bottomwear",
    "chinos": "Bottomwear",
    "jeans": "Bottomwear",
    "pants": "Bottomwear",
    "skirts": "Bottomwear",
    "shorts": "Bottomwear",
    "palazzos": "Bottomwear",
    "leggings": "Bottomwear",
    "culottes": "Bottomwear",
    "salwars": "Bottomwear",
    # ── Footwear ──
    "heels": "Footwear",
    "flats": "Footwear",
    "sneakers": "Footwear",
    "loafers": "Footwear",
    "formal-shoes": "Footwear",
    "casual-shoes": "Footwear",
    "boots": "Footwear",
    "sandals": "Footwear",
    "slippers": "Footwear",
    "oxfords": "Footwear",
    "mules": "Footwear",
    "kolhapuris": "Footwear",
    "juttis": "Footwear",
    # ── Accessories ──
    "bags": "Accessories",
    "handbags": "Accessories",
    "clutches": "Accessories",
    "belts": "Accessories",
    "watches": "Accessories",
    "jewellery": "Accessories",
    "jewelry": "Accessories",
    "sunglasses": "Accessories",
    "scarves": "Accessories",
    "stoles": "Accessories",
    "hats": "Accessories",
    "caps": "Accessories",
    "socks": "Accessories",
    "dupatta": "Accessories",
}

# ── Colour harmony matrix ───────────────────────────────────────────────────────
# Maps a base colour → list of harmonious complement colours
COLOR_HARMONY: dict[str, list[str]] = {
    "white":   ["navy", "black", "grey", "blue", "beige", "brown", "red", "green"],
    "black":   ["white", "grey", "red", "gold", "silver", "beige", "pink", "blue"],
    "navy":    ["white", "beige", "grey", "light blue", "tan", "brown", "yellow"],
    "grey":    ["white", "black", "navy", "maroon", "pink", "purple", "teal"],
    "beige":   ["white", "navy", "brown", "olive", "black", "camel", "rust"],
    "brown":   ["beige", "white", "cream", "navy", "olive", "mustard", "orange"],
    "blue":    ["white", "grey", "beige", "navy", "brown", "denim", "orange"],
    "green":   ["white", "beige", "brown", "olive", "cream", "tan", "khaki"],
    "red":     ["white", "black", "grey", "navy", "beige", "gold"],
    "pink":    ["white", "grey", "navy", "beige", "black", "lavender"],
    "yellow":  ["white", "grey", "navy", "black", "brown", "olive"],
    "maroon":  ["white", "grey", "beige", "gold", "cream", "black"],
    "orange":  ["white", "navy", "brown", "grey", "beige"],
    "purple":  ["white", "grey", "black", "silver", "lavender", "cream"],
    "olive":   ["white", "beige", "brown", "khaki", "orange", "navy"],
    "cream":   ["brown", "navy", "black", "maroon", "olive", "tan"],
}

# ── Occasion map ────────────────────────────────────────────────────────────────
# Maps user intent occasions → dataset occasion keywords
OCCASION_MAP: dict[str, list[str]] = {
    "business":      ["office", "formal", "business", "work", "professional"],
    "office":        ["office", "formal", "work", "business", "professional"],
    "casual":        ["casual", "everyday", "weekend", "relaxed", "informal"],
    "party":         ["party", "night out", "club", "celebration", "cocktail"],
    "wedding":       ["wedding", "festive", "ethnic", "ceremony", "reception"],
    "date":          ["date", "dinner", "romantic", "evening", "smart casual"],
    "summer":        ["summer", "beach", "vacation", "holiday", "resort"],
    "sport":         ["sport", "gym", "workout", "active", "athletic"],
    "ethnic":        ["ethnic", "festival", "traditional", "puja", "religious"],
    "smart casual":  ["smart casual", "semi-formal", "dinner", "brunch"],
}

# ── Season map ─────────────────────────────────────────────────────────────────
SEASON_MAP: dict[str, list[str]] = {
    "summer": ["summer", "spring", "warm", "tropical", "light"],
    "winter": ["winter", "fall", "autumn", "cold", "warm", "layered"],
    "all":    ["all season", "year-round", "versatile"],
}

# ── Streamlit app settings ──────────────────────────────────────────────────────
APP_TITLE = "👗 AI Fashion Assistant"
APP_ICON = "👗"
MAX_HISTORY = 20              # max messages kept in session
PLACEHOLDER_IMAGE = "https://via.placeholder.com/300x400?text=No+Image"
