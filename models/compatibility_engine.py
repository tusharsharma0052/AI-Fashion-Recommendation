"""
models/compatibility_engine.py — Outfit Compatibility & Ranking Engine.

For each user query this engine:
  1. Receives FAISS-retrieved candidates per category (Topwear, Bottomwear, Footwear, Accessories)
  2. Scores every candidate against the user profile using:

        compatibility_score =
            0.40 × embedding_similarity
          + 0.20 × occasion_match
          + 0.15 × style_match
          + 0.15 × color_match
          + 0.10 × season_match

  3. Ranks and selects the best item per category
  4. Assembles TOP_K_OUTFITS complete outfit combinations
"""

from __future__ import annotations

import logging
import sys
from itertools import product as itertools_product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    COLOR_HARMONY,
    COMPAT_WEIGHT_COLOR,
    COMPAT_WEIGHT_EMBED,
    COMPAT_WEIGHT_OCCASION,
    COMPAT_WEIGHT_SEASON,
    COMPAT_WEIGHT_STYLE,
    MIN_COMPATIBILITY_SCORE,
    OCCASION_MAP,
    SEASON_MAP,
    TOP_K_OUTFITS,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL SCORING COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def _occasion_score(item: dict, profile: dict) -> float:
    """
    Return 1.0 if the product's occasion field matches the user's intent,
    0.5 for a partial match, 0.0 for no match.
    """
    user_occ = profile.get("occasion", "").lower()
    item_occ = str(item.get("occasion", "")).lower()

    if not item_occ:
        return 0.3  # unknown → neutral

    keywords = OCCASION_MAP.get(user_occ, [user_occ])
    for kw in keywords:
        if kw in item_occ:
            return 1.0
    # Partial — check if ANY OCCASION_MAP keyword appears
    for occ_keywords in OCCASION_MAP.values():
        for kw in occ_keywords:
            if kw in item_occ:
                return 0.3
    return 0.0


def _style_score(item: dict, profile: dict) -> float:
    """
    Simple keyword overlap between the profile style and the item's tags /
    description / category_label.
    """
    user_style = profile.get("style", "").lower()
    item_text = " ".join([
        str(item.get("tags", "")),
        str(item.get("description", "")),
        str(item.get("category_label", "")),
        str(item.get("wear_type", "")),
    ]).lower()

    if not user_style:
        return 0.5

    style_words = user_style.split()
    matched = sum(1 for w in style_words if w in item_text)
    return min(1.0, matched / max(len(style_words), 1))


def _color_score(item: dict, anchor_color: Optional[str]) -> float:
    """
    Assess colour harmony between the item's dominant colour and an anchor
    colour (typically the topwear colour).

    Returns:
        1.0 — anchor is None (first item, no constraint yet)
        1.0 — same colour family
        0.8 — colour is listed as harmonious with anchor
        0.5 — anchor unknown
        0.2 — no known harmony
    """
    if anchor_color is None:
        return 1.0

    item_color = str(item.get("dominant_color", "unknown")).lower()
    anchor_color = anchor_color.lower()

    if item_color == "unknown" or anchor_color == "unknown":
        return 0.5
    if item_color == anchor_color:
        return 1.0

    harmonious = COLOR_HARMONY.get(anchor_color, [])
    if item_color in harmonious:
        return 0.8

    # Check reverse direction
    if anchor_color in COLOR_HARMONY.get(item_color, []):
        return 0.7

    return 0.2


def _season_score(item: dict, profile: dict) -> float:
    """
    Match the item's seasonal metadata against the user's season preference.
    """
    user_season = profile.get("season", "all").lower()
    item_text = " ".join([
        str(item.get("tags", "")),
        str(item.get("description", "")),
        str(item.get("wear_type", "")),
    ]).lower()

    if user_season == "all":
        return 0.8  # neutral — no specific preference

    keywords = SEASON_MAP.get(user_season, [user_season])
    for kw in keywords:
        if kw in item_text:
            return 1.0

    # All-season item also works
    for kw in SEASON_MAP.get("all", []):
        if kw in item_text:
            return 0.8

    return 0.4  # might still work but no explicit mention


def _gender_penalty(item: dict, profile: dict) -> float:
    """
    Return 1.0 if gender matches (or is unisex), 0.0 otherwise.
    Used as a hard multiplier to filter gender-mismatched items.
    """
    user_gender = profile.get("gender", "unisex").lower()
    item_gender = str(item.get("gender", "unisex")).lower()

    if user_gender == "unisex" or item_gender == "unisex":
        return 1.0
    if user_gender in item_gender or item_gender in user_gender:
        return 1.0
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE COMPATIBILITY SCORE
# ══════════════════════════════════════════════════════════════════════════════

def score_item(
    item: dict,
    profile: dict,
    anchor_color: Optional[str] = None,
) -> float:
    """
    Compute the composite compatibility score for a single item.

    Parameters
    ----------
    item         : product dict row (from products_df)
    profile      : UserProfile from intent_parser
    anchor_color : dominant colour of the already-selected topwear (for colour harmony)

    Returns
    -------
    float in [0, 1]
    """
    embed_sim = float(item.get("similarity_score", 0.0))
    # Clamp to [0,1] — FAISS inner product on unit vectors is in [-1,1]
    embed_sim = (embed_sim + 1.0) / 2.0

    occ  = _occasion_score(item, profile)
    sty  = _style_score(item, profile)
    col  = _color_score(item, anchor_color)
    sea  = _season_score(item, profile)
    gpen = _gender_penalty(item, profile)

    score = (
        COMPAT_WEIGHT_EMBED    * embed_sim
        + COMPAT_WEIGHT_OCCASION * occ
        + COMPAT_WEIGHT_STYLE    * sty
        + COMPAT_WEIGHT_COLOR    * col
        + COMPAT_WEIGHT_SEASON   * sea
    ) * gpen

    return round(float(score), 4)


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY-LEVEL RANKING
# ══════════════════════════════════════════════════════════════════════════════

def _rank_candidates(
    candidates: pd.DataFrame,
    profile: dict,
    anchor_color: Optional[str],
    top_n: int = 5,
) -> list[dict]:
    """
    Score and rank a DataFrame of candidates, returning the top-n as dicts.
    Items below MIN_COMPATIBILITY_SCORE are discarded.
    """
    if candidates.empty:
        return []

    rows = candidates.to_dict(orient="records")
    for row in rows:
        row["compatibility_score"] = score_item(row, profile, anchor_color)

    rows = [r for r in rows if r["compatibility_score"] >= MIN_COMPATIBILITY_SCORE]
    rows.sort(key=lambda r: r["compatibility_score"], reverse=True)
    return rows[:top_n]


def recommend_topwear(
    candidates: pd.DataFrame,
    profile: dict,
    top_n: int = 5,
) -> list[dict]:
    """Return ranked topwear candidates."""
    return _rank_candidates(candidates, profile, anchor_color=None, top_n=top_n)


def recommend_bottomwear(
    candidates: pd.DataFrame,
    profile: dict,
    topwear_color: Optional[str] = None,
    top_n: int = 5,
) -> list[dict]:
    """Return ranked bottomwear candidates, harmonised with topwear colour."""
    return _rank_candidates(candidates, profile, anchor_color=topwear_color, top_n=top_n)


def recommend_footwear(
    candidates: pd.DataFrame,
    profile: dict,
    anchor_color: Optional[str] = None,
    top_n: int = 5,
) -> list[dict]:
    """Return ranked footwear candidates."""
    return _rank_candidates(candidates, profile, anchor_color=anchor_color, top_n=top_n)


def recommend_accessories(
    candidates: pd.DataFrame,
    profile: dict,
    anchor_color: Optional[str] = None,
    top_n: int = 3,
) -> list[dict]:
    """Return ranked accessory candidates."""
    return _rank_candidates(candidates, profile, anchor_color=anchor_color, top_n=top_n)


# ══════════════════════════════════════════════════════════════════════════════
# FULL OUTFIT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_complete_outfit(
    topwear_candidates: pd.DataFrame,
    bottomwear_candidates: pd.DataFrame,
    footwear_candidates: pd.DataFrame,
    accessories_candidates: pd.DataFrame,
    profile: dict,
    k: int = TOP_K_OUTFITS,
) -> list[dict]:
    """
    Assemble and rank complete outfit combinations.

    Strategy:
    1. Rank each category independently
    2. Pick the top-k topwear options as anchors
    3. For each anchor, select best bottomwear / footwear / accessories
       using colour harmony from the anchor
    4. Score each full outfit as the weighted average of its parts
    5. Return the top-k outfits sorted by total score

    Parameters
    ----------
    *_candidates  : DataFrames retrieved by FAISS for each category
    profile       : UserProfile dict
    k             : number of complete outfits to return

    Returns
    -------
    list of dicts, each with keys:
        topwear, bottomwear, footwear, accessories, total_score
    """
    # ── Rank individual categories ─────────────────────────────────────────
    tops     = recommend_topwear(topwear_candidates, profile, top_n=k + 2)
    if not tops:
        logger.warning("No suitable topwear found.")
        tops = topwear_candidates.head(1).to_dict(orient="records") if not topwear_candidates.empty else [{}]

    outfits: list[dict] = []

    for top in tops[:k + 2]:
        anchor_color = top.get("dominant_color")

        bottoms = recommend_bottomwear(
            bottomwear_candidates, profile, topwear_color=anchor_color, top_n=3
        )
        feet = recommend_footwear(
            footwear_candidates, profile, anchor_color=anchor_color, top_n=3
        )
        accs = recommend_accessories(
            accessories_candidates, profile, anchor_color=anchor_color, top_n=2
        )

        best_bottom = bottoms[0] if bottoms else ({} if bottomwear_candidates.empty else bottomwear_candidates.iloc[0].to_dict())
        best_foot   = feet[0]   if feet   else ({} if footwear_candidates.empty   else footwear_candidates.iloc[0].to_dict())

        # Outfit-level score = average of component scores
        scores = [
            top.get("compatibility_score", top.get("similarity_score", 0.5)),
            best_bottom.get("compatibility_score", best_bottom.get("similarity_score", 0.4)) if best_bottom else 0.0,
            best_foot.get("compatibility_score",   best_foot.get("similarity_score", 0.4))   if best_foot   else 0.0,
        ]
        if accs:
            scores.append(accs[0].get("compatibility_score", 0.4))

        total_score = round(float(np.mean(scores)), 4)

        outfit = {
            "topwear":     top,
            "bottomwear":  best_bottom,
            "footwear":    best_foot,
            "accessories": accs[:2],
            "total_score": total_score,
        }
        outfits.append(outfit)

    # De-duplicate by topwear ID and rank
    seen_top_ids: set[str] = set()
    unique_outfits: list[dict] = []
    for outfit in sorted(outfits, key=lambda o: o["total_score"], reverse=True):
        top_id = outfit["topwear"].get("id", "")
        if top_id not in seen_top_ids:
            seen_top_ids.add(top_id)
            unique_outfits.append(outfit)
        if len(unique_outfits) >= k:
            break

    logger.info("Built %d complete outfit(s).", len(unique_outfits))
    return unique_outfits


# ══════════════════════════════════════════════════════════════════════════════
# CLI DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    # Minimal demo with synthetic data
    demo_profile = {
        "gender": "men",
        "occasion": "business",
        "style": "formal",
        "season": "all",
    }
    demo_item = {
        "id": "T001",
        "name": "White Formal Shirt",
        "dominant_color": "white",
        "occasion": "office formal",
        "tags": "formal work office",
        "description": "Crisp white formal shirt for professional settings.",
        "gender": "men",
        "similarity_score": 0.85,
    }
    score = score_item(demo_item, demo_profile, anchor_color=None)
    print(f"Compatibility score for '{demo_item['name']}': {score}")
