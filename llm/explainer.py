"""
llm/explainer.py — Outfit Explainability Engine via Gemini.

Generates 2-4 sentence natural-language explanations for a recommended outfit,
covering why each item was chosen and how the pieces work together.

Falls back to a template-based explanation when Gemini is unavailable.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
)

logger = logging.getLogger(__name__)


# ── Gemini client ──────────────────────────────────────────────────────────────
_gemini_model: Optional[genai.GenerativeModel] = None


def _get_model() -> genai.GenerativeModel:
    global _gemini_model
    if _gemini_model is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError("GEMINI_API_KEY not set.")
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=0.65,           # slightly more creative for explanations
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )
    return _gemini_model


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

_EXPLAIN_SYSTEM = """
You are an expert fashion stylist who provides concise, warm, and professional
outfit recommendations.

Given:
  • User's occasion / style intent
  • The selected outfit items (topwear, bottomwear, footwear, accessories)

Write a 2-4 sentence explanation covering:
  1. Why the topwear suits the occasion
  2. How the bottomwear complements it
  3. How the footwear finishes the look
  4. (If accessories present) How accessories elevate the outfit

Tone: friendly, confident, and practical.
Length: 2-4 sentences only.
Do NOT use bullet points or lists — write flowing prose.
Do NOT mention product IDs or prices.
""".strip()


def _build_explain_prompt(
    profile: dict,
    topwear: Optional[dict],
    bottomwear: Optional[dict],
    footwear: Optional[dict],
    accessories: list[dict],
) -> str:
    def item_desc(item: Optional[dict]) -> str:
        if not item:
            return "not selected"
        return (
            f"{item.get('name', 'Unknown item')} "
            f"({item.get('dominant_color', '')} {item.get('category_label', '')})"
        ).strip()

    acc_str = (
        ", ".join(item_desc(a) for a in accessories)
        if accessories
        else "none"
    )

    return (
        f"{_EXPLAIN_SYSTEM}\n\n"
        f"User Intent:\n"
        f"  Occasion : {profile.get('occasion', 'casual')}\n"
        f"  Style    : {profile.get('style', 'casual')}\n"
        f"  Gender   : {profile.get('gender', 'unisex')}\n"
        f"  Season   : {profile.get('season', 'all')}\n\n"
        f"Selected Outfit:\n"
        f"  Topwear     : {item_desc(topwear)}\n"
        f"  Bottomwear  : {item_desc(bottomwear)}\n"
        f"  Footwear    : {item_desc(footwear)}\n"
        f"  Accessories : {acc_str}\n\n"
        f"Write the outfit explanation:"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE FALLBACK
# ══════════════════════════════════════════════════════════════════════════════

def _template_explanation(
    profile: dict,
    topwear: Optional[dict],
    bottomwear: Optional[dict],
    footwear: Optional[dict],
    accessories: list[dict],
) -> str:
    """
    Rule-based fallback explanation when Gemini is unavailable.
    Produces a readable sentence without any API call.
    """
    occasion = profile.get("occasion", "this occasion")
    style    = profile.get("style", "the chosen style")

    top_name = topwear.get("name", "the topwear") if topwear else "the top"
    bot_name = bottomwear.get("name", "the bottomwear") if bottomwear else "the bottom"
    foot_name = footwear.get("name", "the footwear") if footwear else "the shoes"
    top_color = topwear.get("dominant_color", "") if topwear else ""
    bot_color = bottomwear.get("dominant_color", "") if bottomwear else ""

    sentences: list[str] = []

    # Sentence 1 – topwear
    color_mention = f" in {top_color}" if top_color and top_color != "unknown" else ""
    sentences.append(
        f"The {top_name}{color_mention} is a versatile choice "
        f"that perfectly suits a {occasion} setting with its {style} aesthetic."
    )

    # Sentence 2 – bottomwear
    if bottomwear:
        contrast = (
            "creating a clean contrast" if top_color != bot_color else "maintaining a cohesive palette"
        )
        sentences.append(
            f"Paired with {bot_name}, the combination is polished and well-balanced, "
            f"{contrast} that works beautifully for the occasion."
        )

    # Sentence 3 – footwear
    if footwear:
        sentences.append(
            f"{foot_name} rounds off the outfit with the right level of formality, "
            f"ensuring comfort and style go hand in hand."
        )

    # Sentence 4 – accessories
    if accessories:
        acc_names = ", ".join(a.get("name", "accessory") for a in accessories[:2])
        sentences.append(
            f"The {acc_names} add a thoughtful finishing touch that elevates the overall look."
        )

    return " ".join(sentences)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_outfit_explanation(
    profile: dict,
    topwear: Optional[dict] = None,
    bottomwear: Optional[dict] = None,
    footwear: Optional[dict] = None,
    accessories: Optional[list[dict]] = None,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> str:
    """
    Generate a natural-language explanation for the recommended outfit.

    Parameters
    ----------
    profile     : UserProfile dict from intent_parser
    topwear     : dict of the selected topwear product (or None)
    bottomwear  : dict of the selected bottomwear product (or None)
    footwear    : dict of the selected footwear product (or None)
    accessories : list of selected accessory product dicts (or None / [])
    max_retries : Gemini API retry count
    retry_delay : seconds between retries

    Returns
    -------
    str — 2-4 sentence outfit explanation
    """
    accessories = accessories or []

    prompt = _build_explain_prompt(profile, topwear, bottomwear, footwear, accessories)

    for attempt in range(max_retries + 1):
        try:
            model = _get_model()
            response = model.generate_content(prompt)
            explanation = response.text.strip()

            # Sanity check — reject empty or too-short responses
            if len(explanation) < 30:
                raise ValueError("Explanation too short.")

            # Remove any accidental bullet points
            explanation = re.sub(r"^\s*[-•*]\s*", "", explanation, flags=re.MULTILINE)
            explanation = " ".join(explanation.split())  # collapse whitespace

            logger.info("Explanation generated via Gemini (%d chars).", len(explanation))
            return explanation

        except Exception as exc:
            logger.warning("Gemini explanation error (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries:
                time.sleep(retry_delay)

    logger.warning("Using template fallback for explanation.")
    return _template_explanation(profile, topwear, bottomwear, footwear, accessories)


def generate_style_tips(profile: dict) -> str:
    """
    Generate 2-3 quick style tips based on the user's occasion and style profile.
    Used as a secondary panel in the Streamlit UI.
    """
    prompt = (
        f"You are a fashion stylist. Give exactly 3 quick, practical style tips "
        f"for someone dressing for a '{profile.get('occasion', 'casual')}' occasion "
        f"in a '{profile.get('style', 'casual')}' style. "
        f"Format: numbered list, one sentence per tip. No preamble."
    )
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        logger.warning("Style tips generation failed: %s", exc)
        return (
            "1. Choose colours that complement each other.\n"
            "2. Fit is everything — well-fitted clothes always look better.\n"
            "3. Accessories can transform even the simplest outfit."
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    sample_profile = {
        "gender": "men",
        "occasion": "business",
        "style": "formal",
        "season": "all",
    }
    sample_topwear = {
        "name": "White Oxford Formal Shirt",
        "dominant_color": "white",
        "category_label": "formal-shirts",
    }
    sample_bottomwear = {
        "name": "Navy Blue Slim-Fit Trousers",
        "dominant_color": "navy",
        "category_label": "trousers",
    }
    sample_footwear = {
        "name": "Brown Derby Leather Shoes",
        "dominant_color": "brown",
        "category_label": "formal-shoes",
    }
    sample_accessories = [
        {"name": "Classic Brown Leather Belt", "dominant_color": "brown"},
    ]

    explanation = generate_outfit_explanation(
        profile=sample_profile,
        topwear=sample_topwear,
        bottomwear=sample_bottomwear,
        footwear=sample_footwear,
        accessories=sample_accessories,
    )
    print("\nOutfit Explanation:")
    print(explanation)

    tips = generate_style_tips(sample_profile)
    print("\nStyle Tips:")
    print(tips)
