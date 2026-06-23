"""
llm/intent_parser.py — Natural Language Intent Extraction via Gemini.

Converts a free-form user query into a structured UserProfile JSON that
downstream modules use for retrieval and compatibility scoring.

Example input : "I need a smart casual look for a dinner date this summer."
Example output:
    {
        "gender": "unisex",
        "age": "25-35",
        "occasion": "date",
        "style": "smart casual",
        "season": "summer",
        "color_preference": "",
        "budget_range": "",
        "specific_items": [],
        "raw_query": "I need a smart casual look for a dinner date this summer."
    }
"""

from __future__ import annotations

import json
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
    OCCASION_MAP,
    SEASON_MAP,
)

logger = logging.getLogger(__name__)

# ── Gemini client initialisation ───────────────────────────────────────────────
def _init_gemini() -> genai.GenerativeModel:
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to your .env file or environment."
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai.types.GenerationConfig(
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        ),
    )


_gemini_model: Optional[genai.GenerativeModel] = None


def _get_model() -> genai.GenerativeModel:
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = _init_gemini()
    return _gemini_model


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_SYSTEM_PROMPT = """
You are a fashion intelligence assistant specialised in extracting structured user intent
from natural language fashion queries.

Your task: Analyse the user query and return ONLY a valid JSON object — no markdown,
no code fences, no explanation — with exactly these keys:

{
  "gender":           <"men" | "women" | "unisex" — infer from context, default "unisex">,
  "age":              <age or age range string, e.g. "22" or "25-35"; empty string if unknown>,
  "occasion":         <one of: business, office, casual, party, wedding, date, summer, sport, ethnic, smart casual; infer best match>,
  "style":            <brief style descriptor, e.g. "formal", "smart casual", "boho", "minimalist">,
  "season":           <"summer" | "winter" | "all" — infer from context or default "all">,
  "color_preference": <mentioned colour or empty string>,
  "budget_range":     <e.g. "under 2000" or "1000-3000 INR" or empty string>,
  "specific_items":   <list of specific clothing items mentioned, e.g. ["white shirt", "chinos"]>,
  "raw_query":        <the original query, verbatim>
}

Rules:
- Never add keys not listed above.
- Infer gender from pronouns: "I am a male" → "men", "for my wife" → "women".
- If the user says "wedding" the occasion is "wedding".
- If the query mentions "office" or "meeting" treat occasion as "business".
- Map "dinner date" → occasion "date", style "smart casual".
- Map "festival" or "puja" → occasion "ethnic".
- Return ONLY the JSON object, nothing else.
""".strip()


def _build_intent_prompt(user_query: str) -> str:
    return f"{_INTENT_SYSTEM_PROMPT}\n\nUser Query: {user_query}"


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSING
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_PROFILE: dict = {
    "gender": "unisex",
    "age": "",
    "occasion": "casual",
    "style": "casual",
    "season": "all",
    "color_preference": "",
    "budget_range": "",
    "specific_items": [],
    "raw_query": "",
}


def _parse_json_response(raw: str) -> dict:
    """
    Extract a JSON object from the Gemini response string.
    Handles markdown code fences and stray text gracefully.
    """
    # Strip markdown code fences
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find first { ... } block
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {raw[:200]}")

    return json.loads(match.group())


def _rule_based_fallback(query: str) -> dict:
    """
    Lightweight rule-based intent extraction used when the Gemini API is
    unavailable or returns an unparseable response.
    """
    q = query.lower()
    profile = {**_DEFAULT_PROFILE, "raw_query": query}

    # Gender
    if any(w in q for w in ["male", "man", "men", "boy", "he ", "his "]):
        profile["gender"] = "men"
    elif any(w in q for w in ["female", "woman", "women", "girl", "she ", "her "]):
        profile["gender"] = "women"

    # Occasion
    for occ, keywords in OCCASION_MAP.items():
        if any(kw in q for kw in keywords):
            profile["occasion"] = occ
            break

    # Style
    if "formal" in q or "business" in q or "office" in q:
        profile["style"] = "formal"
    elif "casual" in q:
        profile["style"] = "casual"
    elif "smart" in q:
        profile["style"] = "smart casual"
    elif "ethnic" in q or "traditional" in q:
        profile["style"] = "ethnic"

    # Season
    for season, keywords in SEASON_MAP.items():
        if any(kw in q for kw in keywords):
            profile["season"] = season
            break

    # Age (simple regex: "22 year", "22-year", "age 22")
    age_match = re.search(r"\b(\d{2})\s*(?:-?\s*year|\s*yr|yo\b)", q)
    if age_match:
        profile["age"] = age_match.group(1)

    return profile


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_user_intent(
    user_query: str,
    max_retries: int = 2,
    retry_delay: float = 1.0,
) -> dict:
    """
    Parse a natural language fashion query into a structured UserProfile dict.

    Tries the Gemini API first; falls back to rule-based extraction on failure.

    Parameters
    ----------
    user_query  : raw user input string
    max_retries : number of Gemini API retries on transient errors
    retry_delay : seconds between retries

    Returns
    -------
    dict with keys: gender, age, occasion, style, season, color_preference,
                    budget_range, specific_items, raw_query
    """
    if not user_query.strip():
        return {**_DEFAULT_PROFILE, "raw_query": user_query}

    prompt = _build_intent_prompt(user_query)

    for attempt in range(max_retries + 1):
        try:
            model = _get_model()
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            profile = _parse_json_response(raw_text)

            # Enforce required keys (fill missing with defaults)
            for key, default in _DEFAULT_PROFILE.items():
                if key not in profile:
                    profile[key] = default

            profile["raw_query"] = user_query
            logger.info("Intent extracted via Gemini: %s", profile)
            return profile

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error (attempt %d): %s", attempt + 1, exc)
        except Exception as exc:
            logger.warning("Gemini API error (attempt %d): %s", attempt + 1, exc)
            if attempt < max_retries:
                time.sleep(retry_delay)

    # Fallback
    logger.warning("Falling back to rule-based intent extraction.")
    return _rule_based_fallback(user_query)


def build_retrieval_query(profile: dict) -> str:
    """
    Convert a UserProfile dict into a rich text query for embedding retrieval.

    This string is passed to generate_text_embedding() and used to search
    the FAISS index.
    """
    parts = [
        profile.get("occasion", ""),
        profile.get("style", ""),
        profile.get("season", ""),
        profile.get("gender", ""),
        profile.get("color_preference", ""),
        " ".join(profile.get("specific_items", [])),
    ]
    return " ".join(p for p in parts if p).strip() or profile.get("raw_query", "")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    samples = [
        "I need an outfit for a business meeting.",
        "Suggest a smart casual outfit for a dinner date.",
        "I am attending a wedding next weekend.",
        "I am a 22-year-old male looking for a casual summer outfit.",
        "Something ethnic for Diwali celebrations for women.",
    ]

    for query in samples:
        print(f"\nQuery : {query}")
        profile = extract_user_intent(query)
        print(f"Profile: {json.dumps(profile, indent=2)}")
        print(f"Retrieval Query: {build_retrieval_query(profile)}")
