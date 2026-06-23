"""
assistant/chat_engine.py — Conversational Fashion Assistant Pipeline.

Orchestrates the end-to-end recommendation flow:

    User Query
        → Gemini Intent Extraction        (llm/intent_parser.py)
        → Build retrieval query embedding  (embedding_generator.py)
        → FAISS Retrieval per category     (retrieval/faiss_index.py)
        → Compatibility Scoring & Ranking  (models/compatibility_engine.py)
        → Gemini Explanation               (llm/explainer.py)
        → Structured Response

The ChatEngine is stateful — it keeps a short conversation history so the
user can refine requests ("make it more casual", "I prefer blue").
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FAISS_N_RESULTS, MAX_HISTORY, TOP_K_OUTFITS
from embedding_generator import generate_text_embedding, load_embeddings
from llm.explainer import generate_outfit_explanation, generate_style_tips
from llm.intent_parser import build_retrieval_query, extract_user_intent
from models.compatibility_engine import build_complete_outfit
from retrieval.faiss_index import FashionRetriever, load_index

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """A single turn in the conversation."""
    role: str           # "user" | "assistant"
    content: str
    profile: Optional[dict] = None
    outfits: list[dict] = field(default_factory=list)
    style_tips: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class OutfitRecommendation:
    """Structured recommendation response returned by ChatEngine.chat()."""
    profile: dict
    outfits: list[dict]          # list of {topwear, bottomwear, footwear, accessories, total_score, explanation}
    style_tips: str
    query: str
    processing_time_s: float


# ══════════════════════════════════════════════════════════════════════════════
# CHAT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ChatEngine:
    """
    Stateful conversational engine for fashion outfit recommendations.

    Usage
    -----
    engine = ChatEngine()
    engine.load()
    result = engine.chat("I need an outfit for a business meeting")
    """

    def __init__(self, n_outfits: int = TOP_K_OUTFITS) -> None:
        self.n_outfits = n_outfits
        self.history: list[Message] = []
        self._retriever: Optional[FashionRetriever] = None
        self._loaded = False

    # ── Initialisation ─────────────────────────────────────────────────────

    def load(self) -> "ChatEngine":
        """Load the FAISS index into memory (idempotent)."""
        if not self._loaded:
            logger.info("Loading FAISS retriever …")
            self._retriever = FashionRetriever().load()
            self._loaded = True
            logger.info("ChatEngine ready ✓")
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ── Retrieval helpers ──────────────────────────────────────────────────

    def _retrieve_category(
        self,
        query_embedding: np.ndarray,
        category: str,
        gender: Optional[str],
        k: int = FAISS_N_RESULTS,
    ) -> pd.DataFrame:
        """Retrieve top-k candidates for a single normalised category."""
        try:
            return self._retriever.search(
                query_embedding, category=category, gender=gender, k=k
            )
        except Exception as exc:
            logger.error("Retrieval error for %s: %s", category, exc)
            return pd.DataFrame()

    # ── Context merging ────────────────────────────────────────────────────

    def _merge_with_history(self, profile: dict) -> dict:
        """
        Merge current profile with the most recent assistant turn so the user
        can say "more casual" or "in blue" and the engine remembers gender/occasion.
        """
        if not self.history:
            return profile

        # Find most recent assistant message with a profile
        for msg in reversed(self.history):
            if msg.role == "assistant" and msg.profile:
                prev = msg.profile
                # Only inherit fields the user did NOT explicitly mention
                for key in ("gender", "season"):
                    if not profile.get(key) or profile[key] in ("unisex", "all", ""):
                        if prev.get(key):
                            profile[key] = prev[key]
                break

        return profile

    # ── Main pipeline ──────────────────────────────────────────────────────

    def chat(self, user_query: str) -> OutfitRecommendation:
        """
        Process a user message through the full recommendation pipeline.

        Parameters
        ----------
        user_query : raw natural language input

        Returns
        -------
        OutfitRecommendation
        """
        self._ensure_loaded()
        t_start = time.time()

        # ── 1. Log user message ────────────────────────────────────────────
        self.history.append(Message(role="user", content=user_query))
        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

        # ── 2. Extract intent ──────────────────────────────────────────────
        logger.info("Extracting intent for: %r", user_query[:80])
        profile = extract_user_intent(user_query)
        profile = self._merge_with_history(profile)
        gender = profile.get("gender") or None
        if gender == "unisex":
            gender = None   # don't apply gender filter for unisex

        # ── 3. Build retrieval embedding ───────────────────────────────────
        retrieval_query = build_retrieval_query(profile)
        logger.info("Retrieval query: %r", retrieval_query)
        query_embedding = generate_text_embedding(retrieval_query)

        # ── 4. FAISS retrieval per category ────────────────────────────────
        top_candidates  = self._retrieve_category(query_embedding, "Topwear",    gender)
        bot_candidates  = self._retrieve_category(query_embedding, "Bottomwear", gender)
        foot_candidates = self._retrieve_category(query_embedding, "Footwear",   gender)
        acc_candidates  = self._retrieve_category(query_embedding, "Accessories", None)  # accessories are often unisex

        logger.info(
            "Candidates — top:%d  bot:%d  foot:%d  acc:%d",
            len(top_candidates), len(bot_candidates),
            len(foot_candidates), len(acc_candidates),
        )

        # ── 5. Build & rank complete outfits ───────────────────────────────
        outfits = build_complete_outfit(
            top_candidates,
            bot_candidates,
            foot_candidates,
            acc_candidates,
            profile,
            k=self.n_outfits,
        )

        if not outfits:
            logger.warning("No outfits assembled — returning empty result.")

        # ── 6. Generate explanations ───────────────────────────────────────
        for outfit in outfits:
            outfit["explanation"] = generate_outfit_explanation(
                profile=profile,
                topwear=outfit.get("topwear") or None,
                bottomwear=outfit.get("bottomwear") or None,
                footwear=outfit.get("footwear") or None,
                accessories=outfit.get("accessories", []),
            )

        # ── 7. Style tips ──────────────────────────────────────────────────
        style_tips = generate_style_tips(profile)

        # ── 8. Build assistant message ─────────────────────────────────────
        summary = self._build_summary_text(profile, outfits)
        assistant_msg = Message(
            role="assistant",
            content=summary,
            profile=profile,
            outfits=outfits,
            style_tips=style_tips,
        )
        self.history.append(assistant_msg)

        processing_time = round(time.time() - t_start, 2)
        logger.info("Pipeline completed in %.2fs", processing_time)

        return OutfitRecommendation(
            profile=profile,
            outfits=outfits,
            style_tips=style_tips,
            query=user_query,
            processing_time_s=processing_time,
        )

    # ── Formatting ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_summary_text(profile: dict, outfits: list[dict]) -> str:
        """Build a short text summary for the conversation history."""
        occ = profile.get("occasion", "your occasion")
        style = profile.get("style", "")
        n = len(outfits)
        return (
            f"Here {'are' if n > 1 else 'is'} {n} outfit recommendation"
            f"{'s' if n > 1 else ''} for a {style} {occ} look. "
            f"Each outfit has been scored for colour harmony, style fit, and occasion suitability."
        )

    # ── History helpers ────────────────────────────────────────────────────

    def get_history(self) -> list[dict]:
        """Return conversation history as a list of plain dicts."""
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in self.history
        ]

    def clear_history(self) -> None:
        """Reset the conversation."""
        self.history.clear()
        logger.info("Conversation history cleared.")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON (used by Streamlit app.py)
# ══════════════════════════════════════════════════════════════════════════════

_engine_instance: Optional[ChatEngine] = None


def get_engine() -> ChatEngine:
    """Return a lazily-initialised singleton ChatEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ChatEngine()
        _engine_instance.load()
    return _engine_instance


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    engine = ChatEngine()
    try:
        engine.load()
    except FileNotFoundError as e:
        print(f"\n⚠  {e}")
        print("Run the setup pipeline first:")
        print("  python dataset_analysis.py")
        print("  python embedding_generator.py")
        print("  python retrieval/faiss_index.py")
        sys.exit(1)

    test_queries = [
        "I need an outfit for a business meeting.",
        "Suggest a smart casual outfit for a dinner date.",
        "I am a 22-year-old male looking for a casual summer outfit.",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        result = engine.chat(q)
        print(f"Profile   : {result.profile}")
        print(f"Outfits   : {len(result.outfits)} generated")
        for i, outfit in enumerate(result.outfits, 1):
            print(f"\n  Outfit {i} (score {outfit['total_score']:.3f}):")
            print(f"    Topwear   : {outfit['topwear'].get('name','—')}")
            print(f"    Bottomwear: {outfit['bottomwear'].get('name','—')}")
            print(f"    Footwear  : {outfit['footwear'].get('name','—')}")
            if outfit["accessories"]:
                print(f"    Accessory : {outfit['accessories'][0].get('name','—')}")
            print(f"    Explanation: {outfit['explanation'][:120]}…")
        print(f"\nStyle Tips:\n{result.style_tips}")
