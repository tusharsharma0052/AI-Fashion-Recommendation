"""
api.py — FastAPI REST Backend for the Fashion Recommendation System.

Endpoints:
  POST /recommend      → Full outfit recommendation from a text query
  POST /intent         → Extract user intent only (debugging)
  GET  /health         → Health check
  GET  /stats          → Catalog statistics

Run with:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Project imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import TOP_K_OUTFITS
from dataset_analysis import build_fashion_catalog, generate_statistics, validate_catalog
from llm.intent_parser import extract_user_intent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

# ══════════════════════════════════════════════════════════════════════════════
# APP STATE
# ══════════════════════════════════════════════════════════════════════════════

class AppState:
    engine = None
    catalog_stats: dict = {}


state = AppState()


# ══════════════════════════════════════════════════════════════════════════════
# LIFESPAN (startup / shutdown)
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ChatEngine and catalog stats at startup."""
    logger.info("Starting Fashion AI API …")
    try:
        from assistant.chat_engine import ChatEngine
        state.engine = ChatEngine()
        state.engine.load()
        logger.info("ChatEngine loaded ✓")
    except FileNotFoundError as exc:
        logger.error("ChatEngine could not load: %s", exc)
        logger.warning("Run `python setup_pipeline.py` to build required artefacts.")

    try:
        products, outfits, stats = build_fashion_catalog()
        warnings = validate_catalog(products)
        state.catalog_stats = {**stats, "validation_warnings": warnings}
        logger.info("Catalog stats loaded: %d products", stats["total_products"])
    except Exception as exc:
        logger.warning("Could not load catalog stats: %s", exc)

    yield
    logger.info("Fashion AI API shutting down.")


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Fashion Outfit Recommendation API",
    description=(
        "End-to-end fashion recommendation system powered by Gemini, "
        "FAISS, and Sentence Transformers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class RecommendRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        example="I need an outfit for a business meeting.",
    )
    n_outfits: int = Field(
        default=TOP_K_OUTFITS,
        ge=1,
        le=5,
        description="Number of outfit recommendations to generate.",
    )


class IntentRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class ProductOut(BaseModel):
    id: str
    name: str
    brand: str
    price_inr: float
    rating: float
    dominant_color: str
    norm_category: str
    occasion: str
    image_path: str
    compatibility_score: float


class OutfitOut(BaseModel):
    topwear: Optional[dict] = None
    bottomwear: Optional[dict] = None
    footwear: Optional[dict] = None
    accessories: list[dict] = []
    total_score: float
    explanation: str


class RecommendResponse(BaseModel):
    query: str
    profile: dict
    outfits: list[OutfitOut]
    style_tips: str
    processing_time_s: float


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLEWARE — request timing
# ══════════════════════════════════════════════════════════════════════════════

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.3f}s"
    return response


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
async def health_check():
    """Return API health status."""
    return {
        "status": "ok",
        "engine_loaded": state.engine is not None,
        "catalog_products": state.catalog_stats.get("total_products", 0),
    }


@app.get("/stats", tags=["System"])
async def get_stats():
    """Return dataset catalog statistics."""
    if not state.catalog_stats:
        raise HTTPException(status_code=503, detail="Catalog statistics not yet loaded.")
    return state.catalog_stats


@app.post("/intent", tags=["Debug"], response_model=dict)
async def extract_intent(req: IntentRequest):
    """
    Extract structured user intent from a natural-language fashion query.
    Useful for debugging the intent parser.
    """
    try:
        profile = extract_user_intent(req.query)
        return {"query": req.query, "profile": profile}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/recommend", tags=["Recommendation"], response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    Generate a complete outfit recommendation from a natural-language query.

    Returns up to `n_outfits` complete outfit combinations, each with:
      - topwear, bottomwear, footwear, accessories
      - compatibility score
      - natural-language styling explanation
    """
    if state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Recommendation engine not loaded. "
                "Run `python setup_pipeline.py` and restart the API."
            ),
        )

    try:
        state.engine.n_outfits = req.n_outfits
        result = state.engine.chat(req.query)

        outfits_out = [
            OutfitOut(
                topwear=outfit.get("topwear"),
                bottomwear=outfit.get("bottomwear"),
                footwear=outfit.get("footwear"),
                accessories=outfit.get("accessories", []),
                total_score=outfit.get("total_score", 0.0),
                explanation=outfit.get("explanation", ""),
            )
            for outfit in result.outfits
        ]

        return RecommendResponse(
            query=result.query,
            profile=result.profile,
            outfits=outfits_out,
            style_tips=result.style_tips,
            processing_time_s=result.processing_time_s,
        )

    except Exception as exc:
        logger.exception("Recommendation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
