"""
app.py — AI Fashion Outfit Recommendation System — Streamlit Interface.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config import APP_ICON, APP_TITLE, MAX_HISTORY, TOP_K_OUTFITS
from utils.helpers import color_badge_html, format_price, format_rating, truncate

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
<style>
/* ── Global ── */
:root {
    --primary: #7C3AED;
    --primary-light: #EDE9FE;
    --surface: #FAFAFA;
    --card-bg: #FFFFFF;
    --card-border: #E5E7EB;
    --text-main: #111827;
    --text-muted: #6B7280;
    --accent: #F59E0B;
    --success: #10B981;
    --radius: 14px;
    --shadow: 0 4px 18px rgba(0,0,0,0.08);
}

/* ── Chat bubbles ── */
.user-bubble {
    background: var(--primary);
    color: #fff;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 78%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(124,58,237,.25);
}
.assistant-bubble {
    background: var(--primary-light);
    color: var(--text-main);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 18px;
    margin: 8px 0;
    max-width: 78%;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ── Outfit card ── */
.outfit-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 16px;
    box-shadow: var(--shadow);
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.outfit-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.outfit-card img {
    width: 100%;
    max-height: 220px;
    object-fit: cover;
    border-radius: 10px;
    margin-bottom: 10px;
    background: #f3f4f6;
}
.card-category {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 4px;
}
.card-name {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text-main);
    margin-bottom: 4px;
    line-height: 1.3;
}
.card-brand {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 6px;
}
.card-price {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-main);
}
.card-rating {
    font-size: 0.75rem;
    color: var(--accent);
    margin-bottom: 4px;
}

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    background: var(--primary);
    color: #fff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 8px;
}

/* ── Explanation box ── */
.explanation-box {
    background: linear-gradient(135deg, #F5F3FF 0%, #EDE9FE 100%);
    border-left: 4px solid var(--primary);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 0.9rem;
    line-height: 1.6;
    color: var(--text-main);
    margin: 12px 0;
}

/* ── Profile pill ── */
.profile-chip {
    display: inline-block;
    background: #F3F4F6;
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 3px 3px 3px 0;
}
.profile-chip strong {
    color: var(--primary);
}

/* ── Style tips ── */
.tips-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: var(--radius);
    padding: 14px 18px;
    font-size: 0.88rem;
    line-height: 1.7;
}

/* ── Divider ── */
.outfit-divider {
    border: none;
    height: 2px;
    background: linear-gradient(to right, var(--primary-light), transparent);
    margin: 20px 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface);
}

/* ── Input box ── */
.stChatInputContainer {
    padding-top: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════

def _init_session() -> None:
    defaults = {
        "messages": [],          # list of {role, content, outfits, profile, style_tips}
        "engine": None,
        "engine_loaded": False,
        "engine_error": None,
        "processing": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session()

# ══════════════════════════════════════════════════════════════════════════════
# ENGINE LOADER
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _load_engine():
    """Load ChatEngine once, cache across sessions."""
    from assistant.chat_engine import ChatEngine
    engine = ChatEngine()
    engine.load()
    return engine


def _get_engine():
    if not st.session_state.engine_loaded:
        with st.spinner("🔄 Loading Fashion AI models (first run may take ~30s)…"):
            try:
                st.session_state.engine = _load_engine()
                st.session_state.engine_loaded = True
                st.session_state.engine_error = None
            except FileNotFoundError as exc:
                st.session_state.engine_error = str(exc)
                st.session_state.engine_loaded = False
    return st.session_state.engine


# ══════════════════════════════════════════════════════════════════════════════
# RENDERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _render_product_card(item: dict, label: str) -> None:
    """Render a single product card with image, name, brand, price."""
    if not item:
        st.markdown(
            f'<div class="outfit-card"><p class="card-category">{label}</p>'
            f'<p style="color:#9CA3AF;font-size:0.8rem;">Not available</p></div>',
            unsafe_allow_html=True,
        )
        return

    color_badge = color_badge_html(item.get("dominant_color", "unknown"))
    score_pct = int(item.get("compat_score", 0) * 100)

    st.markdown(
        f"""
<div class="outfit-card">
    <p class="card-category">{label}</p>
    <img src="{item.get('image_src', '')}" alt="{item.get('name', '')}"
        onerror="this.src='https://via.placeholder.com/280x360?text=No+Image'"/>
    <div class="card-name">{truncate(item.get('name',''), 60)}</div>
    <div class="card-brand">{item.get('brand','')}</div>
    <div class="card-rating">{item.get('rating','')}</div>
    <div class="card-price">{item.get('price','')}</div>
    <div style="margin-top:6px;">{color_badge}</div>
    <div style="margin-top:8px;">
        <span class="score-badge">Match {score_pct}%</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_outfit_block(outfit_display: dict, outfit_idx: int) -> None:
    """Render a full outfit (top + bottom + shoes + accessories + explanation)."""
    n = outfit_display["number"]
    score_pct = outfit_display["total_score_pct"]

    st.markdown(f"#### Outfit {n} &nbsp; <span class='score-badge'>Overall Match {score_pct}%</span>",
                unsafe_allow_html=True)

    # ── Explanation ────────────────────────────────────────────────────────
    if outfit_display.get("explanation"):
        st.markdown(
            f'<div class="explanation-box">💡 {outfit_display["explanation"]}</div>',
            unsafe_allow_html=True,
        )

    # ── Product grid ───────────────────────────────────────────────────────
    cols = st.columns(4, gap="medium")
    labels = ["👕 Topwear", "👖 Bottomwear", "👟 Footwear", "👜 Accessory"]
    items = [
        outfit_display.get("topwear", {}),
        outfit_display.get("bottomwear", {}),
        outfit_display.get("footwear", {}),
        (outfit_display.get("accessories") or [{}])[0],
    ]

    for col, item, label in zip(cols, items, labels):
        with col:
            _render_product_card(item, label)

    # Show second accessory if present
    accessories = outfit_display.get("accessories", [])
    if len(accessories) > 1:
        with st.expander("➕ More accessories"):
            acc_cols = st.columns(min(len(accessories) - 1, 3))
            for col, acc in zip(acc_cols, accessories[1:]):
                with col:
                    _render_product_card(acc, "👜 Accessory")

    st.markdown('<hr class="outfit-divider"/>', unsafe_allow_html=True)


def _render_profile_chips(profile: dict) -> None:
    """Render the extracted user profile as compact pills."""
    chips_html = ""
    fields = [
        ("Gender",   profile.get("gender", "")),
        ("Occasion", profile.get("occasion", "")),
        ("Style",    profile.get("style", "")),
        ("Season",   profile.get("season", "")),
        ("Age",      profile.get("age", "")),
    ]
    for label, value in fields:
        if value and value not in ("unisex", "all", ""):
            chips_html += f'<span class="profile-chip"><strong>{label}:</strong> {value}</span> '
    if chips_html:
        st.markdown(
            f'<div style="margin:6px 0 12px;">{chips_html}</div>',
            unsafe_allow_html=True,
        )


def _render_style_tips(tips: str) -> None:
    """Render style tips in a highlighted box."""
    if tips:
        st.markdown(
            f'<div class="tips-box">✨ <strong>Style Tips</strong><br/>{tips}</div>',
            unsafe_allow_html=True,
        )


def _render_chat_history() -> None:
    """Render all messages in the conversation history."""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🧑 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">👗 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Render extracted profile
            if msg.get("profile"):
                _render_profile_chips(msg["profile"])

            # Render outfit recommendations
            from utils.helpers import outfit_to_display
            for i, outfit in enumerate(msg.get("outfits", []), 1):
                _render_outfit_block(outfit_to_display(outfit, i), i)

            # Render style tips
            if msg.get("style_tips"):
                _render_style_tips(msg["style_tips"])


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://play-lh.googleusercontent.com/qpHoKy_VeBQn8Hf2xTSudWqIH2aNZGEFdQFZJX3Dkarwfu7YNnwrmtEQuC_aDUCkF6luGgG53K3QEotWwsUffQ",
             use_container_width=True)
    st.markdown("---")

    st.markdown("### 💬 Conversation")
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        if st.session_state.engine:
            st.session_state.engine.clear_history()
        st.rerun()

    st.markdown("---")
    st.markdown("### 🎯 Example Queries")
    example_queries = [
        "I need an outfit for a business meeting.",
        "Suggest a smart casual dinner date look.",
        "I'm attending a wedding next weekend.",
        "22-year-old male, casual summer outfit.",
        "Something ethnic for Diwali for women.",
        "Chic party outfit for a Saturday night.",
    ]
    for eq in example_queries:
        if st.button(eq, use_container_width=True, key=f"eg_{eq[:20]}"):
            st.session_state["_example_query"] = eq
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    n_outfits = st.slider(
        "Number of outfit options", min_value=1, max_value=3,
        value=TOP_K_OUTFITS, step=1,
    )

    st.markdown("---")
    st.markdown(
        "<small style='color:#9CA3AF;'>"
        "Powered by Gemini · FAISS · Sentence Transformers"
        "</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"# {APP_TITLE}")
st.markdown(
    "<p style='color:#6B7280;margin-top:-10px;'>"
    "Your AI-powered personal stylist. Ask for outfit ideas and I'll curate "
    "complete looks with product recommendations and styling explanations."
    "</p>",
    unsafe_allow_html=True,
)

# ── Engine status ──────────────────────────────────────────────────────────────
engine = _get_engine()

if st.session_state.engine_error:
    st.error(
        f"⚠️ **Could not load the recommendation engine.**\n\n"
        f"{st.session_state.engine_error}\n\n"
        "**Quick fix:**\n"
        "```bash\n"
        "# 1. Place products.csv and outfits.csv in data/\n"
        "# 2. Place product images in images/\n"
        "# 3. Run the setup pipeline:\n"
        "python setup_pipeline.py\n"
        "# 4. Restart the app:\n"
        "streamlit run app.py\n"
        "```"
    )
    st.stop()

# ── Chat container ─────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.markdown(
            """
<div style="text-align:center;padding:40px 20px;color:#9CA3AF;">
    <div style="font-size:3rem;">👗</div>
    <h3 style="color:#374151;">Welcome to your AI Fashion Assistant</h3>
    <p>Tell me what you need — an occasion, your age, gender, preferred style,
    season, or colour — and I'll build a complete outfit just for you.</p>
    <p style="font-size:0.85rem;">Try: <em>"I need an outfit for a business meeting"</em></p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        _render_chat_history()

# ── Handle example query click ─────────────────────────────────────────────────
if "_example_query" in st.session_state:
    user_input = st.session_state.pop("_example_query")
else:
    user_input = None

# ── Chat input ─────────────────────────────────────────────────────────────────
prompt = st.chat_input("Describe the outfit you need…", disabled=st.session_state.processing)

if prompt:
    user_input = prompt

if user_input:
    # ── Add user message ───────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.processing = True
    st.rerun()

# ── Process pending user message ──────────────────────────────────────────────
if st.session_state.processing and st.session_state.messages:
    last_user = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "user"), None
    )
    last_assistant = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None
    )

    # Only process if the last message is from the user
    if last_user and (not last_assistant or
                      st.session_state.messages[-1]["role"] == "user"):

        with st.spinner("🧵 Curating your perfect outfit…"):
            try:
                # Update engine's n_outfits from sidebar slider
                engine.n_outfits = n_outfits

                result = engine.chat(last_user["content"])

                summary = (
                    f"Here {'are' if len(result.outfits) > 1 else 'is'} "
                    f"{len(result.outfits)} outfit recommendation"
                    f"{'s' if len(result.outfits) > 1 else ''} for "
                    f"**{result.profile.get('occasion','your occasion')}** "
                    f"in **{result.profile.get('style','')}** style. "
                    f"_(Processed in {result.processing_time_s}s)_"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": summary,
                    "profile": result.profile,
                    "outfits": result.outfits,
                    "style_tips": result.style_tips,
                })

            except Exception as exc:
                logger.exception("Pipeline error: %s", exc)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"😔 I ran into an issue while generating your outfit: `{exc}`. "
                        "Please try again or rephrase your request."
                    ),
                    "profile": {},
                    "outfits": [],
                    "style_tips": "",
                })
        st.session_state.processing = False
        st.rerun()
