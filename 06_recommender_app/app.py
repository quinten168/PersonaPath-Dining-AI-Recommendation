"""
PersonaPath — AI-Powered Dining Recommendations
================================================
Version: 3.4.1 (Updated: 2026-04-24 07:22 AM)
Run:
    pip install streamlit pandas numpy scikit-learn openai requests plotly pydeck
    streamlit run big_data.py
"""

import json
import csv
import os
import time
import urllib.parse
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import jensenshannon

st.set_page_config(page_title="PersonaPath", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")

import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    function cleanUI() {
        var doc = window.parent.document;
        var selectors = ['[data-testid="stSidebar"]', '[data-testid="stAppViewContainer"]', '[data-testid="collapsedControl"]'];
        selectors.forEach(sel => {
            var els = doc.querySelectorAll(sel);
            els.forEach(el => {
                var walker = doc.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
                var node;
                while (node = walker.nextNode()) {
                    var v = node.nodeValue.trim();
                    if (v.includes('double_arrow_right')) { node.nodeValue = '>>'; node.parentElement.style.color = '#000000'; }
                    else if (v.includes('double_arrow_left')) { node.nodeValue = '<<'; node.parentElement.style.color = '#000000'; }
                    else if (v.includes('keyboard') || v.includes('keyb')) node.nodeValue = '';
                }
                if (sel === '[data-testid="collapsedControl"]') {
                    el.style.color = '#000000';
                    var inner = el.querySelectorAll('*');
                    inner.forEach(c => { c.style.color = '#000000'; c.style.fill = '#000000'; });
                }
            });
        });
    }
    cleanUI(); setInterval(cleanUI, 200);
})();
</script>
""", height=0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@300;400;600;700&display=swap');

/* ── Download Button ── */
.stDownloadButton > button { 
    background-color: #ffffff !important; 
    color: #000000 !important; 
    border: 1px solid #e8ddd0 !important;
    font-weight: 900 !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-color: #c0392b !important;
}

/* ── Global Reset ── */
html, body { background-color: #f5ede0 !important; color: #1a1a1a !important; font-family: 'Inter', sans-serif !important; }
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; text-decoration: none !important; }
h1, h2, h3, .hero-title, .card-name, .section-label { font-family: 'Playfair Display', serif !important; }

.stApp, .main, .block-container, section.main, [data-testid="stAppViewContainer"] { background-color: #f5ede0 !important; }

/* ── Minimal Header Suppression ── */
header[data-testid="stHeader"] { background: transparent !important; }
.stDeployButton { display: none !important; }
#MainMenu { visibility: hidden !important; }

/* ── Sidebar (V3 Command Center) ── */
section[data-testid="stSidebar"] { 
    background-color: #161616 !important; 
    border-right: 2px solid #c0392b !important; 
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { padding-bottom: 120px !important; }
section[data-testid="stSidebar"] * { color: #f5ede0 !important; }
section[data-testid="stSidebar"] .sb-label { font-size: 0.72rem !important; font-weight: 700 !important; letter-spacing: 1.5px !important; color: #c8b89a !important; text-transform: uppercase !important; margin: 20px 0 10px !important; }
section[data-testid="stSidebar"] .stSelectbox > div, section[data-testid="stSidebar"] .stTextInput > div, section[data-testid="stSidebar"] .stMultiSelect > div { background: #262626 !important; border: 1px solid #444 !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] textarea { background: #262626 !important; border: 1px solid #444 !important; color: #f5ede0 !important; border-radius: 8px !important; }
section[data-testid="stSidebar"] .stButton > button { background: #c0392b !important; color: #fff !important; margin-top: 20px; font-weight: 900 !important; letter-spacing: 1px !important; }

/* ── Persistent Toggle Control ── */
button[data-testid="collapsedControl"], 
button[data-testid="collapsedControl"] *,
[data-testid="stSidebar"] button { 
    color: #000000 !important; 
    font-weight: 1000 !important;
    fill: #000000 !important;
}

/* ── Hero (Original Styling) ── */
.hero-wrap {
    background: linear-gradient(135deg, #7a0a00 0%, #c0392b 50%, #e85d4a 100%);
    border-radius: 14px;
    padding: 36px 44px;
    margin-bottom: 24px;
}
.hero-wrap * { color: #ffffff !important; }
.hero-eyebrow { font-size: 0.68rem; font-weight: 700; letter-spacing: 4px; color: rgba(255,255,255,0.65) !important; text-transform: uppercase; margin: 0 0 16px; display: block; }
.hero-title { font-size: 2.6rem; font-weight: 900; color: #ffffff !important; line-height: 1.1; letter-spacing: -1px; margin: 0 0 16px; display: block; }
.hero-tagline { font-size: 0.95rem; font-weight: 300; color: rgba(255,255,255,0.8) !important; margin: 0; display: block; }

/* ── Cards ── */
.rest-card { background: #ffffff !important; border: 1px solid #e8ddd0 !important; border-radius: 12px; padding: 18px 20px 14px; margin-bottom: 12px; }
.rest-card * { color: #1a1a1a !important; }
.card-rank { display: inline-block; background: #c0392b !important; color: #ffffff !important; border-radius: 50%; width: 26px; height: 26px; line-height: 26px; text-align: center; font-size: 0.72rem; font-weight: 700; margin-right: 8px; vertical-align: middle; }
.card-name { font-size: 1.15rem !important; font-weight: 700; color: #1a1a1a; }
.card-meta { font-size: 0.8rem; color: #666; margin: 3px 0 8px; }

/* ── Topic tags ── */
.topic-tag { display: inline-block; background: #fdf0e0; color: #8a4020; border: 1px solid #e8d0b8; border-radius: 20px; padding: 3px 11px; font-size: 0.72rem; font-weight: 600; margin-right: 5px; margin-bottom: 5px; }

/* ── AI reason ── */
.ai-reason { background: #fdf6ee; border-left: 3px solid #c0392b; border-radius: 0 8px 8px 0; padding: 9px 14px; font-size: 0.85rem; line-height: 1.65; color: #3a2010; font-style: normal; margin: 6px 0; }

/* ── Maps button ── */
.maps-btn { display: inline-block; background: #c0392b; color: #fff !important; padding: 6px 16px; border-radius: 6px; font-size: 0.76rem; font-weight: 700; text-decoration: none; margin-top: 8px; }
.maps-btn:hover { background: #a93226; }

/* ── Briefing ── */
.briefing { background: #ffffff; border: 1px solid #e8ddd0; border-left: 4px solid #c0392b; border-radius: 0 10px 10px 0; padding: 14px 20px; margin: 0 0 22px; font-size: 0.93rem; line-height: 1.8; color: #2a1a0a; box-shadow: 0 1px 6px rgba(0,0,0,0.04); }
.briefing strong { color: #1a1a1a; font-weight: 700; }

/* ── Section label ── */
.section-label { font-size: 0.68rem; font-weight: 700; letter-spacing: 2.5px; color: #c0392b; text-transform: uppercase; margin-bottom: 12px; }

/* ── Hours ── */
.hours-open  { color: #27ae60; font-weight: 700; font-size: 0.76rem; }
.hours-closed{ color: #c0392b; font-weight: 700; font-size: 0.76rem; }

/* ── Tabs ── */
button[data-baseweb="tab"] { font-size: 13px !important; font-weight: 600 !important; color: #888 !important; background: transparent !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #1a1a1a !important; background: transparent !important; }
[data-baseweb="tab-highlight"] { background-color: #c0392b !important; }
[data-baseweb="tab-list"] { background-color: transparent !important; }

/* ── Visibility Overrides (Fix White Text) ── */
[data-testid="stExpander"] * { color: #1a1a1a !important; }
.stMarkdownContainer p, .stCaption { color: #1a1a1a !important; }
div[data-testid="stStatusWidget"] * { color: #1a1a1a !important; }
.stSpinner * { color: #1a1a1a !important; }
[data-testid="stChatMessage"] * { color: #1a1a1a !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea { background-color: #ffffff !important; color: #1a1a1a !important; }
[data-testid="stChatInput"] * { background-color: transparent !important; color: #1a1a1a !important; }
div[data-testid="stChatInput"] > div { background-color: #ffffff !important; border-radius: 12px !important; }
.stChatFloatingInputContainer { background-color: #f5ede0 !important; border-top: 1px solid #e8ddd0 !important; }
.stChatInputContainer { background-color: #ffffff !important; border-radius: 12px !important; border: 1px solid #e8ddd0 !important; }
/* ── Fixed Chat Guide ── */
.chat-guide {
    position: fixed;
    bottom: 10px;
    left: 56.5%;
    transform: translateX(-50%);
    background-color: transparent !important;
    padding: 0 !important;
    z-index: 1001;
    width: fit-content;
    max-width: 600px;
    pointer-events: none;
}
.chat-guide div {
    font-size: 0.72rem !important;
    color: #888 !important;
    letter-spacing: 0.5px;
}
@media (max-width: 1200px) { .chat-guide { left: 50%; width: 90%; } }
</style>
""", unsafe_allow_html=True)

OPENAI_API_KEY    = "ENTER"
GOOGLE_PLACES_KEY = "ENTER"
FEEDBACK_FILE     = "feedback.csv"

TOPIC_COLS = [
    "bar_nightlife_crowd","bar_vibes_live_music","brunch_breakfast",
    "cafe_coffee_reading_terminal","casual_payment_ordering","chef_specials_platters",
    "cocktail_bars_speakeasy","comfort_food_sandwiches","craft_beer_sports_bars",
    "customer_service_quality","desserts_bakery","dinner_happy_hour",
    "fine_dining_tasting_menu","food_trucks_trendy_spots","markets_grocery_shopping",
    "outdoor_street_seating","overall_food_quality","philly_neighborhood_gems",
    "quick_fresh_lunch","service_wait_time","small_plates_cocktails",
    "spicy_asian_flavors","unique_quirky_dining","value_portion_size",
    "venue_location_experience",
]

TOPIC_LABELS = {
    "bar_nightlife_crowd":"🍺 Bar & Nightlife","bar_vibes_live_music":"🎵 Live Music",
    "brunch_breakfast":"🥞 Brunch","cafe_coffee_reading_terminal":"☕ Café & Coffee",
    "casual_payment_ordering":"🧾 Casual Dining","chef_specials_platters":"👨‍🍳 Chef Specials",
    "cocktail_bars_speakeasy":"🍸 Cocktails","comfort_food_sandwiches":"🥪 Comfort Food",
    "craft_beer_sports_bars":"🏈 Sports Bar","customer_service_quality":"⭐ Great Service",
    "desserts_bakery":"🧁 Desserts","dinner_happy_hour":"🌅 Happy Hour",
    "fine_dining_tasting_menu":"🍷 Fine Dining","food_trucks_trendy_spots":"🚚 Food Trucks",
    "markets_grocery_shopping":"🛒 Market","outdoor_street_seating":"🌿 Outdoor",
    "overall_food_quality":"🍽️ Food Quality","philly_neighborhood_gems":"📍 Philly Gem",
    "quick_fresh_lunch":"⚡ Quick Lunch","service_wait_time":"⏱️ Fast Service",
    "small_plates_cocktails":"🫒 Small Plates","spicy_asian_flavors":"🌶️ Asian Flavors",
    "unique_quirky_dining":"✨ Unique","value_portion_size":"💰 Great Value",
    "venue_location_experience":"🏙️ Great Location",
}

ALPHA = 0.5
BETA  = 0.3
DAYS  = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]

# ── Embedding-based similarity (additive, alongside JSD) ──────────────────
EMB_DIM       = 384
EMBED_COLS    = [f"emb_{i}" for i in range(EMB_DIM)]
HYBRID_WEIGHT = 0.5   # w in hybrid = w*jsd_sim_norm + (1-w)*embedding_sim_norm

FOOD_IMAGES = [
    "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&q=80",
    "https://images.unsplash.com/photo-1414235077428-338989a2e8c0?w=400&q=80",
    "https://images.unsplash.com/photo-1424847651672-bf20a4b0982b?w=400&q=80",
    "https://images.unsplash.com/photo-1559339352-11d035aa65de?w=400&q=80",
    "https://images.unsplash.com/photo-1544025162-d76694265947?w=400&q=80",
    "https://images.unsplash.com/photo-1555396273-367ea4eb4db5?w=400&q=80",
    "https://images.unsplash.com/photo-1466978913421-dad2ebd01d17?w=400&q=80",
    "https://images.unsplash.com/photo-1537047902294-62a40c20a6ae?w=400&q=80",
    "https://images.unsplash.com/photo-1552566626-52f8b828add9?w=400&q=80",
    "https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=400&q=80",
]


@st.cache_data(show_spinner="Loading profiles…")
def load_data():
    b_path = "data/profiles/business_profiles.csv"
    u_path = "data/profiles/user_profiles.csv"
    return pd.read_csv(b_path), pd.read_csv(u_path)


@st.cache_data(show_spinner="Building vectors…")
def build_matrix(business_df):
    df = business_df.copy()
    df["eas_norm"] = MinMaxScaler().fit_transform(df[["overall_sentiment"]].fillna(0))
    return df, df[TOPIC_COLS].fillna(0).values


@st.cache_data(show_spinner="Loading embeddings…")
def load_embeddings():
    """Additive, alongside the LDA topic vectors above. Returns (None, None)
    if the embedding artifacts haven't been generated yet
    (03_profile_building/embedding/build_profiles.py) so the app degrades
    gracefully to JSD-only rather than crashing."""
    biz_path  = "data/profiles/business_embeddings.csv"
    user_path = "data/profiles/user_embeddings.csv"
    if not (os.path.exists(biz_path) and os.path.exists(user_path)):
        return None, None
    return pd.read_csv(biz_path), pd.read_csv(user_path)


@st.cache_data(show_spinner="Building embedding vectors…")
def build_embedding_matrix(business_emb_df, business_df):
    """Aligns business_embeddings rows to business_df's row order so the
    resulting matrix indexes identically to biz_matrix (matrix[idx] with
    idx = candidate.index). Missing coverage -> zero vector (inert fallback:
    a zero vector always scores ~0 similarity, never crashes)."""
    if business_emb_df is None:
        return None
    aligned = business_df[["business_id"]].merge(business_emb_df, on="business_id", how="left")
    return aligned[EMBED_COLS].fillna(0).values


def get_top3_topics(row):
    scores = {c: float(row.get(c, 0)) for c in TOPIC_COLS}
    top3   = sorted(scores, key=scores.get, reverse=True)[:3]
    return [TOPIC_LABELS.get(t, t) for t in top3]


def format_hours(row):
    today_idx = datetime.now().weekday()
    DAY_ABBR  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    today_day = DAYS[today_idx]
    today_val = row.get(f"hours_{today_day}", None)

    if pd.isna(today_val) or not today_val:
        today_badge = "<span class='hours-closed'>&#x25CF; Closed today</span>"
    else:
        try:
            parts = str(today_val).split("-")
            def fmt_t(t):
                t = t.strip()
                if ":" in t:
                    h, m = t.split(":", 1)
                    return f"{int(h):02d}:{m.zfill(2)}"
                return t
            t = f"{fmt_t(parts[0])}-{fmt_t(parts[1])}" if len(parts)==2 else str(today_val)
            today_badge = f"<span class='hours-open'>&#x25CF; Open &nbsp;{t}</span>"
        except Exception:
            today_badge = "<span class='hours-open'>&#x25CF; Open today</span>"

    rows_html = ""
    for i, day in enumerate(DAYS):
        val  = row.get(f"hours_{day}", None)
        abbr = DAY_ABBR[i]
        is_today = (i == today_idx)
        day_style = "font-weight:700;color:#1a1a1a;" if is_today else "color:#555;"
        if pd.isna(val) or not val:
            h_str = "<span style='color:#c0392b;'>Closed</span>"
        else:
            try:
                parts = str(val).split("-")
                def fmt_time(t):
                    t = t.strip()
                    if ":" in t:
                        h, m = t.split(":", 1)
                        return f"{int(h):02d}:{m.zfill(2)}"
                    return t
                h_val = f"{fmt_time(parts[0])}-{fmt_time(parts[1])}" if len(parts)==2 else str(val)
                h_str = f"<span style='color:#3a2010;'>{h_val}</span>"
            except Exception:
                h_str = f"<span style='color:#3a2010;'>{val}</span>"
        rows_html += (
            f"<tr>"
            f"<td style='{day_style}font-size:0.72rem;padding:1px 12px 1px 0;white-space:nowrap;'>{abbr}</td>"
            f"<td style='font-size:0.72rem;padding:1px 0;'>{h_str}</td>"
            f"</tr>"
        )

    final_html = (
        '<div style="text-align:right;">'
        f'<div style="margin-bottom:4px;">{today_badge}</div>'
        f'<table style="border-collapse:collapse;margin-left:auto;">{rows_html}</table>'
        '</div>'
    )
    return final_html



@st.cache_data(ttl=86400, show_spinner=False)
def geocode(name, address="", city="Philadelphia"):
    try:
        q = urllib.parse.quote(f"{name}, {address}, {city}" if address else f"{name}, {city}")
        r = requests.get(
            f"https://nominatim.openstreetmap.org/search?q={q}&format=json&limit=1",
            headers={"User-Agent": "PersonaPath/1.0"}, timeout=5,
        ).json()
        if r:
            return float(r[0]["lat"]), float(r[0]["lon"])
    except Exception:
        pass
    return None, None

def _mmr_rerank(scores, matrix, top_k, lambda_=0.65):
    """MMR: balance relevance vs diversity. lambda_=0.65 → 65% relevance, 35% diversity."""
    selected, remaining = [], list(range(len(scores)))
    for _ in range(min(top_k, len(remaining))):
        if not selected:
            best = int(np.argmax(scores))
        else:
            redundancy = cosine_similarity(matrix[remaining], matrix[selected]).max(axis=1)
            mmr_scores = lambda_ * scores[remaining] - (1 - lambda_) * redundancy
            best       = remaining[int(np.argmax(mmr_scores))]
        selected.append(best)
        remaining.remove(best)
    return selected


def _verify_like(user_vec, user_tastes, biz_row, sim_score):
    """Predict how likely the user is to actually enjoy this restaurant (0–1)."""
    s_sim     = float(np.clip(sim_score, 0.0, 1.0))
    biz_dom   = str(biz_row.get("dominant_topic", "")).strip()
    s_persona = 1.0 if biz_dom and biz_dom in user_tastes else 0.0
    s_sent    = float(np.clip((float(biz_row.get("overall_sentiment", 0) or 0) + 1) / 2, 0, 1))
    prob = float(np.clip(0.60 * s_sim + 0.25 * s_persona + 0.15 * s_sent, 0, 1))
    return round(prob, 3)


def _embedding_similarity(user_vec, biz_matrix_sub):
    """Cosine similarity via dot product -- vectors are pre-normalized at
    encode time (normalize_embeddings=True in eval/embeddings.py) and
    re-normalized after mean-pooling, so a plain dot product IS cosine
    similarity here. Mirrors the inline JSD block in recommend() below for
    symmetry/readability."""
    norm = np.linalg.norm(user_vec)
    user_unit = user_vec / norm if norm > 0 else user_vec
    return biz_matrix_sub @ user_unit


def recommend(user_id, user_df, business_df, biz_matrix, user_query="", top_n=10, surprise=False,
              scoring_method="jsd", emb_matrix=None, user_emb_df=None, hybrid_weight=HYBRID_WEIGHT):
    # ── User lookup ──────────────────────────────────────────────────
    cols = ["user_id","top_taste_1","top_taste_2","top_taste_3"] + TOPIC_COLS
    avail    = [c for c in cols if c in user_df.columns]
    user_row = user_df[user_df["user_id"] == user_id][avail].head(1)
    if user_row.empty:
        st.error(f"user_id '{user_id}' not found.")
        return pd.DataFrame()

    user_vec_1d = user_row[TOPIC_COLS].fillna(0).values[0]
    user_tastes = [
        str(user_row["top_taste_1"].iloc[0] or "") if "top_taste_1" in user_row else "",
        str(user_row["top_taste_2"].iloc[0] or "") if "top_taste_2" in user_row else "",
        str(user_row["top_taste_3"].iloc[0] or "") if "top_taste_3" in user_row else "",
    ]
    ql        = user_query.lower()
    candidate = business_df.copy()

    # ── Cuisine pre-filter (unchanged) ───────────────────────────────
    cuisine_map = {
        "italian":["italian","pizza","pasta"],"korean":["korean"],"japanese":["japanese","sushi","ramen"],
        "chinese":["chinese","dim sum"],"mexican":["mexican","tacos"],"american":["american"],
        "seafood":["seafood"],"steakhouse":["steakhouse","steak"],"mediterranean":["mediterranean","greek"],
        "french":["french"],"indian":["indian"],"thai":["thai"],
    }
    for _, kws in cuisine_map.items():
        if any(k in ql for k in kws):
            mask = candidate["categories"].str.lower().str.contains("|".join(kws), na=False)
            if mask.sum() >= 5:
                candidate = candidate[mask]
            break

    # ── Layer 1: similarity (scoring_method: "jsd" | "embedding" | "hybrid") ──
    idx            = candidate.index
    biz_matrix_sub = biz_matrix[idx]
    eps       = 1e-10
    user_dist = user_vec_1d + eps
    user_dist = user_dist / user_dist.sum()
    jsd_sims = np.array([
        1.0 - jensenshannon(user_dist, biz_vec + eps)
        for biz_vec in biz_matrix_sub
    ])

    if scoring_method == "jsd":
        # Unchanged from the original JSD-only path -- default, no regression.
        sims = jsd_sims
        mmr_matrix_sub = biz_matrix_sub
    else:
        if emb_matrix is None or user_emb_df is None:
            raise ValueError(
                "scoring_method='%s' requires emb_matrix and user_emb_df "
                "(see load_embeddings()/build_embedding_matrix())" % scoring_method
            )
        emb_matrix_sub = emb_matrix[idx]
        user_emb_row = user_emb_df[user_emb_df["user_id"] == user_id][EMBED_COLS]
        embedding_sims = (
            np.zeros(len(idx)) if user_emb_row.empty
            else _embedding_similarity(user_emb_row.values[0], emb_matrix_sub)
        )
        if scoring_method == "embedding":
            sims, mmr_matrix_sub = embedding_sims, emb_matrix_sub
        elif scoring_method == "hybrid":
            jsd_norm = MinMaxScaler().fit_transform(jsd_sims.reshape(-1, 1)).ravel()
            emb_norm = MinMaxScaler().fit_transform(embedding_sims.reshape(-1, 1)).ravel()
            sims = hybrid_weight * jsd_norm + (1 - hybrid_weight) * emb_norm
            mmr_matrix_sub = emb_matrix_sub
        else:
            raise ValueError(f"unknown scoring_method: {scoring_method!r}")

    # ── Layer 2: Intent boost (expanded) ─────────────────────────────
    boost = pd.Series(0.0, index=idx)
    romantic = any(w in ql for w in ["romantic","date","dating","couple","intimate","cozy",
                                      "girlfriend","boyfriend","partner","anniversary","propose"])
    if romantic:
        boost += candidate["romantic_score"] * 2
        if any(w in ql for w in ["quiet","private","talk"]):
            boost += candidate["romantic_score"] * 1
    if any(w in ql for w in ["friend","group","beer","party","night","drink",
                               "birthday","celebrate","crew","bachelor","gathering"]):
        boost += candidate["group_celebration_score"] * 2
    if any(w in ql for w in ["family","kid","brunch","casual","children","parents"]):
        boost += candidate["family_score"] * 2
    if any(w in ql for w in ["quiet","work","coffee","study","solo","cafe","laptop","focus"]):
        boost += candidate["solo_work_score"] * 2
    if any(w in ql for w in ["hidden","local","gem","unique","authentic","underrated"]) or surprise:
        boost += candidate["hidden_gem_score"] * (4 if surprise else 2)
    if any(w in ql for w in ["fine dining","fancy","upscale","tasting menu","splurge","omakase"]):
        boost += candidate["fine_dining_tasting_menu"] * 1.5
    if any(w in ql for w in ["quick","fast","cheap","budget","affordable","takeout"]):
        boost += candidate["quick_fresh_lunch"] * 1.5
    if any(w in ql for w in ["outdoor","patio","terrace","rooftop","outside","al fresco"]):
        boost += candidate["outdoor_street_seating"] * 1.5
    if any(w in ql for w in ["late night","cocktail","nightlife","speakeasy","happy hour"]):
        boost += candidate["cocktail_bars_speakeasy"] * 1.5
    if boost.max() > 0:
        boost = boost / boost.max()
    boost_vals = boost.clip(0, 1).values

    # ── Layer 3: Final score ─────────────────────────────────────────
    scores = ALPHA * sims + max(1-ALPHA-BETA, 0) * candidate["eas_norm"].values + BETA * boost_vals
    if surprise:
        scores = scores * 0.6 + np.random.uniform(0, 0.15, size=scores.shape)

    # ── Layer 4: MMR reranking ───────────────────────────────────────
    pool_size = min(top_n * 3, len(scores))
    pool_idx  = np.argsort(scores)[::-1][:pool_size]
    mmr_idx   = _mmr_rerank(scores[pool_idx], mmr_matrix_sub[pool_idx], top_k=top_n)
    final_idx = pool_idx[mmr_idx]

    res = candidate.iloc[final_idx].copy()
    res["name"]              = res["business_name"]
    res["city"]              = "Philadelphia"
    res["stars"]             = res["business_stars"]
    res["dominant_vibe"]     = res.get("dominant_topic", "Great Dining")
    res["avg_sentiment"]     = res["overall_sentiment"] if "overall_sentiment" in res.columns else 0.5
    res["intent_romantic"]   = res["romantic_score"]
    res["intent_solo_work"]  = res["solo_work_score"]
    res["intent_family"]     = res["family_score"]
    res["intent_group"]      = res["group_celebration_score"]
    res["intent_hidden_gem"] = res["hidden_gem_score"]
    res["similarity_score"]  = sims[final_idx]
    res["query_boost"]       = boost_vals[final_idx]
    res["eas_norm"]          = candidate["eas_norm"].values[final_idx]
    res["final_score"]       = scores[final_idx]

    # ── Layer 5: verify_like ─────────────────────────────────────────
    res["predicted_like_prob"] = [
        _verify_like(user_vec_1d, user_tastes, row, row["similarity_score"])
        for _, row in res.iterrows()
    ]

    return res.reset_index(drop=True)


def _client(): return OpenAI(api_key=OPENAI_API_KEY)
def has_key(): return OPENAI_API_KEY and not OPENAI_API_KEY.startswith("YOUR_")


def call_briefing(top_k, user_query):
    if not has_key(): return ""
    names = ", ".join(top_k["name"].tolist()[:5])
    try:
        r = _client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"You are PersonaPath, a refined dining concierge."},
                {"role":"user","content":(
                    f'User asked: "{user_query}". Shortlisted: {names}. '
                    "Write 2 elegant sentences explaining why these picks suit the request. Mention restaurant names."
                )},
            ],
            max_tokens=140, temperature=0.7,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ""


def call_card_reasons(top_k, user_query):
    if not has_key(): return {}
    candidates = [{"name":r["name"],"categories":r["categories"],"stars":r["stars"],"dominant_vibe":r["dominant_vibe"]} for _,r in top_k.iterrows()]
    try:
        resp = _client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":(
                    "You are PersonaPath. For each restaurant write ONE vivid sentence (max 18 words) "
                    "explaining why it matches. "
                    'Return ONLY valid JSON: {"recommendations":[{"name":"...","reason":"..."}]}'
                )},
                {"role":"user","content":f'Request: "{user_query}"\n\nCandidates: {candidates}'},
            ],
            max_tokens=500, temperature=0.7,
            response_format={"type":"json_object"},
        )
        recs = json.loads(resp.choices[0].message.content).get("recommendations",[])
        return {r["name"]: r["reason"] for r in recs}
    except Exception:
        return {}


def call_followup(conversation, top_k, user_message):
    if not has_key(): return "Please add your OpenAI key."
    summary = top_k[["name","categories","stars","dominant_vibe"]].to_dict("records")
    messages = [{"role":"system","content":(f"You are PersonaPath, a dining concierge. Shortlist: {summary}. Answer in 2-3 sentences.")}] + conversation + [{"role":"user","content":user_message}]
    try:
        r = _client().chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=200, temperature=0.7)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def radar_chart(user_df, user_id):
    row = user_df[user_df["user_id"] == user_id][TOPIC_COLS].fillna(0)
    if row.empty: return None
    vals   = row.values[0]
    top8   = np.argsort(vals)[-8:][::-1]
    labels = [TOPIC_LABELS.get(TOPIC_COLS[i], TOPIC_COLS[i]) for i in top8]
    v      = vals[top8].tolist() + [vals[top8[0]]]
    labels = labels + [labels[0]]
    fig = go.Figure(go.Scatterpolar(r=v, theta=labels, fill="toself",
        fillcolor="rgba(192,57,43,0.12)", line=dict(color="#c0392b", width=2)))
    fig.update_layout(
        polar=dict(bgcolor="#111", radialaxis=dict(visible=True, range=[0,max(v)*1.15],
            gridcolor="#2a2a2a", tickfont=dict(color="#555",size=9)),
            angularaxis=dict(tickfont=dict(color="#c8b89a",size=10))),
        paper_bgcolor="#0e0e0e", height=320,
        margin=dict(l=50,r=50,t=40,b=40), showlegend=False,
    )
    return fig


def map_view(top_k, places: dict):
    rows = []
    for i, (_, r) in enumerate(top_k.iterrows()):
        pd_data = places.get(r["name"], {})
        lat, lng = pd_data.get("lat"), pd_data.get("lng")
        if lat and lng:
            rows.append({"name":r["name"],"stars":r["stars"],"vibe":r["dominant_vibe"],
                         "score":round(r["final_score"],3),"rank":i+1,"lat":lat,"lon":lng})

    if not rows:
        st.info("Could not locate any restaurants."); return

    df_map = pd.DataFrame(rows)
    layer  = pdk.Layer("ScatterplotLayer", data=df_map,
        get_position=["lon","lat"], get_color=[192,57,43,230],
        get_radius=120, pickable=True, auto_highlight=True)
    view = pdk.ViewState(latitude=df_map["lat"].mean(), longitude=df_map["lon"].mean(), zoom=13)
    tooltip = {"html":"<b>#{rank} {name}</b><br/>⭐ {stars} · {vibe}<br/>Score: {score}",
               "style":{"background":"#111","color":"#f0e8d8","font-family":"Inter",
                        "padding":"8px","border-radius":"6px","font-size":"12px"}}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"))
    st.markdown(f'<p style="color:#555;font-size:0.82rem;margin-top:6px;">📍 Verified Neighborhood Mapping via Google Places Data</p>', unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner=False)
def enrich_places(name: str, city: str = "Philadelphia") -> dict:
    if not GOOGLE_PLACES_KEY or GOOGLE_PLACES_KEY.startswith("YOUR_"): return {}
    try:
        q = urllib.parse.quote(f"{name} {city}")
        resp = requests.get(f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={q}&key={GOOGLE_PLACES_KEY}", timeout=5).json()
        results = resp.get("results", [])
        if not results: return {}
        
        place_id = results[0]["place_id"]
        loc      = results[0].get("geometry", {}).get("location", {})
        lat, lng = loc.get("lat"), loc.get("lng")

        # Step 2 — Place Details
        fields = "opening_hours,formatted_phone_number,website,price_level,photos,reviews,rating"
        detail = requests.get(f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields={fields}&key={GOOGLE_PLACES_KEY}", timeout=5).json().get("result", {})
        
        photo_url = ""
        if detail.get("photos"):
            ref = detail["photos"][0]["photo_reference"]
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={ref}&key={GOOGLE_PLACES_KEY}"
        
        reviews = [rv.get("text", "") for rv in detail.get("reviews", [])[:2] if rv.get("text")]

        return {
            "open_now"   : detail.get("opening_hours", {}).get("open_now"),
            "phone"      : detail.get("formatted_phone_number", ""),
            "website"    : detail.get("website", ""),
            "price_level": detail.get("price_level"),
            "g_rating"   : detail.get("rating"),
            "photo_url"  : photo_url,
            "reviews"    : reviews,
            "lat"        : lat,
            "lng"        : lng
        }
    except Exception: return {}


def price_label(level) -> str:
    return "$" * int(level) if level else ""


def log_feedback(user_id, name, rating):
    exists = os.path.exists(FEEDBACK_FILE)
    with open(FEEDBACK_FILE,"a",newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(["timestamp","user_id","business_name","rating"])
        w.writerow([datetime.now().isoformat(), user_id, name, rating])


def render_card(row, rank, reason, user_id, pd_data: dict):
    name, stars, addr = row["name"], row["stars"], str(row.get("address","")).strip()
    top3, hours = get_top3_topics(row), format_hours(row)
    star_str = "★" * min(int(round(stars)),5) + "☆" * (5 - min(int(round(stars)),5))
    maps_url = f"https://www.google.com/maps/search/?q={urllib.parse.quote(name+' Philadelphia')}"
    
    # Enrichment
    p_url, g_ph = pd_data.get("photo_url", ""), pd_data.get("phone", "")
    price, g_rt = price_label(pd_data.get("price_level")), pd_data.get("g_rating")
    img_url = p_url if p_url else FOOD_IMAGES[(rank-1) % len(FOOD_IMAGES)]
    
    open_html = ""
    if pd_data.get("open_now") is True: open_html = '<span style="color:#27ae60;font-size:0.72rem;font-weight:700;margin-right:8px;">● Open Now</span>'
    elif pd_data.get("open_now") is False: open_html = '<span style="color:#c0392b;font-size:0.72rem;font-weight:700;margin-right:8px;">● Closed</span>'
    
    price_html = f'<span style="background:#f0ede8;color:#5a4a3a;border-radius:4px;padding:1px 6px;font-size:0.7rem;font-weight:700;margin-right:8px;">{price}</span>' if price else ""
    g_rating_html = f'<span style="color:#555;font-size:0.72rem;margin-right:8px;">🔵 Google {g_rt}★</span>' if g_rt else ""
    phone_html = f'<span style="font-size:0.85rem;color:#1a1a1a;font-weight:600;">📞 {g_ph}</span>' if g_ph else ""

    col_img, col_info = st.columns([1, 3])
    with col_img:
        st.markdown(f'<div style="border-radius:10px;overflow:hidden;height:130px;"><img src="{img_url}" style="width:100%;height:130px;object-fit:cover;"/></div>', unsafe_allow_html=True)

    with col_info:
        addr_html = f'&nbsp;&middot;&nbsp;<span style="color:#666;">{addr}</span>' if addr else ""
        topics_html = ''.join(f'<span class="topic-tag">{t}</span>' for t in top3)
        w_link = pd_data.get("website", "")
        w_html = f'<a href="{w_link}" target="_blank" style="font-size:0.76rem;font-weight:700;color:#c0392b;margin-left:10px;text-decoration:none;">🌐 Website</a>' if w_link else ""
        
        # Build main card string separately for safety
        ct = f'<div class="rest-card">'
        ct += f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">'
        ct += f'<div style="flex:1;">'
        ct += f'<div class="card-name"><span class="card-rank">{rank}</span>{name}</div>'
        ct += f'<div class="card-meta">'
        ct += f'<span style="color:#c0392b;">{star_str}</span>&nbsp;{stars}&nbsp;&middot;&nbsp;Philadelphia{addr_html}'
        ct += f'</div>'
        ct += f'<div style="margin:5px 0 8px;">{open_html}{price_html}{g_rating_html}{phone_html}</div>'
        ct += f'<div style="margin:8px 0;">{topics_html}</div>'
        ct += f'<div style="margin-top:12px;">'
        ct += f'<a class="maps-btn" href="{maps_url}" target="_blank">📍 Open in Google Maps</a>{w_html}'
        ct += f'</div></div>'
        ct += f'<div style="min-width:140px;flex-shrink:0;">{hours}</div>'
        ct += f'</div></div>'
        
        st.markdown(ct, unsafe_allow_html=True)

    if reason:
        st.markdown(f'<div class="ai-reason">🍽️ {reason}</div>', unsafe_allow_html=True)

    # Review Expander
    reviews = pd_data.get("reviews", [])
    if reviews:
        st.markdown('<p style="font-family:\'Playfair Display\', serif !important; font-size:0.85rem; font-weight:700; color:#1a1a1a; margin:12px 0 4px;">💬 What People are Saying</p>', unsafe_allow_html=True)
        with st.expander("Read Google Reviews"):
            for rv in reviews:
                st.markdown(
                    f'<div style="background:#fdf6ee;border-left:3px solid #e8d0b0;'
                    f'border-radius:0 8px 8px 0;padding:8px 12px;font-size:0.82rem;'
                    f'color:#3a2010;line-height:1.55;margin-bottom:6px;">'
                    f'"{rv[:260]}{"…" if len(rv)>260 else ""}"</div>',
                    unsafe_allow_html=True,
                )

    fb_key = f"fb_{name}_{rank}"
    if fb_key not in st.session_state: st.session_state[fb_key] = None
    ca, cb, _ = st.columns([1,1,10])
    with ca:
        if st.button("👍", key=f"up_{fb_key}"): st.session_state[fb_key]="up"; log_feedback(user_id,name,"up")
    with cb:
        if st.button("👎", key=f"dn_{fb_key}"): st.session_state[fb_key]="down"; log_feedback(user_id,name,"down")
    if   st.session_state[fb_key]=="up":   st.markdown('<p style="color:#555;font-size:0.8rem;">✅ Thanks!</p>', unsafe_allow_html=True)
    elif st.session_state[fb_key]=="down": st.markdown('<p style="color:#555;font-size:0.8rem;">📝 Noted.</p>', unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #e8ddd0;margin:6px 0;'/>", unsafe_allow_html=True)


def main():
    # ── 1. SIDEBAR COMMAND CENTER ─────────────────────────────────────
    with st.sidebar:
        # Branding
        st.markdown('<div style="text-align: center; padding: 10px 0;"><h1 style="color: #c0392b; margin:0; font-size: 1.8rem; letter-spacing: -1px;">PersonaPath</h1><p style="color: #887a6a; font-size: 0.78rem; margin: 4px 0 0; letter-spacing: 0.5px;">Dining curated for you.</p></div>', unsafe_allow_html=True)
        
        # Profile Switcher
        st.markdown('<p class="sb-label">👤 Dining Profile</p>', unsafe_allow_html=True)
        PRESETS = {
            "🍜 The Foodie"   : "11XTXYmhkEFo7SwEb2ByCQ",
            "🕯️ Date Night"  : "GSDw9czWBk_GL53a5NtcIg",
            "🥂 Group Hangout": "V6BrQCclOjOwkJNVqMFNEg",
            "☕ Solo Explorer": "acMsMOJ-FJtl6mPYCkbQ_g",
            "🛠️ Custom Token" : "CUSTOM"
        }
        persona = st.selectbox("Select Active Persona", list(PRESETS.keys()), label_visibility="collapsed")
        user_id = PRESETS[persona]
        if persona == "🛠️ Custom Token":
            user_id = st.text_input("Enter User Token", placeholder="Yelp ID format...")

        # The Inquiry (Mood + Filters)
        st.markdown('<p class="sb-label">✨ What is your mood today?</p>', unsafe_allow_html=True)
        base_query = st.text_area("Question", value="I'm looking for a romantic dinner spot", 
                                  height=70, label_visibility="collapsed", placeholder="What are you looking for?")
        
        full_query = base_query

        # Discovery Engine
        st.markdown('<p class="sb-label">🎡 Discovery Engine</p>', unsafe_allow_html=True)
        st.markdown("""
        <style>
        section[data-testid="stSidebar"] [data-testid="stToggle"] label {
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            color: #f5ede0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stToggle"] div[data-baseweb="toggle"] {
            background-color: #444 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stToggle"] div[aria-checked="true"] {
            background-color: #c0392b !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] {
            padding: 2px 0 0 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
            background: #c0392b !important;
            width: 13px !important;
            height: 13px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        surprise = st.checkbox("💎 Surprise Me — Hidden Gems", value=False)
        st.markdown('<p class="sb-label" style="margin-top:15px; margin-bottom:5px;">Volume of Results</p>', unsafe_allow_html=True)
        top_n = st.slider("Result Count", 3, 20, 10, label_visibility="collapsed")
        st.markdown("<br/>", unsafe_allow_html=True)
        run_btn = st.button("✦ GET RECOMMENDATIONS", use_container_width=True, type="primary")

    # ── 2. HERO SECTION ───────────────────────────────────────────────
    st.markdown("""
        <div class="hero-wrap">
            <div class="hero-eyebrow">THE ORACLE &nbsp;·&nbsp; PHILADELPHIA</div>
            <div class="hero-title">Find Your PersonaPath</div>
            <div class="hero-tagline">
                AI-Driven Dining Curation for Your Culinary Scenes.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── 3. DATA & STATE ───────────────────────────────────────────────
    try:
        biz_raw, u_df = load_data()
        biz_df, b_mat = build_matrix(biz_raw)
    except Exception as e:
        st.error(f"Data link failure: {e}"); return

    # Embedding-based similarity is additive: if the artifacts don't exist yet
    # (eval/embeddings.py hasn't been run), the app still works exactly as
    # before with scoring_method="jsd" (the default below).
    try:
        biz_emb_raw, u_emb_df = load_embeddings()
        emb_mat = build_embedding_matrix(biz_emb_raw, biz_df)
    except Exception:
        emb_mat, u_emb_df = None, None

    # Init session state
    for k, v in [("top_k",None),("llm_reasons",{}),("briefing",""),
                 ("chat_history",[]),("user_id",user_id),("user_query",full_query)]:
        if k not in st.session_state: st.session_state[k] = v

    # ── 4. RECOMMENDATION TRIGGER ─────────────────────────────────────
    if run_btn:
        st.session_state.chat_history = []
        st.session_state.user_id      = user_id
        st.session_state.user_query   = full_query
        with st.spinner("Calculating culinary vectors…"):
            top_k = recommend(user_id=user_id, user_df=u_df, business_df=biz_df,
                               biz_matrix=b_mat, user_query=full_query, top_n=top_n, surprise=surprise,
                               emb_matrix=emb_mat, user_emb_df=u_emb_df)
        
        if not top_k.empty:
            with st.spinner("Enriching with live Google data…"):
                places = {row["name"]: enrich_places(row["name"]) for _, row in top_k.iterrows()}
            
            with st.spinner("Mapping your taste journey…"):
                briefing    = call_briefing(top_k, full_query)
                llm_reasons = call_card_reasons(top_k, full_query)
            st.session_state.update(top_k=top_k, briefing=briefing, llm_reasons=llm_reasons, places=places)
        else:
            st.error("The Oracle found no matches for this specific combination.")

    # ── 5. RESULT DISPLAY ──────────────────────────────────────────────
    if st.session_state.top_k is not None:
        tk     = st.session_state.top_k
        recs   = st.session_state.llm_reasons
        brief  = st.session_state.get("briefing", "")
        uid    = st.session_state.get("user_id", "")
        places = st.session_state.get("places", {})

        if brief:
            import re
            clean_brief = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', brief)
            st.markdown(f'<div class="briefing">✦ &nbsp;<strong>The Oracle\'s Briefing</strong><br/><br/>{clean_brief}</div>', unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["✦ Recommendations", "🏞️ Flavor Landscape", "💾 Export"])
        
        with t1:
            st.markdown('<p class="section-label">📍 Neighborhood Visualization</p>', unsafe_allow_html=True)
            map_view(tk, places)
            st.markdown("<br/>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">🏆 Your Top Candidates</p>', unsafe_allow_html=True)
            for i in range(len(tk)):
                row = tk.iloc[i]
                render_card(row=row, rank=i+1, reason=recs.get(row["name"],""), user_id=uid, pd_data=places.get(row["name"], {}))

        with t2:
            st.markdown('<p class="section-label">Your Culinary Fingerprint</p>', unsafe_allow_html=True)
            fig = radar_chart(u_df, uid)
            if fig: st.plotly_chart(fig, use_container_width=True)

        with t3:
            st.markdown('<p class="section-label">Data Manifest</p>', unsafe_allow_html=True)
            st.dataframe(tk[["name","stars","categories","final_score"]], use_container_width=True, hide_index=True)
            csv_data = tk.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Result CSV", data=csv_data, file_name="personapath_v3.csv", use_container_width=True)

        # Sticky Chat Guidance - Pinned to absolute bottom
        st.markdown("""
            <div class="chat-guide">
                <div>💬 <strong>Refine Your Path</strong> &nbsp;·&nbsp; Filter, compare, or dig deeper into the results.</div>
            </div>
        """, unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.write(msg["content"])

        if prompt := st.chat_input("Ask the Oracle about these results..."):
            st.session_state.chat_history.append({"role":"user","content":prompt})
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    reply = call_followup(st.session_state.chat_history[:-1], tk, prompt)
                st.write(reply)
            st.session_state.chat_history.append({"role":"assistant","content":reply})

    else:
        st.markdown("""
            <div style="text-align:center;padding:80px 20px;color:#333;">
                <p style="font-size:3rem;">🍽️</p>
                <p style="font-family:'Playfair Display',serif;font-size:1.3rem;color:#555;">
                    Configure your preferences and discover your next favourite table.
                </p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
