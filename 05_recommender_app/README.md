# Project Scripts & Notebooks

This directory contains the complete PersonaPath analytics pipeline, including data processing, modeling, and the user interface.

| File | Purpose (Plain English) |
| :--- | :--- |
| **`01_data_processing.ipynb`** | **The Filter:** Cleans the raw Yelp data to extract the Philadelphia subset. |
| **`02_lda_topic_modeling.ipynb`** | **The Brain:** Trains the 25-topic behavioral model (removes 99 cuisine words). |
| **`03_feature_engineering.ipynb`** | **The Profiler:** Creates "Taste DNA" profiles for restaurants and users. |
| **`04_recommender_rag_and_interface.ipynb`** | **The Oracle:** Handles vector search and natural language explanations and **The Window:** The interactive Streamlit dashboard for the end-user. |

---

PersonaPath matches users to restaurants using a multi-layer recommendation engine:

- **Jensen-Shannon Divergence** similarity between user taste vectors and business profiles
- **Intent boosting** — detects mood signals in natural language (romantic, group, solo, hidden gems, etc.)
- **MMR reranking** — balances relevance with diversity across results
- **Live Google Places enrichment** — open/closed status, photos, reviews, phone, website
- **GPT-4o-mini briefing** — a concise AI-written rationale for each result set
- **Chat interface** — refine results conversationally after the initial recommendation

---

## 🚀 Quickstart

*04_recommender_rag_and_streamlit_interface_(final_code).py* is the final, self-contained interface — just run it.

### 1. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn scipy openai requests plotly pydeck
```

### 2. Add your API keys

Open `big_data.py` and replace the placeholders near the top:

```python
OPENAI_API_KEY    = "sk-..."        # OpenAI (GPT-4o-mini for briefings)
GOOGLE_PLACES_KEY = "AIza..."       # Google Places API (enrichment + map pins)
```

> Both keys are important — the core recommendation engine runs with them. Without OpenAI you lose AI briefings; without Google Places you lose live photos, hours, and map coordinates.

### 3. Point to your data

Update the file paths in `load_data()`:

```python
b_path = "data/business_profiles.csv"
u_path = "data/user_personas.csv"
```

### 4. Run

```bash
streamlit run big_data.py
```

---

## 📁 Data Requirements

The app expects two CSVs placed in a `data/` folder (or wherever you point `load_data()`):

| File | Key Columns |
|---|---|
| `business_profiles.csv` | `business_name`, `business_stars`, `categories`, `overall_sentiment`, `dominant_topic`, 25 topic score columns, `hours_monday` … `hours_sunday`, intent scores |
| `user_personas.csv` | `user_id`, `top_taste_1/2/3`, same 25 topic score columns |

The 25 topic columns are listed in `TOPIC_COLS` at the top of `big_data.py`.

---

## 🧠 How the Scoring Works

```
final_score = α × JSD_similarity + β × intent_boost + (1−α−β) × sentiment_norm
```

| Weight | Component | Default |
|---|---|---|
| α | User–restaurant topic similarity (JSD) | 0.50 |
| β | Query intent boost (romantic, group, etc.) | 0.30 |
| 1−α−β | Ensemble Adjusted Sentiment (EAS) | 0.20 |

Results are reranked with **MMR** (λ=0.65) to prevent similar restaurants from dominating the top 10.

---

## 🖥️ Interface Overview

| Panel | Description |
|---|---|
| Sidebar | Persona selector, mood query, Surprise Me toggle, result count |
| Recommendations tab | Map view + ranked restaurant cards with AI reasoning |
| Flavor Landscape tab | Radar chart of the user's top 8 taste dimensions |
| Export tab | Full result table + CSV download |
| Chat input | Follow-up questions answered in context of current results |

---

## ⚙️ Configuration

| Constant | Location | Purpose |
|---|---|---|
| `ALPHA`, `BETA` | Top of `big_data.py` | Scoring weights |
| `TOPIC_COLS` | Top of `big_data.py` | The 25 taste dimensions |
| `FEEDBACK_FILE` | Top of `big_data.py` | Path for thumbs up/down logging |
| `FOOD_IMAGES` | Top of `big_data.py` | Fallback images when Google Photos unavailable |

---

## 📦 Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| ML | scikit-learn, scipy, numpy |
| Data | pandas |
| Visualisation | Plotly, pydeck |
| AI Briefings | OpenAI GPT-4o-mini |
| Enrichment | Google Places API |

---


