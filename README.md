# PersonaPath: Personalized Dining Recommendation Engine
### *Beyond Proximity: A Behavioral-Driven Discovery Engine for the Modern Diner*

[![PersonaPath](https://img.shields.io/badge/PersonaPath-v1.0-blue?style=for-the-badge&logo=openai)](https://github.com/DhairyaLunia/Team5_Big_data)
[![Apache Spark](https://img.shields.io/badge/Distributed_Processing-Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![scikit-learn](https://img.shields.io/badge/Behavioral_Modeling-TF--IDF_%2B_Embeddings-blue?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)
[![Cosine Similarity](https://img.shields.io/badge/Vector_Similarity-Cosine_Similarity-0433FF?style=for-the-badge)](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity)

---

## 🌟 Executive Summary
Traditional recommendation platforms (like Yelp or Google Maps) answer the question: *"What is nearby?"* **PersonaPath** answers the question: *"Who am I, and where should I go?"*

Built by **Team 5 at the Carlson School of Management (MSBA)**, PersonaPath is a recommendation engine that builds deep behavioral profiles from Yelp reviews. By combining **TF-IDF profile similarity** (interpretable, and the best-performing method on our own evaluation — see [`ARCHITECTURE.md`](./ARCHITECTURE.md)), **sentence embeddings** for semantic matching, and **live free-text prompt matching**, we deliver personalized dining matches a user can ask for in their own words.

> The engine originally launched on a Philadelphia review subset using LDA topic modeling + Jensen-Shannon Divergence similarity. That pipeline has since been retired in favor of TF-IDF, which our own evaluation harness showed outperforms it on every ranking metric — the project is now being extended to a New Orleans review subset.

---

## 🏗️ The PersonaPath Engine: Scoring Architecture
Our recommendation logic blends historical taste, live intent, and business quality into one ranked score.

| Layer | Component | Weight | Logic |
| :--- | :--- | :--- | :--- |
| **Historical Profile Similarity** | TF-IDF Behavioral DNA | primary | Cosine similarity between a user's and a business's TF-IDF review-term profile — literal, interpretable, and (per our own evaluation) the best-performing similarity method we've tried. |
| | + Sentence-Embedding Similarity | optional blend | Cosine similarity over `all-MiniLM-L6-v2` review embeddings, blended in alongside TF-IDF for semantic/paraphrase coverage TF-IDF alone misses. |
| **Live Prompt Match** | Free-text query matching | 0.75 within the blend above | A user's live prompt ("find me a romantic spot") is vectorized with the *same* fitted TF-IDF vectorizer + embedding model, then blended with the historical profile match — weighted so the explicit ask dominates over general taste history. |
| **Popularity Prior** | `stars × log(1 + review_count)` | 0.50 vs. similarity | Rewards established, well-rated businesses without needing a sentiment or topic model. |
| **LLM Concierge** | Natural-language explanation | N/A | GPT-4o-mini explanation layer — the ranked shortlist is passed as structured context to generate personalized justifications for every recommendation. |

---

## 🔬 Key Technical Innovations

### 1. TF-IDF as the Primary Similarity Signal
Rather than compressing reviews into a small set of pre-named topics, we keep the full literal term/bigram vocabulary (~20,000 features) as each business's and user's profile representation. On our own offline evaluation, this **beat both LDA-topic similarity and sentence-embedding similarity on every ranking metric** (Precision@K, Recall@K, NDCG@K, HitRate@K, MRR) — see [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the numbers. It's also the only one of the three that's directly interpretable: every profile ships with its own top-6-term label (e.g. *"happy hour, craft beer, patio"*), no separate labeling step required.

### 2. Live Free-Text Query Matching
Rather than a fixed set of hand-picked "mood" categories, a user's live prompt is vectorized at request time with the same fitted TF-IDF vectorizer and sentence-transformer used to build the profiles — no retraining, no closed taxonomy. TF-IDF catches literal keyword overlap; embeddings catch paraphrases the literal match misses (a prompt saying "romantic" matching reviews that say "candlelit" or "intimate"). The two are blended, weighted toward the live prompt over general taste history, since an explicit ask should outweigh what a user has liked in the past.

### 3. Popularity Prior
Business quality is scored as `stars × log(1 + review_count)` — rewarding established, consistently well-rated businesses without depending on a sentiment model or topic-model output.

### 4. Distributed Scale
Built on **Apache Spark on Databricks**, with review/business/user data and generated profiles stored in **Unity Catalog** (managed tables + Volumes) for schema reliability and shared access across pipeline stages.

---

## 👥 The Team
*   **Saloni Jain:** Lead for LDA Behavioral Modeling and Feature Engineering (original Philadelphia pipeline).
*   **Quinten:** Lead for Recommender System Design, Similarity Scoring, and LLM Explanation Pipeline.
*   **Dhairya:** Lead for Model Optimization, LLM Integration, data processing, LLM Explanation Pipeline to Streamlit Dashboard Development.
*   **Esther:** Lead for PersonaPath Flier and Presentation.
*   **Lear:** Lead for Data Pipeline orchestration (original Philadelphia pipeline), Streamlit Dashboard Interface Design and Presentation Deck & Strategy.

---

## 📊 Project Impact
**Original Philadelphia pilot** (LDA + JSD, since retired — see above):
*   **1,962** Restaurants profiled across 48 behavioral and intent features.
*   **20,017** Unique User Personas built from historical dining patterns.
*   **261,000** High-quality reviews analyzed for sentiment and topic distribution.

**New Orleans** (current, TF-IDF + embeddings): pipeline built; evaluation pending an actual end-to-end run — numbers to follow once the New Orleans dataset has been processed.

---

## 🔗 Resources & Navigation
*   🚀 **[Quick Start Guide](INSTRUCTIONS.md)** - How to run the pipeline.
*   📓 **Pipeline stages** - [`01_data_processing/`](./01_data_processing/) → [`02_profile_building/`](./02_profile_building/) → [`03_scoring/`](./03_scoring/) → [`04_evaluation/`](./04_evaluation/) → [`05_recommender_app/`](./05_recommender_app/) (Data Processing to Interface — see [`ARCHITECTURE.md`](./ARCHITECTURE.md) for stage-by-stage detail).
*   📁 **[Demo Walkthrough](./demo/)** - See the engine and walkthrough video in action.
*   📄 **[Project Flyer](./flier/Team5_PersonaPath_Flier.pdf)** - Executive summary PDF.
*   📚 **[Bibliography](./BIBLIOGRAPHY.md)** - Data and library credits.

---

> This project repository is created in partial fulfillment of the requirements for the Big Data Analytics course offered by the Master of Science in Business Analytics program at the Carlson School of Management, University of Minnesota.
