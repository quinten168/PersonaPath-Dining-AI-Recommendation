# PersonaPath Recommendation Pipeline — Architecture

> TF-IDF review-term profiles are the primary similarity signal end to end. Sentence
> embeddings are an optional secondary signal, blended in for historical
> profile-to-profile matching and used for semantic live-query matching. The original
> LDA + JSD pipeline (`02_topic_modeling_lda/`) has been retired: on our own offline
> evaluation (Stage 05), TF-IDF outperformed both LDA-topic similarity and embedding
> similarity on every ranking metric — see the numbers below.

Every review becomes both a TF-IDF sparse vector (~20,000 literal term/bigram features)
and a 384-dimensional dense embedding; every business and user becomes a profile in
both spaces. A live user prompt is vectorized the same way at request time and blended
with the historical profile match, weighted toward the live prompt.

- **Root:** `01_data_processing/` … `06_recommender_app/` (stage-based layout, see repo root)
- **Primary model:** TF-IDF (`scikit-learn TfidfVectorizer`, 20k features, uni+bigrams, English stopwords)
- **Secondary model:** `all-MiniLM-L6-v2` (sentence-transformers, local, no API cost)
- **Storage:** Unity Catalog managed table (raw reviews) + Volumes (generated artifacts) — Databricks-hosted, not local files

Each stage's *output* is the next stage's *input* — stages run top to bottom, once,
offline, except live query matching and final scoring, which run per request.

---

## Stage 00 — Ingestion & Filtering *(Databricks notebook, not a local script)*

**Notebook:** `01_data_processing/01_data_processing.ipynb`

Loads the raw Yelp business/review/user JSON via Spark, filters to a target city
(currently New Orleans, `review_count >= 50` per business), joins in business and user
metadata, filters to reviews with `text` length > 50 characters and users with
`review_count >= 3`, and writes the result to a Unity Catalog **managed table** —
`persona-path.default.new_orleans_reviews`. This replaces the previous local
`prep_reviews.py` + parquet-file approach; every downstream stage now reads this table
via a live Spark session rather than a local file.

| | |
|---|---|
| Input | Yelp `business`/`review`/`user` JSON (Unity Catalog Volume) |
| Notebook | `01_data_processing.ipynb` |
| Output | Managed table `persona-path.default.new_orleans_reviews` — `review_id, business_id, user_id, stars, date, text` + business/user metadata |

---

## Stage 01 — TF-IDF Vectorization & Profile Building *(primary similarity signal)*

**Scripts:** `03_profile_building/tfidf/vectorize.py`, `build_profiles.py`, `aggregation.py`

Fits one global `TfidfVectorizer` (20,000 features, uni+bigrams, English stopwords)
over the full review corpus, then aggregates per-review TF-IDF rows into one
L2-normalized profile vector per business and per user (mean- or recency-weighted sum),
so cosine similarity at query time reduces to a dot product. Each profile also gets a
top-6-term label for free (e.g. *"happy hour, craft beer, patio"*) — interpretable by
construction, no separate labeling step needed.

| | |
|---|---|
| Input | `persona-path.default.new_orleans_reviews` (managed table) |
| Script | `vectorize.py` — fit + transform → `review_tfidf.npz`/`_meta.csv`/`tfidf_vectorizer.joblib`<br>`build_profiles.py` — `aggregation.build_group_tfidf_profiles()` → per-group profile matrix + label |
| Output (Volume) | `business_tfidf_profiles.npz`/`_meta.csv`, `user_tfidf_profiles.npz`/`_meta.csv` |

---

## Stage 02 — Review Embedding *(secondary signal, optional)*

**Script:** `03_profile_building/embedding/embeddings.py`

Encodes every review independently with `sentence-transformers` (`all-MiniLM-L6-v2`,
`normalize_embeddings=True`). Additive: the app degrades gracefully to TF-IDF-only for
both historical similarity and live query matching if these artifacts don't exist.

| | |
|---|---|
| Input | `persona-path.default.new_orleans_reviews` (managed table) |
| Script | `embeddings.py` — all-MiniLM-L6-v2, local, no API cost |
| Output (Volume) | `review_embeddings.npy` (float32), `review_embeddings_meta.csv` |

---

## Stage 03 — Embedding Profile Aggregation

**Scripts:** `03_profile_building/embedding/aggregation.py`, `build_profiles.py`, `profile_labels.py`

Pools per-review embeddings into one centroid per business/user (same mean/recency
weighting convention as Stage 01's TF-IDF aggregation), plus a category-prior cold-start
blend for under-reviewed groups (inert on the current review-count floors) and a
TF-IDF-derived top-terms label for interpretability, since raw embedding dimensions
carry no human meaning.

| | |
|---|---|
| Input | `review_embeddings.npy`/`_meta.csv`, review text (managed table) |
| Script | `build_group_profiles()`, `blend_with_category_prior()`, `profile_labels.build_group_labels()` |
| Output (Volume) | `business_embeddings.csv`, `user_embeddings.csv` — `emb_0…emb_383, n_reviews, low_confidence, label` |

---

## Stage 04 — Live Query Matching & Similarity Scoring *(runs live, per request)*

**Scripts:** `04_scoring/similarity_tfidf.py`, `similarity_embedding.py`, `query_matching.py`

The part that runs at request time. A user's historical TF-IDF profile (optionally
blended with their embedding profile) is compared against every business via cosine
similarity — the pre-computed part. A live free-text prompt is vectorized on the spot
with the *same* fitted TF-IDF vectorizer and embedding model (never re-fit), matched
against business profiles the same way, then blended with the historical match —
weighted toward the live prompt (`prompt_weight=0.75` by default), since an explicit ask
should outweigh general taste history.

| | |
|---|---|
| Input | `business/user_tfidf_profiles.npz`, `business/user_embeddings.csv`, live query text |
| Script | `cosine_similarity_via_dot()` (tfidf + embedding) · `match_prompt_to_businesses()` · `blend_prompt_and_profile()` |
| Output | score per business — computed live, not persisted |

---

## Stage 05 — Evaluation

**Script:** `05_evaluation/evaluate_recommender.py`

An 80/20 leave-out harness — Precision@K, Recall@K, NDCG@K, HitRate@K, MRR — comparing
`tfidf`, `embedding`, `cbf` (the retired LDA+JSD baseline), `popularity`, and `random`
arms on the same held-out split.

| System | K | Precision | Recall | NDCG | HitRate | MRR |
|---|---|---|---|---|---|---|
| **tfidf** | 5 | **0.0263** | **0.0318** | **0.0361** | **0.1177** | **0.0648** |
| **tfidf** | 10 | **0.0226** | **0.0573** | **0.0438** | **0.1830** | **0.0737** |
| embedding | 5 | 0.0245 | 0.0308 | 0.0307 | 0.1089 | 0.0509 |
| embedding | 10 | 0.0207 | 0.0529 | 0.0377 | 0.1695 | 0.0590 |
| popularity | 5 | 0.0228 | 0.0289 | 0.0315 | 0.1037 | 0.0564 |
| popularity | 10 | 0.0177 | 0.0417 | 0.0349 | 0.1493 | 0.0623 |
| cbf (LDA+JSD, retired) | 5 | 0.0207 | 0.0262 | 0.0245 | 0.0975 | 0.0406 |
| cbf (LDA+JSD, retired) | 10 | 0.0206 | 0.0515 | 0.0343 | 0.1742 | 0.0508 |
| random | 5/10 | ~0.002 | ~0.005 | ~0.004 | ~0.02 | ~0.007 |

*(Philadelphia dataset, from `eval_per_user.csv`. `tfidf` beats every other arm on every
metric at both K=5 and K=10 — including `cbf`, despite `cbf` having a structural
leakage advantage: its LDA model was trained once on the full corpus and never saved to
disk, so it couldn't be retrained per evaluation fold the way `tfidf`/`embedding` were.
This result is why TF-IDF is the primary method going forward, and why LDA was retired
rather than carried over to New Orleans. These numbers have not yet been re-validated
on the New Orleans dataset — re-running this stage once New Orleans profiles exist is
an open task, not an assumption that the Philadelphia result automatically generalizes.)*

| | |
|---|---|
| Input | `business/user_tfidf_profiles.npz`, `business/user_embeddings.csv` |
| Script | `evaluate_recommender.py` |
| Output | `eval_per_user.csv` |

---

## Stage 06 — Recommender *(terminal stage)*

**Script:** `06_recommender_app/app.py`

Blends TF-IDF profile similarity (+ optional embedding blend) with live prompt matching
(Stage 04), applies a `stars × log(1+review_count)` popularity prior, MMR-reranks for
diversity, and hands the final list to an LLM (GPT-4o-mini) for a plain-language
explanation — served in the Streamlit interface. Runs as a Databricks App (needs a live
Spark session to read the managed table for business display metadata — name, address,
categories, stars, review_count; business hours have no New Orleans data source yet).

| | |
|---|---|
| Input | `business/user_tfidf_profiles.npz`, `business/user_embeddings.csv`, managed table (business metadata), live query text |
| Script | `recommend()` |
| Output | ranked list + explanation — Streamlit |

---

## Legend

- **Input** — file/table read
- **Script** — transformation run
- **Output** — file/table written

## Source files

`01_data_processing/01_data_processing.ipynb` ·
`03_profile_building/tfidf/vectorize.py` · `build_profiles.py` · `aggregation.py` ·
`03_profile_building/embedding/embeddings.py` · `aggregation.py` · `profile_labels.py` · `build_profiles.py` ·
`04_scoring/similarity_tfidf.py` · `similarity_embedding.py` · `query_matching.py` ·
`05_evaluation/evaluate_recommender.py` ·
`06_recommender_app/app.py`
