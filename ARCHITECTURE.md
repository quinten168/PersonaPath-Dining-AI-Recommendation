# PersonaPath Embedding Pipeline — Architecture

> Additive to the existing LDA + JSD recommender (`01_data_processing/`–`02_topic_modeling_lda/`). Nothing in this
> pipeline replaces that path — it adds a second, vector-based similarity signal that
> gets blended with the topic-model score at recommendation time.

Every review becomes a point in a 384-dimensional space; every business and user becomes
a centroid in that same space. This is the path from raw Yelp text to a cosine-similarity
score sitting next to the existing topic-model score.

- **Stages:** 8 (+2 sub-stages)
- **Root:** `01_data_processing/` … `06_recommender_app/` (stage-based layout, see repo root)
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers, local, no API cost)

Each stage's *output* is the next stage's *input* — Stage 00 through 07 run top to
bottom, once, offline. Nothing runs at request time except **Stage 05**.

---

## Stage 00 — Raw Data

*No processing.*

Source exports, dropped in as-is. The Philadelphia business/user universe was already
established by the topic-model pipeline (notebooks 01–02); this pipeline only adds a
second signal on top of it.

| Input | Description |
|---|---|
| `yelp_academic_dataset_review.json` | ~6.9M reviews, full Yelp corpus |
| `business_profiles.csv` | 1,962 Philadelphia businesses + LDA topics |
| `user_profiles.csv` | 20,017 Philadelphia users |

---

## Stage 01 — Review Filtering

**Script:** `01_data_processing/prep_reviews.py`

Streams the full Yelp corpus down to just the reviews touching the existing
Philadelphia business/user universe, keeping raw review text — everything downstream
reads text, not IDs.

| | |
|---|---|
| Input | `review.json`, `business/user_profiles.csv` |
| Script | `prep_reviews.py` |
| Output | `philly_reviews.parquet` — 274,563 reviews · `review_id, user_id, business_id, stars, date, text` |

---

## Stage 02 — Review Embedding

**Script:** `03_profile_building/embedding/embeddings.py`

Encodes every review independently with `sentence-transformers`
(`all-MiniLM-L6-v2`), `normalize_embeddings=True` so cosine similarity later reduces to
a plain dot product. Offline batch job (`BATCH_SIZE=256`) — never run at request time,
matching the batch-only scope of this pipeline.

| | |
|---|---|
| Input | `philly_reviews.parquet` |
| Script | `embeddings.py` — all-MiniLM-L6-v2, local, no API cost |
| Output | `review_embeddings.npy` — 274,563 × 384, float32 memmap<br>`review_embeddings_meta.csv` — `review_id, business_id, user_id, stars, date` |

---

## Stage 03 — Profile Aggregation

**Script:** `03_profile_building/embedding/aggregation.py`

Pools per-review vectors into one centroid per business and per user, then folds in a
fallback for anything under-reviewed. A single shared weight function drives both this
stage and Stage 04, so a strategy swap (`"mean"` ↔ `"recency"`) never lets the label and
the vector describe different things.

### 3a — Centroid (`build_group_profiles()`)

| | |
|---|---|
| Input | `review_embeddings.npy`, `review_embeddings_meta.csv` |
| Script | `compute_review_weights()` — mean = uniform · recency = `1/(1+days/30)` |
| Output | `emb_0…emb_383` + `n_reviews, low_confidence` |

### 3b — Cold-Start Blend (`blend_with_category_prior()`)

| | |
|---|---|
| Input | `low_confidence` rows (`n_reviews < 5` biz / `< 3` user), `categories` (raw Yelp tags, business only) |
| Script | `blend_with_category_prior()` — swappable `weight_fn` + `category_key_fn` |
| Output | blended `emb_` cols — low-confidence rows only — currently 0 in this corpus |

---

## Stage 04 — Interpretability Labels

**Script:** `03_profile_building/embedding/profile_labels.py`

None of the 384 embedding dimensions carry human meaning, unlike LDA's named topic
columns. This stage fits one TF-IDF vectorizer over the full corpus, sums each group's
*weighted* TF-IDF rows in a single sparse matmul (same weights as 3a, every review, not
a nearest-to-centroid sample) and keeps the top terms as a plain-language label.

| | |
|---|---|
| Input | `philly_reviews.parquet` (`text`), `review_embeddings_meta.csv` |
| Script | `fit_tfidf()` — 20k features, uni+bigrams<br>`build_group_labels()` — weighted sparse matmul, top-6 terms |
| Output | `label` — e.g. *"market, terminal, vendors, reading terminal, food, produce"* |

---

## Stage 05 — Similarity Scoring *(runs live)*

**Script:** `04_scoring/similarity_embedding.py`

The one stage that runs live, per request: a user vector against the full business
matrix, one dot product — valid only because every vector was pre-normalized back in
Stage 02.

| | |
|---|---|
| Input | `business_embeddings.csv` (matrix), `user_embeddings.csv` (one row) |
| Script | `cosine_similarity_via_dot()` |
| Output | score per business — computed live, not persisted |

---

## Stage 06 — Evaluation

**Script:** `05_evaluation/evaluate_recommender.py`

An 80/20 leave-out harness — precision, recall, NDCG, hit-rate, MRR — reused as-is from
the topic-model baseline, so the embedding signal is judged on the exact same eval set,
not a new one.

| | |
|---|---|
| Input | `business/user_embeddings.csv` |
| Script | `rank_embedding()` via `similarity.py` |
| Output | `eval_per_user.csv`<br>NDCG@5 0.0322 · @10 0.0389 — mean strategy, beats LDA+JSD |

> `rank_cbf()` — the pre-existing LDA + JSD baseline — runs in the same harness for
> comparison, unchanged. Its topic model was trained once on the full corpus in
> notebook 02 and isn't part of this pipeline.

---

## Stage 07 — Recommender *(terminal stage)*

**Script:** `06_recommender_app/app.py`

Blends the embedding similarity score with the LDA/JSD score (`MinMaxScaler`),
MMR-reranks the blend for diversity, and hands the final list to an LLM for a
plain-language explanation — served in the Streamlit interface.

| | |
|---|---|
| Input | `business/user_embeddings.csv`, `business_profiles.csv` (LDA topics) |
| Script | `recommend(scoring_method=…)` |
| Output | ranked list + explanation — Streamlit |

---

## Legend

- **Input** — file read
- **Script** — transformation run
- **Output** — file written
- **Pre-existing path** — context only (not built in this pass)

## Source files

`03_profile_building/embedding/embeddings.py` · `03_profile_building/embedding/aggregation.py` · `03_profile_building/embedding/profile_labels.py` ·
`04_scoring/similarity_embedding.py` · `05_evaluation/evaluate_recommender.py` — additive to the pre-existing
LDA + JSD pipeline (notebooks 01–02), which none of this replaces.
