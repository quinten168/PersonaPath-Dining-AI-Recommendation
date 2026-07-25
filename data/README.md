# Data

This directory holds the raw source data, intermediate pipeline artifacts, and final
profile outputs for PersonaPath. **The full data is not stored in GitHub per project
guidelines** — everything in `raw/`, `interim/`, and most of `profiles/` is
gitignored. Refer to the [Yelp Open Dataset download page](https://www.yelp.com/dataset)
to obtain the raw data. ETL process outputs are typically managed via Databricks
FileStore / Delta Lake in the full pipeline.

## Layout

| Folder | Contents |
| :--- | :--- |
| `raw/` | Unmodified Yelp Open Dataset exports (`yelp_academic_dataset_*.json`), the dataset's user agreement. |
| `interim/` | Generated intermediates: filtered Philadelphia review corpus (`philly_reviews.parquet`), per-review embedding/TF-IDF matrices and their metadata. Reproducible from `raw/` via the `01_data_processing/` and `03_profile_building/` scripts. |
| `profiles/` | Final per-business and per-user profile outputs consumed by scoring and evaluation — `business_profiles.csv`/`user_profiles.csv` (LDA topic vectors), `business_embeddings.csv`/`user_embeddings.csv` (sentence-embedding centroids), and the TF-IDF equivalents. |
| `schema/` | Human-readable schema docs for the profile tables (below). |

## Schema — `business_profiles`
*   **Total Records:** 1,962 restaurants
*   **Total Columns:** 48

| Column Group | Description |
| :--- | :--- |
| **Metadata** | `business_id`, `name`, `address`, `city`, `is_open`, `stars`, `review_count`, plus hours for all 7 days. |
| **Topic Vectors (25 cols)** | Normalized probability scores for each of the 25 behavioral topics (e.g., `topic_01`, `topic_02`...). |
| **Intent Scores (5 cols)** | Proprietary scores for: `romantic`, `solo_work`, `family`, `group`, `hidden_gem`. |
| **Quality Metrics** | `vader_sentiment` (aggregated average) and the final `EAS_score`. |

## Schema — `user_personas`
*   **Total Records:** 20,017 users

| Column | Description |
| :--- | :--- |
| **`user_id`** | Unique identifier for the Yelp user. |
| **Topic Vectors (25 cols)** | The user's historical behavioral DNA based on their review history. |
| **`top_3_topics`** | The most dominant behavioral preferences for the user. |
| **`confidence_score`** | Statistical confidence in the persona assignment (based on review count and topic consistency). |

See `schema/` for the full per-table schema docs (`schema_business_profiles.md`,
`schema_lda_topic_results.md`, `schema_user_personas.md`).

---

### Delta Lake Configuration (full pipeline)
*   **Location:** `msbabigdata.default`
*   **Write Mode:** `overwriteSchema=True` (Ensures clean updates even if columns are added or modified during feature engineering).
