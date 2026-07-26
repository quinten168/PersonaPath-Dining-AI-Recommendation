"""
Fit a single global TF-IDF vectorizer over every New Orleans review and
transform the corpus into a sparse per-review TF-IDF matrix.

This is the TF-IDF pipeline's counterpart to
03_profile_building/embedding/embeddings.py -- same input corpus, same role
in the pipeline (per-review vector generation), but a standalone
implementation: it does not import from the embedding pipeline, and it
writes its own review-level metadata CSV rather than reusing
review_embeddings_meta.csv, so the two pipelines share no code or generated
artifacts end to end. Deliberate, for genuine independence -- see
03_profile_building/tfidf/aggregation.py's docstring for the same note on
compute_review_weights().

Why TF-IDF at all, alongside embeddings: sentence embeddings capture
semantic/paraphrase similarity well but are not interpretable (384
meaningless dimensions). TF-IDF profiles are interpretable by construction
-- every dimension is a literal word or bigram -- at the cost of losing
semantic matching (synonyms/paraphrases score no similarity). This pipeline
exists to evaluate that trade-off, not to replace the embedding pipeline.

Input:  persona-path.default.new_orleans_reviews   (Unity Catalog managed table,
                                                     from 01_data_processing/01_data_processing.ipynb)
Output (Unity Catalog Volume, so these survive past this job's cluster):
    /Volumes/persona-path/default/interim/review_tfidf.npz          (scipy sparse CSR, n_reviews x n_features)
    /Volumes/persona-path/default/interim/review_tfidf_meta.csv     (review_id, business_id, user_id,
                                                                       stars, date -- same row order as the .npz)
    /Volumes/persona-path/default/interim/tfidf_vectorizer.joblib   (fitted TfidfVectorizer, for
                                                                       feature names + future reuse)

Requires a live Spark session (run as a Databricks notebook/job) -- reads
via spark.table(), not a local file, so this can no longer run as a plain
`python3 vectorize.py`.

Vectorizer settings (max_features=20000, ngram_range=(1,2), English stopwords)
match the embedding pipeline's label-generation TF-IDF
(03_profile_building/embedding/profile_labels.fit_tfidf) so the two are
comparable on vocabulary size/shape -- not because of a code dependency
between them.
"""
import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

TABLE_NAME = "persona-path.default.new_orleans_reviews"
VOLUME_ROOT = "/Volumes/persona-path/default"

MAX_TFIDF_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)

OUT_REVIEW_TFIDF_NPZ = f"{VOLUME_ROOT}/interim/review_tfidf.npz"
OUT_REVIEW_TFIDF_META = f"{VOLUME_ROOT}/interim/review_tfidf_meta.csv"
OUT_VECTORIZER = f"{VOLUME_ROOT}/interim/tfidf_vectorizer.joblib"


def fit_tfidf(texts, max_features=MAX_TFIDF_FEATURES, ngram_range=TFIDF_NGRAM_RANGE):
    """One TfidfVectorizer over the full review corpus (one global
    vocabulary/IDF) so business and user profiles are on the same scale.
    Standalone copy of the embedding pipeline's identically-named function
    (profile_labels.fit_tfidf) -- see this file's module docstring."""
    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=ngram_range,
        stop_words="english", lowercase=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def main():
    print(f"Loading review corpus: {TABLE_NAME}")
    df = spark.table(TABLE_NAME).select(
        "review_id", "business_id", "user_id", "stars", "date", "text",
    ).toPandas()
    df["text"] = df["text"].fillna("")
    print(f"  {len(df):,} reviews")

    print(f"Fitting TF-IDF (max_features={MAX_TFIDF_FEATURES}, "
          f"ngram_range={TFIDF_NGRAM_RANGE}) ...")
    vectorizer, tfidf_matrix = fit_tfidf(df["text"])
    tfidf_matrix = tfidf_matrix.tocsr()
    print(f"  matrix shape: {tfidf_matrix.shape}, "
          f"nnz={tfidf_matrix.nnz:,} ({tfidf_matrix.nnz / tfidf_matrix.shape[0]:.1f} terms/review avg)")

    sp.save_npz(OUT_REVIEW_TFIDF_NPZ, tfidf_matrix)
    print(f"Saved: {OUT_REVIEW_TFIDF_NPZ}")

    df[["review_id", "business_id", "user_id", "stars", "date"]].to_csv(
        OUT_REVIEW_TFIDF_META, index=False
    )
    print(f"Saved: {OUT_REVIEW_TFIDF_META}")

    joblib.dump(vectorizer, OUT_VECTORIZER)
    print(f"Saved: {OUT_VECTORIZER}")

    print("Run build_profiles.py next to aggregate these into "
          "per-business/per-user profile vectors.")


if __name__ == "__main__":
    main()
