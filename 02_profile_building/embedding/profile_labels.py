"""
TF-IDF-based interpretability labels for business/user profiles: a short
human-readable string per profile (e.g. "happy hour, craft beer, patio"),
surfaced as a new `label` column alongside the embedding profiles.

Uses ALL of a group's reviews, weighted by the same per-review weights as
the embedding centroid (aggregation.compute_review_weights) -- not a
top-K-nearest-centroid sample. Centroid-nearest reviews are, by
construction, the most generic ones (closest to the mean); a fixed top-K
also badly under-covers high-volume businesses (up to ~5,700 reviews).
Full-corpus aggregation guarantees the label always matches what the
paired embedding vector actually represents.

Kept separate from aggregation.py (operates on sparse TF-IDF rows/text,
not dense embedding vectors) and from embeddings.py (first code to
re-read review TEXT, which embeddings.py discards after encoding).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from aggregation import compute_review_weights


def fit_tfidf(texts, max_features=20000, ngram_range=(1, 2)):
    """One TfidfVectorizer over the full review corpus (one global
    vocabulary/IDF) so business and user labels are on the same scale."""
    vectorizer = TfidfVectorizer(
                    max_features=max_features, 
                    ngram_range=ngram_range,
                    stop_words="english", 
                    lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def build_group_labels(tfidf_matrix, meta_df: pd.DataFrame, group_col: str,
                        feature_names, strategy: str = "mean",
                        decay_constant: float = 30.0, reference_date=None,
                        top_n: int = 6) -> pd.DataFrame:
    """
    Aggregate per-review TF-IDF rows into one label per group, using the
    exact same per-review weights as the embedding centroid for that group
    (via aggregation.compute_review_weights).

    Weighted-sum TF-IDF rows per group via one sparse matmul (a
    (n_groups x n_reviews) weighted indicator matrix times tfidf_matrix),
    then take each group's top_n terms by weight. No division by the
    weight sum is needed -- top-N ranking within a row is invariant to a
    positive scalar, so only the weighted sum is computed, not the mean.

    Returns a DataFrame: [group_col, label].
    """
    weights = compute_review_weights(meta_df, strategy, decay_constant, reference_date)

    codes, uniques = pd.factorize(meta_df[group_col], sort=True)
    n_groups, n_reviews = len(uniques), len(meta_df)
    indicator = sp.coo_matrix(
        (weights, (codes, np.arange(n_reviews))), shape=(n_groups, n_reviews),
    ).tocsr()
    weighted_sums = (indicator @ tfidf_matrix).tocsr()

    feature_names = np.asarray(feature_names)
    labels = []
    for row_start, row_end in zip(weighted_sums.indptr[:-1], weighted_sums.indptr[1:]):
        row_idx = weighted_sums.indices[row_start:row_end]
        row_val = weighted_sums.data[row_start:row_end]
        if len(row_val) == 0:
            labels.append("")
            continue
        k = min(top_n, len(row_val))
        top_local = np.argpartition(-row_val, k - 1)[:k]
        top_local = top_local[np.argsort(-row_val[top_local])]
        labels.append(", ".join(feature_names[row_idx[top_local]]))

    return pd.DataFrame({group_col: uniques, "label": labels})
