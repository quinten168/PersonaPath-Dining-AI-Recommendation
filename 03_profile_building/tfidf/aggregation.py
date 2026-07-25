"""
Aggregation strategy for turning per-review TF-IDF vectors (vectorize.py)
into one profile vector per business/user.

Standalone counterpart to 03_profile_building/embedding/aggregation.py --
same "mean"/"recency" weighting scheme, deliberately duplicated rather than
imported so this pipeline shares no code with the embedding pipeline (see
vectorize.py's module docstring for why). ~15 lines, pure function of
meta_df alone, unlikely to drift; if you change the weighting scheme here,
change it in the embedding-pipeline copy too if you want the two pipelines
to stay comparable under "recency".

No cold-start category-prior blending here (unlike the embedding pipeline's
blend_with_category_prior) -- explicitly out of scope for this pass. It
currently blends 0 rows in the embedding pipeline on this corpus (every
business/user already clears the min_reviews floor upstream), so skipping
it here is a documented simplification, not a regression relative to what
the embedding pipeline actually does today.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def compute_review_weights(meta_df: pd.DataFrame, strategy: str,
                            decay_constant: float = 30.0,
                            reference_date=None) -> np.ndarray:
    """
    Per-review weight, aligned to meta_df's row order -- how much does a
    review count toward its group's profile.

    "mean" -> all ones. "recency" -> 1 / (1 + days_old / decay_constant),
    days_old measured against reference_date (defaults to the latest date
    in meta_df).
    """
    if strategy not in ("mean", "recency"):
        raise ValueError(f"unknown aggregation strategy: {strategy!r}")
    if strategy != "recency":
        return np.ones(len(meta_df), dtype=np.float64)
    dates = pd.to_datetime(meta_df["date"])
    if reference_date is None:
        reference_date = dates.max()
    days_old = (reference_date - dates).dt.total_seconds().values / 86400.0
    return 1.0 / (1.0 + days_old / decay_constant)


def build_group_tfidf_profiles(tfidf_matrix, meta_df: pd.DataFrame, group_col: str,
                                strategy: str = "mean", min_reviews: int = 5,
                                decay_constant: float = 30.0, reference_date=None):
    """
    Aggregate per-review TF-IDF rows into one sparse profile vector per
    group (business_id or user_id), via a weighted-sum sparse matmul, then
    L2-row-normalize so cosine similarity reduces to a dot product at query
    time (mirrors the embedding pipeline's normalize_embeddings=True
    convention -- see similarity_tfidf.py).

    tfidf_matrix: (n_reviews, n_features) sparse CSR, row-aligned with
        meta_df (i.e. vectorize.py's output).
    meta_df: DataFrame with at least [group_col, "date"], same row order as
        tfidf_matrix.

    Returns (profile_matrix, profile_meta):
        profile_matrix: (n_groups, n_features) sparse CSR, L2-row-normalized.
        profile_meta: DataFrame [group_col, n_reviews, low_confidence], row
            order matches profile_matrix (NOT necessarily meta_df's or any
            external DataFrame's row order -- pd.factorize(sort=True) below
            orders groups lexicographically by ID. Always resolve rows via
            an explicit {group_id: row} dict, never assume positional
            alignment with another table).
    """
    weights = compute_review_weights(meta_df, strategy, decay_constant, reference_date)

    codes, uniques = pd.factorize(meta_df[group_col], sort=True)
    n_groups, n_reviews = len(uniques), len(meta_df)
    indicator = sp.coo_matrix(
        (weights, (codes, np.arange(n_reviews))), shape=(n_groups, n_reviews),
    ).tocsr()
    weighted_sums = (indicator @ tfidf_matrix).tocsr()
    profile_matrix = normalize(weighted_sums, norm="l2", axis=1)

    counts = np.bincount(codes, minlength=n_groups)
    profile_meta = pd.DataFrame({
        group_col: uniques,
        "n_reviews": counts,
    })
    profile_meta["low_confidence"] = profile_meta["n_reviews"] < min_reviews
    return profile_matrix, profile_meta
