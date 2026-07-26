"""
Aggregation strategies for turning per-review embeddings (embeddings.py)
into one profile vector per business or per user.

Kept deliberately separate from embedding generation (embeddings.py) and
similarity scoring (similarity.py) so a new aggregation strategy can be
swapped in -- or the cold-start blend ratio / category grouping changed --
without touching either of those. No config constants live in this file;
callers (build_profiles.py) pass everything in as parameters.
"""
import numpy as np
import pandas as pd


def _renormalize(vecs: np.ndarray) -> np.ndarray:
    """Rescale each row to unit length. Needed after any pooling/blending --
    normalize_embeddings=True at encode time only guarantees unit length for
    the ORIGINAL per-review vectors, not a mean or weighted blend of them."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def mean_centroid(vectors: np.ndarray, weights=None, **_ignored) -> np.ndarray:
    """
    Unweighted mean-pool + renormalize. Accepts (and ignores) `weights` so it
    can be dispatched uniformly alongside recency_weighted_centroid.

    KNOWN SIMPLIFICATION: averaging can wash out mixed signals -- e.g. a
    restaurant with both "romantic date-night" and "loud Friday-night
    sports-bar" reviews lands on a bland midpoint resembling neither well.
    Revisit with a medoid or multi-vector/cluster approach if this turns out
    to matter.
    """
    centroid = vectors.mean(axis=0, keepdims=True)
    return _renormalize(centroid)[0]


def recency_weighted_centroid(vectors: np.ndarray, weights: np.ndarray, **_ignored) -> np.ndarray:
    """
    Recency-weighted mean-pool + renormalize. `weights` is precomputed by
    compute_review_weights() -- 1 / (1 + days_old / decay_constant), a
    reciprocal/hyperbolic decay (not a literal exponential one, despite the
    "exponential decay" description this was requested under; the explicit
    formula takes precedence).

    Same mixed-signal caveat as mean_centroid, just weighted toward recent
    reviews rather than all reviews equally.
    """
    wsum = weights.sum()
    if wsum == 0:
        return mean_centroid(vectors)
    centroid = (vectors * weights[:, None]).sum(axis=0, keepdims=True) / wsum
    return _renormalize(centroid)[0]


AGGREGATORS = {
    "mean": mean_centroid,
    "recency": recency_weighted_centroid,
}


def compute_review_weights(meta_df: pd.DataFrame, strategy: str,
                            decay_constant: float = 30.0,
                            reference_date=None) -> np.ndarray:
    """
    Per-review weight, aligned to meta_df's row order -- single source of
    truth for "how much does a review count" toward its group's profile.
    Shared by the embedding centroid (build_group_profiles, below) and the
    TF-IDF label (profile_labels.build_group_labels), so a label always
    describes exactly what its paired embedding vector represents.

    "mean" -> all ones. "recency" -> 1 / (1 + days_old / decay_constant),
    days_old measured against reference_date (defaults to the latest date
    in meta_df).
    """
    if strategy not in AGGREGATORS:
        raise ValueError(f"unknown aggregation strategy: {strategy!r}")
    if strategy != "recency":
        return np.ones(len(meta_df), dtype=np.float64)
    dates = pd.to_datetime(meta_df["date"])
    if reference_date is None:
        reference_date = dates.max()
    days_old = (reference_date - dates).dt.total_seconds().values / 86400.0
    return 1.0 / (1.0 + days_old / decay_constant)


def build_group_profiles(review_embeddings, meta_df: pd.DataFrame, group_col: str,
                          strategy: str = "mean", min_reviews: int = 5,
                          decay_constant: float = 30.0, reference_date=None) -> pd.DataFrame:
    """
    Aggregate per-review vectors into one vector per group (business_id or
    user_id).

    review_embeddings: (n_reviews, dim) array-like, row-aligned with meta_df
        (e.g. a memmap opened via np.load(path, mmap_mode="r") -- fancy
        indexing on it here only reads the rows actually needed per group,
        so the full array is never pulled into RAM at once).
    meta_df: DataFrame with at least [group_col, "date"], same row order as
        review_embeddings.

    Returns a DataFrame: [group_col, emb_0..emb_{dim-1}, n_reviews, low_confidence].
    """
    if strategy not in AGGREGATORS:
        raise ValueError(f"unknown aggregation strategy: {strategy!r}")
    agg_fn = AGGREGATORS[strategy]
    weights_all = compute_review_weights(meta_df, strategy, decay_constant, reference_date)

    dim = review_embeddings.shape[1]
    group_ids, counts, centroids = [], [], []
    for gid, idxs in meta_df.groupby(group_col).indices.items():
        idxs = np.asarray(idxs)
        vecs = np.asarray(review_embeddings[idxs])
        centroid = agg_fn(vecs, weights=weights_all[idxs])
        group_ids.append(gid)
        counts.append(len(idxs))
        centroids.append(centroid)

    emb_df = pd.DataFrame(np.asarray(centroids, dtype=np.float32),
                           columns=[f"emb_{i}" for i in range(dim)])
    out = pd.DataFrame({group_col: group_ids, "n_reviews": counts})
    out = pd.concat([out, emb_df], axis=1)
    out["low_confidence"] = out["n_reviews"] < min_reviews
    return out


def default_coldstart_blend_weight(n_reviews: int, min_reviews: int) -> float:
    """
    Default blend-ratio function for blend_with_category_prior(): weight put
    on the entity's OWN centroid, growing with review count.

        weight_on_own = n_reviews / (n_reviews + min_reviews)

    At n_reviews=0 -> 0 (all prior). At n_reviews=min_reviews (the
    low-confidence boundary) -> 0.5. Swappable: pass a different callable to
    blend_with_category_prior(weight_fn=...) for a different curve (e.g. a
    hard cutoff, or a steeper/gentler ramp).
    """
    return n_reviews / (n_reviews + min_reviews)


def first_category_tag(categories_str) -> str:
    """
    KNOWN SIMPLIFICATION: takes the first comma-separated Yelp category tag
    as a business's "primary" category for cold-start grouping (e.g.
    "Bars, Nightlife, Restaurants" -> "Bars"). Yelp's category field order is
    not guaranteed to reflect specificity or primacy, so this is an
    approximation -- swap in a smarter parse (e.g. least-frequent/most
    specific tag) via blend_with_category_prior(category_key_fn=...) if this
    turns out to produce noisy category priors.
    """
    if not isinstance(categories_str, str) or not categories_str.strip():
        return "Unknown"
    return categories_str.split(",")[0].strip()


def blend_with_category_prior(profile_df: pd.DataFrame, category_by_id: dict,
                               min_reviews: int,
                               weight_fn=default_coldstart_blend_weight,
                               category_key_fn=first_category_tag) -> pd.DataFrame:
    """
    Cold-start fallback for low-confidence business profiles: blend a
    low-confidence centroid with the mean centroid of other businesses
    sharing its category tag, weighted by review count.

    profile_df: output of build_group_profiles() for businesses -- must have
        the group id column (first column), "n_reviews", "low_confidence",
        and emb_ columns.
    category_by_id: {business_id: raw categories string}, e.g. sourced from
        data/profiles/business_profiles.csv's "categories" column.

    High-confidence rows pass through unchanged. For a low-confidence row
    with no OTHER business sharing its category tag, falls back further to
    the global mean centroid across all businesses.
    """
    id_col = profile_df.columns[0]
    emb_cols = [c for c in profile_df.columns if c.startswith("emb_")]
    df = profile_df.copy()
    df["_category"] = df[id_col].map(category_by_id).map(category_key_fn)

    global_prior = _renormalize(df[emb_cols].values.mean(axis=0, keepdims=True))[0]
    category_sums = df.groupby("_category")[emb_cols].sum()
    category_counts = df.groupby("_category").size()

    new_vecs = df[emb_cols].values.copy()
    for i, row in df.reset_index(drop=True).iterrows():
        if not row["low_confidence"]:
            continue
        cat = row["_category"]
        own_vec = row[emb_cols].values.astype(np.float64)
        others_sum = category_sums.loc[cat].values - own_vec
        others_count = category_counts.loc[cat] - 1
        if others_count > 0:
            prior = _renormalize((others_sum / others_count).reshape(1, -1))[0]
        else:
            prior = global_prior
        w = weight_fn(row["n_reviews"], min_reviews)
        blended = w * own_vec + (1 - w) * prior
        new_vecs[i] = _renormalize(blended.reshape(1, -1))[0]

    out = df.drop(columns=["_category"])
    out[emb_cols] = new_vecs
    return out


def _demo():
    """
    Synthetic self-check: the current dataset has no businesses/users below
    the low_reviews threshold (business floor is 50, user floor is 3 --
    already filtered upstream), so blend_with_category_prior() is correctly
    implemented but INERT on real data. This demonstrates the blend math
    directly on fabricated low-review-count rows instead.
    """
    rng = np.random.default_rng(0)
    dim = 8
    raw = rng.normal(size=(5, dim))
    vecs = _renormalize(raw)
    df = pd.DataFrame({
        "business_id": ["A", "B", "C", "D", "E"],
        "n_reviews":   [2,   40,  1,   60,  3],
    })
    df = pd.concat([df, pd.DataFrame(vecs, columns=[f"emb_{i}" for i in range(dim)])], axis=1)
    min_reviews = 5
    df["low_confidence"] = df["n_reviews"] < min_reviews
    category_by_id = {
        "A": "Pizza, Restaurants",       # low-confidence, shares category with D
        "B": "Sushi, Japanese",          # high-confidence
        "C": "Pizza, Italian",           # low-confidence, shares category with D
        "D": "Pizza, Restaurants",       # high-confidence prior for A
        "E": "Tacos, Mexican",           # low-confidence, no category peers -> global prior
    }
    blended = blend_with_category_prior(df, category_by_id, min_reviews)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    print("business  n_reviews  low_conf  moved_toward_prior(cos to own original)")
    for i, row in df.iterrows():
        own = row[emb_cols].values.astype(float)
        new = blended.loc[i, emb_cols].values.astype(float)
        cos_to_own = float(np.dot(own, new))  # both unit vectors -> dot = cosine
        print(f"{row.business_id:>8}  {row.n_reviews:>9}  {str(row.low_confidence):>8}  "
              f"{cos_to_own:.4f}"
              f"{'  (unchanged, expected 1.0)' if not row.low_confidence else ''}")


if __name__ == "__main__":
    _demo()
