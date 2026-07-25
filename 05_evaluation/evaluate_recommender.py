"""
80/20 offline evaluation of the PersonaPath recommender.

For each qualifying user:
  - 80% of their Philly reviews -> rebuild user vector (train)
  - 20% of their reviews -> hold out as ground truth (test)
  - Ground truth "liked" = test reviews with stars >= 4
  - Recommend Top-K from remaining businesses
  - Compute Precision@K, Recall@K, NDCG@K, HitRate@K, MRR

Systems compared:
  - cbf         : cosine similarity over LDA topic vectors + EAS (original system)
  - embedding   : cosine similarity over sentence-transformer review embeddings + EAS (additive)
  - tfidf       : cosine similarity over TF-IDF review-term vectors + EAS (additive,
                  standalone pipeline -- see 03_profile_building/tfidf/)
  - popularity  : stars x log(1+review_count), no personalization
  - random      : sanity floor

Adapted from an earlier session's version (originally ran against a remote
sandbox mount; paths updated here to this repo's local layout). The cbf/
popularity/random logic is UNCHANGED from that version -- the "embedding"
and "tfidf" systems and their wiring are additive.

KNOWN LEAKAGE ASYMMETRY (read before trusting a head-to-head "winner"):
  "embedding" rebuilds each user's vector from ONLY their train-split review
  embeddings -- a true leave-out split, no leakage.
  "cbf" rebuilds each user's vector from LDA topic scores in
  data/profiles/business_profiles.csv, which were derived from the training
  review's BUSINESS-level topic vector (not the user's own historical LDA
  vector), so cbf itself does not leak the held-out review either -- both
  arms are leak-free in that specific sense. The real asymmetry is upstream:
  the LDA topic model itself was trained once on the full corpus (including
  these same reviews) back in notebook 02, and was never saved, so it cannot
  be retrained per-split locally (out of scope here). This gives the LDA
  topic vectors a mild, structural, hard-to-quantify advantage baked in at
  the corpus level that "embedding" does not share, since embeddings are
  generated fresh, per-review, with no cross-review training step at all.
  Keep this in mind when comparing the two -- it is not a perfectly
  apples-to-apples comparison, and the bias runs in cbf's favor.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# No package structure in this repo (flat scripts run via `python3 script.py`
# from inside their own directory) -- similarity_embedding.py/similarity_tfidf.py
# now live in a sibling stage folder (04_scoring/), not next to this file, so
# they need an explicit, __file__-anchored sys.path entry rather than a plain
# same-folder import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_scoring"))
from similarity_embedding import cosine_similarity_via_dot
from similarity_tfidf import cosine_similarity_via_dot as tfidf_cosine_similarity_via_dot

ROOT = ".."  # run this script from inside 05_evaluation/
BUSINESS_PROFILES = f"{ROOT}/data/profiles/business_profiles.csv"
USER_PROFILES = f"{ROOT}/data/profiles/user_profiles.csv"
REVIEWS_PARQUET = f"{ROOT}/data/interim/philly_reviews.parquet"
BUSINESS_EMBEDDINGS = f"{ROOT}/data/profiles/business_embeddings.csv"
REVIEW_EMBEDDINGS_NPY = f"{ROOT}/data/interim/review_embeddings.npy"
REVIEW_EMBEDDINGS_META = f"{ROOT}/data/interim/review_embeddings_meta.csv"
BUSINESS_TFIDF_NPZ = f"{ROOT}/data/profiles/business_tfidf_profiles.npz"
BUSINESS_TFIDF_META = f"{ROOT}/data/profiles/business_tfidf_profiles_meta.csv"
REVIEW_TFIDF_NPZ = f"{ROOT}/data/interim/review_tfidf.npz"
REVIEW_TFIDF_META = f"{ROOT}/data/interim/review_tfidf_meta.csv"

TOPIC_COLS = [
    "bar_nightlife_crowd", "bar_vibes_live_music", "brunch_breakfast",
    "cafe_coffee_reading_terminal", "casual_payment_ordering",
    "chef_specials_platters", "cocktail_bars_speakeasy",
    "comfort_food_sandwiches", "craft_beer_sports_bars",
    "customer_service_quality", "desserts_bakery", "dinner_happy_hour",
    "fine_dining_tasting_menu", "food_trucks_trendy_spots",
    "markets_grocery_shopping", "outdoor_street_seating",
    "overall_food_quality", "philly_neighborhood_gems", "quick_fresh_lunch",
    "service_wait_time", "small_plates_cocktails", "spicy_asian_flavors",
    "unique_quirky_dining", "value_portion_size", "venue_location_experience",
]

# ── Tunables ────────────────────────────────────────────────────────────
K_VALUES        = [5, 10]
MIN_REVIEWS     = 10       # skip users with too few reviews to split
EVAL_SAMPLE     = 2000     # cap users evaluated (speed); set None for all
SEED            = 42
LIKED_THRESHOLD = 4.0      # stars >= 4 counts as liked
ALPHA           = 0.6      # recommender weight on similarity
BETA_EAS        = 0.4      # recommender weight on EAS (no query here)

# ── Load data ───────────────────────────────────────────────────────────
print("Loading data...")
rev   = pd.read_parquet(REVIEWS_PARQUET)
biz   = pd.read_csv(BUSINESS_PROFILES)
users = pd.read_csv(USER_PROFILES)

# Keep only reviews for businesses we actually have in the profile
rev = rev[rev.business_id.isin(biz.business_id)]
print(f"Reviews : {len(rev):,}")
print(f"Users   : {rev.user_id.nunique():,}")
print(f"Biz     : {biz.shape[0]:,}")

# Business matrix + quality score (EAS rebuild, same as the notebook)
biz = biz.reset_index(drop=True)
biz_matrix = biz[TOPIC_COLS].fillna(0).values
sent_shift = (biz["overall_sentiment"].fillna(0) + 1) / 2
log_rev    = np.log1p(biz["review_count"].fillna(0))
focus      = biz["dominant_topic_score"].fillna(0)
eas_raw    = (sent_shift * log_rev * focus).clip(lower=0)
biz["eas_norm"]    = (eas_raw - eas_raw.min()) / (eas_raw.max() - eas_raw.min() + 1e-9)
biz["popularity"]  = biz["business_stars"].fillna(0) * np.log1p(biz["review_count"].fillna(0))

# Fast lookups
bid_to_row    = {bid: i for i, bid in enumerate(biz.business_id.values)}
biz_ids_array = biz.business_id.values
eas_array     = biz["eas_norm"].values
pop_array     = biz["popularity"].values

# ── Embedding artifacts (additive) ───────────────────────────────────────
biz_emb_csv = pd.read_csv(BUSINESS_EMBEDDINGS)
EMB_COLS    = [c for c in biz_emb_csv.columns if c.startswith("emb_")]
EMB_DIM     = len(EMB_COLS)
# Align to the SAME row order as `biz` / bid_to_row, so exclude_rows/ranking
# indices are shared across the cbf and embedding systems. Businesses with no
# surviving embedding (e.g. filtered out by prep_reviews.py's text-length
# floor) fall back to a zero vector -- cosine similarity against a zero
# vector is ~0, i.e. inert, never crashes.
biz_emb_aligned = biz[["business_id"]].merge(biz_emb_csv, on="business_id", how="left")
biz_emb_matrix  = biz_emb_aligned[EMB_COLS].fillna(0).values

review_embeddings = np.load(REVIEW_EMBEDDINGS_NPY)
review_meta       = pd.read_csv(REVIEW_EMBEDDINGS_META)
review_id_to_row  = {rid: i for i, rid in enumerate(review_meta.review_id.values)}
review_ids_with_emb = set(review_id_to_row.keys())

# ── TF-IDF artifacts (additive, standalone pipeline) ─────────────────────
tfidf_profile_matrix = sp.load_npz(BUSINESS_TFIDF_NPZ).tocsr()
tfidf_profile_meta   = pd.read_csv(BUSINESS_TFIDF_META)
TFIDF_DIM = tfidf_profile_matrix.shape[1]
# Align to the SAME row order as `biz` / bid_to_row via an explicit sparse
# selection matmul -- NOT the merge()+fillna(0) trick used for
# biz_emb_matrix above. A -1 "missing" sentinel in sparse fancy-indexing
# silently wraps to the matrix's LAST row instead of producing a zero row,
# so a genuine permutation/selection matrix is required here; businesses
# absent from tfidf_profile_meta get a real all-zero row (inert, cosine
# similarity ~0, never crashes).
tfidf_bid_to_row = {bid: i for i, bid in enumerate(tfidf_profile_meta.business_id.values)}
_sel = np.array([tfidf_bid_to_row.get(bid, -1) for bid in biz_ids_array])
_present = _sel != -1
_P = sp.coo_matrix(
    (np.ones(_present.sum()), (np.arange(len(biz_ids_array))[_present], _sel[_present])),
    shape=(len(biz_ids_array), tfidf_profile_matrix.shape[0]),
).tocsr()
biz_tfidf_matrix = (_P @ tfidf_profile_matrix).tocsr()

review_tfidf         = sp.load_npz(REVIEW_TFIDF_NPZ).tocsr()
review_tfidf_meta    = pd.read_csv(REVIEW_TFIDF_META)
review_tfidf_id_to_row = {rid: i for i, rid in enumerate(review_tfidf_meta.review_id.values)}
review_ids_with_tfidf = set(review_tfidf_id_to_row.keys())

# Qualifying users
counts = rev.groupby("user_id").size()
pool = counts[counts >= MIN_REVIEWS].index.to_numpy()
rng = np.random.default_rng(SEED)
if EVAL_SAMPLE is not None and len(pool) > EVAL_SAMPLE:
    eval_users = rng.choice(pool, size=EVAL_SAMPLE, replace=False)
else:
    eval_users = pool
print(f"Evaluating {len(eval_users):,} users (min {MIN_REVIEWS} reviews each)")

# Group reviews once for fast access
rev_by_user = {u: df for u, df in rev.groupby("user_id")}

# User -> persona mapping for breakdown
persona_by_user = dict(zip(users.user_id, users.persona_label.fillna("Unknown")))

# ── Metrics ─────────────────────────────────────────────────────────────
def metrics_at_k(ranked_ids, test_liked, k):
    top_k = ranked_ids[:k]
    hits  = [b for b in top_k if b in test_liked]
    prec  = len(hits) / k
    rec   = len(hits) / len(test_liked) if test_liked else 0.0
    dcg   = sum(1.0 / np.log2(i + 2) for i, b in enumerate(top_k) if b in test_liked)
    idcg  = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(test_liked))))
    ndcg  = dcg / idcg if idcg > 0 else 0.0
    hit   = 1.0 if hits else 0.0
    mrr   = 0.0
    for i, b in enumerate(top_k):
        if b in test_liked:
            mrr = 1.0 / (i + 1)
            break
    return prec, rec, ndcg, hit, mrr


def split_reviews(u, seed):
    """Deterministic 80/20 split per user."""
    df = rev_by_user[u]
    shuf = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_tr = int(round(len(shuf) * 0.8))
    n_tr = max(1, min(n_tr, len(shuf) - 1))  # ensure both sides non-empty
    return shuf.iloc[:n_tr], shuf.iloc[n_tr:]


def build_user_vector(train_df):
    """
    Weighted average of the business topic vectors for training reviews.
    Weight = max(stars - 2.5, 0) / 2.5   -> 5*=1.0, 4*=0.6, 3*=0.2, <=2.5*=0
    Falls back to uniform weighting if the user has no >=3* reviews.
    """
    idxs   = [bid_to_row[b] for b in train_df.business_id if b in bid_to_row]
    stars  = train_df.stars.values
    if len(idxs) == 0:
        return np.zeros(len(TOPIC_COLS))
    weights = np.maximum(stars - 2.5, 0) / 2.5
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    weights = weights[:len(idxs)]
    return (biz_matrix[idxs] * weights[:, None]).sum(axis=0) / (weights.sum() + 1e-9)


def build_user_embedding_vector(train_df):
    """
    Weighted average of REVIEW-level embeddings for training reviews only
    (true leave-out: held-out test reviews are never included here, since
    train_df already excludes them via split_reviews()). Same star-weighting
    scheme as build_user_vector(), for a like-for-like comparison.
    Returns a zero vector if none of the user's train reviews have an
    embedding (e.g. filtered out upstream) -- caller skips this user for the
    embedding system in that case.
    """
    mask = train_df.review_id.isin(review_ids_with_emb)
    sub = train_df[mask]
    if sub.empty:
        return np.zeros(EMB_DIM)
    rows = [review_id_to_row[rid] for rid in sub.review_id]
    weights = np.maximum(sub.stars.values - 2.5, 0) / 2.5
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    vecs = review_embeddings[rows]
    return (vecs * weights[:, None]).sum(axis=0) / (weights.sum() + 1e-9)


def build_user_tfidf_vector(train_df):
    """
    Weighted average of REVIEW-level TF-IDF vectors for training reviews
    only (true leave-out, same as build_user_embedding_vector() -- train_df
    already excludes held-out test reviews via split_reviews()). Same
    star-weighting scheme, for a like-for-like comparison across all three
    systems. Returns a dense zero vector if none of the user's train
    reviews have a TF-IDF row -- caller skips this user for the tfidf
    system in that case.

    Returned vector is DENSE, not sparse: it's built from at most a few
    hundred train reviews (this dataset's MIN_REVIEWS floor), so a dense
    TFIDF_DIM-length float vector is trivial (~160KB at 20K features), and
    keeping it dense simplifies rank_tfidf()/similarity_tfidf.py (sparse @
    dense returns a plain 1-D ndarray directly).
    """
    mask = train_df.review_id.isin(review_ids_with_tfidf)
    sub = train_df[mask]
    if sub.empty:
        return np.zeros(TFIDF_DIM)
    rows = [review_tfidf_id_to_row[rid] for rid in sub.review_id]
    weights = np.maximum(sub.stars.values - 2.5, 0) / 2.5
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    vecs = review_tfidf[rows]  # sparse (n_sub, TFIDF_DIM)
    weighted = vecs.multiply(weights[:, None])
    # scipy sparse .sum(axis=0) returns a numpy.matrix, not a flat ndarray --
    # must ravel() before treating it as a plain vector.
    return np.asarray(weighted.sum(axis=0)).ravel() / (weights.sum() + 1e-9)


def rank_cbf(user_vec, exclude_rows):
    """Original recommender's ranking: cosine similarity (LDA topics) + EAS."""
    sim = cosine_similarity(user_vec.reshape(1, -1), biz_matrix)[0]
    score = ALPHA * sim + BETA_EAS * eas_array
    score[list(exclude_rows)] = -np.inf
    order = np.argsort(-score)
    return biz_ids_array[order]


def rank_embedding(user_emb_vec, exclude_rows):
    """New system: cosine similarity (sentence embeddings, via similarity.py) + EAS."""
    sim = cosine_similarity_via_dot(user_emb_vec, biz_emb_matrix)
    score = ALPHA * sim + BETA_EAS * eas_array
    score[list(exclude_rows)] = -np.inf
    order = np.argsort(-score)
    return biz_ids_array[order]


def rank_tfidf(user_tfidf_vec, exclude_rows):
    """Standalone system: cosine similarity (TF-IDF review terms, via
    similarity_tfidf.py) + EAS. Mirrors rank_embedding() exactly."""
    sim = tfidf_cosine_similarity_via_dot(user_tfidf_vec, biz_tfidf_matrix)
    score = ALPHA * sim + BETA_EAS * eas_array
    score[list(exclude_rows)] = -np.inf
    order = np.argsort(-score)
    return biz_ids_array[order]


def rank_popularity(exclude_rows):
    """Baseline: popularity only, no personalization."""
    score = pop_array.copy().astype(float)
    score[list(exclude_rows)] = -np.inf
    order = np.argsort(-score)
    return biz_ids_array[order]


def rank_random(exclude_rows, rng_local):
    """Sanity floor: random ranking."""
    score = rng_local.random(len(biz_ids_array))
    score[list(exclude_rows)] = -np.inf
    order = np.argsort(-score)
    return biz_ids_array[order]


# ── Run evaluation ──────────────────────────────────────────────────────
rows = []
rng_local = np.random.default_rng(SEED + 1)
skipped = 0
skipped_embedding_only = 0
skipped_tfidf_only = 0

for i, uid in enumerate(eval_users):
    if i % 500 == 0:
        print(f"  eval progress: {i:,} / {len(eval_users):,}")
    train_df, test_df = split_reviews(uid, SEED)
    test_liked = set(test_df[test_df.stars >= LIKED_THRESHOLD].business_id)
    if not test_liked:
        skipped += 1
        continue

    # Exclude anything the user already saw in training (can't "recommend" it)
    excl_rows = {bid_to_row[b] for b in train_df.business_id if b in bid_to_row}

    uvec = build_user_vector(train_df)
    if uvec.sum() == 0:
        skipped += 1
        continue

    rankings = {
        "cbf":        rank_cbf(uvec, excl_rows),
        "popularity": rank_popularity(excl_rows),
        "random":     rank_random(excl_rows, rng_local),
    }

    uemb = build_user_embedding_vector(train_df)
    if uemb.sum() != 0:
        rankings["embedding"] = rank_embedding(uemb, excl_rows)
    else:
        skipped_embedding_only += 1

    utfidf = build_user_tfidf_vector(train_df)
    if utfidf.sum() != 0:
        rankings["tfidf"] = rank_tfidf(utfidf, excl_rows)
    else:
        skipped_tfidf_only += 1

    persona = persona_by_user.get(uid, "Unknown")
    for sys_name, ranked in rankings.items():
        for k in K_VALUES:
            p, r, n, h, m = metrics_at_k(ranked, test_liked, k)
            rows.append({
                "user_id": uid, "persona": persona,
                "system":  sys_name, "k": k,
                "precision": p, "recall": r, "ndcg": n,
                "hit_rate":  h, "mrr": m,
                "n_test_liked": len(test_liked),
                "n_train": len(train_df),
            })

print(f"Evaluated {len(eval_users) - skipped:,} users ({skipped} skipped for no test-liked / empty vector)")
print(f"  of which {skipped_embedding_only} had no embedding coverage and were skipped for 'embedding' only "
      f"(still counted for cbf/popularity/random)")
print(f"  of which {skipped_tfidf_only} had no TF-IDF coverage and were skipped for 'tfidf' only "
      f"(still counted for cbf/popularity/random)")

# ── Aggregate ───────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

print("\n" + "=" * 78)
print("OVERALL RESULTS (mean across users)")
print("=" * 78)
agg = (
    df.groupby(["system", "k"])[["precision", "recall", "ndcg", "hit_rate", "mrr"]]
      .mean()
      .round(4)
)
print(agg.to_string())

# Lift vs baselines
print("\n" + "=" * 78)
print("LIFT — cbf & embedding vs baselines")
print("=" * 78)
for k in K_VALUES:
    cbf  = df[(df.system == "cbf")        & (df.k == k)][["precision","recall","ndcg","hit_rate","mrr"]].mean()
    emb  = df[(df.system == "embedding")  & (df.k == k)][["precision","recall","ndcg","hit_rate","mrr"]].mean()
    pop  = df[(df.system == "popularity") & (df.k == k)][["precision","recall","ndcg","hit_rate","mrr"]].mean()
    rnd  = df[(df.system == "random")     & (df.k == k)][["precision","recall","ndcg","hit_rate","mrr"]].mean()
    print(f"\n@K={k}")
    print(f"  {'metric':<10} {'cbf':>8} {'embed':>8} {'popul':>8} {'random':>8}   "
          f"lift cbf-pop  lift emb-pop  lift emb-cbf")
    for m in ["precision", "recall", "ndcg", "hit_rate", "mrr"]:
        lift_cbf_pop = (cbf[m] / pop[m] - 1) * 100 if pop[m] > 0 else float("inf")
        lift_emb_pop = (emb[m] / pop[m] - 1) * 100 if pop[m] > 0 else float("inf")
        lift_emb_cbf = (emb[m] / cbf[m] - 1) * 100 if cbf[m] > 0 else float("inf")
        print(f"  {m:<10} {cbf[m]:>8.4f} {emb[m]:>8.4f} {pop[m]:>8.4f} {rnd[m]:>8.4f}  "
              f"{lift_cbf_pop:>+11.1f}%  {lift_emb_pop:>+11.1f}%  {lift_emb_cbf:>+11.1f}%")

print("\n" + "=" * 78)
print("LEAKAGE CAVEAT — read before treating 'embedding vs cbf' as a clean A/B")
print("=" * 78)
print(
    "embedding: leave-out user vectors, rebuilt fresh per split, no cross-review\n"
    "  training step.\n"
    "cbf: user vectors are a weighted average of each training business's LDA\n"
    "  topic vector -- also leave-out at the user level -- BUT the underlying\n"
    "  LDA topic model was trained once on the full corpus back in notebook 02\n"
    "  and never saved, so it could not be retrained per-split here (out of\n"
    "  scope). That gives cbf's topic vectors a mild, structural, hard-to-\n"
    "  quantify advantage that embedding does not share. If embedding matches\n"
    "  or beats cbf despite this, that's a stronger result than the raw\n"
    "  numbers show."
)

# Persona breakdown
print("\n" + "=" * 78)
print("BY PERSONA  (cbf only, @K=10)")
print("=" * 78)
sub = df[(df.system == "cbf") & (df.k == 10)]
pbreak = (
    sub.groupby("persona")[["precision","recall","ndcg","hit_rate","mrr"]]
       .mean()
       .join(sub.groupby("persona").size().rename("n_users"))
       .sort_values("precision", ascending=False)
       .round(4)
)
print(pbreak.head(15).to_string())

# Save per-user results
out = "eval_per_user.csv"
df.to_csv(out, index=False)
print(f"\nPer-user results saved: {out}")
