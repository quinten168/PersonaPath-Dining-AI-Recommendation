"""
One-off: stream through the 6.9M-row Yelp reviews JSON and keep only the rows
that matter for our Philly recommender evaluation + embedding generation.

Input:  data/raw/yelp_academic_dataset_review.json  (JSON lines, one review per line)
Output: data/interim/philly_reviews.parquet

Adapted from an earlier session's version (originally had to stream through a
zipped/tarred mount due to FUSE read limits on a remote sandbox). Locally the
JSON is already extracted, so this just reads it directly line-by-line.

Difference from the original: also keeps `text`, since the embedding step
(02_profile_building/embedding/embeddings.py) and the TF-IDF vectorizer
(02_profile_building/tfidf/vectorize.py) need real review text, not just stars.
"""
import json
import time

import pandas as pd

ROOT = ".."  # run this script from inside 01_data_processing/
REVIEW_JSON = f"{ROOT}/data/raw/yelp_academic_dataset_review.json"
BUSINESS_PROFILES = f"{ROOT}/data/profiles/business_profiles.csv"
USER_PROFILES = f"{ROOT}/data/profiles/user_profiles.csv"
OUT_PARQUET = f"{ROOT}/data/interim/philly_reviews.parquet"

MIN_TEXT_LEN = 50  # mirrors notebook 01's review_text length filter

# ── Load the ID universes we care about ─────────────────────────────────
biz_ids = set(pd.read_csv(BUSINESS_PROFILES)["business_id"])
user_ids = set(pd.read_csv(USER_PROFILES)["user_id"])
print(f"Target businesses: {len(biz_ids):,}")
print(f"Target users     : {len(user_ids):,}")

# ── Stream reviews line-by-line (never load the 5.3GB file into memory) ──
rows = []
start = time.time()
kept = 0
scanned = 0

with open(REVIEW_JSON, "r", encoding="utf-8") as f:
    for ln in f:
        scanned += 1
        if scanned % 500_000 == 0:
            elapsed = time.time() - start
            print(f"  scanned {scanned:>10,}  kept {kept:>7,}  "
                  f"({scanned/max(elapsed,1e-3):,.0f} lines/s)")
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        text = r.get("text", "") or ""
        if (r.get("business_id") in biz_ids and r.get("user_id") in user_ids
                and len(text) > MIN_TEXT_LEN):
            rows.append((
                r["review_id"], r["user_id"], r["business_id"],
                float(r["stars"]), r.get("date", ""), text,
            ))
            kept += 1

print(f"\nTotal scanned: {scanned:,}  kept: {kept:,}  elapsed: {time.time()-start:.1f}s")

df = pd.DataFrame(rows, columns=["review_id", "user_id", "business_id", "stars", "date", "text"])
df.to_parquet(OUT_PARQUET, index=False)
print(f"Saved: {OUT_PARQUET}  ({len(df):,} rows, {df.memory_usage(deep=True).sum()/1e6:.1f} MB in RAM)")

print("\nDistribution of reviews per user (after filter):")
print(df.groupby("user_id").size().describe([.5, .75, .9, .95, .99]).to_string())
print("\nStars distribution:")
print(df["stars"].value_counts().sort_index().to_string())
