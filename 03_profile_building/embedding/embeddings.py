"""
Generate sentence-embedding vectors for every review in the New Orleans
review corpus — an additive alternative to the LDA topic vectors already in
data/profiles/business_profiles.csv and user_profiles.csv (Philadelphia-era;
not carried over for New Orleans -- see the project plan).

Input:  persona-path.default.new_orleans_reviews   (Unity Catalog managed table,
                                                     from 01_data_processing/01_data_processing.ipynb)
Output (Unity Catalog Volume, so these survive past this job's cluster):
    /Volumes/persona-path/default/interim/review_embeddings.npy             (per-review vectors, float32)
    /Volumes/persona-path/default/interim/review_embeddings_meta.csv        (review_id, business_id, user_id,
                                             stars, date — same row order as
                                             the .npy)

Model: sentence-transformers/all-MiniLM-L6-v2 (local, free, CPU, 384-dim).

Per-business / per-user profile vectors (mean-pooled or recency-weighted,
with cold-start handling) are built separately in build_profiles.py
using aggregation.py — kept out of this file so the aggregation
strategy can be swapped without touching embedding generation.

Requires a live Spark session (run as a Databricks notebook/job) -- reads
via spark.table(), not a local file, so this can no longer run as a plain
`python3 embeddings.py`.

MEMORY NOTE: unlike the original Philadelphia version of this script (which
streamed through ~275K reviews in row-batches via pyarrow, writing directly
into a pre-sized on-disk memmap to keep peak RAM proportional to one batch),
this loads the whole New Orleans corpus into memory at once via
`.toPandas()`. New Orleans is a much smaller city than Philadelphia with the
same review_count>=50 business floor, so the corpus is expected to be far
below the ~275K-review scale that motivated the streaming design. If this
corpus turns out to be large enough to risk OOM, reintroduce
chunked/batched processing (e.g. Spark `mapInPandas` per partition) rather
than assuming this will always fit in memory.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

TABLE_NAME = "persona-path.default.new_orleans_reviews"
VOLUME_ROOT = "/Volumes/persona-path/default"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256          # sentence-transformers encode() micro-batch size

OUT_REVIEW_EMB_NPY = f"{VOLUME_ROOT}/interim/review_embeddings.npy"
OUT_REVIEW_EMB_META = f"{VOLUME_ROOT}/interim/review_embeddings_meta.csv"


def main():
    print(f"Loading review corpus: {TABLE_NAME}")
    df = spark.table(TABLE_NAME).select(
        "review_id", "business_id", "user_id", "stars", "date", "text",
    ).toPandas()
    df["text"] = df["text"].fillna("")
    n_rows = len(df)
    print(f"Reviews: {n_rows:,}")

    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    review_embeddings = model.encode(
        df["text"].tolist(),
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(OUT_REVIEW_EMB_NPY, review_embeddings)
    print(f"Saved per-review embeddings: {OUT_REVIEW_EMB_NPY}")

    df[["review_id", "business_id", "user_id", "stars", "date"]].to_csv(
        OUT_REVIEW_EMB_META, index=False
    )
    print(f"Saved: {OUT_REVIEW_EMB_META}")
    print("Run build_profiles.py next to aggregate these into "
          "per-business/per-user profile vectors.")


if __name__ == "__main__":
    main()
