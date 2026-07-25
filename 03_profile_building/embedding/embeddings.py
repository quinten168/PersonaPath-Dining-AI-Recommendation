"""
Generate sentence-embedding vectors for every review in the Philadelphia
review corpus — an additive alternative to the LDA topic vectors already in
data/profiles/business_profiles.csv and user_profiles.csv.

Input:  data/interim/philly_reviews.parquet   (from 01_data_processing/prep_reviews.py)
Output:
    data/interim/review_embeddings.npy             (per-review vectors, float32)
    data/interim/review_embeddings_meta.csv        (review_id, business_id, user_id,
                                             stars, date — same row order as
                                             the .npy)

Model: sentence-transformers/all-MiniLM-L6-v2 (local, free, CPU, 384-dim).

Per-business / per-user profile vectors (mean-pooled or recency-weighted,
with cold-start handling) are built separately in build_profiles.py
using aggregation.py — kept out of this file so the aggregation
strategy can be swapped without touching embedding generation.

MEMORY NOTE: this streams through the parquet file in row-batches (default
5,000 rows) rather than loading all ~275K review texts and their embeddings
into RAM at once. Per-review vectors are written directly into a pre-sized
on-disk memmap (data/interim/review_embeddings.npy) slice-by-slice, so peak
RAM stays roughly proportional to one batch, not the whole corpus.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

ROOT = "../.."  # run this script from inside 03_profile_building/embedding/
REVIEWS_PARQUET = f"{ROOT}/data/interim/philly_reviews.parquet"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256          # sentence-transformers encode() micro-batch size
READ_BATCH_SIZE = 5000    # rows pulled from parquet per streaming chunk

OUT_REVIEW_EMB_NPY = f"{ROOT}/data/interim/review_embeddings.npy"
OUT_REVIEW_EMB_META = f"{ROOT}/data/interim/review_embeddings_meta.csv"


def main():
    pf = pq.ParquetFile(REVIEWS_PARQUET)
    n_rows = pf.metadata.num_rows
    print(f"Reviews: {n_rows:,} (streaming in batches of {READ_BATCH_SIZE:,})")

    print(f"Loading model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_embedding_dimension()

    # Pre-sized on-disk memmap -- per-review vectors are written directly to
    # disk as each batch is encoded, so the full (n_rows, dim) array never
    # exists as an in-RAM numpy array.
    review_embeddings = np.lib.format.open_memmap(
        OUT_REVIEW_EMB_NPY, mode="w+", dtype=np.float32, shape=(n_rows, dim)
    )

    meta_header_written = False
    row_offset = 0
    for batch in pf.iter_batches(batch_size=READ_BATCH_SIZE,
                                  columns=["review_id", "business_id", "user_id", "stars", "date", "text"]):
        chunk = batch.to_pandas()
        n = len(chunk)

        chunk_embeddings = model.encode(
            chunk["text"].tolist(),
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        review_embeddings[row_offset:row_offset + n] = chunk_embeddings

        chunk[["review_id", "business_id", "user_id", "stars", "date"]].to_csv(
            OUT_REVIEW_EMB_META, mode="a", index=False, header=not meta_header_written
        )
        meta_header_written = True

        row_offset += n
        print(f"  encoded {row_offset:,} / {n_rows:,}")

    review_embeddings.flush()
    print(f"Saved per-review embeddings: {OUT_REVIEW_EMB_NPY}")
    print("Run build_profiles.py next to aggregate these into "
          "per-business/per-user profile vectors.")


if __name__ == "__main__":
    main()
