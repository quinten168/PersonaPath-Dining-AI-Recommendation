"""
Cosine similarity scoring between a user profile vector and a matrix of
business profile vectors (aggregation.py). Kept separate from
embedding generation and aggregation so the similarity function can be
reused (or swapped) without touching either.
"""
import numpy as np


def cosine_similarity_via_dot(user_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Cosine similarity via a plain dot product.

    VALID ONLY because both `user_vec` and the rows of `matrix` are unit
    length by construction (every aggregator in aggregation.py renormalizes
    after pooling/blending). Defensively renormalizes `user_vec` in case it
    drifted or wasn't pre-normalized; does NOT renormalize `matrix` rows --
    those are fixed at profile-build time, so renormalizing them on every
    call here would be wasted work at query time.
    """
    norm = np.linalg.norm(user_vec)
    unit = user_vec / norm if norm > 0 else user_vec
    return matrix @ unit
