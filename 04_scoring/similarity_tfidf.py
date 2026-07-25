"""
Cosine similarity scoring between a user TF-IDF profile vector and a sparse
matrix of business TF-IDF profile vectors (03_profile_building/tfidf/).
Standalone counterpart to similarity_embedding.py -- same contract, sparse
business matrix instead of dense.
"""
import numpy as np
import scipy.sparse as sp


def cosine_similarity_via_dot(user_vec: np.ndarray, matrix: sp.csr_matrix) -> np.ndarray:
    """
    Cosine similarity via a plain dot product.

    VALID ONLY because the rows of `matrix` are unit length by construction
    (build_group_tfidf_profiles L2-normalizes every profile after the
    weighted-sum aggregation). `user_vec` is a dense array -- the aggregated
    query vector is built from at most a few hundred train reviews, so a
    dense 20,000-float vector is trivial (~160KB), and `matrix @ user_vec`
    (sparse @ dense) already returns a plain 1-D ndarray directly, with no
    `.toarray()`/`.T` needed. Defensively renormalizes `user_vec` in case it
    drifted or wasn't pre-normalized; does NOT renormalize `matrix` rows --
    those are fixed at profile-build time.
    """
    norm = np.linalg.norm(user_vec)
    unit = user_vec / norm if norm > 0 else user_vec
    return matrix @ unit
