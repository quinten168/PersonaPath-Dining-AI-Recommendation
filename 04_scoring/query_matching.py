"""
Blend a live free-text user prompt against business profiles.

Reuses the existing cosine-similarity primitives (similarity_tfidf.py,
similarity_embedding.py) unchanged -- a vectorized/encoded query is just
another "user_vec" in the same shape those functions already expect, so no
new similarity math is needed, only new vectorization of the query text.

Two prompt-match signals are computed -- TF-IDF (literal keyword overlap)
and embedding (paraphrase/semantic overlap, e.g. prompt says "romantic",
reviews say "candlelit") -- and blended together, then combined with the
user's own historical taste-profile similarity as a secondary
personalization signal. The live prompt dominates by design: an explicit
ask should outweigh general taste history, not be averaged equally with it.

No LLM API call is involved in the vectorization/matching step itself --
a local TF-IDF transform() or sentence-transformer encode() is faster and
free compared to a network round trip. An LLM's place in this pipeline is
downstream of this module: re-ranking or explaining the already-narrowed
shortlist this produces, not computing the match.
"""
import numpy as np
import scipy.sparse as sp

from similarity_tfidf import cosine_similarity_via_dot as tfidf_cosine_similarity
from similarity_embedding import cosine_similarity_via_dot as embedding_cosine_similarity


def match_prompt_to_businesses(
    query_text: str,
    tfidf_vectorizer,
    business_tfidf_matrix: sp.csr_matrix,
    embedding_model=None,
    business_embedding_matrix: np.ndarray = None,
    tfidf_weight: float = 0.5,
) -> np.ndarray:
    """
    Vectorize a live free-text prompt with the SAME fitted TfidfVectorizer
    and sentence-transformer used to build the business profiles (transform/
    encode only -- never re-fit), then blend TF-IDF cosine similarity against
    embedding cosine similarity into one prompt<->business score per
    business, row-aligned with business_tfidf_matrix / business_embedding_matrix.

    `tfidf_weight` blends the two prompt-match signals against each other
    (0.5 = equal weight) -- independent of the prompt-vs-profile blend done
    in `blend_prompt_and_profile` below.

    `embedding_model` / `business_embedding_matrix` are optional: if either
    is None, this returns TF-IDF-only prompt matching, mirroring this
    codebase's existing degrade-gracefully-without-embeddings convention
    (e.g. 06_recommender_app/app.py's load_embeddings()).
    """
    query_tfidf_vec = np.asarray(tfidf_vectorizer.transform([query_text]).todense()).ravel()
    tfidf_scores = tfidf_cosine_similarity(query_tfidf_vec, business_tfidf_matrix)

    if embedding_model is None or business_embedding_matrix is None:
        return tfidf_scores

    query_embedding_vec = embedding_model.encode(
        [query_text], normalize_embeddings=True, convert_to_numpy=True,
    )[0]
    embedding_scores = embedding_cosine_similarity(query_embedding_vec, business_embedding_matrix)

    return tfidf_weight * tfidf_scores + (1 - tfidf_weight) * embedding_scores


def blend_prompt_and_profile(
    prompt_match_scores: np.ndarray,
    profile_match_scores: np.ndarray,
    prompt_weight: float = 0.75,
) -> np.ndarray:
    """
    Combine the live prompt<->business match with the user's own historical
    taste-profile<->business similarity into one final ranking score.

    The explicit live ask dominates by design (default prompt_weight=0.75):
    the user's general historical taste is a smaller personalization nudge
    on top of it, not an equal partner. Tune prompt_weight against real
    query examples rather than treating 0.75 as fixed -- same spirit as the
    tunable similarity/EAS blend in evaluate_recommender.py.
    """
    return prompt_weight * prompt_match_scores + (1 - prompt_weight) * profile_match_scores
