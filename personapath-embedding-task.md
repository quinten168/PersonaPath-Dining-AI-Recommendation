# Task: Add embedding-based retrieval to PersonaPath

## Objective
Add sentence-embedding similarity as an alternate/additional scoring signal in the
recommendation pipeline, alongside the existing LDA + JSD approach. Do not remove or
break the existing LDA pipeline — this is additive, evaluated side by side.

## Context
Current pipeline: LDA topic modeling → JSD similarity → MMR reranking → LLM explanation.
LDA topics are kept because they feed the LLM explanation layer with interpretable labels.
Goal: test whether embedding-based cosine similarity improves ranking quality (NDCG)
over JSD, while keeping LDA topics available for explanations regardless of which
scoring path wins.

## Scope

In scope:
- New embedding generation step for restaurant reviews (batch, offline/precomputed — not real-time)
- Cosine similarity scoring function as an alternative to JSD
- A config flag or parameter to switch scoring method between `jsd` and `embedding`
  (and ideally `hybrid`, a weighted blend of both) without duplicating the MMR reranking logic
- Aggregation strategy for combining multiple reviews per restaurant into one representative vector
- Evaluation harness reuse: run the exact same NDCG eval set/metric used for the LDA baseline
  against the new embedding scores, so results are directly comparable

Out of scope (do not touch):
- LDA topic modeling code itself
- LLM explanation generation prompts/logic
- MMR reranking algorithm structure (only the similarity function it calls should change)
- Any UI/frontend layer

## Technical requirements
- Library: `sentence-transformers`, model `all-MiniLM-L6-v2` (local, free, fast — no external
  API calls, no added cost/latency budget for this stage)
- Use `normalize_embeddings=True` at encode time so cosine similarity reduces to a dot product
- Store precomputed embeddings (don't re-embed on every request) — [FILL IN: existing storage
  convention, e.g. pickle file / parquet column / vector column in DB]
- Per-restaurant aggregation: default to mean-pooling across that restaurant's review embeddings,
  but implement this as a swappable function — flag it clearly as a simplification, since
  averaging can wash out mixed signals (e.g. "romantic" + "loud on weekends" reviews)
- Keep the JSD path fully intact and runnable — new code should live alongside it, not replace it

## Files likely affected
[FILL IN — point Claude Code at your actual module paths, e.g.:]
- `pipeline/similarity.py` — add `cosine_similarity()` alongside existing `jsd_similarity()`
- `pipeline/embeddings.py` — new file, embedding generation + aggregation
- `pipeline/rerank.py` — parameterize which similarity function MMR calls
- `eval/ndcg_eval.py` — reuse for the new scoring path, no changes to the metric itself

## Acceptance criteria
- [ ] Can run the pipeline end-to-end with `scoring_method="embedding"` and get ranked results
- [ ] Can run with `scoring_method="jsd"` and get identical output to current behavior (no regression)
- [ ] NDCG reported for both methods on the same eval set, printed/logged side by side
- [ ] LLM explanation step still works unmodified, still pulling from LDA topics regardless of
      which scoring method produced the ranking
- [ ] No new external API dependencies or costs introduced

## Notes for Claude Code
- Ask before restructuring existing pipeline files — prefer additive changes over refactors
- Match existing code style and naming conventions in the repo
- Flag any place where the embedding aggregation strategy is a known simplification, so it's
  easy to find and revisit later
