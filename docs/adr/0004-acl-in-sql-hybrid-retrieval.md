# ADR-004: Permissions in the retrieval SQL, hybrid search with RRF, not an agent

Status: accepted · 2026-11-15

## Context

Legal's requirement: manager-only, HR-only and finance-only content must never reach regular
staff. Employees phrase questions in ways keyword search misses. Policies differ by region.

## Decision

1. **Access control lives where the data lives.** Every lexical and vector query carries
   `acl_groups && :user_groups` (plus "not superseded"), so restricted chunks never leave
   Postgres for that user. The model can't leak what it never receives. Prompts aren't access
   control.
2. **Hybrid retrieval:** Postgres full-text search (`tsvector`, OR-of-lexemes query) and pgvector
   HNSW (cosine, 768-dim Gemini embeddings), fused with Reciprocal Rank Fusion (k = 60). RRF needs
   no score calibration between the two. With pgvector ≥ 0.8, iterative scan
   (`hnsw.iterative_scan = relaxed_order`) keeps the ANN result count up when the ACL filter
   removes many rows.
3. **Gate before generation:** if the best vector similarity is under the calibrated threshold
   (0.59 on dev), abstain without calling the model.
4. **One generation step, then code:** structured answer, citation validator, output safety
   filter. This is not an agent: no query rewriting or multi-hop retrieval unless the eval
   shows misses that need it.

## Measured outcome (dev ablation, 2026-11-15)

The hypothesis "hybrid ≥ either retriever alone" was **wrong on this corpus**:

| Mode | R@1 | R@5 | MRR |
|---|---|---|---|
| lexical | 0.60 | 0.87 | 0.74 |
| vector | **0.94** | **1.00** | **0.98** |
| hybrid (RRF) | 0.78 | 0.98 | 0.88 |
| hybrid + LLM rerank | 0.94 | 1.00 | 0.98 (p50 3.7 s) |

Employees paraphrase ("vacation days" for "annual leave"), so lexical ranks are noisy, and
equal-weight RRF lets them pull good vector hits down. The default is therefore **vector**.
Lexical stays as the automatic fallback when the embedding provider fails (degraded mode), and
hybrid stays selectable. Revisit with weighted RRF, or when the corpus has many exact
identifiers (policy numbers, product codes), where lexical usually wins.

## Consequences

- Zero-leak is a property of the query, testable without a model (the property test covers
  6 personas × 3 retrievers).
- One store for text, vectors and ACLs, updated in one transaction.
- The LLM reranker is optional and controlled by a flag; the ablation decides whether it earns
  its latency.

## Alternatives considered

- **Filtering after retrieval in Python:** a restricted chunk is fetched first, and a bug
  leaks it.
- **A dedicated vector DB with metadata filters:** another system, and its ACL sync must be
  trusted.
- **A per-group index:** explodes with overlapping groups.
