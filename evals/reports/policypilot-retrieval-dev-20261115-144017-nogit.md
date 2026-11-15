# Retrieval ablation (dev, 63 answerable questions)

| Mode | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.603 | 0.865 | 0.897 | 0.968 | 0.743 | 0.773 | 29 | 36 |
| vector | 0.937 | 1.000 | 1.000 | 1.000 | 0.984 | 0.988 | 548 | 825 |
| hybrid | 0.778 | 0.984 | 1.000 | 1.000 | 0.883 | 0.910 | 46 | 51 |
| hybrid+rerank | 0.937 | 1.000 | 1.000 | 1.000 | 0.981 | 0.986 | 3673 | 35948 |

Structural leaks (retrieved chunk outside the asker's groups, all 97 queries x modes): **0**
