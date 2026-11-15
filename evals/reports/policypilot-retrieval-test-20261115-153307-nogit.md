# Retrieval ablation (test, 32 answerable questions)

| Mode | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.562 | 0.906 | 0.969 | 1.000 | 0.734 | 0.784 | 24 | 51 |
| vector | 0.906 | 1.000 | 1.000 | 1.000 | 0.969 | 0.977 | 539 | 865 |
| hybrid | 0.781 | 1.000 | 1.000 | 1.000 | 0.896 | 0.920 | 45 | 64 |

Structural leaks (retrieved chunk outside the asker's groups, all 48 queries x modes): **0**
