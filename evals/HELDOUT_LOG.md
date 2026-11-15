# Held-out test set runs

Every run of a frozen test split is recorded here. More than one row for the same
suite and dataset hash means the set was re-run (forced) and must be disclosed.

| UTC time | suite | split | dataset sha256 | git | forced |
|---|---|---|---|---|---|
| 2026-11-15T15:32:22+00:00 | policypilot-answers | test | 807f306aeeddba166b68d188f5d432f11f0e297220e0517b3348aa1371f10d88 | nogit | no |
| 2026-11-15T15:33:07+00:00 | policypilot-retrieval | test | 807f306aeeddba166b68d188f5d432f11f0e297220e0517b3348aa1371f10d88 | nogit | no |

Results: `evals/reports/policypilot-answers-test-20261115-153222-nogit.{json,md}` and
`evals/reports/policypilot-retrieval-test-20261115-153307-nogit.{json,md}`. The git field
says `nogit` because the repository had no commits yet; the code is the v1.0.0 commit.

## Disclosure: leak detector revised after the held-out run (2026-11-15)

The answers run reported **2 permission leaks** with leak detector v1 ("all numbers of the
restricted fact appear in the answer"). Manual review showed both were false positives:

1. `p-f1-0`, emp_us, "How many vacation days do I get each year?": the answer gave the **US**
   policy ("15 days ... 20 days after five years") from a document the persona may see. The
   restricted **PK** fact "20 working days of annual leave" shares only the number 20.
2. `p-f88-0`, mgr_us, "Who signs off on payments over 50,000 dollars?": the answer said the
   visible policies do not cover it and cited the expenses policy (visible to the persona);
   "50,000" was repeated from the question.

Independent evidence: 0 structural leaks (no retrieved chunk outside the asker's groups) in the
held-out retrieval run, and both citations point to permitted documents.

Detector v2 ignores numbers present in the question and requires a distinctive word of the
restricted fact for a numeric match (a close phrase match always counts). The **saved** answers
were re-scored offline with no new model calls (`scripts/rescore_leaks.py`): held-out 2 -> 0,
dev unchanged at 0. Output: `*.rescored.json` next to each report. The held-out set was **not**
re-run. Both numbers are reported in the README.
