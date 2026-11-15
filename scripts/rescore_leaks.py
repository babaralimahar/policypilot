"""Re-score saved answer reports with the current leak detector. No model calls.

uv run python scripts/rescore_leaks.py evals/reports/policypilot-answers-test-*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from policypilot.evals.metrics import leaks


def main(paths: list[str]) -> int:
    for p in paths:
        report = json.loads(Path(p).read_text(encoding="utf-8"))
        split = report["split"]
        items = {
            json.loads(line)["id"]: json.loads(line)["expected"]
            for line in Path(f"evals/datasets/policy/{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        before = sum(1 for r in report["rows"] if r["leak"])
        flagged = [
            r["id"]
            for r in report["rows"]
            if leaks(r["answer"], items[r["id"]]["restricted"], items[r["id"]]["question"])
        ]
        out = {
            "report": Path(p).name,
            "split": split,
            "detector": "leaks v2 (text)",
            "flagged_v1": before,
            "flagged_v2": len(flagged),
            "flagged_v2_ids": flagged,
        }
        Path(p).with_suffix(".rescored.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8"
        )
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
