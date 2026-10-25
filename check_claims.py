"""CLI for the evidence-check skill: run deterministic checks on a JSON file.

Usage: python check_claims.py input.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_checks import ISSUE_LABELS, check_claim  # noqa: E402


def main(path: str) -> int:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    evidence = {e["id"].upper(): e for e in data.get("evidence", [])}
    report = []
    for h in data.get("hypotheses", []):
        claims = []
        for c in h.get("claims", []):
            r = check_claim(c, evidence)
            r["text"] = c.get("text", "")
            r["reasons"] = [ISSUE_LABELS[i] for i in r["issues"]]
            claims.append(r)
        report.append({"id": h.get("id"), "title": h.get("title"), "claims": claims,
                       "passed_deterministic": all(not c["issues"] for c in claims)})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
