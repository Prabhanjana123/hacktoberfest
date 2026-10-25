"""Evidence checker: grade every Gemma explanation against the evidence it cites.

Two independent layers:
  1. Deterministic checks (from the `evidence-check` Agent Skill in skills/evidence-check):
     citations exist, quotes appear verbatim, numbers match.
  2. A separate Gemma 4 judge that sees only each claim plus its cited evidence.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SKILL = Path(__file__).resolve().parent.parent / "skills" / "evidence-check" / "scripts" / "evidence_checks.py"
_spec = importlib.util.spec_from_file_location("evidence_checks", _SKILL)
evidence_checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evidence_checks)

VERDICT_SCORE = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "UNSUPPORTED": 0.0, "CONTRADICTED": -0.5}
SUPPORTED_MIN, PARTIAL_MIN, MIN_DISTINCT = 0.65, 0.40, 2


def prepare_claims(hypotheses: list[dict], evidence_by_id: dict[str, dict]) -> list[dict]:
    flat = []
    for h in hypotheses:
        for j, c in enumerate(h.get("claims") or [], 1):
            det = evidence_checks.check_claim(c, evidence_by_id)
            flat.append({"cid": f"{h['id']}{j}", "hid": h["id"], "text": c.get("text", ""),
                         "quote": c.get("quote", ""), **det})
    return flat


def grade(hypotheses: list[dict], claims: list[dict], verdicts: dict[str, dict] | None) -> list[dict]:
    by_h: dict[str, list[dict]] = {}
    for c in claims:
        v = (verdicts or {}).get(c["cid"])
        verdict = v["verdict"] if v else None
        score = VERDICT_SCORE[verdict] if verdict else (0.5 if not c["issues"] else 0.0)
        if "fabricated_citation" in c["issues"] or "no_citation" in c["issues"]:
            score = 0.0
        elif {"quote_not_found", "missing_quote", "number_not_in_evidence"} & set(c["issues"]):
            score *= 0.5
        reasons = [evidence_checks.ISSUE_LABELS[i] for i in c["issues"]]
        if verdict in {"UNSUPPORTED", "CONTRADICTED", "PARTIAL"} and v.get("rationale"):
            reasons.append(f"Judge: {v['rationale']}")
        by_h.setdefault(c["hid"], []).append({
            **c, "verdict": verdict or "NOT_JUDGED", "judge_rationale": (v or {}).get("rationale", ""),
            "score": score, "reasons": reasons,
        })

    graded = []
    for h in hypotheses:
        cs = by_h.get(h["id"], [])
        score = sum(c["score"] for c in cs) / len(cs) if cs else 0.0
        support_ids = sorted({i for c in cs if c["score"] >= 0.5 for i in c["valid_ids"]})
        fabricated = any("fabricated_citation" in c["issues"] for c in cs)
        contradicted = any(c["verdict"] == "CONTRADICTED" for c in cs)

        if score >= SUPPORTED_MIN and len(support_ids) >= MIN_DISTINCT and not (fabricated or contradicted):
            status = "supported"
        elif score >= PARTIAL_MIN and not (fabricated or contradicted):
            status = "partial"
        else:
            status = "insufficient"

        why = []
        if status == "insufficient":
            if fabricated:
                why.append("cites evidence that does not exist")
            if contradicted:
                why.append("a claim is contradicted by its own evidence")
            if len(support_ids) < MIN_DISTINCT:
                why.append(f"only {len(support_ids)} evidence item(s) actually support it")
            weak = sum(1 for c in cs if c["score"] < 0.5)
            if weak:
                why.append(f"{weak} of {len(cs)} claims are not backed by what they cite")
        elif status == "partial" and len(support_ids) < MIN_DISTINCT:
            why.append(f"relies on {len(support_ids)} supporting item(s)")

        graded.append({**h, "claims": cs, "support_score": round(score, 2), "status": status,
                       "flagged": status == "insufficient", "supporting_ids": support_ids,
                       "flag_reasons": why})
    order = {"supported": 0, "partial": 1, "insufficient": 2}
    return sorted(graded, key=lambda x: (order[x["status"]], -x["support_score"]))
