"""Deterministic evidence checks (standard library only).

Used by StockPulse's evidence checker and by the `evidence-check` Agent Skill.
Given a claim, the evidence IDs it cites, an optional verbatim quote and the
evidence corpus, these functions answer questions a model should not be
trusted to answer about itself:

* Do the cited evidence IDs actually exist?
* Does the quoted text actually appear in the cited evidence?
* Do the percentages in the claim actually appear in the cited evidence?
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

QUOTE_MATCH_THRESHOLD = 0.85
NUMBER_TOLERANCE = 0.15  # percentage points

# Evidence IDs whose content is not text (e.g. the chart image). A quote is
# not required when a claim cites only these.
NON_TEXT_IDS = {"C1"}


def normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^\w%.\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_match(quote: str, text: str) -> float:
    """Return 0..1: how well `quote` appears verbatim inside `text`."""
    q, t = normalize(quote), normalize(text)
    if not q or not t:
        return 0.0
    if q in t:
        return 1.0
    n = len(q)
    if n > len(t):
        return SequenceMatcher(None, q, t).ratio()
    best = 0.0
    step = max(1, n // 5)
    for i in range(0, len(t) - n + 1, step):
        best = max(best, SequenceMatcher(None, q, t[i : i + n]).ratio())
        if best >= 0.99:
            break
    return round(best, 3)


_PCT = re.compile(r"[-+]?\d+(?:\.\d+)?\s?%")


def percentages(text: str) -> list[float]:
    return [abs(float(m.replace("%", "").replace(" ", ""))) for m in _PCT.findall(text or "")]


def check_claim(claim: dict, evidence_by_id: dict[str, dict]) -> dict:
    """Run every deterministic check on one claim.

    claim: {"text": str, "evidence_ids": [str], "quote": str}
    evidence_by_id: {"H1": {"text": ...}, ...}
    """
    cited = [str(e).strip().upper() for e in (claim.get("evidence_ids") or [])]
    valid = [e for e in cited if e in evidence_by_id]
    invalid = [e for e in cited if e not in evidence_by_id]
    issues: list[str] = []

    if not cited:
        issues.append("no_citation")
    if invalid:
        issues.append("fabricated_citation")

    text_ids = [e for e in valid if e not in NON_TEXT_IDS]
    cited_text = " ".join(evidence_by_id[e].get("text", "") for e in text_ids)

    quote = (claim.get("quote") or "").strip()
    q_score = None
    if text_ids:
        if not quote:
            issues.append("missing_quote")
        else:
            q_score = max(quote_match(quote, evidence_by_id[e].get("text", "")) for e in text_ids)
            if q_score < QUOTE_MATCH_THRESHOLD:
                issues.append("quote_not_found")

    claimed_pcts = percentages(claim.get("text", ""))
    evidence_pcts = percentages(cited_text)
    unmatched = [
        p for p in claimed_pcts if not any(abs(p - e) <= NUMBER_TOLERANCE for e in evidence_pcts)
    ]
    if claimed_pcts and text_ids and unmatched:
        issues.append("number_not_in_evidence")

    return {
        "cited": cited,
        "valid_ids": valid,
        "invalid_ids": invalid,
        "quote_score": q_score,
        "unmatched_numbers": unmatched,
        "issues": issues,
    }


ISSUE_LABELS = {
    "no_citation": "Cites no evidence",
    "fabricated_citation": "Cites evidence that does not exist",
    "missing_quote": "No supporting quote given",
    "quote_not_found": "Quote does not appear in the cited evidence",
    "number_not_in_evidence": "Uses a number the cited evidence does not contain",
}
