"""Contradiction detector: price direction vs. headline tone. Pure, deterministic rules."""
from __future__ import annotations

import math

DROP_THRESHOLD = -0.03      # price fell at least 3% over the window
POSITIVE_SHARE = 0.50       # at least half of headlines read positive
MIN_HEADLINES = 3


def detect(summary: dict, labels: dict[str, dict], n_headlines: int) -> dict:
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for lab in labels.values():
        counts[lab["sentiment"]] += 1
    total = max(n_headlines, sum(counts.values()))
    pos_share = counts["positive"] / total if total else 0.0
    net = (counts["positive"] - counts["negative"]) / total if total else 0.0
    ret = summary["window_return"]

    checks = {
        "price_fell": ret <= DROP_THRESHOLD,
        "mostly_positive": pos_share >= POSITIVE_SHARE and counts["positive"] > counts["negative"],
        "enough_headlines": total >= MIN_HEADLINES,
    }
    active = all(checks.values())

    price_part = min(1.0, max(0.0, -ret) / 0.10)
    sent_part = max(0.0, net)
    strength = round(100 * math.sqrt(price_part * sent_part)) if active else 0

    if active:
        msg = (f"Price is {ret * 100:+.1f}% over {summary['window']} sessions while "
               f"{counts['positive']} of {total} headlines are positive.")
    elif not checks["price_fell"]:
        msg = f"No contradiction: price moved {ret * 100:+.1f}% (needs a drop of {DROP_THRESHOLD * 100:.0f}% or more)."
    elif not checks["enough_headlines"]:
        msg = f"Not enough headlines to judge ({total} found, need {MIN_HEADLINES})."
    else:
        msg = f"No contradiction: only {pos_share:.0%} of headlines are positive, so the news agrees with the drop."

    return {"active": active, "strength": strength, "message": msg, "checks": checks,
            "total": total, "positive_share": pos_share, "net_sentiment": net, **counts,
            "thresholds": {"drop": DROP_THRESHOLD, "positive_share": POSITIVE_SHARE, "min_headlines": MIN_HEADLINES}}
