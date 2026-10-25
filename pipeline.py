"""Glue: analyze a ticker, then investigate a detected contradiction.

The evidence set is frozen at analysis time, so the explanations and the checker
always work from exactly the evidence shown on screen. Every result is saved to
snapshots/ so a demo can be replayed if the venue Wi-Fi fails.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from . import checker, detector, gemma, market, news

SNAPSHOTS = Path(__file__).resolve().parent.parent / "snapshots"
_cache: dict[str, dict] = {}


def _snap_path(ticker: str) -> Path:
    return SNAPSHOTS / f"{ticker.upper().replace('^', '')}.json"


def save_snapshot(ticker: str, state: dict) -> None:
    SNAPSHOTS.mkdir(exist_ok=True)
    _snap_path(ticker).write_text(json.dumps(state, indent=1, default=str), encoding="utf-8")


def load_snapshot(ticker: str) -> dict | None:
    p = _snap_path(ticker)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def analyze(ticker: str, window: int = 5, client: gemma.GemmaClient | None = None,
            *, df=None, bench=None, headlines=None, name=None) -> dict:
    ticker = ticker.upper().strip()
    g = client or gemma.GemmaClient()
    bench_sym, bench_name = market.benchmark_for(ticker)
    if df is None:
        df = market.fetch_history(ticker)
    if bench is None:
        try:
            bench = market.fetch_history(bench_sym)
        except Exception:
            bench = None
    name = name or market.company_name(ticker)
    if headlines is None:
        headlines = news.collect_headlines(ticker, name)

    price_ev, summary = market.price_evidence(ticker, df, bench, bench_name, window)
    head_ev = news.headline_evidence(headlines)
    labels = gemma.classify_headlines(g, ticker, name, head_ev)
    for e in head_ev:
        e["sentiment"] = labels.get(e["id"], {"sentiment": "neutral", "confidence": 0, "reason": "not labelled"})
    det = detector.detect(summary, labels, len(head_ev))
    png = market.chart_png(ticker, df, summary["window"])

    state = {
        "ticker": ticker, "name": name, "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "series": market.series_for_ui(df), "detector": det,
        "evidence": price_ev + head_ev, "chart_png_b64": base64.b64encode(png).decode(),
        "trace": list(g.trace), "investigation": None,
    }
    _cache[ticker] = state
    save_snapshot(ticker, state)
    return state


def investigate(ticker: str, client: gemma.GemmaClient | None = None) -> dict:
    ticker = ticker.upper().strip()
    state = _cache.get(ticker) or load_snapshot(ticker)
    if not state:
        raise ValueError("Analyze the stock first.")
    g = client or gemma.GemmaClient()
    png = base64.b64decode(state["chart_png_b64"])
    evidence = state["evidence"]
    ev_by_id = {e["id"]: e for e in evidence}
    ev_by_id["C1"] = {"id": "C1", "kind": "chart", "text": "Attached chart image of daily closes and volume."}

    raw = gemma.explain_divergence(g, ticker, state["name"], state["summary"], state["detector"], evidence, png)
    hyps = []
    for i, h in enumerate(raw.get("hypotheses") or []):
        h["id"] = chr(ord("A") + i)  # normalise IDs so claim IDs are unique
        hyps.append(h)

    claims = checker.prepare_claims(hyps, ev_by_id)
    try:
        verdicts = gemma.verify_claims(g, claims, ev_by_id, png)
        judge_ok = True
    except Exception as e:  # deterministic layer still runs
        verdicts, judge_ok = {}, False
        g.trace.append({"purpose": "Independent claim verification", "error": str(e)[:200]})
    graded = checker.grade(hyps, claims, verdicts)

    state["investigation"] = {
        "investigated_at": datetime.now(timezone.utc).isoformat(),
        "chart_reading": raw.get("chart_reading", ""), "hypotheses": graded, "judge_ok": judge_ok,
        "flagged": sum(1 for h in graded if h["flagged"]), "trace": list(g.trace),
    }
    _cache[ticker] = state
    save_snapshot(ticker, state)
    return state
