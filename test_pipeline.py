"""Offline end-to-end test: synthetic prices + fake Gemma. Run: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stockpulse import checker, pipeline  # noqa: E402


def _prices(drop: float) -> pd.DataFrame:
    idx = pd.bdate_range(end="2026-10-23", periods=250)
    close = np.linspace(180, 225, 250)
    close[-5:] = close[-6] * (1 + np.linspace(drop / 5, drop, 5))
    vol = np.full(250, 1_000_000.0)
    vol[-3] = 2_600_000
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": vol}, index=idx)


HEADLINES = [
    {"title": "Acme beats earnings estimates, raises full-year guidance", "summary": "", "publisher": "Reuters",
     "published": "2026-10-21T10:00:00+00:00", "url": "", "source": "test"},
    {"title": "Analysts upgrade Acme to Buy after record quarter", "summary": "", "publisher": "CNBC",
     "published": "2026-10-22T10:00:00+00:00", "url": "", "source": "test"},
    {"title": "Acme launches new AI product line", "summary": "", "publisher": "Bloomberg",
     "published": "2026-10-22T12:00:00+00:00", "url": "", "source": "test"},
    {"title": "Acme faces regulatory probe in Europe", "summary": "", "publisher": "FT",
     "published": "2026-10-23T09:00:00+00:00", "url": "", "source": "test"},
]


class FakeGemma:
    def __init__(self):
        self.trace = []

    def generate_json(self, model, prompt, **kw):
        self.trace.append({"purpose": kw.get("purpose"), "model": model, "image_attached": bool(kw.get("image_png"))})
        if "Label each headline" in prompt:
            return {"labels": [{"id": "H1", "sentiment": "positive"}, {"id": "H2", "sentiment": "positive"},
                               {"id": "H3", "sentiment": "positive"}, {"id": "H4", "sentiment": "negative"}]}
        if "COMPETING explanations" in prompt:
            return {"chart_reading": "Steady climb, then a sharp drop in the shaded window on high volume.",
                    "hypotheses": [
                        {"title": "Good news priced in", "claims": [
                            {"text": "The stock had already rallied before the window.", "evidence_ids": ["P5"],
                             "quote": "In the 20 trading days before the window"},
                            {"text": "Earnings beat was the headline catalyst.", "evidence_ids": ["H1"],
                             "quote": "Acme beats earnings estimates"}]},
                        {"title": "Regulatory fear", "claims": [
                            {"text": "A probe weighed on the shares.", "evidence_ids": ["H4"],
                             "quote": "faces regulatory probe in Europe"},
                            {"text": "Selling happened on heavy volume.", "evidence_ids": ["P3"],
                             "quote": "the average daily volume of the previous 20 sessions"}]},
                        {"title": "Insider selling", "claims": [
                            {"text": "Executives sold 12% of their stake.", "evidence_ids": ["H9"],
                             "quote": "insiders dumped shares"},
                            {"text": "The CEO hinted at weaker demand.", "evidence_ids": ["H2"],
                             "quote": "CEO warns of slowing demand"}]},
                    ]}
        if "CLAIM" in prompt:
            return {"verdicts": [
                {"claim_id": "A1", "verdict": "SUPPORTED", "rationale": "Prior run-up shown"},
                {"claim_id": "A2", "verdict": "PARTIAL", "rationale": "Beat shown, causality not"},
                {"claim_id": "B1", "verdict": "PARTIAL", "rationale": "Probe exists, impact unknown"},
                {"claim_id": "B2", "verdict": "SUPPORTED", "rationale": "Volume spike shown"},
                {"claim_id": "C1", "verdict": "UNSUPPORTED", "rationale": "No such evidence"},
                {"claim_id": "C2", "verdict": "UNSUPPORTED", "rationale": "H2 is about an upgrade"}]}
        raise AssertionError("unexpected prompt")


def test_full_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SNAPSHOTS", tmp_path)
    g = FakeGemma()
    state = pipeline.analyze("ACME", 5, g, df=_prices(-0.08), bench=_prices(-0.005), headlines=HEADLINES,
                             name="Acme Corp")
    det = state["detector"]
    assert det["active"], det
    assert det["positive"] == 3 and det["negative"] == 1
    ids = [e["id"] for e in state["evidence"]]
    assert ids[:6] == ["P1", "P2", "P3", "P4", "P5", "P6"] and "H4" in ids

    state = pipeline.investigate("ACME", g)
    inv = state["investigation"]
    by_title = {h["title"]: h for h in inv["hypotheses"]}
    assert by_title["Insider selling"]["flagged"]
    assert "fabricated_citation" in by_title["Insider selling"]["claims"][0]["issues"]
    assert "quote_not_found" in by_title["Insider selling"]["claims"][1]["issues"]
    assert not by_title["Regulatory fear"]["flagged"]
    assert inv["flagged"] == 1
    assert any(t["image_attached"] for t in g.trace), "Gemma must receive the chart image"
    assert pipeline.load_snapshot("ACME")["investigation"]


def test_no_contradiction_when_news_negative(tmp_path, monkeypatch):
    from stockpulse import detector
    d = detector.detect({"window_return": -0.06, "window": 5},
                        {"H1": {"sentiment": "negative"}, "H2": {"sentiment": "negative"},
                         "H3": {"sentiment": "positive"}}, 3)
    assert not d["active"]


def test_quote_matching():
    ec = checker.evidence_checks
    assert ec.quote_match("beats earnings estimates", 'Headline: "Acme beats earnings estimates, raises"') == 1.0
    assert ec.quote_match("CEO warns of slowing demand", "Analysts upgrade Acme to Buy") < 0.85
