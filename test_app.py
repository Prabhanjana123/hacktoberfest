"""Offline tests for the server, credentials, error handling, scoring and the skill CLI.

These use mocks. They prove the code paths work; they do NOT prove the live Gemini API works.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import app as server  # noqa: E402
from stockpulse import checker, gemma, market, pipeline  # noqa: E402
from test_pipeline import HEADLINES, FakeGemma, _prices  # noqa: E402

CLI = ROOT / "skills" / "evidence-check" / "scripts" / "check_claims.py"


@pytest.fixture
def client():
    return TestClient(server.app)


def test_index_and_static_files_load(client):
    assert client.get("/").status_code == 200
    for f in ("app.js", "style.css", "index.html"):
        assert client.get(f"/static/{f}").status_code == 200


def test_health_reports_models_without_leaking_key(client, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-value")
    r = client.get("/api/health").json()
    assert r["gemini_key"] is True
    assert r["deep_model"] == "gemma-4-31b-it" and r["fast_model"] == "gemma-4-26b-a4b-it"
    assert "secret-test-value" not in json.dumps(r)


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemma.GemmaClient()


def test_analyze_error_returns_502_with_reason(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Set GEMINI_API_KEY")
    monkeypatch.setattr(pipeline, "analyze", boom)
    r = client.get("/api/analyze/NVDA")
    assert r.status_code == 502 and "GEMINI_API_KEY" in r.json()["detail"]


def test_replay_without_snapshot_is_404(client, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "SNAPSHOTS", tmp_path)
    assert client.get("/api/analyze/NOPE?replay=true").status_code == 404


def test_bad_ticker_price_error(monkeypatch):
    class Empty:
        def history(self, **k):
            return pd.DataFrame()
    monkeypatch.setattr(market.yf, "Ticker", lambda t: Empty())
    with pytest.raises(ValueError, match="No price data"):
        market.fetch_history("ZZZZ")


def test_empty_news_does_not_trigger_detector(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SNAPSHOTS", tmp_path)
    s = pipeline.analyze("ACME", 5, FakeGemma(), df=_prices(-0.08), bench=_prices(0), headlines=[], name="Acme")
    assert not s["detector"]["active"] and s["detector"]["total"] == 0


def test_judge_failure_falls_back_to_deterministic_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "SNAPSHOTS", tmp_path)

    class JudgeDown(FakeGemma):
        def generate_json(self, model, prompt, **kw):
            if "CLAIM" in prompt:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return super().generate_json(model, prompt, **kw)

    g = JudgeDown()
    pipeline.analyze("ACME", 5, g, df=_prices(-0.08), bench=_prices(0), headlines=HEADLINES, name="Acme")
    inv = pipeline.investigate("ACME", g)["investigation"]
    assert inv["judge_ok"] is False
    assert any("error" in t for t in inv["trace"])
    flagged = [h["title"] for h in inv["hypotheses"] if h["flagged"]]
    assert flagged == ["Insider selling"]


def test_status_thresholds():
    ev = {"P1": {"text": "fell -8.0% over 5 days"}, "H1": {"text": 'Headline: "Acme beats estimates"'}}
    hyps = [{"id": "A", "claims": [{"text": "Fell 8.0%.", "evidence_ids": ["P1"], "quote": "fell -8.0%"},
                                   {"text": "Beat.", "evidence_ids": ["H1"], "quote": "Acme beats estimates"}]},
            {"id": "B", "claims": [{"text": "Fell 12.5%.", "evidence_ids": ["P1"], "quote": "fell -8.0%"}]}]
    claims = checker.prepare_claims(hyps, ev)
    graded = {h["id"]: h for h in checker.grade(hyps, claims, {
        "A1": {"verdict": "SUPPORTED", "rationale": ""}, "A2": {"verdict": "SUPPORTED", "rationale": ""},
        "B1": {"verdict": "SUPPORTED", "rationale": ""}})}
    assert graded["A"]["status"] == "supported"
    assert "number_not_in_evidence" in graded["B"]["claims"][0]["issues"]
    # 12.5% appears nowhere in the evidence: documented rule halves the claim score (1.0 -> 0.5)
    assert graded["B"]["support_score"] == 0.5 and graded["B"]["status"] == "partial"


def _run_cli(path):
    return subprocess.run([sys.executable, str(CLI), str(path)], capture_output=True, text=True)


def test_skill_cli_valid_and_invalid(tmp_path):
    good = tmp_path / "in.json"
    good.write_text(json.dumps({
        "evidence": [{"id": "H1", "text": 'Headline: "Acme beats earnings estimates"'}],
        "hypotheses": [{"id": "A", "title": "ok", "claims": [
            {"text": "Beat.", "evidence_ids": ["H1"], "quote": "beats earnings estimates"}]},
            {"id": "B", "title": "bad", "claims": [
            {"text": "Fraud.", "evidence_ids": ["H7"], "quote": "fraud"}]}]}))
    r = _run_cli(good)
    assert r.returncode == 0, r.stderr
    out = {h["id"]: h for h in json.loads(r.stdout)}
    assert out["A"]["passed_deterministic"] is True
    assert "fabricated_citation" in out["B"]["claims"][0]["issues"]

    assert _run_cli(tmp_path / "missing.json").returncode != 0
