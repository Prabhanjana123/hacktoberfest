import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stockpulse import quantproof  # noqa: E402


def make_csv(rows=180, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=rows)
    rets = rng.normal(0.0004, 0.012, rows)
    close = 100 * np.cumprod(1 + rets)
    open_ = close * (1 + rng.normal(0, 0.002, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, rows))
    vol = rng.integers(100_000, 1_000_000, rows)
    return pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"), "open": open_, "high": high,
        "low": low, "close": close, "volume": vol,
    }).to_csv(index=False)


class FakeGemma:
    def __init__(self):
        self.trace = []

    def generate_json(self, model, prompt, **kwargs):
        purpose = kwargs.get("purpose", "")
        self.trace.append({
            "purpose": purpose, "model": model,
            "thinking_level": kwargs.get("thinking", "minimal"),
            "image_attached": bool(kwargs.get("image_png")),
            "latency_s": 0.01, "input_tokens": 100, "output_tokens": 50,
        })
        if "hypothesis-to-strategy mapping" in purpose:
            return {
                "supported": True, "strategy_type": "momentum", "strategy_title": "Short-horizon momentum",
                "lookback": 10, "threshold_bps": 50,
                "hypothesis_interpretation": "Hold long when the trailing 10-bar return exceeds 50 bps.",
                "assumptions": ["Long-only, otherwise cash."],
                "limitations": ["Only one fixed split is tested."],
            }
        if "out-of-sample and robustness interpretation" in purpose:
            return {
                "headline": "Out-of-sample results require caution.",
                "assessment": "This synthetic example is a software test, not market evidence.",
                "key_findings": [{"title": "Generalisation", "detail": "Compare in-sample and out-of-sample metrics.", "metric_keys": ["strategy.out_of_sample.total_return"]}],
                "caveats": ["A single split is not sufficient to establish robustness."],
                "next_experiments": ["Use multiple chronological folds."],
                "not_financial_advice": True,
            }
        raise AssertionError(f"Unexpected Gemma call purpose: {purpose}")


def test_parse_ohlcv_csv_returns_normalized_frame_and_quality():
    frame, quality = quantproof.parse_ohlcv_csv(make_csv())
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.is_monotonic_increasing
    assert quality["rows_used"] == 180
    assert quality["frequency_estimate"].startswith("daily")


def test_parse_ohlcv_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing"):
        quantproof.parse_ohlcv_csv("date,close\n2024-01-01,100\n")


def test_parse_ohlcv_rejects_short_sample():
    with pytest.raises(ValueError, match="at least 80"):
        quantproof.parse_ohlcv_csv(make_csv(rows=50))


def test_spec_is_constrained_and_rejects_unknown_template():
    valid = quantproof.validate_strategy_spec({"strategy_type": "momentum", "lookback": 10, "threshold_bps": 50})
    assert valid["supported"] is True
    assert valid["lookback"] == 10
    with pytest.raises(ValueError, match="Unsupported strategy"):
        quantproof.validate_strategy_spec({"strategy_type": "arbitrary_python", "lookback": 10, "threshold_bps": 50})


def test_signals_are_long_or_cash_and_breakout_uses_prior_highs():
    frame, _ = quantproof.parse_ohlcv_csv(make_csv())
    for strategy in ("momentum", "mean_reversion", "breakout"):
        spec = {"strategy_type": strategy, "lookback": 10, "threshold_bps": 50}
        signals = quantproof.generate_signals(frame, spec)
        assert set(signals.unique()).issubset({0.0, 1.0})


def test_run_research_returns_oos_cost_and_parameter_checks():
    fake = FakeGemma()
    result = quantproof.run_research(
        hypothesis="Test whether short-horizon positive momentum persists after costs.",
        csv_text=make_csv(), filename="sample.csv", cost_bps=10, train_fraction=0.7, client=fake,
    )
    assert result["supported"] is True
    assert result["strategy_spec"]["strategy_type"] == "momentum"
    assert result["metrics"]["strategy"]["in_sample"]["observations"] > 0
    assert result["metrics"]["strategy"]["out_of_sample"]["observations"] >= 20
    assert len(result["parameter_sensitivity"]) >= 3
    assert result["metrics"]["double_costs_out_of_sample"]
    assert result["metrics"]["market_regime_analysis"]["status"] in {"computed", "inconclusive"}
    assert result["equity_curve"]
    assert len(fake.trace) == 2
    assert fake.trace[1]["image_attached"] is True


def test_unsupported_hypothesis_does_not_run_backtest():
    class UnsupportedGemma(FakeGemma):
        def generate_json(self, model, prompt, **kwargs):
            self.trace.append({"purpose": kwargs.get("purpose"), "model": model, "image_attached": bool(kwargs.get("image_png"))})
            return {"supported": False, "strategy_type": "unsupported", "reason": "Requires order-book data."}
    result = quantproof.run_research(
        hypothesis="Test whether order-book imbalance predicts the next tick.",
        csv_text=make_csv(), filename="sample.csv", client=UnsupportedGemma(),
    )
    assert result["supported"] is False
    assert "order-book" in result["reason"]
    assert "metrics" not in result


class BrokenGemma:
    """Simulate a persistent provider-side 500 for every Gemma request."""
    def __init__(self):
        self.trace = []

    def generate_json(self, model, prompt, **kwargs):
        raise RuntimeError("ServerError: 500 INTERNAL. Internal error encountered.")


def test_explicit_hypothesis_fallback_completes_backtest_when_gemma_500s():
    hypothesis = (
        "On BEL 5-minute OHLCV data, test whether a long-only momentum strategy that holds "
        "the stock when its trailing 10-bar close-to-close return exceeds +1.0% can outperform "
        "buy-and-hold after transaction costs. Use a 10-bar lookback and a 100 basis-point "
        "threshold. Execute signals on the next bar, use a chronological 70% in-sample / "
        "30% out-of-sample split."
    )
    result = quantproof.run_research(
        hypothesis=hypothesis,
        csv_text=make_csv(),
        filename="sample.csv",
        cost_bps=10,
        train_fraction=0.7,
        client=BrokenGemma(),
    )
    assert result["supported"] is True
    assert result["strategy_spec"]["strategy_type"] == "momentum"
    assert result["strategy_spec"]["lookback"] == 10
    assert result["strategy_spec"]["threshold_bps"] == 100
    assert result["metrics"]["strategy"]["out_of_sample"]["observations"] > 0
    assert result["ai_status"]["mapping_source"] == "explicit_hypothesis_fallback"
    assert result["ai_status"]["report_source"] == "deterministic_fallback"
    assert result["report"]["review_source"] == "deterministic_fallback"
    assert "Gemma" in result["report"]["headline"]
    assert any("mapping failed" in w.lower() for w in result["ai_status"]["warnings"])


def test_explicit_fallback_refuses_ambiguous_hypothesis():
    assert quantproof._infer_explicit_strategy_spec(
        "Test whether price action can predict returns."
    ) is None
    assert quantproof._infer_explicit_strategy_spec(
        "Test momentum over 10 bars with a 100 bps threshold and breakout over 20 bars."
    ) is None
