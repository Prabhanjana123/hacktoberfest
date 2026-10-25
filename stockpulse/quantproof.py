"""QuantProof: constrained strategy research with deterministic backtesting.

Gemma maps a natural-language hypothesis to one of a few supported strategy
families and explains computed results. It never writes or executes Python and it
never computes the backtest metrics. All reported numbers come from this module.

MVP limitations: long-only, daily or regular intraday OHLCV candles, close-based
signals executed on the next bar, simple proportional transaction costs, and a
single chronological train/test split. This is a research prototype, not a
production-grade trading simulator.
"""
from __future__ import annotations

import json
import math
import re
from io import StringIO, BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUPPORTED_STRATEGIES = {"momentum", "mean_reversion", "breakout"}
MIN_ROWS = 80
MAX_CSV_CHARS = 3_500_000


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def parse_ohlcv_csv(csv_text: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse and validate user-supplied OHLCV CSV; no data is silently invented."""
    if not csv_text or not csv_text.strip():
        raise ValueError("Upload a non-empty OHLCV CSV file.")
    if len(csv_text) > MAX_CSV_CHARS:
        raise ValueError("CSV is too large for this prototype (maximum about 3.5 MB of text).")

    try:
        raw = pd.read_csv(StringIO(csv_text))
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if raw.empty:
        raise ValueError("CSV contains no data rows.")

    normalized = {_normalize_column(c): c for c in raw.columns}
    # Accept common aliases while retaining explicit OHLCV semantics.
    aliases = {
        "timestamp": ("timestamp", "datetime", "date", "time", "dateandtime", "unnamed0"),
        "open": ("open", "openingprice", "o"),
        "high": ("high", "highprice", "h"),
        "low": ("low", "lowprice", "l"),
        "close": ("close", "closingprice", "adjclose", "adjustedclose", "c"),
        "volume": ("volume", "vol", "v"),
    }
    found: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in normalized:
                found[canonical] = normalized[candidate]
                break

    missing = [k for k in ("timestamp", "open", "high", "low", "close", "volume") if k not in found]
    if missing:
        raise ValueError(
            "CSV must contain date/time and OHLCV columns. Missing: " + ", ".join(missing) +
            ". Expected headers such as timestamp, open, high, low, close, volume."
        )

    frame = pd.DataFrame({k: raw[v] for k, v in found.items()})
    # Parse timestamps while preserving the displayed local wall-clock date when possible.
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce")
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
    except (AttributeError, TypeError, ValueError):
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
    frame["timestamp"] = parsed

    raw_rows = len(frame)
    numeric_columns = ["open", "high", "low", "close", "volume"]
    for col in numeric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    before_drop = len(frame)
    frame = frame.dropna(subset=["timestamp", *numeric_columns]).copy()
    dropped_invalid = before_drop - len(frame)
    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    dropped_duplicates = before_drop - dropped_invalid - len(frame)

    if frame.empty:
        raise ValueError("No valid rows remained after parsing dates and OHLCV values.")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive.")
    if (frame["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    bad_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) |
        (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    if bad_ohlc.any():
        raise ValueError(f"{int(bad_ohlc.sum())} rows have inconsistent OHLC values (high/low do not enclose open/close).")
    frame = frame.set_index("timestamp")
    frame.index.name = "timestamp"

    if len(frame) < MIN_ROWS:
        raise ValueError(
            f"Need at least {MIN_ROWS} valid OHLCV bars to make a useful chronological train/test split; got {len(frame)}."
        )

    bars_per_year, frequency = _infer_annualization(frame.index)
    quality = {
        "rows_received": int(raw_rows),
        "rows_used": int(len(frame)),
        "invalid_rows_removed": int(dropped_invalid),
        "duplicate_timestamps_removed": int(max(0, dropped_duplicates)),
        "start_date": frame.index[0].isoformat(),
        "end_date": frame.index[-1].isoformat(),
        "frequency_estimate": frequency,
        "bars_per_year_assumption": bars_per_year,
        "limitations": [
            "The strategy engine assumes decisions made from a bar close are executed on the next bar.",
            "Transaction costs are a simplified proportional estimate and do not model market impact or liquidity.",
            "Results depend on the supplied dataset and are not a prediction of future returns.",
        ],
    }
    return frame, quality


def _infer_annualization(index: pd.DatetimeIndex) -> tuple[int, str]:
    """Use a conservative, transparent bars/year heuristic for annualized metrics."""
    if len(index) < 3:
        return 252, "daily (assumed)"
    diffs = pd.Series(index).diff().dropna().dt.total_seconds() / 60.0
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return 252, "daily (assumed)"
    med = float(diffs.median())
    if med >= 8 * 60:
        return 252, "daily bars (estimated)"
    if med <= 2:
        return 252 * 375, "approximately 1-minute intraday bars"
    if med <= 6:
        return 252 * 75, "approximately 5-minute intraday bars"
    if med <= 20:
        return 252 * 25, "approximately 15-minute intraday bars"
    if med <= 65:
        return 252 * 6, "approximately hourly intraday bars"
    return 252, "frequency uncertain; 252 bars/year assumed"


def validate_strategy_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Validate Gemma's strategy mapping against a fixed, non-executable schema."""
    if not isinstance(spec, dict):
        raise ValueError("Gemma did not return a structured strategy specification.")
    if spec.get("supported") is False or spec.get("strategy_type") == "unsupported":
        return {
            "supported": False,
            "reason": str(spec.get("reason") or "This hypothesis could not be mapped safely to a supported strategy template."),
            "strategy_type": "unsupported",
        }
    kind = str(spec.get("strategy_type", "")).strip().lower()
    if kind not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy template '{kind}'. Allowed: {', '.join(sorted(SUPPORTED_STRATEGIES))}.")
    try:
        lookback = int(spec.get("lookback", 10))
        threshold = float(spec.get("threshold_bps", 100))
    except (TypeError, ValueError) as exc:
        raise ValueError("The strategy lookback and threshold must be numeric.") from exc
    if not 3 <= lookback <= 60:
        raise ValueError("Strategy lookback must be between 3 and 60 bars.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1000:
        raise ValueError("Strategy threshold must be between 0 and 1000 basis points.")
    return {
        "supported": True,
        "strategy_type": kind,
        "lookback": lookback,
        "threshold_bps": round(threshold, 2),
        "hypothesis_interpretation": str(spec.get("hypothesis_interpretation", ""))[:600],
        "assumptions": [str(x)[:200] for x in (spec.get("assumptions") or [])[:6]],
        "limitations": [str(x)[:200] for x in (spec.get("limitations") or [])[:6]],
        "strategy_title": str(spec.get("strategy_title") or kind.replace("_", " ").title())[:100],
    }


def generate_signals(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    """Generate long/cash signals with past information only; never executes AI code."""
    kind = spec["strategy_type"]
    lookback = int(spec["lookback"])
    threshold = float(spec["threshold_bps"]) / 10_000.0
    close = frame["close"]
    if kind == "momentum":
        past_return = close / close.shift(lookback) - 1
        signal = (past_return > threshold).astype(float)
    elif kind == "mean_reversion":
        past_return = close / close.shift(lookback) - 1
        signal = (past_return < -threshold).astype(float)
    elif kind == "breakout":
        # Compare today's close to the highest prior high; shift prevents including today's high.
        prior_high = frame["high"].rolling(lookback, min_periods=lookback).max().shift(1)
        signal = (close > prior_high * (1 + threshold)).astype(float)
    else:  # guarded by validate_strategy_spec
        raise ValueError(f"Unsupported strategy type: {kind}")
    return signal.fillna(0.0).clip(0.0, 1.0)


def _metrics(returns: pd.Series, positions: pd.Series, bars_per_year: int) -> dict[str, Any]:
    returns = pd.Series(returns, index=returns.index, dtype=float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if returns.empty:
        return {"observations": 0, "total_return": 0.0, "annualized_return": None,
                "annualized_volatility": None, "sharpe": None, "max_drawdown": 0.0,
                "exposure_pct": 0.0, "position_changes": 0}
    equity = (1.0 + returns).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    n = len(returns)
    if equity.iloc[-1] <= 0:
        annualized_return = -1.0
    else:
        annualized_log = (bars_per_year / n) * math.log(float(equity.iloc[-1]))
        # Very short intraday samples can make annualisation numerically meaningless.
        annualized_return = math.exp(annualized_log) - 1.0 if annualized_log < 700 else None
    std = float(returns.std(ddof=1)) if n > 1 else 0.0
    ann_vol = std * math.sqrt(bars_per_year) if std > 0 else 0.0
    mean = float(returns.mean())
    sharpe = mean / std * math.sqrt(bars_per_year) if std > 0 else None
    dd = equity / equity.cummax() - 1.0
    p = pd.Series(positions, index=returns.index, dtype=float).fillna(0.0)
    changes = int(p.diff().abs().fillna(p.abs()).gt(0).sum())
    return {
        "observations": int(n),
        "total_return": total,
        "annualized_return": annualized_return,
        "annualized_volatility": float(ann_vol),
        "sharpe": float(sharpe) if sharpe is not None and math.isfinite(sharpe) else None,
        "max_drawdown": float(dd.min()),
        "exposure_pct": float(p.mean() * 100.0),
        "position_changes": changes,
    }


def _evaluate(frame: pd.DataFrame, spec: dict[str, Any], cost_bps: float,
              train_fraction: float, bars_per_year: int) -> dict[str, Any]:
    close = frame["close"]
    market_returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    desired = generate_signals(frame, spec)
    # A signal formed at close t is applied to the next bar's close-to-close return.
    position = desired.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (float(cost_bps) / 10_000.0)
    strategy_returns = (position * market_returns - cost).clip(lower=-0.999999)
    buy_hold_returns = market_returns

    split = int(math.floor(len(frame) * train_fraction))
    split = max(1, min(split, len(frame) - 1))
    train_idx = frame.index[:split]
    test_idx = frame.index[split:]
    strategy = {
        "full": _metrics(strategy_returns, position, bars_per_year),
        "in_sample": _metrics(strategy_returns.loc[train_idx], position.loc[train_idx], bars_per_year),
        "out_of_sample": _metrics(strategy_returns.loc[test_idx], position.loc[test_idx], bars_per_year),
    }
    baseline = {
        "in_sample": _metrics(buy_hold_returns.loc[train_idx], pd.Series(1.0, index=train_idx), bars_per_year),
        "out_of_sample": _metrics(buy_hold_returns.loc[test_idx], pd.Series(1.0, index=test_idx), bars_per_year),
    }
    curve_df = pd.DataFrame({
        "date": [x.isoformat() for x in frame.index[split:]],
        "strategy_equity": (1 + strategy_returns.loc[test_idx]).cumprod().to_numpy(),
        "buy_hold_equity": (1 + buy_hold_returns.loc[test_idx]).cumprod().to_numpy(),
    })
    # Keep JSON responses reasonable for large intraday CSVs.
    if len(curve_df) > 500:
        positions = np.linspace(0, len(curve_df) - 1, 500).astype(int)
        curve_df = curve_df.iloc[positions].reset_index(drop=True)
    curve = [{"date": str(row.date), "strategy_equity": float(row.strategy_equity),
              "buy_hold_equity": float(row.buy_hold_equity)} for row in curve_df.itertuples(index=False)]
    return {
        "strategy": strategy,
        "buy_hold": baseline,
        "split_index": split,
        "split_date": frame.index[split].isoformat(),
        "test_start": frame.index[split].isoformat(),
        "test_end": frame.index[-1].isoformat(),
        "curve": curve,
        "strategy_returns": strategy_returns,
        "position": position,
        "buy_hold_returns": buy_hold_returns,
    }


def _market_regime_analysis(frame: pd.DataFrame, evaluated: dict[str, Any],
                            train_fraction: float, lookback: int) -> dict[str, Any]:
    """Conditional OOS bar-return summary by volatility regime.

    Volatility cutoffs are learned from the training slice only. The regime label
    is shifted one bar so the return being scored is not used to define its own
    regime. This is a descriptive check, not a causal model.
    """
    close_returns = frame["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol_window = max(5, min(20, int(lookback)))
    regime_vol = close_returns.rolling(vol_window, min_periods=vol_window).std().shift(1)
    split = int(math.floor(len(frame) * train_fraction))
    split = max(1, min(split, len(frame) - 1))
    training_vol = regime_vol.iloc[:split].dropna()
    if len(training_vol) < 15:
        return {
            "status": "inconclusive", "method": "Training-only rolling-volatility terciles, lagged by one bar.",
            "reason": "Not enough in-sample volatility observations to define stable regime thresholds.",
            "groups": [],
        }
    low_cut = float(training_vol.quantile(1 / 3))
    high_cut = float(training_vol.quantile(2 / 3))
    oos_idx = frame.index[split:]
    oos_vol = regime_vol.loc[oos_idx]
    strategy_returns = evaluated["strategy_returns"].loc[oos_idx]
    buy_hold_returns = evaluated["buy_hold_returns"].loc[oos_idx]
    position = evaluated["position"].loc[oos_idx]
    groups = []
    for name, mask in [
        ("Low volatility", oos_vol <= low_cut),
        ("Medium volatility", (oos_vol > low_cut) & (oos_vol <= high_cut)),
        ("High volatility", oos_vol > high_cut),
    ]:
        mask = mask.fillna(False)
        n = int(mask.sum())
        if n == 0:
            groups.append({"regime": name, "oos_bars": 0, "status": "no_observations"})
            continue
        strat_slice = strategy_returns.loc[mask]
        bh_slice = buy_hold_returns.loc[mask]
        pos_slice = position.loc[mask]
        groups.append({
            "regime": name,
            "oos_bars": n,
            "mean_strategy_return_bps_per_bar": float(strat_slice.mean() * 10_000),
            "mean_buy_hold_return_bps_per_bar": float(bh_slice.mean() * 10_000),
            "positive_strategy_bar_pct": float((strat_slice > 0).mean() * 100),
            "market_exposure_pct": float(pos_slice.mean() * 100),
            "status": "descriptive_small_sample" if n < 10 else "descriptive",
        })
    return {
        "status": "computed",
        "method": f"OOS bars grouped by prior {vol_window}-bar realised volatility; low/high cutoffs from training-slice terciles only.",
        "low_volatility_cutoff": low_cut,
        "high_volatility_cutoff": high_cut,
        "groups": groups,
        "warning": "Conditional averages do not prove regime predictability or causation. Small regime buckets are especially uncertain.",
    }


def _robustness_assessment(in_sample: dict, out_sample: dict, stressed: dict,
                           sensitivity: list[dict], min_test_bars: int) -> dict[str, Any]:
    is_ret = in_sample.get("total_return", 0.0)
    oos_ret = out_sample.get("total_return", 0.0)
    is_sharpe, oos_sharpe = in_sample.get("sharpe"), out_sample.get("sharpe")
    if min_test_bars < 20:
        performance_status = "inconclusive_small_test_sample"
        performance_text = "The out-of-sample period is too short for a meaningful performance assessment."
    elif is_ret > 0 and oos_ret < 0:
        performance_status = "out_of_sample_loss_after_positive_training"
        performance_text = "The strategy gained in-sample but lost money out-of-sample; this is a material generalisation warning."
    elif is_sharpe is not None and oos_sharpe is not None and is_sharpe > 0 and oos_sharpe < is_sharpe - 0.5:
        performance_status = "out_of_sample_deterioration"
        performance_text = "Risk-adjusted performance deteriorated out-of-sample by more than 0.5 Sharpe units."
    elif oos_ret > 0 and (oos_sharpe is None or oos_sharpe > 0):
        performance_status = "positive_out_of_sample_not_proven_robust"
        performance_text = "Out-of-sample results are positive, but a single split does not prove future robustness."
    else:
        performance_status = "mixed_or_weak"
        performance_text = "The out-of-sample evidence is mixed or weak; avoid treating the strategy as validated."

    base_oos = out_sample.get("total_return", 0.0)
    stressed_oos = stressed.get("total_return", 0.0)
    if stressed_oos < 0 <= base_oos:
        cost_status = "cost_sensitive"
        cost_text = "Doubling transaction costs changes the out-of-sample result from non-negative to negative."
    elif base_oos - stressed_oos > 0.03:
        cost_status = "material_cost_impact"
        cost_text = "Doubling costs reduces out-of-sample total return by more than 3 percentage points."
    else:
        cost_status = "no_large_flip_under_2x_costs"
        cost_text = "Doubling the assumed costs did not trigger the specified large-impact flags; this is not proof costs are realistic."

    returns = [x.get("oos_total_return") for x in sensitivity if x.get("oos_total_return") is not None]
    sharpe_values = [x.get("oos_sharpe") for x in sensitivity if x.get("oos_sharpe") is not None]
    sign_flip = bool(returns and min(returns) < 0 < max(returns))
    spread = (max(sharpe_values) - min(sharpe_values)) if len(sharpe_values) >= 2 else None
    if sign_flip or (spread is not None and spread > 1.0):
        param_status = "parameter_sensitive"
        param_text = "Nearby parameter settings materially change the out-of-sample result; the strategy may be parameter-sensitive."
    else:
        param_status = "no_large_sensitivity_flag"
        param_text = "The tested nearby settings did not trigger the chosen sensitivity flags; this small grid cannot rule out overfitting."

    return {
        "performance": {"status": performance_status, "message": performance_text},
        "transaction_costs": {"status": cost_status, "message": cost_text},
        "parameters": {"status": param_status, "message": param_text},
        "warning": "These diagnostics are heuristic research checks, not statistical proof of edge or absence of overfitting.",
    }


def _equity_chart_png(curve: list[dict[str, Any]]) -> bytes:
    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=110)
    if curve:
        dates = [row["date"] for row in curve]
        strategy = [row["strategy_equity"] for row in curve]
        baseline = [row["buy_hold_equity"] for row in curve]
        ax.plot(dates, strategy, label="QuantProof strategy (after costs)", linewidth=2)
        ax.plot(dates, baseline, label="Buy and hold", linewidth=1.6, alpha=0.85)
        ax.axhline(1.0, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        ax.set_title("Out-of-sample equity curves (start = 1.0)")
        ax.set_ylabel("Equity multiple")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
    else:
        ax.text(0.5, 0.5, "No out-of-sample curve", ha="center", va="center")
    fig.tight_layout()
    buff = BytesIO()
    fig.savefig(buff, format="png")
    plt.close(fig)
    return buff.getvalue()


def _infer_explicit_strategy_spec(hypothesis: str) -> dict[str, Any] | None:
    """Conservative local fallback only when the user explicitly named all parameters.

    This is used only if Gemma's hypothesis-mapping request fails. It deliberately
    refuses vague hypotheses rather than silently inventing a strategy.
    """
    text = (hypothesis or "").lower()
    kinds = []
    if re.search(r"\bmomentum\b", text):
        kinds.append("momentum")
    if re.search(r"\bmean[ -]?reversion\b", text):
        kinds.append("mean_reversion")
    if re.search(r"\bbreakout\b", text):
        kinds.append("breakout")
    if len(kinds) != 1:
        return None

    lookbacks = {int(v) for v in re.findall(r"\b(\d+)\s*[- ]?bars?\b", text)}
    if len(lookbacks) != 1:
        return None
    lookback = next(iter(lookbacks))

    # Only treat values in a threshold context as percentage thresholds. This avoids
    # mistaking the train/test percentages for trading thresholds.
    bps_values = [
        float(v) for v in re.findall(
            r"\b(\d+(?:\.\d+)?)\s*(?:basis[ -]*points?|bps)\b", text
        )
    ]
    pct_values = [
        float(v) * 100.0 for v in re.findall(
            r"(?:exceed(?:s)?|greater than|more than|above|threshold of)"
            r"[^\n]{0,45}?([+-]?\d+(?:\.\d+)?)\s*%", text
        )
    ]
    candidates = bps_values + pct_values
    if not candidates:
        return None
    if max(candidates) - min(candidates) > 1.0:
        # Conflicting explicit thresholds: ask the user to simplify the hypothesis.
        return None
    threshold = candidates[0]
    try:
        return validate_strategy_spec({
            "supported": True,
            "strategy_type": kinds[0],
            "strategy_title": f"Explicit {kinds[0].replace('_', ' ').title()} Specification",
            "lookback": lookback,
            "threshold_bps": threshold,
            "hypothesis_interpretation": (
                "Mapped locally from the user's explicit strategy type, lookback, and threshold "
                "because the Gemma mapping request failed."
            ),
            "assumptions": [
                "This specification was parsed from the user's wording, not generated by Gemma.",
                "Long-only or cash; signals use close data and enter on the next bar.",
            ],
            "limitations": [
                "Gemma hypothesis mapping was unavailable for this run.",
                "A successful backtest is not evidence of future profitability.",
            ],
        })
    except (ValueError, TypeError):
        return None


def _local_quant_report(metrics: dict[str, Any], robustness: dict[str, Any], reason: str) -> dict[str, Any]:
    """Transparent deterministic report used when both Gemma review attempts fail."""
    strategy = metrics["strategy"]
    ins = strategy["in_sample"]
    oos = strategy["out_of_sample"]
    hold = metrics["buy_hold"]["out_of_sample"]
    def pct(value):
        return "unavailable" if value is None else f"{float(value) * 100:+.2f}%"
    assessment = (
        f"The deterministic backtest completed. Strategy return was {pct(ins.get('total_return'))} "
        f"in-sample and {pct(oos.get('total_return'))} out-of-sample; buy-and-hold returned "
        f"{pct(hold.get('total_return'))} over the same out-of-sample interval. "
        f"Python's diagnostic status is '{robustness['performance']['status']}': "
        f"{robustness['performance']['message']}"
    )
    return {
        "headline": "Backtest completed; Gemma's research review was unavailable.",
        "assessment": assessment,
        "key_findings": [
            {"title": "Out-of-sample comparison", "detail": assessment,
             "metric_keys": ["strategy.in_sample.total_return", "strategy.out_of_sample.total_return", "buy_hold.out_of_sample.total_return"]},
            {"title": "Transaction-cost stress", "detail": robustness["transaction_costs"]["message"],
             "metric_keys": ["double_costs_out_of_sample.total_return"]},
            {"title": "Parameter sensitivity", "detail": robustness["parameters"]["message"],
             "metric_keys": ["parameter_sensitivity"]},
        ],
        "caveats": [
            "Gemma's final interpretation could not be generated: " + str(reason)[:220],
            "A single chronological split and a small sample cannot establish generalisable trading performance.",
            "Transaction costs are simplified and do not model market impact or all execution frictions.",
        ],
        "next_experiments": [
            "Repeat the evaluation across multiple chronological walk-forward splits.",
            "Use a longer, independently sourced dataset and verify data quality.",
            "Use realistic commissions, taxes, slippage and liquidity assumptions.",
        ],
        "not_financial_advice": True,
        "review_source": "deterministic_fallback",
    }


def run_research(hypothesis: str, csv_text: str, filename: str, cost_bps: float = 10.0,
                 train_fraction: float = 0.7, client=None) -> dict[str, Any]:
    """Run constrained research, degrading gracefully if the hosted Gemma API errors."""
    hypothesis = (hypothesis or "").strip()
    if len(hypothesis) < 10:
        raise ValueError("Describe the trading hypothesis in at least 10 characters.")
    if len(hypothesis) > 2000:
        raise ValueError("Hypothesis is too long (maximum 2,000 characters).")
    if not math.isfinite(float(cost_bps)) or not 0 <= float(cost_bps) <= 100:
        raise ValueError("Transaction costs must be between 0 and 100 basis points per position change.")
    if not 0.6 <= float(train_fraction) <= 0.85:
        raise ValueError("Training fraction must be between 60% and 85%.")

    from . import gemma
    frame, data_quality = parse_ohlcv_csv(csv_text)
    trace: list[dict[str, Any]] = []
    pipeline_error_trace: list[dict[str, Any]] = []
    ai_status: dict[str, Any] = {
        "mapping_source": "gemma",
        "report_source": "gemma",
        "warnings": [],
    }
    init_error = None
    try:
        g = client if client is not None else gemma.GemmaClient()
    except Exception as exc:
        g = None
        init_error = f"Gemma client unavailable ({type(exc).__name__}: {str(exc)[:180]})"
        ai_status["warnings"].append(init_error)

    # Stage 1: ask Gemma to map the hypothesis. If the API fails, only fall back when
    # the user explicitly supplied an unambiguous supported strategy + lookback + threshold.
    raw_spec = None
    mapping_error = init_error
    if g is not None:
        try:
            raw_spec = gemma.map_quant_hypothesis(g, hypothesis)
        except Exception as exc:
            mapping_error = f"{type(exc).__name__}: {str(exc)[:220]}"
            pipeline_error_trace.append({
                "purpose": "QuantProof hypothesis mapping (failed)",
                "model": gemma.FAST_MODEL,
                "thinking_level": "minimal",
                "image_attached": False,
                "error": mapping_error,
            })
            ai_status["warnings"].append(
                "Gemma hypothesis mapping failed; checking whether the wording contains a complete explicit specification."
            )
    if raw_spec is None:
        fallback_spec = _infer_explicit_strategy_spec(hypothesis)
        if fallback_spec is None:
            detail = (
                "Gemma's API request failed and the hypothesis does not contain one unambiguous "
                "supported strategy type, lookback in bars, and threshold. Please state those values explicitly."
            )
            if mapping_error:
                detail += f" API error: {mapping_error}"
            raise RuntimeError(detail)
        raw_spec = fallback_spec
        ai_status["mapping_source"] = "explicit_hypothesis_fallback"
        ai_status["warnings"].append(
            "Strategy settings were parsed from your explicit hypothesis because the Gemma mapping call failed; verify them before interpreting the backtest."
        )
    spec = validate_strategy_spec(raw_spec)
    if not spec.get("supported"):
        return {
            "supported": False,
            "filename": (filename or "uploaded.csv")[:200],
            "hypothesis": hypothesis,
            "reason": spec.get("reason", "Could not map this hypothesis to a supported template."),
            "supported_templates": sorted(SUPPORTED_STRATEGIES),
            "data_quality": data_quality,
            "ai_status": ai_status,
            "trace": list(getattr(g, "trace", [])) if g is not None else trace,
        }

    base = _evaluate(frame, spec, float(cost_bps), float(train_fraction), data_quality["bars_per_year_assumption"])
    stressed = _evaluate(frame, spec, float(cost_bps) * 2, float(train_fraction), data_quality["bars_per_year_assumption"])

    lookbacks = sorted(set([
        max(3, min(60, int(round(spec["lookback"] * 0.75)))),
        int(spec["lookback"]),
        max(3, min(60, int(round(spec["lookback"] * 1.25)))),
    ]))
    base_threshold = float(spec["threshold_bps"])
    thresholds = sorted(set([
        round(max(0.0, base_threshold * 0.75), 2),
        round(base_threshold, 2),
        round(min(1000.0, base_threshold * 1.25), 2),
    ]))
    sensitivity: list[dict[str, Any]] = []
    for lookback in lookbacks:
        for threshold in thresholds:
            variant = {**spec, "lookback": lookback, "threshold_bps": threshold}
            ev = _evaluate(frame, variant, float(cost_bps), float(train_fraction), data_quality["bars_per_year_assumption"])
            m = ev["strategy"]["out_of_sample"]
            sensitivity.append({
                "lookback": lookback, "threshold_bps": threshold,
                "oos_total_return": m["total_return"], "oos_sharpe": m["sharpe"],
                "oos_max_drawdown": m["max_drawdown"], "oos_observations": m["observations"],
                "is_reference": lookback == spec["lookback"] and threshold == spec["threshold_bps"],
            })

    regime_analysis = _market_regime_analysis(frame, base, float(train_fraction), spec["lookback"])
    metrics = {
        "strategy": base["strategy"],
        "buy_hold": base["buy_hold"],
        "double_costs_out_of_sample": stressed["strategy"]["out_of_sample"],
        "market_regime_analysis": regime_analysis,
        "cost_bps_one_way": float(cost_bps),
        "training_fraction": float(train_fraction),
        "split_date": base["split_date"],
        "test_start": base["test_start"],
        "test_end": base["test_end"],
        "bars_per_year_assumption": data_quality["bars_per_year_assumption"],
    }
    robustness = _robustness_assessment(
        base["strategy"]["in_sample"], base["strategy"]["out_of_sample"],
        stressed["strategy"]["out_of_sample"], sensitivity,
        base["strategy"]["out_of_sample"]["observations"],
    )
    curve = base["curve"]
    chart_png = _equity_chart_png(curve)

    # Stage 2: prefer Gemma's multimodal review. If it fails, try the lighter Gemma 4
    # model without an image. If that also fails, preserve the computed research results
    # and return a deterministic review that clearly discloses the fallback.
    report = None
    report_error = None
    if g is not None:
        try:
            report = gemma.explain_quantproof_results(
                g, hypothesis, spec, metrics, robustness, sensitivity, data_quality, chart_png
            )
        except Exception as exc:
            report_error = f"{type(exc).__name__}: {str(exc)[:220]}"
            pipeline_error_trace.append({
                "purpose": "QuantProof research review (failed)",
                "model": gemma.DEEP_MODEL,
                "thinking_level": "high",
                "image_attached": True,
                "error": report_error,
            })
            try:
                # Short text-only review fallback: still Gemma 4, but no chart/image and less thinking.
                compact_prompt = (
                    "Review these already-computed quantitative research diagnostics conservatively. "
                    "Do not recompute values or recommend trades. State in-sample vs out-of-sample differences, "
                    "cost sensitivity, parameter sensitivity, sample limitations, and next experiments. "
                    "Return JSON with headline, assessment, key_findings (array of title/detail/metric_keys), "
                    "caveats (array), next_experiments (array), not_financial_advice=true.\n\n" +
                    json.dumps({"hypothesis": hypothesis[:1000], "spec": spec, "metrics": metrics,
                                "robustness": robustness, "sensitivity": sensitivity,
                                "data_quality": data_quality}, default=str)[:14000]
                )
                report = g.generate_json(
                    gemma.FAST_MODEL, compact_prompt,
                    system=gemma.QUANT_REPORT_SYSTEM, thinking="minimal", temperature=0.1,
                    purpose="QuantProof text-only research review (fallback Gemma 4 model)",
                )
                ai_status["report_source"] = "gemma_text_only_fallback"
                ai_status["warnings"].append(
                    "The multimodal Gemma review failed; a text-only review from the lighter Gemma 4 model was used."
                )
            except Exception as fallback_exc:
                report_error = (
                    report_error + "; alternate Gemma review also failed: " +
                    f"{type(fallback_exc).__name__}: {str(fallback_exc)[:180]}"
                )
                pipeline_error_trace.append({
                    "purpose": "QuantProof text-only research review (failed)",
                    "model": gemma.FAST_MODEL,
                    "thinking_level": "minimal",
                    "image_attached": False,
                    "error": str(fallback_exc)[:220],
                })
                report = _local_quant_report(metrics, robustness, report_error)
                ai_status["report_source"] = "deterministic_fallback"
                ai_status["warnings"].append(
                    "Both Gemma research-review attempts failed; metrics and the displayed review are deterministic Python outputs."
                )
    else:
        report = _local_quant_report(metrics, robustness, init_error or "Gemma client unavailable")
        ai_status["report_source"] = "deterministic_fallback"
        ai_status["warnings"].append(
            "Gemma was unavailable; backtest metrics were still computed locally and the report is a deterministic fallback."
        )

    # A failed initial mapping may still have created a trace on a real GemmaClient.
    trace = list(getattr(g, "trace", [])) if g is not None else []
    # Include the fallback/error entries produced by the pipeline, without duplicating them.
    existing_purposes = {(x.get("purpose"), x.get("error")) for x in trace}
    # Pipeline errors are kept separately so a later successful request cannot erase them.
    for item in pipeline_error_trace:
        if (item.get("purpose"), item.get("error")) not in existing_purposes:
            trace.append(item)

    # Remove pandas series, which are internal implementation details and not JSON-serialisable.
    return {
        "supported": True,
        "filename": (filename or "uploaded.csv")[:200],
        "hypothesis": hypothesis,
        "strategy_spec": spec,
        "data_quality": data_quality,
        "metrics": metrics,
        "robustness": robustness,
        "parameter_sensitivity": sensitivity,
        "equity_curve": curve,
        "report": report,
        "ai_status": ai_status,
        "trace": trace,
    }
