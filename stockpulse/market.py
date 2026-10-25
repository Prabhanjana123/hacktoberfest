"""Price data, computed price evidence, and the chart image Gemma reads."""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

DEFAULT_WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "AMD", "INTC",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "BPCL.NS",
]


def benchmark_for(ticker: str) -> tuple[str, str]:
    t = ticker.upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return "^NSEI", "NIFTY 50 index"
    return "SPY", "SPY (S&P 500 ETF)"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"]).copy()
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    return df


def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No price data found for {ticker}. Check the symbol (Indian stocks need .NS).")
    return _clean(df)


def company_name(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).get_info()
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def scan_watchlist(tickers: list[str], window: int = 5) -> list[dict]:
    """5-day (window) change for each ticker, biggest decliners first."""
    out = []
    try:
        raw = yf.download(tickers, period="1mo", interval="1d", group_by="ticker",
                          progress=False, auto_adjust=False, threads=True)
    except Exception:
        raw = None
    for t in tickers:
        try:
            closes = (raw[t]["Close"] if raw is not None and len(tickers) > 1 else raw["Close"]).dropna()
            if len(closes) <= window:
                continue
            last, base = float(closes.iloc[-1]), float(closes.iloc[-1 - window])
            out.append({"ticker": t, "price": round(last, 2), "change": round(last / base - 1, 4)})
        except Exception:
            continue
    return sorted(out, key=lambda r: r["change"])


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def price_evidence(ticker: str, df: pd.DataFrame, bench: pd.DataFrame | None,
                   bench_name: str, window: int = 5) -> tuple[list[dict], dict]:
    """Turn raw prices into numbered, checkable facts (P1..Pn)."""
    closes = df["Close"]
    if len(closes) < window + 2:
        raise ValueError("Not enough price history.")
    window = min(window, len(closes) - 1)
    end_date, start_date = closes.index[-1], closes.index[-1 - window]
    end_px, start_px = float(closes.iloc[-1]), float(closes.iloc[-1 - window])
    ret = end_px / start_px - 1

    daily = closes.pct_change().iloc[-window:]
    worst_day = daily.idxmin()
    worst = float(daily.min())

    vol = df["Volume"]
    base_vol = vol.iloc[max(0, len(vol) - window - 20): len(vol) - window]
    avg20 = float(base_vol.mean()) if len(base_vol) else 0.0
    vol_ratio = float(vol.loc[worst_day]) / avg20 if avg20 > 0 else None

    ev: list[dict] = []

    def add(text: str, **data):
        ev.append({"id": f"P{len(ev) + 1}", "kind": "price", "text": text, "data": data})

    add(f"{ticker} closed at {end_px:.2f} on {end_date:%Y-%m-%d}, a change of {_pct(ret)} over the last "
        f"{window} trading days (from {start_px:.2f} on {start_date:%Y-%m-%d}).",
        window_return=ret)
    add(f"The largest single-day move in the window was {_pct(worst)} on {worst_day:%Y-%m-%d}.",
        worst_day=f"{worst_day:%Y-%m-%d}", worst=worst)
    if vol_ratio:
        add(f"Trading volume on {worst_day:%Y-%m-%d} was {vol_ratio:.1f}x the average daily volume of the "
            f"previous 20 sessions.", vol_ratio=vol_ratio)

    bench_ret = None
    if bench is not None and len(bench) > window:
        b = bench["Close"]
        bench_ret = float(b.iloc[-1]) / float(b.iloc[-1 - window]) - 1
        gap = (ret - bench_ret) * 100
        add(f"The {bench_name} changed {_pct(bench_ret)} over the same {window} trading days, so {ticker} "
            f"{'underperformed' if gap < 0 else 'outperformed'} the market by {abs(gap):.1f} percentage points.",
            bench_return=bench_ret)

    prior_ret = None
    if len(closes) > window + 21:
        prior_ret = float(closes.iloc[-1 - window]) / float(closes.iloc[-1 - window - 20]) - 1
        add(f"In the 20 trading days before the window, {ticker} changed {_pct(prior_ret)}.", prior_return=prior_ret)

    hi = float(closes.max())
    add(f"The current price is {_pct(end_px / hi - 1)} from its 52-week closing high of {hi:.2f}.",
        from_high=end_px / hi - 1)

    summary = {
        "ticker": ticker, "price": round(end_px, 2), "window": window,
        "window_return": ret, "bench_return": bench_ret, "bench_name": bench_name,
        "prior_return": prior_ret, "worst_day": f"{worst_day:%Y-%m-%d}", "worst_move": worst,
        "start_date": f"{start_date:%Y-%m-%d}", "end_date": f"{end_date:%Y-%m-%d}",
    }
    return ev, summary


def series_for_ui(df: pd.DataFrame, days: int = 90) -> dict:
    d = df.iloc[-days:]
    return {"dates": [f"{i:%Y-%m-%d}" for i in d.index],
            "close": [round(float(x), 2) for x in d["Close"]],
            "volume": [int(x) for x in d["Volume"]]}


def chart_png(ticker: str, df: pd.DataFrame, window: int, days: int = 60) -> bytes:
    """Price + volume chart with the analysis window shaded. Sent to Gemma as an image."""
    d = df.iloc[-days:]
    fig, (ax, axv) = plt.subplots(2, 1, figsize=(8, 4.5), dpi=110, sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(d.index, d["Close"], color="#1f4e79", lw=1.8)
    ax.axvspan(d.index[-1 - window], d.index[-1], color="#d9534f", alpha=0.12, label=f"Last {window} sessions")
    ax.set_title(f"{ticker} daily close, last {len(d)} sessions", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    axv.bar(d.index, d["Volume"], color="#7f8c8d", width=0.8)
    axv.set_ylabel("Volume", fontsize=8)
    axv.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
