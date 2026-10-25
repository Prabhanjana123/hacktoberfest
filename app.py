"""StockPulse web server. Run: uvicorn app:app --reload  then open http://localhost:8000"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from stockpulse import gemma, market, pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="StockPulse")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"gemini_key": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
            "fast_model": gemma.FAST_MODEL, "deep_model": gemma.DEEP_MODEL,
            "snapshots": sorted(p.stem for p in pipeline.SNAPSHOTS.glob("*.json")) if pipeline.SNAPSHOTS.exists() else []}


@app.get("/api/watchlist")
def watchlist(tickers: str | None = None, window: int = 5):
    syms = [t.strip().upper() for t in tickers.split(",")] if tickers else market.DEFAULT_WATCHLIST
    return {"items": market.scan_watchlist(syms, window)}


@app.get("/api/analyze/{ticker}")
def analyze(ticker: str, window: int = Query(5, ge=2, le=30), replay: bool = False):
    if replay:
        snap = pipeline.load_snapshot(ticker)
        if not snap:
            raise HTTPException(404, f"No saved snapshot for {ticker}.")
        pipeline._cache[ticker.upper()] = snap
        return {**snap, "replayed": True}
    try:
        return pipeline.analyze(ticker, window)
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.post("/api/investigate/{ticker}")
def investigate(ticker: str, replay: bool = False):
    if replay:
        snap = pipeline.load_snapshot(ticker)
        if snap and snap.get("investigation"):
            return {**snap, "replayed": True}
    try:
        return pipeline.investigate(ticker)
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")
