"""StockPulse web server. Run: uvicorn app:app --reload  then open http://localhost:8000"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from stockpulse import gemma, market, pipeline, quantproof  # noqa: E402

log = logging.getLogger("stockpulse")
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
        log.exception("Request for %s failed", ticker)  # full traceback in the terminal
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
        log.exception("Request for %s failed", ticker)  # full traceback in the terminal
        raise HTTPException(502, f"{type(e).__name__}: {e}")


class QuantProofRequest(BaseModel):
    hypothesis: str = Field(min_length=10, max_length=2000)
    csv_text: str = Field(min_length=50, max_length=quantproof.MAX_CSV_CHARS)
    filename: str = Field(default="uploaded.csv", max_length=200)
    transaction_cost_bps: float = Field(default=10.0, ge=0.0, le=100.0)
    train_fraction: float = Field(default=0.7, ge=0.6, le=0.85)


@app.post("/api/quantproof/analyze")
def quantproof_analyze(request: QuantProofRequest):
    """Run a constrained, AI-assisted backtest on user-provided OHLCV CSV."""
    try:
        return quantproof.run_research(
            hypothesis=request.hypothesis,
            csv_text=request.csv_text,
            filename=request.filename,
            cost_bps=request.transaction_cost_bps,
            train_fraction=request.train_fraction,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log.exception("QuantProof request failed")
        raise HTTPException(502, f"QuantProof failed: {type(exc).__name__}: {exc}") from exc


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatTurn] = Field(min_length=1, max_length=12)
    context: str = Field(default="", max_length=6000)
    depth: Literal["quick", "deep"] = "quick"


@app.post("/api/chat")
def chat(request: ChatRequest):
    """Answer a conversational question with Gemma 4; history stays client-side."""
    if not request.messages or request.messages[-1].role != "user":
        raise HTTPException(422, "The last chat message must be from the user.")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise HTTPException(503, "Gemini API key is not configured. Add GEMINI_API_KEY to the local .env file and restart the server.")
    try:
        client = gemma.GemmaClient()
        return gemma.chat_reply(
            client,
            [turn.model_dump() for turn in request.messages],
            context=request.context,
            depth=request.depth,
        )
    except Exception as exc:
        log.exception("Gemma chatbot request failed")
        message = str(exc).replace(os.getenv("GEMINI_API_KEY", "__NO_KEY__"), "[redacted]")
        raise HTTPException(502, f"Gemma chat failed: {type(exc).__name__}: {message[:240]}") from exc
