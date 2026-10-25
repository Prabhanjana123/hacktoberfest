"""Gemma 4 through the Gemini API (google-genai SDK).

Three separate Gemma 4 jobs:
  1. classify_headlines  - gemma-4-26b-a4b-it, thinking minimal (fast, batched)
  2. explain_divergence  - gemma-4-31b-it, thinking high, text + chart IMAGE
  3. verify_claims       - gemma-4-26b-a4b-it, thinking high, independent judge
Every call is logged in `trace` so the UI can show exactly what Gemma did.
"""
from __future__ import annotations

import json
import os
import re
import time

from google import genai
from google.genai import types

FAST_MODEL = os.getenv("GEMMA_FAST_MODEL", "gemma-4-26b-a4b-it")
DEEP_MODEL = os.getenv("GEMMA_DEEP_MODEL", "gemma-4-31b-it")


def extract_json(text: str):
    """Pull the first JSON object out of a model reply (handles ``` fences and prose)."""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("No JSON in model reply")
    depth, in_str, esc = 0, False, False
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise ValueError("Unterminated JSON in model reply")


class GemmaClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (get one free at https://aistudio.google.com/apikey).")
        self.client = genai.Client(api_key=key)
        self.trace: list[dict] = []

    def generate(self, model: str, prompt: str, *, system: str | None = None,
                 image_png: bytes | None = None, thinking: str = "minimal",
                 temperature: float = 0.2, purpose: str = "") -> str:
        contents: list = []
        if image_png:
            contents.append(types.Part.from_bytes(data=image_png, mime_type="image/png"))
        contents.append(prompt)

        def cfg(with_thinking: bool):
            kw = dict(system_instruction=system, temperature=temperature)
            if with_thinking:
                kw["thinking_config"] = types.ThinkingConfig(thinking_level=thinking)
            return types.GenerateContentConfig(**kw)

        t0 = time.time()
        try:
            resp = self.client.models.generate_content(model=model, contents=contents, config=cfg(True))
        except Exception as e:  # older SDKs / models without thinking_level
            if "thinking" not in str(e).lower():
                raise
            resp = self.client.models.generate_content(model=model, contents=contents, config=cfg(False))
        text = resp.text or ""
        usage = getattr(resp, "usage_metadata", None)
        self.trace.append({
            "purpose": purpose, "model": model, "thinking_level": thinking,
            "image_attached": bool(image_png), "latency_s": round(time.time() - t0, 2),
            "prompt_chars": len(prompt),
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
        })
        return text

    def generate_json(self, model: str, prompt: str, **kw):
        text = self.generate(model, prompt, **kw)
        try:
            return extract_json(text)
        except (ValueError, json.JSONDecodeError):
            fix = self.generate(model, "Return ONLY valid JSON, no prose, fixing this reply:\n" + text[:6000],
                                thinking="minimal", purpose=kw.get("purpose", "") + " (JSON repair)")
            return extract_json(fix)


# ----------------------------------------------------------------- 1. sentiment

SENTIMENT_SYSTEM = (
    "You label financial news headlines by their tone for a company's shareholders. "
    "Judge only the wording of each headline and summary. Reply with JSON only."
)


def classify_headlines(g: GemmaClient, ticker: str, name: str, headline_ev: list[dict]) -> dict[str, dict]:
    if not headline_ev:
        return {}
    lines = "\n".join(f"[{e['id']}] {e['text']}" for e in headline_ev)
    prompt = f"""Company: {name} ({ticker})

Label each headline's tone for {ticker} shareholders as "positive", "negative" or "neutral".
Headlines about other companies or general market news that do not mention {name} are "neutral".

{lines}

Reply as JSON:
{{"labels": [{{"id": "H1", "sentiment": "positive", "confidence": 0.0, "reason": "max 12 words"}}]}}"""
    data = g.generate_json(FAST_MODEL, prompt, system=SENTIMENT_SYSTEM, thinking="minimal",
                           temperature=0.0, purpose="Headline sentiment")
    out = {}
    for item in data.get("labels", []):
        hid = str(item.get("id", "")).upper()
        s = str(item.get("sentiment", "neutral")).lower()
        out[hid] = {"sentiment": s if s in {"positive", "negative", "neutral"} else "neutral",
                    "confidence": float(item.get("confidence") or 0), "reason": item.get("reason", "")}
    return out


# ------------------------------------------------- 2. competing explanations

EXPLAIN_SYSTEM = """You are an evidence-bound equity analyst.
Rules:
- Use ONLY the numbered evidence and the attached chart image. No outside knowledge, no memory of real events.
- Every claim must cite evidence IDs and give a short quote copied VERBATIM from one cited text item.
- If an explanation is plausible but the evidence for it is thin, still include it, and say so honestly.
  An independent checker will grade every claim. Never invent evidence, IDs, numbers or quotes.
Reply with JSON only."""


def explain_divergence(g: GemmaClient, ticker: str, name: str, summary: dict,
                       detector: dict, evidence: list[dict], chart: bytes | None) -> dict:
    ev_lines = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)
    prompt = f"""{name} ({ticker}) changed {summary['window_return'] * 100:+.1f}% over {summary['window']} trading days,
yet {detector['positive']} of {detector['total']} recent headlines read as positive
({detector['negative']} negative, {detector['neutral']} neutral). Explain the contradiction.

The attached image (evidence C1) is the {ticker} price and volume chart; the shaded area is the analysis window.

EVIDENCE
{ev_lines}
[C1] Attached chart image of daily closes and volume.

TASK
1. In "chart_reading", describe in 1-2 sentences what the chart image shows about the move.
2. Give 3 or 4 COMPETING explanations for why the price fell while headlines were positive. Make them
   genuinely different mechanisms, for example: a market-wide or sector move; good news already priced in
   ("sell the news") after a prior run-up; a negative development the headlines play down; the headline
   collection being skewed or off-topic; or flows and technical selling. Choose what fits the evidence.
3. Each explanation has 2-3 claims. Each claim cites 1-3 evidence IDs and quotes up to 20 words verbatim
   from one cited text item (claims citing only C1 may leave the quote empty).

JSON format:
{{
  "chart_reading": "...",
  "hypotheses": [
    {{
      "id": "A",
      "title": "max 8 words",
      "mechanism": "1-2 sentences",
      "claims": [{{"text": "...", "evidence_ids": ["P1"], "quote": "verbatim words"}}],
      "would_confirm": "what new evidence would confirm it",
      "would_refute": "what new evidence would refute it",
      "model_confidence": 0.0
    }}
  ]
}}"""
    return g.generate_json(DEEP_MODEL, prompt, system=EXPLAIN_SYSTEM, image_png=chart, thinking="high",
                           temperature=0.4, purpose="Competing explanations (text + chart image)")


# ---------------------------------------------------------- 3. independent judge

VERIFY_SYSTEM = """You are a strict fact-checker. For each claim you see ONLY the claim and the evidence it cites.
Decide whether that evidence, by itself, establishes the claim. Do not use outside knowledge.
SUPPORTED = the cited evidence directly states or clearly implies it.
PARTIAL = the evidence is related and consistent but does not establish the claim (e.g. a causal leap).
UNSUPPORTED = the evidence does not say this.
CONTRADICTED = the evidence says the opposite.
Reply with JSON only."""


def verify_claims(g: GemmaClient, claims: list[dict], evidence_by_id: dict[str, dict],
                  chart: bytes | None) -> dict[str, dict]:
    if not claims:
        return {}
    blocks = []
    for c in claims:
        cited = "\n".join(f"    [{i}] {evidence_by_id[i]['text']}" for i in c["valid_ids"]) or "    (none)"
        blocks.append(f"CLAIM {c['cid']}: {c['text']}\n  CITED EVIDENCE:\n{cited}")
    prompt = ("Evidence [C1], where cited, is the attached chart image.\n\n" + "\n\n".join(blocks) +
              '\n\nJSON format: {"verdicts": [{"claim_id": "A1", "verdict": "SUPPORTED", '
              '"rationale": "max 20 words"}]}')
    data = g.generate_json(FAST_MODEL, prompt, system=VERIFY_SYSTEM, image_png=chart, thinking="high",
                           temperature=0.0, purpose="Independent claim verification")
    out = {}
    for v in data.get("verdicts", []):
        verdict = str(v.get("verdict", "UNSUPPORTED")).upper()
        if verdict not in {"SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"}:
            verdict = "UNSUPPORTED"
        out[str(v.get("claim_id", "")).upper()] = {"verdict": verdict, "rationale": v.get("rationale", "")}
    return out
