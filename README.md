# StockPulse

**When the news and the price disagree, StockPulse finds out why, and tells you which explanations the evidence actually supports.**

Most finance chatbots summarise headlines and sound confident. StockPulse does the opposite: it notices when a
stock is falling while the headlines are mostly positive, asks Gemma 4 for *competing* explanations grounded in
numbered evidence (including the price chart as an image), and then runs an independent evidence checker that
flags any explanation the evidence does not support.

Built at **Hacktoberfest x MOSS Hack Day Udupi**, 25 October 2026.
Challenges entered: **Best Use of Gemma 4** and **Best Open-Source AI Project**.

## Team

| Name (GitHub) | Role / part built |
| --- | --- |
| TODO: full name ([@Prabhanjana123](https://github.com/Prabhanjana123)) | TODO: actual contribution |
| TODO: full name ([@sushanthrs360](https://github.com/sushanthrs360)) | TODO: actual contribution |
| TODO: third member and GitHub handle | TODO: actual contribution |

Replace every TODO with the real registered team details before submission.

## Demo flow

1. Open a stock with a recent price decline (the sidebar ranks the watchlist by biggest fall).
2. Most of the collected headlines are labelled positive by Gemma 4.
3. The **contradiction detector** activates: price ≤ −3% over the window while ≥ 50% of headlines are positive.
4. Click **Investigate**.
5. Gemma 4 reads the evidence list **and the chart image** and produces 3–4 competing explanations. Every claim cites evidence IDs (`P1`, `H3`, `C1`) and quotes the evidence verbatim.
6. The **evidence checker** grades every claim and flags the explanation that lacks sufficient support, with plain-language reasons.

## How it works

```
 yfinance prices ──► computed price facts P1..P6 ──┐          (deterministic, not generated)
 Yahoo + Google News ► headlines H1..Hn ───────────┤
 matplotlib chart.png ► C1 (image) ─────────────────┤
                                                    ▼
            Gemma 4 26B A4B  ── headline sentiment (batched, thinking: minimal)
                                                    ▼
            Contradiction detector (pure rules: price direction vs headline tone)
                                                    ▼  Investigate
            Gemma 4 31B  ── competing explanations, text + chart image (thinking: high)
                                                    ▼
            Evidence checker
              1. deterministic checks (skills/evidence-check): IDs exist, quote is verbatim, numbers match
              2. independent Gemma 4 26B A4B judge: sees only claim + cited evidence (+ chart)
                                                    ▼
            Supported · Partially supported · Insufficient support (flagged)
```

**Why two checking layers?** A model grading itself is not a check. The deterministic layer catches what a model
cannot be trusted to self-report (a cited headline that does not exist, a quote that is not in the source, a number
that appears nowhere in the evidence). The judge is a *separate* Gemma call that never sees the generator's
reasoning or confidence, only the claim and the evidence it cites.

**Scoring** (`stockpulse/checker.py`): claim score SUPPORTED 1 · PARTIAL 0.5 · UNSUPPORTED 0 · CONTRADICTED −0.5;
a fabricated citation scores 0, a missing or unmatched quote halves the score. An explanation is *Supported* at
≥ 0.65 with ≥ 2 distinct supporting items, *Partial* at ≥ 0.40, otherwise flagged. The UI shows Gemma's own
confidence next to the checker's score so you can see where they disagree.

## How to verify the Gemma 4 integration during the demo

1. The header shows the two configured Gemma 4 model IDs once `GEMINI_API_KEY` is set (from `GET /api/health`).
2. Open **Gemma 4 call trace** at the bottom of the page. Every row is a real `generate_content` call recorded by
   `stockpulse/gemma.py`: model ID, thinking level, whether `chart.png` was attached, latency and token counts
   returned by the API.
3. After **Investigate**, the "What Gemma saw in the chart" panel shows the exact PNG sent to Gemma and its reading of it.
4. In the code, every model call goes through `GemmaClient.generate()` in `stockpulse/gemma.py`. There is no
   fallback to any other model: if Gemma 4 is unreachable, the request fails with the API's error.

**Offline safety net:** every run is saved to `snapshots/`. Tick **Replay** to reload the last run of a ticker
without network access.

## Gemma 4 and the Gemini API

| Model | Used for | How it's called |
| --- | --- | --- |
| `gemma-4-31b-it` | Competing explanations, multimodal (evidence text + chart PNG) | `client.models.generate_content(model="gemma-4-31b-it", contents=[Part.from_bytes(png, "image/png"), prompt], config=GenerateContentConfig(system_instruction=..., thinking_config=ThinkingConfig(thinking_level="high")))` |
| `gemma-4-26b-a4b-it` | Headline sentiment (thinking `minimal`) and the independent claim judge (thinking `high`, with chart image) | same SDK, `model="gemma-4-26b-a4b-it"` |

All calls are in [`stockpulse/gemma.py`](stockpulse/gemma.py) using the official `google-genai` SDK. The UI's
**Gemma 4 call trace** panel lists every call made: model, thinking level, whether the chart image was attached,
latency and token counts.

- Gemma 4 license: **Apache 2.0**, <https://ai.google.dev/gemma/apache_2>
- Gemma terms and prohibited use policy: <https://ai.google.dev/gemma/terms>, <https://ai.google.dev/gemma/prohibited_use_policy>
- Model IDs can be overridden with `GEMMA_DEEP_MODEL` and `GEMMA_FAST_MODEL` in `.env`.


## QuantProof Lab (added research module)

QuantProof extends StockPulse with a constrained, AI-assisted quantitative research workflow. A researcher uploads a timestamped OHLCV CSV and describes a hypothesis in natural language. Gemma 4 maps the hypothesis to one fixed strategy template; deterministic Python then runs the strategy, compares in-sample and out-of-sample periods, applies user-specified transaction costs, compares with buy-and-hold, tests nearby parameter settings and doubled costs, and summarizes out-of-sample performance by volatility regime using thresholds learned only from the training slice. Gemma 4 then reviews the computed metrics and an equity-curve image. Performance numbers come from Python, never from generated prose.

### Supported templates in this MVP

- **Momentum:** long when the trailing return over a selected lookback exceeds a threshold.
- **Mean reversion:** long when the trailing return falls below the negative threshold.
- **Breakout:** long when the close exceeds the highest high of the preceding lookback by a threshold.

The strategy engine is deliberately constrained to long/cash positions and does not execute generated code. It assumes the close-based signal is applied to the next bar. Costs are a simple basis-point estimate per position change. The split is chronological. This MVP is not a production backtester: it does not fully model spreads, market impact, borrowing, corporate actions, partial fills, survivorship bias, or walk-forward nested model selection. A positive holdout result is not proof of a persistent edge.

### Using QuantProof

1. Open the **QuantProof Lab** tab.
2. Describe one testable hypothesis that can be represented as momentum, mean reversion, or breakout.
3. Upload a CSV with timestamp/date, open, high, low, close and volume columns (minimum 80 valid bars; daily and regular intraday bars are accepted).
4. Set a one-way transaction-cost estimate and chronological training fraction. You can download `static/quantproof_sample_synthetic.csv` to verify the workflow; it is synthetic and must not be presented as actual market history.
5. Click **Map hypothesis & run research**.
6. Inspect the strategy specification, in-sample versus out-of-sample metrics, buy-and-hold baseline, doubled-cost test, parameter-sensitivity grid, volatility-regime summary and Gemma's cautious report.

Gemma is used for constrained natural-language hypothesis mapping and multimodal interpretation of the Python-generated equity curve plus metrics. Python remains authoritative for all prices, signals, returns, Sharpe, drawdowns, costs and sensitivity calculations.

### QuantProof tests

```bash
python -m pytest -q tests/test_quantproof.py
```

The tests use a fake Gemma client and synthetic data. They test deterministic code paths and do **not** verify live Gemini API availability or strategy profitability.

## Agent Skill

[`skills/evidence-check/`](skills/evidence-check/SKILL.md) is a reusable skill following the Agent Skills standard
(`SKILL.md` with `name`/`description` frontmatter, plus `scripts/`). It packages the evidence-checking procedure so
any agent can grade claims against a fixed evidence set:

```bash
python skills/evidence-check/scripts/check_claims.py input.json
```

StockPulse itself loads its deterministic checks from this skill, so the app and the skill share one implementation.

## Run it

```bash
git clone https://github.com/Prabhanjana123/hacktoberfest && cd hacktoberfest
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # Windows: copy .env.example .env   — then paste your key from https://aistudio.google.com/apikey
uvicorn app:app --reload
```

Open <http://localhost:8000>. Click a red ticker in the sidebar, or type any symbol (`NVDA`, `INFY.NS`).

Tests: `python -m pytest -q` (run from the repository root).
They are offline and use a fake Gemma client and synthetic prices, so they verify the pipeline, checker, API
error handling and skill CLI, **not** the live Gemini API.

## Known limitations

- Headlines come from Yahoo Finance and Google News RSS. Coverage varies by ticker, and some stocks return few
  or off-topic headlines, so the detector may not fire.
- The detector thresholds (−3% move, ≥ 50% positive, ≥ 3 headlines) are fixed rules chosen for this demo, not
  tuned on historical data.
- The quote check is a fuzzy text match (≥ 0.85). It confirms the words are in the source, not that the claim
  follows from them; that is the Gemma judge's job, and the judge can also be wrong.
- Gemma 4 on the Gemini API free tier has low rate limits. One analysis + investigation makes 3 calls.
- Snapshots in `snapshots/` are real past runs; Replay shows those, not live data.
- StockPulse explains price moves from public headlines. It is not financial advice.

## Open-source components

| Component | Purpose | License |
| --- | --- | --- |
| [Gemma 4](https://ai.google.dev/gemma) | All AI reasoning | [Apache 2.0](https://ai.google.dev/gemma/apache_2) |
| [google-genai](https://github.com/googleapis/python-genai) | Gemini API SDK | Apache 2.0 |
| [yfinance](https://github.com/ranaroussi/yfinance) | Prices and Yahoo Finance headlines | Apache 2.0 |
| [feedparser](https://github.com/kurtmckee/feedparser) | Google News RSS headlines | BSD-2-Clause |
| [FastAPI](https://github.com/fastapi/fastapi) / [Uvicorn](https://github.com/encode/uvicorn) | Web server | MIT / BSD-3-Clause |
| [pandas](https://github.com/pandas-dev/pandas) | Price calculations | BSD-3-Clause |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | Chart image sent to Gemma | Matplotlib (PSF-based) |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` loading | BSD-3-Clause |
| [Chart.js](https://github.com/chartjs/Chart.js) | Interactive chart in the UI | MIT |
| [IBM Plex](https://github.com/IBM/plex) | Fonts | SIL OFL 1.1 |

**Data sources:** Yahoo Finance (via yfinance) and Google News RSS. Data is used for a non-commercial demo; see each
provider's terms. StockPulse is not financial advice.

## AI tools used to build this

As the rulebook requires, every AI tool used during the event is listed here.

| Tool | What it produced |
| --- | --- |
| Claude (Anthropic), via claude.ai | Generated the initial version of every source file during build time (25 Oct 2026, 10:50–10:56 IST): backend (`stockpulse/`), frontend (`static/`), `app.py`, the `evidence-check` skill, `tests/test_pipeline.py` and this README. Later in the event it restored the folder structure after the web upload flattened it, removed a delisted ticker, added error logging in `app.py`, added `tests/test_app.py` and updated this README. |
| ChatGPT (OpenAI), via ChatGPT | Generated the QuantProof MVP: constrained strategy mapping/report prompts, deterministic backtest and robustness module, API endpoint, dashboard tab, synthetic demo CSV, tests and documentation integration. Later added the Gemma Chat backend endpoint, conversation UI, optional session context, model fallback handling, tests and documentation. The team must review and understand the code and accurately disclose all AI-assisted contributions. |

**Changes made by the team:** TODO: list the actual changes, tests, prompt tuning and live-data testing performed by your team. Fill in all remaining TODOs before submission.

## License

[MIT](LICENSE)

## QuantProof: handling Gemma API `500 INTERNAL` errors

QuantProof is designed to preserve a completed deterministic backtest if the hosted Gemma review endpoint fails. It tries the configured Gemma 4 model for hypothesis mapping, then the alternate configured Gemma 4 model. If both mapping requests fail, it only parses a local fallback when the user's hypothesis explicitly contains one supported strategy family, a single lookback in bars, and an unambiguous threshold. It never silently invents missing settings.

For the final report, the multimodal review is attempted first, followed by a text-only review using the lighter configured Gemma 4 model. If both fail, QuantProof returns a clearly labelled deterministic report and adds an AI availability warning to the UI. Check the call trace and warning before demonstrating results; do not claim Gemma generated a fallback report.


## Gemma Chat

StockPulse includes an interactive Gemma 4 chat tab for quantitative-finance questions, research interpretation, statistics, and help using the app. It supports Quick mode (configured fast model) and Deep mode (configured reasoning model); if the requested model fails, it attempts the alternate configured Gemma model and visibly discloses the fallback.

The browser sends the recent conversation (up to 12 turns), the selected mode, and—if the user leaves the context checkbox enabled—a compact summary of any loaded StockPulse analysis or QuantProof research result to `POST /api/chat`. The server does not persist chat messages. Do not enter API keys or other secrets in the chat.

The chatbot is a research/educational assistant, not a live market-data feed, financial adviser, or trading-signal service. It may be wrong; numeric performance metrics should be verified against the deterministic Python research engine.
