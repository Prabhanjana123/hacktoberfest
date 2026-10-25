# StockPulse

**When the news and the price disagree, StockPulse finds out why, and tells you which explanations the evidence actually supports.**

Most finance chatbots summarise headlines and sound confident. StockPulse does the opposite: it notices when a
stock is falling while the headlines are mostly positive, asks Gemma 4 for *competing* explanations grounded in
numbered evidence (including the price chart as an image), and then runs an independent evidence checker that
flags any explanation the evidence does not support.

Built at **Hacktoberfest x MOSS Hack Day Udupi**, 25 October 2026.
Challenges entered: **Best Use of Gemma 4** and **Best Open-Source AI Project**.

## Team

| Name | Role / part built |
| --- | --- |
| _Member 1_ | _e.g. Gemma prompts and evidence checker_ |
| _Member 2_ | _e.g. frontend and demo_ |
| _Member 3 (optional)_ | |
| _Member 4 (optional)_ | |

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
git clone <this repo> && cd stockpulse
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # paste your key from https://aistudio.google.com/apikey
uvicorn app:app --reload
```

Open <http://localhost:8000>. Click a red ticker in the sidebar, or type any symbol (`NVDA`, `INFY.NS`).

Tests (offline, fake Gemma + synthetic prices): `python -m pytest -q`

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
| Claude (Anthropic) | Initial scaffold of the backend (`stockpulse/`), frontend (`static/`), the `evidence-check` skill, tests and this README draft |
| _add any others, e.g. Copilot_ | _which parts_ |

_Update this table honestly with what your team then changed, rewrote or added yourselves._

## License

[MIT](LICENSE)
