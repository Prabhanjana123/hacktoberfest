---
name: evidence-check
description: Verify that AI-generated explanations are actually supported by a fixed set of evidence. Use when a model has produced claims that cite evidence IDs (headlines, data points, documents) and you need to flag fabricated citations, quotes that are not in the source, invented numbers, or explanations with too little support.
license: MIT
metadata:
  project: StockPulse
  version: "1.0"
---

# Evidence check

This skill grades explanations against evidence the model was given. It never
adds new facts. It answers one question per claim: *does the cited evidence
actually say this?*

## Inputs

A JSON file with two keys:

```json
{
  "evidence": [{"id": "H1", "text": "Headline (Reuters, 2026-10-21): \"...\""}],
  "hypotheses": [
    {
      "id": "A",
      "title": "Market-wide sell-off",
      "claims": [
        {"text": "The index fell 2.1% in the same window.", "evidence_ids": ["P4"], "quote": "changed -2.1%"}
      ]
    }
  ]
}
```

## Procedure

1. Run the deterministic checks:

   ```bash
   python scripts/check_claims.py input.json
   ```

   For each claim it reports:
   - `fabricated_citation`: a cited ID is not in the evidence list.
   - `quote_not_found`: the quote is not (fuzzy ≥ 0.85) a substring of the cited evidence.
   - `number_not_in_evidence`: a percentage in the claim is not in the cited evidence (±0.15 pts).
   - `no_citation` / `missing_quote`.

2. For each claim that passes, judge entailment: read only the claim and the
   text of the evidence it cites, and label it `SUPPORTED`, `PARTIAL`,
   `UNSUPPORTED` or `CONTRADICTED`. Do not use outside knowledge. In
   StockPulse this step is done by a separate Gemma 4 call that never sees
   the generator's reasoning or confidence.

3. Score each hypothesis as the mean claim score
   (SUPPORTED 1, PARTIAL 0.5, UNSUPPORTED 0, CONTRADICTED −0.5; any
   fabricated citation scores 0, a missing or unmatched quote halves the score).

4. Verdict:
   - **Supported**: score ≥ 0.65 and at least 2 distinct supporting evidence items.
   - **Partially supported**: score ≥ 0.40.
   - **Insufficient support**: anything else, or any contradicted or fabricated claim. Flag it.

## Output

Report each hypothesis with its verdict, score, and one plain-language reason
per failed claim. Never upgrade a verdict because an explanation sounds
plausible; plausibility is exactly what this check exists to separate from support.
