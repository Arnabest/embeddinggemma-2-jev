# Dataset Schema & Format Specification

This directory defines the input schema for training and fine-tuning `EmbeddingGemma-2-Jev` decision adapters.

## Data Isolation Policy
Private conversational logs, proprietary code diffs, and local user trajectories are strictly isolated from the open-source repository. Only generic, sanitized engineering examples are stored in `sample_dataset.jsonl`.

## JSONL Format
Each line corresponds to one JSON object with the following schema:

```json
{
  "id": "sample_unique_id",
  "source": "source_category",
  "state": "Background context, error trace, or code diff (up to 8K tokens)",
  "noul": {
    "statement": "Hypothesis or safety condition to judge",
    "y": 1
  },
  "choice": {
    "question": "Which action should the dispatch engine take?",
    "options": [
      {"t": "Action Option A", "d": "Rationale for option A"},
      {"t": "Action Option B", "d": "Rationale for option B"}
    ],
    "y": 0
  },
  "score": {
    "question": "Assess the risk/danger tier of this situation:",
    "levels": [
      "1=Safe / Normal",
      "2=Low risk",
      "3=Moderate caution",
      "4=High danger / OOM imminent",
      "5=Fatal / Data corruption"
    ],
    "y": 3
  }
}
```

- `noul.y`: Integer (`0` = False/Safe, `1` = True/Anomaly).
- `choice.y`: 0-indexed integer pointing to the ground-truth choice in `options`.
- `score.y`: 0-indexed integer pointing to the ground-truth severity in `levels`.
