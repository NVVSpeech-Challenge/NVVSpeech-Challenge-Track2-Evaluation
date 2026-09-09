# NVVSpeech Challenge Track 2 Evaluation

This repository provides the Codabench-compatible evaluation program for Track 2 of the NVVSpeech Challenge @ ISCSLP 2026. A Large Audio-Language Model (LALM) evaluates each synthesized audio sample on five components using a 1–5 scale:

- A: NVV Accuracy, weighted at 30%
- P: NVV Perceptual Effect, weighted at 25%
- N: Overall Naturalness, weighted at 15%
- Q: Overall Quality, weighted at 15%
- E: Overall Expression, weighted at 15%

Each component score is normalized from 1–5 to 0–1 using `(x - 1) / 4`, and the Track 2 scores are calculated as follows:

```text
Track2Score = 100 * (0.30*A + 0.25*P + 0.15*N + 0.15*Q + 0.15*E)
FinalTrack2Score = (Track2Score_ZH + Track2Score_EN) / 2
```

In the formula above, A/P/N/Q/E denote the normalized component scores.

## Repository Contents

- `score.py`: Codabench evaluation entry point, Gemini LALM requests, and input validation.
- `track2_scorer.py`: component parsing, normalization, and Track 2 score calculation.
- `metadata.yaml`: Codabench scoring-program launch configuration.
- `requirements.txt`: Python dependencies.
- `results/`: publishable aggregate evaluation results without test data or per-sample records.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Environment Variables

Set the API configuration in the runtime environment before evaluation. Never write real credentials into the source code or commit them to GitHub.

```bash
export GEMINI_API_BASE_URL="<your-api-base-url>"
export GEMINI_API_KEY="<your-api-key>"
export GEMINI_MODEL="gemini-2.5-pro"
```

Optional concurrency and retry settings:

```bash
export MAX_WORKERS=4
export MAX_RETRIES=6
export GLOBAL_SEED=1234
```

## Local Evaluation

```bash
python3 score.py /path/to/ref /path/to/res /path/to/output
```

- `ref`: an organizer-authorized reference directory containing the Chinese and English `prepared_data_*.json` files.
- `res`: a directory of submitted WAV files or a submission archive.
- `output`: the directory in which evaluation outputs will be written.

A successful evaluation produces:

- `scores.json`: Chinese, English, and final Track 2 scores.
- `detailed_results.json`: A/P/N/Q/E component scores, normalized values, weights, and sample statistics.
- `gemini_checkpoint.jsonl`: per-sample records used for resuming interrupted runs; this file should not be published.

## Codabench Packaging

The `program.zip` archive must preserve `program/` as its top-level directory:

```bash
mkdir -p program
cp score.py track2_scorer.py metadata.yaml requirements.txt program/
zip -r program.zip program
```

Reference data must be configured separately by the challenge organizers and is not included in this repository.

## Data and Security

This repository does not contain the test set, reference data, generated audio, API keys, API base URLs, or per-sample LALM checkpoints. Do not commit any of these materials to a public repository.
