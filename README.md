# NVVSpeech Challenge Track 2 Evaluation

This repository provides the official evaluation code for Track 2 of the NVVSpeech Challenge @ ISCSLP 2026. A Large Audio-Language Model (LALM) evaluates each synthesized audio sample on five components using a 1–5 scale:

- A: NVV Accuracy, weighted at 30%
- P: NVV Perceptual Effect, weighted at 25%
- N: Overall Naturalness, weighted at 15%
- Q: Overall Quality, weighted at 15%
- E: Overall Expression, weighted at 15%

```text
Track2Score = 100 * (0.30*A + 0.25*P + 0.15*N + 0.15*Q + 0.15*E)
FinalTrack2Score = (Track2Score_ZH + Track2Score_EN) / 2
```

In the formula above, A/P/N/Q/E denote the normalized component scores.

Detailed metric definitions can be found in the official [NVVSpeech Challenge website](https://nvvspeech-challenge.github.io/).

## Repository Contents

- `score.py`: Evaluation entry point, Gemini LALM requests, and input validation.
- `track2_scorer.py`: component parsing, normalization, and Track 2 score calculation.
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
- `gemini_checkpoint.jsonl`: per-sample records used for resuming interrupted runs.

## Test set
Huggingface: [Track 2 test set](https://huggingface.co/datasets/NVVSpeech-Challenge/NVVSpeech-Challenge-Track2-Test-Set)

## Citation
The evaluation framework behind the challenge is described in the [NVV-SuperBench](https://arxiv.org/abs/2604.16211) paper. If you use this challenge or its evaluation protocol in your work, please cite:

@article{xue2026nvv,
  title={NVV-SuperBench: Beyond Words, Beyond Quality-Benchmarking Nonverbal Vocalizations in Speech Generation},
  author={Xue, Liumeng and Bian, Weizhen and Pan, Jiahao and Wu, Wenxuan and Ren, Yilin and Kang, Boyi and Hu, Jingbin and Ma, Ziyang and Wang, Shuai and Qian, Xinyuan and others},
  journal={arXiv preprint arXiv:2604.16211},
  year={2026}
}
