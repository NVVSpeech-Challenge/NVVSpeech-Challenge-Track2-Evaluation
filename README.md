# NVVSpeech Challenge Track 2 Evaluation

本仓库提供 NVVSpeech Challenge @ ISCSLP 2026 Track 2 的 Codabench 兼容评测程序。评委模型对合成音频进行 LALM 评测，并给出五项 1–5 分指标：

- A：NVV Accuracy（非语言声音准确性），权重 30%
- P：NVV Perceptual Effect（非语言声音感知效果），权重 25%
- N：Overall Naturalness（整体自然度），权重 15%
- Q：Overall Quality（整体音质），权重 15%
- E：Overall Expression（整体表现力），权重 15%

各项分数先通过 `(x - 1) / 4` 从 1–5 分归一化至 0–1，然后按以下公式计算：

```text
Track2Score = 100 * (0.30*A + 0.25*P + 0.15*N + 0.15*Q + 0.15*E)
FinalTrack2Score = (Track2Score_ZH + Track2Score_EN) / 2
```

上式中的 A/P/N/Q/E 表示归一化后的分数。

## 文件说明

- `score.py`：Codabench 评测入口、Gemini LALM 请求及输入检查。
- `track2_scorer.py`：五项指标解析、归一化与 Track 2 得分计算。
- `metadata.yaml`：Codabench scoring program 启动配置。
- `requirements.txt`：Python 依赖。
- `results/`：可公开的汇总评测结果，不包含测试数据和逐样本记录。

## 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 环境变量

评测前需要在运行环境中设置 API 信息。不要把真实值写入代码或提交到 GitHub。

```bash
export GEMINI_API_BASE_URL="<your-api-base-url>"
export GEMINI_API_KEY="<your-api-key>"
export GEMINI_MODEL="gemini-2.5-pro"
```

可选并发与重试参数：

```bash
export MAX_WORKERS=4
export MAX_RETRIES=6
export GLOBAL_SEED=1234
```

## 本地运行

```bash
python3 score.py /path/to/ref /path/to/res /path/to/output
```

- `ref`：由赛事组织方授权提供的参考数据目录，应包含中英文 `prepared_data_*.json`。
- `res`：待评测的 WAV 文件目录或提交压缩包。
- `output`：评测输出目录。

评测成功后会生成：

- `scores.json`：中文、英文和最终 Track 2 得分。
- `detailed_results.json`：A/P/N/Q/E 小分、归一化值、权重和样本统计。
- `gemini_checkpoint.jsonl`：用于断点续跑的逐样本记录，不建议公开提交。

## Codabench 打包

`program.zip` 中应保留 `program/` 顶层目录：

```bash
mkdir -p program
cp score.py track2_scorer.py metadata.yaml requirements.txt program/
zip -r program.zip program
```

参考数据应由赛事组织方单独配置，不包含在本仓库中。

## 数据与安全说明

本仓库不包含测试集、参考数据、生成音频、API Key、API Base URL 或逐样本 LALM checkpoint。请勿将上述内容提交到公开仓库。
