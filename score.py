#!/usr/bin/env python3
"""NVVSpeech Challenge Track 2 — CodaBench Scoring Program.

Flow:
  1. Load prepared_data (GT texts) from ref/
  2. Find contestant WAV files from res/ (zip or directory)
  3. Send each WAV + reference text to Gemini LALM for 5-dimension scoring
  4. Compute Track2Score per language, then FinalTrack2Score
  5. Write scores.json

Requires: GEMINI_API_KEY environment variable set on the compute worker.
"""

import asyncio
import base64
from collections import defaultdict
import glob
import json
import math
import mimetypes
import os
import posixpath
import re
import random
import sys
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

# Ensure $program is on path for track2_scorer import
_program_dir = os.path.dirname(os.path.abspath(__file__))
if _program_dir not in sys.path:
    sys.path.insert(0, _program_dir)

# ---------------------------------------------------------------------------
# Try to install aiohttp if missing (PyPI was confirmed reachable)
# ---------------------------------------------------------------------------
try:
    import aiohttp  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp", "-q"])
    import aiohttp  # noqa: F401

# ---------------------------------------------------------------------------
# Constants (mirror llm_eval.py for consistency)
# ---------------------------------------------------------------------------
ALLOWED_ISSUES = [
    "clipping", "distortion", "background_noise", "codec_artifacts", "harsh_sibilance",
    "metallic_timbre", "robotic_prosody", "glitches_breaks", "unnatural_phonemes"
]
QUALITY_CAP_ISSUES = {"clipping", "distortion", "background_noise", "codec_artifacts", "harsh_sibilance"}
NATURALNESS_CAP_ISSUES = {"metallic_timbre", "robotic_prosody", "glitches_breaks", "unnatural_phonemes"}

KEYWORD_FALLBACK = {
    # Keyword fallback is only a safety net.
    # Keep terms specific (avoid generic 'artifact/compression/codec/break') to reduce false triggers.
    "clipping": ["clipping", "clipped", "削波", "削顶", "爆音削波"],
    "distortion": ["distortion", "distorted", "overdriven", "失真", "破音", "过载"],
    "background_noise": [
        "background noise", "noise floor", "hiss", "hum", "static",
        "底噪", "电流声", "嗡嗡声", "嘶嘶声", "噪声很大", "环境噪声很大"
    ],
    # Avoid broad 'compression/codec/artifact' — use strong-signal phrases only.
    "codec_artifacts": [
        "heavy compression", "strong compression artifacts", "bitrate artifacts",
        "warbling", "watery", "swirling", "robotic artifacts",
        "严重压缩伪影", "明显压缩伪影", "码率伪影", "水声感", "涟漪感"
    ],
    "harsh_sibilance": ["sibilance", "harsh s", "piercing s", "齿音重", "尖锐齿音", "刺耳齿音"],
    "metallic_timbre": ["metallic", "tinny", "metallic timbre", "金属音", "铁皮音", "很金属"],
    "robotic_prosody": ["robotic", "machine-like", "mechanical", "flat prosody", "机械感", "机器味", "很机器人", "语气很平"],
    # Avoid generic 'break' (e.g., 'break immersion'). Use specific audio glitch terms.
    "glitches_breaks": ["glitch", "dropout", "audio drop", "stutter", "click", "pop", "crackle", "断音", "卡顿", "爆裂声", "噼啪声"],
    # Avoid generic 'phoneme'. Use specific mispronunciation / intelligibility signals.
    "unnatural_phonemes": [
        "mispronunciation", "wrong pronunciation", "garbled", "mumbled", "slurred",
        "错读", "读错", "发音不准", "吐字不清", "口齿不清", "含混", "声调不对", "不太听得清"
    ],
}

METRIC_TO_JSON_KEY = {
    "Overall Naturalness": "overall_naturalness",
    "Overall Quality": "overall_quality",
    "Overall Expression": "overall_expression",
    "NVC Accuracy": "nvv_accuracy",
    "NVC Perceptual Effect (PE)": "nvv_pe",
}

TAG_METRICS = [
    "Overall Naturalness",
    "Overall Quality",
    "Overall Expression",
    "NVC Accuracy",
    "NVC Perceptual Effect (PE)",
]

METRICS_INFO_EN = {
    "Overall Naturalness": "How natural and human-like does the speech sound overall? Focus on whether it feels 'real'. Ignore background noise (rated in Quality).",
    "Overall Quality": "Signal-level audio quality. Rate how clean, clear, and comfortable to listen to the audio signal itself is, ignoring naturalness or content.",
    "Caption–Audio Match (CAM)": "Does the synthesized audio match the caption/prompt in terms of speaker, emotion, and scenario? Ignore detailed NVC timing (rated in IF).",
    "Overall Expression": "Overall expressive effect of speech + NVC, and how well the performance conveys emotion and attitude. Judge by listening alone, not comparing to the caption.",
    "NVC Instruction Following (IF)": "In prompt-based synthesis, does the model follow the NVC-related instructions (type, position) in the text? If no NVC is heard, score 0.",
    "NVC Accuracy": "In tag-based TTS, do the generated NVCs match the input NVC tags (type, position)? If no NVC is heard, score 0.",
    "NVC Perceptual Effect (PE)": "Regardless of instructions, how natural and expressive do the NVCs sound? Do they enhance the performance? If no NVC is heard, score 0.",
}


METRIC_SCORING_RULES_EN = {
    "Overall Naturalness": {
        "5": "Indistinguishable from real human speech.",
        "4": "Sounds like a real person with only slight unnaturalness.",
        "3": "Intelligible, but synthetic qualities are clearly noticeable.",
        "2": "Rather unnatural, sounds more like a machine.",
        "1": "Extremely unnatural or almost intolerable."
    },
    "Overall Quality": {
        "5": "Very clean and clear audio, like a studio recording.",
        "4": "Clear and comfortable audio with only slight, non-intrusive noise/artifacts.",
        "3": "Noticeable issues (noise, compression) that may distract.",
        "2": "Serious quality problems (heavy noise, distortion).",
        "1": "Very poor or unusable audio quality."
    },
    "Caption–Audio Match (CAM)": {
        "5": "Extremely well-matched to caption in voice, emotion, and scene.",
        "4": "Good match with only minor deviations.",
        "3": "Roughly matches, but with at least one obvious core mismatch (e.g., gender, emotion).",
        "2": "Poor match with multiple clear inconsistencies.",
        "1": "Almost no match with the caption."
    },
    "Overall Expression": {
        "5": "Very vivid, with clear emotion and strong dynamics.",
        "4": "Expression is generally good and engaging.",
        "3": "Some emotion is present, but it's relatively flat.",
        "2": "Expression is rather stiff or disjointed.",
        "1": "Almost no perceivable emotion or expressive variation."
    },
    "NVC Instruction Following (IF)": {
        "5": "Fully follows the NVC instruction (type and position).",
        "4": "Generally matches the instruction with minor deviations.",
        "3": "Recognizable NVC, but with clear mismatches in type or timing.",
        "2": "NVC heard, but it does not match most of the instruction.",
        "1": "NVC is almost unrelated to the instruction.",
        "0": "No NVC or almost no NVC is audible."
    },
    "NVC Accuracy": {
        "5": "The NVC matches the tag perfectly (type and timing).",
        "4": "The NVC generally matches the tag well.",
        "3": "The NVC is only partially consistent with the tag.",
        "2": "The NVC mostly does not match the expected tag.",
        "1": "The NVC is almost completely unrelated to the tag.",
        "0": "No NVC or almost no NVC is audible."
    },
    "NVC Perceptual Effect (PE)": {
        "5": "The NVC sounds very natural, expressive, and enhances emotion.",
        "4": "The NVC sounds generally natural and works as a plus.",
        "3": "Naturalness is average, with some synthetic feeling.",
        "2": "The NVC sounds rather stiff, overacted, or connects poorly.",
        "1": "The NVC is extremely unnatural and breaks immersion.",
        "0": "No NVC or almost no NVC is audible."
    },
}


SYSTEM_INSTRUCTION = "You are a STRICT professional MOS rater for TTS audio. Do not inflate scores. Use the full scale. Score 5 is extremely rare. Follow the rubric and output JSON only (no markdown)."

# Configurable via env
API_BASE_URL = os.environ.get(
    "GEMINI_API_BASE_URL", ""
).rstrip("/")
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
MAX_WORKERS = max(1, int(os.environ.get("MAX_WORKERS", "100")))
MAX_RETRIES = max(1, int(os.environ.get("MAX_RETRIES", "6")))
MAX_RECOVERY_ROUNDS = max(
    1, int(os.environ.get("MAX_RECOVERY_ROUNDS", "10"))
)
RECOVERY_COOLDOWN = max(
    1.0, float(os.environ.get("RECOVERY_COOLDOWN", "30"))
)
TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.2"))
GLOBAL_SEED = int(os.environ.get("GLOBAL_SEED", "1234"))


class RetryableEvaluationError(RuntimeError):
    """A temporary API/network/model error; the sample must be retried."""


class PermanentSampleError(RuntimeError):
    """The submitted sample itself cannot be evaluated."""


class FatalEvaluationError(RuntimeError):
    """A global configuration/authentication error that retries cannot fix."""


# ---------------------------------------------------------------------------
# Prompt builders (adapted from llm_eval.py)
# ---------------------------------------------------------------------------
STRICTNESS_BLOCK = f"""
--- STRICTNESS & CONSISTENCY RULES (CRITICAL) ---
You must be a careful, consistent judge. Use the full 1–5 scale and avoid both inflation and unnecessary harshness.

A) IMPORTANT: Separate Naturalness vs Quality
- Overall Naturalness measures human-likeness (prosody, pronunciation, flow). Do NOT penalize Naturalness for mild background hiss/compression; those belong to Quality.
- Overall Quality measures audio fidelity/cleanliness (noise, distortion/clipping, codec artifacts, clicks/pops/dropouts, harsh sibilance). Do NOT penalize Quality just because the voice sounds synthetic/robotic; that belongs to Naturalness.

B) Score meaning (use these anchors)
Overall Quality:
- 5: Very clean and clear, like a studio recording. No noticeable noise/artifacts/distortion.
- 4: Clear and comfortable; only slight, non-intrusive noise/artifacts.
- 3: Noticeable issues (noise/codec artifacts/etc.) that may distract.
- 2: Serious quality problems (heavy noise/distortion/clipping) that reduce comfort/clarity.
- 1: Very poor or unusable quality.

Overall Naturalness:
- 5: Indistinguishable from real human speech (VERY rare).
- 4: Very natural with minor synthetic cues.
- 3: Noticeably synthetic/robotic traits.
- 2: Strongly synthetic or awkward.
- 1: Very unnatural.

C) Issues (must output)
Output an "issues" list using ONLY labels from this allowed set:
{ALLOWED_ISSUES}
If no issues are present, output [].
Only list issues you can actually hear.

D) Consistency rules you MUST follow
- If you list ANY Quality-related issue (clipping/distortion/background_noise/codec_artifacts/harsh_sibilance/glitches_breaks),
  you MUST NOT give Overall Quality = 5.
  * slight/minor/non-intrusive -> Quality is usually 4
  * noticeable/distracting     -> Quality is usually 3
  * heavy/strong/severe        -> Quality is usually 1–2

- If you list ANY Naturalness-related issue (metallic_timbre/robotic_prosody/unnatural_phonemes/glitches_breaks),
  you MUST NOT give Overall Naturalness = 5.

E) Sanity check
- If you give any metric a 5, the reason must explicitly state: "no noticeable issues".
"""

def build_json_schema(active_metrics):
    # valid JSON example (no trailing commas)
    obj = {
        "heard_summary": "",
        "issues": [],
    }
    for metric in active_metrics:
        key = METRIC_TO_JSON_KEY[metric]
        obj[f"{key}_score"] = 1 if "NVC" not in metric else 0
        obj[f"{key}_reason"] = ""
    obj["caps_applied"] = {}
    return json.dumps(obj, indent=2)

def build_evaluation_prompt() -> str:
    is_prompt_task = False
    active_metrics = [m for m in (PROMPT_METRICS if is_prompt_task else TAG_METRICS) 
                      if m in METRICS_INFO_EN]

    base_prompt = (
        "You are an expert evaluator for a tag-based text-to-speech system.\n"
        "You will be given ONE audio sample and its source text.\n"
        "IMPORTANT: The system identity is UNKNOWN and IRRELEVANT. Do not assume any system.\n"
        "Rate the sample on the specified metrics.\n\n"
        "Carefully review the metric definitions and scoring rubric below.\n"
    )

    definitions_block = "--- METRIC DEFINITIONS ---\n"
    for metric in active_metrics:
        definitions_block += f"\n### {metric}\n{METRICS_INFO_EN[metric]}\n"

    rules_block = "\n--- SCORING RUBRIC ---\n"
    for metric in active_metrics:
        rules_block += f"\n#### {metric}\n"
        for score, desc in METRIC_SCORING_RULES_EN[metric].items():
            rules_block += f"- Score {score}: {desc}\n"

    schema = build_json_schema(active_metrics)

    output_instructions = (
        "\n" + STRICTNESS_BLOCK + "\n"
        "\n--- OUTPUT FORMAT ---\n"
        "Return ONLY a valid JSON object matching this schema example (same keys). No extra text.\n"
        f"{schema}\n\n"
        "--- SOURCE TEXT FOR THIS SAMPLE ---\n"
    )
    return base_prompt + definitions_block + rules_block + output_instructions
def clamp_int(x, lo, hi):
    try:
        xi = int(x)
    except Exception:
        return None
    return max(lo, min(hi, xi))


def extract_issues(pred: dict) -> set:
    """
    Extract issues from model output.
    Priority:
      1) Structured 'issues' field (must be from ALLOWED_ISSUES)
      2) Keyword fallback on ONLY: heard_summary, overall_quality_reason, overall_naturalness_reason
         (avoid scanning CAM/IF/PE/Expression reasons to reduce false triggers)

    NOTE: This function is used only for post-processing consistency on Overall Quality/Naturalness.
    """
    issues = set()
    if not isinstance(pred, dict):
        return issues

    # 1) Structured issues (preferred)
    raw = pred.get("issues", [])
    if isinstance(raw, list):
        for it in raw:
            if isinstance(it, str) and it.strip() in ALLOWED_ISSUES:
                issues.add(it.strip())
    elif isinstance(raw, str) and raw.strip() in ALLOWED_ISSUES:
        issues.add(raw.strip())

    # 2) Keyword fallback (safety net) — limited fields to reduce false triggers
    # Negation guard: if a negation appears right before the keyword, do not count it as a hit.
    neg_pat = re.compile(r"\b(no|not|without)\b|没有|无", re.IGNORECASE)

    def contains_kw(text: str, kw: str) -> bool:
        if not text or not kw:
            return False
        t = text.lower()
        k = kw.lower()
        idx = t.find(k)
        if idx < 0:
            return False
        left = t[max(0, idx - 12):idx]
        if neg_pat.search(left):
            return False
        return True

    fields = []
    for k in ("heard_summary", "overall_quality_reason", "overall_naturalness_reason"):
        v = pred.get(k)
        if isinstance(v, str) and v.strip():
            fields.append(v.lower())
    joined = " ".join(fields)

    for issue, kws in KEYWORD_FALLBACK.items():
        for kw in kws:
            if contains_kw(joined, kw):
                issues.add(issue)
                break

    return issues
def apply_hard_caps(pred: dict, task_mode: str) -> dict:
    """
    Apply rubric-consistent caps on Overall Quality / Overall Naturalness.

    Goal:
      - Avoid score collapse at 3 caused by over-eager hard caps.
      - Enforce consistency only when the judge's own text indicates clear problems,
        especially *severe/heavy* problems.

    Three-tier logic (cap is an upper bound; never boosts scores):
      Overall Quality:
        - heavy/serious noise or distortion -> cap <= 2
        - noticeable, distracting issues     -> cap <= 3
        - any stated issue but score==5      -> cap <= 4  (since 5 means no audible issues)

      Overall Naturalness: analogous tiers.

    Notes:
      - Severity is inferred from the judge's wording in:
          heard_summary + overall_quality_reason / overall_naturalness_reason
      - If the judge mentions an issue without any severity cues, we avoid over-capping;
        we only prevent contradictory '5' in that case.
    """
    if not isinstance(pred, dict):
        return pred

    out = dict(pred)
    issues = extract_issues(out)
    caps_applied = {}

    def cap(score_key: str, cap_value: int, lo: int, hi: int, why: str):
        s = clamp_int(out.get(score_key), lo, hi)
        if s is None:
            return
        if s > cap_value:
            out[score_key] = cap_value
            caps_applied[score_key] = why

    def joined_text(*keys: str) -> str:
        parts = []
        for k in keys:
            v = out.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
        return " ".join(parts).lower()

    def severity_flags(text: str):
        # Strong/severe signals -> "heavy"
        heavy = bool(re.search(
            r"\b(heavy|strong|severe|serious|extreme)\b|明显|严重|重度|持续|频繁|一直|影响理解|听不清|不舒服|刺耳|难以忍受|hard to listen|uncomfortable|painful|hurts",
            text, flags=re.IGNORECASE
        ))
        # Moderate signals -> "noticeable"
        moderate = bool(re.search(
            r"\b(noticeable|distract|distracting|annoying|bothersome|may distract)\b|干扰|分散注意|明显可闻|听得出|听出来|比较明显",
            text, flags=re.IGNORECASE
        ))
        # Mild signals -> "slight"
        slight = bool(re.search(
            r"\b(slight|minor|subtle|non-intrusive)\b|轻微|不明显|不太影响|基本不影响|偶尔",
            text, flags=re.IGNORECASE
        ))
        return heavy, moderate, slight

    # ----------------------------
    # Overall Quality (1–5)
    # ----------------------------
    quality_issue_any = issues & (set(QUALITY_CAP_ISSUES) | {"glitches_breaks"})
    qt = joined_text("heard_summary", "overall_quality_reason")
    q_heavy, q_mod, q_slight = severity_flags(qt)

    # If any issue is stated but score is 5, it's inconsistent with the rubric definition of 5.
    if quality_issue_any:
        cap("overall_quality_score", 4, 1, 5, "issue_present_so_not_5")

    # Three-tier caps only when severity cues exist (avoid over-capping vague wording)
    if q_heavy and (issues & {"background_noise", "distortion", "clipping"}):
        cap("overall_quality_score", 2, 1, 5, "heavy_noise_or_distortion")
    elif q_mod and quality_issue_any:
        cap("overall_quality_score", 3, 1, 5, "noticeable_distracting_issue")

    # ----------------------------
    # Overall Naturalness (1–5)
    # ----------------------------
    nat_issue_any = issues & set(NATURALNESS_CAP_ISSUES)
    nt = joined_text("heard_summary", "overall_naturalness_reason")
    n_heavy, n_mod, n_slight = severity_flags(nt)

    if nat_issue_any:
        cap("overall_naturalness_score", 4, 1, 5, "issue_present_so_not_5")

    if n_heavy and (issues & {"glitches_breaks", "unnatural_phonemes", "robotic_prosody"}):
        cap("overall_naturalness_score", 2, 1, 5, "severely_unnatural_or_breaks")
    elif n_mod and nat_issue_any:
        cap("overall_naturalness_score", 3, 1, 5, "noticeably_synthetic")

    # Keep original conservative cap for tag mode if NVC totally absent (does NOT affect Quality/Naturalness).
    if task_mode == "tag":
        nvv_acc = clamp_int(out.get("nvv_accuracy_score"), 0, 5)
        nvv_pe = clamp_int(out.get("nvv_pe_score"), 0, 5)
        if nvv_acc == 0 or nvv_pe == 0:
            s = clamp_int(out.get("overall_expression_score"), 1, 5)
            if s is not None and s > 3:
                out["overall_expression_score"] = 3
                caps_applied["overall_expression_score"] = "tag_mode_nvv_absent"

    out["issues"] = sorted(list(issues))
    out["caps_applied"] = caps_applied
    return out

# ==============================================================================
# 3) Prompt builder (point #1, #2)
def _encode_audio_part(audio_path: str) -> dict:
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except OSError as exc:
        # Do not expose the filename / utterance ID in user-visible errors.
        raise PermanentSampleError(
            f"Cannot read submitted audio file: {type(exc).__name__}"
        ) from exc
    if not audio_bytes:
        raise PermanentSampleError(
            "Submitted audio file is empty"
        )
    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        ext = os.path.splitext(audio_path)[1].lower()
        mime_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
        mime_type = mime_map.get(ext, "audio/wav")
    b64_data = base64.b64encode(audio_bytes).decode("utf-8")
    return {"inline_data": {"mime_type": mime_type, "data": b64_data}}


def _clean_json_response(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


async def _call_gemini(
    session: aiohttp.ClientSession,
    parts: list,
) -> str:
    api_url = f"{API_BASE_URL}/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    payload = {
        "contents": [{"parts": parts}],
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "generationConfig": {
            "temperature": TEMPERATURE,
            "response_mime_type": "application/json",
        },
    }
    delay = 3.0
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(api_url, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError, TypeError) as exc:
                        raise RetryableEvaluationError(
                            f"Malformed Gemini response: {json.dumps(data)[:500]}"
                        ) from exc
                    return _clean_json_response(text)

                response_text = await resp.text()
                if resp.status in (408, 409, 425, 429, 500, 502, 503, 504):
                    if attempt < MAX_RETRIES - 1:
                        retry_after = resp.headers.get("Retry-After")
                        try:
                            wait = float(retry_after) if retry_after else delay
                        except ValueError:
                            wait = delay
                        await asyncio.sleep(wait + random.uniform(0, wait * 0.25))
                        delay = min(delay * 2, 60)
                        continue
                    raise RetryableEvaluationError(
                        f"Temporary HTTP {resp.status} after retry exhaustion"
                    )

                if resp.status in (400, 413, 415, 422):
                    raise PermanentSampleError(
                        f"Submitted audio was rejected with HTTP {resp.status}"
                    )
                if resp.status in (401, 403):
                    raise FatalEvaluationError(
                        f"API authentication/permission error HTTP {resp.status}"
                    )
                raise FatalEvaluationError(
                    f"Non-retryable API error HTTP {resp.status}"
                )
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay + random.uniform(0, delay * 0.25))
                delay = min(delay * 2, 60)
                continue
            raise RetryableEvaluationError(
                f"Network error after {MAX_RETRIES} attempts: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    raise RetryableEvaluationError("Gemini request exhausted retries")


async def evaluate_one(
    session: aiohttp.ClientSession,
    audio_path: str,
    source_text: str,
    eval_prompt: str,
) -> dict:
    parts = [
        {"text": eval_prompt + source_text + "\n"},
        _encode_audio_part(audio_path),
    ]
    raw = await _call_gemini(session, parts)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RetryableEvaluationError(
            f"Invalid JSON response: {raw[:500]}"
        ) from exc
    return validate_prediction(apply_hard_caps(parsed, "tag"))


REQUIRED_SCORE_KEYS = (
    "nvv_accuracy_score",
    "nvv_pe_score",
    "overall_naturalness_score",
    "overall_quality_score",
    "overall_expression_score",
)


def validate_prediction(pred: dict) -> dict:
    """Require one complete, finite and in-range five-dimensional result."""
    if not isinstance(pred, dict):
        raise ValueError("Gemini prediction must be a JSON object")

    for key in REQUIRED_SCORE_KEYS:
        value = pred.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RetryableEvaluationError(
                f"Missing or non-numeric score: {key}"
            )
        if not math.isfinite(float(value)):
            raise RetryableEvaluationError(f"Non-finite score: {key}={value}")
        lower = 0.0 if key.startswith("nvv_") else 1.0
        if not lower <= float(value) <= 5.0:
            raise RetryableEvaluationError(
                f"Out-of-range score: {key}={value}; expected [{lower}, 5]"
            )
    return pred


# ---------------------------------------------------------------------------
# Scoring (uses track2_scorer)
# ---------------------------------------------------------------------------
def compute_scores(per_lang_scores: Dict[str, Dict[str, List[float]]]) -> dict:
    """Compute Track2Score per lang and bilingual final from collected raw scores."""
    from track2_scorer import compute_track2_score, compute_final_bilingual_score

    lang_results = {}
    for lang in ("zh", "en"):
        comps = per_lang_scores.get(lang, {})
        means = {k: sum(v) / len(v) for k, v in comps.items() if v}
        result = compute_track2_score(
            A=means.get("A", 0),
            P=means.get("P", 0),
            N=means.get("N", 0),
            Q=means.get("Q", 0),
            E=means.get("E", 0),
            normalize=True,
        )
        lang_results[lang] = result

    zh_score = lang_results["zh"]["track2_score"]
    en_score = lang_results["en"]["track2_score"]
    bilingual = compute_final_bilingual_score(zh_score, en_score)

    return {
        "Track2Score_ZH": round(zh_score, 6),
        "Track2Score_EN": round(en_score, 6),
        "FinalTrack2Score": round(bilingual["final_track2_score"], 6),
        "details": {
            "zh": lang_results["zh"],
            "en": lang_results["en"],
        },
    }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
IGNORED_SUBMISSION_FILENAMES = {".DS_Store"}


def _is_ignored_submission_path(path: str) -> bool:
    """Ignore only known packaging artifacts; never inspect or print IDs."""
    normalized = path.replace("\\", "/")
    basename = posixpath.basename(normalized)
    return (
        basename in IGNORED_SUBMISSION_FILENAMES
        or normalized.startswith("__MACOSX/")
        or "/__MACOSX/" in normalized
    )


def _zip_member_basename(name: str) -> str:
    """Return a platform-independent basename for a ZIP member."""
    return posixpath.basename(name.replace("\\", "/").rstrip("/"))


def _looks_like_zip(path: str) -> bool:
    """Recognize ZIP archives even when CodaBench removes the .zip suffix."""
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def find_wavs(res_dir: str) -> Dict[str, str]:
    """Find submitted audio recursively or inside ZIP archives.

    The filename stem is still used as the sample ID, exactly as in the
    original scorer. The only discovery change is that CodaBench wrapper
    directories and extension-less ZIP archives are supported.
    """
    zip_files: List[str] = []
    loose_audio_files: List[str] = []

    for root, _dirs, files in os.walk(res_dir):
        for filename in files:
            fpath = os.path.join(root, filename)
            relpath = os.path.relpath(fpath, res_dir)
            if _is_ignored_submission_path(relpath):
                continue

            ext = os.path.splitext(filename)[1].lower()
            if ext in SUPPORTED_AUDIO_EXTENSIONS:
                loose_audio_files.append(fpath)
            elif ext == ".zip" or _looks_like_zip(fpath):
                zip_files.append(fpath)

    wav_map: Dict[str, str] = {}

    # Preserve the original precedence: archive entries are registered first,
    # then loose files are used only when that ID is not already present.
    for zpath in sorted(zip_files):
        try:
            with zipfile.ZipFile(zpath) as zf:
                import tempfile
                tmpdir = tempfile.mkdtemp(prefix="wav_")
                for info in zf.infolist():
                    if info.is_dir() or _is_ignored_submission_path(info.filename):
                        continue
                    member_name = info.filename.replace("\\", "/")
                    ext = os.path.splitext(member_name)[1].lower()
                    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
                        continue
                    basename = _zip_member_basename(member_name)
                    sample_id = os.path.splitext(basename)[0]
                    extracted = zf.extract(info, tmpdir)
                    wav_map[sample_id] = extracted
        except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError):
            # inspect_submission_files() reports the archive error without IDs.
            continue

    for fpath in sorted(loose_audio_files):
        sample_id = os.path.splitext(os.path.basename(fpath))[0]
        if sample_id not in wav_map:
            wav_map[sample_id] = fpath

    return wav_map


def inspect_submission_files(res_dir: str) -> Dict[str, Any]:
    """Collect ID-free diagnostics without changing scoring semantics.

    Internal filename stems are retained only for comparison with reference
    IDs. They are never printed or written to validation_error.json.
    """
    info: Dict[str, Any] = {
        "total_files": 0,
        "zip_files": 0,
        "invalid_zip_files": 0,
        "empty_audio_zips": 0,
        "unsupported_files": 0,
        "unsupported_extensions": defaultdict(int),
        "unsupported_records": [],       # internal: (stem, extension)
        "nested_supported_audio": 0,
        "supported_audio_occurrences": 0,
        "loose_stem_counts": defaultdict(int),
        "zip_stem_counts": defaultdict(int),
        "empty_audio_stems": [],         # internal only
        "encrypted_audio_entries": 0,
    }

    for root, _dirs, files in os.walk(res_dir):
        for filename in files:
            fpath = os.path.join(root, filename)
            relpath = os.path.relpath(fpath, res_dir)
            if _is_ignored_submission_path(relpath):
                continue

            info["total_files"] += 1
            ext = os.path.splitext(filename)[1].lower()
            is_archive = ext == ".zip" or (
                ext not in SUPPORTED_AUDIO_EXTENSIONS and _looks_like_zip(fpath)
            )

            if is_archive:
                info["zip_files"] += 1
                try:
                    with zipfile.ZipFile(fpath) as zf:
                        audio_count = 0
                        for member in zf.infolist():
                            if member.is_dir() or _is_ignored_submission_path(
                                member.filename
                            ):
                                continue

                            member_name = member.filename.replace("\\", "/")
                            member_ext = os.path.splitext(member_name)[1].lower()
                            basename = _zip_member_basename(member_name)
                            stem = os.path.splitext(basename)[0]

                            if member_ext in SUPPORTED_AUDIO_EXTENSIONS:
                                audio_count += 1
                                info["supported_audio_occurrences"] += 1
                                info["zip_stem_counts"][stem] += 1
                                if member.file_size == 0:
                                    info["empty_audio_stems"].append(stem)
                                if member.flag_bits & 0x1:
                                    info["encrypted_audio_entries"] += 1
                            else:
                                info["unsupported_files"] += 1
                                shown_ext = member_ext or "<no extension>"
                                info["unsupported_extensions"][shown_ext] += 1
                                info["unsupported_records"].append(
                                    (stem, shown_ext)
                                )

                        if audio_count == 0:
                            info["empty_audio_zips"] += 1
                except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError):
                    info["invalid_zip_files"] += 1
                continue

            if ext in SUPPORTED_AUDIO_EXTENSIONS:
                stem = os.path.splitext(filename)[0]
                info["supported_audio_occurrences"] += 1
                info["loose_stem_counts"][stem] += 1
                if root != res_dir:
                    info["nested_supported_audio"] += 1
                try:
                    if os.path.getsize(fpath) == 0:
                        info["empty_audio_stems"].append(stem)
                except OSError:
                    # The later sample-read stage returns a specific INVALID_SAMPLE.
                    pass
            else:
                shown_ext = ext or "<no extension>"
                stem = os.path.splitext(filename)[0]
                info["unsupported_files"] += 1
                info["unsupported_extensions"][shown_ext] += 1
                info["unsupported_records"].append((stem, shown_ext))

    # A ZIP plus its CodaBench-extracted copy is not treated as a duplicate.
    # Only repeated IDs within loose files or within archive entries are
    # ambiguous and therefore invalid.
    duplicate_stems = set()
    duplicate_occurrences = 0
    for counts_key in ("loose_stem_counts", "zip_stem_counts"):
        for stem, count in info[counts_key].items():
            if count > 1:
                duplicate_stems.add(stem)
                duplicate_occurrences += count - 1

    info["duplicate_stems"] = duplicate_stems       # internal only
    info["duplicate_occurrences"] = duplicate_occurrences
    info["unsupported_extensions"] = dict(info["unsupported_extensions"])
    info["loose_stem_counts"] = dict(info["loose_stem_counts"])
    info["zip_stem_counts"] = dict(info["zip_stem_counts"])
    return info


def format_extension_summary(extension_counts: Dict[str, int]) -> str:
    """Return only extension/count information; never filenames or IDs."""
    return ", ".join(
        f"{ext}: {count}"
        for ext, count in sorted(
            extension_counts.items(), key=lambda item: (-item[1], item[0])
        )
    )


def classify_prevalidation_error(
    diagnostics: Dict[str, Any],
    submitted_count: int,
    matched_count: int,
    missing_count: int,
    unexpected_count: int,
) -> Tuple[str, str]:
    """Return one precise, ID-free pre-validation error."""
    invalid_zips = diagnostics.get("invalid_zip_files", 0)
    empty_zips = diagnostics.get("empty_audio_zips", 0)
    zip_count = diagnostics.get("zip_files", 0)
    required_unsupported = diagnostics.get(
        "unsupported_matching_required_ids", 0
    )
    required_unsupported_exts = diagnostics.get(
        "unsupported_matching_extensions", {}
    )
    empty_required = diagnostics.get("empty_matching_required_ids", 0)
    duplicate_required = diagnostics.get(
        "duplicate_matching_required_ids", 0
    )
    encrypted_entries = diagnostics.get("encrypted_audio_entries", 0)

    if duplicate_required:
        return (
            "DUPLICATE_AUDIO_IDS",
            f"Found {duplicate_required} required sample ID(s) represented by "
            "more than one submitted audio file. Each required ID must appear "
            "exactly once.",
        )

    if encrypted_entries:
        return (
            "ENCRYPTED_ZIP_ARCHIVE",
            f"Found {encrypted_entries} encrypted audio file(s) inside a ZIP "
            "archive. Password-protected submissions cannot be evaluated.",
        )

    # If required files are still missing, an unreadable archive is a more
    # specific cause than a generic missing-file message.
    if invalid_zips and missing_count:
        return (
            "INVALID_ZIP_ARCHIVE",
            f"{invalid_zips} submitted ZIP archive(s) cannot be opened. "
            f"Only {matched_count} of {matched_count + missing_count} required "
            "audio files could be matched.",
        )

    if empty_required:
        return (
            "EMPTY_AUDIO_FILE",
            f"Found {empty_required} required audio file(s) with zero bytes. "
            "Every required audio file must contain valid audio data.",
        )

    if submitted_count == 0:
        if invalid_zips:
            return (
                "INVALID_ZIP_ARCHIVE",
                f"{invalid_zips} submitted ZIP archive(s) cannot be opened.",
            )

        if required_unsupported:
            suffix = (
                f" ({format_extension_summary(required_unsupported_exts)})"
                if required_unsupported_exts else ""
            )
            return (
                "UNSUPPORTED_FILE_FORMAT",
                f"Found {required_unsupported} required sample file(s) with "
                f"unsupported format{suffix}. Allowed audio extensions are "
                ".wav, .mp3, and .flac.",
            )

        if zip_count and empty_zips == zip_count:
            return (
                "EMPTY_AUDIO_ARCHIVE",
                f"Found {zip_count} ZIP archive(s), but none contains a "
                "supported audio file.",
            )

        unsupported_count = diagnostics.get("unsupported_files", 0)
        if unsupported_count:
            extension_summary = format_extension_summary(
                diagnostics.get("unsupported_extensions", {})
            )
            suffix = f" ({extension_summary})" if extension_summary else ""
            return (
                "UNSUPPORTED_FILE_FORMAT",
                f"No supported audio files were found. Found "
                f"{unsupported_count} unsupported file(s){suffix}. Allowed "
                "audio extensions are .wav, .mp3, and .flac.",
            )

        return (
            "EMPTY_SUBMISSION",
            "No submitted audio files were found.",
        )

    if matched_count == 0 and unexpected_count:
        extra = ""
        if required_unsupported:
            extra = (
                f" In addition, {required_unsupported} required sample "
                "file(s) use an unsupported format."
            )
        return (
            "INVALID_AUDIO_IDS",
            f"Found {submitted_count} supported audio file(s), but none of "
            f"their filename stems matches a reference ID.{extra}",
        )

    if missing_count and unexpected_count and required_unsupported:
        return (
            "MULTIPLE_SUBMISSION_ERRORS",
            f"Matched {matched_count} required audio file(s); "
            f"{unexpected_count} submitted filename stem(s) do not match any "
            f"reference ID; {required_unsupported} required sample file(s) "
            f"use an unsupported format; and {missing_count} required audio "
            "file(s) remain unavailable.",
        )

    if missing_count and unexpected_count:
        return (
            "MISMATCHED_AUDIO_IDS",
            f"Matched {matched_count} required audio file(s); "
            f"{unexpected_count} submitted filename stem(s) do not match any "
            f"reference ID, and {missing_count} required audio file(s) are "
            "missing.",
        )

    if missing_count and required_unsupported:
        suffix = (
            f" ({format_extension_summary(required_unsupported_exts)})"
            if required_unsupported_exts else ""
        )
        return (
            "UNSUPPORTED_FILE_FORMAT",
            f"Matched {matched_count} required audio file(s), but "
            f"{required_unsupported} required sample file(s) use an "
            f"unsupported format{suffix}; {missing_count} required audio "
            "file(s) remain unavailable.",
        )

    if missing_count and zip_count and empty_zips == zip_count:
        return (
            "EMPTY_AUDIO_ARCHIVE",
            f"The submitted ZIP archive(s) contain no supported audio files. "
            f"Only {matched_count} of {matched_count + missing_count} required "
            "audio files could be matched.",
        )

    return (
        "MISSING_SUBMISSION_FILES",
        f"Matched {matched_count} of {matched_count + missing_count} required "
        f"audio file(s); {missing_count} required audio file(s) are missing.",
    )


def load_prepared_data(ref_dir: str) -> dict:
    """Load prepared_data_{lang}_*.json files from ref/.
    Returns: {lang: {sample_id: {text, text_with_mark, ...}}}
    """
    data = {}
    for fpath in sorted(glob.glob(os.path.join(ref_dir, "prepared_data_*.json"))):
        fname = os.path.basename(fpath)
        lang = "zh" if "_zh" in fname else "en"
        with open(fpath, "r") as f:
            items = json.load(f)
        data[lang] = {}
        for item in items:
            sid = item["id"]
            text_with_mark = item.get("text_with_mark_original") or \
                             item.get("text_with_mark_voxcpm") or \
                             item.get("text_with_mark") or \
                             item.get("text", "")
            data[lang][sid] = {
                "text": item.get("text", ""),
                "text_with_mark": text_with_mark,
            }
    return data


def write_error_outputs(
    output_dir: str,
    code: str,
    message: str,
    errors: Optional[Dict[str, str]] = None,
) -> None:
    """Write validation/error info without exposing utterance IDs."""
    os.makedirs(output_dir, exist_ok=True)

    # A failed run must never leave a score from a previous run behind.
    for filename in ("scores.json", "detailed_results.json"):
        stale_path = os.path.join(output_dir, filename)
        if os.path.isfile(stale_path):
            os.remove(stale_path)

    # Internal error dictionaries are keyed by sample ID. Keep their count and
    # details, but replace user-visible keys with anonymous sequence numbers.
    anonymized_errors = {}
    for index, (sample_id, detail) in enumerate((errors or {}).items(), start=1):
        safe_detail = str(detail).replace(str(sample_id), "<utt_id>")
        anonymized_errors[f"error_{index}"] = safe_detail

    safe_message = str(message)
    for sample_id in (errors or {}):
        safe_message = safe_message.replace(str(sample_id), "<utt_id>")

    with open(os.path.join(output_dir, "validation_error.json"), "w") as f:
        json.dump(
            {
                "status": "failed",
                "error_code": code,
                "message": safe_message,
                "num_errors": len(errors or {}),
                "errors": anonymized_errors,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_checkpoint(path: str, expected_ids: set) -> Dict[str, dict]:
    """Load only valid results belonging to this exact evaluation set."""
    if not os.path.isfile(path):
        return {}
    loaded: Dict[str, dict] = {}
    try:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                sid = record.get("sample_id")
                if sid in expected_ids:
                    loaded[sid] = validate_prediction(record.get("result"))
    except Exception:
        # The exception may contain a checkpoint record or sample ID.
        print("  Warning: invalid checkpoint ignored")
        return {}
    return loaded


def append_checkpoint(path: str, sample_id: str, result: dict) -> None:
    """Append one successful result before the next queue item is processed."""
    with open(path, "a") as f:
        f.write(json.dumps(
            {"sample_id": sample_id, "result": result},
            ensure_ascii=False,
        ) + "\n")
        f.flush()
        os.fsync(f.fileno())


async def evaluate_batch(
    items: List[Tuple[str, str, str, str]],
    eval_prompt: str,
    checkpoint_path: str,
    already_completed: int,
    total_count: int,
) -> Tuple[Dict[str, dict], Dict[str, str], Dict[str, str]]:
    """Evaluate a batch with exactly MAX_WORKERS long-lived workers."""
    queue: asyncio.Queue = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    results: Dict[str, dict] = {}
    retryable_errors: Dict[str, str] = {}
    permanent_errors: Dict[str, str] = {}
    progress_lock = asyncio.Lock()
    checkpoint_lock = asyncio.Lock()
    completed = already_completed

    connector = aiohttp.TCPConnector(
        limit=MAX_WORKERS,
        force_close=True,
    )

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=1000),
    ) as session:

        async def worker() -> None:
            nonlocal completed
            while True:
                try:
                    lang, sid, wav_path, source_text = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

                try:
                    prediction = await evaluate_one(
                        session, wav_path, source_text, eval_prompt
                    )
                    results[sid] = prediction
                    retryable_errors.pop(sid, None)
                    async with checkpoint_lock:
                        try:
                            append_checkpoint(checkpoint_path, sid, prediction)
                        except OSError as exc:
                            raise FatalEvaluationError(
                                "Checkpoint output cannot be written"
                            ) from exc
                    async with progress_lock:
                        completed += 1
                        if completed % 200 == 0 or completed == total_count:
                            print(
                                f"  [{completed}/{total_count}] done, "
                                f"{len(retryable_errors)} pending retries"
                            )
                except PermanentSampleError as exc:
                    permanent_errors[sid] = f"{type(exc).__name__}: {exc}"
                except FatalEvaluationError:
                    raise
                except Exception as exc:
                    # Network, rate-limit, malformed model output, and other
                    # non-sample failures are recoverable and must be retried.
                    retryable_errors[sid] = f"{type(exc).__name__}: {exc}"
                finally:
                    queue.task_done()

        workers = [
            asyncio.create_task(worker())
            for _ in range(min(MAX_WORKERS, len(items)))
        ]
        if workers:
            await asyncio.gather(*workers)

    return results, retryable_errors, permanent_errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    random.seed(GLOBAL_SEED)

    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <ref_dir> <res_dir> <output_dir>", file=sys.stderr)
        return 1

    ref_dir = sys.argv[1]
    res_dir = sys.argv[2]
    output_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)

    if not API_KEY:
        write_error_outputs(
            output_dir, "MISSING_API_KEY", "GEMINI_API_KEY is not set"
        )
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        return 1

    # ---- 1. Load data ----
    print("Loading prepared data ...")
    try:
        ref_data = load_prepared_data(ref_dir)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        message = "Reference data cannot be loaded or has an invalid format."
        write_error_outputs(output_dir, "REFERENCE_DATA_ERROR", message)
        print("ERROR TYPE: REFERENCE_DATA_ERROR", file=sys.stderr)
        print(f"ERROR: {message}", file=sys.stderr)
        return 1

    if not ref_data:
        message = "No prepared_data files were found in the reference input."
        write_error_outputs(output_dir, "REFERENCE_DATA_ERROR", message)
        print("ERROR TYPE: REFERENCE_DATA_ERROR", file=sys.stderr)
        print(f"ERROR: {message}", file=sys.stderr)
        return 1

    missing_reference_languages = [
        lang for lang in ("zh", "en") if not ref_data.get(lang)
    ]
    if missing_reference_languages:
        message = (
            "Reference data is incomplete: both ZH and EN subsets must be "
            "present and non-empty."
        )
        write_error_outputs(output_dir, "REFERENCE_DATA_ERROR", message)
        print("ERROR TYPE: REFERENCE_DATA_ERROR", file=sys.stderr)
        print(f"ERROR: {message}", file=sys.stderr)
        return 1

    expected_count = sum(len(samples) for samples in ref_data.values())
    print(
        f"  Reference : {expected_count} samples "
        f"({', '.join(f'{lang.upper()}: {len(d)}' for lang, d in ref_data.items())})"
    )

    print("Finding WAV files ...")
    wav_map = find_wavs(res_dir)
    submission_diagnostics = inspect_submission_files(res_dir)
    print(f"  Submitted : {len(wav_map)} supported audio IDs")
    if submission_diagnostics["nested_supported_audio"]:
        print(
            f"  Found recursively : "
            f"{submission_diagnostics['nested_supported_audio']} supported "
            "audio file(s) in subdirectories"
        )
    if submission_diagnostics["invalid_zip_files"]:
        print(
            "  Warning type: INVALID_ZIP_ARCHIVE — "
            f"{submission_diagnostics['invalid_zip_files']} invalid ZIP "
            "archive(s)"
        )
    if submission_diagnostics["empty_audio_zips"]:
        print(
            "  Warning type: EMPTY_AUDIO_ARCHIVE — "
            f"{submission_diagnostics['empty_audio_zips']} ZIP archive(s) "
            "contain no supported audio"
        )

    # ---- 2. Match samples ----
    eval_items = []  # [(lang, sample_id, audio_path, source_text)]
    missing = []
    for lang, samples in ref_data.items():
        for sid, info in samples.items():
            if sid in wav_map:
                eval_items.append((lang, sid, wav_map[sid], info["text_with_mark"]))
            else:
                missing.append(f"{lang}/{sid}")

    expected_ids = {item[1] for item in eval_items}
    all_reference_ids = {
        sid for samples in ref_data.values() for sid in samples
    }

    unsupported_matching = [
        (stem, ext)
        for stem, ext in submission_diagnostics.get(
            "unsupported_records", []
        )
        if stem in all_reference_ids
    ]
    unsupported_matching_extensions: Dict[str, int] = defaultdict(int)
    for _stem, ext in unsupported_matching:
        unsupported_matching_extensions[ext] += 1
    submission_diagnostics["unsupported_matching_required_ids"] = len(
        unsupported_matching
    )
    submission_diagnostics["unsupported_matching_extensions"] = dict(
        unsupported_matching_extensions
    )
    submission_diagnostics["empty_matching_required_ids"] = len(
        [
            stem
            for stem in submission_diagnostics.get("empty_audio_stems", [])
            if stem in all_reference_ids
        ]
    )
    submission_diagnostics["duplicate_matching_required_ids"] = len(
        set(submission_diagnostics.get("duplicate_stems", set()))
        & all_reference_ids
    )

    unexpected = sorted(set(wav_map) - all_reference_ids)

    print("=" * 60)
    print("  PRE-VALIDATION")
    print("=" * 60)
    if unexpected:
        print(
            "  Warning type: INVALID_AUDIO_IDS — "
            f"{len(unexpected)} submitted audio filename stem(s) do not "
            "match any reference ID"
        )
    if submission_diagnostics["unsupported_matching_required_ids"]:
        print(
            "  Warning type: UNSUPPORTED_FILE_FORMAT — "
            f"{submission_diagnostics['unsupported_matching_required_ids']} "
            "required sample file(s) use an unsupported extension"
        )
    elif not wav_map and submission_diagnostics["unsupported_files"]:
        extension_summary = format_extension_summary(
            submission_diagnostics["unsupported_extensions"]
        )
        suffix = f" ({extension_summary})" if extension_summary else ""
        print(
            "  Warning type: UNSUPPORTED_FILE_FORMAT — "
            f"{submission_diagnostics['unsupported_files']} unsupported "
            f"file(s){suffix}"
        )
    if submission_diagnostics["duplicate_matching_required_ids"]:
        print(
            "  Warning type: DUPLICATE_AUDIO_IDS — "
            f"{submission_diagnostics['duplicate_matching_required_ids']} "
            "required sample ID(s) occur more than once"
        )
    if submission_diagnostics["empty_matching_required_ids"]:
        print(
            "  Warning type: EMPTY_AUDIO_FILE — "
            f"{submission_diagnostics['empty_matching_required_ids']} "
            "required audio file(s) are empty"
        )
    if submission_diagnostics["encrypted_audio_entries"]:
        print(
            "  Warning type: ENCRYPTED_ZIP_ARCHIVE — "
            f"{submission_diagnostics['encrypted_audio_entries']} encrypted "
            "audio file(s) found inside ZIP archive(s)"
        )
    print(f"  Reference : {expected_count} samples")
    print(f"  Matched   : {len(eval_items)} samples")
    print(f"  Missing   : {len(missing)}")

    has_blocking_prevalidation_error = bool(
        missing
        or submission_diagnostics["duplicate_matching_required_ids"]
        or submission_diagnostics["empty_matching_required_ids"]
        or submission_diagnostics["encrypted_audio_entries"]
    )

    if has_blocking_prevalidation_error:
        error_code, message = classify_prevalidation_error(
            diagnostics=submission_diagnostics,
            submitted_count=len(wav_map),
            matched_count=len(eval_items),
            missing_count=len(missing),
            unexpected_count=len(unexpected),
        )
        write_error_outputs(output_dir, error_code, message)
        print("  PRE-VALIDATION FAILED")
        print(f"  Error type: {error_code}")
        print(f"  Details   : {message}")
        print("=" * 60)
        print(f"ERROR TYPE: {error_code}", file=sys.stderr)
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    print("  PRE-VALIDATION PASSED")
    print("=" * 60)

    # ---- 3. Run Gemini evaluation ----
    eval_prompt = build_evaluation_prompt()
    print(f"\nStarting Gemini evaluation ({len(eval_items)} samples, {MAX_WORKERS} workers) ...")
    t_start = time.time()

    # LALM key → Track2 component mapping
    LALM_KEY_MAP = {
        "nvv_accuracy_score": "A",
        "nvv_pe_score": "P",
        "overall_naturalness_score": "N",
        "overall_quality_score": "Q",
        "overall_expression_score": "E",
    }

    checkpoint_path = os.path.join(output_dir, "gemini_checkpoint.jsonl")
    predictions = load_checkpoint(checkpoint_path, expected_ids)
    if predictions:
        print(f"  Resuming from checkpoint: {len(predictions)} valid samples")

    pending = [item for item in eval_items if item[1] not in predictions]
    retryable_errors: Dict[str, str] = {}
    permanent_errors: Dict[str, str] = {}

    retry_items = pending
    recovery_round = 0
    while retry_items:
        if recovery_round:
            print(
                f"\nRecovery round {recovery_round}: retrying "
                f"{len(retry_items)} sample(s) after "
                f"{RECOVERY_COOLDOWN:.0f}s cooldown"
            )
            time.sleep(RECOVERY_COOLDOWN)

        try:
            recovered, retryable_errors, batch_permanent = asyncio.run(
                evaluate_batch(
                    retry_items,
                    eval_prompt,
                    checkpoint_path,
                    already_completed=len(predictions),
                    total_count=len(eval_items),
                )
            )
        except FatalEvaluationError as exc:
            write_error_outputs(
                output_dir, "FATAL_EVALUATION_ERROR", str(exc)
            )
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        predictions.update(recovered)
        permanent_errors.update(batch_permanent)
        if permanent_errors:
            break

        retry_ids = set(retryable_errors)
        retry_items = [
            item for item in eval_items if item[1] in retry_ids
        ]
        recovery_round += 1
        if retry_items and recovery_round >= MAX_RECOVERY_ROUNDS:
            print(
                f"  Error type: EVALUATION_RETRIES_EXHAUSTED — "
                f"{len(retry_items)} sample(s) remain unscored after "
                f"{MAX_RECOVERY_ROUNDS} recovery round(s).",
                file=sys.stderr,
            )
            break

    elapsed = time.time() - t_start
    print(
        f"\nEvaluation complete in {elapsed:.0f}s: "
        f"{len(predictions)}/{len(eval_items)} scored, "
        f"{len(permanent_errors)} permanent sample errors"
    )

    # ---- 4. Require complete results before computing any score ----
    missing_results = sorted(all_reference_ids - set(predictions))
    if permanent_errors or missing_results or len(predictions) != expected_count:
        error_code = (
            "INVALID_SAMPLE" if permanent_errors
            else "EVALUATION_RETRIES_EXHAUSTED"
        )
        message = (
            f"{len(missing_results)} sample(s) have no valid Gemini result. "
            f"All {expected_count} samples must be scored successfully. "
            f"Permanent sample errors: {len(permanent_errors)}."
        )
        write_error_outputs(
            output_dir, error_code, message, permanent_errors
        )
        print(f"  VALIDATION FAILED — {error_code}")
        print(f"  Error type: {error_code}", file=sys.stderr)
        print(f"  Details   : {message}", file=sys.stderr)
        return 1

    per_lang_scores = {
        "zh": {"A": [], "P": [], "N": [], "Q": [], "E": []},
        "en": {"A": [], "P": [], "N": [], "Q": [], "E": []},
    }
    sid_to_lang = {
        sid: lang for lang, sid, _wav_path, _source_text in eval_items
    }
    for sid, prediction in predictions.items():
        lang = sid_to_lang[sid]
        for json_key, component in LALM_KEY_MAP.items():
            per_lang_scores[lang][component].append(
                float(prediction[json_key])
            )

    expected_lang_counts = {
        lang: len(samples) for lang, samples in ref_data.items()
    }
    bad_counts = []
    for lang in ("zh", "en"):
        for component in ("A", "P", "N", "Q", "E"):
            actual = len(per_lang_scores[lang][component])
            expected = expected_lang_counts.get(lang, 0)
            if actual != expected:
                bad_counts.append(
                    f"{lang}/{component}: {actual}, expected {expected}"
                )
    if bad_counts:
        message = "Incomplete metric counts: " + "; ".join(bad_counts)
        write_error_outputs(
            output_dir, "INCOMPLETE_METRICS", message
        )
        print(f"ERROR: {message}", file=sys.stderr)
        return 1

    # ---- 5. Compute scores only after completeness validation ----
    result = compute_scores(per_lang_scores)

    # ---- 6. Write outputs ----
    stale_error_path = os.path.join(output_dir, "validation_error.json")
    if os.path.isfile(stale_error_path):
        os.remove(stale_error_path)

    scores_out = {
        "Track2Score_ZH": result["Track2Score_ZH"],
        "Track2Score_EN": result["Track2Score_EN"],
        "FinalTrack2Score": result["FinalTrack2Score"],
    }
    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(scores_out, f)

    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump({
            "status": "success",
            "scores": scores_out,
            "per_lang_counts": {
                lang: {k: len(v) for k, v in comps.items()}
                for lang, comps in per_lang_scores.items()
            },
            "num_errors": 0,
            "num_scored_samples": len(predictions),
            "elapsed_s": round(elapsed, 1),
            "details": result.get("details", {}),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nTrack2Score_ZH    : {scores_out['Track2Score_ZH']:.6f}")
    print(f"Track2Score_EN    : {scores_out['Track2Score_EN']:.6f}")
    print(f"FinalTrack2Score  : {scores_out['FinalTrack2Score']:.6f}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        if len(sys.argv) >= 4:
            write_error_outputs(
                sys.argv[3],
                "INTERNAL_SCORING_ERROR",
                f"Scoring failed because of an internal {type(exc).__name__}.",
            )
        print(
            f"ERROR TYPE: INTERNAL_SCORING_ERROR ({type(exc).__name__})",
            file=sys.stderr,
        )
        raise SystemExit(1)
