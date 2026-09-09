#!/usr/bin/env python3
"""
Track 2 Official Scoring Module.

Implements the NVVSpeech Challenge Track 2 scoring formula:

    Track2Score = 100 * (0.30*A + 0.25*P + 0.15*N + 0.15*Q + 0.15*E)

where all component scores are rated on a five-point scale [1,5] and
normalized to [0,1] via (score - 1) / 4.

The final bilingual score is:

    FinalTrack2Score = (Track2Score_ZH + Track2Score_EN) / 2

Reference: NVV-SuperBench (https://github.com/lmxue/NVV-SuperBench)
"""

import json
import re
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Official Track 2 scoring weights
# ============================================================
WEIGHTS = {
    "A": 0.30,  # NVV Accuracy
    "P": 0.25,  # NVV Perceptual Effect
    "N": 0.15,  # Overall Naturalness
    "Q": 0.15,  # Overall Quality
    "E": 0.15,  # Overall Expression
}

# Component names
COMPONENT_NAMES = {
    "A": "NVV Accuracy",
    "P": "NVV Perceptual Effect",
    "N": "Overall Naturalness",
    "Q": "Overall Quality",
    "E": "Overall Expression",
}

# Tie-breaking priority order
TIE_BREAK_ORDER = ["A", "P", "N"]


def normalize_score(score_1_to_5: float) -> float:
    """Normalize a 1-5 scale score to [0, 1].

    Formula: (score - 1) / 4
    Score of 0 (absent/inaudible) maps to 0.
    """
    if score_1_to_5 <= 0:
        return 0.0
    return max(0.0, min(1.0, (score_1_to_5 - 1.0) / 4.0))


def compute_track2_score(
    A: float,
    P: float,
    N: float,
    Q: float,
    E: float,
    normalize: bool = True,
) -> Dict[str, Any]:
    """Compute the Track 2 score from five component scores.

    Args:
        A: NVV Accuracy (1-5 scale, 0 if absent)
        P: NVV Perceptual Effect (1-5 scale, 0 if absent)
        N: Overall Naturalness (1-5 scale)
        Q: Overall Quality (1-5 scale)
        E: Overall Expression (1-5 scale)
        normalize: If True, normalize scores from [1,5] to [0,1].

    Returns:
        Dict with raw/normalized scores and final Track2Score.
    """
    if normalize:
        A_norm = normalize_score(A)
        P_norm = normalize_score(P)
        N_norm = normalize_score(N)
        Q_norm = normalize_score(Q)
        E_norm = normalize_score(E)
    else:
        A_norm, P_norm, N_norm, Q_norm, E_norm = A, P, N, Q, E

    raw_score = (
        WEIGHTS["A"] * A_norm
        + WEIGHTS["P"] * P_norm
        + WEIGHTS["N"] * N_norm
        + WEIGHTS["Q"] * Q_norm
        + WEIGHTS["E"] * E_norm
    )

    track2_score = 100.0 * raw_score

    return {
        "components": {
            "A": {"name": "NVV Accuracy", "raw": A, "normalized": A_norm, "weight": WEIGHTS["A"]},
            "P": {"name": "NVV Perceptual Effect", "raw": P, "normalized": P_norm, "weight": WEIGHTS["P"]},
            "N": {"name": "Overall Naturalness", "raw": N, "normalized": N_norm, "weight": WEIGHTS["N"]},
            "Q": {"name": "Overall Quality", "raw": Q, "normalized": Q_norm, "weight": WEIGHTS["Q"]},
            "E": {"name": "Overall Expression", "raw": E, "normalized": E_norm, "weight": WEIGHTS["E"]},
        },
        "weighted_sum": raw_score,
        "track2_score": track2_score,
    }


def compute_final_bilingual_score(score_zh: float, score_en: float) -> Dict[str, Any]:
    """Compute the final bilingual Track 2 score.

    FinalTrack2Score = (Track2Score_ZH + Track2Score_EN) / 2
    """
    final = (score_zh + score_en) / 2.0
    return {
        "track2_score_zh": score_zh,
        "track2_score_en": score_en,
        "final_track2_score": final,
    }


# ============================================================
# LALM result parsing (from llm_eval.py output JSON)
# ============================================================

# Mapping from LLM eval JSON keys to Track 2 components
LALM_KEY_TO_COMPONENT_TAG = {
    "nvc_accuracy_score": "A",
    "nvc_pe_score": "P",
    "overall_naturalness_score": "N",
    "overall_quality_score": "Q",
    "overall_expression_score": "E",
    "nvc_if_score": "A",  # prompt-based: Instruction Following → Accuracy
    "cam_score": "E",     # prompt-based: Caption-Audio Match → Expression
}

LALM_KEY_TO_COMPONENT_PROMPT = {
    "nvc_if_score": "A",
    "nvc_pe_score": "P",
    "overall_naturalness_score": "N",
    "overall_quality_score": "Q",
    "cam_score": "E",
}


def parse_lalm_results(
    eval_json_path: str,
    task_mode: str = "tag",
) -> Dict[str, Any]:
    """Parse LALM evaluation results and compute Track 2 scores.

    Args:
        eval_json_path: Path to llm_eval.py output JSON.
        task_mode: "tag" or "prompt".

    Returns:
        Dict with per-sample and aggregate Track 2 scores.
    """
    with open(eval_json_path, "r") as f:
        data = json.load(f)

    items = data.get("items", {})
    if not isinstance(items, dict):
        raise ValueError(f"Unexpected format in {eval_json_path}: items must be a dict")

    # Collect per-sample scores
    per_sample_scores: Dict[str, Dict[str, float]] = {}
    component_sums: Dict[str, float] = defaultdict(float)
    component_counts: Dict[str, int] = defaultdict(int)

    key_mapping = LALM_KEY_TO_COMPONENT_TAG if task_mode == "tag" else LALM_KEY_TO_COMPONENT_PROMPT

    for audio_path, rec in items.items():
        if rec.get("status") != "ok":
            continue

        pred = rec.get("prediction_capped") or rec.get("prediction") or {}
        sample_id = rec.get("ground_truth", {}).get("sample_id", audio_path)

        scores_for_sample: Dict[str, float] = {}
        for json_key, component in key_mapping.items():
            val = pred.get(json_key)
            if isinstance(val, (int, float)):
                scores_for_sample[component] = float(val)
                component_sums[component] += float(val)
                component_counts[component] += 1

        if scores_for_sample:
            per_sample_scores[sample_id] = scores_for_sample

    # Compute mean component scores
    mean_components: Dict[str, float] = {}
    for comp in ["A", "P", "N", "Q", "E"]:
        if component_counts[comp] > 0:
            mean_components[comp] = component_sums[comp] / component_counts[comp]
        else:
            mean_components[comp] = 0.0

    # Compute Track 2 score from means
    track2_result = compute_track2_score(
        A=mean_components.get("A", 0),
        P=mean_components.get("P", 0),
        N=mean_components.get("N", 0),
        Q=mean_components.get("Q", 0),
        E=mean_components.get("E", 0),
        normalize=True,
    )

    return {
        "source": eval_json_path,
        "task_mode": task_mode,
        "num_samples": len(per_sample_scores),
        "mean_components": mean_components,
        "component_counts": dict(component_counts),
        **track2_result,
        "per_sample_scores": per_sample_scores,
    }


# ============================================================
# Objective metrics → Track 2 component mapping
# ============================================================

def dnsmos_to_quality(dnsmos_ovrl: float) -> float:
    """Map DNSMOS OVRL score (~1-5) to Quality component.

    DNSMOS OVRL is roughly on a 1-5 scale. We keep it as-is
    for the Track 2 formula.
    """
    return max(1.0, min(5.0, dnsmos_ovrl))


def cer_to_naturalness(cer: float) -> float:
    """Map CER (0-1+, lower is better) to Naturalness (1-5, higher is better).

    CER 0%   → Naturalness 5
    CER 5%   → Naturalness 4
    CER 15%  → Naturalness 3
    CER 30%  → Naturalness 2
    CER 50%+ → Naturalness 1
    """
    cer_pct = cer * 100
    if cer_pct <= 2:
        return 5.0
    elif cer_pct <= 8:
        return 4.0
    elif cer_pct <= 20:
        return 3.0
    elif cer_pct <= 40:
        return 2.0
    else:
        return 1.0


def clap_to_expression(clap_score: float) -> float:
    """Map CLAP score (~0-1) to Expression (1-5).

    CLAP ≥ 0.4  → Expression 5
    CLAP ≥ 0.3  → Expression 4
    CLAP ≥ 0.2  → Expression 3
    CLAP ≥ 0.1  → Expression 2
    CLAP < 0.1  → Expression 1
    """
    if clap_score >= 0.40:
        return 5.0
    elif clap_score >= 0.30:
        return 4.0
    elif clap_score >= 0.20:
        return 3.0
    elif clap_score >= 0.10:
        return 2.0
    else:
        return 1.0


def nvv_f1_to_accuracy_perceptual(f1: float, coverage: float) -> Tuple[float, float]:
    """Map NVV F1/Coverage metrics to Accuracy (A) and Perceptual Effect (P).

    Accuracy (A) is approximated by Coverage-Adjusted F1:
      CA-F1 ≥ 0.8  → A=5
      CA-F1 ≥ 0.6  → A=4
      CA-F1 ≥ 0.4  → A=3
      CA-F1 ≥ 0.2  → A=2
      CA-F1 < 0.2  → A=1

    Perceptual Effect (P) is approximated by F1 score:
      F1 ≥ 0.8  → P=5
      F1 ≥ 0.6  → P=4
      F1 ≥ 0.4  → P=3
      F1 ≥ 0.2  → P=2
      F1 < 0.2  → P=1
    """
    # Accuracy from CA-F1
    if f1 >= 0.8:
        A = 5.0
    elif f1 >= 0.6:
        A = 4.0
    elif f1 >= 0.4:
        A = 3.0
    elif f1 >= 0.2:
        A = 2.0
    else:
        A = 1.0

    # Perceptual Effect from F1
    if f1 >= 0.8:
        P = 5.0
    elif f1 >= 0.6:
        P = 4.0
    elif f1 >= 0.4:
        P = 3.0
    elif f1 >= 0.2:
        P = 2.0
    else:
        P = 1.0

    # Adjust by coverage: if coverage is low, cap the scores
    if coverage < 0.5:
        A = min(A, 3.0)
        P = min(P, 3.0)
    if coverage < 0.3:
        A = min(A, 2.0)
        P = min(P, 2.0)

    return A, P


# ============================================================
# Tie-breaking
# ============================================================

def compare_systems(scores_a: Dict[str, float], scores_b: Dict[str, float]) -> int:
    """Compare two systems by Track 2 tie-breaking rules.

    Returns:
        -1 if A wins, 1 if B wins, 0 if tied on all criteria.
    """
    # Primary: Final Track2Score
    if scores_a["final_track2_score"] > scores_b["final_track2_score"]:
        return -1
    elif scores_a["final_track2_score"] < scores_b["final_track2_score"]:
        return 1

    # Tie-breaking: A → P → N
    for comp in TIE_BREAK_ORDER:
        a_val = scores_a.get(f"score_{comp}", 0)
        b_val = scores_b.get(f"score_{comp}", 0)
        if a_val > b_val:
            return -1
        elif a_val < b_val:
            return 1

    return 0


def rank_systems(system_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank systems by Track 2 score with tie-breaking.

    Each item in system_scores should have:
      - system_name: str
      - final_track2_score: float
      - track2_score_zh: float
      - track2_score_en: float
      - score_A, score_P, score_N: float (for tie-breaking)
    """
    from functools import cmp_to_key

    def cmp(a, b):
        return compare_systems(a, b)

    ranked = sorted(system_scores, key=cmp_to_key(cmp))
    for i, s in enumerate(ranked):
        s["rank"] = i + 1
    return ranked


# ============================================================
# Utility: Summary report generation
# ============================================================

def print_track2_report(result: Dict[str, Any], title: str = "Track 2 Score Report") -> None:
    """Pretty-print a Track 2 score report."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    if "num_samples" in result:
        print(f"  Samples evaluated: {result['num_samples']}")

    components = result.get("components", {})
    if components:
        print(f"\n  {'Component':<30} {'Raw':>6} {'Norm':>6} {'Weight':>8} {'Contrib':>8}")
        print(f"  {'-'*60}")
        for comp_key in ["A", "P", "N", "Q", "E"]:
            c = components.get(comp_key, {})
            raw = c.get("raw", 0)
            norm = c.get("normalized", 0)
            weight = c.get("weight", 0)
            contrib = weight * norm * 100
            print(f"  {c.get('name', comp_key):<30} {raw:>6.2f} {norm:>6.4f} {weight:>8.2f} {contrib:>8.2f}")

        print(f"  {'-'*60}")
        print(f"  {'Weighted Sum':<30} {'':>6} {'':>6} {'':>8} {result.get('weighted_sum', 0)*100:>8.2f}")

    track2 = result.get("track2_score", 0)
    print(f"\n  >>> Track2Score = {track2:.4f}")

    if "final_track2_score" in result:
        print(f"  >>> Final Bilingual Track2Score = {result['final_track2_score']:.4f}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Quick test
    result = compute_track2_score(A=4.0, P=3.5, N=4.0, Q=4.5, E=3.5)
    print_track2_report(result, "Track 2 Score Example")

    bilingual = compute_final_bilingual_score(
        score_zh=result["track2_score"],
        score_en=result["track2_score"] * 0.9,
    )
    print(f"Bilingual: ZH={bilingual['track2_score_zh']:.4f}, "
          f"EN={bilingual['track2_score_en']:.4f}, "
          f"Final={bilingual['final_track2_score']:.4f}")
