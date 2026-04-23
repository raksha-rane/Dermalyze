"""Model trust policy utilities for calibrated confidence and abstention."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class TrustThresholds:
    classify_confidence: float
    reject_confidence: float
    review_entropy: float
    reject_entropy: float
    review_margin: float


@dataclass(frozen=True)
class TrustConfig:
    temperature: float
    thresholds: TrustThresholds
    source: str


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _softmax_with_temperature_from_probs(
    probs: np.ndarray,
    temperature: float,
) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-6)
    safe_probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    adjusted = safe_probs ** (1.0 / safe_temperature)
    denom = float(np.sum(adjusted))
    if denom <= 0.0:
        return np.full_like(adjusted, 1.0 / len(adjusted), dtype=np.float64)
    return adjusted / denom


class ModelTrustLayer:
    """Provides calibrated confidence, uncertainty and abstention recommendation."""

    def __init__(self, config: Optional[TrustConfig] = None):
        self.config = config or TrustConfig(
            temperature=1.0,
            thresholds=TrustThresholds(
                classify_confidence=0.70,
                reject_confidence=0.45,
                review_entropy=0.55,
                reject_entropy=0.75,
                review_margin=0.15,
            ),
            source="built_in_defaults",
        )

    @classmethod
    def from_json_path(cls, config_path: Optional[Path]) -> "ModelTrustLayer":
        if config_path is None or not config_path.exists():
            return cls()

        with open(config_path, "r") as handle:
            payload = json.load(handle)

        temperature = float(payload.get("temperature", 1.0))
        thresholds = payload.get("thresholds", {})
        conformal = payload.get("conformal", {})

        conformal_threshold = float(
            conformal.get("confidence_threshold", thresholds.get("classify_confidence", 0.70))
        )
        classify_confidence = float(
            thresholds.get("classify_confidence", conformal_threshold)
        )
        reject_confidence = float(
            thresholds.get("reject_confidence", max(0.0, classify_confidence - 0.15))
        )

        config = TrustConfig(
            temperature=temperature,
            thresholds=TrustThresholds(
                classify_confidence=_clamp01(classify_confidence),
                reject_confidence=_clamp01(reject_confidence),
                review_entropy=_clamp01(float(thresholds.get("review_entropy", 0.55))),
                reject_entropy=_clamp01(float(thresholds.get("reject_entropy", 0.75))),
                review_margin=_clamp01(float(thresholds.get("review_margin", 0.15))),
            ),
            source=str(config_path),
        )
        return cls(config=config)

    def assess(self, probabilities: Dict[str, float]) -> Dict[str, Any]:
        labels = list(probabilities.keys())
        probs = np.array([float(probabilities[label]) for label in labels], dtype=np.float64)

        calibrated = _softmax_with_temperature_from_probs(probs, self.config.temperature)
        top_idx = int(np.argmax(calibrated))
        top_prob = float(calibrated[top_idx])

        sorted_probs = np.sort(calibrated)[::-1]
        second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
        margin = max(0.0, top_prob - second_prob)

        entropy = float(-np.sum(calibrated * np.log(np.clip(calibrated, 1e-12, 1.0))))
        max_entropy = math.log(max(len(calibrated), 2))
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        variation_ratio = float(1.0 - top_prob)

        thresholds = self.config.thresholds
        quality_flags: List[str] = []

        if top_prob < thresholds.classify_confidence:
            quality_flags.append("low_calibrated_confidence")
        if top_prob < thresholds.reject_confidence:
            quality_flags.append("very_low_confidence")
        if normalized_entropy > thresholds.review_entropy:
            quality_flags.append("high_predictive_entropy")
        if normalized_entropy > thresholds.reject_entropy:
            quality_flags.append("extreme_predictive_entropy")
        if margin < thresholds.review_margin:
            quality_flags.append("ambiguous_top2_margin")

        reject = (
            top_prob < thresholds.reject_confidence
            or normalized_entropy > thresholds.reject_entropy
        )
        if reject:
            recommendation = "reject"
        elif (
            top_prob >= thresholds.classify_confidence
            and normalized_entropy <= thresholds.review_entropy
            and margin >= thresholds.review_margin
        ):
            recommendation = "classify"
        else:
            recommendation = "review_required"

        uncertainty_score = _clamp01(
            0.50 * normalized_entropy + 0.30 * (1.0 - margin) + 0.20 * variation_ratio
        )

        return {
            "prediction": labels[top_idx],
            "calibrated_confidence": top_prob,
            "uncertainty": {
                "score": uncertainty_score,
                "normalized_entropy": normalized_entropy,
                "top2_margin": margin,
                "variation_ratio": variation_ratio,
            },
            "quality_flags": quality_flags,
            "recommendation": recommendation,
            "trust_metadata": {
                "temperature": float(self.config.temperature),
                "config_source": self.config.source,
            },
        }
