"""
Clinical configuration for the Holter / Comprehensive ECG pipeline.

This layer externalizes the clinically important thresholds so the analysis
code can be tuned per deployment without editing Python source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import logging
import yaml

_logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    # First check physical parent containing src (parents[3])
    try_path = Path(__file__).resolve().parents[3]
    if (try_path / "config").exists():
        return try_path
    # Fallback to parents[4] / "qww_new"
    fallback_path = Path(__file__).resolve().parents[4] / "qww_new"
    if (fallback_path / "config").exists():
        return fallback_path
    return try_path


def default_clinical_config_path() -> Path:
    import sys
    # 1. PyInstaller frozen path resolution
    if getattr(sys, "frozen", False):
        base_dir = Path(getattr(sys, "_MEIPASS", os.getcwd()))
        return base_dir / "config" / "clinical_config.yaml"

    # 2. Dynamic execution path resolution
    return _repo_root() / "config" / "clinical_config.yaml"


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_tuple(value: Any, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        return tuple(item for item in parts if item)
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return tuple(default)


@dataclass(frozen=True)
class ClinicalConfig:
    sampling_rate_hz: float = 500.0
    min_signal_seconds: float = 2.0
    primary_detection_leads: Tuple[str, ...] = ("II", "V5")

    signal_quality_threshold: float = 0.6
    signal_flatline_std: float = 0.05
    signal_saturation_mv: float = 4.5
    signal_saturation_ratio: float = 0.01
    signal_baseline_wander_ratio: float = 0.30
    signal_hf_noise_ratio: float = 0.40
    signal_regularity_cv: float = 0.8

    tachy_threshold_bpm: float = 100.0
    brady_threshold_bpm: float = 60.0
    pause_threshold_ms: float = 2000.0

    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    notch_hz: float = 50.0

    qrs_validator_min_width_ms: float = 25.0
    qrs_validator_max_width_ms: float = 200.0
    qrs_validator_min_amplitude: float = 0.04

    template_similarity_threshold: float = 0.90
    qrs_detection_window_ms: float = 150.0
    qrs_detection_refractory_ms: float = 120.0
    qrs_refine_radius_ms: float = 80.0
    qrs_detection_threshold_scale: float = 0.25

    extra: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    @classmethod
    def from_mapping(cls, data: Dict[str, Any] | None = None) -> "ClinicalConfig":
        data = dict(data or {})

        signal = dict(data.get("signal") or {})
        arrhythmia = dict(data.get("arrhythmia") or {})
        qrs_validator = dict(data.get("qrs_validator") or {})
        quality = dict(data.get("quality") or {})

        sampling_rate_hz = _as_float(data.get("sampling_rate_hz"), 500.0)
        min_signal_seconds = _as_float(signal.get("min_signal_seconds", data.get("min_signal_seconds")), 2.0)
        primary_detection_leads = _as_tuple(
            signal.get("primary_detection_leads", data.get("primary_detection_leads")),
            ("II", "V5"),
        )

        signal_quality_threshold = _as_float(signal.get("quality_threshold", data.get("signal_quality_threshold")), 0.6)
        signal_flatline_std = _as_float(quality.get("flatline_std", signal.get("flatline_std")), 0.05)
        signal_saturation_mv = _as_float(quality.get("saturation_mv", signal.get("saturation_mv")), 4.5)
        signal_saturation_ratio = _as_float(quality.get("saturation_ratio", signal.get("saturation_ratio")), 0.01)
        signal_baseline_wander_ratio = _as_float(quality.get("baseline_wander_ratio", signal.get("baseline_wander_ratio")), 0.30)
        signal_hf_noise_ratio = _as_float(quality.get("hf_noise_ratio", signal.get("hf_noise_ratio")), 0.40)
        signal_regularity_cv = _as_float(quality.get("regularity_cv", signal.get("regularity_cv")), 0.8)

        tachy_threshold_bpm = _as_float(arrhythmia.get("tachy_threshold_bpm", data.get("tachy_threshold_bpm")), 100.0)
        brady_threshold_bpm = _as_float(arrhythmia.get("brady_threshold_bpm", data.get("brady_threshold_bpm")), 60.0)
        pause_threshold_ms = _as_float(arrhythmia.get("pause_threshold_ms", data.get("pause_threshold_ms")), 2000.0)

        bandpass_low_hz = _as_float(signal.get("bandpass_low_hz", data.get("bandpass_low_hz")), 0.5)
        bandpass_high_hz = _as_float(signal.get("bandpass_high_hz", data.get("bandpass_high_hz")), 40.0)
        notch_hz = _as_float(signal.get("notch_hz", data.get("notch_hz")), 50.0)

        qrs_validator_min_width_ms = _as_float(qrs_validator.get("min_width_ms", data.get("qrs_validator_min_width_ms")), 25.0)
        qrs_validator_max_width_ms = _as_float(qrs_validator.get("max_width_ms", data.get("qrs_validator_max_width_ms")), 200.0)
        qrs_validator_min_amplitude = _as_float(qrs_validator.get("min_amplitude", data.get("qrs_validator_min_amplitude")), 0.04)

        template_similarity_threshold = _as_float(
            data.get("template_similarity_threshold"),
            0.90,
        )

        qrs_detection_window_ms = _as_float(arrhythmia.get("qrs_detection_window_ms", data.get("qrs_detection_window_ms")), 150.0)
        qrs_detection_refractory_ms = _as_float(arrhythmia.get("qrs_detection_refractory_ms", data.get("qrs_detection_refractory_ms")), 120.0)
        qrs_refine_radius_ms = _as_float(arrhythmia.get("qrs_refine_radius_ms", data.get("qrs_refine_radius_ms")), 80.0)
        qrs_detection_threshold_scale = _as_float(arrhythmia.get("threshold_scale", data.get("qrs_detection_threshold_scale")), 0.25)

        known_keys = {
            "sampling_rate_hz",
            "signal",
            "arrhythmia",
            "qrs_validator",
            "quality",
            "min_signal_seconds",
            "primary_detection_leads",
            "signal_quality_threshold",
            "tachy_threshold_bpm",
            "brady_threshold_bpm",
            "pause_threshold_ms",
            "bandpass_low_hz",
            "bandpass_high_hz",
            "notch_hz",
            "qrs_validator_min_width_ms",
            "qrs_validator_max_width_ms",
            "qrs_validator_min_amplitude",
            "template_similarity_threshold",
            "qrs_detection_window_ms",
            "qrs_detection_refractory_ms",
            "qrs_refine_radius_ms",
            "qrs_detection_threshold_scale",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            sampling_rate_hz=sampling_rate_hz,
            min_signal_seconds=min_signal_seconds,
            primary_detection_leads=primary_detection_leads,
            signal_quality_threshold=signal_quality_threshold,
            signal_flatline_std=signal_flatline_std,
            signal_saturation_mv=signal_saturation_mv,
            signal_saturation_ratio=signal_saturation_ratio,
            signal_baseline_wander_ratio=signal_baseline_wander_ratio,
            signal_hf_noise_ratio=signal_hf_noise_ratio,
            signal_regularity_cv=signal_regularity_cv,
            tachy_threshold_bpm=tachy_threshold_bpm,
            brady_threshold_bpm=brady_threshold_bpm,
            pause_threshold_ms=pause_threshold_ms,
            bandpass_low_hz=bandpass_low_hz,
            bandpass_high_hz=bandpass_high_hz,
            notch_hz=notch_hz,
            qrs_validator_min_width_ms=qrs_validator_min_width_ms,
            qrs_validator_max_width_ms=qrs_validator_max_width_ms,
            qrs_validator_min_amplitude=qrs_validator_min_amplitude,
            template_similarity_threshold=template_similarity_threshold,
            qrs_detection_window_ms=qrs_detection_window_ms,
            qrs_detection_refractory_ms=qrs_detection_refractory_ms,
            qrs_refine_radius_ms=qrs_refine_radius_ms,
            qrs_detection_threshold_scale=qrs_detection_threshold_scale,
            extra=extra,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sampling_rate_hz": self.sampling_rate_hz,
            "min_signal_seconds": self.min_signal_seconds,
            "primary_detection_leads": list(self.primary_detection_leads),
            "signal_quality_threshold": self.signal_quality_threshold,
            "signal_flatline_std": self.signal_flatline_std,
            "signal_saturation_mv": self.signal_saturation_mv,
            "signal_saturation_ratio": self.signal_saturation_ratio,
            "signal_baseline_wander_ratio": self.signal_baseline_wander_ratio,
            "signal_hf_noise_ratio": self.signal_hf_noise_ratio,
            "signal_regularity_cv": self.signal_regularity_cv,
            "tachy_threshold_bpm": self.tachy_threshold_bpm,
            "brady_threshold_bpm": self.brady_threshold_bpm,
            "pause_threshold_ms": self.pause_threshold_ms,
            "bandpass_low_hz": self.bandpass_low_hz,
            "bandpass_high_hz": self.bandpass_high_hz,
            "notch_hz": self.notch_hz,
            "qrs_validator_min_width_ms": self.qrs_validator_min_width_ms,
            "qrs_validator_max_width_ms": self.qrs_validator_max_width_ms,
            "qrs_validator_min_amplitude": self.qrs_validator_min_amplitude,
            "template_similarity_threshold": self.template_similarity_threshold,
            "qrs_detection_window_ms": self.qrs_detection_window_ms,
            "qrs_detection_refractory_ms": self.qrs_detection_refractory_ms,
            "qrs_refine_radius_ms": self.qrs_refine_radius_ms,
            "qrs_detection_threshold_scale": self.qrs_detection_threshold_scale,
            **dict(self.extra),
        }


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        _logger.warning(
            "[ClinicalConfig] clinical_config.yaml NOT FOUND at %s -- "
            "using Python hardcoded defaults. Arrhythmia detection thresholds "
            "may differ from expected clinical values.",
            path,
        )
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            _logger.warning("[ClinicalConfig] clinical_config.yaml is not a valid mapping -- using defaults.")
            return {}
        _logger.info("[ClinicalConfig] Loaded clinical_config.yaml from %s", path)
        return data
    except Exception as exc:
        _logger.error("[ClinicalConfig] Failed to load clinical_config.yaml: %s -- using defaults.", exc)
        return {}


@lru_cache(maxsize=8)
def load_clinical_config(path: str | None = None) -> ClinicalConfig:
    """
    Load the clinical configuration from YAML, with env override support.

    Order of precedence:
      1. explicit path argument
      2. CARDIOX_CLINICAL_CONFIG env var
      3. bundled qww_new/config/clinical_config.yaml
    """
    selected = path or os.environ.get("CARDIOX_CLINICAL_CONFIG") or str(default_clinical_config_path())
    mapping = _load_yaml_config(Path(selected))
    return ClinicalConfig.from_mapping(mapping)


def reload_clinical_config(path: str | None = None) -> ClinicalConfig:
    load_clinical_config.cache_clear()
    return load_clinical_config(path)
