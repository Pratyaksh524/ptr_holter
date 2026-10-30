"""
Holter analysis pipeline helpers.

This module keeps the Holter worker focused on the architecture described in
the reference PDF:
- preprocessing
- signal quality gating
- multi-lead selection
- beat validation
- RR cleaning
- morphology clustering
- rule-based arrhythmia heuristics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .clinical_config import ClinicalConfig, load_clinical_config


def _safe_array(signal: Optional[Sequence[float]]) -> np.ndarray:
    if signal is None:
        return np.array([], dtype=float)
    arr = np.asarray(signal, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_filter(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    if signal.size < max(len(a), len(b)) * 3:
        return signal.copy()
    padlen = min(signal.size - 1, max(len(a), len(b)) * 3)
    if padlen <= 0:
        return signal.copy()
    return filtfilt(b, a, signal, padlen=padlen)


def _bandpass(signal: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    if signal.size < 3:
        return signal.copy()
    nyquist = float(fs) / 2.0
    low = max(0.001, float(low_hz) / nyquist)
    high = min(0.99, float(high_hz) / nyquist)
    if low >= high:
        return signal.copy()
    b, a = butter(order, [low, high], btype="band")
    return _safe_filter(signal, b, a)


def _highpass(signal: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    if signal.size < 3:
        return signal.copy()
    nyquist = float(fs) / 2.0
    cutoff = max(0.001, float(cutoff_hz) / nyquist)
    if cutoff >= 0.99:
        return signal.copy()
    b, a = butter(order, cutoff, btype="high")
    return _safe_filter(signal, b, a)


def _lowpass(signal: np.ndarray, fs: float, cutoff_hz: float, order: int = 2) -> np.ndarray:
    if signal.size < 3:
        return signal.copy()
    nyquist = float(fs) / 2.0
    cutoff = max(0.001, min(float(cutoff_hz) / nyquist, 0.99))
    if cutoff <= 0.001:
        return signal.copy()
    b, a = butter(order, cutoff, btype="low")
    return _safe_filter(signal, b, a)


def _normalize_template(signal: Sequence[float]) -> np.ndarray:
    arr = _safe_array(signal)
    if arr.size == 0:
        return np.array([], dtype=float)
    arr = arr - float(np.median(arr))
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 0:
        arr = arr / peak
    return arr


def _median_absolute_deviation(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    med = float(np.median(signal))
    return float(np.median(np.abs(signal - med)))


@dataclass
class HolterConfig:
    sampling_rate_hz: float = 500.0
    min_signal_seconds: float = 2.0
    primary_detection_leads: Tuple[str, ...] = ("II", "V5")
    tachy_threshold_bpm: float = 100.0
    brady_threshold_bpm: float = 60.0
    pause_threshold_ms: float = 2000.0
    sqi_min: float = 0.6
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 40.0
    notch_hz: float = 50.0
    signal_flatline_std: float = 0.05
    signal_saturation_mv: float = 4.5
    signal_saturation_ratio: float = 0.01
    signal_baseline_wander_ratio: float = 0.30
    signal_hf_noise_ratio: float = 0.40
    signal_regularity_cv: float = 0.8
    qrs_validator_min_width_ms: float = 25.0
    qrs_validator_max_width_ms: float = 200.0
    qrs_validator_min_amplitude: float = 0.04
    template_similarity_threshold: float = 0.90
    qrs_detection_window_ms: float = 150.0
    qrs_detection_refractory_ms: float = 120.0
    qrs_refine_radius_ms: float = 80.0
    qrs_detection_threshold_scale: float = 0.25

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, object]] = None) -> "HolterConfig":
        cfg = ClinicalConfig.from_mapping(data or {})
        return cls(
            sampling_rate_hz=cfg.sampling_rate_hz,
            min_signal_seconds=cfg.min_signal_seconds,
            primary_detection_leads=cfg.primary_detection_leads,
            tachy_threshold_bpm=cfg.tachy_threshold_bpm,
            brady_threshold_bpm=cfg.brady_threshold_bpm,
            pause_threshold_ms=cfg.pause_threshold_ms,
            sqi_min=cfg.signal_quality_threshold,
            bandpass_low_hz=cfg.bandpass_low_hz,
            bandpass_high_hz=cfg.bandpass_high_hz,
            notch_hz=cfg.notch_hz,
            signal_flatline_std=cfg.signal_flatline_std,
            signal_saturation_mv=cfg.signal_saturation_mv,
            signal_saturation_ratio=cfg.signal_saturation_ratio,
            signal_baseline_wander_ratio=cfg.signal_baseline_wander_ratio,
            signal_hf_noise_ratio=cfg.signal_hf_noise_ratio,
            signal_regularity_cv=cfg.signal_regularity_cv,
            qrs_validator_min_width_ms=cfg.qrs_validator_min_width_ms,
            qrs_validator_max_width_ms=cfg.qrs_validator_max_width_ms,
            qrs_validator_min_amplitude=cfg.qrs_validator_min_amplitude,
            template_similarity_threshold=cfg.template_similarity_threshold,
            qrs_detection_window_ms=cfg.qrs_detection_window_ms,
            qrs_detection_refractory_ms=cfg.qrs_detection_refractory_ms,
            qrs_refine_radius_ms=cfg.qrs_refine_radius_ms,
            qrs_detection_threshold_scale=cfg.qrs_detection_threshold_scale,
        )

    @classmethod
    def from_clinical_config(cls, clinical_config: Optional[ClinicalConfig] = None) -> "HolterConfig":
        cfg = clinical_config or load_clinical_config()
        return cls.from_mapping(cfg.to_dict())


class SignalPreprocessor:
    def __init__(self, fs: float = 500.0, config: Optional[HolterConfig] = None):
        self.fs = float(fs or 500.0)
        self.config = config or HolterConfig()

    def remove_baseline(self, raw_signal: Sequence[float]) -> np.ndarray:
        signal = _safe_array(raw_signal)
        if signal.size == 0:
            return signal
        return signal - float(np.median(signal))

    def bandpass_filter(self, signal: Sequence[float]) -> np.ndarray:
        arr = _safe_array(signal)
        if arr.size == 0:
            return arr
        return _bandpass(arr, self.fs, self.config.bandpass_low_hz, self.config.bandpass_high_hz, order=2)

    def remove_powerline(self, signal: Sequence[float]) -> np.ndarray:
        arr = _safe_array(signal)
        if arr.size == 0:
            return arr
        nyquist = self.fs / 2.0
        if self.config.notch_hz <= 0 or self.config.notch_hz >= nyquist:
            return arr
        b, a = iirnotch(w0=self.config.notch_hz / nyquist, Q=30.0)
        return _safe_filter(arr, b, a)

    def process(self, raw_signal: Sequence[float]) -> np.ndarray:
        signal = self.remove_baseline(raw_signal)
        signal = self.bandpass_filter(signal)
        signal = self.remove_powerline(signal)
        return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


class SignalQuality:
    def __init__(self, fs: float = 500.0, config: Optional[HolterConfig] = None):
        self.fs = float(fs or 500.0)
        self.config = config or HolterConfig()

    def compute_sqi(self, signal: Sequence[float], r_peaks: Optional[Sequence[int]] = None) -> float:
        arr = _safe_array(signal)
        if arr.size < int(self.fs * max(self.config.min_signal_seconds, 0.5)):
            return 0.0
        if not np.any(arr):
            return 0.0

        signal_std = float(np.std(arr))
        if signal_std < self.config.signal_flatline_std:
            return 0.0

        baseline = _lowpass(arr, self.fs, self.config.bandpass_low_hz, order=2)
        drift_energy = float(np.var(baseline) / max(np.var(arr), 1e-9))
        drift_score = max(0.0, 1.0 - min(1.0, drift_energy / max(self.config.signal_baseline_wander_ratio, 1e-6)))

        high_freq = _highpass(arr, self.fs, self.config.bandpass_high_hz, order=2)
        noise_energy = float(np.var(high_freq) / max(np.var(arr), 1e-9))
        noise_score = max(0.0, 1.0 - min(1.0, noise_energy / max(self.config.signal_hf_noise_ratio, 1e-6)))

        stability_score = 0.5
        if r_peaks is not None:
            peaks = np.asarray(r_peaks, dtype=float).reshape(-1)
            if peaks.size >= 3:
                rr = np.diff(peaks)
                mean_rr = float(np.mean(rr))
                if mean_rr > 0:
                    cv = float(np.std(rr) / mean_rr)
                    stability_score = max(0.0, 1.0 - min(1.0, cv / max(self.config.signal_regularity_cv, 1e-6)))

        amplitude_score = max(
            0.0,
            min(1.0, float(np.ptp(arr)) / max(self.config.signal_saturation_mv, 1e-6)),
        )
        sqi = float(np.clip(np.mean([drift_score, noise_score, stability_score, amplitude_score]), 0.0, 1.0))
        return sqi


class MultiLeadSelector:
    def __init__(self, fs: float = 500.0, config: Optional[HolterConfig] = None):
        self.fs = float(fs or 500.0)
        self.config = config or HolterConfig()
        self._preprocessor = SignalPreprocessor(self.fs, self.config)
        self._quality = SignalQuality(self.fs, self.config)

    def select_best_lead(self, leads: Dict[str, Sequence[float]]) -> Tuple[Optional[str], np.ndarray, Dict[str, float]]:
        scores: Dict[str, float] = {}
        best_name: Optional[str] = None
        best_signal = np.array([], dtype=float)
        best_score = -1.0
        for lead_name, lead_signal in (leads or {}).items():
            processed = self._preprocessor.process(lead_signal)
            score = self._quality.compute_sqi(processed)
            scores[str(lead_name)] = float(score)
            if score > best_score:
                best_name = str(lead_name)
                best_signal = processed
                best_score = float(score)
        return best_name, best_signal, scores


class QRSValidator:
    def __init__(self, min_width_ms: float = 25.0, max_width_ms: float = 200.0, min_amplitude: float = 0.04):
        self.min_width_ms = float(min_width_ms)
        self.max_width_ms = float(max_width_ms)
        self.min_amplitude = float(min_amplitude)

    @classmethod
    def from_config(cls, config: Optional[HolterConfig] = None) -> "QRSValidator":
        cfg = config or HolterConfig()
        return cls(
            min_width_ms=cfg.qrs_validator_min_width_ms,
            max_width_ms=cfg.qrs_validator_max_width_ms,
            min_amplitude=cfg.qrs_validator_min_amplitude,
        )

    def is_valid(self, beat: Dict[str, float]) -> bool:
        width = float(beat.get("width_ms", 0.0) or beat.get("qrs_ms", 0.0) or 0.0)
        amplitude = float(beat.get("amplitude", 0.0) or beat.get("qrs_amplitude", 0.0) or 0.0)
        if width < self.min_width_ms or width > self.max_width_ms:
            return False
        if amplitude < self.min_amplitude:
            return False
        return True


def clean_rr(rr_intervals: Sequence[float], max_mad_z: float = 3.5) -> np.ndarray:
    rr = np.asarray(rr_intervals, dtype=float).reshape(-1)
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0]
    if rr.size == 0:
        return rr
    median = float(np.median(rr))
    mad = _median_absolute_deviation(rr)
    if mad <= 1e-9:
        return rr.copy()
    z = 0.6745 * (rr - median) / mad
    keep = np.abs(z) <= float(max_mad_z)
    filtered = rr[keep]
    if filtered.size >= 2:
        return filtered
    return rr.copy()


class TemplateCluster:
    def __init__(self, similarity_threshold: float = 0.90):
        self.similarity_threshold = float(similarity_threshold)

    @staticmethod
    def _similarity(a: Sequence[float], b: Sequence[float]) -> float:
        from ecg.template_matcher import compute_similarity

        try:
            return float(compute_similarity(np.asarray(a, dtype=float), np.asarray(b, dtype=float)))
        except Exception:
            a_arr = _normalize_template(a)
            b_arr = _normalize_template(b)
            n = min(a_arr.size, b_arr.size)
            if n < 3:
                return 0.0
            a_arr = a_arr[:n]
            b_arr = b_arr[:n]
            denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
            if denom <= 1e-9:
                return 0.0
            return float(np.dot(a_arr, b_arr) / denom)

    def cluster(self, beats: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        clusters: List[Dict[str, object]] = []
        for beat in beats:
            segment = np.asarray(beat.get("segment", []), dtype=float).reshape(-1)
            if segment.size == 0:
                continue
            placed = False
            for cluster in clusters:
                similarity = self._similarity(segment, cluster["template"])
                if similarity >= self.similarity_threshold:
                    cluster["beats"].append(dict(beat))
                    cluster["count"] += 1
                    cluster["template"] = self._blend_template(cluster["template"], segment, cluster["count"])
                    placed = True
                    break
            if not placed:
                clusters.append(
                    {
                        "template": _normalize_template(segment),
                        "beats": [dict(beat)],
                        "count": 1,
                    }
                )
        return clusters

    @staticmethod
    def _blend_template(existing: np.ndarray, new_segment: np.ndarray, count: int) -> np.ndarray:
        existing = _normalize_template(existing)
        incoming = _normalize_template(new_segment)
        n = min(existing.size, incoming.size)
        if n <= 0:
            return existing
        existing = existing[:n]
        incoming = incoming[:n]
        alpha = 1.0 / max(1, int(count))
        blended = (1.0 - alpha) * existing + alpha * incoming
        peak = float(np.max(np.abs(blended))) if blended.size else 0.0
        if peak > 0:
            blended = blended / peak
        return blended


@dataclass
class Event:
    time: float
    type: str
    label: str = ""
    payload: Optional[dict] = None


class ArrhythmiaDetector:
    def __init__(self, config: Optional[HolterConfig] = None):
        self.config = config or HolterConfig()

    def detect(
        self,
        rr_intervals_ms: Sequence[float],
        qrs_widths_ms: Optional[Sequence[float]] = None,
        p_wave_present: Optional[Sequence[bool]] = None,
        beat_count: Optional[int] = None,
    ) -> List[str]:
        rr = np.asarray(rr_intervals_ms, dtype=float).reshape(-1)
        rr = rr[np.isfinite(rr)]
        rr = rr[rr > 0]
        if rr.size == 0:
            return []

        hr = 60000.0 / float(np.mean(rr))
        width = np.asarray(qrs_widths_ms, dtype=float).reshape(-1) if qrs_widths_ms is not None else np.array([], dtype=float)
        p_flags = np.asarray(p_wave_present, dtype=bool).reshape(-1) if p_wave_present is not None else np.array([], dtype=bool)

        flags: List[str] = []
        irregular = float(np.std(rr) / max(np.mean(rr), 1.0)) > 0.18
        p_missing = bool(p_flags.size and not np.any(p_flags))
        pause = bool(np.any(rr > self.config.pause_threshold_ms))
        wide = bool(width.size and np.any(width > 120.0))

        if irregular and p_missing:
            flags.append("AF")
        if pause:
            flags.append("Pause")
        if hr > self.config.tachy_threshold_bpm:
            flags.append("Tachycardia")
        if hr < self.config.brady_threshold_bpm:
            flags.append("Bradycardia")
        if wide and hr > self.config.tachy_threshold_bpm:
            flags.append("Wide Complex Tachycardia")
        if beat_count is not None and beat_count < 2:
            flags.append("Insufficient Data")

        return flags


__all__ = [
    "ArrhythmiaDetector",
    "Event",
    "HolterConfig",
    "MultiLeadSelector",
    "QRSValidator",
    "SignalPreprocessor",
    "SignalQuality",
    "TemplateCluster",
    "clean_rr",
]
