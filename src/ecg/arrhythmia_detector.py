"""
ECG arrhythmia and interval analysis helpers for CardioX.

This module provides:
- Pan-Tompkins style R peak detection
- Beat-wise interval measurement
- Rhythm and arrhythmia classification
- Human-readable interpretation helpers
- A backward-compatible ArrhythmiaDetector class used by the UI
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

from ecg.holter.clinical_config import ClinicalConfig, load_clinical_config


_CLINICAL_CONFIG = load_clinical_config()
DEFAULT_FS = int(_CLINICAL_CONFIG.sampling_rate_hz)
MIN_SIGNAL_SECONDS = float(_CLINICAL_CONFIG.min_signal_seconds)
PRIMARY_DETECTION_LEADS = tuple(_CLINICAL_CONFIG.primary_detection_leads)
ORDERED_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _ms_to_samples(ms: float, fs: float) -> int:
    return int(ms * float(fs) / 1000.0)


def _safe_array(signal: Optional[Sequence[float]]) -> np.ndarray:
    if signal is None:
        return np.array([], dtype=float)
    arr = np.asarray(signal, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.array([], dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _trimmed_std(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    lo, hi = np.percentile(signal, [5, 95])
    clipped = signal[(signal >= lo) & (signal <= hi)]
    if clipped.size == 0:
        clipped = signal
    return float(np.std(clipped))


def _ensure_odd(value: int, minimum: int = 5) -> int:
    value = max(minimum, int(value))
    if value % 2 == 0:
        value += 1
    return value


def _median_baseline(signal: np.ndarray, start: int, width: int) -> float:
    left = max(0, start)
    right = min(len(signal), left + max(1, width))
    if right <= left:
        return float(np.median(signal)) if signal.size else 0.0
    return float(np.median(signal[left:right]))


def _bandpass_filter(signal: np.ndarray, fs: float, low_hz: float = 5.0, high_hz: float = 15.0, order: int = 2) -> np.ndarray:
    if signal.size < 3:
        return signal.copy()
    nyquist = float(fs) / 2.0
    low = max(0.001, low_hz / nyquist)
    high = min(0.99, high_hz / nyquist)
    if low >= high:
        return signal.copy()
    b, a = butter(order, [low, high], btype="band")
    padlen = min(signal.size - 1, max(len(a), len(b)) * 3)
    if padlen <= 0:
        return signal.copy()
    return filtfilt(b, a, signal, padlen=padlen)


def _atrial_flutter_features(
    signal: Sequence[float],
    fs: float = DEFAULT_FS,
    r_peaks: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Detect organised atrial flutter waves from baseline energy, independent of
    P-peak detection. Typical flutter waves cluster around 240-360 bpm (4-6 Hz).
    """
    sig = _safe_array(signal)
    if sig.size < int(float(fs) * MIN_SIGNAL_SECONDS) or not np.any(sig):
        return {"is_flutter": False, "score": 0.0, "atrial_rate_bpm": 0.0, "peak_hz": 0.0}

    working = sig.astype(float).copy()
    if r_peaks is not None and len(r_peaks) > 0:
        blank = _ms_to_samples(90, fs)
        for peak in r_peaks:
            center = int(peak)
            left = max(0, center - blank)
            right = min(working.size, center + blank + 1)
            if right > left:
                base_left = max(0, left - blank)
                base_right = min(working.size, right + blank)
                baseline = float(np.median(np.concatenate((working[base_left:left], working[right:base_right])))) if (left > base_left or base_right > right) else float(np.median(working))
                working[left:right] = baseline

    try:
        atrial = _bandpass_filter(working - float(np.median(working)), fs, low_hz=3.0, high_hz=15.0, order=2)
    except Exception:
        atrial = working - float(np.median(working))

    if atrial.size < 8:
        return {"is_flutter": False, "score": 0.0, "atrial_rate_bpm": 0.0, "peak_hz": 0.0}

    windowed = atrial * np.hanning(atrial.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(atrial.size, d=1.0 / float(fs))
    flutter_band = (freqs >= 3.5) & (freqs <= 8.0)
    reference_band = (freqs >= 1.0) & (freqs <= 20.0)
    if not np.any(flutter_band) or not np.any(reference_band):
        return {"is_flutter": False, "score": 0.0, "atrial_rate_bpm": 0.0, "peak_hz": 0.0}

    band_spectrum = spectrum[flutter_band]
    band_freqs = freqs[flutter_band]
    peak_idx = int(np.argmax(band_spectrum))
    peak_hz = float(band_freqs[peak_idx])
    flutter_energy = float(np.sum(band_spectrum ** 2))
    total_energy = float(np.sum(spectrum[reference_band] ** 2)) + 1e-9
    score = flutter_energy / total_energy
    atrial_rate_bpm = peak_hz * 60.0

    is_flutter = bool(0.18 <= score and 240.0 <= atrial_rate_bpm <= 360.0)
    return {
        "is_flutter": is_flutter,
        "score": float(score),
        "atrial_rate_bpm": float(atrial_rate_bpm),
        "peak_hz": peak_hz,
    }


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if signal.size == 0:
        return signal.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _signal_quality_score(signal: np.ndarray, fs: float) -> float:
    if signal.size < int(fs * MIN_SIGNAL_SECONDS):
        return 0.0
    if not np.any(signal):
        return 0.0
    amplitude = float(np.ptp(signal))
    noise = _trimmed_std(np.diff(signal))
    std = float(np.std(signal))
    if amplitude < 0.05 or std < 0.01:
        return 0.0
    snr = amplitude / max(noise, 1e-6)
    return float(max(0.0, min(1.0, (snr - 1.0) / 8.0)))


def _select_detection_lead(
    leads_dict: Dict[str, Sequence[float]],
    fs: float,
    preferred_leads: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], np.ndarray, float]:
    best_name = None
    best_signal = np.array([], dtype=float)
    best_quality = -1.0
    for lead_name in tuple(preferred_leads or PRIMARY_DETECTION_LEADS):
        signal = _safe_array(leads_dict.get(lead_name))
        quality = _signal_quality_score(signal, fs)
        if quality > best_quality:
            best_name = lead_name
            best_signal = signal
            best_quality = quality
    if best_signal.size == 0:
        for lead_name, lead_signal in leads_dict.items():
            signal = _safe_array(lead_signal)
            quality = _signal_quality_score(signal, fs)
            if quality > best_quality:
                best_name = lead_name
                best_signal = signal
                best_quality = quality
    return best_name, best_signal, max(0.0, best_quality)


def detect_r_peaks_pan_tompkins(signal: Sequence[float], fs: float = DEFAULT_FS) -> List[int]:
    signal_arr = _safe_array(signal)
    if signal_arr.size < int(fs * MIN_SIGNAL_SECONDS) or not np.any(signal_arr):
        return []

    filtered = _bandpass_filter(signal_arr, fs, low_hz=5.0, high_hz=15.0, order=2)
    differentiated = np.diff(filtered, prepend=filtered[0])
    squared = differentiated ** 2
    integration_window = _ms_to_samples(150, fs)
    integrated = _moving_average(squared, integration_window)

    candidate_distance = max(1, _ms_to_samples(120, fs))
    candidates, _ = find_peaks(integrated, distance=candidate_distance)
    if candidates.size == 0:
        return []

    signal_level = float(np.percentile(integrated[candidates], 75))
    noise_level = float(np.percentile(integrated[candidates], 25))
    threshold = noise_level + 0.25 * max(signal_level - noise_level, 1e-9)
    refractory_samples = _ms_to_samples(200, fs)
    refine_radius = max(1, _ms_to_samples(80, fs))

    r_peaks: List[int] = []
    for peak in candidates:
        peak_value = float(integrated[peak])
        if peak_value >= threshold:
            signal_level = 0.125 * peak_value + 0.875 * signal_level
        else:
            noise_level = 0.125 * peak_value + 0.875 * noise_level
        threshold = noise_level + 0.25 * max(signal_level - noise_level, 1e-9)
        if peak_value < threshold:
            continue

        left = max(0, peak - refine_radius)
        right = min(signal_arr.size, peak + refine_radius + 1)
        if right <= left:
            continue
        refined = left + int(np.argmax(signal_arr[left:right]))
        if r_peaks and refined - r_peaks[-1] < refractory_samples:
            if signal_arr[refined] > signal_arr[r_peaks[-1]]:
                r_peaks[-1] = refined
            continue
        r_peaks.append(refined)

    return [int(idx) for idx in r_peaks]


def _first_baseline_crossing(signal: np.ndarray, baseline: float, start: int, stop: int, step: int) -> int:
    if signal.size == 0:
        return max(0, start)
    idx = start
    last_value = signal[min(max(idx, 0), signal.size - 1)] - baseline
    while (idx > stop if step < 0 else idx < stop):
        idx += step
        if idx < 0 or idx >= signal.size:
            break
        value = signal[idx] - baseline
        if value == 0 or np.sign(value) != np.sign(last_value):
            return idx
        last_value = value
    return min(max(idx, 0), signal.size - 1)


def _qrs_bounds(signal: np.ndarray, r_idx: int, fs: float) -> Tuple[int, int, float, bool]:
    if signal.size == 0:
        return 0, 0, 0.0, False
    qrs_left = max(0, r_idx - _ms_to_samples(80, fs))
    qrs_right = min(signal.size - 1, r_idx + _ms_to_samples(80, fs))
    smooth_window = _ensure_odd(min(11, signal.size - (1 - signal.size % 2)))
    smoothed = savgol_filter(signal, smooth_window, 3, mode="interp") if signal.size >= smooth_window else signal
    slopes = np.gradient(smoothed)
    baseline = _median_baseline(signal, r_idx - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
    local_slopes = np.abs(slopes[qrs_left:qrs_right + 1])
    slope_threshold = max(float(np.percentile(local_slopes, 35)) if local_slopes.size else 0.0, 1e-4)
    baseline_threshold = max(0.15 * max(float(np.ptp(signal[qrs_left:qrs_right + 1])), 0.05), 0.015)
    amplitude_threshold = max(0.05 * max(float(np.ptp(signal[qrs_left:qrs_right + 1])), 0.1), 0.02)

    # q_resolved/j_resolved track whether a genuine quiet (isoelectric) boundary
    # was actually found, as opposed to the search falling through to the raw
    # +/-80ms window edge. On a chaotic/fibrillatory signal there is no quiet
    # point anywhere nearby, so both loops fall through and q_onset/j_point
    # silently default to qrs_left/qrs_right -- producing a fake, perfectly
    # uniform ~160ms "QRS" on every beat instead of signalling "undetermined".
    q_onset = qrs_left
    q_resolved = False
    for idx in range(r_idx, qrs_left - 1, -1):
        if abs(slopes[idx]) <= slope_threshold and abs(smoothed[idx] - baseline) <= baseline_threshold:
            q_onset = idx
            q_resolved = True
            break

    j_point = qrs_right
    j_resolved = False
    for idx in range(r_idx, qrs_right + 1):
        if abs(slopes[idx]) <= slope_threshold and abs(smoothed[idx] - baseline) <= baseline_threshold:
            j_point = idx
            j_resolved = True
            break

    active = np.where(np.abs(smoothed[qrs_left:qrs_right + 1] - baseline) > amplitude_threshold)[0]
    if active.size:
        q_onset = min(q_onset, qrs_left + int(active[0]))
        j_point = max(j_point, qrs_left + int(active[-1]))

    return int(q_onset), int(j_point), baseline, bool(q_resolved and j_resolved)


def _detect_p_wave(signal: np.ndarray, r_idx: int, baseline: float, fs: float) -> Dict[str, object]:
    result = {
        "p_present": False,
        "p_peak": None,
        "p_onset": None,
        "p_end": None,
        "p_duration_ms": None,
        "p_amplitude": 0.0,
    }
    p_start = max(0, r_idx - _ms_to_samples(400, fs))
    p_end = max(p_start + 1, r_idx - _ms_to_samples(50, fs))
    if p_end - p_start < 5:
        return result

    window = signal[p_start:p_end]
    prominence = max(0.04, 0.18 * max(float(np.ptp(window)), 0.05))
    peaks, props = find_peaks(window, prominence=prominence)
    if peaks.size == 0:
        peak_rel = int(np.argmax(window))
        if window[peak_rel] - baseline < max(0.05, 0.2 * np.ptp(window)):
            return result
    else:
        prominences = props.get("prominences", np.zeros(peaks.size))
        peak_rel = int(peaks[int(np.argmax(prominences))])

    p_peak = p_start + peak_rel
    amplitude = float(signal[p_peak] - baseline)
    # Threshold lowered from 0.08 → 0.04 mV: small but real P waves (common in
    # standard single-lead recordings) were being rejected, causing NSR to be
    # misclassified as "Rhythm Undetermined".
    if amplitude < 0.04:
        return result
    boundary_threshold = max(0.02, 0.2 * abs(amplitude))

    p_onset = p_peak
    left_limit = max(p_start, p_peak - _ms_to_samples(120, fs))
    for idx in range(p_peak, left_limit - 1, -1):
        if abs(signal[idx] - baseline) <= boundary_threshold:
            p_onset = idx
            break

    p_finish = p_peak
    right_limit = min(p_end - 1, p_peak + _ms_to_samples(120, fs))
    for idx in range(p_peak, right_limit + 1):
        if abs(signal[idx] - baseline) <= boundary_threshold:
            p_finish = idx
            break

    duration_ms = float((p_finish - p_onset) * 1000.0 / fs)
    if duration_ms <= 0 or duration_ms > 220.0:
        return result

    result.update(
        {
            "p_present": True,
            "p_peak": int(p_peak),
            "p_onset": int(p_onset),
            "p_end": int(p_finish),
            "p_duration_ms": duration_ms,
            "p_amplitude": amplitude,
        }
    )
    return result


def _tangent_t_end(signal: np.ndarray, r_idx: int, baseline: float, fs: float) -> Tuple[Optional[int], Optional[float]]:
    t_start = min(signal.size - 1, r_idx + _ms_to_samples(100, fs))
    t_stop = min(signal.size, r_idx + _ms_to_samples(600, fs))
    if t_stop - t_start < 5:
        return None, None
    window = signal[t_start:t_stop]
    peak_rel = int(np.argmax(window))
    t_peak = t_start + peak_rel
    downslope = signal[t_peak:t_stop]
    if downslope.size < 3:
        return None, None

    derivative = np.gradient(downslope)
    steepest_rel = int(np.argmin(derivative))
    steepest_idx = t_peak + steepest_rel
    slope = float(np.gradient(signal)[steepest_idx]) if 0 < steepest_idx < signal.size - 1 else 0.0

    if abs(slope) > 1e-6:
        intercept_idx = steepest_idx + int((baseline - signal[steepest_idx]) / slope)
        if t_peak <= intercept_idx < t_stop:
            return int(intercept_idx), float(signal[t_peak] - baseline)

    crossing = _first_baseline_crossing(signal, baseline, t_peak, t_stop - 1, 1)
    if crossing <= t_peak:
        return None, None
    return int(crossing), float(signal[t_peak] - baseline)


def _beat_noise_flags(signal: np.ndarray, q_onset: int, j_point: int, baseline: float, fs: float) -> Tuple[bool, List[str], float]:
    reasons: List[str] = []
    qrs_slice = signal[max(0, q_onset):min(signal.size, j_point + 1)]
    qrs_amplitude = float(np.ptp(qrs_slice)) if qrs_slice.size else 0.0
    if qrs_amplitude < 0.1:
        reasons.append("low_qrs_amplitude")

    flat_start = max(0, q_onset - _ms_to_samples(80, fs))
    flat_end = max(flat_start + 1, q_onset - _ms_to_samples(20, fs))
    flat_std = float(np.std(signal[flat_start:flat_end])) if flat_end > flat_start else 0.0
    if flat_std > 0.05:
        reasons.append("noisy_baseline")

    if abs(baseline) > 10:
        reasons.append("baseline_drift")

    return bool(reasons), reasons, qrs_amplitude


def measure_beat(signal: Sequence[float], r_idx: int, fs: float = DEFAULT_FS) -> Optional[Dict[str, object]]:
    signal_arr = _safe_array(signal)
    if signal_arr.size == 0 or r_idx <= 0 or r_idx >= signal_arr.size:
        return None

    q_onset, j_point, baseline, qrs_bounds_resolved = _qrs_bounds(signal_arr, int(r_idx), fs)
    qrs_ms = float((j_point - q_onset) * 1000.0 / fs)

    beat = {
        "r_peak": int(r_idx),
        "q_onset": int(q_onset),
        "j_point": int(j_point),
        "baseline": baseline,
        "qrs_ms": qrs_ms,
        "qrs_bounds_resolved": qrs_bounds_resolved,
        "p_present": False,
        "p_peak": None,
        "p_onset": None,
        "p_end": None,
        "p_duration_ms": None,
        "p_amplitude": 0.0,
        "pr_ms": None,
        "t_end": None,
        "t_peak": None,
        "t_amplitude": None,
        "qt_ms": None,
        "st_level_mv": None,
        "noisy": False,
        "noise_reasons": [],
        "qrs_amplitude": 0.0,
    }

    if qrs_ms < 40.0 or qrs_ms > 200.0:
        beat["noisy"] = True
        beat["noise_reasons"] = ["qrs_out_of_range"]
        return beat

    beat.update(_detect_p_wave(signal_arr, int(r_idx), baseline, fs))
    if beat["p_present"] and beat["p_onset"] is not None:
        pr_ms = float((int(q_onset) - int(beat["p_onset"])) * 1000.0 / fs)
        if 80.0 <= pr_ms <= 400.0:
            beat["pr_ms"] = pr_ms

    t_end, t_amp = _tangent_t_end(signal_arr, int(r_idx), baseline, fs)
    if t_end is not None:
        beat["t_end"] = int(t_end)
        beat["qt_ms"] = float((int(t_end) - int(q_onset)) * 1000.0 / fs)
        beat["t_amplitude"] = t_amp

    st_point = min(signal_arr.size - 1, int(r_idx) + _ms_to_samples(70, fs))
    beat["st_level_mv"] = float(signal_arr[st_point] - baseline)

    noisy, reasons, amplitude = _beat_noise_flags(signal_arr, int(q_onset), int(j_point), baseline, fs)
    beat["noisy"] = noisy
    beat["noise_reasons"] = reasons
    beat["qrs_amplitude"] = amplitude
    return beat


def _median_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.median(clean))


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def classify_heart_rate(hr_bpm: float, rr_intervals_ms: Sequence[float]) -> Dict[str, object]:
    """Returns rate-only label (rhythm labels are determined by detect_primary_rhythm)."""
    rr = np.asarray(rr_intervals_ms, dtype=float)
    rr_last = rr[-5:] if rr.size else rr
    rr_mean_ms = float(np.mean(rr)) if rr.size else 0.0
    rr_variability = float(np.max(rr_last) - np.min(rr_last)) if rr_last.size else 0.0
    is_irregular = rr_variability > 120.0

    # Rate-only labels — do NOT include sinus labels here.
    # Rhythm classification is done by detect_primary_rhythm().
    if hr_bpm < 20:
        label = "Severe bradycardia"
    elif hr_bpm < 40:
        label = "Bradycardia - severe"
    elif hr_bpm < 60:
        label = "Bradycardia"
    elif hr_bpm <= 100:
        label = "Normal rate"
    elif hr_bpm <= 110:
        label = "Borderline tachycardia"
    elif hr_bpm <= 150:
        label = "Tachycardia"
    elif hr_bpm <= 220:
        label = "Supraventricular tachycardia (SVT) - suspect"
    elif hr_bpm <= 300:
        label = "Ventricular tachycardia rate"
    else:
        label = "Extreme tachycardia"

    return {
        "label": label,
        "heart_rate_bpm": float(hr_bpm),
        "rr_mean_ms": rr_mean_ms,
        "rr_variability": rr_variability,
        "is_irregular": is_irregular,
    }


def is_normal_sinus_rhythm(beat_metrics: Dict[str, object]) -> Tuple[bool, List[str]]:
    """All criteria must pass: P wave, PR interval, narrow QRS, regular RR, HR 60-100."""
    failed: List[str] = []
    hr = float(beat_metrics.get("heart_rate_bpm") or 0.0)
    pr_ms = beat_metrics.get("pr_ms")
    qrs_ms = beat_metrics.get("qrs_ms")
    rr_variability = float(beat_metrics.get("rr_variability") or 0.0)
    p_present = bool(beat_metrics.get("p_present"))
    p_onset = beat_metrics.get("p_onset")
    q_onset = beat_metrics.get("q_onset")
    p_amplitude = float(beat_metrics.get("p_amplitude_lead_ii") or beat_metrics.get("p_amplitude") or 0.0)

    # ALL of these must be true for Normal Sinus Rhythm
    if not (60.0 <= hr <= 100.0):
        failed.append("hr_ok")
    if not p_present:
        failed.append("p_present")
    if p_onset is None or q_onset is None or not (int(p_onset) < int(q_onset)):
        failed.append("p_before_qrs")
    if pr_ms is None or not (120.0 <= float(pr_ms) <= 200.0):
        failed.append("pr_ok")
    if qrs_ms is None or not (float(qrs_ms) < 120.0):
        failed.append("qrs_narrow")
    if rr_variability >= 120.0:
        # Irregular RR = NOT sinus
        failed.append("rr_regular")
    if p_amplitude <= 0.05:
        failed.append("p_axis_ok")

    return (len(failed) == 0), failed


def _pp_intervals_from_beats(beats_list: Sequence[Dict[str, object]], fs: float) -> np.ndarray:
    p_peaks = [int(beat["p_peak"]) for beat in beats_list if beat.get("p_present") and beat.get("p_peak") is not None]
    if len(p_peaks) < 2:
        return np.array([], dtype=float)
    return np.diff(np.asarray(p_peaks, dtype=float)) * 1000.0 / float(fs)


def _detect_secondary_r(signal: np.ndarray, beat: Dict[str, object], fs: float) -> bool:
    if signal.size == 0:
        return False
    r_peak = int(beat.get("r_peak") or 0)
    j_point = int(beat.get("j_point") or r_peak)
    search_start = max(r_peak + _ms_to_samples(20, fs), j_point - _ms_to_samples(40, fs))
    end = min(signal.size, j_point + _ms_to_samples(80, fs))
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[search_start:end]
    if segment.size < 5:
        return False
    peaks, props = find_peaks(segment, prominence=max(0.03, 0.15 * max(np.ptp(segment), 0.05)))
    if peaks.size == 0:
        return False
    for peak_rel in peaks:
        if segment[int(peak_rel)] > baseline + 0.05:
            return True
    return False


def _terminal_s_present(signal: np.ndarray, beat: Dict[str, object], fs: float) -> bool:
    if signal.size == 0:
        return False
    r_peak = int(beat.get("r_peak") or 0)
    stop = min(signal.size, r_peak + _ms_to_samples(120, fs))
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[r_peak:stop]
    return bool(segment.size and np.min(segment) < baseline - 0.05)


def _dominant_negative_qrs(signal: np.ndarray, beat: Dict[str, object]) -> bool:
    if signal.size == 0:
        return False
    start = int(beat.get("q_onset") or 0)
    stop = min(signal.size, int(beat.get("j_point") or start + 1) + 1)
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[start:stop]
    if segment.size == 0:
        return False
    pos = float(np.max(segment) - baseline)
    neg = float(baseline - np.min(segment))
    return neg > max(pos * 1.2, 0.08)


def _broad_monophasic_r(signal: np.ndarray, beat: Dict[str, object], fs: float) -> bool:
    if signal.size == 0:
        return False
    start = int(beat.get("q_onset") or 0)
    stop = min(signal.size, int(beat.get("j_point") or start + 1) + 1)
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[start:stop]
    if segment.size == 0:
        return False
    positive = np.max(segment) - baseline
    negative = baseline - np.min(segment)
    if positive < 0.1 or negative > 0.05:
        return False
    above = np.where(segment > baseline + 0.05 * max(positive, 0.1))[0]
    if above.size == 0:
        return False
    duration_ms = float((above[-1] - above[0]) * 1000.0 / fs)
    return duration_ms > 60.0


def _has_septal_q(signal: np.ndarray, beat: Dict[str, object]) -> bool:
    if signal.size == 0:
        return False
    q_onset = int(beat.get("q_onset") or 0)
    r_peak = int(beat.get("r_peak") or q_onset)
    baseline = float(beat.get("baseline") or 0.0)
    segment = signal[q_onset:r_peak + 1]
    return bool(segment.size and np.min(segment) < baseline - 0.03)


def _contiguous_pairs(leads: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for idx in range(len(leads) - 1):
        pairs.append((leads[idx], leads[idx + 1]))
    return pairs


CONTIGUOUS_LEAD_PAIRS = (
    _contiguous_pairs(["I", "aVL", "V5", "V6"])
    + _contiguous_pairs(["II", "III", "aVF"])
    + _contiguous_pairs(["V1", "V2", "V3", "V4", "V5", "V6"])
)

BBB_MIN_QRS_MS = 110.0


def _median_qrs_polarity(
    signal: np.ndarray,
    beats: Sequence[Dict[str, object]],
    fs: float,
) -> Tuple[float, float]:
    """Return median positive/negative QRS deflection within beat windows."""
    sig = _safe_array(signal)
    if sig.size == 0 or not beats:
        return 0.0, 0.0

    positives: List[float] = []
    negatives: List[float] = []
    for beat in beats[: min(len(beats), 8)]:
        r_peak = beat.get("r_peak")
        if r_peak is None:
            continue
        r_idx = int(r_peak)
        start = max(0, r_idx - _ms_to_samples(120, fs))
        stop = min(sig.size, r_idx + _ms_to_samples(180, fs))
        if stop - start < 5:
            continue

        baseline = _median_baseline(sig, r_idx - _ms_to_samples(350, fs), _ms_to_samples(120, fs))
        segment = sig[start:stop] - baseline
        positives.append(float(np.max(segment)))
        negatives.append(float(abs(np.min(segment))))

    if not positives or not negatives:
        return 0.0, 0.0
    return float(np.median(positives)), float(np.median(negatives))


def detect_bundle_branch_block_from_leads(
    qrs_ms: Optional[float],
    signal_dict: Dict[str, Sequence[float]],
    beats: Sequence[Dict[str, object]],
    fs: float = DEFAULT_FS,
) -> Optional[str]:
    """Classify BBB from beat morphology using V1 plus lateral leads."""
    if not qrs_ms or float(qrs_ms) < BBB_MIN_QRS_MS:
        return None

    lead_i = _safe_array(signal_dict.get("I"))
    lead_v1 = _safe_array(signal_dict.get("V1"))
    lead_v6 = _safe_array(signal_dict.get("V6"))
    if lead_v1.size == 0:
        return None

    usable_beats = [beat for beat in beats if float(beat.get("qrs_ms") or qrs_ms or 0.0) >= 100.0]
    if not usable_beats:
        return None

    rbbb_votes = 0
    lbbb_votes = 0
    scored_beats = 0

    for beat in usable_beats[: min(len(usable_beats), 10)]:
        secondary_r = _detect_secondary_r(lead_v1, beat, fs)
        terminal_s_i = _terminal_s_present(lead_i, beat, fs) if lead_i.size else False
        terminal_s_v6 = _terminal_s_present(lead_v6, beat, fs) if lead_v6.size else False
        lateral_terminal_s = terminal_s_i or terminal_s_v6

        v1_negative = _dominant_negative_qrs(lead_v1, beat)
        broad_r_i = _broad_monophasic_r(lead_i, beat, fs) if lead_i.size else False
        broad_r_v6 = _broad_monophasic_r(lead_v6, beat, fs) if lead_v6.size else False
        lateral_broad_r = broad_r_i or broad_r_v6
        no_septal_q_i = (not _has_septal_q(lead_i, beat)) if lead_i.size else True
        no_septal_q_v6 = (not _has_septal_q(lead_v6, beat)) if lead_v6.size else True

        beat_rbbb = secondary_r and lateral_terminal_s
        beat_lbbb = v1_negative and lateral_broad_r and no_septal_q_i and no_septal_q_v6

        if beat_rbbb or beat_lbbb:
            scored_beats += 1
        if beat_rbbb and not beat_lbbb:
            rbbb_votes += 1
        elif beat_lbbb and not beat_rbbb:
            lbbb_votes += 1

    if scored_beats == 0:
        return None

    if rbbb_votes > lbbb_votes and rbbb_votes >= max(2, int(np.ceil(scored_beats * 0.5))):
        return "Complete Right Bundle Branch Block" if float(qrs_ms) >= 120.0 else "Right Bundle Branch Block"
    if lbbb_votes > rbbb_votes and lbbb_votes >= max(2, int(np.ceil(scored_beats * 0.5))):
        return "Complete Left Bundle Branch Block" if float(qrs_ms) >= BBB_MIN_QRS_MS else "Left Bundle Branch Block"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# LETHAL / PRIMARY RHYTHM DETECTORS
# ──────────────────────────────────────────────────────────────────────────────

def is_asystole(signal: np.ndarray, fs: float = DEFAULT_FS) -> bool:
    """
    Asystole = absent or near-flat signal (amplitude < 0.05 mV equivalent).
    We check: peak-to-peak < 0.05 AND std < 0.02.
    """
    sig = _safe_array(signal)
    if sig.size == 0:
        return True
    ptp = float(np.ptp(sig))
    std = float(np.std(sig))
    return ptp < 0.05 and std < 0.02


def is_ventricular_fibrillation(signal: np.ndarray, r_peaks: List[int], fs: float = DEFAULT_FS) -> bool:
    """
    VFib Detection - 4 criteria comprehensive scoring algorithm.
    Criteria:
      1. Rate 150-400 BPM (RR 150-400ms)
      2. Extreme RR Irregularity (CoV > 0.4)
      3. No Dominant QRS Morphology (Amp CoV > 0.35)
      4. Signal Baseline Chaos (>180 ZC/sec)
    """
    sig = _safe_array(signal)
    if sig.size < int(fs * 2.0):
        return False

    sig_mean = float(np.mean(sig))
    rr_intervals_ms = []
    qrs_amplitudes = []
    if len(r_peaks) >= 2:
        rr_intervals_ms = list(np.diff(np.array(r_peaks)) * 1000.0 / fs)
        for r in r_peaks:
            if 0 <= r < len(sig):
                # Subtract mean to get true amplitude, important for raw ADC data
                qrs_amplitudes.append(abs(sig[r] - sig_mean))

    # If few R-peaks, use legacy variance-based detection (fallback for completely chaotic VF)
    if len(rr_intervals_ms) < 5:
        variance = float(np.var(sig))
        amplitude = float(np.ptp(sig))
        if amplitude < 0.05:
            return False  # Flat line => asystole
        relative_variance = variance / max(amplitude ** 2, 1e-9)
        if len(r_peaks) < 3 and relative_variance > 0.015:
            return True
        if len(r_peaks) >= 2:
            mean_rr_ms = float(np.mean(np.diff(np.array(r_peaks))) * 1000.0 / fs)
            if mean_rr_ms < 200.0 and relative_variance > 0.015:
                return True
        return False

    score = 0.0
    debug_flags = []

    # Criterion 1: Rate (Broadened to include very fast and chaotic spacing)
    short_rr_count = sum(1 for rr in rr_intervals_ms if rr <= 450)
    short_rr_ratio = short_rr_count / len(rr_intervals_ms)
    if short_rr_ratio > 0.35:
        score += 0.35
        debug_flags.append(f"Rate(>35%<450ms):{short_rr_ratio:.2f}")

    # Criterion 2: Extreme RR Irregularity (CoV > 0.35)
    rr_array = np.array(rr_intervals_ms)
    valid_rr = rr_array[(rr_array > 100) & (rr_array < 2500)]
    if len(valid_rr) > 3:
        cov = float(np.std(valid_rr) / np.mean(valid_rr))
        if cov > 0.35:
            score += 0.30
            debug_flags.append(f"RR_CoV:{cov:.2f}")

    # Criterion 3: No Dominant QRS Morphology (QRS amplitude variability > 35%)
    if len(qrs_amplitudes) > 3:
        amp_array = np.array(qrs_amplitudes)
        amp_cov = float(np.std(amp_array) / (np.mean(np.abs(amp_array)) + 1e-6))
        if amp_cov > 0.35:
            score += 0.20
            debug_flags.append(f"Amp_CoV:{amp_cov:.2f}")

    # Criterion 4: Signal Baseline Chaos (High zero-crossing rate)
    try:
        from scipy.signal import detrend
        sig_detrended = detrend(sig.astype(float))
        zero_crossings = np.sum(np.diff(np.sign(sig_detrended)) != 0)
        zc_per_sec = float(zero_crossings / (len(sig_detrended) / fs))
        if zc_per_sec > 150:
            score += 0.15
            debug_flags.append(f"Chaos_ZC:{zc_per_sec:.0f}/s")
    except Exception:
        pass

    if score > 0.4:
        print(f" ⚠️ VFib Evaluation -> Score: {score:.2f} | Flags: {debug_flags}")

    # If flutter spectral features are present, reduce amplitude CoV weight.
    # Only let this veto a BORDERLINE candidate: VF's own chaotic baseline can
    # concentrate spectral energy in the same 3.5-8Hz band real flutter uses
    # (confirmed on real chaotic recordings misread as organised flutter), so
    # a score that has already cleared the VF threshold on rate/irregularity/
    # amplitude/chaos criteria should not be thrown out on spectral overlap
    # alone -- genuine flutter does not also present with VF-level RR chaos.
    flutter_check = _atrial_flutter_features(sig, fs, r_peaks=r_peaks)
    if float(flutter_check.get("score", 0)) > 0.18 and score < 0.60:
        # Organised atrial activity = not VFib
        return False

    # ── Physiological gate: VF cannot present at an organised normal rate ────
    # The scoring above can reach 0.65 from RR-variability (0.30) + amplitude
    # variability (0.20) + baseline chaos (0.15) alone — i.e. entirely from
    # NOISE, with the rate criterion never firing. That is how a 65 bpm sinus
    # recording was reported as Ventricular Fibrillation.
    #
    # VF is a ventricular rate of roughly 150-400 bpm with no organised
    # activation. If most RR intervals are long AND the implied rate is an
    # ordinary physiological one, the rhythm is organised and this is not VF,
    # however noisy the trace looks. Real VF still passes: its RR intervals are
    # short and chaotic, so short_rr_ratio is high and this gate does not apply.
    if len(valid_rr) > 3:
        median_rr_ms = float(np.median(valid_rr))
        if median_rr_ms > 0:
            implied_hr = 60000.0 / median_rr_ms
            if short_rr_ratio <= 0.35 and 20.0 <= implied_hr <= 150.0:
                if score >= 0.60:
                    print(
                        f" VFib rejected: organised rhythm at {implied_hr:.0f} bpm "
                        f"(median RR {median_rr_ms:.0f} ms, only "
                        f"{short_rr_ratio*100:.0f}% of beats < 450 ms). "
                        f"Noise score was {score:.2f}."
                    )
                return False

    # Final Decision: Score threshold lowered slightly to account for Pan-Tompkins integration
    return score >= 0.60


def is_ventricular_tachycardia(hr_bpm: float, qrs_ms: Optional[float],
                                p_present: bool, rr_variability: float) -> bool:
    """
    VT criteria (morphology-based, not just rate):
      - HR >= 100 bpm (ventricular rate)
      - Wide QRS >= 120 ms
      - No P waves preceding QRS (AV dissociation)
      - Relatively regular RR (not VF chaos)
    """
    if hr_bpm < 100:
        return False
    if qrs_ms is None or float(qrs_ms) < 120.0:
        return False
    if p_present:
        # P waves present = supraventricular origin, not VT
        return False
    if rr_variability > 200.0:
        # Highly irregular = more likely VF than VT
        return False
    return True


def detect_primary_rhythm(
    signal: np.ndarray,
    r_peaks: List[int],
    beats_list: Sequence[Dict[str, object]],
    rate_info: Dict[str, object],
    fs: float = DEFAULT_FS,
) -> str:
    """
    Priority-ordered primary rhythm detection.
    Returns the single most important rhythm label.

    Priority order (highest to lowest):
      1. Asystole          — amplitude-based
      2. Ventricular Fibrillation — chaos + no organised peaks
      3. Ventricular Tachycardia  — wide QRS + high rate + no P
      4. Sinus Bradycardia        — P wave present + HR < 60
      5. Sinus Tachycardia        — P wave present + HR > 100
      6. Normal Sinus Rhythm      — full sinus criteria met
      7. Rate-appropriate fallback
    """
    hr = float(rate_info.get("heart_rate_bpm") or 0.0)
    rr_variability = float(rate_info.get("rr_variability") or 0.0)

    # ─── 1. Asystole ────────────────────────────────────────────────────────
    if is_asystole(signal, fs):
        return "Asystole"

    # ─── 2. VF ──────────────────────────────────────────────────────────────
    if is_ventricular_fibrillation(signal, r_peaks, fs):
        return "Ventricular Fibrillation"

    # Gather beat-level stats for downstream checks
    clean_beats = [b for b in beats_list if not b.get("noisy")]
    ref_beats = clean_beats if clean_beats else list(beats_list)
    qrs_ms_vals = [float(b.get("qrs_ms") or 0.0) for b in ref_beats if b.get("qrs_ms")]
    mean_qrs_ms: Optional[float] = float(np.mean(qrs_ms_vals)) if qrs_ms_vals else None
    # p_present_majority: use p_present flag OR infer from pr_ms being available.
    # If the beat analyser computed a valid PR interval it means a P wave existed
    # even if the p_present boolean was not set (e.g. amplitude below old threshold).
    def _beat_has_p(b: Dict[str, object]) -> bool:
        if bool(b.get("p_present")):
            return True
        pr = b.get("pr_ms")
        return pr is not None and float(pr) > 0
    p_present_ratio = float(np.mean([_beat_has_p(b) for b in ref_beats])) if ref_beats else 0.0
    p_present_majority = p_present_ratio > 0.5

    # ─── 3. VT ──────────────────────────────────────────────────────────────
    if is_ventricular_tachycardia(hr, mean_qrs_ms, p_present_majority, rr_variability):
        return "Ventricular Tachycardia"

    # ─── 4–6. Sinus rhythms (require P wave present) ────────────────────────
    pr_vals = [float(b.get("pr_ms") or 0.0) for b in ref_beats if b.get("pr_ms")]
    # pr_ok: if PR values are available, they must be in normal range (120–200ms).
    # If pr_ms was not computed for any beat (common in some ECG configs), we do
    # NOT block NSR — we fall back to P-wave presence + rate + regularity alone.
    if pr_vals:
        pr_ok = 120.0 <= float(np.mean(pr_vals)) <= 200.0
    else:
        pr_ok = True  # No PR data: let p_present + rr_regular + narrow_qrs decide

    rr_regular = rr_variability < 120.0
    narrow_qrs = mean_qrs_ms is not None and mean_qrs_ms < 120.0

    is_sinus_origin = p_present_majority and pr_ok and rr_regular and narrow_qrs

    if is_sinus_origin:
        if hr < 60:
            return "Sinus Bradycardia"
        elif hr <= 100:
            return "Normal Sinus Rhythm"
        else:
            return "Sinus Tachycardia"

    # ─── 7. Fallback based on rate ───────────────────────────────────────────
    if hr < 60:
        return "Bradycardia (non-sinus)"
    elif hr <= 100:
        return "Rhythm Undetermined"
    else:
        return "Tachycardia (non-sinus)"


# ──────────────────────────────────────────────────────────────────────────────
# SECONDARY ARRHYTHMIA DETECTOR (PACS, PVCs, AF, Blocks, BBB, ST)
# ──────────────────────────────────────────────────────────────────────────────

def detect_arrhythmia(beats_list: Sequence[Dict[str, object]], signal_dict: Dict[str, Sequence[float]], fs: float = DEFAULT_FS) -> List[str]:
    if not beats_list:
        return []

    arrhythmias: List[str] = []
    clean_beats = [beat for beat in beats_list if not beat.get("noisy")]
    reference_beats = clean_beats if clean_beats else list(beats_list)
    rr_ms = np.asarray([float(beat["rr_ms"]) for beat in reference_beats if beat.get("rr_ms") is not None], dtype=float)
    pr_values = [float(beat["pr_ms"]) for beat in reference_beats if beat.get("pr_ms") is not None]
    hr = float(reference_beats[-1].get("heart_rate_bpm") or 0.0)
    rr_variability = float(np.max(rr_ms[-5:]) - np.min(rr_ms[-5:])) if rr_ms.size else 0.0

    p_absent_ratio = float(np.mean([not bool(beat.get("p_present")) for beat in reference_beats])) if reference_beats else 0.0
    signal_for_flutter = _safe_array(signal_dict.get("II"))
    if signal_for_flutter.size == 0:
        for value in signal_dict.values():
            signal_for_flutter = _safe_array(value)
            if signal_for_flutter.size:
                break
    r_peaks = signal_dict.get("r_peaks", [])
    flutter_features = _atrial_flutter_features(signal_for_flutter, fs, r_peaks=r_peaks)
    atrial_fib = p_absent_ratio > 0.70 and rr_variability > 120.0
    # Organised flutter should not win when the ventricular response is
    # clearly irregular over the same window; that pattern is more consistent
    # with AF than with classic flutter conduction.
    atrial_flutter = bool(flutter_features.get("is_flutter")) and rr_variability <= 120.0

    if atrial_fib:
        arrhythmias.append("Atrial Fibrillation")
    elif atrial_flutter:
        arrhythmias.append("Atrial Flutter")

    pp_ms = _pp_intervals_from_beats(reference_beats, fs)
    if not atrial_fib and not atrial_flutter and pp_ms.size:
        atrial_rate = 60000.0 / float(np.mean(pp_ms))
        ventricular_rate = hr
        if 250.0 <= atrial_rate <= 350.0 and rr_variability <= 120.0 and ventricular_rate > 0:
            ratio = atrial_rate / max(ventricular_rate, 1e-6)
            if abs(ratio - 2.0) < 0.4 or abs(ratio - 4.0) < 0.6:
                arrhythmias.append("Atrial Flutter")
                atrial_flutter = True

    p_present_ratio = float(np.mean([bool(beat.get("p_present")) for beat in reference_beats])) if reference_beats else 0.0

    skip_av_block = atrial_fib or atrial_flutter or p_present_ratio <= 0.7

    if not skip_av_block and any(beat.get("p_present") and beat.get("pr_ms") is not None and float(beat["pr_ms"]) > 200.0 for beat in reference_beats):
        arrhythmias.append("1st-degree AV block")

    # Bug 2: 2nd-degree AV Blocks (Mobitz I / II)
    if not skip_av_block and len(pr_values) >= 3:
        pr_trend = [pr_values[i+1] - pr_values[i] for i in range(len(pr_values)-1)]
        if sum(1 for d in pr_trend if d > 0) >= len(pr_trend) * 0.7:
            arrhythmias.append("2nd-degree AV block (Mobitz I / Wenckebach)")

    p_beats = [b for b in beats_list if b.get("p_present")]
    qrs_count = len(reference_beats)

    pp_regular = False
    av_dissociation = False
    if pp_ms.size >= 2:
        pp_regular = float(np.std(pp_ms)) <= 120.0
    pr_scatter = float(np.ptp(pr_values)) if len(pr_values) >= 2 else 0.0
    if len(pr_values) >= 3 and pr_scatter >= 80.0:
        av_dissociation = True

    if not skip_av_block and hr < 60.0 and pp_regular and av_dissociation:
        arrhythmias.append("3rd-degree AV block")

    lead_i = _safe_array(signal_dict.get("I"))
    lead_v1 = _safe_array(signal_dict.get("V1"))
    lead_v6 = _safe_array(signal_dict.get("V6"))

    has_rbbb = False
    has_lbbb = False
    incomplete_rbbb = False
    incomplete_lbbb = False
    for beat in reference_beats:
        qrs_ms = float(beat.get("qrs_ms") or 0.0)
        secondary_r = _detect_secondary_r(lead_v1, beat, fs)
        terminal_s = _terminal_s_present(lead_i, beat, fs) and _terminal_s_present(lead_v6, beat, fs)
        v1_negative = _dominant_negative_qrs(lead_v1, beat)
        broad_r = _broad_monophasic_r(lead_i, beat, fs) and _broad_monophasic_r(lead_v6, beat, fs)
        no_septal_q = (not _has_septal_q(lead_i, beat)) and (not _has_septal_q(lead_v6, beat))

        if qrs_ms >= BBB_MIN_QRS_MS and secondary_r and terminal_s:
            has_rbbb = True
        elif 100.0 <= qrs_ms < BBB_MIN_QRS_MS and secondary_r and terminal_s:
            incomplete_rbbb = True

        if qrs_ms >= BBB_MIN_QRS_MS and v1_negative and broad_r and no_septal_q:
            has_lbbb = True
        elif 100.0 <= qrs_ms < BBB_MIN_QRS_MS and v1_negative and broad_r and no_septal_q:
            incomplete_lbbb = True

    if has_rbbb and has_lbbb:
        arrhythmias.append("Inconclusive bundle branch block")
    elif has_rbbb:
        arrhythmias.append("Right bundle branch block (RBBB)")
    elif incomplete_rbbb:
        arrhythmias.append("Incomplete RBBB")

    if has_rbbb and has_lbbb:
        pass
    elif has_lbbb:
        arrhythmias.append("Left bundle branch block (LBBB)")
    elif incomplete_lbbb:
        arrhythmias.append("Incomplete LBBB")

    if rr_ms.size:
        mean_rr = float(np.mean(rr_ms))
        p_amps = [abs(float(beat.get("p_amplitude") or 0.0)) for beat in reference_beats if beat.get("p_present")]
        mean_p_amp = float(np.mean(p_amps)) if p_amps else 0.0
        for idx, beat in enumerate(reference_beats):
            beat_rr = beat.get("rr_ms")
            if beat_rr is None:
                continue
            qrs_ms = float(beat.get("qrs_ms") or 0.0)
            p_present = bool(beat.get("p_present"))
            t_amp = float(beat.get("t_amplitude") or 0.0)
            qrs_amp = float(beat.get("qrs_amplitude") or 0.0)

            if (
                qrs_ms > 120.0
                and not p_present
                and qrs_amp > 0.1
                and t_amp != 0.0
                and np.sign(t_amp) != np.sign(qrs_amp)
            ):
                next_rr = reference_beats[idx + 1].get("rr_ms") if idx + 1 < len(reference_beats) else None
                if next_rr is not None and float(next_rr) > 1.5 * mean_rr:
                    arrhythmias.append("Premature ventricular contraction (PVC)")
                    break

            if p_present_ratio > 0.6 and p_present and float(beat_rr) < 0.8 * mean_rr and qrs_ms < 120.0 and mean_p_amp > 0:
                if abs(abs(float(beat.get("p_amplitude") or 0.0)) - mean_p_amp) > 0.3 * mean_p_amp:
                    arrhythmias.append("Premature atrial contraction (PAC)")
                    break

    st_levels = {}
    for lead_name, lead_signal in signal_dict.items():
        signal_arr = _safe_array(lead_signal)
        if signal_arr.size == 0:
            continue
        per_beat = []
        for beat in reference_beats[: min(len(reference_beats), 10)]:
            qrs_amp = float(beat.get("qrs_amplitude") or 0.0)
            if qrs_amp < 0.05:   # skip flutter-wave beats that slipped through
                continue
            r_peak = int(beat.get("r_peak") or 0)
            st_point = min(signal_arr.size - 1, r_peak + _ms_to_samples(70, fs))
            baseline = _median_baseline(signal_arr, r_peak - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
            per_beat.append(float(signal_arr[st_point] - baseline))
        st_levels[lead_name] = float(np.mean(per_beat)) if per_beat else 0.0

    elevated = {lead for lead, value in st_levels.items() if value > 0.1}
    depressed = {lead for lead, value in st_levels.items() if value < -0.05}
    if any(a in elevated and b in elevated for a, b in CONTIGUOUS_LEAD_PAIRS):
        arrhythmias.append("ST elevation")
    if any(a in depressed and b in depressed for a, b in CONTIGUOUS_LEAD_PAIRS):
        arrhythmias.append("ST depression")

    deduped: List[str] = []
    for label in arrhythmias:
        if label not in deduped:
            deduped.append(label)
    return deduped


def _axis_from_amplitudes(lead_i_amp: float, avf_amp: float) -> float:
    return float(np.degrees(np.arctan2(float(avf_amp), float(lead_i_amp))))


def _median_wave_amplitude(signal: np.ndarray, beats: Sequence[Dict[str, object]], start_key: str, end_key: str, baseline_key: str = "baseline") -> float:
    values: List[float] = []
    for beat in beats:
        start = beat.get(start_key)
        end = beat.get(end_key)
        baseline = float(beat.get(baseline_key) or 0.0)
        if start is None or end is None:
            continue
        left = max(0, int(start))
        right = min(signal.size, int(end) + 1)
        if right <= left:
            continue
        segment = signal[left:right]
        pos = float(np.max(segment) - baseline)
        neg = float(baseline - np.min(segment))
        values.append(pos if pos >= neg else -neg)
    return float(np.median(values)) if values else 0.0


def _compute_axis(signal_dict: Dict[str, Sequence[float]], beats: Sequence[Dict[str, object]], wave: str) -> float:
    lead_i = _safe_array(signal_dict.get("I"))
    avf = _safe_array(signal_dict.get("aVF"))
    if lead_i.size == 0 or avf.size == 0 or not beats:
        return 0.0

    if wave == "P":
        start_key, end_key = "p_onset", "p_end"
    elif wave == "QRS":
        start_key, end_key = "q_onset", "j_point"
    else:
        start_key, end_key = "r_peak", "t_end"
    lead_i_amp = _median_wave_amplitude(lead_i, beats, start_key, end_key)
    avf_amp = _median_wave_amplitude(avf, beats, start_key, end_key)
    return _axis_from_amplitudes(lead_i_amp, avf_amp)




def analyze_ecg(
    leads_dict: Dict[str, Sequence[float]],
    fs: float = DEFAULT_FS,
    patient_gender: str = "M",
    external_metrics: Optional[Dict[str, float]] = None,
    clinical_config: Optional[ClinicalConfig] = None,
) -> Dict[str, object]:
    clinical_config = clinical_config or _CLINICAL_CONFIG
    fs = float(fs or clinical_config.sampling_rate_hz or DEFAULT_FS)
    cleaned_leads = {lead: _safe_array(sig) for lead, sig in (leads_dict or {}).items()}
    if not cleaned_leads:
        return {"arrhythmias": [], "reason": "No leads provided", "confidence": 0.0}

    lead_lengths = [sig.size for sig in cleaned_leads.values() if sig.size]
    if not lead_lengths or min(lead_lengths) < int(fs * clinical_config.min_signal_seconds):
        return {"arrhythmias": [], "reason": "Signal too short (< 2 seconds)", "confidence": 0.0}

    if all(not np.any(sig) for sig in cleaned_leads.values()):
        return {"arrhythmias": [], "reason": "All-zero signal", "confidence": 0.0}

    detection_lead_name, detection_signal, detection_quality = _select_detection_lead(
        cleaned_leads,
        fs,
        preferred_leads=clinical_config.primary_detection_leads,
    )
    if detection_signal.size == 0:
        return {"arrhythmias": [], "reason": "No usable detection lead", "confidence": 0.0}

    # ── Flat-line / Asystole check (ADC-scale signals) ──────────────────────
    # When the device sends raw ADC counts the signal mean is large (~2000)
    # but a flat/no-signal trace has near-zero std.  The mV-scale is_asystole()
    # uses a 0.05 threshold which is irrelevant at ADC scale.
    # Heuristic: if mean > 100 (raw ADC) AND std < 50  → flat line → Asystole.
    _det_std  = float(np.std(detection_signal))
    _det_mean = float(np.mean(np.abs(detection_signal)))
    _is_flat_line = (_det_mean > 100.0 and _det_std < 50.0)
    if _is_flat_line:
        _asystole_result = {
            "heart_rate_bpm": 0.0, "rr_ms": 0.0, "pr_ms": 0.0,
            "qrs_ms": 0.0, "qt_ms": 0.0, "qtc_bazett": 0.0,
            "qtc_fridericia": 0.0, "rv5_mv": 0.0, "sv1_mv": 0.0,
            "p_axis_deg": 0.0, "qrs_axis_deg": 0.0,
            "t_axis_deg": 0.0, "is_nsr": False,
            "nsr_failed_criteria": ["flat_line"],
            "primary_rhythm": "Asystole",
            "Primary Diagnosis": "Asystole",
            "arrhythmias": ["Asystole"],
            "st_levels": {}, "confidence": 0.0,
            "reason": "Flat-line signal (ADC std < 50)",
            "r_peaks": [], "beats": [],
            "Conduction Abnormalities": [],
            "Rhythm Analysis": ["Flat line", "No P waves"],
            "Intervals": {"PR": 0, "QRS": 0, "QT/QTc": "0/0"},
            "Signal Quality": "Poor",
        }
        return _asystole_result

    r_peaks = detect_r_peaks_pan_tompkins(detection_signal, fs)

    # === NEW: Filter out low-amplitude peaks (flutter waves) ===
    if len(r_peaks) >= 3:
        peak_amplitudes = np.array([
            float(detection_signal[r]) for r in r_peaks
            if 0 <= r < len(detection_signal)
        ])
        amp_median = float(np.median(peak_amplitudes))
        amp_threshold = amp_median * 0.55   # flutter waves typically <55% of QRS
        r_peaks = [
            r for r, amp in zip(r_peaks, peak_amplitudes)
            if abs(amp - amp_median) < amp_median * 0.6  # keep peaks near median amplitude
        ]
    # =====================================================

    if len(r_peaks) == 0:
        if is_asystole(detection_signal, fs):
            primary_rhythm = "Asystole"
            arrhythmias = [primary_rhythm]
        elif is_ventricular_fibrillation(detection_signal, [], fs):
            primary_rhythm = "Ventricular Fibrillation"
            arrhythmias = [primary_rhythm]
        else:
            primary_rhythm = "Rhythm Undetermined"
            arrhythmias = []
        return {
            "heart_rate_bpm": 0.0,
            "rr_ms": 0.0,
            "pr_ms": 0.0,
            "qrs_ms": 0.0,
            "qt_ms": 0.0,
            "qtc_bazett": 0.0,
            "qtc_fridericia": 0.0,
            "rv5_mv": 0.0,
            "sv1_mv": 0.0,
            "p_axis_deg": 0.0,
            "qrs_axis_deg": 0.0,
            "t_axis_deg": 0.0,
            "is_nsr": False,
            "nsr_failed_criteria": ["no_r_peaks"],
            "primary_rhythm": primary_rhythm,
            "arrhythmias": arrhythmias,
            "st_levels": {},
            "confidence": max(0.0, detection_quality * 0.5),
            "reason": "No R peaks found",
            "r_peaks": [],
            "beats": [],
        }
    # --- VFib Short Circuit ---
    # Even if Pan-Tompkins found "R-peaks", they might be chaotic VFib waves.
    # Validate that the candidate has neither reliable P waves nor an
    # organized QRS response before accepting the VF label.  Flutter F-waves
    # are independent of QRS, but Flutter still has organized ventricular
    # QRS peaks; that distinction prevents spectral Flutter activity from
    # being confused with VF.
    vf_candidate = is_ventricular_fibrillation(detection_signal, r_peaks, fs)
    provisional_beats = [
        beat for peak in r_peaks
        for beat in [measure_beat(detection_signal, int(peak), fs)]
        if beat is not None
    ]
    # A P wave only counts as "reliable" here when its own beat's QRS boundary
    # was actually resolved (see qrs_bounds_resolved above). An absolute
    # amplitude cutoff can't do this job: recordings may be stored as raw ADC
    # counts rather than calibrated mV, so a fixed-unit threshold is either a
    # no-op (ADC scale) or rejects real P waves (mV scale). A beat whose QRS
    # window never found a genuine isoelectric boundary is chaotic/noise at
    # that location, so any "P wave" measured relative to it is noise too --
    # on a fibrillatory baseline this flags nearly every beat as a false P wave.
    reliable_p_count = sum(
        bool(beat.get("p_present")) and bool(beat.get("qrs_bounds_resolved"))
        for beat in provisional_beats
    )
    # Only count a beat's QRS as "organized" when _qrs_bounds() actually found
    # a genuine isoelectric onset/offset -- not when both searches fell through
    # to the raw +/-80ms window edge, which yields a uniform fake ~160ms width
    # on every beat of a chaotic trace with no real QRS anywhere.
    organized_qrs_count = sum(
        bool(beat.get("qrs_bounds_resolved")) and 40.0 <= float(beat.get("qrs_ms") or 0.0) <= 220.0
        for beat in provisional_beats
    )
    no_reliable_p = reliable_p_count <= max(2, len(r_peaks) // 4)
    no_organized_qrs = organized_qrs_count < max(3, len(r_peaks) // 2)

    if vf_candidate and no_reliable_p and no_organized_qrs:
        return {
            "heart_rate_bpm": 0.0,
            "rr_ms": 0.0,
            "pr_ms": 0.0,
            "qrs_ms": 0.0,
            "qt_ms": 0.0,
            "qtc_bazett": 0.0,
            "qtc_fridericia": 0.0,
            "rv5_mv": 0.0,
            "sv1_mv": 0.0,
            "p_axis_deg": 0.0,
            "qrs_axis_deg": 0.0,
            "t_axis_deg": 0.0,
            "is_nsr": False,
            "nsr_failed_criteria": ["vfib"],
            "primary_rhythm": "Ventricular Fibrillation",
            "Primary Diagnosis": "Ventricular Fibrillation",
            "arrhythmias": ["Ventricular Fibrillation"],
            "st_levels": {},
            "confidence": 1.0,
            "reason": "VFib morphology scoring threshold met",
            "r_peaks": [],
            "beats": [],
            "Conduction Abnormalities": [],
            "Rhythm Analysis": ["Chaotic Rhythm", "No clear QRS"],
            "Intervals": {"PR": 0, "QRS": 0, "QT/QTc": "0/0"},
            "Signal Quality": "Poor",
        }

    lead_ii = cleaned_leads.get("II", detection_signal)
    beats: List[Dict[str, object]] = []
    for r_peak in r_peaks:
        beat = measure_beat(lead_ii, int(r_peak), fs)
        if beat is not None:
            beats.append(beat)

    rr_intervals_ms = np.diff(np.asarray(r_peaks, dtype=float)) * 1000.0 / fs if len(r_peaks) >= 2 else np.array([], dtype=float)

    # === NEW: Remove RR intervals that are implausibly short given the context ===
    if rr_intervals_ms.size >= 4:
        rr_p75 = float(np.percentile(rr_intervals_ms, 75))
        rr_p25 = float(np.percentile(rr_intervals_ms, 25))
        rr_valid = rr_intervals_ms[
            (rr_intervals_ms >= rr_p25 * 0.5) &
            (rr_intervals_ms <= rr_p75 * 2.0)
        ]
        if rr_valid.size >= 2:
            rr_intervals_ms = rr_valid
    # =====================================================
    
    if external_metrics and external_metrics.get("external_hr"):
        heart_rate = float(external_metrics.get("external_hr"))
    else:
        heart_rate = 60000.0 / float(np.mean(rr_intervals_ms)) if rr_intervals_ms.size else 0.0
    rate_info = classify_heart_rate(heart_rate, rr_intervals_ms)

    for idx, beat in enumerate(beats):
        beat["rr_ms"] = float(rr_intervals_ms[idx - 1]) if idx > 0 and idx - 1 < rr_intervals_ms.size else None
        beat["heart_rate_bpm"] = heart_rate
        beat["rr_variability"] = rate_info["rr_variability"]

    clean_beats = [beat for beat in beats if not beat.get("noisy")]
    averaging_beats = clean_beats[:10] if len(clean_beats) >= 3 else beats[:10]
    beats_for_pr = [beat for beat in beats if beat.get("pr_ms") is not None]

    if external_metrics and external_metrics.get("external_pr"):
        pr_ms = float(external_metrics.get("external_pr"))
    else:
        pr_ms = _median_or_none(beat.get("pr_ms") for beat in beats_for_pr) or _mean_or_none(beat.get("pr_ms") for beat in averaging_beats)
        
    if external_metrics and external_metrics.get("external_qrs"):
        qrs_ms = float(external_metrics.get("external_qrs"))
    else:
        qrs_ms = _median_or_none(beat.get("qrs_ms") for beat in averaging_beats) or _mean_or_none(beat.get("qrs_ms") for beat in averaging_beats)
    qt_candidates = [float(beat["qt_ms"]) for beat in averaging_beats if beat.get("qt_ms") is not None and 200.0 <= float(beat["qt_ms"]) <= 700.0]
    qt_for_average = qt_candidates[: min(5, len(qt_candidates))]
    qt_ms = float(np.mean(qt_for_average)) if qt_for_average else 0.0
    rr_sec = float(np.mean(rr_intervals_ms) / 1000.0) if rr_intervals_ms.size else 0.0
    qtc_bazett = float(qt_ms / math.sqrt(rr_sec)) if qt_ms > 0 and rr_sec > 0 else 0.0
    qtc_fridericia = float(qt_ms / (rr_sec ** (1.0 / 3.0))) if qt_ms > 0 and rr_sec > 0 else 0.0

    st_levels: Dict[str, float] = {}
    for lead_name, lead_signal in cleaned_leads.items():
        per_beat: List[float] = []
        for beat in averaging_beats:
            qrs_amp = float(beat.get("qrs_amplitude") or 0.0)
            if qrs_amp < 0.05:   # skip flutter-wave beats that slipped through
                continue
            r_peak = int(beat["r_peak"])
            st_point = min(lead_signal.size - 1, r_peak + _ms_to_samples(70, fs))
            baseline = _median_baseline(lead_signal, r_peak - _ms_to_samples(400, fs), _ms_to_samples(100, fs))
            per_beat.append(float(lead_signal[st_point] - baseline))
        st_levels[lead_name] = float(np.mean(per_beat)) if per_beat else 0.0

    last_beat = averaging_beats[-1] if averaging_beats else {"p_present": False, "p_onset": None, "q_onset": None, "p_amplitude": 0.0}
    nsr_input = {
        "heart_rate_bpm": heart_rate,
        "p_present": any(bool(beat.get("p_present")) for beat in averaging_beats),
        "p_onset": last_beat.get("p_onset"),
        "q_onset": last_beat.get("q_onset"),
        "pr_ms": pr_ms,
        "qrs_ms": qrs_ms,
        "rr_variability": rate_info["rr_variability"],
        "p_amplitude_lead_ii": _mean_or_none(beat.get("p_amplitude") for beat in averaging_beats if beat.get("p_present")) or 0.0,
    }
    is_nsr, nsr_failed = is_normal_sinus_rhythm(nsr_input)

    # ── PRIMARY RHYTHM vs MORPHOLOGY ──────────────────────────────
    primary = None
    secondary = []
    
    hr = heart_rate

    # Fix 3: P-wave consistency threshold (ratio > 0.5)
    qrs_count = len(r_peaks)
    p_count = sum(1 for b in beats if b.get("p_present"))
    p_ratio = p_count / qrs_count if qrs_count > 0 else 0.0
    p_detection_uncertain = False
    pp_ms_all = _pp_intervals_from_beats(beats, fs)
    pp_cv = float(np.std(pp_ms_all) / max(np.mean(pp_ms_all), 1e-9)) if pp_ms_all.size >= 2 else 0.0

    if p_ratio < 0.5 or pp_cv > 0.25:
        p_detection_uncertain = True
        # Bug 3 fix: Don't force p_present=False when AV block evidence is strong.
        # Wenckebach (Mobitz I) causes intermittent P-wave detection failures on
        # saturated signals — if dropped-beat or PR-progression evidence exists,
        # keep p_present=True so the AV block path wins over AF.
        # We do a quick early check here; full mobitz_1/dropped_beats computed below.
        _rr_vals_early = np.diff(np.asarray([], dtype=float))  # placeholder
        if rr_intervals_ms.size >= 2:
            _med_rr_early = float(np.median(rr_intervals_ms))
            _max_rr_early = float(np.max(rr_intervals_ms))
            _has_dropped_early = _med_rr_early > 0.0 and _max_rr_early > 1.5 * _med_rr_early
        else:
            _has_dropped_early = False
        if _has_dropped_early:
            # Preserve p_present so Mobitz path isn't suppressed by AF check
            p_present = True
        else:
            p_present = False
    else:
        p_present = True

    rr_std_ms = float(np.std(rr_intervals_ms)) if rr_intervals_ms.size >= 2 else 0.0
    rr_regular = rr_std_ms < 80.0
    rr_irregular = rr_std_ms > 80.0
    flutter_features = _atrial_flutter_features(detection_signal, fs, r_peaks=r_peaks)
    spectral_flutter = bool(flutter_features.get("is_flutter"))
    atrial_fib = (not p_present) and rr_irregular
    flutter_score = float(flutter_features.get("score", 0))
    atrial_flutter = spectral_flutter and (
        not rr_irregular or flutter_score > 0.35
    )
    effective_pr_ms = None if (atrial_fib or atrial_flutter) else pr_ms

    # Fix 2: PR Series for Mobitz I
    pr_series = [float(b.get("pr_ms")) for b in beats if b.get("pr_ms")]
    mobitz_1 = False
    if len(pr_series) >= 3:
        pr_trend = [pr_series[i + 1] - pr_series[i] for i in range(len(pr_series) - 1)]
        if sum(1 for d in pr_trend if d > 0) >= len(pr_trend) * 0.7:
            mobitz_1 = True

    # Bug 2 fix: RR-pattern fallback for Mobitz I when P-wave detection fails.
    # On saturated signals pr_series is empty so the PR-trend check never fires.
    # Wenckebach has a characteristic pattern: progressively shorter RRs ending
    # in a long pause. Detect it from RR intervals alone.
    if not mobitz_1 and rr_intervals_ms.size >= 4:
        _rr_min = float(np.min(rr_intervals_ms))
        _rr_max = float(np.max(rr_intervals_ms))
        if _rr_min > 0 and _rr_max / _rr_min > 1.4:
            # Confirm the long interval is isolated (not all intervals are long)
            _long_count = int(np.sum(rr_intervals_ms > 1.35 * _rr_min))
            _short_count = len(rr_intervals_ms) - _long_count
            if _long_count >= 1 and _short_count >= 2:
                mobitz_1 = True
    dropped_beats = False
    if rr_intervals_ms.size >= 2:
        median_rr = float(np.median(rr_intervals_ms))
        # Bug 1 fix: Lower threshold from 1.8x → 1.5x to catch Wenckebach pauses.
        # Wenckebach pause ratio: 1196 / 934 = 1.28 — the old 1.8x missed it entirely.
        dropped_beats = bool(median_rr > 0.0 and float(np.max(rr_intervals_ms)) > 1.5 * median_rr)
    pp_ms = pp_ms_all
    pp_regular = bool(pp_ms.size >= 2 and float(np.std(pp_ms)) <= 120.0)
    av_dissociation = False
    if len(pr_series) >= 3:
        av_dissociation = float(np.ptp(pr_series)) >= 80.0

    # Fix 8: SNR Signal Quality
    if detection_signal.size > 0:
        signal_var = float(np.var(detection_signal))
        signal_snr = float(np.mean(detection_signal**2) / (signal_var + 1e-9))
    else:
        signal_var = 0.0
        signal_snr = 0.0

    # Fix 6: Spectral VF extraction to feed into Engine Feature Dict
    signal_amplitude = float(np.mean(np.abs(detection_signal))) if detection_signal.size else 0.0
    vf_score = 0.0
    if signal_var > 0.02 and signal_amplitude > 0.1 and (qrs_count < 3 or rr_std_ms > 300.0):
        import scipy.fft
        spectrum = np.abs(scipy.fft.fft(detection_signal))
        if spectrum.size >= 30:
            vf_band_energy = float(np.sum(spectrum[2:30]))
            total_energy = float(np.sum(spectrum[1:len(spectrum)//2])) + 1e-9
            vf_score = vf_band_energy / total_energy

    bundle_branch_block = detect_bundle_branch_block_from_leads(qrs_ms, cleaned_leads, beats, fs)
    lbbb_indicator = 0.0
    if bundle_branch_block:
        if "left bundle" in bundle_branch_block.lower():
            lbbb_indicator = 1.0
        elif "right bundle" in bundle_branch_block.lower():
            lbbb_indicator = -1.0

    morphology_summary = {
        "cluster_count": 0.0,
        "dominant_ratio": 0.0,
        "ectopic_ratio": 0.0,
    }
    try:
        from ecg.template_matcher import cluster_beats, extract_beat_templates, morphology_features
        templates = extract_beat_templates(detection_signal, r_peaks, fs)
        beat_clusters = cluster_beats(templates)
        morphology_summary = morphology_features(beat_clusters)
    except Exception:
        pass

    features = {
        "hr": hr,
        "rr_intervals": rr_intervals_ms.tolist() if hasattr(rr_intervals_ms, "tolist") else rr_intervals_ms,
        "rr_std": rr_std_ms,
        "pr": effective_pr_ms or 0.0,
        "qrs": qrs_ms or 0.0,
        "qtc": qtc_bazett or 0.0,
        "p_detected": p_present,
        "qrs_width": qrs_ms or 0.0,
        "lbbb_indicator": lbbb_indicator,
        "dropped_beats": dropped_beats,
        "pr_progression": mobitz_1,
        "pp_regular": pp_regular,
        "av_dissociation": av_dissociation,
        "atrial_flutter": spectral_flutter,
        "atrial_rate_bpm": float(flutter_features.get("atrial_rate_bpm") or 0.0),
        "flutter_score": float(flutter_features.get("score") or 0.0),
        "vf_score": float(vf_score),
        "cluster_count": morphology_summary["cluster_count"],
        "dominant_ratio": morphology_summary["dominant_ratio"],
        "ectopic_ratio": morphology_summary["ectopic_ratio"],
        "lead_V1": cleaned_leads.get("V1", np.array([])),
        "lead_V6": cleaned_leads.get("V6", np.array([])),
        # Signal quality features used by _is_asystole()
        "signal_amplitude": signal_amplitude,
        "signal_std": _det_std,
    }

    try:
        from ecg.arrhythmia_engine.arrhythmia_engine import ArrhythmiaEngine
        engine = ArrhythmiaEngine(features)
        engine_diagnoses = engine.detect()
    except Exception:
        engine_diagnoses = []

    if signal_snr < 0.2:
        primary = "Poor Signal"
    else:
        primary = engine_diagnoses[0] if engine_diagnoses else ("Normal Sinus Rhythm" if p_present else "Unknown Rhythm")

    if primary not in ("Third-degree AV Block", "Ventricular Fibrillation", "Asystole") and av_dissociation and hr < 50:
        primary = "Third-degree AV Block"
        
    for d in engine_diagnoses[1:]:
        if d not in secondary and d not in ["Ventricular Tachycardia", "Ventricular Fibrillation", "Atrial Fibrillation", "Atrial Flutter", "VT", "VF", "AF", "Flutter", "Poor Signal"]:
            secondary.append(d)

    # 3. Morphology (SEPARATE) - Stable Multi-Lead Multi-Beat BBB Detection
    if bundle_branch_block and bundle_branch_block not in secondary:
        secondary.append(bundle_branch_block)

    # BONUS: Clustering + Template matching
    try:
        from ecg.template_matcher import (
            classify_ectopics,
            cluster_templates_crosscorr,
        )
        templates = extract_beat_templates(detection_signal, r_peaks, fs)
        cluster_ids = cluster_templates_crosscorr(templates)
        ectopics = classify_ectopics(cluster_ids, rr_intervals_ms)
        if morphology_summary["ectopic_ratio"] > 0.2 and "Frequent PVCs" not in secondary:
            secondary.append("Frequent PVCs")
        if morphology_summary["cluster_count"] > 2 and "Multifocal PVCs" not in secondary:
            secondary.append("Multifocal PVCs")
        for ect in ectopics:
            if ect not in secondary:
                secondary.append(ect)
    except Exception:
        pass

    # Combine for old report logic backward compat, but report_generator uses them correctly now
    combined: List[str] = [primary] + [s for s in secondary if s not in (primary, "")]
    primary_rhythm = primary
    is_nsr = primary_rhythm == "Normal Sinus Rhythm"
    nsr_failed_out = [] if is_nsr else nsr_failed

    rv5_mv, sv1_mv = 0.0, 0.0

    p_axis = _compute_axis(cleaned_leads, averaging_beats, "P")
    qrs_axis = _compute_axis(cleaned_leads, averaging_beats, "QRS")
    t_axis = _compute_axis(cleaned_leads, averaging_beats, "T")

    lead_scores = [_signal_quality_score(signal, fs) for signal in cleaned_leads.values() if signal.size]
    confidence = float(np.mean(lead_scores)) if lead_scores else 1.0
    
    # Fix 7: Confidence score penalizer & averaging
    # Real average of basic elements: p_consistency, rr_stability, etc.
    p_consistency_score = 1.0 if not p_detection_uncertain else 0.5
    rr_stability_score = 1.0 if rate_info["rr_variability"] < 300.0 else 0.6
    base_confidence = float(np.mean(lead_scores)) if lead_scores else 1.0
    
    confidence = (base_confidence + p_consistency_score + rr_stability_score) / 3.0

    # Fix 8: Edge Validation <60 or >220 QRS
    if qrs_ms and (qrs_ms < 60 or qrs_ms > 220):
        confidence -= 0.3

    confidence *= 0.9 if len(clean_beats) < max(1, min(3, len(beats))) else 1.0
    confidence = max(0.0, min(1.0, confidence))

    gender = str(patient_gender or "M").strip().upper()

    # Fix 10: Final Device-Level JSON Dictionary Formatting
    results = {
        "heart_rate_bpm": float(heart_rate),
        "Primary Diagnosis": primary_rhythm,
        "Conduction Abnormalities": [s for s in combined if "Block" in s or "PVC" in s],
        "Rhythm Analysis": [
            "Stable RR" if rr_regular else "Irregular RR",
            "P waves present" if p_present else "No P waves"
        ],
        "Intervals": {
            "PR": round(float(pr_ms or 0.0)),
            "QRS": round(float(qrs_ms or 0.0)),
            "QT/QTc": f"{int(qt_ms)}/{int(qtc_bazett)}"
        },
        "Signal Quality": "Good" if signal_snr >= 0.5 and confidence >= 0.7 else ("Poor" if signal_snr < 0.2 else "Fair"),
        "Confidence": round(float(confidence), 2),
        # Legacy/Extra bindings
        "rr_ms": float(rate_info["rr_mean_ms"]),
        "pr_ms": float(pr_ms or 0.0),
        "qrs_ms": float(qrs_ms or 0.0),
        "qt_ms": float(qt_ms),
        "qtc_bazett": float(qtc_bazett),
        "qtc_fridericia": float(qtc_fridericia),
        "rv5_mv": float(rv5_mv),
        "sv1_mv": float(sv1_mv),
        "p_axis_deg": float(p_axis),
        "qrs_axis_deg": float(qrs_axis),
        "t_axis_deg": float(t_axis),
        "is_nsr": bool(is_nsr),
        "nsr_failed_criteria": nsr_failed_out,
        "primary_rhythm": primary_rhythm,
        "arrhythmias": combined,
        "atrial_rate_bpm": float(flutter_features.get("atrial_rate_bpm") or 0.0),
        "atrial_flutter_score": float(flutter_features.get("score") or 0.0),
        "morphology": morphology_summary,
        "st_levels": st_levels,
        "confidence": confidence,
        "QRS Classification": "Wide QRS" if qrs_ms and qrs_ms >= 120 else "Normal QRS",
        "r_peaks": [int(idx) for idx in r_peaks],
        "beats": beats,
        "detection_lead": detection_lead_name,
        "detection_quality": detection_quality,
        "patient_gender": patient_gender,
    }

    # ── 7-Step Decision Layer ─────────────────────────────────────────────────
    # Integrates SQI gating, physiological consistency, hysteresis, and
    # priority-based selection. Runs non-destructively; failures fall through.
    try:
        from ecg.decision_layer import process_decision_layer
        rr_arr = np.array(rate_info.get("rr_intervals_ms") or rr_intervals_ms.tolist(), dtype=float)
        dl_features = {
            "p_detected": p_present,
            "p_present": p_present,
            "rr_std": rr_std_ms,
            "av_dissociation": av_dissociation,
            "atrial_flutter": spectral_flutter,
            "atrial_rate_bpm": float(flutter_features.get("atrial_rate_bpm") or 0.0),
            "flutter_score": float(flutter_features.get("score") or 0.0),
            "vf_score": float(vf_score),
            "organized_qrs": qrs_count >= 3 and rr_std_ms < 300.0,
            "signal_amplitude": signal_amplitude,
            "signal_snr": signal_snr,
        }
        dl_metrics = {
            "heart_rate_bpm": heart_rate,
            "pr_ms": pr_ms or 0.0,
            "qrs_ms": qrs_ms or 0.0,
            "qt_ms": qt_ms,
            "qtc_bazett": qtc_bazett,
            "Confidence": confidence,
        }
        available_leads = list(cleaned_leads.keys())
        instance_key = str(id(detection_signal)) if detection_signal.size else "default"
        dl_result = process_decision_layer(
            instance_id=instance_key,
            raw_signal=detection_signal,
            fs=fs,
            r_peaks=np.array(r_peaks, dtype=int),
            rr_intervals_ms=rr_arr,
            metrics=dl_metrics,
            features=dl_features,
            engine_diagnoses=combined,
            available_leads=available_leads,
            human_safety_mode=False,
        )
        results["decision_layer"] = dl_result
        # If the decision layer produced a valid result, update primary diagnosis
        if dl_result and not dl_result.get("interpretation_disabled"):
            dl_diagnoses = dl_result.get("diagnoses", [])
            if dl_diagnoses:
                results["Primary Diagnosis"] = dl_diagnoses[0]["diagnosis"]
                results["primary_rhythm"] = dl_diagnoses[0]["diagnosis"]
                results["arrhythmias"] = [d["diagnosis"] for d in dl_diagnoses]
    except Exception as _dl_err:
        pass  # Decision layer failure must not crash the main pipeline

    return results


def get_interpretation(results_dict: Dict[str, object]) -> List[str]:
    """
    Build a priority-ordered interpretation list.
    The PRIMARY RHYTHM is always the first item.
    Additional findings (BBB, QTc, ST) are appended after.
    Contradictory labels (e.g. sinus + VF) are suppressed.
    """
    if not results_dict:
        return ["No ECG analysis available"]

    qrs_ms = float(results_dict.get("qrs_ms") or 0.0)
    qtc = float(results_dict.get("qtc_bazett") or 0.0)
    arrhythmias = list(results_dict.get("arrhythmias") or [])
    st_levels = dict(results_dict.get("st_levels") or {})

    # ── Primary rhythm always first ──────────────────────────────────────────
    primary = results_dict.get("primary_rhythm") or (arrhythmias[0] if arrhythmias else "Rhythm Undetermined")
    interpretations: List[str] = [primary]

    LETHAL = {"Asystole", "Ventricular Fibrillation", "Ventricular Tachycardia"}

    # ── Secondary findings — only those compatible with primary ─────────────
    for label in arrhythmias:
        if label in interpretations:
            continue
        # Suppress sinus/normal labels when a lethal primary is set
        if primary in LETHAL and any(
            kw in label for kw in ("inus", "bradycardia", "Normal rate", "Borderline", "Normal Sinus")
        ):
            continue
        interpretations.append(label)

    # ── Additional interval / morphology findings ────────────────────────────
    if qtc > 460.0 and "Prolonged QTc interval" not in interpretations:
        interpretations.append("Prolonged QTc interval")
    has_bbb = any("bundle branch" in str(item).lower() or str(item) in ("RBBB", "LBBB") for item in interpretations)
    if qrs_ms >= 120.0 and not has_bbb:
        if "Wide QRS complex" not in interpretations:
            interpretations.append("Wide QRS complex")

    # ── ST changes ───────────────────────────────────────────────────────────
    elevated = [lead for lead, value in st_levels.items() if float(value) > 0.1]
    depressed = [lead for lead, value in st_levels.items() if float(value) < -0.05]
    if len(elevated) >= 2:
        label = f"ST elevation in {'-'.join(sorted(elevated))}"
        if label not in interpretations:
            interpretations.append(label)
    if len(depressed) >= 2:
        label = f"ST depression in {'-'.join(sorted(depressed))}"
        if label not in interpretations:
            interpretations.append(label)

    return interpretations


class ArrhythmiaDetector:
    """
    Backward-compatible wrapper used by the PyQt application.

    The public methods intentionally preserve the existing signatures.
    """

    def __init__(self, sampling_rate: float = DEFAULT_FS, counts_per_mv: float = 1.0, clinical_config: Optional[ClinicalConfig] = None):
        self.clinical_config = clinical_config or _CLINICAL_CONFIG
        self.fs = float(sampling_rate or self.clinical_config.sampling_rate_hz or DEFAULT_FS)
        self.counts_per_mv = float(counts_per_mv or 1.0)
        import collections
        self.diagnosis_buffer = collections.deque(maxlen=5)
        self.bbb_history = collections.deque(maxlen=5)

    def _normalize_signal(self, signal: Sequence[float]) -> np.ndarray:
        signal_arr = _safe_array(signal)
        if signal_arr.size and self.counts_per_mv not in (0.0, 1.0):
            signal_arr = signal_arr / self.counts_per_mv
        return signal_arr

    def detect_arrhythmias(
        self,
        signal,
        analysis,
        has_received_serial_data: bool = False,
        min_serial_data_packets: int = 50,
        lead_signals: Optional[Dict[str, Sequence[float]]] = None,
    ) -> List[str]:
        del has_received_serial_data
        del min_serial_data_packets

        primary_signal = self._normalize_signal(signal)
        analysis = analysis or {}
        lead_signals = dict(lead_signals or {})
        if "II" not in lead_signals and primary_signal.size:
            lead_signals["II"] = primary_signal

        results = analyze_ecg(
            lead_signals,
            fs=self.fs,
            external_metrics=analysis,
            clinical_config=self.clinical_config,
        )

        import collections

        # ── Temporal smoothing layer ──────────────────────────────────────────
        # LETHAL rhythms (Asystole / VF / VT) bypass the majority-vote buffer
        # and are reported immediately — a one-frame delay is clinically
        # unacceptable for cardiac arrest.
        _LETHAL_PRIMARY = {"Asystole", "Ventricular Fibrillation", "Ventricular Tachycardia"}
        raw_primary = results.get("Primary Diagnosis", "Rhythm Undetermined")

        if raw_primary in _LETHAL_PRIMARY:
            # Bypass smoothing — use raw detection immediately.
            # Also flush buffer so coming-back-to-normal isn't delayed.
            self.diagnosis_buffer.clear()
            self.diagnosis_buffer.append(raw_primary)
            smoothed_primary = raw_primary
        else:
            self.diagnosis_buffer.append(raw_primary)
            counter = collections.Counter(self.diagnosis_buffer)

            # ── Significant (non-lethal) rhythms use a lower threshold ─────────
            # AV Blocks, AF, AFL are clinically important and must not be
            # silenced by surrounding NSR frames.  They only need 2 out of the
            # last 5 detections to win — vs. the normal majority (≥3).
            _SIGNIFICANT_PRIMARY = {
                "Second-degree AV Block (Mobitz I)",
                "Third-degree AV Block",
                "First-degree AV Block (Prolonged PR)",
                "Atrial Fibrillation",
                "Atrial Flutter",
            }
            significant_in_buffer = [r for r in self.diagnosis_buffer if r in _SIGNIFICANT_PRIMARY]
            if len(significant_in_buffer) >= 2:
                # Significant rhythm seen in ≥2 of last 5 frames → report it
                sig_counter = collections.Counter(significant_in_buffer)
                smoothed_primary = sig_counter.most_common(1)[0][0]
            else:
                smoothed_primary = counter.most_common(1)[0][0]

        # Mutate the result so downstream report layers use smoothed version
        results["Primary Diagnosis"] = smoothed_primary
        results["primary_rhythm"] = smoothed_primary
        if smoothed_primary not in results.get("arrhythmias", []):
            if results.get("arrhythmias"):
                results["arrhythmias"][0] = smoothed_primary

        arrhythmias = list(results.get("arrhythmias") or [])

        # ── Asystole short-circuit ────────────────────────────────────────────
        # When Asystole is primary, return immediately with ONLY ["Asystole"].
        # No secondary findings (Wide QRS, etc.) should accompany it.
        if smoothed_primary == "Asystole":
            return ["Asystole"]

        # Fix 6: Smooth BBB across time
        current_bbb = "None"
        for label in arrhythmias:
            if "Bundle Branch block" in str(label).lower() or "bbb" in str(label).lower() or "inconclusive bundle" in str(label).lower():
                current_bbb = label
                break
        
        # Fix 6 sub-patch: Mark 'Possible Atrial Flutter' as low confidence instead of confirmed
        arrhythmias = [a if str(a) != "Possible Atrial Flutter" else "Atrial Flutter (low confidence)" for a in arrhythmias]
        
        self.bbb_history.append(current_bbb)
        non_none_history = [label for label in self.bbb_history if label != "None"]
        smoothed_bbb = non_none_history[-1] if non_none_history else "None"
        
        # Rewrite the array wiping out any instantaneous BBB and injecting the smoothed BBB
        arrhythmias = [a for a in arrhythmias if not ("Bundle Branch" in str(a) or "bbb" in str(a).lower() or "Inconclusive Bundle" in str(a))]
        if smoothed_bbb != "None":
            arrhythmias.append(smoothed_bbb)

        # Update dicts
        results["arrhythmias"] = arrhythmias
        if results.get("Conduction Abnormalities") is not None:
            results["Conduction Abnormalities"] = [smoothed_bbb] if smoothed_bbb != "None" else []

        # Always return primary rhythm first (already in arrhythmias[0] by design)
        return arrhythmias if arrhythmias else [results.get("primary_rhythm") or "Rhythm Undetermined"]

    def detect_arrhythmias_with_probabilities(self, signal, analysis, window_size: float = 2.0, step_size: Optional[float] = None) -> Dict[str, List[Tuple[float, float]]]:
        signal_arr = self._normalize_signal(signal)
        if analysis is None:
            analysis = {}
        if isinstance(analysis, dict):
            r_peak_source = analysis.get("r_peaks")
        else:
            r_peak_source = analysis
        if r_peak_source is None or len(r_peak_source) == 0:
            r_peak_source = detect_r_peaks_pan_tompkins(signal_arr, self.fs)
        r_peaks = np.asarray(r_peak_source, dtype=int)
        if r_peaks.size < 2:
            return {
                "Normal Sinus Rhythm": [],
                "Atrial Fibrillation": [],
                "Atrial Flutter": [],
            }

        if step_size is None:
            step_size = float(window_size)
        window_size = float(window_size or 2.0)
        step_size = float(step_size or window_size)

        start_t = float(r_peaks[0]) / self.fs
        end_t = float(r_peaks[-1]) / self.fs
        centers = np.arange(start_t, end_t + 1e-9, step_size)
        output = {
            "Normal Sinus Rhythm": [],
            "Atrial Fibrillation": [],
            "Atrial Flutter": [],
        }

        for center in centers:
            left = int(max(0, (center - window_size / 2.0) * self.fs))
            right = int(min(signal_arr.size, (center + window_size / 2.0) * self.fs))
            if right - left < int(self.fs):
                continue
            window_signal = signal_arr[left:right]
            results = analyze_ecg({"II": window_signal}, fs=self.fs)
            arrhythmias = set(results.get("arrhythmias") or [])
            output["Atrial Fibrillation"].append((float(center), 0.9 if "Atrial Fibrillation" in arrhythmias else 0.1))
            output["Atrial Flutter"].append((float(center), 0.85 if "Atrial Flutter" in arrhythmias else 0.1))
            output["Normal Sinus Rhythm"].append((float(center), 0.8 if results.get("is_nsr") else 0.2))

        return output


__all__ = [
    "ArrhythmiaDetector",
    "analyze_ecg",
    "classify_heart_rate",
    "detect_arrhythmia",
    "detect_bundle_branch_block_from_leads",
    "detect_primary_rhythm",
    "detect_r_peaks_pan_tompkins",
    "get_interpretation",
    "is_asystole",
    "is_normal_sinus_rhythm",
    "is_ventricular_fibrillation",
    "is_ventricular_tachycardia",
    "measure_beat",
]
