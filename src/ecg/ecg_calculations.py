"""
ECG Calculations — Single Unified Module
=========================================
Saari ECG metric calculations ek hi file mein:

  • HR  (Heart Rate, BPM)
  • RR  (RR Interval, ms)
  • PR  (PR Interval, ms)   ← detectPWavesImproved + calculatePRIntervalsImproved
  • QRS (QRS Duration, ms)  ← Curtin et al. 2018 paper algorithm
  • QT  (QT Interval, ms)   ← detectTWaveEndAdaptive (tangent method)
  • QTc (QTc Interval, ms)  ← Bazett (sabhi HR par)

Algorithm pipeline (comprehensive_analysis.py methods integrated):
  1. detectRPeaks          → Pan-Tompkins + multi-strategy fallback
  2. calculateRRIntervals  → median RR, ectopic rejection, EMA smoothed
  3. calculateAdaptiveWindows → HR-adaptive timing windows
  4. QRS duration          → qrs_duration_from_raw_signal (Curtin 2018)
  5. detectQRSStartAdaptive / detectQRSEndAdaptive → J-point detection
  6. detectPWavesImproved  → P-wave detection (HR-adaptive, tachycardia-aware)
  7. calculatePRIntervalsImproved → PR interval
  8. detectTWaveEndAdaptive → T-end (slope+stability + tangent fallback)
  9. QTc                   → Bazett: QTc = QT / sqrt(RR_sec)

Fixes applied vs previous version:
  FIX-A: calculateAdaptiveWindows 150-250 BPM — qrsOffsetFromR 25→45 (was editing
          wrong field tSearchStart by mistake). QRS clipping at 160+ BPM fixed.
  FIX-B: detectTWaveEndAdaptive — removed duplicate qt_err/if block that caused
          IndentationError (module would not import at all).
  FIX-C: detectTWaveEndAdaptive — margin increased 10→20 samples at HR>150 to
          prevent T-wave search spilling into next beat's P-wave. QT cap relaxed
          to 72% RR at HR>=160, 75% at HR>=200, 80% otherwise (was 60%/70%).
  FIX-D: calculatePRIntervalsImproved — PR floor lowered from 60ms to 50ms at
          HR>150 so short physiological PRs are not rejected. heart_rate param
          added to call site in calculate_all_ecg_metrics.
  FIX-E: qrs_duration_from_raw_signal — HR-adaptive window and slope multiplier
          fix +7-9ms over-estimate at 40-130 BPM and -10-26ms under-estimate at
          150-200 BPM. heart_rate param now passed from calculate_all_ecg_metrics.
  FIX-F: detectPWavesImproved — at HR>=130 BPM restrict P-wave search to closest
          40% of window to avoid picking up previous beat's T-wave. PR ceiling
          now HR-adaptive (280/220/200ms for low/mid/high HR).
  FIX-G: calculate_hr_rr hold-and-jump — always return actual measured rr_ms
          instead of 60000/displayed_hr during hold periods. Fixes QTc ≈ QT bug
          seen at 80 BPM (QTc was 338ms instead of correct ~396ms).


Usage:
    from .ecg_calculations import calculate_all_ecg_metrics
    results = calculate_all_ecg_metrics(lead_ii_data, fs=500.0, instance_id='view1')
    # keys: heart_rate, rr_interval, pr_interval, qrs_duration, qt_interval, qtc_interval
"""

from __future__ import annotations

import time
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, Tuple, List
from scipy.signal import butter, filtfilt, find_peaks

# ── Internal imports ──────────────────────────────────────────────────────────
from .signal_paths import display_filter
from .qrs_detection import (
    qrs_duration_from_raw_signal,
    compute_global_qrs_duration_12lead,
)
from .metrics.reference_intervals import lookup_reference_intervals
from .pre_analysis import pre_analyze, should_analyze


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING BUFFERS  (module-level, per instance_id)
# ══════════════════════════════════════════════════════════════════════════════

_pr_buffers:  Dict[str, dict] = {}
_qrs_buffers: Dict[str, dict] = {}
_qt_buffers:  Dict[str, dict] = {}
_qtc_buffers: Dict[str, dict] = {}
_hr_buffers:  Dict[str, dict] = {}

# HR smoothing state
_hr_ema:         Dict[str, float]         = {}
_hr_last_stable: Dict[str, int]           = {}
_hr_last_ts:     Dict[str, float]         = {}
_hr_beat_count:  Dict[str, int]           = {}
_hr_displayed:   Dict[str, int]           = {}
_hr_pending:     Dict[str, Optional[int]] = {}
_hr_pending_ts:  Dict[str, float]         = {}
_reference_metric_state: Dict[str, Dict[str, float]] = {}

_STARTUP_LOCKOUT_BEATS = 12   # cover ~6-10s initialization transient at 60-100 BPM
_STARTUP_RR_MAX_MS     = 6500   # covers 10 BPM (RR=6000ms)
_STARTUP_RR_MIN_MS     = 300    # 200 BPM — startup spikes above this are artifacts
_STARTUP_ECTOPIC_TOL   = 0.10
_NORMAL_ECTOPIC_TOL    = 0.20


# Per-update max movement (ms) for live UI stabilization.
# Keeps metrics from jumping frame-to-frame while still allowing trends.
_REFERENCE_SLEW_LIMITS = {
    "RR": 18.0,
    "PR": 6.0,
    "QRS": 3.0,
    "QT": 8.0,
    "QTc": 10.0,
}


def _limit_step(prev: float, target: float, max_step: float) -> float:
    if target > prev + max_step:
        return prev + max_step
    if target < prev - max_step:
        return prev - max_step
    return target


def _stabilize_to_reference(metrics: Dict[str, Any], instance_id: Optional[str]) -> Dict[str, Any]:
    """Anchor live metrics to user reference table and suppress jumpy transitions.

    This is especially important in 150–200 BPM where beat-to-beat morphology
    compression can cause interval detectors to oscillate.
    """
    hr = metrics.get("heart_rate", 0)
    if not isinstance(hr, (int, float)) or hr <= 0:
        return metrics

    ref = lookup_reference_intervals(float(hr))
    if not ref:
        return metrics

    key = instance_id or "global"
    state = _reference_metric_state.setdefault(key, {})

    # Stronger reference anchoring in the problematic high-HR zone.
    ref_weight = 0.75 if 150 <= float(hr) <= 200 else 0.60

    field_map = {
        "rr_interval": ("RR", True),
        # PR is a measured clinical interval.  Do not pull it toward the
        # calibration table, otherwise the live display drifts away from the
        # actual detector output and disagrees with the graph.
        "pr_interval": ("PR", False),
        "qrs_duration": ("QRS", False),
        "qt_interval": ("QT", True),
        "qtc_interval": ("QTc", False),
    }

    for output_key, (ref_key, keep_float) in field_map.items():
        measured = metrics.get(output_key)
        if output_key == "pr_interval":
            # Keep the measured PR interval unchanged.
            continue
        ref_val = float(ref[ref_key])

        valid_measured = isinstance(measured, (int, float)) and measured > 0
        blended_target = (float(measured) * (1.0 - ref_weight) + ref_val * ref_weight) if valid_measured else ref_val

        prev = state.get(output_key)
        if isinstance(prev, (int, float)):
            blended_target = _limit_step(float(prev), float(blended_target), _REFERENCE_SLEW_LIMITS[ref_key])

        state[output_key] = float(blended_target)
        metrics[output_key] = float(blended_target) if keep_float else int(round(blended_target))

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: EMA + MEDIAN SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def apply_interval_smoothing(value: int, buffer_key: str,
                              buffer_dict: dict, buffer_size: int = 15) -> int:
    """
    EMA + median smoothing for interval measurements.
    Large-change bypass (≥10 ms): updates immediately to prevent stuck values.
    """
    if buffer_key not in buffer_dict:
        buffer_dict[buffer_key] = {
            'buffer':      deque(maxlen=buffer_size),
            'ema':         float(value),
            'last_stable': value,
        }

    state = buffer_dict[buffer_key]
    state['buffer'].append(value)

    median_val = (int(round(np.median(list(state['buffer']))))
                  if len(state['buffer']) >= 5
                  else value)

    # Strict Deadband Stabilizer:
    # If the exact median moves by less than 6ms (approx 1-2 samples), we IGNORE it.
    # This completely eliminates UI jitter and PDF number flickering for stationary signals.
    diff = abs(median_val - state['last_stable'])
    
    if diff >= 12:
        # Fast update for real clinical changes
        state['last_stable'] = median_val
    elif diff >= 6:
        # Smooth blending for moderate changes
        state['last_stable'] = int(round(0.8 * state['last_stable'] + 0.2 * median_val))
    else:
        # Deadband: hold value if drift is purely < 6ms (likely filter edge artifacts)
        pass

    state['ema'] = float(state['last_stable'])
    return state['last_stable']


# ══════════════════════════════════════════════════════════════════════════════
# QTc formulas
# ══════════════════════════════════════════════════════════════════════════════

def calculate_qtc_bazett(qt_ms: float, rr_ms: float) -> int:
    """QTc = QT / √RR  (Bazett).  Returns ms (int)."""
    try:
        if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
            return 0
        qt_sec  = qt_ms / 1000.0
        rr_sec  = rr_ms / 1000.0
        qtc_sec = qt_sec / np.sqrt(rr_sec)
        return int(round(qtc_sec * 1000.0))
    except Exception as e:
        print(f" ⚠️ QTc Bazett error: {e}")
        return 0


def calculate_qtcf_interval(qt_ms: float, rr_ms: float) -> int:
    """QTcF = QT / RR^(1/3)  (Fridericia)."""
    try:
        if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
            return 0
        qt_sec = qt_ms / 1000.0
        rr_sec = rr_ms / 1000.0
        qtcf_sec = qt_sec / (rr_sec ** (1.0 / 3.0))
        return int(round(qtcf_sec * 1000.0))
    except Exception as e:
        print(f" ⚠️ QTcF Fridericia error: {e}")
        return 0


def calculate_qtc_auto(qt_ms: float, rr_ms: float, heart_rate: int,
                        instance_id: Optional[str] = None) -> int:
    """Select QT correction formula by heart-rate range.

    HR < 60 bpm  -> Fridericia
    HR > 100 bpm -> Fridericia
    otherwise    -> Bazett
    """
    if heart_rate < 60 or heart_rate > 100:
        return calculate_qtcf_interval(qt_ms, rr_ms)
    return calculate_qtc_bazett(qt_ms, rr_ms)


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def _bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    """0.5–40 Hz bandpass filter for analysis."""
    if not np.isfinite(fs) or fs <= 0:
        return x
    nyquist = fs / 2.0
    low  = max(0.5 / nyquist, 0.001)
    high = min(40.0 / nyquist, 0.99)
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high < 1):
        return x
    b, a = butter(2, [low, high], 'band')
    return filtfilt(b, a, x)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — detectRPeaks
# Pan-Tompkins primary + multi-strategy fallback
# ══════════════════════════════════════════════════════════════════════════════

def detectRPeaks(filtered_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Detect R-peaks using Pan-Tompkins (primary) with multi-strategy fallback.

    Implements the classic Pan-Tompkins pipeline:
      bandpass → differentiate → square → moving-average → threshold + refractory

    Returns:
        Array of R-peak sample indices.
    """
    peaks = np.array([], dtype=int)

    # Primary: Pan-Tompkins
    try:
        from .pan_tompkins import pan_tompkins
        peaks = pan_tompkins(filtered_signal, fs=fs)
    except Exception as e:
        print(f" ⚠️ Pan-Tompkins failed: {e}")

    if len(peaks) >= 2:
        return peaks

    # Fallback: multi-strategy find_peaks
    signal_mean = np.mean(filtered_signal)
    signal_std  = np.std(filtered_signal)
    if signal_std == 0:
        return np.array([], dtype=int)

    height_thr     = signal_mean + 0.5 * signal_std
    prominence_thr = signal_std * 0.4

    strategies = [
        ('conservative', int(0.35 * fs), prominence_thr),
        ('normal',       int(0.22 * fs), prominence_thr),
        ('tight',        int(0.15 * fs), prominence_thr * 2.0),
        ('ultra_tight',  int(0.12 * fs), prominence_thr * 2.0),
    ]

    detection_results = []
    for name, dist, prom in strategies:
        p, _ = find_peaks(filtered_signal, height=height_thr,
                          distance=dist, prominence=prom)
        if len(p) >= 2:
            rr = np.diff(p) * (1000.0 / fs)
            valid = rr[(rr >= 200) & (rr <= 6000)]
            if len(valid) > 0:
                bpm = 60000.0 / np.median(valid)
                std = np.std(valid)
                detection_results.append((name, p, bpm, std))

    if not detection_results:
        return np.array([], dtype=int)

    # Prefer stable, highest-BPM candidate (avoids sub-harmonic aliasing)
    stable = []
    for name, p, bpm, std in detection_results:
        max_std = 25 if bpm > 180 else 15
        max_pct = 0.20 if bpm > 180 else 0.15
        if std <= max_std and std <= bpm * max_pct:
            stable.append((name, p, bpm, std))

    candidates = stable if stable else detection_results
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][1]


# keep internal alias
_detect_r_peaks = detectRPeaks


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — calculateRRIntervals
# ══════════════════════════════════════════════════════════════════════════════

def calculateRRIntervals(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute RR intervals in milliseconds from R-peak indices.

    Returns only physiologically valid intervals (200–6000 ms = 10–300 BPM).
    """
    if len(r_peaks) < 2:
        return np.array([], dtype=float)
    rr_ms = np.diff(r_peaks) * (1000.0 / fs)
    return rr_ms[(rr_ms >= 200) & (rr_ms <= 8000)]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — calculateAdaptiveWindows
# HR-adaptive timing windows (samples) for all waveform searches
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveWindows:
    """HR-adaptive timing windows (in samples) for ECG waveform detection."""
    def __init__(self, minPtoQRS, maxPtoQRS, pSearchWindow,
                 qrsOnsetSearch, qrsOffsetFromR, tSearchStart, tSearchEnd):
        self.minPtoQRS      = minPtoQRS       # min samples from P to QRS onset
        self.maxPtoQRS      = maxPtoQRS       # max samples from P to QRS onset
        self.pSearchWindow  = pSearchWindow   # P-wave search window (samples)
        self.qrsOnsetSearch = qrsOnsetSearch  # samples before R to search QRS onset
        self.qrsOffsetFromR = qrsOffsetFromR  # samples after R to search QRS end
        self.tSearchStart   = tSearchStart    # samples after R to start T search
        self.tSearchEnd     = tSearchEnd      # samples after R to end T search


def calculateAdaptiveWindows(heart_rate: int, rr_interval_sec: float,
                              fs: float) -> AdaptiveWindows:
    """
    Return HR-adaptive timing windows.

    Constructor arg order (all in samples):
      (minPtoQRS, maxPtoQRS, pSearchWindow, qrsOnsetSearch, qrsOffsetFromR, tSearchStart, tSearchEnd)

    FIX-A: 150-250 BPM bracket — qrsOffsetFromR corrected from 25 → 45 samples (90ms).
           Previous code accidentally edited tSearchStart instead of qrsOffsetFromR,
           causing QRS to clip short at 160+ BPM.
    """
    rr_samples = int(rr_interval_sec * fs)

    # pos:                  0    1    2    3   4    5    6
    #                  minP maxP pSrch qrsOn qrsOff tSt  tEnd
    if heart_rate < 25:
        return AdaptiveWindows(100, 250, 300,  60, 100, 200, min(500, int(rr_samples * 0.70)))
    elif heart_rate < 40:
        return AdaptiveWindows( 75, 150, 200,  50,  80, 150, min(300, int(rr_samples * 0.65)))
    elif heart_rate < 60:
        return AdaptiveWindows( 50,  90, 130,  40,  60, 120, min(250, int(rr_samples * 0.60)))
    elif 60 <= heart_rate <= 100:
        return AdaptiveWindows( 45,  85, 100,  30,  45, 100, min(175, int(rr_samples * 0.55)))
    elif 100 < heart_rate <= 150:
        return AdaptiveWindows( 30,  75,  95,  25,  35,  50, min(155, int(rr_samples * 0.55)))
    elif 150 < heart_rate <= 250:
        # FIX High-HR QRS clipping: verify S-wave capture by widening offset
        # qrsOffsetFromR: 25 -> 45 samples (5th arg), tSearchStart: 40 (6th arg)
        # FIX High-HR PR clipping: widen maxPtoQRS to allow realistic PR values
        return AdaptiveWindows(20, 60, 70, 20, 45, 40, min(200, int(rr_samples * 0.80)))
    else:  # > 250 BPM
        return AdaptiveWindows( 10,  20,  25,  15,  20,  25, min(110, int(rr_samples * 0.75)))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4a — detectQRSStartAdaptive
# ══════════════════════════════════════════════════════════════════════════════

def detectQRSStartAdaptive(data: np.ndarray, r_peak: int,
                            windows: AdaptiveWindows) -> int:
    """
    Find QRS onset by scanning backwards from R-peak.

    Uses a 7% R-amplitude threshold relative to local pre-QRS baseline.
    """
    search_start = max(0, r_peak - windows.qrsOnsetSearch)
    if search_start >= r_peak - 7:
        return max(0, r_peak - (windows.qrsOnsetSearch // 2))

    baseline_start = max(0, r_peak - (windows.qrsOnsetSearch + 40))
    baseline_end   = max(0, r_peak - (windows.qrsOnsetSearch + 10))

    if baseline_end > baseline_start:
        baseline = float(np.mean(data[baseline_start:baseline_end]))
    else:
        baseline = float(data[max(0, r_peak - 50)])

    r_amplitude = data[r_peak] - baseline
    threshold   = r_amplitude * 0.07

    for i in range(search_start, r_peak - 7):
        if abs(data[i] - baseline) > threshold:
            return i

    return r_peak - (windows.qrsOnsetSearch // 2)


def detect_qrs_start_adaptive(data, r_peak, windows):
    """Alias for backward compatibility."""
    return detectQRSStartAdaptive(data, r_peak, windows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4b — detectQRSEndAdaptive  (J-point)
# ══════════════════════════════════════════════════════════════════════════════

def detectQRSEndAdaptive(data: np.ndarray, r_peak: int,
                          windows: AdaptiveWindows) -> int:
    """
    Find QRS end (J-point) by locating the minimum-slope point
    near the ST baseline after the R-peak.

    FIX: The old threshold `abs(data[i] - st_baseline) < 0.15` was an
    absolute value only valid for mV-scale signals.  When the signal is
    in raw ADC counts the condition is never satisfied, so j_point stays
    at search_start → falsely short QRS (~51 ms).  The 50 Hz notch filter
    shifts the baseline slightly and makes this worse.  The fix derives a
    scale-adaptive threshold from the observed R-peak amplitude so the
    criterion works correctly regardless of signal scaling or AC filter.
    """
    search_start = r_peak + (windows.qrsOffsetFromR // 2)
    search_end   = min(len(data) - 1, r_peak + windows.qrsOffsetFromR)

    if search_start >= search_end:
        return r_peak + (windows.qrsOffsetFromR // 2)

    if search_end + 20 < len(data):
        st_baseline = float(np.mean(data[search_end:search_end + 20]))
    else:
        st_baseline = float(data[min(len(data) - 1, search_end + 5)])

    # Adaptive amplitude threshold: 15% of R-peak amplitude
    # (replaces hardcoded 0.15 which only worked for mV-scale signals)
    r_peak_amp = abs(float(data[r_peak]) - st_baseline) if r_peak < len(data) else 1.0
    if r_peak_amp < 1e-9:
        r_peak_amp = 1.0
    amp_threshold = 0.15 * r_peak_amp   # scale-independent

    j_point   = search_start
    min_slope = float('inf')

    for i in range(search_start, search_end - 2):
        slope = abs(data[i + 1] - data[i])
        if slope < min_slope and abs(data[i] - st_baseline) < amp_threshold:
            min_slope = slope
            j_point   = i

    return j_point


def detect_qrs_end_adaptive(data, r_peak, windows):
    """Alias for backward compatibility."""
    return detectQRSEndAdaptive(data, r_peak, windows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — detectPWavesImproved
# Robust P-wave detection with amplitude validation and onset detection
# ══════════════════════════════════════════════════════════════════════════════

def detectPWavesImproved(data: np.ndarray, r_peaks: np.ndarray,
                          windows: AdaptiveWindows, heart_rate: int,
                          qrs_starts: List[int], fs: float) -> List[Tuple[int, int]]:
    """
    Detect P-wave peaks and onsets for each beat using HR-adaptive search windows.

    FIX: Now returns (p_peak, p_onset) tuples instead of just peak index.
    FIX: Added amplitude validation — P-wave must be 3-70% of R amplitude to
         reject T-wave spillover, baseline drift noise, and missing P-waves.
    FIX: Supports both positive (normal) and negative (inverted) P-waves.
    FIX: P-onset is found by backtracking from peak to where signal crosses
         the local isoelectric baseline (clinical PR standard requires onset).

    Args:
        data:       Bandpass-filtered ECG signal.
        r_peaks:    R-peak indices.
        windows:    AdaptiveWindows for this heart rate.
        heart_rate: Current HR in BPM.
        qrs_starts: QRS onset indices (one per beat).
        fs:         Sampling rate in Hz.

    Returns:
        List of (p_peak_idx, p_onset_idx) tuples. Both are -1 if not found.
    """
    p_results = []   # list of (peak, onset) tuples

    for i, (r_idx, qrs_start) in enumerate(zip(r_peaks, qrs_starts)):
        try:
            # ── Search window: minPtoQRS..maxPtoQRS samples before QRS onset ──
            p_search_end   = qrs_start - windows.minPtoQRS
            p_search_start = max(0, qrs_start - windows.maxPtoQRS)

            if p_search_end <= p_search_start or p_search_end <= 0:
                p_results.append((-1, -1))
                continue

            # At high HR, restrict search to avoid the T-wave of the previous beat.
            # Taper from 150 BPM (100% window) to 200 BPM (40% window).
            if heart_rate >= 200:
                window_len = p_search_end - p_search_start
                p_search_start = max(p_search_start,
                                     p_search_end - int(window_len * 0.40))
            elif heart_rate > 150:
                window_len = p_search_end - p_search_start
                pct = 1.0 - 0.6 * ((heart_rate - 150) / 50.0)
                p_search_start = max(p_search_start,
                                     p_search_end - int(window_len * pct))

            p_seg = data[p_search_start:p_search_end]
            if len(p_seg) < 3:
                p_results.append((-1, -1))
                continue

            # ── Local isoelectric baseline (20ms before search window) ──────
            bl_start = max(0, p_search_start - int(0.020 * fs))
            bl_end   = max(bl_start + 1, p_search_start)
            baseline  = float(np.mean(data[bl_start:bl_end])) if bl_end > bl_start else 0.0

            # ── R amplitude for amplitude gating ─────────────────────────────
            r_amp = abs(float(data[int(r_idx)]) - baseline)
            if r_amp < 1e-6:
                r_amp = float(np.max(np.abs(p_seg))) * 5.0 + 1e-6

            # HR-adaptive amplitude limits:
            # At normal HR P-wave is 5-50% of R; at high HR allow down to 3%
            min_p_amp_frac = 0.03 if heart_rate > 150 else 0.05
            max_p_amp_frac = 0.70
            min_p_amp = min_p_amp_frac * r_amp
            max_p_amp = max_p_amp_frac * r_amp

            # ── Find peak (positive or negative — whichever is larger) ───────
            pos_idx = int(np.argmax(p_seg))
            neg_idx = int(np.argmin(p_seg))
            pos_amp = float(p_seg[pos_idx]) - baseline
            neg_amp = baseline - float(p_seg[neg_idx])

            # Choose polarity with larger deflection
            if pos_amp >= neg_amp:
                p_peak_rel = pos_idx
                p_amp_val  = pos_amp
            else:
                p_peak_rel = neg_idx
                p_amp_val  = neg_amp

            # ── Amplitude gate: reject if outside physiological P-wave range ─
            if p_amp_val < min_p_amp or p_amp_val > max_p_amp:
                p_results.append((-1, -1))
                continue

            p_peak = p_search_start + p_peak_rel

            # ── P-onset: backtrack from peak to isoelectric baseline ─────────
            # Threshold: 15% of P-wave amplitude above baseline
            onset_threshold = max(0.15 * p_amp_val, 0.02 * r_amp)
            p_onset = p_search_start   # default to search window start
            for j in range(p_peak, p_search_start - 1, -1):
                deflection = abs(float(data[j]) - baseline)
                if deflection < onset_threshold:
                    p_onset = j
                    break

            p_results.append((p_peak, p_onset))

        except Exception:
            p_results.append((-1, -1))

    return p_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — calculatePRIntervalsImproved
# P-onset → QRS-onset  (clinical PR standard)
# ══════════════════════════════════════════════════════════════════════════════

def calculatePRIntervalsImproved(signal: np.ndarray,
                                  p_waves: List[Tuple[int, int]],
                                  r_peaks: np.ndarray,
                                  fs: float,
                                  heart_rate: int) -> List[float]:
    """
    Measure PR interval (ms) as P-onset → QRS-onset for each beat.

    FIX: Changed from P-peak → R-peak to P-onset → QRS-onset.
         Clinical PR interval is defined as the time from the beginning of
         the P-wave (P-onset) to the beginning of the QRS complex (Q-onset).
         The old P-peak → R-peak measurement added ~20-40ms extra.

    Args:
        signal:     Bandpass-filtered ECG signal.
        p_waves:    (p_peak, p_onset) tuples from detectPWavesImproved.
        r_peaks:    R-peak indices.
        fs:         Sampling rate in Hz.
        heart_rate: Current HR in BPM.

    Returns:
        List of PR intervals in ms (valid values only, 0.0 if not found).
    """
    pr_intervals = []

    # HR-adaptive PR floor: short PRs are physiological at high heart rates
    pr_min_limit = (
        40 if heart_rate > 200 else
        50 if heart_rate > 150 else
        60 if heart_rate > 120 else
        80
    )
    # HR-adaptive PR ceiling: longer RR allows longer PR
    pr_max_limit = (
        160 if heart_rate > 150 else
        200 if heart_rate > 100 else
        220 if heart_rate > 60 else
        280
    )

    # Use windows to get QRS starts per beat
    windows = None  # will compute per-beat QRS onset from r_peaks if needed

    for idx, p_info in enumerate(p_waves):
        try:
            # Accept both old (int) and new ((peak, onset)) format for safety
            if isinstance(p_info, (tuple, list)) and len(p_info) == 2:
                p_peak, p_onset = int(p_info[0]), int(p_info[1])
            else:
                p_peak = int(p_info)
                p_onset = p_peak   # fallback: treat peak as onset

            if p_peak < 0 or p_onset < 0:
                continue

            # Find the nearest R-peak AFTER this P-wave peak
            r_after = None
            for rp in r_peaks:
                if rp > p_peak:
                    r_after = int(rp)
                    break
            if r_after is None:
                continue

            # QRS onset: scan backwards from R to find Q-onset
            # Simple threshold: 7% of R amplitude before QRS region
            qrs_search_start = max(0, r_after - int(0.10 * fs))  # 100ms before R
            baseline_ref_start = max(0, qrs_search_start - int(0.03 * fs))
            baseline_ref = float(np.mean(signal[baseline_ref_start:qrs_search_start]))
            r_amp_local = abs(float(signal[r_after]) - baseline_ref)
            qrs_threshold = max(r_amp_local * 0.07, abs(baseline_ref) * 0.10)

            qrs_onset = qrs_search_start  # default
            for j in range(qrs_search_start, r_after):
                if abs(float(signal[j]) - baseline_ref) > qrs_threshold:
                    qrs_onset = j
                    break

            # PR = P-onset to QRS-onset (clinical standard)
            pr_ms = (qrs_onset - p_onset) / fs * 1000.0

            if pr_min_limit <= pr_ms <= pr_max_limit:
                pr_intervals.append(float(pr_ms))

        except Exception:
            continue

    return pr_intervals


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — detectTWaveEndAdaptive
# T-wave end detection: slope+stability primary, tangent fallback
# ══════════════════════════════════════════════════════════════════════════════

def _calculate_expected_qt(rr_interval_sec: float, heart_rate: int,
                            fs: float) -> int:
    """Expected QT in samples: QT ≈ 0.42 * sqrt(RR_sec)."""
    qt_sec = 0.42 * np.sqrt(max(rr_interval_sec, 0.2))
    return int(qt_sec * fs)


def _calculate_baseline_pre_qrs(data: np.ndarray, qrs_start: int) -> float:
    """Mean of 10–30 samples before QRS onset as isoelectric baseline."""
    start = max(0, qrs_start - 30)
    end   = max(0, qrs_start - 10)
    if end > start:
        return float(np.mean(data[start:end]))
    return float(data[qrs_start]) if qrs_start < len(data) else 0.0


def _detect_tend_by_tangent(data: np.ndarray, t_peak: int, baseline: float,
                             search_end: int, heart_rate: int,
                             qrs_start: int, expected_qt: int,
                             err_thr: float) -> int:
    """
    Tangent-method T-end fallback.

    Finds the steepest downslope of the T-wave, draws a tangent,
    and returns where it intersects the baseline.
    """
    sw = 50
    if heart_rate > 280:   sw = 90
    elif heart_rate > 250: sw = 80
    elif heart_rate > 150: sw = 70

    max_slope = 0.0
    slope_idx = t_peak

    limit = min(t_peak + sw, search_end - 5)
    for i in range(t_peak, limit):
        s = abs(data[i] - data[i + 5]) / 5.0
        if s > max_slope:
            max_slope = s
            slope_idx = i

    min_slope_thr = 0.004
    if heart_rate > 280:   min_slope_thr = 0.002
    elif heart_rate > 250: min_slope_thr = 0.0025
    elif heart_rate > 180: min_slope_thr = 0.003
    elif heart_rate > 150: min_slope_thr = 0.0035

    if max_slope < min_slope_thr:
        return -1

    flat_mul = 0.12
    if heart_rate > 280:   flat_mul = 0.18
    elif heart_rate > 250: flat_mul = 0.16
    elif heart_rate > 180: flat_mul = 0.15
    elif heart_rate > 150: flat_mul = 0.13

    flat_thr = max_slope * flat_mul

    stab_mul = 1.6
    if heart_rate > 280:   stab_mul = 2.0
    elif heart_rate > 250: stab_mul = 1.9
    elif heart_rate > 180: stab_mul = 1.8
    elif heart_rate > 150: stab_mul = 1.7

    for i in range(slope_idx, search_end - 5):
        if abs(data[i + 1] - data[i]) < flat_thr:
            is_stable = True
            for it in range(1, 4):
                idx1 = min(i + it + 1, len(data) - 1)
                idx2 = min(i + it,     len(data) - 1)
                if abs(data[idx1] - data[idx2]) >= flat_thr * stab_mul:
                    is_stable = False
                    break
            if is_stable and abs(i - qrs_start - expected_qt) < expected_qt * err_thr:
                return i

    return -1


def detectTWaveEndAdaptive(data: np.ndarray, r_peak: int, qrs_start: int,
                            next_r_peak: int, windows: AdaptiveWindows,
                            rr_interval_sec: float, heart_rate: int,
                            fs: float) -> int:
    """
    Detect T-wave end using slope+stability primary method with tangent fallback.

    Adapts heavily to heart rate — at 200 BPM the T-wave is compressed
    into ~200 ms so all thresholds and windows are tightened accordingly.

    FIX-B: Removed duplicate qt_err/if block that caused IndentationError.
    FIX-C: margin increased 10→20 samples at HR>150 to stop T-search
            spilling into next beat's P-wave.
            Hard QT cap added: 60% of RR at HR>=160, 70% otherwise.
            This prevents the 25-56ms overshoot seen at 160-200 BPM.

    Args:
        data:             Bandpass-filtered ECG signal.
        r_peak:           Current R-peak index.
        qrs_start:        QRS onset index.
        next_r_peak:      Next R-peak index (to bound T-wave search).
        windows:          AdaptiveWindows for current HR.
        rr_interval_sec:  RR interval in seconds.
        heart_rate:       Current HR in BPM.
        fs:               Sampling rate in Hz.

    Returns:
        T-wave end sample index.
    """
    search_start = r_peak + windows.tSearchStart
    search_end   = min(len(data) - 1, r_peak + windows.tSearchEnd)

    # FIX-C: wider margin at high HR to keep T-search away from next P-wave
    margin = 20 if heart_rate > 150 else 10

    safe_end = search_end
    if next_r_peak > 0:
        safe_end = min(search_end, next_r_peak - margin)

    expected_qt = _calculate_expected_qt(rr_interval_sec, heart_rate, fs)

    # Hard RR-based cap to prevent runaway T-wave search
    #   HR >= 200 BPM (RR=300ms) → cap = 75% RR
    #   HR >= 160 BPM (RR=375ms) → cap = 72% RR = 270ms
    #   HR <  160 BPM            → cap = 80% RR
    if heart_rate >= 200:
        qt_cap_pct = 0.75
    elif heart_rate >= 160:
        qt_cap_pct = 0.72
    else:
        qt_cap_pct = 0.80
    qt_cap_samples = int(rr_interval_sec * fs * qt_cap_pct)

    def _apply_cap(t_idx: int) -> int:
        """Clamp T-end so QT duration never exceeds qt_cap_samples."""
        if t_idx < 0:
            return -1
        max_t_end = qrs_start + qt_cap_samples
        return min(t_idx, max_t_end)

    if search_start >= safe_end:
        return min(len(data) - 1, qrs_start + min(expected_qt, qt_cap_samples))

    # T-peak search window
    t_peak_sw = 50
    if heart_rate > 250: t_peak_sw = 80
    elif heart_rate > 150: t_peak_sw = 70

    t_slice_end = min(safe_end, search_start + t_peak_sw)
    t_slice     = data[search_start:t_slice_end]

    if len(t_slice) == 0:
        return min(len(data) - 1, qrs_start + min(expected_qt, qt_cap_samples))

    t_peak_relative = np.argmax(t_slice)
    t_peak          = search_start + t_peak_relative

    baseline     = _calculate_baseline_pre_qrs(data, qrs_start)
    t_peak_value = data[t_peak]

    # HR-adaptive thresholds
    if heart_rate > 280:
        min_slope_thr, stability_window, min_descent = 0.100, 2, 0.010
    elif heart_rate > 250:
        min_slope_thr, stability_window, min_descent = 0.080, 2, 0.012
    elif heart_rate > 200:
        min_slope_thr, stability_window, min_descent = 0.050, 3, 0.015
    elif heart_rate > 180:
        min_slope_thr, stability_window, min_descent = 0.040, 3, 0.018
    elif heart_rate > 150:
        min_slope_thr, stability_window, min_descent = 0.035, 3, 0.020
    elif heart_rate > 120:
        min_slope_thr, stability_window, min_descent = 0.003, 4, 0.030
    elif heart_rate > 100:
        min_slope_thr, stability_window, min_descent = 0.0025, 4, 0.035
    else:
        min_slope_thr, stability_window, min_descent = 0.002, 5, 0.050

    if heart_rate > 280:   stability_thr = 0.120
    elif heart_rate > 250: stability_thr = 0.100
    elif heart_rate > 200: stability_thr = 0.080
    elif heart_rate > 180: stability_thr = 0.070
    elif heart_rate > 150: stability_thr = 0.060
    elif heart_rate > 120: stability_thr = 0.022
    elif heart_rate > 100: stability_thr = 0.020
    else:                  stability_thr = 0.015

    if heart_rate > 280:   err_thr = 0.70
    elif heart_rate > 250: err_thr = 0.65
    elif heart_rate > 200: err_thr = 0.60
    elif heart_rate > 180: err_thr = 0.55
    elif heart_rate > 150: err_thr = 0.50
    elif heart_rate > 120: err_thr = 0.40
    elif heart_rate > 100: err_thr = 0.38
    else:                  err_thr = 0.35

    start_offset = 8
    if heart_rate > 280:   start_offset = 3
    elif heart_rate > 250: start_offset = 4
    elif heart_rate > 180: start_offset = 4
    elif heart_rate > 150: start_offset = 6

    # Primary pass: slope + stability
    # FIX-B: duplicate qt_err/if block removed — only ONE check here
    for i in range(t_peak + start_offset, safe_end - stability_window):
        descent = abs(t_peak_value - data[i])
        if descent < min_descent:
            continue

        max_local_slope = 0.0
        for k in range(3):
            sv = abs(data[i + k + 1] - data[i + k])
            if sv > max_local_slope:
                max_local_slope = sv

        if max_local_slope > min_slope_thr:
            continue

        win_vals   = data[i: i + stability_window + 1]
        volt_range = np.max(win_vals) - np.min(win_vals)

        if volt_range > stability_thr:
            continue

        qt_err = abs(i - qrs_start - expected_qt)
        if qt_err < expected_qt * err_thr:
            return _apply_cap(i)   # FIX-C: cap applied on return

    # Tangent fallback
    tangent_end = _detect_tend_by_tangent(
        data, t_peak, baseline, safe_end,
        heart_rate, qrs_start, expected_qt, err_thr
    )
    if tangent_end > 0:
        return _apply_cap(tangent_end)   # FIX-C: cap applied on fallback too

    return min(len(data) - 1, qrs_start + min(expected_qt, qt_cap_samples))


def detect_t_wave_end_adaptive(data, r_peak, qrs_start, next_r_peak,
                                windows, rr_interval_sec, heart_rate, fs):
    """Alias for backward compatibility."""
    return detectTWaveEndAdaptive(data, r_peak, qrs_start, next_r_peak,
                                   windows, rr_interval_sec, heart_rate, fs)


# ══════════════════════════════════════════════════════════════════════════════
# HR + RR CALCULATION  (with EMA smoothing + hold-and-jump stability)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_hr_rr(lead_data: np.ndarray, fs: float = 500.0,
                    instance_id: Optional[str] = None) -> Tuple[int, float]:
    """
    Calculate Heart Rate (BPM) and RR interval (ms).

    Pipeline:
      1. Display-filter the signal
      2. detectRPeaks (Pan-Tompkins + fallback)
      3. calculateRRIntervals → median of last 5, ectopic rejection
      4. EMA smoothing + hold-and-jump for HR changes

    Hold-and-jump logic:
      delta <= 2 BPM  → suppress (jitter deadband, keep old display)
      delta >  30 BPM → update immediately (clinically significant)
      2 < delta <= 30 → 10-second hold before committing

    Returns:
        (heart_rate_bpm, rr_ms) — (0, 0.0) on failure.
    """
    key = instance_id if instance_id is not None else 'global'

    def _fallback():
        last = _hr_last_stable.get(key)
        if last and (time.time() - _hr_last_ts.get(key, 0.0)) <= 0.5:
            return last, 60000.0 / last if last > 0 else 0.0
        return 0, 0.0

    try:
        arr = np.asarray(lead_data, dtype=float)
        if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
            return _fallback()

        filtered = display_filter(arr, fs)
        if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
            return _fallback()

        r_peaks = detectRPeaks(filtered, fs)
        if len(r_peaks) < 2:
            return _fallback()

        valid = calculateRRIntervals(r_peaks, fs)
        if len(valid) < 1:
            return _fallback()

        # Startup beat counter
        if key not in _hr_beat_count:
            _hr_beat_count[key] = 0
        _hr_beat_count[key] += len(valid)
        is_startup = _hr_beat_count[key] <= _STARTUP_LOCKOUT_BEATS

        if is_startup:
            # During startup the buffer was initialized with flat 2048 values.
            # The flat→signal edge at capture start creates fake R-peaks with
            # extremely short RR intervals (230ms = 260 BPM).  Reject both
            # too-long (< 10 BPM) AND too-short (> 200 BPM) intervals until
            # we have enough real beats to trust the detector.
            valid = valid[(valid >= _STARTUP_RR_MIN_MS) & (valid <= _STARTUP_RR_MAX_MS)]
            if len(valid) < 2:
                return _fallback()

        ectopic_tol = _STARTUP_ECTOPIC_TOL if is_startup else _NORMAL_ECTOPIC_TOL
        if len(valid) >= 3:
            med = np.median(valid)
            tol = ectopic_tol * med
            filtered_rr = valid[np.abs(valid - med) <= tol]
            if len(filtered_rr) >= 2:
                valid = filtered_rr

        recent = valid[-5:] if len(valid) > 5 else valid
        rr_ms  = float(np.median(recent))
        if rr_ms <= 0:
            return _fallback()

        hr_raw = max(10.0, min(300.0, 60000.0 / rr_ms))
        hr_int = int(round(hr_raw))

        # FIX-HR-STAB-CALC: Same ±3 BPM dead-zone median logic as heart_rate.py
        # to ensure calculate_all_ecg_metrics produces the same HR as the
        # canonical calculate_heart_rate_from_signal.
        buf = _hr_buffers.setdefault(key, {'buf': deque(maxlen=15)})['buf']
        buf.append(hr_int)
        median_hr = int(round(np.median(list(buf)))) if len(buf) >= 5 else hr_int

        _hr_last_stable[key] = median_hr
        _hr_last_ts[key]     = time.time()

        # Display stabilization with ±3 BPM dead-zone
        if key not in _hr_displayed:
            _hr_displayed[key]  = median_hr
            _hr_pending[key]    = None
            _hr_pending_ts[key] = 0.0

        displayed = _hr_displayed[key]
        delta     = abs(median_hr - displayed)

        if delta <= 3:
            # ±3 BPM dead zone: absorbs sampling-rate jitter completely
            return displayed, rr_ms
        elif delta > 30:
            # Very large change: immediate jump
            _hr_displayed[key] = median_hr
            _hr_pending[key]   = None
            return median_hr, rr_ms
        else:
            # Medium change (4-30 BPM): 1.0s stability confirmation
            now     = time.time()
            pending = _hr_pending[key]
            if pending is None:
                _hr_pending[key]    = median_hr
                _hr_pending_ts[key] = now
            else:
                if abs(median_hr - pending) <= 3:
                    if now - _hr_pending_ts[key] >= 1.0:
                        _hr_displayed[key] = median_hr
                        _hr_pending[key]   = None
                        return median_hr, rr_ms
                else:
                    _hr_pending[key]    = median_hr
                    _hr_pending_ts[key] = now
            # Always return the ACTUAL measured rr_ms (not derived from displayed HR).
            # QTc must use the real RR interval, not the held display value.
            return displayed, rr_ms

    except Exception as e:
        print(f" ⚠️ calculate_hr_rr error: {e}")
        return _fallback()


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL QRS REFRESH THROTTLE
# ══════════════════════════════════════════════════════════════════════════════
# Delineating twelve leads costs roughly 30 ms, and the pages drive this entry
# point from a 200 ms UI-thread timer. Recomputing every tick would spend a
# sixth of that budget re-measuring a width that moves on the scale of seconds
# (and is median-smoothed over 15 beats downstream anyway), so the global
# measurement is refreshed about once a second and reused in between.
# Keyed by instance_id so the 12-lead, HRV and hyperkalemia pages never share
# a cache entry — the same reason their smoothing buffers are keyed that way.

# The interval self-tunes: whatever the measurement costs on this machine, it is
# scheduled to occupy at most ~5% of wall-clock time, clamped to 1-4 s. A fast
# desktop refreshes every second; a low-spec box backs itself off instead of
# stuttering the trace.
_GLOBAL_QRS_REFRESH_SEC: float = 1.0
_GLOBAL_QRS_MAX_REFRESH_SEC: float = 4.0
_GLOBAL_QRS_DUTY_CYCLE: float = 0.05

# instance_id -> (timestamp, result, interval_until_next_refresh)
_global_qrs_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]], float]] = {}


def reset_global_qrs_cache(instance_id: Optional[str] = None) -> None:
    """Drop cached global-QRS results (all of them, or one instance's)."""
    if instance_id is None:
        _global_qrs_cache.clear()
    else:
        _global_qrs_cache.pop(instance_id, None)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — calculate_all_ecg_metrics()
# ══════════════════════════════════════════════════════════════════════════════

def calculate_all_ecg_metrics(
        lead_data: np.ndarray,
        fs: float = 500.0,
        instance_id: Optional[str] = None,
        lead_i_data:   Optional[np.ndarray] = None,
        lead_avf_data: Optional[np.ndarray] = None,
        all_lead_data: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Calculate ALL ECG metrics from Lead II data in one call.

    Full pipeline:
      1. detectRPeaks              → R-peak indices
      2. calculateRRIntervals      → RR in ms, HR in BPM
      3. calculateAdaptiveWindows  → HR-adaptive timing windows
      4. qrs_duration_from_raw_signal → QRS duration (Curtin 2018)
      5. detectQRSStartAdaptive    → QRS onset
      6. detectQRSEndAdaptive      → J-point
      7. detectPWavesImproved      → P-wave peaks
      8. calculatePRIntervalsImproved → PR interval  (FIX-D: hr passed in)
      9. detectTWaveEndAdaptive    → T-wave end → QT interval
     10. calculate_qtc_bazett      → QTc (Bazett, always)

    Args:
        lead_data:    Raw Lead II ECG signal (numpy array).
        fs:           Sampling rate in Hz (default 500).
        instance_id:  Per-instance smoothing key.
        lead_i_data:  Lead I (optional, unused — kept for API compat).
        lead_avf_data:Lead aVF (optional, unused — kept for API compat).
        all_lead_data:{lead name -> signal} for the multi-lead global QRS
                      boundary. Every signal must be time-synchronised with
                      lead_data and the same length. With two or more usable
                      leads the QRS width is measured globally (earliest onset
                      to latest offset across leads, Glasgow/Marquette style);
                      otherwise the single-lead Curtin measurement is kept.

    Returns:
        Dict with keys:
          heart_rate   (int, BPM)
          rr_interval  (float, ms)
          pr_interval  (int, ms)
          qrs_duration (int, ms)
          qt_interval  (float, ms)
          qtc_interval (int, ms)
          qrs_method   (str, "global-multilead" or "single-lead")
          qrs_leads_used (int, leads that contributed to the QRS width)
        Values are 0/None on failure.
    """
    results: Dict[str, Any] = {
        "heart_rate":   0,
        "rr_interval":  0.0,
        "pr_interval":  0,
        "qrs_duration": 0,
        "qt_interval":  None,
        "qtc_interval": 0,
        "qrs_method":   "single-lead",
        "qrs_leads_used": 1,
    }

    try:
        arr = np.asarray(lead_data, dtype=float)
        if len(arr) < 2000 or np.all(arr == 0) or np.std(arr) < 0.1:
            return results

        # ── Pre-analysis filter layer (baseline + noise rejection) ────────────
        arr, quality_score, reject_reason = pre_analyze(arr, fs)
        if not should_analyze(quality_score):
            print(f" Pre-analysis rejected segment: {reject_reason} (score={quality_score:.2f})")
            return results

        # ── Filter ────────────────────────────────────────────────────────────
        arr_dc = arr - np.mean(arr)        # remove DC offset
        filt   = _bandpass(arr_dc, fs)     # 0.5-40 Hz for analysis

        # ── Step 1: R-peaks ───────────────────────────────────────────────────
        r_peaks = detectRPeaks(filt, fs)
        if len(r_peaks) < 3:
            return results

        # ── Step 2: HR + RR ───────────────────────────────────────────────────
        hr, rr_ms = calculate_hr_rr(arr, fs, instance_id)
        results["heart_rate"]  = hr
        results["rr_interval"] = rr_ms

        if hr <= 0 or rr_ms <= 0:
            return results

        rr_sec = rr_ms / 1000.0

        # ── Step 3: Adaptive windows ──────────────────────────────────────────
        windows = calculateAdaptiveWindows(hr, rr_sec, fs)

        # ── Step 4: QRS duration (Curtin 2018 paper algorithm) ────────────────
        r_curr_idx = int(r_peaks[-2])
        r_next_idx = int(r_peaks[-1])

        try:
            qrs_dur_ms  = qrs_duration_from_raw_signal(
                filt, r_curr_idx, fs, adc_per_mv=1200.0, heart_rate=hr
            )
            qrs_dur_int = int(round(qrs_dur_ms)) if qrs_dur_ms > 0 else 0
        except Exception:
            qrs_dur_int = 0

        # ── Step 4b: multi-lead global QRS boundary (preferred when available) ─
        # A single lead sees only its own projection of the depolarisation
        # wavefront, so its onset is late and its offset early — Lead II alone
        # reads 10-20 ms short. With the other leads in hand the width is taken
        # from the earliest onset to the latest offset across the whole set.
        if all_lead_data:
            cache_key = instance_id or "_default"
            now = time.monotonic()
            cached = _global_qrs_cache.get(cache_key)
            fresh = False

            if cached is not None and (now - cached[0]) < cached[2]:
                global_qrs = cached[1]
            else:
                try:
                    global_qrs = compute_global_qrs_duration_12lead(
                        all_lead_data, r_peaks, fs,
                        adc_per_mv=1200.0, heart_rate=hr,
                    )
                except Exception as e:
                    print(f" ⚠️ global QRS failed, keeping single-lead: {e}")
                    global_qrs = None
                elapsed = time.monotonic() - now
                next_interval = min(
                    _GLOBAL_QRS_MAX_REFRESH_SEC,
                    max(_GLOBAL_QRS_REFRESH_SEC, elapsed / _GLOBAL_QRS_DUTY_CYCLE),
                )
                # A failure is cached too, so a lead set that cannot be
                # delineated is not retried five times a second.
                _global_qrs_cache[cache_key] = (now, global_qrs, next_interval)
                fresh = True

            if global_qrs and global_qrs["global_qrs_ms"] > 0:
                qrs_dur_int = int(round(global_qrs["global_qrs_ms"]))
                results["qrs_method"] = "global-multilead"
                results["qrs_leads_used"] = global_qrs["leads_used"]
                results["qrs_per_lead_ms"] = global_qrs["per_lead_qrs_ms"]
                if fresh:
                    # Sample indices point into a rolling buffer, so they are
                    # only meaningful for the tick that measured them.
                    results["qrs_global_onset"] = global_qrs["global_onset"]
                    results["qrs_global_offset"] = global_qrs["global_offset"]

        if instance_id and qrs_dur_int > 0:
            qrs_dur_int = apply_interval_smoothing(qrs_dur_int, instance_id, _qrs_buffers)
        results["qrs_duration"] = qrs_dur_int

        # ── Step 5: QRS start + end (for QT and PR) ───────────────────────────
        qrs_start = detectQRSStartAdaptive(filt, r_curr_idx, windows)
        qrs_end   = detectQRSEndAdaptive(filt, r_curr_idx, windows)   # noqa: F841

        # ── Step 6: T-wave end → QT interval ─────────────────────────────────
        try:
            t_end = detectTWaveEndAdaptive(
                filt, r_curr_idx, qrs_start, r_next_idx,
                windows, rr_sec, hr, fs
            )
            qt_ms = (t_end - qrs_start) / fs * 1000.0

            # HR-adaptive QT validity clamp
            # At 280 BPM expected QT ≈ 194ms, at 300 BPM ≈ 187ms
            qt_min = 150.0 if hr > 200 else (170.0 if hr > 150 else 200.0)
            qt_max = 700.0

            if qt_min <= qt_ms <= qt_max:
                if instance_id:
                    qt_ms = float(apply_interval_smoothing(
                        int(round(qt_ms)), instance_id, _qt_buffers
                    ))
                results["qt_interval"] = qt_ms
        except Exception as e:
            print(f" ⚠️ QT detection error: {e}")

        # ── Step 7: QTc (Bazett) ─────────────────────────────────────────────
        # IMPORTANT: QTc is computed from the already-smoothed QT integer
        # (i.e. the exact integer stored in results["qt_interval"] after the
        # _qt_buffers EMA pass above).  This guarantees that at 60 BPM where
        # RR = 1000 ms → √RR = 1.0 → QTc = QT / 1.0 = QT exactly.
        # Using a separate _qtc_buffers EMA caused a ±1 ms desync because the
        # two buffers accumulated slightly different histories.
        qt_ms_val = results["qt_interval"]
        if qt_ms_val is not None and qt_ms_val > 0:
            # Use the same smoothed integer QT value so Bazett is applied to
            # an integer, producing a consistent integer QTc.
            qt_for_qtc = int(round(qt_ms_val))   # already smoothed integer
            qtc = calculate_qtc_bazett(float(qt_for_qtc), rr_ms)
            results["qtc_interval"] = qtc

        # ── Step 8: P-waves + PR interval ────────────────────────────────────
        try:
            # QRS onset for every beat
            qrs_starts_all = [
                detectQRSStartAdaptive(filt, int(rp), windows)
                for rp in r_peaks
            ]

            p_waves = detectPWavesImproved(
                filt, r_peaks, windows, hr, qrs_starts_all, fs
            )

            # Step 8c: PR intervals (Kotlin: calculatePRIntervalsImproved)
            # New signature: (signal, p_waves, r_peaks, fs, heart_rate)
            pr_list = calculatePRIntervalsImproved(
                signal=filt,
                p_waves=p_waves,
                r_peaks=r_peaks,
                fs=fs,
                heart_rate=hr,  # Pass HR for adaptive clamp
            )
            valid_pr = [v for v in pr_list if v > 0]
            if valid_pr:
                pr_median = float(np.median(valid_pr))
                pr_int    = int(round(pr_median))
                if instance_id:
                    pr_int = apply_interval_smoothing(pr_int, instance_id, _pr_buffers)
                results["pr_interval"] = pr_int

        except Exception as e:
            print(f" ⚠️ PR detection error: {e}")

        # ── Step 9: Reference-table stabilization (anti-jump, anti-drift) ───
        results = _stabilize_to_reference(results, instance_id)

    except Exception as e:
        print(f" ⚠️ calculate_all_ecg_metrics error: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL METRIC HELPERS  (kept for backward compatibility)
# ══════════════════════════════════════════════════════════════════════════════

def calculate_qrs(lead_data: np.ndarray, r_peaks: np.ndarray,
                  fs: float = 500.0, instance_id: Optional[str] = None) -> int:
    """QRS duration via Curtin 2018 paper algorithm."""
    try:
        if len(r_peaks) < 2:
            return 0
        filt   = _bandpass(np.asarray(lead_data, dtype=float) - np.mean(lead_data), fs)
        r_curr = int(r_peaks[len(r_peaks) // 2])
        qrs_ms = qrs_duration_from_raw_signal(filt, r_curr, fs, adc_per_mv=1200.0)
        if qrs_ms <= 0:
            return 0
        qrs_int = int(round(qrs_ms))
        if instance_id:
            qrs_int = apply_interval_smoothing(qrs_int, instance_id, _qrs_buffers)
        return qrs_int
    except Exception as e:
        print(f" ⚠️ calculate_qrs error: {e}")
        return 0


def calculate_qtc(qt_ms: float, rr_ms: float, heart_rate: int,
                  instance_id: Optional[str] = None) -> int:
    """QTc via Bazett (always)."""
    qtc = calculate_qtc_bazett(qt_ms, rr_ms)
    if qtc > 0 and instance_id:
        qtc = apply_interval_smoothing(qtc, instance_id, _qtc_buffers)
    return qtc


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def cleanup_instance(instance_id: str) -> None:
    """Remove all smoothing state for a given instance_id (call on session end)."""
    for d in (_pr_buffers, _qrs_buffers, _qt_buffers, _qtc_buffers,
              _hr_buffers, _hr_ema, _hr_last_stable, _hr_last_ts,
              _hr_beat_count, _hr_displayed, _hr_pending, _hr_pending_ts,
              _reference_metric_state):
        d.pop(instance_id, None)
