"""
ECG Filter Module - Medical-Grade Filtering for ECG Signals

This module provides medical-grade filtering for ECG signals:
- Respiration-Preserving Baseline Correction (MONITOR-GRADE): Removes motion/electrode drift while preserving respiration
- Baseline Wander Removal (GOLD STANDARD): Median + Mean filter (commercial ECG monitor standard)
- AC Filter: Notch filter to remove 50Hz or 60Hz power line interference
- EMG Filter: High-pass filter to remove muscle artifacts (25Hz, 35Hz, 40Hz, 75Hz, 100Hz, 150Hz)
- DFT Filter: High-pass filter to remove baseline wander (0.05Hz, 0.5Hz)

Usage:
    from ecg.ecg_filters import apply_ecg_filters, ecg_with_respiratory_baseline
    
    # MONITOR-GRADE: Respiration-preserving baseline correction (RECOMMENDED)
    clean_ecg, respiration = ecg_with_respiratory_baseline(signal, sampling_rate=500)
    
    # GOLD STANDARD: Median + Mean filter for baseline removal
    clean_signal = apply_baseline_wander_median_mean(signal, sampling_rate=500)
    
    # Or use full filter chain
    filtered_signal = apply_ecg_filters(
        signal, 
        sampling_rate=500,
        ac_filter="50",  # "off", "50", or "60"
        emg_filter="150",  # "25", "35", "40", "75", "100", "150"
        dft_filter="0.5"  # "off", "0.05", or "0.5"
    )
"""

import numpy as np
import re
from scipy.signal import butter, filtfilt, iirnotch, medfilt, find_peaks
from scipy.ndimage import uniform_filter1d
from numpy.lib.stride_tricks import sliding_window_view
from typing import List, Optional, Tuple, Union


def normalize_adc_signal(signal: np.ndarray, preserve_amplitude: bool = True) -> np.ndarray:
    """
    Normalize ADC signal at the very start of processing pipeline.
    Removes DC offset only (variance normalization disabled to prevent instability).
    
    Algorithm:
    1. Remove DC offset (mean subtraction)
    2. DO NOT normalize variance (preserves signal amplitude, prevents flickering)
    
    Args:
        signal: Raw ADC signal
        preserve_amplitude: If True, only remove DC offset (default: True)
    
    Returns:
        Normalized signal (zero mean, original amplitude preserved)
    """
    signal = np.asarray(signal, dtype=float)
    
    if len(signal) == 0:
        return signal
    
    # Remove DC offset only (do NOT normalize variance to prevent instability)
    normalized = signal - np.mean(signal)
    
    # Variance normalization disabled - causes waves to "come and go" when applied to sliding windows
    # The variance changes between updates, causing amplitude instability
    
    return normalized


def detect_qrs_regions(ecg: np.ndarray, fs: float = 500.0) -> np.ndarray:
    """
    Detect QRS regions for gated sharpening.
    Returns boolean mask: True in QRS regions, False elsewhere.
    
    Algorithm:
    1. Find R-peaks using peak detection
    2. Create QRS windows (±80 ms around each R-peak)
    3. Return mask indicating QRS regions
    
    Args:
        ecg: ECG signal
        fs: Sampling rate (Hz)
    
    Returns:
        Boolean array: True where QRS regions are located
    """
    if len(ecg) < 10:
        return np.zeros(len(ecg), dtype=bool)
    
    try:
        # Find R-peaks (simple peak detection)
        # Use absolute value to handle inverted leads
        abs_ecg = np.abs(ecg)
        threshold = np.percentile(abs_ecg, 75)  # 75th percentile as threshold
        
        # Find peaks with minimum distance (corresponds to ~200 BPM max)
        min_distance = int(0.3 * fs)  # 300 ms minimum between peaks
        
        peaks, _ = find_peaks(abs_ecg, height=threshold, distance=min_distance)
        
        # Create QRS region mask (±80 ms around each R-peak)
        qrs_window_ms = 80.0  # ±80 ms QRS window
        qrs_window_samples = int(qrs_window_ms * fs / 1000.0)
        
        mask = np.zeros(len(ecg), dtype=bool)
        for peak_idx in peaks:
            start = max(0, peak_idx - qrs_window_samples)
            end = min(len(ecg), peak_idx + qrs_window_samples + 1)
            mask[start:end] = True
        
        return mask
    except Exception:
        # Fallback: return all False (no sharpening)
        return np.zeros(len(ecg), dtype=bool)


def sharpen_qrs_gated(ecg: np.ndarray, fs: float = 500.0, alpha: float = 0.3) -> np.ndarray:
    """
    Sharpen QRS complexes with gating - only applies to QRS regions, not P or T waves.
    Monitor-grade: Preserves ST segment and QT interval.
    
    Algorithm:
    1. Detect QRS regions
    2. Calculate first derivative (highlights QRS edges)
    3. Apply sharpening only in QRS regions
    4. Leave P and T waves unchanged
    
    Args:
        ecg: EMG-suppressed ECG signal
        fs: Sampling rate (Hz)
        alpha: Sharpening factor (0.2-0.4, default 0.3)
    
    Returns:
        Sharpened ECG with preserved P and T waves
    """
    if len(ecg) < 3:
        return ecg
    
    try:
        # Detect QRS regions
        qrs_mask = detect_qrs_regions(ecg, fs)
        
        if not np.any(qrs_mask):
            # No QRS detected, return original
            return ecg
        
        # Calculate first derivative (emphasizes QRS edges)
        dt = 1.0 / fs
        derivative = np.gradient(ecg, dt)
        
        # Normalize derivative to prevent amplitude distortion
        if np.std(derivative) > 1e-10:
            derivative = derivative / np.std(derivative) * np.std(ecg)
        
        # Apply sharpening only in QRS regions
        sharpened = ecg.copy()
        sharpened[qrs_mask] = ecg[qrs_mask] + alpha * derivative[qrs_mask]
        
        return sharpened
    except Exception:
        return ecg


# One mains cycle is 20 ms at 50 Hz, so a two-second window holds a hundred of
# them against two to six heartbeats: long enough that a QRS cannot pull the
# fit far, short enough to follow a mains amplitude that drifts, and re-fitted
# every second because the windows overlap by half.
#
# Measured on three recordings from this device, as waveform added to the ST
# segment of the median beat (% of R height, 150 Hz muscle filter):
#
#            window   0.5s   1.0s   2.0s   3.0s   5.0s
#   paed 180/min      2.55   1.45   1.27   1.29   1.41
#   adult 60/min      3.70   2.15   1.58   0.98   0.85
#   paed 180/min (2)  2.20   0.76   0.71   0.67   0.60
#
# Past two seconds the gain is small and the filter grows slower to react to
# interference that comes and goes, so this is where it stops.
_MAINS_FIT_WINDOW_S = 2.0


# How far the supply may be found from its nominal 50 or 60 Hz before the
# measurement is treated as noise rather than as mains. Grid frequency is held
# to a few hundredths of a hertz by statute; anything past half a hertz is not
# the supply and the nominal value is kept instead.
_MAINS_MAX_DRIFT_HZ = 0.5

# Below this the supply is on its nominal frequency for our purposes and the
# second fit is skipped - a millihertz of fitting noise is not worth another
# pass over twelve leads on every plot tick.
_MAINS_REFIT_MIN_DRIFT_HZ = 0.001


def _fit_phasor_track(signal: np.ndarray, t: np.ndarray, f0: float,
                      starts: List[int], win: int):
    """Fit cos/sin at `f0` over each window. Returns (centres, cos_c, sin_c).

    THE SAME FIT, ARRIVED AT WITHOUT THE WASTE.

    This runs on every lead on every plot tick - twelve times, several times a
    second - and was measured as the single largest cost in the render path.
    Two things in the original loop were pure overhead:

      the cosine and sine were recomputed for each window, though the windows
      overlap by half and every sample's value is the same whichever window it
      is seen from. They are now evaluated once for the whole signal and
      sliced.

      np.linalg.lstsq solves by SVD, which is the right tool when a design
      matrix may be rank-deficient. This one is four columns of cos, sin, a
      constant and a ramp over at least four mains cycles, which is very well
      conditioned, so the normal equations give the identical answer from a
      4x4 solve instead. lstsq is kept as the fallback for the singular case
      rather than removed, so a degenerate window still behaves as before.

    Verified identical, not merely close: across 480 lead-buffers from this
    device the largest difference between the two forms is 3.3e-11 ADC counts,
    which is 7.6e-13% of the signal range - floating-point summation order and
    nothing else. Measured 1.8x faster on the same buffers.
    """
    cos_all = np.cos(2.0 * np.pi * f0 * t)
    sin_all = np.sin(2.0 * np.pi * f0 * t)
    ones_win = np.ones(win)

    centres, cos_c, sin_c = [], [], []
    for s in starts:
        seg = signal[s:s + win]
        tt = t[s:s + win]
        ramp = tt - float(np.mean(tt))
        design = np.column_stack([cos_all[s:s + win],
                                  sin_all[s:s + win],
                                  ones_win, ramp])
        gram = design.T @ design
        try:
            coef = np.linalg.solve(gram, design.T @ seg)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(design, seg, rcond=None)
        centres.append(s + win / 2.0)
        cos_c.append(float(coef[0]))
        sin_c.append(float(coef[1]))
    return centres, cos_c, sin_c


# The supply frequency, once measured, reused until it goes stale.
#
# Refining costs a second full phasor-track fit, and it is paid per lead per
# frame: measured at 3.22 ms a lead on a 3 s window, 38.6 ms for twelve - more
# than a whole 30 ms frame budget, for a number that is a property of the wall
# socket and does not move. Measuring it once and reusing it brings the per
# lead cost back to a single fit.
#
# Keyed by nominal frequency only, deliberately: every lead is looking at the
# same supply, so lead I's measurement is lead V6's answer. The TTL exists
# because a generator or an inverter UPS can drift over minutes, not because
# the value is expected to change between frames.
# Below this fitted amplitude there is no mains tone worth measuring a
# frequency from, so the estimate is discarded rather than locked in. In ADC
# counts, against a mains that reads 150-320 counts on this device when it is
# present at all.
_MAINS_FIT_MIN_AMPLITUDE = 25.0

# How close two successive estimates must be before the frequency is trusted.
# The supply is steady to far better than this; the tolerance is here to reject
# a window that is still filling, not to track real drift.
_MAINS_FREQ_AGREE_HZ = 0.02

_MAINS_FREQ_CACHE: dict = {}
_MAINS_FREQ_PENDING: dict = {}

# Long, and reset at acquisition start instead (see reset_mains_freq_cache,
# called from the 12-lead Start path next to prepare_gc_for_realtime).
#
# Re-measuring costs ~28 ms on the lead that pays for it, against a 30-50 ms
# frame budget - one dropped frame. At a 20 s TTL that is a visible hitch three
# times a minute, which is the same periodic stutter this filter work set out
# to remove. Measuring once per acquisition puts that cost in the device
# handshake, which already takes hundreds of milliseconds and where nothing is
# on screen to stutter. The TTL that remains is a backstop for a session left
# running for hours on a supply that has since drifted.
_MAINS_FREQ_TTL_S = 900.0


def _cached_mains_freq(f0: float):
    """The measured supply frequency for this nominal, or None if stale."""
    import time as _time

    hit = _MAINS_FREQ_CACHE.get(f0)
    if hit is None:
        return None
    f_true, stamp = hit
    if _time.monotonic() - stamp > _MAINS_FREQ_TTL_S:
        return None
    return f_true


def _store_mains_freq(f0: float, f_true: float) -> None:
    import time as _time

    _MAINS_FREQ_CACHE[f0] = (f_true, _time.monotonic())


def reset_mains_freq_cache() -> None:
    """Forget the measured supply frequency (call when acquisition restarts)."""
    _MAINS_FREQ_CACHE.clear()
    _MAINS_FREQ_PENDING.clear()


def _refine_mains_freq(f0: float, centres: List[float],
                       cos_c: List[float], sin_c: List[float],
                       sampling_rate: float,
                       centre_times: Optional[np.ndarray] = None) -> float:
    """The supply's REAL frequency, read off the drift of the fit itself.

    A fit made at f0 against a tone that is really at f0 + df produces a phasor
    that rotates at df cycles per second. Unwrapping the phase across the
    window centres and taking its slope recovers df directly - one polyfit over
    a handful of points, with no search over candidate frequencies and no
    second pass over the signal.
    """
    if len(centres) < 3:
        return f0
    try:
        phase = np.unwrap(np.arctan2(np.asarray(sin_c, dtype=float),
                                     np.asarray(cos_c, dtype=float)))
        if centre_times is not None:
            t_c = np.asarray(centre_times, dtype=float)      # slip-corrected
        else:
            t_c = np.asarray(centres, dtype=float) / float(sampling_rate)
        slope = float(np.polyfit(t_c, phase, 1)[0])          # radians per second
    except Exception:
        return f0
    df = -slope / (2.0 * np.pi)
    if not np.isfinite(df) or abs(df) > _MAINS_MAX_DRIFT_HZ:
        return f0
    if abs(df) < _MAINS_REFIT_MIN_DRIFT_HZ:
        return f0
    return f0 + df


def _track_to_samples(idx: np.ndarray, centres: List[float],
                      coeffs: List[float]) -> np.ndarray:
    """np.interp, except the two ends CONTINUE the trend instead of holding flat.

    np.interp clamps outside the data it is given, so every sample before the
    first window centre and after the last was reconstructed with a phasor
    frozen at up to a full window's age. That is precisely the first and last
    half second of the strip - the part the reader sees at the edges of the
    printed page.
    """
    out = np.interp(idx, centres, coeffs)
    if len(centres) < 2:
        return out
    head = idx < centres[0]
    if np.any(head):
        slope = (coeffs[1] - coeffs[0]) / (centres[1] - centres[0])
        out[head] = coeffs[0] + slope * (idx[head] - centres[0])
    tail = idx > centres[-1]
    if np.any(tail):
        slope = (coeffs[-1] - coeffs[-2]) / (centres[-1] - centres[-2])
        out[tail] = coeffs[-1] + slope * (idx[tail] - centres[-1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE SLIPS: THE STREAM IS NOT ALWAYS CONTINUOUS IN TIME
#
# The tone subtraction below assumes sample n was taken at time n / fs. On the
# RhythmUltra that is not quite true: the device drops one sample roughly every
# 2,100 (about every 4.2 s at 500 Hz) while its packet counter shows no gap, so
# the host cannot see it as packet loss. Measured on four raw captures of a
# Fluke at 60/min with 50 Hz line noise (captures/ecg_capture_20260919_1009*):
# RR is 499.95 samples on three beats in four and exactly one sample shorter,
# 498.95, on the fourth, and the 50 Hz tone - a steady 578 counts - jumps by
# one sample's worth of phase (36 degrees) at the same instants, every
# 2,104-2,114 samples.
#
# A one-sample slip is clinically nothing (2 ms, once every four seconds), but
# a fitted sinusoid cannot follow a phase step, so the mains came back for a
# second or two around every slip: 13 counts left between slips, 120-240
# around them. That patchy "50 Hz still there with AC 50 on" is what the grid
# and the report showed.
#
# The fix restores the time base rather than the samples: find each slip from
# the phase of the tone itself, and fit on t = (n + slips so far) / fs. The
# tone is then continuous in the time it is fitted against, and the unchanged
# windowed fit removes it across the slip - on the same capture every 1 s
# window after start-up comes out at 13 counts. Nothing is detected, and
# nothing changes, on a trace without mains or without slips.
# ─────────────────────────────────────────────────────────────────────────────

# The tone's phase is read once per mains cycle (10 samples at 50 Hz / 500 Hz)
# and median-smoothed over this many cycles, which removes the one or two
# cycles a QRS disturbs without rounding off a step.
_SLIP_SMOOTH_CYCLES = 7
# Smallest timing step worth correcting, in samples (a quarter sample is 9
# degrees of 50 Hz). Below this the windowed fit follows it anyway.
_SLIP_MIN_STEP = 0.25
# Largest step believed to be a timing step rather than a disturbance.
_SLIP_MAX_STEP = 2.5
# A step must hold for this many cycles to count.
_SLIP_PERSIST_CYCLES = 4
# The step is kept only if modelling it cuts the local fit error to below this
# fraction of the error without it.
_SLIP_MIN_GAIN = 0.7
# A step this close to a whole number of samples is taken to be exactly that.
_SLIP_SNAP = 0.15
# How far either side of its first estimate the step's sample is searched, in
# mains cycles. One cycle: the first estimate comes from whole-cycle phases,
# and a QRS next to the step can put it most of a cycle out (7 samples was
# seen; a +-4 search could not reach back from there).
_SLIP_PIN_CYCLES = 1.0


def _verify_slip(y: np.ndarray, s: int, step: float, a: int, b: int,
                 w: float, amp: float) -> Optional[Tuple[int, float, bool]]:
    """(exact index, size, unsure) of the timing step near `s`, or None if not
    real. `unsure`: a QRS covered samples the index was chosen on.

    One sinusoid is fitted over y[a:b] on the time base with the step and
    without it. Samples much larger than the tone - a QRS - are left out so
    they cannot decide the answer.

    SIZE. Refined over a small grid around the estimate - the phase medians it
    starts from are a few hundredths of a sample out, and at the edge of a
    short window the fit's extrapolation magnifies that into visible ripple -
    then snapped to a whole sample when within _SLIP_SNAP of one, because a
    dropped sample is exactly one sample: across the twelve leads of one
    capture the raw estimates of the same drop ranged 0.90-1.16.

    WHERE. Checked sample by sample within one mains cycle of the estimate against
    the fitted tone. One sample out leaves a one-sample spike of up to 0.6x the
    mains amplitude; the twelve leads' estimates of one drop spread over three
    samples before this.

    Kept if, at its best size and place, it cuts the fit error below
    _SLIP_MIN_GAIN of the error without it.
    """
    seg = y[a:b]
    keep = np.abs(seg) <= 3.0 * amp
    if keep.sum() < 20:
        return None
    nn = np.arange(a, b, dtype=float)
    v = seg[keep]

    def _fit(tn):
        A = np.column_stack([np.cos(w * tn), np.sin(w * tn)])
        coef, *_ = np.linalg.lstsq(A[keep], v, rcond=None)
        r = v - A[keep] @ coef
        return float(r @ r), coef

    e_plain, _ = _fit(nn)
    if e_plain <= 0.0:
        return None
    after = nn >= s
    grid = step + np.linspace(-0.15, 0.15, 7)
    errs = [_fit(nn + g * after)[0] for g in grid]
    i = int(np.argmin(errs))
    best = float(grid[i])
    if 0 < i < len(grid) - 1:                      # parabola through the minimum
        e0, e1, e2 = errs[i - 1], errs[i], errs[i + 1]
        den = e0 - 2.0 * e1 + e2
        if den > 0:
            best += 0.5 * (e0 - e2) / den * (grid[1] - grid[0])
    if errs[i] >= _SLIP_MIN_GAIN * e_plain:
        return None
    # The same floor as detection, applied to the refined size: a QRS can
    # nudge the first estimate over a quarter sample and the refit then finds
    # a fraction of that (-0.17 on two leads of a capture with no slips).
    if abs(best) < _SLIP_MIN_STEP:
        return None
    whole = round(best)
    if whole != 0 and abs(best - whole) <= _SLIP_SNAP:
        best = float(whole)

    cands, errs_at, unsure = _pin_errors(y, amp, s, best, w, a, b)
    if cands.size == 0:
        return s, best, True
    return _pick_pin(cands, errs_at, s), best, unsure


def _pin_errors(y: np.ndarray, amp: float, s: int, step: float, w: float,
                a: int, b: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Fit error of each candidate index for a step of `step` near `s`.

    With the tone fitted over y[a:b] (step at s), only the samples between two
    candidate positions differ, so the candidates are compared on the samples
    around them alone. Samples the tone does not explain to within its own
    amplitude - the QRS - are left out: a slip that lands on a QRS (about one
    in five do) was otherwise pinned by the QRS's slope, not by the tone, and
    put up to nine samples out, which leaves a short 50 Hz burst on the QRS
    upstroke that reads like notching. Errors are divided by amp**2 so leads
    can be summed (see _shared_sample_slips). The third value says whether any
    sample near the candidates had to be left out - i.e. whether this lead
    alone could be misled.
    """
    seg = y[a:b]
    nn = np.arange(a, b, dtype=float)
    keep = np.abs(seg) <= 3.0 * amp
    if keep.sum() < 20:
        return np.array([], dtype=int), np.array([]), True
    tn0 = nn + step * (nn >= s)
    A = np.column_stack([np.cos(w * tn0), np.sin(w * tn0)])
    coef, *_ = np.linalg.lstsq(A[keep], seg[keep], rcond=None)
    pin = max(1, int(round(_SLIP_PIN_CYCLES * 2.0 * np.pi / w)))
    lo = max(a + 1, s - pin)
    hi = min(b - 1, s + pin)
    local = np.arange(max(a, lo - pin), min(b, hi + pin))
    yl = y[local]
    model0 = coef[0] * np.cos(w * (local + step * (local >= s))) +         coef[1] * np.sin(w * (local + step * (local >= s)))
    ok = np.abs(yl - model0) <= amp
    cands = np.arange(lo, hi + 1)
    # All candidates at once: (candidates, samples).
    tn = local[None, :] + step * (local[None, :] >= cands[:, None])
    r = (yl[None, :] - coef[0] * np.cos(w * tn) - coef[1] * np.sin(w * tn)) * ok[None, :]
    errs = np.einsum("ij,ij->i", r, r) / (amp * amp)
    return cands, errs, not bool(ok.all())


def _pick_pin(cands: np.ndarray, errs: np.ndarray, s: int) -> int:
    """Lowest error; among equal ones (all samples between them excluded as
    QRS, so nothing to choose on) the one nearest the first estimate."""
    best = float(np.min(errs))
    tied = cands[errs <= best * (1.0 + 1e-9) + 1e-12]
    return int(tied[np.argmin(np.abs(tied - s))])


def _detect_sample_slips(signal: np.ndarray, sampling_rate: float,
                         f0: float) -> List[Tuple[int, float]]:
    """See _detect_sample_slips_ex; this drops the per-slip `unsure` flag."""
    return [(s, st) for s, st, _ in _detect_sample_slips_ex(signal, sampling_rate, f0)]


def _detect_sample_slips_ex(signal: np.ndarray, sampling_rate: float,
                            f0: float) -> List[Tuple[int, float, bool]]:
    """[(index, step), ...]: from `index` on, each sample was really taken
    `step` samples later than its position says (+1.0 for a dropped sample;
    the device also makes fractional steps - half a sample back, then a
    whole one forward, 150 samples apart, on the captures this was built on).

    Returns [] - after one pass of per-cycle sums - when there is no mains
    tone to read the timing from, which is every trace without interference.
    """
    x = np.asarray(signal, dtype=float)
    n = x.size
    if n == 0 or f0 <= 0 or not np.all(np.isfinite(x)):
        return []
    period = sampling_rate / f0
    P = int(round(period))
    C = n // P if P > 0 else 0
    need = _SLIP_SMOOTH_CYCLES + 2 * _SLIP_PERSIST_CYCLES + 2
    if P < 4 or C < need:
        return []

    # Everything slower than the mains is removed by subtracting a one-period
    # moving average (itself blind to the mains); the tone is then brought to
    # DC and summed per cycle: one phasor per mains cycle.
    y = x - np.convolve(x, np.ones(P) / P, mode="same")
    w = 2.0 * np.pi * f0 / sampling_rate          # one sample, in radians
    z = (y[:C * P] * np.exp(-1j * w * np.arange(C * P))).reshape(C, P).sum(axis=1)
    amp = float(np.median(2.0 * np.abs(z) / P))
    if amp < _MAINS_FIT_MIN_AMPLITUDE:
        return []

    # Cycle-to-cycle phase, wrapped (so a QRS cycle's garbage phase is undone
    # by the next one), steady drift removed (supply a little off nominal),
    # accumulated, then median-smoothed.
    d = np.angle(z[1:] * np.conj(z[:-1]))
    d = np.angle(np.exp(1j * (d - float(np.median(d)))))
    psi = np.concatenate(([0.0], np.cumsum(d)))
    from scipy.ndimage import median_filter as _median_filter
    ps = _median_filter(psi, size=_SLIP_SMOOTH_CYCLES, mode="nearest")

    thr = _SLIP_MIN_STEP * w
    top = _SLIP_MAX_STEP * w
    Q = _SLIP_PERSIST_CYCLES

    # A step at cycle c: the median of the Q smoothed cycles from c on differs
    # from the median of the Q before it by a timing-sized amount. Scored for
    # every c at once; each run of neighbouring hits is one step, taken where
    # the difference peaks.
    win = sliding_window_view(ps, Q)
    cs = np.arange(Q, C - Q + 1)
    mb = np.median(win[cs - Q], axis=1)
    ma = np.median(win[cs], axis=1)
    D = ma - mb
    hit = (np.abs(D) >= thr) & (np.abs(D) <= top)
    if not hit.any():
        return []

    slips: List[Tuple[int, float, bool]] = []
    idx = np.flatnonzero(hit)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    for run in runs:
        j = int(run[int(np.argmax(np.abs(D[run])))])
        c = int(cs[j])
        level, delta = float(mb[j]), float(D[j])
        # Where inside the cycles around c: the transition cycle sits between
        # the two levels in proportion to how much of it came after the step,
        # so the unshifted fractions sum to the distance from the start of the
        # range to the step.
        j0 = max(0, c - 3)
        j1 = min(C, c + 2)
        frac = np.clip((psi[j0:j1] - level) / delta, 0.0, 1.0)
        s = j0 * P + int(round(P * float(np.sum(1.0 - frac))))
        a = max(0, s - 15 * P)
        b = min(n, s + 15 * P)
        if 0 < s < n:
            found = _verify_slip(y, s, delta / w, a, b, w, amp)
            if found is not None:
                slips.append(found)
    return slips


def _shared_sample_slips(signals: np.ndarray, sampling_rate: float,
                         f0: float) -> List[List[Tuple[int, float]]]:
    """One slip list for a (L, n) block, read from its strongest-mains lead.

    Leads with NaN in them (a sweep-mode gap) keep an empty list, exactly as
    _detect_sample_slips gives them on their own.
    """
    L, n = signals.shape
    if L == 0 or n == 0:
        return [[] for _ in range(L)]
    finite = np.all(np.isfinite(signals), axis=1)
    if not finite.any():
        return [[] for _ in range(L)]
    ph = np.exp(-1j * 2.0 * np.pi * f0 / sampling_rate * np.arange(n))
    centred = signals[finite] - np.mean(signals[finite], axis=1, keepdims=True)
    strength = np.abs(centred @ ph)
    best = int(np.flatnonzero(finite)[int(np.argmax(strength))])
    found = _detect_sample_slips_ex(signals[best], sampling_rate, f0)
    shared = [(s, st) for s, st, _ in found]
    unsure = [(s, st) for s, st, u in found if u]
    if unsure:
        # Only where a QRS got in the way on the strongest lead - about one
        # slip in five. The rest are already pinned on clean samples, and
        # re-pinning them on every lead every frame doubled the filter's cost.
        repinned = dict(zip(unsure, _pin_across_leads(signals[finite], sampling_rate,
                                                        f0, unsure)))
        shared = [repinned.get(sl, sl) for sl in shared]
    return [list(shared) if finite[i] else [] for i in range(L)]


def _pin_across_leads(signals: np.ndarray, sampling_rate: float, f0: float,
                      slips: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Re-pin each slip on every lead carrying mains at once.

    The drop is one instant on all twelve leads, but each lead's QRS has its
    own shape and size, so the QRS bias that can mislead one lead's pin does
    not repeat across them; summing the (amplitude-normalised) candidate
    errors lets the leads outvote it.
    """
    n = signals.shape[1]
    P = max(2, int(round(sampling_rate / f0)))
    w = 2.0 * np.pi * f0 / sampling_rate
    kernel = np.ones(P) / P
    ys, amps = [], []
    for x in signals:
        y = x - np.convolve(x, kernel, mode="same")
        C = n // P
        if C < 4:
            continue
        z = (y[:C * P] * np.exp(-1j * w * np.arange(C * P))).reshape(C, P).sum(axis=1)
        amp = float(np.median(2.0 * np.abs(z) / P))
        if amp >= _MAINS_FIT_MIN_AMPLITUDE:
            ys.append(y)
            amps.append(amp)
    if len(ys) < 2:
        return slips
    out = []
    for s, step in slips:
        a = max(0, s - 15 * P)
        b = min(n, s + 15 * P)
        total, cands = None, None
        for y, amp in zip(ys, amps):
            c, e, _ = _pin_errors(y, amp, s, step, w, a, b)
            if c.size == 0:
                continue
            if total is None:
                cands, total = c, e.copy()
            elif c.size == cands.size:
                total += e
        out.append((s if total is None else _pick_pin(cands, total, s), step))
    return out


def _slip_time_base(n: int, sampling_rate: float,
                    slips: List[Tuple[int, float]]) -> np.ndarray:
    """Sample times in seconds with every slip put back: (n + shift) / fs."""
    idx = np.arange(n, dtype=float)
    for s, st in slips:
        idx[s:] += st
    return idx / sampling_rate


def _has_flat_run(signal: np.ndarray, min_len: int) -> bool:
    """True if some stretch of at least min_len samples holds one exact value.

    Real acquisition with mains on it never does that; the unfilled part of a
    ring buffer (zeros, or a constant after offset removal) always does.
    """
    x = np.asarray(signal)
    if x.size < min_len or min_len < 2:
        return False
    same = np.diff(x) == 0
    if not same.any():
        return False
    edges = np.diff(np.concatenate(([0], same.astype(np.int8), [0])))
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return bool(runs.max() + 1 >= min_len)


def _unfilled_window(signal: np.ndarray) -> bool:
    """A window still partly padding: no frequency may be measured from it.

    The two-estimates-must-agree check below does not protect against this on
    its own, because the two estimates are usually two leads of the SAME frame
    - they share the same padding, so they agree with each other while both
    being wrong, and the wrong value is then held for the session.
    """
    n = np.asarray(signal).size
    return _has_flat_run(signal, max(25, n // 20))


def _mains_fit_grid(n: int, sampling_rate: float, f0: float, window_s: float):
    """Time base, fit-window length and window starts for a signal of n samples.

    Shared by _mains_tone (one lead) and _mains_tone_multi (all leads at once)
    so the two cannot drift apart: the batched form exists only because the
    grid depends on n, fs and f0 and not on the signal, which is exactly what
    lets twelve leads share one design matrix.
    """
    t = np.arange(n, dtype=float) / sampling_rate
    win = int(round(window_s * sampling_rate))
    win = max(win, int(round(4.0 * sampling_rate / f0)))   # >= 4 mains cycles
    win = min(win, n)
    if win < 4:
        return t, win, []

    hop = max(1, win // 2)

    # ENOUGH FIT WINDOWS FOR THE FREQUENCY REFINEMENT TO RUN AT ALL.
    #
    # _refine_mains_freq() needs three window centres to take a slope through;
    # with fewer it returns the nominal frequency untouched. At the default
    # 2 s window and a half-window hop that needs 4 s of signal - but the live
    # 12-lead view filters a 3 s window, which yields exactly two centres. So
    # on screen the refinement never ran and the fit was always at exactly
    # 50.000 Hz, however far the supply actually was.
    #
    # Measured on captures/ecg_capture_20260916_181945.npz, lead II, where the
    # supply is at 49.65 Hz and 309 counts:
    #
    #     3 s window,  2 centres, refinement skipped -> 196.7 counts left
    #     6 s window,  5 centres, refinement ran     ->   0.2 counts left
    #
    # A thousandfold difference, decided by nothing but strip length. That
    # residual is what the operator sees as fuzz on every lead of the live
    # grid with "AC 50 Hz" switched on, while the printed report - built from
    # a 10 s snapshot - comes out clean from the same recording.
    #
    # Tightening the hop on short signals costs one or two extra least-squares
    # solves over a few hundred samples and changes nothing for long ones.
    if n > win:
        hop = min(hop, max(1, (n - win) // 2))

    starts = list(range(0, max(1, n - win + 1), hop))
    if starts[-1] != n - win:
        # NEVER TWO WINDOWS A FEW SAMPLES APART AT THE END.
        #
        # The track is continued past the last centre along the slope between
        # the last two. A 10,005-sample strip got starts 9000 and 9005 - centres
        # five samples apart - so any difference between those two fits was
        # divided by 5 and extrapolated 500 samples: a hundredfold lever. On a
        # steady tone the fits are equal and nothing shows; with a sample slip
        # inside both windows it drew the tail's mains back at twice its size.
        # When the final window would start within half a hop of the previous
        # one, it replaces it instead of joining it.
        if len(starts) > 1 and (n - win) - starts[-1] < hop / 2.0:
            starts[-1] = n - win
        else:
            starts.append(n - win)
    return t, win, starts


def _mains_tone(signal: np.ndarray, sampling_rate: float, f0: float,
                window_s: float = _MAINS_FIT_WINDOW_S,
                exact: bool = False,
                slips: Optional[List[Tuple[int, float]]] = None) -> np.ndarray:
    """The power-line tone present in this signal, to be subtracted from it.

    Least squares against cos and sin at the mains frequency, re-fitted over
    overlapping windows so a slowly changing interference amplitude is
    tracked. A constant and a ramp are fitted alongside them and thrown away,
    so that baseline offset and drift inside the window cannot lean on the
    two coefficients that matter.

    The point of doing it this way rather than with a notch filter is that
    what comes back is a pure sinusoid. Subtracting it removes the mains and
    changes nothing else, because a sinusoid has no impulse response: there
    is no transient for a QRS edge to excite. See apply_ac_filter().
    """
    n = signal.size
    t, win, starts = _mains_fit_grid(n, sampling_rate, f0, window_s)
    if win < 4:
        return np.zeros(n)
    # Fit against real time, with any dropped samples put back. See
    # "SAMPLE SLIPS" above; with no slips this is the plain n / fs.
    if slips:
        t = _slip_time_base(n, sampling_rate, slips)

    # THE SUPPLY IS NOT AT EXACTLY 50.000 Hz, AND FITTING AS THOUGH IT WERE IS
    # WHAT LEFT THIS FILTER'S OUTPUT DIRTY AT THE TWO ENDS OF EVERY STRIP.
    #
    # Measured on a Fluke recording from this device (captures/ecg_capture_
    # 20260914_163043.npz, mains at 695 counts on lead II - 29% of R wave
    # height): the supply was running at 50.024 Hz. Across one 2-second fit
    # window that difference is invisible, but the fitted phasor rotates 8.6
    # degrees per second against it, and the coefficient track was then handed
    # to np.interp, WHICH CLAMPS rather than extrapolates - so the first and
    # last half second of every strip was rebuilt from a phasor frozen up to a
    # full window stale. Measured residual there: 91 counts against 4.4 counts
    # in the middle of the same strip, twenty times worse, and the same story
    # on all twelve leads (17-91 counts at the edges).
    #
    # That residual is the wobble still on the trace after "AC 50 Hz" has been
    # switched on, and it sits on the ST segment at the strip edges, where it
    # is read as a J point.
    #
    # Two corrections, both paid for out of work already being done: take the
    # supply's real frequency from the drift of the first fit, and continue the
    # coefficient track at the ends instead of holding it flat. Re-measured on
    # the same recording, all twelve leads, the edge residual falls from
    # 17-91 counts to 1.3-8.3.
    centres, cos_c, sin_c = _fit_phasor_track(signal, t, f0, starts, win)
    # `exact` means the caller already knows the frequency and there is nothing
    # to estimate - the harmonic takes this path, at twice the measured
    # fundamental. Otherwise reuse a recent measurement, or make one now and
    # keep it. See _MAINS_FREQ_CACHE.
    if exact:
        f_true = f0
    else:
        f_true = _cached_mains_freq(f0)
        if f_true is None:
            # The amplitude does not depend on the refinement, and below the
            # threshold the refinement's answer is thrown away (f_true = f0,
            # further down) - so it is only run when it can be used. It is a
            # pure function; skipping it changes nothing but the time.
            _amp_pre = float(np.median(np.hypot(np.asarray(cos_c, dtype=float),
                                                np.asarray(sin_c, dtype=float))))
            if _amp_pre < _MAINS_FIT_MIN_AMPLITUDE or _unfilled_window(signal):
                f_true = f0
            else:
                centre_times = (np.interp(centres, np.arange(n, dtype=float), t)
                                if slips else None)
                f_true = _refine_mains_freq(f0, centres, cos_c, sin_c, sampling_rate,
                                            centre_times)

            # ONLY LOCK IT IN IF THERE WAS REALLY A TONE TO MEASURE.
            #
            # The estimate comes from the rotation of the fitted phasor, which
            # is meaningless when the fit had nothing to lock onto - and the
            # first frames after Start are exactly that case: the ring buffer is
            # still mostly zeros, so the window is part silence and part signal.
            # Measured on captures/ecg_capture_20260916_181945.npz, whose supply
            # is at 49.671 Hz:
            #
            #     zeros + 0.6 s of data  -> 49.562 Hz   (0.11 Hz out)
            #     half-filled buffer     -> 49.821 Hz   (0.15 Hz out)
            #     full window of signal  -> 49.671 Hz   (right)
            #
            # and what that costs on the same strip, 48-52 Hz residual:
            #
            #     locked at 49.671 -> 16.9 counts
            #     locked at 49.562 -> 33.4 counts
            #     locked at 49.821 -> 59.4 counts
            #
            # Because the value is then reused for every lead and held for the
            # rest of the session, one bad first frame quietly degraded the
            # whole recording. Requiring a real tone means startup frames fall
            # back to the nominal frequency and simply try again next frame,
            # once the buffer has filled.
            # Two independent measurements have to agree before the value is
            # trusted. The amplitude test alone is not enough: a half-filled
            # buffer still carries plenty of mains in the part that has arrived,
            # and measured 49.821 Hz against a true 49.671 - wrong by 0.15 Hz,
            # and locked in for the session. A window that is part silence gives
            # a different answer every frame as it fills, so it never agrees
            # with itself; a full one repeats the same figure. Consecutive calls
            # are usually different leads of the same frame, which makes this a
            # cross-check between leads as well - they see one supply, so they
            # must agree on its frequency.
            amp = float(np.median(np.hypot(np.asarray(cos_c, dtype=float),
                                           np.asarray(sin_c, dtype=float))))
            if amp < _MAINS_FIT_MIN_AMPLITUDE or _unfilled_window(signal):
                f_true = f0
            else:
                pending = _MAINS_FREQ_PENDING.get(f0)
                _MAINS_FREQ_PENDING[f0] = f_true
                if pending is not None and abs(pending - f_true) <= _MAINS_FREQ_AGREE_HZ:
                    _store_mains_freq(f0, f_true)
                else:
                    f_true = f0        # not confirmed yet - try again next call

    # Refit at the measured frequency. Skipped when the supply really is on its
    # nominal value, so a clean grid costs nothing extra.
    if f_true != f0:
        centres, cos_c, sin_c = _fit_phasor_track(signal, t, f_true, starts, win)

    idx = np.arange(n, dtype=float)
    if len(centres) == 1:
        a_t = np.full(n, cos_c[0])
        b_t = np.full(n, sin_c[0])
    else:
        a_t = _track_to_samples(idx, centres, cos_c)
        b_t = _track_to_samples(idx, centres, sin_c)
    return a_t * np.cos(2.0 * np.pi * f_true * t) + b_t * np.sin(2.0 * np.pi * f_true * t)


def apply_ac_filter(signal: np.ndarray, sampling_rate: float, ac_filter: str,
                    slips: Optional[List[Tuple[int, float]]] = None) -> np.ndarray:
    """
    Remove power line interference by SUBTRACTING THE MAINS TONE, not by
    notching the band out.

    WHY NOT A NOTCH. What stood here was scipy's iirnotch at Q=5 run through
    sosfiltfilt. Every filter has an impulse response and a zero-phase one
    rings symmetrically on both sides of an edge; a notch's impulse response
    is a decaying 50 Hz oscillation, so each QRS complex was followed - and
    preceded - by a 20 ms-period wobble lasting into the ST segment. Measured
    on real recordings from this device, on the median beat, it put 12-13% of
    R-wave height of 50 Hz ripple onto the ST segment with a 150 Hz muscle
    filter, and 3-3.5% with a 25 Hz one (which only looks better because a
    25 Hz low-pass is itself removing most of a 50 Hz ripple afterwards).

    That ripple is drawn as a J wave. It is most obvious exactly where it is
    least welcome: at 180/min the ST segment is short and already sloping up
    into the T wave, so a bump lands on it as a discrete notch, while at
    60/min the same ripple spreads along a long flat ST segment and reads as
    mild waviness. Hence "adult is fine, paediatric at 180 shows a J point" -
    the filter behaved the same in both, the waveform it was drawn on did not.

    Worse, the notch ran whether or not there was anything to remove. On the
    three device recordings this was measured against there was no mains at
    all - the 50 Hz line sat at or below the level of the spectrum either
    side of it - so the notch was removing nothing and inventing a J wave.

    WHAT THIS DOES INSTEAD. Fit the amplitude and phase of a 50 (or 60) Hz
    sinusoid to the signal over overlapping one-second windows and subtract
    it. Real interference is a coherent tone and is fitted and removed; a QRS
    complex is broadband and transient, contributes almost nothing to a fit
    made over 50 mains cycles, and is left alone. There is no impulse
    response, so there is nothing to ring.
    """
    if ac_filter == "off" or not ac_filter:
        return signal

    try:
        ac_text = str(ac_filter).strip().lower()
        match = re.search(r"(\d+(?:\.\d+)?)", ac_text)
        notch_freq = float(match.group(1)) if match else float(ac_filter)  # 50 or 60 Hz

        nyquist = sampling_rate / 2.0
        if notch_freq <= 0 or notch_freq >= nyquist:
            print(f" AC filter frequency {notch_freq}Hz is invalid for sampling rate {sampling_rate}Hz")
            return signal

        work = np.asarray(signal, dtype=float)
        if work.size < 4:
            return signal

        # Read at the NOMINAL frequency, never the cached one, so the answer
        # does not depend on what earlier calls locked - which is what keeps
        # this and apply_ac_filter_multi identical lead for lead.
        # `slips` may be passed in when the caller already knows the stream's
        # timing (tests hold this path to the batched one that way).
        if slips is None:
            slips = _detect_sample_slips(work, sampling_rate, notch_freq)
        work = work - _mains_tone(work, sampling_rate, notch_freq, slips=slips)

        # Only a CONFIRMED supply earns a harmonic subtraction. See below.
        confirmed = _cached_mains_freq(notch_freq)
        measured = confirmed or notch_freq

        # THE SECOND HARMONIC IS TAKEN OFF TOO.
        #
        # A mains-frequency fit removes 50 Hz and leaves 100 Hz untouched, and
        # switch-mode supplies and half-wave rectifiers put a real second
        # harmonic on the signal. Measured: a 40-count 100 Hz tone comes out of
        # apply_ac_filter at 40.0 counts - completely unaffected.
        #
        # It went unnoticed because the muscle filter was hiding it at the
        # shipped default. Residual 100 Hz after apply_emg_filter:
        #
        #     EMG 25 Hz  ->   0.16 counts     (default - harmonic invisible)
        #     EMG 40 Hz  ->   4.59 counts
        #     EMG 150 Hz ->  36.31 counts     (harmonic essentially intact)
        #     EMG off    ->  40.00 counts
        #
        # So on any recording taken with a wide muscle filter - which is what a
        # diagnostic-quality trace wants - the harmonic was drawn and printed.
        # The CardioX Android app already handles this, with a second IIR notch
        # at 100 Hz (EnhancedECGFilter.notch2); this is the same idea expressed
        # the way this module removes mains, by subtracting a fitted tone
        # rather than notching, so it inherits the reason apply_ac_filter does
        # not use a notch in the first place - see this function's docstring.
        # Twice the MEASURED fundamental, not twice the nominal, and passed as
        # exact so it is not estimated a second time. A supply at 49.67 Hz puts
        # its harmonic at 99.34, not 100.00, and refining that independently
        # would both cost another full fit per lead and fail anyway: the 0.66 Hz
        # offset exceeds _MAINS_MAX_DRIFT_HZ, so the estimator would reject it
        # and hand back the nominal it started from.
        #
        # AND ONLY WHEN THE SUPPLY HAS BEEN CONFIRMED TO BE THERE.
        #
        # A QRS complex is a sharp edge, and a sharp edge carries real energy
        # around 100 Hz. Subtracting a fitted 100 Hz tone from a signal that has
        # no harmonic interference does not remove noise - it removes part of
        # the QRS, blunting its edges so it measures wider. Caught by
        # tests/test_qrs_width_fast_rate.py: with the harmonic always
        # subtracted the median QRS read 68 ms against 56 ms, twelve
        # milliseconds wider on a strip with no mains to speak of, and against
        # a 120 ms wide-QRS diagnostic threshold that is not a rounding error.
        #
        # The gate is per signal: fit the harmonic, and subtract it only if what
        # was fitted is big enough to be interference. It must NOT be gated on
        # the shared frequency cache alone - that records whether a supply was
        # seen on ANY signal in this process, so once the live grid had locked
        # one, every later call subtracted a harmonic whether its own signal
        # carried one or not. That is exactly how the QRS test above passed on
        # its own and failed in the full suite: an earlier test had locked a
        # frequency, and the cache leaked it in.
        #
        # The threshold separates the two cases by three orders of magnitude.
        # Fitted harmonic amplitude, measured:
        #
        #     real capture (ecg_capture_20260916_181945, lead II)  177.8
        #     clean fast-rate QRS, no interference                    0.2
        #     clean normal-rate QRS, no interference                  0.0
        #
        # so a QRS edge's own 100 Hz content is left alone and a real harmonic
        # is still removed.
        harmonic = measured * 2.0
        if harmonic < nyquist:
            # Same slips: a dropped sample shifts the harmonic in time too.
            harmonic_tone = _mains_tone(work, sampling_rate, harmonic, exact=True,
                                        slips=slips)
            if np.sqrt(2.0) * float(np.std(harmonic_tone)) >= _MAINS_FIT_MIN_AMPLITUDE:
                work = work - harmonic_tone

        return work

    except Exception as e:
        print(f" Error applying AC filter ({ac_filter}Hz): {e}")
        return signal


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-LEAD FORMS OF THE DISPLAY FILTERS
#
# The live 12-lead grid runs the display chain once per lead per frame. Each of
# those calls is many small numpy operations, and every numpy operation drops
# the interpreter lock for its C work and then has to take it back. On a live
# acquisition that added up to roughly a thousand lock hand-offs per frame on
# the thread that draws the trace - and whenever any other thread was running
# Python bytecode at that moment (the rhythm analysis, a cloud sync), each of
# those hand-offs waited for it. py-spy put the chain's real work at about 8 ms
# a frame; the PERF log showed the same stage at 300-500 ms, and the waveform
# showed it as a periodic freeze.
#
# Batching the twelve leads into one (12, n) array turns a thousand hand-offs
# into a few dozen. The arithmetic is the same: every stage below is row-wise
# independent, so lead i of the batched result is what the single-lead call
# returns for lead i (tests/test_batched_display_chain.py holds each of these
# to the single-lead form on real device captures). Where a stage carries
# module state - the measured mains frequency - the batched form replays the
# single-lead sequence lead by lead, so the cache ends in the same state.
# ─────────────────────────────────────────────────────────────────────────────

def _fit_phasor_track_multi(signals: np.ndarray, t: np.ndarray, f0: float,
                            starts: List[int], win: int):
    """_fit_phasor_track for a (L, n) block: one design matrix, L right-hand sides.

    The normal-equation matrix depends on the window and the frequency only,
    so the LU factorisation is shared and each lead is one back-substitution.
    Returns centres (list), cos_c and sin_c as (len(starts), L) arrays.
    """
    L = signals.shape[0]
    cos_all = np.cos(2.0 * np.pi * f0 * t)
    sin_all = np.sin(2.0 * np.pi * f0 * t)
    ones_win = np.ones(win)
    centres = []
    cos_c = np.empty((len(starts), L))
    sin_c = np.empty((len(starts), L))
    for k, s0 in enumerate(starts):
        seg = signals[:, s0:s0 + win].T                    # (win, L)
        tt = t[s0:s0 + win]
        ramp = tt - float(np.mean(tt))
        design = np.column_stack([cos_all[s0:s0 + win],
                                  sin_all[s0:s0 + win],
                                  ones_win, ramp])
        gram = design.T @ design
        rhs = design.T @ seg                              # (4, L)
        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(design, seg, rcond=None)
        centres.append(s0 + win / 2.0)
        cos_c[k] = coef[0]
        sin_c[k] = coef[1]
    return centres, cos_c, sin_c


def _track_to_samples_multi(idx: np.ndarray, centres: List[float],
                            coeffs: np.ndarray) -> np.ndarray:
    """_track_to_samples for a (S, L) coefficient block; returns (L, n).

    Same piecewise line as np.interp between the centres, and the same
    continued trend beyond the first and last centre, expressed as one
    gather so all leads are interpolated in a single pass.
    """
    xp = np.asarray(centres, dtype=float)
    j = np.searchsorted(xp, idx, side='right') - 1
    j = np.clip(j, 0, len(xp) - 2)
    dx = (xp[j + 1] - xp[j])[:, None]
    slope = (coeffs[j + 1, :] - coeffs[j, :]) / dx
    out = coeffs[j, :] + (idx - xp[j])[:, None] * slope     # (n, L)
    return out.T


def _mains_tone_multi(signals: np.ndarray, sampling_rate: float, f0: float,
                      window_s: float = _MAINS_FIT_WINDOW_S,
                      exact: bool = False,
                      slips: Optional[List[List[Tuple[int, float]]]] = None,
                      min_amplitude: Optional[float] = None) -> np.ndarray:
    """_mains_tone for a (L, n) block. Returns the (L, n) tones to subtract.

    The supply-frequency estimate and its cache are module state that the
    single-lead form updates once per call, in lead order - lead 0 leaves a
    pending estimate, lead 1 confirms it, leads 2..11 read the cache. That
    sequence is replayed here lead by lead on the batched fit, so the cache
    is left exactly as twelve single-lead calls would leave it. Leads that
    resolve to different frequencies (only ever during lock-in) are refitted
    in groups.

    `slips` is one slip list per lead (see _detect_sample_slips). Leads that
    share a list share a time base and one batched fit; normally that is every
    lead with the mains on it, since a dropped sample is dropped from all of
    them at once.

    `min_amplitude`: a lead whose fitted tone provably cannot reach this
    amplitude gets a zero row instead of a synthesised one. The bound is exact:
    the coefficient track is piecewise linear between the window centres and
    the two ends, so its magnitude is largest at one of those points, and no
    sample of the tone exceeds it. The caller's gate (sqrt(2) x std >= the
    same threshold) is therefore false for that lead either way - on a clean
    signal the harmonic was synthesised for all twelve leads every frame and
    then thrown away.
    """
    L, n = signals.shape
    t0, win, starts = _mains_fit_grid(n, sampling_rate, f0, window_s)
    if win < 4:
        return np.zeros_like(signals)

    idx = np.arange(n, dtype=float)
    keys = [tuple(slips[i]) if slips and slips[i] else () for i in range(L)]
    groups: dict = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    tbase = {k: (_slip_time_base(n, sampling_rate, list(k)) if k else t0)
             for k in groups}
    fits = {k: _fit_phasor_track_multi(signals[cols], tbase[k], f0, starts, win)
            for k, cols in groups.items()}
    pos = {i: groups[k].index(i) for i, k in enumerate(keys)}

    f_true = np.full(L, float(f0))
    if not exact:
        for i in range(L):
            ft = _cached_mains_freq(f0)
            if ft is None:
                k = keys[i]
                centres, cos_c, sin_c = fits[k]
                ci, si = cos_c[:, pos[i]], sin_c[:, pos[i]]
                amp = float(np.median(np.hypot(ci, si)))
                # Refinement only when its answer can be used - see _mains_tone.
                # On a clean signal (no mains worth measuring) this skipped
                # twelve refinements every display frame.
                if amp < _MAINS_FIT_MIN_AMPLITUDE or _unfilled_window(signals[i]):
                    ft = f0
                else:
                    centre_times = np.interp(centres, idx, tbase[k]) if k else None
                    ft = _refine_mains_freq(f0, centres, ci, si, sampling_rate, centre_times)
                    pending = _MAINS_FREQ_PENDING.get(f0)
                    _MAINS_FREQ_PENDING[f0] = ft
                    if pending is not None and abs(pending - ft) <= _MAINS_FREQ_AGREE_HZ:
                        _store_mains_freq(f0, ft)
                    else:
                        ft = f0
            f_true[i] = ft

    tones = np.empty_like(signals)
    for k, members in groups.items():
        members = np.asarray(members)
        t = tbase[k]
        centres, cos_c, sin_c = fits[k]
        for ft in np.unique(f_true[members]):
            sel = np.flatnonzero(f_true[members] == ft)     # positions in group
            cols = members[sel]
            if ft != f0:
                c2, cc, sc = _fit_phasor_track_multi(signals[cols], t, ft, starts, win)
            else:
                c2, cc, sc = centres, cos_c[:, sel], sin_c[:, sel]
            if min_amplitude is not None:
                if len(c2) == 1:
                    peak = np.hypot(cc[0], sc[0])
                else:
                    ends = np.array([0.0, float(n - 1)])
                    ea = _track_to_samples_multi(ends, c2, cc)          # (m, 2)
                    eb = _track_to_samples_multi(ends, c2, sc)
                    peak = np.maximum(np.hypot(cc, sc).max(axis=0),
                                      np.hypot(ea, eb).max(axis=1))
                keep = np.sqrt(2.0) * peak >= float(min_amplitude)
                if not np.all(keep):
                    tones[cols[~keep]] = 0.0
                    if not np.any(keep):
                        continue
                    cols, cc, sc = cols[keep], cc[:, keep], sc[:, keep]
            if len(c2) == 1:
                a_t = np.repeat(cc[0][:, None], n, axis=1)
                b_t = np.repeat(sc[0][:, None], n, axis=1)
            else:
                a_t = _track_to_samples_multi(idx, c2, cc)
                b_t = _track_to_samples_multi(idx, c2, sc)
            tones[cols] = (a_t * np.cos(2.0 * np.pi * ft * t)
                           + b_t * np.sin(2.0 * np.pi * ft * t))
    return tones


def apply_ac_filter_multi(signals: np.ndarray, sampling_rate: float, ac_filter: str) -> np.ndarray:
    """apply_ac_filter for a (L, n) block of leads. See that function.

    One difference, confined to the first frame or two of a session: the
    second-harmonic frequency is taken from the cache after ALL leads' fits
    have run, where the single-lead form reads it after each lead's own. Once
    the supply is confirmed - within two calls - the two agree exactly.
    """
    if ac_filter == "off" or not ac_filter:
        return signals
    try:
        ac_text = str(ac_filter).strip().lower()
        match = re.search(r"(\d+(?:\.\d+)?)", ac_text)
        notch_freq = float(match.group(1)) if match else float(ac_filter)
        nyquist = sampling_rate / 2.0
        if notch_freq <= 0 or notch_freq >= nyquist:
            print(f" AC filter frequency {notch_freq}Hz is invalid for sampling rate {sampling_rate}Hz")
            return signals
        work = np.asarray(signals, dtype=float)
        if work.ndim != 2 or work.shape[1] < 4:
            return signals
        # A dropped sample is dropped from every lead at once, so the timing is
        # read once, from the lead carrying the most mains, and shared - one
        # time base, one batched fit. apply_ac_filter reads it from its own
        # lead, which on a lead with the mains on it finds the same steps to
        # within a few hundredths of a sample.
        slips = _shared_sample_slips(work, sampling_rate, notch_freq)
        work = work - _mains_tone_multi(work, sampling_rate, notch_freq, slips=slips)

        confirmed = _cached_mains_freq(notch_freq)
        measured = confirmed or notch_freq
        harmonic = measured * 2.0
        if harmonic < nyquist:
            harmonic_tone = _mains_tone_multi(work, sampling_rate, harmonic, exact=True,
                                              slips=slips,
                                              min_amplitude=_MAINS_FIT_MIN_AMPLITUDE)
            # Per lead, exactly as the single-lead gate: subtract only where
            # what was fitted is big enough to be interference.
            gate = np.sqrt(2.0) * harmonic_tone.std(axis=1) >= _MAINS_FIT_MIN_AMPLITUDE
            if np.any(gate):
                work[gate] = work[gate] - harmonic_tone[gate]
        return work
    except Exception as e:
        print(f" Error applying AC filter ({ac_filter}Hz) to lead block: {e}")
        return signals


def _gaussian_lowpass_multi(signals: np.ndarray, sampling_rate: float, cutoff_freq: float) -> np.ndarray:
    """_gaussian_lowpass for a (L, n) block: one reflect-pad, one matmul."""
    n = signals.shape[1]
    if n == 0:
        return signals
    sigma = _GAUSSIAN_SIGMA_AT_MINUS_3DB * sampling_rate / cutoff_freq
    if sigma <= 0.05:
        return signals
    half = max(1, int(np.ceil(_GAUSSIAN_HALF_WIDTH_SIGMAS * sigma)))
    half = min(half, (n - 1) // 2)
    if half < 1:
        return signals
    offsets = np.arange(2 * half + 1) - half
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(signals, ((0, 0), (half, half)), mode="reflect")
    windows = sliding_window_view(padded, 2 * half + 1, axis=1)      # (L, n, K) view
    return windows @ kernel[::-1]                                     # np.convolve flips the kernel


def apply_emg_filter_multi(signals: np.ndarray, sampling_rate: float, emg_filter: str) -> np.ndarray:
    """apply_emg_filter for a (L, n) block of leads. See that function."""
    if not emg_filter or emg_filter == "off":
        return signals
    try:
        cutoff_freq = float(emg_filter)
        nyquist = sampling_rate / 2.0
        if sampling_rate < 100:
            nyquist = 500.0 / 2.0
            sampling_rate = 500.0
        max_allowed = nyquist * 0.9
        if cutoff_freq >= max_allowed:
            cutoff_freq = max_allowed
        if cutoff_freq <= 0:
            return signals
        return _gaussian_lowpass_multi(np.asarray(signals, dtype=float), sampling_rate, cutoff_freq)
    except Exception as e:
        print(f" Error applying EMG filter ({emg_filter}Hz) to lead block: {e}")
        return signals


def apply_baseline_wander_median_mean_multi(signals: np.ndarray, sampling_rate: float = 500) -> np.ndarray:
    """apply_baseline_wander_median_mean for a (L, n) block. See that function.

    The median stage runs medfilt once per row; the mean stage and the
    subtraction are one call over the block.
    """
    n = signals.shape[1]
    if n < 50:
        return signals - np.mean(signals, axis=1, keepdims=True)
    try:
        median_window_ms = 120.0
        median_window = int(median_window_ms * sampling_rate / 1000.0)
        median_window = max(3, min(median_window, n // 2))
        if median_window % 2 == 0:
            median_window += 1
        # One medfilt per row, not ndimage.median_filter over the block: the
        # N-d rank filter is generic and measured 9.7 ms on a 12 x 1500 block
        # against 1.0 ms for twelve 1-D medfilt calls, for the same output.
        b1 = np.array([medfilt(row, kernel_size=median_window) for row in signals])
        mean_window_ms = 800.0
        mean_window = int(mean_window_ms * sampling_rate / 1000.0)
        mean_window = max(10, min(mean_window, n // 2))
        baseline = uniform_filter1d(b1.astype(float), size=mean_window, mode='nearest', axis=-1)
        return signals - baseline
    except Exception as e:
        print(f" Error applying median+mean baseline filter to lead block: {e}")
        return signals - np.mean(signals, axis=1, keepdims=True)


def apply_dft_filter_multi(signals: np.ndarray, sampling_rate: float, dft_filter: str) -> np.ndarray:
    """apply_dft_filter for a (L, n) block of leads. See that function."""
    if dft_filter == "off" or not dft_filter:
        return signals
    try:
        # No 0.5 Hz special case, and the double filtfilt - both for the reasons
        # given in apply_dft_filter. This block path and the per-lead path must
        # produce the same trace; a lead drawn through one and a lead drawn
        # through the other appear side by side on the same sheet.
        cutoff_freq = float(dft_filter)
        nyquist = sampling_rate / 2.0
        normalized_cutoff = cutoff_freq / nyquist
        if normalized_cutoff <= 0 or normalized_cutoff >= 0.5:
            print(f" DFT filter cutoff {cutoff_freq}Hz is invalid for sampling rate {sampling_rate}Hz")
            return signals
        b, a = butter(2, normalized_cutoff, btype='high')
        return filtfilt(b, a, filtfilt(b, a, signals, axis=-1), axis=-1)
    except Exception as e:
        print(f" Error applying DFT filter ({dft_filter}Hz) to lead block: {e}")
        return signals


def lead_noise_ratio(signal: np.ndarray, sampling_rate: float = 500.0) -> float:
    """High-frequency content of a lead as a fraction of its own signal span.

    Used to decide whether a lead is clean enough for its ST segment to be worth
    reporting. Philips DXL carries quality-monitor codes for exactly this reason
    and will decline to interpret data it considers unsuitable; a screening
    device that states an ST finding off a noisy trace is worse than one that
    says it could not measure.

    The high-pass sits at 100 Hz, NOT the 60 Hz used elsewhere. A second-order
    high-pass at 60 Hz still passes about 31% of a 50 Hz mains component, so it
    counts power-line interference as muscle noise; at 100 Hz that falls to a few
    percent and the number measures what it claims to.

    Returns 0.0 when the lead is too short or too flat to judge.

    SUPERSEDED FOR THE INTERPRETATION RULES by `lead_noise_mm`, which measures
    the same thing absolutely; dividing by the lead's own span makes this an
    amplitude test as well as a noise test (see that function). Kept because the
    muscle filter's QRS gate and `interpretation.artifact_statement` are both
    keyed to this ratio, and because the diagnostic captures carry it.
    """
    try:
        x = np.asarray(signal, dtype=float)
        if x.size < int(0.5 * sampling_rate):
            return 0.0
        nyq = sampling_rate / 2.0
        cut = min(100.0, nyq * 0.9)
        b, a = butter(2, cut / nyq, btype="high")
        hf = float(np.std(filtfilt(b, a, x)))
        span = float(np.percentile(x, 99) - np.percentile(x, 1))
        return hf / span if span > 0 else 0.0
    except Exception:
        return 0.0


# A lead whose signal is this close to its own noise is not readable at any
# absolute level: there is no ECG in it to measure, only interference. The two
# cases are an electrode that is off the patient and one that is barely on.
LEAD_NOISE_DOMINANCE_MAX = 0.25
# What such a lead reports, in mm, so a single "> limit" comparison catches both
# it and an ordinarily contaminated lead. Chosen far above any real noise level.
LEAD_NOISE_UNREADABLE_MM = 99.0


def lead_noise_mm(signal: np.ndarray, sampling_rate: float = 500.0,
                  adc_per_mv: float = 1361.0) -> float:
    """High-frequency noise in a lead, in MILLIMETRES at 10 mm/mV.

    Absolute, not a fraction of the lead's own amplitude, and that is the whole
    point of it. The thresholds this number is compared against - the ST bars at
    1 mm and 2 mm, the T wave bars, the tall-P bar - are absolute millimetres, so
    what decides whether a lead can be read is how many millimetres the noise is
    worth, not how it compares with that lead's R wave.

    WHY THE RATIO IT REPLACES WAS WRONG. `lead_noise_ratio` divided the same
    high-frequency measurement by the lead's own peak-to-peak span, which turned
    a noise test into an amplitude test: the numerator is a roughly constant
    front-end floor, so the ratio is really `floor / lead amplitude` and a
    perfectly clean recording of a small-amplitude heart fails it. Measured on
    one device's own 12-lead recording (A091, 2026-09-22 10:59:09, RV5+SV1 =
    0.72 mV, every lead visibly clean on the sheet):

        lead    span      noise      old ratio      verdict
         I      0.45 mV   38.8 uV      0.087        NOISY
         II     1.15 mV   37.9 uV      0.033        clean
         III    0.87 mV    4.4 uV      0.005        clean
         V2     0.48 mV   36.3 uV      0.076        NOISY

    Nine of the twelve failed, all of them on amplitude, leaving three readable
    leads - under the four the ST rule needs - so ST analysis was switched off
    and a report carrying a real elevation would have printed "ST analysis
    limited by signal quality" instead of the finding. The noise is the same
    ~38 uV in every lead; only the denominator moved.

    THE BAND IS THE SAME 100 Hz, and for the same reason: a second-order
    high-pass at 60 Hz still passes about 31% of a 50 Hz mains component, so it
    would count power-line interference as muscle noise; at 100 Hz that falls to
    a few percent and the number measures what it claims to. Muscle artefact
    extends well past 100 Hz, so it is still seen.

    STATED PLAINLY, BECAUSE IT IS A REAL LIMIT: this measures only what is ABOVE
    100 Hz. Baseline wander and a slow (5-10 Hz) tremor are invisible to it -
    measured, an 8 Hz tremor of 1-3 mm leaves this number unchanged at 0.37 mm
    while moving the ST reading by 0.5-0.7 mm. The ratio it replaces had exactly
    the same blind spot; nothing has been given up here, and nothing about
    low-frequency artefact has been gained.

    Returns 0.0 when the lead is too short to judge, and
    LEAD_NOISE_UNREADABLE_MM when the noise is within LEAD_NOISE_DOMINANCE_MAX
    of the lead's own span - a lead with no ECG in it, which no absolute limit
    would catch because amplifier noise on a flat line is small in millimetres.
    """
    try:
        x = np.asarray(signal, dtype=float)
        if x.size < int(0.5 * sampling_rate):
            return 0.0
        scale = float(adc_per_mv) if adc_per_mv else 1361.0
        if scale <= 0:
            scale = 1361.0
        nyq = sampling_rate / 2.0
        cut = min(100.0, nyq * 0.9)
        b, a = butter(2, cut / nyq, btype="high")
        hf_mm = float(np.std(filtfilt(b, a, x))) / scale * 10.0
        span_mm = float(np.percentile(x, 99) - np.percentile(x, 1)) / scale * 10.0
        if span_mm <= 0 or hf_mm > LEAD_NOISE_DOMINANCE_MAX * span_mm:
            return LEAD_NOISE_UNREADABLE_MM
        return hf_mm
    except Exception:
        return 0.0


def stabilize_report_edges(signal: np.ndarray, sampling_rate: float, edge_ms: float = 180.0) -> np.ndarray:
    """
    Reduce edge transients in report strips so the waveform ends smoothly
    (no visible terminal jump/spike in printed reports).
    """
    arr = np.asarray(signal, dtype=float)
    if arr.size < 20:
        return arr

    try:
        fs = float(sampling_rate)
    except Exception:
        fs = 500.0
    if fs <= 0:
        fs = 500.0

    edge_n = max(8, int((edge_ms / 1000.0) * fs))
    edge_n = min(edge_n, max(8, arr.size // 6))
    if edge_n * 3 >= arr.size:
        return arr

    stable_tail_start = arr.size - (3 * edge_n)
    stable_tail_end = arr.size - edge_n
    stable_head_start = edge_n
    stable_head_end = 3 * edge_n

    head_target = float(np.median(arr[stable_head_start:stable_head_end]))
    tail_target = float(np.median(arr[stable_tail_start:stable_tail_end]))

    out = arr.copy()
    ramp = np.linspace(0.0, 1.0, edge_n, endpoint=True)
    out[:edge_n] = (1.0 - ramp) * head_target + ramp * out[:edge_n]
    out[-edge_n:] = (1.0 - ramp) * out[-edge_n:] + ramp * tail_target
    return out


# THE GAUSSIAN IS THE ANDROID APP'S, NOT THIS MODULE'S OWN.
#
# EnhancedECGFilter.kt derives its width as sigma = fs / (2*pi*cutoff), which is
# 1 / (2*pi) = 0.15915 in the form used here, and takes a kernel half-width of
# 3*sigma. Both devices now smooth the trace identically, which is the point: a
# CardioX report and the Android chart drawn from the same recording have to be
# the same picture, or a clinician comparing them has no way to tell which one
# is the device and which is the patient.
#
# WHAT THIS COST, STATED PLAINLY. The value that stood here was 0.13250, which is
# the mathematically exact -3 dB width: H(f) = exp(-2 pi^2 sigma^2 f^2) reaches
# 1/sqrt(2) at sigma * f = sqrt(ln(2)/2) / (2*pi). The Kotlin's comment claims
# its formula "matches the -3 dB point" and it does not - at its own cutoff it
# measures 0.607, which is -4.3 dB, so the Kotlin Gaussian is about 20% wider and
# smooths slightly harder than its label says. The label on the report is the
# cutoff the user picked either way; it is now the Android app's definition of
# that cutoff rather than the textbook one. If the Android side is ever corrected
# to sqrt(ln(2)/2)/(2*pi), change this constant with it and the two stay matched.
_GAUSSIAN_SIGMA_KOTLIN = 1.0 / (2.0 * np.pi)          # 0.159155
_GAUSSIAN_SIGMA_AT_MINUS_3DB = _GAUSSIAN_SIGMA_KOTLIN

# Kernel half-width in sigmas. The Kotlin uses 3 (99.7% of the Gaussian's
# energy); this module used 4. At 3 sigma the truncation leaves a floor around
# -60 dB, which is far below anything on an ECG, and matching it keeps the two
# kernels identical sample for sample.
_GAUSSIAN_HALF_WIDTH_SIGMAS = 3.0


def _gaussian_lowpass(signal: np.ndarray, sampling_rate: float, cutoff_freq: float) -> np.ndarray:
    """
    Zero-phase low-pass with a Gaussian kernel.

    A Gaussian kernel is strictly positive and its frequency response has no
    side lobes, so unlike a Butterworth it CANNOT overshoot a sharp edge. That
    matters here: the QRS is a step-like edge, and any overshoot lands in the
    40-90 ms window after it and gets drawn as a J wave that is not in the
    patient's signal. See apply_emg_filter() for the full explanation.
    """
    n = signal.size
    if n == 0:
        return signal

    sigma = _GAUSSIAN_SIGMA_AT_MINUS_3DB * sampling_rate / cutoff_freq
    if sigma <= 0.05:
        # Cutoff is at/above the usable band — nothing meaningful to remove.
        return signal

    # The kernel must fit inside the signal, and np.pad(mode="reflect") requires
    # the pad width to be smaller than the array. Shrink rather than fail, and
    # leave signals too short to carry a kernel untouched.
    half = max(1, int(np.ceil(_GAUSSIAN_HALF_WIDTH_SIGMAS * sigma)))
    half = min(half, (n - 1) // 2)
    if half < 1:
        return signal

    offsets = np.arange(2 * half + 1) - half
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()

    # Reflect the ends before convolving so the first and last beats are not
    # pulled toward zero by the implicit zero-padding of a plain convolution.
    # "valid" over the padded copy returns exactly n samples; "same" would
    # return max(len(signal), len(kernel)) and silently lengthen short inputs.
    padded = np.pad(signal, half, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def apply_emg_filter(signal: np.ndarray, sampling_rate: float, emg_filter: str) -> np.ndarray:
    """
    Apply EMG Filter (Low-pass filter) to suppress muscle artifacts.

    Uses a zero-phase GAUSSIAN low-pass, not a Butterworth.

    The previous implementation was butter(4) applied through filtfilt. filtfilt
    runs the filter forwards and backwards, so that design has the magnitude
    response of an 8th-order Butterworth, which overshoots a step edge by ~6.8%
    and — being zero-phase — rings symmetrically on both sides of it. The
    overshoot amplitude is set by the order, but the ringing WIDTH is set by the
    cutoff: at 150 Hz the lobe is ~14 ms and hides inside the QRS, while at 25 Hz
    it is ~42 ms and lands squarely on the J point. It was then drawn as a J wave
    / notch with pseudo ST elevation that is absent from the same patient's
    150 Hz recording.

    A Gaussian kernel has no overshoot at all (measured 0.00% against 6.83% for
    the old design), so the strip follows the real signal back to baseline after
    the S wave. It trades some stop-band steepness for that, which is the correct
    trade for a diagnostic ECG: a muscle filter must not invent waveform.

    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz
        emg_filter: Cutoff frequency - "25", "35", "40", "75", "100", or "150" (Hz)

    Returns:
        Filtered signal
    """
    if not emg_filter or emg_filter == "off":
        return signal

    try:
        cutoff_freq = float(emg_filter)
        nyquist = sampling_rate / 2.0

        # Safety: if sampling_rate is unexpectedly low, fall back to 500 Hz
        if sampling_rate < 100:  # hardware should be 500 Hz; avoid unstable filters
            nyquist = 500.0 / 2.0
            sampling_rate = 500.0

        # Clamp cutoff so it is always below Nyquist for a valid low‑pass filter
        max_allowed = nyquist * 0.9
        if cutoff_freq >= max_allowed:
            cutoff_freq = max_allowed
        if cutoff_freq <= 0:
            return signal

        return _gaussian_lowpass(np.asarray(signal, dtype=float), sampling_rate, cutoff_freq)

    except Exception as e:
        print(f" Error applying EMG filter ({emg_filter}Hz): {e}")
        return signal


def apply_dft_filter(signal: np.ndarray, sampling_rate: float, dft_filter: str) -> np.ndarray:
    """
    Apply DFT Filter (High-pass filter) to remove baseline wander
    
    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz
        dft_filter: Cutoff frequency - "off", "0.05", or "0.5" (Hz)
    
    Returns:
        Filtered signal
    """
    if dft_filter == "off" or not dft_filter:
        return signal

    try:
        cutoff_freq = float(dft_filter)  # Low cutoff frequency (0.05 or 0.5 Hz)

        # THE 0.5 Hz CASE NO LONGER TAKES A DIFFERENT ROUTE.
        #
        # It used to be special-cased to apply_baseline_wander_median_mean - a
        # median+mean baseline estimate rather than a high-pass. That is a
        # perfectly good baseline remover, but it is not what the Android app
        # does, so the same recording came off the two devices with differently
        # shaped baselines and there was no way to tell from either picture
        # which shape belonged to the patient.
        #
        # EnhancedECGFilter.kt uses a 2nd-order Butterworth high-pass at 0.5 Hz
        # run through filtfilt TWICE - see applyHighPassFilter, which calls
        # filtfilt on its own output. That is a 4th-order zero-phase response,
        # and both passes are reproduced below because dropping one would make
        # this the gentler filter of the two and put the difference straight
        # back. apply_baseline_wander_median_mean is still exported and still
        # used where a caller asks for it by name.
        nyquist = sampling_rate / 2.0
        normalized_cutoff = cutoff_freq / nyquist

        # The Kotlin's own guard: a cutoff at or above half of Nyquist is not a
        # baseline filter and is refused rather than designed.
        if normalized_cutoff <= 0 or normalized_cutoff >= 0.5:
            print(f" DFT filter cutoff {cutoff_freq}Hz is invalid for sampling rate {sampling_rate}Hz")
            return signal

        b, a = butter(2, normalized_cutoff, btype='high')
        return filtfilt(b, a, filtfilt(b, a, signal))

    except Exception as e:
        print(f" Error applying DFT filter ({dft_filter}Hz): {e}")
        return signal


def process_cardiox_grade(ecg: np.ndarray, fs: float = 500.0, apply_sharpening: bool = False) -> np.ndarray:
    """
    Complete monitor-grade ECG processing pipeline with all corrections.
    
    Pipeline order:
    1. ADC normalization (DC offset removal only - variance preserved to prevent instability)
    2. Powerline removal (50 Hz notch, Q=25)
    3. Baseline correction (respiration preserved, 120 ms median)
    4. EMG suppression (35-40 Hz low-pass)
    5. QRS sharpening (gated, only in QRS regions) - OPTIONAL to prevent instability
    
    Args:
        ecg: Raw ECG ADC signal
        fs: Sampling rate (Hz, default 500)
        apply_sharpening: If True, apply QRS sharpening (default: False to prevent instability)
    
    Returns:
        Processed ECG with sharp P-QRS-T waves, preserved ST/QT
    """
    ecg = np.asarray(ecg, dtype=float)
    
    if len(ecg) < 10:
        return ecg
    
    # Step 1: ADC normalization (remove DC offset only - preserve amplitude to prevent flickering)
    ecg = normalize_adc_signal(ecg, preserve_amplitude=True)
    
    # Step 2: Remove powerline interference (50 Hz, Q=25 to avoid ringing)
    ecg = notch_filter_butterworth(ecg, fs, freq=50.0, q=25.0)
    
    # Step 3: Correct baseline (preserve respiration, 120 ms median to avoid QRS erosion)
    ecg, _ = ecg_with_respiratory_baseline(ecg, fs)
    
    # Step 4: Suppress EMG noise (35-40 Hz low-pass, not high-pass)
    ecg = apply_emg_filter(ecg, fs, "35")
    
    # Step 5: Sharpen QRS (gated, only in QRS regions, preserves P/T waves)
    # DISABLED by default - can cause instability if QRS detection is inconsistent
    if apply_sharpening:
        ecg = sharpen_qrs_gated(ecg, fs, alpha=0.3)
    
    return ecg


def apply_ecg_filters(
    signal: Union[np.ndarray, list],
    sampling_rate: float = 500,
    ac_filter: Optional[str] = None,
    emg_filter: Optional[str] = None,
    dft_filter: Optional[str] = None
) -> np.ndarray:
    """
    Apply all ECG filters in the correct order:
    1. DFT Filter (baseline wander removal) - first
    2. AC Filter (power line interference removal)
    3. EMG Filter (muscle artifact removal) - last

    The AC filter runs BEFORE the muscle filter, not after it. See the note
    beside it in the body for the measurement that settled the order; this
    docstring described the old sequence long after the code had changed.
    
    Args:
        signal: Input ECG signal (numpy array or list)
        sampling_rate: Sampling frequency in Hz (default: 500)
        ac_filter: AC filter setting - "off", "50", or "60"
        emg_filter: EMG filter setting - "25", "35", "40", "75", "100", "150"
        dft_filter: DFT filter setting - "off", "0.05", or "0.5"
    
    Returns:
        Filtered signal as numpy array
    """
    # Convert to numpy array if needed
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal, dtype=float)
    
    # Check minimum signal length
    if len(signal) < 10:
        return signal
    
    # Apply filters in correct order
    filtered = signal.copy()
    
    # 1. DFT Filter first (removes slow baseline wander)
    if dft_filter:
        filtered = apply_dft_filter(filtered, sampling_rate, dft_filter)
    
    # 2. AC filter BEFORE the muscle filter. Mains interference is not signal,
    #    and nothing downstream should have to work around it.
    #
    #    This used to run last and be SKIPPED whenever the EMG cutoff was under
    #    60 Hz, on the reasoning that a 25 Hz low-pass removes 50 Hz anyway. That
    #    held for the old butter(4) design, which left 2.7% of a 50 Hz component.
    #    It does not hold for the Gaussian that replaced it: a Gaussian is gentle
    #    by construction - that is why it does not ring - and it leaves 25.3% of
    #    50 Hz standing, nine times as much. With the notch skipped there was
    #    then nothing else to remove it.
    #
    #    Measured on a real recording carrying heavy mains (lead I at 138% of
    #    signal): EMG 25 alone left mains at 54.4, EMG 25 "+ AC 50" left exactly
    #    the same 54.4 because the notch never ran, and the notch on its own
    #    brought it to 0.2. The skip was costing everything and saving nothing -
    #    a narrow bandstop is cheap, so it now always runs when asked for.
    if ac_filter:
        filtered = apply_ac_filter(filtered, sampling_rate, ac_filter)

    # 3. EMG Filter last (removes muscle artifacts)
    if emg_filter and str(emg_filter).lower() != "off":
        filtered = apply_emg_filter(filtered, sampling_rate, emg_filter)

    return filtered


def apply_baseline_wander_median_mean(signal: np.ndarray, sampling_rate: float = 500) -> np.ndarray:
    """
    GOLD STANDARD: Median Filter + Mean Filter for baseline wander removal
    Used in many commercial ECG monitors.
    
    Why it's special:
    - Removes baseline without touching QRS or ST segments
    - No phase distortion
    - Excellent for real-time display
    
    Algorithm:
    1. Median filter (200-300 ms) → removes QRS influence
    2. Moving average (600-1000 ms) → smooths baseline
    3. Subtract baseline from original signal
    
    Typical parameters (500 Hz):
    - Median filter: 200-300 ms (100-150 samples)
    - Mean filter: 600-1000 ms (300-500 samples)
    
    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz (default: 500)
    
    Returns:
        Clean ECG signal with baseline wander removed
    """
    if len(signal) < 50:  # Need minimum samples
        return signal - np.mean(signal)
    
    try:
        # Step 1: Median filter (120 ms window - reduced from 200 ms to avoid QRS erosion)
        median_window_ms = 120.0  # milliseconds
        median_window = int(median_window_ms * sampling_rate / 1000.0)
        median_window = max(3, min(median_window, len(signal) // 2))  # Ensure odd and reasonable
        
        # Make window odd (required for median filter)
        if median_window % 2 == 0:
            median_window += 1
        
        # Apply median filter to remove QRS influence
        b1 = medfilt(signal, kernel_size=median_window)
        
        # Step 2: Moving average (600-1000 ms window)
        # Use 800 ms as middle value: 0.8 * Fs
        mean_window_ms = 800.0  # milliseconds
        mean_window = int(mean_window_ms * sampling_rate / 1000.0)
        mean_window = max(10, min(mean_window, len(b1) // 2))  # Ensure reasonable
        
        # Apply moving average to smooth baseline
        baseline = uniform_filter1d(b1.astype(float), size=mean_window, mode='nearest')
        
        # Step 3: Subtract baseline from original signal
        clean_ecg = signal - baseline
        
        return clean_ecg
    
    except Exception as e:
        print(f" Error applying median+mean baseline filter: {e}")
        # Fallback: simple mean subtraction
        return signal - np.mean(signal)


def notch_filter_butterworth(ecg: np.ndarray, fs: float, freq: float = 50.0, q: float = 25.0) -> np.ndarray:
    """
    Notch filter using Butterworth design (for India → 50 Hz)
    
    Args:
        ecg: Input ECG signal
        fs: Sampling frequency in Hz
        freq: Notch frequency (default: 50.0 Hz for India)
        q: Quality factor (default: 30.0)
    
    Returns:
        Filtered signal with powerline noise removed
    """
    if len(ecg) < 10:
        return ecg
    
    try:
        w0 = freq / (fs / 2.0)
        if w0 <= 0 or w0 >= 1:
            return ecg
        
        b, a = butter(2, [w0 - w0/q, w0 + w0/q], btype='bandstop')
        return filtfilt(b, a, ecg)
    except Exception as e:
        print(f" Error applying notch filter: {e}")
        return ecg


def estimate_baseline_drift(ecg: np.ndarray, fs: float) -> np.ndarray:
    """
    Estimate baseline drift (motion + electrode drift) using median + mean filter.
    This is the clinical gold standard method.
    
    Algorithm:
    1. Median filter (200 ms) → removes QRS influence
    2. Moving average (1.8 s) → smooths baseline drift
    
    Args:
        ecg: Input ECG signal
        fs: Sampling frequency in Hz
    
    Returns:
        Estimated baseline drift signal
    """
    if len(ecg) < 50:
        return np.zeros_like(ecg)
    
    try:
        # Median filter removes QRS influence (120 ms window - reduced from 200 ms to avoid QRS erosion)
        median_window = int(0.12 * fs) | 1  # Ensure odd
        if median_window < 3:
            median_window = 3
        if median_window > len(ecg) // 2:
            median_window = (len(ecg) // 2) | 1
        
        med = medfilt(ecg, kernel_size=median_window)
        
        # Moving average removes remaining slow drift (1.8 s window)
        mean_window = int(1.8 * fs)
        if mean_window < 10:
            mean_window = 10
        if mean_window > len(med):
            mean_window = len(med)
        
        # Use convolution for moving average
        kernel = np.ones(mean_window) / mean_window
        drift = np.convolve(med, kernel, mode='same')
        
        return drift
    except Exception as e:
        print(f" Error estimating baseline drift: {e}")
        return np.zeros_like(ecg)


def extract_respiration(drift_signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Extract respiration component from baseline drift signal (EDR - ECG-Derived Respiration).
    CORRECTED: Extracts from drift signal, not raw ECG, for better accuracy.
    
    Respiration band: 0.1 - 0.35 Hz
    
    Args:
        drift_signal: Baseline drift signal (from estimate_baseline_drift)
        fs: Sampling frequency in Hz
    
    Returns:
        Respiration waveform (EDR) with safe amplitude scaling
    """
    if len(drift_signal) < 10:
        return np.zeros_like(drift_signal)
    
    try:
        # Low-pass filter at 0.35 Hz to extract respiration from drift signal
        nyquist = fs / 2.0
        cutoff = 0.35 / nyquist
        
        if cutoff <= 0 or cutoff >= 1:
            return np.zeros_like(drift_signal)
        
        b, a = butter(2, cutoff, btype='low')
        resp = filtfilt(b, a, drift_signal)
        
        # Remove DC offset
        resp = resp - np.mean(resp)
        
        # Safe amplitude scaling instead of hard clipping
        # Scale to ±0.6 mV range if amplitude exceeds threshold
        max_amplitude = np.max(np.abs(resp)) if len(resp) > 0 else 0.0
        if max_amplitude > 0.6:
            scale_factor = 0.6 / max_amplitude
            resp = resp * scale_factor
        
        return resp
    except Exception as e:
        print(f" Error extracting respiration: {e}")
        return np.zeros_like(drift_signal)


def ecg_with_respiratory_baseline(ecg: np.ndarray, fs: float = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    🫀 MONITOR-GRADE: ECG with Respiration-Controlled Baseline
    
    This is exactly how bedside ECG monitors behave:
    - ✅ Stable ECG baseline
    - ✅ Baseline moves only with respiration
    - ✅ Motion drift suppressed
    - ✅ Safe for ST segment
    - ✅ Real-time friendly
    
    Algorithm:
    1. Remove powerline noise (50 Hz notch for India)
    2. Estimate unwanted baseline drift (motion + electrode drift)
    3. Extract respiration component (0.1-0.35 Hz)
    4. Clamp respiration amplitude (prevents crazy baseline swing)
    5. Reconstruct ECG: clean_ecg = ecg_notched - drift + respiration
    
    Args:
        ecg: Input ECG signal
        fs: Sampling frequency in Hz (default: 500)
    
    Returns:
        tuple: (clean_ecg, respiration)
            - clean_ecg: Stable ECG with breathing baseline
            - respiration: Respiration waveform (EDR)
    
    Example:
        clean_ecg, respiration = ecg_with_respiratory_baseline(signal, fs=500)
        # clean_ecg: Flat baseline at rest, smooth sinusoidal motion while breathing
        # respiration: Can be used to calculate respiration rate
    """
    ecg = np.asarray(ecg, dtype=float)
    
    if len(ecg) < 50:
        # Too short for filtering, just center it
        centered = ecg - np.mean(ecg)
        return centered, np.zeros_like(centered)
    
    try:
        # 1. Remove powerline noise (50 Hz for India, Q=25 to avoid ringing)
        ecg_notched = notch_filter_butterworth(ecg, fs, freq=50.0, q=25.0)
        
        # 2. Estimate unwanted baseline drift (motion + electrode drift)
        drift = estimate_baseline_drift(ecg_notched, fs)
        
        # 3. Extract respiration component from drift signal (0.1-0.35 Hz)
        # CORRECTED: Extract from drift, not raw ECG, with safe amplitude scaling
        respiration = extract_respiration(drift, fs)
        
        # 5. Reconstruct ECG: remove drift, add back respiration
        clean_ecg = ecg_notched - drift + respiration
        
        return clean_ecg, respiration
    
    except Exception as e:
        print(f" Error in respiration-preserving baseline correction: {e}")
        # Fallback: simple mean subtraction
        centered = ecg - np.mean(ecg)
        return centered, np.zeros_like(centered)


def respiration_rate(resp: np.ndarray, fs: float) -> float:
    """
    Calculate respiration rate from respiration waveform (EDR).
    
    Args:
        resp: Respiration waveform (from extract_respiration or ecg_with_respiratory_baseline)
        fs: Sampling frequency in Hz
    
    Returns:
        Respiration rate in breaths per minute (BPM)
    """
    if len(resp) < 10:
        return 0.0
    
    try:
        # Count zero crossings (positive-going)
        zero_crossings = np.where(np.diff(np.sign(resp)) > 0)[0]
        breaths = len(zero_crossings)
        
        # Calculate duration in minutes
        duration_min = len(resp) / fs / 60.0
        
        if duration_min > 0:
            return breaths / duration_min
        else:
            return 0.0
    except Exception as e:
        print(f" Error calculating respiration rate: {e}")
        return 0.0


def apply_ecg_filters_from_settings(
    signal: Union[np.ndarray, list],
    sampling_rate: float = 500,
    settings_manager=None
) -> np.ndarray:
    """
    Apply ECG filters using settings from SettingsManager
    
    Args:
        signal: Input ECG signal
        sampling_rate: Sampling frequency in Hz
        settings_manager: SettingsManager instance (optional, will create if not provided)
    
    Returns:
        Filtered signal
    """
    # Import here to avoid circular imports
    if settings_manager is None:
        from utils.settings_manager import SettingsManager
        settings_manager = SettingsManager()
    
    # Get filter settings
    ac_filter = settings_manager.get_setting("filter_ac", "50")
    # 150, matching every screen's own default. This read 25, so a caller that
    # relied on the fallback drew a different bandwidth from the one beside it.
    emg_filter = settings_manager.get_setting("filter_emg", "150")
    dft_filter = settings_manager.get_setting("filter_dft", "off")
    
    # Apply filters
    return apply_ecg_filters(
        signal=signal,
        sampling_rate=sampling_rate,
        ac_filter=ac_filter,
        emg_filter=emg_filter,
        dft_filter=dft_filter
    )
