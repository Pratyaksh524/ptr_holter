"""Baseline wander removal ahead of the 0.5 Hz high-pass: two-stage median.

Baseline wander (breathing, electrode / cable movement, the simulator's
"baseline wander" artifact) is a slow drift of a lead's level, 0.05-0.5 Hz
and up to several mV. The strip's 0.5 Hz high-pass (ecg_filters.
apply_dft_filter, a 2nd-order Butterworth run through filtfilt twice) leaves
~1 mV of a 5-8 mV, 0.12-0.2 Hz wander, and cannot follow the steps the
simulator's wander holds (3.7 mV in 0.7 s at 09:52:44, 2026-11-13_09-52-26).
Raising its cutoff is no answer: at 1.0 Hz the ST segment moved 2.7 mm and
the T wave lost 5% on clean recordings.

The two-stage median (de Chazal / Sornmo & Laguna) estimates the baseline
itself and subtracts it:
  1. a median over BASELINE_MEDIAN_1_SEC (0.2 s) takes the QRS and P waves
     out (both shorter than half the window),
  2. a median over BASELINE_MEDIAN_2_SEC (0.6 s) of that takes the T wave
     out; what is left is the baseline.
Being medians they do not smear a step the way a linear filter does, and
they keep the P / QRS / ST / T shape of a narrow beat (ST error <= 0.4 mm,
QRS height <= 1% on clean simulator recordings).

Wide beats (a PVC, a VT beat: QRS 160-270 ms plus its T wave) are longer
than the 0.6 s window and bend the medians -- their QRS height changed
3-16% in tests. Callers that know where such beats are pass them as
mask_spans; the baseline is interpolated linearly across each span from
the baseline on either side, so the beat itself is not read.

Where it runs: in front of every 0.5 Hz high-pass that reads the
recording for a measurement -- the auto segment detector's sinus P check
(_sinus_gate_filter), PR / QRS / QT (pqrst_measure.measure_beats), ST / T
(ischemia_detection._fd_filter), the QRS axis (axis_detection.
_strip_filter) and the Full Disclosure strip -- each with its beats
masked (remove_baseline_wander_r). Not in the band-pass the window rules
(VFib / VT / AF / Flutter) and the ectopic beats detector use: an
unmasked median there changed the auto segments of 35 of 156 recordings
and the ectopic counts of 15 (wide beats bent the medians); the wander's
measured harm was only in P / PR / QT.
"""
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter

BASELINE_MEDIAN_1_SEC = 0.2   # removes P and QRS
BASELINE_MEDIAN_2_SEC = 0.6   # removes T
# Samples on each side of a masked (wide-beat) span whose baseline is used
# to bridge it.
_BRIDGE_SEC = 0.05


def _odd(n: int) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def baseline(signal: Sequence[float], fs: float,
             mask_spans: Optional[Iterable[Tuple[int, int]]] = None) -> np.ndarray:
    """The baseline wander of one lead (same length as `signal`).

    mask_spans: (start_index, end_index) sample ranges of wide beats
    (QRS + T) to bridge across instead of reading."""
    x = np.asarray(signal, dtype=float)
    if x.size < 8:
        return np.full_like(x, float(np.median(x)) if x.size else 0.0)
    fs = float(fs or 500.0)
    b = median_filter(x, size=_odd(BASELINE_MEDIAN_1_SEC * fs), mode='nearest')
    b = median_filter(b, size=_odd(BASELINE_MEDIAN_2_SEC * fs), mode='nearest')
    if mask_spans:
        n = x.size
        br = max(1, int(_BRIDGE_SEC * fs))
        for s, e in sorted(mask_spans):
            s, e = max(0, int(s)), min(n, int(e))
            if e <= s:
                continue
            left = float(np.median(b[max(0, s - br):s])) if s > 0 else None
            right = float(np.median(b[e:min(n, e + br)])) if e < n else None
            if left is None and right is None:
                continue
            if left is None:
                left = right
            if right is None:
                right = left
            b[s:e] = np.linspace(left, right, e - s, endpoint=False)
    return b


def remove_baseline_wander(signal: Sequence[float], fs: float,
                           mask_spans: Optional[Iterable[Tuple[int, int]]] = None) -> np.ndarray:
    """`signal` minus its baseline(); the lead's own median level is kept
    so the strip stays where it is drawn."""
    x = np.asarray(signal, dtype=float)
    if x.size < 8:
        return x.copy()
    return x - baseline(x, fs, mask_spans) + float(np.median(x))


# Mask around every R peak when estimating the baseline: QRS (from
# MASK_PRE_SEC before R) and T wave (to MASK_POST_SEC after R) -- long enough
# for a 270 ms PVC QRS and its T. Where the masks cover more than
# MASK_MAX_FRACTION of the signal (fast rhythms) nothing would be left to
# estimate from, and the plain medians are used.
MASK_PRE_SEC = 0.12
MASK_POST_SEC = 0.40
MASK_MAX_FRACTION = 0.85


def _r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """R peaks (sample index) of one lead for the mask: Pan-Tompkins on the
    lead itself."""
    try:
        from ecg.pan_tompkins import pan_tompkins
        return np.asarray(pan_tompkins(np.asarray(x, dtype=float), fs=fs), dtype=int)
    except Exception:
        return np.zeros(0, dtype=int)


def remove_baseline_wander_r(signal: Sequence[float], fs: float,
                             r_idx: Optional[Sequence[int]] = None) -> np.ndarray:
    """remove_baseline_wander() with the beats masked: r_idx are the R peaks
    (sample index into `signal`), found on the lead itself when None."""
    x = np.asarray(signal, dtype=float)
    if x.size < int(2 * fs):
        return x.copy()
    fs = float(fs or 500.0)
    r = np.asarray(r_idx, dtype=int) if r_idx is not None else _r_peaks(x, fs)
    pre, post = int(MASK_PRE_SEC * fs), int(MASK_POST_SEC * fs)
    spans = [(max(0, int(i) - pre), min(x.size, int(i) + post)) for i in r]
    covered = np.zeros(x.size, dtype=bool)
    for a, b in spans:
        covered[a:b] = True
    if r.size == 0 or float(covered.mean()) > MASK_MAX_FRACTION:
        spans = None
    return remove_baseline_wander(x, fs, spans)
