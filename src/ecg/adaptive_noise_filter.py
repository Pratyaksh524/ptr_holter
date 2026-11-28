"""Small, morphology-preserving adaptive mains cleaner.

This module is deliberately separate from the ECG detector.  It removes only
coherent power-line tones (the measured fundamental and its second harmonic)
from a block, then leaves the existing display and detector filters to do
their normal work.  It never changes the raw recording on disk.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _as_rows(signal: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError("signal must be one- or two-dimensional")


def estimate_mains_frequency(signal: np.ndarray, fs: float) -> Optional[float]:
    """Estimate a coherent 50/60-Hz supply tone from one or more leads.

    The estimate is intentionally limited to the usual supply bands.  A
    quadratic interpolation around the FFT peak gives useful values such as
    50.05 Hz without assuming the nominal frequency is exactly 50 Hz.
    ``None`` means that no sufficiently concentrated tone was found.
    """
    rows, _ = _as_rows(signal)
    n = rows.shape[1]
    fs = float(fs)
    if n < max(256, int(round(fs))):
        return None

    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    rows = rows - np.median(rows, axis=1, keepdims=True)
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(rows * window, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    # Search both possible mains bands.  Normalising each lead prevents one
    # high-amplitude lead from hiding a coherent tone on the others.
    scores = []
    for lo, hi in ((45.0, 55.0), (55.0, 65.0)):
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask):
            continue
        band = spectrum[:, mask]
        scale = np.maximum(np.median(band, axis=1, keepdims=True), 1e-12)
        scores.append((band / scale, np.flatnonzero(mask)))

    if not scores:
        return None
    score = np.concatenate([item[0] for item in scores], axis=1)
    bins = np.concatenate([item[1] for item in scores])
    aggregate = np.median(score, axis=0)
    best = int(np.argmax(aggregate))
    # A broad/noisy spectrum should not be treated as mains.  A real supply
    # tone is a concentrated peak relative to nearby bins.
    left = max(0, best - 2)
    right = min(aggregate.size, best + 3)
    local = np.delete(aggregate[left:right], best - left)
    if local.size and aggregate[best] < max(3.0, 2.5 * float(np.median(local))):
        return None

    k = int(bins[best])
    delta = 0.0
    if 1 <= k < len(freqs) - 1:
        # Parabolic interpolation of log power around the FFT maximum.
        p0 = np.log(max(float(np.sum(spectrum[:, k - 1])), 1e-12))
        p1 = np.log(max(float(np.sum(spectrum[:, k])), 1e-12))
        p2 = np.log(max(float(np.sum(spectrum[:, k + 1])), 1e-12))
        denom = p0 - 2.0 * p1 + p2
        if abs(denom) > 1e-12:
            delta = float(np.clip(0.5 * (p0 - p2) / denom, -0.5, 0.5))

    frequency = float(freqs[k] + delta * fs / n)
    if 45.0 <= frequency <= 55.0 or 55.0 <= frequency <= 65.0:
        return frequency
    return None


def remove_adaptive_mains(
    signal: np.ndarray,
    fs: float,
    mains_hz: Optional[float] = None,
    min_tone_ratio: float = 0.01,
) -> np.ndarray:
    """Subtract the measured mains tone and its second harmonic.

    The subtraction is a least-squares sinusoid fit, not a broad low-pass.  It
    therefore preserves QRS/ST morphology better than lowering the display
    cutoff.  The fit is per lead, while the frequency is shared by the block.
    If no coherent tone is present, the input is returned unchanged.
    """
    rows, was_1d = _as_rows(signal)
    n = rows.shape[1]
    fs = float(fs)
    if n < max(256, int(round(fs))) or fs <= 0:
        return np.asarray(signal, dtype=float).copy()

    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    frequency = float(mains_hz) if mains_hz is not None else estimate_mains_frequency(rows, fs)
    if frequency is None or frequency <= 0.0 or 2.0 * frequency >= fs / 2.0:
        out = rows.copy()
        return out[0] if was_1d else out

    # Fit short overlapping windows rather than one coefficient set for the
    # whole block. Cable pickup can change amplitude/phase as the wire moves;
    # local fits remove that variation without touching the QRS band.
    fit_window = min(n, max(int(round(fs)), int(round(2.0 * fs))))
    hop = max(1, fit_window // 2)
    starts = list(range(0, max(1, n - fit_window + 1), hop))
    final_start = max(0, n - fit_window)
    if starts[-1] != final_start:
        starts.append(final_start)

    centers = np.asarray(starts, dtype=float) + fit_window / 2.0
    coefficients = np.zeros((len(starts), 4, rows.shape[0]), dtype=float)
    for wi, start in enumerate(starts):
        stop = min(n, start + fit_window)
        t_local = np.arange(start, stop, dtype=float) / fs
        design = np.column_stack((
            np.sin(2.0 * np.pi * frequency * t_local),
            np.cos(2.0 * np.pi * frequency * t_local),
            np.sin(4.0 * np.pi * frequency * t_local),
            np.cos(4.0 * np.pi * frequency * t_local),
        ))
        local = rows[:, start:stop]
        centered = local - np.median(local, axis=1, keepdims=True)
        fitted, *_ = np.linalg.lstsq(design, centered.T, rcond=None)

        # Do not touch leads/windows where the fitted tone is negligible. This
        # keeps the cleaner a no-op on clean ECG and protects morphology.
        tone_amp = np.hypot(fitted[0], fitted[1])
        tone_amp = np.hypot(tone_amp, np.hypot(fitted[2], fitted[3]))
        signal_pp = np.maximum(np.ptp(centered, axis=1), 1e-9)
        use = tone_amp >= float(min_tone_ratio) * signal_pp
        fitted[:, ~use] = 0.0
        coefficients[wi] = fitted

    if not np.any(coefficients):
        out = rows.copy()
        return out[0] if was_1d else out

    # Interpolate the local sin/cos coefficients at every sample.  This avoids
    # seams at window boundaries and has no causal delay in the display or
    # offline recording analysis paths.
    sample_times = np.arange(n, dtype=float)
    coefficient_track = np.empty((4, rows.shape[0], n), dtype=float)
    for component in range(4):
        for lead in range(rows.shape[0]):
            coefficient_track[component, lead] = np.interp(
                sample_times, centers, coefficients[:, component, lead]
            )
    t = np.arange(n, dtype=float) / fs
    design = np.column_stack((
        np.sin(2.0 * np.pi * frequency * t),
        np.cos(2.0 * np.pi * frequency * t),
        np.sin(4.0 * np.pi * frequency * t),
        np.cos(4.0 * np.pi * frequency * t),
    ))
    tone = np.einsum("nk,kln->ln", design, coefficient_track)
    out = rows.copy()
    out -= tone
    return out[0] if was_1d else out


__all__ = ["estimate_mains_frequency", "remove_adaptive_mains"]
