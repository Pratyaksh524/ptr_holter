"""A port of the Android app's EnhancedECGFilter.kt, coefficient for coefficient.

WHY A PORT RATHER THAN A NEW FILTER. The Kotlin filter is what the Android
CardioX chart draws through, so it is the reference for what a filtered trace on
this hardware is supposed to look like. Writing a second filter here with
different coefficients would mean two devices disagreeing about the same signal
for reasons nobody could later reconstruct. Everything below follows
EnhancedECGFilter.kt: the same biquad designs, the same cascade order, the same
quality factor, the same Gaussian kernel.

Two paths, exactly as in the Kotlin:

    CausalFilter      single pass, state persists between calls. This is the
                      chart path - no backward pass means no edge artefact and
                      no cold-start delay when a lead reconnects mid-stream.
                      One instance per lead; the six biquad slots hold that
                      lead's history and must never be shared.

    apply_complete_filter()
                      zero-phase filtfilt, batch, no persistent state. This is
                      the analysis path. At 25 and 40 Hz it uses a Gaussian FIR
                      rather than a Butterworth, which is what the Kotlin does
                      and is the reason its 25 Hz trace has no ringing.

ONE DELIBERATE DIFFERENCE, and it is the only one. The Kotlin causal path always
applies a low-pass - `lp` is not nullable there. Here `lowpass="off"` bypasses
it, because the caller this was written for is a viewer whose entire purpose is
showing the signal with nothing applied. Every other setting, including "off"
for AC, EMG and the high-pass, behaves as the Kotlin does.

The cascade order is load-bearing and is the Kotlin's:

    high-pass → mains notch → 2nd harmonic notch → low-pass → EMG (4th order)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

# From the Kotlin companion object.
QUALITY_FACTOR = 25.0
LOWPASS_CUTOFF = 40.0
HIGHPASS_CUTOFF = 0.5


class Biquad:
    __slots__ = ("b", "a")

    def __init__(self, b, a):
        self.b = [float(x) for x in b]
        self.a = [float(x) for x in a]


# ──────────────────────────────────────────────────────────────────────────────
# Coefficient design - one function per makeXxx() in the Kotlin
# ──────────────────────────────────────────────────────────────────────────────

def make_notch(freq_hz: float, fs: float) -> Biquad:
    w0 = freq_hz / (fs / 2.0) * math.pi
    bw = w0 / QUALITY_FACTOR
    gb = 1.0 / math.sqrt(2.0)
    beta = math.sqrt(1.0 - gb * gb) / gb * math.tan(bw / 2.0)
    gain = 1.0 / (1.0 + beta)
    return Biquad(
        b=[gain, -2.0 * gain * math.cos(w0), gain],
        a=[1.0, -2.0 * gain * math.cos(w0), 2.0 * gain - 1.0],
    )


def make_lowpass(cutoff_hz: float, fs: float) -> Biquad:
    wc = min(max(cutoff_hz / (fs / 2.0), 0.001), 0.999) * math.pi
    k = math.tan(wc / 2.0)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    return Biquad(
        b=[k * k * norm, 2.0 * k * k * norm, k * k * norm],
        a=[1.0, 2.0 * (k * k - 1.0) * norm, (1.0 - math.sqrt(2.0) * k + k * k) * norm],
    )


def make_highpass(cutoff_hz: float, fs: float) -> Biquad:
    wc = min(max(cutoff_hz / (fs / 2.0), 0.001), 0.499) * math.pi
    k = math.tan(wc / 2.0)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    return Biquad(
        b=[norm, -2.0 * norm, norm],
        a=[1.0, 2.0 * (k * k - 1.0) * norm, (1.0 - math.sqrt(2.0) * k + k * k) * norm],
    )


def make_lowpass_4th(cutoff_hz: float, fs: float):
    """4th-order Butterworth as two cascaded biquads - the EMG filter."""
    wc = min(max(cutoff_hz / (fs / 2.0), 0.001), 0.999) * math.pi
    k = math.tan(wc / 2.0)
    k2 = k * k
    out = []
    for sigma in (-math.sin(math.pi / 8.0), -math.sin(3.0 * math.pi / 8.0)):
        norm = 1.0 / (1.0 - 2.0 * k * sigma + k2)
        out.append(Biquad(
            b=[k2 * norm, 2.0 * k2 * norm, k2 * norm],
            a=[1.0, 2.0 * (k2 - 1.0) * norm, (1.0 + 2.0 * k * sigma + k2) * norm],
        ))
    return out[0], out[1]


def _parse_lowpass(value: str) -> Optional[float]:
    """"off"/"" means bypass; anything else is a cutoff in Hz."""
    v = (value or "").strip().lower()
    if v in ("off", "", "none", "0"):
        return None
    try:
        return float(v)
    except ValueError:
        return LOWPASS_CUTOFF


def _parse_ac(value: str) -> float:
    v = (value or "").strip().lower()
    if v == "50":
        return 50.0
    if v == "60":
        return 60.0
    return 0.0


def _parse_emg(value: str) -> float:
    v = (value or "").strip().lower()
    if v in ("off", "", "none", "0"):
        return 0.0
    try:
        return float(v)
    except ValueError:
        return 25.0


# ──────────────────────────────────────────────────────────────────────────────
# The causal path - one instance per lead
# ──────────────────────────────────────────────────────────────────────────────

class CausalFilter:
    """Direct Form II Transposed cascade with persistent state.

    Six slots, in the Kotlin's order: 0 high-pass, 1 mains notch, 2 second
    harmonic, 3 low-pass, 4 and 5 the two EMG biquads. Coefficients are rebuilt
    only when a setting changes, and the state is reset when they are - the
    Kotlin does the same, so a settings change does not fire a transient
    through the trace.
    """

    def __init__(self, fs: float = 500.0):
        self.fs = float(fs)
        self._w1 = [0.0] * 6
        self._w2 = [0.0] * 6
        self._coeffs = None
        self._key = None

    def reset(self) -> None:
        self._w1 = [0.0] * 6
        self._w2 = [0.0] * 6

    def _step(self, slot: int, bq: Biquad, x: float) -> float:
        y = bq.b[0] * x + self._w1[slot]
        self._w1[slot] = bq.b[1] * x - bq.a[1] * y + self._w2[slot]
        self._w2[slot] = bq.b[2] * x - bq.a[2] * y
        return y

    def _ensure(self, high_pass: bool, ac: str, lowpass: str, emg: str):
        key = (bool(high_pass), str(ac), str(lowpass), str(emg), self.fs)
        if self._coeffs is not None and key == self._key:
            return self._coeffs

        ac_freq = _parse_ac(ac)
        lp_hz = _parse_lowpass(lowpass)
        emg_hz = _parse_emg(emg)

        emg1 = emg2 = None
        if emg_hz > 0.0:
            emg1, emg2 = make_lowpass_4th(emg_hz, self.fs)

        coeffs = {
            "hp": make_highpass(HIGHPASS_CUTOFF, self.fs) if high_pass else None,
            "notch": make_notch(ac_freq, self.fs) if ac_freq > 0.0 else None,
            # The Kotlin adds the 100 Hz harmonic for 50 Hz mains only.
            "notch2": (make_notch(100.0, self.fs)
                       if ac_freq == 50.0 and self.fs > 200 else None),
            "lp": make_lowpass(lp_hz, self.fs) if lp_hz is not None else None,
            "emg1": emg1,
            "emg2": emg2,
        }
        self.reset()
        self._coeffs = coeffs
        self._key = key
        return coeffs

    def process(self, samples: Sequence[float], high_pass: bool = True,
                ac: str = "50", lowpass: str = "40", emg: str = "off") -> np.ndarray:
        """Filter a chunk, continuing from where the last chunk ended.

        A NaN sample - which is how a disconnected lead reaches here - is passed
        through untouched and does NOT enter the filter state. Feeding NaN into
        a recursive filter poisons every sample after it, so a two-second lead
        drop would otherwise blank the trace for the rest of the recording.
        """
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            return arr
        c = self._ensure(high_pass, ac, lowpass, emg)

        chain = [(0, c["hp"]), (1, c["notch"]), (2, c["notch2"]),
                 (3, c["lp"]), (4, c["emg1"]), (5, c["emg2"])]
        chain = [(slot, bq) for slot, bq in chain if bq is not None]
        if not chain:
            return arr.copy()

        out = np.empty(arr.size, dtype=float)
        for i in range(arr.size):
            y = arr[i]
            if not np.isfinite(y):
                out[i] = y
                continue
            for slot, bq in chain:
                y = self._step(slot, bq, y)
            out[i] = y
        return out


# ──────────────────────────────────────────────────────────────────────────────
# The zero-phase path - batch, no state
# ──────────────────────────────────────────────────────────────────────────────

def _iir(b, a, x: np.ndarray) -> np.ndarray:
    a0 = a[0]
    bn = [v / a0 for v in b]
    an = [v / a0 for v in a]
    y = np.zeros(x.size, dtype=float)
    for i in range(x.size):
        acc = 0.0
        for j in range(len(bn)):
            if i - j >= 0:
                acc += bn[j] * x[i - j]
        for j in range(1, len(an)):
            if i - j >= 0:
                acc -= an[j] * y[i - j]
        y[i] = acc
    return y


def filtfilt(b, a, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x
    pad = min(x.size - 1, 3 * max(len(b), len(a)))
    if pad > 0:
        padded = np.concatenate([x[:pad][::-1], x, x[-pad:][::-1]])
    else:
        padded = x
    fwd = _iir(b, a, padded)
    back = _iir(b, a, fwd[::-1])[::-1]
    return back[pad:back.size - pad] if pad > 0 else back


def gaussian_lowpass(x: np.ndarray, cutoff_hz: float, fs: float) -> np.ndarray:
    """Gaussian FIR, sigma derived so the -3 dB point lands on the cutoff.

    This is what the Kotlin uses at 25 and 40 Hz instead of a Butterworth, and
    it is why its 25 Hz trace does not ring around a sharp QRS.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    sigma = fs / (2.0 * math.pi * cutoff_hz)
    half = max(1, int(math.ceil(3.0 * sigma)))
    n = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (n / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([x[1:half + 1][::-1], x, x[-half - 1:-1][::-1]])
    if padded.size < x.size + 2 * half:            # very short records
        padded = np.pad(x, half, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[:x.size]


def apply_complete_filter(signal, fs: float = 500.0, high_pass: bool = False,
                          ac: str = "50", lowpass: str = "25",
                          emg: str = "off") -> np.ndarray:
    """The Kotlin applyCompleteFilter: zero-phase, Gaussian at 25 and 40 Hz."""
    out = np.asarray(signal, dtype=float)
    if out.size == 0:
        return out

    if high_pass:
        wc = HIGHPASS_CUTOFF / (fs / 2.0)
        if 0 < wc < 0.5:
            bq = make_highpass(HIGHPASS_CUTOFF, fs)
            # The Kotlin runs the high-pass filtfilt twice - a 4th-order
            # zero-phase response. Kept, so the baselines match.
            out = filtfilt(bq.b, bq.a, filtfilt(bq.b, bq.a, out))

    ac_freq = _parse_ac(ac)
    if ac_freq > 0:
        bq = make_notch(ac_freq, fs)
        out = filtfilt(bq.b, bq.a, out)

    lp = (lowpass or "").strip().lower()
    if lp in ("25", "40"):
        out = gaussian_lowpass(out, float(lp), fs)
    elif lp not in ("off", "", "none", "0"):
        try:
            hz = float(lp)
        except ValueError:
            hz = None
        if hz is not None and hz / (fs / 2.0) < 1.0:
            bq = make_lowpass(hz, fs)
            out = filtfilt(bq.b, bq.a, out)

    emg_hz = _parse_emg(emg)
    if emg_hz > 0:
        safe = min(max(emg_hz, 1.0), fs / 2.0 * 0.9)
        b1, b2 = make_lowpass_4th(safe, fs)
        out = filtfilt(b1.b, b1.a, out)
        out = filtfilt(b2.b, b2.a, out)

    return out
