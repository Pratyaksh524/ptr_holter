"""Lead I in place of Lead II, only where Lead II is not usable for P waves.

TODO(conduction-lead-fallback): the Lead II -> Lead I switch is a separate,
removable condition on purpose. It is NOT a general lead chooser: Lead II
stays everywhere it is fine; Lead I stands in for it only in the 20 s blocks
where Lead II's P wave is as tall as its QRS (seen: ProSim conduction
recording 2026-11-13_10-56-05, 10:58:05-10:59:30: Lead II P 364 ADC against
QRS 472 ADC = 77%, Lead I P 23% of QRS). On such a lead
 - the beat detector counts the P wave as a beat (Lead II gave 27 beats in a
   10 s window where V4 / V5 / Lead I gave 13), so VFib was called on regular
   First-degree AV block / Mobitz / RBBB, and the AV block rules found no P
   waves outside "QRS zones";
 - the P detector's T-wave collision test (a wave before the QRS above 55% of
   R is "the previous T wave, not a P", arrhythmia_detector._detect_p_wave)
   rejects the real P (RBBB P found on 6 of 14 beats, 11 needed).
Revisit once the beat detector / P detector handle a tall Lead II P wave
themselves; then this file and its one call site (the first lines of
auto_segment_arrthymia_detection._detect_arrhythmias_uncached) can go.

How: LeadIIFallbackReader wraps the reader the segment detector is given.
Wherever read_range() covers a bad span, the Lead II row is replaced by the
Lead I row (shifted by the two leads' median difference over that span, so
there is no DC step), so everything the detector does with "Lead II" -- the
beat detection leads, the P / PR measurement, the AV block and bundle branch
block rules -- sees Lead I there and Lead II everywhere else. Nothing else
in the detector knows about it.

Bad span = run of BLOCK_SEC blocks (on a fixed grid from 0 s, so a window
worker that only scans its own range finds the same blocks) in which, on the
beats whose previous RR is >= MIN_RR_MS (so the previous beat's T wave is
not in the zone), the share where the P detector's own T-collision test fires
(a wave between R-333 ms and R-70 ms above 55% of |R - baseline|) is
>= II_SHARE_MIN on Lead II and <= I_SHARE_MAX on Lead I. Beats come from
Pan-Tompkins on the strongest of V5 / V6 / V4 / V1 / I of the block, never
Lead II; a block with fewer than MIN_BEATS beats on either lead stays Lead II
(fast rhythms leave none), and so does every block where Lead I is no better.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

BLOCK_SEC = 20.0
# The T-collision test of _detect_p_wave.
COLLISION_RATIO = 0.55
MIN_RR_MS = 700.0
MIN_BEATS = 4
II_SHARE_MIN = 0.5
I_SHARE_MAX = 0.2
_BEAT_LEADS = ('V5', 'V6', 'V4', 'V1', 'I')


def _collision_counts(x: np.ndarray, fs: float, r_idx: np.ndarray, lo: int, hi: int):
    """(eligible beats, beats where the T-collision test fires) of one lead,
    for the beats with lo <= R < hi (the beat before gives the RR)."""
    import ecg.arrhythmia_detector as ad
    from ecg.ecg_filters import apply_dft_filter
    from .two_stage_median_baseline_filter import remove_baseline_wander_r
    ms = lambda v: int(round(v * fs / 1000.0))
    x = np.asarray(apply_dft_filter(remove_baseline_wander_r(x, fs, r_idx), fs, "0.5"), dtype=float)
    n = c = 0
    for k in range(1, len(r_idx)):
        r = int(r_idx[k])
        if r < lo or r >= hi or r - ms(400.0) < 0 or r + ms(100.0) > x.size:
            continue
        if (r - int(r_idx[k - 1])) * 1000.0 / fs < MIN_RR_MS:
            continue
        base = ad._qrs_bounds(x, r, fs)[2]
        r_amp = abs(float(x[r]) - base)
        zone = x[r - ms(333.3):r - ms(70.0)]
        if r_amp > 1e-4 and zone.size:
            n += 1
            c += float(np.max(np.abs(zone - base))) > COLLISION_RATIO * r_amp
    return n, c


def lead_ii_bad_spans(reader, t_from: float = 0.0, t_to: Optional[float] = None) -> List[Tuple[float, float]]:
    """[(start_sec, end_sec)] of the stretches in [t_from, t_to] (whole
    recording by default) where Lead II is unusable for P waves (see the
    module docstring). Adjacent bad blocks are one span."""
    from ecg.pan_tompkins import pan_tompkins
    names = list(getattr(reader, 'lead_names', None) or [])
    if 'I' not in names or 'II' not in names:
        return []
    fs = float(getattr(reader, 'fs', 500) or 500)
    total = float(getattr(reader, 'duration_sec', 0.0) or 0.0)
    t_to = total if t_to is None else min(total, float(t_to))
    i_I, i_II = names.index('I'), names.index('II')
    cand = [names.index(n) for n in _BEAT_LEADS if n in names]
    blocks = []
    t = float(np.floor(max(0.0, float(t_from)) / BLOCK_SEC) * BLOCK_SEC)
    while t < t_to - 5.0:
        a, b = max(0.0, t - 1.0), min(total, t + BLOCK_SEC + 1.0)
        d = reader.read_range(a, b)
        if d is None or d.shape[1] < int(5 * fs) or d.shape[0] <= max(i_I, i_II):
            t += BLOCK_SEC
            continue
        li = max((i for i in cand if i < d.shape[0]), key=lambda i: float(np.ptp(d[i])))
        pk = np.asarray(sorted(pan_tompkins(np.asarray(d[li], dtype=float), fs=fs)), dtype=int)
        lo, hi = int((t - a) * fs), int((min(t + BLOCK_SEC, total) - a) * fs)
        n_ii, c_ii = _collision_counts(np.asarray(d[i_II], dtype=float), fs, pk, lo, hi)
        n_i, c_i = _collision_counts(np.asarray(d[i_I], dtype=float), fs, pk, lo, hi)
        if n_ii >= MIN_BEATS and n_i >= MIN_BEATS and c_ii / n_ii >= II_SHARE_MIN and c_i / n_i <= I_SHARE_MAX:
            blocks.append((float(t), float(min(t + BLOCK_SEC, total))))
        t += BLOCK_SEC
    spans: List[Tuple[float, float]] = []
    for s, e in blocks:
        if spans and abs(spans[-1][1] - s) < 1e-6:
            spans[-1] = (spans[-1][0], e)
        else:
            spans.append((s, e))
    return spans


class LeadIIFallbackReader:
    """The segment detector's reader with Lead I's samples in the Lead II row
    over the bad spans (see the module docstring). Everything but read_range()
    is the wrapped reader's own."""

    def __init__(self, reader, spans: Sequence[Tuple[float, float]]):
        self._reader = reader
        self.fallback_spans = [(float(s), float(e)) for s, e in spans]
        names = list(getattr(reader, 'lead_names', None) or [])
        self._i_I, self._i_II = names.index('I'), names.index('II')
        self._fs = float(getattr(reader, 'fs', 500) or 500)
        # Lead I -> Lead II DC shift of each span, from its first block.
        self._offset = []
        for s, e in self.fallback_spans:
            d = reader.read_range(s, min(e, s + BLOCK_SEC))
            self._offset.append(float(np.median(d[self._i_II]) - np.median(d[self._i_I]))
                                if d is not None and d.shape[1] else 0.0)

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def __enter__(self):
        self._reader.__enter__()
        return self

    def __exit__(self, *exc):
        return self._reader.__exit__(*exc)

    def read_range(self, start_sec, end_sec, *args, **kwargs):
        data = self._reader.read_range(start_sec, end_sec, *args, **kwargs)
        if data is None or not self.fallback_spans or data.ndim != 2 or data.shape[1] == 0:
            return data
        out = None
        for (s, e), off in zip(self.fallback_spans, self._offset):
            lo, hi = max(float(start_sec), s), min(float(end_sec), e)
            if hi <= lo:
                continue
            i0 = max(0, int(round((lo - float(start_sec)) * self._fs)))
            i1 = min(data.shape[1], int(round((hi - float(start_sec)) * self._fs)))
            if i1 <= i0:
                continue
            if out is None:
                out = np.array(data, copy=True)
            out[self._i_II, i0:i1] = (np.asarray(out[self._i_I, i0:i1], dtype=float) + off).astype(out.dtype)
        return data if out is None else out


def wrap_reader(reader, t_from: float = 0.0, t_to: Optional[float] = None):
    """`reader` itself when Lead II is fine everywhere in [t_from, t_to] (or
    the check cannot run), else a LeadIIFallbackReader."""
    try:
        spans = lead_ii_bad_spans(reader, t_from, t_to)
        return LeadIIFallbackReader(reader, spans) if spans else reader
    except Exception:
        return reader
