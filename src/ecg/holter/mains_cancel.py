"""Mains interference (50 / 60 Hz) removed by subtraction, not by a notch.

A narrow notch filter has a long memory: where the interference starts,
stops or switches 50 <-> 60 Hz, the notch rings for ~1 s on both sides (it
runs zero-phase), as big as a QRS. Seen on 2026-11-04_18-22-58 (simulator,
mains switched on, off and 50 <-> 60 Hz 17 times): every rule reading P, QRS
or ST inside that second misread it -- a P wave lost under the ringing broke
a Mobitz I run for 4 s, a QRS end lost under it broke RBBB / LBBB for 1.5 s
six times.

Here the interference itself is modelled and subtracted, so there is no
filter memory to ring:
  1. On one reference lead (II), every 0.1 s (0.5 s window): mains present
     when the 48-52 or 58-62 Hz band is >= 50 ADC RMS and >= 20% of the
     1-150 Hz power. Measured: recordings with no mains <= 36 ADC (95th
     percentile 28); mains stretches median 500, 5th percentile 56.
  2. A change (start, stop, 50 <-> 60 Hz) where that reading changes and
     holds >= 1 s on both sides; placed to the sample: first the step in the
     50 / 60 Hz amplitude (40 ms fits every 2 ms, least absolute deviation
     from each side's median, so a QRS spike doesn't pull it), then each
     side's sinusoid fitted away from the change and the switch moved sample
     by sample for the least residual.
  3. Inside each stretch with mains, on every lead: sin / cos at f and 2f
     fitted in 1 s windows every 0.5 s (anchored to the stretch start, so a
     sample gets the same correction whatever range is read), coefficients
     interpolated between window centres, never across a change, and
     subtracted.
Where no mains is found nothing is touched: read_range returns exactly what
the wrapped reader returns.
"""
import numpy as np

MAINS_RMS_MIN = 50.0
MAINS_SHARE_MIN = 0.2
MAINS_HOLD_SEC = 1.0
# A single step (or a couple) misread as 0 right at a 50<->60 switch --
# the switch instant's 0.5 s window straddles both frequencies, so
# neither's share reaches MAINS_SHARE_MIN there -- stops BOTH sides from
# ever qualifying as a change (each needs MAINS_HOLD_SEC of its own kind
# right up to the point), so the two stretches were never cut apart and
# the shorter one was never subtracted. Seen on 2026-11-05_10-53-47: one
# 0.1 s step between a 15.2 s run of 60 Hz and a 6.3 s run of 50 Hz.
# Filled only when BOTH neighbours already independently hold for
# MAINS_HOLD_SEC (the same bar an ordinary change point must clear) --
# never inside a genuinely noisy/flickering stretch, where neither side
# holds (seen on 2026-11-04_18-22-58: a 0.3 s gap there sits between a
# 0.6 s and a 0.1 s run, both short, so it is correctly left alone and
# that whole messy stretch still waits for its own real 1 s read).
MAINS_GAP_FILL_MAX_STEPS = 3
_STEP_WIN_SEC = 0.5
_STEP_SEC = 0.1
_FIT_WIN_SEC = 1.0
_FIT_HOP_SEC = 0.5
_CHUNK_SEC = 60.0


def _fill_short_mains_gaps(kinds, hold_steps):
    """kinds with a short run of 0 (<= MAINS_GAP_FILL_MAX_STEPS) filled in
    wherever both the run right before it and the run right after it are
    non-zero and already >= hold_steps long on their own -- see
    MAINS_GAP_FILL_MAX_STEPS. The fill is split down the middle (first half
    takes the kind before, second half the kind after); only used so the
    change-point search below has a kind to compare at every step, the
    search itself (and the exact sample it settles on) is unchanged."""
    out = kinds.copy()
    n = len(kinds)
    i = 0
    while i < n:
        j = i
        while j < n and kinds[j] == kinds[i]:
            j += 1
        run_kind, run_start, run_len = int(kinds[i]), i, j - i
        if (run_kind == 0 and run_len <= MAINS_GAP_FILL_MAX_STEPS
                and run_start > 0 and j < n):
            before_kind, before_len = int(kinds[run_start - 1]), 1
            k = run_start - 2
            while k >= 0 and kinds[k] == before_kind:
                before_len += 1
                k -= 1
            after_kind, after_len = int(kinds[j]), 1
            k = j + 1
            while k < n and kinds[k] == after_kind:
                after_len += 1
                k += 1
            if before_kind != 0 and after_kind != 0 and before_len >= hold_steps and after_len >= hold_steps:
                mid = run_start + (run_len + 1) // 2
                out[run_start:mid] = before_kind
                out[mid:j] = after_kind
        i = j
    return out


def _mains_kinds(x, fs):
    """Mains kind (0 / 50 / 60) of each 0.5 s window starting every 0.1 s."""
    win, step = int(_STEP_WIN_SEC * fs), max(1, int(_STEP_SEC * fs))
    if x.size < win:
        return np.zeros(0, dtype=int)
    freqs = np.fft.rfftfreq(win, 1.0 / fs)
    band = (freqs >= 1.0) & (freqs <= 150.0)
    mains = [(50, (freqs >= 48.0) & (freqs <= 52.0)), (60, (freqs >= 58.0) & (freqs <= 62.0))]
    hann = np.hanning(win)
    # Parseval: band RMS of the windowed segment, Hann gain removed.
    scale = 2.0 / (win * win * float(np.mean(hann ** 2)))
    starts = np.arange(0, x.size - win + 1, step)
    segs = x[starts[:, None] + np.arange(win)[None, :]]
    segs = (segs - segs.mean(axis=1, keepdims=True)) * hann
    p = np.abs(np.fft.rfft(segs, axis=1)) ** 2
    total = p[:, band].sum(axis=1)
    kinds = np.zeros(starts.size, dtype=int)
    for hz, sel in reversed(mains):   # 50 Hz wins when both qualify
        s = p[:, sel].sum(axis=1)
        ok = (total > 0) & (s >= MAINS_SHARE_MIN * total) & (np.sqrt(s * scale) >= MAINS_RMS_MIN)
        kinds[ok] = hz
    return kinds


def _design(t, hz):
    """sin / cos at hz and 2*hz (no constant)."""
    return np.stack([np.sin(2 * np.pi * hz * t), np.cos(2 * np.pi * hz * t),
                     np.sin(4 * np.pi * hz * t), np.cos(4 * np.pi * hz * t)], 1)


def _side_model(x, t, hz):
    """Fitted mains of kind hz on (x, t) as a function of time (zero for 0)."""
    if hz == 0:
        return lambda tt: np.zeros(len(tt))
    A = np.concatenate([np.stack([np.ones_like(t), t - t.mean()], 1), _design(t, hz)], 1)
    cf, *_ = np.linalg.lstsq(A, x, rcond=None)
    return lambda tt: _design(np.asarray(tt), hz) @ cf[2:]


def _place_change(x, fs, c, k0, k1):
    """Sample (index into x) of the change near c from kind k0 to k1."""
    win, hop, scan = int(0.04 * fs), max(1, int(0.002 * fs)), int(0.6 * fs)
    lo, hi = max(0, c - scan - win), min(x.size, c + scan + win)
    t = np.arange(x.size) / fs
    pos, amp = [], []
    for s in range(lo, hi - win, hop):
        ts = t[s:s + win]
        A = np.stack([np.ones_like(ts), ts - ts.mean(),
                      np.sin(2 * np.pi * 50 * ts), np.cos(2 * np.pi * 50 * ts),
                      np.sin(2 * np.pi * 60 * ts), np.cos(2 * np.pi * 60 * ts)], 1)
        cf, *_ = np.linalg.lstsq(A, x[s:s + win], rcond=None)
        pos.append(s + win // 2)
        amp.append((np.hypot(cf[2], cf[3]), np.hypot(cf[4], cf[5])))
    if len(pos) < 12:
        return c
    pos, amp = np.asarray(pos), np.asarray(amp)
    best, s0 = None, c
    for m in range(5, len(pos) - 5):
        left, right = amp[:m], amp[m:]
        e = float(np.abs(left - np.median(left, 0)).sum() + np.abs(right - np.median(right, 0)).sum())
        if best is None or e < best:
            best, s0 = e, int(pos[m])
    # To the sample: each side's sinusoid fitted 30-500 ms away from the
    # change, then the switch moved +-15 ms for the least residual.
    g, sp, se = int(0.03 * fs), int(0.5 * fs), int(0.015 * fs)
    if s0 - sp < 0 or s0 + sp > x.size:
        return s0
    left = _side_model(x[s0 - sp:s0 - g], t[s0 - sp:s0 - g], k0)
    right = _side_model(x[s0 + g:s0 + sp], t[s0 + g:s0 + sp], k1)
    zone = np.arange(s0 - g - se, s0 + g + se)
    m_left, m_right = left(t[zone]), right(t[zone])
    trend = np.stack([np.ones(zone.size), zone - zone.mean()], 1)
    best, best_s = None, s0
    for s in range(s0 - se, s0 + se + 1):
        r = x[zone] - np.where(zone < s, m_left, m_right)
        cf, *_ = np.linalg.lstsq(trend, r, rcond=None)
        e = float(np.sum((r - trend @ cf) ** 2))
        if best is None or e < best:
            best, best_s = e, s
    return best_s


class MainsCancelledReader:
    """Wraps a recording reader; read_range returns the samples with mains
    interference subtracted (see the module docstring). Every other
    attribute is the wrapped reader's."""

    def __init__(self, reader, ref_lead='II'):
        self._reader = reader
        self._fs = float(getattr(reader, 'fs', 500) or 500)
        names = list(getattr(reader, 'lead_names', []) or [])
        self._ref = names.index(ref_lead) if ref_lead in names else 0
        self._stretches = None   # [(start_sample, end_sample, hz)], hz 50 / 60 only

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def _read_samples(self, s0, s1):
        """Samples [s0, s1) of every lead (the wrapped reader takes seconds
        and floors them to a sample; half a sample of margin keeps a
        float round-off from landing one sample early)."""
        fs = self._fs
        d = self._reader.read_range((s0 + 0.5) / fs, (s1 + 0.9) / fs)
        if d is None or d.ndim != 2:
            return np.zeros((0, 0))
        return np.asarray(d, dtype=float)

    # -- finding the mains stretches, once per recording ------------------
    def _read_ref(self, s0, s1):
        d = self._read_samples(s0, s1)
        if d.shape[0] <= self._ref:
            return np.zeros(0)
        return d[self._ref]

    def mains_stretches(self):
        if self._stretches is not None:
            return self._stretches
        fs = self._fs
        total = int(round(float(getattr(self._reader, 'duration_sec', 0.0) or 0.0) * fs))
        win, step = int(_STEP_WIN_SEC * fs), max(1, int(_STEP_SEC * fs))
        chunk = int(_CHUNK_SEC * fs) // step * step
        kinds = []
        s = 0
        while s < total:
            core_end = min(total, s + chunk)
            x = self._read_ref(s, min(total, core_end + win))
            k = _mains_kinds(x, fs)
            n_core = (core_end - s + step - 1) // step
            kinds.extend(k[:n_core].tolist())
            s = core_end
        kinds = np.asarray(kinds, dtype=int)
        self._stretches = []
        if not kinds.size or not np.any(kinds):
            return self._stretches
        centers = np.arange(kinds.size) * step + win // 2
        hold = int(round(MAINS_HOLD_SEC / _STEP_SEC))
        kinds = _fill_short_mains_gaps(kinds, hold)
        changes = []
        for j in range(hold, kinds.size - hold + 1):
            if (kinds[j] != kinds[j - 1] and np.all(kinds[j - hold:j] == kinds[j - 1])
                    and np.all(kinds[j:j + hold] == kinds[j])):
                c = int((centers[j - 1] + centers[j]) // 2)
                lo = max(0, c - int(2.0 * fs))
                x = self._read_ref(lo, min(total, c + int(2.0 * fs)))
                changes.append(lo + _place_change(x, fs, c - lo, int(kinds[j - 1]), int(kinds[j])))
        cuts = [0] + changes + [total]
        for a, b in zip(cuts, cuts[1:]):
            inside = kinds[(centers >= a) & (centers < b)]
            hz = int(np.bincount(inside // 10).argmax() * 10) if inside.size else 0
            if hz and b - a >= int(0.1 * fs):
                self._stretches.append((a, b, hz))
        return self._stretches

    # -- reading ------------------------------------------------------------
    def read_range(self, start_sec, end_sec):
        data = self._reader.read_range(start_sec, end_sec)
        stretches = self.mains_stretches()
        if not stretches or data is None or data.ndim != 2 or data.shape[1] == 0:
            return data
        # the sample the wrapped reader started at (it floors start * fs)
        s0 = int(max(0.0, start_sec) * getattr(self._reader, 'fs', self._fs))
        s1 = s0 + data.shape[1]
        out = None
        for a, b, hz in stretches:
            lo, hi = max(a, s0), min(b, s1)
            if hi <= lo:
                continue
            if out is None:
                out = np.asarray(data, dtype=float).copy()
            out[:, lo - s0:hi - s0] -= self._mains_part(a, b, hz, lo, hi, out.shape[0])
        return data if out is None else out

    def _mains_part(self, a, b, hz, lo, hi, n_leads):
        """Fitted mains on every lead for samples [lo, hi) of stretch [a, b)."""
        fs = self._fs
        win = min(b - a, int(_FIT_WIN_SEC * fs))
        hop = int(_FIT_HOP_SEC * fs)
        starts = list(range(a, b - win + 1, hop))
        if starts[-1] != b - win:
            starts.append(b - win)
        starts = np.asarray(starts)
        centers = starts + win / 2.0
        # windows whose centres bracket [lo, hi)
        i0 = max(0, int(np.searchsorted(centers, lo, side='right')) - 1)
        i1 = min(len(starts), int(np.searchsorted(centers, hi, side='left')) + 1)
        use = range(i0, i1)
        r0, r1 = int(starts[i0]), int(starts[i1 - 1] + win)
        raw = self._read_samples(r0, r1)[:n_leads]
        if raw.shape[1] < r1 - r0:
            raw = np.pad(raw, ((0, 0), (0, r1 - r0 - raw.shape[1])), mode='edge')
        coefs, cs = [], []
        for i in use:
            s = int(starts[i])
            t = np.arange(s, s + win) / fs
            A = np.concatenate([np.ones((win, 1)), _design(t, hz)], 1)
            cf, *_ = np.linalg.lstsq(A, raw[:, s - r0:s - r0 + win].T, rcond=None)
            coefs.append(cf[1:])   # (4, n_leads)
            cs.append(centers[i])
        coefs, cs = np.asarray(coefs), np.asarray(cs)
        idx = np.arange(lo, hi)
        basis = _design(idx / fs, hz)            # (n, 4)
        part = np.zeros((n_leads, idx.size))
        for j in range(4):
            for li in range(n_leads):
                part[li] += basis[:, j] * np.interp(idx, cs, coefs[:, j, li])
        return part
