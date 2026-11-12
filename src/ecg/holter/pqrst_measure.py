"""P / QRS / T measurement of recorded beats, beat by beat, for display
(Full Disclosure top bar, report). Not used by arrhythmia detection.

Measured against the Fluke ProSim 8 reference table
(ecg.metrics.reference_intervals) on the "Test_case TT" recording
(2026-10-30_16-03-13), normal sinus 60-160 bpm: PR 168 ms (ref 166), QRS 90
(ref 85-91), QT 242-346 (ref 246-363, 4-17 ms short; tangent T end).

  - Beats: Pan-Tompkins on leads I, V1 and V6; a beat counts when >= 2 of
    them find it within 100 ms. One lead alone drops beats where its QRS
    shrinks (seen: V5 halved on LBBB-shaped beats, 7.5 s with no beat).
  - P / PR: Lead II, 50/60 Hz notch + 0.5 Hz DFT high-pass + 25 Hz low-pass,
    measure_beat(strict_pr=True) at Lead II's own dominant QRS deflection
    (40 ms before to 100 ms after the beat). Without the high-pass a 0.36 Hz
    baseline wander moved PR 96-300 ms beat to beat at 150 bpm. The P / R
    amplitude collision check runs only when the beat's RR is under 333.3 ms
    (>= 180 bpm): on a 150 bpm sinus run with a small R it rejected the
    real P on every beat. Where the strict search finds no PR and RR >= 500
    ms, the non-strict search (most prominent wave) is used, as the bundle
    branch block rule does.
  - QRS: global, across leads I, II, aVL, V1, V2, V5, V6 (0.5-40 Hz +
    notch): from where their combined slope first rises above 10% of its
    peak to where it stays under 10% for 16 ms (20% when a discordant ST-T
    keeps it up) -- the bundle branch block rule's measurement. A single
    lead starts the QRS late (lead II: 32 ms before R, global: ~48 ms) and
    cuts off a slurred terminal S / R'.
  - QT: global QRS onset to the latest T end on leads II, V2, V3, V5; T end
    by the tangent method (see _t_end). The T wave is searched within the
    beat's own RR -- a fixed 600 ms window ran into the next beat's QRS at
    >= 120 bpm (QT 465-637 ms read). Lead II alone read QT 30 ms short.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

FRAME_SEC = 10.0
UNIDENTIFIED = 'Unidentified'
UNIDENTIFIED_COLOR = '#808080'

# A real ECG's lead I and V1 differ (correlation 0.37-0.72 on the test
# recording). The ProSim 8 performance waveforms (triangle / pulse / square
# trains) are the same signal on every lead: 0.96-1.00. Its VFib is too, so
# a frame with a rhythm label of its own (VFib ...) keeps that label.
IDENTICAL_LEADS = ('I', 'V1', 'V5')
IDENTICAL_LEADS_CORR = 0.95
# Among those, a test signal is mostly flat: the share of Lead II samples
# whose step is <= 0.2% of the range -- square wave 0.84, triangle / pulse
# trains 0.64-0.74; the simulator's VFib 0.03-0.21.
TEST_SIGNAL_FLAT_STEP = 0.002
TEST_SIGNAL_FLAT_FRACTION = 0.5
# A test wave that is not flat-stepped (a 10 Hz sine, a 2 Hz triangle) is
# still the same on every lead AND repeats cycle for cycle: the normalised
# autocorrelation of Lead II (_periodicity) is 0.99-1.0 for it. The
# simulator's VFib is the same on every lead too, but repeats nothing: 0.45-
# 0.50 in every VFib window of every recording (113 recordings, 10 s frames).
TEST_SIGNAL_PERIODICITY = 0.97
STATUS_CONTEXT_SEC = 2.0


def _periodicity(x, fs: float) -> float:
    """How exactly `x` repeats itself: the highest normalised (unbiased)
    autocorrelation at a lag past the first drop below 0.5 and up to half the
    signal -- ~1.0 for a periodic wave (sine / triangle / square, any
    frequency with >= 2 cycles), ~0.5 for the simulator's VFib, lower for
    anything else. 0.0 when it never drops below 0.5 or is too short."""
    x = np.asarray(x, dtype=float)
    x = x - float(np.mean(x))
    n = x.size
    if n < int(2 * fs) or not np.any(x):
        return 0.0
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    r = (ac / np.maximum(n - np.arange(n), 1)) / (ac[0] / n)
    below = np.flatnonzero(r < 0.5)
    if below.size == 0:
        return 0.0
    lo, hi = int(below[0]), n // 2
    if hi <= lo + 2:
        return 0.0
    return float(np.max(r[lo:hi]))


def span_status(reader, start_sec: float, end_sec: float) -> str:
    """frame_status of a short stretch (one RR interval), judged on at least
    STATUS_CONTEXT_SEC of signal centred on it."""
    mid = 0.5 * (start_sec + end_sec)
    total = float(getattr(reader, 'duration_sec', end_sec) or end_sec)
    lo = max(0.0, min(start_sec, mid - STATUS_CONTEXT_SEC / 2.0))
    hi = min(total, max(end_sec, mid + STATUS_CONTEXT_SEC / 2.0))
    return frame_status(reader, lo, hi)

# Rhythms with no P wave to measure a PR from (or P waves not conducted).
NO_PR_RHYTHMS = (
    'atrial fibrillation', 'atrial flutter', 'ventricular fibrillation',
    'ventricular tachycardia', 'supraventricular tachycardia', 'third-degree av block', 'asystole',
)
NO_QRS_RHYTHMS = ('ventricular fibrillation', 'asystole')
# summarize(narrow_qrs_non_ecg=True): on a 'non_ecg' / 'test_signal' frame
# with one of these rhythm segments, QRS is shown without the duty-cycle test.
# Matched as the whole label: 'ventricular tachycardia' is also inside
# 'supraventricular tachycardia (svt)'.
VT_QRS_RHYTHMS = ('ventricular tachycardia',)
# A VT's QRS is wide by definition; a narrower value is not taken from this
# rule (seen: 68 ms on a wide, large VT near the ADC rails,
# 2026-11-04_17-31-21 at 300 s) -- the frame goes on to the duty-cycle test
# as before.
VT_QRS_MIN_MS = 120.0
# PR changes beat to beat by definition: shown as a range.
VARIABLE_PR_RHYTHMS = ('second-degree av block (mobitz i)',)

# A value is shown for a frame when >= 75% of its beats have it and their
# spread (interquartile range) is within this many ms.
MIN_BEAT_FRACTION = 0.75
SPREAD_MS = {'pr_ms': 25.0, 'qrs_ms': 20.0, 'qt_ms': 30.0}
# Max QRS/RR duty cycle (each beat's own QRS as a fraction of its own RR)
# for summarize(narrow_qrs_non_ecg=True). A real QRS complex is always a
# small fraction of its own cycle, at any rate; confirmed directly: two
# genuine QRS-shaped (R+S) test pulses measured 5.6-14.6% on every beat,
# a square test wave measured 63% (160 ms QRS at a 250 ms RR) -- well
# clear of this bound either way.
QRS_DUTY_CYCLE_MAX = 0.25

BEAT_LEADS = ('I', 'V1', 'V6')
QRS_LEADS = ('I', 'II', 'aVL', 'V1', 'V2', 'V5', 'V6')
T_LEADS = ('II', 'V2', 'V3', 'V5')
PAD_SEC = 2.0
BEAT_MERGE_SEC = 0.10
COLLISION_RR_MS = 333.3
RELAXED_P_MIN_RR_MS = 500.0
T_LIMB_FRACTION = 0.25   # _t_end: the T wave's descending limb ends where it falls to this fraction of its peak
# _t_end above 250 bpm (next RR under this): T peak searched from 30 ms (not
# 60 ms) after the QRS end, and the T end must come 40 ms before the next R.
FAST_T_MAX_RR_MS = 240.0


def _beat_samples(data: np.ndarray, lead_names: Sequence[str], fs: float) -> List[int]:
    """Beats (sample index into data) found on >= 2 of BEAT_LEADS within
    BEAT_MERGE_SEC; Lead II alone when fewer than two of them exist."""
    from ecg.pan_tompkins import pan_tompkins
    idx = [lead_names.index(n) for n in BEAT_LEADS if n in lead_names]
    if len(idx) < 2:
        idx = [lead_names.index('II')] if 'II' in lead_names else idx[:1]
        need = 1
    else:
        need = 2
    found = sorted((int(p), li) for li in idx for p in pan_tompkins(np.asarray(data[li], dtype=float), fs=fs))
    merge = max(1, int(BEAT_MERGE_SEC * fs))
    beats = []
    k = 0
    while k < len(found):
        group = [found[k]]
        while k + 1 < len(found) and found[k + 1][0] - group[0][0] < merge:
            k += 1
            group.append(found[k])
        k += 1
        if len({li for _, li in group}) >= need:
            beats.append(group[0][0])
    return beats


def _global_qrs(leads: np.ndarray, qrs_idx: Sequence[int], r: int, fs: float):
    """(onset, offset) sample indices of one beat's global QRS, or None."""
    w_lo, w_hi = r - int(0.15 * fs), r + int(0.20 * fs)
    if w_lo < 0 or w_hi > leads.shape[1] or len(qrs_idx) < 3:
        return None
    p_lo, p_hi = int(0.07 * fs), int(0.27 * fs)
    velocity = np.zeros(w_hi - w_lo)
    for i in qrs_idx:
        x = np.asarray(leads[i][w_lo:w_hi], dtype=float)
        if x.size >= 11:
            x = savgol_filter(x, 11, 3)
        size = float(np.ptp(x[p_lo:p_hi])) or 1.0
        velocity += np.abs(np.gradient(x)) / size
    peak = p_lo + int(np.argmax(velocity[p_lo:p_hi]))
    quiet = max(1, int(0.016 * fs))

    def edge(threshold, step):
        run = 0
        rng = range(peak, -1, -1) if step < 0 else range(peak, velocity.size)
        for k in rng:
            run = run + 1 if velocity[k] < threshold else 0
            if run >= quiet:
                return k + quiet - 1 if step < 0 else k - quiet + 1
        return None

    onset = edge(0.10 * float(velocity[peak]), -1)
    offset = edge(0.10 * float(velocity[peak]), +1)
    if offset is None:
        offset = edge(0.20 * float(velocity[peak]), +1)
    if onset is None or offset is None or offset <= onset:
        return None
    return w_lo + onset, w_lo + offset


QUIET_LEAD_ACTIVITY = 0.02
QUIET_WINDOW_MIN_RR_MS = 700.0


def _quiet_qrs_leads(leads: np.ndarray, qrs_idx: Sequence[int], r: int, prev_rr_ms: Optional[float],
                     fs: float) -> List[int]:
    """The QRS-width leads whose signal is quiet before the QRS (median slope
    150-400 ms before the beat <= 2% of the QRS's steepest slope), at least
    three. In Atrial Fibrillation the f waves on some leads run into the QRS
    edges and moved them 50-60 ms (seen: 88 ms QRS read 124-184 ms on a
    third of the beats; 88 ms on all of them without those leads). Only
    when the previous RR is >= 700 ms -- faster, that window holds the
    previous beat's T wave (130 bpm sinus read 78 ms instead of 90, SVT
    230 bpm the same)."""
    lo, hi = r - int(0.40 * fs), r - int(0.15 * fs)
    if prev_rr_ms is None or prev_rr_ms < QUIET_WINDOW_MIN_RR_MS or lo < 0 or r + int(0.10 * fs) > leads.shape[1]:
        return list(qrs_idx)
    activity = []
    for i in qrs_idx:
        steepest = float(np.max(np.abs(np.gradient(leads[i][r - int(0.07 * fs):r + int(0.10 * fs)])))) or 1.0
        activity.append((float(np.median(np.abs(np.gradient(leads[i][lo:hi])))) / steepest, i))
    activity.sort()
    quiet = [i for a, i in activity if a <= QUIET_LEAD_ACTIVITY]
    return quiet if len(quiet) >= 3 else [i for _, i in activity[:3]]


def _t_end(x: np.ndarray, onset: int, offset: int, r: int, next_rr_ms: float, fs: float) -> Optional[int]:
    """T end (sample index) on one lead, or None: the tangent method -- where
    the tangent at the steepest descent of the T wave crosses the PR
    baseline. The T peak is searched from 60 ms after the QRS end to 55% of
    the next RR (at most 450 ms) -- later, the next P wave rising toward the
    window end was taken for the T peak at 150 bpm; its end up to 75% of the
    next RR (at most 600 ms).

    The steepest descent is looked for only on the T wave's own descending
    limb (peak to where it first falls to 25% of the peak, + 20 ms), with
    the slope over +-10 ms. Was: the first point under 5% of the peak, the
    tangent only where that never came. On human recordings the signal after
    the T wave does not come back to the PR baseline -- the PR segment sits
    30-48 ADC below the TP segment (atrial repolarization) and a U wave
    follows -- and settled right at the 5% level (2026-10-28_17-39-47, V2 /
    V3: T peak ~1230 ADC, after it +55..+120), so the end jumped between
    ~400 and the window edge (~600 ms) beat to beat: QT shown on 20 of 153
    frames, mostly ~575 ms; now all 153, 362-394 ms. Simulator (no Ta / U
    wave): QT unchanged within -12..+1 ms (Test_case TT -6..+2 ms). A
    one-sample slope over the whole tail picked noise spikes after the T
    wave (T end ~580-620 ms on single beats), hence the limb and the
    +-10 ms slope.

    Above 250 bpm (next RR < FAST_T_MAX_RR_MS) the T peak (~R+95 ms at 270
    bpm) came before R + QRS end + 60 ms (R+108 ms), so no QT was read from
    260 bpm up. There the search starts 30 ms after the QRS end -- the ST
    segment is gone at that rate -- and a T end at or after the next R
    - 40 ms is dropped. 55% of the RR still keeps the next P out. Slower
    beats are measured as before."""
    fast = next_rr_ms < FAST_T_MAX_RR_MS
    base_lo, base_hi = onset - int(0.04 * fs), onset - int(0.01 * fs)
    lo = offset + int((0.03 if fast else 0.06) * fs)
    peak_hi = min(x.size, r + int(min(0.45, 0.55 * next_rr_ms / 1000.0) * fs))
    end_hi = min(x.size, r + int(min(0.60, 0.75 * next_rr_ms / 1000.0) * fs))
    if base_lo < 0 or peak_hi - lo < 10:
        return None
    base = float(np.median(x[base_lo:base_hi]))
    w = x[lo:peak_hi] - base
    k = int(np.argmax(np.abs(w)))
    if abs(float(w[k])) <= 0:
        return None
    t_peak = lo + k
    seg = (x[t_peak:end_hi] - base) * np.sign(w[k])
    # Only up to the trough after the T wave: at 150 bpm the next P wave
    # starts right after the T, the signal never fell under 5% between them
    # and the end was read at the window edge (QT ~345 ms on half the beats
    # beside ~240 ms on the rest).
    seg = seg[:int(np.argmin(seg)) + 1]
    if seg.size < 5:
        return None
    below = np.where(seg <= T_LIMB_FRACTION * seg[0])[0]
    limb_end = min(seg.size, (int(below[0]) if below.size else seg.size) + int(0.02 * fs) + 1)
    hw = max(1, int(0.01 * fs))
    steepest, slope = None, 0.0
    for i in range(hw, limb_end - hw):
        s = (seg[i + hw] - seg[i - hw]) / (2.0 * hw)
        if s < slope:
            steepest, slope = i, s
    if steepest is None:
        return None
    level = float(np.mean(seg[steepest - hw:steepest + hw + 1]))
    cross = steepest + level / -slope
    if t_peak + cross > end_hi + 0.05 * fs:
        return None
    end = t_peak + int(round(cross))
    if fast and end >= r + int((next_rr_ms - 40.0) / 1000.0 * fs):
        return None
    return end


def measure_beats(reader, start_sec: float, end_sec: float) -> List[Dict[str, Any]]:
    """Every beat whose time falls in [start_sec, end_sec), measured as in the
    module docstring. Each beat: 't' (s), 'rr_ms' (from the previous beat,
    or None), 'pr_ms', 'qrs_ms', 'qt_ms' (None where not measurable),
    'p_present'."""
    from ecg.arrhythmia_detector import measure_beat
    from ecg.enhanced_filter import make_notch
    from ecg.ecg_filters import apply_dft_filter
    from .analysis_pipeline import SignalPreprocessor, HolterConfig

    lead_names = list(getattr(reader, 'lead_names', []) or [])
    fs = float(getattr(reader, 'fs', 500) or 500)
    total = float(getattr(reader, 'duration_sec', end_sec) or end_sec)
    if 'II' not in lead_names:
        return []
    p0 = max(0.0, start_sec - PAD_SEC)
    p1 = min(total, end_sec + PAD_SEC)
    data = reader.read_range(p0, p1)
    if data is None or data.shape[1] < int(fs):
        return []
    data = np.asarray(data, dtype=float)

    lead_ii = data[lead_names.index('II')]
    for notch in (make_notch(50.0, fs), make_notch(60.0, fs)):
        lead_ii = filtfilt(notch.b, notch.a, lead_ii)
    lead_ii = np.asarray(apply_dft_filter(lead_ii, fs, "0.5"), dtype=float)
    b, a = butter(2, 25.0 / (fs / 2.0), btype='lowpass')
    lead_ii = filtfilt(b, a, lead_ii)

    # The preprocessor notches 50 Hz only; 60 Hz is notched here too, for
    # these measurements alone. Seen: 60 Hz interference kept the QRS slope
    # above its 10% end level -- QRS read 114-154 ms (true 90) or not at
    # all, and QT with it. The detector's own use of the preprocessor is
    # unchanged.
    preproc = SignalPreprocessor(fs=fs, config=HolterConfig.from_clinical_config())
    notch_60 = make_notch(60.0, fs)
    leads = np.vstack([filtfilt(notch_60.b, notch_60.a, np.asarray(preproc.process(lead), dtype=float))
                       for lead in data])
    qrs_idx = [lead_names.index(n) for n in QRS_LEADS if n in lead_names]
    t_idx = [lead_names.index(n) for n in T_LEADS if n in lead_names]

    beats = _beat_samples(data, lead_names, fs)
    out = []
    for k, r in enumerate(beats):
        t = p0 + r / fs
        if not (start_sec <= t < end_sec):
            continue
        prev_rr = (r - beats[k - 1]) * 1000.0 / fs if k > 0 else None
        next_rr = (beats[k + 1] - r) * 1000.0 / fs if k + 1 < len(beats) else None
        beat = {'t': t, 'rr_ms': prev_rr, 'pr_ms': None, 'qrs_ms': None, 'qt_ms': None, 'p_present': False}
        m = None

        a_lo, a_hi = max(0, r - int(0.04 * fs)), min(lead_ii.size, r + int(0.10 * fs))
        if a_hi > a_lo:
            local = lead_ii[max(0, r - int(fs)):r + int(fs)]
            anchor = a_lo + int(np.argmax(np.abs(lead_ii[a_lo:a_hi] - np.median(local))))
            collision = prev_rr is None or prev_rr < COLLISION_RR_MS
            m = measure_beat(lead_ii, anchor, fs, strict_pr=True, prev_rr_ms=prev_rr,
                             p_collision_check=collision)
            if m and m.get('pr_ms') is None and prev_rr is not None and prev_rr >= RELAXED_P_MIN_RR_MS:
                relaxed = measure_beat(lead_ii, anchor, fs, strict_pr=False, prev_rr_ms=prev_rr,
                                       p_collision_check=collision)
                if relaxed and relaxed.get('pr_ms') is not None:
                    m = relaxed
            if m and m.get('pr_ms') is not None:
                beat['pr_ms'] = float(m['pr_ms'])
                beat['p_present'] = True

        bounds = _global_qrs(leads, _quiet_qrs_leads(leads, qrs_idx, r, prev_rr, fs), r, fs)
        # Lead II's own QRS onset can't come before the earliest onset over
        # all leads: where it does, its search ran into the PR segment (seen:
        # 150 bpm, lead II R shrunk to ~200-390, onset 52 ms before the beat
        # vs 42 ms globally, PR 106 ms instead of ~116 -- ref 125). PR is then
        # measured to the global onset; elsewhere lead II's own onset stays
        # (PR 168 at 60 bpm, ref 166).
        if bounds is not None and beat['pr_ms'] is not None and m and m.get('q_onset') is not None \
                and int(m['q_onset']) < bounds[0]:
            pr = (bounds[0] - int(m['p_onset'])) * 1000.0 / fs
            beat['pr_ms'] = pr if 60.0 <= pr <= 400.0 else None
            beat['p_present'] = beat['pr_ms'] is not None
        if bounds is not None:
            onset, offset = bounds
            beat['qrs_ms'] = (offset - onset) * 1000.0 / fs
            if next_rr is not None:
                ends = [e for e in (_t_end(leads[i], onset, offset, r, next_rr, fs) for i in t_idx) if e is not None]
                if ends:
                    beat['qt_ms'] = (max(ends) - onset) * 1000.0 / fs
        out.append(beat)
    return out


def frame_status(reader, start_sec: float, end_sec: float) -> str:
    """'test_signal' (the same waveform on every lead and Lead II flat over
    >= TEST_SIGNAL_FLAT_FRACTION of the samples -- a square / triangle /
    pulse test train -- or repeating cycle for cycle, _periodicity >=
    TEST_SIGNAL_PERIODICITY -- a sine / triangle), 'non_ecg' (the same
    waveform on every lead otherwise -- e.g. the simulator's VFib), 'noise' (Lead II signal quality under
    sqi_min) or 'ok'. At least 2 s of signal; shorter reads 'ok'."""
    from .analysis_pipeline import SignalPreprocessor, SignalQuality, HolterConfig
    lead_names = list(getattr(reader, 'lead_names', []) or [])
    fs = float(getattr(reader, 'fs', 500) or 500)
    data = reader.read_range(start_sec, end_sec)
    if data is None or data.shape[1] < int(2 * fs):
        return 'ok'
    cfg = HolterConfig.from_clinical_config()
    idx = [lead_names.index(n) for n in IDENTICAL_LEADS if n in lead_names]
    if len(idx) == len(IDENTICAL_LEADS):
        preproc = SignalPreprocessor(fs=fs, config=cfg)
        x = np.vstack([np.asarray(preproc.process(np.asarray(data[i], dtype=float)), dtype=float) for i in idx])
        if np.all(np.std(x, axis=1) > 0):
            corr = np.corrcoef(x)
            if np.all(corr[np.triu_indices(len(idx), 1)] >= IDENTICAL_LEADS_CORR):
                if 'II' in lead_names:
                    lead_ii = np.asarray(data[lead_names.index('II')], dtype=float)
                    step = np.abs(np.diff(lead_ii))
                    flat = float(np.mean(step <= TEST_SIGNAL_FLAT_STEP * float(np.ptp(lead_ii))))
                    if flat >= TEST_SIGNAL_FLAT_FRACTION:
                        return 'test_signal'
                    if _periodicity(lead_ii, fs) >= TEST_SIGNAL_PERIODICITY:
                        return 'test_signal'
                return 'non_ecg'
    if 'II' in lead_names:
        scores, _ = SignalQuality(fs=fs, config=cfg).lead_scores(
            np.asarray(data[lead_names.index('II')], dtype=float)[None, :])
        if float(scores[0]) < cfg.sqi_min:
            return 'noise'
    return 'ok'


def _robust_value(beats: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(b[key]) for b in beats if b.get(key) is not None]
    if not beats or len(values) < max(2, MIN_BEAT_FRACTION * len(beats)):
        return None
    q1, q3 = np.percentile(values, [25, 75])
    if q3 - q1 > SPREAD_MS[key]:
        return None
    return float(np.median(values))


def summarize(beats: List[Dict[str, Any]], rhythms: Iterable[str] = (), status: str = 'ok',
              narrow_qrs_non_ecg: bool = False) -> Dict[str, Any]:
    """PR / QRS / QT of one frame for display: {'pr_ms', 'qrs_ms', 'qt_ms'}
    (None = not reliably measurable, shown as '--') plus 'pr_range' (min,
    max) for a rhythm whose PR varies beat to beat. `rhythms`: labels of
    the rhythm segments in the frame. See the rules at the top of the
    module.

    narrow_qrs_non_ecg: on a 'non_ecg' or 'test_signal' frame (the same
    waveform on every lead, the second additionally flat-stepped), still
    give QRS -- only QRS -- when every measured beat's own QRS is a small
    fraction of that beat's own RR (QRS_DUTY_CYCLE_MAX), i.e. it looks
    like a real QRS complex regardless of its exact width or the
    recording's rate, and the usual _robust_value checks still pass.
    'noise' status is not included -- that is signal actually too
    degraded to trust, not a same-everywhere test waveform.
    Seen: two genuine QRS-shaped (R+S) simulator pulses, 56-64 ms at a
    1000 ms RR and 144-146 ms at a 1000 ms RR, both shown as '--' before
    this; a test square wave (160 ms QRS at a 250 ms RR -- 64% duty
    cycle) and a triangle wave (QRS never resolved on any beat) both
    correctly stay '--' with this on. Off by default: only the on-screen
    label asks for it.
    In a Ventricular Tachycardia segment (VT_QRS_RHYTHMS) the duty-cycle
    test is skipped -- QRS only, with the _robust_value checks: the
    simulator's VT is the same waveform on I / V1 / V5 (0.99) and repeats
    exactly, so its frames are 'test_signal' / 'non_ecg', and its wide QRS
    is 67% of the RR (250 ms at 374 ms), as much as a square wave's -- it
    read '--' on every VT frame (2026-11-12_13-54-02). The VT detector has
    already judged those beats a rhythm; a test wave gets no VT segment.
    A value under VT_QRS_MIN_MS (a VT's QRS is wide) is not shown by this
    rule; the frame then goes through the duty-cycle test as before."""
    out = {'pr_ms': None, 'qrs_ms': None, 'qt_ms': None, 'pr_range': None}
    labels = [str(r).lower() for r in rhythms]
    has = lambda names: any(any(n in lbl for n in names) for lbl in labels)
    if narrow_qrs_non_ecg and status in ('non_ecg', 'test_signal') and beats:
        if has(NO_QRS_RHYTHMS):
            return out
        if any(lbl in VT_QRS_RHYTHMS for lbl in labels):
            qrs = _robust_value(beats, 'qrs_ms')
            if qrs is not None and qrs >= VT_QRS_MIN_MS:
                out['qrs_ms'] = qrs
                return out
        duty = [
            float(b['qrs_ms']) / float(b['rr_ms'])
            for b in beats
            if b.get('qrs_ms') is not None and b.get('rr_ms')
        ]
        if duty and all(d <= QRS_DUTY_CYCLE_MAX for d in duty):
            out['qrs_ms'] = _robust_value(beats, 'qrs_ms')
        return out
    if status != 'ok' or not beats:
        return out
    if not has(NO_QRS_RHYTHMS):
        out['qrs_ms'] = _robust_value(beats, 'qrs_ms')
        out['qt_ms'] = _robust_value(beats, 'qt_ms')
    if has(VARIABLE_PR_RHYTHMS):
        prs = [float(b['pr_ms']) for b in beats if b.get('pr_ms') is not None]
        if len(prs) >= 2:
            out['pr_range'] = (min(prs), max(prs))
    elif not has(NO_PR_RHYTHMS):
        out['pr_ms'] = _robust_value(beats, 'pr_ms')
    return out


_CACHE_FILENAME = 'pqrst_cache.json'
_frames_memory: Dict[str, List[Dict[str, Any]]] = {}


def _frames_cache_key(path: str) -> List[Any]:
    """The recording file (size, modification time), the code the frames
    depend on (this module and everything of the ecg package it imports --
    cache_fingerprint.code_fingerprint follows the imports itself; the
    hand-bumped _CACHE_VERSION it replaces missed every change to a
    dependency) and the clinical configuration the run loaded. The code part
    is '' when the sources cannot be read (a frozen build): the disk cache
    is then not used."""
    import os
    from .cache_fingerprint import clinical_config_fingerprint, code_fingerprint
    stat = os.stat(path)
    code = code_fingerprint([os.path.abspath(__file__)])
    return [os.path.abspath(path), stat.st_size, stat.st_mtime_ns, code, clinical_config_fingerprint()]


def recording_frames(path: str) -> List[Dict[str, Any]]:
    """Every FRAME_SEC frame of the recording at `path`: {'start', 'end',
    'status', 'beats': [[t, rr_ms, pr_ms, qrs_ms, qt_ms], ...]}. Cached in
    memory and next to the recording (pqrst_cache.json), keyed by the file,
    the code and the clinical configuration (_frames_cache_key)."""
    import json
    import os
    import threading
    from .file_format import ECGHFileReader
    key = _frames_cache_key(path)
    mem_key = json.dumps(key)
    if mem_key in _frames_memory:
        return _frames_memory[mem_key]
    disk = bool(key[3])
    cache_path = os.path.join(os.path.dirname(os.path.abspath(path)), _CACHE_FILENAME)
    if disk:
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            if isinstance(saved, dict) and saved.get('key') == key and isinstance(saved.get('frames'), list):
                _frames_memory[mem_key] = saved['frames']
                return saved['frames']
        except (OSError, ValueError):
            pass
    reader = ECGHFileReader(path)
    try:
        total = float(reader.duration_sec)
        frames = []
        t = 0.0
        while t < total:
            chunk_end = min(t + 6 * FRAME_SEC, total)
            beats = measure_beats(reader, t, chunk_end)
            f0 = t
            while f0 < chunk_end:
                f1 = min(f0 + FRAME_SEC, chunk_end)
                frames.append({
                    'start': f0,
                    'end': f1,
                    'status': frame_status(reader, f0, f1),
                    'beats': [[b['t'], b['rr_ms'], b['pr_ms'], b['qrs_ms'], b['qt_ms']]
                              for b in beats if f0 <= b['t'] < f1],
                })
                f0 = f1
            t = chunk_end
    finally:
        reader.close()
    _frames_memory[mem_key] = frames
    if not disk:
        return frames
    # a temp name of its own: the report and another process can save at once
    tmp = f"{cache_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'key': key, 'frames': frames}, f)
        os.replace(tmp, cache_path)
    except OSError as e:
        print(f"[PQRST] Could not save cache: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass
    return frames


def interval_stats(path: str, rhythm_segments: Iterable[Dict[str, Any]] = ()) -> Dict[str, List[float]]:
    """{'pr_ms', 'qrs_ms', 'qt_ms', 'qtc_ms'}: one value per frame where it
    can be read (summarize), over the whole recording. QTc is Bazett's, from
    the frame's QT and median RR. `rhythm_segments`: the auto-marked
    segments ({'label', 'start_sec', 'end_sec'})."""
    segments = [(float(s.get('start_sec', 0.0) or 0.0), float(s.get('end_sec', 0.0) or 0.0),
                 str(s.get('label', '') or '')) for s in rhythm_segments or []]
    stats = {'pr_ms': [], 'qrs_ms': [], 'qt_ms': [], 'qtc_ms': []}
    for frame in recording_frames(path):
        beats = [{'t': b[0], 'rr_ms': b[1], 'pr_ms': b[2], 'qrs_ms': b[3], 'qt_ms': b[4]} for b in frame['beats']]
        rhythms = [lbl for s, e, lbl in segments if s < frame['end'] and e > frame['start']]
        values = summarize(beats, rhythms, frame['status'])
        for key in ('pr_ms', 'qrs_ms', 'qt_ms'):
            if values[key] is not None:
                stats[key].append(values[key])
        rrs = [b['rr_ms'] for b in beats if b['rr_ms']]
        if values['qt_ms'] is not None and rrs:
            stats['qtc_ms'].append(values['qt_ms'] / np.sqrt(float(np.median(rrs)) / 1000.0))
    return stats
