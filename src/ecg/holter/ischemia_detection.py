"""
Ischemia findings for Holter recordings: ST Elevation / ST Depression and T
Inversion. These are VALUES shown in the Overview / Replay tables and the
report -- not auto arrhythmia marks.

Independent of the auto arrhythmia detection (auto marking): it is not called
from it and does not read its rhythm segments. It reads the recording itself
(ECGHFileReader.read_range, the raw 12 leads the Full Disclosure view shows)
and filters every lead exactly as the Full Disclosure strip does
(_fd_filter): 0.5 Hz DFT high-pass (zero-phase, 2 s mirror padding), 3-point
median, Gaussian low-pass 35 Hz twice (zero-phase).

ST Elevation / ST Depression (detect_st_episodes), frame by frame:
  1. The recording is read in 10 s frames (1.5 s padding each side).
  2. Skipped frames -- where the ST segment can't be read -- decided from the
     frame's own beats, not from any rhythm label:
       - beats that don't share one shape (median correlation < 0.9 with the
         frame's template): ectopics, noise, Ventricular Fibrillation;
       - median beat with QRS >= 120 ms: bundle branch block, ventricular
         rhythm, paced beats (secondary ST-T changes; the J point sits in
         the slurred end of the QRS);
       - irregular RR (median beat-to-beat RR change > 12 % of the RR):
         Atrial Fibrillation, frequent ectopics. In Atrial Fibrillation the
         PR segment holds f waves, so it is not a baseline.
  3. R-peaks found on the reference lead; beats aligned to each other. The
     median beat is used only to locate the QRS onset and J point (combined
     slope of all leads, as the bundle branch block width).
  4. Per beat and per lead: baseline = PR segment (40-10 ms before QRS
     onset); ST at J+40 ms (J+30 ms above 120 bpm) and at J+20 ms (slope).
     The value of a lead in the frame is the one that occurs most often
     among its beats (0.1 mm bins, mean of the fullest bin). A lead whose
     beats spread more than 1 mm (interquartile range) is noisy in that
     frame and is not judged there.
     All in the millimetres of the Full Disclosure graph paper
     (_SCREEN_ADC_PER_MV).
  5. Criteria per frame, limb leads and chest leads judged separately:
       limb group  (I, II, III, aVL, aVF), any 2 leads
       chest group (V1-V6), any 2 leads
       ST Elevation : limb >= 1 mm, chest >= 2 mm.
       ST Depression: limb and chest <= -1 mm, horizontal or downsloping
                      (not rising more than 0.5 mm from J+20 to the ST
                      point).
     aVR is not judged. A frame with both is ST Elevation; its depression
     is shown with the event, not as an ST Depression event:
       Reciprocal: leads facing the wall opposite an elevated wall
         (inferior II/III/aVF <-> I/aVL, anterior V1-V4 -> II/III/aVF,
         lateral V5-V6 -> III/aVF), one lead is enough;
       Reciprocal (posterior): V1-V3 with inferior / lateral elevation;
       Additional depression: other leads, any 2 of one group.
  6. Events: consecutive frames of one label, a gap of up to one frame
     (10 s) joined, kept when the qualifying frames add up to >= 30 s.
     Per event and per lead that qualified: the average over all frames of
     the event, and the largest value in the frames in which it qualified
     (max elevation / deepest depression).
  7. A recording shorter than 30 s can't hold an event; the Overview says
     so instead of "None".
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.signal import savgol_filter

ST_ELEVATION = 'ST Elevation'
ST_DEPRESSION = 'ST Depression'
T_INVERSION = 'T Inversion'
ST_COLORS = {
    ST_ELEVATION: '#00E5A0',   # mint green
    ST_DEPRESSION: '#00BFFF',  # deep sky blue
    T_INVERSION: '#FFB020',    # amber
}
# every label this module produces (they are ischemia findings, not rhythms)
ST_LABELS = (ST_ELEVATION, ST_DEPRESSION, T_INVERSION)

# Same ADC scale the Full Disclosure measuring ruler uses (theme.ADC_TO_MV).
try:
    from .theme import ADC_TO_MV as _ADC_TO_MV
except Exception:  # pragma: no cover - theme needs Qt in some environments
    _ADC_TO_MV = 1.0 / 200.0

_CHUNK_SEC = 10.0
_CHUNK_PAD_SEC = 1.5
_MIN_CHUNK_SEC = 5.0
_MIN_BEATS = 4
_MIN_EPISODE_SEC = 60.0          # T Inversion episodes
_MERGE_GAP_SEC = 60.0            # T Inversion episodes
_MODE_BIN_MV = 0.05              # T Inversion

_MM_PER_MV = 10.0

# ST is measured in the millimetres of the Full Disclosure graph paper. The
# strip draws 4096 ADC over its plot height (row height - 8 px margin,
# holter_widgets.ECGStripCanvas), and the graph paper draws 1 mm from the
# monitor's physical DPI. On a 2560x1440 27" monitor at 100 % scale: row
# 163 px -> plot 155 px = 26.4 ADC per px, 1 mm = 4.29 px (~109 DPI) ->
# ~113 ADC per graph-paper mm. Checked against a recording on that screen
# (V2 / V3 / V5 / aVF peak-to-peak: 111-116 ADC per mm). A fixed number, so
# the values don't change with the window size.
_SCREEN_ADC_PER_MV = 1133.0
_ADC_TO_MM = _MM_PER_MV / _SCREEN_ADC_PER_MV

_ST_MODE_BIN_MM = 0.1            # beats within 0.1 mm of each other count as the same value
_ELEVATION_MM = {'limb': 1.0, 'chest': 2.0}
_DEPRESSION_MM = -1.0            # limb and chest leads
_UPSLOPE_LIMIT_MM = 0.5          # depression rising more than this J+20 -> ST point is upsloping
_LEAD_SPREAD_MM = 1.0            # beats of one lead spread wider than this: lead is noisy
_IRREGULAR_RR = 0.12             # median |RR change| / median RR above this: irregular rhythm
_MIN_LEADS_IN_GROUP = 2
_ST_EVENT_GAP_SEC = 10.0         # one frame
_ST_MIN_EVENT_SEC = 30.0

# Limb and chest leads are measured and judged as two separate arrays
# (limb leads: frontal plane; chest leads: horizontal plane).
LIMB_LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
CHEST_LEADS = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')

# Depression shown with an ST Elevation event (_depression_kinds), sorted by
# the wall each lead faces. A wall is elevated when at least _MIN_WALL_LEADS
# of its leads are among the event's elevated leads (contiguous leads).
_WALL_LEADS = {
    'inferior': ('II', 'III', 'aVF'),
    'high lateral': ('I', 'aVL'),
    'anterior': ('V1', 'V2', 'V3', 'V4'),
    'lateral': ('V5', 'V6'),
}
_MIN_WALL_LEADS = 2
# Leads facing the wall opposite an elevated wall: their depression is the
# mirror image of the same injury (reciprocal).
_RECIPROCAL_LEADS = {
    'inferior': ('I', 'aVL'),
    'high lateral': ('II', 'III', 'aVF'),
    'anterior': ('II', 'III', 'aVF'),
    'lateral': ('III', 'aVF'),
}
# V1-V3 face the front of the heart: with inferior / lateral elevation their
# depression is the mirror image of the posterior wall.
_POSTERIOR_LEADS = ('V1', 'V2', 'V3')
_POSTERIOR_WITH = ('inferior', 'lateral')
DEP_RECIPROCAL = 'Reciprocal'
DEP_POSTERIOR = 'Reciprocal (posterior)'
DEP_ADDITIONAL = 'Additional depression'
DEP_KINDS = (DEP_RECIPROCAL, DEP_POSTERIOR, DEP_ADDITIONAL)

_MAX_QRS_MS = 120.0

# T inversion (detect_t_inversion_episodes)
_T_INVERSION_MV = -0.10          # T wave 1 mm (0.1 mV) or more below the PR baseline
_T_MAX_HR = 120.0                # above this the T wave runs into the next P wave
_T_START_AFTER_J_SEC = 0.06      # T search window starts here after the J point
_T_MIN_PROMINENCE_MV = 0.02      # a real T wave is a turning point, not a slope
_T_NOT_JUDGED = {'limb': ('aVR',), 'chest': ('V1',)}   # inverted T is normal there


def _fd_filter(x: np.ndarray, fs: float) -> np.ndarray:
    """One lead filtered exactly as the Full Disclosure strip displays it
    (panels/holter_widgets.ECGStripCanvas.set_data): 0.5 Hz DFT high-pass
    (zero-phase, mirrored 2 s padding), 3-point median, Gaussian low-pass
    35 Hz twice (zero-phase, no ringing at the J point)."""
    from ..ecg_filters import apply_dft_filter
    from ..enhanced_filter import gaussian_lowpass
    x = np.asarray(x, dtype=float)
    if x.size > 30:
        pad = int(min(x.size - 1, 2.0 * fs))
        mirrored = np.pad(x, (pad, pad), mode='reflect')
        x = np.asarray(apply_dft_filter(mirrored, float(fs), "0.5"), dtype=float)[pad:pad + x.size]
    if x.size >= 3:
        p = np.pad(x, (1, 1), mode='edge')
        x = np.median(np.stack((p[:-2], p[1:-1], p[2:]), axis=0), axis=0)
    x = gaussian_lowpass(x, 35.0, float(fs))
    return gaussian_lowpass(x, 35.0, float(fs))


def _read_chunk(reader, t: float, t_end: float, total: float, fs: float):
    """The chunk [t, t_end) plus _CHUNK_PAD_SEC padding, every lead through
    _fd_filter, or None when too short."""
    p0 = max(0.0, t - _CHUNK_PAD_SEC)
    p1 = min(total, t_end + _CHUNK_PAD_SEC)
    raw = reader.read_range(p0, p1)
    if raw is None or raw.shape[1] < int(fs * _MIN_CHUNK_SEC):
        return None
    return np.vstack([_fd_filter(x, fs) for x in np.asarray(raw, dtype=float)])


def _chunks(total: float):
    """10 s chunks over the whole recording (a shorter last piece >= 5 s is
    kept)."""
    t = 0.0
    while t < total:
        t_end = min(t + _CHUNK_SEC, total)
        if t_end - t < _MIN_CHUNK_SEC:
            break
        yield t, t_end
        t = t_end


def _qrs_bounds(median_beats: np.ndarray, r_idx: int, fs: float):
    """QRS onset / J point on the median beats from the combined slope of all
    leads (each scaled by its own QRS size): where it stays under 10% of its
    peak for 16 ms (20% if not found), as the bundle branch block width."""
    n = median_beats.shape[1]
    w_lo, w_hi = r_idx - int(0.15 * fs), r_idx + int(0.20 * fs)
    if w_lo < 0 or w_hi > n:
        return None
    velocity = np.zeros(w_hi - w_lo)
    p_lo, p_hi = int(0.07 * fs), int(0.27 * fs)
    for x in median_beats:
        seg = np.asarray(x[w_lo:w_hi], dtype=float)
        if seg.size >= 11:
            seg = savgol_filter(seg, 11, 3)
        size = float(np.ptp(seg[p_lo:p_hi])) or 1.0
        velocity += np.abs(np.gradient(seg)) / size
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

    for frac in (0.10, 0.20):
        onset = edge(frac * float(velocity[peak]), -1)
        offset = edge(frac * float(velocity[peak]), +1)
        if onset is not None and offset is not None and offset > onset:
            return w_lo + onset, w_lo + offset
    return None


def _median_beats(data: np.ndarray, peaks: List[int], ref_idx: int, fs: float):
    """Align beats on the reference lead (correlation, +/-20 ms) and return
    (median beat per lead, beat-centre index, beat-shape consistency)."""
    built = _aligned_beats(data, peaks, ref_idx, fs)
    if built is None:
        return None
    beats, pre, corr = built
    return np.median(beats, axis=0), pre, corr


def _aligned_beats(data: np.ndarray, peaks: List[int], ref_idx: int, fs: float):
    """Align beats on the reference lead (correlation, +/-20 ms) and return
    (every beat of every lead as an array [beat, lead, sample], beat-centre
    index, beat-shape consistency)."""
    pre, post = int(0.35 * fs), int(0.55 * fs)
    max_lag = int(0.02 * fs)
    n = data.shape[1]
    usable = [p for p in peaks if p - pre - max_lag >= 0 and p + post + max_lag <= n]
    if len(usable) < _MIN_BEATS:
        return None
    ref = data[ref_idx]
    core = int(0.06 * fs)
    snippets = np.array([ref[p - core:p + core] for p in usable], dtype=float)
    template = np.median(snippets, axis=0)
    template = template - template.mean()
    aligned = []
    for p in usable:
        best_lag, best_c = 0, -2.0
        for lag in range(-max_lag, max_lag + 1):
            s = ref[p + lag - core:p + lag + core]
            s = s - s.mean()
            denom = float(np.linalg.norm(s) * np.linalg.norm(template))
            c = float(np.dot(s, template) / denom) if denom > 0 else -1.0
            if c > best_c:
                best_lag, best_c = lag, c
        aligned.append((p + best_lag, best_c))
    corr = float(np.median([c for _, c in aligned]))
    beats = np.array([[lead[p - pre:p + post] for lead in data] for p, _ in aligned], dtype=float)
    return beats, pre, corr


def _lead_order(name: str) -> int:
    order = LIMB_LEADS + CHEST_LEADS
    return order.index(name) if name in order else len(order)


def _modal_st(st_level: np.ndarray, st_early: np.ndarray, bin_size: float = _MODE_BIN_MV):
    """The ST value that occurs most often among the beats of one window (one
    lead): the beats' ST levels are put in bin_size wide bins, the fullest
    bin wins (a tie goes to the bin nearest the median), and the value is the
    mean of the beats in that bin -- ST level and ST at J+20 ms of the same
    beats. Returns (st_level, st_early) in the units given."""
    bins = np.floor(st_level / bin_size).astype(int)
    keys, counts = np.unique(bins, return_counts=True)
    top = keys[counts == counts.max()]
    if top.size > 1:
        mid = float(np.median(st_level))
        top = top[[int(np.argmin(np.abs((top + 0.5) * bin_size - mid)))]]
    pick = bins == top[0]
    return float(np.mean(st_level[pick])), float(np.mean(st_early[pick]))


def _lead_group(names: List[str], st_early: List[float], st_level: List[float],
                noisy: List[bool]) -> Dict[str, Any]:
    """One lead group as arrays: lead names, ST at J+20 ms and ST at the ST
    point (screen mm), and whether the lead was too noisy to judge,
    index-aligned."""
    return {'names': list(names),
            'st_early': np.asarray(st_early, dtype=float),
            'st_level': np.asarray(st_level, dtype=float),
            'noisy': list(noisy)}


def _rr_irregular(peaks: List[int]) -> bool:
    """True when the RR intervals jump from beat to beat (Atrial
    Fibrillation, frequent ectopics): median |RR change| > _IRREGULAR_RR of
    the median RR. Sinus rhythm changes gradually, a few % per beat."""
    rr = np.diff(np.asarray(peaks, dtype=float))
    if rr.size < 3:
        return False
    return float(np.median(np.abs(np.diff(rr)))) > _IRREGULAR_RR * float(np.median(rr))


def _chunk_st(data: np.ndarray, lead_names: List[str], fs: float, pan_tompkins,
              ref_idx: int, skip_leads: set):
    """ST deviation of one 10 s frame in screen mm, limb leads and chest
    leads in separate arrays: ({'limb': group, 'chest': group}, heart rate),
    or None when the frame can't be measured (see the module docstring).
    group = see _lead_group (ST at J+20 ms and at the ST point J+40 ms, J+30
    ms above 120 bpm). The ST value of a lead is the value that occurs most
    often among the beats of the frame (_modal_st), not the value of an
    averaged beat."""
    peaks = [int(p) for p in pan_tompkins(np.asarray(data[ref_idx], dtype=float), fs=fs)]
    if len(peaks) < _MIN_BEATS:
        return None
    # Irregular rhythm: no flat PR segment to measure from (f waves).
    if _rr_irregular(peaks):
        return None
    rr = np.diff(peaks) / float(fs)
    hr = 60.0 / float(np.median(rr)) if rr.size else 0.0
    built = _aligned_beats(data, peaks, ref_idx, fs)
    if built is None:
        return None
    beats, r_idx, corr = built
    # Beats that don't share one shape (ectopics, noise) don't give a
    # meaningful ST reading.
    if corr < 0.9:
        return None
    # The median beat is used only to LOCATE the QRS onset / J point; the ST
    # values are measured on every beat (below).
    med = np.median(beats, axis=0)
    bounds = _qrs_bounds(med, r_idx, fs)
    if bounds is None:
        return None
    onset, j_point = bounds
    if (j_point - onset) * 1000.0 / fs >= _MAX_QRS_MS:
        return None
    base_lo, base_hi = onset - int(0.04 * fs), onset - int(0.01 * fs)
    early = j_point + int(0.02 * fs)
    st_point = j_point + int((0.03 if hr > 120.0 else 0.04) * fs)
    half = max(1, int(0.005 * fs))
    if base_lo < 0 or st_point + half >= med.shape[1]:
        return None
    limb = ([], [], [], [])
    chest = ([], [], [], [])
    for li, name in enumerate(lead_names):
        if name in skip_leads or li >= med.shape[0]:
            continue
        group = limb if name in LIMB_LEADS else chest if name in CHEST_LEADS else None
        if group is None:
            continue
        x = beats[:, li, :]                                   # [beat, sample]
        baseline = x[:, base_lo:base_hi].mean(axis=1)
        beat_early = (x[:, early - half:early + half].mean(axis=1) - baseline) * _ADC_TO_MM
        beat_level = (x[:, st_point - half:st_point + half].mean(axis=1) - baseline) * _ADC_TO_MM
        st_level, st_early = _modal_st(beat_level, beat_early, _ST_MODE_BIN_MM)
        q1, q3 = np.percentile(beat_level, [25, 75])
        group[0].append(name)
        group[1].append(st_early)
        group[2].append(st_level)
        group[3].append(bool(q3 - q1 > _LEAD_SPREAD_MM))
    return {'limb': _lead_group(*limb), 'chest': _lead_group(*chest)}, hr


def _chunk_label(st: Dict[str, Dict[str, Any]]):
    """('ST Elevation' | 'ST Depression' | None, {lead: ST mm}, {lead: ST mm})
    for one frame.

    The limb-lead array and the chest-lead array are judged separately; a
    label needs at least _MIN_LEADS_IN_GROUP leads of ONE group past its
    threshold. Noisy leads and aVR are not judged. The first dict holds
    the leads of the groups that qualified. The second, only for an ST
    Elevation frame, holds every lead past the depression criteria (each lead
    on its own, no group count) -- sorted per event by _depression_kinds.
    """
    elevated, depressed, dep_any = {}, {}, {}
    for key in ('limb', 'chest'):
        group = st[key]
        elev, dep = {}, {}
        for name, st_early, st_level, noisy in zip(group['names'], group['st_early'],
                                                   group['st_level'], group['noisy']):
            if name == 'aVR' or noisy:
                continue
            if st_level >= _ELEVATION_MM[key]:
                elev[name] = float(st_level)
            if st_level <= _DEPRESSION_MM and (st_level - st_early) <= _UPSLOPE_LIMIT_MM:
                dep[name] = float(st_level)
        dep_any.update(dep)
        if len(elev) >= _MIN_LEADS_IN_GROUP:
            elevated.update(elev)
        if len(dep) >= _MIN_LEADS_IN_GROUP:
            depressed.update(dep)
    if elevated:
        return ST_ELEVATION, elevated, dep_any
    if depressed:
        return ST_DEPRESSION, depressed, {}
    return None, {}, {}


def _depression_kinds(elevated_leads, dep_frames: List[Dict[str, float]]) -> Dict[str, str]:
    """Depression in the frames of one ST Elevation event: {lead: kind}.

    Walls elevated: _WALL_LEADS with at least _MIN_WALL_LEADS of their leads
    among the event's elevated leads.
      Reciprocal: the lead faces the wall opposite an elevated wall
        (_RECIPROCAL_LEADS) -- one lead is enough, it is the mirror image of
        the injury already found.
      Reciprocal (posterior): V1-V3 with inferior / lateral elevation.
      Additional depression: any other lead; like an ST Depression frame it
        needs _MIN_LEADS_IN_GROUP such leads of one group in the same frame.
    A lead elevated in the event is left out."""
    elevated = set(elevated_leads)
    walls = [w for w, names in _WALL_LEADS.items()
             if sum(n in elevated for n in names) >= _MIN_WALL_LEADS]
    mirror = {n for w in walls for n in _RECIPROCAL_LEADS[w]}
    posterior = set(_POSTERIOR_LEADS) if any(w in _POSTERIOR_WITH for w in walls) else set()
    kinds: Dict[str, str] = {}
    for dep in dep_frames:
        other = []
        for name in dep:
            if name in elevated:
                continue
            if name in mirror:
                kinds[name] = DEP_RECIPROCAL
            elif name in posterior:
                kinds[name] = DEP_POSTERIOR
            else:
                other.append(name)
        for names in (LIMB_LEADS, CHEST_LEADS):
            group = [n for n in other if n in names]
            if len(group) >= _MIN_LEADS_IN_GROUP:
                for n in group:
                    kinds[n] = DEP_ADDITIONAL
    return kinds


def _chunk_all_leads_mm(st: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """ST level (screen mm) of every lead of one frame."""
    out = {}
    for key in ('limb', 'chest'):
        for name, level in zip(st[key]['names'], st[key]['st_level']):
            out[name] = float(level)
    return out


def detect_st_episodes(
    reader,
    make_time_strs: Optional[Callable[[float, float], tuple]] = None,
    info: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return ST Elevation / ST Depression events (start_sec, end_sec, label,
    color, start_time_str, end_time_str, plus 'st_leads', 'st_lead_mm' =
    qualifying leads, average over all the event's frames,
    'st_lead_max_mm' = their largest value in the frames they qualified in,
    'st_mv' = the largest value in mV, 'st_all_mm' = every lead, average
    over the event's frames; ST Elevation events also 'st_dep_kind' /
    'st_dep_mm' / 'st_dep_max_mm' = the depressed leads with their kind
    (_depression_kinds), average and deepest value). Graph-paper mm. When `info` is given it is filled with 'duration_sec', 'start_time',
    'frames' and 'measured' (frames that could be read). See the module
    docstring."""
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:  # pragma: no cover
        from ..pan_tompkins import pan_tompkins

    fs = float(getattr(reader, 'fs', 500) or 500)
    total = float(getattr(reader, 'duration_sec', 0.0) or 0.0)
    lead_names = list(getattr(reader, 'lead_names', []) or [])
    if info is not None:
        info.update({'duration_sec': total, 'frames': 0, 'measured': 0,
                     'start_time': float(getattr(reader, 'start_time', 0.0) or 0.0)})
    if total <= 0 or not lead_names:
        return []
    ref_name = next((n for n in ('II', 'V5', 'I', 'V6') if n in lead_names), lead_names[0])
    ref_idx = lead_names.index(ref_name)

    if make_time_strs is None:
        def make_time_strs(s_sec, e_sec):
            def fmt(t):
                return f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}"
            return fmt(s_sec), fmt(e_sec)

    # Positive frames: (start, end, label, {lead: mm}, {every lead: mm},
    # {depressed lead: mm} -- ST Elevation frames only)
    positives = []
    n_frames = n_measured = 0
    for t, t_end in _chunks(total):
        n_frames += 1
        try:
            data = _read_chunk(reader, t, t_end, total, fs)
            if data is None:
                continue
            result = _chunk_st(data, lead_names[:data.shape[0]], fs, pan_tompkins, ref_idx, set())
            if result is None:
                continue
            n_measured += 1
            st, _hr = result
            label, lead_mm, dep_mm = _chunk_label(st)
            if label is not None:
                positives.append((t, t_end, label, lead_mm, _chunk_all_leads_mm(st), dep_mm))
        except Exception as e:
            print(f"[Ischemia Detect] Chunk error {t:.1f}s: {e}")
    if info is not None:
        info.update({'frames': n_frames, 'measured': n_measured})

    # Events: same label, a gap of up to one frame joined.
    events = []
    for s_sec, e_sec, label, lead_mm, all_mm, dep_mm in positives:
        ev = events[-1] if events else None
        if ev is None or ev['label'] != label or s_sec - ev['end_sec'] > _ST_EVENT_GAP_SEC + 1e-6:
            ev = {'start_sec': s_sec, 'end_sec': e_sec, 'label': label,
                  'qual_sec': 0.0, 'lead': {}, 'all': {}, 'dep_frames': []}
            events.append(ev)
        ev['dep_frames'].append(dep_mm)
        ev['end_sec'] = e_sec
        ev['qual_sec'] += e_sec - s_sec
        for lead, v in lead_mm.items():
            ev['lead'].setdefault(lead, []).append(v)
        for lead, v in all_mm.items():
            ev['all'].setdefault(lead, []).append(v)

    out = []
    for ev in events:
        if ev['qual_sec'] < _ST_MIN_EVENT_SEC - 1e-6:
            continue
        elevation = ev['label'] == ST_ELEVATION
        s_str, e_str = make_time_strs(ev['start_sec'], ev['end_sec'])
        leads = sorted(ev['lead'], key=_lead_order)
        peak = {lead: (max(v) if elevation else min(v)) for lead, v in ev['lead'].items()}
        worst = max(peak.values()) if elevation else min(peak.values())
        # Depression with an ST Elevation event (empty for ST Depression):
        # kind per lead, average over all frames of the event (as
        # 'st_lead_mm'), deepest value in the frames the lead was depressed.
        dep_kind = _depression_kinds(leads, ev['dep_frames']) if elevation else {}
        dep_leads = sorted(dep_kind, key=_lead_order)
        dep_deepest = {lead: min(d[lead] for d in ev['dep_frames'] if lead in d) for lead in dep_leads}
        out.append({
            'start_sec': float(ev['start_sec']),
            'end_sec': float(ev['end_sec']),
            'label': ev['label'],
            'color': ST_COLORS[ev['label']],
            'start_time_str': s_str,
            'end_time_str': e_str,
            'detected_rhythm': ev['label'],
            'st_leads': leads,
            'st_mv': round(float(worst) / _MM_PER_MV, 3),
            'st_lead_mm': {lead: round(float(np.mean(ev['all'][lead])), 1) for lead in leads},
            'st_lead_max_mm': {lead: round(float(peak[lead]), 1) for lead in leads},
            'st_all_mm': {lead: round(float(np.mean(v)), 1) for lead, v in
                          sorted(ev['all'].items(), key=lambda kv: _lead_order(kv[0]))},
            'st_dep_kind': {lead: dep_kind[lead] for lead in dep_leads},
            'st_dep_mm': {lead: round(float(np.mean(ev['all'][lead])), 1) for lead in dep_leads},
            'st_dep_max_mm': {lead: round(float(dep_deepest[lead]), 1) for lead in dep_leads},
        })
    return out


def _chunk_t_wave(data: np.ndarray, lead_names: List[str], fs: float, pan_tompkins, ref_idx: int):
    """T wave amplitude (mV, signed, relative to the PR-segment baseline) of
    every lead of one chunk: ({'limb': {lead: mV}, 'chest': {lead: mV}}, heart
    rate), or None when the chunk can't be measured (too few beats, beats of
    different shapes, wide QRS, heart rate above _T_MAX_HR).

    Measured on the median beat (same beats / QRS bounds / baseline as the ST
    measurement). Search window: from J + 60 ms to 0.55 x RR after the R peak
    (at most 0.50 s), so it never reaches the next P wave. The T wave is the
    largest turning point (local peak or trough) of the smoothed beat inside
    that window; a slope that only runs into the window edge (a downsloping ST
    segment, the next P wave) is not a T wave and counts as 0."""
    from scipy.signal import find_peaks

    peaks = [int(p) for p in pan_tompkins(np.asarray(data[ref_idx], dtype=float), fs=fs)]
    if len(peaks) < _MIN_BEATS:
        return None
    rr = np.diff(peaks) / float(fs)
    if not rr.size:
        return None
    rr_med = float(np.median(rr))
    hr = 60.0 / rr_med
    if hr > _T_MAX_HR:
        return None
    built = _median_beats(data, peaks, ref_idx, fs)
    if built is None:
        return None
    med, r_idx, corr = built
    if corr < 0.9:
        return None
    bounds = _qrs_bounds(med, r_idx, fs)
    if bounds is None:
        return None
    onset, j_point = bounds
    if (j_point - onset) * 1000.0 / fs >= _MAX_QRS_MS:
        return None
    base_lo, base_hi = onset - int(0.04 * fs), onset - int(0.01 * fs)
    t_lo = j_point + int(_T_START_AFTER_J_SEC * fs)
    t_hi = min(r_idx + int(min(0.50, 0.55 * rr_med) * fs), med.shape[1] - 1)
    if base_lo < 0 or t_hi - t_lo < int(0.08 * fs):
        return None
    win = int(0.04 * fs) | 1                      # ~40 ms smoothing, odd
    min_prom = _T_MIN_PROMINENCE_MV / _ADC_TO_MV
    out = {'limb': {}, 'chest': {}}
    for li, name in enumerate(lead_names):
        group = 'limb' if name in LIMB_LEADS else 'chest' if name in CHEST_LEADS else None
        if group is None or li >= med.shape[0]:
            continue
        x = np.asarray(med[li], dtype=float)
        x = savgol_filter(x, win, 3) if x.size > win else x
        seg = x[t_lo:t_hi + 1] - float(np.mean(x[base_lo:base_hi]))
        cand = []
        for sign in (1.0, -1.0):
            idx, _ = find_peaks(sign * seg, prominence=min_prom)
            cand.extend(float(seg[i]) for i in idx)
        amp = max(cand, key=abs) if cand else 0.0
        out[group][name] = amp * _ADC_TO_MV
    return out, hr


def _t_chunk_label(t_mv: Dict[str, Dict[str, float]]):
    """(T_INVERSION | None, {lead: T mV}) for one chunk. Limb leads and chest
    leads are judged separately, never mixed: a group qualifies when at least
    _MIN_LEADS_IN_GROUP of its leads have a T wave at or below
    _T_INVERSION_MV. aVR and V1 are not judged (an inverted T is normal
    there). The returned dict holds the leads of the groups that qualified."""
    inverted = {}
    for key in ('limb', 'chest'):
        inv = {name: float(mv) for name, mv in t_mv[key].items()
               if name not in _T_NOT_JUDGED[key] and mv <= _T_INVERSION_MV}
        if len(inv) >= _MIN_LEADS_IN_GROUP:
            inverted.update(inv)
    return (T_INVERSION, inverted) if inverted else (None, {})


def detect_t_inversion_episodes(
    reader,
    make_time_strs: Optional[Callable[[float, float], tuple]] = None,
) -> List[Dict[str, Any]]:
    """Return T Inversion episodes as segment dicts, same format as
    detect_st_episodes() ('st_leads', 'st_mv', 'st_lead_mm' = qualifying
    leads, 'st_all_mm' = T amplitude of all 12 leads, all in mm / worst = most
    negative).

    Reads the recording itself in 10 s chunks, every lead through the Full
    Disclosure filter, exactly like detect_st_episodes(); chunks whose beats
    differ in shape, with a wide QRS or above _T_MAX_HR are skipped; episodes
    need >= _MIN_EPISODE_SEC and gaps up to _MERGE_GAP_SEC are joined."""
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:  # pragma: no cover
        from ..pan_tompkins import pan_tompkins

    fs = float(getattr(reader, 'fs', 500) or 500)
    total = float(getattr(reader, 'duration_sec', 0.0) or 0.0)
    lead_names = list(getattr(reader, 'lead_names', []) or [])
    if total <= 0 or not lead_names:
        return []
    ref_name = next((n for n in ('II', 'V5', 'I', 'V6') if n in lead_names), lead_names[0])
    ref_idx = lead_names.index(ref_name)

    if make_time_strs is None:
        def make_time_strs(s_sec, e_sec):
            def fmt(t):
                return f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}"
            return fmt(s_sec), fmt(e_sec)

    positives = []     # (start, end, {lead: mV qualifying}, {lead: mV all})
    for t, t_end in _chunks(total):
        try:
            data = _read_chunk(reader, t, t_end, total, fs)
            if data is None:
                continue
            result = _chunk_t_wave(data, lead_names[:data.shape[0]], fs, pan_tompkins, ref_idx)
            if result is not None:
                t_mv, _hr = result
                label, lead_mv = _t_chunk_label(t_mv)
                if label is not None:
                    all_mv = {**t_mv['limb'], **t_mv['chest']}
                    positives.append((t, t_end, lead_mv, all_mv))
        except Exception as e:
            print(f"[T Inversion Detect] Chunk error {t:.1f}s: {e}")

    # Episodes: gaps up to _MERGE_GAP_SEC joined; per lead the most negative T
    # of the episode is kept.
    episodes = []
    for s_sec, e_sec, lead_mv, all_mv in positives:
        if episodes and s_sec - episodes[-1]['end_sec'] <= _MERGE_GAP_SEC:
            ep = episodes[-1]
            ep['end_sec'] = e_sec
            for lead, v in lead_mv.items():
                ep['lead_mv'][lead] = min(ep['lead_mv'][lead], v) if lead in ep['lead_mv'] else v
            for lead, v in all_mv.items():
                ep['all_mv'][lead] = min(ep['all_mv'][lead], v) if lead in ep['all_mv'] else v
            continue
        episodes.append({'start_sec': s_sec, 'end_sec': e_sec,
                         'lead_mv': dict(lead_mv), 'all_mv': dict(all_mv)})

    out = []
    for ep in episodes:
        if ep['end_sec'] - ep['start_sec'] < _MIN_EPISODE_SEC:
            continue
        s_str, e_str = make_time_strs(ep['start_sec'], ep['end_sec'])
        leads = sorted(ep['lead_mv'], key=_lead_order)
        out.append({
            'start_sec': float(ep['start_sec']),
            'end_sec': float(ep['end_sec']),
            'label': T_INVERSION,
            'color': ST_COLORS[T_INVERSION],
            'start_time_str': s_str,
            'end_time_str': e_str,
            'detected_rhythm': T_INVERSION,
            'st_leads': leads,
            'st_mv': round(float(min(ep['lead_mv'].values())), 3),
            'st_lead_mm': {lead: round(ep['lead_mv'][lead] * _MM_PER_MV, 1) for lead in leads},
            'st_all_mm': {lead: round(v * _MM_PER_MV, 1) for lead, v in
                          sorted(ep['all_mv'].items(), key=lambda kv: _lead_order(kv[0]))},
        })
    return out


_CACHE_FILENAME = 'ischemia_cache.json'
_memory_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_jobs: Dict[str, Any] = {}
_code_hash: Optional[str] = None


def _cache_key(path: str) -> Optional[str]:
    """Recording file (path / size / mtime) + the code this result depends
    on (this module, the filters, the R-peak detector), so it is recomputed
    when either changes. Nothing from the arrhythmia detection."""
    import hashlib
    import os
    global _code_hash
    try:
        st = os.stat(path)
    except OSError:
        return None
    if _code_hash is None:
        here = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.sha1()
        for f in (os.path.join(here, 'ischemia_detection.py'),
                  os.path.join(here, '..', 'ecg_filters.py'),
                  os.path.join(here, '..', 'enhanced_filter.py'),
                  os.path.join(here, '..', 'pan_tompkins.py')):
            try:
                with open(f, 'rb') as fh:
                    h.update(fh.read())
            except OSError:
                pass
        _code_hash = h.hexdigest()
    return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{_code_hash}"


def get_cached_ischemia(path: str) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Cached compute_ischemia() result for this recording --
    {'st': ST episodes, 't': T Inversion episodes} -- or None. Cheap."""
    import copy
    import json
    import os
    key = _cache_key(path)
    if key is None:
        return None
    if key in _memory_cache:
        return copy.deepcopy(_memory_cache[key])
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(path)), _CACHE_FILENAME), 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if saved.get('key') != key or not isinstance(saved.get('result'), dict):
        return None
    _memory_cache[key] = saved['result']
    return copy.deepcopy(_memory_cache[key])


def compute_ischemia(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """ST and T Inversion episodes of the recording at `path`:
    {'st': [...], 't': [...], 'st_info': {...}} (opens and closes the reader), cached in memory
    and next to the recording. Blocking; ischemia_async() runs it on a worker
    thread."""
    import json
    import os
    from .file_format import ECGHFileReader
    reader = ECGHFileReader(path)
    try:
        st_info: Dict[str, Any] = {}
        result = {'st': detect_st_episodes(reader, info=st_info),
                  't': detect_t_inversion_episodes(reader), 'st_info': st_info}
    finally:
        reader.close()
    key = _cache_key(path)
    if key is not None:
        _memory_cache[key] = result
        try:
            cache_path = os.path.join(os.path.dirname(os.path.abspath(path)), _CACHE_FILENAME)
            with open(cache_path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump({'key': key, 'result': result}, f, default=float)
            os.replace(cache_path + '.tmp', cache_path)
        except OSError:
            pass
    return result


def ischemia_async(path: str, on_done: Callable[[Dict[str, List[Dict[str, Any]]]], None]) -> None:
    """Run compute_ischemia() for the recording at `path` on a worker thread
    and call on_done(result) on the Qt main thread. Must be called from the Qt
    main thread."""
    import copy
    import os
    import threading
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

    cached = get_cached_ischemia(path)
    if cached is not None:
        QTimer.singleShot(0, lambda: on_done(cached))
        return
    key = _cache_key(path) or os.path.abspath(path)
    job = _jobs.get(key)
    if job is not None:
        job.callbacks.append(on_done)
        return

    class _Job(QObject):
        done = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.callbacks = [on_done]
            self.done.connect(self._deliver)

        @pyqtSlot(object)
        def _deliver(self, result):
            _jobs.pop(key, None)
            for cb in self.callbacks:
                try:
                    cb(copy.deepcopy(result))
                except RuntimeError:
                    pass  # the receiving widget was closed meanwhile
                except Exception as e:
                    print(f"[Ischemia Detect] Callback error: {e}")

    job = _Job()
    _jobs[key] = job

    def _run():
        result = {'st': [], 't': [], 'st_info': None}
        try:
            result = compute_ischemia(path)
        except Exception as e:
            print(f"[Ischemia Detect] Failed: {e}")
        job.done.emit(result)

    threading.Thread(target=_run, daemon=True, name='ischemia-detect').start()


_ST_OVERVIEW_EVENTS = 5


def format_st_summary(episodes: Optional[List[Dict[str, Any]]], label: str, multiline: bool = False,
                      info: Optional[Dict[str, Any]] = None) -> str:
    """Text for the Overview / Replay "ST Elevation" / "ST Depression" /
    "T Inversion" rows. episodes: every detected event (any label) or None
    when detection hasn't run -- then "-".

    ST: the event list -- per event its system start - end time, duration
    and the qualifying leads as "<lead> <average>" in graph-paper mm; the
    longest _ST_OVERVIEW_EVENTS events, in time order, then "+N more". `info`
    (detect_st_episodes) turns "None" into "Recording too short" / "Not
    measurable" where that is why.

    T Inversion: episode count and total duration, then the six limb leads
    and the six chest leads with their worst value in mm.

    multiline=True puts the parts on separate lines (for the tables),
    otherwise they are joined with " | "."""
    if episodes is None:
        return "-"
    sep = "\n" if multiline else " | "
    eps = [e for e in episodes if e.get('label') == label]

    def dur(sec):
        return f"{int(sec // 60)}m{int(sec % 60):02d}s"

    if label != T_INVERSION:
        if not eps:
            if info and float(info.get('duration_sec', 0.0) or 0.0) < _ST_MIN_EVENT_SEC:
                return f"Recording too short (< {int(_ST_MIN_EVENT_SEC)} s)"
            if info and int(info.get('frames', 0) or 0) > 0 and int(info.get('measured', 0) or 0) == 0:
                return "Not measurable (wide QRS / irregular / noise)"
            return "None"
        start = float((info or {}).get('start_time', 0.0) or 0.0)

        def clock(sec):
            if start > 0:
                from datetime import datetime
                return datetime.fromtimestamp(start + sec).strftime('%H:%M:%S')
            return f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}:{int(sec % 60):02d}"

        total = sum(max(0.0, float(e['end_sec']) - float(e['start_sec'])) for e in eps)
        longest = sorted(eps, key=lambda e: float(e['start_sec']) - float(e['end_sec']))[:_ST_OVERVIEW_EVENTS]
        shown = sorted(longest, key=lambda e: float(e['start_sec']))
        lines = [f"{len(eps)} event{'s' if len(eps) != 1 else ''}, {dur(total)} (mm)"]
        for i, e in enumerate(shown, 1):
            avg = e.get('st_lead_mm') or {}
            s, en = float(e['start_sec']), float(e['end_sec'])
            lines.append(f"{i}) {clock(s)} - {clock(en)} ({dur(en - s)})")
            for group, names in (('Limb', LIMB_LEADS), ('Chest', CHEST_LEADS)):
                leads = ", ".join(f"{lead} {float(avg[lead]):+.1f}"
                                  for lead in names if lead in avg)
                if leads:
                    lines.append(f"    {group}: {leads}")
            dep_mm, dep_kind = e.get('st_dep_mm') or {}, e.get('st_dep_kind') or {}
            for kind in DEP_KINDS:
                leads = ", ".join(f"{lead} {float(dep_mm[lead]):+.1f}"
                                  for lead in dep_mm if dep_kind.get(lead) == kind)
                if leads:
                    lines.append(f"    {kind}: {leads}")
        if len(eps) > len(shown):
            lines.append(f"+{len(eps) - len(shown)} more (report)")
        return sep.join(lines)

    if not eps:
        return "None"
    total = sum(max(0.0, float(e['end_sec']) - float(e['start_sec'])) for e in eps)
    per_lead: Dict[str, float] = {}
    for e in eps:
        for lead, mm in (e.get('st_lead_mm') or {}).items():
            mm = float(mm)
            per_lead[lead] = min(per_lead[lead], mm) if lead in per_lead else mm

    def line(leads):
        return ", ".join(f"{lead}: {per_lead[lead]:+.1f}" if lead in per_lead else f"{lead}: -" for lead in leads)

    head = f"{len(eps)} ep, {dur(total)} (mm)"
    return sep.join([head, line(LIMB_LEADS), line(CHEST_LEADS)])
