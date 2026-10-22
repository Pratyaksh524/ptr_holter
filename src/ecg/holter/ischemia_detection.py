"""
Ischemia (ST elevation / ST depression) episode detection for Holter
recordings.

Called from auto_segment_arrthymia_detection._detect_arrhythmias_uncached()
after the rhythm segments are final, and runs on the same recording data the
Full Disclosure view shows (ECGHFileReader.read_range). It is its own layer:
it never changes, trims or suppresses any rhythm segment; the rhythm segments
are only used to split the recording rhythm by rhythm and to skip stretches
where the ST segment can't be read.

Method, rhythm by rhythm, chunk by chunk:
  1. The recording is split wherever the rhythm label changes (the rhythm
     segments passed in, plus the unlabeled stretches between them). Each
     rhythm stretch is read in 10 s chunks aligned to its own start, so no
     chunk mixes two rhythms.
  2. Skipped: Ventricular Fibrillation, Ventricular Tachycardia and every
     Bundle Branch Block (right/left, complete/incomplete), and any chunk
     whose median beat has QRS >= 120 ms even without such a label (an
     undetected bundle branch block, a ventricular escape rhythm, paced
     beats). A wide QRS has secondary ST-T changes, and its J point sits in
     the slurred end of the QRS (seen: RBBB/LBBB stretches reading
     +1.3..+3.1 mV "ST elevation").
  3. Per chunk: every lead filtered with the same SignalPreprocessor as the
     AFib/Flutter/AV-block/BBB rules (0.5-40 Hz zero-phase bandpass + 50 Hz
     notch); R-peaks found on the reference lead; beats aligned to each
     other; one median beat per lead (noise averages out).
  4. On the median beats: QRS onset and J point (QRS end) from the combined
     slope of all leads -- the same global multi-lead method the bundle
     branch block width uses. Baseline = PR segment (40-10 ms before QRS
     onset). ST level measured at J+40 ms (J+30 ms above 120 bpm), slope
     from J+20 ms to that point, in mV relative to the baseline.
     Why not at J or J+80: measured on normal sinus beats, the global J
     point sits 10-20 ms before the end of the QRS in V1-V3 (+0.09..+0.17 mV
     at J, ~0 at J+20), and by J+60..J+80 the T wave has started
     (+0.24 mV at J+80 at 105 bpm). J+20..J+40 was flat at ~0 on every lead.
  5. Criteria, limb leads and chest leads judged separately (never mixed):
       limb group  (I, II, III, aVL, aVF)  : any 2 leads
       chest group (V1-V6)                 : any 2 leads
       ST Elevation : limb leads >= 1 mm (0.1 mV), chest leads >= 2 mm
                      (0.2 mV).
       ST Depression: limb and chest leads <= -1 mm (-0.1 mV), horizontal or
                      downsloping (not more than 0.05 mV above the J+20
                      level).
     aVR is not judged. A chunk with both is labeled ST Elevation (the
     depression is reciprocal). Values are reported per lead, in mm
     (10 mm = 1 mV, the standard 10 mm/mV).
  6. Episodes: consecutive positive chunks of one label, joined across gaps
     up to _MERGE_GAP_SEC, kept when they last >= _MIN_EPISODE_SEC (the
     Holter "1 mm x 1 min" convention).
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
_MIN_EPISODE_SEC = 60.0
_MERGE_GAP_SEC = 60.0

_MM_PER_MV = 10.0
_ELEVATION_MV = {'limb': 0.10, 'chest': 0.20}     # 1 mm limb leads, 2 mm chest leads
_DEPRESSION_MV = -0.10                            # 1 mm, limb and chest leads
_UPSLOPE_LIMIT_MV = 0.05
_MIN_LEADS_IN_GROUP = 2

# Limb and chest leads are measured and judged as two separate arrays
# (limb leads: frontal plane; chest leads: horizontal plane).
LIMB_LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
CHEST_LEADS = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')

_SKIP_RHYTHMS = (
    'ventricular fibrillation',
    'ventricular tachycardia',
    'bundle branch block',
)
_MAX_QRS_MS = 120.0

# T inversion (detect_t_inversion_episodes)
_T_INVERSION_MV = -0.10          # T wave 1 mm (0.1 mV) or more below the PR baseline
_T_MAX_HR = 120.0                # above this the T wave runs into the next P wave
_T_START_AFTER_J_SEC = 0.06      # T search window starts here after the J point
_T_MIN_PROMINENCE_MV = 0.02      # a real T wave is a turning point, not a slope
_T_NOT_JUDGED = {'limb': ('aVR',), 'chest': ('V1',)}   # inverted T is normal there
# AFib / Flutter: the f-waves / flutter waves (seen up to -2 mV deflections in
# the T window) are not T waves and would read as T inversion.
_T_SKIP_RHYTHMS = _SKIP_RHYTHMS + ('atrial fibrillation', 'atrial flutter')


def _rhythm_chunks(total_duration: float, rhythm_segments: List[Dict[str, Any]]):
    """Split [0, total_duration] wherever any rhythm segment starts or ends,
    and return (start, end, labels-active) for each piece."""
    cuts = {0.0, float(total_duration)}
    for seg in rhythm_segments:
        s = float(seg.get('start_sec', 0.0))
        e = float(seg.get('end_sec', s))
        if 0.0 < s < total_duration:
            cuts.add(s)
        if 0.0 < e < total_duration:
            cuts.add(e)
    edges = sorted(cuts)
    out = []
    for a, b in zip(edges, edges[1:]):
        if b - a <= 0:
            continue
        mid = 0.5 * (a + b)
        labels = [str(seg.get('label', '')) for seg in rhythm_segments
                  if float(seg.get('start_sec', 0.0)) <= mid < float(seg.get('end_sec', 0.0))]
        out.append((a, b, labels))
    return out


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
    return np.median(beats, axis=0), pre, corr


def _lead_order(name: str) -> int:
    order = LIMB_LEADS + CHEST_LEADS
    return order.index(name) if name in order else len(order)


def _lead_group(names: List[str], st_early: List[float], st_level: List[float]) -> Dict[str, Any]:
    """One lead group as arrays: lead names, ST at J+20 ms and ST at the ST
    point, in mV, index-aligned."""
    return {'names': list(names),
            'st_early': np.asarray(st_early, dtype=float),
            'st_level': np.asarray(st_level, dtype=float)}


def _chunk_st(data: np.ndarray, lead_names: List[str], fs: float, pan_tompkins,
              ref_idx: int, skip_leads: set):
    """ST deviation (mV) of one chunk, limb leads and chest leads in separate
    arrays: ({'limb': group, 'chest': group}, heart rate), or None when the
    chunk can't be measured. group = see _lead_group (ST at J+20 ms and at
    the ST point J+40 ms, J+30 ms above 120 bpm)."""
    peaks = [int(p) for p in pan_tompkins(np.asarray(data[ref_idx], dtype=float), fs=fs)]
    if len(peaks) < _MIN_BEATS:
        return None
    rr = np.diff(peaks) / float(fs)
    hr = 60.0 / float(np.median(rr)) if rr.size else 0.0
    built = _median_beats(data, peaks, ref_idx, fs)
    if built is None:
        return None
    med, r_idx, corr = built
    # Beats that don't share one shape (ectopics, noise) don't make a
    # meaningful median beat.
    if corr < 0.9:
        return None
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
    limb = ([], [], [])
    chest = ([], [], [])
    for li, name in enumerate(lead_names):
        if name in skip_leads or li >= med.shape[0]:
            continue
        group = limb if name in LIMB_LEADS else chest if name in CHEST_LEADS else None
        if group is None:
            continue
        x = med[li]
        baseline = float(np.mean(x[base_lo:base_hi]))
        st_early = float(np.mean(x[early - half:early + half])) - baseline
        st_level = float(np.mean(x[st_point - half:st_point + half])) - baseline
        group[0].append(name)
        group[1].append(st_early * _ADC_TO_MV)
        group[2].append(st_level * _ADC_TO_MV)
    return {'limb': _lead_group(*limb), 'chest': _lead_group(*chest)}, hr


def _chunk_label(st: Dict[str, Dict[str, Any]]):
    """('ST Elevation' | 'ST Depression' | None, {lead: ST mV}) for one chunk.

    The limb-lead array and the chest-lead array are judged separately; a
    label needs at least _MIN_LEADS_IN_GROUP leads of ONE group past its
    threshold. The returned dict holds the leads of the groups that
    qualified.
    """
    elevated, depressed = {}, {}
    for key in ('limb', 'chest'):
        group = st[key]
        elev, dep = {}, {}
        for name, st_early, st_level in zip(group['names'], group['st_early'], group['st_level']):
            if name == 'aVR':
                continue
            if st_level >= _ELEVATION_MV[key]:
                elev[name] = float(st_level)
            if st_level <= _DEPRESSION_MV and (st_level - st_early) <= _UPSLOPE_LIMIT_MV:
                dep[name] = float(st_level)
        if len(elev) >= _MIN_LEADS_IN_GROUP:
            elevated.update(elev)
        if len(dep) >= _MIN_LEADS_IN_GROUP:
            depressed.update(dep)
    if elevated:
        return ST_ELEVATION, elevated
    if depressed:
        return ST_DEPRESSION, depressed
    return None, {}


def _chunk_all_leads_mv(st: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """ST level (mV) of every lead of one chunk, for the 6 + 6 lead display."""
    out = {}
    for key in ('limb', 'chest'):
        for name, level in zip(st[key]['names'], st[key]['st_level']):
            out[name] = float(level)
    return out


def detect_st_episodes(
    reader,
    rhythm_segments: Optional[List[Dict[str, Any]]] = None,
    make_time_strs: Optional[Callable[[float, float], tuple]] = None,
) -> List[Dict[str, Any]]:
    """Return ST Elevation / ST Depression episodes as segment dicts in the
    same format as the rhythm segments (start_sec, end_sec, label, color,
    start_time_str, end_time_str, detected_rhythm), plus 'st_leads' and
    'st_mv' (worst deviation, mV)."""
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:  # pragma: no cover
        from ..pan_tompkins import pan_tompkins
    try:
        from .analysis_pipeline import SignalPreprocessor, HolterConfig
        fs0 = getattr(reader, 'fs', 500) or 500
        preproc = SignalPreprocessor(fs=fs0, config=HolterConfig.from_clinical_config())
    except Exception:
        preproc = None

    fs = float(getattr(reader, 'fs', 500) or 500)
    total = float(getattr(reader, 'duration_sec', 0.0) or 0.0)
    lead_names = list(getattr(reader, 'lead_names', []) or [])
    if total <= 0 or not lead_names:
        return []
    ref_name = next((n for n in ('II', 'V5', 'I', 'V6') if n in lead_names), lead_names[0])
    ref_idx = lead_names.index(ref_name)
    rhythm_segments = list(rhythm_segments or [])

    if make_time_strs is None:
        def make_time_strs(s_sec, e_sec):
            def fmt(t):
                return f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}"
            return fmt(s_sec), fmt(e_sec)

    # Positive chunks: (start, end, label, {lead: ST mV})
    positives = []
    for c_start, c_end, labels in _rhythm_chunks(total, rhythm_segments):
        low = [l.lower() for l in labels]
        if any(skip in l for l in low for skip in _SKIP_RHYTHMS):
            continue
        skip_leads = set()
        t = c_start
        while t < c_end:
            t_end = min(t + _CHUNK_SEC, c_end)
            if t_end - t < _MIN_CHUNK_SEC:
                break
            try:
                p0 = max(0.0, t - _CHUNK_PAD_SEC)
                p1 = min(total, t_end + _CHUNK_PAD_SEC)
                raw = reader.read_range(p0, p1)
                if raw is None or raw.shape[1] < int(fs * _MIN_CHUNK_SEC):
                    t = t_end
                    continue
                data = np.asarray(raw, dtype=float)
                if preproc is not None:
                    data = np.vstack([np.asarray(preproc.process(x), dtype=float) for x in data])
                result = _chunk_st(data, lead_names[:data.shape[0]], fs, pan_tompkins, ref_idx, skip_leads)
                if result is not None:
                    st, _hr = result
                    label, lead_mv = _chunk_label(st)
                    if label is not None:
                        positives.append((t, t_end, label, lead_mv, _chunk_all_leads_mv(st)))
            except Exception as e:
                print(f"[Ischemia Detect] Chunk error {t:.1f}s: {e}")
            t = t_end

    # Episodes: same label, gaps up to _MERGE_GAP_SEC joined. Per lead the
    # worst value over the episode is kept.
    def worse(label, a, b):
        return max(a, b) if label == ST_ELEVATION else min(a, b)

    episodes = []
    for s_sec, e_sec, label, lead_mv, all_mv in positives:
        if episodes and episodes[-1]['label'] == label and s_sec - episodes[-1]['end_sec'] <= _MERGE_GAP_SEC:
            ep = episodes[-1]
            ep['end_sec'] = e_sec
            for lead, v in lead_mv.items():
                ep['lead_mv'][lead] = worse(label, ep['lead_mv'][lead], v) if lead in ep['lead_mv'] else v
            for lead, v in all_mv.items():
                ep['all_mv'][lead] = worse(label, ep['all_mv'][lead], v) if lead in ep['all_mv'] else v
            continue
        episodes.append({'start_sec': s_sec, 'end_sec': e_sec, 'label': label,
                         'lead_mv': dict(lead_mv), 'all_mv': dict(all_mv)})

    out = []
    for ep in episodes:
        if ep['end_sec'] - ep['start_sec'] < _MIN_EPISODE_SEC:
            continue
        s_str, e_str = make_time_strs(ep['start_sec'], ep['end_sec'])
        leads = sorted(ep['lead_mv'], key=_lead_order)
        values = list(ep['lead_mv'].values())
        worst = max(values) if ep['label'] == ST_ELEVATION else min(values)
        out.append({
            'start_sec': float(ep['start_sec']),
            'end_sec': float(ep['end_sec']),
            'label': ep['label'],
            'color': ST_COLORS[ep['label']],
            'start_time_str': s_str,
            'end_time_str': e_str,
            'detected_rhythm': ep['label'],
            'st_leads': leads,
            'st_mv': round(float(worst), 3),
            'st_lead_mm': {lead: round(ep['lead_mv'][lead] * _MM_PER_MV, 1) for lead in leads},
            # every lead (worst value over the episode), qualifying or not
            'st_all_mm': {lead: round(v * _MM_PER_MV, 1) for lead, v in
                          sorted(ep['all_mv'].items(), key=lambda kv: _lead_order(kv[0]))},
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
    rhythm_segments: Optional[List[Dict[str, Any]]] = None,
    make_time_strs: Optional[Callable[[float, float], tuple]] = None,
) -> List[Dict[str, Any]]:
    """Return T Inversion episodes as segment dicts, same format as
    detect_st_episodes() ('st_leads', 'st_mv', 'st_lead_mm' = qualifying
    leads, 'st_all_mm' = T amplitude of all 12 leads, all in mm / worst = most
    negative).

    Runs on the same Full Disclosure recording data (all 12 leads, same
    SignalPreprocessor filter), rhythm by rhythm in 10 s chunks, exactly like
    the ST detection: VT / VFib / bundle branch block stretches and wide-QRS
    chunks are skipped; episodes need >= _MIN_EPISODE_SEC and gaps up to
    _MERGE_GAP_SEC are joined. It never changes any other segment.

    Pass the RHYTHM segments only (without ST / T episodes)."""
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:  # pragma: no cover
        from ..pan_tompkins import pan_tompkins
    try:
        from .analysis_pipeline import SignalPreprocessor, HolterConfig
        fs0 = getattr(reader, 'fs', 500) or 500
        preproc = SignalPreprocessor(fs=fs0, config=HolterConfig.from_clinical_config())
    except Exception:
        preproc = None

    fs = float(getattr(reader, 'fs', 500) or 500)
    total = float(getattr(reader, 'duration_sec', 0.0) or 0.0)
    lead_names = list(getattr(reader, 'lead_names', []) or [])
    if total <= 0 or not lead_names:
        return []
    ref_name = next((n for n in ('II', 'V5', 'I', 'V6') if n in lead_names), lead_names[0])
    ref_idx = lead_names.index(ref_name)
    rhythm_segments = [s for s in (rhythm_segments or []) if s.get('label') not in ST_LABELS]

    if make_time_strs is None:
        def make_time_strs(s_sec, e_sec):
            def fmt(t):
                return f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{int(t % 60):02d}"
            return fmt(s_sec), fmt(e_sec)

    positives = []     # (start, end, {lead: mV qualifying}, {lead: mV all})
    for c_start, c_end, labels in _rhythm_chunks(total, rhythm_segments):
        low = [l.lower() for l in labels]
        if any(skip in l for l in low for skip in _T_SKIP_RHYTHMS):
            continue
        t = c_start
        while t < c_end:
            t_end = min(t + _CHUNK_SEC, c_end)
            if t_end - t < _MIN_CHUNK_SEC:
                break
            try:
                p0 = max(0.0, t - _CHUNK_PAD_SEC)
                p1 = min(total, t_end + _CHUNK_PAD_SEC)
                raw = reader.read_range(p0, p1)
                if raw is None or raw.shape[1] < int(fs * _MIN_CHUNK_SEC):
                    t = t_end
                    continue
                data = np.asarray(raw, dtype=float)
                if preproc is not None:
                    data = np.vstack([np.asarray(preproc.process(x), dtype=float) for x in data])
                result = _chunk_t_wave(data, lead_names[:data.shape[0]], fs, pan_tompkins, ref_idx)
                if result is not None:
                    t_mv, _hr = result
                    label, lead_mv = _t_chunk_label(t_mv)
                    if label is not None:
                        all_mv = {**t_mv['limb'], **t_mv['chest']}
                        positives.append((t, t_end, lead_mv, all_mv))
            except Exception as e:
                print(f"[T Inversion Detect] Chunk error {t:.1f}s: {e}")
            t = t_end

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


_T_CACHE_FILENAME = 't_inversion_cache.json'
_t_memory_cache: Dict[str, List[Dict[str, Any]]] = {}
_t_jobs: Dict[str, Any] = {}


def _t_cache_key(path: str) -> Optional[str]:
    """Same key as the arrhythmia detection cache (file size / mtime / code
    fingerprint), so the result is recomputed when the recording or any code
    changes."""
    try:
        from .auto_segment_arrthymia_detection import _cache_key
        return _cache_key(path)
    except Exception:
        return None


def get_cached_t_inversion(path: str) -> Optional[List[Dict[str, Any]]]:
    """Cached detect_t_inversion_episodes() result for this recording, or
    None. Cheap."""
    import copy
    import json
    import os
    key = _t_cache_key(path)
    if key is None:
        return None
    if key in _t_memory_cache:
        return copy.deepcopy(_t_memory_cache[key])
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(path)), _T_CACHE_FILENAME), 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if saved.get('key') != key:
        return None
    _t_memory_cache[key] = saved.get('episodes', [])
    return copy.deepcopy(_t_memory_cache[key])


def compute_t_inversion(path: str, rhythm_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """detect_t_inversion_episodes() on the recording at `path` (opens and
    closes the reader), cached in memory and next to the recording. Blocking;
    t_inversion_async() runs it on a worker thread."""
    import json
    import os
    from .file_format import ECGHFileReader
    reader = ECGHFileReader(path)
    try:
        episodes = detect_t_inversion_episodes(reader, rhythm_segments=list(rhythm_segments or []))
    finally:
        reader.close()
    ck = _t_cache_key(path)
    if ck is not None:
        _t_memory_cache[ck] = episodes
        try:
            cache_path = os.path.join(os.path.dirname(os.path.abspath(path)), _T_CACHE_FILENAME)
            with open(cache_path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump({'key': ck, 'episodes': episodes}, f, default=float)
            os.replace(cache_path + '.tmp', cache_path)
        except OSError:
            pass
    return episodes


def t_inversion_async(path: str, rhythm_segments: List[Dict[str, Any]],
                      on_done: Callable[[List[Dict[str, Any]]], None]) -> None:
    """Run detect_t_inversion_episodes() on the recording at `path` on a
    worker thread and call on_done(episodes) on the Qt main thread (the
    result is cached in memory and next to the recording). Must be called
    from the Qt main thread."""
    import copy
    import json
    import os
    import threading
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

    cached = get_cached_t_inversion(path)
    if cached is not None:
        QTimer.singleShot(0, lambda: on_done(cached))
        return
    key = _t_cache_key(path) or os.path.abspath(path)
    job = _t_jobs.get(key)
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
        def _deliver(self, episodes):
            _t_jobs.pop(key, None)
            for cb in self.callbacks:
                try:
                    cb(copy.deepcopy(episodes))
                except RuntimeError:
                    pass  # the receiving widget was closed meanwhile
                except Exception as e:
                    print(f"[T Inversion Detect] Callback error: {e}")

    job = _Job()
    _t_jobs[key] = job
    segments = copy.deepcopy(list(rhythm_segments or []))

    def _run():
        episodes = []
        try:
            episodes = compute_t_inversion(path, segments)
        except Exception as e:
            print(f"[T Inversion Detect] Failed: {e}")
        job.done.emit(episodes)

    threading.Thread(target=_run, daemon=True, name='t-inversion-detect').start()


def format_st_summary(episodes: Optional[List[Dict[str, Any]]], label: str, multiline: bool = False) -> str:
    """Text for the Overview / Replay "ST Elevation" / "ST Depression" rows.
    episodes: every detected segment (any label) or None when detection
    hasn't run -- then "-". No episode of this label: "None". Otherwise the
    episode count and total duration, then the six limb leads and below them
    the six chest leads, each as "<lead>: <worst value in mm>" separated by
    commas (the worst qualifying value of each lead over all episodes of this
    label; a lead that did not meet the criteria shows "-"). multiline=True puts the three parts on
    three lines (for the tables), otherwise they are joined with " | "."""
    if episodes is None:
        return "-"
    eps = [e for e in episodes if e.get('label') == label]
    if not eps:
        return "None"
    total = sum(max(0.0, float(e['end_sec']) - float(e['start_sec'])) for e in eps)
    m, s = int(total // 60), int(total % 60)
    per_lead: Dict[str, float] = {}
    for e in eps:
        # Only the leads that met the criteria, with their qualifying value --
        # the same values the report's ST Event list shows.
        for lead, mm in (e.get('st_lead_mm') or {}).items():
            mm = float(mm)
            if lead not in per_lead:
                per_lead[lead] = mm
            else:
                per_lead[lead] = max(per_lead[lead], mm) if label == ST_ELEVATION else min(per_lead[lead], mm)

    def line(leads):
        return ", ".join(f"{lead}: {per_lead[lead]:+.1f}" if lead in per_lead else f"{lead}: -" for lead in leads)

    head = f"{len(eps)} ep, {m}m{s:02d}s (mm)"
    parts = [head, line(LIMB_LEADS), line(CHEST_LEADS)]
    return ("\n" if multiline else " | ").join(parts)
