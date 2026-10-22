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
  5. Criteria (>= 2 contiguous leads of one territory: inferior II/III/aVF,
     lateral I/aVL/V5/V6, anterior V1-V4):
       ST Elevation : >= 0.1 mV (V2, V3: >= 0.2 mV).
       ST Depression: <= -0.1 mV, horizontal or downsloping (not more than
                      0.05 mV above the J+20 level).
     A chunk with both is labeled ST Elevation (the depression is
     reciprocal).
  6. Episodes: consecutive positive chunks of one label, joined across gaps
     up to _MERGE_GAP_SEC, kept when they last >= _MIN_EPISODE_SEC (the
     Holter "1 mm x 1 min" convention).
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.signal import savgol_filter

ST_ELEVATION = 'ST Elevation'
ST_DEPRESSION = 'ST Depression'
ST_COLORS = {
    ST_ELEVATION: '#00E5A0',   # mint green
    ST_DEPRESSION: '#00BFFF',  # deep sky blue
}
ST_LABELS = (ST_ELEVATION, ST_DEPRESSION)

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

_ELEVATION_MV = 0.10
_ELEVATION_V2_V3_MV = 0.20
_DEPRESSION_MV = -0.10
_UPSLOPE_LIMIT_MV = 0.05

_TERRITORIES = {
    'inferior': ('II', 'III', 'aVF'),
    'lateral': ('I', 'aVL', 'V5', 'V6'),
    'anterior': ('V1', 'V2', 'V3', 'V4'),
}

_SKIP_RHYTHMS = (
    'ventricular fibrillation',
    'ventricular tachycardia',
    'bundle branch block',
)
_MAX_QRS_MS = 120.0


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


def _chunk_st(data: np.ndarray, lead_names: List[str], fs: float, pan_tompkins,
              ref_idx: int, skip_leads: set):
    """ST deviation (mV) at J+20 ms and at the ST point (J+40, J+30 ms above
    120 bpm) for every lead of one chunk, or None when it can't be measured."""
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
    out = {}
    for li, name in enumerate(lead_names):
        if name in skip_leads or li >= med.shape[0]:
            continue
        x = med[li]
        baseline = float(np.mean(x[base_lo:base_hi]))
        st_early = float(np.mean(x[early - half:early + half])) - baseline
        st_level = float(np.mean(x[st_point - half:st_point + half])) - baseline
        out[name] = (st_early * _ADC_TO_MV, st_level * _ADC_TO_MV)
    return out, hr


def _chunk_label(st: Dict[str, tuple]):
    """('ST Elevation' | 'ST Depression' | None, leads, worst mV)."""
    elevated, depressed = [], []
    for name, (st_early, st_level) in st.items():
        if name == 'aVR':
            continue
        limit = _ELEVATION_V2_V3_MV if name in ('V2', 'V3') else _ELEVATION_MV
        if st_level >= limit:
            elevated.append((name, st_level))
        if st_level <= _DEPRESSION_MV and (st_level - st_early) <= _UPSLOPE_LIMIT_MV:
            depressed.append((name, st_level))

    def territory_hit(hits):
        names = {n for n, _ in hits}
        leads = []
        for group in _TERRITORIES.values():
            in_group = [n for n in group if n in names]
            if len(in_group) >= 2:
                leads.extend(in_group)
        return leads

    elev_leads = territory_hit(elevated)
    if elev_leads:
        worst = max(v for n, v in elevated if n in elev_leads)
        return ST_ELEVATION, elev_leads, worst
    dep_leads = territory_hit(depressed)
    if dep_leads:
        worst = min(v for n, v in depressed if n in dep_leads)
        return ST_DEPRESSION, dep_leads, worst
    return None, [], 0.0


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

    # Positive chunks: (start, end, label, leads, worst mV)
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
                    label, leads, worst = _chunk_label(st)
                    if label is not None:
                        positives.append((t, t_end, label, leads, worst))
            except Exception as e:
                print(f"[Ischemia Detect] Chunk error {t:.1f}s: {e}")
            t = t_end

    # Episodes: same label, gaps up to _MERGE_GAP_SEC joined.
    episodes = []
    for s, e, label, leads, worst in positives:
        if episodes and episodes[-1]['label'] == label and s - episodes[-1]['end_sec'] <= _MERGE_GAP_SEC:
            ep = episodes[-1]
            ep['end_sec'] = e
            ep['st_leads'] = sorted(set(ep['st_leads']) | set(leads))
            ep['st_mv'] = max(ep['st_mv'], worst) if label == ST_ELEVATION else min(ep['st_mv'], worst)
            continue
        episodes.append({'start_sec': s, 'end_sec': e, 'label': label,
                         'st_leads': sorted(set(leads)), 'st_mv': worst})

    out = []
    for ep in episodes:
        if ep['end_sec'] - ep['start_sec'] < _MIN_EPISODE_SEC:
            continue
        s_str, e_str = make_time_strs(ep['start_sec'], ep['end_sec'])
        out.append({
            'start_sec': float(ep['start_sec']),
            'end_sec': float(ep['end_sec']),
            'label': ep['label'],
            'color': ST_COLORS[ep['label']],
            'start_time_str': s_str,
            'end_time_str': e_str,
            'detected_rhythm': ep['label'],
            'st_leads': ep['st_leads'],
            'st_mv': round(float(ep['st_mv']), 3),
        })
    return out


def format_st_summary(episodes: Optional[List[Dict[str, Any]]], label: str) -> str:
    """One-line text for the Overview / Replay "ST Elevation" / "ST Depression"
    rows. episodes: every detected segment (any label) or None when detection
    hasn't run -- then "-". No episode of this label: "None". Otherwise
    "<count> ep, <total duration>, <worst mV> (<leads>)"."""
    if episodes is None:
        return "-"
    eps = [e for e in episodes if e.get('label') == label]
    if not eps:
        return "None"
    total = sum(max(0.0, float(e['end_sec']) - float(e['start_sec'])) for e in eps)
    m, s = int(total // 60), int(total % 60)
    values = [float(e.get('st_mv', 0.0) or 0.0) for e in eps]
    worst = max(values) if label == ST_ELEVATION else min(values)
    leads = sorted({lead for e in eps for lead in (e.get('st_leads') or [])})
    lead_str = f" ({','.join(leads)})" if leads else ""
    return f"{len(eps)} ep, {m}m{s:02d}s, {worst:+.2f} mV{lead_str}"
