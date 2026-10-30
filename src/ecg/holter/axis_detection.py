"""
QRS axis for Holter recordings: the dominant frontal-plane axis, its range,
the time spent in each axis category and the Axis Deviation events. VALUES
shown in the Overview table and the report -- not auto arrhythmia marks.

Independent of the auto arrhythmia detection. Run with the ischemia findings
(ischemia_detection.compute_ischemia, same cache and background job). It
reads the recording itself (ECGHFileReader.read_range, the raw 12 leads the
Full Disclosure view shows) and filters every lead the way the Full
Disclosure strip displays it (panels/holter_widgets.ECGStripCanvas.set_data):
50 / 60 Hz notches (_remove_mains), 0.5 Hz DFT high-pass (zero-phase), 3-point
median, Gaussian low-pass 35 Hz twice -- with 2 s of real neighbouring
samples on each side, as the strip pads. Only the beats inside the frame
count.

Frame by frame (detect_qrs_axis):
  1. The recording is read in 10 s frames.
  2. R-peaks on lead II; beats aligned to each other (the ischemia
     module's beat alignment). Skipped frames: fewer than 4 beats, or beats
     that don't share one shape (median correlation < 0.9 with the frame's
     template: ectopics, noise). Irregular rhythm (Atrial Fibrillation) is
     NOT skipped: the QRS axis does not need a P wave. A wide QRS (>= 120 ms,
     bundle branch block) is measured and flagged.
  3. QRS onset / J point on the median beat (combined slope of all leads,
     ischemia_detection._qrs_bounds).
  4. Net QRS area (onset -> J point, baseline = 20 ms before the onset) in
     lead I (0 deg) and aVF (+90 deg): axis = atan2(2/sqrt(3) x area aVF,
     area I), -180 to +180 deg -- an augmented lead is sqrt(3)/2 of the
     frontal vector's projection (plain atan2(aVF, I) reads up to ~4 deg
     towards 0 / 180). A frame whose QRS area is too small in both leads has no
     measurable axis. Neither has a frame whose net-area axis is more than
     60 deg from its maximal QRS vector (R +/- 40 ms, baseline = median of
     the beat outside the QRS): the baseline before the QRS is not clean
     there (atrial flutter waves, noise) and the net area would give a
     false axis.
  5. Categories: Normal -30 to +90, Left Axis Deviation -30 to -90, Right
     Axis Deviation +90 to +180, Extreme Axis -90 to -180.
  6. Summary over the measured frames: dominant axis = circular mean,
     range = 5th - 95th percentile around it (posture and breathing move the
     axis 10-30 deg), % of measured time per category.
  7. Axis Deviation events: consecutive frames of one deviated category, a
     gap of up to one frame (10 s) joined, kept when the qualifying frames
     add up to >= 30 s. Per event: median axis and its lowest / highest
     frame.
"""

from typing import Any, Dict, List, Optional

import numpy as np

AXIS_NORMAL = 'Normal'
AXIS_LAD = 'Left Axis Deviation'
AXIS_RAD = 'Right Axis Deviation'
AXIS_EXTREME = 'Extreme Axis'
AXIS_CATEGORIES = (AXIS_NORMAL, AXIS_LAD, AXIS_RAD, AXIS_EXTREME)
AXIS_SHORT = {AXIS_NORMAL: 'Normal', AXIS_LAD: 'LAD', AXIS_RAD: 'RAD', AXIS_EXTREME: 'Extreme'}
AXIS_NORMAL_RANGE = '-30° to +90°'

_CTX_SEC = 2.0                   # real samples on each side, as the strip pads
_BASELINE_SEC = 0.02             # baseline: this long before the QRS onset
_MIN_AREA_MV_MS = 2.0            # |net area| of I and aVF combined below this: no axis
_MAX_QRS_MS = 120.0              # at or above: wide QRS (flagged, still measured)
_MAX_VECTOR_SEC = 0.04           # maximal QRS vector searched within R +/- this
# net-area axis vs maximal vector further apart than this: frame not measured.
# Sinus / AF / SVT / AV block frames of the recordings: <= ~15 deg apart;
# atrial flutter (F waves under the baseline): ~160 deg.
_MAX_VECTOR_DIFF_DEG = 60.0
_EVENT_GAP_SEC = 10.0            # one frame
_MIN_EVENT_SEC = 30.0
_RANGE_PCT = (5.0, 95.0)
_MAX_EVENTS_SHOWN = 5


def axis_category(deg: float) -> str:
    """Axis category of one angle (-180 to +180 deg)."""
    if -30.0 <= deg <= 90.0:
        return AXIS_NORMAL
    if -90.0 <= deg < -30.0:
        return AXIS_LAD
    if deg > 90.0:
        return AXIS_RAD
    return AXIS_EXTREME


def _wrap(deg):
    """Angle(s) to -180 .. +180 deg."""
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def _strip_filter(x: np.ndarray, fs: float) -> np.ndarray:
    """One lead, already padded with real neighbouring samples, filtered as
    the Full Disclosure strip displays it."""
    from ..ecg_filters import apply_dft_filter
    from ..enhanced_filter import gaussian_lowpass
    try:
        from .panels.holter_widgets import _remove_mains
    except Exception:  # pragma: no cover - holter_widgets needs Qt
        _remove_mains = None
    x = np.asarray(x, dtype=float)
    if x.size > 30:
        if _remove_mains is not None:
            x = np.asarray(_remove_mains(x, float(fs)), dtype=float)
        x = np.asarray(apply_dft_filter(x, float(fs), "0.5"), dtype=float)
    if x.size >= 3:
        p = np.pad(x, (1, 1), mode='edge')
        x = np.median(np.stack((p[:-2], p[1:-1], p[2:]), axis=0), axis=0)
    x = gaussian_lowpass(x, 35.0, float(fs))
    return gaussian_lowpass(x, 35.0, float(fs))


def _read_frame(reader, t: float, t_end: float, total: float, fs: float):
    """The frame [t, t_end) with _CTX_SEC real samples on each side (mirrored
    only where the recording starts / ends), every lead through
    _strip_filter: (data, frame start index, frame end index), or None when
    too short."""
    p0 = max(0.0, t - _CTX_SEC)
    p1 = min(total, t_end + _CTX_SEC)
    raw = reader.read_range(p0, p1)
    if raw is None or raw.shape[1] < int(fs * 5.0):
        return None
    raw = np.asarray(raw, dtype=float)
    want = int(_CTX_SEC * fs)
    lead_in = int(t * fs) - int(p0 * fs)
    frame_n = int(t_end * fs) - int(t * fs)
    left = max(0, want - lead_in)
    right = max(0, want - (raw.shape[1] - lead_in - frame_n))
    if left or right:
        left, right = min(left, raw.shape[1] - 1), min(right, raw.shape[1] - 1)
        raw = np.pad(raw, ((0, 0), (left, right)), mode='reflect')
    data = np.vstack([_strip_filter(x, fs) for x in raw])
    return data, left + lead_in, left + lead_in + frame_n


def _frame_axis(data: np.ndarray, lo: int, hi: int, lead_names: List[str], fs: float, pan_tompkins):
    """(axis deg, wide QRS) of the beats with their R-peak in data[:, lo:hi],
    or None when it can't be measured."""
    from .ischemia_detection import _ADC_TO_MV, _aligned_beats, _qrs_bounds
    try:
        i_idx, avf_idx = lead_names.index('I'), lead_names.index('aVF')
    except ValueError:
        return None
    ref_idx = lead_names.index('II') if 'II' in lead_names else i_idx
    peaks = [int(p) for p in pan_tompkins(np.asarray(data[ref_idx], dtype=float), fs=fs)
             if lo <= int(p) < hi]
    built = _aligned_beats(data, peaks, ref_idx, fs)
    if built is None:
        return None
    beats, r_idx, corr = built
    if corr < 0.9:
        return None
    med = np.median(beats, axis=0)
    bounds = _qrs_bounds(med, r_idx, fs)
    if bounds is None:
        return None
    onset, j_point = bounds
    base_lo = onset - int(_BASELINE_SEC * fs)
    if base_lo < 0:
        return None
    ms_per_sample = 1000.0 / fs

    def net_area(idx):
        x = med[idx]
        return float(np.sum(x[onset:j_point + 1] - np.mean(x[base_lo:onset]))) * _ADC_TO_MV * ms_per_sample

    a_i, a_avf = net_area(i_idx), net_area(avf_idx)
    if np.hypot(a_i, a_avf) < _MIN_AREA_MV_MS:
        return None
    # aVF = II - I/2 is (sqrt 3)/2 of the frontal vector's +90 deg projection
    # (Einthoven triangle), lead I its full 0 deg projection
    k_avf = 2.0 / np.sqrt(3.0)
    axis = float(np.degrees(np.arctan2(a_avf * k_avf, a_i)))
    # Consistency check: the maximal QRS vector (R +/- _MAX_VECTOR_SEC,
    # baseline = median of the beat outside the QRS) hardly depends on the
    # baseline; when the net-area axis points elsewhere, the baseline / QRS
    # window is not clean (flutter waves, noise) -- no axis for this frame.
    outside = np.ones(med.shape[1], dtype=bool)
    outside[onset:j_point + 1] = False
    w = int(_MAX_VECTOR_SEC * fs)
    v_i = med[i_idx][r_idx - w:r_idx + w] - np.median(med[i_idx][outside])
    v_f = (med[avf_idx][r_idx - w:r_idx + w] - np.median(med[avf_idx][outside])) * k_avf
    m = int(np.argmax(np.hypot(v_i, v_f)))
    max_vector = float(np.degrees(np.arctan2(v_f[m], v_i[m])))
    if abs(float(_wrap(axis - max_vector))) > _MAX_VECTOR_DIFF_DEG:
        return None
    return axis, bool((j_point - onset) * ms_per_sample >= _MAX_QRS_MS)


def _summary(axes: np.ndarray):
    """(dominant deg, (low, high) deg) of the measured frames' axes."""
    rad = np.radians(axes)
    dominant = float(np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))
    spread = _wrap(axes - dominant)
    lo, hi = np.percentile(spread, _RANGE_PCT)
    return dominant, (float(dominant + lo), float(dominant + hi))


def detect_qrs_axis(reader) -> Dict[str, Any]:
    """QRS axis of the recording (see the module docstring):
    {'dominant': deg | None, 'category': str | None, 'range': [low, high] | None,
     'sd': spread around the dominant axis (deg),
     'pct': {category: % of measured time}, 'wide_pct': % of measured time
     with a wide QRS, 'events': [{start_sec, end_sec, label, axis, axis_min,
     axis_max}], 'frames': n, 'measured': n, 'duration_sec': s,
     'start_time': epoch}."""
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:  # pragma: no cover
        from ..pan_tompkins import pan_tompkins
    from .ischemia_detection import _chunks

    fs = float(getattr(reader, 'fs', 500) or 500)
    total = float(getattr(reader, 'duration_sec', 0.0) or 0.0)
    lead_names = list(getattr(reader, 'lead_names', []) or [])
    out = {'dominant': None, 'category': None, 'range': None, 'pct': {}, 'wide_pct': 0.0,
           'events': [], 'frames': 0, 'measured': 0, 'duration_sec': total,
           'start_time': float(getattr(reader, 'start_time', 0.0) or 0.0)}
    if total <= 0 or not lead_names:
        return out

    measured = []          # (start, end, axis, wide)
    for t, t_end in _chunks(total):
        out['frames'] += 1
        try:
            frame = _read_frame(reader, t, t_end, total, fs)
            if frame is None:
                continue
            data, lo, hi = frame
            res = _frame_axis(data, lo, hi, lead_names[:data.shape[0]], fs, pan_tompkins)
            if res is not None:
                measured.append((t, t_end, res[0], res[1]))
        except Exception as e:
            print(f"[Axis Detect] Frame error {t:.1f}s: {e}")
    out['measured'] = len(measured)
    if not measured:
        return out

    axes = np.array([m[2] for m in measured])
    secs = np.array([m[1] - m[0] for m in measured])
    dominant, rng = _summary(axes)
    out['dominant'] = round(dominant, 1)
    out['category'] = axis_category(dominant)
    out['range'] = [round(rng[0], 1), round(rng[1], 1)]
    out['sd'] = round(float(np.std(_wrap(axes - dominant))), 1)
    cats = [axis_category(a) for a in axes]
    out['pct'] = {c: round(float(secs[[k == c for k in cats]].sum() / secs.sum() * 100.0), 1)
                  for c in AXIS_CATEGORIES}
    out['wide_pct'] = round(float(secs[[m[3] for m in measured]].sum() / secs.sum() * 100.0), 1)

    # Axis Deviation events: same deviated category, a gap of up to one frame joined.
    events = []
    for (s_sec, e_sec, axis, _wide), cat in zip(measured, cats):
        if cat == AXIS_NORMAL:
            continue
        ev = events[-1] if events else None
        if ev is None or ev['label'] != cat or s_sec - ev['end_sec'] > _EVENT_GAP_SEC + 1e-6:
            ev = {'start_sec': s_sec, 'end_sec': e_sec, 'label': cat, 'qual_sec': 0.0, 'axes': []}
            events.append(ev)
        ev['end_sec'] = e_sec
        ev['qual_sec'] += e_sec - s_sec
        ev['axes'].append(axis)
    for ev in events:
        if ev['qual_sec'] < _MIN_EVENT_SEC - 1e-6:
            continue
        a = np.array(ev['axes'])
        # Extreme Axis straddles +/-180: median / range around the circular mean
        centre, (lo, hi) = _summary(a)
        out['events'].append({
            'start_sec': float(ev['start_sec']), 'end_sec': float(ev['end_sec']),
            'label': ev['label'],
            'axis': round(float(_wrap(centre + np.median(_wrap(a - centre)))), 1),
            'axis_min': round(float(centre + np.min(_wrap(a - centre))), 1),
            'axis_max': round(float(centre + np.max(_wrap(a - centre))), 1),
        })
    return out


def _deg(v: float) -> str:
    return f"{float(_wrap(v)):+.0f}°"


def _clock(sec: float, start: float) -> str:
    if start > 0:
        from datetime import datetime
        return datetime.fromtimestamp(start + sec).strftime('%H:%M:%S')
    return f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}:{int(sec % 60):02d}"


def _dur(sec: float) -> str:
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


def format_axis_summary(axis: Optional[Dict[str, Any]], multiline: bool = False) -> str:
    """Text for the Overview "QRS Axis" row: dominant axis and category,
    range, % of time outside Normal. None (not calculated yet) -> "-"."""
    if axis is None:
        return "-"
    if axis.get('dominant') is None:
        if float(axis.get('duration_sec', 0.0) or 0.0) <= 0:
            return "None"
        return "Not measurable (noise / ectopics)"
    sep = "\n" if multiline else " | "
    lo, hi = axis.get('range') or (axis['dominant'], axis['dominant'])
    # the range runs from lo up to hi; past +180 it continues at -180
    via = " (via ±180°)" if lo < -180.0 or hi > 180.0 else ""
    lines = [f"{_deg(axis['dominant'])} ({axis['category']})",
             f"Range {_deg(lo)} to {_deg(hi)}{via}"]
    pct = axis.get('pct') or {}
    off = [f"{AXIS_SHORT[c]} {pct[c]:.1f}%" for c in AXIS_CATEGORIES[1:] if pct.get(c, 0.0) > 0]
    if off:
        lines.append("Time: " + ", ".join(off))
    if float(axis.get('wide_pct', 0.0) or 0.0) > 0:
        lines.append(f"Wide QRS {axis['wide_pct']:.1f}% of time")
    return sep.join(lines)


def format_axis_events(axis: Optional[Dict[str, Any]], multiline: bool = False) -> str:
    """Text for the Overview "Axis Deviation" row: the Axis Deviation events
    (start - end, duration, category, median axis); the longest
    _MAX_EVENTS_SHOWN in time order, then "+N more"."""
    if axis is None:
        return "-"
    events = axis.get('events') or []
    if not events:
        return "None" if axis.get('dominant') is not None else "-"
    sep = "\n" if multiline else " | "
    start = float(axis.get('start_time', 0.0) or 0.0)
    total = sum(e['end_sec'] - e['start_sec'] for e in events)
    longest = sorted(events, key=lambda e: e['start_sec'] - e['end_sec'])[:_MAX_EVENTS_SHOWN]
    lines = [f"{len(events)} event{'s' if len(events) != 1 else ''}, {_dur(total)}"]
    for i, e in enumerate(sorted(longest, key=lambda e: e['start_sec']), 1):
        lines.append(f"{i}) {_clock(e['start_sec'], start)} - {_clock(e['end_sec'], start)} "
                     f"({_dur(e['end_sec'] - e['start_sec'])}) {AXIS_SHORT[e['label']]} {_deg(e['axis'])}")
    if len(events) > len(longest):
        lines.append(f"+{len(events) - len(longest)} more (report)")
    return sep.join(lines)
