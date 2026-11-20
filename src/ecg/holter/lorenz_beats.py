"""Lorenz (RR scatter) points of a recording, one per beat, with the class
the filter buttons of both Lorenz plots (REPLAY and the LORENZ tab) use.

A point is beat n: x = RR(n), the interval that ends on it, y = RR(n+1), the
interval to the very next beat -- always the real next beat, whatever filter
is on (the old plots filtered the RRs first and then paired each with the
next one of the same class: a PVC's RR with the RR of a PVC minutes later).

The beats are the PVC / PAC detector's (ectopic_beats_detector -- the same
beats as the report's VE / SVE, the Template cards and Full Disclosure), not
the live 5 s chunks' labels (a healthy human recording read 124 S there, 13
here). The class of beat n, first rule that holds:
  U      inside an auto "Unidentified" segment (no readable QRS: noise)
  X      inside an auto "Artifact" segment or an Asystole span
  Other  inside an auto Ventricular Fibrillation segment (waves, not beats)
  V      a PVC (detector 'V') or a beat of an auto Ventricular Tachycardia
  S      a PAC (detector 'S') or a beat of an auto SVT segment
  AF     a beat of an auto Atrial Fibrillation / Atrial Flutter segment
  P      a paced beat (detector 'P' -- no detector gives it yet)
  N      a normal beat (detector 'N') anywhere else (sinus rhythm, sinus
         brady / tachy, AV blocks, BBB: their QRS is the conducted one)
  Other  anything else (a 'Q' beat the detector could not classify)
Results are read from the saved analysis (recording_analysis); None when it
is not there, and the plots then keep their old live-chunk points.
"""

import os
from typing import List, Optional, Tuple

CLASS_NAMES = {
    'N': 'normal (sinus)', 'S': 'supraventricular', 'V': 'ventricular', 'P': 'paced',
    'AF': 'AF / atrial flutter', 'X': 'artifact', 'U': 'unidentified', 'Other': 'other',
}

_U_LABELS = ('Unidentified',)
_X_LABELS = ('Artifact', 'Asystole')
_VF_LABELS = ('Ventricular Fibrillation',)
_VT_LABELS = ('Ventricular Tachycardia',)
_SVT_LABELS = ('Supraventricular Tachycardia (SVT)',)
_AF_LABELS = ('Atrial Fibrillation', 'Atrial Flutter')


def _spans(segments, labels):
    out = []
    for s in segments or []:
        if str(s.get('label', '')) in labels:
            try:
                out.append((float(s['start_sec']), float(s['end_sec'])))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _inside(t: float, spans) -> bool:
    return any(a <= t <= b for a, b in spans)


def lorenz_points(session_dir: str) -> Optional[List[Tuple[float, float, float, str]]]:
    """[(t of beat n (s), RR(n) ms, RR(n+1) ms, class)] of every beat that
    has a beat before and after it; None when the recording's PVC / PAC beats
    or auto segments are not saved."""
    if not session_dir:
        return None
    path = os.path.join(session_dir, 'recording.ecgh')
    if not os.path.exists(path):
        return None
    try:
        from .auto_segment_arrthymia_detection import get_cached_asystole, get_cached_detection
        from .ectopic_beats_detector import get_cached_ectopic_beats
        beats = get_cached_ectopic_beats(path)
        segments = get_cached_detection(path)
        asystole = get_cached_asystole(path) or []
    except Exception as e:
        print(f"[Lorenz] Could not read the saved beats: {e}")
        return None
    if beats is None or segments is None:
        return None
    beats = sorted(beats, key=lambda b: float(b.get('timestamp', 0.0) or 0.0))
    u_spans = _spans(segments, _U_LABELS)
    x_spans = _spans(segments, _X_LABELS) + _spans(asystole, _X_LABELS)
    vf_spans = _spans(segments, _VF_LABELS)
    vt_spans = _spans(segments, _VT_LABELS)
    svt_spans = _spans(segments, _SVT_LABELS)
    af_spans = _spans(segments, _AF_LABELS)

    def classify(b) -> str:
        t = float(b.get('timestamp', 0.0) or 0.0)
        lab = str(b.get('label', 'N'))
        if _inside(t, u_spans):
            return 'U'
        if _inside(t, x_spans):
            return 'X'
        if _inside(t, vf_spans):
            return 'Other'
        if lab == 'V' or _inside(t, vt_spans):
            return 'V'
        if lab == 'S' or _inside(t, svt_spans):
            return 'S'
        if _inside(t, af_spans):
            return 'AF'
        if lab == 'P':
            return 'P'
        if lab == 'N':
            return 'N'
        return 'Other'

    times = [float(b.get('timestamp', 0.0) or 0.0) for b in beats]
    points = []
    for n in range(1, len(beats) - 1):
        rr = (times[n] - times[n - 1]) * 1000.0
        rr_next = (times[n + 1] - times[n]) * 1000.0
        if rr <= 0 or rr_next <= 0:
            continue
        points.append((times[n], rr, rr_next, classify(beats[n])))
    return points


def plot_range(values) -> Tuple[float, float]:
    """The plots' axis range: 5th..95th percentile of the shown RRs, at least
    1000 ms wide, 50 ms margin (as before)."""
    import numpy as np
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi - lo < 250:
        center = float(np.median(values))
        lo, hi = center - 500.0, center + 500.0
    return max(0.0, lo - 50.0), hi + 50.0


def plot_data(points, filter_key: str):
    """(xs, ys, classes, rr_points, x_range, empty_text) for LorenzCanvas.
    set_data: the points of the filter's class (all of them for "all").
    With none: empty lists and the text the canvas shows ("No ventricular
    beats") -- never the other classes' points in their place."""
    shown = [p for p in points if filter_key == 'all' or p[3] == filter_key]
    if not shown:
        name = CLASS_NAMES.get(filter_key, filter_key)
        return [], [], [], [], None, (f"No {name} beats" if filter_key != 'all' else "No RR data")
    xs = [p[1] for p in shown]
    ys = [p[2] for p in shown]
    return xs, ys, [p[3] for p in shown], [(p[0], p[1], p[3]) for p in shown], plot_range(xs + ys), None
