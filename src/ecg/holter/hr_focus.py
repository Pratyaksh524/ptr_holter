"""Heart-rate focus points of a recording -- the EDIT STRIPS cards: Max / Min
HR and Sinus Max / Min HR, each with the moment (s) it happens.

Max / Min: the report's page-1 Heart Rate (report_generator._hr_min_avg_max):
the Full Disclosure strip's own beats, 60 / RR of every beat pair, the pairs
inside Ventricular Fibrillation / Unidentified / Asystole left out -- the
same value at the same time as the report and the strip.

Sinus Max / Min: sinus beats only. An interval counts when
  - both of its beats are the PVC / PAC detector's 'N' and next to each
    other (no PVC, PAC or unclassified beat between them),
  - neither beat is inside a non-sinus auto segment (NON_SINUS_LABELS, with
    the doctor's label changes) or an Asystole span,
  - it is within SINUS_STEP_FRACTION of the sinus interval before it: a
    dropped beat, a missed or an extra beat is not the sinus rate. On
    2026-11-12_13-54-02 a wide beat 326 ms early at 14:07:59.21 and the
    2248 ms gap after it read 184 / 27 bpm, and every 8th beat dropped
    (1500 ms) at 13:57:51-13:58:21 read 40 bpm.
Was (summary_utils.derive_hr_focus_summary): the live 5 s chunk with the
highest / lowest mean HR among chunks whose live labels looked sinus -- 50 bpm
at 14:08:42 on that recording, inside Ventricular Fibrillation.

Everything is read from the saved analysis (recording_analysis); a value
whose source is not saved is None and the caller keeps its old one.
"""

import os
from typing import Dict, Optional, Tuple

# Auto segment labels that are not sinus rhythm (auto_segment_edits.LABEL_GROUPS
# names). Sinus rhythm with 1st-degree AV block or a bundle branch block,
# sinus brady / tachy / arrhythmia stay sinus. Plain "Tachycardia" /
# "Bradycardia" are the detector's labels when it could not confirm P waves.
NON_SINUS_LABELS = (
    'Ventricular Fibrillation', 'Ventricular Tachycardia',
    'Third-degree AV Block', 'Second-degree AV Block (Mobitz II)',
    'Second-degree AV Block (2:1)', 'Second-degree AV Block (Mobitz I)',
    'Atrial Fibrillation', 'Atrial Flutter', 'Supraventricular Tachycardia (SVT)',
    'Atrial Tachycardia', 'Tachycardia', 'Bradycardia',
    'Unidentified', 'Artifact',
)

# An N-N interval more than this far from the one before it is not counted.
SINUS_STEP_FRACTION = 0.20

# Pairs the report's HR leaves out (report_generator, page-1 Heart Rate).
_REPORT_HR_SKIP_LABELS = ('Unidentified', 'Ventricular Fibrillation')

Point = Tuple[float, float]          # (bpm, time of the pair's first beat, s)


def _spans(items, labels=None):
    out = []
    for s in items or []:
        if labels is not None and str(s.get('label', '')) not in labels:
            continue
        try:
            out.append((float(s['start_sec']), float(s['end_sec'])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _sinus_extremes(beats, bad_spans) -> Tuple[Optional[Point], Optional[Point]]:
    """(fastest, slowest) sinus N-N interval as (bpm, start s); None when
    there is none."""
    def inside(t):
        return any(a <= t <= b for a, b in bad_spans)

    beats = sorted(beats, key=lambda b: float(b.get('timestamp', 0.0) or 0.0))
    fastest = slowest = None
    prev_rr = None                     # the previous counted-or-candidate sinus interval
    for b0, b1 in zip(beats, beats[1:]):
        t0 = float(b0.get('timestamp', 0.0) or 0.0)
        t1 = float(b1.get('timestamp', 0.0) or 0.0)
        rr = t1 - t0
        sinus = (b0.get('label') == 'N' and b1.get('label') == 'N' and rr > 0
                 and not inside(t0) and not inside(t1))
        if not sinus:
            prev_rr = None             # the next interval has no sinus one before it
            continue
        if prev_rr is not None and abs(rr - prev_rr) <= SINUS_STEP_FRACTION * prev_rr:
            bpm = 60.0 / rr
            if fastest is None or bpm > fastest[0]:
                fastest = (bpm, t0)
            if slowest is None or bpm < slowest[0]:
                slowest = (bpm, t0)
        prev_rr = rr
    return fastest, slowest


def hr_focus_points(session_dir: str) -> Dict[str, Optional[Point]]:
    """{'max_hr', 'min_hr', 'sinus_max_hr', 'sinus_min_hr'} -> (bpm, s) or
    None (its source not saved / nothing to read)."""
    out: Dict[str, Optional[Point]] = {'max_hr': None, 'min_hr': None,
                                       'sinus_max_hr': None, 'sinus_min_hr': None}
    path = os.path.join(session_dir or '', 'recording.ecgh')
    if not session_dir or not os.path.exists(path):
        return out
    try:
        from .auto_segment_arrthymia_detection import get_cached_asystole, get_cached_detection
        from .auto_segment_edits import apply_edits_to_segments
        segments = apply_edits_to_segments(get_cached_detection(path), session_dir) or []
        asystole = get_cached_asystole(path) or []
    except Exception as e:
        print(f"[HR focus] auto segments unavailable: {e}")
        segments, asystole = [], []

    # Max / Min: exactly the report's page-1 numbers
    try:
        from .report_generator import _full_disclosure_beat_times, _full_disclosure_hr_pairs, _hr_min_avg_max
        fd_beats = _full_disclosure_beat_times(session_dir, compute=False)
        if fd_beats:
            skip = _spans(segments, _REPORT_HR_SKIP_LABELS) + _spans(asystole)
            hr = _hr_min_avg_max(_full_disclosure_hr_pairs(fd_beats, skip))
            if hr:
                out['max_hr'] = (float(hr['max']), float(hr['max_t']))
                out['min_hr'] = (float(hr['min']), float(hr['min_t']))
    except Exception as e:
        print(f"[HR focus] Full Disclosure beats unavailable: {e}")

    # Sinus Max / Min: the detector's sinus N-N intervals
    try:
        from .ectopic_beats_detector import get_cached_ectopic_beats
        beats = get_cached_ectopic_beats(path)
        if beats:
            bad = _spans(segments, NON_SINUS_LABELS) + _spans(asystole)
            out['sinus_max_hr'], out['sinus_min_hr'] = _sinus_extremes(beats, bad)
    except Exception as e:
        print(f"[HR focus] PVC / PAC beats unavailable: {e}")
    return out
