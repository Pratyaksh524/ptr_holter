"""A doctor's changes to the labels of auto segments (auto arrhythmia
marking), kept with the recording.

In Full Disclosure, the small arrow after an auto segment's label opens the
list of labels below; the one picked replaces the segment's label (and
colour) over the same start and end time. The change is saved in the
recording folder (relabeled_auto_segments.json) and applied wherever the auto
segments are shown or reported: Full Disclosure, AF ANALYSIS, EVENTS, the
Template cards, Lorenz and the report. A changed segment carries
'edited': True and 'original_label' (the detector's label); Full Disclosure
draws it with a dashed border and a pencil mark, and the report's Event
Timeline lists it as "Auto (Edited)".

The doctor decides: the change stays when "Auto Arrhythmia Detection" is run
again (the detector's result for the same file and code is the same, so each
change still finds its segment). The PVC / PAC detector keeps reading the
detector's own segments -- a relabel does not re-run beat detection.

This module is not imported by the detectors, so it is not part of their
cache fingerprint: changing it does not throw saved analyses away.
"""

import json
import os
from typing import Any, Dict, List, Optional

EDITS_FILENAME = 'relabeled_auto_segments.json'

# The labels a doctor can give an auto segment, grouped as in the menu, with
# the colour Full Disclosure draws each one in (its _add_auto_arrhythmia_
# segments targets -- the strip's colours, not the detector's).
LABEL_GROUPS = (
    ('VENTRICULAR', (
        ('Ventricular Fibrillation', '#FF3333'),
        ('Ventricular Tachycardia', '#E67E22'),
    )),
    ('AV BLOCK', (
        ('Third-degree AV Block', '#C0FF00'),
        ('Second-degree AV Block (Mobitz II)', '#FF00FF'),
        ('Second-degree AV Block (2:1)', '#DA70D6'),
        ('Second-degree AV Block (Mobitz I)', '#4169E1'),
        ('First-degree AV Block', '#FFD700'),
    )),
    ('ATRIAL / SUPRAVENTRICULAR', (
        ('Atrial Fibrillation', '#B36BFF'),
        ('Atrial Flutter', '#FF69B4'),
        ('Supraventricular Tachycardia (SVT)', '#FFA500'),
        ('Atrial Tachycardia', '#FF7F50'),
    )),
    ('SINUS / RATE', (
        ('Sinus Tachycardia', '#00FFFF'),
        ('Sinus Bradycardia', '#00BFFF'),
        ('Sinus Arrhythmia', '#F5DEB3'),
        ('Tachycardia', '#F08080'),
        ('Bradycardia', '#20B2AA'),
    )),
    ('BUNDLE BRANCH BLOCK', (
        ('Complete Right Bundle Branch Block', '#E0E0E0'),
        ('Complete Left Bundle Branch Block', '#1ABC9C'),
    )),
)
LABEL_COLORS = {label: color for _, items in LABEL_GROUPS for label, color in items}

_SAME_SEC = 0.05     # a saved change finds its segment by start / end within this


def _path(session_dir: str) -> str:
    return os.path.join(session_dir, EDITS_FILENAME)


def load_edits(session_dir: Optional[str]) -> List[Dict[str, Any]]:
    """[{start_sec, end_sec, original_label, label}] -- the detector's segment
    (its times and label) and the label the doctor gave it."""
    if not session_dir:
        return []
    try:
        with open(_path(session_dir), 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def save_edits(session_dir: str, edits: List[Dict[str, Any]]) -> None:
    path = _path(session_dir)
    try:
        if not edits:
            if os.path.exists(path):
                os.remove(path)
            return
        with open(path + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(edits, f, indent=2)
        os.replace(path + '.tmp', path)
    except OSError as e:
        print(f"[Auto segment edits] Could not save {EDITS_FILENAME}: {e}")


def _matches(seg_start: float, seg_end: float, seg_label: str, edit: Dict[str, Any]) -> bool:
    try:
        return (abs(seg_start - float(edit['start_sec'])) <= _SAME_SEC
                and abs(seg_end - float(edit['end_sec'])) <= _SAME_SEC
                and str(seg_label).strip().lower() == str(edit.get('original_label', '')).strip().lower())
    except (KeyError, TypeError, ValueError):
        return False


def apply_edits_to_segments(segments, session_dir: Optional[str]):
    """Copies of the detector's segments with the doctor's labels applied
    ('label', 'detected_rhythm', 'color'; plus 'edited' and
    'original_label'); the list itself when there is no change. None stays
    None."""
    if segments is None:
        return None
    edits = load_edits(session_dir)
    if not edits:
        return segments
    out = []
    for seg in segments:
        try:
            start, end = float(seg['start_sec']), float(seg['end_sec'])
        except (KeyError, TypeError, ValueError):
            out.append(seg)
            continue
        label = str(seg.get('label', ''))
        edit = next((e for e in edits if _matches(start, end, label, e)), None)
        if edit is None:
            out.append(seg)
            continue
        new = dict(seg)
        new_label = str(edit.get('label', label))
        new['label'] = new_label
        new['detected_rhythm'] = new_label
        new['color'] = LABEL_COLORS.get(new_label, seg.get('color'))
        new['edited'] = True
        new['original_label'] = label
        out.append(new)
    return out


def set_label(session_dir: str, segments: List[Dict[str, Any]], new_label: str) -> None:
    """Give `segments` (as apply_edits_to_segments returned them: a changed
    one carries its 'original_label') the label `new_label` and save it;
    back to the detector's own label removes the change."""
    edits = load_edits(session_dir)
    for seg in segments:
        try:
            start, end = float(seg['start_sec']), float(seg['end_sec'])
        except (KeyError, TypeError, ValueError):
            continue
        original = str(seg.get('original_label') or seg.get('label', ''))
        edits = [e for e in edits if not _matches(start, end, original, e)]
        if new_label.strip().lower() != original.strip().lower():
            edits.append({'start_sec': start, 'end_sec': end,
                          'original_label': original, 'label': new_label})
    save_edits(session_dir, edits)


def mark_edited_events(engine, segments) -> None:
    """Flag the engine's auto events that come from a changed segment
    ('edited', 'original_label'; convert_to_structured_events does not carry
    them over)."""
    changed = [s for s in (segments or []) if s.get('edited')]
    if not changed:
        return
    for ev in getattr(engine, '_structured_events', []) or []:
        if str(ev.get('source', '')) != 'Auto' or ev.get('end_timestamp') is None:
            continue
        ev_start = float(ev.get('timestamp', 0.0) or 0.0)
        ev_end = float(ev.get('end_timestamp', ev_start) or ev_start)
        ev_label = str(ev.get('label', ev.get('type', ''))).strip().lower()
        for seg in changed:
            if (str(seg.get('label', '')).strip().lower() == ev_label
                    and min(ev_end, float(seg['end_sec'])) - max(ev_start, float(seg['start_sec'])) > 0):
                ev['edited'] = True
                ev['original_label'] = seg.get('original_label', '')
                break


def apply_auto_segments_with_edits(engine, segments=None):
    """auto_segment_arrthymia_detection.apply_auto_segments_to_engine with
    the doctor's label changes applied first (and the events marked)."""
    from .auto_segment_arrthymia_detection import apply_auto_segments_to_engine, detect_arrhythmias
    reader = getattr(engine, '_reader', None)
    path = getattr(reader, 'path', None) if reader is not None else None
    session_dir = os.path.dirname(os.path.abspath(path)) if path else None
    if segments is None and reader is not None:
        segments = detect_arrhythmias(reader)
    edited = apply_edits_to_segments(segments, session_dir)
    added = apply_auto_segments_to_engine(engine, segments=edited)
    mark_edited_events(engine, edited)
    return added


def edited_segments_for(path: Optional[str]):
    """get_cached_detection(path) with the doctor's label changes applied --
    for the views that show the auto segments; None when not saved."""
    from .auto_segment_arrthymia_detection import get_cached_detection
    if not path:
        return None
    return apply_edits_to_segments(get_cached_detection(path), os.path.dirname(os.path.abspath(path)))
