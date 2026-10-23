"""
ecg/holter/report_generator.py
================================
Generates a clinical Holter report PDF from a completed recording session.

Reads:  metrics.jsonl (produced by HolterAnalysisWorker)
        recording.ecgh (for representative ECG strip)
Writes: holter_report.pdf

Reuses existing infrastructure:
  - ecg_report_generator.py utilities (patient info formatting, PDF setup)
  - matplotlib for charts (already a dependency)
"""

import os
import sys
import json
import time
import traceback
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from .session_store import load_events, load_metrics

# Add project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _detector_owns_label(label: str) -> bool:
    """Labels decided only by the waveform auto-segment detector. For these,
    the live recording pipeline's stored tally (summary['arrhythmia_counts'])
    is not used -- it has logged e.g. Sinus Bradycardia at a steady 60 bpm,
    which showed up here as episodes with no occurring time."""
    lower = str(label or '').lower()
    return lower in {
        'ventricular fibrillation', 'ventricular tachycardia', 'atrial fibrillation', 'atrial flutter',
        'supraventricular tachycardia (svt)', 'sinus bradycardia', 'sinus tachycardia',
    } or 'av block' in lower or 'bundle branch block' in lower


def _load_auto_segment_events(session_dir: str) -> List[Dict[str, object]]:
    """Run the same waveform auto-segment detector used by Full Disclosure.

    Only does anything if the user actually clicked "Auto Arrhythmia
    Detection" in Full Disclosure for this recording (see
    is_auto_marking_enabled / set_auto_marking_enabled) -- otherwise the
    report shows plain NSR, same as Full Disclosure does before that click.
    """
    ecgh_path = os.path.join(session_dir, 'recording.ecgh')
    if not os.path.exists(ecgh_path):
        return []
    try:
        from .auto_segment_arrthymia_detection import is_auto_marking_enabled
        if not is_auto_marking_enabled(session_dir):
            return []
    except Exception as exc:
        print(f"[HolterReport] Could not check auto-marking flag: {exc}")
        return []
    try:
        from .file_format import ECGHFileReader
        from .auto_segment_arrthymia_detection import detect_arrhythmias
        reader = ECGHFileReader(ecgh_path)
        events = []

        # Load any auto segments that the user deleted in Full Disclosure
        deleted_segments = []
        del_path = os.path.join(session_dir, 'deleted_auto_segments.json')
        if os.path.exists(del_path):
            try:
                with open(del_path, 'r', encoding='utf-8') as df:
                    deleted_segments = json.load(df)
            except Exception as e:
                print(f"[HolterReport] Error reading deleted_auto_segments: {e}")

        def _is_deleted(start_s, end_s, lbl):
            lbl_lower = lbl.strip().lower()
            for ds in deleted_segments:
                ds_start = float(ds.get('start_sec', 0.0) or 0.0)
                ds_end = float(ds.get('end_sec', ds_start) or ds_start)
                ds_lbl = str(ds.get('label', '')).strip().lower()
                overlap = max(0.0, min(end_s, ds_end) - max(start_s, ds_start))
                if overlap > 0.5 or (abs(start_s - ds_start) < 1.0 and abs(end_s - ds_end) < 1.0):
                    if ds_lbl in lbl_lower or lbl_lower in ds_lbl:
                        return True
            return False

        from .ischemia_detection import ST_LABELS
        for segment in detect_arrhythmias(reader):
            start_sec = float(segment.get('start_sec', 0.0) or 0.0)
            end_sec = float(segment.get('end_sec', 0.0) or 0.0)
            rhythm_lbl = str(segment.get('detected_rhythm', segment.get('label', 'Event')))
            # ST Elevation / Depression episodes have their own ST ANALYSIS
            # page (_load_st_episodes); they are not arrhythmia rows.
            if rhythm_lbl in ST_LABELS:
                continue
            if _is_deleted(start_sec, end_sec, rhythm_lbl):
                continue
            events.append({
                'timestamp': start_sec,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'end_timestamp': end_sec,
                'label': rhythm_lbl,
                'event_type': rhythm_lbl,
                'start_time_str': segment.get('start_time_str', ''),
                'end_time_str': segment.get('end_time_str', ''),
                'duration': max(0.0, end_sec - start_sec),
                'source': 'Auto',
                'confidence': 1.0,
            })
        return events
    except Exception as exc:
        print(f"[HolterReport] Could not load auto segments: {exc}")
        return []


def _load_ischemia(session_dir: str) -> Dict[str, List[Dict[str, object]]]:
    """{'st': ST episodes, 't': T Inversion episodes} of this recording from
    ischemia_detection (reads the recording itself; nothing to do with the
    auto arrhythmia detection). Uses the cached result when there is one."""
    ecgh_path = os.path.join(session_dir, 'recording.ecgh')
    if not os.path.exists(ecgh_path):
        return {'st': [], 't': []}
    try:
        from .ischemia_detection import compute_ischemia, get_cached_ischemia
        result = get_cached_ischemia(ecgh_path)
        if result is None:
            result = compute_ischemia(ecgh_path)
        return result
    except Exception as exc:
        print(f"[HolterReport] Could not load ST / T Inversion episodes: {exc}")
        return {'st': [], 't': []}


def _load_st_episodes(session_dir: str) -> List[Dict[str, object]]:
    """ST Elevation / ST Depression episodes, for the ST ANALYSIS page."""
    episodes = _load_ischemia(session_dir).get('st', [])
    return sorted(episodes, key=lambda s: float(s.get('start_sec', 0.0) or 0.0))


def _load_t_episodes(session_dir: str) -> List[Dict[str, object]]:
    """T Inversion episodes, for the T wave section of the ST ANALYSIS page."""
    episodes = _load_ischemia(session_dir).get('t', [])
    return sorted(episodes, key=lambda s: float(s.get('start_sec', 0.0) or 0.0))


def generate_holter_report(session_dir: str,
                            patient_info: dict,
                            summary: dict,
                            settings_manager=None) -> str:
    """
    Main entry point. Generates holter_report.pdf in session_dir.
    Returns path to generated PDF.
    """
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib import colors
        from reportlab.lib.units import mm, cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image, PageBreak,
                                         HRFlowable)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        _has_reportlab = True
    except ImportError:
        _has_reportlab = False

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(session_dir, f'holter_report_{timestamp_str}.pdf')

    if not _has_reportlab:
        # Fallback: text report
        _generate_text_report(session_dir, patient_info, summary, output_path.replace('.pdf', '.txt'))
        return output_path.replace('.pdf', '.txt')

    try:
        return _generate_pdf_report(session_dir, patient_info, summary,
                                     output_path, settings_manager)
    except Exception as e:
        print(f"[HolterReport] PDF generation error: {e}")
        traceback.print_exc()
        return _generate_text_report(session_dir, patient_info, summary,
                                      output_path.replace('.pdf', '.txt'))


_NSR_CONFIRM_READERS: Dict[str, object] = {}


def _first_confirmed_nsr_beat(session_dir: str, t_from: float, t_to: float):
    """Report wrapper around first_confirmed_nsr_beat() for a session folder.
    Returns False when the recording can't be read."""
    try:
        from .file_format import ECGHFileReader
        reader = _NSR_CONFIRM_READERS.get(session_dir)
        if reader is None:
            path = os.path.join(session_dir, 'recording.ecgh')
            if not os.path.exists(path):
                return False
            reader = ECGHFileReader(path)
            _NSR_CONFIRM_READERS[session_dir] = reader
    except Exception as e:
        print(f"[HolterReport] NSR confirmation skipped: {e}")
        return False
    return first_confirmed_nsr_beat(reader, t_from, t_to)


def first_confirmed_nsr_beat(reader, t_from: float, t_to: float):
    """Time (s) of the first beat in [t_from, t_to) that opens a run of 3
    consecutive confirmed normal sinus beats, or None if there is none.

    Shared by the report's Event Timeline and Full Disclosure's rhythm
    label. A beat is confirmed sinus when, measured on Lead II with the same
    measure_beat(strict_pr=True) the auto-segment detector uses: a P wave
    precedes it with PR 120-200 ms, its QRS is narrow (< 120 ms), and the RR
    from the previous beat is 600-1000 ms (60-100 bpm; 10 ms of slack above
    1000 so an exact 60 bpm rhythm measuring 1000-1002 ms doesn't flicker
    between confirmed and not); the 3 beats' PRs agree within 50 ms. Returns False when the recording can't be read, so
    the caller can keep its previous behaviour.
    """
    try:
        from ecg.pan_tompkins import pan_tompkins
        from ecg.arrhythmia_detector import measure_beat
        names = list(getattr(reader, 'lead_names', []) or [])
        if 'II' not in names:
            return False
        fs = float(getattr(reader, 'fs', 500) or 500)
        total = float(getattr(reader, 'duration_sec', 0.0) or 0.0)
        t_to = min(t_to, t_from + 120.0, total)
        read_start = max(0.0, t_from - 1.5)
        read_end = min(total, t_to + 1.0)
        if read_end - read_start < 2.0:
            return None
        data = reader.read_range(read_start, read_end)
        lead_ii = np.asarray(data[names.index('II')], dtype=float)
        peaks = [int(p) for p in pan_tompkins(lead_ii, fs=fs)]
        half = int(fs)
        run = []  # (time, pr) of the current run of confirmed beats
        for k in range(1, len(peaks)):
            p = peaks[k]
            t = read_start + p / fs
            if t < t_from:
                continue
            if t >= t_to:
                break
            rr_ms = (p - peaks[k - 1]) * 1000.0 / fs
            lo, hi = max(0, p - half), min(lead_ii.size, p + half)
            beat = measure_beat(lead_ii[lo:hi], p - lo, fs, strict_pr=True, prev_rr_ms=rr_ms)
            pr = beat.get('pr_ms') if beat else None
            qrs = float(beat.get('qrs_ms') or 0.0) if beat else 0.0
            confirmed = (
                beat is not None and pr is not None and 120.0 <= float(pr) <= 200.0
                and 0.0 < qrs < 120.0 and 600.0 <= rr_ms <= 1010.0
            )
            if not confirmed:
                run = []
                continue
            run.append((t, float(pr)))
            if len(run) >= 3:
                prs = [v for _, v in run[-3:]]
                if max(prs) - min(prs) <= 50.0:
                    return run[-3][0]
                run = run[-2:]
        return None
    except Exception as e:
        print(f"[HolterReport] NSR confirmation skipped: {e}")
        return False


def _build_timeline_events(session_dir: str,
                           auto_segment_events=None,
                           metrics=None) -> List[Dict[str, object]]:
    """
    Builds a clean, streamlined event timeline for the report:
    1. Loads manual markings (manual_segments.json and manual_beats.json) - EXACT MANUAL MARKING LOGIC.
    2. Loads auto-detected rhythm segments (Sinus Tachycardia, Sinus Bradycardia, AFib, VFib, VTach, etc.).
    3. Creates structured start/end markers for rhythm segments.
    4. Establishes baseline Normal Sinus Rhythm transitions outside segments without repetitive periodic polling spam.
    5. Loads auto-detected badge events (SVT, VT, Couplet, Bigeminy, PVC, PAC, etc.) outside manual/segment areas.
    6. Appends manual beat and segment markings (unaltered).
    7. Chronologically sorts and deduplicates timeline events.
    """
    timeline_events = []

    # 1. Load manual segments (do NOT alter manual marking logic)
    manual_segments = []
    manual_segments_path = os.path.join(session_dir, 'manual_segments.json')
    if os.path.exists(manual_segments_path):
        try:
            with open(manual_segments_path, 'r') as _ms_f:
                manual_segments = json.load(_ms_f)
            print(f"[HolterReport] Loaded {len(manual_segments)} manual segments for timeline")
        except Exception as _ms_e:
            print(f"[HolterReport] Could not load manual segments for timeline: {_ms_e}")

    # 2. Load manual beats (do NOT alter manual marking logic)
    manual_beats = []
    manual_beats_path = os.path.join(session_dir, 'manual_beats.json')
    if os.path.exists(manual_beats_path):
        try:
            with open(manual_beats_path, 'r') as _mb_f:
                loaded_beats = json.load(_mb_f)
                manual_beats = [
                    mb for mb in loaded_beats 
                    if mb.get('is_manual', False) or (mb.get('marking_mode') is not None) or (mb.get('batch_id') is not None)
                ]
            print(f"[HolterReport] Loaded {len(manual_beats)} manual beats for timeline")
        except Exception as _mb_e:
            print(f"[HolterReport] Could not load manual beats for timeline: {_mb_e}")

    # 3. Load auto-detected rhythm segments
    auto_segments = (
        auto_segment_events
        if auto_segment_events is not None
        else _load_auto_segment_events(session_dir)
    )

    # 4. Extract all_beats from metrics for badge detection
    metrics = metrics if metrics is not None else load_metrics(session_dir)
    if not metrics:
        jsonl_path = os.path.join(session_dir, 'metrics.jsonl')
        if os.path.exists(jsonl_path):
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as _jf:
                    for line in _jf:
                        line = line.strip()
                        if line:
                            metrics.append(json.loads(line))
            except Exception as _je:
                print(f"[HolterReport] Could not read metrics.jsonl: {_je}")

    all_beats = []
    for metric in metrics or []:
        for b in metric.get('all_beats', []) or []:
            if isinstance(b, dict):
                all_beats.append(b)

    # 5. Build auto segment events (start and end marker rows)
    # Suppress auto segments if they fall inside manual segments
    active_segments_ranges = []
    for seg in manual_segments:
        s_t = float(seg.get('start_sec', 0.0) or 0.0)
        e_t = float(seg.get('end_sec', s_t) or s_t)
        active_segments_ranges.append((s_t, e_t))
    manual_segment_ranges = list(active_segments_ranges)
    bundle_branch_block_ranges = []

    for seg in auto_segments:
        s_t = float(seg.get('start_sec', seg.get('timestamp', 0.0)) or 0.0)
        e_t = float(seg.get('end_sec', seg.get('end_timestamp', s_t)) or s_t)
        lbl = str(seg.get('label', 'Event'))

        # A bundle branch block is a QRS-shape finding that runs alongside
        # the rhythm (e.g. Sinus Bradycardia with Complete Right Bundle
        # Branch Block), so it overlaps an auto rhythm segment by design.
        # It is only held back by a manual segment, and is not added to
        # active_segments_ranges, so the rhythm rows and the Normal Sinus
        # Rhythm return markers below are exactly as without it.
        if 'bundle branch block' in lbl.lower():
            if any(ms < e_t and s_t < me for ms, me in manual_segment_ranges):
                continue
            bundle_branch_block_ranges.append((s_t, e_t))
            timeline_events.append({
                'timestamp': s_t,
                'sort_ts': s_t,
                'label': f"Auto segment marked ({lbl})",
                'event_type': lbl,
                'source': 'Auto'
            })
            if e_t > s_t + 0.1:
                timeline_events.append({
                    'timestamp': e_t,
                    'sort_ts': e_t,
                    'label': f"Auto segment marked end ({lbl})",
                    'event_type': lbl,
                    'source': 'Auto'
                })
            continue

        # Check if inside manual segment (or a previously-processed auto
        # segment -- this list accumulates both). A strict/open-interval
        # overlap test (ms < e_t and s_t < me), not an inclusive endpoint
        # check: the inclusive version treated two segments that merely
        # TOUCH at a shared boundary (e.g. one auto-detected rhythm ending
        # at the exact same timestamp the next one starts, which is the
        # normal, correct way adjacent segments now meet) as "overlapping",
        # silently dropping the second segment from the timeline entirely --
        # confirmed directly: an Atrial Fibrillation segment starting
        # exactly where the preceding Ventricular Fibrillation segment
        # ended never appeared in the Event Timeline at all, even though it
        # showed correctly in the Arrhythmia Summary above (built
        # separately, without this check).
        in_manual = any(ms < e_t and s_t < me for ms, me in active_segments_ranges)
        if in_manual:
            continue

        active_segments_ranges.append((s_t, e_t))

        timeline_events.append({
            'timestamp': s_t,
            'sort_ts': s_t,
            'label': f"Auto segment marked ({lbl})",
            'event_type': lbl,
            'source': 'Auto'
        })
        if e_t > s_t + 0.1:
            timeline_events.append({
                'timestamp': e_t,
                'sort_ts': e_t,
                'label': f"Auto segment marked end ({lbl})",
                'event_type': lbl,
                'source': 'Auto'
            })

    # 6. Baseline rhythm / transition events
    # Start with baseline Normal Sinus Rhythm at t=0 if recording has data and t=0 is not in a segment
    if metrics or auto_segments or manual_segments:
        # Do not append a trailing NSR marker when the preceding segment
        # reaches the recording end.  There is no post-segment data from
        # which to establish a return to sinus rhythm; adding one here made
        # the report show a false final NSR row after VFib/VT.
        recording_end_sec = 0.0
        try:
            # Chunk start time: metrics.jsonl rows store it as 't' (seconds
            # from recording start); 'timestamp' is the older key. Reading
            # only 'timestamp' gave 0 for every row, so the "recording end"
            # came out as one chunk (5 s) and every Normal Sinus Rhythm
            # return marker after a segment was skipped as "at the end".
            recording_end_sec = max(
                float(metric.get('timestamp', metric.get('t', 0.0)) or 0.0)
                + float(metric.get('duration_sec', metric.get('duration', 0.0)) or 0.0)
                for metric in (metrics or [])
                if isinstance(metric, dict)
            )
        except Exception:
            recording_end_sec = 0.0
        if recording_end_sec <= 0.0:
            recording_end_sec = max(
                [float(seg.get('end_sec', seg.get('end_timestamp', 0.0)) or 0.0)
                 for seg in auto_segments]
                + [float(seg.get('end_sec', 0.0) or 0.0) for seg in manual_segments]
                + [0.0]
            )
        starts_in_segment = any(ms <= 0.5 and me > 0.0 for ms, me in active_segments_ranges)

        # Normal Sinus Rhythm rows are placed at the first CONFIRMED sinus
        # beat (_first_confirmed_nsr_beat: P with PR 120-200 ms, QRS < 120
        # ms, 60-100 bpm, 3 beats in a row) up to the next segment. With no
        # such beat (or an unreadable recording) the row stays where it was
        # before -- recording start, or the second after a segment ends --
        # matching Full Disclosure's label, which falls back to Normal Sinus
        # Rhythm whenever no segment is active.
        def _next_segment_start(after):
            starts = [ms for ms, me in active_segments_ranges if ms > after + 0.01]
            starts += [bs for bs, be in bundle_branch_block_ranges if bs > after + 0.01]
            if recording_end_sec > 0.0:
                starts.append(recording_end_sec)
            return min(starts) if starts else after + 120.0

        start_row_sec = 0.0
        if not starts_in_segment:
            confirmed_start = _first_confirmed_nsr_beat(session_dir, 0.0, _next_segment_start(-1.0))
            if confirmed_start not in (None, False):
                start_row_sec = float(confirmed_start)
        if not starts_in_segment:
            timeline_events.append({
                'timestamp': start_row_sec,
                'sort_ts': start_row_sec,
                'label': 'Normal Sinus Rhythm',
                'event_type': 'Normal Sinus Rhythm',
                'source': 'Auto'
            })

        # Add baseline return after segments if there's a gap before next segment
        #
        # After an AUTO segment the row is placed where Full Disclosure's
        # rhythm label itself switches to Normal Sinus Rhythm: that label
        # keeps showing the segment for as long as the visible frame's
        # wall-clock second is the segment's own end second, so the first
        # second it reads Normal Sinus Rhythm is the end second + 1 (end
        # 12:27:24.2 -> NSR 12:27:25). Rows after a manual segment are left
        # exactly as before.
        _report_start_wall = _report_recording_start_time(session_dir)

        def _display_resume(seg_end):
            if _report_start_wall:
                return float(np.floor(_report_start_wall + seg_end)) + 1.0 - _report_start_wall
            return float(np.floor(seg_end)) + 1.0

        manual_resume_times = set()
        for ms, me in active_segments_ranges:
            if (ms, me) in manual_segment_ranges:
                t_resume = me + 0.05
            else:
                confirmed = _first_confirmed_nsr_beat(session_dir, me, _next_segment_start(me))
                t_resume = _display_resume(me) if confirmed in (None, False) else float(confirmed)
            if recording_end_sec > 0.0 and t_resume >= recording_end_sec - 0.05:
                continue
            in_another = any(other_s <= t_resume <= other_e for other_s, other_e in active_segments_ranges if (other_s, other_e) != (ms, me))
            if not in_another:
                if (ms, me) in manual_segment_ranges:
                    manual_resume_times.add(t_resume)
                timeline_events.append({
                    'timestamp': t_resume,
                    'sort_ts': t_resume,
                    'label': 'Normal Sinus Rhythm',
                    'event_type': 'Normal Sinus Rhythm',
                    'source': 'Auto'
                })

        # Bundle branch block: while one is marked, Full Disclosure's label
        # reads the bundle branch block, not Normal Sinus Rhythm, so no
        # Normal Sinus Rhythm row falls inside one (rows after a manual
        # segment are left as they are). Where it ends with no rhythm
        # segment there, Normal Sinus Rhythm resumes by the same end-second
        # + 1 rule as after any other auto segment.
        if bundle_branch_block_ranges:
            def _in_bundle_branch_block(t):
                return any(bs <= t <= be for bs, be in bundle_branch_block_ranges)

            timeline_events[:] = [
                ev for ev in timeline_events
                if not (
                    ev.get('label') == 'Normal Sinus Rhythm'
                    and float(ev.get('timestamp', 0.0) or 0.0) not in manual_resume_times
                    and _in_bundle_branch_block(float(ev.get('timestamp', 0.0) or 0.0))
                )
            ]
            for bs, be in bundle_branch_block_ranges:
                confirmed = _first_confirmed_nsr_beat(session_dir, be, _next_segment_start(be))
                t_resume = _display_resume(be) if confirmed in (None, False) else float(confirmed)
                if recording_end_sec > 0.0 and t_resume >= recording_end_sec - 0.05:
                    continue
                if any(other_s <= t_resume <= other_e for other_s, other_e in active_segments_ranges) \
                        or _in_bundle_branch_block(t_resume):
                    continue
                timeline_events.append({
                    'timestamp': t_resume,
                    'sort_ts': t_resume,
                    'label': 'Normal Sinus Rhythm',
                    'event_type': 'Normal Sinus Rhythm',
                    'source': 'Auto'
                })

    #    Ectopic beat pattern badge events (Single PVC, PVC Couplet, Bigeminy, Trigeminy,
    #    Quadrigeminy, VT Run, Single PAC, PAC Couplet, SVT Run, AFib from sliding window)
    #    are intentionally NOT added to the Event Timeline.
    #    Only auto-segment rhythm events (VF, AFib, AFlutter, Sinus Brady, Sinus Tachy)
    #    and manual markings are shown in the timeline.

    # 7. Generate auto-detection badge events (SVT, VT, Couplet, Bigeminy, PVC, PAC, etc.)
    # TODO: Baad mein enable karna — abhi Event Timeline mein sirf auto-segment rhythm events
    #       (VF, AFib, AFlutter, Sinus Brady, Sinus Tachy) aur manual markings dikhne chahiye.
    #       Ectopic beat pattern labels (Single PVC, Couplet, Bigeminy, VT Run, PAC, SVT Run)
    #       timeline mein add karne ka kaam baad mein karein.
    # try:
    #     from .holter_summary_calc import get_template_beats_for_badges
    #     badges = get_template_beats_for_badges(all_beats, [])
    #     for b in badges:
    #         badge_name = b.get('name', '').strip()
    #         if not badge_name:
    #             continue
    #         b_ts = float(b.get('timestamp', 0.0) or 0.0)
    #
    #         # Check if inside any manual marking
    #         in_manual = False
    #         for ms, me in [(float(s.get('start_sec', 0.0)), float(s.get('end_sec', 0.0))) for s in manual_segments]:
    #             if ms <= b_ts <= me:
    #                 in_manual = True
    #                 break
    #         if not in_manual and manual_beats:
    #             for mb in manual_beats:
    #                 if abs(b_ts - float(mb.get('timestamp', 0.0))) < 0.15:
    #                     in_manual = True
    #                     break
    #         if in_manual:
    #             continue
    #
    #         if (b.get('code') in ['V', 'S', 'AF', 'P'] or
    #             b.get('critical', False) or
    #             any(k in badge_name.lower() for k in ['run', 'couplet', 'bigeminy', 'trigeminy', 'quadrigeminy', 'tachycardia', 'fibrillation', 'flutter'])):
    #             timeline_events.append({
    #                 'timestamp': b_ts,
    #                 'sort_ts': b_ts,
    #                 'label': badge_name,
    #                 'event_type': b.get('code', 'Arrhythmia'),
    #                 'source': 'Auto',
    #                 'confidence': 1.0,
    #             })
    # except Exception as _badge_e:
    #     print(f"[HolterReport] Error generating badge events: {_badge_e}")

    # 8. Load manual beats and append non-normal ones to timeline (parallel manual marking)
    # [EXACT ORIGINAL MANUAL MARKING LOGIC]
    for mb in manual_beats:
        if not (mb.get('is_manual', False) or (mb.get('marking_mode') is not None) or (mb.get('batch_id') is not None)):
            continue
        lbl = mb.get('label', 'N')
        short_code = lbl
        if '(' in lbl and ')' in lbl:
            short_code = lbl.split('(')[1].split(')')[0]
        if short_code != 'N':
            marking_mode = mb.get('marking_mode', 'parallel_single')
            if marking_mode == 'parallel_multi':
                label_text = f"Parallel multiple beat marked ({lbl})"
            else:
                label_text = f"Parallel single beat marked ({lbl})"
            timeline_events.append({
                'timestamp': float(mb.get('timestamp', 0.0)),
                'sort_ts': float(mb.get('timestamp', 0.0)),
                'label': label_text,
                'event_type': lbl,
                'source': 'Manual'
            })

    # 9. Load manual segments and append to timeline (segment manual marking)
    # [EXACT ORIGINAL MANUAL MARKING LOGIC]
    for seg in manual_segments:
        lbl = seg.get('label', 'Unknown')
        start_sec = seg.get('start_sec', 0.0)
        end_sec = seg.get('end_sec', 0.0)
        timeline_events.append({
            'timestamp': float(start_sec),
            'sort_ts': float(start_sec),
            'label': f"Segment manual marked ({lbl})",
            'event_type': lbl,
            'source': 'Manual'
        })
        timeline_events.append({
            'timestamp': float(end_sec),
            'sort_ts': float(end_sec),
            'label': f"Segment manual marked end ({lbl})",
            'event_type': lbl,
            'source': 'Manual'
        })

    # 10. Chronologically sort all events
    timeline_events.sort(key=lambda x: float(x.get("sort_ts", x.get("timestamp", 0.0)) or 0.0))

    # 11. Deduplicate consecutive identical rows
    deduped_timeline = []
    for ev in timeline_events:
        if deduped_timeline:
            last_ev = deduped_timeline[-1]
            if last_ev.get('label') == ev.get('label') and abs(float(last_ev.get('timestamp', 0.0)) - float(ev.get('timestamp', 0.0))) < 0.2:
                continue
            if last_ev.get('label') == 'Normal Sinus Rhythm' and ev.get('label') == 'Normal Sinus Rhythm':
                continue
        deduped_timeline.append(ev)

    return deduped_timeline


#    PDF Report                                                                  

def _generate_pdf_report(session_dir, patient_info, summary, output_path, settings_manager):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, Image, PageBreak,
                                     HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    PAGE_W, PAGE_H = A4
    # Keep the existing orange identity, but pair it with the restrained
    # navy/blue-gray palette used by clinical monitoring reports.
    ORANGE = colors.HexColor('#E65100')
    NAVY   = colors.HexColor('#17324D')
    TEAL   = colors.HexColor('#197C86')
    DARK   = colors.HexColor('#243447')
    LIGHT  = colors.HexColor('#F8F5F0')
    PANEL  = colors.HexColor('#F7FAFC')
    GRAY   = colors.HexColor('#EEF2F5')
    BORDER = colors.HexColor('#C9D3DC')
    GREEN  = colors.HexColor('#2E7D32')
    RED    = colors.HexColor('#B71C1C')
    BLUE   = colors.HexColor('#1565C0')

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=14*mm, bottomMargin=14*mm,
        title="Comprehensive ECG Analysis Report",
    )

    def _draw_page_chrome(canvas, _doc):
        """Draw a quiet clinical header/footer without changing report data."""
        canvas.saveState()
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(15*mm, PAGE_H - 11*mm, PAGE_W - 15*mm, PAGE_H - 11*mm)
        canvas.setFont('Helvetica-Bold', 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(15*mm, PAGE_H - 8*mm, 'HOLTER ECG  |  COMPREHENSIVE ANALYSIS')
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(colors.HexColor('#607080'))
        canvas.drawRightString(PAGE_W - 15*mm, PAGE_H - 8*mm, 'CONFIDENTIAL CLINICAL REPORT')
        canvas.line(15*mm, 11*mm, PAGE_W - 15*mm, 11*mm)
        canvas.setFont('Helvetica', 7.5)
        canvas.drawString(15*mm, 7*mm, 'For clinical review • Generated by Holter ECG Analysis')
        canvas.drawRightString(PAGE_W - 15*mm, 7*mm, f'Page {canvas.getPageNumber()}')
        canvas.restoreState()

    styles = getSampleStyleSheet()
    # Load the session-wide inputs once. Auto-segment detection and metrics
    # parsing are among the most expensive report steps; the same results are
    # consumed by the summary grid, arrhythmia table and event timeline.
    _report_metrics = load_metrics(session_dir) or []
    _report_auto_segment_events = _load_auto_segment_events(session_dir)

    title_style = ParagraphStyle('Title', parent=styles['Normal'],
                                  fontSize=18, leading=22, textColor=NAVY,
                                  fontName='Helvetica-Bold',
                                  alignment=TA_CENTER, spaceAfter=3*mm)
    h1_style = ParagraphStyle('H1', parent=styles['Normal'],
                               fontSize=12, leading=15, textColor=NAVY,
                               fontName='Helvetica-Bold',
                               spaceBefore=5*mm, spaceAfter=2*mm,
                               borderPad=(0, 0, 2, 0))
    h2_style = ParagraphStyle('H2', parent=styles['Normal'],
                               fontSize=10, leading=13, textColor=TEAL,
                               fontName='Helvetica-Bold',
                               spaceBefore=4*mm, spaceAfter=1.5*mm)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=8.6, leading=11.2, textColor=DARK, spaceAfter=0.8*mm)
    small_style = ParagraphStyle('Small', parent=styles['Normal'],
                                  fontSize=7.5, leading=10, textColor=colors.HexColor('#607080'))

    story = []

    #    PAGE 1: FINAL SUMMARY                                                   
    story.append(Paragraph("COMPREHENSIVE ECG ANALYSIS REPORT", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))

    # Patient info table
    dur_h = int(summary.get('duration_sec', 0) // 3600)
    dur_m = int((summary.get('duration_sec', 0) % 3600) // 60)
    dur_s = int(summary.get('duration_sec', 0) % 60)
    pname = patient_info.get('name', patient_info.get('patient_name', 'Unknown'))
    
    st_time_str, end_time_str = _get_recording_start_end(session_dir, float(summary.get('duration_sec', 0)))
    
    pinfo_data = [
        ['Patient Name', pname,              'Recording Duration', f"{dur_h}h {dur_m}m {dur_s}s"],
        ['Age / Gender', f"{patient_info.get('age','--')} / {patient_info.get('gender','--')}",
         'Report Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
        ['Doctor',       patient_info.get('doctor', '--'),
         'Recording Start', st_time_str],
        ['Organisation', patient_info.get('Org.', patient_info.get('org', '--')),
         'Recording End', end_time_str],
        ['Email',        patient_info.get('email', '--'),
         'Phone',        patient_info.get('phone', patient_info.get('doctor_mobile', '--'))],
    ]

    pinfo_table = Table(pinfo_data, colWidths=[35*mm, 55*mm, 45*mm, 45*mm])
    pinfo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), GRAY),
        ('BACKGROUND', (2, 0), (2, -1), GRAY),
        ('TEXTCOLOR', (0, 0), (-1, -1), DARK),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, LIGHT]),
        ('PADDING', (0, 0), (-1, -1), 4.5),
    ]))
    story.append(pinfo_table)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("1. RECORDING SUMMARY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    # ------------------------------------------------------------------
    # Six-panel clinical summary grid (Brief Summary / Heart Rate / AFib-
    # Flutter / VE & Rhythm / SVE & Rhythm / HRV). Purely additive: sits
    # above the existing Parameter/Value table below and doesn't change it.
    # Loads its own beats/RR intervals locally so it doesn't depend on
    # anything computed later in this function.
    # ------------------------------------------------------------------
    try:
        from .hrv_metrics import compute_hrv_summary
        from .holter_summary_calc import label_beat_sequences

        _grid_metrics = _report_metrics
        _grid_beats = []
        _grid_rr_ms = []
        for _m in _grid_metrics:
            for _b in (_m.get('all_beats') or []):
                if isinstance(_b, dict):
                    _grid_beats.append(dict(_b))
            for _rr in (_m.get('rr_intervals_list') or []):
                try:
                    _rr_f = float(_rr)
                    if _rr_f > 200:
                        _grid_rr_ms.append(_rr_f)
                except (TypeError, ValueError):
                    pass
        _grid_beats.sort(key=lambda b: float(b.get('timestamp', 0.0) or 0.0))
        label_beat_sequences(_grid_beats)

        _ve_patterns = {'single': 0, 'couplet': 0, 'bigeminy': 0, 'trigeminy': 0, 'run': 0}
        _sve_patterns = {'single': 0, 'couplet': 0, 'bigeminy': 0, 'trigeminy': 0, 'run': 0}
        _ve_total = _sve_total = 0
        _ve_run_lengths = []
        _sve_run_lengths = []
        for _b in _grid_beats:
            code = _b.get('short_code', '')
            if code not in ('V', 'S'):
                continue
            bucket = _ve_patterns if code == 'V' else _sve_patterns
            if code == 'V':
                _ve_total += 1
            else:
                _sve_total += 1
            pl = str(_b.get('rhythm_pattern', '') or '').lower()
            if 'run' in pl:
                bucket['run'] += 1
                (_ve_run_lengths if code == 'V' else _sve_run_lengths).append(int(_b.get('run_length', 0) or 0))
            elif 'quadrigeminy' in pl or 'trigeminy' in pl:
                bucket['trigeminy'] += 1
            elif 'bigeminy' in pl:
                bucket['bigeminy'] += 1
            elif 'couplet' in pl:
                bucket['couplet'] += 1
            else:
                bucket['single'] += 1

        _longest_ve_run = max(_ve_run_lengths) if _ve_run_lengths else 0
        _longest_sve_run = max(_sve_run_lengths) if _sve_run_lengths else 0

        # Paced beats: short_code 'P' (see calculate_summary_metrics(), the
        # same definition used elsewhere in this module). Instant rate per
        # paced beat comes from the RR gap to the immediately preceding
        # beat in the chronologically-sorted list.
        _paced_total = 0
        _paced_rates_bpm = []
        _prev_beat_ts = None
        for _b in _grid_beats:
            _ts = float(_b.get('timestamp', 0.0) or 0.0)
            if _b.get('short_code') == 'P':
                _paced_total += 1
                if _prev_beat_ts is not None and _ts > _prev_beat_ts:
                    _paced_rates_bpm.append(60.0 / (_ts - _prev_beat_ts))
            _prev_beat_ts = _ts

        def _grid_pct(part, total):
            return f"{(100.0 * part / total):.1f}%" if total else "0.0%"

        _hrv = compute_hrv_summary(_grid_rr_ms) if _grid_rr_ms else {}

        # AFib/Flutter burden from the auto-segment detector -- same source
        # as the Arrhythmia Summary table further below, so this only shows
        # real numbers once "Auto Arrhythmia Detection" has been run for
        # this session (see _load_auto_segment_events); otherwise "-".
        def _is_afib_or_aflutter(label: str) -> bool:
            lbl = str(label or '').lower()
            # 'fibrillation' alone also matches "Ventricular Fibrillation" --
            # this panel is atrial-only, so exclude anything ventricular.
            return ('fibrillation' in lbl or 'flutter' in lbl) and 'ventricular' not in lbl

        _auto_segment_events_all = _report_auto_segment_events
        _afib_events = [e for e in _auto_segment_events_all if _is_afib_or_aflutter(e.get('label', ''))]
        _afib_total_sec = sum(
            max(0.0, float(e.get('end_sec', 0.0) or 0.0) - float(e.get('start_sec', 0.0) or 0.0))
            for e in _afib_events
        )
        _afib_longest_sec = max(
            (float(e.get('end_sec', 0.0) or 0.0) - float(e.get('start_sec', 0.0) or 0.0) for e in _afib_events),
            default=0.0,
        )
        _rec_dur_sec = float(summary.get('duration_sec', 0.0) or 0.0)
        _afib_burden_pct = (_afib_total_sec / _rec_dur_sec * 100.0) if _rec_dur_sec > 0 else 0.0

        # AV block episode counts, same auto-segment source as AFib/Flutter
        # above (auto_segment_arrthymia_detection.py labels 2nd-degree block
        # as "... (Mobitz I)" or "... (Mobitz II)").
        _av_block_2_1_count = sum(
            1 for e in _auto_segment_events_all
            if str(e.get('label', '')).lower() == 'second-degree av block (mobitz i)'
        )
        _av_block_2_2_count = sum(
            1 for e in _auto_segment_events_all
            if str(e.get('label', '')).lower() == 'second-degree av block (mobitz ii)'
        )
        _av_block_3_count = sum(
            1 for e in _auto_segment_events_all
            if ('third' in str(e.get('label', '')).lower() or '3rd' in str(e.get('label', '')).lower())
            and 'av block' in str(e.get('label', '')).lower()
        )

        def _grid_hms(sec):
            sec = max(0, int(sec))
            h, rem = divmod(sec, 3600)
            m, s = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def _grid_wallclock(sec):
            """MM-DD HH:MM:SS using the recording's real start time, same
            source _format_system_time reads; falls back to elapsed
            HH:MM:SS if the index file isn't available."""
            try:
                idx_path = os.path.join(session_dir, 'recording_index.json')
                if os.path.exists(idx_path):
                    with open(idx_path, 'r') as f:
                        start_time = json.load(f).get('start_time')
                    if start_time:
                        return datetime.fromtimestamp(float(start_time) + sec).strftime('%m-%d %H:%M:%S')
            except Exception:
                pass
            return _grid_hms(sec)

        # Longest RR Interval, with the real date/time it occurred at --
        # found from whichever chunk's own "longest_rr" matches the overall
        # summary value, same way max_hr_chunk/min_hr_chunk are found above.
        _longest_rr_chunk = max(_grid_metrics, key=lambda m: float(m.get('longest_rr', 0) or 0), default={})
        _longest_rr_ms = float(_longest_rr_chunk.get('longest_rr', 0) or 0) or float(summary.get('longest_rr_ms', 0) or 0)
        _longest_rr_t = float(_longest_rr_chunk.get('t', 0.0) or 0.0)
        _longest_rr_str = f"{_longest_rr_ms/1000:.2f}s At {_grid_wallclock(_longest_rr_t)}"

        # Brief Summary as prose, not a table cell -- the same numbers the
        # grid panel below used to show for this, pulled into one sentence.
        story.append(Paragraph("Brief Summary", h2_style))
        _afib_sentence = (
            f"The atrial fibrillation/flutter burden was <b>{_afib_burden_pct:.2f}%</b> of the recording."
            if _afib_events else
            "Atrial fibrillation/flutter burden was not assessed for this recording."
        )
        _brief_summary_text = (
            f"This recording captured a total of <b>{summary.get('total_beats', 0):,}</b> beats, "
            f"including <b>{_ve_total}</b> ventricular ectopic (VE) beat{'s' if _ve_total != 1 else ''} and "
            f"<b>{_sve_total}</b> supraventricular ectopic (SVE) beat{'s' if _sve_total != 1 else ''}. "
            f"{_afib_sentence}"
        )
        story.append(Paragraph(_brief_summary_text, body_style))
        story.append(Spacer(1, 3*mm))

        # Made larger per request -- bigger title/body text, more padding,
        # and 2 wide columns instead of 3 narrower ones (Brief Summary above
        # is no longer a panel here, so there are 5 panels left; Heart Rate
        # Variability, the longest one, spans the full width on its own row).
        grid_title_style = ParagraphStyle('GridTitle', parent=styles['Normal'], fontSize=10.2,
                                           textColor=TEAL, alignment=TA_CENTER, fontName='Helvetica-Bold',
                                           spaceAfter=3)
        grid_body_style = ParagraphStyle('GridBody', parent=styles['Normal'], fontSize=8.5,
                                          leading=11.2, textColor=DARK)

        def _grid_panel(title, rows):
            lines = "<br/>".join(f"<b>{k}:</b> {v}" for k, v in rows)
            return [Paragraph(title, grid_title_style), Paragraph(lines, grid_body_style)]

        hr_panel = _grid_panel("Heart Rate (bpm)", [
            ("Avg", f"{summary.get('avg_hr', 0):.0f}"),
            ("Max", f"{summary.get('max_hr', 0):.0f} @ {summary.get('max_hr_time', '-')}"),
            ("Min", f"{summary.get('min_hr', 0):.0f} @ {summary.get('min_hr_time', '-')}"),
            ("Longest RR Interval", _longest_rr_str),
            ("Pause (&gt;2.0s)", str(summary.get('pauses', 0))),
        ])
        afib_panel = _grid_panel("Atrial Fibrillation/Flutter", [
            ("Total Event", str(len(_afib_events))),
            ("Burden (Hour)", f"{_afib_total_sec/3600:.2f}" if _afib_events else "-"),
            ("Burden (%)", f"{_afib_burden_pct:.2f}%" if _afib_events else "-"),
            ("Longest", _grid_hms(_afib_longest_sec) if _afib_events else "-"),
        ])
        ve_panel = _grid_panel("VE and Rhythm", [
            ("Single", f"{_ve_patterns['single']} ({_grid_pct(_ve_patterns['single'], _ve_total)})"),
            ("Couplet", str(_ve_patterns['couplet'])),
            ("Bigeminy", str(_ve_patterns['bigeminy'])),
            ("Trigeminy", str(_ve_patterns['trigeminy'])),
            ("VT run", str(_ve_patterns['run'])),
            ("Longest VT run", str(_longest_ve_run)),
            ("Total VE", str(_ve_total)),
        ])
        sve_panel = _grid_panel("SVE and Rhythm", [
            ("Single", f"{_sve_patterns['single']} ({_grid_pct(_sve_patterns['single'], _sve_total)})"),
            ("Couplet", str(_sve_patterns['couplet'])),
            ("Bigeminy", str(_sve_patterns['bigeminy'])),
            ("Trigeminy", str(_sve_patterns['trigeminy'])),
            ("SVT run", str(_sve_patterns['run'])),
            ("Longest SVT run", str(_longest_sve_run)),
            ("Total SVE", str(_sve_total)),
        ])
        hrv_panel = _grid_panel("Heart Rate Variability", [
            ("SDNN (ms)", f"{_hrv.get('sdnn', 0):.1f}" if _hrv else "-"),
            ("rMSSD (ms)", f"{_hrv.get('rmssd', 0):.1f}" if _hrv else "-"),
            ("pNN50 (%)", f"{_hrv.get('pnn50', 0):.2f}" if _hrv else "-"),
            ("Triangular Index", f"{_hrv.get('triangular_index', 0):.2f}" if _hrv else "-"),
            ("VLF", f"{_hrv.get('vlf', 0):.1f}" if _hrv else "-"),
            ("LF", f"{_hrv.get('lf', 0):.1f}" if _hrv else "-"),
            ("HF", f"{_hrv.get('hf', 0):.1f}" if _hrv else "-"),
            ("LF/HF", f"{_hrv.get('lf_hf_ratio', 0):.2f}" if _hrv else "-"),
        ])
        paced_panel = _grid_panel("Paced", [
            ("Total", str(_paced_total)),
            ("Max paced rate", f"{max(_paced_rates_bpm):.0f} bpm" if _paced_rates_bpm else "-"),
            ("Min paced rate", f"{min(_paced_rates_bpm):.0f} bpm" if _paced_rates_bpm else "-"),
            ("Ave paced rate", f"{sum(_paced_rates_bpm)/len(_paced_rates_bpm):.0f} bpm" if _paced_rates_bpm else "-"),
        ])
        others_panel = _grid_panel("Others", [
            ("2nd deg. AV block type I", str(_av_block_2_1_count) if _auto_segment_events_all else "-"),
            ("2nd deg. AV block type II", str(_av_block_2_2_count) if _auto_segment_events_all else "-"),
            ("3rd deg. AV block", str(_av_block_3_count) if _auto_segment_events_all else "-"),
        ])

        grid_table = Table(
            [[hr_panel, afib_panel],
             [ve_panel, sve_panel],
             [hrv_panel, ''],
             [paced_panel, others_panel]],
            colWidths=[87*mm, 87*mm],
        )
        grid_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.6, BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 0), (-1, -1), PANEL),
            ('SPAN', (0, 2), (1, 2)),
        ]))
        story.append(grid_table)
        story.append(Spacer(1, 4*mm))
    except Exception as _grid_err:
        print(f"[HolterReport] Could not build the summary grid panel: {_grid_err}")

    # Parameter/Value table removed per request -- its numbers are now
    # covered by the summary grid above (Brief Summary text + the 5-panel
    # grid). avg_hr/min_hr/max_hr are kept as plain variables since they
    # still feed the Clinical Impression paragraph below.
    avg_hr = summary.get('avg_hr', 0)
    max_hr = summary.get('max_hr', 0)
    min_hr = summary.get('min_hr', 0)

    # -------------------------------------------------------------------------
    # Compute Arrhythmias (combining auto detection with exact manual markings)
    # -------------------------------------------------------------------------
    manual_beats = []
    manual_beats_path = os.path.join(session_dir, 'manual_beats.json')
    if os.path.exists(manual_beats_path):
        try:
            with open(manual_beats_path, 'r') as _mb_f:
                manual_beats = json.load(_mb_f)
            print(f"[HolterReport] Loaded {len(manual_beats)} manual beats for arrhythmia summary")
        except Exception as _mb_e:
            print(f"[HolterReport] Could not load manual beats for arrhythmia summary: {_mb_e}")

    manual_segments = []
    manual_segments_path = os.path.join(session_dir, 'manual_segments.json')
    if os.path.exists(manual_segments_path):
        try:
            with open(manual_segments_path, 'r') as _ms_f:
                manual_segments = json.load(_ms_f)
            print(f"[HolterReport] Loaded {len(manual_segments)} manual segments for arrhythmia summary")
        except Exception as _ms_e:
            print(f"[HolterReport] Could not load manual segments for arrhythmia summary: {_ms_e}")

    # Count manual markings by label type
    manual_arrhy_counts = {}
    for mb in manual_beats:
        if not (mb.get('is_manual', False) or (mb.get('marking_mode') is not None) or (mb.get('batch_id') is not None)):
            continue
        lbl = mb.get('label', 'N')
        short_code = lbl
        if '(' in lbl and ')' in lbl:
            short_code = lbl.split('(')[1].split(')')[0]
        if short_code != 'N':
            manual_arrhy_counts[lbl] = manual_arrhy_counts.get(lbl, 0) + 1

    for seg in manual_segments:
        lbl = seg.get('label', 'Unknown')
        manual_arrhy_counts[lbl] = manual_arrhy_counts.get(lbl, 0) + 1

    auto_segment_events = _report_auto_segment_events
    auto_segment_counts = {}
    auto_segment_times = {}
    for event in auto_segment_events:
        label = str(event.get('label', 'Unknown'))
        auto_segment_counts[label] = auto_segment_counts.get(label, 0) + 1
        s_t = float(event.get('start_sec', event.get('timestamp', 0.0)) or 0.0)
        e_t = float(event.get('end_sec', event.get('end_timestamp', s_t)) or s_t)
        s_str = _format_system_time(session_dir, s_t)
        e_str = _format_system_time(session_dir, e_t)
        if e_t > s_t + 0.5:
            auto_segment_times.setdefault(label, []).append(f"{s_str} - {e_str}")
        else:
            auto_segment_times.setdefault(label, []).append(s_str)

    # Merge auto-detected and manual arrhythmia counts
    arrhy_counts = summary.get('arrhythmia_counts', {})
    combined_arrhy_counts = {}
    for label, count in arrhy_counts.items():
        if label not in auto_segment_counts and not _detector_owns_label(label):
            combined_arrhy_counts[label] = count
    for label, count in auto_segment_counts.items():
        combined_arrhy_counts[label] = count
    for label, count in manual_arrhy_counts.items():
        combined_arrhy_counts[label] = combined_arrhy_counts.get(label, 0) + count

    # Filter out noise/conduction terms
    excluded_summary_terms = (
        'long qt',
        'prolonged qt',
        'wide qrs',
        'premature ventricular contraction',
        'pvc',
        'second-degree',
        'second degree',
        'third-degree',
        'third degree',
        # Ventricular Tachycardia is intentionally never reported here --
        # auto_segment_arrthymia_detection.py's own VFib rule folds any
        # window chaotic enough to be VT into Ventricular Fibrillation by
        # design (see that module's own comment on the point), so this
        # detector never produces a "Ventricular Tachycardia" auto segment
        # with real timing of its own. Without this exclusion, the summary
        # fell back to the recording's separate, older stored real-time-
        # pipeline tally for that same label (arrhy_counts further above,
        # from `summary['arrhythmia_counts']`) purely because it wasn't in
        # auto_segment_counts -- showing an orphaned "Ventricular
        # Tachycardia" row with an episode count but no coherent start/end
        # time (the stored tally has no segment timing of its own to show).
        'ventricular tachycardia',
    )
    filtered_arrhy_counts = {}
    for label, count in combined_arrhy_counts.items():
        label_lower = label.lower()
        is_generic_sinus_rhythm = (
            'sinus rhythm' in label_lower
            and 'normal sinus rhythm' not in label_lower
        )
        # AV blocks (First-degree, Mobitz I, Mobitz II, Third-degree) and
        # Ventricular Tachycardia from the waveform auto-segment detector
        # carry their own segment timing and are reported. The note on
        # 'ventricular tachycardia' above predates the detector's own VT
        # rule (_is_ventricular_tachycardia_window); the terms above still
        # filter every other source (stored tally, manual) as before.
        # 'Supraventricular Tachycardia (SVT)' contains the text "ventricular
        # tachycardia", so the substring filter above removed it from the
        # summary even though the detector reported it with its own timing.
        is_auto_reported = label in auto_segment_counts and (
            'av block' in label_lower
            or label_lower in ('ventricular tachycardia', 'supraventricular tachycardia (svt)')
        )
        if is_generic_sinus_rhythm or (
            not is_auto_reported and any(term in label_lower for term in excluded_summary_terms)
        ):
            continue
        filtered_arrhy_counts[label] = count

    story.append(Paragraph("Clinical Impression", h2_style))
    top_events = ", ".join(f"{label} ({count} episode{'s' if count > 1 else ''})" for label, count in sorted(filtered_arrhy_counts.items(), key=lambda item: -item[1])[:4]) or "No significant arrhythmias detected"
    avg_quality = summary.get('avg_quality', 0) * 100
    impression_text = (
        f"This Comprehensive ECG Analysis study for <b>{pname}</b> covers <b>{dur_h}h {dur_m}m {dur_s}s</b> with an average heart rate of "
        f"<b>{avg_hr:.0f} bpm</b> (minimum <b>{min_hr:.0f} bpm</b>, maximum <b>{max_hr:.0f} bpm</b>). "
        f"Overall signal quality was <b>{avg_quality:.1f}%</b>. The rhythm and arrhythmia summary shows: <b>{top_events}</b>."
    )
    story.append(Paragraph(impression_text, body_style))
    
    auto_conclusion = _auto_conclusion(summary, filtered_arrhy_counts)
    story.append(Paragraph(auto_conclusion, body_style))

    # Signature box
    sig_data = [
        ['Reference Report Confirmed by', 'Doctor Name', 'Doctor Sign'],
        ['', patient_info.get('doctor', ''), ''],
    ]
    sig_table = Table(sig_data, colWidths=[70*mm, 50*mm, 60*mm])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('ROWBACKGROUNDS', (0, 0), (-1, 0), [GRAY]),
        ('MINROWHEIGHT', (0, 1), (-1, 1), 18*mm),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sig_table)

    # Start the detailed statistics and numerical summary on page 2 while
    # keeping the confirmation/signature block attached to page 1.
    story.append(PageBreak())

    # Compact Holter statistics overview, based on the supplied reference
    # format, placed before the numerical summary on page 2.
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph("HOLTER STATISTICS OVERVIEW", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=3*mm))

    overview_chart_path = _generate_hourly_hr_chart(session_dir, summary.get('hourly_hr', {}))
    if overview_chart_path and os.path.exists(overview_chart_path):
        story.append(Image(overview_chart_path, width=170*mm, height=44*mm))

    _snapshot_ve = locals().get('_ve_patterns', {'single': 0, 'couplet': 0, 'run': 0})
    _snapshot_sve = locals().get('_sve_patterns', {'single': 0, 'couplet': 0, 'run': 0})
    _snapshot_ve_total = int(locals().get('_ve_total', 0) or 0)
    _snapshot_sve_total = int(locals().get('_sve_total', 0) or 0)
    _snapshot_pct = lambda value, total: f"{100.0 * value / total:.1f}%" if total else "0.0%"
    overview_data = [
        ['Rhythm', 'Min HR', 'Avg HR', 'Max HR', 'Single', 'Couplets', 'Runs', 'Total', '%'],
        ['VE', f"{min_hr:.0f}", f"{avg_hr:.0f}", f"{max_hr:.0f}",
         str(_snapshot_ve.get('single', 0)), str(_snapshot_ve.get('couplet', 0)),
         str(_snapshot_ve.get('run', 0)), str(_snapshot_ve_total),
         _snapshot_pct(_snapshot_ve_total, summary.get('total_beats', 0))],
        ['SVE', f"{min_hr:.0f}", f"{avg_hr:.0f}", f"{max_hr:.0f}",
         str(_snapshot_sve.get('single', 0)), str(_snapshot_sve.get('couplet', 0)),
         str(_snapshot_sve.get('run', 0)), str(_snapshot_sve_total),
         _snapshot_pct(_snapshot_sve_total, summary.get('total_beats', 0))],
        ['Pause (>2.0s)', '', '', '', '', '', '', str(summary.get('pauses', 0)), ''],
    ]
    overview_table = Table(
        overview_data,
        colWidths=[27*mm, 18*mm, 18*mm, 18*mm, 18*mm,
                   18*mm, 18*mm, 18*mm, 18*mm],
        repeatRows=1,
    )
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.2),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PANEL]),
        ('GRID', (0, 0), (-1, -1), 0.45, BORDER),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('SPAN', (1, 3), (7, 3)),
    ]))
    story.append(overview_table)

    # Detailed hourly statistics sheet. This mirrors the reference Holter
    # statistic layout: HR min/average/max plus VE/SVE breakdown and pauses.
    _hourly_stats = {}
    for _metric in _report_metrics:
        try:
            _hour = int(float(_metric.get('t', 0.0) or 0.0) // 3600)
        except (TypeError, ValueError):
            continue
        _row = _hourly_stats.setdefault(_hour, {
            'beats': 0, 'hr_min': [], 'hr_avg': [], 'hr_max': [],
            've': 0, 'sve': 0, 'pauses': 0,
        })
        _row['beats'] += int(_metric.get('beat_count', _metric.get('n_beats', 0)) or 0)
        for _key, _field in (('hr_min', 'hr_min'), ('hr_avg', 'hr_mean'), ('hr_max', 'hr_max')):
            try:
                _value = float(_metric.get(_field, 0.0) or 0.0)
                if _value > 0:
                    _row[_key].append(_value)
            except (TypeError, ValueError):
                pass
        _row['ve'] += int(_metric.get('ve_beats', 0) or 0)
        _row['sve'] += int(_metric.get('sve_beats', 0) or 0)
        _row['pauses'] += int(_metric.get('pauses', 0) or 0)

    _detail_rows = [[
        'Start time', 'Beats', 'HR', '', '',
        'VE premature', '', '', '', '', '',
        'SVE premature', '', '', '', '', '', 'Pause\n(>2.0s)'
    ], [
        '', '', 'Min', 'Avg', 'Max',
        'Single', 'Couplets', 'Triplets', 'Runs', 'Total', '%',
        'Single', 'Couplets', 'Triplets', 'Runs', 'Total', '%', ''
    ]]
    _detail_hours = max(1, int(np.ceil(float(summary.get('duration_sec', 0) or 0) / 3600.0)))
    for _hour in range(_detail_hours):
        _row = _hourly_stats.get(_hour, {})
        _mins = _row.get('hr_min', [])
        _avgs = _row.get('hr_avg', [])
        _maxs = _row.get('hr_max', [])
        _ve_count = int(_row.get('ve', 0) or 0)
        _sve_count = int(_row.get('sve', 0) or 0)
        _detail_rows.append([
            _format_system_time(session_dir, _hour * 3600),
            str(_row.get('beats', 0)),
            f"{min(_mins):.0f}" if _mins else '--',
            f"{sum(_avgs) / len(_avgs):.0f}" if _avgs else '--',
            f"{max(_maxs):.0f}" if _maxs else '--',
            '0', '0', '0', '0', str(_ve_count),
            _snapshot_pct(_ve_count, _row.get('beats', 0)),
            '0', '0', '0', '0', str(_sve_count),
            _snapshot_pct(_sve_count, _row.get('beats', 0)),
            str(_row.get('pauses', 0)),
        ])
    # Keep the sheet visually filled like a clinical statistic form while
    # retaining real rows at the top for the actual recording duration.
    for _ in range(max(0, 24 - _detail_hours)):
        _detail_rows.append([''] * 18)

    # Widened to use the full page width (was 169mm of the 180mm available
    # inside the 15mm margins) and given bigger font/padding so the hourly
    # grid reads clearly instead of looking cramped.
    _detail_table = Table(
        _detail_rows,
        colWidths=[20.2*mm, 10.7*mm, 8.5*mm, 8.5*mm, 8.5*mm,
                   8.5*mm, 9.6*mm, 9.6*mm, 8.5*mm, 9.6*mm, 8.5*mm,
                   8.5*mm, 9.6*mm, 9.6*mm, 8.5*mm, 9.6*mm, 8.5*mm, 14.9*mm],
        repeatRows=2,
    )
    _detail_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 1), GRAY),
        ('TEXTCOLOR', (0, 0), (-1, 1), DARK),
        ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('SPAN', (0, 0), (0, 1)),
        ('SPAN', (1, 0), (1, 1)),
        ('SPAN', (2, 0), (4, 0)),
        ('SPAN', (5, 0), (10, 0)),
        ('SPAN', (11, 0), (16, 0)),
        ('SPAN', (17, 0), (17, 1)),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 2), (-1, -1), [colors.white, GRAY]),
        ('PADDING', (0, 0), (-1, -1), 4.5),
    ]))
    story.append(Spacer(1, 4*mm))
    story.append(_detail_table)

    # -------------------------------------------------------------------------
    # PAGE 3: ST ANALYSIS
    # -------------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("ST ANALYSIS", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))

    # ST episodes from ischemia_detection (12 leads; limb leads >= 1 mm, chest
    # leads >= 2 mm elevation, >= 1 mm depression, any 2 leads of one group,
    # >= 1 min). Per episode 'st_lead_mm' holds the leads that met the
    # criteria and their worst value in mm (10 mm = 1 mV).
    _st_episodes = _load_st_episodes(session_dir)
    _st_dep = [e for e in _st_episodes if e.get('label') == 'ST Depression']
    _st_elev = [e for e in _st_episodes if e.get('label') == 'ST Elevation']
    _st_channel = '12'

    def _st_group_summary(eps, elevation):
        """[episodes, total duration, max (mV), start time of the episode
        holding that max] for one ST type."""
        if not eps:
            return ['0', '-', '-', '-']
        total = sum(max(0.0, float(e['end_sec']) - float(e['start_sec'])) for e in eps)
        best_val, best_ep = None, None
        for e in eps:
            for mm in (e.get('st_lead_mm') or {}).values():
                mv = float(mm) / 10.0
                if best_val is None or (mv > best_val if elevation else mv < best_val):
                    best_val, best_ep = mv, e
        return [
            str(len(eps)),
            _sec_to_hms(total),
            f"{best_val:+.2f}" if best_val is not None else '-',
            _format_system_time(session_dir, float(best_ep['start_sec'])) if best_ep is not None else '-',
        ]

    _st_summary_data = [
        ['Channel', 'ST Depression', '', '', '', 'ST Elevation', '', '', ''],
        ['', 'Episodes', 'Duration', 'Max (mV)', 'Start Time',
         'Episodes', 'Duration', 'Max (mV)', 'Start Time'],
        [_st_channel] + _st_group_summary(_st_dep, False) + _st_group_summary(_st_elev, True),
    ]
    _st_summary_table = Table(_st_summary_data,
                              colWidths=[15*mm, 15*mm, 20*mm, 17*mm, 23*mm,
                                         15*mm, 20*mm, 17*mm, 23*mm])
    _st_summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), GRAY),
        ('GRID', (0, 0), (-1, -1), 0.5, DARK),
        ('FONTNAME', (0, 0), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('SPAN', (1, 0), (4, 0)),
        ('SPAN', (5, 0), (8, 0)),
        ('SPAN', (0, 0), (0, 1)),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(_st_summary_table)
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("ST Event list", h2_style))

    # One row per lead of each episode that met the criteria: its value in
    # mV, the lead name (At), the episode's start - end system time, and
    # whether it is an elevation or a depression.
    _st_lead_order = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    _st_event_rows = [['Index', 'Channel', 'Type', 'mV', 'At', 'Duration', 'ST slope']]
    _st_idx = 0
    for _event in _st_episodes:
        _span = (f"{_format_system_time(session_dir, float(_event['start_sec']))} - "
                 f"{_format_system_time(session_dir, float(_event['end_sec']))}")
        _slope = 'Elevation' if _event.get('label') == 'ST Elevation' else 'Depression'
        _lead_mm = _event.get('st_lead_mm') or {}
        for _lead in sorted(_lead_mm, key=lambda n: _st_lead_order.index(n) if n in _st_lead_order else 99):
            _st_idx += 1
            _st_event_rows.append([
                str(_st_idx), _st_channel, '12 lead', f"{float(_lead_mm[_lead]) / 10.0:+.2f}",
                _lead, _span, _slope,
            ])
    # 20 rows (was 28) so the T wave section below fits on this page
    for _ in range(max(0, 20 - _st_idx)):
        _st_event_rows.append([''] * 7)
    _st_event_table = Table(_st_event_rows,
                            colWidths=[12*mm, 16*mm, 22*mm, 20*mm,
                                       18*mm, 46*mm, 26*mm],
                            repeatRows=1)
    _st_event_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GRAY),
        ('GRID', (0, 0), (-1, -1), 0.45, DARK),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, GRAY]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(_st_event_table)

    # T wave section (T Inversion, ischemia_detection): one row per episode,
    # small, under the ST Event list on the same page. mV = the most negative
    # T wave of the episode, At = the leads that met the criteria, Duration =
    # the episode's start - end system time.
    _t_episodes = _load_t_episodes(session_dir)
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph("T wave", h2_style))
    _t_cell_style = ParagraphStyle('TWaveCell', parent=styles['Normal'], fontSize=7, leading=8.5,
                                   alignment=TA_CENTER)
    _t_rows = [['Index', 'Channel', 'Type', 'Max (mV)', 'At', 'Duration', 'T wave']]
    _T_MAX_ROWS = 4
    for _i, _ep in enumerate(_t_episodes[:_T_MAX_ROWS], start=1):
        _t_leads = sorted(_ep.get('st_lead_mm') or {},
                          key=lambda n: _st_lead_order.index(n) if n in _st_lead_order else 99)
        _t_rows.append([
            str(_i), _st_channel, '12 lead', f"{float(_ep.get('st_mv', 0.0)):+.2f}",
            Paragraph(', '.join(_t_leads), _t_cell_style),
            f"{_format_system_time(session_dir, float(_ep['start_sec']))} - "
            f"{_format_system_time(session_dir, float(_ep['end_sec']))}",
            'Inversion',
        ])
    if len(_t_episodes) > _T_MAX_ROWS:
        _t_rows.append(['', '', '', '', f"+{len(_t_episodes) - _T_MAX_ROWS} more episodes", '', ''])
    for _ in range(max(0, 3 - (len(_t_rows) - 1))):
        _t_rows.append([''] * 7)
    _t_table = Table(_t_rows,
                     colWidths=[11*mm, 15*mm, 20*mm, 18*mm, 40*mm, 36*mm, 20*mm],
                     repeatRows=1)
    _t_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GRAY),
        ('GRID', (0, 0), (-1, -1), 0.45, DARK),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, GRAY]),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(_t_table)

    # -------------------------------------------------------------------------
    # PAGE 4: HRV TIME DOMAIN REPORT
    # -------------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("HRV TIME DOMAIN REPORT", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))

    _hrv_total_beats = int(summary.get('total_beats', 0) or 0)
    _hrv_avg_hr = float(summary.get('avg_hr', 0) or 0)
    _hrv_mean_nn = (60000.0 / _hrv_avg_hr) if _hrv_avg_hr > 0 else 0.0
    _hrv_sdnn = float(summary.get('sdnn', 0) or 0)
    _hrv_rmssd = float(summary.get('rmssd', 0) or 0)
    _hrv_pnn50 = float(summary.get('pnn50', 0) or 0)
    _hrv_triidx = float(summary.get('triidx', 0) or 0)
    _hrv_duration = _grid_hms(float(summary.get('duration_sec', 0) or 0))
    # Keep the report tables readable: the full ISO timestamp is too long
    # for the compact "Start at" column used by the reference layout.
    _hrv_start_display = str(st_time_str).split()[-1] if st_time_str else '--'
    _hrv_plot_path = _generate_hrv_rr_plot(session_dir, _report_metrics)
    if _hrv_plot_path and os.path.exists(_hrv_plot_path):
        story.append(Image(_hrv_plot_path, width=78*mm, height=52*mm))
    _hrv_time_rows = [[
        'Type', 'Start at', 'Duration', 'NNs', 'Mean NN\nms', 'SDNN\nms',
        'SDANN\nms', 'SDNNIDX\nms', 'TRI IDX', 'rMSSD\nms', 'pNN50'
    ], [
        'Entire', _hrv_start_display, _hrv_duration, str(_hrv_total_beats),
        f"{_hrv_mean_nn:.0f}", f"{_hrv_sdnn:.0f}", '--', '--',
        f"{_hrv_triidx:.2f}", f"{_hrv_rmssd:.0f}", f"{_hrv_pnn50:.2f}%"
    ], [
        'D1', _hrv_start_display, _hrv_duration, str(_hrv_total_beats),
        f"{_hrv_mean_nn:.0f}", f"{_hrv_sdnn:.0f}", '--', '--',
        f"{_hrv_triidx:.2f}", f"{_hrv_rmssd:.0f}", f"{_hrv_pnn50:.2f}%"
    ], [
        'Hour', _hrv_start_display, _hrv_duration, str(_hrv_total_beats),
        f"{_hrv_mean_nn:.0f}", f"{_hrv_sdnn:.0f}", '--', '--',
        f"{_hrv_triidx:.2f}", f"{_hrv_rmssd:.0f}", f"{_hrv_pnn50:.2f}%"
    ]]
    for _ in range(20):
        _hrv_time_rows.append([''] * 11)
    _hrv_time_table = Table(
        _hrv_time_rows,
        colWidths=[15*mm, 24*mm, 16*mm, 14*mm, 17*mm, 15*mm,
                   15*mm, 17*mm, 14*mm, 16*mm, 17*mm],
        repeatRows=1,
    )
    _hrv_time_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, GRAY]),
        ('GRID', (0, 0), (-1, -1), 0.45, BORDER),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(_hrv_time_table)

    # -------------------------------------------------------------------------
    # PAGE 5: HRV FREQUENCY DOMAIN REPORT
    # -------------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("HRV FREQUENCY DOMAIN REPORT", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))

    _vlf = float(summary.get('vlf_power', 0) or 0)
    _lf = float(summary.get('lf_power', 0) or 0)
    _hf = float(summary.get('hf_power', 0) or 0)
    _lf_hf = float(summary.get('lf_hf_ratio', 0) or 0)
    _hrv_freq_rows = [[
        'Type', 'Start at', 'Duration', 'Total Power', 'VLF', 'LF', 'HF',
        'LFnu', 'HFnu', 'LF/HF'
    ], [
        'Entire', _hrv_start_display, _hrv_duration, '--', f"{_vlf:.3f}",
        f"{_lf:.3f}", f"{_hf:.3f}", '--', '--', f"{_lf_hf:.3f}"
    ]]
    for _ in range(27):
        _hrv_freq_rows.append([''] * 10)
    _hrv_freq_table = Table(
        _hrv_freq_rows,
        colWidths=[16*mm, 25*mm, 18*mm, 25*mm, 15*mm,
                   15*mm, 15*mm, 15*mm, 15*mm, 21*mm],
        repeatRows=1,
    )
    _hrv_freq_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TEAL),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, GRAY]),
        ('GRID', (0, 0), (-1, -1), 0.45, BORDER),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(_hrv_freq_table)

    # Numerical summary is intentionally a separate page (page 6).
    story.append(PageBreak())
    
    # =========================================================================
    # NUMERICAL SUMMARY TABLE
    # =========================================================================
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph("NUMERICAL SUMMARY TABLE", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))
    
    story.append(Paragraph("2. ARRHYTHMIA SUMMARY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    arrhy_hdr_style = ParagraphStyle('ArrhyHdr', parent=styles['Normal'],
                                     fontSize=9, leading=11, textColor=colors.white, bold=True)
    arrhy_hdr_center = ParagraphStyle('ArrhyHdrCenter', parent=styles['Normal'],
                                      fontSize=9, leading=11, textColor=colors.white, bold=True, alignment=TA_CENTER)
    arrhy_cell_style = ParagraphStyle('ArrhyCell', parent=styles['Normal'],
                                      fontSize=8.5, leading=11, textColor=DARK)
    arrhy_cell_center = ParagraphStyle('ArrhyCellCenter', parent=styles['Normal'],
                                       fontSize=8.5, leading=11, textColor=DARK, alignment=TA_CENTER)

    if filtered_arrhy_counts:
        arrhy_data = [[
            Paragraph("Arrhythmia Type", arrhy_hdr_style),
            Paragraph("Episodes", arrhy_hdr_center),
            Paragraph("Occurring Time", arrhy_hdr_style),
            Paragraph("Source", arrhy_hdr_center)
        ]]

        for label, count in sorted(filtered_arrhy_counts.items(), key=lambda x: -x[1]):
            # Determine source
            source = []
            if label in arrhy_counts or label in auto_segment_counts:
                source.append("Auto")
            if label in manual_arrhy_counts:
                source.append("Manual")
            source_str = ", ".join(dict.fromkeys(source)) or "Auto"

            occurring = []
            if label in auto_segment_times:
                occurring.extend(auto_segment_times[label])
            if label in manual_arrhy_counts:
                for seg in manual_segments:
                    if str(seg.get('label', '')) == label:
                        s_t = float(seg.get('start_sec', 0.0) or 0.0)
                        e_t = float(seg.get('end_sec', s_t) or s_t)
                        s_str = _format_system_time(session_dir, s_t)
                        e_str = _format_system_time(session_dir, e_t)
                        if e_t > s_t + 0.5:
                            occurring.append(f"{s_str} - {e_str}")
                        else:
                            occurring.append(s_str)
                for mb in manual_beats:
                    if mb.get('label') == label:
                        occurring.append(_format_system_time(session_dir, float(mb.get('timestamp', 0.0) or 0.0)))

            # Format each occurring time on its own separate line below the previous one
            occ_html = "<br/>".join(occurring) if occurring else "-"

            type_para = Paragraph(f"<b>{label}</b>", arrhy_cell_style)
            episodes_para = Paragraph(str(count), arrhy_cell_center)
            occ_para = Paragraph(occ_html, arrhy_cell_style)
            source_para = Paragraph(source_str, arrhy_cell_center)

            arrhy_data.append([type_para, episodes_para, occ_para, source_para])

        arrhy_table = Table(arrhy_data, colWidths=[55*mm, 20*mm, 75*mm, 25*mm])
        arrhy_table.setStyle(TableStyle([
            # Red is reserved for clinically abnormal status values; the
            # table itself uses the report's calm clinical navy treatment.
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PANEL]),
            ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(arrhy_table)
    else:
        story.append(Paragraph("No significant arrhythmias detected during this recording.", body_style))

    # Build complete timeline events (Badge labels for auto-detection + exact Manual Markings)
    timeline_events = _build_timeline_events(
        session_dir,
        auto_segment_events=_report_auto_segment_events,
        metrics=_report_metrics,
    )
    print(f"[HolterReport] Total timeline events: {len(timeline_events)}")

    if timeline_events:
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("2B. EVENT TIMELINE", h2_style))
        timeline_rows = [[
            Paragraph("Time", arrhy_hdr_style),
            Paragraph("Label", arrhy_hdr_style),
            Paragraph("Source", arrhy_hdr_center)
        ]]
        seen_events = set()
        for event in timeline_events:
            t_str = _format_system_time(session_dir, float(event.get("timestamp", 0.0) or 0.0))
            lbl   = str(event.get("label", event.get("event_type", "Event")))
            src   = str(event.get("source", ""))
            dedup_key = (t_str, lbl, src)
            if dedup_key in seen_events:
                continue
            seen_events.add(dedup_key)
            t_para = Paragraph(t_str, arrhy_cell_style)
            lbl_para = Paragraph(lbl, arrhy_cell_style)
            src_para = Paragraph(src, arrhy_cell_center)
            timeline_rows.append([t_para, lbl_para, src_para])

        timeline_table = Table(timeline_rows, colWidths=[40*mm, 105*mm, 30*mm],
                               repeatRows=1)          # repeat header on each page
        timeline_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), TEAL),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 7),     # compact font
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5FAFF')]),
            ('GRID',       (0, 0), (-1, -1), 0.3, BORDER),
            ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING',    (0, 0), (-1, -1), 2),     # tight padding for more rows per page
        ]))
        story.append(timeline_table)


    # Keep the complete HRV section together on its own page. This prevents
    # the heading/header row from being stranded at the bottom of the prior
    # numerical-summary page.
    story.append(PageBreak())

    #    HRV ANALYSIS                                                    
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph("HEART RATE VARIABILITY SUMMARY (HRV)", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))

    sdnn  = summary.get('sdnn', 0)
    rmssd = summary.get('rmssd', 0)
    pnn50 = summary.get('pnn50', 0)
    triidx = summary.get('triidx', 0)
    vlf = summary.get('vlf_power', 0)
    lf = summary.get('lf_power', 0)
    hf = summary.get('hf_power', 0)
    lf_hf = summary.get('lf_hf_ratio', 0)

    def hrv_status(sdnn):
        if sdnn > 100: return 'Normal', GREEN
        if sdnn > 50:  return 'Borderline', colors.orange
        return 'Reduced', RED

    hrv_label, hrv_color = hrv_status(sdnn)

    hrv_data = [
        ['Metric', 'Value', 'Reference', 'Status'],
        ['SDNN',   f"{sdnn:.1f} ms",  '>100 ms',  hrv_label],
        ['rMSSD',  f"{rmssd:.1f} ms", '>42 ms',   'Normal' if rmssd > 42 else 'Low'],
        ['pNN50',  f"{pnn50:.2f}%",   '>20%',     'Normal' if pnn50 > 20 else 'Low'],
        ['TriIdx', f"{triidx:.2f}",   'Higher is better', '--'],
        ['VLF',    f"{vlf:.3f}",      '0.0033-0.04 Hz',   '--'],
        ['LF',     f"{lf:.3f}",       '0.04-0.15 Hz',     '--'],
        ['HF',     f"{hf:.3f}",       '0.15-0.40 Hz',     '--'],
        ['LF/HF',  f"{lf_hf:.3f}",    'Balance marker',   '--'],
    ]
    hrv_table = Table(hrv_data, colWidths=[40*mm, 35*mm, 35*mm, 30*mm])
    hrv_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('TEXTCOLOR', (3, 1), (3, 1), hrv_color),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(hrv_table)
    story.append(Spacer(1, 10*mm))
    
    #    QT / QTC ANALYSIS                                               
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph("QT / QTc ANALYSIS", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))
    
    jsonl_path = os.path.join(session_dir, 'metrics.jsonl')
    interval_stats = _compute_interval_stats(jsonl_path)

    story.append(Paragraph("3. ECG INTERVAL STATISTICS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    int_data = [['Interval', 'Mean', 'Std Dev', 'Min', 'Max', 'Normal Range']]
    for label, key, ref in [
        ('PR Interval',  'pr_ms',  '120--200 ms'),
        ('QRS Duration', 'qrs_ms', '60--120 ms'),
        ('QT Interval',  'qt_ms',  '350--450 ms'),
        ('QTc Interval', 'qtc_ms', '<440 ms'),
    ]:
        vals = interval_stats.get(key, [])
        if vals:
            int_data.append([label,
                             f"{np.mean(vals):.0f} ms",
                             f"{np.std(vals):.0f} ms",
                             f"{np.min(vals):.0f} ms",
                             f"{np.max(vals):.0f} ms",
                             ref])
        else:
            int_data.append([label, '--', '--', '--', '--', ref])

    int_table = Table(int_data, colWidths=[35*mm, 22*mm, 22*mm, 22*mm, 22*mm, 32*mm])
    int_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), TEAL),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(int_table)

    #    FULL DISCLOSURE ECG STRIPS                                              
    # User requested: "all waves should be attached to the report for how many time it has run all 12 lead data should go with time stamp"
    ecgh_path = os.path.join(session_dir, 'recording.ecgh')
    if os.path.exists(ecgh_path):
        story.append(PageBreak())
        # The waveform image already contains its own "Full Disclosure"
        # title. Keeping a second title flowable here can push the image to
        # the next page and leave an otherwise empty heading-only page.
        
        try:
            from .replay_engine import HolterReplayEngine
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            engine = HolterReplayEngine(ecgh_path)
            dur = int(engine.duration_sec)
            
            chunk_sec = 60.0 # 1 minute per row
            rows_per_page = 30 # 30 minutes per page
            fs = engine.fs if hasattr(engine, 'fs') else 250
            
            # --- Load manually marked QRS beats (saved by Full Disclosure dialog) ---
            manual_beats = []  # list of {timestamp, label, color}
            manual_beats_path = os.path.join(session_dir, 'manual_beats.json')
            if os.path.exists(manual_beats_path):
                try:
                    with open(manual_beats_path, 'r') as _mb_f:
                        raw_beats = json.load(_mb_f)
                        manual_beats = [
                            mb for mb in raw_beats 
                            if mb.get('is_manual', False) or (mb.get('marking_mode') is not None) or (mb.get('batch_id') is not None)
                        ]
                    print(f"[HolterReport] Loaded {len(manual_beats)} manual beats from {manual_beats_path}")
                except Exception as _mb_e:
                    print(f"[HolterReport] Could not load manual beats: {_mb_e}")
                    manual_beats = []
            
            n_chunks = int(np.ceil(dur / chunk_sec))
            if n_chunks == 0: n_chunks = 1

            source_mtime = os.path.getmtime(ecgh_path)
            if os.path.exists(manual_beats_path):
                source_mtime = max(source_mtime, os.path.getmtime(manual_beats_path))
            
            # Iterate by pages
            for page_idx in range(int(np.ceil(n_chunks / rows_per_page))):
                start_chunk = page_idx * rows_per_page
                end_chunk = min((page_idx + 1) * rows_per_page, n_chunks)
                actual_rows = end_chunk - start_chunk
                
                # Scale height dynamically so short recordings don't create huge blank spaces
                fig_height = 10.5 * (max(1, actual_rows) / rows_per_page)

                chart_path = os.path.join(session_dir, f'full_disclosure_page_{page_idx}.png')
                # Full-disclosure rendering is expensive. Reuse a page image
                # when the recording/markings have not changed since it was
                # generated; this makes repeated Generate Report clicks fast.
                if (os.path.exists(chart_path)
                        and os.path.getmtime(chart_path) >= source_mtime):
                    story.append(Image(chart_path, width=190*mm, height=250*mm *
                                       (max(1, actual_rows) / rows_per_page)))
                    if page_idx < int(np.ceil(n_chunks / rows_per_page)) - 1:
                        story.append(PageBreak())
                    continue
                
                fig, axes = plt.subplots(actual_rows, 1, figsize=(8.27, fig_height), gridspec_kw={'hspace': 0.0})
                if actual_rows == 1: axes = [axes]
                
                # Title
                fig.suptitle("Full Disclosure(Lead II)", fontsize=14, fontweight='bold', y=0.98 if actual_rows > 5 else 1.1)
                
                for r in range(actual_rows):
                    chunk_i = start_chunk + r
                    start_t = chunk_i * chunk_sec
                    end_t = start_t + chunk_sec
                    engine._current_sec = start_t + (chunk_sec/2)
                    # Full Disclosure renders Lead II only. Reading one lead
                    # avoids decoding all 12 channels for every minute while
                    # keeping the report output unchanged.
                    ch2_data = engine.get_lead_data(1, window_sec=chunk_sec)
                    
                    if ch2_data is None or ch2_data.size == 0:
                        continue

                    x_time = np.linspace(start_t, end_t, len(ch2_data))
                    
                    ax = axes[r]
                    ax.plot(x_time, ch2_data, color='black', linewidth=0.5)
                    
                    # --- Overlay manually marked QRS beats for this time window ---
                    if manual_beats:
                        # Compute y-range for placing label text above the waveform
                        y_min = float(np.min(ch2_data)) if len(ch2_data) > 0 else -1.0
                        y_max = float(np.max(ch2_data)) if len(ch2_data) > 0 else 1.0
                        y_range = y_max - y_min if (y_max - y_min) > 0 else 1.0
                        tick_top = y_max + y_range * 0.05   # slightly above the signal
                        text_y   = y_max + y_range * 0.15   # label sits above the tick
                        
                        for mb in manual_beats:
                            ts = float(mb.get('timestamp', -1.0))
                            if start_t <= ts < end_t:
                                lbl   = str(mb.get('label', 'N'))
                                # Convert stored hex color (#RRGGBB) to matplotlib-compatible tuple
                                raw_col = str(mb.get('color', '#FF0000'))
                                try:
                                    import matplotlib.colors as mcolors
                                    marker_color = mcolors.to_rgba(raw_col)
                                except Exception:
                                    marker_color = 'red'
                                
                                # Draw a short vertical tick line at the QRS position
                                ax.axvline(x=ts, color=marker_color, linewidth=0.8,
                                           alpha=0.85, ymin=0.85, ymax=1.0)
                                
                                # Draw the beat label and its actual system time above the tick
                                beat_time_str = _format_system_time(session_dir, ts)
                                label_text = f"{lbl}\n{beat_time_str}"
                                ax.text(ts, text_y, label_text,
                                        color=marker_color, fontsize=3.5,
                                        ha='center', va='bottom',
                                        fontweight='bold', clip_on=True)

                                # Highlight the QRS peak segment on the waveform itself
                                qrs_start_t = ts - 0.06
                                qrs_end_t = ts + 0.06
                                mask = (x_time >= qrs_start_t) & (x_time <= qrs_end_t)
                                if np.sum(mask) >= 2:
                                    ax.plot(x_time[mask], ch2_data[mask], color=marker_color, linewidth=1.0)
                    
                    # Format timestamp
                    time_str = _format_system_time(session_dir, start_t)
                    
                    ax.set_ylabel(time_str, fontsize=6, rotation=0, labelpad=25, va='center')
                    ax.axis('off') # remove borders
                    
                    # Add back the ylabel so it stays visible even with axis('off')
                    ax.text(-0.01, 0.5, time_str, transform=ax.transAxes, 
                            fontsize=8, fontweight='bold', va='center', ha='right')
                            
                plt.tight_layout(rect=[0.05, 0.02, 0.98, 0.92])
                
                plt.savefig(chart_path, dpi=140, bbox_inches='tight')
                plt.close(fig)
                
                # Scale Image height in PDF so it aligns cleanly at the top
                pdf_img_height = 250 * mm * (max(1, actual_rows) / rows_per_page)
                story.append(Image(chart_path, width=190*mm, height=pdf_img_height))
                
                if page_idx < int(np.ceil(n_chunks / rows_per_page)) - 1:
                    story.append(PageBreak())
                    
        except Exception as e:
            print(f"Error generating full disclosure: {e}")
            story.append(Paragraph(f"Could not generate waveforms: {e}", body_style))


    doc.build(story, onFirstPage=_draw_page_chrome, onLaterPages=_draw_page_chrome)
    print(f"[HolterReport] PDF saved: {output_path}")
    return output_path


#    Helper functions                                                             

def _generate_hrv_rr_plot(session_dir: str, metrics: list) -> str:
    """Create the compact RR-interval distribution used on the HRV page."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        rr_ms = []
        for metric in metrics or []:
            for value in (metric.get('rr_intervals_list') or []):
                try:
                    value = float(value)
                    if 250.0 <= value <= 3000.0:
                        rr_ms.append(value / 1000.0)
                except (TypeError, ValueError):
                    pass
        if not rr_ms:
            return ""

        plot_path = os.path.join(session_dir, 'hrv_rr_distribution.png')
        metrics_path = os.path.join(session_dir, 'metrics.jsonl')
        if (os.path.exists(plot_path)
                and (not os.path.exists(metrics_path)
                     or os.path.getmtime(plot_path) >= os.path.getmtime(metrics_path))):
            return plot_path

        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.hist(rr_ms, bins=30, color='#17324D', edgecolor='#17324D')
        ax.set_title('D1', fontsize=9, color='#17324D')
        ax.set_xlabel('RR interval (s)', fontsize=7)
        ax.set_ylabel('Count', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, axis='y', alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return plot_path
    except Exception as exc:
        print(f"[HolterReport] HRV plot error: {exc}")
        return ""


def _generate_hourly_hr_chart(session_dir: str, hourly_hr: dict) -> str:
    """Generate bar chart of hourly mean HR, save as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not hourly_hr:
            return ""

        chart_path = os.path.join(session_dir, 'hourly_hr_chart.png')
        metrics_path = os.path.join(session_dir, 'metrics.jsonl')
        if (os.path.exists(chart_path)
                and (not os.path.exists(metrics_path)
                     or os.path.getmtime(chart_path) >= os.path.getmtime(metrics_path))):
            return chart_path

        hours = sorted(hourly_hr.keys())
        values = [hourly_hr[h] for h in hours]

        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.bar(hours, values, color='#197C86', alpha=0.86, width=0.7)
        ax.axhline(y=60, color='#E65100', linestyle='--', alpha=0.7, label='60 bpm')
        ax.axhline(y=100, color='#B71C1C', linestyle='--', alpha=0.7, label='100 bpm')
        ax.set_xlabel('Hour of Recording', fontsize=9)
        ax.set_ylabel('Mean HR (bpm)', fontsize=9)
        ax.set_title('Hourly Heart Rate Trend', fontsize=10, fontweight='bold', color='#17324D')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(hours)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        plt.savefig(chart_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return chart_path
    except Exception as e:
        print(f"[HolterReport] Chart error: {e}")
        return ""


def _compute_interval_stats(jsonl_path: str) -> dict:
    """Load all interval values from JSONL for statistics."""
    stats = {'pr_ms': [], 'qrs_ms': [], 'qt_ms': [], 'qtc_ms': []}
    if not os.path.exists(jsonl_path):
        return stats
    try:
        with open(jsonl_path) as f:
            for line in f:
                m = json.loads(line.strip())
                for key in stats:
                    val = m.get(key, 0)
                    if val > 0:
                        stats[key].append(val)
    except Exception:
        pass
    return stats


def _auto_conclusion(summary: dict, filtered_arrhy_counts: Optional[dict] = None) -> str:
    """
    Generate an auto-summary conclusion text.
    User request: Only show rhythm/arrhythmia findings, nothing extra.
    """
    lines = []
    arrhy = filtered_arrhy_counts if filtered_arrhy_counts is not None else summary.get('arrhythmia_counts', {})

    if not arrhy:
        lines.append("Normal sinus rhythm throughout the recording with no significant arrhythmias.")
    else:
        # Print detected issues/arrhythmias
        arrhy_list = ', '.join(f"{k} ({v} episode{'s' if v > 1 else ''})" for k, v in arrhy.items())
        lines.append(f"Arrhythmias detected: {arrhy_list}.")

    return " ".join(lines)


def _generate_text_report(session_dir, patient_info, summary, output_path) -> str:
    """Fallback plain-text report when reportlab is unavailable."""
    lines = [
        "=" * 60,
        "COMPREHENSIVE ECG ANALYSIS REPORT",
        "=" * 60,
        f"Patient: {patient_info.get('name', 'Unknown')}",
        f"Doctor: {patient_info.get('doctor', '--')}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "SUMMARY",
        f"  Duration: {summary.get('duration_sec',0)/3600:.1f} hours",
        f"  Total Beats: {summary.get('total_beats',0):,}",
        f"  Avg HR: {summary.get('avg_hr',0):.0f} bpm",
        f"  Max HR: {summary.get('max_hr',0):.0f} bpm",
        f"  Min HR: {summary.get('min_hr',0):.0f} bpm",
        "",
        "HRV",
        f"  SDNN:  {summary.get('sdnn',0):.1f} ms",
        f"  rMSSD: {summary.get('rmssd',0):.1f} ms",
        f"  pNN50: {summary.get('pnn50',0):.2f}%",
        f"  TriIdx: {summary.get('triidx',0):.2f}",
        f"  VLF: {summary.get('vlf_power',0):.3f}",
        f"  LF: {summary.get('lf_power',0):.3f}",
        f"  HF: {summary.get('hf_power',0):.3f}",
        f"  LF/HF: {summary.get('lf_hf_ratio',0):.3f}",
        "",
        "ARRHYTHMIAS",
    ]
    # Calculate combined and filtered arrhythmia counts
    auto_segment_events = _load_auto_segment_events(session_dir)
    auto_segment_counts = {}
    for event in auto_segment_events:
        label = str(event.get('label', 'Unknown'))
        auto_segment_counts[label] = auto_segment_counts.get(label, 0) + 1

    manual_beats = []
    manual_beats_path = os.path.join(session_dir, 'manual_beats.json')
    if os.path.exists(manual_beats_path):
        try:
            with open(manual_beats_path, 'r') as _mb_f:
                manual_beats = json.load(_mb_f)
        except Exception:
            pass

    manual_segments = []
    manual_segments_path = os.path.join(session_dir, 'manual_segments.json')
    if os.path.exists(manual_segments_path):
        try:
            with open(manual_segments_path, 'r') as _ms_f:
                manual_segments = json.load(_ms_f)
        except Exception:
            pass

    manual_arrhy_counts = {}
    for mb in manual_beats:
        if not (mb.get('is_manual', False) or (mb.get('marking_mode') is not None) or (mb.get('batch_id') is not None)):
            continue
        lbl = mb.get('label', 'N')
        short_code = lbl
        if '(' in lbl and ')' in lbl:
            short_code = lbl.split('(')[1].split(')')[0]
        if short_code != 'N':
            manual_arrhy_counts[lbl] = manual_arrhy_counts.get(lbl, 0) + 1

    for seg in manual_segments:
        lbl = seg.get('label', 'Unknown')
        manual_arrhy_counts[lbl] = manual_arrhy_counts.get(lbl, 0) + 1

    arrhy_counts = summary.get('arrhythmia_counts', {})
    combined_arrhy_counts = {}
    for label, count in arrhy_counts.items():
        if label not in auto_segment_counts and not _detector_owns_label(label):
            combined_arrhy_counts[label] = count
    for label, count in auto_segment_counts.items():
        combined_arrhy_counts[label] = count
    for label, count in manual_arrhy_counts.items():
        combined_arrhy_counts[label] = combined_arrhy_counts.get(label, 0) + count

    excluded_summary_terms = (
        'long qt', 'prolonged qt', 'wide qrs', 'premature ventricular contraction', 'pvc',
        'second-degree', 'second degree', 'third-degree', 'third degree',
    )
    # Auto-segment AV blocks are reported, same as in the PDF summary.
    filtered_arrhy_counts = {
        k: v for k, v in combined_arrhy_counts.items()
        if (
            (k in auto_segment_counts and 'av block' in k.lower())
            or not any(term in k.lower() for term in excluded_summary_terms)
        )
        and not ('sinus rhythm' in k.lower() and 'normal sinus rhythm' not in k.lower())
    }

    if filtered_arrhy_counts:
        for label, count in sorted(filtered_arrhy_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {label}: {count} episode(s)")
    else:
        lines.append("  None detected")

    # Build complete timeline events (Badge labels for auto-detection + exact Manual Markings)
    timeline_events = _build_timeline_events(session_dir)
    
    if timeline_events:
        lines += ["", "EVENT TIMELINE"]
        for event in timeline_events:
            lines.append(
                f"  {_format_system_time(session_dir, float(event.get('timestamp', 0.0) or 0.0))} | "
                f"{event.get('label', event.get('event_type', 'Event'))} | "
                f"{event.get('source', '')}"
            )

    lines += ["", "=" * 60, "Physician Signature: _______________", "Date: _______________"]

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return output_path


def _sec_to_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# Called once per event/beat row while building the report (easily
# thousands of times on a long recording) -- caches recording_index.json's
# start_time per session_dir instead of re-reading and re-parsing that file
# (a per-second index, multiple MB on a 14h recording) on every call. Safe:
# the file is written once by ECGHFileWriter.finalize() and never changes
# for a completed session; the mtime check only guards a stale cache across
# repeated finalize()/reopen cycles within the same process.
_report_system_time_start_cache: dict = {}


def _report_recording_start_time(session_dir: str):
    index_path = os.path.join(session_dir, 'recording_index.json')
    try:
        mtime = os.path.getmtime(index_path)
    except OSError:
        return None
    cached = _report_system_time_start_cache.get(session_dir)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        with open(index_path, 'r') as f:
            start_time = json.load(f).get('start_time')
    except Exception as e:
        print(f"[HolterReport] Error reading system time: {e}")
        start_time = None
    _report_system_time_start_cache[session_dir] = (mtime, start_time)
    return start_time


def _format_system_time(session_dir: str, chunk_timestamp: float) -> str:
    """
    Convert chunk timestamp to system recording time.
    Reads start_time from recording.ecgh index and adds chunk timestamp.
    """
    start_time = _report_recording_start_time(session_dir)
    if start_time:
        system_time = datetime.fromtimestamp(start_time + chunk_timestamp)
        return system_time.strftime('%H:%M:%S')
    # Fallback to chunk time in HH:MM:SS format
    return _sec_to_hms(chunk_timestamp)


def _get_recording_start_end(session_dir: str, duration_sec: float) -> tuple:
    """Return recording start and end date-time strings."""
    start_time = None
    try:
        import json
        index_path = os.path.join(session_dir, 'recording_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            start_time = index_data.get('start_time')
    except Exception as e:
        print(f"[HolterReport] Error reading start time: {e}")

    if not start_time:
        ecgh_path = os.path.join(session_dir, 'recording.ecgh')
        if os.path.exists(ecgh_path):
            start_time = os.path.getmtime(ecgh_path) - duration_sec
        else:
            import time
            start_time = time.time() - duration_sec

    from datetime import datetime
    start_dt = datetime.fromtimestamp(start_time)
    end_dt = datetime.fromtimestamp(start_time + duration_sec)
    return start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')
