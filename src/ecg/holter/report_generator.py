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
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
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


def _build_timeline_events(session_dir: str) -> List[Dict[str, object]]:
    """
    Builds the event timeline for the report:
    1. Loads manual markings (manual_segments.json and manual_beats.json) - EXACT MANUAL MARKING LOGIC.
    2. Loads auto-detected metrics/beats and computes rich Badge labels (get_template_beats_for_badges).
    3. Retains 'Normal Sinus Rhythm' for auto-detection windows.
    4. Filters out older/generic auto labels (e.g. raw PAC, Second degree AV, Wide QRS, Long QT, etc.).
    5. Suppresses auto-detected events if they fall inside manually marked areas.
    6. Appends manual beat and segment markings (unaltered).
    7. Chronologically sorts all events.
    """
    # Load manual segments and beats to filter automated events within manually marked areas
    manual_segments = []
    manual_segments_path = os.path.join(session_dir, 'manual_segments.json')
    if os.path.exists(manual_segments_path):
        try:
            with open(manual_segments_path, 'r') as _ms_f:
                manual_segments = json.load(_ms_f)
            print(f"[HolterReport] Loaded {len(manual_segments)} manual segments for filtering")
        except Exception as _ms_e:
            print(f"[HolterReport] Could not load manual segments for filtering: {_ms_e}")

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
            print(f"[HolterReport] Loaded {len(manual_beats)} manual beats for filtering")
        except Exception as _mb_e:
            print(f"[HolterReport] Could not load manual beats for filtering: {_mb_e}")

    # Load metrics to extract all_beats and auto-detected rhythm (e.g. Normal Sinus Rhythm)
    metrics = load_metrics(session_dir)
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
    auto_events = []

    for metric in metrics or []:
        for b in metric.get('all_beats', []) or []:
            if isinstance(b, dict):
                all_beats.append(b)

        # Keep Normal Sinus Rhythm / Sinus Rhythm from auto detection
        base_t = float(metric.get('t', 0.0) or 0.0)
        for label in metric.get('arrhythmias', []) or []:
            lbl = str(label).strip()
            lbl_lower = lbl.lower()
            if 'sinus' in lbl_lower or 'normal' in lbl_lower:
                auto_events.append({
                    'timestamp': base_t,
                    'label': lbl,
                    'event_type': lbl,
                    'source': 'analysis',
                    'confidence': float(metric.get('quality', 1.0) or 1.0),
                })

    # Generate auto-detection badge events (SVT, VT, Couplet, Bigeminy, PVC, PAC, etc.)
    try:
        from .holter_summary_calc import get_template_beats_for_badges
        badges = get_template_beats_for_badges(all_beats, [])
        for b in badges:
            badge_name = b.get('name', '').strip()
            if not badge_name:
                continue
            # Filter for arrhythmia badge beats/patterns (V, S, AF, P, runs, couplets, bigeminy, etc.)
            if (b.get('code') in ['V', 'S', 'AF', 'P'] or 
                b.get('critical', False) or 
                any(k in badge_name.lower() for k in ['run', 'couplet', 'bigeminy', 'trigeminy', 'quadrigeminy', 'tachycardia', 'fibrillation', 'flutter'])):
                auto_events.append({
                    'timestamp': float(b.get('timestamp', 0.0) or 0.0),
                    'label': badge_name,
                    'event_type': b.get('code', 'Arrhythmia'),
                    'source': 'analysis',
                    'confidence': 1.0,
                })
    except Exception as _badge_e:
        print(f"[HolterReport] Error generating badge events: {_badge_e}")

    # Fallback if no beats in metrics: check load_events but keep only NSR
    if not auto_events and not all_beats:
        raw_events = load_events(session_dir)
        for ev in raw_events:
            ev_lbl = str(ev.get('label', ev.get('event_type', ''))).lower()
            if 'sinus' in ev_lbl or 'normal' in ev_lbl:
                auto_events.append(ev)

    # Filter auto-detected events if they fall inside manually marked areas
    filtered_events = []
    for event in auto_events:
        event_ts = float(event.get('timestamp', 0.0))
        should_filter = False

        # Check segment ranges
        for seg in manual_segments:
            start_sec = float(seg.get('start_sec', 0.0))
            end_sec = float(seg.get('end_sec', 0.0))
            if start_sec <= event_ts <= end_sec:
                should_filter = True
                break

        # Check parallel marking timestamps (within 0.15s tolerance)
        if not should_filter and manual_beats:
            for mb in manual_beats:
                mb_ts = float(mb.get('timestamp', 0.0))
                if abs(event_ts - mb_ts) < 0.15:
                    should_filter = True
                    break

        if not should_filter:
            filtered_events.append(event)

    timeline_events = filtered_events

    # Load manual beats and append non-normal ones to timeline (parallel manual marking)
    # [EXACT ORIGINAL MANUAL MARKING LOGIC]
    manual_beats_path = os.path.join(session_dir, 'manual_beats.json')
    if os.path.exists(manual_beats_path):
        try:
            with open(manual_beats_path, 'r') as _mb_f:
                manual_beats = json.load(_mb_f)
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
                        'label': label_text,
                        'event_type': lbl,
                        'source': 'Manual'
                    })
        except Exception as _mb_e:
            print(f"[HolterReport] Could not load manual beats for timeline: {_mb_e}")

    # Load manual segments and append to timeline (segment manual marking)
    # [EXACT ORIGINAL MANUAL MARKING LOGIC]
    manual_segments_path = os.path.join(session_dir, 'manual_segments.json')
    if os.path.exists(manual_segments_path):
        try:
            with open(manual_segments_path, 'r') as _ms_f:
                manual_segments = json.load(_ms_f)
            for seg in manual_segments:
                lbl = seg.get('label', 'Unknown')
                start_sec = seg.get('start_sec', 0.0)
                end_sec = seg.get('end_sec', 0.0)
                start_time_str = seg.get('start_time_str', '')
                end_time_str = seg.get('end_time_str', '')
                # Start-of-segment entry
                timeline_events.append({
                    'timestamp': float(start_sec),
                    'sort_ts': float(start_sec),
                    'label': f"Segment manual marked ({lbl})",
                    'event_type': lbl,
                    'source': 'Manual'
                })
                # End-of-segment entry
                timeline_events.append({
                    'timestamp': float(end_sec),
                    'sort_ts': float(start_sec) + 1e-6,
                    'label': f"Segment manual marked end ({lbl})",
                    'event_type': lbl,
                    'source': 'Manual'
                })
        except Exception as _ms_e:
            print(f"[HolterReport] Could not load manual segments for timeline: {_ms_e}")

    # Sort all events chronologically by timestamp (sort_ts overrides timestamp for manual segment end-rows)
    timeline_events = sorted(timeline_events, key=lambda x: float(x.get("sort_ts", x.get("timestamp", 0.0)) or 0.0))
    return timeline_events


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
    ORANGE = colors.HexColor('#E65100')
    DARK   = colors.HexColor('#1A1A2E')
    LIGHT  = colors.HexColor('#FFF8F0')
    GRAY   = colors.HexColor('#F5F5F5')
    GREEN  = colors.HexColor('#2E7D32')
    RED    = colors.HexColor('#B71C1C')
    BLUE   = colors.HexColor('#1565C0')

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=15*mm, leftMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
        title="Comprehensive ECG Analysis Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Normal'],
                                  fontSize=20, textColor=ORANGE, bold=True,
                                  alignment=TA_CENTER, spaceAfter=4*mm)
    h1_style = ParagraphStyle('H1', parent=styles['Normal'],
                               fontSize=13, textColor=DARK, bold=True,
                               spaceBefore=6*mm, spaceAfter=2*mm,
                               borderPad=(0, 0, 2, 0))
    h2_style = ParagraphStyle('H2', parent=styles['Normal'],
                               fontSize=11, textColor=BLUE, bold=True,
                               spaceBefore=4*mm, spaceAfter=1*mm)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=9, textColor=DARK, spaceAfter=1*mm)
    small_style = ParagraphStyle('Small', parent=styles['Normal'],
                                  fontSize=8, textColor=colors.gray)

    story = []

    #    PAGE 1: FINAL SUMMARY                                                   
    story.append(Paragraph("COMPREHENSIVE ECG ANALYSIS REPORT", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))

    # Patient info table
    dur_h = int(summary.get('duration_sec', 0) // 3600)
    dur_m = int((summary.get('duration_sec', 0) % 3600) // 60)
    pname = patient_info.get('name', patient_info.get('patient_name', 'Unknown'))
    
    st_time_str, end_time_str = _get_recording_start_end(session_dir, float(summary.get('duration_sec', 0)))
    
    pinfo_data = [
        ['Patient Name', pname,              'Recording Duration', f"{dur_h}h {dur_m}m"],
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
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, LIGHT]),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(pinfo_table)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("1. RECORDING SUMMARY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    total_beats = summary.get('total_beats', 0)
    avg_hr      = summary.get('avg_hr', 0)
    max_hr      = summary.get('max_hr', 0)
    min_hr      = summary.get('min_hr', 0)
    pauses      = summary.get('pauses', 0)
    longest_rr  = summary.get('longest_rr_ms', 0)

    stats_data = [
        ['Parameter', 'Value', 'Parameter', 'Value'],
        ['Total Beats', f"{total_beats:,}",      'Avg Heart Rate',    f"{avg_hr:.0f} bpm"],
        ['Max Heart Rate', f"{max_hr:.0f} bpm",  'Min Heart Rate',    f"{min_hr:.0f} bpm"],
        ['Longest RR', f"{longest_rr:.0f} ms",   'Pauses (RR>2s)',    str(pauses)],
        ['Avg Quality', f"{summary.get('avg_quality',1)*100:.0f}%",
         'Chunks Analyzed', str(summary.get('chunks_analyzed', 0))],
    ]
    stats_table = Table(stats_data, colWidths=[45*mm, 35*mm, 45*mm, 35*mm])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ORANGE),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME',   (2, 1), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(stats_table)

    story.append(Paragraph("Clinical Impression", h2_style))
    arrhy_counts = summary.get('arrhythmia_counts', {})
    top_events = ", ".join(f"{label} ({count})" for label, count in sorted(arrhy_counts.items(), key=lambda item: -item[1])[:4]) or "No significant arrhythmias detected"
    avg_quality = summary.get('avg_quality', 0) * 100
    impression_text = (
        f"This Comprehensive ECG Analysis study for <b>{pname}</b> covers <b>{dur_h}h {dur_m}m</b> with an average heart rate of "
        f"<b>{avg_hr:.0f} bpm</b> (minimum <b>{min_hr:.0f} bpm</b>, maximum <b>{max_hr:.0f} bpm</b>). "
        f"Overall signal quality was <b>{avg_quality:.1f}%</b>. The automated event summary shows: <b>{top_events}</b>."
    )
    story.append(Paragraph(impression_text, body_style))
    
    auto_conclusion = _auto_conclusion(summary)
    story.append(Paragraph(auto_conclusion, body_style))
    story.append(Spacer(1, 10*mm))

    # Signature box
    sig_data = [
        ['Reference Report Confirmed by', 'Doctor Name', 'Doctor Sign'],
        ['', patient_info.get('doctor', ''), ''],
    ]
    sig_table = Table(sig_data, colWidths=[70*mm, 50*mm, 60*mm])
    sig_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 0), (-1, 0), [GRAY]),
        ('MINROWHEIGHT', (0, 1), (-1, 1), 20*mm),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(sig_table)
    
    #    NUMERICAL SUMMARY TABLE                                         
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph("NUMERICAL SUMMARY TABLE", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))
    
    story.append(Paragraph("2. ARRHYTHMIA SUMMARY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    # Load manual beats and segments to include in arrhythmia summary
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
        # Extract short code from full label name (e.g., "Normal(N)" -> "N")
        short_code = lbl
        if '(' in lbl and ')' in lbl:
            short_code = lbl.split('(')[1].split(')')[0]
        if short_code != 'N':
            # Use full label as key for display
            manual_arrhy_counts[lbl] = manual_arrhy_counts.get(lbl, 0) + 1

    for seg in manual_segments:
        lbl = seg.get('label', 'Unknown')
        manual_arrhy_counts[lbl] = manual_arrhy_counts.get(lbl, 0) + 1

    # Merge auto-detected and manual arrhythmia counts
    combined_arrhy_counts = dict(arrhy_counts)
    for label, count in manual_arrhy_counts.items():
        combined_arrhy_counts[label] = combined_arrhy_counts.get(label, 0) + count

    # Keep the summary focused on clinically significant arrhythmias.  Rhythm
    # context and beat/conduction classifications below are intentionally not
    # included in this table (they may still appear elsewhere in the report).
    excluded_summary_terms = (
        'long qt',
        'wide qrs',
        'premature ventricular contraction',
        'pvc',
        'second-degree',
        'second degree',
        'third-degree',
        'third degree',
    )
    filtered_arrhy_counts = {}
    for label, count in combined_arrhy_counts.items():
        label_lower = label.lower()
        # Retain Normal Sinus Rhythm, but omit the generic Sinus Rhythm label.
        is_generic_sinus_rhythm = (
            'sinus rhythm' in label_lower
            and 'normal sinus rhythm' not in label_lower
        )
        if is_generic_sinus_rhythm or any(term in label_lower for term in excluded_summary_terms):
            continue
        filtered_arrhy_counts[label] = count

    if filtered_arrhy_counts:
        arrhy_data = [['Arrhythmia Type', 'Episodes', 'Burden', 'Source']]
        total_chunks = max(1, summary.get('chunks_analyzed', 1))
        for label, count in sorted(filtered_arrhy_counts.items(), key=lambda x: -x[1]):
            burden = f"{count / total_chunks * 100:.1f}%"
            # Determine source
            source = []
            if label in arrhy_counts:
                source.append("Auto")
            if label in manual_arrhy_counts:
                source.append("Manual")
            source_str = ", ".join(source)
            arrhy_data.append([label, str(count), burden, source_str])

        arrhy_table = Table(arrhy_data, colWidths=[80*mm, 25*mm, 25*mm, 20*mm])
        arrhy_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), RED),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#FFEBEE')]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ('PADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(arrhy_table)
    else:
        story.append(Paragraph("No significant arrhythmias detected during this recording.", body_style))

    # Build complete timeline events (Badge labels for auto-detection + exact Manual Markings)
    timeline_events = _build_timeline_events(session_dir)
    print(f"[HolterReport] Total timeline events: {len(timeline_events)}")

    if timeline_events:
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("2B. EVENT TIMELINE", h2_style))
        timeline_rows = [["Time", "Label", "Source"]]
        seen_events = set()
        for event in timeline_events:
            t_str = _format_system_time(session_dir, float(event.get("timestamp", 0.0) or 0.0))
            lbl   = str(event.get("label", event.get("event_type", "Event")))
            src   = str(event.get("source", ""))
            dedup_key = (t_str, lbl, src)
            if dedup_key in seen_events:
                continue
            seen_events.add(dedup_key)
            timeline_rows.append([t_str, lbl, src])


        timeline_table = Table(timeline_rows, colWidths=[35*mm, 100*mm, 25*mm],
                               repeatRows=1)          # repeat header on each page
        timeline_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), BLUE),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0, 0), (-1, -1), 7),     # compact font
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5FAFF')]),
            ('GRID',       (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ('PADDING',    (0, 0), (-1, -1), 2),     # tight padding for more rows per page
        ]))
        story.append(timeline_table)


    #    HOURLY ANALYSIS                                                 
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph("HOURLY ANALYSIS", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))
    
    story.append(Paragraph("3. HOURLY HEART RATE TREND", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

    hourly_chart_path = _generate_hourly_hr_chart(session_dir, summary.get('hourly_hr', {}))
    if hourly_chart_path and os.path.exists(hourly_chart_path):
        story.append(Image(hourly_chart_path, width=170*mm, height=55*mm))
    else:
        story.append(Paragraph("Hourly HR chart not available.", small_style))

    #    HRV ANALYSIS                                                    
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph("HEART RATE VARIABILITY (HRV)", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))
    
    story.append(Paragraph("4. HRV TIME DOMAIN", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ORANGE, spaceAfter=2*mm))

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
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('TEXTCOLOR', (3, 1), (3, 1), hrv_color),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(hrv_table)
    
    #    QT / QTC ANALYSIS                                               
    story.append(Spacer(1, 15*mm))
    story.append(Paragraph("QT / QTc ANALYSIS", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))
    
    jsonl_path = os.path.join(session_dir, 'metrics.jsonl')
    interval_stats = _compute_interval_stats(jsonl_path)

    story.append(Paragraph("5. ECG INTERVAL STATISTICS", h1_style))
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
        ('BACKGROUND', (0, 0), (-1, 0), GREEN),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',   (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(int_table)

    #    FULL DISCLOSURE ECG STRIPS                                              
    # User requested: "all waves should be attached to the report for how many time it has run all 12 lead data should go with time stamp"
    ecgh_path = os.path.join(session_dir, 'recording.ecgh')
    if os.path.exists(ecgh_path):
        story.append(PageBreak())
        story.append(Paragraph("FULL DISCLOSURE ECG", title_style))
        story.append(HRFlowable(width="100%", thickness=2, color=ORANGE, spaceAfter=4*mm))
        
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
            
            # Iterate by pages
            for page_idx in range(int(np.ceil(n_chunks / rows_per_page))):
                start_chunk = page_idx * rows_per_page
                end_chunk = min((page_idx + 1) * rows_per_page, n_chunks)
                actual_rows = end_chunk - start_chunk
                
                # Scale height dynamically so short recordings don't create huge blank spaces
                fig_height = 10.5 * (max(1, actual_rows) / rows_per_page)
                
                fig, axes = plt.subplots(actual_rows, 1, figsize=(8.27, fig_height), gridspec_kw={'hspace': 0.0})
                if actual_rows == 1: axes = [axes]
                
                # Title
                fig.suptitle("Full Disclosure(Lead II)", fontsize=14, fontweight='bold', y=0.98 if actual_rows > 5 else 1.1)
                
                for r in range(actual_rows):
                    chunk_i = start_chunk + r
                    start_t = chunk_i * chunk_sec
                    end_t = start_t + chunk_sec
                    engine._current_sec = start_t + (chunk_sec/2)
                    data = engine.get_all_leads_data(window_sec=chunk_sec)
                    
                    if data is None or data.shape[0] == 0:
                        continue
                        
                    ch2_data = data[1] # Lead II
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
                
                chart_path = os.path.join(session_dir, f'full_disclosure_page_{page_idx}.png')
                plt.savefig(chart_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                
                # Scale Image height in PDF so it aligns cleanly at the top
                pdf_img_height = 250 * mm * (max(1, actual_rows) / rows_per_page)
                story.append(Image(chart_path, width=190*mm, height=pdf_img_height))
                
                if page_idx < int(np.ceil(n_chunks / rows_per_page)) - 1:
                    story.append(PageBreak())
                    
        except Exception as e:
            print(f"Error generating full disclosure: {e}")
            story.append(Paragraph(f"Could not generate waveforms: {e}", body_style))


    doc.build(story)
    print(f"[HolterReport] PDF saved: {output_path}")
    return output_path


#    Helper functions                                                             

def _generate_hourly_hr_chart(session_dir: str, hourly_hr: dict) -> str:
    """Generate bar chart of hourly mean HR, save as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if not hourly_hr:
            return ""

        hours = sorted(hourly_hr.keys())
        values = [hourly_hr[h] for h in hours]

        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.bar(hours, values, color='#1565C0', alpha=0.8, width=0.7)
        ax.axhline(y=60, color='orange', linestyle='--', alpha=0.7, label='60 bpm')
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100 bpm')
        ax.set_xlabel('Hour of Recording', fontsize=9)
        ax.set_ylabel('Mean HR (bpm)', fontsize=9)
        ax.set_title('Hourly Heart Rate Trend', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(hours)
        ax.tick_params(labelsize=8)
        plt.tight_layout()

        chart_path = os.path.join(session_dir, 'hourly_hr_chart.png')
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


def _auto_conclusion(summary: dict) -> str:
    """
    Generate an auto-summary conclusion text.
    User request: Only show rhythm/arrhythmia findings, nothing extra.
    """
    lines = []
    arrhy = summary.get('arrhythmia_counts', {})

    if not arrhy:
        lines.append("Normal sinus rhythm.")
    else:
        # Print detected issues/arrhythmias
        arrhy_list = ', '.join(f"{k}" for k, v in arrhy.items())
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
    arrhy = summary.get('arrhythmia_counts', {})
    if arrhy:
        for label, count in arrhy.items():
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


def _format_system_time(session_dir: str, chunk_timestamp: float) -> str:
    """
    Convert chunk timestamp to system recording time.
    Reads start_time from recording.ecgh index and adds chunk timestamp.
    """
    try:
        import json
        index_path = os.path.join(session_dir, 'recording_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            start_time = index_data.get('start_time')
            if start_time:
                from datetime import datetime
                system_time = datetime.fromtimestamp(start_time + chunk_timestamp)
                return system_time.strftime('%H:%M:%S')
    except Exception as e:
        print(f"[HolterReport] Error reading system time: {e}")
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
