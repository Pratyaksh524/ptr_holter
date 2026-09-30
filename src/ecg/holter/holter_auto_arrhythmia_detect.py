"""
ecg/holter/holter_auto_arrhythmia_detect.py
===========================================
Auto-detection module for arrhythmias in Holter full disclosure view.

This module provides functionality to automatically detect arrhythmias from
ECG waveforms using the arrhythmia_detector module and add them as structured
events for waveform coloring.
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional


def detect_arrhythmias(
    reader,
    progress_callback: Optional[Callable[[int], None]] = None
) -> List[Dict[str, Any]]:
    """
    Auto-detect arrhythmias from the waveform using arrhythmia_detector.py.
    
    Args:
        reader: ECGHFileReader instance with read_range method
        progress_callback: Optional callback function(int) for progress updates (0-100)
    
    Returns:
        List of detected arrhythmia segments with:
            - start_sec: Start time in seconds
            - end_sec: End time in seconds
            - label: Short label code (V, AF, S, P, X)
            - color: Color hex code for waveform coloring
            - detected_rhythm: Full arrhythmia name
    """
    from ecg.arrhythmia_detector import analyze_ecg
    
    # Raw 10-second detections are only candidates.  Do not paint them
    # immediately: one noisy window must not become a complete arrhythmia
    # segment.  Candidates are confirmed below by comparing them with the
    # previous overlapping window.
    candidate_segments = []
    
    # Get sampling rate and duration from reader
    fs = getattr(reader, 'fs', 500)
    if fs is None or fs <= 0:
        fs = 500
    
    total_duration = getattr(reader, 'duration_sec', 0)
    if total_duration <= 0:
        print("[Auto Arrhythmia Detect] Invalid duration")
        return []
    
    # Get lead names from reader
    lead_names = getattr(reader, 'lead_names', [])
    if not lead_names:
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    # Process in 10-second windows with 5-second overlap
    window_size = 10.0  # seconds
    overlap = 5.0  # seconds
    step_size = window_size - overlap
    
    # Process each window
    current_time = 0.0
    window_count = 0
    
    while current_time < total_duration:
        window_end = min(current_time + window_size, total_duration)
        
        try:
            # Read window data from reader
            data_array = reader.read_range(current_time, window_end)
            
            if data_array is not None and data_array.shape[1] > 0:
                # Convert to leads dictionary for analyze_ecg
                window_leads = {}
                for i, lead_name in enumerate(lead_names):
                    if i < data_array.shape[0]:
                        window_leads[lead_name] = data_array[i, :]
                
                if window_leads:
                    # Run arrhythmia detection
                    results = analyze_ecg(window_leads, fs=fs)
                    
                    # Get detected arrhythmias
                    arrhythmias = results.get('arrhythmias', [])
                    primary_rhythm = results.get('primary_rhythm', '')

                    # AF segment gate: require absent/insufficient P waves and
                    # an irregular ventricular rhythm.  This prevents a
                    # missing-P-wave result caused by noise from becoming an
                    # AF region by itself.
                    af_waveform_confirmed = False
                    beats = [b for b in (results.get('beats') or []) if isinstance(b, dict)]
                    rr_values = np.asarray(
                        [float(b.get('rr_ms')) for b in beats if b.get('rr_ms') is not None],
                        dtype=float,
                    )
                    rate_label = ''
                    if len(beats) >= 3 and rr_values.size >= 2:
                        p_present_ratio = float(np.mean([
                            bool(b.get('p_present')) for b in beats
                        ]))
                        qrs_values = np.asarray(
                            [float(b.get('qrs_ms')) for b in beats
                             if b.get('qrs_ms') is not None],
                            dtype=float,
                        )
                        pr_values = np.asarray(
                            [float(b.get('pr_ms')) for b in beats
                             if b.get('pr_ms') is not None],
                            dtype=float,
                        )
                        organized_qrs = bool(
                            qrs_values.size >= max(3, len(beats) // 2)
                            and np.all((qrs_values >= 40.0) & (qrs_values <= 220.0))
                        )
                        consistent_pr = bool(
                            pr_values.size >= max(3, len(beats) // 2)
                            and np.std(pr_values) <= 80.0
                        )
                        sinus_rate_evidence = bool(
                            p_present_ratio >= 0.60
                            and organized_qrs
                            and consistent_pr
                        )
                        # Use the requested strict RR rule.  Every valid RR
                        # interval in the window must satisfy the threshold;
                        # a mixed-rate window is left unclassified. Rate
                        # labels require sinus P/QRS evidence.
                        if sinus_rate_evidence and bool(np.all(rr_values > 1000.0)):
                            rate_label = 'Sinus Bradycardia'
                        elif sinus_rate_evidence and bool(np.all(rr_values < 600.0)):
                            rate_label = 'Sinus Tachycardia'
                    if len(beats) >= 3 and rr_values.size >= 2:
                        p_absent_ratio = float(np.mean([
                            not bool(b.get('p_present')) for b in beats
                        ]))
                        rr_std_ms = float(np.std(rr_values))
                        rr_range_ms = float(np.max(rr_values) - np.min(rr_values))
                        af_waveform_confirmed = bool(
                            p_absent_ratio > 0.50 and (
                                rr_std_ms > 80.0
                                or (p_absent_ratio > 0.70 and rr_range_ms > 120.0)
                            )
                        )
                    
                    if arrhythmias or primary_rhythm or rate_label:
                        # Map detected arrhythmias to segment labels
                        label_map = {
                            'Asystole': ('X', '#0000FF'),
                            'Ventricular Fibrillation': ('V', '#FF3333'),
                            'Ventricular Tachycardia': ('V', '#FF3333'),
                            'Atrial Fibrillation': ('AF', '#FF00FF'),
                            'Atrial Flutter': ('AF', '#FF00FF'),
                            'Sinus Bradycardia': ('S', '#00BFFF'),
                            'Sinus Tachycardia': ('S', '#00FFFF'),
                            'Normal Sinus Rhythm': ('N', '#00FF00'),
                            'Bradycardia (non-sinus)': ('S', '#00FFFF'),
                            'Tachycardia (non-sinus)': ('S', '#00FFFF'),
                            '1st-degree AV block': ('P', '#FF00FF'),
                            '2nd-degree AV block (Mobitz I / Wenckebach)': ('P', '#FF00FF'),
                            '3rd-degree AV block': ('P', '#FF00FF'),
                            'Right bundle branch block (RBBB)': ('P', '#FF00FF'),
                            'Left bundle branch block (LBBB)': ('P', '#FF00FF'),
                            'Premature ventricular contraction (PVC)': ('V', '#FF3333'),
                            'Premature atrial contraction (PAC)': ('S', '#00FFFF'),
                            'ST elevation': ('X', '#FFFF00'),
                            'ST depression': ('X', '#FFFF00'),
                        }
                        
                        # Prefer clinically important atrial rhythm findings
                        # from the arrhythmia list over a generic primary label
                        # such as Normal Sinus Rhythm.  Flutter/AF can be a
                        # secondary finding when the ventricular rate remains
                        # organized, but it must still create a waveform region.
                        atrial_label = next(
                            (
                                str(label) for label in arrhythmias
                                if 'atrial fibrillation' in str(label).lower()
                                or 'atrial flutter' in str(label).lower()
                            ),
                            '',
                        )
                        detected_label = atrial_label or rate_label or primary_rhythm or (
                            arrhythmias[0] if arrhythmias else ''
                        )

                        # Never trust a classifier's Brady/Tachy label when
                        # the waveform did not satisfy the strong sinus
                        # P-wave/QRS/PR checks above.
                        if (not rate_label and str(detected_label).lower()
                                in {'sinus bradycardia', 'sinus tachycardia',
                                    'bradycardia (non-sinus)', 'tachycardia (non-sinus)'}):
                            detected_label = 'Rhythm Undetermined'

                        # Use the beat-to-beat waveform measurement for rate
                        # segments instead of relying only on a classifier
                        # label.  A minimum beat count prevents one bad RR
                        # interval from creating a Brady/Tachy region.
                        protected_rate_labels = {
                            'Ventricular Fibrillation',
                            'Ventricular Tachycardia',
                            'Atrial Fibrillation',
                            'Atrial Flutter',
                        }
                        if (detected_label not in protected_rate_labels
                                and rate_label):
                            detected_label = rate_label

                        # Never create an AF segment from the label alone.
                        # The waveform-derived gate above must also pass.
                        if 'atrial fibrillation' in str(detected_label).lower() and not af_waveform_confirmed:
                            detected_label = 'Rhythm Undetermined'

                        # Map to segment label
                        label_code, color = label_map.get(detected_label, ('X', '#0000FF'))
                        
                        # Only add if not Normal Sinus Rhythm (to avoid clutter)
                        if detected_label != 'Normal Sinus Rhythm' and detected_label != 'Rhythm Undetermined':
                            # Compute real-world timestamp strings
                            start_time_str = ''
                            end_time_str = ''
                            try:
                                if hasattr(reader, 'start_time') and reader.start_time is not None:
                                    start_real = datetime.fromtimestamp(reader.start_time + current_time)
                                    end_real = datetime.fromtimestamp(reader.start_time + window_end)
                                    start_time_str = start_real.strftime('%H:%M:%S')
                                    end_time_str = end_real.strftime('%H:%M:%S')
                                else:
                                    h = int(current_time // 3600)
                                    m = int((current_time % 3600) // 60)
                                    s = int(current_time % 60)
                                    start_time_str = f"{h:02d}:{m:02d}:{s:02d}"
                                    
                                    h = int(window_end // 3600)
                                    m = int((window_end % 3600) // 60)
                                    s = int(window_end % 60)
                                    end_time_str = f"{h:02d}:{m:02d}:{s:02d}"
                            except Exception as e:
                                print(f"[Auto Arrhythmia Detect] Error formatting time: {e}")
                            
                            candidate_segments.append({
                                'start_sec': current_time,
                                'end_sec': window_end,
                                'label': detected_label,
                                'color': color,
                                'start_time_str': start_time_str,
                                'end_time_str': end_time_str,
                                'detected_rhythm': detected_label
                            })
                            
                            print(f"[Auto Arrhythmia Detect] Detected: {detected_label} at {current_time:.1f}s - {window_end:.1f}s")
        
        except Exception as e:
            print(f"[Auto Arrhythmia Detect] Error analyzing window {current_time:.1f}s: {e}")
        
        current_time += step_size
        window_count += 1
        
        # Update progress periodically
        if window_count % 10 == 0 and progress_callback:
            progress = int((current_time / total_duration) * 100)
            progress_callback(progress)
    
    # ------------------------------------------------------------------
    # Temporal confirmation
    # ------------------------------------------------------------------
    # Windows advance by five seconds and overlap by five seconds.  Therefore
    # two matching windows represent at least five seconds of persistence in
    # the same rhythm.  A single candidate is deliberately rejected for the
    # rhythms most vulnerable to noise (VF/VT/Flutter/AF).
    if not candidate_segments:
        return []

    candidate_segments.sort(key=lambda x: x['start_sec'])
    persistent_labels = {
        'Ventricular Fibrillation',
        'Ventricular Tachycardia',
        'Atrial Fibrillation',
        'Sinus Bradycardia',
        'Sinus Tachycardia',
    }
    confirmed = []
    run = []

    def flush_run(items):
        if not items:
            return
        label = items[0]['label']
        # Require a previous-window match for high-risk/atrial candidate
        # labels.  Other labels retain the old single-window behavior.
        if label in persistent_labels and len(items) < 2:
            return
        first = dict(items[0])
        first['end_sec'] = max(item['end_sec'] for item in items)
        first['end_time_str'] = items[-1].get('end_time_str', first.get('end_time_str', ''))
        confirmed.append(first)

    for seg in candidate_segments:
        if not run:
            run = [seg]
            continue
        previous = run[-1]
        same_label = seg['label'] == previous['label']
        overlapping = seg['start_sec'] <= previous['end_sec'] + 0.25
        if same_label and overlapping:
            run.append(seg)
        else:
            flush_run(run)
            run = [seg]
    flush_run(run)

    # A confirmed run is already merged, but keep this final pass so adjacent
    # confirmed runs of the same label become one visual region.
    merged_segments = []
    for seg in confirmed:
        if merged_segments:
            last = merged_segments[-1]
            if (seg['label'] == last['label'] and
                    seg['start_sec'] <= last['end_sec'] + 2.0):
                last['end_sec'] = max(last['end_sec'], seg['end_sec'])
                last['end_time_str'] = seg.get('end_time_str', last.get('end_time_str', ''))
                continue
        merged_segments.append(seg)

    return merged_segments


def convert_to_structured_events(detected_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert detected segments to structured_events format for waveform coloring.
    
    Args:
        detected_segments: List of detected arrhythmia segments from detect_arrhythmias()
    
    Returns:
        List of structured events with timestamp, type, label, end_timestamp, color
    """
    structured_events = []
    
    for seg in detected_segments:
        structured_events.append({
            'timestamp': seg['start_sec'],
            'type': seg['label'],
            'label': seg['label'],
            'end_timestamp': seg['end_sec'],
            'color': seg['color'],
            'source': 'Auto',
        })
    
    # Sort by timestamp
    structured_events.sort(key=lambda x: float(x.get('timestamp', 0.0) or 0.0))
    
    return structured_events
