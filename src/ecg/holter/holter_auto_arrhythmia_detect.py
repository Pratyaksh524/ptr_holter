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
    Auto-detect arrhythmias and rate-based rhythm segments (Sinus Tachycardia / Sinus Bradycardia)
    directly from ECG waveforms with beat-exact QRS peak boundary alignment.
    
    Args:
        reader: ECGHFileReader instance with read_range method
        progress_callback: Optional callback function(int) for progress updates (0-100)
    
    Returns:
        List of detected arrhythmia segments with:
            - start_sec: Start time in seconds (snapped exactly to start QRS peak)
            - end_sec: End time in seconds (snapped exactly to end QRS peak)
            - label: Full arrhythmia name
            - color: Color hex code for waveform coloring
            - start_time_str: Formatted HH:MM:SS string
            - end_time_str: Formatted HH:MM:SS string
            - detected_rhythm: Full arrhythmia name
    """
    from ecg.arrhythmia_detector import analyze_ecg
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:
        from ..pan_tompkins import pan_tompkins

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

    # ------------------------------------------------------------------
    # 1. Detect all R-peaks across the entire recording using best rhythm lead
    # ------------------------------------------------------------------
    lead_candidates = [l for l in ['II', 'V5', 'V6', 'V4', 'V1', 'I'] if l in lead_names]
    if not lead_candidates:
        lead_candidates = lead_names

    test_chunk = reader.read_range(0.0, min(30.0, total_duration))
    best_lead_idx = 0
    best_ptp = 0.0
    for cand in lead_candidates:
        idx = lead_names.index(cand)
        if test_chunk is not None and idx < test_chunk.shape[0]:
            ptp = float(np.ptp(test_chunk[idx]))
            if ptp > best_ptp:
                best_ptp = ptp
                best_lead_idx = idx

    chunk_len_sec = 60.0
    all_r_peaks = []
    t_cursor = 0.0

    while t_cursor < total_duration:
        t_end = min(t_cursor + chunk_len_sec, total_duration)
        chunk_data = reader.read_range(t_cursor, t_end)
        if chunk_data is not None and best_lead_idx < chunk_data.shape[0]:
            lead_sig = np.asarray(chunk_data[best_lead_idx], dtype=float)
            if len(lead_sig) > 50:
                p_indices = pan_tompkins(lead_sig, fs=fs)
                for p_idx in p_indices:
                    peak_sec = t_cursor + (float(p_idx) / float(fs))
                    if not all_r_peaks or (peak_sec - all_r_peaks[-1]) >= 0.20:
                        all_r_peaks.append(peak_sec)
        t_cursor += chunk_len_sec

    all_r_peaks = sorted(all_r_peaks)

    # Calculate beat-to-beat RR intervals
    all_rr_ms = []
    if len(all_r_peaks) >= 2:
        for i in range(len(all_r_peaks) - 1):
            rr = (all_r_peaks[i + 1] - all_r_peaks[i]) * 1000.0
            all_rr_ms.append(rr)

    detected_segments = []

    def make_time_strs(s_sec, e_sec):
        s_str = ''
        e_str = ''
        try:
            if hasattr(reader, 'start_time') and reader.start_time is not None:
                s_real = datetime.fromtimestamp(reader.start_time + s_sec)
                e_real = datetime.fromtimestamp(reader.start_time + e_sec)
                s_str = s_real.strftime('%H:%M:%S')
                e_str = e_real.strftime('%H:%M:%S')
            else:
                h = int(s_sec // 3600)
                m = int((s_sec % 3600) // 60)
                s = int(s_sec % 60)
                s_str = f"{h:02d}:{m:02d}:{s:02d}"
                h = int(e_sec // 3600)
                m = int((e_sec % 3600) // 60)
                s = int(e_sec % 60)
                e_str = f"{h:02d}:{m:02d}:{s:02d}"
        except Exception:
            pass
        return s_str, e_str

    # ------------------------------------------------------------------
    # 2. Extract Beat-Exact Rate Segments (Sinus Tachycardia & Sinus Bradycardia)
    # ------------------------------------------------------------------
    if len(all_rr_ms) >= 3:
        # A. Sinus Tachycardia: consecutive beats with RR <= 600ms (HR >= 100 bpm)
        in_tachy = False
        tachy_start_idx = None

        for i, rr in enumerate(all_rr_ms):
            is_fast = (rr <= 600.0) or (rr <= 630.0 and in_tachy and i + 1 < len(all_rr_ms) and all_rr_ms[i + 1] <= 600.0)
            if is_fast:
                if not in_tachy:
                    in_tachy = True
                    tachy_start_idx = i
            else:
                if in_tachy:
                    tachy_end_idx = i
                    beat_count = tachy_end_idx - tachy_start_idx + 1
                    dur = all_r_peaks[tachy_end_idx] - all_r_peaks[tachy_start_idx]
                    if beat_count >= 4 and dur >= 2.0:
                        s_sec = all_r_peaks[tachy_start_idx]
                        e_sec = all_r_peaks[tachy_end_idx]
                        s_str, e_str = make_time_strs(s_sec, e_sec)
                        detected_segments.append({
                            'start_sec': s_sec,
                            'end_sec': e_sec,
                            'label': 'Sinus Tachycardia',
                            'color': '#00FFFF',
                            'start_time_str': s_str,
                            'end_time_str': e_str,
                            'detected_rhythm': 'Sinus Tachycardia'
                        })
                    in_tachy = False
                    tachy_start_idx = None

        if in_tachy and tachy_start_idx is not None:
            tachy_end_idx = len(all_rr_ms)
            beat_count = tachy_end_idx - tachy_start_idx + 1
            dur = all_r_peaks[tachy_end_idx] - all_r_peaks[tachy_start_idx]
            if beat_count >= 4 and dur >= 2.0:
                s_sec = all_r_peaks[tachy_start_idx]
                e_sec = all_r_peaks[tachy_end_idx]
                s_str, e_str = make_time_strs(s_sec, e_sec)
                detected_segments.append({
                    'start_sec': s_sec,
                    'end_sec': e_sec,
                    'label': 'Sinus Tachycardia',
                    'color': '#00FFFF',
                    'start_time_str': s_str,
                    'end_time_str': e_str,
                    'detected_rhythm': 'Sinus Tachycardia'
                })

        # B. Sinus Bradycardia: consecutive beats with RR >= 1050ms (HR < 57.1 bpm)
        # Note: 60 bpm (1000ms RR) is Normal Sinus Rhythm (NSR: 60-100 bpm), NOT Bradycardia.
        # True Bradycardia occurs when HR < 60 bpm (RR > 1000ms, e.g. 50 bpm / 1200ms).
        in_brady = False
        brady_start_idx = None

        for i, rr in enumerate(all_rr_ms):
            is_slow = (rr >= 1050.0) or (rr >= 1020.0 and in_brady and i + 1 < len(all_rr_ms) and all_rr_ms[i + 1] >= 1050.0)
            if is_slow:
                if not in_brady:
                    in_brady = True
                    brady_start_idx = i
            else:
                if in_brady:
                    brady_end_idx = i
                    beat_count = brady_end_idx - brady_start_idx + 1
                    dur = all_r_peaks[brady_end_idx] - all_r_peaks[brady_start_idx]
                    if beat_count >= 3 and dur >= 2.5:
                        s_sec = all_r_peaks[brady_start_idx]
                        e_sec = all_r_peaks[brady_end_idx]
                        s_str, e_str = make_time_strs(s_sec, e_sec)
                        detected_segments.append({
                            'start_sec': s_sec,
                            'end_sec': e_sec,
                            'label': 'Sinus Bradycardia',
                            'color': '#00BFFF',
                            'start_time_str': s_str,
                            'end_time_str': e_str,
                            'detected_rhythm': 'Sinus Bradycardia'
                        })
                    in_brady = False
                    brady_start_idx = None

        if in_brady and brady_start_idx is not None:
            brady_end_idx = len(all_rr_ms)
            beat_count = brady_end_idx - brady_start_idx + 1
            dur = all_r_peaks[brady_end_idx] - all_r_peaks[brady_start_idx]
            if beat_count >= 3 and dur >= 2.5:
                s_sec = all_r_peaks[brady_start_idx]
                e_sec = all_r_peaks[brady_end_idx]
                s_str, e_str = make_time_strs(s_sec, e_sec)
                detected_segments.append({
                    'start_sec': s_sec,
                    'end_sec': e_sec,
                    'label': 'Sinus Bradycardia',
                    'color': '#00BFFF',
                    'start_time_str': s_str,
                    'end_time_str': e_str,
                    'detected_rhythm': 'Sinus Bradycardia'
                })

    # ------------------------------------------------------------------
    # 3. Detect Morphological Arrhythmias (VFib, VTach, AFib, Flutter, AV Block)
    # ------------------------------------------------------------------
    window_size = 10.0
    overlap = 5.0
    step_size = window_size - overlap
    current_time = 0.0
    window_count = 0

    morphological_candidates = []

    while current_time < total_duration:
        window_end = min(current_time + window_size, total_duration)
        try:
            data_array = reader.read_range(current_time, window_end)
            if data_array is not None and data_array.shape[1] > 0:
                window_leads = {}
                for i, lead_name in enumerate(lead_names):
                    if i < data_array.shape[0]:
                        window_leads[lead_name] = data_array[i, :]

                if window_leads:
                    results = analyze_ecg(window_leads, fs=fs)
                    arrhythmias = results.get('arrhythmias', [])

                    # Look for non-rate morphological arrhythmias
                    target_morph = None
                    for arr in arrhythmias:
                        arr_lower = str(arr).lower()
                        if 'ventricular fibrillation' in arr_lower or 'vfib' in arr_lower:
                            target_morph = ('Ventricular Fibrillation', '#FF3333')
                            break
                        elif 'ventricular tachycardia' in arr_lower or 'vtach' in arr_lower:
                            target_morph = ('Ventricular Tachycardia', '#FF3333')
                            break
                        elif 'atrial fibrillation' in arr_lower or 'afib' in arr_lower:
                            target_morph = ('Atrial Fibrillation', '#FF00FF')
                            break
                        elif 'atrial flutter' in arr_lower or 'aflutter' in arr_lower:
                            target_morph = ('Atrial Flutter', '#FF00FF')
                            break
                        elif 'av block' in arr_lower:
                            target_morph = (str(arr), '#FF00FF')
                            break

                    if target_morph:
                        s_str, e_str = make_time_strs(current_time, window_end)
                        morphological_candidates.append({
                            'start_sec': current_time,
                            'end_sec': window_end,
                            'label': target_morph[0],
                            'color': target_morph[1],
                            'start_time_str': s_str,
                            'end_time_str': e_str,
                            'detected_rhythm': target_morph[0]
                        })
        except Exception as e:
            print(f"[Auto Arrhythmia Detect] Window error {current_time:.1f}s: {e}")

        current_time += step_size
        window_count += 1
        if window_count % 10 == 0 and progress_callback:
            progress = int((current_time / total_duration) * 100)
            progress_callback(progress)

    # Merge contiguous morphological candidate windows
    if morphological_candidates:
        morphological_candidates.sort(key=lambda x: x['start_sec'])
        merged_morph = []
        for m in morphological_candidates:
            if merged_morph:
                last = merged_morph[-1]
                if last['label'] == m['label'] and m['start_sec'] <= last['end_sec'] + 0.25:
                    last['end_sec'] = max(last['end_sec'], m['end_sec'])
                    last['end_time_str'] = m['end_time_str']
                    continue
            merged_morph.append(m)
        detected_segments.extend(merged_morph)

    # Sort all detected segments by start time
    detected_segments.sort(key=lambda x: x['start_sec'])

    # Merge adjacent segments of same label
    final_segments = []
    for seg in detected_segments:
        if final_segments:
            last = final_segments[-1]
            if last['label'] == seg['label'] and seg['start_sec'] <= last['end_sec'] + 0.50:
                last['end_sec'] = max(last['end_sec'], seg['end_sec'])
                last['end_time_str'] = seg.get('end_time_str', last.get('end_time_str', ''))
                continue
        final_segments.append(seg)

    print(f"[Auto Arrhythmia Detect] Found {len(final_segments)} exact rhythm segments:")
    for s in final_segments:
        print(f"  - {s['label']}: {s['start_sec']:.2f}s to {s['end_sec']:.2f}s ({s['start_time_str']} - {s['end_time_str']})")

    return final_segments


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
