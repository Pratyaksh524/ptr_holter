"""
ecg/holter/auto_segment_arrthymia_detection.py
================================================
Auto-detected arrhythmia segment pipeline for the Holter Full Disclosure
view: waveform-based detection, and merging that result into a replay
engine's structured event list.

This module owns the full auto-segment-marking pipeline so that
holter_full_disclosure.py only has to call apply_auto_segments_to_engine()
and otherwise deal with rendering (SegmentOverlay, label-to-color mapping).

Pipeline:
    1. detect_arrhythmias()            -- waveform-based rate detection
                                           (Sinus Bradycardia/Tachycardia via
                                           RR intervals) and morphological
                                           detection (Ventricular
                                           Fibrillation via a direct P-wave/
                                           QRS-organization rule; Atrial
                                           Fibrillation/Flutter/AV Block via
                                           ecg.arrhythmia_detector.analyze_ecg).
    2. convert_to_structured_events()  -- segment dicts -> structured events.
    3. apply_auto_segments_to_engine() -- merges that output into
                                           engine._structured_events, which
                                           may already hold events loaded
                                           from the recording's own stored
                                           replay data. Handles dedup,
                                           rate-vs-morphology suppression,
                                           and same-label/cross-source
                                           merging.
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
    from ecg.arrhythmia_detector import analyze_ecg, measure_beat
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:
        from ..pan_tompkins import pan_tompkins

    # One-time numerical warm-up. analyze_ecg()'s spectral checks (atrial
    # flutter score, VF spectral score) go through scipy/numpy FFT code
    # paths whose very first invocation in a process was confirmed, by
    # direct testing, to occasionally produce a different result than every
    # subsequent call with the exact same input samples -- almost certainly
    # a one-time algorithm/plan-selection step in the FFT backend rather
    # than anything data-dependent. Running analyze_ecg() once here, on
    # throwaway synthetic data before any real window is analyzed, absorbs
    # that one-time cost up front so every real classification below always
    # runs on the "warm", stable code path and is reproducible regardless of
    # whether this is the first arrhythmia analysis done in the process.
    try:
        _warmup_fs = 500
        _warmup_t = np.linspace(0, 2.0, int(_warmup_fs * 2.0), endpoint=False)
        _warmup_sig = 0.5 * np.sin(2 * np.pi * 1.2 * _warmup_t)
        analyze_ecg({"II": _warmup_sig}, fs=_warmup_fs)
    except Exception:
        pass

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

    # Pan-Tompkins is run on each 60s chunk independently for memory/
    # performance reasons, but QRS detectors have edge artifacts (filter
    # settling) right at the start/end of whatever signal they're handed --
    # a beat sitting close to a chunk boundary can be missed entirely. Since
    # chunk_len_sec is just an I/O chunking choice, not a true rhythm
    # boundary, each chunk is read with a few seconds of context on both
    # sides; Pan-Tompkins runs on the padded signal, but only peaks that
    # fall within the chunk's own [t_cursor, t_end) core are kept, so
    # padding from adjacent chunks never causes double-counting -- it only
    # gives the detector enough settling room to not miss the boundary beat.
    chunk_len_sec = 60.0
    chunk_pad_sec = 3.0
    all_r_peaks = []
    t_cursor = 0.0

    while t_cursor < total_duration:
        t_end = min(t_cursor + chunk_len_sec, total_duration)
        pad_start = max(0.0, t_cursor - chunk_pad_sec)
        pad_end = min(total_duration, t_end + chunk_pad_sec)
        chunk_data = reader.read_range(pad_start, pad_end)
        if chunk_data is not None and best_lead_idx < chunk_data.shape[0]:
            lead_sig = np.asarray(chunk_data[best_lead_idx], dtype=float)
            if len(lead_sig) > 50:
                p_indices = pan_tompkins(lead_sig, fs=fs)
                for p_idx in p_indices:
                    peak_sec = pad_start + (float(p_idx) / float(fs))
                    if peak_sec < t_cursor or peak_sec >= t_end:
                        continue
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

    def suppress_rate_overlap(rate_segments, morph_segments):
        """
        Morphological rhythms (VFib, VTach, AFib, Flutter, AV Block) are detected
        from actual QRS/P-wave morphology and take clinical priority over the
        pure RR-interval-based Sinus Tachycardia/Bradycardia segments above.
        Pan-Tompkins can mis-fire "R-peaks" on a chaotic VFib/VTach baseline at
        spacing that happens to look fast or slow, which previously caused
        Sinus Tachycardia/Bradycardia labels to be drawn on top of a confirmed
        morphological arrhythmia. Trim (or drop) any rate-based segment so it
        never overlaps a morphological segment's time range.

        A fixed buffer is added around each morphological interval before
        checking overlap (not to the interval's own displayed boundaries,
        only to this comparison). Right at the transition into/out of a
        chaotic episode, Pan-Tompkins can briefly catch one or two
        wide-spaced peaks before/after the window that actually crossed the
        VFib threshold, producing a short "Sinus Bradycardia"/"Sinus
        Tachycardia" blip that sits immediately next to -- but doesn't
        strictly overlap -- the morphological region. That blip is a
        transition artifact, not a real rhythm, so the exclusion zone
        extends a bit past the morphological region's own edges to catch it.
        """
        if not morph_segments:
            return rate_segments
        buffer_sec = 5.0
        morph_intervals = sorted(
            (max(0.0, m['start_sec'] - buffer_sec), m['end_sec'] + buffer_sec)
            for m in morph_segments
        )
        result = []
        for seg in rate_segments:
            pieces = [(seg['start_sec'], seg['end_sec'])]
            for m_start, m_end in morph_intervals:
                next_pieces = []
                for p_start, p_end in pieces:
                    if m_end <= p_start or m_start >= p_end:
                        next_pieces.append((p_start, p_end))
                        continue
                    if m_start > p_start:
                        next_pieces.append((p_start, m_start))
                    if m_end < p_end:
                        next_pieces.append((m_end, p_end))
                pieces = next_pieces
            for p_start, p_end in pieces:
                if p_end - p_start < 1.0:
                    continue
                s_str, e_str = make_time_strs(p_start, p_end)
                trimmed = dict(seg)
                trimmed['start_sec'] = p_start
                trimmed['end_sec'] = p_end
                trimmed['start_time_str'] = s_str
                trimmed['end_time_str'] = e_str
                result.append(trimmed)
        return result

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
    # 3. Detect Morphological Arrhythmias (VFib, AFib, Flutter, AV Block)
    # ------------------------------------------------------------------
    # Ventricular Fibrillation rule (simple, direct -- applied to every
    # rhythm window the same way): a window is VFib when its beats show
    # BOTH hallmark absences --
    #   1. no P wave reliably precedes a QRS (p_onset before q_onset), and
    #   2. the QRS complex itself is not organized/resolvable (irregular,
    #      chaotic -- _qrs_bounds() in arrhythmia_detector.py could not find
    #      a genuine isoelectric boundary for most beats).
    # Ventricular Tachycardia is intentionally not reported as a separate
    # label here -- a window this chaotic renders as Ventricular Fibrillation.
    def _is_vfib_window(window_signal, fs):
        r_peaks = pan_tompkins(window_signal, fs=fs)
        if len(r_peaks) < 2:
            return False
        beats = [b for b in (measure_beat(window_signal, int(p), fs) for p in r_peaks) if b is not None]
        if not beats:
            return False

        # A P wave only counts as reliably preceding QRS when that same
        # beat's QRS boundary was genuinely resolved. The P-search window is
        # positioned before q_onset by construction, so on a chaotic/noisy
        # beat (QRS boundary unresolved) almost any noise bump trivially
        # satisfies p_onset < q_onset -- that check only means something once
        # the QRS itself is confirmed real.
        p_follows_qrs = sum(
            1 for b in beats
            if b.get('p_present') and b.get('qrs_bounds_resolved')
            and b.get('p_onset') is not None and b.get('q_onset') is not None
            and b['p_onset'] < b['q_onset']
        )
        organized_qrs = sum(1 for b in beats if b.get('qrs_bounds_resolved'))

        # The P-wave search window is positioned before q_onset by
        # construction, so any detected bump -- real or a noise peak on a
        # chaotic baseline -- almost always trivially satisfies
        # p_onset < q_onset. Requiring literally zero such beats never fires
        # on real data; a majority-based threshold (most beats must fail to
        # show a P-wave reliably preceding QRS) is the meaningful check.
        no_p_follows_qrs = p_follows_qrs < max(1, len(beats) // 2)
        no_organized_qrs = organized_qrs < max(1, len(beats) // 2)
        return no_p_follows_qrs and no_organized_qrs

    # Tried shrinking this to 6s/3s to reduce detection latency; verified
    # directly against real recordings that it did NOT detect episodes any
    # earlier (the rhythm's own transition is gradual, not instant -- a
    # smaller window doesn't see the onset sooner), but it DID shrink each
    # window to as few as 3 beats, which is too small a sample for the
    # majority vote below to be stable -- a single PVC or one noisy beat
    # could flip an otherwise-clear VFib window to "not VFib", fragmenting
    # one continuous episode into pieces with gaps, and conversely could
    # flip a single isolated window to "VFib" around an ectopic beat inside
    # an otherwise organized rhythm. Reverted to 10s/5s, which has enough
    # beats per window to make that vote statistically meaningful.
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
                target_morph = None

                if best_lead_idx < data_array.shape[0]:
                    window_signal = np.asarray(data_array[best_lead_idx], dtype=float)
                    if len(window_signal) > 50 and _is_vfib_window(window_signal, fs):
                        target_morph = ('Ventricular Fibrillation', '#FF3333')

                if target_morph is None:
                    window_leads = {}
                    for i, lead_name in enumerate(lead_names):
                        if i < data_array.shape[0]:
                            window_leads[lead_name] = data_array[i, :]

                    if window_leads:
                        results = analyze_ecg(window_leads, fs=fs)
                        arrhythmias = results.get('arrhythmias', [])

                        # Only the non-VFib morphological arrhythmias are
                        # looked up here; VFib is decided by the direct rule
                        # above, applied the same way to every window.
                        for arr in arrhythmias:
                            arr_lower = str(arr).lower()
                            if 'atrial fibrillation' in arr_lower or 'afib' in arr_lower:
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

    # Merge contiguous morphological candidate windows. The gap tolerance is
    # one full window_size, not a tiny fraction of a second: each window's
    # pass/fail is a per-window majority vote over a handful of beats, so a
    # single borderline window right in the middle of a sustained episode can
    # narrowly miss the threshold by chance (one or two beats' worth of
    # noise) even though the episode never actually stopped. A short gap
    # between two windows that both independently fired as the same
    # morphological label is almost always that kind of sampling noise, not
    # a real few-second recovery in the middle of e.g. Ventricular
    # Fibrillation -- so it gets bridged into one continuous segment.
    if morphological_candidates:
        morphological_candidates.sort(key=lambda x: x['start_sec'])
        merged_morph = []
        for m in morphological_candidates:
            if merged_morph:
                last = merged_morph[-1]
                if last['label'] == m['label'] and m['start_sec'] <= last['end_sec'] + window_size:
                    last['end_sec'] = max(last['end_sec'], m['end_sec'])
                    last['end_time_str'] = m['end_time_str']
                    continue
            merged_morph.append(m)
        # Morphological arrhythmias take priority: trim/drop any Sinus
        # Tachycardia/Bradycardia segments that overlap them before merging.
        detected_segments = suppress_rate_overlap(detected_segments, merged_morph)
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


def _suppress_rate_overlap_in_structured_events(events):
    """
    Drop/trim rate-based Sinus Tachycardia/Bradycardia auto events so
    they never overlap a morphological auto event (VFib, VTach, AFib,
    Flutter, AV Block). `events` can carry auto events from two
    independent sources -- the real-time recording pipeline (stored
    replay metrics) and this module's own waveform-based
    detect_arrhythmias() -- and either source's rate-based labels can
    predate a detector fix or simply miss a morphological arrhythmia
    the other source caught. Without this pass, a stale/independent
    Sinus Tachycardia region keeps rendering on top of a correctly
    detected Ventricular Fibrillation region.

    Only a morphological finding from THIS module's own fresh waveform
    analysis (source 'auto'/'arrhythmia') is trusted to trim a rate
    segment. The stored real-time pipeline's own morphological labels
    (source 'analysis'/'classified') are not used as an exclusion trigger
    here: that pipeline runs its own smoothing/hysteresis and has been
    observed to keep reporting a finding (e.g. AV Block) for several
    seconds after a fresh re-analysis of that same window no longer finds
    it -- trusting it would wrongly eat into the front of a rate segment
    (e.g. a genuinely tachycardic run) that starts right after.
    """
    RATE_LABELS = {'sinus tachycardia', 'sinus bradycardia'}
    TRUSTED_MORPH_SOURCES = {'auto', 'arrhythmia'}

    def _is_morph(ev):
        lbl = str(ev.get('label', ev.get('type', ''))).lower()
        if str(ev.get('source', '')).lower() not in TRUSTED_MORPH_SOURCES:
            return False
        return 'av block' in lbl or lbl in {
            'ventricular fibrillation', 'ventricular tachycardia',
            'atrial fibrillation', 'atrial flutter',
        }

    morph_intervals = sorted(
        (float(ev.get('timestamp', 0.0) or 0.0),
         float(ev.get('end_timestamp', ev.get('timestamp', 0.0)) or 0.0))
        for ev in events if _is_morph(ev)
    )
    if not morph_intervals:
        return events

    result = []
    for ev in events:
        label = str(ev.get('label', ev.get('type', ''))).lower()
        if label not in RATE_LABELS:
            result.append(ev)
            continue
        start = float(ev.get('timestamp', 0.0) or 0.0)
        end = float(ev.get('end_timestamp', start) or start)
        pieces = [(start, end)]
        for m_start, m_end in morph_intervals:
            next_pieces = []
            for p_start, p_end in pieces:
                if m_end <= p_start or m_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                if m_start > p_start:
                    next_pieces.append((p_start, m_start))
                if m_end < p_end:
                    next_pieces.append((m_end, p_end))
            pieces = next_pieces
        for p_start, p_end in pieces:
            if p_end - p_start < 1.0:
                continue
            trimmed = dict(ev)
            trimmed['timestamp'] = p_start
            trimmed['end_timestamp'] = p_end
            result.append(trimmed)
    return result


def apply_auto_segments_to_engine(
    engine,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Optional[int]:
    """
    Run detect_arrhythmias() against engine._reader and merge the result
    into engine._structured_events in place.

    engine._structured_events may already hold events loaded from the
    recording's own stored replay data (e.g. via HolterReplayEngine's
    _load_layered_store()/_load_metrics()). This merges the two sources:
    dedup, rate-vs-morphology suppression (see
    _suppress_rate_overlap_in_structured_events), and same-label merging
    across both sources, preserving this detector's own 'Auto'/
    'arrhythmia' source tag on the merged region so
    HolterFullDisclosureDialog._update_time_and_arrhythmia_labels() (which
    only looks at events with that source) can find it.

    Returns the number of newly-added regions, or None if the engine has
    no reader (nothing was done).
    """
    reader = getattr(engine, '_reader', None)
    if reader is None:
        return None

    auto_segments = detect_arrhythmias(reader, progress_callback=progress_callback)
    auto_events = convert_to_structured_events(auto_segments)
    existing = getattr(engine, '_structured_events', []) or []
    existing_keys = {
        (round(float(ev.get('timestamp', 0.0) or 0.0), 3),
         str(ev.get('label', '')).lower())
        for ev in existing
    }
    added = 0
    for event in auto_events:
        key = (
            round(float(event.get('timestamp', 0.0) or 0.0), 3),
            str(event.get('label', '')).lower(),
        )
        # A region-type event (has end_timestamp) is always added even if
        # its (timestamp, label) exactly matches an existing stored one --
        # the merge/suppression pass below safely reconciles overlapping
        # same-label regions by extending coverage, not duplicating it.
        # Skipping it here on an exact-key "duplicate" silently discarded
        # this detector's properly merged, multi-window region (e.g. a
        # continuous 60s Ventricular Fibrillation episode) whenever the
        # stored real-time pipeline happened to log its own, far more
        # fragmented per-chunk snapshot at that same starting instant --
        # leaving only the fragmented version on screen. Point-in-time
        # events (no end_timestamp, e.g. manual markers) still dedup as
        # before since they don't go through that merge.
        if event.get('end_timestamp') is not None or key not in existing_keys:
            existing.append(event)
            existing_keys.add(key)
            added += 1

    # Collapse duplicate/overlapping automatic regions from the stored
    # replay metrics and the current waveform detector. Manual events do
    # not have an end_timestamp and are never included in this merge.
    # Non-manual auto-origin events come through with several different
    # source labels depending on which pipeline produced them ('Auto',
    # 'arrhythmia', 'classified', 'analysis', ...) -- an allow-list here
    # missed 'analysis'-sourced regions (the layered event store's default
    # when no source was recorded), letting a stale Sinus Tachycardia
    # region from that path bypass suppression and keep rendering on top
    # of a correctly detected VFib region. Exclude only clearly
    # manual-origin events instead.
    auto_events_all = [
        ev for ev in existing
        if 'manual' not in str(ev.get('source', '')).lower()
        and ev.get('end_timestamp') is not None
    ]
    non_auto_events = [
        ev for ev in existing if ev not in auto_events_all
    ]
    auto_events_all.sort(
        key=lambda ev: float(ev.get('timestamp', 0.0) or 0.0)
    )
    auto_events_all = _suppress_rate_overlap_in_structured_events(auto_events_all)

    # Merge each label's own timeline independently, not one single
    # interleaved pass. The stored real-time pipeline logs several labels
    # at the exact same timestamp (e.g. at t=110: "Ventricular
    # Fibrillation", "Wide QRS (non-specific)", "Long QT Syndrome", "PVC
    # Morphology" all together) -- sorted by timestamp alone, a
    # differently-labeled entry can land between two same-labeled ones and
    # break the "is this adjacent to the last entry" check, fragmenting
    # what should be one continuous region into several, even though
    # nothing about the underlying rhythm actually changed.
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for ev in auto_events_all:
        label = str(ev.get('label', ev.get('type', ''))).lower()
        by_label.setdefault(label, []).append(ev)

    merged_auto = []
    for label, group in by_label.items():
        group.sort(key=lambda ev: float(ev.get('timestamp', 0.0) or 0.0))
        label_merged = []
        for ev in group:
            start = float(ev.get('timestamp', 0.0) or 0.0)
            end = float(ev.get('end_timestamp', start) or start)
            ev_source = str(ev.get('source', '')).lower()
            if label_merged:
                last = label_merged[-1]
                last_end = float(last.get('end_timestamp', 0.0) or 0.0)
                if start <= last_end + 0.25:
                    last['end_timestamp'] = max(last_end, end)
                    # Prefer this waveform detector's own 'auto'/
                    # 'arrhythmia' source over a stored replay source
                    # ('analysis', 'classified', ...) once merged into the
                    # same region. The bottom-right rhythm indicator only
                    # looks at events with source in {'auto','arrhythmia'}
                    # (_update_time_and_arrhythmia_labels); if the merge
                    # kept whichever source happened to sort first, a
                    # region this detector correctly identified (e.g. a
                    # 60s Ventricular Fibrillation episode) could end up
                    # permanently invisible to that indicator even though
                    # the segment overlay renders it fine.
                    if ev_source in {'auto', 'arrhythmia'}:
                        last['source'] = ev.get('source')
                    continue
            label_merged.append(ev)
        merged_auto.extend(label_merged)

    existing = non_auto_events + merged_auto
    engine._structured_events = sorted(
        existing,
        key=lambda ev: float(ev.get('timestamp', 0.0) or 0.0),
    )
    return added
