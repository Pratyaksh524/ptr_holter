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
from scipy.signal import find_peaks


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
    candidate_lead_ptp = []
    for cand in lead_candidates:
        idx = lead_names.index(cand)
        if test_chunk is not None and idx < test_chunk.shape[0]:
            ptp = float(np.ptp(test_chunk[idx]))
            candidate_lead_ptp.append((idx, ptp))
            if ptp > best_ptp:
                best_ptp = ptp
                best_lead_idx = idx

    # The top few candidate leads by this same early-amplitude ranking,
    # used later by the morphological window loop to detect R-peaks across
    # more than just best_lead_idx alone -- that single lead is picked once
    # from the first 30s and stays fixed for the whole recording, so a
    # stretch later on where it happens to have a transient amplitude dip
    # can make Pan-Tompkins miss real beats that other leads see just
    # fine. Confirmed directly: a VFib-labeled stretch in one recording had
    # 3 beats (~457s, ~460s, ~462s) completely missing from best_lead_idx
    # while present at normal amplitude on 7 of the other 11 leads.
    peak_detection_lead_indices = [
        idx for idx, _ in sorted(candidate_lead_ptp, key=lambda item: item[1], reverse=True)[:3]
    ]
    if best_lead_idx not in peak_detection_lead_indices:
        peak_detection_lead_indices = [best_lead_idx] + peak_detection_lead_indices

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
    #
    # r_peaks is supplied by the caller rather than detected here. The
    # caller reads a couple of seconds of padding on each side of this
    # window before detecting peaks, purely so Pan-Tompkins has settling
    # room and doesn't miss or mismeasure a real beat sitting right at this
    # window's own start/end (the same reason the whole-recording R-peak
    # pass earlier in this function pads its own chunk reads) -- only
    # peaks whose own timestamp falls inside this window's true
    # [start, end) are ever passed in, so this is strictly a detection-
    # quality fix, never a way to borrow beats from a neighboring window.
    def _is_vfib_window(window_signal, r_peaks, fs):
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

    # Atrial Fibrillation / Atrial Flutter rule (simple, direct -- same style
    # as the VFib rule above, and completely independent of the Sinus/Brady/
    # Tachy P-wave logic, which stays untouched). Both require, as a shared
    # pre-condition, that the QRS itself IS organized (ventricular activation
    # is normal -- this is what separates it from Ventricular Fibrillation).
    #
    # Within that, the decision is read directly off the baseline BETWEEN
    # beats -- the segment a clinician actually looks at to call this --
    # instead of any frequency-domain score (an FFT-based sawtooth/flutter-
    # band score was tried first and rejected: on real recordings its
    # "spectral energy" in the flutter band turned out to just be harmonics
    # of the QRS complex itself, and it fired on plain Sinus Bradycardia/
    # Tachycardia test recordings whenever a harmonic of the ordinary heart
    # rate happened to land in that band):
    #   - take the stretch of each beat's own RR interval that lies between
    #     the end of the previous beat's QRS/T and that beat's own P wave (or
    #     a fallback point just before its QRS, if no P was found) -- this is
    #     exactly the normal "quiet" TP segment in sinus rhythm;
    #   - a beat whose TP segment is genuinely quiet (flat, no wave sitting
    #     in it) is a normal beat with an isolated P wave -- not this rule's
    #     territory;
    #   - once MOST beats in the window instead show continuous low-level
    #     waves filling that segment -- no quiet stretch at all, something
    #     (F/f waves) occupying it throughout -- the window is AFib/Flutter
    #     territory, and counting those waves gives the atrial rate:
    #       - a steady atrial rate forming a fixed 2:1/3:1/4:1/5:1 ratio to
    #         the QRS rate (a regular sawtooth) is Atrial Flutter;
    #       - anything else that still fills the TP segment throughout is
    #         Atrial Fibrillation (irregular fibrillatory f-waves).
    #
    # r_peaks is supplied by the caller (padded detection, filtered to this
    # window's own true [start, end)) for the same detection-quality reason
    # _is_vfib_window takes it -- see that function's own comment.
    def _classify_atrial_window(window_signal, r_peaks, fs):
        # A window needs a real sample of beats before its own majority
        # vote below means anything. Confirmed directly on a real
        # recording, right at a genuine VFib-to-AFib transition: a window
        # with only 5 beats in it (not from missed detection -- the
        # recording's own beat rate was genuinely still sparse there,
        # confirmed by re-reading with extra padding and finding the exact
        # same 5) had 3 resolved, just barely clearing the 2/3-majority
        # bar below and calling the window AFib several seconds before the
        # rhythm had actually settled -- every window from there onward
        # with 6 or more beats was unambiguously organized. At n=5 the
        # 2/3 rule only demands 3 right answers, which a genuinely
        # still-chaotic stretch can hand it by chance; requiring a few
        # more beats first is a direct, simple guard against exactly that.
        if len(r_peaks) < 6:
            return None
        beats = [measure_beat(window_signal, int(p), fs) for p in r_peaks]
        if any(b is None for b in beats):
            return None

        organized_qrs = sum(1 for b in beats if b.get('qrs_bounds_resolved'))
        if organized_qrs < max(2, (len(beats) * 2) // 3):
            return None  # QRS not reliably resolvable -- not this rule's territory (e.g. VFib)

        rr_ms = np.diff(np.array(r_peaks, dtype=float)) * 1000.0 / float(fs)
        rr_mean = float(np.mean(rr_ms)) if rr_ms.size else 0.0
        ventricular_rate = 60000.0 / rr_mean if rr_mean > 0 else 0.0
        if ventricular_rate <= 0:
            return None

        qrs_amp_ref = float(np.median([b.get('qrs_amplitude') or 0.0 for b in beats])) or 0.1
        noise_threshold = max(0.05, 0.15 * qrs_amp_ref)

        quiet_count = 0
        scored = 0
        total_waves = 0
        total_active_sec = 0.0

        for i in range(1, len(beats)):
            beat = beats[i]
            r_prev = r_peaks[i - 1]
            rr_local = r_peaks[i] - r_prev
            q_onset = int(beat.get('q_onset') or r_peaks[i])
            p_onset = beat.get('p_onset')

            # Skip past the previous beat's own QRS+T before looking for a
            # quiet TP stretch -- fixed 280ms, or 38% of this RR interval
            # when that's longer (T duration scales with RR, not a constant).
            #
            # Tried loosening this (150ms/30%) to recover coverage on a
            # real accelerating Atrial Flutter episode (RR falling from
            # 600ms to 400ms) where these stricter numbers left too little
            # room to score almost any beat once RR dropped below ~700ms,
            # leaving a genuine organized-QRS Flutter stretch completely
            # unclassified. That looser version worked for that case, but
            # directly broke a recording that must stay false-positive-free:
            # a plain Sinus Tachycardia run started reading as Atrial
            # Fibrillation/Flutter, because at a fast-but-genuinely-sinus
            # rate the T and P waves sit just as close together as they do
            # in real Flutter -- there is no margin value that reliably
            # tells the two apart from TP-segment room alone. Reverted:
            # missing a real episode in that one narrow accelerating-rate
            # case is a smaller problem than false-labeling a clean
            # recording the detector is specifically required not to touch.
            tp_start = r_prev + max(int(0.28 * fs), int(0.38 * rr_local))
            tp_end = int(p_onset) if (beat.get('p_present') and p_onset is not None) else max(tp_start, q_onset - int(0.15 * fs))
            tp_end = min(tp_end, q_onset)
            if tp_end - tp_start < int(0.06 * fs):
                continue  # no room left in this RR interval to judge

            scored += 1
            segment = window_signal[tp_start:tp_end]
            if float(np.ptp(segment)) < noise_threshold:
                quiet_count += 1
                continue

            waves, _ = find_peaks(segment, distance=max(1, int(0.09 * fs)), prominence=max(0.02, 0.2 * max(float(np.ptp(segment)), 0.05)))
            total_waves += len(waves)
            total_active_sec += (tp_end - tp_start) / float(fs)

        # At a fast rate the fixed 280ms blanked straight after each R can
        # leave no room at all before the next beat's own P-wave onset --
        # T and P legitimately crowd together with little or no isoelectric
        # gap between them at high heart rates, which is normal, not
        # pathological. Most beats then get skipped above ("no room"),
        # and scored can collapse to just one or two beats that did happen
        # to have room -- far too small a sample to trust a verdict from.
        # Confirmed directly: a window of 15 clean 120bpm sinus beats had
        # only ONE scored (the rest all skipped for exactly this reason),
        # and that one happened to be a stray artifact right at the
        # recording's own abrupt end, whose few found "waves" by chance
        # landed close enough to a flutter ratio to call the window
        # Atrial Flutter.
        if scored < 4:
            return None

        if (quiet_count / scored) < 0.4 and total_active_sec > 0:
            atrial_rate = (total_waves / total_active_sec) * 60.0
            ratio = atrial_rate / ventricular_rate
            if any(abs(ratio - whole) < 0.5 for whole in (2.0, 3.0, 4.0, 5.0)):
                return 'Atrial Flutter'
            return 'Atrial Fibrillation'

        # Most beats read as having a "quiet" TP segment -- but that quiet
        # reading can itself be fooled: on a genuinely fibrillatory
        # baseline, the generic P-wave finder often still latches onto one
        # of the small irregular f-wave bumps sitting right before each
        # QRS and calls it a P wave, which shrinks the measured TP window
        # down to nothing and hides the real irregularity from the check
        # above (confirmed directly: a recording with f-waves measuring
        # ~230-270 units of amplitude, just under the noise floor used
        # above for that recording's scale, was missed this way). The
        # direct check here is the other hallmark of Atrial Fibrillation --
        # an irregularly irregular RR -- but a plain variability number
        # (e.g. coefficient of variation) does not actually distinguish
        # that from a normal, single, organized rhythm whose rate is
        # simply changing (a Sinus Bradycardia window sliding into a Sinus
        # Tachycardia one looks just as "variable" over one window; tried
        # that first and it fired on real Brady/Tachy transition windows).
        # What actually separates them is the SHAPE of the variability: a
        # genuine rate change is a handful of beats settling from one
        # fairly steady RR to another (the big jumps all point the same
        # way), while true AFib has no settled rate at all -- consecutive
        # RR intervals keep reversing direction, beat after beat. So: take
        # only the RR changes big enough to be real (ignore sub-20ms
        # jitter), and require that most of them flip direction from the
        # last one, not just that a few big changes exist.
        rr_diffs = np.diff(rr_ms)
        significant_diffs = rr_diffs[np.abs(rr_diffs) > 60.0]
        if significant_diffs.size >= 3:
            diff_signs = np.sign(significant_diffs)
            reversals = np.sum(diff_signs[1:] != diff_signs[:-1])
            reversal_ratio = reversals / (diff_signs.size - 1)
            if reversal_ratio >= 0.5:
                return 'Atrial Fibrillation'

        return None  # most beats have a genuinely quiet/consistent baseline -- normal

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
    #
    # Also tried shrinking only the STEP (to 1s, keeping the 10s window) to
    # pin down a transition boundary more precisely without touching that
    # per-window vote's own stability. It barely moved any boundary (the
    # vote itself is what decides the boundary, not how often it's taken),
    # and it has its own real cost: 5x more windows means 5x more windows
    # falling through to the analyze_ecg() AV-Block fallback below whenever
    # neither direct rule above fires -- confirmed directly that this
    # surfaced false "Third-degree AV Block" calls on a plain Sinus
    # Tachycardia recording that had never shown them at the 5s step. Not
    # worth it for no real precision gain; reverted to 5s.
    window_size = 10.0
    overlap = 5.0
    step_size = window_size - overlap
    current_time = 0.0
    window_count = 0

    morphological_candidates = []

    # How far past each window's own edges to read before detecting R-peaks
    # for it, purely so Pan-Tompkins has settling room and doesn't miss or
    # mismeasure a real beat sitting right at the window boundary (see
    # _is_vfib_window's own comment for the real transition this was
    # confirmed to fix). Only peaks whose own timestamp lands inside the
    # window's true [current_time, window_end) are ever handed to the
    # window's classifiers below -- the padding is purely to detect those
    # edge beats correctly, never to borrow beats from the next window over.
    morph_pad_sec = 2.0

    while current_time < total_duration:
        window_end = min(current_time + window_size, total_duration)
        try:
            pad_start = max(0.0, current_time - morph_pad_sec)
            pad_end = min(total_duration, window_end + morph_pad_sec)
            padded_array = reader.read_range(pad_start, pad_end)
            if padded_array is not None and padded_array.shape[1] > 0:
                # The unpadded core slice, for the AV-Block/analyze_ecg()
                # fallback below, which wants exactly this window's own
                # samples -- sliced out of the padded read instead of a
                # second reader call.
                core_start_idx = int(round((current_time - pad_start) * fs))
                core_end_idx = core_start_idx + int(round((window_end - current_time) * fs))
                data_array = padded_array[:, core_start_idx:core_end_idx]
                target_morph = None

                if best_lead_idx < padded_array.shape[0]:
                    padded_signal = np.asarray(padded_array[best_lead_idx], dtype=float)
                    if len(padded_signal) > 50:
                        # Union R-peaks detected across the top few
                        # candidate leads (not just best_lead_idx alone --
                        # see peak_detection_lead_indices above), merging
                        # any two detections within 100ms as the same
                        # physical beat. Each beat is still MEASURED on
                        # best_lead_idx's own signal below (padded_signal),
                        # keeping morphology/amplitude comparisons
                        # consistent on one lead -- only which timestamps
                        # count as a beat in the first place is made
                        # robust against one lead's transient dropout.
                        merge_tol_samples = max(1, int(0.1 * fs))
                        all_peak_times = []
                        for lead_idx in peak_detection_lead_indices:
                            if lead_idx >= padded_array.shape[0]:
                                continue
                            lead_sig = np.asarray(padded_array[lead_idx], dtype=float)
                            if len(lead_sig) <= 50:
                                continue
                            all_peak_times.extend(pan_tompkins(lead_sig, fs=fs))
                        all_peak_times.sort()
                        all_peaks = []
                        for p in all_peak_times:
                            if all_peaks and (p - all_peaks[-1]) < merge_tol_samples:
                                continue
                            all_peaks.append(p)
                        core_r_peaks = [
                            p for p in all_peaks
                            if current_time <= pad_start + p / fs < window_end
                        ]
                        if _is_vfib_window(padded_signal, core_r_peaks, fs):
                            target_morph = ('Ventricular Fibrillation', '#FF3333')
                        else:
                            atrial_label = _classify_atrial_window(padded_signal, core_r_peaks, fs)
                            if atrial_label:
                                target_morph = (atrial_label, '#FF00FF')

                if target_morph is None:
                    window_leads = {}
                    for i, lead_name in enumerate(lead_names):
                        if i < data_array.shape[0]:
                            window_leads[lead_name] = data_array[i, :]

                    if window_leads:
                        results = analyze_ecg(window_leads, fs=fs)
                        arrhythmias = results.get('arrhythmias', [])

                        # Only AV Block is looked up here. VFib is decided by
                        # its own direct rule above; Atrial Fibrillation /
                        # Atrial Flutter are now decided by the direct
                        # _classify_atrial_window() rule above as well, not by
                        # analyze_ecg()'s internal spectral/ratio scoring.
                        for arr in arrhythmias:
                            arr_lower = str(arr).lower()
                            if 'av block' in arr_lower:
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

    # Atrial Fibrillation / Atrial Flutter get one more check here, specific
    # to those two labels: a SINGLE isolated window carrying one of them,
    # with no neighboring window agreeing on the same label, is dropped.
    # Verified directly on real recordings that the two failure modes this
    # catches are real: right at the on-ramp into a Ventricular Fibrillation
    # episode, the QRS can still look just barely "organized" for one window
    # before the next window clearly crosses into VFib -- that one window's
    # destabilizing baseline was reading as Atrial Fibrillation by this
    # rule's own logic, even though it is really just the VFib transition,
    # not a separate rhythm. The genuine Atrial Flutter test recording never
    # had this problem: every true episode showed many consecutive windows
    # agreeing, never a lone one. VFib's own rule is untouched by this.
    if morphological_candidates:
        _sorted_by_time = sorted(morphological_candidates, key=lambda x: x['start_sec'])
        _confirmed = []
        for _i, _m in enumerate(_sorted_by_time):
            if _m['label'] not in ('Atrial Fibrillation', 'Atrial Flutter'):
                _confirmed.append(_m)
                continue
            # Gap tolerance is 2x window_size here (see the merge step right
            # below, which uses the same reasoning and the same value) --
            # confirmed directly against a real recording: a brief
            # ~10-second Atrial Flutter episode whose conduction rate was
            # itself accelerating left a couple of windows in the middle
            # too data-starved (too few beats landed with room to judge
            # their own TP segment -- see _classify_atrial_window's
            # min-scored guard) to vote either way, widening the real gap
            # between two confidently-Flutter windows to about 2 window
            # lengths even though the episode itself never stopped.
            _neighbor_tolerance = window_size * 2
            _prev_match = _i > 0 and _sorted_by_time[_i - 1]['label'] == _m['label'] and (_m['start_sec'] - _sorted_by_time[_i - 1]['start_sec']) <= _neighbor_tolerance
            _next_match = _i + 1 < len(_sorted_by_time) and _sorted_by_time[_i + 1]['label'] == _m['label'] and (_sorted_by_time[_i + 1]['start_sec'] - _m['start_sec']) <= _neighbor_tolerance
            if _prev_match or _next_match:
                _confirmed.append(_m)
        morphological_candidates = _confirmed

    # Merge contiguous morphological candidate windows. The gap tolerance is
    # 2x window_size, not a tiny fraction of a second: each window's
    # pass/fail is a per-window majority vote over a handful of beats, so a
    # single borderline window right in the middle of a sustained episode can
    # narrowly miss the threshold by chance (one or two beats' worth of
    # noise) even though the episode never actually stopped -- and for
    # Atrial Fibrillation/Flutter specifically, a window can also go
    # unclassified simply for lacking enough beats with room to judge their
    # own TP segment (see _classify_atrial_window's min-scored guard),
    # which is more likely right when a Flutter episode's conduction rate
    # is itself changing. A short gap between two windows that both
    # independently fired as the same morphological label is almost always
    # one of those two things, not a real few-second recovery in the middle
    # of the episode -- so it gets bridged into one continuous segment.
    if morphological_candidates:
        morphological_candidates.sort(key=lambda x: x['start_sec'])
        merged_morph = []
        for m in morphological_candidates:
            if merged_morph:
                last = merged_morph[-1]
                if last['label'] == m['label'] and m['start_sec'] <= last['end_sec'] + window_size * 2:
                    last['end_sec'] = max(last['end_sec'], m['end_sec'])
                    last['end_time_str'] = m['end_time_str']
                    continue
            merged_morph.append(m)

        # Beat-exact boundary refinement for Ventricular Fibrillation only.
        # The window vote above locates a VFib region to within one
        # window/step's own resolution (a whole 10s window either votes for
        # it or doesn't), but the real disorganized-QRS stretch on the strip
        # rarely starts and ends exactly on a window edge -- confirmed
        # directly on this module's own test recording: the window vote
        # called a 5s VFib region (e.g. 455-460s) while the raw signal's
        # actual chaotic stretch ran from about 452s to 465s, so 460-465s
        # was left mislabeled as whatever rhythm follows. Snap each VFib
        # region's own start/end to the nearest BEAT where organized vs.
        # disorganized QRS actually flips, by re-measuring individual beats
        # (not whole windows) out to one window_size past each edge --
        # requiring two consecutive beats to agree before accepting the
        # flip, so one noisy/ectopic beat can't drag the boundary.
        def _beat_is_organized(peak_sec):
            half = 0.4
            seg_start = max(0.0, peak_sec - half)
            seg_end = min(total_duration, peak_sec + half)
            sig = reader.read_range(seg_start, seg_end)
            if sig is None or best_lead_idx >= sig.shape[0]:
                return None
            lead_sig = np.asarray(sig[best_lead_idx], dtype=float)
            local_idx = int(round((peak_sec - seg_start) * fs))
            if local_idx <= 0 or local_idx >= len(lead_sig):
                return None
            beat = measure_beat(lead_sig, local_idx, fs)
            if beat is None:
                return None
            return bool(beat.get('qrs_bounds_resolved'))

        def _refine_vfib_edge(boundary_sec, edge):
            lo, hi = boundary_sec - window_size, boundary_sec + window_size
            nearby = sorted(float(p) for p in all_r_peaks if lo <= p <= hi)
            states = [(p, _beat_is_organized(p)) for p in nearby]
            for i in range(len(states) - 1):
                ts0, o0 = states[i]
                _, o1 = states[i + 1]
                if edge == 'start' and o0 is False and o1 is False:
                    return ts0
                if edge == 'end' and o0 is True and o1 is True:
                    return ts0
            return boundary_sec

        for idx, region in enumerate(merged_morph):
            if region['label'] != 'Ventricular Fibrillation':
                continue
            try:
                new_start = _refine_vfib_edge(region['start_sec'], 'start')
                new_end = _refine_vfib_edge(region['end_sec'], 'end')
                if new_start < new_end - 0.5:
                    region['start_sec'], region['end_sec'] = new_start, new_end
                    region['start_time_str'], region['end_time_str'] = make_time_strs(new_start, new_end)
                    # Pull the immediate neighbors' own edge in to meet this
                    # freshly beat-verified boundary too -- otherwise the
                    # generic overlap-trim pass right below, which always
                    # cuts back whichever of two overlapping regions comes
                    # FIRST, would chop this refined (and more trustworthy)
                    # VFib edge straight back down to the neighbor's own
                    # unrefined, coarse window edge.
                    if idx > 0 and merged_morph[idx - 1]['end_sec'] > new_start:
                        merged_morph[idx - 1]['end_sec'] = new_start
                    if idx + 1 < len(merged_morph) and merged_morph[idx + 1]['start_sec'] < new_end:
                        merged_morph[idx + 1]['start_sec'] = new_end
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] VFib boundary refine error: {e}")

        # The same 50%-overlapping sliding window that lets one episode span
        # several windows also means a transition from one morphological
        # label to another (e.g. Ventricular Fibrillation settling into
        # Atrial Flutter) has a few seconds where consecutive windows
        # disagree -- the last window still called the old label, the next
        # window already calls the new one. Each becomes its own candidate
        # above, so without this pass the two finished segments overlap in
        # time (both get drawn, stacked, over that shared stretch). Trim the
        # earlier segment back to where the next one starts so segments of
        # different labels never overlap -- the boundary is simply where the
        # window-by-window classification actually changed.
        # The same per-window voting that can leave two different-label
        # regions overlapping (handled above) can also leave a short GAP
        # between them: the single window straddling the true transition
        # often confirms neither the old label (its QRS/TP picture is
        # already decaying) nor the new one (not chaotic/organized enough
        # yet to clear that rule's own threshold), or it does vote for the
        # old label but gets discarded by the isolated-window guard above
        # once its neighbor on the new-label side stops agreeing. Either
        # way, that stretch never becomes its own candidate, so without
        # this pass it renders as a blank, unlabeled strip between two
        # confirmed episodes even though the recording never actually left
        # the earlier rhythm until the next one was confirmed. Bridging is
        # capped at one window length -- that is the largest span a single
        # ambiguous transition window can account for; any larger gap is a
        # genuine unclassified stretch, not a transition artifact, and is
        # left alone.
        for idx in range(len(merged_morph) - 1):
            cur = merged_morph[idx]
            nxt = merged_morph[idx + 1]
            if nxt['start_sec'] < cur['end_sec']:
                cur['end_sec'] = nxt['start_sec']
                _, cur['end_time_str'] = make_time_strs(cur['start_sec'], cur['end_sec'])
            elif 0.0 < (nxt['start_sec'] - cur['end_sec']) <= window_size:
                cur['end_sec'] = nxt['start_sec']
                cur['end_time_str'] = nxt['start_time_str']
        merged_morph = [m for m in merged_morph if m['end_sec'] > m['start_sec']]

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

    # This function is not always called just once per engine: the engine
    # object itself can outlive a single Full Disclosure dialog (e.g. the
    # main window keeps one engine alive per loaded recording across
    # close/reopen), and apply_auto_segments_to_engine() is re-invoked on
    # that same engine each time the dialog opens. Every call used to
    # simply APPEND this run's fresh detections on top of whatever was
    # already in engine._structured_events -- including this exact
    # function's own output from an earlier call, before a fix like the
    # overlap-trim or min-beat-count changes. The by-label merge below
    # then fused the OLD (stale) 'Auto'-sourced region together with the
    # NEW one (same label, close in time), producing a widened/overlapping
    # ghost segment that a later, corrected detection run could never
    # actually clear -- it only ever got added to, never replaced. Confirmed
    # directly: a Ventricular Fibrillation region kept showing a stale
    # wider boundary indefinitely, surviving across dialog close/reopen
    # (though not a full process restart, which starts a fresh engine with
    # an empty _structured_events instead). Dropping this function's own
    # prior output before adding this run's result makes each call
    # idempotent -- only genuinely external data (stored replay metrics,
    # manual annotations) can ever persist from before this call.
    # Beyond this function's own repeated-call leftovers (just handled
    # above), the recording's ORIGINAL stored replay data -- loaded by
    # HolterReplayEngine before this function ever runs, from the live
    # recording session's own real-time pipeline -- can carry its own
    # 'analysis'-sourced entries for the exact same morphological labels
    # this detector directly rules on (Ventricular Fibrillation, Atrial
    # Fibrillation, Atrial Flutter, AV Block). That real-time pipeline uses
    # an older, less-reliable scoring approach; this module exists
    # specifically to replace it as the authority on those labels with a
    # directly-verified rule. Left in place, the by-label merge below
    # treats an 'analysis' entry as just another same-label region to fuse
    # with this run's fresh one -- confirmed directly: on a brand new
    # engine's very first call (no repeat-call pollution possible yet), a
    # stored 'analysis' Ventricular Fibrillation entry widened a genuine
    # 120-240s fresh detection into 120-250s by simple virtue of being
    # close enough in time to merge with, and did the same to a 455-460s
    # region, widening it to 455-470s. Entries for any OTHER label (Sinus
    # Bradycardia/Tachycardia, or anything this detector doesn't itself
    # classify) are left alone -- only these four morphological categories
    # are this detector's own territory to decide.
    _owned_labels = ('ventricular fibrillation', 'atrial fibrillation', 'atrial flutter')

    def _is_owned_label(label_lower):
        return label_lower in _owned_labels or 'av block' in label_lower

    existing = [
        ev for ev in (getattr(engine, '_structured_events', []) or [])
        if str(ev.get('source', '')) != 'Auto'
        and not (
            'manual' not in str(ev.get('source', '')).lower()
            and _is_owned_label(str(ev.get('label', ev.get('type', ''))).lower())
        )
    ]
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
