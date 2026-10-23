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
                                           P/PR/QRS conduction plus RR
                                           intervals) and morphological
                                           detection (Ventricular
                                           Fibrillation/Tachycardia, Atrial
                                           Fibrillation/Flutter, SVT, and
                                           First-degree / Mobitz I / Mobitz
                                           II / Third-degree AV Block, each
                                           by its own direct rule here).
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

import copy
import hashlib
import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
from scipy.signal import find_peaks, savgol_filter


# Detection takes ~50s on a 23-minute recording and was being run again by
# every caller (AF panel on load, Full Disclosure on open, report). Results
# are cached in memory for this process and on disk next to the recording,
# keyed on the recording file's size/mtime and a hash of the ecg package's
# source, so any change to the recording or to detection code recomputes.
_CACHE_FILENAME = 'auto_segments_cache.json'
_memory_cache: Dict[str, List[Dict[str, Any]]] = {}
_code_fingerprint_value: Optional[str] = None


def _code_fingerprint() -> str:
    global _code_fingerprint_value
    if _code_fingerprint_value is None:
        ecg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        h = hashlib.sha1()
        n_files = 0
        for root, dirs, files in os.walk(ecg_dir):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__')
            for name in sorted(files):
                if not name.endswith('.py'):
                    continue
                try:
                    with open(os.path.join(root, name), 'rb') as f:
                        h.update(os.path.relpath(os.path.join(root, name), ecg_dir).encode('utf-8'))
                        h.update(f.read())
                    n_files += 1
                except OSError:
                    continue
        # No readable sources (e.g. a frozen build): disk cache is skipped,
        # since there's no way to tell whether a saved result is stale.
        _code_fingerprint_value = h.hexdigest() if n_files else ''
    return _code_fingerprint_value


def _cache_key(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{_code_fingerprint()}"


def _disk_cache_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), _CACHE_FILENAME)


# Whether the user has actually clicked "Auto Arrhythmia Detection" in Full
# Disclosure for this recording, persisted next to the recording so that
# generate_holter_report() -- which loads a session fresh from disk and never
# touches the live dialog's engine -- can tell. Without this, the report
# unconditionally ran detect_arrhythmias() and showed every auto-detected
# region regardless of whether the button was ever pressed, which is exactly
# what the button is meant to gate (see holter_full_disclosure.py's
# _auto_marking_visible). Report generation reports plain NSR for a session
# this flag isn't set on.
_MARKING_ENABLED_FILENAME = 'auto_arrhythmia_marking_enabled.json'


def _marking_enabled_path(session_dir: str) -> str:
    return os.path.join(session_dir, _MARKING_ENABLED_FILENAME)


def is_auto_marking_enabled(session_dir: str) -> bool:
    """True once set_auto_marking_enabled() has been called for this
    session (i.e. the user clicked the button at least once)."""
    try:
        with open(_marking_enabled_path(session_dir), 'r', encoding='utf-8') as f:
            return bool(json.load(f).get('enabled'))
    except (OSError, ValueError):
        return False


def set_auto_marking_enabled(session_dir: str) -> None:
    """Record that auto arrhythmia marking has been enabled for this
    session. Called once the button's detection result has been applied."""
    try:
        os.makedirs(session_dir, exist_ok=True)
        with open(_marking_enabled_path(session_dir), 'w', encoding='utf-8') as f:
            json.dump({'enabled': True}, f)
    except OSError as e:
        print(f"[Auto Arrhythmia Detect] Could not persist marking-enabled flag: {e}")


def clear_auto_marking_enabled(session_dir: str) -> None:
    """Undo counterpart to set_auto_marking_enabled() -- called when the
    user undoes the "Auto Arrhythmia Detection" click itself, so report
    generation goes back to reporting plain NSR for this session."""
    try:
        path = _marking_enabled_path(session_dir)
        if os.path.exists(path):
            os.remove(path)
    except OSError as e:
        print(f"[Auto Arrhythmia Detect] Could not clear marking-enabled flag: {e}")


def get_cached_detection(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Cached detect_arrhythmias() result for this recording, or None. Cheap."""
    key = _cache_key(path)
    if key is None:
        return None
    if key in _memory_cache:
        return copy.deepcopy(_memory_cache[key])
    if not _code_fingerprint():
        return None
    try:
        with open(_disk_cache_path(path), 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if saved.get('key') != key or not isinstance(saved.get('segments'), list):
        return None
    _memory_cache[key] = saved['segments']
    return copy.deepcopy(saved['segments'])


def detect_arrhythmias(
    reader,
    progress_callback: Optional[Callable[[int], None]] = None
) -> List[Dict[str, Any]]:
    """Cached wrapper around _detect_arrhythmias_uncached (see the cache note above)."""
    path = getattr(reader, 'path', None)
    key = _cache_key(path)
    if key is None:
        return _detect_arrhythmias_uncached(reader, progress_callback)

    cached = get_cached_detection(path)
    if cached is not None:
        if progress_callback:
            progress_callback(100)
        return cached

    segments = _detect_arrhythmias_uncached(reader, progress_callback)
    _memory_cache[key] = copy.deepcopy(segments)
    if _code_fingerprint():
        cache_path = _disk_cache_path(path)
        try:
            tmp_path = cache_path + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump({'key': key, 'segments': segments}, f, default=float)
            os.replace(tmp_path, cache_path)
        except (OSError, TypeError, ValueError) as e:
            print(f"[Auto Arrhythmia Detect] Could not save cache: {e}")
    return segments


# Background detection, so opening a recording / Full Disclosure doesn't
# block the UI the first time a recording is analysed. One job per recording:
# a second request while it is running just waits for the same result.
_async_jobs: Dict[str, Dict[str, Any]] = {}


def detect_arrhythmias_async(path: str, on_done: Callable[[List[Dict[str, Any]]], None]) -> None:
    """Run detect_arrhythmias() for the recording at `path` on a worker
    thread and call on_done(segments) on the Qt main thread. Must be called
    from the Qt main thread."""
    import threading
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

    cached = get_cached_detection(path)
    if cached is not None:
        QTimer.singleShot(0, lambda: on_done(cached))
        return

    key = _cache_key(path) or os.path.abspath(path)
    job = _async_jobs.get(key)
    if job is not None:
        job.callbacks.append(on_done)
        return

    class _Job(QObject):
        # Created on the main thread; done is emitted from the worker thread,
        # so _deliver (a slot of this main-thread object) runs queued on the
        # main thread.
        done = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.callbacks = [on_done]
            self.done.connect(self._deliver)

        @pyqtSlot(object)
        def _deliver(self, segments):
            _async_jobs.pop(key, None)
            for cb in self.callbacks:
                try:
                    cb(copy.deepcopy(segments))
                except RuntimeError:
                    pass  # the receiving widget was closed meanwhile
                except Exception as e:
                    print(f"[Auto Arrhythmia Detect] Callback error: {e}")

    job = _Job()
    _async_jobs[key] = job
    notifier = job

    def _run():
        try:
            from .file_format import ECGHFileReader
            reader = ECGHFileReader(path)
            try:
                segments = detect_arrhythmias(reader)
            finally:
                reader.close()
        except Exception as e:
            print(f"[Auto Arrhythmia Detect] Background detection failed: {e}")
            segments = []
        notifier.done.emit(segments)

    threading.Thread(target=_run, name='auto-segment-detection', daemon=True).start()


def _detect_arrhythmias_uncached(
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
    from ecg.arrhythmia_detector import measure_beat
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:
        from ..pan_tompkins import pan_tompkins

    # Noise/artifact gate for the morphological rules below (VFib, AFib,
    # Atrial Flutter, AV Block). Those rules are judged purely off waveform
    # shape -- e.g. VFib is "no P wave + no organized QRS for most beats" --
    # which a genuinely chaotic but non-physiological signal (motion
    # artifact, a loose/falling-off electrode, EMG noise) satisfies just as
    # well as real Ventricular Fibrillation does. There was no independent
    # check that the window is even a usable ECG signal before classifying
    # it, so sustained noise could get auto-marked as a lethal arrhythmia.
    # Reuses the exact same SQI scoring and 0.6 cutoff as the real-time
    # recording pipeline (SignalQuality.compute_sqi / signal_quality_threshold
    # in config/clinical_config.yaml) so "noisy" means the same thing
    # everywhere in the app, not a second, separately-tuned definition.
    # The same config also drives _atrial_preproc: 0.5-40 Hz bandpass + mains
    # notch, applied inside _classify_atrial_window only. Its TP-segment and
    # P-wave checks were reading the raw .ecgh samples, where 50 Hz mains
    # ripple alone exceeded the "quiet baseline" threshold on every beat of
    # a clean, regular NSR recording and got called Atrial Fibrillation.
    try:
        from .analysis_pipeline import SignalQuality, SignalPreprocessor, HolterConfig
        _sqi_config = HolterConfig.from_clinical_config()
        _signal_quality = SignalQuality(fs=getattr(reader, 'fs', 500) or 500, config=_sqi_config)
        _sqi_min = _sqi_config.sqi_min
        _atrial_preproc = SignalPreprocessor(fs=getattr(reader, 'fs', 500) or 500, config=_sqi_config)
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Signal-quality gate unavailable, proceeding without it: {e}")
        _signal_quality = None
        _sqi_min = 0.6
        _atrial_preproc = None

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

    # Rate labels are sinus labels, so an RR pattern alone is not enough.
    # Measure every R-peak on the same best lead before rate segmentation and
    # use these beat-level measurements as a conduction gate below. A small
    # amount of padding keeps P/QRS searches away from each read chunk's
    # artificial edges.
    rate_beat_metrics = [None] * len(all_r_peaks)
    # The Sinus Bradycardia / Tachycardia gate measures P/PR/QRS on this
    # parallel list instead: the same beats measured on the signal after a
    # 25 Hz zero-phase low-pass, which is the signal the top-bar PR/QRS and
    # measure_beat's PR logic were tuned on. On the raw samples a small bump
    # (60-70 counts) ~55 ms before the QRS is closer than the real P wave, so
    # the strict P search takes it, its PR (10-20 ms) fails the 60-400 ms
    # range, and a clean 45 bpm run had PR on 0 of 12 beats and no Bradycardia
    # label (the top bar showed PR 152 ms for the same beats, as it reads the
    # filtered strip). rate_beat_metrics itself stays raw: the Third-degree
    # AV Block edge refinement reads it and is unchanged.
    rate_gate_metrics = [None] * len(all_r_peaks)
    from scipy.signal import butter as _butter, filtfilt as _filtfilt
    _gate_ba = _butter(2, 25.0 / (float(fs) / 2.0), btype='lowpass')

    def _gate_filter(x):
        try:
            return _filtfilt(_gate_ba[0], _gate_ba[1], x)
        except Exception:
            return x
    # Lead II is the preferred conduction lead: it is the most reliable lead
    # for visible P waves and PR intervals in these recordings. R-peaks still
    # come from the strongest detection lead, but P/PR/QRS gating must not be
    # judged on (for example) V5 where the preceding T-wave can be mistaken
    # for P during a slow rhythm.
    conduction_lead_idx = lead_names.index('II') if 'II' in lead_names else best_lead_idx
    metric_cursor = 0
    metric_pad_sec = 1.0
    metric_chunk_sec = 60.0
    while metric_cursor < len(all_r_peaks):
        metric_start = max(0.0, all_r_peaks[metric_cursor] - metric_pad_sec)
        metric_end = min(
            total_duration,
            max(metric_start + metric_chunk_sec, all_r_peaks[metric_cursor] + metric_pad_sec),
        )
        metric_data = reader.read_range(metric_start, metric_end)
        if metric_data is None or conduction_lead_idx >= metric_data.shape[0]:
            metric_cursor += 1
            continue
        metric_signal = np.asarray(metric_data[conduction_lead_idx], dtype=float)
        gate_signal = _gate_filter(metric_signal)
        while metric_cursor < len(all_r_peaks):
            peak_sec = all_r_peaks[metric_cursor]
            if peak_sec >= metric_end - metric_pad_sec and metric_end < total_duration:
                break
            if peak_sec < metric_start:
                metric_cursor += 1
                continue
            local_peak = int(round((peak_sec - metric_start) * fs))
            prev_rr_ms = (all_r_peaks[metric_cursor] - all_r_peaks[metric_cursor - 1]) * 1000.0 if metric_cursor > 0 else None
            if 0 < local_peak < len(metric_signal):
                rate_beat_metrics[metric_cursor] = measure_beat(
                    metric_signal, local_peak, fs, strict_pr=True, prev_rr_ms=prev_rr_ms
                )
                metric = rate_beat_metrics[metric_cursor]
                if metric and metric.get('p_peak') is not None:
                    metric['_p_time_sec'] = metric_start + float(metric['p_peak']) / float(fs)
                if metric:
                    metric['_r_time_sec'] = peak_sec
                gate_metric = measure_beat(
                    gate_signal, local_peak, fs, strict_pr=True, prev_rr_ms=prev_rr_ms
                )
                rate_gate_metrics[metric_cursor] = gate_metric
                if gate_metric and gate_metric.get('p_peak') is not None:
                    gate_metric['_p_time_sec'] = metric_start + float(gate_metric['p_peak']) / float(fs)
                if gate_metric:
                    gate_metric['_r_time_sec'] = peak_sec
            metric_cursor += 1

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

    def sinus_conduction_gate(start_idx, end_idx, metrics=None):
        """Require organized P-to-QRS conduction before a rate label.

        Bradycardia/Tachycardia are sinus labels, therefore an RR pattern by
        itself must not create them. Most beats in the candidate must have a
        real P before QRS, a valid and consistent PR, and a resolved,
        consistent QRS. P polarity/duration consistency is an additional
        guard against counting random baseline bumps as P waves.

        metrics: per-beat measurements the indices refer to; defaults to
        rate_beat_metrics (all_r_peaks). The Sinus Bradycardia rule passes
        its own list, see brady_r_peaks below.
        """
        if metrics is None:
            metrics = rate_gate_metrics
        if start_idx is None or end_idx is None:
            return False
        left = max(0, int(start_idx))
        right = min(len(metrics) - 1, int(end_idx))
        if right < left:
            return False
        beats = [b for b in metrics[left:right + 1] if b]
        if len(beats) < 3:
            return False

        valid = [
            b for b in beats
            if bool(b.get('p_present'))
            and b.get('p_onset') is not None
            and b.get('p_end') is not None
            and b.get('qrs_bounds_resolved')
            and b.get('q_onset') is not None
            and 40.0 <= float(b.get('qrs_ms') or 0.0) <= 200.0
        ]
        # A sinus label needs P/QRS evidence on at least 75% of the beats;
        # for the shortest permitted segment this means all three beats.
        required = max(3, int(np.ceil(len(beats) * 0.75)))
        if len(valid) < required:
            return False

        # PR is a beat-to-beat conduction measurement. Reject only when a
        # majority of the candidate's beats disagree; an occasional missed
        # P boundary or noise bump must not erase a genuine long brady run.
        # For short runs (<= 30 beats), use the candidate median; for long
        # sustained episodes (> 30 beats), use a rolling local median (+-15 beats)
        # so natural multi-minute autonomic PR drift does not erase true sinus rhythm.
        pr_valid = [b for b in valid if b.get('pr_ms') is not None]
        if len(pr_valid) < required:
            return False
        pr_values = np.asarray([float(b['pr_ms']) for b in pr_valid], dtype=float)
        if len(pr_values) <= 30:
            pr_median = float(np.median(pr_values))
            pr_consistent_count = int(np.count_nonzero(np.abs(pr_values - pr_median) <= 25.0))
        else:
            w = 15
            pr_consistent_count = 0
            for i in range(len(pr_values)):
                lo = max(0, i - w)
                hi = min(len(pr_values), i + w + 1)
                loc_med = float(np.median(pr_values[lo:hi]))
                if abs(pr_values[i] - loc_med) <= 25.0:
                    pr_consistent_count += 1
        if pr_consistent_count < required:
            return False

        qrs_values = np.asarray([float(b['qrs_ms']) for b in valid], dtype=float)
        if float(np.std(qrs_values)) > 20.0:
            return False

        # P-to-P timing should follow the ventricular cycle. This rejects
        # isolated noise bumps that happen to pass the individual P-wave
        # shape test, while allowing normal beat-to-beat rate variability.
        p_times = np.asarray(
            [float(b['_p_time_sec']) for b in valid if b.get('_p_time_sec') is not None],
            dtype=float,
        )
        if len(p_times) < required:
            return False
        r_times = np.asarray(
            [float(b['_r_time_sec']) for b in valid if b.get('_r_time_sec') is not None],
            dtype=float,
        )
        if len(r_times) < required:
            return False
        pp_values = np.diff(p_times) * 1000.0
        rr_for_p_values = np.diff(r_times) * 1000.0
        if pp_values.size and rr_for_p_values.size:
            pair_count = min(pp_values.size, rr_for_p_values.size)
            # In sinus rhythm the atrial cycle follows the ventricular cycle;
            # compare P-P to the matching R-R interval. This permits a real
            # rate transition inside a long brady run while rejecting a
            # random baseline bump whose timing does not track QRS.
            pp_rr_error = np.abs(pp_values[:pair_count] - rr_for_p_values[:pair_count])
            pp_rr_ok = pp_rr_error <= np.maximum(120.0, 0.15 * rr_for_p_values[:pair_count])
            if int(np.count_nonzero(pp_rr_ok)) < max(2, int(np.ceil(pair_count * 0.75))):
                return False

        # P morphology should not randomly flip polarity or change duration
        # from beat to beat. Amplitude magnitude is deliberately not used as
        # a hard equality because physiological P amplitude can vary.
        p_amplitudes = np.asarray([float(b.get('p_amplitude') or 0.0) for b in valid])
        p_durations = np.asarray([float(b.get('p_duration_ms') or 0.0) for b in valid])
        if len(p_amplitudes) <= 30:
            if max(np.count_nonzero(p_amplitudes >= 0.0), np.count_nonzero(p_amplitudes < 0.0)) < required:
                return False
        else:
            w = 15
            p_pol_consistent_count = 0
            for i in range(len(p_amplitudes)):
                lo = max(0, i - w)
                hi = min(len(p_amplitudes), i + w + 1)
                win = p_amplitudes[lo:hi]
                dom_pos = np.count_nonzero(win >= 0.0) >= np.count_nonzero(win < 0.0)
                if (p_amplitudes[i] >= 0.0) == dom_pos:
                    p_pol_consistent_count += 1
            if p_pol_consistent_count < required:
                return False
        if len(p_durations) <= 30:
            p_duration_median = float(np.median(p_durations))
            p_dur_consistent_count = int(np.count_nonzero(np.abs(p_durations - p_duration_median) <= 40.0))
        else:
            w = 15
            p_dur_consistent_count = 0
            for i in range(len(p_durations)):
                lo = max(0, i - w)
                hi = min(len(p_durations), i + w + 1)
                loc_med = float(np.median(p_durations[lo:hi]))
                if abs(p_durations[i] - loc_med) <= 40.0:
                    p_dur_consistent_count += 1
        if p_dur_consistent_count < required:
            return False
        return True

    def suppress_rate_overlap(rate_segments, morph_segments):
        """
        Morphological rhythms (VFib, VTach, AFib, Flutter, AV Block) are detected
        from actual QRS/P-wave morphology and take clinical priority over the
        P/PR/QRS-gated, RR-rate-based Sinus Tachycardia/Bradycardia segments above.
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
        transition artifact, not a real rhythm, so a piece that sits entirely
        within buffer_sec of a morphological edge is dropped.

        The buffer is NOT used for trimming itself: trimming against the
        buffered interval cut every real rate segment 5s short of the
        morphological one (a 120 bpm Sinus Tachycardia that ran right up to
        a VFib onset at 600.2s was cut at 595.2s, leaving an unlabeled gap
        of steady tachycardia beats). Rate segments are trimmed at the exact
        morphological edge.
        """
        if not morph_segments:
            return rate_segments
        buffer_sec = 5.0
        morph_intervals = sorted((m['start_sec'], m['end_sec']) for m in morph_segments)

        def _is_transition_blip(p_start, p_end):
            return any(
                (p_end <= m_start and p_start >= m_start - buffer_sec)
                or (p_start >= m_end and p_end <= m_end + buffer_sec)
                for m_start, m_end in morph_intervals
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
                if p_end - p_start < 1.0 or _is_transition_blip(p_start, p_end):
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
        # A. Sinus Tachycardia: first pass through the P/PR/QRS sinus gate,
        # then consecutive beats with RR <= 600ms (HR >= 100 bpm).
        in_tachy = False
        tachy_start_idx = None

        # Lower bound 340 ms, not 333.3: _detect_p_wave suppresses P whenever
        # the previous QRS sits within 333.3 ms, and RR at ~180 bpm jitters
        # around that edge. Those P-less beats sat inside the run and failed
        # the 75% P gate for the whole run, dropping valid 100-175 bpm sinus
        # tachycardia along with them.
        for i, rr in enumerate(all_rr_ms):
            is_fast = (340.0 <= rr <= 600.5) or (rr <= 630.0 and in_tachy and i + 1 < len(all_rr_ms) and 340.0 <= all_rr_ms[i + 1] <= 600.5)
            if is_fast:
                if not in_tachy:
                    in_tachy = True
                    tachy_start_idx = i
            else:
                if in_tachy:
                    tachy_end_idx = i
                    beat_count = tachy_end_idx - tachy_start_idx + 1
                    dur = all_r_peaks[tachy_end_idx] - all_r_peaks[tachy_start_idx]
                    if beat_count >= 3 and dur >= 1.0 \
                            and sinus_conduction_gate(tachy_start_idx, tachy_end_idx):
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
            if beat_count >= 3 and dur >= 1.0 \
                    and sinus_conduction_gate(tachy_start_idx, tachy_end_idx):
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

        # B. Sinus Bradycardia: first pass through the same P/PR/QRS sinus
        # gate, then consecutive beats with RR >= 1050ms (HR < 57.1 bpm)
        # Note: 60 bpm (1000ms RR) is Normal Sinus Rhythm (NSR: 60-100 bpm), NOT Bradycardia.
        # This detector's configured brady range is 10-57 bpm:
        # RR 1050-6000ms. Anything slower is treated as a pause/possible
        # signal-loss interval rather than sustained sinus bradycardia.
        #
        # An RR this "slow" must also stay under a ceiling, or a stretch
        # where Pan-Tompkins simply finds no peaks at all -- because the
        # rhythm has degenerated into chaos, e.g. a VFib onset -- reads as
        # one single multi-second "RR interval" and clears the >=1050ms bar
        # easily, mislabelling the start of VFib as Sinus Bradycardia.
        # Confirmed directly: a real recording had a 6.7s gap with zero
        # detected peaks right where VFib began, reported as a ~7s Sinus
        # Bradycardia episode because nothing capped how slow "slow" was
        # allowed to be.
        #
        # The app's own pause_threshold_ms (2000ms, used for "RRI
        # (>=2.0s)"/pause counting elsewhere) is NOT the right ceiling here
        # -- tried first and confirmed wrong directly: a different real
        # recording has genuine, sustained 30 bpm Sinus Bradycardia (RR
        # steady at ~2000ms for 2 straight minutes), which that cutoff
        # fragmented into disconnected ~4s pieces since roughly half its
        # real beats landed right at/above 2000ms. 6000ms (10 bpm) instead:
        # comfortably below the 6.7s chaos gap above while allowing the
        # requested lower boundary.
        _brady_max_rr_ms = 6000.0

        # Missed-beat recovery, for this Sinus Bradycardia rule ONLY. The
        # whole-recording R-peak pass above runs Pan-Tompkins on 60 s chunks,
        # and pan_tompkins() widens its refractory period to a third of the
        # chunk's MEDIAN RR. A chunk that is mostly a very slow rhythm (RR
        # ~4000 ms) then gets a ~1320 ms refractory period, which drops
        # beats of a 60 bpm rhythm in the same chunk -- two 1000 ms RR
        # intervals read as one 2000 ms "slow" RR, and Sinus Bradycardia was
        # drawn over plain 60 bpm beats (seen: 179.15/180.15/181.15/182.16 s,
        # only 179.15 and 181.15 detected, Bradycardia from 179.15 s instead
        # of 182.16 s). Each gap this rule could count as slow is re-checked
        # by running the same pan_tompkins() on just that gap +/-3 s, where
        # the median RR reflects the local rhythm; beats found inside the gap
        # are added to this rule's own beat list and measured exactly like
        # rate_beat_metrics. all_r_peaks itself is left untouched, so Sinus
        # Tachycardia and every morphological rule see the same beats as
        # before.
        _recovered_beats = []
        for i, rr in enumerate(all_rr_ms):
            if not (1050.0 <= rr <= _brady_max_rr_ms):
                continue
            gap_start, gap_end = all_r_peaks[i], all_r_peaks[i + 1]
            read_start = max(0.0, gap_start - 3.0)
            read_end = min(total_duration, gap_end + 3.0)
            gap_data = reader.read_range(read_start, read_end)
            if gap_data is None or best_lead_idx >= gap_data.shape[0]:
                continue
            gap_sig = np.asarray(gap_data[best_lead_idx], dtype=float)
            if len(gap_sig) <= 50:
                continue
            for p_idx in pan_tompkins(gap_sig, fs=fs):
                peak_sec = read_start + float(p_idx) / float(fs)
                if gap_start + 0.25 < peak_sec < gap_end - 0.25:
                    _recovered_beats.append(peak_sec)

        if _recovered_beats:
            brady_r_peaks = sorted(set(all_r_peaks) | set(_recovered_beats))
            _metric_at = dict(zip(all_r_peaks, rate_gate_metrics))
            brady_beat_metrics = []
            for k, peak_sec in enumerate(brady_r_peaks):
                if peak_sec in _metric_at:
                    brady_beat_metrics.append(_metric_at[peak_sec])
                    continue
                # Same measurement as rate_beat_metrics above: conduction
                # lead, strict P/PR, 1 s of signal on each side.
                metric = None
                m_start = max(0.0, peak_sec - 1.0)
                m_end = min(total_duration, peak_sec + 1.0)
                m_data = reader.read_range(m_start, m_end)
                if m_data is not None and conduction_lead_idx < m_data.shape[0]:
                    m_sig = _gate_filter(np.asarray(m_data[conduction_lead_idx], dtype=float))
                    local_peak = int(round((peak_sec - m_start) * fs))
                    prev_rr_ms = (peak_sec - brady_r_peaks[k - 1]) * 1000.0 if k > 0 else None
                    if 0 < local_peak < len(m_sig):
                        metric = measure_beat(m_sig, local_peak, fs, strict_pr=True, prev_rr_ms=prev_rr_ms)
                        if metric and metric.get('p_peak') is not None:
                            metric['_p_time_sec'] = m_start + float(metric['p_peak']) / float(fs)
                        if metric:
                            metric['_r_time_sec'] = peak_sec
                brady_beat_metrics.append(metric)
            brady_rr_ms = [
                (brady_r_peaks[k + 1] - brady_r_peaks[k]) * 1000.0
                for k in range(len(brady_r_peaks) - 1)
            ]
        else:
            brady_r_peaks, brady_beat_metrics, brady_rr_ms = all_r_peaks, rate_gate_metrics, all_rr_ms

        in_brady = False
        brady_start_idx = None

        for i, rr in enumerate(brady_rr_ms):
            next_rr = brady_rr_ms[i + 1] if i + 1 < len(brady_rr_ms) else None
            is_slow = (
                (1050.0 <= rr <= _brady_max_rr_ms)
                or (
                    1020.0 <= rr <= _brady_max_rr_ms and in_brady
                    and next_rr is not None and 1050.0 <= next_rr <= _brady_max_rr_ms
                )
            )
            if is_slow:
                if not in_brady:
                    in_brady = True
                    brady_start_idx = i
            else:
                if in_brady:
                    brady_end_idx = i
                    beat_count = brady_end_idx - brady_start_idx + 1
                    dur = brady_r_peaks[brady_end_idx] - brady_r_peaks[brady_start_idx]
                    if beat_count >= 3 and dur >= 2.5 \
                            and sinus_conduction_gate(brady_start_idx, brady_end_idx, brady_beat_metrics):
                        s_sec = brady_r_peaks[brady_start_idx]
                        e_sec = brady_r_peaks[brady_end_idx]
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
            brady_end_idx = len(brady_rr_ms)
            beat_count = brady_end_idx - brady_start_idx + 1
            dur = brady_r_peaks[brady_end_idx] - brady_r_peaks[brady_start_idx]
            if beat_count >= 3 and dur >= 2.5 \
                    and sinus_conduction_gate(brady_start_idx, brady_end_idx, brady_beat_metrics):
                s_sec = brady_r_peaks[brady_start_idx]
                e_sec = brady_r_peaks[brady_end_idx]
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
    # 3. Detect Morphological Arrhythmias (VFib, VT, AFib, Flutter, AV Block)
    # ------------------------------------------------------------------
    # Ventricular Fibrillation rule (simple, direct -- applied to every
    # rhythm window the same way): a window is VFib when its beats show
    # BOTH hallmark absences --
    #   1. no P wave reliably precedes a QRS (p_onset before q_onset), and
    #   2. the QRS complex itself is not organized/resolvable (irregular,
    #      chaotic -- _qrs_bounds() in arrhythmia_detector.py could not find
    #      a genuine isoelectric boundary for most beats).
    # Ventricular Tachycardia is evaluated separately below. It must not be
    # confused with VFib: VT has an organized, usually regular ventricular
    # rhythm with consistently wide QRS complexes, while VFib has no
    # resolvable/organized QRS.
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
    def _aligned_beat_shape_corr(signal, r_peaks, fs_):
        """Median correlation of the beats (120 ms around each detected beat)
        with their median template, each beat shifted up to +/-40 ms to its
        best match first. None when fewer than 3 beats fit."""
        x = np.asarray(signal, dtype=float)
        h, lag = int(0.06 * fs_), int(0.04 * fs_)
        peaks = [int(q) for q in r_peaks if q - h - lag >= 0 and q + h + lag <= x.size]
        if len(peaks) < 3:
            return None
        snips = [x[q - h:q + h] - float(np.mean(x[q - h:q + h])) for q in peaks]
        template = np.median(np.asarray(snips), axis=0)
        t_norm = float(np.linalg.norm(template))
        if t_norm <= 0:
            return None
        corrs = []
        for q in peaks:
            best = -1.0
            for shift in range(-lag, lag + 1, 2):
                seg = x[q + shift - h:q + shift + h]
                seg = seg - float(np.mean(seg))
                n = float(np.linalg.norm(seg))
                if n > 0:
                    best = max(best, float(np.dot(seg, template)) / (n * t_norm))
            corrs.append(best)
        return float(np.median(corrs))

    # Which test called the last window VFib: 'coarse' when only the shape
    # test below did (read right after the call; see the guard after the
    # window loop).
    _vfib_window_kind = {'last': None}

    def _is_vfib_window(window_signal, r_peaks, fs):
        _vfib_window_kind['last'] = None
        if len(r_peaks) < 2:
            return False
        # VF is a chaotic, disorganized ventricular rhythm. A regular rhythm
        # (such as regular VT or sinus tachycardia) cannot be VFib.
        if len(r_peaks) >= 3:
            rr_diffs = np.diff(np.asarray(r_peaks, dtype=float)) * 1000.0 / float(fs)
            mean_rr = float(np.mean(rr_diffs))
            if mean_rr > 0 and (float(np.std(rr_diffs)) / mean_rr) < 0.20:
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
        if no_p_follows_qrs and no_organized_qrs:
            return True
        # Coarse, slow VFib: big rounded waves whose sides flatten enough for
        # the QRS boundary search to "resolve" on half or more of them, so the
        # organized-QRS test above fails although no two waves look alike
        # (seen: ~100 s of VFib, 400-495 s, P before QRS on 0 beats, organized
        # 5/11, 6/11, 7/8 ...). Such a window is VFib when NOT ONE beat has a P
        # wave before it (irregular RR already required above) and the beats
        # don't share one shape: median correlation, each beat aligned
        # +/-40 ms to the window's own template (beat positions merged from
        # several leads shift by 20-50 ms), below 0.95. Measured on windows
        # reaching this rule: AFib and Flutter 0.97-1.00 (and they rarely have
        # no P at all), the missed VFib 0.48-0.93; windows of other rhythms
        # below 0.95 had P waves on 2-4 beats.
        # A window called VFib only by this test is kept only when a window
        # next to it is VFib too (see the coarse-VFib guard after the window
        # loop): extra detections on P / T waves in one plain sinus window can
        # look the same (seen: recording start, 12 "beats" for 10 real ones,
        # P missed on the best lead, correlation 0.69 -- one lone window).
        if p_follows_qrs == 0 and len(beats) >= 5:
            shape_corr = _aligned_beat_shape_corr(window_signal, r_peaks, fs)
            if shape_corr is not None and shape_corr < 0.95:
                _vfib_window_kind['last'] = 'coarse'
                return True
        return False

    def _is_ventricular_tachycardia_window(window_signal, r_peaks, fs):
        """Detect an organized fast wide-complex ventricular run."""
        if len(r_peaks) < 4:
            return False
        measured = [(int(p), measure_beat(window_signal, int(p), fs)) for p in r_peaks]
        beat_peaks = [p for p, b in measured if b is not None]
        beats = [b for _, b in measured if b is not None]
        if len(beats) < 4:
            return False

        rr_ms = np.diff(np.asarray(r_peaks, dtype=float)) * 1000.0 / float(fs)
        if rr_ms.size < 3 or np.any(rr_ms <= 0.0):
            return False
        # A window can straddle the onset of VT and contain a few preceding
        # transition beats. Do not let those mixed intervals reject the
        # stable ventricular rhythm that follows. Find the longest contiguous
        # RR run around the dominant (median) cycle and evaluate only that
        # rhythm core; transition beats are deliberately excluded.
        median_rr = float(np.median(rr_ms))
        stable_intervals = np.abs(rr_ms - median_rr) <= (0.15 * median_rr)
        best_start = best_end = None
        run_start = None
        for interval_idx, is_stable in enumerate(stable_intervals):
            if is_stable and run_start is None:
                run_start = interval_idx
            if (not is_stable or interval_idx == len(stable_intervals) - 1) and run_start is not None:
                run_end = interval_idx if is_stable and interval_idx == len(stable_intervals) - 1 else interval_idx - 1
                if best_start is None or (run_end - run_start) > (best_end - best_start):
                    best_start, best_end = run_start, run_end
                run_start = None
        # A window split about evenly between two rhythms (seen: 8 sinus RRs
        # at 750 ms, then 8 VT RRs at 375 ms) has its median RR in between
        # (526 ms), so no run sits within 15% of it. Only then, the longest
        # run of consecutive RRs within 15% of that run's own median is
        # taken instead -- and only if it is at least 6 RRs long: inside
        # VFib, 3-4 RRs can match by chance (seen: 318/352/310/304 ms amid
        # RR 228-3060 ms) and must not read as VT. A window whose core was
        # found above is unaffected.
        if best_start is None or (best_end - best_start + 1) < 3:
            best_start = best_end = None
            for s in range(len(rr_ms)):
                e = s
                while e + 1 < len(rr_ms):
                    run_median = float(np.median(rr_ms[s:e + 2]))
                    if np.all(np.abs(rr_ms[s:e + 2] - run_median) <= 0.15 * run_median):
                        e += 1
                    else:
                        break
                if best_start is None or (e - s) > (best_end - best_start):
                    best_start, best_end = s, e
            if best_start is not None and (best_end - best_start + 1) < 6:
                best_start = best_end = None
        if best_start is None or (best_end - best_start + 1) < 3:
            return False
        core_beats = beats[best_start:best_end + 2]
        core_rr_ms = rr_ms[best_start:best_end + 1]
        median_rr = float(np.median(core_rr_ms))
        if median_rr <= 0.0 or (60000.0 / median_rr) < 100.0:
            return False
        # Regular VT; irregular/chaotic ventricular activity remains outside
        # this rule and can be handled by the VFib/no-label paths.
        if float(np.std(core_rr_ms) / median_rr) > 0.12:
            return False
        if float(np.median(np.abs(core_rr_ms - median_rr)) / median_rr) > 0.10:
            return False

        # A bounded 160 ms estimate is still useful for VT when the broad
        # complex has no clean isoelectric S-offset. Do not relabel the beat
        # as globally "resolved"; use this controlled fallback only after
        # the independent VT regularity and P/PR gates have passed.
        wide_beats = [
            b for b in core_beats
            if b.get('qrs_ms') is not None
            and 120.0 <= float(b['qrs_ms']) <= 220.0
        ]
        if len(wide_beats) < max(3, int(np.ceil(0.75 * len(core_beats)))):
            return False
        qrs_ms = np.asarray([float(b['qrs_ms']) for b in wide_beats], dtype=float)
        if float(np.std(qrs_ms)) > 25.0:
            return False

        # Incidental noise bumps or dissociated atrial activity are tolerated;
        # only an organized P/PR majority with physiological AV conduction
        # (PR >= 100 ms) means the rhythm is supraventricular, not VT.
        # Wide VT recordings can make the generic P-finder report coincidental
        # sub-100ms bumps before QRS which are not conducted sinus P waves.
        # A P wave can't sit inside the previous beat's QRS: a "P" less than
        # 120 ms after the previous beat is that beat itself. At 160 bpm the
        # P search, run from a QRS onset placed late, reached back 370 ms
        # onto the previous VT complex (PR "310 ms" on 22 of 22 beats) and
        # called the run supraventricular. Sinus P waves (~150 ms before
        # their QRS, hundreds of ms after the previous beat) are unaffected.
        min_after_prev = int(round(0.12 * fs))

        def real_p(i, b):
            if b.get('p_peak') is None:
                return True
            prev = beat_peaks[best_start + i - 1] if best_start + i - 1 >= 0 else None
            return prev is None or int(b['p_peak']) - prev >= min_after_prev

        p_present = sum(
            1 for i, b in enumerate(core_beats)
            if b.get('p_present') and b.get('p_onset') is not None
            and b.get('q_onset') is not None and b['p_onset'] < b['q_onset']
            and (b.get('pr_ms') is not None and b['pr_ms'] >= 100.0)
            and real_p(i, b)
        )
        valid_pr = sum(
            1 for i, b in enumerate(core_beats)
            if b.get('pr_ms') is not None and b['pr_ms'] >= 100.0 and real_p(i, b)
        )
        return p_present <= 0.50 * len(core_beats) and valid_pr <= 0.50 * len(core_beats)

    def _ventricular_tachycardia_core_start(r_peaks, fs):
        """Return the first beat of the dominant regular VT run.

        The VT classifier may inspect a window that begins during a mixed
        transition.  Segment drawing must start at the stable ventricular
        run, not at the window boundary.
        """
        peaks = np.asarray(r_peaks, dtype=float)
        if peaks.size < 4:
            return None
        rr_ms = np.diff(peaks) * 1000.0 / float(fs)
        if rr_ms.size < 3 or np.any(rr_ms <= 0.0):
            return None
        median_rr = float(np.median(rr_ms))
        stable = np.abs(rr_ms - median_rr) <= (0.15 * median_rr)
        best_start = best_end = None
        run_start = None
        for idx, is_stable in enumerate(stable):
            if is_stable and run_start is None:
                run_start = idx
            if (not is_stable or idx == len(stable) - 1) and run_start is not None:
                run_end = idx if is_stable and idx == len(stable) - 1 else idx - 1
                if best_start is None or (run_end - run_start) > (best_end - best_start):
                    best_start, best_end = run_start, run_end
                run_start = None
        # Same fallback as _is_ventricular_tachycardia_window (>= 6 RRs): a
        # window split about evenly between sinus and VT has its median RR in
        # between.
        if best_start is None or (best_end - best_start + 1) < 3:
            best_start = best_end = None
            for s in range(len(rr_ms)):
                e = s
                while e + 1 < len(rr_ms):
                    run_median = float(np.median(rr_ms[s:e + 2]))
                    if np.all(np.abs(rr_ms[s:e + 2] - run_median) <= 0.15 * run_median):
                        e += 1
                    else:
                        break
                if best_start is None or (e - s) > (best_end - best_start):
                    best_start, best_end = s, e
            if best_start is not None and (best_end - best_start + 1) < 6:
                best_start = best_end = None
        if best_start is None or (best_end - best_start + 1) < 3:
            return None
        return float(peaks[best_start]) / float(fs)

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
    def _prepare_atrial_window_context(window_signal, r_peaks, fs):
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
        if _atrial_preproc is not None:
            try:
                window_signal = _atrial_preproc.process(window_signal)
            except Exception:
                pass
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

        return {
            'beats': beats,
            'rr_ms': rr_ms,
            'ventricular_rate': ventricular_rate,
            'quiet_count': quiet_count,
            'scored': scored,
            'total_waves': total_waves,
            'total_active_sec': total_active_sec,
        }

    def _classify_atrial_flutter_window(context):
        """Apply only the existing Atrial Flutter rule."""
        quiet_count = context['quiet_count']
        scored = context['scored']
        total_active_sec = context['total_active_sec']
        total_waves = context['total_waves']
        ventricular_rate = context['ventricular_rate']

        if (quiet_count / scored) < 0.4 and total_active_sec > 0:
            atrial_rate = (total_waves / total_active_sec) * 60.0
            ratio = atrial_rate / ventricular_rate
            if any(abs(ratio - whole) < 0.5 for whole in (2.0, 3.0, 4.0, 5.0)):
                return 'Atrial Flutter'
        return None

    def _classify_atrial_fibrillation_window(context):
        """Apply only the existing Atrial Fibrillation rule."""
        beats = context['beats']
        rr_ms = context['rr_ms']

        # Atrial Fibrillation means NO organized P wave: the atria fibrillate
        # instead of depolarizing as one, so no beat has a P wave at a fixed
        # distance before its QRS. "Organized" = most beats have a P wave AND
        # their PR intervals agree (sinus conduction). A P-finder that latches
        # onto a random f-wave bump in real AFib gives scattered PRs, so that
        # still counts as no organized P. AV blocks (P present, PR varying)
        # fall through to the AV block rules since this returns None for them.
        pr_values = [float(b['pr_ms']) for b in beats if b.get('p_present') and b.get('pr_ms') is not None]
        organized_p = (
            len(pr_values) >= max(3, int(0.6 * len(beats)))
            and float(np.std(pr_values)) <= 25.0
        )
        if organized_p:
            return None

        rr_diffs = np.diff(rr_ms)
        significant_diffs = rr_diffs[np.abs(rr_diffs) > 60.0]
        if significant_diffs.size >= 3:
            diff_signs = np.sign(significant_diffs)
            reversals = np.sum(diff_signs[1:] != diff_signs[:-1])
            reversal_ratio = reversals / (diff_signs.size - 1)
            if reversal_ratio >= 0.5:
                return 'Atrial Fibrillation'

        return None

    def _classify_atrial_window(window_signal, r_peaks, fs):
        """Preserve the original Flutter-then-AFib classification order."""
        context = _prepare_atrial_window_context(window_signal, r_peaks, fs)
        if context is None:
            return None

        flutter_label = _classify_atrial_flutter_window(context)
        if flutter_label is not None:
            return flutter_label
        return _classify_atrial_fibrillation_window(context)

    # Supraventricular Tachycardia (SVT) rule (own function, checked only after the VFib and
    # AFib/Flutter rules found nothing). Textbook picture: a regular narrow-
    # complex rhythm at 100-250/min, QRS complexes the same shape as the
    # sinus beats, and the P waves superimposed on the T waves of the
    # preceding beats. That last point is what separates it from Sinus
    # Tachycardia, which has the same rate and the same QRS: in sinus
    # tachycardia the T wave and the next P wave are still two separate
    # waves with the signal dipping back between them (measured on every
    # Sinus Tachycardia test recording, 105-160 bpm: two humps between
    # beats in 100% of intervals), whereas in Supraventricular Tachycardia they merge
    # into a single hump.
    #   1. at least 6 beats, rate 100-250/min;
    #   2. regular RR (coefficient of variation <= 0.08) -- rules out AFib,
    #      and variable-conduction Flutter (measured CV 0.19-0.47);
    #   3. organized QRS on >= 80% of beats, all of one shape (median
    #      correlation to the window's own QRS template >= 0.9);
    #   4. P on T: between each beat's QRS end (J point) and the next beat's
    #      QRS onset there is only ONE wave (T and P fused) in >= 70% of the
    #      judged intervals. A wave counts when its prominence is >= 30% of
    #      that stretch's own range, so a notch on the fused wave doesn't
    #      split it in two. Both polarities are tried (inverted T/P leads).
    # QRS width from measure_beat is not used: on these recordings it reads
    # >= 120 ms even for clearly narrow sinus beats.
    def _is_atrial_tachycardia_window(window_signal, r_peaks, fs):
        if len(r_peaks) < 6:
            return False
        if _atrial_preproc is not None:
            try:
                window_signal = _atrial_preproc.process(window_signal)
            except Exception:
                pass
        sig = np.asarray(window_signal, dtype=float)
        beats = [measure_beat(sig, int(p), fs) for p in r_peaks]
        if any(b is None for b in beats):
            return False

        rr_ms = np.diff(np.asarray(r_peaks, dtype=float)) * 1000.0 / float(fs)
        rr_mean = float(np.mean(rr_ms))
        if rr_mean <= 0:
            return False
        rate = 60000.0 / rr_mean
        if not (180.0 <= rate <= 300.0):
            return False
        if float(np.std(rr_ms)) / rr_mean > 0.08:
            return False

        organized = sum(1 for b in beats if b.get('qrs_bounds_resolved'))
        if organized < 0.8 * len(beats):
            return False

        half = int(0.05 * fs)
        qrs_shapes = [sig[int(p) - half:int(p) + half] for p in r_peaks
                      if int(p) - half >= 0 and int(p) + half <= len(sig)]
        if len(qrs_shapes) < 3:
            return False
        qrs_shapes = np.array([q - np.mean(q) for q in qrs_shapes])
        template = np.median(qrs_shapes, axis=0)
        if not np.any(template):
            return False
        corrs = [float(np.corrcoef(q, template)[0, 1]) for q in qrs_shapes if np.any(q)]
        if not corrs or float(np.median(corrs)) < 0.9:
            return False

        # The stretch between beats is cut at fixed distances from the R
        # peaks, not at measure_beat's own q_onset/j_point: when the P wave
        # sits close to its QRS, q_onset can land on the P wave itself,
        # which then drops out of the stretch and leaves only the T wave --
        # reading sinus tachycardia as "one fused wave" (seen on a test
        # waveform with q_onset 80 ms before R).
        edge = int(0.06 * fs)
        fused = 0
        judged = 0
        for i in range(1, len(beats)):
            seg_start = max(int(r_peaks[i - 1]) + edge, int(beats[i - 1].get('j_point') or 0))
            seg_end = int(r_peaks[i]) - edge
            if seg_end - seg_start < int(0.12 * fs):
                continue
            segment = sig[seg_start:seg_end]
            seg_range = float(np.ptp(segment))
            if seg_range <= 0:
                continue
            waves = max(
                len(find_peaks(pol * segment, prominence=0.3 * seg_range, distance=max(1, int(0.06 * fs)))[0])
                for pol in (1.0, -1.0)
            )
            judged += 1
            if waves <= 1:
                fused += 1
        # At HR >= 180 the P-on-T fusion test above gets unreliable on its
        # own merits: the TP stretch shrinks to almost nothing, so a wave
        # finder has very little room to tell "one fused hump" from "two
        # waves crowded together" -- the same ambiguity that separates
        # Sinus Tachycardia from Supraventricular Tachycardia in the first place, per
        # the textbook criteria this rule is built from. Practically,
        # resting sinus tachycardia this fast and sustained is not
        # physiological for an adult Holter recording; once rate, QRS shape,
        # and RR regularity (gates 1-3 above) already match, a sustained
        # rate >= 180 bpm is itself the deciding factor, independent of how
        # the fused-wave count happens to land.
        if rate >= 180.0:
            return True

        if judged < 4:
            return False
        return fused / judged >= 0.7

    # ------------------------------------------------------------------
    # AV conduction blocks: First-degree, Second-degree Mobitz I,
    # Second-degree Mobitz II, Third-degree. Each has its own rule function
    # below, all reading one shared per-window picture of the atrial (P wave)
    # rhythm and how each P wave is -- or is not -- conducted to a QRS.
    #
    # Checked only after VT, VFib, AFib/Flutter and SVT all found nothing for
    # the window, and in severity order (Third -> Mobitz II -> Mobitz I ->
    # First), so a window only ever gets one label. Every rule requires a
    # regular atrial rhythm of discrete P waves (atrial rate 30-200/min), so
    # AFib (no P waves) and Flutter (atrial rate 250-350/min) can never reach
    # them, and >= 80% organized QRS, so VFib can't either. The same regular
    # P-wave train, when found, also vetoes an AFib/Flutter call for the
    # window (see the window loop), so a block isn't taken for AFib either.
    #
    # Why P waves are found here instead of reading measure_beat()'s own P:
    # measure_beat() only looks 50-400 ms before each QRS, so it can never
    # see a P wave that is NOT followed by a QRS -- the blocked P wave that
    # defines 2nd- and 3rd-degree block. Instead, P waves measure_beat() did
    # find are used as a template, and the whole window is scanned for that
    # shape. P waves due (by the steady P-P interval) inside a QRS or T wave,
    # where the shape can't be matched, are inferred and flagged hidden.
    # ------------------------------------------------------------------
    _AV_FIRST = 'First-degree AV Block'
    _AV_MOBITZ_I = 'Second-degree AV Block (Mobitz I)'
    _AV_MOBITZ_II = 'Second-degree AV Block (Mobitz II)'
    _AV_THIRD = 'Third-degree AV Block'
    _AV_COLORS = {
        _AV_FIRST: '#FFD700',      # gold
        _AV_MOBITZ_I: "#3D3AD4",   # royal blue
        _AV_MOBITZ_II: '#FF00FF',  # magenta
        _AV_THIRD: '#C0FF00',      # chartreuse
    }
    # Most severe first. A region of a less severe block is cut wherever it
    # overlaps a more severe one (or any non-AV rhythm), see below.
    _AV_LABELS = (_AV_THIRD, _AV_MOBITZ_II, _AV_MOBITZ_I, _AV_FIRST)
    # Beat-exact AV spans from overlapping windows touch or overlap; anything
    # further apart than this is a real stretch of normal conduction between
    # two episodes and must not be bridged over.
    _AV_MERGE_GAP_SEC = 2.5

    def _prepare_av_conduction_context(conduction_signal, qrs_peaks, core_r_peaks, fs):
        """Map every P wave in the window to the QRS it conducts to (or none).

        conduction_signal: Lead II of the padded window read.
        qrs_peaks: every R-peak in the padded read (sample indices into
            conduction_signal) -- used so a P wave near the window's own edge
            still finds the QRS that follows it.
        core_r_peaks: the R-peaks inside this window's true bounds; only P
            waves from just before the first of these up to the last are judged.
        """
        core = sorted(int(p) for p in core_r_peaks)
        if len(core) < 3:
            return None
        sig = np.asarray(conduction_signal, dtype=float)
        if _atrial_preproc is not None:
            try:
                sig = np.asarray(_atrial_preproc.process(sig), dtype=float)
            except Exception:
                pass
        n = sig.size

        def ms(v):
            return int(round(v * fs / 1000.0))

        peaks = sorted(set(int(p) for p in qrs_peaks if 0 < int(p) < n) | set(core))
        beats = {p: measure_beat(sig, p, fs) for p in peaks}
        if any(beats.get(p) is None for p in core):
            return None
        # Organized ventricular activity: >= 80% of QRS boundaries resolved,
        # OR one QRS shape beat to beat (median correlation to the window's
        # own QRS template >= 0.9, the SVT rule's test). A wide, abnormally
        # shaped QRS -- a ventricular escape rhythm in complete block, or a
        # bundle branch block -- often never returns cleanly to baseline, so
        # its boundary doesn't "resolve" even though every beat is the same
        # organized complex; VFib's chaotic beats fail both tests.
        qrs_resolved_majority = sum(1 for p in core if beats[p].get('qrs_bounds_resolved')) >= 0.8 * len(core)
        if not qrs_resolved_majority:
            half = ms(50)
            shapes_q = [sig[p - half:p + half] for p in core if p - half >= 0 and p + half <= n]
            if len(shapes_q) < 3:
                return None
            shapes_q = np.array([s - np.mean(s) for s in shapes_q])
            template_q = np.median(shapes_q, axis=0)
            if not np.any(template_q):
                return None
            corrs_q = [float(np.corrcoef(s, template_q)[0, 1]) for s in shapes_q if np.any(s)]
            if not corrs_q or float(np.median(corrs_q)) < 0.9:
                return None

        def q_onset_of(p):
            b = beats.get(p)
            q = b.get('q_onset') if b else None
            return int(q) if q is not None else p - ms(40)

        # Where a P wave can't be seen: inside a QRS, or inside the T wave
        # that follows it.
        qrs_zones, t_zones = [], []
        for i, p in enumerate(peaks):
            b = beats.get(p)
            if b is not None and b.get('qrs_bounds_resolved'):
                z0 = min(int(b['q_onset']), p - ms(40)) - ms(20)
                z1 = max(int(b['j_point']), p + ms(40)) + ms(20)
            else:
                z0, z1 = p - ms(80), p + ms(80)
            qrs_zones.append((z0, z1))
            t_end = b.get('t_end') if b is not None else None
            # The T wave ends within ~450 ms of R even at slow rates. After a
            # long pause measure_beat()'s T-end search can run on into the
            # next P wave, which would then be hidden as "inside a T wave".
            t1 = min(int(t_end), p + ms(450)) if t_end is not None else p + ms(400)
            if i + 1 < len(peaks):
                t1 = min(t1, q_onset_of(peaks[i + 1]) - ms(20))
            if t1 > z1:
                t_zones.append((z1, t1))

        def in_zone(x, zones):
            return any(z0 <= x <= z1 for z0, z1 in zones)

        # P-wave template from the P waves measure_beat() found in front of
        # their QRS, keeping only ones that share one shape (a T wave or a
        # noise bump picked up instead of a real P won't).
        h = ms(60)
        # A "P" measure_beat() reports inside a QRS zone is the QRS's own
        # upstroke, not a P wave: in complete block with a wide ventricular
        # escape the real P often lies outside the 50-400 ms search window,
        # which then falls back to the QRS edge -- seen: those false seeds,
        # all one shape, outvoted the real P waves and built a QRS-onset
        # template that rejected every real P.
        seeds = [
            b for b in beats.values()
            if b is not None and b.get('p_present') and b.get('p_peak') is not None
            and b.get('p_onset') is not None and h <= int(b['p_peak']) < n - h
            and not in_zone(int(b['p_peak']), qrs_zones)
        ]
        if len(seeds) < 2:
            return None
        shapes = np.array([sig[int(b['p_peak']) - h:int(b['p_peak']) + h + 1] for b in seeds])
        shapes = shapes - shapes.mean(axis=1, keepdims=True)
        median_shape = np.median(shapes, axis=0)
        if not np.any(median_shape):
            return None
        keep = [
            i for i, s in enumerate(shapes)
            if np.any(s) and float(np.corrcoef(s, median_shape)[0, 1]) >= 0.7
        ]
        if len(keep) < 2:
            return None
        template = np.mean(shapes[keep], axis=0)
        template_amp = float(np.ptp(template))
        if template_amp <= 0:
            return None
        onset_offset = int(np.median([int(seeds[i]['p_peak']) - int(seeds[i]['p_onset']) for i in keep]))

        # Normalized cross-correlation of the template along the window.
        from numpy.lib.stride_tricks import sliding_window_view
        if n < 2 * h + 3:
            return None
        win = sliding_window_view(sig, 2 * h + 1)
        win_c = win - win.mean(axis=1, keepdims=True)
        t_c = template - template.mean()
        denom = np.linalg.norm(win_c, axis=1) * float(np.linalg.norm(t_c))
        ncc = (win_c @ t_c) / np.maximum(denom, 1e-12)
        win_amp = np.ptp(win, axis=1)
        hits, _ = find_peaks(ncc, height=0.6, distance=max(1, ms(200)))
        candidates = []  # (center sample, ncc, in_t_zone)
        for k in hits:
            c = int(k) + h
            if not (0.5 * template_amp <= float(win_amp[k]) <= 2.0 * template_amp):
                continue
            if in_zone(c, qrs_zones):
                continue
            # Before the first QRS of the read there is no beat to place a T
            # wave against, so a wave there can't be told apart from a P.
            if c < peaks[0] - ms(500):
                continue
            in_t = in_zone(c, t_zones)
            # Outside a T wave a real P matches its own template closely; inside
            # one the T wave distorts it, so a weaker match is accepted there,
            # but only where the P-P rhythm says a P wave is due (chain below).
            if not in_t and float(ncc[k]) < 0.8:
                continue
            candidates.append((c, float(ncc[k]), in_t))
        clear = [c for c in candidates if not c[2]]
        if len(clear) < 3:
            return None

        # Steady atrial (P-P) interval: the shortest common spacing between
        # clearly visible P waves (longer spacings are a P hidden in between).
        diffs = np.diff([c[0] for c in clear]).astype(float)
        base = float(np.percentile(diffs, 25))
        pp = float(np.median(diffs[diffs <= 1.3 * base]))
        if not (ms(300) <= pp <= ms(2000)):
            return None
        tol = max(ms(60), 0.12 * pp)

        judge_lo = core[0] - ms(600)
        judge_hi = core[-1]

        # Walk the atrial rhythm out from one clear P wave in both directions,
        # one P-P interval at a time.
        def walk_chain(anchor):
            chain = {anchor: False}  # sample -> hidden
            used = {anchor}
            mismatches = 0
            # Mismatches right after a hidden P whose real P sits within two
            # P-P tolerances (see hidden_drift_mismatches below). Counted
            # only; the walk itself is unchanged.
            hidden_drift = 0
            for direction in (1, -1):
                t = anchor
                while True:
                    exp = t + direction * pp
                    if exp < h or exp > n - h - 1:
                        break
                    near = [c for c in candidates if abs(c[0] - exp) <= tol]
                    if near:
                        best = max(near, key=lambda c: c[1])
                        chain[best[0]] = False
                        used.add(best[0])
                        t = best[0]
                        continue
                    e = int(round(exp))
                    if in_zone(e, qrs_zones) or in_zone(e, t_zones):
                        chain[e] = True
                        t = e
                        continue
                    # Clean baseline exactly where a P wave was due, and none there.
                    if judge_lo <= e <= judge_hi:
                        mismatches += 1
                        if chain.get(t) and any(abs(c[0] - exp) <= 2 * tol for c in clear):
                            hidden_drift += 1
                    ahead = [c[0] for c in clear if (c[0] - exp) * direction > 0]
                    if not ahead:
                        break
                    t = ahead[0] if direction > 0 else ahead[-1]
                    chain[t] = False
                    used.add(t)
            extras = [c for c in clear if c[0] not in used and judge_lo <= c[0] <= judge_hi]
            walk_chain.hidden_drift = hidden_drift
            return chain, mismatches, extras

        # A T wave can match the P template too (both are smooth humps), and
        # T waves are just as regular, so a chain started on one would follow
        # T waves throughout. The real atrial rhythm is the chain that
        # accounts for the most clearly visible P waves.
        best_chain = None
        for anchor, _, _ in clear:
            chain, mismatches, extras = walk_chain(anchor)
            key = (sum(1 for c in clear if c[0] in chain), -mismatches, -sum(chain.values()))
            if best_chain is None or key > best_chain[0]:
                best_chain = (key, chain, mismatches, extras, walk_chain.hidden_drift)
        _, chain, mismatches, extras, hidden_drift_mismatches = best_chain

        chain_times = sorted(chain)
        judged = [t for t in chain_times if judge_lo <= t <= judge_hi]
        if len(judged) < 3:
            return None
        judged_pp = np.diff(judged).astype(float)
        pp_cv = float(np.std(judged_pp) / np.mean(judged_pp)) if judged_pp.size else 1.0
        hidden_count = sum(1 for t in judged if chain[t])
        atrial_regular = (
            mismatches == 0 and not extras and pp_cv <= 0.12
            and hidden_count <= 0.5 * len(judged)
        )
        # Third-degree AV block only: a P wave that falls inside a QRS can't be
        # seen, so the chain places it at its expected time -- a guess. The
        # next P is then judged from that guess with one interval's tolerance,
        # but the uncertainty spans two intervals; seen: P-P stretching ~120 ms
        # around the QRS that hides a P (tolerance 90 ms) in complete block,
        # one mismatch per window, and no window passed. Here such a mismatch
        # is forgiven when the real P is within two tolerances. atrial_regular
        # itself, and every rule that reads it, is unchanged.
        atrial_regular_complete_block = atrial_regular or (
            mismatches == hidden_drift_mismatches and not extras and pp_cv <= 0.12
            and hidden_count <= 0.5 * len(judged)
        )
        # AFib/Flutter veto only (see the call site): the clearly visible P
        # waves, in time order, form ONE steady P-P train with at most one
        # phase reset. Gaps of 2-3 P-P intervals (a P hidden in a QRS) are
        # part of the train. Seen: at a Mobitz I -> First-degree change the
        # P timing reset once (gap 1.23 s in a 750 ms train), the chain walk
        # left the P waves before the reset as "extras", atrial_regular went
        # false and the window read as Atrial Fibrillation although every P
        # sat exactly 750 ms from the last. Fibrillatory waves don't form
        # such a train. atrial_regular itself, and every rule reading it, is
        # unchanged.
        clear_times = sorted(c[0] for c in clear)
        phase_resets = sum(
            1 for d in np.diff(clear_times)
            if not any(abs(d - m * pp) <= tol for m in (1, 2, 3))
        )
        atrial_regular_for_afib_veto = atrial_regular or (
            len(clear_times) >= 6 and phase_resets <= 1
            and hidden_count <= 0.5 * len(judged)
        )

        # PR is measured between the two sharpest landmarks, P peak and R
        # peak, then shifted by this window's typical P-onset-to-P-peak and
        # Q-onset-to-R-peak distances. Each beat's own measured Q onset moves
        # by ~20 ms from beat to beat, which alone would read as a changing
        # PR and break the fixed-PR (Mobitz II) / lengthening-PR (Mobitz I)
        # comparisons below.
        q_to_r = int(np.median([
            p - int(beats[p]['q_onset']) for p in core
            if beats[p].get('qrs_bounds_resolved') and beats[p].get('q_onset') is not None
        ] or [ms(40)]))

        # Conduction: a P wave conducts to the first QRS after it when that
        # QRS starts 80-450 ms after the P onset and before the next P wave.
        events = []
        for t in judged:
            idx = chain_times.index(t)
            next_p = chain_times[idx + 1] if idx + 1 < len(chain_times) else None
            onset = t - onset_offset
            following = [p for p in peaks if p - q_to_r > t]
            long_pr_qrs, long_pr_ms = None, None
            if not following:
                conducted, qrs, pr_ms = None, None, None  # can't tell yet
            else:
                p_r = following[0]
                q_on = p_r - q_to_r
                ok = ms(80) <= q_on - onset <= ms(450) and (next_p is None or q_on < next_p - onset_offset)
                conducted = bool(ok) and not in_zone(t, qrs_zones)
                qrs = p_r if conducted else None
                pr_ms = (q_on - onset) * 1000.0 / fs if conducted and not chain[t] else None
                # Mobitz I only (see _is_mobitz_i_window): a visible P whose
                # QRS follows 450-600 ms later, still before the next P wave.
                # 'conducted' above is unchanged for every other rule.
                if (not conducted and not chain[t] and not in_zone(t, qrs_zones)
                        and ms(450) < q_on - onset <= ms(600)
                        and (next_p is None or q_on < next_p - onset_offset)):
                    long_pr_qrs, long_pr_ms = p_r, (q_on - onset) * 1000.0 / fs
            events.append({
                't': t, 'hidden': chain[t], 'conducted': conducted,
                'qrs': qrs, 'pr_ms': pr_ms,
                'mobitz_i_long_pr_qrs': long_pr_qrs, 'mobitz_i_long_pr_ms': long_pr_ms,
            })

        qrs_pr = {e['qrs']: e['pr_ms'] for e in events if e['conducted'] and e['qrs'] is not None}
        # For each QRS: time from the nearest P onset before it, whatever that
        # P's relation to it -- fixed in any conducted rhythm, scattered when
        # atria and ventricles beat independently (complete block).
        nearest_p_to_qrs = []
        for p in core:
            q_on = p - q_to_r
            before = [t for t in chain_times if t - onset_offset < q_on - ms(20)]
            nearest_p_to_qrs.append((q_on - (before[-1] - onset_offset)) * 1000.0 / fs if before else None)

        return {
            'fs': fs,
            'qrs': core,
            'rr_ms': np.diff(core).astype(float) * 1000.0 / fs,
            'events': events,
            'qrs_pr': qrs_pr,
            'pp_ms': pp * 1000.0 / fs,
            'atrial_regular': atrial_regular,
            'atrial_regular_complete_block': atrial_regular_complete_block,
            'atrial_regular_for_afib_veto': atrial_regular_for_afib_veto,
            # True when the context came through the original resolved-QRS
            # gate (not the QRS-shape one); the AFib/Flutter veto uses only
            # those, exactly as before the shape gate was added.
            'qrs_resolved_majority': qrs_resolved_majority,
            'nearest_p_to_qrs_ms': nearest_p_to_qrs,
        }

    # First-degree AV Block: every P wave conducts, just slowly.
    #   1. regular atrial rhythm, every P wave visible and conducted (1:1, no
    #      blocked P -- that would be 2nd/3rd degree);
    #   2. >= 5 QRS, >= 80% of them with a measured PR;
    #   3. median PR > 200 ms (and <= 450 ms), > 200 ms on >= 80% of beats;
    #   4. PR constant: std <= 25 ms (lengthening PR is Mobitz I, scattered PR
    #      is complete block);
    #   5. regular RR, none >= 1.5x the median (no dropped beat), HR < 100
    #      (above that the P wave sits in the T wave and PR isn't reliable).
    # Returns the (first, last) R-peak sample of the window's beats, or None.
    def _is_first_degree_av_block_window(ctx):
        if not ctx['atrial_regular']:
            return None
        qrs = ctx['qrs']
        if len(qrs) < 5:
            return None
        rr = ctx['rr_ms']
        median_rr = float(np.median(rr))
        if median_rr <= 0 or 60000.0 / median_rr >= 100.0:
            return None
        if float(np.std(rr) / np.mean(rr)) > 0.15 or float(np.max(rr)) >= 1.5 * median_rr:
            return None
        events = ctx['events']
        if any(e['conducted'] is False or e['hidden'] for e in events):
            return None
        prs = [ctx['qrs_pr'].get(q) for q in qrs]
        prs = [v for v in prs if v is not None]
        if len(prs) < max(4, int(np.ceil(0.8 * len(qrs)))):
            return None
        median_pr = float(np.median(prs))
        if not (200.0 < median_pr <= 450.0):
            return None
        if sum(1 for v in prs if v > 200.0) < int(np.ceil(0.8 * len(prs))):
            return None
        if float(np.std(prs)) > 25.0:
            return None
        return qrs[0], qrs[-1]

    def _blocked_p_cycles(ctx):
        """Each blocked (non-conducted) P wave with the conducted beats right
        before it and the first conducted beat after it.

        Returns None when the window has a pattern neither Mobitz rule covers:
        a hidden blocked P (can't be confirmed), two blocked P waves in a row
        (high-grade block), or more blocked than conducted P waves.
        """
        events = ctx['events']
        blocked = [i for i, e in enumerate(events) if e['conducted'] is False]
        if not blocked:
            return None
        conducted = sum(1 for e in events if e['conducted'])
        if conducted < len(blocked):
            return None
        cycles = []
        for i in blocked:
            if events[i]['hidden']:
                return None
            if i + 1 < len(events) and events[i + 1]['conducted'] is False:
                return None
            run = []
            j = i - 1
            while j >= 0 and events[j]['conducted'] and events[j]['pr_ms'] is not None:
                run.insert(0, events[j])
                j -= 1
            # Complete = the run starts right after the previous blocked P,
            # not cut off by the window's own start.
            complete = j >= 0 and events[j]['conducted'] is False
            after = None
            if i + 1 < len(events) and events[i + 1]['conducted'] and events[i + 1]['pr_ms'] is not None:
                after = events[i + 1]
            cycles.append((run, complete, after))
        return cycles

    # Second-degree AV Block, Mobitz I (Wenckebach): PR lengthens beat by
    # beat until one P wave is not conducted, then resets.
    #   1. regular atrial rhythm, at least one visible blocked P wave, never
    #      two in a row;
    #   2. before the blocked P: >= 2 conducted beats whose PR never shortens
    #      (by > 10 ms) and grows >= 40 ms in total;
    #   3. after it: the first conducted PR is >= 40 ms shorter (reset);
    #   4. the pause around the blocked P is < 2x the P-P interval.
    # A cycle that clearly fails these (e.g. constant PR = Mobitz II) vetoes
    # the window; a cycle cut off by the window edge is just not judged.
    # Returns (first R of the earliest cycle, first R after the last blocked
    # P), or None.
    # In Wenckebach the PR is longest on the beat right before the dropped P,
    # and can pass 450 ms there; seen: 5:4 cycles with PR 146 -> 206 -> 386
    # -> 456 ms, then a blocked P. With the 450 ms conduction limit the
    # 456 ms beat read as blocked, two "blocked" P waves in a row, and no
    # window passed. This rule alone counts a P conducted at a 450-600 ms PR
    # (QRS still before the next P wave); every other rule sees 'conducted'
    # unchanged.
    def _is_mobitz_i_window(ctx):
        if not ctx['atrial_regular']:
            return None
        events = [
            dict(e, conducted=True, qrs=e['mobitz_i_long_pr_qrs'], pr_ms=e['mobitz_i_long_pr_ms'])
            if e['conducted'] is False and e['mobitz_i_long_pr_ms'] is not None else e
            for e in ctx['events']
        ]
        cycles = _blocked_p_cycles(dict(ctx, events=events))
        if not cycles:
            return None
        fs_ = ctx['fs']
        spans = []
        for run, complete, after in cycles:
            if len(run) < 2 or after is None:
                continue
            prs = [e['pr_ms'] for e in run]
            lengthening = (
                all(d >= -10.0 for d in np.diff(prs))
                and prs[-1] - prs[0] >= 40.0
                and after['pr_ms'] <= prs[-1] - 40.0
            )
            pause_ms = (after['qrs'] - run[-1]['qrs']) * 1000.0 / fs_
            if not (lengthening and pause_ms < 2.0 * ctx['pp_ms']):
                if complete:
                    return None
                continue  # cycle cut off by the window start: not judged
            spans.append((run[0]['qrs'], after['qrs']))
        if not spans:
            return None
        return min(s for s, _ in spans), max(e for _, e in spans)

    # Second-degree AV Block, Mobitz II: PR stays fixed, then a P wave is
    # suddenly not conducted.
    #   1. regular atrial rhythm, at least one visible blocked P wave, never
    #      two in a row;
    #   2. PR of the conducted beats before AND after the blocked P all
    #      within 20 ms of each other (no lengthening, no reset);
    #   3. the pause around the blocked P is 2x the P-P interval (+/-15%).
    # 2:1 block (one conducted beat between blocked P waves) can't show a PR
    # trend; it is reported here when PR is fixed and the 2:1 pattern repeats
    # (>= 2 such cycles in the window).
    # Returns (R before the conducted run's last two beats, first R after the
    # last blocked P), or None.
    def _is_mobitz_ii_window(ctx):
        if not ctx['atrial_regular']:
            return None
        cycles = _blocked_p_cycles(ctx)
        if not cycles:
            return None
        fs_ = ctx['fs']
        spans = []
        two_to_one = []
        for run, complete, after in cycles:
            if not run or after is None:
                continue
            prs = [e['pr_ms'] for e in run] + [after['pr_ms']]
            fixed = max(prs) - min(prs) <= 20.0
            pause_ms = (after['qrs'] - run[-1]['qrs']) * 1000.0 / fs_
            doubled = 1.7 * ctx['pp_ms'] <= pause_ms <= 2.3 * ctx['pp_ms']
            if not (fixed and doubled):
                if len(run) >= 2 and complete:
                    return None
                continue
            span = (run[max(0, len(run) - 2)]['qrs'], after['qrs'])
            (spans if len(run) >= 2 else two_to_one).append(span)
        if len(two_to_one) >= 2:
            spans.extend(two_to_one)
        if not spans:
            return None
        return min(s for s, _ in spans), max(e for _, e in spans)

    # Third-degree (complete) AV Block: atria and ventricles beat
    # independently -- no P wave is conducted, a slow escape rhythm drives
    # the ventricles.
    #   1. regular atrial rhythm with >= 4 visible P waves;
    #   2. regular ventricular rhythm (RR CV <= 0.12), >= 3 QRS, rate < 60;
    #   3. atrial rate >= 1.3x the ventricular rate;
    #   4. AV dissociation: the time from the nearest P wave to each QRS is
    #      scattered (std > 40 ms) -- in 2:1 block it stays fixed.
    # Mobitz I can also scatter that time, but its ventricular rhythm is
    # grouped/irregular, so gate 2 keeps it out.
    # Returns the (first, last) R-peak sample of the window's beats, or None.
    def _is_third_degree_av_block_window(ctx):
        if not ctx['atrial_regular_complete_block']:
            return None
        qrs = ctx['qrs']
        if len(qrs) < 3:
            return None
        if sum(1 for e in ctx['events'] if not e['hidden']) < 4:
            return None
        rr = ctx['rr_ms']
        median_rr = float(np.median(rr))
        if median_rr <= 0:
            return None
        ventricular_rate = 60000.0 / median_rr
        if ventricular_rate >= 60.0 or float(np.std(rr) / np.mean(rr)) > 0.12:
            return None
        if 60000.0 / ctx['pp_ms'] < 1.3 * ventricular_rate:
            return None
        intervals = [v for v in ctx['nearest_p_to_qrs_ms'] if v is not None]
        if len(intervals) < 3 or float(np.std(intervals)) <= 40.0:
            return None
        return qrs[0], qrs[-1]

    def _classify_av_block_window(ctx):
        """Return (label, (start_sample, end_sample)) for the most severe AV
        block rule this window's conduction context satisfies, or None."""
        for label, rule in (
            (_AV_THIRD, _is_third_degree_av_block_window),
            (_AV_MOBITZ_II, _is_mobitz_ii_window),
            (_AV_MOBITZ_I, _is_mobitz_i_window),
            (_AV_FIRST, _is_first_degree_av_block_window),
        ):
            span = rule(ctx)
            if span is not None:
                return label, span
        return None

    # ------------------------------------------------------------------
    # Bundle branch blocks: Complete Right Bundle Branch Block, Incomplete Right Bundle Branch Block, Complete Left Bundle Branch Block,
    # Incomplete Left Bundle Branch Block. A bundle branch block is a QRS-shape finding that sits
    # ON TOP of the rhythm (sinus rhythm/bradycardia/tachycardia, an AV block
    # ...), so it is a separate layer: these rules run on every window on
    # their own, never enter the rhythm priority chain above, never trim or
    # get trimmed by a rhythm segment, and never suppress Sinus Bradycardia/
    # Tachycardia. The only exception: no Bundle Branch Block is drawn over Ventricular
    # Tachycardia/Fibrillation, where the wide QRS is ventricular in origin.
    #
    # Every rule uses the same measure_beat() P / PR / QRS measurements as the
    # rest of this module:
    #   - P/PR (Lead II, strict, as the sinus-conduction gate): >= 75% of
    #     beats with a P wave and a PR, >= 75% of PRs within 25 ms of their
    #     median -- a conducted supraventricular rhythm, which Bundle Branch Block requires
    #     (rules out VT, VFib, AFib, junctional/escape rhythms);
    #   - QRS shape consistent beat to beat (median correlation to the
    #     window's own QRS template >= 0.9, the SVT rule's test);
    #   - QRS duration: each beat's measure_beat() qrs_ms taken on I, II, aVL,
    #     V1, V2, V5, V6 and the median used (QRS is widest in some leads,
    #     narrowest in others; the median keeps one lead's mis-measure out).
    #     Complete >= 120 ms, incomplete 110-119 ms, < 110 ms none.
    # The QRS boundary itself is not required to be "resolved": a Bundle Branch Block's
    # slurred terminal S / R' often never returns cleanly to baseline.
    # ------------------------------------------------------------------
    _COMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK = 'Complete Right Bundle Branch Block'
    _INCOMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK = 'Incomplete Right Bundle Branch Block'
    _COMPLETE_LEFT_BUNDLE_BRANCH_BLOCK = 'Complete Left Bundle Branch Block'
    _INCOMPLETE_LEFT_BUNDLE_BRANCH_BLOCK = 'Incomplete Left Bundle Branch Block'
    _BUNDLE_BRANCH_BLOCK_COLORS = {
        _COMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK: '#E0E0E0',   # light gray
        _INCOMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK: '#9E9E9E',   # mid gray
        _COMPLETE_LEFT_BUNDLE_BRANCH_BLOCK: '#1ABC9C',   # strong teal
        _INCOMPLETE_LEFT_BUNDLE_BRANCH_BLOCK: '#76D7C4',   # light teal
    }
    # Complete outranks incomplete where two Bundle Branch Block regions overlap.
    _BUNDLE_BRANCH_BLOCK_LABELS = (_COMPLETE_LEFT_BUNDLE_BRANCH_BLOCK, _COMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK, _INCOMPLETE_LEFT_BUNDLE_BRANCH_BLOCK, _INCOMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK)
    _BUNDLE_BRANCH_BLOCK_MERGE_GAP_SEC = 2.5

    def _lead_index(*names):
        for name in names:
            if name in lead_names:
                return lead_names.index(name)
        return None

    _bundle_branch_block_v1_idx = _lead_index('V1', 'V2')
    _bundle_branch_block_lateral_idx = [i for i in (_lead_index('I'), _lead_index('V6', 'V5')) if i is not None]
    _bundle_branch_block_width_idx = [i for i in (_lead_index(n) for n in ('I', 'II', 'aVL', 'V1', 'V2', 'V5', 'V6')) if i is not None]

    def _longest_run_ms(mask, fs_):
        best = run = 0
        for v in mask:
            run = run + 1 if v else 0
            best = max(best, run)
        return best * 1000.0 / fs_

    def _v1_bundle_branch_block_shape(x, fs_):
        """V1 QRS (baseline-subtracted). Returns (right_bundle_branch_block_like, left_bundle_branch_block_like).

        Right Bundle Branch Block: rsR' (two positive waves with a dip between) or a terminal
        positive (R') deflection -- the right ventricle activates last and
        its activation points toward V1.
        Left Bundle Branch Block: QRS dominantly negative (QS / rS) -- septal and left
        ventricular activation point away from V1.
        """
        amp = float(np.ptp(x))
        if x.size < 5 or amp <= 0:
            return False, False
        peaks, _ = find_peaks(x, prominence=0.15 * amp)
        peaks = [int(p) for p in peaks if x[p] > 0.1 * amp]
        rsr = any(
            b - a >= int(0.03 * fs_)
            and float(np.min(x[a:b + 1])) < 0.6 * min(float(x[a]), float(x[b]))
            for a, b in zip(peaks, peaks[1:])
        )
        terminal = x[x.size // 2:]
        terminal_r = (
            float(np.clip(terminal, 0, None).sum()) > float(np.clip(-terminal, 0, None).sum())
            and int(np.argmax(x)) >= x.size // 2
        )
        positive = float(np.clip(x, 0, None).sum())
        negative = float(np.clip(-x, 0, None).sum())
        right_bundle_branch_block = rsr or terminal_r
        left_bundle_branch_block = (not rsr) and negative >= 2.0 * positive
        return right_bundle_branch_block, left_bundle_branch_block

    def _lateral_bundle_branch_block_shape(x, fs_):
        """Lead I / V6 QRS (baseline-subtracted). Returns (wide_s, broad_r).

        Right Bundle Branch Block: wide, slurred terminal S (>= 40 ms below baseline after the R).
        Left Bundle Branch Block: broad (>= 60 ms) dominantly positive R with no septal q before
        it and no S after it.
        """
        amp = float(np.ptp(x))
        if x.size < 5 or amp <= 0:
            return False, False
        thr = 0.10 * amp
        peak = int(np.argmax(x))
        after = x[peak:]
        wide_s = (
            float(-np.min(after)) >= 0.15 * amp
            and _longest_run_ms(after < -thr, fs_) >= 40.0
        )
        positive = float(np.clip(x, 0, None).sum())
        negative = float(np.clip(-x, 0, None).sum())
        septal_q = peak > 0 and float(np.min(x[:peak])) < -thr
        broad_r = (
            positive >= 3.0 * negative
            and _longest_run_ms(x > thr, fs_) >= 60.0
            and not septal_q
            and float(-np.min(after)) < 0.15 * amp
        )
        return wide_s, broad_r

    def _bundle_branch_block_window_beats(padded_array, core_r_peaks, best_signal, fs_):
        """Shared P/PR/QRS gate for the four Bundle Branch Block rules. Returns
        (median QRS ms, [(right_bundle_branch_block_like, left_bundle_branch_block_like) per beat]) or None."""
        core = sorted(int(p) for p in core_r_peaks)
        if len(core) < 5 or _bundle_branch_block_v1_idx is None or not _bundle_branch_block_lateral_idx \
                or conduction_lead_idx >= padded_array.shape[0]:
            return None
        # Same filter the AFib/Flutter/AV-block P-wave rules already use on
        # this exact signal (0.5-40 Hz bandpass + 50 Hz notch) -- this gate
        # alone still read raw samples. Mains hum on the chest leads (seen:
        # V1/V5/V6 2-4x noisier on one recording than another with the same
        # LBBB) sits inside the 10% slope threshold the QRS onset/offset
        # search uses, so on a noisy recording the onset was never found on
        # 50-90% of beats and the window failed outright; on a quieter one
        # the same LBBB passed. Filtered, QRS onset/offset were found on
        # every beat on both, width consistent within a few ms either way.
        if _atrial_preproc is not None:
            try:
                padded_array = np.vstack([
                    np.asarray(_atrial_preproc.process(np.asarray(lead, dtype=float)), dtype=float)
                    for lead in padded_array
                ])
            except Exception:
                pass
        n = padded_array.shape[1]
        half_sec = int(fs_)
        required = max(3, int(np.ceil(0.75 * len(core))))

        # P / PR on Lead II (strict, exactly as the sinus-conduction gate).
        # The beat is measured at Lead II's own dominant QRS deflection
        # (largest excursion 40 ms before to 100 ms after the detected beat)
        # rather than at the detected R-peak itself: R-peaks are merged from
        # several leads, and in a wide Bundle Branch Block QRS the earliest lead's peak sits
        # near the QRS onset (seen ~70 ms before Lead II's R). The P-wave
        # search is positioned relative to that point, so measured there it
        # found no P on a complete Left Bundle Branch Block beat. On normal-width beats both
        # points coincide and PR comes out identical.
        lead_ii = np.asarray(padded_array[conduction_lead_idx], dtype=float)
        prs, with_p = [], 0
        for k, r in enumerate(core):
            lo, hi = max(0, r - half_sec), min(n, r + half_sec)
            prev_rr = (r - core[k - 1]) * 1000.0 / fs_ if k else None
            seg_ii = lead_ii[lo:hi]
            a_lo = max(0, r - lo - int(0.04 * fs_))
            a_hi = min(seg_ii.size, r - lo + int(0.10 * fs_))
            if a_hi <= a_lo:
                continue
            anchor = a_lo + int(np.argmax(np.abs(seg_ii[a_lo:a_hi] - np.median(seg_ii))))
            b = measure_beat(seg_ii, anchor, fs_, strict_pr=True, prev_rr_ms=prev_rr)
            # Second try, only where the strict search found no PR: the strict
            # search takes the candidate wave CLOSEST to the QRS and gives up
            # if that one is too small, so a little PR-segment noise just
            # before the QRS hides a clear P wave further back (seen: P ~170
            # at R-130 ms, a -10..-20 noise dip at R-80 ms chosen instead, PR
            # missing on 9 of 20 beats and every BBB window failing). The
            # non-strict search takes the most PROMINENT wave instead -- the
            # real P. Only at <= 120 bpm (RR >= 500 ms): faster, the previous
            # beat's T wave sits close enough before the QRS to be taken for
            # a P. The PR-consistency check below still has to pass.
            if (b is None or b.get('pr_ms') is None) and prev_rr is not None and prev_rr >= 500.0:
                b_relaxed = measure_beat(seg_ii, anchor, fs_, strict_pr=False, prev_rr_ms=prev_rr)
                if b_relaxed is not None and b_relaxed.get('pr_ms') is not None:
                    b = b_relaxed
            if b and b.get('p_present') and b.get('p_onset') is not None:
                with_p += 1
                if b.get('pr_ms') is not None:
                    prs.append(float(b['pr_ms']))
        if with_p < required or len(prs) < required:
            return None
        pr_median = float(np.median(prs))
        if sum(1 for v in prs if abs(v - pr_median) <= 25.0) < required:
            return None

        # One QRS shape throughout (SVT rule's template test). Each beat's
        # snippet is centred on the best lead's own dominant QRS deflection
        # (largest excursion 40 ms before to 100 ms after the detected beat,
        # the same anchoring as the P / PR measurement above), not on the
        # detected beat itself: beat positions are merged from several leads,
        # and in a wide bundle branch block QRS the lead that is found first
        # changes from beat to beat, so the positions jumped by ~20 ms (seen:
        # dominant deflection +30 ms after some beats, +50 ms after others).
        # With a +/-50 ms snippet that misalignment alone dropped identical
        # LBBB beats to correlation 0.74-0.92 and two windows failed, leaving a
        # 5 s gap inside one LBBB episode.
        h = int(0.05 * fs_)
        median_best = float(np.median(best_signal)) if best_signal.size else 0.0
        aligned = []
        for r in core:
            a_lo, a_hi = max(0, r - int(0.04 * fs_)), min(best_signal.size, r + int(0.10 * fs_))
            if a_hi <= a_lo:
                continue
            aligned.append(a_lo + int(np.argmax(np.abs(best_signal[a_lo:a_hi] - median_best))))
        shapes = [best_signal[r - h:r + h] for r in aligned if r - h >= 0 and r + h <= best_signal.size]
        if len(shapes) < required:
            return None
        shapes = np.array([s - np.mean(s) for s in shapes])
        template = np.median(shapes, axis=0)
        if not np.any(template):
            return None
        corrs = [float(np.corrcoef(s, template)[0, 1]) for s in shapes if np.any(s)]
        if not corrs or float(np.median(corrs)) < 0.9:
            return None

        # QRS duration and shape per beat. measure_beat() still decides that
        # each lead holds a real QRS (40-200 ms, as everywhere in this
        # module), but the duration is the GLOBAL QRS across leads: from the
        # first lead's activation to the last lead's, found where the leads'
        # combined slope rises above / falls back below 10% of its peak for
        # 16 ms. measure_beat()'s own J point stops at the first flat point
        # within 15% of baseline, which cuts off the slow, slurred terminal S
        # / R' that defines a bundle branch block -- measured on synthetic
        # beats: true ~150 ms Right Bundle Branch Block read 116 ms by measure_beat, 162 ms here;
        # normal beats on real recordings 84-86 ms here, 90-94 ms there.
        widths, shapes_per_beat, shape_peaks = [], [], []
        quiet = max(1, int(0.016 * fs_))
        for r in core:
            lo, hi = max(0, r - half_sec), min(n, r + half_sec)
            measured = {}
            for li in set(_bundle_branch_block_width_idx) | {_bundle_branch_block_v1_idx} | set(_bundle_branch_block_lateral_idx):
                seg = np.asarray(padded_array[li][lo:hi], dtype=float)
                # Same anchoring as Lead II above: this lead's own dominant
                # QRS deflection near the detected beat.
                a_lo = max(0, r - lo - int(0.04 * fs_))
                a_hi = min(seg.size, r - lo + int(0.10 * fs_))
                if a_hi <= a_lo:
                    continue
                anchor = a_lo + int(np.argmax(np.abs(seg[a_lo:a_hi] - np.median(seg))))
                b = measure_beat(seg, anchor, fs_)
                if b is not None and 40.0 <= float(b.get('qrs_ms') or 0.0) <= 200.0:
                    measured[li] = seg
            width_leads = [li for li in _bundle_branch_block_width_idx if li in measured]
            if len(width_leads) < 3 or _bundle_branch_block_v1_idx not in measured:
                continue

            # Combined slope of the width leads, each scaled by its own QRS
            # size; its peak is searched 80 ms before to 120 ms after the
            # detected beat, onset/offset within 150 ms before / 200 ms after.
            r_rel = r - lo
            w_lo, w_hi = r_rel - int(0.15 * fs_), r_rel + int(0.20 * fs_)
            if w_lo < 0 or w_hi > hi - lo:
                continue
            velocity = np.zeros(w_hi - w_lo)
            for li in width_leads:
                x = measured[li][w_lo:w_hi]
                if x.size >= 11:
                    x = savgol_filter(x, 11, 3)
                size = float(np.ptp(x[int(0.07 * fs_):int(0.27 * fs_)])) or 1.0
                velocity += np.abs(np.gradient(x)) / size
            p_lo, p_hi = int(0.07 * fs_), int(0.27 * fs_)
            peak = p_lo + int(np.argmax(velocity[p_lo:p_hi]))
            threshold = 0.10 * float(velocity[peak])
            onset = offset = None
            run = 0
            for k in range(peak, -1, -1):
                run = run + 1 if velocity[k] < threshold else 0
                if run >= quiet:
                    onset = k + quiet - 1
                    break
            run = 0
            for k in range(peak, velocity.size):
                run = run + 1 if velocity[k] < threshold else 0
                if run >= quiet:
                    offset = k - quiet + 1
                    break
            # Left bundle branch block: a large discordant ST-T starts right
            # where the QRS ends, so the combined slope never settles under
            # 10% of its peak (measured: it dips to 11-15% between QRS and T
            # and rises again) and no QRS end was found -- every beat was
            # skipped. Only then, the QRS end is the first point the slope
            # stays under 20% of its peak for the same 16 ms: the dip
            # between QRS and T. Beats whose end was found above are
            # unaffected.
            if offset is None:
                fallback_threshold = 0.20 * float(velocity[peak])
                run = 0
                for k in range(peak, velocity.size):
                    run = run + 1 if velocity[k] < fallback_threshold else 0
                    if run >= quiet:
                        offset = k - quiet + 1
                        break
            if onset is None or offset is None or offset <= onset:
                continue
            widths.append((offset - onset) * 1000.0 / fs_)
            onset += w_lo
            offset += w_lo

            def qrs_of(li):
                seg = measured[li]
                # Each lead's own baseline: the PR segment just before QRS onset.
                base = float(np.median(seg[max(0, onset - int(0.06 * fs_)):max(1, onset - int(0.01 * fs_))]))
                return seg[onset:offset + 1] - base

            v1_right_bundle_branch_block, v1_left_bundle_branch_block = _v1_bundle_branch_block_shape(qrs_of(_bundle_branch_block_v1_idx), fs_)
            lateral = [_lateral_bundle_branch_block_shape(qrs_of(li), fs_) for li in _bundle_branch_block_lateral_idx if li in measured]
            if not lateral:
                continue
            beat_right_bundle_branch_block = v1_right_bundle_branch_block and any(s for s, _ in lateral)
            beat_left_bundle_branch_block = v1_left_bundle_branch_block and all(r_ for _, r_ in lateral)
            shapes_per_beat.append((beat_right_bundle_branch_block, beat_left_bundle_branch_block))
            shape_peaks.append(r)
        if len(widths) < required or len(shapes_per_beat) < required:
            return None
        return float(np.median(widths)), shapes_per_beat, shape_peaks

    def _bundle_branch_block_majority(shapes_per_beat, side):
        need = int(np.ceil(0.75 * len(shapes_per_beat)))
        if side == 'R':
            return sum(1 for r_, l_ in shapes_per_beat if r_ and not l_) >= need
        return sum(1 for r_, l_ in shapes_per_beat if l_ and not r_) >= need

    # Complete Right Bundle Branch Block: QRS >= 120 ms; V1 rsR' / terminal R'; wide slurred S in
    # I or V6; >= 75% of beats show it.
    def _is_complete_right_bundle_branch_block_window(ctx):
        width, shapes_per_beat = ctx[0], ctx[1]
        return width >= 120.0 and _bundle_branch_block_majority(shapes_per_beat, 'R')

    # Incomplete Right Bundle Branch Block: same Right Bundle Branch Block shape, QRS 110-119 ms.
    def _is_incomplete_right_bundle_branch_block_window(ctx):
        width, shapes_per_beat = ctx[0], ctx[1]
        return 110.0 <= width < 120.0 and _bundle_branch_block_majority(shapes_per_beat, 'R')

    # Complete Left Bundle Branch Block: QRS >= 120 ms; V1 dominantly negative (QS/rS); broad
    # monophasic R in I and V6 with no septal q and no S; >= 75% of beats.
    def _is_complete_left_bundle_branch_block_window(ctx):
        width, shapes_per_beat = ctx[0], ctx[1]
        return width >= 120.0 and _bundle_branch_block_majority(shapes_per_beat, 'L')

    # Incomplete Left Bundle Branch Block: same Left Bundle Branch Block shape, QRS 110-119 ms.
    def _is_incomplete_left_bundle_branch_block_window(ctx):
        width, shapes_per_beat = ctx[0], ctx[1]
        return 110.0 <= width < 120.0 and _bundle_branch_block_majority(shapes_per_beat, 'L')

    def _classify_bundle_branch_block_window(padded_array, core_r_peaks, best_signal, fs_, votes_out=None):
        """Return (label, (first_beat, last_beat)) for this window, or None.

        votes_out: if given, every beat this window judged is appended as
        (r_peak_sample, right_pattern, left_pattern) -- also when the window
        gets no label -- for the beat-level edge extension further below.

        The span runs from the first to the last beat in the window whose own
        shape is that block's pattern, not the whole window: a window passes
        with >= 75% such beats, so its first or last beats can still be the
        rhythm before or after it (seen: 2 narrow sinus beats at 10:20:32.69
        and 10:20:34.69 opened a Right Bundle Branch Block segment whose
        first wide beat was 10:20:35.21)."""
        ctx = _bundle_branch_block_window_beats(padded_array, core_r_peaks, best_signal, fs_)
        if ctx is None:
            return None
        if votes_out is not None:
            votes_out.extend((peak, r_, l_) for peak, (r_, l_) in zip(ctx[2], ctx[1]))
        for label, rule in (
            (_COMPLETE_LEFT_BUNDLE_BRANCH_BLOCK, _is_complete_left_bundle_branch_block_window),
            (_COMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK, _is_complete_right_bundle_branch_block_window),
            (_INCOMPLETE_LEFT_BUNDLE_BRANCH_BLOCK, _is_incomplete_left_bundle_branch_block_window),
            (_INCOMPLETE_RIGHT_BUNDLE_BRANCH_BLOCK, _is_incomplete_right_bundle_branch_block_window),
        ):
            if rule(ctx):
                right = 'Right' in label
                matching = [
                    peak for peak, (r_, l_) in zip(ctx[2], ctx[1])
                    if ((r_ and not l_) if right else (l_ and not r_))
                ]
                if not matching:
                    return None
                return label, (min(matching), max(matching))
        return None

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
    # Window starts called VFib only by the coarse-VFib shape test.
    _coarse_vfib_window_starts = set()
    # Ventricular rate (bpm) of each Supraventricular Tachycardia (SVT)
    # window, keyed by id() of its candidate dict -- for the drop rule after
    # the loop only; never part of the output segments.
    _svt_window_rate = {}
    bundle_branch_block_candidates = []
    # Every beat any window judged for a bundle branch block pattern:
    # beat time (s, 2 decimals) -> [(right_pattern, left_pattern), ...].
    bundle_branch_block_beat_votes = {}

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
                # The unpadded core slice, for the signal-quality gate below,
                # which wants exactly this window's own samples -- sliced
                # out of the padded read instead of a second reader call.
                core_start_idx = int(round((current_time - pad_start) * fs))
                core_end_idx = core_start_idx + int(round((window_end - current_time) * fs))
                data_array = padded_array[:, core_start_idx:core_end_idx]
                target_morph = None
                av_span_sec = None
                core_r_peaks = []

                # Skip morphological classification entirely for a window
                # whose own (unpadded) signal doesn't look like a usable
                # ECG -- same SQI/0.6 cutoff used everywhere else in the
                # app (see the comment above _signal_quality near the top
                # of this function). A noisy window simply produces no
                # morphological label here, same as if nothing matched.
                window_is_noisy = False
                if _signal_quality is not None and best_lead_idx < data_array.shape[0]:
                    try:
                        core_signal = np.asarray(data_array[best_lead_idx], dtype=float)
                        if core_signal.size > 0:
                            window_is_noisy = _signal_quality.compute_sqi(core_signal) < _sqi_min
                    except Exception:
                        window_is_noisy = False

                if not window_is_noisy and best_lead_idx < padded_array.shape[0]:
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
                        # Bundle branch block: its own layer, judged on every
                        # (non-noisy) window before the rhythm chain below. Its
                        # only effect on that chain is the VFib veto there.
                        try:
                            window_votes = []
                            bundle_branch_block_result = _classify_bundle_branch_block_window(
                                padded_array, core_r_peaks, padded_signal, fs, votes_out=window_votes)
                            for vote_peak, vote_right, vote_left in window_votes:
                                bundle_branch_block_beat_votes.setdefault(
                                    round(float(pad_start + vote_peak / fs), 2), []
                                ).append((bool(vote_right), bool(vote_left)))
                        except Exception as e:
                            bundle_branch_block_result = None
                            print(f"[Auto Arrhythmia Detect] Bundle Branch Block window error {current_time:.1f}s: {e}")
                        # Resolve the organized-wide-complex overlap first:
                        # the same unresolved QRS flag can otherwise make a
                        # regular monomorphic VT window look like VFib.
                        if _is_ventricular_tachycardia_window(padded_signal, core_r_peaks, fs):
                            # Burnt orange distinguishes organized VT from
                            # VFib red and Supraventricular Tachycardia orange.
                            target_morph = ('Ventricular Tachycardia', '#E67E22')
                        # VFib is not called where this same window is a
                        # proven conducted bundle branch block: P waves with
                        # a consistent PR on >= 75% of beats, one QRS shape,
                        # QRS >= 110 ms and >= 75% RBBB/LBBB morphology
                        # (_classify_bundle_branch_block_window). VFib has no
                        # AV conduction; seen: a 30 -> 80 bpm change into
                        # RBBB read as VFib only because the window's RR
                        # spread was large and the wide QRS never 'resolved'.
                        # _is_vfib_window itself is unchanged; a vetoed
                        # window goes on through the rest of the chain.
                        elif (_is_vfib_window(padded_signal, core_r_peaks, fs)
                              and bundle_branch_block_result is None):
                            target_morph = ('Ventricular Fibrillation', '#FF3333')
                            if _vfib_window_kind['last'] == 'coarse':
                                _coarse_vfib_window_starts.add(current_time)
                        else:
                            # P waves are judged on Lead II (see the
                            # conduction_lead_idx comment above), beats still
                            # come from the same R-peaks.
                            av_ctx = None
                            if conduction_lead_idx < padded_array.shape[0]:
                                av_ctx = _prepare_av_conduction_context(
                                    np.asarray(padded_array[conduction_lead_idx], dtype=float),
                                    all_peaks, core_r_peaks, fs,
                                )
                            atrial_label = _classify_atrial_window(padded_signal, core_r_peaks, fs)
                            # AFib has no discrete P waves and Flutter's atrial
                            # rate is 250-350/min; a steady train of discrete P
                            # waves at <= 200/min is neither. Mobitz I's grouped
                            # beating plus its changing PR otherwise reads as
                            # "irregular RR, no consistent PR" = AFib.
                            if atrial_label and av_ctx is not None and av_ctx['atrial_regular_for_afib_veto'] \
                                    and av_ctx['qrs_resolved_majority']:
                                atrial_label = None
                            if atrial_label:
                                target_morph = (atrial_label, '#FF00FF')
                            elif _is_atrial_tachycardia_window(padded_signal, core_r_peaks, fs):
                                target_morph = ('Supraventricular Tachycardia (SVT)', '#FFA500')
                            elif av_ctx is not None:
                                av_result = _classify_av_block_window(av_ctx)
                                if av_result is not None:
                                    av_label, (av_s, av_e) = av_result
                                    target_morph = (av_label, _AV_COLORS[av_label])
                                    av_span_sec = (pad_start + av_s / fs, pad_start + av_e / fs)

                        if bundle_branch_block_result is not None:
                            # Beat-exact: first to last beat showing the block's
                            # own pattern (see _classify_bundle_branch_block_window).
                            bundle_branch_block_label, (bbb_first, bbb_last) = bundle_branch_block_result
                            b_start = float(pad_start + bbb_first / fs)
                            b_end = float(pad_start + bbb_last / fs)
                            b_s_str, b_e_str = make_time_strs(b_start, b_end)
                            bundle_branch_block_candidates.append({
                                'start_sec': b_start,
                                'end_sec': b_end,
                                'label': bundle_branch_block_label,
                                'color': _BUNDLE_BRANCH_BLOCK_COLORS[bundle_branch_block_label],
                                'start_time_str': b_s_str,
                                'end_time_str': b_e_str,
                                'detected_rhythm': bundle_branch_block_label,
                            })

                if target_morph:
                    # A VT classifier window can include onset-transition
                    # beats before the stable wide-complex run.  Align only
                    # VT's visual/event boundary to that run; all classifier
                    # gates and the other rhythm boundaries remain unchanged.
                    candidate_start = current_time
                    candidate_end = window_end
                    # AV block rules return the exact beats their evidence
                    # spans (e.g. one Wenckebach cycle), not the whole window.
                    if av_span_sec is not None:
                        candidate_start, candidate_end = av_span_sec
                    if target_morph[0] == 'Ventricular Tachycardia':
                        core_start_rel = _ventricular_tachycardia_core_start(
                            core_r_peaks, fs
                        )
                        if core_start_rel is not None:
                            candidate_start = max(
                                current_time,
                                min(window_end, pad_start + core_start_rel),
                            )
                    s_str, e_str = make_time_strs(candidate_start, candidate_end)
                    morphological_candidates.append({
                        'start_sec': candidate_start,
                        'end_sec': candidate_end,
                        'label': target_morph[0],
                        'color': target_morph[1],
                        'start_time_str': s_str,
                        'end_time_str': e_str,
                        'detected_rhythm': target_morph[0]
                    })
                    if target_morph[0] == 'Supraventricular Tachycardia (SVT)' and len(core_r_peaks) >= 2:
                        _svt_rr = float(np.median(np.diff(np.asarray(core_r_peaks, dtype=float)))) / float(fs)
                        if _svt_rr > 0:
                            _svt_window_rate[id(morphological_candidates[-1])] = 60.0 / _svt_rr
        except Exception as e:
            print(f"[Auto Arrhythmia Detect] Window error {current_time:.1f}s: {e}")

        current_time += step_size
        window_count += 1
        if window_count % 10 == 0 and progress_callback:
            progress = int((current_time / total_duration) * 100)
            progress_callback(progress)

    # Coarse-VFib guard: a window called VFib only by the shape test in
    # _is_vfib_window is dropped unless another VFib window (either test)
    # starts within one window of it -- one lone window is not an episode
    # (the missed VFib this test is for ran 16+ windows in a row).
    if _coarse_vfib_window_starts:
        # A VFib candidate starts at its window's start (current_time).
        _vfib_window_starts = [m['start_sec'] for m in morphological_candidates
                               if m['label'] == 'Ventricular Fibrillation']
        morphological_candidates = [
            m for m in morphological_candidates
            if not (m['label'] == 'Ventricular Fibrillation'
                    and m['start_sec'] in _coarse_vfib_window_starts
                    and not any(o != m['start_sec'] and abs(o - m['start_sec']) <= window_size
                                for o in _vfib_window_starts))
        ]

    # AV conduction can't be graded during Atrial Fibrillation/Flutter --
    # there are no discrete P waves. Flutter whose conduction ratio changes
    # (RR 1000 -> 600 -> 400ms) reads as "dropped beats" and analyze_ecg()
    # calls it Second-degree AV Block, which then split real Flutter
    # episodes apart. Drop AV-block windows next to AFib/Flutter windows,
    # before the isolated-window guard below, so they can't make the
    # surrounding Flutter windows look isolated. Only atrial windows with a
    # same-label window nearby count -- a lone AFib window at the edge of a
    # real dropped-beat AV block (dropped by the isolated-window guard
    # anyway) was wiping out the block itself.
    _atrial = [
        m for m in morphological_candidates
        if m['label'] in ('Atrial Fibrillation', 'Atrial Flutter')
    ]
    _atrial_starts = [
        m['start_sec'] for m in _atrial
        if any(o is not m and o['label'] == m['label']
               and abs(o['start_sec'] - m['start_sec']) <= window_size * 2
               for o in _atrial)
    ]
    # Supraventricular Tachycardia (SVT) windows next to AFib/Flutter are dropped the same
    # way: 2:1 Atrial Flutter at 150/min is regular with one F wave visible
    # between beats, which reads exactly like P-on-T (seen: one such window
    # inside a Flutter episode).
    # Only SVT windows at a 2:1-Flutter ventricular rate (atrial 250-350/min
    # halved: ~125-175 bpm) are dropped. A faster one can't be 2:1 Flutter
    # (seen: real SVT at 200 bpm, RR 300 ms, starting right after a 5:1
    # Flutter -- every SVT window within 20 s was dropped and Flutter was
    # stretched 14 s over it). A window whose rate is unknown is dropped as
    # before.
    def _svt_could_be_2to1_flutter(m):
        rate = _svt_window_rate.get(id(m))
        return rate is None or 125.0 <= rate <= 175.0

    # First-degree AV Block is not dropped here: its own rule already needs a
    # regular atrial rhythm with every P wave conducted 1:1, no dropped beat
    # and a constant PR (std <= 25 ms), which Flutter / AFib can't pass --
    # the "dropped beats" misreading this guard is for is a 2nd / 3rd degree
    # one. Seen: First-degree AV Block right after Atrial Flutter (PR 234-240
    # ms, RR 750 ms, windows 235-245 s and 240-250 s) wiped out because the
    # last Flutter windows started 10-15 s earlier.
    morphological_candidates = [
        m for m in morphological_candidates
        if not ((('av block' in m['label'].lower() and m['label'] != _AV_FIRST)
                 or (m['label'] == 'Supraventricular Tachycardia (SVT)' and _svt_could_be_2to1_flutter(m)))
                and any(abs(m['start_sec'] - s) <= window_size * 2 for s in _atrial_starts))
    ]

    # Same for an AV-block window overlapping or touching a Ventricular
    # Fibrillation window: P-to-QRS conduction can't be graded on chaotic
    # beats. A window straddling the edge of VFib picks up the chaotic
    # beats' scattered "PR" values, which analyze_ecg() reads as PR
    # variability -- seen: a clean 30 bpm Sinus Bradycardia window
    # (115-125 s) whose last 3 s were VFib onset, called Third-degree AV
    # Block.
    _vfib_starts = [
        m['start_sec'] for m in morphological_candidates
        if m['label'] == 'Ventricular Fibrillation'
    ]
    morphological_candidates = [
        m for m in morphological_candidates
        if not ('av block' in m['label'].lower()
                and any(abs(m['start_sec'] - s) <= window_size for s in _vfib_starts))
    ]

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
    # Supraventricular Tachycardia (SVT) goes through the same guard: one lone window is
    # not an episode.
    if morphological_candidates:
        _sorted_by_time = sorted(morphological_candidates, key=lambda x: x['start_sec'])
        _confirmed = []
        for _i, _m in enumerate(_sorted_by_time):
            if _m['label'] not in ('Atrial Fibrillation', 'Atrial Flutter', 'Supraventricular Tachycardia (SVT)'):
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
                gap_tolerance = _AV_MERGE_GAP_SEC if m['label'] in _AV_LABELS else window_size * 2
                if last['label'] == m['label'] and m['start_sec'] <= last['end_sec'] + gap_tolerance:
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
        # require_narrow: where a VFib edge borders unlabeled (normal) rhythm,
        # a resolvable QRS boundary alone doesn't mean the beat is organized --
        # wide (>=120ms) chaotic VFib beats frequently still get a resolved
        # boundary, which cut a real VFib tail ~10s short on a test recording
        # (beats stayed 150-160ms wide with wildly irregular RR until the
        # rhythm genuinely narrowed to ~100ms at a steady 1000ms RR). Not
        # applied next to AFib/Flutter, whose own beats measure wide here too.
        def _beat_is_organized(peak_sec, require_narrow=False):
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
            # A "QRS" under 40 ms is not a real QRS (measure_beat itself
            # flags it qrs_out_of_range): the peak sits on a chaotic wave
            # whose top happens to look flat enough for the boundary search
            # to stop right at it. Seen on VFib beats measuring 0-30 ms,
            # two of which (301.26 s and 302.26 s) read as an organized
            # pair and ended a VFib episode 10 s early.
            if (beat.get('qrs_ms') or 0.0) < 40.0:
                return False
            if require_narrow and (beat.get('qrs_ms') or 0.0) >= 120.0:
                return False
            return bool(beat.get('qrs_bounds_resolved'))

        _sorted_peaks = sorted(float(p) for p in all_r_peaks)

        # Pan-Tompkins often finds no peaks at all in the first seconds of
        # VFib, so the first detected beat after the last real beat can sit
        # seconds later with chaos in between (seen: 302.18 s -> 308.86 s,
        # 120.39 s -> 126.06 s, and a 452.1 s -> 465.9 s burst inside Atrial
        # Flutter with no peak at all), leaving that chaos unlabeled. If the
        # beat before such a silent gap is organized and the gap is much
        # longer than that rhythm's own RR, VFib started right after it.
        #
        # A lone beat inside the chaos -- a silent gap both before AND after
        # it -- is not the end of the organized rhythm even when it happens
        # to measure narrow; it is skipped, and the search continues back to
        # the last beat that still belongs to the organized run (seen: last
        # sinus beat 600.23 s, then single peaks at 603.83 s and 607.64 s
        # inside VFib, which pushed VFib's start to 609 s).
        def _silent_gap_start(after_ts, require_narrow):
            prev_peaks = [p for p in _sorted_peaks if p < after_ts]
            nxt = after_ts
            for k in range(len(prev_peaks) - 1, -1, -1):
                cand = prev_peaks[k]
                if after_ts - cand > window_size:
                    return None
                prior_rr = np.diff(prev_peaks[max(0, k - 4):k + 1])
                threshold = max(2.0, 1.5 * float(np.median(prior_rr))) if prior_rr.size else 2.0
                if nxt - cand <= threshold:
                    return None
                isolated = prior_rr.size > 0 and prior_rr[-1] > threshold
                if not isolated:
                    return cand if _beat_is_organized(cand, require_narrow) is True else None
                nxt = cand
            return None

        def _refine_vfib_edge(boundary_sec, edge, require_narrow=False):
            lo, hi = boundary_sec - window_size, boundary_sec + window_size
            nearby = sorted(float(p) for p in all_r_peaks if lo <= p <= hi)
            states = [(p, _beat_is_organized(p, require_narrow)) for p in nearby]
            for i in range(len(states) - 1):
                ts0, o0 = states[i]
                ts1, o1 = states[i + 1]
                if edge == 'start' and o0 is False and o1 is False:
                    gap_start = _silent_gap_start(ts0, require_narrow)
                    return gap_start if gap_start is not None else ts0
                # Two organized beats only mark the end of VFib if they are
                # truly back-to-back. Pan-Tompkins often finds no peaks at
                # all in the chaotic tail of VFib, so a lone, by-chance
                # "organized" VFib beat and the first real beat after it
                # can sit seconds apart with chaos in between (seen: 239.4 s
                # and 243.0 s, 3.6 s apart, which drew AFib starting ~3 s
                # early over still-chaotic waves).
                if edge == 'end' and o0 is True and o1 is True and (ts1 - ts0) <= 2.0:
                    return ts0
            if edge == 'start':
                # No two chaotic beats detected at all -- the whole burst
                # may sit in one silent gap. Use that gap if it spans the
                # window-vote boundary.
                # Also the gap right after the last organized beat before the
                # boundary, even if a stray chaotic beat follows it (seen:
                # last sinus beat 100.44 s, one wide VFib beat 104.45 s, then
                # nothing until past the 105.0 s window edge).
                candidates = [p for p in _sorted_peaks if p > boundary_sec][:1]
                organized_before = [s[0] for s in states if s[0] <= boundary_sec and s[1] is True]
                if organized_before:
                    later = [p for p in _sorted_peaks if p > organized_before[-1]]
                    if later:
                        candidates.append(later[0])
                for after_ts in candidates:
                    gap_start = _silent_gap_start(after_ts, require_narrow)
                    if gap_start is not None and gap_start <= boundary_sec:
                        return gap_start
                # No beat detected at all from the last organized beat up to
                # the window-vote boundary, and the next one more than a window
                # away (seen: last First-degree beat 396.30 s, VFib waves from
                # ~396.6 s, the R-peak pass found nothing until past 410 s, so
                # the gap search above -- capped at one window -- gave up and
                # VFib started at the 400.0 s window edge). That silence,
                # longer than 1.5x the organized rhythm's RR (at least 2 s),
                # is VFib too.
                if organized_before:
                    last_org = organized_before[-1]
                    if not any(last_org < p <= boundary_sec for p in _sorted_peaks):
                        prior = [p for p in _sorted_peaks if p <= last_org][-5:]
                        prior_rr = np.diff(prior)
                        threshold = max(2.0, 1.5 * float(np.median(prior_rr))) if prior_rr.size else 2.0
                        if boundary_sec - last_org > threshold:
                            return last_org
            return boundary_sec

        # VFib right after Ventricular Tachycardia: the start search above
        # counts every wide beat as disorganized (require_narrow), and VT
        # beats are wide by definition, so it pulled VFib's start back a whole
        # window into the still-regular VT run and cut the VT short (seen:
        # identical VT beats, RR 374 ms, shape correlation 1.00, until 174.96 s
        # and ~781 s; VFib drawn from 165.2 s and 770 s, where every window up
        # to 175-185 s / 775-785 s had itself passed the VT rule). VFib's
        # start is not moved before the last beat that still continues that
        # VT run: same RR (within 15% of the VT's own median RR, as the VT
        # rule's stable core) and same QRS shape (correlation >= 0.90 with the
        # VT's median beat). A single extra detection inside the run is
        # skipped when the next beat continues it. Only this one VT -> VFib
        # boundary is affected; it can only give back beats to VT, never take
        # any, and the VT / VFib rules themselves are unchanged.
        def _vt_run_last_beat(vt_region, vfib_start_sec):
            ref_lo = max(float(vt_region['start_sec']), vfib_start_sec - window_size)
            ref = [p for p in _sorted_peaks if ref_lo <= p < vfib_start_sec]
            if len(ref) < 5:
                return None
            ref_rr = np.diff(np.asarray(ref, dtype=float))
            vt_rr = float(np.median(ref_rr))
            if vt_rr <= 0 or float(np.std(ref_rr)) / vt_rr > 0.12:
                return None
            limit = max(float(vt_region['end_sec']), vfib_start_sec) + window_size
            ahead = [p for p in _sorted_peaks if vfib_start_sec <= p <= limit]
            if not ahead:
                return None
            seg_start = max(0.0, ref[0] - 0.5)
            seg_end = min(total_duration, ahead[-1] + 0.5)
            sig = reader.read_range(seg_start, seg_end)
            if sig is None or best_lead_idx >= sig.shape[0]:
                return None
            lead_sig = np.asarray(sig[best_lead_idx], dtype=float)
            try:
                _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
                lead_sig = _filtfilt(_b, _a, lead_sig)
            except Exception:
                pass
            half_w = int(0.12 * fs)

            def _snippet(t):
                i = int(round((t - seg_start) * fs))
                if i - half_w < 0 or i + half_w > len(lead_sig):
                    return None
                s = lead_sig[i - half_w:i + half_w]
                return s - float(np.mean(s))

            ref_snips = [s for s in (_snippet(t) for t in ref) if s is not None]
            if len(ref_snips) < 5:
                return None
            template = np.median(np.asarray(ref_snips), axis=0)
            t_norm = float(np.linalg.norm(template))
            if t_norm <= 0:
                return None

            t_ptp = float(np.ptp(template))

            def _same_shape(t):
                s = _snippet(t)
                if s is None:
                    return False
                # same shape AND about the same size (correlation alone
                # ignores size)
                if not (0.5 * t_ptp <= float(np.ptp(s)) <= 2.0 * t_ptp):
                    return False
                n = float(np.linalg.norm(s))
                return n > 0 and float(np.dot(s, template)) / (n * t_norm) >= 0.90

            def _in_run(prev_t, t):
                return abs((t - prev_t) - vt_rr) <= 0.15 * vt_rr and _same_shape(t)

            last = ref[-1]
            i = 0
            while i < len(ahead):
                t = ahead[i]
                if t <= last:
                    i += 1
                    continue
                if _in_run(last, t):
                    last = t
                    i += 1
                    continue
                # one extra (false) detection inside the run
                if i + 1 < len(ahead) and _in_run(last, ahead[i + 1]):
                    last = ahead[i + 1]
                    i += 2
                    continue
                break
            return last

        for idx, region in enumerate(merged_morph):
            if region['label'] != 'Ventricular Fibrillation':
                continue
            try:
                atrial = ('Atrial Fibrillation', 'Atrial Flutter')
                prev_atrial = idx > 0 and merged_morph[idx - 1]['label'] in atrial and \
                    region['start_sec'] - merged_morph[idx - 1]['end_sec'] <= window_size
                next_atrial = idx + 1 < len(merged_morph) and merged_morph[idx + 1]['label'] in atrial and \
                    merged_morph[idx + 1]['start_sec'] - region['end_sec'] <= window_size
                new_start = _refine_vfib_edge(region['start_sec'], 'start', require_narrow=not prev_atrial)
                if idx > 0 and merged_morph[idx - 1]['label'] == 'Ventricular Tachycardia' \
                        and region['start_sec'] - merged_morph[idx - 1]['end_sec'] <= window_size:
                    vt_last = _vt_run_last_beat(merged_morph[idx - 1], new_start)
                    if vt_last is not None and vt_last > new_start:
                        new_start = vt_last
                new_end = _refine_vfib_edge(region['end_sec'], 'end', require_narrow=not next_atrial)
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

        # Beat-exact START of Ventricular Tachycardia. Its window vote can't
        # see the first VT beats when Pan-Tompkins finds no peak at all in the
        # window holding the onset -- small sinus beats followed by large VT
        # complexes upset its thresholds (seen: window 690-700 s with 0 peaks,
        # VT drawn from 695.36 s while identical VT beats ran from ~691 s).
        # So the onset is searched on the signal itself instead of on R-peaks:
        # the VT's own first confirmed beats give its QRS template and RR, and
        # going backward one RR at a time (+/-15%, the VT rule's stable-core
        # tolerance) the template is looked for in the signal; a match
        # (correlation >= 0.90) is one more VT beat, the first miss ends the
        # search. Everything comes from that VT's own beats, nothing fixed.
        # The search reaches back at most one window. Only the VT start can
        # move, only earlier, and only over beats with this VT's shape and
        # rate; the VT rule itself is unchanged.
        def _vt_run_first_beat(vt_region, floor_sec):
            v_start = float(vt_region['start_sec'])
            ref = [p for p in _sorted_peaks if v_start - 0.05 <= p < v_start + window_size]
            if len(ref) < 5:
                # The whole-recording R-peak pass (60 s chunks) can miss the
                # VT beats too (seen: none at all in 695-705 s while the
                # window pass found 26). Re-detect them on just this VT's
                # first window, as the window vote itself did.
                r_start = max(0.0, v_start - 0.5)
                r_data = reader.read_range(r_start, min(total_duration, v_start + window_size))
                if r_data is None or best_lead_idx >= r_data.shape[0]:
                    return None
                ref = sorted(r_start + float(q) / float(fs)
                             for q in pan_tompkins(np.asarray(r_data[best_lead_idx], dtype=float), fs=fs))
                ref = [p for p in ref if v_start - 0.05 <= p < v_start + window_size]
            if len(ref) < 5:
                return None
            ref_rr = np.diff(np.asarray(ref, dtype=float))
            vt_rr = float(np.median(ref_rr))
            if vt_rr <= 0 or float(np.std(ref_rr)) / vt_rr > 0.12:
                return None
            seg_start = max(0.0, v_start - window_size - 1.0)
            seg_end = min(total_duration, ref[-1] + 0.5)
            sig = reader.read_range(seg_start, seg_end)
            if sig is None or best_lead_idx >= sig.shape[0]:
                return None
            lead_sig = np.asarray(sig[best_lead_idx], dtype=float)
            try:
                _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
                lead_sig = _filtfilt(_b, _a, lead_sig)
            except Exception:
                pass
            half_w = int(0.12 * fs)

            def _snippet_at(i):
                if i - half_w < 0 or i + half_w > len(lead_sig):
                    return None
                s = lead_sig[i - half_w:i + half_w]
                return s - float(np.mean(s))

            ref_snips = [s for s in (_snippet_at(int(round((t - seg_start) * fs))) for t in ref) if s is not None]
            if len(ref_snips) < 5:
                return None
            template = np.median(np.asarray(ref_snips), axis=0)
            t_norm = float(np.linalg.norm(template))
            if t_norm <= 0:
                return None

            first = ref[0]
            tol = 0.15 * vt_rr
            t_ptp = float(np.ptp(template))
            while first - vt_rr >= max(v_start - window_size, floor_sec):
                lo_i = int(round((first - vt_rr - tol - seg_start) * fs))
                hi_i = int(round((first - vt_rr + tol - seg_start) * fs))
                best_c, best_i = -1.0, None
                for i in range(max(0, lo_i), max(0, hi_i) + 1):
                    s = _snippet_at(i)
                    if s is None:
                        continue
                    n = float(np.linalg.norm(s))
                    if n <= 0:
                        continue
                    # Correlation ignores size: the previous beat's T wave is
                    # a smooth hump too and matched a sine-like VT complex at
                    # 0.96 (seen: 690.17 s, ~7x smaller). A VT beat is about
                    # the VT's own size.
                    if not (0.5 * t_ptp <= float(np.ptp(s)) <= 2.0 * t_ptp):
                        continue
                    c = float(np.dot(s, template)) / (n * t_norm)
                    if c > best_c:
                        best_c, best_i = c, i
                if best_i is None or best_c < 0.90:
                    break
                first = seg_start + best_i / float(fs)
            return first

        for idx, region in enumerate(merged_morph):
            if region['label'] != 'Ventricular Tachycardia':
                continue
            try:
                # Never into another morphological label: the walk stops at
                # the end of the region before (VFib, AFib, a block, ...), so
                # no other rhythm's segment is shortened by this.
                floor_sec = float(merged_morph[idx - 1]['end_sec']) if idx > 0 else 0.0
                vt_first = _vt_run_first_beat(region, floor_sec)
                if vt_first is not None and vt_first < region['start_sec'] - 0.05:
                    region['start_sec'] = vt_first
                    region['start_time_str'], _ = make_time_strs(vt_first, region['end_sec'])
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] VT start refine error: {e}")

        # Beat-exact edges for Atrial Fibrillation / Atrial Flutter where they
        # border normal rhythm (no other morphological region within one
        # window). The window vote leaves these on the 5s window grid, so a
        # few seconds of the normal rhythm after AFib ended were drawn as
        # AFib (seen: last AFib beat 1160.5s, steady 1000ms sinus from
        # 1161.5s, segment ran to 1165s).
        #
        # A beat counts as atrial-active when its TP segment (between the
        # previous beat's T wave and this beat's P/QRS) carries waves of at
        # least 0.3x the noise threshold _classify_atrial_window uses. Across
        # the test recordings AFib beats measured >= 0.57 and Flutter >= 1.14,
        # Sinus Brady/Tachy <= 0.26. Normal rhythm can occasionally read
        # higher (fast sinus arrhythmia, P and T crowding the TP segment), so
        # each edge is searched starting from INSIDE the episode: the end is
        # the first of two consecutive quiet beats going forward from inside
        # the last AFib/Flutter window; the start is the beat right after the
        # first two consecutive quiet beats going backward from inside the
        # first window. If no such pair exists (e.g. no measurable TP at a
        # fast rate), the window edge is kept.
        def _beat_tp_active(prev_sec, peak_sec):
            seg_start = max(0.0, prev_sec - 0.1)
            seg_end = min(total_duration, peak_sec + 0.4)
            sig = reader.read_range(seg_start, seg_end)
            if sig is None or best_lead_idx >= sig.shape[0]:
                return None
            lead_sig = np.asarray(sig[best_lead_idx], dtype=float)
            prev_idx = int(round((prev_sec - seg_start) * fs))
            cur_idx = int(round((peak_sec - seg_start) * fs))
            if cur_idx <= prev_idx or cur_idx >= len(lead_sig):
                return None
            beat = measure_beat(lead_sig, cur_idx, fs)
            if beat is None:
                return None
            rr = cur_idx - prev_idx
            q_onset = int(beat.get('q_onset') or cur_idx)
            p_onset = beat.get('p_onset')
            tp_start = prev_idx + max(int(0.28 * fs), int(0.38 * rr))
            tp_end = int(p_onset) if (beat.get('p_present') and p_onset is not None) else max(tp_start, q_onset - int(0.15 * fs))
            tp_end = min(tp_end, q_onset)
            if tp_end - tp_start < int(0.06 * fs):
                return None
            noise_threshold = max(0.05, 0.15 * float(beat.get('qrs_amplitude') or 0.1))
            return float(np.ptp(lead_sig[tp_start:tp_end])) >= 0.3 * noise_threshold

        # End of AFib / Flutter where the TP test can't decide: at a fast
        # following rhythm the T wave runs into the next P / QRS and there is
        # no TP segment to measure, so no two quiet beats were found and the
        # end stayed on the window grid (seen: last AFib beat 666.63 s, a
        # 160 bpm run of identical beats from 666.86 s, AFib drawn to 670.0 s).
        # Then the rhythm that follows decides: if its beats in the next window
        # are regular (RR CV <= 0.12), going backward from the boundary a beat
        # belongs to that run while its RR is within 15% of the run's median
        # and its QRS has the run's shape (correlation >= 0.90 with the run's
        # median beat, about the same size). AFib ends at the run's first beat.
        def _regular_run_first_beat(boundary_sec):
            after = [p for p in _sorted_peaks if boundary_sec <= p <= boundary_sec + window_size]
            if len(after) < 5:
                return None
            rr = np.diff(np.asarray(after, dtype=float))
            run_rr = float(np.median(rr))
            if run_rr <= 0 or float(np.std(rr)) / run_rr > 0.12:
                return None
            before = [p for p in _sorted_peaks if boundary_sec - window_size <= p < boundary_sec]
            if not before:
                return None
            seg_start = max(0.0, before[0] - 0.5)
            seg_end = min(total_duration, after[-1] + 0.5)
            sig = reader.read_range(seg_start, seg_end)
            if sig is None or best_lead_idx >= sig.shape[0]:
                return None
            lead_sig = np.asarray(sig[best_lead_idx], dtype=float)
            try:
                _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
                lead_sig = _filtfilt(_b, _a, lead_sig)
            except Exception:
                pass
            half_w = int(0.06 * fs)

            def _snippet(t):
                i = int(round((t - seg_start) * fs))
                if i - half_w < 0 or i + half_w > len(lead_sig):
                    return None
                s = lead_sig[i - half_w:i + half_w]
                return s - float(np.mean(s))

            ref = [s for s in (_snippet(t) for t in after) if s is not None]
            if len(ref) < 5:
                return None
            template = np.median(np.asarray(ref), axis=0)
            t_norm, t_ptp = float(np.linalg.norm(template)), float(np.ptp(template))
            if t_norm <= 0:
                return None
            first = after[0]
            for p in reversed(before):
                if abs((first - p) - run_rr) > 0.15 * run_rr:
                    break
                s = _snippet(p)
                if s is None or not (0.5 * t_ptp <= float(np.ptp(s)) <= 2.0 * t_ptp):
                    break
                n = float(np.linalg.norm(s))
                if n <= 0 or float(np.dot(s, template)) / (n * t_norm) < 0.90:
                    break
                first = p
            return first

        def _refine_atrial_edge(boundary_sec, edge):
            lo, hi = boundary_sec - window_size, boundary_sec + window_size
            peaks = sorted(float(p) for p in all_r_peaks if lo - 5.0 <= p <= hi)
            states = [
                (peaks[i], _beat_tp_active(peaks[i - 1], peaks[i]))
                for i in range(1, len(peaks)) if lo <= peaks[i] <= hi
            ]
            if edge == 'end':
                for i in range(len(states) - 1):
                    if states[i][1] is False and states[i + 1][1] is False:
                        return states[i][0]
                run_first = _regular_run_first_beat(boundary_sec)
                if run_first is not None and run_first < boundary_sec:
                    return run_first
            else:
                for i in range(len(states) - 1, 0, -1):
                    if states[i][1] is False and states[i - 1][1] is False:
                        return states[i + 1][0] if i + 1 < len(states) else boundary_sec
            return boundary_sec

        for idx, region in enumerate(merged_morph):
            if region['label'] not in ('Atrial Fibrillation', 'Atrial Flutter'):
                continue
            try:
                new_start, new_end = region['start_sec'], region['end_sec']
                if idx == 0 or region['start_sec'] - merged_morph[idx - 1]['end_sec'] > window_size:
                    new_start = _refine_atrial_edge(region['start_sec'], 'start')
                if idx + 1 >= len(merged_morph) or merged_morph[idx + 1]['start_sec'] - region['end_sec'] > window_size:
                    new_end = _refine_atrial_edge(region['end_sec'], 'end')
                if new_start < new_end - 0.5:
                    region['start_sec'], region['end_sec'] = new_start, new_end
                    region['start_time_str'], region['end_time_str'] = make_time_strs(new_start, new_end)
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] Atrial boundary refine error: {e}")

        # Beat-exact edge between ADJACENT Atrial Fibrillation and Atrial
        # Flutter regions. The window vote put this edge up to ~4 s early
        # (seen: last AFib beat 319.15 s, Flutter called from 315.0 s). Each
        # RR interval is judged on its own: Flutter's sawtooth F waves repeat
        # at a steady ~200 ms (atrial rate ~240-350/min), so the interval's
        # autocorrelation has a clear peak at a 170-250 ms lag; AFib f-waves
        # are irregular (measured <= 0.27 at that lag on the test
        # recordings, Flutter >= 0.47). Intervals too short to judge (fast
        # 2:1 Flutter) count as unknown and never move the edge.
        def _interval_is_flutter(prev_sec, peak_sec):
            s_sec, e_sec = prev_sec + 0.12, peak_sec - 0.08
            if e_sec - s_sec < 0.4:
                return None
            sig = reader.read_range(s_sec, e_sec)
            if sig is None or best_lead_idx >= sig.shape[0]:
                return None
            seg = np.asarray(sig[best_lead_idx], dtype=float)
            n = np.arange(len(seg))
            seg = seg - np.polyval(np.polyfit(n, seg, 1), n)
            ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
            if ac[0] <= 0:
                return None
            ac = ac / ac[0]
            lo_lag, hi_lag = int(0.15 * fs), int(0.35 * fs)
            lag = lo_lag + int(np.argmax(ac[lo_lag:hi_lag]))
            return bool(ac[lag] >= 0.4 and 0.17 * fs <= lag <= 0.25 * fs)

        _atrial_labels = ('Atrial Fibrillation', 'Atrial Flutter')
        for idx in range(len(merged_morph) - 1):
            cur, nxt = merged_morph[idx], merged_morph[idx + 1]
            if cur['label'] not in _atrial_labels or nxt['label'] not in _atrial_labels \
                    or cur['label'] == nxt['label'] \
                    or nxt['start_sec'] - cur['end_sec'] > window_size:
                continue
            try:
                boundary = nxt['start_sec']
                lo = max(cur['start_sec'], boundary - window_size)
                hi = min(nxt['end_sec'], max(boundary, cur['end_sec']) + window_size)
                peaks = [p for p in _sorted_peaks if lo <= p <= hi]
                states = [(peaks[i - 1], _interval_is_flutter(peaks[i - 1], peaks[i]))
                          for i in range(1, len(peaks))]
                # Two consecutive intervals in the NEW rhythm mark the switch;
                # it happened right after the beat that opens the first one.
                want = nxt['label'] == 'Atrial Flutter'
                new_edge = None
                for i in range(len(states) - 1):
                    if states[i][1] is want and states[i + 1][1] is want:
                        new_edge = states[i][0]
                        break
                if new_edge is not None and cur['start_sec'] + 0.5 < new_edge < nxt['end_sec'] - 0.5:
                    cur['end_sec'] = nxt['start_sec'] = new_edge
                    _, cur['end_time_str'] = make_time_strs(cur['start_sec'], new_edge)
                    nxt['start_time_str'], _ = make_time_strs(new_edge, nxt['end_sec'])
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] AFib/Flutter boundary refine error: {e}")

        # Beat-exact edges for Supraventricular Tachycardia (SVT). Its onset and end are
        # abrupt (textbook: "after three sinus beats, Supraventricular Tachycardia develops"),
        # so the edge is simply where the RR enters / leaves the episode's
        # rate: the first beat that opens two consecutive RR intervals within
        # +/-15% of the median RR, and the last beat that closes two such
        # intervals. Searched out to one window past each window-vote edge,
        # since that edge sits on the 5 s grid.
        # The median is taken from the beats next to each edge, not the whole
        # episode: when the rate keeps climbing (200 -> 270 bpm), the episode
        # median only reflects the fast end and its band rejected the slower
        # opening SVT beats, pulling the start ~80 s inward. Each RR must also
        # stay in SVT range (<= 340 ms, the Sinus Tachycardia run's lower
        # bound), so a local band can't extend SVT over 170 bpm sinus beats.
        _svt_max_rr = 0.340

        def _at_rr_in_band(a, b, median_rr):
            rr = b - a
            return 0.85 * median_rr <= rr <= 1.15 * median_rr and rr <= _svt_max_rr

        for region in merged_morph:
            if region['label'] != 'Supraventricular Tachycardia (SVT)':
                continue
            try:
                inside = [p for p in _sorted_peaks if region['start_sec'] <= p <= region['end_sec']]
                if len(inside) < 3:
                    continue
                head = [p for p in inside if p <= region['start_sec'] + window_size]
                tail = [p for p in inside if p >= region['end_sec'] - window_size]
                head_rr = float(np.median(np.diff(head if len(head) >= 3 else inside)))
                tail_rr = float(np.median(np.diff(tail if len(tail) >= 3 else inside)))
                near = [p for p in _sorted_peaks
                        if region['start_sec'] - window_size <= p <= region['end_sec'] + window_size]
                new_start = new_end = None
                for i in range(len(near) - 2):
                    if _at_rr_in_band(near[i], near[i + 1], head_rr) and \
                            _at_rr_in_band(near[i + 1], near[i + 2], head_rr):
                        new_start = near[i]
                        break
                for i in range(len(near) - 1, 1, -1):
                    if _at_rr_in_band(near[i - 1], near[i], tail_rr) and \
                            _at_rr_in_band(near[i - 2], near[i - 1], tail_rr):
                        new_end = near[i]
                        break
                if new_start is not None and new_end is not None and new_start < new_end - 0.5:
                    region['start_sec'], region['end_sec'] = new_start, new_end
                    region['start_time_str'], region['end_time_str'] = make_time_strs(new_start, new_end)
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] Supraventricular Tachycardia (SVT) boundary refine error: {e}")

        # SVT next to Atrial Fibrillation / Flutter: the SVT edge just found
        # is beat-exact (its own RR band), the AFib / Flutter edge is still on
        # the 5 s window grid -- the AFib / Flutter beat-exact edge search
        # above only runs next to normal rhythm. Where the two overlap, the
        # generic overlap trim below always cuts the FIRST region, which put a
        # whole window of regular SVT under AFib (seen: SVT RR 300 ms steady
        # until 114.95 s, AFib's irregular beats from 115.37 s, AFib drawn
        # from 110.0 s because window 110-120 s mixed both). Here the SVT's
        # beat edge wins: the AFib / Flutter region starts (or ends) at that
        # SVT beat instead. Same for a short gap (up to one window) between
        # them, which the gap bridging below would otherwise fill by
        # stretching the SVT up to the AFib's window-grid start (seen: last SVT
        # beat 184.35 s, first AFib beat 184.77 s, SVT drawn to 185.0 s). Only
        # this boundary moves; the SVT and AFib / Flutter rules themselves are
        # unchanged.
        _svt_label = 'Supraventricular Tachycardia (SVT)'
        _atrial_pair = ('Atrial Fibrillation', 'Atrial Flutter')
        for idx in range(len(merged_morph) - 1):
            cur, nxt = merged_morph[idx], merged_morph[idx + 1]
            try:
                if cur['label'] == _svt_label and nxt['label'] in _atrial_pair \
                        and nxt['start_sec'] - window_size <= cur['end_sec'] < nxt['end_sec'] - 0.5 \
                        and cur['end_sec'] != nxt['start_sec']:
                    nxt['start_sec'] = cur['end_sec']
                    nxt['start_time_str'], _ = make_time_strs(nxt['start_sec'], nxt['end_sec'])
                elif cur['label'] in _atrial_pair and nxt['label'] == _svt_label \
                        and cur['start_sec'] + 0.5 < nxt['start_sec'] <= cur['end_sec'] + window_size \
                        and cur['end_sec'] != nxt['start_sec']:
                    cur['end_sec'] = nxt['start_sec']
                    _, cur['end_time_str'] = make_time_strs(cur['start_sec'], cur['end_sec'])
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] SVT / atrial boundary error: {e}")

        # Beat-exact edges for Third-degree AV Block. The window right at the
        # onset holds P waves of both the old rhythm and the new atrial rhythm,
        # so it never reads as a regular atrial rhythm and the region starts
        # one window late, at the next window's first beat (seen: escape
        # rhythm from 110.87 s, region from 115.61 s). The ventricles are
        # driven by a regular, slow escape rhythm, so each edge is moved out
        # beat by beat while the RR stays within +/-15% of the region's own
        # median RR, at most one window past the window-vote edge.
        # RR alone can't stop at a sinus bradycardia of about the same rate
        # (40 bpm sinus vs a 38 bpm escape rhythm), so the walk also stops at
        # a conducted sinus beat: one whose PR matches the next beat further
        # out to within 20 ms. In complete block the P waves march through the
        # QRS complexes, so consecutive "PR" values differ beat to beat.
        _pr_at = {
            float(p): (m.get('pr_ms') if m else None)
            for p, m in zip(all_r_peaks, rate_beat_metrics)
        }

        def _sinus_conducted(beat, further_out):
            a, b = _pr_at.get(float(beat)), _pr_at.get(float(further_out))
            return a is not None and b is not None and abs(a - b) <= 20.0

        # A beat inside a proven bundle branch block (P waves with a consistent
        # PR, _classify_bundle_branch_block_window) is conducted, so it can't
        # belong to a complete block either. The PR brake above can't see it
        # there: the strict whole-recording P search often loses the P in front
        # of a wide QRS (seen: the first RBBB beat after a 30 bpm escape rhythm
        # came 2.27 s later, inside the RR band, with no PR, and the region
        # ended on it -- both labels at once).
        _third_degree_bbb_spans = [(c['start_sec'], c['end_sec']) for c in bundle_branch_block_candidates]

        def _in_proven_bundle_branch_block(beat):
            return any(s - 0.05 <= beat <= e + 0.05 for s, e in _third_degree_bbb_spans)

        for region in merged_morph:
            if region['label'] != _AV_THIRD:
                continue
            try:
                inside = [p for p in _sorted_peaks if region['start_sec'] <= p <= region['end_sec']]
                if len(inside) < 3:
                    continue
                median_rr = float(np.median(np.diff(inside)))
                lo, hi = 0.85 * median_rr, 1.15 * median_rr
                new_start, new_end = inside[0], inside[-1]
                before = [p for p in _sorted_peaks
                          if region['start_sec'] - window_size <= p < inside[0]]
                for k in range(len(before) - 1, -1, -1):
                    p = before[k]
                    if not (lo <= new_start - p <= hi):
                        break
                    if k > 0 and _sinus_conducted(p, before[k - 1]):
                        break
                    if _in_proven_bundle_branch_block(p):
                        break
                    new_start = p
                after = [p for p in _sorted_peaks
                         if inside[-1] < p <= region['end_sec'] + window_size]
                for k, p in enumerate(after):
                    if not (lo <= p - new_end <= hi):
                        break
                    if k + 1 < len(after) and _sinus_conducted(p, after[k + 1]):
                        break
                    if _in_proven_bundle_branch_block(p):
                        break
                    new_end = p
                if new_start < new_end - 0.5:
                    region['start_sec'], region['end_sec'] = new_start, new_end
                    region['start_time_str'], region['end_time_str'] = make_time_strs(new_start, new_end)
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] Third-degree AV Block boundary refine error: {e}")

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
        #
        # Beat-exact boundary where Atrial Fibrillation / Flutter is followed
        # right away by First-degree AV Block. The window that mixes the last
        # flutter beats with the first First-degree beats passes neither rule
        # (its P waves aren't all regular/conducted), so those First-degree
        # beats stayed inside the Flutter segment, which ends on the 5 s window
        # grid (seen: beats at RR 850/750/750 ms with flat baseline and PR
        # ~236 ms, 233-235 s, painted Flutter until 235.0 s). Per-beat P
        # detection missed the P on two of those beats, so the beats are
        # compared by shape instead: the block's own beats give a template of
        # P wave + PR segment + QRS (Lead II, from PR + 80 ms before the R peak
        # to 60 ms after it). Going backward from the block's start, a beat
        # joins the block when it has that shape (correlation >= 0.90, about
        # the same size) and an RR within 15% of the block's own median RR --
        # a Flutter / AFib beat has flutter / fibrillation waves where the P
        # wave is and doesn't match. The first beat that doesn't, stops it.
        # The Flutter / AFib segment then ends where the block now starts.
        # Only this one boundary moves; the Flutter / AFib and AV block rules
        # themselves are unchanged.
        _gate_metric_at = dict(zip((float(p) for p in all_r_peaks), rate_gate_metrics))

        def _first_degree_first_beat(block, prev):
            b_start = float(block['start_sec'])
            ref = [p for p in _sorted_peaks if b_start - 0.05 <= p <= min(float(block['end_sec']), b_start + window_size)]
            ref_pr = [float(_gate_metric_at[p]['pr_ms']) for p in ref
                      if _gate_metric_at.get(p) and _gate_metric_at[p].get('pr_ms') is not None]
            if len(ref) < 4 or len(ref_pr) < 3:
                return None
            pr_med = float(np.median(ref_pr))
            rr_med = float(np.median(np.diff(np.asarray(ref, dtype=float))))
            if rr_med <= 0:
                return None
            earlier = [p for p in _sorted_peaks
                       if max(float(prev['start_sec']), b_start - window_size) <= p < ref[0] - 0.05]
            if not earlier:
                return None
            pre = (pr_med + 80.0) / 1000.0
            post = 0.06
            seg_start = max(0.0, earlier[0] - pre - 0.5)
            seg_end = min(total_duration, ref[-1] + post + 0.5)
            sig = reader.read_range(seg_start, seg_end)
            if sig is None or conduction_lead_idx >= sig.shape[0]:
                return None
            lead_sig = np.asarray(sig[conduction_lead_idx], dtype=float)
            try:
                _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
                lead_sig = _filtfilt(_b, _a, lead_sig)
            except Exception:
                pass
            n_pre, n_post = int(round(pre * fs)), int(round(post * fs))

            def _beat_shape(t):
                i = int(round((t - seg_start) * fs))
                if i - n_pre < 0 or i + n_post > len(lead_sig):
                    return None
                s = lead_sig[i - n_pre:i + n_post]
                return s - float(np.mean(s))

            ref_shapes = [s for s in (_beat_shape(t) for t in ref) if s is not None]
            if len(ref_shapes) < 4:
                return None
            template = np.median(np.asarray(ref_shapes), axis=0)
            t_norm, t_ptp = float(np.linalg.norm(template)), float(np.ptp(template))
            if t_norm <= 0:
                return None
            first = ref[0]
            for p in reversed(earlier):
                if abs((first - p) - rr_med) > 0.15 * rr_med:
                    break
                s = _beat_shape(p)
                if s is None or not (0.5 * t_ptp <= float(np.ptp(s)) <= 2.0 * t_ptp):
                    break
                n = float(np.linalg.norm(s))
                if n <= 0 or float(np.dot(s, template)) / (n * t_norm) < 0.90:
                    break
                first = p
            return first

        for idx in range(1, len(merged_morph)):
            block, prev = merged_morph[idx], merged_morph[idx - 1]
            if block['label'] != _AV_FIRST or prev['label'] not in ('Atrial Fibrillation', 'Atrial Flutter'):
                continue
            if block['start_sec'] - prev['end_sec'] > window_size:
                continue
            try:
                first = _first_degree_first_beat(block, prev)
                if first is not None and first < block['start_sec'] - 0.05 and first > prev['start_sec'] + 0.5:
                    block['start_sec'] = first
                    block['start_time_str'], _ = make_time_strs(first, block['end_sec'])
                    if prev['end_sec'] > first:
                        prev['end_sec'] = first
                        _, prev['end_time_str'] = make_time_strs(prev['start_sec'], first)
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] First-degree AV Block start refine error: {e}")

        # AV block regions are beat-exact and follow their own priority
        # instead: every non-AV rhythm outranks any AV block, and among AV
        # blocks the more severe degree wins (Third > Mobitz II > Mobitz I >
        # First). A lower-ranked AV region is cut out wherever it overlaps a
        # higher-ranked one -- possibly into two pieces -- and is never
        # stretched across a gap, which would paint normal conduction as a
        # block (or the next block's label over the previous one).
        def _av_rank(label):
            return _AV_LABELS.index(label) + 1 if label in _AV_LABELS else 0

        _resolved = []
        for region in merged_morph:
            rank = _av_rank(region['label'])
            pieces = [(region['start_sec'], region['end_sec'])]
            if rank:
                for other in merged_morph:
                    if other is region or _av_rank(other['label']) >= rank:
                        continue
                    next_pieces = []
                    for p_start, p_end in pieces:
                        if other['end_sec'] <= p_start or other['start_sec'] >= p_end:
                            next_pieces.append((p_start, p_end))
                            continue
                        if other['start_sec'] > p_start:
                            next_pieces.append((p_start, other['start_sec']))
                        if other['end_sec'] < p_end:
                            next_pieces.append((other['end_sec'], p_end))
                    pieces = next_pieces
            for p_start, p_end in pieces:
                if p_end - p_start < 0.5:
                    continue
                piece = dict(region)
                piece['start_sec'], piece['end_sec'] = p_start, p_end
                piece['start_time_str'], piece['end_time_str'] = make_time_strs(p_start, p_end)
                _resolved.append(piece)
        merged_morph = sorted(_resolved, key=lambda x: x['start_sec'])

        for idx in range(len(merged_morph) - 1):
            cur = merged_morph[idx]
            nxt = merged_morph[idx + 1]
            if nxt['start_sec'] < cur['end_sec']:
                cur['end_sec'] = nxt['start_sec']
                _, cur['end_time_str'] = make_time_strs(cur['start_sec'], cur['end_sec'])
            elif cur['label'] in _AV_LABELS or nxt['label'] in _AV_LABELS:
                continue
            elif 0.0 < (nxt['start_sec'] - cur['end_sec']) <= window_size:
                cur['end_sec'] = nxt['start_sec']
                cur['end_time_str'] = nxt['start_time_str']
        merged_morph = [m for m in merged_morph if m['end_sec'] > m['start_sec']]

        # Every morphological rhythm, AV blocks included, takes priority over
        # a Sinus Bradycardia/Tachycardia rate segment: e.g. a 55 bpm rhythm
        # with PR 280 ms passes the sinus gate as Bradycardia too, but is
        # marked First-degree AV Block only, never both stacked.
        detected_segments = suppress_rate_overlap(detected_segments, merged_morph)
        detected_segments.extend(merged_morph)

    # Bundle branch block regions, added only now -- after every rhythm
    # segment above is final -- so they can't trim, suppress or merge with
    # any of them. Same-label windows merge only when they touch (beat-
    # exact spans); where two Bundle Branch Block labels overlap, complete outranks
    # incomplete; and no Bundle Branch Block is drawn over Ventricular Tachycardia/
    # Fibrillation (a ventricular wide QRS is not a bundle branch block).
    # Kept out of detected_segments until after the final same-label merge
    # below, so a Bundle Branch Block region sorted between two pieces of one rhythm can't
    # stop those pieces from merging.
    bundle_branch_block_segments = []
    if bundle_branch_block_candidates:
        bundle_branch_block_candidates.sort(key=lambda x: x['start_sec'])
        merged_bundle_branch_block = []
        for c in bundle_branch_block_candidates:
            if merged_bundle_branch_block and merged_bundle_branch_block[-1]['label'] == c['label'] \
                    and c['start_sec'] <= merged_bundle_branch_block[-1]['end_sec'] + _BUNDLE_BRANCH_BLOCK_MERGE_GAP_SEC:
                merged_bundle_branch_block[-1]['end_sec'] = max(merged_bundle_branch_block[-1]['end_sec'], c['end_sec'])
                continue
            merged_bundle_branch_block.append(dict(c))

        # Beat-level edge extension. A window is labelled only when >= 75% of
        # its beats show one pattern, so beats right at a change between two
        # patterns can sit in no labelled window at all (seen: the 2 first
        # Left Bundle Branch Block beats at 614.21/614.96 s -- the 610-620 s
        # window that judged them Left had only 62% Left beats, and the
        # 615-625 s window that was labelled started after them). Each
        # region's edges move outward one beat at a time while the next beat
        # was itself judged the same pattern (majority of its votes), with no
        # missing beat in between (gap <= 1.5x the region's own RR), at most
        # one window, never into another bundle branch block region.
        vote_times = sorted(bundle_branch_block_beat_votes)

        def _beat_side(t):
            votes = bundle_branch_block_beat_votes.get(t, [])
            right = sum(1 for r_, l_ in votes if r_ and not l_)
            left = sum(1 for r_, l_ in votes if l_ and not r_)
            if right > len(votes) / 2:
                return 'R'
            if left > len(votes) / 2:
                return 'L'
            return None

        for region in merged_bundle_branch_block:
            side = 'R' if 'Right' in region['label'] else 'L'
            others = [(o['start_sec'], o['end_sec']) for o in merged_bundle_branch_block if o is not region]
            inside = [t for t in vote_times if region['start_sec'] - 0.01 <= t <= region['end_sec'] + 0.01]
            rr = float(np.median(np.diff(inside))) if len(inside) >= 2 else 1.0
            max_gap = 1.5 * rr
            new_start = region['start_sec']
            for t in reversed([t for t in vote_times if region['start_sec'] - window_size <= t < region['start_sec'] - 0.01]):
                if new_start - t > max_gap or _beat_side(t) != side \
                        or any(o_s - 0.01 <= t <= o_e + 0.01 for o_s, o_e in others):
                    break
                new_start = t
            new_end = region['end_sec']
            for t in [t for t in vote_times if region['end_sec'] + 0.01 < t <= region['end_sec'] + window_size]:
                if t - new_end > max_gap or _beat_side(t) != side \
                        or any(o_s - 0.01 <= t <= o_e + 0.01 for o_s, o_e in others):
                    break
                new_end = t
            region['start_sec'], region['end_sec'] = float(new_start), float(new_end)
            region['start_time_str'], region['end_time_str'] = make_time_strs(region['start_sec'], region['end_sec'])

        # Two bundle branch block regions that meet beat to beat (one ends on
        # a beat, the next starts on the very next beat, e.g. Right -> Left)
        # leave that one RR interval in neither region: seen as a 0.76 s gap
        # (613.44 -> 614.20 s) where Full Disclosure's label fell back to
        # "Normal Sinus Rhythm" for one second. Where the gap is no more than
        # one beat (<= 1.5x the earlier region's own RR), the earlier region
        # now ends where the next one starts.
        merged_bundle_branch_block.sort(key=lambda x: x['start_sec'])
        for cur, nxt in zip(merged_bundle_branch_block, merged_bundle_branch_block[1:]):
            gap = nxt['start_sec'] - cur['end_sec']
            if gap <= 0:
                continue
            inside = [t for t in vote_times if cur['start_sec'] - 0.01 <= t <= cur['end_sec'] + 0.01]
            rr = float(np.median(np.diff(inside))) if len(inside) >= 2 else 1.0
            if gap <= 1.5 * rr:
                cur['end_sec'] = nxt['start_sec']
                cur['start_time_str'], cur['end_time_str'] = make_time_strs(cur['start_sec'], cur['end_sec'])

        def _bundle_branch_block_rank(label):
            return _BUNDLE_BRANCH_BLOCK_LABELS.index(label) if label in _BUNDLE_BRANCH_BLOCK_LABELS else -1

        # A Ventricular Fibrillation region is cut back off any stretch that
        # a window proved to be a conducted bundle branch block (labelled
        # window: P waves with a consistent PR, one QRS shape, wide QRS and
        # RBBB/LBBB morphology on >= 75% of beats; span = its first to last
        # such beat). VFib has no AV conduction. Its window vote and edge
        # refinement can still run it over such beats where a window mixes
        # the end of real VFib with the bundle branch block that follows --
        # seen: real VFib ending ~38 s, first Right Bundle Branch Block beat
        # 40.22 s, VFib drawn to 45.0 s, because the wide RBBB beats never
        # count as "organized" for the VFib end search. Pieces under 1 s are
        # dropped. Rule functions are unchanged; only VFib regions that
        # overlap a proven bundle branch block move.
        proven_bundle_branch_block = [(c['start_sec'], c['end_sec']) for c in bundle_branch_block_candidates]
        cut_segments = []
        for seg in detected_segments:
            if seg['label'] != 'Ventricular Fibrillation':
                cut_segments.append(seg)
                continue
            pieces = [(seg['start_sec'], seg['end_sec'])]
            for b_start, b_end in proven_bundle_branch_block:
                next_pieces = []
                for p_start, p_end in pieces:
                    if b_end <= p_start or b_start >= p_end:
                        next_pieces.append((p_start, p_end))
                        continue
                    if b_start > p_start:
                        next_pieces.append((p_start, b_start))
                    if b_end < p_end:
                        next_pieces.append((b_end, p_end))
                pieces = next_pieces
            if pieces == [(seg['start_sec'], seg['end_sec'])]:
                cut_segments.append(seg)
                continue
            bbb_edges = [e for span in proven_bundle_branch_block for e in span]
            for p_start, p_end in pieces:
                if p_end - p_start < 1.0:
                    continue
                # A piece left right against a proven bundle branch block
                # and shorter than one window step is the transition out of
                # (or into) that block -- the window that mixed its last wide
                # beats with the next rhythm read as VFib -- not VFib itself
                # (seen: 359.30-362.45 s, RBBB beats then 30 bpm narrow beats).
                touches_bbb = any(abs(p_start - e) < 0.01 or abs(p_end - e) < 0.01 for e in bbb_edges)
                if touches_bbb and p_end - p_start < step_size:
                    continue
                piece = dict(seg)
                piece['start_sec'], piece['end_sec'] = p_start, p_end
                piece['start_time_str'], piece['end_time_str'] = make_time_strs(p_start, p_end)
                cut_segments.append(piece)
        detected_segments = cut_segments

        ventricular = [
            (s['start_sec'], s['end_sec']) for s in detected_segments
            if s['label'] in ('Ventricular Tachycardia', 'Ventricular Fibrillation')
        ]
        for region in merged_bundle_branch_block:
            pieces = [(region['start_sec'], region['end_sec'])]
            blockers = ventricular + [
                (o['start_sec'], o['end_sec']) for o in merged_bundle_branch_block
                if o is not region and _bundle_branch_block_rank(o['label']) < _bundle_branch_block_rank(region['label'])
            ]
            for b_start, b_end in blockers:
                next_pieces = []
                for p_start, p_end in pieces:
                    if b_end <= p_start or b_start >= p_end:
                        next_pieces.append((p_start, p_end))
                        continue
                    if b_start > p_start:
                        next_pieces.append((p_start, b_start))
                    if b_end < p_end:
                        next_pieces.append((b_end, p_end))
                pieces = next_pieces
            for p_start, p_end in pieces:
                if p_end - p_start < 0.5:
                    continue
                piece = dict(region)
                piece['start_sec'], piece['end_sec'] = p_start, p_end
                piece['start_time_str'], piece['end_time_str'] = make_time_strs(p_start, p_end)
                bundle_branch_block_segments.append(piece)

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

    final_segments.extend(bundle_branch_block_segments)
    final_segments.sort(key=lambda x: x['start_sec'])

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
        return lbl in {
            'ventricular fibrillation', 'ventricular tachycardia',
            'atrial fibrillation', 'atrial flutter', 'supraventricular tachycardia (svt)',
            'first-degree av block', 'second-degree av block (mobitz i)',
            'second-degree av block (mobitz ii)', 'third-degree av block',
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
    segments: Optional[List[Dict[str, Any]]] = None,
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

    # `segments` lets a caller pass a result it already has (e.g. from
    # detect_arrhythmias_async) instead of running detection here.
    auto_segments = segments if segments is not None else detect_arrhythmias(reader, progress_callback=progress_callback)
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
    # region, widening it to 455-470s. Sinus Bradycardia/Tachycardia are
    # owned too: this detector's own RR-interval rule decides them, and the
    # stored real-time labels for them are not reliable -- confirmed on a
    # real recording where the live pipeline logged "Sinus Bradycardia" for
    # three 5s chunks at a steady 60 bpm (RR ~1000ms, brady_beats=0), which
    # then rendered as a 15s Bradycardia segment over plain sinus rhythm.
    # Entries for any label this detector doesn't itself classify are left
    # alone.
    #
    # Matched as substrings, the same way Full Disclosure's segment overlay
    # maps labels to regions: a stored "Ventricular Tachycardia" is drawn as
    # Ventricular Fibrillation and "Bradycardia (non-sinus)" as Sinus
    # Bradycardia there. A stored 600.0-605.0s "Ventricular Tachycardia"
    # merged into a detected VFib starting at 600.23s, so the overlay showed
    # VFib starting at 12:31:19 while the rhythm label (detector only)
    # showed 12:31:20.
    _owned_substrings = (
        'fibrillation', 'vfib', 'afib', 'flutter',
        'ventricular tachycardia', 'vtach',
        'bradycardia', 'tachycardia',
        'av block',
        'bundle branch block',
    )

    def _is_owned_label(label_lower):
        return any(s in label_lower for s in _owned_substrings)

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
