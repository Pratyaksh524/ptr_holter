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

import bisect
import copy
import hashlib
import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Callable, Optional
from scipy.ndimage import convolve1d
from scipy.signal import find_peaks, savgol_coeffs, savgol_filter


# savgol_filter(x, 11, 3) (mode "interp") for the per-beat QRS width in the
# Bundle Branch Block rule, called for every beat on 3-7 leads. scipy rebuilds
# the kernel and solves two least-squares edge fits on every call; both are
# fixed linear maps of the samples, so they are built once here: the kernel,
# and each edge's output-from-input matrix (savgol_filter of the identity).
# Same filter, without the per-call setup.
_SAVGOL_11_3_KERNEL = savgol_coeffs(11, 3)
_SAVGOL_11_3_EDGES = savgol_filter(np.eye(11), 11, 3, axis=0)


def _savgol_11_3(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = convolve1d(x, _SAVGOL_11_3_KERNEL, mode='constant')
    y[:5] = _SAVGOL_11_3_EDGES[:5] @ x[:11]
    y[-5:] = _SAVGOL_11_3_EDGES[6:] @ x[-11:]
    return y


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
    return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{_code_fingerprint()}|age={_patient_age(path)}"


def _patient_age(path: Optional[str]) -> Optional[float]:
    """Age in years from the recording's patient.json, or None when it is
    missing, blank or not a number in 1-120 (93 of 98 recordings here)."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(path)), 'patient.json'), encoding='utf-8') as f:
            age = float(str(json.load(f).get('age', '')).strip())
    except Exception:
        return None
    return age if 1.0 <= age <= 120.0 else None


# Supraventricular Tachycardia (SVT) rate bound -- Philips DXL ECG Algorithm
# Physician's Guide, p. B-8: "Supraventricular tachycardia ... V-rate >
# (220 - age), QRSd < **". With no usable age the bound stays 180 bpm and
# every rate edge tied to it (SVT edge RR <= 340 ms, Sinus Tachycardia RR >=
# 340 ms) stays as it was.
_SVT_DEFAULT_MIN_HR = 180.0


def _svt_age_min_hr(reader) -> Optional[float]:
    """220 - age for the recording, or None without a usable age."""
    age = _patient_age(getattr(reader, 'path', None))
    return None if age is None else 220.0 - age


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


# Long recordings: the morphological window loop in _detect_arrhythmias_uncached
# (10 s windows, 5 s step) is ~90% of the run time and every window is judged
# on its own samples only, so it is split into jobs of consecutive windows and
# run in worker processes. Each worker calls _detect_arrhythmias_uncached with
# _window_job=(first, stop) window indices: it skips the whole-recording R-peak
# pass and rate rules and returns just its windows' raw votes, which are put
# back together in window order -- the same lists the serial loop builds, so
# the cleanup after the loop and the output segments are unchanged. The jobs
# start before the whole-recording pass so both run at the same time.
# Processes, not threads: the per-beat work is small numpy/scipy calls that
# hold the GIL. Short recordings, or any failure here, use the serial loop.
_PARALLEL_MIN_DURATION_SEC = 15 * 60.0
_PARALLEL_WINDOW_STEP_SEC = 5.0       # must equal step_size in the window loop
_PARALLEL_JOB_WINDOWS = 120           # 10 minutes of recording per job
_PARALLEL_MAX_WORKERS = 8
# A worker peaks at ~215 MB (numpy/scipy ~100 MB, plus one job's working
# set). Workers are limited by memory free at that moment, two ways:
#   - commit (RAM + page file) is a hard limit: running out of it fails
#     allocations ("Unable to allocate ...") in every process on the PC, so
#     _PARALLEL_COMMIT_RESERVE always stays free;
#   - physical RAM is a soft limit: past it Windows pages idle programs out
#     and everything slows, so no more workers than fit, but at least 2.
_PARALLEL_WORKER_RAM = 250 * 2**20
_PARALLEL_COMMIT_RESERVE = 1024 * 2**20
_PARALLEL_RAM_RESERVE = 256 * 2**20


def _parallel_worker_limit() -> Optional[int]:
    """Most workers free memory allows right now, or None if it can't be read."""
    commit, ram = _available_memory_bytes()
    if commit is None:
        return None
    by_commit = (commit - _PARALLEL_COMMIT_RESERVE) // _PARALLEL_WORKER_RAM
    by_ram = (ram - _PARALLEL_RAM_RESERVE) // _PARALLEL_WORKER_RAM if ram is not None else by_commit
    return int(min(by_commit, max(2, by_ram)))


def _available_memory_bytes() -> tuple:
    """(commit, physical RAM) free right now in bytes; None where it can't be read."""
    try:
        if os.name == 'nt':
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None, None
            return int(status.ullAvailPageFile), int(status.ullAvailPhys)
        ram = int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        return ram, ram
    except Exception:
        return None, None


def _window_loop_job(path: str, first_window: int, stop_window: int) -> Dict[str, Any]:
    from .file_format import ECGHFileReader
    with ECGHFileReader(path) as reader:
        return _detect_arrhythmias_uncached(reader, None, _window_job=(first_window, stop_window))


def _start_parallel_window_loop(reader, total_duration: float):
    """Submit the window loop's jobs to a process pool, or None to run it serially."""
    path = getattr(reader, 'path', None)
    if not path or not os.path.isfile(path) or total_duration < _PARALLEL_MIN_DURATION_SEC:
        return None
    workers = min(_PARALLEL_MAX_WORKERS, (os.cpu_count() or 1) - 1)   # one core left for the UI
    memory_limit = _parallel_worker_limit()
    if memory_limit is not None:
        workers = min(workers, memory_limit)
    if workers < 2:
        if memory_limit is not None:
            print("[Auto Arrhythmia Detect] Not enough free memory for worker processes, running serially")
        return None
    try:
        import math
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        n_windows = int(math.ceil(total_duration / _PARALLEL_WINDOW_STEP_SEC))
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
        futures = [
            pool.submit(_window_loop_job, path, first, min(first + _PARALLEL_JOB_WINDOWS, n_windows))
            for first in range(0, n_windows, _PARALLEL_JOB_WINDOWS)
        ]
        print(f"[Auto Arrhythmia Detect] {n_windows} windows in {len(futures)} jobs on {workers} processes")
        return pool, futures
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Parallel window loop unavailable, running serially: {e}")
        return None


def _collect_parallel_window_loop(handle, progress_callback=None) -> Optional[List[Dict[str, Any]]]:
    """Each job's window votes in window order, or None if any job failed."""
    from concurrent.futures import as_completed
    pool, futures = handle
    try:
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if progress_callback:
                progress_callback(int(done * 100 / len(futures)))
        return [future.result() for future in futures]
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Parallel window loop failed, running serially: {e}")
        return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
# ──────────────────────────────────────────────────────────────────────────
# ASYSTOLE DETECTION -- intentionally separate from the pipeline below
# ──────────────────────────────────────────────────────────────────────────
# A flatline (no P, no QRS, no T -- the ECG goes silent) sustained for
# several seconds is its own finding. It must not be decided, or even
# touched, by the morphological rules in _detect_arrhythmias_uncached():
# _is_vfib_window() there only ever looks at beats Pan-Tompkins actually
# found inside its own 10s window, so a silent run with zero detected beats
# is invisible to it and simply gets inherited into whichever window it
# falls inside -- confirmed directly on a recording where a ~5s flatline
# right after a Ventricular Tachycardia run, followed a few seconds later
# by the first chaotic beats, got folded whole into that window's
# Ventricular Fibrillation call, purely because the window vote has no way
# to "see" the silence itself, only the beats on either side of it.
#
# Per the architecture rule (see this module's own docstring and the
# arrhythmia-change-rules memory), a fix for one arrhythmia's logic must
# not touch any other arrhythmia's output -- teaching _is_vfib_window()
# about silence would risk exactly that kind of cross-label regression.
# Asystole is reported here instead as a fully independent finding: its
# own detector, its own cache file, never called by detect_arrhythmias()/
# _detect_arrhythmias_uncached() and never merged by
# apply_auto_segments_to_engine() into engine._structured_events -- so it
# cannot change a single byte of what the pipeline above outputs, and is
# not gated by (or counted as part of) "Auto Arrhythmia Detection" /
# _auto_marking_visible either. Callers that want it (the Full Disclosure
# rhythm label, the report's Event Timeline) call
# detect_asystole_segments()/detect_asystole_segments_async() directly and
# display its result on their own, alongside -- never inside -- the
# auto-marking pipeline.

_ASYSTOLE_CACHE_FILENAME = 'asystole_segments_cache.json'
_asystole_memory_cache: Dict[str, List[Dict[str, Any]]] = {}
_asystole_async_jobs: Dict[str, Dict[str, Any]] = {}

# Minimum sustained flatline duration to call it Asystole rather than a
# brief dropout/artifact blip.
_ASYSTOLE_MIN_SEC = 3.0


def _asystole_disk_cache_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), _ASYSTOLE_CACHE_FILENAME)


def get_cached_asystole(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Cached detect_asystole_segments() result for this recording, or
    None. Mirrors get_cached_detection(), but reads/writes its own cache
    file -- entirely independent of the morphological pipeline's cache."""
    key = _cache_key(path)
    if key is None:
        return None
    if key in _asystole_memory_cache:
        return copy.deepcopy(_asystole_memory_cache[key])
    if not _code_fingerprint():
        return None
    try:
        with open(_asystole_disk_cache_path(path), 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if saved.get('key') != key or not isinstance(saved.get('segments'), list):
        return None
    _asystole_memory_cache[key] = saved['segments']
    return copy.deepcopy(saved['segments'])


def _asystole_time_strs(reader, s_sec: float, e_sec: float):
    """HH:MM:SS start/end strings for a [s_sec, e_sec) span -- same format
    detect_arrhythmias()'s own make_time_strs() produces."""
    try:
        if hasattr(reader, 'start_time') and reader.start_time is not None:
            s_real = datetime.fromtimestamp(reader.start_time + s_sec)
            e_real = datetime.fromtimestamp(reader.start_time + e_sec)
            return s_real.strftime('%H:%M:%S'), e_real.strftime('%H:%M:%S')
    except Exception:
        pass

    def _hms(sec):
        sec = max(0.0, sec)
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    return _hms(s_sec), _hms(e_sec)


def _detect_asystole_uncached(reader) -> List[Dict[str, Any]]:
    """Standalone flatline scan -- see the banner comment above this
    section for why this is deliberately independent of
    _detect_arrhythmias_uncached()."""
    fs = getattr(reader, 'fs', 500)
    if fs is None or fs <= 0:
        fs = 500
    total_duration = getattr(reader, 'duration_sec', 0)
    if total_duration <= 0:
        return []

    lead_names = getattr(reader, 'lead_names', []) or [
        "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    ]
    # Pick the lead with the largest early amplitude, the same ranking
    # detect_arrhythmias() uses for its own best_lead_idx -- a strong lead
    # gives a reliable reference amplitude for "normal" beats below, so the
    # flatline threshold scales with this recording's own signal instead of
    # a fixed count (this reader returns raw ADC counts, not calibrated mV,
    # so a fixed-unit floor like is_asystole()'s in arrhythmia_detector.py
    # does not transfer here).
    lead_candidates = [l for l in ['II', 'V5', 'V6', 'V4', 'V1', 'I'] if l in lead_names] or lead_names
    test_chunk = reader.read_range(0.0, min(30.0, total_duration))
    best_lead_idx, best_ptp = 0, 0.0
    for cand in lead_candidates:
        idx = lead_names.index(cand)
        if test_chunk is not None and idx < test_chunk.shape[0]:
            ptp = float(np.ptp(test_chunk[idx]))
            if ptp > best_ptp:
                best_ptp, best_lead_idx = ptp, idx

    block_sec = 1.0
    chunk_len_sec = 120.0
    block_samples = max(1, int(round(block_sec * fs)))
    blocks = []  # (block_start_sec, ptp, std)
    t_cursor = 0.0
    while t_cursor < total_duration:
        t_end = min(t_cursor + chunk_len_sec, total_duration)
        chunk = reader.read_range(t_cursor, t_end)
        if chunk is not None and best_lead_idx < chunk.shape[0]:
            sig = np.asarray(chunk[best_lead_idx], dtype=float)
            for b_start in range(0, sig.size, block_samples):
                seg = sig[b_start:b_start + block_samples]
                if seg.size < block_samples * 0.5:
                    continue
                blocks.append((t_cursor + b_start / fs, float(np.ptp(seg)), float(np.std(seg))))
        t_cursor += chunk_len_sec

    if not blocks:
        return []

    # Reference amplitude: the 75th percentile of 1s-block peak-to-peak
    # values across the whole recording -- what a normal beating stretch
    # looks like on this lead/device, whatever its raw units happen to be.
    all_ptp = np.asarray([b[1] for b in blocks], dtype=float)
    reference_ptp = float(np.percentile(all_ptp, 75))
    if reference_ptp <= 0:
        return []

    # A block is flat when BOTH its peak-to-peak and its std sit well below
    # that reference -- the same two-sided amplitude check is_asystole()
    # (arrhythmia_detector.py) uses, scaled to this recording's own signal
    # rather than that function's fixed mV floor.
    flat_ptp_limit = max(1.0, 0.08 * reference_ptp)
    flat_std_limit = max(1.0, 0.04 * reference_ptp)

    runs = []
    run_start = None
    run_end = None
    for b_start, ptp, std in blocks:
        is_flat = ptp < flat_ptp_limit and std < flat_std_limit
        b_end = b_start + block_sec
        if is_flat:
            if run_start is None:
                run_start = b_start
            run_end = b_end
        elif run_start is not None:
            runs.append((run_start, run_end))
            run_start = None
    if run_start is not None:
        runs.append((run_start, run_end))

    # Each run's edges above are only accurate to the whole second (the
    # scan runs in 1s blocks), even though the real flat-to-chaos
    # transition usually sits partway through the first/last block -- the
    # exact same limitation _vfib_flatline_start_trim/_vt_flatline_end_trim
    # (auto_segment_arrthymia_detection.py's VFib/VT edge trims) hit on
    # their own copy of this same boundary, and fix the same way: finish in
    # smaller 0.2s steps from here. Kept in sync deliberately -- those two
    # trims report where a VFib/VT region's own edge leaves a confirmed
    # flatline, and an Asystole run reported here can sit right next to
    # one; without this, the two independent detectors could disagree by
    # up to a second on exactly the same transition, leaving a sliver where
    # neither claims the flatline (or neither claims the rhythm after it).
    def _ptp_std(t0, t1):
        seg = reader.read_range(max(0.0, t0), min(total_duration, t1))
        if seg is None or best_lead_idx >= seg.shape[0]:
            return None
        x = np.asarray(seg[best_lead_idx], dtype=float)
        if x.size == 0:
            return None
        return float(np.ptp(x)), float(np.std(x))

    fine_step = 0.2

    def _refine_run_start(s, hi_bound):
        t = s
        limit = max(0.0, t - 1.0)
        while t > limit:
            blk = _ptp_std(t - fine_step, t)
            if blk is None or not (blk[0] < flat_ptp_limit and blk[1] < flat_std_limit):
                break
            t -= fine_step
        return min(t, hi_bound)

    def _refine_run_end(e, lo_bound):
        t = e
        limit = min(total_duration, t + 1.0)
        while t < limit:
            blk = _ptp_std(t, t + fine_step)
            if blk is None or not (blk[0] < flat_ptp_limit and blk[1] < flat_std_limit):
                break
            t += fine_step
        return max(t, lo_bound)

    # Keep each run's whole-second core too: the all-leads check below runs
    # on the core, not the refined edges. The refinement only looks at
    # best_lead_idx, so it can pull in the first 0.2 s where the signal is
    # back -- seen: lead II 129 counts (under its 165 limit) at 787.0-787.2 s
    # while the other 11 leads were already at 750-2100 counts. That partial
    # last block then failed on every lead and dropped a 130 s flatline
    # (658-787 s, ptp 0 on all 12 leads) as 0/12 flat.
    runs = [
        ((_refine_run_start(s, e - fine_step), _refine_run_end(e, s + fine_step)), (s, e))
        for s, e in runs
    ]

    # The flatline must be on most leads, not only on best_lead_idx: a
    # loose / detached electrode flattens just the leads it feeds while the
    # heart keeps beating on the rest. Seen on a human recording: V4 flat
    # for 4.2 s (270.8-275.0 s) after its electrode lost contact, normal
    # P-QRS-T on the other 11 leads -- called Asystole. Each lead is judged
    # with the same 1 s block test, against its own 90th-percentile block
    # amplitude in the 60 s around the run (75th over 30 s was too low
    # where the run sits among other pauses -- seen: 15 bpm with 3 s pauses,
    # most 1 s blocks around the run flat themselves). Measured: every
    # Asystole run on the test recordings was flat on 9-12 of 12 leads; the
    # electrode one on 1 of 12.
    asystole_context_sec = 60.0

    def _flat_on_most_leads(start_sec, end_sec):
        c0 = max(0.0, start_sec - asystole_context_sec)
        c1 = min(total_duration, end_sec + asystole_context_sec)
        data = reader.read_range(c0, c1)
        if data is None or data.shape[0] == 0:
            return True
        i0, i1 = int(round((start_sec - c0) * fs)), int(round((end_sec - c0) * fs))
        flat_leads = 0
        for li in range(data.shape[0]):
            x = np.asarray(data[li], dtype=float)
            ref = [float(np.ptp(x[b:b + block_samples]))
                   for b in range(0, x.size - block_samples + 1, block_samples)]
            if not ref:
                continue
            lead_ref = float(np.percentile(ref, 90))
            ptp_limit = max(1.0, 0.08 * lead_ref)
            std_limit = max(1.0, 0.04 * lead_ref)
            run = x[i0:i1]
            blocks = [run[b:b + block_samples] for b in range(0, run.size, block_samples)]
            blocks = [blk for blk in blocks if blk.size >= block_samples * 0.5]
            if blocks and all(float(np.ptp(blk)) < ptp_limit and float(np.std(blk)) < std_limit
                              for blk in blocks):
                flat_leads += 1
        return flat_leads * 2 >= data.shape[0]

    segments = []
    for (start_sec, end_sec), (core_start, core_end) in runs:
        if end_sec - start_sec < _ASYSTOLE_MIN_SEC:
            continue
        if not _flat_on_most_leads(core_start, core_end):
            continue
        s_str, e_str = _asystole_time_strs(reader, start_sec, end_sec)
        segments.append({
            'start_sec': start_sec,
            'end_sec': end_sec,
            'label': 'Asystole',
            'color': '#777777',
            'start_time_str': s_str,
            'end_time_str': e_str,
            'detected_rhythm': 'Asystole',
        })
    return segments


def detect_asystole_segments(reader) -> List[Dict[str, Any]]:
    """Cached wrapper around _detect_asystole_uncached(). Entirely separate
    from detect_arrhythmias(): its own cache file, never invoked by or
    merged into the morphological/rate pipeline above."""
    path = getattr(reader, 'path', None)
    key = _cache_key(path)
    if key is None:
        return _detect_asystole_uncached(reader)

    cached = get_cached_asystole(path)
    if cached is not None:
        return cached

    segments = _detect_asystole_uncached(reader)
    _asystole_memory_cache[key] = copy.deepcopy(segments)
    if _code_fingerprint():
        cache_path = _asystole_disk_cache_path(path)
        try:
            tmp_path = cache_path + '.tmp'
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump({'key': key, 'segments': segments}, f, default=float)
            os.replace(tmp_path, cache_path)
        except (OSError, TypeError, ValueError) as e:
            print(f"[Asystole Detect] Could not save cache: {e}")
    return segments


def detect_asystole_segments_async(path: str, on_done: Callable[[List[Dict[str, Any]]], None]) -> None:
    """Run detect_asystole_segments() for the recording at `path` on a
    worker thread and call on_done(segments) on the Qt main thread. Must be
    called from the Qt main thread. Mirrors detect_arrhythmias_async(), with
    its own job table so neither detector's background run can block or be
    confused with the other's."""
    import threading
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

    cached = get_cached_asystole(path)
    if cached is not None:
        QTimer.singleShot(0, lambda: on_done(cached))
        return

    key = _cache_key(path) or os.path.abspath(path)
    job = _asystole_async_jobs.get(key)
    if job is not None:
        job.callbacks.append(on_done)
        return

    class _AsystoleJob(QObject):
        done = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.callbacks = [on_done]
            self.done.connect(self._deliver)

        @pyqtSlot(object)
        def _deliver(self, segments):
            _asystole_async_jobs.pop(key, None)
            for cb in self.callbacks:
                try:
                    cb(copy.deepcopy(segments))
                except RuntimeError:
                    pass  # the receiving widget was closed meanwhile
                except Exception as e:
                    print(f"[Asystole Detect] Callback error: {e}")

    job = _AsystoleJob()
    _asystole_async_jobs[key] = job
    notifier = job

    def _run():
        try:
            from .file_format import ECGHFileReader
            reader = ECGHFileReader(path)
            try:
                segments = detect_asystole_segments(reader)
            finally:
                reader.close()
        except Exception as e:
            print(f"[Asystole Detect] Background detection failed: {e}")
            segments = []
        notifier.done.emit(segments)

    threading.Thread(target=_run, name='asystole-detection', daemon=True).start()


class _MainsNotchedPreprocessor:
    """SignalPreprocessor.process() (0.5-40 Hz band-pass + the config's one
    mains notch, notch_hz) followed by a notch at the other mains frequency,
    so 50 and 60 Hz are both removed -- as the Full Disclosure strip does
    (holter_widgets._remove_mains). A recording does not store its mains
    frequency, and one simulator recording switched 50 -> 60 Hz mid-way.
    Seen: 60 Hz mains (88% of Lead II's power) left in by the 50 Hz-only
    notch read as atrial activity -- Mobitz I called Atrial Fibrillation
    (17:28:00-17:28:20), and the AV-block P-wave context failed in Third-
    degree, which then read as plain Bradycardia. Detector only: the config
    and the live pipeline's own SignalPreprocessor are unchanged."""

    _MAINS_HZ = (50.0, 60.0)

    def __init__(self, base, fs, config_notch_hz):
        self._base = base
        self._notches = []
        try:
            from ..enhanced_filter import make_notch
        except ImportError:
            from ecg.enhanced_filter import make_notch
        for hz in self._MAINS_HZ:
            if abs(hz - config_notch_hz) > 0.5 and hz < fs / 2.0:
                bq = make_notch(hz, fs)
                self._notches.append((np.asarray(bq.b, dtype=float), np.asarray(bq.a, dtype=float)))

    def process(self, raw_signal):
        from scipy.signal import filtfilt
        y = self._base.process(raw_signal)
        if y.size > 15:
            for b, a in self._notches:
                y = filtfilt(b, a, y)
        return y


def _detect_arrhythmias_uncached(
    reader,
    progress_callback: Optional[Callable[[int], None]] = None,
    _window_job: Optional[tuple] = None,
) -> List[Dict[str, Any]]:
    """
    Auto-detect arrhythmias and rate-based rhythm segments (Sinus Tachycardia / Sinus Bradycardia)
    directly from ECG waveforms with beat-exact QRS peak boundary alignment.

    Args:
        reader: ECGHFileReader instance with read_range method
        progress_callback: Optional callback function(int) for progress updates (0-100)
        _window_job: internal, set only in a worker process of the parallel
            window loop (see _start_parallel_window_loop): (first, stop)
            window indices; returns those windows' raw votes, not segments.

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

    # SVT rate bound for this patient (_SVT_DEFAULT_MIN_HR without an age).
    _svt_age_hr = _svt_age_min_hr(reader)
    _svt_min_hr = _svt_age_hr if _svt_age_hr is not None else _SVT_DEFAULT_MIN_HR

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
        _atrial_preproc = _MainsNotchedPreprocessor(
            SignalPreprocessor(fs=getattr(reader, 'fs', 500) or 500, config=_sqi_config),
            float(getattr(reader, 'fs', 500) or 500), float(_sqi_config.notch_hz))
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

    # Every rule below reads the recording with mains interference (50 / 60
    # Hz) subtracted instead of only notched: a notch rings for ~1 s where
    # the interference starts, stops or switches frequency, and that ringing
    # hid P waves and QRS ends from the rules (see mains_cancel). Where no
    # mains is found the samples are exactly the recording's own. Only the
    # mains Artifact layer reads raw_reader -- it measures the interference.
    raw_reader = reader
    try:
        from .mains_cancel import MainsCancelledReader
    except ImportError:
        from ecg.holter.mains_cancel import MainsCancelledReader
    reader = MainsCancelledReader(raw_reader)

    # Long recording: the window loop below starts now in worker processes
    # and runs alongside the whole-recording pass (see _PARALLEL_* above).
    _parallel_window_handle = None
    if _window_job is None:
        _parallel_window_handle = _start_parallel_window_loop(reader, total_duration)

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

    # A window-loop worker needs none of the whole-recording beats: with no
    # R-peaks, the beat measurements and the rate rules below are skipped too.
    while _window_job is None and t_cursor < total_duration:
        t_end = min(t_cursor + chunk_len_sec, total_duration)
        pad_start = max(0.0, t_cursor - chunk_pad_sec)
        pad_end = min(total_duration, t_end + chunk_pad_sec)
        chunk_data = reader.read_range(pad_start, pad_end)
        if chunk_data is not None and best_lead_idx < chunk_data.shape[0]:
            lead_sig = np.asarray(chunk_data[best_lead_idx], dtype=float)
            if len(lead_sig) > 50:
                p_indices, p_without_shape = pan_tompkins(lead_sig, fs=fs, split_shape_match=True)
                if len(p_indices) != len(p_without_shape):
                    p_indices = _keep_confirmed_shape_beats(p_indices, p_without_shape, [
                        pan_tompkins(np.asarray(chunk_data[li], dtype=float), fs=fs)
                        for li in peak_detection_lead_indices
                        if li != best_lead_idx and li < chunk_data.shape[0]], fs)
                for p_idx in p_indices:
                    peak_sec = pad_start + (float(p_idx) / float(fs))
                    if peak_sec < t_cursor or peak_sec >= t_end:
                        continue
                    if not all_r_peaks or (peak_sec - all_r_peaks[-1]) >= 0.20:
                        all_r_peaks.append(peak_sec)
            all_r_peaks.extend(_beats_missed_on_best_lead(
                chunk_data, best_lead_idx, peak_detection_lead_indices, all_r_peaks,
                pad_start, t_cursor, t_end, fs, pan_tompkins))
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

    # The Sinus Bradycardia / Tachycardia gate reads sinus_gate_metrics:
    # the same beats, the same 25 Hz low-pass, plus the 0.5 Hz DFT high-pass
    # the Full Disclosure strip draws with (ecg_filters.apply_dft_filter).
    # The recording holds raw samples, and without the high-pass a ~0.36 Hz
    # baseline wander moved the P / QRS boundaries beat to beat -- seen: 150
    # bpm sinus, lead II baseline swinging ~1265 ADC, PR 96-300 ms and P
    # polarity flipping inside 6-beat runs, so no run passed the PR +-25 ms
    # check. rate_gate_metrics stays as it was: the First-degree AV Block
    # boundary step reads its PR.
    sinus_gate_metrics = [None] * len(all_r_peaks)
    try:
        from ..ecg_filters import apply_dft_filter as _apply_dft_filter
    except Exception:
        _apply_dft_filter = None

    try:
        from ..enhanced_filter import make_notch as _make_notch
    except Exception:
        _make_notch = None

    # 50 and 60 Hz notches first, as the Full Disclosure strip does
    # (holter_widgets._remove_mains): the 25 Hz low-pass below still passes
    # part of the mains, and with the simulator's mains on, that hid every
    # P wave -- seen: sinus 30 bpm, P on 10/10 beats before a 50 Hz stretch,
    # 0/10 inside it (10/10, PR 152-174 ms, with the notch), so Sinus
    # Bradycardia stopped there and only the rate-only Bradycardia remained.
    def _mains_dft_filter(x):
        """50 + 60 Hz notches and the 0.5 Hz DFT high-pass, as the Full
        Disclosure strip filters (no low-pass)."""
        if _make_notch is not None:
            for _mains_hz in (50.0, 60.0):
                if _mains_hz < float(fs) / 2.0:
                    try:
                        _bq = _make_notch(_mains_hz, float(fs))
                        x = _filtfilt(_bq.b, _bq.a, np.asarray(x, dtype=float))
                    except Exception:
                        pass
        if _apply_dft_filter is not None:
            try:
                x = np.asarray(_apply_dft_filter(x, float(fs), "0.5"), dtype=float)
            except Exception:
                pass
        return x

    def _sinus_gate_filter(x):
        return _gate_filter(_mains_dft_filter(x))
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
        sinus_signal = _sinus_gate_filter(metric_signal)
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
                sinus_metric = measure_beat(
                    sinus_signal, local_peak, fs, strict_pr=True, prev_rr_ms=prev_rr_ms,
                    p_collision_skip_prev_qrs_ms=_SINUS_P_SKIP_PREV_QRS_MS,
                )
                sinus_gate_metrics[metric_cursor] = sinus_metric
                if sinus_metric and sinus_metric.get('p_peak') is not None:
                    sinus_metric['_p_time_sec'] = metric_start + float(sinus_metric['p_peak']) / float(fs)
                    _set_p_span_sec(sinus_metric, metric_start, fs)
                if sinus_metric:
                    sinus_metric['_r_time_sec'] = peak_sec
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

    # Beat-exact Sinus Tachycardia / Sinus Bradycardia segments. Moved out
    # to _sinus_rate_segments() (with its own helpers _sinus_beat() /
    # _sinus_conduction_gate() / _append_sinus_rate_segments(), defined at
    # module level next to _plain_rate_segments) -- pure extraction, same
    # computation, same order, same inputs; only how it's organized on disk
    # changed. _sinus_conduction_gate() is shared: the Sinus Arrhythmia rule
    # further below calls it too. brady_r_peaks is returned alongside the
    # segments: the plain Bradycardia/Tachycardia pass further below reuses
    # it as its own Bradycardia candidate beat list.
    _sinus_rate_result, brady_r_peaks = _sinus_rate_segments(
        reader, fs, best_lead_idx, conduction_lead_idx, total_duration,
        all_r_peaks, all_rr_ms, sinus_gate_metrics, _sinus_gate_filter, make_time_strs,
        tachy_min_rr_ms=340.0 if _svt_age_hr is None else 60000.0 / _svt_age_hr,
    )
    detected_segments.extend(_sinus_rate_result)

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
    _vfib_window_kind = {'last': None, 'beats': 0}

    def _is_vfib_window(window_signal, r_peaks, fs):
        _vfib_window_kind['last'] = None
        _vfib_window_kind['beats'] = 0
        if len(r_peaks) < 2:
            return False
        # VF is a chaotic, disorganized ventricular rhythm. A regular rhythm
        # (such as regular VT or sinus tachycardia) cannot be VFib.
        if len(r_peaks) >= 3:
            rr_diffs = np.diff(np.asarray(r_peaks, dtype=float)) * 1000.0 / float(fs)
            mean_rr = float(np.mean(rr_diffs))
            if mean_rr > 0 and (float(np.std(rr_diffs)) / mean_rr) < 0.20:
                return False
            # Regular beats with dropped-beat pauses are not VFib either:
            # with the RRs at ~2x / 3x the median RR left out, >= 80% of the
            # rest are within 15% of the median (one artifact spike counted
            # as a beat must not undo it), and every beat has one shape.
            # Seen: RBBB at 80 bpm with Mobitz II pauses (2.25 s, two blocked
            # P waves) -- one pause put the window's RR spread at 0.49-0.58,
            # its wide QRS never 'resolved' and P was not found on the
            # R-peak lead (V4), and 39 s of clean RBBB were called VFib.
            if len(rr_diffs) >= 4:
                med_rr = float(np.median(rr_diffs))
                ratio = rr_diffs / med_rr if med_rr > 0 else rr_diffs
                pause = ((ratio >= 1.8) & (ratio <= 2.3)) | ((ratio >= 2.7) & (ratio <= 3.3))
                rest = ratio[~pause]
                if pause.any() and rest.size >= 3 \
                        and float(np.mean(np.abs(rest - 1.0) <= 0.15)) >= 0.8:
                    shape_corr = _aligned_beat_shape_corr(window_signal, r_peaks, fs)
                    if shape_corr is not None and shape_corr >= 0.95:
                        return False
        beats = [b for b in (measure_beat(window_signal, int(p), fs) for p in r_peaks) if b is not None]
        if not beats:
            return False
        _vfib_window_kind['beats'] = len(beats)

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
        # > 120/min: Hampton p60 ("not VT unless the rate exceeds 120");
        # a slower wide regular rhythm is accelerated idioventricular
        # (Luthra p196: 60-100), not VT. Was 100.
        if median_rr <= 0.0 or (60000.0 / median_rr) < 120.0:
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
        # Upper bound 300 ms: VT QRS has no textbook ceiling (Hampton p74
        # shows a VT at 280 ms; Luthra: QRS > 0.14 s); was 220.
        wide_beats = [
            b for b in core_beats
            if b.get('qrs_ms') is not None
            and 120.0 <= float(b['qrs_ms']) <= 300.0
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

        # F waves are counted on a 60 Hz-notched copy (_atrial_preproc
        # notches 50 Hz only); the beats above stay measured as before.
        # Seen: Flutter 5:1 under simulator 60 Hz mains, 18 "waves" in 6
        # TP stretches (atrial rate ~548/min) as the ripple split each F wave.
        tp_signal = window_signal
        if _make_notch is not None and 60.0 < float(fs) / 2.0:
            try:
                _bq60 = _make_notch(60.0, float(fs))
                tp_signal = _filtfilt(_bq60.b, _bq60.a, np.asarray(window_signal, dtype=float))
            except Exception:
                tp_signal = window_signal

        quiet_count = 0
        scored = 0
        scored_rr = []
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
            scored_rr.append(rr_local)
            segment = tp_signal[tp_start:tp_end]
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
            # ventricular rate of the beats whose TP segment was judged --
            # the same beats the atrial rate comes from
            'scored_ventricular_rate': 60.0 * float(fs) / float(np.mean(scored_rr)),
        }

    def _classify_atrial_flutter_window(context):
        """Apply only the existing Atrial Flutter rule."""
        quiet_count = context['quiet_count']
        scored = context['scored']
        total_active_sec = context['total_active_sec']
        total_waves = context['total_waves']
        # The atrial rate is counted only on the beats whose TP segment could
        # be judged, so the ventricular rate it is compared with comes from
        # those same beats. With the rate of every beat in the window, a
        # window straddling a fast regular rhythm and AFib compared AFib's
        # f waves (judged on the few slow AFib beats) with mostly the fast
        # beats' rate (seen: SVT 200 bpm -> AFib, 4 of 24 beats judged,
        # 471/min vs 148 bpm = 3.18 -> "Flutter" on the SVT beats). Over all
        # recordings only 4 of 114 Flutter-voting windows change, all at an
        # atrial rate of 442-471/min (AFib f waves, not Flutter).
        ventricular_rate = context['scored_ventricular_rate']

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

    # Beat-by-beat check on all 12 leads, run only on a window the VFib or the
    # AFib / Atrial Flutter rule has already called. Those rules judge one
    # lead (the best-amplitude lead), so an artifact on that lead alone (a
    # loose electrode, motion) or one mis-picked P wave per window made plain
    # sinus rhythm read as VFib / AFib / Flutter. Seen on a 30 min human
    # recording (24 y, sinus rhythm with respiratory sinus arrhythmia, RR
    # 540-1070 ms): 13 min of AFib, 23 s of VFib and 8 s of Flutter, while
    # Lead II showed a P wave at a fixed PR before every QRS throughout.
    # Per lead (filtered like _classify_atrial_window), per beat:
    #   organized QRS -- QRS boundary resolved (measure_beat) and the beat's
    #     QRS (120 ms) correlates >= 0.9 with the lead's own median QRS of the
    #     window, each beat aligned +/-40 ms first;
    #   sinus beat -- the stretch 300-70 ms before the R peak (where the P wave
    #     sits) correlates >= 0.7 with the lead's median of that stretch, AND
    #     the diastole before it (after the previous beat's T: max(300 ms,
    #     45% RR) after its R, up to 300 ms before this R) is quiet:
    #     detrended peak-to-peak < 0.7 x the median P stretch's own.
    #     One repeated P wave on an otherwise flat baseline. Fibrillatory
    #     waves don't repeat at a fixed place before each QRS; Flutter waves
    #     do (fixed conduction), but they fill the diastole too -- measured
    #     on the Flutter test recordings, diastole / P stretch 0.93-0.97.
    #     (Diastole from 38% RR instead still held the end of the T wave:
    #     with a small P wave (V4, P ~3% of QRS) that alone failed the quiet
    #     test on clean sinus beats.)
    # A beat is sinus when it is on any lead; undecided when it isn't, and
    # every lead where its P stretch matched had no diastole left to judge
    # (rate ~100/min and up: T runs into P); otherwise fibrillatory / flutter.
    # Undecided beats don't vote; with fewer than 3 deciding beats the old
    # rule's label stands.
    # Measured over every window the old rules called (all recordings):
    #   AFib  -- sinus share of deciding beats in windows kept in the output:
    #            at most 33%; the human recordings' false windows: >= 50% in
    #            all but 2 lone windows (dropped by the lone-window guard).
    #   Flutter -- 0% on the test recordings.
    #   VFib  -- the lead with the most organized beats: at most 56% in
    #            windows kept in the output; 75% and 83% (Lead II) in the
    #            human recording's two false windows.
    _BEAT_QRS_CORR = 0.9
    _BEAT_P_CORR = 0.7
    _BEAT_DIASTOLE_TO_P = 0.7
    _BEAT_SINUS_MAJORITY = 0.5
    _BEAT_MIN_DECIDING = 3
    _BEAT_FLUTTER_LOCKED = 0.4
    _BEAT_VFIB_ORGANIZED = 0.7

    def _detrended(x):
        x = np.asarray(x, dtype=float)
        if x.size < 2:
            return x - (float(x.mean()) if x.size else 0.0)
        idx = np.arange(x.size)
        return x - np.polyval(np.polyfit(idx, x, 1), idx)

    def _beat_by_beat_evidence(padded_array, r_peaks, fs_):
        """Per beat of r_peaks: 'atrial' -- 'sinus' / 'undecided' /
        'fibrillatory' (None for the first beat, which has no diastole);
        'organized_by_lead' -- per lead, the organized-QRS flag per beat."""
        n = len(r_peaks)
        sinus = [False] * n
        locked_leads = [0] * n          # leads whose P stretch matched (corr only)
        p_match_judged = [False] * n    # P stretch matched on a lead with diastole to judge
        p_match_unjudged = [False] * n  # ... on a lead without
        organized_by_lead = []
        h, lag = int(0.06 * fs_), int(0.04 * fs_)
        p_lo, p_hi = int(0.30 * fs_), int(0.07 * fs_)
        for li in range(padded_array.shape[0]):
            x = np.asarray(padded_array[li], dtype=float)
            if _atrial_preproc is not None:
                try:
                    x = np.asarray(_atrial_preproc.process(x), dtype=float)
                except Exception:
                    pass
            organized = [False] * n
            organized_by_lead.append(organized)
            fits = [k for k, q in enumerate(r_peaks) if q - h - lag >= 0 and q + h + lag <= x.size]
            if len(fits) < 3:
                continue
            snips = [x[int(r_peaks[k]) - h:int(r_peaks[k]) + h] for k in fits]
            template = np.median(np.asarray([s - float(np.mean(s)) for s in snips]), axis=0)
            t_norm = float(np.linalg.norm(template))
            if t_norm <= 0:
                continue
            aligned_r = {}
            for k in fits:
                q = int(r_peaks[k])
                best, best_shift = -1.0, 0
                for shift in range(-lag, lag + 1, 2):
                    seg = x[q + shift - h:q + shift + h]
                    seg = seg - float(np.mean(seg))
                    nrm = float(np.linalg.norm(seg))
                    if nrm > 0:
                        c = float(np.dot(seg, template)) / (nrm * t_norm)
                        if c > best:
                            best, best_shift = c, shift
                aligned_r[k] = q + best_shift
                if best >= _BEAT_QRS_CORR:
                    beat = measure_beat(x, q, fs_)
                    organized[k] = bool(beat and beat.get('qrs_bounds_resolved'))

            p_segs = {k: _detrended(x[r - p_lo:r - p_hi]) for k, r in aligned_r.items() if r - p_lo >= 0}
            if len(p_segs) < 4:
                continue
            p_template = np.median(np.asarray(list(p_segs.values())), axis=0)
            p_norm = float(np.linalg.norm(p_template))
            p_ptp = float(np.ptp(p_template))
            if p_norm <= 0 or p_ptp <= 0:
                continue
            for k in range(1, n):
                if k not in p_segs or (k - 1) not in aligned_r:
                    continue
                seg = p_segs[k]
                nrm = float(np.linalg.norm(seg))
                if nrm <= 0 or float(np.dot(seg, p_template)) / (nrm * p_norm) < _BEAT_P_CORR:
                    continue
                locked_leads[k] += 1
                if sinus[k]:
                    continue
                r_prev, r_cur = aligned_r[k - 1], aligned_r[k]
                d_start = r_prev + max(int(0.30 * fs_), int(0.45 * (r_cur - r_prev)))
                d_end = r_cur - p_lo
                if d_end - d_start < int(0.04 * fs_):
                    p_match_unjudged[k] = True  # no diastole left at this rate
                    continue
                p_match_judged[k] = True
                if float(np.ptp(_detrended(x[d_start:d_end]))) < _BEAT_DIASTOLE_TO_P * p_ptp:
                    sinus[k] = True
        atrial = [None] + [
            'sinus' if sinus[k]
            else 'undecided' if (p_match_unjudged[k] and not p_match_judged[k])
            else 'fibrillatory'
            for k in range(1, n)
        ]
        return {'atrial': atrial, 'organized_by_lead': organized_by_lead,
                'locked_leads': locked_leads, 'n_leads': padded_array.shape[0]}

    def _flutter_waves_locked(padded_array, r_peaks, fs_):
        """True when >= 40% of the window's beats have the stretch before
        their QRS matching on at least half the leads.

        Flutter waves are conducted at a fixed ratio, so each QRS sits at
        the same place in the F wave cycle on every lead -- measured on the
        Flutter test recordings: 50-100% of beats (mostly 100%) matched on
        >= 6 of 12 leads. Motion artifact runs independently of the heart:
        a 2 min human recording with ~5 Hz artifact on I / II / V1 read as
        Flutter for 47 s with only 6-24% of beats matching."""
        if len(r_peaks) < 3:
            return True
        ev = _beat_by_beat_evidence(padded_array, r_peaks, fs_)
        locked = sum(1 for c in ev['locked_leads'][1:] if c * 2 >= ev['n_leads'])
        return locked >= _BEAT_FLUTTER_LOCKED * (len(r_peaks) - 1)

    def _atrial_beats_mostly_sinus(padded_array, r_peaks, fs_):
        """True when >= half the window's deciding beats are sinus beats --
        the window is not AFib / Flutter."""
        if len(r_peaks) < 3:
            return False
        atrial = _beat_by_beat_evidence(padded_array, r_peaks, fs_)['atrial']
        n_sinus = atrial.count('sinus')
        n_deciding = n_sinus + atrial.count('fibrillatory')
        return n_deciding >= _BEAT_MIN_DECIDING and n_sinus >= _BEAT_SINUS_MAJORITY * n_deciding

    # VFib needs a fibrillating stretch somewhere in a slow window: a 2 s
    # piece (1 s steps) where, on the median of the 12 leads, the signal sits
    # still for < _VFIB_STILL_SHARE of the time. VFib is one unbroken wave
    # with no isoelectric baseline; sinus, escape and bundle branch block
    # beats sit on a flat baseline between them. Measured over every VFib
    # window of the recordings: real VFib pieces 0.00-0.05, while windows of
    # separate slow beats called VFib had no such piece at all (seen: two
    # escape beats and RBBB beats in one window, 65-75 s, RR 930 -> 2248 ->
    # 750 ms and wide QRS unresolved on 9/11 beats -> VFib; still share
    # 0.20-0.57). Judged only where the window's median RR is over
    # _VFIB_STILL_MIN_RR_MS (under 100 bpm): fast wide complexes with short
    # flat gaps between them (seen: polymorphic wide complexes ~130 bpm,
    # still share 0.15-0.27) are left to the VFib rule as before.
    # "Still": 0.5-25 Hz signal (50/60 Hz notched) moving under one piece
    # range per second for >= 80 ms in a row.
    _VFIB_STILL_SHARE = 0.10
    _VFIB_STILL_MIN_RR_MS = 600.0

    def _vfib_filtered(x, fs_):
        from scipy.signal import butter as _bt, filtfilt as _ff
        try:
            from ..enhanced_filter import make_notch as _mk_notch
        except ImportError:
            from ecg.enhanced_filter import make_notch as _mk_notch
        y = np.asarray(x, dtype=float)
        for mains_hz in (50.0, 60.0):
            if mains_hz < fs_ / 2.0:
                bq = _mk_notch(mains_hz, fs_)
                y = _ff(bq.b, bq.a, y)
        b, a = _bt(2, [0.5 / (fs_ / 2.0), 25.0 / (fs_ / 2.0)], 'band')
        return _ff(b, a, y)

    def _vfib_still_share(y, fs_):
        """Still share of an already filtered piece (_vfib_filtered)."""
        rng = float(np.percentile(y, 99) - np.percentile(y, 1))
        if rng <= 0:
            return 1.0
        still = np.abs(np.gradient(y)) * fs_ < rng
        run = int(0.08 * fs_)
        flat = 0
        i, n = 0, still.size
        while i < n:
            if still[i]:
                j = i
                while j < n and still[j]:
                    j += 1
                if j - i >= run:
                    flat += j - i
                i = j
            else:
                i += 1
        return flat / float(n)

    def _vfib_piece_fibrillating(filtered_leads, start, length, fs_):
        shares = [_vfib_still_share(y[start:start + length], fs_) for y in filtered_leads]
        return float(np.median(shares)) < _VFIB_STILL_SHARE

    def _vfib_has_fibrillating_piece(core_array, r_peaks, fs_):
        """True when the window is fast (median RR <= _VFIB_STILL_MIN_RR_MS,
        or under 3 beats to tell) or some 2 s piece of it (all leads,
        unpadded) is fibrillating -- see above. A window under 2 s is not
        judged (True)."""
        if len(r_peaks) >= 3 and float(np.median(np.diff(np.asarray(r_peaks, dtype=float)))) * 1000.0 / fs_                 <= _VFIB_STILL_MIN_RR_MS:
            return True
        if len(r_peaks) < 3:
            return True
        piece, n = int(2.0 * fs_), core_array.shape[1]
        if n < piece:
            return True
        filtered = [_vfib_filtered(core_array[i], fs_) for i in range(core_array.shape[0])]
        for start in range(0, n - piece + 1, int(fs_)):
            if _vfib_piece_fibrillating(filtered, start, piece, fs_):
                return True
        return False

    def _vfib_beats_organized(padded_array, r_peaks, fs_):
        """True when one lead shows an organized QRS on >= 70% of the
        window's beats -- the window is not VFib."""
        if len(r_peaks) < 3:
            return False
        by_lead = _beat_by_beat_evidence(padded_array, r_peaks, fs_)['organized_by_lead']
        return any(sum(flags) >= _BEAT_VFIB_ORGANIZED * len(r_peaks) for flags in by_lead)

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
    #   1. at least 6 beats, rate 180-300/min (see the rate check below);
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
        # 180, not LITFL's 140: tried 140 (AVNRT 140-280), and the P-on-T
        # test below then read the Sinus Tachycardia test recordings
        # (st_check22 / st_check_23, 150-170 bpm) as SVT -- 26 s and 65 s.
        # _svt_min_hr: 220 - age (book), 180 without an age.
        if not (_svt_min_hr <= rate <= 300.0):
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
        if rate >= _svt_min_hr:
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
    _AV_TWO_TO_ONE = 'Second-degree AV Block (2:1)'
    _AV_THIRD = 'Third-degree AV Block'
    _AV_COLORS = {
        _AV_FIRST: '#FFD700',      # gold
        _AV_MOBITZ_I: "#3D3AD4",   # royal blue
        _AV_MOBITZ_II: '#FF00FF',  # magenta
        _AV_TWO_TO_ONE: '#DA70D6',  # orchid
        _AV_THIRD: '#C0FF00',      # chartreuse
    }
    # Most severe first. A region of a less severe block is cut wherever it
    # overlaps a more severe one (or any non-AV rhythm), see below.
    _AV_LABELS = (_AV_THIRD, _AV_MOBITZ_II, _AV_TWO_TO_ONE, _AV_MOBITZ_I, _AV_FIRST)
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
        # First-degree AV Block only: one missed P (a real P right at the
        # window's own edge, often just outside where the walk still has a
        # neighbour to anchor to) or one false-positive P-like bump (a T
        # wave matching the template, _is_first_degree_av_block_window's own
        # events/qrs_pr are unaffected by an extra -- it is simply never
        # part of the chain) must not fail the whole window when the P-P
        # train is otherwise clean. _is_first_degree_av_block_window still
        # separately requires every QRS conducted and not hidden, and >= 80%
        # of beats with a consistent PR, so a real blocked/hidden P (2nd/3rd
        # degree) still fails it regardless of this flag. atrial_regular
        # itself, and every other rule reading it, is unchanged. Seen: PR
        # 240-258 ms, RR 750 ms throughout, 80 bpm -- one window with the
        # first P undetected (a real mismatch at the window's leading edge)
        # and the next with one extra T-wave match, neither close to
        # mismatches == 0, so the whole 11 s read as no AV label at all.
        atrial_regular_first_degree_block = atrial_regular or (
            mismatches <= 1 and len(extras) <= 1 and pp_cv <= 0.12
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
            'atrial_regular_first_degree_block': atrial_regular_first_degree_block,
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
        if not ctx['atrial_regular_first_degree_block']:
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

    def _blocked_p_cycles(ctx, allow_double=False):
        """Each blocked (non-conducted) P wave with the conducted beats right
        before it and the first conducted beat after it.

        Returns None when the window has a pattern neither Mobitz rule covers:
        a hidden blocked P (can't be confirmed), two blocked P waves in a row
        (high-grade block), or more blocked than conducted P waves.
        allow_double (Mobitz II only): two blocked P waves in a row are one
        cycle (pause 3x the P-P interval); three in a row still return None.
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
            if allow_double and i > 0 and events[i - 1]['conducted'] is False:
                continue  # second P of a double block, judged with the first
            if i + 1 < len(events) and events[i + 1]['conducted'] is False:
                if not allow_double or events[i + 1]['hidden']:
                    return None
                if i + 2 < len(events) and events[i + 2]['conducted'] is False:
                    return None
            # Last blocked P of the cycle: the cycle's first conducted beat
            # comes after it.
            last = i + 1 if allow_double and i + 1 < len(events) and events[i + 1]['conducted'] is False else i
            run = []
            j = i - 1
            while j >= 0 and events[j]['conducted'] and events[j]['pr_ms'] is not None:
                run.insert(0, events[j])
                j -= 1
            # Complete = the run starts right after the previous blocked P,
            # not cut off by the window's own start.
            complete = j >= 0 and events[j]['conducted'] is False
            after = None
            if last + 1 < len(events) and events[last + 1]['conducted'] and events[last + 1]['pr_ms'] is not None:
                after = events[last + 1]
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
    # trend, so it can't be told Mobitz I from Mobitz II: it is its own label,
    # _AV_TWO_TO_ONE (as Hampton), see _is_two_to_one_av_block_window. Was:
    # reported as Mobitz II.
    # Returns (R before the conducted run's last two beats, first R after the
    # last blocked P), or None.
    # allow_double (Mobitz II only): also takes two blocked P waves in a row
    # after a fixed-PR run of >= 2 beats, pause 3x the P-P interval
    # (+/-10%). Seen: RBBB at 75 bpm, PR fixed, 2250 ms pauses with two
    # non-conducted P waves, which the 2x-only check left unlabelled. A
    # double block after a single beat (3:1) is not judged here.
    def _fixed_pr_block_spans(ctx, allow_double=False):
        """(Mobitz II spans, 2:1 spans) of the window's blocked-P cycles with
        a fixed PR and a doubled pause, or None when a complete cycle breaks
        that pattern."""
        if not ctx['atrial_regular']:
            return None
        cycles = _blocked_p_cycles(ctx, allow_double=allow_double)
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
            if allow_double and len(run) >= 2 and 2.7 * ctx['pp_ms'] <= pause_ms <= 3.3 * ctx['pp_ms']:
                doubled = True
            if not (fixed and doubled):
                if len(run) >= 2 and complete:
                    return None
                continue
            span = (run[max(0, len(run) - 2)]['qrs'], after['qrs'])
            (spans if len(run) >= 2 else two_to_one).append(span)
        return spans, two_to_one

    def _is_mobitz_ii_window(ctx):
        found = _fixed_pr_block_spans(ctx, allow_double=True)
        if not found or not found[0]:
            return None
        spans = found[0]
        return min(s for s, _ in spans), max(e for _, e in spans)

    # Second-degree AV Block 2:1: every other P wave blocked, PR of the
    # conducted beats fixed, pause 2x the P-P interval; the pattern repeats
    # (>= 2 such cycles in the window). Checked after Mobitz II.
    def _is_two_to_one_av_block_window(ctx):
        found = _fixed_pr_block_spans(ctx)
        if not found or len(found[1]) < 2:
            return None
        spans = found[1]
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
            (_AV_TWO_TO_ONE, _is_two_to_one_av_block_window),
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

    def _bundle_branch_block_beat_pr(lead_ii, r, prev_rr, fs_):
        """One beat's P / PR on Lead II for the bundle branch block rules:
        (has_p, pr_ms), or None when the beat can't be measured. See the
        P / PR comments in _bundle_branch_block_window_beats."""
        n = lead_ii.size
        half_sec = int(fs_)
        lo, hi = max(0, r - half_sec), min(n, r + half_sec)
        seg_ii = lead_ii[lo:hi]
        a_lo = max(0, r - lo - int(0.04 * fs_))
        a_hi = min(seg_ii.size, r - lo + int(0.10 * fs_))
        if a_hi <= a_lo:
            return None
        anchor = a_lo + int(np.argmax(np.abs(seg_ii[a_lo:a_hi] - np.median(seg_ii))))
        b = measure_beat(seg_ii, anchor, fs_, strict_pr=True, prev_rr_ms=prev_rr)
        if (b is None or b.get('pr_ms') is None) and prev_rr is not None and prev_rr >= 500.0:
            b_relaxed = measure_beat(seg_ii, anchor, fs_, strict_pr=False, prev_rr_ms=prev_rr)
            if b_relaxed is not None and b_relaxed.get('pr_ms') is not None:
                b = b_relaxed
        if b and b.get('p_present') and b.get('p_onset') is not None:
            return True, (float(b['pr_ms']) if b.get('pr_ms') is not None else None)
        return False, None

    def _bundle_branch_block_beat_width_shape(padded_array, r, fs_):
        """One beat's global QRS width (ms) and its (right_pattern,
        left_pattern) shape, or (width, None) when no lateral lead could be
        read, or None when the beat can't be measured. See the QRS duration
        and shape comments in _bundle_branch_block_window_beats."""
        n = padded_array.shape[1]
        half_sec = int(fs_)
        quiet = max(1, int(0.016 * fs_))
        lo, hi = max(0, r - half_sec), min(n, r + half_sec)
        measured = {}
        for li in set(_bundle_branch_block_width_idx) | {_bundle_branch_block_v1_idx} | set(_bundle_branch_block_lateral_idx):
            seg = np.asarray(padded_array[li][lo:hi], dtype=float)
            # Same anchoring as Lead II: this lead's own dominant QRS
            # deflection near the detected beat.
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
            return None
        # Combined slope of the width leads, each scaled by its own QRS
        # size; its peak is searched 80 ms before to 120 ms after the
        # detected beat, onset/offset within 150 ms before / 200 ms after.
        r_rel = r - lo
        w_lo, w_hi = r_rel - int(0.15 * fs_), r_rel + int(0.20 * fs_)
        if w_lo < 0 or w_hi > hi - lo:
            return None
        velocity = np.zeros(w_hi - w_lo)
        for li in width_leads:
            x = measured[li][w_lo:w_hi]
            if x.size >= 11:
                x = _savgol_11_3(x)
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
            return None
        width = (offset - onset) * 1000.0 / fs_
        onset += w_lo
        offset += w_lo

        def qrs_of(li):
            seg = measured[li]
            # Each lead's own baseline: the PR segment just before QRS onset.
            base = float(np.median(seg[max(0, onset - int(0.06 * fs_)):max(1, onset - int(0.01 * fs_))]))
            return seg[onset:offset + 1] - base

        v1_right, v1_left = _v1_bundle_branch_block_shape(qrs_of(_bundle_branch_block_v1_idx), fs_)
        lateral = [_lateral_bundle_branch_block_shape(qrs_of(li), fs_) for li in _bundle_branch_block_lateral_idx if li in measured]
        if not lateral:
            return width, None
        return width, (v1_right and any(s for s, _ in lateral), v1_left and all(r_ for _, r_ in lateral))

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
        # (Both searches: _bundle_branch_block_beat_pr.)
        lead_ii = np.asarray(padded_array[conduction_lead_idx], dtype=float)
        prs, with_p = [], 0
        for k, r in enumerate(core):
            prev_rr = (r - core[k - 1]) * 1000.0 / fs_ if k else None
            beat_p = _bundle_branch_block_beat_pr(lead_ii, r, prev_rr, fs_)
            if beat_p is None:
                continue
            has_p, pr = beat_p
            if has_p:
                with_p += 1
                if pr is not None:
                    prs.append(pr)
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
        # (Per beat: _bundle_branch_block_beat_width_shape.)
        widths, shapes_per_beat, shape_peaks = [], [], []
        for r in core:
            width_shape = _bundle_branch_block_beat_width_shape(padded_array, r, fs_)
            if width_shape is None:
                continue
            widths.append(width_shape[0])
            if width_shape[1] is None:
                continue
            shapes_per_beat.append(width_shape[1])
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

    # Parallel window loop (see _PARALLEL_* above): a worker judges only its
    # own windows; the main process takes all workers' votes, in window
    # order, instead of running the loop itself.
    window_loop_end = total_duration
    if _window_job is not None:
        current_time = _window_job[0] * step_size
        window_loop_end = min(total_duration, _window_job[1] * step_size)
    elif _parallel_window_handle is not None:
        job_votes = None
        if step_size == _PARALLEL_WINDOW_STEP_SEC:
            job_votes = _collect_parallel_window_loop(_parallel_window_handle, progress_callback)
        else:
            _parallel_window_handle[0].shutdown(wait=False, cancel_futures=True)
        if job_votes is not None:
            for votes in job_votes:
                for candidate, svt_rate in zip(votes['morphological_candidates'], votes['svt_window_rate']):
                    morphological_candidates.append(candidate)
                    if svt_rate is not None:
                        _svt_window_rate[id(candidate)] = svt_rate
                _coarse_vfib_window_starts.update(votes['coarse_vfib_window_starts'])
                bundle_branch_block_candidates.extend(votes['bundle_branch_block_candidates'])
                for beat_time, beat_votes in votes['bundle_branch_block_beat_votes']:
                    bundle_branch_block_beat_votes.setdefault(beat_time, []).extend(beat_votes)
            current_time = total_duration   # windows already judged

    while current_time < window_loop_end:
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
                        # The same beats without the ones Pan-Tompkins adds
                        # by shape match at a beat-size step: the VFib rule
                        # below is judged on these (see STEP_SHAPE_* in
                        # pan_tompkins -- shape cannot tell VFib waves from
                        # beats).
                        vfib_peak_times = []
                        lead_peak_pairs = []
                        for lead_idx in peak_detection_lead_indices:
                            if lead_idx >= padded_array.shape[0]:
                                continue
                            lead_sig = np.asarray(padded_array[lead_idx], dtype=float)
                            if len(lead_sig) <= 50:
                                continue
                            lead_peak_pairs.append(pan_tompkins(lead_sig, fs=fs, split_shape_match=True))
                        for k, (lead_peaks, lead_vfib_peaks) in enumerate(lead_peak_pairs):
                            if len(lead_peaks) != len(lead_vfib_peaks):
                                lead_peaks = _keep_confirmed_shape_beats(
                                    lead_peaks, lead_vfib_peaks,
                                    [p for j, (p, _) in enumerate(lead_peak_pairs) if j != k], fs)
                            all_peak_times.extend(lead_peaks)
                            vfib_peak_times.extend(lead_vfib_peaks)

                        def _window_core_peaks(peak_times):
                            merged = []
                            for p in sorted(peak_times):
                                if merged and (p - merged[-1]) < merge_tol_samples:
                                    continue
                                merged.append(p)
                            return merged, [p for p in merged
                                            if current_time <= pad_start + p / fs < window_end]

                        all_peaks, core_r_peaks = _window_core_peaks(all_peak_times)
                        _, vfib_core_r_peaks = _window_core_peaks(vfib_peak_times)
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
                        # The VT and VFib rules measure beats on the strip's
                        # own filtering (50 + 60 Hz notch, 0.5 Hz DFT high-
                        # pass), not the raw samples: mains ripple and
                        # baseline wander the strip does not show must not
                        # read as an unresolved QRS / missing P. R-peaks
                        # above stay from the raw signal.
                        rule_signal = _mains_dft_filter(padded_signal)
                        # Where the shape match added beats to this window,
                        # whether it is VFib is decided exactly as without
                        # them (the VT check first, then the VFib checks
                        # below, all on vfib_core_r_peaks); None = no such
                        # beats here, the chain below runs as it always did.
                        vfib_without_shape_beats = None
                        if [int(p) for p in vfib_core_r_peaks] != [int(p) for p in core_r_peaks]:
                            vfib_without_shape_beats = bool(
                                not _is_ventricular_tachycardia_window(rule_signal, vfib_core_r_peaks, fs)
                                and _is_vfib_window(rule_signal, vfib_core_r_peaks, fs)
                                and _classify_bundle_branch_block_window(
                                    padded_array, vfib_core_r_peaks, padded_signal, fs) is None
                                and not _vfib_beats_organized(padded_array, vfib_core_r_peaks, fs)
                                and _vfib_has_fibrillating_piece(data_array, vfib_core_r_peaks, fs))
                        # Resolve the organized-wide-complex overlap first:
                        # the same unresolved QRS flag can otherwise make a
                        # regular monomorphic VT window look like VFib.
                        if not vfib_without_shape_beats and _is_ventricular_tachycardia_window(rule_signal, core_r_peaks, fs):
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
                        # Nor where one lead shows an organized QRS on most
                        # beats (_vfib_beats_organized, beat by beat on all
                        # 12 leads); such a window goes on through the chain.
                        # And only where the window holds a fibrillating
                        # stretch (_vfib_has_fibrillating_piece).
                        elif (vfib_without_shape_beats if vfib_without_shape_beats is not None else (
                                _is_vfib_window(rule_signal, core_r_peaks, fs)
                                and bundle_branch_block_result is None
                                and not _vfib_beats_organized(padded_array, core_r_peaks, fs)
                                and _vfib_has_fibrillating_piece(data_array, core_r_peaks, fs))):
                            target_morph = ('Ventricular Fibrillation', '#FF3333')
                            # Called VFib on fewer than 4 measured beats: same
                            # neighbor guard (seen: the recording's last,
                            # 3 s window held 2 cut-off normal beats -- read
                            # as "no P, no organized QRS" -- and became VFib).
                            if _vfib_window_kind['last'] == 'coarse' or _vfib_window_kind['beats'] < 4:
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
                            # AFib / Flutter only when most beats show the
                            # fibrillatory / flutter pattern, not a sinus P
                            # (_atrial_beats_mostly_sinus, beat by beat).
                            if atrial_label and _atrial_beats_mostly_sinus(padded_array, core_r_peaks, fs):
                                atrial_label = None
                            # Flutter waves must be locked to the QRS on most
                            # leads (_flutter_waves_locked), not artifact.
                            if atrial_label == 'Atrial Flutter' \
                                    and not _flutter_waves_locked(padded_array, core_r_peaks, fs):
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

    if _window_job is not None:
        return {
            'morphological_candidates': morphological_candidates,
            'svt_window_rate': [_svt_window_rate.get(id(m)) for m in morphological_candidates],
            'coarse_vfib_window_starts': sorted(_coarse_vfib_window_starts),
            'bundle_branch_block_candidates': bundle_branch_block_candidates,
            'bundle_branch_block_beat_votes': list(bundle_branch_block_beat_votes.items()),
        }

    # VFib neighbor guard: a window called VFib only by the shape test in
    # _is_vfib_window, or called VFib on fewer than 4 measured beats, is
    # dropped unless another VFib window starts within one window of it --
    # one lone window is not an episode (the missed VFib the shape test is
    # for ran 16+ windows in a row; real VFib windows with 2-3 beats sat next
    # to other VFib windows, at a VFib's end or inside it).
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
    # One RR interval judged on its own for Atrial Flutter: its sawtooth F
    # waves repeat at a steady ~200 ms (atrial rate ~240-350/min), so the
    # interval's autocorrelation has a clear peak at a 170-250 ms lag; AFib
    # f-waves are irregular (measured <= 0.27 at that lag on the test
    # recordings, Flutter >= 0.47). True / False, or None for an interval too
    # short to judge (fast 2:1 Flutter). Used by the isolated-window guard,
    # the AFib / Flutter edge and the beat-by-beat AFib / Flutter step below.
    def _interval_is_flutter(prev_sec, peak_sec):
        s_sec, e_sec = prev_sec + 0.12, peak_sec - 0.08
        if e_sec - s_sec < 0.4:
            return None
        # Read 0.5 s wider and notch 50 / 60 Hz (as the sinus gate does)
        # before cutting the interval out -- seen: 3:1 Flutter under 60 Hz
        # mains read as no F wave cycle here, and so as AFib, beat by beat.
        pad_lo = min(0.5, s_sec)
        sig = reader.read_range(s_sec - pad_lo, e_sec + 0.5)
        if sig is None or best_lead_idx >= sig.shape[0]:
            return None
        wide = np.asarray(sig[best_lead_idx], dtype=float)
        if _make_notch is not None:
            for _mains_hz in (50.0, 60.0):
                if _mains_hz < float(fs) / 2.0:
                    try:
                        _bq = _make_notch(_mains_hz, float(fs))
                        wide = _filtfilt(_bq.b, _bq.a, wide)
                    except Exception:
                        pass
        i0 = int(round(pad_lo * fs))
        seg = wide[i0:i0 + int(round((e_sec - s_sec) * fs))]
        if len(seg) < int(0.4 * fs):
            return None
        n = np.arange(len(seg))
        seg = seg - np.polyval(np.polyfit(n, seg, 1), n)
        ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
        if ac[0] <= 0:
            return None
        ac = ac / ac[0]
        lo_lag, hi_lag = int(0.15 * fs), int(0.35 * fs)
        lag = lo_lag + int(np.argmax(ac[lo_lag:hi_lag]))
        return bool(ac[lag] >= 0.4 and 0.17 * fs <= lag <= 0.25 * fs)

    _FLUTTER_MIN_INTERVALS = 4

    def _flutter_run_in(t0, t1):
        """True when >= _FLUTTER_MIN_INTERVALS consecutive RR intervals in
        [t0, t1] are Flutter intervals (_interval_is_flutter)."""
        peaks = sorted(float(p) for p in all_r_peaks if t0 <= p <= t1)
        run = 0
        for a, b in zip(peaks, peaks[1:]):
            run = run + 1 if _interval_is_flutter(a, b) else 0
            if run >= _FLUTTER_MIN_INTERVALS:
                return True
        return False

    def _svt_run_in(t0, t1):
        """True when the beats of all_r_peaks (the whole-recording beat
        list, not this window's own core_r_peaks) inside [t0, t1] still
        pass _is_atrial_tachycardia_window's own checks (rate, RR
        regularity, one QRS shape, P-on-T) on their own, independent of any
        neighboring window. Mirrors _flutter_run_in's exception for the same
        lone-window guard below: a short SVT run can fill only one window
        with no same-label neighbor to confirm it (seen: 13 beats at a
        steady 300 ms RR, 59.8-63.7 s, right at a recording's own end --
        the one window before it mixed a decelerating AFib tail with the
        new fast beats and failed the regularity check, so the real SVT
        run never got a second window's vote and was dropped, falling back
        to the bare 'Tachycardia' rate label)."""
        peaks = sorted(float(p) for p in all_r_peaks if t0 - 0.01 <= p <= t1 + 0.01)
        if len(peaks) < 6:
            return False
        pad_start = max(0.0, peaks[0] - 1.0)
        pad_end = min(total_duration, peaks[-1] + 1.0)
        sig = reader.read_range(pad_start, pad_end)
        if sig is None or best_lead_idx >= sig.shape[0]:
            return False
        local_peaks = [int(round((p - pad_start) * fs)) for p in peaks]
        try:
            return bool(_is_atrial_tachycardia_window(
                np.asarray(sig[best_lead_idx], dtype=float), local_peaks, fs))
        except Exception:
            return False

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
            # A lone Flutter window is kept when its own beats are Flutter
            # beat by beat (>= 4 consecutive Flutter RR intervals): a short
            # episode fills only one window (seen: 4:1 Flutter, RR 1000 ms,
            # 1185.5-1192.6 s, one Flutter window 1185-1195 s between AFib
            # windows -- dropped, and the AFib before it ran on to 1193.4 s).
            # A lone Supraventricular Tachycardia (SVT) window gets the same
            # exception (_svt_run_in): kept when all_r_peaks' own beats in
            # it still pass the SVT window rule on their own (see
            # _svt_run_in's docstring). The VFib on-ramp windows this guard
            # is for have no steady 170-250 ms F-wave cycle and fail SVT's
            # own rate/regularity/QRS-shape checks the same way.
            # NOTE: a matching lone-Atrial-Fibrillation exception
            # (_afib_run_in, same shape as the two above) was tried and
            # reverted -- confirmed directly it also un-drops a lone AFib
            # window that is itself a false call on a genuine grouped
            # Mobitz I run (PR cycling 164/224/368/dropped in a repeating
            # 4-beat pattern, read as "irregular enough" by the window
            # rule's own AFib check), which then merged with a real AFib
            # window 20 s later into one wrong AFib span covering both.
            # Unlike Flutter/SVT, whose own lone-window false positives are
            # specifically VFib on-ramps (too chaotic to reach
            # _classify_atrial_window's QRS-resolved entry gate at all),
            # AFib's lone-window false positives include exactly this kind
            # of regular-but-grouped rhythm, which does clear that gate.
            # _afib_run_in re-checking the same window's own beats with the
            # same rule that was already wrong about them confirms nothing.
            if _prev_match or _next_match or (
                    _m['label'] == 'Atrial Flutter' and _flutter_run_in(_m['start_sec'], _m['end_sec'])) or (
                    _m['label'] == 'Supraventricular Tachycardia (SVT)'
                    and _svt_run_in(_m['start_sec'], _m['end_sec'])):
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

        # A sustained FLATLINE (no cardiac activity at all -- not chaotic
        # activity) sitting right at a VFib region's own start must not be
        # credited to VFib. _is_vfib_window() only ever looks at the beats
        # Pan-Tompkins actually found inside its 10s window, so a silent
        # run with zero detected beats is invisible to it and simply gets
        # inherited whole into whichever window it falls inside -- confirmed
        # directly on a recording where ~5s of true flatline right after a
        # Ventricular Tachycardia run, followed a few seconds later by the
        # first chaotic beats, got drawn as VFib starting at the window
        # edge (the flat stretch itself), not at the first chaotic beat.
        # The _refine_vfib_edge()/_silent_gap_start() search above already
        # tries to snap the start to a real beat transition, but it needs an
        # "organized" reference beat before the gap and (with
        # require_narrow=True, as used coming out of a VT run) a wide VT
        # beat never qualifies as organized -- so it has nothing to search
        # from and leaves the boundary at the window edge.
        #
        # This walks forward from the already-refined start in 1s blocks,
        # using the signal's own amplitude in the 2s right before the region
        # (the rhythm this VFib call followed) as the reference for what
        # "normal" looks like here -- same scaling detect_asystole_segments()
        # uses, just computed locally instead of over the whole recording,
        # since this only ever needs to look at one region's own edge. Only
        # acts on a flat stretch at least _ASYSTOLE_MIN_SEC long (the same
        # bar used to call something Asystole at all -- see the Asystole
        # section of this module), so an ordinary quiet beat gap inside
        # VFib's own chaos is never mistaken for this. If trimming it away
        # would leave less than 2s of region, the trim is skipped instead
        # (a boundary call this module isn't confident enough to make) --
        # the existing refinement above still stands. Only a Ventricular
        # Fibrillation region's own start can move, only later, and only
        # over a confirmed flat stretch; the VFib rule and every other
        # label are untouched.
        def _vfib_flatline_start_trim(start_sec, end_sec):
            if best_lead_idx is None or fs <= 0:
                return start_sec

            def _ptp_std(t0, t1):
                seg = reader.read_range(max(0.0, t0), min(total_duration, t1))
                if seg is None or best_lead_idx >= seg.shape[0]:
                    return None
                x = np.asarray(seg[best_lead_idx], dtype=float)
                if x.size == 0:
                    return None
                return float(np.ptp(x)), float(np.std(x))

            ref = _ptp_std(start_sec - 2.0, start_sec)
            if ref is None or ref[0] <= 0:
                return start_sec
            ref_ptp = ref[0]
            flat_ptp_limit = 0.08 * ref_ptp
            flat_std_limit = 0.04 * ref_ptp

            t = start_sec
            flat_run = 0.0
            while t < end_sec - 1.0:
                blk = _ptp_std(t, t + 1.0)
                if blk is None or not (blk[0] < flat_ptp_limit and blk[1] < flat_std_limit):
                    break
                t += 1.0
                flat_run += 1.0
            if flat_run < _ASYSTOLE_MIN_SEC:
                return start_sec
            # `t` is only accurate to the whole second: the walk above stops
            # at the first 1s block that ISN'T entirely flat, even when the
            # real flat-to-chaos transition sits partway through that block
            # -- confirmed directly on a recording where the signal was
            # flat (<1.5% of reference) right up to 295.3s and then rose
            # sharply (>20% of reference by 295.4s, >55% by 295.6s), while
            # this walk alone reported 295.0s, a full second early. Finish
            # the walk in smaller 0.2s steps from here (capped at one more
            # whole second) so the boundary lands within ~0.2s of where the
            # signal actually leaves the flat threshold instead of up to 1s
            # early. Same thresholds, just finer steps; nothing about what
            # counts as "flat" changes.
            fine_step = 0.2
            fine_limit = min(end_sec - fine_step, t + 1.0)
            tf = t
            while tf < fine_limit:
                blk = _ptp_std(tf, tf + fine_step)
                if blk is None or not (blk[0] < flat_ptp_limit and blk[1] < flat_std_limit):
                    break
                tf += fine_step
            t = tf
            if end_sec - t < 2.0:
                return start_sec
            return t

        def _vfib_pulled_start(new_start, old_start):
            """_refine_vfib_edge may move a VFib start back over beats it
            could not resolve (seen: 320.0 -> 310.1 s across clean 60 bpm
            beats carrying 50 Hz mains on the raw samples). A pull of more
            than 2 s is kept only as far back as the stretch is
            fibrillating: 1 s pieces in 0.25 s steps going back from the old
            start (_vfib_piece_fibrillating), the start set where the last
            fibrillating piece begins."""
            if old_start - new_start <= 2.0:
                return new_start
            read_start = max(0.0, new_start - 1.0)
            blk = reader.read_range(read_start, min(total_duration, old_start + 1.0))
            if blk is None or blk.shape[1] == 0:
                return new_start
            filtered = [_vfib_filtered(blk[i], fs) for i in range(blk.shape[0])]
            piece, step = int(fs), 0.25
            t = old_start
            while t - 1.0 >= new_start - 1e-6:
                i0 = int(round((t - 1.0 - read_start) * fs))
                if i0 < 0 or i0 + piece > blk.shape[1]                         or not _vfib_piece_fibrillating(filtered, i0, piece, fs):
                    break
                t -= step
            if t >= old_start:
                return old_start
            return max(new_start, t + step - 1.0)

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
                new_start = _vfib_pulled_start(new_start, region['start_sec'])
                if idx > 0 and merged_morph[idx - 1]['label'] == 'Ventricular Tachycardia' \
                        and region['start_sec'] - merged_morph[idx - 1]['end_sec'] <= window_size:
                    vt_last = _vt_run_last_beat(merged_morph[idx - 1], new_start)
                    if vt_last is not None and vt_last > new_start:
                        new_start = vt_last
                new_end = _refine_vfib_edge(region['end_sec'], 'end', require_narrow=not next_atrial)
                trimmed_start = _vfib_flatline_start_trim(new_start, new_end)
                flatline_trimmed = trimmed_start > new_start + 0.01
                new_start = trimmed_start
                if new_start < new_end - 0.5:
                    region['start_sec'], region['end_sec'] = new_start, new_end
                    region['start_time_str'], region['end_time_str'] = make_time_strs(new_start, new_end)
                    if flatline_trimmed:
                        # A confirmed flatline, not VFib's own chaos, now
                        # sits in the gap this trim opened up before this
                        # region's start (see _vfib_flatline_start_trim).
                        # Tagged so the generic gap-closing pass further
                        # below -- which otherwise stretches the PREVIOUS
                        # region across any gap up to one window long --
                        # leaves this one alone instead of re-absorbing the
                        # flatline into whatever rhythm preceded it. Only
                        # read where that pass checks this exact flag; every
                        # other label's gap-closing is unaffected.
                        region['_flatline_gap_before'] = True
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

        # A sustained FLATLINE at a Ventricular Tachycardia region's own END
        # is the same window-vote inheritance issue as
        # _vfib_flatline_start_trim above (see its own comment), just at
        # VT's trailing edge instead of VFib's leading edge: a window whose
        # core VT beats stop partway through still has its full
        # [start, window_end) span merged into the VT candidate, so a
        # silent run after those beats (confirmed directly: VT's own last
        # real beat at ~289.65s, then true flatline to ~295s, immediately
        # before the next window's first chaotic VFib beats) gets credited
        # to VT instead of being left out of every rhythm label -- which is
        # exactly what would happen here if only VFib's own leading edge
        # were trimmed: the flatline would simply end up inside VT instead
        # of VFib. Same reference-amplitude scaling and _ASYSTOLE_MIN_SEC
        # bar as that function (here taken from the 2s right AFTER this
        # region, i.e. whatever follows the flatline -- typically VFib's
        # own already-start-trimmed chaotic activity). Only a VT region's
        # own end can move, only earlier, over a confirmed flat stretch;
        # the VT rule itself is unchanged.
        def _vt_flatline_end_trim(start_sec, end_sec):
            if best_lead_idx is None or fs <= 0:
                return end_sec

            def _ptp_std(t0, t1):
                seg = reader.read_range(max(0.0, t0), min(total_duration, t1))
                if seg is None or best_lead_idx >= seg.shape[0]:
                    return None
                x = np.asarray(seg[best_lead_idx], dtype=float)
                if x.size == 0:
                    return None
                return float(np.ptp(x)), float(np.std(x))

            ref = _ptp_std(end_sec, end_sec + 2.0)
            if ref is None or ref[0] <= 0:
                return end_sec
            ref_ptp = ref[0]
            flat_ptp_limit = 0.08 * ref_ptp
            flat_std_limit = 0.04 * ref_ptp

            t = end_sec
            flat_run = 0.0
            while t > start_sec + 1.0:
                blk = _ptp_std(t - 1.0, t)
                if blk is None or not (blk[0] < flat_ptp_limit and blk[1] < flat_std_limit):
                    break
                t -= 1.0
                flat_run += 1.0
            if flat_run < _ASYSTOLE_MIN_SEC:
                return end_sec
            # Same whole-second-only limitation as
            # _vfib_flatline_start_trim's own walk, and the same fix: finish
            # in smaller 0.2s steps (backward here) from `t`, capped at one
            # more whole second, so this boundary lands within ~0.2s of
            # where the signal actually enters the flat threshold instead
            # of up to 1s late. Same thresholds, just finer steps.
            fine_step = 0.2
            fine_limit = max(start_sec + fine_step, t - 1.0)
            tf = t
            while tf > fine_limit:
                blk = _ptp_std(tf - fine_step, tf)
                if blk is None or not (blk[0] < flat_ptp_limit and blk[1] < flat_std_limit):
                    break
                tf -= fine_step
            t = tf
            if t - start_sec < 2.0:
                return end_sec
            return t

        # Beat-exact END of Ventricular Tachycardia, the mirror of
        # _vt_run_first_beat: the window vote carries a VT region on to its
        # last window's edge, over whatever beats follow the VT inside that
        # window (seen: wide 174 ms VT at 200 bpm to 363.84 s, then narrow
        # 54-64 ms beats at 60-120 bpm; VT drawn to the 370.0 s window edge,
        # 6 narrow beats inside it). The VT's own first window gives the
        # QRS template and RR (as _vt_run_first_beat); going forward from
        # there, a beat continues the VT while its shape matches (correlation
        # >= 0.90, size 0.5-2x) and its RR is within 15% of the VT's; the
        # VT ends on the last such beat before two in a row that don't.
        # Only when no other morphological region follows within a window
        # (VT -> VFib keeps its own handling) and only moving the end
        # earlier.
        def _vt_run_last_beat_fwd(vt_region):
            v_start, v_end = float(vt_region['start_sec']), float(vt_region['end_sec'])
            peaks = [p for p in _sorted_peaks if v_start - 0.05 <= p <= v_end + 0.05]
            ref = [p for p in peaks if p < min(v_end, v_start + window_size)]
            if len(ref) < 5:
                # The whole-recording R-peak pass can miss the VT beats
                # (seen: none at all in 344-363.8 s of a 200 bpm VT);
                # re-detect them on the region itself, as
                # _vt_run_first_beat does on its first window.
                r_start = max(0.0, v_start - 0.5)
                r_data = reader.read_range(r_start, min(total_duration, v_end + 0.5))
                if r_data is None or best_lead_idx >= r_data.shape[0]:
                    return None
                peaks = sorted(r_start + float(q) / float(fs)
                               for q in pan_tompkins(np.asarray(r_data[best_lead_idx], dtype=float), fs=fs))
                peaks = [p for p in peaks if v_start - 0.05 <= p <= v_end + 0.05]
                ref = [p for p in peaks if p < min(v_end, v_start + window_size)]
            if len(ref) < 5:
                return None
            ref_rr = np.diff(np.asarray(ref, dtype=float))
            vt_rr = float(np.median(ref_rr))
            if vt_rr <= 0 or float(np.std(ref_rr)) / vt_rr > 0.12:
                return None
            cand = [p for p in peaks if p >= ref[-1] - 1e-6]
            seg_start = max(0.0, ref[0] - 0.5)
            seg_end = min(total_duration, v_end + 0.5)
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

            def _snip(t):
                i = int(round((t - seg_start) * fs))
                if i - half_w < 0 or i + half_w > len(lead_sig):
                    return None
                x = lead_sig[i - half_w:i + half_w]
                return x - float(np.mean(x))

            ref_snips = [x for x in (_snip(t) for t in ref) if x is not None]
            if len(ref_snips) < 5:
                return None
            template = np.median(np.asarray(ref_snips), axis=0)
            t_norm = float(np.linalg.norm(template))
            t_ptp = float(np.ptp(template))
            if t_norm <= 0 or t_ptp <= 0:
                return None

            def _is_vt_beat(t, prev_t):
                if abs((t - prev_t) - vt_rr) > 0.15 * vt_rr:
                    return False
                x = _snip(t)
                if x is None:
                    return False
                n = float(np.linalg.norm(x))
                if n <= 0 or not (0.5 <= float(np.ptp(x)) / t_ptp <= 2.0):
                    return False
                return float(np.dot(x, template)) / (n * t_norm) >= 0.90

            last, misses = cand[0] if cand else None, 0
            for k in range(1, len(cand)):
                if _is_vt_beat(cand[k], cand[k - 1]):
                    last, misses = cand[k], 0
                else:
                    misses += 1
                    if misses >= 2:
                        return last
            return None

        for idx, region in enumerate(merged_morph):
            if region['label'] != 'Ventricular Tachycardia':
                continue
            try:
                if not (idx + 1 < len(merged_morph)
                        and float(merged_morph[idx + 1]['start_sec']) - float(region['end_sec']) <= window_size):
                    vt_end = _vt_run_last_beat_fwd(region)
                    if vt_end is not None and float(region['start_sec']) + 1.0 < vt_end < float(region['end_sec']) - 0.05:
                        region['end_sec'] = vt_end
                        _, region['end_time_str'] = make_time_strs(region['start_sec'], vt_end)
                # Never into another morphological label: the walk stops at
                # the end of the region before (VFib, AFib, a block, ...), so
                # no other rhythm's segment is shortened by this.
                floor_sec = float(merged_morph[idx - 1]['end_sec']) if idx > 0 else 0.0
                vt_first = _vt_run_first_beat(region, floor_sec)
                if vt_first is not None and vt_first < region['start_sec'] - 0.05:
                    region['start_sec'] = vt_first
                    region['start_time_str'], _ = make_time_strs(vt_first, region['end_sec'])
                trimmed_end = _vt_flatline_end_trim(region['start_sec'], region['end_sec'])
                if trimmed_end < region['end_sec'] - 0.01:
                    region['end_sec'] = trimmed_end
                    _, region['end_time_str'] = make_time_strs(region['start_sec'], trimmed_end)
                    if idx + 1 < len(merged_morph):
                        # Same flag _vfib_flatline_start_trim sets, read by
                        # the same generic gap-closing pass -- see its
                        # comment there.
                        merged_morph[idx + 1]['_flatline_gap_before'] = True
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
        # RR interval is judged on its own (_interval_is_flutter); intervals
        # too short to judge (fast 2:1 Flutter) count as unknown and never
        # move the edge.
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
                # A gap between the two (another rhythm in between) is not
                # one edge: each region moves only its own side, and the gap
                # stays theirs to neither. Seen: AFib to 50 s, a regular 160
                # bpm run, Flutter from 61.2 s -- AFib was stretched to 61.2 s.
                if nxt['start_sec'] - cur['end_sec'] > 0.5:
                    if new_edge is not None and cur['end_sec'] < new_edge < nxt['end_sec'] - 0.5:
                        nxt['start_sec'] = new_edge
                        nxt['start_time_str'], _ = make_time_strs(new_edge, nxt['end_sec'])
                    own_end = _regular_run_first_beat(cur['end_sec'])
                    if own_end is not None and cur['start_sec'] + 0.5 < own_end < cur['end_sec']:
                        cur['end_sec'] = own_end
                        _, cur['end_time_str'] = make_time_strs(cur['start_sec'], own_end)
                elif new_edge is not None and cur['start_sec'] + 0.5 < new_edge < nxt['end_sec'] - 0.5:
                    cur['end_sec'] = nxt['start_sec'] = new_edge
                    _, cur['end_time_str'] = make_time_strs(cur['start_sec'], new_edge)
                    nxt['start_time_str'], _ = make_time_strs(new_edge, nxt['end_sec'])
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] AFib/Flutter boundary refine error: {e}")

        # Supraventricular Tachycardia (SVT) carved out of an Atrial
        # Fibrillation / Atrial Flutter region the window vote already
        # claimed. A short SVT run -- its own window never covers it alone,
        # see _svt_run_in above -- sitting between two AFib/Flutter windows
        # gets bridged over by the merge step above, and inside that merged
        # region the beat-by-beat AFib/Flutter split right below reads
        # SVT's very fast, very regular RR as "undetermined" (its TP
        # segment has no room to judge, _beat_tp_active) and the run simply
        # inherits whichever AFib/Flutter label surrounds it (seen directly:
        # a 200 bpm, 8-9 s SVT run swallowed whole into AFib on two
        # different recordings).
        # Reuses the same evidence the standalone SVT gap-fill lower in
        # this function requires (_svt_runs_in_gap: steady 200-333 ms RR
        # within 8% of the run's own median, one QRS shape, a sharp narrow
        # QRS by slope) at a shorter minimum run length -- 5 beats (4 RR
        # intervals), not 15: a run already sitting inside an AFib/Flutter
        # call, meeting that same shape/regularity bar, is lower risk to
        # mistake for genuine AFib/Flutter than free time is, and the
        # AFib/Flutter beat-by-beat split right after this already drops
        # any run of its own under 4 intervals on its own merits anyway.
        # Only ever carves into an AFib/Flutter region; every other label
        # and rule is untouched.
        _SVT_ATRIAL_MIN_BEATS = 5
        _svt_carve_label = 'Supraventricular Tachycardia (SVT)'

        def _svt_runs_in_atrial(region_start, region_end):
            peaks = [p for p in _sorted_peaks if region_start <= p < region_end]
            if len(peaks) < _SVT_ATRIAL_MIN_BEATS:
                return []
            read_lo = max(0.0, region_start - 1.0)
            sig = reader.read_range(read_lo, min(total_duration, region_end + 2.5))
            if sig is None or best_lead_idx >= sig.shape[0]:
                return []
            _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
            x = _filtfilt(_b, _a, np.asarray(sig[best_lead_idx], dtype=float))
            half = int(0.05 * fs)
            runs, i = [], 0
            while i < len(peaks) - 1:
                j, rrs = i, []
                while j + 1 < len(peaks):
                    rr = peaks[j + 1] - peaks[j]
                    if not (0.200 <= rr <= 0.3334):
                        break
                    if len(rrs) >= 3 and abs(rr - float(np.median(rrs))) > 0.08 * float(np.median(rrs)):
                        break
                    rrs.append(rr)
                    j += 1
                run = peaks[i:j + 1]
                if len(run) >= _SVT_ATRIAL_MIN_BEATS:
                    idx = [int(round((p - read_lo) * fs)) for p in run]
                    shapes = np.array([x[k - half:k + half] - float(np.mean(x[k - half:k + half]))
                                       for k in idx if k - half >= 0 and k + half <= len(x)])
                    sharp = []
                    for k in (idx[::4] or idx[:1]):
                        w = (x[k - half:k - half + int(2.0 * fs)]
                             if k - half >= 0 and k - half + int(2.0 * fs) <= len(x) else None)
                        if w is not None and float(np.ptp(w)) > 0:
                            sharp.append(float(np.max(np.abs(np.diff(w)))) * float(fs) / float(np.ptp(w)))
                    if len(shapes) >= 3 and sharp:
                        template = np.median(shapes, axis=0)
                        corrs = [float(np.corrcoef(s, template)[0, 1]) for s in shapes if np.any(s)]
                        if corrs and float(np.median(corrs)) >= 0.9 and float(np.median(sharp)) >= 40.0:
                            runs.append((run[0], run[-1]))
                i = max(j, i + 1)
            return runs

        try:
            _svt_carved = []
            for region in merged_morph:
                if region['label'] not in _atrial_labels:
                    _svt_carved.append(region)
                    continue
                cuts = _svt_runs_in_atrial(float(region['start_sec']), float(region['end_sec']))
                if not cuts:
                    _svt_carved.append(region)
                    continue
                pieces = [(float(region['start_sec']), float(region['end_sec']))]
                for c0, c1 in cuts:
                    nxt_pieces = []
                    for p0, p1 in pieces:
                        if c1 <= p0 or c0 >= p1:
                            nxt_pieces.append((p0, p1))
                            continue
                        if c0 > p0:
                            nxt_pieces.append((p0, c0))
                        if c1 < p1:
                            nxt_pieces.append((c1, p1))
                    pieces = nxt_pieces
                for p0, p1 in pieces:
                    if p1 - p0 < 0.5:
                        continue
                    piece = dict(region)
                    piece['start_sec'], piece['end_sec'] = p0, p1
                    piece['start_time_str'], piece['end_time_str'] = make_time_strs(p0, p1)
                    _svt_carved.append(piece)
                for c0, c1 in cuts:
                    c_s_str, c_e_str = make_time_strs(c0, c1)
                    _svt_carved.append({
                        'start_sec': c0, 'end_sec': c1, 'label': _svt_carve_label,
                        'color': '#FFA500', 'start_time_str': c_s_str, 'end_time_str': c_e_str,
                        'detected_rhythm': _svt_carve_label,
                    })
            merged_morph = sorted(_svt_carved, key=lambda x: x['start_sec'])
        except Exception as e:
            print(f"[Auto Arrhythmia Detect] SVT-in-AFib/Flutter carve error: {e}")

        # Atrial Fibrillation / Atrial Flutter, beat by beat, inside each
        # region the windows found: every RR interval is Flutter (steady F
        # wave cycle, _interval_is_flutter), AFib (active TP segment and no F
        # wave cycle, _beat_tp_active) or neither (quiet TP); consecutive
        # intervals of one kind make a segment, kept with >= 4 of them; an
        # interval that can't be judged continues the run it is in. A region
        # with fewer than 4 intervals that could be judged (fast 2:1 Flutter:
        # no room between beats) keeps its window extent. Seen: 4:1 Flutter
        # 1185.5-1192.6 s at the end of an AFib episode, all of it drawn as
        # AFib to 1193.4 s.
        try:
            rebuilt = []
            for region in merged_morph:
                if region['label'] not in _atrial_labels:
                    rebuilt.append(region)
                    continue
                peaks = [p for p in _sorted_peaks if region['start_sec'] - 0.01 <= p <= region['end_sec'] + 0.01]
                states = []
                for a, b in zip(peaks, peaks[1:]):
                    flutter = _interval_is_flutter(a, b)
                    if flutter:
                        states.append((a, b, 'Atrial Flutter'))
                        continue
                    active = _beat_tp_active(a, b)
                    if active is None:
                        states.append((a, b, None))
                    elif active:
                        states.append((a, b, 'Atrial Fibrillation' if flutter is False else None))
                    else:
                        states.append((a, b, ''))
                if sum(1 for _, _, s in states if s is not None) < 4:
                    rebuilt.append(region)
                    continue
                runs = []   # [label, start, end, decided intervals]
                for a, b, s in states:
                    if s is None:
                        if runs and runs[-1][0]:
                            runs[-1][2] = b
                        continue
                    if runs and runs[-1][0] == s:
                        runs[-1][2] = b
                        runs[-1][3] += 1
                    else:
                        runs.append([s, a, b, 1])
                for label, s_sec, e_sec, count in runs:
                    if not label or count < 4:
                        continue
                    piece = dict(region)
                    piece['label'] = piece['detected_rhythm'] = label
                    piece['start_sec'], piece['end_sec'] = s_sec, e_sec
                    piece['start_time_str'], piece['end_time_str'] = make_time_strs(s_sec, e_sec)
                    rebuilt.append(piece)
            merged_morph = sorted(rebuilt, key=lambda x: x['start_sec'])
        except Exception as e:
            print(f"[Auto Arrhythmia Detect] AFib/Flutter beat-by-beat error: {e}")

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
        _svt_max_rr = 0.340 if _svt_age_hr is None else 60.0 / _svt_age_hr

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

        # Beat-exact edges for Mobitz I and First-degree AV Block. Both
        # regions end on the 5 s window grid, so the first and last stretch
        # of the block were left unlabelled (seen: a whole Wenckebach cycle,
        # PR 164 -> 226 -> 356 -> dropped P, before the region's 90.16 s
        # start and another after its 116.39 s end; First-degree PR 254 ms
        # beats left out after its 84.25 s end).
        #   - Mobitz I grows by whole cycles: the RRs of the cycle next to
        #     the region must match a cycle inside it (each RR within 12%)
        #     and its first beat's PR must be >= 40 ms shorter than the
        #     longest PR in that cycle (the Wenckebach reset).
        #   - First-degree grows beat by beat while PR > 200 ms, within
        #     25 ms of the region's median PR, and RR within 15% of its own.
        # Neither grows into another region.
        _sg_pr_at = {float(p): (m.get('pr_ms') if m else None)
                     for p, m in zip(all_r_peaks, sinus_gate_metrics)}

        def _free_limits(region):
            lo = max([o['end_sec'] for o in merged_morph
                      if o is not region and o['end_sec'] <= region['start_sec'] + 1e-6] + [0.0])
            hi = min([o['start_sec'] for o in merged_morph
                      if o is not region and o['start_sec'] >= region['end_sec'] - 1e-6] + [total_duration])
            return lo, hi

        def _cycle_matches(peaks, template):
            rr = np.diff(np.asarray(peaks, dtype=float))
            if len(rr) != len(template) or not np.all(np.abs(rr - template) <= 0.12 * template):
                return False
            prs = [_sg_pr_at.get(float(p)) for p in peaks[:-1]]
            measured = [v for v in prs if v is not None]
            return prs[0] is not None and measured and max(measured) - prs[0] >= 40.0

        for region in merged_morph:
            try:
                if region['label'] == _AV_MOBITZ_I:
                    idx = [i for i, p in enumerate(_sorted_peaks)
                           if region['start_sec'] - 1e-6 <= p <= region['end_sec'] + 1e-6]
                    if len(idx) < 4:
                        continue
                    rr_in = np.diff(np.asarray([_sorted_peaks[i] for i in idx], dtype=float))
                    med = float(np.median(rr_in))
                    starts = [idx[k + 1] for k in range(len(rr_in)) if rr_in[k] > 1.3 * med]
                    if len(starts) < 2:
                        continue
                    template = np.diff(np.asarray(_sorted_peaks[starts[0]:starts[1] + 1], dtype=float))
                    L = len(template)
                    lo, hi = _free_limits(region)
                    a = starts[0]
                    while a - L >= 0 and _sorted_peaks[a - L] >= lo - 1e-6 \
                            and _cycle_matches(_sorted_peaks[a - L:a + 1], template):
                        a -= L
                    b = starts[-1]
                    while b + L < len(_sorted_peaks) and _sorted_peaks[b + L] <= hi + 1e-6 \
                            and _cycle_matches(_sorted_peaks[b:b + L + 1], template):
                        b += L
                    new_start = min(region['start_sec'], float(_sorted_peaks[a]))
                    new_end = max(region['end_sec'], float(_sorted_peaks[b]))
                elif region['label'] == _AV_FIRST:
                    inside = [p for p in _sorted_peaks if region['start_sec'] - 1e-6 <= p <= region['end_sec'] + 1e-6]
                    prs = [_sg_pr_at.get(float(p)) for p in inside]
                    prs = [v for v in prs if v is not None]
                    if len(inside) < 3 or len(prs) < 3:
                        continue
                    med_pr = float(np.median(prs))
                    med_rr = float(np.median(np.diff(inside)))
                    lo, hi = _free_limits(region)

                    # 'yes' / 'unknown' (no PR measured: one is let through,
                    # never two in a row, and never as the outermost beat)
                    # / 'no'.
                    def _first_degree_beat(p, prev):
                        if abs(abs(p - prev) - med_rr) > 0.15 * med_rr:
                            return 'no'
                        pr = _sg_pr_at.get(float(p))
                        if pr is None:
                            return 'unknown'
                        return 'yes' if pr > 200.0 and abs(pr - med_pr) <= 25.0 else 'no'

                    def _walk(edge, beats):
                        last_ok, pending, prev = edge, False, edge
                        for p in beats:
                            state = _first_degree_beat(p, prev)
                            if state == 'no' or (state == 'unknown' and pending):
                                break
                            pending = state == 'unknown'
                            if state == 'yes':
                                last_ok = p
                            prev = p
                        return last_ok

                    new_start = _walk(inside[0], [q for q in _sorted_peaks if lo - 1e-6 <= q < inside[0]][::-1])
                    new_end = _walk(inside[-1], [q for q in _sorted_peaks if inside[-1] < q <= hi + 1e-6])
                    new_start = min(region['start_sec'], float(new_start))
                    new_end = max(region['end_sec'], float(new_end))
                else:
                    continue
                if (new_start, new_end) != (region['start_sec'], region['end_sec']):
                    region['start_sec'], region['end_sec'] = new_start, new_end
                    region['start_time_str'], region['end_time_str'] = make_time_strs(new_start, new_end)
            except Exception as e:
                print(f"[Auto Arrhythmia Detect] AV block beat-exact edge error: {e}")

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
            # A gap just before a region whose own leading edge was trimmed
            # back off a confirmed flatline (_vfib_flatline_start_trim) --
            # or just after a region whose own trailing edge was
            # (_vt_flatline_end_trim) -- is left alone instead of being
            # closed like an ordinary detection gap: closing it would just
            # hand the flatline right back to whichever rhythm happened to
            # border it, which is exactly what those trims were for. Only
            # read here; it never changes any other pair's gap-closing.
            elif nxt.get('_flatline_gap_before'):
                continue
            # Only a gap between two pieces of the SAME label is closed --
            # the real target is a region that this step's own earlier
            # passes (VFib's beat-exact edge refine, the AV-rank cut just
            # above) split back into several pieces or shifted a boundary
            # on, not two genuinely different rhythms that happen to sit
            # close together. Without this check, a short but real
            # in-between rhythm neither confirmed rule actually covers gets
            # silently handed to whichever one of the two sits first (seen:
            # Ventricular Fibrillation's own edge correctly stopped at
            # 74.22 s, where the beats become organized/narrow, and Atrial
            # Fibrillation's own edge correctly started at 75.95 s -- this
            # step then closed the 74.22-75.95 s gap by stretching VFib
            # across it, although neither rule actually found VFib or AFib
            # there).
            elif cur['label'] == nxt['label'] \
                    and 0.0 < (nxt['start_sec'] - cur['end_sec']) <= window_size:
                cur['end_sec'] = nxt['start_sec']
                cur['end_time_str'] = nxt['start_time_str']
        merged_morph = [m for m in merged_morph if m['end_sec'] > m['start_sec']]

        # Every morphological rhythm, AV blocks included, takes priority over
        # a Sinus Bradycardia/Tachycardia rate segment: e.g. a 55 bpm rhythm
        # with PR 280 ms passes the sinus gate as Bradycardia too, but is
        # marked First-degree AV Block only, never both stacked.
        detected_segments = _suppress_sinus_rate_overlap(detected_segments, merged_morph, make_time_strs)
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

    # Bundle branch block segments, beat by beat. Was: the 10 s window vote
    # (>= 75% of a window's beats) plus a beat-level edge extension, on the
    # window loop's beats. Two problems, seen on one recording where the
    # QRS switched between RBBB and LBBB shape every 4-30 beats:
    #   - the window loop's beats come from the three strongest leads of the
    #     first 30 s (V5, II, aVL there); on the LBBB-shaped beats V5 halved
    #     (peak-to-peak ~1778 -> ~877) and Pan-Tompkins dropped them -- no
    #     beat from 1085.6 to 1093.1 s while leads I, V1 and V6 had all of
    #     them -- so no LBBB was ever judged;
    #   - a 4-10 beat stretch never reaches 75% of a 10 s window, and a
    #     region ran on to the 5 s window grid past its last beat (RBBB to
    #     1090.0 s, last RBBB beat 1085.7 s).
    # Now: beats from leads I, V1 and V6 (a beat counts when >= 2 of them
    # find it within 100 ms -- one lead's noise is not a beat); every beat
    # judged on its own with the same measurements as the window rule
    # (_bundle_branch_block_beat_pr / _bundle_branch_block_beat_width_shape):
    # a P with a PR, and a Complete / Incomplete Right / Left shape and
    # width. Consecutive beats with the same label make one segment; a beat
    # that fails or has another label, or a missing beat (RR > 1.5x the
    # segment's own), ends it. Kept with >= 3 beats that pass the window
    # rule's consistency tests on their own: PRs within 25 ms of their median
    # and one QRS shape (median correlation >= 0.9), on >= 75% of them.
    # The window vote itself still runs and still decides the VFib veto, the
    # Third-degree AV Block edge and the VFib cut below
    # (bundle_branch_block_candidates); only the drawn segments come from here.
    _BUNDLE_BRANCH_BLOCK_BEAT_MIN = 3

    def _beat_bundle_branch_block_label(width, shape):
        right, left = shape
        if right and not left:
            side = 'Right'
        elif left and not right:
            side = 'Left'
        else:
            return None
        if width >= 120.0:
            return f'Complete {side} Bundle Branch Block'
        if width >= 110.0:
            return f'Incomplete {side} Bundle Branch Block'
        return None

    def _beat_level_bundle_branch_block_segments():
        beat_lead_idx = [i for i in (_lead_index('I'), _lead_index('V1'), _lead_index('V6', 'V5')) if i is not None]
        if len(beat_lead_idx) < 2 or _bundle_branch_block_v1_idx is None or not _bundle_branch_block_lateral_idx:
            return []
        chunk_sec, pad_sec = 60.0, 2.0
        merge_tol = max(1, int(0.1 * fs))
        h = int(0.05 * fs)
        beats = []   # (time_sec, label or None, pr_ms, snippet)
        t = 0.0
        while t < total_duration:
            t_end = min(t + chunk_sec, total_duration)
            p0, p1 = max(0.0, t - pad_sec), min(total_duration, t_end + pad_sec)
            try:
                raw = reader.read_range(p0, p1)
            except Exception:
                raw = None
            if raw is None or raw.shape[1] <= 50:
                t = t_end
                continue
            found = []
            for li in beat_lead_idx:
                if li < raw.shape[0]:
                    found.extend((int(p), li) for p in pan_tompkins(np.asarray(raw[li], dtype=float), fs=fs))
            found.sort()
            chunk_beats = []
            k = 0
            while k < len(found):
                group = [found[k]]
                while k + 1 < len(found) and found[k + 1][0] - group[0][0] < merge_tol:
                    k += 1
                    group.append(found[k])
                k += 1
                if len({li for _, li in group}) >= 2 and t <= p0 + group[0][0] / fs < t_end:
                    chunk_beats.append(group[0][0])
            filtered = raw
            if _atrial_preproc is not None:
                try:
                    filtered = np.vstack([
                        np.asarray(_atrial_preproc.process(np.asarray(lead, dtype=float)), dtype=float)
                        for lead in raw
                    ])
                except Exception:
                    filtered = raw
            # 60 Hz notch, on top of _atrial_preproc's own 50 Hz-only one,
            # for this beat-level Bundle Branch Block measurement alone --
            # _atrial_preproc itself (shared by the AFib/Flutter/VFib
            # window rules) is untouched. Confirmed directly: under 60 Hz
            # interference, every beat's width/shape measurement here
            # failed outright (None) for ~60 s straight -- a genuine
            # RBBB-> First-degree AV Block-> LBBB transition underneath
            # went completely unmarked except for First-degree (its own,
            # different measurement happened to tolerate the 60 Hz).
            try:
                from ..enhanced_filter import make_notch as _bbb_make_notch
                _bbb_notch = _bbb_make_notch(60.0, fs)
                filtered = np.vstack([
                    _filtfilt(_bbb_notch.b, _bbb_notch.a, np.asarray(lead, dtype=float))
                    for lead in filtered
                ])
            except Exception:
                pass
            lead_ii = np.asarray(filtered[conduction_lead_idx], dtype=float) \
                if conduction_lead_idx < filtered.shape[0] else None
            best = np.asarray(raw[best_lead_idx], dtype=float) if best_lead_idx < raw.shape[0] else None
            median_best = float(np.median(best)) if best is not None and best.size else 0.0
            for r in chunk_beats:
                t_sec = p0 + r / fs
                prev_rr = (t_sec - beats[-1][0]) * 1000.0 if beats else None
                label, pr, snippet = None, None, None
                try:
                    beat_p = _bundle_branch_block_beat_pr(lead_ii, r, prev_rr, fs) if lead_ii is not None else None
                    width_shape = _bundle_branch_block_beat_width_shape(filtered, r, fs)
                    if beat_p is not None and beat_p[0] and beat_p[1] is not None \
                            and width_shape is not None and width_shape[1] is not None:
                        label = _beat_bundle_branch_block_label(width_shape[0], width_shape[1])
                        pr = beat_p[1]
                    if label is not None and best is not None:
                        a_lo, a_hi = max(0, r - int(0.04 * fs)), min(best.size, r + int(0.10 * fs))
                        if a_hi > a_lo:
                            c = a_lo + int(np.argmax(np.abs(best[a_lo:a_hi] - median_best)))
                            if c - h >= 0 and c + h <= best.size:
                                snippet = best[c - h:c + h] - float(np.mean(best[c - h:c + h]))
                except Exception as e:
                    print(f"[Auto Arrhythmia Detect] Bundle Branch Block beat error {t_sec:.2f}s: {e}")
                    label = None
                beats.append((t_sec, label, pr, snippet))
            t = t_end

        def _consistent(run):
            n_run = len(run)
            need = max(3, int(np.ceil(0.75 * n_run)))
            prs = [b[2] for b in run]
            pr_median = float(np.median(prs))
            if sum(1 for v in prs if abs(v - pr_median) <= 25.0) < need:
                return False
            shapes = [b[3] for b in run if b[3] is not None and np.any(b[3])]
            if len(shapes) < need:
                return False
            template = np.median(np.array(shapes), axis=0)
            if not np.any(template):
                return False
            corrs = [float(np.corrcoef(s, template)[0, 1]) for s in shapes]
            return float(np.median(corrs)) >= 0.9

        segments = []
        run = []

        def _append_bbb_segment(run):
            label = run[0][1]
            s_sec, e_sec = float(run[0][0]), float(run[-1][0])
            s_str, e_str = make_time_strs(s_sec, e_sec)
            segments.append({
                'start_sec': s_sec,
                'end_sec': e_sec,
                'label': label,
                'color': _BUNDLE_BRANCH_BLOCK_COLORS[label],
                'start_time_str': s_str,
                'end_time_str': e_str,
                'detected_rhythm': label,
                '_rr': float(np.median(np.diff([b[0] for b in run]))),
            })

        def _close(run):
            if len(run) < _BUNDLE_BRANCH_BLOCK_BEAT_MIN:
                return
            if _consistent(run):
                _append_bbb_segment(run)
                return
            # A run otherwise one QRS shape/width can still fail _consistent
            # purely on PR: two genuinely different, each internally stable
            # PR levels in one run (seen: 12 beats PR 238-242 ms, then 14
            # beats PR 148-152 ms, step right at the 13th beat) fail the
            # single-median-PR test as a whole (only the larger group's
            # beats are "near the median", 54% < the 75% bar) even though
            # each half easily clears that same bar on its own. Try the
            # biggest PR jumps in the run as a split point (not every k --
            # a real step is the actual cause, not an arbitrary valid-but-
            # wrong boundary some other k might also happen to pass at);
            # split there only when BOTH halves independently pass the
            # exact same _consistent() check the whole run already had to.
            # No match: the run is dropped, same as before this existed.
            prs = [b[2] for b in run]
            jumps = sorted(
                range(1, len(prs)),
                key=lambda i: abs(prs[i] - prs[i - 1]),
                reverse=True,
            )
            for k in jumps[:3]:
                if k < _BUNDLE_BRANCH_BLOCK_BEAT_MIN or (len(run) - k) < _BUNDLE_BRANCH_BLOCK_BEAT_MIN:
                    continue
                left, right = run[:k], run[k:]
                if _consistent(left) and _consistent(right):
                    _append_bbb_segment(left)
                    _append_bbb_segment(right)
                    return

        for beat in beats:
            if run and beat[1] == run[0][1]:
                gaps = np.diff([b[0] for b in run])
                limit = 1.5 * float(np.median(gaps)) if gaps.size else None
                if limit is None or beat[0] - run[-1][0] <= limit:
                    run.append(beat)
                    continue
            _close(run)
            run = [beat] if beat[1] is not None else []
        _close(run)
        return segments

    try:
        beat_bundle_branch_block = _beat_level_bundle_branch_block_segments()
    except Exception as e:
        beat_bundle_branch_block = []
        print(f"[Auto Arrhythmia Detect] Bundle Branch Block beat-level error: {e}")

    if bundle_branch_block_candidates or beat_bundle_branch_block:
        bundle_branch_block_candidates.sort(key=lambda x: x['start_sec'])
        merged_bundle_branch_block = sorted(beat_bundle_branch_block, key=lambda x: x['start_sec'])
        _bundle_branch_block_rr = {id(seg): seg.pop('_rr') for seg in merged_bundle_branch_block}

        # Two bundle branch block regions that meet beat to beat (one ends on
        # a beat, the next starts on the very next beat, e.g. Right -> Left)
        # leave that one RR interval in neither region: seen as a 0.76 s gap
        # (613.44 -> 614.20 s) where Full Disclosure's label fell back to
        # "Normal Sinus Rhythm" for one second. Where the gap is no more than
        # one beat (<= 1.5x the earlier region's own RR), the earlier region
        # now ends where the next one starts.
        for cur, nxt in zip(merged_bundle_branch_block, merged_bundle_branch_block[1:]):
            gap = nxt['start_sec'] - cur['end_sec']
            if gap <= 0:
                continue
            rr = _bundle_branch_block_rr.get(id(cur), 1.0)
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

    # Atrial Flutter continuing at a faster conduction ratio. The Flutter
    # rule (_classify_atrial_window) reads F waves in each beat's TP
    # segment, which needs RR >= ~530 ms: at 2:1 (RR ~400 ms, 150 bpm) no
    # beat has TP room, the window gets no verdict and a Flutter episode
    # that speeds up from 4:1 / 3:1 to 2:1 ends at the last slow window
    # (seen: 3:1 -> 2:1 at 105.0 s, the 2:1 beats to 109.7 s unmarked). That
    # rule is unchanged; this step only lengthens an Atrial Flutter segment
    # it already found, beat by beat from its end, into time no other
    # segment covers. A beat continues the Flutter when, against the last
    # 10 s of the segment (lead II, 0.5-40 Hz):
    #   - its waveform after the QRS (R + 60..340 ms) correlates >= 0.90 with
    #     the segment's: the ventricle is triggered by an F wave, so the
    #     sawtooth after the QRS repeats at every conduction ratio. Sinus
    #     Tachycardia there has T + P instead -- at most 0.55 against the
    #     Flutter segments of the recordings, 2:1 Flutter 0.97-0.98;
    #   - its RR is k x the flutter cycle (k 2-5, within 0.10 k), the cycle
    #     measured in the segment (F-wave autocorrelation, 170-250 ms, as the
    #     AFib / Flutter edge refinement);
    #   - its QRS correlates >= 0.90 with the segment's (same conduction).
    # Extended only when at least _FLUTTER_CONT_MIN_BEATS beats in a row
    # pass; it ends at the last beat that passes.
    # backward=True runs the same test from the segment's start into the
    # uncovered time before it, against its first 10 s -- seen: Flutter 3:1
    # (100 bpm) then 2:1 (150 bpm) then 5:1, only the 5:1 part found by the
    # window rule, the 10 s of 3:1 / 2:1 before it left as plain Tachycardia.
    _FLUTTER_CONT_MIN_BEATS = 3

    def _extend_atrial_flutter_end(region, limit_sec, backward=False):
        if backward:
            ref_lo = region['start_sec']
            ref_hi = min(region['end_sec'], region['start_sec'] + 10.0)
            read_lo = max(0.0, limit_sec - 1.0)
            read_hi = min(total_duration, ref_hi + 1.5)
        else:
            ref_lo = max(region['start_sec'], region['end_sec'] - 10.0)
            ref_hi = region['end_sec']
            read_lo = max(0.0, ref_lo - 1.0)
            read_hi = min(total_duration, limit_sec + 1.5)
        sig = reader.read_range(read_lo, read_hi)
        if sig is None or conduction_lead_idx >= sig.shape[0]:
            return None
        x = np.asarray(sig[conduction_lead_idx], dtype=float)
        if len(x) < int(2.0 * fs):
            return None
        _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
        x = _filtfilt(_b, _a, x)
        idx = lambda t: int(round((t - read_lo) * fs))

        def _snip(t, w0, w1):
            i0, i1 = idx(t) + int(w0 * fs), idx(t) + int(w1 * fs)
            if i0 < 0 or i1 > len(x):
                return None
            s = x[i0:i1]
            return s - float(np.mean(s))

        def _corr(a, b):
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            return float(np.dot(a, b)) / (na * nb) if na > 0 and nb > 0 else -1.0

        ref_peaks = [p for p in all_r_peaks if ref_lo <= p <= ref_hi]
        lags = []
        for p0, p1 in zip(ref_peaks[:-1], ref_peaks[1:]):
            i0, i1 = idx(p0) + int(0.12 * fs), idx(p1) - int(0.08 * fs)
            if i1 - i0 < int(0.4 * fs) or i0 < 0 or i1 > len(x):
                continue
            seg = x[i0:i1]
            n = np.arange(len(seg))
            seg = seg - np.polyval(np.polyfit(n, seg, 1), n)
            ac = np.correlate(seg, seg, 'full')[len(seg) - 1:]
            if ac[0] <= 0:
                continue
            ac = ac / ac[0]
            lo_lag, hi_lag = int(0.15 * fs), int(0.35 * fs)
            lag = lo_lag + int(np.argmax(ac[lo_lag:hi_lag]))
            if ac[lag] >= 0.4 and 0.17 * fs <= lag <= 0.25 * fs:
                lags.append(lag / float(fs))
        if len(lags) < 3 or max(lags) - min(lags) > 0.02:
            return None
        cycle = float(np.median(lags))
        after_ref = [s for s in (_snip(p, 0.06, 0.34) for p in ref_peaks) if s is not None]
        qrs_ref = [s for s in (_snip(p, -0.06, 0.06) for p in ref_peaks) if s is not None]
        if len(after_ref) < 4 or len(qrs_ref) < 4:
            return None
        after_tpl = np.median(np.asarray(after_ref), axis=0)
        qrs_tpl = np.median(np.asarray(qrs_ref), axis=0)

        if backward:
            chain = [p for p in all_r_peaks if limit_sec <= p < region['start_sec']]
            first_in = next((p for p in all_r_peaks if p >= region['start_sec']), None)
            if first_in is not None:
                chain.append(first_in)
            pairs = list(zip(chain[:-1], chain[1:]))[::-1]
        else:
            # from the segment's own last beat, as the backward chain does
            last_in = next((p for p in reversed(all_r_peaks) if p <= region['end_sec']), None)
            cand = ([last_in] if last_in is not None else []) + \
                [p for p in all_r_peaks if region['end_sec'] < p <= limit_sec]
            pairs = list(zip(cand[:-1], cand[1:]))
        passed = []
        for p0, p1 in pairs:
            rr = p1 - p0
            k = rr / cycle
            s_after, s_qrs = _snip(p0, 0.06, 0.34), _snip(p0, -0.06, 0.06)
            if (rr < 0.36 or s_after is None or s_qrs is None
                    or not 2 <= round(k) <= 5 or abs(k - round(k)) > 0.10
                    or _corr(s_after, after_tpl) < 0.90 or _corr(s_qrs, qrs_tpl) < 0.90):
                break
            passed.append(p0)
        if len(passed) < _FLUTTER_CONT_MIN_BEATS:
            return None
        return passed[-1]

    for region in final_segments:
        if region['label'] != 'Atrial Flutter':
            continue
        try:
            # up to the next segment of any label (or the recording end)
            limit_sec = min([o['start_sec'] for o in final_segments
                             if o is not region and o['end_sec'] > region['end_sec']]
                            + [total_duration])
            new_end = None
            if limit_sec > region['end_sec'] + 0.5:
                new_end = _extend_atrial_flutter_end(region, limit_sec)
            if new_end is not None and new_end > region['end_sec']:
                region['end_sec'] = float(new_end)
                _, region['end_time_str'] = make_time_strs(region['start_sec'], region['end_sec'])
        except Exception as e:
            print(f"[Auto Arrhythmia Detect] Atrial Flutter continuation error: {e}")
        try:
            # back to the previous segment of any label (or the recording start)
            start_limit = max([o['end_sec'] for o in final_segments
                               if o is not region and o['start_sec'] < region['start_sec']]
                              + [0.0])
            if start_limit >= region['start_sec'] - 0.5:
                continue
            new_start = _extend_atrial_flutter_end(region, start_limit, backward=True)
            if new_start is not None and new_start < region['start_sec']:
                region['start_sec'] = float(new_start)
                region['start_time_str'], _ = make_time_strs(region['start_sec'], region['end_sec'])
        except Exception as e:
            print(f"[Auto Arrhythmia Detect] Atrial Flutter backward continuation error: {e}")

    # Ventricular Fibrillation in time no segment covers. The VFib rule
    # (_is_vfib_window) votes on the beats Pan-Tompkins finds in a 10 s
    # window; its threshold, set by the sharp QRS before, picks up only a
    # few of the rounded VF waves, so a short VF episode between two other
    # rhythms is outvoted by their beats in every window (seen: ~7.5 s of VF
    # between Sinus Bradycardia and AFib, 3 of ~30 waves detected, 3 of 5
    # beats with a P wave -> no VFib). That rule is unchanged; this step
    # judges the waveform itself, without R-peaks, on 2 s windows (1 s step,
    # 0.5-40 Hz) of the unlabelled stretches. A window is VF-like on a lead
    # when all hold (values over the labelled rhythms of the recordings):
    #   - < 0.45 of the samples near the baseline (within 10% of its
    #     peak-to-peak of the median): no isoelectric line (VFib <= 0.42,
    #     every rhythm with QRS complexes >= 0.49);
    #   - steepest slope < 40 x peak-to-peak per second: rounded waves, no
    #     sharp QRS (VFib <= 34, narrow-QRS rhythms >= 46);
    #   - >= 0.6 of the 1-30 Hz power at 2.5-7 Hz (VFib >= 0.65, sinus /
    #     AFib / Flutter <= 0.45);
    #   - autocorrelation (lag 0.12-0.5 s) < 0.70: not a regular rhythm
    #     (VFib median 0.34, Ventricular Tachycardia >= 0.80).
    # VFib where >= 3 of the leads I, II, V1, V3, V5 agree (artefact is
    # rarely in every lead) for >= 4 windows in a row. Over all unlabelled
    # time of the recordings this found only real VF (4 stretches), no other
    # rhythm.
    _VF_GAP_LEADS = ('I', 'II', 'V1', 'V3', 'V5')
    _VF_GAP_MIN_LEADS = 3
    _VF_GAP_MIN_WINDOWS = 4
    _VF_GAP_WIN = 2.0

    def _vf_like_window(w):
        w = w - float(np.median(w))
        ptp = float(np.ptp(w))
        if ptp <= 0:
            return False
        if float(np.mean(np.abs(w) < 0.1 * ptp)) >= 0.45:
            return False
        if float(np.max(np.abs(np.diff(w)))) * float(fs) / ptp >= 40.0:
            return False
        spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
        freqs = np.fft.rfftfreq(len(w), 1.0 / float(fs))
        total = float(spec[(freqs >= 1.0) & (freqs <= 30.0)].sum())
        if total <= 0 or float(spec[(freqs >= 2.5) & (freqs <= 7.0)].sum()) / total < 0.6:
            return False
        ac = np.correlate(w, w, 'full')[len(w) - 1:]
        if ac[0] <= 0:
            return False
        ac = ac / ac[0]
        return float(np.max(ac[int(0.12 * fs):int(0.5 * fs)])) < 0.70

    def _vf_runs_in_gap(gap_start, gap_end):
        if gap_end - gap_start < _VF_GAP_WIN + (_VF_GAP_MIN_WINDOWS - 1):
            return []
        leads = [lead_names.index(n) for n in _VF_GAP_LEADS if n in lead_names]
        if len(leads) < _VF_GAP_MIN_LEADS:
            return []
        read_lo = max(0.0, gap_start - 1.0)
        sig = reader.read_range(read_lo, min(total_duration, gap_end + 1.0))
        if sig is None:
            return []
        _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
        filtered = {li: _filtfilt(_b, _a, np.asarray(sig[li], dtype=float)) for li in leads if li < sig.shape[0]}
        n_win = int(_VF_GAP_WIN * fs)
        runs, run = [], []
        t = gap_start
        while t + _VF_GAP_WIN <= gap_end:
            i = int(round((t - read_lo) * fs))
            votes = sum(1 for x in filtered.values()
                        if i + n_win <= len(x) and _vf_like_window(x[i:i + n_win]))
            if votes >= _VF_GAP_MIN_LEADS:
                run.append(t)
            else:
                if len(run) >= _VF_GAP_MIN_WINDOWS:
                    runs.append((run[0], run[-1] + _VF_GAP_WIN))
                run = []
            t += 1.0
        if len(run) >= _VF_GAP_MIN_WINDOWS:
            runs.append((run[0], run[-1] + _VF_GAP_WIN))
        return runs

    try:
        covered = sorted((s['start_sec'], s['end_sec']) for s in final_segments)
        gaps, cursor = [], 0.0
        for c_start, c_end in covered:
            if c_start > cursor:
                gaps.append((cursor, c_start))
            cursor = max(cursor, c_end)
        if cursor < total_duration:
            gaps.append((cursor, total_duration))
        for gap_start, gap_end in gaps:
            for vf_start, vf_end in _vf_runs_in_gap(gap_start, gap_end):
                vf_end = min(vf_end, gap_end)
                # touching an existing VFib segment: lengthen that one
                touching = [s for s in final_segments if s['label'] == 'Ventricular Fibrillation'
                            and (abs(s['end_sec'] - vf_start) <= 1.0 or abs(s['start_sec'] - vf_end) <= 1.0)]
                if touching:
                    seg = touching[0]
                    seg['start_sec'] = min(seg['start_sec'], vf_start)
                    seg['end_sec'] = max(seg['end_sec'], vf_end)
                    seg['start_time_str'], seg['end_time_str'] = make_time_strs(seg['start_sec'], seg['end_sec'])
                    continue
                s_str, e_str = make_time_strs(vf_start, vf_end)
                final_segments.append({
                    'start_sec': float(vf_start),
                    'end_sec': float(vf_end),
                    'label': 'Ventricular Fibrillation',
                    'color': '#FF3333',
                    'start_time_str': s_str,
                    'end_time_str': e_str,
                    'detected_rhythm': 'Ventricular Fibrillation'
                })
        final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Ventricular Fibrillation gap check error: {e}")

    # Supraventricular Tachycardia (SVT) in time no segment covers. The SVT
    # rule (_is_atrial_tachycardia_window) judges 10 s windows, and one lone
    # SVT window is dropped as "not an episode" (the isolated-window guard):
    # a ~10 s SVT episode between two other rhythms fills only one window
    # completely, the windows across its edges mix in the other rhythm's RR
    # (seen: Flutter -> SVT 200 bpm 87.3-97.8 s -> VFib: only 90-100 s
    # passed, dropped). That rule and that guard are unchanged; this step
    # marks, beat-exact, a run of R-peaks in an unlabelled stretch when
    #   - >= _SVT_GAP_MIN_BEATS beats over >= _SVT_GAP_MIN_SEC,
    #   - every RR 200-333 ms (180-300 bpm, the SVT rule's rate band; Sinus
    #     Tachycardia and 2:1 Flutter stay below 180) and within 8% of the
    #     run's median RR (regular; AFib / variable Flutter are not),
    #   - one QRS shape (median correlation >= 0.9 with the run's median
    #     QRS, as the SVT rule),
    #   - sharp narrow QRS: steepest slope >= 40 x peak-to-peak per second
    #     on 2 s stretches of the run (SVT 50-54 on the recordings,
    #     Ventricular Tachycardia 14-17, LBBB ~25).
    # Over the recordings it fired only on SVT: every labelled SVT run, no
    # Sinus Tachycardia / VT / AFib / Flutter region, and 3 unlabelled SVT
    # runs.
    _SVT_GAP_MIN_BEATS = 15
    _SVT_GAP_MIN_SEC = 5.0

    def _svt_runs_in_gap(gap_start, gap_end):
        peaks = [p for p in all_r_peaks if gap_start <= p < gap_end]
        if len(peaks) < _SVT_GAP_MIN_BEATS:
            return []
        read_lo = max(0.0, gap_start - 1.0)
        sig = reader.read_range(read_lo, min(total_duration, gap_end + 2.5))
        if sig is None or best_lead_idx >= sig.shape[0]:
            return []
        _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
        x = _filtfilt(_b, _a, np.asarray(sig[best_lead_idx], dtype=float))
        half = int(0.05 * fs)
        runs, i = [], 0
        while i < len(peaks) - 1:
            j, rrs = i, []
            while j + 1 < len(peaks):
                rr = peaks[j + 1] - peaks[j]
                if not (0.200 <= rr <= 0.3334):
                    break
                if len(rrs) >= 3 and abs(rr - float(np.median(rrs))) > 0.08 * float(np.median(rrs)):
                    break
                rrs.append(rr)
                j += 1
            run = peaks[i:j + 1]
            if len(run) >= _SVT_GAP_MIN_BEATS and run[-1] - run[0] >= _SVT_GAP_MIN_SEC:
                idx = [int(round((p - read_lo) * fs)) for p in run]
                shapes = np.array([x[k - half:k + half] - float(np.mean(x[k - half:k + half]))
                                   for k in idx if k - half >= 0 and k + half <= len(x)])
                sharp = []
                for k in idx[::4]:
                    w = x[k - half:k - half + int(2.0 * fs)] if k - half >= 0 and k - half + int(2.0 * fs) <= len(x) else None
                    if w is not None and float(np.ptp(w)) > 0:
                        sharp.append(float(np.max(np.abs(np.diff(w)))) * float(fs) / float(np.ptp(w)))
                if len(shapes) >= 3 and sharp:
                    template = np.median(shapes, axis=0)
                    corrs = [float(np.corrcoef(s, template)[0, 1]) for s in shapes if np.any(s)]
                    if corrs and float(np.median(corrs)) >= 0.9 and float(np.median(sharp)) >= 40.0:
                        runs.append((run[0], run[-1]))
            i = max(j, i + 1)
        return runs

    try:
        covered = sorted((s['start_sec'], s['end_sec']) for s in final_segments)
        gaps, cursor = [], 0.0
        for c_start, c_end in covered:
            if c_start > cursor:
                gaps.append((cursor, c_start))
            cursor = max(cursor, c_end)
        if cursor < total_duration:
            gaps.append((cursor, total_duration))
        _svt_label = 'Supraventricular Tachycardia (SVT)'
        for gap_start, gap_end in gaps:
            for svt_start, svt_end in _svt_runs_in_gap(gap_start, gap_end):
                # touching an existing SVT segment: lengthen that one
                touching = [s for s in final_segments if s['label'] == _svt_label
                            and (abs(s['end_sec'] - svt_start) <= 1.0 or abs(s['start_sec'] - svt_end) <= 1.0)]
                if touching:
                    seg = touching[0]
                    seg['start_sec'] = min(seg['start_sec'], svt_start)
                    seg['end_sec'] = max(seg['end_sec'], svt_end)
                    seg['start_time_str'], seg['end_time_str'] = make_time_strs(seg['start_sec'], seg['end_sec'])
                    continue
                s_str, e_str = make_time_strs(svt_start, svt_end)
                final_segments.append({
                    'start_sec': float(svt_start),
                    'end_sec': float(svt_end),
                    'label': _svt_label,
                    'color': '#FFA500',
                    'start_time_str': s_str,
                    'end_time_str': e_str,
                    'detected_rhythm': _svt_label
                })
        final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Supraventricular Tachycardia gap check error: {e}")

    # Ventricular Tachycardia (VT) carved out of a Ventricular Fibrillation
    # (VFib) call already made above. The window loop's own R-peaks
    # (core_r_peaks) come from one Pan-Tompkins call per window, which sets
    # ONE amplitude threshold for that whole padded window. A VT run's QRS
    # is often 5-7x smaller than a neighbouring sinus or VFib-onset beat's
    # (measured directly: ~170-300 vs ~1300-2000), so a window straddling
    # VT and either neighbour sets its threshold off the bigger beats and
    # misses most of VT's own -- the broken RR that is left then satisfies
    # VFib's "no organized QRS, no steady RR" rule instead (confirmed: a
    # window scoped to VT's own 5 s found 13/13 of its beats at a steady
    # 374 ms RR; the production 10 s window, padded +/-2 s into the
    # neighbouring sinus/VFib amplitude, found only 1-6 of the same beats).
    # This pass re-reads short, local stretches (padded a beat's width, not
    # a whole window) and runs Pan-Tompkins fresh on each, across the same
    # top-3 amplitude leads the window loop itself uses
    # (peak_detection_lead_indices) -- avoiding that dilution rather than
    # reusing all_r_peaks (same single-threshold problem, over a 60 s
    # chunk and one lead -- confirmed it misses this same VT run entirely)
    # or the window loop's own core_r_peaks.
    # A run of >= _VT_LOCAL_MIN_BEATS (5) consecutive beats is VT when its
    # RR is regular (every interval within 12% of the run's own median,
    # the same regularity _is_ventricular_tachycardia_window asks of its
    # own window-found core) and its rate is >= 100/min, and >= 75% of the
    # run's beats measure a wide QRS (120-220 ms, std <= 25 ms across
    # them) -- the same two gates that window rule already requires,
    # evaluated on a run of beats instead of on one window. Only ever
    # carves VT out of an existing Ventricular Fibrillation call; every
    # other label and rule is untouched, and this runs last, after every
    # other rhythm and boundary rule above has already settled.
    _VT_LOCAL_MIN_BEATS = 5
    _vt_carve_label = 'Ventricular Tachycardia'

    def _vt_local_peaks(scan_start, scan_end):
        """R-peaks in [scan_start, scan_end), found fresh on short, tightly
        local reads -- not all_r_peaks' 60 s single-lead chunks, whose one
        threshold dilutes the same way a whole window's does. Union across
        the recording's top-3 amplitude leads, merged within 100 ms as one
        beat, the same leads and merge rule the window loop's own
        core_r_peaks use."""
        out = []
        # Measured on the VT run this pass exists for: step=4/pad=1 still
        # dilutes (its first read spans the VT run's own neighbouring loud
        # beat, same problem as the 10 s window), step<=1.5/pad<=0.5 does
        # not -- every true VT beat recovered, none from the loud beats
        # bleeding in.
        step, pad = 1.5, 0.5
        t = scan_start
        while t < scan_end:
            w1 = min(scan_end, t + step)
            p0, p1 = max(0.0, t - pad), min(total_duration, w1 + pad)
            blk = reader.read_range(p0, p1)
            if blk is not None:
                found = []
                for li in peak_detection_lead_indices:
                    if li < blk.shape[0]:
                        lead_sig = np.asarray(blk[li], dtype=float)
                        if len(lead_sig) > 50:
                            found.extend(p0 + float(idx) / fs for idx in pan_tompkins(lead_sig, fs=fs))
                found.sort()
                for p in found:
                    if t <= p < w1 and (not out or p - out[-1] >= 0.1):
                        out.append(p)
            t = w1
        return out

    def _vt_beat_qrs_ms(peak_sec):
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
        return beat.get('qrs_ms') if beat else None

    def _vt_trim_to_core(run):
        """Trim a beat run (absolute seconds) to its own longest
        stable-RR core, dropping transition beats at either edge -- a
        sinus/VFib-onset beat still closing in on VT's own rate must not
        get carved in as VT. Reuses _ventricular_tachycardia_core_start
        exactly as the window rule already uses it to trim VT's start
        (forwards on the run, then again on the run reversed, for the
        end) -- the same definition of "VT's own core", not a new one."""
        if len(run) < 4:
            return run
        start_t = _ventricular_tachycardia_core_start([p * fs for p in run], fs)
        if start_t is None:
            return run
        trimmed = [p for p in run if p >= start_t - 1e-9]
        if len(trimmed) < 4:
            return trimmed
        rev_samples = [(trimmed[-1] - p) * fs for p in reversed(trimmed)]
        end_rel = _ventricular_tachycardia_core_start(rev_samples, fs)
        if end_rel is None:
            return trimmed
        end_t = trimmed[-1] - end_rel
        return [p for p in trimmed if start_t - 1e-9 <= p <= end_t + 1e-9]

    def _vt_is_test_signal(run_start, run_end):
        try:
            from .pqrst_measure import span_status
            # Not just 'test_signal' (flat-step square/pulse train): the
            # square wave this check exists for actually read 'non_ecg'
            # (leads I/V1/V5 correlate >= 0.95 -- the square repeats
            # identically on every lead, just without the flat step
            # 'test_signal' itself looks for). Either way the span isn't
            # distinguishable heart activity; only 'ok' is let through.
            return span_status(reader, max(0.0, run_start - 0.5),
                                min(total_duration, run_end + 0.5)) != 'ok'
        except Exception:
            return False

    def _vt_runs_in(scan_start, scan_end):
        peaks = _vt_local_peaks(scan_start, scan_end)
        if len(peaks) < _VT_LOCAL_MIN_BEATS:
            return []
        runs, i = [], 0
        while i < len(peaks) - 1:
            j, rrs = i, []
            while j + 1 < len(peaks):
                rr = peaks[j + 1] - peaks[j]
                if not (0.0 < rr <= 0.5):
                    break
                if len(rrs) >= 3 and abs(rr - float(np.median(rrs))) > 0.12 * float(np.median(rrs)):
                    break
                rrs.append(rr)
                j += 1
            run = _vt_trim_to_core(peaks[i:j + 1])
            if len(run) >= _VT_LOCAL_MIN_BEATS:
                qrs_vals = [_vt_beat_qrs_ms(p) for p in run]
                wide = [q for q in qrs_vals if q is not None and 120.0 <= q <= 300.0]
                # A run this regular and this uniform in width (std == 0
                # across every beat, not just low) is itself suspicious --
                # confirmed on a test recording: a repeating square/triangle
                # wave inside a VFib call gave 12 beats of exactly 250 ms RR
                # and exactly 160.0 ms QRS, bit for bit, no real beat is
                # that perfectly identical. Same check the VFib trim already
                # uses for the same reason (span_status == 'test_signal':
                # the waveform repeats identically across leads I/V1/V5,
                # flat steps). A real wide-complex run is unaffected.
                if (len(wide) >= max(3, int(np.ceil(0.75 * len(run)))) and float(np.std(wide)) <= 25.0
                        and not _vt_is_test_signal(run[0], run[-1])):
                    runs.append((run[0], run[-1]))
            i = max(j, i + 1)
        return runs

    try:
        _vt_carved = []
        for seg in final_segments:
            if seg['label'] != 'Ventricular Fibrillation':
                _vt_carved.append(seg)
                continue
            cuts = _vt_runs_in(float(seg['start_sec']), float(seg['end_sec']))
            if not cuts:
                _vt_carved.append(seg)
                continue
            pieces = [(float(seg['start_sec']), float(seg['end_sec']))]
            for c0, c1 in cuts:
                nxt_pieces = []
                for p0, p1 in pieces:
                    if c1 <= p0 or c0 >= p1:
                        nxt_pieces.append((p0, p1))
                        continue
                    if c0 > p0:
                        nxt_pieces.append((p0, c0))
                    if c1 < p1:
                        nxt_pieces.append((c1, p1))
                pieces = nxt_pieces
            for p0, p1 in pieces:
                if p1 - p0 < 1.0:
                    continue
                piece = dict(seg)
                piece['start_sec'], piece['end_sec'] = p0, p1
                piece['start_time_str'], piece['end_time_str'] = make_time_strs(p0, p1)
                _vt_carved.append(piece)
            for c0, c1 in cuts:
                c_s_str, c_e_str = make_time_strs(c0, c1)
                _vt_carved.append({
                    'start_sec': c0, 'end_sec': c1, 'label': _vt_carve_label,
                    'color': '#E67E22', 'start_time_str': c_s_str, 'end_time_str': c_e_str,
                    'detected_rhythm': _vt_carve_label,
                })
        final_segments = sorted(_vt_carved, key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] VT-in-VFib carve error: {e}")

    # ------------------------------------------------------------------
    # Artifact: its own rule. Marks stretches where some leads carry
    # non-cardiac signal (electrode motion, a loose / detached electrode,
    # muscle tremor) while the other leads still show the heart beating
    # normally. Its segments are added alongside the rhythm labels (like
    # the bundle branch block layer) and never change them.
    # Per 10 s window:
    #   beats -- R-peaks found by Pan-Tompkins on at least 3 of the leads
    #     I, II, III, V1, V3, V5 within 80 ms (between them these see every
    #     electrode, so an artifact from one electrode can't make or hide a
    #     beat on its own);
    #   per lead, every beat (250 ms before to 450 ms after R) minus the
    #     median of the lead's beats within 3 s of it = the non-cardiac
    #     residual (the median of the whole window instead mixed two
    #     morphologies where a test recording switched rhythm);
    #   per 2 s block, a lead is
    #     noisy  -- residual RMS >= 25% of the window's median QRS amplitude
    #               (all leads), or flat (lead-off: peak-to-peak < 8% of
    #               that median) while the heart beats on the others;
    #     clean  -- residual RMS < 15% of its own QRS amplitude, and that
    #               QRS >= 30% of the median one.
    #   A block is Artifact when >= 1 lead is noisy AND >= 1 lead is
    #   clean. In VFib (and any rhythm without one repeating beat) no lead
    #   is clean, so it is never called Artifact.
    # Blocks merge across a gap of up to one block; a segment is kept when
    # >= 4 s (two blocks): a lone 2 s block was a rhythm change or one
    # ectopic beat changing shape more on some leads than others (test
    # recordings), electrode artifact runs on. Each segment records its
    # noisy leads ('artifact_leads').
    # Measured on a 30 min human recording: electrode artifact at 140-170 s
    # (I, III, aVL, V1, V4...; Lead II clean), V4 lead-off at 270-275 s,
    # RA electrode artifact at 750-780 s (I, II, aVR; III clean). The
    # test recordings' VFib windows: no clean lead.
    # ------------------------------------------------------------------
    _ART_WIN_SEC = 10.0
    _ART_LOCAL_SEC = 3.0
    _ART_BLOCK_SEC = 2.0
    _ART_NOISY = 0.25
    _ART_FLAT = 0.08
    _ART_CLEAN = 0.15
    _ART_CLEAN_MIN_AMP = 0.30
    _ART_MIN_CLEAN_LEADS = 1
    _ART_MIN_SEC = 4.0
    # Artifact marking is switched off for now: no Artifact segments are
    # added. Set True to bring the rule back unchanged.
    _ARTIFACT_RULE_ON = False
    _art_label = 'Artifact'
    _art_beat_leads = [lead_names.index(n) for n in ('I', 'II', 'III', 'V1', 'V3', 'V5') if n in lead_names]

    def _artifact_blocks():
        """[(block_start_sec, block_end_sec, [noisy lead names])]"""
        out = []
        h0, h1 = int(0.25 * fs), int(0.45 * fs)
        qrs_half = int(0.08 * fs)
        _ab, _aa = _butter(2, [0.5 / (float(fs) / 2.0), 40.0 / (float(fs) / 2.0)], btype='band')
        t = 0.0
        while t < total_duration:
            w_end = min(total_duration, t + _ART_WIN_SEC)
            if w_end - t < _ART_BLOCK_SEC:
                break
            pad_lo, pad_hi = max(0.0, t - 1.0), min(total_duration, w_end + 1.0)
            data = reader.read_range(pad_lo, pad_hi)
            t_next = t + _ART_WIN_SEC
            if data is None or data.shape[1] < int(fs):
                t = t_next
                continue
            sig = np.asarray([_filtfilt(_ab, _aa, np.asarray(data[i], dtype=float))
                              for i in range(data.shape[0])])
            found = []
            for li in _art_beat_leads:
                if li < sig.shape[0]:
                    found.extend((int(p), li) for p in pan_tompkins(sig[li], fs=fs))
            found.sort()
            groups = []
            for p, li in found:
                if groups and p - groups[-1][0] <= int(0.08 * fs):
                    groups[-1][1].add(li)
                    groups[-1][2].append(p)
                else:
                    groups.append([p, {li}, [p]])
            peaks = [int(np.median(g[2])) for g in groups if len(g[1]) >= 3]
            peaks = [p for p in peaks if p - h0 >= 0 and p + h1 <= sig.shape[1]
                     and t <= pad_lo + p / fs < w_end]
            if len(peaks) < 3:
                t = t_next
                continue
            beat_t = np.asarray([pad_lo + p / fs for p in peaks])
            per_lead = []
            for li in range(sig.shape[0]):
                beats = np.asarray([sig[li][p - h0:p + h1] for p in peaks])
                beats = beats - np.median(beats, axis=1, keepdims=True)
                template = np.median(beats, axis=0)
                q_amp = float(np.ptp(template[h0 - qrs_half:h0 + qrs_half]))
                residual = []
                for j in range(len(peaks)):
                    near = np.abs(beat_t - beat_t[j]) <= _ART_LOCAL_SEC
                    local = np.median(beats[near], axis=0) if int(np.sum(near)) >= 3 else template
                    residual.append(float(np.sqrt(np.mean((beats[j] - local) ** 2))))
                residual_rms = np.asarray(residual)
                per_lead.append((q_amp, residual_rms, sig[li]))
            ref_amp = float(np.median([q for q, _, _ in per_lead])) or 1.0
            b0 = t
            while b0 + 0.5 * _ART_BLOCK_SEC <= w_end:
                b1 = min(w_end, b0 + _ART_BLOCK_SEC)
                sel = (beat_t >= b0) & (beat_t < b1)
                i0, i1 = int((b0 - pad_lo) * fs), int((b1 - pad_lo) * fs)
                noisy, clean = [], 0
                for li, (q_amp, res, x) in enumerate(per_lead):
                    flat = float(np.ptp(x[i0:i1])) < _ART_FLAT * ref_amp
                    res_b = float(np.sqrt(np.mean(res[sel] ** 2))) if np.any(sel) else None
                    if flat or (res_b is not None and res_b >= _ART_NOISY * ref_amp):
                        noisy.append(lead_names[li] if li < len(lead_names) else str(li))
                    elif (res_b is not None and q_amp >= _ART_CLEAN_MIN_AMP * ref_amp
                          and res_b < _ART_CLEAN * q_amp):
                        clean += 1
                if noisy and clean >= _ART_MIN_CLEAN_LEADS:
                    out.append((b0, b1, noisy))
                b0 = b1
            t = t_next
        return out

    try:
        _art_segments = []
        for b0, b1, leads in (_artifact_blocks() if _ARTIFACT_RULE_ON else []):
            if _art_segments and b0 <= _art_segments[-1]['end_sec'] + _ART_BLOCK_SEC + 1e-6:
                last = _art_segments[-1]
                last['end_sec'] = float(b1)
                last['artifact_leads'] = sorted(set(last['artifact_leads']) | set(leads),
                                                key=lambda n: lead_names.index(n) if n in lead_names else 99)
            else:
                _art_segments.append({'start_sec': float(b0), 'end_sec': float(b1), 'artifact_leads': list(leads)})
        _art_segments = [seg for seg in _art_segments
                         if seg['end_sec'] - seg['start_sec'] >= _ART_MIN_SEC - 1e-6]
        for seg in _art_segments:
            s_str, e_str = make_time_strs(seg['start_sec'], seg['end_sec'])
            seg.update({'label': _art_label, 'color': '#A0A0A0', 'start_time_str': s_str,
                        'end_time_str': e_str, 'detected_rhythm': _art_label})
        final_segments.extend(_art_segments)
        final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Artifact check error: {e}")

    # Mains interference (50 / 60 Hz) Artifact, RR interval by RR interval
    # (1 s steps where no beat is found): the share of the raw signal's
    # 1-150 Hz power that sits at 48-52 Hz or 58-62 Hz, over at least 1 s
    # around the interval, on leads I, II, V1, V5; the interval is Artifact
    # when that share is >= _MAINS_SHARE on >= 2 of them. Measured on a test
    # recording with 50 / 60 Hz switched on: clean stretches <= 0.05,
    # interference 0.3-0.94, on every lead at once. The rule above can't see
    # it (it filters to 0.5-40 Hz first and needs a clean lead). Only 10 s
    # blocks whose Lead II share reaches _MAINS_SCREEN (and the blocks next
    # to them) are walked interval by interval. Consecutive intervals of one
    # frequency merge; a stretch is kept when >= 1 s. Like the rule above, a
    # layer next to the rhythm labels: the measurements notch 50 / 60 Hz out
    # and the rhythm under it is still read. 'artifact_kind' = '50 Hz' /
    # '60 Hz'.
    def _is_layer_label(label):
        """Labels drawn next to the rhythm, not as one: bundle branch
        blocks and Artifact. The rhythm steps after this treat their time
        as free."""
        return 'bundle branch block' in label.lower() or label == _art_label

    _MAINS_SHARE = 0.2
    _MAINS_SCREEN = 0.05
    _MAINS_MIN_LEADS = 2
    _MAINS_MIN_SEC = 1.0
    try:
        from scipy.signal import welch as _welch
        _mains_leads = [lead_names.index(n) for n in ('I', 'II', 'V1', 'V5') if n in lead_names]
        _mains_ii = lead_names.index('II') if 'II' in lead_names else (_mains_leads[0] if _mains_leads else None)

        def _mains_share(x):
            x = np.asarray(x, dtype=float)
            if x.size < int(0.5 * fs):
                return 0.0, 0.0
            f, p = _welch(x - float(np.mean(x)), fs=fs, nperseg=min(x.size, int(fs)))
            total = float(p[(f >= 1.0) & (f <= 150.0)].sum())
            if total <= 0:
                return 0.0, 0.0
            return (float(p[(f >= 48.0) & (f <= 52.0)].sum()) / total,
                    float(p[(f >= 58.0) & (f <= 62.0)].sum()) / total)

        zones = []
        if _mains_ii is not None:
            b0 = 0.0
            while b0 < total_duration:
                b1 = min(total_duration, b0 + 10.0)
                blk = raw_reader.read_range(b0, b1)
                if blk is not None and blk.shape[1] >= int(fs) and max(_mains_share(blk[_mains_ii])) >= _MAINS_SCREEN:
                    z0, z1 = max(0.0, b0 - 10.0), min(total_duration, b1 + 10.0)
                    if zones and z0 <= zones[-1][1]:
                        zones[-1][1] = z1
                    else:
                        zones.append([z0, z1])
                b0 = b1
        mains_runs = []   # [start, end, kind]
        for z0, z1 in zones:
            points = [z0] + [float(p) for p in all_r_peaks if z0 < p < z1] + [z1]
            filled = [points[0]]
            for p in points[1:]:
                while p - filled[-1] > 1.5:
                    filled.append(filled[-1] + 1.0)
                filled.append(p)
            for a, b in zip(filled, filled[1:]):
                if b <= a:
                    continue
                mid = 0.5 * (a + b)
                lo, hi = max(0.0, min(a, mid - 0.5)), min(total_duration, max(b, mid + 0.5))
                data = raw_reader.read_range(lo, hi)
                if data is None:
                    continue
                shares = [_mains_share(data[i]) for i in _mains_leads if i < data.shape[0]]
                kind = None
                if sum(1 for s50, _ in shares if s50 >= _MAINS_SHARE) >= _MAINS_MIN_LEADS:
                    kind = '50 Hz'
                elif sum(1 for _, s60 in shares if s60 >= _MAINS_SHARE) >= _MAINS_MIN_LEADS:
                    kind = '60 Hz'
                if kind is None:
                    continue
                if mains_runs and mains_runs[-1][2] == kind and abs(mains_runs[-1][1] - a) < 1e-6:
                    mains_runs[-1][1] = b
                else:
                    mains_runs.append([a, b, kind])
        for a, b, kind in mains_runs:
            if b - a < _MAINS_MIN_SEC:
                continue
            s_str, e_str = make_time_strs(a, b)
            final_segments.append({
                'start_sec': float(a), 'end_sec': float(b), 'label': _art_label, 'color': '#A0A0A0',
                'start_time_str': s_str, 'end_time_str': e_str, 'detected_rhythm': _art_label,
                'artifact_kind': kind,
            })
        final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Mains Artifact check error: {e}")

    # ------------------------------------------------------------------
    # Sinus Arrhythmia: its own rule, run last, only in time no other label
    # covers -- so it never changes any other label's segments.
    # Sinus rhythm whose rate rises and falls (respiratory, mostly in the
    # young): every QRS still has its P wave at a constant PR, only the
    # cycle length changes. Per run of 10 consecutive beats:
    #   1. sinus conduction -- sinus_conduction_gate (the same P / PR / QRS
    #      gate as Sinus Bradycardia / Tachycardia);
    #   2. RR varies: longest - shortest >= 120 ms (the usual P-P > 0.12 s
    #      definition);
    #   3. gradually: no beat-to-beat change > 20% of the run's median RR
    #      and no RR outside 0.7-1.4x median -- a premature beat, a pause or
    #      a missed / extra R-peak is not sinus arrhythmia;
    #   4. phasic: >= 3 beat-to-beat changes of >= 40 ms, both lengthening
    #      and shortening -- the rate rises and falls. One rate step (seen:
    #      a test recording going 60 -> 70 bpm, RR 1000 -> 856 ms once, flat
    #      on both sides) is a rate change, not sinus arrhythmia.
    # Qualifying runs are merged (gap <= 3 s), cut out of any other label's
    # time, and kept when >= 8 s long.
    # ------------------------------------------------------------------
    _SA_BEATS = 10
    _SA_MIN_RANGE_MS = 120.0
    _SA_MAX_STEP = 0.20
    _SA_RR_BOUNDS = (0.7, 1.4)
    _SA_MIN_STEP_MS = 40.0
    _SA_MIN_STEPS = 3
    _SA_MERGE_GAP_SEC = 3.0
    _SA_MIN_SEC = 8.0
    _sa_label = 'Sinus Arrhythmia'

    def _sinus_arrhythmia_runs():
        runs = []
        for i in range(0, len(all_r_peaks) - _SA_BEATS + 1):
            j = i + _SA_BEATS - 1
            rr = np.diff(np.asarray(all_r_peaks[i:j + 1], dtype=float)) * 1000.0
            med = float(np.median(rr))
            if med <= 0:
                continue
            if float(rr.max() - rr.min()) < _SA_MIN_RANGE_MS:
                continue
            if rr.min() < _SA_RR_BOUNDS[0] * med or rr.max() > _SA_RR_BOUNDS[1] * med:
                continue
            steps = np.diff(rr)
            if steps.size and float(np.max(np.abs(steps))) > _SA_MAX_STEP * med:
                continue
            big = steps[np.abs(steps) >= _SA_MIN_STEP_MS]
            if big.size < _SA_MIN_STEPS or not (np.any(big > 0) and np.any(big < 0)):
                continue
            if not _sinus_conduction_gate(sinus_gate_metrics, i, j):
                continue
            s, e = all_r_peaks[i], all_r_peaks[j]
            if runs and s <= runs[-1][1] + _SA_MERGE_GAP_SEC:
                runs[-1][1] = max(runs[-1][1], e)
            else:
                runs.append([s, e])
        return runs

    try:
        _sa_covered = sorted((s['start_sec'], s['end_sec']) for s in final_segments
                             if s['label'] != _art_label)
        for sa_start, sa_end in _sinus_arrhythmia_runs():
            pieces = [(sa_start, sa_end)]
            for c_start, c_end in _sa_covered:
                next_pieces = []
                for p_start, p_end in pieces:
                    if c_end <= p_start or c_start >= p_end:
                        next_pieces.append((p_start, p_end))
                        continue
                    if c_start > p_start:
                        next_pieces.append((p_start, c_start))
                    if c_end < p_end:
                        next_pieces.append((c_end, p_end))
                pieces = next_pieces
            for p_start, p_end in pieces:
                if p_end - p_start < _SA_MIN_SEC:
                    continue
                s_str, e_str = make_time_strs(p_start, p_end)
                final_segments.append({
                    'start_sec': float(p_start),
                    'end_sec': float(p_end),
                    'label': _sa_label,
                    'color': '#7FFFD4',
                    'start_time_str': s_str,
                    'end_time_str': e_str,
                    'detected_rhythm': _sa_label,
                })
        final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Sinus Arrhythmia check error: {e}")

    # First-degree AV Block, beat by beat, in time no other rhythm label
    # holds. The window rule (_is_first_degree_av_block_window) needs a whole
    # 10 s window of it; a shorter run between two other rhythms never fills
    # one -- seen: PR 238 ms (std 1) on every beat 1042-1051.9 s, between a
    # Mobitz I ending 1042.0 s and a slow rhythm from 1051.9 s; each window
    # held a blocked P from one side or the other and the stretch got no
    # label. A beat joins the run when it has a P with PR 200-450 ms, its RR
    # gives HR < 100, and its PR / RR are within 25 ms / 15% of the run's
    # median; the first beat that isn't ends it. Kept with >= 5 beats (the
    # window rule's minimum). Sinus Bradycardia / Tachycardia / Arrhythmia
    # are cut where it lands (an AV block outranks a rate label, as in
    # suppress_rate_overlap).
    try:
        _rate_only = ('Sinus Bradycardia', 'Sinus Tachycardia', 'Sinus Arrhythmia', _AT_LABEL)
        other_rhythm = [
            (float(s['start_sec']), float(s['end_sec'])) for s in final_segments
            if s['label'] not in _rate_only and not _is_layer_label(s['label'])
        ]

        def _first_degree_beat(i):
            m = sinus_gate_metrics[i] if i < len(sinus_gate_metrics) else None
            if not m or m.get('pr_ms') is None or i == 0:
                return None
            pr = float(m['pr_ms'])
            rr = (all_r_peaks[i] - all_r_peaks[i - 1]) * 1000.0
            t = all_r_peaks[i]
            if not (200.0 < pr <= 450.0) or rr <= 600.0 \
                    or any(s - 0.01 <= t <= e + 0.01 for s, e in other_rhythm):
                return None
            return pr, rr

        first_degree_runs = []
        run = []
        for i in range(len(all_r_peaks)):
            beat = _first_degree_beat(i)
            if beat is not None and run:
                prs = [b[1][0] for b in run]
                rrs = [b[1][1] for b in run]
                if abs(beat[0] - float(np.median(prs))) <= 25.0 \
                        and abs(beat[1] - float(np.median(rrs))) <= 0.15 * float(np.median(rrs)):
                    run.append((i, beat))
                    continue
            if len(run) >= 5:
                first_degree_runs.append((all_r_peaks[run[0][0]], all_r_peaks[run[-1][0]]))
            run = [(i, beat)] if beat is not None else []
        if len(run) >= 5:
            first_degree_runs.append((all_r_peaks[run[0][0]], all_r_peaks[run[-1][0]]))

        if first_degree_runs:
            kept = []
            for seg in final_segments:
                if seg['label'] not in _rate_only:
                    kept.append(seg)
                    continue
                pieces = [(float(seg['start_sec']), float(seg['end_sec']))]
                for r0, r1 in first_degree_runs:
                    nxt = []
                    for p0, p1 in pieces:
                        if r1 <= p0 or r0 >= p1:
                            nxt.append((p0, p1))
                            continue
                        if r0 > p0:
                            nxt.append((p0, r0))
                        if r1 < p1:
                            nxt.append((r1, p1))
                    pieces = nxt
                for p0, p1 in pieces:
                    if p1 - p0 < 1.0:
                        continue
                    piece = dict(seg)
                    if (p0, p1) != (float(seg['start_sec']), float(seg['end_sec'])):
                        piece['start_sec'], piece['end_sec'] = p0, p1
                        piece['start_time_str'], piece['end_time_str'] = make_time_strs(p0, p1)
                    kept.append(piece)
            final_segments = kept
            for r0, r1 in first_degree_runs:
                s_str, e_str = make_time_strs(r0, r1)
                final_segments.append({
                    'start_sec': r0, 'end_sec': r1, 'label': _AV_FIRST, 'color': _AV_COLORS[_AV_FIRST],
                    'start_time_str': s_str, 'end_time_str': e_str, 'detected_rhythm': _AV_FIRST,
                })
            final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] First-degree AV Block beat check error: {e}")

    # Mobitz I and Third-degree AV Block, beat by beat, in time no other
    # rhythm label holds. Their window rules need a 10 s window holding the
    # block alone; an episode of 5-10 s between two other rhythms never gets
    # one (seen on 5 / 7 / 10 s pieces of a simulator sequence: Mobitz I
    # missed at every length, Third-degree read plain Bradycardia at 5-7 s).
    #   Mobitz I, one Wenckebach cycle at a time: >= 3 conducted beats whose
    #   PR rises every beat (by >= 60 ms in all; one beat with no PR -- the
    #   P lost in the T at the longest PR -- may close the run), then a pause
    #   of 1.3-2.0x the cycle's median RR (the blocked P), and the beat after
    #   it with a PR >= 60 ms shorter than the cycle's longest (the reset).
    #   Back-to-back cycles make one segment.
    #   Mobitz II, one pause at a time: 3 beats at a steady RR (within 10%),
    #   then a pause of 2x (one blocked P) or 3x (two blocked P) that RR
    #   (+/-15% / +/-10%), >= that many waves in the pause on Lead II (the
    #   blocked P waves plus the conducted one), and the P-peak-to-R time of
    #   the two beats before the pause and the one after within 20 ms (PR
    #   fixed). Measured from the P peak, not with measure_beat(): on a wide
    #   bundle branch block QRS that finds no PR at all (seen: RBBB at
    #   75 bpm, 2250 ms pauses with two blocked P waves, P peak to R 240 ms
    #   on every beat, left unlabelled).
    #   Third-degree: >= 4 beats, HR < 60, each RR within 15% of the run's
    #   median; the "PR" before each QRS scattered (std > 40 ms or range >
    #   80 ms over >= 2 measured -- P and QRS unrelated); and >= 2 waves
    #   per RR, >= 300 ms apart (more P than QRS, at an atrial rate below
    #   200/min -- Flutter's F waves are ~200 ms apart).
    # Sinus Bradycardia / Tachycardia / Arrhythmia are cut where they land.
    try:
        _rate_only = ('Sinus Bradycardia', 'Sinus Tachycardia', 'Sinus Arrhythmia', _AT_LABEL)
        # Every label blocks here, Sinus Bradycardia / Tachycardia /
        # Arrhythmia too: the sinus gate's consistent PR there is evidence
        # against a block (seen: 30 bpm Sinus Bradycardia, PR 172 ms on every
        # beat, read as Third-degree for 332 s off a few stray PRs).
        _taken = [
            (float(s['start_sec']), float(s['end_sec'])) for s in final_segments
            if not _is_layer_label(s['label'])
        ]
        _n = len(all_r_peaks)

        def _free(i):
            t = all_r_peaks[i]
            return not any(s - 0.01 <= t <= e + 0.01 for s, e in _taken)

        def _rr(i):
            return all_r_peaks[i] - all_r_peaks[i - 1]

        # PR without strict_pr (the sinus gate's own is strict and drops the
        # long PRs these blocks are made of -- seen: 356 ms at the end of a
        # Wenckebach cycle read as no PR), same filtering as the sinus gate,
        # measured only on beats no other label holds, 60 s at a time.
        _ns_pr = {}
        # free beats, plus the beat right after each (a Wenckebach cycle's
        # reset beat may be the first beat of the next segment)
        _free_idx = [q for q in range(_n) if _free(q) or (q > 0 and _free(q - 1))]
        _c = 0
        while _c < len(_free_idx):
            t_lo = all_r_peaks[_free_idx[_c]]
            chunk = [q for q in _free_idx[_c:] if all_r_peaks[q] < t_lo + 60.0]
            _c += len(chunk)
            r0 = max(0.0, t_lo - 1.5)
            r1 = min(total_duration, all_r_peaks[chunk[-1]] + 1.0)
            sig = reader.read_range(r0, r1)
            if sig is None or conduction_lead_idx >= sig.shape[0] or sig.shape[1] < int(fs):
                continue
            xs = _sinus_gate_filter(np.asarray(sig[conduction_lead_idx], dtype=float))
            for q in chunk:
                ri = int(round((all_r_peaks[q] - r0) * fs))
                if 0 < ri < len(xs):
                    b = measure_beat(xs, ri, fs, strict_pr=False,
                                     prev_rr_ms=(_rr(q) * 1000.0 if q > 0 else None))
                    if b and b.get('pr_ms') is not None:
                        _ns_pr[q] = float(b['pr_ms'])

        def _pr(i):
            return _ns_pr.get(i)

        new_av = []

        # Mobitz I cycles
        cycles = []
        i = 1
        while i < _n - 3:
            if not _free(i) or _pr(i) is None:
                i += 1
                continue
            k = i
            prs = [_pr(i)]
            while k + 1 < _n and _free(k + 1) and _pr(k + 1) is not None and _pr(k + 1) > prs[-1]:
                k += 1
                prs.append(_pr(k))
            j = k + 1
            if j < _n and _free(j) and _pr(j) is None:
                j += 1
            ok = False
            if len(prs) >= 3 and prs[-1] - prs[0] >= 60.0 and j < _n and (_free(j) or _free(j - 1)):
                cyc_rr = [_rr(q) for q in range(i + 1, j)]
                med = float(np.median(cyc_rr)) if cyc_rr else 0.0
                p_after = _pr(j)
                # no early beat inside the cycle: a premature beat and its
                # compensatory pause (seen: RR 750, 750, 562, 936) is not a
                # blocked P -- Wenckebach RRs stay close (810 / 930 / 820)
                ok = (med > 0 and min(cyc_rr) >= 0.8 * max(cyc_rr)
                      and 1.3 * med <= _rr(j) <= 2.0 * med
                      and p_after is not None and p_after <= max(prs) - 60.0)
            if ok:
                cycles.append((i, j))
                i = j
            else:
                i += 1
        for a, b in cycles:
            if new_av and new_av[-1][0] == _AV_MOBITZ_I and abs(new_av[-1][2] - all_r_peaks[a]) < 1e-6:
                new_av[-1][2] = all_r_peaks[b]
            else:
                new_av.append([_AV_MOBITZ_I, all_r_peaks[a], all_r_peaks[b]])

        # Mobitz II pauses
        _mob1 = [(s, e) for _, s, e in new_av]

        def _pause_evidence(j):
            """(waves between the T wave after beat j-1 and beat j's QRS,
            P-peak-to-R time in ms of beats j-2, j-1, j -- None where no P)."""
            t0, t1 = all_r_peaks[j - 2] - 0.5, all_r_peaks[j] + 0.5
            sig = reader.read_range(max(0.0, t0), min(total_duration, t1))
            if sig is None or conduction_lead_idx >= sig.shape[0]:
                return 0, [None] * 3
            x = np.asarray(sig[conduction_lead_idx], dtype=float)
            _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 30.0 / (float(fs) / 2.0)], btype='band')
            x = _filtfilt(_b, _a, x)
            base = max(0.0, t0)

            def _peaks(lo_s, hi_s):
                i0, i1 = int((lo_s - base) * fs), int((hi_s - base) * fs)
                if i1 - i0 < int(0.1 * fs) or i0 < 0 or i1 > len(x):
                    return None, []
                seg = x[i0:i1]
                if np.ptp(seg) <= 0:
                    return None, []
                pk, props = find_peaks(seg, prominence=0.3 * float(np.ptp(seg)), distance=int(0.3 * fs))
                return i0, list(zip(pk, props['prominences']))

            waves = 0
            if all_r_peaks[j] - all_r_peaks[j - 1] - 0.51 >= 0.3:
                _, pk = _peaks(all_r_peaks[j - 1] + 0.45, all_r_peaks[j] - 0.06)
                waves = len(pk)
            p_to_r = []
            for q in (j - 2, j - 1, j):
                i0, pk = _peaks(all_r_peaks[q] - 0.35, all_r_peaks[q] - 0.06)
                if not pk:
                    p_to_r.append(None)
                    continue
                best = max(pk, key=lambda v: v[1])[0]
                p_to_r.append((all_r_peaks[q] - (base + (i0 + best) / fs)) * 1000.0)
            return waves, p_to_r

        for j in range(4, _n):
            run = list(range(j - 3, j + 1))
            if not all(_free(q) for q in run):
                continue
            if any(s - 0.01 <= all_r_peaks[q] <= e + 0.01 for q in run for s, e in _mob1):
                continue
            pre_rr = [_rr(q) for q in range(j - 2, j)]
            base_rr = float(np.median(pre_rr))
            if base_rr <= 0 or max(pre_rr) - min(pre_rr) > 0.1 * base_rr:
                continue
            k = 2 if 1.7 * base_rr <= _rr(j) <= 2.3 * base_rr else (
                3 if 2.7 * base_rr <= _rr(j) <= 3.3 * base_rr else 0)
            if not k:
                continue
            waves, prs = _pause_evidence(j)
            if waves < k or any(v is None for v in prs) or max(prs) - min(prs) > 20.0:
                continue
            r0, r1 = all_r_peaks[j - 2], all_r_peaks[j]
            if new_av and new_av[-1][0] == _AV_MOBITZ_II and r0 <= new_av[-1][2] + 1e-6:
                new_av[-1][2] = r1
            else:
                new_av.append([_AV_MOBITZ_II, r0, r1])

        # Third-degree runs
        _mob = [(s, e) for _, s, e in new_av]

        def _free3(i):
            return _free(i) and not any(s - 0.01 <= all_r_peaks[i] <= e + 0.01 for s, e in _mob)

        def _third_degree_run(run):
            prs = [_pr(q) for q in run if _pr(q) is not None]
            if len(prs) < 2 or not (float(np.std(prs)) > 40.0 or max(prs) - min(prs) > 80.0):
                return False
            # scattered, not one PR with a stray or two: fewer than half
            # within 20 ms of the median
            med_pr = float(np.median(prs))
            if sum(1 for v in prs if abs(v - med_pr) <= 20.0) >= 0.5 * len(prs):
                return False
            t0, t1 = all_r_peaks[run[0]] - 0.5, all_r_peaks[run[-1]] + 0.5
            sig = reader.read_range(max(0.0, t0), min(total_duration, t1))
            if sig is None or conduction_lead_idx >= sig.shape[0]:
                return False
            x = np.asarray(sig[conduction_lead_idx], dtype=float)
            _b, _a = _butter(2, [0.5 / (float(fs) / 2.0), 30.0 / (float(fs) / 2.0)], btype='band')
            x = _filtfilt(_b, _a, x)
            base = max(0.0, t0)
            waves = []
            for q in run[1:]:
                i0 = int((all_r_peaks[q - 1] + 0.40 - base) * fs)
                i1 = int((all_r_peaks[q] - 0.06 - base) * fs)
                if i1 - i0 < int(0.3 * fs) or i0 < 0 or i1 > len(x):
                    continue
                seg = x[i0:i1]
                pk, _ = find_peaks(seg, prominence=0.3 * float(np.ptp(seg)) if np.ptp(seg) > 0 else 1.0,
                                   distance=int(0.3 * fs))
                waves.append(len(pk))
            return len(waves) >= 3 and float(np.median(waves)) >= 2

        run = []
        for q in range(1, _n):
            ok = _free3(q) and _rr(q) > 1.0
            if ok and run:
                med = float(np.median([_rr(z) for z in run]))
                ok = abs(_rr(q) - med) <= 0.15 * med
                if not ok and _free3(q) and _rr(q) > 1.0:
                    if len(run) >= 4 and _third_degree_run(run):
                        new_av.append([_AV_THIRD, all_r_peaks[run[0]], all_r_peaks[run[-1]]])
                    run = [q]
                    continue
            if ok:
                run.append(q)
                continue
            if len(run) >= 4 and _third_degree_run(run):
                new_av.append([_AV_THIRD, all_r_peaks[run[0]], all_r_peaks[run[-1]]])
            run = []
        if len(run) >= 4 and _third_degree_run(run):
            new_av.append([_AV_THIRD, all_r_peaks[run[0]], all_r_peaks[run[-1]]])

        new_av = [r for r in new_av if r[2] - r[1] >= 2.0]
        # only on readable signal (pqrst_measure.frame_status 'ok' over
        # every 10 s of it): a noisy stretch reads as scattered PRs too
        try:
            from .pqrst_measure import frame_status as _fs_status

            def _readable(t0, t1):
                t = t0
                while t < t1 - 0.5:
                    if _fs_status(reader, t, min(t1, t + 10.0)) != 'ok':
                        return False
                    t += 10.0
                return True

            new_av = [r for r in new_av if _readable(r[1], r[2])]
        except Exception:
            new_av = []
        if new_av:
            kept = []
            for seg in final_segments:
                if seg['label'] not in _rate_only:
                    kept.append(seg)
                    continue
                pieces = [(float(seg['start_sec']), float(seg['end_sec']))]
                for _, r0, r1 in new_av:
                    nxt = []
                    for p0, p1 in pieces:
                        if r1 <= p0 or r0 >= p1:
                            nxt.append((p0, p1))
                            continue
                        if r0 > p0:
                            nxt.append((p0, r0))
                        if r1 < p1:
                            nxt.append((r1, p1))
                    pieces = nxt
                for p0, p1 in pieces:
                    if p1 - p0 < 1.0:
                        continue
                    piece = dict(seg)
                    if (p0, p1) != (float(seg['start_sec']), float(seg['end_sec'])):
                        piece['start_sec'], piece['end_sec'] = p0, p1
                        piece['start_time_str'], piece['end_time_str'] = make_time_strs(p0, p1)
                    kept.append(piece)
            final_segments = kept
            for label, r0, r1 in new_av:
                touching = next((s for s in final_segments if s['label'] == label and (
                    abs(float(s['start_sec']) - r1) < 0.02 or abs(float(s['end_sec']) - r0) < 0.02)), None)
                if touching is not None:
                    touching['start_sec'] = min(float(touching['start_sec']), r0)
                    touching['end_sec'] = max(float(touching['end_sec']), r1)
                    touching['start_time_str'], touching['end_time_str'] = make_time_strs(
                        touching['start_sec'], touching['end_sec'])
                    continue
                s_str, e_str = make_time_strs(r0, r1)
                final_segments.append({
                    'start_sec': r0, 'end_sec': r1, 'label': label, 'color': _AV_COLORS[label],
                    'start_time_str': s_str, 'end_time_str': e_str, 'detected_rhythm': label,
                })
            final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Mobitz I / Third-degree beat check error: {e}")

    # Unidentified: stretches with no heart signal that P / QRS / T can be
    # read from (pqrst_measure.frame_status): a simulator test signal (the
    # same waveform on every lead -- seen: triangle / pulse trains read as
    # Tachycardia 1276-1287 s, a square wave read as VFib 2005-2020 s) or
    # Lead II too noisy. Judged RR interval by RR interval
    # (pqrst_measure.span_status, 2 s of signal around each; 1 s steps where
    # no beat is found), so the stretch starts and ends on a beat -- was 10 s
    # frames, which took in the normal beats before a test signal. Only
    # frames that are unreadable, and the frames next to them, are walked
    # beat by beat. A VFib region that is mostly test signal is dropped (the
    # VFib rule fires on a square wave's steps); a stretch inside any other
    # rhythm label (VFib, AFib, an AV block ... -- the simulator's VFib is
    # the same on every lead too) keeps that label. Rate labels (Sinus
    # Bradycardia / Tachycardia / Arrhythmia) are cut out of it, and plain
    # Bradycardia / Tachycardia below never start in it.
    def _cut_spans(spans, blockers):
        """`spans` minus `blockers`, pieces of >= 2 s."""
        out = []
        for s0, s1 in spans:
            pieces = [(s0, s1)]
            for b0, b1 in blockers:
                nxt = []
                for p0, p1 in pieces:
                    if b1 <= p0 or b0 >= p1:
                        nxt.append((p0, p1))
                        continue
                    if b0 > p0:
                        nxt.append((p0, b0))
                    if b1 < p1:
                        nxt.append((b1, p1))
                pieces = nxt
            out.extend(p for p in pieces if p[1] - p[0] >= 2.0)
        return out

    unreadable_spans = []
    try:
        from .pqrst_measure import frame_status, span_status, FRAME_SEC, UNIDENTIFIED, UNIDENTIFIED_COLOR
        _rate_labels = ('Sinus Bradycardia', 'Sinus Tachycardia', 'Sinus Arrhythmia', _AT_LABEL)
        zones = []
        f0 = 0.0
        while f0 < total_duration:
            f1 = min(f0 + FRAME_SEC, total_duration)
            if f1 - f0 >= 2.0 and frame_status(reader, f0, f1) != 'ok':
                z0, z1 = max(0.0, f0 - FRAME_SEC), min(total_duration, f1 + FRAME_SEC)
                if zones and z0 <= zones[-1][1]:
                    zones[-1][1] = z1
                else:
                    zones.append([z0, z1])
            f0 = f1
        test_spans = []
        for z0, z1 in zones:
            points = [z0] + [float(p) for p in all_r_peaks if z0 < p < z1] + [z1]
            filled = [points[0]]
            for p in points[1:]:
                while p - filled[-1] > 1.5:
                    filled.append(filled[-1] + 1.0)
                filled.append(p)
            for a, b in zip(filled, filled[1:]):
                if b - a <= 0:
                    continue
                status = span_status(reader, a, b)
                if status == 'ok':
                    continue
                if unreadable_spans and abs(unreadable_spans[-1][1] - a) < 1e-6:
                    unreadable_spans[-1][1] = b
                else:
                    unreadable_spans.append([a, b])
                if status == 'test_signal':
                    if test_spans and abs(test_spans[-1][1] - a) < 1e-6:
                        test_spans[-1][1] = b
                    else:
                        test_spans.append([a, b])
        unreadable_spans = [(a, b) for a, b in unreadable_spans if b - a >= 2.0]

        def _test_overlap(seg):
            s0, s1 = float(seg['start_sec']), float(seg['end_sec'])
            return sum(max(0.0, min(s1, b) - max(s0, a)) for a, b in test_spans)

        final_segments = [
            seg for seg in final_segments
            if not (seg['label'] == 'Ventricular Fibrillation'
                    and _test_overlap(seg) >= 0.5 * (float(seg['end_sec']) - float(seg['start_sec'])))
        ]

        # A VFib segment that survives the drop above (< 50% test_signal)
        # can still have a test-signal (square / pulse-train) piece inside
        # it: the check above only flags a whole FRAME_SEC (10 s) frame, so
        # a short test-signal stretch sharing a 10 s frame with genuine
        # chaotic beats on either side never raises a zone there and
        # test_spans never sees it (seen: a 6 s VFib call, 69.95-75.95 s --
        # 69.95-73.0 s was a plain square wave, only 73.0-75.95 s was real
        # chaotic VFib; the whole 10-20 s frame read 'ok' because the
        # chaotic tail broke the same-every-lead correlation for the frame
        # as a whole, so neither zones nor test_spans ever formed there).
        # Each surviving VFib segment is rechecked here on its own, in
        # beat-snapped pieces, and any piece span_status reads as
        # 'test_signal' is cut out of it -- this is a square/pulse-train
        # wave (same on every lead AND flat steps), never a real beat.
        # Genuine chaotic VFib is unaffected: it differs lead to lead, so
        # span_status reads it 'ok', never 'test_signal'.
        # The cut piece itself is kept, relabelled Unidentified (>= 2 s,
        # same floor the rest of this pass uses) rather than just dropped:
        # left unclaimed, it has nowhere else to go (the zones/test_spans
        # pass above never saw it either, for the same 10 s-frame reason),
        # and fell through to the generic rate-based Tachycardia/
        # Bradycardia label below, which reads a square wave's own regular
        # steps as a plausible rate just as easily as real beats.
        _vfib_trimmed = []
        for seg in final_segments:
            if seg['label'] != 'Ventricular Fibrillation':
                _vfib_trimmed.append(seg)
                continue
            s0, s1 = float(seg['start_sec']), float(seg['end_sec'])
            points = [s0] + [float(p) for p in all_r_peaks if s0 < p < s1] + [s1]
            filled = [points[0]]
            for p in points[1:]:
                while p - filled[-1] > 1.5:
                    filled.append(filled[-1] + 1.0)
                filled.append(p)
            cut_spans = []
            for a, b in zip(filled, filled[1:]):
                if b - a <= 0 or span_status(reader, a, b) != 'test_signal':
                    continue
                if cut_spans and abs(cut_spans[-1][1] - a) < 1e-6:
                    cut_spans[-1][1] = b
                else:
                    cut_spans.append([a, b])
            # Only a test-signal stretch of >= 2 s is cut (a square / pulse
            # train runs for seconds): a lone 1 s piece reading test_signal
            # is the flat join between VFib and a flatline, and cutting it
            # just split one VFib episode into pieces with 1 s holes (seen:
            # 594.3-595.3 s and 607.3-608.3 s).
            cut_spans = [c for c in cut_spans if c[1] - c[0] >= 2.0]
            pieces = [(s0, s1)]
            for a, b in cut_spans:
                nxt = []
                for p0, p1 in pieces:
                    if b <= p0 or a >= p1:
                        nxt.append((p0, p1))
                        continue
                    if a > p0:
                        nxt.append((p0, a))
                    if b < p1:
                        nxt.append((b, p1))
                pieces = nxt
            for p0, p1 in pieces:
                if p1 - p0 < 1.0:
                    continue
                piece = dict(seg)
                if (p0, p1) != (s0, s1):
                    piece['start_sec'], piece['end_sec'] = p0, p1
                    piece['start_time_str'], piece['end_time_str'] = make_time_strs(p0, p1)
                _vfib_trimmed.append(piece)
            for c0, c1 in cut_spans:
                if c1 - c0 < 2.0:
                    continue
                c_s_str, c_e_str = make_time_strs(c0, c1)
                _vfib_trimmed.append({
                    'start_sec': c0, 'end_sec': c1, 'label': UNIDENTIFIED, 'color': UNIDENTIFIED_COLOR,
                    'start_time_str': c_s_str, 'end_time_str': c_e_str, 'detected_rhythm': UNIDENTIFIED,
                })
        final_segments = sorted(_vfib_trimmed, key=lambda x: x['start_sec'])

        own_rhythm = [
            (float(s['start_sec']), float(s['end_sec'])) for s in final_segments
            if s['label'] not in _rate_labels and not _is_layer_label(s['label'])
        ]
        unidentified = _cut_spans(unreadable_spans, own_rhythm)
        if unidentified:
            kept = []
            for seg in final_segments:
                if seg['label'] not in _rate_labels:
                    kept.append(seg)
                    continue
                pieces = [(float(seg['start_sec']), float(seg['end_sec']))]
                for u0, u1 in unidentified:
                    nxt = []
                    for p0, p1 in pieces:
                        if u1 <= p0 or u0 >= p1:
                            nxt.append((p0, p1))
                            continue
                        if u0 > p0:
                            nxt.append((p0, u0))
                        if u1 < p1:
                            nxt.append((u1, p1))
                    pieces = nxt
                for p0, p1 in pieces:
                    if p1 - p0 < 1.0:
                        continue
                    piece = dict(seg)
                    if (p0, p1) != (float(seg['start_sec']), float(seg['end_sec'])):
                        piece['start_sec'], piece['end_sec'] = p0, p1
                        piece['start_time_str'], piece['end_time_str'] = make_time_strs(p0, p1)
                    kept.append(piece)
            final_segments = kept
            for u0, u1 in unidentified:
                s_str, e_str = make_time_strs(u0, u1)
                final_segments.append({
                    'start_sec': u0,
                    'end_sec': u1,
                    'label': UNIDENTIFIED,
                    'color': UNIDENTIFIED_COLOR,
                    'start_time_str': s_str,
                    'end_time_str': e_str,
                    'detected_rhythm': UNIDENTIFIED,
                })
            final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Unidentified check error: {e}")

    # Plain Bradycardia / Tachycardia: rate runs no other label claimed (no
    # sinus P/PR/QRS, so not Sinus Bradycardia / Tachycardia). Added last, on
    # beats outside every rhythm segment above, so it never shortens or
    # blocks another label (e.g. the Atrial Flutter 2:1 extension into
    # uncovered time). See _plain_rate_segments.
    try:
        rhythm_spans = [
            (float(s['start_sec']), float(s['end_sec'])) for s in final_segments
            if not _is_layer_label(s['label'])
        ]
        plain_brady_peaks = brady_r_peaks if len(all_rr_ms) >= 3 else all_r_peaks
        final_segments.extend(_plain_rate_segments(
            plain_brady_peaks, lambda hr: _PLAIN_BRADY_HR[0] < hr < _PLAIN_BRADY_HR[1],
            2.5, 'Bradycardia', '#20B2AA', rhythm_spans, make_time_strs))
        final_segments.extend(_plain_rate_segments(
            all_r_peaks, lambda hr: hr > _PLAIN_TACHY_HR,
            1.0, 'Tachycardia', '#F08080', rhythm_spans, make_time_strs))
        final_segments.sort(key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Bradycardia / Tachycardia check error: {e}")

    # One RR interval between two segments, no beat inside it: segments end
    # on R peaks, so the interval from one rhythm's last beat to the next
    # one's first beat belonged to neither and was drawn unlabelled (seen:
    # Third-degree escape 182.81 s -> LBBB 185.09 s; an RBBB run split by a
    # 2250 ms pause with two blocked P waves). The earlier segment runs on
    # to the next one's start; the same label on both sides joins into one.
    # Bundle branch block is paired only with bundle branch block, a rhythm
    # only with a rhythm (the two are drawn side by side). Up to 3 s: a
    # longer gap with no beat is a pause of its own.
    try:
        _peaks_sorted = sorted(float(p) for p in all_r_peaks)

        def _beat_inside(t0, t1):
            k = bisect.bisect_right(_peaks_sorted, t0 + 0.05)
            return k < len(_peaks_sorted) and _peaks_sorted[k] < t1 - 0.05

        def _close_single_interval_gaps(segs):
            segs = sorted(segs, key=lambda s: s['start_sec'])
            out = []
            for seg in segs:
                if out:
                    last = out[-1]
                    gap = float(seg['start_sec']) - float(last['end_sec'])
                    if 0.0 < gap <= 3.0 and not _beat_inside(float(last['end_sec']), float(seg['start_sec'])):
                        if last['label'] == seg['label']:
                            last['end_sec'] = max(float(last['end_sec']), float(seg['end_sec']))
                            _, last['end_time_str'] = make_time_strs(last['start_sec'], last['end_sec'])
                            continue
                        last['end_sec'] = float(seg['start_sec'])
                        _, last['end_time_str'] = make_time_strs(last['start_sec'], last['end_sec'])
                out.append(seg)
            return out

        _bbb = [s for s in final_segments if 'Bundle Branch Block' in s['label']]
        _rhythm = [s for s in final_segments if 'Bundle Branch Block' not in s['label']
                   and not _is_layer_label(s['label']) and s['label'] != 'Unidentified']
        _other = [s for s in final_segments if s not in _bbb and s not in _rhythm]
        _rhythm = _close_single_interval_gaps(_rhythm)
        _bbb = _close_single_interval_gaps(_bbb)
        # Then across the two kinds: a beatless interval that no label of
        # either kind covers (Third-degree ending 182.81 s, LBBB from
        # 185.09 s) goes to the segment that ends where it starts.
        _labelled = sorted(_rhythm + _bbb, key=lambda s: s['start_sec'])
        _cover_end = None
        for seg in _labelled:
            if _cover_end is not None:
                gap = float(seg['start_sec']) - float(_cover_end['end_sec'])
                if 0.0 < gap <= 3.0 and not _beat_inside(float(_cover_end['end_sec']), float(seg['start_sec'])):
                    _cover_end['end_sec'] = float(seg['start_sec'])
                    _, _cover_end['end_time_str'] = make_time_strs(_cover_end['start_sec'], _cover_end['end_sec'])
            if _cover_end is None or float(seg['end_sec']) >= float(_cover_end['end_sec']):
                _cover_end = seg
        final_segments = sorted(_rhythm + _bbb + _other, key=lambda s: s['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Single-interval gap error: {e}")

    # ------------------------------------------------------------------
    # Noise gate, applied to every label above: no auto marking where the
    # signal is not clean. Per 10 s block, every lead's SQI is computed on
    # the raw samples (analysis_pipeline.SignalQuality: flat / clipping
    # outside the QRS / EMG / baseline wander); a block is noisy and is cut
    # out of every segment when Lead II or the R-peak lead is below sqi_min,
    # or at least _NOISE_MIN_FAILING_LEADS leads are (_noise_cuts_block). All leads flat
    # at once is no signal rather than noise, so it is not cut. A piece left
    # shorter than 2 s (or than the whole original segment) is dropped.
    # Simulator recordings (incl. VFib / AFib / Flutter) had no noisy block.
    # ------------------------------------------------------------------
    _NOISE_BLOCK_SEC = 10.0
    _NOISE_READ_SEC = 60.0
    _NOISE_MIN_PIECE_SEC = 2.0
    noisy_blocks = []
    # Blocks where any one lead fails (the gate's old rule): they still cut
    # the plain Bradycardia / Tachycardia labels -- rate only, the weakest
    # evidence -- and their cut pieces read Unidentified. Seen: electrode
    # artifact on every lead at 57-60 s that SQI does not flag on Lead II
    # (10 s block 0.97, each 2 s piece >= 0.96); only V6 failed (0.54), and
    # its swings read as a 2.9 s plain Bradycardia without this.
    any_lead_noisy_blocks = []
    plain_noise_cut_spans = []
    try:
        from .analysis_pipeline import SignalQuality, HolterConfig, DERIVED_LEAD_SOURCES
        _nq = SignalQuality(fs=fs, config=HolterConfig.from_clinical_config())
        # III/aVR/aVL/aVF are not measured leads: they're computed from I/II
        # (DERIVED_LEAD_SOURCES), so whatever noise they carry is already
        # I/II's own noise. Scored on their own ADC samples they can still
        # fail sqi_min purely from being small (III/aVL's amplitude in a
        # given frontal axis can be a fraction of I/II's, e.g. one real
        # recording: I/II SQI 0.71/0.81, aVL SQI 0.45 on the SAME noise --
        # ~0.1 mV in each, just a bigger fraction of aVL's own ~1.4 mV
        # peak-to-peak -- cutting 20 s of otherwise clean Mobitz I /
        # First-degree AV Block labels). Only the source leads actually
        # measured are checked here; a genuinely noisy I or II still cuts
        # the block (derived leads inherit the same noise and fail too).
        noisy_blocks.clear()
        any_lead_noisy_blocks.clear()
        # Noisy blocks where at least one failing lead is flat (a flatline,
        # not chaos) -- the only ones that still cut VFib / VT, see below.
        flat_noisy_blocks = []
        # Scored after a 25 Hz zero-phase low-pass (SQI itself notches 50 /
        # 60 Hz): noise the display filters take out does not count, only
        # what is left after them -- wander, clipping, a flat lead. Seen:
        # EMG-like noise on I / V1 only (SQI 0.31-0.54 raw), Lead II clean,
        # cut 20 s of Flutter / Sinus Tachycardia.
        _noise_lp = _butter(2, 25.0 / (float(fs) / 2.0), btype='lowpass')
        r0 = 0.0
        while r0 < total_duration and final_segments:
            r1 = min(total_duration, r0 + _NOISE_READ_SEC)
            data = reader.read_range(r0, r1)
            if data is not None and data.shape[1] > int(0.1 * fs):
                data = np.asarray(data, dtype=float)
                for _li in range(data.shape[0]):
                    if _li < len(lead_names) and lead_names[_li] not in DERIVED_LEAD_SOURCES:
                        try:
                            data[_li] = _filtfilt(_noise_lp[0], _noise_lp[1], data[_li])
                        except Exception:
                            pass
                b0 = r0
                while b0 < r1:
                    b1 = min(r1, b0 + _NOISE_BLOCK_SEC)
                    i0, i1 = int(round((b0 - r0) * fs)), int(round((b1 - r0) * fs))
                    block = data[:, i0:i1]
                    if block.shape[1] >= int(2.0 * fs):
                        leads = {lead_names[i]: block[i] for i in range(min(len(lead_names), block.shape[0]))
                                 if lead_names[i] not in DERIVED_LEAD_SOURCES}
                        noisy = _nq.noisy_leads(leads)
                        if noisy:
                            # Any failing lead: still cuts the plain rate
                            # labels and (flat) VFib / VT, as before.
                            if any_lead_noisy_blocks and abs(any_lead_noisy_blocks[-1][1] - b0) < 1e-6:
                                any_lead_noisy_blocks[-1][1] = b1
                            else:
                                any_lead_noisy_blocks.append([b0, b1])
                            if _noise_cuts_block(noisy, lead_names[best_lead_idx]
                                                 if best_lead_idx < len(lead_names) else None):
                                if noisy_blocks and abs(noisy_blocks[-1][1] - b0) < 1e-6:
                                    noisy_blocks[-1][1] = b1
                                else:
                                    noisy_blocks.append([b0, b1])
                            _names, _blk, _clip = _nq._lead_block(leads)
                            _scores, _flat = _nq.lead_scores(_blk, _clip)
                            if any(bool(f) for n, f in zip(_names, _flat) if n in noisy):
                                if flat_noisy_blocks and abs(flat_noisy_blocks[-1][1] - b0) < 1e-6:
                                    flat_noisy_blocks[-1][1] = b1
                                else:
                                    flat_noisy_blocks.append([b0, b1])
                    b0 = b1
            r0 = r1
        if noisy_blocks or any_lead_noisy_blocks:
            gated = []
            for seg in final_segments:
                # Unidentified already marks a stretch with nothing to read
                # (noisy, or no heart signal): it stays. Ventricular
                # Fibrillation / Tachycardia are cut only by a block whose
                # failing lead is flat (flat_noisy_blocks), not by one that
                # fails from chaos: their own evidence is
                # an unresolved/organized-wide QRS on the detection leads
                # (_is_vfib_window / _is_ventricular_tachycardia_window),
                # not a clean per-lead SQI -- VFib's genuinely chaotic
                # waveform is exactly what SQI's own HF-noise/wander checks
                # are built to flag, on ANY one of all 12 leads, including
                # ones the rule never looked at. Seen: a real VFib episode
                # (window-confirmed, beat-exact edges 0.0-14.2 s) cut to
                # 10.0-14.2 s because lead V6 alone read SQI 0.37 (< 0.6)
                # over 0-10 s, while I/II/aVL/aVF/V1-V5 were all >= 0.7 --
                # the 1.0-10.0 s remnant then had no label at all and read
                # back as Unidentified. VFib's own test-signal/non-ECG
                # override (_test_overlap above) and the fibrillating-piece
                # check (_vfib_has_fibrillating_piece) are untouched and
                # still the real guards against a false VFib/VT call; this
                # exemption only removes a second, unrelated SQI cut that a
                # genuine episode can fail purely by being chaotic. AFib /
                # Flutter / AV blocks / SVT / BBB / the rate labels keep
                # going through this gate as before -- no evidence yet that
                # they are cut the same way, and AV blocks / SVT lean on
                # fine P-wave timing that a truly noisy lead can corrupt.
                # A flat failing lead is no signal, not chaos, so it still
                # cuts VFib / VT as before (seen: VFib 595.4-607.3 s, then a
                # flatline on 6-8 of 8 leads -- without this, VFib ran on
                # over the flatline to 645 s instead of being cut at 630 s;
                # the real VFib blocks above failed on one non-flat lead).
                if seg['label'] == 'Unidentified':
                    gated.append(seg)
                    continue
                if seg['label'] in ('Ventricular Fibrillation', 'Ventricular Tachycardia'):
                    cut_blocks = flat_noisy_blocks
                elif seg['label'] in ('Bradycardia', 'Tachycardia'):
                    cut_blocks = any_lead_noisy_blocks
                    plain_noise_cut_spans.extend(
                        [max(n0, float(seg['start_sec'])), min(n1, float(seg['end_sec']))]
                        for n0, n1 in cut_blocks
                        if n0 < float(seg['end_sec']) and n1 > float(seg['start_sec']))
                else:
                    cut_blocks = noisy_blocks
                pieces = [(float(seg['start_sec']), float(seg['end_sec']))]
                for n0, n1 in cut_blocks:
                    nxt = []
                    for p0, p1 in pieces:
                        if n1 <= p0 or n0 >= p1:
                            nxt.append((p0, p1))
                            continue
                        if n0 > p0:
                            nxt.append((p0, n0))
                        if n1 < p1:
                            nxt.append((n1, p1))
                    pieces = nxt
                min_len = min(_NOISE_MIN_PIECE_SEC, float(seg['end_sec']) - float(seg['start_sec']))
                for p0, p1 in pieces:
                    if p1 - p0 < min_len - 1e-6:
                        continue
                    piece = dict(seg)
                    if (p0, p1) != (float(seg['start_sec']), float(seg['end_sec'])):
                        s_str, e_str = make_time_strs(p0, p1)
                        piece.update({'start_sec': p0, 'end_sec': p1, 'start_time_str': s_str, 'end_time_str': e_str})
                    gated.append(piece)
            print(f"[Auto Arrhythmia Detect] Noise gate: {len(noisy_blocks)} noisy stretch(es), "
                  f"{sum(b - a for a, b in noisy_blocks):.0f}s; segments {len(final_segments)} -> {len(gated)}")
            final_segments = sorted(gated, key=lambda x: x['start_sec'])
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Noise gate error: {e}")

    # Unidentified, second pass: an unreadable stretch left with no rhythm
    # label once the noise gate cut it -- it was skipped above because a
    # label covered it then (seen: VFib to 650 s cut back to 630 s, 630-650 s
    # noisy and blank).
    try:
        from .pqrst_measure import UNIDENTIFIED, UNIDENTIFIED_COLOR
        added = False
        labelled = [(float(s['start_sec']), float(s['end_sec'])) for s in final_segments
                    if not _is_layer_label(s['label'])]
        # A block the noise gate cut is unreadable too: shown as Unidentified,
        # not left blank.
        _unread = sorted([tuple(s) for s in unreadable_spans] + [tuple(b) for b in noisy_blocks]
                         + [tuple(b) for b in plain_noise_cut_spans])
        _merged_unread = []
        for s0, s1 in _unread:
            if _merged_unread and s0 <= _merged_unread[-1][1] + 1e-6:
                _merged_unread[-1][1] = max(_merged_unread[-1][1], s1)
            else:
                _merged_unread.append([s0, s1])
        for f0, f1 in _cut_spans([tuple(s) for s in _merged_unread], labelled):
            s_str, e_str = make_time_strs(f0, f1)
            final_segments.append({
                'start_sec': f0, 'end_sec': f1, 'label': UNIDENTIFIED, 'color': UNIDENTIFIED_COLOR,
                'start_time_str': s_str, 'end_time_str': e_str, 'detected_rhythm': UNIDENTIFIED,
            })
            added = True
        if added:
            final_segments.sort(key=lambda x: x['start_sec'])
            merged = []
            for seg in final_segments:
                last = merged[-1] if merged else None
                if last is not None and seg['label'] == UNIDENTIFIED and last['label'] == UNIDENTIFIED \
                        and seg['start_sec'] <= last['end_sec'] + 1e-6:
                    last['end_sec'] = max(last['end_sec'], seg['end_sec'])
                    last['end_time_str'] = seg['end_time_str']
                    continue
                merged.append(seg)
            final_segments = merged
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Unidentified second pass error: {e}")

    # Last step: no two rhythm labels over the same time (layer labels --
    # bundle branch blocks, Artifact -- are drawn alongside and are left
    # alone). See _resolve_rhythm_overlaps.
    try:
        final_segments = _resolve_rhythm_overlaps(final_segments, _is_layer_label, make_time_strs)
    except Exception as e:
        print(f"[Auto Arrhythmia Detect] Rhythm overlap guard error: {e}")

    print(f"[Auto Arrhythmia Detect] Found {len(final_segments)} exact rhythm segments:")
    for s in final_segments:
        print(f"  - {s['label']}: {s['start_sec']:.2f}s to {s['end_sec']:.2f}s ({s['start_time_str']} - {s['end_time_str']})")

    return final_segments


# Noise gate: a 10 s block is cut only when the leads the rhythm is read
# from fail (Lead II -- P / PR -- or the R-peak lead), or when at least this
# many measured leads fail at once. Was: any one lead. Seen on INCART
# 12-lead Holter records (real patients): one lead with baseline wander (I at
# SQI 0.60, V3 0.67, the other 10 at 0.82-0.95) cut 270 of 300 s of a
# record whose beats were found 100%, and 24 of 72 records read mostly
# Unidentified.
_NOISE_MIN_FAILING_LEADS = 3


def _noise_cuts_block(failing_leads, rpeak_lead):
    """True when the noise gate's failing leads (below sqi_min) are enough
    to cut the block -- see _NOISE_MIN_FAILING_LEADS."""
    failing = set(failing_leads)
    return 'II' in failing or (rpeak_lead in failing) or len(failing) >= _NOISE_MIN_FAILING_LEADS


# Which rhythm label keeps a stretch two of them were both drawn over, most
# severe / most specific first: ventricular, then AV blocks by degree (a
# Second-degree block always beats First-degree), then the atrial rhythms,
# then the rate-only labels; Unidentified (signal not readable) last. A label
# not listed ranks just above Unidentified.
_RHYTHM_PRIORITY = (
    'Ventricular Fibrillation',
    'Ventricular Tachycardia',
    'Third-degree AV Block',
    'Second-degree AV Block (Mobitz II)',
    'Second-degree AV Block (2:1)',
    'Second-degree AV Block (Mobitz I)',
    'Atrial Fibrillation',
    'Atrial Flutter',
    'Supraventricular Tachycardia (SVT)',
    'Atrial Tachycardia',
    'First-degree AV Block',
    'Sinus Tachycardia',
    'Sinus Bradycardia',
    'Sinus Arrhythmia',
    'Tachycardia',
    'Bradycardia',
)
_RHYTHM_OVERLAP_MIN_SEC = 1.0


def _resolve_rhythm_overlaps(segments, is_layer_label, make_time_strs):
    """Guard on the detector's final output: where two rhythm segments of
    different labels overlap, the one higher in _RHYTHM_PRIORITY keeps the
    time and the other is cut there; a piece left under
    _RHYTHM_OVERLAP_MIN_SEC is dropped. A segment no other one overlaps is
    returned unchanged. Seen: Unidentified 150.0-152.3 s drawn over the end
    of a Ventricular Tachycardia run."""
    def rank(label):
        if label in _RHYTHM_PRIORITY:
            return _RHYTHM_PRIORITY.index(label)
        return len(_RHYTHM_PRIORITY) + (1 if label == 'Unidentified' else 0)

    layer = [seg for seg in segments if is_layer_label(seg['label'])]
    rhythm = sorted((seg for seg in segments if not is_layer_label(seg['label'])),
                    key=lambda seg: (rank(seg['label']), float(seg['start_sec'])))
    kept = []
    for seg in rhythm:
        s0, s1 = float(seg['start_sec']), float(seg['end_sec'])
        blockers = [(float(k['start_sec']), float(k['end_sec'])) for k in kept
                    if k['label'] != seg['label']
                    and float(k['start_sec']) < s1 - 1e-6 and float(k['end_sec']) > s0 + 1e-6]
        if not blockers:
            kept.append(seg)
            continue
        pieces = [(s0, s1)]
        for b0, b1 in sorted(blockers):
            nxt = []
            for p0, p1 in pieces:
                if b1 <= p0 or b0 >= p1:
                    nxt.append((p0, p1))
                    continue
                if b0 > p0:
                    nxt.append((p0, b0))
                if b1 < p1:
                    nxt.append((b1, p1))
            pieces = nxt
        for p0, p1 in pieces:
            if p1 - p0 < _RHYTHM_OVERLAP_MIN_SEC:
                continue
            piece = dict(seg)
            piece['start_sec'], piece['end_sec'] = p0, p1
            piece['start_time_str'], piece['end_time_str'] = make_time_strs(p0, p1)
            kept.append(piece)
    return sorted(kept + layer, key=lambda seg: float(seg['start_sec']))


# Bradycardia, sinus or not: HR < 60 (textbook), as RR >= 1010 ms. A 60 bpm
# simulator rhythm reads RR 998-1002 ms beat to beat; a bound at exactly
# 1000 ms would mark plain 60 bpm sinus rhythm as bradycardia on every beat
# that came out at 1001-1002 ms. Was RR >= 1050 ms (HR < 57.1).
_BRADY_MIN_RR_MS = 1010.0
_PLAIN_BRADY_HR = (10.0, 60000.0 / _BRADY_MIN_RR_MS)    # Bradycardia: 10 < HR < 59.4 bpm
# Tachycardia: HR > 100 bpm, no upper limit. Above 180 bpm the SVT rules
# get there first (their segments block this label); a capped band left
# beats alternating 179.6 / 180.7 bpm (RR 334 / 332 ms) with no label at all.
_PLAIN_TACHY_HR = 100.0


# Sinus gate P search (sinus_gate_metrics and its missed-beat recovery): the
# P-on-T collision zone starts no earlier than this long after the previous
# R, so the previous beat's own QRS is never read as a wave colliding with
# the P. Only changes anything when RR < 333 + 80 ms (HR > ~145): slower,
# the zone already starts after the previous QRS. Seen: 175 bpm sinus
# tachycardia, P upright before every QRS (PR ~100 ms), rejected on every
# beat, labelled plain Tachycardia.
_SINUS_P_SKIP_PREV_QRS_MS = 80.0


def _set_p_span_sec(metric, chunk_start_sec, fs):
    """P onset / end as recording seconds (_p_on_sec / _p_end_sec) on a
    sinus gate metric, for the P axis read in _split_atrial_tachycardia."""
    if metric.get('p_onset') is not None and metric.get('p_end') is not None:
        metric['_p_on_sec'] = chunk_start_sec + float(metric['p_onset']) / float(fs)
        metric['_p_end_sec'] = chunk_start_sec + float(metric['p_end']) / float(fs)


# Atrial Tachycardia vs Sinus Tachycardia, by the frontal P axis -- Philips
# DXL ECG Algorithm Physician's Guide, p. 2-2: a normal P axis (-30 to 120
# deg) is a sinus P, an abnormal one an atrial or junctional origin; p. B-6:
# "Ectopic atrial tachycardia, unifocal ... abnormal P axis, V-rate > **",
# and "Sinus or ectopic atrial tachycardia ... P axis (-45, 135)" for the
# borderline band. Only an axis outside -45..135 is called atrial; the
# borderline band stays Sinus Tachycardia. Read the way axis_detection.py
# reads the QRS axis: net P area (onset -> end, over a straight onset-end
# baseline) in lead I (0 deg) and aVF (+90 deg), axis =
# atan2(2/sqrt(3) x area aVF, area I). Sinus P here reads ~49 deg.
_AT_LABEL = 'Atrial Tachycardia'
_AT_COLOR = '#FF7F50'
_AT_SINUS_P_AXIS_DEG = (-45.0, 135.0)
_AT_MIN_BEATS = 3
_AT_MIN_SEC = 1.0


def _split_atrial_tachycardia(segments, peaks, metrics, reader, fs, sinus_gate_filter, make_time_strs):
    """Sinus Tachycardia segments (from _append_sinus_rate_segments, on
    `peaks` / `metrics`) with the runs of beats whose P axis is abnormal
    relabelled Atrial Tachycardia. A beat with no P axis takes its
    neighbours' call. A run of >= _AT_MIN_BEATS abnormal beats over >=
    _AT_MIN_SEC is Atrial Tachycardia, from the R before its first beat
    (that RR ends in the ectopic P) to its last R. A Sinus Tachycardia
    piece too short for that between two such runs joins them; one at the
    segment's edge is dropped. Segments are returned unchanged when lead I
    or aVF is missing."""
    names = [str(n) for n in (getattr(reader, 'lead_names', None) or [])]
    if not segments or 'I' not in names or 'aVF' not in names:
        return segments
    i_idx, avf_idx = names.index('I'), names.index('aVF')
    k_avf = 2.0 / np.sqrt(3.0)
    lo_deg, hi_deg = _AT_SINUS_P_AXIS_DEG

    def _area(x, on, end):
        piece = x[on:end + 1]
        return float(np.sum(piece - np.linspace(piece[0], piece[-1], piece.size)))

    out = []
    for seg in segments:
        s_sec, e_sec = float(seg['start_sec']), float(seg['end_sec'])
        idx = [k for k, p in enumerate(peaks) if s_sec - 1e-6 <= p <= e_sec + 1e-6]
        if len(idx) < _AT_MIN_BEATS:
            out.append(seg)
            continue
        r0 = max(0.0, s_sec - 1.0)
        data = reader.read_range(r0, min(float(reader.duration_sec), e_sec + 1.0))
        if data is None or max(i_idx, avf_idx) >= data.shape[0]:
            out.append(seg)
            continue
        lead_i = sinus_gate_filter(np.asarray(data[i_idx], dtype=float))
        lead_avf = sinus_gate_filter(np.asarray(data[avf_idx], dtype=float))

        # Per beat: True = abnormal P axis, False = sinus, None = no axis.
        calls = []
        for k in idx:
            m = metrics[k] if k < len(metrics) else None
            call = None
            if m and m.get('_p_on_sec') is not None:
                on = int(round((m['_p_on_sec'] - r0) * fs))
                end = int(round((m['_p_end_sec'] - r0) * fs))
                if on >= 0 and end - on >= 2 and end < lead_i.size:
                    a_i, a_f = _area(lead_i, on, end), _area(lead_avf, on, end)
                    if a_i or a_f:
                        axis = float(np.degrees(np.arctan2(k_avf * a_f, a_i)))
                        call = not (lo_deg <= axis <= hi_deg)
            calls.append(call)
        if not any(c for c in calls if c is not None):
            out.append(seg)
            continue
        for j in range(len(calls)):
            if calls[j] is None and j > 0:
                calls[j] = calls[j - 1]
        for j in range(len(calls) - 1, -1, -1):
            if calls[j] is None:
                calls[j] = calls[j + 1] if j + 1 < len(calls) else False

        # Runs of one call, as [call, first, last] positions in idx.
        runs = []
        for j, c in enumerate(calls):
            if runs and runs[-1][0] == c:
                runs[-1][2] = j
            else:
                runs.append([c, j, j])

        def _long(r):
            return (r[2] - r[1] + 1 >= _AT_MIN_BEATS
                    and peaks[idx[r[2]]] - peaks[idx[r[1]]] >= _AT_MIN_SEC)

        for r in runs:
            if r[0] and not _long(r):
                r[0] = False
        for j in range(1, len(runs) - 1):
            if not runs[j][0] and runs[j - 1][0] and runs[j + 1][0] and not _long(runs[j]):
                runs[j][0] = True
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] = r[2]
            else:
                merged.append(list(r))
        if not any(r[0] for r in merged):
            out.append(seg)
            continue
        # Pieces touch: each starts on the R the previous one ended on.
        for n, (c, a, b) in enumerate(merged):
            p0 = peaks[idx[a - 1]] if a > 0 else peaks[idx[a]]
            p1 = peaks[idx[b]]
            if not c and (b - a + 1 < _AT_MIN_BEATS or p1 - p0 < _AT_MIN_SEC):
                continue
            piece = dict(seg)
            piece['start_sec'], piece['end_sec'] = p0, p1
            piece['start_time_str'], piece['end_time_str'] = make_time_strs(p0, p1)
            if c:
                piece['label'] = piece['detected_rhythm'] = _AT_LABEL
                piece['color'] = _AT_COLOR
            out.append(piece)
    return out


# A beat only Pan-Tompkins' shape match at a beat-size step found
# (pan_tompkins STEP_SHAPE_*) is kept only when another peak-detection lead
# has a beat within this distance: an amplitude change moves every lead
# together, noise on one lead does not. Seen: V6 alone full of narrow-band
# noise (lead II clean sinus) on 2026-10-30_14-55-20 -- the match took 9
# noise peaks there for beats and a 10 s label changed.
_SHAPE_BEAT_AGREE_SEC = 0.08


def _keep_confirmed_shape_beats(beats, beats_without_shape, other_leads_beats, fs):
    """beats_without_shape plus each beat of `beats` not in it that some
    lead in other_leads_beats (sample indices, same array) also has within
    _SHAPE_BEAT_AGREE_SEC."""
    base = sorted(int(p) for p in beats_without_shape)
    tol = _SHAPE_BEAT_AGREE_SEC * fs
    others = [np.asarray(o, dtype=float) for o in other_leads_beats if len(o)]
    extra = sorted(set(int(p) for p in beats) - set(base))
    keep = [p for p in extra if any(float(np.min(np.abs(o - p))) <= tol for o in others)]
    return np.array(sorted(base + keep), dtype=int)


# Beats the best lead's Pan-Tompkins missed, recovered when every other
# peak-detection lead finds one within this distance of each other.
_MISSED_BEAT_AGREE_SEC = 0.08


def _beats_missed_on_best_lead(chunk_data, best_lead_idx, lead_indices, peaks_so_far,
                               pad_start, t_cursor, t_end, fs, pan_tompkins):
    """R-peaks inside [t_cursor, t_end) of one Pan-Tompkins chunk that the
    best lead missed but the other peak-detection leads (at least two) all
    found, within _MISSED_BEAT_AGREE_SEC of each other and >= 0.20 s from
    every best-lead peak. Returned at the median of those leads' times; the
    best lead's own peaks are never moved or dropped. The best lead is
    picked once from the first 30 s by amplitude -- seen: that stretch was
    VFib and noise, lead II won, and its Pan-Tompkins found 0 of 52 RBBB
    beats (lead II R ~+200, S ~-550, after narrow beats with R ~+1000)
    that V5, V6, V1 and I each found."""
    others = [i for i in lead_indices if i != best_lead_idx and i < chunk_data.shape[0]]
    if len(others) < 2:
        return []
    per_lead = []
    for li in others:
        sig = np.asarray(chunk_data[li], dtype=float)
        if len(sig) <= 50:
            return []
        per_lead.append(sorted(pad_start + float(p) / float(fs) for p in pan_tompkins(sig, fs=fs)))
    existing = sorted(p for p in peaks_so_far if t_cursor - 1.0 <= p < t_end + 1.0)
    added = []
    for t in per_lead[0]:
        if t < t_cursor or t >= t_end:
            continue
        times = [t]
        for lead_peaks in per_lead[1:]:
            k = bisect.bisect_left(lead_peaks, t)
            near = [lead_peaks[j] for j in (k - 1, k) if 0 <= j < len(lead_peaks)
                    and abs(lead_peaks[j] - t) <= _MISSED_BEAT_AGREE_SEC]
            if not near:
                break
            times.append(min(near, key=lambda x: abs(x - t)))
        if len(times) < len(per_lead):
            continue
        beat = float(np.median(times))
        if all(abs(beat - p) >= 0.20 for p in existing + added):
            added.append(beat)
    return added


def _qrs_ok_for_sinus(b):
    """QRS evidence for a sinus beat: bounds resolved, or -- when a P wave
    with a measured PR leads into it -- the fallback width alone. Seen:
    140 bpm sinus tachycardia, P / PR 136-140 ms on every beat, QRS 110 ms,
    but 64 of 167 beats with qrs_bounds_resolved False (the T wave too
    close for the slope test): the run broke into 25 pieces alternating
    Sinus Tachycardia / plain Tachycardia."""
    return bool(b.get('qrs_bounds_resolved') or b.get('pr_ms') is not None)


def _sinus_beat(b):
    """One beat's own sinus evidence: a P with onset/end, a QRS of 40-200
    ms (see _qrs_ok_for_sinus) and a measured PR --
    _sinus_conduction_gate()'s per-beat "valid" test plus the PR it needs."""
    return bool(
        b
        and b.get('p_present')
        and b.get('p_onset') is not None
        and b.get('p_end') is not None
        and _qrs_ok_for_sinus(b)
        and b.get('q_onset') is not None
        and 40.0 <= float(b.get('qrs_ms') or 0.0) <= 200.0
        and b.get('pr_ms') is not None
    )


def _sinus_conduction_gate(metrics, start_idx, end_idx):
    """Require organized P-to-QRS conduction before a rate label.

    Bradycardia/Tachycardia are sinus labels, therefore an RR pattern by
    itself must not create them. Most beats in the candidate must have a
    real P before QRS, a valid and consistent PR, and a resolved,
    consistent QRS. P polarity/duration consistency is an additional
    guard against counting random baseline bumps as P waves.

    metrics: per-beat measurements the indices refer to -- sinus_gate_metrics
    for Sinus Tachycardia and Sinus Arrhythmia, the Sinus Bradycardia rule's
    own (possibly missed-beat-recovered) list for Sinus Bradycardia. Every
    caller passes its own list explicitly; there is no default here.
    """
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
        and _qrs_ok_for_sinus(b)
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


def _append_sinus_rate_segments(peaks, metrics, start_idx, end_idx, min_dur_sec, label, color, make_time_strs):
    """Sinus Tachycardia / Bradycardia judged beat by beat inside one
    rate run (beats start_idx..end_idx of `peaks`): a beat that fails
    _sinus_beat() ends the current segment and the next passing beat
    starts a new one. Each segment needs >= 3 beats, >= min_dur_sec and
    _sinus_conduction_gate() on its own beats (PR / QRS / P-P / P shape
    consistency). Was: the gate on the whole run at once, so one bad
    stretch dropped the good one with it -- seen: Sinus Tachycardia 150
    bpm 100.1-148.6 s, 37 of 122 beats valid (92 needed) because P was
    rejected on every beat after 120 s, while 101-111 s alone had 21
    of 25 and got no label.

    Returns the new segments (the caller extends its own list with them --
    this does not append to anything itself)."""
    segments = []
    k = start_idx
    while k <= end_idx:
        if not _sinus_beat(metrics[k]):
            k += 1
            continue
        seg_start = k
        while k + 1 <= end_idx and _sinus_beat(metrics[k + 1]):
            k += 1
        seg_end = k
        k += 1
        if seg_end - seg_start + 1 >= 3 \
                and peaks[seg_end] - peaks[seg_start] >= min_dur_sec \
                and _sinus_conduction_gate(metrics, seg_start, seg_end):
            s_sec = peaks[seg_start]
            e_sec = peaks[seg_end]
            s_str, e_str = make_time_strs(s_sec, e_sec)
            segments.append({
                'start_sec': s_sec,
                'end_sec': e_sec,
                'label': label,
                'color': color,
                'start_time_str': s_str,
                'end_time_str': e_str,
                'detected_rhythm': label
            })
    return segments


def _suppress_sinus_rate_overlap(rate_segments, morph_segments, make_time_strs):
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


def _sinus_rate_segments(reader, fs, best_lead_idx, conduction_lead_idx, total_duration,
                         all_r_peaks, all_rr_ms, sinus_gate_metrics, sinus_gate_filter, make_time_strs,
                         tachy_min_rr_ms=340.0):
    """Beat-exact Sinus Tachycardia / Sinus Bradycardia segments for the
    whole recording (both built the same way: a first pass through the
    RR-rate window, then _append_sinus_rate_segments() re-checks every
    candidate run beat by beat through the P/PR/QRS sinus gate). Needs
    >= 3 whole-recording beats; returns [] otherwise. Pulled out of
    _detect_arrhythmias_uncached() as a pure extraction -- same
    computation, same order, same inputs, only where it lives changed."""
    from ecg.arrhythmia_detector import measure_beat
    try:
        from ecg.pan_tompkins import pan_tompkins
    except ImportError:
        from ..pan_tompkins import pan_tompkins

    segments = []
    if len(all_rr_ms) < 3:
        # brady_r_peaks is returned too: the plain Bradycardia/Tachycardia
        # pass later reuses the (possibly missed-beat-recovered) Sinus
        # Bradycardia beat list as its own Bradycardia candidate beats.
        # Unused by the caller in this branch (same as before extraction).
        return segments, all_r_peaks

    # A. Sinus Tachycardia: first pass through the P/PR/QRS sinus gate,
    # then consecutive beats with RR <= 600ms (HR >= 100 bpm).
    #
    # tachy_min_rr_ms: 60000 / (220 - age) with a patient age -- above that
    # rate it is SVT (_SVT_DEFAULT_MIN_HR) -- else 340 ms.
    # Lower bound 340 ms, not 333.3: _detect_p_wave suppresses P whenever
    # the previous QRS sits within 333.3 ms, and RR at ~180 bpm jitters
    # around that edge. Those P-less beats sat inside the run and failed
    # the 75% P gate for the whole run, dropping valid 100-175 bpm sinus
    # tachycardia along with them.
    def _fast_ranges(rr_ms):
        ranges = []
        in_run = False
        start = None
        for i, rr in enumerate(rr_ms):
            is_fast = (tachy_min_rr_ms <= rr <= 600.5) or (rr <= 630.0 and in_run and i + 1 < len(rr_ms) and tachy_min_rr_ms <= rr_ms[i + 1] <= 600.5)
            if is_fast:
                if not in_run:
                    in_run = True
                    start = i
            else:
                if in_run:
                    ranges.append((start, i))
                    in_run = False
        if in_run:
            ranges.append((start, len(rr_ms)))
        return ranges

    _tachy_ranges = _fast_ranges(all_rr_ms)

    # Missed-beat recovery for this Sinus Tachycardia rule, reactive
    # instead of a blanket pre-scan the way Sinus Bradycardia's below
    # is: a blanket scan over every RR that COULD be 2-4 merged fast
    # beats would also match huge stretches of plain NSR -- a 700 ms
    # NSR RR and two merged ~350 ms beats are the same number; only
    # re-reading the raw signal tells them apart, and doing that for
    # every such RR in a whole recording is far more work than this
    # rare chunk-boundary artifact (same root cause as the Sinus
    # Bradycardia recovery's own comment below: pan_tompkins() widens
    # its refractory period to a third of the 60 s chunk's MEDIAN RR;
    # a chunk that is mostly a slow rhythm right before a fast run
    # starts gets a refractory period wide enough to drop every other
    # beat of that fast run -- confirmed directly: two 428 ms RR
    # intervals read as one 856 ms, even three as one 1286 ms, with no
    # mains/signal noise anywhere near it, purely from a preceding ~150 s
    # 41 bpm run dominating that 60 s chunk's own median -- 173.1-180.4 s
    # read as if it were still under 100 bpm, and Sinus Tachycardia
    # started 7.3 s late).
    # Instead: once a run above is already confirmed fast, walk backward
    # from its own start while the RR interval just before it divides
    # evenly (within 15%) by 2, 3 or 4 into the run's own established
    # rate -- re-read just that one gap (+/-3 s, same technique as the
    # Sinus Bradycardia recovery) and re-run Pan-Tompkins locally, where
    # the gap's own short local median RR keeps the refractory period
    # correctly tight. Beats found whose resulting intervals are
    # themselves in the fast band are spliced in and the run is
    # extended one gap further back; this repeats until a gap doesn't
    # divide cleanly, or nothing new turns up, or the signal can't be
    # read. all_r_peaks itself is left untouched, so every other rule
    # sees the same beats as before.
    _tachy_recovered = {}
    for start_idx, end_idx in _tachy_ranges:
        run_rrs = all_rr_ms[start_idx:end_idx]
        if not run_rrs:
            continue
        run_rr = float(np.median(run_rrs))
        if run_rr <= 0:
            continue
        j = start_idx
        while j > 0:
            prev_rr = all_rr_ms[j - 1]
            k = round(prev_rr / run_rr)
            if k < 2 or k > 4 or abs(prev_rr - k * run_rr) > 0.15 * k * run_rr:
                break
            gap_start, gap_end = all_r_peaks[j - 1], all_r_peaks[j]
            read_start = max(0.0, gap_start - 3.0)
            read_end = min(total_duration, gap_end + 3.0)
            gap_data = reader.read_range(read_start, read_end)
            if gap_data is None or best_lead_idx >= gap_data.shape[0]:
                break
            gap_sig = np.asarray(gap_data[best_lead_idx], dtype=float)
            if len(gap_sig) <= 50:
                break
            found = sorted(
                read_start + float(p_idx) / float(fs)
                for p_idx in pan_tompkins(gap_sig, fs=fs)
                if gap_start + 0.25 < read_start + float(p_idx) / float(fs) < gap_end - 0.25
            )
            if not found:
                break
            new_points = [gap_start] + found + [gap_end]
            new_rrs = [1000.0 * (b - a) for a, b in zip(new_points, new_points[1:])]
            if not all(tachy_min_rr_ms <= x <= 630.0 for x in new_rrs):
                break
            _tachy_recovered.setdefault(j - 1, []).extend(found)
            j -= 1

    if _tachy_recovered:
        tachy_r_peaks = []
        for idx, p in enumerate(all_r_peaks):
            tachy_r_peaks.append(p)
            tachy_r_peaks.extend(sorted(_tachy_recovered.get(idx, [])))
        tachy_r_peaks = sorted(set(tachy_r_peaks))
        _metric_at = dict(zip(all_r_peaks, sinus_gate_metrics))
        tachy_beat_metrics = []
        for k, peak_sec in enumerate(tachy_r_peaks):
            if peak_sec in _metric_at:
                tachy_beat_metrics.append(_metric_at[peak_sec])
                continue
            # Same measurement as sinus_gate_metrics above: conduction
            # lead, strict P/PR, 0.5 Hz high-pass + 25 Hz low-pass.
            metric = None
            m_start = max(0.0, peak_sec - 3.0)
            m_end = min(total_duration, peak_sec + 3.0)
            m_data = reader.read_range(m_start, m_end)
            if m_data is not None and conduction_lead_idx < m_data.shape[0]:
                m_sig = sinus_gate_filter(np.asarray(m_data[conduction_lead_idx], dtype=float))
                local_peak = int(round((peak_sec - m_start) * fs))
                prev_rr_ms = (peak_sec - tachy_r_peaks[k - 1]) * 1000.0 if k > 0 else None
                if 0 < local_peak < len(m_sig):
                    metric = measure_beat(m_sig, local_peak, fs, strict_pr=True, prev_rr_ms=prev_rr_ms,
                                          p_collision_skip_prev_qrs_ms=_SINUS_P_SKIP_PREV_QRS_MS)
                    if metric and metric.get('p_peak') is not None:
                        metric['_p_time_sec'] = m_start + float(metric['p_peak']) / float(fs)
                        _set_p_span_sec(metric, m_start, fs)
                    if metric:
                        metric['_r_time_sec'] = peak_sec
            tachy_beat_metrics.append(metric)
        tachy_rr_ms = [
            (tachy_r_peaks[k + 1] - tachy_r_peaks[k]) * 1000.0
            for k in range(len(tachy_r_peaks) - 1)
        ]
        _tachy_ranges = _fast_ranges(tachy_rr_ms)
    else:
        tachy_r_peaks, tachy_beat_metrics = all_r_peaks, sinus_gate_metrics

    tachy_segments = []
    for start_idx, end_idx in _tachy_ranges:
        tachy_segments.extend(_append_sinus_rate_segments(
            tachy_r_peaks, tachy_beat_metrics, start_idx, end_idx,
            1.0, 'Sinus Tachycardia', '#00FFFF', make_time_strs))
    segments.extend(_split_atrial_tachycardia(
        tachy_segments, tachy_r_peaks, tachy_beat_metrics, reader, fs,
        sinus_gate_filter, make_time_strs))

    # B. Sinus Bradycardia: first pass through the same P/PR/QRS sinus
    # gate, then consecutive beats with RR >= _BRADY_MIN_RR_MS (HR < 60,
    # the textbook bound -- was RR >= 1050 ms, HR < 57.1).
    # Note: 60 bpm (1000ms RR) is Normal Sinus Rhythm (NSR: 60-100 bpm), NOT Bradycardia.
    # Brady range: RR _BRADY_MIN_RR_MS-6000ms. Anything slower is treated
    # as a pause/possible signal-loss interval rather than sustained sinus
    # bradycardia.
    #
    # An RR this "slow" must also stay under a ceiling, or a stretch
    # where Pan-Tompkins simply finds no peaks at all -- because the
    # rhythm has degenerated into chaos, e.g. a VFib onset -- reads as
    # one single multi-second "RR interval" and clears the brady RR bar
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
        if not (_BRADY_MIN_RR_MS <= rr <= _brady_max_rr_ms):
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
        _metric_at = dict(zip(all_r_peaks, sinus_gate_metrics))
        brady_beat_metrics = []
        for k, peak_sec in enumerate(brady_r_peaks):
            if peak_sec in _metric_at:
                brady_beat_metrics.append(_metric_at[peak_sec])
                continue
            # Same measurement as sinus_gate_metrics above: conduction
            # lead, strict P/PR, 0.5 Hz high-pass + 25 Hz low-pass; 3 s
            # of signal on each side so the high-pass has room to settle.
            metric = None
            m_start = max(0.0, peak_sec - 3.0)
            m_end = min(total_duration, peak_sec + 3.0)
            m_data = reader.read_range(m_start, m_end)
            if m_data is not None and conduction_lead_idx < m_data.shape[0]:
                m_sig = sinus_gate_filter(np.asarray(m_data[conduction_lead_idx], dtype=float))
                local_peak = int(round((peak_sec - m_start) * fs))
                prev_rr_ms = (peak_sec - brady_r_peaks[k - 1]) * 1000.0 if k > 0 else None
                if 0 < local_peak < len(m_sig):
                    metric = measure_beat(m_sig, local_peak, fs, strict_pr=True, prev_rr_ms=prev_rr_ms,
                                          p_collision_skip_prev_qrs_ms=_SINUS_P_SKIP_PREV_QRS_MS)
                    if metric and metric.get('p_peak') is not None:
                        metric['_p_time_sec'] = m_start + float(metric['p_peak']) / float(fs)
                        _set_p_span_sec(metric, m_start, fs)
                    if metric:
                        metric['_r_time_sec'] = peak_sec
            brady_beat_metrics.append(metric)
        brady_rr_ms = [
            (brady_r_peaks[k + 1] - brady_r_peaks[k]) * 1000.0
            for k in range(len(brady_r_peaks) - 1)
        ]
    else:
        brady_r_peaks, brady_beat_metrics, brady_rr_ms = all_r_peaks, sinus_gate_metrics, all_rr_ms

    in_brady = False
    brady_start_idx = None

    for i, rr in enumerate(brady_rr_ms):
        next_rr = brady_rr_ms[i + 1] if i + 1 < len(brady_rr_ms) else None
        is_slow = (
            (_BRADY_MIN_RR_MS <= rr <= _brady_max_rr_ms)
            or (
                _BRADY_MIN_RR_MS - 30.0 <= rr <= _brady_max_rr_ms and in_brady
                and next_rr is not None and _BRADY_MIN_RR_MS <= next_rr <= _brady_max_rr_ms
            )
        )
        if is_slow:
            if not in_brady:
                in_brady = True
                brady_start_idx = i
        else:
            if in_brady:
                segments.extend(_append_sinus_rate_segments(
                    brady_r_peaks, brady_beat_metrics, brady_start_idx, i,
                    2.5, 'Sinus Bradycardia', '#00BFFF', make_time_strs))
                in_brady = False
                brady_start_idx = None

    if in_brady and brady_start_idx is not None:
        segments.extend(_append_sinus_rate_segments(
            brady_r_peaks, brady_beat_metrics, brady_start_idx, len(brady_rr_ms),
            2.5, 'Sinus Bradycardia', '#00BFFF', make_time_strs))

    return segments, brady_r_peaks


def _plain_rate_segments(peaks, hr_ok, min_dur_sec, label, color, blocked_spans, make_time_strs):
    """Plain 'Bradycardia' / 'Tachycardia' segments, beat by beat: a beat
    joins the current segment while its RR gives hr_ok(60000 / RR) and the
    RR lies outside every blocked span (the other rhythm segments); the
    first beat that fails ends the segment. Kept with >= 3 beats over
    >= min_dur_sec, edges on the R peaks. Rate only -- no P/PR/QRS check:
    a run with sinus evidence is already Sinus Bradycardia / Tachycardia
    and is among the blocked spans.

    One beat whose own rate misses hr_ok by under 1 bpm (it sits on the
    cutoff) does not end the run when the beat after it qualifies again --
    seen: a steady 100 bpm simulator run, RR alternating 598 / 600 ms, split
    into three 1 s Tachycardia pieces because every 600 ms beat read exactly
    100.0 bpm, not > 100. A beat in a blocked span still ends the run."""
    def not_blocked(i):
        a, b = peaks[i], peaks[i + 1]
        return not any(a < s_end and b > s_start for s_start, s_end in blocked_spans)

    def beat_ok(i):
        rr_ms = (peaks[i + 1] - peaks[i]) * 1000.0
        if rr_ms <= 0 or not hr_ok(60000.0 / rr_ms):
            return False
        return not_blocked(i)

    def on_cutoff(i):
        rr_ms = (peaks[i + 1] - peaks[i]) * 1000.0
        if rr_ms <= 0:
            return False
        hr = 60000.0 / rr_ms
        return (hr_ok(hr + 1.0) or hr_ok(hr - 1.0)) and not_blocked(i)

    segments = []
    n = len(peaks) - 1
    i = 0
    while i < n:
        if not beat_ok(i):
            i += 1
            continue
        first = i
        while i + 1 < n:
            if beat_ok(i + 1):
                i += 1
            elif i + 2 < n and on_cutoff(i + 1) and beat_ok(i + 2):
                i += 2
            else:
                break
        s_sec, e_sec = peaks[first], peaks[i + 1]
        if i - first + 2 >= 3 and e_sec - s_sec >= min_dur_sec:
            s_str, e_str = make_time_strs(s_sec, e_sec)
            segments.append({
                'start_sec': s_sec,
                'end_sec': e_sec,
                'label': label,
                'color': color,
                'start_time_str': s_str,
                'end_time_str': e_str,
                'detected_rhythm': label,
            })
        i += 1
    return segments


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
            # Carried through so apply_auto_segments_to_engine's same-label
            # merge (below) can tell two touching Artifact regions of a
            # DIFFERENT mains kind (50 Hz vs 60 Hz) apart -- see its own
            # comment. None on every other label; harmless there.
            'artifact_kind': seg.get('artifact_kind'),
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
    RATE_LABELS = {'sinus tachycardia', 'sinus bradycardia', 'atrial tachycardia'}
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
        'sinus arrhythmia',
        'artifact',
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
                # Two touching/close Artifact regions of a DIFFERENT mains
                # kind (50 Hz vs 60 Hz) are not the same event and must not
                # merge: confirmed directly this erased the one edge
                # between them (e.g. 0-16.08s 60 Hz, 16.08-35.57s 50/60
                # mix) that apply_artifact_wiggle_from_recording needs to
                # find the wiggle at -- merged into one 0-53.47s span, that
                # internal edge no longer exists to check at all, and the
                # "A" mark there never appears. Same-kind Artifact pairs
                # (the normal case) are unaffected. Every other label has
                # no artifact_kind (None == None), so this never changes
                # their own merge behaviour.
                if label == 'artifact' and last.get('artifact_kind') != ev.get('artifact_kind'):
                    pass
                elif start <= last_end + 0.25:
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
