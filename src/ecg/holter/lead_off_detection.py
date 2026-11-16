"""Lead-off spans of a recording, for the Full Disclosure "Lead off: ..." label.

How a lead-off is in the recording: the device reports each electrode's
contact bit (ecg/serial/packet_parser.py, la/ra/ll_present and each chest
lead's own bit), the parser turns a lead with a missing electrode into None,
and main.py writes Live12LeadPlotter.fill_lead_off(packet): a lead that is
off holds its last valid value. So in the .ecgh file a lead-off is a run of
exactly identical consecutive samples -- set by the electrode bit, not by
the signal being small.

A run counts as lead-off from LEAD_OFF_MIN_SEC on. Measured on all 166
recordings in src/recordings (the 14 h and 72 h ones left out), the
longest run of identical samples of each lead of each recording (1992):
1893 under 0.5 s, one 0.8 s, none 1-2 s, and 98 of 2 s or longer -- the
lead-off ones (e.g. V4 1490-1505 s and V5 1170-1185 s on
2026-11-13_12-30-24, all 12 leads 657-787 s on 2026-10-30_13-51-44).

All 12 leads off means no electrode is reporting -- nothing is known about
the heart then. This module only reports the spans; it does not decide
Asystole (that is detect_asystole_segments in
auto_segment_arrthymia_detection.py, a separate detector).

Independent of every arrhythmia detector: own cache file, own background
thread, read only by the Full Disclosure bottom bar.
"""
import copy
import json
import os
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

LEAD_OFF_MIN_SEC = 1.0
_CHUNK_SEC = 300
_CACHE_FILENAME = 'lead_off_cache.json'
# Bump when the scan changes, so old cache files are not reused.
_CACHE_VERSION = 1

_memory_cache: Dict[str, List[Dict[str, object]]] = {}
_async_jobs: Dict[str, object] = {}


def _cache_key(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|v{_CACHE_VERSION}|min{LEAD_OFF_MIN_SEC}"


def _cache_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), _CACHE_FILENAME)


def _scan(reader) -> List[Dict[str, object]]:
    """[{'lead', 'start_sec', 'end_sec'}] of every run of identical samples
    >= LEAD_OFF_MIN_SEC, per lead, sorted by start."""
    fs = float(getattr(reader, 'fs', 500) or 500)
    names = list(getattr(reader, 'lead_names', []) or [])
    total = float(getattr(reader, 'duration_sec', 0.0) or 0.0)
    if not names or total <= 0:
        return []
    min_len = int(round(LEAD_OFF_MIN_SEC * fs))
    # Per lead: value of the last sample read, and the sample index where
    # the current run of that value started (runs continue across chunks).
    last_val = [None] * len(names)
    run_start = [0] * len(names)
    spans = []
    pos = 0  # sample index of the chunk's first sample
    t = 0
    while t < total:
        data = reader.read_range(float(t), float(min(total, t + _CHUNK_SEC)))
        if data is None or data.shape[1] == 0:
            break
        n = data.shape[1]
        for i in range(min(len(names), data.shape[0])):
            x = np.asarray(data[i])
            # Sample indices (within the chunk) where the value changes from
            # the sample before; index 0 compares with the previous chunk.
            change = np.flatnonzero(np.diff(x) != 0) + 1
            if last_val[i] is not None and x[0] != last_val[i]:
                change = np.concatenate(([0], change))
            for c in change:
                start = run_start[i]
                if pos + c - start >= min_len:
                    spans.append({'lead': names[i], 'start_sec': float(start / fs), 'end_sec': float((pos + c) / fs)})
                run_start[i] = pos + int(c)
            last_val[i] = x[-1]
        pos += n
        t += _CHUNK_SEC
    for i in range(len(names)):
        if last_val[i] is not None and pos - run_start[i] >= min_len:
            spans.append({'lead': names[i], 'start_sec': float(run_start[i] / fs), 'end_sec': float(pos / fs)})
    spans.sort(key=lambda s: (s['start_sec'], names.index(s['lead'])))
    return spans


def get_cached_lead_off(path: Optional[str]) -> Optional[List[Dict[str, object]]]:
    key = _cache_key(path)
    if key is None:
        return None
    if key in _memory_cache:
        return copy.deepcopy(_memory_cache[key])
    try:
        with open(_cache_path(path), 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if saved.get('key') != key or not isinstance(saved.get('spans'), list):
        return None
    _memory_cache[key] = saved['spans']
    return copy.deepcopy(saved['spans'])


def detect_lead_off_spans(reader) -> List[Dict[str, object]]:
    """Cached _scan() of the recording the reader has open."""
    path = getattr(reader, 'path', None)
    cached = get_cached_lead_off(path)
    if cached is not None:
        return cached
    spans = _scan(reader)
    key = _cache_key(path)
    if key is not None:
        _memory_cache[key] = copy.deepcopy(spans)
        try:
            tmp = _cache_path(path) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'key': key, 'spans': spans}, f)
            os.replace(tmp, _cache_path(path))
        except (OSError, TypeError, ValueError) as e:
            print(f"[Lead Off] Could not save cache: {e}")
    return spans


def detect_lead_off_spans_async(path: str, on_done: Callable[[List[Dict[str, object]]], None]) -> None:
    """detect_lead_off_spans() on a worker thread; on_done(spans) is called
    on the Qt main thread. Must be called from the Qt main thread (same
    pattern as detect_asystole_segments_async)."""
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

    cached = get_cached_lead_off(path)
    if cached is not None:
        QTimer.singleShot(0, lambda: on_done(cached))
        return
    key = _cache_key(path) or os.path.abspath(path)
    job = _async_jobs.get(key)
    if job is not None:
        job.callbacks.append(on_done)
        return

    class _LeadOffJob(QObject):
        done = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.callbacks = [on_done]
            self.done.connect(self._deliver)

        @pyqtSlot(object)
        def _deliver(self, spans):
            _async_jobs.pop(key, None)
            for cb in self.callbacks:
                try:
                    cb(copy.deepcopy(spans))
                except RuntimeError:
                    pass  # the receiving widget was closed meanwhile
                except Exception as e:
                    print(f"[Lead Off] Callback error: {e}")

    job = _LeadOffJob()
    _async_jobs[key] = job

    def _run():
        try:
            from .file_format import ECGHFileReader
            reader = ECGHFileReader(path)
            try:
                spans = detect_lead_off_spans(reader)
            finally:
                close = getattr(reader, 'close', None)
                if close:
                    close()
        except Exception as e:
            print(f"[Lead Off] Background scan failed: {e}")
            spans = []
        job.done.emit(spans)

    threading.Thread(target=_run, name='lead-off-scan', daemon=True).start()


def leads_off_in(spans, start_sec: float, end_sec: float, lead_order) -> List[str]:
    """Lead names (in lead_order) off at any time in [start_sec, end_sec)."""
    off = {s['lead'] for s in spans or [] if s['start_sec'] < end_sec and s['end_sec'] > start_sec}
    return [n for n in lead_order if n in off]


def lead_off_text(leads: List[str], all_leads) -> str:
    """'' when no lead is off; 'Lead off: All leads' when every lead is;
    otherwise 'Lead off: V4, V5'."""
    if not leads:
        return ''
    if all_leads and len(leads) >= len(list(all_leads)):
        return 'Lead off: All leads'
    return 'Lead off: ' + ', '.join(leads)
