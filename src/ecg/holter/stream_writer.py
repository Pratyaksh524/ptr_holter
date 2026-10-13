"""
ecg/holter/stream_writer.py
============================
HolterStreamWriter: central coordinator between the serial reader and
the Holter analysis pipeline.

Responsibilities:
  - Write every packet to disk via ECGHFileWriter (real-time, 500 Hz)
  - Maintain a 120-second circular RAM buffer for live display
  - Accumulate 30-second analysis chunks and enqueue to HolterAnalysisWorker
  - Track elapsed time, live BPM, detected arrhythmias

Integration (3 lines in twelve_lead_test.py):
    # In __init__:  self._holter = None
    # In packet loop:  if self._holter: self._holter.push(packet)
    # In Holter button:  self._holter = HolterStreamWriter(...)
"""

import os
import time
import queue
import threading
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict

from .file_format import ECGHFileWriter, LEAD_NAMES
from .session_store import ensure_events_db, write_session_metadata
from .sleep_guard import SleepGuard

# Analysis chunk cadence for live Holter updates.
CHUNK_SECONDS = 5
FS_DEFAULT = 500
LEADS = 12
DISPLAY_BUFFER_SECONDS = 120   # how much raw data to keep in RAM for display


class HolterStreamWriter:
    """
    Sits between the serial reader and the rest of the Holter system.
    Must be fast -- called 500x per second from the Qt main thread.
    """

    def __init__(self, output_dir: str, patient_info: dict,
                 fs: int = FS_DEFAULT,
                 on_chunk_ready=None,
                 on_arrhythmia=None,
                 chunk_seconds: int = CHUNK_SECONDS,
                 analysis_worker=None):
        """
        Args:
            output_dir: Directory to save .ecgh + .jsonl files
            patient_info: dict with name, dob, gender, doctor
            fs: sampling rate (500 Hz)
            on_chunk_ready: callback(chunk_data) when 30s chunk is ready
            on_arrhythmia: callback(label, timestamp) for live arrhythmia ticker
        """
        self.fs = fs
        self.patient_info = patient_info
        self.output_dir = output_dir
        self.on_chunk_ready = on_chunk_ready
        self.on_arrhythmia = on_arrhythmia
        self._analysis_worker = analysis_worker
        self.chunk_seconds = max(5, int(chunk_seconds))

        # State
        self._running = False
        self._start_time: Optional[float] = None
        self._total_frames = 0
        self._session_dir = ""
        self._ecgh_path = ""
        self._jsonl_path = ""
        self._session_json_path = ""
        self._events_db_path = ""

        # File writer (created on start)
        self._writer: Optional[ECGHFileWriter] = None

        # Display circular buffer: last 120s per lead
        self._display_buf_size = DISPLAY_BUFFER_SECONDS * fs
        self._display_buf = np.zeros((LEADS, self._display_buf_size), dtype=np.float32)
        self._display_ptr = 0    # next write position

        # 30-second analysis accumulator
        self._chunk_size = self.chunk_seconds * fs
        self._chunk_buf = np.zeros((LEADS, self._chunk_size), dtype=np.float32)
        self._chunk_ptr = 0

        # Analysis queue (consumed by HolterAnalysisWorker)
        self.analysis_queue: queue.Queue = queue.Queue(maxsize=10)

        # Live stats (updated by analysis worker via callbacks)
        self._live_bpm: float = 0.0
        self._live_arrhythmias: List[str] = []
        self._metrics_history: List[dict] = []
        self._summary_cache: Dict[str, object] = {
            'duration_sec': 0.0,
            'total_beats': 0,
            'avg_hr': 0.0,
            'max_hr': 0.0,
            'min_hr': 0.0,
            'sdnn': 0.0,
            'rmssd': 0.0,
            'pnn50': 0.0,
            'avg_quality': 1.0,
            'arrhythmia_counts': {},
            'longest_rr_ms': 0.0,
            'tachy_beats': 0,
            'brady_beats': 0,
            'pauses': 0,
            'avg_st_mv': 0.0,
            'avg_t_mv': 0.0,
            'patient_info': dict(patient_info or {}),
            'chunks_analyzed': 0,
            'beat_class_totals': {},
            've_beats': 0,
            'sve_beats': 0,
            'template_count': 0,
        }
        self._analysis_seq = 0
        self._lock = threading.Lock()

        # Flush timer for disk writes
        self._flush_thread: Optional[threading.Thread] = None

        # Blocks OS idle-sleep for as long as the recording runs: a sleeping
        # PC freezes the process, so frames stop while the wall clock runs on.
        self._sleep_guard = SleepGuard()

    #    Start / Stop                                                           

    def start(self) -> str:
        """Creates session directory + .ecgh file. Returns session directory path."""
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Replacing spaces is not enough: a name of '../../../Users/Public/x'
        # walks out of output_dir once os.path.join and normpath are done with
        # it. The shared sanitiser strips every character that could form a
        # path separator or a traversal sequence.
        try:
            from utils.input_validation import sanitize_filename_component
            patient_name = sanitize_filename_component(
                self.patient_info.get('name', 'Unknown'), fallback='Unknown')
        except Exception:
            import re as _re
            patient_name = _re.sub(
                r'[^A-Za-z0-9_\-]', '_',
                str(self.patient_info.get('name', 'Unknown'))) or 'Unknown'
        self._session_dir = os.path.join(self.output_dir, f"{ts}_{patient_name}")
        os.makedirs(self._session_dir, exist_ok=True)

        self._ecgh_path = os.path.join(self._session_dir, "recording.ecgh")
        self._jsonl_path = os.path.join(self._session_dir, "metrics.jsonl")
        self._session_json_path = os.path.join(self._session_dir, "session.json")
        self._events_db_path = ensure_events_db(self._session_dir)
        write_session_metadata(self._session_dir, {
            "session_dir": self._session_dir,
            "patient_info": dict(self.patient_info or {}),
            "fs": self.fs,
            "chunk_seconds": self.chunk_seconds,
            "output_dir": self.output_dir,
            "ecgh_path": self._ecgh_path,
            "jsonl_path": self._jsonl_path,
            "events_db_path": self._events_db_path,
            "status": "recording",
        })

        self._writer = ECGHFileWriter(
            path=self._ecgh_path,
            patient_info=self.patient_info,
            fs=self.fs
        )

        self._start_time = time.time()
        self._running = True
        self._sleep_guard.acquire()
        self._total_frames = 0
        self._chunk_ptr = 0
        self._display_ptr = 0

        # Reset buffers
        self._display_buf[:] = 0
        self._chunk_buf[:] = 0
        with self._lock:
            self._live_bpm = 0.0
            self._live_arrhythmias = []
            self._metrics_history = []
            self._analysis_seq = 0
            self._summary_cache = {
                'duration_sec': 0.0,
                'total_beats': 0,
                'avg_hr': 0.0,
                'max_hr': 0.0,
                'min_hr': 0.0,
                'sdnn': 0.0,
                'rmssd': 0.0,
                'pnn50': 0.0,
                'avg_quality': 1.0,
                'arrhythmia_counts': {},
                'longest_rr_ms': 0.0,
                'tachy_beats': 0,
                'brady_beats': 0,
                'pauses': 0,
                'avg_st_mv': 0.0,
                'avg_t_mv': 0.0,
                'patient_info': dict(self.patient_info or {}),
                'chunks_analyzed': 0,
                'beat_class_totals': {},
                've_beats': 0,
                'sve_beats': 0,
                'template_count': 0,
            }

        print(f"[Holter] Recording started -> {self._session_dir}")
        return self._session_dir

    def stop(self) -> dict:
        """Flush, finalize file, return session summary."""
        if not self._running:
            return {}

        self._running = False
        self._sleep_guard.release()

        # Flush partial chunk to analysis queue
        if self._chunk_ptr > 0:
            partial = self._chunk_buf[:, :self._chunk_ptr].copy()
            try:
                self.analysis_queue.put_nowait({
                    'data': partial,
                    'start_sec': self._total_frames / self.fs - self._chunk_ptr / self.fs,
                    'fs': self.fs,
                    'partial': True,
                    'jsonl_path': self._jsonl_path,
                })
            except queue.Full:
                pass

        # Finalize .ecgh file
        summary = {}
        if self._writer:
            summary = self._writer.finalize()
            summary['session_dir'] = self._session_dir
            summary['jsonl_path'] = self._jsonl_path
            summary['session_json_path'] = self._session_json_path
            summary['events_db_path'] = self._events_db_path
            try:
                write_session_metadata(self._session_dir, {
                    "session_dir": self._session_dir,
                    "patient_info": dict(self.patient_info or {}),
                    "fs": self.fs,
                    "chunk_seconds": self.chunk_seconds,
                    "output_dir": self.output_dir,
                    "ecgh_path": self._ecgh_path,
                    "jsonl_path": self._jsonl_path,
                    "events_db_path": self._events_db_path,
                    "status": "stopped",
                    "summary": dict(summary),
                })
            except Exception:
                pass
        with self._lock:
            summary.update(dict(self._summary_cache))
            summary['patient_info'] = dict(self.patient_info or {})
            summary['chunks_analyzed'] = self._summary_cache.get('chunks_analyzed', len(self._metrics_history))

        # Signal analysis worker to stop
        try:
            self.analysis_queue.put_nowait(None)   # sentinel
        except queue.Full:
            pass

        if self._analysis_worker is not None:
            try:
                self._analysis_worker.stop(wait=True, timeout=5.0)
            except Exception:
                pass

        print(f"[Holter] Recording stopped. Duration: {self.elapsed_seconds:.1f}s")
        return summary

    def attach_analysis_worker(self, analysis_worker) -> None:
        self._analysis_worker = analysis_worker

    def close(self, wait_for_analysis: bool = True) -> dict:
        if not wait_for_analysis:
            return self.stop()
        return self.stop()

    #    Main push method (called 500x per second)                               

    def push(self, packet: dict):
        """
        Fast path -- must return in <0.1ms.
        Called from Qt main thread on every serial packet.
        """
        if not self._running or self._writer is None:
            return

        # 1. Write to disk
        self._writer.write_packet(packet)

        # 2. Update display circular buffer
        dp = self._display_ptr % self._display_buf_size
        for i, lead in enumerate(LEAD_NAMES):
            self._display_buf[i, dp] = float(packet.get(lead, 2048))
        self._display_ptr += 1

        # 3. Accumulate analysis chunk
        cp = self._chunk_ptr
        for i, lead in enumerate(LEAD_NAMES):
            self._chunk_buf[i, cp] = float(packet.get(lead, 2048))
        self._chunk_ptr += 1
        self._total_frames += 1

        # 4. Chunk full -> enqueue for analysis
        if self._chunk_ptr >= self._chunk_size:
            chunk_data = self._chunk_buf.copy()
            chunk_start = (self._total_frames - self._chunk_size) / self.fs
            try:
                self.analysis_queue.put_nowait({
                    'data': chunk_data,
                    'start_sec': chunk_start,
                    'fs': self.fs,
                    'partial': False,
                    'jsonl_path': self._jsonl_path,
                })
            except queue.Full:
                pass   # analysis worker is slow, drop chunk (data still on disk)
            self._chunk_ptr = 0

    #    Live display data                                                       

    def get_display_data(self, lead_idx: int, n_samples: int) -> np.ndarray:
        """
        Returns last n_samples for the given lead from the circular display buffer.
        Safe to call from Qt main thread.
        """
        n = min(n_samples, self._display_buf_size)
        end = self._display_ptr % self._display_buf_size
        start = (end - n) % self._display_buf_size
        if start < end:
            return self._display_buf[lead_idx, start:end].copy()
        else:
            return np.concatenate([
                self._display_buf[lead_idx, start:],
                self._display_buf[lead_idx, :end]
            ])

    #    Live stats (updated by analysis worker)                                 

    def update_live_stats(self, bpm: float, arrhythmias: List[str]):
        with self._lock:
            self._live_bpm = bpm
            for a in arrhythmias:
                if a not in self._live_arrhythmias:
                    self._live_arrhythmias.insert(0, a)
            self._live_arrhythmias = self._live_arrhythmias[:10]
            if self.on_arrhythmia:
                for a in arrhythmias:
                    self.on_arrhythmia(a, time.time())

    def update_live_analysis(self, result: dict):
        """Persist the latest analysis result in memory for live UI panels."""
        bpm = float(result.get('hr_mean', 0.0) or 0.0)
        arrhythmias = list(result.get('arrhythmias', []) or [])
        with self._lock:
            self._live_bpm = bpm
            for a in arrhythmias:
                if a not in self._live_arrhythmias:
                    self._live_arrhythmias.insert(0, a)
            self._live_arrhythmias = self._live_arrhythmias[:10]
            self._metrics_history.append(dict(result))
            self._analysis_seq += 1
            self._summary_cache = self._build_summary_locked()
            seq = self._analysis_seq
        if self.on_chunk_ready:
            try:
                self.on_chunk_ready(dict(result), seq)
            except Exception:
                pass
        if self.on_arrhythmia:
            for a in arrhythmias:
                self.on_arrhythmia(a, time.time())

    def _build_summary_locked(self) -> Dict[str, object]:
        ml = self._metrics_history
        if not ml:
            return {
                'duration_sec': 0.0,
                'total_beats': 0,
                'avg_hr': 0.0,
                'max_hr': 0.0,
                'min_hr': 0.0,
                'sdnn': 0.0,
                'rmssd': 0.0,
                'pnn50': 0.0,
                'avg_quality': 1.0,
                'arrhythmia_counts': {},
                'longest_rr_ms': 0.0,
                'tachy_beats': 0,
                'brady_beats': 0,
                'pauses': 0,
                'avg_st_mv': 0.0,
                'patient_info': dict(self.patient_info or {}),
                'chunks_analyzed': 0,
                'beat_class_totals': {},
                've_beats': 0,
                'sve_beats': 0,
                'template_count': 0,
            }

        hr_vals = [m.get('hr_mean', 0) for m in ml if m.get('hr_mean', 0) > 0]
        beat_counts = [m.get('beat_count', 0) for m in ml]
        rr_stds = [m.get('rr_std', 0) for m in ml if m.get('rr_std', 0) > 0]
        rmssds = [m.get('rmssd', 0) for m in ml if m.get('rmssd', 0) > 0]
        pnn50s = [m.get('pnn50', 0) for m in ml if m.get('pnn50', 0) >= 0]
        qualities = [m.get('quality', 0) for m in ml if m.get('quality', 0) > 0]
        all_rr = [m.get('longest_rr', 0) for m in ml]
        st_vals = [m.get('st_mv', 0.0) for m in ml]
        t_vals = [m.get('t_mv', 0.0) for m in ml]
        duration_sec = float(sum(m.get('duration', 0.0) or 0.0 for m in ml))

        arrhy_counts: Dict[str, int] = {}
        beat_class_totals: Dict[str, int] = {}
        template_counts: List[int] = []
        for metric in ml:
            for label in metric.get('arrhythmias', []):
                arrhy_counts[label] = arrhy_counts.get(label, 0) + 1
            for cls, count in (metric.get('beat_class_counts', {}) or {}).items():
                beat_class_totals[cls] = beat_class_totals.get(cls, 0) + int(count or 0)
            template_counts.append(int(metric.get('template_count', 0) or 0))

        return {
            'duration_sec': duration_sec,
            'total_beats': sum(beat_counts),
            'avg_hr': float(np.mean(hr_vals)) if hr_vals else 0.0,
            'max_hr': float(np.max(hr_vals)) if hr_vals else 0.0,
            'min_hr': float(np.min(hr_vals)) if hr_vals else 0.0,
            'sdnn': float(np.mean(rr_stds)) if rr_stds else 0.0,
            'rmssd': float(np.mean(rmssds)) if rmssds else 0.0,
            'pnn50': float(np.mean(pnn50s)) if pnn50s else 0.0,
            'avg_quality': float(np.mean(qualities)) if qualities else 1.0,
            'arrhythmia_counts': arrhy_counts,
            'longest_rr_ms': max(all_rr) if all_rr else 0.0,
            'tachy_beats': sum(m.get('tachy_beats', 0) for m in ml),
            'brady_beats': sum(m.get('brady_beats', 0) for m in ml),
            'pauses': sum(m.get('pauses', 0) for m in ml),
            'avg_st_mv': float(np.mean(st_vals)) if st_vals else 0.0,
            'avg_t_mv': float(np.mean(t_vals)) if t_vals else 0.0,
            'patient_info': dict(self.patient_info or {}),
            'chunks_analyzed': len(ml),
            'beat_class_totals': beat_class_totals,
            've_beats': int(beat_class_totals.get('VE', 0)),
            'sve_beats': int(beat_class_totals.get('SVE', 0)),
            'template_count': max(template_counts) if template_counts else 0,
        }

    def get_live_stats(self) -> dict:
        with self._lock:
            return {
                'bpm': self._live_bpm,
                'arrhythmias': list(self._live_arrhythmias),
                'elapsed': self.elapsed_seconds,
                'frames': self._total_frames,
                'analysis_seq': self._analysis_seq,
            }

    def get_live_analysis_snapshot(self, since_seq: int = -1) -> Optional[dict]:
        with self._lock:
            if self._analysis_seq <= since_seq:
                return None
            return {
                'seq': self._analysis_seq,
                'metrics': [dict(m) for m in self._metrics_history],
                'summary': dict(self._summary_cache),
            }

    #    Properties                                                             

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def ecgh_path(self) -> str:
        return self._ecgh_path

    @property
    def jsonl_path(self) -> str:
        return self._jsonl_path

    @property
    def session_dir(self) -> str:
        return self._session_dir
