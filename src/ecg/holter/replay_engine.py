"""
ecg/holter/replay_engine.py
============================
Loads a saved .ecgh recording and feeds it to the existing
expanded_lead_view.py display pipeline -- no new display code needed.

Features:
  - Fast seek to any second via the index file
  - Load JSONL metrics and map timestamps to arrhythmia events
  - Provide navigation: Prev/Next event by type
"""

import os
import json
import time
import threading
import numpy as np
from typing import Optional, List, Dict, Tuple

from .file_format import ECGHFileReader, LEAD_NAMES
from .session_store import load_events, load_metrics, read_session_metadata
from .hrv_metrics import compute_hrv_summary
from .summary_utils import derive_hr_focus_summary


class HolterReplayEngine:
    """
    Controls playback of a saved .ecgh recording.
    Attach to expanded_lead_view by calling set_ecg_data_callback().
    """

    def __init__(self, ecgh_path: str, fs: int = 500):
        self.ecgh_path = ecgh_path
        self.fs = fs
        self._reader = ECGHFileReader(ecgh_path)
        self.duration_sec = self._reader.get_duration_seconds()
        self.patient_info = self._reader.patient_info
        self.lead_names = self._reader.lead_names

        # Playback state
        self._current_sec: float = 0.0
        self._playing = False
        self._playback_speed = 1.0
        self._playback_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._window_sec: float = 10.0

        # Metrics
        self._metrics: List[dict] = []
        self._arrhythmia_events: List[Tuple[float, str]] = []
        self._structured_events: List[dict] = []
        self._summary: Dict[str, object] = {}
        self._load_metrics(ecgh_path)
        self._load_layered_store()
        self._load_session_metadata()

        # Callbacks
        self._on_data: Optional[callable] = None        # (lead_idx, data_array)
        self._on_position: Optional[callable] = None    # (current_sec)
        self._on_arrhythmia_event: Optional[callable] = None  # (event)
        self._last_event_emit_sec: float = -1.0

    #    Metrics loading                                                        

    def _load_metrics(self, ecgh_path: str):
        jsonl_path = os.path.join(os.path.dirname(ecgh_path), 'metrics.jsonl')
        if not os.path.exists(jsonl_path):
            return
        try:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        m = json.loads(line)
                        self._metrics.append(m)
                        # Extract arrhythmia events with proper color coding for waveform display
                        for a in m.get('arrhythmias', []):
                            self._arrhythmia_events.append((m['t'], a))
                            
                            # Determine label code and color for waveform coloring
                            arr_label = str(a).lower()
                            label_code = 'N'
                            color = '#00FF00'  # Default green for NSR
                            
                            # Color mapping for arrhythmia types
                            if 'ventricular fibrillation' in arr_label or 'vfib' in arr_label or 'vf' in arr_label:
                                label_code = 'V'
                                color = '#FF3333'  # Red
                            elif 'ventricular tachycardia' in arr_label or 'vtach' in arr_label:
                                label_code = 'V'
                                color = '#FFA500'  # Orange
                            elif 'atrial fibrillation' in arr_label or 'afib' in arr_label:
                                label_code = 'AF'
                                color = '#9932CC'  # Purple
                            elif 'atrial flutter' in arr_label or 'aflutter' in arr_label:
                                label_code = 'AF'
                                color = '#FF69B4'  # Pink
                            elif 'sinus bradycardia' in arr_label or 'bradycardia' in arr_label:
                                label_code = 'S'
                                color = '#0000FF'  # Blue
                            elif 'sinus tachycardia' in arr_label or 'tachycardia' in arr_label:
                                label_code = 'S'
                                color = '#00BFFF'  # Deep Sky Blue
                            elif '1st-degree av block' in arr_label:
                                label_code = 'P'
                                color = '#FFFF00'  # Yellow
                            elif '2nd-degree av block' in arr_label:
                                label_code = 'P'
                                color = '#FFD700'  # Gold
                            elif '3rd-degree av block' in arr_label:
                                label_code = 'P'
                                color = '#FF8C00'  # Dark Orange
                            elif 'right bundle branch block' in arr_label:
                                label_code = 'P'
                                color = '#32CD32'  # Lime Green
                            elif 'left bundle branch block' in arr_label:
                                label_code = 'P'
                                color = '#9370DB'  # Medium Purple
                            elif 'premature ventricular contraction' in arr_label or 'pvc' in arr_label:
                                label_code = 'V'
                                color = '#8B4513'  # Brown
                            elif 'premature atrial contraction' in arr_label or 'pac' in arr_label:
                                label_code = 'S'
                                color = '#00FFFF'  # Cyan
                            
                            # Calculate end timestamp (use chunk duration)
                            chunk_start = float(m.get('t', 0.0) or 0.0)
                            chunk_duration = float(m.get('duration', 60.0) or 60.0)
                            
                            self._structured_events.append({
                                'timestamp': chunk_start,
                                'end_timestamp': chunk_start + chunk_duration,
                                'type': str(a),
                                'label': str(a),
                                'event_type': label_code,
                                'color': color,
                                'source': 'arrhythmia',
                            })
                        for ev in (m.get('classified_events', []) or []):
                            ts = float(ev.get('timestamp', m.get('t', 0.0)) or 0.0)
                            event_type = str(ev.get('template_label') or ev.get('label') or 'Event')
                            self._structured_events.append({
                                'timestamp': ts,
                                'type': event_type,
                                'label': str(ev.get('label') or event_type),
                                'template_label': event_type,
                                'source': 'classified',
                            })
        except Exception as e:
            print(f"[Replay] Could not load metrics: {e}")

        self._structured_events.sort(key=lambda item: float(item.get('timestamp', 0.0) or 0.0))

    def _load_layered_store(self):
        session_dir = os.path.dirname(self.ecgh_path)
        try:
            layered_metrics = load_metrics(session_dir)
            if layered_metrics:
                self._metrics = layered_metrics
        except Exception as e:
            print(f"[Replay] Could not load layered metrics: {e}")
        try:
            layered_events = load_events(session_dir)
            if layered_events:
                self._structured_events = []
                self._arrhythmia_events = []
                sorted_items = sorted(
                    layered_events,
                    key=lambda it: float(it.get("timestamp", it.get("t", 0.0)) or 0.0),
                )
                for idx, item in enumerate(sorted_items):
                    ts = float(item.get("timestamp", item.get("t", 0.0)) or 0.0)
                    label = str(item.get("label", item.get("event_type", "Event")))
                    label_lower = label.lower()
                    
                    # Determine label code and color for waveform coloring
                    label_code = 'N'
                    color = '#00FF00'  # Default green for NSR
                    
                    # Color mapping for arrhythmia types
                    if 'ventricular fibrillation' in label_lower or 'vfib' in label_lower or 'vf' in label_lower:
                        label_code = 'V'
                        color = '#FF3333'  # Red
                    elif 'ventricular tachycardia' in label_lower or 'vtach' in label_lower:
                        label_code = 'V'
                        color = '#FFA500'  # Orange
                    elif 'atrial fibrillation' in label_lower or 'afib' in label_lower:
                        label_code = 'AF'
                        color = '#9932CC'  # Purple
                    elif 'atrial flutter' in label_lower or 'aflutter' in label_lower:
                        label_code = 'AF'
                        color = '#FF69B4'  # Pink
                    elif 'sinus bradycardia' in label_lower or 'bradycardia' in label_lower:
                        label_code = 'S'
                        color = '#0000FF'  # Blue
                    elif 'sinus tachycardia' in label_lower or 'tachycardia' in label_lower:
                        label_code = 'S'
                        color = '#00BFFF'  # Deep Sky Blue
                    elif '1st-degree av block' in label_lower:
                        label_code = 'P'
                        color = '#FFFF00'  # Yellow
                    elif '2nd-degree av block' in label_lower:
                        label_code = 'P'
                        color = '#FFD700'  # Gold
                    elif '3rd-degree av block' in label_lower:
                        label_code = 'P'
                        color = '#FF8C00'  # Dark Orange
                    elif 'right bundle branch block' in label_lower:
                        label_code = 'P'
                        color = '#32CD32'  # Lime Green
                    elif 'left bundle branch block' in label_lower:
                        label_code = 'P'
                        color = '#9370DB'  # Medium Purple
                    elif 'premature ventricular contraction' in label_lower or 'pvc' in label_lower:
                        label_code = 'V'
                        color = '#8B4513'  # Brown
                    elif 'premature atrial contraction' in label_lower or 'pac' in label_lower:
                        label_code = 'S'
                        color = '#00FFFF'  # Cyan
                    
                    # Check if item has end_timestamp, otherwise calculate from
                    # duration -- or, if neither is stored, cap the fallback
                    # span at the next *chunk-level* snapshot's own start
                    # time. These rows can be discrete, closely-spaced
                    # real-time classifier snapshots (e.g. one label every
                    # ~5s) with no duration field persisted; a flat 60s guess
                    # made an old snapshot's rendered region balloon far past
                    # where a newer, later classification already superseded
                    # it, which then visually collided with a genuinely
                    # separate, later arrhythmia episode on the Full
                    # Disclosure timeline.
                    #
                    # The same real-time pipeline also logs finer-grained,
                    # beat-level annotations (e.g. "PVC Candidate") at the
                    # precise instant that beat occurred, which can be
                    # anywhere within a chunk-level label's own ~5s window --
                    # not necessarily close in time to the chunk label itself
                    # (seen up to ~3s away in practice, so a short minimum-gap
                    # heuristic isn't reliable). Sorted by timestamp, that
                    # beat-level row would otherwise get treated as "the next
                    # chunk", capping the rhythm label's span early and
                    # leaving the rest of that chunk unrendered. Chunk-level
                    # rhythm labels are logged on whole-second boundaries
                    # (105.00, 110.00, ...); beat-level annotations are logged
                    # at the beat's precise, generally fractional timestamp.
                    # So the "next" row used for capping is the next one that
                    # itself looks like a whole-second chunk boundary, not
                    # simply the next row in sorted order.
                    end_ts = item.get("end_timestamp")
                    if end_ts is None:
                        duration = item.get("duration")
                        if duration is None:
                            next_ts = None
                            for later in sorted_items[idx + 1:]:
                                later_ts = float(later.get("timestamp", later.get("t", 0.0)) or 0.0)
                                if later_ts <= ts:
                                    continue
                                if abs(later_ts - round(later_ts)) < 0.01:
                                    next_ts = later_ts
                                    break
                            end_ts = next_ts if next_ts is not None and next_ts > ts else ts + 5.0
                        else:
                            end_ts = ts + float(duration or 60.0)
                    
                    event_dict = {
                        "timestamp": ts,
                        "end_timestamp": float(end_ts),
                        "type": str(item.get("event_type", label)),
                        "label": label,
                        "event_type": label_code,
                        "color": color,
                        "template_label": str(item.get("template_label", item.get("event_type", label))),
                        "source": str(item.get("source", "analysis")),
                        "confidence": float(item.get("confidence", 0.0) or 0.0),
                    }
                    self._structured_events.append(event_dict)
                    
                    if "arrhythmia" in str(item.get("source", "")).lower() or "analysis" in str(item.get("source", "")).lower():
                        self._arrhythmia_events.append((ts, label))
                self._structured_events.sort(key=lambda item: float(item.get('timestamp', 0.0) or 0.0))
        except Exception as e:
            print(f"[Replay] Could not load layered events: {e}")

    def _load_session_metadata(self):
        session_dir = os.path.dirname(self.ecgh_path)
        try:
            metadata = read_session_metadata(session_dir)
            if metadata:
                if isinstance(metadata.get("patient_info"), dict) and metadata["patient_info"]:
                    self.patient_info = metadata["patient_info"]
                if isinstance(metadata.get("summary"), dict) and metadata["summary"]:
                    self._summary = dict(metadata["summary"])
        except Exception as e:
            print(f"[Replay] Could not load session metadata: {e}")
    
    def get_events_with_manual_priority(self, start_sec=None, end_sec=None):
        """
        Get all events with MANUAL PRIORITY over AUTO-DETECTION.
        
        When both manual and auto marks exist:
        - Manual marks are always shown
        - Auto marks that overlap with manual marks (within 150ms) are suppressed
        - This ensures manual marks take precedence in reports
        
        Args:
            start_sec: Start time (None = from beginning)
            end_sec: End time (None = to end)
            
        Returns:
            List[dict]: Combined events with manual marks having priority
        """
        if start_sec is None:
            start_sec = 0.0
        if end_sec is None:
            end_sec = self.duration_sec
            
        SUPPRESS_TOLERANCE_SEC = 0.15
        
        # 1. Load manual beats from disk if they exist
        manual_events = []
        manual_timestamps = set()
        session_dir = os.path.dirname(self.ecgh_path)
        manual_beats_path = os.path.join(session_dir, 'manual_beats.json')
        
        try:
            if os.path.exists(manual_beats_path):
                with open(manual_beats_path, 'r') as f:
                    manual_beats = json.load(f)
                    for beat in manual_beats:
                        ts = float(beat.get('timestamp', 0.0))
                        lbl = str(beat.get('label', 'N'))
                        is_manual = beat.get('is_manual', False)
                        if lbl != 'N' and start_sec <= ts <= end_sec and is_manual:
                            manual_events.append({
                                'timestamp': ts,
                                'label': lbl,
                                'source': 'Manual',
                                'is_manual': True,
                                'color': beat.get('color', '#FFFF00')
                            })
                            manual_timestamps.add(round(ts, 2))
        except Exception as e:
            print(f"[ReplayEngine] Could not load manual beats: {e}")
        
        # 2. Load manual segments from disk if they exist
        manual_segments = []
        manual_segments_path = os.path.join(session_dir, 'manual_segments.json')
        try:
            if os.path.exists(manual_segments_path):
                with open(manual_segments_path, 'r') as f:
                    segments = json.load(f)
                    for seg in segments:
                        start = float(seg.get('start_sec', 0.0))
                        end = float(seg.get('end_sec', 0.0))
                        # Check if segment overlaps with the requested window
                        if start <= end_sec and end >= start_sec:
                            manual_segments.append({
                                'timestamp': start,
                                'end_timestamp': end,
                                'label': str(seg.get('label', 'Event')),
                                'source': 'Manual',
                                'is_manual': True,
                                'color': seg.get('color', '#FFFF00'),
                                'segment': True  # Mark as segment type
                            })
                            # Add segment start/end to manual timestamps for suppression
                            for ts in np.linspace(start, end, max(2, int((end - start) / 0.5))):
                                manual_timestamps.add(round(float(ts), 2))
        except Exception as e:
            print(f"[ReplayEngine] Could not load manual segments: {e}")
        
        # 3. Collect auto-detected events, suppressing those that overlap with manual marks
        auto_events = []
        for ev in self._structured_events:
            ts = float(ev.get('timestamp', 0.0) or 0.0)
            
            # Only include events in the requested window
            if not (start_sec <= ts <= end_sec):
                continue
            
            # Check if this auto event falls within tolerance of any manual mark
            is_suppressed = False
            for manual_ts in manual_timestamps:
                if abs(ts - manual_ts) < SUPPRESS_TOLERANCE_SEC:
                    is_suppressed = True
                    break
            
            if not is_suppressed:
                auto_events.append({
                    'timestamp': ts,
                    'label': ev.get('label', 'Event'),
                    'source': 'Auto',
                    'is_manual': False,
                    'type': ev.get('type', 'Event')
                })
        
        # Combine all events and return sorted by timestamp
        all_events = sorted(manual_events + manual_segments + auto_events, key=lambda x: x['timestamp'])
        return all_events

    #    Callbacks                                                              

    def set_data_callback(self, callback):
        """callback(lead_data_array_12xN) called every display frame."""
        self._on_data = callback

    def set_position_callback(self, callback):
        """callback(current_sec, duration_sec)"""
        self._on_position = callback

    def set_arrhythmia_callback(self, callback):
        """callback(timestamp, label)"""
        self._on_arrhythmia_event = callback

    def set_window_length(self, window_sec: float):
        """Set the display window length in seconds for emitted frames."""
        self._window_sec = max(0.5, float(window_sec))

    def get_window_length(self) -> float:
        return float(self._window_sec)

    def current_position(self) -> float:
        return float(self._current_sec)

    def _window_bounds(self, window_sec: float) -> Tuple[float, float]:
        """Return a bounded window, keeping the requested span when possible."""
        window_sec = max(0.5, float(window_sec))
        if self.duration_sec <= window_sec:
            return 0.0, float(self.duration_sec)

        start = max(0.0, min(self._current_sec, self.duration_sec - window_sec))
        end = min(self.duration_sec, start + window_sec)
        return start, end

    #    Seek & navigation                                                       
    def seek(self, target_sec: float):
        """Jump to any timestamp in the recording."""
        self._current_sec = max(0.0, min(target_sec, self.duration_sec))
        self._emit_frame()

    def seek_to_event(self, event_type: str, direction: str = 'next') -> float:
        """
        Jump to next/prev arrhythmia event of a given type.
        event_type: 'AF', 'VT', 'Brady', 'Tachy', etc. (substring match)
        direction: 'next' or 'prev'
        Returns the timestamp jumped to, or current if not found.
        """
        needle = str(event_type or "").strip().lower()
        events = [
            (float(item.get('timestamp', 0.0) or 0.0), str(item.get('label', '')))
            for item in self._structured_events
            if needle and (
                needle in str(item.get('type', '')).lower()
                or needle in str(item.get('label', '')).lower()
                or needle in str(item.get('template_label', '')).lower()
            )
        ]
        if not events:
            return self._current_sec

        if direction == 'next':
            candidates = [(t, a) for t, a in events if t > self._current_sec + 1]
            if candidates:
                target = candidates[0][0]
                self.seek(target)
                return target
        else:
            candidates = [(t, a) for t, a in events if t < self._current_sec - 1]
            if candidates:
                target = candidates[-1][0]
                self.seek(target)
                return target

        return self._current_sec

    def get_events_list(self) -> List[dict]:
        """Returns all detected arrhythmia events for the event navigator."""
        result = []
        for item in self._structured_events:
            ts = float(item.get('timestamp', 0.0) or 0.0)
            result.append({
                'timestamp': ts,
                'label': str(item.get('label', item.get('type', 'Event'))),
                'type': str(item.get('type', item.get('label', 'Event'))),
                'template_label': str(item.get('template_label', item.get('type', 'Event'))),
                'time_str': self._sec_to_hms(ts),
                'source': item.get('source', ''),
            })
        return result

    #    Data retrieval                                                          

    def get_lead_data(self, lead_idx: int, window_sec: float = 10.0) -> np.ndarray:
        """
        Returns window_sec of data for the given lead starting at current position.
        Returns shape (N,) float32 array.
        """
        start, end = self._window_bounds(window_sec)
        data = self._reader.read_range(start, end)
        if data.shape[0] > lead_idx:
            return data[lead_idx]
        return np.zeros(int(window_sec * self.fs), dtype=np.float32)

    def get_all_leads_data(self, window_sec: float = 10.0) -> np.ndarray:
        """Returns (12, N) array for current window."""
        start, end = self._window_bounds(window_sec)
        return self._reader.read_range(start, end)

    def get_beat_annotations(self, window_sec: float = 10.0) -> List[dict]:
        """
        Returns beat annotations (R-peaks with labels) for the current window.
        Returns list of dicts with 'timestamp' and 'label' keys.
        """
        start, end = self._window_bounds(window_sec)
        beats = []
        
        # Gather all beats from metrics that fall within the window
        for m in self._metrics:
            all_beats = m.get('all_beats', [])
            if not all_beats:
                continue
            
            for beat in all_beats:
                ts = float(beat.get('timestamp', 0.0))
                if start <= ts <= end:
                    beats.append({
                        'timestamp': ts,
                        'label': str(beat.get('label', 'N'))
                    })
        
        # Sort by timestamp
        beats.sort(key=lambda b: b['timestamp'])
        return beats

    def get_metrics_at(self, target_sec: float) -> Optional[dict]:
        """Returns the metrics chunk closest to target_sec."""
        if not self._metrics:
            return None
        closest = min(self._metrics, key=lambda m: abs(m['t'] - target_sec))
        return closest

    def _emit_frame(self):
        """Emit current frame data to registered callbacks."""
        if self._on_data:
            data = self.get_all_leads_data(window_sec=self._window_sec)
            self._on_data(data)
        if self._on_position:
            self._on_position(self._current_sec, self.duration_sec)
        if self._on_arrhythmia_event and self._arrhythmia_events:
            for t, label in self._arrhythmia_events:
                if self._last_event_emit_sec < t <= self._current_sec:
                    self._on_arrhythmia_event(t, label)
            self._last_event_emit_sec = self._current_sec

    #    Playback loop                                                          

    def play(self):
        """Start replay loop."""
        if self._playing:
            return
        self._playing = True
        self._stop_event.clear()
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True, name="HolterReplayLoop")
        self._playback_thread.start()

    def pause(self):
        """Pause replay without resetting position."""
        self._playing = False
        self._stop_event.set()
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=0.5)
        self._playback_thread = None

    def set_speed(self, speed: float):
        self._playback_speed = max(0.25, min(float(speed), 8.0))

    def is_playing(self) -> bool:
        return self._playing

    def _playback_loop(self):
        # Use a UI-friendly refresh cadence for replay frames.
        target_fps = 25.0
        dt_wall = 1.0 / target_fps
        while self._playing and not self._stop_event.is_set():
            self._current_sec += dt_wall * self._playback_speed
            if self._current_sec >= self.duration_sec:
                self._current_sec = self.duration_sec
                self._emit_frame()
                self._playing = False
                break
            self._emit_frame()
            time.sleep(dt_wall)

    #    Summary statistics                                                      

    def get_summary(self) -> dict:
        """Compute overall recording summary from all JSONL chunks."""
        if not self._metrics:
            return {}

        hr_values = [m['hr_mean'] for m in self._metrics if m.get('hr_mean', 0) > 0]
        beat_counts = [m.get('beat_count', 0) for m in self._metrics]
        qualities = [m['quality'] for m in self._metrics if m.get('quality', 0) > 0]

        # Properly aggregate ALL RR intervals for true global HRV metrics
        all_rr_intervals = []
        for m in self._metrics:
            if 'rr_intervals_list' in m and m['rr_intervals_list']:
                all_rr_intervals.extend(m['rr_intervals_list'])
        
        all_rr = np.array(all_rr_intervals, dtype=float)
        hrv = compute_hrv_summary(all_rr)
        global_sdnn = hrv.get("sdnn", 0.0)
        global_rmssd = hrv.get("rmssd", 0.0)
        global_pnn50 = hrv.get("pnn50", 0.0)

        # Arrhythmia counts
        arrhy_counts: Dict[str, int] = {}
        beat_class_totals: Dict[str, int] = {}
        template_counts: List[int] = []
        for m in self._metrics:
            for a in m.get('arrhythmias', []):
                arrhy_counts[a] = arrhy_counts.get(a, 0) + 1
            for cls, count in (m.get('beat_class_counts', {}) or {}).items():
                beat_class_totals[cls] = beat_class_totals.get(cls, 0) + int(count or 0)
            template_counts.append(int(m.get('template_count', 0) or 0))

        # ST and T per-lead averages
        st_vals = [m.get('st_mv', 0) for m in self._metrics]
        t_vals = [m.get('t_mv', 0) for m in self._metrics]

        # HR per hour
        hourly_hr: Dict[int, List[float]] = {}
        for m in self._metrics:
            hour = int(m['t'] // 3600)
            if m.get('hr_mean', 0) > 0:
                hourly_hr.setdefault(hour, []).append(m['hr_mean'])
        hourly_avg = {h: round(np.mean(vals), 1) for h, vals in hourly_hr.items()}

        # Longest RR interval
        all_rr = [m.get('longest_rr', 0) for m in self._metrics]
        longest_rr = max(all_rr) if all_rr else 0

        total_beats = sum(beat_counts)
        total_tachy = sum(m.get('tachy_beats', 0) for m in self._metrics)
        total_brady = sum(m.get('brady_beats', 0) for m in self._metrics)
        total_pauses = sum(m.get('pauses', 0) for m in self._metrics)
        focus = derive_hr_focus_summary(self._metrics)

        return {
            'duration_sec': self.duration_sec,
            'total_beats': total_beats,
            'avg_hr': round(float(np.mean(hr_values)), 1) if hr_values else 0.0,
            'max_hr': round(float(np.max(hr_values)), 1) if hr_values else 0.0,
            'min_hr': round(float(np.min(hr_values)), 1) if hr_values else 0.0,
            'max_hr_time': focus.get('max_hr_time', ''),
            'max_hr_timestamp': focus.get('max_hr_timestamp', 0.0),
            'min_hr_time': focus.get('min_hr_time', ''),
            'min_hr_timestamp': focus.get('min_hr_timestamp', 0.0),
            'sinus_max_hr': focus.get('sinus_max_hr', round(float(np.max(hr_values)), 1) if hr_values else 0.0),
            'sinus_min_hr': focus.get('sinus_min_hr', round(float(np.min(hr_values)), 1) if hr_values else 0.0),
            'sinus_max_hr_time': focus.get('sinus_max_hr_time', ''),
            'sinus_max_hr_timestamp': focus.get('sinus_max_hr_timestamp', 0.0),
            'sinus_min_hr_time': focus.get('sinus_min_hr_time', ''),
            'sinus_min_hr_timestamp': focus.get('sinus_min_hr_timestamp', 0.0),
            'sdnn': global_sdnn,
            'rmssd': global_rmssd,
            'pnn50': global_pnn50,
            'triidx': hrv.get("triangular_index", 0.0),
            'vlf_power': hrv.get("vlf", 0.0),
            'lf_power': hrv.get("lf", 0.0),
            'hf_power': hrv.get("hf", 0.0),
            'lf_hf_ratio': hrv.get("lf_hf_ratio", 0.0),
            'total_power': hrv.get("total_power", 0.0),
            'avg_quality': round(float(np.mean(qualities)), 3) if qualities else 1.0,
            'arrhythmia_counts': arrhy_counts,
            'hourly_hr': hourly_avg,
            'longest_rr_ms': longest_rr,
            'tachy_beats': total_tachy,
            'brady_beats': total_brady,
            'pauses': total_pauses,
            'avg_st_mv': round(float(np.mean(st_vals)), 4) if st_vals else 0.0,
            'avg_t_mv': round(float(np.mean(t_vals)), 4) if t_vals else 0.0,
            'patient_info': self.patient_info,
            'chunks_analyzed': len(self._metrics),
            'beat_class_totals': beat_class_totals,
            've_beats': int(beat_class_totals.get('VE', 0)),
            'sve_beats': int(beat_class_totals.get('SVE', 0)),
            'template_count': max(template_counts) if template_counts else 0,
        }

    #    Helpers                                                                 

    @staticmethod
    def _sec_to_hms(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def close(self):
        self.pause()
        self._reader.close()
