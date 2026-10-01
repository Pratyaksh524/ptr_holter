from __future__ import annotations

import gc
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, QThread, QTimer, Qt, QObject, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QAbstractItemView, QButtonGroup, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QScrollBar, QSlider, QSizePolicy, QSpinBox, QSplitter,
    QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QToolButton, QVBoxLayout,
    QWidget, QInputDialog
)

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    pg = None
    HAS_PG = False

from ..theme import (
    ADC_TO_MV, COL_BEAT_S, COL_BG, COL_BLACK, COL_BTN_ACTIVE_BG, COL_BTN_ACTIVE_TEXT, COL_DARK,
    COL_GRAY, COL_GREEN, COL_GREEN_DRK, COL_GREEN_MID, COL_GRID_MAJOR, COL_GRID_MINOR, COL_RED,
    COL_TEXT, COL_TIMESTAMP, COL_WAVE_ORANGE, COL_WAVE_RED, COL_WHITE, COL_YELLOW, GAINS,
    PAPER_SPEEDS, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_RULER, TOOL_SELECT,
)
from ..holter_helpers import (
    UI_ACCENT, UI_ACCENT_HOVER, UI_BG, UI_BORDER, UI_CARD, UI_MUTED, UI_PANEL, UI_PANEL_ALT,
    UI_SUCCESS, UI_TEXT, UI_WARNING, _class_matches_filter, _find_latest_completed_session,
    _format_system_time, _get_recording_start_end_times, _metrics_duration_sec,
    _normalize_beat_class, _normalize_patient_info, _resolve_recordings_dir, _sec_to_hms,
    _style_active_btn, _style_btn, _style_filter_btn, _table_style, _template_filter_key,
)

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas, HolterRRTrendCanvas, HRTrendCanvas

# 11b. HOLTER LORENZ PANEL
class HolterLorenzPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None, duration_sec: float = 86400, replay_engine=None):
        super().__init__(parent)
        self.duration_sec = duration_sec
        self.replay_engine = replay_engine
        self.setStyleSheet(f"background:{COL_BG};")
        self._metrics_list = []
        self._all_beat_data = []  # Store (time, rr, beat_class, metric_ref) for beat selection
        self._current_replay_data = None  # Store current 12-lead frame for beat selection display
        self._current_replay_start_sec = 0.0
        self._current_replay_beat_annotations = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top Section: Single RR Trend Canvas for Full Recording
        self.rr_trend_canvas = HolterRRTrendCanvas(self, title="RR Trend (Full Recording)")
        self.rr_trend_canvas.setFixedHeight(120)
        
        layout.addWidget(self.rr_trend_canvas)

        # Middle Section Splitter
        mid_splitter = QSplitter(Qt.Horizontal)
        
        # Left side of mid splitter: Lorenz Plot + Buttons
        lorenz_container = QWidget()
        l_layout = QVBoxLayout(lorenz_container)
        l_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lorenz_canvas = LorenzCanvas()
        self.lorenz_canvas.beats_selected.connect(self._on_beats_selected)
        self.lorenz_canvas.beats_reclassified.connect(self._on_beats_reclassified)
        self.lorenz_canvas.beats_deleted.connect(self._on_beats_deleted)
        self.lorenz_canvas.setFixedHeight(350)
        l_layout.addWidget(self.lorenz_canvas)

        # Filter buttons matching the Template tab
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_row.setContentsMargins(0, 4, 0, 4)
        self._filter_btns = {}
        self._current_filter = "all"
        for key, lbl in [
            ("all", "All"),
            ("N", "N"),
            ("S", "S"),
            ("V", "V"),
            ("P", "P"),
            ("AF", "AF"),
            ("X", "X"),
            ("Other", "Other"),
        ]:
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setChecked(key == "all")
            btn.setToolTip(f"Filter by {lbl}")
            btn.setFixedHeight(28)
            btn.setStyleSheet(_style_filter_btn(active=(key == "all"), fg=COL_GREEN if key != "all" else UI_TEXT))
            btn.clicked.connect(lambda checked=False, k=key: self._set_lorenz_filter(k))
            self._filter_btns[key] = btn
            btn_row.addWidget(btn)
        l_layout.addLayout(btn_row)

        # Internal state for view/display toggle helpers (used by right-click context menu)
        self._pixel_cycle = [2, 3, 5]
        self._pixel_idx   = 1

        mid_splitter.addWidget(lorenz_container)
        
        # Right side of mid splitter: 12-Lead ECG Grid (3 columns x 4 rows)
        leads_container = QWidget()
        leads_container.setStyleSheet(f"background:{COL_BLACK};border:1px solid {UI_BORDER};border-radius:6px;")
        leads_layout = QVBoxLayout(leads_container)
        leads_layout.setContentsMargins(4, 4, 4, 4)
        leads_layout.setSpacing(2)
        
        # Create 12-lead grid: 4 rows x 3 columns
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        grid_layout.setContentsMargins(6, 6, 6, 6)
        
        self._lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self._lead_strips = {}  # Dictionary to store ECG strip canvases by lead name
        
        for idx, lead_name in enumerate(self._lead_names):
            row = idx // 3  # 0-3 (4 rows)
            col = idx % 3   # 0-2 (3 columns)
            
            # Create a frame for each lead
            lead_frame = QFrame()
            lead_frame.setStyleSheet(f"background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;")
            lead_frame_layout = QVBoxLayout(lead_frame)
            lead_frame_layout.setContentsMargins(2, 2, 2, 2)
            lead_frame_layout.setSpacing(1)
            
            # Lead label
            lead_label = QLabel(lead_name)
            lead_label.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
            lead_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lead_frame_layout.addWidget(lead_label)
            
            # ECG strip canvas for this lead - with show_annotations=False to hide N labels and numbers
            strip = ECGStripCanvas(height=60, color=COL_GREEN, pen_width=0.8, lead_name=lead_name, show_vertical_lines=False, show_annotations=False)
            strip.set_gain(1.0)
            self._lead_strips[lead_name] = strip
            lead_frame_layout.addWidget(strip)
            
            grid_layout.addWidget(lead_frame, row, col)
        
        leads_layout.addLayout(grid_layout)
        mid_splitter.addWidget(leads_container)

        lorenz_container.setMaximumWidth(450)
        mid_splitter.setSizes([400, 1200])
        mid_splitter.setStretchFactor(0, 0)
        mid_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(mid_splitter, stretch=2)

        # Bottom Section: ECG Strip
        self.ecg_canvas = ECGStripCanvas()
        self.ecg_canvas.setFixedHeight(90)
        layout.addWidget(self.ecg_canvas)

    def update_from_metrics(self, metrics_list, duration_sec=None):
        self._metrics_list = list(metrics_list or [])
        if duration_sec:
            self.duration_sec = duration_sec
            
        pts = []
        rr_all = []
        rr_classes = []
        
        for m in self._metrics_list:
            t0 = float(m.get('t', 0.0) or 0.0)
            beat_labels = list(m.get('all_beats') or [])
            if 'rr_intervals_list' in m:
                rr_list = [float(v) for v in (m.get('rr_intervals_list') or []) if float(v) > 0]
                if rr_list:
                    dur = float(m.get('duration', 0.0) or 0.0)
                    step = (dur / max(1, len(rr_list))) if dur > 0 else 0.2
                    for i, rr in enumerate(rr_list):
                        beat_info = beat_labels[i] if i < len(beat_labels) and isinstance(beat_labels[i], dict) else {}
                        beat_class = _normalize_beat_class(
                            beat_info.get("label")
                            or beat_info.get("template_label")
                            or beat_info.get("auto_label")
                            or m.get("label")
                            or m.get("template_label")
                            or m.get("arrhythmia")
                            or (m.get("arrhythmias") or [None])[0]
                        )
                        t = t0 + i * step
                        pts.append((t, rr))
                        rr_all.append(rr)
                        rr_classes.append(beat_class)
                    continue
            rr_val = float(m.get('rr_ms', 0) or 0)
            if rr_val > 200:
                beat_class = _normalize_beat_class(
                    m.get("label")
                    or m.get("template_label")
                    or m.get("arrhythmia")
                    or (m.get("arrhythmias") or [None])[0]
                )
                pts.append((t0, rr_val))
                rr_all.append(rr_val)
                rr_classes.append(beat_class)
                
        start_epoch = None
        if self._metrics_list:
            start_epoch = self._metrics_list[0].get('timestamp')
            
        self.rr_trend_canvas.set_points(pts, start_epoch)
        self._pts = pts
        self._rr_all = rr_all
        self._rr_classes = rr_classes
        self._apply_lorenz_filter()

    def _set_lorenz_filter(self, key: str):
        self._current_filter = key
        for k, btn in self._filter_btns.items():
            is_active = (k == key)
            btn.setChecked(is_active)
            btn.setStyleSheet(_style_filter_btn(active=is_active, fg=COL_GREEN if not is_active else UI_TEXT))
        self._apply_lorenz_filter()

    def _apply_lorenz_filter(self):
        pts = getattr(self, "_pts", [])
        rr_all = getattr(self, "_rr_all", [])
        rr_classes = getattr(self, "_rr_classes", [])
        filt = getattr(self, "_current_filter", "all")

        if not rr_all:
            self.lorenz_canvas.set_data([], [])
            return

        if filt == "all":
            rr_n = [r for r, c in zip(rr_all, rr_classes) if r > 200]
            rr_n_classes = [c for r, c in zip(rr_all, rr_classes) if r > 200]
            rr_pts_full = [(t, r, c) for (t, r), c in zip(pts, rr_classes) if r > 200]
        else:
            filtered_indices = [i for i, (r, c) in enumerate(zip(rr_all, rr_classes)) if r > 200 and _class_matches_filter(c, filt)]
            rr_n = [rr_all[i] for i in filtered_indices]
            rr_n_classes = [rr_classes[i] for i in filtered_indices]
            rr_pts_full = [(pts[i][0], pts[i][1], rr_classes[i]) for i in filtered_indices if i < len(pts)]

        if len(rr_n) >= 2:
            rr_x = rr_n[:-1]
            rr_y = rr_n[1:]
            plot_beat_classes = rr_n_classes[:-1]

            lo = float(np.percentile(rr_n, 5))
            hi = float(np.percentile(rr_n, 95))
            if hi - lo < 250:
                center = float(np.median(rr_n))
                lo = center - 500.0
                hi = center + 500.0
            lo = max(0.0, lo - 50.0)
            hi = hi + 50.0
            self.lorenz_canvas.set_data(rr_x, rr_y, x_range=(lo, hi), y_range=(lo, hi),
                                        beat_classes=plot_beat_classes, rr_points=rr_pts_full)
        elif len(rr_n) == 1:
            self.lorenz_canvas.set_data([rr_n[0]], [rr_n[0]], x_range=(200, 1500), y_range=(200, 1500),
                                        beat_classes=rr_n_classes, rr_points=rr_pts_full)
        else:
            self.lorenz_canvas.set_data([], [])

    def set_replay_frame(self, data, beat_annotations=None, start_sec=0.0):
        """Store the current 12-lead frame data and display if beats are selected."""
        if data is None or data.shape[0] < 1:
            return
        
        # Store the current 12-lead data and timing info
        self._current_replay_data = data.copy() if hasattr(data, 'copy') else data
        self._current_replay_start_sec = start_sec
        self._current_replay_beat_annotations = beat_annotations
        
        # If beats were just selected, update the 12-lead strips with this data
        if getattr(self, '_beats_selected_for_display', False):
            self._beats_selected_for_display = False
            
            selected_beat_class = getattr(self, '_selected_beat_class', 'N')
            
            # Color mapping for arrhythmia types
            color_map = {
                "N":     "#00d215",  # Green
                "S":     "#ff3333",  # Red
                "V":     "#ff9900",  # Orange
                "P":     "#00e5ff",  # Cyan
                "AF":    "#ffd700",  # Gold
                "X":     "#888888",  # Gray
                "Other": "#00d215",  # Green
            }
            qrs_color = color_map.get(selected_beat_class, color_map["N"])
            
            print(f"DEBUG: Displaying 12-lead data for beat class: {selected_beat_class}, QRS color: {qrs_color}")
            
            if data.shape[0] >= 12:
                # Create time axis (assuming 500 Hz sampling rate)
                fs = 500  # Hz
                n_samples = data.shape[1]
                time_axis = np.linspace(0, n_samples / fs, n_samples)
                
                # Detect R-peaks in lead II and create beat annotations for highlighting
                r_peaks_ts = self._detect_qrs_peaks(data, fs=fs, start_sec=start_sec)
                
                # Create beat annotations for each detected R-peak with the anomaly color
                peak_annotations = []
                for peak_ts in r_peaks_ts:
                    peak_annotations.append({
                        'timestamp': peak_ts,
                        'label': selected_beat_class,
                        'color': qrs_color
                    })
                
                print(f"DEBUG: set_replay_frame - Found {len(peak_annotations)} R-peaks, Loading {len(self._lead_names)} leads")
                
                # Display data in each lead strip
                for idx, lead_name in enumerate(self._lead_names):
                    if idx < data.shape[0]:
                        strip = self._lead_strips[lead_name]
                        lead_signal = data[idx]
                        
                        # Set the data with beat annotations for QRS coloring
                        strip.set_data(time_axis, lead_signal, beat_annotations=peak_annotations, start_sec=start_sec)
                        
                        # Keep normal green color for the base waveform (QRS peaks will be colored by annotations)
                        strip.color = "#00d215"
                        strip.pen_width = 0.8
                        strip.update()
                
                print(f"DEBUG: set_replay_frame - Successfully updated all {len(self._lead_names)} lead strips with QRS highlighting")
            else:
                print(f"DEBUG: set_replay_frame - data.shape[0]={data.shape[0]} is less than 12 leads")
        
        # Always update the bottom ECG strip
        N = data.shape[1]
        x = np.linspace(0, N / 500.0, N) if N > 0 else []
        if N > 0:
            ch_idx = 1 if data.shape[0] > 1 else 0
            self.ecg_canvas.set_data(x, data[ch_idx].copy())
    
    def _detect_qrs_peaks(self, data, fs=500, start_sec=0.0):
        """Detect QRS/R-peaks in the ECG data using lead II and return timestamps."""
        try:
            from scipy.signal import find_peaks
            
            # Use lead II (index 1) for peak detection - it typically has the most prominent R-peak
            lead_idx = 1 if data.shape[0] > 1 else 0
            signal = np.asarray(data[lead_idx], dtype=float)
            
            # Normalize signal for better peak detection
            sig_mean = np.mean(signal)
            sig_std = np.std(signal)
            if sig_std > 0:
                normalized = (signal - sig_mean) / sig_std
            else:
                normalized = signal - sig_mean
            
            # Find peaks with:
            # - Minimum distance of 300ms (150 samples at 500Hz) between R-peaks
            # - Height threshold above 0.5 standard deviations
            # - Prominence to avoid false positives
            min_distance = int(0.3 * fs)  # 300ms minimum between R-peaks
            peaks, properties = find_peaks(
                normalized,
                distance=min_distance,
                height=0.5,
                prominence=0.3
            )
            
            # Convert peak indices to timestamps
            peak_times = []
            for peak_idx in peaks:
                ts = start_sec + (peak_idx / fs)
                peak_times.append(ts)
            
            print(f"DEBUG: _detect_qrs_peaks - Found {len(peak_times)} peaks in lead {lead_idx}")
            return peak_times
            
        except Exception as e:
            print(f"DEBUG: Error detecting QRS peaks: {e}")
            import traceback
            traceback.print_exc()
            return []

    #     View / display toggle helpers                                         

    def _set_display_mode(self, mode):
        self.lorenz_canvas._set_display(mode)
        self._btn_complete.setChecked(mode == "complete")
        self._btn_timeshare.setChecked(mode == "timesharing")
        # Update button visual state
        active = _style_active_btn() if hasattr(__builtins__, '__name__') else _style_btn()
        try:
            self._btn_complete.setStyleSheet(
                _style_active_btn() if mode == "complete" else _style_btn())
            self._btn_timeshare.setStyleSheet(
                _style_active_btn() if mode == "timesharing" else _style_btn())
        except Exception:
            pass

    def _set_view_mode(self, mode):
        self.lorenz_canvas._set_view(mode)
        self._btn_lorenz.setChecked(mode == "lorenz")
        self._btn_delta.setChecked(mode == "delta_rr")
        try:
            self._btn_lorenz.setStyleSheet(
                _style_active_btn() if mode == "lorenz" else _style_btn())
            self._btn_delta.setStyleSheet(
                _style_active_btn() if mode == "delta_rr" else _style_btn())
        except Exception:
            pass

    def _cycle_pixel_size(self):
        self._pixel_idx = (self._pixel_idx + 1) % len(self._pixel_cycle)
        sz = self._pixel_cycle[self._pixel_idx]
        self.lorenz_canvas._set_pixel(sz)

    #     Signal handlers from LorenzCanvas                                     

    def _on_beats_selected(self, indices):
        """Display 12-lead ECG for selected beats or clear if no beats selected."""
        # If no beats are selected (empty click), clear the waveforms
        if not indices:
            print("DEBUG: Empty space clicked - clearing waveforms")
            self._clear_12lead_displays()
            return
        
        if not self.lorenz_canvas._all_rr_points:
            return
        
        pts = self.lorenz_canvas._all_rr_points
        times = [pts[i][0] for i in indices if i < len(pts)]
        beat_classes = [pts[i][2] for i in indices if i < len(pts)]
        
        if not times:
            return
        
        # Store which beat class is selected for use when data arrives
        self._selected_beat_class = beat_classes[0] if beat_classes else "N"
        self._beats_selected_for_display = True
        
        # Use median time to display ECG window
        target_t = float(np.median(times))
        
        print(f"DEBUG: Beats selected at time {target_t}, beat class: {self._selected_beat_class}")
        
        # Emit seek signal to trigger data broadcast
        self.seek_requested.emit(target_t)
    
    def _clear_12lead_displays(self):
        """Clear all 12-lead waveform displays by showing flatlines instead of disappearing."""
        fs = 500  # Hz
        # Create a short flatline signal (5 seconds at baseline)
        window_sec = 5.0
        n_samples = int(fs * window_sec)
        time_axis = np.linspace(0, window_sec, n_samples)
        
        # Create flatline signal (constant baseline value)
        flatline = np.full(n_samples, 2048.0)  # Baseline value for most leads
        
        for lead_name in self._lead_names:
            strip = self._lead_strips[lead_name]
            
            # Special case for aVR which has different baseline
            if lead_name == "aVR":
                signal = np.full(n_samples, -2048.0)  # Baseline for aVR
            else:
                signal = flatline.copy()
            
            # Set the flatline data
            strip.set_data(time_axis, signal)
            
            # Keep normal green color
            strip.color = "#00d215"
            strip.pen_width = 0.8
            strip.update()

    def _on_beats_reclassified(self, indices, new_class):
        """Reclassification is already applied directly in LorenzCanvas._beat_classes;
        just trigger a replot to refresh colors."""
        self.lorenz_canvas.update()

    def _on_beats_deleted(self, indices):
        """Beats were removed from the canvas data; nothing to persist back to metrics
        in this lightweight implementation -- just refresh the display."""
        self.lorenz_canvas.update()



