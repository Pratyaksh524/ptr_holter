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
    _style_active_btn, _style_btn, _table_style, _template_filter_key,
)

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas

# 11. HOLTER HISTOGRAM PANEL
class HolterHistogramPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._metrics = []
        self._rank_mode = "rri"
        self._selected_point = None
        self._build_ui()

    def _find_template_host(self):
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, "_show_template_card_menu"):
                return parent
            parent = parent.parentWidget()
        window = self.window()
        if window is not None and hasattr(window, "_show_template_card_menu"):
            return window
        return None

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        self._title_label = QLabel("Histogram - RR Interval Distribution")
        self._title_label.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;border:none;")
        title_row.addWidget(self._title_label, 1)
        self._type_combo = QComboBox()
        self._type_combo.addItems(["RR Interval", "Heart Rate", "RRI Ratio"])
        self._type_combo.setStyleSheet(f"""
            QComboBox {{
                background:{COL_DARK}; color:{COL_GREEN};
                border:1px solid {COL_GREEN_DRK}; padding:4px; border-radius:4px;
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        title_row.addWidget(self._type_combo)
        layout.addLayout(title_row)

        btn_row = QHBoxLayout()
        self._rank_buttons = {}
        for lbl, mode in [("RRI Ranking", "rri"), ("Time Ranking", "time"),
                          ("Prematurity Ranking", "prematurity"), ("Similarity Ranking", "similarity")]:
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setStyleSheet(_style_btn())
            btn.clicked.connect(lambda _, m=mode: self._set_rank_mode(m))
            self._rank_buttons[mode] = btn
            btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._hist_canvas = HistogramCanvas()
        self._hist_canvas.bar_clicked.connect(self._on_bar_clicked)
        layout.addWidget(self._hist_canvas, 1)
        self._selected_info = QLabel("Selected: none")
        self._selected_info.setStyleSheet(
            f"color:{COL_WHITE};font-size:11px;font-weight:bold;border:none;padding:2px 0;"
        )
        layout.addWidget(self._selected_info)


        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(10, 6, 10, 6)
        self._hist_stats = {}
        for i, (key, lbl) in enumerate([("nns","NNs"),("mean_nn","Mean NN"),
                                         ("sdnn","SDNN"),("sdann","SDANN"),
                                         ("rmssd","rMSSD"),("pnn50","pNN50"),
                                         ("triidx","TRIIDX"),("sdnnidx","SDNNIDX")]):
            col = i % 4
            row = i // 4
            l = QLabel(f"{lbl}:")
            l.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            v = QLabel("?")
            v.setStyleSheet(f"color:{COL_WHITE};font-size:13px;font-weight:bold;border:none;")
            stats_layout.addWidget(l, row*2, col)
            stats_layout.addWidget(v, row*2+1, col)
            self._hist_stats[key] = v
        layout.addWidget(stats_frame)

        self._strip = ECGStripCanvas(height=60)
        layout.addWidget(self._strip)
        self._set_rank_mode("rri")

    def _set_rank_mode(self, mode: str):
        self._rank_mode = mode
        for key, btn in getattr(self, "_rank_buttons", {}).items():
            active = key == mode
            btn.setChecked(active)
            btn.setStyleSheet(_style_active_btn() if active else _style_btn())
        self._draw()
    
    def _on_type_changed(self, text):
        """Update title label when histogram type changes."""
        if text == "RR Interval":
            self._title_label.setText("Histogram - RR Interval Distribution")
        elif text == "Heart Rate":
            self._title_label.setText("Histogram - Heart Rate Distribution")
        elif text == "RRI Ratio":
            self._title_label.setText("Histogram - RRI Ratio Distribution")
        self._draw()

    def update_from_metrics(self, metrics_list: list):
        self._metrics = list(metrics_list or [])
        self._draw()

    def _build_points(self):
        points = []
        for metric_idx, m in enumerate(self._metrics):
            base_t = float(m.get('t', 0.0) or 0.0)
            label = str((m.get('arrhythmias') or [m.get('label', '')])[0] or '')
            rr_list = [float(v) for v in (m.get('rr_intervals_list') or []) if float(v) > 0]
            if rr_list:
                dur = float(m.get('duration', 0.0) or 0.0)
                step = (dur / max(1, len(rr_list))) if dur > 0 else 0.2
                for sample_idx, rr in enumerate(rr_list):
                    points.append({
                        't': base_t + sample_idx * step,
                        'rr': rr,
                        'metric_idx': metric_idx,
                        'sample_idx': sample_idx,
                        'label': label,
                    })
                continue
            rr_val = float(m.get('rr_ms', 0) or 0)
            if rr_val > 200:
                points.append({
                    't': base_t,
                    'rr': rr_val,
                    'metric_idx': metric_idx,
                    'sample_idx': 0,
                    'label': label,
                })
        return points

    def _draw(self):
        points = self._build_points()
        rr_vals = [p['rr'] for p in points if p['rr'] > 200]
        if not rr_vals:
            self._hist_canvas.set_histogram_data([], mode=self._rank_mode)
            for key in self._hist_stats:
                self._hist_stats[key].setText('?')
            return

        rr_arr = np.array(rr_vals, dtype=float)
        median_rr = float(np.median(rr_arr)) if rr_arr.size else 0.0
        mean_rr = float(np.mean(rr_arr)) if rr_arr.size else 0.0
        
        # Get selected histogram type from dropdown
        hist_type = self._type_combo.currentText()
        
        # Build the value that is actually plotted.  Ranking must change the
        # histogram variable itself; sorting the points alone has no effect
        # on np.histogram().
        axis_label = hist_type
        if hist_type == "Heart Rate":
            # Convert RR intervals (ms) to Heart Rate (bpm): HR = 60000 / RR_ms
            for p in points:
                if p['rr'] > 0:
                    p['hr'] = 60000.0 / p['rr']
                else:
                    p['hr'] = 0.0
            for p in points:
                p['hist_value'] = p['hr']
            ranked = points
        elif hist_type == "RRI Ratio":
            # RRI Ratio = current RR / previous RR
            for i, p in enumerate(points):
                if i > 0:
                    prev_rr = points[i-1]['rr']
                    if prev_rr > 0:
                        p['rri_ratio'] = p['rr'] / prev_rr
                    else:
                        p['rri_ratio'] = 1.0
                else:
                    p['rri_ratio'] = 1.0
            for p in points:
                p['hist_value'] = p['rri_ratio']
            ranked = points
        else:
            # Default: RR Interval
            for p in points:
                p['hist_value'] = p['rr']
            ranked = points

        if self._rank_mode == 'time':
            axis_label = 'Time (s)'
            for p in ranked:
                p['hist_value'] = max(0.0, float(p['t']))
        elif self._rank_mode == 'prematurity':
            # Holter prematurity ratio: current RR / running reference RR.
            # Values below 1.0 are premature; 1.0 is the reference beat.
            axis_label = 'Prematurity Ratio'
            reference = mean_rr if mean_rr > 0 else 1.0
            for p in ranked:
                p['hist_value'] = p['rr'] / reference
        elif self._rank_mode == 'similarity':
            # No waveform template is carried by the histogram metrics, so
            # use interval similarity to the median NN interval as the stable
            # available proxy: 1.0 = identical, approaching 0 = dissimilar.
            axis_label = 'Similarity (%)'
            reference = max(median_rr, 1.0)
            for p in ranked:
                distance = abs(p['rr'] - reference) / reference
                p['hist_value'] = 100.0 / (1.0 + distance)

        self._hist_canvas.set_histogram_data(ranked, mode=self._rank_mode, data_type=axis_label)

        self._hist_stats['nns'].setText(str(len(rr_arr)))
        self._hist_stats['mean_nn'].setText(f"{rr_arr.mean():.0f} ms")
        self._hist_stats['sdnn'].setText(f"{rr_arr.std():.0f} ms")
        self._hist_stats['sdann'].setText('?')
        d = np.diff(rr_arr)
        rmssd = np.sqrt(np.mean(d ** 2)) if len(d) > 0 else 0.0
        self._hist_stats['rmssd'].setText(f"{rmssd:.0f} ms")
        pnn50 = 100.0 * np.sum(np.abs(d) > 50) / len(d) if len(d) > 0 else 0.0
        self._hist_stats['pnn50'].setText(f"{pnn50:.2f}%")
        self._hist_stats['triidx'].setText('?')
        self._hist_stats['sdnnidx'].setText('?')

    def _on_bar_clicked(self, payload: dict):
        if not payload:
            self._selected_point = None
            self._selected_info.setText("Selected: none")
            return
        count = int(payload.get('count', 0) or 0)
        lo, hi = payload.get('range', (0.0, 0.0))
        rr_center = float(payload.get('center_rr', 0.0) or 0.0)
        times = [float(t) for t in (payload.get('times', []) or []) if float(t) >= 0]
        if times:
            target = float(np.median(times))
        else:
            target = float(payload.get('center_t', 0.0) or 0.0)
        self._selected_point = payload
        self._selected_info.setText(
            f"Selected: {count} beats in {lo:.0f}-{hi:.0f} ms range | center {rr_center:.0f} ms"
        )
        if target > 0:
            self.seek_requested.emit(target)

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 1:
            return
        N = data.shape[1]
        x = np.linspace(0, N / 500.0, N) if N > 0 else []
        if N > 0:
            self._strip.set_data(x, data[0].copy())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._hist_canvas._selected_index = -1
            self._hist_canvas.update()
            self._on_bar_clicked({})  # Pass empty dict instead of None
        super().mousePressEvent(event)
