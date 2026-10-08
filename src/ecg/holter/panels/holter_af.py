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

# 12. HOLTER AF ANALYSIS PANEL
class HolterAFPanel(QWidget):
    def __init__(self, parent=None, session_dir: str = ""):
        super().__init__(parent)
        self.session_dir = session_dir
        self.setStyleSheet(f"background:{COL_BG};")
        self._replay_engine = None
        self._duration_sec = 0.0
        self._af_rows = []
        self._auto_segments_ready_for = None
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
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: AF event list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        title = QLabel("AF Analysis")
        title.setStyleSheet(f"color:#07111F;font-size:13px;font-weight:bold;background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #28E37B, stop:1 #89F7C5);padding:6px 10px;border-radius:6px;")
        left_layout.addWidget(title)

        cols = ["Start time", "Duration", "Type"]
        self._af_table = QTableWidget(0, len(cols))
        self._af_table.setHorizontalHeaderLabels(cols)
        self._af_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._af_table.setStyleSheet(_table_style())
        self._af_table.verticalHeader().setVisible(False)
        self._af_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._af_table.cellClicked.connect(self._on_af_row_clicked)
        left_layout.addWidget(self._af_table, 1)

        no_items = QLabel("There are no items to show.")
        no_items.setStyleSheet(f"color:{COL_GREEN_DRK};font-style:italic;padding:8px;border:none;")
        self._no_items_lbl = no_items
        left_layout.addWidget(no_items)

        nav_row = QHBoxLayout()
        for lbl in ["AF Analysis", "Parameters", "Prev Event", "Next Event", "Remove All", "Remove"]:

            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            nav_row.addWidget(btn)
        left_layout.addLayout(nav_row)
        layout.addWidget(left, 2)

        # Right: ECG strip + Lorenz
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background:transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(4)
        
        self._thumb_strips = []
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for i in range(12):
            lbl = QLabel(f"Lead {lead_names[i]}")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            scroll_layout.addWidget(lbl)
            
            # Disable vertical lines and labels for AF Analysis panel
            strip = ECGStripCanvas(height=120, show_vertical_lines=False)
            strip.lead_name = lead_names[i]
            scroll_layout.addWidget(strip)
            self._thumb_strips.append(strip)
            
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area, 1)

        # Disable vertical lines for bottom ECG strip too
        self._af_ecg_strip = ECGStripCanvas(height=70, show_vertical_lines=False)
        right_layout.addWidget(self._af_ecg_strip)

        self._af_lorenz = LorenzCanvas()
        self._af_lorenz.setFixedHeight(160)
        right_layout.addWidget(self._af_lorenz)
        layout.addWidget(right, 3)

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._auto_segments_ready_for = None
        self._refresh_af_events()

    def update_from_metrics(self, metrics_list: list, duration_sec: float = 0):
        self._duration_sec = float(duration_sec or 0.0)
        self._refresh_af_events()

    @staticmethod
    def _af_type_from_label(label: str) -> Optional[str]:
        label_lower = str(label or "").lower()
        # "Ventricular Fibrillation"/"Ventricular Tachycardia" must never
        # land here -- "fibrillation" is a substring of "Ventricular
        # Fibrillation" too, so without this guard a VFib episode gets
        # mislabeled "Atrial Fibrillation" in this table (and its row then
        # plays back the real, chaotic VFib strip under an AFib label).
        if "ventricular" in label_lower or "vfib" in label_lower or "vtach" in label_lower:
            return None
        if "flutter" in label_lower:
            return "Atrial Flutter"
        if "fibrillation" in label_lower or "afib" in label_lower:
            return "Atrial Fibrillation"
        return None

    def _load_auto_af_segments(self) -> list:
        """AF/AFlutter episodes auto-detected and merged on the replay engine
        (same source Full Disclosure's segment overlay draws from)."""
        segments = []
        engine = self._replay_engine
        if engine is None:
            return segments
        if self._auto_segments_ready_for is not engine:
            try:
                from ..auto_segment_arrthymia_detection import apply_auto_segments_to_engine
                apply_auto_segments_to_engine(engine)
            except Exception as e:
                print(f"[HolterAFPanel] Auto segment detection unavailable: {e}")
            self._auto_segments_ready_for = engine
        for ev in (getattr(engine, "_structured_events", []) or []):
            if ev.get("end_timestamp") is None:
                continue
            af_type = self._af_type_from_label(ev.get("label", ev.get("type", "")))
            if af_type is None:
                continue
            start_sec = float(ev.get("timestamp", 0.0) or 0.0)
            end_sec = float(ev.get("end_timestamp", start_sec) or start_sec)
            if end_sec > start_sec:
                segments.append({"start_sec": start_sec, "end_sec": end_sec, "type": af_type})
        return segments

    def _load_manual_af_segments(self) -> list:
        """AF/AFlutter episodes manually marked on Full Disclosure (persisted
        to manual_segments.json alongside the session)."""
        segments = []
        if not self.session_dir:
            return segments
        try:
            path = os.path.join(self.session_dir, "manual_segments.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    segments_data = json.load(f) or []
                for seg in segments_data:
                    af_type = self._af_type_from_label(seg.get("label", ""))
                    if af_type is None:
                        continue
                    start_sec = float(seg.get("start_sec", 0.0) or 0.0)
                    end_sec = float(seg.get("end_sec", start_sec) or start_sec)
                    if end_sec > start_sec:
                        segments.append({"start_sec": start_sec, "end_sec": end_sec, "type": af_type})
        except Exception as e:
            print(f"[HolterAFPanel] Error loading manual segments: {e}")
        return segments

    def _recording_start_timestamp(self) -> Optional[float]:
        engine = self._replay_engine
        reader = getattr(engine, "_reader", None) if engine is not None else None
        reader_start = getattr(reader, "start_time", None) if reader is not None else None
        if reader_start:
            return float(reader_start)
        if self.session_dir:
            try:
                ecgh_path = os.path.join(self.session_dir, "recording.ecgh")
                if os.path.exists(ecgh_path):
                    return os.path.getmtime(ecgh_path) - self._duration_sec
            except Exception as e:
                print(f"[HolterAFPanel] Error getting recording start time: {e}")
        return None

    def _refresh_af_events(self):
        segments = self._load_auto_af_segments() + self._load_manual_af_segments()
        segments.sort(key=lambda s: s["start_sec"])
        self._af_rows = segments

        recording_start_timestamp = self._recording_start_timestamp()

        self._af_table.setRowCount(len(segments))
        if segments:
            self._no_items_lbl.hide()
            for i, seg in enumerate(segments):
                start_sec = seg["start_sec"]
                duration_sec = seg["end_sec"] - start_sec
                if recording_start_timestamp:
                    start_str = datetime.fromtimestamp(recording_start_timestamp + start_sec).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    start_str = _sec_to_hms(start_sec)
                duration_str = _sec_to_hms(duration_sec)
                for j, val in enumerate([start_str, duration_str, seg["type"]]):
                    item = QTableWidgetItem(val)
                    item.setForeground(QColor(COL_WHITE))
                    if j == 0:
                        item.setData(Qt.UserRole, start_sec)
                    self._af_table.setItem(i, j, item)
        else:
            self._no_items_lbl.show()

    def _on_af_row_clicked(self, row: int, _col: int):
        if row < 0 or row >= len(self._af_rows):
            return
        engine = self._replay_engine
        reader = getattr(engine, "_reader", None) if engine is not None else None
        if reader is None:
            return
        start_sec = float(self._af_rows[row]["start_sec"])
        try:
            end_sec = min(float(getattr(engine, "duration_sec", start_sec + 10.0)), start_sec + 10.0)
            data = reader.read_range(start_sec, end_sec)
            if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] > 0:
                self.set_replay_frame(data)
        except Exception as e:
            print(f"[HolterAFPanel] Error loading AF strip: {e}")

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        fs = 500.0
        n_samples = min(N, int(10 * fs))
        x = np.linspace(0, n_samples/fs, n_samples) if n_samples > 0 else []
        if n_samples > 0:
            lead2_idx = 1 if data.shape[0] > 1 else 0
            self._af_ecg_strip.set_data(x, data[lead2_idx, :n_samples].copy())
            for i, ts in enumerate(self._thumb_strips):
                if i < data.shape[0]:
                    ts.set_data(x, data[i, :n_samples].copy())


