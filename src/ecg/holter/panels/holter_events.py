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

# 7. HOLTER EVENTS PANEL
class HolterEventsPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._events = []
        self._session_dir = ""
        self._selected_payload = {}
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

        # Left: event list + stats
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        ev_title = QLabel("Events")
        ev_title.setStyleSheet(f"color:#07111F;font-size:13px;font-weight:bold;background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #28E37B, stop:1 #89F7C5);padding:6px 10px;border-radius:6px;")
        left_layout.addWidget(ev_title)

        cols = ["Event name", "Start Time", "Chan.", "Print Len.", "Source", "Conf."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.setStyleSheet(
            _table_style() +
            """
            QTableWidget::item:selected {
                background-color: rgba(66, 153, 225, 70);
                color: #F3F7FB;
                border: 1px solid rgba(255, 255, 255, 90);
            }
            QTableWidget::item:selected:active {
                background-color: rgba(66, 153, 225, 110);
            }
            """
        )
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._ev_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ev_table.cellClicked.connect(self._on_event_clicked)
        left_layout.addWidget(self._ev_table, 1)

        # Stats below table
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(8, 6, 8, 6)
        sf_layout.setSpacing(6)
        self._stat_labels = {}
        for i, (key, label) in enumerate([
            ("hr_max","HR Max"),("hr_min","HR Min"),("hr_smax","Sinus Max HR"),
            ("hr_smin","Sinus Min HR"),("brady","Bradycardia"),("user_ev","User Event"),
        ]):
            row, col = divmod(i, 2)
            l = QLabel(f"{label}:")
            l.setStyleSheet(f"color:{COL_WHITE};font-size:10px;font-weight:bold;border:none;")
            v = QLabel("-")
            v.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
            sf_layout.addWidget(l, row * 2, col)
            sf_layout.addWidget(v, row * 2 + 1, col)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 1)

        # Right: navigation
        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)
        for label in ["Prev Event", "Next Event", "Remove All", "Remove"]:
            btn = QPushButton(label)
            btn.setStyleSheet(_style_btn())
            btn.setFixedHeight(38)
            if label == "Prev Event":
                btn.clicked.connect(self._go_prev_event)
            elif label == "Next Event":
                btn.clicked.connect(self._go_next_event)
            elif label == "Remove All":
                btn.clicked.connect(self._remove_all_events)
            elif label == "Remove":
                btn.clicked.connect(self._remove_selected_event)
            nav_layout.addWidget(btn)
        nav_layout.addStretch()
        layout.addWidget(nav)

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir or ""

    def load_events(self, events: list, summary: dict):
        # Filter out hidden event labels from display
        hidden_labels = ["Long QT Syndrome", "Wide QRS (non-specific)", "Frequent PVCs", "Multifocal PVCs"]
        filtered_events = []
        for ev in events:
            label = str(ev.get('label', '')).lower()
            if not any(hl.lower() in label for hl in hidden_labels):
                filtered_events.append(ev)
        
        self._events = filtered_events
        self._ev_table.setRowCount(len(filtered_events))
        for i, ev in enumerate(filtered_events):
            t_str = _format_system_time(self._session_dir, ev['timestamp'])
            source = ev.get("source", "analysis")
            conf = ev.get("confidence", 0.0)
            for j, val in enumerate([ev['label'], t_str, "12", "7s", source, f"{float(conf or 0.0):.2f}"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                self._ev_table.setItem(i, j, item)
        s = summary
        for key, fmt in [("hr_max",f"{s.get('max_hr',0):.0f} bpm"),
                          ("hr_min",f"{s.get('min_hr',0):.0f} bpm"),
                          ("hr_smax",f"{s.get('max_hr',0):.0f} bpm"),
                          ("hr_smin",f"{s.get('min_hr',0):.0f} bpm"),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    def _on_event_clicked(self, row, col):
        self._select_and_seek(row)

    def _selected_row(self) -> int:
        selection = self._ev_table.selectionModel()
        if selection:
            rows = selection.selectedRows()
            if rows:
                return int(rows[0].row())
        return -1

    def _select_and_seek(self, row: int):
        if row < 0 or row >= len(self._events):
            return
        self._ev_table.selectRow(row)
        self._selected_payload = dict(self._events[row] or {})
        self.seek_requested.emit(float(self._events[row].get('timestamp', 0.0) or 0.0))

    def _go_prev_event(self):
        if not self._events:
            return
        row = self._selected_row()
        if row < 0:
            row = 0
        else:
            row = max(0, row - 1)
        self._select_and_seek(row)

    def _go_next_event(self):
        if not self._events:
            return
        row = self._selected_row()
        if row < 0:
            row = 0
        else:
            row = min(len(self._events) - 1, row + 1)
        self._select_and_seek(row)

    def _remove_selected_event(self):
        row = self._selected_row()
        if row < 0 or row >= len(self._events):
            return
        self._events.pop(row)
        self._ev_table.removeRow(row)
        self._selected_payload = {}
        if self._events:
            next_row = min(row, len(self._events) - 1)
            self._select_and_seek(next_row)

    def _remove_all_events(self):
        self._events = []
        self._selected_payload = {}
        self._ev_table.setRowCount(0)
        
