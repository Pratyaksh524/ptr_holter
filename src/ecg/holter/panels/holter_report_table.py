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
    _card_style, _card_title_style,
)

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas

# 16. HOLTER REPORT TABLE PANEL
class HolterReportTablePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{UI_BG};")
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
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(0)
        card = QFrame()
        card.setObjectName("rtCard")
        card.setStyleSheet(_card_style("QFrame#rtCard"))
        outer.addWidget(card)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Title; on the right the recording's span and how many hour rows.
        title_row = QHBoxLayout()
        title = QLabel("Hour-by-Hour Report Table")
        title.setStyleSheet(_card_title_style())
        title_row.addWidget(title)
        title_row.addStretch(1)
        self._span_chip = QLabel("")
        self._span_chip.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:bold;background:{UI_PANEL_ALT};"
                                      f"border:1px solid {UI_BORDER};border-radius:6px;padding:3px 10px;")
        self._span_chip.hide()
        title_row.addWidget(self._span_chip)
        layout.addLayout(title_row)

        cols = ["Hour", "Beats", "HR Min", "HR Avg", "HR Max",
                "VE Iso.", "VE Coup.", "VE Runs", "VE Total", "VE %",
                "SVE Iso.", "SVE Coup.", "SVE Total", "Pauses"]

        # Group bar over the columns: every column stretches to the same
        # width, so the groups' stretch factors are their column counts.
        group_bar = QHBoxLayout()
        group_bar.setSpacing(2)
        group_bar.setContentsMargins(1, 0, 1, 0)
        for text, n_cols in [("", 2), ("HEART RATE (bpm)", 3), ("VENTRICULAR ECTOPY (VE)", 5),
                             ("SUPRAVENTRICULAR ECTOPY (SVE)", 3), ("", 1)]:
            cell = QLabel(text)
            cell.setAlignment(Qt.AlignCenter)
            if text:
                cell.setStyleSheet(f"color:{UI_MUTED};font-size:9px;font-weight:700;letter-spacing:1px;"
                                   f"background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                                   f"border-radius:4px;padding:3px 0;")
            else:
                cell.setStyleSheet("background:transparent;border:none;")
            group_bar.addWidget(cell, n_cols)
        layout.addLayout(group_bar)

        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self._table.setAlternatingRowColors(True)
        self._table.setShowGrid(False)
        self._table.setStyleSheet(_table_style())
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(30)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

        # Empty state, centred over the table.
        self._no_items_lbl = QLabel("No hourly data")
        self._no_items_lbl.setParent(self._table.viewport())
        self._no_items_lbl.setAlignment(Qt.AlignCenter)
        self._no_items_lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._no_items_lbl.setStyleSheet(f"color:{UI_MUTED};font-size:12px;font-style:italic;padding:8px;"
                                         f"border:none;background:transparent;")
        self._no_items_lbl.hide()
        self._table.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        # Keep the empty-state text centred over the table's viewport.
        lbl = getattr(self, "_no_items_lbl", None)
        if lbl is not None and obj is self._table.viewport() and event.type() == QEvent.Resize:
            lbl.setGeometry(obj.rect())
        return super().eventFilter(obj, event)

    def update_from_metrics(self, metrics_list: list, session_dir: str = None, duration_sec: float = 0.0):
        """`session_dir` gives the hour rows their system time (the start of
        the hour; the recording's last, partial hour "start - end", as the
        PDF report's hourly table); without it the elapsed "00:00".
        `duration_sec` is the recording's length (the report's end time);
        the live chunks' span when not given."""
        duration = float(duration_sec or 0.0)
        if duration <= 0 and metrics_list:
            duration = float(_metrics_duration_sec(metrics_list) or 0.0)

        def hour_label(h):
            if not session_dir:
                return f"{h:02d}:00"
            label = _format_system_time(session_dir, h * 3600.0)
            if 0 < duration < (h + 1) * 3600:
                label += " \u2013 " + _format_system_time(session_dir, duration)
            return label

        hourly: Dict[int, list] = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        rows = []
        total_beats = total_pauses = 0
        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            beats = sum(c.get('beat_count', 0) for c in chunks)
            hr_vals = [c.get('hr_mean', 0) for c in chunks if c.get('hr_mean', 0) > 0]
            hr_min_vals = [c.get('hr_min', 0) for c in chunks if c.get('hr_min', 0) > 0]
            hr_max_vals = [c.get('hr_max', 0) for c in chunks if c.get('hr_max', 0) > 0]
            pauses = sum(c.get('pauses', 0) for c in chunks)
            avg_hr = int(np.mean(hr_vals)) if hr_vals else 0
            min_hr = int(np.min(hr_min_vals)) if hr_min_vals else 0
            max_hr = int(np.max(hr_max_vals)) if hr_max_vals else 0
            total_beats += beats
            total_pauses += pauses
            rows.append([hour_label(h), str(beats), str(min_hr), str(avg_hr), str(max_hr),
                          "0","0","0","0","0%","0","0","0", str(pauses)])

        # Total row
        rows.append(["Total", str(total_beats), "-", "-", "-",
                      "0","0","0","0","0%","0","0","0", str(total_pauses)])

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            is_total = (i == len(rows) - 1)
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                # numbers centred under their header; the hour on the left
                item.setTextAlignment((Qt.AlignLeft if j == 0 else Qt.AlignCenter) | Qt.AlignVCenter)
                item.setForeground(QColor(UI_TEXT))
                if is_total:
                    # total row: bold on the panel shade (was green on blue)
                    item.setBackground(QColor(UI_PANEL_ALT))
                    f = item.font(); f.setBold(True); item.setFont(f)
                self._table.setItem(i, j, item)

        has_data = len(rows) > 1
        self._no_items_lbl.setVisible(not has_data)
        self._no_items_lbl.setGeometry(self._table.viewport().rect())
        if has_data and session_dir and duration > 0:
            n_hours = len(rows) - 1
            self._span_chip.setText(f"{_format_system_time(session_dir, 0.0)} \u2192 "
                                    f"{_format_system_time(session_dir, duration)}  \u00b7  "
                                    f"{n_hours} hour row" + ("" if n_hours == 1 else "s"))
            self._span_chip.show()
        else:
            self._span_chip.hide()

