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

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas, HolterRRTrendCanvas, HRTrendCanvas

# 4. HOLTER OVERVIEW PANEL
class HolterOverviewPanel(QWidget):
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Overview")
        title.setStyleSheet(
            f"color:{UI_TEXT};font-size:14px;font-weight:700;background:{UI_PANEL_ALT};"
            f"padding:8px;border-radius:6px;border:1px solid {UI_BORDER};"
        )
        layout.addWidget(title)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Name", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.setStyleSheet(_table_style())
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

    def update_summary(self, s: dict):
        rows = [
            ("Total Beats",          f"{s.get('total_beats', 0):,}"),
            ("AVG Heart Rate",       f"{s.get('avg_hr', 0):.0f} bpm"),
            ("Max HR",               f"{s.get('max_hr', 0):.0f} bpm"),
            ("Min HR",               f"{s.get('min_hr', 0):.0f} bpm"),
            ("Sinus Max HR",         f"{s.get('max_hr', 0):.0f} bpm"),
            ("Sinus Min HR",         f"{s.get('min_hr', 0):.0f} bpm"),
            ("Longest RR Interval",  f"{s.get('longest_rr_ms', 0)/1000:.2f}s"),
            ("RRI (>=2.0s)",          str(pauses)),
            ("Tachycardia Beats",    str(s.get('tachy_beats', 0))),
            ("Bradycardia Beats",    str(s.get('brady_beats', 0))),
            ("Ventricular Beats",    str(s.get('ve_beats', 0))),
            ("Supraventricular Beats", str(s.get('sve_beats', 0))),
            ("Template Clusters",    str(s.get('template_count', 0))),
            ("X Total",              str(s.get('pauses', 0))),
            ("SDNN (HRV)",           f"{s.get('sdnn', 0):.1f} ms"),
            ("rMSSD (HRV)",          f"{s.get('rmssd', 0):.1f} ms"),
            ("pNN50 (HRV)",          f"{s.get('pnn50', 0):.2f}%"),
            ("ST Elevation",         "-"),
            ("ST Depression",        "-"),
            ("Signal Quality",       f"{s.get('avg_quality', 1.0)*100:.1f}%"),
            ("Chunks Analyzed",      str(s.get('chunks_analyzed', 0))),
        ]
        self._table.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            ni = QTableWidgetItem(name)
            ni.setForeground(QColor(UI_MUTED))
            ni.setBackground(QColor(UI_PANEL if i % 2 == 0 else UI_PANEL_ALT))
            vi = QTableWidgetItem(value)
            vi.setForeground(QColor(UI_TEXT))
            vi.setBackground(QColor(UI_PANEL if i % 2 == 0 else UI_PANEL_ALT))
            vi.setFont(QFont("Arial", 12, QFont.Bold))
            self._table.setItem(i, 0, ni)
            self._table.setItem(i, 1, vi)
        self._table.resizeRowsToContents()


# 4b. HOLTER EXPERT REVIEW PANEL
class HolterExpertReviewPanel(QWidget):
    """HolterExpert-inspired review layout: trend + Lorenz + strips + overview."""
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metrics = []
        self._summary = {}
        self._template_rows = []
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
        layout.setSpacing(8)

        self._rr_trend_full = HolterRRTrendCanvas(title="RR Interval Trend (Full Recording)")
        layout.addWidget(self._rr_trend_full)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(1)
        body.setStyleSheet(f"QSplitter{{background:{UI_BG};}} QSplitter::handle{{background:{UI_BORDER};}}")

        left = QFrame()
        left.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:8px;}}")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(8, 8, 8, 8)
        left_l.setSpacing(8)
        self._lorenz = LorenzCanvas()
        self._lorenz.setMinimumHeight(280)
        left_l.addWidget(self._lorenz, 2)
        self._template_table = QTableWidget(0, 3)
        self._template_table.setHorizontalHeaderLabels(["Template", "Class", "Beats"])
        self._template_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._template_table.verticalHeader().setVisible(False)
        self._template_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._template_table.setStyleSheet(_table_style())
        self._template_table.cellClicked.connect(self._on_template_clicked)
        left_l.addWidget(self._template_table, 1)
        body.addWidget(left)

        #  "  "  Center: scrollable 12-lead ECG grid  "  " 
        center = QFrame()
        center.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {UI_BORDER};border-radius:8px;}}")
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(4, 4, 4, 4)
        center_l.setSpacing(0)

        # 12-lead header
        hdr = QLabel("12-Lead ECG Overview")
        hdr.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:700;padding:4px 6px;border:none;")
        center_l.addWidget(hdr)

        leads_scroll = QScrollArea()
        leads_scroll.setWidgetResizable(True)
        leads_scroll.setFrameShape(QFrame.NoFrame)
        leads_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        leads_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        leads_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        leads_container = QWidget()
        leads_container.setStyleSheet(f"background:{COL_BLACK};")
        leads_vbox = QVBoxLayout(leads_container)
        leads_vbox.setContentsMargins(2, 2, 2, 2)
        leads_vbox.setSpacing(2)

        self._lead_names_12 = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        self._expert_lead_strips = {}  # lead_name -> ECGStripCanvas
        for lead in self._lead_names_12:
            row_h = QHBoxLayout()
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(4)
            lbl = QLabel(lead)
            lbl.setFixedWidth(34)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
            # Expert/Overview mode: disable ALL coloring (disable_all_coloring=True) - no arrhythmia colors, just green
            strip = ECGStripCanvas(height=60, color="#00FF00", pen_width=0.9, lead_name=lead, show_vertical_lines=False, show_annotations=False, disable_all_coloring=True)
            strip.set_gain(1.0)
            self._expert_lead_strips[lead] = strip
            row_h.addWidget(lbl)
            row_h.addWidget(strip, 1)
            leads_vbox.addLayout(row_h)

        leads_scroll.setWidget(leads_container)
        center_l.addWidget(leads_scroll, 1)

        # Rhythm strip at bottom (Lead II) - also disable vertical lines and ALL coloring
        rhythm_row = QHBoxLayout()
        rhythm_row.setSpacing(4)
        rlbl = QLabel("II")
        rlbl.setFixedWidth(34)
        rlbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rlbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
        self._mini = ECGStripCanvas(height=40, color="#00AA00", pen_width=0.9, show_vertical_lines=False, show_annotations=False, disable_all_coloring=True)
        rhythm_row.addWidget(rlbl)
        rhythm_row.addWidget(self._mini, 1)
        center_l.addLayout(rhythm_row)

        body.addWidget(center)

        right = QFrame()
        right.setStyleSheet(f"""
            QFrame {{
                background: #0A0D10;
                border: 1px solid #1E2832;
                border-radius: 4px;
            }}
        """)
        right_l = QVBoxLayout(right)
        ttl = QLabel("Overview")
        ttl.setStyleSheet("color: #FFFFFF; font-weight: bold; font-size: 13px; padding: 5px 8px; background: #131A22; border: 1px solid #232D38; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px;")
        right_l.addWidget(ttl)

        # Overview Table matching reference workstation layout
        self._overview = QTableWidget(0, 2)
        self._overview.setHorizontalHeaderLabels(["Name", "Value"])
        self._overview.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._overview.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._overview.verticalHeader().setVisible(False)
        self._overview.setEditTriggers(QTableWidget.NoEditTriggers)
        self._overview.setSelectionMode(QAbstractItemView.NoSelection)
        self._overview.setFocusPolicy(Qt.NoFocus)
        self._overview.setShowGrid(True)
        self._overview.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._overview.horizontalHeader().setFixedHeight(28)

        self._overview.setStyleSheet("""
            QTableWidget {
                background-color: #0A0E13;
                color: #DCE3EA;
                border: 1px solid #232D38;
                gridline-color: #1A222C;
                font-size: 11px;
                font-family: 'Segoe UI', Arial, sans-serif;
                outline: none;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1F2833, stop:1 #131A22);
                color: #FFFFFF;
                font-weight: bold;
                font-size: 11px;
                padding-left: 8px;
                border: 1px solid #2A3644;
                height: 26px;
            }
            QTableWidget::item {
                padding-left: 8px;
                padding-right: 8px;
                border-bottom: 1px solid #141C24;
            }
        """)

        right_l.addWidget(self._overview, 1)
        body.addWidget(right)
        body.setSizes([360, 760, 310])
        layout.addWidget(body, 1)

    def update_from_metrics(self, metrics_list: list, summary: dict):
        self._metrics = list(metrics_list or [])
        self._summary = dict(summary or {})
        rr_points = []
        for m in self._metrics:
            t0 = float(m.get("t", 0.0) or 0.0)
            rr_list = list(m.get("rr_intervals_list", []) or [])
            dur = float(m.get("duration", 0.0) or 0.0)
            n = max(1, len(rr_list))
            step = (dur / n) if dur > 0 else 0.2
            for i, rr in enumerate(rr_list):
                rr_points.append((t0 + i * step, float(rr)))



        start_epoch = summary.get("start_time_epoch", None)
        self._rr_trend_full.set_points(rr_points, start_epoch)
        
        # Update Lorenz plot with all RR intervals from the full recording
        x = [p[1] for p in rr_points[:-1]] if len(rr_points) > 1 else []
        y = [p[1] for p in rr_points[1:]] if len(rr_points) > 1 else []
        self._lorenz.set_data(x, y)

        tm = {}
        for m in self._metrics:
            for row in (m.get("template_summary", []) or []):
                key = row.get("template_key") or row.get("template_id") or row.get("label") or "T"
                r = tm.setdefault(key, {"id": row.get("template_id", "T"), "label": row.get("label", "N"), "count": 0, "first": float(row.get("first_timestamp", m.get("t", 0.0)) or 0.0)})
                r["count"] += int(row.get("count", 0) or 0)
                r["first"] = min(r["first"], float(row.get("first_timestamp", r["first"]) or r["first"]))
        self._template_rows = sorted(tm.values(), key=lambda it: it["count"], reverse=True)
        self._template_table.setRowCount(len(self._template_rows))
        for i, row in enumerate(self._template_rows):
            vals = [str(row["id"]), str(row["label"]), str(row["count"])]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setForeground(QColor(UI_TEXT))
                self._template_table.setItem(i, j, item)

        # Compute tachy/brady durations from summary
        _dur_sec = float(summary.get('duration_sec', 0) or 0)
        _total_beats = int(summary.get('total_beats', 0) or 1)
        # Use pre-computed sec values if available (set by _build_summary_from_metrics)
        if 'tachy_sec' in summary:
            _tachy_sec = float(summary['tachy_sec'] or 0)
            _brady_sec = float(summary['brady_sec'] or 0)
            _tachy_pct = float(summary.get('tachy_pct', 0) or 0)
            _brady_pct = float(summary.get('brady_pct', 0) or 0)
        else:
            _tachy_beats = int(summary.get('tachy_beats', 0) or 0)
            _brady_beats = int(summary.get('brady_beats', 0) or 0)
            _avg_rr_sec = (_dur_sec / _total_beats) if _total_beats > 0 and _dur_sec > 0 else 0.0
            _tachy_sec = _tachy_beats * _avg_rr_sec
            _brady_sec = _brady_beats * _avg_rr_sec
            _tachy_pct = (_tachy_beats / _total_beats * 100) if _total_beats > 0 else 0.0
            _brady_pct = (_brady_beats / _total_beats * 100) if _total_beats > 0 else 0.0
        def _fmt_dur(sec):
            s = int(sec); h = s // 3600; m = (s % 3600) // 60; ss = s % 60
            return f"{h:02d}:{m:02d}:{ss:02d}"
        _tachy_str = f"{_fmt_dur(_tachy_sec)} {_tachy_pct:.2f}%"
        _brady_str = f"{_fmt_dur(_brady_sec)} {_brady_pct:.2f}%"

        # Sinus HR from summary (falls back to max_hr/min_hr)
        _sinus_max = summary.get('sinus_max_hr', summary.get('max_hr', 0))
        _sinus_min = summary.get('sinus_min_hr', summary.get('min_hr', 0))
        _sinus_max_t = summary.get('sinus_max_hr_time', '')
        _sinus_min_t = summary.get('sinus_min_hr_time', '')
        _max_hr_t = summary.get('max_hr_time', '')
        _min_hr_t = summary.get('min_hr_time', '')

        # AF duration from arrhythmia_counts
        _af_count = int((summary.get('arrhythmia_counts') or {}).get('AF', 0))
        _af_chunk_dur = 4.0  # each metric chunk ~4s
        _af_sec = _af_count * _af_chunk_dur
        _af_pct = (_af_sec / _dur_sec * 100) if _dur_sec > 0 else 0.0
        _af_str = f"{_fmt_dur(_af_sec)} {_af_pct:.2f}%" if _af_count > 0 else "-"

        def _hr_with_time(hr, t):
            hr_str = f"{int(round(float(hr or 0)))}bpm"
            return f"{hr_str} {t}" if t else hr_str

        rows = [
            ("Total", f"{summary.get('total_beats', 0)}"),
            ("X Total", f"{summary.get('pauses', 0)}"),
            ("AVG HR", f"{summary.get('avg_hr', 0):.0f} bpm"),
            ("Max HR", _hr_with_time(summary.get('max_hr', 0), _max_hr_t)),
            ("Min HR", _hr_with_time(summary.get('min_hr', 0), _min_hr_t)),
            ("Sinus Max HR", _hr_with_time(_sinus_max, _sinus_max_t)),
            ("Sinus Min HR", _hr_with_time(_sinus_min, _sinus_min_t)),
            ("During of Tachy.", _tachy_str),
            ("During of Brady.", _brady_str),
            ("V Total", f"{summary.get('ve_beats', 0)}"),
            ("S Total", f"{summary.get('sve_beats', 0)}"),
            ("AVG HR of Af/AF", "-"),
            ("Duration of Af/AF", _af_str),
            ("Paced Beats", "0"),
            ("Pacing AVG HR", "-"),
            ("Pacing Max HR", "-"),
            ("Pacing Min HR", "-"),
            ("Longest RR", f"{summary.get('longest_rr_ms', 0)/1000:.2f}s"),
            ("RRI (\u22652.0s)", f"{summary.get('pauses', 0)}"),
            ("ST Elevation", "-"),
            ("ST Depression", "-"),
        ]
        self._overview.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self._overview.setRowHeight(i, 24)
            ki = QTableWidgetItem(k)
            vi = QTableWidgetItem(v)
            bg_col = QColor("#121822") if i % 2 == 0 else QColor("#0A0D13")
            ki.setBackground(bg_col)
            vi.setBackground(bg_col)
            ki.setForeground(QColor("#FFFFFF"))
            vi.setForeground(QColor("#FFFFFF"))
            self._overview.setItem(i, 0, ki)
            self._overview.setItem(i, 1, vi)

    def set_replay_frame(self, data, beat_annotations=None, start_sec=0.0):
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] == 0:
            return
        n = data.shape[1]
        x = np.arange(n, dtype=float) / 500.0
        # Feed all 12 leads to the new lead-strip dict
        for i, lead in enumerate(self._lead_names_12):
            if i < data.shape[0]:
                strip = self._expert_lead_strips.get(lead)
                if strip:
                    strip.set_data(x, data[i].copy(), beat_annotations=beat_annotations, start_sec=start_sec)
        # Rhythm strip: Lead II (index 1)
        if data.shape[0] > 1:
            self._mini.set_data(x, data[1].copy(), beat_annotations=beat_annotations, start_sec=start_sec)
        elif data.shape[0] > 0:
            self._mini.set_data(x, data[0].copy(), beat_annotations=beat_annotations, start_sec=start_sec)

    def _on_template_clicked(self, row, _col):
        if 0 <= row < len(self._template_rows):
            self.seek_requested.emit(float(self._template_rows[row].get("first", 0.0)))

