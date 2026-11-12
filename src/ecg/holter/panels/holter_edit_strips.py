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


def _es_tool_btn_style() -> str:
    # Same look as _style_btn but without the horizontal padding so the
    # fixed-size (28x28) tool buttons don't clip their text.
    return _style_btn() + "QPushButton{padding:0px 8px;border-radius:6px;}"


def _es_danger_btn_style() -> str:
    return _style_btn()

from ..summary_utils import derive_hr_focus_summary
from .holter_widgets import ClickableSummaryTile, ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas

# 15. HOLTER EDIT STRIPS PANEL
class HolterEditStripsPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{UI_BG};")
        self._events = []
        self._summary = {}
        self._metrics_list = []
        self._focus_cards = {}
        self._selected_focus_key = "max_hr"
        self._selected_payload = {}
        self._tile_widgets = {}
        self._stat_labels = {}
        self._session_dir = ""
        self._build_ui()

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir or ""


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

        # --- LEFT: Event List (20% width) ---
        left = QFrame()
        left.setObjectName("esLeftCard")
        left.setStyleSheet(_card_style("QFrame#esLeftCard"))
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        ev_title = QLabel("Event List")
        ev_title.setStyleSheet(_card_title_style())
        left_layout.addWidget(ev_title)

        cols = ["Event name", "Start Time", "Chan.", "Print Len."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._ev_table.setStyleSheet(
            _table_style() +
            f"""
            QTableWidget::item:selected {{
                background-color: #1E3A5F;
                color: {UI_TEXT};
                border: none;
            }}
            QTableWidget::item:selected:active {{
                background-color: #22457A;
            }}
            """
        )
        self._ev_table.setAlternatingRowColors(True)
        self._ev_table.setShowGrid(False)
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._ev_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ev_table.cellClicked.connect(self._on_event_clicked)
        left_layout.addWidget(self._ev_table, 1)

        nav_row = QHBoxLayout()
        for lbl in ["Prev", "Next", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_es_danger_btn_style() if lbl.startswith("Remove") else _style_btn())
            btn.setCursor(Qt.PointingHandCursor)
            nav_row.addWidget(btn)
        left_layout.addLayout(nav_row)

        stats_title = QLabel("Summary")
        stats_title.setStyleSheet(_card_title_style())
        left_layout.addWidget(stats_title)
        stats_frame = QFrame()
        stats_frame.setObjectName("esStatsCard")
        stats_frame.setStyleSheet(
            f"QFrame#esStatsCard{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};border-radius:6px;}}"
        )
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 8, 10, 8)
        sf_layout.setSpacing(6)
        for i, (key, label) in enumerate([
            ("hr", "Avg HR"),
            ("max_hr", "HR Max"),
            ("min_hr", "HR Min"),
            ("smax_hr", "Sinus Max HR"),
            ("smin_hr", "Sinus Min HR"),
            ("rr_int", "Longest RR"),
            ("brady", "Brady"),
            ("user_ev", "User Event"),
        ]):
            row, col = divmod(i, 2)
            l = QLabel(f"{label}:")
            l.setStyleSheet(f"color:{UI_MUTED};font-size:10px;font-weight:bold;letter-spacing:0.5px;border:none;background:transparent;")
            v = QLabel("-")
            v.setStyleSheet(f"color:{UI_TEXT};font-size:12px;font-weight:bold;border:none;background:transparent;")
            sf_layout.addWidget(l, row * 2, col)
            sf_layout.addWidget(v, row * 2 + 1, col)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 2)

        # --- CENTER: 2x2 Thumbnail Boxes (30% width) ---
        center = QFrame()
        center.setObjectName("esCenterCard")
        center.setStyleSheet(_card_style("QFrame#esCenterCard"))
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        # Tool buttons
        tool_row = QHBoxLayout()
        for icon in ["Refresh", "Menu", "Prev", "Next", "<<", ">>", "Up", "Down", "End"]:
            btn = QPushButton(icon)
            btn.setStyleSheet(_es_tool_btn_style())
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(28)
            btn.setMinimumWidth(36)
            tool_row.addWidget(btn)
        tool_row.addStretch()
        center_layout.addLayout(tool_row)

        thumb_grid = QGridLayout()
        thumb_grid.setSpacing(12)
        thumb_grid.setHorizontalSpacing(12)
        thumb_grid.setVerticalSpacing(12)
        self._thumb_frames = []
        self._tile_style_normal = (
            f"QFrame#esTile{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};border-radius:8px;}}"
            f"QFrame#esTile:hover{{border:1px solid {UI_ACCENT};}}"
        )
        self._tile_style_active = f"QFrame#esTile{{background:#16294A;border:2px solid {UI_ACCENT};border-radius:8px;}}"
        self._tile_widgets = {}
        tile_specs = [
            (0, 0, "max_hr", "Maximum Heart Rate"),
            (0, 1, "min_hr", "Minimum Heart Rate"),
            (1, 0, "smax_hr", "Sinus Max HR"),
            (1, 1, "smin_hr", "Sinus Min HR"),
        ]
        for row, col, key, title in tile_specs:
            frame = ClickableSummaryTile(key)
            frame.setObjectName("esTile")
            frame.setMinimumHeight(210)
            frame.setMinimumWidth(280)
            frame.setStyleSheet(self._tile_style_normal)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 8, 8, 8)
            fl.setSpacing(4)

            header_w = QWidget()
            header_w.setStyleSheet("border:none;background:transparent;")
            hl = QHBoxLayout(header_w)
            hl.setContentsMargins(0, 0, 0, 0)
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet(f"color:{UI_TEXT};font-size:12px;font-weight:bold;border:none;background:transparent;")
            hl.addWidget(t_lbl)
            hl.addStretch()
            hr_lbl = QLabel("HR: --")
            hr_lbl.setStyleSheet(f"color:{UI_WARNING};font-size:11px;font-weight:bold;border:none;background:transparent;")
            hl.addWidget(hr_lbl)
            fl.addWidget(header_w)

            time_lbl = QLabel("--:--:--")
            time_lbl.setStyleSheet(f"color:{UI_MUTED};font-size:10px;border:none;background:transparent;")
            fl.addWidget(time_lbl)

            strips = []
            for _ in range(3):
                strip = ECGStripCanvas(height=40)
                strip.setStyleSheet("border:1px solid #444444;")
                fl.addWidget(strip)
                strips.append(strip)
                self._thumb_frames.append(strip)

            self._tile_widgets[key] = {
                "frame": frame,
                "title": t_lbl,
                "hr": hr_lbl,
                "time": time_lbl,
                "strips": strips,
            }
            frame.clicked.connect(self._select_focus_tile)
            thumb_grid.addWidget(frame, row, col)

        thumb_grid.setColumnStretch(0, 1)
        thumb_grid.setColumnStretch(1, 1)
        center_layout.addLayout(thumb_grid)
        center_layout.addStretch()
        layout.addWidget(center, 4)

        # --- RIGHT: Large Clinical Strip View (50% width) ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        detail_frame = QFrame()
        detail_frame.setObjectName("esDetailCard")
        detail_frame.setStyleSheet(_card_style("QFrame#esDetailCard"))
        df_layout = QGridLayout(detail_frame)
        df_layout.setContentsMargins(10, 8, 10, 8)
        df_layout.setHorizontalSpacing(10)
        df_layout.setVerticalSpacing(4)
        self._detail_title = QLabel("Select a heart-rate tile")
        self._detail_title.setStyleSheet(
            f"color:{UI_TEXT};font-size:14px;font-weight:bold;border:none;background:transparent;"
            f"border-left:3px solid {UI_ACCENT};padding-left:8px;"
        )
        self._detail_value = QLabel("-- bpm")
        self._detail_value.setStyleSheet(
            f"color:{UI_SUCCESS};font-size:26px;font-weight:bold;border:none;background:transparent;"
        )
        self._detail_time = QLabel("-")
        self._detail_time.setStyleSheet(
            f"color:{UI_WARNING};font-size:11px;font-weight:bold;border:none;background:transparent;"
        )
        self._detail_note = QLabel("Click any tile to expand the matching strip view here.")
        self._detail_note.setStyleSheet(
            f"color:{UI_MUTED};font-size:11px;border:none;background:transparent;"
        )
        self._detail_note.setWordWrap(True)
        df_layout.addWidget(self._detail_title, 0, 0, 1, 2)
        df_layout.addWidget(self._detail_value, 1, 0, 1, 1)
        df_layout.addWidget(self._detail_time, 1, 1, 1, 1, Qt.AlignRight | Qt.AlignVCenter)
        df_layout.addWidget(self._detail_note, 2, 0, 1, 2)
        right_layout.addWidget(detail_frame, 0)

        main_frame = QFrame()
        main_frame.setObjectName("esMainCard")
        main_frame.setStyleSheet(f"QFrame#esMainCard{{background:{COL_BLACK};border:1px solid {UI_BORDER};border-radius:8px;}}")
        ml = QVBoxLayout(main_frame)
        ml.setContentsMargins(8, 8, 8, 8)

        self._detail_header_lbl = QLabel("Detailed View")
        self._detail_header_lbl.setStyleSheet(_card_title_style())
        ml.addWidget(self._detail_header_lbl)

        self._main_strips = []
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(6)
        
        lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        for name in lead_names:
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{COL_GREEN};border:none;font-weight:bold;background:transparent;")
            strip = ECGStripCanvas(height=95)
            scroll_layout.addWidget(lbl)
            scroll_layout.addWidget(strip)
            self._main_strips.append(strip)
            
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        ml.addWidget(scroll_area)

        right_layout.addWidget(main_frame, 1)

        self._mini_lbl = QLabel("Lead II")
        self._mini_lbl.setStyleSheet(f"color:{COL_GREEN};border:none;font-weight:bold;background:transparent;")
        right_layout.addWidget(self._mini_lbl)

        self._mini = ECGStripCanvas(height=40, color="#00AA00")
        right_layout.addWidget(self._mini)
        layout.addWidget(right, 5)

    def _format_bpm(self, value: float) -> str:
        if value and value > 0:
            return f"{float(value):.0f} bpm"
        return "-- bpm"

    def _on_event_clicked(self, row, col):
        if row < len(self._events):
            self._ev_table.selectRow(row)
            self.seek_requested.emit(self._events[row]['timestamp'])

    def _build_focus_cards(self):
        summary = self._summary or {}
        focus = derive_hr_focus_summary(self._metrics_list or [])

        def _metric_value(*keys, default=0.0):
            for key in keys:
                for source in (summary, focus):
                    if source.get(key) is not None:
                        try:
                            value = float(source.get(key) or 0.0)
                            if value > 0:
                                return value
                        except Exception:
                            continue
            return float(default)

        def _metric_time(*keys):
            for key in keys:
                for source in (summary, focus):
                    val = source.get(f"{key}_time")
                    if val:
                        return str(val)
            return ""

        def _metric_timestamp(*keys):
            for key in keys:
                for source in (summary, focus):
                    val = source.get(f"{key}_timestamp")
                    if val is not None:
                        try:
                            ts = float(val or 0.0)
                            if ts > 0:
                                return ts
                        except Exception:
                            continue
            return 0.0

        self._focus_cards = {
            "max_hr": {
                "title": "Maximum Heart Rate",
                "value": _metric_value("max_hr"),
                "time": _metric_time("max_hr"),
                "timestamp": _metric_timestamp("max_hr"),
                "note": "Highest overall heart rate found in the recording.",
            },
            "min_hr": {
                "title": "Minimum Heart Rate",
                "value": _metric_value("min_hr"),
                "time": _metric_time("min_hr"),
                "timestamp": _metric_timestamp("min_hr"),
                "note": "Lowest overall heart rate found in the recording.",
            },
            "smax_hr": {
                "title": "Sinus Max HR",
                "value": _metric_value("sinus_max_hr", "max_hr"),
                "time": _metric_time("sinus_max_hr", "max_hr"),
                "timestamp": _metric_timestamp("sinus_max_hr", "max_hr"),
                "note": "Highest heart rate from sinus-like beats.",
            },
            "smin_hr": {
                "title": "Sinus Min HR",
                "value": _metric_value("sinus_min_hr", "min_hr"),
                "time": _metric_time("sinus_min_hr", "min_hr"),
                "timestamp": _metric_timestamp("sinus_min_hr", "min_hr"),
                "note": "Lowest heart rate from sinus-like beats.",
            },
        }

    def _refresh_focus_tiles(self):
        for key, widget in self._tile_widgets.items():
            card = self._focus_cards.get(key, {})
            widget["hr"].setText(f"HR: {self._format_bpm(card.get('value', 0.0))}")
            widget["time"].setText(card.get("time") or "-")
            widget["frame"].setStyleSheet(
                self._tile_style_active if key == self._selected_focus_key else self._tile_style_normal
            )

    def _update_detail_panel(self, key: str):
        card = self._focus_cards.get(key) or self._focus_cards.get("max_hr") or {}
        title = card.get("title", "Heart Rate Detail")
        value = card.get("value", 0.0)
        time_str = card.get("time") or "-"
        note = card.get("note", "")
        self._detail_title.setText(title)
        self._detail_value.setText(self._format_bpm(value))
        self._detail_time.setText(time_str)
        self._detail_note.setText(note or "Click a tile to expand the matching strip view here.")
        self._detail_header_lbl.setText(f"Detailed View  -  {title}  -  {time_str}")

    def _select_focus_tile(self, tile_key: str, emit_seek: bool = True):
        if tile_key not in self._focus_cards:
            return
        self._selected_focus_key = tile_key
        self._refresh_focus_tiles()
        self._update_detail_panel(tile_key)
        if emit_seek:
            timestamp = float(self._focus_cards.get(tile_key, {}).get("timestamp", 0.0) or 0.0)
            if timestamp > 0:
                self.seek_requested.emit(timestamp)

    def load_events(self, events: list, summary: dict, metrics_list: Optional[list] = None):
        # Filter out hidden event labels from display
        hidden_labels = ["Long QT Syndrome", "Wide QRS (non-specific)", "Frequent PVCs", "Multifocal PVCs"]
        filtered_events = []
        for ev in events:
            label = str(ev.get('label', '')).lower()
            if not any(hl.lower() in label for hl in hidden_labels):
                filtered_events.append(ev)
        
        self._events = filtered_events
        self._summary = dict(summary or {})
        self._metrics_list = list(metrics_list or [])
        self._build_focus_cards()
        self._ev_table.setRowCount(len(filtered_events))
        for i, ev in enumerate(filtered_events):
            t_str = _format_system_time(self._session_dir, ev['timestamp'])
            for j, val in enumerate([ev['label'], t_str, "12", "7s"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(UI_TEXT))
                self._ev_table.setItem(i, j, item)
        if self._selected_focus_key not in self._focus_cards:
            self._selected_focus_key = "max_hr"
        self._refresh_focus_tiles()
        self._update_detail_panel(self._selected_focus_key)
        s = self._summary
        max_card = self._focus_cards.get("max_hr", {})
        min_card = self._focus_cards.get("min_hr", {})
        smax_card = self._focus_cards.get("smax_hr", {})
        smin_card = self._focus_cards.get("smin_hr", {})
        for key, fmt in [("hr",f"{s.get('avg_hr',0):.0f} bpm"),
                          ("max_hr",self._format_bpm(float(max_card.get('value', s.get('max_hr',0)) or 0.0))),
                          ("min_hr",self._format_bpm(float(min_card.get('value', s.get('min_hr',0)) or 0.0))),
                          ("smax_hr",self._format_bpm(float(smax_card.get('value', s.get('sinus_max_hr', s.get('max_hr',0))) or 0.0))),
                          ("smin_hr",self._format_bpm(float(smin_card.get('value', s.get('sinus_min_hr', s.get('min_hr',0))) or 0.0))),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1"),("event","1"),
                          ("rr_int",f"{s.get('longest_rr_ms',0):.0f} ms"),
                          ("atrial_ecto","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    def set_replay_frame(self, data, metrics_dict=None, current_sec=0.0):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        x = np.linspace(0, N/250.0, N) if N > 0 else []
        
        start_sec = max(0.0, current_sec - 5.0) # 10s window centered
        all_beats = metrics_dict.get('all_beats', []) if metrics_dict else []
        
        if N > 0:
            # Update thumbnails (cycle through CH1, CH2, CH3 if available)
            for i, strip in enumerate(self._thumb_frames):
                ch_idx = i % 3
                if ch_idx < data.shape[0]:
                    strip.set_data(x, data[ch_idx].copy(), beat_annotations=all_beats, start_sec=start_sec)
                else:
                    strip.set_data(x, data[0].copy(), beat_annotations=all_beats, start_sec=start_sec)
            
            # Update large main strips
            if hasattr(self, "_main_strips"):
                for idx, strip in enumerate(self._main_strips):
                    if idx < data.shape[0]:
                        strip.set_data(x, data[idx].copy(), beat_annotations=all_beats, start_sec=start_sec)
                    elif data.shape[0] > 0:
                        strip.set_data(x, np.zeros_like(data[0]), beat_annotations=all_beats, start_sec=start_sec)
                
            if data.shape[0] > 1:
                self._mini.set_data(x, data[1].copy(), beat_annotations=all_beats, start_sec=start_sec)
            else:
                self._mini.set_data(x, data[0].copy(), beat_annotations=all_beats, start_sec=start_sec)


