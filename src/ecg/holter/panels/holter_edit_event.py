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

# Visual-only helpers for this panel (palette colours from holter_helpers).
_EE_INPUT_CSS = (
    f"background:{UI_BG};color:{UI_TEXT};border:1px solid {UI_BORDER};"
    f"padding:5px;border-radius:6px;selection-background-color:#1E3A5F;"
)
_EE_FIELD_LABEL_CSS = f"color:{UI_MUTED};font-weight:bold;border:none;background:transparent;"


def _ee_tool_btn_style() -> str:
    # Same look as _style_btn but without the horizontal padding so the
    # fixed-size (30x30) tool buttons don't clip their text.
    return _style_btn() + "QPushButton{padding:0px 8px;border-radius:6px;}"


def _ee_danger_btn_style() -> str:
    return _style_btn()

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas

# 14. HOLTER EDIT EVENT PANEL
class HolterEditEventPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{UI_BG};")
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
        left = QFrame()
        left.setObjectName("eeLeftCard")
        left.setStyleSheet(_card_style("QFrame#eeLeftCard"))
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)
        ev_title = QLabel("Event List")
        ev_title.setStyleSheet(_card_title_style())
        left_layout.addWidget(ev_title)

        cols = ["Event name", "Start Time", "Chan.", "Print Len.", "Source", "Conf."]
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
        self._ev_table.cellClicked.connect(self._on_click)
        left_layout.addWidget(self._ev_table, 1)

        # Stats
        stats_title = QLabel("Summary")
        stats_title.setStyleSheet(_card_title_style())
        left_layout.addWidget(stats_title)
        stats_frame = QFrame()
        stats_frame.setObjectName("eeStatsCard")
        stats_frame.setStyleSheet(f"QFrame#eeStatsCard{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};border-radius:6px;}}")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(10, 8, 10, 8)
        sf_layout.setSpacing(4)
        self._stat_labels = {}
        for i, (key, lbl) in enumerate([
            ("atrial_ecto","Atrial Ectopic"),("rr_int","Longest RR Interval"),
            ("hr","HR"),("max_hr","HR Max"),("min_hr","HR Min"),
            ("smax_hr","Sinus Max HR"),("smin_hr","Sinus Min HR"),
            ("brady","Bradycardia"),("user_ev","User Event"),("event","Event"),
        ]):
            r, c = divmod(i, 2)
            l = QLabel(f"{lbl}:")
            l.setStyleSheet(f"color:{UI_MUTED};font-size:10px;font-weight:bold;letter-spacing:0.5px;border:none;background:transparent;")
            v = QLabel("-")
            v.setStyleSheet(f"color:{UI_TEXT};font-size:12px;font-weight:bold;border:none;background:transparent;")
            sf_layout.addWidget(l, r*2, c)
            sf_layout.addWidget(v, r*2+1, c)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 1)

        # Right: ECG strip + thumbnail
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Tool buttons row
        tool_row = QHBoxLayout()
        for icon in ["Refresh", "Menu", "Prev", "Next", "<<", ">>", "Up", "Down"]:
            btn = QPushButton(icon)
            btn.setStyleSheet(_ee_tool_btn_style())
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(30)
            btn.setMinimumWidth(36)
            tool_row.addWidget(btn)
        tool_row.addStretch()
        right_layout.addLayout(tool_row)

        # 12-lead ECG strips in a vertical scroll area (3 visible at once)
        leads_scroll = QScrollArea()
        leads_scroll.setWidgetResizable(True)
        leads_scroll.setFrameShape(QFrame.NoFrame)
        leads_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        leads_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:1px solid {UI_BORDER};border-radius:8px;}}")
        leads_host = QWidget()
        leads_host.setStyleSheet(f"background:{COL_BLACK};")
        leads_layout = QVBoxLayout(leads_host)
        leads_layout.setContentsMargins(4, 4, 4, 4)
        leads_layout.setSpacing(2)
        self._ch_strips = []
        _all_lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF",
                           "V1", "V2", "V3", "V4", "V5", "V6"]
        for name in _all_lead_names:
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;padding:2px 0;background:transparent;")
            leads_layout.addWidget(lbl)
            strip = ECGStripCanvas(height=80)
            strip.setFixedHeight(80)
            leads_layout.addWidget(strip)
            self._ch_strips.append(strip)
        leads_scroll.setWidget(leads_host)
        right_layout.addWidget(leads_scroll, 1)

        annot_box = QFrame()
        annot_box.setObjectName("eeAnnotCard")
        annot_box.setStyleSheet(_card_style("QFrame#eeAnnotCard"))
        annot_layout = QGridLayout(annot_box)
        annot_layout.setContentsMargins(10, 8, 10, 8)
        annot_layout.setSpacing(6)
        annot_title = QLabel("Beat Annotation Editor")
        annot_title.setStyleSheet(_card_title_style())
        annot_layout.addWidget(annot_title, 0, 0, 1, 2)
        self._annot_event_id = QLineEdit()
        self._annot_event_id.setPlaceholderText("beat_id / event id")
        self._annot_event_id.setStyleSheet(f"QLineEdit{{{_EE_INPUT_CSS}}} QLineEdit:focus{{border:1px solid {UI_ACCENT};}}")
        self._annot_auto = QLineEdit()
        self._annot_auto.setPlaceholderText("auto label")
        self._annot_auto.setStyleSheet(self._annot_event_id.styleSheet())
        self._annot_clin = QComboBox()
        self._annot_clin.addItems(["", "N", "S", "V", "AF", "Pause", "Tachy", "Brady", "Other"])
        self._annot_clin.setStyleSheet(f"""
            QComboBox {{
                background:{UI_BG}; color:{UI_TEXT}; border:1px solid {UI_BORDER};
                padding:5px; border-radius:6px;
            }}
            QComboBox:hover, QComboBox:focus {{ border:1px solid {UI_ACCENT}; }}
            QComboBox QAbstractItemView {{
                background:{UI_PANEL}; color:{UI_TEXT}; border:1px solid {UI_BORDER};
                selection-background-color:#1E3A5F; selection-color:{UI_TEXT}; outline:none;
            }}
        """)
        self._annot_conf = QDoubleSpinBox()
        self._annot_conf.setRange(0.0, 1.0)
        self._annot_conf.setSingleStep(0.05)
        self._annot_conf.setDecimals(2)
        self._annot_conf.setValue(0.0)
        self._annot_conf.setStyleSheet(f"QDoubleSpinBox{{{_EE_INPUT_CSS}}} QDoubleSpinBox:focus{{border:1px solid {UI_ACCENT};}}")
        self._annot_editor = QTextEdit()
        self._annot_editor.setPlaceholderText("Optional note or reviewer context")
        self._annot_editor.setFixedHeight(56)
        self._annot_editor.setStyleSheet(f"QTextEdit{{{_EE_INPUT_CSS}}} QTextEdit:focus{{border:1px solid {UI_ACCENT};}}")
        self._annot_save_btn = QPushButton("Save Annotation")
        self._annot_save_btn.setStyleSheet(_style_active_btn())
        self._annot_save_btn.setCursor(Qt.PointingHandCursor)
        self._annot_save_btn.clicked.connect(self._save_annotation)
        annot_lbl = QLabel("Beat ID:")
        annot_lbl.setStyleSheet(_EE_FIELD_LABEL_CSS)
        annot_layout.addWidget(annot_lbl, 1, 0)
        annot_layout.addWidget(self._annot_event_id, 1, 1)
        annot_lbl = QLabel("Auto label:")
        annot_lbl.setStyleSheet(_EE_FIELD_LABEL_CSS)
        annot_layout.addWidget(annot_lbl, 2, 0)
        annot_layout.addWidget(self._annot_auto, 2, 1)
        annot_lbl = QLabel("Clinician label:")
        annot_lbl.setStyleSheet(_EE_FIELD_LABEL_CSS)
        annot_layout.addWidget(annot_lbl, 3, 0)
        annot_layout.addWidget(self._annot_clin, 3, 1)
        annot_lbl = QLabel("Confidence:")
        annot_lbl.setStyleSheet(_EE_FIELD_LABEL_CSS)
        annot_layout.addWidget(annot_lbl, 4, 0)
        annot_layout.addWidget(self._annot_conf, 4, 1)
        annot_layout.addWidget(self._annot_editor, 5, 0, 1, 2)
        annot_layout.addWidget(self._annot_save_btn, 6, 0, 1, 2)
        right_layout.addWidget(annot_box)

        nav_row = QHBoxLayout()
        for lbl in ["Prev Event", "Next Event", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_ee_danger_btn_style() if lbl.startswith("Remove") else _style_btn())
            btn.setCursor(Qt.PointingHandCursor)
            nav_row.addWidget(btn)
        right_layout.addLayout(nav_row)
        layout.addWidget(right, 2)

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
            source = str(ev.get("source", "analysis"))
            conf = float(ev.get("confidence", 0.0) or 0.0)
            t_str = _format_system_time(self._session_dir, ev['timestamp'])
            for j, val in enumerate([ev['label'], t_str, "12", "7s", source, f"{conf:.2f}"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(UI_TEXT))
                if j == 0:
                    item.setData(Qt.UserRole, {
                        "beat_id": str(ev.get("beat_id", ev.get("template_label", ev.get("label", "")))),
                        "auto_label": str(ev.get("template_label", ev.get("label", ""))),
                        "clinician_label": str(ev.get("label", "")),
                        "confidence": conf,
                        "timestamp": float(ev.get("timestamp", 0.0) or 0.0),
                        "source": source,
                    })
                self._ev_table.setItem(i, j, item)
        s = summary
        for key, fmt in [("hr",f"{s.get('avg_hr',0):.0f} bpm"),
                          ("max_hr",f"{s.get('max_hr',0):.0f} bpm"),
                          ("min_hr",f"{s.get('min_hr',0):.0f} bpm"),
                          ("smax_hr",f"{s.get('max_hr',0):.0f} bpm"),
                          ("smin_hr",f"{s.get('min_hr',0):.0f} bpm"),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1"),("event","1"),
                          ("rr_int",f"{s.get('longest_rr_ms',0):.0f} ms"),
                          ("atrial_ecto","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    def _on_click(self, row, col):
        if row < len(self._events):
            self._ev_table.selectRow(row)
            self.seek_requested.emit(self._events[row]['timestamp'])
            item = self._ev_table.item(row, 0)
            if item:
                payload = item.data(Qt.UserRole) or {}
                self._selected_payload = dict(payload)
                self._annot_event_id.setText(str(payload.get("beat_id", "")))
                self._annot_auto.setText(str(payload.get("auto_label", "")))
                self._annot_clin.setCurrentText(str(payload.get("clinician_label", "")))
                self._annot_conf.setValue(float(payload.get("confidence", 0.0) or 0.0))
                self._annot_editor.setPlainText(
                    f"Source: {payload.get('source', '')}\n"
                    f"Timestamp: {_sec_to_hms(float(payload.get('timestamp', 0.0) or 0.0))}"
                )

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir or ""

    def _save_annotation(self):
        if not self._session_dir:
            QMessageBox.information(self, "Annotation", "No session directory is available for saving.")
            return
        beat_id = self._annot_event_id.text().strip()
        if not beat_id:
            QMessageBox.information(self, "Annotation", "Select an event or enter a beat ID first.")
            return
        annotation = {
            "beat_id": beat_id,
            "auto_label": self._annot_auto.text().strip(),
            "clinician_label": self._annot_clin.currentText().strip() or self._annot_auto.text().strip(),
            "confidence": float(self._annot_conf.value()),
            "edited_by": "clinician",
            "timestamp": float(self._selected_payload.get("timestamp", self._events[0].get("timestamp", 0.0) if self._events else 0.0)),
            "note": self._annot_editor.toPlainText().strip(),
        }
        try:
            append_annotation(self._session_dir, annotation)
            QMessageBox.information(self, "Annotation", "Annotation saved to session database.")
        except Exception as e:
            QMessageBox.warning(self, "Annotation", f"Could not save annotation: {e}")

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        x = np.linspace(0, N/500.0, N) if N > 0 else []
        for i, strip in enumerate(self._ch_strips):
            if i < data.shape[0] and N > 0:
                strip.set_data(x, data[i].copy())
            elif N > 0:
                strip.set_data([], [])
