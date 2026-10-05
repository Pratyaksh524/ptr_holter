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
from ..session_store import read_session_metadata
from ..holter_helpers import (
    UI_ACCENT, UI_ACCENT_HOVER, UI_BG, UI_BORDER, UI_CARD, UI_MUTED, UI_PANEL, UI_PANEL_ALT,
    UI_SUCCESS, UI_TEXT, UI_WARNING, _class_matches_filter, _find_latest_completed_session,
    _format_system_time, _get_recording_start_end_times, _metrics_duration_sec,
    _normalize_beat_class, _normalize_patient_info, _resolve_recordings_dir, _sec_to_hms,
    _style_active_btn, _style_btn, _style_filter_btn, _table_style, _template_filter_key,
)

from ..tool_engine import ECGToolEngine
from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas, HolterRRTrendCanvas, HRTrendCanvas

# 6. HOLTER REPLAY PANEL
class HolterReplayPanel(QWidget):
    playback_state_changed = pyqtSignal(bool)
    seek_requested = pyqtSignal(float)
    lead_changed   = pyqtSignal(int)
    section_requested = pyqtSignal(str)
    frame_received = pyqtSignal(object)

    def __init__(self, parent=None, duration_sec: float = 86400):
        super().__init__(parent)
        self.duration_sec = max(1, duration_sec)
        self._strip_length_sec = 10.0
        self.setStyleSheet(f"background:{COL_DARK};")
        self._replay_engine = None
        self._tool_engine = ECGToolEngine()
        self._current_replay_frame = None
        self._selected_lead_idx = 1
        self._slider_units_per_sec = 100
        self._last_slider_seek_raw = None
        self._class_filter = "all"
        self._last_metrics_list = []
        self._build_ui()
        self._magnifier_overlay = MagnifierOverlay(self)
        self._magnifier_overlay.setGeometry(self.rect())
        self._magnifier_overlay.hide()
        self._install_magnifier_dismiss_filters()

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

        self._ribbon_buttons = {}

        self._rr_mode = "RR"
        self._time_scope = "whole"

        #  "  "  48-hour session summary bar (replaces the two RR trend canvases)  "  " 
        summary_frame = QFrame()
        summary_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #172D4D, stop:0.48 #10213A, stop:1 #080D16);
                border: 1px solid #315783;
                border-radius: 8px;
            }
        """)
        summary_frame.setMinimumHeight(76)
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(12, 6, 12, 6)
        summary_layout.setSpacing(0)
        self._summary_labels = {}
        for key, title in [("duration","Duration"),("total_beats","Total Beats"),("avg_hr","Avg HR"),("max_hr","Max HR"),("min_hr","Min HR"),("pauses","Pauses"),("ve","VE Beats"),("sve","SVE Beats"),("sdnn","SDNN"),("rmssd","rMSSD")]:
            col = QVBoxLayout()
            col.setContentsMargins(12, 0, 12, 0)
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet("color:#D5E5F7;font-size:9px;font-weight:700;border:none;")
            v = QLabel("-")
            v.setStyleSheet("color:#FFFFFF;font-size:13px;font-weight:700;border:none;")
            col.addWidget(t)
            col.addWidget(v)
            self._summary_labels[key] = v
            summary_layout.addLayout(col)
        summary_layout.addStretch()
        layout.addWidget(summary_frame)

        #    HR trend mini-chart (40h-48h of data at a glance)   
        self._hr_trend_canvas = HolterRRTrendCanvas(title="Heart Rate Trend (full recording)")
        self._hr_trend_canvas.setFixedHeight(126)
        layout.addWidget(self._hr_trend_canvas)
        # Keep these as dummy attrs so update_lorenz doesn't crash
        self._rr_trend_full = self._hr_trend_canvas
        self._rr_trend_zoom = self._hr_trend_canvas

        time_row = QHBoxLayout()
        self._btn_time_whole = QPushButton("Time-whole")
        self._btn_time_share = QPushButton("Time-share")
        self._btn_goto_time = QPushButton("Goto Time")
        self._btn_rr = QPushButton("RR")
        self._btn_hr = QPushButton("HR")
        for b in [self._btn_time_whole, self._btn_time_share, self._btn_goto_time]:
            b.setFixedHeight(30)
            b.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            time_row.addWidget(b)
        time_row.addStretch()
        for b in [self._btn_rr, self._btn_hr]:
            b.setFixedHeight(28)
            b.setFixedWidth(52)
            b.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            time_row.addWidget(b)
        layout.addLayout(time_row)
        self._btn_time_whole.clicked.connect(lambda: self._set_time_scope("whole"))
        self._btn_time_share.clicked.connect(lambda: self._set_time_scope("share"))
        self._btn_goto_time.clicked.connect(self._goto_time)
        self._btn_rr.clicked.connect(lambda: self._set_rr_mode("RR"))
        self._btn_hr.clicked.connect(lambda: self._set_rr_mode("HR"))
        self._set_time_scope("whole")
        self._set_rr_mode("RR")

        # Top: left (lorenz + templates) + right (focused CH strips)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setHandleWidth(1)
        top_splitter.setStyleSheet(f"QSplitter{{background:{UI_BG};}} QSplitter::handle{{background:{UI_BORDER};}}")

        left_wrap = QFrame()
        left_wrap.setStyleSheet(f"QFrame{{background:#292D32;border:1px solid {UI_BORDER};border-radius:8px;}}")
        lw_l = QVBoxLayout(left_wrap)
        lw_l.setContentsMargins(4, 4, 4, 4)
        lw_l.setSpacing(6)
        self._lorenz_canvas = LorenzCanvas(parent=left_wrap)
        lw_l.addWidget(self._lorenz_canvas, 3)
        lorenz_filter_row = QHBoxLayout()
        lorenz_filter_row.setSpacing(4)
        lorenz_filter_row.setContentsMargins(0, 4, 0, 4)
        self._lorenz_class_btns = {}
        for key, lbl in [("all", "All"), ("N", "N"), ("S", "S"), ("V", "V"), ("P", "P"), ("AF", "AF"), ("X", "X"), ("Other", "Other")]:
            b = QPushButton(lbl)
            b.setCheckable(True)
            b.setToolTip(f"Show {lbl} beats" if key != "all" else "Show all beats")
            b.setFixedHeight(28)
            b.setStyleSheet(_style_filter_btn(active=(key == "all"), fg=COL_GREEN if key != "all" else UI_TEXT))
            b.clicked.connect(lambda checked=False, k=key: self._set_lorenz_class_filter(k))
            self._lorenz_class_btns[key] = b
            lorenz_filter_row.addWidget(b)
        lorenz_filter_row.addStretch()
        lw_l.addLayout(lorenz_filter_row)
        self._set_lorenz_class_filter("all")

        self._template_thumbs = []
        top_splitter.addWidget(left_wrap)

        ecg_right = QFrame()
        ecg_right.setStyleSheet(f"QFrame{{background:#292D32;border:1px solid {UI_BORDER};border-radius:8px;}}")
        ecg_right_layout = QVBoxLayout(ecg_right)
        ecg_right_layout.setContentsMargins(4, 4, 4, 4)
        ecg_right_layout.setSpacing(2)

        #  "  "  12-lead scrollable grid (1 column, 12 rows)  "  " 
        leads_scroll = QScrollArea()
        leads_scroll.setWidgetResizable(True)
        leads_scroll.setFrameShape(QFrame.NoFrame)
        leads_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        leads_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        leads_scroll.setStyleSheet("QScrollArea{background:#25292E;border:none;}")
        leads_container = QWidget()
        leads_container.setStyleSheet("background:#25292E;")
        leads_vbox = QVBoxLayout(leads_container)
        leads_vbox.setContentsMargins(2, 2, 2, 2)
        leads_vbox.setSpacing(2)

        self._lead_names_ordered = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        self._lead_strips = {}   # lead_name -> ECGStripCanvas
        self._ch_strips = []     # backward-compat list for gain/speed handlers
        self._detected_r_peaks = []  # Store R-peak timestamps for vertical line overlay
        
        for lead in self._lead_names_ordered:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            lbl = QLabel(lead)
            lbl.setFixedWidth(34)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
            # Height 60px makes them compact enough to fit well, but scrollable if needed
            # Replay mode: disable ALL coloring (disable_all_coloring=True) - no arrhythmia colors, just green
            strip = ECGStripCanvas(height=60, color="#00FF00", pen_width=0.9, lead_name=lead, show_annotations=False, disable_all_coloring=True)
            strip.set_gain(1.0)
            self._lead_strips[lead] = strip
            self._ch_strips.append(strip)
            row.addWidget(lbl)
            row.addWidget(strip, 1)
            leads_vbox.addLayout(row)

        leads_scroll.setWidget(leads_container)
        ecg_right_layout.addWidget(leads_scroll, 3)

        # Full-width rhythm strip (Lead II) at bottom
        rhythm_row = QHBoxLayout()
        rhythm_row.setContentsMargins(0, 0, 0, 0)
        rhythm_row.setSpacing(4)
        rhythm_lbl = QLabel("II")
        self._mini_lead_lbl = rhythm_lbl
        rhythm_lbl.setFixedWidth(34)
        rhythm_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rhythm_lbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
        # Replay mode: disable ALL coloring
        self._mini_strip = ECGStripCanvas(height=60, color="#00AA00", pen_width=0.9, show_annotations=False, disable_all_coloring=True)
        rhythm_row.addWidget(rhythm_lbl)
        rhythm_row.addWidget(self._mini_strip, 1)
        ecg_right_layout.addLayout(rhythm_row)

        top_splitter.addWidget(ecg_right)


        ov_frame = QFrame()
        ov_frame.setStyleSheet("""
            QFrame {
                background: #0A0D10;
                border: 1px solid #1E2832;
                border-radius: 4px;
            }
        """)
        ov_layout = QVBoxLayout(ov_frame)
        ov_layout.setContentsMargins(4, 4, 4, 4)
        ov_layout.setSpacing(0)
        ov_title = QLabel("Overview")
        ov_title.setStyleSheet("color: #FFFFFF; font-weight: bold; font-size: 13px; padding: 5px 8px; background: #131A22; border: 1px solid #232D38; border-bottom: none; border-top-left-radius: 3px; border-top-right-radius: 3px;")
        ov_layout.addWidget(ov_title)

        self._overview_table = QTableWidget(0, 2)
        self._overview_table.setHorizontalHeaderLabels(["Name", "Value"])
        self._overview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._overview_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._overview_table.verticalHeader().setVisible(False)
        self._overview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._overview_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._overview_table.setFocusPolicy(Qt.NoFocus)
        self._overview_table.setShowGrid(True)
        self._overview_table.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._overview_table.horizontalHeader().setFixedHeight(28)

        self._overview_table.setStyleSheet("""
            QTableWidget {
                background-color: #0A0E13;
                color: #FFFFFF;
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
        ov_layout.addWidget(self._overview_table, 1)
        top_splitter.addWidget(ov_frame)
        top_splitter.setSizes([430, 980, 280])
        layout.addWidget(top_splitter, 2)

        # Scrub slider row
        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(2, 0, 2, 0)
        slider_row.setSpacing(8)
        self._time_start_label = QLabel("00:00:00")
        self._time_start_label.setStyleSheet(f"color:{COL_TIMESTAMP};font-family:monospace;font-size:12px;border:none;")
        slider_row.addWidget(self._time_start_label)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._slider_sec_to_value(self.duration_sec))
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal{{height:8px;background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}
            QSlider::handle:horizontal{{background:{COL_GREEN};border:1px solid {COL_WHITE};border-radius:9px;
                width:18px;height:18px;margin:-6px 0;}}
            QSlider::sub-page:horizontal{{background:{COL_GREEN_DRK};border-radius:4px;}}
        """)
        self._slider.setTracking(True)
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderMoved.connect(self._on_slider)
        slider_row.addWidget(self._slider, 1)
        self._pos_label = QLabel("00:00:00")
        self._pos_label.setStyleSheet(f"color:{COL_TIMESTAMP};font-family:monospace;font-size:14px;font-weight:bold;"
                                      f"background:{COL_BLACK};padding:4px;border:1px solid {COL_GREEN_DRK};border-radius:4px;border:none;")
        slider_row.addWidget(self._pos_label)
        layout.addLayout(slider_row)

        # Transport + controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(7)

        self._play_btn = QPushButton("Play")
        self._play_btn.setStyleSheet(_style_btn())
        self._play_btn.setFixedHeight(30)
        self._play_btn.setMinimumWidth(100)
        self._play_btn.clicked.connect(self._toggle_playback)
        ctrl_row.addWidget(self._play_btn)

        for speed_lbl in ["0.5x", "1x", "2x", "4x"]:
            btn = QPushButton(speed_lbl)
            btn.setStyleSheet(_style_btn(COL_DARK, COL_GREEN, COL_GREEN_DRK))
            btn.setFixedHeight(30)
            btn.setMinimumWidth(56)
            btn.clicked.connect(lambda _, s=speed_lbl: self._set_speed(s))
            ctrl_row.addWidget(btn)
        sep = QLabel("|")
        sep.setStyleSheet(f"color:{COL_GREEN_DRK};")
        ctrl_row.addWidget(sep)

        lbl_lead = QLabel("Lead:")
        lbl_lead.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;border:none;font-size:13px;")
        ctrl_row.addWidget(lbl_lead)
        self._lead_combo = QComboBox()
        self._lead_combo.addItems(["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])
        self._lead_combo.setCurrentIndex(1)
        self._lead_combo.setFixedWidth(70)
        self._lead_combo.setStyleSheet(f"background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                                       f"padding:4px;border-radius:4px;font-weight:bold;")
        self._lead_combo.currentIndexChanged.connect(self._on_lead_changed)
        ctrl_row.addWidget(self._lead_combo)

        ctrl_row.addSpacing(22)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Bottom toolbar (like reference image)
        toolbar = QFrame()
        toolbar.setStyleSheet(f"""
            QFrame {{
                background: #090B0D;
                border: 1px solid #3A4148;
                border-radius: 8px;
            }}
        """)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 4, 5, 4)
        toolbar_layout.setSpacing(5)
        self._tool_btns = {}
        tool_min_widths = {
            "Patient information": 140,
            "Full Disc.": 90,
            "Goto Template": 114,
            "Measuring Ruler": 120,
            "Parallel Ruler": 112,
            "Magnifying Glass": 126,
            "Gain Settings": 110,
            "Paper speed:25mm/s": 144,
            "Add Event(space)": 130,
            "Adjust strip position": 146,
            "Strip Length:10s": 120,
        }
        tool_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4A5158, stop:0.5 #292E33, stop:1 #15181B);
                color: #FFFFFF;
                border: 1px solid #68727C;
                border-bottom: 2px solid #101214;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #69747E, stop:1 #353C43);
                color: #FFFFFF;
                border: 1px solid #AAB4BD;
            }
            QPushButton:pressed, QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #171A1D, stop:1 #59636C);
                color: #FFFFFF;
                border: 1px solid #D0D7DD;
                padding-top: 6px;
                padding-bottom: 4px;
            }
        """
        for tool in ["Patient information", "Full Disc.", "Goto Template"]:
            tbtn = QPushButton(tool)
            tbtn.setStyleSheet(tool_button_style)
            tbtn.setMinimumHeight(32)
            tbtn.setMinimumWidth(tool_min_widths.get(tool, 110))
            tbtn.clicked.connect(lambda _, t=tool, b=tbtn: self._set_tool_mode(t, b))
            toolbar_layout.addWidget(tbtn)
            self._tool_btns[tool] = tbtn
        for tool in ["Measuring Ruler", "Parallel Ruler", "Magnifying Glass", "Gain Settings",
                     "Paper speed:25mm/s", "Add Event(space)", "Adjust strip position", "Strip Length:10s"]:
            tbtn = QPushButton(tool)
            tbtn.setStyleSheet(tool_button_style)
            tbtn.setMinimumHeight(32)
            tbtn.setMinimumWidth(tool_min_widths.get(tool, 110))
            if tool == "Gain Settings":
                tbtn.setText("Gain: 10mm/mV")
            tbtn.clicked.connect(lambda _, t=tool, b=tbtn: self._set_tool_mode(t, b))
            toolbar_layout.addWidget(tbtn)
            self._tool_btns[tool] = tbtn
        self._tool_btns["Gain Settings"].setToolTip(
            "Cycle gain (5/10/20/40 mm/mV equivalent) to improve waveform visibility."
        )
        toolbar_layout.addStretch()
        layout.addWidget(toolbar)



    def _install_magnifier_dismiss_filters(self):
        """Dismiss the magnifier on any non-wave click inside the replay panel."""
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

    def _clear_magnifier_if_needed(self, event) -> bool:
        if event.type() != QEvent.MouseButtonPress or event.button() != Qt.LeftButton:
            return False
        overlay = getattr(self, "_magnifier_overlay", None)
        if overlay is None or not getattr(overlay, "_visible", False):
            return False
        source = self.sender()
        if isinstance(source, ECGStripCanvas):
            return False
        overlay.clear_focus()
        for strip in getattr(self, "_ch_strips", []):
            if hasattr(strip, "_magnify_locked"):
                strip._magnify_locked = False
                strip._magnify_pos = None
                strip.update()
        mini_strip = getattr(self, "_mini_strip", None)
        if mini_strip is not None and hasattr(mini_strip, "_magnify_locked"):
            mini_strip._magnify_locked = False
            mini_strip._magnify_pos = None
            mini_strip.update()
        return False

    def eventFilter(self, obj, event):
        if self._clear_magnifier_if_needed(event):
            return False
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.setGeometry(self.rect())

    def set_magnifier_focus(self, source_widget, payload: dict, focus_pos: QPoint):
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.setGeometry(self.rect())
            self._magnifier_overlay.set_focus(source_widget, payload, focus_pos)

    def clear_magnifier_focus(self, source_widget=None):
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.clear_focus(source_widget)

    def clear_strip_tools(self, source_widget=None):
        """Clear transient ruler/caliper overlays and any locked magnifier."""
        self.clear_magnifier_focus(source_widget)
        for strip in getattr(self, "_ch_strips", []):
            if hasattr(strip, "clear_interaction"):
                strip.clear_interaction()
        mini_strip = getattr(self, "_mini_strip", None)
        if mini_strip is not None and hasattr(mini_strip, "clear_interaction"):
            mini_strip.clear_interaction()

    def _set_tool_mode(self, tool_name: str, btn: QPushButton = None):
        """
        Main tool mode dispatcher - delegates to HolterToolHandlers.
        Keeps holter_ui.py clean by moving implementations to holter_full_disclosure.py.
        """
        from ..holter_full_disclosure import HolterToolHandlers
        
        # Patient Information
        if "Patient information" in tool_name:
            HolterToolHandlers.handle_patient_information(self)
            return
        
        # Full Disclosure
        if "Full Disc." in tool_name:
            HolterToolHandlers.handle_full_disclosure(self)
            return
        
        # Goto Template (Info Dialog)
        if "Goto Template" in tool_name:
            HolterToolHandlers.handle_goto_template(self)
            return
        
        # Add Event
        if "Add Event" in tool_name or "space" in tool_name.lower():
            HolterToolHandlers.handle_add_event(self)
            return

        # Tool name conversions
        if "Measuring Ruler" in tool_name:
            tool_name = TOOL_RULER
        elif "Parallel Ruler" in tool_name:
            tool_name = TOOL_CALIPER
        elif "Magnifying Glass" in tool_name:
            tool_name = TOOL_MAGNIFY

        # Gain Settings
        if "Gain Settings" in tool_name:
            HolterToolHandlers.handle_gain_settings(self, btn)
            return
        
        # Paper Speed
        elif "Paper speed" in tool_name:
            HolterToolHandlers.handle_paper_speed(self, btn)
            return
        
        # Strip Length
        elif "Strip Length" in tool_name:
            HolterToolHandlers.handle_strip_length(self, btn)
            return

        # Apply tool mode to strips (Ruler, Caliper, Magnify)
        HolterToolHandlers.apply_tool_mode_to_strips(self, tool_name)

    def _set_time_scope(self, scope: str):
        self._time_scope = scope
        self._btn_time_whole.setStyleSheet(_style_active_btn() if scope == "whole" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._btn_time_share.setStyleSheet(_style_active_btn() if scope == "share" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._strip_length_sec = 10.0 if scope == "whole" else 3.6
        if getattr(self, "_replay_engine", None):
            try:
                self._replay_engine.set_window_length(self._strip_length_sec)
            except Exception:
                pass
        self._refresh_wave_window()

    def _refresh_wave_window(self):
        if getattr(self, '_replay_engine', None):
            try:
                current_pos = float(self._replay_engine.current_position())
                self._replay_engine.seek(current_pos)
                window_sec = float(self._strip_length_sec)
                data = self._replay_engine.get_all_leads_data(window_sec=window_sec)
                beat_annotations = self._replay_engine.get_beat_annotations(window_sec=window_sec)
                start_sec = self._replay_engine.current_position() - window_sec / 2
                self._replay_panel.set_replay_frame(data, beat_annotations=beat_annotations, start_sec=start_sec)
            except Exception:
                pass
        elif getattr(self, '_live_source', None) is not None and hasattr(self, '_replay_panel'):
            try:
                raw = getattr(self._live_source, 'data', None)
                if raw is not None and hasattr(raw, '__len__') and len(raw) > 0:
                    import numpy as _np
                    n_samp = max(len(raw[i]) for i in range(min(12, len(raw))))
                    arr = _np.full((12, n_samp), 2048.0)
                    for i in range(min(12, len(raw))):
                        ch = _np.asarray(raw[i], dtype=float)
                        arr[i, :len(ch)] = ch
                    self._replay_panel.set_replay_frame(arr)
            except Exception:
                pass

    def _set_rr_mode(self, mode: str):
        self._rr_mode = mode
        self._btn_rr.setStyleSheet(_style_active_btn() if mode == "RR" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._btn_hr.setStyleSheet(_style_active_btn() if mode == "HR" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        if hasattr(self, '_hr_trend_canvas'):
            self._hr_trend_canvas.set_mode(mode)
        if self._last_metrics_list:
            self.update_lorenz(self._last_metrics_list)

    def _build_info_table(self, rows):
        table = QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Field", "Value"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.setStyleSheet(_table_style())
        table.setAlternatingRowColors(True)
        for row, (label, value) in enumerate(rows):
            label_item = QTableWidgetItem(str(label))
            label_item.setForeground(QColor(UI_TEXT))
            value_item = QTableWidgetItem(str(value))
            value_item.setForeground(QColor(COL_WHITE))
            table.setItem(row, 0, label_item)
            table.setItem(row, 1, value_item)
        table.resizeRowsToContents()
        return table

    def _show_patient_information(self):
        summary = dict(getattr(self, "_summary", {}) or {})
        engine = getattr(self, "_replay_engine", None)
        session_dir = summary.get("session_dir", "") or ""
        if not session_dir and engine is not None:
            session_dir = os.path.dirname(getattr(engine, "ecgh_path", "") or "")
        metadata = read_session_metadata(session_dir) if session_dir else {}
        if isinstance(metadata, dict):
            meta_summary = metadata.get("summary")
            if isinstance(meta_summary, dict):
                merged = dict(meta_summary)
                merged.update(summary)
                summary = merged
            meta_patient = metadata.get("patient_info")
            if isinstance(meta_patient, dict) and meta_patient:
                summary["patient_info"] = dict(meta_patient)

        patient_info = _normalize_patient_info(
            summary.get("patient_info") or getattr(engine, "patient_info", {}) or {}
        )

        if not summary and not patient_info:
            QMessageBox.information(self, "Patient information", "No recording is loaded.")
            return

        session_name = os.path.basename(session_dir) if session_dir else "Current session"
        record_time = session_name
        parts = session_name.split("_", 3)
        if len(parts) >= 2:
            record_time = "_".join(parts[:2]).replace("_", " ")

        def fmt_num(value, suffix="", digits=1):
            try:
                return f"{float(value):.{digits}f}{suffix}"
            except Exception:
                return "-" if value in (None, "") else f"{value}{suffix}"

        def fmt_duration(seconds):
            try:
                sec = max(0, int(float(seconds)))
            except Exception:
                return "-"
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            if h > 0:
                return f"{h}h {m:02d}m"
            if m > 0:
                return f"{m}m {s:02d}s"
            return f"{s}s"

        patient_rows = [
            ("Patient Name", patient_info.get("patient_name") or patient_info.get("name") or "-"),
            ("Age", patient_info.get("age", "-")),
            ("Gender", patient_info.get("gender") or patient_info.get("sex") or "-"),
            ("Doctor", patient_info.get("doctor") or "-"),
            ("Email", patient_info.get("email") or "-"),
            ("Phone", patient_info.get("doctor_mobile") or patient_info.get("phone") or "-"),
            ("Organisation", patient_info.get("org") or patient_info.get("Org.") or "-"),
            ("Study Duration", fmt_duration(summary.get("duration_sec", 0))),
        ]

        dlg = QDialog(self)
        dlg.setWindowTitle("Patient Information")
        dlg.setMinimumSize(760, 620)
        dlg.setStyleSheet(f"""
            QDialog {{
                background: {UI_BG};
                color: {UI_TEXT};
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
            QGroupBox {{
                color: {UI_TEXT};
                font-weight: 700;
                border: 1px solid {UI_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                background: {COL_BLACK};
            }}
            QHeaderView::section {{
                background: {UI_PANEL_ALT};
                color: {UI_TEXT};
                border: 1px solid {UI_BORDER};
                padding: 6px;
                font-weight: 700;
            }}
        """)
        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QLabel("Patient Information")
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{UI_TEXT};")
        outer.addWidget(title)

        subtitle = QLabel(f"Selected recording: {session_name}")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:{UI_MUTED};font-size:12px;")
        outer.addWidget(subtitle)

        patient_box = QGroupBox("Patient Details")
        patient_box.setStyleSheet(f"""
            QGroupBox {{
                color: {UI_TEXT};
                font-weight: 700;
                border: 1px solid {UI_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                background: {UI_PANEL};
            }}
        """)
        patient_layout = QVBoxLayout(patient_box)
        patient_layout.setContentsMargins(10, 16, 10, 10)
        patient_layout.addWidget(self._build_info_table(patient_rows))
        outer.addWidget(patient_box)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedSize(94, 32)
        close_btn.setStyleSheet(_style_btn())
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        dlg.exec_()

    def _goto_time(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Go to Time")
        dlg.setFixedSize(420, 160)
        dlg.setStyleSheet(f"""
            QDialog {{
                background: {UI_BG};
                border: 1px solid {UI_BORDER};
                border-radius: 10px;
            }}
            QLabel {{
                color: {UI_TEXT};
                font-size: 13px;
                font-weight: bold;
                border: none;
                background: transparent;
            }}
            QLineEdit {{
                background: {UI_PANEL};
                color: {UI_TEXT};
                border: 1px solid {UI_BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                selection-background-color: {UI_ACCENT};
            }}
            QLineEdit:focus {{
                border: 1px solid {UI_ACCENT};
            }}
        """)
        v_layout = QVBoxLayout(dlg)
        v_layout.setContentsMargins(20, 18, 20, 18)
        v_layout.setSpacing(12)

        lbl = QLabel("Enter time (HH:MM:SS or seconds):")
        lbl.setStyleSheet(f"color:{UI_MUTED};font-size:12px;font-weight:normal;border:none;background:transparent;")
        v_layout.addWidget(lbl)

        inp = QLineEdit()
        inp.setPlaceholderText("e.g.  01:23:45  or  83")
        inp.setClearButtonEnabled(True)
        v_layout.addWidget(inp)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()
        ok_btn = QPushButton("Go")
        ok_btn.setFixedSize(90, 34)
        ok_btn.setStyleSheet(_style_active_btn())
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedSize(90, 34)
        cancel_btn.setStyleSheet(_style_btn())
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        v_layout.addLayout(btn_row)

        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        inp.returnPressed.connect(dlg.accept)

        if dlg.exec_() != QDialog.Accepted:
            return
        t = inp.text().strip()
        if not t:
            return
        sec = 0.0
        try:
            if ":" in t:
                parts = [int(p) for p in t.split(":")]
                if len(parts) == 3:
                    sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
                elif len(parts) == 2:
                    sec = parts[0] * 60 + parts[1]
                else:
                    sec = float(t)
            else:
                sec = float(t)
        except Exception:
            warn = QDialog(self)
            warn.setWindowTitle("Invalid Time")
            warn.setFixedSize(340, 120)
            warn.setStyleSheet(f"QDialog{{background:{UI_BG};border:1px solid {UI_BORDER};border-radius:8px;}} QLabel{{color:{UI_TEXT};font-size:12px;border:none;background:transparent;}}")
            wl = QVBoxLayout(warn)
            wl.setContentsMargins(18, 16, 18, 16)
            wl.setSpacing(12)
            wl.addWidget(QLabel("Use HH:MM:SS, MM:SS, or plain seconds."))
            wb = QPushButton("OK")
            wb.setStyleSheet(_style_btn())
            wb.clicked.connect(warn.accept)
            wr = QHBoxLayout()
            wr.addStretch()
            wr.addWidget(wb)
            wl.addLayout(wr)
            warn.exec_()
            return
        sec = max(0.0, min(sec, float(self.duration_sec)))
        self.seek_requested.emit(sec)

    def update_summary(self, summary: dict):
        self._summary = dict(summary or {})
        self._patient_info = _normalize_patient_info(self._summary.get('patient_info') or getattr(self._replay_engine, 'patient_info', {}) or {})
        if hasattr(self, '_summary_labels'):
            def set_lbl(k, v):
                if k in self._summary_labels: self._summary_labels[k].setText(str(v))
            
            dur = summary.get('duration_sec', 0)
            h = int(dur // 3600)
            m = int((dur % 3600) // 60)
            set_lbl("duration", f"{h:02d}h {m:02d}m")
            set_lbl("total_beats", summary.get('total_beats', 0))
            set_lbl("avg_hr", f"{summary.get('avg_hr', 0)} bpm")
            set_lbl("max_hr", f"{summary.get('max_hr', 0)} bpm")
            set_lbl("min_hr", f"{summary.get('min_hr', 0)} bpm")
            set_lbl("pauses", summary.get('pauses', 0))
            set_lbl("ve", summary.get('ve_beats', 0))
            set_lbl("sve", summary.get('sve_beats', 0))
            set_lbl("sdnn", f"{summary.get('sdnn', 0)} ms")
            set_lbl("rmssd", f"{summary.get('rmssd', 0)} ms")

    def _slider_sec_to_value(self, seconds: float) -> int:
        units = max(1, int(getattr(self, "_slider_units_per_sec", 100)))
        return int(round(max(0.0, float(seconds)) * units))

    def _slider_value_to_sec(self, value: int) -> float:
        units = max(1, int(getattr(self, "_slider_units_per_sec", 100)))
        return max(0.0, float(value) / float(units))

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._slider.setRange(0, self._slider_sec_to_value(engine.duration_sec))
        engine.set_position_callback(self._on_position_update)
        engine.set_window_length(getattr(self, "_strip_length_sec", 10.0))
        
        # Wire up data callback safely for thread-safe playback updates
        try:
            self.frame_received.disconnect()
        except TypeError:
            pass
        self.frame_received.connect(self.set_replay_frame)
        engine.set_data_callback(lambda data: self.frame_received.emit(data))

    def _on_slider(self, value):
        raw = int(value)
        sec = self._slider_value_to_sec(raw)
        self._pos_label.setText(_sec_to_hms(sec))
        if self._last_slider_seek_raw == raw:
            return
        self._last_slider_seek_raw = raw
        self.seek_requested.emit(float(sec))

    def _on_lead_changed(self, idx: int):
        self._selected_lead_idx = max(0, int(idx))
        frame = getattr(self, "_current_replay_frame", None)
        if frame is not None:
            try:
                self.set_replay_frame(frame)
            except Exception:
                pass
        try:
            self.lead_changed.emit(int(idx))
        except Exception:
            pass

    def _on_position_update(self, current_sec, duration_sec):
        self._last_slider_seek_raw = self._slider_sec_to_value(current_sec)
        self._slider.blockSignals(True)
        self._slider.setValue(self._slider_sec_to_value(current_sec))
        self._slider.blockSignals(False)
        self._pos_label.setText(_sec_to_hms(current_sec))

    def _toggle_playback(self):
        if not self._replay_engine: return
        if self._replay_engine.is_playing():
            self._replay_engine.pause()
            self._play_btn.setText("Play")
        else:
            self._replay_engine.play()
            self._play_btn.setText("Pause")

    def _set_speed(self, text: str):
        if self._replay_engine:
            try:
                self._replay_engine.set_speed(float(text.replace("x", "")))
            except Exception:
                pass

    def _set_lorenz_class_filter(self, key: str):
        key = str(key or "all")
        if key not in self._lorenz_class_btns:
            key = "all"
        self._class_filter = key
        for btn_key, btn in self._lorenz_class_btns.items():
            is_active = (btn_key == key)
            btn.setChecked(is_active)
            btn.setStyleSheet(_style_filter_btn(active=is_active, fg=COL_GREEN if not is_active else UI_TEXT))
        if self._last_metrics_list:
            self.update_lorenz(self._last_metrics_list)

    def _jump_event(self, ev_type: str, direction: str):
        if self._replay_engine:
            t = self._replay_engine.seek_to_event(ev_type, direction)
            self.seek_requested.emit(t)

    def update_lorenz(self, metrics_list: list):
        """Update the Lorenz/scatter plot from all individual RR data."""
        self._last_metrics_list = list(metrics_list or [])
        rr_all = []
        rr_points = []
        for m in metrics_list:
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
                        rr_all.append(rr)
                        rr_points.append((t0 + i * step, rr, beat_class))
                    continue  # skip fallback if list data available
            # Fallback: use single rr_ms value per chunk
            rr_val = float(m.get('rr_ms', 0) or 0)
            if rr_val > 200:
                rr_all.append(rr_val)
                beat_class = _normalize_beat_class(
                    m.get("label")
                    or m.get("template_label")
                    or m.get("arrhythmia")
                    or (m.get("arrhythmias") or [None])[0]
                )
                rr_points.append((t0, rr_val, beat_class))

        rr_n = [r for r in rr_all if r > 200]
        rr_n_classes = [p[2] for p in rr_points if p[1] > 200]
        filtered_points = [p for p in rr_points if _class_matches_filter(p[2], self._class_filter)]
        filtered_rr = [p[1] for p in filtered_points if p[1] > 200]
        filtered_classes = [p[2] for p in filtered_points if p[1] > 200]
        plot_rr = filtered_rr if len(filtered_rr) >= 2 else rr_n
        plot_classes = filtered_classes if len(filtered_rr) >= 2 else rr_n_classes
        if len(plot_rr) >= 2:
            rr_x = plot_rr[:-1]
            rr_y = plot_rr[1:]
            plot_beat_classes = plot_classes[:-1]  # use the first beat's class for each point
            lo = float(np.percentile(plot_rr, 5))
            hi = float(np.percentile(plot_rr, 95))
            if hi - lo < 250:
                center = float(np.median(plot_rr))
                lo = center - 500.0
                hi = center + 500.0
            lo = max(0.0, lo - 50.0)
            hi = hi + 50.0
            self._lorenz_canvas.set_data(rr_x, rr_y, x_range=(lo, hi), y_range=(lo, hi), beat_classes=plot_beat_classes)
        else:
            self._lorenz_canvas.set_data([], [])
        if hasattr(self, "_rr_trend_full"):
            trend_points = [(t, rr) for t, rr, _cls in filtered_points] if len(filtered_points) >= 2 else [(t, rr) for t, rr, _cls in rr_points]
            if self._rr_mode == "HR":
                # Keep the trend consistent with the summary card.  Raw RR
                # data can contain a short, rejected interval (for example
                # 200 ms -> 300 bpm), while the summary's max HR is based on
                # cleaned per-beat values. Do not plot those rejected values.
                valid_hr_max = max(
                    (float(m.get("hr_max", 0.0) or 0.0) for m in metrics_list),
                    default=0.0,
                )
                hr_points = []
                for t, rr in trend_points:
                    if rr <= 0:
                        continue
                    hr = 60000.0 / rr
                    if valid_hr_max <= 0.0 or hr <= valid_hr_max + 0.5:
                        hr_points.append((t, hr))
                trend_points = hr_points
            self._rr_trend_full.set_points(trend_points)
            # Only update zoom canvas separately if it's a different widget (OVERVIEW panel has two separate canvases;
            # the REPLAY panel aliases both to the same _hr_trend_canvas -- calling set_points on it twice would
            # overwrite the full data with the recent subset)
            if hasattr(self, "_rr_trend_zoom") and self._rr_trend_zoom is not self._rr_trend_full:
                recent = trend_points[-1200:] if len(trend_points) > 1200 else trend_points
                self._rr_trend_zoom.set_points(recent)
        self._update_overview_table(metrics_list, rr_n)
            
    def set_replay_frame(self, data, beat_annotations=None, start_sec=0.0):
        """Update all 12 lead strips and compute Lorenz from data when no RR metrics."""
        if data is None or data.shape[0] < 12:
            return

        self._current_replay_frame = data
        strip_len = int(max(1, getattr(self, "_strip_length_sec", 10.0)) * 500)
        if data.shape[1] > strip_len:
            data = data[:, -strip_len:]

        N = data.shape[1]
        x = np.linspace(0, N / 500.0, N) if N > 0 else []
        if N <= 0:
            return

        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

        # Ensure data has 12 channels
        if data.shape[0] < 12:
            new_data = np.zeros((12, data.shape[1]), dtype=data.dtype)
            new_data[:data.shape[0], :] = data
            for i in range(data.shape[0], 12):
                new_data[i, :] = 2048.0
            data = new_data

        # --- Force mathematical derivation of augmented limb leads ---
        # Einthoven's Law dictates these leads are perfectly locked to I and II.
        # We unconditionally derive them to overwrite any floating hardware noise.
        I_lead = data[0]
        II_lead = data[1]
        
        data[2] = (II_lead - 2048.0) - (I_lead - 2048.0) + 2048.0  # III = II - I
        # User wants aVR mapped to 0 to -4096. We synthesize it inverted and centered at -2048.
        data[3] = -((I_lead - 2048.0) + (II_lead - 2048.0)) / 2.0 - 2048.0
        data[4] = (I_lead - 2048.0) - (II_lead - 2048.0) / 2.0 + 2048.0  # aVL = I - II/2
        data[5] = (II_lead - 2048.0) - (I_lead - 2048.0) / 2.0 + 2048.0  # aVF = II - I/2

        # --- Feed all 12 lead strips ---
        lead_strips = getattr(self, "_lead_strips", {})
        for idx, lead in enumerate(lead_names):
            if idx < data.shape[0] and lead in lead_strips:
                lead_strips[lead].set_data(x, data[idx].copy(), beat_annotations=beat_annotations, start_sec=start_sec)

        # Mini strip follows the selected lead from the dropdown
        if hasattr(self, "_mini_strip") and data.shape[0] > 0:
            lead_idx = max(0, min(int(getattr(self, "_selected_lead_idx", 1)), data.shape[0] - 1))
            selected_lead = self._lead_combo.currentText() if hasattr(self, "_lead_combo") else lead_names[lead_idx]
            self._mini_strip.set_data(x, data[lead_idx].copy(), beat_annotations=beat_annotations, start_sec=start_sec)
            try:
                self._mini_strip.lead_name = selected_lead
            except Exception:
                pass
            if hasattr(self, "_mini_lead_lbl"):
                self._mini_lead_lbl.setText(selected_lead)

        # Template thumbnails (use first 4 leads: I, II, III, aVR)
        for i, ts in enumerate(getattr(self, "_template_thumbs", [])):
            src = min(i, data.shape[0] - 1)
            center = N // 2
            w = min(220, max(80, N // 6))
            a = max(0, center - w // 2)
            b = min(N, center + w // 2)
            seg = data[src, a:b] if b > a else data[src, :]
            tx = np.linspace(0, len(seg) / 500.0, len(seg)) if len(seg) > 0 else []
            ts.set_data(tx, seg.copy() if len(seg) > 0 else data[src].copy())

        # --- On-the-fly Lorenz from data when Lorenz canvas has no data ---
        lorenz = getattr(self, "_lorenz_canvas", None)
        if lorenz is not None and (not lorenz._x):  # no RR data loaded from metrics
            self._compute_lorenz_from_signal(data, N)

    def _compute_lorenz_from_signal(self, data: np.ndarray, N: int):
        """Detect R-peaks in Lead II and populate the Lorenz scatter from the raw ECG data."""
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 500.0
            lead_ii = np.asarray(data[1], dtype=float) if data.shape[0] > 1 else np.asarray(data[0], dtype=float)
            # Bandpass 5-20 Hz to isolate QRS
            nyq = fs / 2.0
            b, a = butter(2, [5.0 / nyq, 20.0 / nyq], btype='band')
            filtered = filtfilt(b, a, lead_ii)
            squared = filtered ** 2
            win = max(1, int(0.15 * fs))
            mwa = np.convolve(squared, np.ones(win) / win, mode='same')
            threshold = max(np.mean(mwa) * 0.5, 1e-6)
            min_dist = max(1, int(0.3 * fs))
            peaks, _ = find_peaks(mwa, height=threshold, distance=min_dist)
            if len(peaks) >= 3:
                rr_ms = np.diff(peaks) / fs * 1000.0
                rr_valid = [float(r) for r in rr_ms if 250 < r < 2500]
                if len(rr_valid) >= 2:
                    rr_x = rr_valid[:-1]
                    rr_y = rr_valid[1:]
                    self._lorenz_canvas.set_data(rr_x, rr_y)
                    # Only fill the trend chart from live signal if full-recording metrics haven't already populated it
                    if hasattr(self, "_rr_trend_full") and not getattr(self._rr_trend_full, '_points', []):
                        rr_points = [(i * 0.5, rr) for i, rr in enumerate(rr_valid)]
                        self._rr_trend_full.set_points(rr_points)
                        self._rr_trend_zoom.set_points(rr_points[-400:] if len(rr_points) > 400 else rr_points)
        except Exception:
            pass

    def apply_record_settings(self, settings: dict):
        """Apply lead gains and invert states from Record Settings to all lead strips."""
        # Persist so Full Disclosure dialog can read these settings when it opens
        self._last_record_settings = settings
        lead_gains = settings.get("lead_gains", {})
        lead_inverts = settings.get("lead_inverts", {})
        for lead, strip in self._lead_strips.items():
            if lead in lead_gains:
                # Map gain mm/mV (e.g. 5, 10, 20) to scaling factor (10 mm/mV is 1.0)
                g_val = lead_gains[lead]
                scale = float(g_val) / 10.0
                strip.set_gain(scale)
            if lead in lead_inverts:
                strip.set_invert(lead_inverts[lead])
        if hasattr(self, "_mini_strip"):
            selected_lead = self._lead_combo.currentText() if hasattr(self, "_lead_combo") else "II"
            if selected_lead in lead_gains:
                self._mini_strip.set_gain(float(lead_gains[selected_lead]) / 10.0)
            if selected_lead in lead_inverts:
                self._mini_strip.set_invert(lead_inverts[selected_lead])
        # Update toolbar Gain button text if uniform gain across leads
        if hasattr(self, "_tool_btns") and "Gain Settings" in self._tool_btns:
            g_vals = set(lead_gains.values()) if lead_gains else set()
            if len(g_vals) == 1:
                val = int(list(g_vals)[0])
                self._tool_btns["Gain Settings"].setText(f"Gain: {val}mm/mV")
                try:
                    from ecg.holter.theme import GAINS
                    if val in GAINS:
                        self._curr_gain_idx = GAINS.index(val)
                except Exception:
                    pass
            else:
                self._tool_btns["Gain Settings"].setText("Gain: 10mm/mV")
                self._curr_gain_idx = 1

    def _update_overview_table(self, metrics_list: list, rr_n: list):
        if not hasattr(self, "_overview_table"):
            return
        summary_rows = self._compute_replay_overview(metrics_list, rr_n)
        self._overview_table.setRowCount(len(summary_rows))
        for r, (name, value) in enumerate(summary_rows):
            self._overview_table.setRowHeight(r, 24)
            n_item = QTableWidgetItem(name)
            v_item = QTableWidgetItem(value)
            bg_col = QColor("#121822") if r % 2 == 0 else QColor("#0A0D13")
            n_item.setBackground(bg_col)
            v_item.setBackground(bg_col)
            n_item.setForeground(QColor("#FFFFFF"))
            v_item.setForeground(QColor("#FFFFFF"))
            self._overview_table.setItem(r, 0, n_item)
            self._overview_table.setItem(r, 1, v_item)

    def _compute_replay_overview(self, metrics_list: list, rr_n: list) -> list:
        total_beats = len(rr_n)
        avg_hr = (60000.0 / float(np.mean(rr_n))) if rr_n else 0.0
        max_hr = (60000.0 / float(min(rr_n))) if rr_n else 0.0
        min_hr = (60000.0 / float(max(rr_n))) if rr_n else 0.0
        pauses = int(sum(1 for rr in rr_n if rr >= 2000))
        longest_rr = (max(rr_n) / 1000.0) if rr_n else 0.0
        ve = int(sum(1 for m in metrics_list if any("V" in str(a) for a in (m.get("arrhythmias", []) or []))))
        sve = int(sum(1 for m in metrics_list if any(("PAC" in str(a)) or ("SVE" in str(a)) for a in (m.get("arrhythmias", []) or []))))
        
        # Calculate percentages and durations
        total_chunks = len(metrics_list) if metrics_list else 1
        tachy_chunks = sum(1 for m in metrics_list if m.get('hr_mean', 0) > 100)
        brady_chunks = sum(1 for m in metrics_list if 0 < m.get('hr_mean', 0) < 60)
        af_chunks = sum(1 for m in metrics_list if any("AF" in str(a) for a in (m.get("arrhythmias", []) or [])))
        
        # Each chunk is roughly 4 seconds.
        chunk_dur = 4.0
        tachy_pct = (tachy_chunks / total_chunks) * 100.0
        brady_pct = (brady_chunks / total_chunks) * 100.0
        af_dur_str = f"{(af_chunks * chunk_dur) / 60.0:.1f} m"
        
        return [
            ("Total NNs", str(total_beats)),
            ("AVG HR", f"{avg_hr:.0f} bpm"),
            ("Max HR", f"{max_hr:.0f} bpm"),
            ("Min HR", f"{min_hr:.0f} bpm"),
            ("Longest RR Interval", f"{longest_rr:.2f}s"),
            ("RRI (>=2.0s)", str(pauses)),
            ("During Tachy (>100)", f"{tachy_pct:.1f}%"),
            ("During Brady (<60)", f"{brady_pct:.1f}%"),
            ("AF Duration", af_dur_str),
            ("V Total", str(ve)),
            ("S Total", str(sve)),
        ]


#  "  "  Helper canvas widgets  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "

