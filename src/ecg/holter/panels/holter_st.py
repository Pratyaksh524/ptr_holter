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
    _show_message_box, _style_active_btn, _style_btn, _table_style, _template_filter_key,
    _card_style, _card_title_style,
)

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas, HolterRRTrendCanvas, HRTrendCanvas, TendencyCanvas

def _seg_btn_style(active: bool) -> str:
    """One button of a segmented toggle (its group frame draws the outline)."""
    if active:
        return (f"QPushButton{{background:{UI_ACCENT};color:#FFFFFF;border:none;border-radius:6px;"
                "padding:4px 14px;font-size:12px;font-weight:bold;}")
    return (f"QPushButton{{background:transparent;color:{UI_MUTED};border:none;border-radius:6px;"
            "padding:4px 14px;font-size:12px;font-weight:bold;}"
            f"QPushButton:hover{{color:{UI_TEXT};background:#1A2C49;}}")


def _seg_group(object_name: str, buttons) -> QFrame:
    """A rounded frame holding `buttons` side by side (a segmented toggle)."""
    group = QFrame()
    group.setObjectName(object_name)
    group.setStyleSheet(f"QFrame#{object_name}{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                        f"border-radius:8px;}}")
    lay = QHBoxLayout(group)
    lay.setContentsMargins(3, 3, 3, 3)
    lay.setSpacing(2)
    for b in buttons:
        b.setCursor(Qt.PointingHandCursor)
        b.setFixedHeight(26)
        lay.addWidget(b)
    return group


def _caption(text: str) -> QLabel:
    c = QLabel(text)
    c.setStyleSheet(f"color:{UI_MUTED};font-size:9px;font-weight:700;letter-spacing:1px;"
                    f"background:transparent;border:none;")
    return c


def _vline() -> QFrame:
    line = QFrame()
    line.setFixedWidth(1)
    line.setStyleSheet(f"background:{UI_BORDER};border:none;")
    return line


_VIEW_HINTS = {
    "ST": "Left: 12 leads at the replay position  \u00b7  Right: ST of the whole recording",
    "T": "Left: 12 leads at the replay position  \u00b7  Right: T wave of the whole recording",
    "TEND_CHART": "Click the HR graph or a trend to open that moment  \u00b7  drag I / J / K on the strip",
}


# 13. HOLTER ST TENDENCY PANEL
class HolterSTPanel(QWidget):
    def __init__(self, parent=None, replay_engine=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{UI_BG};")
        self._metrics = []
        self._current_tendency_mode = "ST"
        self._replay_engine = replay_engine
        self._current_view_mode = "ST"  # can be "ST", "T", or "TEND_CHART"
        
        # Tend chart variables
        self._tend_current_tendency_mode = "ST"
        self._tend_scan_start_index = 0
        self._tend_is_scanning = False
        self._tend_current_metric_index = 0
        self._tend_auto_update = False
        self._tend_adjust_k_by_hr = True
        self._tend_st_threshold_mv = 0.1
        self._tend_i_offset_ms = -120
        self._tend_j_offset_ms = 80
        self._tend_k_offset_ms = 120
        self._tend_default_i_offset_ms = -120
        self._tend_default_j_offset_ms = 80
        self._tend_default_k_offset_ms = 120
        self._tend_scan_results = []
        
        self._build_ui()
        
    def set_replay_engine(self, replay_engine):
        self._replay_engine = replay_engine

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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Mode row (always visible)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        self._tend_chart_btn = QPushButton("Tend chart")
        self._tend_chart_btn.setStyleSheet(_seg_btn_style(False))
        self._st_btn = QPushButton("ST")
        self._st_btn.setStyleSheet(_seg_btn_style(True))
        self._st_btn.setFixedWidth(50)
        self._t_btn = QPushButton("T")
        self._t_btn.setStyleSheet(_seg_btn_style(False))
        self._t_btn.setFixedWidth(50)

        self._tend_chart_btn.clicked.connect(lambda: self._set_view_mode("TEND_CHART"))
        self._st_btn.clicked.connect(lambda: self._set_view_mode("ST"))
        self._t_btn.clicked.connect(lambda: self._set_view_mode("T"))

        mode_row.addWidget(_seg_group("stModeGroup", [self._tend_chart_btn, self._st_btn, self._t_btn]))
        # what the current view shows (changes with the toggle)
        self._mode_hint = QLabel(_VIEW_HINTS["ST"])
        self._mode_hint.setStyleSheet(f"color:{UI_MUTED};font-size:11px;background:transparent;border:none;")
        mode_row.addWidget(self._mode_hint)
        mode_row.addStretch()
        main_layout.addLayout(mode_row)

        # Main content stacked widget
        self._main_stack = QStackedWidget()
        
        # --- Normal view (ST/T with left ECG strips) ---
        normal_view_widget = QWidget()
        normal_layout = QHBoxLayout(normal_view_widget)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.setSpacing(8)

        scroll_css = (
            f"QScrollArea{{background:{COL_BLACK};border:1px solid {UI_BORDER};border-radius:6px;}}"
            f"QScrollBar:vertical{{background:{UI_PANEL};width:10px;margin:0;border:none;}}"
            f"QScrollBar::handle:vertical{{background:{UI_BORDER};min-height:24px;border-radius:5px;margin:2px;}}"
            f"QScrollBar::handle:vertical:hover{{background:#3A5378;}}"
            f"QScrollBar::add-line,QScrollBar::sub-line{{width:0;height:0;}}")

        def _lead_block(name, widget, object_name):
            """One lead: "Lead X" on a slim header line (green accent), the
            strip / trend under it, a faint rule below."""
            block = QFrame()
            block.setObjectName(object_name)
            block.setStyleSheet(f"QFrame#{object_name}{{background:transparent;border:none;"
                                f"border-bottom:1px solid {UI_BORDER};}}")
            block_l = QVBoxLayout(block)
            block_l.setContentsMargins(0, 0, 0, 4)
            block_l.setSpacing(2)
            lbl = QLabel(f"Lead {name}")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;background:transparent;"
                              f"border:none;border-left:2px solid {COL_GREEN};padding:0px 6px;")
            block_l.addWidget(lbl)
            block_l.addWidget(widget)
            return block

        # Left: ECG strip + controls
        left = QWidget()
        left.setObjectName("stLeftCard")
        left.setAttribute(Qt.WA_StyledBackground, True)
        left.setStyleSheet(_card_style("QWidget#stLeftCard"))
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        ecg_ttl = QLabel("12-Lead ECG")
        ecg_ttl.setStyleSheet(_card_title_style())
        left_layout.addWidget(ecg_ttl)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(scroll_css)

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background:transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(6, 6, 6, 6)
        scroll_layout.setSpacing(6)

        self._ch_strips = []
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for i in range(12):
            # Disable vertical lines for ST Tendency panel
            strip = ECGStripCanvas(height=80, show_vertical_lines=False)
            strip.lead_name = lead_names[i]
            scroll_layout.addWidget(_lead_block(lead_names[i], strip, "stLeadBlock"))
            self._ch_strips.append(strip)

        scroll_area.setWidget(scroll_content)
        left_layout.addWidget(scroll_area, 1)

        # Mini overview (Lead II rhythm strip) in its own box
        rhythm_box = QFrame()
        rhythm_box.setObjectName("stRhythmBox")
        rhythm_box.setStyleSheet(f"QFrame#stRhythmBox{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                                 f"border-radius:6px;}}")
        rhythm_l = QVBoxLayout(rhythm_box)
        rhythm_l.setContentsMargins(8, 6, 8, 8)
        rhythm_l.setSpacing(4)
        rhythm_ttl = QLabel("Rhythm Strip  \u00b7  Lead II")
        rhythm_ttl.setStyleSheet(_card_title_style())
        rhythm_l.addWidget(rhythm_ttl)
        self._mini_strip = ECGStripCanvas(height=60, color="#00AA00")
        rhythm_l.addWidget(self._mini_strip)
        left_layout.addWidget(rhythm_box)

        # Actions: rescan | step | remove | reset
        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        for g, names in enumerate([["ReScan"], ["Next Event"], ["Remove All", "Remove"], ["Reset"]]):
            if g:
                nav_row.addSpacing(8)
            for lbl in names:
                btn = QPushButton(lbl)
                btn.setStyleSheet(_style_btn())
                btn.setFixedHeight(30)
                btn.setCursor(Qt.PointingHandCursor)
                nav_row.addWidget(btn, 1)
        left_layout.addLayout(nav_row)
        normal_layout.addWidget(left, 3)

        # Right: ST tendency charts + conclusion
        right = QWidget()
        right.setObjectName("stRightCard")
        right.setAttribute(Qt.WA_StyledBackground, True)
        right.setStyleSheet(_card_style("QWidget#stRightCard"))
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        st_head = QHBoxLayout()
        self._st_title = QLabel("ST tendency(mV)")
        self._st_title.setStyleSheet(_card_title_style())
        st_head.addWidget(self._st_title)
        st_hint = QLabel("Full recording, per lead")
        st_hint.setStyleSheet(f"color:{UI_MUTED};font-size:11px;background:transparent;border:none;")
        st_hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        st_hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        st_hint.setMinimumWidth(0)
        st_head.addWidget(st_hint, 1)
        right_layout.addLayout(st_head)

        self._st_canvases = []
        st_scroll = QScrollArea()
        st_scroll.setWidgetResizable(True)
        st_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        st_scroll.setStyleSheet(scroll_css)
        st_scroll_content = QWidget()
        st_scroll_content.setStyleSheet("background:transparent;")
        st_scroll_layout = QVBoxLayout(st_scroll_content)
        st_scroll_layout.setContentsMargins(6, 6, 6, 6)
        st_scroll_layout.setSpacing(6)
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for name in lead_names:
            canvas = STCanvas(height=70)
            st_scroll_layout.addWidget(_lead_block(name, canvas, "stTrendBlock"))
            self._st_canvases.append(canvas)
        st_scroll.setWidget(st_scroll_content)
        right_layout.addWidget(st_scroll, 1)

        conclusion_frame = QFrame()
        conclusion_frame.setObjectName("stConclusionCard")
        conclusion_frame.setStyleSheet(f"QFrame#stConclusionCard{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};border-radius:6px;}}")
        cf_layout = QVBoxLayout(conclusion_frame)
        cf_layout.setContentsMargins(8, 6, 8, 8)
        cf_layout.setSpacing(6)
        cf_lbl = QLabel("Conclusion")
        cf_lbl.setStyleSheet(_card_title_style())
        cf_layout.addWidget(cf_lbl)
        self._conclusion_edit = QTextEdit()
        self._conclusion_edit.setFixedHeight(64)
        self._conclusion_edit.setPlaceholderText("Write the ST / T findings for the report")
        self._conclusion_edit.setStyleSheet(f"QTextEdit{{background:{UI_BG};color:{UI_TEXT};"
                                             f"border:1px solid {UI_BORDER};border-radius:4px;font-size:11px;padding:4px;}}"
                                             f"QTextEdit:focus{{border:1px solid {UI_ACCENT};}}")
        cf_layout.addWidget(self._conclusion_edit)
        save_row = QHBoxLayout()
        save_row.setSpacing(6)
        for lbl in ["Save as template", "Quote templates"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn.setFixedHeight(30)
            save_row.addWidget(btn, 1)
        cf_layout.addLayout(save_row)
        right_layout.addWidget(conclusion_frame)
        normal_layout.addWidget(right, 2)
        
        # --- Tend chart view widget ---
        tend_widget = QWidget()
        tend_layout = QVBoxLayout(tend_widget)
        tend_layout.setContentsMargins(0,0,0,0)
        tend_layout.setSpacing(4)
        
        # Tend chart toolbar: one card -- SCAN | I / J / K | ST THRESHOLD
        toolbar_card = QFrame()
        toolbar_card.setObjectName("stTendToolbar")
        toolbar_card.setStyleSheet(f"QFrame#stTendToolbar{{background:{UI_PANEL};border:1px solid {UI_BORDER};"
                                   f"border-radius:8px;}}")
        toolbar = QHBoxLayout(toolbar_card)
        toolbar.setContentsMargins(10, 6, 10, 6)
        toolbar.setSpacing(6)
        toolbar.addWidget(_caption("SCAN"))

        self._tend_start_btn = QPushButton("Start")
        self._tend_start_btn.setStyleSheet(_style_btn())
        self._tend_start_btn.setFixedHeight(32)
        self._tend_start_btn.clicked.connect(self._tend_on_start_scan)
        toolbar.addWidget(self._tend_start_btn)
        
        self._tend_confirm_btn = QPushButton("Confirm")
        self._tend_confirm_btn.setStyleSheet(_style_btn())
        self._tend_confirm_btn.setFixedHeight(32)
        self._tend_confirm_btn.clicked.connect(self._tend_on_confirm_scan)
        toolbar.addWidget(self._tend_confirm_btn)
        
        self._tend_default_btn = QPushButton("Default")
        self._tend_default_btn.setStyleSheet(_style_btn())
        self._tend_default_btn.setFixedHeight(32)
        self._tend_default_btn.clicked.connect(self._tend_on_reset_defaults)
        toolbar.addWidget(self._tend_default_btn)
        
        self._tend_auto_update_btn = QPushButton("Update automatically")
        self._tend_auto_update_btn.setCheckable(True)
        self._tend_auto_update_btn.setStyleSheet(_style_btn() + f"QPushButton:checked{{background:#1E3A5F;color:{UI_TEXT};border:1px solid {UI_ACCENT};}}")
        self._tend_auto_update_btn.setFixedHeight(32)
        self._tend_auto_update_btn.clicked.connect(self._tend_on_toggle_auto_update)
        toolbar.addWidget(self._tend_auto_update_btn)
        toolbar.addSpacing(6)
        toolbar.addWidget(_vline())
        toolbar.addSpacing(6)
        toolbar.addWidget(_caption("I / J / K"))
        # Default belongs with the markers it resets
        toolbar.removeWidget(self._tend_default_btn)
        toolbar.addWidget(self._tend_default_btn)
        
        self._tend_adjust_k_btn = QPushButton("Adjust K by HR")
        self._tend_adjust_k_btn.setCheckable(True)
        self._tend_adjust_k_btn.setChecked(True)
        self._tend_adjust_k_btn.setStyleSheet(_style_btn() + f"QPushButton:checked{{background:#1E3A5F;color:{UI_TEXT};border:1px solid {UI_ACCENT};}}")
        self._tend_adjust_k_btn.setFixedHeight(32)
        self._tend_adjust_k_btn.clicked.connect(self._tend_on_toggle_adjust_k)
        toolbar.addWidget(self._tend_adjust_k_btn)
        
        toolbar.addSpacing(6)
        toolbar.addWidget(_vline())
        toolbar.addSpacing(6)
        threshold_lbl = QLabel("ST Threshold (mV):")
        threshold_lbl.setStyleSheet(f"color:{UI_MUTED};font-weight:bold;background:transparent;border:none;")
        toolbar.addWidget(threshold_lbl)
        self._tend_threshold_spin = QDoubleSpinBox()
        self._tend_threshold_spin.setRange(0.01, 1.0)
        self._tend_threshold_spin.setSingleStep(0.01)
        self._tend_threshold_spin.setValue(self._tend_st_threshold_mv)
        self._tend_threshold_spin.valueChanged.connect(self._tend_on_threshold_changed)
        self._tend_threshold_spin.setFixedHeight(32)
        self._tend_threshold_spin.setStyleSheet(f"QDoubleSpinBox{{background:{UI_PANEL_ALT};color:{UI_TEXT};border:1px solid {UI_BORDER};padding:5px;border-radius:6px;}}"
                                                f"QDoubleSpinBox:hover, QDoubleSpinBox:focus{{border:1px solid {UI_ACCENT};}}")
        toolbar.addWidget(self._tend_threshold_spin)
        
        toolbar.addStretch()
        for b in (self._tend_start_btn, self._tend_confirm_btn, self._tend_default_btn,
                  self._tend_auto_update_btn, self._tend_adjust_k_btn):
            b.setCursor(Qt.PointingHandCursor)
        tend_layout.setSpacing(8)
        tend_layout.addWidget(toolbar_card)
        
        # Tend chart main content
        main_content = QHBoxLayout()
        
        # Left: HR graph + ECG strip
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_panel_layout.setSpacing(8)
        
        # HR graph
        hr_frame = QFrame()
        hr_frame.setObjectName("stHrCard")
        hr_frame.setStyleSheet(_card_style("QFrame#stHrCard"))
        hr_layout = QVBoxLayout(hr_frame)
        hr_layout.setContentsMargins(10, 10, 10, 10)
        hr_layout.setSpacing(6)
        self._tend_hr_title = QLabel("Entire HR Graph")
        self._tend_hr_title.setStyleSheet(_card_title_style())
        self._tend_hr_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        hr_layout.addWidget(self._tend_hr_title)
        self._tend_hr_canvas = HRTrendCanvas()
        self._tend_hr_canvas.clicked.connect(self._tend_on_hr_canvas_clicked)
        hr_layout.addWidget(self._tend_hr_canvas, 1)
        left_panel_layout.addWidget(hr_frame, 1)
        
        # ECG strip with I/J/K markers
        ecg_frame = QFrame()
        ecg_frame.setObjectName("stEcgCard")
        ecg_frame.setStyleSheet(_card_style("QFrame#stEcgCard"))
        ecg_layout = QVBoxLayout(ecg_frame)
        ecg_layout.setContentsMargins(10, 10, 10, 10)
        ecg_layout.setSpacing(6)
        self._tend_ecg_title = QLabel("ECG Strip (drag I/J/K markers)")
        self._tend_ecg_title.setStyleSheet(_card_title_style())
        self._tend_ecg_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        ecg_layout.addWidget(self._tend_ecg_title)
        marker_legend = QLabel("I: baseline before the QRS  \u00b7  J: J point (end of QRS)  \u00b7  "
                               "K: ST measuring point  \u00b7  T: T wave")
        marker_legend.setWordWrap(True)
        marker_legend.setStyleSheet(f"color:{UI_MUTED};font-size:10px;background:transparent;border:none;")
        ecg_layout.addWidget(marker_legend)
        self._tend_ecg_strip = STTMarkerCanvas(height=200)
        self._tend_ecg_strip.markerMoved.connect(self._tend_on_marker_moved)
        ecg_layout.addWidget(self._tend_ecg_strip, 1)
        # its own height (the strip is a fixed 200 px); the HR graph takes the rest
        left_panel_layout.addWidget(ecg_frame, 0)
        
        main_content.addWidget(left_panel, 1)
        
        # Right: Tendency charts
        right_panel = QFrame()
        right_panel.setObjectName("stTrendCard")
        right_panel.setStyleSheet(_card_style("QFrame#stTrendCard"))
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(10, 10, 10, 10)
        right_panel_layout.setSpacing(8)

        mode_row_tend = QHBoxLayout()
        self._tend_tendency_title = QLabel("ST Trends")
        self._tend_tendency_title.setStyleSheet(_card_title_style())
        mode_row_tend.addWidget(self._tend_tendency_title)
        mode_row_tend.addStretch()
        self._tend_st_mode_btn = QPushButton("ST")
        self._tend_st_mode_btn.setStyleSheet(_seg_btn_style(True))
        self._tend_t_mode_btn = QPushButton("T")
        self._tend_t_mode_btn.setStyleSheet(_seg_btn_style(False))
        for btn in [self._tend_st_mode_btn, self._tend_t_mode_btn]:
            btn.setFixedWidth(50)
        mode_row_tend.addWidget(_seg_group("stTendModeGroup", [self._tend_st_mode_btn, self._tend_t_mode_btn]))
        right_panel_layout.addLayout(mode_row_tend)

        self._tend_tendency_scroll = QScrollArea()
        self._tend_tendency_scroll.setWidgetResizable(True)
        self._tend_tendency_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._tend_tendency_scroll.setStyleSheet(scroll_css)
        
        self._tend_tendency_content = QWidget()
        self._tend_tendency_layout = QVBoxLayout(self._tend_tendency_content)
        self._tend_tendency_content.setStyleSheet("background:transparent;")
        self._tend_tendency_layout.setContentsMargins(4,4,4,4)
        self._tend_tendency_layout.setSpacing(4)
        
        lead_names_tend = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self._tend_tendency_canvases = []
        for name in lead_names_tend:
            canvas = TendencyCanvas(height=70)
            canvas.set_threshold(self._tend_st_threshold_mv)
            # the signal carries the index of the clicked point
            canvas.clicked.connect(lambda idx, c=canvas: self._tend_on_tendency_clicked(idx, c))
            self._tend_tendency_layout.addWidget(_lead_block(name, canvas, "stTendLeadBlock"))
            self._tend_tendency_canvases.append(canvas)
        
        self._tend_tendency_scroll.setWidget(self._tend_tendency_content)
        right_panel_layout.addWidget(self._tend_tendency_scroll, 1)
        
        main_content.addWidget(right_panel, 2)
        tend_layout.addLayout(main_content, 1)
        
        # Connect tend chart mode buttons
        self._tend_st_mode_btn.clicked.connect(lambda: self._tend_set_mode("ST"))
        self._tend_t_mode_btn.clicked.connect(lambda: self._tend_set_mode("T"))
        
        # Add widgets to main stack
        self._main_stack.addWidget(normal_view_widget)
        self._main_stack.addWidget(tend_widget)
        self._main_stack.setCurrentIndex(0)
        
        main_layout.addWidget(self._main_stack, 1)

    def _set_view_mode(self, mode):
        self._current_view_mode = mode
        
        # Update button styles
        self._tend_chart_btn.setStyleSheet(_seg_btn_style(False))
        self._st_btn.setStyleSheet(_seg_btn_style(False))
        self._t_btn.setStyleSheet(_seg_btn_style(False))
        self._mode_hint.setText(_VIEW_HINTS.get(mode, ""))

        if mode == "TEND_CHART":
            self._tend_chart_btn.setStyleSheet(_seg_btn_style(True))
            self._main_stack.setCurrentIndex(1)
            self._tend_update_hr_graph()
            self._tend_update_tendency_graphs()
            if self._replay_engine:
                self._tend_update_ecg_strip()
        elif mode == "ST":
            self._st_btn.setStyleSheet(_seg_btn_style(True))
            self._main_stack.setCurrentIndex(0)
            self._set_tendency_mode("ST")
        elif mode == "T":
            self._t_btn.setStyleSheet(_seg_btn_style(True))
            self._main_stack.setCurrentIndex(0)
            self._set_tendency_mode("T")

    def _set_tendency_mode(self, mode: str):
        self._current_tendency_mode = mode
        if mode == "ST":
            self._st_title.setText("ST tendency(mV)")
        else:
            self._st_title.setText("T tendency(mV)")
        self._refresh_plot()

    def set_replay_frame(self, data):
        if self._current_view_mode in ["ST", "T"]:
            if data is None or data.shape[0] < 1: return
            N = data.shape[1]
            fs = 500.0
            n_samples = min(N, int(10 * fs))
            x = np.linspace(0, n_samples/fs, n_samples) if n_samples > 0 else []
            if n_samples > 0:
                lead2_idx = 1 if data.shape[0] > 1 else 0
                self._mini_strip.set_data(x, data[lead2_idx, :n_samples].copy())
                for i, ts in enumerate(self._ch_strips):
                    if i < data.shape[0]:
                        ts.set_data(x, data[i, :n_samples].copy())
        elif self._current_view_mode == "TEND_CHART":
            self._tend_update_ecg_strip()

    def update_from_metrics(self, metrics_list):
        self._metrics = metrics_list
        self._refresh_plot()
        if self._current_view_mode == "TEND_CHART":
            self._tend_update_hr_graph()
            self._tend_update_tendency_graphs()

    def _refresh_plot(self):
        if self._current_tendency_mode == "ST":
            vals = [m.get('st_mv', 0.0) for m in self._metrics]
        else:
            vals = [m.get('t_mv', 0.0) for m in self._metrics]
        for canvas in self._st_canvases:
            canvas.set_data(vals)

    def update_st_ecg_display(self, data):
        """Update ECG display for ST/T normal view mode."""
        if data is None or data.shape[0] < 1: 
            return
        
        N = data.shape[1]
        fs = 500.0
        n_samples = min(N, int(10 * fs))
        x = np.linspace(0, n_samples/fs, n_samples) if n_samples > 0 else []
        
        if n_samples > 0:
            lead2_idx = 1 if data.shape[0] > 1 else 0
            self._mini_strip.set_data(x, data[lead2_idx, :n_samples].copy())
            for i, ts in enumerate(self._ch_strips):
                if i < data.shape[0]:
                    ts.set_data(x, data[i, :n_samples].copy())

    def set_replay_frame(self, data):
        """Main entry point for updating ECG displays based on current view mode."""
        if data is None or data.shape[0] < 1: 
            return
        
        if self._current_view_mode in ["ST", "T"]:
            # Update left-side ECG strips (all 12 leads) for ST/T mode
            self.update_st_ecg_display(data)
        
        elif self._current_view_mode == "TEND_CHART":
            # Update tendency chart ECG strip
            self._tend_update_ecg_strip()
            
    # --- Tend chart methods ---
    def _tend_set_mode(self, mode):
        self._tend_current_tendency_mode = mode
        if mode == "ST":
            self._tend_st_mode_btn.setStyleSheet(_seg_btn_style(True))
            self._tend_t_mode_btn.setStyleSheet(_seg_btn_style(False))
            self._tend_tendency_title.setText("ST Trends")
        else:
            self._tend_t_mode_btn.setStyleSheet(_seg_btn_style(True))
            self._tend_st_mode_btn.setStyleSheet(_seg_btn_style(False))
            self._tend_tendency_title.setText("T Trends")
        self._tend_update_tendency_graphs()
        
    def _tend_update_hr_graph(self):
        """HR of every chunk at its own time (s) on the bpm scale -- was the
        chunk number on the time axis (178 chunks of a 14:51 recording read
        0:00-2:30) and the 0-3000 RR scale (HR a flat line at the bottom)."""
        points = [(float(m.get('t', i) or 0.0), float(m.get('hr_mean', 0) or 0.0))
                  for i, m in enumerate(self._metrics)]
        self._tend_hr_canvas.set_mode("HR")
        self._tend_hr_canvas.set_points(points)
        self._tend_hr_canvas.set_current_index(self._tend_current_metric_index)
        
    def _tend_update_tendency_graphs(self):
        # each value at its chunk's time (s), as the HR graph above
        if self._tend_scan_results:
            times = [float(res.get('t_sec', i) or 0.0) for i, res in enumerate(self._tend_scan_results)]
            st_vals = [res['st_mv'] for res in self._tend_scan_results]
            t_vals = [res['t_mv'] for res in self._tend_scan_results]
        else:
            times = [float(m.get('t', i) or 0.0) for i, m in enumerate(self._metrics)]
            st_vals = [m.get('st_mv', 0.0) for m in self._metrics]
            t_vals = [m.get('t_mv', 0.0) for m in self._metrics]

        for i, canvas in enumerate(self._tend_tendency_canvases):
            if self._tend_current_tendency_mode == "ST":
                canvas.set_tendency_mode("ST")
                canvas.set_points(list(zip(times, st_vals)))
            else:
                canvas.set_tendency_mode("T")
                canvas.set_points(list(zip(times, t_vals)))
            canvas.set_current_index(self._tend_current_metric_index)
                
    def _tend_update_ecg_strip(self):
        if self._replay_engine:
            data = self._replay_engine.get_all_leads_data(window_sec=3.0)
            if data is not None:
                self._tend_ecg_strip.set_data(data, self._replay_engine.fs)
                r_peak_idx = data.shape[1] // 2
                fs = self._replay_engine.fs
                self._tend_ecg_strip.set_marker_positions(
                    i_pos=r_peak_idx + int(self._tend_i_offset_ms * fs / 1000),
                    j_pos=r_peak_idx + int(self._tend_j_offset_ms * fs / 1000),
                    k_pos=r_peak_idx + int(self._tend_get_k_offset_ms() * fs / 1000),
                    t_pos=r_peak_idx + int(250 * fs / 1000)
                )
    
    def _tend_get_k_offset_ms(self):
        if self._tend_adjust_k_by_hr and self._metrics:
            current_metric = self._metrics[self._tend_current_metric_index]
            hr = current_metric.get('hr_mean', 75)
            if hr > 100:
                return self._tend_k_offset_ms - int((hr - 100) * 0.5)
            elif hr < 60:
                return self._tend_k_offset_ms + int((60 - hr) * 0.5)
        return self._tend_k_offset_ms
        
    def _tend_on_start_scan(self):
        self._tend_is_scanning = True
        self._tend_scan_results = []
        if self._replay_engine:
            self._tend_scan_thread = threading.Thread(target=self._tend_run_scan, daemon=True)
            self._tend_scan_thread.start()
            
    def _tend_run_scan(self):
        fs = self._replay_engine.fs
        for metric in self._metrics:
            t_sec = metric.get('t', 0.0)
            self._replay_engine.seek(t_sec)
            data = self._replay_engine.get_all_leads_data(window_sec=2.0)
            
            lead_results = {}
            for i in range(min(12, data.shape[0])):
                lead_data = data[i]
                st_mv, t_mv = self._tend_analyze_lead(lead_data, fs)
                lead_results[i] = {'st_mv': st_mv, 't_mv': t_mv}
                
            self._tend_scan_results.append({
                't_sec': t_sec,
                'st_mv': lead_results.get(1, {}).get('st_mv', 0.0),
                't_mv': lead_results.get(1, {}).get('t_mv', 0.0),
                'lead_results': lead_results
            })
            
            if self._tend_auto_update:
                QMetaObject.invokeMethod(self, "_tend_update_tendency_graphs", Qt.QueuedConnection)
                time.sleep(0.01)
                
        self._tend_is_scanning = False
        
    def _tend_analyze_lead(self, lead_data: np.ndarray, fs: int):
        from scipy.signal import butter, filtfilt, find_peaks
        
        if len(lead_data) < fs * 0.5:
            return 0.0, 0.0
            
        b, a = butter(2, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype='band')
        filtered = filtfilt(b, a, lead_data)
        
        peaks, _ = find_peaks(np.abs(filtered), distance=int(0.2 * fs))
        if len(peaks) == 0:
            return 0.0, 0.0
            
        r_peak = peaks[len(peaks) // 2]
        
        fs = fs
        i_idx = max(0, r_peak + int(self._tend_i_offset_ms * fs / 1000))
        j_idx = min(len(lead_data) - 1, r_peak + int(self._tend_j_offset_ms * fs / 1000))
        k_idx = min(len(lead_data) - 1, r_peak + int(self._tend_get_k_offset_ms() * fs / 1000))
        
        # T wave: measure from J point (end of QRS) to T peak
        t_start = min(len(lead_data) - 1, r_peak + int(150 * fs / 1000))
        t_end = min(len(lead_data), r_peak + int(400 * fs / 1000))
        
        # ST baseline: measure at I point (before J)
        baseline = lead_data[i_idx] if 0 <= i_idx < len(lead_data) else 0.0
        
        # ST measurement: ST deflection at K point relative to baseline
        st_value = lead_data[k_idx] if 0 <= k_idx < len(lead_data) else 0.0
        st = (st_value - baseline)
        
        # T wave: measure from J point baseline (more accurate)
        j_baseline = lead_data[j_idx] if 0 <= j_idx < len(lead_data) else baseline
        t_amp = 0.0
        if t_start < t_end:
            t_seg = lead_data[t_start:t_end]
            if len(t_seg) > 0:
                peak_idx = np.argmax(np.abs(t_seg - j_baseline))
                t_amp = t_seg[peak_idx] - j_baseline
        
        adc_scale = 1.0 / 200.0
        return st * adc_scale, t_amp * adc_scale
        
    def _tend_on_confirm_scan(self):
        if not self._tend_scan_results:
            return
        _show_message_box(self, QMessageBox.Information, "Scan Complete", f"Processed {len(self._tend_scan_results)} segments")
        
    def _tend_on_reset_defaults(self):
        self._tend_i_offset_ms = self._tend_default_i_offset_ms
        self._tend_j_offset_ms = self._tend_default_j_offset_ms
        self._tend_k_offset_ms = self._tend_default_k_offset_ms
        self._tend_update_ecg_strip()
        
    def _tend_on_toggle_auto_update(self, checked):
        self._tend_auto_update = checked
        
    def _tend_on_toggle_adjust_k(self, checked):
        self._tend_adjust_k_by_hr = checked
        self._tend_update_ecg_strip()
        
    def _tend_on_threshold_changed(self, value):
        self._tend_st_threshold_mv = value
        for canvas in self._tend_tendency_canvases:
            canvas.set_threshold(value)
        
    def _tend_on_hr_canvas_clicked(self, index):
        self._tend_current_metric_index = index
        self._tend_update_hr_graph()
        self._tend_update_tendency_graphs()
        if self._replay_engine:
            t_sec = self._metrics[index].get('t', 0.0)
            self._replay_engine.seek(t_sec)
            self._tend_update_ecg_strip()
            
    def _tend_on_tendency_clicked(self, index, canvas):
        """A click on a lead's trend opens that point (the canvas's clicked
        signal carries its index) -- was the canvas's current point
        (get_clicked_index), so a click on a trend went nowhere new."""
        if index is not None and 0 <= int(index) < len(self._metrics):
            self._tend_on_hr_canvas_clicked(int(index))
            
    def _tend_on_marker_moved(self, marker_name, pos_samples):
        if not self._replay_engine:
            return
            
        fs = self._replay_engine.fs
        center_idx = self._tend_ecg_strip._data.shape[1] // 2 if len(self._tend_ecg_strip._data.shape) >1 else len(self._tend_ecg_strip._data)//2
        delta_ms = int((pos_samples - center_idx) * 1000 / fs)
        
        if marker_name == 'I':
            self._tend_i_offset_ms = delta_ms
        elif marker_name == 'J':
            self._tend_j_offset_ms = delta_ms
        elif marker_name == 'K':
            self._tend_k_offset_ms = delta_ms

