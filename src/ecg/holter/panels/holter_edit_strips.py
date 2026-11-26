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
        ev_head = QHBoxLayout()
        ev_title = QLabel("Event List")
        ev_title.setStyleSheet(_card_title_style())
        ev_head.addWidget(ev_title)
        ev_head.addStretch(1)
        self._ev_count_badge = QLabel("0 events")
        self._ev_count_badge.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:700;background:{UI_PANEL_ALT};"
                                           f"border:1px solid {UI_BORDER};border-radius:10px;padding:2px 10px;")
        ev_head.addWidget(self._ev_count_badge)
        left_layout.addLayout(ev_head)
        # event-type filter (as in EVENTS): each name with its count
        self._type_filter = QComboBox()
        self._type_filter.setCursor(Qt.PointingHandCursor)
        self._type_filter.setStyleSheet(f"""
            QComboBox {{
                background:{UI_PANEL_ALT}; color:{UI_TEXT};
                border:1px solid {UI_BORDER}; padding:4px 8px; border-radius:6px; font-size:11px;
            }}
            QComboBox:hover {{ border:1px solid {UI_ACCENT}; }}
            QComboBox QAbstractItemView {{
                background:{UI_PANEL}; color:{UI_TEXT}; border:1px solid {UI_BORDER};
                selection-background-color:#1E3A5F; selection-color:{UI_TEXT}; outline:0;
            }}
        """)
        self._type_filter.currentIndexChanged.connect(lambda _i: self._apply_type_filter())
        left_layout.addWidget(self._type_filter)

        cols = ["Event name", "Start Time", "Chan.", "Print Len."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        # the name takes the spare width (was cut to "Sinus ..."); the short
        # columns fit their text
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._ev_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
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
        self._ev_table.verticalHeader().setDefaultSectionSize(28)
        self._ev_table.setCursor(Qt.PointingHandCursor)
        self._ev_table.cellClicked.connect(self._on_event_clicked)
        left_layout.addWidget(self._ev_table, 1)

        # Empty state, centred over the table (kept centred on resize).
        self._no_items_lbl = QLabel("No events")
        self._no_items_lbl.setParent(self._ev_table.viewport())
        self._no_items_lbl.setAlignment(Qt.AlignCenter)
        self._no_items_lbl.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._no_items_lbl.setStyleSheet(f"color:{UI_MUTED};font-size:12px;font-style:italic;padding:8px;"
                                         f"border:none;background:transparent;")
        self._ev_table.viewport().installEventFilter(self)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        for lbl in ["Prev", "Next", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_es_danger_btn_style() if lbl.startswith("Remove") else _style_btn())
            btn.setCursor(Qt.PointingHandCursor)
            if lbl == "Prev":
                btn.clicked.connect(lambda _=False: self._step_event(-1))
            elif lbl == "Next":
                btn.clicked.connect(lambda _=False: self._step_event(1))
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
        sf_layout.setHorizontalSpacing(8)
        sf_layout.setVerticalSpacing(6)
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
            l = QLabel(label)
            l.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:bold;border:none;background:transparent;")
            v = QLabel("-")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v.setStyleSheet(f"color:{UI_TEXT};font-size:12px;font-weight:bold;border:none;background:transparent;")
            sf_layout.addWidget(l, row, col * 3)
            sf_layout.addWidget(v, row, col * 3 + 1)
            self._stat_labels[key] = v
        # a thin rule between the two columns
        col_rule = QFrame()
        col_rule.setFixedWidth(1)
        col_rule.setStyleSheet(f"background:{UI_BORDER};border:none;")
        sf_layout.addWidget(col_rule, 0, 2, 4, 1)
        sf_layout.setColumnStretch(1, 1)
        sf_layout.setColumnStretch(4, 1)
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 2)

        # --- CENTER: 2x2 Thumbnail Boxes (30% width) ---
        center = QFrame()
        center.setObjectName("esCenterCard")
        center.setStyleSheet(_card_style("QFrame#esCenterCard"))
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(8, 8, 8, 8)
        center_layout.setSpacing(8)

        strips_ttl = QLabel("Heart Rate Strips")
        strips_ttl.setStyleSheet(_card_title_style())
        center_layout.addWidget(strips_ttl)

        # Tool buttons, in groups
        tool_row = QHBoxLayout()
        tool_row.setSpacing(4)
        for g, group in enumerate([["Refresh", "Menu"], ["Prev", "Next"], ["<<", ">>"], ["Up", "Down", "End"]]):
            if g:
                sep = QFrame()
                sep.setFixedSize(1, 20)
                sep.setStyleSheet(f"background:{UI_BORDER};border:none;")
                tool_row.addSpacing(4)
                tool_row.addWidget(sep)
                tool_row.addSpacing(4)
            for icon in group:
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
                # no lead_name: a strip named "I" prints the RR numbers over itself
                strip = ECGStripCanvas(height=56)
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
        strips_hint = QLabel("Click a tile to open that moment: Replay moves there and the Detailed View "
                             "shows its 12 leads.")
        strips_hint.setWordWrap(True)
        strips_hint.setStyleSheet(f"color:{UI_MUTED};font-size:11px;background:transparent;border:none;")
        center_layout.addWidget(strips_hint)
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
            f"color:{UI_TEXT};font-size:26px;font-weight:bold;border:none;background:transparent;"
        )
        self._detail_time = QLabel("-")
        self._detail_time.setStyleSheet(
            f"color:{UI_MUTED};font-size:12px;font-weight:bold;border:none;background:transparent;"
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
        main_frame.setStyleSheet(_card_style("QFrame#esMainCard"))
        ml = QVBoxLayout(main_frame)
        ml.setContentsMargins(10, 10, 10, 10)
        ml.setSpacing(8)

        self._detail_header_lbl = QLabel("Detailed View")
        self._detail_header_lbl.setStyleSheet(_card_title_style())
        ml.addWidget(self._detail_header_lbl)

        self._main_strips = []
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(
            f"QScrollArea{{background:{COL_BLACK};border:1px solid {UI_BORDER};border-radius:6px;}}"
            f"QScrollBar:vertical{{background:{UI_PANEL};width:10px;margin:0;border:none;}}"
            f"QScrollBar::handle:vertical{{background:{UI_BORDER};min-height:24px;border-radius:5px;margin:2px;}}"
            f"QScrollBar::handle:vertical:hover{{background:#3A5378;}}"
            f"QScrollBar::add-line,QScrollBar::sub-line{{width:0;height:0;}}")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(6, 6, 6, 6)
        scroll_layout.setSpacing(6)

        lead_names = ["Lead I", "Lead II", "Lead III", "Lead aVR", "Lead aVL", "Lead aVF",
                      "Lead V1", "Lead V2", "Lead V3", "Lead V4", "Lead V5", "Lead V6"]

        for name in lead_names:
            # one block per lead: the name on a slim header line, the strip under it
            block = QFrame()
            block.setObjectName("esLeadBlock")
            block.setStyleSheet(f"QFrame#esLeadBlock{{background:transparent;border:none;"
                                f"border-bottom:1px solid {UI_BORDER};}}")
            block_l = QVBoxLayout(block)
            block_l.setContentsMargins(0, 0, 0, 4)
            block_l.setSpacing(2)
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;background:transparent;"
                              f"border:none;border-left:2px solid {COL_GREEN};padding:0px 6px;")
            block_l.addWidget(lbl)
            strip = ECGStripCanvas(height=95)
            block_l.addWidget(strip)
            scroll_layout.addWidget(block)
            self._main_strips.append(strip)
            
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        ml.addWidget(scroll_area)

        right_layout.addWidget(main_frame, 1)

        # Lead II rhythm strip in its own box (as in ST TENDENCY / AF ANALYSIS)
        rhythm_box = QFrame()
        rhythm_box.setObjectName("esRhythmBox")
        rhythm_box.setStyleSheet(f"QFrame#esRhythmBox{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                                 f"border-radius:6px;}}")
        rhythm_l = QVBoxLayout(rhythm_box)
        rhythm_l.setContentsMargins(8, 6, 8, 8)
        rhythm_l.setSpacing(4)
        self._mini_lbl = QLabel("Rhythm Strip  \u00b7  Lead II")
        self._mini_lbl.setStyleSheet(_card_title_style())
        rhythm_l.addWidget(self._mini_lbl)

        self._mini = ECGStripCanvas(height=40, color="#00AA00")
        rhythm_l.addWidget(self._mini)
        right_layout.addWidget(rhythm_box)
        layout.addWidget(right, 5)

    def _card_time_text(self, card) -> str:
        """The tile's moment as system time with the elapsed time:
        "13:59:02 (00:05:00)" -- was the elapsed time alone."""
        ts = float(card.get("timestamp", 0.0) or 0.0)
        if ts > 0:
            return f"{_format_system_time(self._session_dir, ts)} ({card.get('time') or _sec_to_hms(ts)})"
        return card.get("time") or "-"

    def _format_bpm(self, value: float) -> str:
        if value and value > 0:
            return f"{float(value):.0f} bpm"
        return "-- bpm"

    def _on_event_clicked(self, row, col):
        if row < len(self._events):
            self._ev_table.selectRow(row)
            self.seek_requested.emit(self._events[row]['timestamp'])

    def eventFilter(self, obj, event):
        # Keep the empty-state text centred over the table's viewport.
        lbl = getattr(self, "_no_items_lbl", None)
        if lbl is not None and obj is self._ev_table.viewport() and event.type() == QEvent.Resize:
            lbl.setGeometry(obj.rect())
        return super().eventFilter(obj, event)

    def _rebuild_type_filter(self):
        """The Type list: "All events (n)" and each event name with its count
        (arrhythmia names first, then the sinus / normal ones); keeps the
        chosen name while it is still listed."""
        current = self._type_filter.currentData() if self._type_filter.count() else None
        counts: Dict[str, int] = {}
        for ev in self._events:
            name = str(ev.get('label', ''))
            counts[name] = counts.get(name, 0) + 1

        def order(name):
            low = name.lower()
            return (1 if ('sinus' in low or 'normal' in low) else 0, low)
        self._type_filter.blockSignals(True)
        self._type_filter.clear()
        self._type_filter.addItem(f"All events ({len(self._events):,})", None)
        for name in sorted(counts, key=order):
            self._type_filter.addItem(f"{name} ({counts[name]:,})", name)
        idx = self._type_filter.findData(current) if current is not None else 0
        self._type_filter.setCurrentIndex(max(0, idx))
        self._type_filter.blockSignals(False)
        self._apply_type_filter()

    def _apply_type_filter(self):
        """Show only the rows of the chosen event name (all for "All events");
        the badge says how many rows show."""
        name = self._type_filter.currentData()
        shown = 0
        for row, ev in enumerate(self._events):
            hide = name is not None and str(ev.get('label', '')) != name
            self._ev_table.setRowHidden(row, hide)
            shown += 0 if hide else 1
        total = len(self._events)
        if name is None:
            self._ev_count_badge.setText(f"{total:,} event" + ("" if total == 1 else "s"))
        else:
            self._ev_count_badge.setText(f"{shown:,} of {total:,}")
        self._no_items_lbl.setVisible(shown == 0)
        self._no_items_lbl.setText("No events" if total == 0 else "No events of this type")
        self._no_items_lbl.setGeometry(self._ev_table.viewport().rect())
        row = self._selected_row()
        if row >= 0 and self._ev_table.isRowHidden(row):
            self._ev_table.clearSelection()

    def _visible_rows(self) -> List[int]:
        return [r for r in range(len(self._events)) if not self._ev_table.isRowHidden(r)]

    def _selected_row(self) -> int:
        selection = self._ev_table.selectionModel()
        if selection:
            rows = selection.selectedRows()
            if rows:
                return int(rows[0].row())
        return -1

    def _step_event(self, step: int):
        """Prev / Next: the shown row before / after the selected one (the
        first when none is) -- opened as a click on it does."""
        rows = self._visible_rows()
        if not rows:
            return
        row = self._selected_row()
        if row < 0 or row not in rows:
            new = rows[0]
        else:
            new = rows[max(0, min(len(rows) - 1, rows.index(row) + step))]
        self._ev_table.scrollToItem(self._ev_table.item(new, 0))
        self._on_event_clicked(new, 0)

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
                "note": "Highest heart rate in sinus rhythm.",
            },
            "smin_hr": {
                "title": "Sinus Min HR",
                "value": _metric_value("sinus_min_hr", "min_hr"),
                "time": _metric_time("sinus_min_hr", "min_hr"),
                "timestamp": _metric_timestamp("sinus_min_hr", "min_hr"),
                "note": "Lowest heart rate in sinus rhythm.",
            },
        }
        # The values above are the live 5 s chunks' (a chunk's mean HR, and
        # "sinus" by its live labels: Sinus Min 50 bpm at 14:08:42 on
        # 2026-11-12_13-54-02, inside Ventricular Fibrillation). hr_focus
        # gives the saved analysis's: Max / Min the report's page-1 HR (Full
        # Disclosure beats), Sinus Max / Min sinus N-N beats only. Each
        # replaces its live value when its source is saved.
        if self._session_dir:
            try:
                from ..hr_focus import hr_focus_points
                points = hr_focus_points(self._session_dir)
            except Exception as e:
                print(f"[HolterEditStrips] HR focus unavailable: {e}")
                points = {}
            for key, focus_key in (("max_hr", "max_hr"), ("min_hr", "min_hr"),
                                   ("smax_hr", "sinus_max_hr"), ("smin_hr", "sinus_min_hr")):
                point = points.get(focus_key)
                if point:
                    bpm, t_sec = point
                    self._focus_cards[key].update({"value": float(bpm), "timestamp": float(t_sec),
                                                   "time": _sec_to_hms(float(t_sec))})

    def _refresh_focus_tiles(self):
        for key, widget in self._tile_widgets.items():
            card = self._focus_cards.get(key, {})
            widget["hr"].setText(f"HR: {self._format_bpm(card.get('value', 0.0))}")
            widget["time"].setText(self._card_time_text(card))
            widget["frame"].setStyleSheet(
                self._tile_style_active if key == self._selected_focus_key else self._tile_style_normal
            )

    def _load_tile_strips(self, window_sec: float = 10.0):
        """Every card's three strips (leads I, II, III): `window_sec` around
        the card's own moment, read from the recording -- were the replay's
        current frame in all four cards (the same waves in each: a click on
        Min HR put its waves in the Max HR card too)."""
        path = os.path.join(self._session_dir, "recording.ecgh") if self._session_dir else ""
        reader = None
        try:
            if path and os.path.exists(path):
                from ..file_format import ECGHFileReader
                reader = ECGHFileReader(path)
            for key, widget in self._tile_widgets.items():
                card = self._focus_cards.get(key, {})
                ts = float(card.get("timestamp", 0.0) or 0.0)
                data = None
                if reader is not None and ts > 0:
                    t0 = max(0.0, ts - window_sec / 2.0)
                    t1 = min(float(reader.duration_sec), t0 + window_sec)
                    try:
                        data = reader.read_range(t0, t1)
                    except Exception as e:
                        print(f"[HolterEditStrips] Could not read the {key} strip: {e}")
                for lead, strip in enumerate(widget["strips"]):
                    if data is not None and getattr(data, "ndim", 0) == 2 and lead < data.shape[0] and data.shape[1] > 0:
                        n = data.shape[1]
                        strip.set_data(np.linspace(0, n / float(reader.fs), n), data[lead].copy())
                    else:
                        strip.set_data([], [])
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

    def _update_detail_panel(self, key: str):
        card = self._focus_cards.get(key) or self._focus_cards.get("max_hr") or {}
        title = card.get("title", "Heart Rate Detail")
        value = card.get("value", 0.0)
        time_str = self._card_time_text(card)
        note = card.get("note", "")
        self._detail_title.setText(title)
        self._detail_value.setText(self._format_bpm(value))
        self._detail_time.setText(time_str)
        self._detail_note.setText(note or "Click a tile to expand the matching strip view here.")
        self._detail_header_lbl.setText(f"Detailed View  \u00b7  {title}  \u00b7  {time_str}")

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
                # name and time in full colour; the per-row constants muted
                item.setForeground(QColor(UI_TEXT if j < 2 else UI_MUTED))
                self._ev_table.setItem(i, j, item)
        self._rebuild_type_filter()
        if self._selected_focus_key not in self._focus_cards:
            self._selected_focus_key = "max_hr"
        self._refresh_focus_tiles()
        self._load_tile_strips()
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
            # The cards keep their own moment's strips (_load_tile_strips);
            # the replay's frame goes to the detailed view and the rhythm strip.
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


