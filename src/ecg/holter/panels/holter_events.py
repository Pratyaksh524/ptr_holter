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

# 7. HOLTER EVENTS PANEL
class HolterEventsPanel(QWidget):
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
        left = QWidget()
        left.setObjectName("evLeftCard")
        left.setAttribute(Qt.WA_StyledBackground, True)
        left.setStyleSheet(_card_style("QWidget#evLeftCard"))
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        # Title; on the right the event-type filter and how many rows show.
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        ev_title = QLabel("PAC/PVC Events")
        ev_title.setStyleSheet(_card_title_style())
        title_row.addWidget(ev_title)
        title_row.addStretch(1)
        type_lbl = QLabel("Type")
        type_lbl.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:bold;background:transparent;border:none;")
        title_row.addWidget(type_lbl)
        self._type_filter = QComboBox()
        self._type_filter.setMinimumWidth(240)
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
        title_row.addWidget(self._type_filter)
        self._count_badge = QLabel("0 events")
        self._count_badge.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:700;background:{UI_PANEL_ALT};"
                                        f"border:1px solid {UI_BORDER};border-radius:10px;padding:2px 10px;")
        title_row.addWidget(self._count_badge)
        left_layout.addLayout(title_row)

        cols = ["Event name", "Start Time", "Chan.", "Print Len.", "Source", "Conf."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        hdr = self._ev_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for c, w in [(1, 140), (2, 90), (3, 110), (4, 140), (5, 90)]:
            hdr.setSectionResizeMode(c, QHeaderView.Fixed)
            hdr.resizeSection(c, w)
        hdr.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._ev_table.setAlternatingRowColors(True)
        self._ev_table.setShowGrid(False)
        self._ev_table.setCursor(Qt.PointingHandCursor)
        self._ev_table.setStyleSheet(
            _table_style() +
            f"""
            QTableWidget::item:selected {{
                background-color: #1E3A5F;
                color: {UI_TEXT};
                border-top: 1px solid {UI_ACCENT};
                border-bottom: 1px solid {UI_ACCENT};
            }}
            QTableWidget::item:selected:active {{
                background-color: #1E4470;
            }}
            """
        )
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.verticalHeader().setDefaultSectionSize(30)
        self._ev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._ev_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
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

        # Stats below the table: one tile per figure.
        stats_row = QGridLayout()
        stats_row.setHorizontalSpacing(8)
        stats_row.setVerticalSpacing(8)
        self._stat_labels = {}
        for i, (key, label) in enumerate([
            ("hr_max","HR Max"),("hr_min","HR Min"),("hr_smax","Sinus Max HR"),
            ("hr_smin","Sinus Min HR"),("brady","Bradycardia"),("user_ev","User Event"),
        ]):
            tile = QFrame()
            tile.setObjectName("evStatTile")
            tile.setStyleSheet(f"QFrame#evStatTile{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                               f"border-radius:8px;}}")
            tile_l = QVBoxLayout(tile)
            tile_l.setContentsMargins(12, 6, 12, 6)
            tile_l.setSpacing(1)
            l = QLabel(label)
            l.setStyleSheet(f"color:{UI_MUTED};font-size:10px;font-weight:bold;border:none;background:transparent;")
            v = QLabel("-")
            v.setStyleSheet(f"color:{UI_TEXT};font-size:16px;font-weight:bold;border:none;background:transparent;")
            tile_l.addWidget(l)
            tile_l.addWidget(v)
            stats_row.addWidget(tile, 0, i)
            self._stat_labels[key] = v
        left_layout.addLayout(stats_row)
        layout.addWidget(left, 1)

        # Right: navigation, remove, the selected event
        nav = QWidget()
        nav.setObjectName("evNavCard")
        nav.setAttribute(Qt.WA_StyledBackground, True)
        nav.setStyleSheet(_card_style("QWidget#evNavCard"))
        nav.setFixedWidth(240)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(10, 10, 10, 10)
        nav_layout.setSpacing(8)
        nav_ttl = QLabel("Navigate")
        nav_ttl.setStyleSheet(_card_title_style())
        nav_layout.addWidget(nav_ttl)

        def caption(text):
            c = QLabel(text)
            c.setStyleSheet(f"color:{UI_MUTED};font-size:9px;font-weight:700;letter-spacing:1px;"
                            f"background:transparent;border:none;padding:6px 2px 0 2px;")
            return c

        actions = {"Prev Event": self._go_prev_event, "Next Event": self._go_next_event,
                   "Remove All": self._remove_all_events, "Remove": self._remove_selected_event}
        for cap, names in [("EVENT", ["Prev Event", "Next Event"]), ("REMOVE", ["Remove All", "Remove"])]:
            nav_layout.addWidget(caption(cap))
            for label in names:
                btn = QPushButton(label)
                btn.setStyleSheet(_style_btn())
                btn.setFixedHeight(38)
                btn.setCursor(Qt.PointingHandCursor)
                btn.clicked.connect(actions[label])
                nav_layout.addWidget(btn)

        # The selected event, so the doctor sees what Replay was moved to.
        nav_layout.addWidget(caption("SELECTED EVENT"))
        sel_box = QFrame()
        sel_box.setObjectName("evSelectedBox")
        sel_box.setStyleSheet(f"QFrame#evSelectedBox{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                              f"border-radius:8px;}}")
        sel_l = QVBoxLayout(sel_box)
        sel_l.setContentsMargins(10, 8, 10, 8)
        sel_l.setSpacing(3)
        self._sel_name = QLabel("No event selected")
        self._sel_name.setWordWrap(True)
        self._sel_name.setStyleSheet(f"color:{UI_TEXT};font-size:13px;font-weight:bold;background:transparent;border:none;")
        self._sel_time = QLabel("")
        self._sel_time.setStyleSheet(f"color:{UI_TEXT};font-size:12px;background:transparent;border:none;")
        self._sel_meta = QLabel("Click an event to move Replay to it")
        self._sel_meta.setWordWrap(True)
        self._sel_meta.setStyleSheet(f"color:{UI_MUTED};font-size:11px;background:transparent;border:none;")
        for w in (self._sel_name, self._sel_time, self._sel_meta):
            sel_l.addWidget(w)
        nav_layout.addWidget(sel_box)
        nav_layout.addStretch()
        layout.addWidget(nav)

    def eventFilter(self, obj, event):
        # Keep the empty-state text centred over the table's viewport.
        lbl = getattr(self, "_no_items_lbl", None)
        if lbl is not None and obj is self._ev_table.viewport() and event.type() == QEvent.Resize:
            lbl.setGeometry(obj.rect())
        return super().eventFilter(obj, event)

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
        self._selected_payload = {}
        self._ev_table.setRowCount(len(filtered_events))
        muted = QColor(UI_MUTED)
        white = QColor(COL_WHITE)
        for i, ev in enumerate(filtered_events):
            t_str = _format_system_time(self._session_dir, ev['timestamp'])
            source = ev.get("source", "analysis")
            conf = ev.get("confidence", 0.0)
            for j, val in enumerate([ev['label'], t_str, "12", "7s", source, f"{float(conf or 0.0):.2f}"]):
                item = QTableWidgetItem(val)
                # name and time in white; the per-row constants muted
                item.setForeground(white if j < 2 else muted)
                self._ev_table.setItem(i, j, item)
        self._rebuild_type_filter()
        self._show_selected(-1)
        s = summary
        for key, fmt in [("hr_max",f"{s.get('max_hr',0):.0f} bpm"),
                          ("hr_min",f"{s.get('min_hr',0):.0f} bpm"),
                          ("hr_smax",f"{s.get('max_hr',0):.0f} bpm"),
                          ("hr_smin",f"{s.get('min_hr',0):.0f} bpm"),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    # ---- event-type filter ------------------------------------------------

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
        """Show only the rows of the chosen event name (all of them for "All
        events"); the badge says how many rows show."""
        name = self._type_filter.currentData()
        shown = 0
        for row, ev in enumerate(self._events):
            hide = name is not None and str(ev.get('label', '')) != name
            self._ev_table.setRowHidden(row, hide)
            shown += 0 if hide else 1
        total = len(self._events)
        if name is None:
            self._count_badge.setText(f"{total:,} event" + ("" if total == 1 else "s"))
        else:
            self._count_badge.setText(f"{shown:,} of {total:,}")
        self._no_items_lbl.setVisible(shown == 0)
        self._no_items_lbl.setText("No events" if total == 0 else "No events of this type")
        self._no_items_lbl.setGeometry(self._ev_table.viewport().rect())
        row = self._selected_row()
        if row >= 0 and self._ev_table.isRowHidden(row):
            self._ev_table.clearSelection()
            self._selected_payload = {}
            self._show_selected(-1)
        elif row >= 0:
            self._show_selected(row)     # its "Event n of m" in the new list

    def _visible_rows(self) -> List[int]:
        return [r for r in range(len(self._events)) if not self._ev_table.isRowHidden(r)]

    # ---- selection / navigation --------------------------------------------

    def _show_selected(self, row: int):
        """The Navigate card's SELECTED EVENT box."""
        if row < 0 or row >= len(self._events):
            self._sel_name.setText("No event selected")
            self._sel_time.setText("")
            self._sel_meta.setText("Click an event to move Replay to it")
            return
        ev = self._events[row]
        visible = self._visible_rows()
        pos = visible.index(row) + 1 if row in visible else row + 1
        self._sel_name.setText(str(ev.get('label', '')))
        self._sel_time.setText(_format_system_time(self._session_dir, float(ev.get('timestamp', 0.0) or 0.0)))
        conf = float(ev.get('confidence', 0.0) or 0.0)
        self._sel_meta.setText(f"Event {pos:,} of {len(visible):,}  \u00b7  {ev.get('source', 'analysis')}"
                               f"  \u00b7  Conf. {conf:.2f}")

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
        self._ev_table.scrollToItem(self._ev_table.item(row, 0))
        self._selected_payload = dict(self._events[row] or {})
        self._show_selected(row)
        self.seek_requested.emit(float(self._events[row].get('timestamp', 0.0) or 0.0))

    def _go_prev_event(self):
        """The shown row before the selected one (the first when none is)."""
        rows = self._visible_rows()
        if not rows:
            return
        row = self._selected_row()
        if row < 0 or row not in rows:
            self._select_and_seek(rows[0])
        else:
            self._select_and_seek(rows[max(0, rows.index(row) - 1)])

    def _go_next_event(self):
        """The shown row after the selected one (the first when none is)."""
        rows = self._visible_rows()
        if not rows:
            return
        row = self._selected_row()
        if row < 0 or row not in rows:
            self._select_and_seek(rows[0])
        else:
            self._select_and_seek(rows[min(len(rows) - 1, rows.index(row) + 1)])

    def _remove_selected_event(self):
        row = self._selected_row()
        if row < 0 or row >= len(self._events):
            return
        self._events.pop(row)
        self._ev_table.removeRow(row)
        self._selected_payload = {}
        self._rebuild_type_filter()
        # the next shown row takes its place (else the one before it)
        rows = self._visible_rows()
        if rows:
            after = [r for r in rows if r >= row]
            self._select_and_seek(after[0] if after else rows[-1])
        else:
            self._show_selected(-1)

    def _remove_all_events(self):
        self._events = []
        self._selected_payload = {}
        self._ev_table.setRowCount(0)
        self._rebuild_type_filter()
        self._show_selected(-1)
