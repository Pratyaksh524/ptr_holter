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

def _episode_avg_hr(beat_times, start_sec: float, end_sec: float) -> Optional[float]:
    """Average HR (bpm) of an episode from the Full Disclosure beats inside it:
    beat-to-beat pairs counted / the time they span (as the report's HR)."""
    import bisect
    i = bisect.bisect_left(beat_times, start_sec)
    j = bisect.bisect_right(beat_times, end_sec)
    inside = beat_times[i:j]
    if len(inside) < 2 or inside[-1] <= inside[0]:
        return None
    return 60.0 * (len(inside) - 1) / (inside[-1] - inside[0])


# 12. HOLTER AF ANALYSIS PANEL
class HolterAFPanel(QWidget):
    def __init__(self, parent=None, session_dir: str = ""):
        super().__init__(parent)
        self.session_dir = session_dir
        self.setStyleSheet(f"background:{UI_BG};")
        self._replay_engine = None
        self._duration_sec = 0.0
        self._af_rows = []
        self._auto_segments_ready_for = None
        self._auto_segments_pending_for = None
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
        left.setObjectName("afLeftCard")
        left.setAttribute(Qt.WA_StyledBackground, True)
        left.setStyleSheet(_card_style("QWidget#afLeftCard"))
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        # Title, and how many episodes the list holds on the right.
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title = QLabel("AF Analysis")
        title.setStyleSheet(_card_title_style())
        title_row.addWidget(title)
        title_row.addStretch(1)
        self._count_badge = QLabel("0 episodes")
        self._count_badge.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:700;background:{UI_PANEL_ALT};"
                                        f"border:1px solid {UI_BORDER};border-radius:10px;padding:2px 10px;")
        title_row.addWidget(self._count_badge)
        left_layout.addLayout(title_row)

        # Summary of the listed episodes (same sums as the report's
        # Atrial Fibrillation / Flutter box: total, longest, burden).
        stats = QGridLayout()
        stats.setHorizontalSpacing(8)
        stats.setVerticalSpacing(8)
        self._af_stats = {}
        for i, (key, lbl) in enumerate([("count", "AF / AFL"), ("total", "TOTAL TIME"),
                                        ("longest", "LONGEST"), ("burden", "BURDEN")]):
            tile = QFrame()
            tile.setObjectName("afStatTile")
            tile.setStyleSheet(f"QFrame#afStatTile{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                               f"border-radius:8px;}}")
            tile_l = QVBoxLayout(tile)
            tile_l.setContentsMargins(12, 6, 12, 6)
            tile_l.setSpacing(1)
            cap = QLabel(lbl)
            cap.setStyleSheet(f"color:{UI_MUTED};font-size:9px;font-weight:700;letter-spacing:1px;"
                              f"border:none;background:transparent;")
            val = QLabel("--")
            val.setStyleSheet(f"color:{UI_TEXT};font-size:16px;font-weight:bold;border:none;background:transparent;")
            tile_l.addWidget(cap)
            tile_l.addWidget(val)
            stats.addWidget(tile, 0, i)
            self._af_stats[key] = val
        left_layout.addLayout(stats)

        cols = ["#", "Start time", "Duration", "Type", "Avg HR", "Source"]
        self._af_table = QTableWidget(0, len(cols))
        self._af_table.setHorizontalHeaderLabels(cols)
        hdr = self._af_table.horizontalHeader()
        for c, mode in enumerate([QHeaderView.ResizeToContents, QHeaderView.Stretch, QHeaderView.ResizeToContents,
                                  QHeaderView.Stretch, QHeaderView.ResizeToContents, QHeaderView.ResizeToContents]):
            hdr.setSectionResizeMode(c, mode)
        hdr.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._af_table.setAlternatingRowColors(True)
        self._af_table.setShowGrid(False)
        self._af_table.setStyleSheet(_table_style())
        self._af_table.verticalHeader().setVisible(False)
        self._af_table.verticalHeader().setDefaultSectionSize(30)
        self._af_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._af_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._af_table.setCursor(Qt.PointingHandCursor)
        self._af_table.cellClicked.connect(self._on_af_row_clicked)
        left_layout.addWidget(self._af_table, 1)

        # Empty state, centred over the empty table (kept centred on resize).
        no_items = QLabel("No AF / Atrial Flutter episodes\n"
                          "Auto-detected and manually marked episodes appear here")
        no_items.setParent(self._af_table.viewport())
        no_items.setAlignment(Qt.AlignCenter)
        no_items.setWordWrap(True)
        no_items.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        no_items.setStyleSheet(f"color:{UI_MUTED};font-size:12px;font-style:italic;padding:8px;"
                               f"border:none;background:transparent;")
        self._no_items_lbl = no_items
        self._af_table.viewport().installEventFilter(self)

        # Actions: step through events, remove.
        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        for lbl in ["Prev Event", "Next Event", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn.setCursor(Qt.PointingHandCursor)
            if lbl == "Prev Event":
                btn.clicked.connect(lambda _=False: self._step_episode(-1))
            elif lbl == "Next Event":
                btn.clicked.connect(lambda _=False: self._step_episode(1))
            nav_row.addWidget(btn, 1)
        left_layout.addLayout(nav_row)
        layout.addWidget(left, 2)

        # Right: ECG strip + Lorenz
        right = QWidget()
        right.setObjectName("afRightCard")
        right.setAttribute(Qt.WA_StyledBackground, True)
        right.setStyleSheet(_card_style("QWidget#afRightCard"))
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)

        ecg_ttl = QLabel("12-Lead ECG")
        ecg_ttl.setStyleSheet(_card_title_style())
        ttl_row = QHBoxLayout()
        ttl_row.setSpacing(8)
        ttl_row.addWidget(ecg_ttl)
        ttl_row.addStretch(1)
        self._episode_chip = QLabel("")
        self._episode_chip.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:bold;background:{UI_PANEL_ALT};"
                                         f"border:1px solid {UI_BORDER};border-radius:6px;padding:3px 10px;")
        self._episode_chip.hide()
        ttl_row.addWidget(self._episode_chip)
        right_layout.addLayout(ttl_row)

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
        scroll_content.setStyleSheet("background:transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(6, 6, 6, 6)
        scroll_layout.setSpacing(8)

        self._thumb_strips = []
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for i in range(12):
            # One block per lead: "Lead X" on a slim header line, the strip under it.
            block = QFrame()
            block.setObjectName("afLeadBlock")
            block.setStyleSheet(f"QFrame#afLeadBlock{{background:transparent;border:none;"
                                f"border-bottom:1px solid {UI_BORDER};}}")
            block_l = QVBoxLayout(block)
            block_l.setContentsMargins(0, 0, 0, 6)
            block_l.setSpacing(2)
            lbl = QLabel(f"Lead {lead_names[i]}")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;background:transparent;"
                              f"border:none;border-left:2px solid {COL_GREEN};padding:0px 6px;")
            block_l.addWidget(lbl)

            # Disable vertical lines and labels for AF Analysis panel
            strip = ECGStripCanvas(height=120, show_vertical_lines=False)
            strip.lead_name = lead_names[i]
            block_l.addWidget(strip)
            scroll_layout.addWidget(block)
            self._thumb_strips.append(strip)

        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area, 1)

        # Bottom: Lead II rhythm strip, full width.
        def _section(title_text, widget):
            box = QFrame()
            box.setObjectName("afBottomBox")
            box.setStyleSheet(f"QFrame#afBottomBox{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                              f"border-radius:6px;}}")
            box_l = QVBoxLayout(box)
            box_l.setContentsMargins(8, 6, 8, 8)
            box_l.setSpacing(4)
            cap = QLabel(title_text)
            cap.setStyleSheet(_card_title_style())
            box_l.addWidget(cap)
            box_l.addWidget(widget, 1)
            return box

        # Disable vertical lines for bottom ECG strip too
        self._af_ecg_strip = ECGStripCanvas(height=70, show_vertical_lines=False)
        self._af_ecg_strip.setMinimumHeight(90)
        self._af_ecg_strip.setMaximumHeight(130)
        self._af_ecg_strip.setMinimumWidth(120)
        self._af_ecg_strip.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(_section("Rhythm Strip  \u00b7  Lead II", self._af_ecg_strip))
        layout.addWidget(right, 3)

    def eventFilter(self, obj, event):
        # Keep the empty-state text centred over the table's viewport.
        lbl = getattr(self, "_no_items_lbl", None)
        if lbl is not None and obj is self._af_table.viewport() and event.type() == QEvent.Resize:
            lbl.setGeometry(obj.rect())
        return super().eventFilter(obj, event)

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        # the folder of the recording the engine plays: the panel is built
        # once, and session_dir from then was another (or no) recording's
        path = getattr(getattr(engine, "_reader", None), "path", None)
        if path:
            self.session_dir = os.path.dirname(os.path.abspath(path))
        self._auto_segments_ready_for = None
        self._shown_af_start = None      # another recording: its first episode is selected
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
            # First analysis of a long recording takes a while, so without a
            # cached result it runs in the background and the table refreshes
            # when it's done, instead of blocking the recording from opening.
            if self._auto_segments_pending_for is not engine:
                try:
                    from ..auto_segment_arrthymia_detection import (
                        detect_arrhythmias_async, get_cached_detection,
                    )
                    # with the doctor's label changes (auto_segment_edits)
                    from ..auto_segment_edits import apply_auto_segments_with_edits as _apply_with_edits
                    path = getattr(getattr(engine, "_reader", None), "path", None)
                    cached = get_cached_detection(path) if path else None
                    if cached is not None or not path:
                        _apply_with_edits(engine, segments=cached)
                        self._auto_segments_ready_for = engine
                    else:
                        self._auto_segments_pending_for = engine

                        def _on_ready(segs, eng=engine):
                            if self._replay_engine is not eng:
                                return
                            _apply_with_edits(eng, segments=segs)
                            self._auto_segments_ready_for = eng
                            self._auto_segments_pending_for = None
                            self._refresh_af_events()

                        detect_arrhythmias_async(path, _on_ready)
                except Exception as e:
                    print(f"[HolterAFPanel] Auto segment detection unavailable: {e}")
                    self._auto_segments_ready_for = engine
            if self._auto_segments_ready_for is not engine:
                return segments  # still analysing; stored live labels aren't shown meanwhile
        for ev in (getattr(engine, "_structured_events", []) or []):
            if ev.get("end_timestamp") is None:
                continue
            af_type = self._af_type_from_label(ev.get("label", ev.get("type", "")))
            if af_type is None:
                continue
            start_sec = float(ev.get("timestamp", 0.0) or 0.0)
            end_sec = float(ev.get("end_timestamp", start_sec) or start_sec)
            if end_sec > start_sec:
                segments.append({"start_sec": start_sec, "end_sec": end_sec, "type": af_type,
                                 "source": "Auto (Edited)" if ev.get("edited") else "Auto"})
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
                        segments.append({"start_sec": start_sec, "end_sec": end_sec, "type": af_type,
                                         "source": "Manual"})
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
        self._count_badge.setText(f"{len(segments)} episode" + ("" if len(segments) == 1 else "s"))
        self._update_af_stats(segments)
        beat_times = self._full_disclosure_beats() if segments else None
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
                hr = _episode_avg_hr(beat_times, start_sec, seg["end_sec"]) if beat_times else None
                source = seg.get("source", "Auto")
                for j, val in enumerate([str(i + 1), start_str, duration_str, seg["type"],
                                         f"{hr:.0f} bpm" if hr else "--", source]):
                    item = QTableWidgetItem(val)
                    muted = j == 0 or (j == 5 and source == "Auto") or (j == 4 and not hr)
                    item.setForeground(QColor(UI_MUTED if muted else COL_WHITE))
                    if j == 1:
                        item.setData(Qt.UserRole, start_sec)
                    self._af_table.setItem(i, j, item)
            self._select_af_row_after_refresh(segments)
        else:
            self._no_items_lbl.show()
            self._shown_af_start = None
            self._episode_chip.hide()

    def _update_af_stats(self, segments):
        """The tiles: AF / AFL episodes, their total and longest time, and
        the burden (total / recording) -- "--" with no episode, as the
        report's "-"."""
        n_af = sum(1 for s in segments if s["type"] == "Atrial Fibrillation")
        n_afl = sum(1 for s in segments if s["type"] == "Atrial Flutter")
        durations = [max(0.0, s["end_sec"] - s["start_sec"]) for s in segments]
        engine = self._replay_engine
        rec = self._duration_sec or float(getattr(engine, "duration_sec", 0.0) or 0.0)
        self._af_stats["count"].setText(f"{n_af} / {n_afl}")
        if durations:
            total = sum(durations)
            self._af_stats["total"].setText(_sec_to_hms(total))
            self._af_stats["longest"].setText(_sec_to_hms(max(durations)))
            self._af_stats["burden"].setText(f"{100.0 * total / rec:.2f}%" if rec > 0 else "--")
        else:
            for key in ("total", "longest", "burden"):
                self._af_stats[key].setText("--")

    def _full_disclosure_beats(self):
        """The Full Disclosure strip's beat times (s) saved with the
        recording's analysis; None when they are not saved (Avg HR "--")."""
        if not self.session_dir:
            return None
        try:
            from ..report_generator import _full_disclosure_beat_times
            return _full_disclosure_beat_times(self.session_dir, compute=False)
        except Exception as e:
            print(f"[HolterAFPanel] Full Disclosure beats unavailable: {e}")
            return None

    def _step_episode(self, step: int):
        """Prev / Next Event: the episode before / after the selected one
        (the first when none is), its waves on the right."""
        n = len(self._af_rows)
        if not n:
            return
        rows = self._af_table.selectionModel().selectedRows()
        cur = rows[0].row() if rows else -1
        new = 0 if cur < 0 else max(0, min(n - 1, cur + step))
        self._af_table.selectRow(new)
        self._on_af_row_clicked(new, 0)

    def _update_episode_chip(self, row: int):
        """Over the 12 leads, which episode the waves are from: 'Episode 2
        of 3 . Atrial Flutter . 13:56:26 -> 13:57:01 . first 10 s shown'."""
        if row < 0 or row >= len(self._af_rows):
            self._episode_chip.hide()
            return
        seg = self._af_rows[row]
        start = self._recording_start_timestamp()

        def at(sec):
            if start:
                return datetime.fromtimestamp(start + sec).strftime('%H:%M:%S')
            return _sec_to_hms(sec)
        shown = min(10.0, seg["end_sec"] - seg["start_sec"])
        self._episode_chip.setText(f"Episode {row + 1} of {len(self._af_rows)}  \u00b7  {seg['type']}  \u00b7  "
                                   f"{at(seg['start_sec'])} \u2192 {at(seg['end_sec'])}  \u00b7  "
                                   f"first {shown:.0f} s shown")
        self._episode_chip.show()

    def _select_af_row_after_refresh(self, segments):
        """After the table is filled: the episode whose waves are on the
        right stays selected when it is still listed; otherwise (the tab's
        first fill, another recording) the first episode is selected and its
        waves are loaded -- was nothing selected and the right side showing
        whatever second the replay stood on (at 13:54:02, 1 s before the
        first AF of 2026-11-12_13-54-02, it only looked like AF)."""
        shown = getattr(self, "_shown_af_start", None)
        if shown is not None:
            for i, seg in enumerate(segments):
                if abs(float(seg["start_sec"]) - shown) < 1e-6:
                    self._af_table.selectRow(i)
                    self._update_episode_chip(i)
                    return
        self._af_table.selectRow(0)
        self._on_af_row_clicked(0, 0)

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
                self._show_frame(data)
                self._shown_af_start = start_sec
                self._update_episode_chip(row)
        except Exception as e:
            print(f"[HolterAFPanel] Error loading AF strip: {e}")

    def set_replay_frame(self, data):
        """The replay's current frame (HolterMainWindow._broadcast_replay_frame).
        Ignored while an AF episode's waves are shown (_shown_af_start): the
        replay frame arriving after the tab opened replaced the selected
        episode's waves with whatever second the replay stood on."""
        if getattr(self, "_shown_af_start", None) is not None:
            return
        self._show_frame(data)

    def _show_frame(self, data):
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


