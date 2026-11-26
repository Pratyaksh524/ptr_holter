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

def _seg_btn_style(active: bool) -> str:
    """One button of a segmented toggle (its group frame draws the outline)."""
    if active:
        return (f"QPushButton{{background:{UI_ACCENT};color:#FFFFFF;border:none;border-radius:6px;"
                "padding:4px 14px;font-size:12px;font-weight:bold;}")
    return (f"QPushButton{{background:transparent;color:{UI_MUTED};border:none;border-radius:6px;"
            "padding:4px 14px;font-size:12px;font-weight:bold;}"
            f"QPushButton:hover{{color:{UI_TEXT};background:#1A2C49;}}")


# 5. HOLTER HRV PANEL
class HolterHRVPanel(QWidget):
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

        tab_row = QHBoxLayout()
        self._hrv_event_btn = QPushButton("HRV Event")
        self._hrv_event_btn.setStyleSheet(_seg_btn_style(True))
        self._hrv_trend_btn = QPushButton("HRV Tendency")
        self._hrv_trend_btn.setStyleSheet(_seg_btn_style(False))
        # one segmented toggle (a rounded frame holding the two buttons)
        view_group = QFrame()
        view_group.setObjectName("hrvViewGroup")
        view_group.setStyleSheet(f"QFrame#hrvViewGroup{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                                 f"border-radius:8px;}}")
        vg_l = QHBoxLayout(view_group)
        vg_l.setContentsMargins(3, 3, 3, 3)
        vg_l.setSpacing(2)
        for b in (self._hrv_event_btn, self._hrv_trend_btn):
            b.setCursor(Qt.PointingHandCursor)
            b.setFixedHeight(26)
            vg_l.addWidget(b)
        tab_row.addWidget(view_group)
        tab_row.setSpacing(6)
        tab_row.addStretch()
        layout.addLayout(tab_row)

        table_card = QFrame()
        table_card.setObjectName("hrvTableCard")
        table_card.setStyleSheet(_card_style("QFrame#hrvTableCard"))
        table_card_l = QVBoxLayout(table_card)
        table_card_l.setContentsMargins(10, 10, 10, 10)
        table_card_l.setSpacing(8)
        ttl_row = QHBoxLayout()
        table_ttl = QLabel("HRV Analysis")
        table_ttl.setStyleSheet(_card_title_style())
        ttl_row.addWidget(table_ttl)
        ttl_row.addStretch(1)
        # the recording's span (set in update_hrv)
        self._span_chip = QLabel("")
        self._span_chip.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:bold;background:{UI_PANEL_ALT};"
                                      f"border:1px solid {UI_BORDER};border-radius:6px;padding:3px 10px;")
        self._span_chip.hide()
        ttl_row.addWidget(self._span_chip)
        table_card_l.addLayout(ttl_row)

        cols = ["Type", "Start at", "Duration", "Mean NN", "SDNN", "SDANN", "TRIIDX", "pNN50", "LF", "HF", "LF/HF", "Status"]
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
        table_card_l.addWidget(self._table, 1)
        layout.addWidget(table_card, 1)

        # Bottom figures: one tile each, time domain | frequency domain
        stats_frame = QFrame()
        stats_frame.setObjectName("hrvStatsCard")
        stats_frame.setStyleSheet(_card_style("QFrame#hrvStatsCard"))
        stats_row = QHBoxLayout(stats_frame)
        stats_row.setContentsMargins(12, 10, 12, 10)
        stats_row.setSpacing(16)
        self._summary_labels = {}
        groups = [
            ("TIME DOMAIN", 4, [("NNs", "nns"), ("Mean NN", "mean_nn"), ("SDNN", "sdnn"), ("SDANN", "sdann"),
                                ("rMSSD", "rmssd"), ("pNN50", "pnn50"), ("TRIIDX", "triidx"),
                                ("SDNNIDX", "sdnnidx")]),
            ("FREQUENCY DOMAIN", 2, [("VLF", "vlf"), ("LF", "lf"), ("HF", "hf"), ("LF/HF", "lf_hf_ratio")]),
        ]
        for g, (caption, n_cols, stat_defs) in enumerate(groups):
            if g:
                sep = QFrame()
                sep.setFixedWidth(1)
                sep.setStyleSheet(f"background:{UI_BORDER};border:none;")
                stats_row.addWidget(sep)
            box = QVBoxLayout()
            box.setSpacing(6)
            cap = QLabel(caption)
            cap.setStyleSheet(f"color:{UI_MUTED};font-size:9px;font-weight:700;letter-spacing:1px;"
                              f"border:none;background:transparent;")
            box.addWidget(cap)
            grid = QGridLayout()
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(8)
            for i, (label, key) in enumerate(stat_defs):
                tile = QFrame()
                tile.setObjectName("hrvStatTile")
                tile.setStyleSheet(f"QFrame#hrvStatTile{{background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};"
                                   f"border-radius:8px;}}")
                tile_l = QVBoxLayout(tile)
                tile_l.setContentsMargins(12, 6, 12, 6)
                tile_l.setSpacing(1)
                lbl = QLabel(label)      # HRV names keep their own case (rMSSD, pNN50, NNs)
                lbl.setStyleSheet(f"color:{UI_MUTED};font-size:10px;font-weight:bold;border:none;background:transparent;")
                val = QLabel("-")
                val.setStyleSheet(f"color:{UI_TEXT};font-size:16px;font-weight:bold;border:none;background:transparent;")
                tile_l.addWidget(lbl)
                tile_l.addWidget(val)
                grid.addWidget(tile, i // n_cols, i % n_cols)
                self._summary_labels[key] = val
            box.addLayout(grid)
            stats_row.addLayout(box, 2 if g == 0 else 1)
        layout.addWidget(stats_frame)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        for lbl in ["Insert", "Reset", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn.setCursor(Qt.PointingHandCursor)
            btn_row.addWidget(btn, 1)
        layout.addLayout(btn_row)

    def update_hrv(self, metrics_list: list, summary: dict, session_dir: str = None):
        """`session_dir` gives "Start at" its system time (the recording's
        start; each hour's start); without it "-" / the elapsed "00:00"."""
        hourly: dict = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        rows = []
        all_rr = [m.get('rr_ms', 0) for m in metrics_list if m.get('rr_ms', 0) > 0]
        total_duration_sec = int(_metrics_duration_sec(metrics_list)) if metrics_list else 0
        entire_mean_nn = None
        if all_rr:
            from ..hrv_metrics import compute_hrv_summary
            hrv = compute_hrv_summary(all_rr)
            entire_mean_nn = int(np.mean(all_rr))
            rows.append(("Entire", _format_system_time(session_dir, 0.0) if session_dir else "-",
                         _sec_to_hms(total_duration_sec),
                         f"{int(np.mean(all_rr))}ms", f"{hrv.get('sdnn', summary.get('sdnn', 0)):.0f}ms",
                         f"{hrv.get('sdnn', summary.get('sdnn', 0))*0.82:.0f}ms", f"{hrv.get('triangular_index', 0.0):.2f}",
                         f"{hrv.get('pnn50', summary.get('pnn50', 0)):.2f}%",
                         f"{hrv.get('lf', 0.0):.3f}", f"{hrv.get('hf', 0.0):.3f}",
                         f"{hrv.get('lf_hf_ratio', 0.0):.3f}", ""))
        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            rr_vals = [c.get('rr_ms', 0) for c in chunks if c.get('rr_ms', 0) > 0]
            rr_stds = [c.get('rr_std', 0) for c in chunks if c.get('rr_std', 0) > 0]
            pnn50s = [c.get('pnn50', 0) for c in chunks]
            if not rr_vals: continue
            from ..hrv_metrics import compute_hrv_summary
            hrv = compute_hrv_summary(rr_vals)
            # its own length: the recording's last hour usually ends early
            hour_len = max(0, min(3600, total_duration_sec - h * 3600)) if total_duration_sec else 3600
            rows.append(("Hour", _format_system_time(session_dir, h * 3600.0) if session_dir else f"{h:02d}:00",
                         _sec_to_hms(hour_len),
                         f"{int(np.mean(rr_vals))}ms",
                         f"{hrv.get('sdnn', 0):.0f}ms",
                         f"{hrv.get('sdnn', 0)*0.82:.0f}ms",
                         f"{hrv.get('triangular_index', 0.0):.2f}",
                         f"{hrv.get('pnn50', np.mean(pnn50s) if pnn50s else 0.0):.2f}%",
                         f"{hrv.get('lf', 0.0):.3f}", f"{hrv.get('hf', 0.0):.3f}",
                         f"{hrv.get('lf_hf_ratio', 0.0):.3f}", ""))

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                # numbers centred under their header; the type bold (was green)
                item.setTextAlignment((Qt.AlignLeft if j == 0 else Qt.AlignCenter) | Qt.AlignVCenter)
                item.setForeground(QColor(UI_TEXT))
                if j == 0:
                    f = item.font(); f.setBold(True); item.setFont(f)
                self._table.setItem(i, j, item)
        # the recording's own length for the end (as the window header); the
        # live chunks' span when the summary has none
        span_sec = float((summary or {}).get('duration_sec', 0) or 0) or float(total_duration_sec)
        if session_dir and span_sec > 0:
            self._span_chip.setText(f"{_format_system_time(session_dir, 0.0)} \u2192 "
                                    f"{_format_system_time(session_dir, span_sec)}")
            self._span_chip.show()
        else:
            self._span_chip.hide()

        s = summary
        for key, fmt in [("nns", str(s.get('total_beats', 0))),
                          # the table's Mean NN (was the average HR with "ms")
                          ("mean_nn", f"{entire_mean_nn}ms" if entire_mean_nn else "-"),
                          ("sdnn", f"{s.get('sdnn', 0):.0f}ms"),
                          ("sdann", f"{s.get('sdnn', 0)*0.82:.0f}ms"),
                          ("rmssd", f"{s.get('rmssd', 0):.0f}ms"),
                          ("pnn50", f"{s.get('pnn50', 0):.2f}%"),
                          ("triidx", f"{s.get('triidx', 0.0):.2f}" if s.get('triidx', 0) else "-"),
                          ("sdnnidx", "-"),
                          ("vlf", f"{s.get('vlf_power', 0.0):.3f}" if s.get('vlf_power', 0) else "-"),
                          ("lf", f"{s.get('lf_power', 0.0):.3f}" if s.get('lf_power', 0) else "-"),
                          ("hf", f"{s.get('hf_power', 0.0):.3f}" if s.get('hf_power', 0) else "-"),
                          ("lf_hf_ratio", f"{s.get('lf_hf_ratio', 0.0):.3f}" if s.get('lf_hf_ratio', 0) else "-")]:
            if key in self._summary_labels:
                self._summary_labels[key].setText(fmt)


