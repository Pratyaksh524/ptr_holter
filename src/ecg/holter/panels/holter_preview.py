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

# 9. HOLTER PREVIEW PANEL
class HolterInsightPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pvCard")
        self.setStyleSheet(_card_style("QFrame#pvCard"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title = QLabel("Comprehensive Report Preview")
        title.setStyleSheet(f"color:{UI_TEXT};font-size:14px;font-weight:bold;background:transparent;"
                            f"padding:4px 8px;border:none;border-left:3px solid {UI_ACCENT};")
        layout.addWidget(title)
        self._report = QTextEdit()
        self._report.setReadOnly(True)
        self._report.setStyleSheet(f"""
            QTextEdit{{background:{UI_BG};color:{UI_TEXT};
              border:1px solid {UI_BORDER};border-radius:8px;padding:10px;font-size:13px;
              selection-background-color:#1E3A5F;selection-color:{UI_TEXT};}}
        """)
        layout.addWidget(self._report)

    def update_text(self, patient_info: dict, summary: dict):
        name = patient_info.get("patient_name") or patient_info.get("name") or "Unknown patient"
        age = patient_info.get("age", "-")
        sex = patient_info.get("gender") or patient_info.get("sex") or "-"
        email = patient_info.get("email", "-")
        dur = summary.get("duration_sec", 0) / 3600
        avg_hr = summary.get("avg_hr", 0)
        min_hr = summary.get("min_hr", 0)
        max_hr = summary.get("max_hr", 0)
        quality = summary.get("avg_quality", 0) * 100
        arrhythmias = summary.get("arrhythmia_counts", {})
        top = ", ".join(f"{k} ({v})" for k, v in sorted(arrhythmias.items(), key=lambda x: -x[1])[:4]) \
              or "No clinically significant arrhythmia burden detected."
        rhythm = ("predominantly tachycardic trend" if avg_hr >= 100
                  else "predominantly bradycardic trend" if 0 < avg_hr <= 60
                  else "predominantly sinus-range rhythm")
        text = (
            f"Patient: {name} | Age/Sex: {age}/{sex} | Email: {email}\n\n"
            f"Study summary:\n"
            f"- Recording duration: {dur:.1f} hours\n"
            f"- Average heart rate: {avg_hr:.0f} bpm (range {min_hr:.0f}-{max_hr:.0f} bpm)\n"
            f"- Signal quality: {quality:.1f}%\n"
            f"- Longest RR interval: {summary.get('longest_rr_ms',0):.0f} ms\n"
            f"- HRV profile: SDNN {summary.get('sdnn',0):.1f} ms, "
            f"rMSSD {summary.get('rmssd',0):.1f} ms, pNN50 {summary.get('pnn50',0):.2f}%\n\n"
            f"Interpretation:\n"
            f"The recording demonstrates a {rhythm}. Key events: {top}\n\n"
            f"Suggested final report wording:\n"
            f'"Comprehensive ECG Analysis monitoring for {name} shows {rhythm} with an average heart rate of '
            f'{avg_hr:.0f} bpm. The minimum recorded rate was {min_hr:.0f} bpm and the '
            f'maximum recorded rate was {max_hr:.0f} bpm. Overall signal quality was '
            f'{quality:.1f}%, enabling comprehensive review of the 12-lead trends and event strips."'
        )
        self._report.setPlainText(text)


