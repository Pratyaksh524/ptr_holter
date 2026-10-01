"""
ecg/holter/holter_ui.py
========================
Complete Holter Monitor UI - Professional Medical Software
Matches reference images: black/green medical workstation style.

Screens:
  1. HolterStartDialog        - patient info + duration + start
  2. HolterStatusBar          - REC indicator, elapsed, live BPM, arrhythmia ticker
  3. HolterSummaryCards       - KPI cards (Avg HR, Min/Max, Beats, Pauses, Quality, SDNN)
  4. HolterOverviewPanel      - full stats table (Name/Value pairs)
  5. HolterHRVPanel           - HRV table per hour + bottom stats strip
  6. HolterReplayPanel        - RR scatter/Lorenz + scrub slider + ECG strip
  7. HolterEventsPanel        - Arrhythmia events list with strip nav
  8. HolterWaveGridPanel      - 12-lead live/replay grid (3 rows  -- 4 cols)
  9. HolterInsightPanel       - Comprehensive report preview narrative
 10. HolterRecordManagementPanel - searchable session browser
 11. HolterHistogramPanel     - RR-interval histogram
 12. HolterAFPanel            - AF episode browser
 13. HolterSTPanel            - ST tendency per channel
 14. HolterEditEventPanel     - Edit events with strip thumbnails
 15. HolterEditStripsPanel    - Edit strips (max HR, min HR, sinus max/min thumbnails)
 16. HolterReportTablePanel   - Hour-by-hour report table
 17. HolterMainWindow         - Orchestrates all panels in tabbed layout
"""

import os
import sys
import json
import time
import math
import shutil
import gc
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QLineEdit, QComboBox, QSlider, QGroupBox, QFrame,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QScrollArea, QGridLayout, QSpinBox, QMessageBox,
    QFileDialog, QApplication, QProgressBar, QSplitter, QTextEdit, QInputDialog, QDoubleSpinBox,
    QAbstractItemView, QToolButton, QButtonGroup, QMenu, QScrollBar, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint, QPointF, QRect, QObject, QEvent, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QPen, QBrush, QPixmap, QPainterPath, QIcon

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    pg = None
    HAS_PG = False


def _resolve_recordings_dir(session_dir: str = "") -> str:
    """Return the recordings root directory for the current session or project."""
    normalized = os.path.dirname(session_dir) if os.path.isfile(session_dir) else session_dir
    if normalized and os.path.isdir(normalized):
        if os.path.basename(os.path.normpath(normalized)).lower() == "recordings":
            return normalized
        parent_dir = os.path.dirname(normalized)
        if os.path.basename(os.path.normpath(parent_dir)).lower() == "recordings":
            return parent_dir

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    preferred_dir = os.path.join(src_root, "recordings")
    fallback_dir = os.path.join(os.getcwd(), "recordings")
    if os.path.isdir(preferred_dir):
        return preferred_dir
    if os.path.isdir(fallback_dir):
        return fallback_dir
    return preferred_dir


def _find_latest_completed_session(output_dir: str) -> str:
    """Return the newest completed session directory, or empty string."""
    if not output_dir or not os.path.isdir(output_dir):
        return ""

    candidates = []
    try:
        for name in os.listdir(output_dir):
            session_dir = os.path.join(output_dir, name)
            ecgh_path = os.path.join(session_dir, "recording.ecgh")
            if not os.path.isdir(session_dir):
                continue
            if not os.path.exists(ecgh_path):
                continue
            try:
                sort_key = os.path.getmtime(ecgh_path)
            except Exception:
                sort_key = os.path.getmtime(session_dir)
            candidates.append((sort_key, session_dir))
    except Exception:
        return ""

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _normalize_patient_info(info: Optional[dict]) -> dict:
    normalized = dict(info or {})
    if not normalized:
        return {}

    display_name = ""
    for key in ("patient_name", "name", "full_name", "patientName"):
        raw_value = normalized.get(key)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value and value.lower() not in {"unknown", "unknown patient"}:
            display_name = value
            break

    if display_name:
        normalized["patient_name"] = display_name
        normalized["name"] = display_name
        normalized["full_name"] = display_name
        normalized["patientName"] = display_name

    return normalized


def _load_patient_info_from_session(session_dir: str, fallback_info: Optional[dict] = None) -> dict:
    merged = _normalize_patient_info(fallback_info)
    if not session_dir:
        return merged

    metadata = read_session_metadata(session_dir) if session_dir else {}
    if isinstance(metadata, dict):
        patient_sources = [metadata.get("patient_info")]
        summary = metadata.get("summary")
        if isinstance(summary, dict):
            patient_sources.append(summary.get("patient_info"))
        for candidate in patient_sources:
            if isinstance(candidate, dict) and candidate:
                merged.update(_normalize_patient_info(candidate))

    patient_json = os.path.join(session_dir, "patient.json")
    if os.path.exists(patient_json):
        try:
            with open(patient_json, "r", encoding="utf-8") as handle:
                patient_data = json.load(handle) or {}
            if isinstance(patient_data, dict) and patient_data:
                merged.update(_normalize_patient_info(patient_data))
        except Exception:
            pass

    return _normalize_patient_info(merged)



def _metrics_duration_sec(metrics_list: list) -> float:
    return float(sum(m.get('duration', 0.0) or 0.0 for m in metrics_list))


def _normalize_beat_class(label) -> str:
    """Map recorded beat labels into the compact class buttons used by the UI."""
    raw = str(label or "").strip().upper()
    if not raw:
        return "Other"
    if raw in {"N", "NORMAL", "SINUS"}:
        return "N"
    if raw.startswith("S") or "SV" in raw or "PAC" in raw:
        return "S"
    if raw.startswith("V") or "PVC" in raw or "VENT" in raw:
        return "V"
    if raw.startswith("P") or "PACED" in raw:
        return "P"
    if "AF" in raw or "AFL" in raw or raw in {"F", "Q"}:
        return "AF"
    if raw in {"X", "ART", "ARTIFACT", "NOISE"} or "ARTIFACT" in raw or "NOISE" in raw:
        return "X"
    if raw == "OTHER":
        return "Other"
    return "Other"


def _class_matches_filter(beat_class: str, filter_key: str) -> bool:
    if filter_key == "all":
        return True
    if filter_key == "AF":
        return beat_class == "AF"
    if filter_key == "Other":
        return beat_class == "Other"
    return beat_class == filter_key


def _template_filter_key(label: str) -> str:
    """Normalize template labels into the UI filter keys."""
    return _normalize_beat_class(label)

try:
    from .theme import (
        ADC_TO_MV,
        COL_BEAT_S,
        COL_BG,
        COL_BLACK,
        COL_BTN_ACTIVE_BG,
        COL_BTN_ACTIVE_TEXT,
        COL_DARK,
        COL_GRAY,
        COL_GREEN,
        COL_GREEN_DRK,
        COL_GREEN_MID,
        COL_GRID_MAJOR,
        COL_GRID_MINOR,
        COL_RED,
        COL_TEXT,
        COL_TIMESTAMP,
        COL_WAVE_ORANGE,
        COL_WAVE_RED,
        COL_WHITE,
        COL_YELLOW,
        GAINS,
        PAPER_SPEEDS,
        TOOL_CALIPER,
        TOOL_MAGNIFY,
        TOOL_RULER,
        TOOL_SELECT,
    )
    from .tool_engine import (
        ECGToolEngine,
        amplitude_mv_from_pixels,
        caliper_label,
        canonical_tool,
        hint as tool_hint,
        interval_ms_from_pixels,
        ruler_label,
        tool_specs,
        tooltip as tool_tooltip,
    )
    from .session_store import append_annotation, load_annotations, load_events, load_metrics, read_session_metadata
    from .summary_utils import derive_hr_focus_summary
except ImportError:
    from ecg.holter.theme import (
        ADC_TO_MV,
        COL_BEAT_S,
        COL_BG,
        COL_BLACK,
        COL_BTN_ACTIVE_BG,
        COL_BTN_ACTIVE_TEXT,
        COL_DARK,
        COL_GRAY,
        COL_GREEN,
        COL_GREEN_DRK,
        COL_GREEN_MID,
        COL_GRID_MAJOR,
        COL_GRID_MINOR,
        COL_RED,
        COL_TEXT,
        COL_TIMESTAMP,
        COL_WAVE_ORANGE,
        COL_WAVE_RED,
        COL_WHITE,
        COL_YELLOW,
        GAINS,
        PAPER_SPEEDS,
        TOOL_CALIPER,
        TOOL_MAGNIFY,
        TOOL_RULER,
        TOOL_SELECT,
    )
    from ecg.holter.tool_engine import (
        ECGToolEngine,
        amplitude_mv_from_pixels,
        caliper_label,
        canonical_tool,
        hint as tool_hint,
        interval_ms_from_pixels,
        ruler_label,
        tool_specs,
        tooltip as tool_tooltip,
    )
    from ecg.holter.session_store import append_annotation, load_annotations, load_events, load_metrics, read_session_metadata
    from ecg.holter.summary_utils import derive_hr_focus_summary

# Professional UI palette (kept separate from signal colors).
UI_BG = "#0B1220"
UI_PANEL = "#0F1A2E"
UI_PANEL_ALT = "#13213A"
UI_CARD = "#101B2F"
UI_BORDER = "#243552"
UI_TEXT = "#E6EDF7"
UI_MUTED = "#9AAECB"
UI_ACCENT = "#2F80ED"
UI_ACCENT_HOVER = "#4B96FA"
UI_SUCCESS = "#16C172"
UI_WARNING = "#F59E0B"


def _style_btn(bg=UI_PANEL_ALT, fg=UI_TEXT, hover="#1A2C49"):
    return f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border: 1px solid {UI_BORDER};
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background: {hover};
            color: {UI_TEXT};
        }}
        QPushButton:pressed {{ background: {bg}; border: 1px solid {UI_ACCENT_HOVER}; }}
        QPushButton:disabled {{ background: #1A2233; color: #5F708A; border: 1px solid #2A3953; }}
    """


def _style_active_btn():
    return f"""
        QPushButton {{
            background: {UI_ACCENT};
            color: {UI_TEXT};
            border: 1px solid #5EA4FF;
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background: {UI_ACCENT_HOVER}; }}
    """


def _table_style():
    return f"""
        QTableWidget {{
            background: {UI_PANEL};
            alternate-background-color: {UI_PANEL_ALT};
            color: {UI_TEXT};
            gridline-color: {UI_BORDER};
            font-size: 12px;
            border: 1px solid {UI_BORDER};
            selection-background-color: #1E3A5F;
            selection-color: {UI_TEXT};
        }}
        QHeaderView::section {{
            background: {UI_PANEL_ALT};
            color: {UI_MUTED};
            font-size: 11px;
            font-weight: bold;
            padding: 6px;
            border: 1px solid {UI_BORDER};
        }}
        QTableWidget::item {{ padding: 5px; border: none; }}
        QScrollBar:vertical {{
            border: none;
            background: {UI_PANEL_ALT};
            width: 10px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {UI_BORDER};
            min-height: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {UI_MUTED};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}
        QScrollBar:horizontal {{
            border: none;
            background: {UI_PANEL_ALT};
            height: 10px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background: {UI_BORDER};
            min-width: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {UI_MUTED};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
    """


def _show_message_box(parent, icon, title, text, buttons=QMessageBox.Ok, default_button=QMessageBox.NoButton):
    msg_box = QMessageBox(parent)
    msg_box.setIcon(icon)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    msg_box.setStandardButtons(buttons)
    msg_box.setDefaultButton(default_button)
    # Set dark theme style with white text
    msg_box.setStyleSheet(f"""
        QMessageBox {{
            background-color: {UI_PANEL};
        }}
        QMessageBox QLabel {{
            color: {UI_TEXT};
            background-color: transparent;
        }}
        QMessageBox QPushButton {{
            background-color: {UI_PANEL_ALT};
            color: #FFFFFF;
            border: 1px solid {UI_BORDER};
            border-radius: 5px;
            padding: 7px 20px;
            min-width: 70px;
            font-weight: bold;
        }}
        QMessageBox QPushButton:hover {{
            background-color: #1A2C49;
            color: #FFFFFF;
        }}
        QMessageBox QPushButton:pressed {{
            background-color: {UI_BORDER};
            color: #FFFFFF;
        }}
    """)
    return msg_box.exec_()
    return msg_box.exec_()


def _sec_to_hms(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _format_system_time(session_dir: str, chunk_timestamp: float) -> str:
    """Convert relative timestamp to actual system wall-clock time."""
    if not session_dir:
        return _sec_to_hms(chunk_timestamp)
    try:
        index_path = os.path.join(session_dir, 'recording_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            start_time = index_data.get('start_time')
            if start_time:
                system_time = datetime.fromtimestamp(start_time + chunk_timestamp)
                return system_time.strftime('%H:%M:%S')
    except Exception as e:
        print(f"[HolterUI] Error reading system time: {e}")
    return _sec_to_hms(chunk_timestamp)


def _get_recording_start_end_times(session_dir: str, duration_sec: float) -> tuple:
    """Return recording start and end date-time strings."""
    if not session_dir:
        return "Unknown", "Unknown"
    try:
        index_path = os.path.join(session_dir, 'recording_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            start_time = index_data.get('start_time')
            if start_time:
                start_dt = datetime.fromtimestamp(start_time)
                end_dt = datetime.fromtimestamp(start_time + duration_sec)
                return start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"[HolterUI] Error reading recording start/end times: {e}")
    return "Unknown", "Unknown"



# -----------------------------------------------------------------------------
# 1. HOLTER START DIALOG
# -----------------------------------------------------------------------------

class HolterStartDialog(QDialog):
    def __init__(self, parent=None, patient_info: dict = None, output_dir: str = "recordings"):
        super().__init__(parent)
        self.setWindowTitle("Comprehensive ECG Analysis - Setup")
        self.setMinimumWidth(640)
        self.setStyleSheet(f"background: #0F1724; color: {COL_WHITE};")
        self.output_dir = output_dir
        self._result_info = None
        self._result_duration = 24
        self._result_dir = output_dir
        self._build_ui(patient_info or {})

    def _build_ui(self, info: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Comprehensive ECG Analysis - Professional Setup")
        title.setStyleSheet(f"background:{COL_GRAY};color:{COL_GREEN};border:2px solid {COL_GREEN};"
                            f"font-size:20px;font-weight:bold;padding:16px;border-radius:8px;")
        layout.addWidget(title)

        subtitle = QLabel("Enter patient details, choose study duration, and launch the 12-lead ECG workspace.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:#cccccc;font-size:13px;padding:4px;")
        layout.addWidget(subtitle)

        # Patient info group
        pg = QGroupBox("Patient Information")
        pg.setStyleSheet(f"QGroupBox{{font-weight:bold;color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                         f"border-radius:8px;margin-top:12px;padding-top:20px;background:{COL_BLACK};}}")
        pg_layout = QGridLayout(pg)
        pg_layout.setSpacing(10)

        fields = [
            ("Patient Name", "patient_name", info.get("patient_name", "")),
            ("Age", "age", str(info.get("age", ""))),
            ("Email", "email", info.get("email", "")),
            ("Doctor", "doctor", info.get("doctor", "")),
            ("Organisation", "org", info.get("Org.", info.get("org", ""))),
            ("Phone", "phone", info.get("doctor_mobile", info.get("phone", ""))),
        ]
        self._fields = {}
        for row, (label, key, default) in enumerate(fields):
            lbl = QLabel(label + ":")
            lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
            edit = QLineEdit(default)
            edit.setStyleSheet(f"QLineEdit{{border:1px solid {COL_GREEN_DRK};border-radius:4px;padding:8px;"
                               f"font-size:13px;background:{COL_DARK};color:{COL_GREEN};}}"
                               f"QLineEdit:focus{{border-color:{COL_GREEN};}}")
            pg_layout.addWidget(lbl, row, 0)
            pg_layout.addWidget(edit, row, 1)
            self._fields[key] = edit

        lbl_g = QLabel("Gender:")
        lbl_g.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        self._gender = QComboBox()
        self._gender.addItems(["Select", "Male", "Female", "Other"])
        idx = self._gender.findText(info.get("gender", info.get("sex", "Select")))
        if idx >= 0: self._gender.setCurrentIndex(idx)
        self._gender.setStyleSheet(f"""
            QComboBox {{
                border:1px solid {COL_GREEN_DRK}; border-radius:4px; padding:8px;
                background:{COL_DARK}; color:{COL_GREEN};
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        pg_layout.addWidget(lbl_g, len(fields), 0)
        pg_layout.addWidget(self._gender, len(fields), 1)
        layout.addWidget(pg)

        # Recording settings group
        rg = QGroupBox("Recording Settings")
        rg.setStyleSheet(f"QGroupBox{{font-weight:bold;color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                         f"border-radius:8px;margin-top:12px;padding-top:20px;background:{COL_BLACK};}}")
        rg_layout = QGridLayout(rg)
        rg_layout.setSpacing(10)

        dur_lbl = QLabel("Duration:")
        dur_lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        self._duration = QComboBox()
        self._duration.addItems(["24 hours", "48 hours", "Custom"])
        self._duration.setStyleSheet(f"""
            QComboBox {{
                border:1px solid {COL_GREEN_DRK}; border-radius:4px; padding:8px;
                background:{COL_DARK}; color:{COL_GREEN};
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        self._duration.currentTextChanged.connect(lambda t: self._custom_hours.setVisible(t == "Custom"))
        rg_layout.addWidget(dur_lbl, 0, 0)
        rg_layout.addWidget(self._duration, 0, 1)

        self._custom_hours = QSpinBox()
        self._custom_hours.setRange(1, 72)
        self._custom_hours.setValue(24)
        self._custom_hours.setSuffix(" hours")
        self._custom_hours.setVisible(False)
        self._custom_hours.setStyleSheet(f"border:1px solid {COL_GREEN_DRK};border-radius:4px;padding:8px;"
                                         f"font-size:13px;background:{COL_DARK};color:{COL_GREEN};")
        rg_layout.addWidget(self._custom_hours, 1, 1)

        out_lbl = QLabel("Output Directory:")
        out_lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        rg_layout.addWidget(out_lbl, 2, 0)
        dir_row = QHBoxLayout()
        self._dir_label = QLabel(self.output_dir)
        self._dir_label.setStyleSheet(f"font-size:12px;color:{COL_GREEN};")
        dir_row.addWidget(self._dir_label, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(_style_btn())
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        rg_layout.addLayout(dir_row, 2, 1)

        self._rec_count_label = QLabel("")
        self._rec_count_label.setStyleSheet(f"font-size:13px;color:{COL_GREEN};font-weight:700;")
        rg_layout.addWidget(QLabel("Recorded Sessions:"), 3, 0)
        rg_layout.addWidget(self._rec_count_label, 3, 1)
        self._refresh_rec_count()
        layout.addWidget(rg)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(_style_btn(COL_GRAY, COL_WHITE, COL_GREEN_DRK))
        cancel_btn.clicked.connect(self.reject)
        start_btn = QPushButton("Open ECG Workspace")
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #ff6600, stop:1 #e65c00);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 14px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: #ff7a26; }}
            QPushButton:pressed {{ background: #cc5200; }}
        """)
        start_btn.setMinimumHeight(48)
        start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(start_btn, 1)
        layout.addLayout(btn_row)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if d:
            self._result_dir = d
            self._dir_label.setText(d)
            self._refresh_rec_count()

    def _refresh_rec_count(self):
        root = getattr(self, '_result_dir', self.output_dir)
        count = 0
        try:
            if os.path.isdir(root):
                for name in os.listdir(root):
                    if os.path.exists(os.path.join(root, name, "recording.ecgh")):
                        count += 1
        except Exception:
            pass
        self._rec_count_label.setText(f"{count} completed recording(s)")

    def _on_start(self):
        info = {key: field.text().strip() for key, field in self._fields.items()}
        info['gender'] = self._gender.currentText()
        info['sex'] = info['gender']
        info['name'] = info.get('patient_name', 'Unknown')
        info['Org.'] = info.get('org', '')
        if not info.get('patient_name'):
            QMessageBox.warning(self, "Missing Name", "Please enter the patient name.")
            return
        dur_text = self._duration.currentText()
        if dur_text == "24 hours": self._result_duration = 24
        elif dur_text == "48 hours": self._result_duration = 48
        else: self._result_duration = self._custom_hours.value()
        self._result_info = info
        self._result_dir = self._dir_label.text()
        self.accept()

    def get_result(self):
        if self._result_info:
            return self._result_info, self._result_duration, self._result_dir
        return None


# -----------------------------------------------------------------------------
# 2. HOLTER STATUS BAR  (Live recording indicator)
# -----------------------------------------------------------------------------

class HolterStatusBar(QFrame):
    stop_requested = pyqtSignal()

    def __init__(self, parent=None, target_hours: int = 24):
        super().__init__(parent)
        self.target_hours = target_hours
        self._start_time = time.time()
        self._blink_state = True
        self.setFixedHeight(52)
        self.setStyleSheet(f"QFrame{{background:{COL_BLACK};border-bottom:2px solid {COL_GREEN};}}")
        self._build_ui()
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_timer.start(800)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(1000)

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
        layout.setContentsMargins(15, 4, 15, 4)
        layout.setSpacing(18)

        self._rec_label = QLabel("REC")
        self._rec_label.setStyleSheet(f"color:{COL_GREEN};font-size:15px;font-weight:bold;")
        layout.addWidget(self._rec_label)

        self._time_label = QLabel("00:00:00")
        self._time_label.setStyleSheet(f"color:{COL_GREEN};font-size:18px;font-weight:bold;font-family:monospace;")
        layout.addWidget(self._time_label)

        tgt = QLabel(f"/ {self.target_hours:02d}:00:00")
        tgt.setStyleSheet(f"color:{COL_GREEN_DRK};font-size:12px;")
        layout.addWidget(tgt)

        sep = QLabel("|")
        sep.setStyleSheet(f"color:{COL_GREEN_DRK};")
        layout.addWidget(sep)

        bpm_lbl = QLabel("BPM:")
        bpm_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:12px;")
        layout.addWidget(bpm_lbl)
        self._bpm_label = QLabel("-")
        self._bpm_label.setStyleSheet(f"color:{COL_GREEN};font-size:18px;font-weight:bold;")
        layout.addWidget(self._bpm_label)

        sep2 = QLabel("|")
        sep2.setStyleSheet(f"color:{COL_GREEN_DRK};")
        layout.addWidget(sep2)

        ev_lbl = QLabel("Events:")
        ev_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:12px;")
        layout.addWidget(ev_lbl)
        self._arrhy_label = QLabel("None detected")
        self._arrhy_label.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;")
        self._arrhy_label.setMaximumWidth(380)
        layout.addWidget(self._arrhy_label, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, self.target_hours * 3600)
        self._progress.setValue(0)
        self._progress.setFixedWidth(140)
        self._progress.setFixedHeight(12)
        self._progress.setStyleSheet(f"""
            QProgressBar{{background:{COL_DARK};border-radius:6px;border:1px solid {COL_GREEN_DRK};}}
            QProgressBar::chunk{{background:{COL_GREEN};border-radius:5px;}}
        """)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        stop_btn = QPushButton("Stop")
        stop_btn.setStyleSheet(_style_btn(COL_GREEN_DRK, COL_WHITE, COL_GREEN))
        stop_btn.setFixedHeight(34)
        stop_btn.clicked.connect(self.stop_requested)
        layout.addWidget(stop_btn)

    def _blink(self):
        self._blink_state = not self._blink_state
        color = COL_GREEN if self._blink_state else COL_GREEN_DRK
        self._rec_label.setStyleSheet(f"color:{color};font-size:15px;font-weight:bold;")

    def _update_elapsed(self):
        elapsed = int(time.time() - self._start_time)
        h = elapsed // 3600; m = (elapsed % 3600) // 60; s = elapsed % 60
        self._time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        self._progress.setValue(min(elapsed, self.target_hours * 3600))

    def update_stats(self, bpm: float, arrhythmias: List[str]):
        if bpm > 0:
            self._bpm_label.setText(f"{bpm:.0f}")
        if arrhythmias:
            self._arrhy_label.setText("  |  ".join(arrhythmias[:3]))

    def cleanup(self):
        self._blink_timer.stop()
        self._elapsed_timer.stop()


# -----------------------------------------------------------------------------
# 3. HOLTER SUMMARY CARDS
# -----------------------------------------------------------------------------

class HolterSummaryCards(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value_labels = {}
        self._card_frames = []
        self._grid = None
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
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(12, 10, 12, 10)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        cards = [
            ("Average HR", "avg_hr", "bpm"),
            ("Min / Max HR", "range_hr", "bpm"),
            ("Total Beats", "beats", ""),
            ("Pauses", "pauses", "events"),
            ("Signal Quality", "quality", "%"),
            ("HRV SDNN", "sdnn", "ms"),
            ("rMSSD", "rmssd", "ms"),
            ("Longest RR", "longest_rr", "s"),
        ]
        for idx, (title, key, unit) in enumerate(cards):
            frame = QFrame()
            frame.setStyleSheet(
                f"QFrame{{background:{UI_CARD};border:1px solid {UI_BORDER};border-radius:10px;}}"
            )
            frame.setMinimumHeight(74)
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 10, 12, 10)
            box.setSpacing(3)
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:600;border:none;")
            val = QLabel("-")
            val.setStyleSheet(f"color:{UI_TEXT};font-size:21px;font-weight:700;border:none;")
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet(f"color:{UI_SUCCESS};font-size:10px;font-weight:700;border:none;")
            box.addWidget(lbl)
            box.addWidget(val)
            box.addWidget(unit_lbl)
            self._value_labels[key] = val
            self._card_frames.append(frame)
        self._relayout_cards()

    def _relayout_cards(self):
        if self._grid is None:
            return
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(self)

        width = max(1, self.width())
        if width >= 1400:
            cols = 4
        elif width >= 1050:
            cols = 3
        elif width >= 700:
            cols = 2
        else:
            cols = 1

        for idx, frame in enumerate(self._card_frames):
            self._grid.addWidget(frame, idx // cols, idx % cols)

        rows = int(math.ceil(len(self._card_frames) / float(cols)))
        self.setMinimumHeight(rows * 84 + 24)

    def update_summary(self, s: dict):
        self._value_labels["avg_hr"].setText(f"{s.get('avg_hr', 0):.0f}")
        self._value_labels["range_hr"].setText(f"{s.get('min_hr', 0):.0f} / {s.get('max_hr', 0):.0f}")
        self._value_labels["beats"].setText(f"{s.get('total_beats', 0):,}")
        self._value_labels["pauses"].setText(str(s.get("pauses", 0)))
        self._value_labels["quality"].setText(f"{s.get('avg_quality', 0) * 100:.1f}")
        self._value_labels["sdnn"].setText(f"{s.get('sdnn', 0):.1f}")
        self._value_labels["rmssd"].setText(f"{s.get('rmssd', 0):.1f}")
        self._value_labels["longest_rr"].setText(f"{s.get('longest_rr_ms', 0) / 1000:.2f}")
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_cards()

from .panels.holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas, HolterRRTrendCanvas, HRTrendCanvas
from .panels.holter_overview import HolterOverviewPanel, HolterExpertReviewPanel
from .panels.holter_replay import HolterReplayPanel
from .panels.holter_template import HolterBeatTemplatePanel, TemplateCardWidget, _TemplateMetricCard
from .panels.holter_histogram import HolterHistogramPanel
from .panels.holter_lorenz import HolterLorenzPanel
from .panels.holter_af import HolterAFPanel
from .panels.holter_events import HolterEventsPanel
from .panels.holter_st import HolterSTPanel
from .panels.holter_edit_event import HolterEditEventPanel
from .panels.holter_edit_strips import HolterEditStripsPanel
from .panels.holter_report_table import HolterReportTablePanel
from .panels.holter_recordings import HolterRecordManagementPanel
from .panels.holter_preview import HolterInsightPanel
from .panels.holter_hrv import HolterHRVPanel

# -----------------------------------------------------------------------------
# 17. HOLTER MAIN WINDOW  - Orchestrates everything
# -----------------------------------------------------------------------------

class HolterMainWindow(QDialog):
    def __init__(self, parent=None, session_dir: str = "",
                 patient_info: dict = None,
                 writer=None,
                 live_source=None,
                 duration_hours: int = 24):
        super().__init__(parent)
        self.setWindowTitle("Comprehensive ECG Analysis Monitor & Analysis")
        self.setMinimumSize(900, 620)

        screen = QApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.resize(max(1100, int(g.width() * 0.92)), max(750, int(g.height() * 0.92)))
        else:
            self.resize(1400, 900)

        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(f"QDialog{{background:{UI_BG};}}")
        self.showMaximized()

        self.session_dir = session_dir
        self.patient_info = _normalize_patient_info(patient_info or (writer.patient_info if writer else {}))
        self._writer = writer
        self._live_source = live_source
        self._duration_hours = duration_hours
        self._replay_engine = None
        self._metrics_list = []
        self._summary = {}
        self._last_live_seq = -1
        self._tab_name_map = {}

        if not self.session_dir and writer:
            self.session_dir = getattr(writer, 'session_dir', '')

        self._load_session()
        self._build_ui()

        if self._writer:
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._update_live_ui)
            self._live_timer.start(1000)

    # ---- Session loading -----------------------------------------------------------------------------

    def _load_session(self):
        self._metrics_list = []
        metadata = read_session_metadata(self.session_dir) if self.session_dir else {}
        self.patient_info = _load_patient_info_from_session(self.session_dir, self.patient_info)
        layered_metrics = load_metrics(self.session_dir) if self.session_dir else []
        if layered_metrics:
            self._metrics_list = layered_metrics
        jsonl_path = os.path.join(self.session_dir, 'metrics.jsonl') if self.session_dir else ''
        if not self._metrics_list and os.path.exists(jsonl_path):
            try:
                with open(jsonl_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._metrics_list.append(json.loads(line))
            except Exception as e:
                print(f"[HolterUI] Could not load metrics: {e}")

        ecgh_path = os.path.join(self.session_dir, 'recording.ecgh') if self.session_dir else ''
        if os.path.exists(ecgh_path):
            try:
                from .file_format import ECGHFileReader
                from .replay_engine import HolterReplayEngine
                self._replay_engine = HolterReplayEngine(ecgh_path)
                self._summary = self._replay_engine.get_summary()
            except Exception as e:
                print(f"[HolterUI] Replay engine error: {e}")
                self._summary = self._build_summary_from_metrics()
        else:
            self._summary = self._build_summary_from_metrics()
        if metadata and isinstance(metadata.get("summary"), dict):
            meta_sum = dict(metadata.get("summary"))
            if self._summary:
                meta_sum.update(self._summary)
            self._summary = meta_sum
        if self._summary is not None:
            self._summary["patient_info"] = dict(self.patient_info or {})

    def _build_summary_from_metrics(self) -> dict:
        if not self._metrics_list:
            return {}
        ml = self._metrics_list
        hr_vals = [m['hr_mean'] for m in ml if m.get('hr_mean', 0) > 0]
        beat_counts = [m.get('beat_count', 0) for m in ml]
        rr_stds = [m['rr_std'] for m in ml if m.get('rr_std', 0) > 0]
        rmssds = [m['rmssd'] for m in ml if m.get('rmssd', 0) > 0]
        pnn50s = [m['pnn50'] for m in ml if m.get('pnn50', 0) >= 0]
        qualities = [m['quality'] for m in ml if m.get('quality', 0) > 0]
        arrhy_counts: Dict[str, int] = {}
        beat_class_totals: Dict[str, int] = {}
        template_counts = []
        tachy_sec = 0.0
        brady_sec = 0.0
        for m in ml:
            for a in m.get('arrhythmias', []):
                arrhy_counts[a] = arrhy_counts.get(a, 0) + 1
            for cls, count in (m.get('beat_class_counts', {}) or {}).items():
                beat_class_totals[cls] = beat_class_totals.get(cls, 0) + int(count or 0)
            template_counts.append(int(m.get('template_count', 0) or 0))
            chunk_dur = float(m.get('duration', 4.0) or 4.0)
            hr_m = float(m.get('hr_mean', 0) or 0)
            if hr_m > 100:
                tachy_sec += chunk_dur
            elif 0 < hr_m < 60:
                brady_sec += chunk_dur
        all_rr = [m.get('longest_rr', 0) for m in ml]
        # Find max/min HR chunks for timestamps
        max_hr_chunk = max((m for m in ml if m.get('hr_mean', 0) > 0), key=lambda m: m.get('hr_mean', 0), default={})
        min_hr_chunk = min((m for m in ml if m.get('hr_mean', 0) > 0), key=lambda m: m.get('hr_mean', 0), default={})
        max_hr_t = float(max_hr_chunk.get('t', 0.0) or 0.0)
        min_hr_t = float(min_hr_chunk.get('t', 0.0) or 0.0)
        focus = derive_hr_focus_summary(ml)
        total_dur = _metrics_duration_sec(ml)
        tachy_pct = (tachy_sec / total_dur * 100) if total_dur > 0 else 0.0
        brady_pct = (brady_sec / total_dur * 100) if total_dur > 0 else 0.0
        return {
            'duration_sec': total_dur,
            'total_beats': sum(beat_counts),
            'avg_hr': float(np.mean(hr_vals)) if hr_vals else 0.0,
            'max_hr': float(np.max(hr_vals)) if hr_vals else 0.0,
            'min_hr': float(np.min(hr_vals)) if hr_vals else 0.0,
            'max_hr_time': _sec_to_hms(max_hr_t),
            'max_hr_timestamp': max_hr_t,
            'min_hr_time': _sec_to_hms(min_hr_t),
            'min_hr_timestamp': min_hr_t,
            'sinus_max_hr': focus.get('sinus_max_hr', float(np.max(hr_vals)) if hr_vals else 0.0),
            'sinus_min_hr': focus.get('sinus_min_hr', float(np.min(hr_vals)) if hr_vals else 0.0),
            'sinus_max_hr_time': focus.get('sinus_max_hr_time', _sec_to_hms(max_hr_t)),
            'sinus_max_hr_timestamp': focus.get('sinus_max_hr_timestamp', max_hr_t),
            'sinus_min_hr_time': focus.get('sinus_min_hr_time', _sec_to_hms(min_hr_t)),
            'sinus_min_hr_timestamp': focus.get('sinus_min_hr_timestamp', min_hr_t),
            'sdnn': float(np.mean(rr_stds)) if rr_stds else 0.0,
            'rmssd': float(np.mean(rmssds)) if rmssds else 0.0,
            'pnn50': float(np.mean(pnn50s)) if pnn50s else 0.0,
            'avg_quality': float(np.mean(qualities)) if qualities else 1.0,
            'arrhythmia_counts': arrhy_counts,
            'longest_rr_ms': max(all_rr) if all_rr else 0,
            'tachy_beats': sum(m.get('tachy_beats', 0) for m in ml),
            'brady_beats': sum(m.get('brady_beats', 0) for m in ml),
            'tachy_sec': tachy_sec,
            'brady_sec': brady_sec,
            'tachy_pct': tachy_pct,
            'brady_pct': brady_pct,
            'pauses': sum(m.get('pauses', 0) for m in ml),
            'avg_st_mv': float(np.mean([m.get('st_mv', 0) for m in ml])),
            'patient_info': self.patient_info,
            'chunks_analyzed': len(ml),
            'beat_class_totals': beat_class_totals,
            've_beats': int(beat_class_totals.get('VE', 0)),
            'sve_beats': int(beat_class_totals.get('SVE', 0)),
            'template_count': max(template_counts) if template_counts else 0,
        }

    def _tab_index_for(self, name: str) -> int:
        if not hasattr(self, '_tabs'):
            return -1
        target = (name or '').strip().lower()
        aliases = {
            'overview': 'overview',
            'view': 'preview',
            'report': 'preview',
            'preview': 'preview',
            'lorenz': 'lorenz',
            'histogram': 'histogram',
            'template': 'template',
            'af analysis': 'af analysis',
            'st tendency': 'st tendency',
            'edit event': 'edit event',
            'edit strips': 'edit strips',
            'report table': 'report table',
            'hrv': 'hrv',
            'recordings': 'recordings',
            'record settings': 'record settings',
            'replay': 'replay',
            'lorenz': 'replay',
        }
        target = aliases.get(target, target)
        for idx in range(self._tabs.count()):
            if self._tabs.tabText(idx).strip().lower() == target:
                return idx
        return -1

    def _focus_tab(self, name: str):
        idx = self._tab_index_for(name)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def _recordings_panel(self):
        return getattr(self, '_record_mgmt_panel', None)

    def _open_recordings_folder(self):
        if self._is_replay_active():
            return
        self._focus_tab('RECORDINGS')

    def _search_recordings(self):
        if self._is_replay_active():
            return
        self._focus_tab('RECORDINGS')
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_search'):
            panel._search.setFocus()
            panel._search.selectAll()

    def _apply_recordings_filter(self, label: str):
        if self._is_replay_active():
            return
        self._focus_tab('RECORDINGS')
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_filter'):
            idx = panel._filter.findText(label)
            if idx >= 0:
                panel._filter.setCurrentIndex(idx)

    def _import_recording(self):
        if self._is_replay_active():
            return
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_import_session'):
            panel._import_session()

    def _backup_recordings(self):
        if self._is_replay_active():
            return
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_backup_root'):
            panel._backup_root()

    def _delete_recording(self):
        if self._is_replay_active():
            return
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_delete_session'):
            panel._delete_session()

    def _is_replay_active(self) -> bool:
        panel = getattr(self, "_replay_panel", None)
        engine = getattr(self, "_replay_engine", None)
        return bool(panel and engine and engine.is_playing())

    def _set_record_browser_enabled(self, enabled: bool):
        panel = self._recordings_panel()
        if panel is not None:
            for name in ("Browse", "Import", "Export", "Backup", "Delete"):
                btn = getattr(panel, "_action_buttons", {}).get(name)
                if btn is not None:
                    btn.setEnabled(bool(enabled))
            table = getattr(panel, "_table", None)
            if table is not None:
                table.setEnabled(bool(enabled))
        top_browse = getattr(self, "_action_buttons", {}).get("Browse")
        if top_browse is not None:
            top_browse.setEnabled(bool(enabled))

    def _generate_from_current(self):
        self._focus_tab('PREVIEW')
        self._generate_report()

    def _refresh_current_session(self):
        self._load_session()
        self._refresh_ui()

    def _confirm_reanalysis(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Reanalyse Data")
        msg_box.setText("Would you like to reanalyse the data?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        msg_box.setStyleSheet(f"""
            QMessageBox {{
                background:{COL_BG};
                border:2px solid {COL_GREEN_DRK};
            }}
            QMessageBox QLabel {{
                color:{COL_WHITE};
                font-size:13px;
                font-weight:bold;
                border:none;
            }}
            QPushButton {{
                background:{COL_DARK};
                color:{COL_GREEN};
                border:1px solid {COL_GREEN_DRK};
                padding:6px 22px;
                border-radius:5px;
                font-size:12px;
                font-weight:bold;
            }}
            QPushButton:hover {{
                background:{COL_GREEN_DRK};
                color:{COL_WHITE};
            }}
            QPushButton:pressed {{
                background:{COL_GREEN};
                color:{COL_BLACK};
            }}
        """)
        reply = msg_box.exec_()
        if reply == QMessageBox.Yes:
            self._run_reanalysis_with_progress()

    def _run_reanalysis_with_progress(self):
        # --- Progress dialog ---
        dlg = QDialog(self, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        dlg.setFixedSize(480, 160)
        dlg.setStyleSheet("""
            QDialog {
                background: #0D1117;
                border: 1px solid #00CC66;
                border-radius: 12px;
            }
        """)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(28, 22, 28, 22)
        dlg_layout.setSpacing(10)

        # Header row
        hdr_row = QHBoxLayout()
        dot = QLabel(" ")
        dot.setStyleSheet("color:#00FF00; font-size:14px; border:none;")
        hdr_row.addWidget(dot)
        title_lbl = QLabel("ECG Reanalysis")
        title_lbl.setStyleSheet("color:#00FF00; font-size:14px; font-weight:bold; border:none; letter-spacing:1px;")
        hdr_row.addWidget(title_lbl)
        hdr_row.addStretch()
        pct_lbl = QLabel("0%")
        pct_lbl.setStyleSheet("color:#00CC66; font-size:12px; font-weight:bold; border:none;")
        hdr_row.addWidget(pct_lbl)
        dlg_layout.addLayout(hdr_row)

        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(False)
        progress.setFixedHeight(10)
        progress.setStyleSheet("""
            QProgressBar {
                background: #161B22;
                border: 1px solid #444444;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00CC66, stop:0.6 #00FF00, stop:1 #66FFB3);
                border-radius: 5px;
            }
        """)
        dlg_layout.addWidget(progress)

        # Status label
        status_lbl = QLabel("Initialising...")
        status_lbl.setStyleSheet("color:#888888; font-size:11px; border:none;")
        dlg_layout.addWidget(status_lbl)

        # Sub info
        sub_lbl = QLabel("This may take a few moments depending on recording length.")
        sub_lbl.setStyleSheet("color:#444444; font-size:10px; border:none;")
        dlg_layout.addWidget(sub_lbl)

        dlg.setModal(True)
        dlg.show()
        QApplication.processEvents()

        # --- Worker thread ---
        class _ReanalysisWorker(QThread):
            progress_changed = pyqtSignal(int, str)
            finished_ok = pyqtSignal()

            def __init__(self, fn):
                super().__init__()
                self._fn = fn

            def run(self):
                self.progress_changed.emit(10, "Loading session metadata...")
                try:
                    self._fn()
                    self.progress_changed.emit(90, "Finalising...")
                except Exception as e:
                    print(f"[Reanalysis] Error: {e}")
                self.progress_changed.emit(100, "Done!")
                self.finished_ok.emit()

        def _on_progress(val, txt):
            progress.setValue(val)
            pct_lbl.setText(f"{val}%")
            status_lbl.setText(txt)
            QApplication.processEvents()

        def _on_done():
            progress.setValue(100)
            pct_lbl.setText("100%")
            status_lbl.setText("Complete -- loading replay...")
            QApplication.processEvents()
            QTimer.singleShot(450, lambda: (dlg.accept(), _navigate_replay()))

        def _navigate_replay():
            try:
                self._refresh_ui()
            except Exception as e:
                print(f"[Reanalysis] _refresh_ui error: {e}")
            self._focus_tab('REPLAY')

        self._reanalysis_worker = _ReanalysisWorker(self._load_session)
        self._reanalysis_worker.progress_changed.connect(_on_progress)
        self._reanalysis_worker.finished_ok.connect(_on_done)

        # Animate progress bar smoothly with a timer while the thread runs
        _tick_val = [10]
        _anim_timer = QTimer(self)
        def _tick():
            if _tick_val[0] < 88:
                _tick_val[0] += 1
                progress.setValue(_tick_val[0])
                msg = ("Loading recorded data..." if _tick_val[0] < 30 else
                       "Running beat classification..." if _tick_val[0] < 55 else
                       "Computing HRV metrics..." if _tick_val[0] < 75 else
                       "Building summary...")
                status_lbl.setText(msg)
                pct_lbl.setText(f"{_tick_val[0]}%")
        _anim_timer.timeout.connect(_tick)
        _anim_timer.start(40)  # ~25 fps smooth fill

        self._reanalysis_worker.finished_ok.connect(_anim_timer.stop)
        self._reanalysis_worker.start()

    def _on_workspace_section_requested(self, section: str):
        key = (section or '').strip().lower()
        if key == 'quit':
            self.close()
        elif key in {'reanalysis', 'replay'}:
            self._confirm_reanalysis()
        elif key in {'overview'}:
            self._focus_tab('OVERVIEW')
        elif key in {'preview', 'view', 'report', 'edit report'}:
            self._focus_tab('PREVIEW')
        elif key == 'template':
            self._focus_tab('TEMPLATE')
        elif key == 'histogram':
            self._focus_tab('HISTOGRAM')
        elif key == 'lorenz':
            self._focus_tab('REPLAY')
        elif key == 'af analysis':
            self._focus_tab('AF ANALYSIS')
        elif key in {'tend. chart', 'st tendency'}:
            self._focus_tab('ST TENDENCY')
        elif key in {'edit event', 'pace spike', 'add event'}:
            self._focus_tab('EDIT EVENT')
        elif key == 'edit strips':
            self._focus_tab('EDIT STRIPS')
        elif key == 'report table':
            self._focus_tab('REPORT TABLE')
        elif key == 'hrv':
            self._focus_tab('HRV')
        elif key == 'record settings':
            self._focus_tab('RECORD SETTINGS')
        elif key == 'advance tools':
            self._focus_tab('RECORDINGS')
        elif key == 'print':
            self._generate_report()
        else:
            self._focus_tab('REPLAY')

    # ----- Build UI -----------------------------------------------------------------------------

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
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        #  "  "  Top title bar  "  " 
        title_bar = QFrame()
        title_bar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #07111F, stop:0.55 #102A4A, stop:1 #07111F);
                border-bottom: 1px solid #315783;
            }
        """)
        title_bar.setFixedHeight(52)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        tb_layout.setSpacing(14)
        self._mode_badge = QLabel("LIVE" if self._writer else "REVIEW")
        self._mode_badge.setStyleSheet(
            f"background:{'#7A2633' if self._writer else '#124936'};color:{UI_TEXT};"
            f"border:1px solid {UI_BORDER};border-radius:12px;padding:5px 12px;font-size:11px;font-weight:700;"
        )
        tb_layout.addWidget(self._mode_badge)
        app_title = QLabel("Comprehensive ECG Analysis")
        app_title.setStyleSheet(f"color:{UI_TEXT};font-size:18px;font-weight:700;border:none;")
        tb_layout.addWidget(app_title)
        dur_text = self._summary.get('duration_sec', 0)
        dur_h = int(dur_text // 3600)
        dur_m = int((dur_text % 3600) // 60)
        self._dur_label = QLabel(f"{dur_h:02d}h {dur_m:02d}m")
        self._dur_label.setStyleSheet(
            f"color:{UI_TEXT};font-size:13px;font-weight:700;background:{UI_PANEL_ALT};"
            f"padding:7px 12px;border-radius:8px;border:1px solid {UI_BORDER};"
        )
        tb_layout.addWidget(self._dur_label)
        tb_layout.addStretch()
        gen_report_btn = QPushButton("Generate Report")
        gen_report_btn.setStyleSheet(
            f"QPushButton{{background:{UI_ACCENT};color:{UI_TEXT};border:1px solid #61a8ff;border-radius:8px;padding:8px 14px;font-size:12px;font-weight:700;}}"
            f"QPushButton:hover{{background:{UI_ACCENT_HOVER};}}"
        )
        gen_report_btn.setFixedHeight(34)
        gen_report_btn.clicked.connect(self._generate_report)
        tb_layout.addWidget(gen_report_btn)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            f"QPushButton{{background:{UI_PANEL_ALT};color:{UI_TEXT};border:1px solid {UI_BORDER};border-radius:8px;padding:8px 14px;font-size:12px;font-weight:600;}}"
            "QPushButton:hover{background:#2A3D61;}"
        )
        close_btn.setFixedHeight(34)
        close_btn.clicked.connect(self.close)
        tb_layout.addWidget(close_btn)
        main_layout.addWidget(title_bar)

        session_bar = QFrame()
        session_bar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #111B2B, stop:0.5 #1B3558, stop:1 #111B2B);
                border-bottom: 1px solid #294B72;
            }
        """)
        sb_layout = QHBoxLayout(session_bar)
        sb_layout.setContentsMargins(14, 8, 14, 8)
        sb_layout.setSpacing(10)
        patient_name = self.patient_info.get("patient_name") or self.patient_info.get("name") or "Unknown Patient"
        doctor_name = self.patient_info.get("doctor") or "No referring doctor"
        session_name = os.path.basename(self.session_dir) if self.session_dir else "Active Session"
        self._patient_chip = QLabel(f"Patient: {patient_name}")
        self._patient_chip.setStyleSheet(f"background:#123B2D;color:{UI_TEXT};border:1px solid #206B51;border-radius:14px;padding:6px 12px;font-size:11px;font-weight:700;")
        sb_layout.addWidget(self._patient_chip)
        self._doctor_chip = QLabel(f"Doctor: {doctor_name}")
        self._doctor_chip.setStyleSheet(f"background:{UI_PANEL};color:{UI_MUTED};border:1px solid {UI_BORDER};border-radius:14px;padding:6px 12px;font-size:11px;font-weight:600;")
        sb_layout.addWidget(self._doctor_chip)
        self._session_chip = QLabel(f"Session: {session_name}")
        self._session_chip.setStyleSheet(f"background:{UI_PANEL};color:{UI_MUTED};border:1px solid {UI_BORDER};border-radius:14px;padding:6px 12px;font-size:11px;font-weight:600;")
        sb_layout.addWidget(self._session_chip)
        self._rec_time_chip = QLabel("Recording: --")
        self._rec_time_chip.setStyleSheet(f"background:{UI_PANEL};color:{UI_MUTED};border:1px solid {UI_BORDER};border-radius:14px;padding:6px 12px;font-size:11px;font-weight:600;")
        sb_layout.addWidget(self._rec_time_chip)
        sb_layout.addStretch()
        self._analysis_state = QLabel("Clinical review mode")
        self._analysis_state.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:600;border:none;")
        sb_layout.addWidget(self._analysis_state)
        main_layout.addWidget(session_bar)

        action_bar = QFrame()
        action_bar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3D444B, stop:0.5 #20252A, stop:1 #0B0D0F);
                border-bottom: 1px solid #56616B;
            }
        """)
        ab_layout = QHBoxLayout(action_bar)
        ab_layout.setContentsMargins(8, 6, 8, 6)
        ab_layout.setSpacing(6)
        self._action_buttons = {}
        for label in ["Browse", "Search", "Analyse", "View", "Import", "Backup"]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #59636D, stop:0.5 #30363C, stop:1 #171A1D);
                    color: #FFFFFF; border: 1px solid #78838D;
                    border-radius: 5px; padding: 5px 13px; font-weight: 700;
                }
                QPushButton:hover { background: #657481; border-color: #AEB8C0; }
                QPushButton:pressed { background: #171B1F; }
            """)
            self._action_buttons[label] = btn
            ab_layout.addWidget(btn)
        ab_layout.addStretch()
        self._filter_buttons = {}
        for label in ["All", "Today", "Yesterday", "This Week", "This Month", "This Year"]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4B545D, stop:1 #20252A);
                    color: #FFFFFF; border: 1px solid #68747E;
                    border-radius: 5px; padding: 5px 11px; font-weight: 600;
                }
                QPushButton:hover { background: #5E6A75; color: #FFFFFF; }
            """)
            self._filter_buttons[label] = btn
            ab_layout.addWidget(btn)
        main_layout.addWidget(action_bar)

        # ----- Status bar (if recording) ------------------------------------------------------------
        if self._writer:
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            main_layout.addWidget(self._status_bar)

        # ----- Summary KPI cards ----------------------------------------------------------------------------

        # ----- Body: tabs fill full width (12-lead grid is inside HolterReplayPanel) -----------------------------------------
        right_frame = QFrame()
        right_frame.setStyleSheet(f"QFrame{{background:{UI_BG};}}")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setUsesScrollButtons(True)

        # Make the tab bar scrollable with the mouse wheel instead of changing tabs
        class TabBarScroller(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Wheel:
                    # In QTabBar, there is a private QScrollArea, but sending the wheel event 
                    # as horizontal scroll to the tabbar doesn't always work cleanly.
                    # Alternatively, if we just want to suppress changing tabs, we can return True,
                    # but to scroll, we need to find the scroll buttons or adjust the scroll offset.
                    # A simple way to trigger the internal scroll is to post a wheel event to the 
                    # tab widget's internal scroll widget, but PyQt doesn't expose it easily.
                    # Wait, if usesScrollButtons is True, QTabBar has two QToolButtons as children.
                    # We can simulate clicks on them based on wheel direction.
                    delta = event.angleDelta().y()
                    if delta == 0:
                        delta = event.angleDelta().x()
                    if delta != 0:
                        buttons = obj.findChildren(QToolButton)
                        if len(buttons) >= 2:
                            # Usually button 0 is left, 1 is right
                            if delta > 0:
                                buttons[0].click()
                                buttons[0].click()
                            else:
                                buttons[1].click()
                                buttons[1].click()
                    return True # Consume event to prevent changing the selected tab
                return super().eventFilter(obj, event)

        self._tab_scroller = TabBarScroller(self)
        self._tabs.tabBar().installEventFilter(self._tab_scroller)
        self._tabs.setIconSize(QSize(18, 18))

        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background:{UI_BG};
                border:1px solid {UI_BORDER};
                border-top:none;
            }}
            QTabBar::tab {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4B545D, stop:0.5 #282E34, stop:1 #111417);
                color:#FFFFFF;
                border:1px solid #68747E;
                border-radius:8px;
                padding:8px 10px;
                font-size:11px;
                font-weight:700;
                margin:6px 6px 8px 0;
                min-width:115px;
                text-align:center;
            }}
            QTabBar::tab:selected {{
                color:#FFFFFF;
                background:{UI_ACCENT};
                border-color:#6EB4FF;
            }}
            QTabBar::tab:hover:!selected {{
                background:#566572;
                color:#FFFFFF;
            }}
            QGroupBox {{
                border: none;
                border-top: 1px solid {UI_BORDER};
                margin-top: 20px;
                font-weight: bold;
                font-size: 14px;
                color: {UI_TEXT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }}
        """)

        def _make_icon(tab_type: str) -> QIcon:
            pix = QPixmap(24, 24)
            pix.fill(Qt.transparent)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor('#4DB6AC'), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)

            if tab_type == "OVERVIEW":
                # Monitor screen / grid icon
                painter.drawRect(3, 4, 18, 13)
                painter.drawLine(8, 20, 16, 20)
                painter.drawLine(12, 17, 12, 20)
                path = QPainterPath()
                path.moveTo(5, 11)
                path.lineTo(8, 11)
                path.lineTo(10, 7)
                path.lineTo(13, 15)
                path.lineTo(15, 11)
                path.lineTo(19, 11)
                painter.setPen(QPen(QColor('#00E676'), 1.8))
                painter.drawPath(path)

            elif tab_type == "REPLAY":
                # Play / replay curve icon
                path = QPainterPath()
                path.moveTo(6, 12)
                path.lineTo(9, 12)
                path.lineTo(11, 7)
                path.lineTo(13, 17)
                path.lineTo(15, 12)
                path.lineTo(19, 12)
                painter.setPen(QPen(QColor('#FFCA28'), 1.8))
                painter.drawPath(path)
                painter.drawEllipse(16, 15, 5, 5)

            elif tab_type == "TEMPLATE":
                # Beat template pulse
                path = QPainterPath()
                path.moveTo(3, 14)
                path.lineTo(7, 14)
                path.lineTo(9, 18)
                path.lineTo(12, 4)
                path.lineTo(15, 19)
                path.lineTo(17, 14)
                path.lineTo(21, 14)
                painter.setPen(QPen(QColor('#29B6F6'), 2))
                painter.drawPath(path)

            elif tab_type == "HISTOGRAM":
                # Bar chart histogram
                painter.setPen(QPen(QColor('#AB47BC'), 1.8))
                painter.drawRect(4, 13, 3, 7)
                painter.drawRect(9, 7, 3, 13)
                painter.drawRect(14, 10, 3, 10)
                painter.drawRect(19, 15, 3, 5)

            elif tab_type == "LORENZ":
                # Scatter plot
                painter.setPen(QPen(QColor('#26A69A'), 1.8))
                painter.drawLine(4, 20, 20, 20)
                painter.drawLine(4, 4, 4, 20)
                painter.setPen(QPen(QColor('#26C6DA'), 3))
                painter.drawPoint(8, 16)
                painter.drawPoint(11, 12)
                painter.drawPoint(13, 10)
                painter.drawPoint(16, 7)
                painter.drawPoint(17, 14)

            elif tab_type == "AF ANALYSIS":
                # Irregular AF rhythm wave
                path = QPainterPath()
                path.moveTo(3, 12)
                path.lineTo(6, 9)
                path.lineTo(8, 15)
                path.lineTo(11, 5)
                path.lineTo(14, 18)
                path.lineTo(17, 10)
                path.lineTo(21, 12)
                painter.setPen(QPen(QColor('#EF5350'), 1.8))
                painter.drawPath(path)

            elif tab_type == "EVENTS":
                # List / warning event icon
                painter.setPen(QPen(QColor('#FFA726'), 1.8))
                painter.drawLine(4, 7, 7, 7)
                painter.drawLine(10, 7, 20, 7)
                painter.drawLine(4, 12, 7, 12)
                painter.drawLine(10, 12, 20, 12)
                painter.drawLine(4, 17, 7, 17)
                painter.drawLine(10, 17, 20, 17)

            elif tab_type == "ST TENDENCY":
                # ST elevation trend line
                painter.setPen(QPen(QColor('#66BB6A'), 1.8))
                painter.drawLine(3, 17, 8, 17)
                painter.drawLine(8, 17, 13, 8)
                painter.drawLine(13, 8, 21, 8)

            elif tab_type == "EDIT EVENT":
                # Pencil / edit icon
                painter.setPen(QPen(QColor('#FF7043'), 1.8))
                painter.drawLine(5, 19, 19, 5)
                painter.drawLine(15, 5, 19, 9)
                painter.drawLine(3, 21, 7, 20)

            elif tab_type == "EDIT STRIPS":
                # Strip scissors / edit
                painter.setPen(QPen(QColor('#8D6E63'), 1.8))
                painter.drawRect(4, 6, 16, 12)
                painter.drawLine(8, 6, 8, 18)
                painter.drawLine(16, 6, 16, 18)

            elif tab_type == "REPORT TENDENCY":
                # Report graph line
                painter.setPen(QPen(QColor('#26C6DA'), 1.8))
                path = QPainterPath()
                path.moveTo(4, 16)
                path.lineTo(9, 10)
                path.lineTo(14, 14)
                path.lineTo(19, 6)
                painter.drawPath(path)

            elif tab_type == "REPORT TABLE":
                # Grid table icon
                painter.setPen(QPen(QColor('#7E57C2'), 1.8))
                painter.drawRect(4, 4, 16, 16)
                painter.drawLine(4, 10, 20, 10)
                painter.drawLine(4, 15, 20, 15)
                painter.drawLine(12, 4, 12, 20)

            elif tab_type == "HRV":
                # Heart + pulse icon
                painter.setPen(QPen(QColor('#EC407A'), 1.8))
                path = QPainterPath()
                path.moveTo(12, 19)
                path.cubicTo(6, 14, 3, 10, 6, 6)
                path.cubicTo(8, 4, 11, 5, 12, 8)
                path.cubicTo(13, 5, 16, 4, 18, 6)
                path.cubicTo(21, 10, 18, 14, 12, 19)
                painter.drawPath(path)

            elif tab_type == "RECORD SETTINGS":
                # Gear icon
                painter.setPen(QPen(QColor('#78909C'), 1.8))
                painter.drawEllipse(7, 7, 10, 10)
                painter.drawLine(12, 3, 12, 7)
                painter.drawLine(12, 17, 12, 21)
                painter.drawLine(3, 12, 7, 12)
                painter.drawLine(17, 12, 21, 12)

            elif tab_type == "RECORDINGS":
                # Folder / archive icon
                painter.setPen(QPen(QColor('#FFA726'), 1.8))
                painter.drawRect(4, 8, 16, 12)
                path = QPainterPath()
                path.moveTo(4, 8)
                path.lineTo(8, 4)
                path.lineTo(13, 4)
                path.lineTo(15, 8)
                painter.drawPath(path)

            elif tab_type == "PREVIEW":
                # Eye / document preview
                painter.setPen(QPen(QColor('#42A5F5'), 1.8))
                painter.drawRect(6, 4, 12, 16)
                painter.drawLine(9, 8, 15, 8)
                painter.drawLine(9, 12, 15, 12)
                painter.drawLine(9, 16, 13, 16)

            elif tab_type == "PRINT":
                # Printer icon
                painter.setPen(QPen(QColor('#9E9E9E'), 1.8))
                painter.drawRect(6, 9, 12, 8)
                painter.drawRect(8, 4, 8, 5)
                painter.drawRect(8, 14, 8, 6)

            elif tab_type == "REANALYSIS":
                # Refresh / sync icon
                painter.setPen(QPen(QColor('#26A69A'), 1.8))
                painter.drawArc(4, 4, 16, 16, 30 * 16, 280 * 16)
                painter.drawLine(18, 6, 20, 10)
                painter.drawLine(18, 6, 14, 9)

            elif tab_type == "QUIT":
                # Power / exit icon
                painter.setPen(QPen(QColor('#E53935'), 1.8))
                painter.drawArc(5, 5, 14, 14, -40 * 16, 260 * 16)
                painter.drawLine(12, 3, 12, 11)

            else:
                painter.setPen(QPen(QColor('#00E676'), 1.8))
                painter.drawEllipse(6, 6, 12, 12)

            painter.end()
            return QIcon(pix)

        #  "  "  OVERVIEW tab (12-lead scrollable expert view)  "  " 
        self._expert_panel = HolterExpertReviewPanel()
        self._expert_panel.update_from_metrics(self._metrics_list, self._summary)
        self._expert_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._expert_panel, _make_icon("OVERVIEW"), "OVERVIEW")

        # Replay
        duration = self._summary.get('duration_sec', self._duration_hours * 3600)
        self._replay_panel = HolterReplayPanel(duration_sec=duration)
        if self._replay_engine:
            self._replay_panel.set_replay_engine(self._replay_engine)
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        self._replay_panel.section_requested.connect(self._on_workspace_section_requested)
        self._replay_panel.playback_state_changed.connect(self._set_record_browser_enabled)
        self._replay_panel.update_lorenz(self._metrics_list)
        self._replay_panel.update_summary(self._summary)
        self._tabs.addTab(self._replay_panel, _make_icon("REPLAY"), "REPLAY")
        self._lorenz_panel = self._replay_panel

        # Beat Templates
        self._template_panel = HolterBeatTemplatePanel()
        if self._replay_engine:
            try:
                self._template_panel.set_replay_engine(self._replay_engine)
            except Exception:
                pass
        self._template_panel.update_from_metrics(self._metrics_list, self._summary)
        self._template_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._template_panel, _make_icon("TEMPLATE"), "TEMPLATE")

        # Histogram
        self._hist_panel = HolterHistogramPanel()
        self._hist_panel.update_from_metrics(self._metrics_list)
        self._hist_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._hist_panel, _make_icon("HISTOGRAM"), "HISTOGRAM")

        # Lorenz
        self._lorenz_tab_panel = HolterLorenzPanel(replay_engine=self._replay_engine)
        self._lorenz_tab_panel.update_from_metrics(self._metrics_list)
        self._lorenz_tab_panel.seek_requested.connect(self._on_seek_requested)
        if self._replay_engine:
            self._lorenz_tab_panel.set_replay_engine(self._replay_engine)
        self._tabs.addTab(self._lorenz_tab_panel, _make_icon("LORENZ"), "LORENZ")


        # AF Analysis
        self._af_panel = HolterAFPanel(session_dir=self.session_dir)
        self._af_panel.update_from_metrics(self._metrics_list, duration)
        self._tabs.addTab(self._af_panel, _make_icon("AF ANALYSIS"), "AF ANALYSIS")

        events = self._build_linked_events()

        # Event timeline
        self._events_panel = HolterEventsPanel()
        self._events_panel.set_session_dir(self.session_dir)
        self._events_panel.load_events(events, self._summary)
        self._events_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._events_panel, _make_icon("EVENTS"), "EVENTS")

        # ST Tendency
        self._st_panel = HolterSTPanel(replay_engine=self._replay_engine)
        self._st_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._st_panel, _make_icon("ST TENDENCY"), "ST TENDENCY")

        # Edit Event
        self._edit_event_panel = HolterEditEventPanel()
        self._edit_event_panel.set_session_dir(self.session_dir)
        self._edit_event_panel.load_events(events, self._summary)
        self._edit_event_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._edit_event_panel, _make_icon("EDIT EVENT"), "EDIT EVENT")

        # Edit Strips
        self._edit_strips_panel = HolterEditStripsPanel()
        self._edit_strips_panel.set_session_dir(self.session_dir)
        self._edit_strips_panel.seek_requested.connect(self._on_seek_requested)
        self._edit_strips_panel.load_events(events, self._summary, self._metrics_list)
        self._tabs.addTab(self._edit_strips_panel, _make_icon("EDIT STRIPS"), "EDIT STRIPS")

        # Report Tendency
        self._report_tendency_panel = HolterSTPanel(replay_engine=self._replay_engine)
        self._tabs.addTab(self._report_tendency_panel, _make_icon("REPORT TENDENCY"), "REPORT TENDENCY")

        # Report Table
        self._report_table_panel = HolterReportTablePanel()
        self._report_table_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._report_table_panel, _make_icon("REPORT TABLE"), "REPORT TABLE")

        # Expert Review (OVERVIEW tab) already added as first tab above
        # kept here as comment for clarity   " see OVERVIEW tab creation at top of tabs section

        # HRV Analysis
        self._hrv_panel = HolterHRVPanel()
        self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        self._tabs.addTab(self._hrv_panel, _make_icon("HRV"), "HRV")

        # Record settings
        settings_panel = QWidget()
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(18, 18, 18, 18)
        settings_layout.setSpacing(10)
        settings_title = QLabel("Record Settings")
        settings_title.setStyleSheet(f"font-size:16px;font-weight:700;color:{UI_TEXT};")
        settings_layout.addWidget(settings_title)
        settings_info = QLabel(
            f"Recording duration: {self._duration_hours:g} hours\n"
            f"Session: {os.path.basename(os.path.normpath(self.session_dir)) if self.session_dir else 'Current session'}"
        )
        settings_info.setStyleSheet(f"font-size:13px;color:{UI_MUTED};")
        settings_info.setWordWrap(True)
        settings_layout.addWidget(settings_info)
        settings_layout.addStretch(1)
        self._record_settings_panel = settings_panel
        self._tabs.addTab(settings_panel, _make_icon("RECORD SETTINGS"), "RECORD SETTINGS")

        # Record browser
        self._record_mgmt_panel = HolterRecordManagementPanel(
            output_dir=_resolve_recordings_dir(self.session_dir)
        )
        self._record_mgmt_panel.session_selected.connect(self.load_completed_session)
        if self.session_dir:
            self._record_mgmt_panel.set_active_session(self.session_dir)
        self._tabs.addTab(self._record_mgmt_panel, _make_icon("RECORDINGS"), "RECORDINGS")

        # Report Preview
        scroll_insight = QScrollArea()
        scroll_insight.setWidgetResizable(True)
        scroll_insight.setFrameShape(QFrame.NoFrame)
        scroll_insight.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        self._insight_panel = HolterInsightPanel()
        self._insight_panel.update_text(self.patient_info, self._summary)
        scroll_insight.setWidget(self._insight_panel)
        self._tabs.addTab(scroll_insight, _make_icon("PREVIEW"), "PREVIEW")
        self._tabs.addTab(QWidget(), _make_icon("PRINT"), "PRINT")
        self._tabs.addTab(QWidget(), _make_icon("REANALYSIS"), "REANALYSIS")
        self._tabs.addTab(QWidget(), _make_icon("QUIT"), "QUIT")

        # Track the last active content tab
        self._last_active_tab_name = "OVERVIEW"
        def _on_tab_changed(index):
            tab_name = self._tabs.tabText(index)
            if tab_name in {"PRINT", "REANALYSIS", "QUIT"}:
                if hasattr(self, "_last_active_tab_name"):
                    self._focus_tab(self._last_active_tab_name)
                if tab_name == "PRINT":
                    QTimer.singleShot(0, self._generate_report)
                elif tab_name == "REANALYSIS":
                    QTimer.singleShot(0, self._confirm_reanalysis)
                elif tab_name == "QUIT":
                    QTimer.singleShot(0, self.close)
            else:
                self._last_active_tab_name = tab_name
                if tab_name == "RECORDINGS":
                    self._tabs.tabBar().setVisible(False)
                else:
                    self._tabs.tabBar().setVisible(True)
        self._tabs.currentChanged.connect(_on_tab_changed)
        # Ensure initial state is correct
        _on_tab_changed(self._tabs.currentIndex())

        right_layout.addWidget(self._tabs)
        self._tabs.currentChanged.connect(
            lambda idx: hasattr(self, '_analysis_state') and self._analysis_state.setText(
                f"Focused view: {self._tabs.tabText(idx)}"
            )
        )
        action_handlers = {
            "Browse": self._open_recordings_folder,
            "Search": self._search_recordings,
            "Analyse": lambda: self._focus_tab("REPLAY"),
            "View": lambda: self._focus_tab("PREVIEW"),
            "Import": self._import_recording,
            "Backup": self._backup_recordings,
            "Delete": self._delete_recording,
        }
        for name, handler in action_handlers.items():
            btn = self._action_buttons.get(name)
            if btn is not None:
                btn.clicked.connect(handler)
        for label, btn in self._filter_buttons.items():
            btn.clicked.connect(lambda _, t=label: self._apply_recordings_filter(t))

        main_layout.addWidget(right_frame, 1)
        if hasattr(self, '_analysis_state'):
            self._analysis_state.setText(f"Focused view: {self._tabs.tabText(self._tabs.currentIndex())}")


    # ----- Callbacks ----------------------------------------------------------------------------

    def _current_replay_window_sec(self) -> float:
        panel = getattr(self, '_replay_panel', None)
        if panel is not None:
            try:
                return float(getattr(panel, '_strip_length_sec', 10.0) or 10.0)
            except Exception:
                pass
        return float(getattr(self, '_strip_length_sec', 10.0) or 10.0)

    def _sync_replay_window_length(self):
        if getattr(self, '_replay_engine', None) and hasattr(self._replay_engine, 'set_window_length'):
            try:
                self._replay_engine.set_window_length(self._current_replay_window_sec())
            except Exception:
                pass

    def _on_seek_requested(self, target_sec: float):
        if self._replay_engine:
            self._sync_replay_window_length()
            self._replay_engine.seek(target_sec)
            try:
                # Use the replay panel's current strip length (changes with paper speed)
                window_sec = self._current_replay_window_sec()
                data = self._replay_engine.get_all_leads_data(window_sec=float(window_sec))
                beat_annotations = self._replay_engine.get_beat_annotations(window_sec=float(window_sec))
                start_sec = self._replay_engine.current_position() - window_sec / 2
                if hasattr(self, '_wave_panel'):
                    self._wave_panel.set_replay_frame(data, beat_annotations=beat_annotations, start_sec=start_sec)
                self._broadcast_replay_frame(data, beat_annotations=beat_annotations, start_sec=start_sec)
            except Exception:
                pass


    def _broadcast_replay_frame(self, data, beat_annotations=None, start_sec=0.0):
        for panel in [getattr(self, p, None) for p in [
            '_replay_panel', '_lorenz_panel', '_hist_panel', '_af_panel',
            '_st_panel', '_edit_event_panel', '_edit_strips_panel', '_events_panel',
            '_expert_panel', '_template_panel', '_report_tendency_panel', '_hrv_panel',
            '_lorenz_tab_panel'
        ]]:
            if panel and hasattr(panel, 'set_replay_frame'):
                try:
                    panel.set_replay_frame(data, beat_annotations=beat_annotations, start_sec=start_sec)
                except TypeError:
                    # Fallback for panels that don't accept beat_annotations
                    try:
                        panel.set_replay_frame(data)
                    except Exception:
                        pass
                except Exception:
                    pass

    def _build_linked_events(self) -> list:
        events = []
        # 1. Manual marks and user-saved events from replay engine
        if self._replay_engine:
            try:
                for ev in (self._replay_engine.get_events_list() or []):
                    if ev.get('is_manual', False) or ev.get('source') == 'Manual':
                        events.append(dict(ev))
            except Exception:
                pass

        # 2. Collect all beats from metrics
        all_beats = []
        for metric in self._metrics_list or []:
            for b in metric.get('all_beats', []) or []:
                if isinstance(b, dict):
                    all_beats.append(b)

            # Keep NSR / Sinus Rhythm detection as requested
            base_t = float(metric.get('t', 0.0) or 0.0)
            for label in metric.get('arrhythmias', []) or []:
                lbl = str(label).strip()
                lbl_lower = lbl.lower()
                if 'sinus' in lbl_lower or 'normal' in lbl_lower:
                    events.append({
                        'timestamp': base_t,
                        'label': lbl,
                        'source': 'analysis',
                        'confidence': 1.0,
                        'time_str': _sec_to_hms(base_t),
                    })

        # 3. Generate auto-detection badge events (SVT, VT, Couplet, Bigeminy, PVC, PAC, etc.)
        try:
            from .holter_summary_calc import get_template_beats_for_badges
            badges = get_template_beats_for_badges(all_beats, [])
            for b in badges:
                badge_name = b.get('name', '').strip()
                if not badge_name:
                    continue
                # Skip normal badges (only include auto detection arrhythmia badges: SVT, VT, Couplet, etc.)
                if b.get('code') in ['V', 'S', 'AF', 'P'] or b.get('critical', False) or 'run' in badge_name.lower() or 'couplet' in badge_name.lower() or 'bigeminy' in badge_name.lower() or 'trigeminy' in badge_name.lower():
                    events.append({
                        'timestamp': float(b.get('timestamp', 0.0) or 0.0),
                        'label': badge_name,
                        'source': 'analysis',
                        'confidence': 1.0,
                        'time_str': _sec_to_hms(float(b.get('timestamp', 0.0) or 0.0)),
                    })
        except Exception as e:
            print(f"[HolterUI] Error generating badge events: {e}")

        events.sort(key=lambda e: float(e.get('timestamp', 0.0) or 0.0))
        dedup = []
        seen = set()
        for ev in events:
            key = (round(float(ev.get('timestamp', 0.0) or 0.0), 3), str(ev.get('label', '')))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(ev)
        return dedup

    def _update_live_ui(self):
        if not self._writer or not self._writer.is_running:
            if hasattr(self, '_live_timer'):
                self._live_timer.stop()
            self._load_session()
            self._refresh_ui()
            return
        stats = self._writer.get_live_stats()
        if hasattr(self, '_status_bar'):
            self._status_bar.update_stats(stats['bpm'], stats['arrhythmias'])
        snapshot = None
        if hasattr(self._writer, 'get_live_analysis_snapshot'):
            snapshot = self._writer.get_live_analysis_snapshot(getattr(self, '_last_live_seq', -1))
        if snapshot:
            self._last_live_seq = snapshot.get('seq', self._last_live_seq)
            self._metrics_list = snapshot.get('metrics', [])
            self._summary = snapshot.get('summary', {})
            self._refresh_ui()
        if hasattr(self, '_wave_panel'):
            self._wave_panel.refresh_waveforms()
        # Also refresh _replay_panel lead strips from live data on every tick
        if hasattr(self, '_replay_panel') and not self._replay_engine and self._live_source is not None:
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
        if snapshot is None and stats['elapsed'] % 15 < 2:
            self._load_session()
            self._refresh_ui()

    def _refresh_ui(self):
        if hasattr(self, '_summary_cards'):
            self._summary_cards.update_summary(self._summary)
        if hasattr(self, '_insight_panel'):
            self._insight_panel.update_text(self.patient_info, self._summary)
        if hasattr(self, '_overview_panel'):
            self._overview_panel.update_summary(self._summary)
        if hasattr(self, '_expert_panel'):
            self._expert_panel.update_from_metrics(self._metrics_list, self._summary)
        if hasattr(self, '_hrv_panel'):
            self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        if hasattr(self, '_replay_panel'):
            self._replay_panel.update_lorenz(self._metrics_list)
            self._replay_panel.update_summary(self._summary)
            # During live recording (no replay engine), push current live ECG frame
            if not self._replay_engine and self._live_source is not None:
                try:
                    raw = getattr(self._live_source, 'data', None)
                    if raw is not None and hasattr(raw, '__len__') and len(raw) > 0:
                        import numpy as _np
                        # Build a 12-channel array from live_source.data
                        n_samp = max(len(raw[i]) for i in range(min(12, len(raw))))
                        arr = _np.full((12, n_samp), 2048.0)
                        for i in range(min(12, len(raw))):
                            ch = _np.asarray(raw[i], dtype=float)
                            arr[i, :len(ch)] = ch
                        self._replay_panel.set_replay_frame(arr)
                except Exception:
                    pass
        if hasattr(self, '_hist_panel'):
            self._hist_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_lorenz_tab_panel'):
            self._lorenz_tab_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_af_panel'):
            self._af_panel.update_from_metrics(self._metrics_list, self._summary.get('duration_sec', 0))
        if hasattr(self, '_st_panel'):
            self._st_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_report_table_panel'):
            self._report_table_panel.update_from_metrics(self._metrics_list)
        events = self._build_linked_events()
        if hasattr(self, '_events_panel'):
            self._events_panel.set_session_dir(self.session_dir)
            self._events_panel.load_events(events, self._summary)
        if hasattr(self, '_template_panel'):
            if self._replay_engine:
                try:
                    self._template_panel.set_replay_engine(self._replay_engine)
                except Exception:
                    pass
            self._template_panel.update_from_metrics(self._metrics_list, self._summary)
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.set_session_dir(self.session_dir)
            self._edit_event_panel.load_events(events, self._summary)
        if hasattr(self, '_edit_strips_panel'):
            self._edit_strips_panel.set_session_dir(self.session_dir)
            self._edit_strips_panel.load_events(events, self._summary, self._metrics_list)
        if hasattr(self, '_wave_panel'):
            self._wave_panel.set_live_source(self._live_source)
        if hasattr(self, '_record_mgmt_panel'):
            self._record_mgmt_panel.output_dir = _resolve_recordings_dir(self.session_dir)
            self._record_mgmt_panel.refresh_records()
        if hasattr(self, '_wave_panel') and self._replay_engine:
            try:
                self._wave_panel.set_replay_engine(self._replay_engine)
            except Exception:
                pass
            self._wave_panel.refresh_waveforms()
        if hasattr(self, '_expert_panel') and self._replay_engine:
            try:
                self._expert_panel.set_replay_frame(self._replay_engine.get_all_leads_data(window_sec=self._current_replay_window_sec()))
            except Exception:
                pass
        if self._replay_engine:
            try:
                self._broadcast_replay_frame(self._replay_engine.get_all_leads_data(window_sec=self._current_replay_window_sec()))
            except Exception:
                pass

        # Update duration label
        dur = self._summary.get('duration_sec', 0)
        dur_h = int(dur // 3600)
        dur_m = int((dur % 3600) // 60)
        if hasattr(self, '_dur_label'):
            self._dur_label.setText(f"{dur_h:02d}h {dur_m:02d}m")
        if hasattr(self, '_patient_chip'):
            patient_name = self.patient_info.get("patient_name") or self.patient_info.get("name") or "Unknown Patient"
            self._patient_chip.setText(f"Patient: {patient_name}")
        if hasattr(self, '_doctor_chip'):
            doctor_name = self.patient_info.get("doctor") or "No referring doctor"
            self._doctor_chip.setText(f"Doctor: {doctor_name}")
        if hasattr(self, '_session_chip'):
            session_name = os.path.basename(self.session_dir) if self.session_dir else "Active Session"
            self._session_chip.setText(f"Session: {session_name}")
        if hasattr(self, '_rec_time_chip'):
            st_str, end_str = _get_recording_start_end_times(self.session_dir, dur)
            if st_str != "Unknown":
                self._rec_time_chip.setText(f"Recording: {st_str} to {end_str}")
                self._rec_time_chip.show()
            else:
                self._rec_time_chip.hide()
        if hasattr(self, '_analysis_state') and hasattr(self, '_tabs'):
            self._analysis_state.setText(f"Focused view: {self._tabs.tabText(self._tabs.currentIndex())}")

    def _finalize_live_writer(self) -> dict:
        summary = {}
        if not self._writer:
            return summary
        try:
            stop_fn = getattr(self._writer, "stop", None)
            if callable(stop_fn):
                summary = stop_fn() or {}
            else:
                close_fn = getattr(self._writer, "close", None)
                if callable(close_fn):
                    summary = close_fn() or {}
        finally:
            self._writer = None
        return summary

    def _stop_recording(self):
        if self._writer:
            summary = self._finalize_live_writer()
            if hasattr(self, '_status_bar') and self._status_bar is not None:
                self._status_bar.setVisible(False)
                if hasattr(self._status_bar, 'cleanup'):
                    self._status_bar.cleanup()
            
            # Show dialog to collect patient info AFTER recording
            dialog = HolterStartDialog(self, patient_info=self.patient_info or {}, output_dir=self.session_dir)
            dialog.setWindowTitle("Save Comprehensive ECG Analysis Recording Details")
            if dialog.exec_() == QDialog.Accepted:
                patient_info, dur, out_dir = dialog.get_result()
                summary['patient_info'] = patient_info
                self.patient_info = patient_info
                import json
                try:
                    with open(os.path.join(summary.get('session_dir', ''), "patient.json"), 'w') as f:
                        json.dump(patient_info, f, indent=4)
                except Exception as e:
                    print(f"Failed to save patient.json: {e}")

            QMessageBox.information(self, "Recording Complete",
                                    f"Comprehensive ECG Analysis recording saved to:\n{summary.get('session_dir', '')}")
            self.load_completed_session(summary.get('session_dir', ''), summary.get('patient_info', {}))
            
            # Auto-generate report when recording is stopped
            self._generate_report()

    def _generate_report(self):
        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import QThread, pyqtSignal
        
        progress = QProgressDialog("Generating Comprehensive ECG Analysis Report. Please wait...", None, 0, 0, self)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModal)
        progress.setStyleSheet(f"QProgressDialog{{background:{COL_DARK};color:{COL_GREEN};}}")
        progress.setRange(0, 0)
        progress.show()
        
        class ReportWorker(QThread):
            finished = pyqtSignal(str)
            error = pyqtSignal(str)
            
            def __init__(self, session_dir, patient_info, summary):
                super().__init__()
                self.session_dir = session_dir
                self.patient_info = patient_info
                self.summary = summary
                
            def run(self):
                try:
                    from .report_generator import generate_holter_report
                    path = generate_holter_report(
                        session_dir=self.session_dir,
                        patient_info=self.patient_info,
                        summary=self.summary,
                    )
                    self.finished.emit(path)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.error.emit(str(e))
                    
        self._report_worker = ReportWorker(self.session_dir, self.patient_info, self._summary)
        
        def on_finished(path):
            progress.close()
            try:
                from dashboard.history_window import append_history_entry
                h_pat = self.patient_info.copy() if self.patient_info else {}
                if 'patient_name' not in h_pat and 'name' in h_pat:
                    h_pat['patient_name'] = h_pat['name']
                _p = self.parent()
                _uname = getattr(_p, "username", "") if _p is not None else ""
                _full = (getattr(_p, "user_details", {}) or {}).get("full_name") or _uname
                append_history_entry(
                    h_pat, path, report_type="Comprehensive ECG Analysis",
                    username=_uname, owner_full_name=_full
                )
            except Exception as h_err:
                print(f"Failed to append Holter history: {h_err}")
                
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Generated")
            msg.setText(f"Comprehensive ECG Analysis report saved:\n{path}")
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet(f"QMessageBox {{ background: {COL_BLACK}; }} QLabel {{ color: {COL_WHITE}; font-size: 12px; }}")
            msg.exec_()
            
        def on_error(err_str):
            progress.close()
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Error")
            msg.setText(f"Could not generate report:\n{err_str}")
            msg.setIcon(QMessageBox.Warning)
            msg.setStyleSheet(f"QMessageBox {{ background: {COL_BLACK}; }} QLabel {{ color: {COL_WHITE}; font-size: 12px; }}")
            msg.exec_()
            
        self._report_worker.finished.connect(on_finished)
        self._report_worker.error.connect(on_error)
        self._report_worker.start()

    def attach_writer(self, writer, session_dir: str = "", patient_info: dict = None):
        self._writer = writer
        self._last_live_seq = -1
        if session_dir:
            self.session_dir = session_dir
        if patient_info is not None:
            self.patient_info = _normalize_patient_info(patient_info)
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.set_session_dir(self.session_dir)
        if hasattr(self, '_events_panel'):
            self._events_panel.set_session_dir(self.session_dir)
        if hasattr(self, '_edit_strips_panel'):
            self._edit_strips_panel.set_session_dir(self.session_dir)
        if writer and not hasattr(self, '_status_bar'):
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            self.layout().insertWidget(1, self._status_bar)
        if writer and not hasattr(self, '_live_timer'):
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._update_live_ui)
        if writer and hasattr(self, '_live_timer') and not self._live_timer.isActive():
            self._live_timer.start(1000)
        self._refresh_ui()

    def load_completed_session(self, session_dir: str, patient_info: dict = None):
        self.session_dir = session_dir
        self._last_live_seq = -1
        self.patient_info = _normalize_patient_info(patient_info or {})
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.set_session_dir(self.session_dir)
        if hasattr(self, '_events_panel'):
            self._events_panel.set_session_dir(self.session_dir)
        if hasattr(self, '_edit_strips_panel'):
            self._edit_strips_panel.set_session_dir(self.session_dir)
        if getattr(self, "_replay_engine", None) and self._replay_engine.is_playing():
            self._replay_engine.pause()
        if hasattr(self, "_replay_panel"):
            try:
                self._replay_panel._slider.blockSignals(True)
                self._replay_panel._slider.setValue(self._replay_panel._slider_sec_to_value(0))
                self._replay_panel._slider.blockSignals(False)
                self._replay_panel._pos_label.setText(_sec_to_hms(0))
                self._replay_panel._play_btn.setText("Play")
            except Exception:
                pass
            self._set_record_browser_enabled(True)
        self._writer = None
        self._load_session()
        if hasattr(self, '_record_mgmt_panel'):
            self._record_mgmt_panel.output_dir = _resolve_recordings_dir(session_dir)
            self._record_mgmt_panel.set_active_session(self.session_dir)
        if hasattr(self, '_replay_panel') and getattr(self, '_replay_engine', None):
            self._replay_panel.set_replay_engine(self._replay_engine)
            try:
                self._replay_panel.seek_requested.disconnect(self._on_seek_requested)
            except Exception:
                pass
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        if hasattr(self, '_template_panel') and getattr(self, '_replay_engine', None):
            try:
                self._template_panel.set_replay_engine(self._replay_engine)
            except Exception:
                pass
        self._refresh_ui()
        if hasattr(self, '_tabs'):
            self._tabs.setCurrentIndex(0)

    def close_current_session_if_needed(self, session_to_delete: str) -> bool:
        """
        Closes the current session if it matches the session being deleted.
        Returns True if the session was closed.
        """
        if not hasattr(self, 'session_dir') or not self.session_dir:
            return False
        if not os.path.normpath(self.session_dir) == os.path.normpath(session_to_delete):
            return False
        if hasattr(self, "_replay_engine") and self._replay_engine:
            try:
                self._replay_engine.close()
            except Exception:
                pass
            self._replay_engine = None
        self.session_dir = None
        return True

    def changeEvent(self, event):
        """Handle window state changes to preserve maximized state."""
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            # When window state changes, re-apply maximized state
            if self.isVisible():
                QTimer.singleShot(100, self.showMaximized)

    def showEvent(self, event):
        """Ensure window maintains maximized state when shown."""
        super().showEvent(event)
        # Restore maximized state when window is shown
        QTimer.singleShot(50, self.showMaximized)

    def closeEvent(self, event):
        if self._writer:
            try:
                self._finalize_live_writer()
            except Exception:
                pass
        if self._replay_engine:
            try:
                self._replay_engine.close()
            except Exception:
                pass
        super().closeEvent(event)
