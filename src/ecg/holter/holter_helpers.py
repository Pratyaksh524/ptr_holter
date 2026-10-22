"""Shared helper utilities for the Comprehensive ECG Analysis UI."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from PyQt5.QtWidgets import QMessageBox

from .theme import (
    COL_BLACK,
    COL_BG,
    COL_BTN_ACTIVE_BG,
    COL_BTN_ACTIVE_TEXT,
    COL_DARK,
    COL_GREEN,
    COL_GREEN_DRK,
    COL_GREEN_MID,
    COL_GRAY,
    COL_RED,
    COL_TEXT,
    COL_TIMESTAMP,
    COL_WHITE,
    COL_YELLOW,
)


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
UI_GRID = "#1A2A44"   # faint table grid lines


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


def _metrics_duration_sec(metrics_list: list) -> float:
    return float(sum(m.get("duration", 0.0) or 0.0 for m in metrics_list))


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


def _style_filter_btn(active: bool = False, bg: str = UI_PANEL_ALT, fg: str = UI_TEXT, hover: str = "#1A2C49") -> str:
    if active:
        return f"""
            QPushButton {{
                background: {UI_ACCENT};
                color: #FFFFFF;
                border: 1px solid #5EA4FF;
                border-radius: 6px;
                padding: 2px 6px;
                font-size: 11px;
                font-weight: bold;
                min-height: 24px;
                max-height: 28px;
            }}
            QPushButton:hover {{ background: {UI_ACCENT_HOVER}; color: #FFFFFF; }}
        """
    return f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border: 1px solid {UI_BORDER};
            border-radius: 6px;
            padding: 2px 6px;
            font-size: 11px;
            font-weight: bold;
            min-height: 24px;
            max-height: 28px;
        }}
        QPushButton:hover {{
            background: {hover};
            color: #FFFFFF;
            border: 1px solid {UI_ACCENT};
        }}
        QPushButton:pressed {{ background: {bg}; border: 1px solid {UI_ACCENT_HOVER}; }}
        QPushButton:disabled {{ background: #1A2233; color: #5F708A; border: 1px solid #2A3953; }}
    """


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
            border: 1px solid {UI_ACCENT};
        }}
        QPushButton:pressed {{ background: {bg}; border: 1px solid {UI_ACCENT_HOVER}; }}
        QPushButton:disabled {{ background: #1A2233; color: #5F708A; border: 1px solid #2A3953; }}
    """


def _card_style(selector: str = "QFrame") -> str:
    """Rounded panel card used to group a section of a screen."""
    return f"{selector}{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:8px;}}"


def _card_title_style() -> str:
    """Section heading inside a card: bold text with an accent bar on the left."""
    return (f"QLabel{{color:{UI_TEXT};font-size:12px;font-weight:700;background:transparent;"
            f"border:none;border-left:3px solid {UI_ACCENT};padding:2px 8px;}}")


def _overview_table_style() -> str:
    """Name / Value summary tables (Overview and Replay side panels)."""
    return f"""
        QTableWidget {{
            background-color: {UI_PANEL};
            color: {UI_TEXT};
            border: 1px solid {UI_BORDER};
            border-radius: 6px;
            gridline-color: {UI_GRID};
            font-size: 11px;
            outline: none;
        }}
        QHeaderView {{ background: {UI_PANEL_ALT}; border: none; }}
        QHeaderView::section {{
            background: {UI_PANEL_ALT};
            color: {UI_MUTED};
            font-weight: bold;
            font-size: 11px;
            padding-left: 8px;
            border: none;
            border-bottom: 1px solid {UI_BORDER};
            height: 26px;
        }}
        QTableWidget::item {{
            padding-left: 8px;
            padding-right: 8px;
            border-bottom: 1px solid {UI_GRID};
        }}
    """


def _global_style() -> str:
    """Window-wide defaults for widgets that don't carry their own style:
    scrollbars, splitter handles, tooltips and context menus."""
    return f"""
        QScrollBar:vertical {{ background: {UI_PANEL}; width: 10px; margin: 0; border: none; }}
        QScrollBar::handle:vertical {{ background: {UI_BORDER}; min-height: 24px; border-radius: 5px; margin: 2px; }}
        QScrollBar::handle:vertical:hover {{ background: #3A5378; }}
        QScrollBar:horizontal {{ background: {UI_PANEL}; height: 10px; margin: 0; border: none; }}
        QScrollBar::handle:horizontal {{ background: {UI_BORDER}; min-width: 24px; border-radius: 5px; margin: 2px; }}
        QScrollBar::handle:horizontal:hover {{ background: #3A5378; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
        QSplitter::handle {{ background: {UI_BORDER}; }}
        QSplitter::handle:hover {{ background: {UI_ACCENT}; }}
        QToolTip {{ background: {UI_PANEL_ALT}; color: {UI_TEXT}; border: 1px solid {UI_BORDER}; padding: 5px 8px; border-radius: 4px; }}
        QMenu {{ background: {UI_PANEL}; color: {UI_TEXT}; border: 1px solid {UI_BORDER}; padding: 4px; }}
        QMenu::item {{ padding: 6px 22px 6px 14px; border-radius: 4px; }}
        QMenu::item:selected {{ background: #1E3A5F; }}
        QMenu::separator {{ height: 1px; background: {UI_BORDER}; margin: 4px 6px; }}
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
            gridline-color: {UI_GRID};
            font-size: 12px;
            border: 1px solid {UI_BORDER};
            border-radius: 6px;
            selection-background-color: #1E3A5F;
            selection-color: {UI_TEXT};
            outline: none;
        }}
        QHeaderView {{ background: {UI_PANEL_ALT}; border: none; }}
        QHeaderView::section {{
            background: {UI_PANEL_ALT};
            color: {UI_MUTED};
            font-size: 11px;
            font-weight: bold;
            padding: 6px;
            border: none;
            border-right: 1px solid {UI_GRID};
            border-bottom: 1px solid {UI_BORDER};
        }}
        QHeaderView::section:last {{ border-right: none; }}
        QTableCornerButton::section {{ background: {UI_PANEL_ALT}; border: none; border-bottom: 1px solid {UI_BORDER}; }}
        QTableWidget::item {{ padding: 5px; border: none; }}
        QTableWidget::item:selected {{ background: #1E3A5F; color: {UI_TEXT}; }}
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


def _sec_to_hms(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


# _format_system_time() is called once per row by the Events/Edit Event/Edit
# Strips tables -- thousands of times for a long recording (10k+ rows on a
# 14h session). It used to re-open and re-json.load() recording_index.json
# (a per-second index: ~3MB/50k entries for 14h) on every single call, which
# alone accounted for most of the multi-minute UI freeze when opening a long
# session. recording_index.json is only ever written once, by
# ECGHFileWriter.finalize() (file_format.py) -- a completed session's copy
# never changes -- so caching start_time per session_dir is always safe; the
# mtime check below only exists to tolerate a stale cache across repeated
# live-recording finalize()/reopen cycles within the same process.
_system_time_start_cache: dict = {}


def _recording_start_time(session_dir: str):
    index_path = os.path.join(session_dir, "recording_index.json")
    try:
        mtime = os.path.getmtime(index_path)
    except OSError:
        return None
    cached = _system_time_start_cache.get(session_dir)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            start_time = json.load(f).get("start_time")
    except Exception as e:
        print(f"[HolterUI] Error reading system time: {e}")
        start_time = None
    _system_time_start_cache[session_dir] = (mtime, start_time)
    return start_time


def _format_system_time(session_dir: str, chunk_timestamp: float) -> str:
    """Convert relative timestamp to actual system wall-clock time."""
    if not session_dir:
        return _sec_to_hms(chunk_timestamp)
    start_time = _recording_start_time(session_dir)
    if start_time:
        system_time = datetime.fromtimestamp(start_time + chunk_timestamp)
        return system_time.strftime("%H:%M:%S")
    return _sec_to_hms(chunk_timestamp)


def _get_recording_start_end_times(session_dir: str, duration_sec: float) -> tuple:
    """Return recording start and end date-time strings."""
    if not session_dir:
        return "Unknown", "Unknown"
    try:
        index_path = os.path.join(session_dir, "recording_index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            start_time = index_data.get("start_time")
            if start_time:
                start_dt = datetime.fromtimestamp(start_time)
                end_dt = datetime.fromtimestamp(start_time + duration_sec)
                return start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"[HolterUI] Error reading recording start/end times: {e}")
    return "Unknown", "Unknown"


def _show_message_box(parent, icon, title, text, buttons=QMessageBox.Ok, default_button=QMessageBox.NoButton):
    """Display a styled QMessageBox modal dialog."""
    msg_box = QMessageBox(parent)
    msg_box.setIcon(icon)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    msg_box.setStandardButtons(buttons)
    msg_box.setDefaultButton(default_button)
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
            border-radius: 4px;
            padding: 5px 15px;
            min-width: 60px;
        }}
        QMessageBox QPushButton:hover {{
            background-color: {UI_ACCENT};
            border-color: {UI_ACCENT_HOVER};
        }}
    """)
    return msg_box.exec_()


