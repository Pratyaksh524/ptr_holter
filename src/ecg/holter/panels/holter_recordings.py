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
)

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas

# 10. HOLTER RECORDINGS PANEL
class HolterRecordManagementPanel(QWidget):
    session_selected = pyqtSignal(str)  # session dir path

    def __init__(self, output_dir: str = "recordings"):
        super().__init__()
        self.output_dir = output_dir
        self._selected_session = ""
        self._active_session = ""
        self._build_ui()
        self.refresh_records()
        
    def set_active_session(self, session_dir: str):
        """Set the active session to highlight in the table."""
        self._active_session = session_dir
        self.refresh_records()

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
        layout.setSpacing(6)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search patient / reporter / status")
        self._search.setStyleSheet(f"QLineEdit{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                                   f"border-radius:4px;padding:6px;font-size:12px;}}")
        self._search.textChanged.connect(self.refresh_records)
        self._filter = QComboBox()
        self._filter.addItems(["All", "Today", "Yesterday", "This Week", "This Month", "This Year"])
        self._filter.setStyleSheet(f"""
            QComboBox {{
                background:{COL_DARK}; color:{COL_GREEN}; border:1px solid {COL_GREEN_DRK};
                border-radius:4px; padding:6px; font-size:12px;
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        self._filter.currentTextChanged.connect(self.refresh_records)
        actions.addWidget(QLabel("Search:", styleSheet=f"color:{COL_GREEN};font-size:12px;"))
        actions.addWidget(self._search, 2)
        actions.addWidget(QLabel("Filter:", styleSheet=f"color:{COL_GREEN};font-size:12px;"))
        actions.addWidget(self._filter)
        self._action_buttons = {}
        for txt in ["Browse", "Import", "Export", "Backup", "Delete"]:
            btn = QPushButton(txt)
            btn.setStyleSheet(_style_btn())
            self._action_buttons[txt] = btn
            actions.addWidget(btn)
        layout.addLayout(actions)

        cols = ["Name","Age","Gender","Record Time","Duration","Channel","Import Time","Status","Reporter","Conclusion"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet(
            _table_style() +
            """
            QTableWidget::item:selected {
                background-color: rgba(66, 153, 225, 70);
                color: #F3F7FB;
                border: 1px solid rgba(255, 255, 255, 90);
            }
            QTableWidget::item:selected:active {
                background-color: rgba(66, 153, 225, 110);
            }
            """
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_record_context_menu)
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self._table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._table.verticalScrollBar().setSingleStep(1)
        self._table.itemSelectionChanged.connect(self._sync_selected_session)
        self._table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table, 1)
        self._action_buttons["Browse"].clicked.connect(self._browse_root)
        self._action_buttons["Import"].clicked.connect(self._import_session)
        self._action_buttons["Export"].clicked.connect(self._export_session)
        self._action_buttons["Backup"].clicked.connect(self._backup_root)
        self._action_buttons["Delete"].clicked.connect(self._delete_session)

    def refresh_records(self):
        self._table.setRowCount(0)
        self._selected_session = ""
        if not os.path.isdir(self.output_dir): return
        query = self._search.text().strip().lower()
        filter_label = self._filter.currentText()
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        rows = []
        for name in sorted(os.listdir(self.output_dir), reverse=True):
            session_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(session_dir): continue
            if not os.path.exists(os.path.join(session_dir, "recording.ecgh")): continue
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(session_dir, "recording.ecgh")))
            except Exception:
                mtime = datetime.fromtimestamp(os.path.getmtime(session_dir))
            delta_days = (now - mtime).days
            if filter_label == "Today" and mtime.date() != today:
                continue
            if filter_label == "Yesterday" and mtime.date() != yesterday:
                continue
            if filter_label == "This Week" and delta_days > 6:
                continue
            if filter_label == "This Month" and (mtime.year != now.year or mtime.month != now.month):
                continue
            if filter_label == "This Year" and mtime.year != now.year:
                continue
            parts = name.split("_", 3)
            rec_time = "_".join(parts[:2]).replace("_", " ") if len(parts) >= 2 else name[:19]
            p_name = parts[-1].replace("_", " ") if len(parts) >= 3 else "Unknown"
            age = "-"
            gender = "-"
            dur_str = "-"
            
            import json
            # Check patient.json first (created by Save dialog)
            patient_json = os.path.join(session_dir, "patient.json")
            if os.path.exists(patient_json):
                try:
                    with open(patient_json, 'r') as f:
                        pdata = json.load(f)
                    saved_name = pdata.get('name') or pdata.get('patient_name') or pdata.get('full_name')
                    if saved_name and str(saved_name).strip() and str(saved_name).strip().lower() != "unknown":
                        p_name = str(saved_name).strip()
                    saved_age = pdata.get('age')
                    if saved_age: age = str(saved_age)
                    saved_gender = pdata.get('gender') or pdata.get('sex')
                    if saved_gender: gender = str(saved_gender)
                except Exception:
                    pass

            try:
                session_json = os.path.join(session_dir, "session.json")
                if os.path.exists(session_json):
                    with open(session_json, 'r') as f:
                        sdata = json.load(f)
                        
                    # If we didn't find good name in patient.json, check session.json
                    if p_name.lower() == "unknown":
                        p_info = sdata.get('patient_info') or sdata.get('summary', {}).get('patient_info') or {}
                        saved_name = p_info.get('name') or p_info.get('patient_name') or p_info.get('full_name')
                        if saved_name and str(saved_name).strip() and str(saved_name).strip().lower() != "unknown":
                            p_name = str(saved_name).strip()
                        saved_age = p_info.get('age')
                        if saved_age and age == "-": age = str(saved_age)
                        saved_gender = p_info.get('gender') or p_info.get('sex')
                        if saved_gender and gender == "-": gender = str(saved_gender)

                    dur_sec = sdata.get('summary', {}).get('duration_sec', 0)
                    if dur_sec > 0:
                        h = int(dur_sec // 3600)
                        m = int((dur_sec % 3600) // 60)
                        s = int(dur_sec % 60)
                        if h > 0:
                            dur_str = f"{h}h {m:02d}m"
                        elif m > 0:
                            dur_str = f"{m}m {s:02d}s"
                        else:
                            dur_str = f"{s}s"
            except Exception:
                pass

            row_values = [p_name, age, gender, rec_time, dur_str, "12", rec_time, "Completed", "System", "-"]
            if query and not any(query in str(v).lower() for v in row_values): continue
            rows.append((row_values, session_dir))

        for row_values, session_dir in rows:
            r = self._table.rowCount()
            self._table.insertRow(r)
            
            is_active = os.path.normpath(session_dir) == os.path.normpath(self._active_session)
            for c, v in enumerate(row_values):
                item = QTableWidgetItem(str(v))
                item.setForeground(QColor(COL_GREEN if c == 0 else COL_WHITE))
                item.setData(Qt.UserRole, session_dir)
                if is_active:
                    item.setBackground(QColor(46, 84, 137, 150))  # Semi-transparent blue highlight
                self._table.setItem(r, c, item)
        if self._table.rowCount() > 0:
            self._table.selectRow(0)
            self._sync_selected_session()

    def _open_row(self, row, _column=0):
        item = self._table.item(row, 0)
        if item:
            path = item.data(Qt.UserRole)
            if path:
                self._selected_session = path
                self.session_selected.emit(path)

    def _on_double_click(self, index):
        self._open_row(index.row())

    def _show_record_context_menu(self, position):
        """Show actions for the recording row under the mouse."""
        item = self._table.itemAt(position)
        if item is None:
            return
        self._table.selectRow(item.row())
        menu = QMenu(self)
        edit_action = menu.addAction("Edit Details")
        edit_action.triggered.connect(lambda: self._edit_record_details(item.row()))
        menu.exec_(self._table.viewport().mapToGlobal(position))

    def _edit_record_details(self, row: int):
        """Edit patient metadata for one completed recording."""
        item = self._table.item(row, 0)
        session_dir = item.data(Qt.UserRole) if item else ""
        if not session_dir:
            return

        metadata = {}
        patient_path = os.path.join(session_dir, "patient.json")
        session_path = os.path.join(session_dir, "session.json")
        try:
            if os.path.exists(patient_path):
                with open(patient_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle) or {}
            if not metadata and os.path.exists(session_path):
                with open(session_path, "r", encoding="utf-8") as handle:
                    session_data = json.load(handle) or {}
                metadata = dict(session_data.get("patient_info") or {})
        except Exception as exc:
            _show_message_box(self, QMessageBox.Warning, "Edit Details", f"Could not read patient details:\n{exc}")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Recording Details")
        dialog.setModal(True)
        dialog.setFixedWidth(390)
        dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {UI_BG};
                color: {UI_TEXT};
            }}
            QLabel {{
                color: #FFFFFF;
                font-size: 13px;
                font-weight: 600;
            }}
            QLineEdit, QComboBox {{
                min-height: 30px;
                padding: 3px 8px;
                border: 1px solid {UI_BORDER};
                border-radius: 4px;
                background-color: {UI_PANEL_ALT};
                color: #FFFFFF;
                selection-background-color: {UI_ACCENT};
                selection-color: #FFFFFF;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 1px solid {UI_ACCENT};
            }}
            QComboBox QAbstractItemView {{
                background-color: {UI_PANEL_ALT};
                color: #FFFFFF;
                selection-background-color: {UI_ACCENT};
                selection-color: #FFFFFF;
            }}
            QPushButton {{
                min-width: 82px;
                min-height: 30px;
                padding: 4px 14px;
                border: 1px solid {UI_BORDER};
                border-radius: 4px;
                background-color: {UI_PANEL_ALT};
                color: #FFFFFF;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {UI_ACCENT_HOVER};
            }}
        """)
        form = QGridLayout(dialog)
        form.setContentsMargins(18, 18, 18, 14)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        name_edit = QLineEdit(str(metadata.get("name") or metadata.get("patient_name") or ""))
        age_edit = QLineEdit(str(metadata.get("age") or ""))
        gender_edit = QComboBox()
        gender_edit.addItems(["Select Gender", "Male", "Female", "Other"])
        current_gender = str(metadata.get("gender") or metadata.get("sex") or "")
        gender_index = gender_edit.findText(current_gender, Qt.MatchFixedString)
        if gender_index >= 0:
            gender_edit.setCurrentIndex(gender_index)
        form.addWidget(QLabel("Name:"), 0, 0)
        form.addWidget(name_edit, 0, 1)
        form.addWidget(QLabel("Age:"), 1, 0)
        form.addWidget(age_edit, 1, 1)
        form.addWidget(QLabel("Gender:"), 2, 0)
        form.addWidget(gender_edit, 2, 1)
        buttons = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        save_button.setStyleSheet(f"""
            QPushButton {{ background-color: {UI_ACCENT}; color: #FFFFFF; border: none; }}
            QPushButton:hover {{ background-color: {UI_ACCENT_HOVER}; }}
        """)
        cancel_button.setStyleSheet(f"""
            QPushButton {{ background-color: {UI_PANEL}; color: #FFFFFF; }}
            QPushButton:hover {{ background-color: {UI_PANEL_ALT}; }}
        """)
        buttons.addStretch(1)
        buttons.addWidget(save_button)
        buttons.addWidget(cancel_button)
        form.addLayout(buttons, 3, 0, 1, 2)
        save_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        if dialog.exec_() != QDialog.Accepted:
            return
        name = name_edit.text().strip()
        age = age_edit.text().strip()
        gender = gender_edit.currentText()
        if not name or not age or gender == "Select Gender":
            _show_message_box(self, QMessageBox.Warning, "Edit Details", "Please enter name, age, and gender.")
            return

        updated = dict(metadata)
        updated.update({"name": name, "patient_name": name, "age": age, "gender": gender})
        try:
            with open(patient_path, "w", encoding="utf-8") as handle:
                json.dump(updated, handle, indent=2)
            session_data = {}
            if os.path.exists(session_path):
                with open(session_path, "r", encoding="utf-8") as handle:
                    session_data = json.load(handle) or {}
            session_data["patient_info"] = dict(updated)
            with open(session_path, "w", encoding="utf-8") as handle:
                json.dump(session_data, handle, indent=2)
            self.refresh_records()
            _show_message_box(self, QMessageBox.Information, "Edit Details", "Recording details updated successfully.")
        except Exception as exc:
            _show_message_box(self, QMessageBox.Warning, "Edit Details", f"Could not save patient details:\n{exc}")

    def _sync_selected_session(self):
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if rows:
            item = self._table.item(rows[0].row(), 0)
            if item:
                self._selected_session = item.data(Qt.UserRole) or ""

    def _selected_path(self) -> str:
        self._sync_selected_session()
        return self._selected_session

    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Recordings Root", self.output_dir or os.getcwd())
        if d:
            self.output_dir = d
            self.refresh_records()

    def _import_session(self):
        src = QFileDialog.getExistingDirectory(self, "Import Session Folder")
        if not src:
            return
        if not os.path.exists(os.path.join(src, "recording.ecgh")):
            _show_message_box(self, QMessageBox.Warning, "Import Session", "Select a session folder that contains recording.ecgh.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        dest = os.path.join(self.output_dir, os.path.basename(os.path.normpath(src)))
        if os.path.exists(dest):
            _show_message_box(self, QMessageBox.Warning, "Import Session", "That session already exists in the recordings folder.")
            return
        shutil.copytree(src, dest)
        self.refresh_records()
        self.session_selected.emit(dest)

    def _export_session(self):
        src = self._selected_path()
        if not src:
            _show_message_box(self, QMessageBox.Information, "Export Session", "Select a recording to export first.")
            return
        dest_root = QFileDialog.getExistingDirectory(self, "Export Session To")
        if not dest_root:
            return
        dest = os.path.join(dest_root, os.path.basename(os.path.normpath(src)))
        if os.path.exists(dest):
            _show_message_box(self, QMessageBox.Warning, "Export Session", "That session already exists in the destination.")
            return
        shutil.copytree(src, dest)
        _show_message_box(self, QMessageBox.Information, "Export Session", f"Session exported to:\n{dest}")

    def _backup_root(self):
        if not os.path.isdir(self.output_dir):
            _show_message_box(self, QMessageBox.Information, "Backup", "No recordings folder found.")
            return
        dest_root = QFileDialog.getExistingDirectory(self, "Backup Recordings To")
        if not dest_root:
            return
        dest = os.path.join(dest_root, os.path.basename(os.path.normpath(self.output_dir)) or "recordings_backup")
        if os.path.exists(dest):
            _show_message_box(self, QMessageBox.Warning, "Backup", "That backup folder already exists.")
            return
        shutil.copytree(self.output_dir, dest)
        _show_message_box(self, QMessageBox.Information, "Backup", f"Recordings backed up to:\n{dest}")

    def _delete_session(self):
        src = self._selected_path()
        if not src:
            _show_message_box(self, QMessageBox.Information, "Delete Session", "Select a recording to delete first.")
            return
        if _show_message_box(self, QMessageBox.Question, "Delete Session",
                                f"Delete this recording?\n\n{src}",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        
        # First, try to find the main window and close the session if it's currently loaded
        try:
            # Find the main HolterUI window
            main_window = self.window()
            if main_window and hasattr(main_window, "close_current_session_if_needed"):
                main_window.close_current_session_if_needed(src)
        except Exception:
            pass
        
        # Force garbage collection to clean up any unreferenced file handles
        gc.collect()
        
        # First, try to delete individual files with retries
        max_file_retries = 4
        last_error = None
        
        # Try to delete all files in the directory first
        if os.path.exists(src):
            try:
                for root, dirs, files in os.walk(src, topdown=False):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        # Try to delete this file with retries
                        for file_attempt in range(max_file_retries):
                            try:
                                os.remove(file_path)
                                break
                            except Exception as e:
                                last_error = e
                                if file_attempt < max_file_retries - 1:
                                    gc.collect()
                                    time.sleep(0.3)
                                else:
                                    pass
                    # Try to delete subdirectories
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        for dir_attempt in range(max_file_retries):
                            try:
                                os.rmdir(dir_path)
                                break
                            except Exception as e:
                                last_error = e
                                if dir_attempt < max_file_retries - 1:
                                    gc.collect()
                                    time.sleep(0.3)
                                else:
                                    pass
            except Exception as e:
                last_error = e
        
        # Now try rmtree
        max_retries = 7
        retry_delay = 0.6
        
        for attempt in range(max_retries):
            try:
                gc.collect()
                if os.path.exists(src):
                    shutil.rmtree(src, ignore_errors=False)
                self.refresh_records()
                _show_message_box(self, QMessageBox.Information, "Delete Session", "Recording deleted successfully.")
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    gc.collect()
                    time.sleep(retry_delay)
                    retry_delay *= 1.4  # exponential backoff
                else:
                    break
        
        # If we got here, all retries failed
        _show_message_box(self, QMessageBox.Warning, "Delete Session", f"Failed to delete recording:\n{str(last_error)}")
