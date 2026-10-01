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
    _show_message_box, _style_active_btn, _style_btn, _style_filter_btn, _table_style, _template_filter_key,
)

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas

# 8. HOLTER TEMPLATE PANEL (template gallery)
class _TemplateMetricCard(QFrame):
    def __init__(self, title: str, value: str = "", accent: str = COL_GREEN, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1B2433, stop:1 #111827);border:1px solid {UI_BORDER};border-radius:10px;}}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color:{UI_MUTED};font-size:10px;font-weight:700;border:none;")
        self.value = QLabel(value)
        self.value.setStyleSheet(f"color:{UI_TEXT};font-size:16px;font-weight:800;border:none;")
        layout.addWidget(lbl)
        layout.addWidget(self.value)

class TemplateCardWidget(QFrame):
    clicked = pyqtSignal(object, object)  # (card, event)
    double_clicked = pyqtSignal(object)   # (card)
    template_id_changed = pyqtSignal(object, str)
    class_changed = pyqtSignal(object, str)
    viewed_changed = pyqtSignal(object, bool)

    def __init__(self, parent=None, accent: str = UI_BORDER):
        super().__init__(parent)
        self._accent = accent or UI_BORDER
        self._selected = False
        self._hovered = False
        self._template_key = ""
        self.setObjectName("templateCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)
        self._build_ui()
        self.set_accent(self._accent)

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
        self.setMinimumHeight(170)
        self.setFixedWidth(320)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._card_style = """
            QFrame#templateCard {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {bg_top}, stop:1 {bg_bottom});
                border: 1px solid {border};
                border-radius: 14px;
            }}
            QLineEdit, QComboBox {{
                background: #0C1320;
                color: {ui_text};
                border: 1px solid {border};
                border-radius: 7px;
                padding: 4px 7px;
                font-size: 11px;
            }}
            QComboBox QAbstractItemView {{
                background: #0B1220;
                color: {ui_text};
                selection-background-color: {ui_accent};
                selection-color: #07111F;
                border: 1px solid {border};
                outline: 0;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 4px 8px;
                min-height: 18px;
            }}
            QComboBox::drop-down {{ border: none; width: 16px; }}
            QLineEdit:focus, QComboBox:focus {{ border-color: {ui_accent}; }}
        """

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.id_edit = QLineEdit("T1")
        self.id_edit.setFixedWidth(70)
        self.id_edit.setPlaceholderText("T ID")
        self.class_combo = QComboBox()
        # Updated options: N, S, V, P, AF, X, Other, Unconfirmed
        self.class_combo.addItems(["N", "S", "V", "P", "AF", "X", "Other", "Unconfirmed"])
        self.class_combo.setFixedWidth(95)  # Increased width for "Unconfirmed"
        self.count_pill = QLabel("0 beats")
        self.count_pill.setAlignment(Qt.AlignCenter)
        self.count_pill.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:800;background:#152235;border:1px solid {self._accent};border-radius:10px;padding:4px 8px;")
        header.addWidget(self.id_edit)
        header.addWidget(self.class_combo)
        header.addStretch()
        header.addWidget(self.count_pill)
        layout.addLayout(header)

        self.badge_container = QWidget()
        badge_row = QHBoxLayout(self.badge_container)
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.setSpacing(5)
        badge_row.addStretch()
        self.badge_labels = {}
        badges_info = [
            ("ambiguous", "?", "It means that at least one classification channel of the beat under the template is not similar, so the template ECG waveform is empty, which is represented by '?'"),
            ("inserted", "+", "Templates classified by beats of inserting beats, creating new templates, or merging templates are marked with '+'"),
            ("demix", "DEMIX", "In the figure, DEMIX represents the template generated by overlay beat, new template operation after reclassification or interval reanalysis, and II represents the overlay channel used by the template."),
            ("auto_update", "AUTO", "Template generated by updating the beat is represented by 'N'")
        ]
        for key, txt, tooltip in badges_info:
            badge = QLabel(txt)
            badge.setToolTip(tooltip)
            badge.setAlignment(Qt.AlignCenter)
            badge.setVisible(False)
            badge.setStyleSheet(f"color:{UI_TEXT};font-size:10px;font-weight:800;background:#17243A;border:1px solid {UI_BORDER};border-radius:8px;padding:2px 6px;")
            self.badge_labels[key] = badge
            badge_row.addWidget(badge)
        self.setToolTip("It means that all the beat forms under the template are similar in the template")
        self.badge_container.setVisible(False)
        layout.addWidget(self.badge_container)

        self.thumb = ECGStripCanvas(height=68, color="#39D353", pen_width=1.0)
        self.thumb.setStyleSheet(f"background:{COL_BLACK};border:1px solid {self._accent};border-radius:10px;")
        layout.addWidget(self.thumb, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.template_no = QLabel("#T1")
        self.template_no.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:700;border:none;")
        self.view_toggle = QToolButton()
        self.view_toggle.setCheckable(True)
        self.view_toggle.setChecked(True)
        self.view_toggle.setText("\u25c9")
        self.view_toggle.setToolTip("Viewed / unconfirmed")
        self.view_toggle.setFixedSize(26, 26)
        self.view_toggle.setStyleSheet(f"""
            QToolButton {{
                border: 1px solid {UI_BORDER};
                border-radius: 13px;
                color: {UI_TEXT};
                background: #0C1320;
                font-weight: 900;
            }}
            QToolButton:hover {{
                border-color: {UI_ACCENT_HOVER};
                background: #11203A;
            }}
            QToolButton:checked {{
                background: {UI_SUCCESS};
                color: #07111F;
                border-color: {UI_SUCCESS};
            }}
        """)
        bottom.addWidget(self.template_no)
        bottom.addStretch()
        bottom.addWidget(self.view_toggle)
        layout.addLayout(bottom)

        self.id_edit.editingFinished.connect(self._on_id_edit_finished)
        self.class_combo.currentTextChanged.connect(self._on_class_changed)
        self.view_toggle.toggled.connect(self._on_view_toggle)

        for widget in [self.id_edit, self.class_combo, self.count_pill, self.badge_container, self.thumb, self.template_no, self.view_toggle]:
            try:
                widget.installEventFilter(self)
            except Exception:
                pass
        for badge in self.badge_labels.values():
            try:
                badge.installEventFilter(self)
            except Exception:
                pass

    def eventFilter(self, watched, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if watched not in (self.id_edit, self.class_combo, self.view_toggle):
                self.clicked.emit(self, event)
                return False
        elif event.type() == QEvent.MouseButtonDblClick and event.button() == Qt.LeftButton:
            if watched not in (self.id_edit, self.class_combo, self.view_toggle):
                self.double_clicked.emit(self)
                return True
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self, event)
        return super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self)
        return super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        host = self._find_template_host()
        if host is not None and hasattr(host, "_show_template_card_menu"):
            host._show_template_card_menu(self, event.globalPos())
            event.accept()
            return
        return super().contextMenuEvent(event)

    def _on_view_toggle(self, checked: bool):
        self.view_toggle.setText("\u25c9" if checked else "\u25cb")
        self.viewed_changed.emit(self, bool(checked))

    def _on_id_edit_finished(self):
        try:
            self.template_id_changed.emit(self, self.id_edit.text())
        except Exception as e:
            print(f"Error in _on_id_edit_finished: {e}")

    def _on_class_changed(self, text: str):
        try:
            # Defer processing to avoid destroying widget during signal emission
            QTimer.singleShot(0, lambda: self._handle_class_changed(text))
        except Exception as e:
            print(f"Error in _on_class_changed: {e}")
    
    def _handle_class_changed(self, text: str):
        try:
            self.class_changed.emit(self, text)
        except Exception as e:
            print(f"Error in _handle_class_changed: {e}")

    def _apply_styles(self):
        if self._selected:
            border = UI_ACCENT_HOVER
            bg_top = "#18263C"
            bg_bottom = "#101B2B"
        else:
            border = "#6EB4FF" if self._hovered else self._accent
            bg_top = "#141F31" if self._hovered else "#121C2D"
            bg_bottom = "#101827" if self._hovered else "#0D1521"
        self.setStyleSheet(self._card_style.format(border=border, bg_top=bg_top, bg_bottom=bg_bottom, ui_text=UI_TEXT, ui_accent=UI_ACCENT))
        self.count_pill.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:800;background:#152235;border:1px solid {border};border-radius:10px;padding:4px 8px;")
        self.thumb.setStyleSheet(f"background:{COL_BLACK};border:1px solid {border};border-radius:10px;")

    def set_accent(self, accent: str):
        self._accent = accent or UI_BORDER
        self._apply_styles()

    def set_selected(self, selected: bool):
        self._selected = bool(selected)
        self._apply_styles()

    def enterEvent(self, event):
        self._hovered = True
        self._apply_styles()
        return super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._apply_styles()
        return super().leaveEvent(event)

    def set_template_data(self, data: dict):
        self._template_key = str(data.get("template_key") or data.get("template_id") or data.get("id") or "")
        template_id = str(data.get("template_id", data.get("id", "T?")))
        label = str(data.get("label", "N"))
        count = int(data.get("count", 0) or 0)
        self.id_edit.blockSignals(True)
        self.class_combo.blockSignals(True)
        self.view_toggle.blockSignals(True)
        self.id_edit.setText(template_id)
        if label and self.class_combo.findText(label) < 0:
            self.class_combo.addItem(label)
        self.class_combo.setCurrentText(label if self.class_combo.findText(label) >= 0 else (label[:1] if label else "N"))
        self.count_pill.setText(f"{count} beats")
        idx = data.get("index")
        self.template_no.setText(f"#{idx}" if idx is not None else template_id)

        flags = {
            "ambiguous": bool(data.get("ambiguous")),
            "inserted": bool(data.get("inserted")),
            "demix": bool(data.get("demix")),
            "auto_update": bool(data.get("auto_update")),
        }
        any_badge = False
        for key, badge in self.badge_labels.items():
            visible = flags.get(key, False)
            any_badge = any_badge or visible
            badge.setVisible(visible)
            if key == "ambiguous":
                badge.setStyleSheet(f"color:{UI_WARNING};font-size:10px;font-weight:900;background:#221F12;border:1px solid {UI_WARNING};border-radius:8px;padding:2px 6px;")
            elif key == "inserted":
                badge.setStyleSheet(f"color:{UI_TEXT};font-size:10px;font-weight:900;background:#17311F;border:1px solid {UI_SUCCESS};border-radius:8px;padding:2px 6px;")
            elif key == "demix":
                badge.setStyleSheet(f"color:{UI_TEXT};font-size:10px;font-weight:900;background:#122C46;border:1px solid #2D9CDB;border-radius:8px;padding:2px 6px;")
            elif key == "auto_update":
                badge.setStyleSheet(f"color:{UI_TEXT};font-size:10px;font-weight:900;background:#1F2030;border:1px solid {UI_MUTED};border-radius:8px;padding:2px 6px;")
        self.badge_container.setVisible(any_badge)

        self.view_toggle.blockSignals(True)
        self.view_toggle.setChecked(bool(data.get("viewed", True)))
        self.view_toggle.setText("\u25c9" if self.view_toggle.isChecked() else "\u25cb")
        self.view_toggle.blockSignals(False)
        self.id_edit.blockSignals(False)
        self.class_combo.blockSignals(False)

        waveform = data.get("waveform")
        if waveform is not None:
            x = np.linspace(0, len(waveform) / 500.0, len(waveform)) if len(waveform) else []
            self.thumb.set_data(x, waveform)
        else:
            self.thumb.set_data([], [])

class HolterBeatTemplatePanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._template_rows = []
        self._current_filter = "all"
        self._selected_template_key = ""
        self._selected_template_keys = []
        self._card_widgets = []
        self._waveform_cache = {}
        self._replay_engine = None
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
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Template System")
        title.setStyleSheet(f"color:#07111F;font-size:13px;font-weight:bold;background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #28E37B, stop:1 #89F7C5);padding:7px 12px;border-radius:8px;")
        layout.addWidget(title)

        stats = QGridLayout()
        stats.setHorizontalSpacing(8)
        stats.setVerticalSpacing(8)
        self._stat_cards = {
            "total": _TemplateMetricCard("Total beats", "0", COL_TEXT),
            "templates": _TemplateMetricCard("Templates", "0", COL_TEXT),
            "unconfirmed": _TemplateMetricCard("Unconfirmed", "0", UI_WARNING),
            "beat_distribution": _TemplateMetricCard("Beat distribution", "N: 0", UI_TEXT),
        }
        stats.addWidget(self._stat_cards["total"], 0, 0)
        stats.addWidget(self._stat_cards["templates"], 0, 1)
        stats.addWidget(self._stat_cards["unconfirmed"], 0, 2)
        stats.addWidget(self._stat_cards["beat_distribution"], 0, 3)
        layout.addLayout(stats)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        filter_lbl = QLabel("Filter")
        filter_lbl.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:700;border:none;")
        filter_row.addWidget(filter_lbl)
        self._filter_buttons = {}
        for key, text in [
            ("all", "All"),
            ("N", "N"),
            ("S", "S"),
            ("V", "V"),
            ("P", "P"),
            ("AF", "AF"),
            ("X", "X"),
            ("Other", "Other"),
            ("unconfirmed", "Unconfirmed"),
        ]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setChecked(key == "all")
            btn.setToolTip({
                "all": "Show all templates",
                "N": "Show normal templates",
                "S": "Show supraventricular templates",
                "V": "Show ventricular templates",
                "P": "Show paced templates",
                "AF": "Show atrial fibrillation/flutter templates",
                "X": "Show artifact templates",
                "Other": "Show uncategorized templates",
                "unconfirmed": "Show templates not yet confirmed",
            }.get(key, text))
            btn.setFixedHeight(28)
            btn.setStyleSheet(_style_filter_btn(active=(key == "all")))
            btn.clicked.connect(lambda checked=False, k=key: self._set_filter(k))
            self._filter_buttons[key] = btn
            filter_row.addWidget(btn)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setStyleSheet(f"QScrollArea{{background:{COL_BG};border:none;}}")
        self._cards_host = QWidget()
        self._cards_host.setStyleSheet(f"background:{COL_BG};")
        self._cards_layout = QGridLayout(self._cards_host)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(7)
        self._cards_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._scroll.setWidget(self._cards_host)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._scroll)

        self._detail_grid_host = QWidget()
        self._detail_grid_host.setStyleSheet(f"background:{COL_BLACK};border-left:1px solid {COL_GREEN_DRK};")
        self._detail_layout = QGridLayout(self._detail_grid_host)
        self._detail_layout.setContentsMargins(4, 4, 4, 4)
        self._detail_layout.setSpacing(4)
        self._detail_canvases = []
        lead_names_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for i in range(12):
            lead_name = lead_names_12[i]
            frame = QFrame()
            frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};}}")
            flayout = QVBoxLayout(frame)
            flayout.setContentsMargins(2, 2, 2, 2)
            flayout.setSpacing(0)

            info = QWidget()
            info.setFixedHeight(16)
            info_layout = QHBoxLayout(info)
            info_layout.setContentsMargins(2, 0, 2, 0)
            info_layout.setSpacing(4)
            lbl_type = QLabel(lead_name)
            lbl_type.setStyleSheet(f"color:{COL_WHITE};font-weight:bold;font-size:10px;border:none;background:transparent;")
            info_layout.addWidget(lbl_type)
            info_layout.addStretch()
            flayout.addWidget(info)

            canvas = ECGStripCanvas(height=60, color=COL_GREEN, pen_width=1.0, lead_name=lead_name, show_vertical_lines=False, show_annotations=True, show_rr_numbers=False, disable_all_coloring=False)
            canvas.set_paper_speed(25)
            canvas.set_gain(1.5)
            flayout.addWidget(canvas, 1)

            self._detail_layout.addWidget(frame, i // 3, i % 3)
            self._detail_canvases.append({"frame": frame, "canvas": canvas, "type": lbl_type, "lead_name": lead_name})

        self._splitter.addWidget(self._detail_grid_host)
        self._splitter.setSizes([220, 780])
        layout.addWidget(self._splitter, 1)

    def update_from_metrics(self, metrics_list: list, summary: dict):
        class_totals = dict(summary.get('beat_class_totals', {}) or {})
        if not class_totals:
            for metric in metrics_list or []:
                for cls, count in (metric.get('beat_class_counts', {}) or {}).items():
                    class_totals[cls] = class_totals.get(cls, 0) + int(count or 0)
        total_beats = int(summary.get("total_beats", 0) or sum(class_totals.values()) or 0)
        template_map = {}
        for metric in metrics_list or []:
            for row in (metric.get('template_summary', []) or []):
                tkey = str(row.get('template_key') or row.get('template_id') or row.get('label') or 'T')
                item = template_map.setdefault(tkey, {
                    'template_key': tkey,
                    'template_id': row.get('template_id', 'T'),
                    'label': row.get('label', 'N'),
                    'count': 0,
                    'rr': [],
                    'qrs': [],
                    'first_timestamp': float(row.get('first_timestamp', metric.get('t', 0.0)) or 0.0),
                    'viewed': bool(row.get('viewed', True)),
                    'ambiguous': bool(row.get('ambiguous', False)),
                    'inserted': bool(row.get('inserted', False)),
                    'demix': bool(row.get('demix', False)),
                    'auto_update': bool(row.get('auto_update', False)),
                })
                item['count'] += int(row.get('count', 0) or 0)
                item['rr'].append(float(row.get('avg_rr_ms', 0.0) or 0.0))
                item['qrs'].append(float(row.get('avg_qrs_ms', 0.0) or 0.0))
                item['first_timestamp'] = min(item['first_timestamp'], float(row.get('first_timestamp', item['first_timestamp']) or item['first_timestamp']))
        self._template_rows = sorted(template_map.values(), key=lambda x: x['count'], reverse=True)

        unconfirmed = sum(1 for row in self._template_rows if not row.get("viewed", True))
        self._stat_cards["total"].value.setText(f"{total_beats:,}")
        self._stat_cards["templates"].value.setText(str(len(self._template_rows)))
        self._stat_cards["unconfirmed"].value.setText(str(unconfirmed))
        dist_parts = []
        for key in ("N", "V", "S", "F", "Q"):
            if int(class_totals.get(key, 0) or 0) > 0:
                dist_parts.append(f"{key}: {int(class_totals.get(key, 0)):,}")
        self._stat_cards["beat_distribution"].value.setText("  ".join(dist_parts) if dist_parts else "No beat classes")
        self._refresh_stats()
        self._render_cards()

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._waveform_cache.clear()
        self._render_cards()

    def _row_key(self, row: dict) -> str:
        return str(row.get("template_key") or row.get("template_id") or row.get("label") or "")

    def _filtered_rows(self):
        filtered = []
        for row in self._template_rows:
            label = str(row.get("label", "N") or "N").strip() or "N"
            code = _template_filter_key(label)
            
            if self._current_filter == "all":
                filtered.append(row)
            elif self._current_filter == "unconfirmed" and not row.get("viewed", True):
                filtered.append(row)
            elif self._current_filter == "AF" and code in {"AF", "F", "Q"}:
                filtered.append(row)
            elif self._current_filter == "Other" and code == "Other":
                filtered.append(row)
            elif self._current_filter in {"N", "V", "S", "P", "X"} and code == self._current_filter:
                filtered.append(row)
        return filtered

    def _refresh_stats(self):
        total_beats = sum(int(row.get("count", 0) or 0) for row in self._template_rows)
        unconfirmed = sum(1 for row in self._template_rows if not row.get("viewed", True))
        label_totals = {}
        for row in self._template_rows:
            label = str(row.get("label", "N") or "N").strip() or "N"
            label_totals[label] = label_totals.get(label, 0) + int(row.get("count", 0) or 0)
        self._stat_cards["total"].value.setText(f"{int(total_beats):,}")
        self._stat_cards["templates"].value.setText(str(len(self._template_rows)))
        self._stat_cards["unconfirmed"].value.setText(str(unconfirmed))

        preferred = ["N", "S", "V", "P", "AF", "X", "Unconfirmed"]
        dist_parts = []
        for key in preferred:
            value = int(label_totals.pop(key, 0) or 0)
            if value > 0:
                dist_parts.append(f"{key}: {value:,}")
        for key in sorted(label_totals):
            value = int(label_totals.get(key, 0) or 0)
            if value > 0:
                dist_parts.append(f"{key}: {value:,}")
        self._stat_cards["beat_distribution"].value.setText("  ".join(dist_parts) if dist_parts else "No beat classes")

    def _set_selected_keys(self, keys):
        ordered = []
        for key in keys or []:
            value = str(key or "").strip()
            if value and value not in ordered:
                ordered.append(value)
        self._selected_template_keys = ordered
        self._selected_template_key = ordered[0] if ordered else ""
        self._sync_card_selection()

    def _sync_card_selection(self):
        selected = set(self._selected_template_keys)
        visible = []
        for card in self._card_widgets:
            try:
                is_selected = card._template_key in selected
                card.set_selected(is_selected)
                if is_selected:
                    visible.append(card._template_key)
            except Exception:
                pass
        if visible:
            self._selected_template_key = visible[0]
        elif self._selected_template_keys:
            self._selected_template_key = self._selected_template_keys[0]
        else:
            self._selected_template_key = ""
            
        if self._selected_template_key:
            self._update_detail_grid(self._selected_template_key)

    def _row_for_key(self, template_key: str):
        template_key = str(template_key or "")
        for row in self._template_rows:
            if self._row_key(row) == template_key:
                return row
        return None

    def _selected_rows(self):
        selected = set(self._selected_template_keys)
        return [row for row in self._template_rows if self._row_key(row) in selected]

    def _selected_keys_for_card(self, card):
        card_key = str(getattr(card, "_template_key", "") or "")
        if self._selected_template_keys and card_key in set(self._selected_template_keys):
            return list(self._selected_template_keys)
        return [card_key] if card_key else []

    def _rename_template_key(self, old_key: str, new_key: str):
        old_key = str(old_key or "")
        new_key = str(new_key or "")
        if not old_key or not new_key or old_key == new_key:
            return
        for row in self._template_rows:
            if self._row_key(row) == old_key:
                row["template_key"] = new_key
        if old_key in self._waveform_cache:
            self._waveform_cache[new_key] = self._waveform_cache.pop(old_key)
        self._selected_template_keys = [new_key if key == old_key else key for key in self._selected_template_keys]
        if self._selected_template_key == old_key:
            self._selected_template_key = new_key

    def _apply_template_label(self, keys, label: str):
        key_set = set(keys or [])
        changed = False
        for row in self._template_rows:
            if self._row_key(row) in key_set:
                row["label"] = str(label or "N").strip() or "N"
                changed = True
        if changed:
            self._refresh_stats()
            self._render_cards()

    def _apply_template_viewed(self, keys, viewed: bool):
        key_set = set(keys or [])
        changed = False
        for row in self._template_rows:
            if self._row_key(row) in key_set:
                row["viewed"] = bool(viewed)
                changed = True
        if changed:
            self._refresh_stats()
            self._render_cards()

    def _select_all_visible(self):
        self._set_selected_keys([self._row_key(row) for row in self._filtered_rows()])

    def _reverse_visible_selection(self):
        visible = [self._row_key(row) for row in self._filtered_rows()]
        selected = set(self._selected_template_keys)
        new_keys = [key for key in self._selected_template_keys if key not in visible]
        new_keys.extend(key for key in visible if key not in selected)
        self._set_selected_keys(new_keys)
    def _set_filter(self, key: str):
        try:
            self._current_filter = key
            for k, btn in self._filter_buttons.items():
                is_active = (k == key)
                btn.setChecked(is_active)
                btn.setStyleSheet(_style_filter_btn(active=is_active))
            self._refresh_stats()
            self._render_cards()
        except Exception as e:
            print(f"Error in _set_filter: {e}")
            import traceback
            traceback.print_exc()

    def _on_card_template_id_changed(self, card, text: str):
        try:
            old_key = str(getattr(card, "_template_key", "") or "")
            new_id = str(text or "").strip() or old_key
            row = self._row_for_key(old_key)
            if row is None:
                return
            row["template_id"] = new_id
            row["template_key"] = new_id
            if old_key != new_id:
                self._rename_template_key(old_key, new_id)
            self._refresh_stats()
            # Defer card rendering to avoid destroying card during signal
            QTimer.singleShot(10, self._render_cards)
        except Exception as e:
            print(f"Error in _on_card_template_id_changed: {e}")

    def _on_card_class_changed(self, card, text: str):
        try:
            key = str(getattr(card, "_template_key", "") or "")
            row = self._row_for_key(key)
            if row is None:
                return
            row["label"] = str(text or "N").strip() or "N"
            
            if text == "Unconfirmed":
                row["viewed"] = False
            else:
                row["viewed"] = True  # Mark as viewed for other classifications
            
            self._refresh_stats()
            # Defer card rendering to avoid destroying card during signal
            QTimer.singleShot(10, self._render_cards)
        except Exception as e:
            print(f"Error in _on_card_class_changed: {e}")

    def _on_card_viewed_changed(self, card, viewed: bool):
        try:
            key = str(getattr(card, "_template_key", "") or "")
            row = self._row_for_key(key)
            if row is None:
                return
            row["viewed"] = bool(viewed)
            self._refresh_stats()
            # Defer card rendering to avoid destroying card during signal
            QTimer.singleShot(10, self._render_cards)
        except Exception as e:
            print(f"Error in _on_card_viewed_changed: {e}")

    def _delete_selected_templates(self, keys=None):
        keys = list(keys or self._selected_template_keys)
        if not keys:
            return
        if QMessageBox.question(self, "Delete templates", f"Delete {len(keys)} selected template(s)?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        key_set = set(keys)
        self._template_rows = [row for row in self._template_rows if self._row_key(row) not in key_set]
        for key in key_set:
            self._waveform_cache.pop(key, None)
        self._selected_template_keys = []
        self._selected_template_key = ""
        self._refresh_stats()
        self._render_cards()

    def _confirm_delete_templates(self, count: int) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Delete templates")
        box.setText(f"Delete {count} selected template(s)?")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        box.setStyleSheet(f"""
            QMessageBox {{ background: {COL_BLACK}; }}
            QMessageBox QLabel {{ color: {COL_WHITE}; font-size: 12px; }}
            QMessageBox QPushButton {{
                background: #1B2740;
                color: {COL_WHITE};
                border: 1px solid {UI_BORDER};
                border-radius: 6px;
                padding: 6px 14px;
                min-width: 64px;
            }}
            QMessageBox QPushButton:hover {{
                background: #243552;
                border-color: {UI_ACCENT};
            }}
            QMessageBox QPushButton:pressed {{
                background: #132033;
            }}
        """)
        return box.exec_() == QMessageBox.Yes
    def _merge_selected_templates(self, keys=None):
        keys = list(keys or self._selected_template_keys)
        key_set = set(keys)
        rows = [row for row in self._template_rows if self._row_key(row) in key_set]
        
        # Check if only one template is selected
        if len(rows) < 2:
            _show_message_box(
                self,
                QMessageBox.Information,
                "Merge Templates",
                "Only one template selected.\n\n"
                "Ctrl + Click to select manually to merge template\n"
                "Shift + Click to select all to merge template"
            )
            return
        
        options = [
            ("Normal", "N"),
            ("Atrial Premature", "S"),
            ("Ventricular Premature", "V"),
            ("Artifact", "X"),
            ("Atrial Fibrillation", "Q"),
            ("Atrial Flutter", "R"),
            ("Blocked PAC", "O"),
            ("Paced", "P"),
            ("Other", "Other"),
        ]
        labels = []
        for row in rows:
            label = str(row.get("label", "N") or "N").strip() or "N"
            if label not in labels:
                labels.append(label)
        if len(labels) == 1:
            merged_label = labels[0]
        else:
            items = [f"{name} ({code})" if code != "Other" else name for name, code in options]
            choice, ok = QInputDialog.getItem(self, "Merge template", "Choose the merged template type:", items, 0, False)
            if not ok:
                return
            merged_label = choice[choice.rfind("(") + 1:-1] if "(" in choice and choice.endswith(")") else choice
        base = dict(rows[0])
        total_count = sum(int(row.get("count", 0) or 0) for row in rows)
        rr_values = []
        qrs_values = []
        for row in rows:
            rr_values.extend([float(v) for v in (row.get("rr", []) or []) if v is not None])
            qrs_values.extend([float(v) for v in (row.get("qrs", []) or []) if v is not None])
        base["label"] = merged_label
        base["count"] = total_count
        base["rr"] = rr_values
        base["qrs"] = qrs_values
        base["first_timestamp"] = min(float(row.get("first_timestamp", 0.0) or 0.0) for row in rows)
        base["viewed"] = all(bool(row.get("viewed", True)) for row in rows)
        base["ambiguous"] = any(bool(row.get("ambiguous", False)) for row in rows)
        base["inserted"] = True  # Always show "+" badge for merged templates
        base["demix"] = any(bool(row.get("demix", False)) for row in rows)
        base["auto_update"] = any(bool(row.get("auto_update", False)) for row in rows)
        base["_original_templates"] = [dict(row) for row in rows]  # Store original templates for unmerge
        base.setdefault("template_id", base.get("template_id") or f"T{len(self._template_rows) + 1}")
        base["template_key"] = base.get("template_id") or base.get("label") or "T"
        new_rows = []
        inserted = False
        for row in self._template_rows:
            key = self._row_key(row)
            if key in key_set:
                self._waveform_cache.pop(key, None)
                if not inserted:
                    new_rows.append(base)
                    inserted = True
            else:
                new_rows.append(row)
        self._template_rows = new_rows
        self._selected_template_keys = [base["template_key"]]
        self._selected_template_key = base["template_key"]
        self._refresh_stats()
        self._render_cards()

    def _unmerge_template(self, template_key):
        row = self._row_for_key(template_key)
        if row is None or "_original_templates" not in row:
            return
        
        original_templates = row["_original_templates"]
        if not original_templates:
            return
        
        # Replace the merged template with the original templates
        new_rows = []
        found = False
        for r in self._template_rows:
            key = self._row_key(r)
            if key == template_key:
                found = True
                # Add back the original templates, removing the _original_templates field to avoid recursion
                for orig in original_templates:
                    orig_copy = dict(orig)
                    orig_copy.pop("_original_templates", None)
                    new_rows.append(orig_copy)
            else:
                new_rows.append(r)
        
        if found:
            self._template_rows = new_rows
            self._waveform_cache.pop(template_key, None)
            # Select the first original template
            if original_templates:
                first_key = self._row_key(original_templates[0])
                self._selected_template_keys = [first_key]
                self._selected_template_key = first_key
            self._refresh_stats()
            self._render_cards()

    def _open_overlay_analysis(self, card):
        host = self.window()
        if host is None:
            return
        try:
            if hasattr(host, "_focus_tab"):
                host._focus_tab("REPLAY")
            template_key = str(getattr(card, "_template_key", "") or "")
            row = self._row_for_key(template_key)
            if row is not None and hasattr(host, "set_magnifier_focus"):
                host.set_magnifier_focus(card.thumb, card.thumb._magnifier_source_payload(), QPoint(max(8, card.thumb.width() // 2), max(8, card.thumb.height() // 2)))
                self.seek_requested.emit(float(row.get("first_timestamp", 0.0) or 0.0))
        except Exception:
            pass

    def _open_lorenz_plots(self, card):
        host = self.window()
        if host is None:
            return
        try:
            if hasattr(host, "_focus_tab"):
                host._focus_tab("REPLAY")
            row = self._row_for_key(str(getattr(card, "_template_key", "") or ""))
            if row is not None:
                self.seek_requested.emit(float(row.get("first_timestamp", 0.0) or 0.0))
        except Exception:
            pass

    def _show_template_card_menu(self, card, global_pos):
        card_key = str(getattr(card, "_template_key", "") or "")
        if not card_key:
            return
        if not self._selected_template_keys or card_key not in set(self._selected_template_keys):
            self._set_selected_keys([card_key])
        target_keys = list(self._selected_template_keys) or [card_key]
        menu = QMenu(self)
        menu.setStyleSheet(f"QMenu {{ background: #0B1220; color: {UI_TEXT}; border: 1px solid {UI_BORDER}; padding: 6px; }} QMenu::item {{ padding: 6px 20px 6px 18px; border-radius: 4px; }} QMenu::item:selected {{ background: {UI_ACCENT}; color: #07111F; }} QMenu::separator {{ height: 1px; background: {UI_BORDER}; margin: 6px 4px; }}")
        prop_menu = menu.addMenu("Template properties")
        # Updated options: N, S, V, P, AF, X, Other, Unconfirmed
        for title, code in [
            ("Normal (N)", "N"),
            ("Atrial Premature (S)", "S"),
            ("Ventricular Premature (V)", "V"),
            ("Paced (P)", "P"),
            ("Atrial Fibrillation (AF)", "AF"),
            ("Artifact (X)", "X"),
            ("Other", "Other"),
            ("Unconfirmed", "Unconfirmed")
        ]:
            action = prop_menu.addAction(title)
            action.triggered.connect(lambda checked=False, c=code: self._apply_template_label(target_keys, c))
        func_menu = menu.addMenu("Function")
        func_menu.addAction("Delete").triggered.connect(lambda: self._delete_selected_templates(target_keys))
        func_menu.addAction("Select All").triggered.connect(self._select_all_visible)
        func_menu.addAction("Reverse Selection").triggered.connect(self._reverse_visible_selection)
        merge_action = menu.addAction("Merge template")
        merge_action.setEnabled(True)  # Always enable - let merge function handle validation
        merge_action.triggered.connect(lambda: self._merge_selected_templates(target_keys))
        
        # Add Unmerge Template option
        unmerge_action = menu.addAction("Unmerge template")
        # Enable only if we have a single selected template that has original templates stored
        has_original = len(target_keys) == 1
        if has_original:
            row = self._row_for_key(target_keys[0])
            has_original = row is not None and "_original_templates" in row
        unmerge_action.setEnabled(has_original)
        unmerge_action.triggered.connect(lambda: self._unmerge_template(target_keys[0]))
        
        ok_menu = menu.addMenu("Template OK/Cancel")
        ok_menu.addAction("Confirm").triggered.connect(lambda: self._apply_template_viewed(target_keys, True))
        ok_menu.addAction("Unconfirm").triggered.connect(lambda: self._apply_template_viewed(target_keys, False))
        menu.addAction("Overlay beat analysis").triggered.connect(lambda: self._open_overlay_analysis(card))
        menu.addAction("Lorenz plots").triggered.connect(lambda: self._open_lorenz_plots(card))
        menu.exec_(global_pos)
    def _on_card_clicked(self, card, event):
        template_key = str(getattr(card, "_template_key", "") or "")
        if not template_key:
            return
        
        modifiers = event.modifiers() if event else Qt.NoModifier
        ctrl_pressed = bool(modifiers & Qt.ControlModifier)
        shift_pressed = bool(modifiers & Qt.ShiftModifier)
        
        filtered = self._filtered_rows()
        filtered_keys = [self._row_key(row) for row in filtered]
        
        if ctrl_pressed:
            # Toggle selection
            current = list(self._selected_template_keys)
            if template_key in current:
                current.remove(template_key)
            else:
                current.append(template_key)
            self._set_selected_keys(current)
        elif shift_pressed and self._selected_template_keys:
            # Range selection
            last_key = self._selected_template_keys[-1]
            try:
                last_idx = filtered_keys.index(last_key)
                curr_idx = filtered_keys.index(template_key)
                start = min(last_idx, curr_idx)
                end = max(last_idx, curr_idx)
                self._set_selected_keys(filtered_keys[start:end+1])
            except ValueError:
                self._set_selected_keys([template_key])
        else:
            # Single selection
            self._set_selected_keys([template_key])
        
        row = self._row_for_key(template_key)
        if template_key and row and not ctrl_pressed and not shift_pressed:
            self.seek_requested.emit(float(row.get("first_timestamp", 0.0) or 0.0))
            
        self._update_detail_grid(template_key)

    def _on_card_double_clicked(self, card):
        """On double-click, select the card and immediately load 12-lead waveforms."""
        template_key = str(getattr(card, "_template_key", "") or "")
        if not template_key:
            return
        # Select only this card
        self._set_selected_keys([template_key])
        # Scroll the detail grid into view if splitter is collapsed
        if hasattr(self, "_splitter"):
            sizes = self._splitter.sizes()
            if sizes and len(sizes) == 2 and sizes[1] < 100:
                total = sum(sizes)
                self._splitter.setSizes([220, max(100, total - 220)])
        # Load all 12 leads for this template
        self._update_detail_grid(template_key)

    def _get_replay_engine(self):
        if self._replay_engine is not None:
            return self._replay_engine
        win = self.window()
        if win is not None and getattr(win, "_replay_engine", None) is not None:
            self._replay_engine = win._replay_engine
            return self._replay_engine
        parent = self.parentWidget()
        while parent is not None:
            if getattr(parent, "_replay_engine", None) is not None:
                self._replay_engine = parent._replay_engine
                return self._replay_engine
            parent = parent.parentWidget()
        return None

    def _update_detail_grid(self, template_key: str):
        if not hasattr(self, "_detail_canvases"):
            return

        row = self._row_for_key(template_key)
        if row is None and self._template_rows:
            row = self._template_rows[0]

        label = "N"
        first_ts = 0.0
        rr_med   = 0.0
        qrs_med  = 0.0
        if row:
            label     = str(row.get("label", "N") or "N").strip() or "N"
            first_ts  = float(row.get("first_timestamp", 0.0) or 0.0)
            rr_list   = row.get("rr", []) or []
            qrs_list  = row.get("qrs", []) or []
            rr_med    = float(np.median(rr_list))  if rr_list  else 0.0
            qrs_med   = float(np.median(qrs_list)) if qrs_list else 0.0

        engine = self._get_replay_engine()
        data   = None
        n_real = 0
        pre    = 0.60
        post   = 0.90
        t_start = max(0.0, first_ts - pre)

        if engine is not None and hasattr(engine, "_reader") and engine._reader is not None:
            try:
                t_end = min(float(engine.duration_sec), first_ts + post)
                raw_data = engine._reader.read_range(t_start, t_end)
                if (isinstance(raw_data, np.ndarray)
                        and raw_data.ndim == 2
                        and raw_data.shape[0] >= 1
                        and raw_data.shape[1] > 10):
                    data   = raw_data
                    n_real = data.shape[0]
            except Exception as e:
                print(f"[12-lead] read_range error: {e}")

        beat_annotations = [{"timestamp": first_ts, "label": label, "short_code": label, "type": "beat"}]

        for i, item in enumerate(self._detail_canvases):
            canvas = item["canvas"]
            canvas.set_gain(1.5)

            if data is not None and i < n_real:
                lead_raw = np.asarray(data[i], dtype=float)
                n_pts = len(lead_raw)
                x = np.arange(n_pts, dtype=float) / 500.0
                canvas.set_data(x, lead_raw,
                                beat_annotations=beat_annotations,
                                start_sec=t_start)
            else:
                synthetic = self._make_thumbnail_waveform(rr_med, qrs_med)
                lead_name = item.get("lead_name", "")
                if lead_name == "aVR":
                    lead_wave = 2048.0 - (synthetic - 2048.0) * 0.8
                elif lead_name in ("III", "aVL"):
                    lead_wave = 2048.0 + (synthetic - 2048.0) * 0.65
                elif lead_name in ("V1", "V2"):
                    lead_wave = 2048.0 + (synthetic - 2048.0) * 0.85
                elif lead_name in ("V3", "V4", "V5"):
                    lead_wave = 2048.0 + (synthetic - 2048.0) * 1.3
                else:
                    lead_wave = synthetic
                x = np.arange(len(lead_wave), dtype=float) / 500.0
                canvas.set_data(x, lead_wave,
                                beat_annotations=beat_annotations,
                                start_sec=t_start)

            item["type"].setText(f"{item['lead_name']}  {label}")



    def _render_cards(self):
        try:
            # Disconnect all signals from old cards before deleting
            for card in self._card_widgets:
                try:
                    card.clicked.disconnect()
                    card.template_id_changed.disconnect()
                    card.class_changed.disconnect()
                    card.viewed_changed.disconnect()
                except Exception:
                    pass
            
            while self._cards_layout.count():
                item = self._cards_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            
            self._card_widgets = []
            filtered = self._filtered_rows()
            filtered_keys = [self._row_key(row) for row in filtered]
            selected_visible = [key for key in filtered_keys if key in set(self._selected_template_keys)]
            if filtered and not selected_visible:
                self._set_selected_keys([filtered_keys[0]])
            elif selected_visible:
                self._selected_template_key = selected_visible[0]
            elif not self._selected_template_keys:
                self._selected_template_key = ""

            for idx, row in enumerate(filtered):
                label = str(row.get("label", "N") or "N").strip() or "N"
                label_code = label[:1].upper() if label else "N"
                accent = {"N": "#2D9CDB", "V": "#F2994A", "S": "#E2B93B", "F": "#7F8C8D", "Q": "#B06CFD", "R": "#A97FFF", "O": "#F2C94C", "P": "#56CCF2", "X": "#EB5757"}.get(label_code, UI_BORDER)
                card = TemplateCardWidget(accent=accent)
                rr = row.get("rr", []) or []
                qrs = row.get("qrs", []) or []
                rr_med = float(np.median(rr)) if rr else 0.0
                qrs_med = float(np.median(qrs)) if qrs else 0.0
                waveform = self._resolve_template_waveform(row, rr_med, qrs_med)
                template_key = self._row_key(row)
                card.clicked.connect(self._on_card_clicked)
                card.double_clicked.connect(self._on_card_double_clicked)
                card.template_id_changed.connect(self._on_card_template_id_changed)
                card.class_changed.connect(self._on_card_class_changed)
                card.viewed_changed.connect(self._on_card_viewed_changed)
                card.set_template_data({
                    "template_key": template_key,
                    "index": idx + 1,
                    "template_id": row.get("template_id", f"T{idx+1}"),
                    "label": row.get("label", "N"),
                    "count": row.get("count", 0),
                    "viewed": row.get("viewed", True),
                    "ambiguous": row.get("ambiguous", False),
                    "inserted": row.get("inserted", False),
                    "demix": row.get("demix", False),
                    "auto_update": row.get("auto_update", False),
                    "waveform": waveform,
                })
                card.set_selected(template_key in set(self._selected_template_keys))
                row_idx = idx
                col_idx = 0
                self._cards_layout.addWidget(card, row_idx, col_idx)
                self._card_widgets.append(card)
            
            self._sync_card_selection()
        except Exception as e:
            import traceback
            print(f"Error in _render_cards: {e}")
            traceback.print_exc()

    def _resolve_template_waveform(self, row: dict, rr_ms: float, qrs_ms: float):
        try:
            key = str(row.get("template_key") or row.get("template_id") or row.get("label") or "T")
            cached = self._waveform_cache.get(key)
            if cached is not None:
                return cached
            first_ts = float(row.get("first_timestamp", 0.0) or 0.0)
            waveform = None
            engine = getattr(self, "_replay_engine", None)
            try:
                if engine is not None and hasattr(engine, "_reader") and engine._reader is not None:
                    fs = float(getattr(engine, "fs", 500.0) or 500.0)
                    pre = 0.35
                    post = 0.55
                    data = engine._reader.read_range(max(0.0, first_ts - pre), min(float(engine.duration_sec), first_ts + post))
                    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] > 1 and data.shape[1] > 8:
                        lead = np.asarray(data[1], dtype=float)
                        baseline = float(np.median(lead))
                        centered = lead - baseline
                        if np.ptp(centered) > 1.0:
                            x_old = np.linspace(0.0, 1.0, centered.size)
                            x_new = np.linspace(0.0, 1.0, 300)
                            centered = np.interp(x_new, x_old, centered)
                            centered = centered - float(np.median(centered))
                            peak = max(float(np.max(np.abs(centered))), 1.0)
                            centered = np.clip(centered / peak * 1400.0, -1800.0, 1800.0)
                            waveform = 2048.0 + centered
            except Exception:
                waveform = None
            if waveform is None:
                waveform = self._make_thumbnail_waveform(rr_ms, qrs_ms)
            self._waveform_cache[key] = waveform
            return waveform
        except Exception as e:
            print(f"Error in _resolve_template_waveform: {e}")
            return self._make_thumbnail_waveform(rr_ms, qrs_ms)

    def _make_thumbnail_waveform(self, rr_ms: float, qrs_ms: float):
        t = np.linspace(0, 1.2, 300)
        rr_scale = np.clip(rr_ms / 900.0, 0.55, 1.8) if rr_ms > 0 else 1.0
        qrs_scale = np.clip(qrs_ms / 80.0, 0.6, 1.8) if qrs_ms > 0 else 1.0
        p = 0.04 * np.exp(-((t - 0.14) / 0.028) ** 2)
        q = -0.14 * np.exp(-((t - 0.275) / 0.010) ** 2)
        r = 1.35 * np.exp(-((t - 0.31) / (0.014 / qrs_scale)) ** 2)
        s = -0.36 * np.exp(-((t - 0.335) / 0.014) ** 2)
        tw = 0.22 * np.exp(-((t - 0.63) / (0.09 * rr_scale)) ** 2)
        st = 0.025 * np.exp(-((t - 0.45) / 0.05) ** 2)
        wave = p + q + r + s + tw + st
        return 2048.0 + wave * 1200.0

    def _on_template_clicked(self, row, _col):
        if 0 <= row < len(self._template_rows):
            self.seek_requested.emit(float(self._template_rows[row].get("first_timestamp", 0.0)))


