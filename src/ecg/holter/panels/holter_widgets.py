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

from ..tool_engine import canonical_tool
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
)


# 6. HOLTER RR / HR TREND CANVAS
class HolterRRTrendCanvas(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, parent=None, title: str = ""):
        super().__init__(parent)
        self._title = str(title or "Trend")
        self._points = []
        self._values = []
        self._current_index = -1
        self._start_epoch = None
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, values):
        self.set_points([(idx, float(val)) for idx, val in enumerate(values or [])])

    def set_points(self, points, start_epoch=None):
        parsed = []
        for idx, item in enumerate(points or []):
            x = idx
            y = 0.0
            if isinstance(item, dict):
                x = item.get("x", item.get("t", idx))
                y = item.get("y", item.get("rr", item.get("hr", item.get("value", 0.0))))
            else:
                try:
                    if len(item) >= 2:
                        x, y = item[0], item[1]
                    elif len(item) == 1:
                        x, y = idx, item[0]
                except Exception:
                    x, y = idx, item
            try:
                parsed.append((float(x), float(y)))
            except Exception:
                continue
        self._points = parsed
        self._values = [p[1] for p in parsed]
        self._start_epoch = start_epoch
        self.update()

    def set_current_index(self, index: int):
        try:
            self._current_index = int(index)
        except Exception:
            self._current_index = -1
        self.update()
    def set_mode(self, mode):
        self._mode = str(mode or "").upper()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._points:
            idx = self._nearest_index(event.pos().x())
            if idx is not None:
                self.clicked.emit(idx)
        super().mousePressEvent(event)

    def _nearest_index(self, x_pos: int):
        if not self._points:
            return None
        rect = self.rect().adjusted(42, 18, -12, 30)
        if rect.width() <= 0:
            return 0
        xs = [p[0] for p in self._points]
        lo = 0.0
        hi = 12.0 * 60.0 * 60.0
        span = max(hi - lo, 1e-9)
        best = None
        best_dist = None
        for idx, (x_val, _) in enumerate(self._points):
            xp = rect.left() + int(((x_val - lo) / span) * rect.width())
            dist = abs(xp - x_pos)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = idx
        return best

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Keep the trend comparable across recordings and modes. The replay
        # chart uses seconds on X, milliseconds (RR) or BPM (HR) on Y.
        rect = self.rect().adjusted(42, 18, -12, 30)
        painter.setPen(QPen(QColor(COL_GREEN_DRK), 1))
        painter.drawRect(rect)

        painter.setPen(QPen(QColor(COL_GREEN), 1))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        title = self._title
        mode = str(getattr(self, "_mode", "") or "").upper()
        if mode == "RR" and "Heart Rate" in title:
            title = title.replace("Heart Rate", "RR Interval")
        elif mode == "HR" and "RR Interval" in title:
            title = title.replace("RR Interval", "Heart Rate")
        painter.drawText(12, 12, title)

        if not self._points:
            painter.setPen(QPen(QColor(UI_MUTED)))
            painter.drawText(rect.center(), "No trend data")
            painter.end()
            return

        x_lo = 0.0
        x_hi = 12.0 * 60.0 * 60.0
        y_lo = 0.0
        y_hi = 300.0 if mode == "HR" else 3000.0

        def map_xy(x_val, y_val):
            px = rect.left() + ((x_val - x_lo) / (x_hi - x_lo)) * rect.width()
            py = rect.bottom() - ((y_val - y_lo) / (y_hi - y_lo)) * rect.height()
            return int(px), int(py)

        # Draw fixed-axis tick marks so the chart's scale is visible even when
        # the recording contains only a short or low-variation segment.
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QPen(QColor(UI_MUTED), 1))
        y_ticks = np.arange(y_lo, y_hi + 75.0, 75.0) if mode == "HR" else (0.0, 1500.0, 3000.0)
        for value in y_ticks:
            _, py = map_xy(x_lo, value)
            painter.drawLine(rect.left() - 4, py, rect.left(), py)
            label = f"{int(value)}"
            painter.drawText(4, py + 4, label)

        x_ticks = [(0.0, "0h"), (3.0 * 3600.0, "3h"),
                   (6.0 * 3600.0, "6h"), (9.0 * 3600.0, "9h"),
                   (12.0 * 3600.0, "12h")]
        for value, label in x_ticks:
            px, _ = map_xy(value, y_lo)
            painter.drawLine(px, rect.bottom(), px, rect.bottom() + 4)
            painter.drawText(px - 10, rect.bottom() + 17, label)

        painter.setPen(QPen(QColor(COL_GREEN_MID), 2))
        for i in range(1, len(self._points)):
            x1, y1 = map_xy(*self._points[i - 1])
            x2, y2 = map_xy(*self._points[i])
            painter.drawLine(x1, y1, x2, y2)

        for idx, (x_val, y_val) in enumerate(self._points):
            px, py = map_xy(x_val, y_val)
            if idx == self._current_index:
                painter.setBrush(QBrush(QColor(COL_RED)))
                painter.setPen(QPen(QColor(COL_RED), 2))
                painter.drawEllipse(px - 4, py - 4, 8, 8)
            else:
                painter.setBrush(QBrush(QColor(COL_GREEN)))
                painter.setPen(QPen(QColor(COL_GREEN), 1))
                painter.drawEllipse(px - 2, py - 2, 4, 4)

        painter.setPen(QPen(QColor(UI_MUTED)))
        painter.drawText(rect.right() - 38, rect.bottom() + 17, "Time")


HRTrendCanvas = HolterRRTrendCanvas

class ClickableSummaryTile(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, tile_key: str, parent=None):
        super().__init__(parent)
        self._tile_key = str(tile_key or "")
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._tile_key)
        super().mousePressEvent(event)

class TendencyCanvas(HolterRRTrendCanvas):
    def __init__(self, parent=None, height: int = 70):
        super().__init__(parent, title="Tendency")
        self.setFixedHeight(height)
        self._threshold = None
        self._tendency_mode = "ST"

    def set_threshold(self, value):
        self._threshold = value
        self.update()

    def set_tendency_mode(self, mode):
        self._tendency_mode = str(mode or "ST").upper()
        self._title = "ST" if self._tendency_mode == "ST" else "T"
        self.update()

    def get_clicked_index(self):
        return getattr(self, "_current_index", -1)

# 6a. LORENZ CANVAS
class LorenzCanvas(QWidget):
    """Full-featured RR scatter / Poincar  plot with lasso selection, zoom,
    time-sharing view, DeltaRR mode, and right-click context menu."""

    beats_selected      = pyqtSignal(list)   # list of selected data indices
    beats_reclassified  = pyqtSignal(list, str)  # (indices, new_class)
    beats_deleted       = pyqtSignal(list)   # indices to remove

    # ------------------------------------------------------------------ init
    def __init__(self, parent=None):
        super().__init__(parent)
        # data
        self._x = []
        self._y = []
        self._beat_classes = []
        self._all_rr_points = []   # (t, rr, beat_class) for timesharing
        self._x_range = None
        self._y_range = None

        # display state
        self._view_mode    = "lorenz"     # "lorenz" | "delta_rr"
        self._display_mode = "complete"   # "complete" | "timesharing"
        self._zoom_level   = 1.0
        self._pixel_size   = 3            # dot radius in px
        self._rr_range     = (200, 2000)  # (min_ms, max_ms)

        # selection state
        self._selected_indices = set()
        self._is_dragging      = False
        self._drag_start       = None
        self._drag_current     = None

        self.setMinimumSize(200, 180)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

    # ---------------------------------------------------------------- data
    def set_data(self, x, y, x_range=None, y_range=None, beat_classes=None, rr_points=None):
        self._x = list(x)
        self._y = list(y)
        self._beat_classes = list(beat_classes) if beat_classes is not None else []
        self._x_range = x_range
        self._y_range = y_range
        if rr_points is not None:
            self._all_rr_points = list(rr_points)
        self._selected_indices.clear()
        self.update()

    # ------------------------------------------------------- mouse events
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging  = True
            self._drag_start   = event.pos()
            self._drag_current = event.pos()
            self._selected_indices.clear()
            self.update()

    def mouseMoveEvent(self, event):
        if self._is_dragging:
            self._drag_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_dragging:
            self._is_dragging = False
            self._finalize_lasso_selection()
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selected_indices = set(range(len(self._x)))
            self.beats_selected.emit(list(self._selected_indices))
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_level = max(0.2, self._zoom_level * 0.85)
        else:
            self._zoom_level = min(5.0, self._zoom_level * 1.15)
        self.update()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background: {COL_DARK}; color: white; border: 1px solid {COL_GREEN_DRK}; }}
            QMenu::item:selected {{ background: {COL_GREEN_DRK}; }}
            QMenu::separator {{ height: 1px; background: {COL_GREEN_DRK}; }}
        """)

        #    View submenu   
        view_menu = menu.addMenu("   View")
        act_lorenz   = view_menu.addAction("RR Interval Scatter (Lorenz)")
        act_delta_rr = view_menu.addAction("DeltaRR Difference Scatter")
        act_lorenz.setCheckable(True);   act_lorenz.setChecked(self._view_mode == "lorenz")
        act_delta_rr.setCheckable(True); act_delta_rr.setChecked(self._view_mode == "delta_rr")
        act_lorenz.triggered.connect(lambda: self._set_view("lorenz"))
        act_delta_rr.triggered.connect(lambda: self._set_view("delta_rr"))

        #    Display submenu   
        disp_menu = menu.addMenu("   Display")
        act_complete = disp_menu.addAction("Complete Scatter")
        act_timeshare = disp_menu.addAction("Time-sharing Scatter (16 panels)")
        act_complete.setCheckable(True);   act_complete.setChecked(self._display_mode == "complete")
        act_timeshare.setCheckable(True);  act_timeshare.setChecked(self._display_mode == "timesharing")
        act_complete.triggered.connect(lambda: self._set_display("complete"))
        act_timeshare.triggered.connect(lambda: self._set_display("timesharing"))

        menu.addSeparator()

        #    Zoom   
        zoom_menu = menu.addMenu("   Zoom")
        zoom_menu.addAction("Zoom In  (+)").triggered.connect(lambda: self._do_zoom(True))
        zoom_menu.addAction("Zoom Out (-)").triggered.connect(lambda: self._do_zoom(False))
        zoom_menu.addAction("Reset Zoom").triggered.connect(lambda: self._reset_zoom())

        #    Pixel size   
        pix_menu = menu.addMenu("   Pixel Size")
        pix_menu.addAction("Small  (2 px)").triggered.connect(lambda: self._set_pixel(2))
        pix_menu.addAction("Medium (3 px)").triggered.connect(lambda: self._set_pixel(3))
        pix_menu.addAction("Large  (5 px)").triggered.connect(lambda: self._set_pixel(5))

        #    RR Display Range   
        rng_menu = menu.addMenu("   RR Display Range")
        rng_menu.addAction("Auto (200--2000 ms)").triggered.connect(lambda: self._set_rr_range(200, 2000))
        rng_menu.addAction("Normal (400--1200 ms)").triggered.connect(lambda: self._set_rr_range(400, 1200))
        rng_menu.addAction("Wide (100--3000 ms)").triggered.connect(lambda: self._set_rr_range(100, 3000))
        rng_menu.addAction("Custom...").triggered.connect(self._custom_rr_range_dialog)

        menu.addSeparator()

        #    Beat Attribute   
        has_sel = len(self._selected_indices) > 0
        attr_menu = menu.addMenu("   Beat Attribute")
        attr_menu.setEnabled(has_sel)
        for cls_label, cls_code in [("Normal (N)", "N"), ("Supraventricular (S)", "S"),
                                    ("Ventricular (V)", "V"), ("Paced (P)", "P"),
                                    ("AF / AFl", "AF"), ("Other", "Other"), ("Artifact (X)", "X")]:
            a = attr_menu.addAction(cls_label)
            a.triggered.connect(lambda checked=False, c=cls_code: self._reclassify(c))

        #    Delete   
        del_act = menu.addAction("   Delete Selected")
        del_act.setEnabled(has_sel)
        del_act.triggered.connect(self._delete_selected)

        menu.exec_(event.globalPos())

    # -------------------------------------------------------- private actions
    def _set_view(self, mode):
        self._view_mode = mode
        self._selected_indices.clear()
        self.update()

    def _set_display(self, mode):
        self._display_mode = mode
        self._selected_indices.clear()
        self.update()

    def _do_zoom(self, zoom_in):
        if zoom_in:
            self._zoom_level = max(0.2, self._zoom_level * 0.85)
        else:
            self._zoom_level = min(5.0, self._zoom_level * 1.15)
        self.update()

    def _reset_zoom(self):
        self._zoom_level = 1.0
        self.update()

    def _set_pixel(self, size):
        self._pixel_size = size
        self.update()

    def _set_rr_range(self, lo, hi):
        self._rr_range = (lo, hi)
        self._zoom_level = 1.0
        self.update()

    def _custom_rr_range_dialog(self):
        from PyQt5.QtWidgets import QDialog, QFormLayout, QSpinBox, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Custom RR Display Range")
        dlg.setStyleSheet(f"background:{COL_DARK}; color:white;")
        form = QFormLayout(dlg)
        lo_spin = QSpinBox(); lo_spin.setRange(50, 3000); lo_spin.setValue(self._rr_range[0])
        hi_spin = QSpinBox(); hi_spin.setRange(100, 5000); hi_spin.setValue(self._rr_range[1])
        form.addRow("Min (ms):", lo_spin)
        form.addRow("Max (ms):", hi_spin)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec_() == QDialog.Accepted:
            self._set_rr_range(lo_spin.value(), hi_spin.value())

    def _reclassify(self, new_class):
        indices = list(self._selected_indices)
        for idx in indices:
            if idx < len(self._beat_classes):
                self._beat_classes[idx] = new_class
        self.beats_reclassified.emit(indices, new_class)
        self.update()

    def _delete_selected(self):
        indices = sorted(self._selected_indices, reverse=True)
        for idx in indices:
            if idx < len(self._x):
                self._x.pop(idx)
                self._y.pop(idx)
            if idx < len(self._beat_classes):
                self._beat_classes.pop(idx)
        self.beats_deleted.emit(list(self._selected_indices))
        self._selected_indices.clear()
        self.update()

    def _finalize_lasso_selection(self):
        if self._drag_start is None or self._drag_current is None:
            return
        # Lasso = ellipse defined by drag rect
        x0, y0 = self._drag_start.x(), self._drag_start.y()
        x1, y1 = self._drag_current.x(), self._drag_current.y()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = max(5, abs(x1 - x0) / 2.0)
        ry = max(5, abs(y1 - y0) / 2.0)
        new_sel = set()
        plot_coords = self._compute_plot_coords()
        for idx, (px, py) in enumerate(plot_coords):
            # Inside ellipse?
            if ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1.0:
                new_sel.add(idx)
        self._selected_indices = new_sel
        self.beats_selected.emit(list(new_sel))

    def _compute_plot_coords(self):
        """Return list of (px, py) pixel positions for current data points."""
        w, h = self.width(), self.height()
        left, top, right, bottom = 40, 16, 16, 28
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)
        x_min, x_max, y_min, y_max = self._get_axis_range()
        rng_x = max(x_max - x_min, 1.0)
        rng_y = max(y_max - y_min, 1.0)
        xs, ys = self._get_plot_xy()
        coords = []
        for xv, yv in zip(xs, ys):
            px = int(left + (xv - x_min) / rng_x * plot_w)
            py = int(top + plot_h - (yv - y_min) / rng_y * plot_h)
            coords.append((px, py))
        return coords

    def _get_plot_xy(self):
        """Return (xs, ys) depending on view mode and rr_range filter."""
        rr_lo, rr_hi = self._rr_range
        if self._view_mode == "delta_rr":
            xs, ys, classes = [], [], []
            raw = self._x  # rr_n values
            for i in range(len(raw) - 1):
                rr_n   = raw[i]
                rr_np1 = raw[i+1]
                dv = rr_n - rr_np1
                xs.append(rr_n)
                ys.append(dv)
            return xs, ys
        else:
            filtered_x = [v for v in self._x if rr_lo <= v <= rr_hi]
            filtered_y = [v for v in self._y if rr_lo <= v <= rr_hi]
            n = min(len(filtered_x), len(filtered_y))
            return filtered_x[:n], filtered_y[:n]

    def _get_axis_range(self):
        """Return (x_min, x_max, y_min, y_max) with zoom applied."""
        rr_lo, rr_hi = self._rr_range
        if self._view_mode == "delta_rr":
            xs, ys = self._get_plot_xy()
            if not xs:
                return 200, 1500, -500, 500
            all_x = xs; all_y = ys
            x_center = float(np.median(all_x)) if all_x else 800
            y_center = float(np.median(all_y)) if all_y else 0
            x_half = max(500, float(np.percentile(np.abs(all_x), 95))) * self._zoom_level
            y_half = max(200, float(np.percentile(np.abs(all_y), 95))) * self._zoom_level
            return x_center - x_half, x_center + x_half, y_center - y_half, y_center + y_half

        # Standard Lorenz
        if self._x_range is not None:
            base_lo, base_hi = self._x_range
        else:
            all_vals = [v for v in self._x + self._y if rr_lo <= v <= rr_hi]
            if not all_vals:
                return rr_lo, rr_hi, rr_lo, rr_hi
            lo = float(np.percentile(all_vals, 5))
            hi = float(np.percentile(all_vals, 95))
            if hi - lo < 250:
                c = float(np.median(all_vals))
                lo, hi = c - 500.0, c + 500.0
            base_lo = max(0.0, lo - 50.0)
            base_hi = hi + 50.0

        center = (base_lo + base_hi) / 2.0
        half   = (base_hi - base_lo) / 2.0 * self._zoom_level
        lo = max(0.0, center - half)
        hi = center + half
        return lo, hi, lo, hi

    # ------------------------------------------------------------ paint
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(COL_BLACK))

        if not self._x or not self._y:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No RR data")
            return

        if self._display_mode == "timesharing":
            self._paint_timesharing(painter)
        else:
            self._paint_complete(painter)

        # Draw lasso drag rubber-band
        if self._is_dragging and self._drag_start and self._drag_current:
            x0, y0 = self._drag_start.x(), self._drag_start.y()
            x1, y1 = self._drag_current.x(), self._drag_current.y()
            pen = QPen(QColor(0, 220, 255, 200), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 180, 255, 30)))
            lx, ly = min(x0, x1), min(y0, y1)
            lw, lh = abs(x1-x0), abs(y1-y0)
            painter.drawEllipse(lx, ly, max(4, lw), max(4, lh))

    def _paint_complete(self, painter):
        w, h = self.width(), self.height()
        left, top, right, bottom = 44, 16, 16, 28
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)
        x_min, x_max, y_min, y_max = self._get_axis_range()
        rng_x = max(x_max - x_min, 1.0)
        rng_y = max(y_max - y_min, 1.0)

        def to_px(vx, vy):
            px = int(left + (vx - x_min) / rng_x * plot_w)
            py = int(top + plot_h - (vy - y_min) / rng_y * plot_h)
            return px, py

        # Border
        painter.setPen(QPen(QColor("#2C3E50"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(left, top, plot_w, plot_h)

        # Identity diagonal
        painter.setPen(QPen(QColor("#4A5D6E"), 1, Qt.SolidLine))
        if x_max > x_min:
            s = to_px(x_min, x_min); e = to_px(x_max, x_max)
            painter.drawLine(s[0], s[1], e[0], e[1])

        # Perpendicular grid lines
        range_span = x_max - x_min
        step = 1000.0 if range_span > 2000 else (500.0 if range_span > 600 else 250.0)
        grid_pen = QPen(QColor("#2C3E50"), 1, Qt.SolidLine)
        painter.setPen(grid_pen)
        min_sum = x_min + y_min; max_sum = x_max + y_max
        sum_val = int(np.floor(min_sum / step) * step)
        while sum_val <= max_sum:
            if sum_val > 0:
                pts = []
                yl = sum_val - x_min
                if y_min <= yl <= y_max: pts.append((x_min, yl))
                yr = sum_val - x_max
                if y_min <= yr <= y_max: pts.append((x_max, yr))
                xb = sum_val - y_min
                if x_min <= xb <= x_max: pts.append((xb, y_min))
                xt = sum_val - y_max
                if x_min <= xt <= x_max: pts.append((xt, y_max))
                uniq = []
                for p in pts:
                    if not any(np.allclose(p, up) for up in uniq): uniq.append(p)
                if len(uniq) == 2:
                    p1 = to_px(uniq[0][0], uniq[0][1]); p2 = to_px(uniq[1][0], uniq[1][1])
                    painter.drawLine(p1[0], p1[1], p2[0], p2[1])
            sum_val += step

        # Beat colour map
        color_map = {
            "N":     QColor(0, 210, 21, 255),
            "S":     QColor(255, 51, 51, 255),
            "V":     QColor(255, 153, 0, 255),
            "P":     QColor(0, 229, 255, 255),
            "AF":    QColor(255, 215, 0, 255),
            "X":     QColor(136, 136, 136, 255),
            "Other": QColor(0, 210, 21, 255),
        }
        xs, ys = self._get_plot_xy()
        r = self._pixel_size
        for i, (xv, yv) in enumerate(zip(xs, ys)):
            beat_class = self._beat_classes[i] if i < len(self._beat_classes) else "Other"
            color = color_map.get(beat_class, color_map["Other"])
            px, py = to_px(xv, yv)
            if i in self._selected_indices:
                painter.setPen(QPen(QColor(255, 255, 0, 220), 1))
                painter.setBrush(QBrush(color.lighter(160)))
                painter.drawEllipse(px - r - 2, py - r - 2, (r + 2) * 2, (r + 2) * 2)
            else:
                painter.setPen(QPen(color))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(px - r, py - r, r * 2, r * 2)

        # Axis ticks + labels
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.setFont(QFont("Arial", 8))
        start_tick = int(np.floor(x_min / step) * step)
        end_tick   = int(np.ceil(x_max / step) * step)
        tick = start_tick
        while tick <= end_tick:
            if x_min <= tick <= x_max:
                px = int(left + (tick - x_min) / rng_x * plot_w)
                painter.drawLine(px, top + plot_h, px, top + plot_h + 4)
                painter.drawText(QRect(px - 22, top + plot_h + 5, 44, 14), Qt.AlignCenter, str(int(tick)))
            if y_min <= tick <= y_max:
                py = int(top + plot_h - (tick - y_min) / rng_y * plot_h)
                painter.drawLine(left - 4, py, left, py)
                painter.drawText(QRect(left - 42, py - 7, 38, 14), Qt.AlignRight | Qt.AlignVCenter, str(int(tick)))
            tick += step

        # Axis titles
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        if self._view_mode == "delta_rr":
            painter.drawText(left + 4, h - 8, "RR(n) ms")
            painter.drawText(w - 60, 14, "DeltaRR(n+1)")
        else:
            painter.drawText(left + 4, h - 8, "RR(n) ms")
            painter.drawText(w - 62, 14, "RR(n+1)")

        # Mode badge
        badge = "DeltaRR" if self._view_mode == "delta_rr" else "Lorenz"
        painter.setFont(QFont("Arial", 7))
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.drawText(left + 4, top + 12, badge)

    def _paint_timesharing(self, painter):
        """Draw 16 mini scatter plots in a 4x4 grid."""
        w, h = self.width(), self.height()
        n_cols, n_rows = 4, 4
        cell_w = w // n_cols
        cell_h = h // n_rows

        # Build 16 time segments from _all_rr_points if available, else use _x/_y
        if self._all_rr_points:
            pts = self._all_rr_points
        else:
            pts = [(0, xv, cls) for xv, cls in zip(self._x,
                   self._beat_classes if self._beat_classes else ["N"] * len(self._x))]

        if not pts:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No RR data")
            return

        n_segs = 16
        seg_size = max(1, len(pts) // n_segs)
        segments = [pts[i * seg_size:(i + 1) * seg_size] for i in range(n_segs)]
        # remaining beats go to last segment
        if len(pts) > n_segs * seg_size:
            segments[-1] = segments[-1] + pts[n_segs * seg_size:]

        # Global range from all data
        all_rr = [p[1] for p in pts if self._rr_range[0] <= p[1] <= self._rr_range[1]]
        if not all_rr:
            return
        g_lo = max(0.0, float(np.percentile(all_rr, 5)) - 50.0)
        g_hi = float(np.percentile(all_rr, 95)) + 50.0
        if g_hi - g_lo < 250:
            c = float(np.median(all_rr))
            g_lo, g_hi = max(0, c - 400), c + 400
        g_rng = max(g_hi - g_lo, 1.0)

        color_map = {
            "N": QColor(0, 210, 21, 220), "S": QColor(255, 51, 51, 220),
            "V": QColor(255, 153, 0, 220), "P": QColor(0, 229, 255, 220),
            "AF": QColor(255, 215, 0, 220), "X": QColor(136, 136, 136, 220),
            "Other": QColor(0, 210, 21, 220),
        }

        for seg_idx, seg in enumerate(segments):
            col = seg_idx % n_cols
            row = seg_idx // n_cols
            ox = col * cell_w
            oy = row * cell_h
            pad = 4

            # Cell background + border
            painter.setPen(QPen(QColor("#2C3E50"), 1))
            painter.setBrush(QBrush(QColor("#080C12")))
            painter.drawRect(ox, oy, cell_w, cell_h)

            # Segment number
            painter.setFont(QFont("Arial", 7))
            painter.setPen(QPen(QColor("#4A5D6E")))
            painter.drawText(ox + 3, oy + 10, str(seg_idx + 1))

            # Build local pairs
            local_rr = [p[1] for p in seg if self._rr_range[0] <= p[1] <= self._rr_range[1]]
            local_cls = [p[2] if len(p) > 2 else "N" for p in seg
                         if self._rr_range[0] <= p[1] <= self._rr_range[1]]
            if len(local_rr) < 2:
                continue

            pw = cell_w - 2 * pad
            ph = cell_h - 2 * pad - 4

            def to_mini(vx, vy):
                mx = int(ox + pad + (vx - g_lo) / g_rng * pw)
                my = int(oy + pad + ph - (vy - g_lo) / g_rng * ph)
                return mx, my

            # Identity diagonal
            painter.setPen(QPen(QColor("#2C3E50"), 1))
            s = to_mini(g_lo, g_lo); e = to_mini(g_hi, g_hi)
            painter.drawLine(s[0], s[1], e[0], e[1])

            # Dots
            for i in range(len(local_rr) - 1):
                xv, yv = local_rr[i], local_rr[i+1]
                cls = local_cls[i] if i < len(local_cls) else "N"
                color = color_map.get(cls, color_map["Other"])
                px, py = to_mini(xv, yv)
                painter.setPen(QPen(color))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(px - 1, py - 1, 2, 2)

        # Mode badge
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.drawText(2, h - 4, "Time-sharing  (16 panels)")

# 6b. ECG STRIP CANVAS
class ECGStripCanvas(QWidget):
    """Simple ECG strip renderer with interactive measurement tools."""
    def __init__(self, parent=None, height: int = 80, color: str = "#00FF00", pen_width: float = 0.7, lead_name: str = "", show_vertical_lines: bool = True, show_annotations: bool = True, disable_all_coloring: bool = False, show_rr_numbers: bool = True):
        super().__init__(parent)
        self._data = np.zeros(200)
        self._color = color
        self._pen_width = pen_width
        self.lead_name = lead_name
        self._start_sec = 0.0
        self._show_vertical_lines = show_vertical_lines  # Control whether to show R-peak vertical lines
        self._show_annotations = show_annotations  # Control whether to show beat badge labels
        self._show_rr_numbers = show_rr_numbers  # Control whether to show RR interval numbers (e.g. 476, 632)
        self._disable_all_coloring = disable_all_coloring  # CRITICAL: Disable ALL coloring (for replay/overview)
        self._gain = 1.0
        self._speed = 25
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        self._mode = TOOL_SELECT
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._magnify_locked = False
        self._caliper_line1 = None
        self._caliper_line2 = None
        # Ruler/Measuring ruler state: track start and end points separately
        self._ruler_start = None  # Start point (persists)
        self._ruler_end = None    # End point (current measurement)
        self._magnify_pos = None
        self._fs = 500.0
        self._filter_ba = None
        
        # Click-based vertical line tracking: only show line where user clicked
        self._clicked_beat_timestamp = None  # Timestamp of the clicked peak
        self._clicked_beat_label = None  # Label (N, S, V, etc.) of clicked peak
        self._clicked_beat_x_pos = None  # X pixel position of clicked beat
        
        # Drag selection: multiple selected beats
        self._selected_beats = []  # List of timestamps for all selected beats

    def _find_magnifier_host(self):
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, "set_magnifier_focus") and hasattr(parent, "clear_magnifier_focus"):
                return parent
            parent = parent.parentWidget()
        return None

    def _get_reader_start_time(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "_engine"):
                if hasattr(parent._engine, "_reader") and hasattr(parent._engine._reader, "start_time"):
                    return parent._engine._reader.start_time
            parent = parent.parent()
        return None

    def _magnifier_source_payload(self):
        return {
            "data": np.asarray(self._data, dtype=float).copy(),
            "speed": float(self._speed),
            "gain": float(self._gain),
            "lead_name": getattr(self, "lead_name", ""),
            "fs": float(self._fs),
        }

    def set_gain(self, gain: float):
        self._gain = gain
        self.update()

    def set_paper_speed(self, speed: int):
        self._speed = speed
        self.update()

    def set_mode(self, mode: str):
        host = self._find_magnifier_host()
        if host is not None and self._mode == TOOL_MAGNIFY and canonical_tool(mode) != TOOL_MAGNIFY:
            host.clear_magnifier_focus(self)
        self._mode = canonical_tool(mode)
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._magnify_locked = False
        self._magnify_pos = None
        self._caliper_line1 = None
        self._caliper_line2 = None
        self._ruler_start = None
        self._ruler_end = None
        self.update()

    def clear_interaction(self):
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._magnify_locked = False
        self._magnify_pos = None
        self._caliper_line1 = None
        self._caliper_line2 = None
        self._ruler_start = None
        self._ruler_end = None
        # Clear clicked beat
        self._clicked_beat_timestamp = None
        self._clicked_beat_label = None
        self._clicked_beat_x_pos = None
        self.update()

    def set_data(self, *args, beat_annotations=None, start_sec=0.0, structured_events=None, fast_preview=False):
        if len(args) == 2:
            raw_data = np.asarray(args[1], dtype=float)
        elif len(args) == 1:
            raw_data = np.asarray(args[0], dtype=float)
        else:
            raw_data = np.zeros(0)

        # NOTE: fast_preview no longer skips the low-pass filter below - an
        # earlier version did, to save time on in-between scrub frames, but
        # that displayed the raw unfiltered signal (baseline wander + HF
        # noise the filter normally removes) and made the trace look like
        # pure noise while dragging. Filtering always runs; fast_preview now
        # only skips the RR-interval recompute (self._rr_intervals), which
        # isn't drawn as part of the trace itself.
        if len(raw_data) > 15:
            try:
                if getattr(self, '_filter_ba', None) is None:
                    from scipy.signal import butter
                    self._filter_ba = butter(2, 25.0 / (self._fs / 2.0), btype='lowpass')
                from scipy.signal import filtfilt
                b, a = self._filter_ba
                self._data = filtfilt(b, a, raw_data)
            except Exception:
                self._data = raw_data
        else:
            self._data = raw_data

        self._beat_annotations = beat_annotations or []
        self._structured_events = structured_events or []
        self._start_sec = start_sec
        self._rr_intervals = self._calculate_rr_intervals() if beat_annotations else []
        self.update()
    
    def _calculate_rr_intervals(self):
        """Calculate RR intervals (N-N intervals) between consecutive R-peaks."""
        if not self._beat_annotations:
            return []
        
        # Filter and sort beats by timestamp
        beats_in_window = []
        end_sec = self._start_sec + (len(self._data) / self._fs) if len(self._data) > 0 else self._start_sec
        
        for beat in self._beat_annotations:
            ts = beat.get('timestamp', 0.0)
            if self._start_sec <= ts <= end_sec:
                beats_in_window.append({
                    'timestamp': ts,
                    'label': beat.get('label', 'N')
                })
        
        beats_in_window.sort(key=lambda x: x['timestamp'])
        
        # Calculate intervals between consecutive beats
        intervals = []
        for i in range(len(beats_in_window) - 1):
            curr_beat = beats_in_window[i]
            next_beat = beats_in_window[i + 1]
            
            # Calculate RR interval in ms
            rr_ms = (next_beat['timestamp'] - curr_beat['timestamp']) * 1000.0
            
            intervals.append({
                'start_ts': curr_beat['timestamp'],
                'end_ts': next_beat['timestamp'],
                'rr_ms': rr_ms,
                'start_label': curr_beat['label'],
                'end_label': next_beat['label']
            })
        
        return intervals

    def wheelEvent(self, event):
        """Scroll time scrollbar forward/backward on mouse wheel."""
        delta = event.angleDelta().y()
        if delta != 0:
            step = -1.0 if delta > 0 else 1.0  # Wheel up = scroll forward, wheel down = backward
            parent = self.parentWidget()
            while parent is not None:
                if hasattr(parent, 'time_scrollbar') and parent.time_scrollbar is not None:
                    curr_val = parent.time_scrollbar.value()
                    # Step by 100 units (1.0 second) per wheel click
                    parent.time_scrollbar.setValue(curr_val + int(step * 100))
                    break
                parent = parent.parentWidget()

    def mousePressEvent(self, event):
        # Let the event propagate to parent's eventFilter for vertical line handling
        # Only consume the event if we're in a tool mode
        
        if self._mode == TOOL_MAGNIFY and event.button() == Qt.LeftButton:
            # Click-to-lock magnifier: each click moves the zoom lens to that point.
            # Switching away from the tool clears the lock.
            self._magnify_locked = True
            self._magnify_pos = event.pos()
            self._hover_pos = event.pos()
            host = self._find_magnifier_host()
            if host is not None:
                host.set_magnifier_focus(self, self._magnifier_source_payload(), event.pos())
            self.update()
            return

        if self._mode == TOOL_CALIPER and event.button() == Qt.LeftButton:
            if self._caliper_line1 is None:
                self._caliper_line1 = event.pos().x()
                self._caliper_line2 = None
            else:
                self._caliper_line2 = event.pos().x()
            self.update()
            return

        # Ruler/Measuring ruler: handle two-click measurement
        if self._mode == TOOL_RULER and event.button() == Qt.LeftButton:
            if self._ruler_start is None:
                # First click: set start point
                self._ruler_start = event.pos()
                self._ruler_end = None
            else:
                # Second click: set end point (keeps start point fixed)
                self._ruler_end = event.pos()
            self.update()
            return

        if self._mode != TOOL_SELECT:
            self._start_pos = event.pos()
            self._curr_pos = event.pos()
            self._hover_pos = event.pos()
            self.update()
            return
        
        # In SELECT mode, don't consume the event - let it propagate to eventFilter
        # This allows the Full Disclosure dialog to handle vertical line drawing
        event.ignore()

    def mouseMoveEvent(self, event):
        if self._mode == TOOL_MAGNIFY:
            if self._magnify_locked:
                if event.buttons() & Qt.LeftButton:
                    self._magnify_pos = event.pos()
                    host = self._find_magnifier_host()
                    if host is not None:
                        host.set_magnifier_focus(self, self._magnifier_source_payload(), event.pos())
                self.update()
                return
            self._hover_pos = event.pos()
            self.update()
            return
        
        # For caliper tool, show preview of second line while hovering (if first line is set)
        if self._mode == TOOL_CALIPER:
            if self._caliper_line1 is not None and self._caliper_line2 is None:
                # First line is set, show preview of second line as hover
                self._hover_pos = event.pos()
                self.update()
            return
        
        # For ruler tool, show preview of end point while hovering (if start point is set)
        if self._mode == TOOL_RULER:
            if self._ruler_start is not None and self._ruler_end is None:
                # Start point is set, show preview of end point as hover
                self._hover_pos = event.pos()
                self.update()
            return
            
        self._hover_pos = event.pos()
        if self._mode != TOOL_SELECT and self._start_pos is not None:
            self._curr_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._mode == TOOL_MAGNIFY:
            if not self._magnify_locked:
                self._hover_pos = event.pos()
            self.update()
            return
        if self._mode != TOOL_SELECT:
            self._curr_pos = event.pos()
            self.update()

    def leaveEvent(self, event):
        if not self._magnify_locked:
            self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def _get_display_signal(self):
        if self._data.size < 2:
            return np.array([]), 0.0, 1.0
        sig = np.asarray(self._data, dtype=float)
        baseline = float(np.median(sig))
        
        if self.lead_name == "aVR":
            # aVR: shift so baseline centers at -2048
            target_center = -2048.0
            shift = target_center - baseline
            d = sig + shift
            mn = -4096.0
            rng = 4096.0
        else:
            # All other leads: shift so baseline centers at 2048
            target_center = 2048.0
            shift = target_center - baseline
            d = sig + shift
            mn = 0.0
            rng = 4096.0
        
        # Apply gain: higher gain = taller waves
        # Center around baseline, apply gain, then shift back
        if hasattr(self, '_gain') and self._gain != 1.0:
            center = (mn + rng / 2.0)
            d = center + (d - center) * self._gain
            
        return d, mn, rng

    def _x_to_index(self, x: int, width: int, n: int) -> int:
        if n <= 1 or width <= 1:
            return 0
        x = max(0, min(x, width - 1))
        return int(round((x / float(width - 1)) * (n - 1)))

    def _check_and_store_clicked_beat(self, click_x: int):
        """Check if user clicked near a beat peak and store it for display."""
        w = self.width()
        if w <= 0 or not hasattr(self, '_data') or self._data.size < 2:
            self._clicked_beat_timestamp = None
            self._clicked_beat_label = None
            self._clicked_beat_x_pos = None
            return
        
        end_sec = self._start_sec + len(self._data) / self._fs
        tolerance_pixels = 15  # Reasonable tolerance for user clicking
        
        # Try to snap to the closest detected peak timestamp first
        parent_dialog = None
        curr = self.parent()
        while curr is not None:
            if hasattr(curr, '_detected_r_peaks'):
                parent_dialog = curr
                break
            curr = curr.parent()
            
        # 1. Try to snap to detected R-peaks from parent_dialog
        if parent_dialog and hasattr(parent_dialog, '_detected_r_peaks') and parent_dialog._detected_r_peaks:
            best_ts = None
            best_dist_pixels = 999999
            for ts in parent_dialog._detected_r_peaks:
                pct = (ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                beat_x = int(pct * w)
                dist = abs(click_x - beat_x)
                if dist < best_dist_pixels:
                    best_dist_pixels = dist
                    best_ts = ts
                    
            if best_dist_pixels <= tolerance_pixels:
                self._clicked_beat_timestamp = best_ts
                self._clicked_beat_label = 'N'
                if hasattr(self, '_beat_annotations') and self._beat_annotations:
                    for beat in self._beat_annotations:
                        if abs(beat.get('timestamp', 0) - best_ts) < 0.05:
                            self._clicked_beat_label = beat.get('label', 'N')
                            break
                pct = (best_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                self._clicked_beat_x_pos = int(pct * w)
                self.update()
                return
        
        # 2. Try to snap to provided beat_annotations
        elif hasattr(self, '_beat_annotations') and self._beat_annotations:
            best_ts = None
            best_dist_pixels = 999999
            for beat in self._beat_annotations:
                ts = beat.get('timestamp', 0.0)
                if self._start_sec <= ts <= end_sec:
                    pct = (ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                    beat_x = int(pct * w)
                    dist = abs(click_x - beat_x)
                    if dist < best_dist_pixels:
                        best_dist_pixels = dist
                        best_ts = ts
                        
            if best_dist_pixels <= tolerance_pixels:
                self._clicked_beat_timestamp = best_ts
                self._clicked_beat_label = 'N'
                for beat in self._beat_annotations:
                    if abs(beat.get('timestamp', 0) - best_ts) < 0.05:
                        self._clicked_beat_label = beat.get('label', 'N')
                        break
                pct = (best_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                self._clicked_beat_x_pos = int(pct * w)
                self.update()
                return
        
        # 3. Fallback: Check if click is actually ON or VERY NEAR a local maximum
        # Convert click position to sample index
        click_sample_idx = int((click_x / float(w)) * len(self._data))
        click_sample_idx = max(0, min(click_sample_idx, len(self._data) - 1))
        
        # Look in a small neighborhood (+/-20ms) around the click
        neighborhood_size = max(5, int(0.02 * self._fs))  # 20ms window
        start_check = max(0, click_sample_idx - neighborhood_size)
        end_check = min(len(self._data), click_sample_idx + neighborhood_size)
        
        if end_check > start_check:
            neighborhood = self._data[start_check:end_check]
            
            # Find the maximum value in the neighborhood
            local_max_idx = np.argmax(neighborhood)
            local_max_sample_idx = start_check + local_max_idx
            local_max_value = neighborhood[local_max_idx]
            
            # Check if this is a significant peak (not just noise)
            # Compare to median baseline of entire strip
            baseline = np.median(self._data)
            peak_height = local_max_value - baseline
            
            # Need at least 100 ADC units above baseline (adjust based on your data range)
            # This filters out flat areas and noise
            overall_range = np.max(self._data) - np.min(self._data)
            min_peak_height = 0.2 * overall_range  # Peak must be at least 20% of total range
            
            if peak_height > min_peak_height:
                # Found a significant peak! Now check if it's close enough to click
                ts = self._start_sec + (local_max_sample_idx / self._fs)
                pct = (ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                beat_x = int(pct * w)
                
                distance = abs(click_x - beat_x)
                
                if distance <= tolerance_pixels:
                    # This is a valid peak click!
                    self._clicked_beat_timestamp = ts
                    self._clicked_beat_label = 'N'
                    self._clicked_beat_x_pos = beat_x
                    self.update()
                    return
        
        # No valid peak found near click - clear everything
        self._clicked_beat_timestamp = None
        self._clicked_beat_label = None
        self._clicked_beat_x_pos = None
        self.update()

    def _find_beats_in_range(self, start_x: int, end_x: int):
        """Find all R-peaks EXACTLY between start_x and end_x pixel positions."""
        w = self.width()
        if w <= 0 or not hasattr(self, '_data') or self._data.size < 2:
            return []
        
        # Convert pixel positions to sample indices PRECISELY
        start_sample = int((start_x / float(w)) * len(self._data))
        end_sample = int((end_x / float(w)) * len(self._data))
        
        # Clamp to valid range
        start_sample = max(0, min(start_sample, len(self._data) - 1))
        end_sample = max(0, min(end_sample, len(self._data) - 1))
        
        # Ensure start < end
        if start_sample > end_sample:
            start_sample, end_sample = end_sample, start_sample
        
        # Ensure minimum range
        if (end_sample - start_sample) < 20:
            return []
        
        try:
            # Extract ONLY the data in the selected range
            range_data = self._data[start_sample:end_sample + 1]
            
            beat_timestamps = []
            end_sec = self._start_sec + len(self._data) / self._fs
            
            # Peak detection parameters
            min_samples_between = int(0.3 * self._fs)  # 300ms minimum
            
            # Use FULL DATA baseline for consistency
            full_baseline = np.median(self._data)
            full_range = np.max(self._data) - np.min(self._data)
            threshold = full_baseline + 0.25 * full_range
            
            last_peak_sample = start_sample - min_samples_between - 1
            
            # Find local maxima ONLY within the selected range
            for i in range(len(range_data)):
                current_val = range_data[i]
                
                # Skip if below threshold
                if current_val <= threshold:
                    continue
                
                # Calculate actual sample index (in full data array)
                actual_idx = start_sample + i
                
                # Check if this is a local maximum
                is_local_max = False
                
                if i == 0:
                    # First point in range - only compare with right
                    if len(range_data) > 1 and current_val > range_data[i + 1]:
                        is_local_max = True
                elif i == len(range_data) - 1:
                    # Last point in range - only compare with left
                    if current_val > range_data[i - 1]:
                        is_local_max = True
                else:
                    # Middle points - compare both sides
                    if current_val > range_data[i - 1] and current_val > range_data[i + 1]:
                        is_local_max = True
                
                # Check spacing from previous peak
                if not is_local_max:
                    continue
                
                if (actual_idx - last_peak_sample) < min_samples_between:
                    continue
                
                # FINAL CHECK: Ensure beat is within selected range
                if actual_idx < start_sample or actual_idx > end_sample:
                    continue
                
                ts = self._start_sec + (actual_idx / self._fs)
                
                # Final timestamp validation
                if self._start_sec <= ts <= end_sec:
                    beat_timestamps.append(ts)
                    last_peak_sample = actual_idx
            
            return sorted(beat_timestamps)
            
        except Exception as e:
            return []

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        minor_pen = QPen(QColor(COL_GRID_MINOR))
        minor_pen.setWidth(1)
        major_pen = QPen(QColor(COL_GRID_MAJOR))
        major_pen.setWidth(1)
        for gx in range(0, w, 20):
            painter.setPen(major_pen if gx % 100 == 0 else minor_pen)
            painter.drawLine(gx, 0, gx, h)
        for gy in range(0, h, 20):
            painter.setPen(major_pen if gy % 100 == 0 else minor_pen)
            painter.drawLine(0, gy, w, gy)

        if self._data.size < 2:
            painter.end()
            return
        d, mn, rng = self._get_display_signal()
        if d.size < 2:
            painter.end()
            return
        
        # --- Paper speed: control how many samples are visible per screen width ---
        # At 25mm/s: show all data. At 50mm/s: stretch (show half). At 12.5mm/s: compress (show double).
        speed_factor = max(0.25, min(float(self._speed) / 25.0, 4.0))
        n_visible = max(2, int(round(len(d) / speed_factor)))
        if n_visible < len(d):
            d = d[-n_visible:]   # show the most recent n_visible samples (stretched)
        # (if n_visible >= len(d) we show all data, which appears compressed at slow speed)
        # Apply speed factor to x_scale so the waveform stretches/compresses visually
        x_scale = w / max(1, len(d) - 1) * speed_factor
        
        # Determine colored intervals based on annotations and structured events
        colored_intervals = []
        end_sec = self._start_sec + len(d) / self._fs
        
        # 1. Color full regions for arrhythmias from _structured_events
        # CRITICAL FIX: Only apply structured event coloring if disable_all_coloring is False
        # For replay/overview panels (disable_all_coloring=True), always show plain green waveforms
        # For Full Disclosure (disable_all_coloring=False), show waveform coloring regardless of show_annotations
        if not self._disable_all_coloring and hasattr(self, '_structured_events') and self._structured_events:
            # Expanded color mapping for all arrhythmia types
            label_colors = {
                "V": "#FF3333",      # Ventricular Premature - Red
                "AF": "#FF00FF",     # Atrial Fibrillation - Magenta
                "S": "#00FFFF",      # Sinus Bradycardia/Tachycardia - Cyan
                "P": "#FF00FF",      # Paced/AV Blocks - Magenta
                "C": "#FFA500",      # Conduction Blocks - Orange
                "T": "#9932CC",      # TV Paced - Purple
                "A": "#FFFF00",      # ACLS - Yellow
                "X": "#0000FF",      # Asystole/Artifact - Blue
            }
            # Go through events and find regions
            for i, ev in enumerate(self._structured_events):
                ev_ts = float(ev.get('timestamp', 0.0) or 0.0)
                region_start_ts = ev_ts
                
                # Quickly determine region end to skip invisible events
                region_end_ts = ev.get('end_timestamp')
                if region_end_ts is None:
                    if i + 1 < len(self._structured_events):
                        region_end_ts = float(self._structured_events[i+1].get('timestamp', 0.0) or 0.0)
                    else:
                        region_end_ts = end_sec + 10.0
                else:
                    region_end_ts = float(region_end_ts)
                    
                if region_end_ts < self._start_sec or region_start_ts > end_sec:
                    continue
                    
                ev_lbl = str(ev.get('label', '')).lower()
                ev_lbl_orig = str(ev.get('label', ''))
                source = str(ev.get('source', '')).lower()
                is_manual_structured = source in {'manual', 'manual_parallel_multi', 'restored_parallel_multi'}

                # Check if this event starts an arrhythmia
                active_label = 'N'
                # Check for single-letter labels first (from manual marking)
                if ev_lbl == 'v':
                    active_label = 'V'
                elif ev_lbl == 's':
                    active_label = 'S'
                elif ev_lbl == 'af':
                    active_label = 'AF'
                elif ev_lbl == 'p':
                    active_label = 'P'
                elif ev_lbl == 'c':  # Conduction blocks
                    active_label = 'C'
                elif ev_lbl == 't':  # TV Paced
                    active_label = 'T'
                elif ev_lbl == 'a':  # ACLS
                    active_label = 'A'
                elif ev_lbl == 'x':
                    active_label = 'X'
                # Check for full label names (from auto-detection)
                elif 'asystole' in ev_lbl:
                    active_label = 'X'
                elif 'ventricular fibrillation' in ev_lbl or 'vfib' in ev_lbl:
                    active_label = 'V'
                elif 'ventricular tachycardia' in ev_lbl or 'vtach' in ev_lbl:
                    active_label = 'V'
                elif 'atrial fibrillation' in ev_lbl or 'afib' in ev_lbl:
                    active_label = 'AF'
                elif 'atrial flutter' in ev_lbl or 'aflutter' in ev_lbl:
                    active_label = 'AF'
                elif 'sinus bradycardia' in ev_lbl:
                    active_label = 'S'
                elif 'sinus tachycardia' in ev_lbl:
                    active_label = 'S'
                elif 'bradycardia (non-sinus)' in ev_lbl:
                    active_label = 'S'
                elif 'tachycardia (non-sinus)' in ev_lbl:
                    active_label = 'S'
                elif '1st-degree av block' in ev_lbl or '2nd-degree av block' in ev_lbl or '3rd-degree av block' in ev_lbl:
                    active_label = 'P'
                elif 'right bundle branch block' in ev_lbl or 'left bundle branch block' in ev_lbl:
                    active_label = 'P'
                elif 'premature ventricular contraction' in ev_lbl or 'pvc' in ev_lbl:
                    active_label = 'V'
                elif 'premature atrial contraction' in ev_lbl or 'pac' in ev_lbl:
                    active_label = 'S'
                elif 'st elevation' in ev_lbl or 'st depression' in ev_lbl:
                    active_label = 'X'
                
                # Also check the original label (case-sensitive) for single letters
                if active_label == 'N':
                    if ev_lbl_orig == 'V':
                        active_label = 'V'
                    elif ev_lbl_orig == 'S':
                        active_label = 'S'
                    elif ev_lbl_orig == 'AF':
                        active_label = 'AF'
                    elif ev_lbl_orig == 'P':
                        active_label = 'P'
                    elif ev_lbl_orig == 'X':
                        active_label = 'X'

                if active_label != 'N':
                    # Use color from event if available, otherwise use label_colors
                    color = ev.get('color', label_colors.get(active_label, "#FF3333"))
                    # Calculate overlapping indices
                    if region_end_ts >= self._start_sec and region_start_ts <= end_sec:
                        # Determine if this arrhythmia should color the whole region or just QRS complexes.
                        # Asystole, Artifact, and Ventricular Fibrillation lack normal QRS and should color the whole region.
                        # Tachycardia, Bradycardia, AFib, PVC, PAC etc. should color ONLY the QRS complexes.
                        color_whole_region = is_manual_structured and ('ventricular fibrillation' in ev_lbl or 'vfib' in ev_lbl or 
                                              'asystole' in ev_lbl or 'artifact' in ev_lbl or active_label == 'X')
                        
                        r_peak_tss = []
                        if not color_whole_region:
                            if hasattr(self, '_beat_annotations') and self._beat_annotations:
                                r_peak_tss = [b.get('timestamp', 0.0) for b in self._beat_annotations]
                            else:
                                r_peak_tss = getattr(self, '_detected_peaks_cache', [])
                                if not r_peak_tss:
                                    parent = self.parent()
                                    while parent is not None:
                                        if hasattr(parent, '_detected_r_peaks'):
                                            r_peak_tss = parent._detected_r_peaks
                                            break
                                        parent = parent.parent()
                        
                        # Find peaks inside this region
                        region_peaks = [ts for ts in r_peak_tss if region_start_ts <= ts <= region_end_ts]
                        
                        if color_whole_region or (not region_peaks and is_manual_structured):
                            # Color the entire region (fallback for no peaks, or explicitly requested for X / VFib)
                            start_idx = max(0, int((region_start_ts - self._start_sec) * self._fs))
                            end_idx = min(len(d) - 1, int((region_end_ts - self._start_sec) * self._fs))
                            if start_idx < end_idx:
                                colored_intervals.append((start_idx, end_idx, color))
                        else:
                            # Color ONLY the QRS complexes (+/- 60ms around each R-peak)
                            for ts in region_peaks:
                                qrs_start_ts = max(self._start_sec, ts - 0.06)
                                qrs_end_ts = min(end_sec, ts + 0.06)
                                start_idx = max(0, int((qrs_start_ts - self._start_sec) * self._fs))
                                end_idx = min(len(d) - 1, int((qrs_end_ts - self._start_sec) * self._fs))
                                if start_idx < end_idx:
                                    colored_intervals.append((start_idx, end_idx, color))
        
        # ------------------------------------------------------------------
        # Auto-Detection Pattern-Specific QRS Peak Coloring
        # ------------------------------------------------------------------
        # Colors each auto-detected QRS peak using its EXACT badge pattern color
        # (e.g. VT Run=#FF0000, Bigeminy=#FF6600, Couplet=#FF3333, Single PVC=#FF6666,
        #        SVT Run=#00CCFF, PAC Bigeminy=#00FFFF, PAC Couplet=#66FFFF, Single PAC=#99FFFF)
        #
        # IMPORTANT: Manual beats (is_manual=True) are COMPLETELY EXCLUDED from this pass.
        # Manual marking coloring is handled exclusively by the _structured_events loop above.
        # ------------------------------------------------------------------
        if not self._disable_all_coloring and hasattr(self, '_beat_annotations') and self._beat_annotations:
            try:
                from ecg.holter.holter_summary_calc import label_beat_sequences, _get_badge_appearance

                # Manual Beat Marking QRS Peak Coloring
                manual_label_colors = {
                    "S": "#00FFFF",      # Supraventricular - Cyan
                    "P": "#FF00FF",      # Premature - Magenta
                    "V": "#FF3333",      # Ventricular - Red
                    "C": "#FFA500",      # Conduction - Orange
                    "T": "#9932CC",      # TV Paced - Dark Orchid (Purple)
                    "A": "#FFFF00",      # ACLS - Yellow
                    "X": "#0000FF",      # Artifact - Blue
                }

                for beat in self._beat_annotations:
                    if beat.get('is_manual', False):
                        lbl = str(beat.get('label', 'N')).upper()
                        code = lbl.split('(')[1].split(')')[0] if ('(' in lbl and ')' in lbl) else lbl
                        code = beat.get('short_code', code)

                        if code in ('N', ''):
                            continue

                        color = beat.get('color') or manual_label_colors.get(code, "#FFFF00")
                        if not color or color.upper() in ('#FFFFFF', '#FFF', 'WHITE', '#00FF00'):
                            continue

                        ts = float(beat.get('timestamp', 0.0))
                        if not (self._start_sec <= ts <= end_sec):
                            continue

                        qrs_start_ts = max(self._start_sec, ts - 0.06)
                        qrs_end_ts   = min(end_sec,          ts + 0.06)
                        s_idx = max(0,          int((qrs_start_ts - self._start_sec) * self._fs))
                        e_idx = min(len(d) - 1, int((qrs_end_ts   - self._start_sec) * self._fs))
                        if s_idx < e_idx:
                            colored_intervals.append((s_idx, e_idx, color))

                # Only process auto-detected beats — never touch manual beats
                auto_beats_for_color = [
                    b for b in self._beat_annotations
                    if not b.get('is_manual', False)
                ]

                if auto_beats_for_color:
                    # Run pattern sequencing to assign rhythm_pattern (Bigeminy, Couplet, VT Run etc.)
                    # Safe to call multiple times — idempotent operation
                    label_beat_sequences(auto_beats_for_color)

                    for beat in auto_beats_for_color:
                        # Get short code: V, S, AF, P  (skip N — no coloring needed)
                        lbl = str(beat.get('label', 'N')).upper()
                        code = lbl.split('(')[1].split(')')[0] if ('(' in lbl and ')' in lbl) else lbl
                        # Also check short_code if already set by reclassifier
                        code = beat.get('short_code', code)

                        if code not in ('V', 'S', 'AF', 'P'):
                            continue

                        pattern = beat.get('rhythm_pattern', '')
                        # Get the pattern-specific color (granular: Bigeminy vs Couplet vs Single)
                        color, _ = _get_badge_appearance(code, pattern, beat)

                        # Skip white / normal / suppressed colors
                        if not color or color.upper() in ('#FFFFFF', '#FFF', 'WHITE'):
                            continue

                        ts = float(beat.get('timestamp', 0.0))
                        if not (self._start_sec <= ts <= end_sec):
                            continue

                        # ±60 ms window centered on R-peak
                        qrs_start_ts = max(self._start_sec, ts - 0.06)
                        qrs_end_ts   = min(end_sec,          ts + 0.06)
                        s_idx = max(0,          int((qrs_start_ts - self._start_sec) * self._fs))
                        e_idx = min(len(d) - 1, int((qrs_end_ts   - self._start_sec) * self._fs))
                        if s_idx < e_idx:
                            colored_intervals.append((s_idx, e_idx, color))

            except Exception as _exc:
                pass   # Never crash paint — coloring is decorative

        colored_intervals.sort(key=lambda x: x[0])
        
        default_pen = QPen(QColor(self._color))
        default_pen.setWidthF(self._pen_width)

        n = len(d)
        # Vectorized coordinate computation (numpy) instead of a per-sample Python loop.
        xs = np.arange(n, dtype=np.float64) * x_scale
        ys = h - (d - mn) / rng * h

        # Build the whole trace as a single QPainterPath and draw it in one call.
        # This replaces issuing one drawLine() call per sample (extremely slow for
        # large windows like 1 Min / 2 Min, and the main cause of scrollbar-drag lag),
        # cutting per-frame Qt paint calls from O(n) down to O(1) + a handful for
        # colored arrhythmia/annotation sub-segments.
        painter.setPen(default_pen)
        trace_path = QPainterPath()
        trace_path.moveTo(xs[0], ys[0])
        for xv, yv in zip(xs[1:], ys[1:]):
            trace_path.lineTo(xv, yv)
        painter.drawPath(trace_path)

        # Overlay colored sub-segments (arrhythmia regions / annotated beats) on top
        # of the base trace â€” only a handful of these exist per window, so drawing
        # each as its own short path is cheap.
        for start_idx, end_idx, color in colored_intervals:
            start_idx = max(0, start_idx)
            end_idx = min(n - 1, end_idx)
            if end_idx <= start_idx:
                continue
            seg_pen = QPen(QColor(color))
            seg_pen.setWidthF(self._pen_width * 2.0)
            painter.setPen(seg_pen)
            seg_path = QPainterPath()
            seg_path.moveTo(xs[start_idx], ys[start_idx])
            for xv, yv in zip(xs[start_idx + 1:end_idx + 1], ys[start_idx + 1:end_idx + 1]):
                seg_path.lineTo(xv, yv)
            painter.drawPath(seg_path)

        # --- Draw Auto-Detected Arrhythmia Labels ---
        # (REMOVED: Floating overlay labels have been migrated to the ArrhythmiaBadgeBar at the bottom of the window)


            
        # --- Draw Clinical Beat Annotations ---
        lead_name = getattr(self, 'lead_name', '')
        show_vertical_lines = getattr(self, '_show_vertical_lines', True)
        
        detected_peaks = []
        annotated_beats = {}  # Map timestamp to beat annotation dictionary
        end_sec = self._start_sec + (len(d) / self._fs) if len(d) > 0 else self._start_sec
        
        # First, get annotated beats from annotations if available
        if hasattr(self, '_beat_annotations') and self._beat_annotations:
            for beat in self._beat_annotations:
                ts = beat['timestamp']
                if self._start_sec <= ts <= end_sec:
                    annotated_beats[ts] = beat
        
        if lead_name == 'I' and show_vertical_lines:
            # Only redo R-peak detection when the underlying data has actually
            # changed (id(self._data) changes every time set_data() gets new
            # samples). Otherwise reuse the last result â€” a plain repaint
            # (resize, overlay refresh, etc.) should not re-run scipy.find_peaks
            # over the whole window again.
            data_key = id(self._data)
            if getattr(self, '_peak_cache_key', None) == data_key and getattr(self, '_peak_cache_start_sec', None) == self._start_sec:
                detected_peaks = self._detected_peaks_cache
            else:
                # Detect R-peaks using the full Pan-Tompkins algorithm
                # (same function used by heart_rate.py and comprehensive_analysis.py)
                if len(d) > 50:
                    try:
                        try:
                            from ecg.pan_tompkins import pan_tompkins
                        except ImportError:
                            from ...pan_tompkins import pan_tompkins

                        signal = np.asarray(d, dtype=float)
                        # pan_tompkins() returns sample indices of R-peaks
                        peak_indices = pan_tompkins(signal, fs=self._fs)

                        # Convert peak indices to timestamps
                        for peak_idx in peak_indices:
                            ts = self._start_sec + (int(peak_idx) / self._fs)
                            detected_peaks.append(ts)
                        
                        # Store detected peaks in parent panel for cross-lead vertical lines
                        parent = self.parent()
                        while parent is not None:
                            if hasattr(parent, '_detected_r_peaks'):
                                parent._detected_r_peaks = detected_peaks
                                parent._r_peak_start_sec = self._start_sec
                                parent._r_peak_data_len = len(d)
                                break
                            parent = parent.parent()
                            
                    except Exception as e:
                        print(f"[ECGStripCanvas] R-peak detection error: {e}")

                self._detected_peaks_cache = detected_peaks
                self._peak_cache_key = data_key
                self._peak_cache_start_sec = self._start_sec
            
            # Calculate and store RR intervals from detected peaks (always keep for internal use)
            if detected_peaks:
                self._rr_intervals = []
                for i in range(len(detected_peaks) - 1):
                    curr_ts = detected_peaks[i]
                    next_ts = detected_peaks[i + 1]
                    rr_ms = (next_ts - curr_ts) * 1000.0
                    
                    self._rr_intervals.append({
                        'start_ts': curr_ts,
                        'end_ts': next_ts,
                        'rr_ms': rr_ms,
                        'start_label': 'N',
                        'end_label': 'N'
                    })

        # Draw beat labels ONLY for:
        #   1. Explicitly annotated beats (from _beat_annotations) with non-N labels (manual marks)
        #   2. Detected peaks that fall inside an active arrhythmia structured event (non-N)
        # Default auto-detected N labels are intentionally suppressed in Full Disclosure view.
        if self._show_annotations:
            try:
                from ecg.holter.holter_summary_calc import get_template_beats_for_badges
                
                # Use beat annotations (which includes all beats in window) and structured events
                window_beats = list(getattr(self, '_beat_annotations', []))
                window_events = getattr(self, '_structured_events', [])
                
                # If _beat_annotations is empty, construct beat objects from Pan-Tompkins detected peaks
                # so that Stage 4, 5 & 7 badge logic can run reclassification and generate badges
                if not window_beats and getattr(self, '_detected_peaks_cache', None):
                    peaks = self._detected_peaks_cache
                    for i, ts in enumerate(peaks):
                        rr_ms = (peaks[i] - peaks[i-1]) * 1000.0 if i > 0 else 800.0
                        s_idx = max(0, int((ts - self._start_sec - 0.3) * self._fs))
                        e_idx = min(len(d), int((ts - self._start_sec + 0.3) * self._fs))
                        seg = d[s_idx:e_idx] if s_idx < e_idx else None
                        window_beats.append({
                            'timestamp': ts,
                            'rr_ms': rr_ms,
                            'label': 'N',
                            'short_code': 'N',
                            'segment': seg,
                            'fs': self._fs
                        })
                
                # DEBUG: Print what we have
                if lead_name == 'I':  # Only print once per refresh
                    print(f"[BADGE DEBUG] Lead={lead_name}, window_beats={len(window_beats)}, window_events={len(window_events)}")
                
                badges = get_template_beats_for_badges(window_beats, window_events)
                
                if lead_name == 'I':  # Only print once per refresh
                    print(f"[BADGE DEBUG] Generated {len(badges)} badges")
                    if badges:
                        for b in badges[:3]:  # Show first 3
                            print(f"  Badge: {b['name']} at {b['timestamp']:.2f}s, color={b['color']}")
                
                # Add explicitly manual annotations if not already there
                manual_badges = []
                for annot_ts, beat in annotated_beats.items():
                    if beat.get('is_manual', False) and beat.get('label', 'N') != 'N':
                        lbl = beat.get('label', 'N')
                        short_code = lbl
                        if '(' in lbl and ')' in lbl:
                            short_code = lbl.split('(')[1].split(')')[0]
                        # Only add if not already in badges at similar timestamp
                        if not any(abs(b['timestamp'] - annot_ts) < 0.1 for b in badges):
                            color = beat.get('color', '#FFFF00')
                            if not color or color == '#FFFF00': # fallback
                                 if short_code == 'V': color = "#FF3333"
                                 elif short_code == 'S': color = "#00FFFF"
                                 elif short_code in ['AF', 'P']: color = "#FF00FF"
                            manual_badges.append({
                                'timestamp': annot_ts,
                                'code': short_code,
                                'name': lbl,
                                'color': color
                            })
                
                all_badges = badges + manual_badges
                
                if all_badges:
                    start_time = self._get_reader_start_time()
                    from datetime import datetime
                    
                    for badge in all_badges:
                        ts = badge['timestamp']
                        if not (self._start_sec <= ts <= end_sec):
                            continue

                        color = badge.get('color', '#FFFFFF')
                        # Skip white labels (e.g. [TACHY] Normal) - only render colored arrhythmia badges
                        if color and color.upper() in ['#FFFFFF', '#FFF', 'WHITE']:
                            continue
                        
                        pct = (ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                        bx = int(pct * w)
                        
                        color = badge['color']
                        code_str = f"[{badge['code']}]"
                        name_str = badge['name']
                        time_str = datetime.fromtimestamp(start_time + ts).strftime('%H:%M:%S') if start_time else ""
                        
                        # Set up fonts
                        restore_font = painter.font()
                        font = painter.font()
                        font.setPixelSize(10)
                        font.setBold(True)
                        painter.setFont(font)
                        
                        fm = painter.fontMetrics()
                        code_w = fm.horizontalAdvance(code_str)
                        name_w = fm.horizontalAdvance(name_str)
                        
                        font.setBold(False)
                        painter.setFont(font)
                        fm_time = painter.fontMetrics()
                        time_w = fm_time.horizontalAdvance(time_str)
                        
                        padding = 4
                        spacing = 4
                        total_w = code_w + name_w + time_w + (spacing * 2) + (padding * 2)
                        total_h = fm.height() + padding * 2
                        
                        # Position badge slightly above the trace
                        rect_x = bx - total_w // 2
                        rect_y = 5
                        
                        # Draw pill background (transparent dark with cyan border)
                        painter.setPen(QPen(QColor(color), 1))
                        painter.setBrush(QColor(20, 20, 20, 200)) # Dark transparent
                        painter.drawRoundedRect(rect_x, rect_y, total_w, total_h, 4, 4)
                        
                        # Draw code (cyan, bold)
                        font.setBold(True)
                        painter.setFont(font)
                        curr_x = rect_x + padding
                        painter.setPen(QPen(QColor(color)))
                        painter.drawText(curr_x, rect_y + fm.ascent() + padding, code_str)
                        curr_x += code_w + spacing
                        
                        # Draw name (white, bold)
                        painter.setPen(QPen(QColor("#FFFFFF")))
                        painter.drawText(curr_x, rect_y + fm.ascent() + padding, name_str)
                        curr_x += name_w + spacing
                        
                        # Draw time (light blue, normal)
                        font.setBold(False)
                        painter.setFont(font)
                        painter.setPen(QPen(QColor("#A0C4E8")))
                        painter.drawText(curr_x, rect_y + fm.ascent() + padding, time_str)
                        
                        # Restore font
                        painter.setFont(restore_font)
            except Exception as e:
                print(f"[ECGStripCanvas] Error drawing beat badges: {e}")

        
        # --- Draw square boxes for selected beats (only on Lead I) ---
        # NOTE: Vertical lines are now drawn by VerticalLineOverlay to avoid gaps
        end_sec = self._start_sec + len(d) / self._fs
        
        # ONLY DRAW BOXES ON LEAD I
        if lead_name == 'I':
            # Check if we have multiple selected beats (drag selection)
            if hasattr(self, '_selected_beats') and self._selected_beats and len(self._selected_beats) > 0:
                # Draw boxes for all selected beats with their specific colors
                painter.setBrush(Qt.NoBrush)
                
                for beat_ts in self._selected_beats:
                    # Verify beat is actually in this canvas's visible range
                    if self._start_sec <= beat_ts <= end_sec:
                        pct = (beat_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                        bx = int(pct * w)
                        
                        # Find the beat's color from annotations
                        beat_color = "#FFFF00"  # Default yellow
                        beat_label = "N"
                        if hasattr(self, '_beat_annotations'):
                            for beat in self._beat_annotations:
                                if abs(beat['timestamp'] - beat_ts) < 0.01:  # Match timestamp
                                    beat_color = beat.get('color', "#FFFF00")
                                    beat_label = beat.get('label', 'N')
                                    break
                        
                        # Box removed â€” label + time text is sufficient visual indicator
                        # Label text is already drawn by the main R-peak loop
            
            # Otherwise, draw single beat selection
            elif self._clicked_beat_timestamp is not None:
                if self._start_sec <= self._clicked_beat_timestamp <= end_sec:
                    pct = (self._clicked_beat_timestamp - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                    bx = int(pct * w)
                    
                    # Find the beat's color from annotations
                    beat_color = "#FFFF00"  # Default yellow
                    beat_label = "N"
                    if hasattr(self, '_beat_annotations'):
                        for beat in self._beat_annotations:
                            if abs(beat['timestamp'] - self._clicked_beat_timestamp) < 0.01:
                                beat_color = beat.get('color', "#FFFF00")
                                beat_label = beat.get('label', 'N')
                                break
                    
                    # Box removed â€” label + time text is sufficient visual indicator
        
        # --- Draw N-N (R-R) Interval Labels ONLY on Lead I ---
        if lead_name == 'I' and hasattr(self, '_rr_intervals') and self._rr_intervals and self._show_annotations and getattr(self, '_show_rr_numbers', True):  # Only show if annotations & rr numbers enabled
            font = painter.font()
            font.setPixelSize(9)
            font.setBold(False)
            painter.setFont(font)
            
            end_sec = self._start_sec + len(d) / self._fs
            
            for interval in self._rr_intervals:
                start_ts = interval['start_ts']
                end_ts = interval['end_ts']
                rr_ms = interval['rr_ms']
                
                # Check if interval is within visible window
                if (self._start_sec <= start_ts <= end_sec) or (self._start_sec <= end_ts <= end_sec):
                    # Calculate x coordinates for start and end
                    start_pct = (start_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                    end_pct = (end_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                    
                    start_x = int(start_pct * w)
                    end_x = int(end_pct * w)
                    mid_x = (start_x + end_x) // 2
                    
                    # Draw interval value between the two R-peaks
                    # Position it at mid-height of the waveform, slight offset from top
                    interval_text = f"{int(rr_ms)}"
                    
                    # Color: yellow for long intervals (>1200ms), orange for short (<600ms), white otherwise
                    if rr_ms > 1200:
                        color = "#FFFF00"  # Yellow for pauses
                    elif rr_ms < 600:
                        color = "#FFA500"  # Orange for fast beats
                    else:
                        color = "#CCCCCC"  # Light gray for normal intervals
                    
                    painter.setPen(QPen(QColor(color)))
                    painter.drawText(mid_x - 10, 22, interval_text)
                    
        if self._mode == TOOL_RULER:
            # Draw persistent ruler (start point stays after measurement)
            if self._ruler_start is not None:
                rpen = QPen(QColor("#00FFFF"), 2)
                painter.setPen(rpen)
                # Draw line from start point to end point (or hover if end not set yet)
                end_point = self._ruler_end if self._ruler_end is not None else (self._hover_pos if self._hover_pos else self._ruler_start)
                
                if self._ruler_end is not None:
                    # Final measurement: solid line
                    painter.drawLine(self._ruler_start, self._ruler_end)
                    dx = abs(self._ruler_end.x() - self._ruler_start.x())
                    dy = abs(self._ruler_end.y() - self._ruler_start.y())
                    ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
                    bpm = 60000 / ms if ms > 0 else 0
                    dy_mv = amplitude_mv_from_pixels(dy, max(1, h), rng, ADC_TO_MV)
                    painter.setPen(QPen(QColor("#00FFFF")))
                    painter.drawText(self._ruler_end.x(), max(12, self._ruler_end.y() - 6), ruler_label(ms, dy_mv, bpm))
                elif self._hover_pos is not None:
                    # Preview line while hovering (dashed)
                    pen_preview = QPen(QColor("#00FFFF"), 2, Qt.DashLine)
                    painter.setPen(pen_preview)
                    painter.drawLine(self._ruler_start, self._hover_pos)
                    dx = abs(self._hover_pos.x() - self._ruler_start.x())
                    dy = abs(self._hover_pos.y() - self._ruler_start.y())
                    ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
                    bpm = 60000 / ms if ms > 0 else 0
                    dy_mv = amplitude_mv_from_pixels(dy, max(1, h), rng, ADC_TO_MV)
                    painter.setPen(QPen(QColor("#00FFFF")))
                    painter.drawText(self._hover_pos.x(), max(12, self._hover_pos.y() - 6), ruler_label(ms, dy_mv, bpm))
        elif self._mode == TOOL_CALIPER:

            if self._caliper_line1 is not None:
                ppen = QPen(QColor("#FFFF00"), 1)
                painter.setPen(ppen)
                
                painter.drawLine(self._caliper_line1, 0, self._caliper_line1, h)
                
                
                if self._caliper_line2 is not None:
                    painter.drawLine(self._caliper_line2, 0, self._caliper_line2, h)
                    dx = abs(self._caliper_line2 - self._caliper_line1)
                    ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
                    painter.drawText(min(self._caliper_line1, self._caliper_line2) + dx//2, 12, caliper_label(ms))
                elif self._hover_pos is not None:
                    
                    pen_preview = QPen(QColor("#FFFF00"), 1, Qt.DashLine)
                    painter.setPen(pen_preview)
                    hover_x = self._hover_pos.x()
                    painter.drawLine(hover_x, 0, hover_x, h)
                    dx = abs(hover_x - self._caliper_line1)
                    ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
                    painter.drawText(min(self._caliper_line1, hover_x) + dx//2, 12, caliper_label(ms))
        elif self._mode == TOOL_MAGNIFY:
            host = self._find_magnifier_host()
            if host is not None and hasattr(host, "_magnifier_overlay"):
                painter.end()
                return
            focus_pos = self._magnify_pos if self._magnify_locked else self._hover_pos
            if focus_pos is None:
                painter.end()
                return
            hover_x = max(0, min(focus_pos.x(), w - 1))
            hover_y = max(0, min(focus_pos.y(), h - 1))
            src_center = self._x_to_index(hover_x, w, len(d))
            span = max(12, int(len(d) / max(2.0, self._speed / 12.5) / max(2, getattr(self.parent(), "_curr_length_idx", 1) + 2)))
            half = max(8, int(span / max(2, self._gain * 1.5)))
            i0 = max(0, src_center - half)
            i1 = min(len(d), src_center + half)
            sub = d[i0:i1]

            panel_w = min(320, max(220, int(w * 0.34)))
            panel_h = min(180, max(120, int(h * 0.72)))
            panel_x = min(w - panel_w - 10, hover_x + 24)
            panel_y = max(8, hover_y - panel_h - 18)
            if panel_x < 8:
                panel_x = 8
            if panel_y < 8:
                panel_y = min(h - panel_h - 8, hover_y + 18)

            panel_rect = QRect(panel_x, panel_y, panel_w, panel_h)
            inner = panel_rect.adjusted(10, 10, -10, -10)

            painter.setBrush(QColor(8, 8, 8, 235))
            painter.setPen(QPen(QColor(COL_YELLOW), 3))
            painter.drawRoundedRect(panel_rect, 12, 12)

            painter.setPen(QPen(QColor(COL_GRID_MINOR), 1))
            for frac in (0.25, 0.5, 0.75):
                gx = int(inner.left() + inner.width() * frac)
                gy = int(inner.top() + inner.height() * frac)
                painter.drawLine(gx, inner.top(), gx, inner.bottom())
                painter.drawLine(inner.left(), gy, inner.right(), gy)

            if len(sub) > 1:
                sub_min = float(np.min(sub))
                sub_max = float(np.max(sub))
                
                # Symmetrically frame the magnifier around the local median baseline
                # This prevents asymmetrical waves from pushing the baseline to the bottom/top edge
                base_val = float(np.median(sub))
                max_dev = max(abs(sub_max - base_val), abs(sub_min - base_val))
                pad = max(20.0, max_dev * 0.35)
                view_min = base_val - max_dev - pad
                view_max = base_val + max_dev + pad
                view_rng = max(1.0, view_max - view_min)

                path_pen = QPen(QColor(self._color))
                path_pen.setWidthF(2.0)
                painter.setPen(path_pen)
                x_scale_sub = inner.width() / max(1, len(sub) - 1)
                prev = None
                for i in range(len(sub)):
                    xx = inner.left() + i * x_scale_sub
                    yy = inner.bottom() - ((sub[i] - view_min) / view_rng) * inner.height()
                    if prev is not None:
                        painter.drawLine(QPointF(prev[0], prev[1]), QPointF(xx, yy))
                    prev = (xx, yy)

                focus_x = int(inner.left() + ((src_center - i0) / max(1, len(sub) - 1)) * inner.width())
                focus_y = int(inner.bottom() - ((d[src_center] - view_min) / view_rng) * inner.height())
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawLine(focus_x, inner.top(), focus_x, inner.bottom())
                painter.drawLine(inner.left(), focus_y, inner.right(), focus_y)

            painter.setPen(QPen(QColor(COL_WHITE)))
            painter.drawText(
                panel_rect.left() + 10,
                panel_rect.bottom() - 10,
                f"{getattr(self.parent(), '_curr_gain_idx', 1) + 2}x {'locked' if self._magnify_locked else 'hover'}"
            )

# 6c. MAGNIFIER OVERLAY
class MagnifierOverlay(QWidget):
    """Shared magnifier popup for the replay panel so zoom is never clipped by strip bounds."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._visible = False
        self._panel_rect = QRect()
        self._inner_rect = QRect()
        self._data = None
        self._focus_idx = 0
        self._focus_pos = QPoint(0, 0)
        self._gain = 1.0
        self._speed = 25.0
        self._fs = 500.0
        self._lead_name = ""
        self._source_widget = None
        self.hide()

    def set_focus(self, source_widget, payload: dict, focus_pos: QPoint):
        if payload is None or payload.get("data") is None:
            self.hide()
            return
        self._source_widget = source_widget
        self._data = np.asarray(payload.get("data"), dtype=float)
        self._gain = float(payload.get("gain", 1.0))
        self._speed = float(payload.get("speed", 25.0))
        self._fs = float(payload.get("fs", 500.0))
        self._lead_name = str(payload.get("lead_name", ""))
        self._focus_pos = QPoint(focus_pos)
        if self.parentWidget() is None:
            self.hide()
            return
        host = self.parentWidget()
        self.setGeometry(host.rect())
        self.raise_()
        self._visible = True
        self.show()
        self.update()

    def clear_focus(self, source_widget=None):
        if source_widget is not None and self._source_widget is not None and source_widget is not self._source_widget:
            return
        self._visible = False
        self._data = None
        self._source_widget = None
        self.hide()
        self.update()

    def paintEvent(self, event):
        if not self._visible or self._data is None or self._source_widget is None:
            return
        d = np.asarray(self._data, dtype=float)
        if d.size < 2:
            return

        host = self.parentWidget()
        if host is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Convert the focus point from source-widget coordinates into overlay coordinates.
        try:
            focus_pt = self._source_widget.mapTo(host, self._focus_pos)
        except Exception:
            focus_pt = self._focus_pos
        hover_x = max(0, min(int(focus_pt.x()), max(0, host.width() - 1)))
        hover_y = max(0, min(int(focus_pt.y()), max(0, host.height() - 1)))

        w, h = host.width(), host.height()
        source_w = max(1, int(self._source_widget.width()))
        local_x = max(0, min(int(self._focus_pos.x()), source_w - 1))
        src_center = int(round((local_x / float(max(1, source_w - 1))) * (len(d) - 1)))
        span = max(12, int(len(d) / max(2.0, self._speed / 12.5) / 2.0))
        half = max(8, int(span / max(2, self._gain * 1.5)))
        i0 = max(0, src_center - half)
        i1 = min(len(d), src_center + half)
        sub = d[i0:i1]
        if len(sub) < 2:
            painter.end()
            return

        panel_w = min(360, max(240, int(w * 0.34)))
        panel_h = min(220, max(140, int(h * 0.28)))
        panel_x = hover_x + 24
        panel_y = hover_y - panel_h - 18
        if panel_x + panel_w > w - 8:
            panel_x = hover_x - panel_w - 24
        if panel_x < 8:
            panel_x = 8
        if panel_y < 8:
            panel_y = hover_y + 18
        if panel_y + panel_h > h - 8:
            panel_y = max(8, h - panel_h - 8)
        self._panel_rect = QRect(panel_x, panel_y, panel_w, panel_h)
        self._inner_rect = self._panel_rect.adjusted(12, 12, -12, -12)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 220))
        painter.drawRoundedRect(self._panel_rect, 12, 12)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(COL_YELLOW), 3))
        painter.drawRoundedRect(self._panel_rect, 12, 12)

        painter.setPen(QPen(QColor(COL_GRID_MINOR), 1))
        for frac in (0.25, 0.5, 0.75):
            gx = int(self._inner_rect.left() + self._inner_rect.width() * frac)
            gy = int(self._inner_rect.top() + self._inner_rect.height() * frac)
            painter.drawLine(gx, self._inner_rect.top(), gx, self._inner_rect.bottom())
            painter.drawLine(self._inner_rect.left(), gy, self._inner_rect.right(), gy)

        sub_min = float(np.min(sub))
        sub_max = float(np.max(sub))
        base_val = float(np.median(sub))
        max_dev = max(abs(sub_max - base_val), abs(sub_min - base_val))
        pad = max(20.0, max_dev * 0.35)
        view_min = base_val - max_dev - pad
        view_max = base_val + max_dev + pad
        view_rng = max(1.0, view_max - view_min)

        path_pen = QPen(QColor("#22FF44"))
        path_pen.setWidthF(2.0)
        painter.setPen(path_pen)
        x_scale_sub = self._inner_rect.width() / max(1, len(sub) - 1)
        prev = None
        for i in range(len(sub)):
            xx = self._inner_rect.left() + i * x_scale_sub
            yy = self._inner_rect.bottom() - ((sub[i] - view_min) / view_rng) * self._inner_rect.height()
            if prev is not None:
                painter.drawLine(QPointF(prev[0], prev[1]), QPointF(xx, yy))
            prev = (xx, yy)

        focus_x = int(self._inner_rect.left() + ((src_center - i0) / max(1, len(sub) - 1)) * self._inner_rect.width())
        focus_y = int(self._inner_rect.bottom() - ((d[src_center] - view_min) / view_rng) * self._inner_rect.height())
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawLine(focus_x, self._inner_rect.top(), focus_x, self._inner_rect.bottom())
        painter.drawLine(self._inner_rect.left(), focus_y, self._inner_rect.right(), focus_y)

        painter.setPen(QPen(QColor(COL_WHITE)))
        painter.drawText(
            self._panel_rect.left() + 10,
            self._panel_rect.top() + 18,
            f"{self._lead_name or 'Lead'}  {self._gain:.1f}x"
        )
        painter.drawText(
            self._panel_rect.left() + 10,
            self._panel_rect.bottom() - 10,
            "click another wave to move"
        )


# 11a. HISTOGRAM CANVAS
class HistogramCanvas(QWidget):
    bar_clicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._mode = 'rri'
        self._selected_index = -1
        self._bar_rects = []
        self._bin_payloads = []
        self._bin_edges = []
        self._bins = []
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, rr_values):
        points = [{'t': float(i), 'rr': float(v)} for i, v in enumerate(rr_values or [])]
        self.set_histogram_data(points, mode='rri')

    def set_ranked_data(self, ranked_points, mode: str = 'rri'):
        self.set_histogram_data(ranked_points, mode=mode)

    def set_histogram_data(self, items, mode: str = 'rri', data_type: str = 'RR Interval'):
        self._mode = mode
        self._data_type = data_type  # Store data type for histogram building
        normalized = []
        for item in items or []:
            if isinstance(item, dict):
                normalized.append({
                    't': float(item.get('t', 0.0) or 0.0),
                    'rr': float(item.get('rr', 0.0) or 0.0),
                    'hr': float(item.get('hr', 0.0) or 0.0),
                    'rri_ratio': float(item.get('rri_ratio', 1.0) or 1.0),
                    'label': str(item.get('label', '')),
                })
            else:
                try:
                    t, rr = item
                    normalized.append({'t': float(t), 'rr': float(rr), 'hr': 0.0, 'rri_ratio': 1.0, 'label': ''})
                except Exception:
                    continue
        self._items = normalized
        self.update()

    def _palette(self):
        if self._mode == 'time':
            return QColor('#33D6FF'), QColor('#5AA7FF')
        if self._mode == 'prematurity':
            return QColor('#FF8A1E'), QColor('#FFB347')
        if self._mode == 'similarity':
            return QColor('#8A6CFF'), QColor('#B197FC')
        return QColor('#2F80ED'), QColor('#66A3FF')

    def _build_histogram(self):
        # Extract values based on data type
        data_type = getattr(self, '_data_type', 'RR Interval')
        
        if data_type == 'Heart Rate':
            values = [float(item['hr']) for item in self._items if float(item.get('hr', 0.0) or 0.0) > 0]
        elif data_type == 'RRI Ratio':
            values = [float(item['rri_ratio']) for item in self._items if 0.1 < float(item.get('rri_ratio', 1.0) or 1.0) < 3.0]
        else:
            # Default: RR Interval
            values = [float(item['rr']) for item in self._items if float(item.get('rr', 0.0) or 0.0) > 0]
        
        if not values:
            self._bins = []
            self._bin_edges = []
            self._bin_payloads = []
            self._min_value = 0.0
            self._max_value = 0.0
            return np.array([]), np.array([])
        arr = np.asarray(values, dtype=float)
        
        # Store min/max for later use in paintEvent
        self._min_value = float(np.min(arr))
        self._max_value = float(np.max(arr))
        
        if arr.size < 2:
            self._bins = [(arr[0], arr[0] + 1.0, 1)]
            self._bin_edges = [arr[0], arr[0] + 1.0]
            self._bin_payloads = [list(self._items)]
            return np.array([1]), np.array(self._bin_edges)
        bins = int(min(36, max(12, round(math.sqrt(arr.size) * 2))))
        hist, edges = np.histogram(arr, bins=bins)
        bin_ids = np.clip(np.digitize(arr, edges, right=False) - 1, 0, bins - 1)
        payloads = [[] for _ in range(bins)]
        for item, bidx in zip(self._items, bin_ids):
            payloads[int(bidx)].append(item)
        self._bins = list(hist)
        self._bin_edges = list(edges)
        self._bin_payloads = payloads
        return hist, edges

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            clicked_bar = False
            for idx, rect in enumerate(self._bar_rects):
                if rect.contains(event.pos()) and idx < len(self._bin_payloads):
                    payload = self._payload_for_bin(idx)
                    self._selected_index = idx
                    self.bar_clicked.emit(payload)
                    clicked_bar = True
                    self.update()
                    break
            if not clicked_bar:
                self._selected_index = -1
                # Emit empty dict instead of None to match signal type
                self.bar_clicked.emit({})
                self.update()
        super().mousePressEvent(event)

    def _payload_for_bin(self, idx: int) -> dict:
        points = self._bin_payloads[idx] if idx < len(self._bin_payloads) else []
        if not points:
            left = self._bin_edges[idx] if idx < len(self._bin_edges) else 0.0
            right = self._bin_edges[idx + 1] if idx + 1 < len(self._bin_edges) else left
            return {'index': idx, 'range': (left, right), 'count': 0, 'times': [], 'center_rr': (left + right) / 2.0, 'center_t': 0.0}
        rr_vals = [float(p['rr']) for p in points]
        times = [float(p['t']) for p in points]
        left = self._bin_edges[idx] if idx < len(self._bin_edges) else min(rr_vals)
        right = self._bin_edges[idx + 1] if idx + 1 < len(self._bin_edges) else max(rr_vals)
        return {
            'index': idx,
            'range': (float(left), float(right)),
            'count': len(points),
            'times': times,
            'center_rr': float(np.median(rr_vals)),
            'center_t': float(np.median(times)),
        }

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()

        left, top, right, bottom = 46, 22, 18, 28
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)

        hist, edges = self._build_histogram()
        if hist.size == 0:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, 'No data')
            painter.end()
            return

        counts = hist.astype(int).tolist()
        max_count = max(counts) if counts else 1
        if max_count <= 0:
            max_count = 1

        # Use stored min/max values from _build_histogram
        min_rr = getattr(self, '_min_value', 0.0)
        max_rr = getattr(self, '_max_value', 0.0)

        painter.setPen(QPen(QColor('#22324B'), 1))
        for frac in (0.25, 0.5, 0.75):
            y = int(top + frac * plot_h)
            painter.drawLine(left, y, left + plot_w, y)
        for frac in (0.25, 0.5, 0.75):
            x = int(left + frac * plot_w)
            painter.drawLine(x, top, x, top + plot_h)

        x_label_left = 'low RR'
        x_label_right = 'high RR'
        if self._mode == 'time':
            x_label_left = 'early'
            x_label_right = 'late'
        elif self._mode == 'prematurity':
            x_label_left = 'less premature'
            x_label_right = 'more premature'
        elif self._mode == 'similarity':
            x_label_left = 'less similar'
            x_label_right = 'more similar'

        palette, highlight = self._palette()
        self._bar_rects = []
        bin_count = len(counts)
        gap = 2
        bar_w = max(2, int((plot_w - gap * (bin_count - 1)) / max(1, bin_count)))
        x0 = left
        for idx, count in enumerate(counts):
            bar_h = int((count / max_count) * max(10, plot_h - 8))
            x = int(x0 + idx * (bar_w + gap))
            y = int(top + plot_h - bar_h)
            rect = QRect(x, y, bar_w, bar_h)
            self._bar_rects.append(rect)
            selected = idx == self._selected_index
            color = highlight if selected else palette
            alpha = 255 if selected else 180
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), alpha)))
            painter.setPen(QPen(QColor(color.lighter(130)), 1))
            painter.drawRect(rect)
            if selected:
                painter.setPen(QPen(QColor('#F5D76E'), 2))
                painter.drawRect(rect.adjusted(0, 0, -1, -1))
            if count > 0:
                painter.setPen(QPen(QColor(COL_WHITE)))
                painter.drawText(rect.adjusted(0, -18, 0, -2), Qt.AlignCenter, str(count))

        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.drawText(left, 14, x_label_left)
        painter.drawText(w - 118, 14, x_label_right)
        
        # Get data type and display appropriate labels
        data_type = getattr(self, '_data_type', 'RR Interval')
        
        if data_type == 'Heart Rate':
            painter.drawText(left, h - 5, f'{min_rr:.0f} bpm')
            painter.drawText(w - 70, h - 5, f'{max_rr:.0f} bpm')
            painter.setPen(QPen(QColor(UI_MUTED)))
            painter.drawText(left + 110, 14, f'Heart Rate Range {min_rr:.0f}-{max_rr:.0f} bpm')
        elif data_type == 'RRI Ratio':
            painter.drawText(left, h - 5, f'{min_rr:.2f}')
            painter.drawText(w - 70, h - 5, f'{max_rr:.2f}')
            painter.setPen(QPen(QColor(UI_MUTED)))
            painter.drawText(left + 110, 14, f'RRI Ratio Range {min_rr:.2f}-{max_rr:.2f}')
        else:
            # Default: RR Interval
            painter.drawText(left, h - 5, f'{min_rr:.0f} ms')
            painter.drawText(w - 70, h - 5, f'{max_rr:.0f} ms')
            painter.setPen(QPen(QColor(UI_MUTED)))
            painter.drawText(left + 110, 14, f'RR Interval Range {min_rr:.0f}-{max_rr:.0f} ms')


# 13a. ST CANVAS
class STCanvas(QWidget):
    def __init__(self, parent=None, height: int = 70):
        super().__init__(parent)
        self._data = []
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, vals):
        self._data = vals
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        # Zero line
        pen = QPen(QColor(COL_GREEN_DRK))
        pen.setWidth(1)
        painter.setPen(pen)
        mid = h // 2
        painter.drawLine(0, mid, w, mid)
        if not self._data:
            painter.end()
            return
        d = np.array(self._data)
        mn, mx = min(d.min(), -0.1), max(d.max(), 0.1)
        rng = max(mx - mn, 0.2)
        pen = QPen(QColor(COL_GREEN))
        pen.setWidth(2)
        painter.setPen(pen)
        n = len(d)
        x_scale = w / max(n - 1, 1)
        for i in range(1, n):
            x1 = int((i-1) * x_scale)
            y1 = int(h - 5 - (d[i-1] - mn) / rng * (h - 10))
            x2 = int(i * x_scale)
            y2 = int(h - 5 - (d[i] - mn) / rng * (h - 10))
            painter.drawLine(x1, y1, x2, y2)
        # mV label
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        if len(d) > 0:
            painter.drawText(w - 70, 14, f"{d[min(len(d)//2,len(d)-1)]:.3f}mV")


# 13b. ST/T MARKER CANVAS
class STTMarkerCanvas(QWidget):
    markerMoved = pyqtSignal(str, int)
    def __init__(self, parent=None, height: int = 200):
        super().__init__(parent)
        self._data = np.zeros((1, 100))
        self._fs = 500
        self._i_pos = 0
        self._j_pos = 0
        self._k_pos = 0
        self._t_pos = 0
        self._dragging_marker = None
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        
    def set_data(self, data, fs):
        self._data = data
        self._fs = fs
        self.update()
        
    def set_marker_positions(self, i_pos, j_pos, k_pos, t_pos):
        self._i_pos = i_pos
        self._j_pos = j_pos
        self._k_pos = k_pos
        self._t_pos = t_pos
        self.update()
        
    def mousePressEvent(self, event):
        event.accept()
        x = event.x()
        w = self.width()
        n_samples = self._data.shape[1] if len(self._data.shape) >1 else len(self._data)
        sample_x = int((x / w) * n_samples)
        # Check which marker is near
        markers = [('I', self._i_pos), ('J', self._j_pos), ('K', self._k_pos), ('T', self._t_pos)]
        for name, pos in markers:
            pos_x = (pos / n_samples) * w
            if abs(x - pos_x) < 10:
                self._dragging_marker = name
                break
                
    def mouseMoveEvent(self, event):
        if self._dragging_marker is None:
            return
        x = event.x()
        w = self.width()
        n_samples = self._data.shape[1] if len(self._data.shape) >1 else len(self._data)
        new_pos = int((x / w) * n_samples)
        new_pos = max(0, min(n_samples - 1, new_pos))
        if self._dragging_marker == 'I':
            self._i_pos = new_pos
        elif self._dragging_marker == 'J':
            self._j_pos = new_pos
        elif self._dragging_marker == 'K':
            self._k_pos = new_pos
        elif self._dragging_marker == 'T':
            self._t_pos = new_pos
        self.markerMoved.emit(self._dragging_marker, new_pos)
        self.update()
        
    def mouseReleaseEvent(self, event):
        self._dragging_marker = None
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        
        n_leads = self._data.shape[0] if len(self._data.shape) >1 else 1
        lead_h = h // n_leads
        n_samples = self._data.shape[1] if len(self._data.shape) >1 else len(self._data)
        x_scale = w / max(n_samples - 1, 1) if n_samples > 1 else w
        
        colors = [
            "#00FF00",
            "#0099FF",
            "#FF9900",
            "#FF0000",
            "#9900FF",
            "#00FFFF",
            "#FFFF00",
            "#FF00FF",
            "#00FF99",
            "#99FF00",
            "#FF99FF",
            "#99FFFF"
        ]
        
        for i in range(n_leads):
            lead_data = self._data[i] if n_leads > 1 else self._data
            y0 = i * lead_h
            # Normalize and draw
            if len(lead_data) > 0:
                mn = np.min(lead_data)
                mx = np.max(lead_data)
                rng = mx - mn if (mx - mn) > 0 else 1
                y_offset = y0 + lead_h // 2
                y_scale = (lead_h - 20) / rng * 0.5
                
                # Draw grid lines
                grid_pen = QPen(QColor(COL_GREEN_DRK))
                grid_pen.setWidthF(0.5)
                painter.setPen(grid_pen)
                for grid_y in [y0 + lead_h//4, y0 + lead_h//2, y0 + 3*lead_h//4]:
                    painter.drawLine(0, grid_y, w, grid_y)
                
                # Draw ECG
                pen = QPen(QColor(colors[i % len(colors)]))
                pen.setWidth(1)
                painter.setPen(pen)
                for x in range(1, len(lead_data)):
                    x1 = int((x-1)*x_scale)
                    y1 = int(y_offset - (lead_data[x-1] - mn) * y_scale)
                    x2 = int(x * x_scale)
                    y2 = int(y_offset - (lead_data[x] - mn) * y_scale)
                    painter.drawLine(x1, y1, x2, y2)
        
        # Draw markers
        marker_colors = {
            'I': "#FF9900",
            'J': "#00FFFF",
            'K': "#FF00FF",
            'T': "#00FF00"
        }
        marker_labels = {
            'I': "I",
            'J': "J",
            'K': "K",
            'T': "T"
        }
        
        for name, pos in [('I', self._i_pos), ('J', self._j_pos), ('K', self._k_pos), ('T', self._t_pos)]:
            x = int(pos * x_scale)
            if 0 <= x < w:
                pen = QPen(QColor(marker_colors[name]))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, h)
                
                # Draw label
                font = QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(x + 5, 15, marker_labels[name])



