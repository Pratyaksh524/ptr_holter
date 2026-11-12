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

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, QSize, QThread, QTimer, Qt, QObject, pyqtSignal
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
class _PdfRenderThread(QThread):
    """Renders the pages of one PDF to QImages off the GUI thread."""
    page_ready = pyqtSignal(int, int, object)   # token, page index, QImage
    failed = pyqtSignal(int, str)

    def __init__(self, token: int, path: str, scale: float, parent=None):
        super().__init__(parent)
        self._token = token
        self._path = path
        self._scale = scale
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf
            except ImportError:
                self.failed.emit(self._token, "PDF preview needs PyMuPDF (pip install pymupdf).")
                return
        from PyQt5.QtGui import QImage
        try:
            doc = pymupdf.open(self._path)
            try:
                for i, page in enumerate(doc):
                    if self._cancel:
                        return
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(self._scale, self._scale), alpha=False)
                    img = QImage(pix.samples, pix.width, pix.height, pix.stride,
                                 QImage.Format_RGB888).copy()
                    self.page_ready.emit(self._token, i, img)
            finally:
                doc.close()
        except Exception as e:
            self.failed.emit(self._token, str(e))


class _ElidedLabel(QLabel):
    """One-line label that shrinks with "..." instead of setting a minimum
    width, so a narrow screen does not push the toolbar off the window.
    The full text is in the tooltip."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text):
        self._full = text or ""
        self.setToolTip(self._full)
        self._apply()

    def _apply(self):
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, max(0, self.width() - 4))
        super().setText(shown)

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()


class HolterInsightPanel(QFrame):
    """PREVIEW tab: the Comprehensive ECG Analysis PDF report of the open
    recording, page by page, exactly as "Generate Report" / PRINT would save
    it. The preview is built automatically (in the system temp folder, never
    in the recording folder) whenever the tab is shown and the recording's
    data has changed since the last build; saving stays with Generate
    Report / PRINT."""

    preview_requested = pyqtSignal()   # owner builds a preview PDF, then calls show_preview()
    generate_requested = pyqtSignal()  # kept for older wiring; not emitted

    _DESK_BG = "#1B2433"
    _FIT_MAX = 1.75
    _ZOOM_STEPS = (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pvCard")
        self.setStyleSheet(_card_style("QFrame#pvCard"))
        self._session_dir = None
        self._patient_info = {}
        self._pdf_path = None
        self._pdf_kind = None          # 'preview' or 'saved'
        self._built_key = None         # data state the shown PDF was built from
        self._building = False
        self._rebuild_after = False
        self._page_sizes = []          # (w, h) in PDF points
        self._page_labels = []
        self._zoom = 1.0
        self._fit_width = True
        self._token = 0
        self._threads = []
        self._needs_render = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Toolbar
        bar = QHBoxLayout()
        bar.setSpacing(8)
        title = QLabel("Comprehensive Report Preview")
        title.setStyleSheet(f"color:{UI_TEXT};font-size:14px;font-weight:bold;background:transparent;"
                            f"padding:4px 8px;border:none;border-left:3px solid {UI_ACCENT};")
        bar.addWidget(title)
        self._info = _ElidedLabel("")
        self._info.setStyleSheet(f"color:{UI_MUTED};font-size:11px;background:transparent;border:none;")
        bar.addWidget(self._info, 1)

        def _tool_btn(text, tip, slot):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(_style_btn())
            b.clicked.connect(slot)
            bar.addWidget(b)
            return b

        self._zoom_out_btn = _tool_btn("−", "Zoom out", lambda: self._step_zoom(-1))
        self._zoom_label = QLabel("100%")
        self._zoom_label.setMinimumWidth(48)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setStyleSheet(f"color:{UI_TEXT};font-size:12px;font-weight:700;background:transparent;border:none;")
        bar.addWidget(self._zoom_label)
        self._zoom_in_btn = _tool_btn("+", "Zoom in", lambda: self._step_zoom(1))
        self._fit_btn = _tool_btn("Fit Width", "Fit the page width to the window", self._set_fit_width)
        self._open_btn = _tool_btn("Open PDF", "Open this PDF in the system PDF viewer", self._open_external)
        self._refresh_btn = _tool_btn("Refresh Preview", "Build the preview again from the current data",
                                      lambda: self.refresh(force=True))
        layout.addLayout(bar)

        # Pages
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet(f"QScrollArea{{background:{self._DESK_BG};border:1px solid {UI_BORDER};"
                                   f"border-radius:8px;}}")
        self._desk = QWidget()
        self._desk.setStyleSheet(f"background:{self._DESK_BG};")
        self._pages_layout = QVBoxLayout(self._desk)
        self._pages_layout.setContentsMargins(24, 24, 24, 24)
        self._pages_layout.setSpacing(6)
        self._pages_layout.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self._scroll.setWidget(self._desk)
        layout.addWidget(self._scroll, 1)

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(200)
        self._resize_timer.timeout.connect(self._on_resized)

        self._show_message("Open a recording to preview its report.", busy=False)

    # -- public API ------------------------------------------------------------
    def update_text(self, patient_info: dict, summary: dict):
        """Called on every UI refresh: remember the patient details (part of
        the data a preview is built from)."""
        self._patient_info = dict(patient_info or {})

    def set_session_dir(self, session_dir, force: bool = False):
        """Cheap when unchanged (called on every UI refresh). A new recording
        drops the old preview; it is rebuilt when the tab is shown."""
        session_dir = session_dir or None
        if session_dir == self._session_dir and not force:
            return
        if session_dir != self._session_dir:
            self._session_dir = session_dir
            self._pdf_path = None
            self._pdf_kind = None
            self._built_key = None
            self._show_message("Open a recording to preview its report." if not session_dir
                               else "Preparing the report preview…", busy=bool(session_dir))
        if self.isVisible():
            self.refresh()

    def refresh(self, force: bool = False):
        """Build a new preview if the recording's data changed since the
        shown PDF was built (or always, with force)."""
        if not self._session_dir:
            return
        if not force and self._pdf_path and self._built_key == self._state_key():
            return
        if self._building:
            self._rebuild_after = True
            return
        self._building = True
        self._rebuild_after = False
        self._refresh_btn.setEnabled(False)
        if not self._pdf_path:
            self._show_message("Preparing the report preview…", busy=True)
        else:
            self._info.setText("Updating the preview with the latest data…")
        self.preview_requested.emit()

    def show_preview(self, path: str):
        """The owner's preview build finished."""
        self._finish_build()
        # Taken after the build: a first build may itself refresh a cache
        # file in the recording folder.
        self._built_key = self._state_key()
        self.load_pdf(path, kind='preview')
        if self._rebuild_after:
            self._rebuild_after = False
            QTimer.singleShot(0, self.refresh)

    def preview_failed(self, message: str):
        self._finish_build()
        self._show_message(f"Could not build the report preview:\n{message}", busy=False)

    def load_pdf(self, path: str, kind: str = 'saved'):
        """Show a PDF: a preview build, or a report Generate Report / PRINT
        has just saved (same content, so it also counts as up to date)."""
        if not path or not str(path).lower().endswith('.pdf') or not os.path.isfile(path):
            return
        if kind == 'saved':
            self._built_key = self._state_key()
        self._pdf_path = path
        self._pdf_kind = kind
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf
            except ImportError:
                self._show_message("PDF preview needs PyMuPDF (pip install pymupdf).", busy=False)
                return
        try:
            doc = pymupdf.open(path)
            self._page_sizes = [(p.rect.width, p.rect.height) for p in doc]
            doc.close()
        except Exception as e:
            self._show_message(f"Could not open the report:\n{e}", busy=False)
            return
        n = len(self._page_sizes)
        pages = f"{n} page{'s' if n != 1 else ''}"
        when = datetime.now().strftime("%H:%M:%S")
        if kind == 'saved':
            self._info.setText(f"Saved report: {os.path.basename(path)}   •   {pages}")
        else:
            self._info.setText(f"Live preview (not saved)   •   {pages}   •   built {when}"
                               f"   •   use Generate Report or Print to save")
        self._set_buttons(True)
        self._needs_render = True
        if self.isVisible():
            self._render()

    # -- internals -------------------------------------------------------------
    def _state_key(self):
        """What a preview depends on: the patient details and every data file
        of the recording (name, size, mtime) -- edits, deletions and
        re-analysis all rewrite one of them. Report images / PDFs that a
        build itself writes there are left out."""
        files = []
        sd = self._session_dir
        try:
            for f in sorted(os.listdir(sd)):
                if f.lower().endswith(('.png', '.pdf', '.txt')):
                    continue
                st = os.stat(os.path.join(sd, f))
                files.append((f, st.st_size, int(st.st_mtime)))
        except Exception:
            pass
        return (sd, repr(sorted((str(k), str(v)) for k, v in self._patient_info.items())), tuple(files))

    def _finish_build(self):
        self._building = False
        self._refresh_btn.setEnabled(True)

    def _set_buttons(self, has_pdf: bool):
        for b in (self._zoom_out_btn, self._zoom_in_btn, self._fit_btn, self._open_btn):
            b.setEnabled(has_pdf)

    def _clear_pages(self):
        self._token += 1
        for t in self._threads:
            t.cancel()
        while self._pages_layout.count():
            item = self._pages_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
        self._page_labels = []

    def _show_message(self, text: str, busy: bool):
        self._clear_pages()
        self._set_buttons(False)
        self._info.setText("")
        box = QFrame()
        box.setStyleSheet("background:transparent;border:none;")
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 80, 0, 0)
        v.setSpacing(14)
        icon = QLabel("\U0001F4C4")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size:46px;background:transparent;border:none;")
        msg = QLabel(text)
        msg.setAlignment(Qt.AlignCenter)
        msg.setWordWrap(True)
        msg.setStyleSheet(f"color:{UI_MUTED};font-size:13px;background:transparent;border:none;")
        v.addWidget(icon)
        v.addWidget(msg)
        if busy:
            bar = QProgressBar()
            bar.setRange(0, 0)
            bar.setFixedSize(260, 6)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"QProgressBar{{background:{UI_PANEL_ALT};border:none;border-radius:3px;}}"
                              f"QProgressBar::chunk{{background:{UI_ACCENT};border-radius:3px;}}")
            row = QHBoxLayout()
            row.addStretch(1)
            row.addWidget(bar)
            row.addStretch(1)
            v.addLayout(row)
        self._pages_layout.addWidget(box)

    def _fit_zoom(self):
        if not self._page_sizes:
            return 1.0
        avail = self._scroll.viewport().width() - 48 - 20
        widest = max(w for w, _ in self._page_sizes)
        # Capped so a wide monitor shows a page at a readable size rather
        # than a poster; zoom in for more.
        return max(0.3, min(self._FIT_MAX, avail / float(widest)))

    def _render(self):
        if not self._pdf_path or not self._page_sizes:
            return
        self._needs_render = False
        if self._fit_width:
            self._zoom = self._fit_zoom()
        self._zoom_label.setText(f"{int(round(self._zoom * 100))}%")
        self._clear_pages()
        token = self._token
        dpr = float(self.devicePixelRatioF()) if hasattr(self, 'devicePixelRatioF') else 1.0
        n = len(self._page_sizes)
        for i, (w, h) in enumerate(self._page_sizes):
            page = QLabel()
            page.setFixedSize(int(round(w * self._zoom)), int(round(h * self._zoom)))
            page.setAlignment(Qt.AlignCenter)
            page.setText("Loading page…")
            page.setStyleSheet("background:#FFFFFF;color:#8A94A6;border:1px solid #0E1520;")
            try:
                from PyQt5.QtWidgets import QGraphicsDropShadowEffect
                sh = QGraphicsDropShadowEffect(page)
                sh.setBlurRadius(18)
                sh.setOffset(0, 4)
                sh.setColor(QColor(0, 0, 0, 160))
                page.setGraphicsEffect(sh)
            except Exception:
                pass
            cap = QLabel(f"Page {i + 1} of {n}")
            cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet(f"color:{UI_MUTED};font-size:11px;background:transparent;border:none;"
                              f"padding:6px 0 18px 0;")
            self._pages_layout.addWidget(page, 0, Qt.AlignHCenter)
            self._pages_layout.addWidget(cap, 0, Qt.AlignHCenter)
            self._page_labels.append(page)
        th = _PdfRenderThread(token, self._pdf_path, self._zoom * dpr, self)
        th.page_ready.connect(lambda tok, i, img, d=dpr: self._on_page(tok, i, img, d))
        th.failed.connect(lambda tok, err: tok == self._token and self._show_message(
            f"Could not render the report:\n{err}", busy=False))
        th.finished.connect(lambda t=th: self._threads.remove(t) if t in self._threads else None)
        self._threads.append(th)
        th.start()

    def _on_page(self, token, index, img, dpr):
        if token != self._token or index >= len(self._page_labels):
            return
        pm = QPixmap.fromImage(img)
        pm.setDevicePixelRatio(dpr)
        lbl = self._page_labels[index]
        lbl.setText("")
        lbl.setPixmap(pm)

    def _step_zoom(self, direction: int):
        if not self._pdf_path:
            return
        cur = self._zoom
        steps = self._ZOOM_STEPS
        if direction > 0:
            nxt = next((z for z in steps if z > cur + 1e-3), steps[-1])
        else:
            nxt = next((z for z in reversed(steps) if z < cur - 1e-3), steps[0])
        self._fit_width = False
        self._zoom = nxt
        self._render()

    def _set_fit_width(self):
        self._fit_width = True
        self._render()

    def _open_external(self):
        if self._pdf_path and os.path.isfile(self._pdf_path):
            from PyQt5.QtGui import QDesktopServices
            from PyQt5.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._pdf_path))

    def _on_resized(self):
        if self._fit_width and self._pdf_path and self.isVisible():
            if abs(self._fit_zoom() - self._zoom) > 0.01:
                self._render()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.refresh)
        if self._needs_render:
            QTimer.singleShot(0, self._render)
