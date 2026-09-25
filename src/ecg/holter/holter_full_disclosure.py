"""
ecg/holter/holter_full_disclosure.py
=====================================
Full Disclosure ECG viewer - standalone dialog module.

MANUAL MARKING PRIORITY SYSTEM:
==============================
When user manually marks events in full disclosure and then generates a report:
  1. Manual marks (is_manual=True) always appear in the report
  2. Auto-detected marks within 150ms of a manual mark are SUPPRESSED
  3. This prevents duplicate/conflicting marks from confusing the clinician
  4. Manual marks take absolute priority over auto-detection

Implementation:
  - Manual beats are stored in manual_beats.json with is_manual flag
  - Manual segments are stored in manual_segments.json
  - When displaying events or generating reports, use get_events_with_manual_priority()
  - This method filters out overlapping auto marks based on temporal proximity

Classes:
  - FullDisclosureOverlay         : Transparent selection-box overlay drawn over the ECG canvas
  - HolterFullDisclosureDialog    : 12-lead scrollable Full Disclosure ECG viewer dialog
"""

import os
import json
import copy
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSpinBox, QScrollBar, QSizePolicy, QApplication, QTabBar,
)
from PyQt5.QtCore import Qt, QEvent, QRect, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor

try:
    from .theme import (COL_BLACK, COL_DARK, COL_GREEN, COL_GREEN_DRK, COL_WHITE, COL_GRID_MAJOR,
                        TOOL_RULER, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_SELECT)
    from .holter_ui import ECGStripCanvas, MagnifierOverlay
except ImportError:
    from ecg.holter.theme import (COL_BLACK, COL_DARK, COL_GREEN, COL_GREEN_DRK, COL_WHITE, COL_GRID_MAJOR,
                                   TOOL_RULER, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_SELECT)
    from ecg.holter.holter_ui import ECGStripCanvas, MagnifierOverlay

try:
    from utils.settings_manager import SettingsManager
except ImportError:
    try:
        from ecg.utils.settings_manager import SettingsManager
    except ImportError:
        SettingsManager = None

from PyQt5.QtCore import pyqtSignal

# TODO: Expanded view / Overlay double click handler - Disabled for now
# def _on_overlay_double_clicked(self, start_sec, duration):
#     """Open expanded view for the selected time range."""
#     # TODO: Expanded view disabled for now
#     # dialog = ExpandedViewDialog(self._engine, self._current_start + start_sec, duration, self)
#     # dialog.exec_()
#     pass

# ============================================================================
# SELECTION BOX OVERLAY - COMMENTED OUT (Not needed for now)
# ============================================================================
# class FullDisclosureOverlay(QWidget):
#     """Transparent overlay to draw a fixed-width square selection box over the channels."""
#     
#     double_clicked = pyqtSignal(float, float)  # Emits start_sec, duration_sec
# 
#     def __init__(self, parent=None):
#         super().__init__(parent)
#         self._mouse_enabled = True
#         self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
#         self._selection_center_x = 0.0
#         self._selection_center_y = 0.0
#         self._strip_length_sec = 3.0
#         self._pixels_per_sec = 25.0
#         self._is_dragging = False
#         self.on_selection_made = None
#         
#     def set_mouse_enabled(self, enabled):
#         self._mouse_enabled = enabled
#         self.setAttribute(Qt.WA_TransparentForMouseEvents, not enabled)
#         self.update()
# 
#     def set_pixels_per_sec(self, pps):
#         self._pixels_per_sec = max(1.0, pps)
#         if self._selection_center_x == 0.0:
#             width = self._strip_length_sec * self._pixels_per_sec
#             self._selection_center_x = 48.0 + width / 2.0
#             self._selection_center_y = width / 2.0
#         self.update()
# 
#     def set_strip_length(self, length_sec):
#         self._strip_length_sec = length_sec
#         self.update()
# 
#     def mousePressEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self._is_dragging = True
#             self._selection_center_x = event.pos().x()
#             self._selection_center_y = event.pos().y()
#             self.update()
#             self._emit_selection()
# 
#     def mouseMoveEvent(self, event):
#         if self._is_dragging:
#             self._selection_center_x = event.pos().x()
#             self._selection_center_y = event.pos().y()
#             self.update()
# 
#     def mouseReleaseEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             self._is_dragging = False
#             self._emit_selection()
#             
#     def mouseDoubleClickEvent(self, event):
#         if event.button() == Qt.LeftButton:
#             width = self._strip_length_sec * self._pixels_per_sec
#             start = self._selection_center_x - width / 2.0
#             start = max(48, min(start, self.width() - width))
#             start_sec = max(0.0, (start - 48) / self._pixels_per_sec)
#             self.double_clicked.emit(start_sec, self._strip_length_sec)
# 
#     def _emit_selection(self):
#         if self.on_selection_made and self._selection_center_x is not None:
#             width = self._strip_length_sec * self._pixels_per_sec
#             start = self._selection_center_x - width / 2.0
#             start = max(48, min(start, self.width() - width))
#             start_sec = max(0.0, (start - 48) / self._pixels_per_sec)
#             self.on_selection_made(start_sec, self._strip_length_sec)
# 
#     def paintEvent(self, event):
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.Antialiasing)
#         width = self._strip_length_sec * self._pixels_per_sec
#         height = 186.0
#         start_x = self._selection_center_x - width / 2.0
#         start_y = self._selection_center_y - height / 2.0
#         start_x = max(48, min(start_x, self.width() - width))
#         start_y = max(0, min(start_y, self.height() - height))
#         rect = QRect(int(start_x), int(start_y), int(width), int(height))
#         painter.setBrush(QColor(0, 120, 215, 80))
#         painter.setPen(QPen(QColor(0, 120, 215, 180), 2))
#         painter.drawRect(rect)
# ============================================================================


class VerticalLineOverlay(QWidget):
    """Transparent overlay to draw vertical lines on user click/drag across all leads."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # Let clicks pass through
        self._line_positions = []  # List of X positions for multiple vertical lines
        self.setStyleSheet("background: transparent;")
        
    def set_line_position(self, x: int):
        """Set a single X position for the vertical line (backward compatibility)."""
        self._line_positions = [x] if x is not None else []
        self.update()
    
    def set_line_positions(self, positions: list):
        """Set multiple X positions for vertical lines (for drag selection)."""
        self._line_positions = positions if positions else []
        self.update()
        
    def clear_line(self):
        """Clear all vertical lines."""
        self._line_positions = []
        self.update()
        
    def paintEvent(self, event):
        """Draw vertical yellow lines at all set positions."""
        if not self._line_positions:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw continuous vertical yellow lines starting from y=30 (below label text)
        # This prevents the line from overlapping with the label and time text
        painter.setPen(QPen(QColor("#FFFF00"), 2))
        
        for line_x in self._line_positions:
            if line_x is not None:
                painter.drawLine(line_x, 30, line_x, self.height())


class SegmentOverlay(QWidget):
    """
    Transparent overlay that draws segment selection rectangles across all leads.
    Used in 'Segment Selection' mode — shows a live drag preview and stores labeled segments.
    Mouse events pass through so canvases beneath still receive them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background: transparent;")

        self._drag_start_x = None
        self._drag_end_x   = None
        self._segments     = []   # list of {start_x, end_x, start_sec, end_sec, label, color, start_time_str, end_time_str}

    def set_drag(self, start_x, end_x):
        self._drag_start_x = start_x
        self._drag_end_x   = end_x
        self.update()

    def clear_drag(self):
        self._drag_start_x = None
        self._drag_end_x   = None
        self.update()

    def add_segment(self, start_x, end_x, start_sec, end_sec, label, color, start_time_str='', end_time_str=''):
        self._segments.append({
            'start_x': start_x, 'end_x': end_x,
            'start_sec': start_sec, 'end_sec': end_sec,
            'label': label, 'color': color, 'start_time_str': start_time_str, 'end_time_str': end_time_str
        })
        self.update()

    def update_segment_pixels(self, compute_x_fn):
        for seg in self._segments:
            seg['start_x'] = compute_x_fn(seg['start_sec'])
            seg['end_x']   = compute_x_fn(seg['end_sec'])
        self.update()

    def clear_segments(self):
        self._segments.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        h = self.height()

        # 1. Live drag preview — white translucent fill + dashed border
        if self._drag_start_x is not None and self._drag_end_x is not None:
            x1 = min(self._drag_start_x, self._drag_end_x)
            x2 = max(self._drag_start_x, self._drag_end_x)
            if x2 > x1:
                painter.fillRect(x1, 0, x2 - x1, h, QColor(255, 255, 255, 35))
                painter.setPen(QPen(QColor("#FFFFFF"), 1, Qt.DashLine))
                painter.drawRect(x1, 0, x2 - x1, h - 1)

        # 2. Finalized labeled segments
        for seg in self._segments:
            sx = min(seg['start_x'], seg['end_x'])
            ex = max(seg['start_x'], seg['end_x'])
            if ex <= sx:
                continue
            try:
                c = QColor(seg['color'])
            except Exception:
                c = QColor("#FFFF00")
            painter.fillRect(sx, 0, ex - sx, h, QColor(c.red(), c.green(), c.blue(), 55))
            painter.setPen(QPen(c, 1))
            painter.drawRect(sx, 0, ex - sx, h - 1)

            mid_x = (sx + ex) // 2
            lbl = seg.get('label', '')
            start_sec = seg.get('start_sec', 0)
            end_sec = seg.get('end_sec', 0)
            start_time_str = seg.get('start_time_str', '')
            end_time_str = seg.get('end_time_str', '')

            lbl_font = painter.font()
            lbl_font.setBold(True)
            lbl_font.setPixelSize(11)
            painter.setFont(lbl_font)
            lbl_w = painter.fontMetrics().horizontalAdvance(lbl)

            t_font = painter.font()
            t_font.setPixelSize(9)
            t_font.setBold(True)
            painter.setFont(t_font)
            
            # Calculate duration in seconds and format as HH:MM:SS
            duration_sec = end_sec - start_sec
            dur_h = int(duration_sec // 3600)
            m = int((duration_sec % 3600) // 60)
            s = int(duration_sec % 60)
            time_str = f"{dur_h:02d}:{m:02d}:{s:02d}"
            
            time_w = painter.fontMetrics().horizontalAdvance(time_str) if time_str else 0

            spacer_w = 4 if time_str else 0
            total_w = lbl_w + spacer_w + time_w
            start_text_x = mid_x - (total_w // 2)

            # Draw label (y = 20)
            painter.setFont(lbl_font)
            painter.setPen(QPen(c))
            painter.drawText(start_text_x, 20, lbl)

            # Draw duration string (y = 20, right of label, in white)
            if time_str:
                painter.setFont(t_font)
                painter.setPen(QPen(QColor("#FFFFFF")))
                painter.drawText(start_text_x + lbl_w + spacer_w, 20, time_str)

            # Draw start time on left edge of segment (y = 20, in white)
            if start_time_str:
                painter.setFont(t_font)
                painter.setPen(QPen(QColor("#FFFFFF")))
                painter.drawText(sx + 4, 20, start_time_str)

            # Draw end time on right edge of segment (y = 20, in white)
            if end_time_str:
                end_time_w = painter.fontMetrics().horizontalAdvance(end_time_str)
                painter.setFont(t_font)
                painter.setPen(QPen(QColor("#FFFFFF")))
                painter.drawText(ex - end_time_w - 4, 20, end_time_str)



class HolterFullDisclosureDialog(QDialog):

    """Full Disclosure view: 12-lead scrollable ECG viewer."""

    _GAIN_STEPS  = [(0.5, "5mm/mV"), (1.0, "10mm/mV"), (2.0, "20mm/mV")]
    _SPEED_STEPS = [12.5, 25.0, 50.0]
    _BASE_WIN_SEC = 10.0

    def __init__(self, replay_engine, parent=None):
        super().__init__(parent)
        self._engine      = replay_engine
        self._reader      = replay_engine._reader
        self._paper_speed = 25.0
        self._gain        = 1.0
        self._gain_label  = "10mm/mV"
        self._strip_length = 2.0
        self._current_start = 0.0
        self._window_sec  = self._BASE_WIN_SEC
        self._selected_duration = None
        self._active_tool = TOOL_SELECT
        self._active_tool_btn = None
        
        # Drag selection state (parallel mode)
        self._drag_start_x = None
        self._drag_current_x = None
        self._is_dragging = False
        self._drag_start_timestamp = None
        
        # Parallel multi mode: two vertical lines
        self._multi_line1_x = None  # First line (fixed on press)
        self._multi_line2_x = None  # Second line (follows drag, fixed on release)
        
        # Selection mode: 'parallel_single' | 'parallel_multi' | 'segment'
        # CRITICAL FIX: Restore from saved settings instead of always defaulting to 'parallel_single'
        if SettingsManager is not None:
            try:
                settings = SettingsManager()
                self._selection_mode = settings.get_setting('holter_selection_mode', 'parallel_single')
            except Exception as e:
                print(f"[Full Disclosure] Could not restore selection mode from settings: {e}")
                self._selection_mode = 'parallel_single'
        else:
            self._selection_mode = 'parallel_single'
        
        # Pending manual beats to restore after canvases are created
        self._pending_manual_beats = None
        
        # Segment selection state
        self._seg_start_x   = None   # pixel X where segment drag started
        self._seg_end_x     = None   # pixel X where drag currently ends
        self._seg_dragging  = False
        self._pending_segment = None  # {start_sec, end_sec, start_x, end_x} awaiting label
        self._segment_annotations = []  # all labeled segments (persists across scrolls)
        
        self._detected_r_peaks = []

        # Undo stack: list of state snapshots pushed before each mutating
        # annotation action (beat label/delete, segment label/delete).
        self._undo_stack = []
        self._UNDO_STACK_LIMIT = 50

        self.setWindowTitle("Full Disclosure ECG")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.setWindowState(Qt.WindowMaximized)

        screen = QApplication.primaryScreen()
        if screen:
            self.resize(screen.availableGeometry().size())

        self.setStyleSheet(f"QDialog {{ background: {COL_BLACK}; }}")
        self._build_ui()
        
        # Initialize the recording time label (total recording start-end, fixed)
        self._update_recording_time_label()

        # Shared magnifier overlay (covers the whole dialog, used by TOOL_MAGNIFY)
        self._magnifier_overlay = MagnifierOverlay(self)
        self._magnifier_overlay.setGeometry(self.rect())
        self._magnifier_overlay.hide()

        # Throttling state for smooth scrollbar dragging (updates limited to max ~30 FPS / 33ms)
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._on_scroll_timer_timeout)
        self._pending_scroll_val = None
        self._last_scroll_time = 0

        # True while the user is actively holding the scrollbar handle down.
        # While scrubbing we skip the extras that don't need to update every
        # frame (arrhythmia/event label recompute, segment-overlay reposition)
        # and only redraw the waveforms, so each frame stays cheap and the
        # trace tracks the mouse instead of catching up in jumps.
        self._is_scrubbing = False
        self._last_extras_time = 0.0
        self._EXTRAS_MIN_INTERVAL = 0.15  # refresh labels/overlay at most ~6-7x/sec while scrubbing

        # Cached, timestamp-sorted beat/event index for O(log n) windowed lookups.
        # _update_canvases() used to linear-scan the FULL recording's beat list
        # (all_beats across all metrics) on every single redraw while dragging -
        # for long Holter recordings that's tens of thousands of beats scanned
        # up to ~30x/sec, which is what caused the dragging lag on 30 Sec/1 Min/
        # 2 Min views. These caches are rebuilt lazily (only when marked dirty by
        # an actual beat/event add/delete/undo) and sliced with bisect instead.
        self._beat_index_dirty = True
        self._cached_beat_ts = np.array([], dtype=np.float64)
        self._cached_beat_list = []
        self._event_index_dirty = True
        self._cached_event_ts = np.array([], dtype=np.float64)
        self._cached_event_list = []

        # Load manual_beats.json if it exists and merge into engine metrics on startup
        try:
            session_dir = os.path.dirname(self._engine.ecgh_path)
            manual_beats_path = os.path.join(session_dir, 'manual_beats.json')
            if os.path.exists(manual_beats_path):
                import json
                with open(manual_beats_path, 'r') as f:
                    m_beats = json.load(f)
                if hasattr(self._engine, '_metrics') and self._engine._metrics:
                    for mb in m_beats:
                        ts = float(mb.get('timestamp', 0.0))
                        lbl = str(mb.get('label', 'N'))
                        is_man = mb.get('is_manual', False)
                        color = mb.get('color', '#FFFF00')
                        batch_id = mb.get('batch_id')
                        marking_mode = mb.get('marking_mode')
                        
                        found = False
                        for m in self._engine._metrics:
                            for eb in m.get('all_beats', []):
                                if abs(float(eb.get('timestamp', 0.0)) - ts) < 0.15:
                                    eb['label'] = lbl
                                    eb['is_manual'] = is_man
                                    eb['color'] = color
                                    if batch_id is not None:
                                        eb['batch_id'] = batch_id
                                    if marking_mode:
                                        eb['marking_mode'] = marking_mode
                                    found = True
                                    break
                            if found:
                                break
                        if not found and self._engine._metrics:
                            beat_dict = {
                                'timestamp': ts,
                                'label': lbl,
                                'is_manual': is_man,
                                'color': color
                            }
                            if batch_id is not None:
                                beat_dict['batch_id'] = batch_id
                            if marking_mode:
                                beat_dict['marking_mode'] = marking_mode
                            self._engine._metrics[0].setdefault('all_beats', []).append(beat_dict)
                    
                    self._beat_index_dirty = True
                print(f"[Full Disclosure] Loaded and merged {len(m_beats)} manual beats on startup.")
                
                # Also restore manual beats to canvas _beat_annotations for UI display
                # This needs to happen after canvases are created, so we'll do it in _update_canvases
                self._pending_manual_beats = m_beats
        except Exception as e:
            print(f"[Full Disclosure] Error merging manual beats on startup: {e}")
            self._pending_manual_beats = []

        # Load manual_segments.json if it exists and restore segment annotations
        try:
            session_dir = os.path.dirname(self._engine.ecgh_path)
            manual_segments_path = os.path.join(session_dir, 'manual_segments.json')
            if os.path.exists(manual_segments_path):
                with open(manual_segments_path, 'r') as f:
                    segments_data = json.load(f)
                self._segment_annotations = segments_data
                print(f"[Full Disclosure] Loaded {len(segments_data)} segment annotations on startup.")
                print(f"[Full Disclosure] Segment data: {segments_data}")
        except Exception as e:
            print(f"[Full Disclosure] Error loading segment annotations on startup: {e}")

        self._update_canvases(0.0)
        
        # CRITICAL FIX: Apply the restored selection mode to update button text and UI state
        # This ensures the button shows the correct mode after reopening the dialog
        if hasattr(self, '_selection_mode'):
            # Call _switch_selection_mode to properly initialize the UI for the restored mode
            # But we need to do this AFTER the button is created, so use a short delay
            QTimer.singleShot(50, lambda: self._switch_selection_mode(self._selection_mode))

    def showEvent(self, event):
        """Override showEvent to refresh segment overlay when dialog is shown."""
        super().showEvent(event)
        # Refresh segment overlay to ensure loaded segments are visible
        # Use longer delay to ensure canvas geometry is fully initialized
        QTimer.singleShot(200, self._refresh_segment_overlay)
        # Also raise overlay to ensure it's on top
        QTimer.singleShot(250, lambda: self._segment_overlay.raise_() if hasattr(self, '_segment_overlay') else None)

    # Per-lead row height (px) by gain multiplier. At higher gain the same
    # mV deflection paints taller, so a fixed 90px row clips QRS peaks against
    # the next row - give higher gains more headroom (paid for by the tightened
    # container/row margins above), and let it be honest about the room a low
    # gain trace actually needs.
    _GAIN_CANVAS_HEIGHT = {0.5: 84, 1.0: 92, 2.0: 108}

    def _canvas_height_for_gain(self, gain):
        return self._GAIN_CANVAS_HEIGHT.get(gain, 92)

    def _apply_canvas_height_for_gain(self):
        """Resize every lead row to match the current gain's headroom and
        force a relayout/repaint so peaks stop clipping immediately."""
        new_height = self._canvas_height_for_gain(self._gain)
        if new_height == getattr(self, '_canvas_height', None):
            return
        self._canvas_height = new_height
        for c in getattr(self, '_canvases', []):
            c.setFixedHeight(new_height)
            c.update()
        if hasattr(self, 'canvas_layout'):
            self.canvas_layout.activate()

    def _recalc_window(self):
        idx = self.time_tabs.currentIndex() if hasattr(self, 'time_tabs') else 0
        text = self.time_tabs.tabText(idx) if hasattr(self, 'time_tabs') else "Full disc"
        
        if "Full disc" in text:
            self._window_sec = self._BASE_WIN_SEC * (25.0 / self._paper_speed)
        elif "30 Sec" in text:
            self._window_sec = 30.0
        elif "1 Min" in text:
            self._window_sec = 60.0
        elif "2 Min" in text:
            self._window_sec = 120.0
        # elif "5 Min" in text: self._window_sec = 300.0
        # elif "10 Min" in text: self._window_sec = 600.0
        # elif "15 Min" in text: self._window_sec = 900.0
        else:
            self._window_sec = self._BASE_WIN_SEC * (25.0 / self._paper_speed)

    def _update_scrollbar_range(self):
        total = max(0.0, self._engine.duration_sec - self._window_sec)
        self.time_scrollbar.setRange(0, max(0, int(total * 100)))
        self.time_scrollbar.setSingleStep(100)
        self.time_scrollbar.setPageStep(int(self._window_sec * 100))

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        top_bar = QFrame()
        top_bar.setStyleSheet(f"background: {COL_DARK}; border-bottom: 1px solid {COL_GREEN_DRK};")
        top_bar.setFixedHeight(44)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 4, 14, 4)
        top_layout.setSpacing(12)

        self.lbl_time = QLabel("Time:  00:00:00")
        self.lbl_time.setStyleSheet(f"color: {COL_GREEN}; font-weight: bold; font-size: 15px;")
        top_layout.addWidget(self.lbl_time)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet(f"color: {COL_GREEN_DRK};")
        top_layout.addWidget(sep1)

        # TODO: Selection window option disabled for now
        # lbl_sl = QLabel("Selection window (s):")
        # lbl_sl.setStyleSheet("color: #a0c4e8; font-size: 13px;")
        # top_layout.addWidget(lbl_sl)
        # 
        # self.spin_strip = QSpinBox()
        # self.spin_strip.setRange(1, 60)
        # self.spin_strip.setValue(int(self._strip_length))
        # self.spin_strip.setFixedWidth(58)
        # self.spin_strip.setStyleSheet(f"""
        #     QSpinBox {{
        #         background: #0d1b2a; color: {COL_GREEN};
        #         border: 1px solid {COL_GREEN_DRK}; border-radius: 4px;
        #         padding: 3px 6px; font-size: 13px; font-weight: bold;
        #     }}
        #     QSpinBox::up-button, QSpinBox::down-button {{ width: 16px; background: #162a3a; }}
        # """)
        # self.spin_strip.valueChanged.connect(self._on_strip_length_changed)
        # top_layout.addWidget(self.spin_strip)
        # 
        # sep2 = QFrame()
        # sep2.setFrameShape(QFrame.VLine)
        # sep2.setStyleSheet(f"color: {COL_GREEN_DRK};")
        # top_layout.addWidget(sep2)

        self.time_tabs = QTabBar()
        self.time_tabs.addTab("Full disc")
        self.time_tabs.addTab("30 Sec")
        self.time_tabs.addTab("1 Min")
        self.time_tabs.addTab("2 Min")
        # self.time_tabs.addTab("5 Min")
        # self.time_tabs.addTab("10 Min")
        # self.time_tabs.addTab("15 Min")
        self.time_tabs.setStyleSheet(f"""
            QTabBar::tab {{
                background: #0d1b2a; color: #a0c4e8;
                border: 1px solid {COL_GREEN_DRK};
                padding: 4px 10px;
                border-radius: 4px;
                margin-right: 4px;
                font-size: 13px; font-weight: bold;
            }}
            QTabBar::tab:selected {{
                background: {COL_GREEN_DRK}; color: {COL_GREEN};
            }}
            QTabBar::tab:hover:!selected {{
                background: #162a3a;
            }}
        """)
        self.time_tabs.currentChanged.connect(self._on_time_tab_changed)
        top_layout.addWidget(self.time_tabs)

        top_layout.addStretch()

        # Undo button — left of the Real Time display
        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.setEnabled(False)
        self.btn_undo.setStyleSheet(f"""
            QPushButton {{
                background: #0d1b2a; color: {COL_GREEN};
                border: 1px solid {COL_GREEN_DRK}; padding: 4px 12px;
                font-size: 13px; font-weight: bold; border-radius: 4px;
            }}
            QPushButton:hover:enabled {{ background: #162a3a; }}
            QPushButton:disabled {{ color: #4a5a68; border: 1px solid #2a3a48; }}
        """)
        self.btn_undo.clicked.connect(self._undo_last_action)
        top_layout.addWidget(self.btn_undo)

        sep_undo = QFrame()
        sep_undo.setFrameShape(QFrame.VLine)
        sep_undo.setStyleSheet(f"color: {COL_GREEN_DRK};")
        top_layout.addWidget(sep_undo)

        # Real-time display right of time tabs (current view window)
        self.lbl_real_time = QLabel("Frame Time: --:--:--")
        self.lbl_real_time.setStyleSheet(f"color: {COL_GREEN}; font-weight: bold; font-size: 13px;")
        top_layout.addWidget(self.lbl_real_time)
        layout.addWidget(top_bar)

        canvas_frame = QFrame()
        canvas_frame.setStyleSheet(f"background: {COL_BLACK};")
        self.canvas_layout = QVBoxLayout(canvas_frame)
        # Tightened from (4,4,4,4) - reclaim the outer gap so it can go toward
        # per-lead row height instead (see _canvas_height_for_gain below).
        self.canvas_layout.setContentsMargins(2, 2, 2, 2)
        self.canvas_layout.setSpacing(0)

        self._canvases = []
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self._canvas_height = self._canvas_height_for_gain(self._gain)
        for lead in leads:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            # Tightened from 4 - small reclaim from the label/canvas gap too.
            row.setSpacing(3)

            lbl = QLabel(lead)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet(
                f"color: {COL_GREEN}; font-weight: bold; font-size: 14px;"
                f" background: #0a0f18; border-right: 1px solid {COL_GREEN_DRK};"
                f" padding-right: 4px;"
            )
            lbl.setFixedWidth(44)

            canvas = ECGStripCanvas(canvas_frame, height=self._canvas_height, color=COL_GREEN, lead_name=lead, show_annotations=(lead == "I"))
            canvas.set_paper_speed(25)
            canvas.set_gain(self._gain)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            # Install event filter on each canvas to catch mouse clicks
            canvas.installEventFilter(self)

            row.addWidget(lbl)
            row.addWidget(canvas, 1)
            self.canvas_layout.addLayout(row)
            self._canvases.append(canvas)

        # TODO: Selection box overlay disabled for now
        # self.overlay = FullDisclosureOverlay(canvas_frame)
        # self.overlay.set_strip_length(self._strip_length)
        # self.overlay.on_selection_made = self._on_selection
        # self.overlay.double_clicked.connect(self._on_overlay_double_clicked)
        # # Enable mouse on overlay initially
        # self.overlay.set_mouse_enabled(True)
        
        # Add vertical line overlay for click tracking (spans all leads continuously)
        self._vertical_line_overlay = VerticalLineOverlay(canvas_frame)
        self._vertical_line_overlay.setGeometry(canvas_frame.rect())
        self._vertical_line_overlay.raise_()  # Make sure it's on top
        self._vertical_line_overlay.show()
        self._clicked_vertical_line_x = None  # Track clicked X position
        
        # Add segment overlay for segment selection mode (sits above vertical line overlay)
        self._segment_overlay = SegmentOverlay(canvas_frame)
        self._segment_overlay.setGeometry(canvas_frame.rect())
        self._segment_overlay.raise_()
        self._segment_overlay.show()
        
        # Install event filter BEFORE adding canvas_frame to layout
        canvas_frame.installEventFilter(self)
        self._canvas_frame = canvas_frame
        layout.addWidget(canvas_frame, 1)


        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setFixedHeight(12)
        self.time_scrollbar.setStyleSheet(f"""
            QScrollBar:horizontal {{
                background: #0d1b2a; height: 12px; border-radius: 5px; margin: 0 4px;
            }}
            QScrollBar::handle:horizontal {{
                background: {COL_GREEN_DRK}; min-width: 24px; border-radius: 5px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
        """)
        self._update_scrollbar_range()
        self.time_scrollbar.valueChanged.connect(self._on_scrollbar_moved)
        self.time_scrollbar.sliderPressed.connect(self._on_scrollbar_pressed)
        self.time_scrollbar.sliderReleased.connect(self._on_scrollbar_released)
        layout.addWidget(self.time_scrollbar)

        bot_bar = QFrame()
        bot_bar.setStyleSheet(f"background: {COL_DARK}; border-top: 1px solid {COL_GREEN_DRK};")
        bot_bar.setFixedHeight(40)
        bot_layout = QHBoxLayout(bot_bar)
        bot_layout.setContentsMargins(14, 5, 14, 5)
        bot_layout.setSpacing(8)

        def _tool_btn(text):
            b = QPushButton(text)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: #0d1b2a; color: {COL_GREEN};
                    border: 1px solid {COL_GREEN_DRK}; padding: 5px 14px;
                    font-size: 13px; font-weight: bold; border-radius: 4px;
                }}
            """)
            return b

        self.btn_gain  = _tool_btn(f"Gain: {self._gain_label}")
        self.btn_speed = _tool_btn(f"Speed: {self._paper_speed}mm/s")
        self.btn_gain.clicked.connect(self._cycle_gain)
        self.btn_speed.clicked.connect(self._cycle_speed)
        bot_layout.addWidget(self.btn_gain)
        bot_layout.addWidget(self.btn_speed)

        sep_tools = QFrame()
        sep_tools.setFrameShape(QFrame.VLine)
        sep_tools.setStyleSheet(f"color: {COL_GREEN_DRK};")
        bot_layout.addWidget(sep_tools)

        def _tool_toggle_btn(text, tool_id):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setStyleSheet(f"""
                QPushButton {{
                    background: #0d1b2a; color: #a0c4e8;
                    border: 1px solid {COL_GREEN_DRK}; padding: 5px 14px;
                    font-size: 13px; font-weight: bold; border-radius: 4px;
                }}
                QPushButton:checked {{
                    background: {COL_GREEN_DRK}; color: {COL_GREEN};
                    border: 1px solid {COL_GREEN};
                }}
                QPushButton:hover:!checked {{ background: #162a3a; }}
            """)
            b.clicked.connect(lambda checked, t=tool_id, btn=b: self._set_tool_mode(t, btn))
            return b

        self.btn_ruler   = _tool_toggle_btn("Measuring Ruler",  TOOL_RULER)
        self.btn_caliper = _tool_toggle_btn("Parallel Ruler",   TOOL_CALIPER)
        self.btn_magnify = _tool_toggle_btn("Magnifying Glass", TOOL_MAGNIFY)
        bot_layout.addWidget(self.btn_ruler)
        bot_layout.addWidget(self.btn_caliper)
        bot_layout.addWidget(self.btn_magnify)

        # --- Beat Selection mode dropdown (Parallel / Segment) ---
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtCore import QPoint
        self.btn_sel_mode = QPushButton("▲ Parallel Sel.")
        self.btn_sel_mode.setStyleSheet(f"""
            QPushButton {{
                background: #0d1b2a; color: #a0c4e8;
                border: 1px solid {COL_GREEN_DRK}; padding: 5px 14px;
                font-size: 13px; font-weight: bold; border-radius: 4px;
            }}
            QPushButton:hover {{ background: #162a3a; color: {COL_GREEN}; }}
        """)
        def _show_sel_mode_menu():
            menu = QMenu(self)
            menu.setStyleSheet(f"""
                QMenu {{
                    background-color: {COL_DARK}; color: #a0c4e8;
                    border: 1px solid {COL_GREEN_DRK};
                    font-size: 13px; font-weight: bold;
                }}
                QMenu::item:selected {{ background-color: {COL_GREEN_DRK}; color: {COL_GREEN}; }}
                QMenu::item {{ padding: 6px 20px; }}
            """)
            act_parallel_single = menu.addAction("✔ Parallel single beat selection" if self._selection_mode == 'parallel_single' else "  Parallel single beat selection")
            act_parallel_multi = menu.addAction("✔ Parallel multiple beat selection" if self._selection_mode == 'parallel_multi' else "  Parallel multiple beat selection")
            act_segment  = menu.addAction("✔ Segment Selection"  if self._selection_mode == 'segment'  else "  Segment Selection")
            act_parallel_single.triggered.connect(lambda: self._switch_selection_mode('parallel_single'))
            act_parallel_multi.triggered.connect(lambda: self._switch_selection_mode('parallel_multi'))
            act_segment.triggered.connect(lambda:  self._switch_selection_mode('segment'))
            btn_pos = self.btn_sel_mode.mapToGlobal(QPoint(0, 0))
            menu_h  = menu.sizeHint().height()
            menu.exec_(QPoint(btn_pos.x(), btn_pos.y() - menu_h))
        self.btn_sel_mode.clicked.connect(_show_sel_mode_menu)
        bot_layout.addWidget(self.btn_sel_mode)

        # Arrhythmia indicator left of Recording label
        self.lbl_arrhythmia = QLabel("")
        self.lbl_arrhythmia.setStyleSheet("color: #ff6b6b; font-weight: bold; font-size: 12px;")
        bot_layout.addStretch()
        bot_layout.addWidget(self.lbl_arrhythmia)
        
        # Total Recorded Time (right after arrhythmia label)
        self.lbl_total_recorded_time = QLabel("Recording: --:--:-- - --:--:--")
        self.lbl_total_recorded_time.setStyleSheet(f"color: {COL_GREEN}; font-weight: bold; font-size: 12px;")
        bot_layout.addWidget(self.lbl_total_recorded_time)
        layout.addWidget(bot_bar)

    def _update_recording_time_label(self):
        """Update the total recorded time label (bottom bar, next to arrhythmia)."""
        if hasattr(self._engine, '_reader') and hasattr(self._engine._reader, 'start_time'):
            try:
                start_timestamp = self._engine._reader.start_time
                end_timestamp = start_timestamp + self._engine.duration_sec
                start_real = datetime.fromtimestamp(start_timestamp)
                end_real = datetime.fromtimestamp(end_timestamp)
                self.lbl_total_recorded_time.setText(f"Recording: {start_real.strftime('%H:%M:%S')} - {end_real.strftime('%H:%M:%S')}")
            except Exception as e:
                print(f"[Full Disclosure] Error updating total recorded time label: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.setGeometry(self.rect())

    def set_magnifier_focus(self, source_widget, payload: dict, focus_pos):
        """Called by ECGStripCanvas when magnify tool is active."""
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.setGeometry(self.rect())
            self._magnifier_overlay.set_focus(source_widget, payload, focus_pos)

    def clear_magnifier_focus(self, source_widget=None):
        """Clear the shared magnifier overlay."""
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.clear_focus(source_widget)

    def eventFilter(self, obj, event):
        # Handle resize events for canvas_frame — keep both overlays in sync
        if obj == self._canvas_frame and event.type() == QEvent.Resize:
            if hasattr(self, '_vertical_line_overlay'):
                self._vertical_line_overlay.setGeometry(obj.rect())
            if hasattr(self, '_segment_overlay'):
                self._segment_overlay.setGeometry(obj.rect())
                # IMPORTANT: resizing the overlay only changes its bounding box —
                # each segment's cached pixel start_x/end_x still reflects whatever
                # canvas width was in effect when it was last computed. If the
                # canvas frame was still being laid out (e.g. right after the
                # dialog opens and finishes maximizing), those pixel positions
                # are stale and the segment renders in the wrong place (often
                # off-screen/invisible). Recompute them against the *current*
                # canvas width every time a resize happens so segments always
                # line up, on first open and on every subsequent resize.
                self._refresh_segment_overlay()
        
        # Only process mouse events from canvas_frame and canvases, NOT from other widgets
        is_canvas_event = (obj == self._canvas_frame or obj in self._canvases)
        if not is_canvas_event:
            return super().eventFilter(obj, event)
        
        # ----------------------------------------------------------------
        # RIGHT CLICK
        # ----------------------------------------------------------------
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            if obj in self._canvases:
                click_x = obj.mapTo(self._canvas_frame, event.pos()).x()
            else:
                click_x = event.pos().x()
            
            # Convert click_x (global pixel) to seconds to match segments
            click_sec = 0.0
            ref = self._canvases[0] if self._canvases else None
            if ref and hasattr(ref, '_start_sec') and hasattr(ref, '_data') and hasattr(ref, '_fs'):
                from PyQt5.QtCore import QPoint
                w = ref.width()
                data_len = len(ref._data)
                if w > 0 and data_len > 0:
                    ref_end_sec = ref._start_sec + data_len / ref._fs
                    click_x_local = ref.mapFrom(self._canvas_frame, QPoint(int(click_x), 0)).x()
                    click_x_local = max(0, min(w, click_x_local))
                    click_sec = ref._start_sec + (click_x_local / w) * (ref_end_sec - ref._start_sec)

            # Check if clicked inside an existing segment
            clicked_segment = None
            if self._selection_mode == 'segment' and hasattr(self, '_segment_annotations'):
                for seg in self._segment_annotations:
                    s_sec = min(seg['start_sec'], seg['end_sec'])
                    e_sec = max(seg['start_sec'], seg['end_sec'])
                    if s_sec <= click_sec <= e_sec:
                        clicked_segment = seg
                        break

            if self._selection_mode == 'segment':
                if clicked_segment is not None:
                    # Show segment deletion menu
                    from PyQt5.QtWidgets import QMenu, QAction
                    menu = QMenu(self)
                    menu.setStyleSheet(f"""
                        QMenu {{
                            background-color: {COL_DARK};
                            color: {COL_WHITE};
                            border: 1px solid {COL_GRID_MAJOR};
                            font-size: 13px; font-weight: bold;
                        }}
                        QMenu::item:selected {{ background-color: {COL_GRID_MAJOR}; }}
                        QMenu::item {{ padding: 6px 20px; }}
                    """)
                    del_act = QAction("Delete segment", self)
                    del_act.triggered.connect(lambda: self._delete_segment(clicked_segment))
                    menu.addAction(del_act)
                    menu.exec_(event.globalPos())
                else:
                    # In segment mode: show labeling menu for pending segment
                    self._show_segment_context_menu(event.globalPos())
            else:
                # In parallel mode (single or multi): original beat-level context menu
                self._show_beat_context_menu(click_x, event.globalPos())
            return True  # Consume the event


        # Let canvas handle left click/drag if a measurement tool is active
        if self._active_tool != TOOL_SELECT:
            return super().eventFilter(obj, event)

        # ----------------------------------------------------------------
        # SEGMENT SELECTION MODE — left button press / move / release
        # ----------------------------------------------------------------
        if self._selection_mode == 'segment':
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                if obj in self._canvases:
                    raw_x = obj.mapTo(self._canvas_frame, event.pos()).x()
                else:
                    raw_x = event.pos().x()
                self._seg_start_x  = raw_x
                self._seg_end_x    = raw_x
                self._seg_dragging = False
                self._pending_segment = None
                if hasattr(self, '_segment_overlay'):
                    self._segment_overlay.set_drag(raw_x, raw_x)
                return True

            elif event.type() == QEvent.MouseMove:
                if self._seg_start_x is not None:
                    if obj in self._canvases:
                        raw_x = obj.mapTo(self._canvas_frame, event.pos()).x()
                    else:
                        raw_x = event.pos().x()
                    if not self._seg_dragging and abs(raw_x - self._seg_start_x) > 4:
                        self._seg_dragging = True
                    if self._seg_dragging:
                        self._seg_end_x = raw_x
                        if hasattr(self, '_segment_overlay'):
                            self._segment_overlay.set_drag(self._seg_start_x, raw_x)
                return True

            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self._seg_start_x is not None and self._seg_dragging:
                    if obj in self._canvases:
                        raw_x = obj.mapTo(self._canvas_frame, event.pos()).x()
                    else:
                        raw_x = event.pos().x()
                    self._seg_end_x = raw_x
                    
                    # Convert pixel X → seconds using any canvas as reference
                    ref = self._canvases[0] if self._canvases else None
                    start_sec = end_sec = 0.0
                    if ref and hasattr(ref, '_start_sec') and hasattr(ref, '_data') and hasattr(ref, '_fs'):
                        from PyQt5.QtCore import QPoint
                        w = ref.width()
                        data_len = len(ref._data)
                        if w > 0 and data_len > 0:
                            ref_end_sec = ref._start_sec + data_len / ref._fs
                            sx_local = ref.mapFrom(self._canvas_frame, QPoint(int(min(self._seg_start_x, raw_x)), 0)).x()
                            ex_local = ref.mapFrom(self._canvas_frame, QPoint(int(max(self._seg_start_x, raw_x)), 0)).x()
                            sx_local = max(0, min(w, sx_local))
                            ex_local = max(0, min(w, ex_local))
                            span = ref_end_sec - ref._start_sec
                            start_sec = ref._start_sec + (sx_local / w) * span
                            end_sec   = ref._start_sec + (ex_local / w) * span
                    
                    self._pending_segment = {
                        'start_sec': start_sec, 'end_sec': end_sec,
                        'start_x': int(min(self._seg_start_x, raw_x)),
                        'end_x':   int(max(self._seg_start_x, raw_x)),
                    }
                self._seg_start_x  = None
                self._seg_dragging = False
                return True

            return super().eventFilter(obj, event)

        # ----------------------------------------------------------------
        # PARALLEL SELECTION MODE — existing beat-snap yellow-line logic
        # ----------------------------------------------------------------
        elif event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            if obj == self._canvas_frame or obj in self._canvases:
                if obj in self._canvases:
                    click_pos_local = event.pos()
                    click_pos_global = obj.mapTo(self._canvas_frame, click_pos_local)
                    click_x = click_pos_global.x()
                else:
                    click_x = event.pos().x()
                
                # Parallel multi mode: if lines already exist, don't reset them
                if self._selection_mode == 'parallel_multi' and self._multi_line1_x is not None and self._multi_line2_x is not None:
                    self._drag_start_x = click_x
                    self._drag_current_x = click_x
                    self._is_dragging = False
                    return True
                
                self._drag_start_x = click_x
                self._drag_current_x = click_x
                self._is_dragging = False
                self._clicked_vertical_line_x = click_x
                
                # Parallel multi mode: two vertical lines
                if self._selection_mode == 'parallel_multi':
                    self._multi_line1_x = click_x
                    self._multi_line2_x = click_x  # Initially same position
                    if hasattr(self, '_vertical_line_overlay'):
                        self._vertical_line_overlay.set_line_positions([click_x])
                else:
                    # Parallel single mode: single line
                    if hasattr(self, '_vertical_line_overlay'):
                        self._vertical_line_overlay.set_line_position(click_x)
                
                lead_i_canvas = None
                for c in self._canvases:
                    if c.lead_name == 'I':
                        lead_i_canvas = c
                        break
                
                clicked_timestamp = None
                clicked_label = None
                if lead_i_canvas:
                    from PyQt5.QtCore import QPoint
                    click_x_local = lead_i_canvas.mapFrom(self._canvas_frame, QPoint(click_x, 0)).x()
                    click_x_local = max(0, min(lead_i_canvas.width(), click_x_local))
                    lead_i_canvas._check_and_store_clicked_beat(click_x_local)
                    clicked_timestamp = lead_i_canvas._clicked_beat_timestamp
                    clicked_label = lead_i_canvas._clicked_beat_label
                
                self._drag_start_timestamp = clicked_timestamp
                
                for canvas in self._canvases:
                    if clicked_timestamp is not None:
                        if lead_i_canvas and hasattr(lead_i_canvas, '_clicked_beat_x_pos'):
                            canvas._clicked_beat_x_pos = lead_i_canvas._clicked_beat_x_pos
                        else:
                            canvas._clicked_beat_x_pos = click_x
                        canvas._clicked_beat_timestamp = clicked_timestamp
                        canvas._clicked_beat_label = clicked_label
                        canvas._selected_beats = [clicked_timestamp] if clicked_timestamp else []
                    else:
                        canvas._clicked_beat_x_pos = None
                        canvas._clicked_beat_timestamp = None
                        canvas._clicked_beat_label = None
                        canvas._selected_beats = []
                    canvas.update()
                
                if hasattr(self, '_vertical_line_overlay'):
                    if self._selection_mode == 'parallel_multi':
                        # For multi mode, show both lines if they're different
                        line_positions = []
                        if self._multi_line1_x is not None:
                            line_positions.append(self._multi_line1_x)
                        if self._multi_line2_x is not None and self._multi_line2_x != self._multi_line1_x:
                            line_positions.append(self._multi_line2_x)
                        self._vertical_line_overlay.set_line_positions(line_positions)
                    else:
                        # Single mode behavior
                        if clicked_timestamp is not None and lead_i_canvas and hasattr(lead_i_canvas, '_clicked_beat_x_pos') and lead_i_canvas._clicked_beat_x_pos is not None:
                            from PyQt5.QtCore import QPoint
                            beat_x_global = lead_i_canvas.mapTo(self._canvas_frame, QPoint(lead_i_canvas._clicked_beat_x_pos, 0)).x()
                            self._vertical_line_overlay.set_line_position(beat_x_global)
                        else:
                            self._vertical_line_overlay.set_line_position(click_x)
                
                return True
        
        elif event.type() == QEvent.MouseMove:
            if hasattr(self, '_drag_start_x') and self._drag_start_x is not None and is_canvas_event:
                # Parallel multi mode: second line follows mouse
                if self._selection_mode == 'parallel_multi':
                    if obj in self._canvases:
                        drag_pos_global = obj.mapTo(self._canvas_frame, event.pos())
                        drag_x = drag_pos_global.x()
                    else:
                        drag_x = event.pos().x()
                    
                    self._multi_line2_x = drag_x
                    if hasattr(self, '_vertical_line_overlay'):
                        line_positions = []
                        if self._multi_line1_x is not None:
                            line_positions.append(self._multi_line1_x)
                        if self._multi_line2_x is not None and self._multi_line2_x != self._multi_line1_x:
                            line_positions.append(self._multi_line2_x)
                        self._vertical_line_overlay.set_line_positions(line_positions)
                    return True
                
                # Parallel single mode: disable drag behavior
                if self._selection_mode == 'parallel_single':
                    return True
                
                # Segment mode: allow drag
                if self._selection_mode == 'segment':
                    if obj in self._canvases:
                        drag_pos_global = obj.mapTo(self._canvas_frame, event.pos())
                        drag_x = drag_pos_global.x()
                    else:
                        drag_x = event.pos().x()
                    
                    if not self._is_dragging and abs(drag_x - self._drag_start_x) > 5:
                        self._is_dragging = True
                    
                    if self._is_dragging:
                        self._drag_current_x = drag_x
                        start_x = min(self._drag_start_x, drag_x)
                        end_x   = max(self._drag_start_x, drag_x)
                        
                        self._seg_end_x = end_x
                        if hasattr(self, '_segment_overlay'):
                            self._segment_overlay.update()
                        return True
        
        elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if hasattr(self, '_drag_start_x') and is_canvas_event:
                # Parallel multi mode: keep both lines fixed
                if self._selection_mode == 'parallel_multi':
                    # Second line is already fixed at current position from MouseMove
                    # Clear drag state but keep line positions
                    self._drag_start_x   = None
                    self._drag_current_x = None
                    self._is_dragging    = False
                else:
                    # Single mode: clear everything
                    self._drag_start_x   = None
                    self._drag_current_x = None
                    self._is_dragging    = False
                return True
                
        return super().eventFilter(obj, event)

    # TODO: Strip length change handler disabled for now
    # def _on_strip_length_changed(self, val):
    #     self._strip_length = float(val)
    #     self.overlay.set_strip_length(self._strip_length)

    def _cycle_gain(self):
        multipliers = [g[0] for g in self._GAIN_STEPS]
        try:
            idx = multipliers.index(self._gain)
        except ValueError:
            idx = 0
        next_step = self._GAIN_STEPS[(idx + 1) % len(self._GAIN_STEPS)]
        self._gain, self._gain_label = next_step
        self.btn_gain.setText(f"Gain: {self._gain_label}")
        self._apply_canvas_height_for_gain()
        for c in self._canvases:
            c.set_gain(self._gain)
            c.update()  # Force repaint with new gain
        # Restore selection box when gain is changed
        self._deactivate_tools()

    def _cycle_speed(self):
        try:
            idx = self._SPEED_STEPS.index(self._paper_speed)
        except ValueError:
            idx = 1
        self._paper_speed = self._SPEED_STEPS[(idx + 1) % len(self._SPEED_STEPS)]
        self.btn_speed.setText(f"Speed: {self._paper_speed}mm/s")
        self._recalc_window()
        self._update_scrollbar_range()
        for c in self._canvases:
            c.set_paper_speed(int(self._paper_speed))
        self._update_canvases(self._current_start)
        # Restore selection box when speed is changed
        self._deactivate_tools()

    def _deactivate_tools(self):
        """Deactivate all measurement tools and restore the selection box overlay."""
        self._active_tool = TOOL_SELECT
        self._active_tool_btn = None
        for btn in [self.btn_ruler, self.btn_caliper, self.btn_magnify]:
            btn.setChecked(False)
        for c in self._canvases:
            if hasattr(c, 'set_mode'):
                c.set_mode(TOOL_SELECT)
        self.clear_magnifier_focus()
        # TODO: Overlay show/hide disabled for now
        # self.overlay.show()

    # ------------------------------------------------------------------
    # Selection mode switching
    # ------------------------------------------------------------------
    def _switch_selection_mode(self, mode: str):
        """Switch between 'parallel_single', 'parallel_multi', and 'segment' modes."""
        self._selection_mode = mode
        
        # CRITICAL FIX: Save selection mode to settings so it persists after dialog close/reopen
        if SettingsManager is not None:
            try:
                settings = SettingsManager()
                settings.set_setting('holter_selection_mode', mode)
                print(f"[Full Disclosure] Saved selection mode: {mode}")
            except Exception as e:
                print(f"[Full Disclosure] Could not save selection mode to settings: {e}")
        
        if mode == 'parallel_single':
            self.btn_sel_mode.setText("▲ Parallel Single")
            # Clear any pending segment drag
            self._seg_start_x = None
            self._seg_end_x   = None
            self._seg_dragging = False
            self._pending_segment = None
            if hasattr(self, '_segment_overlay'):
                self._segment_overlay.clear_drag()
            # Restore vertical line overlay visibility
            if hasattr(self, '_vertical_line_overlay'):
                self._vertical_line_overlay.show()
            # DO NOT clear structured events - they should persist across mode switches
        elif mode == 'parallel_multi':
            self.btn_sel_mode.setText("▲ Parallel Multi")
            # Clear any pending segment drag
            self._seg_start_x = None
            self._seg_end_x   = None
            self._seg_dragging = False
            self._pending_segment = None
            if hasattr(self, '_segment_overlay'):
                self._segment_overlay.clear_drag()
            # Clear multi-line state
            self._multi_line1_x = None
            self._multi_line2_x = None
            # Restore vertical line overlay visibility
            if hasattr(self, '_vertical_line_overlay'):
                self._vertical_line_overlay.clear_line()
                self._vertical_line_overlay.show()
            # DO NOT clear structured events - they should persist across mode switches
        else:  # segment mode
            self.btn_sel_mode.setText("▲ Segment Sel.")
            # Clear parallel-mode state
            self._drag_start_x = None
            self._drag_current_x = None
            self._is_dragging = False
            if hasattr(self, '_vertical_line_overlay'):
                self._vertical_line_overlay.clear_line()
                self._vertical_line_overlay.hide()  # Hide in segment mode
            for c in self._canvases:
                c._selected_beats = []
                c._clicked_beat_timestamp = None
                c.update()
            # DO NOT clear structured events when switching to segment mode - they should persist
            # Structured events are only cleared when explicitly deleted or when the dialog is closed
            if hasattr(self, '_segment_overlay'):
                self._segment_overlay.show()
                self._refresh_segment_overlay()

    # ------------------------------------------------------------------
    # Segment context menu + labeling
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Undo support
    # ------------------------------------------------------------------
    def _snapshot_state(self, action_label: str):
        """Push a deep-copy snapshot of all mutable annotation state onto the
        undo stack. Call this BEFORE applying a mutating action (beat label,
        beat delete, segment label, segment delete) so undo can restore the
        exact prior state."""
        beat_annotations_by_lead = {}
        for c in getattr(self, '_canvases', []):
            if hasattr(c, '_beat_annotations') and c._beat_annotations is not None:
                beat_annotations_by_lead[c.lead_name] = copy.deepcopy(c._beat_annotations)

        snapshot = {
            'action': action_label,
            'segment_annotations': copy.deepcopy(getattr(self, '_segment_annotations', [])),
            'beat_annotations_by_lead': beat_annotations_by_lead,
            'engine_metrics': copy.deepcopy(getattr(self._engine, '_metrics', None)),
            'structured_events': copy.deepcopy(getattr(self._engine, '_structured_events', None)),
        }
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._UNDO_STACK_LIMIT:
            self._undo_stack.pop(0)
        if hasattr(self, 'btn_undo'):
            self.btn_undo.setEnabled(True)

    def _persist_segment_annotations(self):
        """Overwrite manual_segments.json with the current in-memory list."""
        try:
            session_dir = os.path.dirname(self._engine.ecgh_path)
            segments_path = os.path.join(session_dir, 'manual_segments.json')
            with open(segments_path, 'w') as f:
                json.dump(self._segment_annotations, f, indent=2)
        except Exception as e:
            print(f"[Full Disclosure] Error persisting segment annotations: {e}")

    def _undo_last_action(self):
        """Revert the most recent beat/segment annotation action."""
        if not self._undo_stack:
            return

        snapshot = self._undo_stack.pop()

        self._segment_annotations = snapshot['segment_annotations']

        for c in self._canvases:
            if c.lead_name in snapshot['beat_annotations_by_lead']:
                c._beat_annotations = snapshot['beat_annotations_by_lead'][c.lead_name]

        if snapshot['engine_metrics'] is not None:
            self._engine._metrics = snapshot['engine_metrics']
            self._beat_index_dirty = True
        if snapshot['structured_events'] is not None:
            self._engine._structured_events = snapshot['structured_events']
            self._event_index_dirty = True

        # Persist the reverted state to disk so it stays consistent with
        # report generation and future reloads.
        self._save_manual_beats()
        self._persist_segment_annotations()

        self.btn_undo.setEnabled(bool(self._undo_stack))

        # Refresh everything on screen: waveforms, beat markers, segment overlay.
        self._update_canvases(self._current_start)
        print(f"[Full Disclosure] Undid action: {snapshot['action']}")

    def _show_segment_context_menu(self, global_pos):
        """Show arrhythmia right-click context menu for labeling a selected segment with hierarchical dropdown menus."""
        from PyQt5.QtWidgets import QMenu, QAction
        if self._pending_segment is None:
            return

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COL_DARK};
                color: {COL_WHITE};
                border: 1px solid {COL_GRID_MAJOR};
                font-size: 13px; font-weight: bold;
            }}
            QMenu::item:selected {{ background-color: {COL_GRID_MAJOR}; }}
            QMenu::item {{ padding: 6px 20px; }}
        """)

        # Define hierarchical arrhythmia structure with main labels and sub-labels
        # Format: sub_label(First capital letter of main label)
        arrhythmia_structure = {
            "Supraventricular": [
                "Atrial Fibrillation 1",
                "Atrial Fibrillation 2",
                "Atrial Flutter",
                "Sinus Arrhythmia",
                "Missed Beat at 80 bpm",
                "Missed Beat at 120 bpm",
                "Atrial Tach",
                "Paroxysmal A Tach",
                "Nodal Rhythm",
                "Supra VTach"
            ],
            "Premature": [
                "Atrial PAC",
                "Nodal PNC",
                "PVC1 left Vent",
                "PVC1 LV Early",
                "PVC1 LV R on T",
                "PVC2 Right Vent",
                "PVC2 RV Early",
                "PVC2 RV R on T",
                "Multi-focal PVCs"
            ],
            "Ventricular": [
                "PVCs",
                "Freq Multi-focal PV",
                "Trigeminy",
                "Bigeminy",
                "R on of PVCs",
                "Mono VTach",
                "Poly VTach",
                "Ventricular Fibrillation",
                "Asystole"
            ],
            "Conduction": [
                "1st Deg AV Block",
                "2nd Deg AV Block T1",
                "2nd Deg AV Block T2",
                "3rd Deg AV Block",
                "Rt Bundle Branch Block",
                "Lt Bundle Branch Block"
            ],
            "TV Paced": [
                "Atrial 80 bpm",
                "Asynchronous 75 bpm",
                "Demand Freq Sinus",
                "Demand Occ Sinus",
                "Atr-Vent Sequential",
                "Non-Capture",
                "Non-Function"
            ],
            "ACLS": [
                "Ventricular Fibrillation",
                "Poly VTach (Unstable)",
                "Asystole",
                "Sinus Bradycardia",
                "2nd Deg AV Block T",
                "3rd Deg AV Block",
                "Rt Bundle Branch B",
                "Lt Bundle Branch B1",
                "Narrow QRS Tach",
                "Wide QRS Tach",
                "Atrial Fibrillation 1",
                "Atrial Fibrillation 2",
                "Atrial Flutter",
                "Mono VTach (Unstable)",
                "Torsade de Pointe"
            ]
        }
        
        # Color mapping for main label first letters
        main_label_colors = {
            "S": "#00FFFF",      # Supraventricular - Cyan
            "P": "#FF00FF",      # Premature - Magenta
            "V": "#FF3333",      # Ventricular - Red
            "C": "#FFA500",      # Conduction - Orange
            "T": "#9932CC",      # TV Paced - Dark Orchid (Purple)
            "A": "#FFFF00",      # ACLS - Yellow
        }
        
        # Create hierarchical menu structure
        for main_label, sub_labels in arrhythmia_structure.items():
            # Get first letter of main label
            main_first_letter = main_label[0]
            
            # Create submenu for this main label
            submenu = menu.addMenu(main_label)
            
            # Add sub-labels to the submenu
            for sub_label in sub_labels:
                # Format: sub_label(First capital letter of main label)
                formatted_label = f"{sub_label}({main_first_letter})"
                color = main_label_colors.get(main_first_letter, "#FFFF00")
                act = QAction(formatted_label, self)
                act.triggered.connect(lambda checked, full_label=formatted_label, col=color: self._label_segment(full_label, col))
                submenu.addAction(act)

        menu.exec_(global_pos)

    def _label_segment(self, full_label: str, color: str):
        """Assign an arrhythmia label to the pending segment and render it."""
        if self._pending_segment is None:
            return

        self._snapshot_state('label_segment')

        seg = self._pending_segment
        start_sec = seg['start_sec']
        end_sec   = seg['end_sec']
        sx        = seg['start_x']
        ex        = seg['end_x']

        # Compute real-world timestamp strings for both start and end using actual system recorded time
        start_time_str = ''
        end_time_str = ''
        try:
            reader = getattr(self._engine, '_reader', None) or getattr(self, '_reader', None)
            if reader and hasattr(reader, 'start_time') and reader.start_time is not None:
                start_real = datetime.fromtimestamp(reader.start_time + start_sec)
                end_real = datetime.fromtimestamp(reader.start_time + end_sec)
                start_time_str = start_real.strftime('%H:%M:%S')
                end_time_str = end_real.strftime('%H:%M:%S')
            else:
                h = int(start_sec // 3600)
                m = int((start_sec % 3600) // 60)
                s = int(start_sec % 60)
                start_time_str = f"{h:02d}:{m:02d}:{s:02d}"
                
                h = int(end_sec // 3600)
                m = int((end_sec % 3600) // 60)
                s = int(end_sec % 60)
                end_time_str = f"{h:02d}:{m:02d}:{s:02d}"
        except Exception as e:
            print(f"[Full Disclosure] Error formatting segment time: {e}")


        # Store in persistent annotations list
        annotation = {
            'start_sec': start_sec, 'end_sec': end_sec,
            'label': full_label, 'color': color, 'start_time_str': start_time_str, 'end_time_str': end_time_str
        }
        self._segment_annotations.append(annotation)

        # Save segment annotations to JSON file for report generation
        try:
            session_dir = os.path.dirname(self._engine.ecgh_path)
            segments_path = os.path.join(session_dir, 'manual_segments.json')
            segments_data = []
            if os.path.exists(segments_path):
                with open(segments_path, 'r') as f:
                    segments_data = json.load(f)
            segments_data.append(annotation)
            with open(segments_path, 'w') as f:
                json.dump(segments_data, f, indent=2)
            print(f"[Full Disclosure] Saved segment annotation to {segments_path}")
            print(f"[Full Disclosure] Total segments in file: {len(segments_data)}")
        except Exception as e:
            print(f"[Full Disclosure] Error saving segment annotations: {e}")

        # Paint it on the overlay - _update_canvases will handle adding all segments
        if hasattr(self, '_segment_overlay'):
            self._segment_overlay.clear_drag()

        self._pending_segment = None
        
        # Trigger canvas update to refresh segment overlay with new segment
        self._update_canvases(self._current_start)

    def _clear_pending_segment(self):
        """Clear the current drag selection without labeling it."""
        self._pending_segment = None
        self._seg_start_x = None
        self._seg_end_x   = None
        self._seg_dragging = False
        if hasattr(self, '_segment_overlay'):
            self._segment_overlay.clear_drag()

    def _delete_segment(self, segment):
        """Remove a finalized segment from annotations and the overlay."""
        self._snapshot_state('delete_segment')
        if segment in self._segment_annotations:
            self._segment_annotations.remove(segment)
        
        # Remove from JSON file as well
        try:
            session_dir = os.path.dirname(self._engine.ecgh_path)
            segments_path = os.path.join(session_dir, 'manual_segments.json')
            if os.path.exists(segments_path):
                with open(segments_path, 'r') as f:
                    segments_data = json.load(f)
                # Remove the segment by matching start_sec, end_sec, and label
                segments_data = [s for s in segments_data if not (
                    abs(s.get('start_sec', 0) - segment.get('start_sec', 0)) < 0.01 and
                    abs(s.get('end_sec', 0) - segment.get('end_sec', 0)) < 0.01 and
                    s.get('label') == segment.get('label')
                )]
                with open(segments_path, 'w') as f:
                    json.dump(segments_data, f, indent=2)
                print(f"[Full Disclosure] Removed segment from {segments_path}")
        except Exception as e:
            print(f"[Full Disclosure] Error removing segment from JSON: {e}")
        
        # Clear and rebuild visible segment list in segment overlay
        if hasattr(self, '_segment_overlay'):
            self._segment_overlay.clear_segments()
            ref = self._canvases[0] if self._canvases else None
            if ref and hasattr(ref, '_start_sec') and hasattr(ref, '_data') and hasattr(ref, '_fs'):
                from PyQt5.QtCore import QPoint
                ref_w  = ref.width()
                data_l = len(ref._data)
                if ref_w > 0 and data_l > 0:
                    ref_end = ref._start_sec + data_l / ref._fs
                    span    = ref_end - ref._start_sec
                    
                    for seg in self._segment_annotations:
                        s_sec = seg['start_sec']
                        e_sec = seg['end_sec']
                        # Check overlap with visible range
                        if max(s_sec, ref._start_sec) <= min(e_sec, ref_end):
                            if span > 0:
                                sx_local = int(((s_sec - ref._start_sec) / span) * ref_w)
                                ex_local = int(((e_sec - ref._start_sec) / span) * ref_w)
                                sx_local = max(0, min(ref_w, sx_local))
                                ex_local = max(0, min(ref_w, ex_local))
                                sx_global = ref.mapTo(self._canvas_frame, QPoint(sx_local, 0)).x()
                                ex_global = ref.mapTo(self._canvas_frame, QPoint(ex_local, 0)).x()
                                
                                self._segment_overlay.add_segment(
                                    sx_global, ex_global, s_sec, e_sec,
                                    seg['label'], seg['color'], seg.get('start_time_str', ''), seg.get('end_time_str', '')
                                )
            self._segment_overlay.update()


    def _show_beat_context_menu(self, click_x: int, global_pos):
        """Show right-click context menu for beat labeling with hierarchical dropdown menus."""
        from PyQt5.QtWidgets import QMenu, QAction
        
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COL_DARK};
                color: {COL_WHITE};
                border: 1px solid {COL_GRID_MAJOR};
            }}
            QMenu::item:selected {{
                background-color: {COL_GRID_MAJOR};
            }}
        """)
        
        # Define hierarchical arrhythmia structure with main labels and sub-labels
        # Format: sub_label(First capital letter of main label)
        arrhythmia_structure = {
            "Supraventricular": [
                "Atrial Fibrillation 1",
                "Atrial Fibrillation 2",
                "Atrial Flutter",
                "Sinus Arrhythmia",
                "Missed Beat at 80 bpm",
                "Missed Beat at 120 bpm",
                "Atrial Tach",
                "Paroxysmal A Tach",
                "Nodal Rhythm",
                "Supra VTach"
            ],
            "Premature": [
                "Atrial PAC",
                "Nodal PNC",
                "PVC1 left Vent",
                "PVC1 LV Early",
                "PVC1 LV R on T",
                "PVC2 Right Vent",
                "PVC2 RV Early",
                "PVC2 RV R on T",
                "Multi-focal PVCs"
            ],
            "Ventricular": [
                "PVCs",
                "Freq Multi-focal PV",
                "Trigeminy",
                "Bigeminy",
                "R on of PVCs",
                "Mono VTach",
                "Poly VTach",
                "Ventricular Fibrillation",
                "Asystole"
            ],
            "Conduction": [
                "1st Deg AV Block",
                "2nd Deg AV Block T1",
                "2nd Deg AV Block T2",
                "3rd Deg AV Block",
                "Rt Bundle Branch Block",
                "Lt Bundle Branch Block"
            ],
            "TV Paced": [
                "Atrial 80 bpm",
                "Asynchronous 75 bpm",
                "Demand Freq Sinus",
                "Demand Occ Sinus",
                "Atr-Vent Sequential",
                "Non-Capture",
                "Non-Function"
            ],
            "ACLS": [
                "Ventricular Fibrillation",
                "Poly VTach (Unstable)",
                "Asystole",
                "Sinus Bradycardia",
                "2nd Deg AV Block T",
                "3rd Deg AV Block",
                "Rt Bundle Branch B",
                "Lt Bundle Branch B1",
                "Narrow QRS Tach",
                "Wide QRS Tach",
                "Atrial Fibrillation 1",
                "Atrial Fibrillation 2",
                "Atrial Flutter",
                "Mono VTach (Unstable)",
                "Torsade de Pointe"
            ]
        }
        
        # Color mapping for main labels
        main_label_colors = {
            "Supraventricular": "#00FFFF",  # Cyan
            "Premature": "#FF00FF",         # Magenta
            "Ventricular": "#FF3333",       # Red
            "Conduction": "#FFA500",        # Orange
            "TV Paced": "#9932CC",          # Dark Orchid (Purple)
            "ACLS": "#FFFF00"               # Yellow
        }
        
        # Create hierarchical menu structure
        for main_label, sub_labels in arrhythmia_structure.items():
            # Get first letter of main label
            main_first_letter = main_label[0]
            
            # Create submenu for this main label
            submenu = menu.addMenu(main_label)
            
            # Add sub-labels to the submenu
            for sub_label in sub_labels:
                # Format: sub_label(First capital letter of main label)
                formatted_label = f"{sub_label}({main_first_letter})"
                action = QAction(formatted_label, self)
                action.triggered.connect(lambda checked, full_label=formatted_label, x=click_x: self._label_beat(x, full_label))
                submenu.addAction(action)
        
        menu.addSeparator()
        
        # Additional options
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_beat_at_position(click_x))
        menu.addAction(delete_action)
        
        menu.addSeparator()
        
        # Other options
        add_strip_action = QAction("Add/Print Strip(space)", self)
        menu.addAction(add_strip_action)
        
        interval_action = QAction("Interval reanalysis", self)
        menu.addAction(interval_action)
        
        add_afib_action = QAction("Add AFib Evt.", self)
        menu.addAction(add_afib_action)
        
        add_af_action = QAction("Add AF Evt.", self)
        menu.addAction(add_af_action)
        
        menu.addSeparator()
        
        amplitude_action = QAction("Amplitude Ruler", self)
        menu.addAction(amplitude_action)
        
        parallel_action = QAction("Parallel Ruler", self)
        menu.addAction(parallel_action)
        
        menu.addSeparator()
        
        set_start_action = QAction("Set as starting", self)
        menu.addAction(set_start_action)
        
        set_end_action = QAction("Set as ending", self)
        menu.addAction(set_end_action)
        
        menu.addSeparator()
        
        equal_interval_action = QAction("equal interval bulkinsert beats", self)
        menu.addAction(equal_interval_action)
        
        menu.addSeparator()
        
        set_max_hr_action = QAction("Set as max HR", self)
        menu.addAction(set_max_hr_action)
        
        set_min_hr_action = QAction("Set as min HR", self)
        menu.addAction(set_min_hr_action)
        
        set_sinus_max_action = QAction("Set as Sinus Max. HR", self)
        menu.addAction(set_sinus_max_action)
        
        set_sinus_min_action = QAction("Set as Sinus Min. HR", self)
        menu.addAction(set_sinus_min_action)
        
        # Show menu at cursor position
        menu.exec_(global_pos)
    
    def _label_beat(self, click_x: int, full_label: str):
        """Label the beat at click_x with the given label."""
        self._snapshot_state('label_beat')
        
        # Extract the short code from full_label (e.g., "Atrial Fibrillation 1(S)" -> "S")
        # The format is: sub_label(First capital letter of main label)
        import re
        match = re.search(r'\(([A-Z])\)$', full_label)
        if match:
            label = match.group(1)  # Extract the letter in parentheses
        else:
            # Fallback: use first letter if no parentheses found
            label = full_label[0] if full_label else "N"
        
        # Define color mapping for main label first letters
        label_colors = {
            "S": "#00FFFF",      # Supraventricular - Cyan
            "P": "#FF00FF",      # Premature - Magenta
            "V": "#FF3333",      # Ventricular - Red
            "C": "#FFA500",      # Conduction - Orange
            "T": "#9932CC",      # TV Paced - Dark Orchid (Purple)
            "A": "#FFFF00",      # ACLS - Yellow
            "N": "#00FF00",      # Normal - Green (legacy)
            "X": "#0000FF",      # Artifact - Blue (legacy)
        }
        
        # Parallel multi mode: label all beats between two vertical lines
        if self._selection_mode == 'parallel_multi' and self._multi_line1_x is not None and self._multi_line2_x is not None:
            # Convert line positions to timestamps
            line1_x = self._multi_line1_x
            line2_x = self._multi_line2_x
            start_x = min(line1_x, line2_x)
            end_x = max(line1_x, line2_x)
            
            # Get reference canvas (Lead I)
            ref_canvas = None
            for canvas in self._canvases:
                if canvas.lead_name == 'I':
                    ref_canvas = canvas
                    break
            
            if ref_canvas:
                from PyQt5.QtCore import QPoint
                w = ref_canvas.width()
                if w > 0:
                    start_x_local = ref_canvas.mapFrom(self._canvas_frame, QPoint(start_x, 0)).x()
                    end_x_local = ref_canvas.mapFrom(self._canvas_frame, QPoint(end_x, 0)).x()
                    start_x_local = max(0, min(w, start_x_local))
                    end_x_local = max(0, min(w, end_x_local))
                    
                    # Convert to timestamps
                    start_sec = ref_canvas._start_sec
                    data_len = len(ref_canvas._data) if hasattr(ref_canvas, '_data') else 0
                    if data_len > 0:
                        end_sec = start_sec + data_len / ref_canvas._fs
                        start_ts = start_sec + (start_x_local / w) * (end_sec - start_sec)
                        end_ts = start_sec + (end_x_local / w) * (end_sec - start_sec)
                        
                        # Label all beats in this range for all leads
                        import time
                        batch_id = int(time.time() * 1000)  # Unique ID for this batch
                        
                        print(f"[Full Disclosure] Parallel multi marking: label='{label}', start_ts={start_ts:.3f}s, end_ts={end_ts:.3f}s")
                        
                        # Add structured event for entire waveform coloring (non-NSR labels only)
                        if label != 'N':
                            if not hasattr(self._engine, '_structured_events'):
                                self._engine._structured_events = []
                            
                            structured_event = {
                                'timestamp': start_ts,
                                'end_timestamp': end_ts,
                                'label': label,  # Use short code (S, V, AF, P, C, T, A) for waveform coloring
                                'label_full': full_label,  # Store full label for reporting
                                'color': label_colors.get(label, "#FFFF00"),
                                'event_type': label,
                                'source': 'manual_parallel_multi'  # Track source for debugging
                            }
                            self._engine._structured_events.append(structured_event)
                            self._event_index_dirty = True
                            
                            print(f"[Full Disclosure] Parallel multi: Added structured event from {start_ts:.3f}s to {end_ts:.3f}s")
                            print(f"[Full Disclosure] Event: code='{label}', label='{full_label}', color='{label_colors.get(label, '#FFFF00')}'")
                            print(f"[Full Disclosure] Total events: {len(self._engine._structured_events)}")
                            
                            # Ensure events are sorted by timestamp for bisect to work correctly
                            self._engine._structured_events.sort(key=lambda x: float(x.get('timestamp', 0.0)))
                            
                            # Refresh UI to apply waveform coloring
                            self._update_canvases(self._current_start)
                        else:
                            print(f"[Full Disclosure] NSR label selected - skipping waveform coloring, only QRS peaks will be colored")
                        
                        # Also label individual beats for report timeline
                        for canvas in self._canvases:
                            if not hasattr(canvas, '_beat_annotations') or canvas._beat_annotations is None:
                                canvas._beat_annotations = []
                            
                            # Find all detected beats in the time range
                            beats_in_range = []
                            if hasattr(self, '_detected_r_peaks') and self._detected_r_peaks:
                                for peak_ts in self._detected_r_peaks:
                                    if start_ts <= peak_ts <= end_ts:
                                        beats_in_range.append(peak_ts)
                            
                            # Label each beat in range
                            snap_tolerance_sec = 0.15
                            for target_ts in beats_in_range:
                                found = False
                                for beat in canvas._beat_annotations:
                                    if abs(beat['timestamp'] - target_ts) < snap_tolerance_sec:
                                        beat['label'] = full_label
                                        beat['color'] = label_colors.get(label, "#FFFF00")
                                        beat['is_manual'] = True
                                        beat['marking_mode'] = 'parallel_multi'
                                        beat['batch_id'] = batch_id
                                        found = True
                                        break
                                
                                if not found:
                                    new_beat = {
                                        'timestamp': target_ts,
                                        'label': full_label,
                                        'color': label_colors.get(label, "#FFFF00"),
                                        'is_manual': True,
                                        'marking_mode': 'parallel_multi',
                                        'batch_id': batch_id
                                    }
                                    canvas._beat_annotations.append(new_beat)
                        
                        # Update engine metrics so manual beats persist across redraws/scrolling
                        if hasattr(self._engine, '_metrics') and self._engine._metrics:
                            for target_ts in beats_in_range:
                                engine_beat_found = False
                                for m in self._engine._metrics:
                                    all_beats = m.get('all_beats', [])
                                    for engine_beat in all_beats:
                                        if abs(float(engine_beat.get('timestamp', 0.0)) - target_ts) < snap_tolerance_sec:
                                            engine_beat['label'] = full_label
                                            engine_beat['color'] = label_colors.get(label, "#FFFF00")
                                            engine_beat['is_manual'] = True
                                            engine_beat['marking_mode'] = 'parallel_multi'
                                            engine_beat['batch_id'] = batch_id
                                            engine_beat_found = True
                                            break
                                    if engine_beat_found:
                                        break
                                
                                if not engine_beat_found and self._engine._metrics:
                                    added = False
                                    for m in self._engine._metrics:
                                        m_start = float(m.get('start_sec', 0.0))
                                        m_dur = float(m.get('duration', 3600.0))
                                        if m_start <= target_ts <= m_start + m_dur:
                                            m.setdefault('all_beats', []).append({
                                                'timestamp': target_ts,
                                                'label': full_label,
                                                'color': label_colors.get(label, "#FFFF00"),
                                                'is_manual': True,
                                                'marking_mode': 'parallel_multi',
                                                'batch_id': batch_id
                                            })
                                            added = True
                                            break
                                    if not added:
                                        self._engine._metrics[0].setdefault('all_beats', []).append({
                                            'timestamp': target_ts,
                                            'label': full_label,
                                            'color': label_colors.get(label, "#FFFF00"),
                                            'is_manual': True,
                                            'marking_mode': 'parallel_multi',
                                            'batch_id': batch_id
                                        })
                            self._beat_index_dirty = True

                        print(f"[Full Disclosure] Labeled all beats between {start_ts:.3f}s and {end_ts:.3f}s as '{label}'")
                        
                        # Clear vertical lines
                        self._multi_line1_x = None
                        self._multi_line2_x = None
                        if hasattr(self, '_vertical_line_overlay'):
                            self._vertical_line_overlay.clear_line()
                        
                        # Update all canvases
                        for canvas in self._canvases:
                            canvas.update()
                        
                        # Save manual beats
                        self._save_manual_beats()
                        return
        
        # Original single beat labeling logic
        # Find the beat at this position and update its label
        for canvas in self._canvases:
            if canvas.lead_name == 'I':
                from PyQt5.QtCore import QPoint
                local_x = canvas.mapFrom(self._canvas_frame, QPoint(click_x, 0)).x()
                local_x = max(0, min(canvas.width(), local_x))
                # Get beat timestamp at this position
                w = canvas.width()
                if w > 0:
                    pct = local_x / float(w)
                    start_sec = canvas._start_sec
                    data_len = len(canvas._data) if hasattr(canvas, '_data') else 0
                    if data_len > 0:
                        end_sec = start_sec + data_len / canvas._fs
                        click_ts = start_sec + pct * (end_sec - start_sec)
                        
                        # --- Snap click_ts to nearest detected R-peak ---
                        snapped_ts = click_ts
                        snap_tolerance_sec = 0.15  # 150ms tolerance
                        
                        # Try snapping to parent dialog's detected peaks first
                        if hasattr(self, '_detected_r_peaks') and self._detected_r_peaks:
                            best_dist = snap_tolerance_sec
                            for peak_ts in self._detected_r_peaks:
                                dist = abs(peak_ts - click_ts)
                                if dist < best_dist:
                                    best_dist = dist
                                    snapped_ts = peak_ts
                        
                        # Determine target timestamps: all selected beats, or just the clicked one
                        target_timestamps = []
                        if hasattr(canvas, '_selected_beats') and canvas._selected_beats:
                            target_timestamps = canvas._selected_beats
                        else:
                            target_timestamps = [snapped_ts]

                        # Ensure _beat_annotations list exists
                        if not hasattr(canvas, '_beat_annotations') or canvas._beat_annotations is None:
                            canvas._beat_annotations = []
                        
                        for target_ts in target_timestamps:
                            # Try to find existing annotation within tolerance
                            found = False
                            for beat in canvas._beat_annotations:
                                if abs(beat['timestamp'] - target_ts) < snap_tolerance_sec:
                                    beat['label'] = full_label
                                    beat['color'] = label_colors.get(label, "#FFFF00")
                                    beat['is_manual'] = True
                                    # Set marking_mode based on current selection mode
                                    if self._selection_mode == 'parallel_single':
                                        beat['marking_mode'] = 'parallel_single'
                                    print(f"[Full Disclosure] Updated beat at {target_ts:.3f}s as '{full_label}'")
                                    found = True
                                    break
                            
                            # If no existing annotation found, create one at the target position
                            if not found:
                                new_beat = {
                                    'timestamp': target_ts,
                                    'label': full_label,
                                    'color': label_colors.get(label, "#FFFF00"),
                                    'is_manual': True
                                }
                                # Set marking_mode based on current selection mode
                                if self._selection_mode == 'parallel_single':
                                    new_beat['marking_mode'] = 'parallel_single'
                                canvas._beat_annotations.append(new_beat)
                                print(f"[Full Disclosure] Created new beat annotation at {target_ts:.3f}s as '{full_label}'")
                                
                            # Update engine metrics for persistence across scrolling
                            if hasattr(self._engine, '_metrics'):
                                engine_beat_found = False
                                for m in self._engine._metrics:
                                    all_beats = m.get('all_beats', [])
                                    for engine_beat in all_beats:
                                        if abs(float(engine_beat.get('timestamp', 0.0)) - target_ts) < snap_tolerance_sec:
                                            engine_beat['label'] = full_label
                                            engine_beat['is_manual'] = True
                                            engine_beat_found = True
                                            break
                                    if engine_beat_found:
                                        break
                                
                                if not engine_beat_found and self._engine._metrics:
                                    added = False
                                    for m in self._engine._metrics:
                                        m_start = float(m.get('start_sec', 0.0))
                                        m_dur = float(m.get('duration', 3600.0))
                                        if m_start <= target_ts <= m_start + m_dur:
                                            m.setdefault('all_beats', []).append({
                                                'timestamp': target_ts,
                                                'label': full_label,
                                                'is_manual': True
                                            })
                                            added = True
                                            break
                                    if not added:
                                        self._engine._metrics[0].setdefault('all_beats', []).append({
                                            'timestamp': target_ts,
                                            'label': full_label,
                                            'is_manual': True
                                        })
                                    # A beat was added -> cached windowed index is stale
                                    self._beat_index_dirty = True
                        
                        # Keep sorted by timestamp for consistent rendering
                        canvas._beat_annotations.sort(key=lambda b: b['timestamp'])
                        
                        # Propagate the updated beat_annotations to ALL other canvases
                        # so that QRS peak colors are updated across every lead strip.
                        for c in self._canvases:
                            if c is not canvas:
                                if not hasattr(c, '_beat_annotations') or c._beat_annotations is None:
                                    c._beat_annotations = []
                                
                                for target_ts in target_timestamps:
                                    # Update or insert the annotation for this beat in each canvas
                                    beat_found_in_c = False
                                    for b in c._beat_annotations:
                                        if abs(b['timestamp'] - target_ts) < snap_tolerance_sec:
                                            b['label'] = full_label
                                            b['color'] = label_colors.get(label, "#FFFF00")
                                            b['is_manual'] = True
                                            beat_found_in_c = True
                                            break
                                    if not beat_found_in_c:
                                        c._beat_annotations.append({
                                            'timestamp': target_ts,
                                            'label': full_label,
                                            'color': label_colors.get(label, "#FFFF00"),
                                            'is_manual': True
                                        })

                                c._beat_annotations.sort(key=lambda b: b['timestamp'])
                            c.update()
                break
        
        # Persist all manual beat edits to disk so the report generator can read them
        self._save_manual_beats()
        
        # Update the arrhythmia status label at the bottom to reflect the newly marked beats
        self._update_time_and_arrhythmia_labels(self._current_start, self._current_start + self._window_sec)
    
    def _save_manual_beats(self):
        """Write all manually-edited beat annotations to manual_beats.json in the session dir."""
        import json
        try:
            session_dir = os.path.dirname(self._engine.ecgh_path)
            manual_beats_path = os.path.join(session_dir, 'manual_beats.json')
            
            # Collect all beats from engine._metrics and lead_i_canvas
            all_beats = []
            label_colors_map = {
                "S": "#00FFFF",      # Supraventricular - Cyan
                "P": "#FF00FF",      # Premature - Magenta
                "V": "#FF3333",      # Ventricular - Red
                "C": "#FFA500",      # Conduction - Orange
                "T": "#9932CC",      # TV Paced - Dark Orchid (Purple)
                "A": "#FFFF00",      # ACLS - Yellow
                "N": "#00FF00",      # Normal - Green (legacy)
                "X": "#0000FF",      # Artifact - Blue (legacy)
            }
            
            seen = set()
            
            # 1. First add from engine._metrics across whole recording
            if hasattr(self._engine, '_metrics') and self._engine._metrics:
                for m in self._engine._metrics:
                    for b in m.get('all_beats', []):
                        ts = float(b.get('timestamp', 0.0))
                        lbl = str(b.get('label', 'N'))
                        is_man = b.get('is_manual', False) or (b.get('marking_mode') is not None) or (b.get('batch_id') is not None)
                        if is_man:
                            seen.add(ts)
                            label_code = lbl
                            if '(' in lbl and ')' in lbl:
                                label_code = lbl.split('(')[1].split(')')[0]
                            beat_color = b.get('color') or label_colors_map.get(label_code, '#FFFF00')
                            beat_entry = {
                                'timestamp': ts,
                                'label': lbl,
                                'color': beat_color,
                                'is_manual': True
                            }
                            if b.get('batch_id') is not None:
                                beat_entry['batch_id'] = b.get('batch_id')
                            if b.get('marking_mode'):
                                beat_entry['marking_mode'] = b.get('marking_mode')
                            all_beats.append(beat_entry)

            # 2. Also check Lead-I canvas annotations for any beats in active view
            lead_i_canvas = None
            for c in self._canvases:
                if c.lead_name == 'I':
                    lead_i_canvas = c
                    break

            if lead_i_canvas and hasattr(lead_i_canvas, '_beat_annotations'):
                for b in lead_i_canvas._beat_annotations:
                    ts = float(b.get('timestamp', 0.0))
                    if ts not in seen:
                        lbl = str(b.get('label', 'N'))
                        is_man = b.get('is_manual', False) or (b.get('marking_mode') is not None) or (b.get('batch_id') is not None)
                        if is_man:
                            seen.add(ts)
                            saved_color = b.get('color', '')
                            if saved_color:
                                color = str(saved_color)
                            else:
                                label_code = lbl
                                if '(' in lbl and ')' in lbl:
                                    label_code = lbl.split('(')[1].split(')')[0]
                                color = label_colors_map.get(label_code, '#FFFF00')
                            
                            beat_entry = {
                                'timestamp': ts,
                                'label': lbl,
                                'color': color,
                                'is_manual': True
                            }
                            if b.get('batch_id') is not None:
                                beat_entry['batch_id'] = b.get('batch_id')
                            if b.get('marking_mode'):
                                beat_entry['marking_mode'] = b.get('marking_mode')
                            all_beats.append(beat_entry)

            
            all_beats.sort(key=lambda b: b['timestamp'])
            with open(manual_beats_path, 'w') as f:
                json.dump(all_beats, f, indent=2)
            print(f"[Full Disclosure] Saved {len(all_beats)} manual beats to {manual_beats_path}")
        except Exception as e:
            print(f"[Full Disclosure] Could not save manual beats: {e}")
    
    def _delete_beat_at_position(self, click_x: int):
        """Delete the beat at the clicked position."""
        self._snapshot_state('delete_beat')
        
        # Parallel multi mode: delete all beats between two vertical lines (if they exist)
        # OR delete all manually marked beats in the current window (after dialog reopens)
        if self._selection_mode == 'parallel_multi':
            # Try to use existing line positions if they're set
            if self._multi_line1_x is not None and self._multi_line2_x is not None:
                # Lines are currently visible - use them
                line1_x = self._multi_line1_x
                line2_x = self._multi_line2_x
                start_x = min(line1_x, line2_x)
                end_x = max(line1_x, line2_x)
                
                # Get reference canvas (Lead I)
                ref_canvas = None
                for canvas in self._canvases:
                    if canvas.lead_name == 'I':
                        ref_canvas = canvas
                        break
                
                if ref_canvas:
                    from PyQt5.QtCore import QPoint
                    w = ref_canvas.width()
                    if w > 0:
                        start_x_local = ref_canvas.mapFrom(self._canvas_frame, QPoint(start_x, 0)).x()
                        end_x_local = ref_canvas.mapFrom(self._canvas_frame, QPoint(end_x, 0)).x()
                        start_x_local = max(0, min(w, start_x_local))
                        end_x_local = max(0, min(w, end_x_local))
                        
                        # Convert to timestamps
                        start_sec = ref_canvas._start_sec
                        data_len = len(ref_canvas._data) if hasattr(ref_canvas, '_data') else 0
                        if data_len > 0:
                            end_sec = start_sec + data_len / ref_canvas._fs
                            start_ts = start_sec + (start_x_local / w) * (end_sec - start_sec)
                            end_ts = start_sec + (end_x_local / w) * (end_sec - start_sec)
                            
                            # Delete all beats in this range for all leads
                            snap_tolerance_sec = 0.15
                            deleted_timestamps = set()
                            deleted_batch_ids = set()
                            
                            print(f"[Full Disclosure] START DELETE: time range {start_ts:.3f}s to {end_ts:.3f}s")
                            
                            for canvas in self._canvases:
                                if hasattr(canvas, '_beat_annotations') and canvas._beat_annotations:
                                    print(f"[Full Disclosure] {canvas.lead_name}: {len(canvas._beat_annotations)} beats before delete")
                                    
                                    # Collect kept beats and track deleted timestamps
                                    # For parallel multi delete: delete beats with matching batch_id OR is_manual in range
                                    kept_beats = []
                                    for b in canvas._beat_annotations:
                                        ts = b.get('timestamp', 0.0)
                                        is_manual = b.get('is_manual', False)
                                        batch_id = b.get('batch_id')
                                        lbl = b.get('label', 'N')
                                        
                                        # Delete if within range AND (manually marked OR has batch_id from parallel marking)
                                        should_delete = start_ts <= ts <= end_ts and (is_manual or batch_id is not None)
                                        
                                        if should_delete:
                                            deleted_timestamps.add(ts)
                                            if batch_id is not None:
                                                deleted_batch_ids.add(batch_id)
                                            print(f"[Full Disclosure] DELETE {canvas.lead_name}: beat at {ts:.3f}s label={lbl} is_manual={is_manual} batch_id={batch_id}")
                                        else:
                                            kept_beats.append(b)
                                    
                                    canvas._beat_annotations[:] = kept_beats
                                    print(f"[Full Disclosure] {canvas.lead_name}: {len(canvas._beat_annotations)} beats after delete")
                                    canvas.update()
                            
                            # CRITICAL FIX: Also remove deleted beats from _pending_manual_beats
                            # Otherwise _update_canvases will restore them from the pending list
                            if hasattr(self, '_pending_manual_beats') and self._pending_manual_beats:
                                self._pending_manual_beats[:] = [
                                    mb for mb in self._pending_manual_beats
                                    if not any(abs(mb.get('timestamp', 0.0) - t_ts) < snap_tolerance_sec for t_ts in deleted_timestamps)
                                ]
                                print(f"[Full Disclosure] Removed {len(deleted_timestamps)} deleted beats from pending manual beats")
                            
                            # CRITICAL FIX #2: Also remove deleted beats from self._engine._metrics
                            # Otherwise _beats_in_window will return them from the cached beat list
                            if hasattr(self._engine, '_metrics') and self._engine._metrics:
                                for m in self._engine._metrics:
                                    if 'all_beats' in m:
                                        m['all_beats'][:] = [
                                            b for b in m['all_beats']
                                            if not any(abs(b.get('timestamp', 0.0) - t_ts) < snap_tolerance_sec for t_ts in deleted_timestamps)
                                        ]
                                print(f"[Full Disclosure] Removed {len(deleted_timestamps)} deleted beats from engine metrics")
                            
                            # Mark beat index dirty so it rebuilds from updated metrics
                            self._beat_index_dirty = True
                            
                            # Also remove manual structured events in this range (auto-detected badges protected)
                            if hasattr(self._engine, '_structured_events'):
                                self._engine._structured_events[:] = [
                                    ev for ev in self._engine._structured_events
                                    if not (
                                        ev.get('timestamp', 0) < end_ts and 
                                        ev.get('end_timestamp', ev.get('timestamp', 0)) > start_ts and
                                        (ev.get('is_manual', False) or ev.get('source') in ['manual', 'manual_parallel_multi', 'restored_parallel_multi'])
                                    )
                                ]
                                self._event_index_dirty = True
                                print(f"[Full Disclosure] Removed manual structured events overlapping with range {start_ts:.3f}s to {end_ts:.3f}s")
                            
                            print(f"[Full Disclosure] Deleted all beats between {start_ts:.3f}s and {end_ts:.3f}s")
                            
                            # CRITICAL FIX: Save deletions to file BEFORE refreshing UI
                            # Otherwise _update_canvases will reload the old beats from the JSON file
                            self._save_manual_beats()
                            
                            # Now refresh UI after deletions are saved
                            self._update_canvases(self._current_start)
                            
                            # Clear vertical lines
                            self._multi_line1_x = None
                            self._multi_line2_x = None
                            if hasattr(self, '_vertical_line_overlay'):
                                self._vertical_line_overlay.clear_line()
                            return
            else:
                # Lines are not currently visible (dialog was reopened)
                # In this case, delete all manually marked beats in the current window
                # by finding beats marked with the batch_id or marking_mode from parallel_multi
                print("[Full Disclosure] Parallel multi mode delete: lines not visible, clearing manually marked beats in current window")
                
                ref_canvas = None
                for canvas in self._canvases:
                    if canvas.lead_name == 'I':
                        ref_canvas = canvas
                        break
                
                if ref_canvas and hasattr(ref_canvas, '_start_sec') and hasattr(ref_canvas, '_data') and hasattr(ref_canvas, '_fs'):
                    start_sec = ref_canvas._start_sec
                    data_len = len(ref_canvas._data) if hasattr(ref_canvas, '_data') else 0
                    if data_len > 0:
                        end_sec = start_sec + data_len / ref_canvas._fs
                        
                        # Delete all manually marked beats in current window
                        deleted_timestamps = set()
                        
                        for canvas in self._canvases:
                            if hasattr(canvas, '_beat_annotations') and canvas._beat_annotations:
                                # Find all manually marked beats (marked with is_manual=True)
                                beats_to_delete = []
                                for beat in canvas._beat_annotations:
                                    ts = float(beat.get('timestamp', 0.0))
                                    is_manual = beat.get('is_manual', False)
                                    marking_mode = beat.get('marking_mode', '')
                                    
                                    # Delete if:
                                    # 1. It's in the current visible window
                                    # 2. It was manually marked (is_manual=True)
                                    # 3. It was marked with parallel_multi mode (has batch_id or marking_mode)
                                    if (start_sec <= ts <= end_sec and is_manual and 
                                        (beat.get('batch_id') is not None or marking_mode == 'parallel_multi')):
                                        beats_to_delete.append(ts)
                                        deleted_timestamps.add(ts)
                                
                                # Remove marked beats
                                snap_tolerance_sec = 0.15
                                canvas._beat_annotations[:] = [
                                    b for b in canvas._beat_annotations 
                                    if not any(abs(b['timestamp'] - t_ts) < snap_tolerance_sec for t_ts in beats_to_delete)
                                ]
                                canvas.update()
                        
                        # CRITICAL FIX: Also remove deleted beats from _pending_manual_beats
                        if hasattr(self, '_pending_manual_beats') and self._pending_manual_beats:
                            self._pending_manual_beats[:] = [
                                mb for mb in self._pending_manual_beats
                                if not any(abs(mb.get('timestamp', 0.0) - t_ts) < 0.15 for t_ts in deleted_timestamps)
                            ]
                            print(f"[Full Disclosure] Removed {len(deleted_timestamps)} deleted beats from pending manual beats")
                        
                        # CRITICAL FIX #2: Also remove deleted beats from self._engine._metrics
                        # Otherwise _beats_in_window will return them from the cached beat list
                        if hasattr(self._engine, '_metrics') and self._engine._metrics:
                            for m in self._engine._metrics:
                                if 'all_beats' in m:
                                    m['all_beats'][:] = [
                                        b for b in m['all_beats']
                                        if not any(abs(b.get('timestamp', 0.0) - t_ts) < 0.15 for t_ts in deleted_timestamps)
                                    ]
                            print(f"[Full Disclosure] Removed {len(deleted_timestamps)} deleted beats from engine metrics")
                        
                        # Mark beat index dirty so it rebuilds from updated metrics
                        self._beat_index_dirty = True
                        
                        # Also remove corresponding manual structured events (auto-detected badges protected)
                        if hasattr(self._engine, '_structured_events') and self._engine._structured_events:
                            # Remove ONLY manual structured events in this range
                            self._engine._structured_events[:] = [
                                ev for ev in self._engine._structured_events
                                if not (
                                    ev.get('timestamp', 0) >= start_sec and 
                                    ev.get('timestamp', 0) <= end_sec and
                                    (ev.get('is_manual', False) or ev.get('source') in ['manual', 'manual_parallel_multi', 'restored_parallel_multi'])
                                )
                            ]
                            self._event_index_dirty = True
                            print(f"[Full Disclosure] Removed manual structured events in window {start_sec:.3f}s to {end_sec:.3f}s")
                        
                        print(f"[Full Disclosure] Deleted manually marked beats in window {start_sec:.3f}s to {end_sec:.3f}s")
                        
                        # CRITICAL FIX: Save BEFORE refreshing UI (same as first delete path)
                        self._save_manual_beats()
                        
                        # Now refresh UI after deletions are saved
                        if hasattr(self._engine, '_structured_events'):
                            self._update_canvases(self._current_start)
                        return
        
        # Original single beat deletion logic
        for canvas in self._canvases:
            if canvas.lead_name == 'I':
                from PyQt5.QtCore import QPoint
                local_x = canvas.mapFrom(self._canvas_frame, QPoint(click_x, 0)).x()
                local_x = max(0, min(canvas.width(), local_x))
                w = canvas.width()
                if w > 0 and hasattr(canvas, '_beat_annotations'):
                    pct = local_x / float(w)
                    start_sec = canvas._start_sec
                    data_len = len(canvas._data) if hasattr(canvas, '_data') else 0
                    if data_len > 0:
                        end_sec = start_sec + data_len / canvas._fs
                        click_ts = start_sec + pct * (end_sec - start_sec)
                        
                        # --- Snap click_ts to nearest detected R-peak ---
                        snapped_ts = click_ts
                        snap_tolerance_sec = 0.15
                        if hasattr(self, '_detected_r_peaks') and self._detected_r_peaks:
                            best_dist = snap_tolerance_sec
                            for peak_ts in self._detected_r_peaks:
                                dist = abs(peak_ts - click_ts)
                                if dist < best_dist:
                                    best_dist = dist
                                    snapped_ts = peak_ts

                        # Determine target timestamps: all selected beats, or just the clicked one
                        target_timestamps = []
                        if hasattr(canvas, '_selected_beats') and canvas._selected_beats:
                            target_timestamps = canvas._selected_beats
                        else:
                            target_timestamps = [snapped_ts]
                            
                        # Remove the target beats from ALL canvases
                        for c in self._canvases:
                            if hasattr(c, '_beat_annotations') and c._beat_annotations:
                                # Keep beats that DO NOT match any of the target timestamps
                                c._beat_annotations[:] = [
                                    b for b in c._beat_annotations 
                                    if not any(abs(b['timestamp'] - t_ts) < snap_tolerance_sec for t_ts in target_timestamps)
                                ]
                            c.update()
                            
                        # Remove from engine metrics for persistence across scrolling
                        if hasattr(self._engine, '_metrics'):
                            for m in self._engine._metrics:
                                if 'all_beats' in m:
                                    m['all_beats'] = [
                                        eb for eb in m['all_beats']
                                        if not any(abs(float(eb.get('timestamp', 0.0)) - t_ts) < snap_tolerance_sec for t_ts in target_timestamps)
                                    ]
                            # Beats were removed -> cached windowed index is stale
                            self._beat_index_dirty = True
                                    
                        # --- Also allow deleting MANUAL Arrhythmia Regions (Auto-detected badges protected) ---
                        if hasattr(self._engine, '_structured_events') and self._engine._structured_events:
                            # Find if click_ts falls inside an active arrhythmia region
                            events = self._engine._structured_events
                            to_remove = None
                            for i, ev in enumerate(events):
                                ev_ts = float(ev.get('timestamp', 0.0) or 0.0)
                                if ev_ts <= click_ts:
                                    next_ts = float(events[i+1].get('timestamp', 0.0) or 0.0) if i+1 < len(events) else (start_sec + 3600*24)
                                    if click_ts < next_ts:
                                        # Only allow deleting manually created structured events/badges
                                        is_manual_ev = ev.get('is_manual', False) or ev.get('source') in ['manual', 'manual_parallel_multi', 'restored_parallel_multi']
                                        if is_manual_ev:
                                            to_remove = ev
                                        break
                                        
                            if to_remove:
                                events.remove(to_remove)
                                self._event_index_dirty = True
                                print(f"[Full Disclosure] Deleted manual arrhythmia event: {to_remove.get('label')} at {to_remove.get('timestamp')}")
                                
                                # Re-update the canvases so the region coloring is removed immediately
                                self._update_canvases(self._current_start)
                                
        # Persist the deletion to disk
        self._save_manual_beats()

    def _rebuild_beat_index(self):
        """Merge all_beats across every engine metric into one timestamp-sorted
        list + parallel numpy array, so windowed lookups can use bisect instead
        of a full linear scan. Call is cheap to skip via the dirty flag - only
        actually rebuilds when a beat was added/removed/restored since last time."""
        merged = []
        if hasattr(self._engine, '_metrics') and self._engine._metrics:
            for m in self._engine._metrics:
                merged.extend(m.get('all_beats', []))
        merged.sort(key=lambda b: float(b.get('timestamp', 0.0)))
        self._cached_beat_list = merged
        self._cached_beat_ts = np.array(
            [float(b.get('timestamp', 0.0)) for b in merged], dtype=np.float64
        )
        self._beat_index_dirty = False

    def _beats_in_window(self, start_sec, end_sec):
        """O(log n + k) windowed beat lookup (k = beats actually in range)."""
        if self._beat_index_dirty:
            self._rebuild_beat_index()
        if len(self._cached_beat_ts) == 0:
            return []
        import bisect
        lo = bisect.bisect_left(self._cached_beat_ts, start_sec)
        hi = bisect.bisect_right(self._cached_beat_ts, end_sec)
        return self._cached_beat_list[lo:hi]

    def _rebuild_event_index(self):
        """Same idea as _rebuild_beat_index but for structured (arrhythmia) events."""
        events = list(getattr(self._engine, '_structured_events', []) or [])
        events.sort(key=lambda e: float(e.get('timestamp', 0.0) or 0.0))
        self._cached_event_list = events
        self._cached_event_ts = np.array(
            [float(e.get('timestamp', 0.0) or 0.0) for e in events], dtype=np.float64
        )
        self._event_index_dirty = False

    def _events_in_window(self, start_sec, end_sec):
        """O(log n + k) windowed structured-event lookup."""
        if self._event_index_dirty:
            self._rebuild_event_index()
        if len(self._cached_event_ts) == 0:
            return []
        import bisect
        lo = bisect.bisect_left(self._cached_event_ts, start_sec)
        hi = bisect.bisect_right(self._cached_event_ts, end_sec)
        return self._cached_event_list[lo:hi]

    def get_events_with_manual_priority(self, start_sec, end_sec):
        """
        Get all events in the window with MANUAL PRIORITY over AUTO-DETECTION.
        
        This method:
        1. Collects all manually marked beats (is_manual=True) in the time window
        2. Filters out any auto-detected events that overlap with manual marks (150ms tolerance)
        3. Returns combined list with manual marks taking precedence
        
        Returns:
            List[dict]: Combined events with manual marks suppressing overlapping auto marks
        """
        SUPPRESS_TOLERANCE_SEC = 0.15
        
        # 1. Collect manually marked beats
        manual_events = []
        manual_timestamps = set()
        lead_i_canvas = None
        if hasattr(self, '_canvases'):
            for c in self._canvases:
                if c.lead_name == 'I':
                    lead_i_canvas = c
                    break
        
        if lead_i_canvas and hasattr(lead_i_canvas, '_beat_annotations') and lead_i_canvas._beat_annotations:
            for b in lead_i_canvas._beat_annotations:
                ts = float(b.get('timestamp', 0.0))
                lbl = b.get('label', 'N')
                is_manual = b.get('is_manual', False)
                if lbl != 'N' and start_sec <= ts <= end_sec and is_manual:
                    manual_events.append({
                        'timestamp': ts,
                        'label': lbl,
                        'source': 'Manual',
                        'is_manual': True
                    })
                    manual_timestamps.add(round(ts, 2))
        
        # 2. Collect auto-detected events, suppressing those that overlap with manual marks
        auto_events = []
        for ev in self._events_in_window(start_sec, end_sec):
            ts = float(ev.get('timestamp', 0.0) or 0.0)
            
            # Check if this auto event falls within tolerance of any manual mark
            is_suppressed = False
            for manual_ts in manual_timestamps:
                if abs(ts - manual_ts) < SUPPRESS_TOLERANCE_SEC:
                    is_suppressed = True
                    break
            
            if not is_suppressed:
                auto_events.append({
                    'timestamp': ts,
                    'label': ev.get('label', 'Event'),
                    'source': 'Auto',
                    'is_manual': False
                })
        
        # Combine and return sorted by timestamp
        return sorted(manual_events + auto_events, key=lambda x: x['timestamp'])

    # Max plausible duration of a single colored arrhythmia region. Padding the
    # window query backward by this much catches events that *started* just
    # before the visible range but still extend into it, without falling back
    # to scanning the whole recording's event list (which is what previously
    # made ECGStripCanvas.paintEvent's structured-event loop scale with total
    # recording length instead of window size - the actual cause of drag lag
    # on large recordings).
    _EVENT_REGION_PAD_SEC = 600.0

    def _structured_events_for_canvas(self, start_sec, end_sec):
        """Structured events to hand to the canvases for this window only.

        Reuses the same sorted/cached index as _events_in_window but pads the
        lower bound so a long-running event that started slightly earlier
        still gets colored in, while still being bounded (not the full
        recording's event list) for large Holter recordings.
        """
        if self._event_index_dirty:
            self._rebuild_event_index()
        if len(self._cached_event_ts) == 0:
            return []
        import bisect
        lo = bisect.bisect_left(self._cached_event_ts, start_sec - self._EVENT_REGION_PAD_SEC)
        hi = bisect.bisect_right(self._cached_event_ts, end_sec)
        return self._cached_event_list[lo:hi]

    def _scroll_throttle_interval(self):
        """Minimum seconds between live redraws while dragging, scaled to window size."""
        win = getattr(self, '_window_sec', self._BASE_WIN_SEC)
        if win <= 15.0:
            return 0.030   # Full disc / 30 Sec — ~33 FPS, very cheap to redraw
        elif win <= 65.0:
            return 0.045   # 1 Min — a bit more data per frame
        else:
            return 0.060   # 2 Min — heaviest window, throttle a little more

    def _on_scrollbar_pressed(self):
        self._is_scrubbing = True

    def _on_scrollbar_released(self):
        self._is_scrubbing = False
        # Do one full-detail, un-throttled pass at the final position so the
        # arrhythmia label / segment overlay are guaranteed accurate once the
        # user lets go, even if a scrub frame was skipped right before release.
        self._pending_scroll_val = self.time_scrollbar.value()
        self._process_scroll_update(force_extras=True)

    def _on_scrollbar_moved(self, val):
        self._pending_scroll_val = val
        if self._scroll_timer.isActive():
            return  # a frame is already queued - it will pick up the latest value
        import time
        now = time.time()
        interval = self._scroll_throttle_interval()
        remaining = interval - (now - self._last_scroll_time)
        # Always hand off to the event loop via singleShot(0, ...) instead of
        # calling _process_scroll_update() directly here. Direct calls run
        # inside QScrollBar's own mouse-move handling, so a slow frame blocks
        # the same call stack that's supposed to move the handle - that's
        # what causes the wave redraw to fall behind the mouse and jump/catch
        # up instead of tracking it smoothly. Deferring by 0ms still runs
        # essentially immediately when we're not throttling, but lets Qt
        # process the pending mouse-move/paint events first.
        delay_ms = 0 if remaining <= 0 else int(remaining * 1000)
        self._scroll_timer.start(max(0, delay_ms))

    def _on_scroll_timer_timeout(self):
        self._process_scroll_update()

    def _process_scroll_update(self, force_extras=False):
        if self._pending_scroll_val is None:
            return
        val = self._pending_scroll_val
        self._pending_scroll_val = None
        import time
        now = time.time()
        self._last_scroll_time = now

        start_sec = float(val) / 100.0

        # Clear any selected beats and vertical lines when scrolling
        self._drag_start_x = None
        self._drag_current_x = None
        self._is_dragging = False
        self._drag_start_timestamp = None

        # Clear vertical lines from overlay
        if hasattr(self, '_vertical_line_overlay'):
            self._vertical_line_overlay.clear_line()

        # Clear beat selection from all canvases
        for canvas in self._canvases:
            canvas._clicked_beat_timestamp = None
            canvas._clicked_beat_label = None
            canvas._clicked_beat_x_pos = None
            canvas._selected_beats = []
            canvas.update()

        # While actively scrubbing, only recompute the arrhythmia/event label
        # and reposition the segment overlay at a reduced cadence - both do
        # extra list building/sorting on top of the waveform redraw, and
        # skipping them on in-between frames keeps each frame cheap so the
        # wave itself stays in sync with the handle. They're always brought
        # up to date on release (see _on_scrollbar_released) and whenever
        # we're not scrubbing (tab change, programmatic seek, etc).
        do_extras = force_extras or (not self._is_scrubbing) or \
            (now - self._last_extras_time >= self._EXTRAS_MIN_INTERVAL)

        self._update_canvases(start_sec, update_extras=do_extras)

        if do_extras:
            self._last_extras_time = now

    def _on_time_tab_changed(self, index):
        text = self.time_tabs.tabText(index)
        if "30 Sec" in text: self._window_sec = 30.0
        elif "1 Min" in text: self._window_sec = 60.0
        elif "2 Min" in text: self._window_sec = 120.0
        # elif "5 Min" in text: self._window_sec = 300.0
        # elif "10 Min" in text: self._window_sec = 600.0
        # elif "15 Min" in text: self._window_sec = 900.0
        else: self._window_sec = self._BASE_WIN_SEC * (25.0 / self._paper_speed)
        
        # Clear any selected beats and vertical lines when tab changes
        self._drag_start_x = None
        self._drag_current_x = None
        self._is_dragging = False
        self._drag_start_timestamp = None
        
        # Clear vertical lines from overlay
        if hasattr(self, '_vertical_line_overlay'):
            self._vertical_line_overlay.clear_line()
        
        # Clear beat selection from all canvases
        for canvas in self._canvases:
            canvas._clicked_beat_timestamp = None
            canvas._clicked_beat_label = None
            canvas._clicked_beat_x_pos = None
            canvas._selected_beats = []
            canvas.update()
        
        # TODO: Overlay mouse enable disabled for now
        # Always enable mouse on overlay for dragging
        # self.overlay.set_mouse_enabled(True)
        
        # Preserve current start position, just ensure it's within bounds
        max_start = max(0.0, self._engine.duration_sec - self._window_sec)
        self._current_start = max(0.0, min(self._current_start, max_start))
        
        self._update_scrollbar_range()
        self.time_scrollbar.setValue(int(self._current_start * 100))
        self._update_canvases(self._current_start)
        
        self.lbl_dur.setText(f"Recording: {self._engine._sec_to_hms(self._engine.duration_sec)}")

    def _update_time_and_arrhythmia_labels(self, start_sec: float, end_sec: float):
        # Update real-time display
        if hasattr(self._engine, '_reader') and hasattr(self._engine._reader, 'start_time'):
            start_real = datetime.fromtimestamp(self._engine._reader.start_time + start_sec)
            end_real = datetime.fromtimestamp(self._engine._reader.start_time + end_sec)
            self.lbl_real_time.setText(f"Capture Frame Time: {start_real.strftime('%H:%M:%S')} - {end_real.strftime('%H:%M:%S')}")
        
        # Update arrhythmia indicator based on BADGE LABELS detected in the current window
        # Collect beats from current window and generate badges
        lead_i_canvas = None
        if hasattr(self, '_canvases'):
            for c in self._canvases:
                if c.lead_name == 'I':
                    lead_i_canvas = c
                    break
        
        arrhythmia_label = ""
        
        if lead_i_canvas and hasattr(lead_i_canvas, '_beat_annotations'):
            # Get beats in current window
            window_beats = [b for b in lead_i_canvas._beat_annotations 
                           if start_sec <= b.get('timestamp', 0.0) <= end_sec]
            window_events = []
            
            # Generate badges for current window using the same logic as badge display
            try:
                from ecg.holter.holter_summary_calc import get_template_beats_for_badges
                badges = get_template_beats_for_badges(window_beats, window_events)
                
                # Filter badges to show V, S, AF, P beats
                arrhythmia_badges = [b for b in badges if b.get('code') in ['V', 'S', 'AF', 'P']]
                
                if arrhythmia_badges:
                    # Sort by timestamp and get the earliest arrhythmia
                    arrhythmia_badges.sort(key=lambda x: x.get('timestamp', 0.0))
                    earliest = arrhythmia_badges[0]
                    
                    badge_name = earliest.get('name', '')
                    badge_ts = earliest.get('timestamp', 0.0)
                    
                    if hasattr(self._engine, '_reader') and hasattr(self._engine._reader, 'start_time'):
                        ts_real = datetime.fromtimestamp(self._engine._reader.start_time + badge_ts)
                        arrhythmia_label = f"Arrhythmia: {badge_name} at {ts_real.strftime('%H:%M:%S')}"
                    else:
                        arrhythmia_label = f"Arrhythmia: {badge_name}"
                elif window_beats:
                    # No arrhythmias detected, show Normal Sinus Rhythm if there are beats in the window
                    if hasattr(self._engine, '_reader') and hasattr(self._engine._reader, 'start_time'):
                        first_beat_ts = window_beats[0].get('timestamp', 0.0)
                        ts_real = datetime.fromtimestamp(self._engine._reader.start_time + first_beat_ts)
                        arrhythmia_label = f"Arrhythmia: Normal Sinus Rhythm at {ts_real.strftime('%H:%M:%S')}"
                    else:
                        arrhythmia_label = "Arrhythmia: Normal Sinus Rhythm"
                        
            except Exception as e:
                print(f"[Full Disclosure] Error updating arrhythmia label from badges: {e}")
                
        self.lbl_arrhythmia.setText(arrhythmia_label)

    def _update_canvases(self, start_sec: float, update_extras: bool = True):
        eff_dur = self._engine.duration_sec
        start_sec = max(0.0, min(start_sec, max(0.0, eff_dur - self._window_sec)))
        self._current_start = start_sec
        end_sec = start_sec + self._window_sec

        # The time label is cheap (string format only) so keep it live every
        # frame; the arrhythmia/event label rebuild involves gathering and
        # sorting events and is gated by update_extras (see _process_scroll_update).
        self.lbl_time.setText(f"Time:  {self._engine._sec_to_hms(start_sec)}")
        self._update_time_and_arrhythmia_labels(start_sec, end_sec)

        read_end_sec = min(end_sec, eff_dur)
        
        # Read data efficiently
        data = self._reader.read_range(start_sec, read_end_sec)
        expected_len = int(self._window_sec * self._engine.fs)
        
        # Get beat annotations for this window
        beat_annotations = []
        # Define color mapping for beat types
        label_colors = {
            "N": "#00FF00",      # Normal - Green
            "S": "#00FFFF",      # Supraventricular - Cyan
            "P": "#FF00FF",      # Premature - Magenta
            "V": "#FF3333",      # Ventricular - Red
            "C": "#FFA500",      # Conduction - Orange
            "T": "#9932CC",      # TV Paced - Purple
            "A": "#FFFF00",      # ACLS - Yellow
            "X": "#0000FF",      # Artifact - Blue
            "AF": "#FFA500",     # Atrial Fibrillation - Orange (legacy)
            "Other": "#FFFF00"   # Other - Yellow (legacy)
        }
        
        try:
            # Windowed lookup via cached sorted index (bisect) instead of
            # scanning every beat in the whole recording on every redraw.
            for beat in self._beats_in_window(start_sec, end_sec):
                ts = float(beat.get('timestamp', 0.0))
                lbl = str(beat.get('label', 'N'))
                # Extract short code from full label name (e.g., "Normal(N)" -> "N")
                short_code = lbl
                if '(' in lbl and ')' in lbl:
                    short_code = lbl.split('(')[1].split(')')[0]
                color = beat.get('color') or label_colors.get(short_code, "#FFFF00")
                beat_annotations.append({
                    'timestamp': ts,
                    'label': lbl,
                    'color': color,
                    'is_manual': beat.get('is_manual', False),
                    'marking_mode': beat.get('marking_mode'),
                    'batch_id': beat.get('batch_id')
                })
            # Already sorted (index is built sorted, and slicing preserves order)
        except Exception as e:
            print(f"[Full Disclosure] Error loading beat annotations: {e}")
        
        # Restore manual beats to canvas annotations if pending (on first load)
        if hasattr(self, '_pending_manual_beats') and self._pending_manual_beats:
            # Define color mapping for label codes (same as in _label_beat)
            beat_label_colors = {
                "S": "#00FFFF",      # Supraventricular - Cyan
                "P": "#FF00FF",      # Premature - Magenta
                "V": "#FF3333",      # Ventricular - Red
                "C": "#FFA500",      # Conduction - Orange
                "T": "#9932CC",      # TV Paced - Dark Orchid (Purple)
                "A": "#FFFF00",      # ACLS - Yellow
                "N": "#00FF00",      # Normal - Green (legacy)
                "X": "#0000FF",      # Artifact - Blue (legacy)
            }
            
            for c in self._canvases:
                if not hasattr(c, '_beat_annotations') or c._beat_annotations is None:
                    c._beat_annotations = []
                for mb in self._pending_manual_beats:
                    ts = float(mb.get('timestamp', 0.0))
                    lbl = str(mb.get('label', 'N'))
                    # Use saved color if available, otherwise derive from label
                    if mb.get('color'):
                        color = str(mb.get('color'))
                    else:
                        # Fallback: extract color from label code
                        # Label might be full like "Atrial Fibrillation 1(S)" or short like "S"
                        label_code = lbl
                        if '(' in lbl and ')' in lbl:
                            label_code = lbl.split('(')[1].split(')')[0]
                        color = beat_label_colors.get(label_code, '#FFFF00')
                    
                    # Check if beat already exists in canvas annotations
                    found = False
                    for b in c._beat_annotations:
                        if abs(b['timestamp'] - ts) < 0.15:
                            b['label'] = lbl
                            b['color'] = color
                            b['is_manual'] = True
                            if mb.get('marking_mode'):
                                b['marking_mode'] = mb.get('marking_mode')
                            if mb.get('batch_id') is not None:
                                b['batch_id'] = mb.get('batch_id')
                            found = True
                            break
                    if not found:
                        c._beat_annotations.append({
                            'timestamp': ts,
                            'label': lbl,
                            'color': color,
                            'is_manual': True,
                            'marking_mode': mb.get('marking_mode'),
                            'batch_id': mb.get('batch_id')
                        })
                c._beat_annotations.sort(key=lambda b: b['timestamp'])
            
            # Also restore structured events (for waveform coloring)
            # Group beats by batch_id to create structured events for parallel multi markings
            if not hasattr(self._engine, '_structured_events'):
                self._engine._structured_events = []
            
            batch_groups = {}  # batch_id -> list of beats
            for mb in self._pending_manual_beats:
                if mb.get('batch_id'):  # Only check batch_id
                    batch_id = mb.get('batch_id')
                    if batch_id not in batch_groups:
                        batch_groups[batch_id] = []
                    batch_groups[batch_id].append(mb)
            
            # CRITICAL FIX: Clear _pending_manual_beats after restoration
            # Otherwise _update_canvases will re-apply them on every refresh, causing deleted beats to reappear!
            self._pending_manual_beats = []
            
            # Define color mapping for all label codes (matching canvas rendering)
            beat_label_colors = {
                "S": "#00FFFF",      # Supraventricular - Cyan
                "P": "#FF00FF",      # Premature - Magenta
                "V": "#FF3333",      # Ventricular - Red
                "C": "#FFA500",      # Conduction - Orange
                "T": "#9932CC",      # TV Paced - Dark Orchid (Purple)
                "A": "#FFFF00",      # ACLS - Yellow
                "N": "#00FF00",      # Normal - Green (legacy)
                "X": "#0000FF",      # Artifact - Blue (legacy)
            }
            
            # Create structured events from batch groups
            for batch_id, beats in batch_groups.items():
                if beats:
                    timestamps = [float(b.get('timestamp', 0.0)) for b in beats]
                    start_ts = min(timestamps)
                    end_ts = max(timestamps)
                    lbl = beats[0].get('label', 'N')
                    color = beats[0].get('color', '')
                    
                    # Extract short code for structured event
                    label_code = lbl
                    if '(' in lbl and ')' in lbl:
                        label_code = lbl.split('(')[1].split(')')[0]
                    
                    # Ensure color is derived correctly if missing or default green
                    if not color or color in ('#00FF00', '#FFFF00', ''):
                        color = beat_label_colors.get(label_code, beats[0].get('color') or '#FFFF00')
                    
                    # Create structured event for waveform coloring
                    structured_event = {
                        'timestamp': start_ts,
                        'end_timestamp': end_ts,
                        'label': label_code,
                        'label_full': lbl,
                        'color': color,
                        'event_type': label_code,
                        'source': 'restored_parallel_multi'
                    }
                    
                    # Update existing event if found, or append new structured event
                    event_exists = False
                    for ev in self._engine._structured_events:
                        if (abs(float(ev.get('timestamp', 0.0)) - start_ts) < 0.5 and
                            abs(float(ev.get('end_timestamp', 0.0)) - end_ts) < 0.5):
                            ev['label'] = label_code
                            ev['label_full'] = lbl
                            ev['color'] = color
                            ev['event_type'] = label_code
                            ev['source'] = 'restored_parallel_multi'
                            event_exists = True
                            break
                    
                    if not event_exists:
                        self._engine._structured_events.append(structured_event)
            
            # Sort structured events by timestamp and rebuild event index immediately
            self._engine._structured_events.sort(key=lambda x: float(x.get('timestamp', 0.0)))
            self._event_index_dirty = True
            self._rebuild_event_index()
            
            # Restore vertical lines for ALL parallel multi markings (not just the first)
            # Process all batches to restore all parallel multi selections
            for batch_id, first_batch_beats in batch_groups.items():
                if first_batch_beats:
                    timestamps = [float(b.get('timestamp', 0.0)) for b in first_batch_beats]
                    min_ts = min(timestamps)
                    max_ts = max(timestamps)
                    
                    # Convert timestamps back to pixel positions
                    ref_canvas = None
                    for canvas in self._canvases:
                        if canvas.lead_name == 'I':
                            ref_canvas = canvas
                            break
                    
                    if ref_canvas and hasattr(ref_canvas, '_start_sec') and hasattr(ref_canvas, '_data') and hasattr(ref_canvas, '_fs'):
                        w = ref_canvas.width()
                        data_len = len(ref_canvas._data) if hasattr(ref_canvas, '_data') else 0
                        if w > 0 and data_len > 0:
                            end_sec = ref_canvas._start_sec + data_len / ref_canvas._fs
                            span = end_sec - ref_canvas._start_sec
                            
                            if span > 0:
                                from PyQt5.QtCore import QPoint
                                # Convert timestamps to pixel positions in canvas frame
                                min_pct = (min_ts - ref_canvas._start_sec) / span
                                max_pct = (max_ts - ref_canvas._start_sec) / span
                                
                                min_x_local = int(min_pct * w)
                                max_x_local = int(max_pct * w)
                                
                                # Convert to global frame coordinates
                                min_x_global = ref_canvas.mapTo(self._canvas_frame, QPoint(min_x_local, 0)).x()
                                max_x_global = ref_canvas.mapTo(self._canvas_frame, QPoint(max_x_local, 0)).x()
                                
                                # Restore the line positions (use the latest batch, which is typically the most relevant)
                                # For the UI, we'll show the lines of the first/most recent batch found
                                if self._multi_line1_x is None:  # Only set if not already set
                                    self._multi_line1_x = min_x_global
                                    self._multi_line2_x = max_x_global
                                    
                                    # Show the vertical lines
                                    if hasattr(self, '_vertical_line_overlay'):
                                        self._vertical_line_overlay.set_line_positions([min_x_global, max_x_global])
                                    
                                    lbl = first_batch_beats[0].get('label', 'N')
                                    print(f"[Full Disclosure] Restored parallel multi vertical lines for {lbl} at x={min_x_global}, x={max_x_global}")
            
            # CRITICAL FIX #2: Set selection mode to parallel_multi so delete works after reopening
            if batch_groups:
                self._selection_mode = 'parallel_multi'
                print(f"[Full Disclosure] Set _selection_mode to 'parallel_multi' for delete functionality")
                
                # Trigger canvas refresh to show all restored markings
                for canvas in self._canvases:
                    canvas.update()
                if hasattr(self, '_vertical_line_overlay'):
                    self._vertical_line_overlay.update()
            
            print(f"[Full Disclosure] Restored {len(self._pending_manual_beats)} manual beats to canvas annotations")
            print(f"[Full Disclosure] Restored {len(batch_groups)} structured events for parallel multi markings")
            self._pending_manual_beats = None  # Clear after restoration

        
        # Generate time array for ECG strips
        x = np.linspace(0, self._window_sec, expected_len)

        # Was: getattr(self._engine, '_structured_events', []) — the FULL
        # recording's event list handed to every one of the 12 canvases.
        # ECGStripCanvas.paintEvent() loops over every item in this list on
        # every repaint; passing the whole recording made that loop (and the
        # scrub redraw) scale with total recording length instead of the
        # visible window. Windowing it here is the actual fix for the lag
        # on large recordings.
        visible_structured_events = self._structured_events_for_canvas(start_sec, end_sec)

        # Optimization: Process data in parallel using numpy vectorization
        for i, c in enumerate(self._canvases):
            if i < data.shape[0] and data.shape[1] > 0:
                d_i = data[i]
                # Use numpy slicing instead of padding for better performance
                if len(d_i) != expected_len:
                    if len(d_i) > expected_len:
                        d_i = d_i[:expected_len]
                    else:
                        # Create padded array more efficiently
                        padded = np.empty(expected_len, dtype=d_i.dtype)
                        padded[:len(d_i)] = d_i
                        padded[len(d_i):] = d_i[-1] if len(d_i) > 0 else 0
                        d_i = padded
                # Convert to float32 for faster rendering and pass beat annotations
                c.set_data(x, np.asarray(d_i, dtype=np.float32), beat_annotations=beat_annotations, start_sec=start_sec, structured_events=visible_structured_events, fast_preview=not update_extras)
            else:
                c.set_data(x, np.zeros(expected_len, dtype=np.float32), beat_annotations=beat_annotations, start_sec=start_sec, structured_events=visible_structured_events, fast_preview=not update_extras)

        # Clear and repopulate segment overlay with only visible segments.
        # CRITICAL FIX: ALWAYS update segment positions on scroll, not just when update_extras=True
        # Otherwise segments appear to move with scrolling instead of staying anchored to timestamps
        if hasattr(self, '_segment_overlay') and hasattr(self, '_segment_annotations'):
            self._segment_overlay.clear_segments()
            ref = self._canvases[0] if self._canvases else None
            if ref and hasattr(ref, '_start_sec') and hasattr(ref, '_data') and hasattr(ref, '_fs'):
                from PyQt5.QtCore import QPoint
                ref_w  = ref.width()
                data_l = len(ref._data)
                if ref_w > 0 and data_l > 0:
                    ref_end = ref._start_sec + data_l / ref._fs
                    span    = ref_end - ref._start_sec

                    for seg in self._segment_annotations:
                        s_sec = seg['start_sec']
                        e_sec = seg['end_sec']
                        # Check overlap with visible range
                        if max(s_sec, ref._start_sec) <= min(e_sec, ref_end):
                            if span > 0:
                                sx_local = int(((s_sec - ref._start_sec) / span) * ref_w)
                                ex_local = int(((e_sec - ref._start_sec) / span) * ref_w)
                                sx_local = max(0, min(ref_w, sx_local))
                                ex_local = max(0, min(ref_w, ex_local))
                                sx_global = ref.mapTo(self._canvas_frame, QPoint(sx_local, 0)).x()
                                ex_global = ref.mapTo(self._canvas_frame, QPoint(ex_local, 0)).x()

                                self._segment_overlay.add_segment(
                                    sx_global, ex_global, s_sec, e_sec,
                                    seg['label'], seg['color'], seg.get('start_time_str', ''), seg.get('end_time_str', '')
                                )
            # Ensure overlay is updated after segments are added
            self._segment_overlay.update()
            # Raise overlay to ensure it's visible on top of canvases
            self._segment_overlay.raise_()

    def _refresh_segment_overlay(self):
        """Force refresh of segment overlay to ensure loaded segments are visible."""
        if hasattr(self, '_segment_overlay') and hasattr(self, '_segment_annotations'):
            print(f"[Full Disclosure] _refresh_segment_overlay called with {len(self._segment_annotations)} segments")
            # Update overlay geometry to match current canvas frame size
            if hasattr(self, '_canvas_frame'):
                self._segment_overlay.setGeometry(self._canvas_frame.rect())
            
            self._segment_overlay.clear_segments()
            ref = self._canvases[0] if self._canvases else None
            if ref and hasattr(ref, '_start_sec') and hasattr(ref, '_data') and hasattr(ref, '_fs'):
                from PyQt5.QtCore import QPoint
                ref_w  = ref.width()
                data_l = len(ref._data)
                print(f"[Full Disclosure] Canvas ref_w={ref_w}, data_l={data_l}, _start_sec={ref._start_sec}")
                if ref_w > 0 and data_l > 0:
                    ref_end = ref._start_sec + data_l / ref._fs
                    span    = ref_end - ref._start_sec
                    print(f"[Full Disclosure] Visible range: {ref._start_sec:.2f} to {ref_end:.2f}, span={span:.2f}")
                    
                    added_count = 0
                    for seg in self._segment_annotations:
                        s_sec = seg['start_sec']
                        e_sec = seg['end_sec']
                        # Check overlap with visible range
                        if max(s_sec, ref._start_sec) <= min(e_sec, ref_end):
                            if span > 0:
                                sx_local = int(((s_sec - ref._start_sec) / span) * ref_w)
                                ex_local = int(((e_sec - ref._start_sec) / span) * ref_w)
                                sx_local = max(0, min(ref_w, sx_local))
                                ex_local = max(0, min(ref_w, ex_local))
                                sx_global = ref.mapTo(self._canvas_frame, QPoint(sx_local, 0)).x()
                                ex_global = ref.mapTo(self._canvas_frame, QPoint(ex_local, 0)).x()
                                
                                self._segment_overlay.add_segment(
                                    sx_global, ex_global, s_sec, e_sec,
                                    seg['label'], seg['color'], seg.get('start_time_str', ''), seg.get('end_time_str', '')
                                )
                                added_count += 1
                    print(f"[Full Disclosure] Added {added_count} segments to overlay")
                else:
                    print(f"[Full Disclosure] Canvas not ready: ref_w={ref_w}, data_l={data_l}")
            else:
                print(f"[Full Disclosure] Canvas ref not available or missing attributes")
            self._segment_overlay.update()
        else:
            print(f"[Full Disclosure] _refresh_segment_overlay: overlay or annotations not available")



    def _set_tool_mode(self, tool_id: str, btn: "QPushButton"):
        """Activate a tool (ruler/caliper/magnify) on all canvases, or deactivate if already active."""
        tool_btns = [self.btn_ruler, self.btn_caliper, self.btn_magnify]
        if self._active_tool == tool_id:
            # Toggle off -- return to select mode
            self._active_tool = TOOL_SELECT
            self._active_tool_btn = None
            for b in tool_btns:
                b.setChecked(False)
        else:
            self._active_tool = tool_id
            self._active_tool_btn = btn
            for b in tool_btns:
                b.setChecked(b is btn)
        for c in self._canvases:
            if hasattr(c, 'set_mode'):
                c.set_mode(self._active_tool)
        # TODO: Show/hide the strip selection overlay based on tool (disabled for now)
        # Show/hide the strip selection overlay based on tool
        # When a measurement tool is active, hide the selection box
        # if self._active_tool == TOOL_SELECT:
        #     self.overlay.show()
        # else:
        #     self.overlay.hide()
        #     self.clear_magnifier_focus()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            event.ignore()
        elif event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self._undo_last_action()
        else:
            super().keyPressEvent(event)

    def _on_selection(self, start_offset, duration):
        sel_abs = self._current_start + start_offset
        print(f"[Full Disclosure] Strip selected: {duration:.1f}s at {sel_abs:.2f}s")
        
    def _on_overlay_double_clicked(self, start_sec, duration):
        """Open expanded view for the selected time range."""
        # TODO: Expanded view disabled for now
        # dialog = ExpandedViewDialog(self._engine, self._current_start + start_sec, duration, self)
        # dialog.exec_()
        pass


# ============================================================================
# EXPANDED VIEW DIALOG - COMMENTED OUT (Not needed for now)
# ============================================================================
# class ExpandedViewDialog(QDialog):
#     """Expanded 12-lead view for selected time range."""
#     
#     def __init__(self, replay_engine, start_sec: float, duration: float, parent=None):
#         super().__init__(parent)
#         self._engine = replay_engine
#         self._reader = replay_engine._reader
#         self._start_sec = start_sec
#         self._duration = duration
#         
#         self.setWindowTitle("Expanded View")
#         self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
#         self.setWindowState(Qt.WindowMaximized)
#         
#         screen = QApplication.primaryScreen()
#         if screen:
#             self.resize(screen.availableGeometry().size())
#             
#         self.setStyleSheet(f"QDialog {{ background: {COL_BLACK}; }}")
#         
#         self._build_ui()
#         self._update_canvases()
#         
#     def _build_ui(self):
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(16, 16, 16, 16)
#         layout.setSpacing(8)
#         
#         top_bar = QFrame()
#         top_bar.setStyleSheet(f"background: {COL_DARK}; border-bottom: 1px solid {COL_GREEN_DRK}; border-radius: 4px;")
#         top_bar.setFixedHeight(44)
#         top_layout = QHBoxLayout(top_bar)
#         top_layout.setContentsMargins(14, 4, 14, 4)
#         top_layout.setSpacing(12)
#         
#         start_real = datetime.fromtimestamp(self._engine._reader.start_time + self._start_sec)
#         end_real = datetime.fromtimestamp(self._engine._reader.start_time + self._start_sec + self._duration)
#         self.lbl_time = QLabel(f"Time Range: {start_real.strftime('%H:%M:%S')} - {end_real.strftime('%H:%M:%S')}")
#         self.lbl_time.setStyleSheet(f"color: {COL_GREEN}; font-weight: bold; font-size: 15px;")
#         top_layout.addWidget(self.lbl_time)
#         
#         top_layout.addStretch()
#         
#         btn_close = QPushButton("Close")
#         btn_close.setStyleSheet("""
#             QPushButton {
#                 background: #0d1b2a; color: #a0c4e8;
#                 border: 1px solid #2a5a6d; padding: 5px 14px;
#                 font-size: 13px; font-weight: bold; border-radius: 4px;
#             }
#             QPushButton:hover {
#                 background: #162a3a;
#             }
#         """)
#         btn_close.clicked.connect(self.accept)
#         top_layout.addWidget(btn_close)
#         
#         layout.addWidget(top_bar)
#         
#         canvas_frame = QFrame()
#         canvas_frame.setStyleSheet(f"background: {COL_BLACK}; border: 1px solid {COL_GREEN_DRK}; border-radius: 4px;")
#         self.canvas_layout = QVBoxLayout(canvas_frame)
#         self.canvas_layout.setContentsMargins(8, 12, 8, 12)
#         self.canvas_layout.setSpacing(6)
#         
#         self._canvases = []
#         leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
#         for lead in leads:
#             row = QHBoxLayout()
#             row.setContentsMargins(0, 0, 0, 0)
#             row.setSpacing(6)
#             
#             lbl = QLabel(lead)
#             lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
#             lbl.setStyleSheet(
#                 f"color: {COL_GREEN}; font-weight: bold; font-size: 16px;"
#                 f" background: #0a0f18; border-right: 1px solid {COL_GREEN_DRK};"
#                 f" padding-right: 8px; padding-top: 4px; padding-bottom:4px;"
#             )
#             lbl.setFixedWidth(52)
#             
#             canvas = ECGStripCanvas(canvas_frame, height=80, color=COL_GREEN, lead_name=lead)
#             canvas.set_paper_speed(25)
#             canvas.set_gain(2.0)  # Higher gain for expanded view
#             canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
#             
#             row.addWidget(lbl)
#             row.addWidget(canvas, 1)
#             self.canvas_layout.addLayout(row)
#             self._canvases.append(canvas)
#             
#         layout.addWidget(canvas_frame, 1)
#         
#     def _update_canvases(self):
#         end_sec = min(self._start_sec + self._duration, self._engine.duration_sec)
#         data = self._reader.read_range(self._start_sec, end_sec)
#         expected_len = int(self._duration * self._engine.fs)
#         
#         # Get beat annotations for this window
#         beat_annotations = []
#         try:
#             for m in self._engine._metrics:
#                 all_beats = m.get('all_beats', [])
#                 if not all_beats:
#                     continue
#                 for beat in all_beats:
#                     ts = float(beat.get('timestamp', 0.0))
#                     if self._start_sec <= ts <= end_sec:
#                         beat_annotations.append({
#                             'timestamp': ts,
#                             'label': str(beat.get('label', 'N'))
#                         })
#             beat_annotations.sort(key=lambda b: b['timestamp'])
#         except Exception as e:
#             print(f"[Expanded View] Error loading beat annotations: {e}")
#         
#         # Generate time array for ECG strips
#         x = np.linspace(0, self._duration, expected_len)
#         
#         # Optimization: Process data efficiently with numpy vectorization
#         for i, c in enumerate(self._canvases):
#             if i < data.shape[0] and data.shape[1] > 0:
#                 d_i = data[i]
#                 # Use numpy slicing instead of padding for better performance
#                 if len(d_i) != expected_len:
#                     if len(d_i) > expected_len:
#                         d_i = d_i[:expected_len]
#                     else:
#                         # Create padded array more efficiently
#                         padded = np.empty(expected_len, dtype=d_i.dtype)
#                         padded[:len(d_i)] = d_i
#                         padded[len(d_i):] = d_i[-1] if len(d_i) > 0 else 0
#                         d_i = padded
#                 # Convert to float32 for faster rendering and pass beat annotations
#                 c.set_data(x, np.asarray(d_i, dtype=np.float32), beat_annotations=beat_annotations, start_sec=self._start_sec)
#             else:
#                 c.set_data(x, np.zeros(expected_len, dtype=np.float32), beat_annotations=beat_annotations, start_sec=self._start_sec)
# ============================================================================

class HolterToolHandlers:
    """
    Centralized button/tool handler implementations for Holter UI.
    Extracted from holter_ui.py to reduce file size and improve maintainability.
    """
    
    @staticmethod
    def handle_patient_information(parent):
        """Show patient information dialog."""
        if hasattr(parent, '_show_patient_information'):
            parent._show_patient_information()
    
    @staticmethod
    def handle_full_disclosure(parent):
        """Open Full Disclosure dialog."""
        if hasattr(parent, "_replay_engine") and parent._replay_engine:
            from .holter_full_disclosure import HolterFullDisclosureDialog
            dialog = HolterFullDisclosureDialog(parent._replay_engine, parent)
            dialog.exec_()
        else:
            from PyQt5.QtWidgets import QMessageBox
            message_box = QMessageBox(parent)
            message_box.setIcon(QMessageBox.Warning)
            message_box.setWindowTitle("No Data")
            message_box.setText("No valid replay engine found for Full Disclosure.")
            message_box.setStyleSheet("""
                QMessageBox {
                    background-color: #2b2b2b;
                }
                QMessageBox QLabel {
                    color: white;
                }
                QMessageBox QPushButton {
                    color: white;
                    background-color: #2b2b2b;
                    border: 1px solid #555555;
                    padding: 5px 18px;
                    min-width: 55px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #444444;
                }
            """)
            message_box.exec_()
    
    @staticmethod
    def handle_goto_template(parent):
        """Navigate to the Template tab."""
        # parent is HolterReplayPanel; _focus_tab lives on the top-level HolterWindow.
        # Walk up the widget tree until we find it.
        target = parent
        while target is not None:
            if hasattr(target, '_focus_tab'):
                target._focus_tab("template")
                return
            target = getattr(target, 'parentWidget', lambda: None)()
    
    @staticmethod
    def handle_gain_settings(parent, btn=None):
        """Cycle through gain settings."""
        try:
            from .theme import GAINS
        except ImportError:
            from ecg.holter.theme import GAINS
        
        gains = [g / 10.0 for g in GAINS]
        curr_g = getattr(parent, '_curr_gain_idx', 1)
        next_g = (curr_g + 1) % len(gains)
        parent._curr_gain_idx = next_g
        val = gains[next_g]
        
        for s in getattr(parent, "_ch_strips", []):
            s.set_gain(val)
            s.update()  # Force repaint with new gain
        if hasattr(parent, "_mini_strip") and parent._mini_strip:
            parent._mini_strip.set_gain(val)
            parent._mini_strip.update()  # Force repaint
        if btn:
            btn.setText(f"Gain: {int(val*10)}mm/mV")
    
    @staticmethod
    def handle_paper_speed(parent, btn=None):
        """Cycle through paper speed settings."""
        try:
            from .theme import PAPER_SPEEDS
        except ImportError:
            from ecg.holter.theme import PAPER_SPEEDS
        
        speeds = PAPER_SPEEDS
        curr_s = getattr(parent, '_curr_speed_idx', 1)
        next_s = (curr_s + 1) % len(speeds)
        parent._curr_speed_idx = next_s
        val = speeds[next_s]
        
        for s in getattr(parent, "_ch_strips", []):
            s.set_paper_speed(int(val))
        if hasattr(parent, "_mini_strip"):
            parent._mini_strip.set_paper_speed(int(val))
        if btn:
            btn.setText(f"Paper speed:{val}mm/s")
        
        # Adjust strip_length_sec so the replay engine delivers the right amount of data.
        parent._strip_length_sec = 10.0 * (25.0 / max(1.0, float(val)))
        if getattr(parent, "_replay_engine", None):
            try:
                parent._replay_engine.set_window_length(parent._strip_length_sec)
            except Exception:
                pass
        
        # Re-seek to force data reload with the new strip length
        try:
            current_pos = parent._slider_value_to_sec(parent._slider.value())
            parent.seek_requested.emit(current_pos)
        except Exception:
            pass
    
    @staticmethod
    def handle_strip_length(parent, btn=None):
        """Cycle through strip length settings."""
        lengths = [3, 7, 10, 15, 30]
        curr_l = getattr(parent, '_curr_length_idx', 1)
        next_l = (curr_l + 1) % len(lengths)
        parent._curr_length_idx = next_l
        val = lengths[next_l]
        parent._strip_length_sec = float(val)
        
        if getattr(parent, "_replay_engine", None):
            try:
                parent._replay_engine.set_window_length(parent._strip_length_sec)
            except Exception:
                pass
        if btn:
            btn.setText(f"Strip Length:{val}s")
    
    @staticmethod
    def apply_tool_mode_to_strips(parent, mode):
        """Apply tool mode to all ECG strips."""
        try:
            from .tool_engine import canonical_tool
        except ImportError:
            from ecg.holter.tool_engine import canonical_tool
        
        canonical_mode = canonical_tool(mode)
        parent._tool_engine.set_tool(canonical_mode)
        
        for strip in getattr(parent, "_ch_strips", []):
            if hasattr(strip, 'set_mode'):
                strip.set_mode(canonical_mode)
        if hasattr(parent._mini_strip, 'set_mode'):
            parent._mini_strip.set_mode(canonical_mode)
        for strip in getattr(parent, "_template_thumbs", []):
            if hasattr(strip, 'set_mode'):
                strip.set_mode(canonical_mode)
    
    @staticmethod
    def handle_add_event(parent):
        """Open Add Event dialog."""
        try:
            from .add_event_dialog import AddEventDialog
        except ImportError:
            from ecg.holter.add_event_dialog import AddEventDialog
        
        from datetime import datetime
        
        # Get current time and position
        start_time = datetime.now()
        current_sec = 0.0
        hr = 0.0
        
        if hasattr(parent, '_replay_engine') and parent._replay_engine:
            try:
                current_sec = parent._replay_engine.current_position()
                # Get start time from reader if available
                if hasattr(parent._replay_engine, '_reader') and hasattr(parent._replay_engine._reader, 'start_time'):
                    start_time = datetime.fromtimestamp(parent._replay_engine._reader.start_time)
            except Exception:
                pass
        
        # Get current HR from status if available
        if hasattr(parent, '_current_bpm'):
            hr = parent._current_bpm
        
        # Open dialog
        dialog = AddEventDialog(parent, start_time=start_time, current_sec=current_sec, hr=hr)
        dialog.event_added.connect(lambda data: HolterToolHandlers._on_event_added(parent, data))
        dialog.exec_()
    
    @staticmethod
    def _on_event_added(parent, event_data: dict):
        """Handle event data from Add Event dialog."""
        print(f"[Add Event] Event added: {event_data}")
        
        # Add event to parent's event list
        if hasattr(parent, '_custom_events'):
            parent._custom_events.append(event_data)
        else:
            parent._custom_events = [event_data]
        
        # Handle different actions
        action = event_data.get('action', 'add_event')
        
        if action == 'instant_print':
            # TODO: Implement instant print
            print("[Add Event] Instant print triggered")
        elif action == 'export_pdf':
            # TODO: Implement PDF export
            print("[Add Event] PDF export triggered")
        elif action == 'add_event':
            # Just add to event list
            print(f"[Add Event] Event added at timestamp {event_data['timestamp']}")
