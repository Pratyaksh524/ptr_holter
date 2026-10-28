"""Live 12-lead plotter for Comprehensive ECG Analysis.

Two pieces:

* ``DeviceAcquisition`` - connects to the ECG device through
  ``ecg.serial.serial_reader`` (port scan -> VERSION -> SERIAL -> START, done in
  ``DeviceStartWorker`` off the UI thread), then polls ``read_packets()`` on a
  QTimer and emits the parsed packets (``ecg.serial.packet_parser`` dicts:
  lead name -> ADC int, or None when that lead is off).

* ``Live12LeadPlotter`` - draws the 12 leads live in a 3x4, 6x2 or 12x1
  layout. Every lead has its own y-axis in ADC counts, 0 to 4096, with the
  baseline at 2048. The trace goes through a 0.5 Hz DFT (baseline) filter done
  as a zero-phase Gaussian: trace = raw - GaussianLowPass(raw, 0.5 Hz) + 2048.

The plotter also keeps the last ``BUFFER_SECONDS`` of raw ADC per lead in
``self.data`` (12 arrays, order of ``file_format.LEAD_NAMES``). That is the
``live_source.data`` that ``HolterMainWindow`` reads during a live recording.
"""

import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QScrollArea,
                             QVBoxLayout, QWidget)
from scipy.ndimage import gaussian_filter1d

from ..ecg_filters import _GAUSSIAN_HALF_WIDTH_SIGMAS, _GAUSSIAN_SIGMA_AT_MINUS_3DB
from .file_format import LEAD_NAMES

FS = 500
BUFFER_SECONDS = 14              # 10 s on screen (12x1) + filter context
POLL_MS = 20
REDRAW_MS = 50

# Y-axis: full ADC range per lead, baseline in the middle.
ADC_MIN, ADC_MAX, ADC_BASELINE = 0, 4096, 2048

# DFT filter: 0.5 Hz high-pass = signal minus its zero-phase Gaussian low-pass.
# Same sigma and kernel width (+-3 sigma) as ecg_filters._gaussian_lowpass;
# scipy's mode="mirror" is that function's np.pad(mode="reflect").
DFT_CUTOFF_HZ = 0.5
DFT_SIGMA_SAMPLES = _GAUSSIAN_SIGMA_AT_MINUS_3DB * FS / DFT_CUTOFF_HZ

# Paper speed sets how many seconds fit across the sheet (10 s at 25 mm/s).
MM_PER_SEC_CHOICES = (12.5, 25.0, 50.0)
SHEET_SECONDS_AT_25 = 10.0

# Monitor style: black screen, light grey grid, only the ECG trace is green.
PAPER_BG = "#000000"
GRID_MINOR = (45, 45, 45)
GRID_MAJOR = (90, 90, 90)
BASELINE_COLOR = (130, 130, 130)
TRACE = "#00FF00"
LABEL_COLOR = "#D0D0D0"
AXIS_COLOR = "#B0B0B0"
LEAD_OFF_COLOR = "#FF3333"
ROWS_PER_PAGE = 6                # lead rows visible at once; more rows scroll vertically
TIME_MINOR_S, TIME_MAJOR_S = 0.04, 0.2       # 1 mm / 5 mm at 25 mm/s
ADC_MINOR, ADC_MAJOR = 256, 1024

# Lead placement: list of columns, each column a list of leads top to bottom.
LAYOUTS = {
    "3x4": [["I", "II", "III"], ["aVR", "aVL", "aVF"], ["V1", "V2", "V3"], ["V4", "V5", "V6"]],
    "6x2": [["I", "II", "III", "aVR", "aVL", "aVF"], ["V1", "V2", "V3", "V4", "V5", "V6"]],
    "12x1": [list(LEAD_NAMES)],
}


def dft_filter_gaussian(block: np.ndarray, fs: float = FS, cutoff_hz: float = DFT_CUTOFF_HZ) -> np.ndarray:
    """DFT (baseline) filter, zero phase: x - GaussianLowPass(x, cutoff_hz).

    ``block`` is (leads, samples) raw ADC. Returns the same shape, centred on 0."""
    sigma = _GAUSSIAN_SIGMA_AT_MINUS_3DB * fs / cutoff_hz
    baseline = gaussian_filter1d(block, sigma, axis=-1, mode="mirror",
                                 truncate=_GAUSSIAN_HALF_WIDTH_SIGMAS)
    return block - baseline


# -----------------------------------------------------------------------------
# Device acquisition
# -----------------------------------------------------------------------------

class DeviceAcquisition(QObject):
    """Owns the serial connection and turns it into a stream of packets."""

    connected = pyqtSignal(bool, str, str)    # success, port, error message
    device_info = pyqtSignal(str, str)        # firmware version, machine serial
    packets_ready = pyqtSignal(list)          # list of packet dicts
    stopped = pyqtSignal()

    def __init__(self, baudrate: int = 115200, parent=None):
        super().__init__(parent)
        self.baudrate = baudrate
        self.reader = None
        self.port = ""
        self._worker = None
        self._version = ""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self.packets_total = 0
        self._rate_t0 = time.time()
        self._rate_n = 0
        self.packets_per_sec = 0.0

    @property
    def is_running(self) -> bool:
        return bool(self.reader is not None and getattr(self.reader, "running", False))

    def start(self, port: str = None):
        """Connect and start streaming. ``port`` None/"" means auto-scan."""
        from ..serial.serial_reader import DeviceStartWorker

        if self.is_running or self._worker is not None:
            return
        self._worker = DeviceStartWorker(port or None, self.baudrate, self.reader, self)
        self._worker.version_ready.connect(self._on_version)
        self._worker.serial_ready.connect(lambda s: self.device_info.emit(self._version, s))
        self._worker.connected.connect(self._on_connected)
        # Drop the worker only once its run() has really returned; dropping it
        # on `connected` (emitted from inside run()) can destroy a live QThread.
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self):
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.deleteLater()

    def _on_version(self, version: str):
        self._version = version or ""

    def _on_connected(self, ok: bool, port: str, err: str):
        from ..serial.serial_reader import GlobalHardwareManager

        self.port = port
        if ok:
            self.reader = GlobalHardwareManager().reader
            self.packets_total = 0
            self._rate_t0, self._rate_n = time.time(), 0
            self._timer.start(POLL_MS)
        self.connected.emit(ok, port, err)

    def stop(self):
        self._timer.stop()
        if self.reader is not None:
            try:
                self.reader.stop()
            except Exception as e:
                print(f"[Acquisition] stop warning: {e}")
        self.stopped.emit()

    def close(self):
        from ..serial.serial_reader import GlobalHardwareManager

        # App closing while the port scan / handshake is still running: let it
        # finish (each port probe has a short timeout) instead of destroying it.
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(15000)
        self.stop()
        GlobalHardwareManager().close_reader()
        self.reader = None

    def _poll(self):
        if self.reader is None:
            return
        if not getattr(self.reader, "running", False):
            # Reader stopped itself (cable pulled / serial error).
            self._timer.stop()
            self.stopped.emit()
            return
        packets = self.reader.read_packets(max_packets=1000)
        if not packets:
            return
        self.packets_total += len(packets)
        self._rate_n += len(packets)
        now = time.time()
        if now - self._rate_t0 >= 1.0:
            self.packets_per_sec = self._rate_n / (now - self._rate_t0)
            self._rate_t0, self._rate_n = now, 0
        self.packets_ready.emit(packets)


# -----------------------------------------------------------------------------
# Plotter
# -----------------------------------------------------------------------------

class _ScrollFriendlyLayoutWidget(pg.GraphicsLayoutWidget):
    """The plots are fixed (no pan/zoom), so hand the mouse wheel to the
    surrounding QScrollArea instead of letting the graphics scene eat it."""

    def wheelEvent(self, event):
        event.ignore()


class Live12LeadPlotter(QWidget):
    """12-lead live view, one ADC axis (0-4096, baseline 2048) per lead."""

    def __init__(self, parent=None, layout_name: str = "3x4"):
        super().__init__(parent)
        self._n = FS * BUFFER_SECONDS
        self._ring = np.zeros((len(LEAD_NAMES), self._n))
        self._ptr = 0                 # next write position in _ring
        self._lead_off = {name: False for name in LEAD_NAMES}
        self._last_valid = {name: float(ADC_BASELINE) for name in LEAD_NAMES}
        self._layout_name = layout_name
        self._mm_per_sec = 25.0
        self._filter_on = True
        self._samples_seen = 0
        self._curves = {}
        self._off_labels = {}

        self._build_ui()
        self._rebuild_sheet()

        self._redraw_timer = QTimer(self)
        self._redraw_timer.timeout.connect(self._redraw)
        self._redraw_timer.start(REDRAW_MS)

    # ---- live_source interface (read by HolterMainWindow) -------------------

    @property
    def data(self):
        """Last BUFFER_SECONDS of raw ADC, one array per lead (LEAD_NAMES order)."""
        return list(self._latest(self._n))

    def _latest(self, n):
        """Newest n samples of every lead, oldest first, shape (12, n)."""
        idx = (self._ptr - n + np.arange(n)) % self._n
        return self._ring[:, idx]

    # ---- input -----------------------------------------------------------

    @staticmethod
    def fill_lead_off(packet: dict, last_valid: dict) -> dict:
        """Packet with every lead set to a number: a lead that is off (None)
        holds its last valid value, so the sample count (and the recording
        clock) never skips. Updates ``last_valid`` in place."""
        out = dict(packet)
        for name in LEAD_NAMES:
            v = packet.get(name)
            if v is None:
                out[name] = int(last_valid[name])
            else:
                last_valid[name] = float(v)
        return out

    def push_packets(self, packets):
        for p in packets:
            col = self._ptr % self._n
            for i, name in enumerate(LEAD_NAMES):
                v = p.get(name)
                self._lead_off[name] = v is None
                if v is not None:
                    self._last_valid[name] = float(v)
                self._ring[i, col] = self._last_valid[name]
            self._ptr = col + 1
        self._samples_seen += len(packets)

    def clear(self):
        self._ring[:] = 0.0
        self._ptr = 0
        self._samples_seen = 0

    # ---- UI --------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        bar = QHBoxLayout()
        bar.setContentsMargins(8, 4, 8, 0)

        self._layout_box = QComboBox()
        self._layout_box.addItems(list(LAYOUTS))
        self._layout_box.setCurrentText(self._layout_name)
        self._layout_box.currentTextChanged.connect(self._on_layout)

        self._speed_box = QComboBox()
        self._speed_box.addItems([f"{s:g} mm/s" for s in MM_PER_SEC_CHOICES])
        self._speed_box.setCurrentText("25 mm/s")
        self._speed_box.currentIndexChanged.connect(self._on_speed)

        self._filter_chk = QCheckBox(f"DFT {DFT_CUTOFF_HZ:g} Hz (zero-phase Gaussian)")
        self._filter_chk.setChecked(True)
        self._filter_chk.toggled.connect(lambda on: setattr(self, "_filter_on", on))

        for label, w in (("Layout", self._layout_box), ("Speed", self._speed_box)):
            bar.addWidget(QLabel(label))
            bar.addWidget(w)
            bar.addSpacing(12)
        bar.addWidget(self._filter_chk)
        bar.addStretch(1)
        bar.addWidget(QLabel(f"Y: {ADC_MIN}-{ADC_MAX} ADC, baseline {ADC_BASELINE}"))
        root.addLayout(bar)

        self._view = _ScrollFriendlyLayoutWidget()
        self._view.setBackground(PAPER_BG)
        self._view.ci.setSpacing(0)
        # ROWS_PER_PAGE rows fill the visible height; a layout with more rows
        # (12x1) gets taller than the viewport and scrolls vertically.
        self._scroll = QScrollArea()
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._scroll.setStyleSheet(f"QScrollArea {{ background: {PAPER_BG}; }}")
        self._scroll.setWidget(self._view)
        self._scroll.viewport().installEventFilter(self)
        root.addWidget(self._scroll, 1)

    def eventFilter(self, obj, event):
        if obj is self._scroll.viewport() and event.type() == QEvent.Resize:
            self._fit_rows_to_page()
        return super().eventFilter(obj, event)

    def _fit_rows_to_page(self):
        n_rows = max(len(c) for c in LAYOUTS[self._layout_name])
        page_h = self._scroll.viewport().height()
        rows_shown = min(n_rows, ROWS_PER_PAGE)
        self._view.setMinimumHeight(int(page_h * n_rows / rows_shown) if n_rows > ROWS_PER_PAGE else 0)

    def _on_layout(self, name):
        self._layout_name = name
        self._rebuild_sheet()
        self._fit_rows_to_page()
        self._scroll.verticalScrollBar().setValue(0)

    def _on_speed(self, idx):
        self._mm_per_sec = MM_PER_SEC_CHOICES[idx]
        self._rebuild_sheet()

    def _seconds_per_column(self):
        sheet = SHEET_SECONDS_AT_25 * 25.0 / self._mm_per_sec
        return min(sheet / len(LAYOUTS[self._layout_name]), BUFFER_SECONDS - 1.0)

    def _rebuild_sheet(self):
        """One PlotItem per lead, laid out as rows x columns."""
        self._view.clear()
        self._curves.clear()
        self._off_labels.clear()

        cols = LAYOUTS[self._layout_name]
        n_rows = max(len(c) for c in cols)
        secs = self._seconds_per_column()
        axis_pen = pg.mkPen(AXIS_COLOR)

        for r in range(n_rows):
            for c, leads in enumerate(cols):
                if r >= len(leads):
                    continue
                name = leads[r]
                plot = self._view.addPlot(row=r, col=c)
                plot.setMouseEnabled(False, False)
                plot.hideButtons()
                plot.setMenuEnabled(False)
                plot.hideAxis("bottom")
                left = plot.getAxis("left")
                if c == 0:
                    left.setTicks([[(v, str(v)) for v in (ADC_MIN, ADC_BASELINE, ADC_MAX)]])
                    left.setPen(axis_pen)
                    left.setTextPen(axis_pen)
                    left.setWidth(40)
                else:
                    plot.hideAxis("left")
                plot.setXRange(0, secs, padding=0)
                plot.setYRange(ADC_MIN, ADC_MAX, padding=0)
                self._add_grid(plot, secs)

                label = pg.TextItem(name, color=LABEL_COLOR, anchor=(0, 0))
                label.setPos(0.02 * secs, ADC_MAX)
                plot.addItem(label)
                off = pg.TextItem("LEAD OFF", color=LEAD_OFF_COLOR, anchor=(1, 0))
                off.setPos(0.98 * secs, ADC_MAX)
                off.setVisible(False)
                plot.addItem(off)
                self._off_labels[name] = off

                curve = pg.PlotCurveItem(pen=pg.mkPen(TRACE, width=1.2))
                plot.addItem(curve)
                self._curves[name] = curve

    @staticmethod
    def _add_grid(plot, secs):
        def lines(t_step, a_step, pen, z):
            xs = np.arange(0, secs + 1e-9, t_step)
            ys = np.arange(ADC_MIN, ADC_MAX + 1, a_step)
            x = np.concatenate([np.repeat(xs, 2), np.tile([0, secs], len(ys))])
            y = np.concatenate([np.tile([ADC_MIN, ADC_MAX], len(xs)), np.repeat(ys, 2)])
            item = pg.PlotCurveItem(x=x, y=y, connect="pairs", pen=pen)
            item.setZValue(z)
            plot.addItem(item)
        lines(TIME_MINOR_S, ADC_MINOR, pg.mkPen(GRID_MINOR, width=0.5), -12)
        lines(TIME_MAJOR_S, ADC_MAJOR, pg.mkPen(GRID_MAJOR, width=1.0), -11)
        base = pg.InfiniteLine(pos=ADC_BASELINE, angle=0, pen=pg.mkPen(BASELINE_COLOR, width=1, style=Qt.DashLine))
        base.setZValue(-10)
        plot.addItem(base)

    # ---- drawing ---------------------------------------------------------

    def _redraw(self):
        if not self.isVisible() or self._samples_seen == 0:
            return
        n_show = int(self._seconds_per_column() * FS)
        # Filter over extra context on the left so the Gaussian kernel
        # (+-3 sigma) sees real signal there, then keep the newest n_show.
        ctx = int(np.ceil(_GAUSSIAN_HALF_WIDTH_SIGMAS * DFT_SIGMA_SAMPLES))
        n_ctx = min(self._n, n_show + ctx, self._samples_seen)
        if n_ctx < 2:
            return
        window = self._latest(n_ctx)
        if self._filter_on:
            sig = dft_filter_gaussian(window)
        else:
            sig = window - np.median(window, axis=1, keepdims=True)
        sig = np.clip(sig[:, -n_show:] + ADC_BASELINE, ADC_MIN, ADC_MAX)
        x = np.arange(sig.shape[1]) / FS
        for i, name in enumerate(LEAD_NAMES):
            self._curves[name].setData(x=x, y=sig[i])
            self._off_labels[name].setVisible(self._lead_off[name])
