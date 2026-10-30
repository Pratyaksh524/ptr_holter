"""Comprehensive ECG Analysis - application entry point.

    python src/main.py

Main window: device connection + live 12-lead plotter. "Connect & Start"
also opens a Holter session (HolterStreamWriter + HolterAnalysisWorker), so
everything shown from connect to "Stop" is saved for all 12 leads.
"Previous Recordings" opens the analysis workspace (HolterMainWindow) on the
saved-recordings list.
See ARCHITECTURE.md at the repo root.
"""

import json
import multiprocessing
import os
import sys

# Auto arrhythmia detection runs long recordings in worker processes. In the
# PyInstaller build each worker starts this exe: here it runs its detection
# job and exits, before the Qt / UI imports below and without opening another
# app window. Does nothing in the app's own process or when run from source.
if __name__ == "__main__":
    multiprocessing.freeze_support()

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# The serial code prints emoji. On a Windows console (cp1252) that print raises
# UnicodeEncodeError inside the port probe and the device is never detected.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QMainWindow,
                             QMessageBox, QPushButton, QVBoxLayout, QWidget)

from ecg.holter.analysis_worker import HolterAnalysisWorker
from ecg.holter.holter_ui import (HolterMainWindow, HolterStartDialog, _find_latest_completed_session,
                                  _resolve_recordings_dir)
from ecg.holter.live_12lead_plotter import FS, DeviceAcquisition, Live12LeadPlotter
from ecg.holter.stream_writer import HolterStreamWriter

APP_NAME = "Comprehensive ECG Analysis"
AUTO_PORT = "Auto-detect"


class ComprehensiveMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        self._acq = DeviceAcquisition(parent=self)
        self._acq.connected.connect(self._on_connected)
        self._acq.device_info.connect(self._on_device_info)
        self._acq.packets_ready.connect(self._on_packets)
        self._acq.stopped.connect(self._on_acq_stopped)

        self._writer = None
        self._worker = None
        self._workspace = None
        self._last_valid = {}
        self._user_stopping = False

        self._build_ui()
        self._refresh_ports()
        self._update_buttons()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_status)
        self._status_timer.start(1000)

    # ---- UI --------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)

        bar = QHBoxLayout()
        self._port_box = QComboBox()
        self._port_box.setMinimumWidth(140)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_ports)
        self._connect_btn = QPushButton("Connect && Start")
        self._connect_btn.clicked.connect(self._start_acquisition)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._stop_acquisition)
        prev_btn = QPushButton("Previous Recordings")
        prev_btn.clicked.connect(self._open_previous)

        bar.addWidget(QLabel("Port"))
        bar.addWidget(self._port_box)
        bar.addWidget(refresh)
        bar.addSpacing(16)
        for b in (self._connect_btn, self._stop_btn, prev_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        self._rec_label = QLabel("")
        self._rec_label.setStyleSheet("color:#CC0000;font-weight:bold;")
        bar.addWidget(self._rec_label)
        root.addLayout(bar)

        self._plotter = Live12LeadPlotter(self)
        root.addWidget(self._plotter, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Not connected")

    def _refresh_ports(self):
        current = self._port_box.currentText()
        self._port_box.clear()
        self._port_box.addItem(AUTO_PORT)
        try:
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                self._port_box.addItem(p.device)
        except Exception:
            pass
        if current:
            self._port_box.setCurrentText(current)

    def _update_buttons(self):
        running = self._acq.is_running
        recording = bool(self._writer and self._writer.is_running)
        self._connect_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running or recording)
        self._port_box.setEnabled(not running)
        self._rec_label.setText("● REC" if recording else "")

    def _update_status(self):
        if self._acq.is_running:
            msg = f"Connected {self._acq.port}  |  {self._acq.packets_per_sec:.0f} packets/s (expected {FS})"
            loss = getattr(self._acq.reader, "packet_loss_percent", 0.0) or 0.0
            msg += f"  |  loss {loss:.1f} %"
            if self._writer and self._writer.is_running:
                msg += f"  |  recording to {self._writer.session_dir}"
            self.statusBar().showMessage(msg)
        self._update_buttons()

    # ---- acquisition ------------------------------------------------------

    def _start_acquisition(self):
        port = self._port_box.currentText()
        self.statusBar().showMessage("Connecting to device…")
        self._connect_btn.setEnabled(False)
        self._plotter.clear()
        self._acq.start(None if port == AUTO_PORT else port)

    def _on_connected(self, ok, port, err):
        if not ok:
            self.statusBar().showMessage("Not connected")
            QMessageBox.warning(self, APP_NAME, f"Could not connect to the ECG device.\n\n{err}")
        else:
            self._plotter.start_elapsed_timer()
            if not (self._writer and self._writer.is_running):
                self._start_recording()
        self._update_buttons()

    def _on_device_info(self, version, serial_no):
        self.setWindowTitle(f"{APP_NAME}  -  device {serial_no or '?'}  fw {version or '?'}")

    def _on_packets(self, packets):
        self._plotter.push_packets(packets)
        if self._writer and self._writer.is_running:
            for p in packets:
                self._writer.push(Live12LeadPlotter.fill_lead_off(p, self._last_valid))

    def _stop_acquisition(self):
        # Stop the device first, so nothing is acquired while the save dialog
        # is open. The flag keeps that "stopped" signal from reading as the
        # device dropping out mid-recording.
        self._plotter.stop_elapsed_timer()
        self._user_stopping = True
        try:
            self._acq.stop()
        finally:
            self._user_stopping = False
        if self._writer and self._writer.is_running:
            self._stop_recording()
        self._update_buttons()

    def _on_acq_stopped(self):
        self._plotter.stop_elapsed_timer()
        if self._writer and self._writer.is_running and not self._user_stopping:
            QMessageBox.warning(self, APP_NAME, "The device stopped sending data. The recording has been stopped.")
            self._stop_recording()
        self.statusBar().showMessage("Not connected")
        self._update_buttons()

    # ---- recording --------------------------------------------------------

    def _start_recording(self):
        out_dir = _resolve_recordings_dir("")
        os.makedirs(out_dir, exist_ok=True)
        self._last_valid = {name: 2048.0 for name in self._plotter._last_valid}
        self._writer = HolterStreamWriter(output_dir=out_dir, patient_info={}, fs=FS, chunk_seconds=5)
        session_dir = self._writer.start()
        self._worker = HolterAnalysisWorker(
            analysis_queue=self._writer.analysis_queue,
            on_chunk_done=self._writer.update_live_analysis,
            fs=FS,
        )
        self._writer.attach_analysis_worker(self._worker)
        self._worker.start()
        print(f"[Comprehensive] recording started in {session_dir}")
        self._update_buttons()

    def _stop_recording(self):
        # The workspace owns the stop flow (writer.stop + patient details dialog
        # + load the finished session). Fall back to stopping the writer here.
        if self._workspace is not None and getattr(self._workspace, "_writer", None) is self._writer:
            self._workspace._stop_recording()
        elif self._writer and self._writer.is_running:
            self._save_capture(self._writer.stop())
        self._writer = None
        self._worker = None
        self._update_buttons()

    def _save_capture(self, summary):
        """Ask for the patient details of a capture saved without the workspace open."""
        session_dir = summary.get("session_dir", "")
        if not session_dir:
            return
        dialog = HolterStartDialog(self, patient_info={}, output_dir=session_dir)
        dialog.setWindowTitle("Save Comprehensive ECG Analysis Recording Details")
        if dialog.exec_() == QDialog.Accepted and dialog.get_result():
            patient_info = dialog.get_result()[0]
            try:
                with open(os.path.join(session_dir, "patient.json"), "w") as f:
                    json.dump(patient_info, f, indent=4)
            except Exception as e:
                print(f"[Comprehensive] could not save patient.json: {e}")
        QMessageBox.information(self, APP_NAME, f"Recording saved to:\n{session_dir}")

    def _open_previous(self):
        if self._writer and self._writer.is_running:
            QMessageBox.information(self, APP_NAME, "Press Stop first to save the current recording.")
            return
        ws = HolterMainWindow(parent=self, session_dir="", patient_info={}, writer=None, live_source=None)
        try:
            latest = _find_latest_completed_session(_resolve_recordings_dir(""))
            if latest:
                ws.load_completed_session(latest)
        except Exception as e:
            print(f"[Comprehensive] could not load latest session: {e}")
        ws._focus_tab("RECORDINGS")
        self._workspace = ws
        ws.showMaximized()
        ws.raise_()

    def closeEvent(self, event):
        if self._writer and self._writer.is_running:
            ans = QMessageBox.question(self, APP_NAME, "A recording is in progress. Stop it and quit?")
            if ans != QMessageBox.Yes:
                event.ignore()
                return
            self._writer.stop()
        self._acq.close()
        super().closeEvent(event)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    win = ComprehensiveMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
