"""
CardioX 12-Lead ECG Acquisition & Analysis Engine — src/ecg/twelve_lead_test.py
==================================================================================
PURPOSE & ARCHITECTURE:
This is the core clinical acquisition and visualization engine (ECGTestWindow) for 12-lead ECG signals.
Key Responsibilities:
1. Real-time 12-Lead Data Buffering: Receives 500Hz serial stream for Leads I, II, III, aVR, aVL, aVF, V1-V6.
2. Clinical Metric Computation: Pan-Tompkins algorithm for HR, PR interval, QRS duration, QT/QTc interval, axes, and RV5/SV1 voltages.
3. Live Rhythm Interpretation: Runs periodic Pan-Tompkins & wave classification on Lead II, updating last_conclusions.json cache every ~1s.
4. Report Snapshotting & PDF Generation (generate_pdf_report): Captures 10-second raw waveform buffer (snap_raw) and renders clinical PDF reports via ecg_report_android.py.

DEVELOPER NOTES & RECENT REFACTORS:
- Fix 1 (Screen Label Fallback): When generating reports, metrics are read from both internal attributes AND screen labels (`max(_live_val, _screen_val)`), preventing phantom-zero PDFs if a read occurs mid-calculation tick.
- Fix 2 (Rate Qualifiers): Automatically appends rate qualifiers ("Bradycardia" for HR < 60, "Sinus Tachycardia" for HR > 100) alongside detected non-NSR arrhythmias.
- Fix 3 (QTc Auto-Warnings): Evaluates QTc against clinical thresholds (≥440ms Borderline, ≥460ms Prolonged, ≥500ms Critical) and auto-injects formatted warnings into report conclusions.
- Safeguard (Flatline Detection): If HR <= 0 or lead std-dev < 0.1, output conclusions default to "No ECG data available" with zeroed metrics to prevent misleading legacy labels.
"""
import os
from utils.app_paths import data_file
import sys
import time
import platform
import numpy as np
import logging
import traceback
from collections import deque
from utils.crash_logger import get_crash_logger
from utils.platform_compat import is_low_spec_mode
from PyQt5.QtWidgets import QMessageBox
# Serial communication now handled by ecg.serial module
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print(" Serial module not available - ECG hardware features disabled")
    SERIAL_AVAILABLE = False
    # Create dummy serial classes
    class Serial:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def readline(self): return b''
    class SerialException(Exception): pass
    serial = type('Serial', (), {'Serial': Serial, 'SerialException': SerialException})()
    class MockComports:
        @staticmethod
        def comports(*args, **kwargs):
            return []
    serial.tools = type('Tools', (), {'list_ports': MockComports()})()

# Import serial communication classes from new modular structure
from .serial import SerialStreamReader, SerialECGReader
from .serial.packet_parser import parse_packet, decode_lead, hex_string_to_bytes, PACKET_SIZE, START_BYTE, END_BYTE, LEAD_NAMES_DIRECT
import csv
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print(" OpenCV (cv2) module not available - some features disabled")
    CV2_AVAILABLE = False
    cv2 = None
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QGroupBox, QFileDialog,
    QStackedLayout, QGridLayout, QSizePolicy, QMessageBox, QFormLayout, QLineEdit, QFrame, QApplication, QDialog
)
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QDateTime, QRect 

# Matplotlib imports (still used for detailed/overlay views)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# PyQtGraph is used for the main real-time plotting grid
import pyqtgraph as pg
import re
from collections import deque
from typing import Deque, Dict, List, Tuple, Optional
from ecg.recording import ECGMenu
from scipy.signal import find_peaks, iirnotch, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from utils.settings_manager import SettingsManager
from utils.patient_profile import resolve_patient_profile
from utils.localization import translate_text
from .demo_manager import DemoManager
from PyQt5.QtWidgets import QGraphicsDropShadowEffect
from functools import partial # For plot clicking
from .clinical_measurements import (
    build_median_beat, get_tp_baseline, measure_qt_from_median_beat,
    measure_rv5_sv1_from_median_beat, measure_st_deviation_from_median_beat, measure_p_duration_from_median_beat,
    calculate_axis_from_median_beat, measure_pr_from_median_beat,
    measure_qrs_duration_from_median_beat
)
# Import from new modular structure
from .metrics.intervals import calculate_qtcf_interval, calculate_rv5_sv1_from_median
from .metrics.axis_calculations import (
    calculate_qrs_axis_from_median, calculate_p_axis_from_median, calculate_t_axis_from_median
)
from .metrics.heart_rate import calculate_heart_rate_from_signal
# ── Unified ECG calculations (HR, RR, PR, QRS, QT, QTc) ──────────────────────
from .ecg_calculations import calculate_all_ecg_metrics
from .ui.display_updates import update_ecg_metrics_display, get_current_metrics_from_labels
from .signal.signal_processing import (
    extract_low_frequency_baseline, detect_signal_source, 
    apply_adaptive_gain, apply_realtime_smoothing
)
from .plotting.plot_widgets import create_plot_grid, LEAD_COLORS_PLOT

# Import lead-off detection (CRITICAL FIX #3)
from .lead_off_detection import detect_lead_off, check_all_leads_quality

# Import smooth display module for jitter-free wave plotting
from .smooth_display import SmoothECGDisplay

# Import constants from utils module
from .utils.constants import HISTORY_LENGTH, NORMAL_HR_MIN, NORMAL_HR_MAX, LEAD_LABELS, LEAD_COLORS, LEADS_MAP

# Import utilities from utils module
from .utils.helpers import SamplingRateCalculator, get_display_gain, generate_realistic_ecg_waveform

# Import serial communication from new modular structure
from .serial import SerialStreamReader, SerialECGReader
from .serial.packet_parser import parse_packet, decode_lead, hex_string_to_bytes, PACKET_SIZE, START_BYTE, END_BYTE, LEAD_NAMES_DIRECT

# ------------------------ Realistic ECG Waveform Generator ------------------------
# NOTE: This function is now in ecg.utils.helpers, but kept here for backward compatibility
def generate_realistic_ecg_waveform(duration_seconds=10, sampling_rate=500, heart_rate=72, lead_name="II"):
    """
    Generate realistic ECG waveform with proper PQRST complexes
    - duration_seconds: Length of waveform in seconds
    - sampling_rate: Samples per second (Hz)
    - heart_rate: Beats per minute
    - lead_name: Lead name for lead-specific characteristics
    """
    import numpy as np
    
    # Calculate parameters
    total_samples = int(duration_seconds * sampling_rate)
    rr_interval = 60.0 / heart_rate  # RR interval in seconds
    samples_per_beat = int(rr_interval * sampling_rate)
    
    # Create time array
    t = np.linspace(0, duration_seconds, total_samples)
    
    # Initialize waveform
    ecg = np.zeros(total_samples)
    
    # Lead-specific characteristics (amplitudes in mV)
    lead_characteristics = {
        "I": {"p_amp": 0.1, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0},
        "II": {"p_amp": 0.15, "qrs_amp": 1.2, "t_amp": 0.3, "baseline": 0.0},
        "III": {"p_amp": 0.05, "qrs_amp": 0.6, "t_amp": 0.15, "baseline": 0.0},
        "aVR": {"p_amp": -0.1, "qrs_amp": -0.8, "t_amp": -0.2, "baseline": 0.0},
        "aVL": {"p_amp": 0.08, "qrs_amp": 0.7, "t_amp": 0.18, "baseline": 0.0},
        "aVF": {"p_amp": 0.12, "qrs_amp": 0.9, "t_amp": 0.25, "baseline": 0.0},
        "V1": {"p_amp": 0.05, "qrs_amp": 0.3, "t_amp": 0.1, "baseline": 0.0},
        "V2": {"p_amp": 0.08, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0},
        "V3": {"p_amp": 0.1, "qrs_amp": 1.0, "t_amp": 0.25, "baseline": 0.0},
        "V4": {"p_amp": 0.12, "qrs_amp": 1.1, "t_amp": 0.3, "baseline": 0.0},
        "V5": {"p_amp": 0.1, "qrs_amp": 1.0, "t_amp": 0.25, "baseline": 0.0},
        "V6": {"p_amp": 0.08, "qrs_amp": 0.8, "t_amp": 0.2, "baseline": 0.0}
    }
    
    char = lead_characteristics.get(lead_name, lead_characteristics["II"])
    
    # Generate beats
    beat_start = 0
    while beat_start < total_samples:
        # P wave (atrial depolarization) - 80-120ms
        p_duration = 0.1  # 100ms
        p_samples = int(p_duration * sampling_rate)
        p_start = beat_start
        p_end = min(p_start + p_samples, total_samples)
        
        if p_start < total_samples:
            p_t = np.linspace(0, p_duration, p_end - p_start)
            p_wave = char["p_amp"] * np.sin(np.pi * p_t / p_duration) * np.exp(-2 * p_t / p_duration)
            ecg[p_start:p_end] += p_wave
        
        # PR interval (isoelectric line) - 120-200ms
        pr_duration = 0.16  # 160ms
        pr_samples = int(pr_duration * sampling_rate)
        pr_start = p_end
        pr_end = min(pr_start + pr_samples, total_samples)
        
        # QRS complex (ventricular depolarization) - 80-120ms
        qrs_duration = 0.08  # 80ms
        qrs_samples = int(qrs_duration * sampling_rate)
        qrs_start = pr_end
        qrs_end = min(qrs_start + qrs_samples, total_samples)
        
        if qrs_start < total_samples:
            qrs_t = np.linspace(0, qrs_duration, qrs_end - qrs_start)
            # Q wave (small negative deflection)
            q_wave = -char["qrs_amp"] * 0.1 * np.exp(-10 * qrs_t / qrs_duration)
            # R wave (large positive deflection)
            r_wave = char["qrs_amp"] * np.sin(np.pi * qrs_t / qrs_duration) * np.exp(-3 * qrs_t / qrs_duration)
            # S wave (negative deflection after R)
            s_wave = -char["qrs_amp"] * 0.3 * np.exp(-5 * qrs_t / qrs_duration)
            
            qrs_complex = q_wave + r_wave + s_wave
            ecg[qrs_start:qrs_end] += qrs_complex
        
        # ST segment (isoelectric) - 80-120ms
        st_duration = 0.08  # 80ms
        st_samples = int(st_duration * sampling_rate)
        st_start = qrs_end
        st_end = min(st_start + st_samples, total_samples)
        
        # T wave (ventricular repolarization) - 160-200ms
        t_duration = 0.16  # 160ms
        t_samples = int(t_duration * sampling_rate)
        t_start = st_end
        t_end = min(t_start + t_samples, total_samples)
        
        if t_start < total_samples:
            t_t = np.linspace(0, t_duration, t_end - t_start)
            t_wave = char["t_amp"] * np.sin(np.pi * t_t / t_duration) * np.exp(-2 * t_t / t_duration)
            ecg[t_start:t_end] += t_wave
        
        # Move to next beat
        beat_start += samples_per_beat
    
    # Add baseline wander (low frequency noise)
    baseline_freq = 0.5  # 0.5 Hz
    baseline_wander = 0.05 * np.sin(2 * np.pi * baseline_freq * t)
    ecg += baseline_wander
    
    # Add high frequency noise (muscle artifact, etc.)
    noise = 0.02 * np.random.normal(0, 1, total_samples)
    ecg += noise
    
    # Add baseline offset
    ecg += char["baseline"]
    
    return ecg, t


def _coerce_numeric_value(value):
    """Best-effort conversion for metric strings like '50', ' 50', or '50 BPM'."""
    try:
        if value is None:
            return None
        if isinstance(value, str):
            cleaned = value.strip().replace("BPM", "").replace("bpm", "")
            if cleaned in ("", "--", "0", "0.0", "Analyzing Rhythm..."):
                return None
            value = cleaned
        numeric = float(value)
        if np.isnan(numeric) or np.isinf(numeric) or numeric <= 0:
            return None
        return numeric
    except Exception:
        return None


def _bpm_rhythm_label(heart_rate):
    """Map BPM to the simple rhythm label we want shown in the UI."""
    hr = _coerce_numeric_value(heart_rate)
    if hr is None:
        return None
    if hr > 100:
        return "Tachycardia"
    if hr < 60:
        return "Bradycardia"
    return "Normal Sinus Rhythm"

# ============================================================================
# NEW PACKET-BASED SERIAL PARSING LOGIC
# Serial communication code moved to ecg.serial module
# Imported above from .serial import SerialStreamReader, SerialECGReader

class LiveLeadWindow(QWidget):
    def __init__(self, lead_name, data_source, buffer_size=80, color="#00ff99"):
        super().__init__()
        self.setWindowTitle(f"Live View: {lead_name}")
        self.resize(900, 300)
        self.lead_name = lead_name
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.color = color

        layout = QVBoxLayout(self)
        self.fig = Figure(facecolor='#000')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#000')
        self.ax.set_xlim(0, self.buffer_size)
        self.ax.set_ylim(-200, 200)
        self.ax.set_title(f"Live {lead_name}", color='white', fontsize=14)
        self.ax.tick_params(colors='white')
        
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        
        self.line, = self.ax.plot([], [], color=color, linewidth=1.0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 20 FPS

    def update_plot(self):
        data = self.data_source()
        if data and len(data) > 0:
            plot_data = np.full(self.buffer_size, np.nan)
            n = min(len(data), self.buffer_size)
            centered = np.array(data[-n:]) - np.mean(data[-n:])
            plot_data[-n:] = centered
            self.line.set_ydata(plot_data)
            self.canvas.draw_idle()

def calculate_st_segment(lead_signal, r_peaks, fs=500, j_offset_ms=40, st_offset_ms=80):
    """
    Calculate mean ST segment amplitude (in mV) at (J-point + st_offset_ms) after R peak.
    - lead_signal: ECG samples (e.g., Lead II)
    - r_peaks: indices of R peaks
    - fs: sampling rate (Hz)
    - j_offset_ms: ms after R peak to estimate J-point (default 40ms)
    - st_offset_ms: ms after J-point to measure ST segment (default 80ms)
    Returns mean ST segment amplitude in mV (float), or '--' if not enough data.
    """
    if len(lead_signal) < 100 or len(r_peaks) == 0:
        return "--"
    j_offset = int(j_offset_ms * fs / 1000)
    st_offset = int(st_offset_ms * fs / 1000)
    st_values = []
    for r in r_peaks:
        st_idx = r + j_offset + st_offset
        if st_idx < len(lead_signal):
            st_values.append(lead_signal[st_idx])
    if len(st_values) == 0:
        return "--"
    st_value = np.mean(st_values)
    if st_value > 0.1:
        return "Elevated"
    elif st_value < -0.1:
        return "Depressed"
    return str(st_value)


class SerialECGReader:
    def __init__(self, port, baudrate):
        if not SERIAL_AVAILABLE:
            raise ImportError("Serial module not available - cannot create ECG reader")
        self.ser = serial.Serial(port, baudrate, timeout=1)
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        print(f" SerialECGReader initialized: Port={port}, Baud={baudrate}")

    def start(self):
        print(" Starting ECG data acquisition...")
        self.ser.reset_input_buffer()
        self.ser.write(b'1\r\n')
        time.sleep(0.5)
        self.running = True
        print(" ECG device started - waiting for data...")

    def stop(self):
        print(" Stopping ECG data acquisition...")
        self.ser.write(b'0\r\n')
        self.running = False
        print(f" Total data packets received: {self.data_count}")

    # ========================================================================
    # OLD read_value() METHOD - COMMENTED OUT
    # Using new packet-based parsing logic instead
    # ========================================================================
    # def read_value(self):
    #     """OLD METHOD - COMMENTED OUT - Using packet-based parsing now"""
    #     if not self.running:
    #         return None
    #     try:
    #         line_raw = self.ser.readline()
    #         line_data = line_raw.decode('utf-8', errors='replace').strip()
    #         
    #         if line_data:
    #             self.data_count += 1
    #             # Print detailed data information
    #             print(f" [Packet #{self.data_count}] Raw data: '{line_data}' (Length: {len(line_data)})")
    #             
    #             # Parse and display ECG value
    #             if line_data.isdigit():
    #                 ecg_value = int(line_data[-3:])
    #                 print(f" ECG Value: {ecg_value} mV")
    #                 return ecg_value
    #             else:
    #                 # Try to parse as multiple values (8-channel data)
    #                 try:
    #                     # Clean the line data - remove any non-numeric characters except spaces and minus signs
    #                     import re
    #                     cleaned_line = re.sub(r'[^\d\s\-]', ' ', line_data)
    #                     values = [int(x) for x in cleaned_line.split() if x.strip() and x.replace('-', '').isdigit()]
    #                     
    #                     if len(values) >= 8:
    #                         print(f" 8-Channel ECG Data: {values}")
    #                         return values  # Return the list of 8 values
    #                     elif len(values) == 1:
    #                         print(f" Single ECG Value: {values[0]} mV")
    #                         return values[0]
    #                     elif len(values) > 0:
    #                         print(f" Unexpected number of values: {len(values)} (expected 8)")
    #                         return None
    #                     else:
    #                         return None
    #                 except ValueError:
    #                     print(f" Non-numeric data received: '{line_data}'")
    #         else:
    #             print(" No data received (timeout)")
    #             
    #     except Exception as e:
    #         self._handle_serial_error(e)
    #     return None
    
    def read_value(self):
        """
        NEW METHOD - Compatibility wrapper that uses packet-based parsing
        Returns data in the same format as old method for backward compatibility
        """
        # If this is actually a SerialStreamReader, use packet-based reading
        if isinstance(self, SerialStreamReader):
            packets = self.read_packets(max_packets=1)
            if packets and len(packets) > 0:
                # Convert packet dict to list format [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
                packet = packets[0]
                lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                values = [packet.get(lead, 0) for lead in lead_order]
                return values[:8] if len(values) >= 8 else values  # Return first 8 for compatibility
        return None

    def close(self):
        print(" Closing serial connection...")
        self.ser.close()
        print(" Serial connection closed")

    def _handle_serial_error(self, error):
        """Handle serial communication errors with alert and logging"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        # Log the error
        error_msg = f"Serial communication error: {error}"
        print(f" {error_msg}")
        
        # Log to crash logger
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        # Show alert if consecutive errors exceed threshold
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self._show_serial_error_alert(error)
            self.last_error_time = current_time
            self.consecutive_errors = 0  # Reset counter after showing alert
    
    def _show_serial_error_alert(self, error):
        """Show alert dialog for serial communication errors"""
        try:
            # Get user details from main application
            user_details = getattr(self, 'user_details', {})
            username = user_details.get('full_name', 'Unknown User')
            phone = user_details.get('phone', 'N/A')
            email = user_details.get('email', 'N/A')
            serial_id = user_details.get('serial_id', 'N/A')
            
            # Create detailed error message
            error_details = f"""
Serial Communication Error Detected!

Error: {str(error)}
User: {username}
Phone: {phone}
Email: {email}
Serial ID: {serial_id}
Machine Serial: {self.crash_logger.machine_serial_id or 'N/A'}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

This error has been logged and an email notification will be sent to the support team.
            """
            
            # Show alert dialog
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Serial Communication Error")
            msg_box.setText("ECG Device Connection Lost")
            msg_box.setDetailedText(error_details)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            
            # Send email notification
            self._send_error_email(error, user_details)
            
        except Exception as e:
            print(f" Error showing serial error alert: {e}")
    
    def _send_error_email(self, error, user_details):
        """Send email notification for serial errors"""
        try:
            # Create error data for email
            error_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error_type': 'Serial Communication Error',
                'error_message': str(error),
                'user_details': user_details,
                'machine_serial': self.crash_logger.machine_serial_id or 'N/A',
                'consecutive_errors': self.consecutive_errors,
                'total_errors': self.error_count
            }
            
            # Send email using crash logger
            self.crash_logger._send_crash_email(error_data)
            print(" Serial error email notification sent")
            
        except Exception as e:
            print(f" Error sending serial error email: {e}")

class LiveLeadWindow(QWidget):
    def __init__(self, lead_name, data_source, buffer_size=80, color="#00ff99"):
        super().__init__()
        self.setWindowTitle(f"Live View: {lead_name}")
        self.resize(900, 300)
        self.lead_name = lead_name
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.color = color

        layout = QVBoxLayout(self)
        self.fig = Figure(facecolor='#000')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#000')
        self.ax.set_ylim(-400, 400)
        self.ax.set_xlim(0, buffer_size)
        self.line, = self.ax.plot([0]*buffer_size, color=self.color, lw=0.7)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

    def update_plot(self):
        data = self.data_source()
        if data and len(data) > 0:
            plot_data = np.full(self.buffer_size, np.nan)
            n = min(len(data), self.buffer_size)
            centered = np.array(data[-n:]) - np.mean(data[-n:])
            plot_data[-n:] = centered
            self.line.set_ydata(plot_data)
            self.canvas.draw_idle()

# ------------------------ Calculate ST Segment ------------------------

def calculate_st_segment(lead_signal, r_peaks, fs=500, j_offset_ms=40, st_offset_ms=80):
    """
    Calculate mean ST segment amplitude (in mV) at (J-point + st_offset_ms) after R peak.
    - lead_signal: ECG samples (e.g., Lead II)
    - r_peaks: indices of R peaks
    - fs: sampling rate (Hz)
    - j_offset_ms: ms after R peak to estimate J-point (default 40ms)
    - st_offset_ms: ms after J-point to measure ST segment (default 80ms)
    Returns mean ST segment amplitude in mV (float), or '--' if not enough data.
    """
    if len(lead_signal) < 100 or len(r_peaks) == 0:
        return "--"
    st_values = []
    j_offset = int(j_offset_ms * fs / 1000)
    st_offset = int(st_offset_ms * fs / 1000)
    for r in r_peaks:
        st_idx = r + j_offset + st_offset
        if st_idx < len(lead_signal):
            st_values.append(lead_signal[st_idx])
    if len(st_values) == 0:
        return "--"
    
    st_value = float(np.mean(st_values))
    # Interpret as medical term
    if 80 <= st_value <= 120:
        return "Isoelectric"
    elif st_value > 120:
        return "Elevated"
    elif st_value < 80:
        return "Depressed"
    return str(st_value)

# ------------------------ Calculate Arrhythmia ------------------------

def detect_arrhythmia(heart_rate, qrs_duration, rr_intervals, pr_interval=None, p_peaks=None, r_peaks=None, ecg_signal=None):
    """
    Expanded arrhythmia detection logic for common clinical arrhythmias.
    - Sinus Bradycardia: HR < 60, regular RR
    - Sinus Tachycardia: HR > 100, regular RR
    - Atrial Fibrillation: Irregular RR, absent/irregular P waves
    - Atrial Flutter: Sawtooth P pattern (not robustly detected here)
    - PAC: Early P, narrow QRS, compensatory pause (approximate)
    - PVC: Early wide QRS, no P, compensatory pause (approximate)
    - VT: HR > 100, wide QRS (>120ms), regular
    - VF: Chaotic, no clear QRS, highly irregular
    - Asystole: Flatline (very low amplitude, no R)
    - SVT: HR > 150, narrow QRS, regular
    - Heart Block: PR > 200 (1°), dropped QRS (2°), AV dissociation (3°)
    - Junctional Rhythm: HR 40-60 with absent or short PR and narrow QRS
    """
    try:
        if rr_intervals is None or len(rr_intervals) < 2:
            return "Detecting..."
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        rr_reg = rr_std < 0.12  # Regular if std < 120ms
        # Asystole: flatline (no R peaks, or very low amplitude)
        if r_peaks is not None and len(r_peaks) < 1:
            if ecg_signal is not None and np.ptp(ecg_signal) < 50:
                return "Asystole (Flatline)"
            return "No QRS Detected"
        # VF: highly irregular, no clear QRS, rapid undulating
        if r_peaks is not None and len(r_peaks) > 5:
            if rr_std > 0.25 and ecg_signal is not None and np.ptp(ecg_signal) > 100 and heart_rate and heart_rate > 180:
                return "Ventricular Fibrillation (VF)"
        # VT: HR > 100, wide QRS (>120ms), regular
        if heart_rate and heart_rate > 100 and qrs_duration and qrs_duration > 120 and rr_reg:
            return "Ventricular Tachycardia (VT)"
        # Junctional Rhythm: rate 40-60, narrow QRS, absent/short PR
        if (
            heart_rate and 40 <= heart_rate <= 60
            and qrs_duration and qrs_duration <= 120
            and rr_reg
        ):
            p_count = len(p_peaks) if p_peaks is not None else 0
            r_count = len(r_peaks) if r_peaks is not None else max(1, len(rr_intervals) + 1)
            p_ratio = p_count / max(r_count, 1)
            pr_short = pr_interval is not None and pr_interval <= 120
            if p_ratio < 0.4 or pr_short:
                return "Junctional Rhythm (possible)"
        # Sinus Bradycardia: HR < 60, regular
        if heart_rate and heart_rate < 60 and rr_reg:
            return "Sinus Bradycardia"
        # Sinus Tachycardia: HR > 100, regular
        if heart_rate and heart_rate > 100 and qrs_duration and qrs_duration <= 120 and rr_reg:
            return "Sinus Tachycardia"
        # SVT: HR > 150, narrow QRS, regular
        if heart_rate and heart_rate > 150 and qrs_duration and qrs_duration <= 120 and rr_reg:
            return "Supraventricular Tachycardia (SVT)"
        # AFib: Irregular RR, absent/irregular P
        if not rr_reg and (p_peaks is None or len(p_peaks) < len(r_peaks) * 0.5):
            return "Atrial Fibrillation (AFib)"
        # Atrial Flutter: (not robust, but if HR ~150, regular, and P waves rapid)
        if heart_rate and 140 < heart_rate < 170 and rr_reg and p_peaks is not None and len(p_peaks) > len(r_peaks):
            return "Atrial Flutter (suggestive)"
        # PAC: Early P, narrow QRS, compensatory pause (approximate)
        if p_peaks is not None and r_peaks is not None and len(p_peaks) > 1 and len(r_peaks) > 1:
            pr_diffs = np.diff([r - p for p, r in zip(p_peaks, r_peaks)])
            if np.any(pr_diffs < -0.15 * len(ecg_signal)) and qrs_duration and qrs_duration <= 120:
                return "Premature Atrial Contraction (PAC)"
        # PVC: Early wide QRS, no P, compensatory pause (approximate)
        if qrs_duration and qrs_duration > 120 and (p_peaks is None or len(p_peaks) < len(r_peaks) * 0.5):
            return "Premature Ventricular Contraction (PVC)"
        # Heart Block: PR > 200ms (1°), dropped QRS (2°), AV dissociation (3°)
        if pr_interval and pr_interval > 200:
            return "Heart Block (1° AV)"
        # If QRS complexes are missing (dropped beats)
        if r_peaks is not None and ecg_signal is not None and len(r_peaks) < len(ecg_signal) / 500 * heart_rate * 0.7:
            return "Heart Block (2°/3° AV, dropped QRS)"
        return "None Detected"
    except Exception as e:
        return "Detecting..."

class SwitchButton(QPushButton):
    """Custom toggle switch with a sliding white circle icon"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setMinimumHeight(40)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Arial", 10, QFont.Bold))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Draw Rounded Background (Pill Shape)
        is_on = self.isChecked()
        bg_color = QColor("#2ecc71") if is_on else QColor("#ff4d4d") # Green if ON, Red if OFF
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        rect = self.rect()
        painter.drawRoundedRect(rect, rect.height()/2, rect.height()/2)
        
        # 2. Draw White Knob (The Icon Circle)
        margin = 4
        knob_size = rect.height() - (margin * 2)
        knob_rect = QRect(0, margin, knob_size, knob_size)
        
        # According to your image:
        # ON (Green) -> Circle is on the LEFT
        # OFF (Red) -> Circle is on the RIGHT
        if is_on:
            knob_rect.moveLeft(margin)
        else:
            knob_rect.moveRight(rect.width() - margin)
            
        painter.setBrush(QBrush(QColor("white")))
        painter.drawEllipse(knob_rect)
        
        # 3. Draw Text
        painter.setPen(QColor("white"))
        text = "Demo Mode: ON" if is_on else "Demo Mode: OFF"
        # Align text opposite to the knob
        if is_on:
            # Knob is left, text on right
            text_rect = rect.adjusted(knob_size + margin, 0, -margin, 0)
        else:
            # Knob is right, text on left
            text_rect = rect.adjusted(margin, 0, -(knob_size + margin), 0)
            
        painter.drawText(text_rect, Qt.AlignCenter, text)

# Import Holter components from new modular structure
try:
    from .holter.stream_writer import HolterStreamWriter
    from .holter.holter_ui import HolterMainWindow, HolterStartDialog
    from .holter.analysis_worker import HolterAnalysisWorker
    HOLTER_AVAILABLE = True
except ImportError as e:
    print(f" Holter modules not available: {e}")
    HOLTER_AVAILABLE = False

def create_pink_grid_brush():
    from PyQt5.QtGui import QBrush, QColor
    return QBrush(QColor("#ffffff"))

    from PyQt5.QtGui import QBrush, QPixmap, QPainter, QPen, QColor
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    dpi = 96.0
    try:
        screen = app.primaryScreen() if app else None
        if screen is not None:
            dpi = float(screen.logicalDotsPerInchX() or screen.logicalDotsPerInch() or dpi)
    except Exception:
        dpi = 96.0

    px_per_mm = max(1.0, dpi / 25.4)
    minor_step = max(2, int(round(px_per_mm)))
    grid_size = minor_step * 5
    pixmap = QPixmap(grid_size, grid_size)
    pixmap.fill(QColor('#FFF5F5'))
    
    painter = QPainter(pixmap)
    try:
        # Minor lines every 1 mm
        minor_pen = QPen(QColor('#FFD9D9'), 0.5, Qt.SolidLine)
        painter.setPen(minor_pen)
        for i in range(1, 5):
            pos = i * minor_step
            painter.drawLine(pos, 0, pos, grid_size)
            painter.drawLine(0, pos, grid_size, pos)
            
        # Major lines every 5 mm
        major_pen = QPen(QColor('#FFB3B3'), 1.0, Qt.SolidLine)
        painter.setPen(major_pen)
        painter.drawLine(0, 0, grid_size, 0)
        painter.drawLine(0, 0, 0, grid_size)
        painter.drawLine(grid_size - 1, 0, grid_size - 1, grid_size)
        painter.drawLine(0, grid_size - 1, grid_size, grid_size - 1)
    finally:
        painter.end()
        
    return QBrush(pixmap)


class ECGTestPage(QWidget):
    LEADS_MAP = {
        "Lead II ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "Lead III ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "7 Lead ECG Test": ["V1", "V2", "V3", "V4", "V5", "V6", "II"],
        "12 Lead ECG Test": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "ECG Live Monitoring": ["II"]
    }
    LEAD_COLORS = {
        "I": "#00ff99",
        "II": "#ff0055", 
        "III": "#0099ff",
        "aVR": "#ff9900",
        "aVL": "#cc00ff",
        "aVF": "#00ccff",
        "V1": "#ffcc00",
        "V2": "#00ffcc",
        "V3": "#ff6600",
        "V4": "#6600ff",
        "V5": "#00b894",
        "V6": "#ff0066"
    }

    def __init__(self, test_name, stacked_widget, settings_manager=None):
        super().__init__()
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)  # Minimum size for usability
        
        self.setWindowTitle("12-Lead CardioX")
        self.stacked_widget = stacked_widget  # Save reference for navigation

        # Use a shared SettingsManager when provided (keeps filters/settings consistent
        # across dashboard mini-wave, overlays, expanded lead view, and reports).
        self.settings_manager = settings_manager if settings_manager is not None else SettingsManager()
        # Ensure AC filter starts at "50" each launch (Set Filter default)
        try:
            self.settings_manager.set_setting("filter_ac", "50")
        except Exception as e:
            print(f" Could not enforce default AC filter state: {e}")
        self.current_language = self.settings_manager.get_setting("system_language", "en")
    
        # Initialize demo manager
        self.demo_manager = DemoManager(self)
        # Timer tracking for countdown timers
        self.countdown_timers = []  # Store active QTimer instances for cancellation

        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background: #f3f4f6;")
        self.detailed_widget = QWidget()
        self.page_stack = QStackedLayout()
        self.page_stack.addWidget(self.grid_widget)
        self.page_stack.addWidget(self.detailed_widget)
        self.setLayout(self.page_stack)

        # Antialiasing: ON to ensure smooth waves without stair-step jagged edges.
        # We use QOpenGLWidget viewport to offload rendering to the GPU to prevent lag.
        pg.setConfigOptions(antialias=True)

        self.test_name = test_name
        self.leads = self.LEADS_MAP[test_name]
        self.display_mode = 'scroll'  # Default display mode
        self.base_buffer_size = 2000  # Base buffer used for speed scaling
        self.buffer_size = self.base_buffer_size  # Increased buffer size for all leads
        # Use GitHub version data structure: list of numpy arrays for all 12 leads
        # Initialize data buffers with memory management
        self.data = [np.zeros(HISTORY_LENGTH, dtype=np.float32) for _ in range(12)]
        # Keep a longer session history for PDF generation so freeze/resume does
        # not leave the report with only the latest short tail of data.
        self._report_buffers = [deque(maxlen=30000) for _ in range(12)]
        # Circular replay buffers used for freeze/replay playback.
        self._replay_buffers = [deque(maxlen=8000) for _ in range(12)]
        # Multi-lead snapshot for ExpandedLeadView (Phase 4 fusion).
        # Updated continuously from the same 12-lead packet stream.
        self.latest_multilead_snapshot = {}
        
        # Filter Pipeline Configuration (from standalone_ecg_plot.py)
        self.SAMPLE_RATE = 500
        self.SMOOTH_SIGMA = 0.8
        self.INTERP_FACTOR = 2
        # 50 Hz NOTCH FILTER
        self.b_notch, self.a_notch = iirnotch(w0=50.0, Q=30.0, fs=self.SAMPLE_RATE)

        # Track overlay state and current layout (12:1 vs 6:2)
        self._overlay_active = False
        self._current_overlay_layout = None
        
        # Memory management
        self.max_buffer_size = 10000  # Maximum buffer size to prevent memory issues
        self.memory_check_interval = 2000  # Check memory every 2000 updates (reduced frequency for better performance)
        self.update_count = 0
        self._plot_update_in_progress = False
        self._plot_render_stride = 4 if is_low_spec_mode() else 2
        self._display_mode = "live"
        self._grid_frozen = False
        self._frozen_metric_texts = {}   # populated by _freeze_current_view, cleared by _resume_live_view
        self._frozen_report_metrics = None  # populated by _freeze_current_view, cleared by _resume_live_view
        self._replay_cursor = 0
        self._replay_window_sec = 10.0
        self._replay_snapshot = None
        self._replay_render_snapshot = None
        # Hold last displayed HR to avoid unnecessary flicker
        self._last_hr_display = None
        # HR smoothing/lock removed; use original calculation
        
        # Initialize crash logger
        self.crash_logger = get_crash_logger()
        self.crash_logger.log_info("ECG Test Page initialized", "ECG_TEST_PAGE_START")

        # Pre-warm matplotlib off-screen renderer so the first report generates
        # instantly without a 200-400ms module-import stall on the report thread.
        def _prewarm_matplotlib():
            try:
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                _fig = Figure(figsize=(1, 1))
                FigureCanvasAgg(_fig)
                del _fig
            except Exception:
                pass
        import threading as _threading
        _t = _threading.Thread(target=_prewarm_matplotlib, daemon=True)
        _t.start()


        self.timer = QTimer()
        # Use CoarseTimer on weak machines: allows Qt to coalesce wake-ups saving ~5-10% CPU.
        # PreciseTimer is only needed on high-spec machines for tightest timing accuracy.
        self.timer.setTimerType(Qt.CoarseTimer if is_low_spec_mode() else Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_plots)
        self.metrics_timer = QTimer(self)
        self.metrics_timer.setTimerType(Qt.CoarseTimer if is_low_spec_mode() else Qt.PreciseTimer)
        self.metrics_timer.timeout.connect(self._on_metrics_timer_timeout)
        self._replay_timer = QTimer(self)
        self._replay_timer.setTimerType(Qt.CoarseTimer if is_low_spec_mode() else Qt.PreciseTimer)
        self._replay_timer.timeout.connect(self._update_replay_plots)
        self.serial_reader = None
        self.stacked_widget = stacked_widget
        self.sampler = SamplingRateCalculator()
        # self.demo_fs = 500  # Increased sampling rate for more realistic ECG
        self.sampling_rate = 500  # Default sampling rate for expanded lead view
        self._latest_rhythm_interpretation = "Analyzing Rhythm..."

        # ── HolterBPMController: stable BPM engine (background thread) ─────────
        try:
            from ecg.holter.holter_bpm_engine import HolterBPMController
            self._bpm_ctrl = HolterBPMController(
                parent_widget=self,
                fs=500,
                chunk_seconds=30,   # 30-second window → ~150 R-peaks, very stable
            )
        except Exception as _bpm_init_err:
            print(f"[ECGTestPage] HolterBPMController init failed: {_bpm_init_err}")
            self._bpm_ctrl = None

        # Flatline detection state: track leads where we've already shown an alert
        self._flatline_alert_shown = [False] * 12
        # Debug/verification: whether display-only DFT was applied for a lead in the last render pass
        self._last_display_dft_applied = [False] * 12
        # Lead connection tracking (from packet connected flags)
        self._lead_connection_state = {lead: True for lead in self.leads}
        self._lead_last_valid_value = {lead: 0.0 for lead in self.leads}
        self._last_off_leads_signature = None
        self._lead_data_received = False
        # Lead-off stability: latch a "Leads Off" state when the firmware chatters
        # (commonly observed when RL/DRL is removed). This prevents UI/BPM flicker.
        self._lead_off_latched = False
        self._lead_off_latch_on_count = 0
        self._lead_off_latch_off_count = 0
        # Tune conservatively: require a few consecutive packets before latching/unlatching.
        # (At 500 pkt/s, 5 packets ≈ 10 ms, 50 packets ≈ 100 ms.)
        self._LEAD_OFF_LATCH_ON_PACKETS = 5
        self._LEAD_OFF_LATCH_OFF_PACKETS = 50
        # Consider "global lead-off" when most leads are off.
        self._LEAD_OFF_GLOBAL_THRESHOLD = 10
        # Fast LL(F) disconnection flag: updated every packet without debounce.
        # Used by update_metrics to immediately suppress BPM on LL removal,
        # preventing the 0↔BPM flicker seen during the 25-packet debounce window.
        self._ll_disconnected = False
        self._la_report_cooldown_armed = False
        self._ll_report_cooldown_armed = False
        # Tracks whether the RA/RL "electrode disconnected" popup has already
        # been shown for the current lead-off event, so it fires once per
        # disconnection rather than every packet while latched.
        self._ra_rl_popup_shown = False
        self._prev_p_axis = None  # Track P-axis for safety assertions
        self._prev_qrs_axis = None
        self._prev_t_axis = None

        # Initialize time tracking for elapsed time
        self.start_time = None
        self.paused_at = None  # Track when pause started
        self.paused_duration = 0  # Total cumulative paused time
        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)

        main_vbox = QVBoxLayout()

        # Enhanced Menu Panel - Make it responsive and compact
        menu_container = QWidget()
        menu_container.setMinimumWidth(200)  # Reduced from 250px
        menu_container.setMaximumWidth(280)  # Reduced from 400px
        menu_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        menu_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                border-right: 2px solid #e9ecef;
            }
        """)

        # Style menu buttons - Make them much more compact
        menu_layout = QVBoxLayout(menu_container)
        menu_layout.setContentsMargins(12, 12, 12, 12)  # Reduced margins
        menu_layout.setSpacing(8)  # Reduced spacing between buttons
        
        # Header - Make it more compact
        self.menu_header_label = QLabel("ECG Control Panel")
        self.menu_header_label.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-size: 18px;  /* Reduced from 24px */
                font-weight: bold;
                padding: 12px 0;  /* Reduced from 20px */
                border-bottom: 2px solid #ff6600;  /* Reduced from 3px */
                margin-bottom: 12px;  /* Reduced from 20px */
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border-radius: 8px;  /* Reduced from 10px */
            }
        """)
        self.menu_header_label.setAlignment(Qt.AlignCenter)
        self.menu_header_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        menu_layout.addWidget(self.menu_header_label)
        
        # Create ECGMenu instance to use its methods
        self.ecg_menu = ECGMenu(parent=self, dashboard=self.stacked_widget.parent())
        # Connect ECGMenu to this ECG test page for data communication
        self.ecg_menu.set_ecg_test_page(self)

        self.ecg_menu.settings_manager = self.settings_manager
        
        # Register demo manager's settings callback so wave gain/speed changes work in demo mode
        if hasattr(self, 'demo_manager'):
            self.ecg_menu.settings_changed_callback = self.demo_manager.on_settings_changed
            print(" Demo manager settings callback registered")

        # Initialize sliding panel for the ECG menu
        self.ecg_menu.sliding_panel = None
        self.ecg_menu.parent_widget = self

        self.ecg_menu.setVisible(False)
        self.ecg_menu.hide()
        
        if self.ecg_menu.parent():
            self.ecg_menu.setParent(None)

        self.ecg_menu.settings_changed_callback = self.on_settings_changed 

        self.apply_display_settings()

        # Create ECG menu buttons
        ecg_menu_buttons = [
            # ("Open ECG", self.ecg_menu.show_open_ecg, "#17a2b8"),  # Commented out - not currently needed
            ("Holter", self.show_holter_menu, "#00FF00"),  # Added Holter button (Green)
            ("Working Mode", self.ecg_menu.show_working_mode, "#ffc107"),
            ("Report Setup", self.ecg_menu.show_report_setup, "#6c757d"),
            ("Set Filter", self.ecg_menu.show_set_filter, "#fd7e14"),
            ("System Setup", self.ecg_menu.show_system_setup, "#6f42c1"),
            ("Load Default", self.ecg_menu.show_load_default, "#20c997"),
            ("Exit", self.ecg_menu.show_exit, "#495057")
        ]
        
        # Create buttons and store them in a list - Make them much smaller
        created_buttons = []
        self.menu_buttons = []
        for text, handler, color in ecg_menu_buttons:
            btn = QPushButton(text)
            btn.setMinimumHeight(40)  # Reduced from 60px - Much more compact
            btn.setMaximumHeight(45)  # Add maximum height constraint
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(handler)
            created_buttons.append(btn)
            menu_layout.addWidget(btn)
            self.menu_buttons.append((btn, text))

        menu_layout.addStretch(1)

        # Holter Monitor State
        self.holter_mode_enabled = False
        self._holter_writer = None
        self._holter_ui = None
        self._holter_worker = None
        self._holter_patient_info = {}
        self._holter_output_dir = os.path.join(os.getcwd(), "recordings")
        self._holter_duration_hours = 24

        self.apply_language(self.current_language)

        # Style menu buttons AFTER they're created - Compact styling
        for i, btn in enumerate(created_buttons):
            color = ecg_menu_buttons[i][2]
            btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #1a1a1a;
                    border: 2px solid #e9ecef;  /* Reduced from 3px */
                    border-radius: 8px;  /* Reduced from 15px */
                    padding: 8px 12px;  /* Reduced from 15px 20px */
                    font-size: 12px;  /* Reduced from 16px */
                    font-weight: bold;
                    text-align: left;
                    margin: 2px 0;  /* Reduced from 4px */
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #fff5f0, stop:1 #ffe0cc);
                    border: 2px solid {color};  /* Reduced from 4px */
                    color: {color};
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #ffe0cc, stop:1 #ffcc99);
                    border: 2px solid {color};  /* Reduced from 4px */
                    color: {color};
                }}
            """)

        # created_buttons[0].clicked.disconnect()  # Open ECG - removed
        # created_buttons[0].clicked.connect(self.ecg_menu.show_open_ecg)  # Open ECG - removed
        
        created_buttons[0].clicked.disconnect()
        created_buttons[0].clicked.connect(self.show_holter_menu)
        # Disable Holter entry from 12-lead ECG menu per request.
        created_buttons[0].setEnabled(False)
        created_buttons[0].setVisible(False)
        
        created_buttons[1].clicked.disconnect()
        created_buttons[1].clicked.connect(self.ecg_menu.show_working_mode)
        
        created_buttons[2].clicked.disconnect()
        created_buttons[2].clicked.connect(self.ecg_menu.show_report_setup)
        
        created_buttons[3].clicked.disconnect()
        created_buttons[3].clicked.connect(self.ecg_menu.show_set_filter)
        
        created_buttons[4].clicked.disconnect()
        created_buttons[4].clicked.connect(self.ecg_menu.show_system_setup)
        
        created_buttons[5].clicked.disconnect()
        created_buttons[5].clicked.connect(self.ecg_menu.show_load_default)

        created_buttons[6].clicked.disconnect()
        created_buttons[6].clicked.connect(self.ecg_menu.show_exit)

        # Recording Toggle Button Section - Make it compact
        recording_frame = QFrame()
        recording_frame.setStyleSheet("""
            QFrame {
                background: transparent;
                border: none;
                padding: 6px;  /* Reduced from 10px */
                margin-top: 3px;  /* Reduced from 5px */
            }
        """)
        recording_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        recording_layout = QVBoxLayout(recording_frame)
        recording_layout.setSpacing(4)  # Reduced spacing

        # Demo toggle button - Custom Switch Style
        # self.demo_toggle = SwitchButton()
        # self.demo_toggle.setChecked(False)
        # self.demo_toggle.setMinimumHeight(40)  # Same as other buttons
        # # self.demo_toggle.setMaximumHeight(40)  # Same as other buttons
        # self.demo_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Set demo button style (toggle-style like recording button)
        # self.demo_toggle.setStyleSheet("""
        #     QPushButton {
        #         background: #ff4d4d; /* Red for OFF */
        #         color: white;
        #         border: 2px solid #e60000;
        #         border-radius: 20px; /* Rounded pill shape like the image */
        #         padding: 8px 16px;
        #         font-size: 13px;
        #         font-weight: bold;
        #         text-align: center;
        #         margin: 4px 0;
        #     }
        #     QPushButton:hover {
        #         background: #ff6666;
        #         border: 2px solid #ff1a1a;
        #     }
        #     QPushButton:checked {
        #         background: #2ecc71; /* Green for ON */
        #         border: 2px solid #27ae60;
        #         color: white;
        #     }
        #     QPushButton:checked:hover {
        #         background: #40e080;
        #         border: 2px solid #2ecc71;
        #     }
        # """)
        
        # Connect demo toggle to demo manager
        # self.demo_toggle.toggled.connect(self.on_demo_toggle_changed)
        
        # recording_layout.addWidget(self.demo_toggle)

        # Capture Screen button - Make it compact
        self.capture_screen_btn = QPushButton("Capture Screen")
        self.capture_screen_btn.setMinimumHeight(35)  # Reduced from 60px
        self.capture_screen_btn.setMaximumHeight(40)  # Add maximum height
        self.capture_screen_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.capture_screen_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                color: #1a1a1a;
                border: 2px solid #e9ecef;  /* Reduced from 3px */
                border-radius: 8px;  /* Reduced from 15px */
                padding: 8px 12px;  /* Reduced from 15px 20px */
                font-size: 12px;  /* Reduced from 16px */
                font-weight: bold;
                text-align: center;
                margin: 2px 0;  /* Reduced from 5px */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border: 2px solid #2453ff;  /* Reduced from 4px */
                color: #2453ff;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #e0e8ff, stop:1 #ccd9ff);
                border: 2px solid #2453ff;  /* Reduced from 4px */
                color: #2453ff;
            }
        """)
        self.capture_screen_btn.clicked.connect(self.capture_screen)
        recording_layout.addWidget(self.capture_screen_btn)
        
        # Toggle-style recording button - Make it compact
        self.recording_toggle = QPushButton("Record Screen")
        self.recording_toggle.setMinimumHeight(35)  # Reduced from 60px
        self.recording_toggle.setMaximumHeight(40)  # Add maximum height
        self.recording_toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.recording_toggle.setCheckable(True)
        self.recording_toggle.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffffff, stop:1 #f8f9fa);
                color: #1a1a1a;
                border: 2px solid #e9ecef;  /* Reduced from 3px */
                border-radius: 8px;  /* Reduced from 15px */
                padding: 8px 12px;  /* Reduced from 15px 20px */
                font-size: 12px;  /* Reduced from 16px */
                font-weight: bold;
                text-align: center;
                margin: 2px 0;  /* Reduced from 5px */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border: 2px solid #ff6600;  /* Reduced from 4px */
                color: #ff6600;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffe0cc, stop:1 #ffcc99);
                border: 2px solid #ff6600;  /* Reduced from 4px */
                color: #ff6600;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #fff5f0, stop:1 #ffe0cc);
                border: 2px solid #dc3545;  /* Reduced from 4px */
                color: #dc3545;
            }
            QPushButton:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ffe0cc, stop:1 #ffcc99);
                border: 2px solid #c82333;  /* Reduced from 4px */
                color: #c82333;
            }
        """)
        self.recording_toggle.clicked.connect(self.toggle_recording)
        recording_layout.addWidget(self.recording_toggle)
        
        menu_layout.addWidget(recording_frame)
        
        # Initialize recording variables
        self.is_recording = False
        self.recording_writer = None
        # Bounded: a screen recording must not be able to consume all of RAM.
        # At 33 fps a 1920x1080 BGR frame is 6.22 MB, so an uncapped list grew at
        # ~187 MB/s and reached ~5.6 GB in 30 s. The cap is applied in
        # start_recording() once the timer interval is known; deque(maxlen=...)
        # then drops the oldest frame instead of growing.
        self._RECORDING_MAX_SECONDS = 120
        self._RECORDING_MAX_WIDTH = 1280
        self.recording_frames = deque()
        
        # Initialize recording button state (disabled by default until acquisition/demo starts)
        QTimer.singleShot(100, self.update_recording_button_state)

        # Add metrics frame above the plot area
        self.metrics_frame = self.create_metrics_frame()
        self.metrics_frame.setMaximumHeight(80)  # Reduced from default
        self.metrics_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.metrics_frame.setMaximumHeight(120)
        self.metrics_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        main_vbox.addWidget(self.metrics_frame)
        
        # Lead connection status (top label only; no popup alerts)
        self.lead_status_label = QLabel("Leads: All connected")
        self.lead_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.lead_status_label.setStyleSheet(
            "color: #2e7d32; font-size: 12px; font-weight: bold; padding: 2px 6px;"
        )
        self.lead_status_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_vbox.addWidget(self.lead_status_label)
        self._set_lead_status_idle()
        
        # Ensure metrics are reset to zero after frame creation
        self.reset_metrics_to_zero()
        
        # --- REPLACED: Matplotlib plot area is replaced with a simple QWidget container ---
        self.plot_area = QWidget()
        self.plot_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_vbox.addWidget(self.plot_area)

        # --- NEW: Create the PyQtGraph plot grid (from GitHub version) ---
        grid = QGridLayout(self.plot_area)
        grid.setSpacing(2)   # Tight spacing so black borders are visible between cells
        grid.setContentsMargins(2, 2, 2, 2)
        self.plot_widgets = []
        self.data_lines = []
        
        # Define colors for each lead type for consistent color coding (all black as requested)
        lead_colors = {
            'I': '#000000',
            'II': '#000000',
            'III': '#000000',
            'aVR': '#000000',
            'aVL': '#000000',
            'aVF': '#000000',
            'V1': '#000000',
            'V2': '#000000',
            'V3': '#000000',
            'V4': '#000000',
            'V5': '#000000',
            'V6': '#000000'
        }
        
        positions = [(i, j) for i in range(4) for j in range(3)]
        for i in range(len(self.leads)):
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground("w")
            plot_widget.setStyleSheet("border: 1px solid black;")
            plot_widget.showGrid(x=False, y=False)
            plot_widget.setClipToView(True)
            plot_widget.setDownsampling(ds=1, auto=False, mode='subsample')
            # Hide Y-axis labels for cleaner display
            plot_widget.getAxis('left').setTicks([])
            plot_widget.getAxis('left').setLabel('')
            plot_widget.getAxis('left').setStyle(showValues=False)
            # Hide X-axis labels (tick numbers) completely
            plot_widget.getAxis('bottom').setTicks([])
            plot_widget.getAxis('bottom').setLabel('')
            plot_widget.getAxis('bottom').setStyle(showValues=False)
            plot_widget.getAxis('bottom').hide()
            # Black 1px border around each plot box
            plot_widget.setStyleSheet("border: 1px solid black;")

            # Get color for this lead
            lead_name = self.leads[i]
            lead_color = lead_colors.get(lead_name, '#000000')
            
            plot_widget.setTitle(self.leads[i], color=lead_color, size='10pt')
            # Clinical 12-lead: symmetric Y-axis centered at 0 for ALL leads.
            # aVR is naturally negative from its formula; no axis inversion needed.
            y_min, y_max = -2048, 2048
            plot_widget.setYRange(y_min, y_max)
            vb = plot_widget.getViewBox()
            if vb is not None:
                # Lock Y-axis limits
                vb.setLimits(yMin=y_min, yMax=y_max)
                # Start with a default X range of 10 seconds and lock Y-range
                try:
                    vb.setRange(xRange=(0.0, 10.0), yRange=(y_min, y_max), padding=0)
                except Exception:
                    try:
                        # Fallback: set ranges separately
                        vb.setRange(xRange=(0.0, 10.0))
                        vb.setRange(yRange=(y_min, y_max), padding=0)
                    except Exception:
                        pass
            
            # --- MAKE PLOT CLICKABLE ---
            plot_widget.scene().sigMouseClicked.connect(partial(self.plot_clicked, i))
            
            row, col = positions[i]
            grid.addWidget(plot_widget, row, col)
            # Enable anti-aliasing for smooth waves (data smoothing is applied separately)
            data_line = plot_widget.plot(pen=pg.mkPen(color=lead_color, width=1.0, antialias=True))

            self.plot_widgets.append(plot_widget)
            self.data_lines.append(data_line)
        
        # R-peaks scatter plot (only if we have at least 2 plots)
        if len(self.plot_widgets) > 1:
            self.r_peaks_scatter = self.plot_widgets[1].plot([], [], pen=None, symbol='o', symbolBrush='r', symbolSize=8)
        else:
            self.r_peaks_scatter = None
        
        main_vbox.setSpacing(12)  # Reduced from 16px
        main_vbox.setContentsMargins(16, 16, 16, 16)  # Reduced from 24px

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Freeze")
        # self.ports_btn = QPushButton("Ports") # Commented out
        self.generate_report_btn = QPushButton("Generate Report")
        # self.export_csv_btn = QPushButton("Export as CSV")  # Commented out
        # self.sequential_btn = QPushButton("Show All Leads Sequentially")  # Commented out
        self.twelve_leads_btn = QPushButton("12:1")
        self.six_leads_btn = QPushButton("6:2")
        self.back_btn = QPushButton("Return to Dashboard")

        # Make all buttons responsive and compact
        for btn in [self.start_btn, self.stop_btn, self.generate_report_btn,
                   self.twelve_leads_btn, self.six_leads_btn, self.back_btn]:
            btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
            btn.setMinimumHeight(32)
            btn.setMaximumHeight(36)

        green_color = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #3d8b40, stop:1 #357a38);
                border: 2px solid #3d8b40;
                color: white;
            }
        """
        # Store common green button style for reuse (demo / real)
        self.green_button_style = green_color
        
        # Apply medical green style to all buttons
        self.start_btn.setStyleSheet(green_color)
        self.stop_btn.setStyleSheet(green_color)
        # self.ports_btn.setStyleSheet(green_color) # Commented out
        self.generate_report_btn.setStyleSheet(green_color)
        # self.export_csv_btn.setStyleSheet(green_color)  # Commented out
        # self.sequential_btn.setStyleSheet(green_color)  # Commented out
        self.twelve_leads_btn.setStyleSheet(green_color)
        self.six_leads_btn.setStyleSheet(green_color)
        self.back_btn.setStyleSheet(green_color)

        btn_layout.setSpacing(4)
        btn_layout.setContentsMargins(4, 4, 4, 4)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        # btn_layout.addWidget(self.ports_btn) # Commented out
        btn_layout.addWidget(self.generate_report_btn)
        # btn_layout.addWidget(self.export_csv_btn)  # Commented out
        # btn_layout.addWidget(self.sequential_btn)  # Commented out
        btn_layout.addWidget(self.twelve_leads_btn)
        btn_layout.addWidget(self.six_leads_btn)
        btn_layout.addWidget(self.back_btn)
        main_vbox.addLayout(btn_layout)

        self.start_btn.clicked.connect(self.start_acquisition)
        self.stop_btn.clicked.connect(self.toggle_freeze_replay)

        # Initial state: Disable Freeze and Generate Report buttons
        self.stop_btn.setEnabled(False)
        self.generate_report_btn.setEnabled(False)
        
        grey_style = """
            QPushButton {
                background: #e0e0e0;
                color: #a0a0a0;
                border: 2px solid #cccccc;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
        """
        self.stop_btn.setStyleSheet(grey_style)
        self.generate_report_btn.setStyleSheet(grey_style)


        self.start_btn.setToolTip("Start ECG recording from the selected port")
        self.stop_btn.setToolTip("Freeze the live ECG screen, then click Resume to play the stored 12-lead history")
        # self.ports_btn.setToolTip("Configure COM port and baud rate settings") # Commented out
        self.generate_report_btn.setToolTip("Generate ECG PDF report and add to Recent Reports")
        # self.export_csv_btn.setToolTip("Export ECG data as CSV file")  # Commented out

        # Add help button
        help_btn = QPushButton("?")
        help_btn.setStyleSheet("""
            QPushButton {
                background: #6c757d;
                color: white;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #495057;
            }
        """)
        help_btn.clicked.connect(self.show_help)

        # self.ports_btn.clicked.connect(self.show_ports_dialog) # Commented out
        self.generate_report_btn.clicked.connect(self.generate_pdf_report)
        # self.export_csv_btn.clicked.connect(self.export_csv)  # Commented out
        # self.sequential_btn.clicked.connect(self.show_sequential_view)  # Commented out
        self.twelve_leads_btn.clicked.connect(self.twelve_leads_overlay)
        self.six_leads_btn.clicked.connect(self.six_leads_overlay)
        self.back_btn.clicked.connect(self.go_back)

        main_hbox = QHBoxLayout(self.grid_widget)
    
        # Add widgets to the layout with responsive sizing - Better proportions
        main_hbox.addWidget(menu_container, 1)  # Menu takes 1 part (compact)
        main_hbox.addLayout(main_vbox, 5)  # Main content takes 5 parts (more space)
        
        # Set spacing and layout
        main_hbox.setSpacing(10)  # Reduced from 15px
        main_hbox.setContentsMargins(8, 8, 8, 8)  # Reduced from 10px
        self.grid_widget.setLayout(main_hbox)
        
        # Make the grid widget responsive
        self.grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def plot_clicked(self, plot_index):
        """Handle plot click events"""
        if plot_index < len(self.leads):
            lead_name = self.leads[plot_index]
            print(f"Clicked on {lead_name} plot")
            
            # Get the ECG data for this lead
            if plot_index < len(self.data) and len(self.data[plot_index]) > 0:
                if getattr(self, '_grid_frozen', False) and getattr(self, '_replay_snapshot', None) is not None:
                    ecg_data = self._replay_snapshot[plot_index]
                else:
                    ecg_data = self.data[plot_index]
                
                # Import and show expanded lead view
                try:
                    from ecg.expanded_lead_view import show_expanded_lead_view
                    show_expanded_lead_view(lead_name, ecg_data, self.sampling_rate, self)
                except ImportError as e:
                    print(f"Error importing expanded lead view: {e}")
                    # Fallback: show a simple message
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Lead Analysis", 
                                          f"Lead {lead_name} analysis would be shown here.\n"
                                          f"Data points: {len(ecg_data)}")
            else:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.warning(self, "No Data", f"No ECG data available for Lead {lead_name}")

    def tr(self, text):
        return translate_text(text, getattr(self, "current_language", "en"))

    def update_demo_toggle_label(self):
        if hasattr(self, 'demo_toggle') and self.demo_toggle:
            is_on = self.demo_toggle.isChecked()
            text = "Demo Mode: ON" if is_on else "Demo Mode: OFF"
            self.demo_toggle.setText(self.tr(text))

    def _can_generate_report(self) -> bool:
        """Generate report is allowed when acquisition has run for >= 10s or demo mode is ON."""
        if getattr(self, "_start_cooldown_active", False):
            return False
        try:
            is_demo_mode = False
            if hasattr(self, 'demo_toggle') and self.demo_toggle is not None:
                is_demo_mode = bool(self.demo_toggle.isChecked())

            is_acquisition_running = False
            if hasattr(self, 'timer') and self.timer is not None:
                is_acquisition_running = bool(self.timer.isActive())
            if not is_acquisition_running and hasattr(self, 'serial_reader') and self.serial_reader is not None:
                is_acquisition_running = bool(getattr(self.serial_reader, 'running', False))

            # Enforce 10-second minimum recording duration before report can be generated
            if is_acquisition_running and not is_demo_mode:
                if hasattr(self, 'start_time') and self.start_time:
                    import time
                    cur_t = time.time()
                    p_dur = getattr(self, 'paused_duration', 0)
                    elapsed_sec = int(max(0, cur_t - self.start_time - p_dur))
                    if elapsed_sec < 10:
                        return False

            frozen_snapshot = getattr(self, "_frozen_report_snapshot", None)
            has_frozen_report = False
            if frozen_snapshot:
                try:
                    has_frozen_report = any(
                        np.asarray(arr).size > 0 for arr in frozen_snapshot
                    )
                except Exception:
                    has_frozen_report = True

            return is_demo_mode or is_acquisition_running or has_frozen_report
        except Exception:
            return False

    def _update_generate_report_button_state(self):
        """Keep the Generate Report button aligned with the current capture state."""
        if not hasattr(self, "generate_report_btn"):
            return

        can_enable = self._can_generate_report()
        self.generate_report_btn.setEnabled(bool(can_enable))
        if can_enable and hasattr(self, "green_button_style") and self.green_button_style:
            self.generate_report_btn.setStyleSheet(self.green_button_style)
        elif not can_enable:
            self.generate_report_btn.setStyleSheet("background: #cccccc; color: #666666; border-radius: 10px; padding: 8px 0; font-size: 10px; font-weight: bold;")
        self._refresh_report_btn_label()

    def _start_generate_report_cooldown(self, seconds: int = 10, reason: str = ""):
        """Start countdown timer on the Generate Report button."""
        self._start_cooldown_active = True
        self._cooldown_seconds_remaining = seconds
        if hasattr(self, "_cooldown_timer") and self._cooldown_timer is not None:
            try:
                self._cooldown_timer.stop()
            except Exception:
                pass
        self._cooldown_timer = QTimer(self)
        self._cooldown_timer.setTimerType(Qt.CoarseTimer if is_low_spec_mode() else Qt.PreciseTimer)

        def update_cooldown_tick():
            self._cooldown_seconds_remaining -= 1
            if self._cooldown_seconds_remaining <= 0:
                self._start_cooldown_active = False
                if hasattr(self, "_cooldown_timer") and self._cooldown_timer is not None:
                    self._cooldown_timer.stop()
            self._update_generate_report_button_state()

        self._cooldown_timer.timeout.connect(update_cooldown_tick)
        self._cooldown_timer.start(1000)
        self._update_generate_report_button_state()
        if reason:
            print(f" Generate Report cooldown started ({seconds}s, reason: {reason})")

    def _start_report_cooldown_on_lead_disconnect(self, lead_names):
        """Start the Generate Report cooldown when LA or LL disconnects."""
        if not hasattr(self, "generate_report_btn"):
            return

        leads = {str(name).strip().upper() for name in (lead_names or []) if str(name).strip()}
        if not leads:
            return

        if leads == {"LA", "LL"}:
            reason = "LA/LL disconnect"
        elif "LA" in leads:
            reason = "LA disconnect"
        else:
            reason = "LL disconnect"

        self._start_generate_report_cooldown(seconds=10, reason=reason)

    def _capture_report_snapshot(self, window_sec: float = 7.0):
        """Store the most recent window of waveform data for report generation."""
        snapshot = []
        fs = 500.0
        try:
            if hasattr(self, 'sampler') and getattr(self.sampler, 'sampling_rate', None):
                fs = float(self.sampler.sampling_rate)
        except Exception:
            fs = 500.0
        if fs < 100.0 or fs > 2000.0:
            fs = 500.0

        sample_count = max(2, int(round(max(1.0, float(window_sec)) * fs)))
        for idx in range(12):
            arr = np.array([], dtype=float)
            try:
                if hasattr(self, "_report_buffers") and idx < len(self._report_buffers) and len(self._report_buffers[idx]) > 0:
                    arr = np.asarray(list(self._report_buffers[idx]), dtype=float)
                elif idx < len(self.data):
                    arr = np.asarray(self.data[idx], dtype=float)
            except Exception:
                arr = np.array([], dtype=float)

            if arr.size > sample_count:
                arr = arr[-sample_count:]
            if arr.size > 0:
                nz = np.flatnonzero(np.abs(arr) > 1e-6)
                if nz.size > 0:
                    arr = arr[nz[0]:]
            snapshot.append(np.asarray(arr, dtype=float))

        self._frozen_report_snapshot = snapshot
        return snapshot

    def _get_replay_view_seconds(self) -> float:
        try:
            wave_speed = 25.0 if getattr(self, '_overlay_active', False) else float(self.settings_manager.get_wave_speed())
        except Exception:
            wave_speed = 25.0
        wave_speed = max(1e-6, wave_speed)
        return 3.0 * (25.0 / wave_speed)

    def _get_sampling_rate_for_display(self) -> float:
        fs = 500.0
        try:
            if hasattr(self, "sampler") and getattr(self.sampler, "sampling_rate", None):
                fs = float(self.sampler.sampling_rate)
        except Exception:
            fs = 500.0
        if fs < 100.0 or fs > 2000.0:
            fs = 500.0
        return fs

    def _style_freeze_button(self, text: str, enabled: bool = True):
        btn = getattr(self, "stop_btn", None)
        if not btn:
            return
        btn.setText(text)
        btn.setEnabled(enabled)
        if enabled:
            btn.setStyleSheet(getattr(self, "green_button_style", btn.styleSheet()))
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background: #e0e0e0;
                    color: #a0a0a0;
                    border: 2px solid #cccccc;
                    border-radius: 6px;
                    padding: 4px 8px;
                    font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
        """)

    def _apply_snapshot_display_pipeline(self, segment, lead_index: int, sampling_rate: float):
        """
        Apply the same display-only transforms used by the live 12-lead grid.

        This keeps replay/frozen views visually aligned with Start, including
        baseline anchoring, optional display filters, gain, smoothing, and
        interpolation.
        """
        try:
            data = np.asarray(segment, dtype=float)
        except Exception:
            data = np.array([], dtype=float)

        if data.size < 2:
            return np.asarray(data, dtype=float)

        if not hasattr(self, "_baseline_anchors"):
            self._baseline_anchors = [0.0] * 12
            self._baseline_alpha_slow = 0.0005

        try:
            from ecg.ecg_filters import apply_ac_filter, apply_baseline_wander_median_mean, apply_dft_filter, apply_emg_filter

            if hasattr(self, "_last_display_dft_applied") and lead_index < len(self._last_display_dft_applied):
                self._last_display_dft_applied[lead_index] = False

            dft_setting = None
            if hasattr(self, "settings_manager"):
                dft_setting = str(self.settings_manager.get_setting("filter_dft", "off")).strip()

            if dft_setting and dft_setting not in ("off", ""):
                if str(dft_setting).strip() == "0.5":
                    data = apply_baseline_wander_median_mean(data, float(sampling_rate))
                else:
                    data = apply_dft_filter(data, float(sampling_rate), dft_setting)
                if hasattr(self, "_last_display_dft_applied") and lead_index < len(self._last_display_dft_applied):
                    self._last_display_dft_applied[lead_index] = True

            emg_setting = None
            if hasattr(self, "settings_manager"):
                emg_setting = str(self.settings_manager.get_setting("filter_emg", "150")).strip()
            if emg_setting and emg_setting.lower() != "off" and data.size >= 10:
                data = apply_emg_filter(data, float(sampling_rate), emg_setting)

            ac_setting = None
            if hasattr(self, "settings_manager"):
                ac_setting = str(self.settings_manager.get_setting("filter_ac", "50")).strip()
            if ac_setting in ("50", "60") and data.size > 30:
                data = apply_ac_filter(data, float(sampling_rate), ac_setting)
        except Exception:
            # Keep replay stable even if one of the optional filters is unavailable.
            pass

        try:
            baseline_estimate = self._extract_low_frequency_baseline(data, sampling_rate)
            current_alpha = getattr(self, "_baseline_alpha_slow", 0.0005)
            if self._baseline_anchors[lead_index] == 0.0:
                self._baseline_anchors[lead_index] = baseline_estimate
            self._baseline_anchors[lead_index] = (
                (1 - current_alpha) * self._baseline_anchors[lead_index]
                + current_alpha * baseline_estimate
            )
            data = data - self._baseline_anchors[lead_index]
            current_dc = np.nanmean(data) if data.size > 0 else 0.0
            data = data - current_dc
        except Exception:
            pass

        try:
            gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
        except Exception:
            gain_factor = 1.0

        data = np.nan_to_num(data * gain_factor, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            if data.size > 5:
                data = gaussian_filter1d(data, sigma=self.SMOOTH_SIGMA)
        except Exception:
            pass

        try:
            data = self.interpolate(data, self.INTERP_FACTOR)
        except Exception:
            pass

        try:
            edge_trim = int(0.5 * sampling_rate)
            if data.size > 2 * edge_trim:
                data = data[edge_trim:-edge_trim]
        except Exception:
            pass

        return np.nan_to_num(np.asarray(data, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    def _render_snapshot_window(self, snapshot, end_index=None, window_sec: float = 3.0, assume_processed: bool = False):
        if not snapshot:
            return

        fs = self._get_sampling_rate_for_display()
        visible_seconds = max(0.5, float(window_sec))
        visible_samples = max(2, int(round(visible_seconds * fs)))

        for i in range(min(len(snapshot), len(self.data_lines))):
            try:
                arr = np.asarray(snapshot[i], dtype=float)
                if arr.size < 2:
                    continue

                current_end = arr.size if end_index is None else min(arr.size, max(1, int(end_index)))
                start = max(0, current_end - visible_samples)
                segment = arr[start:current_end]
                if segment.size < 2:
                    continue

                processed = segment if assume_processed else self._apply_snapshot_display_pipeline(segment, i, fs)

                # Keep the live-like flow: show the actual buffered segment without
                # left-padding, so the trace spans the full box width naturally.
                if processed.size > visible_samples:
                    processed = processed[-visible_samples:]

                x_axis = np.linspace(0.0, visible_seconds, len(processed))
                self.data_lines[i].setData(x_axis, processed, connect='finite')

                if i < len(self.plot_widgets):
                    y_min, y_max = -2048, 2048
                    self.plot_widgets[i].setYRange(y_min, y_max, padding=0)
                    try:
                        vb = self.plot_widgets[i].getViewBox()
                        if vb is not None:
                            vb.setLimits(xMin=0.0, xMax=max(visible_seconds, 0.5), yMin=y_min, yMax=y_max)
                            vb.setRange(xRange=(0.0, visible_seconds), yRange=(y_min, y_max), padding=0)
                    except Exception:
                        pass
            except Exception as e:
                print(f" Error rendering snapshot window for lead {i}: {e}")

    def _freeze_current_view(self):
        """Pause the live display without stopping acquisition."""
        try:
            if hasattr(self, "_replay_timer") and self._replay_timer.isActive():
                self._replay_timer.stop()
        except Exception:
            pass

        try:
            self._display_mode = "frozen"
            self._grid_frozen = True
            self._replay_cursor = 0
            # Pause overlay redraws so Freeze keeps the exact current frame.
            if hasattr(self, '_overlay_timer') and self._overlay_timer.isActive():
                self._overlay_timer.stop()

            # ── FREEZE METRIC LABELS ───────────────────────────────────────
            # Snapshot every visible metric label text right now so the top
            # bar shows the values at the exact moment of freeze and stops
            # updating while the display is paused.
            self._frozen_metric_texts = {}
            try:
                for key, lbl in getattr(self, 'metric_labels', {}).items():
                    self._frozen_metric_texts[key] = lbl.text()
            except Exception:
                pass
            # ──────────────────────────────────────────────────────────────

            # ── FREEZE METRICS FOR REPORT ──────────────────────────────────
            # Snapshot all numeric/text metrics at the moment of freeze so
            # that any report generated while frozen has the exact values that
            # were present on screen at that moment.
            self._frozen_report_metrics = {}
            try:
                is_demo = bool(hasattr(self, 'demo_toggle') and self.demo_toggle and self.demo_toggle.isChecked())
                if is_demo:
                    self._frozen_report_metrics = {
                        'HR':       60,
                        'RR':       1000,
                        'PR':       167,
                        'QRS':      86,
                        'QT':       357,
                        'QTc':      357,
                        'QTcF':     357,
                        'rv5':      float(getattr(self, '_last_rv5', 0.0) or 0.0),
                        'sv1':      float(getattr(self, '_last_sv1', 0.0) or 0.0),
                        'p_axis':   getattr(self, 'last_p_axis',   '--'),
                        'QRS_axis': getattr(self, 'last_qrs_axis', '--'),
                        't_axis':   getattr(self, 'last_t_axis',   '--'),
                    }
                else:
                    self._frozen_report_metrics = {
                        'HR':       int(getattr(self, 'last_heart_rate',    0) or 0),
                        'RR':       int(getattr(self, 'last_rr_interval',   0) or 0),
                        'PR':       int(getattr(self, 'pr_interval',        0) or 0),
                        'QRS':      int(getattr(self, 'last_qrs_duration',  0) or 0),
                        'QT':       int(getattr(self, 'last_qt_interval',   0) or 0),
                        'QTc':      int(getattr(self, 'last_qtc_interval',  0) or 0),
                        'QTcF':     int(getattr(self, 'last_qtcf_interval', 0) or 0),
                        'rv5':      float(getattr(self, '_last_rv5', 0.0) or 0.0),
                        'sv1':      float(getattr(self, '_last_sv1', 0.0) or 0.0),
                        'p_axis':   getattr(self, 'last_p_axis',   '--'),
                        'QRS_axis': getattr(self, 'last_qrs_axis', '--'),
                        't_axis':   getattr(self, 'last_t_axis',   '--'),
                    }
            except Exception as _fme:
                print(f"Error snapshotting frozen metrics: {_fme}")
            try:
                limb_conn = getattr(self, "_lead_connection_state", {}) or {}
                limb_active = limb_conn.get('I', True) or limb_conn.get('II', True)
                lead_off = bool(getattr(self, "_lead_off_latched", False))
                if isinstance(self._frozen_report_metrics, dict):
                    if lead_off or not limb_active:
                        self._frozen_report_metrics["HR"] = 0
                        self._frozen_report_metrics["RR"] = 0
                        self._frozen_report_metrics["PR"] = 0
                        self._frozen_report_metrics["QRS"] = 0
                        self._frozen_report_metrics["QT"] = 0
                        self._frozen_report_metrics["QTc"] = 0
                        self._frozen_report_metrics["QTcF"] = 0
                    if lead_off or limb_conn.get("V5", True) is False:
                        self._frozen_report_metrics["rv5"] = 0.0
                    if lead_off or limb_conn.get("V1", True) is False:
                        self._frozen_report_metrics["sv1"] = 0.0
            except Exception as _fm_lead_err:
                print(f"Error sanitizing frozen metrics: {_fm_lead_err}")

            self._style_freeze_button("Resume", enabled=True)
            if hasattr(self, "stop_btn") and self.stop_btn:
                self.stop_btn.setToolTip("Resume live ECG plotting from the current hardware buffer")

            # Always capture 10 s of waveform data for the frozen report
            self._frozen_report_snapshot = self._capture_report_snapshot(window_sec=10.0)
            self._replay_snapshot = self._capture_replay_snapshot()
            self._replay_render_snapshot = self._build_replay_render_snapshot(self._replay_snapshot)
            # Keep the exact live frame on screen so Freeze does not change
            # the window width or redraw the trace in a stretched state.
            self._update_generate_report_button_state()
        except Exception as e:
            print(f" Freeze render error: {e}")

    def _build_replay_render_snapshot(self, snapshot):
        """
        Precompute replay-ready display arrays once, instead of re-running the
        expensive filter chain on every timer tick.
        """
        rendered = []
        fs = self._get_sampling_rate_for_display()
        for i in range(12):
            try:
                arr = np.asarray(snapshot[i], dtype=float) if i < len(snapshot) else np.array([], dtype=float)
                if arr.size < 2:
                    rendered.append(arr.copy())
                    continue
                rendered.append(self._apply_snapshot_display_pipeline(arr, i, fs))
            except Exception:
                rendered.append(np.asarray(snapshot[i], dtype=float).copy() if i < len(snapshot) else np.array([], dtype=float))
        return rendered

    def _capture_replay_snapshot(self):
        snapshot = []
        for idx in range(12):
            try:
                if hasattr(self, "_replay_buffers") and idx < len(self._replay_buffers) and len(self._replay_buffers[idx]) > 0:
                    arr = np.asarray(list(self._replay_buffers[idx]), dtype=float)
                else:
                    arr = np.asarray(self.data[idx], dtype=float) if idx < len(self.data) else np.array([])
                # Keep the last 8000 samples per lead for replay.
                if arr.size > 8000:
                    arr = arr[-8000:]
            except Exception:
                arr = np.array([], dtype=float)
            snapshot.append(arr.copy())
        return snapshot

    def _resume_live_view(self):
        """Return the 12-lead screen to live plotting from the current hardware buffer."""
        try:
            if hasattr(self, "_replay_timer") and self._replay_timer.isActive():
                self._replay_timer.stop()
        except Exception:
            pass

        self._display_mode = "live"
        self._grid_frozen = False
        self._replay_cursor = 0

        # ── UNFREEZE METRIC LABELS ─────────────────────────────────────────
        # Clear the frozen snapshot so calculate_ecg_metrics() resumes
        # writing live values to the label widgets.
        self._frozen_metric_texts = {}
        self._frozen_report_metrics = None
        # Clear frozen waveform snapshot so Generate Report always uses
        # fresh live data after Resume, not the old frozen 180 BPM capture.
        self._frozen_report_snapshot = None
        # ──────────────────────────────────────────────────────────────────

        # If leads are currently disconnected when resuming, force metric labels to zero
        limb_conn = getattr(self, "_lead_connection_state", {}) or {}
        limb_active = limb_conn.get('I', True) or limb_conn.get('II', True)
        lead_off = bool(getattr(self, "_lead_off_latched", False))
        if lead_off or not limb_active:
            self.reset_metrics_to_zero(force=True)

        self._style_freeze_button("Freeze", enabled=True)
        if hasattr(self, "stop_btn") and self.stop_btn:
            self.stop_btn.setToolTip("Freeze the live ECG screen, then click Resume to continue live plotting")

        # Restart overlay redraws and force an immediate redraw from the current live buffer
        # so Resume feels instant.
        try:
            if hasattr(self, '_overlay_timer'):
                self._overlay_timer.start(100)
        except Exception:
            pass
        try:
            self.update_plots()
        except Exception:
            pass

        self._update_generate_report_button_state()

    def toggle_freeze_replay(self):
        """Freeze live ECG or resume live plotting from the current hardware buffer."""
        if getattr(self, "_display_mode", "live") == "frozen":
            self._resume_live_view()
            return

        self._freeze_current_view()

    def _update_replay_plots(self):
        # Replay mode has been replaced with Resume-to-live behavior.
        return

    def on_demo_toggle_changed(self, checked):
        self.update_demo_toggle_label()
        self.demo_manager.toggle_demo_mode(checked)
        
        # --- HANDLE HOLTER IN DEMO MODE ---
        if self.holter_mode_enabled:
            if checked:
                self._start_holter_session_internal()
            else:
                self._stop_holter_session_internal()
        # ----------------------------------

        if hasattr(self, "generate_report_btn"):
            if checked:
                # Demo mode ON - Start 10-second cooldown
                try:
                    self._start_generate_report_cooldown(seconds=10, reason="Demo Toggle")
                except Exception:
                    self.generate_report_btn.setEnabled(True)
                    try:
                        if hasattr(self, "green_button_style"):
                            self.generate_report_btn.setStyleSheet(self.green_button_style)
                    except Exception:
                        pass
            else:
                # Demo mode OFF: keep report enabled if live acquisition is still running.
                # Cancel any active countdown timers first.
                for timer in self.countdown_timers:
                    if hasattr(timer, "stop") and timer.isActive():
                        timer.stop()
                self.countdown_timers.clear()

                self.generate_report_btn.setText("Generate Report")
                if self._can_generate_report():
                    self.generate_report_btn.setEnabled(True)
                    try:
                        if hasattr(self, "green_button_style"):
                            self.generate_report_btn.setStyleSheet(self.green_button_style)
                    except Exception:
                        pass
                else:
                    self.generate_report_btn.setEnabled(False)
                    self.generate_report_btn.setStyleSheet("background: #cccccc; color: #666666; border-radius: 10px; padding: 8px 0; font-size: 10px; font-weight: bold;")

                timer = getattr(self, "timer", None)
                if timer is None or not timer.isActive():
                    self.update_recording_button_state()

    def apply_language(self, language=None):
        if language:
            self.current_language = language
        translator = self.tr
        if hasattr(self, 'menu_header_label'):
            self.menu_header_label.setText(translator("ECG Control Panel"))
        if hasattr(self, 'menu_buttons'):
            for btn, label in self.menu_buttons:
                btn.setText(translator(label))
        self.update_demo_toggle_label()
        if hasattr(self, 'capture_screen_btn') and self.capture_screen_btn:
            self.capture_screen_btn.setText(translator("Capture Screen"))
        if hasattr(self, 'recording_toggle') and self.recording_toggle:
            self.recording_toggle.setText(translator("Record Screen"))
        for attr, key in [
            ('start_btn', "Start"),
            ('stop_btn', "Freeze"),
            ('generate_report_btn', "Generate Report"),
            ('twelve_leads_btn', "12:1"),
            ('six_leads_btn', "6:2"),
            ('back_btn', "Return to Dashboard"),
        ]:
            btn = getattr(self, attr, None)
            if btn:
                btn.setText(translator(key))
        if hasattr(self, 'demo_toggle') and self.demo_toggle:
            self.update_demo_toggle_label()
        if hasattr(self, 'ecg_menu') and self.ecg_menu:
            update_lang = getattr(self.ecg_menu, "update_language", None)
            if callable(update_lang):
                update_lang(self.current_language)

    def calculate_12_leads_from_8_channels(self, channel_data):
        """
        Calculate 12-lead ECG from 8-channel hardware data
        Hardware sends: [L1, V4, V5, Lead 2, V3, V6, V1, V2] in that order
        """
        try:
            # Validate input data
            if not channel_data or not isinstance(channel_data, (list, tuple, np.ndarray)):
                print(" Invalid channel data format")
                return [0] * 12
            
            # Convert to list if numpy array
            if isinstance(channel_data, np.ndarray):
                channel_data = channel_data.tolist()
            
            # Ensure we have at least 8 channels
            if len(channel_data) < 8:
                # Pad with zeros if not enough channels
                channel_data = channel_data + [0] * (8 - len(channel_data))
                print(f" Padded channel data to 8 channels: {len(channel_data)}")
            
            # Validate all values are numeric
            for i, val in enumerate(channel_data[:8]):
                try:
                    float(val)
                except (ValueError, TypeError):
                    print(f" Invalid numeric value at channel {i}: {val}")
                    channel_data[i] = 0
            
            # Map hardware channels to standard positions with bounds checking
            L1 = float(channel_data[0]) if len(channel_data) > 0 else 0      # Lead I
            V4_hw = float(channel_data[1]) if len(channel_data) > 1 else 0   # V4 from hardware
            V5_hw = float(channel_data[2]) if len(channel_data) > 2 else 0   # V5 from hardware
            II = float(channel_data[3]) if len(channel_data) > 3 else 0      # Lead II
            V3_hw = float(channel_data[4]) if len(channel_data) > 4 else 0   # V3 from hardware
            V6_hw = float(channel_data[5]) if len(channel_data) > 5 else 0   # V6 from hardware
            V1 = float(channel_data[6]) if len(channel_data) > 6 else 0      # V1 from hardware
            V2 = float(channel_data[7]) if len(channel_data) > 7 else 0      # V2 from hardware

            # Calculate derived leads using standard ECG formulas with error handling
            I = L1  # Lead I is directly from hardware

            # Calculate Lead III from Lead I and Lead II
            try:
                III = II - I
            except Exception:
                III = 0

            # Calculate augmented leads using standard Einthoven/Goldberger relations:
            #   aVR = RA - (LA + LL)/2  = -(Lead I + Lead II) / 2
            #   aVL = LA - (RA + LL)/2 = (Lead I - Lead III) / 2
            #   aVF = LL - (RA + LA)/2 = (Lead II + Lead III) / 2
            try:
                aVR = -(I + II) / 2.0
            except Exception:
                aVR = 0.0

            try:
                aVL = (I - III) / 2
            except Exception:
                aVL = 0.0

            try:
                aVF = (II + III) / 2
            except Exception:
                aVF = 0.0

            # Use hardware V leads directly (already named V1, V2; others from *_hw)
            V3 = V3_hw
            V4 = V4_hw
            V5 = V5_hw
            V6 = V6_hw
        
            # Return 12-lead ECG data in standard order
            result = [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
            
            # Validate result
            for i, val in enumerate(result):
                if not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                    print(f" Invalid result value at lead {i}: {val}")
                    result[i] = 0
            
            return result
            
        except Exception as e:
            print(f" Critical error in calculate_12_leads_from_8_channels: {e}")
            # Return safe default values
            return [0] * 12

    def _extract_low_frequency_baseline(self, signal, sampling_rate=500.0):
        """Extract low-frequency baseline - wrapper for modular function"""
        return extract_low_frequency_baseline(signal, sampling_rate)
        """
        Extract very-low-frequency baseline estimate (< 0.3 Hz) for display anchoring.
        
        Uses 2-second moving average SIGNAL (not mean) to remove:
        - Respiration (0.1-0.35 Hz) → filtered out
        - ST/T waves → filtered out
        - QRS complexes → filtered out
        
        Returns only very-low-frequency drift (< 0.1 Hz).
        
        Args:
            signal: ECG signal window
            sampling_rate: Sampling rate in Hz
        
        Returns:
            Low-frequency baseline estimate (single value)
        """
        if len(signal) < 10:
            return np.nanmean(signal) if len(signal) > 0 else 0.0
        
        try:
            # Method: 2-second moving average SIGNAL (not mean)
            window_samples = int(2.0 * sampling_rate)  # 2 seconds
            window_samples = min(window_samples, len(signal))
            
            if window_samples >= 10 and len(signal) >= window_samples:
                # Extract actual low-frequency baseline signal using convolution
                # This is a proper moving average, not just a statistic
                kernel = np.ones(window_samples) / window_samples
                baseline_signal = np.convolve(signal, kernel, mode="valid")
                # Use the last value of the moving-average signal
                baseline_estimate = baseline_signal[-1] if len(baseline_signal) > 0 else np.nanmean(signal)
            else:
                # Fallback: use mean if window too small
                baseline_estimate = np.nanmean(signal)
            
            return baseline_estimate
        
        except Exception:
            # Fallback: simple mean if moving average fails
            return np.nanmean(signal) if len(signal) > 0 else 0.0

    @staticmethod
    def _append_block(buf, values):
        """Append `values` to the end of a fixed-length chronological buffer.

        Replaces `buf = np.roll(buf, -1); buf[-1] = v` repeated once per sample.
        np.roll allocates and copies the entire array every call; at 500 Hz x 12
        leads that was ~240 MB/s of memcpy and ~6000 ndarray allocations per
        second, whose collection produced a measured 10-13 ms hitch roughly every
        12 seconds. One memmove per lead per tick has neither cost.

        The buffer keeps its identity, length and ordering — oldest at index 0,
        newest at index -1 — so every existing reader is unaffected.
        """
        n = len(values)
        if n <= 0:
            return
        size = buf.shape[0]
        if n >= size:
            # Batch longer than the whole window: keep only its most recent tail.
            buf[:] = np.asarray(values[-size:], dtype=buf.dtype)
            return
        buf[:-n] = buf[n:]                                  # slide down, in place
        buf[-n:] = np.asarray(values, dtype=buf.dtype)      # newest at the end

    def _flush_staged_samples(self, staged):
        """Commit one tick's staged samples into the rolling lead buffers."""
        if not staged:
            return
        for i, vals in enumerate(staged):
            if vals and i < len(self.data):
                try:
                    self._append_block(self.data[i], vals)
                except Exception as e:
                    print(f" Error flushing lead buffer {i}: {e}")
                vals.clear()

    # A measurement may stand in for a failed window for this long, then it is
    # discarded. Matches display_updates.HOLD_MAX_SECONDS so the page and the
    # labels cannot disagree about whether a value is still valid.
    PAGE_HOLD_MAX_SECONDS = 4.0

    def _hold_or_zero(self, key, fresh_value):
        """Return `fresh_value` if it is usable, else a recent hold, else 0.

        Without the expiry these holds re-published the last good measurement
        indefinitely. A Fluke simulator moved from 72 bpm to 3 bpm kept
        reporting PR 158 / QRS 81 / QT 338 — the analysis window (20 s) cannot
        contain the three R-peaks needed at that rate, so every subsequent
        window failed and every failure was answered with the pre-change value.
        """
        import time as _time
        try:
            value = float(fresh_value or 0)
        except (TypeError, ValueError):
            value = 0.0

        if not hasattr(self, '_metric_hold_at'):
            self._metric_hold_at = {}

        if value > 0:
            self._metric_hold_at[key] = _time.monotonic()
            return fresh_value

        stamp = self._metric_hold_at.get(key)
        if stamp is None:
            return 0
        if (_time.monotonic() - stamp) <= self.PAGE_HOLD_MAX_SECONDS:
            return getattr(self, key, 0) or 0

        # Hold expired — stop echoing a measurement the signal no longer supports.
        self._metric_hold_at.pop(key, None)
        return 0

    def _build_global_qrs_lead_map(self, reference_length=None):
        """Collect the connected, non-flat leads for the global QRS measurement.

        The multi-lead boundary rule only holds if every signal shares Lead II's
        sample clock, so leads whose buffer length differs are dropped rather
        than resampled. A lead the connection tracker has marked off is dropped
        too: its buffer is held flat at the ADC centre and would otherwise be
        delineated as a spurious, very narrow complex.

        Returns {lead name -> signal}; empty when fewer than two leads qualify,
        which leaves the caller on the single-lead path.
        """
        lead_map = {}
        try:
            data = getattr(self, 'data', None)
            leads = getattr(self, 'leads', None)
            if not data or not leads:
                return {}

            conn_state = getattr(self, '_lead_connection_state', {}) or {}
            off_state = getattr(self, '_lead_off_state', {}) or {}

            for idx, lead_name in enumerate(leads):
                if idx >= len(data):
                    break
                sig = data[idx]
                if sig is None or len(sig) < 1000:
                    continue
                if reference_length is not None and len(sig) != reference_length:
                    continue
                if not conn_state.get(lead_name, True):
                    continue
                if off_state.get(lead_name, False):
                    continue
                arr = np.asarray(sig, dtype=float)
                # Same flat-line gate the rest of the page uses (ADC counts).
                if float(np.std(arr)) < 5.0:
                    continue
                lead_map[lead_name] = arr

            if len(lead_map) < 2:
                return {}
        except Exception as e:
            print(f" ⚠️ _build_global_qrs_lead_map failed: {e}")
            return {}
        return lead_map

    def calculate_ecg_metrics(self):
        """Calculate ECG metrics using median beat (GE/Philips standard).
        
        ⚠️ CLINICAL ANALYSIS: Uses RAW data, median beat, TP baseline
        This function MUST use raw clinical data, NOT display-processed data.
        """
        # Initialize all metric attributes to prevent AttributeError
        if not hasattr(self, 'pr_interval'):
            self.pr_interval = 0
        if not hasattr(self, 'last_qrs_duration'):
            self.last_qrs_duration = 0
        if not hasattr(self, 'last_qt_interval'):
            self.last_qt_interval = 0
        if not hasattr(self, 'last_qtc_interval'):
            self.last_qtc_interval = 0
        if not hasattr(self, 'last_p_duration'):
            self.last_p_duration = 0
        if not hasattr(self, 'last_heart_rate'):
            self.last_heart_rate = 0

        # Lead disconnection guard: Only zero out if BOTH primary limb leads (I and II) are disconnected!
        limb_conn = getattr(self, "_lead_connection_state", {})
        limb_active = limb_conn.get('I', True) or limb_conn.get('II', True)
        if getattr(self, "_lead_off_latched", False) or not limb_active:
            self.reset_metrics_to_zero()
            return

        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            print(" Demo mode active - skipping live ECG metrics calculation")
            return

        # FREEZE guard: Do not recalculate metrics while frozen
        if getattr(self, '_grid_frozen', False):
            return

        # BPM FREEZE: Skip recalculation while report is being generated to keep BPM stable
        if getattr(self, '_report_generating', False):
            return

        if len(self.data) < 2:  # Need at least Lead II for analysis
            return
        
        #  CLINICAL: Use RAW Lead II data (index 1) for clinical analysis
        # This is the raw buffer - NOT display-processed data
        lead_ii_data = self.data[1]
        
        # ── Flat-line / Asystole guard (ADC scale) ────────────────────────────────
        # When the hardware sends raw ADC counts the mean is large (~2060) but a
        # flat/disconnected lead has std < 50 counts. Detect this condition
        # (or completely 0 signal) and return all-zero metrics instead of running the full pipeline.
        _raw_std_ii = float(np.std(lead_ii_data)) if len(lead_ii_data) > 0 else 0.0
        
        # Fallback to Lead I if Lead II is flat (std < 5.0)
        if _raw_std_ii < 5.0 and len(self.data) > 0 and len(self.data[0]) >= 100:
            lead_i_data = self.data[0]
            if float(np.std(lead_i_data)) >= 5.0:
                lead_ii_data = lead_i_data
                _raw_std_ii = float(np.std(lead_ii_data))

        # Check limb lead status: Do NOT treat as flatline if primary limb leads (Lead I or II) are active!
        limb_conn = getattr(self, "_lead_connection_state", {})
        limb_active = limb_conn.get('I', True) or limb_conn.get('II', True)

        # A flat trace means there is nothing to measure — whether the leads are
        # off OR the heart has stopped. The `and not limb_active` condition that
        # used to be here restricted this to disconnected leads only, so a
        # genuine asystole with the electrodes still attached fell through to the
        # full pipeline, which found no R-peaks and then re-published the last
        # good HR/PR/QRS/QT from before the arrest. A flatline was drawn next to
        # a live-looking heart rate.
        #
        # `_raw_std_ii` is in ADC counts and this is checked AFTER the Lead I
        # fallback above, so < 5.0 means both limb leads are flat. No real ECG
        # carrying a QRS has a standard deviation of five ADC counts.
        _signal_is_flat = (
            len(lead_ii_data) < 100
            or np.all(lead_ii_data == 0)
            or _raw_std_ii < 5.0
        )
        # Kept for the log line below: distinguishes "electrodes removed" from
        # "electrodes on, no cardiac activity". Both zero the metrics; only the
        # wording differs.
        _leads_report_connected = bool(limb_active)
        _is_flat_line_ii = _signal_is_flat
        # LL(F) disconnection: use the fast per-packet flag rather than
        # _lead_connection_state['II'] which only flips after 25 consecutive
        # missing packets. Without this, BPM flickers between 0 and a real
        # value during that debounce window when LL is removed.
        if getattr(self, '_ll_disconnected', False):
            _is_flat_line_ii = True
        
        if _is_flat_line_ii:
            # Nothing is measurable on a flat trace. Every value is zeroed —
            # including the ones the old block missed (QTcF, ST, axes, the
            # amplitude figures and the pending/deadband state), because any
            # value left behind here is a pre-arrest number displayed as if it
            # were current.
            _asystole = _leads_report_connected  # electrodes on, no activity
            if not getattr(self, '_flatline_logged', False):
                print(
                    " ASYSTOLE — electrodes attached, no cardiac activity: zeroing all metrics"
                    if _asystole else
                    " FLATLINE — limb leads disconnected: zeroing all metrics"
                )
                self._flatline_logged = True
            self._asystole_active = bool(_asystole)

            self.last_heart_rate   = 0
            self.last_rr_interval  = 0
            self.pr_interval       = 0
            self.last_qrs_duration = 0
            self.last_qt_interval  = 0
            self.last_qtc_interval = 0
            self.last_qtcf_interval = 0
            self.last_p_duration   = 0
            self.last_st_interval  = 0
            self._last_displayed_hr  = 0
            self._last_displayed_qrs = 0
            self._last_displayed_qt  = 0
            self._last_displayed_qtc = 0
            # Pending/deadband state must go too, or a held value can be
            # promoted onto the display one tick after signal returns.
            self._pending_qrs_value = None
            self._pending_qt_value = None
            for _attr in ('last_qrs_axis', 'last_p_axis', 'last_t_axis',
                          'last_rv5', 'last_sv1', 'last_rv5_sv1'):
                if hasattr(self, _attr):
                    setattr(self, _attr, 0)
            # Clear smoothing buffers so they don’t re-inject old values
            for _buf in ('_pr_smooth_buffer_tl', '_qrs_smooth_buffer', '_qt_smooth_buffer'):
                if hasattr(self, _buf):
                    getattr(self, _buf).clear()
            # Drop the cached global-QRS result too, otherwise a width
            # measured before the leads dropped can reappear for a few seconds
            # once signal returns.
            try:
                from .ecg_calculations import reset_global_qrs_cache
                reset_global_qrs_cache(getattr(self, '_instance_id', 'twelve_lead'))
            except Exception:
                pass
            try:
                # Clears the held values AND their freshness timestamps, so a
                # pre-arrest number cannot be echoed back for the hold window.
                from .ui import display_updates as _du
                _du.reset_metric_holds()
            except Exception:
                pass
            # Push zeros to the display
            self.update_ecg_metrics_display(
                0, 0, 0, 0, 0, 0, 0,
                force_immediate=True,
                rr_interval=0,
            )
            try:
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText("  0")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText("  0")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText("  0")
                if 'st_interval' in self.metric_labels:
                    self.metric_labels['st_interval'].setText("0")
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("0/0")
            except Exception:
                pass
            return
        
        # Real signal is present again.
        self._asystole_active = False
        self._flatline_logged = False

        # Get sampling rate
        # Hardware stream is fixed 500 Hz. Keep calculations locked to configured rate
        # to avoid metric/wave instability when UI focus changes and measured UI rate dips.
        fs = 500.0
        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            if hasattr(self, 'demo_fs') and self.demo_fs:
                fs = float(self.demo_fs)
            elif hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                fs = float(self.sampler.sampling_rate)
            
        # Unified ECG metrics (HR, RR, PR, QRS, QT, QTc) — single source of truth
        # Every connected lead is handed over so QRS width can be measured from
        # the earliest onset to the latest offset across the lead set instead of
        # from Lead II alone (which reads 10-20 ms short).
        global_qrs_leads = self._build_global_qrs_lead_map(
            reference_length=len(lead_ii_data)
        )
        try:
            user_metrics = calculate_all_ecg_metrics(
                lead_ii_data, fs,
                instance_id=getattr(self, '_instance_id', 'twelve_lead'),
                all_lead_data=global_qrs_leads or None,
            )
        except Exception as e:
            print(f"Error in calculate_all_ecg_metrics: {e}")
            user_metrics = {k: None for k in ["heart_rate", "rr_interval", "pr_interval", "qrs_duration", "qt_interval", "qtc_interval"]}

        # Remember how the QRS was measured so the report and the metric tooltip
        # can say so rather than implying a single-lead number is a global one.
        self.last_qrs_method = user_metrics.get("qrs_method", "single-lead")
        self.last_qrs_leads_used = int(user_metrics.get("qrs_leads_used", 1) or 1)
        
        # DUAL-PATH ECG ARCHITECTURE (Clinical Standard):
        # 1. DISPLAY CHANNEL (0.5-40 Hz): Used for R-peak detection only
        #    - Preserves clean waveform for peak detection
        #    - R-peaks are accurate with this filter
        # 2. MEASUREMENT CHANNEL (0.05-150 Hz): Used for all clinical calculations
        #    - Preserves Q/S waves (not attenuated)
        #    - Preserves T-wave tail (not truncated)
        #    - Median beat is automatically built from measurement channel in build_median_beat()
        
        # Detect R-peaks.
        # Primary (requested): Pan–Tompkins with search-back to true R indices.
        # Fallback: existing multi-strategy find_peaks logic (for edge cases).
        from scipy.signal import find_peaks
        from .signal_paths import display_filter
        filtered_ii = display_filter(lead_ii_data, fs)  # Display channel for R-peak detection

        r_peaks = np.array([], dtype=int)
        try:
            from .pan_tompkins import pan_tompkins
            r_peaks = pan_tompkins(filtered_ii, fs=fs)
        except Exception as e:
            print(f" ⚠️ Pan-Tompkins failed in calculate_ecg_metrics: {e}")
            r_peaks = np.array([], dtype=int)

        signal_mean = np.mean(filtered_ii)
        signal_std = np.std(filtered_ii)

        # Use adaptive peak detection for 10-300 BPM (legacy fallback)
        # Try multiple strategies and select best based on consistency
        detection_results = []
        height_threshold = signal_mean + 0.5 * signal_std
        prominence_threshold = signal_std * 0.4
        
        # Strategy 1: Conservative (10-120 BPM)
        # Distance set to minimum RR for highest BPM in range (120 BPM = 500ms)
        # RR interval filtering (200-6000ms) will handle the full 10-300 BPM range
        peaks_conservative, _ = find_peaks(
            filtered_ii,
            height=height_threshold,
            distance=int(0.4 * fs),  # 400ms - prevents false peaks, allows 10-300 BPM via RR filtering
            prominence=prominence_threshold
        )
        if len(peaks_conservative) >= 2:
            rr_cons = np.diff(peaks_conservative) * (1000 / fs)
            valid_cons = rr_cons[(rr_cons >= 200) & (rr_cons <= 6000)]
            if len(valid_cons) > 0:
                bpm_cons = 60000 / np.median(valid_cons)
                std_cons = np.std(valid_cons)
                detection_results.append(('conservative', peaks_conservative, bpm_cons, std_cons))
        
        # Strategy 2: Normal (best for 100-180 BPM)
        peaks_normal, _ = find_peaks(
            filtered_ii,
            height=height_threshold,
            distance=int(0.3 * fs),  # 150 samples = 240ms - normal for 100-180 BPM
            prominence=prominence_threshold
        )
        if len(peaks_normal) >= 2:
            rr_norm = np.diff(peaks_normal) * (1000 / fs)
            valid_norm = rr_norm[(rr_norm >= 200) & (rr_norm <= 6000)]
            if len(valid_norm) > 0:
                bpm_norm = 60000 / np.median(valid_norm)
                std_norm = np.std(valid_norm)
                detection_results.append(('normal', peaks_normal, bpm_norm, std_norm))
        
        # Strategy 3: Tight (best for 160-220 BPM)
        peaks_tight, _ = find_peaks(
            filtered_ii,
            height=height_threshold,
            distance=int(0.15 * fs),  # 75 samples = 150ms - FIX: was 0.2*fs which is exactly RR at 300 BPM
            prominence=prominence_threshold
        )
        if len(peaks_tight) >= 2:
            rr_tight = np.diff(peaks_tight) * (1000 / fs)
            valid_tight = rr_tight[(rr_tight >= 200) & (rr_tight <= 6000)]
            if len(valid_tight) > 0:
                bpm_tight = 60000 / np.median(valid_tight)
                std_tight = np.std(valid_tight)
                detection_results.append(('tight', peaks_tight, bpm_tight, std_tight))

        # Strategy 4: Ultra-tight (best for 200-300 BPM) - NEW for high heart rates
        peaks_ultra_tight, _ = find_peaks(
            filtered_ii,
            height=height_threshold,
            distance=int(0.12 * fs),  # 60 samples = 120ms - supports up to 500 BPM
            prominence=prominence_threshold,
        )
        if len(peaks_ultra_tight) >= 2:
            rr = np.diff(peaks_ultra_tight) * (1000.0 / fs)
            valid_rr = rr[(rr >= 200) & (rr <= 6000)]
            if len(valid_rr) > 0:
                bpm = 60000.0 / np.median(valid_rr)
                std = np.std(valid_rr)
                detection_results.append(('ultra_tight', peaks_ultra_tight, bpm, std))
        
        # Select best strategy: FORCE ultra-tight if ANY result shows BPM > 200
        # This prevents sub-harmonic aliasing (detecting every other peak)
        # NOTE: Only run this selection if Pan–Tompkins did not already yield peaks.
        if len(r_peaks) < 2 and detection_results:
            # Candidates that pass the stability gate
            stable_candidates = []
            for r in detection_results:
                method, peaks_result, bpm, std = r
                # Adaptive thresholds based on BPM
                if bpm > 180:
                    # High BPM: allow higher std
                    max_std_abs = 25
                    max_std_pct = 0.20
                else:
                    # Normal BPM: strictly stable
                    max_std_abs = 15
                    max_std_pct = 0.15
                
                if std <= max_std_abs and std <= bpm * max_std_pct:
                    stable_candidates.append(r)

            if stable_candidates:
                # Among stable candidates prefer highest BPM (avoids sub-harmonic aliasing),
                # but only if a faster candidate's BPM is not >10% higher than the next one
                # (which would indicate noise rather than a true faster rate).
                # CRITICAL FIX: Always prefer highest BPM to avoid sub-harmonic detection
                # Strategy detecting every other beat looks "stable" but gives half the rate
                stable_candidates.sort(key=lambda x: x[2], reverse=True)
                best_method, r_peaks, best_bpm, best_std = stable_candidates[0]
                
                # Log RR intervals for debugging
                rr_median_ms = 60000.0 / best_bpm if best_bpm > 0 else 0
                print(f" 🎯 SELECTED (12-lead): '{best_method}' strategy with BPM={best_bpm:.1f}, std={best_std:.1f}ms, RR_median={rr_median_ms:.1f}ms")
            else:
                # Fallback: take the most stable result even if not ideal
                detection_results.sort(key=lambda x: x[3])
                best_method, r_peaks, best_bpm, best_std = detection_results[0]
                rr_median_ms = 60000.0 / best_bpm if best_bpm > 0 else 0
                print(f" ⚠️ FALLBACK (12-lead): No stable candidates, using '{best_method}' with BPM={best_bpm:.1f}, std={best_std:.1f}ms, RR_median={rr_median_ms:.1f}ms")
        else:
            # Fallback to conservative strategy for low BPM (10-120 BPM)
            if len(r_peaks) < 2:
                r_peaks, _ = find_peaks(
                    filtered_ii,
                    height=height_threshold,
                    distance=int(0.4 * fs),  # 400ms - prevents false peaks, allows 10-300 BPM via RR filtering
                    prominence=prominence_threshold
                )
        
        # Calculate BPM first (works with ≥2 beats) - needed for low BPM detection
        # This allows BPM calculation even when we don't have enough beats for median beat
        # CRITICAL: Use most recent R-peaks for accurate BPM calculation (matches Fluke device)
        # Initialize RR interval with last known good value (or 1000 ms ≈ 60 BPM) so that
        # downstream calculations always have a defined rr_ms, even if no valid intervals
        # are found in the current window.
        rr_ms = getattr(self, 'last_rr_interval', 1000.0)

        # ── Rate below the measurable floor ──────────────────────────────────
        # The analysis window is HISTORY_LENGTH samples (20 s at 500 Hz) and at
        # least two R-peaks are needed for one RR interval, so the slowest rate
        # this window can express is about 3 bpm — and the metrics pipeline
        # requires three peaks, i.e. roughly 9 bpm.
        #
        # Below that there IS cardiac activity (visible QRS complexes) but it
        # cannot be measured here. That is a third state, distinct from both a
        # flat trace and a disconnected lead, and the report must not describe
        # it as "no data / please connect device" — the device is working and
        # the patient has a rhythm, just a profoundly slow one.
        self._rate_below_measurable = bool(0 < len(r_peaks) < 3)
        if self._rate_below_measurable and not getattr(self, '_low_rate_logged', False):
            _secs = len(lead_ii_data) / fs if fs else 0.0
            print(
                f" RATE BELOW MEASURABLE RANGE: only {len(r_peaks)} R-peak(s) in a "
                f"{_secs:.0f}s window (>=3 needed, i.e. ~9 bpm). Metrics report 0."
            )
            self._low_rate_logged = True
        elif not self._rate_below_measurable:
            self._low_rate_logged = False

        if len(r_peaks) >= 2:
            # Calculate RR intervals from consecutive R-peaks
            rr_intervals_ms = np.diff(r_peaks) / fs * 1000.0
            # Filter physiologically reasonable intervals (200-8000 ms = 7.5-300 BPM)
            valid_rr = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 8000)]
            if len(valid_rr) > 0:
                prior_rr_ms = getattr(self, 'last_rr_interval', 0)
                has_stable_history = bool(prior_rr_ms and prior_rr_ms > 0)
                provisional_only = len(valid_rr) < 2

                # Do not let a single early RR interval rewrite the live BPM.
                # During startup or after a noisy false R-peak, one short RR can briefly
                # look like 220-280 BPM before the next beats settle. Keep the previous
                # stable RR if we only have one interval so the UI does not spike.
                if provisional_only and has_stable_history:
                    rr_ms = float(prior_rr_ms)
                    estimated_bpm = 60000.0 / rr_ms if rr_ms > 0 else 60
                else:
                    # Use median for robustness, but prioritize recent intervals for accuracy
                    # Take last 5 RR intervals if available for more accurate current BPM
                    recent_rr = valid_rr[-5:] if len(valid_rr) > 5 else valid_rr
                    rr_ms = np.median(recent_rr)
                    estimated_bpm = 60000.0 / rr_ms if rr_ms > 0 else 60
                
                # Debug: Show RR calculation (print once)
                # if not hasattr(self, '_rr_debug_printed'):
                #     self._rr_debug_printed = False
                # if not self._rr_debug_printed and len(valid_rr) > 0:
                if not hasattr(self, '_rr_debug_tick'):
                    self._rr_debug_tick = 0
                    self._rr_debug_last_ms = rr_ms
                self._rr_debug_tick += 1
                rr_change = abs(rr_ms - getattr(self, '_rr_debug_last_ms', rr_ms))
                if self._rr_debug_tick <= 3 or self._rr_debug_tick % 150 == 0 or rr_change >= 80:
                    print(f" 🔍 RR Calculation Debug: {len(r_peaks)} R-peaks, {len(valid_rr)} valid RR intervals")
                    print(f"    RR intervals (ms): {valid_rr[:5].tolist() if len(valid_rr) >= 5 else valid_rr.tolist()}")
                    print(f"    Median RR: {rr_ms:.1f} ms → HR: {estimated_bpm:.1f} BPM")
                self._rr_debug_last_ms = rr_ms
                #     self._rr_debug_printed = True
                
                # Validate BPM is reasonable (10-300 BPM)
                if estimated_bpm < 10 or estimated_bpm > 300:
                    # Fallback to median of all valid intervals if recent calculation is invalid
                    rr_ms = np.median(valid_rr)
                    estimated_bpm = 60000.0 / rr_ms if rr_ms > 0 else 60
            else:
                # No valid RR intervals in the physiological range; reset to 0 BPM
                estimated_bpm = 0
        else:
            estimated_bpm = 0
        
        # Adaptive minimum beat requirement based on BPM and available data window
        # At low BPM (< 40), we need longer windows, so reduce minimum beats
        # At 10 BPM: need 6s per beat, so 2 beats = 12s (not possible in 4s buffer)
        # At 30 BPM: need 2s per beat, so 2 beats = 4s (just possible)
        # Strategy: Use minimum 2 beats for BPM, but require more for median beat if possible
        min_beats_for_bpm = 2  # Always allow BPM calculation with 2 beats
        min_beats_for_median = 8  # Preferred for median beat
        
        # If we have low BPM (< 40), reduce median beat requirement significantly
        if estimated_bpm < 40:
            # At very low BPM, allow building "median" beat from even 1 beat (3 peaks needed if [1:-1])
            # Wait, build_median_beat skips first/last, so 3 peaks -> 1 beat candidate.
            # Set min_beats_for_median to 1 for < 40 BPM to allow 3 peaks to work.
            min_beats_for_median = 1  # Relaxed for low BPM
        elif estimated_bpm < 60:
            min_beats_for_median = 3
        
        # Fallback to V2 if Lead II has insufficient beats (GE/Philips standard)
        if len(r_peaks) < min_beats_for_bpm and len(self.data) > 3:
            lead_v2_data = self.data[3]  # V2 is typically index 3
            if len(lead_v2_data) > 100 and np.std(lead_v2_data) > 0.1:
                # Use display filter for R-peak detection on V2
                from .signal_paths import display_filter
                filtered_v2 = display_filter(lead_v2_data, fs)
                signal_mean_v2 = np.mean(filtered_v2)
                signal_std_v2 = np.std(filtered_v2)
                
                # Use same adaptive detection for V2 (prioritize conservative for low BPM)
                height_v2 = signal_mean_v2 + 0.5 * signal_std_v2
                prominence_v2 = signal_std_v2 * 0.4
                
                # Try tight strategy for V2 (supports 10-300 BPM via RR filtering)
                r_peaks_v2, _ = find_peaks(
                    filtered_v2,
                    height=height_v2,
                    distance=int(0.2 * fs),  # 100 samples = 200ms - minimum RR at 300 BPM
                    prominence=prominence_v2
                )
                
                if len(r_peaks_v2) >= min_beats_for_bpm:
                    r_peaks = r_peaks_v2
                    lead_ii_data = lead_v2_data  # Use V2 for beat alignment
                    # Recalculate estimated BPM from V2
                    if len(r_peaks) >= 2:
                        rr_intervals_ms = np.diff(r_peaks) / fs * 1000.0
                        valid_rr = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 8000)]
                        if len(valid_rr) > 0:
                            rr_ms = np.median(valid_rr)
                            estimated_bpm = 60000.0 / rr_ms if rr_ms > 0 else 60
                            # Update min_beats_for_median based on V2 BPM
                            if estimated_bpm < 40:
                                min_beats_for_median = max(3, min(8, len(r_peaks)))
        
        # Require minimum beats for BPM calculation (≥2 beats)
        if len(r_peaks) < min_beats_for_bpm:
            return
        
        # Build median beat from MEASUREMENT CHANNEL (0.05-150 Hz)
        # build_median_beat() automatically applies measurement filter internally
        # IMPORTANT: At low BPM, we proceed with HR update even if median beat fails,
        # but we need a median beat for PR/QRS/QT clinical metrics.
        time_axis, median_beat_ii = build_median_beat(lead_ii_data, r_peaks, fs, min_beats=min_beats_for_median)
        
        # If median beat failed, we can still update HR if we have ≥2 peaks
        if median_beat_ii is None:
            # Update heart rate even without clinical metrics
            local_bpm = int(round(estimated_bpm))
            if user_metrics["heart_rate"] is not None:
                self.last_heart_rate = user_metrics["heart_rate"]
            else:
                self.last_heart_rate = local_bpm
            
            # No median beat means no clinical intervals from this window. The
            # previous values may stand in briefly — one noisy window should not
            # blank the display — but they expire, otherwise a sustained
            # inability to measure is shown as a live measurement. This is the
            # path a Fluke at 3 bpm takes on every tick: PR 158 / QRS 81 /
            # QT 338 were re-published indefinitely after the rate changed.
            self.pr_interval        = self._hold_or_zero('pr_interval', 0)
            self.last_qrs_duration  = self._hold_or_zero('last_qrs_duration', 0)
            self.last_qt_interval   = self._hold_or_zero('last_qt_interval', 0)
            self.last_qtc_interval  = self._hold_or_zero('last_qtc_interval', 0)
            self.last_p_duration    = self._hold_or_zero('last_p_duration', 0)
            self._last_displayed_qt  = self.last_qt_interval
            self._last_displayed_qtc = self.last_qtc_interval

            self.update_ecg_metrics_display(
                self.last_heart_rate,
                self.pr_interval,
                self.last_qrs_duration,
                self.last_p_duration,
                self._last_displayed_qt,
                self._last_displayed_qtc,
                0,
                force_immediate=True,
                rr_interval=getattr(self, 'last_rr_interval', 0),  # FIX-D1
            )
            return
        
        
        # Get TP baseline from MEASUREMENT CHANNEL (0.05-150 Hz) for consistency
        # get_tp_baseline() automatically applies measurement filter when use_measurement_channel=True (default)
        r_idx = len(median_beat_ii) // 2  # R-peak at center
        r_mid = r_peaks[len(r_peaks) // 2]
        # Use previous R-peak for proper TP segment detection
        prev_r_idx = r_peaks[len(r_peaks) // 2 - 1] if len(r_peaks) > 1 else None
        tp_baseline_ii = get_tp_baseline(lead_ii_data, r_mid, fs, prev_r_peak_idx=prev_r_idx, use_measurement_channel=True)
        
        
        
        # Calculate Heart Rate: HR = 60000 / RR_ms (GE/Philips standard)
        # Formula: BPM = 60000 milliseconds / RR_interval_milliseconds
        if rr_ms > 0:
            heart_rate_raw = int(round(60000.0 / rr_ms))
        else:
            heart_rate_raw = 60  # Fallback if RR is invalid
            
        # OVERRIDE: Use user's comprehensive metrics if available, BUT check for sub-harmonic issues
        local_bpm = int(round(estimated_bpm))
        if user_metrics["heart_rate"] is not None:
            user_bpm = user_metrics["heart_rate"]
            
            # INTELLIGENT OVERRIDE CHECK:
            # If local detection finds high BPM (>180) and user_metrics is approximately half (0.4-0.6x),
            # it's likely a sub-harmonic issue in the user_metrics (double spacing of R-peaks).
            # in this specific case, TRUST THE LOCAL ALGORITHM which has "Ultra-tight" strategy.
            if local_bpm > 180 and (0.4 * local_bpm < user_bpm < 0.6 * local_bpm):
                # Sub-harmonic detected! specific case for ~260 vs ~130
                 print(f" ⚠️ Ignoring user_metrics HR ({user_bpm}) - likely sub-harmonic of local High BPM ({local_bpm})")
                 heart_rate_raw = local_bpm
                 # Do NOT overwrite rr_ms if we are rejecting the HR
            else:
                 # Keep HR and RR mathematically locked: if RR is available, derive HR from RR.
                 if user_metrics["rr_interval"] is not None and user_metrics["rr_interval"] > 0:
                     rr_ms = float(user_metrics["rr_interval"])
                     heart_rate_raw = int(round(60000.0 / rr_ms))
                 else:
                     heart_rate_raw = user_bpm
        else:
             heart_rate_raw = local_bpm
        
        # REAL MODE: Always use real values from hardware calculations
        # Reference table removed - using only real calculated values
        use_reference_values = False
        
        
        # NOTE: Using rr_ms directly from first calculation (lines 1644-1666)
        # DO NOT recalculate or "fix" rr_ms here - it creates circular dependency!
        
        # Store RR interval for access by other functions
        self.last_rr_interval = int(round(rr_ms))
        
        # FIX-TL5: RR/HR consistency check — use ≤15ms tolerance.
        # heart_rate_raw = round(60000/rr_ms) → by definition 60000/HR ≈ rr_ms ± rounding.
        # At 60 BPM: 60000/60=1000ms but actual rr_ms=998.8ms → diff=16ms → spurious warning.
        # The hold-and-jump display logic may show HR=61 while rr_ms=998ms → diff=15ms.
        # Accept ≤20ms difference (< 2 BPM equivalent error).
        expected_rr = 60000.0 / heart_rate_raw if heart_rate_raw > 0 else 600.0
        verification_ok = abs(rr_ms - expected_rr) <= 30.0
        
        # Debug output: Show RR and HR calculation (print first few times and then occasionally)
        if not hasattr(self, '_rr_hr_debug_count'):
            self._rr_hr_debug_count = 0
        self._rr_hr_debug_count += 1
        
        # OPTIMIZED: Reduced print frequency for better performance
        if self._rr_hr_debug_count <= 3 or self._rr_hr_debug_count % 200 == 0:  # Reduced from 10/100 to 3/200
            if verification_ok:
                print(f" ✓ RR Calculation: RR={rr_ms:.0f} ms → HR={heart_rate_raw} BPM (verified: {rr_ms * heart_rate_raw:.0f} ≈ 60000)")
            else:
                print(f" ⚠️ RR/HR inconsistency: RR={rr_ms:.1f} ms, HR={heart_rate_raw} BPM, expected RR={expected_rr:.1f} ms (diff={abs(rr_ms-expected_rr):.1f} ms)")
        
        # Always derive display HR + interval math from the same RR source selected above.
        # This keeps BPM, RR, QT and QTc internally consistent and aligned with
        # reference software that uses median RR from detected beats.
        self.last_heart_rate = heart_rate_raw
        if heart_rate_raw and heart_rate_raw > 0:
            if not hasattr(self, '_metric_hold_at'):
                self._metric_hold_at = {}
            import time as _t
            self._metric_hold_at['last_heart_rate'] = _t.monotonic()

        # FIX-HR-STAB: heart_rate_raw already comes from calculate_hr_rr()
        # (via user_metrics) or local R-peak detection, both of which apply
        # median + dead-zone stabilization.  Do NOT re-smooth here.
        heart_rate = heart_rate_raw
        self._last_displayed_hr = heart_rate

        
        # Calculate PR Interval using atrial vector method (Lead I + aVF) - GE/Philips/Fluke standard
        # CLINICAL-GRADE: Build median beats for Lead I and aVF for atrial vector P-onset detection
        lead_i_data = self.data[0] if len(self.data) > 0 else None
        lead_avf_data = self.data[5] if len(self.data) > 5 else None
        
        median_beat_i = None
        median_beat_avf = None
        
        if lead_i_data is not None and lead_avf_data is not None and len(r_peaks) >= min_beats_for_median:
            try:
                _, median_beat_i = build_median_beat(lead_i_data, r_peaks, fs, min_beats=min_beats_for_median)
                _, median_beat_avf = build_median_beat(lead_avf_data, r_peaks, fs, min_beats=min_beats_for_median)
            except Exception as e:
                print(f" ⚠️ Error building Lead I/aVF median beats for PR: {e}")
                median_beat_i = None
                median_beat_avf = None
        
        # REAL MODE: Measure PR using atrial vector method (clinical-grade)
        pr_interval_raw = measure_pr_from_median_beat(
            median_beat_ii, time_axis, fs, tp_baseline_ii,
            median_beat_i=median_beat_i, 
            median_beat_avf=median_beat_avf,
            rr_ms=rr_ms
        )
        
        # OVERRIDE: Use user's comprehensive metrics if available
        if user_metrics["pr_interval"] is not None:
            pr_interval_raw = user_metrics["pr_interval"]
        # OPTIMIZED: Reduced print frequency for better performance
        if not hasattr(self, '_pr_print_count'):
            self._pr_print_count = 0
        self._pr_print_count += 1
        if self._pr_print_count % 30 == 0:  # Print every 30th calculation
            if pr_interval_raw is None or pr_interval_raw <= 0:
                print(f" ⚠️ PR calculation returned: {pr_interval_raw}, using fallback")
            else:
                print(f" ✓ PR calculated: {pr_interval_raw} ms")
        
        # FIX-TL1: Stabilization — hold last good PR. Do NOT fabricate a formula-based
        # value: a fake PR poisons the EMA buffer and shows wrong values for several seconds.
        if pr_interval_raw is None or pr_interval_raw <= 0:
            pr_interval_raw = self._hold_or_zero('pr_interval', 0)
            # If still 0 at startup, leave as 0 — display will show "--" until real signal arrives.
        
        # FIX-TL4: `calculate_all_ecg_metrics()` already stabilizes PR.  A second
        # smoothing layer here makes the displayed value lag the actual measured
        # interval and can disagree with the graph / console output.
        if pr_interval_raw > 0:
            self.pr_interval = int(round(pr_interval_raw))
        elif not hasattr(self, 'pr_interval'):
            # Keep a defined value for startup / low-signal states.
            self.pr_interval = 0
        
        # FIX-TL3: QRS Blended Approach with High Stability
        # user_metrics_qrs uses Curtin 2018 algorithm (Pan-Tompkins + Adaptive Windows) -> highly accurate & consistent.
        user_metrics_qrs = user_metrics.get("qrs_duration") or 0
        median_beat_qrs = measure_qrs_duration_from_median_beat(
            median_beat_ii, time_axis, fs, tp_baseline_ii
        ) or 0
        
        _global_qrs_used = (
            getattr(self, 'last_qrs_method', 'single-lead') == 'global-multilead'
        )

        if _global_qrs_used and user_metrics_qrs > 0:
            # The global boundary already spans every connected lead. Blending a
            # Lead-II median-beat width back in would drag it toward the very
            # single-lead under-estimate the global rule exists to correct.
            qrs_duration_raw = user_metrics_qrs
        elif user_metrics_qrs > 0 and median_beat_qrs > 0:
            # If median_beat_qrs is unstable or contaminated by T-wave tail (>15ms difference),
            # trust user_metrics_qrs (Curtin 2018) to prevent QRS value jumping.
            if abs(median_beat_qrs - user_metrics_qrs) <= 15:
                qrs_duration_raw = int(round(0.7 * user_metrics_qrs + 0.3 * median_beat_qrs))
            else:
                qrs_duration_raw = user_metrics_qrs
        else:
            qrs_duration_raw = user_metrics_qrs if user_metrics_qrs > 0 else median_beat_qrs

        if not hasattr(self, '_qrs_print_count'):
            self._qrs_print_count = 0
        self._qrs_print_count += 1
        if self._qrs_print_count % 30 == 0:
            if _global_qrs_used:
                _leads = getattr(self, 'last_qrs_leads_used', 1)
                print(f" ✓ QRS: {qrs_duration_raw} ms "
                      f"(global multi-lead boundary, {_leads} leads, "
                      f"lead-II-only={median_beat_qrs} ms)")
            else:
                print(f" ✓ QRS: {qrs_duration_raw} ms (raw_curtin={user_metrics_qrs}, median_beat={median_beat_qrs})")
        
        # Hold last good QRS if current frame fails
        if qrs_duration_raw is None or qrs_duration_raw <= 0:
            qrs_duration_raw = self._hold_or_zero('last_qrs_duration', 0)
        
        # REAL MODE: Smooth QRS with 15-beat median buffer to eliminate noise jitter
        if not hasattr(self, '_qrs_smooth_buffer'):
            self._qrs_smooth_buffer = []
        if qrs_duration_raw > 0:
            self._qrs_smooth_buffer.append(qrs_duration_raw)
            if len(self._qrs_smooth_buffer) > 15:  # Increased to 15 for rock-solid stability
                self._qrs_smooth_buffer.pop(0)
        
        if len(self._qrs_smooth_buffer) > 0:
            smoothed_qrs = int(round(np.median(self._qrs_smooth_buffer)))
        else:
            smoothed_qrs = qrs_duration_raw if qrs_duration_raw > 0 else getattr(self, 'last_qrs_duration', 0)
        
        # Deadband & Hold-and-jump logic for QRS (eliminates 1-5ms UI jitter)
        if not hasattr(self, '_last_displayed_qrs'):
            self._last_displayed_qrs = smoothed_qrs
        if not hasattr(self, '_pending_qrs_value'):
            self._pending_qrs_value = None
        if not hasattr(self, '_pending_qrs_start_time'):
            self._pending_qrs_start_time = 0
        
        qrs_diff = abs(smoothed_qrs - self._last_displayed_qrs)
        if qrs_diff <= 5:
            # Deadband: Hold current value if drift is within 5 ms (eliminates micro-jitter)
            pass
        elif qrs_diff <= 12:
            # Small change (6-12 ms): update smoothly
            self._last_displayed_qrs = int(round(0.8 * self._last_displayed_qrs + 0.2 * smoothed_qrs))
            self._pending_qrs_value = None
        else:
            # Large change (>12 ms): hold old value until new value is stable for 0.5s
            current_time = time.time()
            if self._pending_qrs_value is None:
                self._pending_qrs_value = smoothed_qrs
                self._pending_qrs_start_time = current_time
            else:
                if abs(smoothed_qrs - self._pending_qrs_value) <= 4:
                    if current_time - self._pending_qrs_start_time >= 0.5:
                        self._last_displayed_qrs = smoothed_qrs
                        self._pending_qrs_value = None
                else:
                    self._pending_qrs_value = smoothed_qrs
                    self._pending_qrs_start_time = current_time
        
        self.last_qrs_duration = self._last_displayed_qrs

        # Say which measurement the number came from — a global width and a
        # Lead-II width are different claims and should not look identical.
        try:
            from .qrs_detection import describe_qrs_method
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setToolTip(
                    describe_qrs_method(
                        getattr(self, 'last_qrs_method', 'single-lead'),
                        getattr(self, 'last_qrs_leads_used', 1),
                    )
                )
        except Exception:
            pass
        
        # REAL MODE: Calculate QT Interval from median beat using clinical tangent method
        qt_interval_raw = measure_qt_from_median_beat(median_beat_ii, time_axis, fs, tp_baseline_ii, rr_ms=rr_ms)
        
        # OVERRIDE: Use user's comprehensive metrics if available
        if user_metrics["qt_interval"] is not None:
            qt_interval_raw = user_metrics["qt_interval"]
        # Stabilization: hold last good QT if new one fails
        if qt_interval_raw is None or qt_interval_raw <= 0:
            qt_interval_raw = self._hold_or_zero('last_qt_interval', 0)
        
        # STABILIZATION: Smooth QT with buffer (same as QRS/PR)
        if not hasattr(self, '_qt_smooth_buffer'):
            self._qt_smooth_buffer = []
        if qt_interval_raw > 0:
            self._qt_smooth_buffer.append(qt_interval_raw)
            if len(self._qt_smooth_buffer) > 20:  # Increased from 7 to 20 for better stability
                self._qt_smooth_buffer.pop(0)
        
        if len(self._qt_smooth_buffer) > 0:
            smoothed_qt = int(round(np.median(self._qt_smooth_buffer)))
        else:
            smoothed_qt = qt_interval_raw if qt_interval_raw > 0 else getattr(self, 'last_qt_interval', 0)
        
        # Hold-and-jump logic for QT (same as QRS/PR)
        if not hasattr(self, '_last_displayed_qt'):
            self._last_displayed_qt = smoothed_qt
        if not hasattr(self, '_pending_qt_value'):
            self._pending_qt_value = None
        if not hasattr(self, '_pending_qt_start_time'):
            self._pending_qt_start_time = 0
        
        qt_diff = abs(smoothed_qt - self._last_displayed_qt)
        
        # DEAD ZONE: Only update if change is > 5ms to prevent flickering
        if self._last_displayed_qt == 0 and smoothed_qt > 0:
            # Initial update - update immediately
            self._last_displayed_qt = smoothed_qt
            self._pending_qt_value = None
            qt_interval = smoothed_qt
        elif qt_diff <= 5:
            # Change too small: Keep old value
            qt_interval = self._last_displayed_qt
            self._pending_qt_value = None
        elif qt_diff <= 15:  # Small change: update immediately (allow ±15 ms jitter for QT)
            self._last_displayed_qt = smoothed_qt
            self._pending_qt_value = None
            qt_interval = smoothed_qt
        else:
            # Large change: hold old value until new value is stable
            current_time = time.time()
            if self._pending_qt_value is None:
                self._pending_qt_value = smoothed_qt
                self._pending_qt_start_time = current_time
                qt_interval = self._last_displayed_qt
            else:
                if abs(smoothed_qt - self._pending_qt_value) <= 10:  # Allow ±10 ms jitter
                    if current_time - self._pending_qt_start_time >= 0.5:  # Stable for 0.5 seconds (reduced for real-time)
                        self._last_displayed_qt = smoothed_qt
                        self._pending_qt_value = None
                        qt_interval = smoothed_qt
                    else:
                        qt_interval = self._last_displayed_qt
                else:
                    # Value changed again, reset timer
                    self._pending_qt_value = smoothed_qt
                    self._pending_qt_start_time = current_time
                    qt_interval = self._last_displayed_qt

        self.last_qt_interval = qt_interval
        
        # Calculate QTc (Bazett): QTc = QT / sqrt(RR_sec)
        # IMPORTANT: Always compute from the already-stabilized qt_interval integer
        # (the exact value shown in the display) — do NOT use user_metrics["qtc_interval"]
        # which came from a separate EMA buffer and can diverge by ±1 ms.
        # At 60 BPM: RR = 1000 ms → sqrt(1.0) = 1.0 → QTc = QT exactly.
        if qt_interval > 0 and rr_ms > 0:
            RR = rr_ms / 1000.0  # RR in seconds
            qtc_interval = int(round((qt_interval / 1000.0) / np.sqrt(RR) * 1000.0))

            # Validation: QTc should be in reasonable range (200-600 ms)
            if qtc_interval < 200 or qtc_interval > 600:
                # OPTIMIZED: Reduced print frequency for better performance
                if not hasattr(self, '_qtc_range_warn_count'):
                    self._qtc_range_warn_count = 0
                self._qtc_range_warn_count += 1
                if self._qtc_range_warn_count % 50 == 1:  # Print every 50th warning
                    print(f" ⚠️ QTc out of range: {qtc_interval} ms (QT={qt_interval} ms, RR={rr_ms} ms)")
        else:
            qtc_interval = getattr(self, 'last_qtc_interval', 0)
        
        # Calculate QTcF (Fridericia) using smoothed heart_rate for consistency
        qtcf_interval = self.calculate_qtcf_interval(qt_interval, rr_ms)
        self.last_qtc_interval = qtc_interval
        self.last_qtcf_interval = qtcf_interval
        
        # Calculate axes from median beats (P/QRS/T)
        qrs_axis = self.calculate_qrs_axis_from_median()
        p_axis = self.calculate_p_axis_from_median()
        t_axis = self.calculate_t_axis_from_median()
        self.last_qrs_axis = qrs_axis
        self.last_p_axis = p_axis
        self.last_t_axis = t_axis
        
        # Calculate QRS-T angle (highly valuable clinical metric)
        from .clinical_measurements import calculate_qrs_t_angle
        qrs_t_angle = calculate_qrs_t_angle(qrs_axis, t_axis)
        self.last_qrs_t_angle = qrs_t_angle
        
        # REAL MODE: Calculate P-wave duration from median beat (returns ms)
        p_duration_raw = measure_p_duration_from_median_beat(median_beat_ii, time_axis, fs, tp_baseline_ii)
        # Stabilization: hold last good P if new one fails
        if p_duration_raw is None or p_duration_raw <= 0:
            p_duration_raw = getattr(self, 'last_p_duration', 0)
        
        # STABILIZATION: Smooth P duration with buffer (same as QRS/PR/QT)
        if not hasattr(self, '_p_smooth_buffer'):
            self._p_smooth_buffer = []
        if p_duration_raw > 0:
            self._p_smooth_buffer.append(p_duration_raw)
            if len(self._p_smooth_buffer) > 7:
                self._p_smooth_buffer.pop(0)
        
        if len(self._p_smooth_buffer) > 0:
            smoothed_p = int(round(np.median(self._p_smooth_buffer)))
        else:
            smoothed_p = p_duration_raw if p_duration_raw > 0 else getattr(self, 'last_p_duration', 0)
        
        # Hold-and-jump logic for P duration (same as QRS/PR/QT)
        if not hasattr(self, '_last_displayed_p'):
            self._last_displayed_p = smoothed_p
        if not hasattr(self, '_pending_p_value'):
            self._pending_p_value = None
        if not hasattr(self, '_pending_p_start_time'):
            self._pending_p_start_time = 0
        
        p_diff = abs(smoothed_p - self._last_displayed_p)
        if p_diff <= 10:  # Small change: update immediately (allow ±10 ms jitter for P)
            self._last_displayed_p = smoothed_p
            self._pending_p_value = None
        else:
            # Large change: hold old value until new value is stable
            current_time = time.time()
            if self._pending_p_value is None:
                self._pending_p_value = smoothed_p
                self._pending_p_start_time = current_time
            else:
                if abs(smoothed_p - self._pending_p_value) <= 5:  # Allow ±5 ms jitter
                    if current_time - self._pending_p_start_time >= 0.5:  # Stable for 0.5 seconds (reduced for real-time)
                        self._last_displayed_p = smoothed_p
                        self._pending_p_value = None
                else:
                    # Value changed again, reset timer
                    self._pending_p_value = smoothed_p
                    self._pending_p_start_time = current_time
        
        p_duration = self._last_displayed_p
        self.last_p_duration = p_duration
        
        # Calculate RV5/SV1 from median beats
        rv5_mv, sv1_mv = self.calculate_rv5_sv1_from_median()
        if rv5_mv is not None: self._last_rv5 = float(rv5_mv)
        if sv1_mv is not None: self._last_sv1 = float(sv1_mv)
        # Store for report generation
        if rv5_mv is not None: self._last_rv5 = float(rv5_mv)
        if sv1_mv is not None: self._last_sv1 = float(sv1_mv)
        
        # VALIDATION: Ensure clinical measurements are independent of display filters
        try:
            from .clinical_validation import (
                validate_rv5_sv1_signs, validate_rv5_sv1_sum,
                validate_qtc_formulas, validate_median_beat_beats
            )
            # OPTIMIZED: Reduce validation warning frequency for better performance
            if not hasattr(self, '_validation_warn_count'):
                self._validation_warn_count = 0
            self._validation_warn_count += 1
            
            # Validate RV5/SV1 signs (only print warnings every 50th occurrence)
            if rv5_mv is not None and sv1_mv is not None:
                try:
                    validate_rv5_sv1_signs(rv5_mv, sv1_mv)
                except AssertionError as e:
                    if self._validation_warn_count % 50 == 1:  # Print every 50th warning
                        print(f"⚠️ Clinical validation warning: {e}")
            # Validate QTc formulas
            if qt_interval > 0 and rr_ms > 0:
                try:
                    validate_qtc_formulas(qt_interval, rr_ms, qtc_interval, qtcf_interval)
                except AssertionError as e:
                    if self._validation_warn_count % 50 == 1:  # Print every 50th warning
                        print(f"⚠️ Clinical validation warning: {e}")
            # Validate median beat uses 8-12 beats
            if len(r_peaks) >= 8:
                num_beats_used = min(len(r_peaks), 12)
                try:
                    validate_median_beat_beats(num_beats_used)
                except AssertionError as e:
                    if self._validation_warn_count % 50 == 1:  # Print every 50th warning
                        print(f"⚠️ Clinical validation warning: {e}")
        except ImportError:
            pass  # Validation module not available
        
        # Update UI metrics (dashboard only shows: BPM, PR, P, QT/QTc, timer)
        # Use the stabilized / stored values for PR and QRS so the display matches
        # the same values used elsewhere in the report.
        # Force immediate update on first calculation after acquisition starts
        force_update = not hasattr(self, '_metrics_calculated_once')
        if force_update:
            self._metrics_calculated_once = True
        
        # OPTIMIZED: Reduced print frequency for better performance
        if not hasattr(self, '_final_metrics_print_count'):
            self._final_metrics_print_count = 0
        self._final_metrics_print_count += 1
        if self._final_metrics_print_count % 20 == 0:  # Print every 20th calculation
            print(f" 📊 Final Metrics - HR: {heart_rate}, PR: {self.pr_interval}, QRS: {self.last_qrs_duration}, P: {self.last_p_duration}, QT: {qt_interval}, QTc: {qtc_interval}")
        
        # Use stabilized values for display
        # FIX-D1: pass rr_interval so the RR label is updated
        self.update_ecg_metrics_display(
            heart_rate,
            self.pr_interval,
            self.last_qrs_duration,
            self.last_p_duration,
            qt_interval,
            qtc_interval,
            qtcf_interval,
            force_immediate=force_update,
            rr_interval=getattr(self, 'last_rr_interval', 0),
        )

        # Keep the dashboard and any open detailed ECG window in sync with the
        # same live arrhythmia interpretation, even when the expanded view is not open.
        try:
            self.update_latest_rhythm_interpretation()
        except Exception:
            pass

    def _get_reference_metrics_for_bpm(self, bpm):
        """
        DEPRECATED: This function is no longer used.
        Real hardware values are now always used instead of reference table values.
        
        Returns None to ensure real calculations are always used.
        """
        # Always return None - real values from hardware are used instead
        return None

    def calculate_heart_rate(self, lead_data):
        """Calculate heart rate from Lead II data - wrapper for modular function
        
         CLINICAL ANALYSIS: Must receive RAW clinical data, NOT display-processed data.
        This function is called with self.data[1] which contains raw ECG values.
        """
        sampler = getattr(self, 'sampler', None)
        sampling_rate = getattr(self, 'sampling_rate', 500.0)
        # Use instance id for per-instance smoothing (prevents cross-contamination)
        instance_id = id(self) if hasattr(self, '__class__') else 'ecg_test_page'
        return calculate_heart_rate_from_signal(lead_data, sampling_rate=sampling_rate, sampler=sampler, instance_id=instance_id)
    
    def _calculate_heart_rate_old(self, lead_data):
        """OLD calculate_heart_rate implementation - kept for reference"""
        try:
            # Early exit: if no real signal, report 0 instead of fallback
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                    return 0
            except Exception:
                return 0

            # Validate input data
            if not isinstance(lead_data, (list, np.ndarray)) or len(lead_data) < 200:
                print(" Insufficient data for heart rate calculation")
                return 60  # Default fallback

            # Convert to numpy array for processing
            try:
                lead_data = np.asarray(lead_data, dtype=float)
            except Exception as e:
                print(f" Error converting lead data to array: {e}")
                return 60

            # Check for invalid values
            if np.any(np.isnan(lead_data)) or np.any(np.isinf(lead_data)):
                print(" Invalid values (NaN/Inf) in lead data")
                return 60

            # Use measured sampling rate if available; default to 500 Hz (unified fallback)
            # CRITICAL: If detection works, use detected rate (should be 500 Hz for hardware)
            # If detection fails, use 250 Hz fallback (consistent across platforms)
            import platform
            is_windows = platform.system() == 'Windows'
            platform_tag = "[Windows]" if is_windows else "[macOS/Linux]"
            
            fs = 500.0  # Standard fallback for all platforms
            try:
                if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                    detected_rate = self.sampler.sampling_rate
                    if detected_rate > 10 and np.isfinite(detected_rate):
                        fs = float(detected_rate)
                        # Debug: Log sampling rate detection (first few times only)
                        if not hasattr(self, '_fs_debug_count'):
                            self._fs_debug_count = 0
                        self._fs_debug_count += 1
                        if self._fs_debug_count <= 3:
                            print(f" {platform_tag} Heart rate calculation using detected sampling rate: {fs:.1f} Hz")
                    else:
                        if is_windows:
                            print(f" {platform_tag} Invalid detected sampling rate: {detected_rate}, using fallback 500.0 Hz")
                elif hasattr(self, 'sampling_rate') and self.sampling_rate > 10:
                    fs = float(self.sampling_rate)
                    if not hasattr(self, '_fs_debug_count'):
                        self._fs_debug_count = 0
                    self._fs_debug_count += 1
                    if self._fs_debug_count <= 3:
                        print(f" {platform_tag} Heart rate calculation using sampling rate: {fs:.1f} Hz (from self.sampling_rate)")
                else:
                    if is_windows:
                        print(f" {platform_tag} Sampling rate not available, using fallback 500.0 Hz")
            except Exception as e:
                print(f" Error getting sampling rate: {e}")
                fs = 500.0  # Standard fallback
                if not hasattr(self, '_fs_debug_count'):
                    self._fs_debug_count = 0
                self._fs_debug_count += 1
                if self._fs_debug_count <= 3:
                    print(f" {platform_tag} Using default sampling rate: {fs} Hz (sampling rate detection failed)")
            
            # Validation
            if fs <= 0 or not np.isfinite(fs):
                if is_windows:
                    print(f" {platform_tag} Invalid sampling rate detected: {fs}, using fallback 500.0 Hz")
                fs = 500.0  # Fallback

            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz)
            try:
                from scipy.signal import butter, filtfilt
                nyquist = fs / 2
                low = max(0.001, 0.5 / nyquist)
                high = min(0.999, 40 / nyquist)
                if low >= high:
                    print(" Invalid filter parameters")
                    return 60
                b, a = butter(4, [low, high], btype='band')
                filtered_signal = filtfilt(b, a, lead_data)
                if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                    print(" Filter produced invalid values")
                    return 60
            except Exception as e:
                print(f" Error in signal filtering: {e}")
                return 60

            # Find R-peaks using scipy with robust parameters
            try:
                from scipy.signal import find_peaks
                signal_mean = np.mean(filtered_signal)
                signal_std = np.std(filtered_signal)
                if signal_std == 0:
                    print(" No signal variation detected")
                    return 60
                
                # SMART ADAPTIVE PEAK DETECTION (10-300 BPM with BPM-based selection)
                # Run multiple detections and choose based on CALCULATED BPM
                height_threshold = signal_mean + 0.5 * signal_std
                prominence_threshold = signal_std * 0.4
                
                # Run 3 detection strategies
                detection_results = []
                
                # Strategy 1: Conservative (best for 10-120 BPM)
                # Distance set to minimum RR for highest BPM in range (120 BPM = 500ms)
                # RR interval filtering (200-6000ms) will handle the full 10-300 BPM range
                peaks_conservative, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.4 * fs),  # 400ms - prevents false peaks, allows 10-300 BPM via RR filtering
                    prominence=prominence_threshold
                )
                if len(peaks_conservative) >= 2:
                    rr_cons = np.diff(peaks_conservative) * (1000 / fs)
                    # Accept RR intervals from 200–6000 ms (300–10 BPM)
                    valid_cons = rr_cons[(rr_cons >= 200) & (rr_cons <= 6000)]
                    if len(valid_cons) > 0:
                        bpm_cons = 60000 / np.median(valid_cons)
                        std_cons = np.std(valid_cons)
                        detection_results.append(('conservative', peaks_conservative, bpm_cons, std_cons))
                
                # Strategy 2: Normal (best for 100-180 BPM)
                peaks_normal, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.3 * fs),  # 240ms - medium distance
                    prominence=prominence_threshold
                )
                if len(peaks_normal) >= 2:
                    rr_norm = np.diff(peaks_normal) * (1000 / fs)
                    # Accept RR intervals from 200–6000 ms (300–10 BPM)
                    valid_norm = rr_norm[(rr_norm >= 200) & (rr_norm <= 6000)]
                    if len(valid_norm) > 0:
                        bpm_norm = 60000 / np.median(valid_norm)
                        std_norm = np.std(valid_norm)
                        detection_results.append(('normal', peaks_normal, bpm_norm, std_norm))
                
                # Strategy 3: Tight (best for 160-300 BPM)
                peaks_tight, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.2 * fs),  # 160ms - tight distance for high BPM
                    prominence=prominence_threshold
                )
                if len(peaks_tight) >= 2:
                    rr_tight = np.diff(peaks_tight) * (1000 / fs)
                    # Accept RR intervals from 200–6000 ms (300–10 BPM)
                    valid_tight = rr_tight[(rr_tight >= 200) & (rr_tight <= 6000)]
                    if len(valid_tight) > 0:
                        bpm_tight = 60000 / np.median(valid_tight)
                        std_tight = np.std(valid_tight)
                        detection_results.append(('tight', peaks_tight, bpm_tight, std_tight))
                
                # Select based on BPM consistency (lowest std deviation = most stable)
                if detection_results:
                    # Sort by consistency (lower std = better)
                    detection_results.sort(key=lambda x: x[3])  # Sort by std
                    best_method, peaks, best_bpm, best_std = detection_results[0]
                    # print(f" Selected {best_method}: {best_bpm:.1f} BPM (std={best_std:.1f})")
                else:
                    # Fallback - use conservative distance to handle low BPM (10-120 BPM)
                    peaks, _ = find_peaks(
                        filtered_signal,
                        height=height_threshold,
                        distance=int(0.4 * fs),  # 400ms - prevents false peaks, allows 10-300 BPM via RR filtering
                        prominence=prominence_threshold
                    )
            except Exception as e:
                print(f" Error in peak detection: {e}")
                return 60

            if len(peaks) < 2:
                print(f" Insufficient peaks detected: {len(peaks)}")
                return 60

            # Calculate R-R intervals in milliseconds
            try:
                rr_intervals_ms = np.diff(peaks) * (1000 / fs)
                if len(rr_intervals_ms) == 0:
                    print(" No R-R intervals calculated")
                    return 60
            except Exception as e:
                print(f" Error calculating R-R intervals: {e}")
                return 60

            # Filter physiologically reasonable intervals (200-6000 ms)
            # 200 ms = 300 BPM (max), 6000 ms = 10 BPM (min)
            try:
                valid_intervals = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 6000)]
                if len(valid_intervals) == 0:
                    print(" No valid R-R intervals found")
                    return 60
            except Exception as e:
                print(f" Error filtering intervals: {e}")
                return 60

            # Calculate heart rate from median R-R interval (as in commit 8a6aaee)
            try:
                median_rr = np.median(valid_intervals)
                if median_rr <= 0:
                    print(" Invalid median R-R interval")
                    return 60
                heart_rate = 60000 / median_rr
                # Extended: stable 10–300 BPM range
                heart_rate = max(10, min(300, heart_rate))
                
                # Debug: Log calculated BPM with sampling rate (first few times only)
                if not hasattr(self, '_bpm_calc_debug_count'):
                    self._bpm_calc_debug_count = 0
                self._bpm_calc_debug_count += 1
                if self._bpm_calc_debug_count <= 5:
                    print(f" BPM Calculation: {heart_rate:.1f} BPM (sampling_rate={fs:.1f} Hz, median_RR={median_rr:.1f} ms, peaks={len(peaks)})")
                # Extra guard: avoid falsely reporting very high BPM when real rate is very low
                try:
                    window_sec = len(lead_data) / float(fs)
                except Exception:
                    window_sec = 0
                if heart_rate > 150 and window_sec >= 5.0:
                    # How many beats would we expect at this BPM over the window?
                    expected_peaks = (heart_rate * window_sec) / 60.0
                    # If we have far fewer peaks than expected, this "high BPM" is likely noise
                    if expected_peaks > len(peaks) * 3:
                        # Treat as extreme bradycardia scenario and clamp to minimum (10 bpm)
                        print(f" Suspicious high BPM ({heart_rate:.1f}) with too few peaks "
                              f"(expected≈{expected_peaks:.1f}, got={len(peaks)}). Clamping to 10 bpm.")
                        heart_rate = 10.0
                if np.isnan(heart_rate) or np.isinf(heart_rate):
                    print(" Invalid heart rate calculated")
                    return 60
                
                # STRONG ANTI-FLICKERING: Smooth BPM with strict outlier rejection
                hr_int = int(round(heart_rate))
                
                # Initialize smoothing buffer and EMA
                if not hasattr(self, '_hr_smooth_buffer'):
                    self._hr_smooth_buffer = []
                if not hasattr(self, '_hr_ema'):
                    self._hr_ema = None
                
                # SMOOTH BPM TRANSITION: No fluctuations during change - smooth transition to new value
                import time
                
                # Initialize lock mechanism
                if not hasattr(self, '_hr_locked'):
                    self._hr_locked = False
                if not hasattr(self, '_hr_unlock_buffer'):
                    self._hr_unlock_buffer = []
                if not hasattr(self, '_hr_unlock_start_time'):
                    self._hr_unlock_start_time = None
                if not hasattr(self, '_hr_prelock_buffer'):
                    self._hr_prelock_buffer = []
                if not hasattr(self, '_hr_transition_target'):
                    self._hr_transition_target = None
                
                if hasattr(self, '_last_stable_hr') and self._last_stable_hr is not None:
                    bpm_change = abs(hr_int - self._last_stable_hr)
                    current_time = time.time()
                    
                    # If BPM is LOCKED: Track changes and show smooth transition (no fluctuations)
                    if self._hr_locked:
                        # ANY CHANGE (>= 1 BPM): Start transition tracking
                        if bpm_change >= 1:
                            if self._hr_unlock_start_time is None:
                                # New change detected - start tracking
                                self._hr_unlock_start_time = current_time
                                self._hr_unlock_buffer = []
                                self._hr_transition_target = None
                            
                            self._hr_unlock_buffer.append(hr_int)
                            if len(self._hr_unlock_buffer) > 7:
                                self._hr_unlock_buffer.pop(0)
                            
                            # Check if change persisted
                            elapsed = current_time - self._hr_unlock_start_time
                            
                            # Small changes (1-4 BPM): Require 0.5 seconds persistence
                            # Large changes (>= 5 BPM): Require 0.3 seconds persistence
                            required_time = 0.5 if bpm_change < 5 else 0.3
                            min_readings = 5 if bpm_change < 5 else 3
                            
                            if elapsed >= required_time and len(self._hr_unlock_buffer) >= min_readings:
                                # Check if readings are stable (within 2 BPM)
                                last_readings = self._hr_unlock_buffer[-min_readings:]
                                min_val = min(last_readings)
                                max_val = max(last_readings)
                                if (max_val - min_val) <= 2:
                                    # Change confirmed - calculate target and unlock
                                    median_new_bpm = int(round(np.median(last_readings)))
                                    self._hr_locked = False
                                    self._last_stable_hr = median_new_bpm
                                    self._hr_unlock_buffer = []
                                    self._hr_unlock_start_time = None
                                    self._hr_transition_target = None
                                    smoothed_hr = median_new_bpm
                                else:
                                    # Not stable - show smooth transition to average (prevents fluctuations)
                                    if self._hr_transition_target is None:
                                        self._hr_transition_target = int(round(np.mean(last_readings)))
                                    # Smooth transition: gradually move from current to target
                                    current_bpm = self._last_stable_hr
                                    target_bpm = self._hr_transition_target
                                    # Use weighted average for smooth transition (70% current, 30% target)
                                    smoothed_hr = int(round(0.7 * current_bpm + 0.3 * target_bpm))
                            else:
                                # Still tracking - show smooth MONOTONIC transition (no fluctuations, no overshoot)
                                if len(self._hr_unlock_buffer) >= 2:
                                    # Set target ONCE at the start - don't change it during confirmation (prevents fluctuations)
                                    if self._hr_transition_target is None:
                                        # Calculate initial target from first few readings
                                        avg_new = int(round(np.mean(self._hr_unlock_buffer[-3:])))
                                        self._hr_transition_target = avg_new
                                    
                                    # Smooth MONOTONIC transition display (always moves towards target, never overshoots)
                                    current_bpm = self._last_stable_hr
                                    target_bpm = self._hr_transition_target
                                    
                                    # Gradual transition based on elapsed time (smooth increment/decrement)
                                    transition_progress = min(1.0, elapsed / required_time)
                                    calculated_bpm = int(round(current_bpm + (target_bpm - current_bpm) * transition_progress))
                                    
                                    # Ensure monotonic transition (never overshoot target, always move towards it)
                                    if target_bpm > current_bpm:
                                        # Increasing: clamp to [current_bpm, target_bpm]
                                        smoothed_hr = max(current_bpm, min(calculated_bpm, target_bpm))
                                    else:
                                        # Decreasing: clamp to [target_bpm, current_bpm]
                                        smoothed_hr = max(target_bpm, min(calculated_bpm, current_bpm))
                                else:
                                    # Not enough readings - keep locked
                                    smoothed_hr = self._last_stable_hr
                        
                        # NO CHANGE (< 1 BPM): Reset tracking, keep locked
                        else:
                            smoothed_hr = self._last_stable_hr
                            self._hr_unlock_buffer = []
                            self._hr_unlock_start_time = None
                            self._hr_transition_target = None
                    
                    # If BPM is NOT LOCKED: Collect 5 readings for accuracy, then lock
                    else:
                        # Collect readings before locking for accuracy
                        self._hr_prelock_buffer.append(hr_int)
                        if len(self._hr_prelock_buffer) > 7:
                            self._hr_prelock_buffer.pop(0)
                        
                        # Lock after 5 stable readings (more accurate than 3 readings - reduces 100 vs 102 error)
                        if len(self._hr_prelock_buffer) >= 5:
                            # Use median of last 5 readings for accuracy (reduces 100 vs 102 error)
                            median_bpm = int(round(np.median(self._hr_prelock_buffer[-5:])))
                            self._last_stable_hr = median_bpm
                            self._hr_locked = True  # LOCK after accurate reading
                            self._hr_prelock_buffer = []  # Clear buffer
                            smoothed_hr = self._last_stable_hr
                        else:
                            # Not enough readings yet - use current value temporarily
                            smoothed_hr = hr_int
                            self._last_stable_hr = smoothed_hr
                else:
                    # First reading - use directly
                    self._last_stable_hr = hr_int
                    smoothed_hr = self._last_stable_hr
                    if not hasattr(self, '_hr_ema'):
                        self._hr_ema = float(smoothed_hr)
                    if not hasattr(self, '_hr_smooth_buffer'):
                        self._hr_smooth_buffer = [smoothed_hr]
                
                return self._last_stable_hr
            except Exception as e:
                print(f" Error calculating final BPM: {e}")
                return 60
        except Exception as e:
            print(f" Critical error in calculate_heart_rate: {e}")
            return 60

    def calculate_wave_amplitudes(self):
        """Calculate P, QRS, and T wave amplitudes from all leads for report generation"""
        try:
            amplitudes = {
                'p_amp': 0.0,
                'qrs_amp': 0.0,
                't_amp': 0.0,
                'rv5': 0.0,
                'sv1': 0.0
            }
            
            # Get Lead II for P, QRS, T measurements
            lead_ii_data = self.data[1] if len(self.data) > 1 else None
            if lead_ii_data is None or len(lead_ii_data) < 200:
                return amplitudes
            
            # Check for real signal
            if np.all(lead_ii_data == 0) or np.std(lead_ii_data) < 0.1:
                return amplitudes
            
            # Get sampling rate - default to 250 Hz (unified fallback)
            fs = 500.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                fs = float(self.sampler.sampling_rate)
            
            # Filter signal
            from scipy.signal import butter, filtfilt
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = min(40.0 / nyquist, 0.99)
            b, a = butter(2, [low, high], btype='band')
            filtered_data = filtfilt(b, a, lead_ii_data)
            
            # Detect R-peaks
            from scipy.signal import find_peaks
            squared = np.square(np.diff(filtered_data))
            integrated = np.convolve(squared, np.ones(int(0.15 * fs)) / (0.15 * fs), mode='same')
            threshold = np.mean(integrated) + 0.5 * np.std(integrated)
            r_peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.15 * fs))  # Reduced from 0.6 to 0.15 for high BPM (360 max)
            
            if len(r_peaks) < 2:
                return amplitudes
            
            # Analyze each beat and average the amplitudes
            p_amps = []
            qrs_amps = []
            t_amps = []
            
            for r_idx in r_peaks[1:-1]:  # Skip first and last to avoid edge effects
                try:
                    # P-wave amplitude (120-200ms before R)
                    p_start = max(0, r_idx - int(0.20 * fs))
                    p_end = max(0, r_idx - int(0.12 * fs))
                    if p_end > p_start:
                        p_segment = filtered_data[p_start:p_end]
                        baseline = np.mean(filtered_data[max(0, p_start - int(0.05 * fs)):p_start])
                        p_peak = np.max(p_segment) - baseline
                        if p_peak > 0:
                            p_amps.append(p_peak)
                    
                    # QRS amplitude (Q to peak R to S)
                    qrs_start = max(0, r_idx - int(0.08 * fs))
                    qrs_end = min(len(filtered_data), r_idx + int(0.08 * fs))
                    if qrs_end > qrs_start:
                        qrs_segment = filtered_data[qrs_start:qrs_end]
                        qrs_amp = np.max(qrs_segment) - np.min(qrs_segment)
                        if qrs_amp > 0:
                            qrs_amps.append(qrs_amp)
                    
                    # T-wave amplitude (100-300ms after R)
                    t_start = min(len(filtered_data), r_idx + int(0.10 * fs))
                    t_end = min(len(filtered_data), r_idx + int(0.30 * fs))
                    if t_end > t_start:
                        t_segment = filtered_data[t_start:t_end]
                        baseline = np.mean(filtered_data[r_idx:t_start])
                        t_peak = np.max(t_segment) - baseline
                        if t_peak > 0:
                            t_amps.append(t_peak)
                
                except Exception as e:
                    continue
            
            # Calculate median amplitudes (more robust than mean)
            if len(p_amps) > 0:
                amplitudes['p_amp'] = np.median(p_amps)
            if len(qrs_amps) > 0:
                amplitudes['qrs_amp'] = np.median(qrs_amps)
            if len(t_amps) > 0:
                amplitudes['t_amp'] = np.median(t_amps)
            
            # Calculate RV5 and SV1 for specific leads (GE/Hospital Standard)
            # Lead V5 is index 10, Lead V1 is index 6
            # CRITICAL: Use RAW ECG data (self.data), not display-filtered signals
            # Measurements must be from median beat, relative to TP baseline (isoelectric segment before P-wave)
            
            if len(self.data) > 10:
                lead_v5_data = self.data[10]  # RAW V5 data
                if lead_v5_data is not None and len(lead_v5_data) > 200 and np.std(lead_v5_data) > 0.1:
                    # Apply minimal bandpass filter ONLY for R-peak detection (0.5-40 Hz)
                    # This does NOT affect amplitude measurements - we use raw data for measurements
                    filtered_v5 = filtfilt(b, a, lead_v5_data)
                    # Detect R-peaks in V5
                    squared_v5 = np.square(np.diff(filtered_v5))
                    integrated_v5 = np.convolve(squared_v5, np.ones(int(0.15 * fs)) / (0.15 * fs), mode='same')
                    threshold_v5 = np.mean(integrated_v5) + 0.5 * np.std(integrated_v5)
                    r_peaks_v5, _ = find_peaks(integrated_v5, height=threshold_v5, distance=int(0.15 * fs))
                    
                    # Measure RV5: max(QRS in V5) - TP_baseline_V5 (must be positive, in mV)
                    rv5_amps = []
                    for r_idx in r_peaks_v5[1:-1]:
                        try:
                            # QRS window: ±80ms around R-peak
                            qrs_start = max(0, r_idx - int(0.08 * fs))
                            qrs_end = min(len(lead_v5_data), r_idx + int(0.08 * fs))
                            if qrs_end > qrs_start:
                                # Use RAW data for amplitude measurement
                                qrs_segment = lead_v5_data[qrs_start:qrs_end]
                                
                                # TP baseline: isoelectric segment before P-wave onset
                                # Use longer segment (150-350ms before R) for stable baseline
                                tp_start = max(0, r_idx - int(0.35 * fs))
                                tp_end = max(0, r_idx - int(0.15 * fs))
                                if tp_end > tp_start:
                                    tp_segment = lead_v5_data[tp_start:tp_end]
                                    tp_baseline = np.median(tp_segment)  # Median for robustness
                                else:
                                    # Fallback: short segment before QRS
                                    tp_baseline = np.median(lead_v5_data[max(0, qrs_start - int(0.05 * fs)):qrs_start])
                                
                                # RV5 = max(QRS) - TP_baseline (positive, in mV)
                                # Convert from ADC counts to mV using hardware calibration
                                # Adjusted calibration factor to match GE/Philips reference values
                                # Reference: RV5 should be ~0.972 mV, current gives ~1.316 mV
                                # Adjustment factor: 0.972/1.316 ≈ 0.74, so multiply calibration by 1.35
                                # Original: 1 mV = 1517.2 ADC → Adjusted: 1 mV = 1517.2 * 1.35 ≈ 2048 ADC
                                r_amp_adc = np.max(qrs_segment) - tp_baseline
                                if r_amp_adc > 0:
                                    # Convert ADC to mV: Adjusted calibration factor for GE/Philips alignment
                                    r_amp_mv = r_amp_adc / 2048.0  # Adjusted ADC to mV conversion
                                    rv5_amps.append(r_amp_mv)
                        except:
                            continue
                    
                    if len(rv5_amps) > 0:
                        amplitudes['rv5'] = np.median(rv5_amps)  # Median beat approach
            
            if len(self.data) > 6:
                lead_v1_data = self.data[6]  # RAW V1 data
                if lead_v1_data is not None and len(lead_v1_data) > 200 and np.std(lead_v1_data) > 0.1:
                    # Apply minimal bandpass filter ONLY for R-peak detection (0.5-40 Hz)
                    filtered_v1 = filtfilt(b, a, lead_v1_data)
                    # Detect R-peaks in V1
                    squared_v1 = np.square(np.diff(filtered_v1))
                    integrated_v1 = np.convolve(squared_v1, np.ones(int(0.15 * fs)) / (0.15 * fs), mode='same')
                    threshold_v1 = np.mean(integrated_v1) + 0.5 * np.std(integrated_v1)
                    r_peaks_v1, _ = find_peaks(integrated_v1, height=threshold_v1, distance=int(0.15 * fs))
                    
                    # Measure SV1: min(QRS in V1) - TP_baseline_V1 (must be negative, in mV)
                    sv1_amps = []
                    for r_idx in r_peaks_v1[1:-1]:
                        try:
                            # QRS window: ±80ms around R-peak
                            qrs_start = max(0, r_idx - int(0.08 * fs))
                            qrs_end = min(len(lead_v1_data), r_idx + int(0.08 * fs))
                            if qrs_end > qrs_start:
                                # Use RAW data for amplitude measurement
                                qrs_segment = lead_v1_data[qrs_start:qrs_end]
                                
                                # TP baseline: isoelectric segment before P-wave onset
                                tp_start = max(0, r_idx - int(0.35 * fs))
                                tp_end = max(0, r_idx - int(0.15 * fs))
                                if tp_end > tp_start:
                                    tp_segment = lead_v1_data[tp_start:tp_end]
                                    tp_baseline = np.median(tp_segment)  # Median for robustness
                                else:
                                    # Fallback: short segment before QRS
                                    tp_baseline = np.median(lead_v1_data[max(0, qrs_start - int(0.05 * fs)):qrs_start])
                                
                                # SV1 = min(QRS) - TP_baseline (negative, preserve sign, in mV)
                                # Convert from ADC counts to mV using hardware calibration
                                # Adjusted calibration factor to match GE/Philips reference values
                                # Reference: SV1 should be ~-0.485 mV, current gives ~-0.637 mV
                                # Adjustment factor: 0.485/0.637 ≈ 0.76, so multiply calibration by 1.31
                                # Original: 1 mV = 1100 ADC → Adjusted: 1 mV = 1100 * 1.31 ≈ 1441 ADC
                                s_amp_adc = np.min(qrs_segment) - tp_baseline
                                if s_amp_adc < 0:  # SV1 must be negative
                                    # Convert ADC to mV: Adjusted calibration factor for GE/Philips alignment
                                    s_amp_mv = s_amp_adc / 1441.0  # Adjusted ADC to mV conversion (preserve sign)
                                    sv1_amps.append(s_amp_mv)
                        except:
                            continue
                    
                    if len(sv1_amps) > 0:
                        amplitudes['sv1'] = np.median(sv1_amps)  # Median beat approach, negative value
            
            print(f" Wave Amplitudes Calculated: P={amplitudes['p_amp']:.2f}, QRS={amplitudes['qrs_amp']:.2f}, T={amplitudes['t_amp']:.2f}, RV5={amplitudes['rv5']:.2f}, SV1={amplitudes['sv1']:.2f}")
            
            return amplitudes
            
        except Exception as e:
            print(f" Error calculating wave amplitudes: {e}")
            import traceback
            traceback.print_exc()
            return {
                'p_amp': 0.0,
                'qrs_amp': 0.0,
                't_amp': 0.0,
                'rv5': 0.0,
                'sv1': 0.0
            }

    def calculate_pr_interval(self, lead_data):
        """Calculate PR interval from P wave to QRS complex - LIVE"""
        try:
            # Early exit: no real signal 
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.05:
                    return 0
            except Exception:
                return 0
            
            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz)
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 80
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, lead_data)
            
            # Find R-peaks (lenient for 80 Hz)
            peaks, properties = find_peaks(
                filtered_signal,
                height=np.mean(filtered_signal) + 0.3 * np.std(filtered_signal),
                distance=int(0.2 * fs),
                prominence=np.std(filtered_signal) * 0.2
            )
            
            if len(peaks) > 1:
                pr_intervals = []
                deriv = np.gradient(filtered_signal)
                deriv_std = np.std(deriv)
                deriv_thresh = max(0.2 * deriv_std, 1e-6)
                for i in range(min(5, len(peaks)-1)):
                    r_peak = peaks[i]
                    # Search 80-200 ms before R for P upslope (narrowed from 250ms)
                    win_start = max(0, r_peak - int(0.20 * fs))
                    win_end   = max(win_start, r_peak - int(0.08 * fs))
                    if win_end <= win_start:
                        continue
                    win = deriv[win_start:win_end]
                    if len(win) == 0:
                        continue
                    candidates = np.where(win > deriv_thresh)[0]
                    if candidates.size == 0:
                        p_idx = win_start + int(np.argmax(filtered_signal[win_start:win_end]))
                    else:
                        p_idx = win_start + int(candidates[-1])
                    pr_ms = (r_peak - p_idx) / fs * 1000.0
                    if 80 <= pr_ms <= 240:
                        pr_intervals.append(pr_ms)
                if pr_intervals:
                    return int(round(float(np.median(pr_intervals))))
            
            return 150  # Conservative default if not computable
        except:
            return 150

    def calculate_qrs_duration(self, lead_data):
        """Calculate QRS duration — amplitude-threshold onset/offset (GE/Philips standard).

        Key changes vs old derivative method:
          - Bandpass 5-40 Hz  (was 0.5-40 Hz) — rejects slow P/T wave slope
          - Search windows: 60 ms pre-R, 80 ms post-R  (was ±120 ms)
          - Threshold: 30 % of R-peak amplitude  (was 10 % derivative std)
          - Accepts 50-140 ms; returns median of up to 6 beats
        """
        try:
            arr = np.asarray(lead_data, dtype=float)
            if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                return 0

            from scipy.signal import butter, filtfilt, find_peaks

            fs = 186.5
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                fs = float(self.sampler.sampling_rate)
            elif hasattr(self, 'sampling_rate') and self.sampling_rate > 10:
                fs = float(self.sampling_rate)

            # 5-40 Hz bandpass keeps only QRS energy
            nyq = fs / 2.0
            b, a = butter(3, [max(5.0/nyq, 0.01), min(40.0/nyq, 0.99)], btype='band')
            filt = filtfilt(b, a, arr)

            # Pan-Tompkins squared-gradient envelope for R-peak detection
            env = np.convolve(np.square(np.gradient(filt)),
                              np.ones(max(1, int(0.08 * fs))) / max(1, int(0.08 * fs)),
                              mode='same')
            peaks, _ = find_peaks(
                env,
                height=np.mean(env) + 0.4 * np.std(env),
                distance=int(0.35 * fs),
                prominence=np.std(env) * 0.3,
            )

            if len(peaks) == 0:
                return 0

            # CRITICAL FIX: refine envelope peak -> true R-peak in filtered signal
            # Envelope is smoothed; its peaks are time-shifted from the actual R-peak
            # in filt. Search max|filt| within ±25 ms of each envelope peak.
            refine_win = max(1, int(0.025 * fs))  # 25 ms window
            true_r_peaks = []
            for ep in peaks:
                lo = max(0, ep - refine_win)
                hi = min(len(filt) - 1, ep + refine_win)
                r_true = lo + int(np.argmax(np.abs(filt[lo:hi + 1])))
                true_r_peaks.append(r_true)

            pre_win  = int(0.060 * fs)  # 60 ms before R  -> Q-onset
            post_win = int(0.080 * fs)  # 80 ms after  R  -> S-offset

            qrs_durations = []
            for r in true_r_peaks[:min(6, len(true_r_peaks))]:
                r_amp = abs(filt[r])
                if r_amp < 1e-6:
                    continue
                threshold = 0.30 * r_amp   # 30 % of R amplitude

                # Onset: walk backwards until |signal| drops below threshold
                q_idx = r
                start_lim = max(0, r - pre_win)
                for k in range(r - 1, start_lim - 1, -1):
                    if abs(filt[k]) < threshold:
                        q_idx = k + 1
                        break

                # Offset: walk forward until |signal| drops below threshold
                s_idx = r
                end_lim = min(len(filt) - 1, r + post_win)
                for k in range(r + 1, end_lim + 1):
                    if abs(filt[k]) < threshold:
                        s_idx = k - 1
                        break
                else:
                    s_idx = end_lim

                dur_ms = (s_idx - q_idx) / fs * 1000.0
                if 50.0 <= dur_ms <= 140.0:
                    qrs_durations.append(dur_ms)

            if qrs_durations:
                return int(round(float(np.median(qrs_durations))))

            return 0
        except Exception:
            return 0

    def calculate_st_interval(self, lead_data):
        """Calculate ST segment elevation/depression at J+60ms - FRESH calculation"""
        try:
            # Early exit: no real signal → 0
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                    return 0
            except Exception:
                return 0
            
            # Get sampling rate
            from scipy.signal import butter, filtfilt, find_peaks
            # Get sampling rate - Fixed Bug PR-2 (fs=80 hardcoded)
            fs = 500.0  # Default to hardware sampling rate
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                fs = float(self.sampler.sampling_rate)
            elif hasattr(self, 'sampling_rate') and self.sampling_rate > 10:
                fs = float(self.sampling_rate)
            
            # Filter signal (0.5-40 Hz bandpass)
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, lead_data)
            
            # Find R-peaks (lenient for hardware)
            mean_height = np.mean(filtered_signal)
            std_height = np.std(filtered_signal)
            min_height = mean_height + 0.3 * std_height
            min_distance = int(0.3 * fs)
            min_prominence = std_height * 0.2
            
            peaks, _ = find_peaks(
                filtered_signal,
                height=min_height,
                distance=min_distance,
                prominence=min_prominence
            )
            
            if len(peaks) < 1:
                return 0
            
            st_elevations = []
            for r_peak in peaks[:min(5, len(peaks))]:
                try:
                    # Find J-point (end of S-wave, ~40ms after R)
                    j_start = r_peak
                    j_end = min(len(filtered_signal), r_peak + int(0.04 * fs))
                    if j_end <= j_start:
                        continue
                    j_point = j_start + np.argmin(filtered_signal[j_start:j_end])
                    
                    # Measure ST at J+60ms (standard ST measurement point)
                    st_measure_point = min(len(filtered_signal) - 1, j_point + int(0.06 * fs))
                    
                    # Use TP baseline (isoelectric segment 150-350ms before R)
                    tp_baseline_start = max(0, r_peak - int(0.35 * fs))
                    tp_baseline_end = max(0, r_peak - int(0.15 * fs))
                    if tp_baseline_end > tp_baseline_start:
                        tp_baseline = np.median(filtered_signal[tp_baseline_start:tp_baseline_end])
                    else:
                        # Fallback: short segment before QRS
                        baseline_start = max(0, r_peak - int(0.15 * fs))
                        baseline_end = max(0, r_peak - int(0.05 * fs))
                        if baseline_end > baseline_start:
                            tp_baseline = np.median(filtered_signal[baseline_start:baseline_end])
                        else:
                            tp_baseline = np.median(filtered_signal[max(0, r_peak - int(0.05 * fs)):max(0, r_peak - int(0.01 * fs))])
                        tp_baseline = np.mean(filtered_signal[baseline_start:baseline_end]) if baseline_end > baseline_start else np.mean(filtered_signal)
                    
                    # ST deviation in mV (raw ADC difference, needs conversion)
                    # Convert ADC to mV (approximate: assume 10mm/mV gain, typical ADC scaling)
                    # This is a placeholder - actual conversion should use hardware-specific calibration
                    # For standard 10mm/mV: 1 mV ≈ 1000-1500 ADC counts (varies by lead)
                    st_raw_adc = filtered_signal[st_measure_point] - tp_baseline
                    adc_to_mv_factor = 1000.0  # Placeholder - should be hardware-specific
                    st_mv = st_raw_adc / adc_to_mv_factor
                    
                    # Reasonable ST range: -2.0 to +2.0 mV
                    if -2.0 <= st_mv <= 2.0:
                        st_elevations.append(st_mv)
                    else:
                        pass  # Silently reject extreme outliers
                except Exception as e:
                    print(f" ST: Exception in beat analysis: {e}")
                    continue
            
            if st_elevations:
                # Return mean ST deviation in mV (rounded to 2 decimal places)
                st_mean_mv = np.mean(st_elevations)
                return round(st_mean_mv, 2)
            
            return 0.0  # No ST detected
        except:
            return 0

    def calculate_qt_interval(self, lead_data):
        """Calculate QT interval (Q-wave onset to T-wave end) from Lead II"""
        try:
            # Early exit: no real signal
            try:
                arr = np.asarray(lead_data, dtype=float)
                if len(arr) < 200 or np.all(arr == 0) or np.std(arr) < 0.1:
                    return 0
            except Exception:
                return 0
            
            # Get sampling rate
            from scipy.signal import butter, filtfilt, find_peaks
            from ecg.clinical_measurements import detect_t_wave_end_tangent_method
            
            fs = 186.5
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                fs = float(self.sampler.sampling_rate)
            elif hasattr(self, 'sampling_rate') and self.sampling_rate > 10:
                fs = float(self.sampling_rate)
            
            # Filter signal
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, lead_data)
            
            # Find R-peaks
            mean_height = np.mean(filtered_signal)
            std_height = np.std(filtered_signal)
            peaks, _ = find_peaks(
                filtered_signal,
                height=mean_height + 0.3 * std_height,
                distance=int(0.3 * fs),
                prominence=std_height * 0.2
            )
            
            if len(peaks) < 1:
                return 0
            
            qt_intervals = []
            for r_peak in peaks[:min(5, len(peaks))]:
                try:
                    # Find Q-point: min before R, within 50ms (prevent snagging far P-tail)
                    q_start = max(0, r_peak - int(0.05 * fs))
                    q_end = r_peak
                    if q_end > q_start:
                        q_point = q_start + np.argmin(filtered_signal[q_start:q_end])
                    else:
                        q_point = r_peak

                    # Find T-wave search window
                    t_search_start = r_peak + int(0.08 * fs)  # After QRS
                    t_search_end = min(len(filtered_signal), r_peak + int(0.50 * fs))  # 500ms max (was 600ms)
                    
                    if t_search_end > t_search_start:
                        t_segment = filtered_signal[t_search_start:t_search_end]
                        
                        # TP baseline: isoelectric segment before QRS
                        tp_baseline_start = max(0, r_peak - int(0.35 * fs))
                        tp_baseline_end = max(0, r_peak - int(0.15 * fs))
                        if tp_baseline_end > tp_baseline_start:
                            baseline = np.median(filtered_signal[tp_baseline_start:tp_baseline_end])
                        else:
                            baseline = np.mean(filtered_signal[max(0, r_peak - int(0.15 * fs)):max(0, r_peak - int(0.05 * fs))])
                        
                        # Baseline-correct the signal
                        signal_corrected = filtered_signal - baseline
                        t_seg_corr = signal_corrected[t_search_start:t_search_end]
                        
                        # Find T-peak (POSITIVE deflection only to avoid tracking deep ST-depression/S-wave tails)
                        t_peak_idx = t_search_start + int(np.argmax(t_seg_corr))
                        
                        # Use tangent method to find T-end
                        t_end_idx = detect_t_wave_end_tangent_method(
                            signal_corrected, 
                            t_peak_idx, 
                            t_search_end, 
                            fs, 
                            tp_baseline=0.0
                        )
                        
                        if t_end_idx is not None:
                            # Sanity check: T-wave cannot extend more than ~120ms past its peak
                            t_end_capped = min(t_end_idx, t_peak_idx + int(0.12 * fs))
                            qt_ms = (t_end_capped - q_point) / fs * 1000.0
                            if 200 <= qt_ms <= 600:
                                qt_intervals.append(qt_ms)
                        else:
                            # Fallback: T-peak + 80ms
                            t_end = t_peak_idx + int(0.08 * fs)
                            qt_ms = (t_end - q_point) / fs * 1000.0
                            if 200 <= qt_ms <= 600:
                                qt_intervals.append(qt_ms)
                    else:
                        continue
                except Exception as e:
                    print(f" Error calculating QT for beat: {e}")
                    continue
                
            if qt_intervals:
                return int(round(float(np.median(qt_intervals))))
            
            return 0
        except Exception as e:
            print(f" Error in calculate_qt_interval: {e}")
            return 0

    def calculate_qtc_interval(self, heart_rate, qt_interval):
        """Calculate QTc using Bazett's formula: QTc = QT / sqrt(RR)"""
        try:
            if not heart_rate or heart_rate <= 0:
                return 0
            
            if not qt_interval or qt_interval <= 0:
                return 0
            
            # Calculate RR interval from heart rate (in seconds)
            rr_interval = 60.0 / heart_rate
            
            # QT in seconds
            qt_sec = qt_interval / 1000.0
            
            # Apply Bazett's formula: QTc = QT / sqrt(RR)
            qtc = qt_sec / np.sqrt(rr_interval)
            
            # Convert back to milliseconds
            qtc_ms = int(round(qtc * 1000))
            
            return qtc_ms
            
        except Exception as e:
            return 0

    def calculate_qtcf_interval(self, qt_ms, rr_ms):
        """Calculate QTcF using Fridericia formula: QTcF = QT / RR^(1/3)
        
        Args:
            qt_ms: QT interval in milliseconds
            rr_ms: RR interval in milliseconds
        
        Returns:
            QTcF in milliseconds
        """
        try:
            if not qt_ms or qt_ms <= 0 or not rr_ms or rr_ms <= 0:
                return 0
            
            # Convert to seconds
            qt_sec = qt_ms / 1000.0
            rr_sec = rr_ms / 1000.0
            
            # Fridericia formula: QTcF = QT / RR^(1/3)
            qtcf_sec = qt_sec / (rr_sec ** (1.0 / 3.0))
            
            # Convert back to milliseconds
            qtcf_ms = int(round(qtcf_sec * 1000.0))
            
            return qtcf_ms
        except:
            return 0
    
    def calculate_pr_interval_from_median(self, median_beat, time_axis, fs, tp_baseline):
        """Calculate PR interval from median beat: P onset → QRS onset (GE/Philips standard)."""
        try:
            r_idx = np.argmin(np.abs(time_axis))
            
            # Find P onset: first point before R where signal deviates from TP baseline
            p_search_start = max(0, r_idx - int(0.25 * fs))
            p_search_end = max(0, r_idx - int(0.10 * fs))
            if p_search_end <= p_search_start:
                return 150
            
            p_segment = median_beat[p_search_start:p_search_end]
            p_baseline_diff = np.abs(p_segment - tp_baseline)
            signal_range = np.max(median_beat) - np.min(median_beat)
            threshold = max(0.05 * signal_range, np.std(median_beat) * 0.1) if signal_range > 0 else np.std(median_beat) * 0.1
            
            p_deviations = np.where(p_baseline_diff > threshold)[0]
            if len(p_deviations) > 0:
                p_onset_idx = p_search_start + p_deviations[0]
            else:
                p_onset_idx = p_search_start + np.argmax(p_segment)
            
            # Find QRS onset: first point before R where signal deviates from TP baseline
            qrs_search_start = max(0, r_idx - int(0.04 * fs))
            qrs_search_end = r_idx
            if qrs_search_end <= qrs_search_start:
                return 150
            
            qrs_segment = median_beat[qrs_search_start:qrs_search_end]
            qrs_baseline_diff = np.abs(qrs_segment - tp_baseline)
            qrs_deviations = np.where(qrs_baseline_diff > threshold)[0]
            if len(qrs_deviations) > 0:
                qrs_onset_idx = qrs_search_start + qrs_deviations[0]
            else:
                qrs_onset_idx = qrs_search_start + np.argmin(qrs_segment)
            
            # PR = P onset → QRS onset
            pr_ms = time_axis[qrs_onset_idx] - time_axis[p_onset_idx]
            if 80 <= pr_ms <= 240:
                return int(round(pr_ms))
            return 150
        except:
            return 150
    
    def calculate_qrs_duration_from_median(self, median_beat, time_axis, fs, tp_baseline):
        """Calculate QRS duration from median beat: QRS onset → QRS offset (GE/Philips standard)."""
        try:
            r_idx = np.argmin(np.abs(time_axis))
            signal_range = np.max(median_beat) - np.min(median_beat)
            threshold = max(0.05 * signal_range, np.std(median_beat) * 0.1) if signal_range > 0 else np.std(median_beat) * 0.1
            
            # Find QRS onset: first point before R where signal deviates from TP baseline
            qrs_onset_start = max(0, r_idx - int(0.04 * fs))
            qrs_onset_end = r_idx
            if qrs_onset_end <= qrs_onset_start:
                return 80
            
            qrs_onset_segment = median_beat[qrs_onset_start:qrs_onset_end]
            qrs_onset_diff = np.abs(qrs_onset_segment - tp_baseline)
            qrs_onset_deviations = np.where(qrs_onset_diff > threshold)[0]
            if len(qrs_onset_deviations) > 0:
                qrs_onset_idx = qrs_onset_start + qrs_onset_deviations[0]
            else:
                qrs_onset_idx = qrs_onset_start + np.argmin(qrs_onset_segment)
            
            # Find QRS offset: first point after R where signal returns to TP baseline
            qrs_offset_start = r_idx
            qrs_offset_end = min(len(median_beat), r_idx + int(0.12 * fs))
            if qrs_offset_end <= qrs_offset_start:
                return 80
            
            qrs_offset_segment = median_beat[qrs_offset_start:qrs_offset_end]
            qrs_offset_diff = np.abs(qrs_offset_segment - tp_baseline)
            qrs_offset_deviations = np.where(qrs_offset_diff < threshold)[0]
            if len(qrs_offset_deviations) > 0:
                qrs_offset_idx = qrs_offset_start + qrs_offset_deviations[0]
            else:
                # Fallback: use max in QRS segment (end of S-wave)
                qrs_offset_idx = qrs_offset_start + np.argmax(qrs_offset_segment)
            
            # QRS duration = QRS onset → QRS offset
            qrs_ms = time_axis[qrs_offset_idx] - time_axis[qrs_onset_idx]
            if 40 <= qrs_ms <= 200:
                return int(round(qrs_ms))
            return 80
        except:
            return 80
    
    def calculate_qrs_axis_from_median(self):
        """Calculate QRS axis from median beat vectors - wrapper for modular function"""
        try:
            if len(self.data) < 6:
                return getattr(self, '_prev_qrs_axis', 0) or 0
            
            fs = 500.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                fs = float(self.sampler.sampling_rate)
            
            # Detect R-peaks on Lead II (use display filter for R-peak detection)
            from scipy.signal import find_peaks
            from .signal_paths import display_filter
            lead_ii = self.data[1]
            filtered_ii = display_filter(lead_ii, fs)
            signal_mean = np.mean(filtered_ii)
            signal_std = np.std(filtered_ii)
            r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, 
                                   distance=int(0.3 * fs), prominence=signal_std * 0.4)
            
            if len(r_peaks) < 8:
                return getattr(self, '_prev_qrs_axis', 0) or 0
            
            # Use modular function
            axis_deg = calculate_qrs_axis_from_median(self.data, self.leads, r_peaks, fs)
            if axis_deg is not None:
                self._prev_qrs_axis = axis_deg
                return int(round(axis_deg))
            else:
                return getattr(self, '_prev_qrs_axis', 0) or 0
        except Exception as e:
            # OPTIMIZED: Reduced error print frequency for better performance
            if not hasattr(self, '_qrs_axis_error_count'):
                self._qrs_axis_error_count = 0
            self._qrs_axis_error_count += 1
            if self._qrs_axis_error_count % 100 == 1:  # Print every 100th error
                print(f" Error calculating QRS axis from median: {e}")
            return getattr(self, '_prev_qrs_axis', 0) or 0

    def calculate_p_axis_from_median(self):
        """Calculate P-wave axis from median beat vectors - wrapper for modular function"""
        try:
            if len(self.data) < 6:
                return getattr(self, '_prev_p_axis', 0) or 0
            
            fs = 500.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            
            # Detect R-peaks on Lead II (use display filter for R-peak detection)
            from scipy.signal import find_peaks
            from .signal_paths import display_filter
            lead_ii = self.data[1]
            filtered_ii = display_filter(lead_ii, fs)
            signal_mean = np.mean(filtered_ii)
            signal_std = np.std(filtered_ii)
            r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, 
                                   distance=int(0.25 * fs), prominence=signal_std * 0.4)
            
            if len(r_peaks) < 8:
                return getattr(self, '_prev_p_axis', 0) or 0
            
            pr_ms = getattr(self, 'pr_interval', 0)
            axis_deg = calculate_p_axis_from_median(self.data, self.leads, r_peaks, fs, pr_ms=pr_ms)
            if axis_deg is not None:
                self._prev_p_axis = axis_deg
                return int(round(axis_deg))
            else:
                return getattr(self, '_prev_p_axis', 0) or 0
        except Exception as e:
            # OPTIMIZED: Reduced error print frequency for better performance
            if not hasattr(self, '_p_axis_error_count'):
                self._p_axis_error_count = 0
            self._p_axis_error_count += 1
            if self._p_axis_error_count % 100 == 1:  # Print every 100th error
                print(f" Error calculating P axis: {e}")
            return getattr(self, '_prev_p_axis', 0) or 0

    def calculate_t_axis_from_median(self):
        """Calculate T-wave axis from median beat vectors - wrapper for modular function"""
        try:
            if len(self.data) < 6:
                return getattr(self, '_prev_t_axis', 0) or 0
            
            fs = 500.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
            
            # Detect R-peaks on Lead II (use display filter for R-peak detection)
            from scipy.signal import find_peaks
            from .signal_paths import display_filter
            lead_ii = self.data[1]
            filtered_ii = display_filter(lead_ii, fs)
            signal_mean = np.mean(filtered_ii)
            signal_std = np.std(filtered_ii)
            r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, 
                                   distance=int(0.25 * fs), prominence=signal_std * 0.4)
            
            if len(r_peaks) < 8:
                return getattr(self, '_prev_t_axis', 0) or 0
            
            axis_deg = calculate_t_axis_from_median(self.data, self.leads, r_peaks, fs)
            if axis_deg is not None:
                self._prev_t_axis = axis_deg
                return int(round(axis_deg))
            else:
                return getattr(self, '_prev_t_axis', 0) or 0
        except Exception as e:
            # OPTIMIZED: Reduced error print frequency for better performance
            if not hasattr(self, '_t_axis_error_count'):
                self._t_axis_error_count = 0
            self._t_axis_error_count += 1
            if self._t_axis_error_count % 100 == 1:  # Print every 100th error
                print(f" Error calculating T axis: {e}")
            return getattr(self, '_prev_t_axis', 0) or 0
    
    def calculate_rv5_sv1_from_median(self):
        """Calculate RV5 and SV1 from median beats - wrapper for modular function"""
        try:
            if len(self.data) < 11:
                return None, None
            
            fs = 500.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                fs = float(self.sampler.sampling_rate)
                
            # Detect R-peaks on Lead II for alignment
            from scipy.signal import butter, filtfilt
            lead_ii = self.data[1]
            nyquist = fs / 2
            b, a = butter(4, [0.5/nyquist, 40/nyquist], btype='band')
            filtered_ii = filtfilt(b, a, lead_ii)
            signal_mean = np.mean(filtered_ii)
            signal_std = np.std(filtered_ii)
            r_peaks, _ = find_peaks(filtered_ii, height=signal_mean + 0.5 * signal_std, 
                                   distance=int(0.25 * fs), prominence=signal_std * 0.4)
            
            if len(r_peaks) < 8:
                return None, None
                
            # Use modular function
            return calculate_rv5_sv1_from_median(self.data, r_peaks, fs)
        except Exception as e:
            print(f" Error calculating RV5/SV1 from median: {e}")
            return None, None

    def update_ecg_metrics_display(self, heart_rate, pr_interval, qrs_duration, p_duration,
                                   qt_interval=None, qtc_interval=None, qtcf_interval=None,
                                   force_immediate=False, rr_interval=None):
        """Update the ECG metrics display in the UI - wrapper for modular function"""
        if getattr(self, '_grid_frozen', False):
            return

        if not hasattr(self, '_last_metric_update_ts'):
            self._last_metric_update_ts = 0.0

        if force_immediate:
            self._last_metric_update_ts = 0.0

        metric_labels = getattr(self, 'metric_labels', {})
        # Always update UI heart_rate metric label whenever valid heart rate is measured
        self._last_metric_update_ts = update_ecg_metrics_display(
            metric_labels, heart_rate, pr_interval, qrs_duration, p_duration,
            qt_interval, qtc_interval, qtcf_interval, self._last_metric_update_ts,
            rr_interval=rr_interval,
            skip_heart_rate=False, # Ensure heart rate label is updated on UI!
        )

    def get_current_metrics(self):
        """Get current ECG metrics for dashboard display - wrapper for modular function"""
        metric_labels = getattr(self, 'metric_labels', {})
        last_heart_rate = getattr(self, 'last_heart_rate', None)
        sampler = getattr(self, 'sampler', None)
        try:
            if hasattr(self, 'data') and len(self.data) > 1:
                lead_ii = np.asarray(self.data[1], dtype=float)
                if lead_ii.size > 0:
                    lead_std = float(np.std(lead_ii))
                    lead_mean = float(np.mean(np.abs(lead_ii)))
                    is_flat = (
                        lead_ii.size < 100
                        or np.all(lead_ii == 0)
                        or lead_std < 0.1
                        or (lead_mean > 100.0 and lead_std < 50.0)
                    )
                    if is_flat:
                        return {
                            'heart_rate': '0',
                            'rr_interval': '0',
                            'pr_interval': '0',
                            'qrs_duration': '0',
                            'qt_interval': '0',
                            'qtc_interval': '0',
                            'p_duration': '0',
                            'st_interval': '0',
                            'time_elapsed': metric_labels.get('time_elapsed').text() if 'time_elapsed' in metric_labels else '',
                            'sampling_rate': f"{sampler.sampling_rate:.1f}" if sampler and hasattr(sampler, 'sampling_rate') and sampler.sampling_rate > 0 else "--",
                        }
        except Exception:
            pass
        # return get_current_metrics_from_labels(metric_labels, self.data, last_heart_rate, sampler)
        # 1. Try getting from labels first (legacy behavior)
        metrics = get_current_metrics_from_labels(metric_labels, self.data, last_heart_rate, sampler)
        
        # 2. Fallback/Override with internal attributes for headless use (e.g. Hyperkalemia test)
        # Helper to check if a metric is invalid/missing
        def is_invalid(val):
            return val is None or val == '0' or val == '' or val == '--'

        # Heart Rate
        if is_invalid(metrics.get('heart_rate')) and hasattr(self, 'last_heart_rate') and self.last_heart_rate > 0:
            metrics['heart_rate'] = str(int(self.last_heart_rate))
            
        # PR Interval
        if is_invalid(metrics.get('pr_interval')) and hasattr(self, 'pr_interval') and self.pr_interval > 0:
            metrics['pr_interval'] = str(int(round(self.pr_interval)))

        # QRS Duration
        if is_invalid(metrics.get('qrs_duration')) and hasattr(self, 'last_qrs_duration') and self.last_qrs_duration > 0:
            metrics['qrs_duration'] = str(int(round(self.last_qrs_duration)))
            
        # QTc Interval
        if is_invalid(metrics.get('qtc_interval')) and hasattr(self, 'last_qtc_interval') and self.last_qtc_interval > 0:
            metrics['qtc_interval'] = str(int(round(self.last_qtc_interval)))
            
        return metrics

    def update_latest_rhythm_interpretation(self):
        """Compute and cache the live arrhythmia interpretation for all UI surfaces."""
        try:
            from ecg.arrhythmia_detector import analyze_ecg, get_interpretation

            if not hasattr(self, 'data') or len(self.data) < 2:
                self._latest_rhythm_interpretation = "Analyzing Rhythm..."
                return self._latest_rhythm_interpretation

            fs = 500.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate:
                try:
                    fs = float(self.sampler.sampling_rate)
                except Exception:
                    fs = 500.0

            # Compute the exact same arrhythmia label text as ExpandedLeadView does,
            # without requiring the user to open the expanded view.
            try:
                lead_ii = self.data[1] if len(self.data) > 1 else None
            except Exception:
                lead_ii = None

            if lead_ii is None:
                self._latest_rhythm_interpretation = "Analyzing Rhythm..."
                return self._latest_rhythm_interpretation

            # Expanded view effectively operates on the visible window; use a recent slice.
            try:
                window_sec = float(getattr(self, "_rhythm_window_sec", 10.0) or 10.0)
            except Exception:
                window_sec = 10.0
            window_n = int(max(fs * 2.5, min(fs * window_sec, len(lead_ii))))

            try:
                sig = np.asarray(lead_ii[-window_n:], dtype=float)
            except Exception:
                sig = np.asarray([], dtype=float)

            if sig.size < int(fs * 2.0):
                self._latest_rhythm_interpretation = "Analyzing Rhythm..."
                return self._latest_rhythm_interpretation

            # Same bandpass as ExpandedLeadView._apply_display_bandpass().
            try:
                from scipy.signal import butter, filtfilt
                nyq = 0.5 * float(fs)
                low_n = max(0.05 / nyq, 1e-5)
                high_n = min(40.0 / nyq, 0.999)
                b, a = butter(2, [low_n, high_n], btype="bandpass")
                sig = filtfilt(b, a, sig)
            except Exception:
                pass

            lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
            lead_signals = {}
            for idx, lead_name in enumerate(lead_names):
                try:
                    if idx < len(self.data) and len(self.data[idx]) > 0:
                        lead_arr = np.asarray(self.data[idx], dtype=float)
                        lead_signals[lead_name] = lead_arr[-window_n:] if lead_arr.size > window_n else lead_arr
                except Exception:
                    continue

            if "II" not in lead_signals or lead_signals["II"].size < int(fs * 2.0):
                self._latest_rhythm_interpretation = "Analyzing Rhythm..."
                return self._latest_rhythm_interpretation

            hr_val = getattr(self, 'last_heart_rate', 0) or 0
            pr_val = getattr(self, 'pr_interval', 0) or 0
            qrs_val = getattr(self, 'last_qrs_duration', 0) or 0
            qtc_val = getattr(self, 'last_qtc_interval', 0) or 0
            
            external_metrics = {
                'external_hr': hr_val,
                'external_pr': pr_val,
                'external_qrs': qrs_val
            }

            gender = "M"
            try:
                username = ""
                if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                    username = getattr(self.dashboard_instance, 'username', '') or ''
                patient = resolve_patient_profile(
                    explicit_patient=getattr(getattr(self, 'dashboard_instance', None), 'patient_details', None),
                    username=username,
                    user_details=getattr(getattr(self, 'dashboard_instance', None), 'user_details', {}) or {},
                )
                gender = str(
                    patient.get('gender')
                    or patient.get('sex')
                    or patient.get('Gender')
                    or "M"
                ).strip() or "M"
            except Exception:
                gender = "M"

            results = analyze_ecg(lead_signals, fs=fs, patient_gender=gender, external_metrics=external_metrics)
            arrhythmias = results.get("arrhythmias", [])
            summary_lines = [line for line in get_interpretation(results) if line]
            
            if not hasattr(self, '_last_analysis') or self._last_analysis is None:
                self._last_analysis = {}
            self._last_analysis['arrhythmias'] = arrhythmias

            bpm_label = _bpm_rhythm_label(hr_val)
            significant_keywords = (
                "Atrial Fibrillation",
                "Atrial Flutter",
                "Ventricular Tachycardia",
                "Ventricular Fibrillation",
                "Asystole",
                "Bundle Branch",
                "Pre-excitation",
                "AV Block",
                "Heart Block",
            )
            has_significant_rhythm = any(
                any(keyword.lower() in str(d).lower() for keyword in significant_keywords)
                for d in arrhythmias
            )

            # Ensure the UI string reflects clinical findings
            bbb_dx = [d for d in arrhythmias if any(
                kw in d for kw in ('Bundle Branch', 'Wide QRS', 'Pre-excitation'))]
            if bpm_label and not has_significant_rhythm:
                self._latest_rhythm_interpretation = bpm_label
            elif bbb_dx:
                clean_lines = [line for line in summary_lines if 'Wide QRS' not in line]
                clean_lines.extend(bbb_dx)
                self._latest_rhythm_interpretation = ", ".join(clean_lines)
            else:
                self._latest_rhythm_interpretation = ", ".join(summary_lines) if summary_lines else "No specific arrhythmia detected."

            return self._latest_rhythm_interpretation
        except Exception as e:
            print(f" Error updating live rhythm interpretation: {e}")
            return getattr(self, '_latest_rhythm_interpretation', "Analyzing Rhythm...")

    def get_latest_rhythm_interpretation(self):
        """Expose latest arrhythmia interpretation string for the dashboard."""
        return getattr(self, '_latest_rhythm_interpretation', "Analyzing Rhythm...")

    def update_plot_y_range(self, plot_index):
        """Update Y-axis range for a specific plot using robust stats to avoid cropping"""
        try:
            if plot_index >= len(self.data) or plot_index >= len(self.plot_widgets):
                return

            # Get the data for this plot and apply the current display gain
            data = self.data[plot_index]
            gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
            data = np.asarray(data) * gain_factor
            
            # Remove NaN values and large outliers (robust)
            valid_data = data[~np.isnan(data)]
            
            if len(valid_data) == 0:
                return
            
            # Use percentiles to avoid spikes from clipping the view
            p1 = np.percentile(valid_data, 1)
            p99 = np.percentile(valid_data, 99)
            data_mean = (p1 + p99) / 2.0
            data_std = np.std(valid_data[(valid_data >= p1) & (valid_data <= p99)])
            # Maximum deviation of any point from the mean – we will always cover this
            peak_deviation = np.max(np.abs(valid_data - data_mean))
            
            # Calculate appropriate Y-range with some padding
            if data_std > 0:
                # Use standard deviation within central band
                base_padding = max(data_std * 4, 200)  # Increased padding for better visibility
                padding = base_padding  # Do NOT scale padding with gain; gain already applied to signal
                y_min = data_mean - padding
                y_max = data_mean + padding
                print(f" Basic Y-range: base_padding={base_padding:.1f}, padding={padding:.1f}")
            else:
                # Fallback: use percentile window
                data_range = max(p99 - p1, 300)
                base_padding = max(data_range * 0.3, 200)
                padding = base_padding  # Do NOT scale padding with gain; gain already applied to signal
                y_min = data_mean - padding
                y_max = data_mean + padding
                print(f" Basic Y-range (fallback): base_padding={base_padding:.1f}, padding={padding:.1f}")
            
            # Clinical 12-lead ECG: symmetric Y-axis centred at 0 for ALL leads.
            # aVR's negative polarity comes from the formula (-(I+II)/2), not a flipped axis.
            y_min, y_max = -2048, 2048
            
            # Update the plot's Y-range (fixed, no auto-scaling)
            self.plot_widgets[plot_index].setYRange(y_min, y_max, padding=0)
            
        except Exception as e:
            print(f"Error updating Y-range for plot {plot_index}: {e}")

    def on_settings_changed(self, key, value):
        
        print(f"Setting changed: {key} = {value}")
        
        if key in ["wave_speed", "wave_gain"]:
            # Apply new settings immediately
            self.apply_display_settings()
            
            # CRITICAL: Update all lead titles IMMEDIATELY
            self.update_all_lead_titles()
            
            # Force redraw of all plots
            self.redraw_all_plots()
            
            # Notify demo manager for instant updates (like divyansh.py)
            if hasattr(self, 'demo_manager') and self.demo_manager:
                self.demo_manager.on_settings_changed(key, value)
            
            print(f"Settings applied and titles updated for {key} = {value}")
        elif key == "system_language":
            self.apply_language(value)

    def update_all_lead_titles(self):
        """Update all lead titles with current speed and gain settings"""
        # Safety check: only update if plots are initialized
        if not hasattr(self, 'axs') or not self.axs:
            print(" Plots not initialized yet, skipping title update")
            return
            
        current_speed = 25.0 if getattr(self, '_overlay_active', False) else self.settings_manager.get_wave_speed()
        current_gain = self.settings_manager.get_wave_gain()
        
        print(f"Updating titles: Speed={current_speed}mm/s, Gain={current_gain}mm/mV")
        
        for i, lead in enumerate(self.leads):
            if i < len(self.axs):
                new_title = f"{lead} | Speed: {current_speed}mm/s | Gain: {current_gain}mm/mV"
                self.axs[i].set_title(new_title, fontsize=8, color='#666', pad=10)
                print(f"Updated {lead} title: {new_title}")
        
        # Force redraw of all canvases
        for canvas in self.canvases:
            if canvas:
                canvas.draw_idle()

    def apply_display_settings(self):
        
        wave_speed = 25.0 if getattr(self, '_overlay_active', False) else self.settings_manager.get_wave_speed()
        wave_gain = self.settings_manager.get_wave_gain()
        
        # Higher speed = more samples per second = larger buffer for same time window
        base_buffer = getattr(self, "base_buffer_size", 2000)
        speed_factor = wave_speed / 50.0  # 50mm/s is baseline
        self.buffer_size = int(base_buffer * speed_factor)
        
        # Update y-axis limits based on gain.
        # Higher mm/mV = higher gain = larger waves = need more Y-axis range
        # Use clinical standard helper function (10mm/mV = 1.0x baseline)
        base_ylim = 400
        gain_factor = get_display_gain(wave_gain)
        # Scale Y-axis range with gain (NO CLAMPING - allow all gain values)
        self.ylim = int(base_ylim * gain_factor)

        # Force immediate redraw of all plots with new settings
        self.redraw_all_plots()
        
        print(f"Applied settings: speed={wave_speed}mm/s, gain={wave_gain}mm/mV, buffer={self.buffer_size}, ylim={self.ylim}")

    # ------------------------ Update Dashboard Metrics on the top of the lead graphs ------------------------
    
    def create_metrics_frame(self):
        metrics_frame = QFrame()
        metrics_frame.setObjectName("metrics_frame")
        metrics_frame.setStyleSheet("""
            QFrame {
                background: #000000;
                border: none;
                border-radius: 6px;
                padding: 0;  /* Reduced from 4px */
                margin: 0;  /* Reduced from 2px */
            }
        """)
        
        metrics_layout = QHBoxLayout(metrics_frame)
        metrics_layout.setSpacing(6)  # Reduced from 10px
        metrics_layout.setContentsMargins(6, 6, 6, 6)  # Reduced from 10px
        
        # Store metric labels for live update
        self.metric_labels = {}
        
        # Updated metric info — RR and P Duration removed from display per user request
        metric_info = [
            ("PR",     "  0",  "pr_interval",  "#ffffff"),
            ("QRS",    "  0",  "qrs_duration", "#ffffff"),
            ("QT/QTc", "0",    "qtc_interval", "#ffffff"),
            ("Time",   "00:00","time_elapsed", "#ffffff"),
        ]
        
        for title, value, key, color in metric_info:
            metric_widget = QWidget()
            # FIX-D3: QTc shows "QT/QTc" format (e.g. "380/420") — needs wider box.
            # RR shows up to 4 digits (e.g. "1200"), P shows 2-3 digits.
            if key in ("time_elapsed", "qtc_interval"):
                min_w = "150px"
            elif key in ("rr_interval",):
                min_w = "100px"
            else:
                min_w = "90px"
            metric_widget.setStyleSheet(f"""
                QWidget {{
                    background: transparent;
                    min-width: {min_w};
                    border-right: none;
                }}
            """)

            metric_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            
            # Create vertical layout for the metric widget
            box = QVBoxLayout(metric_widget)
            box.setSpacing(2)  # Reduced from 3px
            box.setAlignment(Qt.AlignCenter)
            
            # Title label with consistent color coding - Make it smaller
            lbl = QLabel(title)
            lbl.setFont(QFont("Arial", 9, QFont.Bold))
            lbl.setStyleSheet(f"color: #000000; margin-bottom: 2px; font-weight: bold;")
            lbl.setAlignment(Qt.AlignCenter)
            
            # Value label with specific colors - Make it smaller
            # Fixed-width initial values prevent layout shift when numbers appear
            if key == "pr_interval":
                fixed_value = "  0"   # 3 chars (e.g. "160")
            elif key == "qrs_duration":
                fixed_value = "  0"   # 3 chars (e.g. " 85")
            elif key in ("rr_interval", "p_duration"):
                fixed_value = "--"    # placeholder until first measurement
            else:
                fixed_value = value
            val = QLabel(fixed_value)
            val.setFont(QFont("Arial", 32, QFont.Bold))
            val.setStyleSheet(f"color: #000000; background: transparent; padding: 0px;")
            val.setAlignment(Qt.AlignCenter)
            
            # Add labels to the metric widget's layout
            box.addWidget(lbl)
            box.addWidget(val)
            
            # Add the metric widget to the horizontal layout
            metrics_layout.addWidget(metric_widget)
            
            # Store reference for live update
            self.metric_labels[key] = val
        
        # Heart rate metric (no emoji, red color)
        heart_rate_widget = QWidget()
        heart_rate_widget.setStyleSheet("""
            QWidget {
                background: transparent;
                min-width: 90px;
                border-right: none;
            }
        """)
        heart_rate_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        heart_box = QVBoxLayout(heart_rate_widget)
        heart_box.setSpacing(2)
        heart_box.setAlignment(Qt.AlignCenter)
        
        # Heart rate title
        hr_title = QLabel("BPM")
        hr_title.setFont(QFont("Arial", 9, QFont.Bold))
        hr_title.setStyleSheet("color: #ff0000; margin-bottom: 2px; font-weight: bold;")
        hr_title.setAlignment(Qt.AlignCenter)
        
        # Heart rate value (red color) - use fixed-width initial value
        heart_rate_val = QLabel("  0")
        heart_rate_val.setFont(QFont("Arial", 32, QFont.Bold))
        heart_rate_val.setStyleSheet("color: #ff0000; background: transparent; padding: 0px;")
        heart_rate_val.setAlignment(Qt.AlignCenter)
        
        heart_box.addWidget(hr_title)
        heart_box.addWidget(heart_rate_val)
        
        # Insert heart rate widget at the beginning
        metrics_layout.insertWidget(0, heart_rate_widget)
        self.metric_labels['heart_rate'] = heart_rate_val
        
        # --- ADD HOLTER STATUS BADGE ---
        self.holter_badge = QLabel(" COMPREHENSIVE ECG ANALYSIS ")
        self.holter_badge.setStyleSheet("""
            QLabel {
                background: #00FF00;
                color: black;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
                font-size: 14px;
                border: 2px solid black;
            }
        """)
        self.holter_badge.setVisible(False)
        metrics_layout.addWidget(self.holter_badge)
        # -------------------------------

        # Reset all metrics to zero after creating the frame
        self.reset_metrics_to_zero()
        
        return metrics_frame

    def _set_lead_status_idle(self, text: str = "Leads: Waiting for device..."):
        """Show a neutral state before any real packets arrive."""
        try:
            if not hasattr(self, 'lead_status_label') or self.lead_status_label is None:
                return
            self._last_off_leads_signature = None
            self.lead_status_label.setText(text)
            self.lead_status_label.setStyleSheet(
                "color: #c77700; font-size: 12px; font-weight: bold; padding: 2px 6px;"
            )
        except Exception:
            pass

    def _set_lead_status_label(self, off_leads: List[str]):
        """Show lead-off status in top label (never use popup for lead-off)."""
        try:
            if not hasattr(self, 'lead_status_label') or self.lead_status_label is None:
                return
            off_leads = list(off_leads or [])
            # Only reset metrics to zero if both primary limb leads (I and II) are off
            limb_off = [l for l in ('I', 'II') if l in off_leads]
            if len(limb_off) == 2 or getattr(self, "_lead_off_latched", False):
                self.reset_metrics_to_zero()

            signature = tuple(off_leads)
            if signature == self._last_off_leads_signature:
                return
            self._last_off_leads_signature = signature
            if not off_leads:
                self.lead_status_label.setText("Leads: All connected")
                self.lead_status_label.setStyleSheet(
                    "color: #2e7d32; font-size: 12px; font-weight: bold; padding: 2px 6px;"
                )
            elif len(off_leads) >= len(self.leads):
                self.lead_status_label.setText("No ECG signal detected. Check patient leads.")
                self.lead_status_label.setStyleSheet(
                    "color: #c62828; font-size: 12px; font-weight: bold; padding: 2px 6px;"
                )
            elif len(off_leads) == 1:
                self.lead_status_label.setText(f"Lead Off: {off_leads[0]}")
                self.lead_status_label.setStyleSheet(
                    "color: #c62828; font-size: 12px; font-weight: bold; padding: 2px 6px;"
                )
            else:
                joined = ", ".join(off_leads)
                self.lead_status_label.setText(f"Leads Off ({len(off_leads)}): {joined}")
                self.lead_status_label.setStyleSheet(
                    "color: #c62828; font-size: 12px; font-weight: bold; padding: 2px 6px;"
                )
        except Exception:
            pass

    def _on_lead_reconnected(self, lead_index: int, lead_name: str, value: float):
        """Reset per-lead state when a previously disconnected lead reconnects (preserves rolling ECG buffer)."""
        try:
            if hasattr(self, '_baseline_anchors') and lead_index < len(self._baseline_anchors):
                self._baseline_anchors[lead_index] = float(value)
            if hasattr(self, '_sweep_cursor_state') and isinstance(self._sweep_cursor_state, dict):
                self._sweep_cursor_state.pop(('serial', lead_index), None)
                self._sweep_cursor_state.pop(('live', lead_index), None)
            if hasattr(self, '_y_range_update_count') and isinstance(self._y_range_update_count, dict):
                self._y_range_update_count[lead_index] = 0
            if lead_index < len(self._flatline_alert_shown):
                self._flatline_alert_shown[lead_index] = False
            if lead_name in ('I', 'II'):
                self._lead_off_latched = False
                self._lead_off_latch_on_count = 0
                self._lead_off_latch_off_count = 0
                self._ra_rl_popup_shown = False
                if lead_name == 'II':
                    self._ll_disconnected = False
                    self._ll_report_cooldown_armed = False
                if lead_name == 'I':
                    self._la_report_cooldown_armed = False
        except Exception:
            pass

    def update_ecg_metrics_on_top_of_lead_graphs(self, intervals):
        if getattr(self, '_report_generating', False) or getattr(self, '_grid_frozen', False):
            return

        limb_conn = getattr(self, "_lead_connection_state", {})
        limb_active = limb_conn.get('I', True) or limb_conn.get('II', True)
        if getattr(self, "_lead_off_latched", False) or not limb_active or not intervals:
            self.reset_metrics_to_zero()
            return

        def _metric_to_int(value):
            if value in (None, "", "--", "--/--", "0", "0/0"):
                return 0
            try:
                val = int(round(float(value)))
                return val if val > 0 else 0
            except Exception:
                return 0

        hr_val = _metric_to_int(intervals.get('Heart_Rate'))
        if hr_val > 0 and hasattr(self, 'metric_labels') and 'heart_rate' in self.metric_labels:
            self.metric_labels['heart_rate'].setText(f"{hr_val:3d}")

        if hasattr(self, 'metric_labels'):
            if 'pr_interval' in self.metric_labels:
                pr_val = _metric_to_int(intervals.get('PR'))
                if pr_val > 0:
                    self.metric_labels['pr_interval'].setText(f"{pr_val:3d}")

            if 'qrs_duration' in self.metric_labels:
                qrs_val = _metric_to_int(intervals.get('QRS'))
                if qrs_val > 0:
                    self.metric_labels['qrs_duration'].setText(f"{qrs_val:2d}")

            if 'qtc_interval' in self.metric_labels:
                qt_val = _metric_to_int(intervals.get('QT'))
                qtc_val = _metric_to_int(intervals.get('QTc'))
                if qt_val > 0 and qtc_val > 0:
                    self.metric_labels['qtc_interval'].setText(f"{qt_val}/{qtc_val}")
                elif qtc_val > 0:
                    self.metric_labels['qtc_interval'].setText(f"{qtc_val}")

    def update_metrics_frame_theme(self, dark_mode=False, medical_mode=False):
       
        if not hasattr(self, 'metrics_frame'):
            return
            
        if dark_mode:
            # Dark mode styling
            self.metrics_frame.setStyleSheet("""
                QFrame#metrics_frame {
                    background: #000000;
                    border: none;
                    border-radius: 6px;
                    padding: 0px;
                    margin: 0px 0;
                    /* Removed unsupported box-shadow property */
                }
            """)
            
            # Update text colors for dark mode
            for key, label in self.metric_labels.items():
                if key == 'heart_rate':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 0; border: none; margin: 0; font-size: 50px;")
                elif key == 'pr_interval':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_duration':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'time_elapsed':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 45px; min-width: 140px;")
                elif key == 'qtc_interval':
                    label.setStyleSheet("color: #ffffff; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
            
            # Update title colors to green for dark mode
            for child in self.metrics_frame.findChildren(QLabel):
                if child != self.metric_labels.get('heart_rate') and child != self.metric_labels.get('time_elapsed') and child != self.metric_labels.get('qtc_interval'):
                    if not any(child == label for label in self.metric_labels.values()):
                        child.setStyleSheet("color: #00ff00; margin-bottom: 5px; border: none;")
                        
        elif medical_mode:
            # Medical mode styling (green theme)
            self.metrics_frame.setStyleSheet("""
                QFrame#metrics_frame {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #f0fff0, stop:1 #e0f0e0);
                    border: none;
                    border-radius: 6px;
                    padding: 0;
                    margin: 0;
                    /* Removed unsupported box-shadow property */
                }
            """)
            
            # Update text colors for medical mode
            for key, label in self.metric_labels.items():
                if key == 'heart_rate':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 0; border: none; margin: 0; font-size: 50px;")
                elif key == 'pr_interval':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_duration':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'time_elapsed':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 45px; min-width: 140px;")
                elif key == 'qtc_interval':
                    label.setStyleSheet("color: #2e7d32; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
            
            # Update title colors to dark green for medical mode
            for child in self.metrics_frame.findChildren(QLabel):
                if child != self.metric_labels.get('heart_rate') and child != self.metric_labels.get('time_elapsed') and child != self.metric_labels.get('qtc_interval'):
                    if not any(child == label for label in self.metric_labels.values()):
                        child.setStyleSheet("color: #2e7d32; margin-bottom: 5px; border: none;")
                        
        else:
            # Light mode (default) styling
            self.metrics_frame.setStyleSheet("""
                QFrame#metrics_frame {
                    background: #ffffff;
                    border: none;
                    border-radius: 6px;
                    padding: 0;
                    margin: 0;
                    /* Removed unsupported box-shadow property */
                }
            """)
            
            # Update text colors for light mode
            for key, label in self.metric_labels.items():
                if key == 'heart_rate':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 0; border: none; margin: 0; font-size: 50px;")
                elif key == 'pr_interval':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'qrs_duration':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'st_interval':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
                elif key == 'time_elapsed':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 45px; min-width: 140px;")
                elif key == 'qtc_interval':
                    label.setStyleSheet("color: #000000; background: transparent; padding: 4px 0px; border: none; font-size: 50px;")
            
            # Update title colors to dark gray for light mode
            for child in self.metrics_frame.findChildren(QLabel):
                if child != self.metric_labels.get('heart_rate') and child != self.metric_labels.get('time_elapsed') and child != self.metric_labels.get('qtc_interval'):
                    if not any(child == label for label in self.metric_labels.values()):
                        child.setStyleSheet("color: #666; margin-bottom: 5px; border: none;")

    def update_elapsed_time(self):
        """Update elapsed time display - only when acquisition or demo mode is active"""
        try:
            # Check if acquisition is running OR demo mode is active
            is_acquisition_running = (hasattr(self, 'serial_reader') and 
                                    self.serial_reader and 
                                    self.serial_reader.running)
            is_demo_mode = (hasattr(self, 'demo_toggle') and 
                          self.demo_toggle and 
                          self.demo_toggle.isChecked())
            
            # Only update time when acquisition is running OR demo mode is active
            if not is_acquisition_running and not is_demo_mode:
                return
            
            if self.start_time and 'time_elapsed' in self.metric_labels:
                current_time = time.time()
                # Subtract paused duration from elapsed time
                paused_duration = getattr(self, 'paused_duration', 0)
                elapsed = max(0, current_time - self.start_time - paused_duration)
                
                # Calculate hours, minutes and seconds
                total_seconds = int(elapsed)
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                
                # Store last displayed time to prevent skipping/duplicate updates
                if not hasattr(self, '_last_displayed_elapsed'):
                    self._last_displayed_elapsed = -1
                
                # Only update if time actually changed (prevents unnecessary UI updates)
                # Update every second (rounded down to prevent flicker)
                current_elapsed_int = total_seconds
                if current_elapsed_int != self._last_displayed_elapsed:
                    # FREEZE GUARD: keep counting internally but do not repaint
                    # the Time label while the display is frozen.
                    if not getattr(self, '_grid_frozen', False):
                        if hours > 0:
                            self.metric_labels['time_elapsed'].setText(f"{hours}:{minutes:02d}:{seconds:02d}")
                        else:
                            self.metric_labels['time_elapsed'].setText(f"{minutes:02d}:{seconds:02d}")
                    self._last_displayed_elapsed = current_elapsed_int

                    # Keep Generate Report button state & countdown label in sync with top metrics timer
                    if current_elapsed_int <= 11 and hasattr(self, '_update_generate_report_button_state'):
                        self._update_generate_report_button_state()
        except Exception as e:
            print(f" Error updating elapsed time: {e}")

    def reset_metrics_to_zero(self, force=False):
        """Reset all ECG metric labels to zero/initial state when limb leads are disconnected."""
        if not force:
            limb_conn = getattr(self, "_lead_connection_state", {})
            lead_off_latched = getattr(self, "_lead_off_latched", False)
            limb_active = (limb_conn.get('I', True) or limb_conn.get('II', True)) and not lead_off_latched
            if limb_active and hasattr(self, 'last_heart_rate') and getattr(self, 'last_heart_rate', 0) > 0:
                return
        try:
            if hasattr(self, 'metric_labels') and isinstance(self.metric_labels, dict):
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText("0")
                if 'rr_interval' in self.metric_labels:
                    self.metric_labels['rr_interval'].setText("0")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText("0")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText("0")
                if 'p_duration' in self.metric_labels:
                    self.metric_labels['p_duration'].setText("0")
                if 'st_interval' in self.metric_labels:
                    self.metric_labels['st_interval'].setText("0")
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("0/0")
                if 'time_elapsed' in self.metric_labels:
                    self.metric_labels['time_elapsed'].setText("00:00")
            
            # Reset numerical metric state
            self.last_heart_rate = 0
            self.last_rr_interval = 0
            self.pr_interval = 0
            self.last_qrs_duration = 0
            self.last_qt_interval = 0
            self.last_qtc_interval = 0
            self.last_p_duration = 0
            self._latest_rhythm_interpretation = "Analyzing Rhythm..."

            try:
                from .ui import display_updates as _du
                _du._last_valid.clear()
            except Exception:
                pass
        except Exception:
            pass

    def showEvent(self, event):
        """Called when the ECG test page is shown - reset metrics to zero"""
        super().showEvent(event)

        # Check if currently running to prevent flicker when returning to the page
        is_acquisition_running = hasattr(self, 'serial_reader') and self.serial_reader and self.serial_reader.running
        is_timer_active = hasattr(self, 'timer') and self.timer and self.timer.isActive()
        
        if is_acquisition_running or is_timer_active:
            # Also force an immediate timer label update so it doesn't wait a full second to refresh if it was paused
            self._last_displayed_elapsed = -1 
            self.update_elapsed_time()
            return  # Do not overwrite metrics while active (prevents "00:00" flicker)

        # Check if demo mode is active
        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            # Demo mode is active - set fixed demo values instead of resetting to zero
            print(" Page shown with demo mode active - setting fixed demo values")
            if hasattr(self, 'metric_labels'):
                # Use fixed-width formatting to prevent text shifting
                self.metric_labels.get('heart_rate', QLabel()).setText(" 60")
                self.metric_labels.get('pr_interval', QLabel()).setText("167")
                self.metric_labels.get('qrs_duration', QLabel()).setText("86")
                self.metric_labels.get('st_interval', QLabel()).setText("92")  # P duration
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("357/357")
        else:
            # Demo mode is not active - reset metrics to zero
            self.reset_metrics_to_zero()

    # ------------------------ Calculate ECG Intervals ------------------------

    def calculate_ecg_intervals(self, lead_ii_data):
        if not lead_ii_data or len(lead_ii_data) < 100 or (isinstance(lead_ii_data, (list, np.ndarray)) and np.std(lead_ii_data) < 5):
            # Fallback to Lead I if Lead II is flat or disconnected
            if hasattr(self, 'data') and len(self.data) > 0 and len(self.data[0]) >= 100 and np.std(self.data[0]) >= 5:
                lead_ii_data = self.data[0]
            else:
                return {}
        
        try:
            from ecg.pan_tompkins import pan_tompkins
            
            # Convert to numpy array
            data = np.array(lead_ii_data)
            
            # Detect R peaks using Pan-Tompkins algorithm
            # Use detected sampling rate - Fixed Bug QRS-2 (fs=186.5 hardcoded)
            fs_report = 500.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                fs_report = float(self.sampler.sampling_rate)

            # Detect R peaks using Pan-Tompkins algorithm
            r_peaks = pan_tompkins(data, fs=fs_report)
            
            if len(r_peaks) < 2:
                return {
                    'Heart_Rate': '--',
                    'PR': '--',
                    'QRS': '--',
                    'QT': '--',
                    'QTc': '--',
                }
            
            # Calculate heart rate
            rr_intervals = np.diff(r_peaks) / fs_report  # Convert to seconds
            mean_rr = np.mean(rr_intervals)
            heart_rate = 60 / mean_rr if mean_rr > 0 else 0
            
            # Use current valid metrics from labels if available (preventing legacy conflict)
            # This ensures consistency with the dashboard and main view
            from .ui.display_updates import get_current_metrics_from_labels
            current_metrics = get_current_metrics_from_labels(getattr(self, 'metric_labels', {}))
            
            pr_val = current_metrics.get('pr_interval')
            qrs_val = current_metrics.get('qrs_duration')
            qt_val = current_metrics.get('qt_interval')
            qtc_val = current_metrics.get('qtc_interval')
            p_val = current_metrics.get('p_duration')
            
            # Fallback to calculated if global metrics unavailable (e.g. offline analysis)
            # Note: storing milliseconds directly as expected by caller
            return {
                'Heart_Rate': heart_rate,
                'PR': pr_val if pr_val is not None else "--",
                'QRS': qrs_val if qrs_val is not None else "--",
                'QT': qt_val if qt_val is not None else "--",
                'QTc': qtc_val if qtc_val is not None else "--"
            }
                
        except Exception as e:
            print(f"Error calculating ECG intervals: {e}")
            return {}

    # ------------------------ Show help dialog ------------------------

    def show_help(self):
        help_text = """
        <h3>12-Lead CardioX Help</h3>
        <p><b>Getting Started:</b></p>
        <ul>
        <li>Configure serial port and baud rate in System Setup</li>
        <li>Click 'Start' to begin recording</li>
        <li>Click on any lead to view it in detail</li>
        <li>Use the menu options for additional features</li>
        </ul>
        <p><b>Features:</b></p>
        <ul>
        <li>Real-time 12-lead ECG monitoring</li>
        <li>Export data as PDF or CSV</li>
        <li>Detailed lead analysis</li>
        <li>Arrhythmia detection</li>
        </ul>
        """
        msg = QMessageBox(self)
        msg.setWindowTitle("Help - 12-Lead CardioX")
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    # ------------------------ Ports Configuration Dialog ------------------------

    def show_ports_dialog(self):
        """Show simple dialog for configuring COM port and baud rate"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Port Configuration")
        dialog.setFixedSize(300, 200)
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        
        # Port selection
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("COM Port:"))
        port_combo = QComboBox()
        port_combo.addItem("Select Port")
        
        # Get available ports
        try:
            ports = serial.tools.list_ports.comports()
            if ports:
                for port in ports:
                    device = getattr(port, 'device', '')
                    desc = getattr(port, 'description', '')
                    hwid = getattr(port, 'hwid', '')
                    label = device
                    extra = " - ".join([s for s in [desc, hwid] if s])
                    if extra:
                        label = f"{device} - {extra}"
                    port_combo.addItem(label, device)
            else:
                port_combo.addItem("No ports found")
        except Exception as e:
            print(f"Error listing ports: {e}")
            port_combo.addItem("Error detecting ports")
        
        # Set current port if available
        current_port = self.settings_manager.get_serial_port()
        if current_port and current_port != "Select Port":
            found_idx = -1
            for i in range(port_combo.count()):
                try:
                    if port_combo.itemData(i) == current_port:
                        found_idx = i
                        break
                except Exception:
                    pass
            if found_idx >= 0:
                port_combo.setCurrentIndex(found_idx)
        
        port_layout.addWidget(port_combo)
        layout.addLayout(port_layout)
        
        # Baud rate selection
        baud_layout = QHBoxLayout()
        baud_layout.addWidget(QLabel("Baud Rate:"))
        baud_combo = QComboBox()
        baud_rates = ["9600", "19200", "38400", "57600", "115200", "230400", "460800", "921600"]
        baud_combo.addItems(baud_rates)
        
        # Set current baud rate if available
        current_baud = self.settings_manager.get_baud_rate()
        if current_baud:
                baud_combo.setCurrentText(current_baud)
        
        baud_layout.addWidget(baud_combo)
        layout.addLayout(baud_layout)
        
        # Refresh ports button
        refresh_btn = QPushButton("🔄 Refresh Ports")
        refresh_btn.clicked.connect(lambda: self.refresh_port_combo(port_combo))
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #ff6600;
                color: white;
                border: 2px solid #ff6600;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #e55a00;
            }
        """)
        layout.addWidget(refresh_btn)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #6c757d;
                color: white;
                border: 2px solid #6c757d;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #5a6268;
            }
        """)
        button_layout.addWidget(cancel_btn)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(lambda: self._save_port_settings(dialog, port_combo, baud_combo))
        save_btn.setStyleSheet("""
            QPushButton {
                background: #ff6600;
                color: white;
                border: 2px solid #ff6600;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #e55a00;
            }
        """)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)
        
        # Show dialog
        dialog.exec_()
        
    def _save_port_settings(self, dialog, port_combo, baud_combo):
        """Save port settings and close dialog"""
        selected_port = port_combo.currentData() or port_combo.currentText()
        selected_baud = baud_combo.currentText()
        
        if selected_port != "Select Port":
            self.settings_manager.set_setting("serial_port", selected_port)
            self.settings_manager.set_setting("baud_rate", selected_baud)
            print(f"Port settings saved: {selected_port} at {selected_baud} baud")
            dialog.accept()
        else:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid COM port.")

    def refresh_port_combo(self, port_combo):
        """Refresh the port combo box with currently available ports"""
        port_combo.clear()
        port_combo.addItem("Select Port")
        
        try:
            ports = serial.tools.list_ports.comports()
            if not ports:
                port_combo.addItem("No ports found")
                QMessageBox.information(self, "Port Refresh", "No serial ports detected.")
            else:
                lines = []
                for port in ports:
                    device = getattr(port, 'device', '')
                    desc = getattr(port, 'description', '')
                    hwid = getattr(port, 'hwid', '')
                    label = device
                    extra = " - ".join([s for s in [desc, hwid] if s])
                    if extra:
                        label = f"{device} - {extra}"
                    port_combo.addItem(label, device)
                    lines.append(label)
                QMessageBox.information(
                    self,
                    "Port Refresh",
                    f"Found {len(ports)} serial ports:\n" + "\n".join(lines)
                )
        except Exception as e:
            port_combo.addItem("Error detecting ports")
            QMessageBox.warning(self, "Port Refresh Error", f"Error detecting ports: {str(e)}")

    def test_serial_connection(self, port, baud_rate):
        """Test the serial connection with the specified port and baud rate"""
        if port == "Select Port":
            QMessageBox.warning(self, "Invalid Port", "Please select a valid COM port first.")
            return
        
        try:
            # Try to open the serial connection
            test_serial = serial.Serial(port, int(baud_rate), timeout=1)
            test_serial.close()
            
            QMessageBox.information(self, "Connection Test", 
                f"✅ Connection successful!\nPort: {port}\nBaud Rate: {baud_rate}")
            
        except serial.SerialException as e:
            QMessageBox.critical(self, "Connection Failed", 
                f"❌ Connection failed!\nPort: {port}\nBaud Rate: {baud_rate}\n\nError: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Connection Error", 
                f"❌ Unexpected error!\nError: {str(e)}")

    # ------------------------ Capture Screen Details ------------------------

    def capture_screen(self):
        try:
            
            # Get the main window
            main_window = self.window()
            
            # Create a timer to delay the capture slightly to ensure UI is ready
            def delayed_capture():
                # Capture the entire window
                pixmap = main_window.grab()
                
                # Show save dialog
                filename, _ = QFileDialog.getSaveFileName(
                    self, 
                    "Save Screenshot", 
                    f"ECG_Screenshot_{QDateTime.currentDateTime().toString('yyyy-MM-dd_hh-mm-ss')}.png",
                    "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
                )
                
                if filename:
                    # Save the screenshot
                    if pixmap.save(filename):
                        QMessageBox.information(
                            self, 
                            "Success", 
                            f"Screenshot saved successfully!\nLocation: {filename}"
                        )
                    else:
                        QMessageBox.warning(
                            self, 
                            "Error", 
                            "Failed to save screenshot."
                        )
            
            # Use a short delay to ensure the UI is fully rendered
            QTimer.singleShot(100, delayed_capture)
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to capture screenshot: {str(e)}"
            )

    # ------------------------ Recording Details ------------------------

    def toggle_recording(self):
        if self.recording_toggle.isChecked():
            # Check if recording is allowed (acquisition running or demo mode on)
            if not self._can_record():
                # Reset button state
                self.recording_toggle.setChecked(False)
                QMessageBox.warning(
                    self, 
                    "Recording Not Available", 
                    "Recording is only available when:\n"
                    "• Data acquisition is running, OR\n"
                    "• Demo mode is enabled"
                )
                return
            self.start_recording()
        else:
            self.stop_recording()
    
    def _can_record(self):
        """Check if recording is allowed - only when acquisition is running or demo mode is on"""
        try:
            # Check if demo mode is on
            is_demo_mode = False
            if hasattr(self, 'demo_toggle') and self.demo_toggle:
                is_demo_mode = self.demo_toggle.isChecked()
            
            # Check if acquisition is running (timer is active)
            is_acquisition_running = False
            if hasattr(self, 'timer') and self.timer:
                is_acquisition_running = self.timer.isActive()
            
            # Recording is allowed if either demo mode is on OR acquisition is running
            return is_demo_mode or is_acquisition_running
        except Exception as e:
            print(f"Error checking recording conditions: {e}")
            return False
    
    def update_recording_button_state(self):
        """Update the recording button enabled state based on acquisition and demo status"""
        try:
            if not hasattr(self, 'recording_toggle') or not self.recording_toggle:
                return
            
            can_record = self._can_record()
            
            # Enable/disable the button based on whether recording is allowed
            self.recording_toggle.setEnabled(can_record)
            
            # If recording is currently active but conditions are no longer met, stop it
            if self.is_recording and not can_record:
                self.stop_recording()
                QMessageBox.warning(
                    self,
                    "Recording Stopped",
                    "Recording stopped because:\n"
                    "• Data acquisition ended, AND\n"
                    "• Demo mode is disabled"
                )
        except Exception as e:
            print(f"Error updating recording button state: {e}")
    
    def start_recording(self):
        try:
            # Double-check conditions before starting
            if not self._can_record():
                self.recording_toggle.setChecked(False)
                QMessageBox.warning(
                    self, 
                    "Recording Not Available", 
                    "Recording is only available when:\n"
                    "• Data acquisition is running, OR\n"
                    "• Demo mode is enabled"
                )
                return
            
            # Initialize recording
            self.is_recording = True
            
            # Update UI - only change button text, no status updates
            self.recording_toggle.setText(self.tr("Stop Recording"))
            
            # Start capture timer
            import time
            self.recording_start_time = time.time()
            self.recording_timer = QTimer()
            self.recording_timer.timeout.connect(self.capture_frame)
            _rec_interval_ms = 33  # ~30 FPS target
            self.recording_timer.start(_rec_interval_ms)

            # Give the frame buffer a hard cap now that the frame rate is known.
            # Without a maxlen this list grew at ~187 MB/s and took the machine
            # into swap, which stalls the serial drain sharing this thread.
            _fps = max(1, int(round(1000.0 / max(1, _rec_interval_ms))))
            _max_frames = max(60, _fps * int(getattr(self, '_RECORDING_MAX_SECONDS', 120)))
            self.recording_frames = deque(maxlen=_max_frames)
            self._recording_frame_shape = None
            print(f" Recording buffer capped at {_max_frames} frames "
                  f"(~{_max_frames / _fps:.0f}s at {_fps} fps)")

            # Show notification as recording started
            QMessageBox.information(self, self.tr("Success"), self.tr("Recording started"))
            
        except Exception as e:
            QMessageBox.warning(self, "Recording Error", f"Failed to start recording: {str(e)}")
            self.is_recording = False
            self.recording_toggle.setChecked(False)
    
    def stop_recording(self):
        try:
            # Stop recording
            self.is_recording = False
            if hasattr(self, 'recording_timer'):
                self.recording_timer.stop()
            
            # Update UI - only change button text, no status updates
            self.recording_toggle.setText(self.tr("Record Screen"))
            
            # Ask user if they want to save the recording
            if len(self.recording_frames) > 0:
                import time
                duration = time.time() - getattr(self, 'recording_start_time', time.time() - 1)
                self._actual_fps = max(1.0, len(self.recording_frames) / max(0.1, duration))
                
                reply = QMessageBox.question(
                    self, 
                    "Save Recording", 
                    "Would you like to save the recording?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.save_recording()
                else:
                    # Discard recording
                    self.recording_frames.clear()
                    QMessageBox.information(self, "Recording Discarded", "Recording has been discarded.")
            
        except Exception as e:
            QMessageBox.warning(self, "Recording Error", f"Failed to stop recording: {str(e)}")
            self.recording_toggle.setChecked(True)

    def capture_frame(self):
        try:
            if self.is_recording:
                # Check if recording should continue (acquisition running or demo mode on)
                if not self._can_record():
                    # Stop recording if conditions are no longer met
                    self.stop_recording()
                    QMessageBox.warning(
                        self,
                        "Recording Stopped",
                        "Recording stopped because:\n"
                        "• Data acquisition ended, AND\n"
                        "• Demo mode is disabled"
                    )
                    return
                
                # Capture the current window
                screen = QApplication.primaryScreen()
                pixmap = screen.grabWindow(self.winId())
                
                # Convert to numpy array for OpenCV
                image = pixmap.toImage()
                width = image.width()
                height = image.height()
                ptr = image.bits()
                ptr.setsize(height * width * 4)
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

                # Downscale wide grabs before they are held in memory. A 1920-wide
                # frame is 6.22 MB; at 1280 it is 2.76 MB, and the recording is a
                # screen capture for review, not a diagnostic artefact.
                max_w = int(getattr(self, '_RECORDING_MAX_WIDTH', 1280))
                if max_w and arr.shape[1] > max_w:
                    scale = max_w / float(arr.shape[1])
                    arr = cv2.resize(
                        arr,
                        (max_w, max(2, int(round(arr.shape[0] * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                    # cv2.VideoWriter needs even dimensions for most codecs.
                    if arr.shape[0] % 2:
                        arr = arr[:-1, :, :]

                # Every frame in one recording must have identical dimensions or
                # cv2.VideoWriter silently writes a corrupt file. The window can be
                # resized mid-recording, so the first frame's geometry is pinned and
                # everything after is forced to match it.
                _shape = getattr(self, '_recording_frame_shape', None)
                if _shape is None:
                    self._recording_frame_shape = (arr.shape[1], arr.shape[0])  # (w, h)
                elif (arr.shape[1], arr.shape[0]) != _shape:
                    arr = cv2.resize(arr, _shape, interpolation=cv2.INTER_AREA)

                # Bounded store — the deque drops the oldest frame at maxlen.
                self.recording_frames.append(np.ascontiguousarray(arr))
                
        except Exception as e:
            print(f"Frame capture error: {e}")
    
    def save_recording(self):
        try:
            if len(self.recording_frames) == 0:
                QMessageBox.warning(self, "No Recording", "No frames to save.")
                return
            
            # Get save file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"ecg_recording_{timestamp}.mp4"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Recording",
                default_filename,
                "MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)"
            )
            
            if file_path:
                # Get video dimensions from first frame
                height, width = self.recording_frames[0].shape[:2]
                
                # Create video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = getattr(self, '_actual_fps', 30.0)
                out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
                
                # Write frames
                for frame in self.recording_frames:
                    out.write(frame)
                
                out.release()
                
                # Clear frames
                self.recording_frames.clear()
                
                QMessageBox.information(
                    self, 
                    "Recording Saved", 
                    f"Recording saved successfully to:\n{file_path}"
                )
            else:
                # User cancelled save
                self.recording_frames.clear()
                QMessageBox.information(self, "Recording Cancelled", "Recording was not saved.")
                
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save recording: {str(e)}")
            self.recording_frames.clear()

    # ------------------------ Get lead figure in pdf ------------------------

    def get_lead_figure(self, lead):
        if hasattr(self, "lead_figures"):
            return self.lead_figures.get(lead)

        ordered_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        if hasattr(self, "figures"):
            if lead in ordered_leads:
                idx = ordered_leads.index(lead)
                if idx < len(self.figures):
                    return self.figures[idx]
        return None

    def center_on_screen(self):
        qr = self.frameGeometry()
        from PyQt5.QtWidgets import QApplication
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def expand_lead(self, idx):
        lead = self.leads[idx]
        def get_lead_data():
            return self.data[lead]
        color = self.LEAD_COLORS.get(lead, "#00ff99")
        if hasattr(self, '_detailed_timer') and self._detailed_timer is not None:
            self._detailed_timer.stop()
            self._detailed_timer.deleteLater()
            self._detailed_timer = None
        old_layout = self.detailed_widget.layout()
        if old_layout is not None:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            QWidget().setLayout(old_layout)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        back_btn = QPushButton("Return to Dashboard")
        back_btn.setFixedHeight(40)
        back_btn.clicked.connect(lambda: self.page_stack.setCurrentIndex(0))
        layout.addWidget(back_btn, alignment=Qt.AlignLeft)
        fig = Figure(facecolor='#fff')  # White background for the figure
        ax = fig.add_subplot(111)
        ax.set_facecolor('#fff')        # White background for the axes
        line, = ax.plot([], [], color=color, lw=0.7)
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(canvas)
        # Create metric labels for cards
        pr_label = QLabel("0 ms")
        qrs_label = QLabel("0 ms")
        qtc_label = QLabel("0 ms")
        arrhythmia_label = QLabel("--")
        # Add metrics card row below the plot (card style)
        metrics_row = QHBoxLayout()
        def create_metric_card(title, label_widget):
            card = QFrame()
            card.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #fff7f0, stop:1 #ffe0cc);
                border-radius: 32px;
                border: 2.5px solid #ff6600;
                padding: 18px 18px;
            """)
            vbox = QVBoxLayout(card)
            vbox.setSpacing(6)
            lbl = QLabel(title)
            lbl.setAlignment(Qt.AlignHCenter)
            lbl.setStyleSheet("color: #ff6600; font-size: 18px; font-weight: bold;")
            label_widget.setStyleSheet("font-size: 32px; font-weight: bold; color: #222; padding: 8px 0;")
            vbox.addWidget(lbl)
            vbox.addWidget(label_widget)
            vbox.setAlignment(Qt.AlignHCenter)
            return card
        metrics_row.setSpacing(32)
        metrics_row.setContentsMargins(32, 16, 32, 24)
        metrics_row.setAlignment(Qt.AlignHCenter)
        cards = [create_metric_card("PR Interval", pr_label),
                 create_metric_card("QRS Duration", qrs_label),
                 create_metric_card("QT/Qtc Interval", qtc_label),
                 create_metric_card("Arrhythmia", arrhythmia_label)]
        for card in cards:
            card.setMinimumWidth(0)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            metrics_row.addWidget(card)
        layout.addLayout(metrics_row)
        self.detailed_widget.setLayout(layout)
        self.detailed_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.page_stack.setCurrentIndex(1)
        self._detailed_timer = QTimer(self)

        def update_detailed_plot():
            detailed_buffer_size = 500  # Reduced to 500 samples for real-time effect
            data = get_lead_data()

            current_gain = self.settings_manager.get_wave_gain()
            current_speed = 25.0 if getattr(self, '_overlay_active', False) else self.settings_manager.get_wave_speed()

            # Robust: Only plot if enough data, else show blank
            if data and len(data) >= 10:
                plot_data = np.array(data[-detailed_buffer_size:])
                x = np.arange(len(plot_data))
                centered = plot_data - np.mean(plot_data)

                # Apply current gain setting
                gain_factor = get_display_gain(current_gain)
                centered = centered * gain_factor

                line.set_data(x, centered)
                ax.set_xlim(0, max(len(centered)-1, 1))
                
                ylim = 500 * gain_factor
                ymin = np.min(centered) - ylim * 0.2
                ymax = np.max(centered) + ylim * 0.2
                if ymin == ymax:
                    ymin, ymax = -ylim, ylim
                ax.set_ylim(ymin, ymax)

                # --- PQRST detection and green labeling for Lead II only ---
                # Remove all extra lines except the main ECG line (robust for all Matplotlib versions)
                try:
                    while len(ax.lines) > 1:
                        ax.lines.remove(ax.lines[-1])
                except Exception as e:
                    print(f"Warning: Could not remove extra lines: {e}")
                for txt in list(ax.texts):
                    try:
                        txt.remove()
                    except Exception as e:
                        print(f"Warning: Could not remove text: {e}")
                # Optionally, clear all lines if you want only labels visible (no ECG trace):
                # ax.lines.clear()
                if lead == "II":
                    # Use the same detection logic as in main.py
                    from scipy.signal import find_peaks
                    sampling_rate = 80
                    ecg_signal = centered
                    window_size = min(500, len(ecg_signal))
                    if len(ecg_signal) > window_size:
                        ecg_signal = ecg_signal[-window_size:]
                        x = x[-window_size:]
                    # R peak detection
                    r_peaks, _ = find_peaks(ecg_signal, distance=int(0.2 * sampling_rate), prominence=0.6 * np.std(ecg_signal))
                    # Q and S: local minima before and after R
                    q_peaks = []
                    s_peaks = []
                    for r in r_peaks:
                        q_start = max(0, r - int(0.06 * sampling_rate))
                        q_end = r
                        if q_end > q_start:  
                            q_idx = np.argmin(ecg_signal[q_start:q_end]) + q_start
                            q_peaks.append(q_idx)
                        s_start = r
                        s_end = min(len(ecg_signal), r + int(0.06 * sampling_rate))
                        if s_end > s_start:
                            s_idx = np.argmin(ecg_signal[s_start:s_end]) + s_start
                            s_peaks.append(s_idx)
                    # P: positive peak before Q (within 0.1-0.2s)
                    p_peaks = []
                    for q in q_peaks:
                        p_start = max(0, q - int(0.2 * sampling_rate))
                        p_end = q - int(0.08 * sampling_rate)
                        if p_end > p_start:
                            p_candidates, _ = find_peaks(ecg_signal[p_start:p_end], prominence=0.1 * np.std(ecg_signal))
                            if len(p_candidates) > 0:
                                p_peaks.append(p_start + p_candidates[-1])
                    # T: positive peak after S (within 0.1-0.4s)
                    t_peaks = []
                    for s in s_peaks:
                        t_start = s + int(0.08 * sampling_rate)
                        t_end = min(len(ecg_signal), s + int(0.4 * sampling_rate))
                        if t_end > t_start:
                            t_candidates, _ = find_peaks(ecg_signal[t_start:t_end], prominence=0.1 * np.std(ecg_signal))
                            if len(t_candidates) > 0:
                                t_peaks.append(t_start + t_candidates[np.argmax(ecg_signal[t_start + t_candidates])])
                    # Only show the most recent peak for each label (if any)
                    peak_dict = {'P': p_peaks, 'Q': q_peaks, 'R': r_peaks, 'S': s_peaks, 'T': t_peaks}
                    for label, idxs in peak_dict.items():
                        if len(idxs) > 0:
                            idx = idxs[-1]
                            ax.plot(idx, ecg_signal[idx], 'o', color='green', markersize=8, zorder=10)
                            y_offset = 0.12 * (np.max(ecg_signal) - np.min(ecg_signal))
                            if label in ['P', 'T']:
                                ax.text(idx, ecg_signal[idx]+y_offset, label, color='green', fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=11, bbox=dict(facecolor='white', edgecolor='none', alpha=2.0, boxstyle='round,pad=0.1'))
                            else:
                                ax.text(idx, ecg_signal[idx]-y_offset, label, color='green', fontsize=12, fontweight='bold', ha='center', va='top', zorder=11, bbox=dict(facecolor='white', edgecolor='none', alpha=2.0, boxstyle='round,pad=0.1'))
                # --- Metrics (for Lead II only, based on R peaks) ---
                # --- Metrics (for Lead II only, based on R peaks) ---
                if lead == "II":
                    # Fixed Bug PR-1: Use centralized metrics instead of legacy inline calculation
                    # Fetch current values from global metric labels (which use the robust median-beat logic)
                    from .ui.display_updates import get_current_metrics_from_labels
                    current_metrics = get_current_metrics_from_labels(getattr(self, 'metric_labels', {}))
                    
                    heart_rate = current_metrics.get('heart_rate')
                    pr_interval = current_metrics.get('pr_interval')
                    qrs_duration = current_metrics.get('qrs_duration')
                    qt_interval = current_metrics.get('qt_interval')
                    qtc_interval = current_metrics.get('qtc_interval')
                    
                    # Update ECG metrics labels with fetched values
                    if isinstance(pr_interval, (int, float)) and pr_interval > 0:
                        pr_label.setText(f"{int(round(pr_interval))} ms")
                    else:
                        pr_label.setText("-- ms")

                    if isinstance(qrs_duration, (int, float)) and qrs_duration > 0:
                        qrs_label.setText(f"{int(round(qrs_duration))} ms")
                    else:
                        qrs_label.setText("-- ms")

                    if isinstance(qtc_interval, (int, float)) and qtc_interval > 0:
                        qtc_label.setText(f"{int(round(qtc_interval))} ms")
                    else:
                        qtc_label.setText("-- ms")
                        
                    # Calculate ST segment using Lead II (keep this as it's separate from intervals)
                    lead_ii = self.data[1] if len(self.data) > 1 else []
                    # Use dynamic sampling rate (Fixed Bug PR-2)
                    fs_st = 500.0
                    if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                        fs_st = float(self.sampler.sampling_rate)
                        
                    # We still need R-peaks for ST calculation if we want a fresh update
                    # But finding peaks on 'centered' (view data) is risky if gain is applied
                    # So we use a simple fallback or reusing global ST if available
                    st_segment = current_metrics.get('st_interval') # Use global ST
                    if st_segment is None:
                         st_segment = 0.0 # fallback

                    # --- Arrhythmia detection using current metrics ---
                    # Keep the detailed window BPM-driven so HR 50 shows Bradycardia,
                    # HR > 100 shows Tachycardia, and the normal range shows NSR.
                    arrhythmia_result = _bpm_rhythm_label(heart_rate)
                    if arrhythmia_result is None:
                        arrhythmia_result = _bpm_rhythm_label(getattr(self, 'last_heart_rate', None))
                    if arrhythmia_result is None:
                        arrhythmia_result = getattr(self, '_latest_rhythm_interpretation', "Analyzing Rhythm...")
                        
                    arrhythmia_label.setText(arrhythmia_result)
                    self._latest_rhythm_interpretation = arrhythmia_result
                else:
                    pr_label.setText("-- ms")
                    qrs_label.setText("-- ms")
                    qtc_label.setText("-- ms")
                    arrhythmia_label.setText("0")
                    self._latest_rhythm_interpretation = "Analyzing Rhythm..."
            else:
                line.set_data([], [])
                ax.set_xlim(0, 1)
                ax.set_ylim(-500, 500)
                pr_label.setText("0 ms")
                qrs_label.setText("0 ms")
                qtc_label.setText("0 ms")
            canvas.draw_idle()
        self._detailed_timer.timeout.connect(update_detailed_plot)
        self._detailed_timer.start(100)
        update_detailed_plot()  # Draw immediately on open

    def refresh_ports(self):
        self.port_combo.clear()
        self.port_combo.addItem("Select Port")
        
        try:
            ports = serial.tools.list_ports.comports()
            if not ports:
                self.port_combo.addItem("No ports found")
                print(" No serial ports detected during refresh")
            else:
                for port in ports:
                    self.port_combo.addItem(port.device)
                print(f" Refreshed: Found {len(ports)} serial ports")
        except Exception as e:
            self.port_combo.addItem("Error detecting ports")
            print(f" Error refreshing ports: {e}")

    def update_lead_layout(self):
        old_layout = self.plot_area.layout()
        if old_layout:
            while old_layout.count():
                item = old_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)
            self.plot_area.setLayout(None)
        self.figures = []
        self.canvases = []
        self.axs = []
        self.lines = []
        grid = QGridLayout()
        grid.setSpacing(8)  # Reduced spacing between graphs
        n_leads = len(self.leads)
        if n_leads == 12:
            rows, cols = 3, 4
        elif n_leads == 7:
            rows, cols = 2, 4
        else:
            rows, cols = 1, 1
        for idx, lead in enumerate(self.leads):
            row, col = divmod(idx, cols)
            group = QGroupBox(lead)
            group.setStyleSheet("""
                QGroupBox {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    border: 2px solid #e9ecef;
                    border-radius: 12px;  /* Reduced from 16px */
                    color: #495057;
                    font: bold 14px 'Arial';  /* Reduced from 16px and changed font */
                    margin-top: 8px;  /* Reduced from 12px */
                    padding: 8px;  /* Reduced from 12px */
                    /* Removed unsupported box-shadow property */
                }
                QGroupBox:hover {
                    border: 2px solid #ff6600;
                    /* Removed unsupported box-shadow and transform properties */
                }
            """)
            vbox = QVBoxLayout(group)
            vbox.setContentsMargins(6, 6, 6, 6)  # Reduced margins
            vbox.setSpacing(4)  # Reduced spacing
            fig = Figure(facecolor='#ffffff', figsize=(5, 2))  # Reduced from (6, 2.5)
            ax = fig.add_subplot(111)
            ax.set_facecolor('#ffffff')
            ylim = self.ylim if hasattr(self, 'ylim') else 400
            ax.set_ylim(-ylim, ylim)
            ax.set_xlim(0, self.buffer_size)
            
            # Modern grid styling
            ax.grid(False)
            ax.set_axisbelow(True)

            # Remove spines for cleaner look
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Style ticks - Make them smaller
            ax.tick_params(axis='both', colors='#6c757d', labelsize=8)  # Reduced from 10
            ax.tick_params(axis='x', length=0)
            ax.tick_params(axis='y', length=0)

            # Enhanced line styling
            import matplotlib.patheffects as path_effects 
            line, = ax.plot([0]*self.buffer_size, 
                            color=self.LEAD_COLORS.get(lead, '#ff6600'), 
                            lw=0.5, 
                            alpha=0.9,
                            path_effects=[path_effects.SimpleLineShadow(offset=(1,1), alpha=0.3),
                                        path_effects.Normal()])

            self.lines.append(line)
            canvas = FigureCanvas(fig)
            vbox.addWidget(canvas)
            grid.addWidget(group, row, col)
            self.figures.append(fig)
            self.canvases.append(canvas)
            self.axs.append(ax)
        self.plot_area.setLayout(grid)
        def make_expand_lead(idx):
            return lambda event: self.expand_lead(idx)
        for i, canvas in enumerate(self.canvases):
            canvas.mpl_connect('button_press_event', make_expand_lead(i))

    def redraw_all_plots(self):
        
        if hasattr(self, 'lines') and self.lines:
            for i, line in enumerate(self.lines):
                if i < len(self.leads):
                    lead = self.leads[i]
                    data = self.data.get(lead, [])
                    
                    if len(data) > 0:
                        # Detect signal source and apply adaptive scaling
                        signal_source = self.detect_signal_source(data)
                        # Calculate gain factor: higher mm/mV = higher gain (10mm/mV = 1.0x baseline)
                        gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                        device_data = np.array(data)
                        centered = self.apply_adaptive_gain(device_data, signal_source, gain_factor)
                        
                        # Apply medical-grade filtering for smooth waves
                        filtered_data = self.apply_ecg_filtering(centered)
                        
                        # Update line data with new buffer size
                        if len(filtered_data) < self.buffer_size:
                            plot_data = np.full(self.buffer_size, np.nan)
                            plot_data[-len(filtered_data):] = filtered_data
                        else:
                            plot_data = filtered_data[-self.buffer_size:]
                        
                        # ALWAYS ensure left-to-right scrolling by explicitly setting x-axis data
                        x_data = np.arange(len(plot_data), dtype=float)
                        line.set_data(x_data, plot_data)
                        
                        # Update axis limits with adaptive Y-range
                        if i < len(self.axs):
                            # ALWAYS set x-limits from 0 to buffer_size to ensure left-to-right scrolling
                            self.axs[i].set_xlim(0, self.buffer_size)
                            
                            # Calculate adaptive Y-range for matplotlib plots
                            valid_data = filtered_data[~np.isnan(filtered_data)]
                            if len(valid_data) > 0:
                                p1 = np.percentile(valid_data, 1)
                                p99 = np.percentile(valid_data, 99)
                                data_mean = (p1 + p99) / 2.0
                                data_std = np.std(valid_data[(valid_data >= p1) & (valid_data <= p99)])
                                
                                if signal_source == "human_body":
                                    padding = max(data_std * 2, 20)
                                elif signal_source == "weak_body":
                                    padding = max(data_std * 1.5, 10)
                                else:
                                    padding = max(data_std * 4, 200)
                                
                                y_min = data_mean - padding
                                y_max = data_mean + padding
                                self.axs[i].set_ylim(y_min, y_max)
                            else:
                                ylim = self.ylim if hasattr(self, 'ylim') else 400
                                self.axs[i].set_ylim(-ylim, ylim)
                            
                            self.axs[i].set_xlim(0, self.buffer_size)
                            
                            # Update plot title with current settings and signal source
                            current_speed = 25.0 if getattr(self, '_overlay_active', False) else self.settings_manager.get_wave_speed()
                            current_gain = self.settings_manager.get_wave_gain()
                            signal_type = "Body" if signal_source in ["human_body", "weak_body"] else "Hardware"
                            new_title = f"{lead} | Speed: {current_speed}mm/s | Gain: {current_gain}mm/mV | {signal_type}"
                            self.axs[i].set_title(new_title, fontsize=8, color='#666', pad=10)
                            print(f"Redraw updated {lead} title: {new_title}")
                        
                        # Redraw canvas
                        if i < len(self.canvases):
                            self.canvases[i].draw_idle()

    def detect_signal_source(self, data):
        """Detect if signal is from hardware or human body - wrapper for modular function"""
        return detect_signal_source(np.asarray(data) if data is not None else np.array([]))

    def apply_adaptive_gain(self, data, signal_source, gain_factor):
        """Apply gain based on signal source - wrapper for modular function"""
        return apply_adaptive_gain(np.asarray(data), signal_source, gain_factor)

    def update_plot_y_range_adaptive(self, plot_index, signal_source, data_override=None):
        """Update Y-axis range based on signal source with adaptive scaling.
        If data_override is provided, use it for statistics (should be the plotted/scaled data).
        Y-axis automatically adjusts to gain to prevent cropping.
        
        NOTE: AUTO-SCALING IS CURRENTLY COMMENTED OUT - USING FIXED RANGE FOR TESTING
        """
        try:
            if plot_index >= len(self.data) or plot_index >= len(self.plot_widgets):
                return

            # ========== AUTO-SCALING CODE COMMENTED OUT ==========
            # # Get the data for this plot
            # if data_override is not None:
            #     data = np.asarray(data_override)
            #     # Data is already scaled with gain, so don't apply gain again
            #     data_already_scaled = True
            # else:
            #     data = self.data[plot_index]
            #     # Data is not scaled, will need to apply gain
            #     data_already_scaled = False
            # 
            # # Remove NaN values and large outliers (robust)
            # valid_data = data[~np.isnan(data)]
            # 
            # if len(valid_data) == 0:
            #     return
            # 
            # # Get current gain setting to properly scale Y-axis
            # current_gain = get_display_gain(self.settings_manager.get_wave_gain())
            # 
            # # Use percentiles to avoid spikes from clipping the view
            # p1 = np.percentile(valid_data, 1)
            # p99 = np.percentile(valid_data, 99)
            # data_mean = (p1 + p99) / 2.0
            # data_std = np.std(valid_data[(valid_data >= p1) & (valid_data <= p99)])
            # # Maximum deviation of any point from the mean – we will always cover this
            # peak_deviation = np.max(np.abs(valid_data - data_mean)) if len(valid_data) > 0 else 0.0
            # 
            # # Calculate appropriate Y-range with adaptive padding based on signal source.
            # # Goal: make peaks visually bigger but still avoid cropping by using robust stats.
            # # Medical standard: Y-axis should scale with gain to accommodate larger amplitudes
            # if signal_source == "human_body":
            #     # Scale fixed range with gain for human body signals
            #     base_range = 600
            #     y_range = base_range * current_gain
            #     y_min = -y_range
            #     y_max = y_range
            #     print(f" Human body Y-range: ±{y_range:.1f} (gain={current_gain:.2f}x)")
            # elif signal_source == "weak_body":
            #     # Scale fixed range with gain for weak body signals
            #     base_range = 400
            #     y_range = base_range * current_gain
            #     y_min = -y_range
            #     y_max = y_range
            #     print(f" Weak body Y-range: ±{y_range:.1f} (gain={current_gain:.2f}x)")
            # else:
            #     # Hardware / unknown – use data-driven range that scales with gain
            #     # Base padding scales with gain to accommodate larger amplitudes
            #     base_padding = max(data_std * 3.0, 250) * current_gain
            #     padding = base_padding
            #     print(f" Hardware Y-range: base_padding={base_padding:.1f}, gain={current_gain:.2f}x")
            # 
            # # FINAL SAFETY: always cover the tallest peak with 15% headroom,
            # # so waves never touch or cross the plot border (no cropping),
            # # regardless of gain/speed combinations.
            # if signal_source not in ["human_body", "weak_body"] and peak_deviation > 0:
            #     min_padding = peak_deviation * 1.15  # 15% headroom to prevent cropping
            #     if padding < min_padding:
            #         padding = min_padding
            #         print(f" Adjusted padding to {padding:.1f} to cover peak deviation {peak_deviation:.1f}")
            # 
            # if signal_source not in ["human_body", "weak_body"]:
            #     if data_std > 0:
            #         y_min = data_mean - padding
            #         y_max = data_mean + padding
            #     else:
            #         # Fallback: use gain-scaled default range
            #         base_range = 400 * current_gain
            #         y_min = -base_range
            #         y_max = base_range
            # else:
            #     # Already set above for body signals
            #     pass
            # 
            # # Ensure reasonable bounds (but allow wider range for high gain)
            # max_range = 10000 * current_gain  # Scale max range with gain
            # y_min = max(y_min, -max_range)
            # y_max = min(y_max, max_range)
            # ========== END OF COMMENTED AUTO-SCALING CODE ==========
            
            # ========== FIXED RANGE (NO AUTO-SCALING) WITH CENTERED Y-AXIS ==========
            # Use fixed Y-axis range regardless of signal source or data characteristics
            # Center the Y-axis around the data mean so all waves are centered in their boxes
            # Ensure peaks don't go outside the box by calculating peak deviation
            current_gain = get_display_gain(self.settings_manager.get_wave_gain())
            
            # Get the data for this plot to calculate center and peak deviation
            if data_override is not None:
                data = np.asarray(data_override)
            else:
                data = self.data[plot_index]
            
            # Calculate the center (median) of the data to center the Y-axis
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                # Use median for more robust centering (less affected by outliers)
                data_center = np.median(valid_data)
                
                # Calculate peak deviation from center to ensure peaks stay within box
                # Use percentiles to avoid outliers affecting the range
                p1 = np.percentile(valid_data, 1)
                p99 = np.percentile(valid_data, 99)
                peak_deviation = max(
                    abs(p99 - data_center),  # Maximum positive deviation
                    abs(p1 - data_center)    # Maximum negative deviation
                )
            else:
                # Fallback to 0 if no valid data
                data_center = 0.0
                peak_deviation = 0.0
            
            # Clinical 12-lead ECG: symmetric Y-axis centred at 0 for ALL leads.
            # aVR's negative polarity comes from the formula (-(I+II)/2), not a flipped axis.
            y_min, y_max = -2048, 2048
            center_str = "0"
            
            # OPTIMIZED: Reduce Y-range print frequency for better performance
            if not hasattr(self, '_y_range_print_count'):
                self._y_range_print_count = {}
            if plot_index not in self._y_range_print_count:
                self._y_range_print_count[plot_index] = 0
            self._y_range_print_count[plot_index] += 1
            
            # Only print every 100th Y-range update per plot
            if self._y_range_print_count[plot_index] % 100 == 1:
                print(f" FIXED Y-range: {y_min} to {y_max} (centered at {center_str}, gain={current_gain:.2f}x, peak_dev={peak_deviation:.1f}, signal_source={signal_source})")
            # ========== END OF FIXED RANGE CODE ==========
            
            # Apply the fixed Y-range using PyQtGraph with NO padding
            self.plot_widgets[plot_index].setYRange(y_min, y_max, padding=0)
            
        except Exception as e:
            print(f" Error updating adaptive Y-range: {e}")

    def update_ecg_lead(self, lead_index, data_array):
        """Update a specific ECG lead with new data from serial communication"""
        try:
            if 0 <= lead_index < len(self.lines) and len(data_array) > 0:
                # Detect signal source first
                signal_source = self.detect_signal_source(data_array)
                
                # Apply current settings to the incoming data
                # Calculate gain factor: higher mm/mV = higher gain (10mm/mV = 1.0x baseline)
                gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                
                # Apply adaptive gain based on signal source
                centered = self.apply_adaptive_gain(data_array, signal_source, gain_factor)
                
                # Apply noise reduction filtering
                filtered_data = self.apply_ecg_filtering(centered)
                
                # Update line data with new buffer size
                if len(filtered_data) < self.buffer_size:
                    plot_data = np.full(self.buffer_size, np.nan)
                    plot_data[-len(filtered_data):] = filtered_data
                else:
                    plot_data = filtered_data[-self.buffer_size:]
                
                # Update the specific lead line - ALWAYS ensure left-to-right scrolling
                # Explicitly set x-axis data to ensure left-to-right direction (0 to buffer_size)
                x_data = np.arange(len(plot_data), dtype=float)
                self.lines[lead_index].set_data(x_data, plot_data)
                
                # Update axis limits with adaptive Y-range
                if lead_index < len(self.axs):
                    # Use adaptive Y-range based on the filtered (plotted) data
                    self.update_plot_y_range_adaptive(lead_index, signal_source, data_override=filtered_data)
                    # ALWAYS set x-limits from 0 to buffer_size to ensure left-to-right scrolling
                    self.axs[lead_index].set_xlim(0, self.buffer_size)
                
                # Redraw the specific canvas
                if lead_index < len(self.canvases):
                    self.canvases[lead_index].draw_idle()
                    
                print(f"Updated ECG lead {lead_index} with {len(data_array)} samples")
                
        except Exception as e:
            print(f"Error updating ECG lead {lead_index}: {str(e)}")
    
    def apply_ecg_filtering(self, signal_data):
        """Apply medical-grade ECG filtering for smooth, clean waves like professional devices"""
        try:
            from scipy.signal import butter, filtfilt, savgol_filter, medfilt, wiener
            from scipy.ndimage import gaussian_filter1d
            from ecg.ecg_filters import apply_ecg_filters_from_settings
            import numpy as np
            
            if len(signal_data) < 10:  # Need minimum data for filtering
                return signal_data
            
            # Convert to numpy array
            signal = np.array(signal_data, dtype=float)
            
            # NOTE: Baseline correction is handled by slow baseline anchor in display paths
            # Do NOT subtract mean here - baseline anchor handles it before this function is called
            
            # Apply AC/EMG/DFT filters based on user settings from SettingsManager
            # This applies filters in correct order: DFT -> EMG -> AC
            sampling_rate = getattr(self, 'demo_fs', 500)  # Get sampling rate, default 500Hz
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate'):
                try:
                    sampling_rate = float(self.sampler.sampling_rate)
                except:
                    pass
            
            # Apply user-configured AC/EMG/DFT filters
            signal = apply_ecg_filters_from_settings(
                signal=signal,
                sampling_rate=sampling_rate,
                settings_manager=self.settings_manager
            )
            
            # 3. Medical-grade bandpass filter (0.5-30 Hz) - tighter range for cleaner signal
            fs = sampling_rate  # Use actual sampling rate
            nyquist = fs / 2
            
            # Low-pass filter to remove high-frequency noise (>30 Hz) - more aggressive
            low_cutoff = 30 / nyquist  # Reduced from 40 to 30 Hz
            b_low, a_low = butter(6, low_cutoff, btype='low')  # Increased order to 6
            signal = filtfilt(b_low, a_low, signal)
            
            # Note: High-pass filter is now handled by DFT filter, so we skip it here
            
            # 3. Wiener filter for medical-grade noise reduction
            if len(signal) > 5:
                signal = wiener(signal, noise=0.05)  # Lower noise parameter for smoother result
            
            # 4. Gaussian smoothing for medical-grade smoothness
            signal = gaussian_filter1d(signal, sigma=1.2)
            
            # 5. Savitzky-Golay filter with optimized parameters for ECG
            if len(signal) >= 15:  # Increased minimum window size
                window_length = min(15, len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
                signal = savgol_filter(signal, window_length, 4)  # Increased polynomial order to 4
            
            # 6. Adaptive median filter for spike removal
            signal = medfilt(signal, kernel_size=7)  # Increased kernel size for better smoothing
            
            # 7. Multi-stage moving average for ultra-smooth baseline
            # Stage 1: Short-term smoothing
            window1 = min(7, len(signal))
            if window1 > 1:
                kernel1 = np.ones(window1) / window1
                signal = np.convolve(signal, kernel1, mode='same')
            
            # Stage 2: Medium-term smoothing for baseline stability
            window2 = min(5, len(signal))
            if window2 > 1:
                kernel2 = np.ones(window2) / window2
                signal = np.convolve(signal, kernel2, mode='same')
            
            # 8. Final Gaussian smoothing for medical device quality
            signal = gaussian_filter1d(signal, sigma=0.8)
            
            return signal
            
        except Exception as e:
            print(f"Medical-grade filtering error: {e}")
            # Return original signal if filtering fails
            return signal_data
    
    def apply_realtime_smoothing(self, new_value, lead_index):
        """Apply real-time smoothing - wrapper for modular function"""
        if not hasattr(self, 'smoothing_buffers'):
            self.smoothing_buffers = {}
        return apply_realtime_smoothing(new_value, lead_index, self.smoothing_buffers)

    # ---------------------- Serial Port Auto-Detection ----------------------

    def get_available_serial_ports(self):
        """Get list of available serial ports"""
        if not SERIAL_AVAILABLE:
            return []
        
        ports = []
        try:
            # Get all available ports
            available_ports = serial.tools.list_ports.comports()
            for port_info in available_ports:
                ports.append(port_info.device)
            print(f" Found {len(ports)} available serial ports: {ports}")
        except Exception as e:
            print(f" Error detecting serial ports: {e}")
        
        return ports

    def auto_detect_serial_port(self):
        """Automatically detect and set the best available serial port"""
        available_ports = self.get_available_serial_ports()
        
        if not available_ports:
            return None, "No serial ports found"
        
        # Look for common ECG device patterns
        preferred_patterns = ['usbserial', 'usbmodem', 'ttyUSB', 'ttyACM']
        
        for pattern in preferred_patterns:
            for port in available_ports:
                if pattern in port.lower():
                    print(f" Auto-detected ECG device port: {port}")
                    return port, f"Auto-detected: {port}"
        
        # If no preferred pattern found, use the first available port
        if available_ports:
            port = available_ports[0]
            print(f" Using first available port: {port}")
            return port, f"Using first available: {port}"
        
        return None, "No suitable ports found"

    def _handle_ra_rl_disconnect(self):
        """RA/RL electrode loss makes every lead invalid (see packet_parser's
        ra_present/rl_absent gating). Rather than silently flatlining the
        display, alert the user with a blocking popup and return them to the
        outer dashboard. Fires once per disconnection event; start_acquisition
        resets the guard flag so the next session (or the next reconnect) can
        trigger it again.
        """
        if getattr(self, "_ra_rl_popup_shown", False):
            return
        self._ra_rl_popup_shown = True

        try:
            self.stop_acquisition()
        except Exception as e:
            print(f" Error stopping acquisition on RA/RL disconnect: {e}")

        popup_title = "Warning"
        popup_text = "ECG Electrode disconnected, please check electrode connection"
        try:
            # Reuse the app's styled dialog for a consistent look; imported
            # lazily to avoid a circular import at module load time.
            from dashboard.dashboard import StyledMessageBox
            StyledMessageBox.show_message(self, popup_title, popup_text, is_critical=True)
        except Exception as e:
            print(f" Falling back to QMessageBox for electrode disconnect alert: {e}")
            try:
                QMessageBox.critical(self, popup_title, popup_text)
            except Exception as e2:
                print(f" Error showing electrode disconnect popup: {e2}")

        # Send the user back to the outer dashboard, same as other
        # device-disconnect flows (e.g. check_device_connection).
        if hasattr(self, "dashboard_instance") and self.dashboard_instance:
            try:
                self.dashboard_instance.go_to_dashboard()
            except Exception as e:
                print(f" Error returning to dashboard after RA/RL disconnect: {e}")

    # ---------------------- Start Button Functionality ----------------------

    def start_acquisition(self):
        # Disable "Generate Report" button with a 10 second countdown after Start is pressed
        self._start_generate_report_cooldown(seconds=10, reason="Start")

        # CHECK: Ensure no other test is running
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            if not self.dashboard_instance.can_start_test("12_lead_test"):
                print(" Start aborted: Another test is currently running.")
                return
            # Set state to running
            self.dashboard_instance.update_test_state("12_lead_test", True)

        # Reset any replay/freeze state before starting live acquisition again.
        try:
            if hasattr(self, "_replay_timer") and self._replay_timer.isActive():
                self._replay_timer.stop()
        except Exception:
            pass
        self._display_mode = "live"
        self._grid_frozen = False
        self._replay_cursor = 0
        self._frozen_report_snapshot = None
        self._frozen_report_metrics = None
        self._replay_snapshot = None
        self._replay_render_snapshot = None

        try:
            if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
                print(" Switching from Demo to Real: turning off demo...")
                self.demo_toggle.setChecked(False)
                if hasattr(self, 'demo_manager'):
                    self.demo_manager.stop_demo_data()
        except Exception as e:
            print(f"[Start Acquisition] Failed to stop demo before real start: {e}")
        
        # Disable demo mode when hardware acquisition starts
        try:
            if hasattr(self, 'demo_toggle'):
                self.demo_toggle.setEnabled(False)
                self.demo_toggle.setStyleSheet("""
                    QPushButton {
                        background: #e0e0e0;
                        color: #a0a0a0;
                        border: 2px solid #cccccc;
                        border-radius: 8px;
                        padding: 8px 12px;
                        font-size: 12px;
                        font-weight: bold;
                        text-align: center;
                    }
                """)
                print(" Demo mode disabled (Hardware acquisition active)")
        except Exception as e:
            print(f" Error disabling demo mode: {e}")

        port = self.settings_manager.get_serial_port()
        baud = self.settings_manager.get_baud_rate()

        print(f"Starting acquisition with configured Port: {port}, Baud: {baud}")
        # Reset lead connection tracking for fresh acquisition session.
        self._lead_connection_state = {lead: True for lead in self.leads}
        self._lead_last_valid_value = {lead: 0.0 for lead in self.leads}
        self._last_off_leads_signature = None
        self._lead_data_received = False
        # Reset lead-off latch state for a fresh session.
        self._lead_off_latched = False
        self._lead_off_latch_on_count = 0
        self._lead_off_latch_off_count = 0
        self._ll_disconnected = False
        self._la_report_cooldown_armed = False
        self._ll_report_cooldown_armed = False
        # Reset the packet-parser RL chatter history for a fresh session.
        try:
            parse_packet.__globals__["_rl_chatter_history"] = []
        except Exception:
            pass
        # Fresh session: allow the electrode-disconnected popup to fire again
        # if RA/RL drop out during this run.
        self._ra_rl_popup_shown = False
        self._set_lead_status_idle()
        self._frozen_report_snapshot = None
        self._frozen_report_metrics = None
        
        self.data = [np.zeros(HISTORY_LENGTH, dtype=np.float32) for _ in range(12)]
        self.last_heart_rate = 0
        self.last_pr_interval = 0
        self.last_qrs_duration = 0
        self.last_qt_interval = 0
        self.last_qtc_interval = 0
        self._latest_rhythm_interpretation = "Analyzing Rhythm..."
        
        # Reset metric labels!
        if hasattr(self, 'metric_labels'):
            if 'heart_rate' in self.metric_labels:
                self.metric_labels['heart_rate'].setText("--")
            if 'pr_interval' in self.metric_labels:
                self.metric_labels['pr_interval'].setText("--")
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setText("--")
            if 'qtc_interval' in self.metric_labels:
                self.metric_labels['qtc_interval'].setText("--/--")

        # --- START HOLTER SESSION IF ENABLED ---
        if self.holter_mode_enabled:
            self._start_holter_session_internal()
        # ---------------------------------------

        # If user has not configured a port/baud, fall back to auto‑scan instead of blocking
        if port in ("Select Port", None):
            print(" No COM port configured in System Setup – will auto‑scan all ports for ECG device.")
        if baud in ("Select Baud Rate", None):
            print(" No baud rate configured in System Setup – will use default 115200 for auto‑scan.")

        # Try to list available ports for logging only (do not block auto‑scan)
        try:
            available_ports = [p.device for p in serial.tools.list_ports.comports()]
            print(f" Available COM ports: {available_ports}")
            if port and port not in ("Select Port",) and port not in available_ports:
                print(f" Configured port {port} not present – will rely on auto‑scan.")
        except Exception as ports_err:
            print(f" Warning: could not list COM ports: {ports_err} (will still auto‑scan)")
        
        try:
            # Convert baud rate to integer with error handling
            try:
                baud_int = int(baud)
            except (ValueError, TypeError):
                self.show_connection_warning(f"Invalid baud rate: {baud}.")
                return
            
            if self.serial_reader:
                self.serial_reader.close()
            
            print(f"Connecting to {port} at {baud_int} baud...")

            # Check if this is a fresh start (no existing serial reader) or a restart
            # Only reset metrics to zero on fresh start, not on restart after stop
            is_fresh_start = (self.serial_reader is None)
            if is_fresh_start and hasattr(self, "_report_buffers"):
                for buf in self._report_buffers:
                    buf.clear()
                if hasattr(self, "_replay_buffers"):
                    for buf in self._replay_buffers:
                        buf.clear()
                
                # Only reset visible metrics to zero on fresh start to avoid losing machine serial data values on restart
                try:
                    if hasattr(self, 'metric_labels'):
                        if 'heart_rate' in self.metric_labels: self.metric_labels['heart_rate'].setText("--")
                        if 'pr_interval' in self.metric_labels: self.metric_labels['pr_interval'].setText("--")
                        if 'qrs_duration' in self.metric_labels: self.metric_labels['qrs_duration'].setText("--")
                        if 'st_interval' in self.metric_labels: self.metric_labels['st_interval'].setText("--")
                        if 'qtc_interval' in self.metric_labels: self.metric_labels['qtc_interval'].setText("--/--")
                        if 'time_elapsed' in self.metric_labels: self.metric_labels['time_elapsed'].setText("00:00")
                        # if 'sampling_rate' in self.metric_labels: self.metric_labels['sampling_rate'].setText("0 Hz")  # Commented out
                    print(" Fresh start - metrics reset to zero")
                except Exception as _:
                    pass
            else:
                print(" Restart - preserving existing metric values from machine serial data")
            
            # --- NEW: Scan all COM ports with START command and pick the one that ACKs ---
            port_to_use = port
            scan_needed = (port in ("Select Port", None) or port == "")
            
            # If we have a configured port, verify it's physically available
            if not scan_needed:
                try:
                    available_ports = [p.device for p in serial.tools.list_ports.comports()]
                    if port not in available_ports:
                        print(f" Configured port {port} not found in available ports. forcing scan.")
                        scan_needed = True
                except Exception:
                    # If we can't list ports, assume we need to scan or just try the configured one
                    pass

            # ─────────────────────────────────────────────────────────────────
            # INSTANT FEEDBACK: disable button & show “Connecting…” right now,
            # before any blocking I/O. The actual port scan + VERSION + START
            # happen in DeviceStartWorker (background QThread).
            # ─────────────────────────────────────────────────────────────────
            self.start_btn.setEnabled(False)
            self.start_btn.setText("Connecting…")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: #ffe0b2;
                    color: #e65100;
                    border: 2px solid #ff6600;
                    border-radius: 6px;
                    padding: 4px 8px;
                    font-size: 10px;
                    font-weight: bold;
                    text-align: center;
                }
            """)

            from ecg.serial.serial_reader import DeviceStartWorker
            self._start_worker = DeviceStartWorker(
                port=port,
                baud_int=baud_int,
                reader=self.serial_reader,
            )
            self._start_worker.version_ready.connect(self._on_device_version)
            self._start_worker.serial_ready.connect(self._on_device_serial_number)
            self._start_worker.connected.connect(
                lambda ok, p, err: self._on_device_connected(ok, p, err, baud_int)
            )
            self._start_worker.start()

        except Exception as e:
            error_msg = f"Failed to start connection worker: {str(e)}"
            print(error_msg)
            # Restore button on unexpected pre-worker error
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start")
            self.show_connection_warning(error_msg)

    def _on_device_version(self, version: str):
        """Slot: device version received from background worker."""
        if version:
            print(f" 🧬 ECG Device Version: {version}")
            # Update version label if it exists on the UI
            if hasattr(self, 'version_label'):
                try:
                    self.version_label.setText(version)
                except Exception:
                    pass

    def _on_device_serial_number(self, serial_number: str):
        """Slot: machine serial number received from background worker."""
        if not serial_number:
            return

        print(f" 🏷️ ECG Machine Serial Number: {serial_number}")

        # Persist it so other UI panels (e.g., Version Information) can display it.
        try:
            if hasattr(self, "settings_manager") and self.settings_manager:
                self.settings_manager.set_setting("machine_serial_number", serial_number)
        except Exception:
            pass

    def _on_device_connected(self, success: bool, port_to_use: str, error_msg: str, baud_int: int):
        """
        Slot: called (on main thread via signal) once the background worker
        finishes. Continues the original post-connect logic from start_acquisition.
        """
        if not success:
            print(f" [start] Connection failed: {error_msg}")
            # Re-enable Start button so the user can retry
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4CAF50, stop:1 #45a049);
                    color: white;
                    border: 2px solid #4CAF50;
                    border-radius: 6px;
                    padding: 4px 8px;
                    font-size: 10px;
                    font-weight: bold;
                    text-align: center;
                }
            """)
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                self.dashboard_instance.update_test_state("12_lead_test", False)
            self.show_connection_warning(f"Failed to connect to any serial port: {error_msg}")
            return

        # ── Connection successful: pull the now-started reader from the manager ──
        from ecg.serial.serial_reader import GlobalHardwareManager
        self.serial_reader = GlobalHardwareManager().get_reader(port_to_use, baud_int)

        # Centralized authorization check
        from utils.license_manager import is_ecg_acquisition_allowed
        if not is_ecg_acquisition_allowed(self):
            # Block and reset UI
            self.start_btn.setEnabled(True)
            self.start_btn.setText("Start")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4CAF50, stop:1 #45a049);
                    color: white;
                    border: 2px solid #4CAF50;
                    border-radius: 6px;
                    padding: 4px 8px;
                    font-size: 10px;
                    font-weight: bold;
                    text-align: center;
                }
            """)
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                self.dashboard_instance.update_test_state("12_lead_test", False)
            if self.serial_reader:
                self.serial_reader.stop()
            return

        if self.serial_reader and hasattr(self, 'user_details'):
            self.serial_reader.user_details = self.user_details

        # Persist discovered port so next Start skips the scan
        if hasattr(self, 'settings_manager') and port_to_use:
            self.settings_manager.set_serial_port(port_to_use)

        print(f" Serial connection established successfully on {port_to_use}!")

        # ── Start timers (was right after serial_reader.start() in original code) ──
        # Match the cadence the HRV and hyperkalemia pages already use. This page
        # is the heaviest of the three and was the only one with bare literals,
        # so on a weak machine it ran the most expensive callbacks the fastest.
        timer_interval = 50 if is_low_spec_mode() else 30   # ~20 / ~33 FPS
        self._metrics_base_interval = 500 if is_low_spec_mode() else 200
        self.timer.start(timer_interval)
        self.metrics_timer.start(self._metrics_base_interval)
        QTimer.singleShot(10, self.update_plots)   # instant first frame

        # ── Start HolterBPMController (stable BPM engine) ─────────────────────
        try:
            if self._bpm_ctrl is not None:
                if self._bpm_ctrl.is_running:
                    self._bpm_ctrl.stop()
                self._bpm_ctrl.start(target_hours=0)
                # Attach display bar at the top of main layout if not already there
                try:
                    main_layout = self.layout()
                    if main_layout and self._bpm_ctrl.display_bar is not None:
                        existing = [main_layout.itemAt(i).widget()
                                    for i in range(main_layout.count())
                                    if main_layout.itemAt(i).widget() is not None]
                        if self._bpm_ctrl.display_bar not in existing:
                            main_layout.insertWidget(0, self._bpm_ctrl.display_bar)
                        self._bpm_ctrl.display_bar.show()
                except Exception:
                    pass

                # ★ 3-second BPM refresh timer — ONLY source that writes to HR label
                if not hasattr(self, '_bpm_refresh_timer'):
                    self._bpm_refresh_timer = QTimer()
                    self._bpm_refresh_timer.timeout.connect(self._refresh_holter_bpm_label)
                if not self._bpm_refresh_timer.isActive():
                    self._bpm_refresh_timer.start(2000)
        except Exception as _bpm_start_err:
            print(f"[ECGTestPage] BPM controller start error: {_bpm_start_err}")
        if hasattr(self, '_12to1_timer'):
            self._12to1_timer.start(100)

        # Reset metric update timestamps for immediate metric updates
        if hasattr(self, '_last_metric_update_ts'):
            self._last_metric_update_ts = 0.0
        if hasattr(self, '_metrics_calculated_once'):
            delattr(self, '_metrics_calculated_once')
        if hasattr(self, '_metrics_update_count'):
            self._metrics_update_count = 0

        # Sync dashboard
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            if hasattr(self.dashboard_instance, '_last_metrics_update_ts'):
                self.dashboard_instance._last_metrics_update_ts = 0.0
            if hasattr(self.dashboard_instance, '_inactive_update_count'):
                self.dashboard_instance._inactive_update_count = 0
            try:
                self.dashboard_instance.update_dashboard_metrics_from_ecg()
                print("✅ Immediate dashboard update triggered on acquisition start")
            except Exception as e:
                print(f"⚠️ Dashboard immediate update failed: {e}")

        self.update_recording_button_state()
        print(f"[DEBUG] ECGTestPage - Timer active: {self.timer.isActive()}")
        print(f"[DEBUG] ECGTestPage - Number of leads: {len(self.leads)}")
        print(f"[DEBUG] ECGTestPage - Number of plot widgets: {len(self.plot_widgets)}")
        print(f"[DEBUG] ECGTestPage - Number of data lines: {len(self.data_lines)}")

        # Start / resume elapsed time
        current_time = time.time()
        if not hasattr(self, 'start_time') or self.start_time is None:
            self.start_time = current_time
            if hasattr(self, 'paused_duration'):
                self.paused_duration = 0
            self.paused_at = None
            print(" Session timer started (first time)")
        else:
            if hasattr(self, 'paused_at') and self.paused_at is not None:
                pause_duration = current_time - self.paused_at
                if not hasattr(self, 'paused_duration') or self.paused_duration is None:
                    self.paused_duration = 0
                self.paused_duration += pause_duration
                print(f" Session timer resumed (was paused for {int(pause_duration)}s)")
                self.paused_at = None
            else:
                print(" Session timer resumed")
        if self.elapsed_timer.isActive():
            self.elapsed_timer.stop()
        self.elapsed_timer.start(1000)

        # Disable Start, enable Stop with grey / green styles
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Start")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #e0e0e0;
                color: #a0a0a0;
                border: 2px solid #cccccc;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
        """)

        green_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3d8b40, stop:1 #357a38);
                border: 2px solid #3d8b40;
                color: white;
            }
        """
        self.stop_btn.setEnabled(True)
        self.stop_btn.setStyleSheet(green_style)
        self._style_freeze_button("Freeze", enabled=True)

        # Enable Generate Report button
        try:
            self.generate_report_btn.setEnabled(True)
            self.generate_report_btn.setStyleSheet(green_style)
            self._refresh_report_btn_label()
        except Exception:
            pass

        # Enforce 10-second cooldown after Start before first report generation.
        self._start_generate_report_cooldown(seconds=10, reason="Start")


    def _on_metrics_timer_timeout(self):
        """Dedicated timer callback to update ECG metrics at a stable rate (200ms)"""
        # Do not recalculate metrics while report is generating or view is frozen.
        if getattr(self, '_report_generating', False) or getattr(self, '_grid_frozen', False):
            # While frozen: restore snapshot texts so label flushes do not overwrite them
            _ftexts = getattr(self, '_frozen_metric_texts', {})
            if _ftexts:
                for _k, _txt in _ftexts.items():
                    try:
                        lbl = self.metric_labels.get(_k)
                        if lbl and lbl.text() != _txt:
                            lbl.setText(_txt)
                    except Exception:
                        pass
            return

        _t0 = time.perf_counter()
        try:
            self.calculate_ecg_metrics()
            if hasattr(self, 'last_heart_rate') and self.last_heart_rate > 0:
                self._last_calculated_bpm = self.last_heart_rate
        except Exception:
            pass

        # Never own more than a quarter of our own period. Metrics are a 5 Hz
        # cosmetic readout over a 20 s window; the plot timer sharing this thread
        # is what the user actually watches. If the callback costs 90 ms it is
        # rescheduled at 360 ms rather than 200 ms, so it stops deleting plot
        # frames on a machine that cannot afford it. The same duty-cycle idiom is
        # used for the global QRS refresh in ecg_calculations.py.
        try:
            _elapsed_ms = (time.perf_counter() - _t0) * 1000.0
            _base = int(getattr(self, '_metrics_base_interval', 200))
            _want = max(_base, int(_elapsed_ms / 0.25))
            _want = min(_want, 2000)          # never slower than 0.5 Hz
            if abs(_want - self.metrics_timer.interval()) >= 50:
                self.metrics_timer.setInterval(_want)
        except Exception:
            pass

    # ---------------------- Stop Button Functionality ----------------------

    def _refresh_holter_bpm_label(self):
        """Called by _bpm_refresh_timer.
        Keep the on-screen BPM synchronized with RR-derived BPM from the ECG
        metrics pipeline (single source of truth).
        """
        try:
            # ── FREEZE GUARD: do not overwrite BPM label while display is frozen ──
            if getattr(self, '_grid_frozen', False):
                # Re-paint the frozen BPM text in case another path cleared it
                _ft = getattr(self, '_frozen_metric_texts', {})
                if _ft and 'heart_rate' in _ft:
                    try:
                        lbl = self.metric_labels.get('heart_rate')
                        if lbl and lbl.text() != _ft['heart_rate']:
                            lbl.setText(_ft['heart_rate'])
                    except Exception:
                        pass
                return
            # ──────────────────────────────────────────────────────────────
            if self._bpm_ctrl is None or not self._bpm_ctrl.is_running:
                return

            # Only force BPM to 0 when the primary limb leads are actually lost.
            # Partial lead disconnects should keep the last stable BPM visible so the
            # dashboard does not flicker between 0 and the real value.
            # _ll_disconnected is a fast per-packet flag (no debounce) so LL removal
            # is caught immediately, before _lead_connection_state['II'] flips.
            limb_conn = getattr(self, "_lead_connection_state", {})
            limb_active = limb_conn.get('I', True) or limb_conn.get('II', True)
            ll_off = getattr(self, '_ll_disconnected', False)
            if getattr(self, "_lead_off_latched", False) or not limb_active or ll_off:
                # Reset internal state so the BPM engine stops propagating values,
                # but do NOT write "0" to the label here — update_metrics owns that
                # write via its flatline path. Writing it from two timer callbacks
                # at different rates is what causes the 0↔BPM flicker on display.
                try:
                    self.last_rr_interval = 0
                except Exception:
                    pass
                try:
                    self.last_heart_rate = 0
                except Exception:
                    pass
                return

            rr_ms = getattr(self, 'last_rr_interval', 0)
            bpm = int(round(60000.0 / rr_ms)) if rr_ms and rr_ms > 0 else 0
            if bpm <= 0:
                bpm = int(round(self._bpm_ctrl.current_bpm()))

            if bpm > 0 and hasattr(self, 'metric_labels') and 'heart_rate' in self.metric_labels:
                self.metric_labels['heart_rate'].setText(f"{bpm:3d}")
                # Keep last_heart_rate in sync so reports get the right value
                self.last_heart_rate = bpm
        except Exception as _e:
            print(f"[ECGTestPage] _refresh_holter_bpm_label error: {_e}")

    def _cleanup_report_generation_thread(self, wait_ms: int = 5000, force_terminate: bool = True):
        """Stop the active report-generation thread safely before teardown."""
        thread = getattr(self, "_report_thread", None)
        if thread is None:
            return
        try:
            if thread.isRunning():
                try:
                    thread.quit()
                except Exception:
                    pass
                try:
                    thread.wait(wait_ms)
                except Exception:
                    pass
            if force_terminate and thread.isRunning():
                try:
                    thread.terminate()
                    thread.wait(1000)
                except Exception:
                    pass
        finally:
            try:
                self._report_thread = None
                self._report_worker = None
            except Exception:
                pass

    def _cleanup_start_worker(self, wait_ms: int = 5000, force_terminate: bool = True):
        """Stop the connection worker safely before teardown."""
        worker = getattr(self, "_start_worker", None)
        if worker is None:
            return
        try:
            if worker.isRunning():
                try:
                    worker.requestInterruption()
                except Exception:
                    pass
                try:
                    worker.quit()
                except Exception:
                    pass
                try:
                    worker.wait(wait_ms)
                except Exception:
                    pass
            if force_terminate and worker.isRunning():
                try:
                    worker.terminate()
                    worker.wait(1000)
                except Exception:
                    pass
        finally:
            try:
                self._start_worker = None
            except Exception:
                pass

    def stop_acquisition(self):
        # UPDATE STATE: Test stopped
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            self.dashboard_instance.update_test_state("12_lead_test", False)

        self._cleanup_start_worker(wait_ms=3000, force_terminate=True)

        # --- STOP HOLTER SESSION IF ENABLED ---
        if self.holter_mode_enabled:
            self._stop_holter_session_internal()
        # --------------------------------------

        # ── Reset all metrics and interpretation when device disconnects ──────
        self._latest_rhythm_interpretation = "Rhythm Undetermined"
        
        # Reset twelve lead test's own metric labels
        if hasattr(self, 'metric_labels'):
            if 'heart_rate' in self.metric_labels:
                self.metric_labels['heart_rate'].setText("--")
            if 'pr_interval' in self.metric_labels:
                self.metric_labels['pr_interval'].setText("--")
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setText("--")
            if 'qtc_interval' in self.metric_labels:
                self.metric_labels['qtc_interval'].setText("--/--")
        
        # Clear all wave plot data (buffers + visible curves)
        if hasattr(self, 'data'):
            self.data = [np.zeros(HISTORY_LENGTH, dtype=np.float32) for _ in range(12)]

        # Visually wipe every plot curve immediately so old waves are not
        # visible when the user reconnects the device for a fresh session.
        if hasattr(self, 'data_lines'):
            for line in self.data_lines:
                try:
                    line.setData([], [])
                except Exception:
                    pass

        # Clear report/replay buffers so reconnection truly starts fresh
        if hasattr(self, '_report_buffers'):
            for buf in self._report_buffers:
                try:
                    buf.clear()
                except Exception:
                    pass
        if hasattr(self, '_replay_buffers'):
            for buf in self._replay_buffers:
                try:
                    buf.clear()
                except Exception:
                    pass

        # Reset last values
        self.last_heart_rate = 0
        self.last_pr_interval = 0
        self.last_qrs_duration = 0
        self.last_qt_interval = 0
        self.last_qtc_interval = 0
        
        # Update dashboard if available
        if hasattr(self, 'dashboard_callback') and self.dashboard_callback:
            self.dashboard_callback({
                'heart_rate': 0,
                'pr_interval': 0,
                'qrs_duration': 0,
                'qtc_interval': 0,
                'st_interval': 0
            })
        
        # Also, if dashboard_instance exists, tell it to reset
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            if hasattr(self.dashboard_instance, 'reset_metrics_and_interpretation'):
                self.dashboard_instance.reset_metrics_and_interpretation()

        # ── Stop HolterBPMController ──────────────────────────────────────────
        try:
            # Stop the 3-second refresh timer first
            if hasattr(self, '_bpm_refresh_timer') and self._bpm_refresh_timer.isActive():
                self._bpm_refresh_timer.stop()
            if self._bpm_ctrl is not None and self._bpm_ctrl.is_running:
                self._bpm_ctrl.stop()
                if self._bpm_ctrl.display_bar is not None:
                    self._bpm_ctrl.display_bar.hide()
        except Exception as _bpm_stop_err:
            print(f"[ECGTestPage] BPM controller stop error: {_bpm_stop_err}")

        port = self.settings_manager.get_serial_port()
        baud = self.settings_manager.get_baud_rate()
            
        if self.serial_reader:
            self.serial_reader.stop()
            # self.serial_reader.close()
            # self.serial_reader = None
        self.timer.stop()
        if hasattr(self, 'metrics_timer') and self.metrics_timer.isActive():
            self.metrics_timer.stop()
        # Cancel any active countdown timers for generate report button
        for timer in self.countdown_timers:
            if hasattr(timer, "stop") and timer.isActive():
                timer.stop()
        self.countdown_timers.clear()
        if hasattr(self, '_12to1_timer'):
            self._12to1_timer.stop()
        try:
            if hasattr(self, "_replay_timer") and self._replay_timer.isActive():
                self._replay_timer.stop()
        except Exception:
            pass
        self._display_mode = "live"
        self._grid_frozen = False
        self._replay_cursor = 0
        self._replay_render_snapshot = None

        # Update recording button state now that acquisition has stopped
        self.update_recording_button_state()
        self._capture_report_snapshot(window_sec=7.0)
        self._update_generate_report_button_state()

        # Pause elapsed time tracking (keep start_time for resume)
        self.elapsed_timer.stop()
        
        # RESET TIMER: When device disconnects, we want the timer to start from 00:00 next time.
        # Check if this stop is due to a device disconnection
        is_disconnected = False
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            if not getattr(self.dashboard_instance, 'device_connected', True):
                is_disconnected = True
        
        if is_disconnected:
            self.start_time = None
            self.paused_duration = 0
            self.paused_at = None
            if 'time_elapsed' in self.metric_labels:
                self.metric_labels['time_elapsed'].setText("00:00")
            print(" Session timer reset due to device disconnection")
        else:
            # Track when pause started (for calculating total paused time on resume)
            if hasattr(self, 'start_time') and self.start_time is not None:
                if not hasattr(self, 'paused_at') or self.paused_at is None:
                    self.paused_at = time.time()
                    print(f" Timer paused")
                # Keep start_time so we can resume from this point

        # --- Calculate and update metrics on dashboard ---
        try:
            if hasattr(self, 'dashboard_callback'):
                # Fixed Bug: get_current_metrics_from_labels requires (metric_labels, data, last_heart_rate, sampler)
                # However, display_updates.py shows it actually needs (metric_labels, data, last_heart_rate=None, sampler=None)
                from .ui.display_updates import get_current_metrics_from_labels
                
                # Use data from the screen if possible, with safety fallbacks
                current_metrics = get_current_metrics_from_labels(
                    getattr(self, 'metric_labels', {}),
                    getattr(self, 'data', []),
                    sampler=getattr(self, 'sampler', None)
                )
                
                self.dashboard_callback({
                    'heart_rate': current_metrics.get('heart_rate'),
                    'pr_interval': current_metrics.get('pr_interval'),
                    'qrs_duration': current_metrics.get('qrs_duration'),
                    'qtc_interval': current_metrics.get('qtc_interval'),
                    'st_interval': current_metrics.get('st_interval')
                })
        except Exception as e:
            print(f"⚠️ Error updating dashboard metrics during stop: {e}")
        
        # Re-enable demo mode when hardware acquisition stops
        try:
            if hasattr(self, 'demo_toggle'):
                self.demo_toggle.setEnabled(True)
                self.demo_toggle.setStyleSheet("""
                    QPushButton {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                            stop:0 #ffffff, stop:1 #f8f9fa);
                        color: #1a1a1a;
                        border: 2px solid #e9ecef;
                        border-radius: 8px;
                        padding: 8px 12px;
                        font-size: 12px;
                        font-weight: bold;
                        text-align: center;
                        margin: 2px 0;
                    }
                    QPushButton:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                            stop:0 #fff5f0, stop:1 #ffe0cc);
                        border: 2px solid #ff6600;
                        color: #ff6600;
                    }
                    QPushButton:checked {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                            stop:0 #fff5f0, stop:1 #ffe0cc);
                        border: 2px solid #dc3545;
                        color: #dc3545;
                    }
                """)
                print(" Demo mode enabled (Hardware acquisition stopped)")
        except Exception as e:
            print(f" Error enabling demo mode: {e}")

        # Cancel and clear all countdown timers
        for timer in self.countdown_timers:
            if hasattr(timer, 'stop') and timer.isActive():
                timer.stop()
        self.countdown_timers.clear()
        
        self._update_generate_report_button_state()

        # Disable Stop button
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #e0e0e0;
                color: #a0a0a0;
                border: 2px solid #cccccc;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
        """)

        # Re-enable Start button and restore green style
        self.start_btn.setEnabled(True)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: bold;
                text-align: center;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
                color: white;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #3d8b40, stop:1 #357a38);
                border: 2px solid #3d8b40;
                color: white;
            }
        """)

        # Reset the lead banner so a stale green status does not survive into the next session.
        self._lead_data_received = False
        self._set_lead_status_idle()

    def stop_holter_recording(self):
        """Public wrapper to stop Holter recording when user switches to HRV or Hyperkalemia tests"""
        if self.holter_mode_enabled:
            try:
                self._stop_holter_session_internal()
                print("[ECGTestPage] Holter recording stopped by user action (switched to HRV/Hyperkalemia)")
            except Exception as e:
                print(f"[ECGTestPage] Error stopping Holter recording: {e}")
            finally:
                self.holter_mode_enabled = False

    def update_plot(self):
        print(f"[DEBUG] ECGTestPage - update_plot called, serial_reader exists: {self.serial_reader is not None}")
        
        if not self.serial_reader:
            print("[DEBUG] ECGTestPage - No serial reader, returning")
            return
        
        # Read raw data directly from serial port
        try:
            line = self.serial_reader.ser.readline()
            line_data = line.decode('utf-8', errors='replace').strip()
            
            if not line_data:
                print("[DEBUG] ECGTestPage - No data received (empty line)")
                return
            
            print(f"[DEBUG] ECGTestPage - Raw hardware data: '{line_data}' (length: {len(line_data)})")
            
            # Parse the 8-channel data (handle multiple spaces)
            try:
                # Split by any whitespace and filter out empty strings
                values = [int(x) for x in line_data.split() if x.strip()]
                print(f"[DEBUG] ECGTestPage - Parsed {len(values)} values: {values}")
                
                if len(values) >= 8:
                    # Extract individual leads from 8-channel data
                    lead1 = values[0]    # Lead I
                    v4    = values[1]    # V4
                    v5    = values[2]    # V5
                    lead2 = values[3]    # Lead II
                    v3    = values[4]    # V3
                    v6    = values[5]    # V6
                    v1    = values[6]    # V1
                    v2    = values[7]    # V2
                    
                    # Calculate derived leads
                    lead3 = lead2 - lead1
                    avr = - (lead1 + lead2) / 2
                    avl = (lead1 - lead3) / 2
                    avf = (lead2 + lead3) / 2
                    
                    lead_data = {
                        "I": lead1, "II": lead2, "III": lead3,
                        "aVR": avr, "aVL": avl, "aVF": avf,
                        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6
                    }
                    
                    print(f"[DEBUG] ECGTestPage - Successfully parsed 8-channel data: {lead_data}")
                    
                elif len(values) == 1:
                    # Single value - generate realistic 12-lead ECG data
                    ecg_value = values[0]
                    print(f"[DEBUG] ECGTestPage - Single value received: {ecg_value}, generating realistic ECG...")
                    
                    # Initialize realistic ECG generation if not already done
                    if not hasattr(self, 'ecg_generators'):
                        self.ecg_generators = {}
                        self.ecg_time_index = 0
                        self.ecg_sampling_rate = self.demo_fs
                        
                        # Generate realistic ECG waveforms for each lead
                        for lead in self.leads:
                            ecg_wave, _ = generate_realistic_ecg_waveform(
                                duration_seconds=60,  # 1 minute of data
                                sampling_rate=self.ecg_sampling_rate,
                                heart_rate=72,
                                lead_name=lead
                            )
                            self.ecg_generators[lead] = ecg_wave
                    
                    # Get current sample from realistic ECG waveforms
                    lead_data = {}
                    for lead in self.leads:
                        if lead in self.ecg_generators:
                            # Scale the realistic ECG to match the input value range
                            realistic_value = self.ecg_generators[lead][self.ecg_time_index % len(self.ecg_generators[lead])]
                            # Scale to match typical ECG range (0-4095 for 12-bit ADC)
                            scaled_value = int(ecg_value + realistic_value * 1000)  # Scale realistic ECG to mV range
                            lead_data[lead] = scaled_value
                    
                    # Move to next time sample
                    self.ecg_time_index += 1
                    
                else:
                    print(f"[DEBUG] ECGTestPage - Unexpected number of values: {len(values)}")
                    return
                    
            except ValueError as e:
                print(f"[DEBUG] ECGTestPage - Error parsing values: {e}")
                # Try to extract numeric part using regex
                import re
                numbers = re.findall(r'-?\d+', line_data)
                if numbers:
                    try:
                        # Use first number as single value
                        ecg_value = int(numbers[0])
                        print(f"[DEBUG] ECGTestPage - Extracted numeric value: {ecg_value}")
                        
                        # Use single value to generate realistic 12-lead ECG data
                        # Initialize realistic ECG generation if not already done
                        if not hasattr(self, 'ecg_generators'):
                            self.ecg_generators = {}
                            self.ecg_time_index = 0
                            self.ecg_sampling_rate = self.demo_fs
                            
                            # Generate realistic ECG waveforms for each lead
                            for lead in self.leads:
                                ecg_wave, _ = generate_realistic_ecg_waveform(
                                    duration_seconds=60,  # 1 minute of data
                                    sampling_rate=self.ecg_sampling_rate,
                                    heart_rate=72,
                                    lead_name=lead
                                )
                                self.ecg_generators[lead] = ecg_wave
                        
                        # Get current sample from realistic ECG waveforms
                        lead_data = {}
                        for lead in self.leads:
                            if lead in self.ecg_generators:
                                # Scale the realistic ECG to match the input value range
                                realistic_value = self.ecg_generators[lead][self.ecg_time_index % len(self.ecg_generators[lead])]
                                # Scale to match typical ECG range (0-4095 for 12-bit ADC)
                                scaled_value = int(ecg_value + realistic_value * 1000)  # Scale realistic ECG to mV range
                                lead_data[lead] = scaled_value
                        
                        # Move to next time sample
                        self.ecg_time_index += 1
                    except ValueError:
                        print(f"[DEBUG] ECGTestPage - Could not parse numeric data from: '{line_data}'")
                        return
                else:
                    print(f"[DEBUG] ECGTestPage - No numeric data found in: '{line_data}'")
                    return
            
            # Update data buffers for all leads
            for lead in self.leads:
                if lead in lead_data:
                    self.data[lead].append(lead_data[lead])
                    if len(self.data[lead]) > self.buffer_size:
                        self.data[lead].pop(0)
            
            print(f"[DEBUG] ECGTestPage - Updated data buffers, Lead II has {len(self.data['II'])} points")
            
            # Write latest Lead II data to file for dashboard
            try:
                import json
                with open('lead_ii_live.json', 'w') as f:
                    json.dump(self.data["II"][-500:], f)
            except Exception as e:
                print("Error writing lead_ii_live.json:", e)
            
            # Calculate and update ECG metrics in real-time
            lead_ii_data = self.data.get("II", [])
            if lead_ii_data:
                intervals = self.calculate_ecg_intervals(lead_ii_data)
                self.update_ecg_metrics_on_top_of_lead_graphs(intervals)
            
            # Update all plots
            for i, lead in enumerate(self.leads):
                if len(self.data[lead]) > 0:
                    print(f"[DEBUG] ECGTestPage - Updating plot for {lead}: {len(self.data[lead])} data points")
                    
                    # Prepare plot data - ALWAYS show oldest data on left, newest on right (left-to-right scrolling)
                    # This ensures waves always move left to right regardless of BPM
                    if len(self.data[lead]) < self.buffer_size:
                        # Buffer not full yet - pad with NaN on the left, data on the right
                        data = np.full(self.buffer_size, np.nan)
                        data[-len(self.data[lead]):] = self.data[lead]
                    else:
                        # Buffer is full - use all data (oldest to newest, left to right)
                        data = np.array(self.data[lead])
                    
                    # Convert device data to ECG range and center around zero
                    device_data = np.array(data)
                    # Scale to typical ECG range (subtract baseline ~2100 and scale)
                    # Calculate gain factor: higher mm/mV = higher gain (10mm/mV = 1.0x baseline)
                    gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                    centered = (device_data - 2100) * gain_factor
                    
                    # Apply noise reduction filtering
                    filtered_data = self.apply_ecg_filtering(centered)
                    
                    # Update the plot line - ALWAYS ensure left-to-right scrolling
                    if i < len(self.lines):
                        # Explicitly set x-axis data to ensure left-to-right direction (0 to buffer_size)
                        x_data = np.arange(len(filtered_data), dtype=float)
                        self.lines[i].set_data(x_data, filtered_data)
                        print(f"[DEBUG] ECGTestPage - Updated {lead} plot with {len(centered)} points, range: {np.min(centered):.2f} to {np.max(centered):.2f}")
                        
                        # Use dynamic y-limits based on current gain setting
                        ylim = self.ylim if hasattr(self, 'ylim') else 400
                        if i < len(self.axs):
                            self.axs[i].set_ylim(-ylim, ylim)
                            
                            # ALWAYS set x-limits from 0 to buffer_size to ensure left-to-right scrolling
                            # This ensures waves always move left to right regardless of BPM
                            self.axs[i].set_xlim(0, self.buffer_size)

                            # Update title with current settings
                            current_speed = 25.0 if getattr(self, '_overlay_active', False) else self.settings_manager.get_wave_speed()
                            current_gain = self.settings_manager.get_wave_gain()
                            self.axs[i].set_title(f"{lead} | Speed: {current_speed}mm/s | Gain: {current_gain}mm/mV", 
                                                fontsize=8, color='#666', pad=10)
                            
                            # Add grid lines to show scale
                            self.axs[i].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                            
                            # Remove any existing labels
                            self.axs[i].set_xlabel("")
                            self.axs[i].set_ylabel("")
                        
                        # Force redraw of the canvas
                        if i < len(self.canvases):
                            self.canvases[i].draw_idle()
                    else:
                        print(f"[DEBUG] ECGTestPage - Warning: No line object for lead {lead} at index {i}")
                    
        except Exception as e:
            print(f"[DEBUG] ECGTestPage - Error in update_plot: {e}")
            import traceback
            traceback.print_exc()

    def _refresh_report_btn_label(self):
        """Update Generate Report button label with selected format and remaining countdown."""
        try:
            if hasattr(self, 'generate_report_btn') and hasattr(self, 'settings_manager'):
                _f   = self.settings_manager.get_setting('report_format', '12_1') or '12_1'
                _lbl = {"12_1": "12:1", "6_2": "6:2", "4_3": "4:3"}.get(_f, "12:1")
                rem = 0
                if getattr(self, "_start_cooldown_active", False):
                    rem = max(0, getattr(self, "_cooldown_seconds_remaining", 0))
                elif hasattr(self, 'start_time') and self.start_time and hasattr(self, 'timer') and self.timer and self.timer.isActive():
                    import time
                    cur_t = time.time()
                    p_dur = getattr(self, 'paused_duration', 0)
                    elapsed_sec = int(max(0, cur_t - self.start_time - p_dur))
                    if elapsed_sec < 10:
                        rem = 10 - elapsed_sec

                if rem > 0:
                    self.generate_report_btn.setText(f"Generate Report ({_lbl}) ({rem}s)")
                else:
                    self.generate_report_btn.setText(f"Generate Report ({_lbl})")
        except Exception:
            pass

    def generate_pdf_report(self):
        """Generate PDF using ecg_report_android — all formats, non-blocking."""
        from PyQt5.QtCore import QThread, pyqtSignal, QObject
        import datetime, os, numpy as np

        # Read format from settings
        fmt = "12_1"
        try:
            if hasattr(self, 'settings_manager') and self.settings_manager:
                fmt = self.settings_manager.get_setting('report_format', '12_1') or '12_1'
        except Exception:
            pass
        fmt_label = {"12_1": "12:1", "6_2": "6:2", "4_3": "4:3"}.get(fmt, "12:1")

        # Prevent crash if thread is already running
        if hasattr(self, "_report_thread") and self._report_thread is not None:
            try:
                if self._report_thread.isRunning():
                    print("⚠️ A report generation thread is already running. Please wait.")
                    return
            except Exception:
                pass

        self._report_generating = True

        # Snapshot ECG arrays (< 5ms on main thread).
        # If we are frozen, reuse the frozen snapshot; otherwise always capture
        # a fresh live window so the PDF reflects the current waveform + metrics.
        try:
            display_mode = getattr(self, "_display_mode", "live")
            frozen_snapshot = getattr(self, "_frozen_report_snapshot", None)
            if display_mode == "frozen" and frozen_snapshot and any(np.asarray(arr).size > 0 for arr in frozen_snapshot):
                snap_raw = [np.asarray(arr, dtype=float).copy() for arr in frozen_snapshot]
            else:
                # Capture exact 10 seconds of waveform data for all report formats
                _snap_sec = 10.0
                snap_raw = self._capture_report_snapshot(window_sec=_snap_sec)
        finally:
            self._report_generating = False

        # Freeze metrics (all reads from self — safe on main thread)
        _sm = self.settings_manager if hasattr(self, 'settings_manager') and self.settings_manager else None
        # Check if demo mode is active and use hardcoded demo values
        is_demo_mode = bool(hasattr(self, 'demo_toggle') and self.demo_toggle and self.demo_toggle.isChecked())
        
        display_mode = getattr(self, "_display_mode", "live")
        frozen_report_metrics = getattr(self, "_frozen_report_metrics", None)

        if display_mode == "frozen" and frozen_report_metrics:
            frozen = frozen_report_metrics.copy()
            frozen['lead_seq'] = (_sm.get_setting('lead_sequence', 'Standard') if _sm else 'Standard')
            frozen['logo_path'] = os.path.join(
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                'assets', 'DeckmountLogo.png')
        elif is_demo_mode:
            
            frozen = {
                'HR':       60,      # Demo BPM
                'RR':       1000,    # Calculated from 60 BPM (60000/60)
                'PR':       167,     # Demo PR interval
                'QRS':      86,      # Demo QRS duration  
                'QT':       357,     # Demo QT interval
                'QTc':      357,     # Demo QTc interval
                'QTcF':     357,     # Demo QTcF interval
                'rv5':      float(getattr(self, '_last_rv5', 0.0) or 0.0),
                'sv1':      float(getattr(self, '_last_sv1', 0.0) or 0.0),
                'p_axis':   getattr(self, 'last_p_axis',   '--'),
                'QRS_axis': getattr(self, 'last_qrs_axis', '--'),
                't_axis':   getattr(self, 'last_t_axis',   '--'),
                'lead_seq': (_sm.get_setting('lead_sequence', 'Standard') if _sm else 'Standard'),
                'logo_path': os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                    'assets', 'DeckmountLogo.png'),
            }
        else:
            # FIX-1: Read metrics from both internal attributes AND screen labels.
            # Take the higher of the two values so a momentary mid-update zero
            # never wipes out what the user can clearly see on screen.
            def _screen_int(label_key, fallback=0):
                try:
                    raw = self.metric_labels[label_key].text().strip()
                    # Strip units like 'BPM', 'ms' etc.
                    clean = raw.replace('BPM','').replace('bpm','').replace('ms','').replace('mV','').split('/')[0].strip()
                    val = int(round(float(clean)))
                    return val if val > 0 else fallback
                except Exception:
                    return fallback

            def _screen_qtc(label_key, fallback=0):
                """Read QTc from combined 'QT/QTc' label — always the second part after '/'."""
                try:
                    raw = self.metric_labels[label_key].text().strip()
                    clean = raw.replace('BPM','').replace('bpm','').replace('ms','').replace('mV','').strip()
                    if '/' in clean:
                        # "400/284" → take QTc (second part)
                        qtc_part = clean.split('/')[-1].strip()
                    else:
                        qtc_part = clean
                    val = int(round(float(qtc_part))) if qtc_part else 0
                    return val if val > 0 else fallback
                except Exception:
                    return fallback

            _live_hr  = int(getattr(self, 'last_heart_rate',    0) or 0)
            _live_pr  = int(getattr(self, 'pr_interval',        0) or 0)
            _live_qrs = int(getattr(self, 'last_qrs_duration',  0) or 0)
            _live_qt  = int(getattr(self, 'last_qt_interval',   0) or 0)
            _live_qtc = int(getattr(self, 'last_qtc_interval',  0) or 0)

            _scr_hr  = _screen_int('heart_rate',  _live_hr)
            _scr_pr  = _screen_int('pr_interval', _live_pr)
            _scr_qrs = _screen_int('qrs_duration', _live_qrs)
            _scr_qtc = _screen_qtc('qtc_interval', _live_qtc)

            frozen = {
                'HR':       max(_live_hr,  _scr_hr),
                'RR':       int(getattr(self, 'last_rr_interval',   0) or 0),
                'PR':       max(_live_pr,  _scr_pr),
                'QRS':      max(_live_qrs, _scr_qrs),
                'QT':       _live_qt,
                'QTc':      max(_live_qtc, _scr_qtc),
                'QTcF':     int(getattr(self, 'last_qtcf_interval', 0) or 0),
                'rv5':      float(getattr(self, '_last_rv5', 0.0) or 0.0),
                'sv1':      float(getattr(self, '_last_sv1', 0.0) or 0.0),
                'p_axis':   getattr(self, 'last_p_axis',   '--'),
                'QRS_axis': getattr(self, 'last_qrs_axis', '--'),
                't_axis':   getattr(self, 'last_t_axis',   '--'),
                'lead_seq': (_sm.get_setting('lead_sequence', 'Standard') if _sm else 'Standard'),
                'logo_path': os.path.join(
                    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')),
                    'assets', 'DeckmountLogo.png'),
            }

        try:
            limb_conn = getattr(self, "_lead_connection_state", {}) or {}
            lead_off = bool(getattr(self, "_lead_off_latched", False))
            if isinstance(frozen, dict):
                if lead_off or limb_conn.get("V5", True) is False:
                    frozen["rv5"] = 0.0
                if lead_off or limb_conn.get("V1", True) is False:
                    frozen["sv1"] = 0.0
                frozen["rv5_mv"] = float(frozen.get("rv5", 0.0) or 0.0)
                frozen["sv1_mv"] = float(frozen.get("sv1", 0.0) or 0.0)
        except Exception as _lead_sanitize_err:
            print(f" Warning: could not sanitize frozen RV5/SV1: {_lead_sanitize_err}")

        username = ''
        try:
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                username = getattr(self.dashboard_instance, 'username', '') or ''
        except Exception:
            pass

        patient = resolve_patient_profile(
            explicit_patient=getattr(getattr(self, 'dashboard_instance', None), 'patient_details', None),
            username=username,
            user_details=getattr(getattr(self, 'dashboard_instance', None), 'user_details', {}) or {},
        )

        # Check if demo mode is active and set name to "Demo Mode"
        if is_demo_mode:
            patient['first_name'] = 'Demo Mode'
            patient['name'] = 'Demo Mode'
            patient['last_name'] = ''

        # Force a fresh evaluation of rhythm & dashboard conclusion so last_conclusions.json is up-to-date
        try:
            self.update_latest_rhythm_interpretation()
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                if hasattr(self.dashboard_instance, 'update_live_conclusion'):
                    self.dashboard_instance.update_live_conclusion()
        except Exception as _ce:
            print(f" Error refreshing live conclusions before report: {_ce}")

        # Resolve sampling rate
        fs = 500.0
        try:
            if hasattr(self, 'sampler') and getattr(self.sampler, 'sampling_rate', None):
                fs = max(100.0, float(self.sampler.sampling_rate))
        except Exception:
            pass

        # Preserve live UI metrics if available; fallback to snapshot calculation only if frozen HR is missing
        lead_ii_snap = snap_raw[1] if (len(snap_raw) > 1 and snap_raw[1].size > 0) else np.array([])
        if frozen.get('HR', 0) <= 0 and lead_ii_snap.size >= int(fs * 2.0) and not is_demo_mode and display_mode != "frozen":
            try:
                from ecg.ecg_calculations import calculate_all_ecg_metrics
                snap_metrics = calculate_all_ecg_metrics(lead_ii_snap, fs=fs)
                if snap_metrics.get("heart_rate", 0) > 0:
                    frozen['HR'] = int(snap_metrics.get("heart_rate", 0))
                    frozen['RR'] = int(round(snap_metrics.get("rr_interval", 0)))
                    frozen['PR'] = int(snap_metrics.get("pr_interval", 0))
                    frozen['QRS'] = int(snap_metrics.get("qrs_duration", 0))
                    frozen['QT'] = int(round(snap_metrics.get("qt_interval", 0) or 0))
                    frozen['QTc'] = int(snap_metrics.get("qtc_interval", 0))
            except Exception as _m_err:
                print(f" Warning: Could not compute snapshot metrics: {_m_err}")

        # Direct analysis of the exact 10-second snapshot being printed in the PDF
        conc_list = []
        is_flatline = (lead_ii_snap.size < int(fs * 2.0)) or (np.std(lead_ii_snap) < 0.1) or (frozen.get('HR', 0) <= 0)

        if is_flatline:
            # Three distinct states, not one. Telling the operator to check a
            # cable while the patient has no cardiac output sends them after
            # equipment that is working correctly.
            if getattr(self, '_asystole_active', False):
                conc_list = ["No cardiac activity detected"]
            elif getattr(self, '_rate_below_measurable', False):
                conc_list = ["Rate below measurable range"]
            else:
                conc_list = ["No ECG data available"]
            frozen['HR'] = 0
            frozen['RR'] = 0
            frozen['PR'] = 0
            frozen['QRS'] = 0
            frozen['QT'] = 0
            frozen['QTc'] = 0
            frozen['QTcF'] = 0
            frozen['rv5'] = 0.0
            frozen['sv1'] = 0.0
            frozen['rv5_mv'] = 0.0
            frozen['sv1_mv'] = 0.0
        else:
            try:
                lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                snapshot_signals = {}
                for i, name in enumerate(lead_names):
                    if i < len(snap_raw) and snap_raw[i].size > 0:
                        snapshot_signals[name] = snap_raw[i]

                if "II" in snapshot_signals and snapshot_signals["II"].size >= int(fs * 2.0):
                    from ecg.arrhythmia_detector import analyze_ecg
                    ext_m = {
                        'external_hr': frozen.get('HR', 0),
                        'external_pr': frozen.get('PR', 0),
                        'external_qrs': frozen.get('QRS', 0),
                    }
                    snap_results = analyze_ecg(snapshot_signals, fs=fs, external_metrics=ext_m)
                    snap_arrhythmias = snap_results.get("arrhythmias", [])
                    if snap_arrhythmias:
                        conc_list = [a for a in snap_arrhythmias if a and a != "Rhythm Undetermined"]
            except Exception as _snap_err:
                print(f" Error analyzing 10s snapshot for report: {_snap_err}")

            if not conc_list or conc_list == ["Rhythm Undetermined"]:
                latest_rhythm = getattr(self, "_latest_rhythm_interpretation", "") or ""
                if latest_rhythm and latest_rhythm not in ("Analyzing Rhythm...", "Detecting...", "Rhythm Undetermined", ""):
                    conc_list = [latest_rhythm]

            hr_val = frozen.get('HR', 0)
            if not conc_list or conc_list == ["Rhythm Undetermined"]:
                if 0 < hr_val < 60:
                    conc_list = ["Sinus Bradycardia"]
                elif hr_val > 100:
                    conc_list = ["Sinus Tachycardia"]
                else:
                    conc_list = ["Normal Sinus Rhythm"]
            else:
                # FIX-2: Sanitize rate-mismatched rhythms AND append rate qualifier
                # for non-NSR arrhythmias so the full clinical picture is in the PDF.
                if 0 < hr_val < 60:
                    # Replace NSR label directly (rate is inconsistent with NSR)
                    conc_list = ["Sinus Bradycardia" if c in ("Normal Sinus Rhythm", "Normal sinus rhythm") else c for c in conc_list]
                    # Also append Bradycardia note if not already present alongside other arrhythmias
                    bradycardia_terms = ("bradycardia", "slow", "junctional", "block")
                    if not any(t in c.lower() for c in conc_list for t in bradycardia_terms):
                        conc_list.append("Bradycardia")
                elif hr_val > 100:
                    # Replace NSR label directly (rate is inconsistent with NSR)
                    conc_list = ["Sinus Tachycardia" if c in ("Normal Sinus Rhythm", "Normal sinus rhythm") else c for c in conc_list]
                    # Also append Tachycardia note if not already present alongside other arrhythmias
                    tachycardia_terms = ("tachycardia", "fast", "flutter", "fibrillation", "svt", "vt")
                    if not any(t in c.lower() for c in conc_list for t in tachycardia_terms):
                        conc_list.append("Sinus Tachycardia")

        # FIX-3: Auto-add QTc warning to conclusion when clinically significant.
        # A prolonged QTc is a risk for life-threatening arrhythmias (Torsades de Pointes).
        # Without this, the doctor only sees the raw number buried in the header — easy to miss.
        try:
            qtc_for_warning = frozen.get('QTc', 0) or 0
            qtc_already_noted = any('qtc' in c.lower() or 'qt' in c.lower() for c in conc_list)
            if not qtc_already_noted and qtc_for_warning > 0:
                if qtc_for_warning >= 500:
                    conc_list.insert(0, f"Critically Prolonged QTc ({qtc_for_warning} ms) — High Risk")
                elif qtc_for_warning >= 460:
                    conc_list.insert(0, f"Prolonged QTc ({qtc_for_warning} ms)")
                elif qtc_for_warning >= 440:
                    conc_list.insert(0, f"Borderline QTc ({qtc_for_warning} ms)")
        except Exception:
            pass

        # ── Findings allow-list ─────────────────────────────────────────────
        # This is the generator the 12-lead page actually uses, so the
        # restriction has to be applied HERE. It was previously only in
        # ecg_report_generator.generate_ecg_report(), which this path never
        # calls — so classifier labels such as "Asystole" still reached the PDF.
        #
        # Status lines ("No cardiac activity detected", "Rate below measurable
        # range", "No ECG data available") are not findings and are preserved.
        try:
            from .ecg_report_generator import restrict_to_allowed_conclusions
            _status_lines = {
                "No cardiac activity detected",
                "Rate below measurable range",
                "No ECG data available",
                "Please connect device",
            }
            _status = [c for c in conc_list if c in _status_lines]

            # The QTc entries carry their measured value, e.g.
            # "Prolonged QTc (470 ms)" — strip it so a real finding is not
            # dropped for wording, then let the allow-list supply the canonical
            # form. "Borderline QTc" is deliberately NOT mapped: 440-459 ms is
            # below the 460 ms threshold the permitted "Prolonged QTc" means.
            _canon = []
            for _c in conc_list:
                _low = str(_c).lower()
                if "critically prolonged qtc" in _low or "prolonged qtc" in _low:
                    _canon.append("Prolonged QTc")
                else:
                    _canon.append(_c)

            _findings = restrict_to_allowed_conclusions(_canon)
            conc_list = _status + [c for c in _findings if c not in _status]

            # Never print an empty conclusion box. If everything the classifier
            # offered was filtered out but a heart rate WAS measured, state the
            # rate: that is always reportable and is derived from a number
            # printed in the same header. This is the path taken when the
            # classifier returns only a suppressed label — e.g. "Asystole"
            # alongside a live HR, which is the self-contradiction that started
            # this work.
            if not conc_list:
                _hr = frozen.get('HR', 0) or 0
                if _hr > 0:
                    conc_list = ["Sinus Bradycardia" if _hr < 60
                                 else "Sinus Tachycardia" if _hr > 100
                                 else "Normal Sinus Rhythm"]
        except Exception as _allow_err:
            print(f" Could not apply conclusion allow-list: {_allow_err}")

        conc_list = conc_list[:5]   # hard cap at 5

        # Sampling rate
        fs = 500.0
        try:
            if hasattr(self, 'sampler') and getattr(self.sampler, 'sampling_rate', None):
                fs = max(100.0, float(self.sampler.sampling_rate))
        except Exception:
            pass

        # Output path
        stamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        rpt_dir = str(data_file("reports"))
        os.makedirs(rpt_dir, exist_ok=True)

        # Resolve machine serial — check all sources in priority order
        machine_serial = (
            getattr(self, "machine_serial_number", "") or ""
        ).strip()
        if not machine_serial and hasattr(self, "settings_manager") and self.settings_manager:
            machine_serial = (self.settings_manager.get_setting("machine_serial_number", "") or "").strip()
        if not machine_serial:
            try:
                from utils.crash_logger import get_crash_logger
                machine_serial = (get_crash_logger().machine_serial_id or "").strip()
            except Exception:
                pass
        if not machine_serial:
            machine_serial = (os.getenv("MACHINE_SERIAL_ID", "") or "").strip()

        serial_part = f"_{machine_serial}" if machine_serial and machine_serial not in ("Not Detected", "") else ""
        filename = os.path.join(rpt_dir, f"ECG_Report_{fmt}{serial_part}_{stamp}.pdf")

        # Capture signup details and machine serial before the worker thread
        # to avoid 'self' shadowing issues inside the QObject.run() method.
        _signup_snap = getattr(getattr(self, "dashboard_instance", None), "user_details", {}) or {}
        _machine_serial_snap = machine_serial

        # Worker thread
        class _Worker(QObject):
            finished = pyqtSignal(str)
            error    = pyqtSignal(str)
            def __init__(self): super().__init__()
            def run(self):
                try:
                    import sys as _sys, os as _os, shutil as _sh, json as _js
                    import datetime as _dt
                    _ecg = _os.path.dirname(_os.path.abspath(__file__))
                    _src = _os.path.abspath(_os.path.join(_ecg, '..'))
                    for _p in [_ecg, _src]:
                        if _p not in _sys.path:
                            _sys.path.insert(0, _p)

                    if is_demo_mode:
                        from ecg.demo_ecg_report_generator import generate_demo_android_ecg_report as _gen_demo

                        _gen_demo(
                            filename=filename,
                            patient=patient,
                            frozen=frozen,
                            fmt=fmt,
                            conc_list=conc_list,
                        )
                    else:
                        from ecg.ecg_report_android import generate_report as _gen

                        _gen(
                            snap_raw=snap_raw,
                            frozen=frozen,
                            patient=patient,
                            filename=filename,
                            fmt=fmt,
                            conc_list=conc_list,
                            fs=fs,
                        )

                    # Copy to Downloads
                    try:
                        import pathlib
                        dl = pathlib.Path.home() / 'Downloads'
                        if dl.exists():
                            ob = _os.path.basename(filename)
                            dp = dl / ob; ctr = 1
                            nm, ext = _os.path.splitext(ob)
                            while dp.exists():
                                dp = dl / f"{nm}_{ctr}{ext}"; ctr += 1
                            _sh.copyfile(filename, str(dp))
                    except Exception: pass

                    # ── Create JSON twin (in thread) ───────────────────────────
                    try:
                        twin_path = _os.path.splitext(filename)[0] + '.json'
                        from utils.ecg_payload_builder import build_12lead_payload
                        
                        # Map snap_raw to lead names for raw samples
                        LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                        raw_leads_map = {}
                        for i, name in enumerate(LEAD_NAMES):
                            if i < len(snap_raw):
                                # Convert numpy arrays to lists for JSON serialization
                                raw_leads_map[name] = snap_raw[i].tolist() if isinstance(snap_raw[i], np.ndarray) else list(snap_raw[i])
                        
                        twin_data = build_12lead_payload(
                            data=frozen,
                            patient=patient,
                            signup_details=_signup_snap,
                            settings_manager=None,
                            raw_leads=raw_leads_map,
                            report_format=fmt,
                            source_report_file=filename
                        )
                        with open(twin_path, 'w') as jf:
                            _js.dump(twin_data, jf, indent=2)
                    except Exception as je:
                        print(f"Error creating JSON twin: {je}")

                    # Update index.json (Recent Reports)
                    try:
                        idx_path = _os.path.join(rpt_dir, 'index.json')
                        items = []
                        if _os.path.exists(idx_path):
                            try:
                                with open(idx_path) as _if: items = _js.load(_if)
                            except Exception: items = []
                        fn = patient.get('first_name','') if isinstance(patient, dict) else ''
                        ln = patient.get('last_name', '') if isinstance(patient, dict) else ''
                        now = _dt.datetime.now()
                        owner_full_name = ""
                        try:
                            dash = getattr(self, "dashboard_instance", None)
                            if dash is not None:
                                owner_full_name = (getattr(dash, "user_details", {}) or {}).get("full_name") or getattr(dash, "username", "") or ""
                        except Exception:
                            owner_full_name = ""
                        meta = {
                            'filename': _os.path.basename(filename),
                            'title':    'ECG Report',
                            'patient':  (fn+' '+ln).strip(),
                            'date':     now.strftime('%Y-%m-%d'),
                            'time':     now.strftime('%H:%M:%S'),
                            'username': username,
                            'owner_full_name': owner_full_name or username or "",
                            'format':   fmt,
                        }
                        items = [meta] + items
                        with open(idx_path, 'w') as _of:
                            _js.dump(items[:20], _of, indent=2)
                    except Exception: pass

                    self.finished.emit(f"✅ {fmt_label} Report saved:\n{filename}")
                except Exception as _e:
                    import traceback as _tb
                    self.error.emit(f"{_e}\n{_tb.format_exc()}")

        thread = QThread()
        worker = _Worker()
        worker.moveToThread(thread)
        self._report_thread  = thread
        self._report_worker  = worker

        def _on_finish(msg):
            print(msg)
            from PyQt5.QtWidgets import QMessageBox
            try:
                QMessageBox.information(self, "Report Saved", msg)
            except Exception:
                pass

            # Refresh Dashboard -> Recent Reports immediately (no restart needed)
            try:
                dash = getattr(self, "dashboard_instance", None)
                if dash is not None and hasattr(dash, "refresh_recent_reports_ui"):
                    dash.refresh_recent_reports_ui()
            except Exception:
                pass
            try:
                if getattr(self, "_report_thread", None) is thread:
                    self._report_thread = None
                    self._report_worker = None
            except Exception:
                pass
            try: thread.quit(); thread.wait(3000)
            except Exception: pass
            self._update_generate_report_button_state()

        def _on_error(msg):
            print(f"❌ Report error: {msg}")
            from PyQt5.QtWidgets import QMessageBox
            try: QMessageBox.critical(self, "Report Failed",
                     f"PDF generation failed.\n\n{msg.split(chr(10))[0]}")
            except Exception: pass
            try:
                if getattr(self, "_report_thread", None) is thread:
                    self._report_thread = None
                    self._report_worker = None
            except Exception:
                pass
            try: thread.quit(); thread.wait(3000)
            except Exception: pass
            self._update_generate_report_button_state()

        worker.finished.connect(_on_finish)
        worker.error.connect(_on_error)
        thread.started.connect(worker.run)
        thread.finished.connect(thread.deleteLater)
        thread.start(QThread.LowPriority)


    def export_csv(self):
        """Export ECG data to CSV file in the same format as dummydata.csv"""
        path, _ = QFileDialog.getSaveFileName(self, "Export ECG Data as CSV", "", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')  # Use tab delimiter like dummydata.csv
                    
                    # Write header exactly like dummydata.csv
                    header = ["Sample", "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
                    writer.writerow(header)
                    
                    # Export from CSV storage (most accurate method)
                    if hasattr(self, 'csv_data_storage') and self.csv_data_storage:
                        print(f" Exporting {len(self.csv_data_storage)} samples from CSV storage")
                        
                        for row_data in self.csv_data_storage:
                            row = [
                                row_data.get('Sample', ''),
                                row_data.get('I', ''),
                                row_data.get('II', ''),
                                row_data.get('III', ''),
                                row_data.get('aVR', ''),
                                row_data.get('aVL', ''),
                                row_data.get('aVF', ''),
                                row_data.get('V1', ''),
                                row_data.get('V2', ''),
                                row_data.get('V3', ''),
                                row_data.get('V4', ''),
                                row_data.get('V5', ''),
                                row_data.get('V6', '')
                            ]
                            writer.writerow(row)
                    
                    # Fallback: Export from numpy arrays if CSV storage is empty
                    else:
                        print(" Exporting from numpy arrays (fallback method)")
                        
                        # Get the actual data length
                        max_length = 0
                        for i in range(len(self.leads)):
                            if i < len(self.data):
                                # Count non-zero values in the numpy array
                                non_zero_count = np.count_nonzero(self.data[i])
                                max_length = max(max_length, non_zero_count)
                        
                        # Export data sample by sample
                        for i in range(max_length):
                            row = [i]  # Sample number
                            
                            # Add data for each lead in the same order as dummydata.csv
                            for lead_idx, lead_name in enumerate(self.leads):
                                if lead_idx < len(self.data):
                                    if i < len(self.data[lead_idx]):
                                        value = self.data[lead_idx][i]
                                        # Only include non-zero values (actual data)
                                        if value != 0:
                                            row.append(int(value))
                                        else:
                                            row.append("")
                                    else:
                                        row.append("")
                                else:
                                    row.append("")
                            
                            writer.writerow(row)
                
                print(f" CSV export completed: {path}")
                QMessageBox.information(
                    self, 
                    "Export Successful", 
                    f"ECG data exported successfully!\n\nFile: {path}\nSamples: {len(self.csv_data_storage) if hasattr(self, 'csv_data_storage') else 'N/A'}"
                )
                
            except Exception as e:
                print(f" Error exporting CSV: {e}")
                QMessageBox.critical(
                    self, 
                    "Export Error", 
                    f"Failed to export CSV:\n{str(e)}"
                )

    def close_serial_connection(self):
        """Safely close the serial connection if it exists"""
        if self.serial_reader:
            print("Cleaning up serial connection...")
            try:
                if hasattr(self.serial_reader, 'stop'):
                    self.serial_reader.stop()
                self.serial_reader.close()
            except Exception as e:
                print(f"Error closing serial reader: {e}")
            self.serial_reader = None

    def closeEvent(self, event):
        """Handle widget closure"""
        self.close_serial_connection()
        event.accept()

    def go_back(self):
        """Go back to the dashboard"""
        try:
            if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
                QMessageBox.information(
                    self,
                    "Demo Mode Enabled",
                    "Please disable Demo Mode if you want to switch to the main dashboard."
                )
                return
        except Exception:
            pass
        if hasattr(self, '_overlay_active') and self._overlay_active:
            self._restore_original_layout()

        # Turn off demo mode if it's ON (only for demo mode, not for real serial data)
        if hasattr(self, 'demo_toggle') and self.demo_toggle and self.demo_toggle.isChecked():
            # Demo mode is ON, turn it OFF before going back
            self.demo_toggle.setChecked(False)
            # This will trigger on_demo_toggle_changed which calls demo_manager.toggle_demo_mode(False)

        # Hide the sliding panel if it's open before going back
        if hasattr(self, 'ecg_menu') and self.ecg_menu:
            self.ecg_menu.hide_sliding_panel()

        # Go back to dashboard (assumes dashboard is at index 0)
        self.stacked_widget.setCurrentIndex(0)

    @staticmethod
    def _friendly_connection_message(extra_msg: str) -> str:
        """Turn low-level serial failures into a user-friendly reconnect prompt."""
        if not extra_msg:
            return ""

        lower_msg = extra_msg.lower()
        if not lower_msg.startswith("failed to connect to any serial port:"):
            return extra_msg

        device_disconnect_markers = (
            "cannot configure port",
            "a device attached to the system is not functioning",
            "write timeout",
            "access is denied",
            "clearcommerror failed",
            "permissionerror(13",
        )
        if any(marker in lower_msg for marker in device_disconnect_markers):
            return (
                "ECG device connection was interrupted. Please check the USB cable, "
                "device power, and reconnect the ECG device. CardioX will try again "
                "automatically."
            )
        return extra_msg

    def show_connection_warning(self, extra_msg=""):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Connection Required")
        msg.setText(self._friendly_connection_message(extra_msg))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def show_main_menu(self):  
        self.clear_content()

    def clear_content(self):
        for i in reversed(range(self.content_layout.count())):
            widget = self.content_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

    def show_sequential_view(self):
        """
        Open sequential 12‑lead view window.
        Import is wrapped in try/except so the dashboard keeps working even if the
        optional module is missing on a given install.
        """
        try:
            from ecg.lead_sequential_view import LeadSequentialView
        except Exception as e:
            QMessageBox.warning(
                self,
                "Sequential View Unavailable",
                f"The sequential view module could not be loaded.\n\nDetails: {e}"
            )
            return

        try:
            win = LeadSequentialView(self.leads, self.data, buffer_size=500)
            win.show()
            self._sequential_win = win
        except Exception as e:
            QMessageBox.critical(
                self,
                "Sequential View Error",
                f"Failed to open sequential view window.\n\nDetails: {e}"
            )

    # ------------------------------------ 12 leads overlay --------------------------------------------

    def twelve_leads_overlay(self):
        # Send stop command to device when opening 12:1 ECG overlay
        try:
            if self.serial_reader and hasattr(self.serial_reader, 'ser') and \
                    self.serial_reader.ser and self.serial_reader.ser.is_open:
                self.serial_reader.ser.write(b'0\r\n')
                print(" Stop command sent to device (12:1 ECG overlay opened)")
        except Exception as _stop_err:
            print(f" Could not send stop command to device: {_stop_err}")

        if getattr(self, "_overlay_active", False):
            # If we're already in 12:1, treat this as a toggle (close overlay)
            if getattr(self, "_current_overlay_layout", None) == "12x1":
                self._restore_original_layout()
                self._current_overlay_layout = None
                return
            # If some other overlay (e.g. 6:2) is active, restore first, then switch
            self._restore_original_layout()
        
        # Store the original plot area layout
        self._store_original_layout()
        
        # Create the overlay widget
        self._create_overlay_widget()
        
        # Replace the plot area with overlay
        self._replace_plot_area_with_overlay()
        
        # Mark overlay as active and record layout type
        self._overlay_active = True
        self._current_overlay_layout = "12x1"

        self._apply_current_overlay_mode()

        # Ensure demo data continues to work in overlay mode
        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            print("Demo mode active - overlay will show demo data")

    def _store_original_layout(self):
        
        # Store the current plot area widget
        self._original_plot_area = self.plot_area
        
        # Store the current layout
        self._original_layout = self.plot_area.layout()
        
        # Store the current figures, canvases, axes, and lines
        self._original_figures = getattr(self, 'figures', [])
        self._original_canvases = getattr(self, 'canvases', [])
        self._original_axs = getattr(self, 'axs', [])
        self._original_lines = getattr(self, 'lines', [])

    def _create_overlay_widget(self):
        
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame
        
        # Create overlay container
        self._overlay_widget = QWidget()
        self._overlay_widget.setStyleSheet("""
            QWidget {
                background: #000;
                border: none;
                border-radius: 15px;
            }
        """)
        
        # Main layout for overlay
        overlay_layout = QVBoxLayout(self._overlay_widget)
        overlay_layout.setContentsMargins(20, 10, 20, 16)
        overlay_layout.setSpacing(15)
        
        # Top control panel with close button
        top_panel = QFrame()
        top_panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 10px;
            }
        """)
        top_layout = QHBoxLayout(top_panel)
        top_layout.setContentsMargins(15, 10, 15, 10)
        top_layout.setSpacing(20)
        
        # Close button
        close_btn = QPushButton("Close Overlay")
        close_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 2px solid #ff6600;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 2px solid #ff8c42;
            }
        """)
        close_btn.clicked.connect(self._restore_original_layout)
        
        # Mode control buttons with highlighting
        self.light_mode_btn = QPushButton("Light Mode")
        self.dark_mode_btn = QPushButton("Dark Mode")
        self.graph_mode_btn = QPushButton("Graph Mode")
        
        # Store current mode for highlighting
        self._current_overlay_mode = "graph"  # Default mode
        
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
            }
        """
        
        active_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 3px solid #ff6600;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
                /* Removed unsupported box-shadow property */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 3px solid #ff8c42;
            }
        """
        
        self.light_mode_btn.setStyleSheet(button_style)
        self.dark_mode_btn.setStyleSheet(button_style)
        self.graph_mode_btn.setStyleSheet(button_style)
        
        # Add widgets to top panel
        top_layout.addWidget(close_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.light_mode_btn)
        top_layout.addWidget(self.dark_mode_btn)
        top_layout.addWidget(self.graph_mode_btn)
        
        overlay_layout.addWidget(top_panel)
        
        # Create the matplotlib figure with all leads
        self._create_overlay_figure(overlay_layout)
        
        # Connect mode buttons
        self.light_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("light"))
        self.dark_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("dark"))
        self.graph_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("graph"))

        # Defer initial graph mode until widget is rendered so figure dimensions are final
        QTimer.singleShot(50, lambda: self._apply_overlay_mode("graph"))

    # def _calculate_adaptive_figsize(self, layout="12x1", num_leads=12):
    #     """
    #     Calculate an adaptive figure size based on the primary screen's resolution and physical DPI.
    #     """
    #     try:
    #         from PyQt5.QtWidgets import QApplication
    #         screen = QApplication.primaryScreen()
    #         if not screen:
    #             return (16, num_leads * 1.2) if layout == "12x1" else (16, 12)
            
    #         geometry = screen.availableGeometry()
    #         width_px = geometry.width()
    #         height_px = geometry.height()
            
    #         dpi_x = screen.physicalDotsPerInchX()
    #         dpi_y = screen.physicalDotsPerInchY()
            
    #         if dpi_x < 40 or dpi_x > 400: dpi_x = 96.0
    #         if dpi_y < 40 or dpi_y > 400: dpi_y = 96.0
            
    #         screen_width_in = width_px / dpi_x
    #         screen_height_in = height_px / dpi_y
            
    #         target_width_in = screen_width_in * 0.9
            
    #         if layout == "12x1":
    #             target_height_in = max(screen_height_in * 0.8, num_leads * 1.0)
    #         else:
    #             target_height_in = max(screen_height_in * 0.8, 8.0)
                
    #         target_width_in = min(target_width_in, 24.0)
            
    #         return (target_width_in, target_height_in)
    #     except Exception as e:
    #         print(f"Error calculating adaptive figsize for {layout}: {e}")
    #         return (16, num_leads * 1.2) if layout == "12x1" else (16, 12)

    def _create_overlay_figure(self, overlay_layout):
        
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import numpy as np
        from PyQt5.QtWidgets import QScrollArea
        
        # Create figure with all leads - adjust spacing for better visibility
        num_leads = len(self.leads)
        fig = Figure(figsize=(16, num_leads * 1.2), facecolor='none')  # Changed to transparent
        # figsize = self._calculate_adaptive_figsize("12x1", num_leads)
        # fig = Figure(figsize=figsize, facecolor='none')  # Changed to transparent
        
        # Adjust subplot parameters for better spacing
        fig.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.06, hspace=0.02)
        
        self._overlay_axes = []
        self._overlay_lines = []
        
        for idx, lead in enumerate(self.leads):
            ax = fig.add_subplot(num_leads, 1, idx+1)
            ax.set_facecolor('none')  # Changed to transparent
            
            # Remove all borders and spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Remove all ticks and labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(lead, color='#00ff00', fontsize=12, fontweight='bold', labelpad=5)
            
            # Create line with initial data
            line, = ax.plot(np.arange(self.buffer_size), [np.nan]*self.buffer_size, color="#00ff00", lw=0.7)
            # Clipping stays ON. With ylim pinned to the full 0-4095 ADC range and
            # the +2048 centering, a normal R-wave at 20 mm/mV reaches ~4248 and,
            # unclipped, paints straight into the lane of the lead above it —
            # measured 51 spilled pixels at 20 mm/mV and 514 at 40 mm/mV. That
            # overlap is the "cluttered" trace: one lead's R-waves appearing in
            # another lead's row. If a taller trace is wanted, widen the y-limits
            # or clamp the data; do not let it draw outside its own axes.
            self._overlay_axes.append(ax)
            self._overlay_lines.append(line)
        
        self._overlay_canvas = FigureCanvas(fig)
        self._overlay_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._overlay_canvas.setStyleSheet("background: transparent;")
        scroll_area = QScrollArea()
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidgetResizable(True)
        scroll_area.setAlignment(Qt.AlignTop)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        lead_height = 80
        self._overlay_canvas.setMinimumHeight(int(num_leads * lead_height))
        scroll_area.setWidget(self._overlay_canvas)
        scroll_area.setMinimumHeight(int(6 * lead_height))
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overlay_layout.addWidget(scroll_area, 1)
        
        # Start update timer for overlay
        self._overlay_timer = QTimer(self)
        self._overlay_timer.timeout.connect(self._update_overlay_plots)
        self._overlay_timer.start(100)

    def _get_overlay_target_buffer_len(self, is_demo_mode):
        """
        Calculate buffer length for overlay modes based on wave speed.
        For real serial data, calculate based on wave speed to ensure peaks align.
        For demo mode, use same calculation as main 12 lead grid view to match wave peaks.
        """
        try:
            wave_speed = 25.0 if getattr(self, '_overlay_active', False) else float(self.settings_manager.get_wave_speed())
        except Exception:
            wave_speed = 25.0

        if not is_demo_mode:
            # For real serial data, calculate buffer length based on wave speed
            # Same logic as in update_plots() for serial data
            # Set different baseline based on layout mode
            layout = getattr(self, "_current_overlay_layout", "12x1")
            baseline_seconds = 3.0 if layout == "6x2" else 6.0
            seconds_scale = (25.0 / max(1e-6, wave_speed))
            seconds_to_show = baseline_seconds * seconds_scale
            
            # Use hardware sampling rate
            sampling_rate = 500.0
            if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                sampling_rate = float(self.sampler.sampling_rate)
            elif hasattr(self, 'sampling_rate') and self.sampling_rate > 10:
                sampling_rate = float(self.sampling_rate)
            samples_to_show = int(sampling_rate * seconds_to_show)
            
            # Return the calculated samples (same as main plots - no buffer size limit)
            # The data selection will handle cases where data is smaller
            return max(1, samples_to_show)

        # Demo mode: Use same calculation as main 12 lead grid view
        # Set different baseline based on layout mode
        layout = getattr(self, "_current_overlay_layout", "12x1")
        baseline_seconds = 3.0 if layout == "6x2" else 6.0
        seconds_scale = (25.0 / max(1e-6, wave_speed))
        seconds_to_show = baseline_seconds * seconds_scale
        
        # Get sampling rate - same logic as update_plots() for demo mode
        sampling_rate = 500  # Default
        if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
            sampling_rate = float(self.sampler.sampling_rate)
        elif hasattr(self, 'sampling_rate') and self.sampling_rate > 10:
            sampling_rate = float(self.sampling_rate)
        elif hasattr(self, 'demo_manager') and self.demo_manager and hasattr(self.demo_manager, 'samples_per_second'):
            sampling_rate = float(self.demo_manager.samples_per_second)
        
        samples_to_show = int(sampling_rate * seconds_to_show)
        return max(1, samples_to_show)

    def _update_overlay_plots(self):
        # If frozen, we only proceed if we have a valid snapshot to draw.
        # Otherwise, if frozen and no snapshot, return.
        if getattr(self, "_grid_frozen", False) and getattr(self, "_replay_snapshot", None) is None:
            return
        if not hasattr(self, '_overlay_lines') or not self._overlay_lines:
            return
        
        # Check if demo mode is active
        is_demo_mode = hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked()
        
        target_buffer_len = self._get_overlay_target_buffer_len(is_demo_mode)
        
        # Select data source based on frozen state
        use_snapshot = getattr(self, "_grid_frozen", False) and getattr(self, "_replay_snapshot", None) is not None

        for idx, lead in enumerate(self.leads):
            if idx < len(self._overlay_lines):
                if use_snapshot:
                    data = self._replay_snapshot[idx] if idx < len(self._replay_snapshot) else np.array([])
                elif idx < len(self.data):
                    data = self.data[idx]
                else:
                    data = np.array([])
                line = self._overlay_lines[idx]
                ax = self._overlay_axes[idx]

                # Ensure overlay line length matches current buffer size
                buffer_len = target_buffer_len
                try:
                    xdata = line.get_xdata()
                    current_len = len(xdata) if xdata is not None else 0
                    if current_len != buffer_len:
                        new_x = np.arange(buffer_len)
                        line.set_xdata(new_x)
                        current_len = buffer_len
                    if current_len:
                        buffer_len = current_len
                except Exception as e:
                    print(f" Overlay line sync error (12-lead): {e}")
                    buffer_len = target_buffer_len
                
                plot_data = np.full(buffer_len, np.nan)
                
                if data is not None and len(data) > 0:
                    # 12:1 overlay waves appear immediately at acquisition start.
                    try:
                        raw_data = np.asarray(data, dtype=float)
                        non_zero_indices = np.where(raw_data != 0)[0]
                        if len(non_zero_indices) > 0:
                            first_real_idx = int(non_zero_indices[0])
                            recent_data = raw_data[first_real_idx:]
                        else:
                            recent_data = raw_data[-min(buffer_len, len(raw_data)):] if len(raw_data) > 0 else raw_data
                    except Exception:
                        raw_data = np.asarray(data, dtype=float)
                        recent_data = raw_data

                    # Take exactly buffer_len samples from the tail of recent_data
                    # so overlay and main grid show the same active window.
                    if len(recent_data) >= buffer_len:
                        data_segment = recent_data[-buffer_len:]
                    else:
                        data_segment = recent_data
                    
                    # Optional AC notch filtering (match main 12-lead grid view)
                    filtered_segment = np.array(data_segment, dtype=float)
                    try:
                        # Prefer hardware sampling rate; fall back to 500.0 Hz
                        sampling_rate = 500.0
                        try:
                            if hasattr(self, "sampler") and hasattr(self.sampler, "sampling_rate") and self.sampler.sampling_rate > 10:
                                sampling_rate = float(self.sampler.sampling_rate)
                            elif hasattr(self, "sampling_rate") and self.sampling_rate > 10:
                                sampling_rate = float(self.sampling_rate)
                        except Exception:
                            pass

                        # EMG Filter
                        emg_applied = False
                        emg_suppresses_ac = False
                        emg_setting = self.settings_manager.get_setting("filter_emg", "150") if hasattr(self, "settings_manager") else "150"
                        if emg_setting and emg_setting.lower() != "off" and len(filtered_segment) >= 10:
                            from ecg.ecg_filters import apply_emg_filter
                            filtered_segment = apply_emg_filter(filtered_segment, sampling_rate, emg_setting)
                            emg_applied = True
                            try:
                                if float(emg_setting) < 60:
                                    emg_suppresses_ac = True
                            except ValueError:
                                pass

                        # AC Filter
                        ac_setting = self.settings_manager.get_setting("filter_ac", "50") if hasattr(self, "settings_manager") else "off"
                        if (not emg_applied or not emg_suppresses_ac) and ac_setting and ac_setting != "off" and len(filtered_segment) >= 10:
                            from ecg.ecg_filters import apply_ac_filter
                            filtered_segment = apply_ac_filter(filtered_segment, sampling_rate, ac_setting)
                    except Exception as filter_error:
                        print(f" Overlay AC filter skipped for lead {lead}: {filter_error}")

                    # Gaussian smoothing
                    try:
                        if len(filtered_segment) > 5:
                            sigma = max(self.SMOOTH_SIGMA * 1.5, 1.3)
                            filtered_segment = gaussian_filter1d(filtered_segment, sigma=sigma)
                    except Exception:
                        pass
                    
                    # Apply same baseline correction as main 12-lead grid view
                    raw = np.array(filtered_segment, dtype=float)
                    try:
                        # Initialize slow anchor if needed
                        if not hasattr(self, '_baseline_anchors'):
                            self._baseline_anchors = [0.0] * 12
                            self._baseline_alpha_slow = 0.0005  # Monitor-grade: ~4 sec time constant at 500 Hz
                        
                        if len(raw) > 0:
                            # Match main grid behavior: when low-frequency respiration/motion drift dominates
                            # AND a DFT cutoff is selected, apply DFT high-pass before baseline anchoring.
                            try:
                                from ecg.ecg_filters import apply_dft_filter, apply_baseline_wander_median_mean
                                dft_setting = None
                                if hasattr(self, "settings_manager"):
                                    dft_setting = str(self.settings_manager.get_setting("filter_dft", "off")).strip()
                                if dft_setting and dft_setting not in ("off", ""):
                                    # Always apply 0.5 Hz in streaming-safe mode when selected.
                                    # Gating on/off per frame can cause visible baseline "wobble" on real body signals.
                                    if str(dft_setting).strip() == "0.5":
                                        raw = apply_baseline_wander_median_mean(raw, float(sampling_rate))
                                    else:
                                        raw = apply_dft_filter(raw, float(sampling_rate), dft_setting)
                            except Exception:
                                pass

                            # Extract low-frequency baseline estimate (removes respiration 0.1-0.35 Hz)
                            baseline_estimate = self._extract_low_frequency_baseline(raw, sampling_rate)
                            
                            # Update anchor with slow EMA (tracks only very-low-frequency drift)
                            self._baseline_anchors[idx] = (1 - self._baseline_alpha_slow) * self._baseline_anchors[idx] + self._baseline_alpha_slow * baseline_estimate
                            
                            # Subtract anchor (NOT raw mean)
                            raw = raw - self._baseline_anchors[idx]
                            
                            # Final zero-centering to ensure perfect centering before gain (so baseline stays fixed at 2048/-2048)
                            current_dc = np.nanmean(raw) if len(raw) > 0 else 0.0
                            raw = raw - current_dc
                    except Exception as filter_error:
                        # Fallback: use original signal
                        print(f" Overlay baseline correction error for lead {lead}: {filter_error}")
                    
                    # Apply current gain setting (match main 12-lead grid)
                    gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                    
                    # Apply gain to zero-centered signal (only amplifies variations, baseline stays at zero)
                    centered = raw * gain_factor
                    centered = np.nan_to_num(centered, copy=False)
                    
                    # Center the wave: baseline stays at 2048/-2048, only variations (peaks) grow with gain
                    # Apply offset AFTER gain so baseline position is fixed, only amplitude variations scale
                    lead_name = self.leads[idx] if idx < len(self.leads) else ""
                    if lead_name == 'aVR':
                        centered = centered - 2048  # Center baseline at -2048 for AVR
                    else:
                        centered = centered + 2048  # Center baseline at 2048 for other leads
                    
                    # Debug logging for first lead in demo mode
                    if is_demo_mode and idx == 1:  # Lead II
                        print(f" Overlay demo mode: Lead {lead}, gain={gain_factor:.2f}, raw_range={np.max(np.abs(raw)):.1f}, gained_range={np.max(np.abs(centered)):.1f}")
                    
                    # Always represent the true sample window:
                    # - if we have >= buffer_len samples, show the newest buffer_len
                    # - if we are frozen in 12:1, left-align the captured snapshot so the
                    #   frozen trace fills the visible area without a leading gap
                    # - otherwise, if we have < buffer_len samples, right-align what we have
                    #   and leave leading NaNs for live scrolling behavior.
                    n = len(centered)
                    if n >= buffer_len:
                        plot_data[:] = centered[-buffer_len:]
                    elif use_snapshot and getattr(self, "_current_overlay_layout", "12x1") == "12x1":
                        plot_data[:n] = centered
                    elif n > 0:
                        plot_data[-n:] = centered
                    
                    # Set fixed Y-axis range: 0-4095 for non-AVR leads (centered at 2048), -4095-0 for AVR (centered at -2048)
                    # Same as main 12-lead grid view
                    lead_name = self.leads[idx] if idx < len(self.leads) else ""
                    if lead_name == 'aVR':
                        ymin, ymax = -4095, 0
                    else:
                        ymin, ymax = 0, 4095
                    
                    ax.set_ylim(ymin, ymax)
                else:
                    # Default range when no data
                    lead_name = self.leads[idx] if idx < len(self.leads) else ""
                    if lead_name == 'aVR':
                        ymin, ymax = -4095, 0
                    else:
                        ymin, ymax = 0, 4095
                    ax.set_ylim(ymin, ymax)
                
                # Set x-limits
                ax.set_xlim(0, max(buffer_len - 1, 1))
                line.set_ydata(plot_data)
        
        if hasattr(self, '_overlay_canvas'):
            self._overlay_canvas.draw_idle()

    def _apply_current_overlay_mode(self):

        if hasattr(self, '_current_overlay_mode'):
            self._apply_overlay_mode(self._current_overlay_mode)

    def _apply_overlay_mode(self, mode):
        
        if not hasattr(self, '_overlay_axes') or not self._overlay_axes:
            return
        
        # Store current mode
        self._current_overlay_mode = mode
        
        self._clear_all_backgrounds()
        
        # Update button highlighting
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
            }
        """
        
        active_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 3px solid #ff6600;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
                /* Removed unsupported box-shadow property */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 3px solid #ff8c42;
            }
        """
        
        # Reset all buttons to normal style
        self.light_mode_btn.setStyleSheet(button_style)
        self.dark_mode_btn.setStyleSheet(button_style)
        self.graph_mode_btn.setStyleSheet(button_style)
        
        # Highlight the active button
        if mode == "light":
            self.light_mode_btn.setStyleSheet(active_button_style)
            self._overlay_widget.setStyleSheet("""
                QWidget {
                    background: rgba(255, 255, 255, 0.95);
                    border: 2px solid #ff6600;
                    border-radius: 15px;
                }
            """)
            
            for ax in self._overlay_axes:
                ax.set_facecolor('#ffffff')
                ax.tick_params(axis='x', colors='#333333', labelsize=10)
                ax.tick_params(axis='y', colors='#333333', labelsize=10)
                ax.set_ylabel(ax.get_ylabel(), color='#333333', fontsize=14, fontweight='bold', labelpad=15)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            for line in self._overlay_lines:
                line.set_color('#0066cc')
                line.set_linewidth(0.7)
        
        elif mode == "dark":
            self.dark_mode_btn.setStyleSheet(active_button_style)
            self._overlay_widget.setStyleSheet("""
                QWidget {
                    background: rgba(0, 0, 0, 0.95);
                    border: 2px solid #ff6600;
                    border-radius: 15px;
                }
            """)
            
            for ax in self._overlay_axes:
                ax.set_facecolor('#000')
                ax.tick_params(axis='x', colors='#00ff00', labelsize=10)
                ax.tick_params(axis='y', colors='#00ff00', labelsize=10)
                ax.set_ylabel(ax.get_ylabel(), color='#00ff00', fontsize=14, fontweight='bold', labelpad=15)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            
            for line in self._overlay_lines:
                line.set_color('#00ff00')
                line.set_linewidth(0.7)
        
        elif mode == "graph":
            self.graph_mode_btn.setStyleSheet(active_button_style)
            self._apply_graph_mode()
        
        if hasattr(self, '_overlay_canvas'):
            self._overlay_canvas.draw_idle()

    def _clear_all_backgrounds(self):
        
        try:
            # Clear figure-level background
            if hasattr(self, '_overlay_canvas') and self._overlay_canvas.figure:
                fig = self._overlay_canvas.figure
                
                # IMPORTANT: Clear grid lines created by graph mode
                if hasattr(fig, '_grid_lines'):
                    for line in fig._grid_lines:
                        try:
                            line.remove()
                        except:
                            pass
                    fig._grid_lines = []
                
                if hasattr(fig, '_figure_background'):
                    try:
                        fig._figure_background.remove()
                        delattr(fig, '_figure_background')
                    except:
                        pass
                
                # Reset figure background to transparent
                fig.patch.set_facecolor('none')
            
            # Clear axis-level backgrounds
            if hasattr(self, '_overlay_axes'):
                for ax in self._overlay_axes:
                    if hasattr(ax, '_background_image'):
                        try:
                            ax._background_image.remove()
                            delattr(ax, '_background_image')
                        except:
                            pass
                    
                    # Reset axis background to transparent
                    ax.set_facecolor('none')
                    ax.patch.set_alpha(0.0)
                    
                    # Disable any grid on axes
                    ax.grid(False)
                    
        except Exception as e:
            print(f"Error clearing backgrounds: {e}")

    def _apply_graph_mode(self):
        """
        Apply graph mode with pink ECG grid lines drawn on the container background.
        Adjust minor grid density for 12x1 vs 6x2 overlay modes.
        """
        try:
            from matplotlib.collections import LineCollection

            bg_color = '#ffe7eb'
            minor_color = '#ffd1d1'
            major_color = '#ffb3b3'
            
            if hasattr(self, '_overlay_widget'):
                self._overlay_widget.setStyleSheet(f"""
                    QWidget {{
                        background: {bg_color};
                        border: 2px solid #ff6600;
                        border-radius: 15px;
                    }}
                """)

            # Apply pink grid background to the figure using Line2D
            if hasattr(self, '_overlay_canvas') and self._overlay_canvas.figure:
                fig = self._overlay_canvas.figure

                # Pink paper background
                fig.patch.set_facecolor(bg_color)

                # Clear any previous grid/background
                if hasattr(fig, '_grid_lines'):
                    for artist in fig._grid_lines:
                        try:
                            artist.remove()
                        except:
                            pass
                    fig._grid_lines = []

                # Remove any existing background image from figure
                if hasattr(fig, '_figure_background'):
                    try:
                        fig._figure_background.remove()
                        delattr(fig, '_figure_background')
                    except:
                        pass

                # --- True 1mm / 5mm grid based on figure physical size ---
                mm_per_inch = 25.4
                fig_width_in = fig.get_figwidth()
                fig_height_in = fig.get_figheight()
                width_mm = max(1.0, fig_width_in * mm_per_inch)
                height_mm = max(1.0, fig_height_in * mm_per_inch)

                # Fraction of figure for 1mm step in each direction
                minor_step_x = 1.0 / width_mm
                minor_step_y = 1.0 / height_mm
                v_lines_minor = []
                v_lines_major = []
                h_lines_minor = []
                h_lines_major = []

                # Vertical lines (time axis)
                x = 0.0
                idx = 0
                while x <= 1.0 + 1e-9:
                    is_major = (idx % 5 == 0)
                    if is_major:
                        v_lines_major.append([(x, 0), (x, 1)])
                    else:
                        v_lines_minor.append([(x, 0), (x, 1)])
                    x += minor_step_x
                    idx += 1

                # Horizontal lines (amplitude axis)
                y = 0.0
                jdx = 0
                while y <= 1.0 + 1e-9:
                    is_major = (jdx % 5 == 0)
                    if is_major:
                        h_lines_major.append([(0, y), (1, y)])
                    else:
                        h_lines_minor.append([(0, y), (1, y)])
                    y += minor_step_y
                    jdx += 1

                grid_artists = []
                
                # Add minor grid (drawn first, behind major)
                if v_lines_minor or h_lines_minor:
                    minor_lc = LineCollection(v_lines_minor + h_lines_minor, 
                                             colors=minor_color, linewidths=0.6, 
                                             alpha=0.9, zorder=0, transform=fig.transFigure)
                    fig.add_artist(minor_lc)
                    grid_artists.append(minor_lc)
                
                # Add major grid
                if v_lines_major or h_lines_major:
                    major_lc = LineCollection(v_lines_major + h_lines_major, 
                                             colors=major_color, linewidths=1.0, 
                                             alpha=0.9, zorder=0, transform=fig.transFigure)
                    fig.add_artist(major_lc)
                    grid_artists.append(major_lc)

                fig._grid_lines = grid_artists

                try:
                    import types

                    wave_speed_mm_s = 25.0
                    try:
                        if hasattr(self, "settings_manager"):
                            wave_speed_mm_s = 25.0 if getattr(self, '_overlay_active', False) else float(self.settings_manager.get_wave_speed())
                    except Exception:
                        wave_speed_mm_s = 25.0

                    sampling_rate_hz = 500.0
                    try:
                        if hasattr(self, "sampler") and hasattr(self.sampler, "sampling_rate") and self.sampler.sampling_rate and self.sampler.sampling_rate > 10:
                            sampling_rate_hz = float(self.sampler.sampling_rate)
                        elif hasattr(self, "sampling_rate") and self.sampling_rate and self.sampling_rate > 10:
                            sampling_rate_hz = float(self.sampling_rate)
                        elif hasattr(self, "demo_manager") and self.demo_manager and hasattr(self.demo_manager, "samples_per_second") and self.demo_manager.samples_per_second:
                            sampling_rate_hz = float(self.demo_manager.samples_per_second)
                    except Exception:
                        sampling_rate_hz = 500.0

                    samples_per_mm = sampling_rate_hz / max(1e-6, wave_speed_mm_s)

                    axis_width_mm = width_mm
                    axes = getattr(self, "_overlay_axes", [])
                    if axes:
                        axis_widths = []
                        for ax in axes:
                            try:
                                bbox = ax.get_position()
                                axis_widths.append(float(bbox.width) * width_mm)
                            except Exception:
                                pass
                        if axis_widths:
                            axis_width_mm = max(1.0, min(axis_widths))

                    target_buffer_len = int(round(axis_width_mm * samples_per_mm)) + 1
                    self._graph_mode_target_buffer_len = max(2, target_buffer_len)

                    if not hasattr(self, "_graph_mode_target_len_wrapped"):
                        self._graph_mode_original_get_overlay_target_buffer_len = self._get_overlay_target_buffer_len

                        def _graph_mode_get_overlay_target_buffer_len(this, is_demo_mode):
                            # Use the computed graph-mode target buffer length not only for
                            # explicit "graph" mode but also for the visual light/dark
                            # overlay modes so they show the same sample window.
                            if getattr(this, "_current_overlay_mode", None) in ("graph", "light", "dark"):
                                v = getattr(this, "_graph_mode_target_buffer_len", None)
                                if v:
                                    return int(v)
                            return this._graph_mode_original_get_overlay_target_buffer_len(is_demo_mode)

                        self._get_overlay_target_buffer_len = types.MethodType(_graph_mode_get_overlay_target_buffer_len, self)
                        self._graph_mode_target_len_wrapped = True
                except Exception as e:
                    print(f"Error applying graph mode scaling: {e}")

            # Make all axes transparent
            for ax in getattr(self, '_overlay_axes', []):
                ax.set_facecolor('none')
                ax.patch.set_alpha(0.0)
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_ylabel(ax.get_ylabel(), color='#333333',
                            fontsize=12, fontweight='bold', labelpad=12)

            # ECG line style
            for line in getattr(self, '_overlay_lines', []):
                line.set_color('#000000')
                line.set_linewidth(0.8)
                line.set_alpha(1.0)
                line.set_zorder(50)

            try:
                if getattr(self, "_current_overlay_layout", "12x1") == "6x2":
                    self._update_two_column_plots()
                else:
                    self._update_overlay_plots()
            except Exception:
                pass

        except Exception as e:
            print(f"Error applying graph mode: {e}")

    def _replace_plot_area_with_overlay(self):
        
        # Get the main horizontal layout
        main_layout = self.grid_widget.layout()
        
        # Find the main_vbox layout item (which contains the plot_area)
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.layout() and hasattr(item.layout(), 'indexOf') and item.layout().indexOf(self.plot_area) >= 0:
                # Found the layout containing plot_area
                main_vbox_layout = item.layout()
                
                # Find and replace the plot_area in main_vbox_layout
                plot_area_index = main_vbox_layout.indexOf(self.plot_area)
                if plot_area_index >= 0:
                    # Remove the plot_area
                    main_vbox_layout.removeWidget(self.plot_area)
                    self.plot_area.hide()
                    
                    # Add the overlay widget at the same position
                    main_vbox_layout.insertWidget(plot_area_index, self._overlay_widget)
                    return
        
        # Fallback: if we can't find the exact position, add to the end of main_vbox
        # Find the main_vbox layout
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.layout() and hasattr(item.layout(), 'indexOf') and item.layout().indexOf(self.plot_area) >= 0:
                main_vbox_layout = item.layout()
                main_vbox_layout.removeWidget(self.plot_area)
                self.plot_area.hide()
                main_vbox_layout.addWidget(self._overlay_widget)
                break

    def _restore_original_layout(self):
        
        if not hasattr(self, '_overlay_active') or not self._overlay_active:
            return
        
        # Stop overlay timer
        if hasattr(self, '_overlay_timer'):
            self._overlay_timer.stop()
            self._overlay_timer.deleteLater()
        
        # Find and remove overlay widget from main_vbox layout
        main_layout = self.grid_widget.layout()
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.layout() and hasattr(item.layout(), 'indexOf'):
                main_vbox_layout = item.layout()
                
                # Check if overlay widget is in this layout
                overlay_index = main_vbox_layout.indexOf(self._overlay_widget)
                if overlay_index >= 0:
                    # Remove overlay widget
                    main_vbox_layout.removeWidget(self._overlay_widget)
                    
                    # Restore original plot area at the exact same position
                    main_vbox_layout.insertWidget(overlay_index, self.plot_area)
                    self.plot_area.show()
                    break
        
        # Clean up overlay references
        if hasattr(self, '_overlay_widget'):
            self._overlay_widget.deleteLater()
            delattr(self, '_overlay_widget')
        
        if hasattr(self, '_overlay_axes'):
            delattr(self, '_overlay_axes')
        
        if hasattr(self, '_overlay_lines'):
            delattr(self, '_overlay_lines')
        
        if hasattr(self, '_overlay_canvas'):
            delattr(self, '_overlay_canvas')
        
        # Mark overlay as inactive and clear current layout type
        self._overlay_active = False
        if hasattr(self, "_current_overlay_layout"):
            self._current_overlay_layout = None
        
        # Force redraw of original plots
        self.redraw_all_plots()

    # ------------------------------------ 6 leads overlay --------------------------------------------

    def six_leads_overlay(self):
        if getattr(self, "_overlay_active", False):
            # If we're already in 6:2, treat this as a toggle (close overlay)
            if getattr(self, "_current_overlay_layout", None) == "6x2":
                self._restore_original_layout()
                self._current_overlay_layout = None
                return
            self._restore_original_layout()
        
        # Store the original plot area layout
        self._store_original_layout()
        
        # Create the 2-column overlay widget
        self._create_two_column_overlay_widget()
        
        # Replace the plot area with overlay
        self._replace_plot_area_with_overlay()
        
        # Mark overlay as active and record layout type
        self._overlay_active = True
        self._current_overlay_layout = "6x2"

        self._apply_current_overlay_mode()

        # Ensure demo data continues to work in overlay mode
        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            print("Demo mode active - overlay will show demo data")

    def _create_two_column_overlay_widget(self):
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QFrame
        
        # Create overlay container
        self._overlay_widget = QWidget()
        self._overlay_widget.setStyleSheet("""
            QWidget {
                background: #000;
                border: 2px solid #ff6600;
                border-radius: 15px;
            }
        """)
        
        # Main layout for overlay
        overlay_layout = QVBoxLayout(self._overlay_widget)
        overlay_layout.setContentsMargins(20, 20, 20, 20)
        overlay_layout.setSpacing(15)
        
        # Top control panel with close button and mode controls
        top_panel = QFrame()
        top_panel.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.1);
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 15px;
                padding: 10px;
            }
        """)
        top_layout = QHBoxLayout(top_panel)
        top_layout.setContentsMargins(15, 10, 15, 10)
        top_layout.setSpacing(20)
        
        # Close button
        close_btn = QPushButton("Close Overlay")
        close_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 2px solid #ff6600;
                border-radius: 10px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 2px solid #ff8c42;
            }
        """)
        close_btn.clicked.connect(self._restore_original_layout)
        
        # Mode control buttons with highlighting
        self.light_mode_btn = QPushButton("Light Mode")
        self.dark_mode_btn = QPushButton("Dark Mode")
        self.graph_mode_btn = QPushButton("Graph Mode")
        
        # Store current mode for highlighting
        self._current_overlay_mode = "graph"  # Default mode
        
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #45a049, stop:1 #4CAF50);
                border: 2px solid #45a049;
            }
        """
        
        active_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff6600, stop:1 #ff8c42);
                color: white;
                border: 3px solid #ff6600;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
                /* Removed unsupported box-shadow property */
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 #ff8c42, stop:1 #ff6600);
                border: 3px solid #ff8c42;
            }
        """
        
        self.light_mode_btn.setStyleSheet(button_style)
        self.dark_mode_btn.setStyleSheet(button_style)
        self.graph_mode_btn.setStyleSheet(button_style)
        
        # Add widgets to top panel
        top_layout.addWidget(close_btn)
        top_layout.addStretch()
        top_layout.addWidget(self.light_mode_btn)
        top_layout.addWidget(self.dark_mode_btn)
        top_layout.addWidget(self.graph_mode_btn)
        
        overlay_layout.addWidget(top_panel)
        
        # Create the 2-column matplotlib figure
        self._create_two_column_figure(overlay_layout)
        
        # Connect mode buttons
        self.light_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("light"))
        self.dark_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("dark"))
        self.graph_mode_btn.clicked.connect(lambda: self._apply_overlay_mode("graph"))

        # Defer initial graph mode until widget is rendered so figure dimensions are final
        QTimer.singleShot(50, lambda: self._apply_overlay_mode("graph"))

    def _create_two_column_figure(self, overlay_layout):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import numpy as np
        
        # Define the two columns of leads
        left_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
        right_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
        
        # Create figure with 2 columns and 6 rows
        fig = Figure(figsize=(16, 12), facecolor='none')
        # figsize = self._calculate_adaptive_figsize("6x2")
        # fig = Figure(figsize=figsize, facecolor='none')
        
        # Adjust subplot parameters for better spacing
        fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.02, hspace=0.05, wspace=0.1)
        
        self._overlay_axes = []
        self._overlay_lines = []
        
        # Create left column (limb leads)
        for idx, lead in enumerate(left_leads):
            ax = fig.add_subplot(6, 2, 2*idx + 1)
            ax.set_facecolor('none')
            
            # Remove all borders and spines

            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Remove all ticks and labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(lead, color='#00ff00', fontsize=12, fontweight='bold', labelpad=12)
            
            # Create line with initial data
            line, = ax.plot(np.arange(self.buffer_size), [np.nan]*self.buffer_size, color="#00ff00", lw=0.7)
            # Clipping stays ON. With ylim pinned to the full 0-4095 ADC range and
            # the +2048 centering, a normal R-wave at 20 mm/mV reaches ~4248 and,
            # unclipped, paints straight into the lane of the lead above it —
            # measured 51 spilled pixels at 20 mm/mV and 514 at 40 mm/mV. That
            # overlap is the "cluttered" trace: one lead's R-waves appearing in
            # another lead's row. If a taller trace is wanted, widen the y-limits
            # or clamp the data; do not let it draw outside its own axes.
            self._overlay_axes.append(ax)
            self._overlay_lines.append(line)
        
        # Create right column (chest leads)
        for idx, lead in enumerate(right_leads):
            ax = fig.add_subplot(6, 2, 2*idx + 2)
            ax.set_facecolor('none')
            
            # Remove all borders and spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Remove all ticks and labels for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(lead, color='#00ff00', fontsize=12, fontweight='bold', labelpad=12)
            
            # Create line with initial data
            line, = ax.plot(np.arange(self.buffer_size), [np.nan]*self.buffer_size, color="#00ff00", lw=0.7)
            # Clipping stays ON. With ylim pinned to the full 0-4095 ADC range and
            # the +2048 centering, a normal R-wave at 20 mm/mV reaches ~4248 and,
            # unclipped, paints straight into the lane of the lead above it —
            # measured 51 spilled pixels at 20 mm/mV and 514 at 40 mm/mV. That
            # overlap is the "cluttered" trace: one lead's R-waves appearing in
            # another lead's row. If a taller trace is wanted, widen the y-limits
            # or clamp the data; do not let it draw outside its own axes.
            self._overlay_axes.append(ax)
            self._overlay_lines.append(line)
        
        self._overlay_canvas = FigureCanvas(fig)
        self._overlay_canvas.setStyleSheet("background: transparent;")
        self._overlay_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        overlay_layout.addWidget(self._overlay_canvas, 1)
        
        # Start update timer for overlay
        self._overlay_timer = QTimer(self)
        self._overlay_timer.timeout.connect(self._update_two_column_plots)
        self._overlay_timer.start(100)

    def _update_two_column_plots(self):
        # If frozen, we only proceed if we have a valid snapshot to draw.
        # Otherwise, if frozen and no snapshot, return.
        if getattr(self, "_grid_frozen", False) and getattr(self, "_replay_snapshot", None) is None:
            return
        if not hasattr(self, '_overlay_lines') or not self._overlay_lines:
            return
        
        # Check if demo mode is active
        is_demo_mode = hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked()
        
        target_buffer_len = self._get_overlay_target_buffer_len(is_demo_mode)
        
        # Define the two columns of leads
        left_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]
        right_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]
        all_leads = left_leads + right_leads
        
        # Select data source based on frozen state
        use_snapshot = getattr(self, "_grid_frozen", False) and getattr(self, "_replay_snapshot", None) is not None

        for idx, lead in enumerate(all_leads):
            if idx < len(self._overlay_lines):
                if lead in self.leads:
                    lead_index = self.leads.index(lead)
                    if use_snapshot:
                        data = self._replay_snapshot[lead_index] if lead_index < len(self._replay_snapshot) else np.array([])
                    elif lead_index < len(self.data):
                        data = self.data[lead_index]
                    else:
                        data = np.array([])
                else:
                    data = np.array([])
                line = self._overlay_lines[idx]
                ax = self._overlay_axes[idx]
                
                # Ensure overlay line length matches current buffer size
                buffer_len = target_buffer_len
                try:
                    xdata = line.get_xdata()
                    current_len = len(xdata) if xdata is not None else 0
                    if current_len != buffer_len:
                        new_x = np.arange(buffer_len)
                        line.set_xdata(new_x)
                        current_len = buffer_len
                    if current_len:
                        buffer_len = current_len
                except Exception as e:
                    print(f" Overlay line sync error (6-lead): {e}")
                    buffer_len = target_buffer_len

                plot_data = np.full(buffer_len, np.nan)
                
                if data is not None and len(data) > 0:
                    # 6:2 overlay waves appear immediately at acquisition start.
                    try:
                        raw_data = np.asarray(data, dtype=float)
                        non_zero_indices = np.where(raw_data != 0)[0]
                        if len(non_zero_indices) > 0:
                            first_real_idx = int(non_zero_indices[0])
                            recent_data = raw_data[first_real_idx:]
                        else:
                            recent_data = raw_data[-min(buffer_len, len(raw_data)):] if len(raw_data) > 0 else raw_data
                    except Exception:
                        raw_data = np.asarray(data, dtype=float)
                        recent_data = raw_data

                    # Take exactly buffer_len samples from the tail of recent_data
                    # so overlay and main grid show the same active window.
                    if len(recent_data) >= buffer_len:
                        data_segment = recent_data[-buffer_len:]
                    else:
                        data_segment = recent_data
                    
                    # Optional AC notch filtering (match main 12-lead grid view)
                    filtered_segment = np.array(data_segment, dtype=float)
                    try:
                        sampling_rate = 500.0
                        try:
                            if hasattr(self, "sampler") and hasattr(self.sampler, "sampling_rate") and self.sampler.sampling_rate > 10:
                                sampling_rate = float(self.sampler.sampling_rate)
                            elif hasattr(self, "sampling_rate") and self.sampling_rate > 10:
                                sampling_rate = float(self.sampler.sampling_rate)
                        except Exception:
                            pass

                        # EMG Filter
                        emg_applied = False
                        emg_suppresses_ac = False
                        emg_setting = self.settings_manager.get_setting("filter_emg", "150") if hasattr(self, "settings_manager") else "150"
                        if emg_setting and emg_setting.lower() != "off" and len(filtered_segment) >= 10:
                            from ecg.ecg_filters import apply_emg_filter
                            filtered_segment = apply_emg_filter(filtered_segment, sampling_rate, emg_setting)
                            emg_applied = True
                            try:
                                if float(emg_setting) < 60:
                                    emg_suppresses_ac = True
                            except ValueError:
                                pass

                        # AC Filter
                        ac_setting = self.settings_manager.get_setting("filter_ac", "50") if hasattr(self, "settings_manager") else "off"
                        if (not emg_applied or not emg_suppresses_ac) and ac_setting and ac_setting != "off" and len(filtered_segment) >= 10:
                            from ecg.ecg_filters import apply_ac_filter
                            filtered_segment = apply_ac_filter(filtered_segment, sampling_rate, ac_setting)
                    except Exception as filter_error:
                        print(f" 6:2 overlay AC filter skipped for lead {lead}: {filter_error}")

                    # Gaussian smoothing
                    try:
                        if len(filtered_segment) > 5:
                            sigma = max(self.SMOOTH_SIGMA * 1.5, 1.3)
                            filtered_segment = gaussian_filter1d(filtered_segment, sigma=sigma)
                    except Exception:
                        pass
                    
                    # Apply same baseline correction as main 12-lead grid view
                    raw = np.array(filtered_segment, dtype=float)
                    try:
                        # Initialize slow anchor if needed
                        if not hasattr(self, '_baseline_anchors'):
                            self._baseline_anchors = [0.0] * 12
                            self._baseline_alpha_slow = 0.0005  # Monitor-grade: ~4 sec time constant at 500 Hz
                        
                        if len(raw) > 0:
                            # Match main grid behavior: when low-frequency respiration/motion drift dominates
                            # AND a DFT cutoff is selected, apply DFT high-pass before baseline anchoring.
                            try:
                                from ecg.ecg_filters import apply_dft_filter, apply_baseline_wander_median_mean
                                dft_setting = None
                                if hasattr(self, "settings_manager"):
                                    dft_setting = str(self.settings_manager.get_setting("filter_dft", "off")).strip()
                                if dft_setting and dft_setting not in ("off", ""):
                                    if str(dft_setting).strip() == "0.5":
                                        raw = apply_baseline_wander_median_mean(raw, float(sampling_rate))
                                    else:
                                        raw = apply_dft_filter(raw, float(sampling_rate), dft_setting)
                            except Exception:
                                pass

                            # Extract low-frequency baseline estimate (removes respiration 0.1-0.35 Hz)
                            baseline_estimate = self._extract_low_frequency_baseline(raw, sampling_rate)
                            
                            # Update anchor with slow EMA (tracks only very-low-frequency drift)
                            self._baseline_anchors[lead_index] = (1 - self._baseline_alpha_slow) * self._baseline_anchors[lead_index] + self._baseline_alpha_slow * baseline_estimate
                            
                            # Subtract anchor (NOT raw mean)
                            raw = raw - self._baseline_anchors[lead_index]
                            
                            # Final zero-centering to ensure perfect centering before gain (so baseline stays fixed at 2048/-2048)
                            current_dc = np.nanmean(raw) if len(raw) > 0 else 0.0
                            raw = raw - current_dc
                    except Exception as filter_error:
                        # Fallback: use original signal
                        print(f" 6:2 overlay baseline correction error for lead {lead}: {filter_error}")
                    
                    # Apply current gain setting (match main 12-lead grid)
                    gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                    
                    # Apply gain to zero-centered signal (only amplifies variations, baseline stays at zero)
                    centered = raw * gain_factor
                    centered = np.nan_to_num(centered, copy=False)
                    
                    # Center the wave: baseline stays at 2048/-2048, only variations (peaks) grow with gain
                    # Apply offset AFTER gain so baseline position is fixed, only amplitude variations scale
                    if lead == 'aVR':
                        centered = centered - 2048  # Center baseline at -2048 for AVR
                    else:
                        centered = centered + 2048  # Center baseline at 2048 for other leads
                    
                    # Debug logging for first lead in demo mode
                    if is_demo_mode and idx == 1:  # Lead II
                        print(f" 6:2 Overlay demo mode: Lead {lead}, gain={gain_factor:.2f}, raw_range={np.max(np.abs(raw)):.1f}, gained_range={np.max(np.abs(centered)):.1f}")
                    
                    # Always represent the true sample window:
                    # - if we have >= buffer_len samples, show the newest buffer_len
                    # - if we are frozen in 12:1, left-align the captured snapshot so the
                    #   frozen trace fills the visible area without a leading gap
                    # - otherwise, if we have < buffer_len samples, right-align what we have
                    #   and leave leading NaNs for live scrolling behavior.
                    n = len(centered)
                    if n >= buffer_len:
                        plot_data[:] = centered[-buffer_len:]
                    elif use_snapshot and getattr(self, "_current_overlay_layout", "12x1") == "12x1":
                        plot_data[:n] = centered
                    elif n > 0:
                        plot_data[-n:] = centered
                    
                    # Set fixed Y-axis range: 0-4095 for non-AVR leads (centered at 2048), -4095-0 for AVR (centered at -2048)
                    # Same as main 12-lead grid view
                    if lead == 'aVR':
                        ymin, ymax = -4095, 0
                    else:
                        ymin, ymax = 0, 4095
                    
                    ax.set_ylim(ymin, ymax)
                else:
                    # Default range when no data
                    if lead == 'aVR':
                        ymin, ymax = -4095, 0
                    else:
                        ymin, ymax = 0, 4095
                    ax.set_ylim(ymin, ymax)
                
                # Set x-limits
                ax.set_xlim(0, max(buffer_len - 1, 1))
                line.set_ydata(plot_data)
        
        if hasattr(self, '_overlay_canvas'):
            self._overlay_canvas.draw_idle()

    def interpolate(self, signal, factor):
        try:
            signal = np.asarray(signal, dtype=float)
            if signal.size < 3 or factor <= 1:
                return signal
            x = np.arange(signal.size, dtype=float)
            xi = np.linspace(0.0, float(signal.size - 1), int(signal.size * factor), dtype=float)
            # Linear interpolation is cheaper than cubic and avoids overshoot during fast updates.
            return np.interp(xi, x, signal)
        except Exception:
            # Fallback to linear
            signal = np.asarray(signal, dtype=float)
            x = np.arange(len(signal), dtype=float)
            xi = np.linspace(0.0, float(len(signal) - 1), int(len(signal) * max(1, factor)), dtype=float)
            return np.interp(xi, x, signal)

    def set_display_mode(self, mode):
        """Set the display mode: 'scroll' or 'sweep'"""
        if mode in ['scroll', 'sweep']:
            self.display_mode = mode

    def _set_holter_button_state(self, enabled: bool):
        holter_btn = None
        for btn, text in self.menu_buttons:
            if text == "Holter" or btn.text().startswith("Holter"):
                holter_btn = btn
                break

        if not holter_btn:
            return

        if enabled:
            holter_btn.setText("Holter: ON")
            holter_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #1b5e20;
                    border: 2px solid #2e7d32;
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 12px;
                    font-weight: bold;
                    text-align: left;
                    margin: 2px 0;
                }
            """)
        else:
            holter_btn.setText("Holter")
            holter_btn.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #1a1a1a;
                    border: 2px solid #e9ecef;
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 12px;
                    font-weight: bold;
                    text-align: left;
                    margin: 2px 0;
                }
            """)

        if hasattr(self, 'generate_report_btn'):
            self.generate_report_btn.setVisible(not enabled)

    def _show_holter_ui(self, is_recording=True, ecgh_path=None, show_record_mgmt=False):
        """Open the professional Holter analysis workspace."""
        if not HOLTER_AVAILABLE:
            QMessageBox.warning(self, "Holter Monitor", "Holter module is not available.")
            return

        if self._holter_ui is None:
            self._holter_ui = HolterMainWindow(
                parent=self,
                session_dir="",
                patient_info=self._holter_patient_info or {},
                writer=self._holter_writer if is_recording else None,
                live_source=self if is_recording else None,
            )
        
        if not is_recording and ecgh_path:
            # Review mode - ecgh_path is the file, but load_completed_session expects session_dir
            session_dir = os.path.dirname(ecgh_path)
            self._holter_ui.load_completed_session(session_dir)
        elif not is_recording:
            # When opening "View Previous Recording", default to the newest
            # completed session so the latest recorded values are visible
            # immediately instead of keeping an older cached session loaded.
            try:
                from .holter.holter_ui import _resolve_recordings_dir, _find_latest_completed_session
                recordings_dir = _resolve_recordings_dir(getattr(self._holter_ui, 'session_dir', ''))
                latest_session = _find_latest_completed_session(recordings_dir)
                if latest_session:
                    self._holter_ui.load_completed_session(latest_session)
            except Exception:
                pass
        
        if show_record_mgmt and hasattr(self._holter_ui, '_record_mgmt_panel'):
            try:
                from .holter.holter_ui import _resolve_recordings_dir
                self._holter_ui._record_mgmt_panel.output_dir = _resolve_recordings_dir(
                    getattr(self._holter_ui, 'session_dir', '')
                )
                self._holter_ui._record_mgmt_panel.refresh_records()
            except Exception:
                pass

        if show_record_mgmt and hasattr(self._holter_ui, '_tabs'):
            for i in range(self._holter_ui._tabs.count()):
                tab_text = self._holter_ui._tabs.tabText(i)
                if "Record Mgmt" in tab_text or "Manage Records" in tab_text or "RECORDINGS" in tab_text:
                    self._holter_ui._tabs.setCurrentIndex(i)
                    break

        self._holter_ui.showMaximized()
        self._holter_ui.raise_()
        self._holter_ui.activateWindow()

    def start_live_holter_from_dashboard(self):
        """Automatically enable Holter mode and start acquisition in inner dashboard"""
        if not HOLTER_AVAILABLE:
            return

        # CRITICAL: Clean up any stale HolterMainWindow from previous "View Recording" session
        # This ensures we always show the inner dashboard (ECG control panel), not a review window
        if self._holter_ui is not None:
            try:
                self._holter_ui.close()
            except Exception:
                pass
            self._holter_ui = None

        # 1. Enable Holter Mode
        self.holter_mode_enabled = True
        self._set_holter_button_state(True)
        
        # 2. Get patient info from dashboard if available
        try:
            if hasattr(self.stacked_widget.parent(), 'patient_details'):
                self._holter_patient_info = dict(self.stacked_widget.parent().patient_details or {})
        except Exception:
            pass

        # 3. Start acquisition (Demo or Real)
        if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
            # If demo mode is already ON, just start the Holter session
            self._start_holter_session_internal()
        else:
            # Try to start real acquisition
            self.start_acquisition()
            
            QMessageBox.information(
                self,
                "Comprehensive ECG Analysis Started",
                "Live Comprehensive ECG Analysis recording has started automatically."
            )

    def show_holter_menu(self):
        """Open professional Holter setup and workspace."""
        if not HOLTER_AVAILABLE:
            QMessageBox.warning(self, "Holter Monitor", "Holter module is not available.")
            return

        # If Holter is already armed AND currently recording (not a stale review window), reopen it
        if (self.holter_mode_enabled and self._holter_ui is not None and 
            self._holter_writer and self._holter_writer.is_running):
            self._holter_ui.showMaximized()
            self._holter_ui.raise_()
            self._holter_ui.activateWindow()
            return

        default_info = {}
        try:
            if hasattr(self.stacked_widget.parent(), 'patient_details'):
                default_info = dict(self.stacked_widget.parent().patient_details or {})
        except Exception:
            default_info = {}

        if self._holter_patient_info:
            default_info.update(self._holter_patient_info)

        os.makedirs(self._holter_output_dir, exist_ok=True)
        
        # User requested to bypass the prompt when starting and only ask on stop
        self._holter_patient_info = default_info
        self._holter_duration_hours = 24
        
        self.holter_mode_enabled = True
        self._set_holter_button_state(True)

        acquisition_running = bool(
            (self.serial_reader and getattr(self.serial_reader, 'running', False))
            or (hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked())
        )

        if acquisition_running and not (self._holter_writer and self._holter_writer.is_running):
            self._start_holter_session_internal()

        if self._holter_ui is None or not self._holter_ui.isVisible():
            self._holter_ui = HolterMainWindow(
                parent=self,
                session_dir=self._holter_writer.session_dir if self._holter_writer else "",
                patient_info=self._holter_patient_info,
                writer=self._holter_writer,
                live_source=self,
                duration_hours=self._holter_duration_hours,
            )
        else:
            self._holter_ui.patient_info = self._holter_patient_info
            self._holter_ui.attach_writer(self._holter_writer,
                                          self._holter_writer.session_dir if self._holter_writer else "",
                                          self._holter_patient_info)

        self._holter_ui.showMaximized()
        self._holter_ui.raise_()
        self._holter_ui.activateWindow()

        if not acquisition_running:
            QMessageBox.information(
                self,
                "Holter Mode Ready",
                "Comprehensive ECG Analysis mode is armed.\n\nClick Start (or enable Demo Mode) to begin recording. "
                "The professional 12‑lead comprehensive ECG workspace has been opened and the report will be generated when the recording stops."
            )

    def _start_holter_session_internal(self):
        """Internal helper to start a Holter session during acquisition"""
        if self._holter_writer and self._holter_writer.is_running:
            return

        # Get patient info from setup dialog or dashboard
        patient_info = dict(self._holter_patient_info or {})
        if not patient_info:
            try:
                if hasattr(self.stacked_widget.parent(), 'patient_details'):
                    patient_info = dict(self.stacked_widget.parent().patient_details or {})
            except Exception:
                patient_info = {}

        # Use configured recording directory
        out_dir = self._holter_output_dir or os.path.join(os.getcwd(), "recordings")
        os.makedirs(out_dir, exist_ok=True)
        
        # Create the stream writer
        self._holter_writer = HolterStreamWriter(
            output_dir=out_dir,
            patient_info=patient_info,
            fs=500,
            chunk_seconds=5,
        )
        
        # Start recording
        session_dir = self._holter_writer.start()
        
        # Start analysis worker thread
        self._holter_worker = HolterAnalysisWorker(
            analysis_queue=self._holter_writer.analysis_queue,
            on_chunk_done=self._holter_writer.update_live_analysis,
            fs=500
        )
        self._holter_writer.attach_analysis_worker(self._holter_worker)
        self._holter_worker.start()
        
        # Show Holter badge
        if hasattr(self, 'holter_badge'):
            self.holter_badge.setVisible(True)

        if self._holter_ui:
            try:
                self._holter_ui.attach_writer(self._holter_writer, session_dir, patient_info)
                self._holter_ui.show()
            except Exception as e:
                print(f"[Holter] UI attach warning: {e}")
            
        print(f"[Comprehensive ECG Analysis] Automatic recording started in {session_dir}")

    def _stop_holter_session_internal(self):
        """Internal helper to stop a Holter session and generate report"""
        if self._holter_writer and self._holter_writer.is_running:
            summary = self._holter_writer.stop()
            
            # Stop the worker thread
            if self._holter_worker:
                self._holter_worker.stop(wait=True)
            
            # Show dialog to collect patient info AFTER recording
            dialog = HolterStartDialog(self, patient_info=summary.get('patient_info', {}), output_dir=self._holter_output_dir)
            dialog.setWindowTitle("Save Comprehensive ECG Analysis Details")
            if dialog.exec_() == QDialog.Accepted:
                patient_info, dur, out_dir = dialog.get_result()
                summary['patient_info'] = patient_info
                import json
                try:
                    with open(os.path.join(summary.get('session_dir', ''), "patient.json"), 'w') as f:
                        json.dump(patient_info, f, indent=4)
                except Exception as e:
                    print(f"Failed to save patient.json: {e}")

            self._holter_writer = None
            self._holter_worker = None
            self.holter_mode_enabled = False
            self._set_holter_button_state(False)
            
            # Hide Holter badge
            if hasattr(self, 'holter_badge'):
                self.holter_badge.setVisible(False)

            # --- NAVIGATE BACK TO DASHBOARD ---
            try:
                if hasattr(self, 'stacked_widget'):
                    # Switch back to the dashboard page (index 0)
                    self.stacked_widget.setCurrentIndex(0)
                    print("[Holter] Navigated back to dashboard after session stop")
            except Exception as e:
                print(f"Error returning to dashboard: {e}")
            # ----------------------------------

            if self._holter_ui:
                try:
                    self._holter_ui.load_completed_session(summary['session_dir'], summary['patient_info'])
                    # We show it after navigation so it stays on top for review
                    self._holter_ui.showMaximized()
                    self._holter_ui.raise_()
                    self._holter_ui.activateWindow()
                except Exception as e:
                    print(f"[Holter] UI completion warning: {e}")

    def _has_respiration_baseline_drift(self, signal: np.ndarray, fs: float) -> bool:
        """
        Display-only detector for respiration/motion baseline drift.

        Returns True when the <0.5 Hz component is a large fraction of the signal
        variance in the current window, which is the condition that causes the
        trace to visually "float" up/down even when EMG/AC filtering is enabled.
        """
        try:
            arr = np.asarray(signal, dtype=float)
            if arr.size < 250 or float(fs) <= 5.0:
                return False

            # Remove DC so the ratio reflects drift rather than a static offset.
            dc = float(np.nanmean(arr)) if arr.size else 0.0
            if np.isfinite(dc):
                arr = arr - dc
            arr = np.nan_to_num(arr, copy=False)

            from scipy.signal import butter, filtfilt

            nyq = float(fs) / 2.0
            if nyq <= 0.0:
                return False

            cutoff_hz = min(0.5, 0.45 * nyq)
            if cutoff_hz <= 0.0 or cutoff_hz >= nyq:
                return False

            b, a = butter(2, cutoff_hz / nyq, btype="low")
            baseline = filtfilt(b, a, arr)
            baseline_var = float(np.var(baseline))
            total_var = float(np.var(arr))
            if total_var <= 1e-9:
                return False

            # Lower threshold so we also catch "small but visible" baseline float.
            return (baseline_var / total_var) > 0.10
        except Exception:
            return False

    def update_plots(self):
        """Update all ECG plots with current data using PyQtGraph (GitHub version)"""
        if getattr(self, '_plot_update_in_progress', False):
            return
        self._plot_update_in_progress = True
        try:
            # ── FIX: Skip rendering during report snapshot — prevents 12-box wave deform ──
            if getattr(self, '_report_generating', False):
                return
            # ─────────────────────────────────────────────────────────────────────────────

            # Memory management - check every N updates
            self.update_count += 1
            if self.update_count % self.memory_check_interval == 0:
                self._manage_memory()
            render_stride = max(1, int(getattr(self, '_plot_render_stride', 2)))
            render_now = (self.update_count % render_stride) == 0

            # DEMO branch
            if not self.serial_reader or not self.serial_reader.running:
                if not render_now:
                    return
                try:
                    wave_speed = 25.0 if getattr(self, '_overlay_active', False) else float(self.settings_manager.get_wave_speed())
                except Exception:
                    wave_speed = 25.0

                # 25 mm/s → 3 s window in 12‑lead view; scale with speed
                baseline_seconds = 3.0
                seconds_scale = (25.0 / max(1e-6, wave_speed))
                seconds_to_show = baseline_seconds * seconds_scale

                for i in range(len(self.data_lines)):
                    try:
                        if i < len(self.data):
                            raw = np.asarray(self.data[i])

                            # ── Resolve fs FIRST — used by every pipeline step below ──
                            fs = 500
                            if hasattr(self, 'sampler') and getattr(self.sampler, 'sampling_rate', None):
                                try:
                                    fs = float(self.sampler.sampling_rate)
                                except Exception:
                                    fs = 500

                            # --- PIPELINE STEP 1: Optional 50/60 Hz AC filter (respect Set Filter) ---
                            try:
                                from ecg.ecg_filters import apply_ac_filter
                                ac_setting = None
                                if hasattr(self, "settings_manager"):
                                    ac_setting = str(self.settings_manager.get_setting("filter_ac", "50")).strip()
                                if ac_setting in ("50", "60") and len(raw) > 30:
                                    raw = apply_ac_filter(raw, float(fs), ac_setting)
                            except Exception:
                                # On any error, fall back to unfiltered raw so display never breaks
                                pass

                            # --- PIPELINE STEP 2: Gaussian Smoothing ---
                            try:
                                if len(raw) > 5:
                                    raw = gaussian_filter1d(raw, sigma=self.SMOOTH_SIGMA)
                            except Exception:
                                pass
                            # ------------------------------------------

                            gain = 1.0
                            try:
                                gain = get_display_gain(self.settings_manager.get_wave_gain())
                            except Exception:
                                pass
                            # 🫀 DISPLAY: Low-frequency baseline anchor (removes respiration from baseline)
                            # Extract very-low-frequency baseline (< 0.3 Hz) to prevent baseline from "breathing"
                            try:
                                # Initialize slow anchor if needed
                                if not hasattr(self, '_baseline_anchors'):
                                    self._baseline_anchors = [0.0] * 12
                                    self._baseline_alpha_slow = 0.0005  # Monitor-grade: ~4 sec time constant at 500 Hz

                                if len(raw) > 0:
                                    # Respiration baseline drift stop (display-only):
                                    # If slow respiration/motion drift dominates this window AND the user has a DFT
                                    # cutoff selected, apply the DFT high-pass before baseline anchoring.
                                    try:
                                        from ecg.ecg_filters import apply_dft_filter, apply_baseline_wander_median_mean
                                        if hasattr(self, "_last_display_dft_applied") and i < len(self._last_display_dft_applied):
                                            self._last_display_dft_applied[i] = False
                                        dft_setting = None
                                        if hasattr(self, "settings_manager"):
                                            dft_setting = str(self.settings_manager.get_setting("filter_dft", "off")).strip()
                                        if dft_setting and dft_setting not in ("off", ""):
                                            # Always apply 0.5 Hz in streaming-safe mode when selected.
                                            # Per-frame gating can cause visible baseline wobble on body signals.
                                            if str(dft_setting).strip() == "0.5":
                                                raw = apply_baseline_wander_median_mean(raw, float(fs))
                                            else:
                                                raw = apply_dft_filter(raw, float(fs), dft_setting)
                                            if hasattr(self, "_last_display_dft_applied") and i < len(self._last_display_dft_applied):
                                                self._last_display_dft_applied[i] = True
                                    except Exception:
                                        pass

                                    # Extract low-frequency baseline estimate (removes respiration 0.1-0.35 Hz)
                                    baseline_estimate = self._extract_low_frequency_baseline(raw, fs)

                                    # Update anchor with slow EMA (tracks only very-low-frequency drift)
                                    self._baseline_anchors[i] = (1 - self._baseline_alpha_slow) * self._baseline_anchors[i] + self._baseline_alpha_slow * baseline_estimate

                                    # Subtract anchor (NOT raw mean, NOT baseline estimate directly)
                                    raw = raw - self._baseline_anchors[i]

                                    # Final zero-centering to ensure perfect centering before gain
                                    current_dc = np.nanmean(raw) if len(raw) > 0 else 0.0
                                    raw = raw - current_dc
                            except Exception as filter_error:
                                # Fallback: use original signal — silent, no print spam
                                pass

                            # Apply gain to zero-centered signal
                            raw = raw * gain
                            # Trim small margins from both ends to reduce filter edge artefacts
                            try:
                                edge_trim = int(0.5 * fs)  # ~200ms on each side at 500 Hz
                                if len(raw) > 2 * edge_trim:
                                    raw = raw[edge_trim:-edge_trim]
                            except Exception:
                                pass

                            window_len = int(max(50, min(len(raw), seconds_to_show * fs)))
                            
                            # Handle Display Mode: Scroll vs Sweep
                            if getattr(self, 'display_mode', 'scroll') == 'sweep' and len(raw) >= window_len:
                                # Sweep Mode: use an internal capped cursor so app focus changes
                                # or slow frames do not cause a huge visual jump on resume.
                                if not hasattr(self, '_sweep_cursor_state'):
                                    self._sweep_cursor_state = {}
                                sweep_key = ('serial', i)
                                now_mono = time.monotonic()
                                sweep_state = self._sweep_cursor_state.get(sweep_key, {})
                                prev_cursor = int(sweep_state.get('cursor', window_len - 1))
                                prev_ts = float(sweep_state.get('ts', now_mono))
                                raw_step = max(1, int(round((now_mono - prev_ts) * fs)))
                                step = min(raw_step, max(1, int(0.08 * fs)))
                                cursor = (prev_cursor + step) % window_len
                                self._sweep_cursor_state[sweep_key] = {
                                    'cursor': cursor,
                                    'ts': now_mono,
                                    'window_len': window_len,
                                }
                                
                                # Take the last window_len samples (the current window of data)
                                raw_window = raw[-window_len:]
                                
                                # Shift to align newest data (at end of raw_window) to cursor
                                # shift = cursor - (window_len - 1)
                                src = np.roll(raw_window, cursor - (window_len - 1))
                                
                                # Insert a small gap of NaNs at the cursor to visualize the sweep bar
                                gap_size = int(0.05 * fs) # 50ms gap
                                if gap_size > 0:
                                    # Create a gap AHEAD of the cursor (the "erase bar")
                                    # Clear data from cursor to cursor + gap
                                    gap_end = min(len(src), cursor + gap_size)
                                    src[cursor:gap_end] = np.nan
                                    
                                    # If gap wraps around
                                    if cursor + gap_size > len(src):
                                        rem = (cursor + gap_size) - len(src)
                                        src[:rem] = np.nan 
                            else:
                                # Scroll Mode (Default): Show the most recent data
                                src = raw[-window_len:]

                            display_len = self.buffer_size if hasattr(self, 'buffer_size') else 1000
                            # --- Flatline detection (display-only) ---
                            # If the recent window for this lead has almost no variation, treat as flatline
                            if src.size >= 50:
                                amp_range = float(np.nanmax(src) - np.nanmin(src))
                                std_val = float(np.nanstd(src))
                                # Thresholds are in display units after gain; very small range and std → flatline
                                is_flat = amp_range < 5.0 and std_val < 1.0
                                lead_name = self.leads[i] if i < len(self.leads) else f"Lead {i+1}"
                                if is_flat and not self._flatline_alert_shown[i]:
                                    self._flatline_alert_shown[i] = True
                                elif not is_flat:
                                    # Reset flag when signal returns
                                    self._flatline_alert_shown[i] = False

                            if src.size < 2:
                                resampled = np.zeros(display_len)
                            else:
                                # --- PIPELINE STEP 3: Interpolation ---
                                try:
                                    # Use fixed factor interpolation (4x) for high-res smoothness
                                    resampled = self.interpolate(src, self.INTERP_FACTOR)
                                except Exception:
                                    # Fallback to display_len if helper fails
                                    x_src = np.linspace(0.0, 1.0, src.size)
                                    x_dst = np.linspace(0.0, 1.0, display_len)
                                    resampled = np.interp(x_dst, x_src, src)
                                # --------------------------------------
                            
                            # Signal is already zero-centred by the baseline subtraction above.
                            # Y-axis is symmetric (-2048 to +2048) for ALL leads — no offset needed.
                            # aVR will naturally sit below baseline due to its formula (-(I+II)/2).

                            # Generate X-axis to ensure correct time scaling on the 0-10s plot
                            # We map the data to [0, seconds_to_show]
                            x_axis = np.linspace(0, seconds_to_show, len(resampled))
                            
                            # Use connect='finite' to handle NaNs (breaks the line at gaps)
                            self.data_lines[i].setData(x_axis, resampled, connect='finite')
                            self.update_plot_y_range(i)
                    except Exception as e:
                        print(f" Error updating plot {i}: {e}")
                        continue
                return

            # SERIAL branch - NEW PACKET-BASED PARSING
            packets_processed = 0
            # At 500 Hz with 20ms timer interval, we need to read 10 packets per cycle
            # CRITICAL: Increase max_packets to allow catching up if we fall behind
            # Process more packets per cycle to prevent buffer accumulation and packet loss
            max_packets = 100  # Increased from 50 to process more packets and catch up faster
            
            # Track packet loss detection
            if not hasattr(self, '_last_packet_count'):
                self._last_packet_count = 0
                self._last_packet_time = time.time()
                self._expected_packets_per_second = 500  # Hardware sends at 500 Hz
            
            # Check if we're using the new packet-based reader
            is_packet_reader = isinstance(self.serial_reader, SerialStreamReader)
            
            if is_packet_reader:
                # NEW: Use packet-based reading with error handling for 500 Hz
                try:
                    # Check connection state before reading
                    was_running = self.serial_reader.running

                    # OPTIMIZED FOR 500 Hz: Read packets aggressively to prevent buffer overflow
                    packets = self.serial_reader.read_packets(max_packets=max_packets)

                    # Check if reader stopped due to error during read_packets (Disconnection detection)
                    if was_running and not self.serial_reader.running:
                        print("CRITICAL: Device disconnected during test!")
                        self.stop_acquisition()
                        return
                    
                    # Safety check: If buffer is accumulating too fast, warn and clear it
                    if hasattr(self.serial_reader, 'buf') and len(self.serial_reader.buf) > 100000:
                        print(f" ⚠️ CRITICAL: Buffer overflow risk ({len(self.serial_reader.buf)} bytes) - clearing buffer")
                        self.serial_reader.buf.clear()
                    
                    # Detect packet loss - comprehensive monitoring
                    current_time = time.time()
                    if hasattr(self.serial_reader, 'data_count'):
                        current_packet_count = self.serial_reader.data_count
                        time_elapsed = current_time - self._last_packet_time
                        
                        if time_elapsed >= 1.0:  # Check every second
                            packets_received = current_packet_count - self._last_packet_count
                            expected_packets = int(self._expected_packets_per_second * time_elapsed)
                            packet_loss = max(0, expected_packets - packets_received)
                            packet_loss_percent = (packet_loss / expected_packets * 100) if expected_packets > 0 else 0
                            
                            # Get overall statistics from SerialStreamReader
                            overall_loss_percent = 0.0
                            if hasattr(self.serial_reader, 'packet_loss_percent'):
                                overall_loss_percent = self.serial_reader.packet_loss_percent
                            
                            # Alert on packet loss (only log significant issues to reduce console spam)
                            if packet_loss_percent > 10.0:  # Only warn on >10% loss (reduced verbosity)
                                if not hasattr(self, '_packet_loss_warned') or (current_time - self._packet_loss_warned) > 5.0:
                                    print(f" ⚠️ Packet loss: {packet_loss}/{expected_packets} ({packet_loss_percent:.1f}%)")
                                    self._packet_loss_warned = current_time
                            
                            # Periodic status report (every 30 seconds - reduced frequency)
                            if not hasattr(self, '_last_status_report'):
                                self._last_status_report = current_time
                            if current_time - self._last_status_report >= 30.0:  # Reduced from 10s to 30s
                                if overall_loss_percent > 5.0:  # Only report if significant loss
                                    print(f" 📊 Packet Stats: {current_packet_count} received, {self.serial_reader.total_packets_lost} lost ({overall_loss_percent:.2f}%)")
                                self._last_status_report = current_time
                            
                            self._last_packet_count = current_packet_count
                            self._last_packet_time = current_time
                    
                    # One list per lead for this tick, committed with a single
                    # slice assignment per lead once the batch is consumed. Every
                    # sample is kept; this defers the copy, it never drops or
                    # reorders a sample.
                    _staged = [[] for _ in range(len(self.data))]

                    for packet in packets:
                        self._lead_data_received = True
                        # Packet contains all 12 leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
                        # Map packet dict to our lead order
                        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

                        # ---------------- Lead-off latch / debouncer ----------------
                        # Some firmware chatters the per-lead "connected" bit when RL/DRL is removed,
                        # causing the "Leads:" banner (and downstream BPM/metrics) to flicker.
                        # Latch into a stable global OFF state when most leads are off for a few
                        # consecutive packets, and only unlatch after a sustained all-connected run.
                        try:
                            raw_off_count = sum(1 for ld in lead_order if packet.get(ld) is None)
                        except Exception:
                            raw_off_count = 0

                        global_threshold = 12  # Only latch global OFF if ALL 12 leads are off
                        raw_ra_present = bool(packet.get('__ra_present__', True))
                        raw_la_present = bool(packet.get('__la_present__', True))
                        raw_ll_present = bool(packet.get('__ll_present__', True))
                        raw_limb_active = raw_ra_present and (raw_la_present or raw_ll_present)
                        limb_active = raw_limb_active or ((packet.get('I') is not None) or (packet.get('II') is not None))

                        # Start the same 10-second report lockout immediately when LA or LL drops.
                        # This greys out Generate Report right away on accidental lead removal,
                        # before the broader lead-off debounce logic fully kicks in.
                        lead_disconnects = []
                        if raw_la_present:
                            self._la_report_cooldown_armed = False
                        elif not getattr(self, "_la_report_cooldown_armed", False):
                            self._la_report_cooldown_armed = True
                            lead_disconnects.append("LA")

                        if raw_ll_present:
                            self._ll_report_cooldown_armed = False
                        elif not getattr(self, "_ll_report_cooldown_armed", False):
                            self._ll_report_cooldown_armed = True
                            lead_disconnects.append("LL")

                        if lead_disconnects:
                            self._start_report_cooldown_on_lead_disconnect(lead_disconnects)

                        # Fast LL(F) flag: Lead II None means LL electrode is off.
                        # Updated every packet with no debounce so update_metrics can
                        # suppress BPM immediately ? before _lead_connection_state's
                        # 25-packet debounce window has elapsed.
                        self._ll_disconnected = (packet.get('II') is None)
                        if limb_active:
                            self._lead_off_latched = False
                            self._lead_off_latch_on_count = 0
                        elif raw_off_count >= global_threshold:
                            self._lead_off_latch_on_count = int(getattr(self, "_lead_off_latch_on_count", 0)) + 1
                        else:
                            self._lead_off_latch_on_count = 0

                        latch_on_packets = int(getattr(self, "_LEAD_OFF_LATCH_ON_PACKETS", 5))
                        latch_off_packets = int(getattr(self, "_LEAD_OFF_LATCH_OFF_PACKETS", 50))

                        _newly_latched = False
                        if not getattr(self, "_lead_off_latched", False):
                            if not limb_active and self._lead_off_latch_on_count >= latch_on_packets:
                                self._lead_off_latched = True
                                self._lead_off_latch_off_count = 0
                                _newly_latched = True
                        else:
                            if limb_active or raw_off_count < global_threshold:
                                self._lead_off_latch_off_count = int(getattr(self, "_lead_off_latch_off_count", 0)) + 1
                            else:
                                self._lead_off_latch_off_count = 0
                            if limb_active or self._lead_off_latch_off_count >= latch_off_packets:
                                self._lead_off_latched = False
                                self._lead_off_latch_on_count = 0
                                self._lead_off_latch_off_count = 0

                        # ── RA/RL electrode disconnected: alert the user and return to the
                        # outer dashboard, rather than silently letting the trace flatline. ──
                        if _newly_latched:
                            # Commit what this tick already staged BEFORE tearing the
                            # page down, so samples that arrived ahead of the
                            # disconnect are not lost with the rest of the batch.
                            self._flush_staged_samples(_staged)
                            self._handle_ra_rl_disconnect()
                            # The page/state was just torn down (acquisition stopped and we
                            # navigated away); stop processing any remaining packets from
                            # this batch against a page that's no longer active.
                            break

                        # ── Feed packet to HolterBPMController (fast, non-blocking) ───────
                        try:
                            if self._bpm_ctrl is not None and not getattr(self, "_lead_off_latched", False):
                                self._bpm_ctrl.push(packet)
                            
                            # ── Feed packet to HolterStreamWriter (fast, non-blocking) ─────
                            if self._holter_writer and self._holter_writer.is_running:
                                self._holter_writer.push(packet)
                        except Exception:
                            pass

                        for i, lead_name in enumerate(lead_order):
                            try:
                                if i < len(self.data) and lead_name in packet:
                                    # While latched, force all leads to OFF to stop flicker.
                                    value = None if getattr(self, "_lead_off_latched", False) else packet[lead_name]
                                    if not hasattr(self, '_lead_missing_counts'):
                                        self._lead_missing_counts = {}

                                    was_connected = self._lead_connection_state.get(lead_name, True)
                                    if value is None:
                                        missing = self._lead_missing_counts.get(lead_name, 0) + 1
                                        self._lead_missing_counts[lead_name] = missing
                                        # Debounce window: require at least 25 consecutive missing packets (50ms)
                                        # before treating the lead as disconnected to prevent brief signal noise recalculation.
                                        if missing >= 25:
                                            self._lead_connection_state[lead_name] = False
                                        write_value = float(self._lead_last_valid_value.get(lead_name, 0.0))
                                    else:
                                        self._lead_missing_counts[lead_name] = 0
                                        write_value = float(value)
                                        self._lead_last_valid_value[lead_name] = write_value
                                        self._lead_connection_state[lead_name] = True
                                        if not was_connected:
                                            self._on_lead_reconnected(i, lead_name, write_value)
                                    # Stage for this tick's block write (see
                                    # _append_block). Raw value; filtering happens
                                    # at display time, exactly as before.
                                    _staged[i].append(write_value)
                                    if hasattr(self, "_report_buffers") and i < len(self._report_buffers):
                                        self._report_buffers[i].append(write_value)
                                    if hasattr(self, "_replay_buffers") and i < len(self._replay_buffers):
                                        self._replay_buffers[i].append(write_value)
                            except Exception as e:
                                print(f" Error updating data buffer {i} ({lead_name}): {e}")
                                continue

                        # Top status label: show all disconnected leads (no popup).
                        if getattr(self, "_lead_off_latched", False):
                            off_leads = list(lead_order)
                        else:
                            off_leads = [ld for ld in lead_order if not self._lead_connection_state.get(ld, True)]
                        self._set_lead_status_label(off_leads)

                        # Publish a multi-lead snapshot for fusion detectors.
                        # (Key names must match what ArrhythmiaDetector expects.)
                        try:
                            if len(self.data) >= 11:
                                self.latest_multilead_snapshot = {
                                    "I": self.data[0],
                                    "aVF": self.data[5],
                                    "V1": self.data[6],
                                    "V5": self.data[10],
                                }
                        except Exception:
                            pass
                        
                        # Update sampling rate counter
                        try:
                            if hasattr(self, 'sampler'):
                                sampling_rate = self.sampler.add_sample()
                                if sampling_rate > 0:
                                    # Debug: Log detected sampling rate (first few times only)
                                    if not hasattr(self, '_sampling_rate_log_count'):
                                        self._sampling_rate_log_count = 0
                                    self._sampling_rate_log_count += 1
                                    if self._sampling_rate_log_count <= 3:
                                        import sys
                                        platform = "Windows" if sys.platform.startswith('win') else "macOS"
                                        print(f"✅ [{platform}] Hardware sampling rate detected: {sampling_rate:.1f} Hz")
                                    
                                    if hasattr(self, 'metric_labels') and 'sampling_rate' in self.metric_labels:
                                        self.metric_labels['sampling_rate'].setText(f"{sampling_rate:.1f} Hz")
                                    
                                    # Keep runtime calculations locked to configured hardware rate
                                    # (device is 500 Hz); measured UI cadence may dip when window focus changes.
                                    self.sampling_rate = 500.0
                        except Exception as e:
                            print(f" Error updating sampling rate: {e}")
                        
                        packets_processed += 1
                        
                    # One block write per lead for the whole tick.
                    self._flush_staged_samples(_staged)

                except Exception as e:
                    print(f" Error reading serial packets: {e}")
                    if hasattr(self, 'serial_reader') and hasattr(self.serial_reader, '_handle_serial_error'):
                        self.serial_reader._handle_serial_error(e)
            else:
                # OLD METHOD: Simple line-based reading (user selects COM port)
                lines_processed = 0
                max_attempts = 20
                while lines_processed < max_attempts:
                    try:
                        # Read serial data - can return 8 or 12 leads
                        all_leads = self.serial_reader.read_value()
                        if all_leads:
                            self._lead_data_received = True
                            # If we got 12 leads directly, use them; otherwise calculate from 8
                            if isinstance(all_leads, list) and len(all_leads) >= 12:
                                all_12_leads = all_leads[:12]
                            elif isinstance(all_leads, list) and len(all_leads) >= 8:
                                all_12_leads = self.calculate_12_leads_from_8_channels(all_leads)
                            elif isinstance(all_leads, (int, float)):
                                # Single value - use for lead II only
                                all_12_leads = [0] * 12
                                all_12_leads[1] = all_leads  # Lead II
                            else:
                                all_12_leads = None
                            
                            if all_12_leads is None:
                                break  # Skip if no valid data
                            
                            for i in range(len(self.leads)):
                                try:
                                    if i < len(self.data) and i < len(all_12_leads):
                                        # Same block-append rule as the packet path.
                                        # Bounded to 20 lines per tick here so the win
                                        # is small, but leaving one np.roll behind
                                        # invites the pattern back.
                                        self._append_block(self.data[i], (all_12_leads[i],))
                                        if hasattr(self, "_report_buffers") and i < len(self._report_buffers):
                                            self._report_buffers[i].append(float(all_12_leads[i]))
                                        if hasattr(self, "_replay_buffers") and i < len(self._replay_buffers):
                                            self._replay_buffers[i].append(float(all_12_leads[i]))
                                except Exception as e:
                                    print(f" Error updating data buffer {i}: {e}")
                                    continue
                            try:
                                if hasattr(self, 'sampler'):
                                    sampling_rate = self.sampler.add_sample()
                                    if sampling_rate > 0 and hasattr(self, 'metric_labels') and 'sampling_rate' in self.metric_labels:
                                        self.metric_labels['sampling_rate'].setText(f"{sampling_rate:.1f} Hz")
                            except Exception as e:
                                print(f" Error updating sampling rate: {e}")
                            lines_processed += 1
                        else:
                            break
                    except Exception as e:
                        print(f" Error reading serial data: {e}")
                        if hasattr(self, 'serial_reader') and hasattr(self.serial_reader, '_handle_serial_error'):
                            self.serial_reader._handle_serial_error(e)
                        continue
                packets_processed = lines_processed
            
            # INSTANT DISPLAY: Update plots immediately if we processed any packets OR if we have any data
            # This ensures waves appear as soon as the first packet arrives
            if not render_now:
                return
            if getattr(self, "_grid_frozen", False):
                return
            if getattr(self, "_overlay_active", False):
                # The 12:1 / 6:2 overlay hides the pyqtgraph plot area and drives
                # its own matplotlib canvas from _overlay_timer. Without this the
                # 30 ms timer kept running the whole filter / interpolate / setData
                # chain for twelve curves into widgets Qt never paints — pure waste
                # competing with a matplotlib redraw measured at 44 ms.
                #
                # This return is placed deliberately AFTER the packet drain and the
                # buffer writes above. Stopping self.timer instead would also stop
                # read_packets(), the Holter writer push and the BPM push, which all
                # live in this same callback — the overlay would silently stop
                # recording. Skip the render, never the acquisition.
                return
            has_any_data = any(len(self.data[i]) > 0 and np.any(self.data[i] != 0) for i in range(min(len(self.data), len(self.leads))))
            if packets_processed > 0 or has_any_data:
                # Detect signal source from a representative lead for adaptive scaling
                signal_source = "hardware"  # Default
                try:
                    if len(self.data) > 1 and hasattr(self, 'leads'):
                        # Prefer Lead II (index 1) if available
                        representative = self.data[1] if len(self.data[1]) > 0 else self.data[0]
                    else:
                        representative = self.data[0] if len(self.data) > 0 else []
                    signal_source = self.detect_signal_source(representative)
                except Exception as e:
                    print(f" Error detecting signal source for serial plots: {e}")
                
                # Get current wave speed for time scaling
                try:
                    wave_speed = 25.0 if getattr(self, '_overlay_active', False) else float(self.settings_manager.get_wave_speed())
                except Exception:
                    wave_speed = 25.0
                
                # Calculate time scaling based on wave speed (same logic as demo mode)
                # 25 mm/s → 3 s window; 12.5 → 6 s; 50 → 1.5 s
                baseline_seconds = 3.0
                seconds_scale = (25.0 / max(1e-6, wave_speed))
                seconds_to_show = baseline_seconds * seconds_scale
                
                for i in range(len(self.leads)):
                    try:
                        if i >= len(self.data_lines):
                            continue
                        has_data = (i < len(self.data) and len(self.data[i]) > 0)
                        # INSTANT DISPLAY: Show data immediately even with just 1 sample
                        if has_data:
                            # Calculate gain factor: higher mm/mV = higher gain (10mm/mV = 1.0x baseline)
                            gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                            scaled_data = self.apply_adaptive_gain(self.data[i], signal_source, gain_factor)

                            # Build time axis and apply wave-speed scaling
                            sampling_rate = 500.0
                            if hasattr(self, 'demo_toggle') and self.demo_toggle.isChecked():
                                if hasattr(self, 'sampler') and hasattr(self.sampler, 'sampling_rate') and self.sampler.sampling_rate > 10:
                                    sampling_rate = float(self.sampler.sampling_rate)
                                elif hasattr(self, 'sampling_rate') and self.sampling_rate > 10:
                                    sampling_rate = float(self.sampling_rate)
                            
                            # Calculate how many samples to show based on wave speed
                            # 25 mm/s → 10s window
                            # 12.5 mm/s → 20s window (show more data, compressed)
                            # 50 mm/s → 5s window (show less data, stretched)
                            samples_to_show = int(sampling_rate * seconds_to_show)
                            
                            # Take only the most recent samples_to_show from the buffer (before gain application)
                            # INSTANT DISPLAY: Use all available data immediately, even if less than samples_to_show
                            raw_data = self.data[i]
                            
                            # INSTANT DISPLAY: Use recent data immediately - don't filter zeros (they might be valid signal)
                            # Find the last non-zero sample to determine how much real data we have
                            non_zero_indices = np.where(raw_data != 0)[0]
                            if len(non_zero_indices) > 0:
                                # We have real data - use from the first non-zero to the end
                                first_real_idx = non_zero_indices[0]
                                recent_data = raw_data[first_real_idx:]
                            else:
                                # All zeros - use recent samples anyway for instant display
                                recent_data = raw_data[-min(100, len(raw_data)):]
                            
                            # Handle Display Mode: Scroll vs Sweep
                            display_mode = getattr(self, 'display_mode', 'scroll')
                            
                            if display_mode == 'sweep' and len(raw_data) >= samples_to_show:
                                # Sweep Mode: use an internal capped cursor so app focus changes
                                # or repeated background work do not deform/jump the live trace.
                                window_len = samples_to_show
                                if not hasattr(self, '_sweep_cursor_state'):
                                    self._sweep_cursor_state = {}
                                sweep_key = ('live', i)
                                now_mono = time.monotonic()
                                sweep_state = self._sweep_cursor_state.get(sweep_key, {})
                                prev_cursor = int(sweep_state.get('cursor', window_len - 1))
                                prev_ts = float(sweep_state.get('ts', now_mono))
                                raw_step = max(1, int(round((now_mono - prev_ts) * sampling_rate)))
                                step = min(raw_step, max(1, int(0.08 * sampling_rate)))
                                cursor = (prev_cursor + step) % window_len
                                self._sweep_cursor_state[sweep_key] = {
                                    'cursor': cursor,
                                    'ts': now_mono,
                                    'window_len': window_len,
                                }
                                
                                raw_window = raw_data[-window_len:]
                                # Shift to align newest data (at end of raw_window) to cursor
                                src = np.roll(raw_window, cursor - (window_len - 1))
                                
                                # Insert gap
                                gap_size = int(0.05 * sampling_rate)
                                gap_end = min(len(src), cursor + gap_size)
                                src[cursor:gap_end] = np.nan
                                if cursor + gap_size > len(src):
                                    src[:cursor + gap_size - len(src)] = np.nan
                                
                                data_slice = src
                            else:
                                if len(recent_data) > samples_to_show:
                                    data_slice = recent_data[-samples_to_show:]
                                else:
                                    data_slice = recent_data
                            
                            # INSTANT DISPLAY: Always ensure we have data to display (even if just a few samples)
                            if len(data_slice) == 0:
                                # Fallback: use the most recent samples from buffer
                                data_slice = raw_data[-min(50, len(raw_data)):]
                            
                            # 🫀 DISPLAY: Low-frequency baseline anchor (removes respiration from baseline)
                            # Extract very-low-frequency baseline (< 0.3 Hz) to prevent baseline from "breathing"
                            filtered_slice = np.array(data_slice, dtype=float)
                            try:
                                # Initialize slow anchor if needed
                                if not hasattr(self, '_baseline_anchors'):
                                    self._baseline_anchors = [0.0] * 12
                                    self._baseline_alpha_slow = 0.0005  # Monitor-grade: ~4 sec time constant at 500 Hz
                                
                                if len(filtered_slice) > 0:
                                    # Respiration baseline drift stop (display-only):
                                    # The 12-box view previously never applied the DFT high-pass, so selecting
                                    # 0.5 Hz removed drift in reports but not in the live grid. Apply it only
                                    # when slow drift dominates this window.
                                    try:
                                        from ecg.ecg_filters import apply_dft_filter, apply_baseline_wander_median_mean
                                        if hasattr(self, "_last_display_dft_applied") and i < len(self._last_display_dft_applied):
                                            self._last_display_dft_applied[i] = False
                                        dft_setting = None
                                        if hasattr(self, "settings_manager"):
                                            dft_setting = str(self.settings_manager.get_setting("filter_dft", "off")).strip()
                                        if dft_setting and dft_setting not in ("off", ""):
                                            if str(dft_setting).strip() == "0.5":
                                                filtered_slice = apply_baseline_wander_median_mean(filtered_slice, float(sampling_rate))
                                            else:
                                                filtered_slice = apply_dft_filter(filtered_slice, float(sampling_rate), dft_setting)
                                            if hasattr(self, "_last_display_dft_applied") and i < len(self._last_display_dft_applied):
                                                self._last_display_dft_applied[i] = True
                                    except Exception:
                                        pass

                                    # Unified Baseline Correction (Steady State Immediately)
                                    # User requested "steady state" behavior from t=0 to avoid wave distortion.
                                    # We use the slow alpha (monitor-grade) immediately.
                                    # The initial "snap" (anchor=0 check) handles the initial DC offset.
                                    
                                    current_alpha = self._baseline_alpha_slow

                                    # Extract baseline estimate (helper handles short signals via fallback to mean)
                                    baseline_estimate = self._extract_low_frequency_baseline(filtered_slice, sampling_rate)
                                    
                                    # Initialize anchor immediately if zero (Snap to center on first frame)
                                    if self._baseline_anchors[i] == 0.0:
                                        self._baseline_anchors[i] = baseline_estimate
                                    
                                    # Update anchor with slow alpha (steady state tracking)
                                    self._baseline_anchors[i] = (1 - current_alpha) * self._baseline_anchors[i] + current_alpha * baseline_estimate
                                    
                                    # Subtract anchor (NOT raw mean)
                                    filtered_slice = filtered_slice - self._baseline_anchors[i]
                                    
                                    # Final zero-centering to ensure perfect centering before gain
                                    current_dc = np.nanmean(filtered_slice) if len(filtered_slice) > 0 else 0.0
                                    filtered_slice = filtered_slice - current_dc
                            except Exception as filter_error:
                                # Fallback: use original signal (baseline anchor handles it, no mean subtraction)
                                pass  # Silently handle errors to avoid console spam
                            
                            # ---------------- DISPLAY FILTERS (match 12:1 / 6:2 overlays) ----------------
                            # Apply EMG first, then AC (unless EMG cutoff is so low it would suppress AC),
                            # so "muscular" filtering behaves the same in the 12‑box grid and overlays.
                            emg_applied = False
                            emg_suppresses_ac = False
                            try:
                                from ecg.ecg_filters import apply_emg_filter
                                emg_setting = None
                                if hasattr(self, "settings_manager"):
                                    emg_setting = str(self.settings_manager.get_setting("filter_emg", "150")).strip()
                                if emg_setting and emg_setting.lower() != "off" and len(filtered_slice) >= 10:
                                    filtered_slice = apply_emg_filter(filtered_slice, float(sampling_rate), emg_setting)
                                    emg_applied = True
                                    try:
                                        if float(emg_setting) < 60:
                                            emg_suppresses_ac = True
                                    except Exception:
                                        pass
                            except Exception:
                                # Keep display running even if EMG filter fails
                                pass

                            # Optional AC notch filtering based on "Set Filter" selection.
                            # Keeps wave peaks intact while removing 50/60 Hz power line noise.
                            try:
                                from ecg.ecg_filters import apply_ac_filter
                                ac_setting = None
                                if hasattr(self, "settings_manager"):
                                    ac_setting = str(self.settings_manager.get_setting("filter_ac", "50")).strip()
                                if ac_setting in ("50", "60") and len(filtered_slice) > 30:
                                    filtered_slice = apply_ac_filter(filtered_slice, float(sampling_rate), ac_setting)
                            except Exception:
                                # If anything fails, fall back to unfiltered slice so UI never crashes
                                pass
                            
                            # Apply wave gain to zero-centered signal (only amplifies variations, baseline stays at zero)
                            gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                            scaled_data = filtered_slice * gain_factor
                            scaled_data = np.nan_to_num(scaled_data, copy=False)

                            # --- Flatline detection (serial/display path) ---
                            if scaled_data.size >= 50:
                                amp_range = float(np.nanmax(scaled_data) - np.nanmin(scaled_data))
                                std_val = float(np.nanstd(scaled_data))
                                # Very small range and std → likely flatline / disconnected lead
                                is_flat = amp_range < 5.0 and std_val < 1.0
                                lead_name = self.leads[i] if i < len(self.leads) else f"Lead {i+1}"
                                if is_flat and not self._flatline_alert_shown[i]:
                                    self._flatline_alert_shown[i] = True
                                elif not is_flat:
                                    # Reset flag when signal returns
                                    self._flatline_alert_shown[i] = False
                            
                            # --- PIPELINE STEP 2: Gaussian Smoothing (Standalone) ---
                            try:
                                if len(scaled_data) > 5:
                                    scaled_data = gaussian_filter1d(scaled_data, sigma=self.SMOOTH_SIGMA)
                            except Exception:
                                pass

                            # --- PIPELINE STEP 3: Interpolation (Standalone) ---
                            try:
                                scaled_data = self.interpolate(scaled_data, self.INTERP_FACTOR)
                            except Exception:
                                pass

                            # Trim small margins from both ends to reduce filter edge artefacts
                            try:
                                edge_trim = int(0.5 * self.SAMPLE_RATE)  # ~200ms on each side at 500 Hz
                                if len(scaled_data) > 2 * edge_trim:
                                    scaled_data = scaled_data[edge_trim:-edge_trim]
                            except Exception:
                                pass

                            # INSTANT DISPLAY: Always create time axis and plot, even with just 1 sample
                            n = len(scaled_data)
                            if n == 0:
                                # Fallback: create minimal data for instant display
                                scaled_data = np.array([0.0])
                                n = 1
                            
                            time_axis = np.arange(n, dtype=float) / sampling_rate
                            
                            # Signal is already zero-centred by the baseline subtraction above.
                            # Y-axis is symmetric (-2048 to +2048) for ALL leads — no offset needed.
                            # aVR will naturally sit below baseline due to its formula (-(I+II)/2).
                            
                            # INSTANT DISPLAY: Update plot immediately, even with minimal data
                            # Avoid cropping: small padding and explicit x-range
                            try:
                                vb = self.plot_widgets[i].getViewBox()
                                if vb is not None:
                                    if len(time_axis) > 1:
                                        vb.setRange(xRange=(time_axis[0], time_axis[-1]), padding=0)
                                    else:
                                        # For single sample, set a small range for instant display
                                        vb.setRange(xRange=(0, max(0.1, 1.0 / sampling_rate)), padding=0)
                            except Exception:
                                pass

                            # INSTANT DISPLAY: Always update the plot, even with minimal data
                            self.data_lines[i].setData(time_axis, scaled_data)
                            
                            # Optimized: Update Y-range only every 10 updates (reduces expensive calculations)
                            if not hasattr(self, '_y_range_update_count'):
                                self._y_range_update_count = {}
                            if i not in self._y_range_update_count:
                                self._y_range_update_count[i] = 0
                            self._y_range_update_count[i] += 1
                            if self._y_range_update_count[i] % 20 == 0:  # Update Y-range every 20 plot updates (reduced from 10 for better performance)
                                self.update_plot_y_range_adaptive(i, signal_source, data_override=scaled_data)

                            # Removed debug print for better performance
                        else:
                            self.data_lines[i].setData(self.data[i] if i < len(self.data) else [])
                            self.update_plot_y_range(i)
                    except Exception as e:
                        print(f" Error updating plot {i}: {e}")
                        continue
                pass
                # Removed heartbeat debug print for better performance

        except Exception as e:
            self.crash_logger.log_crash("Critical error in update_plots", e, "Real-time ECG plotting")
            try:
                if hasattr(self, 'data') and self.data:
                    for i in range(len(self.data)):
                        self.data[i] = np.zeros(self.buffer_size if hasattr(self, 'buffer_size') else 1000)
            except Exception as recovery_error:
                self.crash_logger.log_error("Failed to recover from update_plots error", recovery_error, "Data reset")
        finally:
            self._plot_update_in_progress = False
    
    def _manage_memory(self):
        """Manage memory usage to prevent crashes from large data buffers"""
        try:
            import gc
            import psutil
            import os
            
            # Check current memory usage
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > 500:  # If using more than 500MB
                print(f" High memory usage: {memory_mb:.1f}MB - cleaning up...")
                
                # Force garbage collection
                gc.collect()
                
                # Trim data buffers if they're too large
                for i, data_buffer in enumerate(self.data):
                    if len(data_buffer) > self.max_buffer_size:
                        # Keep only the most recent data
                        self.data[i] = data_buffer[-self.max_buffer_size:].copy()
                        print(f" Trimmed data buffer {i} to {len(self.data[i])} samples")
                
                # Check memory after cleanup
                memory_after = process.memory_info().rss / 1024 / 1024
                print(f" Memory after cleanup: {memory_after:.1f}MB (freed {memory_mb - memory_after:.1f}MB)")
                
        except ImportError:
            # psutil not available, skip memory management
            pass
        except Exception as e:
            print(f" Error in memory management: {e}")
    
    def _log_error(self, error_msg, exception=None, context=""):
        """Comprehensive error logging for debugging crashes"""
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Basic error info
            log_msg = f"[{timestamp}] {error_msg}"
            if context:
                log_msg += f" | Context: {context}"
            
            print(log_msg)
            
            # Detailed exception info
            if exception:
                print(f"Exception Type: {type(exception).__name__}")
                print(f"Exception Message: {str(exception)}")
                print("Full Traceback:")
                traceback.print_exc()
            
            # System state info
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                print(f"System State - Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%")
            except ImportError:
                pass
            
            # ECG state info
            try:
                if hasattr(self, 'data') and self.data:
                    data_lengths = [len(d) for d in self.data]
                    print(f"ECG Data State - Buffer lengths: {data_lengths}")
                
                if hasattr(self, 'serial_reader') and self.serial_reader:
                    print(f"Serial Reader State - Running: {self.serial_reader.running}")
                    print(f"Serial Reader State - Data count: {self.serial_reader.data_count}")
            except Exception:
                pass
                
        except Exception as log_error:
            print(f" Error in error logging: {log_error}")
    
    def closeEvent(self, event):
        """Clean up all resources when the ECG test page is closed"""
        try:
            # Stop demo manager
            if hasattr(self, 'demo_manager'):
                self.demo_manager.stop_demo_data()
            
            # Stop timers
            if hasattr(self, 'timer') and self.timer:
                self.timer.stop()
                self.timer.deleteLater()
            
            if hasattr(self, 'metrics_timer') and self.metrics_timer:
                self.metrics_timer.stop()
                self.metrics_timer.deleteLater()
            
            if hasattr(self, 'elapsed_timer') and self.elapsed_timer:
                self.elapsed_timer.stop()
                self.elapsed_timer.deleteLater()

            # Stop live BPM worker before Qt tears down the page.
            if hasattr(self, '_bpm_ctrl') and self._bpm_ctrl is not None:
                try:
                    if self._bpm_ctrl.is_running:
                        self._bpm_ctrl.stop()
                    if getattr(self._bpm_ctrl, 'display_bar', None) is not None:
                        self._bpm_ctrl.display_bar.cleanup()
                except Exception:
                    pass
                self._bpm_ctrl = None

            # Stop any active Holter recording / analysis threads.
            if hasattr(self, '_holter_writer') and self._holter_writer is not None:
                try:
                    if self._holter_writer.is_running:
                        self._holter_writer.stop()
                except Exception:
                    pass
                self._holter_writer = None

            if hasattr(self, '_holter_worker') and self._holter_worker is not None:
                try:
                    self._holter_worker.stop(wait=True)
                except Exception:
                    pass
                self._holter_worker = None

            # Wait for any worker threads to finish.
            self._cleanup_start_worker(wait_ms=5000, force_terminate=True)
            self._cleanup_report_generation_thread(wait_ms=5000, force_terminate=True)
            
            # Close serial connection
            if hasattr(self, 'serial_reader') and self.serial_reader:
                try:
                    self.serial_reader.close()
                except Exception:
                    pass
            
            # Log cleanup
            if hasattr(self, 'crash_logger'):
                self.crash_logger.log_info("ECG Test Page closed, resources cleaned up", "ECG_TEST_PAGE_CLOSE")
        except Exception as e:
            print(f"Error during ECGTestPage cleanup: {e}")
        finally:
            super().closeEvent(event)
