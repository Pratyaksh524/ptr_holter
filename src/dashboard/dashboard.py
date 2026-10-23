"""
CardioX Main Dashboard — src/dashboard/dashboard.py
===================================================
PURPOSE & ARCHITECTURE:
This module implements the primary Dashboard UI (DashboardWindow) presented to clinicians/operators upon login.
It acts as the central command hub:
1. Patient Data & Exam Navigation: Quick patient setup, history viewing, and direct launch into 12-lead ECG acquisition.
2. Real-Time Rhythm Panel: Receives periodic rhythm interpretation cache updates (last_conclusions.json) to display live diagnostic summaries.
3. System Diagnostics & Controls: Device status, battery levels, cloud sync status indicator, and admin report access.

DEVELOPER NOTES & RECENT REFACTORS:
- Comprehensive ECG Button (self.holter_btn): Disabled and hidden per product specification to prevent incomplete workflow launches from the main hub.
- Live Rhythm Refresh: Listens for rhythm updates every ~1s (update_live_conclusion) to mirror the active 12-lead signal metrics.
"""

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFrame, QGridLayout, QCalendarWidget, QTextEdit,
    QDialog, QLineEdit, QComboBox, QFormLayout, QMessageBox, QSizePolicy, QStackedWidget, QScrollArea, QSpacerItem, QSlider,
    QRadioButton, QButtonGroup, QGraphicsDropShadowEffect, QToolButton, QShortcut
)
from PyQt5.QtGui import QFont, QPixmap, QMovie, QPainter, QColor, QPen, QImage, QIntValidator, QIcon, QDesktopServices, QKeySequence
from PyQt5.QtCore import Qt, QTimer, QSize, QThread, pyqtSignal, QDate, QUrl
try:
    from PyQt5.QtMultimedia import QSound
except ImportError:
    print(" QSound not available - heartbeat sound will be disabled")
    QSound = None
import sys
import platform
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from ecg.serial.serial_reader import SerialStreamReader, SERIAL_AVAILABLE
import serial.tools.list_ports
if SERIAL_AVAILABLE:
    from ecg.serial.hardware_commands import HardwareCommandHandler
    import serial
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import math
import os
from utils.app_paths import data_file
import json
import matplotlib.image as mpimg
import time
import datetime
from dashboard.chatbot_dialog import ChatbotDialog
from utils.settings_manager import SettingsManager
from utils.localization import translate_text
from utils.crash_logger import get_crash_logger, CrashLogDialog
from utils.patient_profile import resolve_patient_profile
from dashboard.admin_reports import AdminLoginDialog, AdminReportsDialog
from ecg.signal.signal_processing import extract_low_frequency_baseline
from ecg.utils.helpers import get_display_gain
from utils.platform_compat import is_low_spec_mode

# Try to import configuration, fallback to defaults if not available
try:
    import sys
    # Add the src directory to the path
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    try:
        from config.settings import get_config
        config = get_config()
        def get_background_config():
            return config.get('ui.background', {"background": "none", "gif": False})
        print(" Dashboard configuration loaded successfully")
    except ImportError as e:
        print(f" Dashboard config import warning: {e}")
        def get_background_config():
            return {"background": "none", "gif": False}
except ImportError:
    print(" Dashboard configuration not found, using default settings")
    def get_background_config():
        return {
            "use_gif_background": False,
            "preferred_background": "none"
        }

def get_asset_path(asset_name):
    """
    Get the absolute path to an asset file in a portable way.
    This function will work regardless of where the script is run from.
    
    Args:
        asset_name (str): Name of the asset file (e.g., 'her.png', 'v.gif')
    
    Returns:
        str: Absolute path to the asset file
    """
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = []
    if getattr(sys, "frozen", False):
        bundle_dir = getattr(sys, "_MEIPASS", "")
        exe_dir = os.path.dirname(sys.executable)
        possible_paths.extend([
            os.path.join(bundle_dir, "assets"),
            os.path.join(exe_dir, "assets"),
        ])
    possible_paths.extend([
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), "assets"),
        os.path.join(script_dir, "assets"),
        os.path.join(os.path.dirname(script_dir), "assets"),
        os.path.join(script_dir, "..", "assets"),
    ])
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return os.path.join(path, asset_name)
    return os.path.join(os.path.dirname(script_dir), "..", "assets", asset_name)

class MplCanvas(FigureCanvas):
    def __init__(self, width=4, height=2, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class DeviceScanWorker(QThread):
    """Worker thread for non-blocking serial port scanning"""
    scan_finished = pyqtSignal(bool, str, str, str) # success, port, version, serial

    def __init__(self, settings_manager=None):
        super().__init__()
        self.settings_manager = settings_manager

    def run(self):
        try:
            import serial
            import serial.tools.list_ports
            from ecg.serial.hardware_commands import HardwareCommandHandler

            ports = list(serial.tools.list_ports.comports())
            if sys.platform == "darwin":
                ports = [p for p in ports if ("usbserial" in p.device) or ("usbmodem" in p.device)]
            else:
                # Avoid probing non-USB / legacy ports that frequently hang or always fail.
                filtered = []
                for p in ports:
                    desc = str(getattr(p, "description", "") or "")
                    dev = str(getattr(p, "device", "") or "")
                    if dev.upper() == "COM1" and "Communications Port" in desc:
                        continue
                    if "Bluetooth" in desc:
                        continue
                    filtered.append(p)
                ports = filtered
            
            if not ports:
                self.scan_finished.emit(False, "", "", "")
                return

            # Prioritize the last saved port
            if self.settings_manager:
                saved_port = self.settings_manager.get_setting("serial_port")
                if saved_port:
                    ports.sort(key=lambda p: 0 if p.device == saved_port else 1)

            for port in ports:
                try:
                    desc = str(getattr(port, "description", "") or "")
                    print(f" Device scan: probing {port.device} ({desc})")

                    # Quick check (keep it short; this runs in a background thread).
                    ser = serial.Serial(
                        port.device,
                        115200,
                        timeout=0.2,
                        write_timeout=0.2,
                    )
                    try:
                        handler = HardwareCommandHandler(ser)
                        # 1. Preferred detection: VERSION command
                        success_v, version, _ = handler.send_version_command(timeout=0.4)
                        
                        # 2. Also try to get MACHINE SERIAL while we have the port open
                        serial_num = ""
                        if success_v:
                            success_s, serial_num, _ = handler.send_machine_serial_command(timeout=0.4)
                            if not success_s:
                                serial_num = ""
                        
                        if success_v and version:
                            self.scan_finished.emit(True, port.device, version, serial_num)
                            return
                    finally:
                        try:
                            ser.close()
                        except Exception:
                            pass
                except Exception as e:
                    print(f"⚠️ Device scan: {port.device} probe failed: {e}")
                    continue

            self.scan_finished.emit(False, "", "", "")
        except Exception as e:
            print(f"Error in DeviceScanWorker: {e}")
            self.scan_finished.emit(False, "", "", "")
class SignInDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sign In")
        self.setMinimumSize(380, 280)
        self.setStyleSheet("""
            QDialog { background: #fff; border-radius: 18px; }
            QLabel { font-size: 15px; color: #222; }
            QLineEdit, QComboBox { border: 2px solid #ff6600; border-radius: 8px; padding: 6px 10px; font-size: 15px; background: #f7f7f7; }
            QPushButton { background: #ff6600; color: white; border-radius: 10px; padding: 8px 0; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background: #ff8800; }
        """)
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        layout.setContentsMargins(28, 24, 28, 24)
        title = QLabel("Sign In to PulseMonitor")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.role_combo = QComboBox()
        self.role_combo.addItems(["Doctor", "Patient"])
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Enter your name")
        self.pass_edit = QLineEdit()
        self.pass_edit.setPlaceholderText("Password")
        self.pass_edit.setEchoMode(QLineEdit.Password)
        form.addRow("Role:", self.role_combo)
        form.addRow("Name:", self.name_edit)
        form.addRow("Password:", self.pass_edit)
        layout.addLayout(form)
        self.signin_btn = QPushButton("Sign In")
        self.signin_btn.clicked.connect(self.accept)
        layout.addWidget(self.signin_btn)
    def get_user_info(self):
        return self.role_combo.currentText(), self.name_edit.text()

class StyledMessageBox(QDialog):
    @staticmethod
    def show_message(parent, title, message, is_critical=False, auto_close_ms=None):
        dialog = QDialog(parent)
        dialog.setWindowTitle(title)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setMinimumSize(450, 220)
        dialog.setStyleSheet("""
            QDialog { 
                background: #111827; 
                border-radius: 12px;
                border: 1px solid #374151;
            }
            QLabel { 
                font-size: 14px; 
                color: #D1D5DB; 
                font-family: 'Segoe UI', Arial;
            }
            QPushButton { 
                background: #3B82F6; 
                color: white; 
                border-radius: 8px; 
                padding: 8px 24px; 
                font-size: 14px; 
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { background: #2563EB; }
        """)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(18)
        layout.setContentsMargins(28, 24, 28, 24)
        
        title_lbl = QLabel(title)
        title_lbl.setFont(QFont("Arial", 18, QFont.Bold))
        title_lbl.setAlignment(Qt.AlignCenter)
        if is_critical:
            title_lbl.setStyleSheet("color: #e74c3c; font-family: 'Segoe UI', Arial;")
        else:
            title_lbl.setStyleSheet("color: #00E676; font-family: 'Segoe UI', Arial;")
        layout.addWidget(title_lbl)
        
        msg_lbl = QLabel(message)
        msg_lbl.setWordWrap(True)
        msg_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(msg_lbl)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(100)
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        # Auto-close after the given number of milliseconds (same behaviour as
        # the "Device connected" notification in on_scan_finished).
        if auto_close_ms is not None:
            from PyQt5.QtCore import QTimer as _QT
            _QT.singleShot(int(auto_close_ms), dialog.accept)
        
        dialog.exec_()
    
class DashboardHomeWidget(QWidget):
    def __init__(self):
        super().__init__()

class Dashboard(QWidget):
    internet_status_changed = pyqtSignal(bool, str)

    def __init__(self, username=None, role=None, user_details=None, parent=None):
        super().__init__(parent)
        # Settings for wave speed/gain
        self.settings_manager = SettingsManager()
        self.current_language = self.settings_manager.get_setting("system_language", "en")
        audio_cfg = {}
        try:
            audio_cfg = get_config().get_audio_config()
        except Exception:
            audio_cfg = {}
        cfg_enabled = str(audio_cfg.get("heartbeat_enabled", "on")).lower() in ("1", "true", "yes", "on")
        settings_enabled = str(self.settings_manager.get_setting("system_beat_vol", "on")).lower() in ("1", "true", "yes", "on")
        if not settings_enabled:
            try:
                self.settings_manager.set_setting("system_beat_vol", "on")
                settings_enabled = True
            except Exception:
                pass
        self.heartbeat_sound_enabled = bool(cfg_enabled and settings_enabled)
        self._had_device_connected = False
        
        # Initialize crash logger
        self.crash_logger = get_crash_logger()
        self.crash_logger.log_info("Dashboard initialized", "DASHBOARD_START")
        
        # ========================================
        # START AUTOMATIC BACKGROUND CLOUD SYNC
        # ========================================
        # This service runs in background and syncs every 5 seconds
        try:
            from utils.auto_sync_service import start_auto_sync
            self.auto_sync_service = start_auto_sync(interval_seconds=15)
            print(" Automatic cloud sync started (every 15 seconds)")
        except Exception as e:
            print(f" Could not start auto-sync service: {e}")
            self.auto_sync_service = None
        
        # ========================================
        # INITIALIZE OFFLINE QUEUE (deferred to background)
        # ========================================
        # FIX: OfflineQueue.__init__ calls is_online() → socket timeout 3s
        # Defer to background thread so login → dashboard is instant
        self.offline_queue = None

        def _init_queue_bg():
            try:
                from utils.offline_queue import get_offline_queue
                q = get_offline_queue()
                self.offline_queue = q
                stats = q.get_stats()
                if stats.get('pending_count', 0) > 0:
                    q.force_sync_now()  # already runs in background thread
            except Exception as e:
                print(f"Offline queue init error: {e}")

        import threading
        threading.Thread(target=_init_queue_bg, daemon=True,
                         name="OfflineQueueInit").start()

        # Prevent overlapping internet checks (timer fires every 3s)
        self._inet_check_lock = threading.Lock()
         
        # Triple-click counter for heart rate metric
        self.heart_rate_click_count = 0
        self.last_heart_rate_click_time = 0
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)  # Minimum size for usability
        
        # Reports filter date
        self.reports_filter_date = None
        
        # Store username, role, and full user details
        self.username = username
        self.role = role
        self.user_details = user_details or {}

        # Flags to track which test is currently running
        self.test_states = {
            "12_lead_test": False,
            "hrv_test": False,
            "hyperkalemia_test": False
        }
        self.closed_by_sign_out = False
        
        # Initialize standard values flag
        self._use_standard_values = True
        print("✅ Standard values flag initialized")
        
        # Initialize BPM tracking for instant updates
        self._last_bpm = None
        self._bpm_change_threshold = 2  # Update if BPM changes by 2 or more
        print("🔄 BPM change detection initialized")
        
        # Initialize theme modes
        self.dark_mode = False
        self.medical_mode = False
        print("🎨 Theme modes initialized")
        
        # Initialize stable RR tracking
        self._last_stable_rr = None
        self._rr_stability_counter = 0
        print("🔒 Stable RR tracking initialized")
        
        self.setWindowTitle("CardioX Dashboard")
        try:
            import os
            from config.settings import resource_path
            from PyQt5.QtGui import QIcon
            icon_path = resource_path("assets/cardiox_logo.ico")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass
        self.setGeometry(100, 100, 1300, 900)
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        self.setWindowState(Qt.WindowMaximized)
        self.center_on_screen()
        
        # Test asset paths at startup for debugging
        self.test_asset_paths()
        
        # Load background settings from configuration file
        config = get_background_config()
        self.use_gif_background = config.get("use_gif_background", False)
        self.preferred_background = config.get("preferred_background", "none")
        
        print(f"Dashboard background: {self.preferred_background} (GIF: {self.use_gif_background})")
        
        # --- Plasma GIF background ---
        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        self.bg_label.lower()
        
        # Try to load background GIFs using portable paths
        if not self.use_gif_background:
            # Use light gray background matching ECG 12 test page
            self.bg_label.setStyleSheet("background: #f8f9fa;")
            print("Using light gray background (matching ECG page)")
        else:
            # Priority order based on user preference
            movie = None
            if self.preferred_background == "plasma.gif":
                plasma_path = get_asset_path("plasma.gif")
                if os.path.exists(plasma_path):
                    movie = QMovie(plasma_path)
                    print("Using plasma.gif as background")
                else:
                    print("plasma.gif not found, trying alternatives...")
                    self.preferred_background = "tenor.gif"  # Fallback
            
            if self.preferred_background == "tenor.gif" and not movie:
                tenor_gif_path = get_asset_path("tenor.gif")
                if os.path.exists(tenor_gif_path):
                    movie = QMovie(tenor_gif_path)
                    print("Using tenor.gif as background")
                else:
                    print("tenor.gif not found, trying alternatives...")
                    self.preferred_background = "v.gif"  # Fallback
            
            if self.preferred_background == "v.gif" and not movie:
                v_gif_path = get_asset_path("v.gif")
                if os.path.exists(v_gif_path):
                    movie = QMovie(v_gif_path)
                    print("Using v.gif as background")
                else:
                    print("v.gif not found, using solid color background")
            
            if movie:
                self.bg_label.setMovie(movie)
                movie.start()
            else:
                # If no GIF found, use light gray background matching ECG page
                self.bg_label.setStyleSheet("background: #f8f9fa;")
                print("Using light gray background (no GIFs found)")
        # --- Central stacked widget for in-place navigation ---
        self.page_stack = QStackedWidget(self)
        
        # --- Dashboard main page widget ---
        self.dashboard_page = DashboardHomeWidget()
        # Set light gray background for main content area (matching ECG page)
        self.dashboard_page.setStyleSheet("background-color: #f8f9fa;")
        dashboard_layout = QVBoxLayout(self.dashboard_page)
        dashboard_layout.setSpacing(20)
        dashboard_layout.setContentsMargins(20, 20, 20, 20)
        
        # --- Header ---
        header = QHBoxLayout()
        logo = QLabel("CardioX Dashboard")
        logo.setFont(QFont("Arial", 24, QFont.Bold))
        logo.setStyleSheet("color: #ff6600;")
        logo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        header.addWidget(logo)
        
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(26, 26)
        self.status_dot.setAlignment(Qt.AlignCenter)
        self.status_dot.setScaledContents(True)
        self.status_dot.setStyleSheet("background: transparent;")
        self.status_dot.setToolTip("Checking connection...")
        header.addWidget(self.status_dot)

        # Wi‑Fi / Internet indicator icons (falls back to dot if icons missing)
        # Support both naming styles: `wi-fi-*.png` and `wifi_*.png`
        self._wifi_pixmap_connected = self._load_status_pixmap_any(
            ["wi-fi-connected.png", "wifi_connected.png"],
            target_px=22,
        )
        self._wifi_pixmap_disconnected = self._load_status_pixmap_any(
            ["wi-fi-disconnected.png", "wifi_disconnected.png"],
            target_px=22,
        )
        if self._wifi_pixmap_disconnected.isNull() and not self._wifi_pixmap_connected.isNull():
            self._wifi_pixmap_disconnected = self._make_wifi_disconnected_pixmap(self._wifi_pixmap_connected, target_px=22)
        if not self._wifi_pixmap_disconnected.isNull():
            # Default to "disconnected" until first check runs (no gray dot)
            self.status_dot.setPixmap(self._wifi_pixmap_disconnected)

        # Thread-safe UI update for internet status checks
        self.internet_status_changed.connect(self._apply_internet_status)
         
        # Kick the first check shortly after the event loop starts. Doing it immediately at startup
        # can occasionally give false negatives on some machines during network initialization.
        QTimer.singleShot(250, self.update_internet_status)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_internet_status)
        self.status_timer.start(3000)

        # Device Status Label
        self.device_status_label = QLabel("RhythmUltra Not Connected")
        self.device_status_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.device_status_label.setStyleSheet("color: red; margin-right: 10px;")
        self.device_status_label.setMinimumWidth(160)
        self.device_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header.addWidget(self.device_status_label)

        self.license_status_label = QLabel("● Software Activated")
        self.license_status_label.setFont(QFont("Arial", 9, QFont.Bold))
        self.license_status_label.setMinimumWidth(160)
        self.license_status_label.setMaximumHeight(28)
        self.license_status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.license_status_label.setStyleSheet("""
            QLabel {
                color: #27ae60;
                background: transparent;
                padding: 2px 8px 2px 8px;
                font-size: 9pt;
                font-weight: 700;
                letter-spacing: 0.3px;
            }
        """)
        # label will be placed in footer, not header
        
        self.medical_btn = QPushButton("Medical Mode")
        self.medical_btn.setCheckable(True)
        self.medical_btn.setStyleSheet("background: #00b894; color: white; border-radius: 10px; padding: 4px 18px;")
        self.medical_btn.clicked.connect(self.toggle_medical_mode)
        self.medical_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Hide button per request while preserving logic
        self.medical_btn.setVisible(False)
        
        self.dark_btn = QPushButton("Dark Mode")
        self.dark_btn.setCheckable(True)
        self.dark_btn.setStyleSheet("background: #222; color: #fff; border-radius: 10px; padding: 4px 18px;")
        self.dark_btn.clicked.connect(self.toggle_dark_mode)
        self.dark_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Hide dark mode button
        self.dark_btn.setVisible(False)
        
        # Removed background control button per request
        
        header.addStretch()
        
        # Cloud Sync Button - styled to match orange UI buttons
        self.cloud_sync_btn = QPushButton("Cloud Sync")
        self.cloud_sync_btn.setStyleSheet("""
            QPushButton {
                background: #ff6600;
                color: white;
                border-radius: 16px;
                padding: 8px 24px;
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #ff7a26;
                min-width: 140px;
            }
            QPushButton:hover { background: #ff7a26; border: 2px solid #ff8e47; }
            QPushButton:pressed { background: #e65c00; }
        """)
        self.cloud_sync_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.cloud_sync_btn.setToolTip("Upload ECG reports and metrics to AWS S3")
        self.cloud_sync_btn.clicked.connect(self.sync_to_cloud)
        header.addWidget(self.cloud_sync_btn)
        # Fully automatic mode: hide manual sync button
        self.cloud_sync_btn.setVisible(False)
        
        # User label removed per request
        # self.user_label = QLabel(f"{username or 'User'}\n{role or ''}")
        # self.user_label.setFont(QFont("Arial", 12))
        # self.user_label.setAlignment(Qt.AlignRight)
        # self.user_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # header.addWidget(self.user_label)
        
        # Admin button (disabled per request; keep logic available)
        self.admin_btn = QPushButton("Admin")
        self.admin_btn.setVisible(False)

        # Patient registration moved from ECG menu ("Save ECG") to outer dashboard header.
        self.doctor_profile_btn = QPushButton("Doctor Profile")
        self.doctor_profile_btn.setStyleSheet(
            "background: #2a2a2a; color: white; border-radius: 10px; padding: 4px 12px; margin-right: 8px;"
        )
        self.doctor_profile_btn.setToolTip("Edit doctor profile (name, organisation, password)")
        self.doctor_profile_btn.clicked.connect(self.show_doctor_profile_dialog)
        self.doctor_profile_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header.addWidget(self.doctor_profile_btn)

        self.new_registration_btn = QPushButton("Patient registration")
        self.new_registration_btn.setStyleSheet(
            "background: #E8650A; color: white; border-radius: 10px; padding: 4px 18px; margin-right: 10px;"
        )
        self.new_registration_btn.clicked.connect(self.show_new_registration_dialog)
        self.new_registration_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header.addWidget(self.new_registration_btn)

        self.support_btn = QPushButton("Help and Support")
        self.support_btn.setStyleSheet(
            "background: #2ecc71; color: white; border-radius: 10px; padding: 4px 18px; margin-right: 10px;"
        )
        self.support_btn.clicked.connect(self.show_help_support_dialog)
        self.support_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header.addWidget(self.support_btn)

        self.version_btn = QPushButton("Version Information")
        self.version_btn.setStyleSheet("background: #3498db; color: white; border-radius: 10px; padding: 4px 18px; margin-right: 10px;")
        self.version_btn.clicked.connect(self.show_version_popup)
        self.version_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header.addWidget(self.version_btn)
        
        self.sign_btn = QPushButton("Sign Out")
        self.sign_btn.setStyleSheet("background: #e74c3c; color: white; border-radius: 10px; padding: 4px 18px;")
        self.sign_btn.clicked.connect(self.handle_sign_out)
        self.sign_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        header.addWidget(self.sign_btn)

        self.apply_language(self.current_language)
        
        dashboard_layout.addLayout(header)

        # ── Phase-3A: Critical arrhythmia alert banner (hidden by default) ───
        self._alert_banner_lbl = QLabel("")
        self._alert_banner_lbl.setAlignment(Qt.AlignCenter)
        self._alert_banner_lbl.setWordWrap(True)
        self._alert_banner_lbl.setStyleSheet(
            "QLabel {"
            "  background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #b91c1c,stop:0.5 #dc2626,stop:1 #b91c1c);"
            "  color: #ffffff;"
            "  font-size: 14px;"
            "  font-weight: bold;"
            "  padding: 8px 16px;"
            "  border-radius: 8px;"
            "  letter-spacing: 0.5px;"
            "}"
        )
        self._alert_banner_lbl.setVisible(False)
        dashboard_layout.addWidget(self._alert_banner_lbl)

        # ── Phase-3B: Offline queue pending badge (hidden when queue empty) ──
        self._queue_badge_lbl = QLabel("")
        self._queue_badge_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._queue_badge_lbl.setStyleSheet(
            "QLabel {"
            "  background: #1d4ed8;"
            "  color: #ffffff;"
            "  font-size: 11px;"
            "  font-weight: bold;"
            "  padding: 4px 14px;"
            "  border-radius: 12px;"
            "}"
        )
        self._queue_badge_lbl.setVisible(False)
        # Place badge in a right-aligned row
        _badge_row = QHBoxLayout()
        _badge_row.addStretch()
        _badge_row.addWidget(self._queue_badge_lbl)
        dashboard_layout.addLayout(_badge_row)

        # --- Device Connection Monitor ---
        self.device_connected = False
        self.device_port = None
        self.device_version = None
        self.settings_manager = SettingsManager()
        self.device_check_timer = QTimer(self)
        self.device_check_timer.timeout.connect(self.check_device_connection)
        # self.device_check_timer.start(100) # Check every 0.1 second
        self.device_check_timer.start(1500 if is_low_spec_mode() else 500) # Reduced frequency on weak machines

        self._device_scan_in_progress = False
        self._last_device_scan_time = 0
        self._initial_scan_completed = False
        self._last_available_ports = [] # Track port changes to avoid redundant scanning

        # Initialize UI as disconnected
        self.update_device_ui(False)

        # --- Cloud Auto Sync: periodically back up reports when online ---
        self._cloud_sync_in_progress = False
        self.cloud_auto_timer = QTimer(self)
        self.cloud_auto_timer.timeout.connect(self.auto_sync_to_cloud)
        self.cloud_auto_timer.start(30000 if is_low_spec_mode() else 15000)  # slower on low-spec systems
        
        # --- Greeting and Date Row ---
        greet_row = QHBoxLayout()
        # Show full name if available, otherwise username
        display_name = self.user_details.get('full_name', username) or username or 'User'
        user_info_lines = [f"<span style='font-size:18pt;font-weight:bold;'>{self._compute_greeting()}, {display_name}</span>"]
        
        # Add user details if available
        if self.user_details:
            details = []
            if self.user_details.get('age'):
                details.append(f"Age: {self.user_details.get('age')}")
            if self.user_details.get('gender'):
                details.append(f"Gender: {self.user_details.get('gender')}")
            if details:
                user_info_lines.append(f"<span style='color:#666; font-size:11pt;'>{' | '.join(details)}</span>")
                
        
        user_info_lines.append("<span style='color:#888;'>Welcome to your ECG dashboard</span>")
        
        self.greet_label = QLabel("<br>".join(user_info_lines))
        self.greet_label.setFont(QFont("Arial", 16))
        self.greet_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        greet_row.addWidget(self.greet_label)
        greet_row.addStretch()
        
        # History button (orange dark suede, left of Hyperkalemia)
        self.history_btn = QPushButton("History")
        self.history_btn.setStyleSheet("background: #2a2a2a; color: white; border-radius: 16px; padding: 8px 24px;")
        self.history_btn.clicked.connect(self.open_history_window)
        self.history_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.history_btn)

        # Hyperkalemia Test button (orange suede color, right of History, left of HRV Test)
        # Define disabled style for initial state to prevent flash of active buttons
        grey_style = "background: #cccccc; color: #666666; border-radius: 16px; padding: 8px 24px;"

        self.hyperkalemia_test_btn = QPushButton("Hyperkalemia Test")
        self.hyperkalemia_test_btn.setEnabled(False)
        self.hyperkalemia_test_btn.setStyleSheet(grey_style)
        self.hyperkalemia_test_btn.clicked.connect(self.open_hyperkalemia_test)
        self.hyperkalemia_test_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.hyperkalemia_test_btn)
        
        # HRV Test button (red color, left of ECG Lead Test 12)
        self.hrv_test_btn = QPushButton("HRV Test")
        self.hrv_test_btn.setEnabled(False)
        self.hrv_test_btn.setStyleSheet(grey_style)
        self.hrv_test_btn.clicked.connect(self.open_hrv_test)
        self.hrv_test_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.hrv_test_btn.setVisible(True)
        greet_row.addWidget(self.hrv_test_btn)
        
        self.date_btn = QPushButton("12-Lead ECG")
        self.date_btn.setEnabled(False)
        self.date_btn.setStyleSheet(grey_style)
        self.date_btn.clicked.connect(self.go_to_lead_test)
        self.date_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.date_btn)

        # --- Add Comprehensive ECG Analysis Button ---
        self.holter_btn = QPushButton("Comprehensive ECG")
        self.holter_btn.setStyleSheet("background: #008000; color: white; border-radius: 16px; padding: 8px 24px; font-weight: bold;")
        self.holter_btn.clicked.connect(self.open_holter_from_dashboard)
        self.holter_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.holter_btn)

        # --- Add Chatbot Button ---
        self.chatbot_btn = QPushButton("AI Chatbot")
        self.chatbot_btn.setStyleSheet("background: #2453ff; color: white; border-radius: 16px; padding: 8px 24px;")
        self.chatbot_btn.clicked.connect(self.open_chatbot_dialog)
        self.chatbot_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # Disabled per request
        self.chatbot_btn.setEnabled(False)
        self.chatbot_btn.setVisible(False)
        greet_row.addWidget(self.chatbot_btn)
        
        # --- Add Waveform Analysis Button ---
        self.analysis_btn = QPushButton("Waveform Analysis")
        self.analysis_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 16px; padding: 8px 24px; font-weight: bold;")
        self.analysis_btn.clicked.connect(self.open_analysis_window)
        self.analysis_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        greet_row.addWidget(self.analysis_btn)



        dashboard_layout.addLayout(greet_row)

        # --- Main Grid ---
        # Create a scroll area for responsive design
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setSpacing(20)
        
        # --- Heart Rate Card --- 
        heart_card = QFrame()
        heart_card.setStyleSheet("background: white; border-radius: 16px;")
        heart_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        heart_layout = QVBoxLayout(heart_card)
        
        self.heart_label = QLabel("Live Heart Rate Overview")
        self.heart_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.heart_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        heart_layout.addWidget(self.heart_label)
        
        heart_img = QLabel()
        # Use portable path for the heart image asset
        heart_img_path = get_asset_path("her.png")
        print(f"Heart image path: {heart_img_path}")  # Debugging line to check the path
        # Ensure os module is available
        import os
        print(f"Heart image exists: {os.path.exists(heart_img_path)}")  # Check if the file exists
        
        # Load the heart image with error handling
        if os.path.exists(heart_img_path):
            self.heart_pixmap = QPixmap(heart_img_path)
            if self.heart_pixmap.isNull():
                print(f"Error: Failed to load heart image from {heart_img_path}")
                # Create a placeholder pixmap
                self.heart_pixmap = QPixmap(220, 220)
                self.heart_pixmap.fill(Qt.lightGray)
        else:
            print(f"Error: Heart image not found at {heart_img_path}")
            # Create a placeholder pixmap
            self.heart_pixmap = QPixmap(220, 220)
            self.heart_pixmap.fill(Qt.lightGray)
        self.heart_base_size = 220
        heart_img.setFixedSize(self.heart_base_size + 20, self.heart_base_size + 20)
        heart_img.setAlignment(Qt.AlignCenter)
        heart_img.setPixmap(self.heart_pixmap.scaled(self.heart_base_size, self.heart_base_size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        heart_img.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        heart_layout.addWidget(heart_img, alignment=Qt.AlignCenter)
        
        # Live stress and HRV labels removed per request
        

        
        grid.addWidget(heart_card, 0, 0, 2, 1)
        
        # --- Heartbeat Animation ---
        self.heart_img = heart_img
        self.heartbeat_phase = 0
        self.current_heart_rate = 60  # Default heart rate
        self.last_beat_time = 0
        self.beat_interval = 1000  # Default 1 second between beats (60 BPM)
        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self.animate_heartbeat)
        self.heartbeat_timer.start(150 if is_low_spec_mode() else 100)  # 6-7 FPS on low-spec, 10 FPS normal
        
        # --- Heartbeat Sound ---
        try:
            if QSound is not None:
                # Try to load heartbeat sound file
                heartbeat_sound_path = get_asset_path("heartbeat.wav")
                if os.path.exists(heartbeat_sound_path):
                    self.heartbeat_sound = QSound(heartbeat_sound_path)
                    print(f" Heartbeat sound loaded: {heartbeat_sound_path}")
                else:
                    print(f" Heartbeat sound not found at: {heartbeat_sound_path}")
                    # Create a synthetic heartbeat sound
                    self.create_heartbeat_sound()
            else:
                print(" QSound not available - heartbeat sound disabled")
                self.heartbeat_sound = None
                self.heartbeat_sound_enabled = False
        except Exception as e:
            print(f" Could not load heartbeat sound: {e}")
            self.heartbeat_sound = None
            self.heartbeat_sound_enabled = False
        
        # --- ECG Recording (Animated Chart) ---
        ecg_card = QFrame()
        ecg_card.setStyleSheet("background: white; border-radius: 16px;")
        ecg_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ecg_layout = QVBoxLayout(ecg_card)
        
        self.ecg_label = QLabel("ECG Recording")
        self.ecg_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.ecg_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ecg_layout.addWidget(self.ecg_label)
        
        self.ecg_canvas = MplCanvas(width=4, height=2)
        self.ecg_canvas.axes.set_facecolor("#eee")
        self.ecg_canvas.axes.set_xticks([])
        self.ecg_canvas.axes.set_yticks([])
        self.ecg_canvas.axes.set_title("Lead II", fontsize=10)
        # Set fixed Y-axis limits to match ECG 12-lead page (0-4096)
        self.ecg_canvas.axes.set_ylim(0, 4096)
        self.ecg_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ecg_layout.addWidget(self.ecg_canvas)
        
        grid.addWidget(ecg_card, 1, 1)
        
        # --- Total Visitors (Medical Stats Panel) ---
        visitors_card = QFrame()
        visitors_card.setStyleSheet("background: white; border-radius: 16px;")
        visitors_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        visitors_layout = QVBoxLayout(visitors_card)
        visitors_layout.setContentsMargins(14, 12, 14, 12)
        visitors_layout.setSpacing(6)

        current_month = datetime.datetime.now().month
        current_year = datetime.datetime.now().year

        # --- Header row: title + total badge ---
        import calendar
        header_row = QHBoxLayout()
        self.visitors_label = QLabel(f"Visitors - Last 6 Months ({current_year})")
        self.visitors_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.visitors_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        header_row.addWidget(self.visitors_label)
        header_row.addStretch()

        # total badge (will be updated after counting)
        self._visitors_total_badge = QLabel("-")
        self._visitors_total_badge.setAlignment(Qt.AlignCenter)
        self._visitors_total_badge.setStyleSheet("""
            QLabel {
                background: #ff6600;
                color: white;
                border-radius: 10px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        header_row.addWidget(self._visitors_total_badge)
        visitors_layout.addLayout(header_row)

        # thin separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #f0f0f0; background: #f0f0f0; max-height: 1px;")
        visitors_layout.addWidget(sep)

        # --- Build month data ---
        month_names = []
        month_data = []
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            sessions_dir = os.path.join(base_dir, 'reports', 'sessions')
            for i in range(5, -1, -1):
                target_month = ((current_month - i - 1) % 12) + 1
                target_year = current_year if (current_month - i) > 0 else current_year - 1
                month_names.append(calendar.month_name[target_month][:3])
                count = 0
                if os.path.exists(sessions_dir):
                    for filename in os.listdir(sessions_dir):
                        if filename.endswith('.jsonl'):
                            try:
                                parts = filename.split('_')
                                if len(parts) >= 3:
                                    date_str = parts[-2]
                                    if len(date_str) == 8:
                                        if int(date_str[:4]) == target_year and int(date_str[4:6]) == target_month:
                                            count += 1
                            except Exception:
                                continue
                month_data.append(max(1, count))
        except Exception as e:
            print(f" Could not calculate visitor stats: {e}")
            month_names = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
            month_data = [1, 1, 1, 1, 1, 1]

        total_visits = sum(month_data)
        self._visitors_total_badge.setText(f"Total  {total_visits}")
        _max_val = max(month_data) if month_data else 1

        # Bar colors (orange gradient, darkest = most recent)
        bar_colors = ["#ffcc99", "#ffaa66", "#ff8c44", "#ff7722", "#ff6600", "#e65c00"]

        # --- Render one row per month ---
        # Column header labels
        col_header = QHBoxLayout()
        col_header.setSpacing(0)

        lbl_mon = QLabel("Month")
        lbl_mon.setFixedWidth(36)
        lbl_mon.setStyleSheet("font-size: 9px; color: #aaa; font-weight: 600;")
        col_header.addWidget(lbl_mon)

        col_header.addSpacing(4)

        lbl_bar_h = QLabel("Sessions")
        lbl_bar_h.setStyleSheet("font-size: 9px; color: #aaa; font-weight: 600;")
        col_header.addWidget(lbl_bar_h, 1)

        lbl_pct_h = QLabel("  Share")
        lbl_pct_h.setFixedWidth(46)
        lbl_pct_h.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_pct_h.setStyleSheet("font-size: 9px; color: #aaa; font-weight: 600;")
        col_header.addWidget(lbl_pct_h)

        lbl_trend_h = QLabel("Trend")
        lbl_trend_h.setFixedWidth(32)
        lbl_trend_h.setAlignment(Qt.AlignCenter)
        lbl_trend_h.setStyleSheet("font-size: 9px; color: #aaa; font-weight: 600;")
        col_header.addWidget(lbl_trend_h)

        visitors_layout.addLayout(col_header)

        # Show latest month on top (descending by recency), but keep trend computed vs the
        # *previous* chronological month (e.g. May vs Apr).
        display_order = list(range(len(month_names) - 1, -1, -1))

        for display_idx, src_idx in enumerate(display_order):
            mon = month_names[src_idx]
            val = month_data[src_idx]
            color = bar_colors[display_idx] if display_idx < len(bar_colors) else "#ff6600"

            pct = (val / total_visits * 100) if total_visits > 0 else 0
            bar_fill = int((val / _max_val) * 100)

            # Trend vs previous month (chronological)
            if src_idx == 0:
                trend_txt, trend_color = "-", "#bbb"
            else:
                diff = val - month_data[src_idx - 1]
                if diff > 0:
                    trend_txt, trend_color = "^", "#27ae60"
                elif diff < 0:
                    trend_txt, trend_color = "v", "#e74c3c"
                else:
                    trend_txt, trend_color = ">", "#f39c12"

            row = QHBoxLayout()
            row.setSpacing(0)
            row.setContentsMargins(0, 0, 0, 0)

            # Month label
            lbl_month = QLabel(mon)
            lbl_month.setFixedWidth(36)
            lbl_month.setStyleSheet(
                "font-size: 11px; font-weight: 700; color: #444;")
            row.addWidget(lbl_month)

            row.addSpacing(4)

            # Progress bar + count inside
            bar_container = QFrame()
            bar_container.setFixedHeight(22)
            bar_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            bar_container.setStyleSheet("background: #f5f5f5; border-radius: 5px;")
            bar_inner_layout = QHBoxLayout(bar_container)
            bar_inner_layout.setContentsMargins(0, 0, 0, 0)
            bar_inner_layout.setSpacing(0)

            # filled portion
            bar_fill_widget = QFrame()
            bar_fill_widget.setFixedHeight(22)
            bar_fill_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            bar_fill_widget.setStyleSheet(
                f"background: {color}; border-radius: 5px;")
            # Use a fixed width ratio approach via QLabel inside
            bar_label = QLabel(f" {val}")
            bar_label.setStyleSheet(
                f"background: {color}; color: {'#333' if display_idx < 3 else '#c44'}; "
                f"font-size: 10px; font-weight: 700; border-radius: 5px; "
                f"min-width: {bar_fill}px;")
            bar_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            bar_inner_layout.addWidget(bar_label, bar_fill)

            # empty portion
            empty = QLabel()
            empty.setStyleSheet("background: #f5f5f5; border-radius: 5px;")
            bar_inner_layout.addWidget(empty, 100 - bar_fill)

            row.addWidget(bar_container, 1)

            # Percentage label
            lbl_pct = QLabel(f"{pct:.0f}%")
            lbl_pct.setFixedWidth(46)
            lbl_pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl_pct.setStyleSheet(
                "font-size: 11px; font-weight: 800; color: #ff6600; padding-right: 4px;")
            row.addWidget(lbl_pct)

            # Trend indicator
            lbl_trend = QLabel(trend_txt)
            lbl_trend.setFixedWidth(32)
            lbl_trend.setAlignment(Qt.AlignCenter)
            lbl_trend.setStyleSheet(
                f"font-size: 13px; font-weight: bold; color: {trend_color};")
            row.addWidget(lbl_trend)

            visitors_layout.addLayout(row)

        visitors_layout.addStretch()
        grid.addWidget(visitors_card, 1, 2)
        
        # --- Schedule Card ---
        schedule_card = QFrame()
        schedule_card.setStyleSheet("background: white; border-radius: 16px;")
        schedule_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        schedule_layout = QVBoxLayout(schedule_card)
        
        schedule_label = QLabel("Calendar")
        schedule_label.setFont(QFont("Arial", 14, QFont.Bold))
        schedule_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        schedule_layout.addWidget(schedule_label)
        
        
        # Create custom navigation header for calendar
        calendar_nav = QFrame()
        calendar_nav.setStyleSheet("background: #f9f9f9; border-radius: 8px; border: 1px solid #e0e0e0;")
        calendar_nav_layout = QHBoxLayout(calendar_nav)
        calendar_nav_layout.setContentsMargins(8, 8, 8, 8)
        calendar_nav_layout.setSpacing(10)
        
        # Previous month button
        self.prev_month_btn = QPushButton("<")
        self.prev_month_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff8533, stop:1 #ff6600);
                color: white;
                border: 1px solid #ff6600;
                border-radius: 15px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffaa66, stop:1 #ff8533);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6600, stop:1 #ff5500);
            }
        """)
        self.prev_month_btn.clicked.connect(self.go_to_prev_month)
        calendar_nav_layout.addWidget(self.prev_month_btn)
        
        # Month label with rounded background
        self.month_label = QLabel("February")
        self.month_label.setAlignment(Qt.AlignCenter)
        self.month_label.setStyleSheet("""
            QLabel {
                background: #ffffff;
                color: #222;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: 600;
                min-width: 80px;
            }
        """)
        calendar_nav_layout.addWidget(self.month_label)
        
        # Year label with rounded background
        self.year_label = QLabel("2026")
        self.year_label.setAlignment(Qt.AlignCenter)
        self.year_label.setStyleSheet("""
            QLabel {
                background: #ffffff;
                color: #222;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 13px;
                font-weight: 600;
                min-width: 60px;
            }
        """)
        calendar_nav_layout.addWidget(self.year_label)
        
        # Next month button
        self.next_month_btn = QPushButton(">")
        self.next_month_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff8533, stop:1 #ff6600);
                color: white;
                border: 1px solid #ff6600;
                border-radius: 15px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffaa66, stop:1 #ff8533);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ff6600, stop:1 #ff5500);
            }
        """)
        self.next_month_btn.clicked.connect(self.go_to_next_month)
        calendar_nav_layout.addWidget(self.next_month_btn)
        
        schedule_layout.addWidget(calendar_nav)
        
        # Calendar widget (hide default navigation)
        self.schedule_calendar = QCalendarWidget()
        self.schedule_calendar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.schedule_calendar.setMinimumHeight(200)
        self.schedule_calendar.setMaximumHeight(260)
        self.schedule_calendar.setMinimumWidth(280)
        self.schedule_calendar.setNavigationBarVisible(False)  # Hide default navigation
        # Never allow selecting/report filtering into future dates.
        self.schedule_calendar.setMaximumDate(QDate.currentDate())
        
        self.schedule_calendar.setStyleSheet("""
        QCalendarWidget QWidget { 
            background: #ffffff; 
            color: #222; 
        }
        QCalendarWidget QAbstractItemView {
            background: #ffffff; 
            color: #222;
            selection-background-color: #ff6600; 
            selection-color: #fff;
            font-size: 12px;
        }
        QCalendarWidget QToolButton { 
            color: #222; 
            background: transparent;
            padding: 4px;
            margin: 2px;
        }
        QCalendarWidget QToolButton:hover {
            background: #ffe6cc;
            border-radius: 4px;
        }
    """)

        # Highlight last ECG usage date in red
        from PyQt5.QtGui import QTextCharFormat, QColor
        from datetime import date as _date_cls
        last_ecg_file = 'last_ecg_date.json'
        today = _date_cls.today()
        # Try to load last ECG date from file
        last_ecg_date = None
        if os.path.exists(last_ecg_file):
            with open(last_ecg_file, 'r') as f:
                try:
                    data = json.load(f)
                    last_ecg_date = data.get('last_ecg_date')
                except Exception:
                    last_ecg_date = None
        if last_ecg_date:
            try:
                y, m, d = map(int, last_ecg_date.split('-'))
                last_date = QDate(y, m, d)
                fmt = QTextCharFormat()
                fmt.setBackground(QColor('red'))
                fmt.setForeground(QColor('white'))
                self.schedule_calendar.setDateTextFormat(last_date, fmt)
            except Exception:
                pass

        # Apply calendar date restrictions for new users
        self._apply_new_user_calendar_restrictions()
        
        # connect date click to filter reports
        self.schedule_calendar.clicked.connect(self.on_calendar_date_selected)
        
        # Update labels when calendar page changes
        self.schedule_calendar.currentPageChanged.connect(self.update_calendar_labels)
        
        # Connect to page change to track month navigation
        self.schedule_calendar.currentPageChanged.connect(self.on_calendar_page_changed)
        
        # Initialize calendar labels
        self.update_calendar_labels()
        
        # Disable double-click activation (prevents popup window)
        try:
            self.schedule_calendar.activated.disconnect()
        except:
            pass  # No activated signal connected

       
        
        schedule_layout.addWidget(self.schedule_calendar)
        grid.addWidget(schedule_card, 2, 0)
        # --- ECG Interpretation Card ---
        issue_card = QFrame()
        issue_card.setStyleSheet("background: white; border-radius: 16px;")
        issue_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        issue_layout = QVBoxLayout(issue_card)
        
        issue_label = QLabel("ECG Interpretation")
        issue_label.setFont(QFont("Arial", 14, QFont.Bold))
        issue_label.setStyleSheet("color: #ff6600;")
        issue_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        issue_layout.addWidget(issue_label)
        
        # Live conclusion box that updates based on ECG analysis - BIGGER SIZE
        self.conclusion_box = QTextEdit()
        self.conclusion_box.setReadOnly(True)
        self.conclusion_box.setStyleSheet("background: #f7f7f7; border: none; font-size: 12px; padding: 10px;")
        self.conclusion_box.setMinimumHeight(300)  # Increased from 180 to 300
        self.conclusion_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Set initial placeholder text
        self.conclusion_box.setHtml("""
            <p style='color: #888; font-style: italic;'>
            No ECG data available yet.<br><br>
            Start an ECG test to see your personalized analysis and recommendations.
            </p>
        """)
        
        issue_layout.addWidget(self.conclusion_box)
        # # Small footer box below the conclusion (~3 cm height)
        # self.conclusion_footer = QFrame()
        # self.conclusion_footer.setStyleSheet("background: #f7f7f7; border: none; border-radius: 10px;")
        # self.conclusion_footer.setFixedHeight(115)
        # self.conclusion_footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # _footer_layout = QHBoxLayout(self.conclusion_footer)
        # _footer_layout.setContentsMargins(10, 8, 10, 8)
        # _footer_layout.addWidget(QLabel(""))
        # issue_layout.addWidget(self.conclusion_footer)

        grid.addWidget(issue_card, 2, 1, 1, 1)

        # Separate card for Additional Notes (outside Conclusion)
        notes_card = QFrame()
        notes_card.setStyleSheet("background: white; border-radius: 16px;")
        notes_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        notes_card.setFixedHeight(160)
        notes_layout = QVBoxLayout(notes_card)
        notes_layout.setContentsMargins(12, 12, 12, 12)
        notes_layout.setSpacing(8)
        notes_title = QLabel("METRICS")
        notes_title.setFont(QFont("Arial", 14, QFont.Bold))
        notes_title.setStyleSheet("color: #ff6600;")
        notes_layout.addWidget(notes_title)
        self.parameters_text = QTextEdit()
        self.parameters_text.setReadOnly(True)
        self.parameters_text.setFixedHeight(90)
        self.parameters_text.setLineWrapMode(QTextEdit.NoWrap)  # Single row, no wrap
        self.parameters_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide horizontal scroll
        self.parameters_text.setStyleSheet("background: #f9f9f9; border: none; font-size: 12px; padding: 6px 8px;")
        notes_layout.addWidget(self.parameters_text)
        # Slider to control horizontal scroll for parameters_text
        self.parameters_slider = QSlider(Qt.Horizontal)
        self.parameters_slider.setMinimum(0)
        self.parameters_slider.setMaximum(0)
        self.parameters_slider.setSingleStep(20)
        self.parameters_slider.setPageStep(self.parameters_text.viewport().width())
        self.parameters_slider.setStyleSheet("QSlider::groove:horizontal { height: 6px; background: #ececec; border-radius: 3px; } QSlider::handle:horizontal { background: #ff6600; width: 14px; border-radius: 7px; margin: -5px 0; } QSlider::sub-page:horizontal { background: #ffd5b3; border-radius: 3px; }")
        notes_layout.addWidget(self.parameters_slider)

        # Keep a reference so we can show/hide the METRICS panel based on user actions
        self.metrics_notes_card = notes_card

        # Sync slider with the hidden horizontal scrollbar
        _hbar = self.parameters_text.horizontalScrollBar()
        _hbar.rangeChanged.connect(lambda _min, _max: self.parameters_slider.setMaximum(_max))
        _hbar.valueChanged.connect(self.parameters_slider.setValue)
        self.parameters_slider.valueChanged.connect(_hbar.setValue)
        # Make the Additional Notes card span all 3 columns (full row width)
        grid.addWidget(notes_card, 3, 0, 1, 3)

        # Hide METRICS panel by default; it will be shown when a report is selected
        self.metrics_notes_card.hide()

        # --- Recent Reports Card ---
        reports_card = QFrame()
        reports_card.setStyleSheet(
            "background: white; border-radius: 16px;"
        )
        reports_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        reports_v = QVBoxLayout(reports_card)
        ttl = QLabel("Recent Reports")
        ttl.setFont(QFont("Arial", 14, QFont.Bold))
        ttl.setStyleSheet("color: #ff6600;")
        reports_v.addWidget(ttl)

        # Scroll area for list
        self.reports_list_widget = QWidget()
        self.reports_list_layout = QVBoxLayout(self.reports_list_widget)
        self.reports_list_layout.setContentsMargins(4, 4, 4, 4)  # Add margins to prevent button cropping
        self.reports_list_layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(180)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # Allow horizontal scroll if needed
        scroll.setWidget(self.reports_list_widget)
        reports_v.addWidget(scroll)

        self.refresh_recent_reports_ui()

        grid.addWidget(reports_card, 2, 2, 1, 1)
        
        # --- CardioX Metrics Cards ---
        metrics_card = QFrame()
        metrics_card.setStyleSheet("background: white; border-radius: 16px;")
        metrics_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        metrics_layout = QHBoxLayout(metrics_card)
        
        # Store metric labels for live update
        self.metric_labels = {}
        metric_info = [
            ("HR", "--", "BPM", "heart_rate"),
            ("PR", "--", "ms", "pr_interval"),
            ("QRS Complex", "--", "ms", "qrs_duration"),
            ("QT/QTc", "--", "ms", "qtc_interval"),
        ]
        
        for title, value, unit, key in metric_info:
            box = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setFont(QFont("Arial", 12, QFont.Bold))
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            val = QLabel(f"{value} {unit}")
            val.setFont(QFont("Arial", 18, QFont.Bold))
            val.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            # Add triple-click functionality to heart rate metric
            if key == "heart_rate":
                val.mousePressEvent = self.heart_rate_triple_click
                try:
                    val.setMinimumWidth(val.fontMetrics().horizontalAdvance("000 BPM"))
                except Exception:
                    pass
            
            box.addWidget(lbl)
            box.addWidget(val)
            metrics_layout.addLayout(box, 1)
            self.metric_labels[key] = val  
        
        grid.addWidget(metrics_card, 0, 1, 1, 2)
        
        # Add the grid widget to the scroll area
        scroll_area.setWidget(grid_widget)
        
        # Add scroll area to dashboard layout
        dashboard_layout.addWidget(scroll_area)

        # --- Professional Status Footer Bar ---
        footer_bar = QFrame()
        footer_bar.setFixedHeight(26)
        footer_bar.setStyleSheet("""
            QFrame {
                background: #FFFFFF;
                border-top: 1px solid #333333;
            }
        """)
        footer_layout = QHBoxLayout(footer_bar)
        footer_layout.setContentsMargins(12, 0, 12, 0)
        footer_layout.setSpacing(0)

        # Divider helper
        def _footer_divider():
            d = QFrame()
            d.setFrameShape(QFrame.VLine)
            d.setStyleSheet("color: #3a3a3a; background: #3a3a3a; max-width: 1px; margin: 4px 8px;")
            return d

        # Software activation status (reuse existing label)
        self.license_status_label.setParent(footer_bar)
        self.license_status_label.setStyleSheet("""
            QLabel {
                color: #3dba6e;
                background: transparent;
                font-size: 8pt;
                font-weight: 700;
                padding: 0px 4px;
                letter-spacing: 0.4px;
            }
        """)
        footer_layout.addWidget(self.license_status_label)

        footer_layout.addWidget(_footer_divider())

        # App name / branding
        _brand = QLabel("CardioX V1.0")
        _brand.setStyleSheet("""
            QLabel {
                color: #555555;
                background: transparent;
                font-size: 8pt;
                padding: 0px 4px;
            }
        """)
        footer_layout.addWidget(_brand)

        footer_layout.addStretch()

        # Current time ticker
        self._footer_clock = QLabel()
        self._footer_clock.setStyleSheet("""
            QLabel {
                color: #555555;
                background: transparent;
                font-size: 8pt;
                padding: 0px 4px;
            }
        """)
        footer_layout.addWidget(self._footer_clock)

        def _tick_clock():
            self._footer_clock.setText(datetime.datetime.now().strftime("%H:%M:%S"))

        _tick_clock()
        self._footer_clock_timer = QTimer(self)
        self._footer_clock_timer.timeout.connect(_tick_clock)
        self._footer_clock_timer.start(1000)

        dashboard_layout.addWidget(footer_bar)
        
        
        # --- ECG Animation Setup ---
        self.ecg_x = np.linspace(0, 2, 500)
        self.ecg_y = 2048 + 150 * np.sin(2 * np.pi * 2 * self.ecg_x) + 30 * np.random.randn(500)  # Centered at 2048 for 0-4096 range
        self.ecg_line, = self.ecg_canvas.axes.plot(self.ecg_x, self.ecg_y, color="#ff6600", linewidth=0.5, antialiased=True)
        # Reduce CPU/GPU usage: lower refresh rate slightly and disable frame caching
        self.anim = FuncAnimation(
            self.ecg_canvas.figure,
            self.update_ecg,
            interval=140 if is_low_spec_mode() else 85,  # lighter refresh on weaker machines
            blit=True,
            cache_frame_data=False,   # prevent unbounded cache growth
            save_count=100
        )
        
        # --- Dashboard Metrics Update Timer ---
        self.metrics_timer = QTimer(self)
        self.metrics_timer.timeout.connect(self.update_dashboard_metrics_from_ecg)
        self.metrics_timer.start(2000 if is_low_spec_mode() else 1000)  # Lighter polling on low-spec systems
        print("⏰ Dashboard metrics timer started - updates every 1 second")
        
        # Force initial metrics update immediately to ensure values appear within 10 seconds
        try:
            self.update_dashboard_metrics_from_ecg()
            print(" Initial metrics update completed - values should appear immediately")
        except Exception as e:
            print(f" Initial metrics update failed: {e}")
        
        # Session timer removed - no longer needed
        # Add dashboard_page to stack
        self.page_stack.addWidget(self.dashboard_page)
        # --- ECG Test Page ---
        try:
            # Add the src directory to the path for ECG imports
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
                print(f" Added src directory to path: {src_dir}")
            
            from ecg.twelve_lead_test import ECGTestPage
            print(" ECG Test Page imported successfully")
                    
        except ImportError as e:
            print(f" ECG Test Page import error: {e}")
            print(" Creating fallback ECG Test Page")
            # Create a fallback ECG test page
            class ECGTestPage(QWidget):
                def __init__(self, title, parent, *args, **kwargs):
                    super().__init__()
                    self.title = title
                    self.parent = parent
                    self.dashboard_callback = None
                    layout = QVBoxLayout()
                    label = QLabel("ECG Test Page - Import Error")
                    label.setAlignment(Qt.AlignCenter)
                    layout.addWidget(label)
                    self.setLayout(layout)
                    print(" Using fallback ECG Test Page")
        # Pass the shared SettingsManager so display filter settings (DFT/EMG/AC)
        # stay consistent across dashboard and ECG test page views.
        self.ecg_test_page = ECGTestPage("12 Lead ECG Test", self.page_stack, settings_manager=self.settings_manager)
        self.ecg_test_page.dashboard_callback = self.update_ecg_metrics
        # Pass username and dashboard reference to ECGTestPage for report filtering
        self.ecg_test_page.dashboard_instance = self
        self.ecg_test_page.current_username = self.username

        if hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
        
        self.page_stack.addWidget(self.ecg_test_page)
        # --- Main layout ---
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.page_stack)
        self.setLayout(main_layout)
        self.page_stack.setCurrentWidget(self.dashboard_page)

        # Add a content_frame for ECGMenu to use
        self.content_frame = QFrame(self)
        self.content_frame.setStyleSheet("background: transparent; border: none;")
        main_layout.addWidget(self.content_frame)

        self.setLayout(main_layout)
        self.page_stack.setCurrentWidget(self.dashboard_page)

        # Connect page stack changes to pause/resume dashboard background timers/animations
        self.page_stack.currentChanged.connect(self.on_page_changed)

        # Check if the license is restricted (offline stable mismatch)
        from utils.license_manager import is_license_restricted
        if is_license_restricted():
            self.set_read_only_license_mode(True, "Hardware change detected. Internet connection required to reverify.")
            
            # Create a prominent warning banner widget inside the dashboard page
            warning_banner = QFrame()
            warning_banner.setStyleSheet("""
                QFrame {
                    background-color: #ffebee;
                    border: 2px solid #ffcdd2;
                    border-radius: 12px;
                }
                QLabel {
                    border: none;
                    background: transparent;
                }
            """)
            wb_layout = QHBoxLayout(warning_banner)
            wb_layout.setContentsMargins(20, 15, 20, 15)
            wb_layout.setSpacing(15)
            
            warning_icon = QLabel("⚠️")
            warning_icon.setFont(QFont("Arial", 22))
            wb_layout.addWidget(warning_icon)
            
            warning_text = QLabel(
                "<b>LICENSE REVERIFICATION REQUIRED</b><br><br>"
                "Hardware change detected. Internet connection required to reverify this license.<br>"
                "ECG acquisition is temporarily disabled. Reports and settings remain available."
            )
            warning_text.setFont(QFont("Segoe UI", 11))
            warning_text.setStyleSheet("color: #c62828;")
            warning_text.setWordWrap(True)
            wb_layout.addWidget(warning_text, 1)
            
            dashboard_layout.insertWidget(2, warning_banner)

        # Register keyboard shortcuts
        self._setup_shortcuts()

    def _setup_shortcuts(self):
        """Register keyboard shortcuts for common dashboard actions."""
        def _sc(key, slot):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(slot)
            return sc

        _sc("Ctrl+N", self.show_new_registration_dialog)
        _sc("Ctrl+H", self.open_history_window)

    # Calendar date selection
    
    def update_calendar_labels(self):
        """Update the custom month and year labels based on current calendar date."""
        try:
            current_date = QDate(self.schedule_calendar.yearShown(), self.schedule_calendar.monthShown(), 1)
            month_names = ["January", "February", "March", "April", "May", "June",
                          "July", "August", "September", "October", "November", "December"]
            self.month_label.setText(month_names[current_date.month() - 1])
            self.year_label.setText(str(current_date.year()))
            self._sync_calendar_nav_buttons()
        except Exception as e:
            print(f"Error updating calendar labels: {e}")
    
    def go_to_prev_month(self):
        """Navigate to previous month."""
        try:
            current_page = QDate(self.schedule_calendar.yearShown(), self.schedule_calendar.monthShown(), 1)
            min_page = QDate(self.schedule_calendar.minimumDate().year(),
                                self.schedule_calendar.minimumDate().month(), 1)
            new_page = current_page.addMonths(-1)
            if new_page < min_page:
                new_page = min_page
            self.schedule_calendar.setCurrentPage(new_page.year(), new_page.month())
            self.update_calendar_labels()
        except Exception as e:
            print(f"Error navigating to previous month: {e}")
    
    def go_to_next_month(self):
        """Navigate to next month."""
        try:
            current_page = QDate(self.schedule_calendar.yearShown(), self.schedule_calendar.monthShown(), 1)
            max_page = QDate(self.schedule_calendar.maximumDate().year(),
                                self.schedule_calendar.maximumDate().month(), 1)
            new_page = current_page.addMonths(1)
            if new_page > max_page:
                new_page = max_page
            self.schedule_calendar.setCurrentPage(new_page.year(), new_page.month())
            self.update_calendar_labels()
        except Exception as e:
            print(f"Error navigating to next month: {e}")

    def on_calendar_date_selected(self, qdate):
        try:
            # Clamp date silently to valid calendar range (avoid intrusive popup windows).
            min_d = self.schedule_calendar.minimumDate()
            max_d = self.schedule_calendar.maximumDate()
            safe_date = qdate
            if safe_date < min_d:
                safe_date = min_d
            if safe_date > max_d:
                safe_date = max_d
            if safe_date != qdate:
                self.schedule_calendar.setSelectedDate(safe_date)
            qdate = safe_date
            
            self.reports_filter_date = qdate.toString("yyyy-MM-dd")
            # Set flag to prevent automatic report opening when calendar is clicked
            self._calendar_triggered = True
            self.refresh_recent_reports_ui(self.reports_filter_date)
            self._calendar_triggered = False
        except Exception:
            self._calendar_triggered = False
            self.refresh_recent_reports_ui()  # safe fallback

    def on_calendar_selection_changed(self):
        try:
            qdate = self.schedule_calendar.selectedDate()
            self.reports_filter_date = qdate.toString("yyyy-MM-dd")
            self.refresh_recent_reports_ui(self.reports_filter_date)
        except Exception:
            pass
    
    def _apply_new_user_calendar_restrictions(self):
        """Apply calendar date restrictions for new users based on signup date"""
        try:
            from PyQt5.QtCore import QDate
            from PyQt5.QtGui import QTextCharFormat, QColor
            from datetime import timedelta
            
            # Check if user has signup_date (new user)
            signup_date_str = self.user_details.get('signup_date') or self.user_details.get('registered_at')
            
            # Also check if user logged in with phone number
            has_phone = bool(self.user_details.get('phone') or self.user_details.get('contact'))
            
            if not signup_date_str or not has_phone:
                # Not a new user or no phone number - no restrictions
                return
            
            # Parse signup date
            try:
                # Try parsing different date formats
                if 'T' in signup_date_str or ' ' in signup_date_str:
                    # ISO format with time: "2024-01-15 10:30:00" or "2024-01-15T10:30:00"
                    signup_date_str = signup_date_str.split('T')[0].split(' ')[0]
                
                signup_date_obj = datetime.datetime.strptime(signup_date_str, "%Y-%m-%d").date()
                signup_qdate = QDate(signup_date_obj.year, signup_date_obj.month, signup_date_obj.day)
            except Exception as e:
                print(f"Error parsing signup date: {e}")
                return
            
            # Always allow navigation up to today's date.
            # Restricting to signup+365 can make the calendar appear "stuck" on old months.
            today_qdate = QDate.currentDate()
            max_qdate = today_qdate
            
            # Set date range
            self.schedule_calendar.setMinimumDate(signup_qdate)
            self.schedule_calendar.setMaximumDate(max_qdate)
            
            # Fade dates before signup date (if any are still visible)
            # Get current displayed month
            current_date = self.schedule_calendar.selectedDate()
            if not current_date.isValid():
                current_date = QDate.currentDate()
            
            # Fade all dates before signup date
            fade_format = QTextCharFormat()
            fade_format.setForeground(QColor(200, 200, 200))  # Light gray
            fade_format.setBackground(QColor(240, 240, 240))  # Light background
            
            # Iterate through dates in the visible month and fade past dates
            year = current_date.year()
            month = current_date.month()
            days_in_month = QDate(year, month, 1).daysInMonth()
            
            for day in range(1, days_in_month + 1):
                check_date = QDate(year, month, day)
                if check_date.isValid() and check_date < signup_qdate:
                    self.schedule_calendar.setDateTextFormat(check_date, fade_format)
            
            # Store signup date and max date for later use
            self._user_signup_date = signup_qdate
            self._user_max_date = max_qdate
            self._user_navigated_months = set()  # Track which months user has navigated to
            
            # Start on the latest available month/day.
            self.schedule_calendar.setSelectedDate(max_qdate)
            self.schedule_calendar.setCurrentPage(max_qdate.year(), max_qdate.month())
            
            # Use string representation of max_qdate (avoid undefined variable)
            try:
                max_date_str = max_qdate.toString("yyyy-MM-dd")
            except Exception:
                max_date_str = str(max_qdate)
            print(f" Calendar restrictions applied for new user. Signup: {signup_date_str}, Max: {max_date_str}")
            
        except Exception as e:
            print(f"Error applying calendar restrictions: {e}")
            import traceback
            traceback.print_exc()
    
    def on_calendar_page_changed(self, year, month):
        """Handle calendar page change - apply date locking after navigation"""
        try:
            # Hard-stop any attempt to navigate beyond maximum allowed month.
            max_d = self.schedule_calendar.maximumDate()
            max_page = QDate(max_d.year(), max_d.month(), 1)
            shown_page = QDate(year, month, 1)
            if shown_page > max_page:
                self.schedule_calendar.setCurrentPage(max_page.year(), max_page.month())
                year, month = max_page.year(), max_page.month()

            # Check if this is a new user with restrictions
            if not hasattr(self, '_user_signup_date'):
                self._sync_calendar_nav_buttons()
                return
            
            from PyQt5.QtCore import QDate
            from PyQt5.QtGui import QTextCharFormat, QColor
            
            # Track that user navigated to this month
            month_key = (year, month)
            self._user_navigated_months.add(month_key)
            
            # Lock dates after the current month (if user navigated forward)
            current_date = QDate.currentDate()
            displayed_date = QDate(year, month, 1)
            
            # If displayed month is in the future (after current month), lock dates after it
            if displayed_date > current_date:
                # Lock all dates in months after the displayed month
                lock_format = QTextCharFormat()
                lock_format.setForeground(QColor(150, 150, 150))  # Darker gray
                lock_format.setBackground(QColor(220, 220, 220))  # Gray background
                
                # Lock dates in the displayed month that are after today
                days_in_month = QDate(year, month, 1).daysInMonth()
                for day in range(1, days_in_month + 1):
                    check_date = QDate(year, month, day)
                    if check_date.isValid() and check_date > current_date:
                        self.schedule_calendar.setDateTextFormat(check_date, lock_format)
            
            # Also ensure dates before signup are still faded
            fade_format = QTextCharFormat()
            fade_format.setForeground(QColor(200, 200, 200))
            fade_format.setBackground(QColor(240, 240, 240))
            
            days_in_month = QDate(year, month, 1).daysInMonth()
            for day in range(1, days_in_month + 1):
                check_date = QDate(year, month, day)
                if check_date.isValid() and check_date < self._user_signup_date:
                    self.schedule_calendar.setDateTextFormat(check_date, fade_format)
            self._sync_calendar_nav_buttons()
            
        except Exception as e:
            print(f"Error in calendar page change: {e}")
            import traceback
            traceback.print_exc()

    def _sync_calendar_nav_buttons(self):
        """Enable/disable custom calendar arrows based on current min/max months."""
        try:
            shown = QDate(self.schedule_calendar.yearShown(), self.schedule_calendar.monthShown(), 1)
            min_d = self.schedule_calendar.minimumDate()
            max_d = self.schedule_calendar.maximumDate()
            min_page = QDate(min_d.year(), min_d.month(), 1)
            max_page = QDate(max_d.year(), max_d.month(), 1)
            self.prev_month_btn.setEnabled(shown > min_page)
            self.next_month_btn.setEnabled(shown < max_page)
        except Exception:
            pass

    def _normalize_report_date_key(self, value: str) -> str:
        """Normalize date text to YYYY-MM-DD for reliable calendar filtering."""
        try:
            s = str(value or "").strip()
            if not s:
                return ""
            # Common formats: YYYY-MM-DD, YYYYMMDD, YYYY/MM/DD, DD-MM-YYYY
            s = s.split(" ")[0].split("T")[0]
            if len(s) == 8 and s.isdigit():
                return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
            if "/" in s:
                parts = s.split("/")
            else:
                parts = s.split("-")
            if len(parts) == 3:
                # Already Y-M-D
                if len(parts[0]) == 4:
                    return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
                # D-M-Y
                if len(parts[2]) == 4:
                    return f"{int(parts[2]):04d}-{int(parts[1]):02d}-{int(parts[0]):02d}"
            return s
        except Exception:
            return str(value or "").strip()

    def show_month_dropdown(self, year, month):
        """Show month selection dropdown"""
        from PyQt5.QtWidgets import QComboBox, QDialog, QVBoxLayout, QPushButton, QLabel
        from PyQt5.QtCore import Qt
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Month")
        dialog.setModal(True)
        dialog.setFixedSize(200, 300)
        
        layout = QVBoxLayout(dialog)
        
        # Year selection
        year_label = QLabel("Year:")
        year_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(year_label)
        
        year_combo = QComboBox()
        current_year = year
        for y in range(current_year - 5, current_year + 6):
            year_combo.addItem(str(y))
        year_combo.setCurrentText(str(year))
        layout.addWidget(year_combo)
        
        # Month selection
        month_label = QLabel("Month:")
        month_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(month_label)
        
        month_combo = QComboBox()
        months = ["January", "February", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]
        for i, month_name in enumerate(months):
            month_combo.addItem(month_name)
        month_combo.setCurrentIndex(month - 1)
        layout.addWidget(month_combo)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        ok_button = QPushButton("OK")
        ok_button.setStyleSheet("background: #ff6600; color: white; border-radius: 5px; padding: 8px;")
        ok_button.clicked.connect(lambda: self.apply_calendar_selection(
            int(year_combo.currentText()), month_combo.currentIndex() + 1, dialog))
        
        cancel_button = QPushButton("Cancel")
        cancel_button.setStyleSheet("background: #666; color: white; border-radius: 5px; padding: 8px;")
        cancel_button.clicked.connect(dialog.reject)
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        dialog.exec_()

    def apply_calendar_selection(self, year, month, dialog):
        """Apply the selected year and month to the calendar"""
        try:
            from PyQt5.QtCore import QDate
            # Set the calendar to the selected month/year
            self.schedule_calendar.setCurrentPage(year, month)
            dialog.accept()
        except Exception as e:
            print(f"Error applying calendar selection: {e}")
            dialog.reject()

    def open_holter_from_dashboard(self):
        """Navigate to ECG page and open Comprehensive ECG Analysis menu with options"""
        try:
            # Check if ECG test page is available
            if not hasattr(self, 'ecg_test_page'):
                QMessageBox.warning(self, "Comprehensive ECG Analysis Monitor", "ECG Test Page not initialized.")
                return

            # Ask user for choice: Live ECG or Previous Recording
            msg = QMessageBox(self)
            msg.setWindowTitle("Comprehensive ECG Analysis")
            msg.setText("Choose an action:")
            msg.setIcon(QMessageBox.Question)
            msg.setStyleSheet("""
                QMessageBox { background-color: #1a1a2e; color: white; }
                QLabel { color: white; font-size: 13px; }
            """)
            
            btn_live = msg.addButton("Live ECG Analysis", QMessageBox.ActionRole)
            btn_prev = msg.addButton("View Previous Recording", QMessageBox.ActionRole)
            btn_cancel = msg.addButton("Cancel", QMessageBox.RejectRole)
            
            # Style the buttons — orange theme
            btn_live.setStyleSheet("background: #ff6600; color: white; padding: 10px 20px; font-weight: bold; border-radius: 6px; border: none;")
            btn_prev.setStyleSheet("background: #7a3000; color: white; padding: 10px 20px; font-weight: bold; border-radius: 6px; border: 1px solid #ff6600;")
            btn_cancel.setStyleSheet("background: #2a2a3e; color: #aaaaaa; padding: 10px 20px; border-radius: 6px; border: 1px solid #555;")
            
            msg.exec_()
            
            if msg.clickedButton() == btn_live:
                # Switch to ECG test page
                self.page_stack.setCurrentWidget(self.ecg_test_page)
                # Enable Holter mode and start acquisition automatically
                if hasattr(self.ecg_test_page, 'start_live_holter_from_dashboard'):
                    self.ecg_test_page.start_live_holter_from_dashboard()
            
            elif msg.clickedButton() == btn_prev:
                # Stay on the main dashboard: the Holter workspace opens as
                # its own window, so closing it (X) brings the user back here
                # and not to the 12-lead ECG test page. Reviewing recordings
                # needs no acquisition page underneath.
                # Open up Holter workspace on Record Management tab
                if hasattr(self.ecg_test_page, '_show_holter_ui'):
                    self.ecg_test_page._show_holter_ui(is_recording=False, ecgh_path=None, show_record_mgmt=True)

        except Exception as e:
            print(f"Error opening Comprehensive ECG Analysis from dashboard: {e}")
            QMessageBox.critical(self, "Error", f"Failed to open Comprehensive ECG Analysis Monitor: {str(e)}")

    def open_analysis_window(self):
        """Open the ECG Analysis Window"""
        try:
            from dashboard.analysis_window import ECGAnalysisWindow
            self.analysis_window = ECGAnalysisWindow(self)
            self.analysis_window.show()
            print("✅ ECG Analysis Window opened successfully")
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Failed to import Analysis Window: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Analysis Window: {str(e)}")
    
    def open_chatbot_dialog(self):
        dlg = ChatbotDialog(self)
        dlg.exec_()

    def refresh_recent_reports_ui(self, filter_date=None):
        import os, json
        
        reports_dir = str(data_file("reports"))
        index_path = os.path.join(reports_dir, "index.json")

        # Clear list
        while self.reports_list_layout.count():
            item = self.reports_list_layout.takeAt(0)
            w = item.widget()
            if w: w.setParent(None)

        entries = []
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    entries = json.load(f) or []
            except Exception:
                entries = []

        # Use the calendar’s current filter if none explicitly provided
        if filter_date is None:
            filter_date = getattr(self, "reports_filter_date", None)

        if filter_date:
            fd = self._normalize_report_date_key(filter_date)
            entries = [
                e for e in entries
                if self._normalize_report_date_key(e.get('date', '')) == fd
            ]

        # Filter to only show ECG Report entries (not detailed JSON metadata)
        # ECG Report entries have 'filename' and 'title' keys
        # Detailed metadata entries have 'timestamp' and 'metrics' keys
        entries = [e for e in entries if 'filename' in e and 'title' in e]
        
        # Filter reports by current user - only show reports generated by this user
        if self.username:
            entries = [e for e in entries if e.get('username', '') == self.username]
        else:
            # If no username, only show reports that also have no username (backward compatibility)
            entries = [e for e in entries if not e.get('username', '')]

        for e in entries[:10]:
            # Build row with hover/touch feedback
            row = QHBoxLayout()
            row.setContentsMargins(6, 6, 12, 6)  # Increased right margin to 12px to prevent button cropping
            row.setSpacing(8)  # Add spacing between elements

            cont = QWidget(self.reports_list_widget)
            meta = QLabel(f"{e.get('date','')} {e.get('time','')}  |  {e.get('patient','')}  |  {e.get('title','Report')}", cont)
            meta.setStyleSheet("color: #333333; font-size: 12px;")
            meta.setCursor(Qt.PointingHandCursor)
            meta.setWordWrap(False)  # Prevent text wrapping
            row.addWidget(meta, 1)

            path = os.path.join(reports_dir, e.get('filename',''))

            # Clicking will be bound after container is created to allow row selection

            # "Params" button to load metrics into Parameters box
            # params_btn = QPushButton("Params")
            # params_btn.setStyleSheet("background: #eeeeee; color: #333333; border-radius: 8px; padding: 4px 10px; font-weight: bold;")
            # params_btn.clicked.connect(lambda _, p=path: self.load_metrics_into_parameters(p))
            # row.addWidget(params_btn)

            # "Open" button strictly opens the PDF
            btn = QPushButton("Open", cont)
            btn.setStyleSheet("background: #ff6600; color: white; border-radius: 8px; padding: 4px 12px; font-weight: bold;")
            btn.setMinimumWidth(65)  # Increased minimum width to prevent cropping
            btn.setMaximumWidth(75)  # Set maximum width
            btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  # Fixed size policy to prevent stretching
            btn.clicked.connect(lambda _, p=path: self.open_report_file(p))
            row.addWidget(btn, 0)  # Don't stretch the button
            row.addSpacing(4)  # Add extra spacing after button

            # Container with hover feedback and full-row click
            cont.setLayout(row)
            cont.setMinimumHeight(35)  # Ensure container has minimum height
            cont.setCursor(Qt.PointingHandCursor)
            cont.setStyleSheet("background: transparent; border-radius: 8px;")
            cont._meta_label = meta
            cont._report_path = path

            # Make label click select row and load parameters
            def _label_click_handler(_evt, p=path, w=cont):
                # Skip if triggered by calendar (to prevent automatic actions)
                if getattr(self, '_calendar_triggered', False):
                    return
                try:
                    self._select_report_row(w, p)
                except Exception:
                    pass
                self.load_metrics_into_parameters(p)
            meta.mousePressEvent = _label_click_handler

            # Make entire row clickable to select and load parameters
            def _row_click_handler(_evt, p=path, w=cont):
                # Skip if triggered by calendar (to prevent automatic actions)
                if getattr(self, '_calendar_triggered', False):
                    return
                try:
                    self._select_report_row(w, p)
                except Exception:
                    pass
                self.load_metrics_into_parameters(p)
            cont.mousePressEvent = _row_click_handler

            # Hover effects (enter/leave)
            def _hover_enter(_e, w=cont, m=meta):
                # Keep selected style if this row is selected
                if getattr(self, '_selected_report_widget', None) is w:
                    w.setStyleSheet("background: #ffe6cc; border-radius: 8px; border: 1px solid #ffb366;")
                    m.setStyleSheet("color: #333333; font-size: 12px;")
                else:
                    w.setStyleSheet("background: #fff3e6; border-radius: 8px;")
                    m.setStyleSheet("color: #333333; font-size: 12px; text-decoration: underline;")

            def _hover_leave(_e, w=cont, m=meta):
                # If selected, maintain selection; else clear
                if getattr(self, '_selected_report_widget', None) is w:
                    w.setStyleSheet("background: #ffe6cc; border-radius: 8px; border: 1px solid #ffb366;")
                    m.setStyleSheet("color: #333333; font-size: 12px;")
                else:
                    w.setStyleSheet("background: transparent; border-radius: 8px;")
                    m.setStyleSheet("color: #333333; font-size: 12px;")

            cont.enterEvent = _hover_enter
            cont.leaveEvent = _hover_leave

            # Preserve selected row highlight on UI refresh - DISABLED to prevent automatic selection on calendar date click
            # try:
            #     if getattr(self, '_selected_report_path', None) and os.path.abspath(path) == os.path.abspath(self._selected_report_path):
            #         self._select_report_row(cont, path)
            # except Exception:
            #     pass

            self.reports_list_layout.addWidget(cont)

        spacer = QWidget(); spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.reports_list_layout.addWidget(spacer)

    def open_report_file(self, path):
        from utils.platform_compat import open_file
        # Prevent automatic opening when triggered by calendar
        if getattr(self, '_calendar_triggered', False):
            print(" Blocked automatic report opening from calendar click")
            return
        if not os.path.exists(path):
            return
        print(f" Opening report: {path}")
        open_file(path)

    def _select_report_row(self, widget, path):
        """Select a Recent Reports row and keep it highlighted until another is clicked."""
        try:
            # Clear previous selection if different
            prev = getattr(self, '_selected_report_widget', None)
            if prev is not None and prev is not widget:
                try:
                    prev.setStyleSheet("background: transparent; border-radius: 8px;")
                    if hasattr(prev, '_meta_label') and prev._meta_label is not None:
                        prev._meta_label.setStyleSheet("color: #333333; font-size: 12px;")
                except Exception:
                    pass

            # Apply selection style to current widget
            if widget is not None:
                widget.setStyleSheet("background: #ffe6cc; border-radius: 8px; border: 1px solid #ffb366;")
                if hasattr(widget, '_meta_label') and widget._meta_label is not None:
                    widget._meta_label.setStyleSheet("color: #333333; font-size: 12px;")

            # Save selection state
            self._selected_report_widget = widget
            self._selected_report_path = path
        except Exception:
            pass

    def load_metrics_into_parameters(self, report_path: str):
        import os, json
        # Add debug output to track when this is called
        print(f" load_metrics_into_parameters called for: {os.path.basename(report_path)}")
        
        reports_dir = str(data_file("reports"))
        metrics_path = os.path.join(reports_dir, "metrics.json")

        line = ""
        has_metrics = False
        try:
            # First try: look for JSON twin (ECG_Report_YYYYMMDD_HHMMSS.json next to .pdf)
            json_path = os.path.splitext(report_path)[0] + ".json"
            
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Extract metrics from JSON twin format
                metrics_data = data.get('metrics', {})
                patient = data.get('patient', {})
                user = data.get('user', {})
                
                if metrics_data:
                    m = {
                        'HR_bpm': metrics_data.get('heart_rate', '--'),
                        'PR_ms': metrics_data.get('pr_interval', '--'),
                        'QRS_ms': metrics_data.get('qrs_duration', '--'),
                        'QT_ms': metrics_data.get('qt_interval', '--'),
                        'QTc_ms': metrics_data.get('qtc_interval', '--'),
                        'ST_mV': metrics_data.get('ST_mV', metrics_data.get('st_interval', '--')),
                        'RR_ms': metrics_data.get('rr_interval', '--'),
                        'RV5_plus_SV1_mV': metrics_data.get('rv5_sv1', '--'),
                        'P_QRS_T_mm': ['--', '--', '--'],  # Placeholder
                        'QTCF': '--',
                        'RV5_SV1_mV': ['--', '--'],
                    }
            else:
                # Fallback: look in old-style metrics.json
                reports_dir = str(data_file("reports"))
                metrics_path = os.path.join(reports_dir, "metrics.json")
                
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        items = json.load(f) or []
                    report_abs = os.path.abspath(report_path)
                    report_name = os.path.basename(report_abs)
                    # Try absolute path match first
                    matches = [m for m in items if os.path.abspath(m.get("file", "")) == report_abs]
                    # Fallback: match by filename only
                    if not matches:
                        matches = [m for m in items if os.path.basename(m.get("file", "")) == report_name]
                    if matches:
                        m = matches[-1]
                    else:
                        raise ValueError("No matching report in metrics.json")
                else:
                    raise ValueError("No metrics.json or JSON twin found")
            
            if m:
                    hr   = m.get("HR_bpm", "--")
                    pr   = m.get("PR_ms", "--")
                    qrs  = m.get("QRS_ms", "--")
                    qt   = m.get("QT_ms", "--")
                    qtc  = m.get("QTc_ms", "--")
                    st   = m.get("ST_mV", m.get("ST_ms", "--"))
                    rr   = m.get("RR_ms", "--")
                    rv5p = m.get("RV5_plus_SV1_mV", "--")
                    pqt  = m.get("P_QRS_T_mm", ["--", "--", "--"])
                    qtcF = None
                    rv5s = m.get("RV5_SV1_mV", ["--", "--"])
                    if isinstance(rv5s, (list, tuple)) and len(rv5s) >= 2:
                        try:
                            rv5s = [f"{float(rv5s[0]):.3f}", f"{float(rv5s[1]):.3f}"]
                        except Exception:
                            pass

                    # Build vertical stacks (label top, value bottom) to match top metrics layout
                    metrics = [
                        ("HR", f"{hr} BPM"),
                        ("PR", f"{pr} ms"),
                        ("QRS Complex", f"{qrs} ms"),
                        ("QT", f"{qt} ms"),
                        ("QTc", f"{qtc} ms"),
                        ("ST", f"{st} mV"),
                        ("RR", f"{rr} ms"),
                        ("RV5+SV1", f"{rv5p} mV"),
                        # ("P/QRS/T", f"{pqt[0]}/{pqt[1]}/{pqt[2]}°"),
                        ("RV5/SV1", f"{rv5s[0]}/{rv5s[1]} mV"),
                    ]

                    # Render as a tight 11-column table (label top, value bottom) to ensure one-row layout
                    labels_row = []

                    values_row = []
                    for label, value in metrics:
                        labels_row.append(
                            f"<td style='width:8%; padding:6px 82px; text-align:center; white-space:nowrap; color:#ff6600; font-size:15px; font-weight:900; letter-spacing:0.2px;'>{label}</td>"
                        )
                        values_row.append(
                            f"<td style='width:8%; padding:4px 16px 8px; text-align:center; white-space:nowrap; color:#222222; font-size:14px; font-weight:800;'>{value}</td>"
                        )
                    table_html = (
                        "<table style='width:100%; border-collapse:collapse; table-layout:fixed;'>"
                        + "<tr>" + "".join(labels_row) + "</tr>"
                        + "<tr>" + "".join(values_row) + "</tr>"
                        + "</table>"
                    )
                    line = table_html
                    has_metrics = True
        except Exception as e:
            print(f"Failed to read metrics for report: {e}")
            import traceback
            traceback.print_exc()

        if hasattr(self, "parameters_text"):
            if not has_metrics:
                self.parameters_text.clear()
            elif line.startswith("<table") or line.startswith("<div"):
                self.parameters_text.setHtml(line)
            else:
                self.parameters_text.setPlainText(line)
        
        # Show METRICS panel only when we actually have metrics to display
        if hasattr(self, "metrics_notes_card"):
            if has_metrics:
                self.metrics_notes_card.show()
            else:
                self.metrics_notes_card.hide()

    def update_live_metrics_panel(self):
        """Update METRICS panel with live data from ECG test page (if not viewing a report)"""
        try:
            # If user is viewing a specific report, don't override with live data
            if getattr(self, '_viewing_report', False):
                return
            if self._is_ecg_frozen():
                return
            
            # Check if ECG test page exists and has data
            if not hasattr(self, 'ecg_test_page') or not self.ecg_test_page:
                return
            
            # Get current metrics from metric labels (top cards)
            if not hasattr(self, 'metric_labels') or not self.metric_labels:
                # Show "Waiting for ECG data..." if no metrics yet
                if hasattr(self, 'parameters_text'):
                    self.parameters_text.setHtml(
                        "<div style='text-align:center; padding:20px; color:#666; font-size:14px;'>"
                        "<b>LIVE METRICS</b><br><br>Waiting for ECG data...<br>"
                        "<small>Start ECG acquisition or demo mode to see real-time metrics</small>"
                        "</div>"
                    )
                return
            
            # Extract metrics from metric labels (with fallback to _dashboard_last_valid)
            hr = self.metric_labels.get('heart_rate', QLabel()).text().replace(' ', '').replace('BPM', '') or '--'
            pr = self.metric_labels.get('pr_interval', QLabel()).text().replace(' ', '').replace('ms', '') or '--'
            qrs = self.metric_labels.get('qrs_duration', QLabel()).text().replace(' ', '').replace('ms', '') or '--'
            qtc_raw = self.metric_labels.get('qtc_interval', QLabel()).text() or '--/--'

            if hasattr(self, '_dashboard_last_valid') and self._dashboard_last_valid:
                hr_num = int(float(hr)) if hr and hr not in ('--', '0', '00', '') else 0
                if hr_num > 0:
                    if (pr in ('--', '0', '00', '') or pr == '0') and 'pr_interval' in self._dashboard_last_valid:
                        pr = self._dashboard_last_valid['pr_interval'].replace(' ', '').replace('ms', '')
                    if (qrs in ('--', '0', '00', '') or qrs == '0') and 'qrs_duration' in self._dashboard_last_valid:
                        qrs = self._dashboard_last_valid['qrs_duration'].replace(' ', '').replace('ms', '')
                    if (qtc_raw in ('--', '--/--', '0', '0/0', '') or qtc_raw == '0') and 'qtc_interval' in self._dashboard_last_valid:
                        qtc_raw = self._dashboard_last_valid['qtc_interval']
            
            # Parse QT/QTc
            qt = '--'
            qtc = '--'
            if '/' in qtc_raw:
                parts = [part.strip() for part in qtc_raw.replace(' ms', '').split('/') if part.strip()]
                if len(parts) >= 1:
                    qt = parts[0]
                if len(parts) >= 2:
                    qtc = parts[1]
            else:
                qtc = qtc_raw
            
            # Calculate RR from HR
            try:
                hr_val = float(hr) if hr != '--' else 0
                rr = int(60000 / hr_val) if hr_val > 0 else '--'
            except:
                rr = '--'
            
            # Get wave amplitudes from ECG test page if available
            rv5_sv1_sum = '--'
            p_qrs_t = '--/--/--'
            rv5_sv1 = '--/--'
            
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                try:
                    # Calculate wave amplitudes in real-time
                    if hasattr(self.ecg_test_page, 'calculate_wave_amplitudes'):
                        wave_amps = self.ecg_test_page.calculate_wave_amplitudes()
                        if wave_amps:
                            p_amp = wave_amps.get('p_amp', 0.0)
                            qrs_amp = wave_amps.get('qrs_amp', 0.0)
                            t_amp = wave_amps.get('t_amp', 0.0)
                            rv5 = wave_amps.get('rv5', 0.0)
                            sv1 = wave_amps.get('sv1', 0.0)
                            
                            # Convert to display format
                            rv5_sv1_sum = f"{(rv5 - abs(sv1)):.3f}" if (rv5 - abs(sv1)) > 0 else '--'
                            p_qrs_t = f"{p_amp:.2f}/{qrs_amp:.2f}/{t_amp:.2f}" if (p_amp + qrs_amp + t_amp) > 0 else '--/--/--'
                            rv5_sv1 = f"{rv5:.2f}/{sv1:.2f}" if (rv5 - abs(sv1)) > 0 else '--/--'
                except Exception as e:
                    print(f"Error calculating wave amplitudes for dashboard: {e}")
            
            # Build metrics table
            metrics = [
                ("HR", f"{hr} BPM" if hr != '--' else '--'),
                ("PR", f"{pr} ms" if pr != '--' else '--'),
                ("QRS Complex", f"{qrs} ms" if qrs != '--' else '--'),
                ("QT", f"{qt} ms" if qt != '--' else '--'),
                ("QTc", f"{qtc} ms" if qtc != '--' else '--'),
                ("RR", f"{rr} ms" if rr != '--' else '--'),
                ("RV5+SV1", f"{rv5_sv1_sum} mV"),
                # ("P/QRS/T", f"{p_qrs_t}°"),
                ("RV5/SV1", f"{rv5_sv1} mV"),
            ]
            
            # Render as HTML table with "LIVE" badge
            labels_row = []
            values_row = []
            for label, value in metrics:
                labels_row.append(
                    f"<td style='width:8%; padding:6px 82px; text-align:center; white-space:nowrap; color:#ff6600; font-size:15px; font-weight:900; letter-spacing:0.2px;'>{label}</td>"
                )
                values_row.append(
                    f"<td style='width:8%; padding:4px 16px 8px; text-align:center; white-space:nowrap; color:#222222; font-size:14px; font-weight:800;'>{value}</td>"
                )
            
            # Add LIVE indicator
            live_badge = (
                "<div style='text-align:right; padding:4px 8px; font-size:11px;'>"
                "<span style='background:#4CAF50; color:white; padding:2px 8px; border-radius:4px; font-weight:bold;'>"
                "● LIVE</span>"
                "</div>"
            )
            
            table_html = (
                live_badge +
                "<table style='width:100%; border-collapse:collapse; table-layout:fixed;'>"
                + "<tr>" + "".join(labels_row) + "</tr>"
                + "<tr>" + "".join(values_row) + "</tr>"
                + "</table>"
            )
            
            if hasattr(self, 'parameters_text'):
                self.parameters_text.setHtml(table_html)
                
        except Exception as e:
            print(f"Error updating live metrics panel: {e}")

    def is_ecg_active(self):
        """Return True if demo is ON or serial acquisition is running."""
        try:
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                # Demo mode active?
                if hasattr(self.ecg_test_page, 'demo_toggle') and self.ecg_test_page.demo_toggle.isChecked():
                    return True
                try:
                    t = getattr(self.ecg_test_page, 'timer', None)
                    if t is not None and t.isActive():
                        return True
                except Exception:
                    pass
                # Serial acquisition running?
                reader = getattr(self.ecg_test_page, 'serial_reader', None)
                if reader and getattr(reader, 'running', False):
                    return True
                # Allow metric updates even when demo/serial not running
                # This ensures dashboard values update from Lead 2 calculation
                return getattr(self, "device_connected", False)  # Only allow updates if device is connected
        except Exception:
            pass
        return True  # Default to True to ensure updates work

    def calculate_stable_rr_interval(self, ecg_signal, sampling_rate):
        """Calculate stabilized RR interval using multiple validation layers"""
        try:
            from scipy.signal import find_peaks
            import numpy as np
            
            if len(ecg_signal) < 1000:  # Need at least 2 seconds at 500Hz
                return None, None
            
            # Step 1: Apply gentle filtering for RR stability
            # Use measurement filter for accurate RR calculation
            try:
                from ecg.signal_paths import measurement_filter
                filtered_signal = measurement_filter(ecg_signal, sampling_rate)
            except ImportError:
                # Fallback: use simple filtering if measurement_filter not available
                filtered_signal = ecg_signal
            
            # Step 2: Multi-strategy peak detection for RR stability
            rr_values = []
            
            # Strategy A: Conservative (most stable)
            try:
                peaks_a, _ = find_peaks(
                    filtered_signal,
                    height=np.std(filtered_signal) * 0.4,
                    distance=int(0.4 * sampling_rate),  # 400ms minimum
                    prominence=np.std(filtered_signal) * 0.4
                )
                if len(peaks_a) >= 2:
                    rr_a = np.diff(peaks_a) * (1000.0 / sampling_rate)
                    # Strict RR filtering: 300-2000ms (30-200 BPM)
                    valid_a = rr_a[(rr_a >= 300) & (rr_a <= 2000)]
                    if len(valid_a) >= 3:  # Need at least 3 intervals
                        rr_values.extend(valid_a)
            except Exception as e:
                print(f" Strategy A failed: {e}")
            
            # Strategy B: Normal (moderate sensitivity)
            try:
                peaks_b, _ = find_peaks(
                    filtered_signal,
                    height=np.std(filtered_signal) * 0.3,
                    distance=int(0.3 * sampling_rate),  # 300ms minimum
                    prominence=np.std(filtered_signal) * 0.3
                )
                if len(peaks_b) >= 2:
                    rr_b = np.diff(peaks_b) * (1000.0 / sampling_rate)
                    # Moderate RR filtering: 250-3000ms (20-240 BPM)
                    valid_b = rr_b[(rr_b >= 250) & (rr_b <= 3000)]
                    if len(valid_b) >= 3:
                        rr_values.extend(valid_b)
            except Exception as e:
                print(f" Strategy B failed: {e}")
            
            # Step 3: RR interval validation and stabilization
            if len(rr_values) < 3:
                return None, None
            
            # Convert to numpy array
            rr_values = np.array(rr_values)
            
            # Step 4: Remove outliers using IQR method
            q25, q75 = np.percentile(rr_values, [25, 75])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            
            # Filter outliers
            clean_rr = rr_values[(rr_values >= lower_bound) & (rr_values <= upper_bound)]
            
            if len(clean_rr) < 2:
                return None, None
            
            # Step 5: Calculate stable RR using median with EMA smoothing
            median_rr = np.median(clean_rr)
            
            # Apply EMA smoothing if we have previous RR value
            if hasattr(self, '_last_stable_rr'):
                alpha = 0.3  # Smoothing factor
                smoothed_rr = alpha * median_rr + (1 - alpha) * self._last_stable_rr
                self._last_stable_rr = smoothed_rr
            else:
                smoothed_rr = median_rr
                self._last_stable_rr = median_rr
            
            # Step 6: Final validation
            if 300 <= smoothed_rr <= 2000:  # 30-200 BPM range
                # Calculate BPM from smoothed RR
                stable_bpm = 60000.0 / smoothed_rr
                
                # Additional BPM validation
                if 40 <= stable_bpm <= 200:
                    print(f"🔒 Stable RR: {smoothed_rr:.1f}ms → BPM: {stable_bpm:.1f}")
                    return smoothed_rr, stable_bpm
                else:
                    print(f"⚠️ BPM out of range: {stable_bpm:.1f}")
                    return None, None
            else:
                print(f"⚠️ RR out of range: {smoothed_rr:.1f}ms")
                return None, None
                
        except Exception as e:
            print(f"❌ Error calculating stable RR: {e}")
            return None, None
    
    def calculate_standard_ecg_metrics(self, bpm):
        """DEPRECATED: Calculate standard ECG metrics based on BPM using simplified formulas.
        
        ⚠️ This function uses simplified BPM-based formulas and is kept only as a fallback.
        Real calculations should use calculate_live_ecg_metrics() which calculates from actual ECG signal.
        """
        try:
            bpm = float(bpm)
            
            # Standard medical formulas based on heart rate
            
            # PR Interval: Normal range 120-200ms, inversely related to HR
            # Formula: PR = 180 - (BPM-60)*0.3 (simplified approximation)
            pr_interval = max(120, min(200, 180 - (bpm - 60) * 0.3))
            
            # 🔧 HR-dependent PR calibration offsets (from calibration guide)
            # These adjustments match the reference table values exactly
            if bpm >= 200:
                pr_interval -= 8.0  # High HR: reduce PR by 8ms
            elif bpm >= 190:
                pr_interval -= 6.0  # Reduce by 6ms
            elif bpm >= 180:
                pr_interval -= 6.0  # Reduce by 6ms
            elif bpm >= 170:
                pr_interval -= 4.0  # Reduce by 4ms
            elif bpm >= 150:
                pr_interval -= 5.0  # Reduce by 5ms
            elif bpm >= 120:
                pr_interval -= 5.0  # Reduce by 5ms
            elif bpm >= 70:
                pr_interval -= 7.0  # 70 BPM needs reduction by 7ms
            
            # Ensure PR stays within physiological limits
            pr_interval = max(80, min(200, pr_interval))
            
            # QRS Duration: Normal range 60-100ms, relatively stable
            # Formula: QRS = 80 + (BPM-60)*0.1 (slight variation with HR)
            qrs_duration = max(60, min(100, 80 + (bpm - 60) * 0.1))
            
            # 🔧 QRS Duration fine-tuning calibration (from calibration guide)
            # Minor threshold adjustment for exact reference table match
            # Most values are already within 1-2ms, this fine-tunes to within ±1ms
            if bpm >= 100:
                qrs_duration -= 1.0  # High HR: slight reduction
            elif bpm >= 80:
                qrs_duration -= 0.5  # Medium HR: minimal reduction
            elif bpm >= 60:
                qrs_duration += 0.0  # Normal HR: no adjustment needed
            else:
                qrs_duration += 1.0  # Low HR: slight increase
            
            # Ensure QRS stays within physiological limits
            qrs_duration = max(60, min(100, qrs_duration))
            
            # QT Interval: Normal range 300-440ms, inversely related to HR
            # Bazett's formula: QTc = QT / sqrt(RR), where RR = 60/BPM
            # Simplified: QT = 400 - (BPM-60)*0.8
            qt_interval = max(300, min(440, 400 - (bpm - 60) * 0.8))
            
            # QT Interval calibration (from calibration guide)
            # According to reference table analysis, QT is already correct
            # Adding minimal verification adjustments for perfect match
            if bpm >= 200:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 180:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 160:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 140:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 120:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 100:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 80:
                qt_interval += 0.0  # Already perfect
            elif bpm >= 60:
                qt_interval += 0.0  # Already perfect
            else:
                qt_interval += 0.0  # Already perfect
            
            # QTc (corrected QT): Using Bazett's formula
            rr_interval = 60000 / bpm  # RR in milliseconds
            qtc_bazett = qt_interval / ((rr_interval / 1000) ** 0.5)
            qtc_bazett = max(350, min(450, qtc_bazett))
            
            # 🔧 QTc verification (automatically correct if QT is correct)
            # Since QTc uses Bazett's formula: QTc = QT / sqrt(RR)
            # If QT is correct, QTc will automatically be correct
            # Adding range validation for safety
            qtc_bazett = max(200, min(600, qtc_bazett))  # Extended safety range (200-600ms)
            
            # P Duration: Normal range 60-120ms, relatively stable
            # Standard P wave duration is typically 80-100ms, slight variation with HR
            p_duration = max(60, min(120, 80 + (bpm - 60) * 0.1))
            
            return {
                'heart_rate': int(round(bpm)),
                'pr_interval': int(round(pr_interval)),
                'qrs_duration': int(round(qrs_duration)),
                'qt_interval': int(round(qt_interval)),
                'qtc_interval': f"{int(round(qt_interval))}/{int(round(qtc_bazett))}",
                'p_duration': int(round(p_duration))
            }
            
        except Exception as e:
            print(f"Error calculating standard ECG metrics: {e}")
            return None
    
    def calculate_live_ecg_metrics(self, ecg_signal, sampling_rate=None):
        """Calculate live ECG metrics from Lead 2 data - ADAPTIVE for 40-300 BPM
        
        CRITICAL: Uses actual sampling rate from ECG test page for accurate BPM calculation.
        On Windows, sampling rate may be 80 Hz (not 500 Hz), so we must detect it correctly.
        """
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            import time

            # Ensure we have enough data
            if len(ecg_signal) < 200:
                return {}
            
            # CRITICAL: Get actual sampling rate from ECG test page
            # Use same fallback as ECG test page (250 Hz) for consistency
            import platform
            is_windows = platform.system() == 'Windows'
            platform_tag = "[Windows]" if is_windows else "[macOS/Linux]"
            
            fs = 500.0  # Hardware stream rate for live acquisition
            demo_mode_active = bool(
                hasattr(self, 'ecg_test_page')
                and self.ecg_test_page
                and hasattr(self.ecg_test_page, 'demo_toggle')
                and self.ecg_test_page.demo_toggle.isChecked()
            )

            if sampling_rate and sampling_rate > 10:
                fs = float(sampling_rate)
            elif demo_mode_active and hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                try:
                    if hasattr(self.ecg_test_page, 'sampler') and hasattr(self.ecg_test_page.sampler, 'sampling_rate'):
                        if self.ecg_test_page.sampler.sampling_rate > 10:
                            fs = float(self.ecg_test_page.sampler.sampling_rate)
                    elif hasattr(self.ecg_test_page, 'sampling_rate') and self.ecg_test_page.sampling_rate > 10:
                        fs = float(self.ecg_test_page.sampling_rate)
                except Exception as e:
                    pass
            
            # Enhanced debugging with platform detection
            if not hasattr(self, '_calc_count'):
                self._calc_count = 0
            self._calc_count += 1
            if self._calc_count <= 5:  # First 5 calculations (increased from 3)
                print(f" {platform_tag} BPM Calculation - Sampling rate: {fs:.1f} Hz, Signal length: {len(ecg_signal)} samples")
            
            # Windows-specific warnings
            if is_windows and fs == 500.0:
                if self._calc_count <= 5:
                    reason = "demo detector unavailable" if demo_mode_active else "live hardware locked to configured 500.0 Hz"
                    print(f" {platform_tag} Using 500.0 Hz ({reason})")
            
            # Validation
            if fs <= 0 or not np.isfinite(fs):
                if is_windows:
                    print(f" {platform_tag} Invalid sampling rate detected: {fs}, using fallback 500.0 Hz")
                fs = 500.0  # Fallback
            
            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz)
            nyquist = fs / 2
            low = 0.5 / nyquist
            high = 40 / nyquist
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, ecg_signal)
            
            # SMART ADAPTIVE PEAK DETECTION (40-300 BPM with BPM-based selection)
            # Run multiple detections and choose based on CALCULATED BPM consistency
            height_threshold = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)
            prominence_threshold = np.std(filtered_signal) * 0.4
            
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
                # Accept RR intervals from 200–6000 ms (300–10 BPM) - changed from 2000 to 6000 to allow 10 BPM
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
                # Accept RR intervals from 200–6000 ms (300–10 BPM) - changed from 2000 to 6000 to allow 10 BPM
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
                # Accept RR intervals from 200–6000 ms (300–10 BPM) - changed from 2000 to 6000 to allow 10 BPM
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
            else:
                # Fallback - use conservative distance to handle low BPM (10-40 BPM)
                peaks, _ = find_peaks(
                    filtered_signal,
                    height=height_threshold,
                    distance=int(0.4 * fs),  # 400ms - prevents false peaks, allows 10-300 BPM via RR filtering
                    prominence=prominence_threshold
                )
            
            metrics = {}
            
            # Calculate Heart Rate (instantaneous, per-beat)
            if len(peaks) >= 2:
                # Calculate R-R intervals in milliseconds
                # CRITICAL: Use correct sampling rate (fs) for accurate BPM calculation
                rr_intervals_ms = np.diff(peaks) * (1000.0 / fs)
                
                # Filter physiologically reasonable intervals (200-6000 ms)
                # 200 ms = 300 BPM (max), 6000 ms = 10 BPM (min)
                # Changed from 120 to 200 to match ECG test page and reduce noise
                min_rr_ms = 200
                max_rr_ms = 6000
                valid_intervals = rr_intervals_ms[(rr_intervals_ms >= min_rr_ms) & (rr_intervals_ms <= max_rr_ms)]
                
                if len(valid_intervals) > 0:
                    # Calculate heart rate from median R-R interval (more stable than instantaneous)
                    median_rr = np.median(valid_intervals)
                    heart_rate = 60000 / median_rr  # Convert to BPM

                    print(" Dashboard Heart Rate", heart_rate)
                    
                    # Ensure reasonable range (10-300 BPM)
                    heart_rate = max(10, min(300, heart_rate))

                    # Focus-switch / heavy-work stabilizer:
                    # if UI callbacks were delayed (e.g. app switch, report generation),
                    # do not allow one noisy frame to jump far away from the prior stable BPM.
                    now_ts = time.time()
                    last_metrics_ts = getattr(self, '_dashboard_last_metrics_ts', None)
                    self._dashboard_last_metrics_ts = now_ts
                    if last_metrics_ts is not None and (now_ts - last_metrics_ts) > 2.0:
                        self._dashboard_resume_grace_until = max(
                            getattr(self, '_dashboard_resume_grace_until', 0.0),
                            now_ts + 2.0,
                        )

                    prev_bpm = getattr(self, '_dashboard_bpm_ema', None)
                    if prev_bpm is not None:
                        grace_until = getattr(self, '_dashboard_resume_grace_until', 0.0)
                        max_jump_bpm = 8.0 if now_ts < grace_until else 18.0
                        bpm_delta = heart_rate - prev_bpm
                        if abs(bpm_delta) > max_jump_bpm:
                            clamped = prev_bpm + np.sign(bpm_delta) * max_jump_bpm
                            print(
                                f" Dashboard BPM jump clamp: raw={heart_rate:.1f}, "
                                f"prev={prev_bpm:.1f}, clamped={clamped:.1f}"
                            )
                            heart_rate = clamped
                    
                    # STABLE BPM WITH EXPONENTIAL MOVING AVERAGE (EMA) - Clinical Standard
                    # EMA provides stability while responding to genuine changes
                    # Alpha = 0.1 gives ~40 second stabilization (updates every 1 second)
                    if not hasattr(self, '_dashboard_bpm_ema'):
                        self._dashboard_bpm_ema = heart_rate  # Initialize with first reading
                        self._dashboard_bpm_alpha = 0.1  # Smoothing factor (0.1 = 40s stabilization)
                        print(f" Dashboard BPM EMA initialized with: {heart_rate}")  # Debug
                    else:
                        # Apply EMA: new_EMA = alpha * new_value + (1 - alpha) * old_EMA
                        self._dashboard_bpm_ema = self._dashboard_bpm_alpha * heart_rate + (1 - self._dashboard_bpm_alpha) * self._dashboard_bpm_ema
                        print(f" Dashboard BPM EMA updated: raw={heart_rate}, ema={self._dashboard_bpm_ema}")  # Debug
                    
                    # Use EMA value for display (stable and accurate)
                    smoothed_bpm = int(round(self._dashboard_bpm_ema))
                    print(f" Dashboard BPM final value: {smoothed_bpm}")  # Debug
                    
                    # Store for next iteration
                    self._last_stable_dashboard_bpm = smoothed_bpm
                    
                    # Add heart rate to metrics
                    metrics['heart_rate'] = smoothed_bpm
                else:
                    metrics['heart_rate'] = 0
            else:
                metrics['heart_rate'] = 0
            
            # ✅ REAL CALCULATIONS: Calculate PR, QRS, P, QT, QTC from actual ECG signal using clinical formulas
            # Uses 0.05-150 Hz measurement channel with median beat and clinical-grade detection methods
            # This replaces reference value lookups with real-time calculations from the signal
            
            try:
                # Import clinical measurement functions (real formulas from reference software)
                try:
                    from ecg.clinical_measurements import (
                        build_median_beat, get_tp_baseline, measure_pr_from_median_beat,
                        measure_qrs_duration_from_median_beat, measure_qt_from_median_beat,
                        measure_p_duration_from_median_beat
                    )
                except ImportError:
                    # Try alternative import path
                    import sys
                    import os
                    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if src_dir not in sys.path:
                        sys.path.insert(0, src_dir)
                    from ecg.clinical_measurements import (
                        build_median_beat, get_tp_baseline, measure_pr_from_median_beat,
                        measure_qrs_duration_from_median_beat, measure_qt_from_median_beat,
                        measure_p_duration_from_median_beat
                    )
                
                # Need at least 8 beats for median beat calculation (GE/Philips standard)
                if len(peaks) >= 8:
                    # Build median beat from Lead II signal (requires ≥8 beats)
                    time_axis, median_beat = build_median_beat(ecg_signal, peaks, fs, min_beats=8)
                    
                    if median_beat is not None and time_axis is not None:
                        # Get TP baseline for accurate measurements
                        r_mid = peaks[len(peaks) // 2]
                        prev_r_idx = peaks[len(peaks) // 2 - 1] if len(peaks) > 1 else None
                        tp_baseline = get_tp_baseline(ecg_signal, r_mid, fs, prev_r_peak_idx=prev_r_idx)
                        
                        # Calculate RR interval for QTC calculation
                        valid_rr = np.array([], dtype=float)
                        if len(peaks) >= 2:
                            rr_intervals_ms = np.diff(peaks) * (1000.0 / fs)
                            valid_rr = rr_intervals_ms[(rr_intervals_ms >= 200) & (rr_intervals_ms <= 6000)]
                            rr_ms = np.median(valid_rr) if len(valid_rr) > 0 else 600.0
                        else:
                            rr_ms = 600.0
                        
                        # Calculate PR Interval from median beat (real formula)
                        pr_val = measure_pr_from_median_beat(median_beat, time_axis, fs, tp_baseline)
                        if pr_val is None or pr_val <= 0:
                            pr_val = 0
                        
                        # PR already arrives stabilized from the ECG pipeline.
                        # Keep the live dashboard aligned with the measured value
                        # instead of introducing another lagging smoothing layer.
                        metrics['pr_interval'] = int(round(pr_val))
                        
                        # Calculate QRS Duration from median beat (real formula)
                        qrs_val = measure_qrs_duration_from_median_beat(median_beat, time_axis, fs, tp_baseline)
                        if qrs_val is None or qrs_val <= 0:
                            qrs_val = 0
                        
                        # Do not smooth QRS: BBB detection depends on the true
                        # measured width, and EMA can keep LBBB stuck near 117 ms.
                        metrics['qrs_duration'] = int(round(qrs_val))
                        
                        # Calculate P Duration from median beat (real formula)
                        p_duration = measure_p_duration_from_median_beat(median_beat, time_axis, fs, tp_baseline)
                        if p_duration is None or p_duration <= 0:
                            p_duration = 0
                        # Store P duration in both p_duration and st_interval (st_interval label shows P)
                        p_duration_int = int(round(p_duration))
                        metrics['p_duration'] = p_duration_int
                        metrics['p_duration'] = p_duration_int
                        
                        # Calculate QT Interval from median beat (real formula)
                        qt_val = measure_qt_from_median_beat(median_beat, time_axis, fs, tp_baseline, rr_ms=rr_ms)
                        if qt_val is None or qt_val <= 0:
                            qt_val = 0
                        
                        metrics['qt_interval'] = int(round(qt_val))
                        
                        # Calculate QTc using Bazett's formula: QTc = QT / sqrt(RR_sec)
                        if qt_val > 0 and rr_ms > 0:
                            rr_sec = rr_ms / 1000.0
                            qtc_val = qt_val / (rr_sec ** 0.5)
                            # Validate QTC range (200-600 ms)
                            qtc_val = max(200, min(600, qtc_val))
                        else:
                            qtc_val = 0
                        
                        # Format QT/QTc display
                        qt_int = int(round(qt_val)) if qt_val > 0 else 0
                        qtc_int = int(round(qtc_val)) if qtc_val > 0 else 0
                        
                        if qt_int > 0 and qtc_int > 0:
                            metrics['qtc_interval'] = f"{qt_int}/{qtc_int}"
                        elif qtc_int > 0:
                            metrics['qtc_interval'] = str(qtc_int)
                        else:
                            metrics['qtc_interval'] = "0"
                    else:
                        # Fallback if median beat cannot be built
                        metrics['pr_interval'] = 0
                        metrics['qrs_duration'] = 0
                        metrics['qt_interval'] = 0
                        metrics['qtc_interval'] = "0"
                        metrics['p_duration'] = 0
                        metrics['st_interval'] = "0"  # P duration = 0
                else:
                    # Not enough beats for median beat calculation
                    metrics['pr_interval'] = 0
                    metrics['qrs_duration'] = 0
                    metrics['qt_interval'] = 0
                    metrics['qtc_interval'] = "0"
                    metrics['p_duration'] = 0
                    metrics['st_interval'] = "0"  # P duration = 0
                    
            except ImportError as e:
                print(f" ⚠️ Clinical measurement functions not available: {e}")
                # Fallback if clinical measurement functions not available
                metrics['pr_interval'] = 0
                metrics['qrs_duration'] = 0
                metrics['qt_interval'] = 0
                metrics['qtc_interval'] = "0"
                metrics['p_duration'] = 0
                metrics['st_interval'] = "0"  # P duration = 0
            except Exception as e:
                print(f" ⚠️ Error calculating real ECG metrics: {e}")
                # Fallback on error
                metrics['pr_interval'] = 0
                metrics['qrs_duration'] = 0
                metrics['qt_interval'] = 0
                metrics['qtc_interval'] = "0"
                metrics['p_duration'] = 0
                metrics['st_interval'] = "0"  # P duration = 0
            
            return metrics
            
        except Exception:
            # Quietly fall back if metrics cannot be calculated
            return {}

    def update_dashboard_metrics_live(self, ecg_metrics):
        """Update dashboard metrics with live calculated values"""
        try:
            import time as _time
            # Throttle: reduced to 0.3s for much faster responsiveness (real-time)
            if not hasattr(self, '_last_metrics_update_ts'):
                self._last_metrics_update_ts = 0.0
            if _time.time() - self._last_metrics_update_ts < 0.3:
                return
            self._last_metrics_update_ts = _time.time()
            # If the dashboard timers are paused (e.g. context switch to HRV /
            # Hyperkalemia test), do not let live signal callbacks overwrite the
            # 0-reset that was applied on the context switch.
            if hasattr(self, 'metrics_timer') and not self.metrics_timer.isActive():
                return
            # Do not clear metrics if serial or acquisition is running or last valid metrics exist
            if not self.is_ecg_active() and not getattr(self, "device_connected", False) and not getattr(self, "_dashboard_last_valid", {}):
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText("--")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText("--")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText("--")
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("--")
                key = 'st_interval' if 'st_interval' in self.metric_labels else 'st_segment'
                if key in self.metric_labels:
                    self.metric_labels[key].setText("--")
                return
            
            # Allow updates in demo mode - display the values set by demo_manager
            # Update Heart Rate
            if 'heart_rate' in ecg_metrics:
                hr_val = ecg_metrics['heart_rate']
                try:
                    hr_int = int(round(float(hr_val))) if hr_val not in (None, "", "--") else 0
                except Exception:
                    hr_int = 0
                if hr_int > 0 and 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText(f"{hr_int} BPM")

            # ------- PR / QRS / P / QT/QTc: update at most every 1 second -------
            pr_val   = ecg_metrics.get('pr_interval')
            qrs_val  = ecg_metrics.get('qrs_duration')
            p_val    = ecg_metrics.get('st_interval')  # st_interval stores P duration
            qtc_raw  = ecg_metrics.get('qtc_interval')

            def _valid_num(v):
                if v in (None, "", "--", "--/--", "0", "0/0"):
                    return 0
                try:
                    val = int(round(float(v)))
                    return val if val > 0 else 0
                except Exception:
                    return 0

            now = _time.time()
            if not hasattr(self, '_last_interval_metrics_ts'):
                self._last_interval_metrics_ts = 0.0

            if now - self._last_interval_metrics_ts >= 1.0:
                p_v = _valid_num(pr_val)
                if p_v > 0 and 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText(f"{p_v} ms")

                q_v = _valid_num(qrs_val)
                if q_v > 0 and 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText(f"{q_v} ms")

                pv_v = _valid_num(p_val)
                if pv_v > 0 and 'st_interval' in self.metric_labels:
                    self.metric_labels['st_interval'].setText(f"{pv_v} ms")

                if qtc_raw not in (None, "", "--", "0", "0/0", "0 ms", "0/0 ms") and 'qtc_interval' in self.metric_labels:
                    qtc_text = str(qtc_raw).replace(" ms", "").strip()
                    if qtc_text and qtc_text not in ("0", "0/0"):
                        self.metric_labels['qtc_interval'].setText(qtc_text)

                self._last_interval_metrics_ts = now

                # Remember timestamp and snapshot
                self._last_interval_metrics_ts = now
                self._last_interval_metrics_values = interval_snapshot
            
            # Update Sampling Rate - Commented out
            # if 'sampling_rate' in ecg_metrics:
            #     self.metric_labels['sampling_rate'].setText(ecg_metrics['sampling_rate'])
            # Record last update time
            self._last_metrics_update_ts = _time.time()

            # Refresh the ECG interpretation immediately after the metrics change.
            # This keeps the dashboard conclusion in sync with the expanded 12-lead
            # view instead of waiting for a slower fallback timer.
            try:
                self.update_live_conclusion()
            except Exception:
                pass
            
            # Keep ECG test page metrics identical to dashboard
            try:
                self.sync_dashboard_metrics_to_ecg_page()
            except Exception:
                pass
            
        except Exception as e:
            print(f"Error updating live dashboard metrics: {e}")




    def _dashboard_has_respiration_baseline_drift(self, signal: np.ndarray, fs: float) -> bool:
        """
        Display-only detector for respiration/motion baseline drift for the dashboard mini chart.

        Uses the same concept as the 12-lead live view: when the <0.5 Hz component
        is a meaningful fraction of variance in the visible window, the trace can
        "float" during acquisition.
        """
        try:
            arr = np.asarray(signal, dtype=float)
            if arr.size < 250 or float(fs) <= 5.0:
                return False

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

            return (baseline_var / total_var) > 0.10
        except Exception:
            return False

    def update_ecg(self, frame):
        try:
            # Try to get data from ECG test page if available
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                if getattr(self.ecg_test_page, '_grid_frozen', False):
                    return [self.ecg_line]
                try:
                    # Validate ECG test page data structure
                    if not hasattr(self.ecg_test_page, 'data') or not self.ecg_test_page.data:
                        print(" ECG test page has no data")
                        return self._fallback_wave_update(frame)
                    
                    if len(self.ecg_test_page.data) <= 1:
                        print(" Insufficient ECG data (need Lead II)")
                        return self._fallback_wave_update(frame)
                    
                    # Get Lead II data from ECG test page (index 1 is Lead II)
                    lead_ii_data = self.ecg_test_page.data[1]
                    
                    # Validate Lead II data
                    if not isinstance(lead_ii_data, (list, np.ndarray)) or len(lead_ii_data) <= 10:
                        # print(" Invalid Lead II data")  # Commented out to suppress log messages
                        return self._fallback_wave_update(frame)
                    
                    # Convert to numpy array safely
                    try:
                        original_data = np.asarray(lead_ii_data, dtype=float)
                    except Exception as e:
                        print(f" Error converting Lead II data to array: {e}")
                        return self._fallback_wave_update(frame)

                    # Get actual sampling rate from ECG test page.
                    # Hardware default is 500 Hz; fall back only when no valid rate is reported.
                    actual_sampling_rate = 500  # Hardware default: 500 Hz
                    try:
                        if (hasattr(self.ecg_test_page, 'sampler') and
                                hasattr(self.ecg_test_page.sampler, 'sampling_rate') and
                                self.ecg_test_page.sampler.sampling_rate):
                            reported_rate = float(self.ecg_test_page.sampler.sampling_rate)
                            # Accept only physiologically sane rates (50–1000 Hz)
                            if 50.0 <= reported_rate <= 1000.0:
                                actual_sampling_rate = reported_rate
                            else:
                                print(f" Reported sampling rate {reported_rate} Hz out of range; using 500 Hz default")
                    except Exception as e:
                        print(f" Error getting sampling rate: {e}")

                    # Check for invalid values
                    if np.any(np.isnan(original_data)) or np.any(np.isinf(original_data)):
                        print(" Invalid values (NaN/Inf) in Lead II data")
                        return self._fallback_wave_update(frame)

                    # Choose settings source
                    settings_src = self.settings_manager
                    try:
                        if hasattr(self.ecg_test_page, 'settings_manager') and self.ecg_test_page.settings_manager is not None:
                            settings_src = self.ecg_test_page.settings_manager
                    except Exception:
                        settings_src = self.settings_manager

                    # Determine visible window based on wave speed (display feature only)
                    try:
                        wave_speed = float(settings_src.get_wave_speed())  # 12.5 / 25 / 50
                        if wave_speed <= 0:
                            wave_speed = 25.0
                    except Exception as e:
                        print(f" Error getting wave speed: {e}")
                        wave_speed = 25.0
                    
                    # Baseline window at 25 mm/s (diagnostic standard)
                    # Smaller window to reduce visible baseline drift
                    # 25 mm/s → 1.5 seconds visible
                    baseline_seconds = 1.5
                    # Scale time window with wave speed:
                    #   12.5 mm/s → 6 s, 25 mm/s → 3 s, 50 mm/s → 1.5 s
                    seconds_to_show = baseline_seconds * (25.0 / max(1e-6, wave_speed))
                    window_samples = int(max(50, min(len(original_data), seconds_to_show * actual_sampling_rate)))

                    # Mirror the stable 12-lead live view pipeline.
                    try:
                        raw_data = np.asarray(original_data, dtype=float)

                        # If acquisition restarts (buffer length drops), reset dashboard display state
                        # so stale filter memory/anchors don't create a visible drift jump.
                        try:
                            prev_len = int(getattr(self, "_dashboard_prev_lead2_len", 0) or 0)
                            curr_len = int(raw_data.size)
                            if prev_len and curr_len and curr_len < prev_len:
                                self._dashboard_lead2_baseline_anchor = None
                                if hasattr(self, "_dashboard_dft_hp_state"):
                                    self._dashboard_dft_hp_state = {}
                            self._dashboard_prev_lead2_len = curr_len
                        except Exception:
                            pass

                        non_zero_indices = np.where(raw_data != 0)[0]
                        if len(non_zero_indices) > 0:
                            recent_data = raw_data[int(non_zero_indices[0]):]
                        else:
                            recent_data = raw_data[-min(window_samples, len(raw_data)):] if len(raw_data) > 0 else raw_data

                        if len(recent_data) > window_samples:
                            data_slice = recent_data[-window_samples:]
                        else:
                            data_slice = recent_data

                        if len(data_slice) == 0:
                            data_slice = raw_data[-min(50, len(raw_data)):]
                        if len(data_slice) == 0:
                            return self._fallback_wave_update(frame)

                        filtered_slice = np.asarray(data_slice, dtype=float)

                        if not hasattr(self, '_dashboard_lead2_baseline_anchor'):
                            self._dashboard_lead2_baseline_anchor = None
                        try:
                            # Apply DFT (baseline HP) BEFORE the baseline anchor when enabled.
                            # Use streaming-safe path at 0.5 Hz to prevent acquisition "wobble"
                            # on sliding windows.
                            try:
                                from ecg.ecg_filters import apply_dft_filter, apply_baseline_wander_median_mean
                                dft_setting = str(settings_src.get_setting("filter_dft", "off")).strip()
                                if dft_setting and dft_setting not in ("off", ""):
                                    if str(dft_setting).strip() == "0.5":
                                        # Display fix: a 0.5 Hz high-pass introduces beat-synchronous baseline
                                        # "droop" between QRS complexes on short sliding windows.
                                        # Use the monitor-grade median+mean baseline removal instead so the
                                        # isoelectric line stays visually straight.
                                        filtered_slice = apply_baseline_wander_median_mean(
                                            filtered_slice, float(actual_sampling_rate)
                                        )
                                    else:
                                        filtered_slice = apply_dft_filter(filtered_slice, float(actual_sampling_rate), dft_setting)
                            except Exception:
                                pass

                            baseline_estimate = extract_low_frequency_baseline(filtered_slice, float(actual_sampling_rate))
                            if self._dashboard_lead2_baseline_anchor is None:
                                self._dashboard_lead2_baseline_anchor = baseline_estimate
                            # Dashboard chart is short-window and very sensitive to baseline "float".
                            # Track baseline a bit faster when DFT=0.5Hz is active.
                            baseline_alpha = 0.0005
                            try:
                                if str(settings_src.get_setting("filter_dft", "off")).strip() == "0.5":
                                    baseline_alpha = 0.002
                            except Exception:
                                pass
                            self._dashboard_lead2_baseline_anchor = (
                                (1.0 - baseline_alpha) * self._dashboard_lead2_baseline_anchor
                                + baseline_alpha * baseline_estimate
                            )
                            filtered_slice = filtered_slice - self._dashboard_lead2_baseline_anchor
                            current_dc = float(np.nanmean(filtered_slice)) if len(filtered_slice) > 0 else 0.0
                            if np.isfinite(current_dc):
                                filtered_slice = filtered_slice - current_dc
                        except Exception:
                            pass

                        try:
                            from ecg.ecg_filters import apply_emg_filter, apply_ac_filter
                            emg_setting = str(settings_src.get_setting("filter_emg", "25")).strip()
                            ac_setting = str(settings_src.get_setting("filter_ac", "50")).strip()
                            if emg_setting.lower() != "off" and len(filtered_slice) >= 10:
                                filtered_slice = apply_emg_filter(filtered_slice, float(actual_sampling_rate), emg_setting)
                            if ac_setting in ("50", "60") and len(filtered_slice) > 30:
                                filtered_slice = apply_ac_filter(filtered_slice, float(actual_sampling_rate), ac_setting)
                        except Exception:
                            pass

                        gain_factor = get_display_gain(settings_src.get_wave_gain())
                        scaled_data = filtered_slice * gain_factor
                        scaled_data = np.nan_to_num(scaled_data, copy=False)

                        sigma = float(getattr(self.ecg_test_page, 'SMOOTH_SIGMA', 0.8) or 0.8)
                        if len(scaled_data) > 5 and sigma > 0:
                            scaled_data = gaussian_filter1d(scaled_data, sigma=sigma)

                        interp_factor = int(getattr(self.ecg_test_page, 'INTERP_FACTOR', 4) or 4)
                        if len(scaled_data) > 1 and interp_factor > 1:
                            try:
                                x = np.arange(len(scaled_data), dtype=float)
                                f = interp1d(x, scaled_data, kind='cubic')
                                xi = np.linspace(0.0, float(len(scaled_data) - 1), len(scaled_data) * interp_factor)
                                scaled_data = np.asarray(f(xi), dtype=float)
                            except Exception:
                                x = np.arange(len(scaled_data), dtype=float)
                                xi = np.linspace(0.0, float(len(scaled_data) - 1), len(scaled_data) * interp_factor)
                                scaled_data = np.interp(xi, x, scaled_data)

                        edge_trim = int(0.5 * float(actual_sampling_rate))
                        if edge_trim > 0 and len(scaled_data) > 2 * edge_trim:
                            scaled_data = scaled_data[edge_trim:-edge_trim]

                        if len(scaled_data) == 0:
                            scaled_data = np.array([0.0], dtype=float)

                        display_y = np.clip(scaled_data + 2048.0, 0.0, 4095.0)
                        x_axis = np.linspace(0.0, seconds_to_show, len(display_y))

                        if np.any(np.isnan(display_y)) or np.any(np.isinf(display_y)):
                            print(" Invalid display data generated")
                            return self._fallback_wave_update(frame)

                        self.ecg_line.set_data(x_axis, display_y)
                        self.ecg_canvas.axes.set_autoscale_on(False)
                        new_xlim = (0.0, seconds_to_show)
                        if not hasattr(self, '_prev_xlim') or self._prev_xlim != new_xlim:
                            self.ecg_canvas.axes.set_xlim(*new_xlim)
                            self._prev_xlim = new_xlim
                        self.ecg_canvas.axes.set_ylim(0, 4096)

                    except Exception as e:
                        print(f" Error processing display data: {e}")
                        return self._fallback_wave_update(frame)

                    
                    # Calculate and update live ECG metrics using ORIGINAL data with SAME sampling rate
                    try:
                        # Use ECG test page's own calculation methods for consistency
                        if hasattr(self.ecg_test_page, 'calculate_ecg_metrics'):
                            self.ecg_test_page.calculate_ecg_metrics()
                        
                        # Get metrics from ECG test page to ensure synchronization
                        if hasattr(self.ecg_test_page, 'get_current_metrics'):
                            ecg_metrics = self.ecg_test_page.get_current_metrics()
                            # Debug: Print metrics to see what's being calculated
                            if hasattr(self, '_debug_counter'):
                                self._debug_counter += 1
                            else:
                                self._debug_counter = 1
                            if self._debug_counter % 50 == 0:  # Optimized: Print every 50 updates (was 10) - reduces console spam
                                print(f" Dashboard ECG metrics: {ecg_metrics}")
                            self.update_dashboard_metrics_from_ecg()
                        
                        # Calculate and update stress level and HRV (throttled to every 3 seconds for stability)
                        if not hasattr(self, '_last_stress_update'):
                            self._last_stress_update = 0
                        if time.time() - self._last_stress_update > 3:
                            self.update_stress_and_hrv(original_data, actual_sampling_rate)
                            self._last_stress_update = time.time()
                        
                    except Exception as e:
                        print(f" Error calculating ECG metrics: {e}")
                        # Continue with display even if metrics fail
                    
                    return [self.ecg_line]
                    
                except Exception as e:
                    print(f" Error getting data from ECG test page: {e}")
                    return self._fallback_wave_update(frame)
            
            # No ECG test page available
            return self._fallback_wave_update(frame)
            
        except Exception as e:
            print(f" Critical error in update_ecg: {e}")
            return self._fallback_wave_update(frame)
    
    def _fallback_wave_update(self, frame):
        """Fallback wave display (flat baseline) when live ECG data is not active"""
        try:
            # Show a clean flat baseline at 2048 (isoelectric line) when no live test is running
            self.ecg_y.fill(2048.0)
            self.ecg_line.set_data(self.ecg_x, self.ecg_y)
            # Ensure axes limits match fallback ranges
            self.ecg_canvas.axes.set_autoscale_on(False)
            new_xlim = (0.0, 2.0)
            if not hasattr(self, '_prev_xlim') or self._prev_xlim != new_xlim:
                self.ecg_canvas.axes.set_xlim(*new_xlim)
                self._prev_xlim = new_xlim
            self.ecg_canvas.axes.set_ylim(0, 4096)
            return [self.ecg_line]
        except Exception as e:
            print(f" Error in fallback wave update: {e}")
            return [self.ecg_line]
    
    def heart_rate_triple_click(self, event):
        """Handle triple-click on heart rate metric to open crash log dialog"""
        # Only count left mouse button clicks
        try:
            if hasattr(event, 'button') and event.button() != Qt.LeftButton:
                return
        except Exception:
            pass
        current_time = time.time()
        
        # Check if this is within 1 second of the last click
        if current_time - self.last_heart_rate_click_time < 1.0:
            self.heart_rate_click_count += 1
        else:
            self.heart_rate_click_count = 1
        
        self.last_heart_rate_click_time = current_time
        
        # Show click count in terminal
        print(f" Heart Rate Metric Click #{self.heart_rate_click_count}")
        
        # If triple-clicked, open crash log dialog
        if self.heart_rate_click_count >= 3:
            self.heart_rate_click_count = 0  # Reset counter
            print(" Triple-click detected! Opening diagnostic dialog...")
            self.crash_logger.log_info("Triple-click detected on heart rate metric", "TRIPLE_CLICK")
            self.open_crash_log_dialog()
        
        # Call original mousePressEvent if it exists
        if hasattr(event, 'original_mousePressEvent'):
            event.original_mousePressEvent(event)
    
    def open_crash_log_dialog(self):
        """Open the crash log diagnostic dialog"""
        try:
            dialog = CrashLogDialog(self.crash_logger, self)
            dialog.exec_()
        except Exception as e:
            self.crash_logger.log_error(f"Failed to open crash log dialog: {str(e)}", e, "DIALOG_ERROR")
            QMessageBox.critical(self, "Error", f"Failed to open diagnostic dialog: {str(e)}")

    def _is_ecg_frozen(self):
        """Return True when the 12-lead page is intentionally frozen."""
        try:
            return bool(
                hasattr(self, 'ecg_test_page')
                and self.ecg_test_page
                and getattr(self.ecg_test_page, '_grid_frozen', False)
            )
        except Exception:
            return False
    
    
    def update_ecg_metrics(self, intervals):
        import time as _time
        # Throttle: reduced to 0.3s for much faster responsiveness (real-time)
        if not hasattr(self, '_last_metrics_update_ts'):
            self._last_metrics_update_ts = 0.0
        if _time.time() - self._last_metrics_update_ts < 0.3:
            return
        if self._is_ecg_frozen():
            return
        # If the dashboard timers are paused (e.g. context switch to HRV /
        # Hyperkalemia test), do not let background signal callbacks overwrite
        # the 0-reset that was applied on the context switch.
        if hasattr(self, 'metrics_timer') and not self.metrics_timer.isActive():
            return
        if not hasattr(self, '_dashboard_last_valid'):
            self._dashboard_last_valid = {}

        def _parse_val(v):
            if v in (None, "", "--", "--/--", "0", "0/0"):
                return 0
            try:
                val = int(round(float(v)))
                return val if val > 0 else 0
            except Exception:
                return 0

        hr_val = _parse_val(intervals.get('Heart_Rate'))
        if hr_val > 0:
            self._dashboard_last_valid['heart_rate'] = f"{hr_val} BPM"
        if 'heart_rate' in self.metric_labels and 'heart_rate' in self._dashboard_last_valid:
            self.metric_labels['heart_rate'].setText(self._dashboard_last_valid['heart_rate'])

        pr_val = _parse_val(intervals.get('PR'))
        if pr_val > 0:
            self._dashboard_last_valid['pr_interval'] = f"{pr_val} ms"
        if 'pr_interval' in self.metric_labels and 'pr_interval' in self._dashboard_last_valid:
            self.metric_labels['pr_interval'].setText(self._dashboard_last_valid['pr_interval'])

        qrs_val = _parse_val(intervals.get('QRS'))
        if qrs_val > 0:
            self._dashboard_last_valid['qrs_duration'] = f"{qrs_val} ms"
        if 'qrs_duration' in self.metric_labels and 'qrs_duration' in self._dashboard_last_valid:
            self.metric_labels['qrs_duration'].setText(self._dashboard_last_valid['qrs_duration'])

        qtc_val = _parse_val(intervals.get('QTc'))
        qt_val = _parse_val(intervals.get('QT'))
        if qt_val > 0 and qtc_val > 0:
            self._dashboard_last_valid['qtc_interval'] = f"{qt_val}/{qtc_val} ms"
        elif qtc_val > 0:
            self._dashboard_last_valid['qtc_interval'] = f"{qtc_val} ms"
        elif 'QTc_interval' in intervals and intervals['QTc_interval'] not in (None, "", "--", "0", "0/0", "0 ms"):
            self._dashboard_last_valid['qtc_interval'] = f"{intervals['QTc_interval']}"

        if 'qtc_interval' in self.metric_labels and 'qtc_interval' in self._dashboard_last_valid:
            self.metric_labels['qtc_interval'].setText(self._dashboard_last_valid['qtc_interval'])

        # Record last update time
        self._last_metrics_update_ts = _time.time()
        
        # OPTIMIZED: Reduce sync frequency to prevent lag - only sync every 10th update
        if not hasattr(self, '_sync_throttle_count'):
            self._sync_throttle_count = 0
        self._sync_throttle_count += 1
        
        # Keep ECG test page metrics identical to dashboard (throttled)
        if self._sync_throttle_count % 10 == 0:  # Sync every 10th update instead of every update
            try:
                self.sync_dashboard_metrics_to_ecg_page()
            except Exception:
                pass
        # Also update the ECG test page theme if it exists
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
        
        # Update recommendations based on new metrics (works in demo mode too!)
        try:
            if hasattr(self, 'update_live_conclusion'):
                self.update_live_conclusion()
        except Exception:
            pass
    
    def sync_dashboard_metrics_to_ecg_page(self):
        """Force sync dashboard's current metric values to ECG test page for consistency"""
        try:
            if not hasattr(self, 'ecg_test_page') or not self.ecg_test_page:
                return
                
            if getattr(self.ecg_test_page, '_grid_frozen', False):
                return
                
            if not hasattr(self.ecg_test_page, 'metric_labels'):
                return
            
            # OPTIMIZED: Reduce sync frequency to prevent lag
            if not hasattr(self, '_sync_count'):
                self._sync_count = 0
            self._sync_count += 1
            
            # Only print sync message every 50th sync to reduce console spam
            if self._sync_count % 50 == 1:
                print(f"FORCE SYNC: Dashboard -> ECG Page")
                
            # Force sync metric values from dashboard to ECG test page ONLY if non-zero/valid
            if 'heart_rate' in self.metric_labels and 'heart_rate' in self.ecg_test_page.metric_labels:
                hr_text = self.metric_labels['heart_rate'].text()
                hr_value = hr_text.split()[0] if ' ' in hr_text else hr_text
                if hr_value and hr_value not in ('0', '00', '--', ''):
                    self.ecg_test_page.metric_labels['heart_rate'].setText(hr_value)
                
            if 'pr_interval' in self.metric_labels and 'pr_interval' in self.ecg_test_page.metric_labels:
                pr_text = self.metric_labels['pr_interval'].text()
                pr_value = pr_text.split()[0] if ' ' in pr_text else pr_text
                if pr_value and pr_value not in ('0', '00', '--', ''):
                    self.ecg_test_page.metric_labels['pr_interval'].setText(pr_value)
                
            if 'qrs_duration' in self.metric_labels and 'qrs_duration' in self.ecg_test_page.metric_labels:
                qrs_text = self.metric_labels['qrs_duration'].text()
                qrs_value = qrs_text.split()[0] if ' ' in qrs_text else qrs_text
                if qrs_value and qrs_value not in ('0', '00', '--', ''):
                    self.ecg_test_page.metric_labels['qrs_duration'].setText(qrs_value)
                
            if 'st_interval' in self.metric_labels and 'st_segment' in self.ecg_test_page.metric_labels:
                st_text = self.metric_labels['st_interval'].text()
                st_value = st_text.split()[0] if ' ' in st_text else st_text
                if st_value and st_value not in ('0', '00', '--', ''):
                    self.ecg_test_page.metric_labels['st_segment'].setText(st_value)
                
            # Handle qtc_interval - dashboard might have "286/369 ms" format
            if 'qtc_interval' in self.metric_labels and 'qtc_interval' in self.ecg_test_page.metric_labels:
                qtc_text = self.metric_labels['qtc_interval'].text()
                if '/' in qtc_text:
                    clean_qtc = qtc_text.replace('ms', '').strip()
                    if clean_qtc and clean_qtc not in ('0/0', '0', '--'):
                        self.ecg_test_page.metric_labels['qtc_interval'].setText(clean_qtc)
                    qtc_value = qtc_text.split()[0] if ' ' in qtc_text else qtc_text
                    self.ecg_test_page.metric_labels['qtc_interval'].setText(qtc_value)
                    if self._sync_count % 50 == 1:
                        print(f"  QT/QTc: {qtc_value}")
                else:
                    # Single value
                    qtc_value = qtc_text.split()[0] if ' ' in qtc_text else qtc_text
                    self.ecg_test_page.metric_labels['qtc_interval'].setText(qtc_value)
                    if self._sync_count % 50 == 1:
                        print(f"  QTc: {qtc_value}")
                    
            if self._sync_count % 50 == 1:
                print("✅ FORCE SYNC COMPLETED - Both pages now show identical values")
            
        except Exception as e:
            print(f"❌ Error syncing dashboard metrics to ECG test page: {e}")
    
    def periodic_sync_to_ecg_page(self):
        """Periodic sync to ensure both pages always show identical values"""
        try:
            # Only sync if ECG is active (either demo or real mode)
            if self.is_ecg_active():
                self.sync_dashboard_metrics_to_ecg_page()
        except Exception as e:
            print(f"❌ Periodic sync error: {e}")
    
    def reset_dashboard_metrics_to_zero(self, force=False):
        """Reset all dashboard metric labels to 0."""
        ecg_page = getattr(self, 'ecg_test_page', None) if hasattr(self, 'ecg_test_page') else None
        limb_conn = getattr(ecg_page, "_lead_connection_state", {}) if ecg_page else {}
        lead_off_latched = getattr(ecg_page, "_lead_off_latched", False) if ecg_page else False
        limb_active = (limb_conn.get('I', True) or limb_conn.get('II', True)) and not lead_off_latched
        ll_off = getattr(ecg_page, '_ll_disconnected', False) if ecg_page else False

        if not force and (limb_active and not ll_off) and hasattr(self, '_dashboard_last_valid') and self._dashboard_last_valid:
            live_hr = getattr(ecg_page, 'last_heart_rate', 0) or 0
            if live_hr > 0:
                return

        if hasattr(self, '_dashboard_last_valid'):
            self._dashboard_last_valid.clear()

        if hasattr(self, "metric_labels") and isinstance(self.metric_labels, dict):
            if 'heart_rate' in self.metric_labels:
                self.metric_labels['heart_rate'].setText("0 BPM")
            if 'pr_interval' in self.metric_labels:
                self.metric_labels['pr_interval'].setText("0 ms")
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setText("0 ms")
            key = 'st_interval' if 'st_interval' in self.metric_labels else 'st_segment'
            if key in self.metric_labels:
                self.metric_labels[key].setText("0 ms")
            if 'qt_interval' in self.metric_labels:
                self.metric_labels['qt_interval'].setText("0 ms")
            if 'qtc_interval' in self.metric_labels:
                self.metric_labels['qtc_interval'].setText("--")
        if hasattr(self, 'conclusion_box'):
            self.conclusion_box.setHtml("""
                <p style='color: #888; font-style: italic;'>
                Waiting for stable ECG data...<br><br>
                Metrics are being analyzed. Please wait a few seconds.
                </p>
            """)
        self._last_valid_conclusion_html = None

    def update_dashboard_metrics_from_ecg(self):
        try:
            import time as _time
            # Throttle: reduced to 0.3s for much faster responsiveness (real-time)
            if not hasattr(self, '_last_metrics_update_ts'):
                self._last_metrics_update_ts = 0.0
            if _time.time() - self._last_metrics_update_ts < 0.3:
                return
            self._last_metrics_update_ts = _time.time()
            if not self.is_ecg_active() or self._is_ecg_frozen():
                return

            if not hasattr(self, '_dashboard_last_valid'):
                self._dashboard_last_valid = {}

            if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'get_current_metrics'):
                ecg_metrics = self.ecg_test_page.get_current_metrics()
                hr_raw = str(ecg_metrics.get('heart_rate', '0')).strip()
                pr_raw = str(ecg_metrics.get('pr_interval', '0')).strip()
                qrs_raw = str(ecg_metrics.get('qrs_duration', '0')).strip()
                p_raw = str(ecg_metrics.get('st_interval', '0')).replace(' ms', '').replace('mV', '').strip()
                qt_raw = str(ecg_metrics.get('qt_interval', '0')).replace(' ms', '').strip()
                qtc_raw = str(ecg_metrics.get('qtc_interval', '0')).replace(' ms', '').strip()
                
                # Check validity
                def is_valid(val):
                    return val not in ('', '0', '00', '0.0', '--', 'None', 'None ms')

                # Only consider OFF if both primary limb leads (Lead I & Lead II) are disconnected.
                # Also check _ll_disconnected (fast per-packet flag) so LL removal is caught
                # immediately — _lead_connection_state['II'] has a 25-packet debounce lag
                # that lets "0 BPM" flash on the dashboard before the latch gate fires.
                limb_conn = getattr(self.ecg_test_page, "_lead_connection_state", {})
                limb_active = limb_conn.get('I', True) or limb_conn.get('II', True)
                ll_off = getattr(self.ecg_test_page, '_ll_disconnected', False)
                if getattr(self.ecg_test_page, "_lead_off_latched", False) or not limb_active or ll_off:
                    self.reset_dashboard_metrics_to_zero(force=True)
                    return

                hr_num = int(float(hr_raw)) if is_valid(hr_raw) else 0
                if hr_num > 0:
                    self._dashboard_last_valid['heart_rate'] = f"{hr_raw} BPM"
                    if 'heart_rate' in self.metric_labels:
                        self.metric_labels['heart_rate'].setText(self._dashboard_last_valid['heart_rate'])
                else:
                    self.metric_labels.get('heart_rate', QLabel()).setText("0 BPM")
                    self.reset_dashboard_metrics_to_zero(force=True)
                    return

                if is_valid(pr_raw) and hr_num > 0:
                    self._dashboard_last_valid['pr_interval'] = f"{pr_raw} ms"
                    if 'pr_interval' in self.metric_labels:
                        self.metric_labels['pr_interval'].setText(self._dashboard_last_valid['pr_interval'])
                else:
                    self._dashboard_last_valid.pop('pr_interval', None)
                    if 'pr_interval' in self.metric_labels:
                        self.metric_labels['pr_interval'].setText("0 ms")

                if is_valid(qrs_raw) and hr_num > 0:
                    self._dashboard_last_valid['qrs_duration'] = f"{qrs_raw} ms"
                    if 'qrs_duration' in self.metric_labels:
                        self.metric_labels['qrs_duration'].setText(self._dashboard_last_valid['qrs_duration'])
                else:
                    self._dashboard_last_valid.pop('qrs_duration', None)
                    if 'qrs_duration' in self.metric_labels:
                        self.metric_labels['qrs_duration'].setText("0 ms")

                key = 'st_interval' if 'st_interval' in self.metric_labels else 'st_segment'
                if is_valid(p_raw) and hr_num > 0:
                    self._dashboard_last_valid[key] = f"{p_raw} ms"
                    if key in self.metric_labels:
                        self.metric_labels[key].setText(self._dashboard_last_valid[key])
                else:
                    self._dashboard_last_valid.pop(key, None)
                    if key in self.metric_labels:
                        self.metric_labels[key].setText("0 ms")

                if is_valid(qt_raw) and is_valid(qtc_raw) and hr_num > 0:
                    self._dashboard_last_valid['qtc_interval'] = f"{qt_raw}/{qtc_raw} ms"
                elif is_valid(qtc_raw) and hr_num > 0:
                    self._dashboard_last_valid['qtc_interval'] = f"{qtc_raw} ms"
                else:
                    self._dashboard_last_valid.pop('qtc_interval', None)

                if 'qtc_interval' in self.metric_labels:
                    if 'qtc_interval' in self._dashboard_last_valid:
                        self.metric_labels['qtc_interval'].setText(self._dashboard_last_valid['qtc_interval'])
                    else:
                        self.metric_labels['qtc_interval'].setText("--")

                self._last_metrics_update_ts = _time.time()
                # Ensure dashboard interpretation updates as soon as live metrics arrive.
                # Previously this only refreshed after visiting the expanded lead view.
                try:
                    if hasattr(self, 'update_live_conclusion'):
                        self.update_live_conclusion()
                except Exception:
                    pass
                # This Prevents Jittering of BPM values in inner dashboard
                # try:
                #     self.sync_dashboard_metrics_to_ecg_page()
                # except Exception:
                #     pass
            else:
                default_metrics = self.calculate_standard_ecg_metrics(75)
                if default_metrics:
                    if 'heart_rate' in self.metric_labels:
                        self.metric_labels['heart_rate'].setText("--")
                    if 'pr_interval' in self.metric_labels:
                        self.metric_labels['pr_interval'].setText("--")
                    if 'qrs_duration' in self.metric_labels:
                        self.metric_labels['qrs_duration'].setText("--")
                    if 'qtc_interval' in self.metric_labels:
                        self.metric_labels['qtc_interval'].setText("--")
                    if 'st_interval' in self.metric_labels:
                        self.metric_labels['st_interval'].setText("--")
        except Exception as e:
            print(f" Error updating dashboard metrics from ECG: {e}")
            
    def pause_dashboard_timers(self):
        print("⏸️ Pausing dashboard timers and animations...")
        if hasattr(self, 'metrics_timer') and self.metrics_timer.isActive():
            self.metrics_timer.stop()
            print("  Dashboard metrics timer stopped.")
        if hasattr(self, 'heartbeat_timer') and self.heartbeat_timer.isActive():
            self.heartbeat_timer.stop()
            print("  Dashboard heartbeat timer stopped.")
        if hasattr(self, 'anim') and self.anim is not None:
            try:
                self.anim.pause()
                print("  Dashboard ECG animation paused.")
            except Exception as e:
                print(f"  Failed to pause dashboard ECG animation: {e}")

    def resume_dashboard_timers(self):
        print("▶️ Resuming dashboard timers and animations...")
        if hasattr(self, 'metrics_timer') and not self.metrics_timer.isActive():
            self.metrics_timer.start(2000 if is_low_spec_mode() else 1000)
            print("  Dashboard metrics timer started.")
        if hasattr(self, 'heartbeat_timer') and not self.heartbeat_timer.isActive():
            self.heartbeat_timer.start(150 if is_low_spec_mode() else 100)
            print("  Dashboard heartbeat timer started.")
        if hasattr(self, 'anim') and self.anim is not None:
            try:
                self.anim.resume()
                print("  Dashboard ECG animation resumed.")
            except Exception as e:
                print(f"  Failed to resume dashboard ECG animation: {e}")

    def _reset_metrics_on_context_switch(self):
        """Force-reset all dashboard metric labels to 0 and clear the ECG
        interpretation panel when the user switches context (e.g. opens the
        HRV or Hyperkalemia test from the dashboard).

        This is intentionally independent of the lead-detach logic — it fires
        unconditionally on a button click and does NOT change any lead-disconnect
        guards or the _lead_connection_state tracking.
        """
        # 1. Wipe the sticky last-valid cache so old values don't re-appear
        if hasattr(self, '_dashboard_last_valid'):
            self._dashboard_last_valid.clear()

        # 2. Reset every metric label to zero
        if hasattr(self, 'metric_labels') and isinstance(self.metric_labels, dict):
            if 'heart_rate' in self.metric_labels:
                self.metric_labels['heart_rate'].setText("0 BPM")
            if 'pr_interval' in self.metric_labels:
                self.metric_labels['pr_interval'].setText("0 ms")
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setText("0 ms")
            _st_key = 'st_interval' if 'st_interval' in self.metric_labels else 'st_segment'
            if _st_key in self.metric_labels:
                self.metric_labels[_st_key].setText("0 ms")
            if 'qt_interval' in self.metric_labels:
                self.metric_labels['qt_interval'].setText("0 ms")
            if 'qtc_interval' in self.metric_labels:
                self.metric_labels['qtc_interval'].setText("0 ms")

        # 3. Reset ECG interpretation panel
        if hasattr(self, 'conclusion_box'):
            self.conclusion_box.setHtml("""
                <p style='color: #888; font-style: italic;'>
                Waiting for stable ECG data...<br><br>
                Metrics are being analyzed. Please wait a few seconds.
                </p>
            """)

        # 4. Invalidate the cached conclusion so it is always regenerated fresh
        self._last_valid_conclusion_html = None
        print(" Dashboard metrics reset on context switch.")

    def on_page_changed(self, index):
        """Handle page stack widget changes. Pause timers if moving away from main dashboard page."""
        try:
            current_widget = self.page_stack.widget(index)
            if current_widget == self.dashboard_page:
                self.resume_dashboard_timers()
            else:
                self.pause_dashboard_timers()
        except Exception as e:
            print(f" Error in on_page_changed: {e}")
    
    def generate_pdf_report(self):
        """Generate ECG PDF report in a separate process."""
        from PyQt5.QtWidgets import QMessageBox
        import copy
        import datetime
        import os

        runner = getattr(self, "_pdf_runner", None)
        if runner is not None and runner.is_running():
            print("ℹ️ Report generation is already running. Please wait for it to finish.")
            return

        try:
            import time as _time
            self._dashboard_resume_grace_until = _time.time() + 1.5
        except Exception:
            pass

        def _extract_metric(label_key, default="0", strip_units=True):
            if not hasattr(self, "metric_labels") or label_key not in self.metric_labels:
                return default
            text = self.metric_labels[label_key].text().strip()
            if not text:
                return default
            if strip_units:
                text = text.replace(" (Unstable)", "")
                for unit in ("BPM", "bpm", "ms", "mV", "°"):
                    text = text.replace(unit, "")
            return text.strip() or default

        def _to_int(value, default=0):
            try:
                return int(float(str(value).split("/")[0].strip()))
            except Exception:
                return default

        def _to_float(value, default=0.0):
            try:
                return float(str(value).replace("mV", "").strip())
            except Exception:
                return default

        hr_text = _extract_metric("heart_rate", "0")
        pr_text = _extract_metric("pr_interval", "0")
        qrs_text = _extract_metric("qrs_duration", "0")
        qtc_label_text = _extract_metric("qtc_interval", "0/0", strip_units=False)
        st_label_text = _extract_metric("st_interval", "", strip_units=False) or _extract_metric(
            "st_segment", "0.0 mV", strip_units=False
        )

        qt_text, qtc_text = "0", "0"
        if "/" in qtc_label_text:
            parts = [p.strip().replace("ms", "").strip() for p in qtc_label_text.split("/") if p.strip()]
            if len(parts) >= 1:
                qt_text = parts[0]
            if len(parts) >= 2:
                qtc_text = parts[1]
        else:
            qtc_text = qtc_label_text.strip()

        hr_value = _to_int(hr_text, 0)
        pr_value = _to_int(pr_text, 0)
        qrs_value = _to_int(qrs_text, 0)
        qt_value = _to_int(qt_text, 0)
        qtc_value = _to_int(qtc_text, 0)
        st_value = _to_float(st_label_text, 0.0)
        qtcf_value = 0

        ecg_page = getattr(self, "ecg_test_page", None)
        if qt_value <= 0 and ecg_page and getattr(ecg_page, "_last_qt_ms", None):
            qt_value = int(ecg_page._last_qt_ms)
        if qtc_value <= 0 and ecg_page and getattr(ecg_page, "_last_qtc_ms", None):
            qtc_value = int(ecg_page._last_qtc_ms)
        if ecg_page and getattr(ecg_page, "_last_qtcf_ms", None):
            qtcf_value = int(ecg_page._last_qtcf_ms or 0)

        sampling_rate = 500.0
        if ecg_page and hasattr(ecg_page, "sampler") and hasattr(ecg_page.sampler, "sampling_rate"):
            try:
                sampling_rate = float(ecg_page.sampler.sampling_rate or 500.0)
            except Exception:
                sampling_rate = 500.0

        from ecg.ecg_report_generator import save_ecg_data_to_file, _normalize_report_conclusions
        from dashboard.history_window import append_history_entry
        from utils.pdf_process_runner import PDFProcessRunner

        report_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
        default_name = f"ECG_Report_12_1{serial_part}_{report_stamp}.pdf"
        downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        if not os.path.isdir(downloads_dir):
            downloads_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "reports"))
        os.makedirs(downloads_dir, exist_ok=True)
        filename = os.path.join(downloads_dir, default_name)

        # First, build patient profile and frozen data first, then save companion JSON with full schema
        frozen_patient = copy.deepcopy(
            resolve_patient_profile(
                explicit_patient=getattr(self, "patient_details", None),
                username=getattr(self, "username", "") or "",
                user_details=getattr(self, "user_details", {}) or {},
            )
        )

        # Now build the rest of the data for the report
        rr_ms = int(round(60000.0 / hr_value)) if hr_value > 0 else 0
        frozen_ecg_data = {
            "HR": hr_value,
            "beat": hr_value,
            "PR": pr_value,
            "QRS": qrs_value,
            "QT": qt_value,
            "QTc": qtc_value,
            "QTcF": qtcf_value,
            "QTc_Fridericia": qtcf_value,
            "ST": st_value,
            "RR_ms": rr_ms,
            "HR_max": hr_value,
            "HR_min": hr_value,
            "HR_avg": hr_value,
            "Heart_Rate": hr_value,
            "HR_bpm": hr_value,
            "user": {
                "name": getattr(self, "user_details", {}).get("full_name", getattr(self, "username", "") or ""),
                "phone": getattr(self, "user_details", {}).get("phone", ""),
            },
            "machine_serial": getattr(self, "user_details", {}).get("serial_id", "") or os.getenv("MACHINE_SERIAL_ID", ""),
        }

        # Now get frozen conclusions:
        frozen_conclusions = []
        try:
            if ecg_page:
                rhythm_text = None
                try:
                    if hasattr(ecg_page, "get_latest_rhythm_interpretation"):
                        rhythm_text = ecg_page.get_latest_rhythm_interpretation()
                except Exception:
                    rhythm_text = getattr(ecg_page, "_latest_rhythm_interpretation", None)
                if rhythm_text:
                    frozen_conclusions.append(rhythm_text)

                last_analysis = getattr(ecg_page, "_last_analysis", None) or {}
                if isinstance(last_analysis, dict):
                    frozen_conclusions.extend(last_analysis.get("arrhythmias", []) or [])

            frozen_conclusions = _normalize_report_conclusions(frozen_conclusions)
            if frozen_conclusions:
                import json
                conclusions_file = str(data_file("last_conclusions.json"))
                with open(conclusions_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "source": "dashboard_report_freeze",
                            "findings": frozen_conclusions,
                            "recommendations": [],
                        },
                        f,
                        indent=2,
                    )
                print(f" Frozen report conclusions: {frozen_conclusions}")
        except Exception as conclusion_err:
            print(f" Could not freeze report conclusions: {conclusion_err}")

        # Now, save companion JSON with the FULL structured schema instead of just raw ECG
        json_filename = os.path.abspath(os.path.splitext(filename)[0] + ".json")
        try:
            from utils.ecg_payload_builder import build_12lead_payload
            
            full_payload = build_12lead_payload(
                data=frozen_ecg_data,
                patient=frozen_patient,
                signup_details=getattr(self, "user_details", {}),
                ecg_test_page=ecg_page,
                source_report_file=filename,
                conclusions=frozen_conclusions,
                report_format="12_1"
            )
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(full_payload, f, indent=2, ensure_ascii=False)
            
            saved_ecg_data_file = json_filename
            print(f"✅ Saved FULL STRUCTURED JSON companion file to: {saved_ecg_data_file}")
            print(f"   Schema matches the expected format with patient details, device info, etc.")
        except Exception as payload_err:
            print(f"⚠️ Could not build full payload: {payload_err}")
            import traceback
            traceback.print_exc()
            # Fall back to raw ECG data if full payload fails
            saved_ecg_data_file = save_ecg_data_to_file(ecg_page, output_file=json_filename)
        
        # We need data to generate the report
        if not saved_ecg_data_file:
            QMessageBox.warning(self, "Report Failed", "No ECG data available to generate report.")
            return

        def _on_finished(fname):
            try:
                append_history_entry(
                    frozen_patient.copy(),
                    fname,
                    report_type="12 Lead",
                    username=getattr(self, "username", "") or "",
                    owner_full_name=(getattr(self, "user_details", {}) or {}).get("full_name") or getattr(self, "username", "") or "",
                )
            except Exception as hist_err:
                print(f" Failed to append ECG history: {hist_err}")
            try:
                self.refresh_recent_reports_ui()
            except Exception:
                pass
            print(f"✅ ECG Report generated successfully: {fname}")
            
            # Interactive prompt to open the generated PDF report
            from PyQt5.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self,
                "Report Generated",
                f"12-Lead ECG Report generated successfully!\n\nSaved at:\n{fname}\n\nWould you like to open it now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                try:
                    from utils.platform_compat import open_file
                    open_file(fname)
                except Exception as open_err:
                    QMessageBox.warning(self, "Error Opening PDF", f"Could not open PDF file:\n{open_err}")

        def _on_failed(err):
            print(f"❌ Failed to generate PDF: {err}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "PDF Generation Failed",
                f"An error occurred while generating the 12-Lead ECG PDF report:\n\n{err}"
            )

        self._pdf_runner = PDFProcessRunner(parent_widget=self)
        spawned = self._pdf_runner.start_ecg_report(
            filename=filename,
            frozen_data={
                "data": frozen_ecg_data,
                "ecg_data_file": saved_ecg_data_file,
                "conclusions": frozen_conclusions,
                "log_history": False,
                "username": getattr(self, "username", "") or "",
                "demo_mode": False,
            },
            patient=frozen_patient,
            sampling_rate=sampling_rate,
            on_success=_on_finished,
            on_failure=_on_failed,
        )

        if not spawned:
            print("❌ Failed to start ECG report background process.")



    def animate_heartbeat(self):
        """Animate heart image synchronized with live heart rate and play sound"""
        import time
        
        current_time = time.time() * 1000  # Convert to milliseconds
        current_hr = 0
        
        # Get current heart rate from metric card
        try:
            if hasattr(self, 'metric_labels') and 'heart_rate' in self.metric_labels:
                hr_text = self.metric_labels['heart_rate'].text()
                # Normalize text (e.g., "86 bpm", "86 BPM", "86")
                if hr_text:
                    cleaned = hr_text.replace("BPM", "").replace("bpm", "").strip()
                    # Treat non-numeric or placeholder values as "no data"
                    if cleaned.isdigit():
                        current_hr = int(cleaned)
                        self.current_heart_rate = current_hr
                        if self.current_heart_rate > 0:
                            # Calculate beat interval based on heart rate
                            self.beat_interval = 60000 / self.current_heart_rate  # Convert BPM to ms between beats
                    else:
                        # No valid heart rate available
                        self.current_heart_rate = 0
        except Exception as e:
            print(f" Error parsing heart rate: {e}")
            self.current_heart_rate = 0
        
        # If there is no valid heart data (HR < 10 bpm or 0 / '--'), 
        # do NOT play heartbeat sound and keep the heart static
        if not isinstance(self.current_heart_rate, (int, float)) or self.current_heart_rate < 10:
            # Optionally, you can keep a very subtle idle animation; here we freeze the icon
            return
        
        # Check if it's time for a heartbeat
        if current_time - self.last_beat_time >= self.beat_interval:
            self.last_beat_time = current_time
            
            # Play heartbeat sound with increased volume
            if self.heartbeat_sound and self.heartbeat_sound_enabled:
                try:
                    # Try to set volume if available (some Qt versions support this)
                    if hasattr(self.heartbeat_sound, 'setVolume'):
                        self.heartbeat_sound.setVolume(100)  # Maximum volume
                    self.heartbeat_sound.play()
                except Exception as e:
                    print(f" Error playing heartbeat sound: {e}")
            
            # Reset heartbeat phase for new beat
            self.heartbeat_phase = 0
        
        # Heartbeat effect: scale up and down based on phase
        # More pronounced beat when close to actual heartbeat
        time_since_beat = current_time - self.last_beat_time
        beat_progress = min(time_since_beat / self.beat_interval, 1.0)
        
        # Create a more realistic heartbeat pattern
        if beat_progress < 0.1:  # First 10% of cycle - sharp beat
            beat = 1 + 0.25 * math.sin(beat_progress * 10 * math.pi)
        elif beat_progress < 0.2:  # Next 10% - second beat
            beat = 1 + 0.15 * math.sin((beat_progress - 0.1) * 10 * math.pi)
        else:  # Rest of cycle - gradual return to normal
            beat = 1 + 0.05 * math.sin(self.heartbeat_phase)
        
        # Apply the beat effect
        size = int(self.heart_base_size * beat)
        self.heart_img.setPixmap(self.heart_pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # Update phase for smooth animation
        self.heartbeat_phase += 0.18
        if self.heartbeat_phase > 2 * math.pi:
            self.heartbeat_phase -= 2 * math.pi
    
    def create_heartbeat_sound(self):
        """Create a synthetic heartbeat sound if no sound file is available.

        This generates a louder, normalized 'lub-dub' sound so it is clearly audible
        across devices. The waveform is normalized to full 16‑bit range.
        """
        try:
            import wave
            import struct
            import math
            
            # Create a simple heartbeat sound (lub-dub pattern)
            sample_rate = 22050
            duration = 0.6  # seconds
            samples = int(sample_rate * duration)
            
            # Generate heartbeat sound data
            sound_data = []
            for i in range(samples):
                t = i / sample_rate
                
                # First beat (lub) - lower frequency (louder envelope)
                if t < 0.1:
                    freq1 = 80  # Hz
                    amplitude = 1.0 * math.sin(2 * math.pi * freq1 * t) * math.exp(-t * 12)
                # Second beat (dub) - higher frequency (louder envelope)
                elif 0.2 < t < 0.3:
                    freq2 = 120  # Hz
                    amplitude = 0.95 * math.sin(2 * math.pi * freq2 * (t - 0.2)) * math.exp(-(t - 0.2) * 12)
                else:
                    amplitude = 0
                
                sound_data.append(amplitude)

            # Normalize to full 16‑bit range
            peak = max(1e-6, max(abs(x) for x in sound_data))
            norm = 32767.0 / peak
            pcm_data = [int(max(-32767, min(32767, x * norm))) for x in sound_data]
            
            # Save as WAV file
            heartbeat_path = get_asset_path("heartbeat.wav")
            with wave.open(heartbeat_path, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(struct.pack('<' + 'h' * len(pcm_data), *pcm_data))
            
            # Load the created sound
            if QSound is not None:
                self.heartbeat_sound = QSound(heartbeat_path)
                print(f" Created synthetic heartbeat sound: {heartbeat_path}")
            else:
                self.heartbeat_sound = None
                print(f" QSound not available - heartbeat sound disabled")
            
        except Exception as e:
            print(f" Could not create heartbeat sound: {e}")
            self.heartbeat_sound = None

    def set_heartbeat_sound_enabled(self, enabled):
        """Enable or disable the audible heartbeat feedback."""
        desired_state = str(enabled).lower() in ("on", "true", "1", "yes")
        self.heartbeat_sound_enabled = desired_state
        if not desired_state and self.heartbeat_sound:
            try:
                self.heartbeat_sound.stop()
            except Exception as e:
                print(f" Unable to stop heartbeat sound: {e}")
        elif desired_state and self.heartbeat_sound is None and QSound is not None:
            # Ensure we have a usable sound source when the user enables heartbeat audio.
            try:
                self.create_heartbeat_sound()
            except Exception as e:
                print(f" Unable to create heartbeat sound: {e}")

    def tr(self, text):
        return translate_text(text, getattr(self, "current_language", "en"))

    def apply_language(self, language=None):
        if language:
            self.current_language = language
        translator = self.tr
        if hasattr(self, 'date_btn') and self.date_btn:
            self.date_btn.setText(translator("12-Lead ECG"))
        if hasattr(self, 'chatbot_btn') and self.chatbot_btn:
            self.chatbot_btn.setText(translator("AI Chatbot"))
        if hasattr(self, 'heart_label') and self.heart_label:
            self.heart_label.setText(translator("Live Heart Rate Overview"))
        if hasattr(self, 'stress_label') and self.stress_label:
            self.stress_label.setText(translator("Stress Level: --"))
        if hasattr(self, 'hrv_label') and self.hrv_label:
            self.hrv_label.setText(translator("Average Variability: --"))
        if hasattr(self, 'ecg_label') and self.ecg_label:
            self.ecg_label.setText(translator("ECG Recording"))
        if hasattr(self, 'visitors_label') and self.visitors_label:
            year = datetime.datetime.now().year
            self.visitors_label.setText(translator("Visitors - Last 6 Months ({year})").format(year=year))
        if hasattr(self, 'sign_btn') and self.sign_btn:
            self.sign_btn.setText(translator("Sign Out"))

    def on_settings_changed(self, key, value):
        """Handle global settings pushed from the ECG menu."""
        if key == "system_beat_vol":
            self.set_heartbeat_sound_enabled(value)
        elif key == "system_language":
            self.apply_language(value)

    def _compute_greeting(self) -> str:
        hour = datetime.datetime.now().hour
        if hour < 12:
            return "Good Morning"
        if hour < 18:
            return "Good Afternoon"
        return "Good Evening"

    def refresh_greeting_label(self) -> None:
        if not hasattr(self, "greet_label") or self.greet_label is None:
            return

        display_name = (self.user_details or {}).get("full_name") or self.username or "User"
        user_info_lines = [f"<span style='font-size:18pt;font-weight:bold;'>{self._compute_greeting()}, {display_name}</span>"]

        if self.user_details:
            details = []
            if self.user_details.get("age"):
                details.append(f"Age: {self.user_details.get('age')}")
            if self.user_details.get("gender"):
                details.append(f"Gender: {self.user_details.get('gender')}")
            if details:
                user_info_lines.append(f"<span style='color:#666; font-size:11pt;'>{' | '.join(details)}</span>")

        user_info_lines.append("<span style='color:#888;'>Welcome to your ECG dashboard</span>")
        self.greet_label.setText("<br>".join(user_info_lines))

    def show_doctor_profile_dialog(self) -> None:
        try:
            from dashboard.doctor_profile_dialog import DoctorProfileDialog
        except Exception as e:
            QMessageBox.warning(self, "Doctor Profile", f"Could not open Doctor Profile dialog: {e}")
            return

        dlg = DoctorProfileDialog(username=self.username or "", user_details=self.user_details or {}, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return

        updated = dlg.get_updated_user_details()
        if isinstance(updated, dict) and updated:
            self.user_details = dict(self.user_details or {})
            self.user_details.update(updated)
            self.refresh_greeting_label()

            # Keep other pages (ECG test, report generators) synced with the latest in-memory profile.
            try:
                if hasattr(self, "ecg_test_page") and self.ecg_test_page:
                    setattr(self.ecg_test_page, "user_details", dict(self.user_details))
            except Exception:
                pass
    def handle_sign(self):
        if self.sign_btn.text() == "Sign In":
            dialog = SignInDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                role, name = dialog.get_user_info()
                if not name.strip():
                    QMessageBox.warning(self, "Input Error", "Please enter your name.")
                    return
                # User label removed per request
                # self.user_label.setText(f"{name}\n{role}")
                self.sign_btn.setText("Sign Out")
        else:
            # User label removed per request
            # self.user_label.setText("Not signed in")
            self.sign_btn.setText("Sign In")
    def update_stress_and_hrv(self, ecg_signal, sampling_rate):
        """Calculate and update stress level and HRV from ECG data with smoothing"""
        try:
            from scipy.signal import find_peaks
            
            if len(ecg_signal) < 500:
                return
            
            # Find R-peaks
            peaks, _ = find_peaks(
                ecg_signal,
                height=np.mean(ecg_signal) + 0.5 * np.std(ecg_signal),
                distance=int(0.15 * sampling_rate)  # Reduced from 0.4 to 0.15 for high BPM (up to 360)
            )
            
            if len(peaks) >= 3:
                # Calculate R-R intervals in milliseconds
                rr_intervals = np.diff(peaks) * (1000 / sampling_rate)
                
                # Filter valid intervals (240-2000 ms) - 240ms = 250 BPM
                valid_rr = rr_intervals[(rr_intervals >= 240) & (rr_intervals <= 2000)]
                
                if len(valid_rr) >= 2:
                    # HRV: Standard deviation of R-R intervals (SDNN)
                    current_hrv_ms = np.std(valid_rr)
                    
                    # Initialize rolling average for HRV smoothing
                    if not hasattr(self, '_hrv_history'):
                        self._hrv_history = []
                    
                    # Add current HRV to history (keep last 5 values for smoothing)
                    self._hrv_history.append(current_hrv_ms)
                    if len(self._hrv_history) > 5:
                        self._hrv_history.pop(0)
                    
                    # Use smoothed HRV value
                    smoothed_hrv_ms = np.mean(self._hrv_history)
                    
                    # Store for conclusion generation
                    self._current_hrv = smoothed_hrv_ms
                    
                    # Stress level based on smoothed HRV
                    # Use dashboard's translation method
                    translator = self.tr
                    
                    if smoothed_hrv_ms > 100:
                        stress = translator("Low")
                        stress_color = "#27ae60"
                    elif smoothed_hrv_ms > 50:
                        stress = translator("Moderate")
                        stress_color = "#f39c12"
                    else:
                        stress = translator("High")
                        stress_color = "#e74c3c"
                    
                    # Update labels with translation
                    if hasattr(self, 'stress_label'):
                        stress_label_text = translator("Stress Level:")
                        self.stress_label.setText(f"{stress_label_text} {stress}")
                        self.stress_label.setStyleSheet(f"font-size: 13px; color: {stress_color}; font-weight: bold;")
                    
                    if hasattr(self, 'hrv_label'):
                        hrv_label_text = translator("Average Variability:")
                        self.hrv_label.setText(f"{hrv_label_text} {int(smoothed_hrv_ms)}ms")
                        self.hrv_label.setStyleSheet("font-size: 13px; color: #666;")
        except Exception as e:
            print(f" Error calculating stress/HRV: {e}")
    
    def update_live_conclusion(self):
        """Generate comprehensive personalized conclusion based on current ECG metrics with detailed BPM analysis"""
        if not getattr(self, "device_connected", False):
            return
        
        # Only reset interpretation when both primary limb leads are off.
        ecg_page = getattr(self, 'ecg_test_page', None)
        is_off = False
        if ecg_page:
            limb_conn = getattr(ecg_page, "_lead_connection_state", {})
            limb_active = limb_conn.get('I', True) or limb_conn.get('II', True)
            is_off = getattr(ecg_page, "_lead_off_latched", False) or not limb_active
        if is_off or not self.is_ecg_active() or self._is_ecg_frozen():
            if hasattr(self, '_last_valid_conclusion_html') and self._last_valid_conclusion_html and limb_active:
                if hasattr(self, 'conclusion_box'):
                    self.conclusion_box.setHtml(self._last_valid_conclusion_html)
                return
            if hasattr(self, 'conclusion_box'):
                self.conclusion_box.setHtml("""
                    <p style='color: #888; font-style: italic;'>
                    Waiting for stable ECG data...<br><br>
                    Metrics are being analyzed. Please wait a few seconds.
                    </p>
                """)
            return

        # ── NEW: Chest-leads (V1–V6) all disconnected ─────────────────────────
        # When every precordial lead is off the interpretation cannot be reliable
        # even though the limb leads (I, II) are still active.
        # Show a waiting message and invalidate the cached conclusion so that when
        # the chest leads are reconnected a fresh interpretation is generated
        # rather than the old (possibly stale) one being shown immediately.
        # NOTE: The metric labels (HR, PR, QRS, QTc) are intentionally left
        # unchanged — they are derived from limb leads and remain valid.
        if ecg_page:
            _lead_state = getattr(ecg_page, '_lead_connection_state', {})
            _chest_leads = ('V1', 'V2', 'V3', 'V4', 'V5', 'V6')
            _all_chest_off = bool(_lead_state) and all(
                not _lead_state.get(v, True) for v in _chest_leads
            )
            if _all_chest_off:
                if hasattr(self, 'conclusion_box'):
                    self.conclusion_box.setHtml("""
                        <p style='color: #888; font-style: italic;'>
                        Waiting for stable ECG data...<br><br>
                        Metrics are being analyzed. Please wait a few seconds.
                        </p>
                    """)
                # Invalidate cache so reconnection forces a fresh interpretation
                self._last_valid_conclusion_html = None
                return
        # ── END chest-leads guard ───────────────────────────────────────────────

        # ── All leads connected: always generate fresh interpretation ───────────
        # When all standard leads (I, II + V1–V6) are confirmed connected,
        # clear any stale cached conclusion so the panel never shows old HTML
        # after a lead-reconnect cycle (e.g., after a brief V1–V6 dropout).
        # The code falls through to the try: block below to build the interpretation.
        if ecg_page:
            _lead_state = getattr(ecg_page, '_lead_connection_state', {})
            _std_leads = ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
            _all_connected = bool(_lead_state) and all(
                _lead_state.get(lead, False) for lead in _std_leads
            )
            if _all_connected:
                # All leads online — invalidate cache so interpretation is always fresh
                self._last_valid_conclusion_html = None
        # ── Falls through to try: block to generate fresh conclusion ────────────

        try:
            findings = []
            recommendations = []
            additional_info = []


            
            rhythm_text = None
            ecg_page = getattr(self, 'ecg_test_page', None)
            # Keep rhythm interpretation updated even when the expanded lead view is never opened.
            # (Previously the cache often stayed at "Analyzing Rhythm..." until navigating away/back.)
            try:
                import time as _time
                now = _time.time()
                last_refresh = getattr(self, "_last_dashboard_rhythm_refresh_ts", 0.0) or 0.0
                cached = getattr(ecg_page, "_latest_rhythm_interpretation", None) if ecg_page else None
                needs_refresh = cached in (None, "", "Analyzing Rhythm...", "Detecting...")
                refresh_due = (now - last_refresh) >= 1.0  # sync with metrics_timer (1 sec)

                ecg_active = True
                try:
                    if hasattr(self, "is_ecg_active") and callable(self.is_ecg_active):
                        ecg_active = bool(self.is_ecg_active())
                except Exception:
                    ecg_active = True

                if ecg_page and ecg_active and (needs_refresh or refresh_due) and (now - last_refresh) >= 1.0 and hasattr(ecg_page, "update_latest_rhythm_interpretation"):
                    self._last_dashboard_rhythm_refresh_ts = now
                    ecg_page.update_latest_rhythm_interpretation()
            except Exception:
                pass

            try:
                if ecg_page and hasattr(ecg_page, 'get_latest_rhythm_interpretation'):
                    rhythm_text = ecg_page.get_latest_rhythm_interpretation()
                elif ecg_page:
                    rhythm_text = getattr(ecg_page, '_latest_rhythm_interpretation', None)
            except Exception:
                rhythm_text = None

            # Get current metrics
            hr_text = self.metric_labels.get('heart_rate', QLabel()).text()
            pr_text = self.metric_labels.get('pr_interval', QLabel()).text()
            qrs_text = self.metric_labels.get('qrs_duration', QLabel()).text()
            st_text = self.metric_labels.get('st_interval', QLabel()).text()
            qtc_text = self.metric_labels.get('qtc_interval', QLabel()).text()
            
            # Parse values
            try:
                hr = int(hr_text.replace(' BPM', '').replace(' bpm', '').strip()) if hr_text and hr_text != '00' else 0
            except:
                hr = 0

            if hr <= 0 or not self.is_ecg_active():
                if hasattr(self, 'conclusion_box'):
                    self.conclusion_box.setHtml("""
                        <p style='color: #888; font-style: italic;'>
                        Waiting for stable ECG data...<br><br>
                        Metrics are being analyzed. Please wait a few seconds.
                        </p>
                    """)
                return

            # Parse PR interval — this line was accidentally removed by commit dd17ee3
            # when it cleaned up the stale-cache restoration block. Without it, `pr`
            # is undefined and causes a silent NameError in every try/except below,
            # preventing any interpretation from being generated even at valid BPM.
            try:
                pr = int(pr_text.replace(' ms', '').strip()) if pr_text and pr_text not in ('0 ms', '0', '', '--') else 0
            except:
                pr = 0

            try:
                pr = int(pr_text.replace(' ms', '').strip()) if pr_text and pr_text not in ('', '0 ms', '0', '--') else 0
            except:
                pr = 0

            try:
                qrs = int(qrs_text.replace(' ms', '').strip()) if qrs_text and qrs_text != '0 ms' else 0
            except:
                qrs = 0
            
            # Parse QTC (can be "QT/QTC" format)
            qt = 0
            qtc = 0
            try:
                if qtc_text and '/' in qtc_text:
                    parts = qtc_text.split('/')
                    qt = int(parts[0].strip().replace(' ms', '')) if len(parts) > 0 else 0
                    qtc = int(parts[1].strip().replace(' ms', '')) if len(parts) > 1 else 0
                elif qtc_text:
                    qtc = int(qtc_text.strip().replace(' ms', ''))
            except:
                pass
            
            # ── LOGIC: If Normal Sinus Rhythm → show only NSR.
            # If NOT NSR → show rhythm (heart-based analysis) + PR interval status + QRS status.

            # ─── Determine rhythm status ────────────────────────────────────────
            rhythm_issue         = None
            is_normal_rhythm     = False
            rhythm_clean         = ""
            ignore_values        = {
                "",
                "Analyzing Rhythm...",
                "Detecting...",
                "Insufficient Data",
                "Insufficient data",
                "No specific arrhythmia detected.",
            }

            if rhythm_text:
                rhythm_clean = rhythm_text.strip()
                if rhythm_clean not in ignore_values:
                    abnormal_rhythm_keywords = (
                        "block", "fibrillation", "flutter", "tachycardia", "bradycardia",
                        "pvc", "pac", "asystole"
                    )
                    # NOTE: Wide QRS, Prolonged QTc, ST changes are MORPHOLOGICAL findings,
                    # not rhythm abnormalities — they coexist with Normal Sinus Rhythm.
                    is_normal_rhythm = any(
                        keyword in rhythm_clean.lower()
                        for keyword in ["normal sinus", "none detected", "sinus rhythm"]
                    ) and not any(keyword in rhythm_clean.lower() for keyword in abnormal_rhythm_keywords)
                    if not is_normal_rhythm:
                        rhythm_issue = rhythm_clean

            # ─── Determine PR interval status ───────────────────────────────────
            pr_status   = None  # None = normal / not measured
            pr_label    = ""
            if pr > 0:
                if pr > 200:
                    pr_status = "prolonged"
                    pr_label  = f"PR Interval: <b>{pr} ms</b> — <span style='color:#e67e22;'>Prolonged (Normal: 120–200 ms)</span>"
                elif pr < 120:
                    pr_status = "short"
                    pr_label  = f"PR Interval: <b>{pr} ms</b> — <span style='color:#e67e22;'>Short (Normal: 120–200 ms)</span>"
                else:
                    pr_label  = f"PR Interval: <b>{pr} ms</b> — <span style='color:#27ae60;'>Normal (120–200 ms)</span>"

            # ─── Determine QRS duration status ──────────────────────────────────
            qrs_status  = None  # None = normal / not measured
            qrs_label   = ""
            if qrs > 0:
                if qrs >= 120:
                    qrs_status = "wide"
                    qrs_label  = f"QRS Duration: <b>{qrs} ms</b> — <span style='color:#e74c3c;'>Wide (Normal: 60–120 ms)</span>"
                elif qrs < 60:
                    qrs_status = "narrow"
                    qrs_label  = f"QRS Duration: <b>{qrs} ms</b> — <span style='color:#e67e22;'>Narrow (Normal: 60–120 ms)</span>"
                elif qrs >= 110:
                    qrs_status = "borderline"
                    qrs_label  = f"QRS Duration: <b>{qrs} ms</b> — <span style='color:#f39c12;'>Borderline Wide (Normal: 60–120 ms)</span>"
                else:
                    qrs_label  = f"QRS Duration: <b>{qrs} ms</b> — <span style='color:#27ae60;'>Normal (60–120 ms)</span>"

            # ─── HR status ──────────────────────────────────────────────────────
            hr_label = ""
            hr_abnormal = False
            if hr >= 10 and hr <= 300:
                if hr > 100:
                    hr_abnormal = True
                    hr_label = f"Heart Rate: <b>{hr} BPM</b> — <span style='color:#e74c3c;'>Tachycardia (&gt;100 BPM)</span>"
                elif hr < 60:
                    hr_abnormal = True
                    hr_label = f"Heart Rate: <b>{hr} BPM</b> — <span style='color:#e67e22;'>Bradycardia (&lt;60 BPM)</span>"
                else:
                    hr_label = f"Heart Rate: <b>{hr} BPM</b> — <span style='color:#27ae60;'>Normal (60–100 BPM)</span>"

            # ─── QTc status ─────────────────────────────────────────────────────
            qtc_label = ""
            qtc_abnormal = False
            if qtc > 0:
                if qtc > 470:
                    qtc_abnormal = True
                    qtc_label = f"QTc: <b>{qtc} ms</b> — <span style='color:#e74c3c;'>Prolonged (&gt;470 ms)</span>"
                elif qtc >= 440:
                    qtc_label = f"QTc: <b>{qtc} ms</b> — <span style='color:#f39c12;'>Borderline (440–470 ms)</span>"
                else:
                    qtc_label = f"QTc: <b>{qtc} ms</b> — <span style='color:#27ae60;'>Normal (&lt;440 ms)</span>"

            # ─── BUILD HTML ─────────────────────────────────────────────────────
            any_metric_abnormal = (
                pr_status is not None
                or qrs_status is not None
                or hr_abnormal
                or qtc_abnormal
            )

            # ── If we have metric data but rhythm_text is still being analysed,
            # show a metric-only interpretation rather than the blank "Waiting..." screen.
            has_metric_data = bool((hr > 0) or (pr > 0) or (qrs > 0) or (qt > 0) or (qtc > 0))
            if not has_metric_data:
                conclusion_html = """
                    <p style='color: #888; font-style: italic;'>
                    Waiting for stable ECG data...<br><br>
                    Metrics are being analyzed. Please wait a few seconds.
                    </p>
                """
                if hasattr(self, 'conclusion_box'):
                    self.conclusion_box.setHtml(conclusion_html)
                return

            # ── If rhythm is still being computed but we DO have metric data,
            # treat it as "rhythm undetermined" so metric labels still show.
            if not rhythm_clean or rhythm_clean in ignore_values:
                # Build a metric-only card without blocking on rhythm detection
                rhythm_issue = None
                is_normal_rhythm = False  # will show CASE 2 (metric-only path)

            # ── Significant-rhythm override ─────────────────────────────────────
            # If the arrhythmia engine's latest analysis contains a clinically
            # significant rhythm (AV Block / AF / AFL) but the rhythm_text from
            # the UI label still says NSR (smoothing lag), force CASE 2 so the
            # finding is shown prominently instead of being hidden under NSR.
            _SIGNIFICANT_LABELS = {
                "Second-degree AV Block (Mobitz I)",
                "Second-degree AV Block (Mobitz II)",
                "Third-degree AV Block",
                "First-degree AV Block (Prolonged PR)",
                "Atrial Fibrillation",
                "Atrial Flutter",
            }
            if is_normal_rhythm:
                try:
                    if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                        _last_sig = getattr(self.ecg_test_page, '_last_analysis', None) or {}
                        _sig_arr = _last_sig.get('arrhythmias', [])
                        _sig_found = [x for x in _sig_arr if x in _SIGNIFICANT_LABELS]
                        if _sig_found:
                            rhythm_issue = _sig_found[0]
                            is_normal_rhythm = False
                except Exception:
                    pass

            if (rhythm_clean in ignore_values or not rhythm_clean) and not has_metric_data:
                pass  # already set conclusion_html above

            elif is_normal_rhythm and not any_metric_abnormal:
                # ── CASE 1: Everything is normal → show Normal Sinus Rhythm only
                conclusion_html = (
                    "<div style='padding:4px;'>"
                    "<span style='font-size:15px; color:#27ae60; font-weight:bold;'>✔ Normal Sinus Rhythm</span><br><br>"
                )
                if hr_label:
                    conclusion_html += f"<span style='color:#555;'>{hr_label}</span><br>"
                if pr_label:
                    conclusion_html += f"<span style='color:#555;'>{pr_label}</span><br>"
                if qrs_label:
                    conclusion_html += f"<span style='color:#555;'>{qrs_label}</span><br>"
                if qtc_label:
                    conclusion_html += f"<span style='color:#555;'>{qtc_label}</span><br>"
                conclusion_html += (
                    "<br><p style='font-size:10px; color:#999; font-style:italic;'>"
                    "<b>NOTE:</b> This is an automated analysis for educational purposes only. "
                    "Not a substitute for professional medical advice."
                    "</p></div>"
                )
                findings.append("Normal Sinus Rhythm")

                # ── Also collect morphological findings (ST, BBB) from
                # ArrhythmiaEngine — these coexist with Normal Sinus Rhythm.
                try:
                    if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                        _last1 = getattr(self.ecg_test_page, '_last_analysis', None) or {}
                        _engine_dx1 = _last1.get('arrhythmias', [])
                        _rhythm_only_labels = {
                            "Normal Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia",
                            "Bradycardia (non-sinus)", "Tachycardia (non-sinus)",
                            "Rhythm Undetermined",
                        }
                        _morph_dx = [x for x in _engine_dx1
                                     if x not in _rhythm_only_labels and x not in _SIGNIFICANT_LABELS]
                        if _morph_dx:
                            findings.extend(_morph_dx)
                except Exception:
                    pass

            else:
                # ── CASE 2: Abnormal rhythm OR any metric abnormal → show everything
                conclusion_html = "<div style='padding:4px;'>"

                # ── Asystole / lethal rhythm short-circuit ──────────────────────────
                # Do NOT append any metric labels (QRS/PR/QTc) alongside a lethal
                # primary — they are meaningless on a flat line and pollute findings.
                _LETHAL_LABELS = {"Asystole", "Ventricular Fibrillation",
                                   "Ventricular Tachycardia"}
                _primary_is_lethal = rhythm_issue and any(
                    lbl.lower() in rhythm_issue.lower() for lbl in _LETHAL_LABELS
                )
                if _primary_is_lethal:
                    _lethal_name = next(
                        (lbl for lbl in _LETHAL_LABELS
                         if lbl.lower() in rhythm_issue.lower()),
                        rhythm_issue,
                    )
                    hr = 0
                    pr = 0
                    qrs = 0
                    qt = 0
                    qtc = 0
                    hr_label = ""
                    pr_label = ""
                    qrs_label = ""
                    qtc_label = ""
                    try:
                        if 'heart_rate' in self.metric_labels:
                            self.metric_labels['heart_rate'].setText("0 BPM")
                        if 'pr_interval' in self.metric_labels:
                            self.metric_labels['pr_interval'].setText("0 ms")
                        if 'qrs_duration' in self.metric_labels:
                            self.metric_labels['qrs_duration'].setText("0 ms")
                        if 'st_interval' in self.metric_labels:
                            self.metric_labels['st_interval'].setText("0 ms")
                        if 'qtc_interval' in self.metric_labels:
                            self.metric_labels['qtc_interval'].setText("0/0")
                    except Exception:
                        pass
                    conclusion_html += (
                        "<b style='color:#cc0000; font-size:15px;'>"
                        f"\u26a0 {_lethal_name}</b><br><br>"
                        "<span style='color:#888; font-size:11px;'>"
                        "All interval calculations: 0 ms (no cardiac activity)<br>"
                        "Immediate clinical attention required."
                        "</span>"
                    )
                    conclusion_html += (
                        "<br><p style='font-size:10px; color:#999; font-style:italic;'>"
                        "<b>NOTE:</b> This is an automated analysis for educational purposes only. "
                        "Not a substitute for professional medical advice."
                        "</p></div>"
                    )
                    findings.append(_lethal_name)
                    # Do NOT append HR / QRS / PR / QTc findings for lethal rhythms
                    # (skip the rest of CASE 2 block)
                else:
                 # 2a. Rhythm (heart-based analysis)
                 if rhythm_issue:
                    conclusion_html += (
                        "<b style='color:#ff6600; font-size:14px;'>\u2665 Heart-Based Rhythm Analysis:</b><br>"
                        f"<span style='color:#e74c3c; font-weight:bold;'>\u26a0 {rhythm_issue}</span><br><br>"
                    )
                    findings.append(f"Rhythm: {rhythm_issue}")
                 elif is_normal_rhythm:
                    # Rhythm is normal but some interval is abnormal
                    conclusion_html += (
                        "<b style='color:#ff6600; font-size:14px;'>\u2665 Heart-Based Rhythm Analysis:</b><br>"
                        "<span style='color:#27ae60; font-weight:bold;'>\u2714 Normal Sinus Rhythm</span><br><br>"
                    )
                    findings.append("Normal Sinus Rhythm")
                 else:
                    # Rhythm engine still computing — show metric-only header
                    conclusion_html += (
                        "<b style='color:#ff6600; font-size:14px;'>♥ ECG Metric Analysis:</b><br>"
                        "<span style='color:#888; font-style:italic;'>Rhythm detection in progress…</span><br><br>"
                    )

                # 2b. HR
                if hr_label:
                    prefix = "⚠ " if hr_abnormal else ""
                    conclusion_html += f"<b style='color:#ff6600;'>Heart Rate:</b> {prefix}{hr_label}<br>"
                    rhythm_is_hr_based = bool(
                        rhythm_clean
                        and any(term in rhythm_clean.lower() for term in ("bradycardia", "tachycardia"))
                    )
                    if hr_abnormal and not rhythm_is_hr_based:
                        findings.append(f"{'Tachycardia' if hr > 100 else 'Bradycardia'} - HR: {hr} BPM")

                # 2c. PR Interval
                if pr_label:
                    conclusion_html += "<br><b style='color:#ff6600; font-size:14px;'>PR Interval:</b><br>"
                    conclusion_html += f"{pr_label}<br>"
                    if pr_status:
                        label_str = "Prolonged PR" if pr_status == "prolonged" else "Short PR"
                        findings.append(f"{label_str} - {pr} ms")

                # 2d. QRS duration / bundle-branch compatible finding
                if qrs_label:
                    conclusion_html += "<br><b style='color:#ff6600; font-size:14px;'>QRS Duration:</b><br>"
                    conclusion_html += f"{qrs_label}<br>"
                    if qrs_status == "wide":
                        findings.append(f"Wide QRS - {qrs} ms")
                    elif qrs_status == "borderline":
                        findings.append(f"Borderline Wide QRS - {qrs} ms")

                # 2e. QTc
                if qtc_label:
                    conclusion_html += "<br><b style='color:#ff6600; font-size:14px;'>QTc Interval:</b><br>"
                    conclusion_html += f"{qtc_label}<br>"
                    if qtc_abnormal:
                        findings.append(f"Prolonged QTc - {qtc} ms")

                # ── ArrhythmiaEngine Priority Sorting ──
                try:
                    engine_dx = []
                    if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                        last = getattr(self.ecg_test_page, '_last_analysis', None) or {}
                        engine_dx = last.get('arrhythmias', [])
                    
                    priority = [
                        "Asystole",
                        "Ventricular Fibrillation",
                        "Ventricular Tachycardia",
                        "Second-degree AV Block (Mobitz II)",
                        "Second-degree AV Block (Mobitz I)",
                        "Third-degree AV Block",
                        "First-degree AV Block (Prolonged PR)",
                        "Atrial Fibrillation",
                        "Complete Left Bundle Branch Block",
                        "Complete Right Bundle Branch Block",
                        "Left Bundle Branch Block",
                        "Right Bundle Branch Block",
                        "Wide QRS",
                        "Sinus Tachycardia",
                        "Sinus Bradycardia",
                        "Normal Sinus Rhythm"
                    ]
                    
                    sorted_arrhythmias = sorted(engine_dx, key=lambda x: priority.index(x) if x in priority else 999)
                    display_text = "<br>".join([f"• {x}" for x in sorted_arrhythmias[:6]])
                    
                    if sorted_arrhythmias:
                        conclusion_html += f"<b style='color:#ff6600;'>Primary Findings:</b><br>{display_text}<br>"
                        findings.extend(sorted_arrhythmias)
                except Exception as e:
                    print("Error sorting clinical priority:", e)

                if recommendations:
                    conclusion_html += "<br><b style='color:#ff6600; font-size:14px;'>Recommendations:</b><br>"
                    for r in recommendations:
                        conclusion_html += f"• {r}<br>"

                conclusion_html += (
                    "<br><p style='font-size:10px; color:#999; font-style:italic;'>"
                    "<b>NOTE:</b> This is an automated analysis for educational purposes only. "
                    "Not a substitute for professional medical advice. Consult a healthcare provider for medical concerns."
                    "</p></div>"
                )

            disconnected_leads = [lead for lead, conn in limb_conn.items() if not conn]
            if disconnected_leads:
                lead_str = ", ".join(disconnected_leads)
                banner = f"<div style='background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 8px 12px; border-radius: 8px; font-weight: bold; margin-bottom: 10px; font-size: 12px;'>⚠️ Signal Lost — Please reconnect lead(s): {lead_str}</div>"
                conclusion_html = banner + conclusion_html

            self._last_valid_conclusion_html = conclusion_html
            if hasattr(self, 'conclusion_box'):
                self.conclusion_box.setHtml(conclusion_html)

            # ─── Save conclusions to JSON for report generation ──────────────────
            try:
                import os
                import json
                import re

                if findings:
                    # Strip any leftover HTML and prefix markers for clean report text
                    clean_findings = []
                    for f in findings:
                        text = re.sub(r'<[^>]+>', '', f).strip()
                        text = re.sub(r'^\[.*?\]\s*', '', text).strip()
                        if text.lower().startswith("rhythm:"):
                            text = text.split(":", 1)[1].strip()
                        for part in text.split(","):
                            label = part.strip()
                            if " - " in label and any(k in label for k in ("Wide QRS", "Borderline Wide QRS", "Prolonged PR", "Short PR")):
                                label = label.split(" - ", 1)[0].strip()
                            if label and label not in clean_findings:
                                clean_findings.append(label)

                    # Only strip "Normal Sinus Rhythm" when a true RHYTHM abnormality is present.
                    # Morphological findings (ST elevation, Wide QRS, QTc) are independent
                    # and can legitimately coexist with Normal Sinus Rhythm.
                    rhythm_abnormal_keywords = (
                        "Block", "Fibrillation", "Flutter", "Tachycardia", "Bradycardia",
                        "PVC", "PAC", "Asystole", "Ventricular"
                    )
                    if any(any(k.lower() in f.lower() for k in rhythm_abnormal_keywords) for f in clean_findings):
                        clean_findings = [f for f in clean_findings if f != "Normal Sinus Rhythm"]

                    clean_recommendations = []
                    for r in recommendations:
                        text = re.sub(r'<[^>]+>', '', r).strip()
                        text = re.sub(r'^[•●○]\s*', '', text).strip()
                        clean_recommendations.append(text)

                    conclusions_data = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "findings": clean_findings,
                        "recommendations": clean_recommendations
                    }

                    conclusions_file = str(data_file("last_conclusions.json"))

                    with open(conclusions_file, 'w') as f:
                        json.dump(conclusions_data, f, indent=2)

                    print(f" Saved {len(clean_findings)} findings to last_conclusions.json")
                    print(f"   Findings: {clean_findings}")

                    # ── Phase-3A: Critical arrhythmia alert banner ────────────────────
                    _CRITICAL_ALERT_KEYWORDS = (
                        "ventricular tachycardia", "ventricular fibrillation",
                        "atrial fibrillation", "afib", "complete heart block",
                        "third degree", "asystole", "stemi", "torsade", "vf", "vt",
                    )
                    combined_lower = " ".join(str(f).lower() for f in clean_findings)
                    is_critical = any(kw in combined_lower for kw in _CRITICAL_ALERT_KEYWORDS)
                    try:
                        if hasattr(self, "_alert_banner_lbl"):
                            self._alert_banner_lbl.setVisible(False)
                    except Exception:
                        pass
                    # ── Phase-3B: Offline queue badge ────────────────────────────────
                    try:
                        if hasattr(self, "_queue_badge_lbl"):
                            oq_dir = str(data_file("offline_queue"))
                            pending = 0
                            if os.path.isdir(oq_dir):
                                pending = sum(
                                    1 for fn in os.listdir(oq_dir)
                                    if fn.lower().endswith((".json", ".pdf"))
                                )
                            if pending > 0:
                                self._queue_badge_lbl.setText(f"📤 {pending} pending")
                                self._queue_badge_lbl.setVisible(True)
                            else:
                                self._queue_badge_lbl.setVisible(False)
                    except Exception:
                        pass

                else:
                    print(" Skipped saving empty findings to last_conclusions.json (waiting for valid ECG data)")

            except Exception as save_err:
                print(f" Error saving conclusions to JSON: {save_err}")


        except Exception as e:
            print(f" Error updating conclusion: {e}")

    def reset_metrics_and_interpretation(self):
        """Reset dashboard metrics, ECG interpretation, and clear waves when device disconnects"""
        # Reset metric labels
        if hasattr(self, 'metric_labels'):
            if 'heart_rate' in self.metric_labels:
                self.metric_labels['heart_rate'].setText("0 BPM")
            if 'pr_interval' in self.metric_labels:
                self.metric_labels['pr_interval'].setText("0 ms")
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setText("0 ms")
            if 'qtc_interval' in self.metric_labels:
                self.metric_labels['qtc_interval'].setText("--")
            if 'st_interval' in self.metric_labels:
                self.metric_labels['st_interval'].setText("--")
        
        # Clear the dashboard's ECG plot (if it has one)
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'axes'):
            try:
                for ax in self.canvas.axes:
                    ax.clear()
                self.canvas.draw()
            except Exception as e:
                print(f"Error clearing dashboard plot: {e}")
        
        # Update the conclusion to show Rhythm Undetermined
        self.update_live_conclusion()

    def show_version_popup(self):
        """Show version information in a popup dialog"""
        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle(self.tr("Version Information"))
        dialog.setMinimumWidth(550)
        dialog.setMinimumHeight(550)
        dialog.setStyleSheet("QDialog { background: #f4f7f6; }")

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        title = QLabel(self.tr("Version Information"))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font: 900 18pt 'Segoe UI', Arial; color: white; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6600, stop:1 #ff8c33); "
            "border-radius: 12px; padding: 14px;"
        )
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(title)

        info_frame = QFrame()
        info_frame.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e0e5eb; border-radius: 16px; }"
        )
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(24, 24, 24, 24)
        info_layout.setSpacing(16)

        hw_v = ""
        sn_v = ""
        is_connected = getattr(self, 'device_connected', True)

        if is_connected:
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                if hasattr(self.ecg_test_page, 'serial_reader') and self.ecg_test_page.serial_reader:
                    if hasattr(self.ecg_test_page.serial_reader, 'device_version') and self.ecg_test_page.serial_reader.device_version:
                        hw_v = self.ecg_test_page.serial_reader.device_version
                    if hasattr(self.ecg_test_page.serial_reader, 'device_serial_number') and self.ecg_test_page.serial_reader.device_serial_number:
                        sn_v = self.ecg_test_page.serial_reader.device_serial_number

            if not hw_v and hasattr(self, 'device_version') and self.device_version:
                hw_v = self.device_version
            if not sn_v and hasattr(self, 'machine_serial_number') and self.machine_serial_number:
                sn_v = self.machine_serial_number

            if hasattr(self, 'settings_manager') and self.settings_manager:
                if not hw_v:
                    hw_v = self.settings_manager.get_setting("hardware_version", "")
                if not sn_v:
                    sn_v = self.settings_manager.get_setting("machine_serial_number", "")
                
                if hw_v and hw_v != "Not Detected":
                    if self.settings_manager.get_setting("hardware_version", "") != hw_v:
                        self.settings_manager.set_setting("hardware_version", hw_v)
                if sn_v and sn_v != "Not Detected":
                    if self.settings_manager.get_setting("machine_serial_number", "") != sn_v:
                        self.settings_manager.set_setting("machine_serial_number", sn_v)

        if not hw_v:
            hw_v = "Not Detected"
        if not sn_v:
            sn_v = "Not Detected"

        version_info = [
            (self.tr("Software Version"), "V 1.1.1"),
            (self.tr("Hardware Version"), hw_v),
            (self.tr("Machine Serial Number"), sn_v),
            (self.tr("Firmware Version"), "V.3.0.1"),
            (self.tr("Build Date"), "2024-08-26"),
            (self.tr("Manufacturer"), "Deckmount Electronics Pvt Ltd"),
            (self.tr("Model"), "CardioX"),
            (self.tr("License"), "Professional Edition")
        ]

        label_style = (
            "QLabel { font: bold 11pt 'Segoe UI', Arial; color: #344054; "
            "border: none; background: transparent; min-width: 180px; }"
        )
        value_style = (
            "QLabel { font: 11pt 'Segoe UI', Arial; color: #ff6600; background: #fcfcfd; padding: 10px 14px;"
            " border: 1px solid #d0d5dd; border-radius: 8px; min-height: 24px; }"
        )

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #f4f7f6; width: 10px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #ff8c33; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #ff6600; }
        """)

        for label, value in version_info:
            row = QHBoxLayout()
            row.setSpacing(16)
            
            lbl = QLabel(label)
            lbl.setStyleSheet(label_style)
            lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            
            val = QLabel(value)
            val.setStyleSheet(value_style)
            val.setWordWrap(True)
            val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            
            row.addWidget(lbl)
            row.addWidget(val)
            info_layout.addLayout(row)

        scroll_area.setWidget(info_frame)
        layout.addWidget(scroll_area)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        close_btn = QPushButton(self.tr("Close"))
        close_btn.setStyleSheet(
            "QPushButton { background: #ff6600; color: white; border-radius: 10px; padding: 10px 24px;"
            " font: bold 11pt 'Segoe UI', Arial; border: none; min-width: 140px; }"
            "QPushButton:hover { background: #e65c00; }"
        )
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)

        layout.addLayout(btn_row)

        try:
            shadow = QGraphicsDropShadowEffect(dialog)
            shadow.setBlurRadius(20)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(16, 24, 40, 30))
            info_frame.setGraphicsEffect(shadow)
        except Exception:
            pass

        dialog.exec_()

    def show_new_registration_dialog(self):
        """Create/update patient registration details (previously 'Save ECG' in ECG menu)."""
        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle(self.tr("Patient Registration"))
        dialog.setMinimumWidth(550)
        dialog.setMinimumHeight(400)
        dialog.setStyleSheet("QDialog { background: #f4f7f6; }")

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        title = QLabel(self.tr("Patient Registration"))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font: 900 18pt 'Segoe UI', Arial; color: white; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6600, stop:1 #ff8c33); "
            "border-radius: 12px; padding: 14px;"
        )
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(title)

        form_frame = QFrame()
        form_frame.setStyleSheet(
            "QFrame { background: #ffffff; border: 1px solid #e0e5eb; border-radius: 16px; }"
        )
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(24, 24, 24, 24)
        form_layout.setSpacing(16)

        label_style = (
            "QLabel { font: bold 11pt 'Segoe UI', Arial; color: #344054; "
            "border: none; background: transparent; min-width: 120px; }"
        )
        input_style = (
            "QLineEdit { font: 11pt 'Segoe UI', Arial; color: #101828; background: #fcfcfd; padding: 10px 14px;"
            " border: 1px solid #d0d5dd; border-radius: 8px; min-height: 24px; }"
            "QLineEdit:focus { border: 2px solid #ff6600; background: #ffffff; }"
        )
        counter_style = (
            "QLabel { font: 9pt 'Segoe UI', Arial; color: rgba(16, 24, 40, 0.45); "
            "background: transparent; border: none; padding: 0px; margin: 0px; }"
        )

        fields = ["Patient Name"]
        entries = {}
        counters = {}
        for field in fields:
            row = QHBoxLayout()
            row.setSpacing(16)
            lbl = QLabel(self.tr(field))
            lbl.setStyleSheet(label_style)

            entry_box = QFrame()
            entry_box.setStyleSheet(
                "QFrame { background: #fcfcfd; border: 1px solid #d0d5dd; border-radius: 8px; }"
                "QFrame:focus-within { border: 2px solid #ff6600; background: #ffffff; }"
            )
            entry_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            entry_box.setFixedHeight(50)

            box_layout = QGridLayout(entry_box)
            box_layout.setContentsMargins(12, 8, 12, 8)
            box_layout.setHorizontalSpacing(0)
            box_layout.setVerticalSpacing(0)

            entry = QLineEdit()
            entry.setPlaceholderText(self.tr(f"Enter {field}"))
            entry.setStyleSheet(
                "QLineEdit { font: 11pt 'Segoe UI', Arial; color: #101828; background: transparent;"
                " border: none; padding: 0px 56px 0px 0px; margin: 0px; }"
                "QLineEdit:focus { border: none; background: transparent; }"
            )
            entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            entry.setMaxLength(20)

            count_label = QLabel("0/20")
            count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            count_label.setStyleSheet(
                "QLabel { font: 9pt 'Segoe UI', Arial; color: rgba(16, 24, 40, 0.45);"
                " background: transparent; border: none; padding: 0px; margin: 0px; min-width: 42px; }"
            )

            box_layout.addWidget(entry, 0, 0, 1, 1)
            box_layout.addWidget(count_label, 0, 0, 1, 1, alignment=Qt.AlignRight | Qt.AlignVCenter)

            entry.textChanged.connect(
                lambda text, lbl=count_label, limit=20: lbl.setText(f"{len(text)}/{limit}")
            )

            row.addWidget(lbl)
            row.addWidget(entry_box)
            form_layout.addLayout(row)
            entries[field] = entry
            counters[field] = count_label

        age_row = QHBoxLayout()
        age_row.setSpacing(16)
        age_lbl = QLabel(self.tr("Age"))
        age_lbl.setStyleSheet(label_style)
        age_entry = QLineEdit()
        age_entry.setPlaceholderText(self.tr("Enter Age"))
        age_entry.setValidator(QIntValidator(0, 100, dialog))
        age_entry.setStyleSheet(input_style)
        age_entry.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        age_entry.setText("1")
        age_row.addWidget(age_lbl)
        age_row.addWidget(age_entry)
        form_layout.addLayout(age_row)
        entries["Age"] = age_entry

        gender_row = QHBoxLayout()
        gender_row.setSpacing(16)
        gender_lbl = QLabel(self.tr("Gender"))
        gender_lbl.setStyleSheet(label_style)
        gender_menu = QComboBox()
        gender_menu.addItems([self.tr("Select Gender"), self.tr("Male"), self.tr("Female"), self.tr("Other")])
        gender_menu.setStyleSheet(
            "QComboBox { font: 11pt 'Segoe UI', Arial; padding: 10px 14px; border: 1px solid #d0d5dd; border-radius: 8px;"
            " background: #fcfcfd; color: #101828; min-height: 24px; }"
            "QComboBox:focus { border: 2px solid #ff6600; background: #ffffff; }"
            "QComboBox::drop-down { border: none; width: 30px; }"
        )
        gender_menu.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        gender_row.addWidget(gender_lbl)
        gender_row.addWidget(gender_menu)
        form_layout.addLayout(gender_row)

        # Prefill previously saved values (same behavior as former Save ECG panel)
        try:
            explicit_patient = getattr(self, "patient_details", None)
            prefill = resolve_patient_profile(
                explicit_patient=explicit_patient,
                username=getattr(self, "username", "") or "",
                user_details=getattr(self, "user_details", {}) or {},
            )

            if not prefill:
                patients_db_file = str(data_file("all_patients.json"))
                if os.path.exists(patients_db_file):
                    with open(patients_db_file, "r") as jf:
                        all_patients = json.load(jf) or {}
                    patients = all_patients.get("patients") or []
                    if patients:
                        prefill = patients[-1]

            if prefill:
                pd = prefill
                first = str(pd.get("first_name") or "")
                last = str(pd.get("last_name") or "")
                full_name = (first + (" " + last if last else "")).strip()
                if full_name:
                    entries["Patient Name"].setText(full_name)
                
                if isinstance(explicit_patient, dict) and explicit_patient.get("age") is not None:
                    entries["Age"].setText(str(explicit_patient.get("age")))
                if pd.get("gender"):
                    idx = gender_menu.findText(str(pd.get("gender")))
                    if idx != -1:
                        gender_menu.setCurrentIndex(idx)
        except Exception:
            pass

        layout.addWidget(form_frame)
        
        # Add a stretch to push form fields upwards and buttons downwards
        layout.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.setStyleSheet(
            "QPushButton { background: #ffffff; color: #344054; border-radius: 10px; padding: 10px 24px;"
            " font: bold 11pt 'Segoe UI', Arial; border: 1px solid #d0d5dd; }"
            "QPushButton:hover { background: #f9fafb; }"
        )
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(cancel_btn)

        save_btn = QPushButton(self.tr("Save"))
        save_btn.setStyleSheet(
            "QPushButton { background: #ff6600; color: white; border-radius: 10px; padding: 10px 24px;"
            " font: bold 11pt 'Segoe UI', Arial; border: none; }"
            "QPushButton:hover { background: #e65c00; }"
        )
        save_btn.setCursor(Qt.PointingHandCursor)
        save_btn.clicked.connect(lambda: self._submit_new_registration(entries, gender_menu, dialog))
        btn_row.addWidget(save_btn)

        layout.addLayout(btn_row)
        
        # Add a subtle drop shadow to the form frame
        try:
            shadow = QGraphicsDropShadowEffect(dialog)
            shadow.setBlurRadius(20)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(16, 24, 40, 30))
            form_frame.setGraphicsEffect(shadow)
        except Exception:
            pass

        dialog.exec_()

    def _submit_new_registration(self, entries, gender_menu, dialog):
        values = {
            label: entries[label].text().strip()
            for label in ["Patient Name", "Age"]
        }
        values["Gender"] = (gender_menu.currentText() or "").strip()

        if any(v == "" for v in values.values()) or gender_menu.currentIndex() == 0:
            QMessageBox.warning(
                dialog,
                self.tr("Missing Data"),
                self.tr("Please fill all the fields and select gender."),
            )
            return

        all_patients = {"patients": []}
        try:
            from datetime import datetime as _dt

            name = values["Patient Name"]
            first, *rest = name.split()
            patient_details = {
                "first_name": first,
                "last_name": " ".join(rest),
                "age": values["Age"],
                "gender": values["Gender"],
                "patient_name": values.get("Patient Name", ""),
                "date_time": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            setattr(self, "patient_details", patient_details)

            # Persist to centralized all_patients.json database
            patients_db_file = str(data_file("all_patients.json"))

            if os.path.exists(patients_db_file):
                try:
                    with open(patients_db_file, "r") as jf:
                        all_patients = json.load(jf) or {"patients": []}
                except Exception:
                    all_patients = {"patients": []}

            patient_details["id"] = len(all_patients.get("patients", []) or []) + 1
            if "patients" not in all_patients:
                all_patients["patients"] = []
            all_patients["patients"].append(patient_details)

            with open(patients_db_file, "w") as jf:
                json.dump(all_patients, jf, indent=2)
        except Exception as e:
            print(f" Could not persist patient details: {e}")

        QMessageBox.information(
            dialog,
            self.tr("Saved"),
            self.tr(
                f" Patient details saved successfully!\n\nTotal patients in database: {len(all_patients.get('patients', []) or [])}"
            ),
        )
        dialog.accept()

    def show_help_support_dialog(self):
        """Open Help & Support hub dialog (more options can be added later)."""
        try:
            from utils.support_api import get_support_api
        except Exception as e:
            QMessageBox.critical(self, "Support", f"Support module not available: {e}")
            return

        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle("Help & Support")
        dialog.setMinimumWidth(680)
        dialog.setMinimumHeight(550)
        dialog.setStyleSheet("""
            QDialog { background: #f4f7f6; }

            QFrame#supportCard {
                background: #ffffff;
                border: 1px solid #e0e5eb;
                border-radius: 16px;
            }

            QLabel { color: #101828; }
            QLabel#supportHint { font-size: 11pt; font-family: 'Segoe UI', Arial; color: #667085; }
            QLabel#supportSection { font-size: 11pt; font-weight: bold; font-family: 'Segoe UI', Arial; color: #101828; }
            QLabel#supportOtherLabel { font-size: 10.5pt; font-weight: bold; font-family: 'Segoe UI', Arial; color: #344054; }

            QScrollArea { background: transparent; border: none; }
            QScrollArea > QWidget > QWidget { background: transparent; }

            QFrame#issueTile {
                background: #fcfcfd;
                border: 1px solid #d0d5dd;
                border-radius: 10px;
            }
            QFrame#issueTile:hover { border: 1px solid #ffb380; background: #ffffff; }
            QFrame#issueTile[selected="true"] { border: 2px solid #ff6600; background: #ffffff; }
            QFrame#otherIssueCard {
                background: #fffaf5;
                border: 1px solid #ffd2b0;
                border-radius: 12px;
            }

            QRadioButton {
                font-size: 11pt;
                font-family: 'Segoe UI', Arial;
                color: #101828;
                padding: 10px 10px;
            }
            QRadioButton::indicator { width: 18px; height: 18px; }
            QRadioButton::indicator:unchecked {
                border: 2px solid #d0d5dd;
                border-radius: 9px;
                background: #ffffff;
            }
            QRadioButton::indicator:checked {
                border: 5px solid #ff6600;
                border-radius: 9px;
                background: #ffffff;
            }

            QLineEdit, QTextEdit {
                font: 11pt 'Segoe UI', Arial; color: #101828; background: #fcfcfd; padding: 10px 14px;
                border: 1px solid #d0d5dd; border-radius: 8px;
            }
            QTextEdit { min-height: 120px; }
            QLineEdit:focus, QTextEdit:focus { border: 2px solid #ff6600; background: #ffffff; }

            QPushButton#supportCancel {
                background: #ffffff; color: #344054; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial; border: 1px solid #d0d5dd; min-width: 120px;
            }
            QPushButton#supportCancel:hover { background: #f9fafb; }
            QPushButton#supportCancel:pressed { background: #e9edf3; }

            QPushButton#supportOutline {
                background: #ffffff;
                color: #ff6600;
                border-radius: 10px;
                padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial;
                border: 2px solid #ff6600;
                min-width: 140px;
            }
            QPushButton#supportOutline:hover { background: #fff3e8; }
            QPushButton#supportOutline:pressed { background: #ffd9bd; }
            QPushButton#supportOutline:disabled { background: #ffffff; color: #98a2b3; border: 2px solid #eaecf0; }

            QPushButton#supportPrimary, QPushButton#supportSubmit {
                background: #ff6600; color: white; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial; border: none; min-width: 140px;
            }
            QPushButton#supportPrimary:hover, QPushButton#supportSubmit:hover { background: #e65c00; }
            QPushButton#supportPrimary:pressed, QPushButton#supportSubmit:pressed { background: #cc5200; }
        """)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(20)

        title = QLabel("Help & Support")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font: 900 18pt 'Segoe UI', Arial; color: white; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6600, stop:1 #ff8c33); "
            "border-radius: 12px; padding: 14px;"
        )
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(title)

        card = QFrame()
        card.setObjectName("supportCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        top = QHBoxLayout()
        hint = QLabel("Please select your issue. If it is not listed, choose Other and describe it below.")
        hint.setObjectName("supportHint")
        hint.setWordWrap(True)
        top.addWidget(hint, 1)

        status_btn = QPushButton("Check Status")
        status_btn.setObjectName("supportOutline")
        status_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        top_actions = QHBoxLayout()
        top_actions.setSpacing(10)
        top_actions.addWidget(status_btn)
        top.addLayout(top_actions, 0)
        card_layout.addLayout(top)

        sec_title = QLabel("Select an issue")
        sec_title.setObjectName("supportSection")
        card_layout.addWidget(sec_title)

        issue_group = QButtonGroup(dialog)
        tiles_container = QWidget()
        tiles_layout = QVBoxLayout(tiles_container)
        tiles_layout.setContentsMargins(2, 2, 2, 2)
        tiles_layout.setSpacing(12)

        issues_data = [
            {"text": "Device Power ON issue", "source": "hardware"},
            {"text": "USB serial port not detected", "source": "hardware"},
            {"text": "Hardware LED blinking abnormally. Device unresponsive", "source": "hardware"},
            {"text": "ECG wave not shown on screen", "source": "software"},
            {"text": "Software crashes / freezes during test", "source": "software"},
            {"text": "Report PDF not generating properly", "source": "software"},
            {"text": "Other", "source": "other"},
        ]

        other_card = QFrame()
        other_card.setObjectName("otherIssueCard")
        other_card.setVisible(False)
        other_layout = QVBoxLayout(other_card)
        other_layout.setContentsMargins(14, 12, 14, 14)
        other_layout.setSpacing(10)

        other_label = QLabel("Describe the other issue")
        other_label.setObjectName("supportOtherLabel")
        other_layout.addWidget(other_label)

        other_input = QTextEdit()
        other_input.setPlaceholderText("Type the issue details here...")
        other_layout.addWidget(other_input)

        for i, item in enumerate(issues_data):
            tile = QFrame()
            tile.setObjectName("issueTile")
            tile.setProperty("selected", "false")
            t_lay = QHBoxLayout(tile)
            t_lay.setContentsMargins(12, 10, 12, 10)
            
            rb = QRadioButton(item["text"])
            rb.setProperty("issue_text", item["text"])
            rb.setProperty("issue_source", item["source"])
            issue_group.addButton(rb, i)
            t_lay.addWidget(rb)

            def _mk_toggled(t=tile, source=item["source"]):
                def _on_t(checked):
                    t.setProperty("selected", "true" if checked else "false")
                    t.style().unpolish(t)
                    t.style().polish(t)
                    if source == "other":
                        other_card.setVisible(checked)
                        if checked:
                            other_input.setFocus()
                    elif checked:
                        other_card.setVisible(False)
                return _on_t
            rb.toggled.connect(_mk_toggled(tile))
            tiles_layout.addWidget(tile)

        tiles_layout.addWidget(other_card)
        tiles_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(tiles_container)
        scroll.setMinimumHeight(280)
        scroll.setStyleSheet("""
            QScrollBar:vertical { background: #f4f7f6; width: 10px; border-radius: 5px; }
            QScrollBar::handle:vertical { background: #ff8c33; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #ff6600; }
        """)
        card_layout.addWidget(scroll, 1)

        def _open_status():
            self.show_complaint_status_dialog()

        def _resolve_identity():
            details = self.user_details or {}
            name = str(details.get("full_name") or details.get("name") or self.username or "").strip()
            machine_id = str(
                details.get("serial_id")
                or details.get("machine_serial_id")
                or details.get("serial_number")
                or ""
            ).strip()
            return name, machine_id

        class _SubmitComplaintThread(QThread):
            done = pyqtSignal(dict)

            def __init__(self, payload: dict, parent=None):
                super().__init__(parent)
                self._payload = payload

            def run(self):
                try:
                    api = get_support_api()
                    result = api.submit_complaint(
                        name=self._payload.get("name", ""),
                        machine_id=self._payload.get("machine_id", ""),
                        complaint=self._payload.get("complaint", ""),
                        source=self._payload.get("source", "software"),
                        queue_if_offline=False,
                    )
                    if not isinstance(result, dict):
                        result = {"success": False, "status": "error", "message": "Unexpected response"}
                except Exception as e:
                    result = {"success": False, "status": "error", "message": str(e)}
                self.done.emit(result)

        def _finish_submission(result: dict):
            submit_btn.setEnabled(True)
            close_btn.setEnabled(True)
            submit_btn.setText(old_submit_text)

            def _show_result_dialog(title_text: str, message_text: str, reference_label: str = "", reference_value: str = "", show_status: bool = False):
                result_dialog = QDialog(dialog)
                result_dialog.setWindowFlags(result_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                result_dialog.setWindowTitle("Help & Support")
                result_dialog.setMinimumWidth(560)
                result_dialog.setStyleSheet(dialog.styleSheet())

                layout = QVBoxLayout(result_dialog)
                layout.setContentsMargins(18, 18, 18, 18)
                layout.setSpacing(14)

                header = QLabel(title_text)
                header.setAlignment(Qt.AlignCenter)
                header.setStyleSheet(
                    "font: 900 15pt 'Segoe UI', Arial; color: white; "
                    "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6600, stop:1 #ff8c33); "
                    "border-radius: 12px; padding: 12px;"
                )
                layout.addWidget(header)

                body = QLabel(message_text)
                body.setWordWrap(True)
                body.setStyleSheet("font: 11pt 'Segoe UI', Arial; color: #344054;")
                layout.addWidget(body)

                ref_box = None
                if reference_value:
                    ref_card = QFrame()
                    ref_card.setObjectName("supportCard")
                    ref_layout = QVBoxLayout(ref_card)
                    ref_layout.setContentsMargins(14, 14, 14, 14)
                    ref_layout.setSpacing(10)

                    ref_label = QLabel(reference_label or "Reference ID")
                    ref_label.setObjectName("supportFieldLabel")
                    ref_layout.addWidget(ref_label)

                    ref_box = QLineEdit(reference_value)
                    ref_box.setReadOnly(True)
                    ref_box.setCursorPosition(0)
                    ref_layout.addWidget(ref_box)

                    layout.addWidget(ref_card)

                button_row = QHBoxLayout()
                button_row.addStretch()

                copy_btn = QPushButton("Copy")
                copy_btn.setObjectName("supportSubmit")
                copy_btn.setFixedWidth(110)
                button_row.addWidget(copy_btn)

                status_btn = None
                if show_status and reference_value:
                    status_btn = QPushButton("Check Status")
                    status_btn.setObjectName("supportCancel")
                    status_btn.setFixedWidth(130)
                    button_row.addWidget(status_btn)

                close_result_btn = QPushButton("Close")
                close_result_btn.setObjectName("supportCancel")
                close_result_btn.setFixedWidth(110)
                button_row.addWidget(close_result_btn)
                layout.addLayout(button_row)

                def _copy_reference():
                    try:
                        QApplication.clipboard().setText(reference_value)
                        copy_btn.setText("Copied")
                    except Exception:
                        pass

                copy_btn.clicked.connect(_copy_reference)
                if status_btn is not None:
                    status_btn.clicked.connect(lambda: (result_dialog.accept(), self.show_complaint_status_dialog(reference_value)))
                close_result_btn.clicked.connect(result_dialog.accept)

                result_dialog.exec_()

            def _show_error_dialog(message_text: str):
                error_dialog = QDialog(dialog)
                error_dialog.setWindowFlags(error_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                error_dialog.setWindowTitle("Help & Support")
                error_dialog.setMinimumWidth(520)
                error_dialog.setStyleSheet(dialog.styleSheet())

                layout = QVBoxLayout(error_dialog)
                layout.setContentsMargins(18, 18, 18, 18)
                layout.setSpacing(14)

                header = QLabel("Connection Required")
                header.setAlignment(Qt.AlignCenter)
                header.setStyleSheet(
                    "font: 900 15pt 'Segoe UI', Arial; color: white; "
                    "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d92d20, stop:1 #f04438); "
                    "border-radius: 12px; padding: 12px;"
                )
                layout.addWidget(header)

                body = QLabel(message_text)
                body.setWordWrap(True)
                body.setStyleSheet("font: 11pt 'Segoe UI', Arial; color: #7a271a;")
                layout.addWidget(body)

                button_row = QHBoxLayout()
                button_row.addStretch()
                close_btn = QPushButton("OK")
                close_btn.setObjectName("supportCancel")
                close_btn.setFixedWidth(110)
                button_row.addWidget(close_btn)
                layout.addLayout(button_row)

                close_btn.clicked.connect(error_dialog.accept)
                error_dialog.exec_()

            if result.get("success") and result.get("status") == "success":
                complaint_id = str(result.get("complaint_id") or "").strip()
                _show_result_dialog(
                    "Complaint Submitted",
                    "Complaint submitted successfully. Copy the complaint ID to check its status later.",
                    "Complaint ID",
                    complaint_id,
                    show_status=True,
                )
                dialog.accept()
                return

            _show_error_dialog(
                str(result.get("message") or "You are offline, please connect to internet to submit the complaint.")
            )

        def _open_contact():
            contact = QDialog(dialog)
            contact.setWindowFlags(contact.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            contact.setWindowTitle("Contact Us")
            contact.setMinimumWidth(520)
            contact.setStyleSheet(dialog.styleSheet())

            lay = QVBoxLayout(contact)
            lay.setContentsMargins(18, 16, 18, 16)
            lay.setSpacing(12)

            t = QLabel("Contact Us")
            t.setStyleSheet(
                "font: 900 16pt 'Segoe UI', Arial; color: white; "
                "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6600, stop:1 #ff8c33); "
                "border-radius: 10px; padding: 10px;"
            )
            t.setAlignment(Qt.AlignCenter)
            lay.addWidget(t)

            c_card = QFrame()
            c_card.setObjectName("supportCard")
            c_lay = QVBoxLayout(c_card)
            c_lay.setContentsMargins(14, 14, 14, 14)
            c_lay.setSpacing(12)

            def _open_link(url: str):
                if not QDesktopServices.openUrl(QUrl(url)):
                    QMessageBox.warning(contact, "Contact Us", "Unable to open the selected contact option.")

            def _make_contact_button(icon_name: str, tooltip: str, url: str, accent: str, hover: str):
                btn = QToolButton()
                btn.setCursor(Qt.PointingHandCursor)
                btn.setToolTip(tooltip)
                btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
                btn.setIconSize(QSize(58, 58))
                btn.setMinimumSize(210, 112)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setStyleSheet(
                    "QToolButton {"
                    "  background: white;"
                    "  border: 1px solid #d8e0ea;"
                    "  border-radius: 18px;"
                    "  padding: 18px;"
                    "}"
                    "QToolButton:hover {"
                    "  background: #f8fbff;"
                    "  border: 1px solid #b7c5d8;"
                    "}"
                    "QToolButton:pressed {"
                    "  background: #eef4fb;"
                    "  padding-top: 20px;"
                    "}"
                )

                icon_path = get_asset_path(icon_name)
                pixmap = QPixmap(icon_path)
                if not pixmap.isNull():
                    btn.setIcon(QIcon(pixmap))
                btn.clicked.connect(lambda _checked=False, link=url: _open_link(link))
                return btn

            icon_row = QHBoxLayout()
            icon_row.setSpacing(14)

            whatsapp_url = "https://web.whatsapp.com/send?phone=919311225195"
            gmail_url = "https://mail.google.com/mail/?view=cm&fs=1&to=wecare@deckmount.in"

            icon_row.addWidget(
                _make_contact_button(
                    "whatsapp.png",
                    "Open WhatsApp chat with support",
                    whatsapp_url,
                    "#25D366",
                    "#1fb757",
                )
            )
            icon_row.addWidget(
                _make_contact_button(
                    "gmail.png",
                    "Open Gmail compose with support address",
                    gmail_url,
                    "#ff8c33",
                    "#ff7a14",
                )
            )
            c_lay.addLayout(icon_row)

            lay.addWidget(c_card)

            btns = QHBoxLayout()
            btns.addStretch()
            ok = QPushButton("Close")
            ok.setObjectName("supportCancel")
            btns.addWidget(ok)
            lay.addLayout(btns)
            ok.clicked.connect(contact.accept)
            contact.exec_()

        footer = QHBoxLayout()
        submit_btn = QPushButton("Submit Issue")
        submit_btn.setObjectName("supportPrimary")
        footer.addWidget(submit_btn)

        contact_btn = QPushButton("Contact Us")
        contact_btn.setObjectName("supportOutline")
        footer.addWidget(contact_btn)

        footer.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setObjectName("supportCancel")
        footer.addWidget(close_btn)
        card_layout.addLayout(footer)

        old_submit_text = submit_btn.text()

        def _open_submit_issue():
            selected = issue_group.checkedButton()
            if not selected:
                QMessageBox.warning(dialog, "Help & Support", "Please select an issue.")
                return

            issue_text = str(selected.property("issue_text") or "").strip()
            issue_source = str(selected.property("issue_source") or "software").strip() or "software"
            complaint_text = issue_text
            if issue_source == "other":
                complaint_text = other_input.toPlainText().strip()
                if not complaint_text:
                    QMessageBox.warning(dialog, "Help & Support", "Please describe the other issue before submitting.")
                    return

            name, machine_id = _resolve_identity()
            if not name or not machine_id:
                QMessageBox.warning(
                    dialog,
                    "Help & Support",
                    "Your signup name and machine serial ID are missing. Please sign up again or update your profile before submitting.",
                )
                return

            payload = {
                "name": name,
                "machine_id": machine_id,
                "complaint": complaint_text,
                "source": issue_source,
            }

            submit_btn.setEnabled(False)
            close_btn.setEnabled(False)
            submit_btn.setText("Submitting...")

            worker = _SubmitComplaintThread(payload, dialog)
            worker.done.connect(_finish_submission)
            worker.start()
            dialog._support_submit_worker = worker

        root.addWidget(card)

        try:
            shadow = QGraphicsDropShadowEffect(dialog)
            shadow.setBlurRadius(20)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(16, 24, 40, 30))
            card.setGraphicsEffect(shadow)
        except Exception:
            pass

        close_btn.clicked.connect(dialog.reject)
        status_btn.clicked.connect(_open_status)
        submit_btn.clicked.connect(_open_submit_issue)
        contact_btn.clicked.connect(_open_contact)

        dialog.exec_()

    def show_complaint_status_dialog(self, prefill_id: str = ""):
        """Open a dialog to check complaint status by complaint_id."""
        try:
            from utils.support_api import get_support_api
        except Exception as e:
            QMessageBox.critical(self, "Support", f"Support module not available: {e}")
            return

        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle("Complaint Status")
        dialog.setMinimumWidth(620)
        dialog.setStyleSheet("""
            QDialog { background: #f4f7f6; }

            QFrame#supportCard {
                background: #ffffff;
                border: 1px solid #e0e5eb;
                border-radius: 16px;
            }

            QLabel { color: #101828; }
            QLabel#supportHint { font-size: 11pt; font-family: 'Segoe UI', Arial; color: #667085; }
            QLabel#statusKey { font-size: 11pt; font-family: 'Segoe UI', Arial; font-weight: bold; color: #344054; min-width: 100px; }
            QLabel#statusValue { font-size: 11pt; font-family: 'Segoe UI', Arial; color: #101828; }
            QLabel#statusBadge {
                padding: 6px 16px;
                border-radius: 14px;
                font-family: 'Segoe UI', Arial;
                font-weight: 900;
                font-size: 11pt;
                color: #b24a00;
                background: #fff3e8;
                border: none;
            }
            QFrame#statusCard {
                background: #fcfcfd;
                border: 1px solid #eaecf0;
                border-radius: 14px;
            }
            QLineEdit {
                font: 11pt 'Segoe UI', Arial; color: #101828; background: #fcfcfd; padding: 10px 14px;
                border: 1px solid #d0d5dd; border-radius: 8px; min-height: 24px;
            }
            QLineEdit:focus { border: 2px solid #ff6600; background: #ffffff; }
            QPushButton#supportCancel {
                background: #ffffff; color: #344054; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial; border: 1px solid #d0d5dd; min-width: 120px;
            }
            QPushButton#supportCancel:hover { background: #f9fafb; }
            QPushButton#supportCancel:pressed { background: #e9edf3; }
            QPushButton#supportPrimary {
                background: #ff6600; color: white; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial; border: none; min-width: 140px;
            }
            QPushButton#supportPrimary:hover { background: #e65c00; }
            QPushButton#supportPrimary:pressed { background: #cc5200; }
        """)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(20)

        title = QLabel("Complaint Status")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font: 900 18pt 'Segoe UI', Arial; color: white; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6600, stop:1 #ff8c33); "
            "border-radius: 12px; padding: 14px;"
        )
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(title)

        card = QFrame()
        card.setObjectName("supportCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        hint = QLabel("Paste your Complaint ID and click Check.")
        hint.setObjectName("supportHint")
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(16)
        cid_input = QLineEdit(str(prefill_id or "").strip())
        cid_input.setPlaceholderText("Complaint ID")
        try:
            cid_input.setClearButtonEnabled(True)
        except Exception:
            pass
        check_btn = QPushButton("Check")
        check_btn.setObjectName("supportPrimary")
        row.addWidget(cid_input, 1)
        row.addWidget(check_btn)
        card_layout.addLayout(row)

        result_box = QFrame()
        result_box.setObjectName("statusCard")
        box_lay = QVBoxLayout(result_box)
        box_lay.setContentsMargins(20, 20, 20, 20)
        box_lay.setSpacing(12)

        status_line = QHBoxLayout()
        status_label = QLabel("Status")
        status_label.setObjectName("statusKey")
        badge = QLabel("-")
        badge.setObjectName("statusBadge")
        status_line.addWidget(status_label)
        status_line.addWidget(badge, 0, Qt.AlignLeft)
        status_line.addStretch()
        box_lay.addLayout(status_line)

        def _kv_row(key_text: str):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            key = QLabel(key_text)
            key.setObjectName("statusKey")
            value = QLabel("-")
            value.setObjectName("statusValue")
            row.addWidget(key)
            row.addSpacing(12)
            row.addWidget(value, 1, Qt.AlignLeft)
            return row, value

        created_row, created_val = _kv_row("Created at")
        updated_row, updated_val = _kv_row("Updated at")
        resolved_row, resolved_val = _kv_row("Resolved at")
        box_lay.addLayout(created_row)
        box_lay.addLayout(updated_row)
        box_lay.addLayout(resolved_row)

        card_layout.addWidget(result_box)

        footer = QHBoxLayout()
        footer.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setObjectName("supportCancel")
        footer.addWidget(close_btn)
        card_layout.addLayout(footer)

        root.addWidget(card)

        try:
            shadow = QGraphicsDropShadowEffect(dialog)
            shadow.setBlurRadius(20)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(16, 24, 40, 30))
            card.setGraphicsEffect(shadow)
        except Exception:
            pass

        close_btn.clicked.connect(dialog.accept)

        def _badge_style(status_value: str) -> str:
            base = "padding:6px 16px;border-radius:14px;font-family:'Segoe UI',Arial;font-weight:900;font-size:11pt;border:none;"
            s = (status_value or "").strip().lower()
            if s == "open":
                return base + "background:#fff3e8;color:#b24a00;"
            if s == "inprogress":
                return base + "background:#fff3e8;color:#b24a00;"
            if s == "resolved":
                return base + "background:#eafaf1;color:#1c7a44;"
            if s == "close" or s == "closed":
                return base + "background:#f2f4f7;color:#344054;"
            return base + "background:#f2f4f7;color:#111;"

        def _render(data: dict):
            status_value = str(data.get("status") or "").strip()
            badge.setText(status_value.title() if status_value else "-")
            badge.setStyleSheet(_badge_style(status_value))

            created_val.setText(str(data.get("created_at") or "-"))
            updated_val.setText(str(data.get("updated_at") or "-"))
            resolved_val.setText(str(data.get("resolved_at") or "-"))

        def _check():
            complaint_id = cid_input.text().strip()
            if not complaint_id:
                QMessageBox.warning(dialog, "Support", "Please enter Complaint ID.")
                return

            check_btn.setEnabled(False)
            try:
                data = get_support_api().get_complaint_status(complaint_id)
            finally:
                check_btn.setEnabled(True)

            if isinstance(data, dict) and (data.get("success") is True or (data.get("status") and data.get("status") != "error")):
                _render(data)
                return

            msg = None
            if isinstance(data, dict):
                msg = data.get("message") or data.get("error") or data.get("detail")
            if not msg or str(msg).strip().lower() == "none":
                msg = f"Complaint ID '{complaint_id}' not found. Please check your Complaint ID and try again."

            QMessageBox.warning(dialog, "Support", str(msg))

        check_btn.clicked.connect(_check)

        # Auto-check if prefilled
        if str(prefill_id or "").strip():
            _check()

        dialog.exec_()

    def _legacy_support_submission_dialog(self, prefill_complaint: str = "", prefill_source: str = ""):
        """Legacy support submission dialog."""
        try:
            from utils.support_api import get_support_api
        except Exception as e:
            QMessageBox.critical(self, "Support", f"Support module not available: {e}")
            return

        dialog = QDialog(self)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setWindowTitle("Help & Support")
        dialog.setMinimumWidth(620)
        dialog.setStyleSheet("""
            QDialog { background: #f4f7f6; }

            QFrame#supportCard {
                background: #ffffff;
                border: 1px solid #e0e5eb;
                border-radius: 16px;
            }

            QLabel { color: #101828; font-family: 'Segoe UI', Arial; }
            QLabel#supportHint { font-size: 11pt; color: #667085; }
            QLabel#supportFieldLabel { font-size: 11pt; font-weight: bold; color: #344054; }

            QLineEdit, QComboBox, QTextEdit {
                font: 11pt 'Segoe UI', Arial; color: #101828; background: #fcfcfd; padding: 10px 14px;
                border: 1px solid #d0d5dd; border-radius: 8px; min-height: 24px;
            }
            QTextEdit { min-height: 120px; }

            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 2px solid #ff6600; background: #ffffff;
            }

            QComboBox::drop-down {
                border: none;
                width: 30px;
            }

            QPushButton#supportCancel {
                background: #ffffff; color: #344054; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial; border: 1px solid #d0d5dd; min-width: 120px;
            }
            QPushButton#supportCancel:hover { background: #f9fafb; }
            QPushButton#supportCancel:pressed { background: #e9edf3; }

            QPushButton#supportSubmit {
                background: #ff6600; color: white; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial; border: none; min-width: 140px;
            }
            QPushButton#supportSubmit:hover { background: #e65c00; }
            QPushButton#supportSubmit:pressed { background: #cc5200; }
        """)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(20)

        title = QLabel("Help & Support")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font: 900 18pt 'Segoe UI', Arial; color: white; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6600, stop:1 #ff8c33); "
            "border-radius: 12px; padding: 14px;"
        )
        title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        root.addWidget(title)

        card = QFrame()
        card.setObjectName("supportCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        card_layout.setSpacing(16)

        hint = QLabel("Send a complaint to support. If offline, it will be queued and sent automatically when online.")
        hint.setWordWrap(True)
        hint.setObjectName("supportHint")
        card_layout.addWidget(hint)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(16)

        name_input = QLineEdit()
        name_input.setPlaceholderText("Your name")
        try:
            name_input.setClearButtonEnabled(True)
        except Exception:
            pass

        machine_input = QLineEdit()
        machine_input.setPlaceholderText("Machine ID / Serial Number")
        try:
            machine_input.setClearButtonEnabled(True)
        except Exception:
            pass

        source_input = QComboBox()
        source_input.addItems(["software", "hardware", "other"])
        preferred_source = str(prefill_source or "").strip().lower()
        if preferred_source in {"software", "hardware", "other"}:
            source_input.setCurrentText(preferred_source)
        else:
            source_input.setCurrentText("software")

        complaint_input = QTextEdit()
        complaint_input.setPlaceholderText("Describe the issue…")
        if str(prefill_complaint or "").strip():
            complaint_input.setPlainText(str(prefill_complaint).strip())

        def _lbl(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setObjectName("supportFieldLabel")
            return lbl

        form.addRow(_lbl("Name"), name_input)
        form.addRow(_lbl("Machine ID"), machine_input)
        form.addRow(_lbl("Source"), source_input)
        form.addRow(_lbl("Complaint"), complaint_input)

        card_layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("supportCancel")
        submit_btn = QPushButton("Submit")
        submit_btn.setObjectName("supportSubmit")
        buttons.addWidget(cancel_btn)
        buttons.addWidget(submit_btn)
        card_layout.addLayout(buttons)

        root.addWidget(card)

        try:
            shadow = QGraphicsDropShadowEffect(dialog)
            shadow.setBlurRadius(20)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(16, 24, 40, 30))
            card.setGraphicsEffect(shadow)
        except Exception:
            pass

        cancel_btn.clicked.connect(dialog.reject)

        class _SubmitComplaintThread(QThread):
            done = pyqtSignal(dict)

            def __init__(self, payload: dict, parent=None):
                super().__init__(parent)
                self._payload = payload

            def run(self):
                try:
                    api = get_support_api()
                    result = api.submit_complaint(
                        name=self._payload.get("name", ""),
                        machine_id=self._payload.get("machine_id", ""),
                        complaint=self._payload.get("complaint", ""),
                        source=self._payload.get("source", "software"),
                        queue_if_offline=False,
                    )
                    if not isinstance(result, dict):
                        result = {"success": False, "status": "error", "message": "Unexpected response"}
                except Exception as e:
                    result = {"success": False, "status": "error", "message": str(e)}
                self.done.emit(result)

        def _submit():
            payload = {
                "name": name_input.text().strip(),
                "machine_id": machine_input.text().strip(),
                "complaint": complaint_input.toPlainText().strip(),
                "source": str(source_input.currentText() or "software").strip() or "software",
            }

            if not payload["name"]:
                QMessageBox.warning(dialog, "Support", "Please enter your name to raise the complaint.")
                return

            if not payload["complaint"]:
                QMessageBox.warning(dialog, "Support", "Please enter your complaint.")
                return

            submit_btn.setEnabled(False)
            cancel_btn.setEnabled(False)
            old_text = submit_btn.text()
            submit_btn.setText("Submitting...")

            def _finish(result: dict):
                submit_btn.setEnabled(True)
                cancel_btn.setEnabled(True)
                submit_btn.setText(old_text)

            def _handle_result(result: dict):
                _finish(result)

                if result.get("success") and result.get("status") == "success":
                    cid = result.get("complaint_id") or ""
                    copy_box = QDialog(dialog)
                    copy_box.setWindowFlags(copy_box.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                    copy_box.setWindowTitle("Complaint Submitted")
                    copy_box.setMinimumWidth(520)
                    copy_box.setStyleSheet(dialog.styleSheet())

                    lay = QVBoxLayout(copy_box)
                    title2 = QLabel("Complaint submitted successfully.")
                    title2.setStyleSheet(
                        "font: 900 14pt 'Segoe UI', Arial; color: white; "
                        "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6600, stop:1 #ff8c33); "
                        "border-radius: 8px; padding: 10px;"
                    )
                    title2.setAlignment(Qt.AlignCenter)
                    lay.addWidget(title2)

                    msg = QLabel("Please copy the Complaint ID for future reference.")
                    msg.setObjectName("supportHint")
                    msg.setWordWrap(True)
                    lay.addWidget(msg)

                    row2 = QHBoxLayout()
                    id_box = QLineEdit(str(cid))
                    id_box.setReadOnly(True)
                    id_box.setSelection(0, len(str(cid)))
                    row2.addWidget(id_box, 1)
                    lay.addLayout(row2)

                    action_row = QHBoxLayout()
                    action_row.addStretch()
                    copy_btn = QPushButton("Copy")
                    copy_btn.setObjectName("supportSubmit")
                    copy_btn.setFixedWidth(110)
                    status_btn = QPushButton("Check Status")
                    status_btn.setObjectName("supportCancel")
                    status_btn.setFixedWidth(110)
                    close_btn2 = QPushButton("Close")
                    close_btn2.setObjectName("supportCancel")
                    close_btn2.setFixedWidth(110)
                    action_row.addWidget(copy_btn)
                    action_row.addSpacing(10)
                    action_row.addWidget(status_btn)
                    action_row.addSpacing(10)
                    action_row.addWidget(close_btn2)
                    lay.addLayout(action_row)

                    def _copy():
                        try:
                            QApplication.clipboard().setText(str(cid))
                            copy_btn.setText("Copied")
                        except Exception:
                            pass

                    copy_btn.clicked.connect(_copy)
                    status_btn.clicked.connect(lambda: self.show_complaint_status_dialog(str(cid)))
                    close_btn2.clicked.connect(copy_box.accept)

                    copy_box.exec_()
                    dialog.accept()
                    return

                if result.get("status") == "rate_limited":
                    QMessageBox.warning(
                        dialog,
                        "Support",
                        "You have reached the limit of 10 complaints per minute for this Machine ID.\n\nPlease wait for 1 minute and try again.",
                    )
                    return

                error_box = QDialog(dialog)
                error_box.setWindowFlags(error_box.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                error_box.setWindowTitle("Help & Support")
                error_box.setMinimumWidth(520)
                error_box.setStyleSheet(dialog.styleSheet())

                lay = QVBoxLayout(error_box)
                lay.setContentsMargins(18, 18, 18, 18)
                lay.setSpacing(14)

                title2 = QLabel("Connection Required")
                title2.setAlignment(Qt.AlignCenter)
                title2.setStyleSheet(
                    "font: 900 14pt 'Segoe UI', Arial; color: white; "
                    "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #d92d20, stop:1 #f04438); "
                    "border-radius: 10px; padding: 10px;"
                )
                lay.addWidget(title2)

                msg = QLabel("You are offline, please connect to internet to submit the complaint.")
                msg.setObjectName("supportHint")
                msg.setWordWrap(True)
                msg.setStyleSheet("color: #7a271a; font: 11pt 'Segoe UI', Arial;")
                lay.addWidget(msg)

                action_row = QHBoxLayout()
                action_row.addStretch()
                close_btn2 = QPushButton("OK")
                close_btn2.setObjectName("supportCancel")
                close_btn2.setFixedWidth(110)
                action_row.addWidget(close_btn2)
                lay.addLayout(action_row)

                close_btn2.clicked.connect(error_box.accept)
                error_box.exec_()

            thread = _SubmitComplaintThread(payload, parent=dialog)
            dialog._support_submit_thread = thread  # keep alive
            thread.done.connect(_handle_result)
            thread.start()

        submit_btn.clicked.connect(_submit)

        dialog.exec_()

    def handle_sign_out(self):
        # User label removed per request
        # self.user_label.setText("Not signed in")
        self.sign_btn.setText("Sign In")
        self.closed_by_sign_out = True
        try:
            recorder = getattr(self, '_session_recorder', None)
            if recorder:
                recorder.close()
                self._session_recorder = None
        except Exception:
            pass
        
        # Clean up ECG test page serial connection before closing
        try:
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                # Stop acquisition if running
                try:
                    if hasattr(self.ecg_test_page, '_cleanup_report_generation_thread'):
                        self.ecg_test_page._cleanup_report_generation_thread(wait_ms=5000, force_terminate=True)
                    if hasattr(self.ecg_test_page, '_cleanup_start_worker'):
                        self.ecg_test_page._cleanup_start_worker(wait_ms=5000, force_terminate=True)
                except Exception as e:
                    print(f" Error stopping report thread on sign out: {e}")
                if hasattr(self.ecg_test_page, 'serial_reader') and self.ecg_test_page.serial_reader:
                    try:
                        print("Closing serial connection on sign out...")
                        self.ecg_test_page.serial_reader.stop()
                        self.ecg_test_page.serial_reader.close()
                        self.ecg_test_page.serial_reader = None
                        print(" Serial connection closed successfully")
                    except Exception as e:
                        print(f" Error closing serial connection: {e}")
                
                # Stop timers
                if hasattr(self.ecg_test_page, 'timer') and self.ecg_test_page.timer:
                    try:
                        self.ecg_test_page.timer.stop()
                    except Exception:
                        pass
                
                # Stop demo manager if active
                if hasattr(self.ecg_test_page, 'demo_manager') and self.ecg_test_page.demo_manager:
                    try:
                        self.ecg_test_page.demo_manager.stop_demo_data()
                    except Exception:
                        pass
                try:
                    self.ecg_test_page.close()
                except Exception:
                    pass
        except Exception as e:
            print(f" Error cleaning up ECG test page: {e}")
        
        self.close()

    def update_test_state(self, test_name, is_running):
        """Update the running state of a specific test."""
        if test_name in self.test_states:
            self.test_states[test_name] = is_running
            print(f" Test State Updated: {test_name} = {is_running}")
            # Force UI update if needed
            QApplication.processEvents()

    def can_start_test(self, test_name):
        """
        Check if a test can be started. 
        Automatically stops any other running test to allow seamless transitions.
        """
        for name, is_running in list(self.test_states.items()):
            if is_running and name != test_name:
                # Another test is running, stop it automatically
                print(f" Auto-stopping {name} to start {test_name}")
                if name == '12_lead_test' and hasattr(self, 'ecg_test_page'):
                    try:
                        self.ecg_test_page.stop_acquisition()
                    except Exception as e:
                        print(f" Error auto-stopping 12_lead_test: {e}")
                elif name == 'hrv_test' and hasattr(self, 'hrv_window'):
                    try:
                        self.hrv_window.stop_capture()
                    except Exception as e:
                        print(f" Error auto-stopping hrv_test: {e}")
                elif name == 'hyperkalemia_test' and hasattr(self, 'hyperkalemia_window'):
                    try:
                        self.hyperkalemia_window.stop_capture()
                    except Exception as e:
                        print(f" Error auto-stopping hyperkalemia_test: {e}")
                
                # Force state update in case the method failed to do it
                self.update_test_state(name, False)
                
        return True
        
    def open_hyperkalemia_test(self):
        """Open Hyperkalemia Test window in a new window"""
        # Stop 12-lead test AND Holter recording cleanly before opening Hyperkalemia test
        try:
            ecg_page = getattr(self, 'ecg_test_page', None)
            if ecg_page:
                # Stop acquisition first
                if hasattr(ecg_page, 'stop_acquisition'):
                    ecg_page.stop_acquisition()
                # Stop Holter recording if active
                if hasattr(ecg_page, 'stop_holter_recording'):
                    ecg_page.stop_holter_recording()
        except Exception as e:
            print(f" Error stopping 12-lead test: {e}")

        try:
            self.pause_dashboard_timers()
            self._reset_metrics_on_context_switch()   # reset labels + interpretation
            from ecg.hyperkalemia_test import HyperkalemiaTestWindow
            self.hyperkalemia_window = HyperkalemiaTestWindow(parent=self, username=self.username)
            self.hyperkalemia_window.showMaximized()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open Hyperkalemia Test window: {str(e)}")
            print(f" Error opening Hyperkalemia test: {e}")

    def open_history_window(self):
        """Open the ECG report history window."""
        try:
            from dashboard.history_window import HistoryWindow
            dlg = HistoryWindow(
                parent=self,
                username=self.username,
                owner_full_name=(self.user_details or {}).get("full_name") or self.username,
            )
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "History", f"Failed to open history window: {e}")
    
    def open_hrv_test(self):
        """Open HRV Test window in a new window"""
        # Stop 12-lead test AND Holter recording cleanly before opening HRV test
        try:
            ecg_page = getattr(self, 'ecg_test_page', None)
            if ecg_page:
                # Stop acquisition first
                if hasattr(ecg_page, 'stop_acquisition'):
                    ecg_page.stop_acquisition()
                # Stop Holter recording if active
                if hasattr(ecg_page, 'stop_holter_recording'):
                    ecg_page.stop_holter_recording()
        except Exception as e:
            print(f" Error stopping 12-lead test: {e}")

        # Pause timers BEFORE showing the dialog so that the metrics_timer
        # cannot fire during the dialog and restore old cached values after
        # stop_acquisition briefly triggers a lead-detach reset (visible flicker).
        self.pause_dashboard_timers()

        duration_minutes = None
        dlg = QDialog(self)
        dlg.setWindowTitle("HRV Test Duration")
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.setModal(True)
        dlg.setMinimumSize(420, 250)
        dlg.setStyleSheet("""
            QDialog { background: #f4f7f6; }
            QLabel { font: 900 14pt 'Segoe UI', Arial; color: #101828; }
            QPushButton { border-radius: 10px; padding: 12px 0; font: bold 12pt 'Segoe UI', Arial; border: none; }
            QPushButton#five { background: #ff6600; color: white; }
            QPushButton#five:hover { background: #e65c00; }
            QPushButton#three { background: #007bff; color: white; }
            QPushButton#three:hover { background: #0069d9; }
            QPushButton#cancel { background: #ffffff; color: #344054; border: 1px solid #d0d5dd; }
            QPushButton#cancel:hover { background: #f9fafb; }
        """)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(20)

        title = QLabel("Select HRV capture duration")
        title.setAlignment(Qt.AlignCenter)
        title.setWordWrap(True)
        outer.addWidget(title)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)

        btn_5 = QPushButton("5 mins")
        btn_5.setObjectName("five")
        btn_5.setCursor(Qt.PointingHandCursor)
        
        btn_3 = QPushButton("3 mins")
        btn_3.setObjectName("three")
        btn_3.setCursor(Qt.PointingHandCursor)

        def _pick(minutes):
            nonlocal duration_minutes
            duration_minutes = minutes
            dlg.accept()

        btn_5.clicked.connect(lambda: _pick(5))
        btn_3.clicked.connect(lambda: _pick(3))

        btn_row.addWidget(btn_5)
        btn_row.addWidget(btn_3)
        outer.addLayout(btn_row)

        cancel = QPushButton("Cancel")
        cancel.setObjectName("cancel")
        cancel.setCursor(Qt.PointingHandCursor)
        cancel.clicked.connect(dlg.reject)
        outer.addWidget(cancel)

        if dlg.exec_() != QDialog.Accepted or not duration_minutes:
            # User cancelled — restore dashboard to normal operation
            self.resume_dashboard_timers()
            return

        try:
            self._reset_metrics_on_context_switch()   # reset labels + interpretation
            from ecg.hrv_test import HRVTestWindow
            self.hrv_window = HRVTestWindow(parent=self, username=self.username, duration_minutes=duration_minutes)
            self.hrv_window.showMaximized()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open HRV Test window: {str(e)}")
            print(f" Error opening HRV test: {e}")

    
    def go_to_lead_test(self):
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)

        if hasattr(self, 'ecg_test_page'):
            self.ecg_test_page.current_username = self.username

        self.page_stack.setCurrentWidget(self.ecg_test_page)
        # Sync dashboard metrics to ECG test page
        self.sync_dashboard_metrics_to_ecg_page()
        # Also update dashboard metrics when opening ECG test page
        self.update_dashboard_metrics_from_ecg()

        # ── AUTO-START: send Start command if acquisition is not already running ──
        # This mirrors what the doctor would do by clicking the "Start" button,
        # so the 12-lead ECG view begins streaming the moment it opens.
        try:
            ecg = self.ecg_test_page
            timer_active = (hasattr(ecg, 'timer') and ecg.timer is not None and ecg.timer.isActive())
            reader_running = (hasattr(ecg, 'serial_reader') and ecg.serial_reader is not None
                              and getattr(ecg.serial_reader, 'running', False))
            demo_active = (hasattr(ecg, 'demo_toggle') and ecg.demo_toggle is not None
                           and ecg.demo_toggle.isChecked())
            # IMPORTANT: Require BOTH timer active AND reader running to skip auto-start.
            # Using OR would incorrectly skip auto-start when timer was stopped by a
            # stop_acquisition call (e.g., triggered after HRV/Hyperkalemia sends a
            # STOP command to hardware) but serial_reader.running is still True.
            already_running = demo_active or (timer_active and reader_running)
            if not already_running and hasattr(ecg, 'start_acquisition'):
                # Small delay so the page fully renders before the connection
                # worker starts (prevents UI jank on slow machines)
                from PyQt5.QtCore import QTimer as _QT
                _QT.singleShot(150, ecg.start_acquisition)
                print("[Dashboard] Auto-started 12-lead acquisition on page switch")
        except Exception as _ae:
            print(f"[Dashboard] Auto-start skipped: {_ae}")
        # ─────────────────────────────────────────────────────────────────────────


    def go_to_dashboard(self):
        # # Close serial connection on ECG page to free up COM port
        # if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
        #     try:
        #         if hasattr(self.ecg_test_page, 'close_serial_connection'):
        #             self.ecg_test_page.close_serial_connection()
        #     except Exception as e:
        #         print(f"Error closing serial connection: {e}")

                
        self.page_stack.setCurrentWidget(self.dashboard_page)
        # Update metrics when returning to dashboard
        self.update_dashboard_metrics_from_ecg()

    # Resume device check if needed
    def check_device_connection(self):
        """Check for device connection or scan if disconnected"""

        if not SERIAL_AVAILABLE:
            return

        # Never let the dashboard probe the serial bus while any test page/window is
        # already streaming live data. Scanning the shared port here can steal
        # or stall the active reader and make the waveform appear frozen.
        active_reader = None
        ecg_page = getattr(self, 'ecg_test_page', None)
        if ecg_page and getattr(getattr(ecg_page, 'serial_reader', None), 'running', False):
            active_reader = ecg_page.serial_reader
        elif hasattr(self, 'hyperkalemia_window') and self.hyperkalemia_window and getattr(getattr(self.hyperkalemia_window, 'serial_reader', None), 'running', False):
            active_reader = self.hyperkalemia_window.serial_reader
        elif hasattr(self, 'hrv_window') and self.hrv_window and getattr(getattr(self.hrv_window, 'serial_reader', None), 'running', False):
            active_reader = self.hrv_window.serial_reader

        if active_reader is not None:
            try:
                live_port = getattr(getattr(active_reader, 'ser', None), 'port', None)
                if live_port:
                    self.device_port = live_port
                if not self.device_connected:
                    self.device_connected = True
                    self.update_device_ui(True)
            except Exception:
                pass
            return


        try:
            # Get current available ports
            current_ports = [p.device for p in serial.tools.list_ports.comports()]
            
            # If device is already connected, check if it's still there
            if self.device_connected and self.device_port:
                if self.device_port not in current_ports:
                    print(f"⚠️ Port {self.device_port} disconnected.")
                    self.device_connected = False
                    self.device_port = None
                    self.update_device_ui(False)
                    # Update port tracking to current state to avoid immediate re-scan
                    self._last_available_ports = current_ports
                    
                    # Inform user if on dashboard and no tests active
                    is_on_dashboard = self.page_stack.currentWidget() == self.dashboard_page
                    hrv_active = hasattr(self, 'hrv_window') and self.hrv_window and self.hrv_window.isVisible()
                    hyper_active = hasattr(self, 'hyperkalemia_window') and self.hyperkalemia_window and self.hyperkalemia_window.isVisible()
                    
                    if is_on_dashboard and not hrv_active and not hyper_active:
                        StyledMessageBox.show_message(self, "Connection Status", "RhythmUltra connection lost. Please ensure the device is properly connected")

                    # If HRV or Hyperkalemia test window is open, show "Test Failed" and close it
                    if hasattr(self, 'hrv_window') and self.hrv_window and self.hrv_window.isVisible():
                        if hasattr(self.hrv_window, 'stop_capture'):
                            try:
                                self.hrv_window.stop_capture(device_disconnected=True)
                            except TypeError:
                                self.hrv_window.stop_capture()
                        StyledMessageBox.show_message(self.hrv_window, "Test Failed", "RhythmUltra disconnected. Test failed.", is_critical=True)
                        self.hrv_window.close()
                    elif hasattr(self, 'hyperkalemia_window') and self.hyperkalemia_window and self.hyperkalemia_window.isVisible():
                        if hasattr(self.hyperkalemia_window, 'stop_capture'):
                            try:
                                self.hyperkalemia_window.stop_capture(device_disconnected=True)
                            except TypeError:
                                self.hyperkalemia_window.stop_capture()
                        StyledMessageBox.show_message(self.hyperkalemia_window, "Test Failed", "RhythmUltra disconnected. Test failed.", is_critical=True)
                        self.hyperkalemia_window.close()
                    # If ECG 12 lead test is running (on the stacked widget), show "Test Failed" and go back to dashboard
                    elif self.page_stack.currentWidget() == getattr(self, 'ecg_test_page', None):
                        if hasattr(self.ecg_test_page, 'stop_acquisition'):
                            self.ecg_test_page.stop_acquisition()

                        # Close any open expanded lead view dialogs
                        try:
                            for widget in QApplication.topLevelWidgets():
                                if widget.__class__.__name__ == 'ExpandedLeadView':
                                    widget.close()
                        except Exception as e:
                            print(f"Error closing expanded views: {e}")

                        StyledMessageBox.show_message(self, "Test Failed", "RhythmUltra disconnected. Test failed.", is_critical=True)
                        self.page_stack.setCurrentWidget(self.dashboard_page)
                return # Already connected and port exists, or just disconnected

            # Not connected, only scan if the ports list has changed (new device plugged in)
            if set(current_ports) != set(self._last_available_ports):
                print(f"COM port change detected: {self._last_available_ports} -> {current_ports}")
                self._last_available_ports = current_ports
                
                # If a new port was added, try to scan
                if len(current_ports) > 0:
                    # Show searching status while scan is in progress
                    if hasattr(self, 'device_status_label'):
                        self.device_status_label.setText("Searching for RhythmUltra...")
                        self.device_status_label.setStyleSheet("color: orange; margin-right: 10px; font-weight: bold;")

                    # Only scan if not already in progress
                    if not getattr(self, "_device_scan_in_progress", False):
                        self._device_scan_in_progress = True
                        
                        # Use DeviceScanWorker to scan in background
                        self.scan_worker = DeviceScanWorker(self.settings_manager)
                        self.scan_worker.scan_finished.connect(self.on_scan_finished)
                        self.scan_worker.start()
                    else:
                        # Ports changed while scanning; request an immediate rescan when current scan finishes.
                        self._device_rescan_requested = True
            
        except Exception as e:
            print(f"Error in check_device_connection: {e}")

        # Skip if any test window is open (HRV or Hyperkalemia)
        if hasattr(self, 'hrv_window') and self.hrv_window and self.hrv_window.isVisible():
            return
        if hasattr(self, 'hyperkalemia_window') and self.hyperkalemia_window and self.hyperkalemia_window.isVisible():
            return
        # Skip if on ECG Test Page (stacked widget)
        if self.page_stack.currentWidget() == getattr(self, 'ecg_test_page', None):
            return

        # If the device is not yet connected, keep retrying even when the port
        # list hasn't changed. Some macOS USB serial adapters need a few probes
        # before the device answers the VERSION/MACHINE_SERIAL commands.
        if not self.device_connected and len(current_ports) > 0:
            now = time.time()
            if getattr(self, "_device_scan_in_progress", False):
                return
            if (now - getattr(self, "_last_device_scan_time", 0)) < 1.0:
                return

            self._last_device_scan_time = now
            if hasattr(self, 'device_status_label'):
                self.device_status_label.setText("Searching for RhythmUltra...")
                self.device_status_label.setStyleSheet("color: orange; margin-right: 10px; font-weight: bold;")

            self._device_scan_in_progress = True
            self.scan_worker = DeviceScanWorker(self.settings_manager)
            self.scan_worker.scan_finished.connect(self.on_scan_finished)
            self.scan_worker.start()
            return

    def on_scan_finished(self, success, port, version, serial_num):
        """Callback for background device scan"""
        was_initial_scan = not getattr(self, '_initial_scan_completed', False)
        self._device_scan_in_progress = False
        self._initial_scan_completed = True

        # If ports changed during the previous scan, run one more scan immediately.
        if not success and getattr(self, "_device_rescan_requested", False):
            self._device_rescan_requested = False
            self._device_scan_in_progress = True
            self.scan_worker = DeviceScanWorker(self.settings_manager)
            self.scan_worker.scan_finished.connect(self.on_scan_finished)
            self.scan_worker.start()
            return

        if success:
            self._had_device_connected = True
            self._last_device_scan_time = time.time()
            
            # Fresh start: reset all metrics, buffers, and interpretation when device reconnects!
            self.reset_metrics_and_interpretation()
            
            # Also reset 12-lead test's data if it exists
            ecg_page = getattr(self, 'ecg_test_page', None)
            if ecg_page:
                ecg_page._latest_rhythm_interpretation = "Analyzing Rhythm..."
                ecg_page.last_heart_rate = 0
                ecg_page.last_pr_interval = 0
                ecg_page.last_qrs_duration = 0
                ecg_page.last_qt_interval = 0
                ecg_page.last_qtc_interval = 0
                if hasattr(ecg_page, 'data'):
                    ecg_page.data = [[] for _ in range(12)]
                if hasattr(ecg_page, 'metric_labels'):
                    if 'heart_rate' in ecg_page.metric_labels:
                        ecg_page.metric_labels['heart_rate'].setText("0 BPM")
                    if 'pr_interval' in ecg_page.metric_labels:
                        ecg_page.metric_labels['pr_interval'].setText("0 ms")
                    if 'qrs_duration' in ecg_page.metric_labels:
                        ecg_page.metric_labels['qrs_duration'].setText("0 ms")
                    if 'qtc_interval' in ecg_page.metric_labels:
                        ecg_page.metric_labels['qtc_interval'].setText("--")
            
            # Inform user if not the initial scan
            if not was_initial_scan:
                msg = QMessageBox(self)
                msg.setWindowTitle("Connection Status")
                msg.setText("Device connected")
                msg.setStandardButtons(QMessageBox.NoButton)
                try:
                    icon_path = get_asset_path("connection_icon_tick.png")
                    pix = QPixmap(icon_path)
                    if not pix.isNull():
                        msg.setIconPixmap(pix)
                    else:
                        msg.setIcon(QMessageBox.Information)
                except Exception:
                    msg.setIcon(QMessageBox.Information)
                # Auto-close after 0.5 seconds — no need for user to press OK
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(500, msg.accept)
                msg.exec_()

            # Update hardware version if it's different from current
            if self.device_version != version:
                print(f"Hardware version changed from {self.device_version} to {version}")
                self.device_version = version
            
            # Update machine serial number if provided
            if serial_num:
                self.machine_serial_number = serial_num
                print(f"Machine serial number detected: {serial_num}")

            self.device_port = port
            self.device_connected = True
            self.update_device_ui(True)

            # Save to settings so test pages can use it
            if hasattr(self, 'settings_manager'):
                self.settings_manager.set_setting("serial_port", port)
                self.settings_manager.set_setting("baud_rate", "115200")
                self.settings_manager.set_setting("hardware_version", version)
                if serial_num:
                    self.settings_manager.set_setting("machine_serial_number", serial_num)
                self.settings_manager.save_settings()
                print(f"✅ RhythmUltra found on {port} and saved to settings with version {version} and serial {serial_num or 'N/A'}.")
        else:
            self._last_device_scan_time = time.time()
            self.update_device_ui(False)

    def update_device_ui(self, connected):
        """Update UI elements based on device connection status"""
        if connected:
            self.device_status_label.setText("RhythmUltra Connected")
            self.device_status_label.setStyleSheet("color: green; margin-right: 10px; font-weight: bold;")
            
            # Enable test buttons only if license is not read-only
            if not getattr(self, "_license_read_only", False):
                if hasattr(self, 'hrv_test_btn'):
                    self.hrv_test_btn.setEnabled(True)
                    self.hrv_test_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 16px; padding: 8px 24px;")

                if hasattr(self, 'hyperkalemia_test_btn'):
                    self.hyperkalemia_test_btn.setEnabled(True)
                    self.hyperkalemia_test_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 16px; padding: 8px 24px;")

                if hasattr(self, 'date_btn'): # ECG Lead Test 12
                    self.date_btn.setEnabled(True)
                    self.date_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 16px; padding: 8px 24px;")
            else:
                grey_style = "background: #cccccc; color: #666666; border-radius: 16px; padding: 8px 24px;"
                if hasattr(self, 'hrv_test_btn'):
                    self.hrv_test_btn.setEnabled(False)
                    self.hrv_test_btn.setStyleSheet(grey_style)
                if hasattr(self, 'hyperkalemia_test_btn'):
                    self.hyperkalemia_test_btn.setEnabled(False)
                    self.hyperkalemia_test_btn.setStyleSheet(grey_style)
                if hasattr(self, 'date_btn'):
                    self.date_btn.setEnabled(False)
                    self.date_btn.setStyleSheet(grey_style)
        else:
            if getattr(self, "_had_device_connected", False):
                self.device_status_label.setText("RhythmUltra Disconnected")
            else:
                self.device_status_label.setText("RhythmUltra Disconnected")
            self.device_status_label.setStyleSheet("color: red; margin-right: 10px; font-weight: bold;")

            # Reset hardware version in settings when disconnected
            if hasattr(self, 'settings_manager'):
                self.settings_manager.set_setting("hardware_version", "")
                self.settings_manager.set_setting("machine_serial_number", "")
            self.device_version = None
            self.machine_serial_number = None
            
            # Disable test buttons
            grey_style = "background: #cccccc; color: #666666; border-radius: 16px; padding: 8px 24px;"

            if hasattr(self, 'hrv_test_btn'):
                self.hrv_test_btn.setEnabled(False)
                self.hrv_test_btn.setStyleSheet(grey_style)

            if hasattr(self, 'hyperkalemia_test_btn'):
                self.hyperkalemia_test_btn.setEnabled(False)
                self.hyperkalemia_test_btn.setStyleSheet(grey_style)

            if hasattr(self, 'date_btn'):
                self.date_btn.setEnabled(False)
                self.date_btn.setStyleSheet(grey_style)

            # Reset metrics and interpretation when disconnected
            if hasattr(self, 'metric_labels'):
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText("0 BPM")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText("0 ms")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText("0 ms")
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("--")
            
            self.current_heart_rate = 0
            self._last_bpm = None
            self._last_stable_rr = None
            self._rr_stability_counter = 0
            
            # Reset cached metric values
            self._last_hr = None
            self._last_pr = None
            self._last_qrs = None
            self._last_p = None
            self._last_qtc = None
            
            if hasattr(self, 'conclusion_box'):
                self.conclusion_box.setHtml("""
                    <p style='color: #888; font-style: italic;'>
                    No ECG data available yet.<br><br>
                    Start an ECG test to see your personalized analysis and recommendations.
                    </p>
                """)

    def set_license_banner(self, status: str, detail: str = "", color: str = "#1b5e20", background: str = "#dcedc8"):
        """Update the footer license status indicator."""
        try:
            status_text = (status or "ONLINE").strip().upper()
            if status_text == "ONLINE":
                label_text = "● Software Activated"
                text_color = "#3dba6e"
            elif status_text == "VERIFICATION REQUIRED":
                label_text = "● Verification Required"
                text_color = "#e05555"
            else:
                label_text = f"● {status_text.title()}"
                text_color = "#aaaaaa"
            detail_text = f"  {detail.strip()}" if detail and detail.strip() else ""
            self.license_status_label.setText(f"{label_text}{detail_text}")
            self.license_status_label.setStyleSheet(f"""
                QLabel {{
                    color: {text_color};
                    background: transparent;
                    font-size: 8pt;
                    font-weight: 700;
                    padding: 0px 4px;
                    letter-spacing: 0.4px;
                }}
            """)
        except Exception:
            pass

    def set_read_only_license_mode(self, enabled: bool, reason: str = ""):
        """Disable acquisition controls while keeping history/report access available."""
        try:
            self._license_read_only = bool(enabled)
            grey_style = "background: #cccccc; color: #666666; border-radius: 16px; padding: 8px 24px;"
            if enabled:
                if hasattr(self, 'date_btn'):
                    self.date_btn.setEnabled(False)
                    self.date_btn.setStyleSheet(grey_style)
                    self.date_btn.setToolTip(reason or "New ECG acquisition is disabled in read-only mode.")
                if hasattr(self, 'hrv_test_btn'):
                    self.hrv_test_btn.setEnabled(False)
                    self.hrv_test_btn.setStyleSheet(grey_style)
                    self.hrv_test_btn.setToolTip(reason or "HRV acquisition is disabled in read-only mode.")
                if hasattr(self, 'hyperkalemia_test_btn'):
                    self.hyperkalemia_test_btn.setEnabled(False)
                    self.hyperkalemia_test_btn.setStyleSheet(grey_style)
                    self.hyperkalemia_test_btn.setToolTip(reason or "Hyperkalemia acquisition is disabled in read-only mode.")
                if hasattr(self, 'analysis_btn'):
                    self.analysis_btn.setEnabled(True)
                self.set_license_banner("VERIFICATION REQUIRED", reason or "Restricted mode", color="#c62828", background="#ffebee")
            else:
                if hasattr(self, 'date_btn') and not self.device_connected:
                    self.date_btn.setEnabled(False)
                    self.date_btn.setStyleSheet(grey_style)
                elif hasattr(self, 'date_btn') and self.device_connected:
                    self.date_btn.setEnabled(True)
                    self.date_btn.setStyleSheet("background: #ff6600; color: white; border-radius: 16px; padding: 8px 24px;")

                if hasattr(self, 'hrv_test_btn') and not self.device_connected:
                    self.hrv_test_btn.setEnabled(False)
                    self.hrv_test_btn.setStyleSheet(grey_style)
                elif hasattr(self, 'hrv_test_btn') and self.device_connected:
                    self.hrv_test_btn.setEnabled(True)
                    self.hrv_test_btn.setStyleSheet("background: #dc3545; color: white; border-radius: 16px; padding: 8px 24px;")

                if hasattr(self, 'hyperkalemia_test_btn') and not self.device_connected:
                    self.hyperkalemia_test_btn.setEnabled(False)
                    self.hyperkalemia_test_btn.setStyleSheet(grey_style)
                elif hasattr(self, 'hyperkalemia_test_btn') and self.device_connected:
                    self.hyperkalemia_test_btn.setEnabled(True)
                    self.hyperkalemia_test_btn.setStyleSheet("background: #d2691e; color: white; border-radius: 16px; padding: 8px 24px;")

                if hasattr(self, 'analysis_btn'):
                    self.analysis_btn.setEnabled(True)
                self.set_license_banner("ONLINE", "", color="#1b5e20", background="#dcedc8")
        except Exception as e:
            print(f"Error setting read-only license mode: {e}")

    def update_internet_status(self):
        """Check internet status in background — never freeze UI."""
        # FIX: was socket.create_connection(timeout=2) on main thread → 2s freeze
        try:
            lock = getattr(self, "_inet_check_lock", None)
            if lock is not None and not lock.acquire(False):
                return  # previous check still running
        except Exception:
            lock = None

        def _check():
            import socket
            try:
                online = False
                source = "none"

                # Prefer OS-connected state on Windows (works even if DNS/53 to 8.8.8.8 is blocked).
                try:
                    import sys
                    if sys.platform.startswith("win"):
                        import ctypes
                        flags = ctypes.c_ulong()
                        online = bool(ctypes.windll.wininet.InternetGetConnectedState(ctypes.byref(flags), 0))
                        if online:
                            source = "wininet"
                except Exception:
                    online = False

                # Windows fallback: treat "has a non-loopback IP" as connected (Wi‑Fi can be up even if ping/DNS blocked).
                if not online:
                    try:
                        import sys
                        if sys.platform.startswith("win"):
                            # Get all local IP addresses in a pure-Python, zero-subprocess way.
                            # This completely avoids spawning cmd.exe or ipconfig.exe, eliminating any console flashes.
                            hostname = socket.gethostname()
                            candidates = set()
                            
                            try:
                                for info in socket.getaddrinfo(hostname, None):
                                    ip = info[4][0]
                                    if ip:
                                        candidates.add(ip)
                            except Exception:
                                pass
                                
                            try:
                                for ip in socket.gethostbyname_ex(hostname)[2]:
                                    if ip:
                                        candidates.add(ip)
                            except Exception:
                                pass
                                
                            for ip in candidates:
                                ip_lower = ip.lower()
                                if ip_lower.startswith("127.") or ip_lower == "::1" or "loopback" in ip_lower:
                                    continue
                                online = True
                                source = "socket_local"
                                break
                    except Exception:
                        online = False

                # Fallback 1: quick HTTPS connect (more likely allowed than DNS/53 in some networks)
                if not online:
                    try:
                        socket.create_connection(("www.google.com", 443), timeout=1.0)
                        online = True
                        source = "tcp443"
                    except Exception:
                        online = False

                # Fallback 2: try quick socket connects (best-effort "internet reachable")
                if not online:
                    for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53)):
                        try:
                            socket.create_connection((host, port), timeout=0.6)
                            online = True
                            source = f"tcp{port}"
                            break
                        except Exception:
                            continue
            except Exception:
                online = False
                source = "error"

            try:
                # Emit from worker thread; Qt delivers to main thread (queued connection).
                self.internet_status_changed.emit(bool(online), str(source))
            except Exception:
                pass
            finally:
                try:
                    if lock is not None:
                        lock.release()
                except Exception:
                    pass
        import threading
        threading.Thread(target=_check, daemon=True, name="InetStatus").start()

    def _apply_internet_status(self, online: bool, source: str = "unknown"):
        try:
            if online:
                if hasattr(self, "_wifi_pixmap_connected") and not self._wifi_pixmap_connected.isNull():
                    self.status_dot.setPixmap(self._wifi_pixmap_connected)
                    self.status_dot.setStyleSheet("background: transparent;")
                else:
                    self.status_dot.setPixmap(QPixmap())
                    self.status_dot.setStyleSheet(
                        "border-radius: 9px; background: #00e676; border: 2px solid #fff;"
                    )
                self.status_dot.setToolTip(f"Wi-Fi/Internet connected ({source})")
            else:
                if hasattr(self, "_wifi_pixmap_disconnected") and not self._wifi_pixmap_disconnected.isNull():
                    self.status_dot.setPixmap(self._wifi_pixmap_disconnected)
                    self.status_dot.setStyleSheet("background: transparent;")
                else:
                    self.status_dot.setPixmap(QPixmap())
                    self.status_dot.setStyleSheet(
                        "border-radius: 9px; background: #e74c3c; border: 2px solid #fff;"
                    )
                self.status_dot.setToolTip(f"Wi-Fi/Internet disconnected ({source})")
        except Exception:
            pass

    def _load_status_pixmap(self, filename: str, target_px: int = 18) -> QPixmap:
        try:
            path = get_asset_path(filename)
            pm = QPixmap(path)
            if pm.isNull():
                return QPixmap()
            return pm.scaled(target_px, target_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception:
            return QPixmap()

    def _load_status_pixmap_any(self, filenames: list[str], target_px: int = 18) -> QPixmap:
        for name in filenames:
            pm = self._load_status_pixmap(name, target_px=target_px)
            if not pm.isNull():
                return pm
        return QPixmap()

    def _make_wifi_disconnected_pixmap(self, base: QPixmap, target_px: int = 18) -> QPixmap:
        try:
            if base.isNull():
                return QPixmap()
            pm = base.scaled(target_px, target_px, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img = pm.toImage().convertToFormat(QImage.Format_Grayscale8).convertToFormat(QImage.Format_ARGB32)
            pm = QPixmap.fromImage(img)
            painter = QPainter(pm)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor("#e74c3c"), 2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(2, pm.height() - 3, pm.width() - 3, 2)
            painter.end()
            return pm
        except Exception:
            return QPixmap()
    def toggle_medical_mode(self):
        self.medical_mode = not self.medical_mode
        if self.medical_mode:
            # Medical color coding: blue/green/white (previous behavior)
            self.setStyleSheet("QWidget { background: #e3f6fd; } QFrame { background: #f8fdff; border-radius: 16px; } QLabel { color: #006266; }")
            self.medical_btn.setText("Normal Mode")
            self.medical_btn.setStyleSheet("background: #0984e3; color: white; border-radius: 10px; padding: 4px 18px;")
        else:
            self.setStyleSheet("")
            self.medical_btn.setText("Medical Mode")
            self.medical_btn.setStyleSheet("background: #00b894; color: white; border-radius: 10px; padding: 4px 18px;")
        # Update ECG test page theme if it exists
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
            
    def open_admin_reports(self):
        try:
            login = AdminLoginDialog(self)
            if login.exec_() == QDialog.Accepted:
                from utils.cloud_uploader import get_cloud_uploader
                cu = get_cloud_uploader()
                cu.reload_config()
                dlg = AdminReportsDialog(cu, self)
                dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unable to open admin reports: {e}")

    def auto_sync_to_cloud(self):
        """Background auto-backup of reports/metrics. Never blocks main thread."""
        if getattr(self, '_cloud_sync_in_progress', False):
            return

        def _bg_upload():
            # FIX: ALL cloud work here — socket + upload off main thread
            try:
                import socket, os, glob, json
                # Quick connectivity check (0.5s max)
                try:
                    socket.create_connection(("8.8.8.8", 53), timeout=0.5)
                except Exception:
                    return  # No internet — silent return

                from utils.cloud_uploader import get_cloud_uploader
                cloud_uploader = get_cloud_uploader()
                if not cloud_uploader.is_configured():
                    return

                reports_dir = str(data_file("reports"))
                os.makedirs(reports_dir, exist_ok=True)
                uploaded_names = set()
                try:
                    history = cloud_uploader.get_upload_history(limit=1000)
                    for item in history:
                        path = item.get('local_path') or ''
                        if path:
                            uploaded_names.add(os.path.basename(path))
                except Exception:
                    pass

                candidates = []
                candidates += glob.glob(os.path.join(reports_dir, "ECG_Report_*.pdf"))
                candidates += [p for p in glob.glob(os.path.join(reports_dir, "*.json"))
                               if ('report' in os.path.basename(p).lower() or
                                   'metric' in os.path.basename(p).lower())]
                pending = [p for p in candidates
                           if os.path.basename(p) not in uploaded_names]
                if not pending:
                    return

                self._cloud_sync_in_progress = True
                for path in pending:
                    try:
                        cloud_uploader.upload_report(path)
                    except Exception:
                        pass
            except Exception as e:
                print(f"Auto-sync error: {e}")
            finally:
                self._cloud_sync_in_progress = False

        import threading
        t = threading.Thread(target=_bg_upload, daemon=True, name="AutoCloudSync")
        t.start()


    def sync_to_cloud(self):
        """Sync ECG reports and metrics to AWS S3"""
        try:
            from utils.cloud_uploader import get_cloud_uploader
            from PyQt5.QtWidgets import QMessageBox
            
            cloud_uploader = get_cloud_uploader()
            # Re-read .env in case the app was launched before keys were added
            try:
                cloud_uploader.reload_config()
                # As an extra safeguard, read .env directly and override fields
                try:
                    from dotenv import dotenv_values
                    from pathlib import Path as _P
                    root = _P(__file__).resolve().parents[2]
                    cfg = dotenv_values(str(root / '.env'))
                    if cfg:
                        cloud_uploader.cloud_service = (cfg.get('CLOUD_SERVICE') or cloud_uploader.cloud_service or 'none').lower()
                        cloud_uploader.upload_enabled = (str(cfg.get('CLOUD_UPLOAD_ENABLED') or cloud_uploader.upload_enabled).lower() == 'true')
                        cloud_uploader.s3_bucket = cfg.get('AWS_S3_BUCKET') or cloud_uploader.s3_bucket
                        cloud_uploader.s3_region = cfg.get('AWS_S3_REGION') or cloud_uploader.s3_region
                        cloud_uploader.aws_access_key = cfg.get('AWS_ACCESS_KEY_ID') or cloud_uploader.aws_access_key
                        cloud_uploader.aws_secret_key = cfg.get('AWS_SECRET_ACCESS_KEY') or cloud_uploader.aws_secret_key
                        _env_path_used = str(root / '.env')
                    else:
                        _env_path_used = '(not found)'
                except Exception:
                    _env_path_used = '(error reading .env)'
            except Exception:
                _env_path_used = '(reload failed)'
            
            if not cloud_uploader.is_configured():
                QMessageBox.warning(
                    self, 
                    "Cloud Not Configured",
                    (
                        "AWS S3 is not configured.\n\nCurrent values read:\n"
                        f"CLOUD_SERVICE={getattr(cloud_uploader,'cloud_service','')}\n"
                        f"CLOUD_UPLOAD_ENABLED={getattr(cloud_uploader,'upload_enabled','')}\n"
                        f"AWS_S3_BUCKET={getattr(cloud_uploader,'s3_bucket','')}\n"
                        f"AWS_S3_REGION={getattr(cloud_uploader,'s3_region','')}\n"
                        f"AWS_ACCESS_KEY_ID set?={'yes' if getattr(cloud_uploader,'aws_access_key',None) else 'no'}\n"
                        f"AWS_SECRET_ACCESS_KEY set?={'yes' if getattr(cloud_uploader,'aws_secret_key',None) else 'no'}\n"
                        f".env path tried: {_env_path_used}\n\n"
                        "Fix: Create .env in project root with:\n"
                        "CLOUD_UPLOAD_ENABLED=true\nCLOUD_SERVICE=s3\n"
                        "AWS_S3_BUCKET=your-bucket-name\nAWS_S3_REGION=us-east-1\n"
                        "AWS_ACCESS_KEY_ID=...\nAWS_SECRET_ACCESS_KEY=...\n\n"
                        "See AWS_REPORTS_ONLY_SETUP.md for details."
                    )
                )
                return
            
            # Show progress
            self.cloud_sync_btn.setText("Syncing...")
            self.cloud_sync_btn.setEnabled(False)
            
            # Find and upload all report files
            import glob
            reports_dir = str(data_file("reports"))
            uploaded_count = 0
            errors = []
            
            # Build file lists (non-blocking upload in background thread)
            pdf_reports = glob.glob(os.path.join(reports_dir, "ECG_Report_*.pdf"))
            json_reports = [p for p in glob.glob(os.path.join(reports_dir, "*.json"))
                            if 'report' in os.path.basename(p).lower() or 'metric' in os.path.basename(p).lower()]

            files_to_upload = pdf_reports + json_reports

            import threading
            def _do_upload():
                nonlocal uploaded_count, errors
                try:
                    for path in files_to_upload:
                        result = cloud_uploader.upload_report(path)
                        if result.get('status') == 'success':
                            uploaded_count += 1
                        elif result.get('status') != 'skipped':
                            errors.append(f"{os.path.basename(path)}: {result.get('message', 'Unknown error')}")
                finally:
                    # Restore UI safely on the main thread
                    try:
                        self.cloud_sync_btn.setText("Cloud Sync")
                        self.cloud_sync_btn.setEnabled(True)
                        if uploaded_count > 0:
                            msg = f" Successfully uploaded {uploaded_count} file(s) to AWS S3!"
                            if errors:
                                msg += f"\n\n{len(errors)} error(s):\n" + "\n".join(errors[:3])
                            QMessageBox.information(self, "Cloud Sync Complete", msg)
                        else:
                            QMessageBox.information(self, "No Files to Sync", "No report files found in the reports directory.")
                    except Exception:
                        pass

            t = threading.Thread(target=_do_upload, daemon=True)
            t.start()
                
        except Exception as e:
            self.cloud_sync_btn.setText("Cloud Sync")
            self.cloud_sync_btn.setEnabled(True)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Sync Error",
                f"Failed to sync to cloud:\n{str(e)}"
            )
            print(f" Cloud sync error: {e}")
    
    def apply_dark_theme(self):
        """Apply dark theme styling to all UI components"""
        self.setStyleSheet("""
            QWidget { background: #181818; color: #fff; }
            QFrame { background: #232323 !important; border-radius: 16px; color: #fff; border: 2px solid #fff; }
            QLabel { color: #fff; }
            QPushButton { background: #333; color: #ff6600; border-radius: 10px; }
            QPushButton:checked { background: #ff6600; color: #fff; }
            QCalendarWidget QWidget { background: #232323; color: #fff; }
            QCalendarWidget QAbstractItemView { background: #232323; color: #fff; selection-background-color: #444; selection-color: #ff6600; }
            QTextEdit { background: #232323; color: #fff; border-radius: 12px; border: 2px solid #fff; }
        """)
        self.dark_btn.setText("Light Mode")
        # Set matplotlib canvas backgrounds to dark
        self.ecg_canvas.axes.set_facecolor("#232323")
        self.ecg_canvas.figure.set_facecolor("#232323")
        for child in self.findChildren(QFrame):
            child.setStyleSheet("background: #232323; border-radius: 16px; color: #fff; border: 2px solid #fff;")
        for key, label in self.metric_labels.items():
            label.setStyleSheet("color: #fff; background: transparent;")
        for canvas in self.findChildren(MplCanvas):
            canvas.axes.set_facecolor("#232323")
            canvas.figure.set_facecolor("#232323")
            canvas.draw()
        for calendar in self.findChildren(QCalendarWidget):
            calendar.setStyleSheet("background: #232323; color: #fff; border-radius: 12px; border: 2px solid #fff;")
        for txt in self.findChildren(QTextEdit):
                    txt.setStyleSheet("background: #232323; color: #fff; border-radius: 12px; border: 2px solid #fff;")
        # Update ECG test page theme if it exists
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
    
    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            self.apply_dark_theme()
        else:
            self.setStyleSheet("")
            self.dark_btn.setText("Dark Mode")
            self.ecg_canvas.axes.set_facecolor("#eee")
            self.ecg_canvas.figure.set_facecolor("#fff")
            for child in self.findChildren(QFrame):
                child.setStyleSheet("")
            for key, label in self.metric_labels.items():
                label.setStyleSheet("color: #222; background: transparent;")
                for canvas in child.findChildren(MplCanvas):
                    canvas.axes.set_facecolor("#fff")
                    canvas.figure.set_facecolor("#fff")
                    canvas.draw()
                for self.schedule_calendar in child.findChildren(QCalendarWidget):
                    self.schedule_calendar.setStyleSheet("")
                for txt in child.findChildren(QTextEdit):
                    txt.setStyleSheet("")
        # Update ECG test page theme if it exists
        if hasattr(self, 'ecg_test_page') and hasattr(self.ecg_test_page, 'update_metrics_frame_theme'):
            self.ecg_test_page.update_metrics_frame_theme(self.dark_mode, self.medical_mode)
    
    def test_asset_paths(self):
        """
        Test all asset paths at startup to ensure they're working correctly.
        This helps with debugging path issues.
        """
        print("=== Testing Asset Paths ===")
        
        # Test common assets
        test_assets = ["her.png", "v.gif", "plasma.gif", "ECG1.png"]
        
        for asset in test_assets:
            path = get_asset_path(asset)
            exists = os.path.exists(path)
            print(f"{asset}: {'✓' if exists else '✗'} - {path}")
            
            if not exists:
                print(f"  Warning: {asset} not found!")
        
        print("=== Asset Path Test Complete ===\n")
    
    def change_background(self, background_type):
        """
        Change the dashboard background dynamically.
        
        Args:
            background_type (str): "plasma.gif", "tenor.gif", "v.gif", "solid", or "none"
        """
        if background_type == "none":
            self.use_gif_background = False
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef);")
            print("Background changed to solid color")
            return
        
        self.use_gif_background = True
        self.preferred_background = background_type
        
        # Stop current movie if any
        if hasattr(self.bg_label, 'movie'):
            self.bg_label.movie().stop()
        
        # Load new background
        movie = None
        if background_type == "plasma.gif":
            plasma_path = get_asset_path("plasma.gif")
            if os.path.exists(plasma_path):
                movie = QMovie(plasma_path)
                print("Background changed to plasma.gif")
            else:
                print("plasma.gif not found, keeping current background")
                return
        elif background_type == "tenor.gif":
            tenor_gif_path = get_asset_path("tenor.gif")
            if os.path.exists(tenor_gif_path):
                movie = QMovie(tenor_gif_path)
                print("Background changed to tenor.gif")
            else:
                print("tenor.gif not found, keeping current background")
                return
        elif background_type == "v.gif":
            v_gif_path = get_asset_path("v.gif")
            if os.path.exists(v_gif_path):
                movie = QMovie(v_gif_path)
                print("Background changed to v.gif")
            else:
                print("v.gif not found, keeping current background")
                return
        
        if movie:
            self.bg_label.setMovie(movie)
            movie.start()
            # Store reference to movie
            self.bg_label.movie = lambda: movie
    
    def cycle_background(self):
        """
        Cycle through different background options when the background button is clicked.
        """
        backgrounds = ["solid", "light_gradient", "dark_gradient", "medical_theme"]
        current_bg = "solid"  # Default to solid
        
        try:
            current_index = backgrounds.index(current_bg)
            next_index = (current_index + 1) % len(backgrounds)
            next_bg = backgrounds[next_index]
        except ValueError:
            next_bg = "solid"
        
        if next_bg == "solid":
            self.change_background("none")
            self.bg_btn.setText("BG: Solid")
        elif next_bg == "light_gradient":
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #ffffff, stop:1 #f0f0f0);")
            self.bg_btn.setText("BG: Light")
        elif next_bg == "dark_gradient":
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e50, stop:1 #34495e);")
            self.bg_btn.setText("BG: Dark")
        elif next_bg == "medical_theme":
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #e8f5e8, stop:1 #d4edda);")
            self.bg_btn.setText("BG: Medical")
        
    def reset_metrics_and_interpretation(self):
        """Reset dashboard metrics and interpretation (called when device disconnects)"""
        if hasattr(self, 'metric_labels'):
            if 'heart_rate' in self.metric_labels:
                self.metric_labels['heart_rate'].setText("0 BPM")
            if 'pr_interval' in self.metric_labels:
                self.metric_labels['pr_interval'].setText("0 ms")
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setText("0 ms")
            if 'qtc_interval' in self.metric_labels:
                self.metric_labels['qtc_interval'].setText("--")
        
        self.current_heart_rate = 0
        self._last_bpm = None
        self._last_stable_rr = None
        self._rr_stability_counter = 0
        
        # Reset cached metric values
        self._last_hr = None
        self._last_pr = None
        self._last_qrs = None
        self._last_p = None
        self._last_qtc = None
        
        if hasattr(self, 'conclusion_box'):
            self.conclusion_box.setHtml("""
                <p style='color: #888; font-style: italic;'>
                No ECG data available yet.<br><br>
                Start an ECG test to see your personalized analysis and recommendations.
                </p>
            """)

    def center_on_screen(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def resizeEvent(self, event):
        """Handle window resize events to maintain responsive design"""
        super().resizeEvent(event)
        
        # Update background label size to match new window size
        if hasattr(self, 'bg_label'):
            self.bg_label.setGeometry(0, 0, self.width(), self.height())
        
        # Ensure all widgets maintain proper proportions
        self.update_layout_proportions()

    def closeEvent(self, event):
        """Handle dashboard closure to ensure hardware is disconnected"""
        print("Dashboard closing...")

        # Clear hardware version on application close
        if hasattr(self, 'settings_manager'):
            self.settings_manager.set_setting("hardware_version", "")
            self.settings_manager.save_settings()
            print("Hardware version cleared on close")

        try:
            if hasattr(self, 'ecg_test_page') and self.ecg_test_page:
                if hasattr(self.ecg_test_page, 'close_serial_connection'):
                    self.ecg_test_page.close_serial_connection()
                elif hasattr(self.ecg_test_page, 'serial_reader') and self.ecg_test_page.serial_reader:
                    self.ecg_test_page.serial_reader.stop()
                    self.ecg_test_page.serial_reader.close()
        except Exception as e:
            print(f"Error cleaning up dashboard resources: {e}")
        try:
            from ecg.serial.serial_reader import GlobalHardwareManager
            GlobalHardwareManager().close_reader()
        except Exception as e:
            print(f"Error closing global serial reader: {e}")
        try:
            if hasattr(self, 'device_check_timer') and self.device_check_timer:
                self.device_check_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, '_license_timer') and self._license_timer:
                self._license_timer.stop()
        except Exception:
            pass
        try:
            license_threads = list(getattr(self, '_license_threads', []) or [])
            for thread in license_threads:
                try:
                    if thread is not None and thread.isRunning():
                        thread.requestInterruption()
                        thread.quit()
                        thread.wait(5000)
                    if thread is not None and thread.isRunning():
                        thread.terminate()
                        thread.wait(1000)
                except Exception:
                    pass
            if hasattr(self, '_license_threads'):
                self._license_threads.clear()
        except Exception as e:
            print(f"Error cleaning up license watchdog threads: {e}")
        event.accept()
    
    def update_layout_proportions(self):
        """Update layout proportions when window is resized"""
        # This method can be used to adjust layout proportions based on window size
        current_width = self.width()
        current_height = self.height()
        
        # Adjust font sizes based on window size for better readability
        if current_width < 1000:
            # Small window - use smaller fonts
            font_size = 12
        elif current_width < 1400:
            # Medium window - use medium fonts
            font_size = 14
        else:
            # Large window - use larger fonts
            font_size = 16
        
        # Update font sizes for better responsiveness
        for child in self.findChildren(QLabel):
            if hasattr(child, 'font'):
                current_font = child.font()
                if current_font.pointSize() > 8:  # Don't make fonts too small
                    current_font.setPointSize(max(8, font_size - 2))
                    child.setFont(current_font)
