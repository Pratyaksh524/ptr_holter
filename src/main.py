"""
CardioX Main Entry Point — src/main.py
======================================
PURPOSE & ARCHITECTURE:
This is the primary application bootstrap and lifecycle controller for the CardioX 12-Lead ECG system.
It handles:
1. System & display initialization (PyQt5, software OpenGL rendering fallback, sys.path resolution).
2. Crash logging setup (ECGCrashLogger) and exception hooks.
3. License management and startup hardware/seat verification.
4. User authentication (Standard Login / Sign Up flow with serial number or credential validation).
5. Instantiation and switching between LoginWindow, DashboardWindow, and ECGTestWindow.

DEVELOPER NOTES & RECENT REFACTORS:
- Security / Authentication: Removed hardcoded admin/divyansh login bypass. All users must authenticate via standard signup/login & valid license seats.
- OpenGL Stability: BUG-05 fix forces QCoreApplication.setAttribute(Qt.AA_UseSoftwareOpenGL) on macOS/Linux to prevent driver crashes during high-frequency graph rendering.
- Crash Logging: Integrates automatic crash payload generation and email error reporting via src/utils/crash_logger.py.
"""

import sys
import os


def _dev_autologin_enabled() -> bool:
    """
    True only when the developer auto-login bypass is explicitly switched on.

    The bypass skips the password prompt AND the licence gate (heartbeat,
    revocation, offline grace), so it defaults to OFF and is never enabled in a
    distributed build — CARDIOX_DEV_AUTOLOGIN is stripped from the installer's
    .env by build_exe.py.
    """
    return os.getenv("CARDIOX_DEV_AUTOLOGIN", "").strip().lower() in ("1", "true", "yes", "on")
import shutil

# Ensure the main src directory is at the absolute front of sys.path
# to prevent subdirectories (like src/ecg) from shadowing root modules (like utils)
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir in sys.path:
    sys.path.remove(_src_dir)
sys.path.insert(0, _src_dir)

for _p in list(sys.path):
    if _p.startswith(_src_dir + os.sep) and _p != _src_dir:
        sys.path.remove(_p)
        sys.path.append(_p)

# Ensure stdout/stderr replace unencodable characters instead of crashing
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(errors='replace')
        except Exception:
            pass

# ── BUG-05 FIX: Force software OpenGL rendering ──────────────────────────────
# MUST be set BEFORE any Qt/PyQtGraph import.
# This fixes blank waves on laptops with Intel HD, AMD integrated, or no GPU.
os.environ['QT_OPENGL'] = 'software'
if sys.platform.startswith("win"):
    os.environ['PYOPENGL_PLATFORM'] = 'win32'
os.environ['QT_SCALE_FACTOR'] = '1'
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '0'
# ─────────────────────────────────────────────────────────────────────────────

# ── Deterministic arrhythmia analysis: force single-threaded BLAS/FFT ────────
# MUST be set BEFORE numpy is imported anywhere (even transitively) -- the
# underlying BLAS/OpenMP libraries read these once at load time.
#
# numpy/scipy's spectral and linear-algebra routines (used throughout
# ecg/arrhythmia_detector.py's flutter/VFib scoring) can multi-thread their
# internal summations. Floating-point addition isn't associative, so the
# result of the same computation on the same input can come out very
# slightly different depending on how the OS happens to schedule those
# threads that run -- normally immaterial, but it was observed to
# occasionally flip a borderline spectral threshold (e.g. the atrial-flutter
# score) and change which arrhythmia a window gets classified as, even
# though the input ECG samples were confirmed byte-for-byte identical run to
# run. Pinning every relevant thread pool to 1 makes that computation fully
# deterministic; the ECG windows here are small enough that single-threaded
# BLAS has no perceptible performance cost.
for _var in (
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
):
    os.environ[_var] = '1'
# ─────────────────────────────────────────────────────────────────────────────

# ── MPLBACKEND: Force Agg (non-GUI) matplotlib backend for all child processes ─
# Belt-and-suspenders: .env sets MPLBACKEND=Agg, but enforce it here too so
# any subprocess or import that happens before dotenv loads uses the right backend.
# Agg is ~2x faster than Qt5Agg for off-screen PDF rendering.
if not os.environ.get('MPLBACKEND'):
    os.environ['MPLBACKEND'] = 'Agg'
# ─────────────────────────────────────────────────────────────────────────────

import json
from dotenv import load_dotenv
from utils.app_paths import data_file
from utils.platform_compat import is_low_spec_mode

def _prepare_runtime_workspace() -> str:
    """
    Ensure a writable runtime directory for packaged installs.
    This avoids permission issues on systems where app is installed in Program Files.
    """
    use_runtime = bool(getattr(sys, "frozen", False)) or (
        str(os.getenv("ECG_FORCE_RUNTIME_DIR", "0")).strip().lower() in {"1", "true", "yes", "on"}
    )
    if not use_runtime:
        return os.getcwd()

    base_dir = os.getenv("ECG_RUNTIME_DIR", "").strip()
    if not base_dir:
        local_appdata = os.getenv("LOCALAPPDATA") or os.path.expanduser("~")
        base_dir = os.path.join(local_appdata, "Deckmount", "CardioX")
    base_dir = os.path.abspath(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    # Ensure required runtime folders exist.
    for rel in ("reports", "logs", "offline_queue", "temp", "src"):
        os.makedirs(os.path.join(base_dir, rel), exist_ok=True)

    # Seed essential config files from bundle/app folder if missing in runtime dir.
    source_roots = []
    if getattr(sys, "frozen", False):
        source_roots.append(os.path.dirname(sys.executable))
        if hasattr(sys, "_MEIPASS"):
            source_roots.append(sys._MEIPASS)
    else:
        source_roots.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    seed_files = [
        ".env",
        "customer_channels.json",
        "users.json",
        "ecg_settings.json",
        "last_conclusions.json",
        os.path.join("src", "users.json"),
        os.path.join("src", "ecg_settings.json"),
    ]
    for rel in seed_files:
        dst = os.path.join(base_dir, rel)
        if os.path.exists(dst):
            # Special check to merge newly bundled .env keys (e.g. LICENSE_SERVER_URL)
            # and correct localhost URLs in the persistent workspace environment file.
            if rel == ".env":
                try:
                    # Read existing local keys
                    local_keys = set()
                    with open(dst, "r", encoding="utf-8", errors="replace") as f:
                        local_lines = f.readlines()
                    for line in local_lines:
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#") and "=" in stripped:
                            key = stripped.split("=", 1)[0].strip()
                            local_keys.add(key)
                except Exception:
                    local_lines = []

                # Find bundled .env
                bundled_env_path = None
                for root in source_roots:
                    src = os.path.join(root, rel)
                    if os.path.exists(src):
                        bundled_env_path = src
                        break

                if bundled_env_path:
                    try:
                        with open(bundled_env_path, "r", encoding="utf-8", errors="replace") as f:
                            bundled_lines = f.readlines()
                        
                        to_append = []
                        for line in bundled_lines:
                            stripped = line.strip()
                            if stripped and not stripped.startswith("#") and "=" in stripped:
                                key, val = stripped.split("=", 1)
                                key = key.strip()
                                val = val.strip()
                                if key not in local_keys:
                                    to_append.append(f"{key}={val}\n")
                        
                        # Overwrite if LICENSE_SERVER_URL in local_lines points to localhost
                        modified = False
                        new_local_lines = []
                        for line in local_lines:
                            stripped = line.strip()
                            if stripped.startswith("LICENSE_SERVER_URL="):
                                val = stripped.split("=", 1)[1].strip().lower()
                                if "localhost" in val or "127.0.0.1" in val:
                                    # Overwrite with bundled or default prod URL
                                    bundled_url = "https://m4qoae4d8e.execute-api.us-east-1.amazonaws.com/prod/api/v1"
                                    for bline in bundled_lines:
                                        if bline.strip().startswith("LICENSE_SERVER_URL="):
                                            bundled_url = bline.strip().split("=", 1)[1].strip()
                                            break
                                    new_local_lines.append(f"LICENSE_SERVER_URL={bundled_url}\n")
                                    modified = True
                                    continue
                            new_local_lines.append(line)

                        if modified:
                            with open(dst, "w", encoding="utf-8") as f:
                                f.writelines(new_local_lines)
                        
                        if to_append:
                            with open(dst, "a", encoding="utf-8") as f:
                                f.write("\n# --- Staged Keys Added on Update ---\n")
                                f.writelines(to_append)
                    except Exception:
                        pass
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for root in source_roots:
            src = os.path.join(root, rel)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, dst)
                except Exception:
                    pass
                break

    os.environ["ECG_RUNTIME_DIR"] = base_dir
    os.chdir(base_dir)
    return base_dir


_RUNTIME_DIR = _prepare_runtime_workspace()

# Load environment variables from .env file.
# Priority: runtime dir (.env) -> executable dir -> _MEIPASS
# Runtime dir is HIGHEST priority — it contains production settings
runtime_env = os.path.join(os.getcwd(), ".env")
if os.path.exists(runtime_env):
    load_dotenv(runtime_env, override=True)   # override=True: runtime .env wins over any bundled values
else:
    load_dotenv(override=False)
if getattr(sys, "frozen", False):
    app_env = os.path.join(os.path.dirname(sys.executable), ".env")
    if os.path.exists(app_env):
        load_dotenv(app_env, override=False)
if hasattr(sys, '_MEIPASS'):
    meipass_env = os.path.join(sys._MEIPASS, '.env')
    if os.path.exists(meipass_env):
        load_dotenv(meipass_env, override=False)

# Safety-net: if LICENSE_SERVER_URL is still empty/localhost after all .env loading,
# force-set the known production URL so update checking always works in frozen builds.
_lic_url = os.environ.get("LICENSE_SERVER_URL", "").strip().lower()
if not _lic_url or "localhost" in _lic_url or "127.0.0.1" in _lic_url:
    os.environ["LICENSE_SERVER_URL"] = (
        "https://m4qoae4d8e.execute-api.us-east-1.amazonaws.com/prod/api/v1"
    )

from PyQt5.QtWidgets import (
    QApplication, QDialog, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, 
    QMessageBox, QStackedWidget, QWidget, QInputDialog, QSizePolicy, QFrame, QScrollArea,
    QFormLayout, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QUrl, QRegularExpression, QThread, pyqtSignal
from PyQt5.QtNetwork import QLocalSocket, QLocalServer
from utils.crash_logger import get_crash_logger
from utils.session_recorder import SessionRecorder
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap, QIntValidator, QRegularExpressionValidator
from utils.ecg_auth_api import get_ecg_auth_api
from utils.offline_queue import get_offline_queue
from utils.settings_manager import SettingsManager

try:
    from version import APP_VERSION, UPDATE_CHANNEL, GITHUB_REPOSITORY
except Exception:
    APP_VERSION = "0.0.0"
    UPDATE_CHANNEL = "stable"
    GITHUB_REPOSITORY = ""

# ── Update Notification helpers ───────────────────────────────────────────────

class UpdateBannerWidget(QWidget):
    """
    A slim, non-blocking floating banner shown at the top of the dashboard
    when a newer version is available.

    The banner is inserted as a child of *parent_widget* (the dashboard window)
    and positions itself at the top-right corner automatically.
    """

    def __init__(self, update_info: dict, parent_widget: QWidget) -> None:
        super().__init__(parent_widget, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self._info = update_info
        self._parent = parent_widget
        self._build_ui()
        self._reposition()
        # Reposition when the parent window moves or resizes.
        parent_widget.installEventFilter(self)

    def _build_ui(self) -> None:
        version        = self._info.get("version", "?")
        notes          = self._info.get("release_notes", "") or "Improvements and bug fixes."
        url            = self._info.get("download_url", "")
        is_rollback    = bool(self._info.get("force_rollback", False))

        # Rollback = urgent red; normal update = orange
        border_colour  = "rgba(220,60,60,0.75)"  if is_rollback else "rgba(255,140,0,0.6)"
        bg_stop0       = "rgba(56,18,18,0.97)"   if is_rollback else "rgba(26,32,56,0.97)"
        bg_stop1       = "rgba(52,12,12,0.97)"   if is_rollback else "rgba(36,24,52,0.97)"
        icon           = "⚠️"                     if is_rollback else "🔔"
        title_colour   = "#ff6b6b"               if is_rollback else "#ffb347"
        if is_rollback:
            title_text = f"Action Required — reinstall v{version}"
        else:
            title_text = f"Update available  —  v{version}"
        dl_label       = "📥  Reinstall"         if is_rollback else "📥  Download"

        self.setFixedWidth(420 if is_rollback else 400)
        self.setAttribute(Qt.WA_TranslucentBackground)

        container = QFrame(self)
        container.setObjectName("UpdateBanner")
        container.setStyleSheet(f"""
            QFrame#UpdateBanner {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {bg_stop0},
                    stop:1 {bg_stop1}
                );
                border: 1px solid {border_colour};
                border-radius: 14px;
            }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(container)

        inner = QVBoxLayout(container)
        inner.setContentsMargins(16, 12, 16, 12)
        inner.setSpacing(6)

        # Header row
        header_row = QHBoxLayout()
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 20px; background: transparent;")
        title = QLabel(title_text)
        title.setStyleSheet(
            f"color: {title_colour}; font-size: 13px; font-weight: bold; background: transparent;"
        )
        header_row.addWidget(icon_lbl)
        header_row.addWidget(title, 1)
        inner.addLayout(header_row)

        # Extra warning line for rollbacks
        if is_rollback:
            warn_lbl = QLabel("⛔  Your current version has a critical issue. Please reinstall the stable release immediately.")
            warn_lbl.setWordWrap(True)
            warn_lbl.setStyleSheet("color: #ff9999; font-size: 11px; font-weight: bold; background: transparent;")
            inner.addWidget(warn_lbl)

        # Release notes (truncated)
        if notes and notes != "Improvements and bug fixes.":
            notes_label = QLabel(notes[:120] + ("…" if len(notes) > 120 else ""))
            notes_label.setWordWrap(True)
            notes_label.setStyleSheet("color: rgba(255,255,255,0.82); font-size: 11px; background: transparent;")
            inner.addWidget(notes_label)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        if url:
            dl_btn = QPushButton(dl_label)
            dl_colour = "#cc2222" if is_rollback else "#ff7a12"
            dl_hover  = "#e03333" if is_rollback else "#ff8a26"
            dl_btn.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                        stop:0 {dl_colour}, stop:1 {dl_colour});
                    color: white; border-radius: 10px;
                    padding: 6px 14px; font-size: 12px; font-weight: bold; border: none;
                }}
                QPushButton:hover {{ background: {dl_hover}; }}
            """)
            dl_btn.setCursor(Qt.PointingHandCursor)
            dl_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(url)))
            btn_row.addWidget(dl_btn)

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setStyleSheet("""
            QPushButton {
                background: rgba(80,80,80,0.55);
                color: rgba(255,255,255,0.75);
                border-radius: 10px; padding: 6px 12px;
                font-size: 12px; border: none;
            }
            QPushButton:hover { background: rgba(110,110,110,0.75); }
        """)
        dismiss_btn.clicked.connect(self.close)
        btn_row.addStretch(1)
        btn_row.addWidget(dismiss_btn)
        inner.addLayout(btn_row)

        self.adjustSize()

    def _reposition(self) -> None:
        """Place the banner at the top-right of the parent window."""
        try:
            parent_rect = self._parent.geometry()
            x = parent_rect.x() + parent_rect.width() - self.width() - 24
            y = parent_rect.y() + 60
            self.move(x, y)
        except Exception:
            pass

    def eventFilter(self, obj, event) -> bool:
        from PyQt5.QtCore import QEvent
        if obj is self._parent and event.type() in (
            QEvent.Move, QEvent.Resize, QEvent.Show
        ):
            self._reposition()
        return super().eventFilter(obj, event)


# Session-level set of versions already shown (so the banner doesn't reappear).
_update_versions_shown: set = set()


def _launch_update_checker(dashboard: QWidget, app_version: str, channel: str) -> None:
    """
    Start the background update checker after the dashboard is shown.
    Fires 5 seconds after being called so the dashboard fully renders first.
    """
    try:
        from utils.update_checker import UpdateCheckerThread

        def _on_update_available(info: dict) -> None:
            version = info.get("version", "")
            if version in _update_versions_shown:
                return
            _update_versions_shown.add(version)
            try:
                banner = UpdateBannerWidget(info, dashboard)
                banner.show()
                # Keep a reference so GC doesn't collect it.
                if not hasattr(dashboard, "_update_banners"):
                    dashboard._update_banners = []
                dashboard._update_banners.append(banner)
            except Exception as _be:
                logger.warning(f"[UpdateChecker] Could not show banner: {_be}")

        _checker = UpdateCheckerThread(
            current_version=app_version,
            channel=channel,
            delay_seconds=5.0,
            parent=dashboard,   # Set dashboard as parent so it's deleted WITH the dashboard
        )
        _checker.update_available.connect(_on_update_available)
        _checker.start()
        # Keep a strong reference and wire clean shutdown to dashboard close.
        dashboard._update_checker = _checker
        
        def safe_stop():
            try:
                if _checker and not _checker.isFinished():
                    _checker.stop()
            except (RuntimeError, ReferenceError):
                pass
        
        try:
            dashboard.destroyed.connect(safe_stop)
        except Exception:
            pass
        logger.info("[UpdateChecker] Background update check scheduled (5s delay).")
    except Exception as e:
        logger.warning(f"[UpdateChecker] Could not start update checker: {e}")

# Import core modules  
try:
    from core.logging_config import get_logger, log_function_call
    from core.exceptions import ECGError, ECGConfigError
    from config.settings import get_config, resource_path
    from core.constants import SUCCESS_MESSAGES, ERROR_MESSAGES
    logger_available = True
except ImportError as e:
    print(f" Core modules not available: {e}")
    print(" Using fallback logging")
    logger_available = False
    
    # Fallback logging
    class FallbackLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}") #msg is messagin g for the self
    
    def log_function_call(func):
        return func
    
    def get_config():
        return type('Config', (), {'get': lambda x, y=None: y})()
    
    def resource_path(relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)
    
    SUCCESS_MESSAGES = {"modules_loaded": " Core modules imported successfully"}
    ERROR_MESSAGES = {"import_error": " Core module import error: {}"}

# Initialize logger
if logger_available:
    logger = get_logger("MainApp")
else:
    logger = FallbackLogger()

# Import application modules with proper error handling
def get_auth_modules():
    try:
        from auth.sign_in import SignIn
        from auth.sign_out import SignOut
        return SignIn, SignOut
    except ImportError as e:
        logger.error(ERROR_MESSAGES["import_error"].format(e))
        logger.error("💡 Make sure you're running from the src directory")
        logger.error("💡 Try: cd src && python main.py")
        sys.exit(1)

def get_dashboard_module():
    try:
        from dashboard.dashboard import Dashboard
        return Dashboard
    except ImportError as e:
        logger.error(ERROR_MESSAGES["import_error"].format(e))
        return None


def _recover_license_in_place(app, window=None, reason: str = "", title: str = "License Invalid") -> bool:
    """
    Disable current features, clear license details, close the dashboard,
    and return the user to the login/signup page.
    """
    message = reason or "License key is revoked. Contact support."
    try:
        if window is not None:
            window.setEnabled(False)
    except Exception:
        pass

    try:
        QMessageBox.warning(
            window,
            title,
            f"{message}\n\nYou will be redirected to the signup page to register again.",
        )
    except Exception:
        pass

    try:
        from utils.license_manager import clear_license_cache, clear_stored_key
        clear_stored_key()
        clear_license_cache()
    except Exception as e:
        logger.warning(f"Error clearing license details: {e}")

    try:
        if window is not None:
            try:
                window.closed_by_sign_out = True
            except Exception:
                pass
            window.close()
    except Exception:
        pass

    return False


def _ecg_session_active(window) -> bool:
    """Return True when the live ECG page is actively acquiring or recording data."""
    try:
        ecg_page = getattr(window, "ecg_test_page", None)
        if ecg_page is None:
            return False

        if bool(getattr(ecg_page, "is_recording", False)):
            return True

        timer = getattr(ecg_page, "timer", None)
        if timer is not None and hasattr(timer, "isActive") and timer.isActive():
            return True

        recorder = getattr(ecg_page, "recording_timer", None)
        if recorder is not None and hasattr(recorder, "isActive") and recorder.isActive():
            return True

        serial_reader = getattr(ecg_page, "serial_reader", None)
        if serial_reader is not None and bool(getattr(serial_reader, "running", False)):
            return True

        holter_writer = getattr(ecg_page, "_holter_writer", None)
        if holter_writer is not None and bool(getattr(holter_writer, "is_running", False)):
            return True

        return False
    except Exception:
        return False


def _defer_license_block_until_safe(app, window, reason: str, title: str) -> bool:
    """
    Keep the app open until the current ECG session finishes, then perform the
    normal license shutdown flow.
    """
    if window is None:
        return _recover_license_in_place(app, window, reason, title)

    try:
        if getattr(window, "_pending_license_block", False):
            return False
        window._pending_license_block = True
        window._pending_license_block_reason = reason
        window._pending_license_block_title = title
    except Exception:
        pass

    try:
        QMessageBox.critical(
            window,
            title,
            f"{reason}\n\n"
            "ECG acquisition is currently active.\n"
            "The application will close after the current recording is finished.",
        )
    except Exception:
        pass

    try:
        timer = getattr(window, "_license_block_timer", None)
        if timer is None:
            timer = QTimer(window)
            timer.setInterval(1000)

            def _poll_license_block():
                if not _ecg_session_active(window):
                    timer.stop()
                    _recover_license_in_place(app, window, reason, title)

            timer.timeout.connect(_poll_license_block)
            window._license_block_timer = timer
        if not timer.isActive():
            timer.start()
    except Exception as e:
        logger.warning(f"Could not start deferred license block timer: {e}")
        return _recover_license_in_place(app, window, reason, title)

    return False

# Import ECG modules with fallback
def get_ecg_modules():
    try:
        from ecg.pan_tompkins import pan_tompkins
        logger.info(SUCCESS_MESSAGES["ecg_modules_loaded"])
        return pan_tompkins
    except ImportError as e:
        if "ecg_import_warning" in ERROR_MESSAGES:
            logger.warning(ERROR_MESSAGES["ecg_import_warning"].format(e))
        else:
            logger.warning(f"ECG module import warning: {e}")
        logger.warning("💡 ECG analysis features may be limited")
        # Create a dummy function to prevent errors
        def pan_tompkins(ecg, fs=500):
            return []
        return pan_tompkins

# Get configuration
config = get_config()
# Store runtime-created data files in a per-user writable directory for packaged builds.
USER_DATA_FILE = str(data_file("users.json"))


@log_function_call
def load_users():
    """Load user data from file with error handling"""
    try:
        if os.path.exists(USER_DATA_FILE):
            with open(USER_DATA_FILE, "r") as f:
                users = json.load(f)
                logger.info(f"Loaded {len(users)} users from {USER_DATA_FILE}")
                return users
        else:
            logger.info(f"User file {USER_DATA_FILE} not found, creating empty user database")
            return {}
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading users: {e}")
        logger.error("Creating empty user database")
        return {}


@log_function_call
def save_users(users):
    """Save user data to file with error handling"""
    try:
        with open(USER_DATA_FILE, "w") as f:
            json.dump(users, f, indent=2)
        logger.info(f"Saved {len(users)} users to {USER_DATA_FILE}")
    except IOError as e:
        logger.error(f"Error saving users: {e}")
        raise ECGError(f"Failed to save user data: {e}")


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


class RegisterWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, name, doctor, org_name, org_address, phone, password, serial_id):
        super().__init__()
        self.name = name
        self.doctor = doctor
        self.org_name = org_name
        self.org_address = org_address
        self.phone = phone
        self.password = password
        self.serial_id = serial_id

    def run(self):
        try:
            import hashlib
            from utils.license_manager import register_device
            pw_hash = hashlib.sha256(self.password.encode("utf-8")).hexdigest()
            res = register_device(
                license_key="",
                full_name=self.name,
                doctor_name=self.doctor,
                org_name=self.org_name,
                org_address=self.org_address,
                phone=self.phone,
                password_hash=pw_hash,
                machine_serial_id=self.serial_id,
            )
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class LoadingOverlayDialog(QDialog):
    """Full-screen overlay shown during signup while license registration runs in the background."""

    _GIF_SIZE = 248
    _GIF_RING = 300

    _STATUS_MSGS = [
        "Creating your account",
        "Setting up your profile",
        "Almost done",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setModal(True)

        if parent:
            self.setGeometry(parent.geometry())
        else:
            self.setFixedSize(900, 650)

        self._step_index = 0
        self._dot_count  = 0
        self._progress   = 0
        self.movie = None
        self._low_spec_mode = is_low_spec_mode()

        self._build_ui()

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick_anim)
        self._anim_timer.start(140 if self._low_spec_mode else 60)

        self._step_timer = QTimer(self)
        self._step_timer.timeout.connect(self._advance_step)
        self._step_timer.start(3200 if self._low_spec_mode else 2200)

    def showEvent(self, event):
        super().showEvent(event)
        if self.parent():
            self.setGeometry(self.parent().geometry())
        self._rescale_animation()

    def _signup_gif_path(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            resource_path("Animation - 1777012518993.gif"),
            os.path.join(root_dir, "Animation - 1777012518993.gif"),
            resource_path("assets/v.gif"),
            os.path.join(root_dir, "assets", "v.gif"),
        ]
        return next((p for p in candidates if p and os.path.exists(p)), None)

    def _rescale_animation(self):
        if self.movie and self.movie.isValid() and hasattr(self, "gif_label"):
            self.movie.setScaledSize(self.gif_label.size())

    def _build_ui(self):
        from PyQt5.QtGui import QMovie

        self.setStyleSheet("QDialog { background: rgba(8, 10, 22, 0.98); }")

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setAlignment(Qt.AlignCenter)

        card = QWidget()
        card.setObjectName("RegCard")
        card.setFixedWidth(640)
        card.setStyleSheet("""
            QWidget#RegCard {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a2238, stop:1 #0e1324);
                border-radius: 32px;
                border: 1px solid rgba(255, 140, 40, 0.45);
            }
        """)

        cl = QVBoxLayout(card)
        cl.setContentsMargins(56, 48, 56, 44)
        cl.setSpacing(0)

        # Hero animation — large centered ring with soft glow
        anim_shell = QFrame()
        anim_shell.setObjectName("AnimShell")
        anim_shell.setFixedSize(self._GIF_RING, self._GIF_RING)
        anim_shell.setStyleSheet("""
            QFrame#AnimShell {
                background: qradialgradient(
                    cx:0.5, cy:0.5, radius:0.85,
                    fx:0.5, fy:0.5,
                    stop:0 rgba(255, 122, 18, 0.22),
                    stop:0.55 rgba(255, 122, 18, 0.06),
                    stop:1 rgba(255, 122, 18, 0));
                border-radius: 150px;
                border: 1px solid rgba(255, 160, 70, 0.28);
            }
        """)
        anim_layout = QVBoxLayout(anim_shell)
        anim_layout.setContentsMargins(0, 0, 0, 0)
        anim_layout.setAlignment(Qt.AlignCenter)

        self.gif_label = QLabel()
        self.gif_label.setFixedSize(self._GIF_SIZE, self._GIF_SIZE)
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setScaledContents(True)
        self.gif_label.setStyleSheet("""
            QLabel {
                background: rgba(12, 16, 30, 0.55);
                border-radius: 124px;
                border: 2px solid rgba(255, 180, 90, 0.35);
            }
        """)

        gif_path = self._signup_gif_path()
        self.movie = QMovie(gif_path) if gif_path else None
        if self.movie and self.movie.isValid():
            self.gif_label.setMovie(self.movie)
            self.movie.setScaledSize(self.gif_label.size())
            self.movie.start()
        else:
            self.gif_label.setText("\u29d7")
            self.gif_label.setStyleSheet("""
                QLabel {
                    font-size: 96px;
                    color: #ff7a12;
                    background: rgba(12, 16, 30, 0.55);
                    border-radius: 124px;
                    border: 2px solid rgba(255, 180, 90, 0.35);
                }
            """)

        anim_layout.addWidget(self.gif_label, alignment=Qt.AlignCenter)

        title = QLabel("Creating Your Account")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #ffc978; background: transparent; "
            "margin-top: 28px; letter-spacing: 0.5px;"
        )

        subtitle = QLabel("Please wait while we finish setting up your account")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet(
            "color: rgba(255,255,255,0.52); background: transparent; "
            "margin-top: 8px; margin-bottom: 28px; line-height: 1.4;"
        )

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background: rgba(255,255,255,0.08);
                border-radius: 5px;
                border: none;
                min-height: 10px;
                max-height: 10px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff7a12, stop:1 #ffe08a);
                border-radius: 5px;
            }
        """)

        self.status_label = QLabel(self._STATUS_MSGS[0] + "...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        self.status_label.setStyleSheet("""
            color: rgba(255,255,255,0.92);
            background: transparent;
            margin-top: 20px;
            margin-bottom: 4px;
            letter-spacing: 0.3px;
        """)

        hint = QLabel("Please keep this window open until setup completes")
        hint.setAlignment(Qt.AlignCenter)
        hint.setFont(QFont("Segoe UI", 9))
        hint.setStyleSheet(
            "color: rgba(255, 150, 60, 0.62); background: transparent; margin-top: 22px;"
        )

        cl.addWidget(anim_shell, alignment=Qt.AlignCenter)
        cl.addWidget(title)
        cl.addWidget(subtitle)
        cl.addWidget(self.progress_bar)
        cl.addWidget(self.status_label)
        cl.addWidget(hint)

        root.addWidget(card, alignment=Qt.AlignCenter)

    # ------------------------------------------------------------------ #
    def _tick_anim(self):
        """Animate progress bar and status dots while registration runs."""
        target = min(int((self._step_index / max(len(self._STATUS_MSGS), 1)) * 92) + 4, 95)
        if self._progress < target:
            self._progress = min(self._progress + 1, target)
            self.progress_bar.setValue(self._progress)

        self._dot_count = (self._dot_count + 1) % 4
        base = self._STATUS_MSGS[min(self._step_index, len(self._STATUS_MSGS) - 1)].rstrip(".")
        self.status_label.setText(base + "." * self._dot_count)

    def _advance_step(self):
        """Cycle through user-facing status messages."""
        self._step_index += 1
        if self._step_index >= len(self._STATUS_MSGS):
            self._step_index = len(self._STATUS_MSGS) - 1
            self._step_timer.stop()


# Login/Register Dialog
class LoginRegisterDialog(QDialog):
    def __init__(self):
        super().__init__()
        self._low_spec_mode = is_low_spec_mode()
        
        # Set responsive size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(800, 600)  # Minimum size for usability
        
        # Set window properties for better responsiveness
        self.setWindowTitle("CardioX by Deckmount - Sign In / Sign Up")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        try:
            from PyQt5.QtGui import QIcon
            icon_path = resource_path("assets/cardiox_logo.ico")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass
        
        # Initialize sign-in logic
        SignIn, _ = get_auth_modules()
        self.sign_in_logic = SignIn()

        # Connection monitoring for auto-populating serial ID
        self.settings_manager = SettingsManager()
        self._device_scan_in_progress = False
        self._low_spec_mode = is_low_spec_mode()
        self.device_check_timer = QTimer(self)
        self.device_check_timer.timeout.connect(self.check_device_connection)
        self.device_check_timer.start(5000 if self._low_spec_mode else 1000) # Slower polling on weak machines
        
        # Resize according to current screen size (~90% of available geometry)
        try:
            screen_geom = QApplication.primaryScreen().availableGeometry()
            target_w = max(int(screen_geom.width() * 0.9), self.minimumWidth())
            target_h = max(int(screen_geom.height() * 0.9), self.minimumHeight())
            self.resize(target_w, target_h)
        except Exception:
            pass
        
        try:
            self.setWindowState(Qt.WindowMaximized)
        except Exception:
            pass
        
        self.ui_initialized = False
        self.result = False
        self.username = None
        self.user_details = {}

    def __getattr__(self, name):
        # Lazy initialize UI if someone accesses a widget/attribute before exec_
        if name in ("stacked", "bg_label") and not getattr(self, "ui_initialized", False):
            self.init_ui()
            self.ui_initialized = True
            return getattr(self, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def exec_(self):
        # Development-only auto-login for the 'cardiomac' test account.
        #
        # SAFETY: this returns QDialog.Accepted without any password check, which
        # also skips the licence gate that runs in the login handler — heartbeat,
        # revocation and offline-grace enforcement all live there. A revoked seat
        # would reach the dashboard unchallenged. It must never be active in a
        # distributed build, so it is gated on CARDIOX_DEV_AUTOLOGIN, which is
        # stripped from the .env staged into the installer by build_exe.py.
        if _dev_autologin_enabled():
            try:
                print("🔑 [DEV] Bypassing login screen for user: cardiomac (9560350477)")
                found = self.sign_in_logic._find_user_record("cardiomac")
                if found:
                    username, record = found
                    self.result = True
                    self.username = username
                    self.user_details = record
                    return QDialog.Accepted
                else:
                    print("⚠️ User 'cardiomac' not found in users.json! Showing login dialog...")
            except Exception as e:
                print(f"⚠️ Bypass login error: {e}")

        # If not bypassed, initialize UI now before showing dialog
        if not getattr(self, "ui_initialized", False):
            self.init_ui()
            self.ui_initialized = True
            
        return super().exec_()

    def init_ui(self):
        # Set up GIF background
        self.bg_label = QLabel(self)
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        self.bg_label.lower()
        
        # Try multiple possible paths for the v.gif file
        possible_gif_paths = [
            resource_path('assets/v.gif'),
            resource_path('../assets/v.gif'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'v.gif'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'v.gif')
        ]
        
        gif_path = None
        for path in possible_gif_paths:
            if os.path.exists(path):
                gif_path = path
                print(f" Found v.gif at: {gif_path}")
                break
        
        # Try loading the GIF first (as requested by user)
        loaded_gif = False
        if gif_path and os.path.exists(gif_path):
            try:
                from PyQt5.QtGui import QMovie
                movie = QMovie(gif_path)
                if movie.isValid():
                    self.bg_label.setMovie(movie)
                    movie.start()
                    print(" v.gif background started successfully")
                    loaded_gif = True
                else:
                    print(" Invalid GIF file")
            except Exception as e:
                print(f" Error loading v.gif: {e}")

        if not loaded_gif:
            # Fallback to gradient if GIF loading was unsuccessful or not found
            self.bg_label.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1a1a2e, stop:1 #16213e);")
        
        self.bg_label.setScaledContents(True)
        # --- Title and tagline above glass ---
        main_layout = QVBoxLayout(self)
        main_layout.addStretch(1)
        # Title (outside glass) - logo style
        title = QLabel("CardioX by Deckmount")
        title.setFont(QFont("Arial", 42, QFont.Black))
        title.setStyleSheet("""
            color: #ffb347;
            letter-spacing: 2px;
            margin-bottom: 0px;
            padding-top: 0px;
            padding-bottom: 0px;
            font-weight: 900;
            border-radius: 18px;
        """)
        title.setAlignment(Qt.AlignHCenter)
        main_layout.addWidget(title)
        # Tagline (outside glass)
        tagline = QLabel("Built to Detect. Designed to Last.")
        tagline.setFont(QFont("Arial", 16, QFont.Bold))
        tagline.setStyleSheet("color: #ff7a12; margin-bottom: 20px; margin-top: 2px; background: transparent;")
        tagline.setAlignment(Qt.AlignHCenter)
        main_layout.addWidget(tagline)
        # --- Glass effect container in center ---
        row = QHBoxLayout()
        row.addStretch(1)
        glass = QWidget(self)
        glass.setObjectName("Glass")
        glass.setStyleSheet("""
            QWidget#Glass {
                background: rgba(255,255,255,0.14);
                border-radius: 30px;
                border: 1px solid rgba(255,255,255,0.26);
            }
        """)
        glass.setMinimumSize(560, 500)
        # Create stacked widget and login/register widgets BEFORE using stacked_col
        self.stacked = QStackedWidget(glass)
        self.login_widget = self.create_login_widget()
        self.register_widget = self.create_register_widget()
        self.stacked.addWidget(self.login_widget)
        self.stacked.addWidget(self.register_widget)
        glass_layout = QHBoxLayout(glass)
        glass_layout.setContentsMargins(28, 28, 28, 24)
        glass_layout.setSpacing(12)
        # Login/Register stacked widget only, centered like the reference
        stacked_col = QVBoxLayout()
        stacked_col.setSpacing(14)
        stacked_col.addWidget(self.stacked, 1)
        # Add sign up/login prompt below
        signup_row = QHBoxLayout()
        signup_row.addStretch(1)
        signup_lbl = QLabel("Don't have an account?")
        signup_lbl.setStyleSheet("color: rgba(255,255,255,0.82); font-size: 14px;")
        signup_btn = QPushButton("Sign up")
        signup_btn.setStyleSheet("color: #ff8d2b; background: transparent; border: none; font-size: 14px; font-weight: bold; text-decoration: underline;")
        signup_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        signup_row.addWidget(signup_lbl)
        signup_row.addWidget(signup_btn)
        signup_row.addStretch(1)
        stacked_col.addLayout(signup_row)
        glass_layout.addLayout(stacked_col, 1)
        row.addWidget(glass, 0, Qt.AlignHCenter)
        row.addStretch(1)
        main_layout.addLayout(row)
        main_layout.addStretch(1)   
        self.setLayout(main_layout)
        # Make glass and all widgets expand responsively
        glass.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Resize background with window
        self.resizeEvent = self._resize_bg
        
        # Ensure background is always visible
        self.ensure_background_visible()

    def closeEvent(self, event):
        """Ensure any hardware connection is explicitly closed when exiting the app from the login screen."""
        try:
            if hasattr(self, 'device_check_timer') and self.device_check_timer:
                self.device_check_timer.stop()
        except Exception:
            pass

        try:
            if hasattr(self, 'scan_worker') and self.scan_worker and self.scan_worker.isRunning():
                self.scan_worker.requestInterruption()
                self.scan_worker.terminate()
        except Exception:
            pass

        # Forcefully send STOP and CLOSE to the saved port to free it
        try:
            if hasattr(self, 'settings_manager'):
                saved_port = self.settings_manager.get_setting("serial_port")
                if saved_port:
                    import serial
                    from ecg.serial.hardware_commands import HardwareCommandHandler
                    print(f"Forcing hardware STOP and CLOSE on {saved_port} during app exit...")
                    # Open port briefly just to send the commands
                    ser = serial.Serial(saved_port, 115200, timeout=0.2, write_timeout=0.2)
                    handler = HardwareCommandHandler(ser)
                    handler.send_stop_command()
                    handler.send_close_command()
                    ser.close()
                    print(f"COM port {saved_port} explicitly freed.")
        except Exception as e:
            print(f"Error sending exit commands on login screen close: {e}")

        try:
            from ecg.serial.serial_reader import GlobalHardwareManager
            manager = GlobalHardwareManager()
            if getattr(manager, 'reader', None):
                manager.close_reader()
        except Exception as e:
            pass

        super().closeEvent(event)

    def check_device_connection(self):
        """Monitor USB connection to auto-populate machine serial ID"""
        if self._device_scan_in_progress:
            return

        try:
            import serial.tools.list_ports
            current_ports = [p.device for p in serial.tools.list_ports.comports()]
            
            # Simple heuristic: only scan if ports changed or we haven't found a device yet
            if not hasattr(self, '_last_ports') or current_ports != self._last_ports:
                self._last_ports = current_ports
                
                self._device_scan_in_progress = True
                self.scan_worker = DeviceScanWorker(self.settings_manager)
                self.scan_worker.scan_finished.connect(self.on_scan_finished)
                self.scan_worker.start()
        except Exception as e:
            print(f"Error checking connection: {e}")

    def on_scan_finished(self, success, port, version, serial_num):
        """Update Sign Up serial field when device is detected"""
        self._device_scan_in_progress = False
        
        if success and serial_num:
            if hasattr(self, 'reg_serial'):
                self.reg_serial.setText(serial_num)
                self.reg_serial.setReadOnly(True)
                self.reg_serial.setStyleSheet(self.reg_serial.styleSheet() + " color: #27ae60; font-weight: bold;")
        else:
            if hasattr(self, 'reg_serial'):
                self.reg_serial.setText("RhythmUltra device connection lost. Please reconnect the device")
                self.reg_serial.setReadOnly(True)
                self.reg_serial.setStyleSheet(self.reg_serial.styleSheet().replace(" color: #27ae60; font-weight: bold;", ""))

    def _resize_bg(self, event):
        """Handle window resize to maintain background coverage"""
        self.bg_label.setGeometry(0, 0, self.width(), self.height())
        # Ensure the background stays behind all other widgets
        self.bg_label.lower()
        event.accept()
    
    def ensure_background_visible(self):
        """Ensure the background is always visible and properly positioned"""
        try:
            # Make sure the background label is at the bottom of the widget stack
            self.bg_label.lower()
            # Ensure it covers the entire window
            self.bg_label.setGeometry(0, 0, self.width(), self.height())
            # Make sure it's visible
            self.bg_label.setVisible(True)
            logger.info(" Background visibility ensured")
        except Exception as e:
            logger.warning(f"Background visibility issue: {e}")

    def create_login_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(12)

        input_style = """
            QLineEdit {
                border: 1px solid rgba(255, 156, 64, 0.85);
                border-radius: 14px;
                padding: 12px 14px;
                font-size: 15px;
                background: rgba(255,255,255,0.92);
                color: #1f1f1f;
                selection-background-color: #ff8a1f;
            }
            QLineEdit:focus {
                border: 2px solid #ff8a1f;
                background: rgba(255,255,255,0.98);
            }
        """
        otp_input_style = """
            QLineEdit {
                border: 1px solid rgba(75, 190, 134, 0.78);
                border-radius: 14px;
                padding: 12px 14px;
                font-size: 15px;
                background: rgba(255,255,255,0.92);
                color: #1f1f1f;
            }
            QLineEdit:focus {
                border: 2px solid #2fa66f;
                background: rgba(255,255,255,0.98);
            }
        """
        primary_button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
                color: white;
                border-radius: 14px;
                padding: 12px 0;
                font-size: 16px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff8a26, stop:1 #ffab31); }
            QPushButton:pressed { background: #e96a00; }
        """
        secondary_button_style = """
            QPushButton {
                background: rgba(58,58,58,0.62);
                color: #ffbe63;
                border: 1px solid rgba(255, 179, 71, 0.42);
                border-radius: 14px;
                padding: 11px 14px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { background: rgba(88,88,88,0.78); }
            QPushButton:pressed { background: rgba(108,108,108,0.92); }
        """

        section_title = QLabel("Sign in to continue")
        section_title.setStyleSheet("color: white; font-size: 30px; font-weight: bold;")

        section_subtitle = QLabel(
            "Use your full name or phone number, with the same password you chose at signup. "
            "No internet is required."
        )
        section_subtitle.setWordWrap(True)
        section_subtitle.setStyleSheet("color: rgba(255,255,255,0.78); font-size: 13px;")

        password_header = QLabel("ACCOUNT LOGIN")
        password_header.setStyleSheet("color: #ffb347; font-size: 12px; font-weight: bold; letter-spacing: 1px;")

        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("Full Name or Phone Number (from signup)")
        self.login_email.setMinimumHeight(44)
        self.login_email.setMaxLength(20)

        password_row = QHBoxLayout()
        password_row.setSpacing(10)
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("Password (from signup)")
        self.login_password.setEchoMode(QLineEdit.Password)
        self.login_password.setMinimumHeight(44)
        password_row.addWidget(self.login_password)

        self.login_eye_btn = QPushButton("View")
        self.login_eye_btn.setFixedSize(72, 46)
        self.login_eye_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
            color: white;
            border-radius: 14px;
            font-size: 13px;
            font-weight: bold;
            border: none;
        """)
        self.login_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.login_password, self.login_eye_btn))
        password_row.addWidget(self.login_eye_btn)

        login_btn = QPushButton("Login")
        login_btn.setObjectName("LoginBtn")
        login_btn.setMinimumHeight(46)
        login_btn.setStyleSheet(primary_button_style)
        login_btn.clicked.connect(self.handle_login)

        # Phone login (OTP) UI is intentionally disabled.
        # (Kept commented for future re-enable if needed.)
        # phone_btn = QPushButton("Send OTP")
        # phone_btn.setObjectName("SignUpBtn")
        # phone_btn.setMinimumHeight(44)
        # phone_btn.setMinimumWidth(170)
        # phone_btn.setStyleSheet(secondary_button_style)
        # phone_btn.clicked.connect(self.handle_phone_login)
        # self.phone_btn = phone_btn
        #
        # self.login_phone = QLineEdit()
        # self.login_phone.setPlaceholderText("Phone number (10 digits)")
        # self.login_phone.setMinimumHeight(44)
        # self.login_phone.setMaxLength(10)
        # self.login_phone.setValidator(QIntValidator(0, 2147483647, self))
        #
        # phone_row = QHBoxLayout()
        # phone_row.setSpacing(10)
        # phone_row.addWidget(self.login_phone, 3)
        # phone_row.addWidget(phone_btn, 0)
        #
        # self.login_otp = QLineEdit()
        # self.login_otp.setPlaceholderText("Enter 4-digit OTP")
        # self.login_otp.setMaxLength(4)
        # self.login_otp.setMinimumHeight(44)
        # self.login_otp.setValidator(QIntValidator(0, 9999, self))
        #
        # self.verify_otp_btn = QPushButton("Verify OTP")
        # self.verify_otp_btn.setMinimumHeight(44)
        # self.verify_otp_btn.setMinimumWidth(132)
        # self.verify_otp_btn.setEnabled(False)
        # self.verify_otp_btn.setStyleSheet("""
        #     QPushButton {
        #         background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1f704f, stop:1 #2fa66f);
        #         color: white;
        #         border-radius: 14px;
        #         padding: 11px 14px;
        #         font-size: 14px;
        #         font-weight: bold;
        #         border: none;
        #     }
        #     QPushButton:hover { background: #2a9b67; }
        #     QPushButton:pressed { background: #1f7b52; }
        #     QPushButton:disabled {
        #         background: rgba(35, 139, 92, 0.30);
        #         color: rgba(255,255,255,0.65);
        #     }
        # """)
        # self.verify_otp_btn.clicked.connect(self.verify_phone_otp)
        #
        # self._otp_cooldown_seconds = 60
        # self._otp_lockout_seconds = 300
        # self._otp_resend_available_at = 0.0
        # self._otp_lockout_until = 0.0
        # self._otp_failed_attempts = 0
        # self._otp_timer = QTimer(self)
        # self._otp_timer.timeout.connect(self._refresh_otp_controls)
        #
        # otp_row = QHBoxLayout()
        # otp_row.setSpacing(10)
        # otp_row.addWidget(self.login_otp, 3)
        # otp_row.addWidget(self.verify_otp_btn, 1)
        #
        # phone_card = QWidget()
        # phone_card.setStyleSheet("""
        #     background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        #         stop:0 rgba(255,255,255,0.10),
        #         stop:1 rgba(255,255,255,0.05));
        #     border: 1px solid rgba(255,255,255,0.18);
        #     border-radius: 18px;
        # """)
        # phone_card_layout = QVBoxLayout(phone_card)
        # phone_card_layout.setContentsMargins(14, 12, 14, 12)
        # phone_card_layout.setSpacing(10)
        # phone_card_title = QLabel("Phone Login")
        # phone_card_title.setStyleSheet("""
        #     color: white;
        #     font-size: 18px;
        #     font-weight: bold;
        #     padding-bottom: 4px;
        #     border-bottom: 1px solid rgba(255,255,255,0.12);
        # """)
        # phone_card_desc = QLabel("Enter your mobile number, request an OTP, then verify it to sign in.")
        # phone_card_desc.setWordWrap(True)
        # phone_card_desc.setStyleSheet("color: rgba(255,255,255,0.74); font-size: 12px;")
        # phone_card_layout.addWidget(phone_card_title)
        # phone_card_layout.addWidget(phone_card_desc)
        # phone_card_layout.addLayout(phone_row)
        # phone_card_layout.addLayout(otp_row)

        for w in [self.login_email, self.login_password, login_btn]:
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.login_email.setStyleSheet(input_style)
        self.login_password.setStyleSheet(input_style)
        # self.login_phone.setStyleSheet(input_style)
        # self.login_otp.setStyleSheet(otp_input_style)

        # divider = QFrame()
        # divider.setFrameShape(QFrame.HLine)
        # divider.setStyleSheet("background: rgba(255,255,255,0.10); max-height: 1px; min-height: 1px; border: none;")

        self.login_email.returnPressed.connect(self.handle_login)
        self.login_password.returnPressed.connect(self.handle_login)
        # self.login_phone.returnPressed.connect(self.handle_phone_login)
        # self.login_otp.returnPressed.connect(self.verify_phone_otp)
        # self.login_otp.textChanged.connect(self._update_verify_otp_button)

        layout.addWidget(section_title)
        layout.addWidget(section_subtitle)
        layout.addWidget(password_header)
        layout.addWidget(self.login_email)
        layout.addLayout(password_row)
        layout.addWidget(login_btn)
        # layout.addSpacing(4)
        # layout.addWidget(divider)
        # layout.addSpacing(4)
        # layout.addWidget(phone_card)

        # COMMENTED OUT: Home, About us, Blog, Pricing navigation menu
        # nav_row = QHBoxLayout()
        # nav_row.setSpacing(10)

        # class NavHome(QWidget):
        #     def __init__(self): super().__init__(); self.setWindowTitle("Home")
        # class NavAbout(QWidget):
        #     def __init__(self): super().__init__(); self.setWindowTitle("About")
        # class NavBlog(QWidget):
        #     def __init__(self): super().__init__(); self.setWindowTitle("Blog")
        # class NavPricing(QWidget):
        #     def __init__(self): super().__init__(); self.setWindowTitle("Pricing")

        # nav_links = [
        #     ("Home", NavHome),
        #     ("About us", NavAbout),
        #     ("Blog", NavBlog),
        #     ("Pricing", NavPricing)
        # ]
        # self.nav_stack = QStackedWidget()
        # self.nav_pages = {}

        # def show_nav_page(page_name):
        #     self.nav_stack.setCurrentWidget(self.nav_pages[page_name])
        #     self.nav_stack.setVisible(True)

        # nav_row.addStretch(1)
        # for text, NavClass in nav_links:
        #     nav_btn = QPushButton(text)
        #     nav_btn.setStyleSheet("""
        #         color: #ff9a3b;
        #         background: transparent;
        #         border: none;
        #         font-size: 14px;
        #         font-weight: bold;
        #         padding: 4px 8px;
        #     """)
        #     page = NavClass()
        #     self.nav_stack.addWidget(page)
        #     self.nav_pages[text] = page
        #     if text == "Pricing":
        #         def show_pricing_dialog():
        #             QMessageBox.information(self, "Pricing", "Pricing information not available.")
        #         nav_btn.clicked.connect(lambda checked, p=self: show_pricing_dialog())
        #     else:
        #         nav_btn.clicked.connect(lambda checked, t=text: show_nav_page(t))
        #     nav_row.addWidget(nav_btn)
        # nav_row.addStretch(1)

        # layout.addLayout(nav_row)
        # layout.addWidget(self.nav_stack)
        # self.nav_stack.setVisible(False)
        self._refresh_otp_controls()
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget

    def create_register_widget(self):
        # Scroll area prevents layout compression on smaller screens, which was cropping the org buttons.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent;")

        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(12)

        register_input_style = """
            QLineEdit {
                border: 1px solid rgba(255, 156, 64, 0.82);
                border-radius: 14px;
                padding: 11px 14px;
                font-size: 15px;
                background: rgba(255,255,255,0.92);
                color: #1f1f1f;
            }
            QLineEdit:focus {
                border: 2px solid #ff8a1f;
                background: rgba(255,255,255,0.98);
            }
        """
        self.reg_serial = QLineEdit()
        self.reg_serial.setPlaceholderText("Machine Serial ID")
        self.reg_serial.setReadOnly(True)
        self.reg_name = QLineEdit()
        self.reg_name.setPlaceholderText("Full Name")
        self.reg_doctor = QLineEdit()
        self.reg_doctor.setPlaceholderText("Doctor Name")
        self.reg_org_name = QLineEdit()
        self.reg_org_name.setPlaceholderText("Organisation Name")
        self.reg_org_address = QLineEdit()
        self.reg_org_address.setPlaceholderText("Organisation Address")
        self.reg_phone = QLineEdit()
        self.reg_phone.setPlaceholderText("Phone Number")
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("Password")
        self.reg_password.setEchoMode(QLineEdit.Password)
        
        self.reg_confirm = QLineEdit()
        self.reg_confirm.setPlaceholderText("Confirm Password")
        self.reg_confirm.setEchoMode(QLineEdit.Password)
        self.reg_confirm.returnPressed.connect(self.handle_register)

        self.reg_name.setMaxLength(20)
        self.reg_doctor.setMaxLength(20)
        self.reg_org_name.setMaxLength(28)
        self.reg_org_address.setMaxLength(45)
        self.reg_phone.setMaxLength(10)
        self.reg_phone.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d{0,10}$"), self))
        
        register_btn = QPushButton("Sign Up")
        register_btn.setObjectName("SignUpBtn")
        register_btn.clicked.connect(self.handle_register)
        
        for w in [
            self.reg_serial,
            self.reg_name,
            self.reg_doctor,
            self.reg_org_name,
            self.reg_org_address,
            self.reg_phone,
            self.reg_password,
            self.reg_confirm,
        ]:
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            w.setMinimumHeight(44)
        
        for w in [
            self.reg_serial,
            self.reg_name,
            self.reg_doctor,
            self.reg_org_name,
            self.reg_org_address,
            self.reg_phone,
            self.reg_password,
            self.reg_confirm,
        ]:
            w.setStyleSheet(register_input_style)
        
        register_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
            color: white;
            border-radius: 14px;
            padding: 11px 0;
            font-size: 16px;
            font-weight: bold;
            border: none;
        """)
        register_btn.setMinimumHeight(46)
        
        # Create password field with eye toggle
        password_row = QHBoxLayout()
        password_row.setSpacing(10)
        password_row.addWidget(self.reg_password)
        self.password_eye_btn = QPushButton("View")
        self.password_eye_btn.setFixedSize(72, 46)
        self.password_eye_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
            color: white;
            border-radius: 14px;
            font-size: 13px;
            font-weight: bold;
            border: none;
        """)
        self.password_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.reg_password, self.password_eye_btn))
        password_row.addWidget(self.password_eye_btn)
        
        # Create confirm password field with eye toggle
        confirm_row = QHBoxLayout()
        confirm_row.setSpacing(10)
        confirm_row.addWidget(self.reg_confirm)
        self.confirm_eye_btn = QPushButton("View")
        self.confirm_eye_btn.setFixedSize(72, 46)
        self.confirm_eye_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f);
            color: white;
            border-radius: 14px;
            font-size: 13px;
            font-weight: bold;
            border: none;
        """)
        self.confirm_eye_btn.clicked.connect(lambda: self.toggle_password_visibility(self.reg_confirm, self.confirm_eye_btn))
        confirm_row.addWidget(self.confirm_eye_btn)
        
        # COMMENTED OUT: Request for New Organization and Existing Organization buttons
        # Organization buttons (imported from organization module)
        # import importlib
        # organization_module = importlib.import_module('organization')
        # create_organization_buttons_layout = getattr(organization_module, 'create_organization_buttons_layout')
        # self.org_buttons_layout, self.new_org_handler, self.existing_org_handler = create_organization_buttons_layout(self)

        # layout.addLayout(self.org_buttons_layout)
        layout.addWidget(self.reg_serial)
        layout.addWidget(self.reg_name)
        layout.addWidget(self.reg_doctor)
        layout.addWidget(self.reg_org_name)
        layout.addWidget(self.reg_org_address)
        layout.addWidget(self.reg_phone)
        layout.addLayout(password_row)
        layout.addLayout(confirm_row)
        layout.addWidget(register_btn)
        layout.addStretch(1)

        # Login prompt inside the register page (so it scrolls with the form content).
        login_row = QHBoxLayout()
        login_row.addStretch(1)
        login_lbl = QLabel("Already have an account?")
        login_lbl.setStyleSheet("color: rgba(255,255,255,0.82); font-size: 14px;")
        login_btn = QPushButton("Login")
        login_btn.setStyleSheet(
            "color: #ff8d2b; background: transparent; border: none; font-size: 14px; "
            "font-weight: bold; text-decoration: underline; padding: 2px 6px;"
        )
        login_btn.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        login_row.addWidget(login_lbl)
        login_row.addWidget(login_btn)
        login_row.addStretch(1)
        layout.addSpacing(12)
        layout.addLayout(login_row)

        scroll.setWidget(widget)
        return scroll

    def _login_limiter(self):
        """Per-window failed-sign-in counter, created on first use."""
        try:
            if getattr(self, "_login_rate_limiter", None) is None:
                from utils.input_validation import LoginRateLimiter
                self._login_rate_limiter = LoginRateLimiter()
            return self._login_rate_limiter
        except Exception:
            return None

    def handle_login(self):
        identifier = self.login_email.text().strip()
        password_or_serial = self.login_password.text()
        if not identifier or not password_or_serial:
            QMessageBox.warning(
                self,
                "Login Required",
                "Please enter your full name or phone number, and the password you used at signup.",
            )
            return

        # Throttle repeated failures. The OTP path already locked after 3 tries;
        # the password path counted nothing, so the form would accept guesses as
        # fast as they could be typed. This bounds guessing at the UI, which is
        # the attack this application can actually see — it is not a substitute
        # for the PBKDF2 work factor protecting users.json on disk.
        limiter = self._login_limiter()
        if limiter is not None and limiter.is_locked(identifier):
            wait_s = limiter.seconds_remaining(identifier)
            QMessageBox.warning(
                self,
                "Too Many Attempts",
                "Too many failed sign-in attempts for this account.\n\n"
                f"Try again in {max(1, wait_s // 60)} minute(s).",
            )
            return

        # ── Start the credential check while the licence gate runs ────────────
        # Both stages make a network round-trip and each takes ~4 s, so running
        # them one after the other left the user staring at the login screen for
        # ~8 s. They are independent — the licence check asks the server about
        # this device, the credential check about this account — so the slower of
        # the two now sets the wait instead of their sum. Users are reloaded first
        # because validate_credentials() reads them on the worker thread.
        import threading as _login_threading

        try:
            self.sign_in_logic.users = self.sign_in_logic.load_users()
        except Exception:
            pass

        _cred_ok = [None]
        _cred_done = _login_threading.Event()

        def _run_credential_check():
            try:
                _cred_ok[0] = bool(
                    self.sign_in_logic.validate_credentials(identifier, password_or_serial)
                )
            except Exception as _ce:
                logger.warning(f"Credential check failed: {_ce}")
                _cred_ok[0] = False
            finally:
                _cred_done.set()

        _login_threading.Thread(
            target=_run_credential_check, daemon=True, name="LoginCredentialCheck"
        ).start()

        # Enforce license key check at login step
        try:
            from utils.license_manager import (
                load_stored_key,
                token_file_exists,
                load_token_file,
                run_startup_checks,
                clear_stored_key,
                clear_license_cache,
                is_stale_seat_error,
                resync_token_from_credentials,
                OFFLINE_WARNING_DAYS,
            )
            stored_key = load_stored_key()
            if not stored_key and not token_file_exists():
                QMessageBox.warning(
                    self,
                    "License Required",
                    "No valid license key found on this system.\n\nPlease register or activate a license in the Sign Up tab first.",
                )
                return

            # Always run the server check. A local token file only proves this
            # machine was activated at some point — it says nothing about the
            # current server-side state, so short-circuiting on it would let
            # revoked seats log in offline-style. Revocation is only detected by
            # the Check-5 heartbeat inside run_startup_checks().
            result = run_startup_checks(force_heartbeat=False)

            # ── Stale seat: repair the cached token instead of the licence ────
            # The server moves a device onto a new seat whenever it registers
            # again and deactivates the old one, so cardiox.lic can point at a
            # dead seat while the licence is perfectly healthy. The user is
            # standing right here with valid credentials, so fetch the current
            # token for this device and re-run the check — silently, because
            # nothing is wrong from their side.
            if (
                not result.ok
                and result.step_failed == 5
                and not getattr(result, "offline_mode", False)
                and is_stale_seat_error(getattr(result, "error_code", ""), result.reason)
            ):
                logger.info(
                    f"Login: stale seat reported ({result.error_code}); "
                    "re-syncing licence token from server."
                )
                if resync_token_from_credentials(identifier, password_or_serial):
                    result = run_startup_checks(force_heartbeat=False)
                    stored_key = load_stored_key() or stored_key

            if not result.ok:
                logger.warning(f"Login blocked due to license failure: {result.reason}")
                is_explicit_revocation = (
                    result.step_failed == 5
                    and getattr(result, "error_code", "") == "LICENSE_REVOKED"
                )

                # ── Seat-missing: offer re-registration ───────────────────────
                # Only for a seat the server genuinely does not have. The generic
                # LICENSE_BLOCKED fallback is NOT included: it is what
                # run_startup_checks assigns to any refusal it cannot name, and
                # treating it as a missing seat offered to wipe healthy licences
                # over transient server errors.
                reason_lower = result.reason.lower()
                error_code_upper = getattr(result, "error_code", "").upper()
                _seat_missing_phrases = (
                    "seat not found",
                    "not registered under this license",
                    "please register first",
                    "seat has been flagged",
                    "seat unavailable",
                )
                is_seat_missing = (
                    result.step_failed == 5
                    and not is_explicit_revocation
                    and (
                        error_code_upper == "SEAT_NOT_FOUND"
                        or any(p in reason_lower for p in _seat_missing_phrases)
                    )
                )

                if is_seat_missing:
                    reply = QMessageBox.question(
                        self,
                        "Re-register Device",
                        f"Your device seat was not found on the license server.\n\n"
                        f"License key on file: {stored_key}\n\n"
                        "This usually happens after a server update or account reset.\n"
                        "Click Yes to clear the stale license and open the Sign Up screen, "
                        "or No to cancel login.",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    if reply == QMessageBox.Yes:
                        clear_stored_key()
                        clear_license_cache()
                        logger.info(
                            "Login: cleared stale seat after seat-not-found; redirecting to Sign Up tab."
                        )
                        # Switch to Sign Up tab (index 1)
                        try:
                            self.stacked.setCurrentIndex(1)
                        except Exception:
                            pass
                    return
                elif is_stale_seat_error(error_code_upper, result.reason):
                    # The re-sync above could not reach or match this device's
                    # current seat. Nothing here is the user's licence being
                    # invalid, so nothing is wiped — tell them what actually
                    # helps instead of sending them back through Sign Up.
                    QMessageBox.critical(
                        self,
                        "License Verification Required",
                        "This device's licence record is out of date and could not "
                        "be refreshed automatically.\n\n"
                        "Check your internet connection and sign in with the phone "
                        "number you used at signup.\n\n"
                        "If this keeps happening, contact Deckmount support and quote "
                        f"license key {stored_key}.",
                    )
                    return
                else:
                    QMessageBox.critical(
                        self,
                        {
                            1: "License Missing",
                            2: "License Integrity Failed",
                            3: "Fingerprint Mismatch",
                            4: "RhythmUltra Device Required",
                            5: "License Verification Required",
                        }.get(result.step_failed, "License Blocked") if not is_explicit_revocation else "License Revoked",
                        # result.reason is REVOKED_MESSAGE for revocation, which
                        # already tells the user to contact Deckmount support.
                        result.reason
                        + ("\n\nAll license details have been removed from this device."
                           if is_explicit_revocation else ""),
                    )
                    if result.step_failed in {1, 2} or is_explicit_revocation:
                        # Wipe every license credential on the device: key file,
                        # legacy cache, sidecar metadata and the signed token.
                        #
                        # Guarded on its own: this block is housekeeping AFTER the
                        # decision to deny has already been made, so nothing in here
                        # may be allowed to propagate out and skip the `return` below.
                        # A revoked seat once reached the dashboard exactly that way —
                        # ECGLogger.info() takes a single pre-formatted message, and
                        # passing %s-style arguments raised TypeError, which the outer
                        # handler swallowed before falling through into normal login.
                        blocked_because = (
                            "revoked" if is_explicit_revocation
                            else f"step {result.step_failed}"
                        )
                        try:
                            clear_stored_key()
                            clear_license_cache()
                            logger.info(
                                f"Login blocked ({blocked_because}): cleared all license "
                                "credentials; redirecting to Sign Up tab."
                            )
                        except Exception as wipe_err:
                            logger.warning(
                                f"License credential wipe reported an error: {wipe_err}"
                            )
                        # Send the user to Sign Up (index 1) so they can register
                        # again — without a license they cannot get past this gate.
                        try:
                            self.stacked.setCurrentIndex(1)
                        except Exception:
                            pass
                    return
            # Licence is valid. If we are running on the offline grace window and it
            # is nearly exhausted, warn on every launch for the final
            # OFFLINE_WARNING_DAYS days. The server heartbeat is mandatory — it is what
            # stops a one-time signup being used indefinitely offline — so the cutoff
            # must never arrive as a surprise.
            if (
                getattr(result, "offline_mode", False)
                and result.days_remaining is not None
                and result.days_remaining <= OFFLINE_WARNING_DAYS
            ):
                logger.warning(
                    f"Offline grace nearly exhausted: {result.days_remaining} day(s) left."
                )
                QMessageBox.warning(self, "Internet Connection Required", result.reason)

        except Exception as le:
            # FAIL CLOSED. Every deny path inside the try above returns on its own,
            # so the only way execution legitimately reaches past it is a license
            # that passed all five checks. Reaching here means the licence gate
            # itself malfunctioned, and continuing would hand out access without a
            # verified licence — which is how a revoked seat previously logged in.
            logger.warning(f"Failed to check license during login: {le}")
            QMessageBox.critical(
                self,
                "License Verification Failed",
                "The license on this device could not be verified.\n\n"
                "Please restart the application. If the problem persists, "
                "contact Deckmount support.",
            )
            return
        # Collect the credential check started before the licence gate. It has had
        # the whole licence round-trip to finish, so this normally returns at once.
        # The generous timeout is a backstop for a stalled socket; on expiry the
        # result is None and login is denied rather than granted.
        _cred_done.wait(timeout=20.0)
        if _cred_ok[0]:
            if limiter is not None:
                limiter.record_success(identifier)
            found = self.sign_in_logic._find_user_record(identifier)
            if found:
                username, record = found
                self.result = True
                self.username = username
                self.user_details = record  # Store full user details
                self.accept()
            else:
                self.result = True
                self.username = identifier
                self.user_details = {}
                self.accept()
        else:
            locked_now = limiter.record_failure(identifier) if limiter is not None else False
            # The wording never distinguishes a wrong identifier from a wrong
            # password. Telling them apart would turn this form into an account
            # enumeration oracle for anyone with access to the machine.
            if locked_now:
                QMessageBox.warning(
                    self,
                    "Too Many Attempts",
                    "Too many failed sign-in attempts for this account.\n\n"
                    f"Sign-in is locked for {max(1, limiter.lockout_seconds // 60)} minute(s).",
                )
            else:
                remaining = limiter.attempts_remaining(identifier) if limiter is not None else 0
                tail = (
                    f"\n\n{remaining} attempt(s) remaining before sign-in is locked."
                    if remaining else ""
                )
                QMessageBox.warning(
                    self,
                    "Login Failed",
                    "Invalid full name / phone number or password.\n\n"
                    "Use the same full name or phone number and password you entered at signup. "
                    "Internet is not required for sign-in." + tail,
                )

    def _upsert_phone_login_user(self, phone: str, token: str):
        from datetime import datetime

        # Use the same user store as normal password login (auth/sign_in.py),
        # otherwise OTP-created passwords can be saved to a different users.json
        # and then fail validation at next login.
        try:
            users = self.sign_in_logic.load_users()
        except Exception:
            users = load_users()
        user_key = phone
        user_record = None
        source_key = None

        for username, record in users.items():
            if str(record.get('phone', '')).strip() == phone:
                user_key = phone
                source_key = username
                user_record = dict(record)
                break

        if not isinstance(user_record, dict):
            user_record = {}

        if not user_record.get('signup_date'):
            user_record['signup_date'] = datetime.now().strftime("%Y-%m-%d")

        user_record['phone'] = phone
        user_record['contact'] = phone
        user_record['username'] = phone
        user_record['login_username'] = phone
        user_record['login_identifier'] = phone
        user_record['canonical_username'] = phone
        user_record['master_phone'] = phone
        user_record['auth_provider'] = 'ecg_otp_backend'
        user_record['jwt_token'] = token
        user_record['last_phone_login_at'] = datetime.now().isoformat()

        users[user_key] = user_record
        if source_key and source_key != user_key and source_key in users:
            try:
                del users[source_key]
            except Exception:
                pass
        try:
            self.sign_in_logic.users = users
            self.sign_in_logic.save_users()
        except Exception:
            save_users(users)
        return user_key, user_record

    def _save_phone_user_password(self, username: str, password: str):
        from datetime import datetime
        canonical_username = self._get_inline_phone_number() or username

        try:
            users = self.sign_in_logic.load_users()
        except Exception:
            users = load_users()
        record = users.get(canonical_username, users.get(username, {}))
        if not isinstance(record, dict):
            record = {}
        record['username'] = canonical_username
        record['login_username'] = canonical_username
        record['login_identifier'] = canonical_username
        record['canonical_username'] = canonical_username
        record['master_phone'] = canonical_username
        if username and username != canonical_username and username in users:
            try:
                del users[username]
            except Exception:
                pass
        record['password'] = password
        record['password_set_via'] = 'phone_otp'
        record['password_set_at'] = datetime.now().isoformat()
        users[canonical_username] = record
        try:
            self.sign_in_logic.users = users
            self.sign_in_logic.save_users()
        except Exception:
            save_users(users)
        return record

    def _prompt_phone_password_setup(self, phone: str) -> str:
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Password")
        dialog.setModal(True)
        dialog.setMinimumWidth(420)
        dialog.setStyleSheet("""
            QDialog { background: #141a2c; border-radius: 16px; }
            QLabel { color: white; font-size: 13px; }
            QLineEdit {
                border: 1px solid rgba(255, 156, 64, 0.82);
                border-radius: 12px;
                padding: 10px 12px;
                font-size: 14px;
                background: rgba(255,255,255,0.94);
                color: #1f1f1f;
            }
            QPushButton {
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 13px;
                font-weight: bold;
                border: none;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)

        title = QLabel(f"Phone {phone} verified successfully.")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        desc = QLabel("Create a password once so next time you can log in directly with your phone number and password.")
        desc.setWordWrap(True)

        password_input = QLineEdit()
        password_input.setPlaceholderText("Create Password")
        password_input.setEchoMode(QLineEdit.Password)

        confirm_input = QLineEdit()
        confirm_input.setPlaceholderText("Confirm Password")
        confirm_input.setEchoMode(QLineEdit.Password)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        skip_btn = QPushButton("Skip")
        skip_btn.setStyleSheet("background: rgba(255,255,255,0.12); color: #ffd2a3;")
        save_btn = QPushButton("Save Password")
        save_btn.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff7a12, stop:1 #ff950f); color: white;")

        btn_row.addWidget(skip_btn)
        btn_row.addWidget(save_btn)

        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addWidget(password_input)
        layout.addWidget(confirm_input)
        layout.addLayout(btn_row)

        result = {"password": ""}

        def _skip():
            dialog.reject()

        def _save():
            password = password_input.text().strip()
            confirm = confirm_input.text().strip()
            if len(password) < 4:
                QMessageBox.warning(dialog, "Password Required", "Password must be at least 4 characters long.")
                return
            if password != confirm:
                QMessageBox.warning(dialog, "Password Mismatch", "Password and confirm password must match.")
                return
            result["password"] = password
            dialog.accept()

        skip_btn.clicked.connect(_skip)
        save_btn.clicked.connect(_save)
        password_input.returnPressed.connect(_save)
        confirm_input.returnPressed.connect(_save)

        dialog.exec_()
        return result["password"]

    def _ensure_phone_user_password(self, username: str, user_record: dict, phone: str):
        if not isinstance(user_record, dict):
            user_record = {}
        if str(user_record.get('password', '')).strip():
            return user_record

        reply = QMessageBox.question(
            self,
            "Create Password",
            "Do you want to create a password for this phone login?\n\nYou can use it next time with phone number + password.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return user_record

        new_password = self._prompt_phone_password_setup(phone)
        if not new_password:
            return user_record

        updated_record = self._save_phone_user_password(username, new_password)
        QMessageBox.information(
            self,
            "Password Saved",
            "Password created successfully. Next time you can sign in using your phone number and password.",
        )
        return updated_record

    def _get_inline_phone_number(self) -> str:
        auth_api = get_ecg_auth_api()
        raw_phone = self.login_phone.text().strip() if hasattr(self, "login_phone") else ""
        return auth_api.normalize_phone(raw_phone)

    def _update_verify_otp_button(self):
        otp = self.login_otp.text().strip() if hasattr(self, "login_otp") else ""
        if hasattr(self, "verify_otp_btn"):
            locked = self._is_otp_locked()
            self.verify_otp_btn.setEnabled(len(otp) == 4 and otp.isdigit() and not locked)

    def _is_otp_locked(self) -> bool:
        import time
        now = time.time()
        return now < getattr(self, "_otp_lockout_until", 0.0)

    def _otp_resend_cooldown_remaining(self) -> int:
        import time
        remaining = int(getattr(self, "_otp_resend_available_at", 0.0) - time.time())
        return max(0, remaining)

    def _refresh_otp_controls(self):
        if not hasattr(self, "phone_btn") or not hasattr(self, "verify_otp_btn"):
            return

        resend_remaining = self._otp_resend_cooldown_remaining()
        locked = self._is_otp_locked()

        if locked:
            import time
            remaining = int(getattr(self, "_otp_lockout_until", 0.0) - time.time())
            remaining = max(0, remaining)
            self.phone_btn.setEnabled(False)
            self.phone_btn.setText(f"Wait {remaining}s")
            self.verify_otp_btn.setEnabled(False)
            self.verify_otp_btn.setText(f"Verify OTP ({remaining}s)")
            if remaining == 0:
                self._otp_lockout_until = 0.0
                self._otp_failed_attempts = 0
                self.phone_btn.setEnabled(True)
                self.phone_btn.setText("Send OTP")
                self.verify_otp_btn.setText("Verify OTP")
                self._update_verify_otp_button()
            return

        if resend_remaining > 0:
            self.phone_btn.setEnabled(False)
            self.phone_btn.setText(f"Wait {resend_remaining}s")
        else:
            self.phone_btn.setEnabled(True)
            self.phone_btn.setText("Send OTP")
            self.verify_otp_btn.setText("Verify OTP")
            if not self._otp_timer.isActive():
                self._otp_timer.stop()

        self._update_verify_otp_button()

    def handle_phone_login(self):
        # Check internet connection first
        try:
            if not get_offline_queue().is_online():
                from utils.ui_feedback import show_critical, offline_action_message
                show_critical(
                    self,
                    "No Internet Connection",
                    offline_action_message(
                        "Sending and verifying OTP",
                        "Phone login is only available when the network is up.",
                    ),
                )
                return
        except Exception as e:
            logger.warning(f"Failed to check connectivity: {e}")

        auth_api = get_ecg_auth_api()
        normalized_phone = self._get_inline_phone_number()
        if len(normalized_phone) != 10:
            QMessageBox.warning(self, "Invalid Phone Number", "Phone number must be exactly 10 digits.")
            return

        if self._is_otp_locked():
            QMessageBox.warning(self, "OTP Locked", "Too many failed OTP attempts. Please wait before trying again.")
            self._refresh_otp_controls()
            return

        resend_remaining = self._otp_resend_cooldown_remaining()
        if resend_remaining > 0:
            QMessageBox.information(self, "Please Wait", f"You can request another OTP in {resend_remaining} seconds.")
            self._refresh_otp_controls()
            return

        try:
            auth_api.send_otp(normalized_phone)
            QMessageBox.information(self, "OTP Sent", f"OTP sent successfully to +91 {normalized_phone}.")
            import time
            self._otp_resend_available_at = time.time() + getattr(self, "_otp_cooldown_seconds", 60)
            self._otp_timer.start(2000 if is_low_spec_mode() else 1000)
            self._refresh_otp_controls()
        except Exception as e:
            logger.error(f"OTP send failed for {normalized_phone}: {e}")
            QMessageBox.warning(self, "OTP Failed", f"Could not send OTP: {e}")
            return

    def verify_phone_otp(self):
        # Check internet connection first
        try:
            if not get_offline_queue().is_online():
                from utils.ui_feedback import show_critical, offline_action_message
                show_critical(
                    self,
                    "No Internet Connection",
                    offline_action_message(
                        "Verifying OTP",
                        "Please connect to the internet and try again.",
                    ),
                )
                return
        except Exception as e:
            logger.warning(f"Failed to check connectivity: {e}")
            
        normalized_phone = self._get_inline_phone_number()
        if len(normalized_phone) != 10:
            QMessageBox.warning(self, "Invalid Phone Number", "Phone number must be exactly 10 digits.")
            return

        otp = self.login_otp.text().strip() if hasattr(self, "login_otp") else ""
        if len(otp) != 4 or not otp.isdigit():
            QMessageBox.warning(self, "OTP Required", "OTP must be exactly 4 digits.")
            return

        if self._is_otp_locked():
            QMessageBox.warning(self, "OTP Locked", "Too many failed OTP attempts. Please wait before trying again.")
            self._refresh_otp_controls()
            return

        auth_api = get_ecg_auth_api()
        try:
            verify_result = auth_api.verify_otp(normalized_phone, otp)
            token = verify_result.get('token', '')
            if not token:
                raise ValueError("JWT token missing from verify OTP response.")

            try:
                from utils.backend_api import get_backend_api
                get_backend_api().set_token(token)
            except Exception as token_error:
                logger.warning(f"Could not propagate JWT token to backend API helper: {token_error}")

            username, user_record = self._upsert_phone_login_user(normalized_phone, token)
            user_record = self._ensure_phone_user_password(username, user_record, normalized_phone)
            self._otp_failed_attempts = 0
            self._otp_lockout_until = 0.0
            self._otp_resend_available_at = 0.0
            if hasattr(self, "_otp_timer"):
                self._otp_timer.stop()
            self._refresh_otp_controls()
            self.result = True
            self.username = username
            self.user_details = user_record
            QMessageBox.information(self, "Phone Login", f"OTP verified for {normalized_phone}.")
            self.accept()
        except Exception as e:
            logger.error(f"OTP verification failed for {normalized_phone}: {e}")
            error_text = str(e).lower()
            if "otp" in error_text or "invalid" in error_text or "incorrect" in error_text:
                self._otp_failed_attempts = getattr(self, "_otp_failed_attempts", 0) + 1
                if self._otp_failed_attempts >= 3:
                    import time
                    self._otp_lockout_until = time.time() + getattr(self, "_otp_lockout_seconds", 300)
                    self._otp_failed_attempts = 0
                    self._otp_timer.start(2000 if is_low_spec_mode() else 1000)
                    self._refresh_otp_controls()
                QMessageBox.warning(
                    self,
                    "Incorrect OTP",
                    "Incorrect OTP. Please enter the 4-digit OTP again.\n"
                    "After 3 failed attempts, OTP verification will pause for a cooling period.",
                )
            else:
                QMessageBox.warning(self, "Verification Failed", f"Could not verify OTP: {e}")

    def handle_register(self):
        serial_id = self.reg_serial.text().strip()
        if serial_id in ("RhythmUltra device connection lost. Please reconnect the device", "RhythmUltra device connection lost. Please reconnect the device", ""):
            serial_id = ""
        name = self.reg_name.text().strip()
        doctor = self.reg_doctor.text().strip()
        org_name = self.reg_org_name.text().strip()
        org_address = self.reg_org_address.text().strip()
        phone = self.reg_phone.text().strip()
        password = self.reg_password.text()
        confirm = self.reg_confirm.text()
        if not all([name, doctor, org_name, org_address, phone, password, confirm]):
            QMessageBox.warning(self, "Error", "All fields are required.")
            return

        # Shared limits (src/utils/input_validation.py). The phone was already
        # checked here; the name, doctor, organisation and address fields were
        # not bounded at all, so an arbitrarily long or control-character-laden
        # value could reach users.json, the cloud payload and the PDF header.
        from utils.input_validation import (
            validate_name, validate_password, validate_phone, validate_text,
            validate_serial, ADDRESS_MAX_LENGTH, ORG_NAME_MAX_LENGTH,
        )

        checks = (
            validate_name(name, "Name"),
            validate_name(doctor, "Doctor name"),
            validate_name(org_name, "Organisation name", max_length=ORG_NAME_MAX_LENGTH),
            validate_text(org_address, "Organisation address", ADDRESS_MAX_LENGTH),
            validate_phone(phone, "Phone number"),
            validate_serial(serial_id, "Serial ID", required=False),
        )
        for ok, _cleaned, err in checks:
            if not ok:
                QMessageBox.warning(self, "Error", err)
                return
        name, doctor, org_name, org_address, phone, serial_id = (c[1] for c in checks)

        if password != confirm:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return

        ok_pwd, password, err_pwd = validate_password(password)
        if not ok_pwd:
            QMessageBox.warning(self, "Error", err_pwd)
            return

        # Show non-blocking translucent loading overlay dialog
        self.loading_overlay = LoadingOverlayDialog(self)
        
        # Instantiate and run RegisterWorker thread
        self.register_worker = RegisterWorker(
            name=name,
            doctor=doctor,
            org_name=org_name,
            org_address=org_address,
            phone=phone,
            password=password,
            serial_id=serial_id
        )
        
        # Connect signals
        self.register_worker.finished.connect(self.on_register_finished)
        self.register_worker.error.connect(self.on_register_error)
        
        # Start worker thread
        self.register_worker.start()
        
        # Display the dialog modal
        self.loading_overlay.exec_()

    def on_register_finished(self, res):
        # Close the loading overlay first
        if hasattr(self, 'loading_overlay') and self.loading_overlay:
            self.loading_overlay.accept()
            self.loading_overlay = None

        # Check if server registration succeeded
        success = res.get("valid") or res.get("success") or res.get("authorized")
        if not success:
            msg = res.get("message") or res.get("error") or "License registration failed."
            err_code = str(res.get("error", "")).strip().upper()
            if err_code == "DEVICE_ALREADY_REGISTERED":
                from utils.ui_feedback import show_critical
                show_critical(
                    self,
                    "Device Limit Reached",
                    "Device limit reached (5/5 machines active). Please contact Deckmount support to free up a slot."
                )
                return
            from utils.ui_feedback import is_network_error, offline_action_message, show_critical
            if is_network_error(msg):
                show_critical(
                    self,
                    "No Internet Connection",
                    offline_action_message(
                        "First-time activation",
                        "Connect to the internet, then retry signup.",
                    ),
                    details=str(msg),
                )
            else:
                show_critical(
                    self,
                    "License Error",
                    f"Registration failed: {msg}",
                )
            return

        # Success! Process the returned token/key
        try:
            from utils.license_manager import save_token_file, save_stored_key, get_hardware_fingerprint, remember_valid_license
            assigned_key = res.get("license_key")
            token_str = res.get("token")
            
            if token_str:
                try:
                    import json as _json
                    if isinstance(token_str, dict):
                        save_token_file(token_str.get("payload", token_str))
                    elif isinstance(token_str, str) and token_str.count(".") == 2:
                        save_token_file(token_str)
                    else:
                        token_envelope = _json.loads(token_str)
                        save_token_file(token_envelope.get("payload", token_envelope))
                except Exception as te:
                    print(f"Could not parse token: {te}")
                    remember_valid_license(assigned_key, get_hardware_fingerprint(), res)
            else:
                remember_valid_license(assigned_key, get_hardware_fingerprint(), res)
                
            save_stored_key(assigned_key)
            print(f"✅ Device registered internally under license key: {assigned_key}")
        except Exception as e:
            QMessageBox.critical(self, "Registration Error", f"Failed to save license/token details: {e}")
            return

        # Save local user credentials in users.json
        name = self.reg_name.text().strip()
        doctor = self.reg_doctor.text().strip()
        org_name = self.reg_org_name.text().strip()
        org_address = self.reg_org_address.text().strip()
        phone = self.reg_phone.text().strip()
        password = self.reg_password.text()
        serial_id = self.reg_serial.text().strip()
        if serial_id in ("RhythmUltra device connection lost. Please reconnect the device", "RhythmUltra device connection lost. Please reconnect the device", "RUM") or not serial_id:
            serial_id = ""

        ok, msg = self.sign_in_logic.register_user_with_details(
            username=phone,
            password=password,
            full_name=name,
            phone=phone,
            serial_id=serial_id,
            email="",
            extra={
                "doctor": doctor,
                "org_name": org_name,
                "org_address": org_address,
                "login_id": phone,
                "login_username": phone,
                "login_identifier": phone,
                "canonical_username": phone,
                "username": phone,
            }
        )
        if not ok:
            QMessageBox.warning(self, "Error", msg)
            return

        # Upload user signup details to cloud
        try:
            from utils.cloud_uploader import get_cloud_uploader
            from datetime import datetime
            
            uploader = get_cloud_uploader()
            user_data = {
                'username': phone,
                'full_name': name,
                'doctor': doctor,
                'org_name': org_name,
                'org_address': org_address,
                'phone': phone,
                'serial_number': serial_id,
                'serial_id': serial_id,
                'machine_serial_id': serial_id,
                'registered_at': datetime.now().isoformat()
            }
            upload_result = uploader.upload_user_signup(user_data)
            print(f" Signup upload status: {upload_result.get('status', 'unknown')}")
        except Exception as e:
            print(f" Error uploading user signup: {e}")

        QMessageBox.information(
            self,
            "Registration Successful",
            "Your account has been created successfully.\n\n"
            "Save these details — use them every time you sign in (no internet needed):\n\n"
            f"Full Name: {name}\n"
            f"Phone Number: {phone}\n"
            f"Password: {password}\n\n"
            "On the login screen, enter your full name OR phone number, "
            "with the same password shown above."
        )

        self.login_email.setText(phone)
        self.login_password.setText(password)
        self.login_password.setFocus()
        
        # Clear register form inputs
        self.reg_name.clear()
        self.reg_doctor.clear()
        self.reg_org_name.clear()
        self.reg_org_address.clear()
        self.reg_phone.clear()
        self.reg_password.clear()
        self.reg_confirm.clear()
        
        # Transition to Login Tab (index 0)
        self.stacked.setCurrentIndex(0)

    def on_register_error(self, err_msg):
        # Close loading dialog
        if hasattr(self, 'loading_overlay') and self.loading_overlay:
            self.loading_overlay.accept()
            self.loading_overlay = None
        from utils.ui_feedback import is_network_error, offline_action_message, show_critical
        if is_network_error(err_msg):
            show_critical(
                self,
                "No Internet Connection",
                offline_action_message(
                    "First-time activation",
                    "The software needs internet for the initial license allocation.",
                ),
                details=str(err_msg),
            )
        else:
            show_critical(
                self,
                "Registration Error",
                f"Failed to connect to the registration server.\n\n{err_msg}",
                details=str(err_msg),
            )
    
    def toggle_password_visibility(self, password_field, eye_button):
        """Toggle password visibility between hidden and visible"""
        if password_field.echoMode() == QLineEdit.Password:
            password_field.setEchoMode(QLineEdit.Normal)
            eye_button.setText("Hide")
        else:
            password_field.setEchoMode(QLineEdit.Password)
            eye_button.setText("View")

    def _show_nav_window(self, NavClass, text):
        nav_win = NavClass()
        nav_win.setWindowTitle(text)
        nav_win.setMinimumSize(400, 300)
        nav_win.show()
        if not hasattr(self, '_nav_windows'):
            self._nav_windows = []
        self._nav_windows.append(nav_win)


@log_function_call
def main():
    """Main application entry point with proper error handling"""
    try:
        # Initialize crash logger first
        crash_logger = get_crash_logger()
        crash_logger.log_info("Application starting", "APP_START")
        
        # Enable High-DPI scaling for Retina displays (e.g. MacBook) and high-res screens
        if hasattr(Qt, "AA_EnableHighDpiScaling"):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, "AA_UseHighDpiPixmaps"):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        if hasattr(Qt, "HighDpiScaleFactorRoundingPolicy"):
            try:
                QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
            except Exception:
                pass

        # Set AppUserModelID so Windows taskbar displays the custom window icon when run from Python
        if sys.platform == "win32":
            try:
                import ctypes
                myappid = "CardioX.1.1.0"
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except Exception as e:
                print(f"[MainApp] Failed to set AppUserModelID: {e}")

        app = QApplication(sys.argv)
        app.setApplicationName("CardioX")
        app.setApplicationVersion(APP_VERSION)

        # TODO: Re-enable single-instance detection before release. It is
        # commented out only so multiple CardioX windows can be opened for
        # testing. Uncomment the block below to restore it.
        # # ── Single-instance detection using QLocalSocket/QLocalServer ──
        # # Prevents multiple CardioX instances and allows restoring minimized window
        # SERVER_NAME = "CardioX-SingleInstance"
        
        # # Try to connect to existing server (another instance is running)
        # socket = QLocalSocket()
        # socket.connectToServer(SERVER_NAME)
        # if socket.waitForConnected(1000):
        #     # Another instance is running - send message to restore it
        #     print("[MainApp] CardioX is already running. Sending restore request...")
        #     socket.write(b"RESTORE")
        #     socket.waitForBytesWritten(1000)
        #     socket.disconnectFromServer()
        #     sys.exit(0)
        
        # # Remove any stale server/socket file left behind by a crash
        # QLocalServer.removeServer(SERVER_NAME)

        # # No existing instance - create server to listen for future instances
        # local_server = QLocalServer()
        # if not local_server.listen(SERVER_NAME):
        #     print(f"[MainApp] Failed to start local server: {local_server.errorString()}")
        #     QMessageBox.warning(
        #         None,
        #         "CardioX Error",
        #         f"Failed to initialize single-instance server: {local_server.errorString()}\n\nPlease ensure no other CardioX instances are running."
        #     )
        #     sys.exit(1)
        
        # # Store reference to server for restoration
        # _local_server = local_server
        
        # def handle_new_connection():
        #     """Called when another instance tries to start - restore this window"""
        #     connection = _local_server.nextPendingConnection()
        #     connection.readyRead.connect(lambda: on_socket_ready_read(connection))
        
        # def on_socket_ready_read(socket):
        #     """Process incoming message from another instance"""
        #     data = socket.readAll()
        #     if b"RESTORE" in data:
        #         from PyQt5.QtWidgets import QApplication
        #         from PyQt5.QtCore import Qt, QTimer

        #         def _do_restore():
        #             main_win = None
        #             for widget in QApplication.topLevelWidgets():
        #                 if not widget.isWindow() or widget.isHidden():
        #                     continue
        #                 if widget.objectName() in ["UpdateBanner", "LoadingOverlayDialog"]:
        #                     continue
        #                 main_win = widget
        #                 if widget.__class__.__name__ in ["Dashboard", "DashboardWindow", "LoginRegisterDialog"]:
        #                     break

        #             if main_win:
        #                 # 1. Un-minimize / restore window state
        #                 if main_win.isMinimized():
        #                     if (main_win.windowState() & Qt.WindowMaximized) or main_win.__class__.__name__ == "Dashboard":
        #                         main_win.showMaximized()
        #                     else:
        #                         main_win.showNormal()
        #                 else:
        #                     if main_win.__class__.__name__ == "Dashboard":
        #                         main_win.showMaximized()
        #                     else:
        #                         main_win.show()

        #                 main_win.raise_()
        #                 main_win.activateWindow()

        #                 # 2. Windows foreground activation bypass
        #                 if sys.platform == "win32":
        #                     try:
        #                         import ctypes
        #                         hwnd = int(main_win.winId())
        #                         if main_win.isMaximized():
        #                             ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_SHOWMAXIMIZED (3)
        #                         else:
        #                             ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE (9)
        #                         ctypes.windll.user32.SetForegroundWindow(hwnd)
        #                     except Exception as e:
        #                         print(f"[MainApp] Error bringing window to front via win32: {e}")

        #         QTimer.singleShot(0, _do_restore)

        #     socket.disconnectFromServer()
        #     socket.deleteLater()
        
        # local_server.newConnection.connect(handle_new_connection)
        # # ──────────────────────────────────────────────────────────────

        # Set application icon
        try:
            from PyQt5.QtGui import QIcon
            icon_path = resource_path("assets/cardiox_logo.ico")
            if os.path.exists(icon_path):
                app.setWindowIcon(QIcon(icon_path))
        except Exception as e:
            logger.warning(f"Failed to set application window icon: {e}")

        # ── Pre-warm heavy imports + WMI in background ──────────────
        # Skip on low-spec machines to avoid an early CPU/RAM spike.
        import threading as _startup_threading
        if not is_low_spec_mode():
            # matplotlib, scipy, pyqtgraph take 2-5s on first import.
            # WMI hardware queries (BIOS/CPU/disk) can each take 1-2s.
            # Start both NOW so results are cached by the time the user
            # finishes typing their password.
            def _prewarm():
                try:
                    import matplotlib; matplotlib.use('Agg')
                    import matplotlib.pyplot
                    import scipy.signal
                    import scipy.ndimage
                    import pyqtgraph
                except Exception:
                    pass

            def _prewarm_wmi():
                """Cache WMI hardware fields so startup license check is instant."""
                try:
                    from utils.license_manager import _collect_wmi_fields
                    _collect_wmi_fields()
                except Exception:
                    pass

            _startup_threading.Thread(target=_prewarm, daemon=True, name="Prewarm").start()
            _startup_threading.Thread(target=_prewarm_wmi, daemon=True, name="PrewarmWMI").start()
        # ──────────────────────────────────────────────────────────────

        # ── Non-blocking startup update check ────────────────────────
        # Run the GitHub release check in a background thread and wait
        # at most 2 seconds.  With no internet the old code would block
        # for the full requests timeout (8 s) causing the black-screen
        # gap before the login window appeared.
        try:
            from utils.update_manager import check_and_install_update, report_update_completion

            _update_result = [None]
            _update_done  = _startup_threading.Event()

            def _run_update_check():
                try:
                    _update_result[0] = check_and_install_update(parent=None, quiet=True)
                except Exception:
                    _update_result[0] = False
                finally:
                    _update_done.set()

            _startup_threading.Thread(
                target=_run_update_check, daemon=True, name="StartupUpdateCheck"
            ).start()
            # Wait max 2 s — if no internet this returns almost instantly
            # once the thread catches the connection error.
            _update_done.wait(timeout=2.0)

            if _update_result[0]:
                logger.info(
                    f"Update launched for channel={UPDATE_CHANNEL}, repo={GITHUB_REPOSITORY or 'unset'}"
                )
                return

            report_update_completion(APP_VERSION, async_mode=True)
        except Exception as e:
            logger.warning(f"Update check failed: {e}")
        # ──────────────────────────────────────────────────────────────

        # ── Startup License Verification Enforcer ──
        try:
            from utils.license_manager import (
                run_startup_checks,
                load_stored_key,
                clear_stored_key,
                clear_license_cache,
                is_stale_seat_error,
            )

            stored_key = load_stored_key()
            if stored_key:
                # ── Run license check in background so no-internet ────────────
                # doesn't freeze the main thread.  _post_json now times out in
                # 5 s; we wait at most 4 s here so the login window never stalls
                # longer than that.
                _lic_result   = [None]
                _lic_done     = _startup_threading.Event()

                def _run_license_check():
                    try:
                        _lic_result[0] = run_startup_checks(force_heartbeat=False)
                    except Exception as _le:
                        logger.warning(f"License check thread error: {_le}")
                    finally:
                        _lic_done.set()

                _startup_threading.Thread(
                    target=_run_license_check, daemon=True, name="StartupLicenseCheck"
                ).start()
                # Wait max 4 s — covers most offline paths (5s _post_json timeout
                # means the heartbeat will fail fast, the thread will finish in <5s)
                _lic_done.wait(timeout=4.0)
                result = _lic_result[0]
                if result is None:
                    # Still running (rare) — use a safe "offline ok" placeholder
                    # so the login window shows immediately; the thread will keep
                    # running in the background (it's daemon so it dies on exit).
                    from utils.license_manager import StartupCheckResult as _SCR
                    result = _SCR()
                    result.ok = True
                    result.offline_mode = True
                    result.reason = "License check in progress..."
                if not result.ok:
                    logger.warning(f"Startup license checks failed: {result.reason}")
                    is_explicit_revocation = (
                        result.step_failed == 5
                        and getattr(result, "error_code", "") == "LICENSE_REVOKED"
                    )

                    # ── Stale token: let the login screen repair it ───────────
                    # A superseded seat (SEAT_INACTIVE and friends) means only
                    # that cardiox.lic is out of date. No credentials exist yet
                    # at startup, so say nothing and open the login screen —
                    # handle_login() re-syncs the token once the user signs in.
                    if (
                        result.step_failed == 5
                        and not is_explicit_revocation
                        and is_stale_seat_error(
                            getattr(result, "error_code", ""), result.reason
                        )
                    ):
                        logger.info(
                            f"Startup: stale seat reported ({result.error_code}); "
                            "deferring to login for token re-sync."
                        )
                        result = None

                if result is not None and not result.ok:
                    # ── Detect "Seat not found" / seat-missing errors ─────────
                    # This happens when the server DB was reset or the seat was
                    # deleted server-side.  The fix is to clear the stale local
                    # token and let the user re-register. LICENSE_BLOCKED — the
                    # catch-all run_startup_checks assigns to refusals it cannot
                    # name — is deliberately excluded, so a transient server
                    # error can no longer offer to wipe a healthy licence.
                    reason_lower = result.reason.lower()
                    error_code_upper = getattr(result, "error_code", "").upper()
                    _seat_missing_phrases = (
                        "seat not found",
                        "not registered under this license",
                        "please register first",
                        "seat has been flagged",
                        "seat unavailable",
                    )
                    is_seat_missing = (
                        result.step_failed == 5
                        and not is_explicit_revocation
                        and (
                            error_code_upper == "SEAT_NOT_FOUND"
                            or any(p in reason_lower for p in _seat_missing_phrases)
                        )
                    )

                    if is_seat_missing:
                        # Offer the user a choice: Re-register or Exit
                        reregister_box = QMessageBox(None)
                        reregister_box.setWindowTitle("License — Seat Not Found")
                        reregister_box.setIcon(QMessageBox.Warning)
                        reregister_box.setText(
                            "<b>Your device seat was not found on the license server.</b><br><br>"
                            "This usually happens after a server update or account reset.<br>"
                            "You can re-register this device using your existing license key."
                        )
                        reregister_box.setInformativeText(
                            f"License key on file: <b>{stored_key}</b><br><br>"
                            "Click <b>Re-register Device</b> to open the registration screen, "
                            "or <b>Exit</b> to close the application."
                        )
                        reregister_btn = reregister_box.addButton(
                            "Re-register Device", QMessageBox.AcceptRole
                        )
                        reregister_box.addButton("Exit", QMessageBox.RejectRole)
                        reregister_box.exec_()

                        if reregister_box.clickedButton() is reregister_btn:
                            # Clear stale token/key so the app opens on Sign Up tab
                            logger.info(
                                "User chose re-registration after seat-not-found error. "
                                "Clearing stale token and key."
                            )
                            clear_stored_key()
                            clear_license_cache()
                            # Fall through — login dialog will show on signup tab below
                        else:
                            return  # User chose Exit
                    else:
                        # All other failures → show error and block/return
                        QMessageBox.critical(
                            None,
                            {
                                1: "License Missing",
                                2: "License Integrity Failed",
                                3: "Fingerprint Mismatch",
                                4: "RhythmUltra Device Required",
                                5: "License Verification Required",
                            }.get(result.step_failed, "License Blocked"),
                            result.reason,
                        )
                        if result.step_failed in {1, 2} or is_explicit_revocation:
                            clear_stored_key()
                            clear_license_cache()
                        return
            else:
                logger.info("No license key found. Redirecting directly to signup page.")
        except Exception as e:
            logger.error(f"Error during startup license check: {e}")
        # ─────────────────────────────────────────────

        # Initialize login dialog
        login = LoginRegisterDialog()

        # If there is no license key, show the Sign Up tab first to guide them to register
        try:
            from utils.license_manager import load_stored_key
            if not load_stored_key():
                login.stacked.setCurrentIndex(1)  # 1 is the register/SignUp widget!
        except Exception as e:
            logger.error(f"Failed to set default tab: {e}")

        # Main application loop
        while True:
            try:
                if login.exec_() == QDialog.Accepted and login.result:
                    logger.info(f"User {login.username} logged in successfully")


                    # Attach machine serial ID to crash logger for email subject/body   tagging
                    try:
                        users = load_users()
                        record = None
                        if isinstance(users, dict) and login.username in users:
                            record = users.get(login.username)
                        else:
                            # Fallback: search by phone/contact stored under 'phone'    
                            for uname, rec in (users or {}).items():
                                try:
                                    if str(rec.get('phone', '')) == str(login.username):
                                        record = rec
                                        break
                                except Exception:
                                    continue
                        serial_id = ''
                        if isinstance(record, dict):
                            serial_id = str(record.get('serial_id', ''))
                            
                        if serial_id:
                            crash_logger.set_machine_serial_id(serial_id)
                            os.environ['MACHINE_SERIAL_ID'] = serial_id
                            logger.info(f"Machine serial ID set for crash reporting: {serial_id}")
                    except Exception as e:
                        logger.warning(f"Could not set machine serial ID for crash reporting: {e}")
                    

                    # ── Show Medical Compliance Loader (non-blocking) ──────
                    # Show the loader first, then build the dashboard while it
                    # animates so there is NO blank-screen gap between login
                    # and the dashboard appearing.
                    _loader = None
                    try:
                        from utils.medical_loader import show_medical_loader_nonblocking
                        _loader = show_medical_loader_nonblocking()
                    except Exception as e:
                        print(f"Failed to show medical loader: {e}")

                    # ── Dashboard construction with live animation ─────────
                    # Dashboard MUST be built on the main thread (Qt rule), but
                    # we still want the ECG animation to keep painting.
                    # We call processEvents() right before and right after the
                    # constructor so the repaint timer gets at least one shot.
                    Dashboard = get_dashboard_module()
                    if Dashboard is None:
                        if _loader is not None:
                            try:
                                _loader.close()
                            except Exception:
                                pass
                        QMessageBox.critical(None, "Error", "Failed to load Dashboard module. Please check logs.")
                        break

                    from PyQt5.QtWidgets import QApplication as _QApp
                    import time as _time

                    # Let the loader paint at least one clean frame before the heavy build
                    _QApp.processEvents()
                    dashboard = Dashboard(
                        username=login.username,
                        role=None,
                        user_details=login.user_details,
                    )
                    # Let the animation catch up after the blocking constructor
                    _QApp.processEvents()


                    # Attach a session recorder for this user
                    try:
                        user_record = None
                        users = load_users()
                        if isinstance(users, dict) and login.username in users:
                            user_record = users.get(login.username)
                        else:
                            for uname, rec in (users or {}).items():
                                try:
                                    if str(rec.get('phone', '')) == str(login.username):
                                        user_record = rec
                                        break
                                except Exception:
                                    continue
                        dashboard._session_recorder = SessionRecorder(username=login.username, user_record=user_record or {})
                    except Exception as e:
                        logger.warning(f"Session recorder init failed: {e}")

                    # Close the loader smoothly, then show dashboard
                    # finish_and_close() marks all steps done, shows "ready",
                    # waits 300 ms, then calls close() — at that point
                    # dashboard.show() fires via QTimer to avoid a blank frame.
                    _splash = None
                    if _loader is not None:
                        try:
                            _loader.finish_and_close(dashboard)
                            # Pump events so the loader's singleShot 300ms close fires
                            # and the ECG animation plays through the handoff.
                            _deadline = _time.monotonic() + 0.45
                            while _time.monotonic() < _deadline:
                                _QApp.processEvents()
                                _time.sleep(0.010)
                        except Exception as e:
                            print(f"Loader close failed: {e}")
                            try:
                                _loader.close()
                            except Exception:
                                pass

                    dashboard.show()
                    # Ensure window appears normally after UAC or initial launch
                    dashboard.showNormal()
                    dashboard.raise_()
                    dashboard.activateWindow()
                    _QApp.processEvents()



                    # Periodic license re-validation so revocations take effect during runtime.
                    # The check (HTTP heartbeat + WMI fingerprint) runs on a background thread
                    # so it NEVER blocks the main UI thread.
                    try:
                        from PyQt5.QtCore import QObject, QThread, pyqtSignal as _pyqtSignal
                        from utils.license_manager import check_license, load_stored_key

                        class _LicenseCheckWorker(QObject):
                            """Runs blocking license check off the main thread."""
                            result_ready = _pyqtSignal(dict)   # emits check_license() result
                            key_missing  = _pyqtSignal()       # emits when stored_key is absent

                            def run(self):
                                try:
                                    stored_key = load_stored_key()
                                    if not stored_key:
                                        self.key_missing.emit()
                                        return
                                    res = check_license(stored_key, force_server=False)
                                    self.result_ready.emit(res)
                                except Exception as _e:
                                    logger.warning(f"Periodic license check failed: {_e}")

                        # Keep strong references so GC doesn't collect mid-run
                        _lic_threads: list = []

                        def _on_license_result(res):
                            """Called on main thread after background check completes."""
                            pending_title = "License Blocked"
                            pending_reason = res.get("message", "License verification required.")
                            if str(res.get("error_code", "")).strip().upper() == "LICENSE_REVOKED" or res.get("revoked"):
                                _license_timer.stop()
                                pending_title = "License Revoked"
                                pending_reason = res.get("message", "License key is revoked. Contact support.")
                                if _ecg_session_active(dashboard):
                                    _defer_license_block_until_safe(
                                        app,
                                        dashboard,
                                        pending_reason,
                                        pending_title,
                                    )
                                else:
                                    if _recover_license_in_place(
                                        app,
                                        dashboard,
                                        pending_reason,
                                        pending_title,
                                    ):
                                        try:
                                            dashboard.closed_by_sign_out = True
                                        except Exception:
                                            pass
                                        app.quit()
                            elif not res.get("valid", False):
                                _license_timer.stop()
                                if _ecg_session_active(dashboard):
                                    _defer_license_block_until_safe(
                                        app,
                                        dashboard,
                                        pending_reason,
                                        pending_title,
                                    )
                                else:
                                    if _recover_license_in_place(
                                        app,
                                        dashboard,
                                        pending_reason,
                                        pending_title,
                                    ):
                                        try:
                                            dashboard.closed_by_sign_out = True
                                        except Exception:
                                            pass
                                        app.quit()
                            else:
                                if res.get("offline"):
                                    try:
                                        dashboard.set_license_banner("ONLINE", "(Offline Mode)")
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        dashboard.set_license_banner("ONLINE", "")
                                    except Exception:
                                        pass

                        def _on_key_missing():
                            """Called on main thread when license key file is gone."""
                            logger.warning("Watchdog: Stored license key is missing! Blocking access.")
                            _license_timer.stop()
                            _recover_license_in_place(
                                app,
                                dashboard,
                                "No valid license key found on this system.",
                                "License Missing",
                            )

                        def _launch_license_check():
                            """Timer callback: spawn a new worker thread for this check cycle."""
                            thread = QThread()
                            worker = _LicenseCheckWorker()
                            worker.moveToThread(thread)
                            worker.result_ready.connect(_on_license_result)
                            worker.key_missing.connect(_on_key_missing)
                            # Clean up thread after it finishes
                            thread.finished.connect(thread.deleteLater)
                            thread.started.connect(worker.run)
                            thread.finished.connect(lambda: _lic_threads.remove(thread) if thread in _lic_threads else None)
                            _lic_threads.append(thread)
                            thread.start()

                        _license_timer = QTimer(dashboard)
                        _license_timer.setInterval(60 * 1000)
                        _license_timer.timeout.connect(_launch_license_check)
                        _license_timer.start()
                        dashboard._license_timer = _license_timer
                        dashboard._license_threads = _lic_threads
                        # First check after 5 s (avoids contention right at app launch)
                        QTimer.singleShot(5000, _launch_license_check)
                    except Exception as e:
                        logger.warning(f"Could not start license watchdog: {e}")

                    # ── Background update availability check ──────────────────
                    _launch_update_checker(dashboard, APP_VERSION, UPDATE_CHANNEL)

                    # Run application
                    app.exec_()
                    
                    if getattr(dashboard, "closed_by_sign_out", False):
                        logger.info(f"User {login.username} logged out")
                        # After dashboard closes via sign out, show login again
                        login = LoginRegisterDialog()
                    else:
                        logger.info("Application closed by user from dashboard")
                        break
                else:
                    logger.info("Application closed by user")
                    break
                    
            except Exception as e:
                logger.error(f"Error in main application loop: {e}")
                QMessageBox.critical(None, "Application Error", 
                                    f"An error occurred: {e}\nThe application will continue.")
                # Continue with new login dialog
                login = LoginRegisterDialog()
                
    except Exception as e:
        logger.critical(f"Fatal error in main application: {e}")
        crash_logger.log_crash(f"Fatal application error: {str(e)}", e, "MAIN_APPLICATION")
        QMessageBox.critical(None, "Fatal Error", 
                           f"A fatal error occurred: {e}\nThe application will exit.")
        sys.exit(1)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
