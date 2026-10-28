import os
import sys
import time
import json
import traceback
import logging
import smtplib
import threading
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QMessageBox, QProgressBar, QGroupBox, QLineEdit, QFormLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon


def _load_env_if_present():
    """Lightweight .env loader: loads KEY=VALUE pairs into os.environ if .env exists.
    Searches multiple locations to support clones and packaged executables."""
    try:
        locations = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        locations.append(os.path.join(project_root, ".env"))
        # Current working directory
        locations.append(os.path.join(os.getcwd(), ".env"))
        # Frozen app directory (PyInstaller)
        if getattr(sys, 'frozen', False):
            app_dir = os.path.dirname(sys.executable)
            locations.append(os.path.join(app_dir, ".env"))
        # User config dir
        home_env = os.path.join(os.path.expanduser("~"), ".cardiox", ".env")
        locations.append(home_env)

        for env_path in locations:
            if os.path.exists(env_path):
                try:
                    with open(env_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            if '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")
                                if key and value and key not in os.environ:
                                    os.environ[key] = value
                except Exception:
                    continue
    except Exception:
        # Silent fallback; environment variables may already be provided by OS
        pass


class CrashLogger:
    """Comprehensive crash logging and email reporting system"""
    
    def __init__(self, app_name="Comprehensive", log_dir="logs"):
        self.app_name = app_name
        self.log_dir = log_dir
        self.crash_log_file = os.path.join(log_dir, "crash_logs.json")
        self.error_log_file = os.path.join(log_dir, "error_logs.txt")
        self.session_log_file = os.path.join(log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load .env if present, then read email configuration from environment
        _load_env_if_present()
        self.email_config = self._load_email_config_from_env(app_name)
        
        # Validate email configuration
        self.email_configured = self._validate_email_config()
        
        # System info
        self.system_info = self._get_system_info()
        
        # Session tracking
        self.session_start = datetime.now()
        self.error_count = 0
        self.crash_count = 0
        # Optional machine serial id to include in reports
        self.machine_serial_id = os.getenv('MACHINE_SERIAL_ID', '')
        
        # Log session start
        self.log_info("Application started", "SESSION_START")

    def _validate_email_config(self):
        """Validate email configuration and provide helpful error messages"""
        # Recipient is auto-configured to Divyansh; only require sender creds
        required_fields = ['sender_email', 'sender_password']
        missing_fields = []
        
        for field in required_fields:
            if not self.email_config[field]:
                missing_fields.append(field)
        
        if missing_fields:
            self.log_warning(f"Email configuration incomplete. Missing: {', '.join(missing_fields)}", "EMAIL_CONFIG")
            self.log_warning("To enable email reporting, create a .env file with email credentials. See email_config_template.txt for instructions.", "EMAIL_CONFIG")
            return False
        
        # Accept config from .env in multiple locations or from user config json
        configured = any([
            os.getenv('EMAIL_SENDER'),
            self._get_user_email_config() is not None
        ])
        if not configured:
            self.log_warning("No email configuration found. You can either create a .env or use the in-app Configure Email button.", "EMAIL_CONFIG")
            return False
        
        self.log_info("Email configuration validated successfully", "EMAIL_CONFIG")
        return True

    def _get_user_config_dir(self):
        base = os.path.join(os.path.expanduser('~'), '.cardiox')
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            pass
        return base

    def _get_user_email_config(self):
        """Load per-user email config from ~/.cardiox/email_config.json if present."""
        try:
            cfg_path = os.path.join(self._get_user_config_dir(), 'email_config.json')
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def _load_email_config_from_env(self, app_name):
        """Load email config using environment variables first, then fall back to per-user json."""
        cfg = {
            'smtp_server': os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('EMAIL_SMTP_PORT', '587')),
            'sender_email': os.getenv('EMAIL_SENDER', ''),
            'sender_password': os.getenv('EMAIL_PASSWORD', ''),
            # recipient will be set to sender below
            'recipient_email': os.getenv('EMAIL_RECIPIENT', ''),
            'subject_prefix': os.getenv('EMAIL_SUBJECT_PREFIX', f'[{app_name}] Crash Report')
        }
        # If missing, try user json
        if not cfg['sender_email'] or not cfg['sender_password']:
            user_cfg = self._get_user_email_config()
            if isinstance(user_cfg, dict):
                cfg['smtp_server'] = user_cfg.get('smtp_server', cfg['smtp_server'])
                cfg['smtp_port'] = int(user_cfg.get('smtp_port', cfg['smtp_port']))
                cfg['sender_email'] = user_cfg.get('sender_email', cfg['sender_email'])
                cfg['sender_password'] = user_cfg.get('sender_password', cfg['sender_password'])
                cfg['subject_prefix'] = user_cfg.get('subject_prefix', cfg['subject_prefix'])
        # Ensure recipient = sender per requirement
        if not cfg['recipient_email']:
            cfg['recipient_email'] = cfg['sender_email']
        return cfg
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for session logs
        file_handler = logging.FileHandler(self.session_log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Setup root logger
        self.logger = logging.getLogger('ECGCrashLogger')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def _get_system_info(self):
        """Get comprehensive system information"""
        try:
            import platform
            import psutil
            import sys
            
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'python_executable': sys.executable,
                'cpu_count': psutil.cpu_count(),
                'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
                'memory_available': f"{psutil.virtual_memory().available / (1024**3):.1f} GB",
                'disk_usage': f"{psutil.disk_usage('/').percent:.1f}%",
                'current_user': os.getenv('USER', 'Unknown'),
                'working_directory': os.getcwd()
            }
        except Exception as e:
            return {'error': f"Failed to get system info: {str(e)}"}
    
    def log_info(self, message, category="INFO"):
        """Log informational message"""
        self.logger.info(f"[{category}] {message}")
    
    def log_warning(self, message, category="WARNING"):
        """Log warning message"""
        self.logger.warning(f"[{category}] {message}")
    
    def log_error(self, message, exception=None, category="ERROR"):
        """Log error message with optional exception details"""
        self.error_count += 1
        
        error_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'category': category,
            'error_count': self.error_count,
            'session_duration': str(datetime.now() - self.session_start)
        }
        
        if exception:
            error_data.update({
                'exception_type': type(exception).__name__,
                'exception_message': str(exception),
                'traceback': traceback.format_exc()
            })
        
        # Log to file
        self.logger.error(f"[{category}] {message}")
        if exception:
            self.logger.error(f"Exception: {type(exception).__name__}: {str(exception)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        # Save to crash log
        self._save_crash_log(error_data)
    
    def log_crash(self, message, exception=None, context=""):
        """Log critical crash with full context"""
        self.crash_count += 1
        
        crash_data = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'exception_type': type(exception).__name__ if exception else None,
            'exception_message': str(exception) if exception else None,
            'traceback': traceback.format_exc() if exception else None,
            'context': context,
            'machine_serial_id': self.machine_serial_id or None,
            'crash_count': self.crash_count,
            'session_duration': str(datetime.now() - self.session_start),
            'system_info': self.system_info,
            'memory_usage': self._get_memory_usage(),
            'recent_logs': self._get_recent_logs(50)  # Last 50 log entries
        }
        
        # Log critical error
        self.logger.critical(f"[CRASH] {message}")
        if exception:
            self.logger.critical(f"Exception: {type(exception).__name__}: {str(exception)}")
            self.logger.critical(f"Traceback:\n{traceback.format_exc()}")
        
        # Save crash log
        self._save_crash_log(crash_data, is_crash=True)
        
        # Auto-send email for critical crashes
        threading.Thread(target=self._send_crash_email, args=(crash_data,), daemon=True).start()

    def set_machine_serial_id(self, serial_id: str):
        """Set the machine serial id to include in subsequent crash reports and emails."""
        try:
            self.machine_serial_id = str(serial_id or '').strip()
            if self.machine_serial_id:
                self.log_info(f"Machine serial set: {self.machine_serial_id}", "SERIAL")
        except Exception:
            pass
    
    def _save_crash_log(self, log_data, is_crash=False):
        """Save log data to JSON file"""
        try:
            logs = []
            if os.path.exists(self.crash_log_file):
                with open(self.crash_log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            
            logs.append(log_data)
            
            # Keep only last 100 entries to prevent file from growing too large
            if len(logs) > 100:
                logs = logs[-100:]
            
            with open(self.crash_log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save crash log: {e}")
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        try:
            import psutil
            process = psutil.Process()
            return {
                'rss': f"{process.memory_info().rss / (1024**2):.1f} MB",
                'vms': f"{process.memory_info().vms / (1024**2):.1f} MB",
                'cpu_percent': f"{process.cpu_percent():.1f}%"
            }
        except Exception:
            return {'error': 'Unable to get memory usage'}
    
    def _get_recent_logs(self, count=50):
        """Get recent log entries"""
        try:
            if os.path.exists(self.session_log_file):
                with open(self.session_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    return lines[-count:] if len(lines) > count else lines
            return []
        except Exception:
            return []
    
    def _send_crash_email(self, crash_data):
        """Send crash report via email"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']
            serial_tag = self.machine_serial_id or 'N/A'
            msg['Subject'] = f"{self.email_config['subject_prefix']} - {crash_data['timestamp']} [Serial: {serial_tag}]"
            
            # Create email body
            body = f"""
CardioX Application Crash Report
====================================

Timestamp: {crash_data['timestamp']}
Session Duration: {crash_data['session_duration']}
Crash Count: {crash_data['crash_count']}
Machine Serial: {self.machine_serial_id or 'N/A'}

Error Details:
--------------
Message: {crash_data['message']}
Context: {crash_data.get('context', 'N/A')}

Exception Information:
---------------------
Type: {crash_data.get('exception_type', 'N/A')}
Message: {crash_data.get('exception_message', 'N/A')}

Traceback:
----------
{crash_data.get('traceback', 'N/A')}

System Information:
------------------
Platform: {crash_data['system_info'].get('platform', 'N/A')}
Python Version: {crash_data['system_info'].get('python_version', 'N/A')}
Memory Usage: {crash_data.get('memory_usage', 'N/A')}

Recent Log Entries:
------------------
{''.join(crash_data.get('recent_logs', []))}

Log Files:
----------
Crash Log: {self.crash_log_file}
Session Log: {self.session_log_file}

This is an automated crash report from the CardioX application.
Please investigate and fix the issue.

Best regards,
CardioX Crash Logger
            """
            
            msg.attach(MIMEText(body, 'plain'))

            # Send email with TLS first, then fallback to SSL
            text = msg.as_string()
            try:
                server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.sendmail(self.email_config['sender_email'], self.email_config['recipient_email'], text)
                server.quit()
            except smtplib.SMTPAuthenticationError as e:
                # Authentication failed via TLS, try SSL (port 465)
                server = smtplib.SMTP_SSL(self.email_config['smtp_server'], 465)
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.sendmail(self.email_config['sender_email'], self.email_config['recipient_email'], text)
                server.quit()
            
            self.log_info("Crash report email sent successfully", "EMAIL_SENT")
            
        except Exception as e:
            self.log_error(f"Failed to send crash email: {str(e)}", e, "EMAIL_ERROR")
    
    def get_all_logs(self):
        """Get all crash logs"""
        try:
            if os.path.exists(self.crash_log_file):
                with open(self.crash_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.log_error(f"Failed to read crash logs: {str(e)}", e)
            return []
    
    def clear_logs(self):
        """Clear all crash logs"""
        try:
            if os.path.exists(self.crash_log_file):
                os.remove(self.crash_log_file)
            self.log_info("All crash logs cleared", "LOGS_CLEARED")
            return True
        except Exception as e:
            self.log_error(f"Failed to clear logs: {str(e)}", e)
            return False


class EmailSenderThread(QThread):
    """Thread for sending emails without blocking UI"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, crash_logger, logs_to_send):
        super().__init__()
        self.crash_logger = crash_logger
        self.logs_to_send = logs_to_send
    
    def run(self):
        try:
            # Check email configuration first
            if not self.crash_logger.email_configured:
                self.finished.emit(False, "Email not configured. Please set up .env file with email credentials.")
                return
            
            self.progress.emit(10)
            
            # Create comprehensive email
            msg = MIMEMultipart()
            msg['From'] = self.crash_logger.email_config['sender_email']
            msg['To'] = self.crash_logger.email_config['recipient_email']
            msg['Subject'] = f"{self.crash_logger.email_config['subject_prefix']} - Manual Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Create detailed email body
            body = f"""
CardioX Application - Manual Log Report
==========================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Log Entries: {len(self.logs_to_send)}

System Information:
------------------
Platform: {self.crash_logger.system_info.get('platform', 'N/A')}
Python Version: {self.crash_logger.system_info.get('python_version', 'N/A')}
Memory Total: {self.crash_logger.system_info.get('memory_total', 'N/A')}
Memory Available: {self.crash_logger.system_info.get('memory_available', 'N/A')}

Session Statistics:
------------------
Session Duration: {str(datetime.now() - self.crash_logger.session_start)}
Total Errors: {self.crash_logger.error_count}
Total Crashes: {self.crash_logger.crash_count}

Log Entries:
============
"""
            
            self.progress.emit(30)
            
            # Add each log entry
            if not self.logs_to_send:
                body += "\nNo crash logs available. Sending session diagnostics only.\n\n"
            for i, log_entry in enumerate(self.logs_to_send):
                body += f"""
Entry {i+1}:
-----------
Timestamp: {log_entry.get('timestamp', 'N/A')}
Category: {log_entry.get('category', 'N/A')}
Message: {log_entry.get('message', 'N/A')}
Context: {log_entry.get('context', 'N/A')}

Exception Type: {log_entry.get('exception_type', 'N/A')}
Exception Message: {log_entry.get('exception_message', 'N/A')}

Traceback:
{log_entry.get('traceback', 'N/A')}

{'='*50}
"""
                self.progress.emit(30 + int((i / len(self.logs_to_send)) * 50))
            
            body += f"""

This report was manually generated by triple-clicking the heart rate metrics.
Please review the logs and address any issues found.

Best regards,
CardioX Application
            """
            
            self.progress.emit(80)
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.crash_logger.email_config['smtp_server'], self.crash_logger.email_config['smtp_port'])
            server.starttls()
            server.login(self.crash_logger.email_config['sender_email'], self.crash_logger.email_config['sender_password'])
            text = msg.as_string()
            server.sendmail(self.crash_logger.email_config['sender_email'], self.crash_logger.email_config['recipient_email'], text)
            server.quit()
            
            self.progress.emit(100)
            self.finished.emit(True, "Email sent successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Failed to send email: {str(e)}")


class CrashLogDialog(QDialog):
    """Dialog for viewing and managing crash logs"""
    
    def __init__(self, crash_logger, parent=None):
        super().__init__(parent)
        self.crash_logger = crash_logger
        self.email_thread = None
        self.init_ui()
        self.load_logs()
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("CardioX - Crash Logs & Diagnostics")
        self.setModal(True)
        self.resize(700, 500)  # Smaller but decent size
        
        # Main layout
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("🔧 CardioX - Diagnostic & Crash Log Center")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #ff6600; margin: 10px;")  # Match software's orange theme
        layout.addWidget(header)
        
        # Stats group
        stats_group = QGroupBox("Session Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel()
        self.update_stats()
        stats_layout.addWidget(self.stats_label)
        
        # Email status
        self.email_status_label = QLabel()
        self.update_email_status()
        stats_layout.addWidget(self.email_status_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Logs display
        logs_group = QGroupBox("Crash Logs & Errors")
        logs_layout = QVBoxLayout()
        
        self.logs_text = QTextEdit()
        self.logs_text.setReadOnly(True)
        self.logs_text.setFont(QFont("Courier", 9))
        logs_layout.addWidget(self.logs_text)
        
        logs_group.setLayout(logs_layout)
        layout.addWidget(logs_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Refresh Logs")
        self.refresh_btn.clicked.connect(self.load_logs)
        button_layout.addWidget(self.refresh_btn)
        
        self.send_email_btn = QPushButton("📧 Send Report via Email")
        self.send_email_btn.clicked.connect(self.send_email_report)
        button_layout.addWidget(self.send_email_btn)
        
        self.clear_logs_btn = QPushButton("🗑️ Clear All Logs")
        self.clear_logs_btn.clicked.connect(self.clear_logs)
        button_layout.addWidget(self.clear_logs_btn)
        
        self.close_btn = QPushButton("❌ Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
        
        # Style the dialog to match the software's theme
        self.setStyleSheet("""
            QDialog {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f8f9fa, stop:1 #e9ecef);
                font-family: Arial, sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ff6600;
                border-radius: 16px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ff6600;
            }
            QPushButton {
                background: #ff6600;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #ff8800;
            }
            QPushButton:pressed {
                background: #e55a00;
            }
            QTextEdit {
                background-color: #f7f7f7;
                border: 2px solid #ff6600;
                border-radius: 8px;
                padding: 8px;
            }
            QProgressBar {
                border: 2px solid #ff6600;
                border-radius: 8px;
                text-align: center;
                background-color: #f7f7f7;
            }
            QProgressBar::chunk {
                background: #ff6600;
                border-radius: 6px;
            }
        """)
        
        # Add quick configure button when not configured
        if not self.crash_logger.email_configured:
            self.configure_btn = QPushButton("⚙️ Configure Email…")
            self.configure_btn.clicked.connect(self.open_email_config_dialog)
            layout.addWidget(self.configure_btn)
    
    def update_stats(self):
        """Update session statistics"""
        session_duration = datetime.now() - self.crash_logger.session_start
        stats_text = f"""
        📊 Session Duration: {str(session_duration).split('.')[0]}
        ⚠️ Total Errors: {self.crash_logger.error_count}
        💥 Total Crashes: {self.crash_logger.crash_count}
        💾 Memory Usage: {self.crash_logger._get_memory_usage().get('rss', 'N/A')}
        """
        self.stats_label.setText(stats_text)
    
    def update_email_status(self):
        """Update email configuration status"""
        if self.crash_logger.email_configured:
            status_text = "📧 Email Reporting: ✅ Configured"
            status_color = "color: #27ae60;"  # Green
        else:
            status_text = "📧 Email Reporting: ❌ Not Configured\n(See email_config_template.txt for setup instructions)"
            status_color = "color: #e74c3c;"  # Red
        
        self.email_status_label.setText(status_text)
        self.email_status_label.setStyleSheet(status_color)
    
    def load_logs(self):
        """Load and display crash logs"""
        try:
            logs = self.crash_logger.get_all_logs()
            
            if not logs:
                self.logs_text.setText("No crash logs found. The application is running smoothly! 🎉")
                return
            
            # Format logs for display
            log_text = "CardioX - Crash Logs & Error Reports\n"
            log_text += "=" * 60 + "\n\n"
            
            for i, log in enumerate(reversed(logs)):  # Show newest first
                log_text += f"Entry #{len(logs) - i}\n"
                log_text += f"Timestamp: {log.get('timestamp', 'N/A')}\n"
                log_text += f"Category: {log.get('category', 'N/A')}\n"
                log_text += f"Message: {log.get('message', 'N/A')}\n"
                
                if log.get('context'):
                    log_text += f"Context: {log.get('context')}\n"
                
                if log.get('exception_type'):
                    log_text += f"Exception: {log.get('exception_type')} - {log.get('exception_message', 'N/A')}\n"
                
                if log.get('traceback'):
                    log_text += f"Traceback:\n{log.get('traceback')}\n"
                
                log_text += "-" * 40 + "\n\n"
            
            self.logs_text.setText(log_text)
            self.update_stats()
            
        except Exception as e:
            self.logs_text.setText(f"Error loading logs: {str(e)}")
    
    def send_email_report(self):
        """Send email report with all logs"""
        # Check if email is configured
        if not self.crash_logger.email_configured:
            QMessageBox.warning(
                self,
                "Email Not Configured",
                "Email reporting is not configured on this computer.\n\n"
                "To enable email reporting:\n"
                "1. Copy 'email_config_template.txt' to '.env'\n"
                "2. Edit .env with your Gmail credentials\n"
                "3. Generate a Gmail App Password\n"
                "4. Restart the application\n\n"
                "See email_config_template.txt for detailed instructions."
            )
            return
        
        logs = self.crash_logger.get_all_logs()
        
        # Allow sending diagnostics even if there are no logs
        if not logs:
            reply = QMessageBox.question(
                self,
                "Send Diagnostics",
                "No crash logs found. Send a diagnostics report with session and system details?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        # Confirm sending
        reply = QMessageBox.question(
            self, 
            "Send Email Report", 
            f"Send {len(logs)} log entries (or diagnostics) to the development team?\n\nThis will include system information and crash details.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.send_email_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Start email thread
            self.email_thread = EmailSenderThread(self.crash_logger, logs)
            self.email_thread.progress.connect(self.progress_bar.setValue)
            self.email_thread.finished.connect(self.email_sent)
            self.email_thread.start()
    
    def email_sent(self, success, message):
        """Handle email sending completion"""
        self.progress_bar.setVisible(False)
        self.send_email_btn.setEnabled(True)
        
        if success:
            QMessageBox.information(self, "Email Sent", message)
        else:
            QMessageBox.critical(self, "Email Failed", message)
    
    def clear_logs(self):
        """Clear all crash logs"""
        reply = QMessageBox.question(
            self, 
            "Clear Logs", 
            "Are you sure you want to clear all crash logs?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.crash_logger.clear_logs():
                QMessageBox.information(self, "Logs Cleared", "All crash logs have been cleared.")
                self.load_logs()
            else:
                QMessageBox.critical(self, "Error", "Failed to clear logs.")

    def open_email_config_dialog(self):
        """Simple inline dialog to capture email settings and save to user config."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Configure Email Reporting")
        form = QFormLayout()
        sender = QLineEdit(self.crash_logger.email_config.get('sender_email', ''))
        password = QLineEdit(self.crash_logger.email_config.get('sender_password', ''))
        password.setEchoMode(QLineEdit.Password)
        recipient = QLineEdit(self.crash_logger.email_config.get('recipient_email', 'divyansh.srivastava@deckmount.in'))
        recipient.setReadOnly(True)
        smtp = QLineEdit(self.crash_logger.email_config.get('smtp_server', 'smtp.gmail.com'))
        port = QLineEdit(str(self.crash_logger.email_config.get('smtp_port', 587)))
        form.addRow("Sender Email", sender)
        form.addRow("App Password", password)
        form.addRow("Recipient Email", recipient)
        form.addRow("SMTP Server", smtp)
        form.addRow("SMTP Port", port)
        btns = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btns.addWidget(save_btn)
        btns.addWidget(cancel_btn)
        wrapper = QVBoxLayout()
        wrapper.addLayout(form)
        wrapper.addLayout(btns)
        dlg.setLayout(wrapper)

        def do_save():
            try:
                cfg_dir = self.crash_logger._get_user_config_dir()
                path = os.path.join(cfg_dir, 'email_config.json')
                data = {
                    'smtp_server': smtp.text().strip() or 'smtp.gmail.com',
                    'smtp_port': int(port.text().strip() or '587'),
                    'sender_email': sender.text().strip(),
                    'sender_password': password.text().strip(),
                    'recipient_email': 'divyansh.srivastava@deckmount.in',
                    'subject_prefix': self.crash_logger.email_config.get('subject_prefix', '[CardioX] Crash Report')
                }
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                # Reload config
                self.crash_logger.email_config = self.crash_logger._load_email_config_from_env(self.crash_logger.app_name if hasattr(self.crash_logger, 'app_name') else 'CardioX')
                self.crash_logger.email_configured = self.crash_logger._validate_email_config()
                self.update_email_status()
                QMessageBox.information(self, "Saved", "Email configuration saved for this user.")
                dlg.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save: {e}")

        save_btn.clicked.connect(do_save)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec_()


# Global crash logger instance
crash_logger = None

def get_crash_logger():
    """Get the global crash logger instance"""
    global crash_logger
    if crash_logger is None:
        crash_logger = CrashLogger()
    return crash_logger

def log_crash(message, exception=None, context=""):
    """Convenience function to log crashes"""
    logger = get_crash_logger()
    logger.log_crash(message, exception, context)

def log_error(message, exception=None, category="ERROR"):
    """Convenience function to log errors"""
    logger = get_crash_logger()
    logger.log_error(message, exception, category)

def log_info(message, category="INFO"):
    """Convenience function to log info"""
    logger = get_crash_logger()
    logger.log_info(message, category)
