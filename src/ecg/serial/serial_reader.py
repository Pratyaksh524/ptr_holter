"""Serial communication classes for ECG hardware"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

# Configuration: Skip VERSION command if device doesn't support it or times out
SKIP_VERSION_CHECK = False  # Enable VERSION command to populate Version panel

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
        def read(self, size): return b''
        def write(self, data): pass
        def reset_input_buffer(self): pass
        def reset_output_buffer(self): pass
        @property
        def is_open(self): return False
    class SerialException(Exception): pass
    serial = type('Serial', (), {'Serial': Serial, 'SerialException': SerialException})()
    class MockComports:
        @staticmethod
        def comports(*args, **kwargs):
            return []
    serial.tools = type('Tools', (), {'list_ports': MockComports()})()

from utils.crash_logger import get_crash_logger
from .packet_parser import parse_packet, PACKET_SIZE, START_BYTE, END_BYTE

# Import hardware command handler (only if serial is available)
if SERIAL_AVAILABLE:
    from .hardware_commands import HardwareCommandHandler
else:
    # Dummy class for when serial is not available
    class HardwareCommandHandler:
        def __init__(self, *args, **kwargs): pass
        def send_start_command(self): return (False, None)
        def send_stop_command(self): return (False, None)
        def send_version_command(self): return (False, None, None)
        def send_machine_serial_command(self): return (False, None, None)
        def send_close_command(self): return (False, None)

class GlobalHardwareManager:
    """Singleton to manage a single shared serial connection across the app"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalHardwareManager, cls).__new__(cls)
            cls._instance.reader = None
        return cls._instance

    def get_reader(self, port=None, baudrate=None):
        """Get or create the shared serial reader"""
        if self.reader is None:
            if port and baudrate:
                self.reader = SerialStreamReader(port, baudrate)
            else:
                return None
        
        # If port changed, close old and open new
        elif port and self.reader.ser.port != port:
            self.reader.close()
            self.reader = SerialStreamReader(port, baudrate)
            
        return self.reader

    def close_reader(self):
        """Explicitly close and destroy the reader"""
        if self.reader:
            self.reader.close()
            self.reader = None
            

class SerialStreamReader:
    """Packet-based serial reader for ECG data - NEW IMPLEMENTATION"""
    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(float(os.getenv(name, str(default))))
        except Exception:
            return int(default)

    # Tunable via environment for cross-laptop stability:
    # ECG_READ_BUDGET_SECONDS: parse budget per UI tick
    # ECG_MAX_BUFFER_BYTES: max parser buffer before trimming
    # ECG_SILENCE_WARNING_SECONDS: warning threshold when no valid packets are seen
    READ_LOOP_BUDGET_SECONDS = _env_float.__func__("ECG_READ_BUDGET_SECONDS", 0.02)
    MAX_BUFFER_BYTES = _env_int.__func__("ECG_MAX_BUFFER_BYTES", 150000)
    SILENCE_WARNING_SECONDS = _env_float.__func__("ECG_SILENCE_WARNING_SECONDS", 3.0)
    
    @staticmethod
    def scan_and_detect_port(baudrate: int = 115200, timeout: float = 0.05) -> Optional[tuple]:
        """
        Scan all available COM ports and detect the ECG device.

        Detection priority:
        1) VERSION command success (preferred; confirms protocol support)
        2) START ACK success (fallback for older firmware)
        
        Args:
            baudrate: Baud rate to use for scanning (default: 115200)
            timeout: Timeout per port in seconds (default: 0.05 = 50ms for instant detection)
            
        Returns:
            tuple: (port_name: str, serial_port: Serial) if found, or None if none found
                   The serial port is already opened and validated
        """
        if not SERIAL_AVAILABLE:
            print("[ERROR] Port scanning failed: Serial module not available")
            return None
        
        print("\n" + "="*70)
        print("[SCAN] PORT SCANNING: Scanning all COM ports for ECG device...")
        print("="*70)
        
        # Get all available COM ports
        try:
            ports = serial.tools.list_ports.comports()
            port_list = [port.device for port in ports]
            print(f"[PARSED] Found {len(port_list)} COM port(s): {', '.join(port_list)}")
        except Exception as e:
            print(f"[ERROR] Error listing COM ports: {e}")
            return None
        
        if not port_list:
            print("[WARN] No COM ports found")
            print("="*70 + "\n")
            return None

        # ── macOS optimisation: only probe USB-serial paths ──────────────────
        # On macOS, comports() returns Bluetooth modems, internal modems, and
        # the USB CDC device. Opening Bluetooth modem paths takes 500–800 ms
        # each on macOS because the system driver negotiates a session.
        # Filtering to /dev/cu.usb* avoids all non-ECG paths immediately.
        if sys.platform == "darwin":
            usb_ports = [
                p for p in port_list
                if any(k in p.lower() for k in ("usbserial", "usbmodem", "tty.usb", "cu.usb"))
            ]
            if usb_ports:
                print(f"🍎 macOS: filtered to {len(usb_ports)} USB port(s): {usb_ports}")
                port_list = usb_ports
            else:
                print("🍎 macOS: no USB ports found – scanning all ports")
        # ─────────────────────────────────────────────────────────────────────

        def _probe_port(port_name: str):
            """Try opening port_name and validating via VERSION (or START fallback)."""
            temp_ser = None
            try:
                print(f"   🔌 Testing port: {port_name}")
                temp_ser = serial.Serial(
                    port=port_name,
                    baudrate=baudrate,
                    timeout=timeout,
                    write_timeout=timeout,
                )
                temp_ser.reset_input_buffer()
                temp_ser.reset_output_buffer()
                temp_handler = HardwareCommandHandler(temp_ser)

                # Preferred detection: VERSION command
                version_ok, version, _ = temp_handler.send_version_command(timeout=timeout)
                if version_ok:
                    version_text = version or "(empty version)"
                    print(f"   [OK] Port {port_name} responded to VERSION: {version_text}")
                    return (port_name, temp_ser)

                # Fallback detection: START ACK (for legacy firmware)
                success, _ = temp_handler.send_start_command(timeout=timeout, quiet=True)
                if success:
                    print(f"   [OK] Port {port_name} responded to START ACK (legacy fallback)")
                    return (port_name, temp_ser)

                print(f"   [ERROR] Port {port_name} did not respond to VERSION/START")
                temp_ser.close()
                return None
            except Exception as exc:
                print(f"   [WARN] Port {port_name} error: {exc}")
                if temp_ser:
                    try:
                        temp_ser.close()
                    except Exception:
                        pass
                return None

        # ── Parallel probing ─────────────────────────────────────────────────
        # Try all candidate ports simultaneously. Whichever replies first wins;
        # the rest are cancelled. On Mac with a single USB device this cuts
        # scan time from (N × timeout) down to ~timeout for one port.
        detected_port = None
        detected_serial = None
        winner_futures = []
        with ThreadPoolExecutor(max_workers=max(1, len(port_list))) as pool:
            futures = {pool.submit(_probe_port, p): p for p in port_list}
            for fut in as_completed(futures):
                result = fut.result()
                if result and detected_port is None:
                    detected_port, detected_serial = result
                    # Cancel remaining (best-effort; already-running threads finish naturally)
                    for f in futures:
                        if f is not fut:
                            f.cancel()
                elif result and result[1] is not None:
                    # Another thread also succeeded – close the extra port
                    try:
                        result[1].close()
                    except Exception:
                        pass
        # ─────────────────────────────────────────────────────────────────────

        # Fallback pass for slower/quirky USB stacks:
        # if ultra-fast probe missed the device, try once more with relaxed timeout.
        if detected_port is None:
            slow_timeout = max(0.35, float(timeout))
            print(f"   🔁 No fast match; retrying sequential probe with timeout={slow_timeout:.2f}s")
            for p in port_list:
                result = _probe_port(p)
                if result:
                    detected_port, detected_serial = result
                    break

        if detected_port and detected_serial:
            print(f"\n[OK] PORT DETECTED: {detected_port}")
        else:
            print(f"\n[ERROR] No port responded to VERSION/START commands")

        print("="*70 + "\n")

        if detected_port and detected_serial:
            return (detected_port, detected_serial)
        return None
    
    def __init__(self, port: str, baudrate: int, timeout: float = 0.1):
        if not SERIAL_AVAILABLE:
            raise RuntimeError("pyserial is required for serial capture. pip install pyserial")
        safe_timeout = max(0.01, min(float(timeout or 0.1), 0.1))
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=safe_timeout, write_timeout=safe_timeout)
        try:
            # Windows driver hint: larger RX buffer reduces burst loss on slower laptops.
            if hasattr(self.ser, "set_buffer_size"):
                self.ser.set_buffer_size(rx_size=262144, tx_size=16384)
        except Exception:
            pass
        self.buf = bytearray()
        self.running = False
        self.data_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.crash_logger = get_crash_logger()
        self.user_details = {}  # For error reporting compatibility
        # Packet loss tracking
        self.start_time = time.time()
        self.last_packet_time = time.time()
        self.total_packets_expected = 0
        self.total_packets_lost = 0
        self.packet_loss_percent = 0.0
        # Sequence-based packet loss detection
        self._last_packet_counter = None
        self._total_sequence_lost = 0
        self._packet_loss_warnings = 0
        self._last_silence_warn_time = 0.0
        self._read_timeout = safe_timeout
        # Initialize hardware command handler
        if SERIAL_AVAILABLE:
            self.command_handler = HardwareCommandHandler(self.ser)
        else:
            self.command_handler = None
        self.device_version = None
        self.device_serial_number = None
        print(f" SerialStreamReader initialized: Port={port}, Baud={baudrate}")

    def is_device_silent(self, silence_seconds: float = None) -> bool:
        """Return True when stream has no valid packet for configured duration."""
        if silence_seconds is None:
            silence_seconds = self.SILENCE_WARNING_SECONDS
        if not self.running:
            return False
        return (time.time() - float(self.last_packet_time or 0)) >= float(silence_seconds)

    def close(self) -> None:
        """Close serial connection"""
        try:
            # Stop data acquisition first
            self.running = False
            
            # Send CLOSE command to hardware if command handler is available
            if self.command_handler:
                try:
                    success, response = self.command_handler.send_close_command()
                    if not success:
                        print(" [WARN] Warning: CLOSE command ACK not received")
                except Exception as e:
                    print(f" [WARN] Error sending CLOSE command: {e}")
            
            # Flush any remaining data
            if hasattr(self.ser, 'reset_input_buffer'):
                try:
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
            
            if hasattr(self.ser, 'reset_output_buffer'):
                try:
                    self.ser.reset_output_buffer()
                except Exception:
                    pass
            
            # Close the serial port
            if self.ser and self.ser.is_open:
                self.ser.close()
                print(" Serial port closed and released")
            
            # Clear buffer
            self.buf.clear()
        except Exception as e:
            print(f" Error closing serial connection: {e}")
    
    def get_device_version(self) -> Optional[str]:
        """
        Get device version from hardware using VERSION command.
        
        Returns:
            str or None: Version string if available, otherwise None.
        """
        # Check if VERSION command is disabled
        if SKIP_VERSION_CHECK:
            print("⏭️  VERSION COMMAND: Skipped (disabled via SKIP_VERSION_CHECK flag)")
            print("   Set SKIP_VERSION_CHECK = False in serial_reader.py to enable")
            return None
        
        # Only available when serial + HardwareCommandHandler are available
        if not hasattr(self, "command_handler") or not self.command_handler:
            print("[ERROR] VERSION COMMAND: Command handler not available")
            return None

        try:
            print("\n" + "=" * 60)
            print("[SCAN] VERSION COMMAND: Requesting device version...")
            print("=" * 60)

            success, version, response = self.command_handler.send_version_command()
            if success and version:
                self.device_version = version
                print(f"[OK] VERSION COMMAND: Success! Device version: '{version}'")
                print("=" * 60 + "\n")
                return version
            else:
                print("[WARN] VERSION COMMAND: Failed or no version received")
                print(f"   Success: {success}, Version: {version}, Response: {response}")
                print("=" * 60 + "\n")
                return None
        except Exception as e:
            print(f"[ERROR] VERSION COMMAND: Error getting device version: {e}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print("=" * 60 + "\n")
            return None

    def get_device_serial_number(self) -> Optional[str]:
        """
        Get machine serial number from hardware using MACHINE SERIAL (0x13) command.

        Returns:
            str or None: Serial number string if available, otherwise None.
        """
        if not hasattr(self, "command_handler") or not self.command_handler:
            print("[ERROR] MACHINE SERIAL COMMAND: Command handler not available")
            return None

        try:
            print("\n" + "=" * 60)
            print("[SCAN] MACHINE SERIAL COMMAND: Requesting machine serial number...")
            print("=" * 60)

            success, serial_number, response = self.command_handler.send_machine_serial_command()
            if success and serial_number:
                self.device_serial_number = serial_number
                print(f"[OK] MACHINE SERIAL COMMAND: Success! Serial number: '{serial_number}'")
                print("=" * 60 + "\n")
                return serial_number

            print("[WARN] MACHINE SERIAL COMMAND: Failed or no serial number received")
            print(f"   Success: {success}, Serial: {serial_number}, Response: {response}")
            print("=" * 60 + "\n")
            return None
        except Exception as e:
            print(f"[ERROR] MACHINE SERIAL COMMAND: Error getting serial number: {e}")
            import traceback

            print(f"   Traceback: {traceback.format_exc()}")
            print("=" * 60 + "\n")
            return None

    def start(self, skip_hardware_start=False):
        """Start data acquisition using packet‑based protocol + hardware START command."""
        print(" Starting packet-based ECG data acquisition...")

        if self.running and not self.ser.is_open:
            # If we think we are running but port is closed, reset state
            self.running = False

        # Get current port info
        current_port = self.ser.port if hasattr(self.ser, "port") else None
        current_baudrate = self.ser.baudrate if hasattr(self.ser, "baudrate") else 115200
        current_port_open = self.ser.is_open if hasattr(self.ser, "is_open") else False

        # Ensure port is open and fresh
        if not current_port_open or not self.running:
            try:
                if current_port_open:
                    print(f"   [RETRY] Closing existing port {current_port} to ensure fresh handle...")
                    self.ser.close()
                
                if current_port:
                    print(f"   [RETRY] Opening port {current_port} at {current_baudrate} baud...")
                    # Re-initialize serial port object to ensure fresh state
                    self.ser.port = current_port
                    self.ser.baudrate = current_baudrate
                    self.ser.timeout = self._read_timeout
                    if hasattr(self.ser, "write_timeout"):
                        self.ser.write_timeout = self._read_timeout
                    self.ser.open()
                    
                    # Re-initialize command handler with the opened port
                    if SERIAL_AVAILABLE:
                        self.command_handler = HardwareCommandHandler(self.ser)
                    print(f"   [OK] Port {current_port} opened")
            except Exception as e:
                print(f"   [ERROR] Failed to open/reopen port {current_port}: {e}")
                # If we failed to reopen, ensure we don't think we're running
                self.running = False
                raise RuntimeError(f"Cannot open port {current_port}: {e}")

        # Verify port is open and ready
        if not self.ser.is_open:
            raise RuntimeError("Serial port is not open. Cannot start data acquisition.")

        # Clear buffers before sending START
        try:
            if hasattr(self.ser, "reset_input_buffer"):
                self.ser.reset_input_buffer()
            self.buf.clear()
        except Exception as e:
            print(f" [WARN] Warning: Could not clear buffers before START: {e}")

        # If already running or requested to skip, don't send hardware commands
        if skip_hardware_start or self.running:
            print("   ⏩ Skipping hardware commands (already running or skip requested)")
            self.running = True
            return

        # First: optionally request and print device version while device is IDLE
        # (send STOP + VERSION, then we will START streaming below)
        # SKIP VERSION if flag is set to prevent UI freezing
        if not SKIP_VERSION_CHECK:
            try:
                version = self.get_device_version()
                if version:
                    print(f" 🧬 ECG Device Version: {version}")
                    # After VERSION is confirmed (device idle), request machine serial number
                    # before starting streaming so handshake frames aren't stolen by the stream reader.
                    try:
                        serial_number = self.get_device_serial_number()
                        if serial_number:
                            print(f" 🏷️ ECG Machine Serial Number: {serial_number}")
                    except Exception as sn_err:
                        print(f" [WARN] MACHINE SERIAL COMMAND skipped due to error: {sn_err}")
            except Exception as e:
                print(f" [WARN] VERSION COMMAND skipped due to error: {e}")
        else:
            print(" ⏭️  VERSION COMMAND: Skipped (SKIP_VERSION_CHECK = True)")

        # Now send hardware START command to begin ECG streaming
        if hasattr(self, "command_handler") and self.command_handler:
            try:
                success, response = self.command_handler.send_start_command()
                if not success:
                    print(" [WARN] Warning: START command ACK not received on current port")
                    print("   Continuing anyway - device may still send data...")
            except Exception as e:
                print(f" [WARN] Warning: Error sending START command: {e}")
                print("   Continuing anyway - device may still send data...")
        else:
            print(" [WARN] Warning: HardwareCommandHandler not available – skipping START command")

        # Mark reader as running and reset statistics
        self.running = True
        self.start_time = time.time()
        self.last_packet_time = time.time()
        self.data_count = 0
        self.total_packets_expected = 0
        self.total_packets_lost = 0
        self.packet_loss_percent = 0.0
        self._last_packet_counter = None
        self._total_sequence_lost = 0
        self._packet_loss_warnings = 0
        self._last_silence_warn_time = 0.0
        self.error_count = 0
        self.consecutive_errors = 0

        port_name = self.ser.port if hasattr(self.ser, "port") else "unknown"
        print(f" [OK] Packet-based ECG device started on port {port_name}")
        print(" 📡 Ready to receive data packets...")

    def _stop_due_to_serial_disconnect(self, error: Exception) -> None:
        """Stop acquisition quietly when the serial port disappears."""
        error_str = str(error)
        if not error_str:
            return
        disconnect_markers = (
            "Device not configured",
            "[Errno 6]",
            "ClearCommError",
            "Access is denied",
            "PermissionError",
            "device disconnected",
            "port is closed",
        )
        if any(marker.lower() in error_str.lower() for marker in disconnect_markers):
            print(f" Serial device disconnected: {error_str}")
            self.running = False
            try:
                if self.ser and getattr(self.ser, "is_open", False):
                    self.ser.close()
                    print(" Serial port closed after disconnect")
            except Exception:
                pass

    def stop(self):
        """Stop data acquisition and send hardware STOP command."""
        print(" Stopping packet-based ECG data acquisition...")
        self.running = False

        # Send STOP command to hardware if handler is available
        if hasattr(self, "command_handler") and self.command_handler:
            try:
                success, response = self.command_handler.send_stop_command()
                if not success:
                    print(" [WARN] Warning: STOP command ACK not received")
            except Exception as e:
                print(f" [WARN] Warning: Error sending STOP command: {e}")

        # Final packet loss statistics
        if hasattr(self, 'start_time') and self.start_time > 0:
            elapsed_time = time.time() - self.start_time
            measured_rate = (self.data_count / elapsed_time) if elapsed_time > 0 else 500.0
            seq_lost = getattr(self, '_total_sequence_lost', 0)
            total_processed = self.data_count + seq_lost
            loss_percent = (seq_lost / total_processed * 100) if total_processed > 0 else 0.0
            print(f" Total data packets received: {self.data_count}")
            print(f" Measured rate: {measured_rate:.1f} pkt/s")
            if seq_lost > 0:
                print(f" Packet loss summary: {seq_lost}/{total_processed} packets lost ({loss_percent:.2f}% loss, sequence-based)")
            else:
                print(" No packet loss detected — perfect packet continuity!")
            
            # Report sequence-based packet loss
            if hasattr(self, '_total_sequence_lost') and self._total_sequence_lost > 0:
                print(f" Sequence-based packet loss: {self._total_sequence_lost} packets lost (detected via counter gaps)")
            else:
                print(f" No sequence gaps detected - perfect packet continuity!")
        else:
            print(f" Total data packets received: {self.data_count}")
            if hasattr(self, '_total_sequence_lost') and self._total_sequence_lost > 0:
                print(f" Sequence-based packet loss: {self._total_sequence_lost} packets lost")

    def read_packets(self, max_packets: int = 100) -> List[Dict[str, int]]:
        """Read and parse ECG packets from serial stream
        
        At 500 Hz, hardware sends 500 packets/second = ~16.67 packets per 33ms timer interval.
        We read up to max_packets to prevent buffer overflow and packet loss.
        """
        if not self.running:
            return []
            
        out: List[Dict[str, int]] = []
        
        try:
            bytes_to_read = 0
            if hasattr(self.ser, 'in_waiting'):
                try:
                    bytes_to_read = self.ser.in_waiting
                except Exception as e:
                    self._stop_due_to_serial_disconnect(e)
                    return []
            
            # If no data is available, return immediately to keep UI responsive
            if bytes_to_read == 0:
                return []
            
            # Always read ALL available bytes to completely drain the OS/hardware UART buffer.
            # Python's bytearray has virtually no memory limit, whereas the OS buffer is small (often 4KB).
            # If we don't completely drain the OS buffer when the UI thread blocks (e.g. tab switching),
            # the OS buffer will overflow and hardware packets will be permanently lost.
            read_size = bytes_to_read
            
            if getattr(self.ser, 'timeout', None) in (None, 0):
                self.ser.timeout = self._read_timeout
            chunk = self.ser.read(read_size)
            if chunk:
                self.buf.extend(chunk)

            # Extract packets - process ALL available packets to prevent buffer overflow
            # At 500 Hz, we need to process packets quickly to avoid accumulation
            # CRITICAL: Process ALL packets in buffer, not limited by max_packets, to prevent packet loss
            # Use max_iterations as safety limit, but process as many as needed
            max_iterations = max(max_packets * 5, 500)  # Allow catching up significantly if we fell behind
            iteration = 0
            packets_processed = 0
            dropped_for_ui = 0
            dynamic_budget = float(self.READ_LOOP_BUDGET_SECONDS)
            if bytes_to_read > 8192 or len(self.buf) > 30000:
                dynamic_budget = max(dynamic_budget, 0.035)
            if len(self.buf) > 80000:
                dynamic_budget = max(dynamic_budget, 0.05)
            parse_deadline = time.perf_counter() + dynamic_budget

            while iteration < max_iterations and len(self.buf) >= PACKET_SIZE:
                if time.perf_counter() >= parse_deadline:
                    # Budget exhausted; continue next timer tick to keep UI responsive
                    break
                iteration += 1
                start_idx = self.buf.find(bytes([START_BYTE]))
                if start_idx == -1:
                    # No start byte found - keep only tail to avoid unbounded growth
                    if len(self.buf) > self.MAX_BUFFER_BYTES:
                        print(f" Serial buffer overflow risk: {len(self.buf)} bytes, trimming garbage")
                        del self.buf[:-PACKET_SIZE]
                    break
                if start_idx > 10000:
                    # Skip too much garbage data before start byte
                    del self.buf[:start_idx]
                    continue
                if len(self.buf) - start_idx < PACKET_SIZE:
                    # Not enough data for a complete packet - keep what we have
                    if start_idx > 0:
                        del self.buf[:start_idx]
                    break
                    
                candidate = bytes(self.buf[start_idx : start_idx + PACKET_SIZE])
                del self.buf[: start_idx + PACKET_SIZE]

                if candidate[-1] != END_BYTE:
                    continue

                parsed = parse_packet(candidate)
                if parsed:
                    self.data_count += 1
                    self.last_packet_time = time.time()
                    packets_processed += 1
                    
                    # Extract packet counter for sequence tracking
                    packet_counter = candidate[1] & 0x3F  # Counter is in lower 6 bits (0-63)
                    
                    # Detect packet loss by checking sequence continuity
                    if self._last_packet_counter is not None:
                        expected_counter = (self._last_packet_counter + 1) % 64
                        if packet_counter != expected_counter:
                            # Calculate how many packets were lost
                            if packet_counter > expected_counter:
                                lost = packet_counter - expected_counter
                            else:
                                # Wrapped around (e.g., 63 -> 0)
                                lost = (64 - expected_counter) + packet_counter
                            
                            if lost > 0:
                                self._total_sequence_lost += lost
                                self._packet_loss_warnings += 1
                                # Only warn every 50th occurrence to avoid spam (optimized for performance)
                                if self._packet_loss_warnings % 50 == 1:
                                    print(f" [WARN] Packet loss: {lost} dropped (Total: {self._total_sequence_lost})")
                    
                    self._last_packet_counter = packet_counter
                    
                    # Log every 250th packet (2 times per second) with numeric values, time and duration
                    if self.data_count % 250 == 0:
                        loss_info = f" (Lost: {self._total_sequence_lost})" if self._total_sequence_lost > 0 else ""
                        elapsed = time.time() - self.start_time
                        hr_time = time.strftime("%H:%M:%S")
                        
                        leads_str = []
                        for lead_key in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
                            val = parsed.get(lead_key, 0)
                            leads_str.append(f"{lead_key}:{val}")
                            
                        print(f" 📡 [{hr_time} | {elapsed:.2f}s] Pkt #{self.data_count}{loss_info} | {' '.join(leads_str)}")
                        
                    # We append EVERYTHING we parse. The safety is the time budget (parse_deadline).
                    # Dropping packets here causes missed heartbeats and false arrhythmias in the backend.
                    out.append(parsed)
            
            # If we processed many packets, we're catching up - this is good
            if packets_processed > max_packets * 2:
                # We're catching up from backlog - this is expected and good
                pass

            if dropped_for_ui > 0:
                if not hasattr(self, '_ui_drop_warn_time') or (time.time() - self._ui_drop_warn_time) > 2.0:
                    print(f" [WARN] UI throttle active: skipped {dropped_for_ui} packets this tick to keep UI responsive")
                    self._ui_drop_warn_time = time.time()
            
            # Warn if buffer is accumulating too much data (indicates we're falling behind)
            if len(self.buf) > 50000:  # >50KB buffer indicates we're not reading fast enough
                if not hasattr(self, '_buffer_warn_time') or (time.time() - self._buffer_warn_time) > 5.0:
                    print(f" [WARN] Buffer: {len(self.buf)} bytes")
                    self._buffer_warn_time = time.time()

            # Device connected but not sending valid packets — surface warning without freezing.
            if self.is_device_silent() and (time.time() - self._last_silence_warn_time) > 3.0:
                print(" [WARN] Device not sending valid ECG packets")
                self._last_silence_warn_time = time.time()
            
            # Update packet loss statistics
            if self.running and self.data_count > 0:
                seq_lost = getattr(self, '_total_sequence_lost', 0)
                total_processed = self.data_count + seq_lost
                self.total_packets_expected = total_processed
                self.total_packets_lost = seq_lost
                if total_processed > 0:
                    self.packet_loss_percent = (seq_lost / total_processed) * 100
                    
        except Exception as e:
            self.error_count += 1
            self.consecutive_errors += 1
            error_msg = f"Packet parsing error: {e}"
            print(f" {error_msg}")
            self.crash_logger.log_error(
                message=error_msg,
                exception=e,
                category="SERIAL_ERROR"
            )
            
            # If device is disconnected (Errno 6) or too many consecutive errors, stop
            error_str = str(e)
            if "Device not configured" in error_str or "[Errno 6]" in error_str or \
               "ClearCommError" in error_str or "Access is denied" in error_str or \
               "PermissionError" in error_str or "element not found" in error_str.lower() or \
               self.consecutive_errors > 20:
                print(f" Critical serial error ({error_str}) - stopping acquisition")
                self.running = False
                self._stop_due_to_serial_disconnect(e)
                
                try:
                    # Explicitly close the port on critical error to allow fresh start
                    if self.ser and self.ser.is_open:
                        self.ser.close()
                        print(" Serial port closed due to critical error")
                except Exception:
                    pass
            
        return out

    def _handle_serial_error(self, error):
        """Handle serial communication errors"""
        current_time = time.time()
        self.error_count += 1
        self.consecutive_errors += 1
        
        error_msg = f"Serial communication error: {error}"
        print(f" {error_msg}")
        
        self.crash_logger.log_error(
            message=error_msg,
            exception=error,
            category="SERIAL_ERROR"
        )
        
        if self.consecutive_errors >= 5 and (current_time - self.last_error_time) > 10:
            self.last_error_time = current_time
            self.consecutive_errors = 0


class SerialECGReader:
    """Legacy serial reader for line-based ECG data"""
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
        # INSTANT START: Removed delay for immediate wave display
        # time.sleep(0.1)  # Removed for instant startup
        self.running = True
        print(" ECG device started - waiting for data...")

    def stop(self):
        print(" Stopping ECG data acquisition...")
        try:
            self.ser.write(b'0\r\n')
        except Exception:
            pass
        self.running = False
        print(f" Total data packets received: {self.data_count}")

    def read_value(self):
        if not self.running:
            return None
        try:
            line_raw = self.ser.readline()
            line_data = line_raw.decode('utf-8', errors='replace').strip()

            if line_data:
                self.data_count += 1
                # Print detailed data information
                print(f" [Packet #{self.data_count}] Raw data: '{line_data}' (Length: {len(line_data)})")
                
                # Parse and display ECG value
                if line_data.isdigit():
                    ecg_value = int(line_data[-3:])
                    print(f" ECG Value: {ecg_value} mV")
                    return ecg_value
                else:
                    # Try to parse as multiple values (8-channel data)
                    try:
                        # Clean the line data - remove any non-numeric characters except spaces and minus signs
                        import re
                        cleaned_line = re.sub(r'[^\d\s\-]', ' ', line_data)
                        values = [int(x) for x in cleaned_line.split() if x.strip() and x.replace('-', '').isdigit()]
                        
                        if len(values) >= 8:
                            print(f" 8-Channel ECG Data: {values}")
                            return values  # Return the list of 8 values
                        elif len(values) == 1:
                            print(f" Single ECG Value: {values[0]} mV")
                            return values[0]
                        elif len(values) > 0:
                            print(f" Unexpected number of values: {len(values)} (expected 8)")
                        else:
                            return None
                    except Exception as e:
                        print(f" Error parsing ECG data: {e}")
                        return None
            else:
                print("[WAIT] No data received (timeout)")
                
        except Exception as e:
            self._handle_serial_error(e)
        return None

    def close(self):
        print(" Closing serial connection...")
        try:
            self.ser.close()
        except Exception:
            pass
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


# ─────────────────────────────────────────────────────────────────────────────
# Background worker: runs the entire connect + start sequence off the Qt
# main thread so the UI never freezes while the device handshakes.
# ─────────────────────────────────────────────────────────────────────────────
class DeviceStartWorker(QThread):
    """
    Runs scan_and_detect_port -> serial_reader.start() in a background thread.

    Signals
    -------
    connected(bool, str, str)
        Emitted when the sequence finishes.
        Args: (success, port_used, error_message)
    version_ready(str)
        Emitted (possibly empty string) once the device version is known.
    """
    connected = pyqtSignal(bool, str, str)   # success, port, error_msg
    version_ready = pyqtSignal(str)           # version string (may be empty)
    serial_ready = pyqtSignal(str)            # machine serial number (may be empty)

    def __init__(self, port: str, baud_int: int, reader, parent=None):
        """
        Parameters
        ----------
        port      : configured port string ("Select Port" / None means auto-scan)
        baud_int  : baud rate as int
        reader    : an already-instantiated SerialStreamReader (or None if
                    GlobalHardwareManager hasn't created one yet)
        """
        super().__init__(parent)
        self._port = port
        self._baud_int = baud_int
        # Keep a reference so the caller can read reader.device_version etc.
        self._reader = reader
        self._port_to_use = None   # filled in by run()

    @property
    def port_to_use(self):
        """The port that was actually used (available after connected signal)."""
        return self._port_to_use
    @staticmethod
    def _should_retry_with_scan(error: Exception) -> bool:
        """Retry auto-scan only for stale/invalid configured-port failures."""
        error_str = str(error).lower()
        retry_markers = (
            "cannot configure port",
            "a device attached to the system is not functioning",
            "access is denied",
            "clearcommerror failed",
            "[errno 6]",
            "device not configured",
            "element not found",
            "port is closed",
        )
        return any(marker in error_str for marker in retry_markers)


    def run(self):
        """Background work: port scan (if needed) -> open -> VERSION/START -> (if version ok) SERIAL -> connected."""
        try:
            port = self._port
            baud_int = self._baud_int

            # ── Determine whether we need to auto-scan ────────────────────
            scan_needed = port in ("Select Port", None, "")
            if not scan_needed and SERIAL_AVAILABLE:
                try:
                    available = [p.device for p in serial.tools.list_ports.comports()]
                    if port not in available:
                        print(f" [Worker] Configured port {port} not found – forcing scan")
                        scan_needed = True
                except Exception:
                    pass

            # Auto-scan with this app's own reader still holding a port (Stop
            # stops the stream but keeps the port open): the scan probe can't
            # open that port again -- Windows answers "Access is denied" (seen:
            # COM4 denied on the second Connect & Start, device not found).
            # Use that port first; reader.start() reopens it itself.
            from ecg.serial.serial_reader import GlobalHardwareManager
            reused_port = False
            if scan_needed and SERIAL_AVAILABLE:
                existing = GlobalHardwareManager().reader
                existing_port = getattr(getattr(existing, "ser", None), "port", None)
                try:
                    present = [p.device for p in serial.tools.list_ports.comports()]
                except Exception:
                    present = []
                if existing_port and existing_port in present:
                    print(f" [Worker] Reusing this app's open port {existing_port}")
                    port = existing_port
                    scan_needed = False
                    reused_port = True

            port_to_use = port
            if scan_needed:
                print(" [Worker] Scanning ports for ECG device…")
                scan_result = SerialStreamReader.scan_and_detect_port(
                    baudrate=baud_int, timeout=0.2
                )
                if scan_result:
                    port_to_use, detected_ser = scan_result
                    print(f" [Worker] Auto-detected port: {port_to_use}")
                    # Close the scan probe connection so the reader can open fresh
                    try:
                        if detected_ser and detected_ser.is_open:
                            detected_ser.close()
                    except Exception:
                        pass
                else:
                    self.connected.emit(False, "", "No ECG device found on any port")
                    return

            self._port_to_use = port_to_use

            # ── Get / create the shared reader ────────────────────────────
            reader = GlobalHardwareManager().get_reader(port_to_use, baud_int)
            if reader is None:
                self.connected.emit(False, port_to_use, "Failed to create serial reader")
                return
            self._reader = reader

            # ── Start (opens port, VERSION handshake, START command) ──────
            try:
                reader.start()
            except Exception as start_exc:
                # Only retry if a configured port failed for the stale/disconnected
                # device cases we want to recover from. Keep every other scenario
                # unchanged so we do not alter normal COM-port handling.
                if scan_needed or not (reused_port or self._should_retry_with_scan(start_exc)):
                    raise
                if reused_port:
                    # Release our own handle first, or the scan is denied it.
                    GlobalHardwareManager().close_reader()
                print(" [Worker] Retrying with full auto-scan across all serial ports…")

                scan_result = SerialStreamReader.scan_and_detect_port(
                    baudrate=baud_int, timeout=0.2
                )
                if not scan_result:
                    self.connected.emit(
                        False,
                        port_to_use,
                        f"Configured port failed ({start_exc}); no ECG device found on any port",
                    )
                    return

                scanned_port, detected_ser = scan_result
                try:
                    if detected_ser and detected_ser.is_open:
                        detected_ser.close()
                except Exception:
                    pass

                self._port_to_use = scanned_port
                reader = GlobalHardwareManager().get_reader(scanned_port, baud_int)
                if reader is None:
                    self.connected.emit(False, scanned_port, "Failed to create serial reader after scan")
                    return
                self._reader = reader
                reader.start()
                port_to_use = scanned_port

            # VERSION must succeed to consider the device connected (per requirement).
            version = reader.device_version or ""
            self.version_ready.emit(version)

            if not version:
                self.serial_ready.emit("")
                self.connected.emit(False, port_to_use, "VERSION handshake failed (no version received)")
                return

            # Serial number is fetched inside reader.start() right after VERSION and before START.
            self.serial_ready.emit(reader.device_serial_number or "")

            self.connected.emit(True, port_to_use, "")

        except Exception as exc:
            import traceback
            err = f"{exc}\n{traceback.format_exc()}"
            print(f" [Worker] Error during start: {err}")
            self.connected.emit(False, self._port_to_use or "", str(exc))
