"""
ecg/holter/sleep_guard.py
=========================
Keeps the machine awake while a Holter recording is running.

A recording only gets frames while the process is actually running. If the OS
goes to sleep (Windows "System Idle" -> S3), the whole process is frozen: no
serial reads, no frames written, but the wall-clock UI timer keeps counting
once it wakes up. A 15 h recording once ended up with only ~1 h of data this
way. SleepGuard asks the OS not to idle-sleep until release() is called.

The display may still turn off; only system sleep is blocked. A user-initiated
sleep (Start menu, lid close, power button) cannot be blocked by an app.
"""

import subprocess
import sys
import threading

_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


class SleepGuard:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_evt = None
        self._thread = None
        self._proc = None

    @property
    def active(self) -> bool:
        return self._thread is not None or self._proc is not None

    def acquire(self) -> bool:
        """Block idle sleep. Safe to call repeatedly. Returns True if held."""
        with self._lock:
            if self.active:
                return True
            try:
                if sys.platform == "win32":
                    return self._acquire_windows()
                if sys.platform == "darwin":
                    self._proc = subprocess.Popen(["caffeinate", "-i"])
                    return True
            except Exception as exc:
                print(f"[SleepGuard] Could not block system sleep: {exc}")
            return False

    def release(self) -> None:
        """Allow the OS to sleep again. Safe to call when not held."""
        with self._lock:
            if self._stop_evt is not None:
                self._stop_evt.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            if self._proc is not None:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
            self._stop_evt = None
            self._thread = None
            self._proc = None

    def _acquire_windows(self) -> bool:
        import ctypes

        stop_evt = threading.Event()
        ready = threading.Event()
        ok = []

        # SetThreadExecutionState(ES_CONTINUOUS) is tied to the calling
        # thread, so a dedicated thread holds it and clears it itself; that
        # keeps release() correct whichever thread calls it, and Windows drops
        # the request automatically if the process dies.
        def _hold():
            kernel32 = ctypes.windll.kernel32
            ok.append(bool(kernel32.SetThreadExecutionState(
                _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED)))
            ready.set()
            stop_evt.wait()
            kernel32.SetThreadExecutionState(_ES_CONTINUOUS)

        thread = threading.Thread(target=_hold, name="holter-sleep-guard", daemon=True)
        thread.start()
        ready.wait(timeout=2.0)
        if not (ok and ok[0]):
            stop_evt.set()
            thread.join(timeout=2.0)
            print("[SleepGuard] SetThreadExecutionState failed; system may sleep")
            return False
        self._stop_evt = stop_evt
        self._thread = thread
        print("[SleepGuard] System sleep blocked while recording")
        return True
