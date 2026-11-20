"""Analyse a recording completely before it is opened.

A new recording (live recording stopped, or imported) waits in the recordings
list, its "Analysis" column "Not analyzed" (is_analysed). The user opens it
by double-click (or View): analysed, it opens at once; not analysed,
holter_ui._open_recording asks "This recording is not analyzed. Would you
like to analyse it?" -- No stays on the list, Yes calls open_when_analysed():
the "Analysing recording... N%" popup runs every signal-only analysis, in
one fixed order, and the recording opens at 100%. analysis_done.json then
marks it analysed for its file, code, patient age and configuration.

Why: the analyses used to start in the background when a recording opened,
each on its own thread, and the report preview called the same functions
directly -- it did not wait for a running background job but started the
same computation again, both on the same CPU. On 87 min of recording the
first preview took 416 s; with every result saved it takes 3 s
(2026-10-30_12-01-08, 2026-11-19). Run once here, every panel and the report
read the saved results.

The analyses (all from the signal alone; the "Auto Arrhythmia Detection"
button only shows or hides the auto segments, it does not compute them):
  1. auto segments            auto_segment_arrthymia_detection.detect_arrhythmias
  2. Asystole                 auto_segment_arrthymia_detection.detect_asystole_segments
  3. lead-off spans           lead_off_detection.detect_lead_off_spans
  4. PVC / PAC beats          ectopic_beats_detector.detect_ectopic_beats (needs 1)
  5. ST / T episodes, QRS axis  ischemia_detection.compute_ischemia
  6. PQRST intervals          pqrst_measure.recording_frames
  7. Full Disclosure beats    report_generator._full_disclosure_beat_times
Each is the module's own cached function: what it computes and saves is
unchanged, this only decides when it runs. Manual markings, events and edited
strips are not analyses and are not touched.
"""

import os
import threading
import time
from typing import Callable, Optional

# (name, seconds of work per second of recording) -- the weight of each step
# in the percentage. Measured with this module, cold (no saved result), on
# 2026-10-30_12-01-08 (5210 s) on the development PC: 102.0, 0.1, 2.4, 29.4,
# 67.9, 17.4, 6.7 s -- 226 s in all (~1 h for 24 h of recording). Only the
# ratios matter: the run scales them by the speed it sees on the steps done.
_STEPS = (
    ('segments', 0.01960),
    ('asystole', 0.00002),
    ('lead_off', 0.00046),
    ('ectopic', 0.00564),
    ('ischemia', 0.01300),
    ('pqrst', 0.00334),
    ('fd_beats', 0.00129),
)

_POPUP_DELAY_MS = 300      # a recording analysed before opens without a popup flash
_busy = False              # one analysis at a time (the popup is modal)


def _ecgh(session_dir: str) -> str:
    return os.path.join(session_dir, 'recording.ecgh')


# Written by run_analysis when every step has its result saved: the key each
# result was saved under. is_analysed compares it with the keys the modules
# would use now -- the file (size / time), the code, the patient's age, the
# clinical configuration -- without reading the results themselves (the
# recordings list asks for every row: reading a 24 h recording's beats and
# frames took seconds, and the first check in a process ~2 s even for 1 min).
_DONE_FILENAME = 'analysis_done.json'
_RESULT_FILES = ('auto_segments_cache.json', 'asystole_segments_cache.json', 'lead_off_cache.json',
                 'ectopic_beats_cache.json', 'ischemia_cache.json', 'pqrst_cache.json',
                 'full_disclosure_beats_cache.json')


def _current_keys(session_dir: str) -> Optional[dict]:
    """The key of every saved result as the modules build it now; None when
    one cannot be built (no file, or code that cannot be read: no disk cache
    then, so nothing can be "analysed" either)."""
    import json
    path = _ecgh(session_dir)
    try:
        from .auto_segment_arrthymia_detection import _cache_key as segments_key, _code_fingerprint
        if not _code_fingerprint():
            return None
        from .ectopic_beats_detector import _cache_key as ectopic_key
        from .ischemia_detection import _cache_key as ischemia_key
        from .lead_off_detection import _cache_key as lead_off_key
        from .pqrst_measure import _frames_cache_key
        from .report_generator import _fd_beats_key
        pqrst = _frames_cache_key(path)
        keys = {
            'segments': segments_key(path),       # also the Asystole result's key
            'lead_off': lead_off_key(path),
            'ectopic': ectopic_key(path),
            'ischemia': ischemia_key(path),
            'pqrst': json.dumps(pqrst) if pqrst[3] else None,
            'fd_beats': _fd_beats_key(session_dir),
        }
    except Exception as e:
        print(f"[Analysis] Could not build the result keys: {e}")
        return None
    if not all(keys.values()) or 'nocode' in str(keys['ischemia']):
        return None
    return keys


def is_analysed(session_dir: str) -> bool:
    """Every analysis of the recording is saved for the current file, code,
    patient age and configuration: analysis_done.json holds the keys the
    modules would use now and every result file is there. Computes nothing
    and reads no result (a few ms). A recording analysed before this file
    existed shows as not analysed; analysing it then only reads its saved
    results and writes the file."""
    import json
    if not os.path.exists(_ecgh(session_dir)):
        return False
    try:
        with open(os.path.join(session_dir, _DONE_FILENAME), 'r', encoding='utf-8') as f:
            done = json.load(f)
    except (OSError, ValueError):
        return False
    if not all(os.path.exists(os.path.join(session_dir, name)) for name in _RESULT_FILES):
        return False
    keys = _current_keys(session_dir)
    return keys is not None and isinstance(done, dict) and done.get('keys') == keys


def _write_done(session_dir: str) -> None:
    """analysis_done.json, after every step of run_analysis has its result."""
    import json
    keys = _current_keys(session_dir)
    if keys is None:
        return
    target = os.path.join(session_dir, _DONE_FILENAME)
    tmp = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'keys': keys, 'analysed_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)
        os.replace(tmp, target)
    except OSError as e:
        print(f"[Analysis] Could not save {_DONE_FILENAME}: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


class _Progress:
    """The run's percentage, read by the popup's timer (main thread) while
    the worker thread moves through the steps."""

    def __init__(self, duration_sec: float):
        self.duration = max(1.0, float(duration_sec))
        self.total = sum(rate for _, rate in _STEPS) * self.duration
        self.done_cost = 0.0          # expected cost of the finished steps
        self.step_cost = 0.0          # expected cost of the running step
        self.step_start = time.time()
        self.speed = 1.0              # actual / expected seconds, from the steps that computed
        self._spent = 0.0
        self._expected = 0.0
        self._shown = 0

    def start_step(self, rate: float):
        self.step_cost = rate * self.duration
        self.step_start = time.time()

    def end_step(self, computed: bool):
        if computed and self.step_cost > 0:
            self._spent += time.time() - self.step_start
            self._expected += self.step_cost
            self.speed = max(0.05, self._spent / self._expected)
        self.done_cost += self.step_cost
        self.step_cost = 0.0

    def percent(self) -> int:
        # the running step: its expected time at the speed seen so far, held
        # below its end until it really ends
        expected = self.step_cost * self.speed
        frac = 0.0 if expected <= 0 else min(0.95, (time.time() - self.step_start) / expected)
        pct = int(100.0 * (self.done_cost + frac * self.step_cost) / self.total)
        self._shown = max(self._shown, min(99, pct))      # never back, 100 only at the end
        return self._shown


def run_analysis(session_dir: str, progress: Optional[_Progress] = None) -> None:
    """Every analysis of the recording, in order; each one saved by its own
    module (a step already saved returns at once)."""
    path = _ecgh(session_dir)
    from .auto_segment_arrthymia_detection import detect_arrhythmias, detect_asystole_segments
    from .ectopic_beats_detector import detect_ectopic_beats
    from .file_format import ECGHFileReader
    from .ischemia_detection import compute_ischemia, get_cached_ischemia
    from .lead_off_detection import detect_lead_off_spans
    from .pqrst_measure import recording_frames
    from .report_generator import _full_disclosure_beat_times

    reader = ECGHFileReader(path)
    try:
        # The percentage of every step is its expected time (no step's own
        # progress callback: detect_arrhythmias reports 100% when its
        # parallel part ends, and its bundle branch block pass -- a third of
        # its time -- then runs at a standing 100%).
        def step(name, run):
            t0 = time.time()
            if progress is not None:
                progress.start_step(dict(_STEPS)[name])
            result = run()
            dt = time.time() - t0
            if progress is not None:
                progress.end_step(computed=dt > 1.0)
            print(f"[Analysis] {os.path.basename(session_dir)}: {name} {dt:.1f} s")
            return result

        segments = step('segments', lambda: detect_arrhythmias(reader))
        step('asystole', lambda: detect_asystole_segments(reader))
        step('lead_off', lambda: detect_lead_off_spans(reader))
        step('ectopic', lambda: detect_ectopic_beats(reader, segments))
        step('ischemia', lambda: get_cached_ischemia(path) or compute_ischemia(path))
        step('pqrst', lambda: recording_frames(path))
        step('fd_beats', lambda: _full_disclosure_beat_times(session_dir))
    finally:
        reader.close()
    _write_done(session_dir)


def open_when_analysed(parent, session_dir: str, on_ready: Callable[[], None]) -> None:
    """Call on_ready() (which opens the recording) once every analysis of the
    recording is saved: at once when it already is, else after the
    "Analysing recording... N%" popup has run them. A failed analysis still
    opens the recording -- each panel then computes what it needs, as before
    this module. Must be called from the Qt main thread."""
    global _busy
    from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
    from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout

    if _busy:
        return
    if not session_dir or not os.path.exists(_ecgh(session_dir)):
        on_ready()
        return
    _busy = True

    try:
        from .file_format import ECGHFileReader
        _rd = ECGHFileReader(_ecgh(session_dir))
        duration = float(_rd.duration_sec)
        _rd.close()
    except Exception:
        duration = 0.0
    progress = _Progress(duration)

    class _Worker(QThread):
        finished_ok = pyqtSignal(bool, str)

        def run(self):
            try:
                if not is_analysed(session_dir):
                    run_analysis(session_dir, progress)
                self.finished_ok.emit(True, '')
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.finished_ok.emit(False, str(e))

    dialog = QDialog(parent, Qt.Dialog | Qt.FramelessWindowHint)
    dialog.setObjectName("analysisProgress")
    dialog.setAttribute(Qt.WA_StyledBackground, True)
    dialog.setFixedSize(420, 110)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setStyleSheet("""
        QDialog#analysisProgress {
            background: #0D1117;
            border: 1px solid #00CC66;
            border-radius: 12px;
        }
        QDialog#analysisProgress QLabel {
            background: transparent; border: none;
            color: #FFFFFF; font-size: 16px; font-weight: bold;
        }
    """)
    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(28, 22, 28, 22)
    label = QLabel("Analysing recording... 0%")
    label.setAlignment(Qt.AlignCenter)
    layout.addWidget(label)
    # no closing it half way: the recording opens when the analysis is done
    dialog.reject = lambda: None

    timer = QTimer(dialog)
    timer.setInterval(200)
    timer.timeout.connect(lambda: label.setText(f"Analysing recording... {progress.percent()}%"))

    worker = _Worker()
    # Whether the run has ended, as a plain Python flag: a run shorter than
    # _POPUP_DELAY_MS (a recording analysed before) has already deleted the
    # worker and the dialog (deleteLater) when show_popup fires, and asking
    # the deleted QThread worker.isRunning() raised "wrapped C/C++ object of
    # type _Worker has been deleted".
    state = {'done': False}

    def show_popup():
        if not state['done']:
            timer.start()
            dialog.show()

    def finished(ok, err):
        global _busy
        state['done'] = True
        timer.stop()
        if dialog.isVisible():
            label.setText("Analysing recording... 100%")
        dialog.hide()
        dialog.deleteLater()
        _busy = False
        if not ok:
            print(f"[Analysis] {os.path.basename(session_dir)} failed, opening anyway: {err}")
        on_ready()

    worker.finished_ok.connect(finished)
    worker.finished.connect(worker.deleteLater)
    if parent is not None:
        parent._recording_analysis_worker = worker      # keep it alive while it runs
    worker.start()
    QTimer.singleShot(_POPUP_DELAY_MS, show_popup)
