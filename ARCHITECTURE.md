# Comprehensive ECG Analysis — Architecture

Comprehensive ECG Analysis is a standalone desktop app (PyQt5) for long-term
(Holter-style) 12-lead ECG recording and review. It reads the ECG device over
USB serial, shows the 12 leads live, records to disk, analyses the recording
in the background and produces a PDF report.

It used to live inside the CardioX app and was started from the CardioX
dashboard. It has been separated: there is no login, cloud upload, licensing
or CardioX 12-lead test page any more. The CardioX code can still be found in
git history before the `comprehensive-standalone` branch.

```
python src/main.py
```

---

## 1. Big picture

```
 ECG device (USB serial, 500 packets/s)
        │  22-byte packets
        ▼
 ecg/serial/serial_reader.py     SerialStreamReader  ── read_packets()
 ecg/serial/packet_parser.py     parse_packet()      ── {lead: ADC | None}
 ecg/serial/hardware_commands.py VERSION / SERIAL / START / STOP
        │
        ▼
 ecg/holter/live_12lead_plotter.py
   DeviceAcquisition  (QTimer, 20 ms)  ── packets_ready(list)
        │
        ├────────────► Live12LeadPlotter   live 3x4 / 6x2 view, keeps 10 s in .data
        │
        └── (while recording) ─► ecg/holter/stream_writer.py  HolterStreamWriter.push()
                                    │  recording.ecgh  (raw, every packet)
                                    │  5 s chunks ─► analysis_queue
                                    ▼
                                 ecg/holter/analysis_worker.py  HolterAnalysisWorker (thread)
                                    │  metrics.jsonl, events.db
                                    ▼
                                 ecg/holter/holter_ui.py  HolterMainWindow
                                    (live summary every 1 s; review, edit, report)
                                    │
                                    ▼
                                 ecg/holter/report_generator.py  ── PDF
```

`src/main.py` (`ComprehensiveMainWindow`) wires this together: port picker,
Connect & Start / Stop, Start / Stop Recording, Open Workspace and Previous
Recordings.

---

## 2. Reading the device

### 2.1 Connection (`ecg/serial/serial_reader.py`)

`DeviceStartWorker` (a `QThread`, so the UI never freezes) does:

1. **Port**: uses the chosen COM port, or auto-scans every port with
   `SerialStreamReader.scan_and_detect_port()` (VERSION reply first, START ACK
   as a fallback for old firmware).
2. **Open**: `GlobalHardwareManager().get_reader(port, 115200)`, a single
   shared `SerialStreamReader`.
3. **Handshake**: `reader.start()` sends VERSION → machine SERIAL → START.
   The device counts as connected only if VERSION answers.
4. Emits `connected(ok, port, error)`, `version_ready`, `serial_ready`.

After that, `read_packets()` is polled. It drains every byte waiting in the OS
buffer (so a busy UI thread doesn't lose packets), re-syncs on the start/end
bytes, tracks packet loss from the 6-bit sequence counter, and returns a list
of parsed packets.

### 2.2 Command frames (`ecg/serial/hardware_commands.py`)

22 bytes: `E8 | counter | 0x11 | opcode | checksum | 16 data bytes | 8E`

| Opcode | Command | Reply |
|---|---|---|
| `0x10` | START streaming | ACK `0x21` |
| `0x11` | STOP streaming | ACK `0x21` |
| `0x13` | Machine serial number | `0x23` + 16 ASCII bytes |
| `0x14` | Firmware version | `0x24` + version bytes |
| `0x15` | Software close | ACK `0x21` |

### 2.3 Data packets (`ecg/serial/packet_parser.py`)

22 bytes at 500 Hz:

```
byte 0      0xE8 start
byte 1      counter (low 6 bits, 0-63, used for loss detection)
bytes 2-4   flags
bytes 5-20  8 leads x (MSB, LSB): I, II, V1, V2, V3, V4, V5, V6
byte 21     0x8E end
```

Each lead: `value = (MSB & 0x1F) << 7 | (LSB & 0x7F)` → 12-bit ADC, 2048 = 0 mV.
`MSB & 0x20` = that electrode is connected.

Electrode bits: LA = lead I MSB bit 5, RA = lead II MSB bit 5, LL = lead II MSB
bit 6. RL is inferred: LA and LL dropping together within the last 25 packets
means RL is off.

Lead-off rules: RA or RL off → every lead off. LA off → I and V1-V6 off. LL off →
II and V1-V6 off. A chest electrode's own bit off → that V lead off. A lead
that is off comes back as `None`, not as noise.

Derived limb leads (Einthoven / Goldberger, an off lead counts as 2048):

```
III = II - I      aVR = -(I + II) / 2      aVL = I - II/2      aVF = II - I/2
```

The parser also returns `__ra_present__`, `__la_present__`, `__ll_present__`
and `__rl_present__` for the UI.

Scale: `ADC_TO_MV = 1/200` (`ecg/holter/theme.py`), i.e. 200 ADC counts = 1 mV.

---

## 3. Live 12-lead plotter (`ecg/holter/live_12lead_plotter.py`)

* **`DeviceAcquisition`** wraps section 2: `start(port)`, `stop()`, `close()`,
  and the signals `connected`, `device_info`, `packets_ready(list)` and
  `stopped`. It polls every 20 ms and reports packets/s.
* **`Live12LeadPlotter`**
  * Layouts: **3x4** (I II III / aVR aVL aVF / V1-V3 / V4-V6, 2.5 s per lead) and
    **6x2** (limb leads / chest leads, 5 s per lead). A whole sheet is 10 s.
  * Speed 12.5 / **25** / 50 mm/s, gain 5 / **10** / 20 mm/mV, with a 1 mV
    calibration pulse per row.
  * Everything is drawn in millimetres with the pyqtgraph aspect ratio locked
    to 1, so the paper is to scale whatever the window size. The grid has 1 mm
    minor and 5 mm major lines.
  * Display filter: `signal_paths.display_filter` (0.5–40 Hz Butterworth,
    zero-phase), run on the newest 4 s and redrawn every 50 ms. It can be
    switched off; the trace then has its median removed instead.
  * A lead that is off holds its last value and shows **LEAD OFF**.
  * `.data`: the last 10 s of raw ADC as 12 arrays in `file_format.LEAD_NAMES`
    order. This is the `live_source` contract `HolterMainWindow` reads during a
    live recording.
  * `fill_lead_off(packet, last_valid)`: the packet sent to the writer has every
    lead numeric (an off lead holds its last value), so no sample is dropped
    and the recording clock stays exact.

---

## 4. Recording and analysis (`ecg/holter/`)

| Module | Role |
|---|---|
| `stream_writer.py` | `HolterStreamWriter`: `push(packet)` on the UI thread (<0.1 ms). Writes `recording.ecgh`, keeps a 120 s display ring, cuts 5 s analysis chunks into `analysis_queue`, holds the PC awake (`sleep_guard.py`). `stop()` flushes, stops the worker and returns the summary. |
| `file_format.py` | `.ecgh`: 256-byte header (magic, version, leads, fs, time, patient) + 26-byte frames (12 × int16 + frame index), plus `_index.json`. |
| `analysis_worker.py` | `HolterAnalysisWorker` thread: QRS, beats, rhythm and HRV per chunk → `metrics.jsonl`, `events.db`, live summary back to the writer. |
| `analysis_pipeline.py`, `beat_models.py`, `holter_bpm_engine.py`, `hrv_metrics.py`, `holter_summary_calc.py`, `summary_utils.py` | Per-chunk analysis, beat/episode models, BPM, HRV and session summary. |
| `auto_segment_arrthymia_detection.py` | Rule-based auto arrhythmia segments (cache: `auto_segments_cache.json`). |
| `ischemia_detection.py` | ST elevation / depression and T inversion episodes (cache: `ischemia_cache.json`). |
| `clinical_config.py` + `config/clinical_config.yaml` | Clinical thresholds. Override the path with `CARDIOX_CLINICAL_CONFIG`. |
| `session_store.py`, `replay_engine.py` | Read finished sessions and random-access the `.ecgh` for review. |
| `report_generator.py` | Final PDF report. |

Rule: **pure signal steps run per chunk; anything with history (RR, rhythm,
AF burden, templates, event fusion) stays in one ordered consumer.**

Shared signal code used by Holter, outside `ecg/holter/`: `arrhythmia_detector`,
`arrhythmia_engine/`, `decision_layer`, `diagnosis_hysteresis`,
`ecg_calculations`, `ecg_filters`, `enhanced_filter`, `clinical_measurements`,
`metrics/`, `pan_tompkins`, `qrs_detection`, `pre_analysis`,
`physiological_consistency`, `signal_paths`, `signal_quality`,
`template_matcher` and `lead_capability_matrix`. Utilities: `utils/app_paths`,
`utils/settings_manager`, `utils/input_validation` and `utils/crash_logger`.

### Session folder (`src/recordings/<date>_<time>_<name>/`)

```
recording.ecgh          raw 12-lead samples
recording_index.json    index into the .ecgh
session.json            metadata
patient.json            patient details (asked when the recording stops)
metrics.jsonl           per-chunk analysis
events.db               events
auto_segments_cache.json, ischemia_cache.json
manual_beats.json, manual_segments.json, deleted_auto_segments.json   user edits
```

Runtime settings and logs go to `%LOCALAPPDATA%\Deckmount\Comprehensive`
(override with `ECG_RUNTIME_DIR`).

---

## 5. Analysis workspace (`ecg/holter/holter_ui.py`, `panels/`)

`HolterMainWindow(parent, session_dir, patient_info, writer, live_source, duration_hours)`

* **Live**: when `writer` is set, it shows a status bar with a Stop button and
  refreshes the summary every second from the writer's live snapshot. It
  draws strips from `live_source.data`. Stop → `writer.stop()` → patient
  details dialog → loads the finished session.
* **Review**: `load_completed_session(session_dir)`.
* Tabs/panels: overview, full disclosure (`holter_full_disclosure.py`),
  replay, events and event editing, strip editing, templates, AF, ST,
  HRV, histogram, Lorenz, report table, recordings and record settings.

---

## 6. Lifecycle (`src/main.py`)

```
Connect & Start ─► DeviceAcquisition.start ─► packets ─► plotter
Start Recording ─► HolterStreamWriter.start + HolterAnalysisWorker.start
                 ─► HolterMainWindow(writer, live_source=plotter)
packets ─► plotter.push_packets  +  writer.push(fill_lead_off(packet))
Stop Recording  ─► HolterMainWindow._stop_recording  (writer.stop → patient dialog → review)
Device drops    ─► acquisition.stopped ─► recording stopped and saved
Previous Recordings ─► HolterMainWindow.load_completed_session(latest)
Quit            ─► asks if recording, writer.stop, STOP + close serial
```

---

## 7. Layout

```
ARCHITECTURE.md
config/clinical_config.yaml
assets/                   scrollbar arrows used by full disclosure
docs/                     algorithm notes (Holter / arrhythmia rules)
tests/                    test_vf_false_positive, test_input_validation
src/
  main.py                 app entry
  ecg/
    serial/               device protocol + packet parsing
    holter/               recording, analysis, workspace, live plotter
      panels/             workspace tabs
    (shared signal and arrhythmia modules)
  utils/                  paths, settings, validation, crash log
  recordings/             sessions
```

Tests: `cd src && python -m pytest ../tests`
