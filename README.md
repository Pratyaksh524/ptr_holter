# Comprehensive ECG Analysis

Standalone desktop app for long-term (Holter-style) 12-lead ECG: live 12-lead
view from the ECG device, continuous recording, background arrhythmia / HRV /
ST analysis, review workspace and PDF report.

## Run

```
pip install -r requirements.txt
python src/main.py
```

1. Pick the COM port (or leave **Auto-detect**) → **Connect & Start**.
2. The 12 leads show live (3x4 or 6x2, 25 mm/s, 10 mm/mV).
3. **Start Recording** opens the analysis workspace. Stop from either window;
   patient details are asked at stop.
4. **Previous Recordings** reviews saved sessions in `src/recordings/`.

## Tests

```
cd src && python -m pytest ../tests
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the device protocol, packet format,
data flow and module map.
