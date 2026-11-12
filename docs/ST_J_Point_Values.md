# ST: J point, J+20, J+40, J+60 values — simulator NSR (BPM-wise) and human recordings

Generated from the recorded data with the current ST code (`ecg/holter/ischemia_detection.py`, filter: 0.5 Hz high-pass, 3-point median, 25 Hz Gaussian once). Nothing in the code was changed for this.

## How the numbers are measured

- Recording read in 10 s frames. At low BPM (fewer than 4 beats in 10 s) the frame is widened to 30 s, then 60 s.
- Frame skipped (same rules as the ST code): fewer than 4 beats, irregular RR, beats of different shape (correlation < 0.9), QRS >= 120 ms.
- QRS onset and J point from the median beat (combined slope of all 12 leads, `_qrs_bounds`).
- Baseline = PR segment (40–10 ms before QRS onset). Each value is the ±5 ms average at that point minus the baseline, per beat; the frame value is the most frequent per-beat value (0.1 mm bins, as `_modal_st`).
- Units: screen graph-paper mm (`_SCREEN_ADC_PER_MV` = 1133 ADC per mV, so 1 mm ≈ 0.1 mV).
- **Code ST point** = what the code uses for ST Elevation: J+40, or J+30 when HR > 120.
- Code rule: ST Elevation when chest leads >= 2 mm (limb >= 1 mm) in 2 contiguous leads. Clinical (4th Universal Definition of MI), at the J point: V2–V3 >= 2.5 mm men < 40, >= 2.0 mm men >= 40, >= 1.5 mm women; other leads >= 1 mm.

## 1. Simulator — Normal sinus rhythm, BPM-wise

Frames used: sinus only (no rhythm label, or Sinus Bradycardia / Sinus Tachycardia / Bradycardia / Tachycardia) and no other label (Artifact, Unidentified, AFib, BBB, blocks ...) in the frame. From 175 BPM up the simulator sinus frames carry the label SVT (with no age set the detector calls any regular narrow rhythm above 180 bpm SVT); they are included and marked SVT. BPM grouped to the nearest 5. Value = median over the frames; J+40 also with min – max.

BPM found: 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 90, 100, 105, 110, 120, 125, 130, 140, 145, 150, 160, 170, 175, 180, 190, 200, 210, 220, 230, 240, 245, 250, 260, 270  (6605 frames from 73 recordings)

**Not in the recorded simulator data (no usable sinus frame at that BPM):** 10, 280, 290, 300

10 BPM: the only 10 bpm frame (2026-11-03_12-43-40, random_8) lies inside an Artifact (mains) segment, so it is left out. 280-300 BPM: no recording at that rate.

### V2

| BPM | frames | recs | J | J+20 | J+40 | J+60 | J+40 min – max | Code ST point | Detector label |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 10 | 3 | -0.31 | +0.03 | +0.04 | +0.04 | -0.02 – +0.07 | +0.04 (J+40) | sinus |
| 20 | 22 | 4 | -0.03 | +0.02 | +0.04 | +0.04 | -0.03 – +0.06 | +0.04 (J+40) | sinus |
| 25 | 3 | 2 | -0.17 | +0.04 | +0.03 | +0.06 | +0.03 – +0.05 | +0.03 (J+40) | sinus |
| 30 | 52 | 12 | -0.44 | +0.04 | +0.04 | +0.04 | -0.05 – +0.08 | +0.04 (J+40) | sinus |
| 35 | 11 | 2 | -0.03 | -0.03 | +0.04 | +0.05 | -0.03 – +0.06 | +0.04 (J+40) | sinus |
| 40 | 80 | 15 | -0.26 | +0.03 | +0.04 | +0.05 | -0.06 – +0.15 | +0.04 (J+40) | sinus |
| 45 | 2 | 2 | -0.27 | +0.00 | +0.04 | +0.04 | +0.04 – +0.05 | +0.04 (J+40) | sinus |
| 50 | 24 | 5 | -0.39 | +0.04 | +0.05 | +0.05 | -0.03 – +0.12 | +0.05 (J+40) | sinus |
| 55 | 2 | 1 | -0.04 | +0.00 | +0.05 | +0.05 | +0.05 – +0.06 | +0.05 (J+40) | sinus |
| 60 | 409 | 52 | -0.44 | +0.04 | +0.05 | +0.06 | -0.25 – +0.65 | +0.05 (J+40) | sinus |
| 70 | 5229 | 7 | -0.45 | -0.01 | -0.03 | -0.05 | -0.46 – +0.21 | -0.03 (J+40) | sinus |
| 75 | 13 | 3 | -0.34 | +0.04 | +0.06 | +0.15 | +0.04 – +0.14 | +0.06 (J+40) | sinus |
| 80 | 84 | 12 | -0.44 | +0.04 | +0.05 | +0.14 | -0.08 – +0.17 | +0.05 (J+40) | sinus |
| 90 | 19 | 4 | -0.06 | +0.04 | +0.05 | +0.15 | +0.00 – +0.08 | +0.05 (J+40) | sinus |
| 100 | 19 | 3 | -0.36 | +0.05 | +0.06 | +0.15 | +0.05 – +0.13 | +0.06 (J+40) | sinus |
| 105 | 5 | 2 | -0.36 | +0.05 | +0.07 | +0.16 | +0.05 – +0.08 | +0.07 (J+40) | sinus |
| 110 | 56 | 6 | -0.10 | +0.04 | +0.08 | +0.17 | +0.00 – +0.15 | +0.08 (J+40) | sinus |
| 120 | 79 | 12 | -0.14 | +0.04 | +0.08 | +0.17 | +0.04 – +0.24 | +0.07 (J+40) | sinus |
| 125 | 1 | 1 | -0.15 | +0.04 | +0.08 | +0.18 | +0.08 – +0.08 | +0.06 (J+30) | sinus |
| 130 | 34 | 9 | -0.35 | +0.05 | +0.14 | +0.24 | -0.02 – +0.27 | +0.06 (J+30) | sinus |
| 140 | 30 | 7 | -0.44 | +0.05 | +0.14 | +0.26 | +0.13 – +0.44 | +0.06 (J+30) | sinus |
| 145 | 1 | 1 | -0.66 | +0.13 | +0.32 | +0.66 | +0.32 – +0.32 | +0.16 (J+30) | sinus |
| 150 | 23 | 5 | -0.36 | +0.05 | +0.15 | +0.35 | +0.05 – +0.45 | +0.07 (J+30) | sinus |
| 160 | 93 | 10 | -0.24 | +0.05 | +0.15 | +0.34 | +0.14 – +0.24 | +0.13 (J+30) | sinus |
| 170 | 28 | 8 | -0.44 | +0.05 | +0.16 | +0.36 | +0.06 – +0.46 | +0.14 (J+30) | sinus |
| 175 | 2 | 2 | -0.40 | +0.02 | +0.19 | +0.40 | +0.14 – +0.23 | +0.09 (J+30) | SVT |
| 180 | 35 | 6 | -0.45 | +0.06 | +0.24 | +0.45 | +0.05 – +0.33 | +0.14 (J+30) | SVT |
| 190 | 28 | 7 | -0.05 | +0.06 | +0.25 | +0.55 | +0.05 – +0.26 | +0.15 (J+30) | SVT |
| 200 | 69 | 15 | -0.45 | +0.05 | +0.16 | +0.46 | -0.05 – +0.55 | +0.07 (J+30) | SVT |
| 210 | 35 | 8 | -0.46 | +0.00 | +0.23 | +0.55 | -0.03 – +0.54 | +0.13 (J+30) | SVT |
| 220 | 49 | 6 | -0.55 | -0.04 | +0.23 | +0.64 | +0.04 – +0.36 | +0.06 (J+30) | SVT |
| 230 | 11 | 2 | -0.34 | -0.06 | +0.24 | +0.74 | +0.17 – +0.35 | +0.06 (J+30) | SVT |
| 240 | 7 | 2 | -0.26 | -0.15 | +0.24 | +0.84 | +0.15 – +0.25 | +0.04 (J+30) | SVT |
| 245 | 2 | 1 | -0.70 | -0.15 | +0.16 | +0.85 | +0.16 – +0.16 | -0.04 (J+30) | SVT |
| 250 | 16 | 3 | -0.35 | -0.15 | +0.25 | +0.86 | +0.16 – +0.26 | +0.04 (J+30) | SVT |
| 260 | 11 | 2 | -0.44 | -0.16 | +0.26 | +0.86 | +0.25 – +0.26 | +0.04 (J+30) | SVT |
| 270 | 11 | 3 | -0.44 | -0.16 | +0.35 | +0.67 | +0.34 – +0.45 | +0.05 (J+30) | SVT |

### V3

| BPM | frames | recs | J | J+20 | J+40 | J+60 | J+40 min – max | Code ST point | Detector label |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 10 | 3 | -0.36 | -0.11 | -0.06 | -0.05 | -0.12 – -0.04 | -0.06 (J+40) | sinus |
| 20 | 22 | 4 | -0.06 | -0.08 | -0.06 | -0.05 | -0.14 – -0.03 | -0.06 (J+40) | sinus |
| 25 | 3 | 2 | -0.06 | -0.11 | -0.07 | -0.05 | -0.14 – -0.06 | -0.07 (J+40) | sinus |
| 30 | 52 | 12 | -0.44 | -0.06 | -0.06 | -0.05 | -0.15 – -0.03 | -0.06 (J+40) | sinus |
| 35 | 11 | 2 | -0.07 | -0.13 | -0.06 | -0.04 | -0.13 – -0.04 | -0.06 (J+40) | sinus |
| 40 | 80 | 15 | -0.27 | -0.07 | -0.06 | -0.04 | -0.24 – +0.07 | -0.06 (J+40) | sinus |
| 45 | 2 | 2 | -0.20 | -0.06 | -0.05 | +0.03 | -0.05 – -0.05 | -0.05 (J+40) | sinus |
| 50 | 24 | 5 | -0.43 | -0.06 | -0.05 | -0.03 | -0.08 – -0.03 | -0.05 (J+40) | sinus |
| 55 | 2 | 1 | -0.05 | -0.11 | -0.05 | +0.01 | -0.06 – -0.05 | -0.05 (J+40) | sinus |
| 60 | 409 | 52 | -0.44 | -0.06 | -0.05 | +0.04 | -0.36 – +0.55 | -0.05 (J+40) | sinus |
| 70 | 5229 | 7 | -0.26 | +0.15 | +0.05 | -0.04 | -0.28 – +0.31 | +0.05 (J+40) | sinus |
| 75 | 13 | 3 | -0.36 | -0.05 | +0.03 | +0.16 | -0.02 – +0.15 | +0.03 (J+40) | sinus |
| 80 | 84 | 12 | -0.43 | -0.05 | +0.03 | +0.07 | -0.15 – +0.26 | +0.03 (J+40) | sinus |
| 90 | 19 | 4 | -0.35 | -0.04 | +0.05 | +0.16 | +0.03 – +0.14 | +0.05 (J+40) | sinus |
| 100 | 19 | 3 | -0.35 | -0.04 | +0.05 | +0.17 | +0.04 – +0.06 | +0.05 (J+40) | sinus |
| 105 | 5 | 2 | -0.35 | -0.04 | +0.06 | +0.24 | +0.05 – +0.07 | +0.06 (J+40) | sinus |
| 110 | 56 | 6 | -0.06 | -0.04 | +0.07 | +0.26 | +0.04 – +0.15 | +0.07 (J+40) | sinus |
| 120 | 79 | 12 | -0.06 | -0.04 | +0.06 | +0.25 | +0.03 – +0.36 | +0.06 (J+40) | sinus |
| 125 | 1 | 1 | -0.05 | -0.03 | +0.06 | +0.27 | +0.06 – +0.06 | +0.05 (J+30) | sinus |
| 130 | 34 | 9 | -0.35 | +0.03 | +0.14 | +0.35 | +0.04 – +0.47 | +0.05 (J+30) | sinus |
| 140 | 30 | 7 | -0.35 | +0.04 | +0.15 | +0.44 | +0.14 – +1.05 | +0.06 (J+30) | sinus |
| 145 | 1 | 1 | -0.54 | +0.06 | +0.45 | +1.17 | +0.45 – +0.45 | +0.25 (J+30) | sinus |
| 150 | 23 | 5 | -0.35 | +0.04 | +0.18 | +0.55 | +0.05 – +0.46 | +0.07 (J+30) | sinus |
| 160 | 93 | 10 | -0.15 | +0.04 | +0.23 | +0.55 | +0.16 – +0.35 | +0.13 (J+30) | sinus |
| 170 | 28 | 8 | -0.36 | +0.05 | +0.26 | +0.66 | +0.05 – +0.74 | +0.15 (J+30) | sinus |
| 175 | 2 | 2 | -0.39 | +0.05 | +0.29 | +0.80 | +0.25 – +0.33 | +0.15 (J+30) | SVT |
| 180 | 35 | 6 | -0.36 | +0.06 | +0.34 | +0.77 | +0.15 – +0.56 | +0.16 (J+30) | SVT |
| 190 | 28 | 7 | -0.05 | +0.06 | +0.37 | +0.95 | +0.15 – +0.45 | +0.23 (J+30) | SVT |
| 200 | 69 | 15 | -0.45 | +0.04 | +0.35 | +0.86 | +0.13 – +1.05 | +0.15 (J+30) | SVT |
| 210 | 35 | 8 | -0.45 | +0.03 | +0.36 | +1.05 | +0.26 – +0.95 | +0.16 (J+30) | SVT |
| 220 | 49 | 6 | -0.54 | -0.04 | +0.44 | +1.16 | +0.16 – +0.95 | +0.15 (J+30) | SVT |
| 230 | 11 | 2 | -0.16 | -0.05 | +0.46 | +1.46 | +0.45 – +0.75 | +0.16 (J+30) | SVT |
| 240 | 7 | 2 | -0.24 | -0.07 | +0.55 | +1.75 | +0.45 – +0.64 | +0.16 (J+30) | SVT |
| 245 | 2 | 1 | -0.70 | -0.15 | +0.55 | +1.75 | +0.55 – +0.55 | +0.15 (J+30) | SVT |
| 250 | 16 | 3 | -0.40 | -0.14 | +0.64 | +1.84 | +0.54 – +0.74 | +0.16 (J+30) | SVT |
| 260 | 11 | 2 | -0.36 | -0.14 | +0.74 | +1.85 | +0.74 – +0.75 | +0.25 (J+30) | SVT |
| 270 | 11 | 3 | -0.35 | -0.13 | +0.85 | +1.54 | +0.84 – +1.04 | +0.34 (J+30) | SVT |

### II

| BPM | frames | recs | J | J+20 | J+40 | J+60 | J+40 min – max | Code ST point | Detector label |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 10 | 3 | -0.23 | -0.13 | -0.13 | -0.07 | -0.15 – -0.08 | -0.13 (J+40) | sinus |
| 20 | 22 | 4 | -0.14 | -0.13 | -0.13 | -0.07 | -0.16 – -0.06 | -0.13 (J+40) | sinus |
| 25 | 3 | 2 | -0.22 | -0.15 | -0.12 | -0.08 | -0.13 – -0.12 | -0.12 (J+40) | sinus |
| 30 | 52 | 12 | -0.26 | -0.13 | -0.12 | -0.07 | -0.26 – -0.04 | -0.12 (J+40) | sinus |
| 35 | 11 | 2 | -0.14 | -0.14 | -0.11 | -0.05 | -0.15 – -0.07 | -0.11 (J+40) | sinus |
| 40 | 80 | 15 | -0.16 | -0.14 | -0.13 | -0.06 | -0.23 – +0.05 | -0.13 (J+40) | sinus |
| 45 | 2 | 2 | -0.21 | -0.13 | -0.06 | -0.04 | -0.07 – -0.05 | -0.06 (J+40) | sinus |
| 50 | 24 | 5 | -0.24 | -0.12 | -0.08 | -0.05 | -0.13 – -0.06 | -0.08 (J+40) | sinus |
| 55 | 2 | 1 | -0.11 | -0.10 | -0.06 | -0.00 | -0.06 – -0.06 | -0.06 (J+40) | sinus |
| 60 | 409 | 52 | -0.25 | -0.12 | -0.06 | -0.02 | -0.34 – +0.57 | -0.06 (J+40) | sinus |
| 70 | 5229 | 7 | -0.25 | +0.07 | -0.03 | -0.08 | -0.11 – +0.04 | -0.03 (J+40) | sinus |
| 75 | 13 | 3 | -0.23 | -0.06 | -0.03 | +0.14 | -0.07 – +0.03 | -0.03 (J+40) | sinus |
| 80 | 84 | 12 | -0.24 | -0.06 | -0.04 | +0.06 | -0.24 – +0.04 | -0.04 (J+40) | sinus |
| 90 | 19 | 4 | -0.15 | -0.05 | +0.04 | +0.15 | -0.04 – +0.04 | +0.04 (J+40) | sinus |
| 100 | 19 | 3 | -0.17 | -0.05 | +0.04 | +0.16 | +0.03 – +0.05 | +0.04 (J+40) | sinus |
| 105 | 5 | 2 | -0.24 | -0.05 | +0.04 | +0.18 | +0.03 – +0.04 | +0.04 (J+40) | sinus |
| 110 | 56 | 6 | -0.07 | -0.05 | +0.06 | +0.26 | +0.04 – +0.15 | +0.06 (J+40) | sinus |
| 120 | 79 | 12 | -0.06 | -0.04 | +0.06 | +0.25 | +0.04 – +0.33 | +0.06 (J+40) | sinus |
| 125 | 1 | 1 | -0.07 | -0.05 | +0.05 | +0.26 | +0.05 – +0.05 | +0.03 (J+30) | sinus |
| 130 | 34 | 9 | -0.16 | -0.04 | +0.08 | +0.34 | +0.04 – +0.44 | +0.04 (J+30) | sinus |
| 140 | 30 | 7 | -0.16 | -0.03 | +0.14 | +0.44 | +0.14 – +0.85 | +0.05 (J+30) | sinus |
| 145 | 1 | 1 | -0.26 | +0.06 | +0.45 | +0.46 | +0.45 – +0.45 | +0.23 (J+30) | sinus |
| 150 | 23 | 5 | -0.16 | -0.02 | +0.17 | +0.55 | -0.05 – +0.36 | +0.06 (J+30) | sinus |
| 160 | 93 | 10 | -0.07 | +0.03 | +0.18 | +0.55 | +0.15 – +0.27 | +0.07 (J+30) | sinus |
| 170 | 28 | 8 | -0.16 | +0.04 | +0.25 | +0.66 | +0.13 – +0.66 | +0.13 (J+30) | sinus |
| 175 | 2 | 2 | -0.17 | +0.00 | +0.26 | +0.65 | +0.24 – +0.27 | +0.14 (J+30) | SVT |
| 180 | 35 | 6 | -0.17 | +0.05 | +0.33 | +0.83 | +0.15 – +0.56 | +0.15 (J+30) | SVT |
| 190 | 28 | 7 | -0.05 | +0.05 | +0.37 | +0.97 | +0.14 – +1.15 | +0.20 (J+30) | SVT |
| 200 | 69 | 15 | -0.25 | -0.04 | +0.33 | +0.86 | +0.13 – +0.95 | +0.13 (J+30) | SVT |
| 210 | 35 | 8 | -0.34 | -0.06 | +0.35 | +1.05 | -0.06 – +0.85 | +0.14 (J+30) | SVT |
| 220 | 49 | 6 | -0.36 | -0.13 | +0.35 | +1.15 | +0.15 – +0.84 | +0.06 (J+30) | SVT |
| 230 | 11 | 2 | -0.35 | -0.15 | +0.36 | +1.44 | +0.35 – +0.56 | +0.06 (J+30) | SVT |
| 240 | 7 | 2 | -0.45 | -0.25 | +0.44 | +1.65 | +0.34 – +0.46 | +0.05 (J+30) | SVT |
| 245 | 2 | 1 | -0.74 | -0.35 | +0.34 | +1.64 | +0.34 – +0.35 | -0.05 (J+30) | SVT |
| 250 | 16 | 3 | -0.55 | -0.26 | +0.45 | +1.80 | +0.35 – +0.55 | +0.04 (J+30) | SVT |
| 260 | 11 | 2 | -0.64 | -0.34 | +0.55 | +1.76 | +0.54 – +0.64 | +0.05 (J+30) | SVT |
| 270 | 11 | 3 | -0.65 | -0.34 | +0.75 | +1.45 | +0.64 – +0.86 | +0.14 (J+30) | SVT |

## 2. Human recordings

Every measured frame of the recording (all rhythms the code measures). Value = median over the frames, with min – max.

### 2026-10-28_17-36-16_Unknown — Sushil_test  (age/gender: 26 / Male)

Frames: 10, HR median 62 (range 60–67), QRS median 88 ms

| Lead | J | J+20 | J+40 | J+60 | Code ST point |
|---|---|---|---|---|---|
| V2 | +0.66 (+0.44 – +0.87) | +2.33 (+2.23 – +2.47) | +2.96 (+2.94 – +3.17) | +3.55 (+3.26 – +3.73) | +2.96 (+2.94 – +3.17) |
| V3 | +0.95 (+0.86 – +1.12) | +2.04 (+1.94 – +2.06) | +2.55 (+2.48 – +2.65) | +2.96 (+2.85 – +3.07) | +2.55 (+2.48 – +2.65) |
| II | +0.89 (+0.75 – +0.96) | +0.54 (+0.32 – +0.64) | +0.55 (+0.34 – +0.64) | +0.60 (+0.47 – +0.76) | +0.55 (+0.34 – +0.64) |

### 2026-10-28_17-39-47_Unknown — Sushil_kumar_25_mins  (age/gender: 24 / Male)

Frames: 151, HR median 59 (range 52–74), QRS median 88 ms

| Lead | J | J+20 | J+40 | J+60 | Code ST point |
|---|---|---|---|---|---|
| V2 | +0.68 (+0.44 – +1.04) | +2.37 (+2.15 – +2.63) | +3.06 (+2.68 – +3.37) | +3.55 (+3.25 – +3.86) | +3.06 (+2.68 – +3.37) |
| V3 | +0.96 (+0.77 – +1.17) | +1.86 (+1.64 – +2.16) | +2.35 (+2.14 – +2.74) | +2.76 (+2.45 – +3.14) | +2.35 (+2.14 – +2.74) |
| II | +0.95 (+0.75 – +1.13) | +0.46 (+0.25 – +0.69) | +0.45 (+0.25 – +0.69) | +0.56 (+0.34 – +0.87) | +0.45 (+0.25 – +0.69) |

### 2026-10-29_16-33-38_Unknown — Sushil  (age/gender: ? / Select)

Frames: 6, HR median 83 (range 73–85), QRS median 96 ms

| Lead | J | J+20 | J+40 | J+60 | Code ST point |
|---|---|---|---|---|---|
| V2 | +0.84 (+0.33 – +2.11) | +1.76 (+0.56 – +3.19) | +2.27 (+1.65 – +2.54) | +2.97 (+0.93 – +3.76) | +2.27 (+1.65 – +2.54) |
| V3 | +1.30 (+1.05 – +2.39) | +2.79 (+2.55 – +4.16) | +3.72 (+3.35 – +4.64) | +4.61 (+4.15 – +5.08) | +3.72 (+3.35 – +4.64) |
| II | +0.65 (-0.42 – +1.37) | +1.04 (+0.27 – +1.15) | +1.39 (+0.81 – +1.55) | +1.91 (+1.61 – +2.05) | +1.39 (+0.81 – +1.55) |

### 2026-10-29_16-36-56_Unknown — Sushil 28_mins  (age/gender: 24 / Male)

Frames: 92, HR median 86 (range 64–126), QRS median 94 ms

| Lead | J | J+20 | J+40 | J+60 | Code ST point |
|---|---|---|---|---|---|
| V2 | +0.26 (-1.12 – +2.07) | +1.40 (+0.02 – +2.54) | +1.85 (-1.71 – +3.45) | +2.36 (-0.03 – +4.03) | +1.85 (-1.71 – +3.45) |
| V3 | +1.13 (+0.14 – +1.86) | +2.65 (+1.37 – +4.44) | +3.45 (+1.85 – +6.21) | +4.34 (+2.86 – +8.95) | +3.45 (+1.85 – +6.21) |
| II | +0.64 (-1.06 – +1.48) | +0.96 (-0.49 – +1.64) | +1.35 (-0.38 – +5.99) | +1.76 (+0.33 – +7.33) | +1.35 (-0.43 – +5.99) |

### 2026-10-30_13-51-44_Unknown — sushil kumar test 35hz  (age/gender: ? / Select)

Frames: 103, HR median 77 (range 64–107), QRS median 86 ms

| Lead | J | J+20 | J+40 | J+60 | Code ST point |
|---|---|---|---|---|---|
| V2 | +0.54 (-1.24 – +3.48) | +2.15 (-0.25 – +4.69) | +2.74 (-0.27 – +6.15) | +3.26 (-2.85 – +5.09) | +2.74 (-0.27 – +6.15) |
| V3 | +0.92 (-3.16 – +1.73) | +1.75 (-5.14 – +2.84) | +2.24 (-9.97 – +2.98) | +2.67 (-2.55 – +3.77) | +2.24 (-9.97 – +2.98) |
| II | +0.65 (+0.13 – +2.75) | +0.26 (-2.45 – +4.94) | +0.35 (-0.45 – +2.04) | +0.55 (-1.54 – +2.74) | +0.35 (-0.45 – +2.04) |

### 2026-10-30_14-30-07_Unknown — sushil_20_mins  (age/gender: ? / Select)

Frames: 119, HR median 70 (range 61–97), QRS median 86 ms

| Lead | J | J+20 | J+40 | J+60 | Code ST point |
|---|---|---|---|---|---|
| V2 | +0.66 (-3.96 – +1.43) | +2.38 (-15.44 – +3.68) | +3.13 (-1.51 – +4.25) | +3.65 (-4.19 – +8.12) | +3.13 (-1.51 – +4.25) |
| V3 | +0.95 (-3.20 – +2.33) | +1.93 (-1.83 – +4.40) | +2.45 (-1.92 – +3.98) | +2.95 (-0.06 – +7.46) | +2.45 (-1.92 – +3.98) |
| II | +0.87 (+0.15 – +1.32) | +0.34 (-0.15 – +0.87) | +0.36 (+0.15 – +0.93) | +0.47 (+0.17 – +1.13) | +0.36 (+0.15 – +0.93) |

### 2026-10-30_14-51-22_Unknown — sushil_N_3_min  (age/gender: ? / Select)

Frames: 20, HR median 67 (range 63–79), QRS median 86 ms

| Lead | J | J+20 | J+40 | J+60 | Code ST point |
|---|---|---|---|---|---|
| V2 | +0.66 (+0.45 – +0.87) | +2.46 (+2.25 – +2.55) | +3.19 (+3.05 – +3.35) | +3.71 (+3.45 – +3.97) | +3.19 (+3.05 – +3.35) |
| V3 | +0.96 (+0.86 – +1.16) | +1.94 (+1.84 – +2.12) | +2.45 (+2.34 – +2.68) | +2.96 (+2.74 – +3.15) | +2.45 (+2.34 – +2.68) |
| II | +0.90 (+0.67 – +1.05) | +0.34 (+0.23 – +0.45) | +0.34 (+0.25 – +0.44) | +0.40 (+0.27 – +0.56) | +0.34 (+0.25 – +0.44) |

### 2026-10-30_14-55-20_Unknown — SUSHIL_LST  (age/gender: ? / Select)

Frames: 52, HR median 75 (range 63–112), QRS median 86 ms

| Lead | J | J+20 | J+40 | J+60 | Code ST point |
|---|---|---|---|---|---|
| V2 | +0.70 (-0.01 – +2.55) | +2.45 (+0.00 – +3.47) | +3.15 (-0.01 – +4.15) | +3.67 (-0.01 – +4.54) | +3.15 (-0.01 – +4.15) |
| V3 | +0.86 (-1.65 – +8.38) | +1.83 (-2.04 – +3.15) | +2.29 (-1.55 – +3.06) | +2.74 (-1.66 – +11.32) | +2.29 (-1.55 – +3.06) |
| II | +0.59 (-0.25 – +1.12) | +0.25 (-0.25 – +0.46) | +0.27 (-0.25 – +0.55) | +0.44 (-0.46 – +0.75) | +0.27 (-0.25 – +0.55) |

### 2026-10-30_12-01-08_Unknown — 1hr_priyam_sir_recording  (age/gender: ? / Select)

**Not usable for ST.** Only Lead II has a signal: Lead I and V1-V6 are completely flat (peak-to-peak 0 ADC, a single value), so III / aVR / aVL / aVF are copies of Lead II and the V2 / V3 values read 0.00. HR is exactly 60 for all 87 min.

## 3. Recordings used / left out

Simulator recordings used (frames):

- 2026-09-28_12-44-19_Unknown — 9_min_fluke (31)
- 2026-09-30_14-43-53_Unknown — 3_min_brady_tachy_test (16)
- 2026-09-30_16-01-22_Unknown — Brady_tachy_TT (20)
- 2026-10-05_17-14-16_Unknown — 5_min_Brady_tach_VFib (19)
- 2026-10-10_12-21-19_Unknown — 23_min_all_test (91)
- 2026-10-13_18-35-55_Unknown — 14_hrs_test (5165)
- 2026-10-14_15-20-19_Unknown — 5_min_Atach (30)
- 2026-10-15_10-55-56_Unknown — 17min_brady_tachy_Atrial_tachy (86)
- 2026-10-16_09-40-05_Unknown — 29_mins_all_BPM_Test (176)
- 2026-10-16_14-26-56_Unknown — 1min_test (7)
- 2026-10-17_14-33-14_Unknown — AFIB_test_TT (9)
- 2026-10-19_12-33-51_Unknown — filter_check_6 (5)
- 2026-10-20_12-26-09_Unknown — 30_min_all_test (118)
- 2026-10-21_14-39-51_Unknown — Conduction_NSR_Tachy_test (5)
- 2026-10-21_17-37-28_Unknown — Filter_check_BPM_test (132)
- 2026-10-21_18-28-06_Unknown — Test@@ (6)
- 2026-10-22_12-07-28_Unknown — baseline_check (7)
- 2026-10-22_12-13-13_Unknown — 20_min_all_test (29)
- 2026-10-23_11-21-01_Unknown — 7_min_multi_test (13)
- 2026-10-23_11-53-46_Unknown — 23_mins_random_rec (29)
- 2026-10-28_13-42-07_Unknown — Notch_check (3)
- 2026-10-28_13-46-44_Unknown — muscular test (10)
- 2026-10-28_14-21-58_Unknown — random_test_2 (19)
- 2026-10-29_11-51-32_Unknown — random test (3)
- 2026-10-29_11-59-11_Axis_Deviation_Demo — Axis_Deviation_Demo (42)
- 2026-10-29_12-43-51_Unknown — random test (3)
- 2026-10-29_15-05-05_Unknown — random_test3 (3)
- 2026-10-30_11-33-35_Unknown — random_new_test_Vfib (2)
- 2026-10-30_15-36-20_Unknown — filter_checknew1 (2)
- 2026-10-30_16-03-13_Unknown — Test_case TT (117)
- 2026-10-31_14-51-40_Unknown — aritfact 50 60 (9)
- 2026-10-31_14-57-58_Unknown — axis fluke (8)
- 2026-10-31_17-46-22_Unknown — tc1 (2)
- 2026-10-31_17-55-39_Unknown — hdhgfgjh (17)
- 2026-10-31_18-04-09_Unknown — TC3 (10)
- 2026-10-31_18-15-00_Unknown — lkjkljlkjk (13)
- 2026-11-02_09-57-22_Unknown — brady_artifact (15)
- 2026-11-02_10-57-19_Unknown — random_test_1 (1)
- 2026-11-02_11-21-36_Unknown — random test (3)
- 2026-11-02_12-58-44_Unknown — random_test4 (6)
- 2026-11-02_14-26-09_Unknown — random_test_5 (6)
- 2026-11-02_14-53-44_Unknown — random (11)
- 2026-11-02_15-00-59_Unknown — random_test_6 (12)
- 2026-11-02_15-29-41_Unknown — random artifact tachy test (3)
- 2026-11-02_15-45-18_Unknown — qrs detection test (2)
- 2026-11-02_17-23-21_Unknown — artifact brady tachy (3)
- 2026-11-02_17-32-07_Unknown — AFib Artifact (2)
- 2026-11-02_17-56-02_Unknown — Superventicular (1)
- 2026-11-03_10-25-10_Unknown — conduction_block 30 (3)
- 2026-11-03_11-24-57_Unknown — ventciular (4)
- 2026-11-03_12-19-03_Unknown — random_test_7 (13)
- 2026-11-03_12-43-40_Unknown — random_8 (1)
- 2026-11-03_13-34-27_Unknown — random_all_test (10)
- 2026-11-03_16-31-21_Unknown — artiNSR (1)
- 2026-11-03_16-46-53_Unknown — random10 (15)
- 2026-11-04_09-33-24_Unknown — random_all_test_new_without artifact (12)
- 2026-11-04_10-13-43_Unknown — performance (1)
- 2026-11-04_11-08-46_Unknown — random_11 (11)
- 2026-11-04_11-31-54_Unknown — artifact_50_60_2 (2)
- 2026-11-04_12-31-46_Unknown — amplitude test (39)
- 2026-11-04_13-51-13_Unknown — Cutie (9)
- 2026-11-04_14-14-07_Unknown — amp_2 (10)
- 2026-11-04_16-05-39_Unknown — amp_check_2 (26)
- 2026-11-04_16-23-00_Unknown — amp_check_2 (32)
- 2026-11-04_16-31-01_Unknown — amp_check_Supervent (15)
- 2026-11-04_16-39-09_Unknown — basline check 5 arti muscular (11)
- 2026-11-04_16-45-30_Unknown — qrs (8)
- 2026-11-04_16-49-27_Unknown — random_ahh_1 (1)
- 2026-11-04_17-31-21_Unknown — random_ytest (9)
- 2026-11-04_18-05-32_Unknown — arti_Arr_10 (31)
- 2026-11-05_10-49-07_Unknown — random_test_5 (9)
- 2026-11-05_12-16-28_Unknown — conduction_arti (2)
- 2026-11-05_12-31-16_Unknown — heinnnnn (8)

Left out:

- 2026-10-22_13-48-04_Unknown — ST Deviation recording: ST test recording (simulator ne ST badla hai)
- 2026-10-23_13-43-55_Unknown — ST_0.5: ST test recording (simulator ne ST badla hai)
- 2026-10-23_13-48-11_Unknown — 30_sec_0.5: ST test recording (simulator ne ST badla hai)
- 2026-10-23_13-54-51_Unknown — ST test all: ST test recording (simulator ne ST badla hai)
- 2026-10-23_16-59-52_Unknown — ST_+0.05mv_1min: ST test recording (simulator ne ST badla hai)
- 2026-10-23_17-06-40_Unknown — ST+0.10mv_1min: ST test recording (simulator ne ST badla hai)
- 2026-10-28_09-42-35_Unknown — ST_+0.02mV: ST test recording (simulator ne ST badla hai)
- 2026-10-28_14-49-28_Unknown — st_check22: ST test recording (simulator ne ST badla hai)
- 2026-10-28_14-53-03_Unknown — st_check_23: ST test recording (simulator ne ST badla hai)
- 2026-10-29_11-14-55_ST_Reciprocal_Demo — ST_Reciprocal_Demo: ST test recording (simulator ne ST badla hai)
- 2026-10-30_11-17-41_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-10-30_11-19-55_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-10-30_11-21-15_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-10-30_11-36-27_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-10-30_13-51-01_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-10-30_14-13-50_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-10-31_14-50-43_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-10-31_15-33-42_Unknown — fluke rec non arrthymia brady tachy types: simulator ST Depression in this recording (ST event 250-300 s)
- 2026-11-03_15-33-22_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-11-03_19-26-42_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-11-04_11-08-19_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-11-04_14-44-43_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-11-04_14-49-51_Unknown — (no name): naam nahi - human/simulator pata nahi
- 2026-10-25_09-00-00_Dummy_72h: 72 h (3.4 GB) dummy file, not run.

Simulator frames left out because the whole ST was shifted in that frame (|J+20| >= 0.8 mm in V2, V3 or II): ST set on the simulator, or noise in the frame (muscular test). Simulator NSR reads J+20 within ±0.35 mm at every rate; at 190-270 bpm only J+40 rises, because the T wave starts early, so those frames are kept:

- 2026-10-23_11-51-05_Unknown — nodal_test: 1 frame(s) from 20 s to 20 s
- 2026-10-28_13-46-44_Unknown — muscular test: 3 frame(s) from 0 s to 100 s
- 2026-10-31_17-17-11_Unknown — hghdbkj: 1 frame(s) from 0 s to 0 s
- 2026-11-02_12-58-44_Unknown — random_test4: 2 frame(s) from 60 s to 70 s
