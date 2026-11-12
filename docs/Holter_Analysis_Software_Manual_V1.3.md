# Holter Analysis Software — User Manual (H12 & H3 Series, V1.3)

> Source: `HQXY-CE-HLT1-14 Holter Operator's Manual of Holter Analysis Software V1.3-20211022.pdf`
> Manufacturer: Shenzhen ECGMAC Medical Electronics Co., Ltd. | Release: October 2021

This document is a structured Markdown transcription/summary of the full PDF manual, covering every section, table, and note in the source document.

---

## Table of Contents

1. [Product Introduction](#1-product-introduction)
2. [System Introduction](#2-system-introduction)
3. [Installation and Startup](#3-installation-and-startup-of-the-software)
4. [Manage Record](#4-manage-record)
5. [View](#5-view)
6. [Edit Tools](#6-edit-tools)
7. [Troubleshooting](#7-troubleshooting)
8. [Appendix A — Shortcut Buttons](#appendix-a-summary-of-software-short-cut-buttons)
9. [Appendix B — Utility Function Operation Guidelines](#appendix-b-operation-guidelines-of-utility-functions-of-software)
10. [Appendix C — Manufacturer Information](#appendix-c-manufacturer-information)
11. [Appendix D — European Authorized Representative](#appendix-d-european-authorized-representative-information)

---

## 1. Product Introduction

### 1.1 Introduction
Holter continuously records and analyzes long-term ECG of the heart in active/quiet states, used to evaluate arrhythmia, myocardial ischemia, and ST-segment changes. Results are only a diagnostic aid — final diagnosis is made by a doctor.

The product = **Holter recorder** + **Holter analysis software**. The analysis software processes:
- 12-lead ECG data (from 12-lead recorder)
- 3-lead ECG data (from 3-lead recorder)
- 3-lead and 12-lead records can coexist in the same software instance

Besides routine arrhythmia and ST-T analysis, the software supports: Afib Analysis, HRV (Heart Rate Variability), HRT (Heart Rate Turbulence), DC (Deceleration Capacity of Rate), TWA (T-wave Alternations), SAP (Sleep Apnea), and Lorenz Plot editing.

### 1.2 Software Features (full list)
1. Display/analysis of 12-lead and 3-lead ECG
2. ECG filters: AC filter, baseline drift filter, lowpass filter
3. Entire 12-lead waveform preview before analysis; QRS start/end and ST J-point settable
4. Analysis start/end time configurable; interference/artifact segments can be excluded (invalid areas)
5. Pre-analysis settings: Pause Duration, SVE Prematurity, SVE Tachycardia Rate threshold, Tachycardia Rate threshold, Bradycardia Rate threshold
6. Single-channel and multi-channel analysis; any channel selectable
7. **Template Editor**: 5 synced windows (template list, Lorenz Plots, Overlay beats, beat list, ECG strips); sortable by Start Time, RR interval, advance rate, similarity; supports reclassification, new-template creation, moving beats between templates
8. **Overlay Beat**: anti-confusion overlay (DEMIX); selectable channels/paper speed/gain; beat projection for batch separation of similar beats
9. **Time Scatter Plot**: 24h complete + time-sharing view, adjustable time window; color-coded beat types; batch update of beat types
10. **HR Graph**: 24h complete + time-sharing, adjustable window; batch update of beat types; auto-selection of Max/Min HR with auto strip insertion
11. **Atrial Fibrillation Editor**: automatic Afib analysis; segment editing via beat type updates
12. **Histogram Editor**: classified into common/VE/supraventricular/paced/others; 55 histogram types (S-R interval, V-R interval, pacing intervals, AF-AF interval, etc.)
13. **Lorenz Plots Editor**: 5 synced windows; filter by All/Normal/SVE/VE/Paced/Artifact/Others; artifact-adjacency toggle; 5 Overlay Beat windows
14. **ST-T Scanning**: 12-channel synchronous ST-segment/T-wave dynamic scan; auto-adjusts ST calc position by instant HR to reduce false positives
15. **Pacemaker Analysis**: auto-distinguishes atrial/ventricular/dual/uncategorized pace types; manual marking of oversensing, sensing failure, pacing dysfunction
16. Batch insertion of beats at equal intervals
17. Measuring tools: measuring ruler + parallel ruler
18. **Auto Strip**: auto-selects Max HR, V Run, Pause, etc. and inserts strips automatically
19. **Instant Print**: direct ECG segment printing with custom strip labels
20. Complete ECG printing: single-channel 24h printing, selectable channel/length per page
21. Custom beat shortcut buttons
22. **Sum. Interp.**: auto-generated summary interpretation with editable template
23. Report: custom title/content, preview support
24. HRV analysis (time + frequency domain, automatic)
25. HRT analysis
26. DC and DRs analysis
27. SAP (Sleep Apnea) analysis
28. TWA (T-wave Alternans) analysis

---

## 2. System Introduction

### 2.1 Software Version
| | Embedded software | Stand-alone software |
|---|---|---|
| Software name | Holter recorder software | Holter analysis software |
| Software model | SWHLT-A | DCECG-ACS |
| Release version | V1 | V1 |

### 2.2 System Composition
Consists of: record viewing, data management, system settings modules. Version V1, loaded on USB Flash Drive.

### 2.3 Software Performance
| Property | Description |
|---|---|
| Applicable objects | Holter ECG data |
| Max concurrency | 1 login/case |
| Data interface | USB |
| Hardware/software | Must be used with ECGMAC-manufactured recorder |
| Clinical functions | Medical record mgmt; entire waveform preview; ECG storage/playback; filter; HR measurement; beat editing; report editing/preview/print; ECG segment printing |
| Service restrictions | Max age: 8 chars; Gender: 7 chars; Name/Outpatient ID/Inpatient ID/Sickbed/Dept/Custom/ID: 20 chars; Search input: 32 chars; Max record length: 24 hours |
| User access control | Login via authorized username/password; dongle encryption required |
| Copyright protection | Dongle-based |
| UI | Graphical |
| Messages | Text prompt boxes |
| Reliability | Auto-backup to hard disk on error; restart recovers operation |
| Maintainability | Usage log / error report log available; updates by authorized personnel only |
| Efficiency | Recorder-computer comms ≤ 15s; login ≤ 3s |

---

## 3. Installation and Startup of the Software

### 3.1 Install Analysis Software
**Notes before installing:**
1. Insert the USB dongle before using the system
2. The computer with the analysis software is a dedicated instrument — no other use
3. Turn off antivirus/firewall before installing

**Steps:**
1. Insert USB dongle
2. Insert USB flash drive → double-click `InstallHolter.exe` (or Run as Administrator)
3. Confirm "Recorder Driver" component selection
4. Follow prompts to finish installation

> If driver install fails, contact ECGMAC after-sales.

### 3.2 Start the Analysis Software
> First-time use requires: dongle + account (**Admin**) + password (**123456**), all authorized by ECGMAC.

1. Power on computer and printer
2. Insert USB dongle
3. Right-click software icon → "Run as Administrator"
4. Enter login credentials → enters main interface

---

## 4. Manage Record

Supports both 3-Channel and 12-Channel recording; UI layout is consistent between them except where channel-specific waveform display differs.

### 4.1 Manage Record Introduction
Three parts: **Toolbar**, **Holter record list**, **Status bar**.

#### 4.1.1 Toolbar (Table 4-1)
| Group | Button | Function |
|---|---|---|
| Holter Record View | Browsing history | Shows Patient ID, Name, Age, Gender, Record Time, etc. |
| | Search record | Find records by Name/ID/etc. |
| | Analyze | Auto-analyze selected records (choose 3-lead combo) |
| | View | Enter record observation interface |
| Holter Record Manage | Initialize | Import patient info, sync recorder time |
| | Import | Import duplicated data into system |
| | Backup Record | Backup selected records |
| | Delete | Delete selected records |
| | Read | Import data from recorder |
| System | Settings | Analysis/Printing/Display settings, etc. |
| | About | Software version info |

#### 4.1.2 Record List
Columns: ID, Patient Name, Age, Gender, Record Time, Record Duration, Record Channels, Import Time, Status.

#### 4.1.3 Status Bar
- **Top**: Manage Record / View (current main interface)
- **Bottom**: hard disk storage status + case statistics
  - Green = sufficient space; **Red = insufficient space (import may fail)**

### 4.2 Manage Record Operations

#### 4.2.1 Holter Record View
- **4.2.1.1 Browse**: lists all records; click column header to sort. Right-click menu: View, Analyze, Backup, Delete, Export (archive file), Import.
- **4.2.1.2 Search**: enter keyword, check Search Options (Name/ID/Gender/Age/Channel/Conclusion/Physician/Status), optional "Only Pace Records" / record time range filter.
- **4.2.1.3 Analyse**:
  - Double-clicking an unanalyzed record prompts confirmation before entering Record Analysis.
  - Analyzing an already-analyzed record prompts Yes (re-enter analysis) / No (stay).
  - **Analysis Channel interface** has 4 main parts:
    1. **Complete Scatter Plot** — entire duration selection (blue box); right-click to mark effective (green)/invalid (white) areas
    2. **Hourly Scatter Plot** — corresponds to blue-box selection; right-click to set analysis start/end point
    3. **ECG Strips** — waveform around the blue positioning line
    4. **Tool Edit Bar** — channel selection, Default/Save As Default, Calibration, Filter, Analysis Settings, arm lead electrode reversal, Low voltage, Paced ECG, Analyze/Save/Cancel

  **Tool Edit Bar details:**
  | Operation | Description |
  |---|---|
  | Analysis Channel Selection | Pick any lead(s); same lead = single-lead; 2 different = 2-lead; 3 different = 3-lead analysis |
  | Default / Save As Default | Update/save current channel selection as default |
  | Calibration | Set I/J/K feature points for ST-segment scan |
  | Filter | For heavy interference; select filter → OK to apply (⚠ affects waveform quality — use carefully) |
  | Analysis Settings | Opens Setting Analysis interface |
  | Arm lead electrode reversal | Auto-correct left/right limb-lead reversal during analysis |
  | Low voltage | Detect low-amplitude info (for low-voltage limb/chest leads) |
  | Paced ECG | Detect pacing info during analysis |
  | Analyze | Runs analysis (shows Record Name, Duration, Analysis Time) |
  | Save / Cancel | Save → View interface; Cancel → back to Manage Record |

  **Notes:** Records can be re-analyzed multiple times before confirming; choose channels with highest amplitude/least interference; filters can degrade waveform quality.

- **4.2.1.4 View**: double-click analyzed record to enter View interface. Unanalyzed records prompt Analyze first.

#### 4.2.2 Holter Record Manage
- **4.2.2.1 Initialize**: connect recorder via USB → "Write Patient Information" (fields: Name, Gender, Age, ID, Outpatient ID, Inpatient ID, Bed, Department, Note, Paced OFF/LOW/HIGH). ⚠ **Initialize empties original recorder data — confirm import first.**
- **4.2.2.2 Read**: confirm (1) valid ECG records exist on recorder, (2) USB cable connected. Read button is grey until recorder is recognized. Data Acquisition dialog shows/edits Patient Info + Record Info (Record from, Duration, Channel type), **Paced** checkbox, and **Convert 12-lead into 3-lead** (12-lead recorders only; recommended when paired with 7-wire 3-lead setup).
  - Notes: data remains on recorder after Read; re-plug to re-read; don't operate other USB devices during Read (risk of interruption).
- **4.2.2.3 Import**: Import Record dialog — Backup Folder (changeable), Read button, Limit time range, Skip existed record, Remove backup record after import, Select All/None. Only software-backed-up records import successfully.
- **4.2.2.4 Backup Record**: Backup Folder, Limit time range, Overwrite Existing Record, Remove Record After Backup (⚠ choose carefully), Select All/None.
- **4.2.2.5 Delete**: via toolbar "Delete" or right-click.

#### 4.2.3 Settings
- **4.2.3.1 System**:
  - General: Record folder, Backup folder paths
  - Custom Settings: Doctor list (Add), Event Label (Add/Update), Record Info. Custom Fields (e.g. mobile/address), Common Terms (Add/Update/Delete/Move Up/Down) — two linked lists for quick conclusion editing
- **4.2.3.2 Display**:
  - 3-Channel / 12-Channel tabs
  - Channel Gain Adjust: 0, 2.5, 5, 10, 20, 40 mm/mV per lead; Invert checkbox
  - Full Disc. Settings: strip line width (duration), visible channels
  - Diag. Strip Display Mode: "3-Channel" or "Custom" grouping (Channel display settings dialog: select leads → name → Add; supports Remove/Update/Up/Down/Reset)
  - Show Instant HR, Enable Antialiasing, Reset
- **4.2.3.3 Auto Strip**: configures Number/Type/Sort By of auto-generated ECG strips (V Isolated, V Couplet, V Run, V Bigeminy/Trigeminy/Quadrigeminy; SV equivalents; Pause, Bradycardia, Tachycardia, ST Depression/Elevation, Longest RR interval, Max HR, Min HR). Valid range 0–100. Sort by Type or other. Clear/Default buttons.
- **4.2.3.4 Print Setting**: Allocation Plan (New/Rename/Delete print solutions); Pages to Print (Cover page, Hour tabular, Event strips, ST summary, Lorenz, HRV Time/Freq Domain, HRT, DC, ST Trend, T Trend, Rhythm Trend, Complete Strip); Other Settings (Cover Page Title, Print Beat Type, Print RRI, Print Instant HR, Print Page Number).
- **4.2.3.5 Analysis Settings** (Clinical setting): Allocation Plan (New/Rename/Delete); **Parameters for Arrhythmia Analysis** — Pause duration ≥ (1.0–5.0s), Atrial prematurity ≥ (10–100%), Atrial Tach. Rate ≥ (30–250bpm), ST Elevation ≥ (0.02–0.5mV), ST Depression ≥ (0.02–0.5mV), ST Duration ≥ (1–300s), ST Reset ≥ (1–300s), Tachycardia rate ≥ (20–250bpm), Bradycardia rate < (20–250bpm); **Advanced** (cycles, basic HR); **Parameters for Other Analysis** — AFib automatic analysis (Shortest duration ≥30s, Min mean rate 30–250bpm), HRT (TO <, TS >).
- **4.2.3.6 HRV Parameters**: Allocation plan; Frequency Domain (Analysis Mode FFT/AR, Min/Max RRI, VLF/LF/HF power band start-stop Hz); Time Domain (Min/Max RRI, Day/Night hour ranges).
- **4.2.3.7 Account**: "Enable User Authentication" checkbox; default user **Admin** / password **123456**.
- **4.2.3.8 Hotkey (Shortcut buttons)**: Allocation plan (New/Rename/Delete); default Hotkey Set (Normal=N, Atrial Premature=S, Ventricular=V, Artifact=X, Junctional=J, Atrial Escape=W, Junctional Escape=G, Ventricular Escape=E, Paced=P, BBB=B, Atrial Fibrillation=Q, Atrial Flutter=R, Fusion=F). None/Default buttons.
- **4.2.3.9 Other Settings**: Global settings —
  - Clear previous interpretation after reanalysing
  - Show R line in template
  - Correct Atrial Premature after artifact beat automatically
  - Don't change user mark templates when reanalyzing arrhythmia
  - Insert strip in event view directly
  - Tag template automatically
  - LORENZ shows all times by default
  - Enable aero effect
  - Maximize when launching
  - Show system tray
  - Default filter (AC filter, Baseline drift filter, Lowpass filter) — for Acquire Record
  - First view of record (e.g. RRGraph)
  - Default button

---

## 5. View

### 5.1 Record Basic Info.
#### 5.1.1 Patient Info.
| Field group | Contents | Modifiable |
|---|---|---|
| Patient basic info | Name, Gender, Age, Height, Weight, ID, Sickbed No., custom fields | Yes |
| Diag. Info | Notes, Medication, Conclusions | Yes |
| Record info | Record Date/Time, Import Date/Time (editable); Record Duration, Analysis Channels (not editable) | Mixed |
| Doctor | Shown | Yes |

#### 5.1.2 Save
Manual save; "Save" button activates only after info is updated/confirmed.

### 5.2 Template
7 template types: **ALL** (all except X), **N** (Normal), **S** (Atrial Premature), **V** (VE), **P** (Paced), **X** (Artifact), **Af/AF** (Atrial Fib/Flutter). Lit icon = record contains that beat type; grey = none.

**Software interface items:**
1. **Template list** — type, positioning line, beat count, Template ID, representative shape, confirm mark (○ unconfirmed / ● confirmed). Templates: solid="similar beats", "?"=dissimilar across channels, DEMIX=overlay-generated, "+"=merge/insert/reclassify/interval-reanalysis-generated, "N"=update-generated.
   - Right-click: Template properties (update all beats to N/S/Af/X/other), Function buttons (Delete/Select All/Reverse), Merge template, Template OK/Cancel, Superimposed wave analysis, Lorenz Plots.
2. **Lorenz Plots** panel — all scatter of selected template; drag-circle select; double-click select all. Right-click: Attribute, RR Interval scatter, RR diff scatter, Complete Scatter Plot, Time-sharing scatter plot, Zoom in/out, Pixel, Selected track, RR display range, beat attribute, Delete.
3. **Beat list** — select/edit/delete beats; numbers 1-5 classify into templates; sort by time/RR/advance rate/similarity. Right-click: Attribute, Function button, Similarity sorting, Re-classification, Create new template, Projection/Cancel projection, Select All/Reverse, length, Display channel, Isochronous insertion of batch beats (dialog: Insert object All/Selected, Insert to CT beat right/left, Insert method Beats number/RR interval, Beat type), Set As Max/Min HR.
4. **Overlay Beat** — right-click: Overlay channels, Gain, Paper speed, Update beat type, Delete, Select track, Interchange with template #1, Insert beat (note: can't insert within 200ms of overlay beat position), Move left/right, Return to original position.
5. **ECG strips** — shows strips for first-sequence beat; HR = 60/RR Interval. Right-click: Edit the beat, Set As Max/Min HR, Print/Add strips, Add Atrial Fibrillation segment, Interval reanalysis, Display channel, Set as starting/ending point, Batch insertion of beats.
6. **Bottom tool bar** — Measuring ruler, Parallel ruler, Set gain, Paper speed, Stripe length.
7. **Icons** — sort switch (time/RR interval/advance rate/similarity), CH (channel switch), refresh, new template, batch processing, beat positioning, display length switch, Select All/Reverse, contraction/expansion.

### 5.3 RR Graph
#### 5.3.1 Time Scatter Plot
Components: (1) entire RR scatter plot, (2) time-shared RR scatter, (3) scatter marking plot (Lorenz-style), (4) ECG strips, (5) ECG preview.
Right-click menu on entire/time-shared scatter: Edit the beat, Select all time, Delete, **Automatic atrial fibrillation analysis**, **Add Atrial Fibrillation segment**, Delete current/all Afib segment, Mark as entire atrial fibrillation, Check Afib events.

#### 5.3.2 HR Graph
Entire hourly HR Graph, Time-shared hourly HR Graph, ECG strips, ECG preview, Setting of Maximum HR (6 options), Setting of Minimum HR (6 options).

#### 5.3.3 Diag. Strip
Display modes: 3X1, 6X2, 12X1, Custom. "Return" button available when entered via double-click from beat list/ECG strip.

#### 5.3.4 Full Disclosure
Info bar: Screen Start Time, Strip Box Start Time, Channels, Strip Box Length, HR (avg in select box). Double-click to enter Diag. Strip. Tool bar: Gain settings.

#### 5.3.5 Histogram
Abscissa = type attribute size (RR interval/ratio/HR); ordinate = count (log or natural coordinates). Classification: **RR Interval**, **RR Ratio**, **Heart Rate**, grouped by Normal/Vent/SVE/Paced/Others.
- RR Interval types: N-N, N-S, N-V, S-N, S-S, S-V, V-N, V-S, V-V, R-R, S-R, V-R, RXR, R-S, R-V, N-J, N-JE, Paced-to-Paced/Intrinsic combos, AF-AF, Af-Af
- RR Ratio: equivalent ratio set (some with -1 suffix meaning positioned on previous beat)
- Heart Rate: Average HR, Sinus HR, VT HR, SVT HR, Paced HR, AF/Af Average HR

> Note: "-1" ratio types position on the previous beat (e.g., V-N Interval positions on V). "Intrinsic" = non-paced/non-artifact beats.

#### 5.3.6 ST Trend Chart
Entire HR Graph (scan start positioning), Trends (ST/T-wave scan results per channel; red=exceeds preset ST change; green=within range; T Trend green=upward/blue=downward), ECG strips (view-only). Tool bar of ST Scan: Start, Confirm, Default, "Update automatically during scanning", "Adjust K value according to heart rate".

#### 5.3.7 Pace Spike
Statistical diagram (column color = pace type; ordinate = spike count; abscissa = ms offset from QRS). Pace Types: SVE Pace Spike, VE Pace Spike, Dual, Uncategorized, Oversensing, Fail to Capture (user judgment required for sensing/capture failures). ECG strip + Beat list panels.

#### 5.3.8 Lorenz Plots
Entire scatter plot (blue box = time window), Lorenz Plots (drag-select beats), Tool bar classes: All, Normal, SVE, VE, Paced, Atrial Fibrillation/Flutter, Others, Artifact, No distinguishing of artifact (3 modes: no distinguishing / artifact related / non-artifact related). Beat List, ECG strip, Superimposed wave panels.

### 5.4 Rhythm Analysis
#### 5.4.1 Vent. Event
Classes: Ventricular Isolated, Bigeminy, Trigeminy, Quadrigeminy, Couplet, Runs (Tachycardia). Event list right-click: Modify, Remove, Remove events in list, Remove All, User event, Sort by Time Ascend/Descend, Select All, Reverse selection.

#### 5.4.2 SV Event
Same layout as Vent. Event. 12 categories: Atrial Isolated/Bigeminy/Trigeminy/Quadrigeminy/Couplet/Runs; Junctional Isolated/Bigeminy/Trigeminy/Quadrigeminy/Couplet/Runs.

#### 5.4.3 Pause Event
Classed by duration: 2.0–2.5S, 2.5–3.0S, 3.0+S. Right-click → Remove inserts an artifact mid-interval.

#### 5.4.4 HR Event
Collective of Bradycardia + Tachycardia. Cannot be edited/updated/deleted directly; shows Occurrence Time, Average HR, Duration.

#### 5.4.5 ST Event
Statistical chart: channel highlight (grey=no change, lit=has event); histogram of ST change amplitude vs. count. Event list: Event Type (ST Elevation/Depression), Start Time, Duration, check column. Right-click: Remove, Edit (Start Time, channel, Max Dep/Elevation range, Duration, ST Slope), Remove events in list/current channel/All, Reset, Select all/Invert.

#### 5.4.6 Afib Event
If no valid Afib mark: prompt to begin automatic Afib analysis (Cancel stays; OK opens Afib Analysis scatter interface). RR Interval scatter plot (blue segment = Afib event; orange = hover highlight). Event list: Index, Start Time, Duration, Type (Atrial Fibrillation/Flutter/Afib-Flutter), check column. Right-click: Edit, Delete, Set as Atrial Fibrillation, Set as Atrial Flutter.

#### 5.4.7 User Event
4 sources of user events: (1) events added in 5.4.1–5.4.6 interfaces, (2) Auto Strip settings, (3) long-press event button (3 lights) during recording, (4) Add Event/Instant Print. Right-click: Remove, Add to Print List, Edit.

### 5.5 Others
#### 5.5.1 Insert Beat
Available on any ECG strip interface except HRT/TWA/SAP. Drag-select blank time → Insert beat button → choose type/insertion mode (Automatic = avg HR before gap; Custom = set HR, etc.).

#### 5.5.2 Insert Afib. / Add Afib Segment
Select ≥3 beats → "Insert Afib" or right-click "Add Afib Segment". Click start/drag to end → "Insert Afib" button → edit start time/event type → OK. If overlapping with existing Afib: prompt to merge (Yes) or abandon (No). Inserted N/S beats become Af (waveform turns purple); line segment mark added to Afib Event scatter interface.

#### 5.5.3 Add Event/Instant Print
"Edit strip" dialog fields:
| Field | Options |
|---|---|
| Strip Start | fixed (not editable) |
| Strip Display Type | 7/14/21/35/42/56s (longitudinal); 10/20/30/40/50/60/70/80s (horizontal) |
| Strip Label | 32 default options + custom |
| HR | default = avg HR of selected beat, editable |
| Print Orientation | Portrait / Landscape |
| Preview channel | adds thumbnail if 12-ch selected |
| Print Len.(Sec.) | centered on or starting from positioning line |
| Print Channel | Single / Triangle(3) / All(12) |
| Buttons | Set as Default, Add Event, Cancel, Instant print, Export Pdf |

#### 5.5.4 Goto Time
Dialog shows Record Start/End Time, Record Duration (fixed), time progress bar + editable time field. OK jumps to time; Cancel retains current view.

#### 5.5.5 Advance Tools
7 advanced analysis tools: **HRV Time**, **HRV Freq.**, **HRT**, **DC**, **SAP**, **TWA**, **Lorenz Plots** (+ automatic atrial fibrillation analysis, found in time scatter plot interface). All support automatic analysis + manual editing.

> ⚠ Note: confirm all beats (incl. deleting artifacts) before using advance tools for accuracy.

##### 5.5.5.1 HRV Time (HRV-TD)
Event list types: Entire, Day, Night, Hour — columns: Start Time, Duration, Mean NN, SDNN, SDANN, Triangular Index, pNN50, Status.

Toolbar "Insert strip" dialog: list of inserted time segments (Start/End/Duration), record time info (fixed), settings (Start Time/End Time/Duration — editing End Time disables Duration & vice versa), buttons Insert/Delete/Update/OK/Cancel.

HRV Time N-N interval diagram: distribution of NN intervals; legend shows NNs, SDNN, rMSSD, Mean NN, SDANN, Triangular index, pNN50, SDNN Index.

**HRV Time Domain Parameters:**
| Index | Unit | Description |
|---|---|---|
| NNs | — | Number of normal NN intervals |
| Mean NN | ms | Mean NN interval (abnormal if day/night mutual deviation < 40ms) |
| SDNN | ms | Std dev of all NN intervals |
| SDANN | ms | Std dev of 5-min mean NN intervals |
| Triangular index | — | Total beats / histogram height (NN 1/128s bins) |
| pNN50 | ms | % of adjacent NN interval diffs > 50ms |
| SDNN Index | ms | Mean of 5-min NN std deviations |
| rMSSD | ms | Root mean square of adjacent NN interval diffs |

HRV Trends tab: time-series of SDNN, SDANN, SDNN Index, rMSSD, Triangular Index, pNN50.

##### 5.5.5.2 HRV Freq. (HRV-FD)
Event list types: Entire, Hour — columns: Start Time, Duration (≥5min), Mean NN, VLF, LF, HF, LFnu, HFnu, Status.

**HRV Frequency Domain Parameters:**
| Index | Unit | Description |
|---|---|---|
| RRs | — | Number of NN intervals |
| Mean NN | ms | Mean NN in segment |
| VLF | — | Very low frequency band power |
| LF | — | Low frequency power |
| HF | — | High frequency power |
| LFnu | — | Normalized LF power |
| HFnu | — | Normalized HF power |

##### 5.5.5.3 HRT
Auto-analyzes each PVC beat. Panels: RR interval change graph (abscissa = position relative to PVC, 0=event, -1=prior PVC, etc.; ordinate = RR interval with Min/Mean/Max), ECG strip, HRT results list.

| Parameter | Meaning |
|---|---|
| TO (turbulence onset) | Neutral = 0; TO<0 → deceleration after PVC; TO>0 → acceleration |
| TS (turbulence slope) | Neutral = 2.5 ms/RR; TS<2.5 → sinus deceleration; else no deceleration |

Event list columns: Index, Start Time, TO, TS, Status (normal/abnormal vs. reference), check column (√ = included in average). Also: Conclusion (manual), Re-analysis, Settings (TO/TS reference).

##### 5.5.5.4 DC
Auto-analyzed. Panels: heart rate acceleration/deceleration diagram (green=acceleration cycle, red=deceleration cycle), DC/AC risk results (Analysis button → DC Options dialog: Record Start/End Time, Analysis from/to, Analysis Duration, Cycles 20–200), DRs (Heart Rate Deceleration runs) analysis + risk judgment, Conclusion (manual editing needed).

##### 5.5.5.5 SAP
Auto-analyzes night-time beats only. Panels: entire RR scatter, RR interval fence diagram (10 min window), ECG strip, Event list + Conclusion (Automatic analysis / Remove / Clear events / auto-generated conclusion).

> ⚠ Note: SAP only analyzes RR interval at night; conclusion is reference only. Night time range set in Settings → HRV Parameters.

##### 5.5.5.6 TWA
Auto-analyzes T-wave of all lead beats. Includes Entire HR Graph, T-wave superimposition graph, power spectrum chart, T-wave amplitude line chart, TWA event list + editing tool bar.

> ⚠ Note: TWA results are reference only — combine with clinical trials/standards.

##### 5.5.5.7 Lorenz Plots (Advance Tools)
Event list (Entire + Hourly scatter results). 4 scatter plot panels: RR interval, NN interval, RR interval difference, NN interval difference (all vs. adjacent interval pairs).

#### 5.5.6 Record Settings
Loads Display, Print Settings, Analysis Settings, HRV Parameter settings — **effective only for the current record** (doesn't affect system-wide settings). Analysis parameter changes require reanalysis; HRV parameter changes take effect via "Reset" in HRV interface.

### 5.6 Report Print
#### 5.6.1 Summary (Sum. Interp.)
Auto-generated summary + user Conclusion.
- Template Editor (edit summary template)
- Enable Summary Template checkbox
- Reset (revert to standard template)
- Copy To Interp. (copy summary → conclusion)
- Conclusion: free text or right-click shortcut menu (categories: Sinus rhythm and arrhythmias, Other supraventricular rhythms, Ventricular arrhythmias, AV conduction, Intraventricular conduction, Changes in ventricular repolarization(ST-T,U), Cardiac pacemaker) + Doctor dropdown.

#### 5.6.2 Edit
- **5.6.2.1 Statistic Info.** — 9 categories: Abstract, Heart Rate, Bradycardia/Tachycardia, Pause, Ventricular ectopia beat, Supraventricular Ectopy Beat, Pace, Atrial Fibrillation/Flutter, Edit Report (unlocks editing of most fields; Afib stats always read-only; only saves if "Edit Report" stays checked).
- **5.6.2.2 Sum. Interp.** — same as 5.6.1.
- **5.6.2.3 Statistic Table** — hourly breakdown: Min/Avg/Max HR, V Isolateds/Couplets/Runs/Total, SVE Isolated/Couplets/Runs/Total. Editable via "Edit Report" checkbox.
- **5.6.2.4 ST Event** — Cover Page ST Channels (max 3, non-null), ST Depression/Elevation statistics per channel, ST Episodes Tabular (filterable by channel/type). Tool bar: Edit ST Event (Event time, Channel, Max Dep./Ele. Amplitude, Duration, ST episode slope Down/Horizontal/Up), Insert ST Event, Delete ST Event, Clear, Reset.
- **5.6.2.5 Trends** — ST event Trends, T Trend, rhythm abnormality chart across entire record.
- **5.6.2.6 Strip List** — all print-queued strips; supports Sort By Time, Insert Strip, Delete Strip, Disable Print, Move Up/Down, Show Interp.

#### 5.6.2 (second instance in source) Preview
Print tool bar: Previous/Next page, Zoom in/out, Export JPEG/PDF, Print all/Current/Odd/Even Page, Close.

**Cover Page contents:**
- Patient Info
- Heart rate stats (Beats, Max/Min/Avg HR + Start Time)
- Tachycardia/Bradycardia total duration + max beat count
- Total pauses + max pause info (RR > 2.0s)
- VE ectopia beat stats (isolated/couplets/coupled rhythm/V run freq/fastest & longest V-Run)
- SVT ectopia beat stats (equivalent set)
- HRV analysis: SDNN, SDANN, SDNN Index, rMSSD, TRIIDX, pNN50
- Atrial fibrillation analysis results
- ST event info for the 3 configured channels

> Note: when no specific event exists, cover page fields display as null. Install printer correctly before printing.

#### 5.6.3 Print Report
Click "Print Report" → Windows default print dialog → confirm to print.

---

## 6. Edit Tools

### 6.1 Beat Overlay Analysis
Purpose: remove dissimilar-morphology beats from a template's superimposed view for easier attribute judgment. Right-click a template → "Beat Overlay Analysis".
- Template #1 = default superposition of all beats; #2–#5 = classified subsets
- Drag-select (blue box) in #1 → press number key 1–5 to reclassify into that sub-template
- **OK**: regenerates templates (new ones marked DEMIX); **Cancel**: discards, reverts to original
- Beat List shows beats of the selected #1–#5 template

### 6.2 Re-classification
Purpose: secondary separation of a template based on the morphology of the currently selected center beat.
- Select beat(s) → right-click → Re-classification → beats reclassified per selected channel waveform
- Results: Create new template (marked DEMIX), Separate beat (enter 1–5), Edit beat directly

### 6.3 Instant Print
Select "Instant Print" → opens the Edit Strip dialog (see §5.5.3) → adds user event (saves fragment) and/or prints real-time ECG fragment.

### 6.4 Atrial Fibrillation Editing
Operations: Automatic Afib Analysis, Insert Afib, Mark Entire Afib, Delete Afib. Available on rhythm/icon analysis interfaces; full set (incl. auto-analysis/mark-all/delete) on time scatter plot.

Right-click menu on scatter/HR graph (requires ≥3 continuous beats for Add):
- **Add AFib Evt.** — multiple methods: drag-select on ECG strip → right-click Add on hourly scatter; shift+click continuous select; scroll+shift+click; Set as Starting/End Point then Add; right-button drag-select on scatter/HR graph then Add
- **Delete current AFib Evt.** — deletes segment at cursor's time point only
- **Delete all AFib Evt.** — deletes every Afib event in record
- **Mark all as an AFib Evt.** — marks entire record as one Afib event
- **Analysis AFib** (Automatic Atrial Fibrillation Analysis) — Analysis/Delete/OK (saves results, marks blue line segments)/Cancel buttons
- **View AFib Evt. information** — "Edit Atrial Fib/Flutter" dialog: event list (Index/Start/End/Type), Record start/end (fixed), Event start/end (editable), Event type dropdown, Delete/Update/OK/Cancel

> Insert Afib (from any rhythm/chart interface): select continuous beats on ECG strip → "Insert Afib" button → OK.

### 6.5 Edit Lorenz Plots
Rapid beat editing + leak-checking via scatter distribution.
Workflow: adjust blue selection box on entire plot → observe Lorenz Plot distribution → identify beats inconsistent with expected classification → right-click to select beat type to display/edit → cross-check beat list + ECG strip → edit.

### 6.6 Measuring Tools
#### 6.6.1 Measuring Ruler
1. Click "Measuring Ruler" button (status bar) to activate
2. Left-click start position on ECG strip → drag to end → measuring box shows amplitude + width
3. Close via button toggle or ESC

#### 6.6.2 Parallel Ruler
1. Click "Parallel Ruler" button to activate
2. Left-click start → drag to measure interval end → shows interval + heart rate
3. Close via button, ESC, or mouse click

### 6.7 Edition of Summary Template
Purpose: quick, customizable statistical conclusions matching institutional habits.
1. Edit → Summary → enter Summary interface
2. Check "Enable Template" → "Edit Summary Template" → opens editor
3. Right-click template list → New → name it (e.g. "Template 1")
4. Enter template text; use macros in format `$(macro_name)` — double-click a parameter-list row to insert its macro automatically (e.g. `$(Record_Length)`)
5. "Update Preview" to see rendered output
6. Right-click template → "Set as Default Template" → OK
7. Summary regenerates using the new default template

> ⚠ Macro call format must be exactly `$(macro_name)` or it won't resolve. Available macros include: HR_Avr, HR_Min, HR_Min_At, HR_Max, HR_Max_At, LG_RR, LG_RR_At, Total_Beat, Sinus_Max_HR, Sinus_Min_HR, Sinus_Avr_HR, Tachy_Threshold, Tachy_Count, Sinus_Tachy_Beat, Sinus_Tachy_Beat_Ratio, Tachy_Duration, Tachy_Duration_Ratio, LG_Tachy_Beats, LG_Tachy_At, Brady_Threshold, Brady_Count, BBB, BBB_Ratio, AVCB_Beats, AVCB_Beats_Ratio, EvtButton, Sinus_Beats, Sinus_Beats_Ratio, Artifact, Artifact_Ratio, SDNN, SDNN_Index, SDANN, rMSSD, pNN50, Triangle_Index, Record_Start_Date/Time, Record_End_Date/Time, Record_Length, Channels, Basilic_Rhythm, and more (SV_Total, SV_Isolated, V_Total, V_Isolated, V_Coupled, V_Bigeminy, V_Trigeminy, V_Quadrigeminy, SV_Coupled, SV_Bigeminy, SV_Tachy runs, etc.)

### 6.8 Paper Speed and Gain
#### 6.8.1 Paper Speed
Click "Paper Speed Adjustment" in the lower toolbar to cycle through speeds.

#### 6.8.2 Gain
Click "Paper Speed Adjustment"-adjacent gain control → dialog to set gain per lead or all leads (0, 2.5, 5, 10, 20, 40 mm/mV). Setting gain = 0 renders a flat line.

### 6.9 Interval Reanalysis
Right-click a selected time segment on ECG strip / time scatter plot / HR Graph → "Interval reanalysis" dialog:
- **Time settings**: Reanalysis Start / End
- **Parameter setting**: Analysis Channels (pick lead with least disturbance), Atrial prematurity ≥ % (10–100, defaults to record analysis settings), Low Voltage checkbox (improves accuracy for very low-voltage leads)
- OK / Cancel

---

## 7. Troubleshooting

| # | Failure | Cause(s) | Solution(s) |
|---|---|---|---|
| 1 | Printer doesn't respond to print command | Printer not set correctly; cable issue; driver not installed correctly | Match printer model in computer settings; reconnect cable; install correct driver |
| 2 | Many beats missed during analysis | Low analysis-lead amplitude; electrode leads off mid-recording | Use higher-gain lead; re-acquire, secure cable connection with tape |
| 3 | Many artifacts judged as beats | High T-wave voltage on analysis lead | Select analysis lead per actual situation |
| 4 | Error exporting medical cases | Insufficient disk capacity at export folder | Reselect export folder |
| 5 | Can't enter analysis software after double-click | Software deleted by mistake; virus infection; software already open elsewhere | Reinstall software; run antivirus check |
| 6 | EMG interference in recorded waveform | Loose leadwire; poor skin contact; nearby electrical interference; unclean skin prep; nervous patient | Keep skin dry, secure contact; tape cable interface; avoid interference sources; clean skin prep (hair removal, conductive paste); calm patient |
| 7 | No signal | Low battery; leads fall off; poor patient-cable/recorder contact; insufficient disk space on import folder | Replace battery; inspect leads; strengthen cable connection; replace import disk |
| 8 | Case cannot be deleted | Case is currently open | Close the observation record (double-click case) |
| 9 | No instant HR shown on ECG waveform | "Show Instant HR" not checked | Check it in Display settings |
| 10 | Incomplete print contents | Print settings not configured | Set required print content (beat Type, RRI, Home Page, Trends) |
| 11 | Patient info cannot be downloaded | Recorder not connected correctly; info already downloaded once | Ensure correct connection & wait for button to light; install battery, power recorder 1 min, re-plug, re-select Download |
| 12 | Entire-lead flatline | Gain set to 0; lead fall-off | Correct gain to non-zero |
| 13 | ARV import fails | Insufficient default storage disk capacity; file incomplete/damaged | Replace storage; re-obtain .arv file |

---

## Appendix A: Summary of Software Short-cut Buttons

| Shortcut | Function | Scope |
|---|---|---|
| Ctrl+A | Select All | Browsing history |
| Esc | Exit current interface | Analysis software (global) |
| Ctrl + left-click | Select multiple templates/beats | Template interface |
| Ctrl+A | Select all templates/beats | Template interface |
| Double-click template | Enter template-beat interface / Diag. Strip | Template interface |
| ← | Next template/beat | Template interface |
| → | Previous template/beat | Template interface |
| Delete | Delete template/beat | Template interface |
| N | Normal beat | View |
| S | Atrial premature | View |
| V | Ventricular premature | View |
| J | Junctional premature | View |
| P | Paced | View |
| C | Supraventricular escape | View |
| B | BBB | View |
| W | Atrial escape | View |
| F | Fusion | View |
| E | Ventricular escape | View |
| G | Junctional escape | View |
| Q | Atrial fibrillation | View |
| R | Atrial flutter | View |
| X | Artifact | View |
| Delete | Delete selected beat | View |
| Click + Shift + click | Continuous select (long interval) | View |
| Drag + click | Continuous select (long interval) | View |
| Left-click + attribute button/letter | Insert beat | View |
| Esc | Cancel excessive beats | View |
| Double-click ECG strip | Enter Diag. Strip | View |
| ← / → | Previous/Next objective or range | View |

> Note: letter shortcuts for beat attributes are customizable and may differ from the table default (see §4.2.3.8).

---

## Appendix B: Operation Guidelines of Utility Functions of Software

A condensed "how-to" index (operation → steps → result):

| Operation | Steps | Result |
|---|---|---|
| Initialize | Connect USB → click "Download Patient Information" → fill dialog → OK | Initializes recorder |
| Read | Connect USB → click "Read Recorder" → update patient info → OK | Reads recorder data |
| Search record | Click Search → choose attribute → enter keyword → Search | Finds record |
| Select beat/template | Left-click | Selects |
| Select multiple | Shift+click, or click+drag | Multi-select |
| Confirm template | Double-click enter/exit (if auto-mark on), or click Confirm Mark button | Marks template as reviewed |
| De-confirm template | Click Confirm Mark again | Cancels mark |
| Select all | Ctrl+A or Select-All button | Selects all |
| Update beat/template attribute | Letter key, or right-click → attribute | Updates type |
| Delete beat/template | Delete key, or right-click → Delete | Deletes (templates only deletable in artifact interface) |
| Insert single beat | Click blank position + letter key, or click + right-click attribute | Inserts beat |
| Insert multiple beats | Select position → right-click batch insert or "Insert beat" button → configure dialog | Batch insert (requires no existing heart rate in that span) |
| Superimposed wave analysis | Right-click template → Beat Overlay Analysis → drag-select → number key 1-4 → OK | Reassembles template via overlay |
| Re-classification | Right-click template-beat interface → Re-classification | Reassembles via classification |
| Enter Diag. Strip | Click Diag. Strip button, or double-click ECG | Opens Diag. Strip |
| Atrial fibrillation analysis | Right-click scatter plot → Automatic Afib Analysis | Runs Afib analysis |
| Adjust gain | Gain button, lower-right | Changes gain |
| Adjust paper speed | Paper Speed button, lower-right | Changes speed |
| Adjust ECG display time | Scroll mouse; select via ECG preview/scatter/HR Graph; "Goto Time" button | Jumps/scrolls view |
| Insert Afib event | "Insert Afib" button (multi-select), or right-click scatter → Add Atrial Fibrillation Segment | Inserts Afib |
| Measuring ruler | Click button or right-click → drag on ECG | Measures amplitude |
| Parallel ruler | Click button or right-click → click + drag | Measures interval/HR |
| Adjust display mode | Click 3/6:6/12/C buttons (Diag. Strip) | Switches layout |
| Adjust display channel | "Display Channel" button or right-click | Changes channels shown |
| Update current record settings | Select option in View interface → configure | Updates record-level settings |
| Reanalyze arrhythmia | Record Settings → analysis params → "Reanalyze Arrhythmia" | Reanalyzes |
| Save analysis results | Click Close or Save in View | Saves |
| Export record | Right-click → Export → choose folder | Exports archive |
| Import | "Import Record" button → select backup folder → Read → select → OK; or double-click exported record; or right-click → Import | Imports record |
| Backup Record | "Backup Record" button → folder → select records → OK | Backs up |
| Update Max/Min HR | Position waveform at time point → "Set as Max/Min HR" (HR Graph or right-click ECG strip) | Sets reference HR point |
| Add user event | Click confirm icon in rhythm analysis, or right-click → Add User Event, or long-press recorder button (3 lights), or "Add Event" dialog | Adds user event |
| Show Instant HR | Check "Show Instant HR" in display settings | Displays instant HR on waveform |
| Add doctor | Settings → Doctor → enter name → Add → OK | Adds doctor to list |
| Select doctor | Dropdown in Patient Info or Sum. Interp. Edit | Assigns doctor |
| Insert ST Event | "Insert ST Event" in ST event Edit interface | Adds ST event |
| Edit statistical data | Check "Edit Report" → edit → Save | Edits Statistic Info/Table |
| Add Custom Patient Info | Settings → System → custom information field → enter → OK | Adds custom field |
| Configure Full Disc. Channels | Full Disclosure Settings | Sets visible channels |
| Configure Diag. Strip Channels | Select 12-channel → configure 3-channel/custom display | Sets Diag. Strip layout |
| Basic heart rate settings | Analysis Settings → Advanced → check Custom → Auto or value | Sets basic HR reference |
| Print type of beat | Check "Print Type of beat" in Print Setting | Enables printing beat type |
| Print RRI | Check "Print RRI" in Print Setting | Enables printing RR interval |
| Print Instant HR | Check "Print Instant HR" in Print Setting | Enables printing instant HR |
| Replace record storage location | Edit default data storage folder in System settings | Changes storage path |

---

## Appendix C: Manufacturer Information

- **Manufacturer**: Shenzhen ECGMAC Medical Electronics Co., Ltd
- **Factory Address**: 2nd Floor of Block 2, Haoye Industrial Park, Tiegang Road, Xixiang Street, Baoan District, 518102 Shenzhen, China
- **Tel**: +86 755-27697821 / 27697823 / 27948579
- **Fax**: +86 755-27697823-616
- **Website**: www.ecgmac.com
- **Email**: info@ecgmac.com

## Appendix D: European Authorized Representative Information

- **Company**: Wellkang Ltd
- **Address**: Enterprise Hub, NW Business Complex, 1 Beraghmore Rd., Derry, BT48 8SE, Northern Ireland, UK
- **Tel**: +44(20)3287 6300, +44(33)3303 1126
- **Fax**: +44(20)76811874
- **Web**: www.wellkang.ltd.uk; www.CE-marking.eu
