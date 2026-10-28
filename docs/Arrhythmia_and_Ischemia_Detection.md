# Arrhythmia and Ischemia Detection

*As of 2026-10-28*

How the Holter software finds arrhythmias and ischemia findings in a recording, and how auto arrhythmia marking works in Full Disclosure, Overview, Replay and the report.

| Part | Module | Output |
| --- | --- | --- |
| A. Auto arrhythmia marking (how it works) | `holter_full_disclosure.py`, `auto_segment_arrthymia_detection.py`, `report_generator.py` | Coloured rhythm segments on the strips, rhythm label, report rows |
| B. Arrhythmia detection rules | `src/ecg/holter/auto_segment_arrthymia_detection.py` | Segments: label, start/end (beat-exact), colour |
| C. Ischemia detection (ST Elevation, ST Depression, T Inversion) | `src/ecg/holter/ischemia_detection.py` | Values shown in Overview, Replay and the report's ST ANALYSIS page |

**The two detections are independent.** Ischemia detection does not call the arrhythmia detection, does not read its segments, and is not affected by the Auto Arrhythmia Detection button. ST / T Inversion are values, not auto marks.

---

# Part A — Auto Arrhythmia Marking

## A1. Complete list of auto-marked labels

| Group | Labels |
| --- | --- |
| Rate (sinus) | Sinus Tachycardia, Sinus Bradycardia |
| Ventricular | Ventricular Tachycardia, Ventricular Fibrillation |
| Atrial | Atrial Fibrillation, Atrial Flutter, Supraventricular Tachycardia (SVT) |
| AV conduction | First-degree AV Block, Second-degree AV Block (Mobitz I), Second-degree AV Block (Mobitz II), Third-degree AV Block |
| Bundle branch (separate layer) | Complete Right Bundle Branch Block, Incomplete Right Bundle Branch Block, Complete Left Bundle Branch Block, Incomplete Left Bundle Branch Block |

Anything not marked is shown as normal rhythm.

## A2. What the user sees

1. **Before the button is clicked.**
   - Full Disclosure shows the plain waveform: no auto colour, no auto rhythm label (`_auto_marking_visible = False`).
   - The recording's own stored real-time labels are not shown either.
   - Manual beat and segment markings are always shown.
2. **Clicking "⚡ Auto Arrhythmia Detection"** (`_run_auto_arrhythmia_detection`).
   - `deleted_auto_segments.json` is removed, so every detected segment comes back.
   - **Cached result available:** it is applied immediately.
   - **No cache:** detection runs on a background thread (`detect_arrhythmias_async`). The button shows "⏳ Detecting..." and is disabled; when it finishes, `_on_auto_segments_ready` applies the result.
   - An undo snapshot named "Auto Arrhythmia Detection" is pushed.
   - The marking becomes visible, the flag file is written (A4), and the strips and segment overlay redraw.
3. **Reopening Full Disclosure** (`_restore_auto_marking_if_enabled`).
   - If the flag file exists, the cached result is re-applied, so the user doesn't click again.
   - Segments the user deleted stay deleted.
   - The restore never runs a fresh detection on open, so the UI is never blocked. If the cache was invalidated by a code change, the user clicks the button again.
4. **Undo of the button click.** It hides all auto marking again and deletes the flag file, so the report goes back to normal rhythm too. Undo of manual edits never touches this.
5. **Deleting an auto segment.**
   - It is saved in `deleted_auto_segments.json`.
   - The deletion is honoured on restore and in the report.
   - It is cleared when the button is run again.
6. **AF panel.** The AF panel applies the same detection to the replay engine when a recording loads (cached result, or in the background) to fill its table.

## A3. Merging into the replay engine — `apply_auto_segments_to_engine()`

The engine's `_structured_events` can already hold events from the recording's stored real-time pipeline. The merge works like this:

1. **Segments to events.** Detected segments are converted to region events (`timestamp`, `end_timestamp`, `label`, `color`, `source = 'Auto'`).
2. **Old entries dropped.** Before adding, these are removed:
   - this function's own output from an earlier call, so each call is idempotent and no stale wider "ghost" region survives;
   - every **non-manual** stored event whose label this detector owns. Labels are matched as substrings: fibrillation, vfib, afib, flutter, ventricular tachycardia, vtach, bradycardia, tachycardia, av block, bundle branch block.

   The real-time pipeline used an older scoring approach and widened or fragmented correct regions. Labels the detector doesn't classify are left alone.
3. **Rate vs morphology.** Sinus Tachycardia/Bradycardia regions are trimmed wherever an auto morphological region (VT, VFib, AFib, Flutter, SVT, any AV block) overlaps. Pieces < 1 s are dropped.
4. **Same-label merge.** Each label's timeline is merged on its own, joining regions ≤ 0.25 s apart. The merged region keeps the detector's `auto`/`arrhythmia` source, so the bottom-right rhythm indicator, which only reads those sources, finds it.

## A4. Files next to the recording

| File | Written by | Purpose |
| --- | --- | --- |
| `auto_segments_cache.json` | `detect_arrhythmias()` | Detection result. Key = recording path, size, mtime, plus a SHA-1 of every `.py` file under `src/ecg`. Any change to the recording or to detection code recomputes it. Also cached in memory for the process. |
| `auto_arrhythmia_marking_enabled.json` | Button click (`set_auto_marking_enabled`) | `{"enabled": true}`. Tells Full Disclosure (on reopen) and the report that the user turned marking on. Deleted on undo of the click. |
| `deleted_auto_segments.json` | Deleting an auto segment in Full Disclosure | Segments the user removed. Matched by > 0.5 s overlap (or start and end within 1 s) and label. |
| `ischemia_cache.json` | `compute_ischemia()` (Part C) | ST / T Inversion result. Not part of auto marking. |

One background job runs per recording. A second request while it is running waits for the same result.

## A5. Report

`_load_auto_segment_events()` in `report_generator.py`:
- **Flag not set:** returns nothing, and the report shows normal rhythm, the same as Full Disclosure before the click.
- **Flag set:** runs `detect_arrhythmias()` (normally answered from the cache), skips deleted segments and any ST labels, and adds each segment as an event with source `Auto`.
- **ST / T Inversion** have their own ST ANALYSIS page (Part C), whatever the flag says.

## A6. Detection pipeline at a glance

```
recording.ecgh
   │
   ├─ 1. Whole-recording R-peak pass (Pan-Tompkins, 60 s chunks, best lead)
   ├─ 2. Per-beat P/PR/QRS on Lead II (raw + 25 Hz low-passed)
   ├─ 3. Sinus Tachycardia / Bradycardia from the RR timeline + sinus gate
   ├─ 4. 10 s windows, 5 s step  ──►  SQI gate
   │        ├─ Bundle Branch Block (separate layer)
   │        └─ Rhythm chain, first match wins:
   │             VT → VFib → Flutter/AFib → SVT → AV Block (3rd → Mobitz II → Mobitz I → 1st)
   ├─ 5. Window cleanup (guards, drops, merge)
   ├─ 6. Beat-exact edge refinement
   ├─ 7. Priority / overlap resolution, rate segments trimmed
   └─ 8. Bundle Branch Block regions added last
   ▼
segments → auto_segments_cache.json → engine / report
```

---

# Part B — Arrhythmia Detection Rules

## B1. Pipeline details

1. **Whole-recording R-peak pass.**
   - Pan-Tompkins runs in 60 s chunks, with 3 s padding each side.
   - It uses the single best-amplitude lead from the first 30 s (candidates II, V5, V6, V4, V1, I).
   - The result is one continuous R-peak timeline. It feeds the rate rules and the edge refinement.
2. **Per-beat measurements.**
   - Every R peak is measured with `measure_beat(strict_pr=True)` on **Lead II** (the conduction lead).
   - Two versions are kept: raw (`rate_beat_metrics`) and after a 25 Hz zero-phase low-pass (`rate_gate_metrics`). The sinus gate reads the low-passed one, which matches the top-bar PR.
3. **Window classification.**
   - 10 s windows, 5 s step, 2 s of padding read on each side for detector settling.
   - R peaks are the union of Pan-Tompkins on the top 3 candidate leads, merged within 100 ms.
   - Only peaks inside the window's true bounds are judged.
   - Shorter windows (6 s) and a 1 s step were tried: they did not find onsets earlier and made the per-window vote unstable.
4. **Cleanup** (B12), **beat-exact edges** (B13), **priority resolution** (B12).

## B2. ECG components every rule uses (`measure_beat()`)

| Component | What it is | How it's found |
| --- | --- | --- |
| R peak | The QRS's dominant deflection | Pan-Tompkins |
| Q onset / J point | Start / end of QRS | `_qrs_bounds()`. Also returns `qrs_bounds_resolved`, which is True only when a genuine isoelectric boundary was found. A chaotic beat fails this. |
| QRS duration (`qrs_ms`) | J point − Q onset | < 40 ms or > 200 ms is flagged noisy and excluded from P/PR/T |
| P wave | Atrial depolarization | Searched 50–400 ms before Q onset. `strict_pr=True` takes the candidate closest to the QRS. |
| PR interval (`pr_ms`) | Q onset − P onset | Kept only in its valid range |
| T wave | Ventricular repolarization | Tangent method after the QRS (`t_peak`, `t_end`) |
| TP segment | Quiet baseline between T and the next P/QRS | Computed by each rule. Skip 280 ms (or 38% of RR) past the previous R, then up to this beat's P onset, or 150 ms before Q onset if there is no P. |

## B3. Signal quality gate

**SQI.**
- `SignalQuality.compute_sqi()` scores each window's own samples on the best lead.
- It uses the same scorer and **0.6 cutoff** as the real-time pipeline (`signal_quality_threshold` in `config/clinical_config.yaml`).
- Below 0.6, the window gets no label at all.
- Every morphological rule is judged off waveform shape, and noise (motion, a loose electrode, EMG) can look like VFib.

**Bandpass + notch.**
- `SignalPreprocessor` applies a 0.5–40 Hz bandpass and a mains notch.
- It runs inside the AFib/Flutter, SVT, AV Block and Bundle Branch Block checks.
- Without it, 50 Hz ripple read as Atrial Fibrillation, and QRS onset couldn't be found on noisy chest leads.

## B4. Sinus Tachycardia / Sinus Bradycardia

Decided from the whole-recording RR timeline, not the window vote.

| Rule | Start condition | Stay-in condition | Minimum |
| --- | --- | --- | --- |
| Sinus Tachycardia | 340 ms ≤ RR ≤ 600 ms (100–176 bpm) | RR ≤ 630 ms, if in a run AND the next RR is 340–600 ms | ≥ 3 beats, ≥ 1.0 s, sinus gate |
| Sinus Bradycardia | 1050 ms ≤ RR ≤ 6000 ms (10–57 bpm) | RR ≥ 1020 ms, if in a run AND the next RR is ≥ 1050 ms | ≥ 3 beats, ≥ 2.5 s, sinus gate |

- **60 bpm (RR 1000 ms) is normal sinus rhythm,** not bradycardia.
- **Why 340 ms.** P is suppressed when the previous QRS is within 333 ms. At ~180 bpm, RR jitters around that edge and the gate would fail.
- **Why 6000 ms.** A stretch with no detected peaks (a VFib onset) reads as one long RR and would become Bradycardia. 2000 ms was too low: real 30 bpm sinus sits at 2000 ms.
- **Hysteresis.** It stops one borderline beat from splitting an episode.

### Sinus-conduction gate (`sinus_conduction_gate`)

An RR pattern alone is not enough for a sinus label. Let *required* = 75% of the run's beats, with a minimum of 3. The run passes only when all of these hold:

1. At least *required* beats have:
   - a P wave with onset and end,
   - a resolved QRS,
   - QRS 40–200 ms.
2. At least *required* beats have a PR within 25 ms of the median PR. Runs over 30 beats use a rolling median of ±15 beats, so slow autonomic PR drift is allowed.
3. QRS duration std ≤ 20 ms.
4. P-P follows R-R: \|P-P − R-R\| ≤ max(120 ms, 15% of RR) on ≥ 75% of beats. This rejects noise bumps that don't track the QRS.
5. P polarity is consistent (rolling for long runs).
6. P duration is within 40 ms of the median (rolling for long runs).

### Bradycardia missed-beat recovery

**The problem.** Pan-Tompkins widens its refractory period to a third of the chunk's median RR. In a chunk dominated by a very slow rhythm, beats of a 60 bpm rhythm get dropped, and two 1000 ms RRs read as one 2000 ms RR.

**The fix.**
1. Every gap Bradycardia could count is re-detected on just that gap ±3 s.
2. Beats found there are added to Bradycardia's own beat list only.

## B5. Ventricular Tachycardia — `_is_ventricular_tachycardia_window()`

**Signature:** an organized, fast, regular, **wide**-complex rhythm without conducted P waves. Checked **first**, ahead of VFib: a regular monomorphic VT's wide QRS often doesn't "resolve" and could otherwise read as VFib.

**1. Stable core.** A window can straddle the VT onset, so only the dominant regular run is judged:
- The core is the longest run of RRs within ±15% of the window's median RR.
- **Fallback.** A window split about evenly between sinus and VT has its median in between, so no run fits. Only then, the longest run within ±15% of its own median is used, and it must be ≥ 6 RRs. Inside VFib, 3–4 RRs can match by chance.
- The core must have ≥ 3 RRs.

**2. Gates.** All must clear:

| Gate | Condition |
| --- | --- |
| Beats | ≥ 4 measured beats |
| Rate | Core rate ≥ 100 bpm |
| Regular | Core RR std / median ≤ 0.12 AND median absolute deviation / median ≤ 0.10 |
| Wide QRS | ≥ 75% (min 3) of core beats with QRS 120–220 ms |
| Uniform QRS | QRS std ≤ 25 ms |
| No conducted P | P with PR ≥ 100 ms on ≤ 50% of core beats, AND valid PR ≥ 100 ms on ≤ 50% |

A "P" less than 120 ms after the previous beat is that beat, not a P wave, and is ignored. At 160 bpm the P search otherwise reached back onto the previous VT complex.

**3. Drawn start.** The segment starts at the first beat of the stable core, not at the window edge. See also the VT start walk in B13.

## B6. Ventricular Fibrillation — `_is_vfib_window()`

Checked second.

**Regular-RR veto.** With ≥ 3 peaks, RR std / mean < 0.20 means **not** VFib. A regular rhythm can't be VFib.

**Main path.** For every beat:
- `p_follows_qrs` counts a beat only when all of these hold:
  - `p_present`,
  - `qrs_bounds_resolved`,
  - `p_onset < q_onset`.

  On a chaotic beat, any noise bump satisfies `p_onset < q_onset`.
- `organized_qrs` counts a beat when `qrs_bounds_resolved` is True.

```
no_p_follows_qrs = p_follows_qrs < beats / 2
no_organized_qrs = organized_qrs < beats / 2
VFib = no_p_follows_qrs AND no_organized_qrs
```

Both are needed together: noisy AFib can also show no clear P.

**Coarse-VFib path.**
- **The problem.** Coarse, slow VFib has big rounded waves whose sides flatten enough for the QRS boundary to "resolve" on half or more of them.
- **The rule.** Such a window is also VFib when **no** beat has a P before it, there are ≥ 5 beats, and the beat-shape correlation is < 0.95.
- **Beat-shape correlation** (`_aligned_beat_shape_corr`):
  - each beat is a 120 ms snippet,
  - each is aligned ±40 ms to the window's median template,
  - the value is the median correlation.
- **Measured values.** AFib and Flutter scored 0.97–1.00. The missed VFib scored 0.48–0.93.

**Bundle Branch Block veto.** VFib is not called where the same window is a proven conducted Bundle Branch Block (B11). VFib has no AV conduction.

At least 2 peaks are needed to judge at all.

## B7. Atrial Flutter / Atrial Fibrillation

Checked third. Both share `_prepare_atrial_window_context`. Flutter is tried first.

### Shared pre-condition and TP scoring

- **Beats:** ≥ 6 beats in the window.
- **Organized QRS:** ≥ 2/3 of beats resolved. This is what separates AFib/Flutter from VFib.
- **Quiet or active TP.** A beat's TP segment is **quiet** when its peak-to-peak is below `noise_threshold = max(0.05, 0.15 × median QRS amplitude)`. Otherwise it is **active**, and the waves inside it are counted.
- **Minimum sample:** ≥ 4 scored beats, or no verdict.

### Atrial Flutter

If quiet beats are < 40% of scored beats:

```
atrial_rate = waves in active TP stretches / their total time
ratio = atrial_rate / ventricular_rate
Atrial Flutter = ratio within 0.5 of 2, 3, 4 or 5
```

This is a fixed-ratio sawtooth read off the waveform. An FFT "flutter band" score was rejected: it was only QRS harmonics and fired on sinus recordings.

### Atrial Fibrillation

Checked if Flutter did not fire. Both gates must clear:

1. **Organized-P veto.** ≥ 60% of beats (min 3) with a PR AND std(PR) ≤ 25 ms is organized conduction, so the window is not AFib.
2. **Irregularly irregular.**
   - Take RR changes > 60 ms; at least 3 are needed.
   - ≥ 50% must reverse direction from the previous one.
   - Changes all in one direction are only a rate change.

### Regular P-train veto

- **The problem.** Mobitz I's grouped beating plus changing PR looks like AFib.
- **The rule.** An AFib/Flutter call is cancelled when the AV conduction context (B10) finds a regular train of discrete P waves and the QRS-resolved majority holds.
- **The train** must have:
  - ≥ 6 clear P waves,
  - at most one phase reset,
  - P hidden in a QRS or T wave on ≤ 50% of positions.
- **Why this can't be real AFib/Flutter.** AFib has no discrete P waves, and Flutter's atrial rate is 250–350/min.

## B8. Supraventricular Tachycardia (SVT) — `_is_atrial_tachycardia_window()`

Checked fourth. All gates must clear:

1. **Beats:** ≥ 6 beats.
2. **Rate:** mean rate **180–300 bpm**.
3. **Regular RR:** RR coefficient of variation ≤ 0.08. This rules out AFib and variable-conduction Flutter (CV 0.19–0.47).
4. **Organized QRS:** ≥ 80% of beats resolved.
5. **Uniform QRS:** median correlation to the window's QRS template (±50 ms around R) ≥ 0.9.

Once these pass, a sustained rate ≥ 180 bpm decides it. Resting sinus tachycardia that fast is not physiological for an adult Holter recording.

The P-on-T fused-wave test (≤ 1 wave between J point and next QRS in ≥ 70% of intervals) is still in the function, but it is only reached below 180 bpm, which gate 2 excludes. **In practice it does not decide anything.**

## B9. (Order reminder)

The rhythm chain per window: **VT → VFib → Flutter/AFib → SVT → AV Block**. The first match wins. AV blocks are tried in severity order: Third-degree → Mobitz II → Mobitz I → First-degree.

## B10. AV Blocks

Checked fifth, only when VT, VFib, AFib/Flutter and SVT found nothing. Each rule returns the exact beats its evidence spans (for example one Wenckebach cycle), not the whole window.

### Shared conduction context — `_prepare_av_conduction_context()`

P waves are found on **Lead II** (bandpass + notch). `measure_beat()` alone can't see a **blocked** P: it only looks 50–400 ms before each QRS.

1. **Organized ventricles.**
   - ≥ 80% of QRS resolved, OR
   - one QRS shape (median correlation ≥ 0.9).

   The shape fallback is there because wide escape beats and Bundle Branch Blocks often don't resolve.
2. **P template.**
   - Built from P waves `measure_beat()` found, excluding any "P" inside a QRS zone.
   - Seeds with correlation ≥ 0.7 to their median are kept, and at least 2 are needed.
3. **Scan.**
   - The template is cross-correlated along the whole window.
   - A hit needs correlation ≥ 0.6 and amplitude 0.5–2× the template.
   - It must not be inside a QRS zone.
   - Outside a T wave it needs correlation ≥ 0.8. Inside a T wave a weaker match is accepted, but only where a P is due.
   - At least 3 clear P waves are needed.
4. **P-P interval.**
   - The median of the shortest common spacing (≤ 1.3 × the 25th percentile).
   - It must be 300–2000 ms.
   - Tolerance = max(60 ms, 12%).
5. **Chain walk.**
   - Step one P-P at a time in both directions from each clear P.
   - A P due inside a QRS or T wave is marked **hidden**.
   - A clean baseline where a P was due is a **mismatch**.
   - The chain that accounts for the most clear P waves wins. This rejects a chain of T waves.
6. **`atrial_regular`** is True when all of these hold:
   - no mismatch,
   - no unexplained extra P,
   - P-P CV ≤ 0.12,
   - hidden ≤ 50%.
7. **Conduction.**
   - A P conducts to the first QRS starting 80–450 ms after its onset and before the next P.
   - PR is measured from P peak to R peak, shifted by the window's typical P-onset→P-peak and Q-onset→R-peak distances, so Q-onset jitter doesn't read as changing PR.

### First-degree AV Block

Every P conducts, just slowly:

1. `atrial_regular`, every P visible and conducted (1:1).
2. ≥ 5 QRS, **HR < 100 bpm** (above that, P sits in the T wave).
3. RR CV ≤ 0.15, no RR ≥ 1.5 × median (no dropped beat).
4. PR measured on ≥ 80% of QRS (min 4).
5. **Median PR > 200 ms and ≤ 450 ms**, > 200 ms on ≥ 80% of beats.
6. **PR constant:** std ≤ 25 ms.

### Second-degree AV Block, Mobitz I (Wenckebach)

PR lengthens until a P is not conducted, then resets.

**Gates:**
1. `atrial_regular`, at least one **visible** blocked P, never two in a row, conducted ≥ blocked.
2. Before the blocked P: ≥ 2 conducted beats, PR never shortening by more than 10 ms, **growing ≥ 40 ms in total**.
3. After it: the first conducted PR is **≥ 40 ms shorter** (the reset).
4. The pause around the blocked P is < 2 × P-P.

**Long-PR exception.** For this rule only, a P whose QRS follows 450–600 ms later (still before the next P) counts as conducted. The PR right before the drop can pass 450 ms; one example cycle was 146 → 206 → 386 → 456 ms.

**Cycle handling.** A complete cycle that fails these gates vetoes the window. A cycle cut off by the window edge is not judged.

### Second-degree AV Block, Mobitz II

PR fixed, then a P is suddenly not conducted.

1. `atrial_regular`, a visible blocked P, never two in a row.
2. PR of conducted beats before **and** after the blocked P all **within 20 ms**.
3. The pause is **1.7–2.3 × P-P**.

**2:1 block** (one conducted beat between blocked P waves) is reported only when the fixed-PR pattern repeats (≥ 2 cycles in the window).

### Third-degree (complete) AV Block

Atria and ventricles beat independently.

1. A regular atrial rhythm with **≥ 4 visible P waves**.
   - This uses the complete-block variant of `atrial_regular`.
   - It forgives one drift mismatch right after a P hidden in a QRS.
2. ≥ 3 QRS, **ventricular rate < 60 bpm**, RR CV ≤ 0.12 (a regular escape rhythm).
3. **Atrial rate ≥ 1.3 × ventricular rate.**
4. **AV dissociation:** the time from the nearest P to each QRS has **std > 40 ms**. In 2:1 block it stays fixed.

Mobitz I can also scatter that time, but its ventricular rhythm is grouped, so gate 2 keeps it out.

## B11. Bundle Branch Blocks (separate layer)

A Bundle Branch Block is a QRS-shape finding that sits **on top of** the rhythm (sinus, bradycardia, tachycardia, an AV block). These rules:
- run on every non-noisy window on their own,
- never enter the rhythm chain,
- never trim or get trimmed by a rhythm segment,
- never suppress Sinus Bradycardia/Tachycardia.

The only exception: no Bundle Branch Block is drawn over VT or VFib.

### Shared gate — `_bundle_branch_block_window_beats()`

All leads are bandpass + notch filtered first.

**1. Beats.** ≥ 5 beats. Let *required* = 75% of beats, with a minimum of 3.

**2. Conducted rhythm (P / PR on Lead II).**
- Each beat is measured at Lead II's own dominant QRS deflection (largest excursion −40..+100 ms from the detected beat). In a wide QRS the merged R peak can sit near the QRS onset.
- The strict P search runs first. If it gives no PR and RR ≥ 500 ms, the relaxed (most prominent wave) search is used.
- P must be present and PR measured on ≥ *required* beats.
- ≥ *required* PRs must be within 25 ms of the median. This rules out VT, VFib, AFib and escape rhythms.

**3. One QRS shape.**
- Median correlation ≥ 0.9.
- Each ±50 ms snippet is centred on the best lead's dominant deflection, since merged beat positions jump ~20 ms in a wide QRS.

**4. Global QRS width.**
- Combined slope of I, II, aVL, V1, V2, V5, V6, each scaled by its own QRS size.
- Onset and end are where the slope stays under **10% of its peak for 16 ms**.
- If no end is found (LBBB's discordant ST-T), **20%** is used for the end.
- The window width is the median over beats.

**5. Per-beat shape.** Each lead's baseline is its PR segment.

| Lead | Right pattern | Left pattern |
| --- | --- | --- |
| V1 (or V2) | rsR' (two positive waves ≥ 30 ms apart with a dip) or terminal positive R' | Dominantly negative (negative area ≥ 2× positive), no rsR' |
| I and V6 (or V5) | Wide slurred S: ≥ 15% of QRS size, ≥ 40 ms below baseline after R | Broad R ≥ 60 ms, positive ≥ 3× negative, no septal q, no S |

- A **Right** beat needs the V1 pattern AND a wide S in at least one lateral lead.
- A **Left** beat needs the V1 pattern AND a broad R in every lateral lead.

### Labels (checked in this order)

| Label | QRS width | Shape |
| --- | --- | --- |
| Complete Left Bundle Branch Block | ≥ 120 ms | Left on ≥ 75% of beats |
| Complete Right Bundle Branch Block | ≥ 120 ms | Right on ≥ 75% of beats |
| Incomplete Left Bundle Branch Block | 110–119 ms | Left on ≥ 75% of beats |
| Incomplete Right Bundle Branch Block | 110–119 ms | Right on ≥ 75% of beats |

The span runs from the first to the last beat with that pattern, not the whole window.

### After the window loop

1. **Merge.** Same-label regions merge when ≤ 2.5 s apart.
2. **Beat-level extension.**
   - Every judged beat keeps its votes.
   - A region's edges move outward one beat at a time while the next beat's majority vote is the same pattern.
   - Limits: gap ≤ 1.5 × RR, at most one window, never into another Bundle Branch Block region.
3. **One-beat gap closed.** Where two Bundle Branch Block regions meet beat to beat (Right → Left), the earlier one ends where the next starts.
4. **Priority.** Complete outranks incomplete where they overlap.
5. **Not over ventricular rhythms.** Nothing is drawn over VT/VFib.
6. **VFib is cut off any proven Bundle Branch Block span.**
   - Pieces < 1 s are dropped.
   - A piece touching the Bundle Branch Block and shorter than 5 s is the transition and is dropped.

## B12. Window cleanup and priority

Applied after all windows have voted:

- **VFib neighbour guard.**
  - Applies to a window called VFib only by the coarse-VFib path, or on fewer than 4 measured beats.
  - It is dropped unless another VFib window starts within one window of it. One lone window at the recording start or end with cut-off beats is not VFib.
- **AV Block next to AFib/Flutter dropped.**
  - Any AV Block window within 2 windows of an AFib/Flutter window is removed. The atrial window counts only if a same-label window is nearby.
  - Flutter's changing ratio reads as dropped beats.
  - **First-degree is exempt.** Its 1:1 constant-PR rule can't pass on Flutter/AFib.
- **SVT next to AFib/Flutter dropped,** only at a 2:1-Flutter ventricular rate (125–175 bpm) or unknown rate.
- **AV Block next to VFib dropped** (within one window). Conduction can't be graded on chaotic beats.
- **Isolated-window guard.** A single AFib, Flutter or SVT window with no same-label window within 2 windows is dropped. This catches the VFib "on-ramp".
- **Merge.**
  - Same-label windows are joined, bridging gaps up to 2 windows.
  - AV blocks bridge only up to 2.5 s, since their spans are beat-exact.
- **AV priority.**
  - Every non-AV rhythm outranks any AV block.
  - Among AV blocks: Third > Mobitz II > Mobitz I > First.
  - A lower one is cut where it overlaps a higher one. It may split into two pieces, and it is never stretched across a gap.
- **Overlap and gap.**
  - Where two different-label regions overlap, the earlier is trimmed to where the next starts.
  - A gap up to one window between two non-AV regions is bridged.
- **Morphology over rate.**
  - Every morphological rhythm overrides Sinus Tachycardia/Bradycardia. For example, 55 bpm with PR 280 ms is First-degree AV Block only, never both.
  - Rate segments are trimmed at the exact edge.
  - A piece within 5 s of a morphological edge is dropped as a transition blip.
- **Final merge.** Same-label segments ≤ 0.5 s apart are joined.

## B13. Beat-exact edge refinement

The window vote places an edge only to within 5 s. Each edge is then moved to the exact beat where the waveform changes.

### Ventricular Fibrillation

- **Start:** the first of two consecutive **disorganized** beats.
- **End:** the first of two consecutive **organized** beats, ≤ 2 s apart.
- **QRS < 40 ms** is never organized.
- **`require_narrow`.**
  - Next to normal rhythm, an organized beat also needs QRS < 120 ms.
  - This is not applied next to AFib/Flutter.
- **Silent gap.**
  - If the organized beat before a gap with no detected peaks is followed by nothing for > max(2 s, 1.5 × recent RR), VFib starts right after that beat.
  - A lone beat with silent gaps on both sides is skipped.
- **Silence fallback.** No beat at all between the last organized beat and the window boundary, for > max(2 s, 1.5 × RR), means VFib starts at that beat.
- **VT → VFib.**
  - VFib's start is not moved before the last beat that continues the VT run.
  - A beat continues the run when both hold:
    - its RR is within 15% of the VT's RR,
    - its QRS correlates ≥ 0.90 with the VT's median beat, at about the same size.
  - One extra detection inside the run is skipped.
- **Neighbours.** The neighbouring regions' edges are pulled in to meet the refined VFib edge.

### Ventricular Tachycardia start — `_vt_run_first_beat()`

1. The VT's first beats give a QRS template and RR. If there are < 5 beats, they are re-detected on that window.
2. Go backward one RR at a time (±15%) and search the **signal** for the template.
3. A match needs correlation ≥ 0.90 and a size 0.5–2× the template; the size check rejects T waves.
4. The first miss ends the walk.

Limits: at most one window back, and never past the previous region's end. This exists because the onset window can have no detected peaks at all.

### Atrial Fibrillation / Flutter vs. normal rhythm

- **The edge.** The first of two consecutive beats with a **quiet** TP segment (< 0.3× the noise threshold), searched from inside the episode outward.
- **End fallback** — `_regular_run_first_beat()`.
  - At a fast following rhythm there is no TP to measure.
  - If the next window's beats are regular (CV ≤ 0.12), go backward while both hold:
    - RR within 15% of the run's median,
    - QRS correlation ≥ 0.90 at about the same size.
  - AFib/Flutter ends at the run's first beat.

### Atrial Fibrillation ↔ Atrial Flutter

- Each RR interval's autocorrelation is computed. Flutter gives a peak ≥ 0.4 at a 170–250 ms lag; AFib does not.
- The edge moves to the first of two consecutive intervals of the new label.

### SVT

- **Edges.** The first beat opening two consecutive RRs within ±15% of the median RR, and the last beat closing two such RRs.
- **Local median.** The median comes from the beats next to each edge, so a climbing rate doesn't reject the opening beats.
- **SVT range.** Each RR must be ≤ 340 ms.

### SVT ↔ AFib/Flutter

Where they overlap, or have a gap up to one window, the SVT's beat-exact edge wins and the AFib/Flutter region starts or ends there.

### Third-degree AV Block

Each edge moves out beat by beat (at most one window) while RR stays within ±15% of the region's median. The walk stops at:
- **A conducted sinus beat:** its PR matches the next beat further out within 20 ms. In complete block "PR" changes every beat.
- **A beat inside a proven Bundle Branch Block span.**

### First-degree AV Block after AFib/Flutter — `_first_degree_first_beat()`

1. The block's own beats give a Lead II template of **P + PR segment + QRS** (PR + 80 ms before R to 60 ms after).
2. Going backward, a beat joins when both hold:
   - correlation ≥ 0.90 at about the same size,
   - RR within 15% of the block's median.
3. The Flutter/AFib region ends where the block now starts.

## B14. Summary table — arrhythmias

| Rhythm | Order | QRS | P wave / TP | RR | Key test |
| --- | --- | --- | --- | --- | --- |
| Sinus Tachycardia | Rate rule | Narrow, consistent | Consistent P and PR | Regular | RR 340–600 ms, ≥ 3 beats, ≥ 1 s |
| Sinus Bradycardia | Rate rule | Narrow, consistent | Consistent P and PR | Regular | RR 1050–6000 ms, ≥ 3 beats, ≥ 2.5 s |
| Ventricular Tachycardia | 1 | Wide 120–220 ms, uniform | No conducted P | Regular (CV ≤ 0.12) | Core rate ≥ 100 |
| Ventricular Fibrillation | 2 | Not organized, or shapes differ | No P before QRS | Irregular (CV ≥ 0.20) | Organized < 50% AND P < 50%; or no P at all + shape corr < 0.95 |
| Atrial Flutter | 3 | Organized | TP filled with steady waves | Fixed ratio | Atrial / ventricular ≈ 2, 3, 4, 5 |
| Atrial Fibrillation | 3 (after Flutter) | Organized | No consistent PR | Irregularly irregular | ≥ 50% of RR changes reverse; no regular P train |
| SVT | 4 | Organized, one shape | — | Regular (CV ≤ 0.08) | Rate 180–300 bpm |
| Third-degree AV Block | 5a | Organized (resolved or one shape) | Regular P, ≥ 4 visible, not linked to QRS | Regular, < 60 bpm | Atrial ≥ 1.3× ventricular; P→QRS std > 40 ms |
| Mobitz II | 5b | Organized | Regular P, blocked P | Pause ≈ 2× P-P | PR fixed within 20 ms |
| Mobitz I | 5c | Organized | Regular P, blocked P | Grouped | PR grows ≥ 40 ms, resets ≥ 40 ms |
| First-degree AV Block | 5d | Organized | Every P conducted | Regular, < 100 bpm | PR 200–450 ms, std ≤ 25 ms |
| Complete / Incomplete RBBB | Layer | ≥ 120 / 110–119 ms | Consistent PR | Any | V1 rsR' / terminal R' + wide S in I/V6 |
| Complete / Incomplete LBBB | Layer | ≥ 120 / 110–119 ms | Consistent PR | Any | V1 dominantly negative + broad R, no q/S in I and V6 |

---

# Part C — Ischemia Detection (ST Elevation, ST Depression, T Inversion)

## C1. How it runs

- **Calculation.**
  - `compute_ischemia(path)` returns `{'st': events, 't': T Inversion episodes, 'st_info': {...}}`.
  - It runs in the background (`ischemia_async`) when a recording opens.
  - It is cached in memory and in `ischemia_cache.json` next to the recording.
  - Cache key = recording path, size, mtime, plus a hash of `ischemia_detection.py`, the filters and the R-peak detector. **Nothing from the arrhythmia detection.**
- **Where it shows.**
  - Overview and Replay rows: T Inversion, ST Elevation, ST Depression.
  - The report's ST ANALYSIS page: summary, ST Event list, T wave section.
  - Before the calculation finishes the rows show "-".
- **Independence.** Not linked to the Auto Arrhythmia Detection button or flag. It doesn't read rhythm segments.

## C2. Data and filter

- The recording is read in **10 s frames**, with 1.5 s padding each side. A last piece ≥ 5 s is kept.
- Every lead goes through the same filter as the Full Disclosure strip (`_fd_filter`):
  1. 0.5 Hz DFT high-pass (zero-phase, 2 s mirror padding),
  2. 3-point median,
  3. Gaussian low-pass 35 Hz, twice.
- **Reference lead** for R peaks: II (else V5, I, V6).

## C3. ST Elevation / ST Depression — `detect_st_episodes()`

### Frame skipped when

| Case | Condition | Why |
| --- | --- | --- |
| Too few beats | < 4 R peaks | Value not reliable |
| Irregular RR | Median beat-to-beat RR change > 12% of RR | AFib / frequent ectopics; the PR segment holds f waves |
| Beats differ in shape | Median correlation < 0.9 (beats aligned ±20 ms) | Ectopics, noise, VFib |
| Wide QRS | ≥ 120 ms | BBB, ventricular rhythm, paced; the J point sits in a slurred QRS end |
| QRS/J not found | — | Can't measure |

### Measurement (per beat, per lead)

1. **QRS onset and J point** from the median beat: the combined slope of all leads, where it stays < 10% of its peak for 16 ms (20% if not found). The median beat is used only to locate these.
2. **Baseline = PR segment:** 40–10 ms before QRS onset, each lead its own.
3. **ST level** at **J + 40 ms** (J + 30 ms above 120 bpm), averaged over ±5 ms. Normal beats were flat from J+20 to J+40; the T wave starts by J+60.
4. **ST early** at J + 20 ms, used for the slope.
5. **Units:** Full Disclosure **screen mm** at 10 mm/mV.
   - The strip draws **933 ADC per mV**, so 1 ADC = 0.0107 mm.
   - Example: 342 ADC = +3.7 mm.
6. **Frame value of a lead** = the most frequent value among its beats.
   - Beat values go into 0.1 mm bins, the fullest bin wins (a tie goes to the bin nearest the median), and the value is the mean of that bin.
   - This is not an averaged beat, so one bad beat can't move it.
7. **Noisy lead:** if a lead's beats spread > 1 mm (interquartile range), that lead is not judged in that frame.

### Criteria per frame

Limb and chest leads are judged separately and never mixed. **aVR is not judged.**

| | Limb (I, II, III, aVL, aVF) | Chest (V1–V6) |
| --- | --- | --- |
| ST Elevation | ≥ +1 mm | ≥ +2 mm |
| ST Depression | ≤ −1 mm, not upsloping | ≤ −1 mm, not upsloping |
| Leads needed | ≥ 2 in the group | ≥ 2 in the group |

- **Not upsloping:** the ST rises no more than 0.5 mm from J+20 to the ST point.
- **Both in one frame:** the frame is **ST Elevation** (the depression is reciprocal).

### Events

- Consecutive frames of one label form an event. A gap of up to **one frame (10 s)** is joined.
- An event is kept when its qualifying frames add up to **≥ 30 s**.
- **Per qualifying lead:**
  - **average** over the frames it qualified in,
  - **max** (largest elevation / deepest depression).
- **Stored fields:**
  - `st_lead_mm`: average,
  - `st_lead_max_mm`: max,
  - `st_mv`: largest value in mV,
  - `st_all_mm`: every lead, average over the event's frames.

### Display (Overview / Replay)

```
1 event, 1m50s (mm avg/max)
1) 13:57:01 - 13:58:51 (1m50s)
    Limb: I +2.1/+2.5, II +3.0/+3.6, III +1.1/+1.2, aVF +2.0/+2.4
    Chest: V3 +2.8/+3.4, V4 +3.8/+4.6, V5 +2.8/+3.4, V6 +2.8/+2.8
```

- The longest 5 events are shown in time order, with "+N more (report)" after them.
- System start–end time and duration per event.
- **No event:** "None".
- **Recording < 30 s:** "Recording too short (< 30 s)".
- **No frame measurable:** "Not measurable (wide QRS / irregular / noise)".

### Report (ST ANALYSIS page)

- **Summary:** episodes, total duration, max (mV, from `st_lead_max_mm`), and start time of that max, for depression and elevation.
- **ST Event list:** one row per qualifying lead of each event, showing the average in mV, the lead, start–end time, and Elevation/Depression.

## C4. T Inversion — `detect_t_inversion_episodes()`

Same frames, filter, beats, QRS bounds and PR baseline as ST.

### Frame skipped when
- < 4 beats,
- heart rate > **120 bpm** (the T wave runs into the next P),
- beats differ in shape (correlation < 0.9),
- QRS ≥ 120 ms,
- QRS/J not found.

### Measurement (median beat, per lead)

1. **T search window:** from **J + 60 ms** to **0.55 × RR** after R (at most 0.50 s), so it never reaches the next P wave. It needs at least 80 ms.
2. **Smoothing:** the median beat is smoothed over ~40 ms (Savitzky–Golay).
3. **T amplitude** = the largest **turning point** (peak or trough, prominence ≥ 0.02 mV) inside the window, relative to the PR baseline.
4. **Not a T wave:** a slope that only runs into the window edge (a downsloping ST, the next P) counts as 0.

### Criteria per frame

- **Inverted lead:** T ≤ **−0.1 mV**.
- **Not judged:** **aVR** (limb) and **V1** (chest). An inverted T is normal there.
- **Groups:** limb and chest are judged separately; ≥ 2 leads of one group must be inverted.

### Episodes

- Gaps up to **60 s** are joined.
- An episode is kept when it lasts **≥ 60 s**.
- Per lead, the most negative T of the episode is kept.

**Display:** Overview/Replay show the episode count, total duration, and per lead the most negative value. The report shows up to 4 episodes in the T wave section.

**Scale note.** T Inversion still uses the older scale of **200 ADC per mV** (`theme.ADC_TO_MV`) and the 60 s rule. ST uses the screen scale (933 ADC/mV) and 30 s. So T values in mm are not on the same scale as ST values and as the Full Disclosure screen.

## C5. Summary table — ischemia

| Finding | Measured on | Baseline | Point | Threshold | Leads | Duration | Units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ST Elevation | Every beat, most frequent value per 10 s frame | PR segment (−40..−10 ms from QRS onset) | J + 40 ms (J + 30 above 120 bpm) | Limb ≥ +1 mm, chest ≥ +2 mm | ≥ 2 in one group, aVR not judged, noisy leads skipped | Events ≥ 30 s, gap ≤ 10 s | Screen mm (933 ADC/mV) |
| ST Depression | Same | Same | Same | ≤ −1 mm, not upsloping (rise ≤ 0.5 mm) | Same | Same | Same |
| T Inversion | Median beat per 10 s frame | Same | Largest turning point J+60 ms .. 0.55 RR | ≤ −0.1 mV | ≥ 2 in one group, aVR and V1 not judged | Episodes ≥ 60 s, gap ≤ 60 s | 200 ADC/mV |
