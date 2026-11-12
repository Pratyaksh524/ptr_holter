# Auto Segment Arrhythmia Detection — Rule-Based Logic

*As of 2026-10-28 (previous version 2026-10-14)*

Reference for the rule-based logic in `src/ecg/holter/auto_segment_arrthymia_detection.py` that powers Full Disclosure's "Auto Arrhythmia Detection" button.

## What changed since 2026-10-14

New labels:

| Label | Section |
| --- | --- |
| Ventricular Tachycardia | 5 |
| Supraventricular Tachycardia (SVT) — replaces "Atrial Tachycardia" | 8 |
| First-degree AV Block | 9 |
| Second-degree AV Block (Mobitz I) | 9 |
| Second-degree AV Block (Mobitz II) | 9 |
| Third-degree AV Block | 9 |
| Complete Right Bundle Branch Block | 10 |
| Incomplete Right Bundle Branch Block | 10 |
| Complete Left Bundle Branch Block | 10 |
| Incomplete Left Bundle Branch Block | 10 |

Changed rules:

- **Ventricular Fibrillation:**
  - Regular-RR veto.
  - Second path for coarse, slow VFib.
  - Bundle Branch Block veto.
  - Neighbour guard for one-window calls.
- **Atrial Fibrillation / Atrial Flutter:** a regular P-wave train cancels the call.
- **Sinus Tachycardia / Bradycardia:**
  - New RR limits.
  - P/PR/QRS sinus-conduction gate.
  - Missed-beat recovery for Bradycardia.
- **Beat-exact edges:**
  - VT start.
  - VT → VFib.
  - SVT.
  - SVT ↔ AFib/Flutter.
  - Third-degree AV Block.
  - First-degree AV Block after AFib/Flutter.
  - AFib/Flutter end before a fast regular rhythm.
  - Bundle Branch Block beat extension.

ST Elevation / ST Depression / T Inversion are **not** part of this module. They are values computed by `ischemia_detection.py` and have no link to auto arrhythmia marking.

## 1. Pipeline Overview

Detection runs once per recording. The result is cached on disk, keyed to the recording file and a hash of every `.py` file under `src/ecg`.

1. **Whole-recording R-peak pass.**
   - Pan-Tompkins runs in 60s chunks, with 3s padding on each side.
   - It uses the single best-amplitude lead from the recording's first 30s (candidates II, V5, V6, V4, V1, I).
   - The result is one continuous R-peak timeline. It feeds the rate rules and the beat-level edge refinement.
2. **Per-beat measurements for the rate rules.**
   - Every R peak is measured with `measure_beat(strict_pr=True)` on **Lead II** (the conduction lead).
   - The measurement is taken on the raw signal (`rate_beat_metrics`) and on a 25 Hz zero-phase low-passed copy (`rate_gate_metrics`).
   - The sinus gate reads the low-passed copy. That is the signal the top-bar PR was tuned on.
3. **10s window / 5s step classification.**
   - Each window reads 2s of padding on each side so Pan-Tompkins has settling room.
   - R peaks are the union of Pan-Tompkins on the top 3 candidate leads. Peaks within 100 ms are merged into one beat.
   - Only peaks inside the window's true bounds are judged.
   - Rhythm rules run in strict priority order, and the first one that fires wins:
     1. Ventricular Tachycardia
     2. Ventricular Fibrillation
     3. Atrial Flutter / Atrial Fibrillation
     4. Supraventricular Tachycardia (SVT)
     5. AV Block: Third-degree → Mobitz II → Mobitz I → First-degree
   - Bundle Branch Block is judged separately on every window. It is a layer on top of the rhythm (section 10).
4. **Window-vote cleanup and merge** (section 11).
5. **Beat-level edge refinement.**
   - A window vote only locates a region to within one 5s step.
   - Each rhythm's start and end is then snapped to the exact beat where the waveform changes (section 12).

**Sinus Tachycardia** and **Sinus Bradycardia** are decided separately, directly from the whole-recording RR timeline (section 4). They are always trimmed wherever a morphological rhythm overlaps them.

## 2. ECG Waveform Components Used by Every Rule

Every rule below reads the same per-beat measurements, produced by `measure_beat()`:

| Component | What it is | How it's found |
| --- | --- | --- |
| R peak | The QRS's dominant deflection | Detected directly by Pan-Tompkins |
| Q onset | Start of the QRS complex | `_qrs_bounds()`. It also returns `qrs_bounds_resolved`, which is True only when a genuine isoelectric boundary was found. A chaotic or noisy beat fails this. |
| J point | End of the QRS complex | `_qrs_bounds()` |
| QRS duration (`qrs_ms`) | J point − Q onset, in ms | A beat with QRS <40ms or >200ms is flagged noisy and excluded from P/PR/T measurement |
| P wave | Atrial depolarization | Searched for 50–400 ms before Q onset. Gives `p_present`, `p_onset`, `p_peak`, `p_amplitude`. `strict_pr=True` takes the candidate closest to the QRS. |
| PR interval | Q onset − P onset, in ms | Only kept (`pr_ms`) when it falls in the valid range; otherwise it is treated as unmeasurable |
| T wave | Ventricular repolarization | Found by a tangent method after the QRS. Gives `t_peak`, `t_end`, `t_amplitude`. |
| QT interval | T end − Q onset, in ms | Derived once T end is found |
| TP segment | The quiet baseline between one beat's T wave and the next beat's P wave/QRS | Computed by each rule. Skip past the previous beat's QRS+T (280ms, or 38% of the RR if longer), then look up to the next beat's P onset, or 150 ms before its Q onset if no P was found. |
| Baseline / isoelectric line | Local reference level | Estimated by `_qrs_bounds()` |

## 3. Signal Quality / Noise Gate

**SQI gate.**
- Before any rhythm or Bundle Branch Block rule runs on a 10s window, `SignalQuality.compute_sqi()` scores the window's own unpadded samples on the best-amplitude lead.
- It uses the same scorer and **0.6 cutoff** as the real-time recording pipeline (`signal_quality_threshold` in `config/clinical_config.yaml`).
- If SQI < 0.6, the window is skipped entirely and no label is produced for it.
- This exists because every morphological rule is judged purely off waveform shape. A chaotic signal that is not physiological (motion artifact, a loose electrode, EMG noise) can satisfy those rules just as well as real Ventricular Fibrillation.

**Bandpass + notch.**
- `SignalPreprocessor` applies a 0.5–40 Hz bandpass and a mains notch.
- It runs inside the AFib/Flutter, SVT, AV Block and Bundle Branch Block checks.
- Without it, 50 Hz mains ripple alone exceeded the "quiet baseline" threshold and read as Atrial Fibrillation. It also made QRS onset impossible to find on noisy chest leads.

## 4. Rate-Based Rules: Sinus Tachycardia / Sinus Bradycardia

Decided directly from the whole-recording RR timeline, independent of the window vote.

| Rule | Start condition | Stay-in condition (hysteresis) | Minimum to report |
| --- | --- | --- | --- |
| Sinus Tachycardia | 340 ms ≤ RR ≤ 600 ms (100–176 bpm) | RR ≤ 630 ms, if already in a run AND the next RR is 340–600 ms | ≥ 3 beats, ≥ 1.0 s, and passes the sinus gate |
| Sinus Bradycardia | 1050 ms ≤ RR ≤ 6000 ms (10–57 bpm) | RR ≥ 1020 ms, if already in a run AND the next RR is back ≥ 1050 ms | ≥ 3 beats, ≥ 2.5 s, and passes the sinus gate |

- **Why the 340 ms lower bound.** The P detector suppresses P whenever the previous QRS is within 333 ms. RR jitters around that edge at ~180 bpm, so those P-less beats would fail the gate for the whole run.
- **Why the 6000 ms ceiling.** A stretch where Pan-Tompkins finds no peaks at all (for example a VFib onset) reads as one multi-second "RR" and would otherwise become Bradycardia. 2000 ms was too low: real 30 bpm sinus bradycardia sits right at 2000 ms.

### Sinus-conduction gate (`sinus_conduction_gate`)

These are sinus labels, so an RR pattern alone is not enough. Let *required* = 75% of the run's beats, with a minimum of 3. The run passes only when all of the following hold:

1. At least *required* beats have:
   - a P wave with onset and end,
   - a resolved QRS,
   - QRS 40–200 ms.
2. At least *required* beats have a PR within 25 ms of the median PR. For runs over 30 beats, the median is a rolling one over ±15 beats, so slow autonomic PR drift doesn't break a long run.
3. QRS duration std ≤ 20 ms.
4. P-P follows R-R: on ≥ 75% of beats, \|P-P − R-R\| ≤ max(120 ms, 15% of RR). This rejects noise bumps whose timing doesn't track the QRS.
5. P polarity is consistent (rolling ±15 beats for long runs).
6. P duration is within 40 ms of the median (rolling for long runs).

### Bradycardia missed-beat recovery

**The problem.** Pan-Tompkins widens its refractory period to a third of the chunk's median RR. A chunk dominated by a very slow rhythm can therefore drop beats of a 60 bpm rhythm in the same chunk. Two 1000 ms RRs then read as one 2000 ms "slow" RR.

**The fix.**
1. Every gap that Bradycardia could count is re-detected with Pan-Tompkins on just that gap ±3 s.
2. Beats found inside the gap are added to Bradycardia's own beat list and measured the same way.
3. `all_r_peaks` itself is untouched.

## 5. Ventricular Tachycardia — `_is_ventricular_tachycardia_window()` (new)

**Defining signature:** an organized, fast, regular, **wide**-complex rhythm without conducted P waves. It is checked **first**, ahead of VFib. Otherwise a regular monomorphic VT, whose wide QRS often doesn't "resolve", could read as VFib.

**1. Stable core.** A window can straddle the VT onset, so only the dominant regular run is judged:
- The core is the longest run of RRs within ±15% of the window's median RR.
- **Fallback.** A window split about evenly between sinus and VT has its median in between, so no run fits. Only then, the core is the longest run within ±15% of the run's own median, and it must be at least 6 RRs. Inside VFib, 3–4 RRs can match by chance.
- The core must have ≥ 3 RRs.

**2. Gates.** All must clear:

| Gate | Condition |
| --- | --- |
| Beats | ≥ 4 measured beats in the window |
| Rate | Core rate ≥ 100 bpm |
| Regular | Core RR std / median ≤ 0.12 AND median absolute deviation / median ≤ 0.10 |
| Wide QRS | ≥ 75% (min 3) of core beats with QRS 120–220 ms |
| Uniform QRS | QRS duration std ≤ 25 ms |
| No conducted P | P with PR ≥ 100 ms on ≤ 50% of core beats, AND valid PR ≥ 100 ms on ≤ 50% |

A "P" less than 120 ms after the previous beat is that beat itself, not a P wave, and is ignored. At 160 bpm the P search otherwise reached back onto the previous VT complex.

**3. Drawn start.** The segment starts at the first beat of the stable core (`_ventricular_tachycardia_core_start`), not at the window edge.

## 6. Ventricular Fibrillation — `_is_vfib_window()`

Checked second, after VT.

**Regular-RR veto (new).** With ≥ 3 peaks, if RR std / mean RR < 0.20, the window is **not** VFib. A regular rhythm (VT, sinus tachycardia) can't be VFib.

**Main path.** For every beat in the window:
- `p_follows_qrs` counts a beat only when all of these hold:
  - `p_present`,
  - `qrs_bounds_resolved`,
  - `p_onset < q_onset`.

  On a chaotic beat whose QRS boundary wasn't found, almost any noise bump trivially satisfies `p_onset < q_onset`.
- `organized_qrs` counts a beat when `qrs_bounds_resolved` is True.

```
no_p_follows_qrs = p_follows_qrs < beats / 2
no_organized_qrs = organized_qrs < beats / 2
VFib = no_p_follows_qrs AND no_organized_qrs
```

**Coarse-VFib path (new).**
- **The problem.** Coarse, slow VFib has big rounded waves whose sides flatten enough for the QRS boundary to "resolve" on half or more of them. The organized-QRS test then fails, although no two waves look alike.
- **The rule.** Such a window is also VFib when **no** beat has a P before its QRS, there are ≥ 5 beats, and the beat-shape correlation is < 0.95.
- **Beat-shape correlation** (`_aligned_beat_shape_corr`):
  - each beat is taken as a 120 ms snippet,
  - each is aligned ±40 ms to the window's median template,
  - the value is the median correlation.
- **Measured values.** AFib and Flutter scored 0.97–1.00. The missed VFib scored 0.48–0.93.

**Bundle Branch Block veto (new).** VFib is not called where the same window is a proven conducted Bundle Branch Block (section 10). VFib has no AV conduction. A vetoed window continues down the rest of the chain.

At least 2 peaks are needed to judge at all.

## 7. Atrial Fibrillation / Atrial Flutter

Checked third. Both share one setup (`_prepare_atrial_window_context`). Flutter is tried first, AFib second.

### Shared pre-condition and TP-segment scoring

- At least **6 beats** in the window.
- **QRS organized** (`qrs_bounds_resolved`) in ≥ 2/3 of beats. This is what separates AFib/Flutter from VFib.
- For each beat, the **TP segment** is measured. Skip 280 ms (or 38% of the RR) past the previous R, then take the stretch up to this beat's P onset, or up to 150 ms before Q onset if there is no P.
- **Quiet or active.** A TP segment is **quiet** when its peak-to-peak is below `noise_threshold = max(0.05, 0.15 × median QRS amplitude)`. Otherwise it is **active**, and the waves inside it are counted.
- **Minimum sample.** Needs ≥ 4 scored beats, or the window gives no verdict.

### Atrial Flutter — `_classify_atrial_flutter_window()`

If quiet beats are < 40% of scored beats, the waves found in the active TP stretches give an **atrial rate**:

```
ratio = atrial_rate / ventricular_rate
Atrial Flutter = ratio within 0.5 of 2, 3, 4 or 5
```

This is a fixed-ratio sawtooth check read off the waveform. A spectral "flutter band" score was tried and rejected: it was only QRS harmonics.

### Atrial Fibrillation — `_classify_atrial_fibrillation_window()`

Checked only if Flutter did not fire. Both gates must clear:

1. **Organized-P veto.** If ≥ 60% of beats (min 3) have a PR AND std(PR) ≤ 25 ms, this is organized conduction, so the window is not AFib.
2. **Irregularly irregular RR.**
   - Take RR changes larger than 60 ms. At least 3 are needed.
   - ≥ 50% of them must reverse direction from the previous one.

### Regular P-train veto (new)

- **The problem.** Mobitz I's grouped beating plus its changing PR reads as "irregular RR, no consistent PR", which is the AFib picture.
- **The rule.** An AFib/Flutter call is cancelled when the AV conduction context (section 9) finds a regular train of discrete P waves (`atrial_regular_for_afib_veto`) and the QRS-resolved majority holds.
- **The train** must have:
  - ≥ 6 clearly visible P waves,
  - at most one phase reset,
  - P waves hidden in a QRS or T wave on ≤ 50% of positions.
- **Why this can't be real AFib/Flutter.** AFib has no discrete P waves, and Flutter's atrial rate is 250–350/min.

## 8. Supraventricular Tachycardia (SVT) — `_is_atrial_tachycardia_window()` (replaces "Atrial Tachycardia")

Checked fourth, only once VT, VFib and AFib/Flutter found nothing. Label: **"Supraventricular Tachycardia (SVT)"**.

All gates must clear:

1. **Beats:** ≥ 6 beats.
2. **Rate:** mean rate **180–300 bpm**.
3. **Regular RR:** RR coefficient of variation ≤ 0.08. This rules out AFib and variable-conduction Flutter (measured CV 0.19–0.47).
4. **Organized QRS:** ≥ 80% of beats with `qrs_bounds_resolved`.
5. **Uniform QRS:** median correlation to the window's QRS template (±50 ms around each R) ≥ 0.9.

Once these pass, a sustained rate ≥ 180 bpm is itself the deciding factor. Resting sinus tachycardia this fast and sustained is not physiological for an adult Holter recording.

The older P-on-T fused-wave test is still in the function: ≤ 1 wave between J point and next QRS in ≥ 70% of intervals. It is only reached below 180 bpm, which gate 2 already excludes, so **in practice it does not decide anything.**

## 9. AV Blocks (new rules; `analyze_ecg()` no longer used)

Checked fifth, only when VT, VFib, AFib/Flutter and SVT all found nothing. The rules run in severity order and the first that passes labels the window:

**Third-degree → Mobitz II → Mobitz I → First-degree**

Every rule returns the exact beats its evidence spans (for example one Wenckebach cycle), not the whole window.

### Shared conduction context — `_prepare_av_conduction_context()`

P waves are found on **Lead II** (bandpass + notch). `measure_beat()` alone can't see a blocked P: it only looks 50–400 ms before each QRS.

1. **Organized ventricles.**
   - ≥ 80% of QRS boundaries resolved, OR
   - one QRS shape (median correlation ≥ 0.9).

   The shape fallback is there because a wide escape rhythm or a Bundle Branch Block often doesn't "resolve".
2. **P template.**
   - Built from the P waves `measure_beat()` found, ignoring any "P" inside a QRS zone.
   - Seeds with correlation ≥ 0.7 to their median are kept, and at least 2 are needed.
3. **Scan.**
   - The template is cross-correlated along the whole window.
   - A hit needs correlation ≥ 0.6 and amplitude 0.5–2× the template.
   - It must not be inside a QRS zone.
   - Outside a T wave it needs correlation ≥ 0.8. Inside a T wave a weaker match is accepted, but only where the P-P rhythm says a P is due.
   - At least 3 clear (non-T) P waves are needed.
4. **Atrial (P-P) interval.**
   - The median of the shortest common spacing (≤ 1.3 × the 25th percentile).
   - It must be 300–2000 ms.
   - Tolerance = max(60 ms, 12% of P-P).
5. **Chain walk.**
   - Starting from each clear P, step one P-P at a time in both directions.
   - A P due inside a QRS or T wave is inferred and marked **hidden**.
   - A clean baseline where a P was due is a **mismatch**.
   - The chain that accounts for the most clear P waves wins. This rejects a chain that follows T waves.
6. **`atrial_regular`** is True when all of these hold:
   - no mismatches,
   - no unexplained extra P waves,
   - P-P CV ≤ 0.12,
   - hidden P ≤ 50% of positions.
7. **Conduction.**
   - A P conducts to the first QRS after it when that QRS starts 80–450 ms after the P onset and before the next P.
   - PR is measured from P peak to R peak, then shifted by the window's typical P-onset→P-peak and Q-onset→R-peak distances. This keeps beat-to-beat Q-onset jitter out of PR.

### First-degree AV Block

Every P conducts, just slowly:

1. `atrial_regular`, and every P visible and conducted (1:1).
2. ≥ 5 QRS, HR < 100 bpm (above that, P sits in the T wave).
3. RR CV ≤ 0.15, and no RR ≥ 1.5 × median (no dropped beat).
4. PR measured on ≥ 80% of QRS (min 4).
5. **Median PR > 200 ms and ≤ 450 ms**, and > 200 ms on ≥ 80% of beats.
6. **PR constant:** std ≤ 25 ms. A lengthening PR is Mobitz I; a scattered PR is complete block.

### Second-degree AV Block, Mobitz I (Wenckebach)

PR lengthens beat by beat until one P is not conducted, then resets.

**Gates:**
1. `atrial_regular`.
2. At least one **visible** blocked P, never two in a row, and conducted ≥ blocked.
3. Before the blocked P: ≥ 2 conducted beats whose PR never shortens by more than 10 ms and **grows ≥ 40 ms in total**.
4. After it: the first conducted PR is **≥ 40 ms shorter** (the reset).
5. The pause around the blocked P is < 2 × P-P.

**Long-PR exception.** For this rule only, a P whose QRS follows 450–600 ms later (still before the next P) counts as conducted. The PR right before the drop can pass 450 ms; one example cycle was 146 → 206 → 386 → 456 ms.

**Cycle handling.** A complete cycle that fails these gates (for example a constant PR) vetoes the window. A cycle cut off by the window edge is just not judged.

### Second-degree AV Block, Mobitz II

PR stays fixed, then a P is suddenly not conducted.

1. `atrial_regular`, a visible blocked P, never two in a row.
2. PR of the conducted beats before **and** after the blocked P all **within 20 ms** of each other (no lengthening, no reset).
3. The pause is **1.7–2.3 × P-P**.

**2:1 block.** One conducted beat between blocked P waves can't show a PR trend. It is reported only when the fixed-PR 2:1 pattern repeats (≥ 2 cycles in the window).

### Third-degree (complete) AV Block

Atria and ventricles beat independently.

1. A regular atrial rhythm with **≥ 4 visible P waves**.
   - This uses `atrial_regular_complete_block`.
   - It forgives a mismatch right after a P hidden in a QRS when the real P is within two tolerances. The guessed position of a hidden P makes the next one drift.
2. ≥ 3 QRS, **ventricular rate < 60 bpm**, RR CV ≤ 0.12 (a regular escape rhythm).
3. **Atrial rate ≥ 1.3 × ventricular rate.**
4. **AV dissociation:** the time from the nearest P to each QRS is scattered, with **std > 40 ms**. In 2:1 block it stays fixed.

Mobitz I can also scatter that time, but its ventricular rhythm is grouped, so gate 2 keeps it out.

## 10. Bundle Branch Blocks (new, separate layer)

A Bundle Branch Block is a QRS-shape finding that sits **on top of** the rhythm. These rules:
- run on every non-noisy window on their own,
- never enter the rhythm priority chain,
- never trim or get trimmed by a rhythm segment,
- never suppress Sinus Bradycardia/Tachycardia.

The only exception: no Bundle Branch Block is drawn over Ventricular Tachycardia or Ventricular Fibrillation.

### Shared gate — `_bundle_branch_block_window_beats()`

All leads are bandpass + notch filtered first.

**1. Beats.** ≥ 5 beats. Let *required* = 75% of beats, with a minimum of 3.

**2. Conducted rhythm (P / PR on Lead II).**
- Each beat is measured at Lead II's own dominant QRS deflection (largest excursion −40..+100 ms from the detected beat), not at the merged R peak. In a wide QRS the earliest lead's peak sits near the QRS onset.
- The strict P search runs first. If it gives no PR and RR ≥ 500 ms, the relaxed (most prominent wave) search is used.
- P must be present and PR measured on ≥ *required* beats.
- ≥ *required* PRs must be within 25 ms of the median. This rules out VT, VFib, AFib and escape rhythms.

**3. One QRS shape.**
- Median correlation to the template ≥ 0.9.
- Each ±50 ms snippet is centred on the best lead's dominant deflection. The merged beat positions jump ~20 ms in a wide QRS.

**4. Global QRS width.**
- Combined slope of leads I, II, aVL, V1, V2, V5, V6, each scaled by its own QRS size.
- QRS onset and end are where the slope stays under **10% of its peak for 16 ms**.
- If no end is found (LBBB's discordant ST-T starts right at the QRS end), **20%** is used for the end.
- The window width is the median over beats.

**5. Per-beat shape.** Each lead's baseline is its PR segment.

| Lead | Right pattern | Left pattern |
| --- | --- | --- |
| V1 (or V2) | rsR' (two positive waves ≥ 30 ms apart with a dip) or a terminal positive R' | Dominantly negative (negative area ≥ 2× positive), no rsR' |
| I and V6 (or V5) | Wide slurred S: ≥ 15% of QRS size, ≥ 40 ms below baseline after the R | Broad R ≥ 60 ms, positive area ≥ 3× negative, no septal q, no S |

- A **Right** beat needs the V1 pattern AND a wide S in at least one lateral lead.
- A **Left** beat needs the V1 pattern AND a broad R in every lateral lead.

### Labels

| Label | QRS width | Shape |
| --- | --- | --- |
| Complete Left Bundle Branch Block | ≥ 120 ms | Left on ≥ 75% of beats |
| Complete Right Bundle Branch Block | ≥ 120 ms | Right on ≥ 75% of beats |
| Incomplete Left Bundle Branch Block | 110–119 ms | Left on ≥ 75% of beats |
| Incomplete Right Bundle Branch Block | 110–119 ms | Right on ≥ 75% of beats |

- Checked in this order.
- The span runs from the first to the last beat with that pattern, not the whole window.

### After the window loop

1. **Merge.** Same-label regions merge when ≤ 2.5 s apart.
2. **Beat-level extension.**
   - Every beat any window judged keeps its vote.
   - A region's edges move outward one beat at a time while the next beat's majority vote is the same pattern.
   - Limits: no gap > 1.5 × RR, at most one window, never into another Bundle Branch Block region.
3. **One-beat gap closed.** Where two Bundle Branch Block regions meet beat to beat (Right → Left), the earlier one ends where the next starts.
4. **Priority.** Complete outranks incomplete where they overlap.
5. **Not drawn over ventricular rhythms.** Nothing is drawn over VT/VFib.
6. **VFib is cut off a proven Bundle Branch Block span.**
   - Pieces < 1 s are dropped.
   - A piece touching the Bundle Branch Block and shorter than one step (5 s) is the transition and is dropped.

## 11. Window-Vote Cleanup and Priority

Applied after every window has voted, before beat-level refinement:

- **VFib neighbour guard (new).**
  - Applies to a window called VFib only by the coarse-VFib path, or on fewer than 4 measured beats.
  - It is dropped unless another VFib window starts within one window of it.
  - Real VFib runs many windows in a row. A lone window at the recording start or end with 2–3 cut-off beats is not VFib.
- **AV Block next to AFib/Flutter is dropped.**
  - Any AV Block window within 2 windows of an AFib/Flutter window is removed. The atrial window counts only if a same-label window is nearby.
  - Flutter's changing conduction ratio reads as dropped beats.
  - **First-degree AV Block is exempt.** Its own rule (1:1, constant PR, no dropped beat) can't pass on Flutter/AFib.
- **SVT next to AFib/Flutter is dropped,** but only at a 2:1-Flutter ventricular rate (125–175 bpm) or when the rate is unknown. 2:1 Flutter reads like a regular SVT; a faster SVT can't be 2:1 Flutter.
- **AV Block next to VFib is dropped** (within one window). Conduction can't be graded on chaotic beats.
- **Isolated-window guard.** A single AFib, Flutter or SVT window with no same-label window within 2 windows is dropped.
- **Merge.**
  - Consecutive same-label windows are joined.
  - Gaps up to 2 windows are bridged, because a borderline window mid-episode can miss the vote by chance.
  - AV blocks bridge only gaps up to 2.5 s, since their spans are beat-exact.
- **AV priority.**
  - Every non-AV rhythm outranks any AV block.
  - Among AV blocks: Third > Mobitz II > Mobitz I > First.
  - A lower-ranked AV region is cut wherever it overlaps a higher one. It may split into two pieces, and it is never stretched across a gap.
- **Overlap and gap.**
  - Where two different-label regions overlap, the earlier is trimmed to where the next starts.
  - A gap up to one window between two non-AV regions is bridged.
- **Morphological priority over rate.**
  - Every morphological rhythm, AV blocks included, overrides Sinus Tachycardia/Bradycardia.
  - Rate segments are trimmed at the exact edge (`suppress_rate_overlap`).
  - A piece within 5 s of a morphological edge is dropped as a transition blip.
- **Final merge.** Same-label segments ≤ 0.5 s apart are joined.

## 12. Beat-Level Edge Refinement

### Ventricular Fibrillation

- **Start:** the first of two consecutive **disorganized** beats.
- **End:** the first of two consecutive **organized** beats, accepted only if they sit ≤ 2 s apart.
- **QRS < 40 ms** is never organized. It is a chaotic wave whose top looks flat.
- **`require_narrow`.**
  - Where VFib borders normal rhythm, an organized beat also needs QRS < 120 ms.
  - This is skipped next to AFib/Flutter, whose beats measure wide too.
- **Silent gap.**
  - Pan-Tompkins often finds no peaks in VFib's first seconds.
  - If the organized beat before such a gap is followed by nothing for > max(2 s, 1.5 × recent median RR), VFib starts right after that beat.
  - A lone beat with silent gaps on both sides is skipped, and the search continues back to the last beat of the organized run.
- **Silence fallback (new).** If no beat at all lies between the last organized beat and the window-vote boundary, and that silence is > max(2 s, 1.5 × RR), VFib starts at that beat.
- **VT → VFib (new).**
  - When VFib follows VT, VFib's start is not moved before the last beat that continues the VT run (`_vt_run_last_beat`).
  - A beat continues the run when both hold:
    - its RR is within 15% of the VT's median RR,
    - its QRS shape correlates ≥ 0.90 with the VT's median beat, at about the same size (0.5–2×).
  - One extra detection inside the run is skipped.
- **Neighbours.** The neighbouring regions' edges are pulled in to meet the refined VFib edge.

### Ventricular Tachycardia start (new) — `_vt_run_first_beat()`

**The problem.** The window holding the VT onset can have no Pan-Tompkins peaks at all: small sinus beats followed by large VT complexes upset its thresholds.

**The walk.**
1. The VT's first confirmed beats give a QRS template and RR. If the whole-recording pass has < 5 beats there, they are re-detected on that window.
2. Go backward one RR at a time (±15%) and search the signal for the template.
3. A match is one more VT beat. It needs correlation ≥ 0.90 and a size 0.5–2× the template; the size check rejects T waves.
4. The first miss ends the walk.

**Limits.** At most one window back, and never past the end of the previous region.

### Atrial Fibrillation / Atrial Flutter vs. normal rhythm

- **The edge.** The first of two consecutive beats whose TP segment is **quiet** (active-wave amplitude < 0.3× the noise threshold), searched from inside the episode outward.
- **End fallback (new)** — `_regular_run_first_beat()`.
  - At a fast following rhythm there is no TP segment, so no quiet pair is found.
  - Then the rhythm that follows decides. Its beats in the next window must be regular (RR CV ≤ 0.12).
  - Going backward, a beat belongs to that run while both hold:
    - its RR is within 15% of the run's median,
    - its QRS shape correlates ≥ 0.90 with the run's median beat, at about the same size.
  - AFib/Flutter ends at the run's first beat.

### Atrial Fibrillation ↔ Atrial Flutter (adjacent)

- Each RR interval's own autocorrelation is computed. Flutter gives a peak ≥ 0.4 at a 170–250 ms lag; AFib does not.
- The edge moves to the first of two consecutive intervals of the **new** label.

### SVT (updated)

- **Start** is the first beat that opens two consecutive RRs within ±15% of the median RR. **End** is the last beat that closes two such RRs. Both are searched out to one window past each edge.
- **Local median.** The median RR is taken from the beats next to each edge (head / tail), not the whole episode. A climbing rate (200 → 270 bpm) otherwise rejected the slower opening beats.
- **SVT range.** Each RR must be ≤ 340 ms, so SVT can't extend over 170 bpm sinus beats.

### SVT ↔ AFib/Flutter (new)

- Where an SVT region and an AFib/Flutter region overlap, or are separated by a gap up to one window, the SVT's beat-exact edge wins.
- The AFib/Flutter region starts (or ends) at that SVT beat.

### Third-degree AV Block (new)

Each edge moves out beat by beat, at most one window. A beat joins while its RR stays within ±15% of the region's median RR. The walk stops at:
- **A conducted sinus beat:** its PR matches the next beat further out within 20 ms. In complete block the "PR" changes beat to beat.
- **A beat inside a proven Bundle Branch Block span,** which is conducted.

### First-degree AV Block after AFib/Flutter (new) — `_first_degree_first_beat()`

**The problem.** The window that mixes the last flutter beats with the first First-degree beats passes neither rule.

**The walk.**
1. The block's own beats give a Lead II template of **P wave + PR segment + QRS**, from PR + 80 ms before R to 60 ms after.
2. Going backward, a beat joins the block when both hold:
   - correlation ≥ 0.90 at about the same size,
   - RR within 15% of the block's median RR.

   Per-beat P detection is not used here, because it missed P on some of these beats.
3. The Flutter/AFib region then ends where the block now starts.

## 13. Summary Table

| Rhythm | Check order | QRS | P wave / TP segment | RR | Key test |
| --- | --- | --- | --- | --- | --- |
| Sinus Tachycardia | Rate rule | Narrow, consistent | Consistent P and PR (sinus gate) | Regular | 340–600 ms RR, ≥ 3 beats, ≥ 1 s |
| Sinus Bradycardia | Rate rule | Narrow, consistent | Consistent P and PR (sinus gate) | Regular | 1050–6000 ms RR, ≥ 3 beats, ≥ 2.5 s |
| Ventricular Tachycardia | 1st | Wide 120–220 ms, uniform | No conducted P (PR ≥ 100 ms on ≤ 50%) | Regular (CV ≤ 0.12) | Rate ≥ 100 on the stable core |
| Ventricular Fibrillation | 2nd | Not organized, or shapes differ | No P before QRS | Irregular (CV ≥ 0.20) | organized < 50% AND P < 50%; or no P at all + shape corr < 0.95 |
| Atrial Flutter | 3rd | Organized | TP filled with steady waves | Fixed ratio | Atrial / ventricular rate ≈ 2, 3, 4, 5 |
| Atrial Fibrillation | 3rd (after Flutter) | Organized | No consistent PR | Irregularly irregular | ≥ 50% of RR changes reverse; no regular P train |
| SVT | 4th | Organized, one shape | — | Regular (CV ≤ 0.08) | Rate 180–300 bpm |
| Third-degree AV Block | 5th | Organized (resolved or one shape) | Regular P, ≥ 4 visible, not linked to QRS | Regular, < 60 bpm | Atrial ≥ 1.3× ventricular; P→QRS std > 40 ms |
| Mobitz II | 5th | Organized | Regular P, blocked P | Pause ~2× P-P | PR fixed within 20 ms |
| Mobitz I | 5th | Organized | Regular P, blocked P | Grouped | PR grows ≥ 40 ms, resets ≥ 40 ms |
| First-degree AV Block | 5th | Organized | Every P conducted | Regular, < 100 bpm | PR 200–450 ms, std ≤ 25 ms |
| Complete / Incomplete RBBB | Separate layer | ≥ 120 / 110–119 ms | Consistent PR | Any | V1 rsR' / terminal R' + wide S in I/V6 |
| Complete / Incomplete LBBB | Separate layer | ≥ 120 / 110–119 ms | Consistent PR | Any | V1 dominantly negative + broad R, no q/S in I and V6 |

**Priority when several rules could apply to one window:**

Ventricular Tachycardia → Ventricular Fibrillation → Atrial Flutter / Atrial Fibrillation → SVT → Third-degree → Mobitz II → Mobitz I → First-degree AV Block.

- Every one of these overrides a Sinus Tachycardia/Bradycardia segment over the same time.
- Bundle Branch Blocks are drawn alongside the rhythm, except over VT/VFib.
