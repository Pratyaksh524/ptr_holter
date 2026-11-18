# Auto Arrhythmia Detection — Complete Pipeline

Reference for `ecg/holter/auto_segment_arrthymia_detection.py` (function
`_detect_arrhythmias_uncached`), as it actually behaves as of this writing.
Every number/rule below was confirmed by reading or tracing the live code,
not from memory of an old design. Supporting modules: `ecg/pan_tompkins.py`,
`ecg/arrhythmia_detector.py` (`measure_beat`), `ecg/holter/analysis_pipeline.py`
(`SignalPreprocessor`, `SignalQuality`), `ecg/holter/pqrst_measure.py`
(`frame_status`, `span_status`, `measure_beats`).

Input: one `.ecgh` recording (`reader`), its lead names, sample rate `fs`
(typically 500 Hz). Output: a flat list of `{label, start_sec, end_sec,
color, ...}` segments.

---

## 1. Lead selection (done once, at the top of the function)

- **`best_lead_idx`**: the lead with the largest peak-to-peak amplitude in
  the first 30 s, chosen from the candidate list `['II', 'V5', 'V6', 'V4',
  'V1', 'I']` (first one present in the recording's lead list that has data
  — picked by actual amplitude, not list order). Used for: `all_r_peaks`
  detection, the window loop's `padded_signal` (what every morphological
  rule — VT, VFib, AFib/Flutter, SVT, AV block context — actually looks
  at), and most beat-exact edge refinements.
- **`peak_detection_lead_indices`**: the top 3 leads by that same early
  amplitude ranking (always includes `best_lead_idx`). Used only for
  R-peak *detection* inside the window loop (`core_r_peaks` — a union
  across these 3 leads, not just `best_lead_idx` alone), so a transient
  amplitude dip on one lead doesn't make Pan-Tompkins miss real beats. The
  beats themselves are still *measured* on `best_lead_idx`'s own signal.
- **`conduction_lead_idx`**: `'II'` if present, else `best_lead_idx`. Used
  for every P-wave / PR / AV-conduction judgement (sinus gate, AV block
  context, bundle branch block PR).

---

## 2. Noise / signal-quality handling — four separate layers

These are **not one check** — each rhythm family answers to a different
one, and they were added at different times for different failures.

### 2a. `SignalPreprocessor` (baseline removal + band-pass + mains notch)
`ecg/holter/analysis_pipeline.py`. Used to pre-filter the signal before:
the window loop's atrial-rhythm context (`_atrial_preproc`, feeds AFib /
Flutter / SVT / bundle-branch-block beat measurement), and nowhere else in
the window loop itself (VT/VFib work on the raw `padded_signal`).
`remove_powerline()` notches **only `config.notch_hz` (50 Hz by default)**
— **60 Hz interference is not removed here.** Confirmed consequence: on a
60 Hz-heavy recording this inflates the Atrial Flutter wave-count (~2.7x),
pushing the computed atrial rate outside the 2:1–5:1 ratio band the
Flutter rule checks, so a real Flutter episode goes unlabelled. Known,
not yet fixed (would need a full regression before touching, since this
preprocessor is shared by many rules).

### 2b. Window-level noise gate (`window_is_noisy`)
Inside the main window loop, before any morphological rule runs for that
window: `SignalQuality.compute_sqi(core_signal)` on `best_lead_idx`'s own
*unpadded* window signal; if `< sqi_min` (0.6, from
`config/clinical_config.yaml`, same threshold the live recording pipeline
uses) the window produces **no morphological label at all** — not even
"no label", it's skipped outright, same as if nothing matched.

### 2c. `frame_status` / `span_status` (`pqrst_measure.py`)
A finer-grained, independent check used only by: the Unidentified pass,
and the VFib test-signal trim. Categories, computed on 10 s frames (or
beat-snapped sub-pieces for `span_status`):
- **`ok`** — usable.
- **`test_signal`** — leads `I`/`V1`/`V5` correlate ≥ 0.95 **and** the
  waveform has flat steps (a square/pulse-train generator, not a heart).
- **`non_ecg`** — leads `I`/`V1`/`V5` correlate ≥ 0.95 (identical shape on
  every lead) but without the flat-step requirement — e.g. a sine/triangle
  test wave, or (confirmed) a genuine ECG frame dominated by un-notched
  60 Hz hum (the hum is identical on every lead, so the leads "agree").
- **`noise`** — Lead II's own SQI < 0.6.

### 2d. Final noise gate (very end of the pipeline, after every label exists)
Per 10 s block (60 s reads, 10 s sub-blocks): every **source** lead (`I`,
`II`, `V1`–`V6`; the 4 *derived* leads `III`/`aVR`/`aVL`/`aVF` are skipped
— they're computed from `I`/`II` so they carry the same noise and can fail
purely from smaller natural amplitude) gets `SignalQuality.lead_scores`.
If **any one** source lead's score is `< sqi_min`, the whole 10 s block is
`noisy_blocks`; if the specific failing lead is additionally **flat** (not
just low-quality/chaotic), the block also goes in `flat_noisy_blocks`.
Adjacent noisy blocks merge. Then, for every segment except
`Unidentified` (left alone — it already means "nothing readable"):
- **Ventricular Fibrillation / Ventricular Tachycardia** are cut only by
  `flat_noisy_blocks` — a genuinely chaotic (not flat) lead is expected
  for real VFib/VT and must not cut it on its own merit.
- **Every other label** (AFib, Flutter, SVT, AV blocks, bundle branch
  block, Sinus Brady/Tachy/Arrhythmia, plain Brady/Tachy) is cut by the
  full `noisy_blocks` — any one bad source lead removes the label there,
  whether or not that lead was used to make the call.

A trimmed piece must be ≥ 2 s (`_NOISE_MIN_PIECE_SEC`) or it's dropped too.

This is the layer that, on a recording where lead V1 saturates (pins at
the ADC ceiling) partway through, silently removed every rhythm label for
the rest of the recording — the rules behind it had all fired correctly;
only the display was suppressed, and only because of V1, not the leads
those rules actually used.

---

## 3. Beat (R-peak) detection — three separate detectors, for three purposes

All three run Pan-Tompkins (`ecg/pan_tompkins.py`: 5–15 Hz band-pass →
derivative → square → 150 ms moving-window integration → adaptive
threshold `mean + k·std` for `k` in `(0.50, 0.30, 0.15, 0.10)`, refractory
≥ 0.22·fs widened to ⅓ of the chunk's own median RR when that median is
slow, search-back ±75 ms to the true peak, de-dup within the refractory
window keeping the larger-amplitude peak) — but on different
reads/leads, which is why their outputs disagree with each other:

1. **`all_r_peaks`**: one Pan-Tompkins call per **60 s chunk** (±3 s pad,
   trimmed to the chunk's own core), on **`best_lead_idx` alone**. Feeds
   the sinus-rate rules, the mains-Artifact rule, the Atrial Flutter
   extension, and the existing `_svt_runs_in_gap` gap-fill. **Known
   weakness**: one amplitude threshold over a whole 60 s chunk — if that
   chunk mixes a small-amplitude fast rhythm with much larger beats
   elsewhere in the same chunk, the small beats can be missed entirely
   (confirmed: completely missed an 8.8 s run of genuine VT beats this
   way).
2. **Window loop `core_r_peaks`**: per 10 s window (5 s step, ±2 s pad),
   union of `peak_detection_lead_indices` (top 3 leads), merged within
   100 ms as one beat, trimmed to the window's own `[current_time,
   window_end)`. Feeds every window-level morphological rule (VT, VFib,
   AFib/Flutter, SVT, AV block). **Same weakness, same reason**: if the
   window (or its ±2 s pad) also contains a much larger-amplitude beat
   (a neighbouring sinus beat, or VFib-onset chaos), the threshold is set
   by that bigger beat and the smaller true beats inside the same window
   get missed — confirmed directly: a window isolated to a VT run alone
   found 13/13 of its own beats; the same stretch inside the real 10 s
   padded window found only 1–6.
3. **Local, narrow-window re-detection** (added for the VT-carve pass
   below only): 1.5 s steps, ±0.5 s pad, union of the same top-3 leads.
   Exists specifically to avoid the amplitude-dilution problem above by
   keeping each individual Pan-Tompkins call's own amplitude range small.

A fourth, **missed-beat recovery** pass exists only for the Sinus
Bradycardia rule: every `all_rr_ms` gap that already looks "slow enough"
to maybe be Bradycardia is re-read ±3 s and re-run through Pan-Tompkins on
its own — because the 60 s-chunk refractory widening (above) can turn two
real 1000 ms RR intervals into one apparent 2000 ms "slow" RR, which would
otherwise draw Bradycardia over plain 60 bpm beats.

---

## 4. Per-beat measurement — `measure_beat()` (`ecg/arrhythmia_detector.py`)

Called throughout the detector (and separately by `pqrst_measure.py` for
the on-screen PR/QRS/QT readout — a different call site, see §9). Per beat,
given a signal and an R-peak sample index, returns a dict including:
`p_present`, `p_onset`, `p_end`, `p_duration_ms`, `p_amplitude`, `pr_ms`,
`q_onset`, `j_point`, `qrs_ms`, `qrs_bounds_resolved`, `qrs_amplitude`,
`t_end`, `qt_ms`, `st_level_mv`, `noisy`/`noise_reasons`.

- **P / PR**: searched in a fixed window before the QRS (50–400 ms before
  R by default); `strict_pr=True` (used by the sinus-rate rules) narrows
  this and requires a cleaner isolated wave. A P/R amplitude-collision
  guard only engages when RR < 333.3 ms (≥ 180 bpm).
- **QRS**: onset/offset found by slope crossing a threshold relative to
  the window's own peak slope (~10%, held for 16 ms, 20% when a
  discordant ST-T keeps the signal elevated); `qrs_bounds_resolved` is
  `True` only when both ends were actually found this way (not a
  fallback estimate).
- **QT**: QRS onset to the T wave's end, searched within that beat's own
  RR interval (not a fixed window — avoids reading into the next beat's
  QRS at fast rates).

`sinus_gate_metrics` (one `measure_beat(..., strict_pr=True)` call per
`all_r_peaks` beat, on the **conduction lead**, filtered first by
`_sinus_gate_filter` — see §6) is the shared input to every sinus-rate
rule (Tachycardia, Bradycardia, Arrhythmia, First-degree AV block
beat-by-beat check).

---

## 5. Sinus gate — `_sinus_conduction_gate()` / `_sinus_beat()`

The shared gate every *sinus* (rate-based) label must pass before an RR
pattern alone is allowed to call it. A beat is individually "valid" when
it has: `p_present`, a found `p_onset`/`p_end`, `qrs_bounds_resolved`,
a resolved `q_onset`, QRS width 40–200 ms, and a measured `pr_ms`.
For a candidate run of beats (`_sinus_conduction_gate`), **all** of the
following must hold (≥ 75% of beats pass each check, `max(3, ceil(0.75 ×
n))`):

1. **P/QRS present & resolved** on ≥ 75% of beats (as above).
2. **PR consistency**: PR within ±25 ms of the run's median (short runs,
   ≤ 30 beats) or of a rolling ±15-beat local median (long runs — lets PR
   genuinely drift over minutes without failing the gate).
3. **QRS width consistency**: `std(qrs_ms) ≤ 20 ms` across the valid
   beats.
4. **P-P tracks R-R**: the atrial cycle must follow the ventricular cycle
   (`|P-P − matching R-R| ≤ max(120 ms, 15% of R-R)` on ≥ 75% of pairs) —
   rejects a random baseline bump that happens to look like a P wave but
   doesn't move with the heartbeat.
5. **P polarity consistency**: the P wave's sign (±) must agree with the
   local majority on ≥ 75% of beats (random local window for long runs).
6. **P duration consistency**: `p_duration_ms` within ±40 ms of the
   (local) median on ≥ 75% of beats.

Any failure → the whole candidate run is rejected (falls through to
whatever the non-sinus rules decide, or stays unlabelled / plain-rate
fallback).

---

## 6. Window-loop morphological rules (10 s window, 5 s step, checked in
this exact order — the first one that fires wins the window)

### 6.1 Ventricular Tachycardia — `_is_ventricular_tachycardia_window`
- Needs ≥ 4 `core_r_peaks` and ≥ 4 measured beats.
- Finds the window's own longest **stable RR core** (consecutive RRs
  within 15% of a local median; falls back to the longest run within 15%
  of its *own* running median if that's < 3 RRs long, and that fallback
  must be ≥ 6 RRs) — deliberately isolates a real stable VT run from
  mixed transition beats at the window's edges.
- That core must have: rate ≥ 100 bpm, `std(RR)/median(RR) ≤ 0.12`,
  `median(|RR − median RR|)/median(RR) ≤ 0.10` (regular, not chaotic).
- ≥ 75% of the core's beats (min 3) must measure QRS 120–220 ms, and
  `std(QRS) ≤ 25 ms` across those wide beats.
- A P/PR check vetoes this as supraventricular-with-aberrancy when an
  organized P/PR majority with PR ≥ 100 ms is found (careful not to treat
  a stray sub-100 ms bump as conducted P).
- The drawn segment's **start** is trimmed to `_ventricular_tachycardia_core_start`
  (same stable-core search, applied again) so onset-transition beats
  aren't included.
- The **end** is cut beat by beat (`_vt_run_last_beat_fwd`: shape corr ≥ 0.90
  against the VT's own template, size 0.5–2×, RR within 15%; two misses in
  a row end the run). The region the window vote joins can hold a break
  (noise on the lead read) and the same VT after it: every later run of
  ≥ 5 beats that matches the template again is kept as its own VT piece
  instead of being thrown away with the break (2026-11-16_15-35-15: VT
  15:36:58 – 15:37:21, muscle noise to 15:37:33, same VT to 15:37:56).
  A later stretch that does not match (VFib after the VT) is still cut.
- **PVC / PAC marks:** beats inside an auto VT segment are not marked V or S
  by `ectopic_beats_detector` (`NO_PVC_SEGMENTS`, `NO_PAC_SEGMENTS`): VT is a
  rhythm of its own, so it no longer reads "PVC Run". Measured: V inside VT
  segments 0 on all 22 simulator recordings with a VT; MIT-BIH PVC Se
  74.7 → 74.4%, +P 88.5 → 91.7% (34 false V gone, 3 real V in record 207's
  VT lost); INCART unchanged.

- **VT over a test signal with no QRS is dropped** (Unidentified pass): a
  VT segment ≥ 50% 'test_signal' whose beats have a global QRS
  (`pqrst_measure.measure_beats`) on < 25% of them. A 4 Hz triangle test
  wave reads 240 bpm, regular, wide, one shape — VT to the window rule —
  but has the same slope all along, so no QRS (2026-11-17_13-39-33,
  13:59:58–14:00:18: 0 of 39). The simulator's real VT also reads
  test_signal / non_ecg but its QRS is found (25–27 of 27 a frame, QRS
  250 ms), so it stays.

### 6.2 Ventricular Fibrillation — `_is_vfib_window`
Fires only when **all** of: the window's own chaos test passes, **and**
no proven bundle-branch-block window covers it (P/PR/QRS-consistent wide
beats aren't VFib), **and** `_vfib_beats_organized` is false (no lead
shows a mostly-organized QRS), **and** `_vfib_has_fibrillating_piece`
finds an actually-fibrillating sub-piece inside the window (slow-window
median RR > 600 ms is exempt from this extra check), **and**
`_vfib_regular_train_under` is false: no regular beat train runs under the
window on ≥ 2 of leads I / V1 / V6 (beats found on ≥ 2 of them within
100 ms, ≥ 5 beats, ≥ 80% of RRs within 15% of the median, median RR
≥ 400 ms, spanning ≥ 70% of the window, one QRS shape — median
correlation ≥ 0.9 on ≥ 2 of those leads, 1–25 Hz). The window's own beat
list takes extra "beats" from muscle noise on a steady rhythm (seen:
2026-11-17_11-37-06, LBBB 80 bpm under 65–150 Hz muscle noise, 21
'beats' for 13 real ones → VFib 11:38:26–36; now Unidentified, muscle).
The simulator's VFib can run at a steady ~740 ms for a few waves but its
waves are not alike (0.69–0.74), so it stays VFib. A lone
coarse/few-beat VFib window is flagged for the neighbour-confirmation
guard below (§7.1).

### 6.3 Atrial Fibrillation / Atrial Flutter — `_classify_atrial_window`
Shared context `_prepare_atrial_window_context`: needs ≥ 6 `r_peaks`;
every beat must be `measure_beat`-able; ≥ ⅔ of beats need
`qrs_bounds_resolved` (else this rule's territory doesn't apply — VFib
territory instead). Per beat, the **TP segment** (from 280 ms / 38% of
that RR after the previous beat's QRS+T, to just before this beat's P
onset or QRS) is read: `ptp < noise_threshold` ⇒ "quiet" (no wave in it,
normal sinus territory); otherwise its peaks are counted
(`find_peaks`, prominence-gated) and added to `total_waves` /
`total_active_sec`. Needs ≥ 4 beats actually scored.
- **Atrial Flutter**: `quiet_count/scored < 0.4` and
  `atrial_rate = total_waves/total_active_sec × 60` is within 0.5 of a
  whole ratio `k ∈ {2,3,4,5}` of the (scored beats') ventricular rate.
- **Atrial Fibrillation** (checked only if Flutter didn't fire):
  no "organized P" majority (≥ 60% of beats with a PR, `std(PR) ≤ 25 ms`)
  **and** the RR series shows frequent sign-reversing jumps (≥ 3
  intervals that differ from their neighbour by > 60 ms, ≥ 50% of those
  reversing direction from the previous one — genuinely irregularly
  irregular, not just variable).
- Two further vetoes back in the main loop: an atrial-regular +
  QRS-resolved-majority window is **not** called AFib/Flutter (keeps
  Mobitz I's grouped beating from reading as AFib); and
  `_atrial_beats_mostly_sinus` (beat-by-beat, all 12 leads) can veto a
  call that's really plain sinus on a lead the window rule didn't check.
  A third veto, AFib only: `_rr_repeats_as_cycle` — AFib's RR is random,
  so a window whose RR series repeats as a cycle (every RR within 4%
  / 30 ms of the RR 2–6 beats earlier, on ≥ 85% of ≥ 4 comparisons) is
  grouped beating (Mobitz I's Wenckebach cycle, bigeminy, fixed 3:2 / 4:3
  block), not AFib. It needs no P wave, so it holds where noise hides them
  (2026-11-16_14-01-13, 14:02:12–14:02:25: Mobitz I under muscle noise read
  AFib). Measured: 178 of 179 simulator recordings unchanged, MIT-BIH
  unchanged, INCART 4 records lost only AFib stretches of sinus rhythm or
  PVC bigeminy (I18, I35, I51, I53).
  A fourth veto, AFib only: **irregularity that is only PVCs**. A beat of
  another shape carries no evidence of the atrial rhythm (its RRs are its
  own coupling interval and pause; it has no P of its own), yet sinus
  rhythm with PVCs passed the RR test above on those short-long pairs, and
  at 100+/min the sinus-beat veto has no diastole to judge (MIT 208 / 215 /
  200). On the window's dominant beats only (QRS on II / V1 / V5, 1–25 Hz,
  ±80 ms, correlation ≥ 0.80 with the window's median QRS):
  `_rr_regular_without_ectopics` — ≥ 1 beat of another shape, ≥ 4 RRs
  between two dominant beats, ≥ 80% of them within 10% of their median;
  `_dominant_beats_fibrillatory` — the `_beat_by_beat_evidence` vote of the
  dominant beats whose previous beat is dominant too (an ectopic's T wave
  sits in the next diastole): ≥ 3 deciding, more than half fibrillatory.
  AFib is vetoed when the first holds and the second does not. Measured on
  every AF window of MIT-BIH / INCART (first 300 s): the RR check alone
  vetoed 13 of 305 windows inside annotated AF (MIT 210, 221: AFib's
  response looks regular over 10 dominant RRs now and then); with the P
  check, 0 of 305, while 65 of 199 windows outside annotated AF went.
  Seconds of AF label: MIT inside annotated AF 1435 → 1434, outside
  748 → 423; INCART inside 90 → 90, outside 753 → 607.
  2026-11-18_12-49-55 12:52:11–12:52:32 (PVCs under muscle noise, read
  AFib) is now Unidentified. Opens up what the false AFib covered: MIT 208
  shows plain Tachycardia and Atrial Tachycardia pieces instead (annotated
  sinus 105–119/min with PVCs and fusion beats — the Atrial Tachycardia
  pieces are false, open), INCART I57 a 3.7 s Mobitz II at 151.2–154.9 s on
  steady RR 610–620 ms (false, Mobitz II rule, open).
  Flutter additionally needs `_flutter_waves_locked` (the sawtooth is
  genuinely phase-locked to the QRS on most leads, not coincidental
  artifact).

### 6.4 Supraventricular Tachycardia (SVT) — `_is_atrial_tachycardia_window`
Checked only if neither AFib/Flutter nor an AV-block context already
claimed the window. Needs ≥ 6 beats, all measurable. Rate must be
**180–300 bpm**, `std(RR)/mean(RR) ≤ 0.08` (tightly regular). ≥ 80% of
beats organized (`qrs_bounds_resolved`). QRS shape consistency: each
beat's QRS snippet correlates ≥ 0.9 (median) with the window's own
template. A fused-wave test (counts distinct peaks between beats, using
the same TP-segment idea as AFib/Flutter) requires ≥ 70% of judged
diastoles to look "fused" (P riding on/near the T) — **except** at rate
≥ 180 bpm this fused-wave test is skipped and rate+regularity+shape alone
decide (the TP gap is too small to judge reliably that fast, and sustained
≥ 180 bpm is itself considered diagnostic).

### 6.5 AV conduction blocks (`_classify_av_block_window`)
Checked last, only once VT/VFib/AFib/Flutter/SVT all found nothing. Shares
one P-wave-train context (`_prepare_av_conduction_context`) built from
every expected P time (steady P-P), classifying each as conducted /
dropped / hidden. Tried most-severe-first: **Third-degree** (P marches
through QRS independently) → **Mobitz II** (constant PR, sudden dropped
beat) → **2:1** → **Mobitz I** (progressively lengthening PR, then a
drop — Wenckebach) → **First-degree** (PR > 200 ms, every beat conducted).
Requires a regular atrial rhythm (30–200/min discrete P waves) — AFib (no
P) and Flutter (250–350/min) structurally can't reach these rules, and
≥ 80% organized QRS keeps VFib out too.

### 6.6 Bundle branch block — its own layer, judged on every non-noisy
window *before* the chain above, but only as a **veto input** (a proven
conducted BBB window vetoes a VFib call there) and as its own separate
beat-level pass (§7's bundle-branch-block section) — it is drawn
alongside the rhythm label, never instead of it.
Its P / PR (the PR-consistency check both the window rule and the
beat-level pass need) is read the way the display reads it:
Lead II with the PR-segment spline baseline taken out
(`pqrst_measure._knot_baseline` through every beat's global QRS onset, at
≤ 120 bpm) and the P chosen by `pqrst_measure.choose_p` (the most
prominent wave instead of a strict pick 2.5× smaller, after the previous
beat's T end — estimated there as QRS onset + 0.46 s × √RR). With only the
0.5–40 Hz band-pass a 7 mV / s respiration wander left PR at 74–166 ms
beside the true 238–258 on half the beats and every RBBB window failed
(2026-11-17_12-10-06, 12:10:41–12:11:06; 2026-11-17_10-15-58,
10:20:28–53).

---

## 7. Post-window-loop assembly (roughly the order it actually runs)

1. **Lone-window guard**: a single AFib/Flutter/SVT window with no
   same-label neighbour within 2×window_size (20 s) is dropped — unless
   it's Flutter with its own beat-level confirmation (`_flutter_run_in`)
   or SVT with its own (`_svt_run_in`, re-checks `all_r_peaks` beats
   directly against the SVT window rule).
2. **Merge** same-label adjacent/near (≤ 2×window_size gap) windows into
   one segment (AV blocks use their own, different gap).
3. **Atrial boundary refine** (`_refine_atrial_edge`) — beat-exact AFib/
   Flutter region edges, only where no same-label neighbour already set it.
4. **Beat-exact edge between adjacent AFib ↔ Flutter** regions
   (`_interval_is_flutter` judged RR by RR).
5. **SVT carved out of an AFib/Flutter region** (added fix — see below).
6. **AFib/Flutter beat-by-beat split**: inside each merged AFib/Flutter
   region, every RR interval is independently judged Flutter /
   AFib / neither via the same TP-segment logic as §6.3; consecutive runs
   of ≥ 4 judged intervals of one kind become their own sub-segment;
   un-judgeable intervals continue whatever run they're inside.
7. **SVT beat-exact edges** (`_at_rr_in_band`): tightens any SVT region's
   start/end to the exact beat where 2 consecutive RRs enter/leave the
   SVT rate band (≤ 340 ms, ±15% of a locally-recomputed median).
8. **SVT ↔ AFib/Flutter boundary rule**: where an SVT region and an
   adjacent AFib/Flutter region's window-grid edges don't meet exactly
   (or there's a ≤ 1-window gap between them), the boundary is moved to
   SVT's own beat-exact edge — SVT's edge wins. **Caveat**: this rule
   assumes the gap between them is pure window-grid slop, not a *third*
   rhythm sitting in between; confirmed it can wrongly absorb a real
   different rhythm (e.g. a run of sinus-tachycardia-rate, P-wave beats)
   into the AFib/Flutter call if that run sits inside the gap.
9. **Third-degree AV block beat-exact edges**, **Ventricular Fibrillation
   gap-fill** (`_vf_runs_in_gap` — local-window chaos re-test on time no
   segment covers, ≥ `_VF_GAP_MIN_WINDOWS` consecutive 1 s windows on
   ≥ `_VF_GAP_MIN_LEADS` of a fixed lead set), **SVT gap-fill**
   (`_svt_runs_in_gap` — same idea, ≥ 15 beats / ≥ 5 s of steady
   200–333 ms RR with one QRS shape and a sharp/narrow slope, on
   `all_r_peaks` directly, only into truly unclaimed time).
10. **Ventricular Tachycardia carved out of a Ventricular Fibrillation
    call** (added fix — see below).
11. **Artifact** (generic per-lead-residual rule — currently switched
    **off**, `_ARTIFACT_RULE_ON = False`) and **Mains (50/60 Hz) Artifact**
    (RR-interval-by-RR-interval 48–52 Hz / 58–62 Hz power share on
    I/II/V1/V5; a layer label, drawn alongside the rhythm underneath it,
    not instead of it; display-side this only shows as the transient
    "wiggle" + "A" mark at its edges plus the bottom-right status text,
    not a coloured band).
12. **Bundle branch block beat-level pass** — leads I/V1/(V6 or V5), a
    beat counted only when ≥ 2 of them agree within 100 ms; each beat's
    own PR + width/shape decide Complete/Incomplete Right/Left; runs of
    ≥ 3 consistent beats (PR within 25 ms, QRS shape correlation ≥ 0.9 on
    ≥ 75%) become a segment. A VFib region is cut back off any stretch a
    window already proved was conducted BBB.
13. **Merge adjacent same-label segments**, add bundle-branch-block
    segments back in.
14. **Atrial Flutter extension at a faster conduction ratio** — lengthens
    an already-found Flutter segment, beat by beat, past its last window
    vote (fast 2:1 Flutter leaves no TP room for the window rule to judge).
15. **Sinus Arrhythmia** — its own rule, run only in time no other label
    claims: 10-beat runs through the sinus gate where RR varies (P-P
    range ≥ 120 ms) but gradually (no beat-to-beat jump > 20% of the
    run's median RR).
16. **First-degree AV Block beat-by-beat** — fills in time no other label
    claims, same PR > 200 ms / 1:1-conducted idea as the window rule,
    beat by beat.
    **Mobitz I / Mobitz II / Third-degree beat-by-beat** run right after it
    in time no other rhythm label holds (Mobitz II: one pause at a time,
    fixed P-peak-to-R PR before and after). Mobitz II is labelled only where
    a pause confirms it. **First-degree next to Mobitz II:** a First-degree
    AV Block segment within 20 s of a Mobitz II span
    (`_FIRST_NEAR_MOBII_SEC`) is that block's conducted beats, so whatever
    the PR reads that time is cut out of the First-degree label (cut to
    beats, readable signal only). It is only cut, NOT relabelled Mobitz II,
    and the cut time stays taken for the plain Bradycardia / Tachycardia
    step (those are the block's own conducted beats, not a free rate run:
    without this a Third-degree block's last slow beats next to a cut
    First-degree read a 4.6 s plain Bradycardia, 2026-11-03_12-19-03).
    Seen on 2026-11-17_10-15-58: 12 s of First-degree right before a Mobitz
    II pause and 9 s right after one. (A Mobitz II episode merge that filled
    the time between two pauses, 18 s of label for 2 pauses 15 s apart, was
    tried and taken out.)
    **Mobitz I cycle, no early beat:** a beat-by-beat Wenckebach cycle
    (≥ 3 beats with PR rising ≥ 60 ms, pause 1.3–2.0× the cycle's median RR,
    PR reset ≥ 60 ms) is not taken when its shortest RR is under
    `_M1_EARLY_FRACTION` (0.85) × the median of up to 4 RRs before it **and**
    under 0.85 × the median of up to 4 RRs after it. A conducted beat's RR
    is the P-P plus the PR's growth, never under the atrial rate; a PVC /
    PAC comes early and the rhythm goes on after its pause. Its "PR" (a P
    found in front of a beat that has none) made the PR rise and its
    compensatory pause the "blocked P": 2026-11-17_13-39-33 at 14:00:39,
    14:00:57 and 14:01:11 (PVC at 606 ms in 750 ms sinus, 0.81), INCART I20
    40.7 / 233 s (two PACs, 0.65 / 0.69), a healthy subject's sinus
    arrhythmia (2026-10-29_16-36-56 at 742 s, 0.71). The PVC detector's list
    (`_is_pvc`) did not hold those PVCs (its repeat test now keeps the ones
    at 14:00:59 and 14:01:13; 14:00:40 has every lead at the ADC rail and
    reads Q), and it holds no PAC at all.
    **PVC repeat test, leads clipped:** in `ectopic_beats_detector` a V that
    can be compared with no other V (< 3 usable leads in common, the rest
    at the ADC rail) is judged as the only V of the recording (very
    different, wide, early against the beat before it, no sinus P), not
    sent back to N as "unlike every other V". 2026-11-17_13-39-33: 11 PVCs
    of the 14:00:59–14:01:45 train back to V, and 4 of them added to the
    list `_is_pvc` reads. MIT-BIH / INCART and every other simulator /
    human recording: no beat changed.
    **PAC, P wave readable:** `_classify_pacs` takes a "P stretch unlike
    the dominant beats'" as PAC evidence only where the P wave can be read:
    the N beats around it (6 each side) match the dominant P at median
    ≥ 0.60 and this beat matches it ≥ 0.30 less. Noisy leads or the
    previous T wave make normal beats' P "different" too; with strong sinus
    arrhythmia that gave 13 false PACs on a healthy subject
    (2026-10-29_16-36-56). INCART false PAC 71 → 13 (true 226 → 216),
    MIT-BIH 304 → 9 (true 34 → 32), simulator / human recordings 19 false
    PACs fewer, none lost; PVCs and segments unchanged (`_is_pvc` reads V
    only).
    **Beat lead off:** `_recording_r_peaks` reads the lead chosen in the
    first 30 s; in a 60 s piece where it is off (peak-to-peak < 10% of its
    first 30 s) the largest lead alive there is read. An 87 min recording
    whose V4 went flat after 6 min (lead II alive) had 361 beats, now 5155
    (2026-10-30_12-01-08); INCART I02 / I58 had a lead off too.
    **Spike, not a beat:** an N "beat" with QRS < 60 ms, unlike the dominant
    beat (< 0.50), whose RR + the next beat's equal one normal RR (±10%) is
    a baseline step / spike: 'Q'. As N it made the next sinus beat read
    PAC. 10 such beats over all recordings, all artifacts.
    **PAC reset must be seen:** the recording's last beat has no next RR,
    so it is no longer taken as "reset after it".
    **Multiform PVC:** a V that matches no other V (repeat test) stays V
    on the full LITFL picture of its own: very different shape, QRS ≥ 120
    ms, early against the beat before it, no sinus P, a full compensatory
    pause (RR + next RR within 10% of 2 × the normal RR) and the dominant
    rhythm right after it (a rhythm change's first beat is followed by
    more of its own shape). MIT-BIH true PVCs 865 → 874 (false 99 → 99),
    INCART 2497 → 2523 (false 46 → 48). Tried and not taken: normal RR
    from dominant-shape beats only (MIT −24 true PVCs), a milder
    prematurity limit for clearly ventricular beats (0.92 / 0.95: as many
    false as true PVCs gained), aberrant PAC vs PVC by the P wave (our P
    measures do not separate them), a P-P / PR timing PAC rule (INCART 180
    PACs / 354 false vs 216 / 13 now).
    Unchanged: every simulator Wenckebach cycle (≥ 0.926), INCART I56
    (0.93), MIT-BIH 208 (1.06).
    **Third-degree run, one early beat:** inside the beat-by-beat Third-degree
    run a single early beat (RR ≤ 0.6× the run's median, next RR back within
    15% of it) is stepped over instead of ending the run (seen:
    2026-11-17_11-25-25, escape beats 2.0 / 2.1 / 2.0 s, one beat 0.75 s
    after the last, then 2.0 s again: the run stopped at 3 beats and the 6 s
    read plain Bradycardia; now Third-degree from 11:29:20).
    **Third-degree run through the window label:** beats a Third-degree
    window label already holds take part in the beat-by-beat Third-degree
    run (PR measured, counted toward its ≥ 4 beats); only the free beats get
    the label, joined to the window label. The window rule's first full 10 s
    window starts several beats into the block, and the 1–3 free beats
    before it used to read plain Bradycardia (seen: 2026-11-17_10-03-23,
    Mobitz II then 30 bpm escape rhythm from 10:07:33; Bradycardia
    10:07:32–40, now Third-degree from 10:07:35).
17. **Unidentified, first pass**: 10 s `frame_status` frames that aren't
    `ok` define "zones"; inside a zone, beat-interval `span_status` is
    checked (1 s filled steps where RR > 1.5 s); any non-`ok` span, minus
    time any *other* rhythm already owns (`own_rhythm` —
    `_cut_by_unreadable`, i.e. the sinus rate labels, Atrial Tachycardia
    and SVT, and layer labels excluded), becomes Unidentified and is cut
    out of those labels. SVT is in that list because it is a regular
    narrow-QRS rhythm: an unreadable stretch inside it is muscle noise /
    mains / a test signal, not the rhythm. Muscle noise is one of the
    unreadable reasons: `ecg/p_reliability.py` `muscle_noise_spans` (2 s
    blocks where the 65-150 Hz power share is ≥ 2% on Lead II, I and V5 and
    Lead II's >40 Hz std is ≥ 2% of the 10 s ECG peak-to-peak, runs of
    ≥ 4 s), read on the raw samples. A P wave cannot be read in it, whatever
    the SQI of the 10 s frame says. Atrial Flutter, the AV blocks (1st
    degree, Mobitz I / II, 3rd degree) and the bundle branch blocks yield to
    these muscle spans too (and only to them, not to the other unreadable
    reasons) -- Ventricular Tachycardia too (muscle noise inside a VT read as VT, 2026-11-17_13-39-33 14:03:03-15; the muscle part is Unidentified, the VT either side stays); Atrial Fibrillation and VFib do not. A muscle span's 2 s block needs the 65-150 Hz share >= 2% on two of Lead II / I / V5 and >= 1.5% on all three (was >= 2% on all three: a big QRS dilutes the share on its biggest lead, Lead I read 1.8-2.4% in that VT). Includes the VFib
    test-signal trim: a VFib segment whose own beat-snapped pieces read
    `test_signal` gets those pieces cut out and relabelled Unidentified
    (≥ 2 s pieces only).
18. **Plain Bradycardia / Tachycardia fallback** — rate only (no P/PR/QRS
    requirement), on beats outside every other rhythm segment: HR < 59.4
    (`_PLAIN_BRADY_HR`) or HR > 100 (`_PLAIN_TACHY_HR`), minimum 2.5 s / 1 s
    respectively. Bradycardia needs ≥ 3 beats; Tachycardia ≥ 4 beats (3
    fast RRs in a row, was 2): two fast RRs are a PAC couplet, a PVC and its
    neighbour, a spurious beat or two quick beats of sinus arrhythmia, not a
    rhythm. Every Tachycardia this removed was a 2-RR one (MIT-BIH 15,
    INCART 31, simulator / human 10); longer runs kept.
19. **Noise gate** (§2d) — the last thing that can remove a label.
20. **Unidentified, second pass** — any unreadable stretch still left
    with no rhythm label after everything above.

---

## 8. Sinus-rate rules — a separate pass, not part of the window loop

Built from `all_r_peaks` / `sinus_gate_metrics` directly (module-level
`_sinus_rate_segments`, called early, well before the window loop's
output is merged in):

- **Sinus Tachycardia**: consecutive beats with RR implying HR > a
  threshold (fast end), through `_sinus_conduction_gate`.
- **Sinus Bradycardia**: consecutive beats with
  `_BRADY_MIN_RR_MS (1010 ms) ≤ RR ≤ 6000 ms` (the 6000 ms ceiling exists
  specifically so a genuine multi-second Pan-Tompkins gap during VFib
  onset can't read as one huge "slow beat" and get mislabelled
  Bradycardia), using the **missed-beat-recovered** peak list (§3,
  item 4), through the same gate.
- **PVC beats are not sinus beats** (the PVC detector's list, `_is_pvc`):
  - A PVC the R-peak pass missed (no R peak within `_PVC_SAME_BEAT_SEC`
    0.15 s) is added to `all_r_peaks`, at the PVC detector's own R peak.
    Missing, its coupling interval and pause read as one long RR (ventricular
    bigeminy: ~2 cycles; 2026-11-18_12-49-55 12:59:44–47 read Bradycardia).
  - Its `sinus_gate_metrics` entry is an `'_ectopic'` marker with no
    measurements (the strict P search found the previous T wave in front of
    it: MIT 233, "PR" 204 ms beside sinus 136). `_sinus_beat` is False on it,
    `_sinus_conduction_gate` leaves it out of every count.
  - `_append_sinus_rate_segments` steps over up to `_ECTOPIC_BRIDGE_MAX` (2)
    of them in a row when a sinus beat follows; they don't count toward the
    ≥ 3 sinus beats. 3 in a row (a VT salvo) ends the segment.
  - `_rr_bridged_over_ectopics`: in the fast run (Sinus Tachycardia) and the
    slow run (Sinus Bradycardia) the RRs that touch such a stretch read as
    max / min of the sinus RRs on either side (two sinus beats on each side
    needed), not as the coupling interval and the pause (> 630 ms at
    100+/min, which ended the fast run at every PVC). Ventricular bigeminy
    has no sinus-sinus RR between its PVCs and is left as it is.
  - Seen before: Sinus Tachycardia on five-PVC runs (2026-11-18_12-49-55
    13:00:15, 13:00:55); MIT 233 (sinus 107/min, a PVC every ~5 s) in pieces,
    now one segment 87.7–115.1 s across three PVCs. Interpolated PVCs (INCART
    I08 119.9–123.0 s, N–V–N inside one ~1000 ms sinus cycle) read on the
    sinus RRs: Sinus Bradycardia, right.
  - Measured (first 300 s / sim 192): simulator 187 of 192 unchanged (the
    new recording's four misfires gone; others ≤ 1.5 s edge moves). MIT-BIH
    and INCART Sinus Bradycardia / Tachycardia now run across PVCs (MIT 114,
    233; INCART I05, I29, I30, I67). Side effects: an added PVC splits MIT
    214's LBBB 200.3–208.1 s in two, MIT 210's AFib starts 1.7 s later; MIT
    232 22.9–28.6 s and INCART I19 71.2–74.0 s read Bradycardia instead of
    Sinus Bradycardia (fewer than 3 sinus beats once a beat the PVC detector
    calls V is left out). GUARD unchanged.
- Both go through `_suppress_sinus_rate_overlap` against the window
  loop's `merged_morph` once that's ready: a sinus-rate segment is
  trimmed (not just dropped) at the exact edge of any overlapping
  morphological segment; a rate-segment piece sitting within 5 s of a
  morphological edge (a Pan-Tompkins transition-artifact blip, not a real
  rhythm) is dropped outright.

---

## 9. Separate: the on-screen PR / QRS / QT readout

Not part of this detector. `pqrst_measure.measure_beats()` re-measures
every beat independently (its own notch: **50 and 60 Hz both**, then a
0.5 Hz high-pass + 25 Hz low-pass on Lead II for P/PR, and
`SignalPreprocessor` + an added 60 Hz notch — `pqrst_measure.py`-local fix
— for QRS/QT's global multi-lead measurement). QT = QRS onset to the
latest T end on II, V2, V3, V5; T end by the tangent method (`_t_end`:
steepest descent of the T wave's descending limb, slope over ±10 ms,
crossing the PR baseline). `summarize()` only shows a
value when ≥ 75% of the frame's beats were measurable and their spread is
tight (`SPREAD_MS`: PR 25 ms, QRS 20 ms, QT 30 ms) — otherwise `--`.
This path does **not** feed back into the detector above; it is
display-only, cached separately (`pqrst_cache.json`).

---

## 10. Recent fixes layered on top of the above (this session)

- **Sinus-gate 50/60 Hz notch** (`_sinus_gate_filter`): Sinus
  Tachycardia/Bradycardia/Arrhythmia, First-degree beat-by-beat, and the
  Bradycardia missed-beat metrics now notch both 50 and 60 Hz before
  measuring — the rest of the detector's own preprocessing is unchanged.
- **`pqrst_measure.py` 60 Hz notch** (display-only, §9) — QRS/QT no
  longer read wrong/`--` under 60 Hz hum; detection algorithms untouched.
- **SVT carved out of an Atrial Fibrillation / Atrial Flutter call**
  (§7.5): a short SVT run (its own 10 s window never covers it alone)
  that the AFib/Flutter merge swallowed is now recovered, beat-exact,
  when ≥ 5 consecutive beats show steady 200–333 ms RR, one QRS shape,
  and a sharp/narrow QRS by slope — the same evidence the existing
  gap-fill SVT rule already required, at a shorter minimum run (5 beats,
  not 15) since this only ever carves inside an existing AFib/Flutter
  call, not free time.
- **VT carved out of a Ventricular Fibrillation call** (§7.10): a VT run
  whose own beats were diluted away by a neighbouring loud beat inside
  the window loop's padded read is now recovered using fresh, tightly
  local re-detection (§3, item 3) plus the same regularity/width gates
  the VT window rule itself uses, at ≥ 5 beats. Guarded against firing on
  a non-physiological test waveform (`span_status != 'ok'` on the
  candidate span) after a regression run caught it doing exactly that on
  a square-wave test recording.

---

## 11. Known, confirmed, not-yet-fixed limitations

- `SignalPreprocessor` (and `_atrial_preproc`) notch only 50 Hz, not 60 —
  affects the Unidentified "non_ecg" check and Atrial Flutter's
  wave-counting under 60 Hz interference. A fix is scoped but not
  applied (would need a full regression; touches many rules at once).
- Triangle/sine test waveforms that read `non_ecg` can still be called
  VFib by the main window rule, and genuine monomorphic VT can be
  indistinguishable from one by this check: leads I/V1/V5 correlating
  ≥ 0.95 (`non_ecg`) was measured directly on both a real VT run and a
  synthetic triangle-wave test recording, with no difference found in
  QRS-width consistency, flat-step fraction, or curve-vs-straight-line
  linearity between the two. The VT-carve's own `span_status != 'ok'`
  guard (added after the carve briefly labelled the triangle wave "VT")
  therefore also blocks this one genuine VT run -- kept anyway, since
  mislabelling non-physiological data as a lethal arrhythmia is the
  worse failure. That run reads plain Ventricular Fibrillation, same as
  before the VT-carve feature existed.
- `all_r_peaks`' 60 s-chunk, single-lead Pan-Tompkins can still drop a
  genuinely fast/small-amplitude run embedded in a slower chunk.
- The window loop's own `core_r_peaks` has the same amplitude-dilution
  weakness as `all_r_peaks`, just at 10 s/±2 s scale instead of 60 s —
  only the VT-carve path (§3, item 3) avoids it; nothing else does yet.
- The same-label window merge (gap ≤ 2×window_size = 20 s) can bridge
  two separate VFib windows straight over a real, different rhythm
  sitting in the gap — confirmed twice on one recording: a genuine 30 bpm
  Sinus Bradycardia run, and separately an Asystole stretch, both got
  absorbed into one continuous VFib call this way.
- A single lone (no same-label neighbour) AFib/Flutter window can be
  dropped by the lone-window guard even where the windows either side of
  it would have voted the same way, if a narrowly-failed Flutter call
  (its wave-lock check just missing) ate those votes instead.
- The beat-level bundle-branch-block run builder rejects a whole run
  when PR genuinely shifts partway through it (e.g. two different
  stable PR values in one otherwise-consistent RBBB run), instead of
  splitting it into two passing sub-runs.

### Regression guard for all of the above

`ecg/holter/corner_case_regression.py` — a standalone script (no pytest
dependency), run as `python -m ecg.holter.corner_case_regression`. Each
confirmed failure mode above is one case in it, marked `GUARD` (already
fixed; must keep passing -- a failure means a new regression) or `KNOWN`
(confirmed broken, tracked on purpose, not a new regression). Run it
after any change to the detector, `pqrst_measure.py`, or
`analysis_pipeline.py`, before considering that change done. When a
`KNOWN` case starts passing, promote it to `GUARD` in that file. New
confirmed failure modes found in the future should be added there too,
not just fixed and forgotten.

A second, complementary tool, `ecg/holter/ground_truth_regression.py`
(`python -m ecg.holter.ground_truth_regression`), scores the detector's
*whole* output against a hand-verified answer key per recording (built
by reading each recording's own beats directly, not from the detector's
own output) -- a coverage percentage per rhythm per recording, not a
single pass/fail. Spans already tracked as `KNOWN` in
`corner_case_regression.py` are deliberately left out of each answer
key's `truth` list (scoring them would just repeat the same tracked
gap) and are listed separately as `known_gap` for visibility. Use this
to see at a glance whether a change moved the needle on recordings
broader than the specific corner cases above, and to add new answer-key
recordings as they come in.
