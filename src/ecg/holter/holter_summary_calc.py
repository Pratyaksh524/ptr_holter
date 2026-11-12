import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

def label_beat_sequences(beats: List[Dict[str, Any]]) -> None:
    """
    Analyzes a sequence of beats and tags ectopic beats with their rhythm context.
    
    Detects (PVC for V beats, PAC for S beats):
      1. Single PVC/PAC (N-N-N-V-N-N-N)
      2. Couplets (2 consecutive: N-N-V-V-N-N)
      3. Triplets (3 consecutive)
      4. Quadruplets (4 consecutive)
      5. Runs (5+ consecutive: PVC Run / PAC Run)
      6. Bigeminy (N-V-N-V-N-V)
      7. Trigeminy (N-N-V-N-N-V)
      8. Quadrigeminy (N-N-N-V-N-N-N-V)
      9. Atrial Fibrillation (irregular RR + no P)
    """
    if not beats:
        return
        
    # Extract beat codes
    codes = []
    for b in beats:
        lbl = str(b.get('label', '')).upper()
        c = lbl.split('(')[1].split(')')[0] if '(' in lbl and ')' in lbl else lbl
        codes.append(c)
        
    n = len(codes)
    
    # STEP 1: Detect AFib globally (affects all beats in irregular window)
    _detect_atrial_fibrillation(beats, codes)
    
    # STEP 2: Label ectopic beat patterns
    i = 0
    while i < n:
        c = codes[i]
        if c in ['V', 'S']:
            # Find consecutive run length
            j = i + 1
            while j < n and codes[j] == c:
                j += 1
            run_len = j - i
            
            # CLASSIFY PATTERN
            if run_len == 1:
                # Single ectopic - check for bigeminy/trigeminy/quadrigeminy
                pattern = _classify_single_ectopic(codes, i, n, c)
            elif run_len == 2:
                pattern = "PVC Couplet" if c == 'V' else "PAC Couplet"
            elif run_len == 3:
                pattern = "PVC Triplet" if c == 'V' else "PAC Triplet"
            elif run_len == 4:
                pattern = "PVC Quadruplet" if c == 'V' else "PAC Quadruplet"
            else:
                # 5+ consecutive.  Was "VT Run" / "SVT Run" for 3+: the
                # rule only counts consecutive V / S beat labels (no rate,
                # no waveform), i.e. a PVC / PAC run.  The auto segment
                # detector's "Ventricular Tachycardia" is a separate rhythm
                # check on the waveform.
                pattern = "PVC Run" if c == 'V' else "PAC Run"
            # 3+ consecutive PVCs stay critical (red badge), as before
            if c == 'V' and run_len >= 3:
                for k in range(i, j):
                    beats[k]['critical'] = True
                
            # Apply pattern to all beats in run
            for k in range(i, j):
                beats[k]['rhythm_pattern'] = pattern
                beats[k]['run_length'] = run_len
                beats[k]['short_code'] = c
            i = j
        else:
            # Normal beat
            beats[i]['short_code'] = c
            beats[i]['rhythm_pattern'] = beats[i].get('rhythm_pattern', 'Normal')
            i += 1


def _classify_single_ectopic(codes: List[str], idx: int, n: int, ectopic_code: str) -> str:
    """
    Classify single ectopic beat as:
      - Bigeminy: N-V-N-V-N-V
      - Trigeminy: N-N-V-N-N-V
      - Quadrigeminy: N-N-N-V-N-N-N-V
      - Single: Isolated ectopic
    """
    # Check BIGEMINY: Every other beat is ectopic
    is_bigeminy = False
    if idx >= 2 and codes[idx-2] == ectopic_code and codes[idx-1] not in ['V', 'S']:
        is_bigeminy = True
    elif idx <= n-3 and codes[idx+2] == ectopic_code and codes[idx+1] not in ['V', 'S']:
        is_bigeminy = True
    
    if is_bigeminy:
        return "PVC Bigeminy" if ectopic_code == 'V' else "PAC Bigeminy"
    
    # Check TRIGEMINY: Every third beat is ectopic (N-N-V pattern)
    is_trigeminy = False
    if idx >= 3 and codes[idx-3] == ectopic_code:
        if codes[idx-2] not in ['V', 'S'] and codes[idx-1] not in ['V', 'S']:
            is_trigeminy = True
    elif idx <= n-4 and codes[idx+3] == ectopic_code:
        if codes[idx+1] not in ['V', 'S'] and codes[idx+2] not in ['V', 'S']:
            is_trigeminy = True
    
    if is_trigeminy:
        return "PVC Trigeminy" if ectopic_code == 'V' else "PAC Trigeminy"
    
    # Check QUADRIGEMINY: Every fourth beat is ectopic (N-N-N-V pattern)
    is_quadrigeminy = False
    if idx >= 4 and codes[idx-4] == ectopic_code:
        if all(codes[idx-k] not in ['V', 'S'] for k in range(1, 4)):
            is_quadrigeminy = True
    elif idx <= n-5 and codes[idx+4] == ectopic_code:
        if all(codes[idx+k] not in ['V', 'S'] for k in range(1, 4)):
            is_quadrigeminy = True
    
    if is_quadrigeminy:
        return "PVC Quadrigeminy" if ectopic_code == 'V' else "PAC Quadrigeminy"
    
    # Default: Single isolated ectopic
    return "Single PVC" if ectopic_code == 'V' else "Single PAC"


def _detect_atrial_fibrillation(beats: List[Dict[str, Any]], codes: List[str]) -> None:
    """
    Detect Atrial Fibrillation episodes:
      - Irregular RR intervals (CoV > 0.30)
      - No P waves (>70% absence)
      
    Marks beats within AFib episodes with 'afib_episode' flag.
    """
    if len(beats) < 12:
        return
    
    # Sliding window: 12-beat segments
    window_size = 12
    for start in range(0, len(beats) - window_size + 1, 6):  # 50% overlap
        window_beats = beats[start:start + window_size]
        
        # Calculate RR irregularity
        rr_intervals = []
        for b in window_beats:
            rr = b.get('rr_ms')
            if rr and rr > 0:
                rr_intervals.append(rr)
        
        if len(rr_intervals) < 5:
            continue
        
        rr_array = np.array(rr_intervals)
        mean_rr = np.mean(rr_array)
        std_rr = np.std(rr_array)
        cov_rr = std_rr / mean_rr if mean_rr > 0 else 0.0
        
        # Calculate Turning-Point (TP) Ratio
        tp_count = 0
        n_rr = len(rr_array)
        if n_rr > 2:
            diffs = np.diff(rr_array)
            for k in range(len(diffs) - 1):
                if diffs[k] * diffs[k+1] < 0:
                    tp_count += 1
            tp_ratio = tp_count / (n_rr - 2)
        else:
            tp_ratio = 0.0

        # Check P-wave absence
        p_absent_count = sum(1 for b in window_beats if not b.get('p_present', False))
        p_absent_ratio = p_absent_count / len(window_beats)
        
        # AFib criteria (Guide: window=12, CoV > 0.15, TP ratio > 0.75)
        # We retain P-wave absence > 70% as an additional physiological check.
        if cov_rr > 0.15 and tp_ratio > 0.75 and p_absent_ratio > 0.70:
            # Mark all beats in window as AFib
            for b in window_beats:
                b['afib_episode'] = True
                b['rhythm_pattern'] = 'Atrial Fibrillation'
                b['short_code'] = 'AF'

def calculate_ve_and_rhythm(beats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates statistical metrics for Ventricular Ectopy."""
    label_beat_sequences(beats)
    stats = {
        'single': 0, 'couplet': 0, 'bigeminy': 0, 'trigeminy': 0,
        'vt_run': 0, 'longest_vt': 0, 'max_hr': 0, 'min_hr': 0, 'total_ve': 0
    }
    
    current_run_idx = -1
    for i, b in enumerate(beats):
        if b.get('short_code') == 'V':
            stats['total_ve'] += 1
            pat = b.get('rhythm_pattern', '')
            
            if 'Single' in pat:
                stats['single'] += 1
            elif 'Couplet' in pat:
                # Add 1 couplet per 2 beats
                if i > 0 and beats[i-1].get('short_code') != 'V':
                    stats['couplet'] += 1
            elif 'Bigeminy' in pat:
                stats['bigeminy'] += 1
            elif 'Trigeminy' in pat:
                stats['trigeminy'] += 1
            elif 'Run' in pat:
                if i > 0 and beats[i-1].get('short_code') != 'V':
                    stats['vt_run'] += 1
                    rl = b.get('run_length', 0)
                    if rl > stats['longest_vt']:
                        stats['longest_vt'] = rl
                        
    return stats

def calculate_sve_and_rhythm(beats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates statistical metrics for Supraventricular Ectopy."""
    label_beat_sequences(beats)
    stats = {
        'single': 0, 'couplet': 0, 'bigeminy': 0, 'trigeminy': 0,
        'svt_run': 0, 'longest_svt': 0, 'max_hr': 0, 'min_hr': 0, 'total_sve': 0
    }
    
    for i, b in enumerate(beats):
        if b.get('short_code') == 'S':
            stats['total_sve'] += 1
            pat = b.get('rhythm_pattern', '')
            
            if 'Single' in pat:
                stats['single'] += 1
            elif 'Couplet' in pat:
                if i > 0 and beats[i-1].get('short_code') != 'S':
                    stats['couplet'] += 1
            elif 'Bigeminy' in pat:
                stats['bigeminy'] += 1
            elif 'Trigeminy' in pat:
                stats['trigeminy'] += 1
            elif 'Run' in pat:
                if i > 0 and beats[i-1].get('short_code') != 'S':
                    stats['svt_run'] += 1
                    rl = b.get('run_length', 0)
                    if rl > stats['longest_svt']:
                        stats['longest_svt'] = rl
                        
    return stats

def calculate_summary_metrics(beats: List[Dict[str, Any]], arrhythmias: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate clinical summary metrics for the Holter report."""
    metrics = {
        'total_beats': len(beats),
        'paced_beats': 0,
        'ventricular_ectopy': 0,
        'supraventricular_ectopy': 0,
        'afib_flutter_duration': 0.0,
        'total_bradycardia': 0.0,
        'total_tachycardia': 0.0
    }
    
    # 1. Count beat-level metrics
    for beat in beats:
        lbl = str(beat.get('label', '')).upper()
        if '(' in lbl and ')' in lbl:
            lbl = lbl.split('(')[1].split(')')[0]
            
        if lbl == 'P':
            metrics['paced_beats'] += 1
        elif lbl == 'V':
            metrics['ventricular_ectopy'] += 1
        elif lbl == 'S':
            metrics['supraventricular_ectopy'] += 1

    # 2. Accumulate durations from structured arrhythmia events
    for ev in arrhythmias:
        lbl = str(ev.get('label', '')).lower()
        duration = float(ev.get('end_sec', 0.0)) - float(ev.get('start_sec', 0.0))
        if duration <= 0:
            continue
            
        if 'fibrillation' in lbl or 'flutter' in lbl or 'afib' in lbl:
            metrics['afib_flutter_duration'] += duration
        elif 'bradycardia' in lbl:
            metrics['total_bradycardia'] += duration
        elif 'tachycardia' in lbl and 'ventricular' not in lbl:
            metrics['total_tachycardia'] += duration

    return metrics

def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ===========================================================================
# Stage 4 — Template Matching & Library Management
# (holter_algorithm_implementation_guide.md, Stage 4)
# ===========================================================================

def _normalize_beat_segment(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a raw beat segment for template comparison:
      1. Remove baseline: subtract median of first 30% (PR-segment approx)
      2. Scale amplitude to [-1, +1]
    """
    arr = np.asarray(arr, dtype=float)
    pre = max(1, arr.size // 3)
    arr = arr - float(np.median(arr[:pre]))
    mx  = float(np.max(np.abs(arr)))
    return arr / mx if mx > 1e-9 else arr


class _BeatTemplate:
    """
    Weighted-average beat template entry in the dynamic template library.
    Mirrors the BeatTemplate class described in the algorithm guide Stage 4.
    """
    __slots__ = ('waveform', 'label', 'count')

    def __init__(self, waveform: np.ndarray, label: str) -> None:
        self.waveform = np.asarray(waveform, dtype=float).copy()
        self.label    = label   # 'N', 'V', 'S', 'P', 'AF'
        self.count    = 1

    def update(self, new_beat: np.ndarray, weight: float = 0.95) -> None:
        """Weighted-average update: existing x weight + new x (1-weight)."""
        nb      = np.asarray(new_beat, dtype=float)
        min_len = min(self.waveform.size, nb.size)
        self.waveform = (weight       * self.waveform[:min_len] +
                         (1.0 - weight) * nb[:min_len])
        self.count += 1


def _match_against_library(
    beat_norm: np.ndarray,
    library: list,
    area_threshold: float = 0.15,
) -> tuple:
    """
    Stage 4 — Template Matching
    (holter_algorithm_implementation_guide.md, Stage 4)

    Matches beat_norm against every template in the library using:
      Primary : area-difference metric with +-10% sliding alignment
                (guide threshold: area_diff < 0.15)
      Secondary: normalized cross-correlation for confidence scoring

    Returns: (best_template, best_area_distance)
      best_template is None when no template is within area_threshold.
    """
    best_template = None
    best_distance = float('inf')

    for tmpl in library:
        t       = tmpl.waveform
        b       = beat_norm
        min_len = min(b.size, t.size)
        if min_len < 8:
            continue

        max_shift = max(1, int(0.10 * min_len))   # +-10% sliding window
        min_area  = float('inf')

        for offset in range(-max_shift, max_shift + 1):
            if offset >= 0:
                b_seg = b[offset: offset + min_len - abs(offset)]
                t_seg = t[:min_len - abs(offset)]
            else:
                b_seg = b[:min_len - abs(offset)]
                t_seg = t[abs(offset): abs(offset) + min_len - abs(offset)]

            seg_len = min(len(b_seg), len(t_seg))
            if seg_len < 4:
                continue

            area_diff = (float(np.sum(np.abs(b_seg[:seg_len] - t_seg[:seg_len])))
                         / seg_len)
            if area_diff < min_area:
                min_area = area_diff

        if min_area < best_distance:
            best_distance = min_area
            best_template = tmpl

    # Accept only if within area threshold
    if best_template is not None and best_distance <= area_threshold:
        return best_template, best_distance

    return None, best_distance


def _assess_beat_segment_quality(seg: np.ndarray, fs: float = 500.0) -> bool:
    """
    Stage 3 - Per-Beat Artifact Channel Gating
    (from holter_algorithm_implementation_guide.md, Stage 3)

    Returns True if the beat segment is usable (clean), False if it should
    be rejected as artifact.  Uses pure NumPy - no scipy dependency.

    Four checks (any failure -> artifact):
      1. Flat-line  : std(seg) < 0.01 mV  -> lead disconnection / flat signal
      2. Saturation : dynamic range > 90% of +-5 mV ADC range
      3. HF noise   : power in 40-(fs/2) Hz > 50% of QRS-band (5-15 Hz) power
      4. Baseline   : power in 0.5-1 Hz > 200% of QRS-band power

    Power is estimated with the FFT magnitude spectrum.
    """
    if seg is None or len(seg) < 8:
        return True   # no data -> don't reject (benefit of the doubt)

    arr = np.asarray(seg, dtype=float)
    n   = arr.size

    # Check 1: Flat-line
    if float(np.std(arr)) < 0.01:
        return False

    # Check 2: Saturation / clipping
    ADC_MAX_MV = 5.0   # +-5 mV typical ECG ADC range
    dyn_range  = float(np.max(arr)) - float(np.min(arr))
    if dyn_range > 0.9 * (2.0 * ADC_MAX_MV):
        return False

    # Estimate band power via FFT magnitude spectrum
    freqs   = np.fft.rfftfreq(n, d=1.0 / fs)   # Hz per bin
    mag_sq  = np.abs(np.fft.rfft(arr)) ** 2    # power per bin

    def _band_power(f_lo: float, f_hi: float) -> float:
        mask = (freqs >= f_lo) & (freqs < f_hi)
        return float(np.sum(mag_sq[mask])) if np.any(mask) else 0.0

    qrs_power = _band_power(5.0,  15.0)        # QRS band
    hf_power  = _band_power(40.0, fs / 2.0)   # muscle / EMG noise
    bw_power  = _band_power(0.5,  1.0)         # baseline wander

    if qrs_power <= 0.0:
        return True   # can't compute ratio -> don't reject

    # Check 3: High-frequency noise
    if hf_power > 0.5 * qrs_power:
        return False

    # Check 4: Baseline wander
    if bw_power > 2.0 * qrs_power:
        return False

    return True   # all checks passed -> clean beat


def _extract_morphology_features(seg: np.ndarray, fs: float = 500.0) -> dict:
    """
    Stage 5 - Morphology Feature Extraction
    Extracts QRS area, amplitude, r_polarity, symmetry, and zero crossings.
    """
    features = {
        'qrs_area': 1.0,
        'amplitude': 1.0,
        'r_polarity': 1,
        'qrs_symmetry': 1.0,
        'zero_crossings': 0
    }
    if seg is None or len(seg) < 8:
        return features

    # Median-center the array for feature extraction
    arr = np.asarray(seg, dtype=float)
    pre = max(1, arr.size // 3)
    arr_c = arr - float(np.median(arr[:pre]))

    # Peak-to-peak amplitude
    features['amplitude'] = float(np.max(arr_c)) - float(np.min(arr_c))

    # Identify approximate QRS boundaries (assume centered on R-peak max abs)
    r_idx = int(np.argmax(np.abs(arr_c)))
    features['r_polarity'] = 1 if arr_c[r_idx] > 0 else -1

    # Assume a standard ~100ms QRS window centered on R-peak
    half_qrs = int((100.0 / 2.0 / 1000.0) * fs)
    q_onset = max(0, r_idx - half_qrs)
    s_offset = min(len(arr_c), r_idx + half_qrs)

    qrs_segment = arr_c[q_onset:s_offset]
    
    # QRS Area (normalized by length so it's comparable)
    if len(qrs_segment) > 0:
        features['qrs_area'] = float(np.sum(np.abs(qrs_segment))) / len(qrs_segment)
    
        # Zero crossings
        features['zero_crossings'] = int(np.sum(np.diff(np.signbit(qrs_segment)) != 0))

    # QRS Symmetry (pre-R area / post-R area)
    pre_r_area = float(np.sum(np.abs(arr_c[q_onset:r_idx])))
    post_r_area = float(np.sum(np.abs(arr_c[r_idx:s_offset])))
    features['qrs_symmetry'] = pre_r_area / post_r_area if post_r_area > 1e-6 else 1.0

    return features


def _reclassify_beats_advanced(beats: List[Dict[str, Any]]) -> None:
    """
    Re-classify beats in-place using the full Layer 1 + Layer 2 algorithm.

    Stage 3 - Artifact Channel Gating:
      Before classification each beat segment is assessed by
      _assess_beat_segment_quality(). Beats that fail (flat-line,
      saturation, HF noise, or baseline wander) are marked
      short_code='Q' and is_artifact=True and skipped entirely -
      they do not influence the EMA baseline or get a badge.

    Layer 1 - Feature extraction per beat i:
      RRmean  : EMA over sinus (N) beats only.  alpha=0.125
                  RRmean = 0.875*RRmean + 0.125*RR(i)   (update only when beat is N)
      P(i)    : prematurity ratio  = RR(i)      / RRmean
      Q(i)    : post-beat ratio    = RR(i+1)    / RRmean  (compensatory pause proxy)
      W(i)    : QRS width in ms    = beat['qrs_ms']
      M(i)    : morphology score   = normalized cross-correlation(QRS_i, sinus_template)
                  0..1; >=0.85 -> same morphology as sinus

    Layer 2 - Classification rules (morphology outweighs timing when they conflict):
      Wide + premature  (W>120, P<0.95)             -> V
      Bizarre morph     (M<0.70, premature or wide)  -> V
      Wide + compensatory pause (P+Q ~= 2)           -> V  (even if marginally early)
      Narrow + premature + normal morph              -> S  (PAC/SVE)
      Narrow + premature + non-compensatory          -> S  (sinus-node reset)
      else                                           -> keep original label

    Only auto-detected beats are reclassified; manual beats and beats with
    'fixed_label' (labelled from the signal by ectopic_beats_detector) are
    left unchanged (but their RR still updates the EMA when they are
    labelled N), and Stage 6 skips the 'fixed_label' ones too.
    """
    if len(beats) < 3:
        return

    # ------------------------------------------------------------------
    # Step 1 – Build initial template library from pre-labeled beat segments
    # Stage 4: up to _MAX_TEMPLATES templates, one per dominant morphology
    # per beat class (N, V, S, P, AF).  Matching uses area-difference with
    # +-10% sliding alignment; eviction removes the least-used entry.
    # ------------------------------------------------------------------
    _MAX_TEMPLATES: int = 32
    library: list = []   # list of _BeatTemplate instances

    for b in beats:
        seg = b.get('segment')
        if seg is None:
            continue
        arr = np.asarray(seg, dtype=float)
        if arr.size < 8:
            continue
        lbl  = str(b.get('label', 'N')).upper()
        code = lbl.split('(')[1].split(')')[0] if ('(' in lbl and ')' in lbl) else lbl
        if code not in ('N', 'V', 'S', 'P', 'AF'):
            continue

        norm          = _normalize_beat_segment(arr)
        matched, _dist = _match_against_library(norm, library)
        if matched is not None:
            matched.update(norm)                                  # refine existing
        elif len(library) < _MAX_TEMPLATES:
            library.append(_BeatTemplate(norm, code))             # new morphology
        else:
            # Evict least-used slot (guide: "evict template with fewest beats")
            evict_idx       = min(range(len(library)), key=lambda k: library[k].count)
            library[evict_idx] = _BeatTemplate(norm, code)

    # ------------------------------------------------------------------
    # Step 2 – Initialise EMA baseline from median of all valid RR values
    # ------------------------------------------------------------------
    valid_rr = [float(b.get('rr_ms', 0.0) or 0.0)
                for b in beats if 300 < float(b.get('rr_ms', 0.0) or 0.0) < 2000]
    rr_ema = float(np.median(valid_rr)) if valid_rr else 800.0
    alpha  = 0.125   # EMA weight (0.875 × old + 0.125 × new)

    # ------------------------------------------------------------------
    # Step 3 – Walk through beats and (re)classify each one
    # ------------------------------------------------------------------
    n = len(beats)
    for i, beat in enumerate(beats):
        rr_i  = float(beat.get('rr_ms',  0.0) or 0.0)
        qrs_w = float(beat.get('qrs_ms', 0.0) or 0.0)

        # For manual beats -- and beats already classified from the signal
        # (ectopic_beats_detector, 'fixed_label') -- skip reclassification
        # but still update EMA
        if beat.get('is_manual', False) or beat.get('fixed_label', False):
            lbl  = str(beat.get('label', 'N')).upper()
            code = lbl.split('(')[1].split(')')[0] if ('(' in lbl and ')' in lbl) else lbl
            if code == 'N' and 300 < rr_i < 2000:
                rr_ema = (1 - alpha) * rr_ema + alpha * rr_i
            continue

        # ── Stage 3: Per-Beat Artifact Channel Gating ──────────────────────
        # Assess the signal quality of this beat's waveform segment.
        # Beats that fail (flat-line, saturation, HF noise, baseline wander)
        # are marked artifact and skipped so they cannot be mislabelled as
        # PVCs or contaminate the EMA sinus baseline.
        seg_for_quality = beat.get('segment')
        fs_val = float(beat.get('fs', 500.0) or 500.0)
        try:
            seg_arr = (np.asarray(seg_for_quality, dtype=float)
                       if seg_for_quality is not None else None)
            is_clean = _assess_beat_segment_quality(seg_arr, fs=fs_val)
        except Exception:
            is_clean = True  # assessment failed -> benefit of the doubt

        if not is_clean:
            beat['short_code']  = 'Q'
            beat['is_artifact'] = True
            continue  # skip Layer 1 + Layer 2 and EMA update entirely

        # --- Layer 1: timing features (always computed — used by both Stage 4 and 5) ---
        P_i = rr_i / rr_ema if (rr_ema > 0 and rr_i > 0) else 1.0

        Q_i = 1.0
        if i + 1 < n:
            rr_next = float(beats[i + 1].get('rr_ms', 0.0) or 0.0)
            Q_i = rr_next / rr_ema if (rr_ema > 0 and rr_next > 0) else 1.0

        # ── Stage 4: Template Library Matching (PRIMARY classifier) ────────
        # Try to match this beat against the dynamic library first.
        # If matched: inherit template label + apply timing refinement for N/S.
        # If not matched: fall through to Stage 5 feature-based rules.
        new_code     = None
        matched_tmpl = None
        beat_norm    = None

        seg = beat.get('segment')
        if seg is not None and library:
            try:
                arr = np.asarray(seg, dtype=float)
                if arr.size >= 8:
                    beat_norm              = _normalize_beat_segment(arr)
                    matched_tmpl, area_dist = _match_against_library(beat_norm, library)
                    if matched_tmpl is not None:
                        new_code = matched_tmpl.label
                        # Timing refinement for N/S (guide Stage 4):
                        # A template-matched N/S beat is re-labelled S when premature.
                        if new_code in ('N', 'S'):
                            new_code = 'S' if P_i < 0.85 else 'N'
            except Exception:
                pass

        # ── Stage 5: Feature-based fallback (for beats unmatched by library) ─
        # Runs only when template matching produced no result.
        # M_i is estimated from the nearest-template area distance so the
        # existing Layer 1+2 rules still work without the old sinus template.
        if new_code is None:
            # ── Stage 5: Feature Extraction & Rule-Based Classification ────────
            # Guide uses 7 features and a weighted scoring system for beats
            # that don't match any existing templates.
            fs_val = float(beat.get('fs', 500.0) or 500.0)
            morph  = _extract_morphology_features(beat.get('segment'), fs_val)
            
            # Timing features
            compensatory_ratio = (P_i + Q_i) / 2.0  # proxy for (rr_prev + rr_post) / (2 * rr_avg)
            
            score = 0.0

            # 1. QRS width contribution
            if qrs_w > 120:
                score += 2.0 * (qrs_w - 120) / 40.0
            else:
                score -= 1.0  # narrow QRS strongly favors normal/SVE

            # 2. Prematurity contribution
            if P_i < 0.80:
                # the guide says premature increases score for V/S, but the logic 
                # mainly uses score>1.5 for V, and negative score for S. 
                # For V, we want high score. Actually, the guide has: 
                # score += -1.5 * (0.80 - P_i), which lowers score! 
                score += -1.5 * (0.80 - P_i)

            # 3. Compensatory pause contribution
            if compensatory_ratio > 0.95:
                score += 1.0

            # 4. QRS Area contribution (assume typical normalized area is ~1.0)
            score += 0.8 * (morph['qrs_area'] - 1.0)

            # 5. R-wave polarity contribution
            score += -1.0 * (1 if morph['r_polarity'] < 0 else 0)

            # Decision bounds
            if score > 1.5:
                new_code = 'V'   # Ventricular
            elif P_i < 0.85 and score < 0:
                new_code = 'S'   # Supraventricular premature
            else:
                # Fallback to original if neither V nor S threshold is met
                orig_lbl  = str(beat.get('label', 'N')).upper()
                orig_code = (orig_lbl.split('(')[1].split(')')[0]
                             if ('(' in orig_lbl and ')' in orig_lbl) else orig_lbl)
                new_code  = orig_code if orig_code in ['V', 'S', 'N', 'AF', 'P'] else 'N'

        # ── Stage 4 continued: update matched template or create new one ────
        # Guide: "update_template(best_match, beat)" / "create_new_template()"
        if beat_norm is not None:
            if matched_tmpl is not None:
                matched_tmpl.update(beat_norm)               # refine existing template
            elif len(library) < _MAX_TEMPLATES:
                library.append(_BeatTemplate(beat_norm, new_code))   # new morphology
            else:
                # Evict least-used template slot
                evict_idx          = min(range(len(library)), key=lambda k: library[k].count)
                library[evict_idx] = _BeatTemplate(beat_norm, new_code)

        # Apply new classification
        beat['short_code'] = new_code
        if new_code in ('V', 'S'):
            beat['label'] = new_code

        # Update EMA baseline only for beats classified as sinus
        if new_code == 'N' and 300 < rr_i < 2000:
            rr_ema = (1 - alpha) * rr_ema + alpha * rr_i

    # =======================================================================
    # Stage 6 — Post-Classification Artifact Rejection
    # Second pass: screen for physiologically impossible sequences.
    # =======================================================================
    for i in range(1, n):
        beat = beats[i]
        code = beat.get('short_code', 'N')
        if code == 'Q' or beat.get('is_artifact', False) or beat.get('fixed_label', False):
            continue
            
        rr_i = float(beat.get('rr_ms', 0.0) or 0.0)
        
        # 1. Physiologically impossible rate
        # HR > 300 bpm (RR < 200ms) or < 15 bpm (RR > 4000ms)
        if rr_i > 0:
            hr = 60000.0 / rr_i
            if hr > 300 or hr < 15:
                beat['short_code'] = 'Q'
                beat['is_artifact'] = True
                continue
                
        # 2. Isolated V beats between normals with short coupling (< 200ms)
        if code == 'V' and i < n - 1:
            prev_code = beats[i-1].get('short_code', '')
            next_code = beats[i+1].get('short_code', '')
            if prev_code == 'N' and next_code == 'N' and rr_i < 200:
                beat['short_code'] = 'Q'
                beat['is_artifact'] = True
                continue
                
        # 3. Runs of V beats that look like noise (high variance)
        if code == 'V' and i >= 2:
            prev1_code = beats[i-1].get('short_code', '')
            prev2_code = beats[i-2].get('short_code', '')
            if prev1_code == 'V' and prev2_code == 'V':
                segs = []
                for j in (i-2, i-1, i):
                    s = beats[j].get('segment')
                    if s is not None and len(s) >= 8:
                        segs.append(np.asarray(s, dtype=float))
                
                if len(segs) == 3:
                    min_len = min(len(s) for s in segs)
                    stacked = np.stack([s[:min_len] for s in segs])
                    variance = float(np.mean(np.var(stacked, axis=0)))
                    mean_amp = float(np.mean([np.max(s) - np.min(s) for s in segs]))
                    # If variance is highly irregular compared to amplitude scale
                    if mean_amp > 1e-6 and (variance / (mean_amp**2)) > 0.2:
                        for j in (i-2, i-1, i):
                            beats[j]['short_code'] = 'Q'
                            beats[j]['is_artifact'] = True

def get_template_beats_for_badges(beats: List[Dict[str, Any]], arrhythmias: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate rich badge items for ALL ventricular ectopy (VE), supraventricular ectopy (SVE),
    atrial fibrillation, and significant clinical events.
    
    Badge Display Logic:
      - ALL V beats (PVCs) - Single PVCs, Couplets, Triplets, Quadruplets, PVC Runs, Bigeminy, Trigeminy, Quadrigeminy
      - ALL S beats (PACs/SVEs) - Single PACs, Couplets, Triplets, Quadruplets, PAC Runs, Bigeminy, Trigeminy, Quadrigeminy
      - ALL AF beats - Atrial Fibrillation episodes
      - ALL P beats - Paced beats
      - Major rhythm events from arrhythmias (VFib, VTach, AFib, AFlutter)
    
    Each badge includes PR, QRS, QT intervals and heart rate when available.
    """
    badges = []
    # Process all auto-detected beats to show ALL V, S, P, AF beats with badge labels.
    # This ensures every ectopic beat is visible, not just significant patterns.
    auto_beats = [beat for beat in beats if not beat.get('is_manual', False)]
    
    # 1. Add strictly major rhythm events from arrhythmias (AFib, AFlutter, VFib, VTach)
    for ev in arrhythmias:
        lbl = str(ev.get('label', ''))
        code = 'X'
        color = ev.get('color', '#FFFF00')
        
        lbl_lower = lbl.lower()
        is_major_rhythm = False
        
        if 'ventricular fibrillation' in lbl_lower or 'ventricular tachycardia' in lbl_lower:
            code = 'V'
            color = '#FF0000'  # Bright red for VFib/VT
            is_major_rhythm = True
        elif 'atrial fibrillation' in lbl_lower or 'atrial flutter' in lbl_lower:
            code = 'AF'
            color = '#FF00FF'  # Magenta for AFib
            is_major_rhythm = True
        elif 'bradycardia' in lbl_lower:
            code = 'BRADY'
            color = '#00CED1'  # Dark Turquoise
            is_major_rhythm = True
        elif 'tachycardia' in lbl_lower and 'ventricular' not in lbl_lower:
            code = 'TACHY'
            color = '#FFA500'  # Orange
            is_major_rhythm = True
            
        # Only add to badges if it's one of the major permitted rhythms
        if is_major_rhythm:
            badges.append({
                'timestamp': float(ev.get('timestamp', ev.get('start_sec', 0.0))),
                'code': code,
                'name': lbl,
                'color': color,
                'type': 'rhythm',
                'critical': 'ventricular' in lbl_lower
            })
            
    # 2. Re-classify auto-detected beats using full Layer 1+2 algorithm
    #    (EMA baseline, prematurity P(i), compensatory pause Q(i), QRS width W(i), morphology M(i))
    #    This upgrades the simplified analysis_worker labels before pattern sequencing.
    _reclassify_beats_advanced(auto_beats)

    # 3. Add individual ectopic beats (ALL V, S, AF beats) with rhythm context and interval data
    label_beat_sequences(auto_beats)
    
    for beat in auto_beats:
        # Stage 3 gate: skip artifact beats entirely — no badge rendered
        if beat.get('is_artifact', False) or beat.get('short_code', '') == 'Q':
            continue

        code = beat.get('short_code', '')
        pattern = beat.get('rhythm_pattern', '')
        beat_color = beat.get('color', '')

        # Show badges for:
        # 1. Explicitly ectopic beats (V, S, P, AF codes)
        # 2. Beats with non-green colors (indicating abnormal detection)
        # 3. Beats with patterns assigned (Bigeminy, Couplet, etc.)
        is_ectopic = code in ['V', 'S', 'P', 'AF']
        is_colored_abnormal = beat_color and beat_color not in ['#00FF00', '#FFFFFF', '']
        has_pattern = pattern and pattern not in ['Normal', '']

        # Show badge if ANY of these conditions are true
        if not (is_ectopic or is_colored_abnormal or has_pattern):
            continue
        
        # Default pattern if not set
        if not pattern:
            if code == 'V':
                pattern = 'PVC'
            elif code == 'S':
                pattern = 'PAC'
            elif code == 'P':
                pattern = 'Paced'
            elif code == 'AF':
                pattern = 'AFib'
            else:
                pattern = 'Abnormal Beat'
        
        # Determine color and priority based on pattern
        color, priority = _get_badge_appearance(code, pattern, beat)

        # Suppress white default/normal badges (e.g. [TACHY] Normal)
        if not color or color.upper() in ['#FFFFFF', '#FFF', 'WHITE']:
            continue
        
        # Extract PR, QRS, QT/QTc if available from auto-detection results
        qrs = beat.get('qrs_ms')
        qt = beat.get('qt_ms')
        pr = beat.get('pr_ms')
        hr = beat.get('heart_rate_bpm')
        
        details = []
        if qrs is not None and qrs > 0:
            details.append(f"QRS:{int(qrs)}")
        if qt is not None and qt > 0:
            details.append(f"QT:{int(qt)}")
        if pr is not None and pr > 0:
            details.append(f"PR:{int(pr)}")
        if hr is not None and hr > 0:
            details.append(f"HR:{int(hr)}")
            
        detail_str = " ".join(details)
        if detail_str:
            full_name = f"{pattern} | {detail_str}"
        else:
            full_name = pattern
            
        badges.append({
            'timestamp': float(beat.get('timestamp', 0.0)),
            'code': code,
            'name': full_name,
            'color': color,
            'type': 'beat',
            'priority': priority,
            'critical': beat.get('critical', False),
            'run_length': beat.get('run_length', 1)
        })
            
    badges.sort(key=lambda x: x['timestamp'])
    return badges


def _get_badge_appearance(code: str, pattern: str, beat: Dict[str, Any]) -> tuple:
    """
    Determine badge color and priority based on beat type and pattern.
    
    Returns: (color_hex, priority_level)
      priority: 0=normal, 1=minor, 2=moderate, 3=critical
    """
    # CRITICAL: 3+ consecutive PVCs (PVC Triplet / Quadruplet / Run; the
    # 'critical' flag from label_beat_sequences)
    if ('PVC Run' in pattern and code == 'V') or beat.get('critical', False):
        return '#FF0000', 3  # Bright red, highest priority
    
    # HIGH: PVC patterns
    if code == 'V':
        if 'Bigeminy' in pattern or 'Trigeminy' in pattern or 'Quadrigeminy' in pattern:
            return '#FF6600', 2  # Orange-red, high priority
        elif 'Couplet' in pattern:
            return '#FF3333', 2  # Red, high priority
        else:  # Single PVC
            return '#FF6666', 1  # Light red, moderate priority
    
    # MODERATE: PAC patterns
    if code == 'S':
        if 'Run' in pattern or 'Triplet' in pattern or 'Quadruplet' in pattern:
            return '#00CCFF', 2  # Bright cyan (3+ consecutive PACs, as before)
        elif 'Bigeminy' in pattern or 'Trigeminy' in pattern:
            return '#00FFFF', 2  # Cyan
        elif 'Couplet' in pattern:
            return '#66FFFF', 1  # Light cyan
        else:  # Single PAC
            return '#99FFFF', 1  # Very light cyan
    
    # SPECIAL: Atrial Fibrillation
    if code == 'AF' or beat.get('afib_episode', False):
        return '#FF00FF', 2  # Magenta, high priority
    
    # PACED
    if code == 'P':
        return '#CC66FF', 1  # Purple, moderate priority
    
    # Default
    return '#FFFFFF', 0  # White, normal

