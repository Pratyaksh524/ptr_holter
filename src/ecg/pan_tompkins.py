import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d


def pan_tompkins(ecg, fs=500):
    """
    Pan-Tompkins QRS detection algorithm implementation.
    Args:
        ecg: 1D numpy array of ECG signal
        fs: Sampling frequency (Hz)
    Returns:
        r_peaks: Indices of detected R peaks (aligned to the bandpassed ECG,
                 not the integrated envelope).
    """
    # 1. Bandpass filter (5-15 Hz) - classic QRS band.
    # Use filtfilt (zero-phase) so peak indices are not delayed by IIR phase shift.
    def bandpass_filter(signal, lowcut, highcut, fs, order=2):
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.99)
        if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high >= 1 or low >= high:
            return signal
        b, a = butter(order, [low, high], btype='band')
        # If signal is too short for filtfilt padding, fall back to causal filtering
        try:
            if len(signal) < max(len(a), len(b)) * 3:
                from scipy.signal import lfilter
                return lfilter(b, a, signal)
            return filtfilt(b, a, signal)
        except Exception:
            return signal

    x = np.asarray(ecg, dtype=float)
    if x.size < 10 or not np.isfinite(fs) or fs <= 0:
        return np.array([], dtype=int)

    filtered = bandpass_filter(x, 5, 15, fs)
    # 2. Differentiate
    diff = np.ediff1d(filtered, to_end=0.0)
    # 3. Square
    squared = diff ** 2
    # 4. Moving window integration (150 ms window)
    window_size = max(1, int(round(0.15 * fs)))
    if window_size > squared.size:
        window_size = max(1, squared.size // 4)
    mwa = np.convolve(squared, np.ones(window_size) / float(window_size), mode='same')

    # 5. Find peaks (adaptive threshold with fallbacks)
    from scipy.signal import find_peaks

    # FIX-PT1: Adaptive refractory period - prevents T-wave double-detection.
    # FIX-PT2: Was 0.20*fs (200ms) = exactly RR at 300 BPM -> find_peaks
    # missed every other peak -> false 150 BPM.
    # Now 0.16*fs (160ms at 500Hz = 80 samples) -> supports up to ~375 BPM.
    # Pass-2 below still adapts refractory for low HR to kill T-wave ghosts.
    min_distance = max(1, int(round(0.22 * fs)))
    mean_mwa = float(np.mean(mwa)) if mwa.size else 0.0
    std_mwa = float(np.std(mwa)) if mwa.size else 0.0

    peaks = np.array([], dtype=int)
    for k in (0.50, 0.30, 0.15, 0.10):
        thr = mean_mwa + k * std_mwa
        try:
            peaks, _ = find_peaks(mwa, height=thr, distance=min_distance)
        except Exception:
            peaks = np.array([], dtype=int)
        if peaks.size >= 2:
            break

    # FIX-PT1 continued: if HR looks < 100 BPM, rerun with tighter refractory
    # to kill T-wave false peaks that appear at ~0.33*RR after true R.
    if peaks.size >= 3:
        ipi = np.diff(peaks) / fs * 1000.0  # inter-peak intervals in ms
        median_ipi = float(np.median(ipi))
        if median_ipi > 600:  # HR < 100 BPM
            adaptive_refrac = max(int(round(0.22 * fs)),
                                  int(round(0.33 * median_ipi / 1000.0 * fs)))
            if adaptive_refrac > min_distance:
                min_distance = adaptive_refrac
                peaks = np.array([], dtype=int)
                for k in (0.50, 0.30, 0.15, 0.10):
                    thr = mean_mwa + k * std_mwa
                    try:
                        peaks, _ = find_peaks(mwa, height=thr, distance=min_distance)
                    except Exception:
                        peaks = np.array([], dtype=int)
                    if peaks.size >= 2:
                        break

    # FIX-PT3: hole rescue. Everything above this point is unchanged from
    # the original algorithm (one mean+std threshold over the whole array
    # handed to this call) -- a call whose peaks have no large gap below
    # behaves exactly as before, bit for bit.
    #
    # The one problem that single whole-array threshold has: a stretch of
    # much larger beats elsewhere in the SAME array raises it high enough
    # that real, normal-looking beats in a smaller-amplitude stretch fall
    # under it and are skipped outright -- not just measured wrong, missed
    # completely, for as long as that stretch lasts. Confirmed directly: a
    # recording whose amplitude dropped for a while lost several
    # consecutive beats in a 10s window straddling the drop, and an entire
    # run of beats in a 60s window straddling it, cutting a rhythm label
    # short by tens of seconds.
    #
    # Rescue only runs inside an actual gap between two already-trusted
    # peaks (HOLE_SEC or longer), and only keeps a recovered peak that is
    # BOTH:
    #   - tall enough, and
    #   - part of a regular run, or (if alone) tall enough on a much
    #     higher bar.
    # A genuine pause (AV block, sinus pause) has no QRS-shaped energy in
    # it at all, so both checks correctly find nothing to recover there;
    # confirmed unchanged against the Mobitz I / Mobitz II / Third-degree
    # pause GUARD cases in corner_case_regression.py.
    HOLE_SEC = 1.5
    ADAPT_SUBWIN_SEC = 1.2
    # A single, isolated recovered beat must clear a fairly high bar
    # (LONE_FLOOR_FRAC) -- calibrated against a real AV-block pause
    # (2026-11-03_10-25-10_Unknown, the quiet stretch around its Mobitz II
    # pauses): ordinary baseline / T-wave noise there peaks at ~8.4% of a
    # real beat's height even with nothing actually happening, so one
    # recovered candidate on its own needs real margin above that to not
    # be that same kind of noise. A RUN of 2+ candidates with regular
    # spacing between each other (RUN_RR_CV_MAX) is held to a much lower
    # floor instead (RUN_FLOOR_FRAC) -- that regularity is itself strong
    # evidence of a real, continuing QRS train (exactly the amplitude-test
    # recording's missed runs, beats at a steady RR down to a few percent
    # of a much larger neighbouring beat's height), which lone noise does
    # not reliably produce.
    #
    # Both floors are fractions of the TRUSTED peak(s) immediately beside
    # the one gap being rescued, not one reference for the whole call -- an
    # amplitude change is sometimes a staircase, not one clean step
    # (confirmed: 100% -> ~16% -> ~3% across a single 60s chunk of one
    # recording), and a single whole-call reference, set by whichever tier
    # happens to be largest, is too strict for a step further down the
    # stairs than the nearest one. Rescuing in PASSES, each one using the
    # previous pass's own newly-rescued points as trusted peaks too, lets
    # each step's floor be set from the step right beside it rather than
    # from a potentially much larger one further away.
    LONE_FLOOR_FRAC = 0.12
    # A run's floor is lower than the lone floor -- the run-consistency
    # check above (>= 2 candidates at a steady RR) is real additional
    # evidence a lone candidate doesn't have. It was tried much lower
    # (0.005, on the reasoning that a real AV-block pause's internal noise
    # never showed consistent RR between any two of its own bumps, so the
    # run check alone looked like enough of a guard) to reach a recording
    # whose amplitude fell to ~0.7% of a nearby trusted beat. That broke a
    # different, more dangerous GUARD case: a square-wave test signal is
    # ITSELF perfectly regular by construction (it is a repeating
    # waveform, not noise), so a low enough floor lets the run check pass
    # on test data too, and that recording's existing test-signal guard
    # (span_status) did not catch every instance -- it read as Ventricular
    # Tachycardia, a real-arrhythmia label on non-physiological data,
    # which this codebase treats as the one failure mode worse than a
    # missed beat. Kept at 4%: safe against the full GUARD suite; a
    # recording with a stretch under about 4% of its nearest trusted
    # beat's height is a known, narrower remaining gap (see docs/notes),
    # not something this floor should be pushed lower to chase.
    RUN_FLOOR_FRAC = 0.04
    RUN_RR_CV_MAX = 0.25
    EDGE_PAD_SEC = 0.3
    MAX_RESCUE_PASSES = 8
    hole_min_gap = max(int(round(HOLE_SEC * fs)), min_distance * 2)
    pad = max(1, int(round(EDGE_PAD_SEC * fs)))

    for _pass in range(MAX_RESCUE_PASSES):
        if not peaks.size or mwa.size < 3:
            break
        trusted = np.sort(peaks)
        bounds = [0] + [int(p) for p in trusted] + [mwa.size - 1]
        rescued = []
        for i in range(len(bounds) - 1):
            g0, g1 = bounds[i], bounds[i + 1]
            if g1 - g0 < hole_min_gap:
                continue
            edge_before = [int(trusted[i - 1])] if i >= 1 else []
            edge_after = [int(trusted[i])] if i <= len(trusted) - 1 else []
            edge_heights = [float(mwa[e]) for e in (edge_before + edge_after)]
            local_ref = min(edge_heights) if edge_heights else 0.0
            if local_ref <= 0:
                continue
            amp_floor = RUN_FLOOR_FRAC * local_ref
            lone_floor = LONE_FLOOR_FRAC * local_ref

            lo, hi = max(0, g0 - pad), min(mwa.size, g1 + pad + 1)
            seg = mwa[lo:hi]
            if seg.size < 3:
                continue
            # A rolling local threshold WITHIN the gap, not one single
            # mean+std over the whole gap -- a gap can itself contain a
            # ramp (amplitude climbing back up through it, not a flat
            # small-and-steady stretch), and one number for the whole gap
            # then gets pulled up by its own far (larger) end, putting its
            # near (still small) end back under threshold, i.e. exactly
            # the bias this function exists to remove, just rescaled down
            # to the gap's own span instead of the whole call's.
            subwin = max(3, min(seg.size, int(round(ADAPT_SUBWIN_SEC * fs))))
            if subwin >= 3:
                local_mean = uniform_filter1d(seg, size=subwin, mode='nearest')
                local_mean_sq = uniform_filter1d(seg * seg, size=subwin, mode='nearest')
                local_std = np.sqrt(np.maximum(local_mean_sq - local_mean * local_mean, 0.0))
            else:
                local_mean = np.full(seg.shape, float(np.mean(seg)))
                local_std = np.full(seg.shape, float(np.std(seg)))
            # The most permissive multiplier (0.10, matching the last
            # fallback used above): amp_floor and the run check below are
            # what keeps this safe, not this multiplier -- an early k that
            # only turns up a peak right at the gap's own edge (re-finding
            # a trusted peak this gap already starts or ends on) would
            # mean the real recoverable beats further in never get tried.
            thr = np.maximum(local_mean + 0.10 * local_std, amp_floor)
            try:
                cand, _ = find_peaks(seg, height=thr, distance=min_distance)
            except Exception:
                cand = np.array([], dtype=int)
            if cand.size == 0:
                continue
            cand_global = sorted(int(lo + c) for c in cand if g0 < lo + c < g1)
            cand_global = [c for c in cand_global if mwa[c] >= amp_floor]
            if not cand_global:
                continue
            # A run is >= 2 RECOVERED CANDIDATES with regular spacing
            # BETWEEN THEMSELVES -- deliberately not counting the trusted
            # peaks at the gap's own edges as run members: a single lone
            # candidate roughly splits the gap, so both of its distances to
            # the two edges end up similar to each other near-automatically
            # (that pair trivially "passes" a same-ness check on its own,
            # regardless of whether the candidate is a real beat), which
            # once let exactly one noise bump inside a genuine AV-block
            # pause through as a false "run" (2026-11-03_10-25-10_Unknown,
            # Third-degree AV Block run, confirmed against the GUARD case
            # for it). A candidate with no neighbouring candidate is always
            # lone and must clear the much higher lone_floor by itself.
            in_run = [False] * len(cand_global)
            if len(cand_global) >= 2:
                rrs = [(b - a) / fs for a, b in zip(cand_global, cand_global[1:])]
                med_rr = float(np.median(rrs))
                if med_rr > 0:
                    regular = [abs(r - med_rr) / med_rr <= RUN_RR_CV_MAX for r in rrs]
                    # regular[j] is whether the RR between cand_global[j]
                    # and cand_global[j+1] is regular; a candidate is in a
                    # run if the RR to its candidate neighbour on either
                    # side is regular.
                    for c_idx in range(len(cand_global)):
                        left_ok = c_idx - 1 >= 0 and regular[c_idx - 1]
                        right_ok = c_idx < len(regular) and regular[c_idx]
                        in_run[c_idx] = bool(left_ok or right_ok)
            cand_global = [c for c, ok in zip(cand_global, in_run) if ok or mwa[c] >= lone_floor]
            if not cand_global:
                continue
            # A candidate's own membership in a run (checked above, against
            # its immediate neighbours) is the evidence it is a genuine
            # beat -- no separate check against the RR to the real peak
            # right at the gap's edge: that edge beat can be mid ramp back
            # up to full amplitude/rate, not yet at the same RR as deeper
            # in the gap. Step 7's own de-dup (largest |amplitude| wins
            # within the refractory period) still protects against a
            # recovered candidate crowding a real peak right at the
            # boundary.
            rescued.extend(cand_global)
        if not rescued:
            break
        peaks = np.array(sorted(set(peaks.tolist()) | set(rescued)), dtype=int)

    if peaks.size == 0:
        return np.array([], dtype=int)

    # 6. Search-back: map each envelope peak to the true R-peak location
    # by finding max(|bandpassed ECG|) within +/-75 ms.
    search_half = max(1, int(round(0.075 * fs)))
    r_locs = []
    for p in peaks:
        left = max(0, int(p) - search_half)
        right = min(filtered.size, int(p) + search_half + 1)
        if right <= left:
            continue
        seg = filtered[left:right]
        # Use abs() to handle inverted leads (R can be negative)
        r = left + int(np.argmax(np.abs(seg)))
        r_locs.append(r)

    if not r_locs:
        return np.array([], dtype=int)

    r_locs = np.array(sorted(set(r_locs)), dtype=int)

    # 7. De-duplicate R-peaks within refractory period: keep the larger |amplitude|
    dedup = []
    for r in r_locs:
        if not dedup:
            dedup.append(int(r))
            continue
        if int(r) - dedup[-1] < min_distance:
            prev = dedup[-1]
            if abs(filtered[int(r)]) > abs(filtered[int(prev)]):
                dedup[-1] = int(r)
        else:
            dedup.append(int(r))

    # FIX-PT4: beats at a beat-size step, ADD-ONLY. Every beat found above is
    # kept exactly as it is; this only adds beats the steps above missed
    # because a much bigger stretch of beats in the same array set the
    # threshold (see _beats_at_size_steps). An array with no size step gets
    # nothing added -- identical to before.
    added = _beats_at_size_steps(filtered, mwa, fs, dedup)
    if added:
        dedup = sorted(dedup + added)
    return np.array(dedup, dtype=int)


# Beat-size step detection (FIX-PT4). Values measured on recorded data
# (2026-11-04, amplitude-test recordings 12-31-46 and 14-14-07 plus 18
# recordings with Asystole):
STEP_RATIO = 1.6        # per-second beat size changes >= 1.6x (or <= 1/1.6)
STEP_HOLD_SEC = 2       # ... and the new size holds (+-35%) for the next 2 s,
STEP_HOLD_TOL = 0.35    # so one spike / artifact does not split anything
STEP_MIN_PIECE_SEC = 3  # two splits at least 3 s apart
STEP_EDGE_SEC = 1.0     # each piece is searched 1 s past its edges (a step
                        # falls mid-second; the beat right at it is covered)
# A piece whose median per-second QRS-band (5-15 Hz) size is under this is a
# flat line (Asystole, lead off): nothing is added there. Measured: Asystole
# 0-3.8, the smallest real beats in these recordings 116-296 (QRS 2-11 mV);
# 20 is about 0.3 mV QRS peak-to-peak at this device's 200 ADC/mV. Without it
# 11-50 fake beats were added inside Asystole on 4 recordings.
STEP_MIN_SIZE = 20.0
# A piece whose QRS-band signal is "active" (over 30% of its own top) more
# than this fraction of the time is a continuous wave, not beats with a
# quiet baseline between them -- a triangle / sine test signal. Nothing is
# added there (that piece keeps exactly the beats found before). Measured:
# triangle 63%, sine 80%; real beats 10-33% (sinus at every amplitude and
# rate tried, AFlutter, VFib), VT 48-50%. Without it the triangle test wave
# on 2026-10-31_15-33-42 got a regular 240 bpm "beat" on every rise and read
# as Ventricular Tachycardia (GUARD: square-wave test signal -> no VT).
STEP_MAX_ACTIVE = 0.55
# Only a piece whose own beats come at a steady rate gets beats added (RR
# spread / mean <= this). An organized rhythm whose size stepped (sinus,
# tachycardia, SVT, flutter, VT, regular escape) qualifies; a chaotic or
# irregular piece (VFib, AFib, dropped beats) keeps exactly the beats found
# before. Seen without it: VF on MIT-BIH 207 has its own size swings; beats
# were added inside it and VF duration sensitivity fell 27.7% -> 6.2%.
STEP_MAX_RR_CV = 0.20


def _beats_at_size_steps(filtered, mwa, fs, existing):
    """Beats (sample indices) to ADD to `existing`. The array is cut where
    the beat size steps up or down; each piece gets its own threshold (the
    same mean + k*std rule and k fallbacks as pan_tompkins, from that
    piece's own envelope only), so a small-amplitude stretch is no longer
    judged against a big one elsewhere in the same array. Seen: amplitude
    dropped to ~1/3, then to ~0.2 of the beats around it on a simulator
    recording -- 49 real, regular beats missed (one right after each step,
    32 in a row in the deepest dip, 15 at the end of a 60 s chunk).

    Candidates are cleaned before use: same QRS (< 0.10 s) keeps the bigger
    point; the classic Pan-Tompkins T-wave rule (within 0.36 s of the last
    beat and under half its slope = its T wave) unless the gap matches the
    following RR (+-25%: a small beat after a big one at a fast rate);
    inside the 0.22 s refractory the narrower wave (slope relative to its
    own height) is the QRS -- a beat at the exact moment of a step can have
    a small QRS and a big T. A candidate within 0.22 s of an existing beat
    is not added."""
    n_sec = int(len(filtered) // fs)
    if n_sec < STEP_HOLD_SEC + 2 or not len(existing):
        return []
    sec = int(fs)
    size = np.array([float(np.max(np.abs(filtered[k * sec:(k + 1) * sec]))) for k in range(n_sec)])
    steps = []
    for k in range(1, n_sec - STEP_HOLD_SEC):
        if size[k - 1] <= 0 or size[k] <= 0:
            continue
        ratio = size[k] / size[k - 1]
        if ratio >= STEP_RATIO or ratio <= 1.0 / STEP_RATIO:
            hold = size[k:k + STEP_HOLD_SEC + 1]
            if np.all(np.abs(hold - size[k]) <= STEP_HOLD_TOL * size[k]):
                if not steps or k - steps[-1] >= STEP_MIN_PIECE_SEC:
                    steps.append(k)
    if not steps:
        return []

    from scipy.signal import find_peaks
    bounds = [0] + [k * sec for k in steps] + [len(filtered)]
    edge = int(round(STEP_EDGE_SEC * fs))
    near = max(1, int(round(0.12 * fs)))
    found = []
    skipped = []
    for s, e in zip(bounds, bounds[1:]):
        piece_size = [size[k] for k in range(s // sec, min(n_sec, max(s // sec + 1, e // sec)))]
        if not piece_size or float(np.median(piece_size)) < STEP_MIN_SIZE:
            skipped.append((s, e))
            continue
        mag = np.abs(filtered[s:e])
        if mag.size and float(np.mean(mag > 0.3 * float(np.percentile(mag, 99)))) > STEP_MAX_ACTIVE:
            skipped.append((s, e))
            continue
        core = mwa[s:e]
        if core.size < 3:
            continue
        lo, hi = max(0, s - edge), min(len(mwa), e + edge)
        pk = np.array([], dtype=int)
        for k in (0.50, 0.30, 0.15, 0.10):
            try:
                pk, _ = find_peaks(mwa[lo:hi], height=float(np.mean(core)) + k * float(np.std(core)), distance=near)
            except Exception:
                pk = np.array([], dtype=int)
            if pk.size >= 2:
                break
        own = np.sort(np.asarray([lo + p for p in pk if s <= lo + p < e], dtype=float))
        rr = np.diff(own)
        if rr.size < 2 or float(np.std(rr)) > STEP_MAX_RR_CV * float(np.mean(rr)):
            skipped.append((s, e))
            continue
        found.extend(int(lo + p) for p in pk)
    # A neighbour's 1 s edge search must not add beats inside a skipped piece.
    found = [p for p in found if not any(a <= p < b for a, b in skipped)]
    if not found:
        return []

    sh = max(1, int(round(0.075 * fs)))
    slope = np.abs(np.gradient(filtered))
    cands = set()
    for p in sorted(set(found)):
        a, b = max(0, p - sh), min(len(filtered), p + sh + 1)
        r = a + int(np.argmax(np.abs(filtered[a:b])))
        w0, w1 = max(0, r - sh), min(len(filtered), r + sh + 1)
        sl = float(slope[w0:w1].max())
        height = float(np.ptp(filtered[w0:w1])) or 1.0
        cands.add((r, sl, sl / height))
    cands = sorted(cands)

    refractory = int(round(0.22 * fs))
    same_qrs = int(round(0.10 * fs))
    t_window = int(round(0.36 * fs))
    keep = []
    for i, (r, sl, sharp) in enumerate(cands):
        if keep:
            pr, psl, psharp = keep[-1]
            gap = r - pr
            ahead = [cands[j][0] - cands[j - 1][0] for j in range(i + 1, min(len(cands), i + 7))]
            ahead = [g for g in ahead if g >= refractory]
            ref = float(np.median(ahead)) if ahead else None
            on_rhythm = ref is not None and abs(gap - ref) <= 0.25 * ref
            if gap < t_window and sl < 0.5 * psl and not on_rhythm:
                continue
            if gap < same_qrs:
                if abs(filtered[r]) > abs(filtered[pr]):
                    keep[-1] = (r, sl, sharp)
                continue
            if gap < refractory:
                if sharp > psharp:
                    keep[-1] = (r, sl, sharp)
                continue
        keep.append((r, sl, sharp))

    ex = np.asarray(sorted(existing), dtype=int)
    added = []
    for r, _, _ in keep:
        j = int(np.searchsorted(ex, r))
        nearest = min(abs(r - ex[q]) for q in (j - 1, j) if 0 <= q < len(ex))
        if nearest > refractory:
            added.append(int(r))
    return added


def detectRPeaks(filtered_signal, fs=500):
    """
    Backward-compatible alias for older callers that still import
    detectRPeaks from this module.
    """
    return pan_tompkins(filtered_signal, fs=fs)
