"""Beat-by-beat ectopic labels (PVC = 'V') for the whole recording, from the
recorded signal.  Separate from the live per-chunk labels (analysis_worker.
_label_beat, metrics.jsonl all_beats) and from the auto arrhythmia segments:
it does not depend on the "Auto Arrhythmia Detection" button.

Why not the live labels: on a ProSim ventricular trigeminy (2026-11-06_15-24-52,
-15-46-16) they kept only the PVCs -- the normal beats' QRS width measured
28-30 ms on one lead (true ~90) and the QRS validator drops < 40 ms -- and
labelled every kept beat V because the 5 s chunk was read as VFib; the PVC
itself measured 88 ms (true ~160) and a beat at a 5 s chunk edge was dropped.
The pattern labels (holter_summary_calc.label_beat_sequences) then read
V V V V as a run instead of N N V N N V (trigeminy).

  - Beats: Pan-Tompkins on the best lead in 60 s pieces with 3 s of context
    (the auto segment detector's R-peak pass, same helpers), then a
    search-back where one RR is >= 1.6x the local RR: a QRS another lead
    shows there is added (seen: a PVC that is mostly negative in lead II
    missed by Pan-Tompkins on 2 of 6 trigeminy cycles).
  - QRS width: pqrst_measure's global QRS across leads I, II, aVL, V1, V2,
    V5, V6 (PVC 160 ms, sinus 90 ms on the trigeminy recordings), retried
    50 / 100 ms either side of the R-peak when not bounded there. Leads
    clipped at the ADC rail around the beat are left out.
  - Shape: the beat's QRS on those leads (-100..+160 ms), each lead scaled
    to unit length, compared lead by lead with a running dominant-beat
    template (best over +-20 ms shifts); the similarity is the mean over
    the usable leads (see _similarity for why not the median).
  - V when the beat is premature (RR <= PREMATURE_FRACTION x the running RR
    of normal beats) and its shape differs from the dominant beat
    (similarity < SHAPE_SAME_CORR) with a QRS >= WIDE_QRS_MS -- or the shape
    is very different (< VERY_DIFFERENT_CORR) with a QRS >= MIN_PVC_QRS_MS.
    A beat right after a V that meets the same shape test and is still
    faster than the normal RR continues the run. A run that turns out
    slower than 100 bpm with a steady RR is a new rhythm (e.g. conducted
    bundle branch block after a slow rhythm), so its beats go back to N.
    Not V: a wide beat at the normal rate (bundle branch block, paced,
    escape, a late PVC -- tried, as many false as true on INCART) or a wide
    beat with the dominant beat's shape.
  - Template: started from the first beats (the narrow ones when there are
    enough, so a bigeminy start does not take the PVC for the template),
    moved only by N beats with similarity >= TEMPLATE_UPDATE_CORR, and
    replaced when TEMPLATE_SWITCH_BEATS beats in a row at the normal rate
    share a new shape (a lasting change of conduction). The normal RR
    follows dominant-shape beats both faster and slower, except the pause
    after a PVC. Template and normal RR are learnt again after a gap >
    RESTART_GAP_SEC or a no-QRS segment (below), never from the beats of an
    auto Ventricular Tachycardia segment (the old ones are kept when
    nothing else is there).
  - Beats inside an auto segment where no QRS can be trusted (Ventricular
    Fibrillation, Asystole, Artifact, Unidentified), with fewer than
    MIN_SHAPE_LEADS usable leads, or (when not V) with no QRS bounds get
    'Q' (not classified). Everything else not V is 'N'.

  Scored against cardiologist annotations (first 5 min of each record):
  INCART 12-lead and MIT-BIH (2 leads; the other 10 are mixes of them, so
  the shape test is weaker) -- see the PVC Se / +P in the change notes.
  Known limit: heavy motion artifact the auto segment detector does not
  mark as Artifact can still give single false PVCs (2026-10-29_16-36-56:
  10 in 31 min, the live labels gave 136). A late PVC (not premature) is
  not found.
"""

import copy
import hashlib
import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np

CACHE_FILENAME = 'ectopic_beats_cache.json'

PIECE_SEC = 60.0
PAD_SEC = 3.0
MIN_BEAT_GAP_SEC = 0.20
SEARCHBACK_RR_RATIO = 1.6        # an RR this many times the local RR is searched
SEARCHBACK_MIN_SLOPE = 0.35      # a QRS there: combined slope >= this x neighbours'
LOCAL_RR_BEATS = 8

WIDE_QRS_MS = 110.0
SHAPE_SAME_CORR = 0.80
VERY_DIFFERENT_CORR = 0.50
MIN_PVC_QRS_MS = 90.0
RETRY_SHIFTS_SEC = (-0.05, 0.05, -0.10, 0.10)
TEMPLATE_UPDATE_CORR = 0.90
PREMATURE_FRACTION = 0.88
RUN_CONTINUE_FRACTION = 0.95
TEMPLATE_INIT_BEATS = 20
TEMPLATE_ALPHA = 0.10
TEMPLATE_SWITCH_BEATS = 8
MIN_SHAPE_LEADS = 3
ADC_RAIL_LO = 2.0                # raw samples at the 12-bit ADC rail
ADC_RAIL_HI = 4093.0
CLIP_FRACTION = 0.02
SLOW_RUN_MIN_RR_MS = 600.0       # 100 bpm
STEADY_RR_FRACTION = 0.10
RESTART_GAP_SEC = 3.0
SHAPE_PRE_SEC = 0.10
SHAPE_POST_SEC = 0.16
SHAPE_SHIFT_SEC = 0.02
NORMAL_RR_MIN_FRACTION = 0.88    # RRs used for the running normal RR
NORMAL_RR_MAX_FRACTION = 1.30

NO_QRS_SEGMENTS = ('Ventricular Fibrillation', 'Asystole', 'Artifact', 'Unidentified')

_DEBUG = False
_memory_cache: Dict[str, List[Dict[str, Any]]] = {}
_async_jobs: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def _code_fingerprint() -> str:
    try:
        with open(__file__, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return ''


def _cache_key(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    from .auto_segment_arrthymia_detection import _code_fingerprint as seg_fp
    return f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{_code_fingerprint()}|{seg_fp()}"


def _disk_cache_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), CACHE_FILENAME)


def get_cached_ectopic_beats(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """detect_ectopic_beats() result for the recording at `path`, or None
    when it has not been computed for this file and code. Cheap."""
    key = _cache_key(path)
    if key is None:
        return None
    if key in _memory_cache:
        return copy.deepcopy(_memory_cache[key])
    try:
        with open(_disk_cache_path(path), 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if saved.get('key') != key or not isinstance(saved.get('beats'), list):
        return None
    _memory_cache[key] = saved['beats']
    return copy.deepcopy(saved['beats'])


def detect_ectopic_beats(reader, segments: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Every beat of the recording: {'timestamp' (s), 'label' 'N'|'V'|'Q',
    'rr_ms', 'qrs_ms'}. Cached per file and code. `segments` (the auto
    segment detection) is fetched when not given and the reader has a path."""
    path = getattr(reader, 'path', None)
    key = _cache_key(path)
    if key is not None:
        cached = get_cached_ectopic_beats(path)
        if cached is not None:
            return cached
    if segments is None and path:
        from .auto_segment_arrthymia_detection import detect_arrhythmias
        segments = detect_arrhythmias(reader)
    beats = _detect_uncached(reader, segments)
    if key is not None:
        _memory_cache[key] = copy.deepcopy(beats)
        try:
            tmp = _disk_cache_path(path) + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump({'key': key, 'beats': beats}, f)
            os.replace(tmp, _disk_cache_path(path))
        except (OSError, TypeError, ValueError) as e:
            print(f"[Ectopic beats] Could not save cache: {e}")
    return beats


def detect_ectopic_beats_async(path: str, on_done: Callable[[List[Dict[str, Any]]], None]) -> None:
    """detect_ectopic_beats() for the recording at `path` on a worker thread;
    on_done(beats) runs on the Qt main thread. Call from the main thread."""
    from PyQt5.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

    cached = get_cached_ectopic_beats(path)
    if cached is not None:
        QTimer.singleShot(0, lambda: on_done(cached))
        return
    key = _cache_key(path) or os.path.abspath(path)
    job = _async_jobs.get(key)
    if job is not None:
        job.callbacks.append(on_done)
        return

    class _Job(QObject):
        done = pyqtSignal(object)

        def __init__(self):
            super().__init__()
            self.callbacks = [on_done]
            self.done.connect(self._deliver)

        @pyqtSlot(object)
        def _deliver(self, beats):
            _async_jobs.pop(key, None)
            for cb in self.callbacks:
                try:
                    cb(copy.deepcopy(beats))
                except Exception as e:
                    print(f"[Ectopic beats] callback error: {e}")

    job = _Job()
    _async_jobs[key] = job

    def _run(segments):
        beats = []
        try:
            from .file_format import ECGHFileReader
            reader = ECGHFileReader(path)
            try:
                beats = detect_ectopic_beats(reader, segments)
            finally:
                try:
                    reader.close()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Ectopic beats] detection error: {e}")
        job.done.emit(beats)

    # The auto segments through their own async job, so a detection Full
    # Disclosure already started is joined, not run a second time.
    from .auto_segment_arrthymia_detection import detect_arrhythmias_async

    def _segments_ready(segments):
        threading.Thread(target=_run, args=(segments,), daemon=True, name='EctopicBeats').start()

    detect_arrhythmias_async(path, _segments_ready)


LIVE_S_MATCH_SEC = 0.10


def pattern_beats(beats: Optional[List[Dict[str, Any]]], metrics_list=None) -> List[Dict[str, Any]]:
    """The beats the PVC / PAC pattern labels are built from
    (holter_summary_calc.label_beat_sequences / get_template_beats_for_badges):
    this module's beats with 'fixed_label' (not reclassified there). PACs are
    not classified here yet: until they are, a live 'S' beat (metrics.jsonl
    all_beats, analysis_worker) within LIVE_S_MATCH_SEC of an N beat keeps
    that beat 'S', so the PAC patterns read as before."""
    live_s = []
    for m in metrics_list or []:
        for b in (m.get('all_beats') or []):
            if isinstance(b, dict) and str(b.get('label', '')) == 'S':
                try:
                    live_s.append(float(b.get('timestamp')))
                except (TypeError, ValueError):
                    pass
    live_s = np.array(sorted(live_s))
    out = []
    for b in beats or []:
        label = b.get('label', 'N')
        ts = float(b['timestamp'])
        if label == 'N' and live_s.size:
            k = int(np.searchsorted(live_s, ts))
            if any(0 <= j < live_s.size and abs(live_s[j] - ts) <= LIVE_S_MATCH_SEC for j in (k - 1, k)):
                label = 'S'
        out.append({'timestamp': ts, 'label': label, 'rr_ms': b.get('rr_ms'), 'fixed_label': True})
    return out


# ---------------------------------------------------------------------------
# beats
# ---------------------------------------------------------------------------
def _recording_r_peaks(reader):
    """R peaks (s) of the whole recording: the auto segment detector's pass
    (best lead by amplitude in the first 30 s, Pan-Tompkins per 60 s piece
    with 3 s of context, shape-match beats kept where another lead agrees,
    beats the best lead missed but the other leads all found)."""
    from ecg.pan_tompkins import pan_tompkins
    from .auto_segment_arrthymia_detection import _keep_confirmed_shape_beats, _beats_missed_on_best_lead
    fs = float(reader.fs)
    names = list(reader.lead_names)
    total = float(reader.duration_sec)
    cands = [n for n in ('II', 'V5', 'V6', 'V4', 'V1', 'I') if n in names] or names
    test = reader.read_range(0.0, min(30.0, total))
    best, best_ptp, ranked = 0, 0.0, []
    for c in cands:
        i = names.index(c)
        if test is not None and i < test.shape[0]:
            p = float(np.ptp(test[i]))
            ranked.append((i, p))
            if p > best_ptp:
                best_ptp, best = p, i
    leads = [i for i, _ in sorted(ranked, key=lambda x: x[1], reverse=True)[:3]]
    if best not in leads:
        leads = [best] + leads
    peaks: List[float] = []
    t = 0.0
    while t < total:
        te = min(t + PIECE_SEC, total)
        ps, pe = max(0.0, t - PAD_SEC), min(total, te + PAD_SEC)
        data = reader.read_range(ps, pe)
        if data is not None and best < data.shape[0] and data.shape[1] > 50:
            sig = np.asarray(data[best], dtype=float)
            p_idx, p_plain = pan_tompkins(sig, fs=fs, split_shape_match=True)
            if len(p_idx) != len(p_plain):
                p_idx = _keep_confirmed_shape_beats(p_idx, p_plain, [
                    pan_tompkins(np.asarray(data[li], dtype=float), fs=fs)
                    for li in leads if li != best and li < data.shape[0]], fs)
            for p in p_idx:
                s = ps + float(p) / fs
                if t <= s < te and (not peaks or s - peaks[-1] >= MIN_BEAT_GAP_SEC):
                    peaks.append(s)
            peaks.extend(_beats_missed_on_best_lead(data, best, leads, peaks, ps, t, te, fs, pan_tompkins))
        t += PIECE_SEC
    return sorted(peaks)


def _slope_energy(leads: np.ndarray, fs: float) -> np.ndarray:
    """Sum over leads of |d/dt| (each lead scaled by its own 99th percentile
    slope), smoothed over 40 ms: high on a QRS, low on P / T waves."""
    e = np.zeros(leads.shape[1])
    for x in leads:
        g = np.abs(np.gradient(x))
        e += g / (float(np.percentile(g, 99)) or 1.0)
    w = max(1, int(0.04 * fs))
    return np.convolve(e, np.ones(w) / w, mode='same')


def _search_back(peaks, energy, t0, fs):
    """Beats added where an RR is >= SEARCHBACK_RR_RATIO x the local RR: the
    highest slope-energy point >= MIN_BEAT_GAP_SEC from both neighbours,
    when it reaches SEARCHBACK_MIN_SLOPE of the neighbours' own."""
    added = []
    for k in range(1, len(peaks)):
        a, b = peaks[k - 1], peaks[k]
        lo = max(0, k - LOCAL_RR_BEATS)
        local = np.diff(peaks[lo:k + 1])
        local = local[local > 0]
        if local.size < 2:
            continue
        ref = float(np.median(local))
        if b - a < SEARCHBACK_RR_RATIO * ref:
            continue
        i0 = int((a + MIN_BEAT_GAP_SEC - t0) * fs)
        i1 = int((b - MIN_BEAT_GAP_SEC - t0) * fs)
        ia, ib = int((a - t0) * fs), int((b - t0) * fs)
        if i0 < 0 or i1 >= energy.size or i1 <= i0 or ia < 0 or ib >= energy.size:
            continue
        w = max(1, int(0.05 * fs))
        neighbour = min(float(np.max(energy[max(0, ia - w):ia + w])),
                        float(np.max(energy[max(0, ib - w):ib + w])))
        j = i0 + int(np.argmax(energy[i0:i1]))
        if energy[j] >= SEARCHBACK_MIN_SLOPE * neighbour:
            added.append(t0 + j / fs)
    return added


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def _clipped_rows(clip: Optional[np.ndarray], a: int, b: int) -> np.ndarray:
    """Leads (rows) whose raw samples sit at the ADC rail on more than
    CLIP_FRACTION of [a, b) -- a lead off or saturated there."""
    if clip is None:
        return np.zeros(0, dtype=bool)
    return clip[:, max(0, a):max(0, b)].mean(axis=1) > CLIP_FRACTION


def _shape(leads: np.ndarray, r: int, fs: float, clip: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """The beat's QRS window (-SHAPE_PRE_SEC..+SHAPE_POST_SEC around r) on
    each lead, each lead centred and scaled to unit length (so a small lead
    counts as much as a big one: a PVC often changes only some leads). A
    lead clipped at the ADC rail there is NaN (left out)."""
    a, b = r - int(SHAPE_PRE_SEC * fs), r + int(SHAPE_POST_SEC * fs)
    if a < 0 or b > leads.shape[1]:
        return None
    seg = leads[:, a:b] - leads[:, a:b].mean(axis=1, keepdims=True)
    n = np.linalg.norm(seg, axis=1, keepdims=True)
    n[n == 0] = 1.0
    seg = seg / n
    bad = _clipped_rows(clip, a, b)
    if bad.size and bad.any():
        seg = seg.copy()
        seg[bad] = np.nan
    return seg


def _similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Mean over leads of the per-lead correlation (leads left out as NaN
    ignored; NaN with fewer than MIN_SHAPE_LEADS). Tried the median, so
    noise on 2-3 leads would not make a normal beat "different": 16 -> 9
    false PVCs on a noisy human recording (2026-10-29_16-36-56), but INCART
    PVC sensitivity fell 89.2% -> 82.8% (PVCs that change only some leads),
    so the mean stays."""
    per = np.sum(u * v, axis=1)
    per = per[~np.isnan(per)]
    if per.size < MIN_SHAPE_LEADS:
        return float('nan')
    return float(np.mean(per))


def _nanmean_rows(stack: np.ndarray) -> np.ndarray:
    """Mean over beats per lead and sample; NaN for a lead no beat has
    (without numpy's empty-slice warning)."""
    ok = ~np.isnan(stack)
    n = ok.sum(axis=0)
    total = np.where(ok, stack, 0.0).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def _normalised(t: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(t, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return t / n


def _blend(template: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    """Template moved toward v by alpha on the leads v has (NaN rows kept)."""
    out = np.where(np.isnan(v), template, (1.0 - alpha) * template + alpha * np.nan_to_num(v))
    return _normalised(out)


def _corr(shape_leads: np.ndarray, r: int, template: np.ndarray, fs: float,
          clip: Optional[np.ndarray] = None) -> float:
    """Best similarity of the beat at r with the template over
    +-SHAPE_SHIFT_SEC; NaN when too few usable leads."""
    best = float('nan')
    step = max(1, int(0.004 * fs))
    s = int(SHAPE_SHIFT_SEC * fs)
    for d in range(-s, s + 1, step):
        v = _shape(shape_leads, r + d, fs, clip)
        if v is not None and v.shape == template.shape:
            c = _similarity(v, template)
            if not np.isnan(c) and (np.isnan(best) or c > best):
                best = c
    return best


def _no_qrs_spans(segments):
    return sorted((float(s['start_sec']), float(s['end_sec'])) for s in (segments or [])
                  if str(s.get('label', '')) in NO_QRS_SEGMENTS
                  and s.get('start_sec') is not None and s.get('end_sec') is not None)


def _detect_uncached(reader, segments) -> List[Dict[str, Any]]:
    from scipy.signal import filtfilt
    from ecg.enhanced_filter import make_notch
    from .analysis_pipeline import SignalPreprocessor, HolterConfig
    from .pqrst_measure import QRS_LEADS, _global_qrs, _quiet_qrs_leads

    fs = float(reader.fs)
    names = list(reader.lead_names)
    total = float(reader.duration_sec)
    qrs_idx = [names.index(n) for n in QRS_LEADS if n in names]
    if len(qrs_idx) < 3:
        return []
    peaks = _recording_r_peaks(reader)
    preproc = SignalPreprocessor(fs=fs, config=HolterConfig.from_clinical_config())
    notch_60 = make_notch(60.0, fs)

    # Per piece: filtered leads, search-back beats, QRS width and shape input.
    feats = []           # (t, qrs_ms or None, piece index, sample in piece)
    pieces = []
    t = 0.0
    all_peaks = list(peaks)
    while t < total:
        te = min(t + PIECE_SEC, total)
        ps, pe = max(0.0, t - PAD_SEC), min(total, te + PAD_SEC)
        data = reader.read_range(ps, pe)
        if data is None or data.shape[1] < int(fs):
            t += PIECE_SEC
            continue
        leads = np.vstack([filtfilt(notch_60.b, notch_60.a, np.asarray(preproc.process(np.asarray(x, dtype=float)), dtype=float))
                           for x in np.asarray(data, dtype=float)])
        q = leads[qrs_idx]
        raw = np.asarray(data, dtype=float)[qrs_idx]
        clip = (raw <= ADC_RAIL_LO) | (raw >= ADC_RAIL_HI)
        local = [p for p in all_peaks if ps <= p < pe]
        extra = [x for x in _search_back(local, _slope_energy(q, fs), ps, fs) if t <= x < te]
        all_peaks.extend(extra)
        pieces.append((ps, q, clip))
        piece_peaks = sorted(p for p in all_peaks if t <= p < te)
        for p in piece_peaks:
            feats.append([p, None, len(pieces) - 1, int(round((p - ps) * fs))])
        t += PIECE_SEC
    feats.sort(key=lambda f: f[0])

    # QRS width (needs the previous RR for the quiet-lead choice)
    for k, f in enumerate(feats):
        ps, q, clip = pieces[f[2]]
        prev_rr = (f[0] - feats[k - 1][0]) * 1000.0 if k > 0 else None
        bad = _clipped_rows(clip, f[3] - int(0.30 * fs), f[3] + int(0.40 * fs))
        usable = [i for i in range(q.shape[0]) if not (bad.size and bad[i])]
        if len(usable) < MIN_SHAPE_LEADS:
            continue
        quiet = _quiet_qrs_leads(q, usable, f[3], prev_rr, fs)
        bounds = _global_qrs(q, quiet, f[3], fs)
        # Not bounded at the detected R: retried 50 / 100 ms either side (a
        # PVC's R-peak time is often off its QRS centre; on INCART I31 / I18
        # 77 of 97 unbounded beats were bounded this way).
        for d in RETRY_SHIFTS_SEC:
            if bounds is not None:
                break
            bounds = _global_qrs(q, quiet, f[3] + int(d * fs), fs)
        if bounds is not None:
            f[1] = (bounds[1] - bounds[0]) * 1000.0 / fs
            f[3] = (bounds[0] + bounds[1]) // 2      # shapes centred on the QRS

    no_qrs = _no_qrs_spans(segments)

    def blocked(ts):
        return any(s <= ts < e for s, e in no_qrs)

    def start(k0):
        """Template and normal RR learnt from the beats from feats[k0] on,
        up to the next restart point: the medoid of the first
        TEMPLATE_INIT_BEATS (the narrow ones when >= 30% are, so a bigeminy
        start does not take the PVC for the template), and the RRs between
        consecutive ones with its shape (so a PVC early on is already
        premature; seen: the first PVC of a trigeminy at beat 3 read N)."""
        group = []
        for k in range(k0, len(feats)):
            if k > k0 and restart_at[k]:
                break
            if feats[k][1] is not None and not blocked(feats[k][0]) and not in_vt(feats[k][0]):
                group.append(feats[k])
            if len(group) >= TEMPLATE_INIT_BEATS:
                break
        narrow = [f for f in group if f[1] < WIDE_QRS_MS]
        pool = narrow if len(narrow) >= 0.3 * max(1, len(group)) else group
        shapes = [v for v in (_shape(pieces[f[2]][1], f[3], fs, pieces[f[2]][2]) for f in pool) if v is not None]
        if not shapes:
            return template, ref_rr      # nothing to learn from: keep the old
        m = np.nan_to_num(np.array(shapes))
        tpl = m[int(np.argmax(np.einsum('ilw,jlw->ij', m, m).sum(axis=1)))].copy()
        like = [f for f in group if _corr(pieces[f[2]][1], f[3], tpl, fs, pieces[f[2]][2]) >= SHAPE_SAME_CORR]
        like_ids = {id(f) for f in like}
        rrs = [(b_[0] - a_[0]) * 1000.0 for a_, b_ in zip(group, group[1:])
               if id(a_) in like_ids and id(b_) in like_ids]
        return tpl, rrs[:LOCAL_RR_BEATS]

    # Restart points: the first beat, a beat after a gap > RESTART_GAP_SEC,
    # and the first beat after a no-QRS segment beat. Seen: sinus
    # tachycardia at 160 bpm right after 24 s of VFib / Unidentified
    # (2026-11-04_11-08-46) was compared with the template and normal RR
    # (750 ms) from before them -- 132 narrow sinus beats read V.
    restart_at = [False] * len(feats)
    for k, f in enumerate(feats):
        if k == 0 or f[0] - feats[k - 1][0] > RESTART_GAP_SEC or blocked(feats[k - 1][0]):
            restart_at[k] = True

    # Beats of an auto Ventricular Tachycardia segment never start a
    # template: restarting right after VFib into VT (2026-11-04_11-08-46)
    # learnt the VT beats as the dominant beat and the run read N.
    vt_spans = [(float(x['start_sec']), float(x['end_sec'])) for x in (segments or [])
                if str(x.get('label', '')) == 'Ventricular Tachycardia']

    def in_vt(ts):
        return any(s_ <= ts < e_ for s_, e_ in vt_spans)

    template, ref_rr = None, []

    out = []
    pending: List[np.ndarray] = []     # new-shape beats at the normal rate, in a row
    run: List[tuple] = []               # the current V run: (beat, rr, shape)
    prev_label = None
    prev_t = None
    for k, f in enumerate(feats):
        ts, qrs = f[0], f[1]
        if restart_at[k] and not blocked(ts):
            template, ref_rr = start(k)
            pending, run, prev_label = [], [], None
        after_ectopic = prev_label == 'V'
        rr = (ts - prev_t) * 1000.0 if prev_t is not None else None
        prev_t = ts
        beat = {'timestamp': round(ts, 3), 'label': 'N',
                'rr_ms': None if rr is None else round(rr, 1),
                'qrs_ms': None if qrs is None else round(qrs, 1)}
        out.append(beat)
        if blocked(ts) or template is None:
            beat['label'] = 'Q'
            prev_label = 'Q'
            pending = []
            run = []
            continue
        shape_leads, clip = pieces[f[2]][1], pieces[f[2]][2]
        corr = _corr(shape_leads, f[3], template, fs, clip)
        if np.isnan(corr):          # too few usable leads to judge the shape
            beat['label'] = 'Q'
            prev_label = 'Q'
            pending = []
            run = []
            continue
        normal_rr = float(np.median(ref_rr)) if len(ref_rr) >= 2 else None
        premature = rr is not None and normal_rr is not None and rr <= PREMATURE_FRACTION * normal_rr
        different = corr < SHAPE_SAME_CORR
        # a beat whose QRS cannot be bounded is no PVC evidence: on INCART
        # most such "beats" were noise / T waves Pan-Tompkins took (146 of
        # 321 false V had no annotated beat at all)
        very_different = corr < VERY_DIFFERENT_CORR and qrs is not None and qrs >= MIN_PVC_QRS_MS
        wide = qrs is not None and qrs >= WIDE_QRS_MS
        run_on = (prev_label == 'V' and rr is not None and normal_rr is not None
                  and rr <= RUN_CONTINUE_FRACTION * normal_rr)
        if _DEBUG:
            beat['_dbg'] = (round(corr, 2), None if normal_rr is None else round(normal_rr), bool(premature), bool(run_on))
        # V: premature (or continuing a ventricular run) with a shape unlike
        # the dominant beat's and a wide QRS -- or a shape so unlike it that
        # a QRS measured under WIDE_QRS_MS (>= MIN_PVC_QRS_MS) does not
        # matter (seen: INCART PVCs measured 108-118 ms, similarity 0.0-0.4).
        if (premature or run_on) and (very_different or (different and wide)):
            beat['label'] = 'V'
            prev_label = 'V'
            pending = []
            if not run_on:
                run = []
            run.append((beat, rr, _shape(shape_leads, f[3], fs, clip)))
            # A "run" slower than 100 bpm with a steady RR is a new rhythm,
            # not PVCs (a PVC run / VT is >= 100 bpm): the beats go back to
            # N and the template and normal RR restart from them. Seen: a
            # 30 bpm complete heart block followed by 80 bpm conducted RBBB
            # (2026-10-20_14-55-50) -- the first RBBB beat was premature
            # against 2000 ms and every later one "continued the run":
            # 1527 of 1560 beats V.
            if len(run) >= TEMPLATE_SWITCH_BEATS:
                # the run's last TEMPLATE_SWITCH_BEATS RRs (its first beat's
                # coupling, or an early jitter, must not keep it a "run"
                # forever: RBBB at 80 bpm stayed V for 16 beats on one 658 ms RR)
                rrs = np.array([x[1] for x in run[1:][-TEMPLATE_SWITCH_BEATS:] if x[1] is not None])
                if (rrs.size and float(np.min(rrs)) >= SLOW_RUN_MIN_RR_MS
                        and float(np.max(np.abs(rrs - np.median(rrs)))) <= STEADY_RR_FRACTION * float(np.median(rrs))):
                    for b_, _, _ in run:
                        b_['label'] = 'N'
                    shapes_ = [x[2] for x in run if x[2] is not None]
                    if shapes_:
                        template = _blend(template, _nanmean_rows(np.array(shapes_)), 1.0)
                    ref_rr = [float(x) for x in rrs][-LOCAL_RR_BEATS:]
                    prev_label = 'N'
                    run = []
            continue
        run = []
        if qrs is None:
            beat['label'] = 'Q'
            prev_label = 'Q'
            pending = []
            continue
        prev_label = 'N'
        # normal beat: running RR and template
        if rr is not None and (normal_rr is None or NORMAL_RR_MIN_FRACTION * normal_rr <= rr <= NORMAL_RR_MAX_FRACTION * normal_rr):
            ref_rr.append(rr)
            ref_rr = ref_rr[-LOCAL_RR_BEATS:]
        elif rr is not None and not different and not after_ectopic:
            # dominant shape, faster or slower: the rate changed (e.g. sinus
            # tachycardia ending). Seen: 107 -> 80 bpm and the normal RR
            # stayed 562 ms for minutes when only faster beats followed.
            # Not the pause after a PVC.
            ref_rr.append(rr)
            ref_rr = ref_rr[-LOCAL_RR_BEATS:]
        v = _shape(shape_leads, f[3], fs, clip)
        if v is None:
            continue
        if corr >= TEMPLATE_UPDATE_CORR:
            # only clearly same-shape beats move the template, so missed
            # PVCs do not drag it toward their shape
            template = _blend(template, v, TEMPLATE_ALPHA)
            pending = []
        elif different and not premature:
            if pending and not (_similarity(pending[-1], v) >= SHAPE_SAME_CORR):
                pending = []
            pending.append(v)
            if len(pending) >= TEMPLATE_SWITCH_BEATS:
                template = _blend(template, _nanmean_rows(np.array(pending)), 1.0)
                pending = []
        elif not different:
            pending = []

    return out
