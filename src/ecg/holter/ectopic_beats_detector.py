"""Beat-by-beat ectopic labels (PVC = 'V', PAC = 'S') for the whole recording,
from the recorded signal.  Separate from the live per-chunk labels (analysis_worker.
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
    A late PVC (not premature) is V when it is wide, its shape is very
    different (< VERY_DIFFERENT_CORR), no sinus P wave comes before it (P
    stretch similarity < LATE_PVC_P_MAX), the beats on both sides are N
    with the dominant shape (a lone beat: a bundle branch block or paced run
    starts with several such beats) and a pause follows it (the next RR >=
    LATE_PVC_PAUSE_FRACTION x the normal RR: a beat neither early nor
    followed by a pause did not break the rhythm) -- and it must repeat: at least
    LATE_MIN_PARTNERS other V beats of the recording match its shape (>=
    LATE_PARTNER_CORR); a late PVC with fewer is dropped.
    Not V: a wide beat at the normal rate with the sinus P (bundle branch
    block), a wide beat with the dominant beat's shape, and any beat inside
    an auto Atrial Flutter or Ventricular Tachycardia segment (NO_PVC_SEGMENTS).
  - A PVC repeats: a V whose shape matches no other V of the recording
    (beyond its own neighbours) >= PARTNER_CORR goes back to N -- noise
    that makes a normal beat "different" does not repeat, a ventricular
    focus does. The only V of a recording stays V when its shape is very
    different, its QRS wide, it came early against the beat before it too
    and no sinus P wave comes before it (_lone_v_is_ectopic). Only V beats
    with >= MIN_SHAPE_LEADS usable leads in common count as compared; a V
    nothing can be compared with (leads clipped at the ADC rail) gets the
    same only-V test.
  - Template: started from the first beats (the narrow ones when there are
    enough, so a bigeminy start does not take the PVC for the template),
    moved only by N beats with similarity >= TEMPLATE_UPDATE_CORR, and
    replaced when TEMPLATE_SWITCH_BEATS beats in a row at the normal rate
    share a new shape (a lasting change of conduction), or when
    TEMPLATE_SWITCH_BEATS V beats in a row (Q beats between them do not
    break the row) come at a steady RR under 100 bpm (a new rhythm, not
    PVCs). The PVC test judges the shape centred on the QRS and on the R
    peak, and takes the better match (f waves can move the QRS edges). The normal RR
    follows dominant-shape beats both faster and slower, except the pause
    after a PVC. Template and normal RR are learnt again after a gap >
    RESTART_GAP_SEC or a no-QRS segment (below), never from the beats of an
    auto Ventricular Tachycardia segment (the old ones are kept when
    nothing else is there).
  - Beats inside an auto segment where no QRS can be trusted (Ventricular
    Fibrillation, Asystole, Artifact, Unidentified -- not a mains 50 / 60 Hz
    Artifact, see _blocks_qrs), with fewer than
    MIN_SHAPE_LEADS usable leads, or (when not V) with no QRS bounds get
    'Q' (not classified), and so does a spike: narrower than MIN_QRS_MS,
    unlike the dominant beat, splitting one normal cycle in two (an
    artifact, not a beat). Everything else not V is 'N'.
  - A narrow R peak < POST_V_MIN_RR_MS after a V is that PVC's T wave: it
    is not returned, and the next beat's RR is measured from the V (when
    the repeat test turns that V back to N, it is returned as 'Q').
  - PACs ('S'): a second pass over the N beats, _classify_pacs (early,
    abruptly, with the dominant QRS shape and a P stretch unlike the
    dominant beats'; not in AF / Flutter / AV block segments).
  - Paced beats ('P', with 'pace' 'V' / 'A' / 'AV'): _label_paced, before
    the PAC pass -- a beat made by a pacemaker pulse, found by its pacing
    spike (see 'paced beats' below). A spike the beat finder took for a
    beat (an atrial spike, one with no capture) is taken out of the beats.

  Scored against cardiologist annotations (first 5 min of each record):
  INCART 12-lead and MIT-BIH (2 leads; the other 10 are mixes of them, so
  the shape test is weaker) -- see the PVC Se / +P in the change notes.
  Known limits: a late PVC with a P wave in front of it (or one the P
  stretch cannot be judged for) is not found; a PVC from a focus that
  fires only once in a recording that has other PVCs is dropped by the
  repeat test. Motion noise on a human recording (2026-10-29_16-36-56,
  31 min) gave 10 false PVCs before the repeat test, 0 after (the live
  labels gave 136).
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
LEAD_DEAD_FRACTION = 0.10      # a lead whose 60 s peak-to-peak falls under this x its first 30 s is off there
SEARCHBACK_RR_RATIO = 1.6        # an RR this many times the local RR is searched
SEARCHBACK_MIN_SLOPE = 0.35      # a QRS there: combined slope >= this x neighbours'
LOCAL_RR_BEATS = 8

WIDE_QRS_MS = 110.0
SHAPE_SAME_CORR = 0.80
VERY_DIFFERENT_CORR = 0.50
MIN_PVC_QRS_MS = 90.0
MIN_QRS_MS = 60.0                # narrower than any QRS
SPLIT_RR_FRACTION = 0.10         # this beat's RR + the next one's within this of the normal RR: it split one cycle
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
POST_V_MIN_RR_MS = 300.0   # a narrow "beat" closer than this after a V is its T wave, not a beat
PARTNER_CORR = 0.70
UNIQUE_PVC_QRS_MS = 120.0        # a V with no partner kept on its own evidence: QRS this wide (LITFL "broad QRS >= 120 ms") ...
FULL_PAUSE_FRACTION = 0.10       # ... and RR + next RR within this of twice the normal RR
PARTNER_MAX_COMPARE = 300
PAC_PREMATURE_FRACTION = 0.85
PAC_ABRUPT_FRACTION = 0.85
PAC_RETURN_FRACTION = 1.10
PAC_RUN_RR_FRACTION = 0.15
P_PRE_SEC = 0.26
P_DIFFERENT_MAX = 0.50
P_READABLE_MIN = 0.60      # neighbours' median P-stretch similarity: the P wave is readable there
P_CONTRAST_MIN = 0.30      # a PAC's P stretch is this much less like the dominant beats' than its neighbours' are
P_LOCAL_BEATS = 6          # N beats on each side for the two tests above
LATE_PVC_P_MAX = 0.50
LATE_PVC_PAUSE_FRACTION = 1.05   # a late PVC is followed by a pause: the next beat's RR >= this x the normal RR
LATE_PARTNER_CORR = 0.90   # a late PVC must match at least LATE_MIN_PARTNERS other V beats this well
LATE_MIN_PARTNERS = 5
P_POST_SEC = 0.06
# Where no PAC is looked for: the RR is irregular by itself (atrial rhythm /
# conduction), or the segment is a rhythm of its own -- a rate segment is a
# rate step, not a run of PACs (the first beat of a step is "early" against
# the old normal RR and its P stretch differs from the old rate's, so the
# whole new rhythm read as one S run: 14-24 S per Sinus Tachycardia / SVT
# segment of the simulator recordings), and first-degree AV block has a PR so
# long the P stretch is not where the P test looks. A ventricular tachycardia
# is no run of PACs either (2026-10-20_12-26-09: 2 S on its beats).
NO_PAC_SEGMENTS = ('Atrial Fibrillation', 'Atrial Flutter', 'Second-degree AV Block (Mobitz I)',
                   'Second-degree AV Block (Mobitz II)', 'Second-degree AV Block (2:1)', 'Third-degree AV Block',
                   'First-degree AV Block', 'Sinus Tachycardia', 'Sinus Bradycardia', 'Tachycardia', 'Bradycardia',
                   'Supraventricular Tachycardia (SVT)', 'Ventricular Tachycardia')
PAC_SEGMENT_MARGIN_SEC = 1.0   # a rate step's first beat comes just before its segment
SHAPE_PRE_SEC = 0.10
SHAPE_POST_SEC = 0.16
SHAPE_SHIFT_SEC = 0.02
NORMAL_RR_MIN_FRACTION = 0.88    # RRs used for the running normal RR
NORMAL_RR_MAX_FRACTION = 1.30

NO_QRS_SEGMENTS = ('Ventricular Fibrillation', 'Asystole', 'Artifact', 'Unidentified')
# Where no PVC is looked for. Flutter is atrial-rate bound and regular, so a
# beat is never "premature" in it: after a rate step (100 -> 150 bpm,
# 2026-11-16_12-09-16 at 12:11:41) every beat is premature against the running
# normal RR, the shape of a conducted flutter beat matches the dominant beat
# at 0.78-0.85 (limit 0.80: the F waves fall in its window) and its QRS reads
# 112-118 ms even when normal, so premature + a shape just under the limit +
# "wide" read V: 10 V in a 150 bpm flutter, 8 of them on the noise after an
# Artifact, 2 on clean beats (12:12:05).
# Atrial Fibrillation is deliberately NOT here: cardiologists annotated 495
# (MIT-BIH) and 226 (INCART) PVCs inside the auto AF / Flutter segments of the
# first 5 min of those records and this module found 380 and 196 of them (false
# V 32 and 1) -- a PVC does occur in AF, and the V beats of the simulator's AF
# recordings look like them (shape similarity median 0.63 vs 0.56, QRS 166 vs
# 168 ms, RR / normal RR 0.66 vs 0.63), so no limit here removes the one and
# not the other. Flutter in those records is 15 s and holds no V.
# Ventricular Tachycardia is a rhythm of its own, not a run of PVCs: its beats
# are premature against the running normal RR and unlike the dominant shape,
# so every one read V and the run read "PVC Run" (2026-11-16_15-35-15,
# 160 bpm wide-complex VT, 22 V at 15:36:58 - 15:37:06). Beats inside an auto
# Ventricular Tachycardia segment are not looked at for PVCs.
NO_PVC_SEGMENTS = ('Atrial Flutter', 'Ventricular Tachycardia')
MAINS_ARTIFACT_KINDS = ('50 Hz', '60 Hz')   # 'artifact_kind' of the segment detector's mains Artifact

_DEBUG = False
_memory_cache: Dict[str, Dict[str, Any]] = {}     # key -> {'beats': [...], 'spans': signature}
_async_jobs: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# cache
#
# The result is the detector's code (this file, pqrst_measure,
# analysis_pipeline, the filters, ... -- every ecg file it imports) applied to
# the recording AND to its auto segments (the no-QRS / VT / no-PAC spans
# below). The segment detection's own cache key already pins all of that: the
# file (path, size, mtime), a hash of the code of both detectors and every
# file they import (auto_segment_arrthymia_detection._code_fingerprint), and
# the patient's age (it moves the SVT bound, so the segments), plus the clinical
# configuration (signal-quality limit, QRS validator, mains notch: read by the
# QRS measurement here; it is YAML, not code). So the key here is that key,
# not a second list of files that could drift. A saved entry also
# carries a signature of the spans it was made with, so segments passed in by
# a caller that are not the recording's own detection do not read a result made
# with other ones.
# ---------------------------------------------------------------------------
def _code_fingerprint() -> str:
    """Hash of the code the result depends on (in a frozen build the hash of
    the sources it was built from, shipped with it; '' when neither can be
    read: nothing then says a saved result is current, so the disk cache is
    not used -- the memory cache still is)."""
    from .auto_segment_arrthymia_detection import _code_fingerprint as seg_fp
    return seg_fp()


def _cache_key(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    from .auto_segment_arrthymia_detection import _cache_key as seg_key
    k = seg_key(path)
    if k is None:
        return None
    from .cache_fingerprint import clinical_config_fingerprint
    return f"ectopic|{k}|cfg={clinical_config_fingerprint()}"


def _blocks_qrs(seg) -> bool:
    """True when auto segment `seg` is one whose beats get 'Q' (no QRS
    trusted): a NO_QRS_SEGMENTS label -- except a mains (50 / 60 Hz)
    Artifact. That one only marks interference at one frequency; this
    module's filters (the preprocessing's 50 Hz notch + a 60 Hz notch) take
    it out and the QRS under it reads as usual. Blocking it hid every PVC /
    PAC under the interference: a recording with 50 / 60 Hz switched on
    throughout had a PVC at 7.68 s (QRS 268 ms) inside Artifact 0.0-11.6 s
    (60 Hz) and the
    label read Normal Sinus Rhythm (2026-11-12_12-30-47).
    Its beats are still not learnt from while a template exists (see
    start() in _detect_uncached)."""
    label = str(seg.get('label', ''))
    if label not in NO_QRS_SEGMENTS:
        return False
    return not (label == 'Artifact' and seg.get('artifact_kind') in MAINS_ARTIFACT_KINDS)


def _spans_signature(segments) -> str:
    """Hash of the auto segments this module reads (label, span and, for an
    Artifact, its kind: a mains one does not block beats, it only keeps them
    out of the template)."""
    used = set(NO_QRS_SEGMENTS) | set(NO_PAC_SEGMENTS) | set(NO_PVC_SEGMENTS) | {'Ventricular Tachycardia'}
    spans = sorted((str(x.get('label', '')), round(float(x['start_sec']), 3), round(float(x['end_sec']), 3),
                    str(x.get('artifact_kind') or ''))
                   for x in (segments or [])
                   if str(x.get('label', '')) in used
                   and x.get('start_sec') is not None and x.get('end_sec') is not None)
    return hashlib.sha1(json.dumps(spans).encode('utf-8')).hexdigest()


def _disk_cache_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(path)), CACHE_FILENAME)


def _load_entry(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """The saved {'beats', 'spans'} for this recording and code, or None."""
    key = _cache_key(path)
    if key is None:
        return None
    if key in _memory_cache:
        return _memory_cache[key]
    if not _code_fingerprint():
        return None
    try:
        with open(_disk_cache_path(path), 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if (not isinstance(saved, dict) or saved.get('key') != key
            or not isinstance(saved.get('beats'), list) or not isinstance(saved.get('spans'), str)):
        return None
    _memory_cache[key] = {'beats': saved['beats'], 'spans': saved['spans']}
    return _memory_cache[key]


def _entry_for_own_segments(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """_load_entry, but only when the spans it was made with are those of the
    recording's own segment detection (when that is at hand: it is cached
    whenever the detection has run). A result made with other segments -- a
    caller that passed its own -- is not handed to one that did not."""
    entry = _load_entry(path)
    if entry is None:
        return None
    from .auto_segment_arrthymia_detection import get_cached_detection
    own = get_cached_detection(path)
    if own is not None and entry['spans'] != _spans_signature(own):
        return None
    return entry


def _save_entry(path: str, beats: List[Dict[str, Any]], spans: str) -> None:
    key = _cache_key(path)
    if key is None:
        return
    _memory_cache[key] = {'beats': copy.deepcopy(beats), 'spans': spans}
    if not _code_fingerprint():
        return
    # a temp name of its own: the report and the UI can save at the same time
    tmp = f"{_disk_cache_path(path)}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({'key': key, 'spans': spans, 'beats': beats}, f)
        os.replace(tmp, _disk_cache_path(path))
    except (OSError, TypeError, ValueError) as e:
        print(f"[Ectopic beats] Could not save cache: {e}")
        try:
            os.remove(tmp)
        except OSError:
            pass


def get_cached_ectopic_beats(path: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """detect_ectopic_beats() result for the recording at `path`, or None
    when it has not been computed for this file, its segments and the code.
    Cheap."""
    entry = _entry_for_own_segments(path)
    return None if entry is None else copy.deepcopy(entry['beats'])


def detect_ectopic_beats(reader, segments: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Every beat of the recording: {'timestamp' (s), 'label' 'N'|'V'|'S'|'Q'|'P',
    'rr_ms', 'qrs_ms'}, a paced beat ('P') also 'pace' ('V' ventricular, 'A'
    atrial, 'AV' both). Cached per file, segments and code. `segments` (the
    auto segment detection) is fetched when not given and the reader has a
    path; when given they must be that detection's (a saved result made with
    other spans is not reused)."""
    path = getattr(reader, 'path', None)
    entry = _entry_for_own_segments(path) if segments is None else _load_entry(path)
    if entry is not None and (segments is None or entry['spans'] == _spans_signature(segments)):
        return copy.deepcopy(entry['beats'])
    if segments is None and path:
        from .auto_segment_arrthymia_detection import detect_arrhythmias
        segments = detect_arrhythmias(reader)
    beats = _detect_uncached(reader, segments)
    if path:
        _save_entry(path, beats, _spans_signature(segments))
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


SEGMENT_PVC_MAX_RUN = 6     # a longer V run is VT / VFib: left to the segment rules for those


def pvc_times_for_segments(reader) -> np.ndarray:
    """Sorted times (s) of the beats the auto segment detector's sinus-rhythm
    beat rules (Mobitz I, Sinus Arrhythmia, plain Tachycardia) must not read
    as conducted sinus beats: this module's V beats in runs of at most
    SEGMENT_PVC_MAX_RUN. A longer run is VT / VFib (the segment detector's own
    rules judge those) and is not returned: taking it out of the rate rules
    would hide a real VT.

    Computed without the auto segments (they are made from this): no
    segment blocks a beat here, so a stretch the segment detector will call
    VFib / Artifact may hold V beats as well -- in runs longer than
    SEGMENT_PVC_MAX_RUN, or where the rules above do not look. Not cached:
    the segment detection's own cache holds the result that used it."""
    beats = _detect_uncached(reader, [])
    out: List[float] = []
    i = 0
    while i < len(beats):
        if beats[i]['label'] == 'V':
            j = i
            while j + 1 < len(beats) and beats[j + 1]['label'] == 'V':
                j += 1
            if j - i + 1 <= SEGMENT_PVC_MAX_RUN:
                out.extend(float(beats[k]['timestamp']) for k in range(i, j + 1))
            i = j + 1
        else:
            i += 1
    return np.array(sorted(out))


def pattern_beats(beats: Optional[List[Dict[str, Any]]], metrics_list=None) -> List[Dict[str, Any]]:
    """The beats the PVC / PAC pattern labels are built from
    (holter_summary_calc.label_beat_sequences / get_template_beats_for_badges):
    this module's beats -- N / V (PVC) / S (PAC, _classify_pacs) / Q -- with
    'fixed_label' (not reclassified there). `metrics_list` is no longer
    read: until PACs were classified here, the live 'S' beats
    (metrics.jsonl all_beats, analysis_worker._label_beat: narrow and RR <
    90% of the chunk median) were laid over the N beats -- 124 S on a
    healthy human recording (2026-10-29_16-36-56), 13 now."""
    return [{'timestamp': float(b['timestamp']), 'label': b.get('label', 'N'),
             'rr_ms': b.get('rr_ms'), 'fixed_label': True}
            for b in beats or []]


# ---------------------------------------------------------------------------
# beats
# ---------------------------------------------------------------------------
def _recording_r_peaks(reader):
    """R peaks (s) of the whole recording: the auto segment detector's pass
    (best lead by amplitude in the first 30 s, Pan-Tompkins per 60 s piece
    with 3 s of context, shape-match beats kept where another lead agrees,
    beats the best lead missed but the other leads all found).
    In a piece where that lead is off (peak-to-peak under
    LEAD_DEAD_FRACTION x its first 30 s) the largest lead still alive there
    is read instead. Seen: V4 chosen, then flat from 6 min to the end of an
    87 min recording with lead II alive -- 361 beats found, now 5155
    (2026-10-30_12-01-08); INCART I58 V4 at 9 counts for 3 min, I02 V6 at
    144 (was 4091): 3 false PVCs fewer. Every other record unchanged."""
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
            b_, leads_ = best, leads
            if float(np.ptp(data[best])) < LEAD_DEAD_FRACTION * best_ptp:
                # the chosen lead is off in this piece: the largest one alive
                alive = sorted(((float(np.ptp(data[i])), i) for i, _ in ranked if i < data.shape[0]), reverse=True)
                alive = [i for p, i in alive if p >= LEAD_DEAD_FRACTION * best_ptp]
                if alive:
                    b_, leads_ = alive[0], alive[:3]
            sig = np.asarray(data[b_], dtype=float)
            p_idx, p_plain = pan_tompkins(sig, fs=fs, split_shape_match=True)
            if len(p_idx) != len(p_plain):
                p_idx = _keep_confirmed_shape_beats(p_idx, p_plain, [
                    pan_tompkins(np.asarray(data[li], dtype=float), fs=fs)
                    for li in leads_ if li != b_ and li < data.shape[0]], fs)
            for p in p_idx:
                s = ps + float(p) / fs
                if t <= s < te and (not peaks or s - peaks[-1] >= MIN_BEAT_GAP_SEC):
                    peaks.append(s)
            peaks.extend(_beats_missed_on_best_lead(data, b_, leads_, peaks, ps, t, te, fs, pan_tompkins))
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


def _p_shape(leads: np.ndarray, r: int, fs: float, clip: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """The stretch before the QRS where the P wave sits (P_PRE_SEC to
    P_POST_SEC before the QRS centre r), per lead centred and scaled to
    unit length; clipped leads NaN."""
    a, b = r - int(P_PRE_SEC * fs), r - int(P_POST_SEC * fs)
    if a < 0 or b > leads.shape[1] or b <= a:
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


def _lone_v_is_ectopic(out, pcorrs, i) -> bool:
    """The only V of a recording (no partner for the repeat test) needs more
    than a very different, wide shape -- noise on a normal beat gives that
    too (shape similarity -0.01, QRS 188 ms read on an 82 ms beat,
    sushil_20_mins 2026-10-30_14-30-07 at 431.5 s). It stays V only when
      - it came early against the beat before it as well, RR <=
        PREMATURE_FRACTION x that beat's RR (not needed after a V: the
        second beat of a couplet). Early only against the running normal
        RR is a rate that drifted: that beat was "premature" (738 ms
        against a normal RR of 863 in Sinus Arrhythmia) at the RR of the
        beat before it (736 ms). The same test drops a normal beat right
        after a VT (RR 1.71 / 1.86 x the one before) and one on a 50/60 Hz
        switch (1.00);
      - no sinus P wave comes before it: P stretch similarity <
        LATE_PVC_P_MAX (not judged when it cannot be measured). Seen: the
        first conducted beat of a new rhythm, P 0.98.
    INCART / MIT-BIH lone PVCs measured RR 0.51-0.65 x the beat before and
    P similarity <= 0.39: all 5 kept."""
    if i <= 0:
        return True
    prev = out[i - 1]
    if prev.get('label') != 'V':
        rr, prev_rr = out[i].get('rr_ms'), prev.get('rr_ms')
        if rr is not None and prev_rr and rr > PREMATURE_FRACTION * prev_rr:
            return False
    pc = pcorrs[i] if i < len(pcorrs) else None
    if pc is not None and pc >= LATE_PVC_P_MAX:
        return False
    return True


def _full_pause(out, i) -> bool:
    """Beat i is followed by a full compensatory pause (LITFL: a PVC is
    "usually followed by a full compensatory pause"): its RR + the next
    beat's within FULL_PAUSE_FRACTION of twice the RR of the N beats around
    it (N after N, up to 8 each side, at least 3)."""
    if i + 1 >= len(out):
        return False
    rr, nxt = out[i].get('rr_ms'), out[i + 1].get('rr_ms')
    if rr is None or nxt is None:
        return False
    nb = [out[j]['rr_ms'] for j in range(max(1, i - 8), min(len(out), i + 9))
          if j not in (i, i + 1) and out[j].get('label') == 'N' and out[j].get('rr_ms')
          and out[j - 1].get('label') == 'N']
    if len(nb) < 3:
        return False
    n = float(np.median(nb))
    return abs(rr + nxt - 2.0 * n) <= FULL_PAUSE_FRACTION * 2.0 * n


def _no_qrs_spans(segments):
    return sorted((float(s['start_sec']), float(s['end_sec'])) for s in (segments or [])
                  if _blocks_qrs(s)
                  and s.get('start_sec') is not None and s.get('end_sec') is not None)


def _classify_pacs(out: List[Dict[str, Any]], corrs: List[Optional[float]], segments,
                   pcorrs: Optional[List[Optional[float]]] = None) -> None:
    """PACs ('S'), beat by beat, on the beats the PVC pass left N (in place).

    A PAC is a supraventricular beat that comes early: it is conducted the
    normal way, so its QRS looks like the dominant beat's -- the opposite of
    a PVC. A beat N becomes S when
      - its shape matches the dominant beat (similarity >= SHAPE_SAME_CORR);
        no QRS width limit (a patient with a wide QRS has wide PACs);
      - it is premature: RR <= PAC_PREMATURE_FRACTION x the running RR of
        normal beats;
      - abruptly: RR <= PAC_ABRUPT_FRACTION x the RR before it (sinus
        arrhythmia speeds up gradually, a few % per beat);
      - and the rhythm resets after it (next RR >= PAC_RETURN_FRACTION x
        its RR, so there must be a next beat) or the next beat is premature
        too (a PAC couplet / run);
      - and the stretch before its QRS where the P wave sits (_p_shape) is
        unlike the dominant beats' (similarity < P_DIFFERENT_MAX): a PAC's
        P comes early and from another place in the atria; a beat that is
        only early (sinus arrhythmia) has the sinus P. (The ProSim "Atrial
        PAC" keeps the sinus P, only earlier -- 2026-11-12_11-07-11, P
        similarity 0.92-0.99 -- so it is not a PAC by this test, on purpose:
        a real PAC's P differs.) Judged only where the P wave is readable:
        the N beats around it match the dominant P (median >=
        P_READABLE_MIN) and this beat matches it by >= P_CONTRAST_MIN less.
    A beat right after an S that is still faster than the normal RR at the
    run's own rate continues the run. A run that holds a steady RR for
    TEMPLATE_SWITCH_BEATS beats is a new rhythm (a sudden rate change, e.g. a
    simulator rate step), so its beats go back to N and the normal RR
    restarts from them.
    Not S: beats inside an auto segment of NO_PAC_SEGMENTS (Atrial
    Fibrillation / Flutter, AV blocks, Sinus / plain Tachycardia and
    Bradycardia, SVT: the RR is irregular by itself there, or the segment is
    a rhythm of its own, not PACs; from PAC_SEGMENT_MARGIN_SEC before its
    start), beats right after a gap, a 'Q' or a paced beat ('P': the rate
    there is the pacemaker's), and the first beats until two normal RRs are
    known -- also again after such a segment.
    The running normal RR takes N beats that do not follow an S or a V
    (not the pause after an ectopic beat)."""
    no_pac = [(float(x['start_sec']), float(x['end_sec'])) for x in (segments or [])
              if str(x.get('label', '')) in NO_PAC_SEGMENTS]

    def in_no_pac(ts):
        return any(s_ - PAC_SEGMENT_MARGIN_SEC <= ts < e_ for s_, e_ in no_pac)

    ref: List[float] = []
    run: List[int] = []
    prev_label, prev_rr = None, None
    for k, b in enumerate(out):
        rr, lbl, c, ts = b.get('rr_ms'), b.get('label'), corrs[k], float(b['timestamp'])
        if rr is None or rr > RESTART_GAP_SEC * 1000.0 or lbl in ('Q', 'P'):
            ref, run = [], []
            prev_label, prev_rr = lbl, rr
            continue
        if lbl == 'V':
            run = []
            prev_label, prev_rr = 'V', rr
            continue
        if in_no_pac(ts):
            # no PAC there, and the normal RR is learnt again after it: the
            # dropped beats of a Mobitz I block (2 s RRs) made the normal RR
            # ~2000 ms, so every conducted beat read "premature" -- inside
            # (PR changing, so the P stretch differed too) and just after
            # the segment (simulator recordings: 15-30 false S each)
            ref, run = [], []
            prev_label, prev_rr = lbl, rr
            continue
        normal = float(np.median(ref)) if len(ref) >= 2 else None
        nxt = out[k + 1].get('rr_ms') if k + 1 < len(out) else None
        # no fixed QRS width limit: the shape test is the conduction test
        # (wide and different beats were taken by the PVC pass). A patient
        # whose own QRS is wide (INCART I70: 122-130 ms) has PACs measured
        # 142-180 ms with the dominant shape (0.83-0.94): 0 of 50 found
        # with a 110 ms limit.
        cand = lbl == 'N' and c is not None and c >= SHAPE_SAME_CORR and normal is not None
        if _DEBUG:
            b['_pac'] = (None if normal is None else round(rr / normal, 3),
                         None if not prev_rr else round(rr / prev_rr, 3),
                         None if not nxt else round(nxt / rr, 3),
                         None if c is None else round(c, 2), bool(cand), prev_label,
                         None if not pcorrs or pcorrs[k] is None else round(pcorrs[k], 2))
        if cand:
            premature = rr <= PAC_PREMATURE_FRACTION * normal
            abrupt = prev_rr is not None and prev_label not in ('V', 'S') and rr <= PAC_ABRUPT_FRACTION * prev_rr
            # the reset must be seen: with no next beat (the recording's
            # last) there is none -- the last beat, after a motion artifact,
            # read PAC (2026-10-29_16-33-38 at 132.87 s)
            resets = nxt is not None and nxt >= PAC_RETURN_FRACTION * rr
            next_early = nxt is not None and nxt <= PAC_PREMATURE_FRACTION * normal
            run_on = (prev_label == 'S' and run and rr <= RUN_CONTINUE_FRACTION * normal
                      and abs(rr - out[run[-1]]['rr_ms']) <= PAC_RUN_RR_FRACTION * out[run[-1]]['rr_ms'])
            # the P wave: a PAC's comes early and from elsewhere, so the
            # stretch before its QRS does not look like the dominant beats'
            # (INCART: 90% of PACs < 0.37 similarity; beats that were only
            # early -- sinus arrhythmia, a rate change -- median 0.79).
            # INCART / MIT-BIH, with this: +P 50 -> 84% / 60 -> 80%.
            p_differs = bool(pcorrs) and pcorrs[k] is not None and pcorrs[k] < P_DIFFERENT_MAX
            # ... and only where the P wave can be read at all: the N beats
            # around it (P_LOCAL_BEATS each side) match the dominant P
            # stretch (median >= P_READABLE_MIN) and this beat clearly less
            # (by >= P_CONTRAST_MIN). Where noisy leads or the previous T
            # wave fill the stretch, normal beats read "different" too and
            # an unreadable P is no evidence of an ectopic one: strong sinus
            # arrhythmia (beat-to-beat RR -30% / +50%) with muscle noise gave
            # 13 S on a healthy subject (2026-10-29_16-36-56; its normal
            # beats' own similarity under 0.50 for 15% of them). Annotated
            # PACs had neighbours at median 0.95 (INCART) / 0.91 (MIT-BIH),
            # false S 0.53 / 0.12. With it: INCART S false 71 -> 13, true
            # 226 -> 216; MIT-BIH false 304 -> 9, true 34 -> 32; simulator /
            # human recordings 19 false S fewer, no PAC lost; PVCs unchanged.
            if p_differs:
                near = [pcorrs[j] for j in range(max(0, k - P_LOCAL_BEATS), min(len(out), k + P_LOCAL_BEATS + 1))
                        if j != k and out[j].get('label') == 'N' and pcorrs[j] is not None]
                local_p = float(np.median(near)) if len(near) >= 3 else None
                p_differs = (local_p is not None and local_p >= P_READABLE_MIN
                             and local_p - pcorrs[k] >= P_CONTRAST_MIN)
            if run_on or (premature and abrupt and (resets or next_early) and p_differs):
                b['label'] = 'S'
                if not run_on:
                    run = []
                run.append(k)
                if len(run) >= TEMPLATE_SWITCH_BEATS:
                    rrs = np.array([out[j]['rr_ms'] for j in run[1:][-TEMPLATE_SWITCH_BEATS:]])
                    # steady: a rate change, not PACs. Tried to keep such a
                    # run S when its P differs from the sinus P (atrial runs,
                    # INCART I20): it found none more there, and read every
                    # simulator rate step that no auto segment names (60 -> 76
                    # bpm: the P stretch differs from the old rate's, ~0.4) as
                    # PACs -- 38 S in one recording.
                    if rrs.size and float(np.max(np.abs(rrs - np.median(rrs)))) <= STEADY_RR_FRACTION * float(np.median(rrs)):
                        for j in run:
                            out[j]['label'] = 'N'
                        ref = [float(x) for x in rrs][-LOCAL_RR_BEATS:]
                        run = []
                        prev_label, prev_rr = 'N', rr
                        continue
                prev_label, prev_rr = 'S', rr
                continue
        run = []
        if lbl == 'N' and prev_label not in ('V', 'S'):
            if normal is None or NORMAL_RR_MIN_FRACTION * normal <= rr <= NORMAL_RR_MAX_FRACTION * normal \
                    or (c is not None and c >= SHAPE_SAME_CORR):
                ref.append(rr)
                ref = ref[-LOCAL_RR_BEATS:]
        prev_label, prev_rr = lbl, rr


# ---------------------------------------------------------------------------
# paced beats
#
# A paced beat is made by a pacemaker pulse: the pacing spike (the pulse,
# ~0.5-2 ms) and right after it the chamber's response. Rule based, two
# ways of seeing the spike:
#   1. find_pacing_spikes -- a spike as this device records it (500 Hz,
#      12 bit, 200 counts per mV). The simulator's paced rhythms (TV Paced:
#      2026-11-27_14-23-01, -14-38-41; atrial paced runs inside
#      2026-11-12_13-54-02 at 14:06:20 and 2026-10-31_17-55-39 at 17:58:57)
#      jump ~10 mV in one sample on every recorded lead at once (most reach
#      the ADC rail), are at their full height within 1-2 samples and over
#      within ~12 ms (with an opposite-side recharge at times), the signal
#      flat right up to the jump, and come again within the pacing interval.
#      Each test (PACE_* below) is there for something else that jumps 1 mV
#      in a sample: a tall sharp human QRS (MIT-BIH 101, 113, 115, 209,
#      212, 220: rising for samples before the jump, most of its height
#      still to come), a clipped lead coming off the rail, the first cycle
#      of a mains burst, the Q notch of a QRS whose R goes to the rail, and
#      lone glitches (1-2 per recording). Over the 177 device recordings
#      and all 48 MIT-BIH records only the four paced ones keep spikes.
#      Each spike is read by what follows it: a QRS within PACE_CAPTURE_SEC
#      = ventricular capture ('V'; with an 'A' spike PACE_CONDUCT_SEC before
#      it, AV sequential); no QRS there but one (or a 'V' spike)
#      PACE_CONDUCT_SEC later = atrial pacing ('A'); nothing = no capture
#      ('none').
#   2. _narrow_spike_before -- a spike recorded smaller and wider: the
#      MIT-BIH paced records (102, 104, 107, 217: 360 Hz tape) show it as a
#      narrow peak (0.3-2 mV, ~8-12 ms wide at half height) just before the
#      wide paced QRS, the signal quiet before it -- in front of 41% of their
#      7028 paced beats (none in front of 95% of 217's: no spike to see
#      there), and of ~20 of the ~100 000 other beats of all 48 records
#      (4-8 per record, scattered). So it makes beats paced only where many
#      such beats (PACE_MIN_TEMPLATE_BEATS) share one wide QRS shape (the
#      paced template); then every wide beat that is not premature and has
#      its own spike, or the paced shape (a spike lost under noise or the
#      tape does not change the paced QRS), is paced. Runs after (1) on
#      every recording.
# Only a pacing spike makes a beat paced: a recording without them is
# labelled exactly as before. The RR between paced beats is no help on its
# own: on the MIT-BIH paced records the RR before a paced beat varies as
# much as sinus (IQR 42 ms, tape speed), so a fixed pacing interval cannot
# be relied on.
# ---------------------------------------------------------------------------
PACE_LEADS = ('I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')  # the recorded leads (the others are sums of I and II)
PACE_JUMP_MV = 1.0            # a spike changes this much in one sample ...
PACE_JUMP_LEADS = 3           # ... on at least this many recorded leads at once (+-1 sample)
PACE_RISE_SEC = 0.004         # ... is at PACE_RISE_FRACTION of its height this soon after its onset (the
PACE_RISE_FRACTION = 0.7      # sample its rise starts from; device spikes: the whole height in 1-2
                              # samples. A tall sharp QRS can also jump 1 mV in a sample -- MIT-BIH 101,
                              # 113, 115, 209, 212, 220 -- but its rise began samples earlier and most
                              # of its height comes later)
PACE_RETURN_SEC = 0.010       # ... and is back under half its height within this
PACE_ALONE_SEC = 0.025        # ... and, until this long after its onset, does not come back to half its
                              # height on the same side: one pulse, not the first cycle of a 50 / 60 Hz
                              # burst (its next peak 17-20 ms later; 2026-11-02_17-23-21 at 39.5 s) ...
PACE_SETTLE_SEC = 0.016       # ... and from this long after its onset to PACE_ALONE_SEC stays within half
                              # its height of the level before it on either side: the pulse, with its
                              # opposite-side recharge (2026-10-31_17-55-39: rail to rail in 12 ms), is
                              # over -- not the Q notch of a tall QRS whose R is still at the rail
                              # (2026-11-04_11-08-46 at 46.4 s)
PACE_QUIET_SEC = 0.035        # this stretch before a spike's onset ...
PACE_QUIET_FRACTION = 0.15    # ... changes per sample by at most this x the spike's own step (device
                              # spikes: flat up to the jump; mains / muscle: steep all along), and is
                              # not at the ADC rail (a clipped lead jumping off the rail is no spike:
                              # 2026-10-29_16-36-56, 2026-11-05_12-16-28)
PACE_REF_SEC = (0.035, 0.010)     # the level before a spike (20 ms mean: the spike itself not in it)
PACE_MERGE_SEC = 0.010        # steps closer together than this are one spike
PACE_PAIR_SEC = 2.0           # a spike counts only with another one this close: a pacemaker that paces
                              # paces again within its lower-rate interval (30/min and up). The spike-
                              # like artifacts left on the non-paced device recordings are lone ones, 1-2
                              # per recording, >= 3 s apart (a single paced beat between the patient's
                              # own beats is missed by this)
PACE_CAPTURE_SEC = (0.030, 0.120)  # a ventricular capture: its QRS is in this window after the spike ...
PACE_CAPTURE_MV = 0.5         # ... at least this far from the level before the spike (20 ms mean: mains out)
PACE_CONDUCT_SEC = (0.10, 0.35)    # an atrial spike: a QRS, or the ventricular spike, this long after it
PACE_V_SHARE = 0.5            # a capture's deflection >= this x the one PACE_CONDUCT_SEC later: the QRS is
                              # right after the spike. Measured (max over the recorded leads): ventricular
                              # 5.9 mV then 5.9; atrial 0.8 (its P wave) then 6.4 (the QRS); none 0.07 / 0.08
PACE_BEAT_TOL_SEC = 0.025     # a beat this close to a spike was taken at the spike
PACE_NARROW_LEADS = ('II', 'V1', 'V5', 'I')  # where (2) looks
PACE_NARROW_MS = 12.0         # (2): width at half prominence at most
PACE_NARROW_MIN_MV = 0.3      # (2): prominence at least
PACE_NARROW_WINDOW_SEC = (0.150, 0.030)  # (2): from this long before the R peak to this long after
PACE_NARROW_QUIET_SEC = (0.035, 0.010)   # (2): quiet stretch before the peak (from, to: seconds before
                                         # it; the peak's own ~5 ms rise not in it), by PACE_QUIET_FRACTION
PACE_NARROW_SWING_SEC = 0.020 # (2): within this after the peak the signal goes at most
PACE_NARROW_SWING = 2.0       # this x the peak's height past the level before it on the other side.
                              # The narrow r of an rS QRS drops straight into its deep S (INCART I73:
                              # 7x its height within 20 ms, every beat); a paced spike's tail stays
                              # near the level (MIT-BIH 102 / 104: none past 1x)
PACE_MIN_TEMPLATE_BEATS = 15  # (2): spiked wide beats of one shape needed for a paced template
PACE_TEMPLATE_CORR = 0.90     # (2): the shape of a paced beat against the paced template
PACE_NOT_PREMATURE = 0.90     # (2): a paced beat's RR >= this x the spiked beats' median (a PVC comes early)


def _mean20(x: np.ndarray, fs: float) -> np.ndarray:
    """Rows of x averaged over 20 ms (one 50 Hz cycle; 60 Hz mostly)."""
    w = max(1, int(round(0.020 * fs)))
    k = np.ones(w) / w
    return np.vstack([np.convolve(r, k, mode='same') for r in x])


def _spike_at(y: np.ndarray, dy: np.ndarray, k: int, rise: int, ret: int, quiet: int,
              settle: int, alone: int) -> bool:
    """Does lead y (dy = its sample-to-sample steps) show a pacing spike
    with the jump dy[k] in it (find_pacing_spikes)? Its onset is the sample
    its rise starts from (at most `rise` - 1 samples before the jump); from
    there PACE_RISE_FRACTION of its height within `rise` samples, back under
    half its height within `ret` of its top, not at half again on the same
    side until `alone` samples after the onset and within half of the level
    before it on either side from `settle` to `alone` samples after it, and
    the `quiet` samples before the onset flat and off the ADC rail."""
    step = float(dy[k])
    sgn = np.sign(step)
    small = PACE_QUIET_FRACTION * abs(step)
    on = k
    while on > k - rise + 1 and sgn * dy[on - 1] > small:
        on -= 1                                  # the rise began a sample earlier
    if on < quiet + 1 or on + 2 * ret + 2 > len(y) or on + alone + 1 > len(y):
        return False
    pre = y[on - quiet:on + 1]
    if np.any((pre <= ADC_RAIL_LO) | (pre >= ADC_RAIL_HI)):
        return False                             # from the rail: a clipped lead, not a spike
    if float(np.max(np.abs(dy[on - quiet:on]))) > small:
        return False                             # mains / muscle, or a QRS rising for longer
    base = y[on]
    top_seg = y[on + 1:on + 2 + ret]
    top_i = int(np.argmax(sgn * (top_seg - base)))
    height = abs(float(top_seg[top_i]) - base)
    if float(np.max(sgn * (y[on + 1:on + 1 + rise] - base))) < PACE_RISE_FRACTION * height:
        return False                             # most of its height still to come: a QRS
    after = sgn * (y[on + 1 + top_i:on + 2 + top_i + ret] - base)
    under = np.where(after < 0.5 * height)[0]
    if under.size == 0:
        return False                             # a step (square wave, electrode), not a spike
    end = on + 1 + top_i + int(under[0])         # first sample back under half
    later = sgn * (y[end + 1:on + alone + 1] - base)
    if later.size and float(np.max(later)) >= 0.5 * height:
        return False                             # a second peak: a burst
    tail = np.abs(y[on + settle:on + alone + 1] - base)
    return not (tail.size and float(np.max(tail)) >= 0.5 * height)     # not over yet: a QRS


def find_pacing_spikes(reader) -> List[Dict[str, Any]]:
    """Pacing spikes of the whole recording as this device records them
    ('paced beats' above, way 1): [{'t' (s), 'kind' 'V' | 'A' | 'none',
    'after_a' (a 'V' with an 'A' spike PACE_CONDUCT_SEC before it)}] in time
    order; a spike with no other one within PACE_PAIR_SEC is left out. From
    the signal alone: for the beat labels here, and for any rule that must
    not read a spike as a beat or as noise."""
    from .analysis_pipeline import ADC_COUNTS_PER_MV
    fs = float(reader.fs)
    names = list(reader.lead_names)
    total = float(reader.duration_sec)
    rec = [names.index(n) for n in PACE_LEADS if n in names] or list(range(len(names)))
    jump = PACE_JUMP_MV * ADC_COUNTS_PER_MV
    level = PACE_CAPTURE_MV * ADC_COUNTS_PER_MV
    ret = max(1, int(round(PACE_RETURN_SEC * fs)))
    rise = max(1, int(round(PACE_RISE_SEC * fs)))
    quiet = int(round(PACE_QUIET_SEC * fs))
    alone = int(round(PACE_ALONE_SEC * fs))
    settle = int(round(PACE_SETTLE_SEC * fs))
    r0, r1 = int(round(PACE_REF_SEC[0] * fs)), int(round(PACE_REF_SEC[1] * fs))
    merge = max(1, int(round(PACE_MERGE_SEC * fs)))
    c0, c1 = int(round(PACE_CAPTURE_SEC[0] * fs)), int(round(PACE_CAPTURE_SEC[1] * fs))
    a0, a1 = int(round(PACE_CONDUCT_SEC[0] * fs)), int(round(PACE_CONDUCT_SEC[1] * fs))
    pad = PACE_CONDUCT_SEC[1] + max(PACE_QUIET_SEC, PACE_REF_SEC[0]) + 0.05
    found = []                     # (t, captured, something later)
    t = 0.0
    while t < total:
        te = min(t + PIECE_SEC, total)
        ps, pe = max(0.0, t - pad), min(total, te + pad)
        data = reader.read_range(ps, pe)
        if data is not None and data.shape[1] > quiet + a1 + ret + 2:
            x = np.asarray(data, dtype=float)[rec]
            d = np.diff(x, axis=1)                       # d[:, k] = x[:, k + 1] - x[:, k]
            big = np.abs(d) >= jump
            near = big.copy()
            near[:, 1:] |= big[:, :-1]
            near[:, :-1] |= big[:, 1:]
            cand = np.where((near.sum(axis=0) >= PACE_JUMP_LEADS) & big.any(axis=0))[0]
            sm = None
            last = -10 * merge
            f0 = int(ps * fs)                            # read_range's first sample
            for k in cand:
                if k - last < merge or k < max(quiet + rise, r0) or k + 1 + a1 >= x.shape[1]:
                    continue
                # a spike on at least one of the jumping leads (muscle noise
                # can cover the others: 2026-11-27_14-38-41 at 365 s)
                if not any(_spike_at(x[lead], d[lead], k, rise, ret, quiet, settle, alone)
                           for lead in np.where(big[:, k])[0]):
                    continue
                last = k
                s_ = k + 1
                ts = (f0 + s_) / fs
                if not (t <= ts < te):
                    continue
                if sm is None:
                    sm = _mean20(x, fs)
                ref = np.median(sm[:, s_ - r0:s_ - r1], axis=1)[:, None]
                now = float(np.max(np.abs(sm[:, s_ + c0:s_ + c1] - ref)))
                then = float(np.max(np.abs(sm[:, s_ + a0:s_ + a1] - ref)))
                # the QRS right after the spike, not only a P wave before a
                # later QRS (atrial pacing: P 0.8 mV, QRS 6.4 mV 200 ms on)
                captured = now >= level and now >= PACE_V_SHARE * then
                found.append((ts, captured, then >= level))
        t += PIECE_SEC
    times = [f[0] for f in found]
    out = []
    for i, (ts, captured, later) in enumerate(found):
        if not ((i > 0 and ts - times[i - 1] <= PACE_PAIR_SEC)
                or (i + 1 < len(times) and times[i + 1] - ts <= PACE_PAIR_SEC)):
            continue                     # a lone spike: an artifact, not pacing
        kind = 'V' if captured else ('A' if later else 'none')
        out.append({'t': round(ts, 4), 'kind': kind, 'after_a': False})
    for i, sp in enumerate(out):
        if sp['kind'] != 'V':
            continue
        for j in range(i - 1, -1, -1):
            dt = sp['t'] - out[j]['t']
            if dt > PACE_CONDUCT_SEC[1]:
                break
            if out[j]['kind'] == 'A' and dt >= PACE_CONDUCT_SEC[0]:
                sp['after_a'] = True
                break
    return out


def _narrow_spike_before(x: np.ndarray, r: int, fs: float) -> bool:
    """A pacing spike recorded small and wide ('paced beats' above, way 2) in
    front of the beat whose R peak is sample r of the raw leads x (rows): on
    one lead, a peak at most PACE_NARROW_MS wide at half its prominence
    (>= PACE_NARROW_MIN_MV), from PACE_NARROW_WINDOW_SEC before r to after
    it, with PACE_NARROW_QUIET_SEC before it quiet (steepest step <=
    PACE_QUIET_FRACTION x its own) and no swing past PACE_NARROW_SWING x its
    height to the other side within PACE_NARROW_SWING_SEC after it."""
    from scipy.signal import find_peaks, peak_widths
    from .analysis_pipeline import ADC_COUNTS_PER_MV
    a, b = r - int(PACE_NARROW_WINDOW_SEC[0] * fs), r + int(PACE_NARROW_WINDOW_SEC[1] * fs)
    q0, q1 = int(round(PACE_NARROW_QUIET_SEC[0] * fs)), int(round(PACE_NARROW_QUIET_SEC[1] * fs))
    swing = max(1, int(round(PACE_NARROW_SWING_SEC * fs)))
    lo, hi = a - q0 - 2, b + max(int(0.04 * fs), swing + 1)
    if lo < 0 or hi > x.shape[1]:
        return False
    prom = PACE_NARROW_MIN_MV * ADC_COUNTS_PER_MV
    max_w = PACE_NARROW_MS / 1000.0 * fs
    for row in x:
        seg = np.asarray(row[lo:hi], dtype=float)
        ds = np.abs(np.diff(seg))
        for pol in (1.0, -1.0):
            y = pol * seg
            pk, _ = find_peaks(y, prominence=prom)
            if pk.size == 0:
                continue
            widths = peak_widths(y, pk, rel_height=0.5)[0]
            for pk_i, w in zip(pk, widths):
                if w > max_w or not (a - lo <= pk_i <= b - lo) or pk_i < q0 + 1:
                    continue
                own = float(ds[max(0, pk_i - 5):pk_i + 2].max())
                quiet = float(ds[pk_i - q0:pk_i - q1].max())
                if not (own > 0 and quiet <= PACE_QUIET_FRACTION * own):
                    continue
                level = float(np.median(y[pk_i - q0:pk_i - q1]))
                height = float(y[pk_i]) - level
                if float(np.min(y[pk_i:pk_i + swing + 1])) - level < -PACE_NARROW_SWING * height:
                    continue                 # the r of an rS QRS, not a spike
                return True
    return False


def _label_paced(reader, out, corrs, pcorrs, feats, pieces, fs):
    """Paced beats in the main loop's beats `out` (feats / pieces alongside,
    one entry per beat): label 'P' and 'pace' ('V' / 'A' / 'AV'), and the
    spikes the beat finder took for beats taken out (an atrial spike, a
    spike with no capture, the second reading of a paced complex) -- the
    next beat's RR is then measured from the beat before. Returns the new
    (out, corrs, pcorrs). See 'paced beats' above."""
    if not out:
        return out, corrs, pcorrs
    t_beats = np.array([float(b['timestamp']) for b in out])
    real = [not b.get('_not_beat') for b in out]
    drop = set()

    def beats_in(t0, t1):
        lo, hi = int(np.searchsorted(t_beats, t0)), int(np.searchsorted(t_beats, t1, side='right'))
        return [i for i in range(lo, hi) if real[i] and i not in drop]

    try:
        spikes = find_pacing_spikes(reader)
    except Exception as e:
        print(f"[Ectopic beats] pacing spike search error: {e}")
        spikes = []
    tol = PACE_BEAT_TOL_SEC
    # ventricular captures first, so an atrial spike's QRS search below
    # sees which beat is already the V spike's
    for sp in spikes:
        if sp['kind'] != 'V':
            continue
        own = beats_in(sp['t'] - tol, sp['t'] + PACE_CAPTURE_SEC[1])
        if own:
            out[own[0]]['label'] = 'P'
            out[own[0]]['pace'] = 'AV' if sp['after_a'] else 'V'
            drop.update(own[1:])                 # the same complex read twice
    for sp in spikes:
        if sp['kind'] == 'V':
            continue
        drop.update(beats_in(sp['t'] - tol, sp['t'] + tol))   # the spike itself: no heartbeat
        if sp['kind'] == 'A':
            nxt = beats_in(sp['t'] + PACE_CONDUCT_SEC[0], sp['t'] + PACE_CONDUCT_SEC[1])
            if nxt and out[nxt[0]].get('pace') is None:
                out[nxt[0]]['label'] = 'P'
                out[nxt[0]]['pace'] = 'A'

    # way 2: spiked wide beats of one shape (the beats way 1 made paced
    # count among them, and keep their pace)
    names = list(reader.lead_names)
    total = float(reader.duration_sec)
    rows = [names.index(n) for n in PACE_NARROW_LEADS if n in names]
    wide = [k for k, b in enumerate(out)
            if real[k] and k not in drop and b.get('qrs_ms') is not None and b['qrs_ms'] >= WIDE_QRS_MS]
    spiked = []
    raw_piece, raw = None, None
    for k in wide:
        pi = feats[k][2]
        if pi != raw_piece:
            ps = pieces[pi][0]
            data = reader.read_range(ps, min(total, ps + PIECE_SEC + 2 * PAD_SEC))
            raw_piece, raw = pi, (None if data is None else np.asarray(data, dtype=float)[rows])
        if raw is not None and rows and _narrow_spike_before(raw, feats[k][4], fs):
            spiked.append(k)
    if len(spiked) >= PACE_MIN_TEMPLATE_BEATS:
        def shape_of(k):
            f = feats[k]
            return _shape(pieces[f[2]][1], f[3], fs, pieces[f[2]][2])

        def corr_of(k, tpl):
            f = feats[k]
            return _corr(pieces[f[2]][1], f[3], tpl, fs, pieces[f[2]][2])
        pool = spiked[::max(1, len(spiked) // 200)]
        shapes = [v for v in (shape_of(k) for k in pool) if v is not None]
        if shapes:
            m = np.nan_to_num(np.array(shapes))
            tpl = m[int(np.argmax(np.einsum('ilw,jlw->ij', m, m).sum(axis=1)))].copy()
            members = [k for k in spiked if corr_of(k, tpl) >= PACE_TEMPLATE_CORR]
            rrs = [out[k]['rr_ms'] for k in members if out[k].get('rr_ms')]
            if len(members) >= PACE_MIN_TEMPLATE_BEATS and rrs:
                paced_rr = float(np.median(rrs))
                spiked_set = set(spiked)
                for k in wide:
                    if out[k]['label'] == 'P':
                        continue                 # way 1's: its pace ('AV', ...) stays
                    rr = out[k].get('rr_ms')
                    # a paced recording: a beat with its own spike, or of the
                    # paced shape (its spike lost), that is not premature
                    if (rr is None or rr >= PACE_NOT_PREMATURE * paced_rr) \
                            and (k in spiked_set or corr_of(k, tpl) >= PACE_TEMPLATE_CORR):
                        out[k]['label'] = 'P'
                        out[k]['pace'] = 'V'

    if not drop:
        return out, corrs, pcorrs
    new_out, new_c, new_p = [], [], []
    prev_t = None
    for k, b in enumerate(out):
        if k in drop:
            continue
        if real[k]:
            if prev_t is not None:
                b['rr_ms'] = round((float(b['timestamp']) - prev_t) * 1000.0, 1)
            prev_t = float(b['timestamp'])
        new_out.append(b)
        new_c.append(corrs[k])
        new_p.append(pcorrs[k])
    return new_out, new_c, new_p


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
            # [time, QRS ms, piece, shape centre (moved to the QRS centre
            # below), R peak sample (kept: see v_corr in the main loop)]
            feats.append([p, None, len(pieces) - 1, int(round((p - ps) * fs)), int(round((p - ps) * fs))])
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
        premature; seen: the first PVC of a trigeminy at beat 3 read N).
        Beats under a mains (50 / 60 Hz) Artifact are classified but learnt
        from only when there is no template yet (a recording that starts
        under the interference): the segment detector does not name a
        rhythm under it -- VT ran on under one (2026-11-03_11-24-57, VT
        135.3-145.0 and 155.5-187.6 s, mains in between), its beats became
        the template and 86 VT beats read N."""
        group = []
        for k in range(k0, len(feats)):
            if k > k0 and restart_at[k]:
                break
            if (feats[k][1] is not None and not blocked(feats[k][0]) and not in_vt(feats[k][0])
                    and (template is None or not in_mains(feats[k][0]))):
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

    mains_spans = [(float(x['start_sec']), float(x['end_sec'])) for x in (segments or [])
                   if str(x.get('label', '')) == 'Artifact' and x.get('artifact_kind') in MAINS_ARTIFACT_KINDS
                   and x.get('start_sec') is not None and x.get('end_sec') is not None]

    def in_mains(ts):
        return any(s_ <= ts < e_ for s_, e_ in mains_spans)

    no_pvc_spans = [(float(x['start_sec']), float(x['end_sec'])) for x in (segments or [])
                    if str(x.get('label', '')) in NO_PVC_SEGMENTS
                    and x.get('start_sec') is not None and x.get('end_sec') is not None]

    def no_pvc(ts):
        return any(s_ <= ts < e_ for s_, e_ in no_pvc_spans)

    template, ref_rr = None, []

    out = []
    corrs: List[Optional[float]] = []  # each beat's similarity to the template (None: not measured)
    pcorrs: List[Optional[float]] = []  # each beat's P-wave stretch vs the dominant beats' (None: not measured)
    nrrs: List[Optional[float]] = []    # the normal RR (ms) each beat was judged against (None: not judged)
    p_template = None
    v_cands: List[tuple] = []           # (index in out, shape, similarity, qrs) of each V
    pending: List[np.ndarray] = []     # new-shape beats at the normal rate, in a row
    pending_first = 0                   # index in out of pending's first beat
    run: List[tuple] = []               # the current V run: (beat, rr, shape)
    spikes: List[tuple] = []           # (index in out, normal RR) of N beats too narrow and unlike any beat
    prev_label = None
    # The last beat whose shape was judged, 'V' or 'N' (a 'Q' beat -- shape
    # or QRS not measurable -- is no evidence either way and leaves it).
    # A V run (`run`, the slow-run test below) goes on over Q beats and
    # ends at an N beat.
    last_judged = None
    prev_t = None
    for k, f in enumerate(feats):
        ts, qrs = f[0], f[1]
        if restart_at[k] and not blocked(ts):
            template, ref_rr = start(k)
            pending, run, prev_label, last_judged = [], [], None, None
            p_template = None
        after_ectopic = prev_label == 'V'
        rr = (ts - prev_t) * 1000.0 if prev_t is not None else None
        # Not a beat: a narrow "beat" < POST_V_MIN_RR_MS after a V is that
        # PVC's own T wave (the ventricles have not recovered yet; its QRS
        # alone is ~270 ms). It is dropped from the result (at the end, if
        # that V is still V) and the next beat's RR is measured from the V.
        # Seen: 250 ms after the last beat of a PVC run, QRS 100 ms, read V
        # and made a 5-beat run 6 (2026-11-12_12-30-47, 87.65 s and
        # 100.96 s). A wide one is kept: a VT faster than 200 bpm has RRs
        # this short.
        if (prev_label == 'V' and rr is not None and rr < POST_V_MIN_RR_MS
                and not (qrs is not None and qrs >= WIDE_QRS_MS)):
            out.append({'timestamp': round(ts, 3), 'label': 'Q', 'rr_ms': round(rr, 1),
                        'qrs_ms': None if qrs is None else round(qrs, 1), '_not_beat': True})
            corrs.append(None)
            pcorrs.append(None)
            nrrs.append(None)
            continue
        prev_t = ts
        beat = {'timestamp': round(ts, 3), 'label': 'N',
                'rr_ms': None if rr is None else round(rr, 1),
                'qrs_ms': None if qrs is None else round(qrs, 1)}
        out.append(beat)
        corrs.append(None)
        pcorrs.append(None)
        nrrs.append(None)
        if blocked(ts) or template is None:
            beat['label'] = 'Q'
            prev_label = 'Q'
            last_judged = None
            pending = []
            run = []
            continue
        shape_leads, clip = pieces[f[2]][1], pieces[f[2]][2]
        corr = _corr(shape_leads, f[3], template, fs, clip)
        corrs[-1] = None if np.isnan(corr) else corr
        pv = _p_shape(shape_leads, f[3], fs, clip)
        if pv is not None and p_template is not None and pv.shape == p_template.shape:
            pc = _similarity(pv, p_template)
            pcorrs[-1] = None if np.isnan(pc) else pc
        if np.isnan(corr):          # too few usable leads to judge the shape
            beat['label'] = 'Q'
            prev_label = 'Q'
            pending = []
            continue
        normal_rr = float(np.median(ref_rr)) if len(ref_rr) >= 2 else None
        nrrs[-1] = normal_rr
        premature = rr is not None and normal_rr is not None and rr <= PREMATURE_FRACTION * normal_rr
        different = corr < SHAPE_SAME_CORR
        # For the PVC test only: the shape also judged centred on the R peak,
        # the better of the two. The shape window sits on the measured QRS
        # centre, and where AF f waves move the QRS edges the window moves
        # with them: a normal 88 ms beat read QRS 172 ms and similarity 0.10
        # (0.99 on its R peak) -- premature in AF, "wide" and "different":
        # V (2026-11-17_13-39-33, 13:50:35 / :40 / :42; the same in the
        # simulator AF of 2026-10-10_12-21-19, 2026-10-20_12-26-09,
        # 2026-10-30_16-03-13). A PVC is unlike the dominant beat however it
        # is aligned. MIT-BIH false V 119 -> 99, true 867 -> 865 (207: its
        # template there is the PVC itself); INCART false V 59 -> 49, true
        # unchanged. Not for the PAC test or the template (similarity there
        # as before): there it made 19 INCART normal beats PACs.
        v_corr = corr
        if f[4] != f[3]:
            corr_r = _corr(shape_leads, f[4], template, fs, clip)
            if not np.isnan(corr_r) and corr_r > corr:
                v_corr = corr_r
        # a beat whose QRS cannot be bounded is no PVC evidence: on INCART
        # most such "beats" were noise / T waves Pan-Tompkins took (146 of
        # 321 false V had no annotated beat at all)
        very_different = v_corr < VERY_DIFFERENT_CORR and qrs is not None and qrs >= MIN_PVC_QRS_MS
        wide = qrs is not None and qrs >= WIDE_QRS_MS
        run_on = (prev_label == 'V' and rr is not None and normal_rr is not None
                  and rr <= RUN_CONTINUE_FRACTION * normal_rr)
        if _DEBUG:
            beat['_dbg'] = (round(corr, 2), None if normal_rr is None else round(normal_rr), bool(premature), bool(run_on))
            v0 = _shape(shape_leads, f[3], fs, clip)
            beat['_dbg_leads'] = None if v0 is None else [round(float(x), 2) for x in np.sum(v0 * template, axis=1)]
            beat['_dbg_shape'] = v0
        # V: premature (or continuing a ventricular run) with a shape unlike
        # the dominant beat's and a wide QRS -- or a shape so unlike it that
        # a QRS measured under WIDE_QRS_MS (>= MIN_PVC_QRS_MS) does not
        # matter (seen: INCART PVCs measured 108-118 ms, similarity 0.0-0.4).
        if (premature or run_on) and (very_different or (v_corr < SHAPE_SAME_CORR and wide)) and not no_pvc(ts):
            beat['label'] = 'V'
            prev_label = 'V'
            pending = []
            if not run_on and last_judged != 'V':
                run = []
            last_judged = 'V'
            v_shape = _shape(shape_leads, f[3], fs, clip)
            v_cands.append((len(out) - 1, v_shape, v_corr, qrs))
            run.append((beat, rr, v_shape))
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
                    last_judged = 'N'
                    run = []
            continue
        if qrs is None:
            beat['label'] = 'Q'
            prev_label = 'Q'
            pending = []
            continue
        # Only an N beat ends the V run: a Q beat broke it, and a rhythm
        # change with clipped leads never reached TEMPLATE_SWITCH_BEATS V in
        # a row (Third-degree block 30 bpm, then 80 bpm LBBB: V Q Q V Q Q V
        # ..., the template stayed the narrow escape beat and 5 LBBB beats
        # read V, 2026-11-17_13-39-33 14:10:40-14:10:54; now the 8th V
        # turns the run into the new rhythm). MIT-BIH / INCART unchanged.
        run = []
        prev_label = 'N'
        last_judged = 'N'
        if qrs < MIN_QRS_MS and v_corr < VERY_DIFFERENT_CORR:
            spikes.append((len(out) - 1, normal_rr))
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
            # the P-wave template: dominant-shape beats at the normal rate
            # (not premature, not right after an ectopic beat)
            if pv is not None and not premature and not after_ectopic:
                p_template = pv.copy() if p_template is None else _blend(p_template, pv, TEMPLATE_ALPHA)
        elif different and not premature:
            if pending and not (_similarity(pending[-1], v) >= SHAPE_SAME_CORR):
                pending = []
            if not pending:
                pending_first = len(out) - 1
            pending.append(v)
            if len(pending) >= TEMPLATE_SWITCH_BEATS:
                template = _blend(template, _nanmean_rows(np.array(pending)), 1.0)
                pending = []
                # The beat right before the new shape's run, when V with that
                # same shape, was the new rhythm's first beat, not a PVC: a
                # rhythm (conduction) change often starts a little early and
                # is premature and unlike the old template only against the
                # old rhythm. Seen: RBBB -> LBBB at 162.3 s, first LBBB beat
                # 648 ms after a 750 ms RR (2026-10-29_15-05-05).
                kp = pending_first - 1
                while kp >= 0 and out[kp].get('_not_beat'):
                    kp -= 1
                if kp >= 0 and out[kp]['label'] == 'V':
                    vs = next((c[1] for c in reversed(v_cands) if c[0] == kp), None)
                    if vs is not None and vs.shape == template.shape:
                        sim_ = _similarity(vs, template)
                        if not np.isnan(sim_) and sim_ >= SHAPE_SAME_CORR:
                            out[kp]['label'] = 'N'
        elif not different:
            pending = []

    # Late PVCs: a beat the pass above left N because it did not come early.
    # No conducted beat is wide, of a shape very unlike the dominant beat's
    # and without a sinus P wave in front of it (P stretch similarity <
    # LATE_PVC_P_MAX) -- and a lone one has dominant-shape N beats on both
    # sides. The neighbours matter: a bundle branch block or paced rhythm
    # starts with a run of such beats (the template needs TEMPLATE_SWITCH_BEATS
    # of them to follow), and with only the three tests above INCART read 301
    # normal beats and MIT-BIH 293 right bundle branch block + 190 paced beats
    # as PVCs (+P 97.6 -> 88.4% / 88.7 -> 56.7%). Even with them the
    # candidates were 23 true in 76 on INCART (16 in 40 on MIT-BIH): a late
    # PVC must repeat, see the repeat test below (>= LATE_MIN_PARTNERS other V
    # beats at similarity >= LATE_PARTNER_CORR: INCART 11 true in 11).
    # And it is followed by a pause (the next beat's RR >= LATE_PVC_PAUSE_
    # FRACTION x the normal RR): a PVC breaks the rhythm -- it comes early or,
    # when it does not, the next beat comes late (the sinus beat it blocked
    # is not conducted). A beat neither early nor followed by a pause is the
    # rhythm's own beat, whatever its shape read. First 5 min of MIT-BIH /
    # INCART: the 21 true late PVCs had a next RR of 1.00-1.76 x normal
    # (all but one >= 1.05), the 13 false ones (MIT 232, aberrant PACs and
    # RBBB beats after long pauses) 0.94-1.88 (10 under 1.05) -> MIT true
    # 10 -> 10, false 13 -> 3; INCART true 11 -> 10, false 0. Seen without
    # it: 7 "late PVCs" in a regular tachycardia, RR 400 ms before and after
    # each (2026-11-12_16-19-27, 16:22:53.91 - 16:23:43.65).
    late_idx = set()
    for k in range(1, len(out) - 1):
        b = out[k]
        if b['label'] != 'N' or corrs[k] is None or pcorrs[k] is None or b['qrs_ms'] is None or no_pvc(feats[k][0]):
            continue
        if not (corrs[k] < VERY_DIFFERENT_CORR and b['qrs_ms'] >= WIDE_QRS_MS and pcorrs[k] < LATE_PVC_P_MAX):
            continue
        next_rr = out[k + 1].get('rr_ms')
        if nrrs[k] is None or next_rr is None or next_rr < LATE_PVC_PAUSE_FRACTION * nrrs[k]:
            continue
        if not all(out[j]['label'] == 'N' and corrs[j] is not None and corrs[j] >= SHAPE_SAME_CORR
                   for j in (k - 1, k + 1)):
            continue
        fk = feats[k]
        b['label'] = 'V'
        late_idx.add(k)
        v_cands.append((k, _shape(pieces[fk[2]][1], fk[3], fs, pieces[fk[2]][2]), corrs[k], b['qrs_ms']))
    v_cands.sort(key=lambda c: c[0])

    # A PVC repeats: beats from one ventricular focus look alike (simulator
    # PVCs 1.00 to each other; INCART: 95% of true PVCs >= 0.84 to another
    # one). Noise that makes a normal beat "different" does not repeat
    # (a human recording's motion noise, 2026-10-29_16-36-56 / -16-33-38:
    # 10 + 9 false V, each <= 0.66 to any other). A V with no other V
    # (beyond its own neighbours) >= PARTNER_CORR goes back to N; a V that
    # is the only one in the recording stays V only with clear evidence
    # (very different and wide, and _lone_v_is_ectopic). INCART, threshold
    # 0.7: true PVCs kept 2485 of 2559, false V 111 -> 58.
    # Only V beats it can be compared with count as "others": two shapes
    # with fewer than MIN_SHAPE_LEADS usable leads in common (the rest
    # clipped at the ADC rail) give no similarity, which is no evidence that
    # the shapes differ. Counted as compared, they made every PVC of a large
    # simulator PVC train "unlike every other V" (I, II, V2 at the rail, 2
    # leads left; 50 Hz on top) and 13 of 14 went back to N
    # (2026-11-17_13-39-33, 14:00:59-14:01:45); a V nothing can be compared
    # with is judged as the only V (the clear-evidence test), 11 of them
    # stay V. MIT-BIH / INCART and every other simulator / human recording
    # unchanged.
    live = [c for c in v_cands if out[c[0]]['label'] == 'V' and c[1] is not None]
    # Late PVCs first: one has no early coupling to vouch for it, so it stays
    # V only when several other V beats of the recording have its shape. A
    # late candidate that fails is N before the test below, which must not
    # count it: still in the list, it made a lone premature PVC (QRS 154 ms,
    # 2026-10-30_13-51-01 at 4.9 s) look not-alone and lose its "clear
    # evidence" pass.
    for i, v, c, q in live:
        if i not in late_idx:
            continue
        matches = 0
        for j, u, *_ in live:
            if abs(j - i) > 1 and u.shape == v.shape:
                sim = _similarity(v, u)
                if not np.isnan(sim) and sim >= LATE_PARTNER_CORR:
                    matches += 1
        if matches < LATE_MIN_PARTNERS:
            out[i]['label'] = 'N'
    live = [cand for cand in live if out[cand[0]]['label'] == 'V']
    for n_, (i, v, c, q) in enumerate(live):
        best, others = -1.0, 0
        if i in late_idx:
            continue
        for step in range(1, len(live)):
            for m in (n_ - step, n_ + step):
                if not 0 <= m < len(live):
                    continue
                j, u = live[m][0], live[m][1]
                if abs(j - i) <= 1 or u.shape != v.shape:
                    continue
                sim = _similarity(v, u)
                if np.isnan(sim):
                    continue          # < MIN_SHAPE_LEADS leads in common: not compared
                others += 1
                if sim > best:
                    best = sim
            if best >= PARTNER_CORR or others >= PARTNER_MAX_COMPARE:
                break
        if best >= PARTNER_CORR:
            continue
        if others == 0 and c < VERY_DIFFERENT_CORR and q is not None and q >= WIDE_QRS_MS and _lone_v_is_ectopic(out, pcorrs, i):
            continue
        # A PVC whose focus fires once among other PVCs (multiform PVCs)
        # has no partner either; it stays V on the full LITFL picture of
        # its own: very different, QRS >= UNIQUE_PVC_QRS_MS, early against
        # the beat before it with no sinus P (_lone_v_is_ectopic), a full
        # compensatory pause (_full_pause), and the dominant rhythm right
        # after it (the next beat N with the dominant shape -- a rhythm
        # change's first beat is followed by more of its own shape: RBBB ->
        # LBBB 2026-11-12_13-54-02 at 858.8 s, RBBB -> AF 2026-10-30_16-03-13
        # at 1150.5 s). MIT-BIH true PVCs 865 -> 874 (false 99 -> 99),
        # INCART 2497 -> 2523 (false 46 -> 48); simulator: two multiform
        # PVCs (2026-11-12_11-19-27 at 11.0 / 14.1 s) and 3 more found, no
        # false V.
        if (c < VERY_DIFFERENT_CORR and q is not None and q >= UNIQUE_PVC_QRS_MS
                and _lone_v_is_ectopic(out, pcorrs, i) and _full_pause(out, i)
                and i + 1 < len(out) and out[i + 1].get('label') == 'N'
                and corrs[i + 1] is not None and corrs[i + 1] >= SHAPE_SAME_CORR):
            continue
        out[i]['label'] = 'N'
    # Not a beat: an N "beat" narrower than any QRS (< MIN_QRS_MS), unlike
    # the dominant beat (< VERY_DIFFERENT_CORR), that splits one normal cycle
    # in two (its RR + the next beat's = the normal RR, within
    # SPLIT_RR_FRACTION) is a step / spike artifact Pan-Tompkins took. As N
    # it made the next real beat "early": an electrode artifact 704 ms into
    # a 1000 ms cycle (QRS 26 ms, similarity 0.06) and the sinus beat 294 ms
    # after it read PAC (2026-10-30_12-01-08 at 160.35 s; 2026-11-16_14-56-18
    # at 383.0 s). It is 'Q' (and _classify_pacs restarts after a Q). Over
    # all recordings 10 such beats, every one a baseline step; MIT-BIH 1,
    # INCART 0; no PVC or PAC count changed except those two false PACs.
    for i, nrr in spikes:
        if out[i]['label'] != 'N' or nrr is None or i + 1 >= len(out):
            continue
        r0, r1 = out[i].get('rr_ms'), out[i + 1].get('rr_ms')
        if r0 is not None and r1 is not None and abs(r0 + r1 - nrr) <= SPLIT_RR_FRACTION * nrr:
            out[i]['label'] = 'Q'
    # Paced beats (and the pacing spikes taken for beats out) before the
    # PAC pass, which restarts at a paced beat.
    out, corrs, pcorrs = _label_paced(reader, out, corrs, pcorrs, feats, pieces, fs)
    _classify_pacs(out, corrs, segments, pcorrs)
    # T waves of a V (see the main loop) are dropped -- when that V is still
    # V. One the repeat test put back to N was no PVC, so what followed it
    # may be a real beat: it stays, unclassified ('Q').
    result: List[Dict[str, Any]] = []
    for b in out:
        if b.pop('_not_beat', False) and result and result[-1]['label'] == 'V':
            continue
        result.append(b)
    return result
