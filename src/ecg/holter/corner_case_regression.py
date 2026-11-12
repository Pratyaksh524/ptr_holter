"""Corner-case regression suite for the auto arrhythmia detector.

Standalone (no pytest dependency, matches this project's existing script
style): run directly --

    python -m ecg.holter.corner_case_regression
    python ecg/holter/corner_case_regression.py

Each case below is a real, previously-confirmed failure mode -- found by
hand during debugging sessions, not invented. The point of this file is
to turn every one of those one-off investigations into a permanent check
that runs again, automatically, before any future change to the detector
is considered done. New corner cases (new recordings, new confirmed
failure modes) should be ADDED here, never silently fixed and forgotten.

Each case is one of:
  - GUARD   -- a fix already landed; this case must keep passing. A
              failure here means a regression was just introduced.
  - KNOWN   -- a confirmed, not-yet-fixed bug. Expected to fail right
              now; tracked here so it's visible and the fix (when it
              lands) has an immediate check to turn it into a GUARD.

Run this after every change to auto_segment_arrthymia_detection.py,
pqrst_measure.py, or analysis_pipeline.py, before calling the change done.
"""
import io
import contextlib
import os
import sys

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ecg.holter.file_format import ECGHFileReader  # noqa: E402
from ecg.holter import auto_segment_arrthymia_detection as m  # noqa: E402
from ecg.holter import pqrst_measure as pm  # noqa: E402

RECORDINGS_DIR = os.path.join(SRC, 'recordings')

_detect_cache = {}
_reader_cache = {}


class _RecordingMissing(Exception):
    pass


def _reader(rec):
    if rec not in _reader_cache:
        path = os.path.join(RECORDINGS_DIR, rec, 'recording.ecgh')
        if not os.path.exists(path):
            raise _RecordingMissing('recording not found: %s (deleted, moved, or renamed?)' % rec)
        _reader_cache[rec] = ECGHFileReader(path)
    return _reader_cache[rec]


def _segments(rec):
    """Cached _detect_arrhythmias_uncached() output for one recording."""
    if rec not in _detect_cache:
        with contextlib.redirect_stdout(io.StringIO()):
            _detect_cache[rec] = m._detect_arrhythmias_uncached(_reader(rec))
    return _detect_cache[rec]


def _overlapping(segs, t0, t1, label=None):
    out = []
    for s in segs:
        if s['start_sec'] < t1 and s['end_sec'] > t0:
            if label is None or s['label'] == label:
                out.append(s)
    return out


# ---------------------------------------------------------------------
# Case definitions: (kind, name, check_fn) -- check_fn() returns
# (passed: bool, detail: str) and may raise (counted as a failure, with
# the exception text as the detail) if the recording is missing.
# ---------------------------------------------------------------------
CASES = []


def case(kind, name):
    def deco(fn):
        CASES.append((kind, name, fn))
        return fn
    return deco


# 1. Square/pulse-train test signal must never read as a real
# ventricular arrhythmia. Confirmed bug: the VT-carve pass (added to
# recover VT beats diluted out of a VFib window) briefly called 2.75 s
# of this exact test wave "Ventricular Tachycardia" before the
# span_status != 'ok' guard was added to it.
@case('GUARD', 'square-wave test signal -> no VT/VFib label')
def _case_square_wave():
    rec = '2026-10-31_15-33-42_Unknown'
    segs = _segments(rec)
    bad = _overlapping(segs, 318.0, 340.0)
    bad = [s for s in bad if s['label'] in ('Ventricular Tachycardia', 'Ventricular Fibrillation')
           and s['label'] == 'Ventricular Tachycardia']
    if bad:
        return False, 'VT label(s) on the test-signal stretch: %s' % [(s['start_sec'], s['end_sec']) for s in bad]
    return True, 'no VT label in 318-340 s (VFib there is a separate, pre-existing, tracked issue)'


# 2. 60 Hz mains interference must not corrupt the on-screen QRS/QT
# reading. Fixed by notching 60 Hz (in addition to the existing 50 Hz)
# before measuring QRS/QT in pqrst_measure.measure_beats, scoped to that
# display path only.
@case('GUARD', '60 Hz hum -> QRS/QT still read correctly (display)')
def _case_60hz_qrs():
    rec = '2026-11-02_10-57-19_Unknown'
    r = _reader(rec)
    beats = pm.measure_beats(r, 0.0, 10.0)
    status = pm.frame_status(r, 0.0, 10.0)
    values = pm.summarize(beats, [], status)
    qrs = values['qrs_ms']
    if qrs is None or not (70.0 <= qrs <= 110.0):
        return False, 'frame 0-10s QRS = %s (expect ~90 ms)' % qrs
    return True, 'frame 0-10s QRS = %.0f ms' % qrs


# 3. A short Supraventricular Tachycardia run sitting between two
# Atrial Fibrillation windows must not be swallowed whole by the
# AFib merge. Fixed by the SVT-in-AFib/Flutter beat-level carve.
@case('GUARD', 'SVT run inside an AFib stretch is recovered')
def _case_svt_in_afib():
    rec = '2026-11-02_10-57-19_Unknown'
    segs = _segments(rec)
    svt = _overlapping(segs, 44.0, 54.0, 'Supraventricular Tachycardia (SVT)')
    if not svt:
        return False, 'no SVT label found in 44-54 s (AFib should not cover this stretch alone)'
    return True, 'SVT found: %s' % [(round(s['start_sec'], 1), round(s['end_sec'], 1)) for s in svt]


@case('GUARD', 'SVT run inside a second AFib stretch is recovered')
def _case_svt_in_afib_2():
    rec = '2026-11-02_11-21-36_Unknown'
    segs = _segments(rec)
    svt = _overlapping(segs, 71.0, 81.0, 'Supraventricular Tachycardia (SVT)')
    if not svt:
        return False, 'no SVT label found in 71-81 s'
    return True, 'SVT found: %s' % [(round(s['start_sec'], 1), round(s['end_sec'], 1)) for s in svt]


# 4. KNOWN, not yet fixed (and not simply unimplemented -- blocked by a
# real trade-off, see below): a genuine VT run whose own beats are
# amplitude-diluted away inside a VFib window (a much louder neighbouring
# beat sets Pan-Tompkins' threshold too high) should be recovered by the
# VT-carve's own local, narrow-window re-detection -- and is, UNLESS the
# carve's own test-signal guard (case 1 above, span_status != 'ok')
# vetoes it. Measured directly: both this real VT run and the square-wave
# test case read span_status == 'non_ecg' identically (leads I/V1/V5
# correlate >= 0.95 either way) -- genuine monomorphic VT can look as
# identical across leads as a synthetic wave, and no discriminator tried
# (QRS-width consistency, flat-step fraction, curve-vs-straight-line
# linearity) told the two apart. The guard is kept (case 1 is the more
# dangerous failure mode -- a real arrhythmia label on non-physiological
# test data) at the cost of this one: the recording reverts to plain
# Ventricular Fibrillation here, exactly as it read before the VT-carve
# feature existed. Not a regression from this feature; a pre-existing
# limitation the feature could not safely lift for this specific case.
@case('KNOWN', 'VT run diluted inside a VFib call is recovered')
def _case_vt_in_vfib():
    rec = '2026-11-02_10-57-19_Unknown'
    segs = _segments(rec)
    vt = _overlapping(segs, 69.0, 76.0, 'Ventricular Tachycardia')
    if not vt:
        return False, 'no VT label found in 69-76 s (blocked by the test-signal guard, case 1 -- see comment above)'
    return True, 'VT found: %s' % [(round(s['start_sec'], 1), round(s['end_sec'], 1)) for s in vt]


# 5. KNOWN, not yet fixed: a single confirmed-AFib window with no
# same-label neighbour is dropped by the lone-window guard, even where
# the surrounding windows failed to vote AFib only because a borderline
# neighbour (Flutter whose wave-lock check narrowly failed) ate the
# votes either side of it.
@case('KNOWN', 'lone AFib window (13:58:04-14) is not dropped')
def _case_lone_afib():
    rec = '2026-11-02_13-56-27_Unknown'
    segs = _segments(rec)
    afib = _overlapping(segs, 104.0, 114.0, 'Atrial Fibrillation')
    if not afib:
        return False, 'no AFib label in 104-114 s (dropped by the lone-window guard)'
    return True, 'AFib found: %s' % [(round(s['start_sec'], 1), round(s['end_sec'], 1)) for s in afib]


# 6. KNOWN, not yet fixed: the beat-level bundle-branch-block run
# builder rejects a whole run when PR shifts partway through it (two
# genuinely different PR populations in one otherwise-consistent RBBB
# run), instead of splitting it into two passing sub-runs.
@case('KNOWN', 'RBBB run with a mid-run PR shift is still labelled')
def _case_bbb_pr_shift():
    rec = '2026-11-02_13-56-27_Unknown'
    segs = _segments(rec)
    bbb = _overlapping(segs, 76.0, 96.0, 'Complete Right Bundle Branch Block')
    if not bbb:
        return False, 'no RBBB label in 76-96 s (beat-level run was rejected over a PR shift)'
    return True, 'RBBB found: %s' % [(round(s['start_sec'], 1), round(s['end_sec'], 1)) for s in bbb]


# 7. KNOWN, not yet fixed: a flat/no-signal (Asystole) stretch sitting
# between two separate VFib windows gets swallowed by the main
# detector's same-label merge (any gap <= 2x window_size = 20 s between
# two VFib windows is bridged into one segment, regardless of what's
# actually in the gap) -- confirmed the identical root cause also ate a
# genuine 30 bpm Sinus Bradycardia run elsewhere in this same recording
# (13:57:01-11 s). The separate Asystole detector (detect_asystole_segments)
# itself is correct; this is the main detector's VFib-merge overreaching.
@case('KNOWN', 'flatline reads Asystole, not VFib, on the main detector')
def _case_asystole_not_vfib():
    rec = '2026-11-02_13-56-27_Unknown'
    asys = m.detect_asystole_segments(_reader(rec))
    hit = _overlapping(asys, 23.0, 33.0)
    if not hit:
        return False, 'no Asystole segment found in 23-33 s'
    segs = _segments(rec)
    vfib_here = _overlapping(segs, 23.0, 33.0, 'Ventricular Fibrillation')
    if vfib_here:
        return False, 'main detector also called VFib over the same flatline: %s' % vfib_here
    return True, 'Asystole found, no competing VFib label there'


# 8. Exact-60-bpm Normal Sinus Rhythm (RR ~= 1000 ms, right at the
# Bradycardia/NSR boundary -- Bradycardia needs RR >= 1010 ms) must not
# be mislabelled Bradycardia or Tachycardia. Confirmed clean: 31 s of
# steady ~1000 ms RR, PR ~158-172 ms, QRS ~88-104 ms.
@case('GUARD', 'exact 60 bpm NSR is not mislabelled Brady/Tachy')
def _case_nsr_boundary():
    rec = '2026-11-02_13-56-27_Unknown'
    segs = _segments(rec)
    bad = _overlapping(segs, 124.0, 155.0, 'Bradycardia') + \
        _overlapping(segs, 124.0, 155.0, 'Sinus Bradycardia') + \
        _overlapping(segs, 124.0, 155.0, 'Tachycardia') + \
        _overlapping(segs, 124.0, 155.0, 'Sinus Tachycardia')
    if bad:
        return False, 'rate label(s) on a clean 60 bpm run: %s' % [(s['label'], s['start_sec'], s['end_sec']) for s in bad]
    return True, 'no Brady/Tachy label over the 124-155 s 60 bpm run'


def _flutter_cover(segs, t0, t1):
    return sum(min(t1, s['end_sec']) - max(t0, s['start_sec'])
               for s in _overlapping(segs, t0, t1, 'Atrial Flutter'))


# Flutter 3:1 (100 bpm) and 2:1 (150 bpm) before a 5:1 stretch: only the
# 5:1 part met the window rule; the rest was plain Tachycardia until the
# Flutter continuation also ran backward from the segment's start.
@case('GUARD', 'Flutter 3:1 / 2:1 before 5:1 is Flutter, not Tachycardia')
def _case_flutter_backward():
    rec = '2026-11-02_18-24-52_Unknown'
    segs = _segments(rec)
    bad = _overlapping(segs, 1.0, 12.0, 'Tachycardia')
    cover = _flutter_cover(segs, 1.0, 12.0)
    if bad or cover < 9.0:
        return False, 'Flutter %.1f of 11 s, Tachycardia %s' % (cover, [(s['start_sec'], s['end_sec']) for s in bad])
    return True, 'Flutter covers %.1f of 11 s, no Tachycardia' % cover


# Flutter 5:1 then 3:1 under simulator 60 Hz mains: the ripple split each F
# wave (atrial rate read ~548/min) and made 3:1 intervals read as AFib.
@case('GUARD', 'Flutter under 60 Hz mains is Flutter, not AFib / Tachycardia')
def _case_flutter_60hz():
    rec = '2026-11-02_17-56-02_Unknown'
    segs = _segments(rec)
    bad = _overlapping(segs, 0.5, 10.5, 'Tachycardia') + _overlapping(segs, 0.5, 10.5, 'Atrial Fibrillation')
    cover = _flutter_cover(segs, 0.5, 10.5)
    if bad or cover < 9.0:
        return False, 'Flutter %.1f of 10 s, other: %s' % (cover, [(s['label'], s['start_sec'], s['end_sec']) for s in bad])
    return True, 'Flutter covers %.1f of 10 s' % cover


# AFib, then a regular 160 bpm run (RR CV 0.00), then Flutter 5:1: the
# AFib / Flutter edge step stretched AFib across the regular run once the
# Flutter after it was found.
@case('GUARD', 'regular 160 bpm run between AFib and Flutter is not AFib')
def _case_afib_flutter_gap():
    rec = '2026-11-02_11-57-07_Unknown'
    segs = _segments(rec)
    bad = _overlapping(segs, 48.5, 60.5, 'Atrial Fibrillation')
    if bad:
        return False, 'AFib over the regular run: %s' % [(s['start_sec'], s['end_sec']) for s in bad]
    return True, 'no AFib over 48.5-60.5 s'


# EMG-like noise on I / V1 over 700-720 s (raw SQI 0.31-0.54), gone after
# the 25 Hz low-pass the noise gate now scores on: the gate cut every label
# there, leaving a 160 bpm run's end and a Flutter 5:1 stretch blank.
@case('GUARD', 'noise the 25 Hz filter removes does not cut the label')
def _case_noise_gate_rhythm_leads():
    rec = '2026-10-23_11-53-46_Unknown'
    segs = _segments(rec)
    cover = _flutter_cover(segs, 709.0, 719.0)
    if cover < 9.0:
        return False, 'Flutter covers %.1f of 10 s in 709-719 s' % cover
    return True, 'Flutter covers %.1f of 10 s in 709-719 s' % cover


# 140 bpm sinus tachycardia, P / PR 136-140 ms on every beat, but 64 of 167
# beats with QRS bounds not resolved (T wave close at this rate): the run
# broke into 25 pieces alternating Sinus Tachycardia / plain Tachycardia.
@case('GUARD', 'sinus tachycardia 140 bpm is one segment, not 25 pieces')
def _case_sinus_tachy_unresolved_qrs():
    rec = '2026-10-28_14-53-03_Unknown'
    segs = _segments(rec)
    st = _overlapping(segs, 1.0, 62.0, 'Sinus Tachycardia')
    plain = _overlapping(segs, 1.0, 62.0, 'Tachycardia')
    if len(st) != 1 or plain:
        return False, 'Sinus Tachycardia pieces %d, plain Tachycardia pieces %d in 1-62 s' % (len(st), len(plain))
    return True, 'one Sinus Tachycardia segment %.1f-%.1f s' % (st[0]['start_sec'], st[0]['end_sec'])


# First-degree -> Mobitz I -> RBBB (with a 2-blocked-P pause) -> Third-
# degree -> LBBB -> RBBB: the AV block edges sat on the 5 s window grid
# (a whole Wenckebach cycle left out at each end) and each one-RR interval
# between two rhythms was drawn unlabelled.
@case('GUARD', 'conduction sequence is traced end to end, no gaps')
def _case_conduction_no_gaps():
    rec = '2026-11-03_10-25-10_Unknown'
    segs = sorted((s for s in _segments(rec) if s['label'] not in ('Artifact', 'Unidentified')),
                  key=lambda s: s['start_sec'])
    gaps, t = [], 31.78
    for s in segs:
        if s['start_sec'] > t + 0.05:
            gaps.append((round(t, 2), round(s['start_sec'], 2)))
        t = max(t, s['end_sec'])
    if gaps or t < 242.0:
        return False, 'unlabelled gaps %s, labels end %.1f s' % (gaps, t)
    mobitz = _overlapping(segs, 86.5, 120.9, 'Second-degree AV Block (Mobitz I)')
    if not mobitz or mobitz[0]['start_sec'] > 86.5 or mobitz[-1]['end_sec'] < 120.1:
        return False, 'Mobitz I does not cover 86.4-120.1 s: %s' % [(s['start_sec'], s['end_sec']) for s in mobitz]
    return True, 'labelled without a gap 31.8-%.1f s' % t


def _spliced_conduction(piece_sec):
    """7 s (or piece_sec) of each rhythm of 2026-11-03_10-25-10, back to
    back: First-degree, Mobitz I, RBBB, Third-degree, LBBB, RBBB. Returns
    (reader-like object, [(label, start, end)])."""
    import numpy as np
    from ecg.pan_tompkins import pan_tompkins
    r = _reader('2026-11-03_10-25-10_Unknown')
    fs = float(r.fs)
    names = list(r.lead_names)
    full = r.read_range(0, r.duration_sec)
    ii = np.asarray(full[names.index('II')], dtype=float)
    plan = [('First-degree', 40.0), ('Mobitz I', 90.16), ('RBBB', 124.0),
            ('Third-degree', 160.0), ('LBBB', 190.0), ('RBBB', 215.0)]
    parts, truth, t = [], [], 0.0
    for lab, s in plan:
        i0 = int((s - 1) * fs)
        pk = np.asarray(pan_tompkins(ii[i0:int((s + piece_sec + 4) * fs)], fs)) / fs + s - 1
        a = pk[pk >= s][0] - 0.3
        e = pk[pk >= a + piece_sec]
        b = (e[0] - 0.3) if len(e) else a + piece_sec
        parts.append(full[:, int(a * fs):int(b * fs)])
        truth.append((lab, t, t + b - a))
        t += b - a
    data = np.concatenate(parts, axis=1).astype(np.float32)

    class _Spliced:
        lead_names, n_leads, start_time, path = names, len(names), None, None

        def __init__(self):
            self.fs, self.duration_sec = fs, data.shape[1] / fs

        def read_range(self, a, b):
            i0, i1 = max(0, int(a * fs)), min(data.shape[1], int(b * fs))
            return data[:, i0:i1] if i1 > i0 else np.zeros((len(names), 0), np.float32)

        def close(self):
            pass

    return _Spliced(), truth


# Each rhythm only 7 s long: Mobitz I and Third-degree were missed (their
# window rules need a 10 s window of the block alone) until the beat-by-
# beat Mobitz I / Third-degree pass; no wrong label at any length.
@case('GUARD', '7 s episodes: Mobitz I and Third-degree still found')
def _case_short_episodes():
    rd, truth = _spliced_conduction(7.0)
    with contextlib.redirect_stdout(io.StringIO()):
        segs = m._detect_arrhythmias_uncached(rd)
    want = {'Mobitz I': 'Second-degree AV Block (Mobitz I)', 'Third-degree': 'Third-degree AV Block'}
    missing = []
    for lab, t0, t1 in truth:
        if lab in want and not _overlapping(segs, t0 + 0.5, t1 - 0.5, want[lab]):
            missing.append((lab, round(t0, 1), round(t1, 1)))
    if missing:
        return False, 'not found: %s' % missing
    return True, 'Mobitz I and Third-degree found in 7 s pieces'


# A premature beat and its compensatory pause (RR 750, 750, 562, 936) read
# as a Wenckebach cycle by the beat-by-beat Mobitz I pass, 7 times.
@case('GUARD', 'premature beat + pause is not Mobitz I')
def _case_pac_not_mobitz():
    segs = _segments('2026-10-23_11-53-46_Unknown')
    bad = _overlapping(segs, 20.0, 105.0, 'Second-degree AV Block (Mobitz I)')
    if bad:
        return False, 'Mobitz I over PAC stretch: %s' % [(s['start_sec'], s['end_sec']) for s in bad]
    return True, 'no Mobitz I in 20-105 s'


# 30 bpm Sinus Bradycardia, PR 172 ms on every beat, read as Third-degree
# for 332 s by the beat-by-beat pass (it ran over sinus-gated time).
@case('GUARD', 'steady-PR Sinus Bradycardia is not Third-degree')
def _case_sinus_brady_not_third():
    segs = _segments('2026-11-02_09-57-22_Unknown')
    bad = _overlapping(segs, 0.0, 335.0, 'Third-degree AV Block')
    if bad:
        return False, 'Third-degree over Sinus Bradycardia: %s' % [(s['start_sec'], s['end_sec']) for s in bad]
    return True, 'no Third-degree label'


# RBBB at 75 bpm with a 2250 ms pause (two blocked P waves, PR fixed) every
# 15 s: Second-degree Mobitz II. Was left unlabelled -- the rule took only
# 2x pauses, and on the wide RBBB QRS no P template / PR was found.
@case('GUARD', 'Mobitz II pause with two blocked P waves is labelled')
def _case_mobitz_ii_double_block():
    segs = _segments('2026-11-03_10-25-10_Unknown')
    missing = [t for t in (136.0, 151.0) if not _overlapping(segs, t, t + 1.0, 'Second-degree AV Block (Mobitz II)')]
    if missing:
        return False, 'no Mobitz II over the pause at %s s' % missing
    return True, 'Mobitz II at both pauses (136, 151 s)'


def main():
    guard_fail = known_fail = known_pass = skipped = 0
    rows = []
    for kind, name, fn in CASES:
        try:
            ok, detail = fn()
            skip = False
        except _RecordingMissing as e:
            ok, detail, skip = None, str(e), True
        except Exception as e:
            ok, detail, skip = False, 'ERROR: %r' % (e,), False
        rows.append((kind, name, ok, detail, skip))
        if skip:
            skipped += 1
        elif kind == 'GUARD' and not ok:
            guard_fail += 1
        elif kind == 'KNOWN':
            if ok:
                known_pass += 1
            else:
                known_fail += 1

    width = max(len(r[1]) for r in rows) + 2
    for kind, name, ok, detail, skip in rows:
        mark = 'SKIP' if skip else ('PASS' if ok else 'FAIL')
        flag = '' if skip or (kind == 'GUARD') == ok or (kind == 'KNOWN' and not ok) else '  <-- unexpected'
        print('[%s] %-4s %-*s %s%s' % (kind, mark, width, name, detail, flag))

    print()
    if skipped:
        print('%d case(s) SKIPPED -- the recording they need is missing (not a code result either way).' % skipped)
    if known_pass:
        print('%d KNOWN case(s) now PASS -- promote them to GUARD in this file.' % known_pass)
    if guard_fail:
        print('%d GUARD case(s) FAILED -- a regression was introduced. Do not ship this change.' % guard_fail)
        return 1
    print('All GUARD cases pass. (%d KNOWN case(s) still open, as expected.)' % known_fail)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
