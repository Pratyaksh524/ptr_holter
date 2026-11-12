"""Amplitude-change check for the auto arrhythmia detector and the PR/QRS/QT
measurements.

Standalone (no pytest dependency, matches this project's existing script
style): run directly --

    python -m ecg.holter.amplitude_regression <recording> [--factors 0.2,0.5,2]
           [--every 20] [--tol-ms 10]

<recording> is a folder name under recordings/ (or a full path to the folder
or its recording.ecgh).

What it checks: a change of amplitude alone -- the same on every lead, as a
gain change on the device or simulator does -- must not change which beats
are found, the arrhythmia marking, or PR/QRS/QT. The recording is replayed
with its amplitude switched between 1x and each factor every --every
seconds (all leads at once, around each lead's own baseline, clipped to the
12-bit ADC range 0-4095 like the device), and compared with the unchanged
recording:

  - beats    -- every beat the strip shows (pqrst_measure.measure_beats, 10 s
                frames) found again within 60 ms; missed and extra beats are
                listed with their clock time.
  - marking  -- the detector's label second by second; a differing stretch of
                1 s or more is listed. Seconds within 2 s of one of the
                unchanged recording's own label boundaries are not counted.
  - PR / QRS / QT -- each 10 s frame's median (as the strip summary shows)
                within --tol-ms of the unchanged recording.

A stretch where the scaled signal hits the ADC limit is marked "(clipped)":
there the waveform itself is cut off, so a difference there is the signal,
not the algorithm.

Exit code 0 = every factor passed, 1 = something differed.
"""
import argparse
import contextlib
import io
import os
import sys
from datetime import datetime

import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ecg.holter.file_format import ECGHFileReader  # noqa: E402
from ecg.holter import auto_segment_arrthymia_detection as m  # noqa: E402
from ecg.holter.pqrst_measure import measure_beats  # noqa: E402

RECORDINGS_DIR = os.path.join(SRC, 'recordings')
ADC_MAX = 4095.0
FRAME_SEC = 10.0
BEAT_TOL_SEC = 0.06
BOUNDARY_SKIP_SEC = 2.0
MIN_LABEL_DIFF_SEC = 1.0


class ScaledReader:
    """The recording with its amplitude switched between 1x and `factor`
    every `every_sec` seconds (from every_sec/2 on), all leads together.
    `path` is left empty so the detector does not hand the file to its
    worker processes, which would read it unscaled."""

    def __init__(self, inner, factor, every_sec):
        self._inner = inner
        self._factor = float(factor)
        self._every = float(every_sec)
        self.fs = inner.fs
        self.lead_names = inner.lead_names
        self.duration_sec = inner.duration_sec
        self.start_time = getattr(inner, 'start_time', None)
        self.path = ''
        # Each lead's own baseline, from the whole recording, so every
        # read_range scales around the same point.
        probe = np.asarray(inner.read_range(0.0, min(inner.duration_sec, 120.0)), dtype=float)
        self._base = np.median(probe, axis=1)[:, None]

    def gain_at(self, t):
        k = int(np.floor((t - self._every / 2.0) / self._every))
        return self._factor if t >= self._every / 2.0 and k % 2 == 0 else 1.0

    def step_times(self):
        t, out = self._every / 2.0, []
        while t < self.duration_sec:
            out.append(t)
            t += self._every
        return out

    def read_range(self, t0, t1):
        d = np.asarray(self._inner.read_range(t0, t1), dtype=float).copy()
        if d.ndim != 2 or d.shape[1] == 0:
            return d
        t = t0 + np.arange(d.shape[1]) / float(self.fs)
        k = np.floor((t - self._every / 2.0) / self._every)
        gain = np.where((t >= self._every / 2.0) & (k % 2 == 0), self._factor, 1.0)
        return np.clip(self._base + (d - self._base) * gain[None, :], 0.0, ADC_MAX)

    def clipped_seconds(self):
        out = set()
        t = 0.0
        while t < self.duration_sec:
            t1 = min(self.duration_sec, t + 60.0)
            d = self.read_range(t, t1)
            raw = np.asarray(self._inner.read_range(t, t1), dtype=float)
            # Only samples the scaling pushed onto the limit (a lead already
            # sitting at 0 / 4095 in the recording is not counted).
            hit = np.any(((d <= 0.0) | (d >= ADC_MAX)) & (raw > 0.0) & (raw < ADC_MAX), axis=0)
            for i in np.flatnonzero(hit):
                out.add(int(t + i / float(self.fs)))
            t += 60.0
        return out

    def close(self):
        pass


def _quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


def _clock(reader, t):
    st = getattr(reader, 'start_time', None)
    if not st:
        return '%.1fs' % t
    return datetime.fromtimestamp(st + t).strftime('%H:%M:%S')


def _frames(reader):
    """[(frame start, beats)] over the recording, 10 s frames as the strip."""
    out, t = [], 0.0
    while t < reader.duration_sec:
        out.append((t, _quiet(measure_beats, reader, t, min(reader.duration_sec, t + FRAME_SEC))))
        t += FRAME_SEC
    return out


def _median(beats, key):
    v = [b[key] for b in beats if b.get(key) is not None]
    return float(np.median(v)) if v else None


def _label_per_second(segs, duration):
    n = int(np.ceil(duration))
    out = [''] * n
    for s in segs:
        for i in range(max(0, int(s['start_sec'])), min(n, int(np.ceil(s['end_sec'])))):
            mid = i + 0.5
            if s['start_sec'] <= mid < s['end_sec']:
                out[i] = s['label']
    return out


def _runs(seconds):
    runs = []
    for s in sorted(seconds):
        if runs and s == runs[-1][1]:
            runs[-1][1] = s + 1
        else:
            runs.append([s, s + 1])
    return runs


def check(reader, factor, every, tol_ms, base_segs, base_frames):
    sr = ScaledReader(reader, factor, every)
    clipped = sr.clipped_seconds()
    tag = lambda a, b: ' (clipped)' if any(int(a) <= c < int(np.ceil(b)) for c in clipped) else ''
    problems = 0
    print('\n--- factor %gx, switched every %g s (steps at %s ...) ---'
          % (factor, every, ', '.join(_clock(reader, t) for t in sr.step_times()[:4])))
    if clipped:
        print('  ADC limit hit in %d s of the scaled recording' % len(clipped))

    frames = _frames(sr)
    base_t = np.array([b['t'] for _, bs in base_frames for b in bs])
    new_t = np.array([b['t'] for _, bs in frames for b in bs])
    missed = [t for t in base_t if not new_t.size or np.min(np.abs(new_t - t)) > BEAT_TOL_SEC]
    extra = [t for t in new_t if not base_t.size or np.min(np.abs(base_t - t)) > BEAT_TOL_SEC]
    print('  beats: %d unchanged, %d after scaling -- %d missed, %d extra'
          % (len(base_t), len(new_t), len(missed), len(extra)))
    for name, ts in (('missed', missed), ('extra', extra)):
        for t in ts:
            problems += 1
            print('    %-6s beat %s (%.2f s, gain %gx)%s' % (name, _clock(reader, t), t, sr.gain_at(t), tag(t, t + 0.01)))

    bad = []
    for (t0, b0), (_, b1) in zip(base_frames, frames):
        for key, name in (('pr_ms', 'PR'), ('qrs_ms', 'QRS'), ('qt_ms', 'QT')):
            v0, v1 = _median(b0, key), _median(b1, key)
            if (v0 is None) != (v1 is None) or (v0 is not None and abs(v0 - v1) > tol_ms):
                bad.append('    %s-%s %-3s %s -> %s ms%s' % (
                    _clock(reader, t0), _clock(reader, min(reader.duration_sec, t0 + FRAME_SEC)), name,
                    '--' if v0 is None else '%.0f' % v0, '--' if v1 is None else '%.0f' % v1,
                    tag(t0, t0 + FRAME_SEC)))
    print('  PR/QRS/QT: %d of %d frame values differ by more than %g ms'
          % (len(bad), 3 * len(base_frames), tol_ms))
    for line in bad:
        problems += 1
        print(line)

    segs = _quiet(m._detect_arrhythmias_uncached, sr)
    a = _label_per_second(base_segs, reader.duration_sec)
    b = _label_per_second(segs, reader.duration_sec)
    edges = set()
    for s in base_segs:
        for e in (s['start_sec'], s['end_sec']):
            edges.update(range(int(e - BOUNDARY_SKIP_SEC), int(np.ceil(e + BOUNDARY_SKIP_SEC))))
    diff = [i for i in range(len(a)) if a[i] != b[i] and i not in edges]
    runs = [r for r in _runs(diff) if r[1] - r[0] >= MIN_LABEL_DIFF_SEC]
    print('  marking: %d stretch(es) differ' % len(runs))
    for s0, s1 in runs:
        problems += 1
        print('    %s-%s  %s -> %s%s' % (_clock(reader, s0), _clock(reader, s1),
                                       a[s0] or '(none)', b[s0] or '(none)', tag(s0, s1)))
    print('  %s' % ('PASS' if problems == 0 else 'FAIL'))
    return problems == 0


def _resolve(arg):
    for p in (arg, os.path.join(arg, 'recording.ecgh'),
              os.path.join(RECORDINGS_DIR, arg, 'recording.ecgh')):
        if os.path.isfile(p):
            return p
    raise SystemExit('recording not found: %s' % arg)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('recording')
    ap.add_argument('--factors', default='0.2,0.5,2')
    ap.add_argument('--every', type=float, default=20.0, help='seconds between amplitude switches')
    ap.add_argument('--tol-ms', type=float, default=10.0)
    args = ap.parse_args(argv)
    path = _resolve(args.recording)
    reader = ECGHFileReader(path)
    print('%s  (%.0f s, %s-%s)' % (os.path.basename(os.path.dirname(path)), reader.duration_sec,
                                  _clock(reader, 0), _clock(reader, reader.duration_sec)))
    # The unchanged run goes through the same reader class at 1x, so both
    # runs take the same (single-process) path through the detector.
    unscaled = ScaledReader(reader, 1.0, args.every)
    base_segs = _quiet(m._detect_arrhythmias_uncached, unscaled)
    base_frames = _frames(unscaled)
    ok = True
    for f in (float(x) for x in args.factors.split(',') if x.strip()):
        ok = check(reader, f, args.every, args.tol_ms, base_segs, base_frames) and ok
    print('\n%s' % ('All factors PASS.' if ok else 'Some factor FAILED -- see the lines above.'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
