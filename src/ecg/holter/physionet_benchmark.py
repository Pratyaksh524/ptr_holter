"""Score the auto arrhythmia detector against cardiologist-annotated
PhysioNet databases (MIT-BIH Arrhythmia, INCART 12-lead, AF / VF databases,
anything in WFDB format), the way ANSI/AAMI EC57 reports it.

Nothing is downloaded: point it at a folder that already holds the records
(<name>.hea + .dat + .atr):

    python -m ecg.holter.physionet_benchmark <folder> [--records 100,101]
           [--max-sec 600] [--jobs 4] [--json out.json]

Per record, and summed over all records, it reports:
  - QRS detection: Sensitivity / Positive Predictivity of Pan-Tompkins on
    lead II against the annotated beats (150 ms match window, EC57).
  - Per rhythm class: duration Se / +P (seconds) and episode Se / +P
    (an episode counts when it overlaps the other side at all).

Truth comes from the annotations: rhythm changes in aux_note ('(AFIB',
'(SVTA', '(VT', ...), Bundle Branch Block from runs of >= 5 'L' / 'R' beats,
and Sinus Tachycardia / Bradycardia from runs of >= 8 normal beats over
100 / under 60 bpm inside '(N'.

Lead handling: a record with the 12 standard leads is used as is. A
2-lead record (MIT-BIH: MLII + V1/V2/V4/V5) is placed in its own slots and
the other 10 are filled with distinct mixes of the two channels, so no lead
is flat or a copy of another. Rules that read lead morphology (Bundle
Branch Block, Flutter lead voting) are therefore only meaningful on
12-lead records -- the report marks those classes '(2-lead: indicative)'.
"""
import argparse
import contextlib
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction

import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

TARGET_FS = 500
COUNTS_PER_MV = 1300.0
ADC_MID = 2048.0
STD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_ALIASES = {'MLII': 'II', 'ML2': 'II', 'MLI': 'I', 'MLIII': 'III', 'D1': 'I', 'D2': 'II', 'D3': 'III'}
BEAT_SYMBOLS = set('NLRBAaJSVrFejnE/fQ?')

CLASSES = {
    'AF': {'Atrial Fibrillation'},
    'AFL': {'Atrial Flutter'},
    'SVT': {'Supraventricular Tachycardia (SVT)'},
    'VT': {'Ventricular Tachycardia'},
    'VF': {'Ventricular Fibrillation'},
    'STACH': {'Sinus Tachycardia', 'Tachycardia'},
    'SBR': {'Sinus Bradycardia', 'Bradycardia'},
    'AVB1': {'First-degree AV Block'},
    'AVB2': {'Second-degree AV Block (Mobitz I)', 'Second-degree AV Block (Mobitz II)',
             'Second-degree AV Block (2:1)'},
    'AVB3': {'Third-degree AV Block'},
    'LBBB': {'Complete Left Bundle Branch Block'},
    'RBBB': {'Complete Right Bundle Branch Block'},
}
MORPHOLOGY_CLASSES = {'LBBB', 'RBBB', 'AFL'}
RHYTHM_MAP = {
    '(AFIB': 'AF', '(AF': 'AF', '(AFL': 'AFL', '(SVTA': 'SVT', '(VT': 'VT',
    '(VFL': 'VF', '(VF': 'VF', '(SBR': 'SBR', '(BI': 'AVB1', '(BII': 'AVB2',
    '(BIII': 'AVB3', '(N': 'N',
}


class PhysioNetReader:
    """Looks like ECGHFileReader to the detector: fs, lead_names,
    duration_sec, start_time, path, read_range(), close()."""

    def __init__(self, data, fs, name):
        self._data = data
        self.fs = fs
        self.lead_names = list(STD_LEADS)
        self.n_leads = len(STD_LEADS)
        self.duration_sec = data.shape[1] / float(fs)
        self.start_time = None
        self.path = None
        self.name = name

    def read_range(self, start_sec, end_sec):
        i0 = max(0, int(start_sec * self.fs))
        i1 = min(self._data.shape[1], int(end_sec * self.fs))
        if i1 <= i0:
            return np.zeros((self.n_leads, 0), dtype=np.float32)
        return self._data[:, i0:i1]

    def get_duration_seconds(self):
        return self.duration_sec

    def close(self):
        pass


def _std_name(name):
    n = name.strip().upper().replace(' ', '')
    n = LEAD_ALIASES.get(n, n)
    for s in STD_LEADS:
        if s.upper() == n:
            return s
    return None


def load_record(folder, rec, max_sec=None):
    import wfdb
    from scipy.signal import resample_poly

    path = os.path.join(folder, rec)
    sampto = None
    hdr = wfdb.rdheader(path)
    fs_in = float(hdr.fs)
    if max_sec:
        sampto = min(hdr.sig_len, int(max_sec * fs_in))
    record = wfdb.rdrecord(path, sampto=sampto, physical=True)
    ann = wfdb.rdann(path, 'atr', sampto=sampto)

    sig = np.nan_to_num(np.asarray(record.p_signal, dtype=float)).T
    ratio = Fraction(TARGET_FS, 1) / Fraction(fs_in).limit_denominator(1000)
    ratio = ratio.limit_denominator(1000)
    sig = np.vstack([resample_poly(ch, ratio.numerator, ratio.denominator) for ch in sig])
    sig -= np.median(sig, axis=1, keepdims=True)
    # The device writes unsigned 12-bit samples around 2048; the noise
    # gate reads <= 1 / >= 4094 as clipped, so keep the signal inside.
    peak = float(np.percentile(np.abs(sig), 99.9)) or 1.0
    sig *= min(COUNTS_PER_MV, 1900.0 / peak)
    sig += ADC_MID

    slots = {}
    for i, name in enumerate(record.sig_name):
        s = _std_name(name)
        if s is not None and s not in slots:
            slots[s] = i
    twelve = all(s in slots for s in STD_LEADS)
    out = np.zeros((len(STD_LEADS), sig.shape[1]), dtype=np.float32)
    c0 = sig[0]
    c1 = sig[1] if sig.shape[0] > 1 else np.gradient(sig[0]) * 5.0 + ADC_MID
    for j, s in enumerate(STD_LEADS):
        if s in slots:
            out[j] = sig[slots[s]]
        else:
            theta = 0.35 + 0.45 * j
            out[j] = np.cos(theta) * (c0 - ADC_MID) + np.sin(theta) * (c1 - ADC_MID) + ADC_MID
    np.clip(out, 2, 4093, out=out)

    scale = TARGET_FS / fs_in
    samples = np.asarray(ann.sample, dtype=float) * scale / TARGET_FS
    return PhysioNetReader(out, TARGET_FS, rec), list(ann.symbol), list(ann.aux_note), samples, twelve


def _merge(spans, gap=0.0):
    spans = sorted(s for s in spans if s[1] > s[0])
    out = []
    for a, b in spans:
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(s) for s in out]


def truth_spans(symbols, aux, times, duration):
    """{class: [(start, end), ...]} from the annotations."""
    truth = {k: [] for k in CLASSES}
    truth['N'] = []
    current, since = None, 0.0
    for sym, note, t in zip(symbols, aux, times):
        note = (note or '').strip().rstrip('\x00')
        if sym == '+' and note.startswith('('):
            if current is not None:
                truth.setdefault(current, []).append((since, t))
            current, since = RHYTHM_MAP.get(note, 'OTHER'), t
    if current is not None:
        truth.setdefault(current, []).append((since, duration))

    beats = [(s, t) for s, t in zip(symbols, times) if s in BEAT_SYMBOLS]
    for sym, cls in (('L', 'LBBB'), ('R', 'RBBB')):
        run = []
        for s, t in beats + [('end', duration)]:
            if s == sym:
                run.append(t)
                continue
            if len(run) >= 5:
                truth[cls].append((run[0], run[-1]))
            run = []

    normal = _merge(truth['N'])
    n_beats = [t for s, t in beats if s == 'N']
    for cls, ok in (('STACH', lambda hr: hr > 100.0), ('SBR', lambda hr: hr < 60.0)):
        run = []
        for t0, t1 in zip(n_beats[:-1], n_beats[1:]):
            inside = any(a <= t0 and t1 <= b for a, b in normal)
            if inside and t1 > t0 and ok(60.0 / (t1 - t0)):
                if not run:
                    run = [t0]
                run.append(t1)
                continue
            if len(run) >= 8:
                truth[cls].append((run[0], run[-1]))
            run = []
        if len(run) >= 8:
            truth[cls].append((run[0], run[-1]))
    return {k: _merge(v) for k, v in truth.items()}


def _overlap(a_spans, b_spans):
    total = 0.0
    for a0, a1 in a_spans:
        for b0, b1 in b_spans:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                total += hi - lo
    return total


def _episodes_hit(a_spans, b_spans):
    return sum(1 for a0, a1 in a_spans if any(min(a1, b1) > max(a0, b0) for b0, b1 in b_spans))


def score_rhythms(truth, segments):
    out = {}
    for cls, labels in CLASSES.items():
        det = _merge([(float(s['start_sec']), float(s['end_sec'])) for s in segments if s['label'] in labels])
        tru = truth.get(cls, [])
        ov = _overlap(tru, det)
        out[cls] = {
            'truth_sec': sum(b - a for a, b in tru),
            'det_sec': sum(b - a for a, b in det),
            'overlap_sec': ov,
            'truth_ep': len(tru),
            'det_ep': len(det),
            'truth_ep_hit': _episodes_hit(tru, det),
            'det_ep_hit': _episodes_hit(det, tru),
        }
    return out


def score_qrs(reader, symbols, times):
    from ecg.pan_tompkins import pan_tompkins

    fs = reader.fs
    lead = reader._data[STD_LEADS.index('II')]
    found = []
    chunk = 60 * fs
    for i0 in range(0, len(lead), chunk):
        part = np.asarray(lead[i0:i0 + chunk], dtype=float)
        if len(part) < 2 * fs:
            continue
        found.extend((i0 + int(p)) / float(fs) for p in pan_tompkins(part, fs))
    found = np.array(sorted(found))
    ref = np.array([t for s, t in zip(symbols, times) if s in BEAT_SYMBOLS])
    tp = 0
    if len(found) and len(ref):
        j = np.searchsorted(found, ref)
        used = set()
        for r, k in zip(ref, j):
            for c in (k - 1, k):
                if 0 <= c < len(found) and c not in used and abs(found[c] - r) <= 0.150:
                    used.add(c)
                    tp += 1
                    break
    return {'ref': int(len(ref)), 'det': int(len(found)), 'tp': int(tp)}


def run_record(args):
    folder, rec, max_sec = args
    from ecg.holter import auto_segment_arrthymia_detection as m

    try:
        reader, symbols, aux, times, twelve = load_record(folder, rec, max_sec)
        with contextlib.redirect_stdout(io.StringIO()):
            segs = m._detect_arrhythmias_uncached(reader)
        truth = truth_spans(symbols, aux, times, reader.duration_sec)
        return {
            'record': rec,
            'twelve_lead': twelve,
            'duration_sec': reader.duration_sec,
            'qrs': score_qrs(reader, symbols, times),
            'rhythm': score_rhythms(truth, segs),
            'unidentified_sec': sum(float(s['end_sec']) - float(s['start_sec'])
                                    for s in segs if s['label'] == 'Unidentified'),
        }
    except Exception as e:
        return {'record': rec, 'error': repr(e)}


def _pct(num, den):
    return '  -  ' if den <= 0 else '%5.1f' % (100.0 * num / den)


def report(results):
    ok = [r for r in results if 'error' not in r]
    for r in results:
        if 'error' in r:
            print('%-10s ERROR %s' % (r['record'], r['error']))
    print('\nQRS detection (lead II, 150 ms window)')
    print('%-10s %7s %7s %7s %6s %6s' % ('record', 'ref', 'det', 'TP', 'Se%', '+P%'))
    tot = {'ref': 0, 'det': 0, 'tp': 0}
    for r in ok:
        q = r['qrs']
        for k in tot:
            tot[k] += q[k]
        print('%-10s %7d %7d %7d %6s %6s' % (r['record'], q['ref'], q['det'], q['tp'],
                                            _pct(q['tp'], q['ref']), _pct(q['tp'], q['det'])))
    print('%-10s %7d %7d %7d %6s %6s' % ('GROSS', tot['ref'], tot['det'], tot['tp'],
                                        _pct(tot['tp'], tot['ref']), _pct(tot['tp'], tot['det'])))

    all_twelve = all(r['twelve_lead'] for r in ok) if ok else True
    print('\nRhythm classes, gross over all records')
    print('%-6s %9s %9s %7s %7s %6s %6s  %s' % ('class', 'truth s', 'det s', 'durSe%', 'dur+P%',
                                              'epSe%', 'ep+P%', ''))
    for cls in CLASSES:
        t = {k: sum(r['rhythm'][cls][k] for r in ok) for k in
             ('truth_sec', 'det_sec', 'overlap_sec', 'truth_ep', 'det_ep', 'truth_ep_hit', 'det_ep_hit')}
        if t['truth_sec'] <= 0 and t['det_sec'] <= 0:
            continue
        note = '(2-lead: indicative)' if cls in MORPHOLOGY_CLASSES and not all_twelve else ''
        print('%-6s %9.0f %9.0f %7s %7s %6s %6s  %s' % (
            cls, t['truth_sec'], t['det_sec'],
            _pct(t['overlap_sec'], t['truth_sec']), _pct(t['overlap_sec'], t['det_sec']),
            _pct(t['truth_ep_hit'], t['truth_ep']), _pct(t['det_ep_hit'], t['det_ep']), note))
    unid = sum(r['unidentified_sec'] for r in ok)
    dur = sum(r['duration_sec'] for r in ok)
    print('\nUnidentified: %.0f s of %.0f s (%s%%)' % (unid, dur, _pct(unid, dur).strip()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('folder')
    ap.add_argument('--records', help='comma-separated record names (default: every .hea with an .atr)')
    ap.add_argument('--max-sec', type=float, default=None, help='score only the first N seconds of each record')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--json', help='also write the raw per-record numbers here')
    a = ap.parse_args(argv)

    if a.records:
        recs = [r.strip() for r in a.records.split(',') if r.strip()]
    else:
        recs = sorted(f[:-4] for f in os.listdir(a.folder)
                      if f.endswith('.hea') and os.path.exists(os.path.join(a.folder, f[:-4] + '.atr')))
    if not recs:
        print('No records with .hea + .atr found in %s' % a.folder)
        return 1
    jobs = [(a.folder, r, a.max_sec) for r in recs]
    if a.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(run_record, jobs))
    else:
        results = [run_record(j) for j in jobs]
    report(results)
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
