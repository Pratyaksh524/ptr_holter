#!/usr/bin/env python
"""
Synthetic 12-lead Holter recording (default 72 h) with known arrhythmia and
ischemia episodes, written as a normal session folder the Holter app opens:

    <out>/<YYYY-MM-DD_HH-MM-SS>_Dummy_<N>h/
        recording.ecgh            same binary format as ECGHFileWriter
        recording_index.json      same per-second index ECGHFileWriter writes
        session.json, patient.json, events.db
        ground_truth_events.csv   every planted episode: label, start/end, params
        ground_truth_events.json  same, with per-lead expected ST / T values
        ground_truth_beats.csv    every QRS: time, beat type, rhythm, PR
        ground_truth_pwaves.csv   every P wave: time, type, conducted or not

Signal scale matches the device: 2048 offset on I, II, V1-V6; III, aVR, aVL,
aVF derived from those offset values exactly like the hardware does.
1 mV = 933 ADC, the Full Disclosure screen scale ischemia_detection.py uses,
so a planted 0.1 mV ST shift reads as exactly 1.0 screen mm.

Usage (from ptr_holter/):
    python tools/generate_dummy_holter.py                      # 72 h -> src/recordings
    python tools/generate_dummy_holter.py --hours 8 --out D:/tmp
    python tools/generate_dummy_holter.py --hours 6 --preview  # + PNG strip per episode
"""

import argparse
import csv
import importlib.util
import json
import math
import os
import struct
import sys
import time
from datetime import datetime, timezone

import numpy as np

FS = 500
ADC_PER_MV = 933.0
ADC_OFFSET = 2048
HEADER_SIZE = 256
FRAME_SIZE = 26
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# independent leads that are synthesised; the other four are derived from I and II
IND = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
LI = {name: i for i, name in enumerate(IND)}
CHUNK_SEC = 60.0

# --------------------------------------------------------------------------
# Beat morphologies.  Times are seconds from the R peak; amplitudes are mV in
# the order I, II, V1, V2, V3, V4, V5, V6.
# --------------------------------------------------------------------------
MORPHS = {
    # Normal narrow QRS (~90 ms)
    'N': dict(
        waves=[(-0.028, 0.007, [-0.05, -0.08, 0.00, 0.00, 0.00, -0.03, -0.06, -0.06]),
               (0.000, 0.0095, [0.70, 1.10, 0.20, 0.45, 0.80, 1.30, 1.20, 1.00]),
               (0.026, 0.008, [-0.10, -0.20, -0.90, -1.30, -0.80, -0.40, -0.20, -0.10])],
        T=([0.20, 0.30, 0.08, 0.45, 0.45, 0.40, 0.30, 0.25], 0.0, 1.0),
        onset=0.042, j=0.046),
    # Complete RBBB (~145 ms): rsR' in V1/V2, wide slurred S in I, V5, V6
    'RBBB': dict(
        waves=[(-0.028, 0.007, [-0.05, -0.08, 0.00, 0.00, 0.00, -0.03, -0.06, -0.06]),
               (-0.004, 0.009, [0.60, 1.00, 0.25, 0.40, 0.70, 1.20, 1.10, 0.90]),
               (0.030, 0.009, [-0.15, -0.20, -0.45, -0.70, -0.50, -0.20, -0.15, -0.10]),
               (0.068, 0.014, [0.00, 0.00, 0.75, 0.50, 0.10, 0.00, 0.00, 0.00]),
               (0.065, 0.018, [-0.30, -0.15, 0.00, 0.00, -0.10, -0.25, -0.35, -0.35])],
        T=([0.18, 0.25, -0.20, -0.10, 0.30, 0.35, 0.28, 0.22], 0.03, 1.0),
        onset=0.042, j=0.105),
    # Incomplete RBBB (~112 ms)
    'iRBBB': dict(
        waves=[(-0.028, 0.007, [-0.05, -0.08, 0.00, 0.00, 0.00, -0.03, -0.06, -0.06]),
               (-0.004, 0.009, [0.65, 1.05, 0.22, 0.42, 0.75, 1.25, 1.15, 0.95]),
               (0.024, 0.008, [-0.12, -0.20, -0.50, -0.80, -0.55, -0.25, -0.15, -0.10]),
               (0.046, 0.010, [0.00, 0.00, 0.55, 0.35, 0.05, 0.00, 0.00, 0.00]),
               (0.046, 0.012, [-0.25, -0.12, 0.00, 0.00, -0.08, -0.20, -0.28, -0.28])],
        T=([0.20, 0.28, -0.10, 0.20, 0.40, 0.38, 0.30, 0.24], 0.015, 1.0),
        onset=0.042, j=0.072),
    # Complete LBBB (~150 ms): deep QS V1-V3, broad notched R in I, V5, V6, no q / no S
    'LBBB': dict(
        waves=[(-0.015, 0.020, [0.55, 0.45, 0.00, 0.00, 0.00, 0.30, 0.65, 0.65]),
               (0.035, 0.020, [0.65, 0.55, 0.00, 0.00, 0.00, 0.40, 0.80, 0.75]),
               (0.010, 0.030, [0.00, 0.00, -1.30, -1.60, -1.00, 0.00, 0.00, 0.00])],
        T=([-0.20, -0.10, 0.35, 0.40, 0.30, -0.05, -0.22, -0.22], 0.04, 1.15),
        onset=0.065, j=0.085),
    # Incomplete LBBB (~110 ms)
    'iLBBB': dict(
        waves=[(-0.015, 0.015, [0.55, 0.50, 0.00, 0.00, 0.00, 0.35, 0.70, 0.70]),
               (0.020, 0.015, [0.60, 0.55, 0.00, 0.00, 0.00, 0.40, 0.75, 0.70]),
               (0.003, 0.022, [0.00, 0.00, -1.10, -1.40, -0.90, 0.00, 0.00, 0.00])],
        T=([-0.10, 0.05, 0.25, 0.35, 0.30, 0.05, -0.10, -0.10], 0.02, 1.1),
        onset=0.0525, j=0.0575),
    # PVC / ventricular escape: wide (~160 ms), LBBB-like, inferior axis
    'V': dict(
        waves=[(-0.010, 0.028, [0.80, 1.40, -0.90, -1.20, -0.40, 0.90, 1.30, 1.10])],
        T=([-0.30, -0.45, 0.40, 0.50, 0.20, -0.30, -0.40, -0.35], 0.06, 1.3),
        onset=0.080, j=0.075),
    # Monomorphic VT: wide (~165 ms), RBBB-like, superior axis
    'VT': dict(
        waves=[(-0.030, 0.020, [-0.30, -0.50, 0.40, 0.30, 0.20, -0.20, -0.30, -0.30]),
               (0.020, 0.025, [-0.60, -1.00, 1.30, 1.10, 0.70, -0.50, -0.90, -1.00])],
        T=([0.35, 0.55, -0.60, -0.50, -0.35, 0.25, 0.40, 0.45], 0.06, 1.3),
        onset=0.080, j=0.083),
}
MORPH_LIST = list(MORPHS)
MORPH_ID = {m: i for i, m in enumerate(MORPH_LIST)}
ISCHEMIA_MORPHS = {'N'}       # ST / T changes are applied to narrow conducted beats

P_SINUS = (0.022, [0.08, 0.15, 0.06, 0.07, 0.07, 0.07, 0.06, 0.05])
P_ECTOPIC = (0.018, [0.04, 0.09, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04])

# Normal-rhythm limits: background HR is clamped so it never meets the
# Sinus Tachycardia (RR <= 600 ms) or Sinus Bradycardia (RR >= 1050 ms) rules.
BG_HR_MIN, BG_HR_MAX = 64.0, 94.0
P_PVC, P_PAC = 0.0015, 0.002


def p_from_r(r, pr, morph):
    """P-wave centre for a QRS at r with PR interval pr (P onset -> QRS onset)."""
    return r - pr + 0.05 - MORPHS[morph]['onset']


def r_from_p(p, pr, morph):
    return p + pr - 0.05 + MORPHS[morph]['onset']


# --------------------------------------------------------------------------
# Timeline: every beat, P wave, continuous signal and ischemia overlay
# --------------------------------------------------------------------------
class Timeline:
    def __init__(self, total_sec, start_epoch, rng):
        self.total = total_sec
        self.start_epoch = start_epoch
        self.rng = rng
        self.R, self.M, self.K, self.PR, self.RH = [], [], [], [], []
        self.P, self.PT, self.PC = [], [], []
        self.cont = []      # dict(t0, t1, kind, ...)
        self.isch = []      # dict(t0, t1, st={lead: (st0, slope)}, t={lead: amp})
        self.gt = []        # ground-truth episodes
        self.last_r = 0.3
        self.last_sinus = 0.3
        self._init_hr_noise()

    # ---- background heart rate ----
    def _init_hr_noise(self):
        n = int(self.total) + 10
        a = math.exp(-1.0 / 120.0)
        w = self.rng.standard_normal(n) * 5.0 * math.sqrt(1 - a * a)
        try:
            from scipy.signal import lfilter
            self._dev = lfilter([1.0], [1.0, -a], w)
        except Exception:
            dev = np.zeros(n)
            for i in range(1, n):
                dev[i] = dev[i - 1] * a + w[i]
            self._dev = dev
        self._start_hour = datetime.fromtimestamp(self.start_epoch).hour + \
            datetime.fromtimestamp(self.start_epoch).minute / 60.0

    def clock_hour(self, t):
        return (self._start_hour + t / 3600.0) % 24.0

    def bg_hr(self, t):
        h = self.clock_hour(t)
        day = 0.25 * (1 + math.tanh((h - 6.5) / 0.7)) * (1 + math.tanh((22.5 - h) / 0.7))
        hr = 66.0 + 14.0 * day + self._dev[min(int(t), len(self._dev) - 1)]
        return min(BG_HR_MAX, max(BG_HR_MIN, hr))

    def bg_rr(self, t):
        return 60.0 / self.bg_hr(t) * (1 + 0.02 * math.sin(2 * math.pi * 0.25 * t))

    # ---- primitives ----
    def beat(self, r, morph, kind, pr=None, rhythm='Normal'):
        self.R.append(r); self.M.append(MORPH_ID[morph]); self.K.append(kind)
        self.PR.append(np.nan if pr is None else pr * 1000.0); self.RH.append(rhythm)
        self.last_r = r

    def pwave(self, p, ptype, conducted):
        self.P.append(p); self.PT.append(ptype); self.PC.append(bool(conducted))

    def add_gt(self, label, start, end, expected='yes', group='', params=None, note=''):
        self.gt.append(dict(label=label, start_s=round(start, 3), end_s=round(end, 3),
                            expected=expected, group=group, params=params or {}, note=note))

    # ---- normal sinus rhythm with occasional PVC / PAC ----
    def nsr(self, t_end, morph='N', ectopy=True, rsa=0.02, rhythm='Normal', hr_scale=1.0):
        seg0 = self.last_r
        t = self.last_r
        prev_ect = False
        while True:
            rr = self.bg_rr(t) * hr_scale
            if rsa != 0.02:
                rr *= (1 + (rsa - 0.02) * math.sin(2 * math.pi * 0.2 * t))
            tn = t + rr
            if tn >= t_end:
                break
            hr = 60.0 / rr
            pr = 0.165 - 0.0006 * (hr - 70) + self.rng.normal(0, 0.002)
            can = ectopy and not prev_ect and tn - seg0 > 30 and t_end - tn > 30
            u = self.rng.random()
            if can and u < P_PVC:
                self.beat(t + 0.6 * rr, 'V', 'V', rhythm=rhythm)
                self.pwave(p_from_r(tn, pr, morph), 0, False)
                t = tn; prev_ect = True
                continue
            if can and u < P_PVC + P_PAC:
                ta = t + 0.68 * rr
                self.pwave(p_from_r(ta, 0.15, morph), 1, True)
                self.beat(ta, morph, 'A', 0.15, rhythm)
                t = ta; prev_ect = True
                continue
            prev_ect = False
            self.pwave(p_from_r(tn, pr, morph), 0, True)
            self.beat(tn, morph, 'N', pr, rhythm)
            t = tn
        self.last_sinus = t

    # ---- sinus with an HR profile (rate episodes, BBB + rate) ----
    def sinus_profile(self, dur, hr_fn, morph='N', pr_fixed=None, rhythm='Sinus'):
        t0 = self.last_r
        t = t0
        idx0 = len(self.R)
        while True:
            hr = hr_fn(t - t0)
            rr = 60.0 / hr * (1 + 0.012 * math.sin(2 * math.pi * 0.25 * t))
            tn = t + rr
            if tn >= t0 + dur:
                break
            pr = pr_fixed if pr_fixed is not None else \
                float(np.clip(0.165 - 0.0008 * (hr - 70), 0.12, 0.18))
            self.pwave(p_from_r(tn, pr, morph), 0, True)
            self.beat(tn, morph, 'N', pr, rhythm)
            t = tn
        return idx0, len(self.R)

    def ramp_fn(self, dur, peak, ramp=90.0):
        h0 = self.bg_hr(self.last_r)
        h1 = self.bg_hr(self.last_r + dur)

        def fn(tau):
            if tau < ramp:
                return h0 + (peak - h0) * 0.5 * (1 - math.cos(math.pi * tau / ramp))
            if tau > dur - ramp:
                x = (tau - (dur - ramp)) / ramp
                return peak + (h1 - peak) * 0.5 * (1 - math.cos(math.pi * x))
            return peak
        return fn

    def rate_gt(self, i0, i1, label, cond, group, params, expected='yes', note=''):
        """Ground truth for a rate episode from the beats actually generated."""
        R = np.asarray(self.R[i0:i1])
        if R.size < 2:
            return
        rr = np.diff(R)
        ok = cond(rr)
        runs, k = [], 0
        while k < ok.size:
            if ok[k]:
                j = k
                while j + 1 < ok.size and ok[j + 1]:
                    j += 1
                if runs and k - runs[-1][1] <= 3:      # 1-2 borderline RRs don't split an episode
                    runs[-1] = (runs[-1][0], j)
                else:
                    runs.append((k, j))
                k = j + 1
            else:
                k += 1
        for a, b in runs:
            if b - a + 1 >= 2:
                self.add_gt(label, R[a], R[b + 1], expected, group,
                            dict(params, beats=int(b - a + 2)), note)

    # ---- ventricular ----
    def vt(self, rate, dur, dissociated_p=False, group='', expected='yes', note='', label='Ventricular Tachycardia'):
        rr = 60.0 / rate
        r = self.last_r + max(0.42, rr)
        start = r
        sinus_rr = self.bg_rr(self.last_r)
        next_p = self.last_r + sinus_rr - 0.15
        while r < start + dur:
            self.beat(r, 'VT', 'VT', rhythm='Ventricular Tachycardia')
            r += rr * (1 + self.rng.normal(0, 0.008))
        end = self.last_r
        if dissociated_p:
            while next_p < end:
                self.pwave(next_p, 0, False)
                next_p += sinus_rr
        self.add_gt(label, start, end, expected, group,
                    dict(rate_bpm=rate, qrs_ms=165, dissociated_p=dissociated_p,
                         beats=int(round((end - start) / rr)) + 1), note)
        self.last_r = end + 0.9
        self.last_sinus = self.last_r
        return start, end

    def vfib(self, dur, amp, f0, group='', expected='yes', note='', start=None):
        t0 = (self.last_r + 0.35) if start is None else start
        t1 = t0 + dur
        seed = int(self.rng.integers(1 << 30))
        self.cont.append(dict(t0=t0, t1=t1, kind='VF', amp=amp, f0=f0, seed=seed))
        kind = 'coarse' if amp >= 0.4 else 'fine'
        self.add_gt('Ventricular Fibrillation', t0, t1, expected, group,
                    dict(type=kind, amp_mv=amp, freq_hz=f0), note)
        self.last_r = t1 + 1.0
        self.last_sinus = self.last_r
        self.beat(self.last_r, 'N', 'N', 0.17, 'Normal')
        self.pwave(p_from_r(self.last_r, 0.17, 'N'), 0, True)
        return t0, t1

    # ---- atrial ----
    def afib(self, mean_hr, dur, cv=0.22, group='', expected='yes', note=''):
        f0 = self.last_r + 0.15
        mean_rr = 60.0 / mean_hr
        sig = math.sqrt(math.log(1 + cv * cv))
        mu = math.log(mean_rr) - sig * sig / 2
        r = self.last_r
        first = None
        while True:
            rr = float(np.clip(self.rng.lognormal(mu, sig), 0.30, 1.6))
            if r + rr > f0 + dur:
                break
            r += rr
            self.beat(r, 'N', 'N', rhythm='Atrial Fibrillation')
            first = r if first is None else first
        f1 = self.last_r + 0.25
        self.cont.append(dict(t0=f0, t1=f1, kind='AF', seed=int(self.rng.integers(1 << 30))))
        self.add_gt('Atrial Fibrillation', first, self.last_r, expected, group,
                    dict(mean_hr=mean_hr, rr_cv=cv, f_wave_start_s=round(f0, 3)), note)
        self.last_sinus = self.last_r
        return first, self.last_r

    def flutter(self, atrial_rate, ratios, dur, group='', expected='yes', note=''):
        T = 60.0 / atrial_rate
        f0 = self.last_r + 0.25
        k, first, i = 0, None, 0
        conduct = 0.26
        while True:
            ratio = ratios[i % len(ratios)]
            k += ratio
            r = f0 + k * T + conduct
            if r > f0 + dur:
                break
            self.beat(r, 'N', 'N', rhythm='Atrial Flutter')
            first = r if first is None else first
            i += 1
        f1 = self.last_r + 0.5 * T
        self.cont.append(dict(t0=f0, t1=f1, kind='AFL', period=T))
        self.add_gt('Atrial Flutter', first, self.last_r, expected, group,
                    dict(atrial_rate=atrial_rate, conduction=':'.join(str(x) for x in ratios) + ' (waves per QRS)',
                         ventricular_rate=round(atrial_rate / np.mean(ratios), 1)), note)
        self.last_sinus = self.last_r
        return first, self.last_r

    def svt(self, rate, dur, group='', expected='yes', note=''):
        rr = 60.0 / rate
        r = self.last_r + 0.42
        self.pwave(p_from_r(r, 0.13, 'N'), 1, True)       # PAC that triggers it
        start = r
        while r < start + dur:
            self.beat(r, 'N', 'N', rhythm='SVT')
            r += rr * (1 + self.rng.normal(0, 0.004))
        self.add_gt('Supraventricular Tachycardia (SVT)', start, self.last_r, expected, group,
                    dict(rate_bpm=rate), note)
        self.last_sinus = self.last_r
        return start, self.last_r

    # ---- AV conduction ----
    def av_block(self, kind, atrial_rate, dur, morph='N', pr=0.18, cycles=None, groups=None,
                 esc_rate=None, esc_morph='N', label=None, group='', expected='yes', note=''):
        pp = 60.0 / atrial_rate
        p = self.last_r + 0.35
        t_end = p + dur
        first_r = None
        params = dict(atrial_rate=atrial_rate)
        rh = label or kind
        if kind == 'complete':
            esc_rr = 60.0 / esc_rate
            r = self.last_r + esc_rr
            while p < t_end:
                self.pwave(p, 0, False)
                p += pp * (1 + self.rng.normal(0, 0.004))
            while r < t_end:
                self.beat(r, esc_morph, 'J' if esc_morph == 'N' else 'E', rhythm=rh)
                first_r = r if first_r is None else first_r
                r += esc_rr * (1 + self.rng.normal(0, 0.006))
            params.update(escape_rate=esc_rate, escape='junctional (narrow)' if esc_morph == 'N' else 'ventricular (wide)')
        else:
            seq = []    # per P: PR or None (blocked)
            if kind == 'first':
                params.update(pr_ms=pr * 1000)
            elif kind == 'wenckebach':
                params.update(pr_cycles_ms=[[int(x * 1000) for x in c] for c in cycles])
            elif kind in ('mobitz2', '2to1'):
                params.update(pr_ms=pr * 1000, conducted_per_group=groups)
            ci = 0
            while p < t_end:
                if kind == 'first':
                    seq = [pr]
                elif kind == 'wenckebach':
                    seq = list(cycles[ci % len(cycles)]) + [None]
                else:
                    seq = [pr] * groups[ci % len(groups)] + [None]
                ci += 1
                for x in seq:
                    if p >= t_end:
                        break
                    if x is None:
                        self.pwave(p, 0, False)
                    else:
                        self.pwave(p, 0, True)
                        r = r_from_p(p, x, morph)
                        self.beat(r, morph, 'N', x, rh)
                        first_r = r if first_r is None else first_r
                    p += pp * (1 + self.rng.normal(0, 0.002))
            # end on a conducted beat
        params['qrs_morph'] = morph
        self.add_gt(label, first_r, self.last_r, expected, group, params, note)
        self.last_sinus = self.last_r
        return first_r, self.last_r


# --------------------------------------------------------------------------
# Episode catalogue.  Each entry: (name, approx duration s, function(tl, gid))
# --------------------------------------------------------------------------
def _nsr_for(tl, dur, morph='N', **kw):
    tl.nsr(tl.last_r + dur, morph=morph, ectopy=False, **kw)


def _isch(tl, dur, spec, group, label, expected='yes', note=''):
    t0 = tl.last_r + 0.01
    tl.isch.append(dict(t0=t0, t1=t0 + dur, **spec))
    _nsr_for(tl, dur)
    tl.add_gt(label, t0, t0 + dur, expected, group, dict(expected_mm=expected_ischemia(spec)), note)


def expected_ischemia(spec):
    """Expected per-lead values for all 12 leads: ST in screen mm at J+40, T in mV."""
    st = {}
    for name in IND:
        st0, slope = spec.get('st', {}).get(name, (0.0, 0.0))
        st[name] = _st_shape(st0, slope, 0.040)
    out = {}
    if spec.get('st'):
        d = derive_dict(st)
        out['st_mm_at_J40'] = {k: round(v * 10.0, 2) for k, v in d.items() if abs(v) > 1e-9}
        d20 = derive_dict({n: _st_shape(*spec.get('st', {}).get(n, (0.0, 0.0)), 0.020) for n in IND})
        out['st_mm_at_J20'] = {k: round(v * 10.0, 2) for k, v in d20.items() if abs(v) > 1e-9}
    if spec.get('t'):
        base = dict(zip(IND, MORPHS['N']['T'][0]))
        base.update(spec['t'])
        d = derive_dict(base)
        out['t_amp_mv_true'] = {k: round(v, 3) for k, v in d.items()}
        out['t_amp_as_read_at_200adc_per_mv'] = {k: round(v * ADC_PER_MV / 200.0, 3) for k, v in d.items()}
    return out


def derive_dict(v):
    I, II = v['I'], v['II']
    d = dict(I=I, II=II, III=II - I, aVR=-(I + II) / 2, aVL=I - II / 2, aVF=II - I / 2)
    for n in IND[2:]:
        d[n] = v[n]
    return d


def _st_shape(st0, slope, dt):
    v = st0 + slope * dt
    if st0 < 0 and slope > 0:
        v = min(v, 0.0)
    if st0 > 0 and slope < 0:
        v = max(v, 0.0)
    if st0 != 0 and abs(v) > 2.5 * abs(st0):
        v = math.copysign(2.5 * abs(st0), v)
    return v


def catalogue():
    E = []
    add = lambda name, dur, fn: E.append((name, dur, fn))

    # ---------------- Rate ----------------
    def tach(peak, dur, exp='yes', note='', morph='N', extra_label=None):
        def f(tl, g):
            fn = tl.ramp_fn(dur, peak)
            i0, i1 = tl.sinus_profile(dur, fn, morph=morph, rhythm='Sinus Tachycardia')
            tl.rate_gt(i0, i1, 'Sinus Tachycardia', lambda rr: rr <= 0.600, g,
                       dict(peak_hr=peak, ramp_s=90), exp, note)
            if extra_label:
                tl.add_gt(extra_label, tl.R[i0], tl.R[i1 - 1], 'yes', g, dict(qrs_morph=morph))
        return f

    def brady(nadir, dur, exp='yes', note='', morph='N', extra_label=None):
        def f(tl, g):
            fn = tl.ramp_fn(dur, nadir)
            i0, i1 = tl.sinus_profile(dur, fn, morph=morph, pr_fixed=0.17, rhythm='Sinus Bradycardia')
            tl.rate_gt(i0, i1, 'Sinus Bradycardia', lambda rr: rr >= 1.050, g,
                       dict(nadir_hr=nadir, ramp_s=90), exp, note)
            if extra_label:
                tl.add_gt(extra_label, tl.R[i0], tl.R[i1 - 1], 'yes', g, dict(qrs_morph=morph))
        return f

    add('Sinus tach 125', 480, tach(125, 480))
    add('Sinus tach 145', 240, tach(145, 240))
    add('Sinus tach 108 (borderline)', 600, tach(108, 600, 'borderline', 'peak RR ~555 ms, just inside 600 ms'))
    add('Sinus brady 48', 900, brady(48, 900))
    add('Sinus brady 40', 600, brady(40, 600))
    add('Sinus brady 55 (borderline)', 360, brady(55, 360, 'borderline', 'RR ~1090 ms, just past 1050 ms'))

    # ---------------- Ventricular ----------------
    add('VT 170', 30, lambda tl, g: tl.vt(170, 30, group=g))
    add('VT 140 + dissociated P', 120, lambda tl, g: tl.vt(140, 120, True, group=g,
        note='sinus P waves keep marching through the VT (AV dissociation)'))
    add('VT 200', 15, lambda tl, g: tl.vt(200, 15, group=g))
    add('NSVT 5 beats', 2, lambda tl, g: tl.vt(160, 1.6, group=g, expected='borderline',
        note='5-beat run; clinically NSVT. VT core needs >=3 RRs, so the rule may fire'))
    add('VFib coarse', 45, lambda tl, g: tl.vfib(45, 0.7, 4.8, group=g))
    add('VFib fine', 30, lambda tl, g: tl.vfib(30, 0.18, 6.5, group=g,
        note='fine VF, low amplitude: tests the SQI gate does not drop it'))
    add('VFib coarse slow', 30, lambda tl, g: tl.vfib(30, 0.9, 3.4, group=g,
        note='big rounded waves: the coarse-VFib path (B6)'))

    def vt_vf(tl, g):
        s, e = tl.vt(180, 20, group=g, note='degenerates into VFib')
        tl.last_r = e
        tl.vfib(40, 0.6, 5.0, group=g, start=e + 0.22, note='VT -> VFib edge (B13)')
    add('VT -> VFib', 62, vt_vf)

    # ---------------- Atrial ----------------
    add('AFib 95', 1200, lambda tl, g: tl.afib(95, 1200, group=g))
    add('AFib RVR 130', 300, lambda tl, g: tl.afib(130, 300, cv=0.18, group=g))
    add('AFib slow 70', 3600, lambda tl, g: tl.afib(70, 3600, cv=0.20, group=g))
    add('Flutter 2:1', 300, lambda tl, g: tl.flutter(300, [2], 300, group=g))
    add('Flutter 4:1', 600, lambda tl, g: tl.flutter(300, [4], 600, group=g))
    add('Flutter 3:1', 240, lambda tl, g: tl.flutter(270, [3], 240, group=g))
    add('Flutter variable 2:1/4:1', 300, lambda tl, g: tl.flutter(300, [2, 2, 4, 2, 4, 4], 300, group=g,
        note='variable conduction: RR irregular but always a multiple of the F-F interval'))

    def af_to_afl(tl, g):
        tl.afib(100, 360, group=g, note='AFib -> Flutter edge (B13)')
        tl.flutter(300, [2], 240, group=g)
    add('AFib -> Flutter', 600, af_to_afl)
    add('SVT 190', 180, lambda tl, g: tl.svt(190, 180, group=g))
    add('SVT 220', 45, lambda tl, g: tl.svt(220, 45, group=g))

    def svt_to_af(tl, g):
        tl.svt(200, 60, group=g, note='SVT -> AFib edge (B13)')
        tl.afib(110, 180, group=g)
    add('SVT -> AFib', 240, svt_to_af)

    def afl_to_1avb(tl, g):
        tl.flutter(300, [2], 180, group=g, note='Flutter -> First-degree AV Block edge (B13)')
        tl.last_r += 0.6
        tl.av_block('first', 72, 300, pr=0.26, label='First-degree AV Block', group=g)
    add('Flutter -> 1st AVB', 480, afl_to_1avb)

    # ---------------- AV blocks ----------------
    add('1st AVB PR 260', 900, lambda tl, g: tl.av_block('first', 70, 900, pr=0.26, label='First-degree AV Block', group=g))
    add('1st AVB PR 320', 600, lambda tl, g: tl.av_block('first', 62, 600, pr=0.32, label='First-degree AV Block', group=g))
    add('1st AVB PR 215 (borderline)', 480, lambda tl, g: tl.av_block('first', 75, 480, pr=0.215,
        label='First-degree AV Block', group=g, expected='borderline', note='PR just above 200 ms'))
    add('Mobitz I 4:3', 240, lambda tl, g: tl.av_block('wenckebach', 80, 240, cycles=[[0.16, 0.25, 0.29]],
        label='Second-degree AV Block (Mobitz I)', group=g))
    add('Mobitz I mixed + long PR', 360, lambda tl, g: tl.av_block('wenckebach', 72, 360,
        cycles=[[0.16, 0.26], [0.146, 0.206, 0.386, 0.456], [0.17, 0.25, 0.30, 0.33]],
        label='Second-degree AV Block (Mobitz I)', group=g,
        note='3:2, 5:4 (incl. the 146->206->386->456 ms long-PR cycle) and 5:4'))
    add('Mobitz II 4:3/3:2', 240, lambda tl, g: tl.av_block('mobitz2', 80, 240, pr=0.18, groups=[3, 2, 3, 3],
        label='Second-degree AV Block (Mobitz II)', group=g))
    add('Mobitz II 2:1', 180, lambda tl, g: tl.av_block('2to1', 90, 180, pr=0.17, groups=[1],
        label='Second-degree AV Block (Mobitz II)', group=g, note='2:1 block, ventricular rate 45'))
    add('3rd AVB junctional 42', 300, lambda tl, g: tl.av_block('complete', 85, 300, esc_rate=42, esc_morph='N',
        label='Third-degree AV Block', group=g))
    add('3rd AVB ventricular escape 32', 240, lambda tl, g: tl.av_block('complete', 75, 240, esc_rate=32, esc_morph='V',
        label='Third-degree AV Block', group=g, note='wide escape: tests the one-shape fallback'))

    # ---------------- Bundle branch blocks ----------------
    def bbb(morph, label, dur, note=''):
        def f(tl, g):
            t0 = tl.last_r
            tl.nsr(t0 + dur, morph=morph, ectopy=False, rhythm='Normal')
            i0 = next(i for i in range(len(tl.R) - 1, -1, -1) if tl.R[i] <= t0) + 1
            tl.add_gt(label, tl.R[i0], tl.last_r, 'yes', g, dict(qrs_morph=morph), note)
        return f
    add('Complete RBBB', 1200, bbb('RBBB', 'Complete Right Bundle Branch Block', 1200))
    add('Complete LBBB', 1200, bbb('LBBB', 'Complete Left Bundle Branch Block', 1200))
    add('Incomplete RBBB', 900, bbb('iRBBB', 'Incomplete Right Bundle Branch Block', 900, 'design QRS ~112 ms'))
    add('Incomplete LBBB', 900, bbb('iLBBB', 'Incomplete Left Bundle Branch Block', 900, 'design QRS ~110 ms'))

    def rbbb_1avb(tl, g):
        s, e = tl.av_block('first', 68, 600, pr=0.24, morph='RBBB', label='First-degree AV Block', group=g)
        tl.add_gt('Complete Right Bundle Branch Block', s, e, 'yes', g, dict(qrs_morph='RBBB'), 'BBB layered on 1st AVB')
    add('RBBB + 1st AVB', 600, rbbb_1avb)
    add('LBBB + sinus brady 50', 600, brady(50, 600, morph='LBBB', extra_label='Complete Left Bundle Branch Block',
        note='BBB layered on bradycardia; both labels expected'))
    add('RBBB + sinus tach 115', 480, tach(115, 480, morph='RBBB', extra_label='Complete Right Bundle Branch Block',
        note='BBB layered on tachycardia; both labels expected'))

    def rbbb_to_lbbb(tl, g):
        bbb('RBBB', 'Complete Right Bundle Branch Block', 300, 'RBBB -> LBBB beat to beat')(tl, g)
        bbb('LBBB', 'Complete Left Bundle Branch Block', 300)(tl, g)
    add('RBBB -> LBBB', 600, rbbb_to_lbbb)

    # ---------------- Ischemia ----------------
    STE, STD, TI = 'ST Elevation', 'ST Depression', 'T Inversion'
    add('STE inferior', 120, lambda tl, g: _isch(tl, 120, dict(st={'I': (-0.05, 0), 'II': (0.25, 0)}), g, STE,
        note='II/III/aVF elevated, aVL reciprocal depression'))
    add('STE anterior', 240, lambda tl, g: _isch(tl, 240, dict(st={'V2': (0.30, 0), 'V3': (0.35, 0), 'V4': (0.25, 0), 'V5': (0.10, 0)}), g, STE))
    add('STD lateral horizontal', 360, lambda tl, g: _isch(tl, 360, dict(st={'V4': (-0.15, 0), 'V5': (-0.18, 0), 'V6': (-0.12, 0)}), g, STD))
    add('STD inferior downsloping', 180, lambda tl, g: _isch(tl, 180, dict(st={'II': (-0.10, -1.0)}), g, STD,
        note='downsloping, 1 mV/s'))

    def std_tach(tl, g):
        dur = 420
        fn = tl.ramp_fn(dur, 116)
        t0 = tl.last_r
        spec = dict(st={'V4': (-0.20, 0), 'V5': (-0.20, 0), 'V6': (-0.15, 0)})
        tl.isch.append(dict(t0=t0 + 120, t1=t0 + dur - 120, **spec))
        i0, i1 = tl.sinus_profile(dur, fn, rhythm='Sinus Tachycardia')
        tl.rate_gt(i0, i1, 'Sinus Tachycardia', lambda rr: rr <= 0.600, g, dict(peak_hr=116, ramp_s=90))
        tl.add_gt(STD, t0 + 120, t0 + dur - 120, 'yes', g, dict(expected_mm=expected_ischemia(spec)),
                  'demand ischemia during sinus tachycardia (HR < 120, so J+40 is used)')
    add('STD during sinus tach', 420, std_tach)
    add('Upsloping J-depression (neg)', 240, lambda tl, g: _isch(tl, 240,
        dict(st={'V3': (-0.30, 4.0), 'V4': (-0.30, 4.0), 'V5': (-0.30, 4.0), 'V6': (-0.25, 4.0)}), g, STD, 'no',
        'upsloping: rises 0.8 mm J+20 -> J+40, must NOT count'))
    add('STE 20 s only (neg)', 20, lambda tl, g: _isch(tl, 20, dict(st={'V2': (0.30, 0), 'V3': (0.30, 0)}), g, STE, 'no',
        'shorter than 30 s'))
    add('STE single lead (neg)', 180, lambda tl, g: _isch(tl, 180, dict(st={'V2': (0.30, 0)}), g, STE, 'no',
        'only one chest lead'))
    add('STE limb 0.9 mm (neg)', 180, lambda tl, g: _isch(tl, 180, dict(st={'II': (0.09, 0)}), g, STE, 'no',
        'II/III/aVF +0.9 mm, below 1 mm'))
    add('T inversion anterior', 900, lambda tl, g: _isch(tl, 900, dict(t={'V2': -0.30, 'V3': -0.35, 'V4': -0.30, 'V5': -0.20}), g, TI))
    add('T inversion inferior', 600, lambda tl, g: _isch(tl, 600, dict(t={'I': 0.05, 'II': -0.15}), g, TI,
        note='II/III/aVF inverted'))
    add('T inversion 40 s (neg)', 40, lambda tl, g: _isch(tl, 40, dict(t={'V3': -0.30, 'V4': -0.30, 'V5': -0.25}), g, TI, 'no',
        'shorter than 60 s'))

    def ste_evolve(tl, g):
        _isch(tl, 180, dict(st={'I': (-0.05, 0), 'II': (0.22, 0)}), g, STE, note='then evolves into T inversion')
        _isch(tl, 300, dict(t={'I': 0.05, 'II': -0.18}), g, TI)
    add('STE -> T inversion', 480, ste_evolve)

    # ---------------- Negative controls (no label expected) ----------------
    def artifact(dur, amp):
        def f(tl, g):
            t0 = tl.last_r + 0.2
            tl.cont.append(dict(t0=t0, t1=t0 + dur, kind='ART', amp=amp, seed=int(tl.rng.integers(1 << 30))))
            _nsr_for(tl, dur + 0.4)
            tl.add_gt('none (motion artifact)', t0, t0 + dur, 'no', g, dict(amp_mv=amp),
                      'normal beats under heavy motion noise; SQI should blank it, no VFib')
        return f
    add('Motion artifact 20 s', 21, artifact(20, 0.8))
    add('Motion artifact 60 s', 61, artifact(60, 1.2))

    def mains(tl, g):
        t0 = tl.last_r + 0.1
        tl.cont.append(dict(t0=t0, t1=t0 + 120, kind='MAINS', amp=0.12))
        _nsr_for(tl, 120.2)
        tl.add_gt('none (50 Hz interference)', t0, t0 + 120, 'no', g, dict(amp_mv=0.12), 'must not read as AFib')
    add('50 Hz interference', 120, mains)

    def bigeminy(tl, g):
        t0 = tl.last_r
        t = t0
        while t < t0 + 180:
            rr = tl.bg_rr(t)
            tn = t + rr
            tl.pwave(p_from_r(tn, 0.16, 'N'), 0, True); tl.beat(tn, 'N', 'N', 0.16, 'Bigeminy')
            tl.beat(tn + 0.55 * rr, 'V', 'V', rhythm='Bigeminy')
            tl.pwave(p_from_r(tn + rr, 0.16, 'N'), 0, False)
            t = tn + rr
        tn = t + tl.bg_rr(t)
        tl.pwave(p_from_r(tn, 0.16, 'N'), 0, True); tl.beat(tn, 'N', 'N', 0.16)
        tl.add_gt('none (ventricular bigeminy)', t0, tl.last_r, 'no', g, {}, 'must not be VT / VFib / AFib')
    add('Ventricular bigeminy', 180, bigeminy)

    def couplets(tl, g):
        t0 = tl.last_r
        end = t0 + 180
        while tl.last_r < end:
            tl.nsr(min(end, tl.last_r + 20), ectopy=False)
            rr = tl.bg_rr(tl.last_r)
            v1 = tl.last_r + 0.55 * rr
            tl.beat(v1, 'V', 'V', rhythm='PVC couplet'); tl.beat(v1 + 0.45, 'V', 'V', rhythm='PVC couplet')
            tl.last_r = v1 + 0.45 + 1.1 * rr - rr   # next sinus beat after the pause
        tl.nsr(tl.last_r + 2, ectopy=False)
        tl.add_gt('none (PVC couplets)', t0, tl.last_r, 'no', g, {}, 'a couplet every ~20 s')
    add('PVC couplets', 182, couplets)

    def pacs(tl, g):
        t0 = tl.last_r
        n = 0
        while tl.last_r < t0 + 300:
            rr = tl.bg_rr(tl.last_r)
            n += 1
            if n % 3 == 0:
                ta = tl.last_r + 0.68 * rr
                tl.pwave(p_from_r(ta, 0.15, 'N'), 1, True); tl.beat(ta, 'N', 'A', 0.15, 'Frequent PACs')
            else:
                tn = tl.last_r + rr
                tl.pwave(p_from_r(tn, 0.165, 'N'), 0, True); tl.beat(tn, 'N', 'N', 0.165, 'Frequent PACs')
        tl.add_gt('none (frequent PACs, every 3rd beat)', t0, tl.last_r, 'no', g, {}, 'irregular but organized P: not AFib')
    add('Frequent PACs', 300, pacs)

    def rsa(tl, g):
        t0 = tl.last_r
        tl.nsr(t0 + 600, ectopy=False, rsa=0.12, rhythm='Sinus arrhythmia')
        tl.add_gt('none (marked sinus arrhythmia)', t0, tl.last_r, 'no', g, dict(rsa_pct=12), 'not AFib')
    add('Marked sinus arrhythmia', 600, rsa)

    def pauses(tl, g):
        t0 = tl.last_r
        for gap in (2.8, 3.4):
            _nsr_for(tl, 30)
            tl.last_r += gap - tl.bg_rr(tl.last_r)   # sinus arrest: no P, no QRS
            tl.last_sinus = tl.last_r
        _nsr_for(tl, 30)
        tl.add_gt('none (two sinus pauses 2.8 s and 3.4 s)', t0, tl.last_r, 'no', g, {},
                  'single long RRs must not become Sinus Bradycardia')
    add('Sinus pauses', 95, pauses)
    return E


# --------------------------------------------------------------------------
# Build the whole timeline
# --------------------------------------------------------------------------
def build_timeline(total_sec, start_epoch, seed):
    rng = np.random.default_rng(seed)
    tl = Timeline(total_sec, start_epoch, rng)
    cat = catalogue()
    order = rng.permutation(len(cat))
    cat = [cat[i] for i in order]
    lead_in = 600.0
    sum_dur = sum(d for _, d, _ in cat)
    gap = max(120.0, (total_sec - lead_in - 600.0 - sum_dur) / len(cat))
    t = lead_in
    placed = 0
    for i, (name, dur, fn) in enumerate(cat):
        start = t + gap * rng.uniform(0.6, 1.0)
        if start + dur + 300 > total_sec:
            print(f"  [skip] no room for '{name}' (use more --hours)")
            continue
        tl.nsr(start)
        n_before = len(tl.gt)
        fn(tl, f"E{placed + 1:02d}")
        for g in tl.gt[n_before:]:
            g['episode'] = name
        placed += 1
        tl.nsr(tl.last_r + 30, ectopy=False)     # clean normal rhythm right after every episode
        t = tl.last_r
    tl.nsr(total_sec - 0.5)
    print(f"  {placed} episodes placed, {len(tl.R)} beats, {len(tl.P)} P waves")
    return tl


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
_TT_Q = np.arange(int(-0.15 * FS), int(0.70 * FS)) / FS
_TT_P = np.arange(int(-0.08 * FS), int(0.08 * FS) + 1) / FS


def _gauss(t, mu, s):
    return np.exp(-0.5 * ((t - mu) / s) ** 2)


def qrst_template(morph, rr, isch):
    m = MORPHS[morph]
    tt = _TT_Q
    out = np.zeros((8, tt.size))
    for mu, s, amps in m['waves']:
        out += np.outer(amps, _gauss(tt, mu, s))
    t_amps, t_delay, t_ws = m['T']
    t_amps = np.array(t_amps, dtype=float)
    rrc = min(1.6, max(0.28, rr))
    sq = math.sqrt(rrc)
    tp = 0.27 * sq + 0.02 + t_delay
    sl, sr = 0.045 * sq * t_ws, 0.030 * sq * t_ws
    tshape = np.where(tt < tp, _gauss(tt, tp, sl), _gauss(tt, tp, sr))
    if isch is not None and morph in ISCHEMIA_MORPHS:
        for name, amp in isch.get('t', {}).items():
            t_amps[LI[name]] = amp
        j = m['j']
        rise = 1.0 / (1.0 + np.exp(-(tt - j) / 0.003))
        fall = np.where(tt > tp, _gauss(tt, tp, sr), 1.0)
        for name, (st0, slope) in isch.get('st', {}).items():
            v = st0 + slope * np.clip(tt - j, 0, None)
            if st0 < 0 and slope > 0:
                v = np.minimum(v, 0.0)
            if st0 > 0 and slope < 0:
                v = np.maximum(v, 0.0)
            v = np.clip(v, -2.5 * abs(st0), 2.5 * abs(st0))
            out[LI[name]] += v * rise * fall
    out += np.outer(t_amps, tshape)
    return out


def p_template(ptype):
    s, amps = P_SINUS if ptype == 0 else P_ECTOPIC
    out = np.outer(amps, _gauss(_TT_P, 0.0, s))
    if ptype == 0:
        out[LI['V1']] += -0.05 * _gauss(_TT_P, 0.028, 0.014)   # biphasic P in V1
    return out


class Renderer:
    def __init__(self, tl, seed):
        self.tl = tl
        self.seed = seed
        self.R = np.asarray(tl.R)
        order = np.argsort(self.R)
        self.R = self.R[order]
        self.M = np.asarray(tl.M)[order]
        rr = np.diff(self.R, prepend=self.R[0] - 0.8)
        # ischemia overlay per beat
        iso = np.full(self.R.size, -1)
        for k, ov in enumerate(tl.isch):
            sel = (self.R >= ov['t0']) & (self.R < ov['t1'])
            iso[sel] = k
        bucket = np.clip(np.round(rr * 100).astype(int), 25, 200)
        self.keys = list(zip(self.M.tolist(), bucket.tolist(), iso.tolist()))
        self._cache = {}
        self.P = np.asarray(tl.P)
        po = np.argsort(self.P)
        self.P = self.P[po]
        self.PT = np.asarray(tl.PT)[po]
        self._ptemp = {0: p_template(0), 1: p_template(1)}
        r = np.random.default_rng(seed + 7)
        # continuous baseline wander + mains, per lead
        self.wander = [(r.uniform(0.02, 0.05), 0.22 + r.uniform(-0.03, 0.03), r.uniform(0, 6.3),
                        [(r.uniform(0.02, 0.07), f, r.uniform(0, 6.3)) for f in (0.013, 0.031, 0.057, 0.11)])
                       for _ in IND]
        self.mains_phase = r.uniform(0, 6.3, len(IND))
        self.vf_w = r.uniform(-1, 1, (len(IND), 2))
        self.vf_w /= np.abs(self.vf_w).max(axis=1, keepdims=True)
        self.af_w = np.array([0.02, 0.05, 0.08, 0.05, 0.03, 0.02, 0.02, 0.02])
        self.afl_w = np.array([0.03, -0.25, 0.15, 0.07, 0.04, 0.03, 0.03, 0.03])

    def template(self, key):
        t = self._cache.get(key)
        if t is None:
            morph_id, bucket, iso = key
            isch = self.tl.isch[iso] if iso >= 0 else None
            t = qrst_template(MORPH_LIST[morph_id], bucket / 100.0, isch)
            self._cache[key] = t
        return t

    def _emg_level(self, t):
        h = self.tl.clock_hour(t)
        return 0.006 if (h >= 23 or h < 6) else 0.011

    def render(self, s0, n):
        """mV array (8, n) for samples s0 .. s0+n."""
        out = np.zeros((8, n))
        t0 = s0 / FS
        t1 = (s0 + n) / FS
        tt = t0 + np.arange(n) / FS
        # QRS-T
        a = np.searchsorted(self.R, t0 - 0.70)
        b = np.searchsorted(self.R, t1 + 0.15)
        off0 = int(round(_TT_Q[0] * FS))
        L = _TT_Q.size
        for i in range(a, b):
            tmp = self.template(self.keys[i])
            st = int(round(self.R[i] * FS)) + off0 - s0
            lo, hi = max(0, st), min(n, st + L)
            if hi > lo:
                out[:, lo:hi] += tmp[:, lo - st:hi - st]
        # P waves
        a = np.searchsorted(self.P, t0 - 0.1)
        b = np.searchsorted(self.P, t1 + 0.1)
        offp = int(round(_TT_P[0] * FS))
        Lp = _TT_P.size
        for i in range(a, b):
            tmp = self._ptemp[int(self.PT[i])]
            st = int(round(self.P[i] * FS)) + offp - s0
            lo, hi = max(0, st), min(n, st + Lp)
            if hi > lo:
                out[:, lo:hi] += tmp[:, lo - st:hi - st]
        # continuous segments
        for c in self.tl.cont:
            if c['t1'] <= t0 or c['t0'] >= t1:
                continue
            sel = (tt >= c['t0']) & (tt < c['t1'])
            ts = tt[sel]
            env = np.minimum(1.0, np.minimum(ts - c['t0'], c['t1'] - ts) / 0.15)
            kind = c['kind']
            if kind == 'VF':
                out[:, sel] += self._vf(c, ts) * env
            elif kind == 'AF':
                out[:, sel] += np.outer(self.af_w, self._af(c, ts) * env)
            elif kind == 'AFL':
                ph = ((ts - c['t0']) / c['period']) % 1.0
                saw = np.where(ph < 0.2, 0.5 - ph / 0.2, -0.5 + (ph - 0.2) / 0.8)
                out[:, sel] += np.outer(self.afl_w, saw * env)
            elif kind == 'ART':
                out[:, sel] += self._art(c, ts) * env
            elif kind == 'MAINS':
                out[:, sel] += c['amp'] * np.sin(2 * np.pi * 50 * ts[None, :] + self.mains_phase[:, None])
        # baseline wander, mains hum, muscle noise
        for li, (ra, rf, rp, slow) in enumerate(self.wander):
            w = ra * np.sin(2 * np.pi * rf * tt + rp)
            for amp, f, ph in slow:
                w += amp * np.sin(2 * np.pi * f * tt + ph)
            out[li] += w
        out += 0.004 * np.sin(2 * np.pi * 50 * tt[None, :] + self.mains_phase[:, None])
        rng = np.random.default_rng((self.seed * 1000003 + s0) % (1 << 63))
        out += rng.standard_normal((8, n)) * self._emg_level(t0)
        return out

    def _vf(self, c, ts):
        u = ts - c['t0']
        r = np.random.default_rng(c['seed'])
        base = []
        for _ in range(2):
            p1, p2, p3, p4 = r.uniform(0, 6.3, 4)
            ph = 2 * np.pi * c['f0'] * u + 2.0 * np.sin(2 * np.pi * 0.37 * u + p1) + 0.8 * np.sin(2 * np.pi * 0.83 * u + p2)
            A = c['amp'] * (0.75 + 0.25 * np.sin(2 * np.pi * 0.15 * u + p3)) * (0.85 + 0.15 * np.sin(2 * np.pi * 0.53 * u + p4))
            base.append(A * (np.sin(ph) + 0.35 * np.sin(2 * ph + 0.7)))
        base = np.vstack(base)
        return self.vf_w @ base

    def _af(self, c, ts):
        r = np.random.default_rng(c['seed'])
        s = np.zeros_like(ts)
        for f, m in ((5.8, 0.31), (6.7, 0.53), (7.4, 0.77)):
            s += r.uniform(0.6, 1.0) * np.sin(2 * np.pi * f * ts + 1.5 * np.sin(2 * np.pi * m * ts + r.uniform(0, 6.3)) + r.uniform(0, 6.3))
        return s / 2.0

    def _art(self, c, ts):
        r = np.random.default_rng(c['seed'])
        out = np.zeros((8, ts.size))
        for li in range(8):
            fs_ = r.uniform(0.5, 12.0, 25)
            ph = r.uniform(0, 6.3, 25)
            am = r.uniform(0.3, 1.0, 25) / fs_ ** 0.5
            sig = (am[:, None] * np.sin(2 * np.pi * fs_[:, None] * ts[None, :] + ph[:, None])).sum(axis=0)
            out[li] = sig / (np.abs(am).sum() * 0.35) * c['amp'] * r.uniform(0.5, 1.0)
        return out


def to_frames(mv, f0):
    """mV (8, n) -> big-endian frame block (n, 13) exactly like the device."""
    adc = np.rint(mv * ADC_PER_MV).astype(np.int64) + ADC_OFFSET
    I, II = adc[0], adc[1]
    leads = np.empty((12, adc.shape[1]), dtype=np.int64)
    leads[0], leads[1] = I, II
    leads[2] = II - I
    leads[3] = -(I + II) // 2
    leads[4] = I - II // 2
    leads[5] = II - I // 2
    leads[6:] = adc[2:]
    np.clip(leads, -32768, 32767, out=leads)
    n = adc.shape[1]
    out = np.empty((n, 13), dtype='>i2')
    out[:, :12] = leads.T
    out[:, 12] = ((f0 + np.arange(n)) % 65536).astype('>u2').view('>i2')
    return out


# --------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------
def write_header(f, start_epoch, patient):
    hdr = bytearray(HEADER_SIZE)
    struct.pack_into('>4sHHI', hdr, 0, b'ECGH', 1, 12, FS)
    struct.pack_into('>d', hdr, 12, start_epoch)
    for i, name in enumerate(LEAD_NAMES):
        hdr[20 + i * 5: 25 + i * 5] = name.encode('ascii')[:5].ljust(5, b'\x00')
    info = json.dumps({'name': patient['name'], 'dob': '', 'gender': patient['gender'], 'doctor': ''}).encode()[:152]
    hdr[84:84 + len(info)] = info
    f.write(bytes(hdr))


def fmt_clock(start_epoch, t):
    dt = datetime.fromtimestamp(start_epoch + t)
    day = int(t // 86400) + 1
    return f"Day{day} {dt.strftime('%Y-%m-%d %H:%M:%S')}"


def write_ground_truth(tl, sess, start_epoch):
    gt = sorted(tl.gt, key=lambda g: g['start_s'])
    for i, g in enumerate(gt, 1):
        g['id'] = i
        g['duration_s'] = round(g['end_s'] - g['start_s'], 3)
        g['start_clock'] = fmt_clock(start_epoch, g['start_s'])
        g['end_clock'] = fmt_clock(start_epoch, g['end_s'])
    with open(os.path.join(sess, 'ground_truth_events.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(start_epoch=start_epoch, adc_per_mv=ADC_PER_MV, fs=FS,
                       note='start_s/end_s are seconds from recording start. Beat episodes run from the '
                            'first to the last QRS (R peak) of the episode; VFib / artifact from signal '
                            'start to end; ischemia from overlay start to end. expected: yes = the '
                            'detector should mark it, no = negative control, borderline = near a threshold.',
                       events=gt), f, indent=1)
    cols = ['id', 'group', 'episode', 'label', 'expected', 'start_s', 'end_s', 'duration_s',
            'start_clock', 'end_clock', 'params', 'note']
    with open(os.path.join(sess, 'ground_truth_events.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for g in gt:
            w.writerow([g['id'], g['group'], g.get('episode', ''), g['label'], g['expected'], g['start_s'],
                        g['end_s'], g['duration_s'], g['start_clock'], g['end_clock'],
                        json.dumps(g['params']), g['note']])
    R = np.asarray(tl.R); o = np.argsort(R)
    with open(os.path.join(sess, 'ground_truth_beats.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['r_time_s', 'sample', 'beat_type', 'morphology', 'rhythm', 'pr_ms'])
        K, M, PRs, RH = tl.K, tl.M, tl.PR, tl.RH
        for i in o:
            pr = PRs[i]
            w.writerow([f"{R[i]:.3f}", int(round(R[i] * FS)), K[i], MORPH_LIST[M[i]], RH[i],
                        '' if np.isnan(pr) else f"{pr:.0f}"])
    P = np.asarray(tl.P); o = np.argsort(P)
    with open(os.path.join(sess, 'ground_truth_pwaves.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['p_peak_time_s', 'p_onset_time_s', 'type', 'conducted'])
        for i in o:
            w.writerow([f"{P[i]:.3f}", f"{P[i] - 0.05:.3f}", 'sinus' if tl.PT[i] == 0 else 'ectopic', int(tl.PC[i])])
    return gt


def write_session_files(sess, start_epoch, duration, frames, patient):
    ecgh = os.path.join(sess, 'recording.ecgh')
    index = []
    for fr in range(0, frames, FS):
        index.append({'time': start_epoch + (fr + 1) / FS, 'frame': fr, 'offset': HEADER_SIZE + fr * FRAME_SIZE})
    idx_path = os.path.join(sess, 'recording_index.json')
    with open(idx_path, 'w') as f:
        json.dump({'path': ecgh, 'start_time': start_epoch, 'duration_sec': duration, 'frames': frames,
                   'fs': FS, 'n_leads': 12, 'index': index}, f)
    pinfo = {"patient_name": patient['name'], "age": patient['age'], "email": "", "doctor": "", "org": "",
             "phone": "", "gender": patient['gender'], "sex": patient['gender'], "name": patient['name'], "Org.": ""}
    with open(os.path.join(sess, 'patient.json'), 'w') as f:
        json.dump(pinfo, f, indent=4)
    events_db = os.path.join(sess, 'events.db')
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        ss_path = os.path.join(here, '..', 'src', 'ecg', 'holter', 'session_store.py')
        spec = importlib.util.spec_from_file_location('_session_store', ss_path)
        ss = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ss)
        events_db = ss.ensure_events_db(sess)
    except Exception as e:
        print(f"  [warn] events.db not created ({e}); the app creates it on first use")
    summary = {'duration_sec': round(duration, 2), 'frames': frames, 'path': ecgh, 'index_path': idx_path,
               'session_dir': sess, 'events_db_path': events_db,
               'jsonl_path': os.path.join(sess, 'metrics.jsonl'),
               'session_json_path': os.path.join(sess, 'session.json')}
    meta = {'chunk_seconds': 5, 'ecgh_path': ecgh, 'events_db_path': events_db, 'fs': FS,
            'jsonl_path': summary['jsonl_path'], 'output_dir': os.path.dirname(sess),
            'patient_info': pinfo, 'schema_version': 1, 'session_dir': sess, 'status': 'stopped',
            'summary': summary, 'updated_at': datetime.now(timezone.utc).isoformat(), 'synthetic': True}
    with open(os.path.join(sess, 'session.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def write_previews(sess, gt, renderer, total_sec):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  [warn] matplotlib missing, no previews")
        return
    pdir = os.path.join(sess, 'preview')
    os.makedirs(pdir, exist_ok=True)
    show = ['I', 'II', 'V1', 'V2', 'V5', 'V6']
    for g in gt:
        c = g['start_s']
        t0 = max(0, c - 4); t1 = min(total_sec, t0 + 12)
        s0, n = int(t0 * FS), int((t1 - t0) * FS)
        fr = to_frames(renderer.render(s0, n), s0)
        tt = t0 + np.arange(n) / FS
        fig, axes = plt.subplots(len(show), 1, figsize=(14, 9), sharex=True)
        for ax, name in zip(axes, show):
            li = LEAD_NAMES.index(name)
            y = fr[:, li].astype(float)
            ax.plot(tt, (y - np.median(y)) / ADC_PER_MV, lw=0.7, color='k')
            ax.set_ylabel(name); ax.axvline(c, color='r', lw=0.8)
            ax.set_ylim(-2, 2); ax.grid(alpha=0.3)
        axes[0].set_title(f"#{g['id']} {g['label']} [{g['expected']}]  start {c:.2f}s  ({g.get('episode', '')})")
        fig.tight_layout()
        safe = ''.join(ch if ch.isalnum() else '_' for ch in f"{g['id']:03d}_{g['label']}")[:60]
        fig.savefig(os.path.join(pdir, safe + '.png'), dpi=80)
        plt.close(fig)
    print(f"  previews -> {pdir}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hours', type=float, default=72.0)
    ap.add_argument('--out', default=os.path.normpath(os.path.join(here, '..', 'src', 'recordings')))
    ap.add_argument('--start', default='2026-10-25 09:00:00', help='recording start, local time')
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--name', default=None, help='patient name (default Dummy_<N>h)')
    ap.add_argument('--preview', action='store_true', help='also save a PNG strip at each episode start')
    args = ap.parse_args()

    total = args.hours * 3600.0
    start_dt = datetime.strptime(args.start, '%Y-%m-%d %H:%M:%S')
    start_epoch = start_dt.timestamp()
    tag = f"Dummy_{args.hours:g}h"
    patient = dict(name=args.name or tag, gender='Male', age='58')
    sess = os.path.join(args.out, f"{start_dt.strftime('%Y-%m-%d_%H-%M-%S')}_{tag}")
    os.makedirs(sess, exist_ok=True)
    frames = int(total * FS)
    print(f"Session : {sess}")
    print(f"Length  : {args.hours:g} h, {frames:,} frames, {frames * FRAME_SIZE / 1e9:.2f} GB")

    t_start = time.time()
    tl = build_timeline(total, start_epoch, args.seed)
    gt = write_ground_truth(tl, sess, start_epoch)
    rd = Renderer(tl, args.seed)

    ecgh = os.path.join(sess, 'recording.ecgh')
    chunk = int(CHUNK_SEC * FS)
    with open(ecgh, 'wb') as f:
        write_header(f, start_epoch, patient)
        s0 = 0
        last_print = 0
        while s0 < frames:
            n = min(chunk, frames - s0)
            f.write(to_frames(rd.render(s0, n), s0).tobytes())
            s0 += n
            if time.time() - last_print > 15 or s0 >= frames:
                last_print = time.time()
                el = time.time() - t_start
                pct = s0 / frames
                eta = el / pct * (1 - pct) if pct > 0 else 0
                print(f"  {pct * 100:5.1f}%  {s0 / FS / 3600:6.2f} h written  elapsed {el / 60:5.1f} min  eta {eta / 60:5.1f} min", flush=True)
    write_session_files(sess, start_epoch, frames / FS, frames, patient)
    if args.preview:
        write_previews(sess, gt, rd, total)
    n_yes = sum(1 for g in gt if g['expected'] == 'yes')
    print(f"Done in {(time.time() - t_start) / 60:.1f} min. {len(gt)} ground-truth rows "
          f"({n_yes} should be detected). See ground_truth_events.csv")


if __name__ == '__main__':
    main()
