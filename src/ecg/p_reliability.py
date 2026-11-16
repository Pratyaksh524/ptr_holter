"""Can a P wave be read here? -- the noise side of it.

The rhythm labels that rest on the P wave (Sinus Bradycardia / Tachycardia, the
AV blocks, Tachycardia without a sinus P ...) read a P and a PR off every beat.
A P wave is about a tenth of the QRS, so noise far smaller than the QRS already
buries it, and the detector then takes a noise bump (or the previous T wave) for
the P: PR jumping 96 / 222 ms, the rhythm label lost or wrong. The scores that
were there to catch it do not see this noise: SignalQuality is noise over the
QRS height of a whole 10 s frame (a bigger QRS at a faster rate hides the same
noise, and a 20-50 s episode split over two frames stays under its limit) and
the noise gate scores the signal after a 25 Hz low-pass, which removes muscle
noise. Seen: 2026-11-16_09-48-46, the same P-zone noise at 120 bpm (SQI
0.68-0.72) as at 30 bpm (SQI 0.33-0.47), only the 30 bpm one was cut.

Muscle noise is told by what it is, not by how big it is against the QRS:
broadband power at 65-150 Hz that an ECG does not have, on every lead at once.

Tried and dropped, on all 170 test recordings: a per-beat noise estimate in
the P zone (std of the signal minus its 42 ms moving mean, or after a 10 / 12 /
15 Hz high-pass, or after taking out the median of the neighbouring beats, or
the 65-95 Hz band, each over the QRS height). Clean beats read as high as muscle
noise -- a sharp P wave and the previous T wave at a fast rate (moving mean /
high-pass), a changing shape in VFib / AF / LBBB (neighbour median), a 65-95 Hz
band spread by test signals -- and a limit that kept those out lost the muscle
noise (30% covered at the limit that left 36 s under the labels it would cut).
"""
import numpy as np

# Muscle noise (EMG): broadband power far above anything an ECG has at
# 65-150 Hz, on every lead at once. A 2 s block counts when
#   - on each of Lead II, I and V5 the 65-150 Hz share of the 1-150 Hz power
#     (Welch, 100 / 120 Hz mains harmonics +-3 Hz left out) is >= 2%, and
#   - on Lead II the std above 40 Hz (SignalQuality's own high-pass, after
#     the 50 / 60 Hz notches) is >= 2% of the ECG peak-to-peak (0.5-40 Hz) of
#     the 10 s around the block.
# Measured on all 170 recordings (26964 blocks): 94% of blocks have a share
# under 0.5%, a clean stretch 0.1%, muscle noise 2.6-32%; the share alone also
# read 2-17% on three very quiet recordings (Lead II HF std 1-4 ADC, Sinus
# Bradycardia), which the amplitude test drops (HF ratio 0.001-0.003 there,
# 0.03-0.08 on muscle noise, only 6 blocks between 0.010 and 0.020). A run
# is kept from _MUSCLE_RUN_BLOCKS blocks (4 s, the Artifact rule's floor):
# a lone block in 5 runs was not a stretch of noise.
# Seen: muscle noise 20-50 s long and under the SQI's 0.05 limit on every
# 10 s frame (HF ratio 0.033-0.043, 2026-11-16_09-48-46, 09:56:30 / 09:57:32 /
# 09:58:48) was never looked at beat by beat and stayed Sinus Tachycardia /
# Tachycardia / SVT.
_MUSCLE_BLOCK_SEC = 2.0
_MUSCLE_RUN_BLOCKS = 2
_MUSCLE_SHARE_MIN = 0.02
_MUSCLE_HF_RATIO_MIN = 0.02
_MUSCLE_CONTEXT_BLOCKS = 2
_MUSCLE_CHUNK_SEC = 120.0
_MUSCLE_LEADS = ('II', 'I', 'V5')


def muscle_noise_spans(reader, fs, total_duration):
    """[[start_sec, end_sec], ...] of the muscle-noise stretches of the
    recording (see the _MUSCLE_* constants above), blocks on a 2 s grid from 0 s."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    from scipy.signal import filtfilt, welch
    from .holter.analysis_pipeline import ADC_COUNTS_PER_MV, HolterConfig, SignalQuality
    names = list(getattr(reader, 'lead_names', None) or [])
    if any(n not in names for n in _MUSCLE_LEADS) or total_duration < 2 * _MUSCLE_BLOCK_SEC:
        return []
    idx = [names.index(n) for n in _MUSCLE_LEADS]
    fs = float(fs)
    blk = int(round(_MUSCLE_BLOCK_SEC * fs))
    ctx = _MUSCLE_CONTEXT_BLOCKS
    pad = ctx * _MUSCLE_BLOCK_SEC
    sq = SignalQuality(fs=fs, config=HolterConfig.from_clinical_config())
    flagged = []
    t = 0.0
    while t < total_duration:
        t_end = min(total_duration, t + _MUSCLE_CHUNK_SEC)
        r0 = max(0.0, t - pad)
        data = reader.read_range(r0, min(total_duration, t_end + pad))
        n_blk = 0 if data is None else data.shape[1] // blk
        if n_blk > 0 and data.shape[0] > max(idx):
            share_ok = np.ones(n_blk, dtype=bool)
            for i in idx:
                x = np.asarray(data[i, :n_blk * blk], dtype=float).reshape(n_blk, blk)
                x = x - x.mean(axis=1, keepdims=True)
                f, pw = welch(x, fs=fs, nperseg=min(blk, 500), axis=-1)
                band = (f >= 65.0) & (f <= 150.0) & ~((f >= 97.0) & (f <= 103.0)) & ~((f >= 117.0) & (f <= 123.0))
                tot = pw[:, (f >= 1.0) & (f <= 150.0)].sum(axis=1)
                share_ok &= (tot > 0) & (pw[:, band].sum(axis=1) >= _MUSCLE_SHARE_MIN * np.maximum(tot, 1e-30))
            y = np.asarray(data[idx[0], :n_blk * blk], dtype=float)[None, :]
            y = (y - np.median(y, axis=1, keepdims=True)) / ADC_COUNTS_PER_MV
            for b, a in sq._notches:
                y = filtfilt(b, a, y, axis=-1)
            ecg = filtfilt(*sq._bp, y, axis=-1)[0].reshape(n_blk, blk)
            hf = filtfilt(*sq._hp, y, axis=-1)[0].reshape(n_blk, blk)
            ptp = (maximum_filter1d(ecg.max(axis=1), 2 * ctx + 1, mode='nearest')
                   - minimum_filter1d(ecg.min(axis=1), 2 * ctx + 1, mode='nearest'))
            noisy = share_ok & (hf.std(axis=1) >= _MUSCLE_HF_RATIO_MIN * np.maximum(ptp, 1e-9))
            first = int(round(r0 / _MUSCLE_BLOCK_SEC))
            for k in np.flatnonzero(noisy):
                start = (first + int(k)) * _MUSCLE_BLOCK_SEC
                if t - 1e-6 <= start < t_end - 1e-6:
                    flagged.append(first + int(k))
        t += _MUSCLE_CHUNK_SEC
    spans = []
    run = []
    for j in sorted(set(flagged)) + [None]:
        if run and (j is None or j != run[-1] + 1):
            if len(run) >= _MUSCLE_RUN_BLOCKS:
                spans.append([run[0] * _MUSCLE_BLOCK_SEC,
                              min(float(total_duration), (run[-1] + 1) * _MUSCLE_BLOCK_SEC)])
            run = []
        if j is not None:
            run.append(j)
    return spans
