# -*- coding: utf-8 -*-
"""音频读写与重采样（不依赖 torch）

重采样优先用 soxr（快且质量好），退回到 scipy，再退回线性插值。
"""
import os

import numpy as np


def load_wav(path, target_sr=None):
    """读音频 → (mono float32, sr)"""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if target_sr and sr != target_sr:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav.astype(np.float32), sr


def save_wav(path, wav, sr):
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)


def resample(wav, sr_in, sr_out):
    """任意采样率重采样，mono float32"""
    if sr_in == sr_out:
        return np.asarray(wav, dtype=np.float32)
    wav = np.asarray(wav, dtype=np.float32)
    try:
        import soxr
        return soxr.resample(wav, sr_in, sr_out).astype(np.float32)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr_in), int(sr_out))
        up, down = int(sr_out // g), int(sr_in // g)
        return resample_poly(wav, up, down).astype(np.float32)
    except ImportError:
        pass
    # 最后兜底：线性插值
    n_out = int(round(len(wav) * sr_out / sr_in))
    x_old = np.linspace(0, 1, len(wav), endpoint=False)
    x_new = np.linspace(0, 1, n_out, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def peak_normalize(wav, target=0.95):
    peak = float(np.abs(wav).max()) if len(wav) else 0.0
    if peak < 1e-6:
        return wav
    return (wav * (target / peak)).astype(np.float32)
