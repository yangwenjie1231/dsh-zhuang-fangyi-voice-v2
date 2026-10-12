# -*- coding: utf-8 -*-
"""校准式亮度补偿（high-shelf）

## 为什么需要

模型输出 32 kHz（v2/v2Pro），而素材是 48 kHz。48k→32k 的下采样在
Nyquist(16 kHz) 附近有抗混叠滤波滚降，实测造成 12–16 kHz 频段比原声低
约 2.8 dB。听感上表现为「糊」。

实测数据（`D:\\A\\voiceclone\\calib_brightness.py`）：
    频段        48k 原声   合成(未处理)   补偿后
    8–12 kHz     -34.7      -33.9        -33.8     （+0.09，几乎不动）
    12–16 kHz    -46.8      -49.6        -46.7     （+2.9，对齐原声）

## 诚实说明

这是**听感补偿**，恢复的是重采样滤波造成的频谱倾斜，
**不是信息恢复** —— 12–16 kHz 的真实内容在 32 kHz 里仍然受限。
参数经网格搜索校准（`fc=12k, gain=3dB, Q=1.0`），12–16 kHz 误差 0.09 dB。

## 补偿量按模型版本取（见 `BY_VERSION`）

**默认不在这里决定** —— API 层用 `"auto"` 按加载到的模型版本选：

| 版本 | 补偿 | 依据 |
|---|---|---|
| v1 / v2 | **+3 dB** | 未补偿时 8–12k 偏亮 4.2 dB、12–16 k 已对齐 → 需要补高频 |
| v2Pro / v2ProPlus / v3 / v4 | **0** | 频谱**本就接近原声**，加 3 dB 会过冲 |

实测（长文本 421 秒全文，目标 = 48k 原声的 8–12k −36.7 / 12–16k −45.25）：

    v2    未补偿 8–12k = −32.5（偏亮 4.2）  12–16k = −45.6（已对齐）
    v2Pro 未补偿 8–12k = −36.6（已对齐）     12–16k = −46.5（仅差 1.2）

v2Pro 加 3 dB 后频谱偏差**从 0.88 升到 1.92** —— 所以它不该补偿。

## 零风险短路

`gain_db=0.0` 时**直接返回入参对象**（逐字节不变，不做任何滤波）。
"""
from __future__ import annotations

import math

import numpy as np

# 校准所得（见 calib_brightness.json）
DEFAULT_FC = 12000.0
DEFAULT_Q = 1.0
DEFAULT_GAIN_DB = 0.0

# 各模型版本的校准补偿量（dB）。
#
# ## 校准口径：**长文本**（这是实际用途）
#
# 用 421 秒全文（45 段拼接）扫描，指标 = |d0-4k| + |d8-12k| + |d12-16k|
# 相对 48k 原声（13 句均值）的合计偏差，越小越好：
#
#   v2     0dB → 6.04 | 1dB → 6.35 | 2dB → 7.35 | 3dB → 8.51   ⇒ **0 最优**
#   v2Pro  0dB → 1.60 | 1dB → 1.12 | 2dB → 2.11 | 3dB → 3.12   ⇒ **1 最优**
#
# v2 未补偿时 8–12k 就**偏亮 4.2 dB**（−32.5 vs 目标 −36.7），
# 而它的 12–16k 已对齐 —— 所以补偿只会让 12–16k 过冲，**总量更差**。
# v2Pro 的两个频段都接近原声，只需极少的 1 dB。
#
# ## ⚠️ 短句与长文本的偏好**相反**（已知口径冲突）
#
# 短句的 12–16k 缺口更大（同一句单测 −47.7，而 421 秒全文 −46.5，
# 段间过渡抬高了长文本的高频）。所以短句扫描的最优是 3 dB。
#
# **这里以长文本为准** —— 实际用途是长朗读。短句场景下 0/1 与 3 的差异
# 在整体听感中占比很小；短句用户可自行调高。
BY_VERSION = {
    "v1": 0.0,
    "v2": 0.0,      # 长文本最优；它的问题是 8–12k 偏亮，补偿解决不了
    "v3": 0.0,
    "v4": 0.0,
    "v2Pro": 1.0,   # 长文本最优
    "v2ProPlus": 1.0,
}


def gain_for_version(version, default=None):
    """按模型版本取校准补偿量（dB）。

    未知版本返回 `default`（默认 0 = 关闭）——
    宁可不动，也不要凭猜测往输出里加高频。
    """
    fallback = DEFAULT_GAIN_DB if default is None else float(default)
    if not version:
        return fallback
    return BY_VERSION.get(str(version), fallback)


def shelf_coeffs(sr: float, fc: float, gain_db: float, q: float):
    """RBJ audio-EQ-cookbook 高通搁架双二阶系数 → (b, a)。

    b/a 均为长度 3 的 float64 数组，a[0] 已归一为 1。
    """
    if sr <= 0:
        raise ValueError(f"采样率必须为正: {sr}")
    if fc <= 0 or fc >= sr / 2:
        raise ValueError(f"截止频率须在 (0, sr/2) 内: fc={fc}, sr={sr}")
    if q <= 0:
        raise ValueError(f"Q 必须为正: {q}")

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * fc / sr
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    sqA = math.sqrt(A)

    b0 = A * ((A + 1) + (A - 1) * cw + 2 * sqA * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * sqA * alpha)
    a0 = (A + 1) - (A - 1) * cw + 2 * sqA * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * sqA * alpha

    return (np.array([b0, b1, b2], dtype=np.float64) / a0,
            np.array([a0, a1, a2], dtype=np.float64) / a0)


def _lfilter_numpy(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """纯 numpy 直接 II 型转置实现（scipy 不可用时的回退）。

    逐样本循环，仅用于没有 scipy 的环境。32 kHz 下 1 秒音频约 0.1 s。
    """
    n = len(x)
    y = np.empty(n, dtype=np.float64)
    z1 = z2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for i in range(n):
        xi = float(x[i])
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        y[i] = yi
    return y


def _lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """优先用 scipy（快），否则回退纯 numpy。"""
    try:
        from scipy.signal import lfilter  # type: ignore
        return lfilter(b, a, x)
    except Exception:  # noqa: BLE001  —— 没装 scipy / 导入异常都走回退
        return _lfilter_numpy(b, a, x)


def apply_brightness(audio, sr: int, gain_db: float = DEFAULT_GAIN_DB,
                     fc: float = DEFAULT_FC, q: float = DEFAULT_Q):
    """对音频施加高通搁架。返回与入参**同 dtype** 的数组。

    gain_db=0 时**原样返回入参**（逐字节不变），不做任何滤波 ——
    这保证「默认关闭」是真正的零风险。

    支持 int16（torch 后端）与 float32/float64（ONNX 后端）输入。
    """
    if gain_db == 0.0:
        return audio                      # 短路：零改动

    arr = np.asarray(audio)
    if arr.size == 0:
        return audio
    if arr.ndim > 1:                       # 多声道：逐声道处理
        chans = [apply_brightness(arr[:, c], sr, gain_db, fc, q)
                 for c in range(arr.shape[1])]
        return np.stack(chans, axis=1).astype(arr.dtype, copy=False)

    is_int = np.issubdtype(arr.dtype, np.integer)
    if is_int:
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        x = arr.astype(np.float64) / scale
    else:
        scale = None
        x = arr.astype(np.float64)

    b, a = shelf_coeffs(sr, fc, gain_db, q)
    y = _lfilter(b, a, x)

    if is_int:
        y = np.clip(y, -1.0, 1.0) * scale
        return np.rint(y).astype(arr.dtype)
    return y.astype(arr.dtype, copy=False)


def describe(gain_db: float = DEFAULT_GAIN_DB, fc: float = DEFAULT_FC,
             q: float = DEFAULT_Q) -> str:
    """一行人类可读描述，用于日志与体检报告。"""
    if gain_db == 0.0:
        return "亮度补偿: 关闭 (0 dB)"
    return f"亮度补偿: +{gain_db:g} dB @ {fc:g} Hz (Q={q:g})"
