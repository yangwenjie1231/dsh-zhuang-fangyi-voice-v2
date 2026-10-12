# -*- coding: utf-8 -*-
"""后端抽象与共享常量（GPT-SoVITS v2）"""
import os

import numpy as np

# ---- v2 模型常量 ----
VERSION = "v2"
EOS = 1024
HZ = 50                    # 语义 token 帧率
MAX_SEC = 54               # 单句最长秒数
EARLY_STOP = HZ * MAX_SEC  # 提前终止阈值
MAX_LOOP = 1500            # 自回归最大步数
SAMPLE_RATE = 32000        # 输出采样率

# 参考音频限制（与上游一致）
REF_MIN_SEC = 3.0
REF_MAX_SEC = 10.0

# 上游怪癖：HuBERT 输入（16k）后面追加「按 32k 计算」的 0.3 秒静音
# 即 9600 个样本。必须精确复刻，否则音色会偏移。
ZERO_PAD_SAMPLES = int(SAMPLE_RATE * 0.3)


class BackendError(RuntimeError):
    pass


class SynthBackend:
    """推理后端接口"""

    name = "base"

    # 实际加载的模型版本（子类在初始化时填）。
    #
    # 为什么必须暴露它：**亮度补偿的校准值随版本而变**（见 postprocess 模块头）——
    # v2 需要 +3dB 才对齐原声，而 v2Pro 的频谱本来就接近原声，加 3dB 会**过冲**。
    # API 层据此自动选补偿量，用户就不必记住"换模型要改数字"。
    model_version = VERSION

    def __init__(self, model_dir=None, **kw):
        self.model_dir = model_dir

    def synth(self, text, ref_wav, ref_text, **params):
        """返回 (wav: np.float32 mono, sr: int)"""
        raise NotImplementedError

    def close(self):
        pass

    # ---- 共用工具 ----
    @staticmethod
    def check_ref_duration(seconds):
        if not (REF_MIN_SEC <= seconds <= REF_MAX_SEC):
            raise ValueError(
                f"参考音频时长 {seconds:.2f}s 超出 {REF_MIN_SEC}~{REF_MAX_SEC}s 范围。"
                "请换一段干净的单句语音。")

    @staticmethod
    def zero_pad_for_hubert(wav16k):
        """复刻上游的静音追加行为"""
        return np.concatenate(
            [wav16k.astype(np.float32),
             np.zeros(ZERO_PAD_SAMPLES, dtype=np.float32)])
