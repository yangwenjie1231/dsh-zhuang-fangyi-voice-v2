# -*- coding: utf-8 -*-
"""torch 后端：复用 GPT-SoVITS 原版实现

优点：GPU 下最快（实测 RTF 0.45，ONNX 约 6.1），支持 fp16 等全部特性
缺点：需要 torch 环境 + 一份 GPT-SoVITS 源码检出，体积大

用法：
    OnnxBackend(...)                                    # 便携
    TorchBackend(gsv_root=r"D:\\GPT-SoVITS", ...)        # 快

gsv_root 需包含 GPT_SoVITS/ 目录（可从官方仓库 clone）。
权重文件放在 <gsv_root>/GPT_weights_v2 与 SoVITS_weights_v2，
或通过 gpt_path / sovits_path 显式指定。
"""
import os
import sys
import threading

# os.chdir 是进程级状态：推理期间要切到 gsv_root（上游用相对路径），
# 所以合成必须串行 —— 否则并发调用会互相把 cwd 搅乱。
_SYNTH_LOCK = threading.Lock()

import numpy as np

from .base import BackendError, SynthBackend


class TorchBackend(SynthBackend):
    name = "torch"

    @staticmethod
    def _pick_weight(d, preferred_name, pattern):
        """在 d 下挑一个权重：先认首选文件名，再按模式扫。

        多个候选时取**文件名排序最大**的（GPT-SoVITS 的轮次/步数后缀
        是递增的，所以字典序最大 ≈ 训练最久的那一个）。
        """
        import re
        if not d or not os.path.isdir(d):
            return os.path.join(d, preferred_name)
        pref = os.path.join(d, preferred_name)
        if os.path.exists(pref):
            return pref
        try:
            names = sorted(n for n in os.listdir(d) if re.search(pattern, n))
        except OSError:
            names = []
        return os.path.join(d, names[-1]) if names else pref

    # 推理采样参数的默认值。
    #
    # 这些直接决定 AR 解码行为（对「念错」和「糊」都有影响）：
    #   · top_k 小 → 只从高概率 token 采样，更稳；大 → 更多样
    #   · temperature 低 → 更确定；高 → 更多样但可能失控
    #   · repetition_penalty 高 → 抑制重复，但过高会让发音含糊
    #
    # 默认值沿用上游/历史上搜得的组合。**想做参数扫描时通过构造参数覆盖** ——
    # 早先这几个值是硬编码的，外部无法调整，导致无法针对新模型重搜。
    DEFAULT_SAMPLING = {
        "top_k": 15,
        "temperature": 1.0,
        "repetition_penalty": 1.35,
    }

    def __init__(self, gsv_root, model_dir=None, gpt_path=None,
                 sovits_path=None, ref_wav=None, ref_text=None,
                 device="cuda", is_half=True, exp_name="zfh",
                 bert_dir=None, hubert_dir=None,
                 top_k=None, temperature=None, repetition_penalty=None,
                 top_p=None, speed=None):
        super().__init__(model_dir)
        # 采样参数：没传就用默认值（保持既有行为不变）
        self.sampling = dict(self.DEFAULT_SAMPLING)
        for key, val in (("top_k", top_k), ("temperature", temperature),
                         ("repetition_penalty", repetition_penalty),
                         ("top_p", top_p), ("speed", speed)):
            if val is not None:
                self.sampling[key] = val
        # 解析模型目录（用于兜底查找 torch_weights/；找不到也不影响，
        # 因为 torch 权重通常直接从 gsv_root 里取）
        if self.model_dir:
            from .. import paths
            try:
                self.model_dir = paths.resolve_model_dir(self.model_dir)
            except FileNotFoundError:
                self.model_dir = os.path.abspath(self.model_dir)
        self.gsv_root = os.path.abspath(gsv_root)
        if not os.path.isdir(os.path.join(self.gsv_root, "GPT_SoVITS")):
            raise BackendError(
                f"gsv_root 下找不到 GPT_SoVITS/ 目录: {self.gsv_root}\n"
                "请 clone https://github.com/RVC-Boss/GPT-SoVITS 并指定其根目录。")

        sys.path[:0] = [self.gsv_root, os.path.join(self.gsv_root, "GPT_SoVITS")]
        prev_cwd = os.getcwd()
        os.chdir(self.gsv_root)          # 上游大量使用相对路径
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS as _TTS
        finally:
            os.chdir(prev_cwd)

        # 权重从哪来？优先级（高→低）：
        #   1. 显式 gpt_path / sovits_path
        #   2. **<model_dir>/torch_weights/** —— 用户明确指定的模型目录
        #   3. <gsv_root>/GPT_weights_v2|SoVITS_weights_v2/ —— 上游默认位置
        #
        # ⚠️ 两条踩过的坑：
        #
        # ① 不能只按固定文件名找。用户重训/换版本后名字会变
        #    （如 `zfh_v2pro_e12_s384.pth`），所以先按原名找，
        #    再按"像 GPT / 像 SoVITS"的模式扫一遍。
        #
        # ② **model_dir 的权重必须优先于 gsv_root 的默认路径。**
        #    早先写成"gsv_root 路径不存在时才回退到 model_dir"，
        #    结果 gsv_root 里恰好有旧的 v2 权重时，**新模型被静默忽略** ——
        #    合成照常出声、听起来"也还行"，但根本不是指定的那个模型。
        #    这种静默错误极难发现，必须让显式指定的目录赢。
        tw = os.path.join(self.model_dir, "torch_weights") if self.model_dir else None
        gpt = gpt_path or (self._pick_weight(tw, f"{exp_name}-e4.ckpt",
                                             r"-e\d+\.ckpt$") if tw else None) \
            or os.path.join(self.gsv_root, "GPT_weights_v2", f"{exp_name}-e4.ckpt")
        sov = sovits_path or (self._pick_weight(tw, f"{exp_name}_e6_s186.pth",
                                                r"_e\d+_s\d+\.pth$") if tw else None) \
            or os.path.join(self.gsv_root, "SoVITS_weights_v2", f"{exp_name}_e6_s186.pth")
        for p in (gpt, sov):
            if not os.path.exists(p):
                raise BackendError(
                    f"找不到 torch 权重: {p}\n"
                    "torch 后端需要 .ckpt/.pth —— **ONNX 文件无法被 torch 加载**。\n"
                    "下载方式：python download_models.py --with-torch\n"
                    "或用 gpt_path / sovits_path 显式指定路径。")

        bert = bert_dir or os.path.join(
            self.gsv_root, "GPT_SoVITS", "pretrained_models",
            "chinese-roberta-wwm-ext-large")
        hubert = hubert_dir or os.path.join(
            self.gsv_root, "GPT_SoVITS", "pretrained_models",
            "chinese-hubert-base")

        # 上游用相对路径，这里换成绝对路径副本
        cfg = {"custom": {
            "device": device, "is_half": is_half, "version": "v2",
            "t2s_weights_path": gpt, "vits_weights_path": sov,
            "bert_base_path": bert, "cnhuhbert_base_path": hubert,
        }}
        prev_cwd = os.getcwd()
        os.chdir(self.gsv_root)
        try:
            self.tts = _TTS(cfg)
        finally:
            os.chdir(prev_cwd)

        self.ref_wav = ref_wav
        self.ref_text = ref_text
        self.device = device
        # 上游会从权重文件头探测版本（`get_sovits_version_from_path_fast`），
        # 这里读回来暴露给 API 层 —— 它据此选亮度补偿量（v2Pro 不需要补偿）。
        try:
            self.model_version = str(self.tts.configs.version)
        except Exception:  # noqa: BLE001
            self.model_version = "v2"

    def synth(self, text, ref_wav, ref_text, seed=None, verbose=False):
        import numpy as np
        ref_wav = ref_wav or self.ref_wav
        ref_text = ref_text or self.ref_text
        if not (ref_wav and ref_text):
            raise BackendError("torch 后端需要指定参考音频与参考文本")

        inputs = {
            "text": text, "text_lang": "zh",
            "ref_audio_path": ref_wav, "prompt_text": ref_text,
            "prompt_lang": "zh",
        }
        inputs.update(self.sampling)      # top_k / temperature / repetition_penalty
        if seed is not None:
            inputs["seed"] = int(seed)
        sr, wav = None, None
        # ⚠️ **推理期间也必须 chdir 到 gsv_root** —— 这不是洁癖，是上游的硬依赖：
        # GPT-SoVITS 内部大量使用**相对路径**找文本前端资源
        #（`GPT_SoVITS/text/G2PWModel/`、pretrained_models 等）。
        # 只在上面的 `__init__` 里 chdir 而这里不切，症状非常隐蔽：
        # 它会去**联网下载** g2pw 模型（打印 "Downloading g2pw model..."），
        # 下载失败就静默卡住 —— 看起来像"模型没装"，其实文件就在 gsv_root 里。
        # 实测：cwd 在 GSV 根目录 → 3.5 秒出音频；cwd 在别处 → 卡在下载。
        #
        # `os.chdir` 是**进程级**状态，所以必须串行（HTTP 服务那边也有 lock，
        # 这里再加一道，防止别的调用方并发进来把 cwd 搅乱）。
        with _SYNTH_LOCK:
            prev_cwd = os.getcwd()
            try:
                os.chdir(self.gsv_root)
                for _sr, _chunk in self.tts.run(inputs):   # run() 是生成器
                    sr, wav = _sr, _chunk
            finally:
                os.chdir(prev_cwd)
        if wav is None:
            raise BackendError("合成失败：未返回音频")
        wav = np.asarray(wav)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
            if np.abs(wav).max() > 2:              # int16
                wav = wav / 32768.0
        return wav.reshape(-1), sr

    def close(self):
        try:
            del self.tts
            import torch
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass
