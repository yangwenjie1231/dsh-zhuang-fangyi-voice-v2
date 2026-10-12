# -*- coding: utf-8 -*-
"""ONNX 后端：仅依赖 onnxruntime，不需要 torch / CUDA

推理链路（全部为 ONNX）：
  BERT(ONNX)   文本 → 语义特征
  HuBERT(ONNX) 参考音频 → SSL 内容特征
  encoder      两者 + 音素 → x, prompts
  fsdec        首段解码
  sdec         自回归逐 token 解码（循环 MAX_LOOP 次）
  vits         语义 token + 参考音频 → 波形

精度可切换：fp32 / fp16 / int8（同目录下的三套模型文件命名一致）
"""
import os
import time

import numpy as np

from .. import audio, frontend, paths
from .base import (EARLY_STOP, EOS, MAX_LOOP, SAMPLE_RATE, VERSION,
                   BackendError, SynthBackend)

# Kaldi mel 滤波器组缓存（构造一次约 5ms，但每句都要用）
_FB_CACHE = {}


def _mel_filterbank(sr=16000, n_fft=512, n_mels=80, low_freq=20.0, high_freq=None):
    """Kaldi 风格 mel 滤波器组 —— **逐行对齐上游 `eres2net/kaldi.py:get_mel_banks`**

    与常见的 librosa/Slaney 实现有两点关键差异（踩过才知道）：

      1. **不做 Slaney 面积归一化**。上游直接用三角斜率 `min(up, down)` 并
         clamp 到 0。加了 `2/(f_hi-f_lo)` 的归一化会让高频通道整体偏低
         （实测逐维均值差从 −3dB 递增到 −5.5dB）。
      2. `num_fft_bins = n_fft // 2`（**不是 n_fft//2 + 1**）—— 上游用的是
         `window_length_padded / 2`，少一个频点。所以返回的列数是 `n_fft//2`，
         用的时候要配合 `spec[..., :n_fft//2]`。

    mel 公式用 Kaldi 的 `1127 * ln(1 + f/700)`。
    """
    if high_freq is None or high_freq <= 0.0:
        high_freq = 0.5 * sr

    num_fft_bins = n_fft // 2
    fft_bin_width = sr / n_fft

    mel_low = 1127.0 * np.log(1.0 + low_freq / 700.0)
    mel_high = 1127.0 * np.log(1.0 + high_freq / 700.0)
    delta = (mel_high - mel_low) / (n_mels + 1)

    bin_idx = np.arange(n_mels)[:, None]                       # (n_mels, 1)
    left = mel_low + bin_idx * delta
    center = mel_low + (bin_idx + 1.0) * delta
    right = mel_low + (bin_idx + 2.0) * delta

    # 各 FFT 频点对应的 mel 值（注意只有 num_fft_bins 个）
    mel = 1127.0 * np.log(1.0 + (fft_bin_width * np.arange(num_fft_bins)) / 700.0)[None, :]

    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    bins = np.maximum(0.0, np.minimum(up, down))
    return bins.astype(np.float32)


def _kaldi_fbank(x16k, n_mels=80, frame_length_ms=25.0, frame_shift_ms=10.0,
                 sample_frequency=16000.0, low_freq=20.0, preemphasis=0.97):
    """Kaldi 兼容的 fbank（**逐项对齐上游 `eres2net/kaldi.py` 的默认参数**）

    为什么要自己实现：上游 `ERes2NetV2` 的 fbank 依赖 `torch.fft.rfft`，
    而该算子**无法导出到 ONNX**，所以 ONNX 侧只能在 Python 里算。

    对齐的默认值（任何一个错了都会让说话人向量偏移）：
      · `snip_edges=True`      —— 帧数 = 1 + (N - frame_len) // shift，不做边界补零
      · `preemphasis_coefficient=0.97` —— **必须做预加重**（曾漏掉，导致与 Kaldi
        相关系数只有 0.37）
      · `window_type=povey`    —— hann^0.85
      · `remove_dc_offset=True`（每帧去均值）、`raw_energy=True`
      · `round_to_power_of_two=True` —— FFT 长度取 2 的幂（400 → 512）
      · `use_power=True`（功率谱）、`use_log_fbank=True`
      · `low_freq=20`、`htk_compat=False`（不做 HTK 的维度换位）
    """
    key = (n_mels, frame_length_ms, frame_shift_ms, sample_frequency, low_freq)
    cached = _FB_CACHE.get(key)
    if cached is None:
        frame_len = int(sample_frequency * frame_length_ms / 1000.0)     # 400
        frame_shift = int(sample_frequency * frame_shift_ms / 1000.0)    # 160
        n_fft = 1
        while n_fft < frame_len:          # round_to_power_of_two
            n_fft *= 2
        cached = _FB_CACHE[key] = (
            frame_len, frame_shift, n_fft,
            _mel_filterbank(sample_frequency, n_fft, n_mels, low_freq))
    frame_len, frame_shift, n_fft, fb = cached

    x = np.asarray(x16k, dtype=np.float64).reshape(-1)
    num_samples = len(x)

    # snip_edges=True：只取完整帧
    if num_samples < frame_len:
        return np.zeros((0, n_mels), dtype=np.float32)
    m = 1 + (num_samples - frame_len) // frame_shift
    idx = np.arange(frame_len)[None, :] + frame_shift * np.arange(m)[:, None]
    frames = x[idx]

    if True:      # remove_dc_offset
        frames = frames - frames.mean(axis=1, keepdims=True)

    # preemphasis：y[n] = x[n] - k * x[n-1]（首样本用 x[0] 自身）
    if preemphasis != 0.0:
        shifted = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
        frames = frames - preemphasis * shifted

    # povey 窗 = hann^0.85（periodic=False）
    w = np.hanning(frame_len) ** 0.85
    frames = frames * w[None, :]

    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2        # use_power=True
    # 滤波器组的列数是 n_fft//2（上游 `window_length_padded / 2`，少一个频点）
    mel = spec[:, : fb.shape[1]] @ fb.T
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


class OnnxBackend(SynthBackend):
    name = "onnx"

    def __init__(self, model_dir=None, providers=None, threads=None,
                 log_level=3):
        super().__init__(model_dir)
        import onnxruntime as ort
        self.ort = ort
        self.model_dir = paths.resolve_model_dir(model_dir)

        if providers is None:
            # 默认：优先 DirectML（Windows 任意显卡），其次 CPU
            avail = ort.get_available_providers()
            if "DmlExecutionProvider" in avail:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.providers = providers

        so = ort.SessionOptions()
        so.log_severity_level = log_level
        if threads:
            so.intra_op_num_threads = int(threads)

        t0 = time.time()
        self.sess = {}
        for key, fn in [
            ("encoder", "zfh_t2s_encoder.onnx"),
            ("fsdec", "zfh_t2s_fsdec.onnx"),
            ("sdec", "zfh_t2s_sdec.onnx"),
            ("vits", "zfh_vits.onnx"),
            ("bert", "bert_encoder.onnx"),
            ("hubert", "hubert_encoder.onnx"),
        ]:
            p = os.path.join(self.model_dir, fn)
            if not os.path.exists(p):
                raise BackendError(f"缺少模型文件: {p}")
            self.sess[key] = ort.InferenceSession(p, so, providers=providers)

        # v2Pro 支持：它比 v2 多一个说话人向量（sv_emb）。
        # 判定方式是看 `zfh_vits.onnx` 是否声明了该输入 —— 比猜文件名可靠。
        vits_inputs = {i.name for i in self.sess["vits"].get_inputs()}
        self.needs_sv = "sv_emb" in vits_inputs
        self.has_sv_model = os.path.exists(
            os.path.join(self.model_dir, "sv_after_fbank.onnx"))
        if self.needs_sv:
            if not self.has_sv_model:
                raise BackendError(
                    "这个 ONNX 包是 v2Pro 的（zfh_vits.onnx 需要 `sv_emb` 输入），"
                    "但缺少说话人编码器 sv_after_fbank.onnx。\n"
                    "请下载完整的 v2Pro ONNX 包，或改用 v2 的包。")
            self.sess["sv"] = ort.InferenceSession(
                os.path.join(self.model_dir, "sv_after_fbank.onnx"),
                so, providers=providers)
        # 供 API 层选亮度补偿量（v2Pro 不需要补偿，加了会过冲）
        self.model_version = "v2Pro" if self.needs_sv else "v2"
        self.load_seconds = time.time() - t0

        # BERT tokenizer（只需 tokenizer 文件，不需要权重）
        from transformers import AutoTokenizer
        tok_dir = paths.bert_dir(self.model_dir, getattr(self, "gsv_root", None))
        self.tokenizer = AutoTokenizer.from_pretrained(tok_dir)

        # 文本前端（会设置 ZFH_G2PW_DIR / ZFH_BERT_DIR）
        # gsv_root 一并传：onnx 用户通常把资源放在模型目录，但有人两个后端
        # 混着装，这时检出一份也应当能用（`paths` 按优先级找）。
        frontend.prepare(self.model_dir, gsv_root=getattr(self, "gsv_root", None))

    # ---------------- 子步骤 ----------------
    def _text_ids(self, text):
        """文本 → (ids, word2ph, norm_text, bert)

        **bert 形状必须是 [text_length, 1024]** —— 这是导出的 ONNX 契约
        （`zfh_t2s_encoder.onnx` 的 `text_bert` 输入是 ['text_length', 1024]）。
        上游 torch 侧内部是 [1024, T]，导出时已转置，这里不要照搬上游布局。

        中英混排按语言分段：中文段走 G2PW 并用真 BERT 特征；
        英文段走 g2p_en（ARPAbet 音素）且 BERT 给**全零** ——
        与上游 `TextPreprocessor.get_bert_inf` 一致（BERT 只对中文有意义，
        英文段的 word2ph 本来就是 None）。
        """
        segs = frontend.text_to_segments(text, "zh", VERSION)
        if not segs:
            return (np.zeros((1, 0), dtype=np.int64), None, "",
                    np.zeros((0, 1024), dtype=np.float32))

        ids_all, bert_parts, norm_all, w2p_all = [], [], [], []
        single_lang = len(segs) == 1
        for lang, phones, word2ph, norm in segs:
            ids_all.extend(frontend.cleaned_ids(phones, VERSION))
            norm_all.append(norm)
            if lang == "zh" and word2ph is not None:
                bert_parts.append(self._bert_feature(norm, word2ph))   # [T,1024]
                w2p_all.extend(word2ph)
            else:
                # 非中文：BERT 全零，行数与音素数一致
                bert_parts.append(np.zeros((len(phones), 1024), dtype=np.float32))
                single_lang = False

        bert = np.concatenate(bert_parts, axis=0) if bert_parts else \
            np.zeros((0, 1024), dtype=np.float32)
        w2p = w2p_all if (single_lang and w2p_all) else None
        return (np.array([ids_all], dtype=np.int64), w2p, "".join(norm_all), bert)

    def _bert_feature(self, norm_text, word2ph):
        """中文 BERT 特征 → [T, 1024]（T = sum(word2ph) = 音素数）"""
        enc = self.tokenizer(norm_text, return_tensors="np")
        ids = enc["input_ids"].astype(np.int64)
        feed = {
            "input_ids": ids,
            "attention_mask": enc["attention_mask"].astype(np.int64),
            "token_type_ids": enc.get(
                "token_type_ids", np.zeros_like(ids)).astype(np.int64),
        }
        hidden = self.sess["bert"].run(None, feed)[0]      # [1, seq, 1024]
        res = hidden[0][1:-1]                              # 去掉首尾特殊 token
        reps = [np.tile(res[i], (word2ph[i], 1)) for i in range(len(word2ph))]
        return np.concatenate(reps, axis=0).astype(np.float32)     # [T, 1024]

    def _ssl_content(self, ref_wav):
        wav, sr = audio.load_wav(ref_wav)
        wav16k = audio.resample(wav, sr, 16000)
        self.check_ref_duration(len(wav) / sr)
        padded = self.zero_pad_for_hubert(wav16k)
        out = self.sess["hubert"].run(
            None, {"wav": padded[None, :].astype(np.float32)})[0]
        return out.astype(np.float32)                      # [1, 768, T]

    def _ref_audio_32k(self, ref_wav):
        wav, sr = audio.load_wav(ref_wav)
        return audio.resample(wav, sr, SAMPLE_RATE)[None, :].astype(np.float32)

    def _sv_embedding(self, ref_wav):
        """v2Pro 说话人向量 [1, 20480]

        ⚠️ fbank **必须在 Python 侧算**：上游 ERes2NetV2 的 fbank 用了
        `torch.fft.rfft`，而 PyTorch 的 ONNX 导出器不支持 `aten::fft_rfft`，
        所以导出的 `sv_after_fbank.onnx` 只包含 fbank **之后**的网络。

        fbank 参数照抄上游 `Kaldi.fbank(num_mel_bins=80, sample_frequency=16000,
        dither=0)`：25ms 窗 / 10ms 跳、去均值、povey 窗、log。
        """
        wav, sr = audio.load_wav(ref_wav)
        wav16k = audio.resample(wav, sr, 16000)
        fb = _kaldi_fbank(wav16k, n_mels=80)          # [T, 80]
        # 导出时输入是 [B, C, F, T]，所以要转置
        fb = fb.T[None, None, :, :].astype(np.float32)
        return self.sess["sv"].run(None, {"fbank": fb})[0]      # [1, 20480]

    # ---------------- 主流程 ----------------
    def synth(self, text, ref_wav, ref_text, seed=None, verbose=False):
        text_seq, text_w2p, text_norm, text_bert = self._text_ids(text)
        ref_seq, ref_w2p, ref_norm, ref_bert = self._text_ids(ref_text)
        ssl = self._ssl_content(ref_wav)
        ref32 = self._ref_audio_32k(ref_wav)
        sv = self._sv_embedding(ref_wav) if self.needs_sv else None

        # 1) encoder
        x, prompts = self.sess["encoder"].run(None, {
            "ref_seq": ref_seq, "text_seq": text_seq,
            "ref_bert": ref_bert, "text_bert": text_bert,
            "ssl_content": ssl})

        # 2) 首段解码
        y, k, v, y_emb, x_example = self.sess["fsdec"].run(
            None, {"x": x, "prompts": prompts})
        prefix_len = prompts.shape[1]

        # 3) 自回归解码
        t0 = time.time()
        n_steps = 0
        for n_steps in range(1, MAX_LOOP):
            y, k, v, y_emb, logits, samples = self.sess["sdec"].run(None, {
                "iy": y, "ik": k, "iv": v,
                "iy_emb": y_emb, "ix_example": x_example})
            if EARLY_STOP != -1 and (y.shape[1] - prefix_len) > EARLY_STOP:
                break
            if (int(np.argmax(logits, axis=-1).reshape(-1)[0]) == EOS
                    or int(np.asarray(samples).reshape(-1)[0]) == EOS):
                break
        y[0, -1] = 0
        pred_semantic = y[:, -n_steps:][None, ...]         # [1,1,T]
        ar_seconds = time.time() - t0

        # 4) 声码器
        vits_feed = {"text_seq": text_seq, "pred_semantic": pred_semantic,
                     "ref_audio": ref32}
        if self.needs_sv:
            vits_feed["sv_emb"] = sv
        wav = self.sess["vits"].run(None, vits_feed)[0]

        if verbose:
            print(f"    [onnx] AR {n_steps} 步 {ar_seconds:.1f}s  "
                  f"总 {ar_seconds:.1f}s")
        return np.asarray(wav).reshape(-1).astype(np.float32), SAMPLE_RATE
