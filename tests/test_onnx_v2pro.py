# -*- coding: utf-8 -*-
"""ONNX 后端的 v2Pro 支持与 Kaldi fbank 回归测试

覆盖两个真实缺口：
  1. ONNX 后端原先**只认 v2**（vits 只有 3 个输入），遇到 v2Pro 的包会报
     `Required inputs (['sv_emb']) are missing from input feed`。
  2. v2Pro 的说话人编码器 `sv_after_fbank.onnx` 需要 **fbank 在 Python 侧算**
     （上游 `Kaldi.fbank` 用 `torch.fft.rfft`，该算子无法导出到 ONNX）。

fbank 的正确性用「与 torch 实现对比」来验（torch 可用时）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice.backends.onnx_backend import (  # noqa: E402
    _kaldi_fbank, _mel_filterbank)

GSV = r"D:\A\voiceclone\GPT-SoVITS"
HAS_GSV = os.path.isdir(os.path.join(GSV, "GPT_SoVITS", "eres2net"))


# ---------- mel 滤波器组 ----------

def test_mel_filterbank_shape():
    """列数是 n_fft//2（**不是** n_fft//2+1）

    上游 `get_mel_banks` 用 `num_fft_bins = window_length_padded / 2`，
    比 rfft 的输出少一个频点。这是它与 librosa 实现的差异之一，
    用的时候必须 `spec[..., :n_fft//2]` 对齐，否则维度对不上。
    """
    fb = _mel_filterbank(16000, 512, 80)
    assert fb.shape == (80, 256), "80 mel × (512/2) 频点"


def test_mel_filterbank_nonnegative():
    fb = _mel_filterbank(16000, 512, 80)
    assert (fb >= 0).all(), "三角滤波器不应有负值"


def test_mel_filterbank_rows_have_energy():
    """每个滤波器至少要覆盖一个频点，否则该通道恒为 0（信息丢失）"""
    fb = _mel_filterbank(16000, 512, 80)
    assert (fb.sum(axis=1) > 0).all()


# ---------- fbank ----------

def test_fbank_shape_and_finiteness():
    x = np.random.default_rng(0).normal(0, 0.1, 16000 * 3).astype(np.float32)
    fb = _kaldi_fbank(x, n_mels=80)
    # 25ms 窗 / 10ms 跳：(48000 - 400)/160 + 1 = 298
    assert fb.shape == (298, 80)
    assert np.isfinite(fb).all(), "log 前有 max(,1e-10)，不该出现 inf/nan"


def test_fbank_too_short_returns_empty():
    """短于一个窗的输入应返回空，而不是崩"""
    fb = _kaldi_fbank(np.zeros(100, dtype=np.float32), n_mels=80)
    assert fb.shape == (0, 80)


def test_fbank_deterministic():
    """同一输入必须给同一输出（它内部有滤波器组缓存，不能因此串味）"""
    x = np.random.default_rng(1).normal(0, 0.1, 16000 * 2).astype(np.float32)
    a = _kaldi_fbank(x, 80)
    b = _kaldi_fbank(x, 80)
    assert np.array_equal(a, b)


def test_fbank_dc_offset_removed():
    """加了直流偏置后，fbank 结果应基本不变（Kaldi 默认 remove_dc_offset）"""
    x = np.random.default_rng(2).normal(0, 0.1, 16000).astype(np.float32)
    a = _kaldi_fbank(x, 80)
    b = _kaldi_fbank(x + 5.0, 80)
    assert np.allclose(a, b, atol=1e-3), "直流分量应被去掉"


@pytest.mark.skipif(not HAS_GSV, reason="需要 GPT-SoVITS 检出")
def test_fbank_matches_kaldi():
    """与上游 Kaldi.fbank 对比 —— **必须高度一致**

    这是导出时数值正确的前提：ONNX 侧的说话人向量靠这个 fbank 算，
    偏一点就会让音色漂移。

    踩过的坑（相关系数 0.37 → 1.000000）：
      · 漏了 preemphasis(0.97)  → 0.37
      · mel 滤波器做了 Slaney 归一化（上游**不归一化**）→ 0.98
      · num_fft_bins 多算一个频点
    """
    torch = pytest.importorskip("torch")
    sys.path[:0] = [GSV, os.path.join(GSV, "GPT_SoVITS")]
    prev = os.getcwd()
    os.chdir(GSV)
    try:
        from GPT_SoVITS.eres2net import kaldi as Kaldi
    except Exception as e:  # noqa: BLE001
        os.chdir(prev)
        pytest.skip(f"无法导入上游 kaldi: {e}")
    try:
        x = np.random.default_rng(3).normal(0, 0.1, 16000 * 2).astype(np.float32)
        ref = Kaldi.fbank(torch.from_numpy(x)[None, :], num_mel_bins=80,
                          sample_frequency=16000, dither=0).numpy()
        got = _kaldi_fbank(x, 80)
        assert ref.shape == got.shape, f"{ref.shape} vs {got.shape}"
        corr = np.corrcoef(ref.ravel(), got.ravel())[0, 1]
        assert corr > 0.9999, f"与 Kaldi 相关系数只有 {corr:.6f}，说明有参数没对齐"
        maxerr = np.abs(ref - got).max()
        assert maxerr < 1e-3, f"最大绝对差 {maxerr:.2e} 偏大"
    finally:
        os.chdir(prev)


# ---------- v2Pro 的 sv_emb 契约 ----------

def test_sv_emb_contract():
    """sv_emb 必须是 [1, 20480] —— 这是 models_onnx 里
    `sv_emb = nn.Linear(20480, gin_channels)` 定下的契约"""
    from zfh_voice.backends.onnx_backend import OnnxBackend
    # 不实际加载模型，只验判定逻辑的字段名存在
    assert hasattr(OnnxBackend, "_sv_embedding")
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "zfh_voice", "backends", "onnx_backend.py"), encoding="utf-8").read()
    assert "20480" in src, "sv_emb 的 20480 维契约应写在注释里"
    assert "sv_after_fbank.onnx" in src
    assert "aten::fft_rfft" in src, "应说明为什么 fbank 要在 Python 侧算"


# ---------- 体检要能发现缺 sv_after_fbank ----------

def _mini_onnx(path, inputs):
    """造一个只有图结构的小 onnx（用于测 needs_sv 判定，不需要真权重）"""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    outs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])]
    ins = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1])
           for n in inputs]
    node = helper.make_node("Identity", [inputs[0]], ["y"])
    g = helper.make_graph([node], "t", ins, outs)
    m = helper.make_model(g, opset_imports=[
        helper.make_opsetid("", 17)])
    onnx.save(m, path)
    return path


def test_onnx_needs_sv_detects_v2pro(tmp_path):
    """有 sv_emb 输入 -> 判定为 v2Pro"""
    from zfh_voice import paths
    p = _mini_onnx(str(tmp_path / "v.onnx"),
                   ["text_seq", "pred_semantic", "ref_audio", "sv_emb"])
    assert paths.onnx_needs_sv(p) is True


def test_onnx_needs_sv_false_for_v2(tmp_path):
    """v2 的 vits 只有 3 个输入 -> 不判成 v2Pro"""
    from zfh_voice import paths
    p = _mini_onnx(str(tmp_path / "v.onnx"),
                   ["text_seq", "pred_semantic", "ref_audio"])
    assert paths.onnx_needs_sv(p) is False


def test_onnx_needs_sv_survives_broken_file(tmp_path):
    """坏文件不该让体检崩 —— 返回 False 让它去报别的错"""
    from zfh_voice import paths
    p = tmp_path / "broken.onnx"
    p.write_bytes(b"not an onnx at all")
    assert paths.onnx_needs_sv(str(p)) is False


def test_onnx_needs_sv_missing_file(tmp_path):
    from zfh_voice import paths
    assert paths.onnx_needs_sv(str(tmp_path / "nope.onnx")) is False


def test_model_status_flags_missing_sv(tmp_path):
    """v2Pro 的包缺 sv_after_fbank 时，体检必须报出来

    真事故场景：用户只下了 7 个 ONNX（v2 的清单），漏了 v2Pro 多的那个说话人
    编码器。若体检说"就绪"，用户要到合成时才见到后端报错 —— 自相矛盾。
    """
    from zfh_voice import paths
    d = str(tmp_path)
    # 造齐 ONNX_FILES + AUX_FILES
    for f in list(paths.ONNX_FILES) + list(paths.AUX_FILES):
        fp = os.path.join(d, f.replace("/", os.sep))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        if f == "zfh_vits.onnx":
            _mini_onnx(fp, ["text_seq", "pred_semantic", "ref_audio", "sv_emb"])
        else:
            with open(fp, "wb") as fh:
                fh.write(b"\0")
    _, miss, ok = paths.model_status(d, backend="onnx")
    assert ok is False, "缺说话人编码器却报就绪"
    assert any("sv_after_fbank" in m for m in miss), f"应指出缺 sv，实际 {miss}"


def test_model_status_ok_when_sv_present(tmp_path):
    """补上 sv_after_fbank 后应就绪"""
    from zfh_voice import paths
    d = str(tmp_path)
    for f in list(paths.ONNX_FILES) + list(paths.AUX_FILES):
        fp = os.path.join(d, f.replace("/", os.sep))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        if f == "zfh_vits.onnx":
            _mini_onnx(fp, ["text_seq", "pred_semantic", "ref_audio", "sv_emb"])
        else:
            with open(fp, "wb") as fh:
                fh.write(b"\0")
    with open(os.path.join(d, "sv_after_fbank.onnx"), "wb") as fh:
        fh.write(b"\0")
    _, miss, ok = paths.model_status(d, backend="onnx")
    assert ok is True, f"应就绪，却报缺 {miss}"


def test_v2_package_does_not_require_sv(tmp_path):
    """v2 的包（vits 无 sv_emb）**不该**要求说话人编码器"""
    from zfh_voice import paths
    d = str(tmp_path)
    for f in list(paths.ONNX_FILES) + list(paths.AUX_FILES):
        fp = os.path.join(d, f.replace("/", os.sep))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        if f == "zfh_vits.onnx":
            _mini_onnx(fp, ["text_seq", "pred_semantic", "ref_audio"])
        else:
            with open(fp, "wb") as fh:
                fh.write(b"\0")
    _, miss, ok = paths.model_status(d, backend="onnx")
    assert ok is True, f"v2 包不该要求 sv，却报缺 {miss}"
