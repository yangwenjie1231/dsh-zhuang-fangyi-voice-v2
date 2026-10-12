# -*- coding: utf-8 -*-
"""冒烟测试：验证模型就绪 + 能合成出非静音音频

运行：
    pytest -q                     # 需先下载模型
    或直接 python tests/test_smoke.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice2 import model_status, resolve_model_dir  # noqa: E402
from zfh_voice2.paths import ONNX_FILES  # noqa: E402


def _models_ready():
    """ONNX 模型是否就绪。

    ## ⚠️ v0.2.0 修正：必须显式 `require="onnx"`

    本文件后面的断言是 `for f in ONNX_FILES: assert exists(...)`，
    所以守卫就该问"ONNX 齐了吗"。

    v0.1.0 时这里写的是 `resolve_model_dir()`（`require=None`），
    当时能工作**只是因为巧合**：机器上没有任何模型目录，
    所以它必抛 `FileNotFoundError` -> 全部 skip。

    v0.2.0 加了**路径自动探测**后，`require=None` 会找到
    "只有 torch 权重、没有 ONNX"的目录（如 `zfh-models-v2pro`）
    并**正常返回** -> 守卫误判为"就绪" -> 断言 ONNX 时失败。

    **API 语义没变**（`require=None` = "有能推理的内容就认"），
    变的是这个**不精确的代理**不再成立了。改成精确提问即可。
    """
    try:
        resolve_model_dir(require="onnx")
        return True
    except FileNotFoundError:
        return False


requires_models = pytest.mark.skipif(
    not _models_ready(), reason="模型未下载，先运行 python download_models.py")


def test_model_status_reports_all_files():
    d, missing, ok = model_status()
    assert isinstance(d, str)
    if not ok:
        # 未下载模型时，缺失项应覆盖到 ONNX 文件
        assert set(missing) & set(ONNX_FILES)


@requires_models
def test_resolve_model_dir():
    d = resolve_model_dir()
    assert os.path.isdir(d)
    for f in ONNX_FILES:
        assert os.path.exists(os.path.join(d, f)), f


@requires_models
def test_synth_short_text():
    from zfh_voice2 import TTS

    tts = TTS()
    try:
        r = tts.say("你好。", use_cache=False)
        assert r.sr in (32000, 24000, 44100, 48000)
        assert r.duration > 0.3
        assert np.abs(r.wav).max() > 0.01, "输出近似静音"
    finally:
        tts.close()


@requires_models
def test_cache_hit_is_fast(tmp_path):
    from zfh_voice2 import TTS

    # 用独立缓存目录，避免被上一次运行的残留缓存干扰
    # （否则第一次 say 就直接命中，断言 not r1.cached 会假失败）
    tts = TTS(cache_dir=str(tmp_path / "cache"))
    try:
        text = "缓存测试：这句话应该被缓存。"
        r1 = tts.say(text, use_cache=True)
        r2 = tts.say(text, use_cache=True)
        assert not r1.cached
        assert r2.cached
        assert r1.duration == pytest.approx(r2.duration, abs=1e-6)
    finally:
        tts.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
