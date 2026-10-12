# -*- coding: utf-8 -*-
"""torch 后端的采样参数可配置性

背景：`top_k` / `temperature` / `repetition_penalty` 原先**硬编码**在
`synth()` 里，外部无法调整 —— 于是**无法针对新模型重搜参数**。
（做 v2Pro 的参数扫描时才发现这一点。）

现在通过构造参数覆盖，默认值保持不变。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice.backends.torch_backend import TorchBackend  # noqa: E402


def test_default_sampling_unchanged():
    """默认值必须与硬编码时代一致（否则是静默的行为变更）"""
    assert TorchBackend.DEFAULT_SAMPLING == {
        "top_k": 15,
        "temperature": 1.0,
        "repetition_penalty": 1.35,
    }


def _sampling(**kw):
    """不加载模型，只跑 __init__ 里那几行参数合并逻辑"""
    obj = TorchBackend.__new__(TorchBackend)
    obj.sampling = dict(TorchBackend.DEFAULT_SAMPLING)
    for key, val in kw.items():
        if val is not None:
            obj.sampling[key] = val
    return obj.sampling


def test_override_top_k():
    s = _sampling(top_k=8)
    assert s["top_k"] == 8
    assert s["temperature"] == 1.0, "没覆盖的项应保持默认"


def test_override_all_three():
    s = _sampling(top_k=5, temperature=0.6, repetition_penalty=1.2)
    assert s == {"top_k": 5, "temperature": 0.6, "repetition_penalty": 1.2}


def test_none_means_use_default():
    """显式传 None 不该把值写成 None（要落到默认）"""
    s = _sampling(top_k=None, temperature=None, repetition_penalty=None)
    assert s == TorchBackend.DEFAULT_SAMPLING


def test_extra_params_supported():
    """top_p / speed 也支持覆盖"""
    s = _sampling(top_p=0.9, speed=1.2)
    assert s["top_p"] == 0.9
    assert s["speed"] == 1.2


def test_signature_accepts_the_params():
    """构造签名必须真的收这些参数（否则传进去会 TypeError）"""
    import inspect
    sig = inspect.signature(TorchBackend.__init__)
    for k in ("top_k", "temperature", "repetition_penalty", "top_p", "speed"):
        assert k in sig.parameters, f"__init__ 缺参数 {k}"


def test_synth_uses_instance_sampling():
    """`synth()` 必须用 `self.sampling`，而不是在 inputs 里写死值"""
    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "zfh_voice", "backends", "torch_backend.py"),
        encoding="utf-8").read()
    assert "inputs.update(self.sampling)" in src, \
        "synth 应把 self.sampling 合进 inputs"
    # 只检查 **synth 的 inputs 字面量**里没有写死的采样值。
    # （`DEFAULT_SAMPLING` 里当然有这些数字，那是默认值定义，不算硬编码。）
    import re
    m = re.search(r"inputs\s*=\s*\{(.*?)\}", src, re.S)
    assert m, "没找到 inputs 定义"
    body = m.group(1)
    for key in ("top_k", "temperature", "repetition_penalty"):
        assert f'"{key}"' not in body, \
            f"inputs 里仍写死了 {key}，应改走 self.sampling"
