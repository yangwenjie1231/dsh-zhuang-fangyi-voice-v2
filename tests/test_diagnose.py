# -*- coding: utf-8 -*-
"""诊断脚本的核心判据：削顶、权重完整性、v5 头。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("diag", ROOT / "tools/diagnose_voice.py")
diag = importlib.util.module_from_spec(spec)
sys.modules["diag"] = diag
spec.loader.exec_module(diag)


def levels():
    return {r["title"]: r["level"] for r in diag.results}


def test_clipped_audio_is_flagged_as_fail(tmp_path):
    diag.results.clear()
    p = tmp_path / "clipped.wav"
    sf.write(p, np.clip(np.sin(np.linspace(0, 200, 48000)) * 10, -1, 1), 48000)
    diag.audio_health(p, "测试音频")
    assert any("削顶" in t and lv == "FAIL" for t, lv in levels().items())


def test_normal_audio_passes(tmp_path):
    diag.results.clear()
    t = np.linspace(0, 2, 96000)
    x = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / "ok.wav"
    sf.write(p, x, 48000)
    diag.audio_health(p, "测试音频")
    assert all(lv != "FAIL" for lv in levels().values())


def test_missing_weight_file_fails(tmp_path):
    diag.results.clear()
    diag.weight_health(tmp_path / "nope.pth")
    assert any(lv == "FAIL" for lv in levels().values())
