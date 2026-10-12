# -*- coding: utf-8 -*-
"""常驻开关（空闲释放）的回归测试

覆盖曾经的真 bug：空闲看门狗在**合成进行中**把模型卸掉 ——
长合成耗时远超 idle_timeout，期间 last_activity 不更新，
看门狗误判为空闲，导致模型在推理过程中被回收。
修法：在途请求计数 inflight，有请求在跑就跳过本轮。

用假 TTS，不加载真实模型，毫秒级完成。
"""
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice import server as srv  # noqa: E402


class FakeResult:
    duration = 1.0
    sr = 32000
    cached = False

    def to_wav_bytes(self):
        return b"RIFF0000WAVE"


class FakeTTS:
    """可观测加载状态的假后端"""

    _backend_kind = "fake"

    def __init__(self):
        self.loaded = True
        self.ref_wav = "ref.wav"
        self.unload_calls = 0
        self.say_calls = 0

    @property
    def is_loaded(self):
        return self.loaded

    def unload(self):
        if self.loaded:
            self.loaded = False
            self.unload_calls += 1
            return True
        return False

    def preload(self):
        self.loaded = True

    def say(self, text, seed=None, use_cache=True):
        self.loaded = True
        self.say_calls += 1
        return FakeResult()

    def clear_cache(self):
        return 0


def _run_watchdog(tts, state, idle, seconds):
    th = threading.Thread(target=srv._idle_watchdog,
                          args=(tts, state, idle, False), daemon=True)
    th.start()
    time.sleep(seconds)
    state["stop"] = True
    th.join(timeout=3)


def test_watchdog_unloads_when_idle():
    tts = FakeTTS()
    state = {"last_activity": time.time(), "unloads": 0, "inflight": 0}
    _run_watchdog(tts, state, idle=1, seconds=3.0)
    assert tts.loaded is False, "空闲超时应释放模型"
    assert state["unloads"] >= 1


def test_watchdog_skips_while_busy():
    """核心回归：有在途请求时绝不能卸载"""
    tts = FakeTTS()
    state = {"last_activity": time.time() - 60,   # 早已"空闲"
             "unloads": 0, "inflight": 1}          # 但正在合成
    _run_watchdog(tts, state, idle=1, seconds=3.0)
    assert tts.loaded is True, "有在途请求时不应卸载模型"
    assert state["unloads"] == 0


def test_watchdog_noop_when_already_unloaded():
    tts = FakeTTS()
    tts.loaded = False
    state = {"last_activity": time.time() - 60, "unloads": 0, "inflight": 0}
    _run_watchdog(tts, state, idle=1, seconds=2.5)
    assert state["unloads"] == 0
    assert tts.unload_calls == 0


def test_handler_state_defaults():
    """make_handler 应初始化计数，且 inflight 默认为 0"""
    tts = FakeTTS()
    srv.make_handler(tts)
    state = {}
    H = srv.make_handler(tts, state)
    assert state["inflight"] == 0
    assert "last_activity" in state
    assert H is not None


def test_tts_unload_then_reload(tmp_path, monkeypatch):
    """TTS.unload 后应能按需重新加载（用假后端替换 make_backend）"""
    from zfh_voice import api

    created = {"n": 0}

    def fake_make_backend(kind, **kw):
        created["n"] += 1
        return FakeTTS()

    monkeypatch.setattr(api, "make_backend", fake_make_backend)
    monkeypatch.setattr(api.paths, "resolve_model_dir",
                        lambda *a, **k: str(tmp_path))
    ref = tmp_path / "ref"
    ref.mkdir()
    (ref / "default.wav").write_bytes(b"\0")
    (ref / "default.txt").write_text("参考文本。", encoding="utf-8")

    t = api.TTS()
    assert t.is_loaded is False          # 懒加载
    t.preload()
    assert t.is_loaded is True
    assert created["n"] == 1
    assert t.unload() is True
    assert t.is_loaded is False
    _ = t.backend                        # 再次访问应重建
    assert t.is_loaded is True
    assert created["n"] == 2, "卸载后应重新创建后端"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
