# -*- coding: utf-8 -*-
"""默认种子与可复现性测试

背景：GPT-SoVITS 的 AR 解码在 seed=None 时每次采样不同，而**句子越短波动越大**
（实测 4 字句时长 CV 23.5%，24 字句 10.3%）。曾观测 `让我想想` 在随机种子下
产出 0.74s（正常 1.22s）—— 近乎截断。

修复：`TTS.default_seed` 默认 42，`say(seed=None)` 时用它兜底。
"""
import numpy as np
import pytest

from zfh_voice import postprocess
from zfh_voice.api import DEFAULT_SEED


class FakeBackend:
    """记录每次调用的 seed，并返回确定性波形（与 seed 相关）。"""

    def __init__(self):
        self.calls = []

    def synth(self, text, ref_wav, ref_text, seed=None, verbose=False):
        self.calls.append(seed)
        n = 3200
        rng = np.random.default_rng(0 if seed is None else int(seed))
        return rng.normal(0, 0.05, n).astype(np.float32), 32000

    def close(self):
        pass


def _tts(tmp_path, monkeypatch, **kw):
    """构造一个不碰真实模型的 TTS：绕过文件校验，直接注入假后端。"""
    from zfh_voice import api
    import os

    tmp_path = str(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    ref = os.path.join(tmp_path, "ref.wav")
    import soundfile as sf
    sf.write(ref, np.zeros(16000, dtype=np.float32), 16000)

    tts = api.TTS.__new__(api.TTS)
    tts._backend_kind = "onnx"
    tts.model_dir = tmp_path
    tts.use_cache = True
    tts.default_seed = kw.pop("default_seed", DEFAULT_SEED)
    # 补偿量是**两段式**的：`_brightness_cfg` 是配置（可以是 "auto"），
    # `brightness_db` 是解析后的数值。这里模拟"已解析"的状态。
    br = float(kw.pop("brightness_db", 0.0))
    tts._brightness_cfg = br
    tts.brightness_db = br
    tts._brightness_version = "v2"
    tts.localize = False
    tts._localize_auto = False
    tts._localize_reason = "test"
    tts._gsv_root = None
    tts._backend_kw = {}
    tts.ref_wav = ref
    tts.ref_text = "测试"
    tts.cache_dir = os.path.join(tmp_path, "cache")
    os.makedirs(tts.cache_dir, exist_ok=True)
    tts._backend = FakeBackend()
    return tts


def test_default_seed_is_42():
    assert DEFAULT_SEED == 42


def test_none_seed_uses_default(tmp_path):
    """seed=None 必须落到 default_seed，而不是随机"""
    tts = _tts(tmp_path, None)
    tts.say("让我想想", seed=None, use_cache=False)
    assert tts._backend.calls == [42]


def test_explicit_seed_wins(tmp_path):
    tts = _tts(tmp_path, None)
    tts.say("让我想想", seed=7, use_cache=False)
    assert tts._backend.calls == [7]


def test_default_seed_none_allows_random(tmp_path):
    """显式把 default_seed 设为 None 时，才允许透传 None（旧行为）"""
    tts = _tts(tmp_path, None, default_seed=None)
    tts.say("让我想想", seed=None, use_cache=False)
    assert tts._backend.calls == [None]


def test_same_text_twice_is_identical(tmp_path):
    """同一文本两次合成必须逐字节相同（可复现性）"""
    tts = _tts(tmp_path, None)
    r1 = tts.say("让我想想", use_cache=False)
    r2 = tts.say("让我想想", use_cache=False)
    assert np.array_equal(r1.wav, r2.wav)
    assert r1.duration == r2.duration


def test_cache_key_includes_brightness(tmp_path):
    """补偿参数必须进缓存键，否则改参数会命中旧缓存"""
    tts = _tts(tmp_path, None)
    k0 = tts._cache_key("你好", 42, 0.0)
    k3 = tts._cache_key("你好", 42, 3.0)
    assert k0 != k3


def test_cache_key_includes_seed(tmp_path):
    tts = _tts(tmp_path, None)
    assert tts._cache_key("你好", 42, 0.0) != tts._cache_key("你好", 7, 0.0)


def test_brightness_zero_does_not_touch_wav(tmp_path):
    """brightness_db=0（默认）时输出必须与不加补偿完全一致"""
    tts = _tts(tmp_path, None, brightness_db=0.0)
    r = tts.say("你好", use_cache=False)
    tts2 = _tts(tmp_path / "b", None, brightness_db=0.0)
    r2 = tts2.say("你好", use_cache=False)
    assert np.array_equal(r.wav, r2.wav)


def test_brightness_applied_when_nonzero(tmp_path):
    """非零补偿确实改变输出，且 12-16k 被抬高"""
    tts0 = _tts(tmp_path / "a", None, brightness_db=0.0)
    tts3 = _tts(tmp_path / "b", None, brightness_db=3.0)
    r0 = tts0.say("你好世界这是一句测试", use_cache=False)
    r3 = tts3.say("你好世界这是一句测试", use_cache=False)

    def band(x, sr, lo, hi):
        X = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64) * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / sr)
        m = (f >= lo) & (f < hi)
        return 10 * np.log10(X[m].mean() + 1e-20)

    d = band(r3.wav, r3.sr, 12000, 16000) - band(r0.wav, r0.sr, 12000, 16000)
    assert d > 1.0, f"补偿后 12-16k 只变化 {d:.2f} dB"


def test_say_many_forwards_brightness(tmp_path):
    tts = _tts(tmp_path, None, brightness_db=0.0)
    rs = tts.say_many(["你好", "世界"], str(tmp_path / "out"),
                      brightness_db=3.0, verbose=False)
    assert len(rs) == 2
    for r in rs:
        assert r.wav is not None


# ---------- 调用点守卫 ----------
#
# 下面这几条守的是「参数有没有真的从上层流到底层」。
# 背景：`buildServeCommand` / `TTS.say` 本身工作正常、函数级单测全绿，
# 但**调用点忘了传新参数** —— 配置到不了命令行、CLI 到不了服务端。
# 只测函数抓不到这种漏，必须测调用链。

def test_remote_say_forwards_brightness(monkeypatch):
    """CLI 走常驻服务时必须把 brightness_db 放进请求体（曾漏传）"""
    from zfh_voice import cli

    captured = {}

    def fake_post(server, path, payload=None, timeout=600):
        captured["path"] = path
        captured["payload"] = payload
        return b"RIFFfake"

    monkeypatch.setattr(cli, "_server_post", fake_post)
    cli._remote_say("http://x", "你好", 42, None, brightness_db=3.0)
    assert captured["payload"]["brightness_db"] == 3.0
    assert captured["payload"]["seed"] == 42


def test_remote_say_default_brightness_is_omitted(monkeypatch):
    """不指定亮度时**不应**在请求体里带该字段

    让服务端用它自己的默认值（同样是 auto，按它加载的模型版本选）。
    显式传 0 才表示"我要关掉" —— 两者语义不同，不能混。
    """
    from zfh_voice import cli

    captured = {}
    monkeypatch.setattr(
        cli, "_server_post",
        lambda s, p, payload=None, timeout=600: (captured.update(payload), b"RIFF")[1])
    cli._remote_say("http://x", "你好", 42, None)
    assert "brightness_db" not in captured, \
        f"未指定时不该带 brightness_db，实际: {captured}"


def test_remote_say_explicit_zero_is_sent(monkeypatch):
    """显式传 0 必须发出去（表示"关掉"，与"未指定"不同）"""
    from zfh_voice import cli

    captured = {}
    monkeypatch.setattr(
        cli, "_server_post",
        lambda s, p, payload=None, timeout=600: (captured.update(payload), b"RIFF")[1])
    cli._remote_say("http://x", "你好", 42, None, brightness_db=0.0)
    assert captured.get("brightness_db") == 0.0


def test_server_forwards_brightness_to_tts():
    """服务端收到 brightness_db 后必须转给 tts.say（曾漏传）"""
    from zfh_voice import server as srv

    class FakeTTS:
        is_loaded = True
        cache_dir = "."
        default_seed = 42
        brightness_db = 0.0

        def __init__(self):
            self.seen = {}

        def say(self, text, seed=None, use_cache=True, brightness_db=None):
            self.seen = dict(text=text, seed=seed, brightness_db=brightness_db)

            class R:
                wav = np.zeros(320, dtype=np.float32)
                sr = 32000
                duration = 0.01
                cached = False
                spoken = text

                def to_wav_bytes(self):
                    return b"RIFF"

            return R()

        def unload(self):
            return False

    tts = FakeTTS()
    handler = srv.make_handler(tts, state={"last_activity": 0})
    inst = handler.__new__(handler)

    class FakeBody:
        def read(self, n=-1):
            return b'{"text":"\xe4\xbd\xa0\xe5\xa5\xbd","brightness_db":3.0}'

    inst.headers = {"Content-Length": "99"}
    inst.rfile = FakeBody()
    sent = {}
    inst._send = lambda code, body, ctype=None: sent.update(code=code, body=body)
    inst._do_tts(binary=False)
    assert tts.seen.get("brightness_db") == 3.0, \
        f"brightness_db 没传到 tts.say: {tts.seen}"


def test_server_default_brightness_is_none():
    """不传 brightness_db 时应是 None（让 TTS 用实例默认值），不是 0 覆盖"""
    from zfh_voice import server as srv

    class FakeTTS:
        is_loaded = True
        cache_dir = "."
        default_seed = 42
        brightness_db = 0.0

        def __init__(self):
            self.seen = {}

        def say(self, text, seed=None, use_cache=True, brightness_db=None):
            self.seen = dict(brightness_db=brightness_db)

            class R:
                wav = np.zeros(320, dtype=np.float32)
                sr = 32000
                duration = 0.01
                cached = False
                spoken = text

                def to_wav_bytes(self):
                    return b"RIFF"

            return R()

        def unload(self):
            return False

    tts = FakeTTS()
    handler = srv.make_handler(tts, state={"last_activity": 0})
    inst = handler.__new__(handler)

    class FakeBody:
        def read(self, n=-1):
            return b'{"text":"\xe4\xbd\xa0\xe5\xa5\xbd"}'

    inst.headers = {"Content-Length": "99"}
    inst.rfile = FakeBody()
    inst._send = lambda code, body, ctype=None: None
    inst._do_tts(binary=False)
    assert tts.seen.get("brightness_db") is None
