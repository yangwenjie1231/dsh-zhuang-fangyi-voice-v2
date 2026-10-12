# -*- coding: utf-8 -*-
"""中英混排（LangSegmenter 分段）的回归测试

背景：中文文本前端对英文是 `re.sub("[a-zA-Z]+", "", seg)` —— **静默删除**。
上游靠 `LangSegmenter` 按语言分段解决：英文段走 `en` 音素、中文段走 `zh`。
本项目此前直接调 `clean_text(text, "zh")`，绕过了分段，所以英文全丢。

本测试钉死分段行为与关键契约：
  · 英文段必须产出音素（不是被删）
  · 中文段的 word2ph 必须存在且 sum(word2ph) == 音素数
  · 非中文段的 word2ph 必须是 None（上游契约，调用方据此给零 BERT）
  · 分段不可用时优雅降级，不抛
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice import frontend, paths  # noqa: E402


def _ready():
    """前端是否可用（需要模型目录里的 G2PW/BERT 辅助文件）"""
    try:
        frontend.prepare()
        return True
    except Exception:
        return False


# ⚠️ 这个 skip 只该盖住"**真的需要模型**"的那些用例。
# 最初写成模块级 `pytestmark`，结果把下面那组**纯参数解析**的回归用例
# 也一起跳过了（15 skipped）—— 而那组恰恰是不依赖模型、最该一直跑的部分。
# 现在改成只给需要的用例加标记。
needs_frontend = pytest.mark.skipif(
    not _ready(), reason="文本前端不可用（缺模型目录或辅助文件）")


@needs_frontend
def test_pure_chinese_single_segment():
    segs = frontend.text_to_segments("今天天气不错。", "zh")
    assert len(segs) == 1
    lang, phones, w2p, norm = segs[0]
    assert lang == "zh"
    assert len(phones) == 13
    assert w2p is not None and sum(w2p) == len(phones)


@needs_frontend
def test_pure_english_gets_phonemes():
    """★ 核心回归：纯英文必须产出音素（旧路径下是 0 音素）"""
    segs = frontend.text_to_segments("Hello world, this is a test.", "zh")
    assert len(segs) == 1
    lang, phones, w2p, _ = segs[0]
    assert lang == "en"
    assert len(phones) > 0, "英文段必须有音素，否则合成会失败"
    assert w2p is None, "非中文段的 word2ph 应为 None（上游契约）"


@needs_frontend
def test_mixed_splits_by_language():
    segs = frontend.text_to_segments("GPU 和 AI 都很重要。", "zh")
    langs = [s[0] for s in segs]
    assert "en" in langs and "zh" in langs, f"应中英分段，实际 {langs}"
    # 英文段有音素，中文段有 word2ph
    for lang, phones, w2p, _ in segs:
        assert len(phones) > 0
        if lang == "zh":
            assert w2p is not None and sum(w2p) == len(phones)
        else:
            assert w2p is None


@needs_frontend
def test_mixed_english_not_deleted():
    """★ 加了英文后音素数必须**变多**（旧路径下不变 → 英文被删）"""
    a = sum(len(p) for _, p, _, _ in frontend.text_to_segments("今天天气不错。", "zh"))
    b = sum(len(p) for _, p, _, _
            in frontend.text_to_segments("今天天气不错，Hello World。", "zh"))
    assert b > a, f"英文应产生额外音素：{a} → {b}"


@needs_frontend
def test_english_phonemes_are_arpabet():
    """英文段应是 ARPAbet 音素（HH/AH0/OW1…），不是中文拼音"""
    segs = frontend.text_to_segments("Hello", "zh")
    phones = segs[0][1]
    assert any(p in ("HH", "AH0", "L", "OW1") for p in phones), phones


@needs_frontend
def test_mixed_available_flag_is_bool():
    assert isinstance(frontend.mixed_available(), bool)


# ── 下面这组钉死一个**静默降级** bug（2026-10 实测踩到）─────────────────────
#
# 症状：中英混排**永远**走中文读法兜底（"Hello World" → "哈喽 达布流欧"），
# 不报任何错。根因有两层，都在"参数没传全"上：
#   1. `paths.g2pw_dir/bert_dir` 原先只看模型目录，而 torch 用户的
#      G2PW/BERT 在 **GPT-SoVITS 检出**里 → prepare() 抛 FileNotFoundError；
#   2. `api._auto_localize()` 里 `mixed_available()` 调用**没传 model_dir**
#      → prepare() 去找默认模型目录（仓库 models/ 或 ~/.cache）→ 同样抛错。
# 两处都被 `except Exception: self._localize_auto = True` 吞掉。
#
# 所以这里断言的是**参数解析**而不是"能不能出声"：只要 gsv_root/model_dir
# 给对了，就必须能定位到 G2PW 与 BERT。

def _find_gsv_root():
    """找一个真实的 GPT-SoVITS 检出（测试环境里可能没有）"""
    for root in paths._gsv_roots():
        if os.path.isdir(os.path.join(root, "GPT_SoVITS", "text", "G2PWModel")):
            return root
    return None


def test_g2pw_dir_falls_back_to_gsv_root(tmp_path):
    """`g2pw_dir` 必须能去 GPT-SoVITS 检出里找（而不是只看模型目录）。

    反向用例：给一个**空的** model_dir + 真实 gsv_root → 仍要找到。
    这正是修复前会失败的那条路径。
    """
    gsv = _find_gsv_root()
    if gsv is None:
        pytest.skip("本机没有 GPT-SoVITS 检出")
    got = paths.g2pw_dir(str(tmp_path), gsv)
    assert os.path.isdir(got), f"应在检出里找到 G2PW，实际 {got}"
    assert "G2PWModel" in got


def test_bert_dir_falls_back_to_gsv_root(tmp_path):
    gsv = _find_gsv_root()
    if gsv is None:
        pytest.skip("本机没有 GPT-SoVITS 检出")
    got = paths.bert_dir(str(tmp_path), gsv)
    assert os.path.isdir(got), f"应在检出里找到 BERT，实际 {got}"


def test_model_dir_wins_over_gsv_root(tmp_path):
    """模型目录里**有**就优先用它（onnx 那套就是这么装的）。"""
    mdir = tmp_path / "models"
    (mdir / "G2PWModel").mkdir(parents=True)
    got = paths.g2pw_dir(str(mdir), r"D:\不存在的检出")
    assert got == os.path.join(str(mdir), "G2PWModel")


def test_explicit_env_wins(tmp_path, monkeypatch):
    """`ZFH_G2PW_DIR` 是用户显式指定 → 优先级最高。"""
    explicit = tmp_path / "my-g2pw"
    explicit.mkdir()
    monkeypatch.setenv("ZFH_G2PW_DIR", str(explicit))
    got = paths.g2pw_dir(str(tmp_path / "models"), None)
    assert got == str(explicit)


def test_mixed_available_accepts_both_args():
    """`mixed_available` 必须同时接受 model_dir 与 gsv_root。

    这是那个静默降级的**直接回归**：只传 gsv_root 时，内部
    `resolve_model_dir(None)` 会去找默认模型目录并抛错（实测），
    于是被判成"混排不可用"。签名里少一个参数就会旧病复发。
    """
    import inspect
    sig = inspect.signature(frontend.mixed_available)
    assert "gsv_root" in sig.parameters
    assert "model_dir" in sig.parameters, (
        "mixed_available 必须接受 model_dir —— 否则 prepare() 会去找默认"
        "模型目录并抛错，中英混排被静默降级成中文读法")


def test_auto_localize_passes_model_dir(monkeypatch):
    """`api._auto_localize` 必须把 model_dir **和** gsv_root 都传给 mixed_available。

    用假 TTS 对象直接验参数传递 —— 不依赖本机有没有真模型。
    """
    from zfh_voice import api

    seen = {}

    def fake_mixed(gsv_root=None, model_dir=None):
        seen["gsv_root"] = gsv_root
        seen["model_dir"] = model_dir
        return True

    monkeypatch.setattr(frontend, "mixed_available", fake_mixed)
    t = api.TTS.__new__(api.TTS)          # 不跑 __init__（那要真模型）
    t._localize_auto = None
    t._localize_reason = None
    t._gsv_root = r"D:\gsv"
    t.model_dir = r"D:\models"
    assert t._auto_localize() is False    # 可用 → 不转换
    assert seen["gsv_root"] == r"D:\gsv"
    assert seen["model_dir"] == r"D:\models", (
        "model_dir 没传下去 → prepare() 会去找默认目录而抛错 → 静默降级")


@needs_frontend
def test_split_falls_back_when_unavailable(monkeypatch):
    """分段不可用时降级为单段，不抛"""
    monkeypatch.setattr(frontend, "_lang_segment", None)
    monkeypatch.setattr(frontend, "_ready", True)
    segs = frontend.split_by_lang("GPU 和 AI", "zh")
    assert segs == [("zh", "GPU 和 AI")]


@needs_frontend
def test_segments_reject_empty():
    """空文本返回 []（不产生空段）"""
    assert frontend.text_to_segments("", "zh") == []
    assert frontend.text_to_segments("   ", "zh") == []
    assert frontend.split_by_lang("", "zh") == []
    # 注意：纯标点**不是**空 —— 上游会把 "。。。" 归一化成一个句号音素
    # （相当于一个停顿），这是合理行为，不该被当成空文本拦掉。
    segs = frontend.text_to_segments("。。。", "zh")
    assert len(segs) == 1 and segs[0][1] == ["."]


@needs_frontend
def test_unknown_language_mapped_to_default():
    """分段器给出未支持语言时，应回落到默认语言而不是崩"""
    def fake(text, default):
        return [{"lang": "fr", "text": "bonjour"}, {"lang": "zh", "text": "你好"}]
    import types
    orig = frontend._lang_segment
    frontend._lang_segment = fake
    try:
        segs = frontend.split_by_lang("x", "zh")
        assert segs[0][0] == "zh", f"未知语言应回落 zh，实际 {segs[0][0]}"
        assert segs[1][0] == "zh"
    finally:
        frontend._lang_segment = orig


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
