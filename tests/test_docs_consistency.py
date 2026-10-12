# -*- coding: utf-8 -*-
"""文档与代码一致性守卫

背景（真事故）：`brightnessDb` 的默认值从 `0` 改成 `'auto'` 之后，
`docs/安装提示词.md` 里还写着「默认 `0` = 关闭」—— **文档与代码矛盾**。
用户照着文档去理解行为就会错。

这类不一致很容易发生（改代码时想不到文档），所以用测试守：
**文档里写的默认值必须与代码里的真值一致**。

只检查"能机械核对的断言"，不检查行文措辞。
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


# ---------- 亮度补偿的默认值 ----------

def test_docs_do_not_claim_brightness_default_is_zero():
    """文档不该再说「brightnessDb 默认 0」——真值是 'auto'

    ⚠️ 正则要**精确**：文档里合法地会出现「默认 'auto'（也可填 0~12）」
    这种句子，粗暴地找 "默认 0" 会误报。所以只在
    「默认」与 `0` 之间**没有 auto** 时才判失败。
    """
    for rel in ("README.md", "docs/安装提示词.md"):
        txt = _read(rel)
        for line in txt.splitlines():
            if "brightnessDb" not in line:
                continue
            # 同一行里声明默认 0，且没有提 auto -> 矛盾
            m = re.search(r"默认[^\n]{0,12}`?0`?", line)
            if m and "auto" not in line:
                pytest.fail(f"{rel} 仍称 brightnessDb 默认 0：{line.strip()!r}")


def test_docs_mention_auto():
    """两份文档都要说明 auto（否则用户不知道默认按版本自动选）"""
    for rel in ("README.md", "docs/安装提示词.md"):
        txt = _read(rel)
        assert "auto" in txt, f"{rel} 没提 auto"


def test_postprocess_module_docstring_not_stale():
    """`postprocess` 模块头不该再写「默认关闭」"""
    txt = _read(os.path.join("src", "zfh_voice", "postprocess.py"))
    head = txt.split('"""')[1] if '"""' in txt else txt[:2000]
    assert "默认关闭" not in head, \
        "模块头还写着「默认关闭」，但默认已改成 auto（按版本选）"


# ---------- 代码里的真值 ----------

def test_gain_default_matches_docs_claim():
    """文档里写的 auto 取值必须与代码真值一致

    真实值见 `postprocess.BY_VERSION`（以长文本校准）：
    v2 → 0、v2Pro → 1。
    """
    from zfh_voice import postprocess as pp
    assert pp.gain_for_version("v2") == 0.0
    assert pp.gain_for_version("v2Pro") == 1.0
    # 文档必须提到这两个版本，且不能把 v2Pro 说成 0
    for rel in ("README.md", "docs/安装提示词.md"):
        txt = _read(rel)
        assert "v2Pro" in txt, f"{rel} 应说明 v2Pro 的取值"


# ---------- ONNX 清单 ----------

def test_docs_list_sv_after_fbank():
    """v2Pro 的 ONNX 包多一个文件 —— 两份文档都该提到

    否则用户照文档下模型会漏掉它，到合成时才失败。
    """
    for rel in ("README.md", "docs/安装提示词.md"):
        txt = _read(rel)
        assert "sv_after_fbank.onnx" in txt, \
            f"{rel} 没提 v2Pro 需要的 sv_after_fbank.onnx"


# ---------- torch 权重命名 ----------

def test_docs_say_weight_names_are_flexible():
    """文档该说明 torch 权重不锁死文件名（否则重训用户以为不能用）"""
    for rel in ("README.md", "docs/安装提示词.md"):
        txt = _read(rel)
        assert re.search(r"-e<N>|_e<N>_s<N>|任意名", txt), \
            f"{rel} 没说清 torch 权重支持自定义命名"


# ---------- 种子 ----------

def test_docs_seed_default_is_42():
    from zfh_voice.api import DEFAULT_SEED
    assert DEFAULT_SEED == 42
    for rel in ("README.md", "docs/安装提示词.md"):
        txt = _read(rel)
        assert "42" in txt, f"{rel} 应写明默认种子 42"
