# -*- coding: utf-8 -*-
"""音色风格（语气）注册表 —— v2 新增。

## 为什么需要
v1 只支持**一个**参考音频（`ref/default.wav`），语气固定。
实际使用中发现：**同一模型 + 不同参考音频 = 不同语气风格**：

    normal  正常说话      ref/default.wav
    firm    语气强硬      剧情_e6m4_026（陈述句结尾，F0 尾部 -4.13）
    gentle  语音娇        剧情_e6m3_016（陈述句结尾，F0 尾部 -4.86）

三种都经**盲听确认可用**，区别在语气而非质量。

## 设计原则（v1 兼容）
    · **默认风格 = normal**，其参考音频就是 v1 的 `ref/default.wav`
      -> 不指定风格时，行为与 v1 **完全一致**
    · 风格是**推理时参数**，不需要重训模型
    · 新增 `style` 参数（HTTP body / CLI / config），**全部可选**

## 与 `paths.AUX_FILES` 的关系
v1 的 `AUX_FILES` 里硬编码了 `ref/default.wav`。
v2 改为**按风格解析**：`ref/styles/<name>.wav`，
但**仍保留** `ref/default.wav` 作为 normal 的兼容路径。

## 用法
    from zfh_voice.styles import STYLES, resolve_style, list_styles

    resolve_style(None)        -> ("normal", Path(".../ref/default.wav"))
    resolve_style("firm")      -> ("firm",   Path(".../ref/styles/firm.wav"))
    list_styles()              -> [{"name","label","file","text","available"}, ...]
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 风格定义
#
# `file` 是相对模型目录的路径；`fallback` 是找不到时的退路。
# normal 的 fallback 指向 v1 的 ref/default.wav —— **保证 v1 行为不变**。
# ---------------------------------------------------------------------------
STYLES: dict[str, dict] = {
    "normal": {
        "label": "正常说话",
        "file": "ref/styles/normal.wav",
        "fallback": "ref/default.wav",
        "text_file": "ref/styles/normal.txt",
        "text_fallback": "ref/ref.txt",
        "desc": "平稳叙述，默认风格（与 v1 一致）",
    },
    "firm": {
        "label": "语气强硬",
        "file": "ref/styles/firm.wav",
        "fallback": "ref/styles/normal.wav",
        "text_file": "ref/styles/firm.txt",
        "text_fallback": "ref/styles/normal.txt",
        "desc": "语气果断、有力，句末下压",
    },
    "gentle": {
        "label": "语音娇",
        "file": "ref/styles/gentle.wav",
        "fallback": "ref/styles/normal.wav",
        "text_file": "ref/styles/gentle.txt",
        "text_fallback": "ref/styles/normal.txt",
        "desc": "语气柔和、偏娇，句末轻收",
    },
}

DEFAULT_STYLE = "normal"

# 风格文本（参考文本必须与音频严格对应，所以内置一份兜底）
STYLE_TEXTS: dict[str, str] = {
    "normal": "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？",
    "firm": "北部禁区才是应龙特勤队的任务目标,再者,他们人手太少.不过他们也承诺过尽可能支援武陵.",
    "gentle": "我问了自己相同的问题,之后才终于下定决心提出申请…",
}


def normalize_style(name: str | None) -> str:
    """把任意输入规范成已知风格名；未知/空 -> 默认风格。

    **不抛异常** —— 与 v1 `synthesize` 的纪律一致（失败返回 null 而非抛）。
    """
    if not name or not isinstance(name, str):
        return DEFAULT_STYLE
    n = name.strip().lower()
    # 允许别名
    alias = {
        "default": "normal", "plain": "normal", "norm": "normal",
        "hard": "firm", "strong": "firm", "tough": "firm",
        "soft": "gentle", "sweet": "gentle", "coy": "gentle",
    }
    n = alias.get(n, n)
    return n if n in STYLES else DEFAULT_STYLE


def _first_existing(base: Path, *rels: str) -> Path | None:
    for r in rels:
        if not r:
            continue
        p = base / r
        if p.is_file():
            return p
    return None


def resolve_style(name: str | None, model_dir: str | os.PathLike | None = None
                  ) -> tuple[str, Path | None, str | None]:
    """解析风格 -> (规范名, 参考音频路径, 参考文本)。

    ## 参数
        name       风格名（None/未知 -> normal）
        model_dir  模型目录（含 ref/）；None 时用 `paths.resolve_model_dir()`

    ## 返回
        (style_name, wav_path | None, text | None)

    ## 兼容性
        未指定风格时返回 `ref/default.wav` —— **与 v1 完全一致**。
        找不到风格文件时**回退到 normal**，不报错。
    """
    key = normalize_style(name)

    if model_dir is None:
        try:
            from . import paths
            model_dir = paths.resolve_model_dir(must_exist=False)
        except Exception:
            model_dir = None

    if model_dir is None:
        return key, None, None
    base = Path(model_dir)

    spec = STYLES[key]
    wav = _first_existing(base, spec["file"], spec["fallback"],
                          "ref/default.wav")
    # 文本：优先同风格 .txt，其次内置兜底
    txt: str | None = None
    tp = _first_existing(base, spec["text_file"], spec["text_fallback"],
                         "ref/ref.txt")
    if tp is not None:
        try:
            txt = tp.read_text(encoding="utf-8").strip() or None
        except Exception:
            txt = None
    if not txt:
        txt = STYLE_TEXTS.get(key)

    # 若风格文件缺失（回退到了 default.wav），把名字也降级为 normal，
    # 免得调用方以为拿到的是 firm 却是 normal 的音频。
    if key != DEFAULT_STYLE and wav is not None and \
            wav.name == "default.wav":
        key = DEFAULT_STYLE

    return key, wav, txt


def list_styles(model_dir: str | os.PathLike | None = None) -> list[dict]:
    """列出所有风格及其可用性（给 `/voices` 端点与设置页用）。"""
    out = []
    for key, spec in STYLES.items():
        resolved, wav, _ = resolve_style(key, model_dir)
        out.append({
            "name": key,
            "label": spec["label"],
            "desc": spec["desc"],
            "file": spec["file"],
            "available": wav is not None and resolved == key,
            "resolved": str(wav) if wav is not None else None,
            "is_default": key == DEFAULT_STYLE,
        })
    return out


def style_text(name: str | None) -> str | None:
    """取风格的参考文本（不需要模型目录）。"""
    return STYLE_TEXTS.get(normalize_style(name))
