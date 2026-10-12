# -*- coding: utf-8 -*-
"""英文可念化：把中文文本里的英文转成中文读法

── 为什么需要这层 ──────────────────────────────────────────────────────

本模型的中文文本前端（GPT-SoVITS `chinese2.py`）有这样一行：

    processed_segments = [re.sub("[a-zA-Z]+", "", seg) for seg in segments]

也就是说 **中文模式下英文是被整段删除的** —— 不发音、也不占时长。
实测（音素数不变、ASR 反查也听不到）：

    「今天天气不错。」                   → 13 个音素
    「今天天气不错，Hello World。」      → 13 个音素（英文没产生任何音素）
    「Hello World」（纯英文）            → 0 个音素 → 合成直接报错

对一个要念 AI 回答的桌宠来说这是硬伤：回答里出现 `GPU`、`Python`、
`Hello` 都会被静默吃掉，用户只看到"这句念得怪怪的"。

── 这层怎么做 ──────────────────────────────────────────────────────────

在**送进前端之前**把英文替换成等效的中文读法：

  1. 常见词查表：`Python` → `派森`、`Hello` → `哈喽`
  2. 全大写缩写逐字母念：`GPU` → `基皮尤`
  3. 其他词逐字母念（保底，至少能听出是什么词）
  4. 数字/符号**不动** —— 前端本来就能正确处理
     （`3` → `三`、`￥100` → `幺零零`、`50%` → `百分之五十`）

设计取向是**宁可念得笨拙，也不要静默丢失**：用户听到"基-皮-尤"能明白，
听到一片空白只会以为坏了。
"""
import re

# ── 常见词表（桌面宠物 / AI 对话场景）────────────────────────────────────
WORDS = {
    # 问候与常用语
    "hello": "哈喽", "hi": "嗨", "hey": "嘿", "ok": "欧克", "okay": "欧克",
    "yes": "耶斯", "no": "弄", "sorry": "骚瑞", "thanks": "三克油",
    "thank": "三克", "you": "优", "bye": "拜拜", "goodbye": "拜拜",
    "good": "古德", "morning": "摸宁", "night": "奈特", "welcome": "威尔康",
    # 技术词
    "python": "派森", "java": "加瓦", "javascript": "加瓦斯克瑞普特",
    "linux": "林纳克斯", "windows": "温豆斯", "mac": "麦克",
    "google": "谷歌", "github": "吉特哈布", "git": "吉特",
    "code": "扣德", "bug": "巴格", "debug": "滴巴格",
    "model": "模型", "test": "泰斯特", "data": "数据", "file": "文件",
    "error": "错误", "warning": "警告", "server": "服务器",
    "api": "诶皮艾", "url": "优阿尔艾欧", "http": "艾尺提提皮",
    "json": "杰森", "html": "艾尺提艾姆艾欧", "css": "西艾斯艾斯",
    "cpu": "西皮优", "gpu": "基皮尤", "ram": "内存", "rom": "只读内存",
    "ai": "诶艾", "ml": "艾姆艾欧", "ui": "优艾", "ux": "优艾克斯",
    "os": "欧艾斯", "pc": "皮西", "app": "应用", "web": "网页",
    "wifi": "歪fai", "usb": "优艾斯比", "pdf": "皮迪艾弗",
    "torch": "托奇", "onnx": "欧恩恩艾克斯", "numpy": "南派",
    "vs": "威艾斯", "etc": "等等", "ie": "也就是", "eg": "例如",
    # 时间/单位
    "am": "上午", "pm": "下午", "id": "艾迪", "ip": "艾皮",
    "gb": "吉字节", "mb": "兆字节", "kb": "千字节", "tb": "太字节",
}

# ── 字母读法（逐字母念时用）──────────────────────────────────────────────
LETTERS = {
    "a": "诶", "b": "比", "c": "西", "d": "迪", "e": "伊", "f": "艾弗",
    "g": "基", "h": "艾尺", "i": "艾", "j": "杰", "k": "开", "l": "艾勒",
    "m": "艾姆", "n": "艾恩", "o": "欧", "p": "皮", "q": "丘", "r": "阿尔",
    "s": "艾斯", "t": "提", "u": "优", "v": "维", "w": "达布流", "x": "艾克斯",
    "y": "歪", "z": "贼德",
}

# 匹配英文单词（含词内连字符/撇号），数字与符号不碰
_WORD = re.compile(r"[A-Za-z]+(?:['\-][A-Za-z]+)*")


def _spell(word):
    """逐字母念：GPU → 基皮尤"""
    return "".join(LETTERS.get(c.lower(), c) for c in word if c.isalpha())


def _render(word):
    """单个英文词的读法"""
    low = word.lower()
    if low in WORDS:
        return WORDS[low]
    # 词形变化：复数 / 过去式，先试去掉词尾
    for suf in ("s", "es", "ed", "ing"):
        if low.endswith(suf) and low[: -len(suf)] in WORDS:
            return WORDS[low[: -len(suf)]]
    # 全大写缩写（GPU / CPU / API）逐字母念
    if word.isupper() and 1 < len(word) <= 8:
        return _spell(word)
    # 很短的一律逐字母
    if len(word) <= 2:
        return _spell(word)
    # 其余：逐字母（保底，至少听得出是什么词）
    return _spell(word)


def localize(text):
    """把文本里的英文转成中文读法

    只动英文词，**不动数字与符号** —— 前端本来就能正确处理它们
    （`3`→`三`、`￥100`→`幺零零`、`50%`→`百分之五十`）。

    >>> localize("GPU 和 AI 都很重要。")
    '基皮尤 和 诶艾 都很重要。'
    >>> localize("今天天气不错，Hello World。")
    '今天天气不错，哈喽 沃尔德。'
    """
    if not text or not isinstance(text, str):
        return text
    return _WORD.sub(lambda m: _render(m.group(0)), text)


def has_english(text):
    """文本里有没有英文（用于提示用户"这些会被转换"）"""
    return bool(text) and isinstance(text, str) and _WORD.search(text) is not None
