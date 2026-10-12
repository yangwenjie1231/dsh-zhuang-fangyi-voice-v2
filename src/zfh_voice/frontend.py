# -*- coding: utf-8 -*-
"""文本前端：文本 → 音素 ID（内嵌 GPT-SoVITS 前端，MIT）

内嵌包位于 _vendor/_gsv_text，已改私有名避免与宿主模块冲突，
并把 G2PW 的硬编码路径改为环境变量驱动（缺失时报错而非联网下载）。

── 中英混排（重要）────────────────────────────────────────────────────

上游 `TextPreprocessor` 用 `LangSegmenter` **按语言分段**：英文段走 `en`
音素、中文段走 `zh` 音素，所以中英混排能各按各的发音念。

但直接调 `clean_text(text, "zh")` 会**绕过分段**，而中文前端对英文是
`re.sub("[a-zA-Z]+", "", seg)` —— **英文被静默删除**（不发音、不占时长、
不报错）。实测：

    今天天气不错。               → 13 音素
    今天天气不错，Hello World。  → 13 音素（英文没产生任何音素）
    Hello World                  →  0 音素 → 合成报错

所以本模块提供 `text_to_ids_mixed()`：走 LangSegmenter 分段后再逐段取音素，
让英文用真英文念。实测 `今天天气不错，Hello World，我已经整理好了。`
ASR 反查为「今天天氣不錯,Hello World已經整理好了」—— 英文完整还原。

参考音频提供的只是**音色**（`ge` 向量），音素用哪种语言与音色无关，
所以"中文音色 + 英文发音"是成立的（实测音色相似度 0.624，与纯中文相当）。

使用前需调用 prepare()，它会：
  1. 解析模型目录
  2. 设置 ZFH_G2PW_DIR / ZFH_BERT_DIR
  3. 把 _vendor 加入 sys.path
"""
import os
import sys
import threading

from . import paths

_lock = threading.Lock()
_ready = False
_clean_text = None
_to_seq = None
_lang_segment = None

# 记住调用方给的 gsv_root。
#
# 为什么不把它加进下面每个公开函数的签名：那些函数（`split_by_lang` /
# `text_to_phonemes_mixed` …）是**纯文本处理**，让它们人人带一个"机器路径"
# 参数很难看，而且调用方（api / 两个 backend）得一路传。
# 这里记一次，`prepare()` 用；`_ready` 之前重复调用会更新它。
_gsv_root = None

# LangSegmenter 返回的语言标签 → 本前端支持的 clean_text 语言
_SUPPORTED = ("zh", "en", "ja", "ko", "yue")


def prepare(model_dir=None, gsv_root=None):
    """初始化文本前端（幂等）

    `gsv_root`：GPT-SoVITS 检出根目录。**必须能传进来** —— torch 用户的
    G2PW / BERT 就在检出里（`<gsv_root>/GPT_SoVITS/text/G2PWModel` 等），
    只按模型目录找会找不到 → `mixed_available()` 静默为假 →
    中英混排被降级成中文读法（"Hello World" 念成"哈喽 达布流欧"）。
    服务进程的命令行本来就带着 `--gsv-root`，所以这条链路是通的。
    """
    global _ready, _clean_text, _to_seq, _lang_segment, _gsv_root
    if gsv_root:
        _gsv_root = gsv_root
    with _lock:
        if _ready:
            return
        mdir = paths.resolve_model_dir(model_dir)
        g2pw = paths.g2pw_dir(mdir, _gsv_root)
        bert = paths.bert_dir(mdir, _gsv_root)
        if not os.path.isdir(g2pw):
            raise FileNotFoundError(f"G2PW 模型目录不存在: {g2pw}")
        if not os.path.isdir(bert):
            raise FileNotFoundError(f"BERT tokenizer 目录不存在: {bert}")
        # 必须在 import 内嵌前端之前设置（chinese2.py 在 import 时即加载 G2PW）
        os.environ["ZFH_G2PW_DIR"] = g2pw
        os.environ["ZFH_BERT_DIR"] = bert

        vd = paths.vendor_dir()
        if vd not in sys.path:
            sys.path.insert(0, vd)

        # nltk 数据（英文词性标注用的 averaged_perceptron_tagger_eng，5.4 MB）
        # **随包分发**，放在 _vendor/nltk_data/。
        # 必须在 import nltk 之前设 NLTK_DATA —— nltk 在导入时就把
        # nltk.data.path 固化下来了，之后再设无效（实测确认）。
        # 否则 nltk 会去用户目录找，找不到就**联网下载**（实测被安全策略拦）。
        nltk_data = os.path.join(vd, "nltk_data")
        if os.path.isdir(nltk_data):
            os.environ["NLTK_DATA"] = nltk_data

        from _gsv_text import cleaned_text_to_sequence  # noqa: E402
        from _gsv_text.cleaner import clean_text  # noqa: E402

        _clean_text = clean_text
        _to_seq = cleaned_text_to_sequence

        # LangSegmenter 是可选能力：装了 nltk/g2p_en/wordsegment 才可用。
        # 拿不到也不影响纯中文合成（只是英文会被删掉，由调用方决定是否提示）。
        try:
            from _gsv_text.LangSegmenter import LangSegmenter  # noqa: E402
            _lang_segment = LangSegmenter.getTexts
        except Exception:
            _lang_segment = None

        _ready = True


def mixed_available(gsv_root=None, model_dir=None):
    """中英混排分段是否可用（依赖 nltk / g2p_en / wordsegment）

    ⚠️ `model_dir` 与 `gsv_root` **都要传**。踩过的坑：只传 gsv_root 时，
    内部 `prepare()` → `resolve_model_dir(None)` 会去找**默认**模型目录
    （仓库 `models/` 或 `~/.cache/zfh-voice/models`），而真实部署用的是
    `--model-dir` 指定的目录 → 找不到就抛 FileNotFoundError →
    被 `api._auto_localize()` 吞掉 → **中英混排永远降级成中文读法**。
    """
    prepare(model_dir=model_dir, gsv_root=gsv_root)
    return _lang_segment is not None


def split_by_lang(text, default_lang="zh"):
    """按语言切分文本 → [(lang, text), ...]

    分段不可用时退化为 [(default_lang, text)]（即原来的"英文会被删"行为）。
    空文本返回 []（不返回空段，避免下游拿到 0 音素）。
    """
    prepare()
    if not text or not str(text).strip():
        return []
    if _lang_segment is None:
        return [(default_lang, text)]
    out = []
    for item in _lang_segment(text, default_lang):
        lang = item.get("lang") or default_lang
        seg = item.get("text") or ""
        if not seg.strip():
            continue
        if lang not in _SUPPORTED:
            lang = default_lang
        out.append((lang, seg))
    return out


def text_to_phonemes(text, language="zh", version="v2"):
    """文本 → (phones, word2ph, norm_text)

    单语言路径（不分段）。中英混排请用 text_to_phonemes_mixed()。
    """
    prepare()
    phones, word2ph, norm_text = _clean_text(text, language, version)
    return phones, word2ph, norm_text


def text_to_segments(text, language="zh", version="v2"):
    """中英混排：按语言分段，逐段产出音素与 BERT 所需信息

    返回 [(lang, phones, word2ph, norm_text), ...]

    ⚠️ **`word2ph` 对非中文语言是 `None`** —— 上游 `cleaner.py` 里
    英文/日文分支只返回 `(phones, None, norm_text)`。这不是 bug，而是
    因为 BERT 特征只对中文有意义：上游 `get_bert_inf` 对非中文段
    直接给 `torch.zeros((1024, len(phones)))`。

    所以调用方必须**按段**决定 BERT 怎么来：
      · zh 段 → 用 norm_text + word2ph 算真 BERT 特征
      · 其他段 → 给 len(phones) 列全零
    拼错会导致维度对不上（本项目的 ONNX 后端据此实现）。
    """
    prepare()
    out = []
    for lang, seg in split_by_lang(text, language):
        try:
            phones, word2ph, norm = _clean_text(seg, lang, version)
        except Exception:
            # 单段失败不该毁掉整句；退化为默认语言处理该段
            if lang == language:
                raise
            phones, word2ph, norm = _clean_text(seg, language, version)
        if not phones:
            continue
        out.append((lang, phones, word2ph, norm))
    return out


def text_to_phonemes_mixed(text, language="zh", version="v2"):
    """中英混排：分段取音素后拼接

    返回 (phones, word2ph, norm_text)。**仅适用于中文段**：
    若含非中文段，word2ph 会返回 None（因为拼接后无法用一个
    word2ph 表达两种语言的 BERT 对应关系）。

    需要完整 BERT 支持时请用 text_to_segments()，由后端按段处理。
    """
    segs = text_to_segments(text, language, version)
    if not segs:
        return [], None, ""
    if len(segs) == 1:
        lang, phones, word2ph, norm = segs[0]
        return phones, word2ph, norm

    all_phones, norm_parts = [], []
    all_zh = True
    all_w2p = []
    for lang, phones, word2ph, norm in segs:
        all_phones.extend(phones)
        norm_parts.append(norm)
        if word2ph is None:
            all_zh = False
        else:
            all_w2p.extend(word2ph)
    return all_phones, (all_w2p if all_zh else None), "".join(norm_parts)


def text_to_ids(text, language="zh", version="v2"):
    """文本 → (ids[list[int]], word2ph, norm_text)（单语言）"""
    phones, word2ph, norm_text = text_to_phonemes(text, language, version)
    return _to_seq(phones, version), word2ph, norm_text


def cleaned_ids(phones, version="v2"):
    """音素列表 → id 列表（供分段拼接时逐段取 id）"""
    prepare()
    return _to_seq(phones, version)


def text_to_ids_mixed(text, language="zh", version="v2"):
    """中英混排 → (ids[list[int]], word2ph, norm_text)

    英文用真英文音素念（推荐）；纯中文时与 text_to_ids 等价。

    ⚠️ 含非中文段时 word2ph 为 None —— 需要 BERT 特征的后端
    请改用 text_to_segments() 逐段处理。
    """
    phones, word2ph, norm_text = text_to_phonemes_mixed(text, language, version)
    return _to_seq(phones, version), word2ph, norm_text
