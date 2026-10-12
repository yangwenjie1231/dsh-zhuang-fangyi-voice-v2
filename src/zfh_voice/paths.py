# -*- coding: utf-8 -*-
"""模型目录解析与完整性检查

解析优先级：
  1. 环境变量 ZFH_MODEL_DIR
  2. 仓库根目录下的 models/
  3. 用户缓存目录 ~/.cache/zfh-voice/models

模型清单固定为 7 个 ONNX + 若干辅助文件，见 ONNX_FILES / AUX_FILES。
"""
import os
import sys

# 推理必需（fp16 为默认精度）
ONNX_FILES = [
    "zfh_t2s_encoder.onnx",
    "zfh_t2s_fsdec.onnx",
    "zfh_t2s_sdec.onnx",
    "zfh_vits.onnx",
    "bert_encoder.onnx",
    "hubert_encoder.onnx",
    "g2pW.onnx",
]

# 文本前端辅助文件
AUX_FILES = [
    os.path.join("G2PWModel", "g2pW.onnx"),      # 与 g2pW.onnx 同源，前端按此路径加载
    os.path.join("G2PWModel", "bopomofo_to_pinyin_wo_tune_dict.json"),
    os.path.join("G2PWModel", "char_bopomofo_dict.json"),
    os.path.join("G2PWModel", "MONOPHONIC_CHARS.txt"),
    os.path.join("G2PWModel", "POLYPHONIC_CHARS.txt"),
    os.path.join("G2PWModel", "config.py"),
    os.path.join("G2PWModel", "version"),
    # fast tokenizer 格式：tokenizer.json + config.json 即可
    os.path.join("chinese-roberta-wwm-ext-large", "tokenizer.json"),
    os.path.join("chinese-roberta-wwm-ext-large", "config.json"),
    os.path.join("ref", "default.wav"),
]


def repo_root():
    """仓库根目录（本文件是 <root>/src/zfh_voice/paths.py）"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def default_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache")
    return os.path.join(base, "zfh-voice", "models")


def has_onnx(d):
    """ONNX 推理所需的 7 个模型是否齐全"""
    if not d or not os.path.isdir(d):
        return False
    return all(os.path.exists(os.path.join(d, f)) for f in ONNX_FILES)


def onnx_needs_sv(vits_path):
    """`zfh_vits.onnx` 是否声明了 `sv_emb` 输入（= v2Pro 的包）

    只读**图结构**（`load_external_data=False`），不把 250 MB 权重读进内存 ——
    实测 153 ms。用于体检时判断要不要额外的说话人编码器。
    """
    try:
        import onnx
        m = onnx.load(vits_path, load_external_data=False)
    except Exception:  # noqa: BLE001  —— 缺 onnx / 文件坏了都不该让体检崩
        return False
    try:
        return any(i.name == "sv_emb" for i in m.graph.input)
    except Exception:  # noqa: BLE001
        return False


def has_torch_weights(d, exp_name=None):
    """torch 后端所需的两个权重是否齐全（在 <d>/torch_weights/ 下）

    兼容两种命名，且**不锁死具体文件名**：

      · 默认发行版：`zfh-e4.ckpt`（GPT）+ `zfh_e6_s186.pth`（SoVITS）
      · 自己重训的：`<任意名>-e<N>.ckpt`（GPT）+ `<任意名>_e<N>_s<N>.pth`（SoVITS）

    为什么放宽：原先硬编码了 `zfh-e4.ckpt` / `zfh_e6_s186.pth`，
    导致用户**重训或换版本（如 v2Pro）后目录被判成"没有模型"** ——
    而报错只说"缺 1 项"，不说缺什么，很难查。
    """
    if not d or not os.path.isdir(d):
        return False
    td = os.path.join(d, "torch_weights")
    if not os.path.isdir(td):
        return False

    # 优先认默认发行版的文件名（最常见，先命中就不用扫目录）
    if exp_name:
        if (os.path.exists(os.path.join(td, f"{exp_name}-e4.ckpt"))
                and os.path.exists(os.path.join(td, f"{exp_name}_e6_s186.pth"))):
            return True

    try:
        names = os.listdir(td)
    except OSError:
        return False

    import re
    has_gpt = any(
        (n.endswith(".ckpt") and re.search(r"-e\d+\.ckpt$", n))
        or n.endswith("s1v3.ckpt")
        for n in names)
    has_sovits = any(
        n.endswith(".pth") and re.search(r"_e\d+_s\d+\.pth$", n)
        for n in names)
    return bool(has_gpt and has_sovits)


def has_aux(d):
    """文本前端辅助文件是否齐全"""
    if not d or not os.path.isdir(d):
        return False
    return (os.path.exists(os.path.join(d, "G2PWModel", "g2pW.onnx"))
            and os.path.exists(os.path.join(d, "chinese-roberta-wwm-ext-large",
                                            "tokenizer.json")))


def has_ref(d):
    return bool(d) and os.path.exists(os.path.join(d, "ref", "default.wav"))


def _matches(d, require):
    if not d or not os.path.isdir(d):
        return False
    if require == "onnx":
        return has_onnx(d)
    if require == "torch":
        return has_torch_weights(d)
    if require == "aux":
        return has_aux(d)
    # require=None：只要目录里有「能支撑推理的内容」就认
    # 注意：仅有 ref/ 音频不算——那无法合成，必须有模型或文本前端辅助文件
    return has_onnx(d) or has_torch_weights(d) or has_aux(d)


def _missing_in(d, require):
    """列出该目录下按 require 还缺什么"""
    if not d:
        return []
    if require == "onnx":
        return [f for f in ONNX_FILES
                if not os.path.exists(os.path.join(d, f))]
    if require == "torch":
        td = os.path.join(d, "torch_weights")
        return [f"{f}（torch 后端用）" for f in ("zfh-e4.ckpt", "zfh_e6_s186.pth")
                if not os.path.exists(os.path.join(td, f))]
    if require == "aux":
        return [f for f in AUX_FILES if not os.path.exists(os.path.join(d, f))]
    return ["该目录里没有任何可识别的模型内容"]


def discover_model_dirs():
    """**自动探测**常见的模型目录位置（v2 新增）。

    ## 为什么需要
    v1 的解析顺序是：
        1. 显式 `model_dir=`
        2. 环境变量 `ZFH_MODEL_DIR`
        3. `<repo>/models`
        4. `~/.cache/zfh-voice/models`

    用户的实际布局往往是 `<某处>/zfh-models` / `zfh-models-v2pro`，
    得靠**在 DSH patch 里手写绝对路径**才行（容易写错、换机就废）。

    ## 探测策略
    按"越具体越优先"扫描候选根目录下的 `zfh-models*` 子目录：
        · `ZFH_MODELS_ROOT` 环境变量（用户可显式指定一个根）
        · 仓库的**父目录**（最常见：仓库与模型并排）
        · 仓库自己
        · 用户主目录

    ## 排序
    同一根下多个候选时，**优先 v2pro**（当前生产版本），其次按名字倒序
    （`zfh-models-v2pro` > `zfh-models-v2` > `zfh-models`）。

    ## 返回
        去重后的目录列表（**只包含存在的**；内容是否齐全交给 `_matches()`）
    """
    cands: list[str] = []

    def _add(p):
        if not p:
            return
        try:
            ap = os.path.abspath(os.path.expanduser(str(p)))
        except Exception:
            return
        if os.path.isdir(ap) and ap not in cands:
            cands.append(ap)

    roots = []
    if os.environ.get("ZFH_MODELS_ROOT"):
        roots.append(os.environ["ZFH_MODELS_ROOT"])
    try:
        parent = os.path.dirname(repo_root())
        roots.append(parent)
        # 再上一层：仓库常在 <X>/<proj>/ 下，而模型在 <X>/<其他>/ 下
        # 例：D:\A\dsh-...-voice 与 D:\A\voiceclone\zfh-models
        roots.append(os.path.dirname(parent))
    except Exception:
        pass
    try:
        roots.append(repo_root())
    except Exception:
        pass
    roots.append(os.path.expanduser("~"))

    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        if os.path.basename(os.path.normpath(root)).startswith("zfh-models"):
            _add(root)
        try:
            entries = sorted(os.listdir(root), reverse=True)
        except OSError:
            continue
        for name in entries:
            full = os.path.join(root, name)
            # ① 根下的 zfh-models*
            if name.startswith("zfh-models"):
                _add(full)
            # ② 再下探一层（<root>/<proj>/zfh-models*）
            #    只对看起来像"项目容器"的目录下探，避免全盘扫描
            elif os.path.isdir(full) and name in (
                    "voiceclone", "voice", "models", "zfy", "zfy-v5",
                    "projects", "work", "ai", "tts"):
                try:
                    for sub in os.listdir(full):
                        if sub.startswith("zfh-models"):
                            _add(os.path.join(full, sub))
                except OSError:
                    pass

    for name in (".zfh-models", os.path.join(".zfh-voice", "models")):
        _add(os.path.join(os.path.expanduser("~"), name))

    def rank(p: str) -> tuple:
        b = os.path.basename(os.path.normpath(p)).lower()
        return (0 if "v2pro" in b else 1, b)

    return sorted(cands, key=rank)


def resolve_model_dir(model_dir=None, must_exist=True, require=None,
                      discover=True):
    """按优先级找到可用的模型目录

    require:
        "onnx"  → 必须齐全 7 个 ONNX（ONNX 后端用）
        "torch" → 必须有 torch_weights/ 下的 .ckpt/.pth
        "aux"   → 必须有文本前端辅助文件
        None    → 只要含任一类内容即可（默认；兼容只用 torch 后端的场景）

    discover:
        v2 新增。True（默认）时，在前 4 个固定位置都没命中后**自动探测**
        常见布局（见 `discover_model_dirs`）。设 False 可回到 v1 行为。

    找不到时会抛出一段**可直接照做**的指引，而不是只说一句"找不到"。
    """
    cands = []
    if model_dir:
        cands.append(model_dir)
    if os.environ.get("ZFH_MODEL_DIR"):
        cands.append(os.environ["ZFH_MODEL_DIR"])
    cands.append(os.path.join(repo_root(), "models"))
    cands.append(default_cache_dir())

    for d in cands:
        if _matches(d, require):
            return os.path.abspath(d)

    # ---------- v2：自动探测 ----------
    #
    # ⚠️ 语义约束（踩过的坑）：v1 的测试用
    #       try: resolve_model_dir(); return True
    #       except FileNotFoundError: return False
    #   来判断"模型是否就绪"，并据此 skip 掉需要 ONNX 的用例。
    #   如果自动探测在**只有 torch 权重**的目录上就返回成功，
    #   那些用例就会跑起来然后因为缺 ONNX 而失败。
    #
    #   所以：自动探测**不得放宽 require 的语义**。
    #   `require=None` 时只认"**含 ONNX** 或 **显式 torch 意图**"的目录 ——
    #   即沿用 `_matches` 的原语义，探测到的目录必须同样通过 `_matches`。
    discovered = discover_model_dirs() if discover else []
    for d in discovered:
        if _matches(d, require):
            return os.path.abspath(d)
    # 探测到但不匹配的，只在**错误指引**里列出来（便于用户看到最接近的候选）
    for d in discovered:
        if d not in cands:
            cands.append(d)

    if not must_exist:
        return os.path.abspath(cands[0]) if cands else None

    # 构造可执行的指引
    lines = []
    lines.append(f"未找到可用的模型目录（需要：{require or '任意一类模型'}）。")
    lines.append("")
    lines.append("已检查这些位置：")
    for c in cands:
        ap = os.path.abspath(c)
        tag = "存在" if os.path.isdir(ap) else "不存在"
        miss = _missing_in(ap, require)
        extra = f"，缺 {len(miss)} 项" if (os.path.isdir(ap) and miss) else ""
        lines.append(f"  [{tag}{extra}] {ap}")
    lines.append("")
    lines.append("解决办法（任选其一）：")
    if require == "onnx":
        lines.append("  1) 装 ONNX 模型（推荐 fp16，约 1.4GB）")
        lines.append("       python download_models.py")
        lines.append("     磁盘紧张可用 int8（0.87GB）：")
        lines.append("       python download_models.py --precision int8")
        lines.append("  2) 改用 torch 后端（不需要 ONNX，需要 .ckpt/.pth）")
        lines.append("       python download_models.py --torch-only")
        lines.append("       python -m zfh_voice --backend torch "
                     "--gsv-root <GPT-SoVITS 目录> say \"你好\"")
    elif require == "torch":
        lines.append("  下载 torch 后端所需的原始权重（约 229MB）：")
        lines.append("       python download_models.py --with-torch")
        lines.append("     （若只用 ONNX 后端，则不需要这些权重）")
    else:
        lines.append("       python download_models.py")
        lines.append("       python download_models.py --torch-only")
    lines.append(f"  3) 指定已有目录：设环境变量 ZFH_MODEL_DIR，或用 --model-dir")
    lines.append("  4) 不确定该装哪套？先跑环境体检：")
    lines.append("       python -m zfh_voice doctor")
    raise FileNotFoundError("\n".join(lines))


def model_status(model_dir=None, backend="onnx"):
    """返回 (目录, 缺失文件列表, 已就绪 bool)

    backend="onnx"  检查 7 个 ONNX + 辅助文件
    backend="torch" 检查 torch 权重 + 辅助文件

    ⚠️ torch 分支**不能硬编码权重文件名**。早先写死
    `zfh-e4.ckpt` / `zfh_e6_s186.pth`，导致用户重训或换版本后
    （如 v2Pro 的 `zfh_v2pro_e12_s384.pth`）`status` 报「缺 1 个文件」，
    但**实际能正常合成** —— 报错与事实相反，很难查。
    这里改成复用 `has_torch_weights()`（它已支持自定义命名）。
    """
    d = model_dir or os.environ.get("ZFH_MODEL_DIR") or \
        os.path.join(repo_root(), "models")
    d = os.path.abspath(d)

    if backend == "onnx":
        required = list(AUX_FILES) + list(ONNX_FILES)
        missing = [f for f in required
                   if not os.path.exists(os.path.join(d, f))]
        # v2Pro 的包多一个说话人编码器。**必须在这里检查** —— 否则用户漏下它时
        # `status` 会说"就绪"，直到真正合成才报错（而且报的是后端里的一句
        # 「缺少 sv_after_fbank.onnx」，与"体检说就绪"自相矛盾）。
        #
        # 判定依据是 `zfh_vits.onnx` 是否声明了 `sv_emb` 输入 —— 比猜文件名可靠：
        # 万一用户混用不同版本的 vits，也能发现。
        if "zfh_vits.onnx" not in missing:
            if onnx_needs_sv(os.path.join(d, "zfh_vits.onnx")):
                if not os.path.exists(os.path.join(d, "sv_after_fbank.onnx")):
                    missing.append("sv_after_fbank.onnx  (v2Pro 的说话人编码器)")
        return d, missing, len(missing) == 0

    # torch：用模式匹配判断，而不是逐文件名
    if has_torch_weights(d):
        return d, [], True
    # 不就绪时给出**具体缺什么**，便于排查
    td = os.path.join(d, "torch_weights")
    import re
    names = []
    if os.path.isdir(td):
        try:
            names = os.listdir(td)
        except OSError:
            names = []
    has_gpt = any(n.endswith(".ckpt") and (re.search(r"-e\d+\.ckpt$", n)
                                           or n.endswith("s1v3.ckpt"))
                  for n in names)
    has_sov = any(re.search(r"_e\d+_s\d+\.pth$", n) for n in names)
    missing = []
    if not has_gpt:
        missing.append(os.path.join("torch_weights", "*-e<N>.ckpt  (GPT)"))
    if not has_sov:
        missing.append(os.path.join("torch_weights", "*_e<N>_s<N>.pth  (SoVITS)"))
    if not missing:
        missing.append(os.path.join("torch_weights", "(权重不完整)"))
    return d, missing, False


def _gsv_candidates(model_dir, rel_in_gsv, name, gsv_root=None, env_key=None):
    """文本前端资源（G2PW / BERT）的候选路径，按优先级排列。

    ⚠️ 这是**踩过的坑**：`g2pw_dir`/`bert_dir` 原先只看模型目录
    （`<model_dir>/G2PWModel`、`<model_dir>/chinese-roberta-wwm-ext-large`），
    但 torch 用户的这两样东西**本来就在 GPT-SoVITS 检出里**：

        <gsv_root>/GPT_SoVITS/text/G2PWModel
        <gsv_root>/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large

    于是 `frontend.prepare()` 抛 FileNotFoundError → `api._auto_localize()`
    捕获后把 `localize` 判成 True → **中英混排永远走中文读法兜底**
    （"Hello World" 念成"哈喽 达布流欧"），而且**不报任何错**。
    表现是"英文能出声但念得怪"，极难定位 —— 所以这里按顺序找：

      1. 显式环境变量（`ZFH_G2PW_DIR` / `ZFH_BERT_DIR`）—— 用户说了算；
      2. 模型目录（onnx 那套就是这么装的）；
      3. `gsv_root` 参数 / `ZFH_GSV_ROOT` / `--gsv-root` 指向的 GPT-SoVITS 检出。
    """
    env_key = env_key or ("ZFH_G2PW_DIR" if name == "G2PW" else "ZFH_BERT_DIR")
    out = []
    explicit = os.environ.get(env_key)
    if explicit:
        out.append(explicit)
    if model_dir:
        out.append(os.path.join(model_dir, name))
    for root in _gsv_roots(gsv_root):
        out.append(os.path.join(root, "GPT_SoVITS", rel_in_gsv))
    return out


def _gsv_roots(extra=None):
    """可能的 GPT-SoVITS 检出根目录（去重，保序）。

    `extra` 是**调用方显式给的**（`--gsv-root` / `TorchBackend(gsv_root=…)`）——
    它优先于环境变量与默认约定。**这条很关键**：服务进程的命令行里明明带着
    `--gsv-root`，但前端原先只认环境变量，
    于是"CLI 知道、前端不知道"，中英混排被静默降级成中文读法。
    """
    roots = []
    if extra:
        roots.append(extra)
    for key in ("ZFH_GSV_ROOT", "GSV_ROOT"):
        v = os.environ.get(key)
        if v:
            roots.append(v)
    # 与 torch_backend 的默认约定一致：仓库同级 / 仓库内 third_party
    rr = repo_root()
    roots.append(os.path.join(os.path.dirname(rr), "GPT-SoVITS"))
    roots.append(os.path.join(rr, "third_party", "GPT-SoVITS"))
    seen, uniq = set(), []
    for r in roots:
        k = os.path.normcase(os.path.abspath(r))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def _first_existing(cands):
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[0] if cands else None


def g2pw_dir(model_dir, gsv_root=None):
    """G2PW 模型目录（**会去 GPT-SoVITS 检出里找**，见 `_gsv_candidates`）。"""
    return _first_existing(_gsv_candidates(
        model_dir, os.path.join("text", "G2PWModel"), "G2PWModel", gsv_root,
        env_key="ZFH_G2PW_DIR"))


def bert_dir(model_dir, gsv_root=None):
    """BERT tokenizer 目录（同上，优先模型目录，回落 GPT-SoVITS 检出）。"""
    return _first_existing(_gsv_candidates(
        model_dir, os.path.join("pretrained_models", "chinese-roberta-wwm-ext-large"),
        "chinese-roberta-wwm-ext-large", gsv_root, env_key="ZFH_BERT_DIR"))


def vendor_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
