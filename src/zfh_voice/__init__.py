# -*- coding: utf-8 -*-
"""dsh-zhuang-fangyi-voice —— 庄方宜专属中文音色推理库

支持多种推理后端与调用方式：

后端（backend）
  onnx   仅依赖 onnxruntime，无 torch / 无 CUDA，跨平台，包体小
  torch  复用 GPT-SoVITS 原版实现，GPU 下最快

入口
  Python API  from zfh_voice import TTS; TTS().say("台词")
  CLI         python -m zfh_voice say "台词" -o out.wav
  HTTP        python -m zfh_voice serve   （POST /tts）

── 为什么这里是懒加载（不要改回顶层 import）──────────────────────────────
`api` → `audio` / `backends.base` 都在模块级 `import numpy`。
若顶层急切导入，**在什么都没装的机器上连 `--help` 和 `doctor` 都跑不起来**：

    python -m zfh_voice doctor   →   ModuleNotFoundError: No module named 'numpy'

而 `doctor` 的全部意义就是"告诉用户缺什么"——它必须在**裸环境**下可用。
所以这里用 PEP 562 的模块级 `__getattr__` 按需导入：
只有真正用到 `TTS`/`SynthResult` 时才拉 numpy 依赖链。

`paths` 只依赖标准库，可以安全地顶层导入。
"""
from .paths import resolve_model_dir, model_status  # noqa: F401

__version__ = "0.1.0"
__all__ = ["TTS", "SynthResult", "resolve_model_dir", "model_status"]

# 惰性属性 → 实际模块（第一次访问时才 import）
_LAZY = {
    "TTS": (".api", "TTS"),
    "SynthResult": (".api", "SynthResult"),
}


def __getattr__(name):
    """PEP 562：按需导入重依赖（numpy / onnxruntime / torch）"""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(target[0], __name__)
    value = getattr(mod, target[1])
    globals()[name] = value          # 缓存，后续访问不再走 __getattr__
    return value


def __dir__():
    return sorted(set(list(globals()) + list(_LAZY)))
