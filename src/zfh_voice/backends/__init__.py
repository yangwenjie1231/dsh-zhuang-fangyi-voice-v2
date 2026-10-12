# -*- coding: utf-8 -*-
"""后端工厂

    from zfh_voice.backends import make_backend
    bk = make_backend("onnx")                    # 便携
    bk = make_backend("torch", gsv_root=...)     # 快

自动选择：torch 可用则用 torch，否则 onnx。
"""
from .base import BackendError, SynthBackend  # noqa: F401

BACKENDS = ("onnx", "torch")


def make_backend(kind="auto", **kw):
    """创建后端实例

    kind: "onnx" | "torch" | "auto"
    """
    if kind == "auto":
        kind = "torch" if _torch_available(kw.get("gsv_root")) else "onnx"

    if kind == "onnx":
        from .onnx_backend import OnnxBackend
        return OnnxBackend(**{k: v for k, v in kw.items()
                              if k in ("model_dir", "providers", "threads",
                                       "log_level")})
    if kind == "torch":
        from .torch_backend import TorchBackend
        return TorchBackend(**{k: v for k, v in kw.items()
                               if k != "model_dir" or True})
    raise ValueError(f"未知后端: {kind}（可选 {BACKENDS}）")


def _torch_available(gsv_root=None):
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    if gsv_root:
        import os
        return os.path.isdir(os.path.join(gsv_root, "GPT_SoVITS"))
    return False
