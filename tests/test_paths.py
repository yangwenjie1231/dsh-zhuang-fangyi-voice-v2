# -*- coding: utf-8 -*-
"""模型目录识别与后端维度判定的回归测试

重点覆盖曾经的 bug：只用 torch 后端时，resolve_model_dir 仍强制要求
7 个 ONNX 文件齐全，导致 .ckpt/.pth 下好了也用不了。

运行：pytest -q tests/test_paths.py    或   python tests/test_paths.py
"""
import os
import shutil
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from zfh_voice import paths  # noqa: E402


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"\0")


@pytest.fixture
def workdir():
    d = tempfile.mkdtemp(prefix="zfh_paths_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _build(root, name, onnx=False, torch_=False, aux=True):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    if onnx:
        for f in paths.ONNX_FILES:
            _touch(os.path.join(d, f))
    if torch_:
        _touch(os.path.join(d, "torch_weights", "zfh-e4.ckpt"))
        _touch(os.path.join(d, "torch_weights", "zfh_e6_s186.pth"))
    if aux:
        for f in paths.AUX_FILES:
            _touch(os.path.join(d, f.replace("/", os.sep)))
    return d


def _selected(d, require):
    """该目录是否会被 resolve_model_dir 选中"""
    try:
        return os.path.abspath(paths.resolve_model_dir(d, require=require)) == \
            os.path.abspath(d)
    except FileNotFoundError:
        return False


def test_torch_only_dir_is_recognized(workdir):
    """torch-only 目录（无 ONNX）应被接受 —— 这是修复的 bug"""
    d = _build(workdir, "torch_only", torch_=True)
    assert _selected(d, None) is True
    assert _selected(d, "torch") is True
    # 但不应被当成完整的 ONNX 目录
    assert _selected(d, "onnx") is False


def test_onnx_only_dir_is_recognized(workdir):
    d = _build(workdir, "onnx_only", onnx=True)
    assert _selected(d, None) is True
    assert _selected(d, "onnx") is True
    assert _selected(d, "torch") is False


def test_empty_dir_rejected(workdir):
    d = _build(workdir, "empty", aux=False)
    assert _selected(d, None) is False
    assert _selected(d, "onnx") is False
    assert _selected(d, "torch") is False


def test_ref_only_dir_rejected(workdir):
    """只有参考音频、没有模型或辅助文件 → 不算可用模型目录"""
    d = os.path.join(workdir, "refonly")
    _touch(os.path.join(d, "ref", "default.wav"))
    assert _selected(d, None) is False


def test_model_status_is_backend_aware(workdir):
    d = _build(workdir, "torch_only", torch_=True)
    _, miss_onnx, ok_onnx = paths.model_status(d, backend="onnx")
    _, miss_torch, ok_torch = paths.model_status(d, backend="torch")
    assert ok_torch is True and not miss_torch
    assert ok_onnx is False and miss_onnx      # ONNX 未装，应如实反映

    d2 = _build(workdir, "onnx_only", onnx=True)
    _, miss_onnx2, ok_onnx2 = paths.model_status(d2, backend="onnx")
    assert ok_onnx2 is True and not miss_onnx2


def test_model_status_accepts_retrained_names(workdir):
    """`status` 必须认自定义命名的权重（曾误报「缺 1 个文件」）

    真事故：`model_status` 硬编码 `zfh-e4.ckpt` / `zfh_e6_s186.pth`，
    于是 v2Pro 的 `zfh_v2pro_e12_s384.pth` 被当成缺失 ——
    **能正常合成却报缺文件**，报错与事实相反。
    """
    d = os.path.join(workdir, "v2pro_style")
    _touch(os.path.join(d, "torch_weights", "zfh-e4.ckpt"))
    _touch(os.path.join(d, "torch_weights", "zfh_v2pro_e12_s384.pth"))
    _, miss, ok = paths.model_status(d, backend="torch")
    assert ok is True, f"自定义命名应被判为就绪，却报缺 {miss}"
    assert not miss


def test_model_status_reports_specific_gaps(workdir):
    """不就绪时要说明**缺哪一类**，而不是只报个数"""
    d = os.path.join(workdir, "partial")
    _touch(os.path.join(d, "torch_weights", "zfh-e4.ckpt"))   # 只有 GPT
    _, miss, ok = paths.model_status(d, backend="torch")
    assert ok is False
    joined = " ".join(miss)
    assert "SoVITS" in joined or "_s" in joined, \
        f"应指出缺 SoVITS 权重，实际: {miss}"
    assert not any("GPT" in m and "ckpt" in m for m in miss), \
        "GPT 权重已存在，不该报缺"


def test_model_status_torch_dir_without_weights(workdir):
    """torch_weights/ 存在但为空 → 应报缺，不该崩"""
    d = os.path.join(workdir, "empty_tw")
    os.makedirs(os.path.join(d, "torch_weights"), exist_ok=True)
    _, miss, ok = paths.model_status(d, backend="torch")
    assert ok is False
    assert len(miss) >= 1


def test_error_message_is_actionable(workdir, monkeypatch):
    """找不到模型时的报错必须包含可照做的指引，而不是一句「找不到目录」"""
    d = os.path.join(workdir, "nothing")
    os.makedirs(d, exist_ok=True)
    monkeypatch.setenv("ZFH_MODEL_DIR", d)
    monkeypatch.setattr(paths, "repo_root", lambda: workdir)
    monkeypatch.setattr(paths, "default_cache_dir",
                        lambda: os.path.join(workdir, "nope"))
    with pytest.raises(FileNotFoundError) as ei:
        paths.resolve_model_dir(require="onnx")
    msg = str(ei.value)
    assert "download_models.py" in msg
    assert "doctor" in msg


def test_has_aux_and_ref(workdir):
    d = _build(workdir, "aux_only", onnx=False, torch_=False, aux=True)
    assert paths.has_aux(d) is True
    assert paths.has_ref(d) is True


# ---------- 自定义/重训权重的命名（曾经的静默失败） ----------
#
# 背景：`has_torch_weights` 曾**硬编码** `zfh-e4.ckpt` / `zfh_e6_s186.pth`。
# 用户重训或换版本（如 v2Pro）后文件名变成 `zfh_v2pro_e12_s384.pth`，
# 目录就被判成"没有模型"；而报错只说"缺 1 项"，不说缺什么，很难查。

def test_retrained_weight_names_recognized(workdir):
    """重训产物的命名（<名>_e<N>_s<N>.pth）也要被认出来"""
    d = os.path.join(workdir, "retrained")
    _touch(os.path.join(d, "torch_weights", "zfh_v2pro_e12_s384.pth"))
    _touch(os.path.join(d, "torch_weights", "zfh-e4.ckpt"))
    assert paths.has_torch_weights(d) is True
    assert _selected(d, "torch") is True


def test_v2pro_style_both_new_names(workdir):
    """GPT 与 SoVITS 都用新命名时也要认"""
    d = os.path.join(workdir, "both_new")
    _touch(os.path.join(d, "torch_weights", "mymodel-e8.ckpt"))
    _touch(os.path.join(d, "torch_weights", "mymodel_e10_s320.pth"))
    assert paths.has_torch_weights(d) is True


def test_only_gpt_weight_is_not_enough(workdir):
    """只有 GPT 没有 SoVITS → 不算完整（否则会在加载时炸）"""
    d = os.path.join(workdir, "gpt_only")
    _touch(os.path.join(d, "torch_weights", "zfh-e4.ckpt"))
    assert paths.has_torch_weights(d) is False


def test_only_sovits_weight_is_not_enough(workdir):
    d = os.path.join(workdir, "sov_only")
    _touch(os.path.join(d, "torch_weights", "zfh_e6_s186.pth"))
    assert paths.has_torch_weights(d) is False


def test_arbitrary_files_do_not_count(workdir):
    """随便两个文件不该被当成权重（避免假阳性）"""
    d = os.path.join(workdir, "junk")
    _touch(os.path.join(d, "torch_weights", "notes.ckpt"))
    _touch(os.path.join(d, "torch_weights", "data.pth"))
    assert paths.has_torch_weights(d) is False


def test_default_names_still_win(workdir):
    """默认发行版命名仍应被优先识别（不破坏现有安装）"""
    d = _build(workdir, "default", torch_=True)
    assert paths.has_torch_weights(d) is True
    assert paths.has_torch_weights(d, exp_name="zfh") is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
