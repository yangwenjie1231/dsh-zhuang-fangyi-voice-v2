# -*- coding: utf-8 -*-
"""环境体检：把「该装哪套模型」所需的判断依据一次性收集出来

用途：
  - 人：`python -m zfh_voice doctor`  看看自己该装什么
  - agent：安装时先跑这个，再据此决定装 onnx / torch、哪种精度

输出为纯文本，人机皆可读。不修改任何东西。
"""
import os
import platform
import shutil
import subprocess
import sys


def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace")
        return p.stdout.strip()
    except Exception:
        return ""


def detect_gpu():
    """返回 (是否有N卡, 名称, 显存MB, 驱动版本)"""
    out = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits"])
    if not out:
        return False, None, None, None
    line = out.splitlines()[0]
    parts = [x.strip() for x in line.split(",")]
    try:
        return True, parts[0], int(float(parts[1])), parts[2]
    except (IndexError, ValueError):
        return True, line, None, None


def has_module(name):
    """该包能否被导入（不真的导入，只看能否找到）

    必须容错：`find_spec` 在包损坏、命名空间冲突、或自定义 meta_path
    finder 抛异常时都会抛出来。体检工具**不能因为"探测过程出错"而崩掉** ——
    那会把"这个包坏了"误报成"体检工具坏了"。
    """
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError, TypeError):
        return False


def ort_providers():
    try:
        import onnxruntime as ort
        return ort.__version__, ort.get_available_providers()
    except Exception:
        return None, []


def torch_info():
    try:
        import torch
        return (torch.__version__, torch.cuda.is_available(),
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        return None, False, None


def disk_free_gb(path):
    try:
        probe = path
        while probe and not os.path.exists(probe):
            probe = os.path.dirname(probe)
        return shutil.disk_usage(probe or ".").free / 1024 ** 3
    except Exception:
        return None


# 各方案的磁盘需求（模型体积 + 余量）
SIZES = {
    "onnx_fp16": 1.40,
    "onnx_int8": 0.87,
    "onnx_fp32": 2.79,
    "torch": 0.23,
    "aux": 0.01,
}


def collect(model_dir=None):
    """收集环境事实，返回 dict"""
    from . import paths
    d = model_dir or os.environ.get("ZFH_MODEL_DIR") or \
        os.path.join(paths.repo_root(), "models")
    d = os.path.abspath(d)

    has_gpu, gpu_name, vram, drv = detect_gpu()
    ort_ver, provs = ort_providers()
    t_ver, t_cuda, t_gpu = torch_info()

    _, missing, _ = paths.model_status(d, backend="onnx")
    _, t_missing, _ = paths.model_status(d, backend="torch")

    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": f"{platform.system()} {platform.release()}",
        "model_dir": d,
        "disk_free_gb": disk_free_gb(d),
        "gpu": {"present": has_gpu, "name": gpu_name, "vram_mb": vram,
                "driver": drv},
        "onnxruntime": {"version": ort_ver, "providers": provs},
        "torch": {"version": t_ver, "cuda": t_cuda, "gpu": t_gpu},
        "installed": {
            "onnx_ok": paths.has_onnx(d),
            "torch_ok": paths.has_torch_weights(d),
            "aux_ok": paths.has_aux(d),
            "ref_ok": paths.has_ref(d),
        },
        "missing": {"onnx": missing, "torch": t_missing},
        "deps": {m: has_module(m) for m in
                 ("onnxruntime", "numpy", "soundfile", "transformers",
                  "pypinyin", "cn2an", "jieba", "opencc", "torch")},
    }


def recommend(info):
    """给出推荐方案 + 理由"""
    r = []
    gpu = info["gpu"]["present"]
    deps = info["deps"]
    t = info["torch"]
    provs = info["onnxruntime"]["providers"]

    # 后端选择
    if gpu and t["cuda"]:
        r.append(("torch", "检测到 NVIDIA GPU 且 torch 已就绪 → "
                           "torch 后端最快（RTF≈0.45，比 ONNX 快约 13 倍）"))
    elif gpu:
        r.append(("torch+onnx", "有 NVIDIA GPU，但 torch 未安装或不可用 → "
                                "建议先装 onnx 跑通，需要提速再装 torch"))
    else:
        r.append(("onnx", "未检测到 NVIDIA GPU → 用 ONNX 后端（不需要 CUDA）"))

    # 加速后端
    if "DmlExecutionProvider" in provs:
        r.append(("dml", "已装 onnxruntime-directml → Windows 任意显卡都能加速"))
    elif gpu and "CUDAExecutionProvider" not in provs:
        r.append(("dml-hint", "想给 ONNX 加 GPU 加速：把 onnxruntime 换成 "
                              "onnxruntime-directml（无需 CUDA/cuDNN）"))

    # 精度
    free = info["disk_free_gb"]
    if free is not None and free < 1.5:
        r.append(("int8", f"磁盘仅剩 {free:.1f} GB → 建议 int8（0.87GB）"))
    else:
        r.append(("fp16", "精度建议 fp16（1.40GB，数值基本无损）"))

    # 缺什么（单独一行，不混进建议）
    m = []
    if not info["installed"]["aux_ok"] or not info["installed"]["ref_ok"]:
        m.append("辅助包（G2PW 数据 + tokenizer + 默认参考音频）")
    if not info["installed"]["onnx_ok"]:
        m.append("ONNX 模型")
    if not info["installed"]["torch_ok"]:
        m.append("torch 权重（仅 torch 后端需要，约 229MB）")
    return r, ("、".join(m) if m else "无，已就绪")


def report_dict(info):
    """机器可读的体检结果（供插件 / agent 读取，不解析文本）

    插件侧拿不到子进程 stdout 的确定性保证，所以走"写文件再读"这条路。
    """
    recs, todo = recommend(info)
    d = dict(info)
    d["recommend"] = [{"tag": t, "text": x} for t, x in recs]
    d["todo"] = todo
    d["deps_missing"] = [k for k, v in info["deps"].items() if not v]
    return d


def format_report(info):
    L = []
    A = L.append
    A("=" * 66)
    A("zfh-voice 环境体检")
    A("=" * 66)
    A(f"Python      : {info['python']}  ({info['executable']})")
    A(f"系统        : {info['platform']}")
    A(f"模型目录    : {info['model_dir']}")
    free = info["disk_free_gb"]
    A(f"磁盘剩余    : {free:.1f} GB" if free is not None else "磁盘剩余    : 未知")

    g = info["gpu"]
    if g["present"]:
        A(f"GPU         : {g['name']}  {g['vram_mb']} MB  驱动 {g['driver']}")
    else:
        A("GPU         : 未检测到 NVIDIA 显卡")

    o = info["onnxruntime"]
    A(f"onnxruntime : {o['version'] or '未安装'}")
    if o["providers"]:
        A(f"  可用 EP   : {', '.join(o['providers'])}")

    t = info["torch"]
    if t["version"]:
        A(f"torch       : {t['version']}  CUDA={t['cuda']}"
          + (f"  ({t['gpu']})" if t["gpu"] else ""))
    else:
        A("torch       : 未安装")

    inst = info["installed"]
    A("")
    A("已安装内容：")
    A(f"  ONNX 模型   : {'是' if inst['onnx_ok'] else '否'}")
    A(f"  torch 权重  : {'是' if inst['torch_ok'] else '否'}")
    A(f"  辅助文件    : {'是' if inst['aux_ok'] else '否'}")
    A(f"  参考音频    : {'是' if inst['ref_ok'] else '否'}")

    deps = info["deps"]
    miss = [k for k, v in deps.items() if not v]
    A("")
    A(f"Python 依赖 : {'全部就绪' if not miss else '缺 ' + ', '.join(miss)}")

    A("")
    A("=" * 66)
    A("建议")
    A("=" * 66)
    recs, todo = recommend(info)
    for tag, text in recs:
        A(f"  · {text}")
    A("")
    A(f"尚缺      : {todo}")

    A("")
    A("下一步（按建议执行）：")
    A("  pip install -r requirements.txt")
    if not inst["onnx_ok"]:
        A("  python download_models.py                     # ONNX（默认 fp16）")
        A("  python download_models.py --precision int8    # 磁盘紧张时")
    if not inst["torch_ok"]:
        A("  python download_models.py --with-torch        # 追加 torch 权重")
        A("  python download_models.py --torch-only        # 只要 torch，跳过 ONNX")
    A("  python -m zfh_voice status           # 确认就绪")
    A('  python -m zfh_voice say "今天天气不错"   # 冒烟测试')
    return "\n".join(L)


def main(model_dir=None, json_path=None):
    """json_path 非空时写 JSON 而不是打印（给程序读）"""
    info = collect(model_dir)
    if json_path:
        import json
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_dict(info), f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"无法写入 {json_path}: {e}", file=sys.stderr)
            return 2
        return 0
    print(format_report(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
