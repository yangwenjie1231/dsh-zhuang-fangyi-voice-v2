# -*- coding: utf-8 -*-
"""下载模型文件（优先 ModelScope，自动回退 GitHub）

用法：
    python download_models.py                    # 默认 fp16
    python download_models.py --precision int8
    python download_models.py --precision fp32
    python download_models.py --dir D:\\models    # 指定目录
    python download_models.py --check            # 只检查不下载
    python download_models.py --source github    # 指定下载源

产出目录结构：
    models/
    ├─ *.onnx                          7 个推理模型
    ├─ G2PWModel/                      多音字模型与数据
    ├─ chinese-roberta-wwm-ext-large/  BERT tokenizer
    └─ ref/default.wav(.txt)           默认参考音频

国内网络建议优先 ModelScope（默认），GitHub 作为备用源。
"""
import argparse
import os
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- 下载源（按顺序尝试）----
MS_REPO = "yangwenjie1231/dsh-zhuang-fangyi-voice"
GH_REPO = "yangwenjie1231/dsh-zhuang-fangyi-voice"
GH_TAG = "models-v1"

SOURCES = {
    "modelscope": f"https://modelscope.cn/models/{MS_REPO}/resolve/master",
    "github": f"https://github.com/{GH_REPO}/releases/download/{GH_TAG}",
}
DEFAULT_ORDER = ("modelscope", "github")

ONNX_FILES = [
    "zfh_t2s_encoder.onnx", "zfh_t2s_fsdec.onnx", "zfh_t2s_sdec.onnx",
    "zfh_vits.onnx", "bert_encoder.onnx", "hubert_encoder.onnx", "g2pW.onnx",
]
PRECISIONS = ("fp16", "int8", "fp32")

# torch 后端所需的原始权重（**不是** ONNX，onnxruntime 无法加载）。
# 只有想用 torch 后端（GPU 下快 13 倍）时才需要下载，约 229 MB。
TORCH_ASSETS = {
    "torch__zfh-e4.ckpt": "GPT_weights_v2/zfh-e4.ckpt",
    "torch__zfh_e6_s186.pth": "SoVITS_weights_v2/zfh_e6_s186.pth",
}


def target_dir(explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    if os.environ.get("ZFH_MODEL_DIR"):
        return os.path.abspath(os.environ["ZFH_MODEL_DIR"])
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _curl_ok(url, dst):
    """用 curl 下载（支持续传）；Windows 上加 --ssl-no-revoke 避免吊销检查失败"""
    import subprocess
    extra = ["--ssl-no-revoke"] if sys.platform == "win32" else []
    curl = "curl.exe" if sys.platform == "win32" else "curl"
    try:
        r = subprocess.run(
            [curl, "-L", "--fail", "--retry", "3", "--retry-delay", "2"]
            + extra + ["-C", "-", "-o", dst, url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        # 服务端不支持 Range 时续传会失败，清掉重来一次
        if os.path.exists(dst):
            os.remove(dst)
        r = subprocess.run(
            [curl, "-L", "--fail", "--retry", "3"] + extra + ["-o", dst, url],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    except FileNotFoundError:
        return False


def _urllib_ok(url, dst):
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=60) as resp, open(dst, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        return os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        return False


def fetch(asset, dst, expect_mb=None, order=DEFAULT_ORDER):
    """按 source 顺序尝试下载 asset → dst"""
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  已存在，跳过: {os.path.basename(dst)}")
        return True
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    print(f"  下载 {asset}" + (f"  (~{expect_mb}MB)" if expect_mb else ""))

    for src in order:
        base = SOURCES[src]
        url = f"{base}/{asset}"
        print(f"    [{src}] ...", end="", flush=True)
        for fn in (_curl_ok, _urllib_ok):
            if fn(url, dst):
                size = os.path.getsize(dst) / 1024 / 1024
                print(f" 完成 ({size:.1f} MB)")
                return True
        print(" 失败")
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
    print(f"  ✗ 所有源均失败: {asset}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="fp16", choices=PRECISIONS)
    ap.add_argument("--dir", default=None)
    ap.add_argument("--check", action="store_true", help="只检查不下载")
    ap.add_argument("--source", default=None,
                    choices=list(SOURCES),
                    help="只用一个下载源（默认 魔搭→GitHub 依次尝试）")
    ap.add_argument("--with-torch", action="store_true",
                    help="额外下载 torch 后端所需的 .ckpt/.pth（约 229MB，"
                         "ONNX 后端用不到）")
    ap.add_argument("--torch-only", action="store_true",
                    help="只下 torch 后端所需（辅助包 + .ckpt/.pth），"
                         "完全跳过 ONNX（省 1.4GB）")
    a = ap.parse_args()
    if a.torch_only:
        a.with_torch = True

    order = (a.source,) if a.source else DEFAULT_ORDER
    d = target_dir(a.dir)
    print(f"目标目录: {d}")
    print(f"精度    : {a.precision}")
    print(f"下载源  : {' → '.join(order)}")

    need = [f for f in ONNX_FILES if not os.path.exists(os.path.join(d, f))]
    aux_ok = os.path.exists(os.path.join(d, "G2PWModel", "g2pW.onnx")) and \
        os.path.exists(os.path.join(d, "chinese-roberta-wwm-ext-large",
                                    "tokenizer.json"))
    ref_ok = os.path.exists(os.path.join(d, "ref", "default.wav"))
    torch_dir = os.path.join(d, "torch_weights")
    torch_need = [k for k, v in TORCH_ASSETS.items()
                  if not os.path.exists(os.path.join(torch_dir,
                                                     os.path.basename(v)))]

    if a.check:
        mode = "torch-only" if a.torch_only else a.precision
        print(f"\n模式  : {mode}")
        if not a.torch_only:
            print(f"ONNX  : {len(ONNX_FILES) - len(need)}/{len(ONNX_FILES)} 就绪")
        print(f"辅助  : {'就绪' if aux_ok else '缺失'}")
        print(f"参考音: {'就绪' if ref_ok else '缺失'}")
        print(f"torch : {len(TORCH_ASSETS) - len(torch_need)}/{len(TORCH_ASSETS)} 就绪"
              + ("" if a.with_torch else "（未请求）"))
        if not a.torch_only and need:
            print("缺失 ONNX: " + ", ".join(need))
        if a.with_torch and torch_need:
            print("缺失 torch: " + ", ".join(torch_need))
        ready = aux_ok and ref_ok
        if not a.torch_only:
            ready = ready and not need
        if a.with_torch:
            ready = ready and not torch_need
        return 0 if ready else 1

    os.makedirs(d, exist_ok=True)
    ok = True

    # 1) 辅助文件包（小）
    if not aux_ok or not ref_ok:
        z = os.path.join(d, "_aux.zip")
        if fetch("zfh-voice-aux.zip", z, 2, order):
            print("  解压辅助文件 ...")
            with zipfile.ZipFile(z) as f:
                f.extractall(d)
            os.remove(z)
        else:
            ok = False
    else:
        print("\n辅助文件已就绪")

    # 2) ONNX 模型（--torch-only 时跳过）
    if a.torch_only:
        print("\n--torch-only：跳过 ONNX 模型")
    elif need:
        print(f"\n需下载 {len(need)} 个模型（{a.precision}）")
        for f in need:
            tmp = os.path.join(d, f + ".part")
            if not fetch(f"{a.precision}__{f}", tmp, order=order):
                ok = False
                continue
            os.replace(tmp, os.path.join(d, f))
    else:
        print("\nONNX 模型已就绪")

    # 3) g2pW.onnx 也需存在于 G2PWModel/（文本前端按此路径加载）
    g_src = os.path.join(d, "g2pW.onnx")
    g_dst_dir = os.path.join(d, "G2PWModel")
    g_dst = os.path.join(g_dst_dir, "g2pW.onnx")
    if os.path.exists(g_src) and not os.path.exists(g_dst):
        os.makedirs(g_dst_dir, exist_ok=True)
        try:
            os.link(g_src, g_dst)                 # 同盘硬链接，不占额外空间
            print("  g2pW.onnx → G2PWModel/ (硬链接)")
        except OSError:
            import shutil
            shutil.copy2(g_src, g_dst)
            print("  g2pW.onnx → G2PWModel/ (复制)")

    # 4) torch 后端权重（可选）
    if a.with_torch:
        print(f"\n下载 torch 权重（{len(torch_need)} 个待下）")
        os.makedirs(torch_dir, exist_ok=True)
        for asset, rel in TORCH_ASSETS.items():
            dst = os.path.join(torch_dir, os.path.basename(rel))
            if os.path.exists(dst):
                print(f"  已存在，跳过: {os.path.basename(dst)}")
                continue
            tmp = dst + ".part"
            if fetch(asset, tmp, 150, order):
                os.replace(tmp, dst)
            else:
                ok = False
        if not torch_need:
            print("  已全部就绪")

    # 5) 校验（按所选模式，只校验实际需要的东西）
    print("\n=== 校验 ===")
    missing = []
    if not a.torch_only:
        for f in ONNX_FILES:
            p = os.path.join(d, f)
            if os.path.exists(p):
                print(f"  OK   {f:<28}{os.path.getsize(p)/1024/1024:>8.1f} MB")
            else:
                print(f"  MISS {f}")
                missing.append(f)
    if a.with_torch:
        for asset, rel in TORCH_ASSETS.items():
            p = os.path.join(torch_dir, os.path.basename(rel))
            if os.path.exists(p):
                print(f"  OK   torch/{os.path.basename(rel):<22}"
                      f"{os.path.getsize(p)/1024/1024:>8.1f} MB")
            else:
                print(f"  MISS torch/{os.path.basename(rel)}")
                missing.append(os.path.basename(rel))
    # 辅助文件（两种后端都需要：文本前端 + 默认参考音频）
    for f in ("G2PWModel/g2pW.onnx",
              "chinese-roberta-wwm-ext-large/tokenizer.json",
              "ref/default.wav"):
        p = os.path.join(d, f.replace("/", os.sep))
        if os.path.exists(p):
            print(f"  OK   {f}")
        else:
            print(f"  MISS {f}")
            missing.append(f)

    if missing:
        print(f"\n仍缺 {len(missing)} 个文件")
        ok = False
    else:
        if a.torch_only:
            print("\ntorch 模式就绪。使用方式：")
            print("  python -m zfh_voice --backend torch \\")
            print("      --gsv-root <GPT-SoVITS 检出目录> say \"今天天气不错\"")
            print(f"\n权重位于 {torch_dir}\\，torch 后端会自动在此查找。")
        else:
            print("\n全部就绪。可以用了：")
            print('  python -m zfh_voice say "今天天气不错"')
            if a.with_torch:
                print("\ntorch 权重已下到 torch_weights/，torch 后端会自动查找；")
                print("也可用 --backend torch --gsv-root <目录> 手动指定。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
