# -*- coding: utf-8 -*-
"""下载 dsh-zhuang-fangyi-voice-v2 的 GPT-SoVITS v5 模型包。

默认下载 GPT + e32 + 三种参考音频。ONNX/v2Pro 由 v1 插件提供，这里不再下载。
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MS_REPO = "yangwenjie1231/dsh-zhuang-fangyi-voice-v2-model"
GH_REPO = "yangwenjie1231/dsh-zhuang-fangyi-voice-v2"
GH_TAG = "model-v2-v5-fullft"
MS_BASE = f"https://modelscope.cn/models/{MS_REPO}/resolve/master"
GH_BASE = f"https://github.com/{GH_REPO}/releases/download/{GH_TAG}"

WEIGHTS = {
    8: "v5_full_e32_final_e8_s432_l0.pth",
    16: "v5_full_e32_final_e16_s864_l0.pth",
    24: "v5_full_e32_final_e24_s1296_l0.pth",
    32: "v5_full_e32_final_e32_s1728_l0.pth",
}
REFS = [
    "ref/default.wav", "ref/default.txt",
    "ref/styles/normal.wav", "ref/styles/normal.txt",
    "ref/styles/firm.wav", "ref/styles/firm.txt",
    "ref/styles/gentle.wav", "ref/styles/gentle.txt",
]


def target_dir(explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    if os.environ.get("ZFH_MODEL_DIR"):
        return os.path.abspath(os.environ["ZFH_MODEL_DIR"])
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def _get(url, dst):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        if os.path.getsize(tmp) == 0:
            raise OSError("empty response")
        os.replace(tmp, dst)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    失败: {type(exc).__name__}: {exc}")
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def fetch(rel, dst, order):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  已存在: {os.path.basename(dst)}")
        return True
    name = os.path.basename(rel)
    for source in order:
        url = f"{MS_BASE}/{rel}" if source == "modelscope" else f"{GH_BASE}/{name}"
        print(f"  [{source}] {name}")
        if _get(url, dst):
            return True
    return False


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(d):
    path = os.path.join(d, "distribution_manifest.json")
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    for item in data.get("weights", []):
        p = os.path.join(d, "torch_weights", "sovits", item["file"])
        if os.path.exists(p):
            ok = os.path.getsize(p) == item["size"] and sha256(p) == item["sha256"]
            print(f"  {'OK' if ok else 'FAIL'} {item['file']}")
            if not ok:
                raise SystemExit(f"校验失败: {item['file']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, choices=sorted(WEIGHTS), default=32)
    ap.add_argument("--all-epochs", action="store_true")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--source", choices=["modelscope", "github"], default=None)
    a = ap.parse_args()
    order = [a.source] if a.source else ["modelscope", "github"]
    d = target_dir(a.dir)
    epochs = list(WEIGHTS) if a.all_epochs else [a.epoch]

    print(f"目标目录: {d}")
    print("版本    : GPT-SoVITS v5dev")
    print(f"epoch   : {epochs}")
    print(f"下载源  : {' → '.join(order)}")

    if a.check:
        for ep in epochs:
            p = os.path.join(d, "torch_weights", "sovits", WEIGHTS[ep])
            print(f"  {'OK' if os.path.exists(p) else 'MISS'} e{ep}")
        gpt = os.path.join(d, "torch_weights", "zfh-e4.ckpt")
        print(f"  {'OK' if os.path.exists(gpt) else 'MISS'} GPT")
        return 0

    ok = True
    ok &= fetch("torch_weights/zfh-e4.ckpt",
                os.path.join(d, "torch_weights", "zfh-e4.ckpt"), order)
    for ep in epochs:
        rel = f"torch_weights/sovits/{WEIGHTS[ep]}"
        ok &= fetch(rel, os.path.join(d, *rel.split("/")), order)
    for rel in REFS:
        ok &= fetch(rel, os.path.join(d, *rel.split("/")), order)
    for name in ("configuration.json", "distribution_manifest.json", "SHA256SUMS.txt"):
        fetch(name, os.path.join(d, name), order)

    zip_path = os.path.join(d, "dsh-zhuang-fangyi-voice-v2-resources.zip")
    if not os.path.exists(os.path.join(d, "ref", "styles", "firm.wav")):
        if fetch("dsh-zhuang-fangyi-voice-v2-resources.zip", zip_path, ["github"]):
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(d)
            os.remove(zip_path)

    verify_manifest(d)
    if not ok:
        raise SystemExit("部分文件下载失败")
    print("\n完成。运行：")
    print(f'  python -m zfh_voice2 --backend torch --model-dir "{d}" '
          '--gsv-root <v5 GPT-SoVITS> say "你好"')


if __name__ == "__main__":
    main()
