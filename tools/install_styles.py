# -*- coding: utf-8 -*-
"""把三种语气参考音频安装到模型目录（v2 部署步骤）。

## 为什么需要
v2 的 `styles.py` 会在模型目录里找：
    ref/styles/normal.wav
    ref/styles/firm.wav
    ref/styles/gentle.wav

而现有模型目录（`D:\\A\\voiceclone\\zfh-models*`）只有 `ref/default.wav`。

## 源
`D:\\A\\zfy-v5\\GPT-SoVITS\\ref\\styles\\`（已固化并上传 GitHub/ModelScope）

## 做法
    1. 备份现有 `ref/default.wav` / `ref/default.txt`（首次）
    2. 复制三种风格到 `ref/styles/`
    3. **同时把 normal 写回 `ref/default.wav`** —— 保证 v1 行为不变
    4. 验证：包络互相关确认三段互不相同

## 用法
    python install_styles.py                    # 装到默认的两个模型目录
    python install_styles.py --model-dir <路径>  # 指定目录
    python install_styles.py --dry-run
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

STYLE_SRC = Path(r"D:\A\zfy-v5\GPT-SoVITS\ref\styles")
DEFAULT_MODEL_DIRS = [
    Path(r"D:\A\voiceclone\zfh-models"),
    Path(r"D:\A\voiceclone\zfh-models-v2pro"),
]

# 风格 -> (源文件名, 参考文本)
STYLES = {
    "normal": ("normal.wav",
               "再好的武器，也得朝夕相处磨合上几天。等用惯了，我再和你说说感受，好吗？"),
    "firm": ("firm.wav",
             "北部禁区才是应龙特勤队的任务目标,再者,他们人手太少.不过他们也承诺过尽可能支援武陵."),
    "gentle": ("gentle.wav",
               "我问了自己相同的问题,之后才终于下定决心提出申请…"),
}


def envelope(p: Path, sr_t: int = 16000, frame: float = 0.02):
    try:
        x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    except Exception:  # noqa: BLE001
        return None
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != sr_t:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), sr_t)
        x = resample_poly(x, sr_t // g, int(sr) // g)
    n = int(frame * sr_t)
    m = (len(x) // n) * n
    if m < n * 4:
        return None
    e = np.sqrt((x[:m].reshape(-1, n) ** 2).mean(axis=1))
    e = e - e.mean()
    nn = np.linalg.norm(e)
    return e / nn if nn > 1e-9 else None


def xcorr(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    if len(a) > len(b):
        a, b = b, a
    if len(a) < 4:
        return float("nan")
    L = len(a)
    cs = np.concatenate([[0.0], np.cumsum(b ** 2)])
    wn = np.sqrt(np.maximum(cs[L:] - cs[:-L], 1e-18))
    nfft = 1 << (len(a) + len(b) - 1).bit_length()
    c = np.fft.irfft(np.fft.rfft(b, nfft) * np.conj(np.fft.rfft(a, nfft)),
                     nfft)[:len(b) - L + 1]
    return float(np.abs(c / wn).max())


def install(model_dir: Path, dry: bool) -> bool:
    print()
    print("=" * 88)
    print(f"安装到: {model_dir}")
    print("=" * 88)
    if not model_dir.is_dir():
        print("  ✗ 目录不存在，跳过")
        return False

    ref = model_dir / "ref"
    styles_dir = ref / "styles"
    if not dry:
        styles_dir.mkdir(parents=True, exist_ok=True)

    # 备份
    for f in ("default.wav", "default.txt"):
        src = ref / f
        bak = ref / f"{f}.bak-v1"
        if src.exists() and not bak.exists():
            if dry:
                print(f"  (dry) 会备份 {f} -> {bak.name}")
            else:
                shutil.copy2(src, bak)
                print(f"  ✓ 备份 {f} -> {bak.name}")

    ok = True
    for name, (fname, text) in STYLES.items():
        src = STYLE_SRC / fname
        if not src.exists():
            print(f"  ✗ 缺源文件 {src}")
            ok = False
            continue
        dst = styles_dir / fname
        if dry:
            print(f"  (dry) {fname}  ->  ref/styles/{fname}  "
                  f"({src.stat().st_size/1024:.0f} KB)")
        else:
            shutil.copy2(src, dst)
            (styles_dir / f"{name}.txt").write_text(text + "\n",
                                                    encoding="utf-8")
            print(f"  ✓ ref/styles/{fname}  ({src.stat().st_size/1024:.0f} KB)")

    # normal 写回 default.wav（保证 v1 行为不变）
    if not dry:
        nw = styles_dir / "normal.wav"
        if nw.exists():
            shutil.copy2(nw, ref / "default.wav")
            (ref / "default.txt").write_text(
                STYLES["normal"][1] + "\n", encoding="utf-8")
            print("  ✓ 已把 normal 写回 ref/default.wav（v1 兼容）")

    # 验证
    if not dry:
        print()
        print("  验证（包络互相关，应全部 < 0.8）:")
        envs = {}
        for name, (fname, _) in STYLES.items():
            envs[name] = envelope(styles_dir / fname)
        names = list(envs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                c = xcorr(envs[a], envs[b])
                flag = "✓" if c < 0.8 else "**⚠️ 过于相似**"
                print(f"    {a:8s} vs {b:8s} : {c:.3f}  {flag}")
        # normal == default ?
        same = (styles_dir / "normal.wav").read_bytes() == \
            (ref / "default.wav").read_bytes()
        print(f"    normal.wav == default.wav ? {same}  "
              f"{'✓' if same else '✗'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", action="append", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 88)
    print("安装三种语气参考音频到模型目录")
    print("=" * 88)
    print(f"  源: {STYLE_SRC}")
    if not STYLE_SRC.is_dir():
        print(f"  ✗ 源目录不存在")
        return 1
    for f in STYLES:
        p = STYLE_SRC / STYLES[f][0]
        print(f"    {f:8s} {p.name:14s} "
              f"{'✓ ' + str(round(p.stat().st_size/1024)) + ' KB' if p.exists() else '✗ 缺失'}")

    dirs = [Path(d) for d in args.model_dir] if args.model_dir \
        else DEFAULT_MODEL_DIRS
    n_ok = sum(1 for d in dirs if install(d, args.dry_run))

    print()
    print("=" * 88)
    print(f"完成: {n_ok}/{len(dirs)} 个目录")
    print("=" * 88)
    if not args.dry_run and n_ok:
        print("""
  ## 用法
      python -m zfh_voice2 say "你好" --style firm
      python -m zfh_voice2 status          # 会列出可用风格

  ## HTTP
      POST /tts  {"text": "...", "style": "firm"}
      GET  /voices                        # 列出风格
""")
    print("=" * 88)
    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
