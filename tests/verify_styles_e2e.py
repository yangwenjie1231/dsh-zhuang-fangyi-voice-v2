# -*- coding: utf-8 -*-
"""验证 v2 的三种风格**确实产出不同音频**（端到端）。

## 为什么必须验证
`styles.py` 只是把参考音频换了，但如果：
    · 缓存键没纳入参考音频 -> 三种风格返回**同一份缓存**
    · 后端没真正读 `ref_wav` -> 输出**完全相同**
那"三种风格"就是假的。

## 判据
    ① 三个 wav 的**字节**不同（md5）
    ② 三个 wav 的**时长**不同或接近
    ③ **F0 尾部斜率**不同（firm/gentle 应比 normal 更下压）
    ④ 谱质心不同

## 用法
    python tests/verify_styles_e2e.py --outdir out/styles_e2e
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR = 16000
TEXT = "这是一句测试，用来对比三种语气风格。"


def f0_slope(x, sr=SR, tail=0.5):
    """末段 F0 斜率（semi/s）。正=上扬，负=下压。"""
    n, h = int(0.04 * sr), int(0.01 * sr)
    if len(x) < n * 4:
        return float("nan")
    ts, f0s = [], []
    for i in range(0, len(x) - n, h):
        seg = x[i:i + n]
        if np.sqrt((seg ** 2).mean()) < 1e-3:
            continue
        z = (seg - seg.mean()) * np.hanning(n)
        ac = np.correlate(z, z, "full")[len(z) - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        lo, hi = int(sr / 350), min(int(sr / 70), len(ac) - 1)
        if hi <= lo:
            continue
        k = int(np.argmax(ac[lo:hi]))
        if ac[lo + k] < 0.3:
            continue
        ts.append(i / sr)
        f0s.append(sr / (lo + k))
    if len(ts) < 5:
        return float("nan")
    ts, f0s = np.array(ts), np.array(f0s)
    m = ts >= ts[-1] - tail
    if m.sum() < 3:
        return float("nan")
    s = 12 * np.log2(f0s[m] / np.median(f0s))
    return float(np.polyfit(ts[m] - ts[m][0], s, 1)[0])


def spec_feats(x, sr=SR):
    nfft = 2048
    acc = np.zeros(nfft // 2 + 1)
    cnt = 0
    for i in range(0, max(1, len(x) - nfft), nfft // 4):
        seg = x[i:i + nfft]
        if len(seg) < nfft or np.sqrt((seg ** 2).mean()) < 1e-4:
            continue
        acc += np.abs(np.fft.rfft(seg * np.hanning(nfft))) ** 2
        cnt += 1
    if cnt == 0:
        return {}
    f = np.fft.rfftfreq(nfft, 1 / sr)
    P = acc / (acc.sum() + 1e-20)
    return {"cent": float((f * P).sum()),
            "body": float(P[(f >= 150) & (f < 800)].sum()) * 100,
            "b500_800": float(P[(f >= 500) & (f < 800)].sum()) * 100}


def load16(p: Path):
    x, sr = sf.read(str(p), always_2d=True, dtype="float64")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(x, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/styles_e2e")
    ap.add_argument("--model-dir", default=r"D:\A\voiceclone\zfh-models")
    ap.add_argument("--gsv-root", default=r"D:\A\voiceclone\GPT-SoVITS")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-synth", action="store_true",
                    help="只分析已有 wav，不重新合成")
    args = ap.parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    styles = ["normal", "firm", "gentle"]
    print("=" * 88)
    print("三种风格端到端验证")
    print("=" * 88)
    print(f"  文本: {TEXT}")
    print(f"  输出: {outdir}")

    if not args.skip_synth:
        for st in styles:
            dst = outdir / f"{st}.wav"
            cmd = [args.python, "-m", "zfh_voice",
                   "--backend", "torch",
                   "--model-dir", args.model_dir,
                   "--gsv-root", args.gsv_root,
                   "--device", "cuda",
                   "--style", st,
                   "--no-cache",           # 关键：禁用缓存，确保真跑合成
                   "say", TEXT, "-o", str(dst)]
            print()
            print(f"  --- 合成 {st} ---")
            r = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               cwd=str(ROOT),
                               env={**__import__("os").environ,
                                    "PYTHONPATH": str(ROOT / "src"),
                                    "PYTHONUTF8": "1"})
            tail = [l for l in (r.stdout or "").split("\n")
                    if l.strip()][-2:]
            for l in tail:
                print(f"      {l.strip()[:90]}")
            if r.returncode != 0:
                print(f"      !! 失败: {(r.stderr or '')[-300:]}")

    # ---------- 分析 ----------
    print()
    print("=" * 88)
    print("分析")
    print("=" * 88)
    print(f"  {'风格':<8s} {'时长':>7s} {'md5':>10s} {'F0斜率':>9s} "
          f"{'谱质心':>8s} {'body':>8s} {'500-800':>9s}")
    print("  " + "-" * 74)
    rows = {}
    for st in styles:
        p = outdir / f"{st}.wav"
        if not p.exists():
            print(f"  {st:<8s} (缺文件)")
            continue
        md5 = hashlib.md5(p.read_bytes()).hexdigest()[:8]
        x = load16(p)
        ft = spec_feats(x)
        rows[st] = {"dur": len(x) / SR, "md5": md5,
                    "slope": f0_slope(x), **ft}
        r = rows[st]
        print(f"  {st:<8s} {r['dur']:6.2f}s {md5:>10s} {r['slope']:+9.2f} "
              f"{r['cent']:8.0f} {r['body']:7.2f}% {r['b500_800']:8.2f}%")

    if len(rows) < 2:
        print("\n  !! 数据不足")
        return 1

    print()
    print("=" * 88)
    print("判据")
    print("=" * 88)
    md5s = {r["md5"] for r in rows.values()}
    ok = True

    # ① 字节必须不同
    if len(md5s) == len(rows):
        print(f"  ✓ 三个 wav 的 md5 互不相同（{len(md5s)} 个）")
    else:
        print(f"  ✗ **有 md5 相同** —— 风格没有真正生效！")
        ok = False

    # ② F0 斜率应有差异
    slopes = {k: r["slope"] for k, r in rows.items()}
    valid = {k: v for k, v in slopes.items() if not np.isnan(v)}
    if len(valid) >= 2:
        spread = max(valid.values()) - min(valid.values())
        if spread > 1.0:
            print(f"  ✓ F0 尾部斜率有差异（跨度 {spread:.2f} semi/s）")
            for k, v in valid.items():
                print(f"      {k:8s} {v:+.2f}")
        else:
            print(f"  ~ F0 斜率差异小（{spread:.2f}）—— 可能句子太短")
    else:
        print("  ~ F0 斜率无法计算（句子太短？）")

    # ③ 谱质心应有差异
    cents = [r["cent"] for r in rows.values()]
    if max(cents) - min(cents) > 5:
        print(f"  ✓ 谱质心有差异（跨度 {max(cents)-min(cents):.0f} Hz）")
    else:
        print(f"  ~ 谱质心接近（跨度 {max(cents)-min(cents):.0f} Hz）")

    print()
    if ok:
        print("  ✓ **三种风格确实产出不同音频** —— 风格切换生效")
    else:
        print("  ✗ 风格切换**未生效**，需排查")
    print("=" * 88)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
