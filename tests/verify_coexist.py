# -*- coding: utf-8 -*-
"""并存验证：v1 与 v2 **确实互不干扰**。

## 为什么必须验证
改标识（服务名/端口/包名/id）很容易漏，漏了就会：
    · 两版抢同一个端口 -> 后启动的失败
    · 两版挂同名服务 -> 宿主报冲突
    · 两版共享缓存目录 -> 风格串味
    · 两版用同一个 Python 包 -> 互相遮蔽

## 检查项
    ① **端口不同**（v1=8765, v2=8766）
    ② **服务名不同**（v1=`zfhVoice`, v2=`zfhVoiceV2`）
    ③ **Loader id 不同**
    ④ **包名不同**
    ⑤ **Python 包名不同**，且**各自能独立导入**
    ⑥ **缓存目录不同**
    ⑦ **两版同时起服务**都能健康（真实起进程）

## 用法
    python tests/verify_coexist.py            # 静态检查
    python tests/verify_coexist.py --live     # 额外真实起两个服务
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V1 = Path(r"D:\A\dsh-zhuang-fangyi-voice")
V2 = Path(r"D:\A\dsh-zhuang-fangyi-voice-v2")

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'✓' if ok else '✗'} {name}" + (f"  ({detail})" if detail else ""))


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="额外真实起两个服务验证端口不冲突")
    ap.add_argument("--model-dir", default=r"D:\A\voiceclone\zfh-models",
                    help="--live 时传给两个服务的模型目录（须含 torch 权重）")
    ap.add_argument("--gsv-root", default=r"D:\A\voiceclone\GPT-SoVITS",
                    help="--live 时传给两个服务的 GPT-SoVITS 检出目录")
    args = ap.parse_args()

    print("=" * 88)
    print("并存验证：v1 与 v2 互不干扰")
    print("=" * 88)

    i1 = read(V1 / "index.js")
    i2 = read(V2 / "index.js")
    c1 = read(V1 / "cordis.patch.yml")
    c2 = read(V2 / "cordis.patch.yml")
    p1 = json.loads(read(V1 / "package.json") or "{}")
    p2 = json.loads(read(V2 / "package.json") or "{}")

    def grep(s, rx, d=""):
        m = re.search(rx, s)
        return m.group(1) if m else d

    # ---------- ① 端口 ----------
    print()
    print("① HTTP 端口")
    port1 = grep(i1, r"port:\s*(\d+)")
    port2 = grep(i2, r"port:\s*(\d+)")
    check(f"v1 端口 = 8765", port1 == "8765", port1)
    check(f"v2 端口 = 8766", port2 == "8766", port2)
    check("两版端口**不同**", port1 != port2, f"{port1} vs {port2}")

    # ---------- ② 服务名 ----------
    print()
    print("② 服务名")
    s1 = grep(i1, r"SERVICE_NAME\s*=\s*'([^']+)'")
    s2 = grep(i2, r"SERVICE_NAME\s*=\s*'([^']+)'")
    check("v1 服务名 = zfhVoice", s1 == "zfhVoice", s1)
    check("v2 服务名 = zfhVoiceV2", s2 == "zfhVoiceV2", s2)
    check("两版服务名**不同**", s1 != s2, f"{s1} vs {s2}")
    # v2 应挂别名
    alias = "zfhVoice2" in i2
    check("v2 挂了短别名 zfhVoice2", alias)

    # ---------- ③ Loader id ----------
    print()
    print("③ Loader 条目 id")
    id1 = grep(c1, r"-\s*id:\s*(\S+)")
    id2 = grep(c2, r"-\s*id:\s*(\S+)")
    check("v1 id = zhuang-fangyi-voice", id1 == "zhuang-fangyi-voice", id1)
    check("v2 id = zhuang-fangyi-voice-v2",
          id2 == "zhuang-fangyi-voice-v2", id2)
    check("两版 id **不同**", id1 != id2, f"{id1} vs {id2}")

    # ---------- ④ 包名 ----------
    print()
    print("④ npm 包名")
    n1, n2 = p1.get("name", ""), p2.get("name", "")
    check("v1 包名 = dsh-zhuang-fangyi-voice",
          n1 == "dsh-zhuang-fangyi-voice", n1)
    check("v2 包名 = dsh-zhuang-fangyi-voice-v2",
          n2 == "dsh-zhuang-fangyi-voice-v2", n2)
    check("两版包名**不同**", n1 != n2, f"{n1} vs {n2}")

    # ---------- ⑤ Python 包 ----------
    print()
    print("⑤ Python 包")
    d1 = V1 / "src" / "zfh_voice"
    d2 = V2 / "src" / "zfh_voice2"
    check("v1 有 src/zfh_voice/", d1.is_dir())
    check("v2 有 src/zfh_voice2/", d2.is_dir())
    check("v2 **没有** src/zfh_voice/（避免遮蔽）",
          not (V2 / "src" / "zfh_voice").exists())
    # 各自独立导入
    for tag, root, pkg in (("v1", V1, "zfh_voice"), ("v2", V2, "zfh_voice2")):
        r = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, r'{root / 'src'}');"
             f"import {pkg} as m; print(m.__version__)"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60)
        ok = r.returncode == 0 and r.stdout.strip() != ""
        check(f"{tag} 的 {pkg} 能独立导入", ok,
              r.stdout.strip() or (r.stderr or "")[-80:])

    # ---------- ⑥ 缓存目录 ----------
    print()
    print("⑥ 缓存目录")
    cd1 = grep(read(V1 / "src" / "zfh_voice" / "paths.py"),
               r'"(\.cache[^"]*)"', "")
    cd2 = grep(read(V2 / "src" / "zfh_voice2" / "paths.py"),
               r'"(\.cache[^"]*)"', "")
    # 直接从常量里找
    def cache_name(p: Path) -> str:
        s = read(p)
        m = re.search(r"zfh-voice2?", s)
        return m.group(0) if m else ""
    c1n = cache_name(V1 / "src" / "zfh_voice" / "paths.py")
    c2n = cache_name(V2 / "src" / "zfh_voice2" / "paths.py")
    check("两版缓存目录名**不同**", c1n != c2n, f"{c1n or '(无)'} vs {c2n or '(无)'}")

    # ---------- ⑦ 真实起服务（可选） ----------
    if args.live:
        print()
        print("⑦ 真实起两个服务（验证端口不冲突）")
        # ⚠️ 必须显式 `--backend torch` + `--model-dir` + `--gsv-root`：
        #    · serve 默认后端是 onnx，而本机模型目录**只有 torch 权重**
        #      -> 不传会因缺 7 个 ONNX 直接退出（实测）
        #    · torch 后端**还要求 `--gsv-root`**（GPT-SoVITS 检出目录），
        #      否则报"torch 后端需要 --gsv-root 指定 GPT-SoVITS 检出目录"（实测）
        #    · `--no-preload` 让服务秒起（不等模型加载），
        #      这样测的是"端口能不能同时占住"而非"模型能不能加载"
        model_dir = args.model_dir
        gsv_root = args.gsv_root
        procs = []
        for tag, root, pkg, port in (
                ("v1", V1, "zfh_voice", 8765),
                ("v2", V2, "zfh_voice2", 8766)):
            cmd = [sys.executable, "-m", pkg,
                   "--backend", "torch",
                   "--model-dir", model_dir,
                   "--gsv-root", gsv_root,
                   "serve", "--port", str(port), "--no-preload"]
            env = {**__import__("os").environ,
                   "PYTHONPATH": str(root / "src"),
                   "PYTHONUTF8": "1"}
            pr = subprocess.Popen(cmd, cwd=str(root), env=env,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            procs.append((tag, pr, port))
        time.sleep(8)
        import urllib.request
        for tag, pr, port in procs:
            alive = pr.poll() is None
            ok = False
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    body = json.loads(r.read().decode("utf-8"))
                    ok = body.get("ok") is True
            except Exception:  # noqa: BLE001
                ok = False
            check(f"{tag} 在 {port} 上健康", alive and ok,
                  "进程存活+健康" if (alive and ok) else
                  f"alive={alive} health={ok}")
        # 关键：两个**同时**健康才算不冲突
        if all(p.poll() is None for _, p, _ in procs):
            check("两版**同时**存活（端口不冲突）", True, "8765 与 8766 并存")
        else:
            check("两版**同时**存活（端口不冲突）", False,
                  "有进程退出，可能端口冲突")
        for _, pr, _ in procs:
            try:
                pr.terminate()
                pr.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass

    # ---------- 汇总 ----------
    print()
    print("=" * 88)
    if FAIL:
        print(f"  ✗ 失败 {len(FAIL)} 项：")
        for f in FAIL:
            print(f"      {f}")
    print(f"  通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    print("=" * 88)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
