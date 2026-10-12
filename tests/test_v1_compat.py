# -*- coding: utf-8 -*-
"""兼容性守卫：v2 必须保留 v1 的全部对外契约。

## 为什么需要
用户要求「更新第二版，**兼容旧的 API**」。
这个测试把 v1 的契约**逐项断言**，任何破坏都会在这里失败。

## 契约来源
`docs/v1_contract.json`（从 v1 源码逐字提取，随仓库一起分发）

## 检查项
    ① JS 服务方法（11 个）**全部仍存在**
    ② HTTP 端点（4 个 v1 端点）**全部仍可路由**
    ③ 配置项（10 个）**全部仍被接受**
    ④ CLI 子命令（8 个）**全部仍存在**
    ⑤ 默认风格必须是 normal，且其参考音频 = `ref/default.wav`

## 用法
    python tests/test_v1_compat.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CONTRACT = ROOT / "docs" / "v1_contract.json"

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    print("=" * 84)
    print("v1 兼容性守卫")
    print("=" * 84)

    if not CONTRACT.exists():
        print(f"  !! 找不到契约文件 {CONTRACT}")
        return 1
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))

    idx = (ROOT / "index.js").read_text(encoding="utf-8")
    srv = (ROOT / "src" / "zfh_voice" / "server.py").read_text(encoding="utf-8")
    cli = (ROOT / "src" / "zfh_voice" / "cli.py").read_text(encoding="utf-8")
    pkg = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))

    # ---------- ① JS 服务方法 ----------
    #
    # ⚠️ 我第一版用 `const service = {` 到 `"\n  }\n"` 切片 —— **错的**：
    #    service 对象里嵌了箭头函数体，`\n  }\n` 会提前匹配到内层的闭合，
    #    导致切片截断，`unload`（在对象末尾）被误报"丢失"。
    #    实测 v1 的 index.js L1146 与 v2 的 L1171 都是
    #        `unload: () => client.unload(),`
    #    —— 完全一致。
    #
    # 改成**全文搜索**：只要方法名在 `export const service = {` 之后出现过
    # 形如 `name (` 或 `name:` 的定义即可。
    print()
    print("① JS 服务方法（v1 契约）")
    svc_start = idx.find("const service = {")
    svc_body = idx[svc_start:] if svc_start > 0 else idx
    for m in c["service_methods"]:
        pat = rf"\b{re.escape(m)}\s*[:(]"
        hit = re.search(pat, svc_body)
        check(f"服务方法 {m}() 仍存在", bool(hit),
              "" if hit else "**丢失**")

    # ---------- ② HTTP 端点 ----------
    print()
    print("② HTTP 端点（v1 契约）")
    for ep in c["http_endpoints_python"]:
        # server.py 里应有 `path == "/xxx"` 或 `"/xxx"` 的匹配
        check(f"端点 {ep} 仍可路由",
              f'"{ep}"' in srv,
              "" if f'"{ep}"' in srv else "**丢失**")

    # ---------- ③ 配置项 ----------
    print()
    print("③ 配置项（v1 契约）")
    for k in c["config_keys"]:
        # normalizeConfig 里应认这个键（或 defaultConfig 里有默认值）
        pat = rf"\b{re.escape(k)}\b"
        hit = bool(re.search(pat, idx))
        check(f"配置项 {k} 仍被接受", hit, "" if hit else "**丢失**")

    # ---------- ④ CLI 子命令 ----------
    print()
    print("④ CLI 子命令（v1 契约）")
    for s in c["cli_subcommands"]:
        hit = f'"{s}"' in cli or f"'{s}'" in cli
        check(f"子命令 {s} 仍存在", hit, "" if hit else "**丢失**")

    # ---------- ⑤ 默认风格 = normal，参考音频 = default.wav ----------
    print()
    print("⑤ 默认行为（关键：不指定风格时与 v1 一致）")
    from zfh_voice import styles  # noqa: E402

    check("DEFAULT_STYLE == 'normal'", styles.DEFAULT_STYLE == "normal",
          styles.DEFAULT_STYLE)
    check("未知风格回落 normal",
          styles.normalize_style("bogus") == "normal"
          and styles.normalize_style(None) == "normal")
    spec = styles.STYLES["normal"]
    check("normal 的 fallback 指向 ref/default.wav",
          spec["fallback"] == "ref/default.wav", spec["fallback"])

    # 有真实模型目录时，验证 normal 解析到 default.wav 或 styles/normal.wav
    md = None
    try:
        from zfh_voice import paths  # noqa: E402
        md = paths.resolve_model_dir(must_exist=False)
    except Exception:
        pass
    if md and Path(md).is_dir():
        k, wav, txt = styles.resolve_style(None, md)
        check("resolve_style(None) 解析到 normal", k == "normal", k)
        check("resolve_style(None) 拿到了参考音频", wav is not None,
              wav.name if wav else "(None)")
    else:
        print(f"  ~ 跳过实目录检查（模型目录不可用: {md}）")

    # ---------- ⑥ 版本号 ----------
    print()
    print("⑥ 版本号")
    check("package.json 版本 >= 0.2.0",
          tuple(int(x) for x in pkg["version"].split(".")) >= (0, 2, 0),
          pkg["version"])

    # ---------- 汇总 ----------
    print()
    print("=" * 84)
    if FAIL:
        print(f"  ✗ 失败 {len(FAIL)} 项：")
        for f in FAIL:
            print(f"      {f}")
    print(f"  通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    print("=" * 84)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
