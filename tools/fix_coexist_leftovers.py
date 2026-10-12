# -*- coding: utf-8 -*-
"""补齐并存转换的遗漏替换（用**词边界正则**，比字面替换可靠）。

## 为什么需要补
`make_coexist.py` 用的是**字面替换**，规则列得不全，导致漏掉：

    pyproject.toml   `"zfh_voice" = [`            （后面是 `"` 不是 `.`/空格）
    tests/*.py       `os.path.join(REPO, "src", "zfh_voice")`
    tools/*.mjs      `src/zfh_voice` 等变体
    index.js         若干

**漏掉的后果**：测试挂 6 个（包数据检查找不到目录）。

## 本脚本的做法
用**词边界正则** `\\bzfh_voice\\b(?!2)` 匹配所有"裸的 zfh_voice"，
一次替换干净。同样处理其它标识。

## 安全
    · 排除 `_vendor`（第三方）、`.git`、本脚本自己
    · 替换前后各打印一次计数，便于核对
    · `--dry-run` 可预览

## 用法
    python tools/fix_coexist_leftovers.py --dry-run
    python tools/fix_coexist_leftovers.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"D:\A\dsh-zhuang-fangyi-voice-v2")
SELF = Path(__file__).resolve()

# 词边界替换表：(正则, 替换)
# ⚠️ 用 `(?!2)` 负向断言，避免把已经是 `zfh_voice2` 的再改一次
RULES = [
    (r"\bzfh_voice\b(?!2)", "zfh_voice2"),
    (r"\bzfh-voice\b(?!2)", "zfh-voice2"),
    (r"\bzfhVoice\b(?!V2|2)", "zfhVoiceV2"),
    (r"\bdsh-zhuang-fangyi-voice\b(?!-v2)", "dsh-zhuang-fangyi-voice-v2"),
    (r"\bzhuang-fangyi-voice\b(?!-v2)", "zhuang-fangyi-voice-v2"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 92)
    print("补齐并存转换的遗漏")
    print("=" * 92)

    targets = []
    for pat in ("*.py", "*.js", "*.mjs", "*.yml", "*.yaml", "*.json",
                "*.md", "*.toml", "*.txt", "*.cfg", "*.ini"):
        targets.extend(ROOT.rglob(pat))
    # ⚠️ 排除要按**文件名**比，不能只比 resolve() ——
    #    实测 `Path(__file__).resolve()` 与 `rglob` 出来的路径
    #    在 Windows 上可能大小写不一致，导致本脚本改到自己。
    SKIP_NAMES = {"make_coexist.py", "fix_coexist_leftovers.py",
                  "build_readme.py"}
    targets = [p for p in targets
               if "_vendor" not in p.parts and ".git" not in p.parts
               and p.name not in SKIP_NAMES
               and p.resolve() != SELF]

    total = 0
    changed = []
    for p in targets:
        try:
            s = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        orig = s
        n = 0
        for rx, rep in RULES:
            s, k = re.subn(rx, rep, s)
            n += k
        if n:
            total += n
            changed.append((p.relative_to(ROOT).as_posix(), n))
            if not args.dry_run:
                p.write_text(s, encoding="utf-8")

    changed.sort(key=lambda t: -t[1])
    print(f"  共 {total} 处，{len(changed)} 个文件")
    for name, n in changed[:20]:
        print(f"    {n:4d}  {name}")

    # ---------- 验证：还有裸的 zfh_voice 吗 ----------
    print()
    print("=" * 92)
    print("验证：是否还有裸标识")
    print("=" * 92)
    left = {}
    for p in targets:
        try:
            s = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        c = 0
        for rx, _ in RULES:
            c += len(re.findall(rx, s))
        if c:
            left[p.relative_to(ROOT).as_posix()] = c
    if left:
        print("  ⚠️ 仍有残留:")
        for k, v in sorted(left.items(), key=lambda t: -t[1])[:12]:
            print(f"    {v:4d}  {k}")
    else:
        print("  ✓ 无残留")

    print()
    print("=" * 92)
    print("完成" if not args.dry_run else "(dry-run)")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
