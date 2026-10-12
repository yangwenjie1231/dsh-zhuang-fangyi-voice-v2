# -*- coding: utf-8 -*-
"""修掉并存转换产生的 `zfh_voice22` 之类**双重替换**产物。

## 根因
`make_coexist.py` 用的是**字面替换**，规则之间有重叠：

    ("zfh_voice.",  "zfh_voice2.")
    ("zfh_voice ",  "zfh_voice2 ")
    ("-m zfh_voice","-m zfh_voice2")
    ...

同一段文本可能被**多条规则依次命中**。例如
    `python -m zfh_voice say`
先被 `("zfh_voice ", "zfh_voice2 ")` 改成
    `python -m zfh_voice2 say`
**这条规则本身不再命中**，但若另一条规则是 `("zfh_voice", "zfh_voice2")`
（无边界），它会把 `zfh_voice2` 里的 `zfh_voice` 再改一次 ->
    `python -m zfh_voice22 say`

**教训**：字面替换做多重改名时，必须用**词边界正则 + 负向断言**
（`\\bzfh_voice\\b(?!2)`），而不是并列的字面规则。

## 本脚本
把所有 `xxx22` / `xxx-v2-v2` 之类的产物收敛回单次后缀。

## 用法
    python tools/fix_double_suffix.py --dry-run
    python tools/fix_double_suffix.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"D:\A\dsh-zhuang-fangyi-voice-v2")

# 产物 -> 正确形式
RULES = [
    (r"zfh_voice2{2,}", "zfh_voice2"),
    (r"zfh-voice2{2,}", "zfh-voice2"),
    (r"zfhVoiceV?2{2,}", "zfhVoiceV2"),
    (r"zfhVoice2{2,}", "zfhVoice2"),
    (r"dsh-zhuang-fangyi-voice-v2(?:-v2)+", "dsh-zhuang-fangyi-voice-v2"),
    (r"zhuang-fangyi-voice-v2(?:-v2)+", "zhuang-fangyi-voice-v2"),
    (r"zfh-voice2{2,}", "zfh-voice2"),
    # ⚠️ 下面这条是**正常形式**，不是产物 —— 它只在"验证残留"时会被误判。
    #    实际含义：用户缓存目录 `~/.cache/zfh-voice2/models`。
    #    所以修复规则里保留它（幂等，无副作用），但验证时要排除。
    (r"zfh-voice2/models", "zfh-voice2/models"),
]

# 验证残留时**排除**这些"正常形式"（否则会把正确内容报成残留）
NOT_ARTIFACTS = [r"zfh-voice2/models"]

SKIP_NAMES = {"fix_double_suffix.py", "make_coexist.py",
              "fix_coexist_leftovers.py", "build_readme.py"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("=" * 88)
    print("修双重后缀产物")
    print("=" * 88)

    targets = []
    for pat in ("*.py", "*.js", "*.mjs", "*.yml", "*.yaml", "*.json",
                "*.md", "*.toml", "*.txt"):
        targets.extend(ROOT.rglob(pat))
    targets = [p for p in targets
               if "_vendor" not in p.parts and ".git" not in p.parts
               and p.name not in SKIP_NAMES]

    total, changed = 0, []
    for p in targets:
        try:
            s = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
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
    print(f"  修复 {total} 处，{len(changed)} 个文件")
    for name, n in changed[:16]:
        print(f"    {n:4d}  {name}")

    # 验证
    print()
    left = {}
    for p in targets:
        try:
            s = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        # 先剔除"正常形式"，再统计
        for ok in NOT_ARTIFACTS:
            s = re.sub(ok, "", s)
        c = sum(len(re.findall(rx, s)) for rx, _ in RULES)
        if c:
            left[p.relative_to(ROOT).as_posix()] = c
    if left:
        print("  ⚠️ 仍有残留:")
        for k, v in sorted(left.items(), key=lambda t: -t[1])[:10]:
            print(f"    {v:4d}  {k}")
    else:
        print("  ✓ 无残留")

    print()
    print("完成" if not args.dry_run else "(dry-run)")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
