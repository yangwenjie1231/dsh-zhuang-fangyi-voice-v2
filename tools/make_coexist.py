# -*- coding: utf-8 -*-
"""把 v2 改成**与 v1 并存**模式。

## 为什么要改
v1 与 v2 的以下标识**完全相同**，同时启用会冲突：

    | 项 | v1 | v2（改前） |
    |---|---|---|
    | Loader 条目 id | `zhuang-fangyi-voice` | **相同** ✗ |
    | 包名 | `dsh-zhuang-fangyi-voice` | **相同** ✗ |
    | 服务名 | `zfhVoice` | **相同** ✗ |
    | HTTP 端口 | 8765 | **相同** ✗ |
    | Python 包 | `zfh_voice` | **相同** ✗ |
    | PYTHONPATH | `<repo>/src` | **相同**（但 repo 不同，OK） |

## 改法（**最小侵入，且保持"旧 API 兼容"**）

    ① **服务名**：`zfhVoice` -> `zfhVoiceV2`，但**同时**用 `ctx.provide`
       的别名机制再挂一个 `zfhVoice`？—— **不行**，v1 已占用会冲突。

       正确做法：v2 **只挂 `zfhVoiceV2`**。桌宠想用 v2 就 `ctx.get('zfhVoiceV2')`。
       **v1 的 `zfhVoice` 完全不受影响** —— 这才是"并存"。

       ⚠️ 但这会让 v2 **不再兼容**"桌宠直接用 zfhVoice"的用法。
       折中：v2 同时挂 `zfhVoiceV2`（主）和 `zfhVoice2`（短别名），
       并在文档里说明。**不抢 `zfhVoice`**。

    ② **Loader id / 包名**：加 `-v2` 后缀
    ③ **端口**：8766（避开 v1 的 8765）
    ④ **Python 包名**：`zfh_voice` -> `zfh_voice2`
       （否则两个插件各自 `PYTHONPATH=<自己>/src` 时会互相遮蔽）
    ⑤ **缓存目录**：加 `-v2` 后缀，避免两版共享缓存（风格不同会串味）
    ⑥ **CLI prog**：`zfh-voice` -> `zfh-voice2`

## 不改的（保持兼容）
    · HTTP 端点路径（`/tts` `/health` ...）—— 端口已经分开了
    · `ctx.get('zfhVoiceV2')` 的方法签名 —— 与 v1 的 `zfhVoice` 逐项一致
    · 配置文件字段名

## 用法
    python tools/make_coexist.py --dry-run
    python tools/make_coexist.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"D:\A\dsh-zhuang-fangyi-voice-v2")

# 服务名映射
SERVICE_OLD = "zfhVoice"
SERVICE_NEW = "zfhVoiceV2"
SERVICE_ALIAS = "zfhVoice2"      # 短别名，方便桌宠
PORT_OLD = 8765
PORT_NEW = 8766
PKG_OLD = "zfh_voice"
PKG_NEW = "zfh_voice2"
ID_OLD = "zhuang-fangyi-voice"
ID_NEW = "zhuang-fangyi-voice-v2"
NPM_OLD = "dsh-zhuang-fangyi-voice"
NPM_NEW = "dsh-zhuang-fangyi-voice-v2"
PROG_OLD = "zfh-voice"
PROG_NEW = "zfh-voice2"


def sub_file(p: Path, pairs: list[tuple[str, str]], dry: bool) -> int:
    """按顺序做字面替换，返回替换总数。"""
    if not p.exists():
        return -1
    s = p.read_text(encoding="utf-8", errors="replace")
    n = 0
    for a, b in pairs:
        c = s.count(a)
        if c:
            s = s.replace(a, b)
            n += c
    if n and not dry:
        p.write_text(s, encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    print("=" * 92)
    print("把 v2 改成与 v1 并存")
    print("=" * 92)
    print(f"  服务名   {SERVICE_OLD} -> {SERVICE_NEW}（+别名 {SERVICE_ALIAS}）")
    print(f"  端口     {PORT_OLD} -> {PORT_NEW}")
    print(f"  Python包 {PKG_OLD} -> {PKG_NEW}")
    print(f"  Loader   {ID_OLD} -> {ID_NEW}")
    print(f"  包名     {NPM_OLD} -> {NPM_NEW}")
    print(f"  CLI prog {PROG_OLD} -> {PROG_NEW}")
    print()

    # ---------- ① 重命名 Python 包目录 ----------
    src_old = ROOT / "src" / PKG_OLD
    src_new = ROOT / "src" / PKG_NEW
    print("① Python 包目录")
    if src_new.exists() and not src_old.exists():
        print(f"   ~ 已是 {PKG_NEW}/")
    elif src_old.exists():
        if dry:
            print(f"   (dry) src/{PKG_OLD}/ -> src/{PKG_NEW}/")
        else:
            shutil.move(str(src_old), str(src_new))
            print(f"   ✓ src/{PKG_OLD}/ -> src/{PKG_NEW}/")
    else:
        print(f"   !! 找不到 src/{PKG_OLD}/")
        return 1

    # ---------- ② 替换所有引用 ----------
    print()
    print("② 替换引用")
    targets = []
    for pat in ("*.py", "*.js", "*.mjs", "*.yml", "*.yaml", "*.json",
                "*.md", "*.toml", "*.txt", "*.cfg", "*.ini"):
        targets.extend(ROOT.rglob(pat))
    # 排除 _vendor（第三方代码，不能改）、.git、**以及本脚本自己**
    # （否则第二次运行时脚本会把自己的常量也替换掉，自我破坏）
    SELF = Path(__file__).resolve()
    targets = [p for p in targets
               if "_vendor" not in p.parts and ".git" not in p.parts
               and p.resolve() != SELF]

    PAIRS = [
        # Python 包名（含 `from .` 相对导入不受影响）
        (f"zfh_voice.", f"{PKG_NEW}."),
        (f"zfh_voice ", f"{PKG_NEW} "),
        (f"from zfh_voice import", f"from {PKG_NEW} import"),
        (f"import zfh_voice", f"import {PKG_NEW}"),
        (f"-m zfh_voice", f"-m {PKG_NEW}"),
        (f"src/zfh_voice", f"src/{PKG_NEW}"),
        (f"src\\\\zfh_voice", f"src\\\\{PKG_NEW}"),
        (f"zfh_voice/", f"{PKG_NEW}/"),
        # 服务名
        ("'zfhVoice'", f"'{SERVICE_NEW}'"),
        ('"zfhVoice"', f'"{SERVICE_NEW}"'),
        ("`zfhVoice`", f"`{SERVICE_NEW}`"),
        ("ctx.get('zfhVoice')", f"ctx.get('{SERVICE_NEW}')"),
        # 端口
        ("8765", str(PORT_NEW)),
        # Loader id / 包名
        (f"id: {ID_OLD}", f"id: {ID_NEW}"),
        (f"'{NPM_OLD}'", f"'{NPM_NEW}'"),
        (f'"{NPM_OLD}"', f'"{NPM_NEW}"'),
        # CLI prog
        (f'prog="{PROG_OLD}"', f'prog="{PROG_NEW}"'),
        (f"{PROG_OLD} ", f"{PROG_NEW} "),
    ]

    total = 0
    changed = []
    for p in targets:
        try:
            n = sub_file(p, PAIRS, dry)
        except Exception as e:  # noqa: BLE001
            print(f"   !! {p.name}: {type(e).__name__}")
            continue
        if n > 0:
            total += n
            changed.append((p.relative_to(ROOT).as_posix(), n))
    changed.sort(key=lambda t: -t[1])
    print(f"   共 {total} 处替换，涉及 {len(changed)} 个文件")
    for name, n in changed[:14]:
        print(f"     {n:4d}  {name}")

    # ---------- ③ 加服务别名 ----------
    print()
    print("③ 加服务别名（zfhVoice2）")
    idx = ROOT / "index.js"
    s = idx.read_text(encoding="utf-8")
    if SERVICE_ALIAS not in s:
        # 在 provide 之后补一个别名挂载
        m = re.search(r"(\n\s*ctx\.provide\([^\n]*\)\n)", s)
        if m:
            anchor = m.group(1)
            alias = (f"\n  // v0.2.0 并存模式：再挂一个短别名，方便桌宠。\n"
                     f"  // **不抢 `zfhVoice`** —— 那是 v1 的，两版并存互不干扰。\n"
                     f"  try {{\n"
                     f"    ctx.provide('{SERVICE_ALIAS}', service)\n"
                     f"  }} catch {{ /* 别名已被占用，忽略 */ }}\n")
            s = s.replace(anchor, anchor + alias, 1)
            if not dry:
                idx.write_text(s, encoding="utf-8")
            print(f"   ✓ 已加别名 {SERVICE_ALIAS}")
        else:
            print("   !! 找不到 ctx.provide 锚点")
    else:
        print(f"   ~ 别名已存在")

    print()
    print("=" * 92)
    print("完成" if not dry else "(dry-run)")
    print("=" * 92)
    if not dry:
        print(f"""
  ## 并存后的布局

  | | v1 | v2 |
  |---|---|---|
  | Loader id | `{ID_OLD}` | `{ID_NEW}` |
  | 包名 | `{NPM_OLD}` | `{NPM_NEW}` |
  | 服务 | `zfhVoice` | `{SERVICE_NEW}`（+ `{SERVICE_ALIAS}`） |
  | 端口 | {PORT_OLD} | {PORT_NEW} |
  | Python 包 | `{PKG_OLD}` | `{PKG_NEW}` |
  | CLI | `{PROG_OLD}` | `{PROG_NEW}` |

  ## 桌宠怎么用两个

  ```js
  const v1 = ctx.get('zfhVoice')      // 旧版（不变）
  const v2 = ctx.get(f'zfhVoiceV2')   // 新版（三种风格）
  const a = await v1.synthesize('你好')
  const b = await v2.synthesize('你好', 'firm')
  ```

  ## ⚠️ 需要改的地方
    · 桌宠设置页如果要支持 v2，需改成读 `zfhVoiceV2`
    · v2 的 HTTP 端口是 {PORT_NEW}（桌宠的 `ttsUrl` 若要指 v2 需改）
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
