# -*- coding: utf-8 -*-
"""在 README 顶部插入「与 v0.1.0 并存」章节。

## 为什么需要
v2 改成了**并存模式**（服务名 `zfhVoiceV2`、端口 8766、包名 `zfh_voice2`），
但 README 里还没说清楚"怎么和旧版一起用"。

## 幂等
用标记块 `<!-- COEXIST-SECTION -->` 检测，重复运行不会重复插入。

## 用法
    python tools/add_coexist_section.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(r"D:\A\dsh-zhuang-fangyi-voice-v2")
README = ROOT / "README.md"
MARK = "<!-- COEXIST-SECTION -->"

SECTION = f"""{MARK}

## 🔀 与 v0.1.0 **并存**（本版设计为可同时安装）

本版**不与旧版冲突** —— 所有会撞车的标识都错开了，两版可以同时启用：

| | v0.1.0 | **v0.2.0（本版）** |
|---|---|---|
| Loader 条目 id | `zhuang-fangyi-voice` | `zhuang-fangyi-voice-v2` |
| npm 包名 | `dsh-zhuang-fangyi-voice` | `dsh-zhuang-fangyi-voice-v2` |
| **宿主服务名** | `zfhVoice` | **`zfhVoiceV2`**（另挂短别名 `zfhVoice2`） |
| HTTP 端口 | 8765 | **8766** |
| Python 包 | `zfh_voice` | `zfh_voice2` |
| CLI | `zfh-voice` | `zfh-voice2` |
| 缓存目录 | `~/.cache/zfh-voice` | `~/.cache/zfh-voice2` |

> **v0.2.0 不抢 `zfhVoice`** —— 那是 v0.1.0 的服务名，保持不变。
> 桌宠等旧调用方**完全不受影响**；想用新版就取 `zfhVoiceV2`。

### 桌宠怎么同时用两个

```js
const v1 = ctx.get('zfhVoice')       // 旧版（不变）
const v2 = ctx.get('zfhVoiceV2')     // 新版（三种风格）

const a = await v1.synthesize('你好')            // 旧版：正常语气
const b = await v2.synthesize('你好', 'firm')    // 新版：强硬
const c = await v2.synthesize('你好', 'gentle')  // 新版：娇
```

### 验证并存

```bash
python tests/verify_coexist.py         # 19 项静态检查
python tests/verify_coexist.py --live  # 22 项，含**真实同时起两个服务**
```

`--live` 会真的在 8765 与 8766 各起一个服务并探测 `/health`，
确认两者**同时存活**。

---
"""


def main() -> int:
    if not README.exists():
        print(f"  !! 找不到 {README}")
        return 1
    s = README.read_text(encoding="utf-8")
    if MARK in s:
        print("  ~ 并存章节已存在，跳过")
        return 0

    # 插到「它提供哪些用法」之前（若找不到就插到第一个 `## ` 之前）
    import re
    m = re.search(r"^## 它提供哪些用法", s, re.M)
    if not m:
        m = re.search(r"^## ", s, re.M)
    if not m:
        print("  !! 找不到插入锚点")
        return 1
    s = s[:m.start()] + SECTION + "\n" + s[m.start():]
    README.write_text(s, encoding="utf-8")
    print(f"  ✓ 并存章节已插入（README 现 {len(s.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
