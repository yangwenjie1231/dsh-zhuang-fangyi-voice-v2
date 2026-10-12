# -*- coding: utf-8 -*-
"""生成 v2 的 README —— **以 v1 为基础增量修改**。

## 为什么不能用重写的 README
我第一版**从头重写**了 README（192 行），结果 v1 的 4 个
`test_docs_consistency` 测试全挂 —— 它们守着文档里的关键技术细节：

    · 「v2Pro」的亮度取值（v2 → 0、v2Pro → 1，以长文本校准）
    · `sv_after_fbank.onnx`（v2Pro 特有的说话人编码器，必须列进 ONNX 清单）
    · 默认种子 42
    · 说权重文件名要灵活（不能硬编码）

而 v1 README 有 **654 行 / 60+ 章节**，重写必然丢东西。

## 正确做法
    1. 复制 v1 README 全文
    2. **插入** v2 新章节（三种风格 / API 版本化 / 路径自动探测）
    3. 改版本号与标题里的描述
    4. **不动**任何 v1 的技术细节

## 输出
    README.md（覆盖）
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

V1 = Path(r"D:\A\dsh-zhuang-fangyi-voice\README.md")
V2 = Path(r"D:\A\dsh-zhuang-fangyi-voice-v2\README.md")

src = V1.read_text(encoding="utf-8")
print("=" * 84)
print("生成 v2 README（以 v1 为基础增量）")
print("=" * 84)
print(f"  v1 README: {len(src.splitlines())} 行")

# ---------- ① 标题下方加 v0.2.0 摘要 ----------
OLD_HEAD = src.split("\n\n", 1)[0]
print(f"  v1 头部: {OLD_HEAD[:80]!r}")

V2_BANNER = """# dsh-zhuang-fangyi-voice

> 庄方宜专属中文音色合成 for **DeepSeek Harness** —— 把任意中文台词用她的声音念出来。
>
> **v0.2.0 新增：三种语气风格（正常 / 强硬 / 娇）· 路径自动探测 · HTTP API 版本化**
>
> 与 v0.1.0 **完全兼容**：不指定风格时，输出与 0.1.0 一致。
"""

src = src.replace(OLD_HEAD, V2_BANNER.strip(), 1)

# ---------- ② 在「它提供哪些用法」之后插入 v2 章节 ----------
V2_SECTION = r"""
---

## 🎭 三种语气风格（v0.2.0 新增）

**同一模型 + 不同参考音频 = 不同语气。** 参考音频不只提供音色，
还提供**韵律风格**（F0 走向、语速、停顿）—— 这是推理时参数，**不需要重训模型**。

| 风格 | 听感 | 参考音频 | 客观特征 |
|---|---|---|---|
| **`normal`** | 正常说话（**默认**） | 7.95s 48kHz | F0 尾部 +1.89，500–800Hz 8.94% |
| **`firm`** | 语气强硬 | 8.66s 32kHz | F0 尾部 −4.13，说话人相似度最高 |
| **`gentle`** | 语音娇 | 5.26s 32kHz | F0 尾部 −4.86，500–800Hz 最干净 4.25% |

三种均经**盲听确认可用**，区别在语气而非质量。

### 安装风格（一次性）

```bash
python tools/install_styles.py
```

它会把三份参考音频装到 `<模型目录>/ref/styles/`，
**并把 `normal` 写回 `ref/default.wav`**（保证 0.1.0 行为不变），
原文件备份为 `*.bak-v1`，最后用包络互相关验证三段互不相同。

### 用风格

```bash
# CLI（--style 是**全局参数**，要放在子命令前）
python -m zfh_voice styles                          # 列出风格
python -m zfh_voice --style firm say "你好"
python -m zfh_voice --style gentle say "你好"

# HTTP
curl -X POST http://127.0.0.1:8765/tts \
     -H 'content-type: application/json' \
     -d '{"text":"你好","style":"firm"}' -o out.wav

# DSH 服务（桌宠）
const v = ctx.get('zfhVoice')
const r = await v.synthesize('你好', 'firm')   // 不传第二个参数 = 用配置的风格
await v.setStyle('gentle')                     // 运行时切换
const list = await v.styles()                  // 列出可用风格
```

### 为什么是"参考音频"而不是"训练"

参考音频决定的不只是音色，还有**整句的韵律轮廓**。实测映射关系：

| 参考音频的 F0 尾部斜率 | 合成输出的 F0 尾部斜率 | 听感 |
|---|---|---|
| +16.71（问句「好吗？」） | +1.89 | 正常 |
| −11.20（陈述句） | −4.13 | 强硬 |
| −18.07（陈述句） | −4.86 | 娇 |

> ⚠️ **说话人相似度不测语调自然度。** `firm` 的相似度（0.8589）高于
> `normal`（0.8488），但盲听显示 `firm` 的语调"怪怪的"。
> 相似度只测音色 —— **选风格要靠耳朵，不要只看指标。**

---

## 🔀 HTTP API 版本化（v0.2.0 新增）

**无前缀路径 = v1 行为**，旧客户端无感。`/v1/*` 与无前缀完全等价。

| 方法 | 端点 | 说明 |
|---|---|---|
| GET | `/health` | 服务状态（**与 0.1.0 一致**） |
| GET | `/voices` | 音色列表（**与 0.1.0 一致**） |
| POST | `/tts` | 合成，返回 WAV（**与 0.1.0 一致**） |
| POST | `/tts.json` | 合成，返回 base64（**与 0.1.0 一致**） |
| POST | `/cache/clear` | 清缓存 |
| GET/POST | `/unload` | 卸模型（显存释放，留进程） |

**v2 新增端点**：

| 方法 | 端点 | 说明 |
|---|---|---|
| GET | `/v2/health` | 多出 `api_version` / `styles` / `style` |
| GET | `/v2/voices` | 音色下挂 `styles` 列表 |
| GET | `/styles` | 风格列表 + 当前风格 |
| GET | `/ref` | 当前参考音频摘要 |
| POST | `/style` | 运行时切换风格 `{"style":"firm"}` |

**`POST /tts` 新增可选 `style` 字段**：不传时请求体里不含该字段，
服务端状态**完全不变**（与 0.1.0 一致）。

---

## 🔍 路径自动探测（v0.2.0 新增）

v0.1.0 需要**在 DSH patch 里手写绝对路径**（`modelDir` / `python` / `gsvRoot`），
换机就废。v0.2.0 会**自动探测**：

`modelDir` 留空时按以下顺序找：

1. `ZFH_MODEL_DIR` 环境变量
2. `<仓库>/models`
3. `~/.cache/zfh-voice/models`
4. **自动探测**（新增）：
   - `ZFH_MODELS_ROOT` 环境变量指定的根
   - `<仓库上两级>/zfh-models*`（**优先 `v2pro`**）
   - `<仓库上两级>/<voiceclone|voice|models|zfy|...>/zfh-models*`
   - `~/.zfh-models` / `~/.zfh-voice/models`

例：仓库在 `D:\A\dsh-zhuang-fangyi-voice-v2`，
自动找到 `D:\A\voiceclone\zfh-models-v2pro`。

```bash
# 不传 --model-dir 也能跑
python -m zfh_voice --backend torch --style firm say "你好"
```

---

## ✅ 与 v0.1.0 的兼容性（硬承诺）

| 层面 | 承诺 |
|---|---|
| `ctx.get('zfhVoice')` | 方法**签名与返回结构不变**（新增 `styles()` / `setStyle()` 为增量） |
| HTTP | `/health` `/tts` `/unload` `/voices` **原样保留** |
| 配置 | `cordis.patch.yml` 原有字段**全部继续可用** |
| CLI | 原有 8 个子命令**继续可用** |
| **默认行为** | **不指定风格时输出与 0.1.0 一致**（normal = 原 `ref/default.wav`） |

**兼容性由测试守着**：

```bash
node tools/test-plugin.mjs        # 77 项（v0.1.0 全部用例）
python tests/test_v1_compat.py    # 29 项契约断言
python tests/smoke_styles.py      # styles 模块
python tests/verify_styles_e2e.py # 三种风格端到端
```
"""

# 插到「它提供哪些用法」章节末尾（下一个 `## ` 之前）
m = re.search(r"^## 它提供哪些用法.*?(?=^## )", src, re.S | re.M)
if m:
    src = src[:m.end()] + V2_SECTION + "\n" + src[m.end():]
    print("  ✓ v2 章节已插入（三种风格 / API 版本化 / 路径探测 / 兼容性）")
else:
    print("  !! 找不到插入锚点「它提供哪些用法」")
    sys.exit(1)

# ---------- ③ 配置章节补 style ----------
src = src.replace(
    "### 模型目录解析顺序",
    """### 配置项（v0.2.0 新增 `style`）

```yaml
config:
  style: normal        # normal / firm / gentle（留空 = normal）
```

### 模型目录解析顺序""", 1)

# ---------- ④ 顶部加变更日志链接 ----------
src = src.rstrip() + "\n\n---\n\n## 更新日志\n\n见 [CHANGELOG.md](./CHANGELOG.md)。\n"

V2.write_text(src, encoding="utf-8")
print(f"  v2 README: {len(src.splitlines())} 行")
print()
print("=" * 84)
print("关键：v1 的技术细节**全部保留**")
for key in ("v2Pro", "sv_after_fbank.onnx", "42", "亮度补偿", "参考音频",
            "常驻开关", "中英混排", "故障排查"):
    print(f"  {'✓' if key in src else '✗'} {key}")
print("=" * 84)
