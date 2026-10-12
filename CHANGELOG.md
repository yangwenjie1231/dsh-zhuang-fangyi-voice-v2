# 更新日志

## 0.2.0

**主题：三种语气风格 + 路径自动探测 + API 版本化 + **与 0.1.0 并存****

### 🔀 与 0.1.0 并存（本版的核心设计）

本版**可与 0.1.0 同时安装启用**，所有会撞车的标识都错开了：

| | 0.1.0 | **0.2.0** |
|---|---|---|
| Loader 条目 id | `zhuang-fangyi-voice` | `zhuang-fangyi-voice-v2` |
| npm 包名 | `dsh-zhuang-fangyi-voice` | `dsh-zhuang-fangyi-voice-v2` |
| **宿主服务名** | `zfhVoice` | **`zfhVoiceV2`**（+ 短别名 `zfhVoice2`） |
| HTTP 端口 | 8765 | **8766** |
| Python 包 | `zfh_voice` | `zfh_voice2` |
| CLI | `zfh-voice` | `zfh-voice2` |
| 缓存目录 | `~/.cache/zfh-voice` | `~/.cache/zfh-voice2` |

> **0.2.0 不抢 `zfhVoice`** —— 那是 0.1.0 的服务名，保持不变。
> 旧调用方（桌宠）**完全不受影响**；想用新版就取 `zfhVoiceV2`。

验证：`python tests/verify_coexist.py --live`
（真实在 8765 与 8766 各起一个服务，确认**同时存活**）

### 新增

- **三种音色风格**（同一音色，不同语气）：
  | 风格 | 听感 | 客观特征 |
  |---|---|---|
  | `normal` | 正常说话（**默认，与 0.1.0 一致**） | F0 尾部 +1.89 |
  | `firm` | 语气强硬 | F0 尾部 −4.13，说话人相似度最高 |
  | `gentle` | 语音娇 | F0 尾部 −4.86，500–800Hz 最干净 |

  三种均经**盲听确认可用**，区别在语气而非质量。

  - CLI：`python -m zfh_voice2 --style firm say "你好"`
  - HTTP：`POST /tts {"text":"...","style":"firm"}`
  - 服务：`ctx.get('zfhVoiceV2').synthesize(text, 'firm')`
  - 新子命令：`python -m zfh_voice2 styles`
  - 新脚本：`python tools/install_styles.py`（把风格装进模型目录）

- **路径自动探测**：`modelDir` 留空时会自动扫描
  `<仓库上两级>/zfh-models*`（优先 `v2pro`），
  无需在 DSH patch 里手写绝对路径。

- **HTTP API 版本化**：
  - `/v1/*` 与无前缀路径**完全等价**（旧客户端无感）
  - `/v2/*` 返回增强字段（`api_version` / `styles` / `style`）
  - 新端点：`GET /styles`、`GET /ref`、`POST /style`

- **ONNX 后端优化**：`doctor` 与 `status` 现在会检查 v2Pro 的说话人编码器。

### 兼容性（**硬承诺**）

- `ctx.get('zfhVoiceV2')` 的 11 个方法**签名与返回结构不变**
- HTTP 的 `/health`、`/tts`、`/unload`、`/voices` **原样保留**
- `cordis.patch.yml` 的原有 config 字段**全部继续可用**
- CLI 的原有 8 个子命令**继续可用**
- **默认风格是 `normal`，其参考音频就是 0.1.0 的 `ref/default.wav`**
  → 不指定风格时，输出与 0.1.0 一致

### 修复

- 修一个依赖外部状态的插件测试（会随环境随机失败）

---

## 0.1.0

首个版本。托管 Python 推理服务的生命周期（起停 / 常驻 / 显存释放），
并通过 `zfhVoiceV2` 服务把合成能力暴露给其它插件。
