# DSH 庄方宜语音 v2：GPT-SoVITS v5 模型选择设计

## 目标

让 `dsh-zhuang-fangyi-voice-v2` 只服务 GPT-SoVITS v5，并在 DSH 插件设置页中自动发现、展示和选择 v5 SoVITS 权重。

v1 与 v2 的边界固定：

| 插件 | 服务 | 端口 | 模型 |
|---|---|---:|---|
| `dsh-zhuang-fangyi-voice` | `zfhVoice` | 8765 | GPT-SoVITS v2 / v2Pro |
| `dsh-zhuang-fangyi-voice-v2` | `zfhVoiceV2` | 8766 | **仅 GPT-SoVITS v5dev** |

v2 不兼容、不下载、不展示 v2/v2Pro 权重；需要旧模型的用户使用 v1 插件。

## 模型仓库

### GitHub

- 代码仓库：`yangwenjie1231/dsh-zhuang-fangyi-voice-v2`
- Release tag：`model-v2-v5-fullft`

### ModelScope

- 新的干净模型仓库：`yangwenjie1231/dsh-zhuang-fangyi-voice-v2-model`
- 不使用旧 `zfy-v5` 工作区或 `zfy-v5-fullft-pkg` 作为用户分发仓库。

## 模型目录

标准布局：

```text
<modelDir>/
  torch_weights/
    zfh-e4.ckpt
    sovits/
      v5_full_e32_final_e8_s432_l0.pth
      v5_full_e32_final_e16_s864_l0.pth
      v5_full_e32_final_e24_s1296_l0.pth
      v5_full_e32_final_e32_s1728_l0.pth
  ref/
    default.wav
    default.txt
    styles/
      normal.wav / normal.txt
      firm.wav / firm.txt
      gentle.wav / gentle.txt
  configuration.json
  distribution_manifest.json
  SHA256SUMS.txt
```

兼容一个扁平后备布局：`<modelDir>/torch_weights/*.pth`。递归扫描范围只限 `torch_weights/`，不扫描整个磁盘。

## v5 权重识别

- 读取 `.pth` 前两个字节；只接受 `b"07"`（GPT-SoVITS v5dev）。
- `b"08"`（v5turbo）当前不接受，避免错误复用 v5dev vocoder与配置。
- 其他头（v2/v2Pro/v3/v4）不进入列表，并在诊断结果中说明：旧模型请使用 v1 插件。
- 文件名解析 epoch：`_e(?P<epoch>\d+)_s(?P<step>\d+)_l0.pth`。
- 默认选择规则：优先 `configuration.json.default_sovits`；缺失时选择 epoch 最大的合法 v5dev 权重。

每个模型记录：

```text
id, path, filename, version, epoch, step, size, recommended, selected
```

## DSH 设置页

设置页提供动态 SoVITS 模型选择器：

- 插件启动后扫描模型目录；
- 显示 epoch、版本、大小和文件名；
- e32 按配置标记为“推荐”；
- 用户选择后持久化为 `sovitsPath`；
- 提供“重新扫描”动作；
- 没有模型时显示下载命令，而不是空白下拉框。

若宿主的 Loader schema 不支持动态枚举，则由插件注册一个轻量设置面板；模型发现和切换逻辑仍通过同一服务 API，避免 UI 与后端耦合。

## 服务与 HTTP API

在 `zfhVoiceV2` 增加：

```text
models()                 列出模型与当前选择
setModel(id_or_path)     切换模型
rescanModels()           重新扫描
```

HTTP v2 增加：

```text
GET  /v2/models
POST /v2/model           {"model": "..."}
POST /v2/models/rescan
```

切换流程：

1. 校验目标位于 `modelDir/torch_weights` 内且头为 `07`；
2. 停止当前由插件启动的 8766 服务；
3. 更新 `sovitsPath`；
4. 持久化选择；
5. 下次合成或用户点启动时加载新权重。

切换失败保持原选择，不留下半更新状态。

## torch 推理适配

- `TorchBackend` 不再写死 `version: "v2"`，改为 `v5dev` 并在加载后核对实际版本。
- 上游仍使用统一的 `TTS.run(inputs)`；v5 的 CFM、48k vocoder由新版 `gsvRoot` 内部实现。
- `gsvRoot` 必须包含：
  - v5-compatible `TTS_infer_pack`；
  - `gsv-v5-pretrained/s2Gv5dev.pth`；
  - `gsv-v5-pretrained/vocoder.pth`。
- 加载非 v5 权重时抛出明确错误，指向 v1 插件。
- v5 输出采样率保持 48 kHz；后处理对 v5 默认增益为 0 dB。

## 下载器

默认来源顺序：ModelScope → GitHub Release。

```text
ModelScope: yangwenjie1231/dsh-zhuang-fangyi-voice-v2-model
GitHub:    yangwenjie1231/dsh-zhuang-fangyi-voice-v2
Release:   model-v2-v5-fullft
```

命令：

```bash
python download_models.py --torch-only       # GPT + 默认 e32 + 参考资源
python download_models.py --epoch 8
python download_models.py --epoch 16
python download_models.py --epoch 24
python download_models.py --epoch 32
python download_models.py --all-epochs
python download_models.py --check
```

v2 下载器不再下载旧 ONNX/v2Pro 资源。当前没有 v5 ONNX 包，因此 v2 设置页只推荐/允许 torch；若用户选择 ONNX，返回“v5 当前仅支持 torch 后端”。

下载完成后依据 `distribution_manifest.json` 校验大小与 SHA256；校验失败不替换现有可用权重。

## 配置

新增并持久化：

```text
sovitsPath: ""
```

保留：`gptPath` 显式覆盖；缺省使用 `torch_weights/zfh-e4.ckpt`。

`modelDir` 自动发现规则保留，但优先识别新仓库布局；候选目录必须含有效 v5dev 权重才算 v2 模型目录。

## 兼容与迁移

- 保持服务名 `zfhVoiceV2`、别名 `zfhVoice2`、端口 8766。
- 保持现有 `/tts`、风格 API、常驻/卸载行为。
- 不保证旧 v2Pro 模型目录在 v2 插件中继续工作；诊断会提示安装 v1 插件。
- v1 仓库和 v1 插件完全不修改。

## 错误处理

- 无合法 v5 权重：列出扫描目录、拒绝启动推理、提供下载命令。
- 缺 v5 pretrained/vocoder：明确报告缺失路径。
- 非 07 头：隐藏于可选列表并记录诊断信息。
- 模型文件在切换前后被删除：启动失败但设置页保留可重新扫描入口。
- 下载中断：写入 `.part`，SHA256 通过后原子替换。

## 测试与验收

### Python

- v5 头识别；v2/v2Pro/v5turbo 拒绝；
- 嵌套和扁平目录扫描；
- e8/e16/e24/e32 排序；
- `configuration.json` 默认 e32；
- 最大 epoch 回退；
- 路径越界拒绝；
- 下载 URL、续传、SHA256、原子替换；
- v5 后端版本核对与 48k 输出。

### JavaScript

- 配置归一化接受 `sovitsPath`；
- `models()` / `setModel()` / `rescanModels()`；
- 切换时停止旧进程并更新启动参数；
- 失败时保留旧选择；
- v1/v2 插件服务名和端口继续并存。

### 回归

- 原 pytest 122 项；
- 原 Node 插件 77 项；
- v1 兼容守卫；
- styles 冒烟；
- 新 v5 模型发现、选择、下载测试；
- 使用真实 e32 权重做一次加载与短句 48k 合成验收。
