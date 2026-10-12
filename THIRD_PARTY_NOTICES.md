# 第三方组件与声明

本项目的**代码**以 MIT 协议发布。其中文本前端部分内嵌了第三方项目，
以下逐项说明来源与改动。

---

## 1. GPT-SoVITS（内嵌文本前端）

- 来源：https://github.com/RVC-Boss/GPT-SoVITS
- 协议：MIT License, Copyright (c) 2024 RVC-Boss
- 内嵌位置：`src/zfh_voice/_vendor/_gsv_text/`
- 内嵌范围：仅中文（zh）路径所需文件
  - `__init__.py`, `cleaner.py`, `chinese2.py`, `tone_sandhi.py`,
    `symbols.py`, `symbols2.py`, `opencpop-strict.txt`
  - `g2pw/`（多音字，不含大模型文件）
  - `zh_normalization/`（数字、日期等文本正则化）

### 所做的改动（均已在代码中标注 `PATCHED`）

1. **包重命名** `text` → `_gsv_text`
   避免与宿主环境中的同名模块冲突。内部 import 已同步改写。

2. **`chinese2.py`：G2PW 路径改为环境变量驱动**
   上游在 import 时用硬编码相对路径 `GPT_SoVITS/text/G2PWModel` 加载模型，
   且当目录不存在时 `g2pw/onnx_api.py` 会**联网下载**模型。
   现改为读取 `ZFH_G2PW_DIR` / `ZFH_BERT_DIR`，缺失时抛出明确错误，
   由 `zfh_voice.frontend.prepare()` 负责解析与设置。

3. **`chinese2.py`：分词器回退**
   `jieba_fast`（C 扩展，部分平台无轮子）导入失败时自动回退到纯 Python 的 `jieba`。

4. **`g2pw/onnx_api.py`：执行提供器改为可配置，默认 CPU**
   上游逻辑是「只要 `CUDAExecutionProvider` 可用就用 CUDA」。
   但本仓库分发的 `g2pW.onnx` 是 **fp16** 模型，CUDA EP 在 float32 输入下会抛
   `Tensor type mismatch. T != PrimitiveDataType<MLFloat16>`；
   而 G2PW 属于每次合成一次性调用的小模型，CPU 完全够快。
   改为默认 `CPUExecutionProvider`，可用 `ZFH_G2PW_PROVIDER=cuda|dml` 显式覆盖。

> 未改动的其余文件保持上游原样，便于日后比对与升级。

---

## 2. 模型文件（不在本仓库内）

模型作为 **GitHub Release 资产** 分发，包含：

| 文件 | 说明 |
|---|---|
| `zfh_t2s_*.onnx`, `zfh_vits.onnx` | 本项目微调训练的音色模型（导出为 ONNX） |
| `bert_encoder.onnx` | 由 `hfl/chinese-roberta-wwm-ext-large` 导出 |
| `hubert_encoder.onnx` | 由 `facebook/hubert-base-ls960` 的中文版 `chinese-hubert-base` 导出 |
| `g2pW.onnx` | 来自 GPT-SoVITS 的 G2PWModel |

各上游模型的协议请遵循其原始仓库声明：

- GPT-SoVITS 预训练权重：MIT（RVC-Boss/GPT-SoVITS）
- chinese-roberta-wwm-ext-large：Apache-2.0（hfl）
- chinese-hubert-base：MIT（见上游 model card）

---

## 3. 音色与素材声明

本项目的音色模型基于**游戏角色语音素材**微调训练，仅供**个人学习、
同人创作**等非商业用途。

若需商业使用或公开发布衍生内容，请自行取得相应权利方授权。
使用者应自行承担合规责任。
