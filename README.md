# dsh-zhuang-fangyi-voice

> 庄方宜专属中文音色合成 for **DeepSeek Harness** —— 把任意中文台词用她的声音念出来。
>
> **v0.2.0 新增：三种语气风格（正常 / 强硬 / 娇）· 路径自动探测 · HTTP API 版本化**
>
> 与 v0.1.0 **完全兼容**：不指定风格时，输出与 0.1.0 一致。

**庄方宜专属中文音色** —— 说任意中文台词，用她的声音。
作为 [`dsh-zhuang-fangyi-pet`](../dsh-zhuang-fangyi-pet) 的语音子插件。

实测性能见文末[基准数据](#基准数据)。**同类工具中体积最小、依赖最轻**：
不需要 CUDA、不需要 torch（ONNX 后端），一条命令即可出声。

---

## 🚀 安装

**新用户看这份就够：[docs/安装提示词.md](docs/安装提示词.md)**
—— 里面有一段**可整段复制给 DSH（或任意编码 agent）的提示词**，
它会自己体检环境、判断该装哪套模型、装好并验证。

最短路径（想自己动手）：

```bash
python -m zfh_voice doctor        # ① 体检：一次拿到全部判断依据与建议
pip install -r requirements.txt   # ② 装依赖（照体检建议，有 N 卡才加装 torch）
python download_models.py         # ③ 下模型（默认 fp16，约 1.4GB，优先走魔搭）
python -m zfh_voice say "今天天气不错"   # ④ 出声
```

> `doctor` **不需要任何第三方依赖**就能跑（裸机器上也行）——
> 它存在的意义就是在你什么都还没装的时候告诉你缺什么。

装成 DSH 插件：见[下文](#装成-dsh-插件推荐)。

---

## 它提供哪些用法

| 方式 | 适用场景 | 命令/代码 |
|---|---|---|
| **命令行** | 快速生成几条 | `python -m zfh_voice say "台词"` |
| **批量** | 一次生成一整批台词 | `python -m zfh_voice batch lines.txt -d out/` |
| **Python API** | 嵌进别的 Python 程序 | `from zfh_voice import TTS` |
| **HTTP 服务** | **桌宠等外部程序调用** | `python -m zfh_voice serve --port 8765` |
| **合成缓存** | 相同台词秒回，不重复推理 | 自动，无需配置 |

两种推理后端可切换：

| 后端 | 依赖 | 速度 | 适用 |
|---|---|---|---|
| **onnx**（默认） | 仅 onnxruntime | 慢（RTF ≈ 6） | 分发、无显卡机器、跨平台 |
| **torch** | torch + GPT-SoVITS 源码 | **快（RTF ≈ 0.45 @GPU）** | 自己机器、批量生成 |

---


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

## 快速开始

```bash
git clone https://github.com/yangwenjie1231/dsh-zhuang-fangyi-voice
cd dsh-zhuang-fangyi-voice

pip install -r requirements.txt
python download_models.py            # 下载模型（约 1.4GB）

python -m zfh_voice status           # 确认模型就绪
python -m zfh_voice say "今天天气不错，我们一起出去走走吧。"
```

输出默认落在 `out/out.wav`。

### 模型下载源

`download_models.py` **默认优先 ModelScope**（国内快），失败自动回退 GitHub：

| 源 | 印尼 |
|---|---|
| ModelScope | `https://modelscope.cn/models/yangwenjie1231/dsh-zhuang-fangyi-voice` |
| GitHub | `https://github.com/yangwenjie1231/dsh-zhuang-fangyi-voice/releases` |

也可手动指定：`python download_models.py --source github`。

### 想更快？用 torch 后端

> ⚠️ **ONNX 模型无法被 torch 加载。** torch 后端需要原始的 `.ckpt` / `.pth` 权重
> （共约 229 MB），要单独下载：
>
> ```bash
> python download_models.py --torch-only   # 只下 torch 所需，跳过 1.4GB 的 ONNX
> # 或已下过 ONNX、想补 torch：
> python download_models.py --with-torch
> ```

```bash
pip install torch torchaudio
git clone https://github.com/RVC-Boss/GPT-SoVITS third_party/GPT-SoVITS

# 权重已下到 models/torch_weights/，后端会自动找到；也可用 gpt_path/sovits_path 指定
python -m zfh_voice --backend torch \
    --gsv-root third_party/GPT-SoVITS say "今天天气不错。"
```

两种后端的差异：

| | onnx（默认） | torch |
|---|---|---|
| 依赖 | 仅 onnxruntime | torch + GPT-SoVITS 源码 |
| 需要 GPU | 否（CPU / DirectML 均可） | 建议有 NVIDIA GPU |
| 速度 | RTF ≈ 6 | **RTF ≈ 0.45（GPU）** |
| 加载的模型 | `*.onnx` | `*.ckpt` / `*.pth` |

---

## HTTP 服务（桌宠对接用）

```bash
python -m zfh_voice serve --host 127.0.0.1 --port 8765
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查；含 `resident`（模型是否常驻）/ `idle_seconds` / `loads` / `unloads` |
| GET | `/voices` | 可用音色列表 |
| POST | `/tts` | body `{"text":"...","seed":null}` → 返回 **audio/wav** |
| POST | `/tts.json` | 同上，但返回 `{ok,duration,cached,cold_start,wav_base64}` JSON |
| POST | `/cache/clear` | 清空合成缓存 |
| POST | `/unload` | 立即释放模型（腾出显存），下次请求按需重载 |

示例：

```bash
curl -X POST http://127.0.0.1:8765/tts \
     -H "Content-Type: application/json" \
     -d '{"text":"管理员，今天的实验数据我已经整理好了。"}' \
     -o reply.wav
```

> 服务已开启 CORS，浏览器端可直接 `fetch` 取 `audio/wav` 播放。
> 请求串行处理（推理后端非线程安全），并发请求会排队。

---

## 中英混排怎么念

### 问题：直接调中文前端会**静默删掉**英文

GPT-SoVITS 的中文前端 `chinese2.py` 有这一行：

```python
processed_segments = [re.sub("[a-zA-Z]+", "", seg) for seg in segments]
```

`GPU`、`Hello`、`Python` 不但不发音，**连时长都不占**，而且不报任何错。
实测（音素数完全不变）：

| 输入 | 音素数 | 结果 |
|---|---:|---|
| `今天天气不错。` | 13 | — |
| `今天天气不错，Hello World。` | **13** | 英文没产生任何音素 |
| `Hello world, this is a test.` | **0** | 合成直接报错 |

**根因不是模型不会英文** —— 是**绕过了上游的 `LangSegmenter`**。
上游 `TextPreprocessor` 会先按语言分段：

```
今天天气不错，Hello World，我已经整理好了。
  → zh:'今天天气不错，' + en:'Hello World，' + zh:'我已经整理好了。'
```

英文段走 `en`（g2p_en + cmudict，产出 ARPAbet 音素），中文段走 `zh`。
**参考音频只提供音色（`ge` 向量），音素用哪种语言与音色无关** ——
所以"中文音色 + 英文发音"成立，实测音色相似度 0.62，与纯中文相当。

### 本库的处理

`frontend.text_to_segments()` 走 `LangSegmenter` 分段，两个后端都用它。

**管线正确性已验证**：我们的音素 id 与上游 `TextPreprocessor.get_phones_and_bert`
**逐位完全相同**（含英文音素），所以英文确实被送进了模型：

```
这个功能叫 Hello World，很简单。
  → [320,133,156,134,156,235,227,146,221,189,      ← 中文段
     51,12,62,68,91,39,62,26,                       ← HH AH0 L OW1 W ER1 L D（Hello World）
     158,141,221,178,127,107,3]                     ← 中文段
```

### ⚠️ 但要如实说明能力边界：**多词英文短语夹在中文里念不清**

实测（v2 与 v2Pro 表现一致）：

| 场景 | 例子 | 英文还原率 |
|---|---|---|
| 纯英文句 | `Hello world, this is a test.` | **6/6 = 100%** |
| 纯英文句 | `OK, no problem.` | **3/3 = 100%** |
| 单个词夹中文 | `这是 OK 的。` | **1/1 = 100%** |
| 缩写 | `GPU 和 AI 都很重要。` | 1/2（`AI` 会念成 `iI`） |
| 缩写 | `这个 API 很好用。` | 0/1（念成 `AP`） |
| **多词短语夹中文** | `这个功能叫 Hello World，很简单。` | **0/2**（念成"就好了我有"） |
| **多词短语夹中文** | `我说的 Hello 就是这个意思。` | **0/1**（念成"很老"） |

**规律**：纯英文整句很好；**1 个英文词**夹中文也可以；
**多词英文短语夹在中文句子里**会散掉。

**原因**：音素序列是对的，但声学模型（SoVITS）只见过中文训练语料，
连续英文音素的建模不足 —— 短串还能凑合，连续多词就糊了。

**如果你的文本里有英文短语**，两个选择：

```python
# 1) 用真英文发音（默认）—— 适合纯英文句、单个英文词
tts.say("这个 API 很好用。")

# 2) 用中文读法兜底 —— 适合多词英文短语
tts.say("这个功能叫 Hello World，很简单。", localize=True)
# 实际念: 这个功能叫 哈喽 沃尔德，很简单。
```

> 这条边界是**实测出来的**，不是推测。早期文档写的 `GPU 和 AI 都很重要。`
> 还原 2/2 是单次测量，重复测试后是 1/2 —— 已按多次测量结果更正。
| `这个功能叫 Hello World，很简单。` | `这个功能叫Hello World很简单` | **2/2** |
| `今天的实验数据我已经整理好了。` | 中文完整 | — |

```bash
python -m zfh_voice say "GPU 和 AI 都很重要。"
# 3.40s  ->  out/out.wav
```

### 所需数据**随包分发，不需要联网**

| 数据 | 体积 | 位置 |
|---|---:|---|
| `cmudict.rep` / `cmudict-fast.rep` | 7.4 MB | `_vendor/_gsv_text/`（`english.py` 读） |
| nltk tagger（词性标注） | 5.4 MB | `_vendor/nltk_data/`（`prepare()` 设 `NLTK_DATA`） |
| `lid.176.ftz`（语言检测） | 916 KB | 随 `fast-langdetect` 包 |

> **不装这些依赖也能跑**，但英文会被前端静默删除。
> 装法：`pip install nltk g2p-en wordsegment fast-langdetect split-lang`

### 两条兜底

1. **`localize` 参数**（默认 `"auto"`）：
   - `"auto"` → 混排可用时**原样送**（真英文发音）；不可用时自动转中文读法兜底
   - `True` → 总是转中文读法（`GPU`→`基皮尤`）—— 纯离线、零 nltk 依赖
   - `False` → 总是不转
2. 转换实现见 [src/zfh_voice/textprep.py](src/zfh_voice/textprep.py)（词表可自行扩充）

### 踩过的坑（供参考）

- **`english.py` 的 4 处 `open()` 没指定 encoding** → 中文 Windows 下按 GBK 读
  UTF-8 的 `cmudict.rep` 直接 `UnicodeDecodeError`。已在 vendor 里补 `encoding="utf-8"`。
- **`word2ph` 对非中文语言是 `None`**（上游契约），BERT 特征对非中文段应给**全零** ——
  照搬中文逻辑会 `'NoneType' object is not iterable`。
- **`text_bert` 的 ONNX 契约是 `[T, 1024]`**，而上游 torch 内部是 `[1024, T]`（导出时转置）。
  按上游布局写会报 `Got invalid dimensions for input`。
- **`LangSegmenter` 硬编码指向 125 MB 的 `lid.176.bin`**，目录不存在时还会**联网下载**。
  实测 lite（916 KB，随 pip 包）与 full 的分段结果**逐条一致**，故改用 lite；
  需要 full 时设 `ZFH_LANGDETECT_MODEL=<lid.176.bin 路径>`。

---

## 常驻开关：省掉反复冷启动

**CUDA 下每次冷启动都要重付"加载 + 预热"的代价**（本机实测：加载约 10 秒、
首句预热约 15 秒）。**在 DSH 里用插件时，这个开关归插件管**（见下）；
下面是命令行/独立使用时的做法。

### 在插件里（DSH / 桌宠）

| 想要什么 | 怎么做 |
|---|---|
| 一直占着，合成立刻开始 | 桌宠设置页打开「让语音模型常驻」，或 `zfhVoice.setResident(true)` |
| 空闲后释放显存 | 关掉常驻，设「空闲多少秒后释放」（默认 300s） |
| **释放的彻底程度** | `idleAction`：`stop` 杀进程（全释放、恢复慢）/ `unload` 只卸模型留进程（**显存同样释放、恢复快**） |
| 现在到底占不占显存 | `zfhVoice.status().modelLoaded`（`true`/`false`/`null`=未知） |

> ⚠️ **`status().resident` 是"配置意图"，`modelLoaded` 才是"真实状态"。**
> 配了常驻但服务刚起、模型还在加载时，前者 `true`、后者 `false`。
> 判断占不占显存要看 `modelLoaded`。

### 在命令行（不经 DSH）

```bash
# 一直常驻（默认）：模型不释放，响应最快
python -m zfh_voice serve

# 空闲 300 秒后自动释放模型，腾出显存；下次请求按需重载
python -m zfh_voice serve --idle-timeout 300

# 启动时不预热，把加载推迟到首次请求
python -m zfh_voice serve --no-preload
```

> 插件模式下**不会**传 `--idle-timeout`，策略完全由插件决定 ——
> 这个服务端参数是给"不经 DSH 直接跑 serve"的场景用的，避免两处打架。

查看当前是否常驻：

```bash
curl http://127.0.0.1:8765/health
# {"ok":true, ..., "resident":true, "idle_seconds":3.2, "loads":1, "unloads":0}

curl -X POST http://127.0.0.1:8765/unload    # 立刻释放模型（保留进程）
```

> 有请求在途时**绝不会释放模型**（长合成可能远超 `idle-timeout`）。

## 音质：两个可调项

### 1. 默认种子：让短句稳定下来

GPT-SoVITS 的 AR 解码**不传 seed 时每次采样都不同**，而且**句子越短波动越大**：

| 文本长度 | 时长变异系数 | 实测现象 |
|---|---|---|
| 4 字（`让我想想`） | **23.5%** | 随机种子下偶发 0.74s（正常 1.22s）——近乎截断 |
| 10 字 | 17.0% | |
| 24 字 | 10.3% | 长句基本稳定 |

所以本库**默认用固定种子 42**（`seed` 配置项）。牺牲一点多样性，
换来可复现的输出：同一句话每次合成时长完全一致。

```bash
# 默认：固定 42
python -m zfh_voice say "让我想想"

# 显式指定别的种子
python -m zfh_voice --seed 7 say "让我想想"

# 恢复上游的随机采样（**不推荐**，短句会不稳）
python -m zfh_voice --random-seed say "让我想想"
```

> **诚实说明**：固定种子后**时长**是 100% 可复现的，但波形仍有约 **−51 dB**
> 的浮点差异（CUDA 原子操作的非确定性；相关系数 0.999997，样本差 0.006
> vs 峰值 0.95）。这个量级**听觉上不可闻**，但严格说不是逐字节可复现。

### 2. 亮度补偿：补回重采样削掉的那一段

模型输出 32 kHz，而素材是 48 kHz。48k→32k 下采样在 Nyquist（16 kHz）
附近的抗混叠滤波滚降，使 **12–16 kHz 比原声低约 2.8 dB** —— 听感偏「糊」。

补偿是**高通搁架**，参数经网格搜索校准（`fc=12kHz, gain=3dB, Q=1.0`）：

| 频段 | 48k 原声 | 未补偿 | 补偿后 | 效果 |
|---|---|---|---|---|
| 8–12 kHz | −36.4 | −31.8 | −31.7 | +0.09 dB（**不会过亮**） |
| **12–16 kHz** | −45.4 | **−47.6** | **−44.7** | **误差 +0.61 dB**（目标 ≤1.5） |

```bash
# 默认 auto：按模型版本自动选校准值（v2 → 3，v2Pro → 0）
python -m zfh_voice say "你好"

# 显式指定（覆盖 auto）
python -m zfh_voice --brightness-db 3 say "你好"

# 显式关闭（与 auto 语义不同：0 = 一定不动输出）
python -m zfh_voice --brightness-db 0 say "你好"
```

插件配置里是 `brightnessDb`，默认 `'auto'`（也可填 0~12 的数字）。

> **诚实说明**：这是**听感补偿**，恢复的是重采样滤波造成的频谱倾斜，
> **不是信息恢复**。经实测，素材本身的有效带宽约 16 kHz
> （16–20 kHz 仅 −60 dB、20–24 kHz 是 −75~−86 dB 噪声底），
> 所以升到 48 kHz 输出**不会**补回真实信息。

#### 补偿量随模型而变 → 已改成 `auto` 自动处理（按**长文本**校准）

不同模型的**频谱起点不同**，同一个数字会一个补齐、一个过冲：

| | 8–12 kHz（目标 −36.7） | 12–16 kHz（目标 −45.25） |
|---|---|---|
| v2 未补偿 | −32.5（**偏亮 +4.2**） | −45.6（对齐） |
| v2Pro 未补偿 | −36.6（**已对齐**） | −46.5（仅差 1.2） |

**v2 的问题是 8–12k 偏亮，补 12–16k 只会让它整体更偏。**
长文本（421 秒全文）实测扫描：

| 补偿 | v2 合计偏差 | v2Pro 合计偏差 |
|---|---|---|
| **0 dB** | **6.04** ← 最优 | 1.60 |
| 1 dB | 6.35 | **1.12** ← 最优 |
| 3 dB | 8.51 | 3.12 |

**所以默认值是 `'auto'`** —— 启动时按加载到的模型版本自动选：

| 模型 | auto 解析为 |
|---|---|
| v2 / v1 | **0**（不补偿） |
| **v2Pro / v2ProPlus** | **1 dB** |
| v3 / v4 | 0 |
| 未知版本 | 0（保守：宁可不动，也不凭猜测加高频） |

**你不必记住"换模型要改数字"。** 想手动控制仍可填数字：

- `brightnessDb: 0` → **显式关闭**（与 auto 语义不同）
- `brightnessDb: 2` → 强制 +2 dB

> ⚠️ **短句与长文本的偏好相反**：短句最优 **3 dB**，长文本最优 **0~1 dB**。
> 因为长文本由多段独立合成后拼接，**段间过渡抬高了高频**
> （同一句单测 12–16 kHz 是 −47.7，而 421 秒全文是 −46.5，天然高 1.2 dB）。
>
> **这里以长文本为准**（实际用途是长朗读）。做短句交互的话可以调到 2~3。

### 一次性命令复用常驻服务

CLI 每次运行都是新进程，会重新加载模型。让它走已在运行的服务即可：

```bash
python -m zfh_voice say "今天天气不错。" --server
# 或指定地址 / 批量
python -m zfh_voice say "..." --server http://127.0.0.1:8765
python -m zfh_voice batch lines.txt -d out/ --server
```

`--server` 不带值（或 `--server auto`）会自动探测 `127.0.0.1:8765`，
探测不到则**回退到本地加载**，不会失败。

实测对比（同一句话）：

| 方式 | 耗时 |
|---|---:|
| 直接 `say`（冷启动） | ~35 s |
| `say --server`（服务常驻 + 缓存命中） | **0.2 s** |

---

## Python API

```python
from zfh_voice import TTS

tts = TTS()                       # 默认 onnx 后端
r = tts.say("管理员，今天的实验数据我已经整理好了。")
print(r.duration, r.sr, r.cached)
r.save("reply.wav")

# 批量
texts = ["早上好。", "该休息一下了。", "任务完成，辛苦啦。"]
tts.say_many(texts, "out/")

# 直接拿字节（发给别处播放）
wav_bytes = r.to_wav_bytes()
```

---

## 配置

### 配置项（v0.2.0 新增 `style`）

```yaml
config:
  style: normal        # normal / firm / gentle（留空 = normal）
```

### 模型目录解析顺序

1. 环境变量 `ZFH_MODEL_DIR`
2. 仓库下的 `models/`
3. `~/.cache/zfh-voice/models`

### 模型目录里该放什么

| 后端 | 必须有 |
|---|---|
| `onnx` | 7 个 ONNX + `G2PWModel/` + `chinese-roberta-wwm-ext-large/` + `ref/default.wav` |
| `onnx`（**v2Pro 包**） | 上面那些 **再加 `sv_after_fbank.onnx`**（说话人编码器，v2Pro 特有） |
| `torch` | `torch_weights/` 下的 GPT + SoVITS 权重，外加 `ref/default.wav` |

**torch 权重不锁死文件名** —— 下面两种命名都认：

```
torch_weights/zfh-e4.ckpt              + zfh_e6_s186.pth          # 默认发行版
torch_weights/<任意名>-e<N>.ckpt        + <任意名>_e<N>_s<N>.pth    # 自己重训的
```

> 为什么放宽：早先硬编码了具体文件名，导致**重训或换版本（如 v2Pro）后**
> 目录被判成"没有模型"，报错只说"缺 1 项" —— 而它其实能正常合成。

不确定缺什么就跑体检，它会明确列出：

```bash
python -m zfh_voice status
# 缺 sv_after_fbank.onnx  (v2Pro 的说话人编码器)
```

### 参考音频

音色由**参考音频**驱动。默认使用模型包里的 `ref/default.wav`（约 8 秒）。
想换参考：

```python
tts = TTS(ref_wav="my_ref.wav", ref_text="这段音频里说的那句话")
```

**硬性要求**（与上游一致）：

- 时长 **3~10 秒**，超出直接报错
- `ref_text` 必须与音频内容**逐字一致**，否则音色会跑偏
- 优先选干净干声；带背景音乐的参考会把底噪一起学进输出

### 精度切换

`download_models.py --precision {fp16,int8,fp32}`

| 精度 | 体积 | 说明 |
|---|---:|---|
| fp16（默认） | 1.40 GB | 数值基本无损 |
| int8 | 0.87 GB | 更小，但音质需自行试听确认 |
| fp32 | 2.79 GB | 基准 |

### 用任意显卡加速

把 `onnxruntime` 换成 `onnxruntime-directml`，代码会自动优先选择
`DmlExecutionProvider`，Windows 上 NVIDIA / AMD / Intel 都能加速，无需 CUDA。

---

## 装成 DSH 插件（推荐）

本仓库除了是 Python 库，**也是一个 DSH 插件**：`index.js` 是它的宿主半边，
负责托管推理服务的进程（启动/保活/空闲释放显存），并把合成能力挂成宿主服务
**`zfhVoice`** —— 桌宠插件用它把 AI 的回答念出来。

```powershell
# 1) 装 Python 依赖与模型（见「快速开始」）
# 2) 复制插件到 DSH profile
powershell -NoProfile -ExecutionPolicy Bypass -File tools\install-dsh-plugin.ps1
# 3) 重启 DSH（ESM 模块有缓存，disable→enable 对入口文件不够）
```

装好后宿主日志会出现「庄方宜语音：已挂载 zfhVoice 服务」。
**若环境还没配好，日志会紧跟一段可照做的指引**（缺什么 + 敲什么命令），
不必去翻源码猜。

**给 AI 助手用的安装提示词**：`docs/安装提示词.md` —— 整段复制给
Claude Code / Cursor / DSH 自己，它会按步骤配好环境（含每一步的确认标志）。

### 挂给宿主的服务面

| 方法 | 用途 |
|---|---|
| `synthesize(text)` | 合成一句；自动确保服务在跑，失败返回 `null`（**绝不抛**） |
| `diagnose()` | 环境体检 → `{ran, ok, failure:{kind,summary,commands}, guidance}` |
| `status()` | 只读状态（含常驻三态，见下） |
| `setResident(bool)` | 运行时切换常驻（桌宠设置页用） |
| `setIdleAction('stop'\|'unload')` | 切换空闲释放档位 |
| `start()` / `stop()` | 显式起 / 停**本插件起的**服务 |
| `configure(patch)` | 批量改配置 |

`status()` 里的**常驻三态必须分清楚**：

| 字段 | 含义 |
|---|---|
| `resident` | **策略意图**（配置里写的要不要常驻） |
| `processAlive` | 服务进程在不在 |
| `modelLoaded` | **模型真实是否在显存里**（`true`/`false`/`null`=未知） |

配了常驻但模型还在加载时：`resident: true` 而 `modelLoaded: false`。
桌宠设置页据此显示「运行中（模型已加载）」/「运行中（已释放显存）」/「未运行」。

### 插件配置（全部可选，留空即自动）

| 键 | 默认 | 说明 |
|---|---|---|
| `python` | 自动 | 找 `.venv` → `ZFH_VOICE_PYTHON` → PATH |
| `repoDir` | 插件自身目录 | 插件知道自己装在哪，**不需要绝对路径** |
| `backend` | `auto` | `auto` / `torch`（快，占显存）/ `onnx`（慢，省显存） |
| `gsvRoot` | 自动 | GPT-SoVITS 检出目录（torch 后端需要） |
| `modelDir` | 自动 | 等价 `ZFH_MODEL_DIR` |
| `host` / `port` | `127.0.0.1` / `8765` | 本地 HTTP 服务 |
| `resident` | `false` | **常驻**：一直占显存但合成立刻开始；关闭则空闲后释放 |
| `idleStopSec` | `300` | 非常驻时空闲多久释放（0 = 不释放） |
| `idleAction` | `stop` | 释放方式：`stop` 杀进程（全释放）/ `unload` 只卸模型（显存同样释放、恢复快） |
| `startTimeoutMs` | `120000` | 等模型加载的上限（CUDA 首次较慢） |

### 三条纪律（有测试盯着）

- **只停自己起的**：用户在终端手动 `python -m zfh_voice serve` 时，
  插件只连接、绝不终止；
- **秒退不重启**：依赖没装/模型缺失时自动重启会变成重启风暴 ——
  10 秒内退出只记错误，等用户显式操作；
- **拿不准就报"未知"**：探测失败时 `modelLoaded` 返回 `null` 而不是猜 `false`，
  失败分类退 `unknown` 并透传原文 —— 宁可说不知道，也不给错结论。

自测：`npm test`（57 项，覆盖 python 解析/命令组装/决策矩阵/监管器/
常驻三态与两级释放/环境探测与失败分类）。

---

## 与桌宠集成

桌宠侧只需调用 HTTP 接口，无需关心本仓库的实现细节：

```
桌宠 → POST http://127.0.0.1:8765/tts {"text": "..."} → audio/wav → 播放
```

若不想常驻服务，也可以直接调 CLI 生成到文件后播放：

```bash
python -m zfh_voice say "台词" -o cache/line.wav
```

固定台词建议走 CLI + 缓存，临时台词走 HTTP。

---

## 基准数据

测试机：RTX 5060 Laptop 8GB / 32 核 CPU / Windows

| 方案 | 模型体积 | RTF | 一句话（6 秒音频）耗时 |
|---|---:|---:|---:|
| torch + CUDA | ~14 GB（含环境） | **0.45** | ~3 s |
| torch + CPU | ~2.6 GB | 1.60 | ~10 s |
| **onnx + CPU**（默认） | **1.5 GB** | 6.1 | ~37 s |

说明：

- **ONNX 的瓶颈是自回归解码**：逐 token 生成（约 130 步），
  每步都要把 24 层 KV cache 传进传出，单步约 170ms。
- 精度对速度几乎无影响（fp32 RTF 6.12 / fp16 6.09）——瓶颈不在算力。
- 首次调用有预热开销（约 10s），之后走缓存或连续批量明显更快。
- **适合批量预生成，不适合逐字实时交互。**

### 质量指标

| 指标 | 数值 |
|---|---|
| 内容准确率（ASR 反查 CER） | 0.000 ~ 0.045 |
| 与 torch 版音色一致度（频谱包络相关） | 0.968 ~ 0.974 |

---

## 故障排查

**第一步永远是 `python -m zfh_voice doctor`** —— 它不需要任何依赖就能跑，
会直接告诉你缺什么、该敲什么命令。

| 现象 | 原因与处理 |
|---|---|
| 不知道缺什么 | `python -m zfh_voice doctor`（人看）/ `doctor --json out.json`（程序读） |
| 插件日志说「环境还没配好 —— 缺 Python 依赖」 | 照日志给的命令 `pip install -r requirements.txt` |
| 插件日志说「模型还没下载」 | `python download_models.py` |
| 插件日志说「找不到 python 解释器」 | 装 Python 3.9+，或设 `ZFH_VOICE_PYTHON`，或填插件配置 `python` |
| `找不到可用的模型目录` | 报错里已列出每个位置缺几项并给出对应命令；或设 `ZFH_MODEL_DIR` |
| 只有 torch 权重、没下 ONNX | 正常：`doctor`/`status` 按后端分别校验，ONNX 与 torch 权重互不通用 |
| `参考音频时长超出 3~10s` | 裁剪参考音频，或换一段 |
| 输出听起来不像本人 | 参考音频不合适：换干净的 3~10 秒单句 |
| 下载模型很慢 | 优先走 ModelScope；慢时可挂代理，或用 `--check` 看已下好的部分 |
| 装完插件仍报未运行 | 重启 DSH（ESM 缓存）；改 `index.js` 后 disable→enable 不够 |
| 明明配了常驻却显示"已释放" | 看 `modelLoaded` 而不是 `resident`：后者是意图，前者是真实状态 |
| `ZFH_G2PW_DIR 未设置` | 直接用 `TTS()` 会自动设置；手动 import 内嵌前端才会遇到 |

---

## 协议

代码 MIT。内嵌的文本前端来自 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)（MIT），
改动已在代码中标注。详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

音色模型基于游戏角色语音微调，**仅供个人学习与同人创作**，
商业使用请自行取得授权。

---

## 更新日志

见 [CHANGELOG.md](./CHANGELOG.md)。
