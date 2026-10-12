/**
 * dsh-zhuang-fangyi-voice · DSH 插件（宿主半边）
 *
 * ── 它是什么 ──────────────────────────────────────────────────────────────
 *
 * 这个仓库的 Python 部分（`src/zfh_voice`）是**推理库**：给一句中文，出 WAV。
 * 本文件是它的 **DSH 插件外壳**，负责三件事：
 *
 *   1. **托管 Python 服务的进程**（启动 / 保活 / 空闲释放显存）；
 *   2. 把合成能力挂成宿主服务 **`zfhVoice`**（其它插件 `ctx.get('zfhVoice')` 就能用）；
 *   3. 给用户一个可配的接缝（后端、python、端口、常驻策略）。
 *
 * 桌宠插件就是这么用的：它拿到 `zfhVoice` 后把 AI 的回答合成出来播放。
 * **没装本插件时桌宠仍然能用固定台词** —— 两个插件彼此独立。
 *
 * ── 为什么不需要任何绝对路径 ──────────────────────────────────────────────
 *
 * 插件**知道自己装在哪**（`import.meta.url`）。所以：
 *   · 仓库目录默认 = 本文件所在目录；
 *   · python 默认 = 仓库里的 `.venv` → 环境变量 → PATH 上的 `python`；
 *   · 模型目录 = Python 侧的 `ZFH_MODEL_DIR` / `<repo>/models` / `~/.cache`。
 *
 * 于是同一份代码在任何人机器上都能跑，配置项全部是**可选覆盖**。
 *
 * ── 常驻 vs 释放（为什么这是个开关而不是常量）─────────────────────────────
 *
 * 启动服务要加载模型：onnx 约 10 秒；torch+CUDA 要把权重搬进显存，更慢，
 * 而且加载完长期占着显存。两个诉求冲突，所以让用户选：
 *
 *   · `resident: true` —— 一直活着；合成立刻开始。代价：显存常占。
 *   · `resident: false` —— 需要时才起，空闲 `idleStopSec` 秒后停掉释放显存。
 *
 * ── 两条纪律 ──────────────────────────────────────────────────────────────
 *
 *   · **只停自己起的**：用户在终端手动 `python -m zfh_voice serve` 时，
 *     插件只连接、绝不终止（`ours` 标志就是这件事的账本）。
 *   · **秒退不重启**：依赖没装/模型缺失/端口占用时自动重启会变成重启风暴。
 *     10 秒内退出只记 `lastError`，等用户显式操作。
 *
 * @module dsh-zhuang-fangyi-voice
 */

import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * 子进程 stdio 的**唯一合法形状**（宿主 `SubprocessStdio` 契约，实测确认）：
 *
 *     interface SubprocessStdio {
 *       stdin:  'ignore' | 'pipe' | {data}
 *       stdout: 'pipe' | 'inherit' | {maxBytes}
 *       stderr: 'pipe' | 'inherit' | {maxBytes}
 *     }
 *
 * 两个坑：
 *   ① `stdio` 必须是**对象** —— 传字符串会让宿主读到 `undefined.maxBytes`，
 *      报 `Cannot read properties of undefined (reading 'maxBytes')`，
 *      还被归类成 spawn-failed（看起来像"进程起不来"，其实是参数形状错）；
 *   ② stdout/stderr **没有 `'ignore'`** —— 只有 pipe / inherit / {maxBytes}。
 *
 * 用 `{maxBytes}` 收集模式而不是 `'pipe'`：能读回输出做诊断
 *（服务会打印"预热推理后端…"/"服务已启动"），而 pipe 需要自己消费流否则会堵。
 */
export const STDIO_CAPTURE = Object.freeze({
  stdin: 'ignore',
  stdout: { maxBytes: 256 * 1024 },
  stderr: { maxBytes: 256 * 1024 }
})
/** 本插件的安装目录 —— 一切相对路径的锚点（因此不需要绝对路径配置）。 */
const HERE = path.dirname(fileURLToPath(import.meta.url))

export const name = 'dsh-zhuang-fangyi-voice'

/** 服务名（桌宠通过它取合成能力）。 */
export const SERVICE_NAME = 'zfhVoice'

/** 空闲多久停掉服务释放显存（秒）—— 默认 5 分钟。 */
export const DEFAULT_IDLE_STOP_SEC = 300

/**
 * 空闲时怎么释放 —— 两档，默认 `stop`（保持既有行为）：
 *
 *   · `stop`   杀掉服务进程：显存全释放，恢复最慢（进程启动+导入+加载+预热）。
 *   · `unload` 只卸模型、保留进程：**显存同样释放**，恢复更快（省掉启动与导入）。
 *
 * 之所以做成开关而不是常量：省显存与快恢复是一对矛盾，不同机器答案不同。
 */
export const DEFAULT_IDLE_ACTION = 'stop'

/** 合法取值（配置校验用白名单）。 */
export const IDLE_ACTIONS = ['stop', 'unload']

/** 跑 `doctor` 体检的超时（首次导入可能慢，给足时间但不无限等）。 */
export const DOCTOR_TIMEOUT_MS = 30000

/**
 * 默认随机种子。
 *
 * GPT-SoVITS 的 AR 解码不传 seed 时每次采样都不同，且**句子越短波动越大**：
 * 实测 4 字句时长变异系数 23.5%、10 字句 17.0%、24 字句 10.3%；
 * `让我想想` 在随机种子下曾产出 0.74s（正常 1.22s）—— 近乎截断。
 *
 * 固定种子牺牲一点多样性，换来可复现的稳定输出。设 `null` 可恢复随机。
 */
/** v0.2.0：音色风格清单（同一音色，不同语气）。 */
export const STYLES = ['normal', 'firm', 'gentle']
/** v0.2.0：默认风格 —— normal 的参考音频就是 v0.1.0 的 ref/default.wav。 */
export const DEFAULT_STYLE = 'normal'
/** v0.2.0：风格显示名（服务没跑时兜底用）。 */
export const STYLE_LABELS = {
  normal: '正常说话',
  firm: '语气强硬',
  gentle: '语音娇'
}

export const DEFAULT_SEED = 42

/**
 * 默认亮度补偿：`'auto'` = **按模型版本自动选校准值**。
 *
 * 为什么要 auto：两种模型的**起点不同**，同一个数字会一个补齐、一个过冲。
 * 实测长文本（421 秒全文，目标为 48k 原声）：
 *
 *   v2    未补偿：8–12k 偏亮 4.2dB、12–16k 已对齐 → 需要 **+3dB**
 *   v2Pro 未补偿：8–12k 已对齐、12–16k 仅差 1.2 → 加 3dB 会**过冲**
 *                （频谱偏差 0.88 → 1.92）
 *
 * 所以默认 `'auto'`（v2 → 3，v2Pro → 0），用户**不必记住"换模型要改数字"**。
 *
 * 想手动控制就给数字：`0` = 显式关闭（逐字节不改动输出），`3` = 强制 +3dB。
 *
 * 说明：这是**听感补偿**（恢复 48k→32k 重采样造成的频谱倾斜），
 * **不是信息恢复** —— 素材本身有效带宽约 16kHz，升采样补不回真实信息。
 */
export const DEFAULT_BRIGHTNESS_DB = 'auto'

/** 启动后等健康检查的最长时间（CUDA 下加载模型可能几十秒）。 */
export const DEFAULT_START_TIMEOUT_MS = 120000

/** 短命退出判定：低于它就认为"起不来"，不再自动重启。 */
export const RAPID_EXIT_MS = 10000

/** 健康探测间隔。 */
export const HEALTH_POLL_MS = 15000

/** 合成结果体积上限（WAV 未压缩；6 秒 ≈ 380KB）。 */
export const MAX_SPEECH_BYTES = 8 * 1024 * 1024

// ─────────────────────────────────────────────────────────────────────────────
// 纯函数部分（可单测，见 tools/test-plugin.mjs）
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 找一个能用的 python（**全部相对插件自身**，不含任何绝对路径硬编码）。
 *
 * 顺序：显式配置 → 环境变量 → 插件目录下的虚拟环境 → PATH。
 *
 * @param {object} o
 * @param {string} [o.python] - 显式配置
 * @param {string} [o.repoDir] - 插件目录
 * @param {(p: string) => boolean} [o.exists]
 * @param {object} [o.env]
 * @returns {string} 可执行文件名或路径（找不到就返回 `python`，让 PATH 兜底）
 */
export function resolvePython (o = {}) {
  const explicit = String(o.python ?? '').trim()
  if (explicit !== '') return explicit
  const env = o.env ?? (typeof process !== 'undefined' ? process.env : {})
  const fromEnv = String(env.ZFH_VOICE_PYTHON ?? '').trim()
  if (fromEnv !== '') return fromEnv
  const repoDir = String(o.repoDir ?? '').trim()
  const exists = typeof o.exists === 'function' ? o.exists : (() => false)
  if (repoDir !== '') {
    // 常见虚拟环境位置（Windows 与 POSIX 都试）——**相对插件目录**，可移植。
    for (const rel of [
      ['.venv', 'Scripts', 'python.exe'], ['.venv', 'bin', 'python'],
      ['venv', 'Scripts', 'python.exe'], ['venv', 'bin', 'python']
    ]) {
      const p = path.join(repoDir, ...rel)
      if (exists(p)) return p
    }
  }
  // 交给 PATH（名字由平台决定）
  return process.platform === 'win32' ? 'python' : 'python3'
}

/**
 * 组装启动命令。
 *
 * CLI 形状（本仓库 `cli.py`）——**全局参数必须在子命令前面**：
 *
 *     python -m zfh_voice --backend torch --gsv-root <dir> serve --host 127.0.0.1 --port 8765
 *
 * @param {object} o
 * @param {string} o.python
 * @param {string} o.repoDir
 * @param {string} [o.backend] - `auto` | `torch` | `onnx`
 * @param {string} [o.gsvRoot]
 * @param {string} [o.modelDir]
 * @param {string} [o.host]
 * @param {number} [o.port]
 * @returns {{argv: string[], cwd: string, env: Record<string,string>}|null}
 */
export function buildServeCommand (o = {}) {
  const python = String(o.python ?? '').trim()
  const repoDir = String(o.repoDir ?? '').trim()
  if (python === '' || repoDir === '') return null
  const host = String(o.host ?? '').trim() || '127.0.0.1'
  const port = Number.isFinite(Number(o.port)) && Number(o.port) > 0
    ? Math.floor(Number(o.port))
    : 8765
  const argv = [python, '-m', 'zfh_voice']
  const backend = String(o.backend ?? '').trim()
  // `auto` = 不传，交给 Python 侧自己选（它有二选一的逻辑）。
  if (backend === 'torch' || backend === 'onnx') argv.push('--backend', backend)
  const gsv = String(o.gsvRoot ?? '').trim()
  if (gsv !== '') argv.push('--gsv-root', gsv)
  const model = String(o.modelDir ?? '').trim()
  if (model !== '') argv.push('--model-dir', model)
  // seed / brightnessDb 必须放在全局参数区（子命令之前），
  // 否则 argparse 会当成 `serve` 自己的参数而报错。
  const seed = o.seed
  if (seed === null) {
    argv.push('--random-seed')            // 显式要求随机
  } else if (Number.isFinite(Number(seed))) {
    argv.push('--seed', String(Math.floor(Number(seed))))
  }
  // brightnessDb：`'auto'`（默认）**不传参数** —— 让 Python 侧按它加载到的
  // 模型版本自己选校准值（v2 → 3，v2Pro → 0）。传数字才显式覆盖。
  const br = o.brightnessDb
  if (typeof br === 'number' && Number.isFinite(br)) {
    argv.push('--brightness-db', String(br))
  }
  argv.push('serve', '--host', host, '--port', String(port))
  return {
    argv,
    cwd: repoDir,
    // `src/` 布局：从仓库直接跑必须把 `src` 放进 PYTHONPATH，否则
    // `python -m zfh_voice` 报 ModuleNotFoundError（没 pip install 时）。
    // 这一条让用户**不必先安装**就能跑起来。
    env: { PYTHONPATH: path.join(repoDir, 'src') }
  }
}

/**
 * **纯函数**：由 `serve` 的启动信息推出对应的 `doctor --json` 命令。
 *
 * 复用同一个 python 与同一批全局参数，保证"体检的就是要跑的那套环境"。
 *
 * @param {object} o
 * @param {{argv: string[], cwd: string, env: object}|null} o.launch
 * @param {string} o.jsonPath
 * @returns {{argv: string[], cwd: string, env: object}|null}
 */
export function buildDoctorCommand (o = {}) {
  const base = o.launch
  const jsonPath = String(o.jsonPath ?? '').trim()
  if (base === null || base === undefined) return null
  if (!Array.isArray(base.argv) || base.argv.length === 0) return null
  if (jsonPath === '') return null
  const python = base.argv[0]
  const at = base.argv.indexOf('serve')
  // ⚠️ serve 的 argv 形状是 [python, '-m', 'zfh_voice', ...全局参数, 'serve', ...]，
  // 要复用的**只有全局参数** —— 必须跳过 `-m zfh_voice`，否则会拼出
  //   python -m zfh_voice -m zfh_voice --backend ... doctor
  // 这种命令（Python 直接报错退出，JSON 永远不生成）。
  let start = 1
  const iM = base.argv.indexOf('-m')
  if (iM >= 0 && base.argv[iM + 1] === 'zfh_voice') start = iM + 2
  const globals = (at > start) ? base.argv.slice(start, at) : []
  return {
    argv: [python, '-m', 'zfh_voice', ...globals, 'doctor', '--json', jsonPath],
    cwd: base.cwd,
    env: base.env
  }
}

/**
 * **纯函数**：把 doctor 的体检结果翻译成"到底是什么问题 + 该敲什么命令"。
 *
 * 分类只在**有把握**时下结论；拿不准一律 `unknown` 并透传原文 ——
 * 宁可说"我不知道"，也不能因为解析失败就谎称"没问题"。
 *
 * @param {object|null} report - doctor --json 的结果
 * @param {string} [fallback] - 没有报告时的兜底描述
 * @param {object} [opts]
 * @param {string} [opts.backend] - 当前配置的后端（`auto`/`torch`/`onnx`）。
 *   **判 aux 是否必需要靠它**：只有 ONNX 路径需要模型目录里的
 *   G2PWModel + tokenizer；torch 路径从 gsvRoot 读，不要求那套文件。
 * @returns {{kind: string, summary: string, commands: string[]}}
 */
export function classifyFailure (report, fallback = '服务起不来，原因未知', opts = {}) {
  if (report === null || report === undefined || typeof report !== 'object') {
    return { kind: 'unknown', summary: fallback, commands: [] }
  }
  const deps = Array.isArray(report.deps_missing) ? report.deps_missing : []
  const inst = report.installed ?? {}
  // aux（G2PW + tokenizer）只有 ONNX 路径要；`auto` 时按"有 torch 权重就走 torch"推。
  const backend = String(opts.backend ?? 'auto')
  const auxRequired = backend === 'onnx' || (backend !== 'torch' && inst.torch_ok !== true)

  if (deps.length > 0) {
    return {
      kind: 'missing-deps',
      summary: `缺 Python 依赖：${deps.join(', ')}`,
      commands: ['pip install -r requirements.txt']
    }
  }
  if (inst.onnx_ok !== true && inst.torch_ok !== true) {
    return {
      kind: 'missing-models',
      summary: 'Python 依赖齐了，但模型还没下载',
      commands: [
        'python download_models.py            # ONNX（默认 fp16，约 1.4GB）',
        'python download_models.py --with-torch   # 需要 torch 后端时'
      ]
    }
  }
  if (inst.aux_ok !== true || inst.ref_ok !== true) {
    // ⭐ **torch 后端不需要模型目录里的辅助包**（G2PWModel + tokenizer）——
    // 那套文本前端它直接从 `gsvRoot`（GPT-SoVITS 检出）读。
    // 所以"torch 权重齐 + 参考音频在"就不该报缺 aux：否则会给出误导性指引
    //（实测：合成明明成功，却打印"环境还没配好，去补齐辅助包"，用户会白折腾）。
    //
    // 只有在**将要走 ONNX 路**时 aux 才是硬需求：
    //   · 显式 `backend: 'onnx'`；
    //   · 或 `auto` 且 torch 权重不在（那就会选 onnx）。
    const wantsTorch = !auxRequired
    if (inst.ref_ok === true && inst.torch_ok === true && wantsTorch) {
      return {
        kind: 'aux-not-needed-torch',
        summary: 'torch 路径就绪（文本前端取自 GPT-SoVITS 检出，模型目录不需要辅助包）',
        commands: []
      }
    }
    return {
      kind: 'missing-aux',
      summary: '模型在，但文本前端辅助文件或默认参考音频缺失',
      commands: ['python download_models.py   # 重新补齐辅助包（约 1MB）']
    }
  }
  return { kind: 'unknown', summary: fallback, commands: [] }
}

/**
 * **纯函数**：把体检结果整理成一段给人/给 agent 看的多行指引。
 *
 * @param {object} probe - createEnvProbe() 的返回值
 * @param {string} [docPath] - 详细文档路径
 * @returns {string}
 */
export function formatSetupGuidance (probe, docPath = 'docs/安装提示词.md', opts = {}) {
  const lines = []
  // ⚠️ "体检没拿到结果"（ran=false / 没 report）与"体检跑了但环境不全"
  //（ok=false 且有 report）是**两件事**，不能合并成一句。
  //
  // 这里踩过一个真实的坑：原先把 `probe.ok !== true` 也归进这个分支，
  // 于是本机（torch 路径完全可用、只是模型目录里没有 ONNX 那套辅助文件）
  // 打印的是"环境还没配好（自动体检没拿到结果）"—— 既吓人又指向错误方向，
  // 而 `classifyFailure` 里刚做好的"torch 路径就绪"判定根本没机会执行。
  const hasReport = probe?.report !== null && probe?.report !== undefined
  if (probe === null || probe.ran !== true || !hasReport) {
    lines.push('庄方宜语音：环境还没配好（自动体检没拿到结果）。')
    lines.push('  先手动跑一次：python -m zfh_voice doctor')
    if (probe?.error) lines.push(`  （体检失败原因：${probe.error}）`)
    lines.push(`  详细步骤见 ${docPath}`)
    return lines.join('\n')
  }
  const cls = classifyFailure(probe.report, undefined, opts)
  // ⭐ torch 路径就绪时 aux 缺失**不是问题** —— 不能打印"环境还没配好"，
  // 那会把用户引去补一个他根本不需要的辅助包（实测：合成成功却报这个）。
  if (cls.kind === 'aux-not-needed-torch') {
    lines.push(`庄方宜语音：${cls.summary}`)
    return lines.join('\n')
  }
  lines.push(`庄方宜语音：环境还没配好 —— ${cls.summary}`)
  if (cls.commands.length > 0) {
    lines.push('  在仓库目录执行：')
    for (const c of cls.commands) lines.push(`    ${c}`)
  } else {
    lines.push('  先跑体检看细节：python -m zfh_voice doctor')
  }
  const rep = probe.report
  if (rep?.gpu?.present === true) {
    lines.push(`  检测到 GPU：${rep.gpu.name ?? 'NVIDIA'}`
      + (rep.gpu.vram_mb ? `（${rep.gpu.vram_mb} MB）` : ''))
  }
  lines.push(`  完整步骤见 ${docPath}`)
  return lines.join('\n')
}

/**
 * 建环境探测器：跑 `doctor --json` 并读回结果。
 *
 * **为什么不收 stdout**：宿主 `ctx.subprocess` 的 stdio 契约未必保证能捕获输出，
 * 所以让 Python 侧把结果写成文件，这里读文件 —— 对宿主实现零假设。
 * 判完成也用"文件出现"而不是 handle.done（done 未必是 promise）。
 *
 * @param {object} o
 * @param {(spec: object) => object} [o.spawn]
 * @param {(p: string) => string} [o.readFile]
 * @param {(p: string) => void} [o.removeFile]
 * @param {(p: string) => string} [o.tmpFile] - 生成一个不冲突的临时文件路径
 * @param {(ms: number) => Promise<void>} [o.sleep]
 * @param {() => number} [o.now]
 * @param {number} [o.timeoutMs]
 */
export function createEnvProbe (o = {}) {
  const spawn = typeof o.spawn === 'function' ? o.spawn : null
  const readFile = typeof o.readFile === 'function' ? o.readFile : null
  const removeFile = typeof o.removeFile === 'function' ? o.removeFile : () => {}
  const tmpFile = typeof o.tmpFile === 'function' ? o.tmpFile : null
  const sleep = typeof o.sleep === 'function'
    ? o.sleep
    : ms => new Promise(r => setTimeout(r, ms))
  const now = typeof o.now === 'function' ? o.now : () => Date.now()
  const timeoutMs = Number.isFinite(o.timeoutMs) ? o.timeoutMs : DOCTOR_TIMEOUT_MS

  /**
   * @param {object|null} launch - serve 的启动信息（用来推 doctor 命令）
   * @returns {Promise<{ran: boolean, ok: boolean|null, report: object|null,
   *                    error?: string, raw?: string}>}
   */
  return async function probeEnvironment (launch) {
    const unavailable = { ran: false, ok: null, report: null }
    if (spawn === null || readFile === null || tmpFile === null) {
      return { ...unavailable, error: 'no-io' }
    }
    const jsonPath = tmpFile()
    const cmd = buildDoctorCommand({ launch, jsonPath })
    if (cmd === null) return { ...unavailable, error: 'no-launch' }

    let handle = null
    try {
      // ⚠️ `stdio` 必须是**对象**，且 stdout/stderr 只接受
      // `'pipe' | 'inherit' | {maxBytes}`（宿主 `SubprocessStdio` 契约，实测确认）。
      // 传字符串 `'ignore'` 会让宿主读到 `undefined.maxBytes` →
      // `Cannot read properties of undefined (reading 'maxBytes')`，
      // 而且错误被归类成 spawn-failed，看起来像"进程起不来"。
      handle = spawn({ ...cmd, stdio: STDIO_CAPTURE, graceMs: 2000 })
    } catch (error) {
      return { ...unavailable, error: `spawn-failed: ${error?.message ?? error}` }
    }

    const deadline = now() + timeoutMs
    let lastRaw = ''
    while (now() < deadline) {
      await sleep(300)
      try {
        const raw = readFile(jsonPath)
        if (typeof raw === 'string' && raw.trim().length > 2) {
          lastRaw = raw
          const report = JSON.parse(raw)
          removeFile(jsonPath)
          return { ran: true, ok: true, report }
        }
      } catch { /* 还没写出来，继续等 */ }
    }
    // 超时：别把进程留着
    try { handle?.terminate?.() } catch { /* 已经退了 */ }
    return {
      ...unavailable,
      error: lastRaw === '' ? 'timeout-no-report' : 'timeout-bad-json',
      raw: lastRaw || undefined
    }
  }
}

/**
 * **纯决策**：这一刻该对服务做什么。
 *
 * 穷举"常驻/非常驻 × 健康/不健康 × 谁起的 × 空闲多久"——这些组合在真实
 * 时序里很难复现，但两个方向的错都很贵且都发生在用户看不见的地方：
 * 该释放时不释放（显存白占）、不该杀时杀了（别人的进程）。
 *
 * @returns {{action: 'none'|'stop'|'restart', why: string}}
 */
export function nextServiceAction (s) {
  const resident = s.resident === true
  const ours = s.ours === true
  const healthy = s.healthy === true
  const launchable = s.launchable === true
  const idleMs = Number.isFinite(s.idleMs) ? s.idleMs : 0
  const idleLimitMs = Number.isFinite(s.idleLimitMs) ? s.idleLimitMs : 0
  // 空闲怎么释放：stop（杀进程）或 unload（只卸模型、留进程）
  const idleAction = IDLE_ACTIONS.includes(s.idleAction)
    ? s.idleAction
    : DEFAULT_IDLE_ACTION

  // ① 不是我们起的 → 永不碰它（用户自己开的服务）。
  if (!ours) return { action: 'none', why: 'not-ours' }

  // ② 我们起的但不健康 → 可能崩了。
  if (!healthy) {
    if (!resident) return { action: 'none', why: 'down-not-resident' }
    if (!launchable) return { action: 'none', why: 'down-no-launch' }
    if (s.rapidExit === true) return { action: 'none', why: 'down-rapid-exit' }
    return { action: 'restart', why: 'down-resident' }
  }

  // ③ 健康 + 我们起的：常驻留着；非常驻空闲够了就释放。
  if (resident) return { action: 'none', why: 'resident-keep' }
  if (idleLimitMs > 0 && idleMs >= idleLimitMs) {
    return { action: idleAction, why: `idle-${idleAction}` }
  }
  return { action: 'none', why: 'in-use' }
}

/** 解析 WAV 时长（ms）—— `POST /tts` 只给字节，时长得自己从头部读。 */
export function wavDurationMs (bytes) {
  try {
    if (bytes === null || bytes === undefined) return 0
    const b = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes)
    if (b.length < 44) return 0
    if (b.toString('ascii', 0, 4) !== 'RIFF' || b.toString('ascii', 8, 12) !== 'WAVE') return 0
    let byteRate = 0
    let dataLen = 0
    let off = 12
    while (off + 8 <= b.length) {
      const id = b.toString('ascii', off, off + 4)
      const size = b.readUInt32LE(off + 4)
      const body = off + 8
      if (id === 'fmt ' && body + 16 <= b.length) byteRate = b.readUInt32LE(body + 8)
      else if (id === 'data') dataLen = Math.min(size, b.length - body)
      off = body + size + (size % 2)
    }
    if (byteRate <= 0 || dataLen <= 0) return 0
    return Math.round((dataLen / byteRate) * 1000)
  } catch {
    return 0
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 配置
// ─────────────────────────────────────────────────────────────────────────────

/** 插件配置默认值（全空 = 自动，见文件头"为什么不需要绝对路径"）。 */
export function defaultConfig () {
  return {
    /** python 可执行文件。留空 = .venv → ZFH_VOICE_PYTHON → PATH。 */
    python: '',
    /** 本仓库目录。留空 = 插件自身目录。 */
    repoDir: '',
    /** 推理后端：`auto`（默认）/ `torch`（快，占显存）/ `onnx`（慢，省显存）。 */
    backend: 'auto',
    /** GPT-SoVITS 检出目录（torch 后端需要）。留空 = 交给 Python 侧找。 */
    gsvRoot: '',
    /** 模型目录（等价 ZFH_MODEL_DIR）。留空 = <repo>/models → ~/.cache。 */
    modelDir: '',
    host: '127.0.0.1',
    port: 8765,
    /** 常驻：true = 一直占着（快）；false = 空闲后释放显存。 */
    resident: false,
    /** 非常驻时，空闲多少秒停掉（0 = 不停）。 */
    idleStopSec: DEFAULT_IDLE_STOP_SEC,
    /**
     * 空闲时怎么释放：`stop`（杀进程，全释放、恢复慢）
     * 或 `unload`（只卸模型、留进程，**显存同样释放**、恢复快）。
     */
    idleAction: DEFAULT_IDLE_ACTION,
    /**
     * 随机种子。GPT-SoVITS 的 AR 解码在随机种子下**句子越短波动越大**
     * （实测 4 字句时长变异 23.5%，24 字句仅 10.3%，偶发截断到正常时长一半）。
     * 默认 42 保证可复现。设 null = 走上游随机采样（不推荐）。
     */
    seed: DEFAULT_SEED,
    /**
     * 亮度补偿（dB）。模型输出 32kHz，而素材是 48kHz；下采样在 Nyquist
     * 附近的抗混叠滚降使 12–16kHz 比原声低约 2.8dB，听感偏「糊」。
     * 校准值 3.0（fc=12kHz, Q=1.0，实测 12–16kHz 误差 0.09dB）。
     * 0 = 关闭（默认，输出逐字节不变）。
     */
    brightnessDb: DEFAULT_BRIGHTNESS_DB,
    /**
     * v0.2.0：音色风格（normal / firm / gentle）。
     * 留空或未知值 -> normal，**行为与 v0.1.0 一致**。
     */
    style: DEFAULT_STYLE,
    /** 等模型加载的上限。 */
    startTimeoutMs: DEFAULT_START_TIMEOUT_MS
  }
}

/** 归一化配置（只认已知键、夹取数值、过滤非法后端）。 */
export function normalizeConfig (input, base = defaultConfig()) {
  const out = { ...base }
  if (input === null || typeof input !== 'object') return out
  for (const key of ['python', 'repoDir', 'gsvRoot', 'modelDir']) {
    if (typeof input[key] === 'string' && input[key].length <= 500) out[key] = input[key].trim()
  }
  if (input.backend === 'auto' || input.backend === 'torch' || input.backend === 'onnx') {
    out.backend = input.backend
  }
  if (typeof input.host === 'string' && /^[0-9a-zA-Z.:-]{1,64}$/.test(input.host)) {
    out.host = input.host
  }
  const port = Number(input.port)
  if (Number.isFinite(port) && port > 0 && port < 65536) out.port = Math.floor(port)
  // v0.2.0：style 只认已知值，未知一律回落 normal（不抛）
  if (typeof input.style === 'string' && STYLES.includes(input.style.trim().toLowerCase())) {
    out.style = input.style.trim().toLowerCase()
  }
  if (typeof input.resident === 'boolean') out.resident = input.resident
  if (IDLE_ACTIONS.includes(input.idleAction)) out.idleAction = input.idleAction
  // seed: 允许 null（显式要求随机）；数字则夹到 int32 范围
  if (input.seed === null) {
    out.seed = null
  } else if (input.seed !== undefined) {
    const sd = Number(input.seed)
    if (Number.isFinite(sd) && sd >= 0 && sd <= 2147483647) out.seed = Math.floor(sd)
  }
  // brightnessDb: `'auto'`（按模型版本自动选）或 0~12 的数字（再高会明显过亮）
  if (input.brightnessDb !== undefined) {
    if (typeof input.brightnessDb === 'string'
        && input.brightnessDb.trim().toLowerCase() === 'auto') {
      out.brightnessDb = 'auto'
    } else {
      const br = Number(input.brightnessDb)
      if (Number.isFinite(br) && br >= 0 && br <= 12) out.brightnessDb = br
    }
  }
  const idle = Number(input.idleStopSec)
  if (Number.isFinite(idle) && idle >= 0 && idle <= 86400) out.idleStopSec = Math.floor(idle)
  const timeout = Number(input.startTimeoutMs)
  if (Number.isFinite(timeout) && timeout >= 1000 && timeout <= 600000) {
    out.startTimeoutMs = Math.floor(timeout)
  }
  return out
}

// ─────────────────────────────────────────────────────────────────────────────
// 运行时（HTTP 客户端 + 进程监管）
// ─────────────────────────────────────────────────────────────────────────────

/** 数字或 null（区分"0"与"没有这个字段"）。 */
function numOrNull (v) {
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

/** 建 TTS HTTP 客户端（只依赖 Python 侧已文档化的契约）。 */
function createClient (cfg, logger) {
  const baseUrl = `http://${cfg.host}:${cfg.port}`
  const fetchImpl = typeof fetch === 'function' ? fetch : null

  const json = async (url, init, timeoutMs) => {
    if (fetchImpl === null) return null
    const ctrl = new AbortController()
    const timer = setTimeout(() => ctrl.abort(), timeoutMs)
    try {
      const res = await fetchImpl(url, { ...init, signal: ctrl.signal })
      if (res === null || res.ok !== true) return null
      return await res.json()
    } catch { return null } finally { clearTimeout(timer) }
  }

  return {
    baseUrl,
    async health () {
      const body = await json(`${baseUrl}/health`, { method: 'GET' }, 5000)
      if (body?.ok !== true) return null
      return {
        ok: true,
        backend: typeof body.backend === 'string' ? body.backend : null,
        version: typeof body.version === 'string' ? body.version : null,
        // ★ 模型真实是否加载在内存/显存里 —— 注意这**不等于**配置里的
        //   `resident`（那是"策略意图"）。两者必须分开上报，否则
        //   "配了常驻但还没加载完"会被误报成"已在显存里"。
        //   旧版 Python 侧没有这些字段 → 落到 null，表示"未知"。
        resident: typeof body.resident === 'boolean' ? body.resident : null,
        idleSeconds: numOrNull(body.idle_seconds),
        loads: numOrNull(body.loads),
        unloads: numOrNull(body.unloads)
      }
    },
    /** 让服务释放模型（保留进程）—— 显存释放，恢复比重启进程快 */
    async unload () {
      const body = await json(`${baseUrl}/unload`, { method: 'POST' }, 10000)
      return body?.ok === true ? body.unloaded !== false : false
    },
    async synthesize (text, style) {
      if (fetchImpl === null) return null
      const ctrl = new AbortController()
      const timer = setTimeout(() => ctrl.abort(), 60000)
      try {
        // v0.2.0：style 可选。不传时 body 里**不含 style 字段**，
        // 服务端行为与 v0.1.0 完全一致（不改变其全局状态）。
        const payload = (typeof style === 'string' && style !== '')
          ? { text, style }
          : { text }
        const res = await fetchImpl(`${baseUrl}/tts`, {
          method: 'POST',
          signal: ctrl.signal,
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload)
        })
        if (res === null || res.ok !== true) return null
        const buf = Buffer.from(await res.arrayBuffer())
        if (buf.length === 0 || buf.length > MAX_SPEECH_BYTES) return null
        if (buf.toString('ascii', 0, 4) !== 'RIFF') {
          logger?.warn?.('zhuang-fangyi-voice: /tts 返回的不是 RIFF/WAV，已丢弃')
          return null
        }
        return { bytes: buf, durationMs: wavDurationMs(buf) }
      } catch (error) {
        logger?.warn?.(`zhuang-fangyi-voice: 合成失败：${error?.message ?? error}`)
        return null
      } finally { clearTimeout(timer) }
    }
  }
}

/**
 * 进程监管器（启动/保活/空闲释放）。
 *
 * @param {object} o
 * @param {(spec: object) => object} o.spawn - 受管子进程接缝
 * @param {() => Promise<object|null>} o.health
 * @param {object} [o.logger]
 * @param {() => number} [o.now]
 */
export function createSupervisor (o = {}) {
  const spawn = typeof o.spawn === 'function' ? o.spawn : null
  const health = typeof o.health === 'function' ? o.health : async () => null
  // 让服务"只卸模型、保留进程"的接缝（与 health 同样是注入的，便于测试）
  const unloadRemote = typeof o.unload === 'function' ? o.unload : async () => false
  // 起不来时做一次环境体检，把"缺什么"问出来（失败路径才调用，慢一点无妨）
  const diagnose = typeof o.diagnose === 'function' ? o.diagnose : async () => null
  const logger = o.logger ?? null
  const now = typeof o.now === 'function' ? o.now : () => Date.now()

  let cfg = defaultConfig()
  let launch = null
  let handle = null
  let startedAt = 0
  let exitedAt = 0
  let lastUseAt = now()
  let lastError = null
  /** 结构化的失败原因（{kind, summary, commands}），供 UI/agent 直接消费 */
  let lastFailure = null
  let backend = null
  let starting = null
  /** 最近一次成功的 /health 快照 —— `status()` 用它回答"模型到底在不在"。 */
  let lastHealth = null

  let _spawnCount = 0
  let _unloadCount = 0

  const launchable = () => launch !== null && spawn !== null

  /**
   * 起不来的**兜底说明**：跑一次体检，把"缺依赖 / 缺模型"分清楚。
   * 探测本身失败不影响主流程 —— 只把已知信息记下来。
   */
  const enrichFailure = async (reason) => {
    let probe = null
    try { probe = await diagnose() } catch { probe = null }
    const cls = (probe !== null && probe.ran === true && probe.ok === true)
      ? classifyFailure(probe.report, reason)
      : { kind: 'unknown', summary: reason, commands: ['python -m zfh_voice doctor'] }
    if (cls.kind === 'unknown' && probe?.ran !== true) {
      cls.commands = ['python -m zfh_voice doctor   # 先看环境缺什么']
    }
    lastFailure = cls
    lastError = cls.summary + (cls.commands.length > 0 ? ` —— 修复：${cls.commands[0]}` : '')
    logger?.warn?.(probe !== null
      ? formatSetupGuidance(probe, undefined, { backend: cfg.backend })
      : `庄方宜语音：${cls.summary}\n  先跑：python -m zfh_voice doctor`)
    return cls
  }

  const probe = async () => {
    const h = await health()
    if (h !== null && h !== undefined && h.ok === true) {
      backend = h.backend ?? backend
      lastHealth = h
      return true
    }
    lastHealth = null
    return false
  }

  const ensure = async (opts = {}) => {
    lastUseAt = now()
    if (await probe()) { exitedAt = 0; return true }
    if (!launchable()) return false
    if (starting !== null) return starting
    starting = (async () => {
      try {
        logger?.info?.('庄方宜语音：启动推理服务…')
        handle = spawn({
          argv: launch.argv,
          cwd: launch.cwd,
          env: launch.env,
          // 同 `createEnvProbe`：stdio 必须是对象；用 `{maxBytes}` 收集模式
          // 还能顺手拿到服务日志（"预热推理后端…"/"服务已启动"），
          // 排查时比"什么都没有"强得多。
          stdio: STDIO_CAPTURE,
          graceMs: 2000
        })
        _spawnCount += 1
        startedAt = now()
        exitedAt = 0
        lastError = null
        lastFailure = null
        try {
          handle?.done?.then(value => {
            exitedAt = now()
            const ms = startedAt > 0 ? exitedAt - startedAt : 0
            if (ms < RAPID_EXIT_MS) {
              const reason = `服务启动后 ${Math.round(ms / 1000)} 秒就退出` +
                `（exit=${value?.exitCode ?? '?'}）`
              // 起不来才做体检：把"缺依赖 / 缺模型"分清楚再报给用户。
              // 异步做，不阻塞本次 ensure 的返回。
              void enrichFailure(reason).catch(() => {})
            }
            handle = null
          }).catch(() => {})
        } catch { /* done 不是 promise 也无妨 */ }
        if (opts.wait === false) return true
        const deadline = now() + cfg.startTimeoutMs
        while (now() < deadline) {
          if (await probe()) {
            logger?.info?.(`庄方宜语音：服务就绪（${Math.round((now() - startedAt) / 1000)}s，backend=${backend ?? '?'}）`)
            return true
          }
          if (handle === null && exitedAt > 0) return false
          await new Promise(r => setTimeout(r, 1000))
        }
        lastFailure = { kind: 'timeout', summary: `等 ${Math.round(cfg.startTimeoutMs / 1000)} 秒仍未就绪`, commands: ['把配置 startTimeoutMs 调大（CUDA 首次加载较慢）'] }
        lastError = lastFailure.summary
        logger?.warn?.(`庄方宜语音：${lastError} —— ${lastFailure.commands[0]}`)
        return false
      } catch (error) {
        const msg = String(error?.message ?? error)
        // spawn 直接抛（而不是"起来了又秒退"）最常见的原因就是解释器不存在
        if (/ENOENT|not found|no such file|不是内部或外部命令/i.test(msg)) {
          lastFailure = {
            kind: 'no-python',
            summary: `找不到 python 解释器：${launch?.argv?.[0] ?? 'python'}`,
            commands: ['装 Python 3.9+，或设置环境变量 ZFH_VOICE_PYTHON 指向解释器',
                       '或在插件配置里填 python']
          }
        } else {
          lastFailure = { kind: 'spawn-failed', summary: msg, commands: [] }
        }
        lastError = lastFailure.summary
        logger?.warn?.(`庄方宜语音：启动服务失败：${lastError}`
          + (lastFailure.commands[0] ? `\n  ${lastFailure.commands[0]}` : ''))
        return false
      } finally {
        starting = null
      }
    })()
    return starting
  }

  const stop = () => {
    if (handle === null) return false
    try {
      handle.terminate?.()
      logger?.info?.('庄方宜语音：已停止服务（释放显存）')
      handle = null
      startedAt = 0
      return true
    } catch (error) {
      logger?.warn?.(`庄方宜语音：停止服务失败：${error?.message ?? error}`)
      return false
    }
  }

  /**
   * 只卸模型、**保留进程** —— 显存同样释放，但下次合成省掉
   * "进程启动 + Python 导入"这段开销（CUDA 下这段并不便宜）。
   *
   * 关键：进程还在，所以 `handle` 不动 —— `ours` 账本必须继续认它，
   * 否则下一次 tick 会判成"不是我们起的"而不再管理。
   */
  const unloadModel = async () => {
    if (handle === null) return false
    if (lastHealth !== null && lastHealth.resident === false) return false // 已经释放过
    try {
      const ok = await unloadRemote()
      if (ok) {
        _unloadCount += 1
        logger?.info?.('庄方宜语音：已释放模型（进程保留，显存已回收）')
      }
      return ok
    } catch (error) {
      logger?.warn?.(`庄方宜语音：释放模型失败：${error?.message ?? error}`)
      return false
    }
  }

  const tick = async () => {
    const healthy = await probe().catch(() => false)
    const decision = nextServiceAction({
      resident: cfg.resident,
      idleAction: cfg.idleAction,
      ours: handle !== null || (exitedAt > 0 && startedAt > 0),
      healthy,
      idleMs: now() - lastUseAt,
      idleLimitMs: Math.max(0, Number(cfg.idleStopSec) || 0) * 1000,
      launchable: launchable(),
      rapidExit: lastError !== null && lastError.includes('就退出')
    })
    if (decision.action === 'stop') stop()
    else if (decision.action === 'unload') await unloadModel()
    else if (decision.action === 'restart') await ensure()
    return decision
  }

  return {
    setConfig (next, nextLaunch) {
      cfg = next
      if (nextLaunch !== undefined) launch = nextLaunch
    },
    ensure,
    stop,
    tick,
    markUse: () => { lastUseAt = now() },
    status () {
      return {
        // ── 常驻三态：意图 / 进程 / 模型 ──
        // 三者必须分开，否则"配了常驻但模型还没加载完"会被误报成已在显存里。
        resident: cfg.resident,                 // 策略意图（配置里写的）
        processAlive: handle !== null,          // 服务进程在不在
        ready: handle !== null,                 // 兼容旧名（= processAlive）
        // 契约：`boolean | null`。老版本 Python 侧没有 `resident` 字段时
        // 必须落到 null（未知），**绝不能**降级成 false（"没加载"）——
        // 前者是"不知道"，后者是错误结论。
        modelLoaded: (lastHealth === null || typeof lastHealth.resident !== 'boolean')
          ? null
          : lastHealth.resident,
        idleSeconds: lastHealth?.idleSeconds ?? null,
        loads: lastHealth?.loads ?? null,
        unloads: lastHealth?.unloads ?? null,
        idleAction: cfg.idleAction,

        launchable: launchable(),
        ours: handle !== null,
        pid: handle?.pid ?? null,
        backend,
        lastError,
        /** 结构化失败原因：{kind, summary, commands} —— UI/agent 可直接消费 */
        lastFailure,
        idleMs: now() - lastUseAt,
        idleStopSec: cfg.idleStopSec,
        launch: launch === null ? null : { argv: launch.argv, cwd: launch.cwd },
        spawnCount: _spawnCount,
        unloadCount: _unloadCount
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 插件入口
// ─────────────────────────────────────────────────────────────────────────────

/**
 * 当前后端**是否依赖 torch**（`onnx` 不依赖）。
 *
 * @param {string} backend
 * @returns {boolean}
 */
export function wantsTorchBackend (backend) {
  return String(backend ?? 'auto') !== 'onnx'
}

/**
 * **纯函数**：解释器不合适时给一条能直接照做的提示；没问题返回 null。
 *
 * 为什么需要它（新用户最容易踩的坑，而且是**静默**失败）：
 * 插件默认按 配置 → `ZFH_VOICE_PYTHON` → `<repo>/.venv` → **PATH** 找解释器。
 * 而 PATH 上那个 `python` 往往**没装 torch**（系统 Python、没激活的 conda…）。
 * 于是 torch 后端起不来，用户看到的是一句莫名其妙的启动失败 ——
 * 完全想不到是"用错了解释器"。
 *
 * @param {object} o
 * @param {string} o.python - 实际会用到的解释器
 * @param {string} o.backend - 配置的后端
 * @param {boolean|null} o.hasTorch - 探测结果（null = 没探测出来）
 * @param {string} [o.gsvRoot]
 * @returns {string|null}
 */
export function interpreterWarning (o = {}) {
  if (!wantsTorchBackend(o.backend)) return null
  if (o.hasTorch !== false) return null
  return [
    `庄方宜语音：解释器 ${o.python} 里**没有 torch**，torch 后端起不来。`,
    '  在插件配置里显式指定带 torch 的解释器（或设环境变量 ZFH_VOICE_PYTHON）：',
    '    python: <你的 python.exe 路径>',
    o.gsvRoot ? `  （当前的 gsvRoot = ${o.gsvRoot}）` : '  （还需要 gsvRoot 指向 GPT-SoVITS 检出目录）',
    '  只想要能出声、不在乎速度：把 backend 改成 onnx（会慢约 13 倍）。'
  ].join('\n')
}

/** 进程级缓存：同一个解释器只探一次（探测要 spawn，别每次 apply 都付一遍）。 */
const torchProbeCache = new Map()

/**
 * 探测解释器是否**装了** torch。
 *
 * ⚠️ 必须用 `find_spec` 而**不是** `import torch` —— 后者会把整个 torch 栈
 * 真的加载进来：**峰值 406MB / 1.9 秒**（实测）。而 find_spec 只定位包：
 * **10MB / 0.1 秒**，便宜 40 倍。
 *
 * 这不是抠性能 —— 我第一版就是 `import torch`，每次 `apply()` 都 spawn 一次，
 * 于是：测试套件 6 次 apply 翻腾 2.4GB；线上叠加常驻服务自己那份 torch 之后
 * 直接把机器的提交量打满，Node 报 `Committing semi space failed`、
 * Electron（DSH 本体）同样分配失败而**崩溃**。
 * 一个"体检"绝不该吃掉 400MB。
 *
 * @param {object} o
 * @param {(spec: object) => object|null} o.spawn
 * @param {string} o.python
 * @param {number} [o.timeoutMs]
 * @returns {Promise<boolean|null>} true/false；探测不了（起不来/超时）→ null
 */
export async function probeTorch (o = {}) {
  const spawn = typeof o.spawn === 'function' ? o.spawn : null
  if (spawn === null) return null
  const timeoutMs = Number.isFinite(o.timeoutMs) ? o.timeoutMs : 15000
  let handle = null
  const probe = 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("torch") else 1)'
  try {
    handle = spawn({
      argv: [o.python, '-c', probe],
      cwd: o.cwd ?? process.cwd(),
      stdio: { stdin: 'ignore', stdout: { maxBytes: 4096 }, stderr: { maxBytes: 4096 } },
      graceMs: 2000
    })
  } catch {
    return null
  }
  if (handle === null || handle === undefined) return null
  const done = handle.done
  if (done === null || done === undefined || typeof done.then !== 'function') return null
  const timeout = new Promise(resolve => setTimeout(() => resolve('timeout'), timeoutMs))
  try {
    const outcome = await Promise.race([Promise.resolve(done).catch(() => 'error'), timeout])
    if (outcome === 'timeout') {
      try { handle.terminate?.() } catch { /* 已退出 */ }
      return null
    }
    if (outcome === 'error') return null
    return outcome?.exitCode === 0
  } catch {
    return null
  }
}

/**
 * **纯函数**：从归一化配置推出 `buildServeCommand` 的入参。
 *
 * 存在的理由：`buildServeCommand` 的入参有 8 个以上，调用点若各写一遍，
 * 新增配置项时极易漏传（`seed` / `brightnessDb` 就漏过一次 —— 单元测试
 * 直接调 `buildServeCommand` 全绿，但配置根本到不了命令行）。
 * 收敛到这里之后，新增配置只需改这一处。
 *
 * @param {object} cfg - normalizeConfig 的产物
 * @param {string} python
 * @param {string} repoDir
 * @returns {object}
 */
export function serveOptions (cfg, python, repoDir) {
  return {
    python,
    repoDir,
    backend: cfg.backend,
    gsvRoot: cfg.gsvRoot,
    modelDir: cfg.modelDir,
    host: cfg.host,
    port: cfg.port,
    seed: cfg.seed,
    brightnessDb: cfg.brightnessDb
  }
}

/**
 * cordis 插件入口。
 *
 * @param {object} ctx - cordis Context
 * @param {object} [rawConfig] - 插件配置（cordis.patch.yml 或设置页给的）
 */
export function apply (ctx, rawConfig) {
  const logger = ctx.logger ?? console
  const cfg = normalizeConfig(rawConfig)

  const repoDir = cfg.repoDir !== '' ? cfg.repoDir : HERE
  const python = resolvePython({
    python: cfg.python,
    repoDir,
    exists: p => { try { return fs.existsSync(p) } catch { return false } }
  })
  const launch = buildServeCommand(serveOptions(cfg, python, repoDir))

  const client = createClient(cfg, logger)

  // ⭐ **解释器自检**：PATH 上那个 `python` 常常没装 torch（系统 Python、
  // 没激活的 conda…），而那是 torch 后端起不来的**静默**原因 ——
  // 用户只会看到一句莫名其妙的启动失败，想不到是"用错了解释器"。
  //
  // 两条纪律（都是踩过才知道的）：
  //   ① 探测用 `find_spec`，**不 import torch**（400MB vs 10MB，见 probeTorch 注释）；
  //   ② **同一解释器只探一次**（进程级缓存）—— apply 可能被调用多次
  //      （设置变化/HMR/测试），每次都 spawn 是浪费。
  if (wantsTorchBackend(cfg.backend)) {
    void (async () => {
      try {
        let probe = torchProbeCache.get(python)
        if (probe === undefined) {
          probe = probeTorch({
            python,
            cwd: repoDir,
            spawn: spec => { try { return subprocessSpawn(spec) } catch { return null } }
          })
          torchProbeCache.set(python, probe)
        }
        const hasTorch = await probe
        const warn = interpreterWarning({
          python, backend: cfg.backend, hasTorch, gsvRoot: cfg.gsvRoot
        })
        if (warn !== null) logger?.warn?.(warn)
      } catch { /* 自检失败不该影响加载 */ }
    })()
  }


  /** 宿主子进程接缝（doctor 与 serve 共用同一份） */
  const subprocessSpawn = spec => {
    let sub = null
    try {
      sub = typeof ctx.get === 'function' ? ctx.get('subprocess') : undefined
    } catch { sub = null }
    if (sub === null || sub === undefined || typeof sub.spawn !== 'function') {
      throw new Error('宿主未提供 ctx.subprocess，无法托管推理服务进程')
    }
    return sub.spawn(spec)
  }

  // 环境体检器：让 doctor 把结果写成 JSON，这里读回来（对宿主 stdio 零假设）
  const envProbe = createEnvProbe({
    spawn: spec => { try { return subprocessSpawn(spec) } catch { return null } },
    readFile: p => fs.readFileSync(p, 'utf8'),
    removeFile: p => { try { fs.rmSync(p, { force: true }) } catch { /* 无所谓 */ } },
    tmpFile: () => path.join(
      os.tmpdir(), `zfh-doctor-${process.pid}-${Date.now()}.json`)
  })
  /**
   * 跑环境体检。
   *
   * ⚠️ 必须传**带 `env` 的那份 launch**（不是 `supervisor.status().launch`）——
   * `status()` 有意只暴露 `{argv, cwd}`（那是对外的诊断快照，不该带环境变量），
   * 而 doctor 和 serve 一样需要 `PYTHONPATH=<repo>/src`（包没 pip install 时
   * 靠它才能 `python -m zfh_voice`）。
   *
   * 症状（实测）：体检永远报 `timeout-no-report`，而手动跑同一条命令 2.7 秒就出
   * JSON —— 因为探针起的那次 `python -m zfh_voice doctor` 直接
   * `No module named zfh_voice` 退出了，JSON 自然永远不生成。
   */
  const runDoctor = () => envProbe(launch)

  const supervisor = createSupervisor({
    spawn: subprocessSpawn,
    health: () => client.health(),
    unload: () => client.unload(),
    diagnose: runDoctor,
    logger
  })
  supervisor.setConfig(cfg, launch)

  // 周期检查：常驻保活 / 非常驻释放显存。
  ctx.effect(() => {
    const timer = setInterval(() => { void supervisor.tick().catch(() => {}) }, HEALTH_POLL_MS)
    timer.unref?.()
    return () => {
      clearInterval(timer)
      try { supervisor.stop() } catch { /* 已退出 */ }
    }
  }, 'zhuang-fangyi-voice: service supervisor')

  // 挂载后异步探一次环境。**不阻塞挂载**（挂载成功 ≠ 环境就绪，
  // 但环境没配好时用户必须立刻看到"缺什么、敲什么"，而不是等到点"试一句"才发现哑了）。
  ctx.effect(() => {
    let alive = true
    const run = async () => {
      const probe = await runDoctor().catch(() => null)
      if (!alive || probe === null) return
      if (probe.ran === true && probe.ok === true) {
        const cls = classifyFailure(probe.report, undefined, opts)
    // ⭐ torch 路径就绪时 aux 缺失**不是问题** —— 不能打印"环境还没配好"，
    // 那会把用户引去补一个他根本不需要的辅助包（实测：合成成功却报这个）。
    if (cls.kind === 'aux-not-needed-torch') {
      lines.push(`庄方宜语音：${cls.summary}`)
      return lines.join('\n')
    }
        if (cls.kind === 'unknown') return          // 环境没问题 → 不打扰
        logger?.warn?.(formatSetupGuidance(probe, undefined, { backend: cfg.backend }))
        return
      }
      logger?.warn?.(
        `庄方宜语音：环境体检没跑完（${probe.error ?? '未知原因'}）。\n` +
        '  手动跑一次：python -m zfh_voice doctor\n' +
        '  安装步骤见 docs/安装提示词.md')
    }
    void run().catch(() => {})
    return () => { alive = false }
  }, 'zhuang-fangyi-voice: startup env probe')

  /**
   * 对外服务：其它插件用它把一句话变成语音。
   *
   * 桌宠的用法：`const v = ctx.get('zfhVoice'); const r = await v.synthesize(text)`。
   */
  const service = {
    /** 合成一句（自动确保服务在跑；失败返回 null，绝不抛）。 */
    async synthesize (text, style) {
      if (typeof text !== 'string' || text.trim() === '') return null
      const ok = await supervisor.ensure()
      if (!ok) return null
      // v0.2.0：style 缺省 -> 用配置里的（cfg.style），再缺省 -> normal。
      // 旧调用方 `synthesize(text)` 的行为**不变**。
      const want = (typeof style === 'string' && style !== '')
        ? style
        : cfg.style
      const r = await client.synthesize(text, want)
      supervisor.markUse()
      if (r === null) return null
      const out = { ...r, backend: supervisor.status().backend }
      if (typeof want === 'string' && want !== '') out.style = want
      return out
    },

    /** v0.2.0：列出可用风格（不启动服务，纯查文件）。 */
    async styles () {
      const body = await json(`${baseUrl}/styles`, { method: 'GET' }, 8000)
        .catch(() => null)
      if (body?.ok === true && Array.isArray(body.styles)) {
        return { current: body.current ?? DEFAULT_STYLE, styles: body.styles }
      }
      // 服务没跑时退回配置里的静态清单
      return {
        current: cfg.style ?? DEFAULT_STYLE,
        styles: STYLES.map(name => ({
          name,
          label: STYLE_LABELS[name] ?? name,
          available: true,
          is_default: name === DEFAULT_STYLE
        }))
      }
    },

    /** v0.2.0：运行时切换风格（同步到服务端，桌宠设置页用）。 */
    async setStyle (name) {
      const want = (typeof name === 'string' && STYLES.includes(name.trim().toLowerCase()))
        ? name.trim().toLowerCase()
        : DEFAULT_STYLE
      this.configure({ style: want })
      const body = await json(`${baseUrl}/style`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ style: want })
      }, 8000).catch(() => null)
      return { style: body?.style ?? want, applied: body?.ok === true }
    },
    /** 健康检查（不启动服务）。 */
    health: () => client.health(),

    /**
     * 环境体检 —— 把"缺什么、该敲什么命令"以结构化形式给出来。
     *
     * DSH / 桌宠可以拿它当行动依据：`cls.kind` 决定怎么提示用户，
     * `cls.commands` 就是可直接执行的修复命令。
     * 探测失败时返回 `{ran:false}` 而不是抛（与 synthesize 的纪律一致）。
     */
    async diagnose () {
      const probe = await runDoctor().catch(() => null)
      if (probe === null) return { ran: false, ok: null, report: null, failure: null }
      const failure = (probe.ran === true && probe.ok === true)
        ? classifyFailure(probe.report)
        : { kind: 'probe-failed', summary: probe.error ?? '体检未完成', commands: ['python -m zfh_voice doctor'] }
      return {
        ran: probe.ran === true,
        ok: probe.ok ?? null,
        report: probe.report ?? null,
        failure,
        error: probe.error ?? null,
        guidance: formatSetupGuidance(probe, undefined, { backend: cfg.backend })
      }
    },

    /**
     * 运行时切换常驻（桌宠设置页的"让语音模型常驻"开关用它）。
     * `true` 时顺手拉起服务，不必等下一次 tick。
     */
    setResident (on) {
      this.configure({ resident: on === true })
      if (on === true) void supervisor.ensure().catch(() => {})
      return this.status()
    },

    /** 运行时切换空闲释放档位：`stop`（杀进程）或 `unload`（只卸模型）。 */
    setIdleAction (action) {
      this.configure({ idleAction: action })
      return this.status()
    },

    /** 只读状态。 */
    status: () => ({
      ...supervisor.status(),
      baseUrl: client.baseUrl,
      python,
      repoDir,
      backend_pref: cfg.backend
    }),
    /** 显式启动（会等模型加载完）。 */
    start: () => supervisor.ensure(),
    /** 停掉**本插件起的**服务（用户自己起的那份不会被动）。 */
    stop: () => supervisor.stop(),
    /** 由宿主插件（如桌宠）驱动配置 —— 让开关只有一处。 */
    configure (patch) {
      const next = normalizeConfig({ ...cfg, ...(patch ?? {}) }, cfg)
      const nextRepo = next.repoDir !== '' ? next.repoDir : HERE
      const nextLaunch = buildServeCommand(serveOptions(next, resolvePython({
        python: next.python,
        repoDir: nextRepo,
        exists: p => { try { return fs.existsSync(p) } catch { return false } }
      }), nextRepo))
      Object.assign(cfg, next)
      supervisor.setConfig(cfg, nextLaunch)
      void supervisor.tick().catch(() => {})
      return this.status()
    }
  }

  ctx.provide(SERVICE_NAME, service)
  ctx.effect(() => () => {
    try { supervisor.stop() } catch { /* 已退出 */ }
  }, 'zhuang-fangyi-voice: dispose service')

  logger.info?.(`庄方宜语音：已挂载 ${SERVICE_NAME} 服务（repo=${repoDir}，python=${python}，` +
    `resident=${cfg.resident}）`)
  return service
}

export default { name, apply }
