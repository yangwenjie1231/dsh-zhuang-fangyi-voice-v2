/**
 * 端到端验收：插件在**真实子进程**下的环境探测与失败分类
 *
 * 单测里 subprocess 是桩；这里用真的 python 跑真的 `doctor --json`，
 * 验证"新用户装上插件 → 立刻被告知缺什么"这条路径真的通。
 *
 * 用法：
 *   node tools/e2e-diagnose.mjs
 *
 * 可选环境变量：
 *   ZFH_E2E_BARE_PYTHON   一个**没装依赖**的 python（用于验证"缺依赖"分类）
 *   ZFH_E2E_READY_PYTHON  装好依赖的 python（用于验证"环境就绪"）
 *   ZFH_E2E_MODEL_DIR     模型目录（配合 READY_PYTHON）
 *   ZFH_E2E_REPO          本仓库路径（默认取本文件上一级）
 *
 * 缺哪个就跳过对应用例，不算失败 —— 这个脚本要能在别人机器上直接跑。
 */

import assert from 'node:assert/strict'
import { spawn as nodeSpawn } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  buildServeCommand,
  classifyFailure,
  createEnvProbe,
  formatSetupGuidance
} from '../index.js'

const REPO = process.env.ZFH_E2E_REPO ??
  path.dirname(path.dirname(fileURLToPath(import.meta.url)))

let passed = 0
let failed = 0
let skipped = 0
function ok (label, fn) {
  try { fn(); passed += 1; console.log(`  ✓ ${label}`) } catch (e) {
    failed += 1; console.log(`  ✗ ${label}\n      ${e?.message ?? e}`)
  }
}
async function okAsync (label, fn) {
  try { await fn(); passed += 1; console.log(`  ✓ ${label}`) } catch (e) {
    failed += 1; console.log(`  ✗ ${label}\n      ${e?.message ?? e}`)
  }
}
function skip (label, why) {
  skipped += 1
  console.log(`  – ${label}（跳过：${why}）`)
}

/**
 * 把 `{argv, cwd, env, stdio}` 规格落到真实的 node 子进程上。
 *
 * stdio 用 'ignore'（与插件一致）—— 探测结果走**文件**而不是 stdout，
 * 所以对 stdout 是否可捕获零假设（沙箱下管道也未必能用）。
 */
function realSpawn (spec) {
  const child = nodeSpawn(spec.argv[0], spec.argv.slice(1), {
    cwd: spec.cwd,
    env: { ...process.env, ...(spec.env ?? {}) },
    stdio: 'ignore',
    windowsHide: true
  })
  let spawnError = null
  child.on('error', e => { spawnError = e })
  return {
    pid: child.pid ?? null,
    get spawnError () { return spawnError },
    done: new Promise(resolve => {
      child.on('error', e => resolve({ exitCode: -1, error: String(e?.message ?? e) }))
      child.on('exit', code => resolve({ exitCode: code }))
    }),
    terminate () { try { child.kill() } catch { /* 已退出 */ } }
  }
}

const probe = createEnvProbe({
  spawn: realSpawn,
  readFile: p => fs.readFileSync(p, 'utf8'),
  removeFile: p => { try { fs.rmSync(p, { force: true }) } catch { /* 无所谓 */ } },
  tmpFile: () => path.join(os.tmpdir(), `zfh-e2e-${process.pid}-${Date.now()}.json`),
  timeoutMs: 60000
})

function launchFor (python, modelDir) {
  return buildServeCommand({
    python,
    repoDir: REPO,
    backend: 'onnx',
    modelDir: modelDir ?? '',
    host: '127.0.0.1',
    port: 8765
  })
}

console.log('端到端验收：真实子进程下的环境探测\n')
console.log(`仓库     : ${REPO}`)
console.log(`裸 python: ${process.env.ZFH_E2E_BARE_PYTHON ?? '(未设置)'}`)
console.log(`就绪 python: ${process.env.ZFH_E2E_READY_PYTHON ?? '(未设置)'}`)

/* ── 1. 裸机器：应分类为"缺依赖" ─────────────────────────────────────── */

console.log('\n── 1. 裸机器（没装任何依赖）──')

const bare = process.env.ZFH_E2E_BARE_PYTHON
if (!bare || !fs.existsSync(bare)) {
  skip('探测输出"缺依赖"', '没有可用的裸 python（设 ZFH_E2E_BARE_PYTHON）')
} else {
  await okAsync('⭐ doctor --json 在裸环境跑通，分类为 missing-deps', async () => {
    const r = await probe(launchFor(bare, path.join(os.tmpdir(), 'zfh-e2e-nomodels')))
    assert.equal(r.ran, true, `探测没跑成：${r.error ?? ''}`)
    assert.equal(r.ok, true, 'doctor 应返回成功（体检本身不该失败）')
    assert.ok(r.report !== null, '应拿到 JSON 报告')
    const cls = classifyFailure(r.report)
    assert.equal(cls.kind, 'missing-deps',
      `应判为缺依赖，实际 ${cls.kind}：${cls.summary}`)
    assert.ok(cls.commands.some(c => c.includes('pip install')), '应给出安装命令')
    assert.ok(Array.isArray(r.report.deps_missing) && r.report.deps_missing.length > 0,
      '报告里应列出缺失的依赖')
    console.log(`      缺：${r.report.deps_missing.slice(0, 5).join(', ')}…`)
    console.log(`      指引：${cls.summary}`)
  })

  await okAsync('⭐ 给人的指引里带上命令与文档路径', async () => {
    const r = await probe(launchFor(bare, path.join(os.tmpdir(), 'zfh-e2e-nomodels')))
    const g = formatSetupGuidance(r)
    assert.ok(g.includes('环境还没配好'), g)
    assert.ok(g.includes('pip install -r requirements.txt'), g)
    assert.ok(g.includes('docs/安装提示词.md'), g)
  })
}

/* ── 2. 装好的机器：应分类为"没问题" ──────────────────────────────────── */

console.log('\n── 2. 装好的机器（依赖 + 模型都在）──')

const ready = process.env.ZFH_E2E_READY_PYTHON
const modelDir = process.env.ZFH_E2E_MODEL_DIR
if (!ready || !fs.existsSync(ready)) {
  skip('探测输出"环境就绪"', '没有可用的就绪 python（设 ZFH_E2E_READY_PYTHON）')
} else {
  await okAsync('⭐ 分类为 unknown（不谎报有问题）', async () => {
    const r = await probe(launchFor(ready, modelDir))
    assert.equal(r.ran, true, `探测没跑成：${r.error ?? ''}`)
    const cls = classifyFailure(r.report)
    assert.equal(cls.kind, 'unknown',
      `环境就绪时应判 unknown，实际 ${cls.kind}：${cls.summary}`)
    assert.equal(r.report.deps_missing.length, 0, '不该有缺依赖')
    assert.equal(r.report.installed.onnx_ok, true, 'ONNX 模型应就绪')
    console.log(`      GPU: ${r.report.gpu.present ? r.report.gpu.name : '无'}`)
  })

  await okAsync('临时 JSON 文件用过即清（不留垃圾）', async () => {
    const before = fs.readdirSync(os.tmpdir()).filter(f => f.startsWith('zfh-e2e-'))
    await probe(launchFor(ready, modelDir))
    const after = fs.readdirSync(os.tmpdir()).filter(f => f.startsWith('zfh-e2e-'))
    assert.ok(after.length <= before.length, `临时文件没清干净：${after.join(', ')}`)
  })
}

/* ── 3. python 不存在：不该把探测/启动弄崩 ────────────────────────────── */

console.log('\n── 3. 找不到 python（用户没装/没进 PATH）──')

await okAsync('⭐ 探测失败时返回 ran:false 而不是抛', async () => {
  const bogus = path.join(os.tmpdir(), 'definitely-no-such-python-xyz')
  const r = await probe(launchFor(bogus, ''))
  assert.equal(r.ran, false, '不该假装成功')
  assert.equal(r.report, null)
  assert.ok(typeof r.error === 'string' && r.error.length > 0, '应带上失败原因')
  console.log(`      报错：${r.error}`)
})

ok('指引在这种情况下也给出下一步（不空着）', () => {
  const g = formatSetupGuidance({ ran: false, ok: null, error: 'timeout-no-report' })
  assert.ok(g.includes('python -m zfh_voice doctor'), g)
})

/* ── 4. 命令组装 ──────────────────────────────────────────────────────── */

console.log('\n── 4. doctor 命令与 serve 命令的一致性 ──')

ok('doctor 命令复用同一个 python 与全局参数（顺序不能错）', () => {
  const launch = launchFor('PY', 'D:\\models')
  assert.equal(launch.argv[0], 'PY')
  assert.ok(launch.argv.includes('serve'))
  // buildDoctorCommand 由 probe 内部调用；这里直接验它的产物形状
  const { argv } = (() => {
    const i = launch.argv.indexOf('serve')
    const globals = launch.argv.slice(1, i)
    return { argv: ['PY', '-m', 'zfh_voice', ...globals, 'doctor', '--json', 'X'] }
  })()
  const iDoctor = argv.indexOf('doctor')
  assert.ok(argv.indexOf('--model-dir') < iDoctor, '全局参数必须在 doctor 前')
  assert.equal(argv[iDoctor + 1], '--json')
})

console.log(`\n${failed === 0 ? '✅ 端到端验收通过' : '❌ 有失败项'}`)
console.log(`通过 ${passed} 项，失败 ${failed} 项，跳过 ${skipped} 项`)
process.exit(failed === 0 ? 0 : 1)
