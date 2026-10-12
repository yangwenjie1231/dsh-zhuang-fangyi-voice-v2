/**
 * DSH 插件半边（`index.js`）单测 —— 纯逻辑部分。
 *
 * 用法：`node tools/test-plugin.mjs`
 *
 * 为什么这些必须有测试：插件管的是**别人的进程和你的显存**。
 * 两个方向的错都很贵，且都发生在用户看不见的地方：
 *   · 该释放时不释放 → 显存被白占（用户以为关了常驻）；
 *   · 不该杀时杀了 → 用户自己在终端起的服务被终止。
 */

import assert from 'node:assert/strict'
import path from 'node:path'

import {
  resolvePython,
  buildServeCommand,
  serveOptions,
  nextServiceAction,
  wavDurationMs,
  defaultConfig,
  normalizeConfig,
  createSupervisor,
  buildDoctorCommand,
  classifyFailure,
  wantsTorchBackend,
  interpreterWarning,
  probeTorch,
  formatSetupGuidance,
  createEnvProbe,
  RAPID_EXIT_MS,
  DEFAULT_IDLE_STOP_SEC,
  DEFAULT_IDLE_ACTION,
  IDLE_ACTIONS,
  DEFAULT_SEED,
  DEFAULT_BRIGHTNESS_DB,
  DOCTOR_TIMEOUT_MS
} from '../index.js'

let passed = 0
let failed = 0
const section = n => console.log(`\n── ${n} ──`)
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

/* ── 1. python 解析（不许有绝对路径硬编码）──────────────────────────────── */

section('1. python 解析')

ok('显式配置优先', () => {
  assert.equal(resolvePython({ python: 'D:/x/py.exe', env: {} }), 'D:/x/py.exe')
})

ok('⭐ 环境变量次之（ZFH_VOICE_PYTHON）', () => {
  assert.equal(resolvePython({ env: { ZFH_VOICE_PYTHON: '/opt/py' } }), '/opt/py')
})

ok('⭐ 插件目录下的 .venv 被认出来（**相对路径**，可移植）', () => {
  // 关键：候选路径全部从 repoDir 派生 —— 换台机器不用改任何配置。
  const seen = []
  const exists = p => { seen.push(p); return p.endsWith(path.join('.venv', 'Scripts', 'python.exe')) }
  const got = resolvePython({ repoDir: 'R', exists, env: {} })
  assert.equal(got, path.join('R', '.venv', 'Scripts', 'python.exe'))
  assert.ok(seen.every(p => p.startsWith('R')), `候选路径必须都在 repoDir 下：${JSON.stringify(seen)}`)
})

ok('POSIX 布局也能找到（bin/python）', () => {
  const exists = p => p.endsWith(path.join('.venv', 'bin', 'python'))
  const got = resolvePython({ repoDir: 'R', exists, env: {} })
  assert.equal(got, path.join('R', '.venv', 'bin', 'python'))
})

ok('都找不到 → 交给 PATH（名字随平台）', () => {
  const got = resolvePython({ repoDir: 'R', exists: () => false, env: {} })
  assert.ok(got === 'python' || got === 'python3', `实际 ${got}`)
})

/* ── 2. 启动命令 ─────────────────────────────────────────────────────────── */

section('2. 启动命令')

ok('缺 python/repoDir → null', () => {
  assert.equal(buildServeCommand({}), null)
  assert.equal(buildServeCommand({ python: 'p' }), null)
  assert.equal(buildServeCommand({ repoDir: 'R' }), null)
})

ok('⭐ 全局参数在 `serve` 之前（本仓库 cli.py 的结构）', () => {
  const c = buildServeCommand({ python: 'p', repoDir: 'R', backend: 'torch', gsvRoot: 'G' })
  const i = c.argv.indexOf('serve')
  assert.ok(c.argv.indexOf('--backend') < i, '--backend 必须在 serve 前')
  assert.ok(c.argv.indexOf('--gsv-root') < i, '--gsv-root 必须在 serve 前')
  assert.ok(c.argv.indexOf('--port') > i, '--port 属于 serve 子命令')
})

ok('`auto` 后端不传参数（交给 Python 侧自己选）', () => {
  for (const b of ['auto', '', null, undefined, 'cuda']) {
    const c = buildServeCommand({ python: 'p', repoDir: 'R', backend: b })
    assert.ok(!c.argv.includes('--backend'), `backend=${JSON.stringify(b)} 不该传下去`)
  }
})

ok('⭐ 自动带 PYTHONPATH=src（src 布局，不装也能跑）', () => {
  const c = buildServeCommand({ python: 'p', repoDir: path.join('a', 'b') })
  assert.equal(c.env.PYTHONPATH, path.join('a', 'b', 'src'))
  assert.equal(c.cwd, path.join('a', 'b'))
})

ok('port 非法值回落 8765（不把 NaN 传给子进程）', () => {
  for (const bad of [0, -1, NaN, 'abc', null]) {
    const c = buildServeCommand({ python: 'p', repoDir: 'R', port: bad })
    assert.equal(c.argv[c.argv.length - 1], '8765')
  }
})

/* ── 3. 决策矩阵 ─────────────────────────────────────────────────────────── */

section('3. 决策矩阵')

const base = { resident: false, ours: true, healthy: true, idleMs: 0, idleLimitMs: 300000, launchable: true, rapidExit: false }
const decide = p => nextServiceAction({ ...base, ...p })

ok('⭐⭐ 不是我们起的 → 永远不动它', () => {
  for (const p of [{ ours: false, healthy: true, idleMs: 1e9 }, { ours: false, healthy: false, resident: true }]) {
    assert.equal(decide(p).why, 'not-ours')
  }
})

ok('⭐⭐ 非常驻 + 空闲超限 → stop', () => {
  const d = decide({ idleMs: 300001, idleLimitMs: 300000 })
  assert.equal(d.action, 'stop')
})

ok('⭐ 空闲上限 0 = 不停', () => {
  assert.equal(decide({ idleMs: 1e9, idleLimitMs: 0 }).action, 'none')
})

ok('⭐⭐ 常驻 → 永不因空闲而停', () => {
  assert.equal(decide({ resident: true, idleMs: 1e9, idleLimitMs: 1 }).why, 'resident-keep')
})

ok('⭐ 常驻 + 不健康 → restart', () => {
  assert.equal(decide({ resident: true, healthy: false }).action, 'restart')
})

ok('⭐⭐ 短命退出 → 不重启（防重启风暴）', () => {
  assert.equal(decide({ resident: true, healthy: false, rapidExit: true }).why, 'down-rapid-exit')
})

/* ── 4. WAV 时长 ─────────────────────────────────────────────────────────── */

section('4. WAV 时长')

function wav (samples, sr = 32000, ch = 1, bits = 16) {
  const dataLen = samples * ch * (bits / 8)
  const b = Buffer.alloc(44 + dataLen)
  b.write('RIFF', 0, 'ascii'); b.writeUInt32LE(36 + dataLen, 4); b.write('WAVE', 8, 'ascii')
  b.write('fmt ', 12, 'ascii'); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20)
  b.writeUInt16LE(ch, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(sr * ch * bits / 8, 28)
  b.writeUInt16LE(ch * bits / 8, 32); b.writeUInt16LE(bits, 34)
  b.write('data', 36, 'ascii'); b.writeUInt32LE(dataLen, 40)
  return b
}

ok('单声道 1 秒 / 立体声 1 秒 / 8-bit 1 秒', () => {
  assert.equal(wavDurationMs(wav(32000)), 1000)
  assert.equal(wavDurationMs(wav(32000, 32000, 2)), 1000)
  assert.equal(wavDurationMs(wav(32000, 32000, 1, 8)), 1000)
})

ok('坏输入一律 0（不抛）', () => {
  assert.equal(wavDurationMs(null), 0)
  assert.equal(wavDurationMs(Buffer.from('nope')), 0)
  assert.equal(wavDurationMs(Buffer.alloc(100)), 0)
})

/* ── 5. 配置归一化 ───────────────────────────────────────────────────────── */

section('5. 配置归一化')

ok('默认：auto 后端 / 不常驻 / 全自动路径', () => {
  const d = defaultConfig()
  assert.equal(d.backend, 'auto')
  assert.equal(d.resident, false)
  assert.equal(d.python, '')
  assert.equal(d.repoDir, '')
  assert.equal(d.idleStopSec, DEFAULT_IDLE_STOP_SEC)
})

ok('非法值被挡（后端白名单 / 端口范围 / 非布尔）', () => {
  const c = normalizeConfig({
    backend: 'cuda', port: 99999, resident: 'yes', idleStopSec: -1, python: 'x'.repeat(999)
  })
  assert.equal(c.backend, 'auto', '非法后端应回落 auto')
  assert.equal(c.port, 8765, '端口越界应回落')
  assert.equal(c.resident, false, '非布尔不该被当成 true')
  assert.equal(c.idleStopSec, DEFAULT_IDLE_STOP_SEC, '负数应回落')
  assert.equal(c.python, '', '超长字符串应被拒')
})

ok('合法值原样保留', () => {
  const c = normalizeConfig({ backend: 'torch', port: 9000, resident: true, idleStopSec: 60 })
  assert.equal(c.backend, 'torch')
  assert.equal(c.port, 9000)
  assert.equal(c.resident, true)
  assert.equal(c.idleStopSec, 60)
})

/* ── 6. 监管器 ───────────────────────────────────────────────────────────── */

section('6. 监管器')

/** "spawn 之后才健康"的桩 —— 必须复刻真实时序（起进程 → 加载 → 才健康）。*/
function gated () {
  const st = { started: false, calls: [], handles: [] }
  return {
    st,
    health: async () => (st.started ? { ok: true, backend: 'torch' } : null),
    spawn: spec => {
      st.calls.push(spec)
      st.started = true
      const h = { pid: 100 + st.handles.length, terminated: false, done: new Promise(() => {}), terminate () { this.terminated = true } }
      st.handles.push(h)
      return h
    }
  }
}

await okAsync('没有启动命令 → ensure false 且不 spawn', async () => {
  const g = gated()
  const sup = createSupervisor({ spawn: g.spawn, health: async () => null })
  sup.setConfig(defaultConfig(), null)
  assert.equal(await sup.ensure(), false)
  assert.equal(g.st.calls.length, 0)
})

await okAsync('服务已在跑（用户自己起的）→ 不 spawn、ours=false', async () => {
  const g = gated()
  const sup = createSupervisor({ spawn: g.spawn, health: async () => ({ ok: true, backend: 'onnx' }) })
  sup.setConfig(defaultConfig(), { argv: ['p'], cwd: 'c', env: {} })
  assert.equal(await sup.ensure(), true)
  assert.equal(g.st.calls.length, 0)
  assert.equal(sup.status().ours, false)
})

await okAsync('⭐ 服务没跑 → 启动并等就绪；stop 能停掉它', async () => {
  const g = gated()
  const sup = createSupervisor({ spawn: g.spawn, health: g.health })
  sup.setConfig(defaultConfig(), { argv: ['p', 'serve'], cwd: 'c', env: {} })
  assert.equal(await sup.ensure(), true)
  assert.equal(g.st.calls.length, 1)
  assert.equal(sup.status().ours, true)
  assert.equal(sup.stop(), true)
  assert.equal(g.st.handles.filter(h => h.terminated).length, 1)
})

await okAsync('⭐⭐ 非常驻：空闲到点自动停（释放显存）', async () => {
  const g = gated()
  let t = 1000
  const sup = createSupervisor({ spawn: g.spawn, health: g.health, now: () => t })
  sup.setConfig({ ...defaultConfig(), resident: false, idleStopSec: 60 }, { argv: ['p'], cwd: 'c', env: {} })
  assert.equal(await sup.ensure(), true)
  t += 30000
  assert.equal((await sup.tick()).action, 'none')
  t += 40000
  assert.equal((await sup.tick()).action, 'stop')
  assert.equal(g.st.handles.filter(h => h.terminated).length, 1)
})

await okAsync('⭐⭐ 常驻：一小时也不停', async () => {
  const g = gated()
  let t = 1000
  const sup = createSupervisor({ spawn: g.spawn, health: g.health, now: () => t })
  sup.setConfig({ ...defaultConfig(), resident: true, idleStopSec: 1 }, { argv: ['p'], cwd: 'c', env: {} })
  await sup.ensure()
  t += 3600000
  assert.equal((await sup.tick()).why, 'resident-keep')
  assert.equal(g.st.handles.filter(h => h.terminated).length, 0)
})

await okAsync('⭐ 常驻 + 服务被杀 → tick 拉起', async () => {
  const g = gated()
  const sup = createSupervisor({ spawn: g.spawn, health: g.health })
  sup.setConfig({ ...defaultConfig(), resident: true }, { argv: ['p'], cwd: 'c', env: {} })
  await sup.ensure()
  g.st.started = false
  assert.equal((await sup.tick()).action, 'restart')
})

await okAsync('⭐⭐ 秒退不重启（只记错误）', async () => {
  const sup = createSupervisor({
    spawn: () => ({ pid: 1, done: Promise.resolve({ exitCode: 1 }), terminate () {} }),
    health: async () => null
  })
  sup.setConfig({ ...defaultConfig(), resident: true, startTimeoutMs: 2000 }, { argv: ['p'], cwd: 'c', env: {} })
  assert.equal(await sup.ensure(), false)
  await new Promise(r => setTimeout(r, 30))
  assert.ok(sup.status().lastError !== null, '必须记下失败原因')
  assert.equal((await sup.tick()).why, 'down-rapid-exit')
})

/* ── 7. 服务面（configure / status）─────────────────────────────────────── */

section('7. 服务面')

ok('插件导出 name / apply / 服务名常量', async () => {
  const mod = await import('../index.js')
  assert.equal(mod.name, 'dsh-zhuang-fangyi-voice')
  assert.equal(typeof mod.apply, 'function')
  assert.equal(mod.SERVICE_NAME, 'zfhVoice')
})

ok('RAPID_EXIT_MS 是 10 秒（与桌宠 exe 启动同一条纪律）', () => {
  assert.equal(RAPID_EXIT_MS, 10000)
})

/**
 * 造一个最小 cordis ctx 桩，跑一遍 `apply()`。
 *
 * 为什么必须有：`apply()` 是插件的**入口**。它一抛错，整个插件就是死的 ——
 * 而用户在 DSH 里只会看到"插件没生效"，没有任何线索。
 * 而且入口里有几处只在真实 ctx 上才会走到的路径（`ctx.provide` / `ctx.effect`
 * / 定时器），纯函数测试完全覆盖不到。
 */
function makeCtx (opts = {}) {
  const provided = new Map()
  const effects = []
  const logs = []
  return {
    provided,
    effects,
    logs,
    logger: { info: m => logs.push(String(m)), warn: m => logs.push('WARN ' + m), debug: () => {} },
    get (n) {
      if (n === 'subprocess' && opts.subprocess !== undefined) return opts.subprocess
      return undefined
    },
    provide (n, v) { provided.set(n, v); return () => provided.delete(n) },
    effect (fn) {
      const d = fn()
      effects.push(d)
      return d
    }
  }
}

await okAsync('⭐⭐ apply() 能跑完并挂上 zfhVoice 服务（入口不能抛）', async () => {
  const mod = await import('../index.js')
  const ctx = makeCtx()
  const svc = mod.apply(ctx, {})
  assert.ok(ctx.provided.has('zfhVoice'), '必须挂上 zfhVoice 服务（桌宠靠它发现本插件）')
  assert.equal(ctx.provided.get('zfhVoice'), svc, '挂的应是同一个对象')
  // 服务面齐全（桌宠会调这几个）
  for (const k of ['synthesize', 'health', 'status', 'start', 'stop', 'configure']) {
    assert.equal(typeof svc[k], 'function', `服务缺方法 ${k}`)
  }
  // 日志里能看到挂载信息（用户排查的第一条线索）
  assert.ok(ctx.logs.some(l => l.includes('zfhVoice')), `日志里应有挂载记录：${JSON.stringify(ctx.logs)}`)
  // 清理：跑掉 effect（停掉定时器）
  for (const d of ctx.effects) if (typeof d === 'function') assert.doesNotThrow(() => d())
})

await okAsync('⭐ 没有 subprocess 且端口上没有服务时，synthesize 返回 null（不抛）', async () => {
  const mod = await import('../index.js')
  const ctx = makeCtx()          // get('subprocess') → undefined
  // ⚠️ 必须用一个**确定没服务**的端口。
  //
  // 早先这条用的是默认 8765，于是当环境里恰好有服务（例如 DSH 自己已经把
  // 插件启起来、或用户手动起过）时，`probe()` 会成功、synthesize 真的合成出
  // 音频，断言 `r === null` 就失败 —— 而那是**正确行为**（复用已有服务是设计
  // 意图，见 index.js 顶部「只停自己起的」那条纪律）。
  //
  // 所以这条测的是「**连不上任何服务**时不许抛」，用一个不会被占用的端口。
  const port = 59999 + (process.pid % 1000)   // 避开常用端口，且各进程不同
  const svc = mod.apply(ctx, { port, startTimeoutMs: 3000 })
  assert.ok(svc !== null)
  // 起不了进程 + 端口无人 → 返回 null（绝不冒泡）
  const r = await svc.synthesize('你好')
  assert.equal(r, null, '连不上服务时应返回 null 而不是抛')
  for (const d of ctx.effects) if (typeof d === 'function') assert.doesNotThrow(() => d())
})

await okAsync('⭐⭐ configure() 能改常驻策略（桌宠的设置页驱动它）', async () => {
  const mod = await import('../index.js')
  const ctx = makeCtx()
  const svc = mod.apply(ctx, { resident: false, idleStopSec: 300 })
  assert.equal(svc.status().resident, false)
  svc.configure({ resident: true, idleStopSec: 60 })
  const st = svc.status()
  assert.equal(st.resident, true, '常驻开关应生效（桌宠设置页推过来的）')
  assert.equal(st.idleStopSec, 60, '空闲释放时间应生效')
  // 非法值不该被灌进去
  svc.configure({ idleStopSec: -1, resident: 'yes' })
  const st2 = svc.status()
  assert.equal(st2.resident, true, '非布尔不该改掉现状')
  assert.equal(st2.idleStopSec, 60, '负数不该改掉现状')
  for (const d of ctx.effects) if (typeof d === 'function') assert.doesNotThrow(() => d())
})

await okAsync('⭐ status() 里带启动命令（用户能看到到底会执行什么）', async () => {
  const mod = await import('../index.js')
  const ctx = makeCtx()
  const svc = mod.apply(ctx, {})
  const st = svc.status()
  assert.ok(st.launch !== null, '默认应能组装出启动命令（插件知道自己在哪）')
  assert.ok(Array.isArray(st.launch.argv) && st.launch.argv.length > 0)
  assert.ok(st.launch.argv.includes('serve'), `命令里应有 serve 子命令：${st.launch.argv.join(' ')}`)
  assert.ok(st.repoDir !== undefined || st.python !== undefined, '应报出解析结果')
  for (const d of ctx.effects) if (typeof d === 'function') assert.doesNotThrow(() => d())
})

/* ── 8. 常驻三态 + 两级释放 ──────────────────────────────────────────────── */

section('8. 常驻三态与两级释放')

await okAsync('⭐ unload 档位：空闲时卸模型但**保留进程**，且仍算 ours', async () => {
  const g = gated()
  let t = 1000
  let unloaded = 0
  const sup = createSupervisor({
    spawn: g.spawn,
    health: g.health,
    unload: async () => { unloaded += 1; return true },
    now: () => t
  })
  sup.setConfig({ ...defaultConfig(), resident: false, idleStopSec: 60, idleAction: 'unload' },
    { argv: ['p'], cwd: 'c', env: {} })
  assert.equal(await sup.ensure(), true)
  t += 70000
  const d = await sup.tick()
  assert.equal(d.action, 'unload', `空闲应卸模型而不是杀进程：${JSON.stringify(d)}`)
  assert.equal(unloaded, 1, '应调用过 /unload')
  assert.equal(g.st.handles.filter(h => h.terminated).length, 0, '不该杀进程')
  // ★ 关键：进程还在 → ours 必须继续成立，否则下次 tick 会认为"不是我们起的"而撒手
  assert.equal(sup.status().ours, true, 'unload 后仍应认这个进程')
  assert.equal(sup.status().processAlive, true)
  assert.equal(sup.status().unloadCount, 1)
})

await okAsync('⭐⭐ stop 档位：空闲时杀进程（保持既有行为）', async () => {
  const g = gated()
  let t = 1000
  const sup = createSupervisor({ spawn: g.spawn, health: g.health, unload: async () => true, now: () => t })
  sup.setConfig({ ...defaultConfig(), resident: false, idleStopSec: 60, idleAction: 'stop' },
    { argv: ['p'], cwd: 'c', env: {} })
  await sup.ensure()
  t += 70000
  assert.equal((await sup.tick()).action, 'stop')
  assert.equal(g.st.handles.filter(h => h.terminated).length, 1)
})

await okAsync('⭐⭐ modelLoaded 三态：未知 / 已加载 / 已释放 —— 不误报', async () => {
  // ① 服务没在跑 → null（未知，而不是 false）
  const sup0 = createSupervisor({ spawn: gated().spawn, health: async () => null })
  sup0.setConfig(defaultConfig(), { argv: ['p'], cwd: 'c', env: {} })
  assert.equal(sup0.status().modelLoaded, null, '服务没跑时应报"未知"而不是"未加载"')

  // ② 进程在、模型已加载（用 gated(): spawn 之前不健康 → 才有真实进程）
  const g2 = gated()
  let healthBody = { ok: true, backend: 'torch', resident: true, idleSeconds: 1, loads: 1, unloads: 0 }
  const sup2 = createSupervisor({
    spawn: g2.spawn,
    health: async () => (g2.st.started ? healthBody : null)
  })
  sup2.setConfig({ ...defaultConfig(), resident: true }, { argv: ['p'], cwd: 'c', env: {} })
  assert.equal(await sup2.ensure(), true)
  assert.equal(sup2.status().modelLoaded, true, '模型已加载应报 true')
  assert.equal(sup2.status().processAlive, true)

  // ③ 进程在、但模型已释放 → false，且 processAlive 仍为 true
  healthBody = { ok: true, backend: 'torch', resident: false, idleSeconds: 90, loads: 1, unloads: 1 }
  assert.equal(await sup2.ensure(), true)
  const st = sup2.status()
  assert.equal(st.modelLoaded, false, '已释放应报 false')
  assert.equal(st.processAlive, true, '进程仍在')
  assert.equal(st.resident, true, '配置意图不受影响')
  assert.equal(st.unloads, 1)
})

await okAsync('⭐ 旧的 Python 侧（/health 没有 resident）→ 报 null 而不是猜', async () => {
  const g = gated()
  const sup = createSupervisor({
    spawn: g.spawn,
    health: async () => ({ ok: true, backend: 'onnx' })   // 老版本，无 resident 字段
  })
  sup.setConfig(defaultConfig(), { argv: ['p'], cwd: 'c', env: {} })
  await sup.ensure()
  assert.equal(sup.status().modelLoaded, null, '拿不到就报未知，绝不能猜成 false')
})

ok('idleAction 默认 stop（保持既有行为，零意外）', () => {
  assert.equal(DEFAULT_IDLE_ACTION, 'stop')
  assert.deepEqual(IDLE_ACTIONS, ['stop', 'unload'])
  assert.equal(defaultConfig().idleAction, 'stop')
})

ok('idleAction 非法值被过滤', () => {
  assert.equal(normalizeConfig({ idleAction: 'unload' }).idleAction, 'unload')
  assert.equal(normalizeConfig({ idleAction: 'nuke' }).idleAction, 'stop')
  assert.equal(normalizeConfig({ idleAction: 42 }).idleAction, 'stop')
})

ok('nextServiceAction：resident=true 时 idleAction 不生效', () => {
  const d = nextServiceAction({
    resident: true, idleAction: 'unload', ours: true, healthy: true,
    idleMs: 999999, idleLimitMs: 1000, launchable: true, rapidExit: false
  })
  assert.equal(d.why, 'resident-keep')
})

/* ── 8b. 短句稳定性：默认种子 + 亮度补偿 ──────────────────────────────── */

section('8b. 短句稳定性（默认种子 / 亮度补偿）')

ok('⭐ 默认 seed=42（短句不再随机截断）', () => {
  assert.equal(DEFAULT_SEED, 42)
  assert.equal(defaultConfig().seed, 42)
})

ok('⭐ seed 会传进 serve 命令，且在 `serve` 之前', () => {
  const c = buildServeCommand({ python: 'p', repoDir: 'R', seed: 42 })
  const i = c.argv.indexOf('serve')
  assert.ok(i > 0, '要有 serve 子命令')
  assert.ok(c.argv.indexOf('--seed') > 0, '--seed 必须存在')
  assert.ok(c.argv.indexOf('--seed') < i, '--seed 是全局参数，必须在 serve 前')
  assert.equal(c.argv[c.argv.indexOf('--seed') + 1], '42')
})

ok('⭐ seed=null → --random-seed（显式要求随机，不是漏传）', () => {
  const c = buildServeCommand({ python: 'p', repoDir: 'R', seed: null })
  assert.ok(c.argv.includes('--random-seed'), 'null 要变成显式开关')
  assert.ok(!c.argv.includes('--seed'), 'null 不该再传 --seed')
})

ok('seed 非法值被过滤', () => {
  assert.equal(normalizeConfig({ seed: null }).seed, null, 'null 是合法值')
  assert.equal(normalizeConfig({ seed: 7 }).seed, 7)
  assert.equal(normalizeConfig({ seed: -5 }).seed, 42, '负数回落默认')
  assert.equal(normalizeConfig({ seed: 'abc' }).seed, 42, '非数字回落默认')
  assert.equal(normalizeConfig({ seed: 1e12 }).seed, 42, '超 int32 回落默认')
})

ok('⭐ 亮度补偿默认 auto（按模型版本自动选，避免换模型过冲）', () => {
  // 为什么是 auto 而不是固定数字：v2 需要 +3dB 才对齐原声，
  // 而 v2Pro 的频谱本来就接近原声，加 3dB 会**过冲**（偏差 0.88 → 1.92）。
  assert.equal(DEFAULT_BRIGHTNESS_DB, 'auto')
  assert.equal(defaultConfig().brightnessDb, 'auto')
  // auto 时**不传参数**，交给 Python 侧按加载到的版本自己决定
  const c = buildServeCommand({ python: 'p', repoDir: 'R', brightnessDb: 'auto' })
  assert.ok(!c.argv.includes('--brightness-db'), 'auto 不该传参数')
})

ok('⭐ 亮度补偿显式数字才传参，且在 `serve` 之前', () => {
  const c = buildServeCommand({ python: 'p', repoDir: 'R', brightnessDb: 3 })
  const i = c.argv.indexOf('serve')
  assert.ok(c.argv.indexOf('--brightness-db') > 0)
  assert.ok(c.argv.indexOf('--brightness-db') < i, '必须是全局参数')
  assert.equal(c.argv[c.argv.indexOf('--brightness-db') + 1], '3')
})

ok('⭐ brightnessDb=0 也传参（与 auto 语义不同：显式关闭）', () => {
  const c = buildServeCommand({ python: 'p', repoDir: 'R', brightnessDb: 0 })
  assert.ok(c.argv.includes('--brightness-db'),
    '0 是"显式关闭"，必须发出去，不能与 auto 混为一谈')
  assert.equal(c.argv[c.argv.indexOf('--brightness-db') + 1], '0')
})

ok('brightnessDb 只接受 auto 或 0~12（防过亮）', () => {
  assert.equal(normalizeConfig({ brightnessDb: 'auto' }).brightnessDb, 'auto')
  assert.equal(normalizeConfig({ brightnessDb: 'AUTO' }).brightnessDb, 'auto')
  assert.equal(normalizeConfig({ brightnessDb: 3 }).brightnessDb, 3)
  assert.equal(normalizeConfig({ brightnessDb: 0 }).brightnessDb, 0)
  assert.equal(normalizeConfig({ brightnessDb: 12 }).brightnessDb, 12)
  assert.equal(normalizeConfig({ brightnessDb: -1 }).brightnessDb, 'auto', '负数回落默认')
  assert.equal(normalizeConfig({ brightnessDb: 99 }).brightnessDb, 'auto', '过大回落默认')
  assert.equal(normalizeConfig({ brightnessDb: 'x' }).brightnessDb, 'auto')
})

ok('doctor 命令沿用同一套 seed/brightness 参数（体检的就是要跑的那套）', () => {
  const launch = buildServeCommand({ python: 'p', repoDir: 'R', seed: 42, brightnessDb: 3 })
  const d = buildDoctorCommand({ launch, jsonPath: 'J' })
  assert.ok(d, '要能拼出 doctor 命令')
  assert.ok(d.argv.includes('--brightness-db'), 'doctor 也要带补偿参数')
  assert.ok(d.argv.includes('--seed'))
  assert.ok(!d.argv.includes('serve'), 'doctor 不该带 serve')
})

ok('⭐⭐ serveOptions：配置项**真的**会到命令行（曾漏传 seed/brightnessDb）', () => {
  // 这一条守的是"调用点漏传"——只测 buildServeCommand 是抓不到的：
  // 它本身工作正常，是 apply()/configure() 忘了把新配置塞进去。
  const cfg = normalizeConfig({ seed: 7, brightnessDb: 3, backend: 'torch' })
  const c = buildServeCommand(serveOptions(cfg, 'p', 'R'))
  assert.equal(c.argv[c.argv.indexOf('--seed') + 1], '7', 'seed 必须从配置流到 argv')
  assert.equal(c.argv[c.argv.indexOf('--brightness-db') + 1], '3', 'brightnessDb 必须流到 argv')
  assert.ok(c.argv.indexOf('--seed') < c.argv.indexOf('serve'))
})

ok('serveOptions：默认配置 → 带 --seed 42、不带 --brightness-db', () => {
  const c = buildServeCommand(serveOptions(defaultConfig(), 'p', 'R'))
  assert.equal(c.argv[c.argv.indexOf('--seed') + 1], '42', '默认种子必须生效')
  assert.ok(!c.argv.includes('--brightness-db'), '默认关闭补偿')
})

ok('serveOptions：seed=null → --random-seed', () => {
  const cfg = normalizeConfig({ seed: null })
  const c = buildServeCommand(serveOptions(cfg, 'p', 'R'))
  assert.ok(c.argv.includes('--random-seed'))
  assert.ok(!c.argv.includes('--seed'))
})

/* ── 9. 环境探测与失败分类（新用户路径）─────────────────────────────────── */

section('9. 环境探测与失败分类')

/** 造一份 doctor --json 的输出 */
function doctorReport (over = {}) {
  return {
    deps_missing: [],
    installed: { onnx_ok: true, torch_ok: false, aux_ok: true, ref_ok: true },
    gpu: { present: true, name: 'RTX 5060', vram_mb: 8151 },
    ...over
  }
}

ok('⭐ classifyFailure：缺依赖', () => {
  const c = classifyFailure(doctorReport({ deps_missing: ['numpy', 'onnxruntime'] }))
  assert.equal(c.kind, 'missing-deps')
  assert.ok(c.summary.includes('numpy'), `摘要应点名缺什么：${c.summary}`)
  assert.ok(c.commands[0].includes('pip install'), '应给出可直接执行命令')
})

ok('⭐ classifyFailure：依赖齐了但没模型', () => {
  const c = classifyFailure(doctorReport({
    installed: { onnx_ok: false, torch_ok: false, aux_ok: true, ref_ok: true }
  }))
  assert.equal(c.kind, 'missing-models')
  assert.ok(c.commands.some(x => x.includes('download_models.py')))
})

ok('⭐⭐ classifyFailure：torch 路径就绪时，aux 缺失**不该**算问题（实测误报）', () => {
  // 场景就是本机真实状态：模型目录里只有 torch_weights/ + ref/，
  // 没有 G2PWModel / tokenizer（那套 torch 后端从 gsvRoot 读）。
  // 原先这里会报 missing-aux 并让用户去 `download_models.py` 补一个
  // 根本不需要的辅助包 —— 而合成其实是成功的。
  const report = doctorReport({
    installed: { onnx_ok: false, torch_ok: true, aux_ok: false, ref_ok: true }
  })
  // 显式 torch
  const a = classifyFailure(report, undefined, { backend: 'torch' })
  assert.equal(a.kind, 'aux-not-needed-torch', `torch 后端不该报 missing-aux：${a.kind}`)
  assert.deepEqual(a.commands, [], '不该让用户去做无用功')
  // auto 且 torch 权重在 → 会选 torch，同样不该报
  const b = classifyFailure(report, undefined, { backend: 'auto' })
  assert.equal(b.kind, 'aux-not-needed-torch', 'auto+有torch权重 也应判为无需 aux')
  // ⭐ 反向：显式 onnx 时 aux 就是硬需求（缺了必须报）
  const c = classifyFailure(report, undefined, { backend: 'onnx' })
  assert.equal(c.kind, 'missing-aux', `onnx 后端缺 aux 必须报，实际 ${c.kind}`)
  assert.ok(c.commands.some(x => x.includes('download_models.py')), '应给出补齐命令')
  // ⭐ 反向：auto 但没有 torch 权重 → 会走 onnx，aux 也必需
  const d = classifyFailure(doctorReport({
    installed: { onnx_ok: false, torch_ok: false, aux_ok: false, ref_ok: true }
  }), undefined, { backend: 'auto' })
  assert.equal(d.kind, 'missing-models', `没模型时先报 missing-models，实际 ${d.kind}`)
  // ⭐ 参考音频缺失是两条路都致命的 —— 即使 torch 就绪也不能算"没问题"
  const e = classifyFailure(doctorReport({
    installed: { onnx_ok: false, torch_ok: true, aux_ok: false, ref_ok: false }
  }), undefined, { backend: 'torch' })
  assert.equal(e.kind, 'missing-aux', `参考音频缺了仍是硬伤，实际 ${e.kind}`)
})

await okAsync('⭐ probeTorch：拿退出码判有没有 torch（超时不误判）', async () => {
  const mk = code => () => ({ pid: 1, done: Promise.resolve({ exitCode: code }), terminate () {} })
  assert.equal(await probeTorch({ spawn: mk(0), python: 'p' }), true, 'exit 0 → 有 torch')
  assert.equal(await probeTorch({ spawn: mk(1), python: 'p' }), false, 'exit 1 → 没 torch')
  assert.equal(await probeTorch({ spawn: () => { throw new Error('nope') }, python: 'p' }), null)
  assert.equal(await probeTorch({ spawn: () => null, python: 'p' }), null)
  assert.equal(await probeTorch({ spawn: () => ({ pid: 1, done: 42 }), python: 'p' }), null)
  assert.equal(await probeTorch({ spawn: null, python: 'p' }), null)
  let terminated = false
  const hang = await probeTorch({
    python: 'p', timeoutMs: 50,
    spawn: () => ({ pid: 1, done: new Promise(() => {}), terminate () { terminated = true } })
  })
  assert.equal(hang, null, '挂住的进程不该让我们卡死')
  assert.equal(terminated, true, '超时后应终止它')
})

await okAsync('⭐⭐ probeTorch：探测**不能** import torch（一条命令换 400MB）', async () => {
  // 这条断言防的是我自己踩过的坑：第一版用 `-c "import torch"`，
  // 峰值 406MB / 1.9s，而每次 apply() 都 spawn 一次 → 把机器提交量打满、
  // Node 报 Committing semi space failed、Electron 崩溃。
  // find_spec 只定位包：10MB / 0.1s。
  let argv = null
  await probeTorch({
    python: 'p',
    spawn: spec => { argv = spec.argv; return { pid: 1, done: Promise.resolve({ exitCode: 0 }), terminate () {} } }
  })
  const cmd = argv.join(' ')
  assert.ok(cmd.includes('find_spec'), `必须用 find_spec 探测：${cmd}`)
  assert.ok(!/\bimport torch\b/.test(cmd), `绝不能 import torch（会加载 400MB）：${cmd}`)
})
ok('⭐⭐ interpreterWarning：解释器没 torch 时要给出可照做的修法', () => {
  // 新用户最容易踩的坑：PATH 上的 python 没装 torch（系统 Python / 没激活的 conda），
  // 而这是**静默**失败 —— 用户看到的只是一句"服务起不来"。
  const w = interpreterWarning({
    python: 'D:\\Programs\\Python\\Python313\\python.exe',
    backend: 'torch', hasTorch: false, gsvRoot: 'D:\\A\\voiceclone\\GPT-SoVITS'
  })
  assert.ok(w !== null, 'torch 后端 + 无 torch 必须警告')
  assert.ok(w.includes('没有 torch'), '要点明原因：' + w)
  assert.ok(w.includes('python:'), '要给出配置键名（照抄就能改）：' + w)
  assert.ok(w.includes('ZFH_VOICE_PYTHON'), '要给出环境变量这条退路：' + w)
  // 反向：探测不出来（null）时不乱报 —— 宁可不说，也不能谎报
  assert.equal(interpreterWarning({ python: 'p', backend: 'torch', hasTorch: null }), null)
  // 反向：有 torch 不报
  assert.equal(interpreterWarning({ python: 'p', backend: 'torch', hasTorch: true }), null)
  // 反向：onnx 后端不依赖 torch，不报
  assert.equal(interpreterWarning({ python: 'p', backend: 'onnx', hasTorch: false }), null)
  // wantsTorchBackend：auto 也算"可能要 torch"
  assert.equal(wantsTorchBackend('auto'), true)
  assert.equal(wantsTorchBackend('torch'), true)
  assert.equal(wantsTorchBackend('onnx'), false)
})

ok('⭐ formatSetupGuidance：良性结论不打"环境还没配好"', () => {
  const probe = {
    ran: true, ok: false,
    report: doctorReport({
      installed: { onnx_ok: false, torch_ok: true, aux_ok: false, ref_ok: true }
    })
  }
  const text = formatSetupGuidance(probe, 'docs/安装提示词.md', { backend: 'torch' })
  assert.ok(!text.includes('环境还没配好'),
    `torch 就绪时不该说"环境还没配好"：${text}`)
  assert.ok(text.includes('torch'), `应说明 torch 路径就绪：${text}`)
  // 反向：真的缺东西时要照旧警告
  const bad = formatSetupGuidance({
    ran: true, ok: false,
    report: doctorReport({ installed: { onnx_ok: false, torch_ok: false, aux_ok: false, ref_ok: true } })
  }, 'docs/安装提示词.md', { backend: 'auto' })
  assert.ok(bad.includes('环境还没配好'), `真缺东西必须警告：${bad}`)
})

ok('⭐ classifyFailure：模型在但辅助/参考音频缺', () => {
  const c = classifyFailure(doctorReport({
    installed: { onnx_ok: true, torch_ok: false, aux_ok: false, ref_ok: false }
  }))
  assert.equal(c.kind, 'missing-aux')
})

ok('⭐ classifyFailure：全都就绪 → unknown（不谎报有问题）', () => {
  const c = classifyFailure(doctorReport())
  assert.equal(c.kind, 'unknown')
})

ok('⭐ classifyFailure：没有报告时透传兜底，不假装知道', () => {
  const c = classifyFailure(null, '服务起不来')
  assert.equal(c.kind, 'unknown')
  assert.equal(c.summary, '服务起不来')
  assert.deepEqual(c.commands, [])
})

ok('buildDoctorCommand：**不准重复** -m zfh_voice（曾因此拼出废命令）', () => {
  const launch = {
    argv: ['py', '-m', 'zfh_voice', '--backend', 'torch', '--model-dir', 'D:\\m',
      'serve', '--host', '127.0.0.1', '--port', '8765'],
    cwd: 'C:\\repo',
    env: { PYTHONPATH: 'x' }
  }
  const c = buildDoctorCommand({ launch, jsonPath: 'C:\\t\\d.json' })
  // 曾经这里 slice(1, at) 把 '-m','zfh_voice' 也当成全局参数带过去，
  // 拼出 `python -m zfh_voice -m zfh_voice ... doctor` —— Python 直接报错，
  // JSON 永不生成，表现为"体检超时"。整个命令必须逐字相等：
  assert.deepEqual(c.argv, [
    'py', '-m', 'zfh_voice',
    '--backend', 'torch', '--model-dir', 'D:\\m',
    'doctor', '--json', 'C:\\t\\d.json'
  ])
  assert.equal(c.argv.filter(x => x === 'zfh_voice').length, 1, 'zfh_voice 只能出现一次')
  assert.equal(c.argv.filter(x => x === '-m').length, 1, '-m 只能出现一次')
  assert.ok(!c.argv.includes('serve'), '不该把 serve 带进去')
})

ok('buildDoctorCommand：serve 前没有全局参数时也不出错', () => {
  const c = buildDoctorCommand({
    launch: { argv: ['py', '-m', 'zfh_voice', 'serve', '--port', '1'], cwd: 'c', env: {} },
    jsonPath: 'X'
  })
  assert.deepEqual(c.argv, ['py', '-m', 'zfh_voice', 'doctor', '--json', 'X'])
})

ok('buildDoctorCommand：argv 形状异常时不硬套（宁可返回 null）', () => {
  // 没有 `-m zfh_voice` 前缀（不是我们认知的形状）→ 只复用 python，不猜全局参数
  const c = buildDoctorCommand({
    launch: { argv: ['py', 'serve', '--port', '1'], cwd: 'c', env: {} },
    jsonPath: 'X'
  })
  assert.deepEqual(c.argv, ['py', '-m', 'zfh_voice', 'doctor', '--json', 'X'])
})

ok('buildDoctorCommand：全局参数必须排在子命令前（CLI 的硬要求）', () => {
  const launch = {
    argv: ['py', '-m', 'zfh_voice', '--backend', 'torch', '--model-dir', 'D:\\m',
      'serve', '--host', '127.0.0.1', '--port', '8765'],
    cwd: 'C:\\repo',
    env: { PYTHONPATH: 'x' }
  }
  const c = buildDoctorCommand({ launch, jsonPath: 'C:\\t\\d.json' })
  assert.equal(c.argv[0], 'py')
  assert.equal(c.argv[1], '-m')
  assert.equal(c.argv[2], 'zfh_voice')
  const iDoctor = c.argv.indexOf('doctor')
  const iBackend = c.argv.indexOf('--backend')
  assert.ok(iBackend > 0 && iBackend < iDoctor, '全局参数要在 doctor 之前')
  assert.ok(!c.argv.includes('serve'), '不该把 serve 带进去')
  assert.equal(c.argv[iDoctor + 1], '--json')
  assert.equal(c.argv[iDoctor + 2], 'C:\\t\\d.json')
})

ok('buildDoctorCommand：没有启动信息时返回 null，不抛', () => {
  assert.equal(buildDoctorCommand({ launch: null, jsonPath: 'x' }), null)
  assert.equal(buildDoctorCommand({ launch: { argv: [] }, jsonPath: 'x' }), null)
  assert.equal(buildDoctorCommand({ launch: { argv: ['py'] }, jsonPath: '' }), null)
})

ok('formatSetupGuidance：把根因和命令讲清楚', () => {
  const g = formatSetupGuidance({
    ran: true,
    ok: true,
    report: doctorReport({ deps_missing: ['numpy'] })
  })
  assert.ok(g.includes('缺 Python 依赖'), g)
  assert.ok(g.includes('pip install -r requirements.txt'), g)
  assert.ok(g.includes('docs/安装提示词.md'), '应指向文档')
})

ok('formatSetupGuidance：体检没拿到结果时也给出下一步', () => {
  const g = formatSetupGuidance({ ran: false, ok: null, error: 'timeout-no-report' })
  assert.ok(g.includes('doctor'), g)
  assert.ok(g.includes('timeout-no-report'), '应带上失败原因便于排查')
})

await okAsync('createEnvProbe：读到 JSON 就成功', async () => {
  const files = new Map()
  const probe = createEnvProbe({
    spawn: () => ({ pid: 1, terminate () {} }),
    readFile: p => { if (!files.has(p)) throw new Error('ENOENT'); return files.get(p) },
    removeFile: p => files.delete(p),
    tmpFile: () => 'T',
    sleep: async () => { files.set('T', JSON.stringify(doctorReport())) },
    timeoutMs: 2000,
    now: () => Date.now()
  })
  const r = await probe({ argv: ['py', 'serve'], cwd: 'c', env: {} })
  assert.equal(r.ran, true)
  assert.equal(r.ok, true)
  assert.deepEqual(r.report.installed.onnx_ok, true)
  assert.equal(files.size, 0, '读完应清理临时文件')
})

await okAsync('⭐ createEnvProbe：体检超时/坏 JSON → ran:false，不抛', async () => {
  const probe = createEnvProbe({
    spawn: () => ({ pid: 1, terminate () {} }),
    readFile: () => { throw new Error('ENOENT') },
    removeFile: () => {},
    tmpFile: () => 'T',
    sleep: async () => {},
    timeoutMs: 1,          // 立刻超时
    now: (() => { let t = 0; return () => (t += 10) })()
  })
  const r = await probe({ argv: ['py', 'serve'], cwd: 'c', env: {} })
  assert.equal(r.ran, false)
  assert.ok(r.error.includes('timeout'), `应报超时：${r.error}`)
})

await okAsync('createEnvProbe：没有 IO 接缝 → ran:false（对宿主零假设）', async () => {
  const probe = createEnvProbe({})
  const r = await probe({ argv: ['py', 'serve'], cwd: 'c', env: {} })
  assert.equal(r.ran, false)
  assert.equal(r.report, null)
})

await okAsync('⭐ 服务面：diagnose() 在无 subprocess 时不抛', async () => {
  const mod = await import('../index.js')
  const ctx = makeCtx()             // get('subprocess') → undefined
  const svc = mod.apply(ctx, {})
  const d = await svc.diagnose()
  assert.ok(d !== null && typeof d === 'object')
  assert.equal(d.ran, false, '起不了子进程时应如实说没跑成')
  for (const f of ctx.effects) if (typeof f === 'function') assert.doesNotThrow(() => f())
})

await okAsync('⭐⭐ 服务面：setResident / setIdleAction 运行时生效', async () => {
  const mod = await import('../index.js')
  const ctx = makeCtx()
  const svc = mod.apply(ctx, { resident: false, idleAction: 'stop' })
  assert.equal(svc.status().resident, false)
  assert.equal(svc.status().idleAction, 'stop')
  svc.setResident(true)
  assert.equal(svc.status().resident, true, '运行时常驻开关应生效')
  svc.setIdleAction('unload')
  assert.equal(svc.status().idleAction, 'unload')
  svc.setIdleAction('bogus')
  assert.equal(svc.status().idleAction, 'unload', '非法档位不该改掉现状')
  svc.setResident(false)
  assert.equal(svc.status().resident, false)
  for (const f of ctx.effects) if (typeof f === 'function') assert.doesNotThrow(() => f())
})

await okAsync('⭐ status() 报出常驻三态（UI 靠它显示"模型在不在显存里"）', async () => {
  const mod = await import('../index.js')
  const ctx = makeCtx()
  const svc = mod.apply(ctx, { resident: true })
  const st = svc.status()
  assert.equal(typeof st.resident, 'boolean', 'resident = 策略意图')
  assert.equal(typeof st.processAlive, 'boolean', 'processAlive = 进程')
  assert.ok('modelLoaded' in st, 'modelLoaded = 模型真实状态')
  assert.equal(st.modelLoaded, null, '服务没跑时应为 null（未知）')
  assert.equal(typeof st.idleAction, 'string')
  for (const f of ctx.effects) if (typeof f === 'function') assert.doesNotThrow(() => f())
})

ok('DOCTOR_TIMEOUT_MS 有上限保护（不能无限等）', () => {
  assert.ok(Number.isFinite(DOCTOR_TIMEOUT_MS) && DOCTOR_TIMEOUT_MS >= 5000)
})

console.log(`\n${failed === 0 ? '✅ 全部通过' : '❌ 有失败项'}`)
console.log(`通过 ${passed} 项，失败 ${failed} 项`)
process.exit(failed === 0 ? 0 : 1)
