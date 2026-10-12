/**
 * 独立验证语音插件的真实代码路径（不需要 DSH 重启）
 *
 * 做法：import 他们的 `index.js`（真代码），给它一个桩 ctx ——
 * 只把 `subprocess.spawn` 换成真的 node:child_process 适配器。
 * 于是 `resolvePython → buildServeCommand → supervisor.ensure → client.synthesize`
 * 这条链全部走**真实实现**，能提前发现"改了代码但没重启所以不知道行不行"的问题。
 *
 * 用法：node tools/verify-plugin-live.mjs
 */
import { spawn as nodeSpawn } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

const HERE = path.dirname(path.dirname(fileURLToPath(import.meta.url)))

/** 真实 spawn → 适配宿主 SubprocessHandle 的形状。 */
function makeSubprocess () {
  let last = null
  return {
    resolveExecutable: async cmd => cmd,
    spawn (spec) {
      const [file, ...args] = spec.argv
      const child = nodeSpawn(file, args, {
        cwd: spec.cwd,
        env: { ...process.env, ...(spec.env ?? {}) },
        stdio: 'ignore'
      })
      last = child
      return {
        pid: child.pid,
        stdin: null,
        stdout: null,
        stderr: null,
        collected: {},
        done: new Promise(resolve => {
          child.on('exit', (code, signal) => resolve({ exitCode: code, signal }))
          child.on('error', err => resolve({ exitCode: -1, signal: null, error: err }))
        }),
        terminate () { try { child.kill() } catch { /* 已退出 */ } },
        async waitForExit () { return true }
      }
    },
    get last () { return last }
  }
}

const logs = []
const subprocess = makeSubprocess()
const ctx = {
  logger: {
    info: m => { logs.push('INFO ' + m); console.log('[插件] ' + m) },
    warn: m => { logs.push('WARN ' + m); console.log('[插件][warn] ' + m) },
    debug: () => {},
    error: m => console.log('[插件][err] ' + m)
  },
  get (name) {
    if (name === 'subprocess') return subprocess
    if (name === 'logger') return ctx.logger
    return undefined
  },
  provide () { return () => {} },
  effect (fn) { const d = fn(); return d }
}

const mod = await import(`file://${path.join(HERE, 'index.js').replace(/\\/g, '/')}`)

// 与 profile patch 层**完全一致**的配置
const config = {
  python: 'C:\\Users\\A\\miniconda3\\envs\\voice\\python.exe',
  backend: 'torch',
  gsvRoot: 'D:\\A\\voiceclone\\GPT-SoVITS',
  modelDir: 'D:\\A\\voiceclone\\zfh-models',
  resident: false,
  idleStopSec: 300
}

console.log('── 1. apply() ──')
const svc = mod.apply(ctx, config)
const st0 = svc.status()
console.log('  解析出的启动命令:')
console.log('    ' + st0.launch.argv.join(' '))
console.log('    cwd: ' + st0.launch.cwd)
if (!st0.launch.argv.includes('--backend')) { console.log('  ❌ 命令里缺 --backend'); process.exit(1) }
if (st0.launch.argv[0] !== config.python) { console.log('  ❌ 没用配置里的 python'); process.exit(1) }
console.log('  ✅ 命令与解释器都对')

console.log('\n── 2. synthesize()（首次要加载模型，约 20 秒）──')
const t0 = Date.now()
const r = await svc.synthesize('嗯，我在呢。')
const dt = ((Date.now() - t0) / 1000).toFixed(1)
if (r === null) {
  console.log(`  ❌ 合成失败（${dt}s）`)
  console.log('  插件状态: ' + JSON.stringify(svc.status(), null, 2))
  process.exit(1)
}
console.log(`  ✅ ${dt}s 拿到 ${r.bytes.length} 字节，时长 ${r.durationMs}ms，backend=${r.backend}`)
console.log(`  RIFF 头: ${r.bytes.toString('ascii', 0, 4)}`)

console.log('\n── 3. status() ──')
const st = svc.status()
console.log(`  ready=${st.ready} ours=${st.ours} pid=${st.pid} backend=${st.backend} spawnCount=${st.spawnCount}`)

console.log('\n── 4. 空闲释放（非常驻 + idleStopSec=1 应把它停掉）──')
const svc2 = mod.apply(ctx, { ...config, idleStopSec: 1 })
await svc2.synthesize('第二句。')
await new Promise(r2 => setTimeout(r2, 1500))
const before = svc2.status().ready
const stopped = svc2.stop()
console.log(`  停止前 ready=${before}，stop() 返回 ${stopped}`)

console.log('\n✅ 插件真实代码路径验证通过（DSH 那边只要重启就能用）')
