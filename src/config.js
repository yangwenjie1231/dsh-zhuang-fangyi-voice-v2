/**
 * 插件 Config schema（DSH 0.2.2 约定）。
 *
 * 0.2.2 起配置就是插件 entry 的 Config：
 *   · 标 `.volatile()` 的字段会进设置表单，改完立即生效；
 *   · 宿主不再注册 settings 命名空间；
 *   · 非 volatile 字段只在加载时读一次。
 *
 * `schemastery` 由宿主提供（peer dependency）。链接安装或精简部署里它可能
 * 解析不到，此时**不能让整个插件加载失败** —— 退化成无 schema（宿主按普通
 * 对象处理配置），其余功能照常工作。
 */
let z = null
try {
  z = (await import('@deepseek-ai/schemastery')).default
} catch {
  z = null
}

/** 必须与 cordis.patch.yml 里的条目 id 逐字一致。 */
export const ENTRY_ID = 'zhuang-fangyi-voice-v2'

/** 没有 schemastery 时的普通默认值，字段集合与有 schema 时保持一致。 */
export const DEFAULTS = {
  backend: 'torch',
  gsvRoot: '',
  modelDir: '',
  gptPath: '',
  sovitsPath: '',
  style: 'normal',
  resident: false,
  idleStopSec: 300,
  idleAction: 'stop',
  host: '127.0.0.1',
  port: 8766,
  python: '',
  seed: 42,
  startTimeoutMs: 120000
}

function build () {
  if (z === null) return undefined
  return z.object({
    backend: z.union([z.const('torch'), z.const('auto')]).default('torch'),
    gsvRoot: z.string().default(''),
    modelDir: z.string().default(''),
    gptPath: z.string().default('').volatile(),
    sovitsPath: z.string().default('').volatile(),
    style: z.union([
      z.const('normal'), z.const('firm'), z.const('gentle')
    ]).default('normal').volatile(),
    resident: z.boolean().default(false).volatile(),
    idleStopSec: z.number().default(300).volatile(),
    idleAction: z.union([z.const('stop'), z.const('unload')]).default('stop').volatile(),
    host: z.string().default('127.0.0.1'),
    port: z.natural().max(65535).default(8766),
    python: z.string().default(''),
    seed: z.number().default(42).volatile(),
    startTimeoutMs: z.natural().default(120000)
  })
}

export const Config = build()

export default Config