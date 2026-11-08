/**
 * 插件 Config schema（DSH 0.2.2 约定）。
 *
 * 0.2.2 起配置就是插件 entry 的 Config：
 *   · 标 `.volatile()` 的字段会进设置表单，改完立即生效；
 *   · 宿主不再注册 settings 命名空间；
 *   · 非 volatile 字段只在加载时读一次。
 *
 * `sovitsPath` / `gptPath` 是用户可选项，所以必须 volatile。
 */
import z from '@deepseek-ai/schemastery'

/** 必须与 cordis.patch.yml 里的条目 id 逐字一致。 */
export const ENTRY_ID = 'zhuang-fangyi-voice-v2'

export const Config = z.object({
  /** 推理后端：v2 插件只支持 torch（v5 无 ONNX 包）。 */
  backend: z.union([z.const('torch'), z.const('auto')]).default('torch'),
  /** GPT-SoVITS 检出目录（含 GPT_SoVITS/ 与 gsv-v5-pretrained）。 */
  gsvRoot: z.string().default(''),
  /** 模型目录；留空走自动探测。 */
  modelDir: z.string().default(''),
  /** GPT/t2s 权重；留空用 <modelDir>/torch_weights/zfh-e4.ckpt。 */
  gptPath: z.string().default('').volatile(),
  /** SoVITS 权重；留空用 configuration.json 默认或最高 epoch。 */
  sovitsPath: z.string().default('').volatile(),
  /** 音色风格：normal / firm / gentle。 */
  style: z.union([
    z.const('normal'), z.const('firm'), z.const('gentle')
  ]).default('normal').volatile(),
  /** 常驻与空闲释放。 */
  resident: z.boolean().default(false).volatile(),
  idleStopSec: z.number().default(300).volatile(),
  idleAction: z.union([z.const('stop'), z.const('unload')]).default('stop').volatile(),
  host: z.string().default('127.0.0.1'),
  port: z.natural().max(65535).default(8766),
  python: z.string().default(''),
  seed: z.number().default(42).volatile(),
  startTimeoutMs: z.natural().default(120000)
})

export default Config
