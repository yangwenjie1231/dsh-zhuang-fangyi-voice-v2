/**
 * 语音插件设置面板的宿主路由。
 *
 * 浏览器端不能直接调用推理服务的 8766 端口（服务可能未启动、且不应把
 * 内部端口暴露给页面），所以由宿主提供同源 JSON API，再转给 zfhVoice 服务。
 */

export const ROUTE_PREFIX = '/api/zhuang-fangyi/voice'
const MAX_BODY = 16 * 1024

function readBody (req) {
  return new Promise((resolve, reject) => {
    let size = 0
    const chunks = []
    req.on('data', chunk => {
      size += chunk.length
      if (size > MAX_BODY) {
        reject(new Error('body too large'))
        req.destroy()
        return
      }
      chunks.push(chunk)
    })
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')))
    req.on('error', reject)
  })
}

function json (res, status, body) {
  const data = Buffer.from(JSON.stringify(body), 'utf8')
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': data.length,
    'cache-control': 'no-store'
  })
  res.end(data)
}

/**
 * @param {object} service - zfhVoiceV2 服务
 * @param {object} logger
 */
export function createVoiceHandler (service, logger) {
  return async function handler (req, res) {
    const url = new URL(req.url, 'http://127.0.0.1')
    const path = url.pathname.slice(ROUTE_PREFIX.length) || '/'
    try {
      if (req.method === 'GET' && (path === '/' || path === '/models')) {
        const models = await service.models()
        const status = service.status?.() ?? {}
        return json(res, 200, {
          ok: true,
          ...models,
          service: {
            running: status.alive === true || status.ours === true,
            baseUrl: status.baseUrl ?? null
          }
        })
      }
      if (req.method === 'POST' && path === '/start') {
        const ok = await service.start()
        return json(res, 200, { ok: ok === true })
      }
      if (req.method === 'POST' && (path === '/model' || path === '/rescan')) {
        let model = null
        if (path === '/model') {
          const raw = await readBody(req)
          try {
            model = JSON.parse(raw || '{}').model ?? null
          } catch {
            return json(res, 400, { ok: false, error: 'invalid json' })
          }
        }
        const result = path === '/model'
          ? await service.setModel(model)
          : await service.rescanModels()
        const ok = path === '/model' ? result?.applied === true : Array.isArray(result?.models)
        return json(res, ok ? 200 : 400, { ok, ...result })
      }
      return json(res, 404, { ok: false, error: 'not found' })
    } catch (error) {
      logger?.warn?.(`zhuang-fangyi-voice-v2: 设置面板请求失败 ${path}: ${error?.message ?? error}`)
      return json(res, 500, { ok: false, error: String(error?.message ?? error) })
    }
  }
}
