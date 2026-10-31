/** 设置面板宿主路由：只暴露模型列表/选择/启动，不泄露内部端口。 */
import assert from 'node:assert/strict'
import { createVoiceHandler, ROUTE_PREFIX } from '../src/voice_routes.js'

function fakeRes () {
  return {
    status: 0,
    headers: {},
    body: '',
    writeHead (s, h) { this.status = s; this.headers = h },
    end (b) { this.body = String(b ?? '') }
  }
}

function fakeReq (method, path, body) {
  const listeners = {}
  return {
    method,
    url: ROUTE_PREFIX + path,
    headers: {},
    on (ev, fn) {
      listeners[ev] = listeners[ev] ?? []
      listeners[ev].push(fn)
      if (ev === 'data' && body !== undefined) fn(Buffer.from(body))
      if (ev === 'end') fn()
      return this
    },
    destroy () {}
  }
}

const service = {
  models: async () => ({ models: [{ epoch: 32, selected: true, path: '/m/e32.pth' }], current: '/m/e32.pth' }),
  status: () => ({ alive: true, baseUrl: 'http://127.0.0.1:8766' }),
  start: async () => true,
  setModel: async id => ({ applied: true, current: id, models: [] }),
  rescanModels: async () => ({ models: [] })
}

const handler = createVoiceHandler(service, { warn () {} })

{
  const res = fakeRes()
  await handler(fakeReq('GET', '/models'), res)
  assert.equal(res.status, 200)
  const body = JSON.parse(res.body)
  assert.equal(body.ok, true)
  assert.equal(body.models[0].epoch, 32)
  assert.equal(body.service.running, true)
}

{
  const res = fakeRes()
  await handler(fakeReq('POST', '/model', JSON.stringify({ model: '/m/e16.pth' })), res)
  assert.equal(res.status, 200)
  assert.equal(JSON.parse(res.body).applied, true)
}

{
  const res = fakeRes()
  await handler(fakeReq('POST', '/model', '{bad json'), res)
  assert.equal(res.status, 400)
}

{
  const res = fakeRes()
  await handler(fakeReq('POST', '/rescan'), res)
  assert.equal(res.status, 200)
}

{
  const res = fakeRes()
  await handler(fakeReq('GET', '/unknown'), res)
  assert.equal(res.status, 404)
}

console.log('✅ voice routes 6 项通过')
