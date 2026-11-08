/**
 * DSH 0.2.2 插件契约守卫。
 *
 * 这些点都是升级后必须成立、且容易回归的：
 *   · package.json 声明 dsh.client.platform=web 与 ./client 导出
 *   · 导出 locale/*.json 与 icon（设置页/插件管理读 meta.title）
 *   · index.js 导出 Config（0.2.2 用 Config 校验行配置）
 *   · client.js 使用 slots / locale / configForms，并注册到 plugins.item
 *   · 不再使用 0.1.x 的 settings.section 直连路由方案
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const pkg = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'))

assert.equal(pkg.dsh.client.platform, 'web')
assert.ok(pkg.dsh.client.inject.includes('@deepseek-ai/dsh-client-ui-settings'))
assert.equal(pkg.exports['./client'], './client.js')
assert.equal(pkg.exports['./icon'], './icon.svg')
assert.ok(pkg.exports['./locale/*.json'])
assert.ok(pkg.engines.dsh.includes('0.2.2'))
assert.ok(pkg.peerDependencies['@deepseek-ai/schemastery'])

for (const f of ['client.js', 'icon.svg', 'locale/zh.json', 'locale/en.json', 'src/config.js']) {
  assert.ok(fs.existsSync(path.join(root, f)), `missing ${f}`)
}

const zh = JSON.parse(fs.readFileSync(path.join(root, 'locale/zh.json'), 'utf8'))
assert.ok(zh.meta.title && zh.meta.description, 'locale meta.title/description required')

const client = fs.readFileSync(path.join(root, 'client.js'), 'utf8')
for (const needle of ['configForms', 'plugins.item', 'ctx.slots.inject', 'ctx.locale.register']) {
  assert.ok(client.includes(needle), `client.js missing ${needle}`)
}
// 注释里可以提到旧名，但不得真的注册到旧插槽。
assert.ok(!/name:\s*['"]settings\.section['"]/.test(client),
  'must not register into legacy settings.section')

const index = fs.readFileSync(path.join(root, 'index.js'), 'utf8')
assert.ok(index.includes("export { Config }"), 'index.js must export Config')

// 面板必须有真实数据源，且不得在服务不可达时伪造空列表当成功。
assert.ok(client.includes('/v2/models'), 'panel must query the v5 model list')
assert.ok(client.includes('offline'), 'panel must distinguish offline from empty')

// schemastery 缺失时不能整包崩：config.js 必须有降级路径。
const config = fs.readFileSync(path.join(root, 'src/config.js'), 'utf8')
assert.ok(config.includes('catch'), 'config.js must tolerate missing schemastery')
assert.ok(config.includes('DEFAULTS'), 'config.js must expose plain defaults')

console.log('✅ DSH 0.2.2 契约 19 项通过')
