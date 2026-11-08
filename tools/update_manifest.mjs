/** 为 DSH 0.2.2 更新 package.json：locale/icon 导出、依赖、peer。 */
import fs from 'node:fs'

const p = new URL('../package.json', import.meta.url)
const j = JSON.parse(fs.readFileSync(p, 'utf8'))

j.exports = {
  '.': './index.js',
  './client': './client.js',
  './locale/*.json': './locale/*.json',
  './icon': './icon.svg',
  './package.json': './package.json'
}
j.files = [
  'index.js', 'client.js', 'cordis.patch.yml', 'src',
  'locale/*.json', 'icon.svg', 'tools', 'docs', 'assets',
  'README.md', 'CHANGELOG.md'
]
j.dependencies = { '@deepseek-ai/schemastery': '^3.18.4' }
j.peerDependencies = {
  '@deepseek-ai/cordis': '~4.1.0',
  '@deepseek-ai/dsh-client-ui-renderer': '>=0.2.0-0',
  '@deepseek-ai/dsh-client-ui-settings': '>=0.2.0-0',
  react: '^18.2.0'
}
j.dsh = {
  bundle: { patch: './cordis.patch.yml' },
  client: {
    platform: 'web',
    immediately: true,
    inject: [
      '@deepseek-ai/dsh-client-locale',
      '@deepseek-ai/dsh-client-ui-renderer',
      '@deepseek-ai/dsh-client-ui-settings'
    ]
  }
}
j.engines = { dsh: '>=0.2.2-0', node: '>=20' }

fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\n', 'utf8')
console.log('package.json updated')