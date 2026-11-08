import fs from 'node:fs'
const p = new URL('../package.json', import.meta.url)
const j = JSON.parse(fs.readFileSync(p, 'utf8'))
delete j.dependencies
j.peerDependencies = {
  ...j.peerDependencies,
  '@deepseek-ai/schemastery': '^3.18.4'
}
fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\n', 'utf8')
console.log('peer deps fixed')