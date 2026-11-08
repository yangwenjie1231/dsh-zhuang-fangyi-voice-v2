import fs from 'node:fs'
const p = new URL('../package.json', import.meta.url)
const j = JSON.parse(fs.readFileSync(p, 'utf8'))
j.devDependencies = { '@deepseek-ai/schemastery': '3.19.0' }
fs.writeFileSync(p, JSON.stringify(j, null, 2) + '\n', 'utf8')
console.log('devDependency added')