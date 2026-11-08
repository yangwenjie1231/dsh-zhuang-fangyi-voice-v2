#!/usr/bin/env node
/** 从 app.asar 里按路径抽文件（只读，不修改安装目录）。 */
import fs from 'node:fs'

const [asarPath, innerPath, outPath] = process.argv.slice(2)
const fd = fs.openSync(asarPath, 'r')
const head = Buffer.alloc(16)
fs.readSync(fd, head, 0, 16, 0)
const headerSize = head.readUInt32LE(12)
const jsonBuf = Buffer.alloc(headerSize)
fs.readSync(fd, jsonBuf, 0, headerSize, 16)
const header = JSON.parse(jsonBuf.toString('utf8'))
const baseOffset = 8 + head.readUInt32LE(4)

function lookup (node, parts) {
  let cur = node
  for (const part of parts) {
    if (!cur.files || !cur.files[part]) return null
    cur = cur.files[part]
  }
  return cur
}

const entry = lookup(header, innerPath.split('/').filter(Boolean))
if (!entry) {
  console.error('not found:', innerPath)
  process.exit(1)
}
if (entry.files) {
  console.error('is a directory:', innerPath)
  process.exit(1)
}
const buf = Buffer.alloc(Number(entry.size))
fs.readSync(fd, buf, 0, buf.length, baseOffset + Number(entry.offset))
fs.closeSync(fd)
fs.writeFileSync(outPath, buf)
console.log(`wrote ${outPath} (${buf.length} bytes)`)
