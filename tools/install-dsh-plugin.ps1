# dsh-zhuang-fangyi-voice · 装进 DSH（把本插件复制到 profile 的 node_modules 并启用）
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\install-dsh-plugin.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\install-dsh-plugin.ps1 -Profile desktop
#
# 为什么需要脚本：
#   1. `Copy-Item -Recurse 源 已存在的目标` **不会合并**，会在目标下再建一层
#      （`<dst>\dsh-zhuang-fangyi-voice\dsh-zhuang-fangyi-voice`）—— 结果是插件
#      半残、而且报错指向莫名其妙的地方；
#   2. DSH 缓存 ESM 模块：改了 index.js 后 disable→enable **不够**，要重启宿主；
#   3. 装完要 enable，而 `plugin_manager` 只认 `include:<短 id>` 这种目标名。
#
# 装完请**重启 DSH**（或在插件管理器里 disable→enable 一次）。

[CmdletBinding()]
param(
    [string]$Profile = 'desktop',
    [string]$DshHome = '',
    [switch]$NoEnable
)

$ErrorActionPreference = 'Stop'
$Src = Split-Path -Parent $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($DshHome)) {
    if ($env:DSH_HOME) { $DshHome = $env:DSH_HOME }
    else { $DshHome = Join-Path $env:USERPROFILE '.dsh' }
}

$Nodes = Join-Path $DshHome "profiles\$Profile\node_modules"
$Dst = Join-Path $Nodes 'dsh-zhuang-fangyi-voice'

Write-Host "源:   $Src"
Write-Host "目标: $Dst"
Write-Host ''

if (-not (Test-Path $Nodes)) {
    Write-Host "找不到 profile 的 node_modules：$Nodes" -ForegroundColor Red
    Write-Host "   确认 DSH 装过、且 profile 名字对（默认 desktop）。" -ForegroundColor Yellow
    exit 1
}

# 要拷的东西：插件入口 + Python 包 + 工具 + 文档。
# **不拷** .git / __pycache__ / models（模型很大，且用户自己下载的位置由
# ZFH_MODEL_DIR 决定 —— 复制一份会让"模型在哪"变得含糊）。
$Items = @('index.js', 'package.json', 'cordis.patch.yml', 'README.md', 'LICENSE', 'pyproject.toml', 'requirements.txt')
$Dirs = @('src', 'tools', 'docs')
$Skip = @('__pycache__')

function Copy-Tree([string]$from, [string]$to) {
    New-Item -ItemType Directory -Force -Path $to | Out-Null
    Get-ChildItem -LiteralPath $from -Force | ForEach-Object {
        if ($Skip -contains $_.Name) { return }
        $target = Join-Path $to $_.Name
        if ($_.PSIsContainer) { Copy-Tree $_.FullName $target }
        else { Copy-Item $_.FullName $target -Force }
    }
}

# 目标存在就先删（避免上面说的"嵌套一层"）
if (Test-Path $Dst) {
    Write-Host '  目标已存在 → 先删除（避免嵌套复制）'
    Remove-Item $Dst -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $Dst | Out-Null

foreach ($f in $Items) {
    $p = Join-Path $Src $f
    if (Test-Path $p) { Copy-Item $p (Join-Path $Dst $f) -Force }
}
foreach ($d in $Dirs) {
    $p = Join-Path $Src $d
    if (Test-Path $p) { Copy-Tree $p (Join-Path $Dst $d) }
}

# 完整性守卫：缺入口的插件装上去只会静默不工作
$need = @('index.js', 'package.json', 'cordis.patch.yml')
$missing = @()
foreach ($f in $need) { if (-not (Test-Path (Join-Path $Dst $f))) { $missing += $f } }
if ($missing.Count -gt 0) {
    Write-Host "部署不完整，缺：$($missing -join ', ')" -ForegroundColor Red
    exit 1
}

$pyFiles = (Get-ChildItem (Join-Path $Dst 'src') -Recurse -File -Filter '*.py' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '__pycache__' }).Count
Write-Host "  已复制：入口 3 个 + Python 源文件 $pyFiles 个" -ForegroundColor Green

if ($NoEnable) {
    Write-Host "`n跳过启用（-NoEnable）。手动启用：plugin_manager set_plugin target=include:zhuang-fangyi-voice enabled=true"
    exit 0
}

Write-Host "`n下一步（DSH 侧）："
Write-Host "  1) 重启 DSH（ESM 模块有缓存，disable→enable 对入口文件不够）"
Write-Host "  2) 或在插件管理器里：target=include:zhuang-fangyi-voice，enabled=false → true"
Write-Host ''
Write-Host "然后确认服务挂上了：宿主日志里应有「庄方宜语音：已挂载 zfhVoice 服务」" -ForegroundColor Cyan
Write-Host "Python 依赖与模型见 docs/安装提示词.md" -ForegroundColor Cyan
exit 0
