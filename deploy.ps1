# 一键部署：结束旧程序 -> 镜像替换程序目录 -> 更新桌面快捷方式 -> 启动
#
# 为什么用 PowerShell 而不是纯 .bat：
#   部署涉及大量中文路径（帧净、桌面快捷方式等），.bat 在不同活动代码页
#   下（尤其是被其它 Shell 以 cmd /c 调用时）会把中文解析乱码，导致
#   robocopy 静默同步失败却仍报“部署完成”。PowerShell 原生按 Unicode
#   处理路径，不依赖代码页，稳定可靠。
#
# 用法：pyinstaller 帧净.spec 打包完成后，双击 部署.bat（或直接运行本脚本）。

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# 1) 结束正在运行的旧程序
Get-Process -Name '帧净' -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

# 2) 镜像同步 dist\帧净 -> 帧净
$src = Join-Path $root 'dist\帧净'
$dst = Join-Path $root '帧净'
if (-not (Test-Path -LiteralPath $src)) {
    Write-Host "部署失败：找不到 $src，请先运行 pyinstaller 帧净.spec" -ForegroundColor Red
    exit 1
}
robocopy $src $dst /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
# robocopy 退出码：0-7 成功，>=8 失败
if ($LASTEXITCODE -ge 8) {
    Write-Host "部署失败：robocopy 退出码 $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# 3) 更新桌面快捷方式（复用现有脚本）
& (Join-Path $root '更新快捷方式.ps1')

# 4) 启动
$exe = Join-Path $dst '帧净.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    Write-Host "部署失败：未找到 $exe" -ForegroundColor Red
    exit 1
}
Start-Process -FilePath $exe -WorkingDirectory $dst
Write-Host '部署完成：程序与桌面快捷方式均已替换，程序已启动。' -ForegroundColor Green
