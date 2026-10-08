# 重建桌面快捷方式（每次部署后都要执行，保证指向最新程序）
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut("$env:USERPROFILE\Desktop\帧净.lnk")
$lnk.TargetPath = (Join-Path $root '帧净\帧净.exe')
$lnk.WorkingDirectory = (Join-Path $root '帧净')
$lnk.IconLocation = (Join-Path $root '帧净\帧净.exe') + ',0'
$lnk.Description = '帧净 · 照片 AI 修复'
$lnk.Save()
Write-Output '桌面快捷方式已更新'
