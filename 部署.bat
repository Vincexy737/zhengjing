@echo off
REM One-click deploy. Rebuild first with:  pyinstaller <spec file>
REM Real logic lives in deploy.ps1 (PowerShell) so Chinese paths stay Unicode-safe.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy.ps1"
if errorlevel 1 (
    echo.
    echo [DEPLOY FAILED - see messages above]
    pause
    exit /b 1
)
echo.
pause
