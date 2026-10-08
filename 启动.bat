@echo off
chcp 65001 >nul
cd /d %~dp0

REM 首次运行自动安装依赖
python -c "import cv2, numpy, PIL, onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo 首次运行，正在安装依赖，请稍候...
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

python app.py
if errorlevel 1 pause
