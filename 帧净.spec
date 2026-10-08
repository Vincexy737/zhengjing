# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = []
datas += collect_data_files('rapidocr_onnxruntime')
# AI 修复模型随包内置，离线开箱即用（fp16 为 GPU 加速版，可选）
_model = os.path.join(SPECPATH, 'models', 'big-lama.onnx')
if os.path.exists(_model):
    datas += [(_model, 'models')]
_fp16 = os.path.join(SPECPATH, 'models', 'big-lama.fp16.onnx')
if os.path.exists(_fp16):
    datas += [(_fp16, 'models')]
# AI 超分辨率模型（动漫 Real-ESRGAN 6B + 通用 SwinIR-M GAN，3通道图片超分）
for _sr_name in ['RealESRGAN_x4plus_anime_6B.onnx', 'SwinIR_M_x4_GAN.onnx']:
    _sr = os.path.join(SPECPATH, 'models', _sr_name)
    if os.path.exists(_sr):
        datas += [(_sr, 'models')]
# AI 抠图 / 上色 / 人脸修复模型（用户已下载，随包内置）
for _extra_name in ['RMBG-1.4.onnx', 'ddcolor.onnx', 'gfpgan.onnx']:
    _extra = os.path.join(SPECPATH, 'models', _extra_name)
    if os.path.exists(_extra):
        datas += [(_extra, 'models')]
# 应用图标（exe 图标 / 窗口任务栏图标 / 顶栏 Logo）
_icon = os.path.join(SPECPATH, 'assets', 'icon.png')
if os.path.exists(_icon):
    datas += [(_icon, 'assets')]

binaries = []
binaries += collect_dynamic_libs('onnxruntime')

# NVIDIA CUDA/cuDNN 运行时 DLL（pip 装在 nvidia/*/bin/），CUDA EP 运行必需。
# 去重：collect_dynamic_libs('onnxruntime') 可能已收集部分依赖；
# 跳过 *_train*.dll（只需推理，不需训练）。
try:
    import glob as _glob
    import site as _site
    _have = {os.path.basename(b[0]) for b in binaries}
    for _sp in _site.getsitepackages() + [_site.getusersitepackages()]:
        for _dll in _glob.glob(os.path.join(_sp, 'nvidia', '*', 'bin', '*.dll')):
            _name = os.path.basename(_dll)
            if _name in _have or 'train' in _name.lower() or '32' in _name:
                continue
            binaries.append((_dll, '.'))
            _have.add(_name)
except Exception:
    pass


# 生成式扩展后端（可选增强）：SD1.5-Inpainting 的 ONNX 模型。
# 缺失时应用自动回退 LaMa 级联，不影响其他功能。
# 运行时推理走已内置的 onnxruntime（CUDA EP），无需 torch/diffusers。
_sd_model = os.path.join(SPECPATH, 'models', 'sd15-inpaint-onnx')
if os.path.isdir(_sd_model):
    datas += [(_sd_model, 'models/sd15-inpaint-onnx')]


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['onnxruntime', 'rapidocr_onnxruntime',
                   'pyclipper', 'shapely', 'shapely.geometry',
                   # setuptools>=78 的 pkg_resources 在 py<3.12 经 jaraco.context
                   # 间接依赖 backports.tarfile；pyi_rth_pkgres 运行期会 import 它，
                   # 缺失会导致打包版启动即 ModuleNotFoundError: No module 'backports'
                   'backports.tarfile'],
    # onnxruntime.training 会静态依赖 torch，运行时用不到，全部排除
    excludes=['torch', 'torchaudio', 'torchvision', 'onnxruntime.training',
              'llvmlite', 'numba', 'scipy', 'matplotlib', 'pandas',
              'IPython', 'jupyter'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='帧净',
    icon=os.path.join(SPECPATH, 'assets', 'icon.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='帧净',
)
