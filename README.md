# 帧净 · 照片 AI 修复

Windows 桌面端的照片 / 视频修复工具：去水印、清晰度增强、降噪、老照片修复、抠图等。
纯 Python + Tkinter 实现，界面控件全部自绘，离线可用（模型随包或本地放置）。

![platform](https://img.shields.io/badge/platform-Windows-blue) ![python](https://img.shields.io/badge/python-3.10+-blue) ![license](https://img.shields.io/badge/license-MIT-green)

## 功能

| 分组 | 功能 | 说明 |
|---|---|---|
| 修复增强 | 去水印 | LaMa AI 修复（纹理结构还原）/ FSR 频率选择性重建 / 经典 TELEA·NS 扩散 |
| | 清晰度增强 | AI 超分辨率 2×（动漫 Real-ESRGAN 6B / 通用 SwinIR-M GAN） |
| | 图片降噪 | 四种模式（快速 / 精细 / 保颗粒 / 自动） |
| | 老照片修复 | 划痕检测、去噪、上色（DDColor）、锐化 |
| 编辑工具 | 裁剪与尺寸 | 预设比例、证件照尺寸、智能裁剪 |
| | 格式转换 | JPG / PNG / WebP / BMP，互转与按体积压缩 |
| | 调色滤镜 | 预设 + 亮度/对比度/饱和度/色温/伽马/锐化等手动调节 |
| AI 实验室 | AI 抠图 | RMBG-1.4 发丝级边缘；U²-Net 轻量备选 |
| | 物体消除 | 框选目标后智能填充 |
| 批量处理 | 批量任务 | 多图队列，支持格式/尺寸/预设批量套用 |

## 视频处理（暂未接入界面）

`core/processor.py` 中实现了完整的视频管线：`process_video` 逐帧修复、
`_temporal` 时域多帧重建、`calibrate_alpha` 半透明水印透明度标定、
`strip_subtitle_stream` 软字幕剥离、`AudioPlayer` 音轨预览。

**这些函数当前没有任何界面入口**（`app.py` 与 `ui/` 均未引用），
需自行调用或另行开发界面。

## 运行

### 方式一：双击启动（Windows）

下载或克隆后双击 **`启动.bat`**，首次运行会自动安装依赖，然后打开界面。

### 方式二：手动

```bash
python -m pip install -r requirements.txt
python app.py
```

### 自检

```bash
python app.py --selftest    # 验证 LaMa 推理链路，需已装 onnxruntime 且 models/big-lama.onnx 就位
python _smoke.py            # core 算法 + 全部功能页冒烟测试，会拉起 Tk 窗口
```

> 两条命令均需先装齐依赖并备好模型；仓库未包含验证记录。

## 模型

模型体积较大，**不随 git 仓库分发**，需自行下载后放入 `models/`。
应用会自动查找该目录（打包后为 exe 同级 `models/`）。

| 文件 | 用途 | 大小 | 必需 |
|---|---|---|---|
| `big-lama.onnx` | 去水印 AI 修复 | ~207 MB | 强烈建议 |
| `big-lama.fp16.onnx` | 同上，GPU 加速版 | ~103 MB | 可选 |
| `RealESRGAN_x4plus_anime_6B.onnx` | 动漫超分 | ~17 MB | 清晰度增强 |
| `SwinIR_M_x4_GAN.onnx` | 通用照片超分 | ~59 MB | 清晰度增强 |
| `RMBG-1.4.onnx` | AI 抠图 | ~176 MB | 抠图页 |
| `ddcolor.onnx` | 老照片上色 | ~136 MB | 上色功能 |
| `gfpgan.onnx` | 人脸修复 | ~341 MB | 可选 |

`core/models.py` 的 `CATALOG` 内置了后四个模型的 HuggingFace 直链与国内镜像，
应用内可直接下载并做 SHA256 校验。LaMa / Real-ESRGAN 请从
[Saafke/LaMaONNX](https://github.com/Saafke/LaMaONNX) 等公开渠道获取。

缺少模型时对应功能会自动降级（去水印退回 FSR，超分按钮置灰等），不会崩溃。

## 打包

```bash
python -m pip install pyinstaller
pyinstaller 帧净.spec
```

产物在 `dist/帧净/`。本机部署（同步到 `帧净/` 并刷新桌面快捷方式）：

```powershell
.\部署.bat
```

`tools/convert_sr.py` 是构建期可选工具，用于把 Real-ESRGAN 权重转成 onnx，
运行应用不需要它。

## 项目结构

```
app.py            主窗口、自绘控件库、去水印页 / 清晰度增强页
core/
  processor.py    修复核心、编解码、编码器探测、视频逐帧管线
  lama.py         LaMa ONNX 引擎（CUDA > DirectML > CPU）
  sr.py           AI 超分（tile 分块 + feather 融合）
  imglib.py       中文路径安全读写、EXIF 纠正、按体积压缩
  models.py       扩展模型注册表 / 下载 / ORT 会话
  ...             抠图、老照片、降噪、基础、调色、滤镜、批量
ui/
  base.py         TabBase：功能页统一骨架（后台任务/进度/取消/撤销）
  widgets.py      侧边导航、滑动对比视图、批量列表、参数行
  tabs/           8 个功能页
tools/            构建期辅助脚本
```

## 依赖

```
opencv-contrib-python   # 需 contrib：FSR 修复在 cv2.xphoto
numpy
Pillow
imageio-ffmpeg          # 内置 ffmpeg，无需单独安装
rapidocr-onnxruntime    # OCR 水印检测，缺失则该功能降级
onnxruntime-gpu         # CUDA > DirectML > CPU 自动选择，均不可用时回退 CPU
```

较新 NVIDIA 显卡（驱动仅支持 CUDA 12）需换用 cu11 系列的替代版本，
见 `requirements.txt` 内注释。

## 说明

- 面向 Windows 开发与验证，未在其它平台测试。视频音轨预览依赖 `winsound`
  （Windows 专有），桌面快捷方式脚本依赖 COM。
- 编码器自动探测 NVENC / QSV / AMF，逐个实测可用性，失败降级到 x264。
- 调试日志：设置环境变量 `ZHENGJING_DEBUG=1`，日志写入 `%LOCALAPPDATA%\帧净\debug.log`。

## License

MIT