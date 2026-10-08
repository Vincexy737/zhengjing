# -*- coding: utf-8 -*-
"""
核心处理算法：
1. 图片去水印 —— OpenCV inpaint 图像修复（TELEA / Navier-Stokes）
2. 视频去水印 —— FFmpeg 解码 -> 逐帧对选区做 inpaint -> FFmpeg 编码 -> 回填原音轨
3. 字幕处理 —— 软字幕直接剥离字幕流；硬字幕（烧录在画面中）通过选区修复去除
"""

import os
import bisect
import json
import shutil
import logging
import tempfile
import threading
import subprocess
from collections import deque

import numpy as np
import cv2
import imageio_ffmpeg

_log = logging.getLogger(__name__)

# 缓存 / 日志等写入用户目录时使用的子目录名（避免装进只读的 Program Files）
_APP_DIR_NAME = "帧净"

INPAINT_TELEA = cv2.INPAINT_TELEA
INPAINT_NS = cv2.INPAINT_NS


class VideoExportCancelled(RuntimeError):
    """用户主动取消视频导出"""


class ProcessingCancelled(RuntimeError):
    """用户主动取消图片修复"""

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

# Windows 上 GUI（无控制台）程序启动 ffmpeg 必须带 CREATE_NO_WINDOW，
# 否则每次调用都会弹出一个黑色控制台窗口
_NO_WIN_FLAGS = 0x08000000 if os.name == "nt" else 0


def nowin_kwargs(**kw):
    """给 subprocess 调用附加「不弹黑窗」标志"""
    if _NO_WIN_FLAGS:
        kw["creationflags"] = _NO_WIN_FLAGS
    return kw


def _temp_path(suffix, prefix="zhengjing_"):
    """在系统 temp 下生成唯一临时路径。

    先用 mkstemp 预留唯一名（避免固定名导致的实例间冲突与本地预创建 /
    符号链接风险），再删除空文件交由 ffmpeg 自行创建。单用户桌面应用下
    预留与使用之间的窗口可忽略。
    """
    fd, p = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    os.close(fd)
    try:
        os.unlink(p)
    except OSError:
        pass
    return p


def _safe_cli_path(p):
    """防止以 '-' 开头的路径被 ffmpeg 当作命令行选项（参数注入）。

    Windows 下文件对话框给出的都是绝对路径（盘符开头），不会触发；
    仅在极端情况下（相对路径 / 挂载点以 '-' 开头）加 './' 前缀。
    """
    p = str(p)
    if p.startswith("-"):
        return os.path.join(".", p)
    return p


# ----------------------------------------------------------------------
# 编码器自动检测：有硬件编码（NVENC/QSV/AMF）就用，又快画质又好
# 注意：ffmpeg 编译支持 ≠ 驱动可用（旧驱动会让 NVENC/QSV/AMF 初始化失败），
# 因此每个候选编码器都会做一次 1 帧实测，失败自动降级到下一个/软件编码
# ----------------------------------------------------------------------
_ENC_CACHE = {}
_ENC_LOCK = threading.Lock()
_ENC_CACHE_VERSION = 1


def _app_data_dir():
    """用户可写数据目录（%LOCALAPPDATA%\\帧净），失败回退系统 temp。"""
    base = (os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
            or tempfile.gettempdir())
    d = os.path.join(base, _APP_DIR_NAME)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = tempfile.gettempdir()
    return d


def _load_enc_cache():
    """读取上次实测可用的编码器，省去每次启动最多 3 次 ffmpeg 实测。

    若缓存编码器后续因驱动更新失效，编码失败处会再次实测并提示，故可信任缓存。
    """
    try:
        with open(os.path.join(_app_data_dir(), "encoder.json"),
                  encoding="utf-8") as f:
            data = json.load(f)
        enc = data.get("enc")
        if data.get("v") == _ENC_CACHE_VERSION and enc and len(enc) == 3:
            return (enc[0], list(enc[1]), enc[2])
    except FileNotFoundError:
        pass
    except Exception:
        _log.debug("读取编码器缓存失败", exc_info=True)
    return None


def _save_enc_cache(enc):
    try:
        with open(os.path.join(_app_data_dir(), "encoder.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"v": _ENC_CACHE_VERSION, "enc": list(enc)}, f)
    except OSError:
        _log.debug("写入编码器缓存失败", exc_info=True)


def _probe_encoder(codec):
    """实测编码器能否真正初始化（驱动/设备可用性）"""
    try:
        r = subprocess.run(
            [FFMPEG_EXE, "-hide_banner", "-v", "error",
             "-f", "lavfi", "-i", "color=c=black:s=128x128:r=1:d=0.1",
             "-c:v", codec, "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, timeout=30,
            **nowin_kwargs())
        return r.returncode == 0
    except Exception:
        return False


def detect_encoder():
    """返回 (codec, output_params, 显示名)；按 实测可用性 优先选择，结果缓存。
    线程安全：可能被后台线程（启动预检）与导出线程同时调用。"""
    if _ENC_CACHE:
        return _ENC_CACHE["enc"]
    with _ENC_LOCK:
        if _ENC_CACHE:
            return _ENC_CACHE["enc"]
        cached = _load_enc_cache()
        if cached is not None:
            _ENC_CACHE["enc"] = cached
            return cached
        try:
            out = subprocess.run([FFMPEG_EXE, "-hide_banner", "-encoders"],
                                 capture_output=True, text=True,
                                 errors="ignore",
                                 **nowin_kwargs()).stdout
        except Exception:
            _log.debug("ffmpeg -encoders 查询失败", exc_info=True)
            out = ""
        candidates = []
        if "h264_nvenc" in out:
            candidates.append(("h264_nvenc", ["-preset", "p5", "-cq", "15"],
                               "NVIDIA NVENC 硬件编码"))
        if "h264_qsv" in out:
            candidates.append(("h264_qsv", ["-preset", "medium",
                                             "-global_quality", "14"],
                               "Intel QSV 硬件编码"))
        if "h264_amf" in out:
            candidates.append(("h264_amf", ["-quality", "quality", "-rc", "cqp",
                                             "-qp_i", "16", "-qp_p", "16"],
                               "AMD AMF 硬件编码"))
        enc = None
        for codec, params, name in candidates:
            if _probe_encoder(codec):
                enc = (codec, params, name)
                break
            _log.debug("硬件编码器 %s 实测不可用，降级", codec)
        if enc is None:
            enc = ("libx264", ["-crf", "14", "-preset", "medium"],
                   "x264 软件编码（多线程优化）")
        _ENC_CACHE["enc"] = enc
        _save_enc_cache(enc)
        return enc


# ----------------------------------------------------------------------
# 图像读写（兼容中文路径）
# ----------------------------------------------------------------------
def load_image(path):
    """读取图片为 BGR ndarray，兼容中文路径"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片：{path}")
    return img


def save_image(img, path):
    """保存图片，兼容中文路径"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"不支持的图片格式：{ext}")
    buf.tofile(path)


# ----------------------------------------------------------------------
# 蒙版工具
# ----------------------------------------------------------------------
def mask_bbox(mask, pad=0):
    """返回蒙版非零区域的外接矩形 (x1, y1, x2, y2)，空蒙版返回 None"""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x1 = max(int(xs.min()) - pad, 0)
    y1 = max(int(ys.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, w)
    y2 = min(int(ys.max()) + pad + 1, h)
    return x1, y1, x2, y2


def _prepare_mask(mask, radius):
    """二值化并适度膨胀蒙版，确保水印边缘被完全覆盖。
    膨胀迭代封顶 5 次：超过 5px 只会白白撑大 ROI、
    放大后续 inpaint/FSR 的耗时，对覆盖边缘无实质帮助。"""
    m = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.dilate(m, kernel, iterations=max(1, min(5, radius // 3)))
    return m


def clean_mask(mask, min_px=48, rel=0.01):
    """剔除蒙版中的细小碎斑（画笔误点/甩出的孤立小点）：
    每个碎斑都会各自触发一次修复，在画面上留下与内容无关的
    可疑小补丁。仅清除「面积 < 主体选区 rel 倍 且 < min_px」
    的连通域——本身就是小选区的蒙版不受影响。就地修改并返回。"""
    if mask is None or cv2.countNonZero(mask) == 0:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    if n <= 2:                       # 背景 + 唯一选区：无碎斑可言
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    keep_min = max(float(min_px), float(areas.max()) * rel)
    for k in np.where(areas < keep_min)[0] + 1:
        mask[labels == k] = 0
    return mask


def detect_watermark_photo(image_bgr, sensitivity=0.55):
    """自动检测图片中的文字 / Logo 水印。

    多尺度形态学 top-hat / black-hat 提取细结构 → 单阈值二值化 →
    闭操作连接同行文字 → 开操作去碎斑 → 连通域形状过滤（面积、
    填充率、长宽比）保留文字状区域 → 膨胀覆盖边缘。

    Args:
        image_bgr: BGR uint8 numpy (H, W, 3)
        sensitivity: 0.2–0.9，越大越敏感

    Returns:
        (mask_u8, coverage): mask uint8 (H, W) 0/255；coverage 0–1
    """
    if image_bgr is None:
        return np.zeros((1, 1), np.uint8), 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = h * w

    combined = np.zeros_like(gray)
    for ksize in (11, 19, 31):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        combined = np.maximum(combined, cv2.add(tophat, blackhat))

    max_val = float(combined.max())
    if max_val < 8:
        return np.zeros_like(gray), 0.0

    thresh = max_val * (0.45 - 0.15 * sensitivity)
    _, mask = cv2.threshold(combined, thresh, 255, cv2.THRESH_BINARY)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask)
    min_area = max(40, img_area * 0.00015)
    max_area = img_area * 0.3
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue
        rw = stats[i, cv2.CC_STAT_WIDTH]
        rh = stats[i, cv2.CC_STAT_HEIGHT]
        fill = area / float(rw * rh) if rw * rh > 0 else 0
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        if 0.04 < fill < 0.95 and aspect < 12:
            result[labels == i] = 255

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    result = cv2.dilate(result, dilate_kernel, iterations=2)

    coverage = float(np.count_nonzero(result)) / img_area
    return result, coverage


_ocr_engine = None


def detect_watermark_ocr(image_bgr, sensitivity=0.55):
    """用 OCR 检测包含 'ai' 字样的文字行作为水印去除。

    RapidOCR 检测文字 → 仅保留文本中含 'ai'（不区分大小写）的行
    → 框内背景色估计（边缘环带中位数）→ 像素与背景色 BGR 距离图
    → Otsu 自适应阈值分割出文字像素 → 与框做交集 → 小膨胀覆盖抗锯齿。
    完整覆盖文字笔画，且不溢出到背景。

    Args:
        image_bgr: BGR uint8 numpy (H, W, 3)
        sensitivity: 0.2–0.9，越大越敏感（检测更多文字）

    Returns:
        (mask_u8, coverage): mask uint8 (H, W) 0/255；coverage 0–1
    """
    global _ocr_engine
    if image_bgr is None:
        return np.zeros((1, 1), np.uint8), 0.0
    try:
        if _ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
    except Exception:
        return None, 0.0

    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    result, _ = _ocr_engine(image_bgr)
    if not result:
        return mask, 0.0

    score_thresh = 0.60 - 0.30 * sensitivity
    for box, text, score in result:
        if score < score_thresh:
            continue
        if "ai" not in text.lower():
            continue
        pts = np.array(box, dtype=np.int32)
        x, y, ww, hh = cv2.boundingRect(pts)
        x2, y2 = min(w, x + ww), min(h, y + hh)
        x = max(0, x); y = max(0, y)
        roi = image_bgr[y:y2, x:x2]
        if roi.size == 0:
            continue
        rh, rw = roi.shape[:2]
        # 背景色估计：框边缘环带（去掉中心 60%）的中位数，避开文字
        if rh > 4 and rw > 4:
            m1, m2 = int(rh * 0.2), int(rh * 0.8)
            n1, n2 = int(rw * 0.2), int(rw * 0.8)
            edge = roi.copy()
            edge[m1:m2, n1:n2] = 0
            em = edge.any(axis=2)
            bg = np.median(roi[em].reshape(-1, 3), axis=0)
        else:
            bg = np.median(roi.reshape(-1, 3), axis=0)
        # 像素与背景色的 BGR 欧氏距离
        diff = roi.astype(np.float32) - bg.astype(np.float32)
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
        # Otsu 自适应阈值分割文字
        _, bw = cv2.threshold(dist_u8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 与框做交集，不溢出框外
        local = np.zeros((rh, rw), np.uint8)
        cv2.fillPoly(local, [pts - np.array([x, y], dtype=np.int32)], 255)
        bw = cv2.bitwise_and(bw, local)
        mask[y:y2, x:x2] = cv2.bitwise_or(mask[y:y2, x:x2], bw)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    mask = cv2.dilate(mask, kernel, iterations=1)
    coverage = float(np.count_nonzero(mask)) / (h * w)
    return mask, coverage


# ----------------------------------------------------------------------
# 修复核心：主修复 + 增强二遍 + 羽化融合
# ----------------------------------------------------------------------
def _fix_region(roi, roi_mask, radius, method, enhance,
                progress_cb=None, cancel_check=None, stage=(0.0, 1.0)):
    """对 ROI 内 mask 区域做修复：
    1) 主修复；2) 增强：内缩蒙版二遍修复 + 锐化补偿；
    3) 羽化融合：修复区与原图软过渡，避免生硬接缝。
    注意：不做全局平滑（如 bilateral），否则会糊掉选区外的清晰画面；
    修复只替换蒙版内部，蒙版外像素保持原样。
    progress_cb(done, total)：按主修复→（增强二遍）→融合的阶段进度上报；
    cancel_check() 返回 True 时抛 ProcessingCancelled（阶段间检测）。"""

    def _check():
        if cancel_check is not None and cancel_check():
            raise ProcessingCancelled("处理已取消")

    def _emit(t):
        if progress_cb is not None:
            lo, hi = stage
            progress_cb(lo + (hi - lo) * t, 1.0)

    _check()
    _emit(0.0)
    fixed = cv2.inpaint(roi, roi_mask, radius, method)
    _check()
    _emit(0.60)
    if enhance:
        inner = cv2.erode(roi_mask, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(inner) > 0:
            fixed = cv2.inpaint(fixed, inner, max(2, radius // 2), method)
        _check()
        _emit(0.85)
        # 锐化补偿：inpaint 本质是扩散填充，会让修复区偏平滑，
        # 对蒙版核心区做轻度 USM 锐化，恢复纹理细节（仅作用核心区）
        core = cv2.erode(roi_mask, np.ones((3, 3), np.uint8))
        if cv2.countNonZero(core) > 0:
            blurred = cv2.GaussianBlur(fixed, (0, 0), sigmaX=1.2)
            sharp = np.clip(fixed.astype(np.float32) * 1.6
                            - blurred.astype(np.float32) * 0.6,
                            0, 255).astype(np.uint8)
            s = (core > 0)[:, :, None]
            fixed = np.where(s, sharp, fixed)
    else:
        _emit(0.85)
    # 羽化融合：sigma 随半径加宽，让修复区与周围自然过渡；
    # rint 四舍五入（勿用截断），保证蒙版外 fixed==roi 的像素逐字节不变
    f = cv2.GaussianBlur(roi_mask.astype(np.float32) / 255.0, (0, 0),
                         sigmaX=max(2.0, radius * 1.2))[:, :, None]
    out = roi.astype(np.float32) * (1 - f) + fixed.astype(np.float32) * f
    _check()
    _emit(1.0)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _inpaint_fsr(image, mask, radius=5, progress_cb=None,
                 cancel_check=None):
    """FSR（频率选择性重建）修复：质感远好于 TELEA/NS 扩散，
    接近 AI 且无需模型，但速度慢（逐通道重建），仅建议照片使用。
    注意 xphoto 接口只支持单通道、且 mask 语义与 cv2.inpaint 相反
    （0 = 待修复区）。
    radius：修复强度，决定蒙版膨胀范围（修正：不再写死为 5）；
    progress_cb：按 RGB 三通道的上报进度；cancel_check：通道间检测取消。"""
    if not hasattr(cv2, "xphoto"):
        raise ValueError("当前 OpenCV 不含 FSR 模块（需要 opencv-contrib）")
    m = _prepare_mask(mask, radius)
    bbox = mask_bbox(m, pad=12)
    if bbox is None:
        raise ValueError("选区为空")
    x1, y1, x2, y2 = bbox
    roi = image[y1:y2, x1:x2]
    mroi = m[y1:y2, x1:x2]
    known = (255 - mroi)                 # xphoto: 非 0 = 已知像素
    out = roi.copy()
    for c in range(3):
        if cancel_check is not None and cancel_check():
            raise ProcessingCancelled("处理已取消")
        dst = np.zeros_like(roi[:, :, c])
        cv2.xphoto.inpaint(roi[:, :, c], known, dst,
                           cv2.xphoto.INPAINT_FSR_BEST)
        out[:, :, c] = dst
        if progress_cb is not None:
            progress_cb((c + 1) * 0.85 / 3.0, 1.0)
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("处理已取消")
    # 羽化融合，避免与周围接缝
    f = cv2.GaussianBlur(mroi.astype(np.float32) / 255.0, (0, 0), 2.0)
    f = f[:, :, None]
    blend = roi.astype(np.float32) * (1 - f) + out.astype(np.float32) * f
    result = image.copy()
    result[y1:y2, x1:x2] = np.clip(np.rint(blend), 0, 255).astype(np.uint8)
    if progress_cb is not None:
        progress_cb(1.0, 1.0)
    return result


def inpaint_image(image, mask, radius=5, method=INPAINT_NS, enhance=True,
                  engine="classic", progress_cb=None, cancel_check=None):
    """对 image 中 mask 标记的区域做图像修复，返回新图像。
    engine: "classic" 经典 TELEA/NS 扩散修复（快）；
            "ai"     LaMa AI 修复（纹理/结构还原，最清晰）；
            "fsr"    FSR 频率选择性重建（质感佳、无需模型，较慢）
    progress_cb(done, total)：经典/FSR 引擎的阶段进度；
    cancel_check()：返回 True 时中止并抛 ProcessingCancelled。"""
    if mask is None or cv2.countNonZero(mask) == 0:
        raise ValueError("请先用矩形或画笔选中要去除的水印区域")
    mask = clean_mask(mask)
    if engine == "ai":
        from core import lama
        return lama.inpaint(image, mask, progress_cb=progress_cb,
                            cancel_check=cancel_check)
    if engine == "fsr":
        return _inpaint_fsr(image, mask, radius, progress_cb, cancel_check)
    m = _prepare_mask(mask, radius)
    bbox = mask_bbox(m, pad=radius * 2 + 3)
    if bbox is None:
        raise ValueError("选区为空")
    x1, y1, x2, y2 = bbox
    result = image.copy()
    result[y1:y2, x1:x2] = _fix_region(
        result[y1:y2, x1:x2], m[y1:y2, x1:x2], radius, method, enhance,
        progress_cb=progress_cb, cancel_check=cancel_check)
    return result


# ----------------------------------------------------------------------
# 视频处理
# ----------------------------------------------------------------------
class VideoPreview:
    """视频预览解码器：支持 -ss 快速定位任意帧 + 顺序播放，兼容中文路径。
    max_w：预览降采样宽度（None=原始分辨率）——播放预览时按小尺寸解码，
    显著降低解码/拷贝开销；直接输出 BGR raw 帧，省去一次色彩转换。"""

    def __init__(self, path, max_w=None):
        gen = imageio_ffmpeg.read_frames(str(path))
        meta = next(gen)
        gen.close()
        self.path = str(path)
        self.fps = meta.get("fps") or 25.0
        self.duration = meta.get("duration") or 0
        self.size = meta["size"]
        w, h = self.size
        if max_w and w > max_w:
            nw = int(max_w) // 2 * 2
            nh = int(h * nw / w) // 2 * 2
            self.out_size = (nw, nh)
        else:
            self.out_size = (w, h)
        self._proc = None
        self.pos = 0.0

    def close(self):
        if self._proc is not None:
            for s in (self._proc.stdout, self._proc.stderr):
                try:
                    if s:
                        s.close()
                except Exception:
                    pass
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def _open(self, t):
        self.close()
        ow, oh = self.out_size
        cmd = [FFMPEG_EXE, "-nostdin", "-loglevel", "error"]
        if t > 0.05:
            cmd += ["-ss", f"{max(0.0, t):.3f}"]
        cmd += ["-i", _safe_cli_path(self.path)]
        if self.out_size != self.size:
            cmd += ["-vf", f"scale={ow}:{oh}"]
        cmd += ["-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            **nowin_kwargs())
        self.pos = max(0.0, t)

    def _read_frame(self):
        if self._proc is None:
            return None
        ow, oh = self.out_size
        n = ow * oh * 3
        try:
            raw = self._proc.stdout.read(n)
        except Exception:
            return None
        if raw is None or len(raw) < n:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((oh, ow, 3))

    def frame_at(self, t):
        """定位到指定时间（秒）并返回该帧 BGR"""
        self._open(t)
        return self._read_frame()

    def frame_at_seq(self, t, skip=0):
        """从 t 秒开始顺序解码，丢弃 skip 帧后返回下一帧（导出用）"""
        self._open(t)
        frame = None
        for _ in range(skip + 1):
            frame = self._read_frame()
            if frame is None:
                break
        return frame

    def next_frame(self):
        """顺序读取下一帧（用于播放），到结尾返回 None。
        约定：解码流中下一个待读帧对应时刻 self.pos"""
        if self._proc is None:
            self._open(self.pos)
        f = self._read_frame()
        if f is None:
            return None
        self.pos += 1.0 / self.fps
        return f


def _build_mask_timeline(segments):
    """把 [{start,end,mask}] 段落列表合并为按帧区间生效的时间线
    返回 [(start_idx, end_idx_exclusive, union_mask_or_None)]"""
    if not segments:
        return [(0, float("inf"), None)]
    bounds = sorted({0} | {max(0, int(s["start"])) for s in segments}
                    | {int(s["end"]) + 1 for s in segments})
    out = []
    for i, b in enumerate(bounds):
        nxt = bounds[i + 1] if i + 1 < len(bounds) else float("inf")
        union = None
        for s in segments:
            if s["start"] <= b <= s["end"]:
                union = s["mask"] if union is None else np.maximum(union, s["mask"])
        out.append((b, nxt, union))
    return out


def _pair_shift(g_a, g_b, max_abs=48):
    """估计整体平移 s：g_a(x) ≈ g_b(x + s)（亚像素）。
    相位相关出候选，但其符号约定与实际采样方向容易搞反，因此
    正负两个候选都用 SSD 实测校验，邻域细扫后抛物线内插到亚像素。
    平滑纹理上纯 Farneback 金字塔常收敛到 0，此法可稳定锁定。"""
    try:
        (dx, dy), resp = cv2.phaseCorrelate(
            g_a.astype(np.float64), g_b.astype(np.float64))
    except Exception:
        return 0.0, 0.0
    if resp < 0.05 or not (np.isfinite(dx) and np.isfinite(dy)):
        return 0.0, 0.0

    def ssd(sx, sy):
        return float(np.mean(np.abs(
            g_a - np.roll(np.roll(g_b, -sy, axis=0), -sx, axis=1))))

    best = None
    for cx, cy in ((dx, dy), (-dx, -dy)):
        ix = int(round(max(-max_abs, min(max_abs, cx))))
        iy = int(round(max(-max_abs, min(max_abs, cy))))
        for sy_ in range(iy - 1, iy + 2):
            for sx_ in range(ix - 1, ix + 2):
                r = ssd(sx_, sy_)
                if best is None or r < best[0]:
                    best = (r, sx_, sy_)
    r0, tx, ty = best

    def sub(cm, c0, cp):
        den = cm - 2 * c0 + cp
        if den <= 1e-9:
            return 0.0
        return float(max(-1.0, min(1.0, 0.5 * (cm - cp) / den)))

    fx = tx + sub(ssd(tx - 1, ty), r0, ssd(tx + 1, ty))
    fy = ty + sub(ssd(tx, ty - 1), r0, ssd(tx, ty + 1))
    return fx, fy


# 半透明水印标定参数（与 _temporal 的对齐阈值保持一致）
_CAL_PAD = 48             # 对齐用的上下文边距
_CAL_OFFS = (2, -2, 4, -4)
_CAL_GATE = 14.0          # 门控环亮度一致性上限


def calibrate_alpha(path, mask, start_f, end_f, fps, samples=12):
    """在 [start_f, end_f] 帧范围内稀疏抽样，为静止半透明水印做
    像素级标定：I = α·W + (1-α)·B，其中 (α, W) 时间上恒定，
    B 为光流对齐取回的真实背景，逐像素线性回归即可解出 α 与 W。
    返回 dict(alpha, wm, valid, bbox, fallback_spec, alpha_med)；
    不可标定时抛 ValueError（信息可直接展示给用户）。"""
    if mask is None or cv2.countNonZero(mask) == 0:
        raise ValueError("选区为空")
    stack_I, stack_P, stack_N, stack_W = [], [], [], []
    fh, fw = mask.shape[:2]
    pv = VideoPreview(path)
    try:
        if pv.duration <= 0:
            raise ValueError("无法读取视频时长")
        total = int(pv.duration * fps)
        s0 = max(0, int(start_f))
        span = min(int(end_f), total - 1) - s0
        if span < 8:
            raise ValueError("选区帧范围太短，无法完成透明度标定\n"
                             "请把帧范围设置得更长一些")
        n = max(6, min(samples, span // 2))
        ys, xs = np.where(mask > 0)
        mx1, my1 = int(xs.min()), int(ys.min())
        mx2, my2 = int(xs.max()) + 1, int(ys.max()) + 1
        # 回归只需蒙版附近；对齐需要上下文，工作框外扩 _CAL_PAD
        px1, py1 = max(0, mx1 - _CAL_PAD), max(0, my1 - _CAL_PAD)
        px2, py2 = min(fw, mx2 + _CAL_PAD), min(fh, my2 + _CAL_PAD)
        rm = cv2.dilate(mask[py1:py2, px1:px2],
                        np.ones((3, 3), np.uint8), iterations=1)
        excl = cv2.dilate(rm, np.ones((5, 5), np.uint8), iterations=1)
        ring = cv2.dilate(rm, np.ones((9, 9), np.uint8),
                          iterations=1) & ~(rm > 0)
        cmask = mask[py1:py2, px1:px2] > 0
        h, w = rm.shape[:2]
        s = min(1.0, 384.0 / max(h, w))
        s = max(s, min(1.0, 64.0 / min(h, w)))
        if s < 1.0:
            sw_, sh_ = max(2, round(w * s)), max(2, round(h * s))
        else:
            sw_, sh_ = w, h
        xs_g, ys_g = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
        xs_s, ys_s = np.meshgrid(np.arange(sw_, dtype=np.float32),
                                 np.arange(sh_, dtype=np.float32))
        kx, ky = w / sw_, h / sh_
        for i in range(n):
            tf = s0 + int(span * (i + 0.5) / n)
            cur = pv.frame_at(tf / fps)
            if cur is None:
                continue
            cur = cur[py1:py2, px1:px2]
            Bp_acc = np.zeros((h, w, 3), np.float32)
            Vp_acc = np.zeros((h, w), np.float32)
            Bn_acc = np.zeros((h, w, 3), np.float32)
            Vn_acc = np.zeros((h, w), np.float32)
            g_cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            # 水印是画面里最大的静止强结构，先填充再算流，否则流场会
            # 被拽向零/伪匹配。正组用 NS、负组用 TELEA：两侧的对齐
            # 误差因此相互独立（IV 回归无偏性的前提）
            fill_ns = cv2.inpaint(g_cur, rm, 3, cv2.INPAINT_NS)
            g_cur_ns = np.where(rm > 0, fill_ns, g_cur).astype(np.uint8)
            fill_te = cv2.inpaint(g_cur, rm, 5, cv2.INPAINT_TELEA)
            g_cur_te = np.where(rm > 0, fill_te, g_cur).astype(np.uint8)
            if s < 1.0:
                g_cur_ns = cv2.resize(g_cur_ns, (sw_, sh_),
                                      interpolation=cv2.INTER_AREA)
                g_cur_te = cv2.resize(g_cur_te, (sw_, sh_),
                                      interpolation=cv2.INTER_AREA)
            for off in _CAL_OFFS:
                nb = pv.frame_at((tf + off) / fps)
                if nb is None:
                    continue
                nb = nb[py1:py2, px1:px2]
                # 两级对齐（与 process_video._temporal 相同的流程）
                g_nb = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY)
                g_nb = np.where(
                    rm > 0, cv2.inpaint(g_nb, rm, 3, cv2.INPAINT_NS),
                    g_nb).astype(np.uint8)
                if off > 0:
                    g_cur_s = g_cur_ns
                else:
                    g_cur_s = g_cur_te
                if s < 1.0:
                    g_nb_s = cv2.resize(g_nb, (sw_, sh_),
                                        interpolation=cv2.INTER_AREA)
                else:
                    g_nb_s = g_nb
                sfx, sfy = _pair_shift(g_cur_s, g_nb_s)
                if abs(sfx) > 48 or abs(sfy) > 48:
                    continue
                nb_al = cv2.remap(g_nb_s, xs_s + sfx, ys_s + sfy,
                                  cv2.INTER_LINEAR)
                flow = cv2.calcOpticalFlowFarneback(
                    g_cur_s, nb_al, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                if s < 1.0:
                    flow = cv2.resize(flow, (w, h),
                                      interpolation=cv2.INTER_LINEAR)
                sx = np.clip(xs_g + (flow[:, :, 0] + sfx) * kx,
                             0.0, w - 1.0)
                sy = np.clip(ys_g + (flow[:, :, 1] + sfy) * ky,
                             0.0, h - 1.0)
                # 源点落回水印内的样本无效（取到的是水印而非背景）
                ok = excl[sy.astype(np.int32), sx.astype(np.int32)] == 0
                if int(ok.sum()) < 30:
                    continue
                warped = cv2.remap(nb, sx, sy, cv2.INTER_LINEAR)
                if int(ring.sum()) >= 100:
                    err = float(np.abs(
                        warped[ring].astype(np.float32)
                        - cur[ring].astype(np.float32)).mean())
                    if err > _CAL_GATE:
                        continue
                if off > 0:
                    Bp_acc += warped * ok[:, :, None]
                    Vp_acc += ok
                else:
                    Bn_acc += warped * ok[:, :, None]
                    Vn_acc += ok
            # 正负偏移各至少一个有效样本，才构成一组独立估计对
            if not ((Vp_acc >= 1) & (Vn_acc >= 1)).any():
                continue
            stack_I.append(cur.astype(np.float32))
            stack_P.append(Bp_acc / np.maximum(Vp_acc, 1.0)[:, :, None])
            stack_N.append(Bn_acc / np.maximum(Vn_acc, 1.0)[:, :, None])
            stack_W.append(((Vp_acc >= 1) & (Vn_acc >= 1)).astype(np.float32))
    finally:
        pv.close()
    if len(stack_I) < 6:
        raise ValueError("有效采样帧不足，无法标定（视频可能过短或解码失败）")

    # 逐像素回归：I = αW + (1-α)B。B̃ 由光流对齐的重采样得到，
    # 携带编码量化噪声，直接 OLS 斜率会被衰减（α 系统性偏高，
    # 实测 +0.1 量级）。var(B̃₊-B̃₋) 给出两侧噪声方差的独立可测
    # 部分，据此做误差变量（errors-in-variables）校正：
    #   m = raw · var(B̃) / (var(B̃) - varε)
    # 校正可覆盖大部分衰减；重压缩片源可能仍有少量残余偏高。
    I = np.stack(stack_I)            # (S,h,w,3)
    Bp = np.stack(stack_P)
    Bn = np.stack(stack_N)
    Wt = np.stack(stack_W)           # (S,h,w) 两侧同时有效的像素
    S = I.shape[0]
    Bmean = (Bp + Bn) / 2.0
    base = Wt.sum(axis=0) >= max(4.0, S * 0.4)
    Kn = np.maximum(Wt.sum(axis=0), 1.0)[:, :, None]
    Bbar = (Bmean * Wt[:, :, :, None]).sum(0) / Kn
    Ibar = (I * Wt[:, :, :, None]).sum(0) / Kn
    varBm = ((Bmean - Bbar) ** 2 * Wt[:, :, :, None]).sum(0) / Kn
    varD = (((Bp - Bn) ** 2) * Wt[:, :, :, None]).sum(0) / Kn
    varE = varD / 2.0                # 单侧 B̃ 噪声方差的独立可测部分
    var_ok = varBm.mean(axis=2) >= 8.0
    raw = ((I - Ibar) * (Bmean - Bbar) * Wt[:, :, :, None]).sum(0) / Kn \
        / np.maximum(varBm, 1e-3)
    m = np.clip(raw * np.maximum(varBm, 1e-3)
                / np.maximum(varBm - varE / 2.0, 1e-3), 0.02, 1.0)
    alpha = 1.0 - np.median(m, axis=2)
    valid = base & var_ok & (alpha > 0.04) & (alpha < 0.97)
    frac = float(valid[cmask].mean()) if cmask.any() else 0.0
    a_med = (float(np.median(alpha[valid & cmask]))
             if (valid & cmask).any() else 0.0)
    if frac < 0.4 or a_med > 0.85:
        if not var_ok[cmask].any() or frac < 0.05:
            raise ValueError("背景几乎静止，透明度无法标定，请改用 AI 修复")
        if a_med > 0.85:
            raise ValueError(f"检测到水印接近不透明（α≈{a_med:.2f}），"
                             "透明还原不适用，请改用 AI 修复")
        raise ValueError(f"可标定像素占比过低（{frac:.0%}），"
                         "请改用 AI 修复")
    # 中值滤波抑制孤立误标
    a8 = cv2.medianBlur(np.clip(alpha * 100, 0, 100).astype(np.uint8), 5)
    alpha = a8.astype(np.float32) / 100.0
    Kn = np.maximum(Wt.sum(axis=0), 1.0)[:, :, None]
    Ibar = (I * Wt[:, :, :, None]).sum(0) / Kn
    Bbar = (Bmean * Wt[:, :, :, None]).sum(0) / Kn
    wm = np.clip((Ibar - m * Bbar)
                 / np.maximum(alpha, 0.04)[:, :, None], 0, 255)
    # 标定失败的像素（不透明核/样本不足）由经典修复兜底
    fb_mask = (cmask & ~valid).astype(np.uint8) * 255
    fb_mask = cv2.dilate(fb_mask, np.ones((3, 3), np.uint8), iterations=1)
    fb_spec = None
    if cv2.countNonZero(fb_mask) > 0:
        bbox = mask_bbox(fb_mask, pad=8)
        if bbox is not None:
            fx1, fy1, fx2, fy2 = bbox
            fb_spec = (px1 + fx1, py1 + fy1, px1 + fx2, py1 + fy2,
                       np.ascontiguousarray(fb_mask[fy1:fy2, fx1:fx2]))
    return {"bbox": (px1, py1, px2, py2), "alpha": alpha, "wm": wm,
            "valid": valid & cmask, "fallback_spec": fb_spec,
            "alpha_med": a_med}


def process_video(input_path, output_path, segments=None, radius=5,
                  method=INPAINT_NS, progress_cb=None, keep_audio=True,
                  enhance=True, engine="classic", cancel_check=None):
    """
    逐帧处理视频：
      FFmpeg 解码 -> 按帧区间对选区做 AI/经典修复 -> 硬件加速编码
      -> 回填原音轨（可选）
    segments: [{"start":帧, "end":帧, "mask":ndarray}]，空则不修复
    engine: "ai" LaMa AI 修复（默认最清晰）；"classic" 经典扩散修复（快）
    enhance: 经典模式的增强修复（二遍+羽化）
    progress_cb(done, total)
    cancel_check: 无参可调用，返回 True 时中止导出并抛
                  VideoExportCancelled（临时文件自动清理）
    """
    from concurrent.futures import ThreadPoolExecutor
    if not segments:
        raise ValueError("请先添加要去除的水印/字幕选区")

    if engine == "ai":
        from core import lama
        if not lama.is_ready():
            raise RuntimeError("AI 模型不可用，请改用经典算法")

    # 帧尺寸以第一个蒙版为准（蒙版均为整帧大小）
    fh, fw = segments[0]["mask"].shape[:2]

    # 预计算每个帧区间的修复参数（alpha 引擎走独立管线，无需准备）：
    #   经典：蒙版外接小矩形 + 局部蒙版
    #   AI：  上下文 ROI 裁剪框 + 膨胀 1px 的局部蒙版（送模型前不再膨胀，
    #         避免 AI 模式把好画面也吃进蒙版）
    prepared = []
    if engine != "alpha":
        for a, b, union in _build_mask_timeline(segments or []):
            spec = None
            if union is not None and cv2.countNonZero(union) > 0:
                clean_mask(union)
                if engine == "ai":
                    # 上下文给足（pad 1.0 / 最少 48px）：LaMa 的全局感受野
                    # 依赖周围结构，上下文越大对纹理/结构的还原越真实
                    x1, y1, x2, y2 = lama.roi_box(union, (fh, fw),
                                                  pad_ratio=1.0, min_pad=48)
                    rm = cv2.dilate(union[y1:y2, x1:x2],
                                    np.ones((3, 3), np.uint8), iterations=1)
                    rm = np.ascontiguousarray(rm)
                    # 复用缓存时只把填充结果透过蒙版（羽化）贴回：
                    # 整块 ROI 直接覆盖会把旧帧背景盖上来，运动画面出现"冻结补丁"
                    alpha = cv2.GaussianBlur(
                        (rm > 0).astype(np.float32), (0, 0), 1.0)[:, :, None]
                    # 时域重建辅助区：
                    #   excl 排除区（蒙版再膨胀）——光流采样落点若仍在水印内，
                    #        取到的是水印像素而非背景，必须丢弃；
                    #   ring  门控环（蒙版外一圈）——评估邻帧对齐质量
                    m5 = np.ones((5, 5), np.uint8)
                    excl = cv2.dilate(rm, m5, iterations=1)
                    ring = cv2.dilate(rm, np.ones((9, 9), np.uint8),
                                      iterations=1) & ~(rm > 0)
                    spec = (x1, y1, x2, y2, rm, alpha, excl, ring)
                else:
                    m = _prepare_mask(union, radius)
                    bbox = mask_bbox(m, pad=radius * 2 + 3)
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        spec = (x1, y1, x2, y2,
                                np.ascontiguousarray(m[y1:y2, x1:x2]))
            prepared.append([a, b, spec])
    _starts = [p[0] for p in prepared]

    # ----------------------------------------------------------------------
    # AI 时域多帧重建：静态水印遮挡住的背景会随画面运动，在相邻帧的
    # 水印区域之外"露出来"。用光流把历史帧对齐回当前帧，从蒙版外的
    # 落点取回真实背景像素。与 AI 生成相比，这是真实画面而非推测：
    #   - 运动充分时（位移 > 水印半宽）可整帧跳过 AI 推理，速度大增
    #   - 运动不足时作为 AI 结果的边缘补强（AI 接缝最显眼处换上真像素）
    # ----------------------------------------------------------------------
    _buf = deque(maxlen=10)          # (帧号, 全局裁剪框内的小帧拷贝)
    _buf_lock = threading.Lock()
    # 取 idx±N 的邻帧：正向与负向都要——单向运动只会从水印带的一侧
    # 露出真实背景，对称取帧才能覆盖整个蒙版
    _T_OFFS = (2, -2, 4, -4)
    _T_GATE = 14.0                   # 邻帧门控环亮度一致性上限
    _T_STD = 9.0                     # 多帧样本一致性（噪声级）上限
    _T_SKIP_AI = 0.92                # 置信覆盖率达到则整帧跳过 AI
    _T_MIN_COV = 0.5                 # 低于则放弃时域结果

    # 全局裁剪框：所有 AI spec 外接矩形的并集。历史帧只存这个小框，
    # 1080P 下约 1MB/帧，避免整帧缓存（4K 一帧就要 24MB）
    _gbbox = None
    if engine == "ai":
        boxes = [p[2][:4] for p in prepared if p[2] is not None]
        if boxes:
            _gbbox = (max(0, min(b[0] for b in boxes)),
                      max(0, min(b[1] for b in boxes)),
                      min(fw, max(b[2] for b in boxes)),
                      min(fh, max(b[3] for b in boxes)))

    def _temporal(roi, rm, excl, ring, box, idx):
        """从历史帧光流对齐重建 ROI 的水印区背景。
        返回 (填充图, 置信掩码, 蒙版内置信覆盖率)；不可用返回 None。"""
        with _buf_lock:
            look = {j - idx: fr for j, fr in _buf
                    if idx - j in _T_OFFS}
        if not look or _gbbox is None:
            return None
        gx1, gy1, gx2, gy2 = _gbbox
        bx1, by1, bx2, by2 = box
        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 水印区内无有效内容用于算流：用 NS 修复沿等照度线把周围结构
        # 延展进蒙版（保留边缘方向；实验对比：比高斯模糊填充的光流
        # 误差小约 6 倍，模糊填充在平坦区几乎算不出流）
        fillg = cv2.inpaint(gray, rm, 3, cv2.INPAINT_NS)
        g_cur = np.where(rm > 0, fillg, gray).astype(np.uint8)
        # 光流计算分辨率：限制最大边（提速），同时保最小边（窄长字幕
        # 条缩太小会让 winsize=15 的窗口失去意义）
        s = min(1.0, 384.0 / max(h, w))
        s = max(s, min(1.0, 64.0 / min(h, w)))
        if s < 1.0:
            sw, sh = max(2, round(w * s)), max(2, round(h * s))
            g_cur_s = cv2.resize(g_cur, (sw, sh),
                                 interpolation=cv2.INTER_AREA)
        else:
            sw, sh = w, h
            g_cur_s = g_cur
        acc = np.zeros((h, w, 3), np.float32)
        sq = np.zeros((h, w, 3), np.float32)
        cnt = np.zeros((h, w), np.float32)
        xs_g, ys_g = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
        xs_s, ys_s = np.meshgrid(np.arange(sw, dtype=np.float32),
                                 np.arange(sh, dtype=np.float32))
        hclip, wclip = h - 1.0, w - 1.0
        kx, ky = w / sw, h / sh
        for fr in look.values():
            nb = fr[by1 - gy1:by2 - gy1, bx1 - gx1:bx2 - gx1]
            g_nb = cv2.cvtColor(nb, cv2.COLOR_BGR2GRAY)
            # 邻帧的水印带必须同样做 NS 延展填充后再参与算流：
            # 原水印是画面里最大的"静止强结构"，会把蒙版内的流场
            # 拽向零/周期混叠伪匹配（采样仍用邻帧原始像素）
            g_nb = np.where(rm > 0, cv2.inpaint(g_nb, rm, 3, cv2.INPAINT_NS),
                            g_nb).astype(np.uint8)
            if s < 1.0:
                g_nb_s = cv2.resize(g_nb, (sw, sh),
                                    interpolation=cv2.INTER_AREA)
            else:
                g_nb_s = g_nb
            # 两级对齐：整体平移预对齐后残余位移趋近 0，
            # Farneback 只负责局部运动的细微修正
            sfx, sfy = _pair_shift(g_cur_s, g_nb_s)
            if abs(sfx) > 48 or abs(sfy) > 48:
                continue          # 跳切级突变：该邻帧不可用
            nb_al = cv2.remap(g_nb_s, xs_s + sfx, ys_s + sfy,
                              cv2.INTER_LINEAR)
            flow = cv2.calcOpticalFlowFarneback(
                g_cur_s, nb_al, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            if s < 1.0:
                flow = cv2.resize(flow, (w, h),
                                  interpolation=cv2.INTER_LINEAR)
            sx = np.clip(xs_g + (flow[:, :, 0] + sfx) * kx, 0.0, wclip)
            sy = np.clip(ys_g + (flow[:, :, 1] + sfy) * ky, 0.0, hclip)
            # 落点仍在外包水印内的样本无效（取到的是水印像素）
            ok = excl[sy.astype(np.int32), sx.astype(np.int32)] == 0
            if int(ok.sum()) < 30:
                continue
            warped = cv2.remap(nb, sx, sy, cv2.INTER_LINEAR)
            # 门控：蒙版外一圈上对齐误差过大（光流失效/光照突变）则弃用
            if int(ring.sum()) >= 100:
                err = float(np.abs(
                    warped[ring].astype(np.float32)
                    - roi[ring].astype(np.float32)).mean())
                if err > _T_GATE:
                    continue
            v3 = ok[:, :, None]
            wf = warped.astype(np.float32)
            acc += np.where(v3, wf, 0.0)
            sq += np.where(v3, wf * wf, 0.0)
            cnt += ok
        conf = cnt >= 2
        if not conf.any():
            return None
        mean = acc / np.maximum(cnt, 1.0)[:, :, None]
        var = np.maximum(sq / np.maximum(cnt, 1.0)[:, :, None]
                         - mean * mean, 0.0)
        conf &= np.sqrt(var.mean(axis=2)) <= _T_STD
        conf &= rm > 0
        if not conf.any():
            return None
        cov = float(cv2.countNonZero(conf.astype(np.uint8))
                    / max(1, cv2.countNonZero(rm)))
        if cov < _T_MIN_COV:
            return None
        fill = np.clip(np.rint(mean), 0, 255).astype(np.uint8)
        fill[rm == 0] = roi[rm == 0]     # 蒙版外一律用当前帧原像素
        return fill, conf, cov

    # AI 静态场景帧复用：水印不动、背后画面也不动时（片头/静态镜头常见），
    # 上下文区域与上一帧几乎一致，直接复用上一帧的修复结果，节省整次推理。
    # 迟滞双阈值：漂移 < _REUSE_ENTER 才进入复用态，进入后要 > _REUSE_EXIT
    # 才退出。单一阈值时，带噪声的缓慢漂移会让判定在阈值附近反复横跳，
    # 「复用旧填充 / 重新推理新填充」交替出现，补丁内容随之跳变（闪烁）。
    _reuse_lock = threading.Lock()
    _reuse = {}                      # i -> (基准sig, 填充结果, 复用态)
    _REUSE_ENTER = 1.2
    _REUSE_EXIT = 2.8

    def _context_sig(roi):
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA
                          ).astype(np.float32)

    def _fix_alpha(frame, idx):
        """半透明水印逐帧反解：B = (I - αW) / (1 - α)。
        只替换标定有效且 α 足够大的像素；标定失败区（不透明核等）
        由经典修复兜底。"""
        for a0, b0, cal in alpha_segs:
            if idx < a0 or idx > b0:
                continue
            x1, y1, x2, y2 = cal["bbox"]
            roi = frame[y1:y2, x1:x2]
            al = cal["alpha"][:, :, None]
            inv = np.maximum(1.0 - al, 0.03)
            rec = (roi.astype(np.float32) - al * cal["wm"]) / inv
            m = (cal["valid"] & (cal["alpha"] >= 0.03))[:, :, None]
            roi[:] = np.where(m, np.clip(np.rint(rec), 0, 255),
                              roi).astype(np.uint8)
            fb = cal["fallback_spec"]
            if fb is not None:
                fx1, fy1, fx2, fy2, frm = fb
                if cv2.countNonZero(frm) > 0:
                    frame[fy1:fy2, fx1:fx2] = _fix_region(
                        frame[fy1:fy2, fx1:fx2], frm, 5,
                        INPAINT_TELEA, True)
        return frame

    def _fix(frame, idx):
        # 二分定位当前帧所属区间（线程安全，不修改共享状态）
        if engine == "alpha":
            return _fix_alpha(frame, idx)
        i = bisect.bisect_right(_starts, idx) - 1
        spec = prepared[i][2] if i >= 0 else None
        if spec is not None and engine == "ai":
            x1, y1, x2, y2, rm, alpha, excl, ring = spec
            box = (x1, y1, x2, y2)
            roi = frame[y1:y2, x1:x2]
            # 复用签名看 ROI 整体：水印条带内部被静态水印占据，
            # 拿它当签名的话，背景动没动根本测不出来（运动场景
            # 会被误判为静态，一直复用首帧填充 → 冻结补丁）
            sig = _context_sig(roi)
            with _reuse_lock:
                rec = _reuse.get(i)
            thr = _REUSE_EXIT if (rec is not None and rec[2]) else _REUSE_ENTER
            if rec is not None and np.abs(sig - rec[0]).mean() < thr:
                # 静态场景：仅水印蒙版区域复用上一帧填充结果（羽化融合），
                # 蒙版外的当前帧画面原样保留
                roi[:] = np.clip(
                    np.rint(roi.astype(np.float32) * (1 - alpha)
                            + rec[1].astype(np.float32) * alpha),
                    0, 255).astype(np.uint8)
                with _reuse_lock:
                    _reuse[i] = (rec[0], rec[1], True)   # 基准不变，置复用态
                return frame
            # 时域重建：从历史帧取回被水印遮挡的真实背景
            tres = _temporal(roi, rm, excl, ring, box, idx)
            if tres is not None:
                fill, conf, cov = tres
                if cov >= _T_SKIP_AI:
                    # 运动充分：真实背景已覆盖几乎所有蒙版区，
                    # 少量孔洞用局部扩散补齐即可，整帧跳过 AI 推理
                    holes = ((rm > 0) & ~conf).astype(np.uint8) * 255
                    if cv2.countNonZero(holes) > 0:
                        fill = cv2.inpaint(fill, holes, 4, cv2.INPAINT_TELEA)
                    roi[:] = np.clip(
                        np.rint(roi.astype(np.float32) * (1 - alpha)
                                + fill.astype(np.float32) * alpha),
                        0, 255).astype(np.uint8)
                    with _reuse_lock:
                        _reuse[i] = (sig, fill, True)
                    return frame
            fixed = lama.infer_tiled(roi, rm)
            if tres is not None:
                # 覆盖不足：AI 兜底，但置信的真实像素盖过 AI 结果
                # （蒙版边缘的 AI 接缝最显眼，换成真像素最有效）
                a2 = cv2.GaussianBlur(conf.astype(np.float32),
                                      (0, 0), 1.2)[:, :, None]
                fixed = np.clip(np.rint(
                    fixed.astype(np.float32) * (1 - a2)
                    + fill.astype(np.float32) * a2), 0, 255).astype(np.uint8)
            with _reuse_lock:
                _reuse[i] = (sig, fixed, False)
            frame[y1:y2, x1:x2] = fixed
        elif spec is not None:
            x1, y1, x2, y2, rm = spec
            # cv2.inpaint 在 C++ 层释放 GIL，多线程可真正并行
            frame[y1:y2, x1:x2] = _fix_region(
                frame[y1:y2, x1:x2], rm, radius, method, enhance)
        return frame

    # AI 推理一次即可吃满 CPU：单 worker 让主线程解码与推理并行，
    # 多 worker 只会争抢核心与 GIL 拖慢整体速度；
    # alpha 反解是纯 numpy（GIL 内），2 worker 够用；
    # 经典修复在 C++ 层释放 GIL，多线程是真并行
    if engine == "ai":
        workers = 1
    elif engine == "alpha":
        workers = 2
    else:
        workers = min(os.cpu_count() or 4, 16)

    reader = imageio_ffmpeg.read_frames(str(input_path))
    meta = next(reader)
    w, h = meta["size"]
    fps = meta.get("fps") or 25.0
    duration = meta.get("duration", 0) or 0
    total = int(duration * fps) if duration > 0 else 0

    # 半透明水印：解码前逐段标定 α 与水印色（随机访问抽帧，可能较慢）
    alpha_segs = []
    if engine == "alpha":
        for seg in segments:
            cal = calibrate_alpha(input_path, seg["mask"],
                                  seg["start"], seg["end"], fps)
            alpha_segs.append((max(0, seg["start"]),
                               min(seg["end"], total - 1), cal))

    # 临时文件用唯一名放在系统 temp 目录：既不在用户文件夹残留 .noaudio
    # 文件，也避免固定名导致的实例间冲突 / 本地预创建（符号链接）风险
    tmp_path = _temp_path(".noaudio.mp4", "zhengjing_export_")
    codec, extra, _name = detect_encoder()
    writer = imageio_ffmpeg.write_frames(
        tmp_path, (w, h), fps=fps, codec=codec, macro_block_size=1,
        quality=None,   # 质量由 output_params 控制，避免自动附加 -qscale:v
        output_params=extra,
    )
    writer.send(None)  # 初始化编码器

    done = 0
    try:
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = []
                idx = 0
                # 前瞻窗口：AI 时域重建需要「未来帧」从水印带另一侧
                # 采样真实背景（单向运动只露出一侧），故延迟 _look 帧
                # 提交修复；经典模式 _look=0，行为与逐帧处理完全一致
                _look = 4 if (engine == "ai" and _gbbox is not None) else 0
                window = deque()
                for raw in reader:
                    if cancel_check is not None and cancel_check():
                        raise VideoExportCancelled()
                    # frombuffer 得到的是只读视图，拷贝为可写数组以便回写
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
                    if _gbbox is not None:
                        # 修复前存档：时域重建要用未修复的邻帧
                        bx0, by0, bx1_, by1_ = _gbbox
                        with _buf_lock:
                            _buf.append((idx, frame[by0:by1_, bx0:bx1_].copy()))
                    window.append((idx, frame))
                    if len(window) > _look:
                        j0, f0 = window.popleft()
                        pending.append(pool.submit(_fix, f0, j0))
                    idx += 1
                    # 控制内存：队列过长时先按顺序写出最早的任务
                    while len(pending) > workers * 2:
                        writer.send(pending.pop(0).result().tobytes())
                        done += 1
                        if progress_cb is not None:
                            progress_cb(done, total)
                # 视频读完：把前瞻窗口里剩余的最后几帧提交修复
                for j0, f0 in window:
                    pending.append(pool.submit(_fix, f0, j0))
                window.clear()
                for fut in pending:
                    if cancel_check is not None and cancel_check():
                        raise VideoExportCancelled()
                    writer.send(fut.result().tobytes())
                    done += 1
                    if progress_cb is not None:
                        progress_cb(done, total)
        except VideoExportCancelled:
            raise
        except Exception as e:
            # 编码中途失败：多半是硬件编码器在启动阶段被 ffmpeg 拒绝
            # （驱动过旧/设备被占用），此时补一次实测把真实原因带上
            hint = ""
            if not _probe_encoder(codec):
                hint = (f"\n编码器 {codec} 实际不可用（显卡驱动过旧或缺少硬件支持），"
                        f"请更新显卡驱动后重启软件；也可稍后再试。")
            raise RuntimeError(f"视频编码失败：{e}{hint}") from e
        finally:
            writer.close()
            reader.close()

        if keep_audio:
            try:
                mux_audio(tmp_path, input_path, output_path)
            except Exception:
                # 回填音轨失败时保留无音频版本
                shutil.move(tmp_path, output_path)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
        else:
            # 临时文件在系统 temp（可能跨盘），shutil.move 兼容跨盘移动
            shutil.move(tmp_path, output_path)
    except VideoExportCancelled:
        _cleanup(tmp_path)
        raise

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"导出失败：未生成输出文件 {output_path}")
    return output_path

def detect_static_mask(path, samples=12, max_w=640):
    """自动检测烧录在画面里的静态水印（硬字幕/台标/角标）：
    均匀抽样做时域分析——「时间轴上不变（时域标准差低）且空间上
    梯度强（文字/图形边缘）」的像素为种子，膨胀合并成完整区域，
    再剔除碎点与超大误检块。返回 (mask 0/255, 覆盖率)。
    镜头整体静止时水印与背景在时域上不可区分，抛 ValueError
    提示改为手动框选（半透明水印同理，边缘特征弱）。"""
    pv = VideoPreview(path, max_w=max_w)
    try:
        if pv.duration <= 0:
            raise ValueError("无法读取视频时长")
        k = max(3, min(samples, int(pv.duration * pv.fps)))
        stack = []
        for i in range(k):
            t = pv.duration * (i + 0.5) / k
            f = pv.frame_at(t)
            if f is None:
                break
            stack.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    finally:
        pv.close()
    if len(stack) < 3:
        raise ValueError("视频过短，无法采样分析")

    arr = np.stack(stack).astype(np.float32)
    tstd = arr.std(axis=0)
    if float(np.percentile(tstd, 90)) < 4.0:
        raise ValueError("画面整体几乎静止，无法区分水印与背景，请手动框选")
    static = tstd < 3.5
    if float(static.mean()) > 0.9:
        raise ValueError("静态区域占比过高（多为静止镜头），请手动框选")

    mean = arr.mean(axis=0)
    gx = cv2.Sobel(mean, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mean, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    gs = grad[static]
    # 编码会模糊边缘，Sobel 响应偏弱：阈值下限取低值，靠后续形态学
    # 与连通域过滤兜底误检
    thr = max(12.0, float(np.percentile(gs, 90)))
    seed = static & (grad > thr)

    # 边缘种子膨胀成完整字形/图标，但不越进「会动」的区域
    k5 = np.ones((5, 5), np.uint8)
    k7 = np.ones((7, 7), np.uint8)
    cand = cv2.dilate(seed.astype(np.uint8), k7, iterations=3) \
        & cv2.dilate(static.astype(np.uint8), k5, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    out = np.zeros_like(cand)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < 48 or a > cand.size * 0.08:   # 碎点噪声 / 大块误检
            continue
        out[labels == i] = 255
    cov = float((out > 0).mean())
    if cov < 0.0005:
        raise ValueError("未检测到明显的静态水印（半透明水印请手动框选）")
    # 轻微膨胀覆盖抗锯齿边缘（UI 侧映射回原分辨率时还会按倍率再膨胀）
    return cv2.dilate(out, k5, iterations=1), cov


def probe_thumbnails(path, count=24, height=64):
    """均匀抽取 count 张缩略图，返回 [(时间秒, BGR小图)]，用于时间线胶片条。
    降采样解码（max_w）抽取，避免与预览播放争抢解码资源"""
    pv = VideoPreview(path, max_w=360)
    out = []
    if pv.duration <= 0:
        return out
    for i in range(count):
        t = pv.duration * i / count
        try:
            frame = pv.frame_at(t)
        except Exception:
            break
        h, w = frame.shape[:2]
        nw = max(1, int(w * height / h))
        out.append((t, cv2.resize(frame, (nw, height),
                                  interpolation=cv2.INTER_AREA)))
    pv.close()
    return out


def mux_audio(video_noaudio, original, output):
    """把原视频的音轨合并到处理后的视频（同时丢弃字幕流）"""
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", _safe_cli_path(video_noaudio),
        "-i", _safe_cli_path(original),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "copy",
        "-movflags", "+faststart",
        _safe_cli_path(output),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="ignore",
                   **nowin_kwargs())
    if proc.returncode != 0 or not os.path.exists(output):
        _log.debug("mux_audio ffmpeg stderr: %s", proc.stderr[-500:])
        raise RuntimeError("合并音轨失败：" + proc.stderr[-500:])


def strip_subtitle_stream(input_path, output_path):
    """直接剥离容器中的软字幕流（画面不重编码，速度极快）"""
    cmd = [
        FFMPEG_EXE, "-y",
        "-i", _safe_cli_path(input_path),
        "-map", "0", "-sn",          # 保留所有流但排除字幕
        "-c", "copy",
        _safe_cli_path(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="ignore",
                   **nowin_kwargs())
    if proc.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError("剥离字幕失败：" + proc.stderr[-500:])
    return output_path


class AudioPlayer:
    """预览音频播放：导入视频时后台把整条音轨解码成 wav，
    播放时从 wav 流复制瞬时切片 + winsound 异步播放——零等待出声。
    零第三方依赖（Windows 标准库 winsound + 自带 ffmpeg）。"""

    def __init__(self):
        self._tmp = None
        self._proc = None
        self._playing = False
        self._full = None          # 整段预解码 wav（唯一临时路径）
        self._full_src = None      # 该 wav 对应的视频路径
        self._lock = threading.Lock()
        # 每个实例独占一段唯一临时路径，避免多实例/多来源互相覆盖
        self._full_path = _temp_path(".wav", "zhengjing_preview_full_")

    @property
    def playing(self):
        return self._playing

    def preload(self, video_path):
        """导入视频后调用：后台整段解码音轨，完成后播放零等待。
        可重复调用（同一路径只解码一次）。"""
        src = str(video_path)
        with self._lock:
            if self._full_src == src and self._full \
                    and os.path.exists(self._full):
                return
            self._full_src = src

        def worker():
            wav = self._full_path
            part = wav + ".part"
            _cleanup(part)
            try:
                proc = subprocess.run(
                    [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
                     "-i", _safe_cli_path(src), "-vn", "-ac", "2",
                     "-ar", "44100", "-f", "wav", _safe_cli_path(part)],
                    capture_output=True, timeout=600, **nowin_kwargs())
                ok = proc.returncode == 0 and os.path.exists(part) \
                    and os.path.getsize(part) > 44
            except Exception:
                _log.debug("预览音轨预解码失败", exc_info=True)
                ok = False
            with self._lock:
                if ok and self._full_src == src:
                    try:
                        os.replace(part, wav)
                        self._full = wav
                    except OSError:
                        _cleanup(part)
                else:
                    _cleanup(part)

        threading.Thread(target=worker, daemon=True).start()

    def play(self, video_path, start_s):
        """从 start_s 秒开始播放音轨（异步，准备好后自动出声）"""
        self.stop()
        try:
            import winsound  # noqa: F401
        except ImportError:
            return          # 非 Windows 平台无音频预览，静默跳过
        src = str(video_path)
        with self._lock:
            full = self._full if (self._full_src == src and self._full
                                  and os.path.exists(self._full)) else None
        self._tmp = _temp_path(".wav", "zhengjing_preview_")
        _cleanup(self._tmp)
        if full is not None:
            # 整段 wav 已就绪：流复制切片，瞬时完成
            cmd = [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", f"{max(0.0, start_s):.3f}", "-i", _safe_cli_path(full),
                   "-c", "copy", _safe_cli_path(self._tmp)]
        else:
            cmd = [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", f"{max(0.0, start_s):.3f}", "-i", _safe_cli_path(src),
                   "-vn", "-ac", "2", "-ar", "44100", "-f", "wav",
                   _safe_cli_path(self._tmp)]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL,
                                      **nowin_kwargs())
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self):
        try:
            self._proc.wait()
        except Exception:
            return
        rc = self._proc.returncode
        self._proc = None
        if rc != 0 or not os.path.exists(self._tmp):
            return
        try:
            import winsound
            winsound.PlaySound(self._tmp,
                               winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._playing = True
        except Exception:
            pass

    def stop(self):
        """停止播放并清理（可重复调用；整段预解码缓存保留）"""
        try:
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
        self._playing = False
        _cleanup(self._tmp)
        self._tmp = None


def _cleanup(*paths):
    """尽力删除临时文件，忽略失败"""
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
