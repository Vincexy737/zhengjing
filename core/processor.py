# -*- coding: utf-8 -*-
"""
核心处理算法：
1. 图片去水印 —— LaMa AI 修复 / FSR 频率选择性重建 / OpenCV inpaint（TELEA / NS）
2. 水印自动检测 —— 形态学 top-hat（照片）与 OCR（文字行）两条通路
"""

import logging
import os

import numpy as np
import cv2

_log = logging.getLogger(__name__)

INPAINT_TELEA = cv2.INPAINT_TELEA
INPAINT_NS = cv2.INPAINT_NS


class ProcessingCancelled(RuntimeError):
    """用户主动取消图片修复"""


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


def fsr_available():
    """当前 OpenCV 是否带 FSR 模块（需 opencv-contrib）。

    界面据此置灰 FSR 选项，避免用户选到才报错。
    """
    return hasattr(cv2, "xphoto")


def _inpaint_fsr(image, mask, radius=5, progress_cb=None,
                 cancel_check=None):
    """FSR（频率选择性重建）修复：质感远好于 TELEA/NS 扩散，
    接近 AI 且无需模型，但速度慢（逐通道重建），仅建议照片使用。
    注意 xphoto 接口只支持单通道、且 mask 语义与 cv2.inpaint 相反
    （0 = 待修复区）。
    radius：修复强度，决定蒙版膨胀范围（修正：不再写死为 5）；
    progress_cb：按 RGB 三通道的上报进度；cancel_check：通道间检测取消。"""
    if not fsr_available():
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
