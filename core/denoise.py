# -*- coding: utf-8 -*-
"""
图片降噪：拍照噪点 / 弱光高感噪点 / 老照片颗粒。

设计要点：
  - 亮度与色彩分开处理：噪点主要出现在亮度通道，色度通道只需温和平滑，
    分开处理比整图一把梭更能保住细节和颜色。
  - 降噪必然伴随细节损失，所以内置「细节回补」：带阈值的 USM 只锐化
    真实边缘，不会把噪点重新放大。
  - 四种模式覆盖不同场景，auto 按像素量自动选，避免大图卡死。
"""

import cv2
import numpy as np

from . import filters, imglib

MODES = ["auto", "fast", "fine", "grain"]


def denoise(img, strength=50, mode="auto", detail=35,
            progress_cb=None, cancel_cb=None):
    """降噪主入口。

    Args:
        img: BGR ndarray
        strength: 0-100 降噪强度
        mode: auto / fast / fine / grain
        detail: 0-100 细节回补强度（0 表示不回补）
        progress_cb: (done, total)
        cancel_cb: () -> bool

    Returns:
        BGR ndarray；被取消时返回 None
    """
    img = imglib.ensure_bgr(img)
    mp = (img.shape[0] * img.shape[1]) / 1e6

    if mode == "auto":
        mode = "fine" if mp <= 3.0 else "hybrid"
    if mode == "hybrid":
        return _denoise_hybrid(img, strength, detail, progress_cb, cancel_cb)
    if mode == "fast":
        return _denoise_fast(img, strength, detail, progress_cb, cancel_cb)
    if mode == "grain":
        return _denoise_grain(img, strength, detail, progress_cb, cancel_cb)
    return _denoise_fine(img, strength, detail, progress_cb, cancel_cb)


def _report(cb, done, total):
    if cb:
        cb(done, total)


def _cancelled(cb):
    return bool(cb and cb())


# ----------------------------------------------------------------------
# 模式实现
# ----------------------------------------------------------------------
def _denoise_fast(img, strength, detail, progress_cb, cancel_cb):
    """保边滤波：秒级出图，适合大图 / 批量预览。"""
    _report(progress_cb, 0, 3)
    s = max(0.0, min(100.0, strength)) / 100.0
    if _cancelled(cancel_cb):
        return None
    # YCrCb：只在亮度通道做保边平滑，色度通道轻度去噪
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    radius = int(3 + s * 6)
    eps = 0.01 + s * 0.06
    y = filters.guided_filter(y, y, radius=radius, eps=eps)
    _report(progress_cb, 1, 3)
    if _cancelled(cancel_cb):
        return None
    cr = cv2.bilateralFilter(cr, 0, 30 + s * 40, 5)
    cb_ = cv2.bilateralFilter(cb, 0, 30 + s * 40, 5)
    _report(progress_cb, 2, 3)
    out = cv2.cvtColor(cv2.merge((y, cr, cb_)), cv2.COLOR_YCrCb2BGR)
    if detail > 0:
        out = filters.unsharp(out, amount=detail / 100.0 * 0.9, radius=1.1,
                              threshold=3)
    _report(progress_cb, 3, 3)
    return out


def _denoise_fine(img, strength, detail, progress_cb, cancel_cb):
    """非局部均值（NLM）：质量最好的通用降噪，速度较慢。"""
    _report(progress_cb, 0, 3)
    s = max(0.0, min(100.0, strength)) / 100.0
    h = 3 + s * 12                 # 亮度滤波强度
    hc = 3 + s * 10                # 色彩滤波强度
    if _cancelled(cancel_cb):
        return None
    out = cv2.fastNlMeansDenoisingColored(
        img, None, h, hc, templateWindowSize=7, searchWindowSize=21)
    _report(progress_cb, 2, 3)
    if _cancelled(cancel_cb):
        return None
    if detail > 0:
        out = filters.unsharp(out, amount=detail / 100.0 * 0.8, radius=1.0,
                              threshold=3)
    _report(progress_cb, 3, 3)
    return out


def _denoise_hybrid(img, strength, detail, progress_cb, cancel_cb):
    """亮度 NLM + 色度双边：质量接近 fine，但只对单通道跑 NLM，快 2-3 倍。"""
    _report(progress_cb, 0, 4)
    s = max(0.0, min(100.0, strength)) / 100.0
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    if _cancelled(cancel_cb):
        return None
    y = cv2.fastNlMeansDenoising(y, None, 3 + s * 12, 7, 21)
    _report(progress_cb, 2, 4)
    if _cancelled(cancel_cb):
        return None
    cr = cv2.bilateralFilter(cr, 0, 25 + s * 45, 5)
    cb_ = cv2.bilateralFilter(cb, 0, 25 + s * 45, 5)
    _report(progress_cb, 3, 4)
    out = cv2.cvtColor(cv2.merge((y, cr, cb_)), cv2.COLOR_YCrCb2BGR)
    if detail > 0:
        out = filters.unsharp(out, amount=detail / 100.0 * 0.85, radius=1.0,
                              threshold=3)
    _report(progress_cb, 4, 4)
    return out


def _denoise_grain(img, strength, detail, progress_cb, cancel_cb):
    """老照片颗粒 / 扫描网点：先抑颗粒再保边，最后自动色阶。"""
    _report(progress_cb, 0, 4)
    s = max(0.0, min(100.0, strength)) / 100.0
    if _cancelled(cancel_cb):
        return None
    # 轻度中值先打散孤立颗粒（ksize 只取 3，避免糊掉五官）
    out = cv2.medianBlur(img, 3) if s > 0.2 else img
    _report(progress_cb, 1, 4)
    ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)
    y = cv2.fastNlMeansDenoising(y, None, 4 + s * 10, 7, 21)
    _report(progress_cb, 2, 4)
    if _cancelled(cancel_cb):
        return None
    y = filters.guided_filter(y, y, radius=int(2 + s * 4), eps=0.02 + s * 0.03)
    cr = cv2.bilateralFilter(cr, 0, 20 + s * 30, 5)
    cb_ = cv2.bilateralFilter(cb, 0, 20 + s * 30, 5)
    out = cv2.cvtColor(cv2.merge((y, cr, cb_)), cv2.COLOR_YCrCb2BGR)
    _report(progress_cb, 3, 4)
    if detail > 0:
        out = filters.unsharp(out, amount=detail / 100.0 * 0.7, radius=1.0,
                              threshold=4)
    _report(progress_cb, 4, 4)
    return out


def estimate_noise(img):
    """粗估噪点水平（0-100），用于给 UI 一个建议强度。

    用高频残差的中值绝对偏差估计：原图减去 3×3 中值后的残差能量。
    """
    g = cv2.cvtColor(imglib.ensure_bgr(img), cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(g, 3)
    resid = g.astype(np.float32) - med.astype(np.float32)
    sigma = float(np.median(np.abs(resid)))
    # sigma 经验值：干净图 <0.6，中等 1-3，重噪 >5
    level = (sigma - 0.4) / 4.6 * 100.0
    return int(max(0, min(100, round(level))))
