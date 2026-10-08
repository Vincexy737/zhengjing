# -*- coding: utf-8 -*-
"""
通用滤波原语，被降噪 / 调色 / 增强等模块共用。

放在独立模块里，避免 denoise ↔ adjust 互相 import 形成循环依赖。
"""

import cv2
import numpy as np


def unsharp(img, amount=0.5, radius=1.2, threshold=3):
    """USM 锐化。amount 0-2，radius 为高斯半径（像素）。

    threshold>0 时只锐化明显边缘，避免把噪点一起放大——
    降噪后回补细节时必须带阈值，否则等于把噪声又加回去。
    """
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    if threshold > 0:
        low = np.abs(img.astype(np.float32) - blur.astype(np.float32))
        k = (low >= threshold).astype(np.float32)
        out = img.astype(np.float32) + (img.astype(np.float32) - blur) * amount * k
    else:
        out = img.astype(np.float32) * (1 + amount) - blur.astype(np.float32) * amount
    return np.clip(out, 0, 255).astype(np.uint8)


def guided_filter(guide, src, radius=4, eps=0.04):
    """引导滤波（保边平滑）。失败时回退到双边滤波。

    opencv-contrib 提供 ximgproc.guidedFilter；纯 opencv 环境用
    盒式均值自行实现，保证功能不缺失。
    """
    try:
        return cv2.ximgproc.guidedFilter(guide, src, radius, eps * 255 * 255)
    except (AttributeError, cv2.error):
        return _guided_box(guide, src, radius, eps)


def _guided_box(guide, src, radius, eps):
    g = guide.astype(np.float32) / 255.0
    s = src.astype(np.float32) / 255.0
    if g.ndim == 3:
        mean_g = cv2.blur(g, (radius, radius))
    else:
        mean_g = cv2.blur(g, (radius, radius))
    mean_s = cv2.blur(s, (radius, radius))
    corr_g = cv2.blur(g * g, (radius, radius)) if g.ndim == 2 else \
        cv2.blur(g * g, (radius, radius))
    corr_gs = cv2.blur(g * s, (radius, radius))
    var_g = corr_g - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s
    if g.ndim == 3:
        var_g = var_g.mean(axis=2, keepdims=True)
        cov_gs = cov_gs.mean(axis=2, keepdims=True)
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = cv2.blur(a, (radius, radius))
    mean_b = cv2.blur(b, (radius, radius))
    if g.ndim == 3:
        out = mean_a * g + mean_b
    else:
        out = mean_a * g + mean_b
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def clahe(img, clip=2.0, grid=8):
    """自适应直方图均衡（提亮暗部细节），在 LAB 的 L 通道上做，不改颜色。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = cl.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def auto_level(img, clip_pct=0.5):
    """自动色阶：按分位拉伸到 0-255，修复发灰 / 褪色的老照片。"""
    lo = np.percentile(img, clip_pct)
    hi = np.percentile(img, 100 - clip_pct)
    if hi - lo < 1e-3:
        return img
    out = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def blur_background(img, mask, ksize=25):
    """把 mask 之外的背景模糊（mask 为 0-255 单通道，255 表示主体）。"""
    m = (mask > 127).astype(np.uint8) * 255
    m = cv2.GaussianBlur(m, (0, 0), 3)
    mf = (m.astype(np.float32) / 255.0)[..., None]
    bg = cv2.GaussianBlur(img, (ksize | 1, ksize | 1), 0)
    out = img.astype(np.float32) * mf + bg.astype(np.float32) * (1 - mf)
    return np.clip(out, 0, 255).astype(np.uint8)
