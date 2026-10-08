# -*- coding: utf-8 -*-
"""
调色与滤镜：基础参数调节 + 一键预设。

所有参数均为 0 中性、可正可负（gamma 除外，1.0 中性），
方便 UI 直接把滑块值 -100..100 绑过来，也方便预设按比例插值。
"""

import cv2
import numpy as np

from . import filters, imglib


def _clip(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def apply_temperature(img, amount):
    """色温：-100 冷（偏蓝）→ +100 暖（偏黄红）。"""
    if not amount:
        return img
    t = amount / 100.0
    b, g, r = cv2.split(img.astype(np.float32))
    r = r + 30 * t
    b = b - 30 * t
    g = g + 6 * t
    return _clip(cv2.merge((b, g, r)))


def apply_brightness_contrast(img, brightness, contrast):
    """亮度 / 对比度：-100..100 线性映射，围绕中灰 127.5 拉伸。"""
    # 标准公式：围绕中灰 127.5 拉伸对比度，再整体平移亮度
    alpha = 1.0 + contrast / 100.0
    off = brightness / 100.0 * 96.0
    out = (img.astype(np.float32) - 127.5) * alpha + 127.5 + off
    return _clip(out)


def apply_gamma(img, gamma):
    if abs(gamma - 1.0) < 1e-3:
        return img
    table = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / max(gamma, 0.05))
    lut = np.clip(table * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def apply_saturation(img, amount):
    """饱和度：-100 完全灰度，0 不变，+100 加倍。"""
    if not amount:
        return img
    s = 1.0 + amount / 100.0
    if s <= 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return _clip(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s, 0, 255)
    return cv2.cvtColor(_clip(hsv), cv2.COLOR_HSV2BGR)


def apply_tone(img, shadow, highlight):
    """阴影提亮 / 高光压暗，只作用于对应亮度区间，中间调不受影响。"""
    if not shadow and not highlight:
        return img
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if shadow:
        w = np.clip(1.0 - lum / 0.55, 0, 1) ** 2 * (shadow / 100.0) * 90.0
        img = np.clip(img.astype(np.float32) + w[..., None], 0, 255)
    if highlight:
        w = np.clip((lum - 0.45) / 0.55, 0, 1) ** 2 * (highlight / 100.0) * 90.0
        img = np.clip(img.astype(np.float32) - w[..., None], 0, 255)
    return img.astype(np.uint8)


def apply_curve(img, lift=(0, 0, 0), gain=(1, 1, 1)):
    """分区色调曲线：lift 抬高黑位（褪色胶片感），gain 控制对比。

    lift 顺序为 BGR，与 OpenCV 通道序一致。
    """
    if max(map(abs, lift)) == 0 and max(abs(g - 1) for g in gain) <= 1e-6:
        return img
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * gain[c] + lift[c]
    return _clip(out)


def adjust(img, brightness=0, contrast=0, saturation=0, temperature=0,
           gamma=1.0, sharpen=0, shadow=0, highlight=0,
           lift=(0, 0, 0), gain=(1, 1, 1)):
    """按顺序应用全部调节项。顺序会影响观感，固定为：
    色温 → 亮度对比 → gamma → 阴影高光 → 饱和度 → 曲线 → 锐化。"""
    img = imglib.ensure_bgr(img)
    img = apply_temperature(img, temperature)
    img = apply_brightness_contrast(img, brightness, contrast)
    img = apply_gamma(img, gamma)
    img = apply_tone(img, shadow, highlight)
    img = apply_saturation(img, saturation)
    img = apply_curve(img, lift, gain)
    if sharpen:
        img = filters.unsharp(img, amount=sharpen / 100.0 * 1.2, radius=1.0,
                              threshold=2)
    return img


# ----------------------------------------------------------------------
# 预设
# ----------------------------------------------------------------------
# 每项：可调参数 + 可选 curve(lift, gain)；bw 表示黑白类（强度插值时不改饱和度方向）
PRESETS = {
    "原图": {},
    "鲜艳": {"saturation": 24, "contrast": 10, "sharpen": 12},
    "柔和": {"contrast": -12, "saturation": -6, "shadow": 14, "highlight": -8},
    "明亮": {"brightness": 10, "shadow": 20, "highlight": -6},
    "日系": {"brightness": 8, "saturation": -12, "contrast": -8,
             "temperature": 5, "shadow": 14, "highlight": -8},
    "复古": {"saturation": -18, "temperature": 14, "contrast": -6,
             "shadow": 10, "lift": (10, 8, 6)},
    "胶片": {"contrast": 14, "saturation": 8, "temperature": -6,
             "lift": (6, 5, 8), "gain": (0.97, 0.98, 0.96)},
    "褪色": {"saturation": -32, "contrast": -10, "brightness": 6,
             "temperature": 8, "lift": (16, 14, 12)},
    "冷调": {"temperature": -20, "saturation": 6, "contrast": 6},
    "暖调": {"temperature": 20, "saturation": 6, "contrast": 4},
    "黑白": {"saturation": -100, "contrast": 16, "sharpen": 14},
    "高反差黑白": {"saturation": -100, "contrast": 38, "gamma": 0.95,
                   "sharpen": 18},
}


def preset_names():
    return list(PRESETS.keys())


def apply_preset(img, name, strength=100):
    """应用预设。strength 0-100 控制滤镜浓度（参数按比例缩放）。"""
    cfg = PRESETS.get(name)
    if not cfg:
        return img
    k = max(0.0, min(100.0, strength)) / 100.0
    p = {key: (v * k if isinstance(v, (int, float)) else v)
         for key, v in cfg.items()}
    lift = p.pop("lift", (0, 0, 0))
    gain = p.pop("gain", (1, 1, 1))
    lift = tuple(v * k for v in lift)
    gain = tuple(1.0 + (v - 1.0) * k for v in gain)
    return adjust(img, lift=lift, gain=gain, **p)


def auto_enhance(img):
    """一键智能增强：自动色阶 + 轻度 CLAHE + 轻饱和 + 轻锐化。"""
    img = imglib.ensure_bgr(img)
    out = filters.auto_level(img, 0.4)
    out = filters.clahe(out, clip=1.6, grid=8)
    out = apply_saturation(out, 8)
    out = filters.unsharp(out, amount=0.35, radius=1.0, threshold=3)
    return out
