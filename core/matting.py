# -*- coding: utf-8 -*-
"""
AI 抠图与背景处理。

两条路线，UI 可按模型可用性自动切换：
  1. 模型抠图（RMBG-1.4 / U²-Net）：任意背景，发丝级边缘
  2. 纯色抠图（漫水填充，无需模型）：证件照蓝 / 白 / 红底，秒出

抠图结果统一为 0-255 单通道 alpha（255 = 主体），
再交给 compose() 合成纯色 / 图片 / 虚化 / 透明背景。
"""

import cv2
import numpy as np

from . import filters, imglib, models

# 归一化参数（ImageNet 标准，RMBG / U²-Net 通用）
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------
# 引擎可用性
# ----------------------------------------------------------------------
def engine_names():
    """返回当前可用的抠图引擎：[(key, 显示名)]，模型缺失的不出现。"""
    out = []
    if models.is_ready("rmbg"):
        out.append(("rmbg", "AI 精准抠图 (RMBG-1.4)"))
    if models.is_ready("u2netp"):
        out.append(("u2netp", "AI 快速抠图 (U²-Net)"))
    out.append(("color", "纯色背景抠图（无需模型）"))
    return out


def ai_ready():
    return models.is_ready("rmbg") or models.is_ready("u2netp")


def default_engine():
    if models.is_ready("rmbg"):
        return "rmbg"
    if models.is_ready("u2netp"):
        return "u2netp"
    return "color"


# ----------------------------------------------------------------------
# 模型推理
# ----------------------------------------------------------------------
def _preprocess(img, size):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    inp = (inp - _MEAN) / _STD
    return inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


def _matte_model(img, key, progress_cb=None, cancel_cb=None):
    sess = models.get_session(key)
    spec = models.CATALOG[key]
    size = spec.input_size or 1024
    x = _preprocess(img, size)
    if progress_cb:
        progress_cb(1, 3)
    if cancel_cb and cancel_cb():
        return None
    name = sess.get_inputs()[0].name
    outs = sess.run(None, {name: x})
    if progress_cb:
        progress_cb(2, 3)
    if cancel_cb and cancel_cb():
        return None
    pred = outs[0]
    while pred.ndim > 2:                 # [1,1,H,W] → [H,W]
        pred = pred[0]
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = _sigmoid(pred)
    h, w = img.shape[:2]
    alpha = cv2.resize(pred.astype(np.float32), (w, h),
                       interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    if progress_cb:
        progress_cb(3, 3)
    return alpha


def matte_color(img, tol=32, refine=True):
    """纯色背景抠图：从四边采样做漫水填充，取反得到主体。

    对证件照（蓝 / 白 / 红底）、商品白底图效果很好，且完全不需要模型。
    """
    img = imglib.ensure_bgr(img)
    h, w = img.shape[:2]
    seed_img = img.copy()
    bg = np.zeros((h, w), np.uint8)
    seeds = [(1, 1), (w - 2, 1), (1, h - 2), (w - 2, h - 2),
             (w // 2, 1), (w // 2, h - 2), (1, h // 2), (w - 2, h // 2)]
    lo = (tol,) * 3
    hi = (tol,) * 3
    for sx, sy in seeds:
        m = np.zeros((h + 2, w + 2), np.uint8)
        try:
            cv2.floodFill(seed_img, m, (int(sx), int(sy)), 255, lo, hi,
                          flags=cv2.FLOODFILL_FIXED_RANGE)
        except cv2.error:
            continue
        bg = cv2.bitwise_or(bg, m[1:-1, 1:-1])
    alpha = cv2.bitwise_not(bg)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, k, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k, iterations=2)
    alpha = _keep_largest(alpha)
    if refine:
        alpha = refine_alpha(img, alpha)
    return alpha


def _keep_largest(mask):
    """只保留最大连通域，去掉背景中残留的孔洞与碎斑。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    # 背景（index 0）面积过大时说明是「多孔图」，此时不强制取单一连通域
    areas = stats[:, cv2.CC_STAT_AREA]
    if areas[0] > mask.size * 0.5 and n <= 2:
        return mask
    idx = int(np.argmax(areas[1:]) + 1)
    out = np.zeros_like(mask)
    out[labels == idx] = 255
    return out


def refine_alpha(img, alpha, radius=4, eps=1e-5):
    """用引导滤波把 alpha 边缘对齐到图像真实边缘（发丝更自然）。"""
    a = alpha.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = filters.guided_filter(g, (a * 255).astype(np.uint8),
                                radius=radius, eps=eps)
    out = np.clip(out.astype(np.float32), 0, 255).astype(np.uint8)
    return cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)


def matte(img, engine="auto", tol=32, refine=True,
          progress_cb=None, cancel_cb=None):
    """抠图主入口，返回 0-255 单通道 alpha。"""
    img = imglib.ensure_bgr(img)
    if engine == "auto":
        engine = default_engine()
    if engine == "color":
        return matte_color(img, tol=tol, refine=refine)
    if not models.is_ready(engine):
        # 模型缺失时静默降级到纯色抠图，保证功能可用
        return matte_color(img, tol=tol, refine=refine)
    a = _matte_model(img, engine, progress_cb, cancel_cb)
    if a is None:
        return None
    if refine:
        a = refine_alpha(img, a)
    return a


# ----------------------------------------------------------------------
# 合成
# ----------------------------------------------------------------------
def _cover(bg, w, h):
    """背景图等比铺满 w×h（居中裁切）。"""
    bh, bw = bg.shape[:2]
    s = max(w / float(bw), h / float(bh))
    nw, nh = int(round(bw * s)), int(round(bh * s))
    rs = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_AREA if s < 1
                    else cv2.INTER_LANCZOS4)
    x, y = max(0, (nw - w) // 2), max(0, (nh - h) // 2)
    return rs[y:y + h, x:x + w]


def compose(img, alpha, mode="color", color=(255, 255, 255), bg_img=None,
            blur=0, shadow=0, edge=0):
    """把 alpha 合成到新背景。

    mode: color（纯色）/ image（图片）/ blur（背景虚化）/ transparent（透明 PNG）
    shadow: 0-100 投影强度；edge: 0-100 边缘羽化/收缩（去白边）
    返回 BGR 或 BGRA（transparent 时）。
    """
    img = imglib.ensure_bgr(img)
    h, w = img.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    a = alpha.astype(np.float32) / 255.0

    if edge > 0:
        # 收缩 alpha，去掉抠图残留的背景色描边
        px = max(1, int(round(edge / 100.0 * 6)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
        a = cv2.erode(a, k, iterations=1)
        a = cv2.GaussianBlur(a, (0, 0), 0.8)

    if mode == "transparent":
        b, g, r = cv2.split(img.astype(np.float32))
        aa = (a * 255).astype(np.float32)
        return np.clip(np.dstack((b, g, r, aa)), 0, 255).astype(np.uint8)

    if mode == "blur":
        bg = cv2.GaussianBlur(img, (51, 51), 0) if not blur else \
            cv2.GaussianBlur(img, (int(blur) | 1, int(blur) | 1), 0)
    elif mode == "image" and bg_img is not None:
        bg = imglib.ensure_bgr(bg_img)
        bg = _cover(bg, w, h)
    else:
        bg = np.full_like(img, np.array(color, dtype=np.uint8))

    out = bg.astype(np.float32)
    if shadow > 0:
        out = _add_shadow(out, a, shadow)
    out = out * (1.0 - a[..., None]) + img.astype(np.float32) * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_shadow(bg, a, strength):
    """在主体下方叠一层柔和投影，证件照 / 商品图更立体。"""
    k = int(max(5, min(a.shape[1], a.shape[0]) * 0.03)) | 1
    sh = cv2.GaussianBlur(a, (k, k), 0)
    sh = np.roll(sh, int(k * 0.4), axis=0)
    sh = np.roll(sh, int(k * 0.2), axis=1)
    s = (strength / 100.0) * 0.55 * sh[..., None]
    return bg * (1.0 - s) + 20.0 * s
