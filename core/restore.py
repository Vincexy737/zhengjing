# -*- coding: utf-8 -*-
"""
老照片修复：划痕 / 破损修补、褪色还原、清晰度回补、（可选）智能上色。

处理链（可按需开关每一环）：
  1. 划痕检测 + 修复：形态学 top-hat / black-hat 找出细长亮暗线，交给 LaMa 重绘
  2. 颗粒与噪点：走 core.denoise 的 grain 模式
  3. 褪色还原：自动色阶 + CLAHE 把发灰的对比找回来
  4. 清晰度回补：必要时叠加超分
  5. 智能上色：DDColor（需下载模型）
"""

import cv2
import numpy as np

from . import adjust, denoise, filters, imglib, lama, models, sr


def detect_scratches(img, sensitivity=50):
    """检测划痕 / 折痕，返回 0-255 单通道蒙版。

    用横向 + 纵向的细长结构元素做 top-hat（亮划痕）与 black-hat（暗划痕），
    再按面积与长宽比过滤掉非线状的误检。
    """
    g = cv2.cvtColor(imglib.ensure_bgr(img), cv2.COLOR_BGR2GRAY)
    t = max(0.0, min(100.0, sensitivity)) / 100.0
    thr = int(28 - t * 18)                    # 灵敏度越高阈值越低
    mask = np.zeros(g.shape, np.uint8)
    for ksize in (9, 15, 25):
        for horiz in (True, False):
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (ksize, 1) if horiz else (1, ksize))
            mask = cv2.bitwise_or(
                mask, cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k))
            mask = cv2.bitwise_or(
                mask, cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    _, mask = cv2.threshold(mask, max(6, thr), 255, cv2.THRESH_BINARY)
    # 只保留细长的连通域（真正的划痕）
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros(g.shape, np.uint8)
    for i in range(1, n):
        x, y, w, h, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                            stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT],
                            stats[i, cv2.CC_STAT_AREA])
        if area < 12:
            continue
        long_side, short_side = max(w, h), min(w, h)
        if long_side >= 8 and (long_side / max(1.0, float(short_side))) >= 2.2:
            out[labels == i] = 255
    if cv2.countNonZero(out) == 0:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(out, k, iterations=1)


def repair_scratches(img, mask, progress_cb=None, cancel_cb=None):
    """用 LaMa 修复划痕区域；LaMa 不可用时退回 OpenCV 的 Navier-Stokes。"""
    if mask is None or cv2.countNonZero(mask) == 0:
        return img
    if lama.is_ready():
        out = lama.infer_region(img, mask, cancel_check=cancel_cb)
        if out is not None:
            return np.asarray(out)
    if progress_cb:
        progress_cb(1, 2)
    out = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    if progress_cb:
        progress_cb(2, 2)
    return out


def restore(img, do_scratch=True, scratch_sens=50, do_denoise=True,
            denoise_strength=55, do_color=True, do_sharpen=True,
            do_upscale=False, do_colorize=False,
            progress_cb=None, cancel_cb=None):
    """老照片一站式修复。

    返回 (结果图, 说明文本)；被取消时返回 (None, ...)。
    """
    img = imglib.ensure_bgr(img)
    steps = []
    total = sum([do_scratch, do_denoise, do_color, do_sharpen, do_upscale,
                 do_colorize]) or 1
    done = 0

    def tick(label):
        nonlocal done
        done += 1
        steps.append(label)
        if progress_cb:
            progress_cb(done, total)

    if do_scratch:
        m = detect_scratches(img, scratch_sens)
        if m is not None:
            img = repair_scratches(img, m, cancel_cb=cancel_cb)
        if cancel_cb and cancel_cb():
            return None, "已取消"
        tick("划痕修复")

    if do_denoise:
        img = denoise.denoise(img, strength=denoise_strength, mode="grain",
                              detail=30, cancel_cb=cancel_cb)
        if img is None:
            return None, "已取消"
        tick("颗粒降噪")

    if do_color:
        img = filters.auto_level(img, 0.5)
        img = filters.clahe(img, clip=1.8, grid=8)
        img = adjust.apply_saturation(img, 12)
        tick("褪色还原")

    if do_sharpen:
        img = filters.unsharp(img, amount=0.45, radius=1.0, threshold=3)
        tick("清晰度回补")

    if do_upscale and sr.is_ready("general"):
        out = sr.upscale(img, cancel_cb=cancel_cb, model="general")
        if out is not None:
            img = out
        tick("AI 放大 2×")

    if do_colorize:
        try:
            col = colorize(img)
            if col is not None:
                img = col
                tick("AI 上色")
        except Exception:
            steps.append("上色模型不可用")

    return img, " → ".join(steps) if steps else "未做任何处理"


# ----------------------------------------------------------------------
# 智能上色（DDColor）
# ----------------------------------------------------------------------
def colorize_ready():
    return models.is_ready("ddcolor")


def colorize(img, progress_cb=None, cancel_cb=None):
    """黑白 / 褪色照片上色。模型缺失时返回 None（调用方负责提示）。"""
    if not colorize_ready():
        return None
    sess = models.get_session("ddcolor")
    inputs = sess.get_inputs()
    h, w = img.shape[:2]
    size = 512

    gray = cv2.cvtColor(imglib.ensure_bgr(img), cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = small.astype(np.float32) / 255.0

    feed = {}
    for inp in inputs:
        shape = inp.shape
        # 形如 [1, 3, H, W] → 灰度三通道；[1, 1, H, W] → 单通道 L
        ch = shape[1] if len(shape) > 1 and isinstance(shape[1], int) else 3
        if ch == 1:
            lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
            arr = lab[:, :, 0:1] / 255.0
        else:
            arr = x
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        feed[inp.name] = arr
    if progress_cb:
        progress_cb(1, 3)

    outs = sess.run(None, feed)
    if progress_cb:
        progress_cb(2, 3)
    pred = outs[0]
    while pred.ndim > 3:
        pred = pred[0]

    if pred.shape[0] == 2:               # 输出 ab 通道 → 拼回 LAB
        lab_small = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_ch = lab_small[:, :, 0:1]
        ab = pred.transpose(1, 2, 0).astype(np.float32)
        ab = cv2.resize(ab, (size, size), interpolation=cv2.INTER_LINEAR)
        lab = np.concatenate([l_ch, ab], axis=2)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    elif pred.shape[0] >= 3:             # 直接输出 RGB
        out = np.clip(pred.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    else:
        return None
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    if progress_cb:
        progress_cb(3, 3)
    # 上色后颜色往往偏淡，轻微提饱和
    return adjust.apply_saturation(out, 10)
