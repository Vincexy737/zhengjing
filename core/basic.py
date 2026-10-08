# -*- coding: utf-8 -*-
"""
基础编辑：裁剪 / 比例 / 尺寸 / 格式转换 / 压缩 / 重命名。

这些功能不需要 AI 模型，全部走 OpenCV 原生实现，
因此即时预览、批量处理都非常快。
"""

import os
import time
from collections import OrderedDict

import cv2
import numpy as np

from . import imglib

# 常用比例：值为 宽/高
RATIOS = OrderedDict([
    ("原始", None),
    ("1:1", 1.0),
    ("4:3", 4.0 / 3.0),
    ("3:4", 3.0 / 4.0),
    ("3:2", 3.0 / 2.0),
    ("2:3", 2.0 / 3.0),
    ("16:9", 16.0 / 9.0),
    ("9:16", 9.0 / 16.0),
    ("21:9", 21.0 / 9.0),
])

# 证件照常用尺寸（像素，按 300dpi 换算）
ID_SIZES = OrderedDict([
    ("一寸 (295×413)", (295, 413)),
    ("二寸 (413×579)", (413, 579)),
    ("小一寸 (260×378)", (260, 378)),
    ("大一寸 (390×567)", (390, 567)),
    ("护照 (354×472)", (354, 472)),
    ("签证 (413×531)", (413, 531)),
])


# ----------------------------------------------------------------------
# 裁剪
# ----------------------------------------------------------------------
def crop_ratio(img, ratio, align="auto"):
    """按目标宽高比裁剪，返回 (裁剪后图像, (x, y, w, h))。

    align:
      center —— 居中裁
      auto   —— 智能取景：优先保住人脸，否则取细节最丰富的区域
    """
    if not ratio:
        return img, (0, 0, img.shape[1], img.shape[0])
    h, w = img.shape[:2]
    cur = w / float(h)
    if abs(cur - ratio) < 1e-4:
        return img, (0, 0, w, h)
    if cur > ratio:            # 太宽，裁左右
        cw = int(round(h * ratio))
        ch = h
        max_x = w - cw
        x = _pick_x(img, cw, ch, align, max_x)
        y = 0
    else:                      # 太高，裁上下
        cw = w
        ch = int(round(w / ratio))
        max_y = h - ch
        x = 0
        y = _pick_y(img, cw, ch, align, max_y)
    x = max(0, min(x, w - cw))
    y = max(0, min(y, h - ch))
    return img[y:y + ch, x:x + cw].copy(), (x, y, cw, ch)


def _energy(img):
    """细节能量图（拉普拉斯绝对值），用于判断画面哪里「有内容」。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (256, max(1, int(256 * g.shape[0] / g.shape[1]))),
                   interpolation=cv2.INTER_AREA)
    return np.abs(cv2.Laplacian(g, cv2.CV_32F))


def _face_boxes(img):
    """Haar 人脸检测；模型文件缺失时静默返回空列表。"""
    try:
        path = os.path.join(cv2.data.haarcascades,
                            "haarcascade_frontalface_default.xml")
        if not os.path.exists(path):
            return []
        det = cv2.CascadeClassifier(path)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (512, max(1, int(512 * g.shape[0] / g.shape[1]))),
                       interpolation=cv2.INTER_AREA)
        g = cv2.equalizeHist(g)
        sx = img.shape[1] / float(g.shape[1])
        sy = img.shape[0] / float(g.shape[0])
        boxes = det.detectMultiScale(g, 1.1, 5, minSize=(24, 24))
        return [(int(x * sx), int(y * sy), int(w * sx), int(h * sy))
                for (x, y, w, h) in boxes]
    except Exception:
        return []


def _pick_x(img, cw, ch, align, max_x):
    if max_x <= 0:
        return 0
    if align not in ("auto", "smart"):
        return max_x // 2
    faces = _face_boxes(img)
    if faces:
        cx = sum(f[0] + f[2] / 2.0 for f in faces) / len(faces)
        return int(max(0, min(max_x, cx - cw / 2.0)))
    e = _energy(img)
    integ = cv2.integral(e)
    eh, ew = e.shape[:2]
    best, best_v = max_x // 2, -1
    for i in range(0, 33):
        fx = int(round(i / 32.0 * (ew - 1)))
        x0 = int(round(fx / ew * max_x))
        x1 = min(ew - 1, fx + max(1, int(round(cw / img.shape[1] * ew))))
        v = integ[eh - 1, x1] - integ[0, fx]
        if v > best_v:
            best_v, best = v, x0
    return int(max(0, min(max_x, best)))


def _pick_y(img, cw, ch, align, max_y):
    if max_y <= 0:
        return 0
    if align not in ("auto", "smart"):
        return max_y // 2
    faces = _face_boxes(img)
    if faces:
        cy = sum(f[1] + f[3] / 2.0 for f in faces) / len(faces)
        return int(max(0, min(max_y, cy - ch / 2.0)))
    e = _energy(img)
    integ = cv2.integral(e)
    eh, ew = e.shape[:2]
    best, best_v = max_y // 2, -1
    for i in range(0, 33):
        fy = int(round(i / 32.0 * (eh - 1)))
        y0 = int(round(fy / eh * max_y))
        y1 = min(eh - 1, fy + max(1, int(round(ch / img.shape[0] * eh))))
        v = integ[y1, ew - 1] - integ[fy, 0]
        if v > best_v:
            best_v, best = v, y0
    return int(max(0, min(max_y, best)))


def crop_rect(img, x, y, w, h):
    H, W = img.shape[:2]
    x0, y0 = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    x1, y1 = max(1, min(x + w, W)), max(1, min(y + h, H))
    return img[y0:y1, x0:x1].copy()


# ----------------------------------------------------------------------
# 尺寸
# ----------------------------------------------------------------------
def resize_img(img, mode="percent", value=100, keep_ar=True,
               other=None, interp=None):
    """尺寸调整。

    mode: percent | width | height | long | short | fit
      percent —— value 为百分比
      width / height —— 指定边，keep_ar 时另一边等比
      long / short —— 指定最长 / 最短边
      fit —— 装进 (value, other) 的框内，不裁剪
    """
    h, w = img.shape[:2]
    if mode == "percent":
        nw, nh = int(round(w * value / 100.0)), int(round(h * value / 100.0))
    elif mode == "fit":
        bw, bh = int(value), int(other or value)
        s = min(bw / float(w), bh / float(h))
        nw, nh = int(round(w * s)), int(round(h * s))
    else:
        if mode == "width":
            nw = int(value)
            nh = int(round(h * nw / float(w))) if keep_ar else h
        elif mode == "height":
            nh = int(value)
            nw = int(round(w * nh / float(h))) if keep_ar else w
        elif mode == "long":
            s = float(value) / max(w, h)
            nw, nh = int(round(w * s)), int(round(h * s))
        else:                                   # short
            s = float(value) / min(w, h)
            nw, nh = int(round(w * s)), int(round(h * s))
    nw, nh = max(1, nw), max(1, nh)
    if nw == w and nh == h:
        return img
    if interp is None:
        interp = cv2.INTER_AREA if (nw < w or nh < h) else cv2.INTER_LANCZOS4
    return cv2.resize(img, (nw, nh), interpolation=interp)


# ----------------------------------------------------------------------
# 格式转换与压缩
# ----------------------------------------------------------------------
FORMATS = ["JPG", "PNG", "WEBP", "BMP", "TIFF"]


def ext_of(fmt):
    fmt = fmt.lower().lstrip(".")
    return {"jpg": ".jpg", "jpeg": ".jpg", "png": ".png",
            "webp": ".webp", "bmp": ".bmp", "tif": ".tif",
            "tiff": ".tiff"}.get(fmt, "." + fmt)


def convert(img, dst_path, fmt=None, quality=92, target_kb=0):
    """转换格式并保存。

    fmt 为空时按扩展名推断；target_kb > 0 时按目标体积压缩。
    返回 (实际保存路径, 输出字节数)。
    """
    ext = ext_of(fmt) if fmt else (os.path.splitext(dst_path)[1].lower() or ".png")
    if os.path.splitext(dst_path)[1].lower() != ext:
        dst_path = os.path.splitext(dst_path)[0] + ext
    out = imglib.ensure_bgr(img) if ext in (".jpg", ".bmp", ".tif", ".tiff") else img
    if target_kb and target_kb > 0:
        out, q, _size = imglib.compress_to_size(out, ext, int(target_kb * 1024))
        buf = imglib.encode(out, ext, q)
        buf.tofile(dst_path)
        return dst_path, int(buf.nbytes)
    imglib.imwrite(dst_path, out, quality)
    return dst_path, int(os.path.getsize(dst_path))


def source_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ----------------------------------------------------------------------
# 重命名
# ----------------------------------------------------------------------
TOKENS = ["{name}", "{index}", "{w}", "{h}", "{date}", "{ext}"]


def rename(path, index=1, pattern="{name}", total=0, start=1, digits=3):
    """按模板生成新文件名（不含目录）。

    可用 token：{name} 原文件名 / {index} 序号 / {w} {h} 尺寸 /
    {date} 日期 / {ext} 原扩展名（含点）
    """
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    idx = str(start + index - 1).zfill(digits)
    out = pattern
    out = out.replace("{name}", name)
    out = out.replace("{index}", idx)
    out = out.replace("{date}", time.strftime("%Y%m%d"))
    out = out.replace("{ext}", ext)
    if "{w}" in out or "{h}" in out:
        try:
            img = imglib.imread(path, fix_exif=False)
            h, w = img.shape[:2]
        except Exception:
            w, h = 0, 0
        out = out.replace("{w}", str(w)).replace("{h}", str(h))
    # 清理非法字符
    for ch in '\\/:*?"<>|':
        out = out.replace(ch, "_")
    return out + ext
