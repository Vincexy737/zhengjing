# -*- coding: utf-8 -*-
"""
图像基础读写层：
  - 中文路径安全的读 / 写 / 内存编解码
  - EXIF 方向自动纠正（手机照片常见）
  - 统一各格式的编码参数（质量、压缩级别、无损开关）
  - 按目标体积压缩（二分搜索质量）
  - 统一的图像列举与尺寸适配

全模块只依赖 numpy / opencv / Pillow，不依赖 UI，可独立单测。
"""

import os

import cv2
import numpy as np

# 支持的输入扩展名
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp",
    ".tif", ".tiff", ".gif", ".ico",
}

# 各格式的默认编码参数（质量 0-100 统一语义）
_DEFAULT_PARAMS = {
    ".jpg": [cv2.IMWRITE_JPEG_QUALITY, 95,
             cv2.IMWRITE_JPEG_OPTIMIZE, 1],
    ".jpeg": [cv2.IMWRITE_JPEG_QUALITY, 95,
              cv2.IMWRITE_JPEG_OPTIMIZE, 1],
    ".webp": [cv2.IMWRITE_WEBP_QUALITY, 92],
    ".png": [cv2.IMWRITE_PNG_COMPRESSION, 3],
}


# ----------------------------------------------------------------------
# 读写
# ----------------------------------------------------------------------
def imread(path, flags=cv2.IMREAD_COLOR, fix_exif=True):
    """读取图片为 BGR ndarray。

    用 np.fromfile + imdecode 而非 cv2.imread，以兼容中文 / 空格路径。
    fix_exif=True 时按 EXIF Orientation 纠正手机照片的旋转。
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"无法读取图片：{path}")
    if fix_exif:
        try:
            img = apply_exif_orientation(img, path)
        except Exception:
            pass
    return img


def imwrite(path, img, quality=95):
    """保存图片（中文路径安全）。quality 对 JPG/WebP 生效。
    PNG 时 quality 被映射为压缩级别（0-9，quality 越高压缩越小体积越大）。"""
    ext = os.path.splitext(path)[1].lower() or ".png"
    ok, buf = cv2.imencode(ext, img, encode_params(ext, quality))
    if not ok:
        raise ValueError(f"不支持的图片格式：{ext}")
    buf.tofile(path)
    return path


def encode_params(ext, quality=95):
    ext = ext.lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext in (".png", ".tif", ".tiff"):
        # quality 100 → 压缩级别 0（最快最大），quality 0 → 级别 9（最慢最小）
        level = int(round((100 - max(0, min(100, quality))) / 100.0 * 9))
        return [cv2.IMWRITE_PNG_COMPRESSION, level]
    base = _DEFAULT_PARAMS.get(ext)
    if base is None:
        return []
    params = list(base)
    if params:
        params[1] = int(max(0, min(100, quality)))
    return params


def encode(img, ext, quality=95):
    """编码为内存字节，失败抛 ValueError。"""
    ok, buf = cv2.imencode(ext, img, encode_params(ext, quality))
    if not ok:
        raise ValueError(f"编码失败：{ext}")
    return buf


def encoded_size(img, ext, quality):
    return int(encode(img, ext, quality).nbytes)


def apply_exif_orientation(img, path):
    """按 EXIF Orientation 旋转 / 翻转（1-8）。无 EXIF 时原样返回。"""
    from PIL import Image
    with Image.open(path) as im:
        ori = im.getexif().get(274, 1)
    if ori == 2:
        return cv2.flip(img, 1)
    if ori == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if ori == 4:
        return cv2.flip(img, 0)
    if ori == 5:
        return cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
    if ori == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if ori == 7:
        return cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_CLOCKWISE)
    if ori == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ----------------------------------------------------------------------
# 体积控制
# ----------------------------------------------------------------------
def compress_to_size(img, ext, target_bytes, lossless_ok=True):
    """把图片压到不超过 target_bytes。

    先在质量区间二分搜索；仍超标则逐步降采样（每次 ×0.85，最多 6 次）。
    返回 (图像, 质量, 实际字节数)；若已触底仍无法达标，抛 ValueError
    （绝不静默交付远超目标的文件）。

    注：lossless_ok 为兼容保留参数，当前恒以「达标或报错」为准则。
    """
    ext = ext.lower()
    cur = img
    quality = 95
    buf = encode(cur, ext, quality)
    scale_try = 0
    while buf.nbytes > target_bytes and scale_try < 6:
        lo, hi = 25, quality
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            test = encode(cur, ext, mid)
            if test.nbytes <= target_bytes:
                best = (test, mid)
                lo = mid + 1          # 还能更高质量
            else:
                hi = mid - 1
        if best is not None:
            return cur, best[1], int(best[0].nbytes)
        # 质量降到 25 仍超标 → 缩尺寸再来一轮
        scale_try += 1
        h, w = cur.shape[:2]
        cur = cv2.resize(cur, (max(1, int(w * 0.85)), max(1, int(h * 0.85))),
                         interpolation=cv2.INTER_AREA)
        buf = encode(cur, ext, quality)
        if buf.nbytes <= target_bytes:
            return cur, quality, int(buf.nbytes)
    if buf.nbytes <= target_bytes:
        return cur, quality, int(buf.nbytes)
    # 质量下限 + 6 次降采样后仍超标：明确报错，交由调用方提示用户放宽目标
    smallest = int(encode(cur, ext, 25).nbytes)
    raise ValueError(
        f"无法压到目标体积（约 {human_size(int(target_bytes))}）："
        f"内容过于复杂，最小也有约 {human_size(smallest)}，"
        f"请放宽目标或先裁剪 / 缩小图片")


def human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# ----------------------------------------------------------------------
# 尺寸与格式
# ----------------------------------------------------------------------
def fit_max(img, max_side, interp=cv2.INTER_AREA):
    """等比缩放到最长边不超过 max_side。"""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side or max_side <= 0:
        return img
    s = max_side / float(m)
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=interp)


def ensure_bgr(img):
    """灰度→BGR，BGRA→BGR（白色底合成，避免透明区变黑）。"""
    if img is None:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        bgra = img
        alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
        bgr = bgra[:, :, :3].astype(np.float32)
        out = bgr * alpha + 255.0 * (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    return img


def ensure_uint8(img):
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def list_images(folder, recursive=True):
    """列出目录下所有图片（中文路径安全，按名称排序）。"""
    out = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    out.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(folder)):
            p = os.path.join(folder, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                out.append(p)
    return sorted(out)


def unique_path(path):
    """若 path 已存在，追加 (1)/(2)… 直到不冲突。"""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while os.path.exists(f"{base}({i}){ext}"):
        i += 1
    return f"{base}({i}){ext}"
