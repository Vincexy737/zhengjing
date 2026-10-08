# -*- coding: utf-8 -*-
"""
AI 超分辨率（ONNX 推理）：
- 双模型：通用照片（SwinIR-M GAN）/ 动漫插画（Real-ESRGAN anime 6B）
- 3 通道输入，x4 放大后缩小到 x2（质量远超直接 x2 插值）
- 小图（≤512）直接整图推理
- 大图用 256×256 tile 分块，feather 混合
- warmup() 预编译 CUDA kernel
"""

import os
import sys
import glob
import time
import threading
import numpy as np
import cv2

MODELS = {
    "anime": "RealESRGAN_x4plus_anime_6B.onnx",
    "general": "SwinIR_M_x4_GAN.onnx",
}
MIN_MODEL_BYTES = 1 * 1024 * 1024
SCALE = 4
TARGET_SCALE = 2
TILE = 256
TILE_OVERLAP = 32
DIRECT_MAX = 256


def _pad_to_tile(img):
    """把 img 边缘填充补齐到 TILE×TILE。

    BORDER_REFLECT 要求每边填充量 < 对应边长且边长 ≥ 2；退化小图
    （某边为 1px，或填充量 ≥ 边长）会让 copyMakeBorder 抛错，此时退回
    BORDER_REPLICATE（对任何尺寸都安全）。
    """
    h, w = img.shape[:2]
    ph, pw = TILE - h, TILE - w
    ok = h >= 2 and w >= 2 and ph < h and pw < w
    bt = cv2.BORDER_REFLECT if ok else cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(img, 0, ph, 0, pw, bt)


def _model_dirs():
    dirs = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(os.path.join(meipass, "models"))
    if getattr(sys, "frozen", False):
        dirs.append(os.path.join(os.path.dirname(sys.executable), "models"))
    else:
        dirs.append(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    return dirs


def model_path(key="anime"):
    name = MODELS.get(key)
    if not name:
        return None
    for d in _model_dirs():
        p = os.path.join(d, name)
        if os.path.exists(p) and os.path.getsize(p) >= MIN_MODEL_BYTES:
            return p
    return None


def available_models():
    return [k for k in MODELS if model_path(k)]


def is_ready(key="anime"):
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_path(key) is not None


_sessions = {}
_lock = threading.Lock()
_provider = None


def _setup_cuda_path():
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = os.environ.get("PATH", "")
        if meipass not in p:
            os.environ["PATH"] = meipass + os.pathsep + p
        return
    try:
        import site
        sps = site.getsitepackages() + [site.getusersitepackages()]
    except Exception:
        return
    for sp in sps:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            if os.path.isdir(d):
                p = os.environ.get("PATH", "")
                if d not in p:
                    os.environ["PATH"] = d + os.pathsep + p


def _ensure_session(key="anime"):
    global _provider
    if key in _sessions:
        return _sessions[key]
    with _lock:
        if key in _sessions:
            return _sessions[key]
        # 不再 clear：保留其他模型会话，避免切换模型后重新加载+重新编译
        _setup_cuda_path()
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        avail = ort.get_available_providers()
        providers = None
        for ep, label in [("CUDAExecutionProvider", "CUDA GPU"),
                          ("DmlExecutionProvider", "DirectML GPU")]:
            if ep in avail:
                providers = [ep, "CPUExecutionProvider"]
                _provider = label
                break
        if providers is None:
            providers = ["CPUExecutionProvider"]
            _provider = f"CPU {os.cpu_count() or 4} 线程"
        _sessions[key] = ort.InferenceSession(
            model_path(key), so, providers=providers)
        return _sessions[key]


def provider_name():
    if _provider is None:
        _ensure_session()
    return _provider


def _run_tile(session, tile):
    x = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)
    x = x[np.newaxis, ...]
    out = session.run(None, {"input": x})[0]
    out = out[0].transpose(1, 2, 0)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def _feather(h, w, overlap):
    wt = np.ones((h, w), dtype=np.float32)
    o = overlap
    if h > o * 2:
        ramp = np.linspace(0, 1, o, dtype=np.float32)
        wt[:o, :] *= ramp[:, np.newaxis]
        wt[-o:, :] *= ramp[::-1, np.newaxis]
    if w > o * 2:
        ramp = np.linspace(0, 1, o, dtype=np.float32)
        wt[:, :o] *= ramp[np.newaxis, :]
        wt[:, -o:] *= ramp[::-1][np.newaxis, :]
    return wt


def warmup(*keys):
    """按实际推理 shape（TILE×TILE）预热：CUDA kernel 按 shape 编译，
    之前用 64x64 预热与 256 tile 不匹配，首次点击仍要编译 ~20 秒（看起来像卡死）"""
    for key in keys or ("anime",):
        if not is_ready(key):
            continue
        try:
            s = _ensure_session(key)
            dummy = np.zeros((TILE, TILE, 3), np.uint8)
            _run_tile(s, dummy)
        except Exception:
            pass


def _infer_padded(session, img, stage_cb, state):
    """统一 pad 到 TILE×TILE 推理再裁剪：CUDA kernel 按输入 shape 编译，
    任意尺寸直推会让每张新尺寸的图都触发一次 ~20 秒编译（表现为卡住）"""
    h, w = img.shape[:2]
    if h == TILE and w == TILE:
        return _run_tile(session, img), h, w
    if stage_cb is not None and not state["noted"]:
        stage_cb("首次运行需编译 GPU 核心（约 20-60 秒），请耐心等待…")
        state["noted"] = True
    padded = _pad_to_tile(img)
    return _run_tile(session, padded)[:h * SCALE, :w * SCALE], h, w


def upscale(img, progress_cb=None, cancel_cb=None, model="anime",
            stage_cb=None):
    """AI 超分辨率放大 2×（内部用 x4 模型放大后缩小到 x2）。
    stage_cb(str)：阶段性状态提示（如首次编译 GPU 核心），可为 None。"""
    first_use = model not in _sessions
    if first_use and stage_cb is not None:
        stage_cb("首次使用需加载 AI 模型并编译 GPU 核心（约 20-60 秒），请耐心等待…")
        time.sleep(0.15)
    session = _ensure_session(model)
    state = {"noted": first_use}
    h, w = img.shape[:2]

    if h <= DIRECT_MAX and w <= DIRECT_MAX:
        if progress_cb:
            progress_cb(0, 2)
        out, _, _ = _infer_padded(session, img, stage_cb, state)
        if cancel_cb and cancel_cb():
            return None
        if progress_cb:
            progress_cb(1, 2)
        result = cv2.resize(out, (w * TARGET_SCALE, h * TARGET_SCALE),
                            interpolation=cv2.INTER_AREA)
        if progress_cb:
            progress_cb(2, 2)
        return result

    # 累加缓冲按 x2 目标尺寸分配（此前按 x4 中间尺寸分配，4K 照片需 ~3GB，
    # 会把 16GB 内存机器拖入换页假死）；每个 tile 推理完立即缩到 x2 再累加
    out_h, out_w = h * TARGET_SCALE, w * TARGET_SCALE
    est_bytes = out_h * out_w * 3 * 4 + out_h * out_w * 4
    if est_bytes > 1_500_000_000:
        raise ValueError(
            f"图片过大（{w}×{h}），放大 2× 需约 {est_bytes/1e9:.1f} GB 内存，"
            f"请先缩小图片或裁剪后再增强")
    accum = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w), dtype=np.float32)

    step = TILE - TILE_OVERLAP
    ys = list(range(0, max(h - TILE_OVERLAP, 1), step))
    if ys[-1] + TILE < h:
        ys.append(max(h - TILE, 0))
    xs = list(range(0, max(w - TILE_OVERLAP, 1), step))
    if xs[-1] + TILE < w:
        xs.append(max(w - TILE, 0))

    total = len(ys) * len(xs)
    done = 0

    if stage_cb is not None and not state["noted"]:
        stage_cb("首次运行需编译 GPU 核心（约 20-60 秒），请耐心等待…")
        state["noted"] = True
        time.sleep(0.15)

    for y0 in ys:
        for x0 in xs:
            if cancel_cb and cancel_cb():
                return None
            y1 = min(y0 + TILE, h)
            x1 = min(x0 + TILE, w)
            th, tw = y1 - y0, x1 - x0
            tile = img[y0:y1, x0:x1]
            pad_h = TILE - th
            pad_w = TILE - tw
            if pad_h > 0 or pad_w > 0:
                tile = _pad_to_tile(tile)
            out = _run_tile(session, tile)
            out = out[:th * SCALE, :tw * SCALE]
            # 立即缩到 x2 目标尺度再累加，避免全图 x4 中间缓冲
            out2 = cv2.resize(out, (tw * TARGET_SCALE, th * TARGET_SCALE),
                              interpolation=cv2.INTER_AREA)
            oy0, ox0 = y0 * TARGET_SCALE, x0 * TARGET_SCALE
            oy1, ox1 = oy0 + th * TARGET_SCALE, ox0 + tw * TARGET_SCALE
            fw = _feather(th * TARGET_SCALE, tw * TARGET_SCALE,
                          TILE_OVERLAP * TARGET_SCALE)
            accum[oy0:oy1, ox0:ox1] += out2.astype(np.float32) \
                * fw[..., np.newaxis]
            weight[oy0:oy1, ox0:ox1] += fw
            done += 1
            if progress_cb:
                progress_cb(done, total)

    result = accum / np.maximum(weight[..., np.newaxis], 1e-6)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
