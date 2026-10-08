# -*- coding: utf-8 -*-
"""
LaMa AI 修复引擎（big-lama，ONNX 推理）：
- 用真实纹理与结构填充选区，效果远超经典 TELEA/NS 扩散修复（不会糊成一团）
- 模型随软件打包内置（_internal/models 或 程序目录/models），离线开箱即用
- 自动选择推理后端：CUDA（NVIDIA 独显，最快）> DirectML（通用 GPU）> CPU
- ROI 裁剪推理：只把水印周围局部区域送入模型，最大限度提速
"""

import os
import sys
import glob
import logging
import threading
import time
from contextlib import contextmanager

import numpy as np
import cv2

from .processor import ProcessingCancelled

_log = logging.getLogger(__name__)

MODEL_NAME = "big-lama.onnx"
MIN_MODEL_BYTES = 190 * 1024 * 1024     # 正常模型约 207MB，过小视为损坏


def _candidate_paths():
    """模型查找路径：打包内置(_MEIPASS) -> exe/项目根目录"""
    dirs = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        dirs.append(os.path.join(meipass, "models"))
    if getattr(sys, "frozen", False):
        dirs.append(os.path.join(os.path.dirname(sys.executable), "models"))
    else:
        dirs.append(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    return [os.path.join(d, MODEL_NAME) for d in dirs]


def model_path():
    for p in _candidate_paths():
        if os.path.exists(p) and os.path.getsize(p) >= MIN_MODEL_BYTES:
            return p
    return None


def _fp16_path():
    """fp16 模型（GPU 加速用，体积减半）；存在且大小合理才返回"""
    name = MODEL_NAME.replace(".onnx", ".fp16.onnx")
    for d in _candidate_paths():
        p = os.path.join(os.path.dirname(d), name)
        if os.path.exists(p) and os.path.getsize(p) >= 90 * 1024 * 1024:
            return p
    return None


def runtime_available():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def is_ready():
    return runtime_available() and model_path() is not None


# ----------------------------------------------------------------------
# 推理会话：单会话即可（一次推理已可吃满 CPU 核心；DML 可用时走 GPU）
# ----------------------------------------------------------------------
_sessions = []
_free = []
_lock = threading.Lock()
_wait = threading.Condition(_lock)
_provider_name = None


def _new_cpu_session():
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = os.cpu_count() or 4
    return ort.InferenceSession(model_path(), so,
                                providers=["CPUExecutionProvider"])


def _probe_ok(sess):
    """会话自检：部分显卡驱动下 DirectML 跑不了本模型的算子。
    用非零随机数据实测（全零输入可能测不出故障），并校验输出有效，
    捕捉 DML 异步失败不抛异常、只返回垃圾的情况"""
    try:
        rng = np.random.default_rng(0)
        x = rng.random((1, 3, INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
        m = np.zeros((1, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        m[:, :, 200:300, 200:300] = 1.0
        names = [i.name for i in sess.get_inputs()]
        out = sess.run(None, {names[0]: x, names[1]: m})[0]
        return bool(np.isfinite(out).all()) and float(out.std()) > 0.01
    except Exception:
        _log.debug("推理会话自检失败", exc_info=True)
        return False


def _setup_cuda_path():
    """把 NVIDIA CUDA/cuDNN 运行时 DLL 路径加到 PATH 前面，
    使 onnxruntime CUDA EP 能找到它们（pip 装在 nvidia/*/bin/）。
    PyInstaller 打包后 DLL 被 collect_dynamic_libs 收集到 _MEIPASS。"""
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
        _log.debug("_setup_cuda_path: 无法定位 site-packages", exc_info=True)
        return
    for sp in sps:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            if os.path.isdir(d):
                p = os.environ.get("PATH", "")
                if d not in p:
                    os.environ["PATH"] = d + os.pathsep + p


def _ensure_pool():
    global _provider_name
    with _lock:
        if not _sessions:
            _setup_cuda_path()
            import onnxruntime as ort
            avail = ort.get_available_providers()
            s = None
            # 优先级：CUDA（NVIDIA 独显，最快）> DirectML（通用 GPU）> CPU
            for ep, label in [("CUDAExecutionProvider", "CUDA GPU 加速"),
                              ("DmlExecutionProvider", "DirectML GPU 加速")]:
                if ep not in avail:
                    continue
                # GPU 会话优先用 fp32 模型（精度最佳）；
                # 显存不够时退回 fp16（体积减半，部分显卡精度有损）
                for path in filter(None, (model_path(), _fp16_path())):
                    tag = "fp16" if ".fp16." in path else "fp32"
                    try:
                        so = ort.SessionOptions()
                        so.graph_optimization_level = \
                            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        so.intra_op_num_threads = 2
                        cand = ort.InferenceSession(
                            path, so,
                            providers=[ep, "CPUExecutionProvider"])
                        # 验证目标 EP 确实启用，而非静默回退到 CPU
                        if ep not in cand.get_providers():
                            continue
                        if _probe_ok(cand):
                            s = cand
                            _provider_name = f"{label}（{tag}）"
                            break
                    except Exception:
                        _log.debug("LaMa %s 后端 %s 初始化失败", ep, path,
                                   exc_info=True)
                        s = None
                if s is not None:
                    break
            if s is None:
                s = _new_cpu_session()
                _provider_name = f"CPU {os.cpu_count() or 4} 线程"
            _sessions.append(s)
            _free.append(s)


def provider_name():
    """当前使用的推理后端显示名"""
    _ensure_pool()
    return _provider_name


@contextmanager
def acquire_session():
    _ensure_pool()
    with _wait:
        while not _free:
            _wait.wait(timeout=0.5)
        sess = _free.pop()
    try:
        yield sess
    finally:
        with _wait:
            _free.append(sess)
            _wait.notify()


# ----------------------------------------------------------------------
# 推理（模型为固定 512×512 输入：ROI 小于 512 时按原比例保持原生分辨率、
# 用画面上下文补齐画布，避免放大再缩小损失清晰度；大于 512 才缩小）
# ----------------------------------------------------------------------
INPUT_SIZE = 512


def _ring_of(mask_bool):
    """蒙版外的环形邻域（膨胀 7px 再挖掉蒙版），用于采样周围画面"""
    ring = cv2.dilate(mask_bool.astype(np.uint8),
                      np.ones((7, 7), np.uint8)) > 0
    return ring & ~mask_bool


def _match_tone(img, res, ring):
    """局部色调匹配：在环形邻域上采样色调差，高斯传播到填充区内部。
    比全局中值偏移更精准——水印横跨不同色调区域（如天空+草地）时，
    各处得到不同校正，消除"补丁色差"。"""
    if int(ring.sum()) < 300:
        return res
    ring_f = ring.astype(np.float32)
    diff = (img.astype(np.float32) - res.astype(np.float32)) * ring_f[:, :, None]
    weight = cv2.GaussianBlur(ring_f, (0, 0), 20)
    weight[weight < 0.01] = 0.01
    delta = cv2.GaussianBlur(diff, (0, 0), 20) / weight[:, :, None]
    delta = np.clip(delta, -12.0, 12.0)
    return np.clip(np.rint(res.astype(np.float32) + delta),
                   0, 255).astype(np.uint8)


def _match_grain(img, res, mask_bool, ring):
    """给填充区补上与周围一致的颗粒/噪点：AI 生成的填充过于干净，
    与带感光颗粒或压缩噪点的真实画面并置会有"干净补丁"感。
    从蒙版外环形邻域估计高频噪声强度，向填充区注入等强度噪点。"""
    if int(ring.sum()) < 300:
        return res
    hp = img.astype(np.float32) - cv2.GaussianBlur(img, (0, 0), 1.5)
    sigma = float(hp[ring].std())
    if sigma < 1.5:
        return res          # 画面本身干净（动画/合成图），无需加颗粒
    sigma = min(sigma * 0.8, 8.0)
    noise = np.random.default_rng().normal(
        0.0, sigma, res.shape[:2])[:, :, None].astype(np.float32)
    return np.clip(res.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _paste_fill(img, res, mask_bool):
    """把模型填充结果贴回 ROI：蒙版内替换，边缘轻微羽化过渡。
    模型以周围画面为上下文生成填充，亮度/色彩天然衔接，直接羽化即可。
    注意：不可用 seamlessClone——OpenCV 5 对笔画状蒙版会静默退化为
    恒等操作（水印原样保留），且平坦填充经泊松求解会还原成边界色。"""
    ring = _ring_of(mask_bool)
    res = _match_tone(img, res, ring)
    res = _match_grain(img, res, mask_bool, ring)
    alpha = cv2.GaussianBlur(mask_bool.astype(np.float32), (0, 0), 1.2)
    alpha = alpha[:, :, None]
    out = img.astype(np.float32) * (1 - alpha) \
        + res.astype(np.float32) * alpha
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _run_model(sess, roi_bgr, roi_mask):
    h, w = roi_bgr.shape[:2]
    scale = min(1.0, INPUT_SIZE / max(h, w))
    if scale < 1.0:
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = cv2.resize(roi_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(roi_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if not np.any(msk):     # 细选区缩小后丢失时改用线性插值保形
            msk = (cv2.resize(roi_mask, (nw, nh),
                              interpolation=cv2.INTER_LINEAR) > 0).astype(np.uint8) * 255
    else:
        img, msk = roi_bgr, roi_mask
    ih, iw = img.shape[:2]

    # 居中放到 512 画布：图像用边缘复制填充，蒙版补 0
    top, left = (INPUT_SIZE - ih) // 2, (INPUT_SIZE - iw) // 2
    canvas = cv2.copyMakeBorder(
        img, top, INPUT_SIZE - ih - top, left, INPUT_SIZE - iw - left,
        cv2.BORDER_REPLICATE)
    mcanvas = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    mcanvas[top:top + ih, left:left + iw] = msk

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))[None]
    m = (mcanvas > 127).astype(np.float32)[None, None]

    names = [i.name for i in sess.get_inputs()]
    out = sess.run(None, {names[0]: x, names[1]: m})[0][0]
    out = np.transpose(out, (1, 2, 0)).astype(np.float32)
    # 本模型（Carve/LaMa-ONNX fp32）输出为 0-255 尺度而非 0-1，
    # 按值域自适应归一，切勿直接 clip(0,1)——那会把整幅结果钳成纯白
    if out.max() > 1.5:
        out = np.clip(out, 0, 255)
    else:
        out = np.clip(out, 0, 1) * 255.0
    res = out[top:top + ih, left:left + iw].astype(np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    if scale < 1.0:
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LINEAR)

    # 在原始分辨率下泊松融合贴回，消除亮度/色差补丁印
    return _paste_fill(roi_bgr, res, roi_mask > 127)


def infer_region(roi_bgr, roi_mask, cancel_check=None):
    """对单个 ROI（BGR uint8 + 0/255 mask）跑 LaMa，返回修复后的 ROI。
    cancel_check() 返回 True 时在推理前后抛 ProcessingCancelled
    （单次 onnx 推理本身不可中断，只能检查间隙）。"""
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("处理已取消")
    with acquire_session() as sess:
        res = _run_model(sess, roi_bgr, roi_mask)
    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("处理已取消")
    return res


TILE_MAX = INPUT_SIZE  # 每个瓦片不超过模型输入尺寸，保证原分辨率处理


def _axis_starts(total, tile, step):
    """沿一轴切瓦片起点：保证无缝覆盖 [0,total)，相邻瓦片重叠 tile-step"""
    if total <= tile:
        return [0]
    n = -(-(total - tile) // step) + 1      # ceil 除法 + 1
    starts = [i * step for i in range(n - 1)]
    starts.append(total - tile)             # 最后一片贴右/下边缘
    return starts


def _edge_weight(n, edge, at_start, at_end):
    """瓦片边缘 weight 渐变：条带内部边缘权重从 ~0 升到 1，避免拼接缝"""
    wgt = np.ones(n, dtype=np.float32)
    e = min(edge, n // 2)
    if e > 0:
        if not at_start:
            wgt[:e] = np.linspace(1.0 / e, 1.0, e)
        if not at_end:
            wgt[-e:] = np.linspace(1.0, 1.0 / e, e)
    return wgt


def infer_tiled(roi_bgr, roi_mask, progress_cb=None, cancel_check=None):
    """大尺寸 ROI 自动分块：宽/高超过 512 的水印（如全幅字幕、竖排台标）
    拆成多个重叠瓦片，每块以原分辨率推理，重叠带加权融合拼接。
    相比整块缩小到 512 再放大（细节尽失、一片模糊），分块保持原生分辨率，
    修复区清晰度与周围画面一致。
    progress_cb(done, total)：按瓦片上报真实进度；cancel_check 瓦片间检测。"""
    h, w = roi_bgr.shape[:2]
    if max(h, w) <= TILE_MAX:
        # 单瓦片一次推理：无内部进度点，由外层阶段进度管理（0.05→0.95）
        if cancel_check is not None and cancel_check():
            raise ProcessingCancelled("处理已取消")
        return infer_region(roi_bgr, roi_mask, cancel_check=cancel_check)

    overlap = 96         # 宽重叠带：跨块水印两侧生成的纹理差异被充分平滑
    step = TILE_MAX - overlap
    xs = _axis_starts(w, TILE_MAX, step)
    ys = _axis_starts(h, TILE_MAX, step)

    tiles = [(x0, x2, y0, y2)
             for x0 in xs for y0 in ys
             for x2 in (min(w, x0 + TILE_MAX),)
             for y2 in (min(h, y0 + TILE_MAX),)
             if np.any(roi_mask[y0:y2, x0:x2] > 0)]
    total = len(tiles)

    acc = np.zeros((h, w, 3), dtype=np.float64)
    wacc = np.zeros((h, w), dtype=np.float64)
    for done, (x0, x2, y0, y2) in enumerate(tiles, 1):
        if cancel_check is not None and cancel_check():
            raise ProcessingCancelled("处理已取消")
        tmsk = roi_mask[y0:y2, x0:x2]
        fixed = infer_region(roi_bgr[y0:y2, x0:x2], tmsk,
                             cancel_check=cancel_check)
        wx = _edge_weight(x2 - x0, overlap, x0 == 0, x2 == w)
        wy = _edge_weight(y2 - y0, overlap, y0 == 0, y2 == h)
        wgt = (wy[:, None] * wx[None, :]).astype(np.float64)
        acc[y0:y2, x0:x2] += fixed.astype(np.float64) * wgt[:, :, None]
        wacc[y0:y2, x0:x2] += wgt
        if progress_cb is not None:
            progress_cb(done, total)

    covered = wacc > 0
    out = roi_bgr.astype(np.float64).copy()
    out[covered] = acc[covered] / wacc[covered][:, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def roi_box(mask, shape, pad_ratio=0.8, min_pad=24, ctx=INPUT_SIZE,
            pad_max=None):
    """由 mask 外接矩形 + 上下文边距计算裁剪框 (x1,y1,x2,y2)。
    - 边距参考取 min(宽,高) 而非 max，避免全宽字幕等宽水印产生超大 ROI
    - ctx：窄长水印（如全幅字幕）在短边方向补足上下文至 ctx，
      让每个推理瓦片接近方形——LaMa 在窄长条上效果差，上下文充足才能真实还原
    - pad_max：上下文边距上限，避免大图上 ROI 爆炸拖慢瓦片推理"""
    h, w = shape[:2]
    ys, xs = np.where(mask > 0)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
    mw, mh = x2 - x1, y2 - y1
    ref = min(mw, mh)
    pad = int(max(min_pad, ref * pad_ratio))
    if pad_max is not None:
        pad = min(pad, int(pad_max))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

    if ctx:
        rw, rh = x2 - x1, y2 - y1
        if rw > ctx > rh:            # 横向超长：纵向补足上下文
            need = min(ctx, h) - rh
            if need > 0:
                up = min(y1, (need + 1) // 2)
                down = min(h - y2, need - up)
                y1 -= up + min(y1 - up, need - up - down)
                y2 += down
        elif rh > ctx > rw:          # 纵向超长：横向补足上下文
            need = min(ctx, w) - rw
            if need > 0:
                left = min(x1, (need + 1) // 2)
                right = min(w - x2, need - left)
                x1 -= left + min(x1 - left, need - left - right)
                x2 += right
    return (x1, y1, x2, y2)


def _appears_natural(res, mask_bool):
    """粗略自检修复区是否与邻域自然衔接（用于决定是否再修一轮）：
    - 拉普拉斯高频能量比（修复区 vs 环带）应在合理区间，过小=糊、过大=接缝
    - 低频亮度差不应过大（补丁色差）
    纹理极弱的纯色区域无判据，直接判通过。"""
    m = mask_bool
    if int(m.sum()) < 200:
        return True
    ring = _ring_of(mask_bool)
    if int(ring.sum()) < 800:
        return True
    g = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    s_in = float(lap[m].mean())
    s_out = float(lap[ring].mean())
    if s_out < 0.5:           # 环带近乎纯色，无纹理可对齐
        return True
    ratio = s_in / max(s_out, 1e-6)
    low = cv2.GaussianBlur(res, (0, 0), 4)
    tone = float(np.abs(low[m].mean(axis=0).astype(np.float32)
                         - low[ring].mean(axis=0).astype(np.float32)).mean())
    return (0.45 <= ratio <= 2.2) and tone <= 8.0


def inpaint(image_bgr, mask_u8, pad_ratio=1.5, min_pad=64, pad_max=384,
            max_iter=3, progress_cb=None, cancel_check=None):
    """对外主接口：对整图中的 mask 区域做 AI 修复。
    自动按 mask 外接矩形 + 上下文边距裁剪 ROI，推理后原样贴回。
    mask 需先做 1~2px 膨胀以覆盖水印抗锯齿边缘。

    面向"看不出痕迹"的调校：
    - 上下文默认给足（pad 1.5 / 最少 64 / 上限 384）：LaMa 全局感受野
      依赖周围结构，上下文越大，填充的纹理与结构延续越真实；
      pad_max 防止大图上 ROI 爆炸拖慢瓦片推理。
    - max_iter：修复后做一次质量自检（高频能量比 + 色调差），
      不自然则把当前结果作为新上下文再修一轮（最多 max_iter 次），
      显著抑制"补丁感/接缝/色差"。

    progress_cb(done, total)：0→准备，5%→95% 按轮次+瓦片真实上报，
    95%→100% 收尾；cancel_check() 返回 True 时抛 ProcessingCancelled。"""
    if not np.any(mask_u8):
        raise ValueError("选区为空")
    if progress_cb is not None:
        progress_cb(0.0, 1.0)

    x1, y1, x2, y2 = roi_box(mask_u8, image_bgr.shape,
                             pad_ratio, min_pad, pad_max=pad_max)
    roi = image_bgr[y1:y2, x1:x2]
    m = mask_u8[y1:y2, x1:x2]
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    if progress_cb is not None:
        progress_cb(0.05, 1.0)

    mask_bool = m > 127
    passes = max(max_iter, 1)
    pass_span = 0.90 / passes
    fixed = roi
    for it in range(passes):
        if cancel_check is not None and cancel_check():
            raise ProcessingCancelled("处理已取消")
        src = fixed if it > 0 else roi        # 后续轮以上轮结果作上下文重修

        lo = 0.05 + pass_span * it
        hi = 0.05 + pass_span * (it + 1)

        def _tile(done, total, _lo=lo, _hi=hi):
            if progress_cb is not None:
                progress_cb(_lo + (_hi - _lo) * (done / max(total, 1)), 1.0)

        fixed = infer_tiled(src, m, progress_cb=_tile,
                            cancel_check=cancel_check)
        if progress_cb is not None:
            progress_cb(hi, 1.0)
        if _appears_natural(fixed, mask_bool):
            break

    if cancel_check is not None and cancel_check():
        raise ProcessingCancelled("处理已取消")
    if progress_cb is not None:
        progress_cb(0.95, 1.0)
    out = image_bgr.copy()
    out[y1:y2, x1:x2] = fixed
    if progress_cb is not None:
        progress_cb(1.0, 1.0)
    return out
