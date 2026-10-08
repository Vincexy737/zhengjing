# -*- coding: utf-8 -*-
"""
扩展模型管理：注册表 / 按需下载 / ONNX Runtime 会话。

与 core.sr、core.lama 平级且互不干扰（各自持有自己的 session 缓存），
这样新增的 AI 功能（抠图、上色、人脸修复）不会挤掉原有模型的显存。

模型存放位置（按优先级）：
  1. PyInstaller 临时目录 _MEIPASS/models
  2. 打包后 exe 同目录 models/
  3. 开发时项目根 models/
"""

import os
import sys
import glob
import hashlib
import logging
import threading
import urllib.request

_log = logging.getLogger(__name__)

MB = 1024 * 1024


class Spec:
    """一个可下载模型的描述。

    sha256: 期望文件的 SHA256（小写十六进制）。留空表示暂未固定校验值，
    下载时只做体积 sanity check；固定后下载产物会逐字节比对，防镜像投毒。
    """

    def __init__(self, key, title, file, size, urls, desc="", input_size=None,
                 sha256=""):
        self.key = key
        self.title = title
        self.file = file
        self.size = size
        self.urls = urls
        self.desc = desc
        self.input_size = input_size
        self.sha256 = (sha256 or "").strip().lower()


def _hf(repo, path):
    """HuggingFace 直连 + 国内镜像两个源，谁快用谁。"""
    return [
        f"https://hf-mirror.com/{repo}/resolve/main/{path}",
        f"https://huggingface.co/{repo}/resolve/main/{path}",
    ]


# ----------------------------------------------------------------------
# 模型目录表
# ----------------------------------------------------------------------
CATALOG = {
    "rmbg": Spec(
        key="rmbg", title="RMBG-1.4 抠图", file="RMBG-1.4.onnx",
        size=176 * MB, input_size=1024,
        urls=_hf("briaai/RMBG-1.4", "onnx/model.onnx"),
        desc="通用主体抠图，发丝级边缘，适合证件照 / 商品图 / 人像",
        sha256="8cafcf770b06757c4eaced21b1a88e57fd2b66de01b8045f35f01535ba742e0f"),
    "u2netp": Spec(
        key="u2netp", title="U²-Net 轻量抠图", file="u2netp.onnx",
        size=5 * MB, input_size=320,
        urls=_hf("Xenova/modnet-onnx", "onnx/model.onnx"),
        desc="体积仅 5MB，CPU 也能秒出，作为 RMBG 的轻量替代"),
    "ddcolor": Spec(
        key="ddcolor", title="DDColor 老照片上色", file="ddcolor.onnx",
        size=136 * MB, input_size=512,
        # 原源 piddnad/ddcolor-onnx 已 404，改用 edgetools 的 fp16 转换版
        # （输入 [1,3,512,512] RGB 0-1，输出 [1,2,512,512] ab，与 colorize() 兼容）
        urls=_hf("edgetools/ddcolor", "ddcolor-tiny-fp16.onnx"),
        desc="黑白 / 褪色老照片智能上色",
        sha256="2653da00dc15e54a45e5200b61dbf82ee9ceaf56b02bb9b9657569ac775e82e6"),
    "gfpgan": Spec(
        key="gfpgan", title="GFPGAN 人脸修复", file="gfpgan.onnx",
        size=341 * MB, input_size=512,
        # 原源 yangchangjie/GFPGAN-ONNX 已 404，改用 GFPGANv1.4 官方权重转换
        urls=_hf("Meeperomi/GFPGANv1.4-onnx", "GFPGANv1.4.onnx"),
        desc="模糊人像面部增强，修复五官细节",
        sha256="cd7311b8d9e13cdb1e208b12363182da58c7bf45e26d1aa67bbeac4751aae92e"),
}

MIN_MODEL_BYTES = 512 * 1024        # 小于此值视为下载残缺


def model_dirs():
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


def writable_dir():
    """模型写入目录：打包后放 exe 同级 models/，开发时放项目 models/。"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(base, "models")
    os.makedirs(d, exist_ok=True)
    return d


def path_of(key):
    spec = CATALOG.get(key)
    if not spec:
        return None
    for d in model_dirs():
        p = os.path.join(d, spec.file)
        if os.path.exists(p) and os.path.getsize(p) >= MIN_MODEL_BYTES:
            return p
    return None


def is_ready(key):
    return path_of(key) is not None


def missing_keys():
    return [k for k in CATALOG if not is_ready(k)]


def human_size(n):
    return f"{n / MB:.0f} MB" if n >= MB else f"{n / 1024:.0f} KB"


# ----------------------------------------------------------------------
# 下载
# ----------------------------------------------------------------------
class DownloadCancelled(Exception):
    pass


def download(key, progress_cb=None, cancel_cb=None, timeout=30):
    """下载模型到 models/ 目录。

    progress_cb(done_bytes, total_bytes)，cancel_cb() 返回 True 则中止。
    多个 URL 依次尝试，全部失败抛 RuntimeError。
    """
    spec = CATALOG.get(key)
    if not spec:
        raise ValueError(f"未知模型：{key}")
    dest = os.path.join(writable_dir(), spec.file)
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_MODEL_BYTES:
        return dest

    tmp = dest + ".part"
    last_err = None
    for url in spec.urls:
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                chunk = 256 * 1024
                with open(tmp, "wb") as f:
                    while True:
                        if cancel_cb and cancel_cb():
                            f.close()
                            _safe_unlink(tmp)
                            raise DownloadCancelled()
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        if progress_cb:
                            progress_cb(done, total or spec.size)
            got = os.path.getsize(tmp)
            # 体积下限：拦截 HTML 错误页 / 空响应
            if got < MIN_MODEL_BYTES:
                raise IOError("下载内容过小，可能不是模型文件")
            # 体积 sanity：与目录标注期望体积相差过大视为残缺（容忍 15%）
            if spec.size and abs(got - spec.size) > max(4 * MB, spec.size * 0.15):
                raise IOError(
                    f"下载体积异常（{got / MB:.0f}MB，期望约 "
                    f"{spec.size / MB:.0f}MB），文件可能不完整")
            # 完整性校验：已固定 SHA256 的模型逐字节比对，防镜像投毒 / MITM。
            # 不匹配按「该源失败」处理，继续尝试下一个镜像。
            if spec.sha256 and _sha256_file(tmp) != spec.sha256:
                raise IOError("下载文件校验失败（SHA256 不匹配），已丢弃")
            if os.path.exists(dest):
                _safe_unlink(dest)
            os.replace(tmp, dest)
            return dest
        except DownloadCancelled:
            raise
        except Exception as e:          # 换下一个源
            last_err = e
            _log.warning("模型 %s 下载源 %s 失败：%s", key, url, e)
            _safe_unlink(tmp)
            continue
    raise RuntimeError(
        f"模型下载失败（{spec.title}）：{last_err}\n"
        f"可手动下载后放到 models/{spec.file}")


def _sha256_file(path):
    """流式计算文件 SHA256（大文件不占内存）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def _safe_unlink(p):
    try:
        if p and os.path.exists(p):
            os.remove(p)
    except OSError:
        pass


# ----------------------------------------------------------------------
# ONNX Runtime 会话（与 sr / lama 独立的缓存）
# ----------------------------------------------------------------------
_sessions = {}
_lock = threading.Lock()
_provider = None


def _setup_cuda_path():
    """把 nvidia-*/bin 加进 PATH，否则 onnxruntime-gpu 找不到 CUDA DLL。"""
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


def provider_name():
    return _provider or "未初始化"


def get_session(key):
    """按 key 惰性创建并缓存 ORT 会话，优先 CUDA → DirectML → CPU。"""
    global _provider
    if key in _sessions:
        return _sessions[key]
    path = path_of(key)
    if not path:
        raise FileNotFoundError(f"模型缺失：{CATALOG[key].title}")
    with _lock:
        if key in _sessions:
            return _sessions[key]
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
        _sessions[key] = ort.InferenceSession(path, so, providers=providers)
        return _sessions[key]


def warmup(key):
    """预热：让 CUDA kernel 提前编译，避免首次使用卡顿数秒。"""
    if not is_ready(key):
        return
    try:
        sess = get_session(key)
        inp = sess.get_inputs()[0]
        shape = [1 if d in (None, 0) else d for d in inp.shape]
        shape = [1 if not isinstance(d, int) else d for d in shape]
        dummy = _zeros_for(inp.type, shape)
        sess.run(None, {inp.name: dummy})
    except Exception:
        _log.debug("warmup(%s) 失败（不影响后续使用）", key, exc_info=True)


def _zeros_for(dtype, shape):
    import numpy as np
    if "float16" in str(dtype):
        return np.zeros(shape, dtype=np.float16)
    return np.zeros(shape, dtype=np.float32)
