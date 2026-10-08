# -*- coding: utf-8 -*-
"""
帧净 · 照片 AI 修复
现代浅色工作台布局：
  顶栏品牌区 | 左侧分组导航 | 左预览画布 | 右侧属性卡片
功能：
  - 去水印：矩形/画笔选区；LaMa AI 智能修复（纹理结构还原，最清晰）
    或 FSR 频率选择性重建（质感佳，无需模型）
  - 清晰度增强：AI 超分辨率放大
  - 更多功能页（降噪/老照片/裁剪/调色/抠图/批量等）注册在
    ui/tabs/__init__.py，由 main() 的侧边导航按需加载
"""

import math
import os
import queue
import sys
import threading
import time

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageTk, ImageDraw

from core import processor

# 以 python app.py / 打包 exe 直接运行时，本模块是 __main__；
# 而 ui.* 子模块都是 `from app import ...`，若不注册会把 app.py 再执行
# 一遍，产生两份类与两份 _DISPATCH_Q——后台线程的回调会投进没人消费
# 的队列，界面永远等不到结果。这里把 __main__ 注册成 "app" 共享同实例。
if __name__ == "__main__":
    sys.modules.setdefault("app", sys.modules[__name__])

# 后台线程 -> 主线程 的安全回调投递。
# Tk 的 widget.after() 不允许在非主线程调用（会抛
# "main thread is not in main loop"），因此后台线程只往队列里投，
# 主线程由 main() 里启动的轮询定时器消费并执行。
_DISPATCH_Q = queue.Queue()


def ui_post(fn):
    """后台线程把 fn 投递到主线程执行（线程安全，无阻塞）"""
    _DISPATCH_Q.put(fn)

IMAGE_FILETYPES = [
    ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("所有文件", "*.*"),
]

# ----------------------------------------------------------------------
# 设计令牌：清溪（Clear Stream）
#   极浅冷灰底 + 纯白卡片，靠留白与细描边分层；全界面只用一个
#   克制的静谧蓝主色，渐变也在同一蓝色家族内过渡，杜绝多色撞色；
#   光斑、光晕、流光统一压低浓度缓慢漂移，整体干净、柔和、舒适。
# ----------------------------------------------------------------------
BG        = "#F5F7FA"   # 应用底色（极浅冷灰，低刺激）
BG_SOFT   = "#FFFFFF"   # 次级底：顶栏
BG_WELL   = "#EDF0F5"   # 预览画布底
PANEL     = "#FFFFFF"   # 卡片 / 面板
PANEL_2   = "#F2F4F8"   # 控件底：按钮、输入框、轨道
PANEL_3   = "#E9EDF4"   # 悬停
PANEL_4   = "#DDE3EE"   # 按下 / 选中
LINE      = "#E4E8F0"   # 常规描边
LINE_HI   = "#D3DAE6"   # 悬停描边
TEXT      = "#232833"   # 主文字（柔和近黑，不刺眼）
TEXT_2    = "#5B6472"   # 次级文字
TEXT_3    = "#99A1AF"   # 弱文字 / 占位
BRAND     = "#4E6BF2"   # 主色·静谧蓝
BRAND_2   = "#5B8DEF"   # 辅色·浅蓝（同族）
BRAND_3   = "#6E7FF5"   # 强调·蓝紫（同族）
BRAND_DK  = "#EAEFFE"   # 徽章底（未选中段）
ON_BRAND  = "#FFFFFF"   # 渐变上的文字
G1        = "#4E6BF2"   # 主色（渐变已取消，三停同色 = 纯色扁平）
G2        = "#4E6BF2"
G3        = "#4E6BF2"
G1H       = "#6480F6"   # 悬停态（明度微抬的纯色）
G2H       = "#6480F6"
G3H       = "#6480F6"
GRAD      = (G1, G2, G3)         # 纯色填充（三停同色即无渐变）
GRAD_H    = (G1H, G2H, G3H)      # 悬停态纯色
GRAD_DIM  = ("#D5DDF7", "#D5DDF7", "#D5DDF7")   # 静息描边（主色的淡色纯色）
DANGER    = "#D9536A"
DANGER_BG = "#FBEDF0"
DANGER_HV = "#F7DEE3"
SEL_COLOR = "#FF5C7A"   # 选区蒙版
TOPBAR    = "#FFFFFF"   # 顶栏
GLOW_C    = "#7C93F5"   # 光晕 / 柔光主色（与主色同族）

FONT_XS   = ("Microsoft YaHei UI", 8)
FONT_SM   = ("Microsoft YaHei UI", 9)
FONT      = ("Microsoft YaHei UI", 10)
FONT_BOLD = ("Microsoft YaHei UI", 10, "bold")
FONT_H    = ("Microsoft YaHei UI", 11, "bold")
FONT_XL   = ("Microsoft YaHei UI", 17, "bold")
FONT_TIP  = ("Microsoft YaHei UI", 8)
MONO      = ("Consolas", 10)

APP_NAME  = "帧净"
APP_TITLE = "帧净 · 照片 AI 修复"


def asset_path(name):
    """随包资源路径：开发环境在项目 assets/，打包后在 _MEIPASS/assets/"""
    base = getattr(sys, "_MEIPASS", None)
    if not base:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "assets", name)


def app_icon_photo(size=32):
    """应用图标 PhotoImage（窗口/任务栏/顶栏 Logo 共用），缺失返回 None"""
    p = asset_path("icon.png")
    if not os.path.exists(p):
        return None
    try:
        img = Image.open(p).convert("RGBA")
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                        (img.width + side) // 2, (img.height + side) // 2))
        return ImageTk.PhotoImage(img.resize((size, size), Image.LANCZOS))
    except Exception:
        return None


def fmt_time(t):
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:04.1f}"


def cv_to_photo(img_bgr):
    """BGR ndarray -> Tk 可显示的 PhotoImage"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


def _user_log_dir():
    """用户可写的日志目录：%LOCALAPPDATA%\\帧净，失败回退 temp / exe 目录。"""
    base = (os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
            or os.path.dirname(sys.executable))
    d = os.path.join(base, APP_NAME)
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = os.path.dirname(sys.executable)
    return d


def _rotate_if_big(path, limit=2 * 1024 * 1024):
    """超过 limit 字节则轮转到 .1，避免日志无限增长。"""
    try:
        if os.path.exists(path) and os.path.getsize(path) > limit:
            os.replace(path, path + ".1")
    except OSError:
        pass


def install_crash_log():
    """打包运行（无控制台）时把未捕获异常写入用户目录的 crash.log。

    窗口化打包下 sys.stdout/stderr 为 None，异常只会弹一个对话框、
    一眼即逝且不便反馈；落盘后可以随时回看。
    日志放在 %LOCALAPPDATA%\\帧净（可写），不再写可能只读的 exe 目录，
    并做简单轮转避免无限增长。
    """
    if not getattr(sys, "frozen", False):
        return
    import traceback

    log = os.path.join(_user_log_dir(), "crash.log")

    def dump(exc_type, exc, tb):
        try:
            _rotate_if_big(log)
            with open(log, "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("".join(traceback.format_exception(exc_type, exc, tb)))
                f.write("-" * 60 + "\n")
        except Exception:
            pass

    sys.excepthook = dump

    def tk_dump(self, exc, val, tb):
        dump(type(val), val, tb)

    tk.Tk.report_callback_exception = tk_dump


def install_debug_log():
    """设置环境变量 ZHENGJING_DEBUG=1 时，把 core.* 的 debug 日志写入
    用户目录 debug.log，便于诊断「功能不可用但无明确原因」的后端降级
    （模型缺失 / GPU 初始化失败 / OpenCV 缺少 contrib 模块等）。"""
    import logging
    if not os.environ.get("ZHENGJING_DEBUG"):
        return
    try:
        path = os.path.join(_user_log_dir(), "debug.log")
        _rotate_if_big(path, limit=4 * 1024 * 1024)
        h = logging.FileHandler(path, encoding="utf-8")
        h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger().addHandler(h)
    except Exception:
        pass


def enable_high_dpi():
    """Windows 高分屏下让界面按物理像素清晰渲染"""
    if os.name != "nt":
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _rr(cv, x0, y0, x1, y1, r, **kw):
    """在画布上画圆角矩形（smooth 多边形）"""
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
           x1, y1 - r, x1, y1, x1 - r, y1,
           x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return cv.create_polygon(pts, smooth=True, **kw)


def _hex2rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def mix(a, b, t=0.5):
    """两个十六进制颜色按比例混合，用于生成同色系的深浅层次"""
    ca, cb = _hex2rgb(a), _hex2rgb(b)
    return "#%02X%02X%02X" % tuple(
        round(ca[i] + (cb[i] - ca[i]) * t) for i in range(3))


_PIL_CACHE = {"root": None, "imgs": {}}


def _cache_photo(key, build):
    """按 key 缓存 PhotoImage；换 Tk 实例时整体作废"""
    try:
        droot = id(tk._get_default_root())
    except Exception:
        droot = None
    if _PIL_CACHE["root"] != droot:
        _PIL_CACHE["root"] = droot
        _PIL_CACHE["imgs"] = {}
    got = _PIL_CACHE["imgs"].get(key)
    if got is not None:
        return got
    ph = build()
    _PIL_CACHE["imgs"][key] = ph
    return ph


def _grad_image(w, h, colors, vertical=False):
    """多色线性渐变底图：先生成 64 级渐变条再拉伸，避免逐像素开销"""
    cols = [_hex2rgb(c) for c in colors]
    segs = len(cols) - 1
    n = 64
    data = []
    for i in range(n):
        t = i / (n - 1) * segs
        k = min(int(t), segs - 1)
        f = t - k
        c1, c2 = cols[k], cols[k + 1]
        data.append(tuple(round(c1[j] + (c2[j] - c1[j]) * f) for j in range(3)))
    if vertical:
        g = Image.new("RGB", (1, n))
    else:
        g = Image.new("RGB", (n, 1))
    g.putdata(data)
    return g.resize((max(int(w), 1), max(int(h), 1)), Image.BILINEAR)


def grad_photo(w, h, colors, vertical=False):
    """纯渐变矩形（顶栏底 / 分隔线）。尺寸随窗口变化，故不缓存"""
    return ImageTk.PhotoImage(_grad_image(w, h, colors, vertical))


def rounded_photo(w, h, radius=12, grad=None, fill=None, outline=None,
                  width=1, vertical=False, pad=0, glow=None):
    """PIL 绘制圆角矩形，返回 PhotoImage。

    grad   : 两色及以上渐变（水平或 vertical=True 垂直）
    fill   : 纯色填充
    outline: 描边色
    pad    : 四周留白（配合 glow 给外发光留位置，图形内缩 pad 像素）
    glow   : (颜色, 模糊半径, 不透明度) 外发光
    """
    w, h, radius = max(int(w), 2), max(int(h), 2), max(int(radius), 1)
    pad = max(int(pad), 0)
    key = (w, h, radius, tuple(grad) if grad else grad, fill, outline, width,
           vertical, pad, glow)

    def build():
        shape = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        if grad:
            base = _grad_image(w, h, grad, vertical)
            mask = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask).rounded_rectangle(
                (0, 0, w - 1, h - 1), radius=radius, fill=255)
            shape = Image.composite(base.convert("RGBA"), shape, mask)
        elif fill:
            ImageDraw.Draw(shape).rounded_rectangle(
                (0, 0, w - 1, h - 1), radius=radius,
                fill=_hex2rgb(fill) + (255,))
        if outline:
            ImageDraw.Draw(shape).rounded_rectangle(
                (0, 0, w - 1, h - 1), radius=radius,
                outline=_hex2rgb(outline) + (255,), width=width)
        if pad or glow:
            ow, oh = w + 2 * pad, h + 2 * pad
            out = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
            if glow:
                gcol, blur, alpha = glow
                gl = Image.new("RGBA", (ow, oh), (0, 0, 0, 0))
                ImageDraw.Draw(gl).rounded_rectangle(
                    (pad, pad, pad + w - 1, pad + h - 1), radius=radius,
                    fill=_hex2rgb(gcol) + (255,))
                gl = gl.filter(ImageFilter.GaussianBlur(max(blur, 0.5)))
                gl.putalpha(gl.split()[3].point(lambda v: int(v * alpha)))
                out = Image.alpha_composite(out, gl)
            out.paste(shape, (pad, pad), shape)
            shape_img = out
        else:
            shape_img = shape
        return ImageTk.PhotoImage(shape_img)

    return _cache_photo(key, build)


def framed_photo(w, h, radius=10, fill=PANEL_2, border=GRAD_DIM, bw=1):
    """纯色底 + 1px 描边；描边给渐变色时自动合成为「渐变描边」"""
    key = ("frame", w, h, radius, fill,
           tuple(border) if isinstance(border, (list, tuple)) else border, bw)

    def build():
        if isinstance(border, (list, tuple)):
            outer = _grad_image(w, h, border)
        else:
            outer = Image.new("RGB", (w, h), _hex2rgb(border or fill))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle(
            (0, 0, w - 1, h - 1), radius=radius, fill=255)
        out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        out.paste(outer, (0, 0), mask)
        if bw > 0:
            inner = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            ImageDraw.Draw(inner).rounded_rectangle(
                (bw, bw, w - 1 - bw, h - 1 - bw), radius=max(radius - bw, 1),
                fill=_hex2rgb(fill) + (255,))
            out = Image.alpha_composite(out, inner)
        return ImageTk.PhotoImage(out)

    return _cache_photo(key, build)


def orb_photo(size, color, peak=0.35, soft=5.5):
    """径向光斑：极光底纹 / 光晕素材"""
    key = ("orb", size, color, peak, soft)

    def build():
        m = Image.new("L", (size, size), 0)
        ImageDraw.Draw(m).ellipse((0, 0, size - 1, size - 1), fill=255)
        m = m.filter(ImageFilter.GaussianBlur(max(size / soft, 1.0)))
        img = Image.new("RGBA", (size, size), _hex2rgb(color) + (0,))
        img.putalpha(m.point(lambda v: int(v * peak)))
        return ImageTk.PhotoImage(img)

    return _cache_photo(key, build)


def shine_photo(w, h, color="#FFFFFF", peak=0.9):
    """中间亮、两端透明的高光条（按钮 / 进度条流光用）"""
    key = ("shine", w, h, color, peak)

    def build():
        ww, hh = max(int(w), 2), max(int(h), 2)
        img = Image.new("RGBA", (ww, hh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        c = _hex2rgb(color)
        for i in range(ww):
            t = i / max(ww - 1, 1)
            a = int(255 * peak * (1 - abs(t * 2 - 1)) ** 2)
            d.line([(i, 0), (i, hh - 1)], fill=c + (a,))
        return ImageTk.PhotoImage(img)

    return _cache_photo(key, build)


# ----------------------------------------------------------------------
# 极光底纹：画布最底层缓慢漂移的彩色光斑，给静态界面注入呼吸感
# ----------------------------------------------------------------------
class AuroraLayer:
    """在指定画布的底层漂移若干光斑。

    只做 create_image + coords 平移，不重绘任何内容，
    因此开销极低（每 90ms 移动 3~4 张预渲染图），
    不影响蒙版涂抹的实时性。
    """

    TAG = "aurora"

    def __init__(self, canvas, colors=GRAD, size=460, peak=0.10,
                 interval=90, speed=1.0):
        self.cv = canvas
        self.size = size
        self.peak = peak
        self.interval = interval
        self.speed = speed
        self._t = 0.0
        self._job = None
        self._ids = []
        self._photos = [orb_photo(size, c, peak) for c in colors]
        # 每个光斑一套独立的利萨如轨迹：角速度 / 相位 / 基准位置
        self._paths = []
        for i in range(len(colors)):
            self._paths.append((
                0.085 + 0.031 * i,    # x 角速度
                0.062 + 0.027 * i,    # y 角速度
                i * 2.09, i * 1.31,   # 相位
                0.16 + 0.30 * ((i * 2) % 3),   # 基准 x（比例）
                0.24 + 0.26 * ((i * 3) % 3),   # 基准 y（比例）
            ))
        self.refresh()

    def reset(self):
        """画布被 delete("all") 清空后调用：丢弃旧元素重建"""
        self._ids = []
        self.refresh()

    def refresh(self):
        """（重）建光斑元素并压到画布最底层"""
        try:
            if not self.cv.winfo_exists():
                return
        except tk.TclError:
            return
        if not self._ids:
            for ph in self._photos:
                self._ids.append(
                    self.cv.create_image(0, 0, image=ph, tags=self.TAG))
        self._move()
        try:
            self.cv.tag_lower(self.TAG)
        except tk.TclError:
            pass
        if self._job is None:
            self._tick()

    def _move(self):
        w = max(self.cv.winfo_width(), 320)
        h = max(self.cv.winfo_height(), 220)
        t = self._t
        for oid, (ax, ay, px, py, bx, by) in zip(self._ids, self._paths):
            x = w * bx + math.cos(t * ax + px) * w * 0.34
            y = h * by + math.sin(t * ay + py) * h * 0.32
            try:
                self.cv.coords(oid, x, y)
            except tk.TclError:
                self._ids = []
                return

    def _tick(self):
        self._job = None
        try:
            if not self.cv.winfo_exists():
                return
        except tk.TclError:
            return
        if self.cv.winfo_ismapped():     # 窗口最小化时不空转
            self._t += self.interval / 1000.0 * self.speed
            self._move()
        self._job = self.cv.after(self.interval, self._tick)

    def stop(self):
        """暂停漂移（画布被不透明内容覆盖时调用，避免底层移动
        强制重绘上层大图，拖慢蒙版涂抹）"""
        if self._job is not None:
            try:
                self.cv.after_cancel(self._job)
            except Exception:
                pass
            self._job = None


class Breath:
    """呼吸灯：在若干预渲染图之间循环切换（Logo 光晕 / 状态点）"""

    def __init__(self, canvas, item, photos, ms=760):
        self.cv = canvas
        self.item = item
        self.photos = list(photos)
        self.ms = ms
        self._i = 0
        self._job = None
        self._tick()

    def _tick(self):
        self._job = None
        try:
            if self.cv.winfo_exists() and self.cv.winfo_ismapped():
                self.cv.itemconfig(self.item, image=self.photos[self._i])
                self._i = (self._i + 1) % len(self.photos)
        except tk.TclError:
            return
        self._job = self.cv.after(self.ms, self._tick)

    def stop(self):
        if self._job is not None:
            try:
                self.cv.after_cancel(self._job)
            except Exception:
                pass
            self._job = None


# ----------------------------------------------------------------------
# ttk 部件样式（仅给输入框、Frame 等少数系统控件打底，观感与自绘件统一）
# ----------------------------------------------------------------------
def apply_theme(root):
    root.configure(bg=BG)
    st = ttk.Style(root)
    try:
        st.theme_use("clam")
    except tk.TclError:
        pass
    st.configure(".", background=BG, foreground=TEXT,
                 fieldbackground=PANEL_2, borderwidth=0, arrowcolor=BRAND,
                 font=FONT)
    st.configure("TFrame", background=BG)
    st.configure("TLabel", background=BG, foreground=TEXT)
    st.configure("TButton", background=PANEL_2, foreground=TEXT,
                 borderwidth=0, relief="flat", padding=(10, 5))
    st.map("TButton", background=[("active", PANEL_3)])
    st.configure("TCombobox", fieldbackground=PANEL_2, foreground=TEXT,
                 background=PANEL_2, borderwidth=0, padding=5,
                 selectbackground=PANEL_2, selectforeground=TEXT,
                 arrowcolor=TEXT_2)
    st.map("TCombobox",
           fieldbackground=[("readonly", PANEL_2)],
           background=[("active", PANEL_2)])
    st.configure("TEntry", fieldbackground=PANEL_2, foreground=TEXT,
                 insertcolor=TEXT, padding=5, borderwidth=0)
    st.configure("TSpinbox", fieldbackground=PANEL_2, foreground=TEXT,
                 insertcolor=TEXT, padding=4, borderwidth=0)
    st.configure("TRadiobutton", background=BG, foreground=TEXT,
                 indicatorcolor=TEXT_2, borderwidth=0)
    st.map("TRadiobutton", background=[("active", BG)],
           foreground=[("active", TEXT)])
    root.option_add("*TCombobox*Listbox.background", PANEL_2)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", PANEL_4)
    root.option_add("*TCombobox*Listbox.selectForeground", TEXT)
    root.option_add("*TCombobox*Listbox.relief", "flat")
    root.option_add("*TCombobox*Listbox.borderWidth", 0)
    root.option_add("*Font", FONT)


# ======================================================================
# 基础自绘控件
# ======================================================================
def ui_font(spec):
    """按 (family, size, weight) 缓存 tk 字体对象"""
    spec = tuple(spec) + ("normal",) * (3 - len(spec))
    f = ui_font._cache.get(spec)
    if f is None:
        f = tkfont.Font(family=spec[0], size=spec[1], weight=spec[2])
        ui_font._cache[spec] = f
    return f


ui_font._cache = {}


class Button(tk.Canvas):
    """统一按钮。

    variant : primary（极光渐变主操作）| secondary | ghost | danger
    size    : sm 30 / md 36 / lg 44
    支持 config(state=/text=/bg=/font=)，并排时高度与圆角完全一致。
    主操作按钮在鼠标悬停时会有一道流光掠过（自绘，不影响布局）。
    """
    SIZES = {"sm": (30, 14, 9, 9), "md": (36, 18, 10, 10),
             "lg": (44, 22, 11, 11)}

    def __init__(self, parent, text="", command=None, variant="secondary",
                 size="md", bg=BG, width=None, stretch=False, **kw):
        self._h, self._padx, self._r, fs = self.SIZES[size]
        spec = kw.pop("font", ("Microsoft YaHei UI", fs,
                               "bold" if variant == "primary" else "normal"))
        self._font = ui_font(spec)
        self._text = text
        self._cmd = command
        self._variant = variant
        self._stretch = stretch
        self._minw = width or 0
        self._enabled = True
        self._hover = False
        self._press = False
        self._cw = None
        self._photo = None
        # 主按钮流光
        self._shine = None
        self._shine_ph = None
        self._shine_w = 0
        self._shine_x = -90
        self._shine_wait = 0
        self._shine_job = None
        kw.pop("padx", None)
        kw.pop("pady", None)
        state = kw.pop("state", "normal")
        super().__init__(parent, height=self._h, bg=bg, highlightthickness=0,
                         bd=0, cursor="hand2", **kw)
        self._layout()
        self.bind("<Enter>", lambda e: self._hover_on(True))
        self.bind("<Leave>", lambda e: self._hover_on(False))
        self.bind("<ButtonPress-1>", self._down)
        self.bind("<ButtonRelease-1>", self._up)
        if stretch:
            self.bind("<Configure>", lambda e: self._layout())
        if state == "disabled":
            self.set_enabled(False)

    # -- 外观 -----------------------------------------------------------
    def _skin(self):
        if not self._enabled:
            if self._variant == "ghost":
                return dict(text=TEXT_3)
            return dict(fill=PANEL_2, text=TEXT_3, line=LINE)
        if self._variant == "primary":
            return dict(grad=GRAD_H if (self._hover or self._press) else GRAD,
                        text=ON_BRAND, shine=self._hover and not self._press)
        if self._variant == "danger":
            if self._hover or self._press:
                return dict(fill=DANGER_HV, text=DANGER,
                            line=mix(DANGER, DANGER_HV, 0.5))
            return dict(fill=DANGER_BG, text=DANGER,
                        line=mix(DANGER, DANGER_BG, 0.45))
        if self._variant == "ghost":
            if self._hover or self._press:
                return dict(fill=PANEL_2, text=TEXT)
            return dict(text=TEXT_2)
        if self._press:
            return dict(fill=PANEL_4, text=TEXT, line=LINE_HI)
        if self._hover:
            return dict(fill=PANEL_3, text=TEXT, border=GRAD)
        return dict(fill=PANEL_2, text=TEXT, line=LINE)

    def _layout(self):
        tw = self._font.measure(self._text)
        w = max(self._minw, tw + self._padx * 2, self._h)
        if self._stretch:
            aw = self.winfo_width()
            if aw > 8:
                w = max(aw, tw + 16)
        if w != self._cw:
            self._cw = w
            tk.Canvas.config(self, width=w)
        self._paint()

    def _paint(self):
        w = self._cw or 1
        s = self._skin()
        self.delete("all")
        if s.get("grad"):
            self._photo = rounded_photo(w, self._h, self._r, grad=s["grad"])
        elif s.get("border"):
            self._photo = framed_photo(w, self._h, self._r, fill=s["fill"],
                                       border=s["border"])
        else:
            self._photo = rounded_photo(w, self._h, self._r,
                                        fill=s.get("fill"),
                                        outline=s.get("line"), width=1)
        self.create_image(0, 0, anchor="nw", image=self._photo)
        self._shine = None
        if s.get("shine"):
            self._shine_w = max(26, int(w * 0.42))
            self._shine_ph = shine_photo(self._shine_w, self._h - 6, peak=0.34)
            self._shine = self.create_image(self._shine_x, 3, anchor="nw",
                                            image=self._shine_ph)
        dy = 1 if self._press else 0
        self.create_text(w / 2, self._h / 2 + dy, text=self._text,
                         font=self._font, fill=s["text"])

    # -- 交互 -----------------------------------------------------------
    def _hover_on(self, on):
        if not on:
            self._press = False
        if not self._enabled:
            return
        if self._hover != on or self._press:
            self._hover = on
            self._layout()
        self._sync_shine()

    # 主按钮流光：悬停时一道高光从左到右掠过，扫完停一拍再来
    def _sync_shine(self):
        want = (self._hover and self._enabled
                and self._variant == "primary" and not self._press)
        if want and self._shine_job is None:
            self._shine_x = -90
            self._shine_wait = 0
            self._layout()
            if self._shine is not None:
                self._shine_step()
        elif not want and self._shine_job is not None:
            if self._shine_job:
                try:
                    self.after_cancel(self._shine_job)
                except Exception:
                    pass
            self._shine_job = None
            self._shine = None
            self._layout()

    def _shine_step(self):
        self._shine_job = None
        if self._shine is None or not self._enabled:
            return
        w = self._cw or 1
        if self._shine_wait > 0:
            self._shine_wait -= 1
        else:
            self._shine_x += max(5, int(w / 16))
            if self._shine_x > w + 8:
                self._shine_x = -self._shine_w - 6
                self._shine_wait = 24
            try:
                self.coords(self._shine, self._shine_x, 3)
            except tk.TclError:
                return
        self._shine_job = self.after(26, self._shine_step)

    def _down(self, _e):
        if self._enabled:
            self._press = True
            self._layout()

    def _up(self, _e):
        if not self._enabled:
            return
        fired, self._press = self._press, False
        self._hover = True
        self._layout()
        self._sync_shine()
        if fired and self._cmd:
            self._cmd()

    # -- 外部接口 -------------------------------------------------------
    def set_enabled(self, on):
        on = bool(on)
        if self._enabled == on:
            return
        self._enabled = on
        if not on:
            self._hover = False
        tk.Canvas.config(self, cursor="hand2" if on else "arrow")
        self._layout()
        self._sync_shine()

    def config(self, cnf=None, **kw):
        if cnf:
            kw = dict(cnf, **kw)
        if "state" in kw:
            self.set_enabled(kw.pop("state") != "disabled")
        if "text" in kw:
            self._text = kw.pop("text")
            self._cw = None
            self._layout()
        if "bg" in kw:
            tk.Canvas.config(self, bg=kw.pop("bg"))
            self._layout()
        if "font" in kw:
            self._font = ui_font(kw.pop("font"))
            self._cw = None
            self._layout()
        if kw:
            tk.Canvas.config(self, **kw)

    configure = config


class _SegItem(tk.Label):
    """分段控件中的单选项：胶囊底 + 文字，激活时铺极光渐变"""

    def __init__(self, parent, text, on_pick, h=30, bg=PANEL_2):
        self._font = ui_font(("Microsoft YaHei UI", 10))
        self._text = text
        self._on_pick = on_pick
        self._h = h
        self._bg = bg
        self._active = False
        self._hover = False
        self._photo = None
        w = self._font.measure(text) + 26
        super().__init__(parent, bg=bg, width=w, height=h, cursor="hand2",
                         compound="center", font=self._font)
        self._draw()
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", lambda e: self._on_pick())

    def _draw(self):
        w = self._font.measure(self._text) + 26
        if self._active:
            self._photo = rounded_photo(w, self._h, 8, grad=GRAD)
            fg = ON_BRAND
        elif self._hover:
            self._photo = rounded_photo(w, self._h, 8, fill=PANEL_3)
            fg = TEXT
        else:
            self._photo = rounded_photo(w, self._h, 8)
            fg = TEXT_2
        self.configure(image=self._photo, text=self._text, fg=fg)

    def _set_hover(self, on):
        if self._hover != on:
            self._hover = on
            self._draw()

    def set_active(self, on):
        if self._active != on:
            self._active = on
            self._draw()


class Segmented(tk.Canvas):
    """分段选择器：胶囊容器 + 内部渐变选中块。options: [(value, label)]"""

    PAD = 3
    ITEM_H = 30

    def __init__(self, parent, options, variable, command=None, bg=BG):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0)
        self.variable = variable
        self.command = command
        self._bg = bg
        self._items = []
        self._cw = 0
        holder = tk.Frame(self, bg=PANEL_2, bd=0)
        first = True
        for val, label in options:
            it = _SegItem(holder, label, lambda v=val: self._pick(v),
                          h=self.ITEM_H, bg=PANEL_2)
            it.pack(side="left", padx=(0 if first else 2, 0))
            first = False
            self._items.append((val, it))
        self._win = self.create_window(self.PAD, self.PAD, window=holder,
                                       anchor="nw")
        self.update_idletasks()
        self._natural = holder.winfo_reqwidth() + self.PAD * 2
        self.configure(width=self._natural,
                       height=self.ITEM_H + self.PAD * 2)
        variable.trace_add("write", lambda *a: self._sync())
        self._sync()
        self.bind("<Configure>", self._on_cfg)

    def _on_cfg(self, e):
        w = max(self.winfo_width(), self._natural)
        h = self.ITEM_H + self.PAD * 2
        if w == self._cw:
            return
        self._cw = w
        self.delete("bg")
        self._photo = framed_photo(w, h, 10, fill=PANEL_2, border=GRAD_DIM)
        self.create_image(0, 0, anchor="nw", image=self._photo, tags="bg")
        self.coords(self._win, max(self.PAD, (w - self._natural) // 2
                                   + self.PAD), self.PAD)
        self.tag_raise(self._win)

    def _pick(self, v):
        self.variable.set(v)
        if self.command:
            self.command()

    def set(self, label):
        """按选项文本（或值）设置选中项"""
        for val, it in self._items:
            if val == label or it._text == label:
                self.variable.set(val)
                if self.command:
                    self.command()
                return

    def _sync(self):
        cur = self.variable.get()
        for val, it in self._items:
            it.set_active(val == cur)


class Card(tk.Canvas):
    """圆角卡片：面板底 + 细描边 + 渐变圆点标题，内容挂到 self.body。
    高度随内容自适应，宽度随布局填充，悬停可亮边（set_hover）。"""

    def __init__(self, parent, title=None, padx=14, pady=12, bg=BG,
                 hoverable=False):
        self._padx, self._pady = padx, pady
        self._title = title
        self._hover = False
        self._key = None
        self._photo = None
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0)
        self._bg = bg
        self.body = tk.Frame(self, bg=PANEL)
        self._title_h = 30 if title else 0
        self._win = self.create_window(0, 0, window=self.body, anchor="nw")
        self.body.bind("<Configure>",
                       lambda e: self.after_idle(self._layout))
        self.bind("<Configure>", lambda e: self._layout())
        if hoverable:
            self.bind("<Enter>", lambda e: self.set_hover(True))
            self.bind("<Leave>", lambda e: self.set_hover(False))
        self._layout()

    def _layout(self):
        try:
            if not self.winfo_exists():
                return
        except tk.TclError:
            return
        w = self.winfo_width()
        if w < 8:
            self.after(20, self._layout)     # 等布局管理器给出真实宽度
            return
        bh = self.body.winfo_reqheight()
        h = self._title_h + bh + self._pady * 2
        self.configure(height=h)
        key = (w, h, self._hover)
        if key == self._key:
            return
        self._key = key
        self.delete("bg")          # 只清装饰层，保留托管内容的 window 元素
        # 1px 渐变描边：静息偏暗，悬停点亮为完整极光渐变
        self._photo = framed_photo(w, h, 14, fill=PANEL,
                                   border=GRAD if self._hover else GRAD_DIM)
        self.create_image(0, 0, anchor="nw", image=self._photo, tags="bg")
        if self._title:
            self._dot = rounded_photo(4, 15, 2, grad=(G1, G3), vertical=True)
            self.create_image(self._padx, self._pady + 5, anchor="nw",
                              image=self._dot, tags="bg")
            self._tf = ui_font(("Microsoft YaHei UI", 11, "bold"))
            self.create_text(self._padx + 13, self._pady + 10,
                             text=self._title, anchor="w", fill=TEXT,
                             font=self._tf, tags="bg")
        self.coords(self._win, self._padx, self._pady + self._title_h)
        self.itemconfigure(self._win, width=max(1, w - self._padx * 2))

    def set_hover(self, on):
        if self._hover != on:
            self._hover = on
            self._key = None
            self._layout()


class ScrollFrame(tk.Frame):
    """滚轮滚动的容器（右侧参数卡片用：矮窗口下内容可滚动，不遮挡）"""

    def __init__(self, parent, bg=BG):
        super().__init__(parent, bg=bg)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner,
                                              anchor="nw")
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")))
        self.bind("<Enter>",
                  lambda e: self.bind_all("<MouseWheel>", self._on_wheel))
        self.bind("<Leave>",
                  lambda e: self.unbind_all("<MouseWheel>"))

    def _on_wheel(self, e):
        try:
            hit = self.winfo_containing(e.x_root, e.y_root)
        except Exception:
            return
        if hit is None or not self._inside(hit):
            return
        y0, y1 = self.canvas.yview()
        if y0 <= 0.0 and y1 >= 1.0:
            return          # 内容没超出，无需滚动
        self.canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")

    def _inside(self, w):
        while w is not None:
            if w is self:
                return True
            w = getattr(w, "master", None)
        return False


class Switch(tk.Frame):
    """现代开关（替代系统复选框）：46×26 胶囊，开启时铺极光渐变并透出光晕"""

    W, H = 46, 26
    PAD = 6

    def __init__(self, parent, text="", variable=None, bg=PANEL,
                 command=None):
        super().__init__(parent, bg=bg, bd=0)
        self._var = variable if variable is not None else tk.BooleanVar()
        self._cmd = command
        self._photo = None
        self._cv = tk.Canvas(self, width=self.W + self.PAD * 2,
                             height=self.H + self.PAD * 2, bg=bg,
                             highlightthickness=0, bd=0, cursor="hand2")
        self._cv.pack(side="right")
        if text:
            lb = tk.Label(self, text=text, bg=bg, fg=TEXT, font=FONT,
                          anchor="w", cursor="hand2")
            lb.pack(side="left", fill="x", expand=True, padx=(0, 10))
            lb.bind("<Button-1>", lambda e: self.toggle())
        self._cv.bind("<ButtonRelease-1>", lambda e: self.toggle())
        self._var.trace_add("write", lambda *a: self._draw())
        self._draw()

    def _draw(self):
        c = self._cv
        c.delete("all")
        on = bool(self._var.get())
        if on:
            self._photo = rounded_photo(self.W, self.H, self.H // 2,
                                        grad=GRAD, pad=self.PAD,
                                        glow=(GLOW_C, 5, 0.32))
        else:
            self._photo = rounded_photo(self.W, self.H, self.H // 2,
                                        fill=PANEL_2, outline=LINE_HI,
                                        pad=self.PAD)
        c.create_image(0, 0, anchor="nw", image=self._photo)
        cy = self.PAD + self.H / 2
        cx = self.PAD + (self.W - 4 - 10 if on else 4 + 10)
        # 浅色底上给白色滑块加一圈投影，避免与轨道糊在一起
        c.create_oval(cx - 10, cy - 8, cx + 10, cy + 12,
                      fill=mix(LINE_HI, BRAND, 0.10), outline="")
        c.create_oval(cx - 10, cy - 10, cx + 10, cy + 10, fill="#FFFFFF",
                      outline=mix(LINE_HI, "#FFFFFF", 0.25), width=1)

    def toggle(self):
        self._var.set(not self._var.get())
        if self._cmd:
            self._cmd()

    def get(self):
        return self._var.get()


class Slider(tk.Canvas):
    """细轨滑块：轨道 5px，已选段主色，圆形滑块带主色描边"""

    H = 26
    TRACK = 5

    def __init__(self, parent, from_=0, to=100, variable=None, width=140,
                 bg=BG, command=None, **kw):
        self._lo, self._hi = float(from_), float(to)
        self._var = variable
        self._val = float(from_)
        self._cmd = command
        self._wid = max(int(width), 48)
        self._int = isinstance(variable, tk.IntVar)
        self._hover = False
        self._drag = False
        self._bg = bg
        super().__init__(parent, width=self._wid, height=self.H, bg=bg,
                         highlightthickness=0, bd=0, **kw)
        self._paint()
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._motion)
        self.bind("<ButtonRelease-1>", self._release)
        if variable is not None:
            variable.trace_add("write", lambda *a: self._paint())

    # -- 值 -------------------------------------------------------------
    def _cur(self):
        return float(self._var.get()) if self._var is not None else self._val

    def _set(self, v):
        v = max(self._lo, min(self._hi, v))
        if self._int:
            v = int(round(v))
        if self._var is not None:
            self._var.set(v)
        else:
            self._val = v
        self._paint()

    def get(self):
        return self._cur()

    def set(self, v):
        self._set(v)

    # -- 绘制 -----------------------------------------------------------
    def _paint(self):
        self.delete("all")
        w, h = self._wid, self.H
        cy = h / 2
        x0, x1 = 8, w - 8
        span = max(self._hi - self._lo, 1e-9)
        t = max(0.0, min(1.0, (self._cur() - self._lo) / span))
        kx = x0 + (x1 - x0) * t
        r = self.TRACK / 2
        _rr(self, x0, cy - r, x1, cy + r, r, fill=PANEL_2, outline="")
        fw = int(kx - x0)
        if fw > 2:
            self._fill = rounded_photo(fw, self.TRACK, int(r), grad=GRAD)
            self.create_image(x0, cy - r, anchor="nw", image=self._fill)
        if self._hover or self._drag:
            self._halo = orb_photo(32, GLOW_C, 0.30)
            self.create_image(kx, cy, image=self._halo)
        kr = 8.5 if (self._hover or self._drag) else 6.5
        self.create_oval(kx - kr, cy - kr + 2, kx + kr, cy + kr + 2,
                         fill=mix(LINE_HI, BRAND, 0.12), outline="")
        self.create_oval(kx - kr, cy - kr, kx + kr, cy + kr,
                         fill="#FFFFFF", outline=BRAND, width=2)

    def _set_hover(self, on):
        if self._hover != on:
            self._hover = on
            self._paint()

    # -- 交互 -----------------------------------------------------------
    def _from_x(self, x):
        x0, x1 = 8, self._wid - 8
        t = max(0.0, min(1.0, (x - x0) / max(x1 - x0, 1)))
        return self._lo + t * (self._hi - self._lo)

    def _press(self, e):
        self._drag = True
        self._set(self._from_x(e.x))
        if self._cmd:
            self._cmd()

    def _motion(self, e):
        if self._drag:
            self._set(self._from_x(e.x))
            if self._cmd:
                self._cmd()

    def _release(self, _e):
        self._drag = False
        self._paint()


class Select(tk.Frame):
    """下拉选择框：自绘圆角字段 + 弹出菜单，替代系统 Combobox"""

    H = 34

    def __init__(self, parent, values=(), width=200, command=None, bg=BG,
                 **kw):
        kw.pop("state", None)
        super().__init__(parent, bg=bg, bd=0)
        self._values = list(values)
        self._var = tk.StringVar(value=self._values[0] if self._values else "")
        self._cmd = command
        self._wid = max(int(width), 90)
        self._hover = False
        self._popup = None
        self._photo = None
        self._font = ui_font(("Microsoft YaHei UI", 10))
        self._cv = tk.Canvas(self, width=self._wid, height=self.H, bg=bg,
                             highlightthickness=0, bd=0, cursor="hand2")
        self._cv.pack()
        self._cv.bind("<Enter>", lambda e: self._set_hover(True))
        self._cv.bind("<Leave>", lambda e: self._set_hover(False))
        self._cv.bind("<ButtonPress-1>", self._open)
        self._paint()

    # -- 数据 -----------------------------------------------------------
    def get(self):
        return self._var.get()

    def set(self, v):
        self._var.set(v)
        self._paint()

    def cget(self, key):
        if key == "values":
            return list(self._values)
        return tk.Frame.cget(self, key)

    def config(self, cnf=None, **kw):
        if cnf:
            kw = dict(cnf, **kw)
        if "values" in kw:
            self._values = list(kw.pop("values"))
            self._paint()
        if "width" in kw:
            self._wid = max(int(kw.pop("width")), 90)
            tk.Canvas.config(self._cv, width=self._wid)
            self._paint()
        if "state" in kw:
            kw.pop("state")
        if kw:
            tk.Frame.config(self, **kw)

    configure = config

    # -- 绘制 -----------------------------------------------------------
    def _ellip(self, text):
        maxw = self._wid - 34
        if self._font.measure(text) <= maxw:
            return text
        s = text
        while s and self._font.measure(s + "…") > maxw:
            s = s[:-1]
        return s + "…"

    def _paint(self):
        c = self._cv
        c.delete("all")
        hot = bool(self._hover or self._popup)
        self._photo = framed_photo(self._wid, self.H, 9, fill=PANEL_2,
                                   border=GRAD if hot else GRAD_DIM)
        c.create_image(0, 0, anchor="nw", image=self._photo)
        c.create_text(12, self.H / 2, anchor="w", text=self._ellip(
            self._var.get()), font=self._font, fill=TEXT)
        ax, ay = self._wid - 18, self.H / 2 - 1
        c.create_polygon(ax - 5, ay - 2, ax + 5, ay - 2, ax, ay + 3,
                         fill=BRAND if hot else TEXT_2)

    def _set_hover(self, on):
        if self._hover != on:
            self._hover = on
            self._paint()

    # -- 弹出菜单 -------------------------------------------------------
    def _open(self, _e=None):
        if self._popup is not None:
            self._close()
            return
        p = tk.Toplevel(self)
        p.wm_overrideredirect(True)
        p.configure(bg=mix(GLOW_C, PANEL, 0.45))
        try:
            p.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        box = tk.Frame(p, bg=PANEL, bd=0)
        box.pack(padx=1, pady=1)
        cur = self._var.get()
        for v in self._values:
            sel = (v == cur)
            row = tk.Label(box, text=v, bg=PANEL_3 if sel else PANEL,
                           fg=BRAND if sel else TEXT, font=FONT, anchor="w",
                           padx=12, pady=7, cursor="hand2",
                           width=max(1, int((self._wid - 30) / 7)))
            row.pack(fill="x")
            row.bind("<Enter>", lambda e, r=row: r.config(bg=PANEL_3,
                                                          fg=TEXT))
            row.bind("<Leave>", lambda e, r=row, s=sel:
                     r.config(bg=PANEL_3 if s else PANEL,
                              fg=BRAND if s else TEXT))
            row.bind("<ButtonRelease-1>", lambda e, val=v: self._choose(val))
        p.update_idletasks()
        h = box.winfo_reqheight() + 2
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.H + 4
        if y + h > p.winfo_screenheight() - 8:
            y = max(0, self.winfo_rooty() - h - 4)
        p.geometry(f"{self._wid}x{h}+{x}+{y}")
        self._popup = p
        p.bind("<Escape>", lambda e: self._close())
        p.bind("<ButtonPress-1>", self._pclick)
        try:
            p.grab_set()
        except tk.TclError:
            pass
        self._paint()

    def _pclick(self, e):
        p = self._popup
        if p is None:
            return
        dx, dy = e.x_root - p.winfo_rootx(), e.y_root - p.winfo_rooty()
        if 0 <= dx <= p.winfo_width() and 0 <= dy <= p.winfo_height():
            return
        self._close()

    def _choose(self, v):
        self._close()
        if v != self._var.get():
            self._var.set(v)
            if self._cmd:
                self._cmd()
        self._paint()

    def _close(self):
        p, self._popup = self._popup, None
        if p is not None:
            try:
                p.grab_release()
            except tk.TclError:
                pass
            try:
                p.destroy()
            except tk.TclError:
                pass
        self._paint()


class ProgressBar(tk.Canvas):
    """圆角进度条：轨道 10px，已走段铺极光渐变，并有流光来回扫过"""

    H = 10

    def __init__(self, parent, width=180, bg=BG, **kw):
        width = kw.pop("length", width)
        self._wid = max(int(width), 40)
        self._max = 100.0
        self._val = 0.0
        self._photo = None
        self._fill = None
        self._shine = None
        self._sx = 0
        self._sw = 0
        self._fw = 0
        self._job = None
        super().__init__(parent, width=self._wid, height=self.H, bg=bg,
                         highlightthickness=0, bd=0)
        self.bind("<Destroy>", lambda e: self._stop())
        self._paint()

    def _paint(self):
        self.delete("all")
        self._photo = rounded_photo(self._wid, self.H, self.H // 2,
                                    fill=PANEL_2)
        self.create_image(0, 0, anchor="nw", image=self._photo)
        t = 0.0 if self._max <= 0 else max(0.0, min(1.0,
                                                    self._val / self._max))
        fw = int(self._wid * t)
        self._shine = None
        if fw > 2:
            self._fw = fw
            self._fill = rounded_photo(fw, self.H, self.H // 2, grad=GRAD)
            self.create_image(0, 0, anchor="nw", image=self._fill)
            if t < 0.999:                       # 未走完时加一道流光
                self._sw = max(16, int(fw * 0.32))
                if not self._sx:                # 进度刷新时不打断流光位置
                    self._sx = -self._sw
                self._shine_ph = shine_photo(self._sw, self.H, peak=0.5)
                self._shine = self.create_image(self._sx, 0, anchor="nw",
                                                image=self._shine_ph)
        self._sync()

    def _sync(self):
        if self._shine is not None and self._job is None:
            self._job = self.after(40, self._step)
        elif self._shine is None and self._job is not None:
            self._stop()

    def _stop(self):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except Exception:
                pass
            self._job = None

    def _step(self):
        self._job = None
        if self._shine is None:
            return
        self._sx += max(2, int(self._fw / 20))
        if self._sx > self._fw:
            self._sx = -self._sw
        try:
            self.coords(self._shine, self._sx, 0)
        except tk.TclError:
            return
        self._job = self.after(40, self._step)

    def config(self, cnf=None, **kw):
        if cnf:
            kw = dict(cnf, **kw)
        if "maximum" in kw:
            self._max = float(kw.pop("maximum") or 100.0)
        if "value" in kw:
            self._val = float(kw.pop("value") or 0.0)
        if "width" in kw:
            self._wid = max(int(kw.pop("width")), 40)
            tk.Canvas.config(self, width=self._wid)
        if kw:
            tk.Canvas.config(self, **kw)
        self._paint()

    configure = config

    def __getitem__(self, key):
        if key == "maximum":
            return self._max
        if key == "value":
            return self._val
        return tk.Canvas.__getitem__

    def cget(self, key):
        if key == "maximum":
            return self._max
        if key == "value":
            return self._val
        return tk.Canvas.cget(self, key)


# ======================================================================
# 可复用的选区编辑画布（去水印页与物体消除页共用）
# ======================================================================
class RegionEditor(ttk.Frame):
    """显示图片，支持矩形框选与画笔涂抹生成蒙版。
    渲染分层：底图 image_item 常驻，刷新时仅 itemconfig 换图，零重建。"""

    def __init__(self, parent):
        super().__init__(parent)
        self.image = None          # 当前底图 BGR
        self.mask = None           # 与底图同尺寸的 uint8 蒙版
        self._mask_has = False     # 蒙版是否含非零像素（避免每帧 countNonZero 全扫描）
        self.scale = 1.0           # 显示缩放比例
        self._base = None          # 降采样缓存：静态图片交互时不必每次重缩放
        self._base_key = None
        self._offset = (0, 0)      # 图像在画布上的居中偏移
        self._last_size = None
        self.tool = tk.StringVar(value="rect")
        self.brush_size = tk.IntVar(value=30)
        self._drag_start = None
        self._last_pt = None
        self._rect = None
        self._tk_img = None
        self.on_release = None      # 选区完成回调（鼠标释放时触发）
        self._img_item = None      # 常驻底图画布元素
        self._ovl = None           # 左上角信息胶囊（文字）
        self._ovl_bgi = None       # 左上角信息胶囊（底板）
        self._ovl_bg = None
        self._ovl_info = None      # 已渲染的信息文本（避免高频重复刷新）
        self._preview_rect = None
        self._overlay_mask = True  # 是否叠加显示红色选区蒙版
        self.placeholder = ""
        self._ph_item = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        """只保留画布：工具/参数统一由右侧属性面板承载，两页结构一致。
        画布外再套一层渐变描边框，底层铺缓慢漂移的极光光斑。"""
        self._aurora = None
        self.wrap = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self.wrap.pack(fill="both", expand=True, padx=12, pady=(10, 12))
        self.canvas = tk.Canvas(self.wrap, bg=BG_WELL, cursor="crosshair",
                                highlightthickness=0, bd=0)
        self._win = self.wrap.create_window(2, 2, window=self.canvas,
                                            anchor="nw")
        self.wrap.bind("<Configure>", self._frame_cfg)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Configure>", self._on_configure)

    def _frame_cfg(self, _e=None):
        """根据外框尺寸重画渐变描边，并把预览画布铺满内侧"""
        w, h = self.wrap.winfo_width(), self.wrap.winfo_height()
        if w < 8 or h < 8:
            return
        self.wrap.delete("chrome")
        self._chrome = framed_photo(w, h, 14, fill=BG_WELL, border=GRAD_DIM)
        self.wrap.create_image(0, 0, anchor="nw", image=self._chrome,
                               tags="chrome")
        self.wrap.itemconfigure(self._win, width=max(2, w - 4),
                                height=max(2, h - 4))
        self.wrap.tag_raise(self._win)      # 画布始终盖在描边之上
        self._ensure_aurora()

    def _ensure_aurora(self):
        """确保光斑底纹存在（画布被 delete("all") 后自动重建）"""
        if self._aurora is None:
            self._aurora = AuroraLayer(self.canvas, size=560, peak=0.08,
                                       interval=95, speed=0.9)
        elif not self._aurora._ids:
            self._aurora.reset()

    def _reset_aurora(self):
        if self._aurora is None:
            self._ensure_aurora()
        else:
            self._aurora.reset()

    # ------------------------------------------------------------------
    def set_placeholder(self, text):
        self.placeholder = text
        if self.image is None:
            self._draw_placeholder()

    def _draw_placeholder(self):
        """空状态：极光光晕 + 渐变描边图标 + 主标题 + 说明行"""
        self.canvas.delete("all")
        self._img_item = None
        self._ovl = None
        self._ovl_bgi = None
        self._ovl_info = None
        if not self.placeholder:
            self._ph_item = None
            return
        cw = max(self.canvas.winfo_width(), 240)
        ch = max(self.canvas.winfo_height(), 180)
        cx, cy = cw // 2, ch // 2
        title, _, hint = self.placeholder.partition("\n\n")
        iy = cy - 54
        # 底层光晕 + 渐变描边图标底板（远山与旭日的取景框）
        self._ph_glow = orb_photo(280, GLOW_C, 0.15)
        self.canvas.create_image(cx, iy, image=self._ph_glow)
        self._ph_box = framed_photo(68, 68, 20, fill=PANEL, border=GRAD)
        self.canvas.create_image(cx, iy, image=self._ph_box)
        self.canvas.create_oval(cx + 6, iy - 20, cx + 20, iy - 6, fill=BRAND,
                                outline="")
        self.canvas.create_polygon(cx - 21, iy + 18, cx - 5, iy - 10,
                                   cx + 8, iy + 18,
                                   fill=mix(BRAND, PANEL, 0.42), outline="")
        self.canvas.create_polygon(cx - 4, iy + 18, cx + 11, iy - 2,
                                   cx + 24, iy + 18,
                                   fill=mix(BRAND_3, PANEL, 0.38), outline="")
        self._ph_item = self.canvas.create_text(
            cx, cy + 26, text=title, fill=TEXT, font=FONT_H, justify="center")
        if hint:
            self.canvas.create_text(cx, cy + 52, text=hint, fill=TEXT_3,
                                    font=FONT_SM, justify="center")
        self._reset_aurora()

    def set_image(self, img_bgr):
        """设置底图并重置蒙版"""
        self.image = img_bgr
        self.mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        self._mask_has = False
        self._pause_aurora()        # 底层光斑被画面盖住，暂停省去强制重绘
        if self._ph_item is not None:
            self.canvas.delete(self._ph_item)
            self._ph_item = None
        self._ovl = None
        self._ovl_bgi = None
        self._ovl_info = None
        self._base = None            # 清掉降采样缓存（底图已换）
        self._base_key = None
        self.update_idletasks()
        self._last_size = None
        self._render()

    def clear_mask(self):
        if self.mask is not None:
            self.mask[:] = 0
            self._mask_has = False
            self._render()

    def set_overlay(self, on):
        """开关红色选区蒙版的叠加显示（效果预览时暂时隐藏）"""
        if self._overlay_mask != on:
            self._overlay_mask = on
            self._render()

    def has_selection(self):
        return self._mask_has

    def get_mask(self):
        return self.mask

    # ------------------------------------------------------------------
    def _render(self):
        """把底图 + 红色半透明蒙版渲染到画布。

        蒙版上色在「显示分辨率」上做：大图不再对全分辨率原图
        做整幅拷贝+上色，只缩放底图与蒙版后着色，画笔涂抹
        与清除选区因此流畅得多（真实蒙版仍保留全分辨率供处理）。"""
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        cw = max(self.canvas.winfo_width(), 240)
        ch = max(self.canvas.winfo_height(), 160)
        self._last_size = (cw, ch)
        self.scale = min(1.0, cw / w, ch / h)
        if self.scale < 1.0:
            nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
            key = (id(self.image), nw, nh)
            if self._base_key != key:
                self._base = cv2.resize(self.image, (nw, nh),
                                        interpolation=cv2.INTER_AREA)
                self._base_key = key
            if self._overlay_mask and self._mask_has:
                disp = self._base.copy()      # 显示分辨率拷贝（便宜）
                self._tint_display(disp, self.mask, nw, nh)
            else:
                disp = self._base
        else:
            if self._overlay_mask and self._mask_has:
                disp = self.image.copy()
                sel = self.mask > 0
                tint = disp[sel].astype(np.float32) * 0.45
                tint[:, 2] += 255 * 0.55          # 叠加红色
                disp[sel] = np.clip(tint, 0, 255).astype(np.uint8)
            else:
                disp = self.image
        self._tk_img = cv_to_photo(disp)
        dw, dh = self._tk_img.width(), self._tk_img.height()
        ox, oy = max(0, (cw - dw) // 2), max(0, (ch - dh) // 2)
        self._offset = (ox, oy)
        if self._img_item is None:
            self._img_item = self.canvas.create_image(ox, oy, anchor="nw",
                                                      image=self._tk_img)
            try:
                self.canvas.tag_lower("aurora")     # 光斑始终在画面之下
            except tk.TclError:
                pass
        else:
            self.canvas.itemconfig(self._img_item, image=self._tk_img)
            self.canvas.coords(self._img_item, ox, oy)
        # 左上角信息胶囊：原图尺寸 + 当前缩放
        # 信息没变化时跳过刷新（画笔涂抹/选区拖动的高频路径上，
        # 省去每帧 itemconfig + 两次 tag_raise 的 Tcl 开销）
        info = f"{w}×{h}" + ("" if self.scale >= 0.999
                             else f" · 缩放 {self.scale:.0%}")
        if self._ovl_info != info or self._ovl_bgi is None:
            self._ovl_info = info
            bw = ui_font(FONT_SM).measure(info) + 18
            self._ovl_bg = framed_photo(bw, 22, 11,
                                        fill=mix(PANEL_2, BG_WELL, 0.45),
                                        border=GRAD_DIM)
            if self._ovl_bgi is None:
                self._ovl_bgi = self.canvas.create_image(
                    12, 12, anchor="nw", image=self._ovl_bg)
                self._ovl = self.canvas.create_text(
                    21, 23, anchor="w", fill=TEXT_2, font=FONT_SM)
            else:
                self.canvas.itemconfig(self._ovl_bgi, image=self._ovl_bg)
            self.canvas.itemconfig(self._ovl, text=info)
            self.canvas.tag_raise(self._ovl_bgi)
            self.canvas.tag_raise(self._ovl)

    def _on_configure(self, event):
        if self.image is None:
            self._draw_placeholder()
            return
        size = (event.width, event.height)
        if size == self._last_size:
            return
        self._render()

    # ------------------------------------------------------------------
    def _to_image_xy(self, event):
        ox, oy = self._offset
        x = int((event.x - ox) / self.scale)
        y = int((event.y - oy) / self.scale)
        h, w = self.image.shape[:2]
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        return x, y

    def _on_press(self, event):
        if self.image is None:
            return
        pt = self._to_image_xy(event)
        if pt is None:
            return
        self._drag_start = pt
        self._last_pt = pt
        if self.tool.get() == "brush":
            self._paint(pt, pt)

    def _on_drag(self, event):
        if self.image is None or self._drag_start is None:
            return
        pt = self._to_image_xy(event)
        if pt is None:
            return
        if self.tool.get() == "rect":
            x0, y0 = self._drag_start
            ox, oy = self._offset
            sx0 = ox + int(x0 * self.scale)
            sy0 = oy + int(y0 * self.scale)
            sx1 = ox + int(pt[0] * self.scale)
            sy1 = oy + int(pt[1] * self.scale)
            if self._preview_rect:
                try:
                    self.canvas.delete(self._preview_rect)
                except tk.TclError:
                    pass
            self._preview_rect = self.canvas.create_rectangle(
                sx0, sy0, sx1, sy1, outline=SEL_COLOR, width=2, dash=(4, 2))
            h, w = self.image.shape[:2]
            self._rect = (max(0, min(x0, pt[0])), max(0, min(y0, pt[1])),
                          min(w - 1, max(x0, pt[0])),
                          min(h - 1, max(y0, pt[1])))
        else:
            self._paint(self._last_pt, pt)
            self._last_pt = pt

    def _on_release(self, event):
        if self.image is None or self._drag_start is None:
            return
        if self.tool.get() == "rect":
            if self._rect is not None:
                x0, y0, x1, y1 = self._rect
                if x1 - x0 >= 2 and y1 - y0 >= 2:
                    self.mask[y0:y1 + 1, x0:x1 + 1] = 255
                    self._mask_has = True
                    self._render()
        else:
            pt = self._to_image_xy(event)
            if pt is not None:
                self._paint(self._last_pt, pt)
        self._drag_start = None
        self._last_pt = None
        self._rect = None
        if self._preview_rect:
            try:
                self.canvas.delete(self._preview_rect)
            except tk.TclError:
                pass
            self._preview_rect = None
        if self.on_release and self._mask_has:
            self.on_release()

    def _paint(self, p0, p1):
        r = max(1, int(self.brush_size.get() / self.scale / 2))
        cv2.line(self.mask, p0, p1, 255, r * 2)
        cv2.circle(self.mask, p1, r, 255, -1)
        self._mask_has = True
        self._render()

    def _pause_aurora(self):
        """底层极光被画面盖住后暂停漂移（否则每次 coords 都会
        强制重绘上层大图区域，拖慢播放 / 涂抹实时性）"""
        if self._aurora is not None:
            self._aurora.stop()

    def _tint_display(self, disp, mask, nw, nh):
        """在显示分辨率上叠加红色蒙版：蒙版与底图同步降采样后再上色，
        避免对全分辨率原图做整幅拷贝 + 上色（大照片上开销可观）"""
        m = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        sel = m > 0
        if sel.any():
            tint = disp[sel].astype(np.float32) * 0.45
            tint[:, 2] += 255 * 0.55
            disp[sel] = np.clip(tint, 0, 255).astype(np.uint8)


# ======================================================================
# 页签一：照片去水印
# ======================================================================
class PhotoTab(ttk.Frame):
    """照片去水印：左侧大预览 + 右侧参数面板，底部状态条"""

    RIGHT_W = 326

    def __init__(self, parent):
        super().__init__(parent)
        self.src_path = None
        self._working = False
        self._history = []         # 修复前的底图栈（支持撤销/还原）
        self.msg_queue = queue.Queue()          # 后台修复线程 → 主线程消息
        self._cancel_evt = threading.Event()    # 取消修复的标志

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # ---- 左：预览编辑区 ----
        left = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, sticky="nsew")
        self.editor = RegionEditor(left)
        self.editor.pack(fill="both", expand=True, padx=(0, 4))
        self.editor.set_placeholder(
            "打开一张图片\n\n点「一键自动去水印」自动检测并去除文字水印")

        # ---- 历史栈（修复前的底图）放在功能页一侧，便于切换）
        right = tk.Frame(self, width=self.RIGHT_W, bg=BG)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)

        self.status = tk.Label(
            right, bg=BG, fg=TEXT_2, font=FONT_SM,
            text="请先打开一张图片，然后点「一键自动去水印」", anchor="w",
            wraplength=self.RIGHT_W - 28, justify="left")
        self.status.pack(fill="x", side="bottom", padx=14, pady=(4, 12))

        c3 = Card(right, "输出")
        c3.pack(fill="x", side="bottom", padx=14, pady=(0, 10))
        self.save_btn = Button(c3.body, "保存结果", size="lg", stretch=True,
                               bg=PANEL, command=self.save_result)
        self.save_btn.pack(fill="x")

        self.cancel_btn = Button(c3.body, "取消处理", variant="danger",
                                 stretch=True, bg=PANEL,
                                 command=self.cancel_photo_work)
        self.cancel_btn.pack(fill="x", pady=(8, 0))
        self.cancel_btn.pack_forget()          # 仅处理中显示
        rpb = tk.Frame(c3.body, bg=PANEL)
        rpb.pack(fill="x", pady=(10, 0))
        self.progress = ProgressBar(rpb, width=self.RIGHT_W - 106, bg=PANEL)
        self.progress.pack(side="left")
        self.pct_label = tk.Label(rpb, text="", bg=PANEL, fg=TEXT_2,
                                  font=FONT_SM, width=5, anchor="e")
        self.pct_label.pack(side="left", padx=(8, 0))

        props = ScrollFrame(right)
        props.pack(fill="both", expand=True, padx=(0, 14), pady=(14, 6))
        W = self.RIGHT_W - 14 - 28          # 卡内可用宽度

        # -- 操作 --
        c0 = Card(props.inner, "操作")
        c0.pack(fill="x")
        self.open_btn = Button(c0.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        self.tool_sel = Segmented(c0.body, [("rect", "矩形框选"),
                                            ("brush", "画笔涂抹")],
                                  variable=self.editor.tool,
                                  command=self._on_tool, bg=PANEL)
        self.tool_sel.pack(fill="x", pady=(10, 0))

        hr = tk.Frame(c0.body, bg=PANEL)
        hr.pack(fill="x", pady=(10, 0))
        self.undo_btn = Button(hr, "撤销一步", size="sm", stretch=True,
                               bg=PANEL, command=self.undo_step,
                               state="disabled")
        self.undo_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.reset_btn = Button(hr, "还原原图", size="sm", stretch=True,
                                bg=PANEL, command=self.reset_image,
                                state="disabled")
        self.reset_btn.pack(side="left", fill="x", expand=True)
        self.inpaint_btn = Button(c0.body, "去除选中水印", size="lg",
                                  stretch=True, variant="primary",
                                  bg=PANEL, command=self.run_inpaint)
        self.inpaint_btn.pack(fill="x", pady=(12, 0))
        self.clear_btn = Button(c0.body, "清除选区", size="sm",
                                stretch=True, bg=PANEL,
                                command=self.editor.clear_mask)
        self.clear_btn.pack(fill="x", pady=(8, 0))


        # -- 修复参数 --
        c1 = Card(props.inner, "修复参数")
        c1.pack(fill="x", pady=(12, 0))
        tk.Label(c1.body, text="算法", bg=PANEL, fg=TEXT_2,
                 font=FONT_SM).pack(anchor="w", pady=(0, 6))
        self.method = Select(
            c1.body,
            values=["AI 智能修复（推荐，最清晰）",
                    "FSR 频率选择性（质感佳，较慢）"],
            width=W, bg=PANEL,
            command=self._update_method_hint)
        self.method.set("AI 智能修复（推荐，最清晰）")
        self.method.pack()
        self.method_hint = tk.Label(c1.body, text="", bg=PANEL, fg=TEXT_3,
                                    font=FONT_XS, wraplength=W - 8,
                                    justify="left")
        self.method_hint.pack(anchor="w", pady=(6, 0))
        self._update_method_hint()
        tk.Label(c1.body, text="修复强度", bg=PANEL, fg=TEXT_2,
                 font=FONT_SM).pack(anchor="w", pady=(14, 4))
        rr = tk.Frame(c1.body, bg=PANEL)
        rr.pack(fill="x")
        self.radius = tk.IntVar(value=5)
        Slider(rr, from_=1, to=25, variable=self.radius, width=W - 36,
               bg=PANEL).pack(side="left")
        self.radius_lbl = tk.Label(rr, text="5", bg=PANEL, fg=TEXT,
                                   font=FONT_SM, width=3, anchor="e")
        self.radius_lbl.pack(side="left", padx=(8, 0))
        self.radius.trace_add("write", lambda *a: self.radius_lbl.config(
            text=str(self.radius.get())))
        tk.Label(c1.body, text="选区向外扩展的像素数，小水印用默认值即可",
                 bg=PANEL, fg=TEXT_3, font=FONT_XS,
                 wraplength=W - 8, justify="left").pack(anchor="w", pady=(4, 0))
        self.enhance = tk.BooleanVar(value=True)
        Switch(c1.body, "增强修复（二遍 + 羽化边缘）", self.enhance,
               bg=PANEL).pack(fill="x", pady=(16, 0))
        tk.Label(c1.body, text="修复两遍并羽化边缘，减少补丁痕迹，大面积水印建议开启",
                 bg=PANEL, fg=TEXT_3, font=FONT_XS,
                 wraplength=W - 8, justify="left").pack(anchor="w", pady=(4, 0))


    # ------------------------------------------------------------------
    def open_image(self):
        path = filedialog.askopenfilename(title="选择图片",
                                          filetypes=IMAGE_FILETYPES)
        if not path:
            return
        try:
            img = processor.load_image(path)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))
            return
        self.src_path = path
        self.editor.set_image(img)

        self.status.config(text=f"已打开：{os.path.basename(path)}　"
                                "请框选/涂抹水印区域，可分多次选取")

    def _method_flag(self):
        return (processor.INPAINT_TELEA
                if self.method.get().startswith("经典 TELEA")
                else processor.INPAINT_NS)

    def _update_method_hint(self):
        t = self.method.get()
        if t.startswith("AI"):
            self.method_hint.config(
                text="用 AI 分析周围画面智能填补，适合文字、图标、人物等"
                     "各种水印。效果最自然，首次需加载模型约 10 秒。")
        elif t.startswith("FSR"):
            self.method_hint.config(
                text="频域重建算法，不需 AI 模型。适合简单背景上的小面积"
                     "水印，大面积修复会模糊。")

    def _auto_recommend(self):
        """选区完成后自动分析画面特征，推荐并切换算法"""
        img = self.editor.image
        mask = self.editor.mask
        if img is None or mask is None or not np.any(mask):
            return
        ys, xs = np.where(mask > 0)
        if len(ys) < 30:
            return
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        mw, mh = x2 - x1, y2 - y1
        area_ratio = float(mask.sum()) / (mask.shape[0] * mask.shape[1] * 255)
        # 选区周围环形区域（用于估计背景复杂度）
        h, w = img.shape[:2]
        pad = max(12, min(mw, mh) // 3)
        rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
        rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)
        ring = (mask[ry1:ry2, rx1:rx2] == 0)
        if ring.sum() > 200:
            bg = img[ry1:ry2, rx1:rx2][ring]
            bg_std = float(bg.std())
        else:
            bg_std = 0
        # AI 几乎总是效果更好（尤其有 GPU 加速），
        # 只有极小面积+极简单背景才考虑 FSR
        if area_ratio > 0.01 or bg_std > 15:
            self.method.set("AI 智能修复（推荐，最清晰）")
            reason = "AI 智能修复效果最佳"
        else:
            self.method.set("AI 智能修复（推荐，最清晰）")
            reason = "小面积简单背景，AI 仍是最优选择"
        self._update_method_hint()
        self.method_hint.config(
            text="✎ " + reason + "\n" + self.method_hint.cget("text"))

    def _engine(self):
        t = self.method.get()
        if t.startswith("AI"):
            return "ai"
        if t.startswith("FSR"):
            return "fsr"
        return "classic"

    def _on_tool(self):
        """切换选区工具（矩形/画笔），同步状态栏提示"""
        if self.editor.tool.get() == "brush":
            self.status.config(text="画笔模式：按住左键涂抹水印区域，"
                                    "适合形状不规则的水印")
        else:
            self.status.config(text="矩形模式：拖拽框选水印区域，"
                                    "适合文字条/台标")


    def run_inpaint(self):
        if self._working:
            return
        if self.editor.image is None:
            messagebox.showinfo("提示", "请先打开一张图片")
            return
        if not self.editor.has_selection():
            messagebox.showinfo("提示", "请先用矩形或画笔选中水印区域")
            return
        engine = self._engine()
        if engine == "ai":
            from core import lama
            if not lama.is_ready():
                messagebox.showerror(
                    "AI 模型不可用",
                    "AI 模型文件缺失或推理库未安装，\n"
                    "请改用「FSR 频率选择性」算法。")
                return
        # 各类算法统一走后台线程 + 实时进度上报：
        # 经典/FSR 报阶段/通道进度，AI 按抹除区瓦片报真实进度。
        # 修复强度越大耗时越长，后台线程化后界面不再假死，
        # 且提供进度条与「取消处理」按钮。
        self._start_photo_work(
            self.editor.image, self.editor.get_mask().copy(),
            engine, self.radius.get(), self._method_flag(),
            self.enhance.get())

    def _start_photo_work(self, img, mask, engine, radius, method, enhance):
        """启动后台修复线程（AI / 经典 / FSR），并把进度经消息队列回报到 UI"""
        self._working = True
        self._cur_engine = engine
        self._cancel_evt.clear()

        self.open_btn.set_enabled(False)
        self.undo_btn.set_enabled(False)
        self.reset_btn.set_enabled(False)
        self.cancel_btn.pack(fill="x", pady=(8, 0))
        self.cancel_btn.config(state="normal")
        self.progress.config(maximum=1, value=0)
        self.pct_label.config(text="0%")
        if engine == "ai":
            self.status.config(text="AI 修复中（首次需加载模型约 10 秒，"
                                    "单块推理期间进度条可能短暂停住；"
                                    "可随时点击「取消处理」）……")
        elif engine == "fsr":
            self.status.config(text="FSR 重建中（逐通道重建，较慢，"
                                    "可随时点击「取消处理」）……")
        else:
            self.status.config(text="正在修复（强度越高越慢，"
                                    "可随时点击「取消处理」）……")

        def cb(done, total):
            self.msg_queue.put(("progress", done, total))

        def worker():
            try:
                result = processor.inpaint_image(
                    img, mask, radius=radius, method=method, enhance=enhance,
                    engine=engine, progress_cb=cb,
                    cancel_check=self._cancel_evt.is_set)
                self.msg_queue.put(("done", result))
            except processor.ProcessingCancelled:
                self.msg_queue.put(("cancelled",))
            except Exception as e:
                self.msg_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll_photo)

    def _poll_photo(self):
        """主线程轮询后台修复线程的消息，刷新进度条 / 状态"""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg[0] == "progress":
                    _, done, total = msg
                    if total > 0:
                        pct = max(0.0, min(1.0, done / total))
                        self.progress.config(maximum=total, value=done)
                        self.pct_label.config(text=f"{pct:.0%}")
                elif msg[0] == "done":
                    self._photo_done(msg[1])
                    return
                elif msg[0] == "cancelled":
                    self._photo_done(None, cancelled=True)
                    return
                elif msg[0] == "error":
                    self._photo_done(None, error=msg[1])
                    return
        except queue.Empty:
            pass
        if self._working:
            self.after(100, self._poll_photo)

    def _photo_done(self, result, error=None, cancelled=False):
        self._working = False

        self.open_btn.set_enabled(True)
        self._sync_undo()
        self.cancel_btn.pack_forget()
        self.progress.config(value=self.progress["maximum"] or 100)
        if cancelled:
            self.status.config(text="已取消处理，可调整参数后重试")
            return
        if error is not None:
            self.status.config(text="处理失败")
            messagebox.showerror("处理失败", error)
            return
        # 结果作为新的底图，支持对残留痕迹继续二次处理
        self._history.append(self.editor.image)
        self._sync_undo()
        self.editor.set_image(result)
        if getattr(self, "_cur_engine", None) == "ai":
            self.status.config(
                text="去除完成！如有残留痕迹可继续选取后再次去除，"
                     "满意后点击「保存结果」　"
                     "小提示：AI 修复时选区尽量贴合水印轮廓，"
                     "选得越小、越贴合，效果越自然")
        else:
            self.status.config(text="去除完成！如有残留痕迹可继续选取后再次去除，"
                                    "满意后点击「保存结果」")

    def cancel_photo_work(self):
        """用户点击取消：通知后台修复线程中止（阶段/通道间生效）"""
        if not self._working:
            return
        self._cancel_evt.set()
        self.cancel_btn.config(state="disabled")
        self.status.config(text="正在取消，请稍候……")

    def _sync_undo(self):
        has = bool(self._history)
        for b in (self.undo_btn, self.reset_btn):
            b.config(state="normal" if has else "disabled")

    def undo_step(self):
        """撤销最近一次修复"""
        if not self._history:
            return
        img = self._history.pop()
        self.editor.set_image(img)
        self._sync_undo()
        self.status.config(text="已撤销一步修复"
                                + ("，已回到原图" if not self._history else ""))

    def reset_image(self):
        """丢弃全部修复，还原打开时的原图"""
        if not self._history:
            return
        img = self._history[0]
        self._history.clear()
        self.editor.set_image(img)
        self._sync_undo()
        self.status.config(text="已还原原图，可重新选取处理")


    def save_result(self):
        if self.editor.image is None:
            messagebox.showinfo("提示", "还没有可保存的结果")
            return
        base = os.path.splitext(os.path.basename(self.src_path or "result"))[0]
        path = filedialog.asksaveasfilename(
            title="保存结果", initialfile=f"{base}_去水印.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                       ("BMP", "*.bmp"), ("WebP", "*.webp")])
        if not path:
            return
        try:
            processor.save_image(self.editor.image, path)
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            return
        self.status.config(text=f"已保存：{path}")


# ======================================================================
# 页签三：清晰度增强（独立页）
# ======================================================================
class EnhanceTab(ttk.Frame):
    """图片清晰度增强：左侧预览 + 右侧增强控制"""

    RIGHT_W = 326

    def __init__(self, parent):
        super().__init__(parent)
        self.src_path = None
        self._history = []
        self._working = False
        self._cancel_evt = threading.Event()
        self._warmed = False
        self._warm_done = threading.Event()
        self._warm_done.set()
        self.msg_queue = queue.Queue()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        left = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, sticky="nsew")
        self.editor = RegionEditor(left)
        self.editor.pack(fill="both", expand=True, padx=(0, 4))
        self.editor.set_placeholder("打开一张图片开始\n\n增强清晰度、提升细节")

        right = tk.Frame(self, width=self.RIGHT_W, bg=BG)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)

        self.status = tk.Label(right, bg=BG, fg=TEXT_2, font=FONT_SM,
                               text="请先打开一张图片", anchor="w",
                               wraplength=self.RIGHT_W - 28, justify="left")
        self.status.pack(fill="x", side="bottom", padx=14, pady=(4, 12))

        c3 = Card(right, "输出")
        c3.pack(fill="x", side="bottom", padx=14, pady=(0, 10))
        self.save_btn = Button(c3.body, "保存结果", size="lg", stretch=True,
                               bg=PANEL, command=self.save_result)
        self.save_btn.pack(fill="x")

        props = ScrollFrame(right)
        props.pack(fill="both", expand=True, padx=(0, 14), pady=(14, 6))
        W = self.RIGHT_W - 14 - 28

        c0 = Card(props.inner, "操作")
        c0.pack(fill="x")
        self.open_btn = Button(c0.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        hr = tk.Frame(c0.body, bg=PANEL)
        hr.pack(fill="x", pady=(10, 0))
        self.undo_btn = Button(hr, "撤销一步", size="sm", stretch=True,
                               bg=PANEL, command=self.undo_step,
                               state="disabled")
        self.undo_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.reset_btn = Button(hr, "还原原图", size="sm", stretch=True,
                                bg=PANEL, command=self.reset_image,
                                state="disabled")
        self.reset_btn.pack(side="left", fill="x", expand=True)

        c2 = Card(props.inner, "AI 超分辨率")
        c2.pack(fill="x", pady=(12, 0))
        mf = tk.Frame(c2.body, bg=PANEL)
        mf.pack(fill="x", pady=(0, 8))
        self.sr_model = tk.StringVar(value="anime")
        tk.Label(mf, text="模型：", bg=PANEL, fg=TEXT_2,
                 font=FONT_SM).pack(side="left")
        for txt, val in [("动漫/插画", "anime"), ("通用照片", "general")]:
            tk.Radiobutton(mf, text=txt, variable=self.sr_model, value=val,
                           bg=PANEL, fg=TEXT_2, font=FONT_SM,
                           selectcolor=PANEL, activebackground=PANEL,
                           highlightthickness=0).pack(side="left", padx=(16, 0))
        self.up2_btn = Button(c2.body, "AI 放大 2×", variant="primary",
                              size="lg", stretch=True, bg=PANEL,
                              command=lambda: self.upscale_ai())
        self.up2_btn.pack(fill="x")
        self.cancel_btn = Button(c2.body, "取消", variant="danger",
                                 stretch=True, bg=PANEL,
                                 command=self.cancel_work)
        self.cancel_btn.pack(fill="x", pady=(8, 0))
        self.cancel_btn.pack_forget()
        rpb = tk.Frame(c2.body, bg=PANEL)
        rpb.pack(fill="x", pady=(10, 0))
        self.progress = ProgressBar(rpb, width=W - 8, bg=PANEL)
        self.progress.pack(side="left")
        self.pct_label = tk.Label(rpb, text="", bg=PANEL, fg=TEXT_2,
                                  font=FONT_SM, width=5, anchor="e")
        self.pct_label.pack(side="left", padx=(8, 0))
        tk.Label(c2.body,
                 text="AI 重建细节，颜色不变，清晰度大幅提升，可撤销",
                 bg=PANEL, fg=TEXT_3, font=FONT_XS,
                 wraplength=W - 8, justify="left").pack(anchor="w", pady=(6, 0))


    def open_image(self):
        path = filedialog.askopenfilename(title="选择图片",
                                          filetypes=IMAGE_FILETYPES)
        if not path:
            return
        try:
            img = processor.load_image(path)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))
            return
        self.src_path = path
        self._history.clear()
        self._sync_undo()
        self.editor.set_image(img)
        self.status.config(text=f"已打开：{os.path.basename(path)}")
        if not self._warmed:
            self._warmed = True
            self._warm_done.clear()
            threading.Thread(target=self._warmup_sr, daemon=True).start()

    def _warmup_sr(self):
        try:
            from core import sr
            sr.warmup("anime")
        except Exception:
            pass
        finally:
            self._warm_done.set()


    def upscale_ai(self):
        img = self.editor.image
        if img is None:
            messagebox.showinfo("提示", "请先打开一张图片")
            return
        if self._working:
            return
        from core import sr
        model = self.sr_model.get()
        if not sr.is_ready(model):
            messagebox.showerror("提示", "AI 模型不可用")
            return
        self._working = True
        self._cancel_evt.clear()
        self.up2_btn.set_enabled(False)
        self.open_btn.set_enabled(False)
        self.cancel_btn.pack(fill="x", pady=(8, 0))
        self.progress.config(value=0)
        self.pct_label.config(text="0%")
        self.status.config(text="AI 超分辨率放大中…")

        def worker():
            try:
                if not self._warm_done.is_set():
                    self.msg_queue.put(("stage", "正在预热 AI 模型，请稍候…"))
                    self._warm_done.wait(timeout=120)
                result = sr.upscale(
                    img,
                    progress_cb=lambda d, t: self.msg_queue.put(("progress", d, t)),
                    cancel_cb=self._cancel_evt.is_set,
                    stage_cb=lambda s: self.msg_queue.put(("stage", s)),
                    model=model)
                if result is not None:
                    self.msg_queue.put(("done", result))
                else:
                    self.msg_queue.put(("cancelled",))
            except Exception as e:
                self.msg_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll_sr)

    def _poll_sr(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg[0] == "progress":
                    _, done, total = msg
                    if total > 0:
                        pct = max(0.0, min(1.0, done / total))
                        self.progress.config(maximum=total, value=done)
                        self.pct_label.config(text=f"{pct:.0%}")
                elif msg[0] == "stage":
                    self.status.config(text=msg[1])
                elif msg[0] == "done":
                    self._sr_done(msg[1])
                    return
                elif msg[0] == "cancelled":
                    self._sr_done(None, cancelled=True)
                    return
                elif msg[0] == "error":
                    self._sr_done(None, error=msg[1])
                    return
        except queue.Empty:
            pass
        if self._working:
            self.after(100, self._poll_sr)

    def _sr_done(self, result, error=None, cancelled=False):
        self._working = False
        self.up2_btn.set_enabled(True)
        self.open_btn.set_enabled(True)
        self._sync_undo()
        self.cancel_btn.pack_forget()
        self.progress.config(value=0)
        self.pct_label.config(text="")
        if cancelled:
            self.status.config(text="已取消")
            return
        if error is not None:
            self.status.config(text=f"失败：{error}")
            messagebox.showerror("错误", error)
            return
        img = self.editor.image
        self._history.append(img)
        self._sync_undo()
        self.editor.set_image(result)
        h, w = result.shape[:2]
        self.status.config(text=f"AI 放大完成 → {w}×{h}，可撤销")

    def cancel_work(self):
        if self._working:
            self._cancel_evt.set()

    def save_result(self):
        if self.editor.image is None:
            messagebox.showinfo("提示", "还没有可保存的结果")
            return
        base = os.path.splitext(os.path.basename(self.src_path or "图片"))[0]
        path = filedialog.asksaveasfilename(
            title="保存结果", initialfile=f"{base}_增强.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"),
                       ("BMP", "*.bmp")])
        if not path:
            return
        try:
            processor.save_image(self.editor.image, path)
            self.status.config(text=f"已保存：{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _sync_undo(self):
        has = bool(self._history)
        for b in (self.undo_btn, self.reset_btn):
            b.config(state="normal" if has else "disabled")

    def undo_step(self):
        if not self._history:
            return
        img = self._history.pop()
        self.editor.set_image(img)
        self._sync_undo()
        self.status.config(text="已撤销"
                                + ("，已回到原图" if not self._history else ""))

    def reset_image(self):
        if not self._history:
            return
        img = self._history[0]
        self._history.clear()
        self.editor.set_image(img)
        self._sync_undo()
        self.status.config(text="已还原原图")


# ======================================================================
# 主程序：顶栏品牌区 + 左侧分组导航（功能页注册在 ui/tabs/__init__.py）
# ======================================================================


def main():
    install_crash_log()
    install_debug_log()
    enable_high_dpi()
    root = tk.Tk()
    root.title(APP_TITLE)
    apply_theme(root)
    icon_img = app_icon_photo(32)      # 窗口/任务栏图标
    if icon_img is not None:
        root.iconphoto(True, icon_img)
        root._icon_ref = icon_img      # 防 GC
    try:
        scale = root.winfo_fpixels("1i") / 96.0
    except Exception:
        scale = 1.0
    root.geometry(f"{int(1440 * scale)}x{int(900 * scale)}")
    root.minsize(int(1240 * scale), int(780 * scale))

    # ---- 顶部导航条：极光底纹 + 流光分隔线 + 呼吸光晕 Logo ----
    TOP_H = 70
    topbar = tk.Canvas(root, height=TOP_H, bg=TOPBAR, highlightthickness=0,
                       bd=0)
    topbar.pack(fill="x")
    topbar.pack_propagate(False)
    AuroraLayer(topbar, size=380, peak=0.06, interval=120, speed=0.7)

    # Logo：底光晕缓慢呼吸 + 应用图标
    logo_box = tk.Canvas(topbar, width=52, height=52, bg=TOPBAR,
                         highlightthickness=0, bd=0)
    glows = [orb_photo(52, GLOW_C, p) for p in (0.20, 0.30, 0.40, 0.30)]
    glow_item = logo_box.create_image(26, 26, image=glows[0])
    logo_img = app_icon_photo(34)
    if logo_img is not None:
        logo_box.create_image(26, 26, image=logo_img)
        logo_box._icon = logo_img       # 防 GC
    else:                               # 图标缺失时的兜底绘制
        p1 = rounded_photo(23, 23, 8, grad=GRAD)
        p2 = rounded_photo(23, 23, 8, fill=BRAND_DK)
        logo_box.create_image(5, 20, anchor="nw", image=p2)
        logo_box.create_image(16, 7, anchor="nw", image=p1)
        logo_box._p1, logo_box._p2 = p1, p2
    Breath(logo_box, glow_item, glows, ms=780)
    topbar.create_window(18, TOP_H / 2, window=logo_box, anchor="w",
                         tags="win")
    # 品牌名直接画在画布上，让极光从字缝里透出来
    topbar.create_text(82, TOP_H / 2 - 11, text=APP_NAME, anchor="w",
                       fill=TEXT, font=FONT_XL, tags="win")
    topbar.create_text(83, TOP_H / 2 + 12, text="照片 AI 修复",
                       anchor="w", fill=TEXT_3, font=FONT_XS, tags="win")

    # 功能页注册表 + 左侧分组导航。页面懒加载：首次切换才构建，
    # 11 个页面若启动时全部构建会让窗口明显卡顿。
    from ui.tabs import NAV_GROUPS, create
    from ui.widgets import SideNav

    views = {}
    current_view = ["photo"]

    def switch(key):
        current_view[0] = key
        w = views.get(key)
        if w is None:                   # 首次进入的页面现场构建
            w = create(key, body)
            views[key] = w
        for k, v in views.items():
            if k == key:
                v.pack(fill="both", expand=True)
            else:
                v.pack_forget()
        nav.set_active(key)

    # 右侧：就绪呼吸灯
    right = tk.Frame(topbar, bg=TOPBAR)
    dot_cv = tk.Canvas(right, width=18, height=18, bg=TOPBAR,
                       highlightthickness=0, bd=0)
    dots = [orb_photo(18, BRAND, p) for p in (0.35, 0.55, 0.75, 0.55)]
    dot_item = dot_cv.create_image(9, 9, image=dots[0])
    Breath(dot_cv, dot_item, dots, ms=620)
    dot_cv.pack(side="right", padx=(0, 8))
    tk.Label(right, text="就绪", bg=TOPBAR, fg=TEXT_2,
             font=FONT_SM).pack(side="right", padx=(0, 16))
    right_win = topbar.create_window(0, TOP_H / 2, window=right, anchor="e",
                                     tags="win")

    # 顶栏底：竖向渐变 + 底部极光分隔线（线上还有一道循环流光）
    top_state = {"w": 0, "sx": -160}
    _top_job = [None]

    def _paint_topbar_now():
        _top_job[0] = None
        w = max(topbar.winfo_width(), 480)
        top_state["w"] = w
        topbar.delete("base")
        topbar.delete("chrome")
        base = grad_photo(w, TOP_H, ("#FFFFFF", "#F8FAFD"))
        topbar.create_image(0, 0, anchor="nw", image=base, tags="base")
        topbar._base = base
        topbar.tag_lower("base")
        line = grad_photo(w, 2, GRAD)
        topbar.create_image(0, TOP_H - 2, anchor="nw", image=line,
                            tags="chrome")
        topbar._line = line
        topbar._shine_ph = shine_photo(140, 2, "#FFFFFF", 0.95)
        topbar.create_image(top_state["sx"], TOP_H - 2, anchor="nw",
                            image=topbar._shine_ph, tags=("chrome", "shine"))
        # 右侧徽章组贴右边缘
        topbar.coords(right_win, w - 20, TOP_H / 2)
        topbar.tag_raise("win")

    def paint_topbar(_e=None):
        """拖拽缩放时 <Configure> 高频触发，合并为 40ms 一次重绘"""
        if _top_job[0] is not None:
            try:
                topbar.after_cancel(_top_job[0])
            except Exception:
                pass
        _top_job[0] = topbar.after(40, _paint_topbar_now)

    def shine_tick():
        top_state["sx"] += 7
        if top_state["sx"] > top_state["w"] + 20:
            top_state["sx"] = -160
        try:
            if topbar.winfo_ismapped():     # 最小化/隐藏时不空转
                topbar.coords("shine", top_state["sx"], TOP_H - 2)
        except tk.TclError:
            return
        topbar.after(40, shine_tick)

    topbar.bind("<Configure>", paint_topbar)
    topbar.after(200, paint_topbar)
    topbar.after(600, shine_tick)

    # ---- 左侧分组导航 + 功能视图（只建一次，保留各自状态）----
    side = tk.Frame(root, bg=BG)
    side.pack(side="left", fill="y", padx=(8, 4), pady=(8, 8))
    nav = SideNav(side, NAV_GROUPS, on_pick=switch)
    nav.pack(fill="y", expand=True)

    body = ttk.Frame(root)
    body.pack(side="left", fill="both", expand=True)

    # photo / enhance 是本文件原有页面，不走 ui.tabs.create()
    views["photo"] = PhotoTab(body)
    views["enhance"] = EnhanceTab(body)

    switch("photo")

    def on_close():
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        root.state("zoomed")      # Windows 下默认最大化
    except tk.TclError:
        pass

    # 后台线程回调投递的消费定时器（见 ui_post / _DISPATCH_Q）
    def _drain_dispatch():
        try:
            while True:
                fn = _DISPATCH_Q.get_nowait()
                try:
                    fn()
                except Exception:
                    pass
        except queue.Empty:
            pass
        root.after(40, _drain_dispatch)

    root.after(40, _drain_dispatch)
    root.mainloop()


def self_test():
    """命令行自检：去水印工具.exe --selftest，验证 AI 引擎链路"""
    import sys
    import numpy as np
    print("模型检查...", flush=True)
    from core import lama
    if not lama.is_ready():
        print("[FAIL] AI 模型不可用")
        return 1
    print("[OK] 模型就绪，推理后端：%s" % lama.provider_name(), flush=True)
    import cv2
    img = np.zeros((720, 1280, 3), np.uint8)
    cv2rand = np.random.default_rng(0)
    img[:] = (cv2rand.integers(80, 120, (720, 1280, 3))).astype(np.uint8)
    cv2.putText(img, "TEST", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (255, 255, 255), 6, cv2.LINE_AA)
    mask = np.zeros((720, 1280), np.uint8)
    cv2.putText(mask, "TEST", (500, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                255, 12, cv2.LINE_AA)
    import time as _t
    t0 = _t.time()
    out = lama.inpaint(img, mask)
    dt = _t.time() - t0
    bb = int((img[mask > 0] > 245).sum())
    ab = int((out[mask > 0] > 245).sum())
    print("[OK] AI 修复完成：%.1fs（含首次模型加载），输出尺寸 %s"
          % (dt, out.shape), flush=True)
    print("[OK] 水印清除：高亮像素 %d -> %d" % (bb, ab), flush=True)
    if ab > bb * 0.1:
        print("[FAIL] 水印残留过多（%d -> %d），填充链路异常" % (bb, ab))
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(self_test())
    main()
