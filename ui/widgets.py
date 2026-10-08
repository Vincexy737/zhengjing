# -*- coding: utf-8 -*-
"""
扩展 UI 组件：
  - SideNav    分组侧边导航（功能变多后顶栏横排放不下）
  - CompareView 原图 / 效果滑动对比（所有处理页共用）
  - FileList   批量文件列表（缩略图 + 逐行状态）
  - ParamRow   滑块 + 标题 + 数值的一体化参数行
"""

import os

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from app import (BG, BG_WELL, PANEL, PANEL_2, PANEL_3, PANEL_4, TEXT, TEXT_2,
                 TEXT_3, BRAND, BRAND_DK, GRAD, GLOW_C,
                 FONT_XS, FONT_SM, FONT_BOLD,
                 ui_font, rounded_photo, orb_photo, mix)


# ======================================================================
# 侧边导航
# ======================================================================
class _NavItem(tk.Canvas):
    """单条导航项：选中态为淡蓝胶囊 + 左侧主色竖条。"""

    H = 34

    def __init__(self, parent, text, on_pick, bg=PANEL):
        self._text = text
        self._bg = bg
        self._active = False
        self._hover = False
        self._on_pick = on_pick
        self._cw = 196
        super().__init__(parent, width=self._cw, height=self.H, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._draw()
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", lambda e: self._on_pick())
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, e):
        if e.width > 8 and abs(e.width - self._cw) > 1:
            self._cw = e.width
            self.configure(width=self._cw)
            self._draw()

    def _draw(self):
        w, h = self._cw, self.H
        self.delete("all")
        if self._active:
            ph = rounded_photo(w - 12, h, 8, fill=BRAND_DK)
            self.create_image(6, 0, anchor="nw", image=ph)
            self._ph = ph
            # 左侧主色竖条
            bar = rounded_photo(3, h - 12, 1, fill=BRAND, pad=0)
            self.create_image(8, 6, anchor="nw", image=bar)
            self._bar = bar
            fg, font = TEXT, ui_font(("Microsoft YaHei UI", 10, "bold"))
        elif self._hover:
            ph = rounded_photo(w - 12, h, 8, fill=PANEL_3)
            self.create_image(6, 0, anchor="nw", image=ph)
            self._ph = ph
            fg, font = TEXT, ui_font(("Microsoft YaHei UI", 10))
        else:
            fg, font = TEXT_2, ui_font(("Microsoft YaHei UI", 10))
        self.create_text(20, h / 2 + 0.5, text=self._text, anchor="w",
                         fill=fg, font=font)

    def _set_hover(self, on):
        if self._hover != on:
            self._hover = on
            self._draw()

    def set_active(self, on):
        if self._active != on:
            self._active = on
            self._draw()

    def set_width(self, w):
        self._cw = w
        self.configure(width=w)
        self._draw()


class SideNav(tk.Frame):
    """分组侧边导航。groups: [(分组标题, [(key, 显示名), ...]), ...]"""

    def __init__(self, parent, groups, on_pick, bg=PANEL, width=196):
        super().__init__(parent, bg=bg, width=width)
        self.pack_propagate(False)
        self._items = {}
        self._cwidth = width

        for gi, (title, entries) in enumerate(groups):
            if gi > 0:
                tk.Frame(self, bg=bg, height=10).pack(fill="x")
            if title:
                tk.Label(self, text=title, bg=bg, fg=TEXT_3,
                         font=ui_font(("Microsoft YaHei UI", 8)),
                         anchor="w").pack(fill="x", padx=(18, 0), pady=(0, 4))
            for key, label in entries:
                it = _NavItem(self, label, lambda k=key: on_pick(k), bg=bg)
                it.pack(fill="x", pady=1)
                self._items[key] = it
        tk.Frame(self, bg=bg).pack(fill="both", expand=True)

    def set_active(self, key):
        for k, it in self._items.items():
            it.set_active(k == key)

    def set_width(self, w):
        self._cwidth = w
        self.configure(width=w)
        for it in self._items.values():
            it.set_width(w)


# ======================================================================
# 滑动对比视图
# ======================================================================
class CompareView(tk.Canvas):
    """原图 / 效果对比：中间分割线可拖动，也可整体切换只看某一边。

    拖动时先移动分割线保证跟手，再用 30ms 节流重绘合成图，
    避免大图下每帧都做一次 PhotoImage 转换导致卡顿。
    """

    def __init__(self, parent, bg=BG_WELL, **kw):
        super().__init__(parent, bg=bg, highlightthickness=0, bd=0, **kw)
        self.before = None            # BGR ndarray
        # 结果图不能命名为 after：Tkinter 用 after() 做定时器，
        # 实例属性会把方法覆盖掉，导致所有定时重绘失效。
        self.after_img = None         # BGR ndarray
        self.ratio = 0.5
        self.mode = "slider"          # slider | before | after
        self._photo = None
        self._job = None
        self._text = "拖入或打开一张图片"
        self._bind_events()
        self.bind("<Configure>", lambda e: self._schedule())

    def _bind_events(self):
        self.bind("<ButtonPress-1>", self._down)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._up)

    # -- 数据 --------------------------------------------------------
    def set_before(self, bgr):
        self.before = None if bgr is None else np.asarray(bgr)
        self._schedule()

    def set_after(self, bgr):
        self.after_img = None if bgr is None else np.asarray(bgr)
        self._schedule()

    def set_mode(self, mode):
        self.mode = mode
        self._schedule()

    def has_pair(self):
        return self.before is not None and self.after_img is not None

    def clear(self):
        self.before = self.after_img = None
        self._photo = None
        self.delete("all")
        self._draw_placeholder()

    # -- 交互 --------------------------------------------------------
    def _down(self, e):
        if self.mode == "slider" and self.has_pair():
            self.ratio = self._ratio_at(e.x)
            self._draw_splitter()
            self._schedule()

    def _move(self, e):
        if self.mode == "slider" and self.has_pair():
            self.ratio = self._ratio_at(e.x)
            self._draw_splitter()
            self._schedule()

    def _up(self, _e):
        self._render()

    def _ratio_at(self, x):
        box = self._img_box()
        if not box:
            return self.ratio
        x0, _y0, w, _h = box
        return max(0.0, min(1.0, (x - x0) / max(1.0, w)))

    def _img_box(self):
        """图片显示区域 (x, y, w, h)；无图时返回 None。"""
        img = self.after_img if self.after_img is not None else self.before
        if img is None:
            return None
        cw, ch = self.winfo_width(), self.winfo_height()
        if cw < 8 or ch < 8:
            return None
        ih, iw = img.shape[:2]
        s = min(cw / float(iw), ch / float(ih), 1.0) if iw and ih else 1.0
        dw, dh = max(1, int(iw * s)), max(1, int(ih * s))
        return ((cw - dw) // 2, (ch - dh) // 2, dw, dh)

    # -- 渲染 --------------------------------------------------------
    def _schedule(self):
        if self._job is not None:
            return
        self._job = self.after(30, self._render)

    def _render(self):
        self._job = None
        self.delete("all")
        box = self._img_box()
        if not box:
            self._draw_placeholder()
            return
        x0, y0, dw, dh = box
        shown = self._compose(dw, dh)
        if shown is None:
            self._draw_placeholder()
            return
        self._photo = ImageTk.PhotoImage(shown)
        self.create_image(x0, y0, anchor="nw", image=self._photo)
        if self.has_pair() and self.mode == "slider":
            self._draw_splitter()
            self._draw_tags(x0, y0, dw, dh)
        elif self.has_pair():
            self._draw_tags(x0, y0, dw, dh)

    def _compose(self, dw, dh):
        """按当前模式合成要显示的 PIL 图。"""
        pair = self.has_pair()
        if not pair:
            img = self.after_img if self.after_img is not None else self.before
            if img is None:
                return None
            return Image.fromarray(cv2.cvtColor(
                _resize_to(img, dw, dh), cv2.COLOR_BGR2RGB))
        if self.mode == "before":
            return Image.fromarray(cv2.cvtColor(
                _resize_to(self.before, dw, dh), cv2.COLOR_BGR2RGB))
        if self.mode == "after":
            return Image.fromarray(cv2.cvtColor(
                _resize_to(self.after_img, dw, dh), cv2.COLOR_BGR2RGB))
        a = _resize_to(self.before, dw, dh)
        b = _resize_to(self.after_img, dw, dh)
        split = int(dw * self.ratio)
        out = np.empty_like(b)
        out[:, :split] = a[:, :split]
        out[:, split:] = b[:, split:]
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    def _draw_splitter(self):
        box = self._img_box()
        if not box:
            return
        x0, y0, dw, dh = box
        sx = x0 + int(dw * self.ratio)
        self.create_line(sx, y0, sx, y0 + dh, fill="#FFFFFF", width=2)
        self.create_oval(sx - 9, y0 + dh / 2 - 9, sx + 9, y0 + dh / 2 + 9,
                         fill="#FFFFFF", outline=BRAND, width=2)
        self.create_text(sx - 13, y0 + dh / 2, text="◀", fill=BRAND,
                         font=ui_font(("Microsoft YaHei UI", 8)))
        self.create_text(sx + 13, y0 + dh / 2, text="▶", fill=BRAND,
                         font=ui_font(("Microsoft YaHei UI", 8)))

    def _draw_tags(self, x0, y0, dw, dh):
        if self.mode == "before":
            self._tag("原图", x0 + 10, y0 + 10)
        elif self.mode == "after":
            self._tag("效果", x0 + 10, y0 + 10)
        else:
            self._tag("原图", x0 + 10, y0 + 10)
            self._tag("效果", x0 + dw - 10, y0 + 10, anchor="e")

    def _tag(self, text, x, y, anchor="w"):
        w = len(text) * 13 + 16
        ox = x if anchor == "w" else x - w
        # 半透明黑底用矩形 + stipple 实现（image 不支持 stipple）
        self.create_rectangle(ox, y, ox + w, y + 20, fill="#1A1A1A",
                              stipple="gray50", outline="")
        self.create_text(ox + w / 2, y + 10, text=text, fill="#FFFFFF",
                         font=ui_font(("Microsoft YaHei UI", 9, "bold")))

    def _draw_placeholder(self):
        cw, ch = max(self.winfo_width(), 10), max(self.winfo_height(), 10)
        self.create_text(cw / 2, ch / 2, text=self._text, fill=TEXT_3,
                         font=ui_font(("Microsoft YaHei UI", 11)),
                         justify="center")

    def set_placeholder(self, text):
        self._text = text
        if not self.has_pair() and self.before is None:
            self.delete("all")
            self._draw_placeholder()


def _resize_to(img, dw, dh):
    ih, iw = img.shape[:2]
    if iw == dw and ih == dh:
        return img
    interp = cv2.INTER_AREA if (dw < iw or dh < ih) else cv2.INTER_LINEAR
    return cv2.resize(img, (dw, dh), interpolation=interp)


# ======================================================================
# 胶囊选项组
# ======================================================================
class _Chip(tk.Canvas):
    H = 26

    def __init__(self, parent, text, on_pick, bg=PANEL):
        self._text = text
        self._on = False
        self._bg = bg
        self._cw = len(text) * 13 + 22
        super().__init__(parent, width=self._cw, height=self.H, bg=bg,
                         highlightthickness=0, bd=0, cursor="hand2")
        self._draw()
        self.bind("<ButtonRelease-1>", lambda e: on_pick())

    def _draw(self):
        self.delete("all")
        if self._on:
            ph = rounded_photo(self._cw, self.H, self.H // 2, fill=BRAND_DK,
                               outline=BRAND, width=1)
            fg = BRAND
            font = ui_font(("Microsoft YaHei UI", 9, "bold"))
        else:
            ph = rounded_photo(self._cw, self.H, self.H // 2, fill=PANEL_2)
            fg = TEXT_2
            font = ui_font(("Microsoft YaHei UI", 9))
        self.create_image(0, 0, anchor="nw", image=ph)
        self._ph = ph
        self.create_text(self._cw / 2, self.H / 2 + 0.5, text=self._text,
                         fill=fg, font=font)

    def set_on(self, on):
        if self._on != on:
            self._on = on
            self._draw()


class OptionChips(tk.Frame):
    """互斥胶囊选项组。options: [(value, 显示名), ...]，自动换行。"""

    def __init__(self, parent, options, variable, command=None, bg=PANEL,
                 per_row=None):
        super().__init__(parent, bg=bg)
        self._var = variable
        self._cmd = command
        self._bg = bg
        self._chips = {}
        # 注意：不能叫 _options，那是 Tkinter 打包布局时要调用的方法名
        self._opts = options
        self._per_row = per_row or max(1, len(options))
        self._build()

    def _build(self):
        for w in self.winfo_children():
            w.destroy()
        self._chips = {}
        row = None
        for i, (val, label) in enumerate(self._opts):
            if i % self._per_row == 0:
                row = tk.Frame(self, bg=self._bg)
                row.pack(fill="x", pady=(0 if i == 0 else 4, 0))
            c = _Chip(row, label, lambda v=val: self._pick(v), bg=self._bg)
            c.pack(side="left", padx=(0, 6))
            self._chips[val] = c
        self._sync()

    def _pick(self, val):
        self._var.set(val)
        self._sync()
        if self._cmd:
            self._cmd()

    def _sync(self):
        cur = self._var.get()
        for k, c in self._chips.items():
            c.set_on(str(k) == str(cur))

    def set_options(self, options):
        self._opts = options
        self._build()


# ======================================================================
# 批量文件列表
# ======================================================================
STATUS_STYLE = {
    "wait": ("待处理", TEXT_3),
    "running": ("处理中…", BRAND),
    "done": ("完成", "#2FA36B"),
    "error": ("失败", "#D9534F"),
    "skip": ("已跳过", TEXT_3),
    "cancel": ("已取消", TEXT_3),
}


class FileList(tk.Frame):
    """批量任务列表：缩略图 + 文件名 + 状态，支持逐行更新。"""

    def __init__(self, parent, bg=PANEL, height=200):
        super().__init__(parent, bg=bg)
        self._bg = bg
        self.rows = []
        self._paths = []
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0,
                                 height=height)
        sb = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._inner = tk.Frame(self._canvas, bg=bg)
        self._cwin = self._canvas.create_window((0, 0), window=self._inner,
                                               anchor="nw")
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self._inner.bind("<Configure>", self._on_cfg)
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfigure(
                              self._cwin, width=e.width))
        self._canvas.bind_all("<MouseWheel>", self._on_wheel, add="+")

    def _on_cfg(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_wheel(self, e):
        """只在鼠标位于本列表上方时才滚动，避免抢走整窗的滚轮。"""
        try:
            if not self.winfo_exists() or not self.winfo_ismapped():
                return
            px, py = self.winfo_pointerxy()
            rx, ry = self.winfo_rootx(), self.winfo_rooty()
            inside = (rx <= px <= rx + self.winfo_width() and
                      ry <= py <= ry + self.winfo_height())
            if not inside:
                return
            self._canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        except (tk.TclError, Exception):
            pass

    def add(self, path):
        idx = len(self.rows)
        row = tk.Frame(self._inner, bg=self._bg)
        row.pack(fill="x", padx=6, pady=2)
        thumb = tk.Label(row, bg=self._bg, width=5)
        thumb.pack(side="left", padx=(4, 8), pady=3)
        txt = tk.Frame(row, bg=self._bg)
        txt.pack(side="left", fill="x", expand=True)
        name = tk.Label(txt, text=os.path.basename(path), bg=self._bg,
                        fg=TEXT, font=ui_font(("Microsoft YaHei UI", 9)),
                        anchor="w")
        name.pack(fill="x")
        sub = tk.Label(txt, text=path, bg=self._bg, fg=TEXT_3,
                       font=FONT_XS, anchor="w")
        sub.pack(fill="x")
        st = tk.Label(row, text="待处理", bg=self._bg, fg=TEXT_3,
                      font=ui_font(("Microsoft YaHei UI", 9)), width=8,
                      anchor="e")
        st.pack(side="right", padx=8)
        self.rows.append({"frame": row, "thumb": thumb, "status": st,
                          "sub": sub})
        self._paths.append(path)
        self._load_thumb(idx, path)
        self.update_idletasks()
        self._on_cfg(None)
        return idx

    def _load_thumb(self, idx, path):
        try:
            from core import imglib
            img = imglib.imread(path, fix_exif=False)
            imglib_fit = cv2.resize(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (36, 36),
                interpolation=cv2.INTER_AREA)
            ph = ImageTk.PhotoImage(Image.fromarray(imglib_fit))
            r = self.rows[idx]
            r["thumb"].configure(image=ph, width=36, height=36)
            r["thumb"].image = ph
        except Exception:
            pass

    def set_status(self, idx, status, message=""):
        if not (0 <= idx < len(self.rows)):
            return
        label, color = STATUS_STYLE.get(status, (status, TEXT_2))
        r = self.rows[idx]
        r["status"].configure(text=label, fg=color)
        if message:
            r["sub"].configure(text=message)

    def paths(self):
        return list(self._paths)

    def clear(self):
        for r in self.rows:
            r["frame"].destroy()
        self.rows = []
        self._paths = []
        self._on_cfg(None)

    def count(self):
        return len(self.rows)


# ======================================================================
# 参数行
# ======================================================================
class ParamRow(tk.Frame):
    """标题 + 数值 + 滑块 的紧凑参数行，command 为拖动回调。"""

    def __init__(self, parent, title, variable, from_=0, to=100,
                 command=None, bg=PANEL, fmt="{:.0f}", width=200,
                 hint=""):
        super().__init__(parent, bg=bg)
        from app import Slider
        self._var = variable
        self._fmt = fmt
        head = tk.Frame(self, bg=bg)
        head.pack(fill="x")
        tk.Label(head, text=title, bg=bg, fg=TEXT_2,
                 font=ui_font(("Microsoft YaHei UI", 9)),
                 anchor="w").pack(side="left")
        self.val = tk.Label(head, text=fmt.format(variable.get()), bg=bg,
                            fg=TEXT, font=ui_font(("Microsoft YaHei UI", 9,
                                                   "bold")))
        self.val.pack(side="right")

        def _on_change():
            self.val.configure(text=self._fmt.format(self._var.get()))
            if command:
                command()

        self.slider = Slider(self, from_=from_, to=to, variable=variable,
                             width=width, bg=bg, command=_on_change)
        self.slider.pack(fill="x", pady=(2, 0))
        if hint:
            tk.Label(self, text=hint, bg=bg, fg=TEXT_3, font=FONT_XS,
                     anchor="w").pack(fill="x")

    def set_enabled(self, on):
        state = "normal" if on else "disabled"
        try:
            self.slider.configure(state=state)
        except tk.TclError:
            pass
