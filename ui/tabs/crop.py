# -*- coding: utf-8 -*-
"""裁剪与尺寸页：常用比例、证件照尺寸、智能取景。"""

import tkinter as tk

from app import PANEL, Button
from core import basic

from ..base import TabBase
from ..widgets import OptionChips, ParamRow

RATIO_OPTS = [(k, k) for k in basic.RATIOS.keys()]
ALIGN_OPTS = [("auto", "智能取景"), ("center", "居中")]
SCALE_OPTS = [("none", "不缩放"), ("percent", "按百分比"),
              ("width", "指定宽"), ("height", "指定高"), ("long", "长边")]
ID_OPTS = [(k, k.split(" ")[0]) for k in basic.ID_SIZES.keys()]


class CropTab(TabBase):
    PLACEHOLDER = ("打开一张图片\n\n"
                   "选比例自动裁，或直接指定输出尺寸")

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.ratio = tk.StringVar(value="原始")
        self.align = tk.StringVar(value="auto")
        self.scale_mode = tk.StringVar(value="none")
        self.percent = tk.IntVar(value=100)
        self.px = tk.IntVar(value=1080)
        self.id_size = tk.StringVar(value=list(basic.ID_SIZES.keys())[0])
        self._px_row = None
        self._pct_row = None
        self._id_row = None
        self._build()

    # -- 界面 ---------------------------------------------------------
    def _build(self):
        c = self.card("操作")
        c.pack(fill="x")
        self.open_btn = Button(c.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        self.apply_btn = Button(c.body, "应用裁剪", size="lg", stretch=True,
                                bg=PANEL, command=self.apply)
        self.apply_btn.pack(fill="x", pady=(10, 0))

        c1 = self.card("裁剪比例")
        c1.pack(fill="x", pady=(10, 0))
        OptionChips(c1.body, RATIO_OPTS, self.ratio,
                    command=self.refresh, bg=PANEL,
                    per_row=3).pack(fill="x", pady=(2, 6))
        tk.Label(c1.body, text="取景方式", bg=PANEL, fg="#5B6472",
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")
        OptionChips(c1.body, ALIGN_OPTS, self.align, command=self.refresh,
                    bg=PANEL, per_row=2).pack(fill="x", pady=(4, 0))

        c2 = self.card("输出尺寸")
        c2.pack(fill="x", pady=(10, 0))
        OptionChips(c2.body, SCALE_OPTS, self.scale_mode,
                    command=self._on_scale_mode, bg=PANEL,
                    per_row=3).pack(fill="x", pady=(2, 6))
        self._pct_row = ParamRow(c2.body, "缩放比例", self.percent, 10, 400,
                                 command=self.refresh, bg=PANEL,
                                 width=self.W, fmt="{:.0f}%")
        self._px_row = tk.Frame(c2.body, bg=PANEL)
        tk.Label(self._px_row, text="像素", bg=PANEL, fg="#5B6472",
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        tk.Spinbox(self._px_row, from_=16, to=20000, width=8,
                   textvariable=self.px, bg="#F2F4F8", relief="flat",
                   font=("Microsoft YaHei UI", 9),
                   command=self.refresh).pack(side="right")

        c3 = self.card("证件照快捷尺寸")
        c3.pack(fill="x", pady=(10, 0))
        OptionChips(c3.body, ID_OPTS, self.id_size, bg=PANEL,
                    per_row=3).pack(fill="x", pady=(2, 6))
        self.id_btn = Button(c3.body, "套用该尺寸输出", stretch=True,
                             bg=PANEL, command=self.apply_id)
        self.id_btn.pack(fill="x")
        self._on_scale_mode()

    # -- 行为 ---------------------------------------------------------
    def _on_scale_mode(self):
        m = self.scale_mode.get()
        if self._pct_row:
            self._pct_row.pack_forget()
        if self._px_row:
            self._px_row.pack_forget()
        if m == "percent":
            self._pct_row.pack(fill="x")
        elif m in ("width", "height", "long"):
            self._px_row.pack(fill="x")
        self.refresh()

    def _snapshot(self):
        """后台线程不能读 Tk 变量，统一在主线程拍快照。"""
        return self.snapshot(("ratio", self.ratio), ("align", self.align),
                             ("scale_mode", self.scale_mode),
                             ("percent", self.percent), ("px", self.px))

    def refresh(self):
        self.schedule_preview(self._preview)

    def _crop_only(self, img, snap):
        if img is None:
            return None
        r = basic.RATIOS.get(snap["ratio"])
        if r:
            img, _box = basic.crop_ratio(img, r, snap["align"])
        return img

    def _apply_scale(self, img, snap):
        m = snap["scale_mode"]
        if m == "none":
            return img
        if m == "percent":
            return basic.resize_img(img, "percent", snap["percent"])
        return basic.resize_img(img, m, snap["px"])

    def _preview(self):
        src = self.src
        if src is None:
            return
        snap = self._snapshot()
        self.run_preview(
            lambda: self._apply_scale(self._crop_only(src, snap), snap))

    def apply(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        snap = self._snapshot()
        img = self._apply_scale(self._crop_only(self.src, snap), snap)
        self.push_history(self.result if self.result is not None else self.src)
        self.result = img
        self._show_result(img)
        self.set_status(f"裁剪完成 · 输出 {img.shape[1]}×{img.shape[0]}")

    def apply_id(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        w, h = basic.ID_SIZES[self.id_size.get()]
        # 先按目标比例裁，再缩放到精确像素，保证不变形
        img, _box = basic.crop_ratio(self.src, w / float(h),
                                     self.align.get())
        img = cv_resize(img, w, h)
        self.push_history(self.result if self.result is not None else self.src)
        self.result = img
        self._show_result(img)
        self.set_status(f"{self.id_size.get()} 已生成 · {w}×{h}")


def cv_resize(img, w, h):
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
