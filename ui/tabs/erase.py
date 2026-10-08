# -*- coding: utf-8 -*-
"""物体消除页：擦掉路人、电线、杂物、文字。

与去水印页的区别是：这里主打「任意物体」，默认用 AI 修复引擎，
并且提供文字自动检测，截图抹字、海报删元素都能用。
"""

import tkinter as tk

from app import PANEL, Button, ui_post
from core import lama, processor

from ..base import NO_IMAGE, TabBase
from ..widgets import OptionChips, ParamRow

TOOLS = [("rect", "矩形框选"), ("brush", "画笔涂抹")]
ENGINES = [("ai", "AI 修复"), ("classic", "经典快速"), ("fsr", "FSR 重建")]
ENGINE_TIP = {
    "ai": "LaMa 模型：纹理与结构还原最完整，适合大面积物体",
    "classic": "经典扩散修复：最快，适合小面积或纯色背景",
    "fsr": "频率选择性重建：无需模型，质感自然，速度较慢",
}


class EraseTab(TabBase):
    PREVIEW = "region"
    PLACEHOLDER = ("打开一张图片\n\n"
                   "用画笔涂抹要消除的物体，再点「消除选中区域」")

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.engine = tk.StringVar(value="ai" if lama.is_ready() else "classic")
        self.sens = tk.IntVar(value=55)
        self.radius = tk.IntVar(value=5)
        self._tip = None
        self._build()
        if not lama.is_ready():
            self.engine.set("classic")
            self._eng._sync()

    # -- 界面 ---------------------------------------------------------
    def _build(self):
        c = self.card("操作")
        c.pack(fill="x")
        self.open_btn = Button(c.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        self.erase_btn = Button(c.body, "消除选中区域", size="lg",
                                stretch=True, bg=PANEL,
                                command=self.erase)
        self.erase_btn.pack(fill="x", pady=(10, 0))
        row = tk.Frame(c.body, bg=PANEL)
        row.pack(fill="x", pady=(8, 0))
        self.detect_btn = Button(row, "自动检测文字", stretch=True, bg=PANEL,
                                 command=self.auto_text)
        self.detect_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.clear_btn = Button(row, "清除选区", stretch=True, bg=PANEL,
                                command=self.clear_mask)
        self.clear_btn.pack(side="left", fill="x", expand=True)

        c1 = self.card("涂抹工具")
        c1.pack(fill="x", pady=(10, 0))
        OptionChips(c1.body, TOOLS, self.view.tool, bg=PANEL,
                    per_row=2).pack(fill="x", pady=(2, 6))
        ParamRow(c1.body, "画笔大小", self.view.brush_size, 5, 200,
                 bg=PANEL, width=self.W).pack(fill="x")

        c2 = self.card("修复引擎")
        c2.pack(fill="x", pady=(10, 0))
        self._eng = OptionChips(c2.body, ENGINES, self.engine,
                                command=self._on_engine, bg=PANEL, per_row=3)
        self._eng.pack(fill="x", pady=(2, 4))
        self._tip = tk.Label(c2.body, text=ENGINE_TIP[self.engine.get()],
                             bg=PANEL, fg="#99A1AF",
                             font=("Microsoft YaHei UI", 8), anchor="w",
                             wraplength=self.W, justify="left")
        self._tip.pack(fill="x")
        ParamRow(c2.body, "边缘扩散", self.radius, 1, 20, bg=PANEL,
                 width=self.W,
                 hint="让修复区域向外多啃一点，接缝更自然").pack(
            fill="x", pady=(6, 0))

        c3 = self.card("文字检测")
        c3.pack(fill="x", pady=(10, 0))
        ParamRow(c3.body, "检测灵敏度", self.sens, 0, 100, bg=PANEL,
                 width=self.W,
                 hint="调高会检测到更多细碎文字，也可能误检").pack(fill="x")

    def _on_engine(self):
        self._tip.configure(text=ENGINE_TIP.get(self.engine.get(), ""))

    # -- 行为 ---------------------------------------------------------
    def clear_mask(self):
        self.view.clear_mask()
        self.set_status("已清除选区")

    def auto_text(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        snap = self.snapshot(("sens", self.sens))
        self.run_bg(lambda p, c: self._detect_worker(snap, p, c),
                    busy=(self.detect_btn,), start_msg="检测文字中…")

    def _detect_worker(self, snap, progress_cb, cancel):
        img = self.view.image
        if img is None:
            raise ValueError("请先打开一张图片")
        s = snap["sens"] / 100.0
        mask = processor.detect_watermark_photo(img, s)
        if progress_cb:
            progress_cb(1, 2)
        if cancel and cancel():
            return None
        if mask is None or not mask.any():
            mask = processor.detect_watermark_ocr(img, s)
        if progress_cb:
            progress_cb(2, 2)
        if mask is None or not mask.any():
            raise ValueError("没有检测到明显的文字区域，请用画笔手动涂抹")
        # 蒙版要写进画布，属于 Tk 操作，必须回到主线程
        ui_post(lambda: self._apply_detected_mask(mask))
        return NO_IMAGE

    def _apply_detected_mask(self, mask):
        self.view.mask = mask
        self.view._render()
        px = int((mask > 0).sum())
        self.set_status(f"检测到约 {px} 像素文字区域，"
                        f"点「消除选中区域」即可去除")

    def erase(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        if self.engine.get() == "fsr" and not processor.fsr_available():
            # 环境缺 FSR 就切回经典算法，别让用户白等一轮再报错
            self.engine.set("classic")
            self._eng._sync()
            self.set_status("当前 OpenCV 不带 FSR 模块，已改用经典算法")
        snap = self.snapshot(("radius", self.radius),
                             ("engine", self.engine))
        self.run_bg(lambda p, c: self._erase_worker(snap, p, c),
                    busy=(self.erase_btn, self.open_btn, self.detect_btn),
                    start_msg="消除中…")

    def _erase_worker(self, snap, progress_cb, cancel):
        img = self.view.image
        mask = self.view.get_mask().copy() if self.view.mask is not None \
            else None
        if mask is None or not mask.any():
            raise ValueError("请先用矩形或画笔选中要消除的内容")
        out = processor.inpaint_image(
            img, mask, radius=snap["radius"],
            method=processor.INPAINT_NS, engine=snap["engine"],
            progress_cb=progress_cb, cancel_check=cancel)
        px = int((mask > 0).sum())
        return out, f"已消除 {px} 像素区域"

    def on_source_loaded(self, img):
        self.set_status("涂抹要消除的部分，然后点「消除选中区域」")
