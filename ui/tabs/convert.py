# -*- coding: utf-8 -*-
"""格式转换与压缩页：JPG / PNG / WEBP 互转，按体积压缩。"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

from app import PANEL, Button, ui_post
from core import basic, imglib

from ..base import TabBase
from ..widgets import OptionChips, ParamRow

FMT_OPTS = [(f, f) for f in basic.FORMATS]


class ConvertTab(TabBase):
    PLACEHOLDER = ("打开一张图片\n\n"
                   "右侧选格式与质量，下方实时显示压缩后体积")

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.fmt = tk.StringVar(value="JPG")
        self.quality = tk.IntVar(value=85)
        self.limit = tk.BooleanVar(value=False)
        self.target = tk.IntVar(value=500)
        self._src_size = 0
        self._est_size = 0
        self._build()

    # -- 界面 ---------------------------------------------------------
    def _build(self):
        c = self.card("操作")
        c.pack(fill="x")
        self.open_btn = Button(c.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        self.export_btn = Button(c.body, "转换并保存…", size="lg",
                                 stretch=True, bg=PANEL,
                                 command=self.save_result)
        self.export_btn.pack(fill="x", pady=(10, 0))

        c1 = self.card("目标格式")
        c1.pack(fill="x", pady=(10, 0))
        OptionChips(c1.body, FMT_OPTS, self.fmt, command=self.refresh,
                    bg=PANEL, per_row=3).pack(fill="x", pady=(2, 6))
        ParamRow(c1.body, "质量", self.quality, 1, 100, command=self.refresh,
                 bg=PANEL, width=self.W,
                 hint="PNG 为无损格式，此处对应压缩级别").pack(fill="x")

        c2 = self.card("限制文件大小")
        c2.pack(fill="x", pady=(10, 0))
        row = tk.Frame(c2.body, bg=PANEL)
        row.pack(fill="x")
        tk.Checkbutton(row, text="压到指定体积以内", variable=self.limit,
                       bg=PANEL, fg="#5B6472", selectcolor="#F2F4F8",
                       activebackground=PANEL,
                       font=("Microsoft YaHei UI", 9),
                       command=self.refresh).pack(side="left")
        tk.Spinbox(row, from_=10, to=20000, width=7, textvariable=self.target,
                   bg="#F2F4F8", relief="flat",
                   font=("Microsoft YaHei UI", 9),
                   command=self.refresh).pack(side="right")
        tk.Label(row, text="KB", bg=PANEL, fg="#5B6472",
                 font=("Microsoft YaHei UI", 9)).pack(side="right", padx=(0, 6))

        c3 = self.card("体积预估")
        c3.pack(fill="x", pady=(10, 0))
        self.info = tk.Label(c3.body, text="—", bg=PANEL, fg="#5B6472",
                             font=("Microsoft YaHei UI", 9), anchor="w",
                             justify="left", wraplength=self.W)
        self.info.pack(fill="x")

    # -- 行为 ---------------------------------------------------------
    def on_source_loaded(self, img):
        try:
            self._src_size = os.path.getsize(self.src_path)
        except OSError:
            self._src_size = 0
        self.refresh()

    def refresh(self):
        if self.src is None:
            return
        self.schedule_preview(self._estimate, 200)

    def _estimate(self):
        ext = basic.ext_of(self.fmt.get())
        q = self.quality.get()
        limit = self.limit.get()
        target = max(1, self.target.get()) * 1024
        img = self.src

        def work():
            try:
                if limit:
                    _o, _qq, size = imglib.compress_to_size(img, ext, target)
                else:
                    size = imglib.encoded_size(img, ext, q)
            except Exception as e:
                ui_post(lambda: self.info.configure(text=f"预估失败：{e}"))
                return
            ui_post(lambda: self._show_estimate(size))

        threading.Thread(target=work, daemon=True).start()

    def _show_estimate(self, size):
        self._est_size = size
        if not self._src_size:
            self.info.configure(text=f"输出约 {imglib.human_size(size)}")
            return
        delta = self._src_size - size
        pct = delta * 100.0 / self._src_size
        if delta >= 0:
            tail = f"减少 {imglib.human_size(delta)}（-{pct:.0f}%）"
        else:
            tail = f"增加 {imglib.human_size(-delta)}（+{-pct:.0f}%）"
        self.info.configure(
            text=f"原图 {imglib.human_size(self._src_size)} → "
                 f"输出约 {imglib.human_size(size)}\n{tail}")

    # -- 导出 ---------------------------------------------------------
    def save_result(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        ext = basic.ext_of(self.fmt.get())
        stem = ""
        if self.src_path:
            stem = os.path.splitext(os.path.basename(self.src_path))[0]
        path = filedialog.asksaveasfilename(
            title="转换并保存", defaultextension=ext,
            filetypes=[(f"{self.fmt.get()} 图片", f"*{ext}"),
                       ("所有文件", "*.*")], initialfile=stem)
        if not path:
            return
        try:
            _dst, nbytes = basic.convert(
                self.src, path, fmt=self.fmt.get(),
                quality=self.quality.get(),
                target_kb=self.target.get() if self.limit.get() else 0)
        except Exception as e:
            messagebox.showerror("转换失败", str(e))
            return
        try:
            before = os.path.getsize(self.src_path)
        except OSError:
            before = 0
        if before:
            pct = (before - nbytes) * 100.0 / before
            self.set_status(f"已保存 {imglib.human_size(nbytes)}"
                            f"（压缩 {pct:.0f}%）\n{path}")
        else:
            self.set_status(f"已保存 {imglib.human_size(nbytes)}\n{path}")
