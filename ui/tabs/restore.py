# -*- coding: utf-8 -*-
"""老照片修复页：划痕修补、颗粒降噪、褪色还原、清晰度回补、AI 上色。"""

import tkinter as tk

from app import PANEL, Button
from core import restore as rs
from core import sr

from ..base import TabBase
from ..widgets import ParamRow


class RestoreTab(TabBase):
    PLACEHOLDER = ("打开一张老照片\n\n"
                   "勾选要修复的项目，一次处理到位")

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.do_scratch = tk.BooleanVar(value=True)
        self.do_denoise = tk.BooleanVar(value=True)
        self.do_color = tk.BooleanVar(value=True)
        self.do_sharpen = tk.BooleanVar(value=True)
        self.do_upscale = tk.BooleanVar(value=False)
        self.do_colorize = tk.BooleanVar(value=False)
        self.sens = tk.IntVar(value=50)
        self.dstrength = tk.IntVar(value=55)
        self._build()

    # -- 界面 ---------------------------------------------------------
    def _build(self):
        c = self.card("操作")
        c.pack(fill="x")
        self.open_btn = Button(c.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        self.go_btn = Button(c.body, "开始修复", size="lg", stretch=True,
                             bg=PANEL, command=self.apply)
        self.go_btn.pack(fill="x", pady=(10, 0))
        self.all_btn = Button(c.body, "全选常用项", stretch=True, bg=PANEL,
                              command=self.pick_common)
        self.all_btn.pack(fill="x", pady=(8, 0))

        c1 = self.card("修复项")
        c1.pack(fill="x", pady=(10, 0))
        self._check(c1.body, "划痕 / 破损修补", self.do_scratch)
        ParamRow(c1.body, "划痕灵敏度", self.sens, 0, 100, bg=PANEL,
                 width=self.W,
                 hint="调高会修补更多细纹，也可能误伤纹理").pack(
            fill="x", pady=(0, 8))
        self._check(c1.body, "颗粒 / 噪点降噪", self.do_denoise)
        ParamRow(c1.body, "降噪强度", self.dstrength, 0, 100, bg=PANEL,
                 width=self.W).pack(fill="x", pady=(0, 8))
        self._check(c1.body, "褪色还原（自动色阶 + 对比）", self.do_color)
        self._check(c1.body, "清晰度回补", self.do_sharpen)
        self._check(c1.body, "AI 放大 2×（较慢）", self.do_upscale)
        self._check(c1.body, "AI 智能上色（需模型）", self.do_colorize)
        if not rs.colorize_ready():
            tk.Label(c1.body,
                     text="上色模型未安装，勾选后会先尝试自动下载"
                          "（DDColor，约 330MB）",
                     bg=PANEL, fg="#99A1AF",
                     font=("Microsoft YaHei UI", 8), anchor="w",
                     justify="left", wraplength=self.W).pack(
                fill="x", pady=(4, 0))
        if not sr.is_ready("general"):
            try:
                self.do_upscale.set(False)
            except Exception:
                pass

    def _check(self, parent, text, var):
        tk.Checkbutton(parent, text=text, variable=var, bg=PANEL,
                       fg="#5B6472", selectcolor="#F2F4F8",
                       activebackground=PANEL,
                       font=("Microsoft YaHei UI", 9),
                       anchor="w", justify="left",
                       wraplength=self.W).pack(fill="x", pady=2)

    # -- 行为 ---------------------------------------------------------
    def pick_common(self):
        for v in (self.do_scratch, self.do_denoise, self.do_color,
                  self.do_sharpen):
            v.set(True)
        self.set_status("已勾选常用项，可再手动调整")

    def apply(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        if not any((self.do_scratch.get(), self.do_denoise.get(),
                    self.do_color.get(), self.do_sharpen.get(),
                    self.do_upscale.get(), self.do_colorize.get())):
            self.set_status("请至少勾选一项修复内容")
            return
        # 后台线程不能读 Tk 变量，先在主线程拍快照
        snap = self.snapshot(
            ("do_scratch", self.do_scratch), ("sens", self.sens),
            ("do_denoise", self.do_denoise), ("dstrength", self.dstrength),
            ("do_color", self.do_color), ("do_sharpen", self.do_sharpen),
            ("do_upscale", self.do_upscale),
            ("do_colorize", self.do_colorize))
        self.run_bg(lambda p, c: self._worker(snap, p, c),
                    busy=(self.go_btn, self.open_btn), start_msg="修复中…")

    def _worker(self, snap, progress_cb, cancel):
        out, note = rs.restore(
            self.src, do_scratch=snap["do_scratch"],
            scratch_sens=snap["sens"], do_denoise=snap["do_denoise"],
            denoise_strength=snap["dstrength"],
            do_color=snap["do_color"], do_sharpen=snap["do_sharpen"],
            do_upscale=snap["do_upscale"],
            do_colorize=snap["do_colorize"],
            progress_cb=progress_cb, cancel_cb=cancel)
        if out is None:
            return None
        return out, note
