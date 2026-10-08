# -*- coding: utf-8 -*-
"""调色与滤镜页：一键预设 + 基础参数精调，全部实时预览。"""

import tkinter as tk

from app import PANEL, Button
from core import adjust

from ..base import TabBase
from ..widgets import OptionChips, ParamRow

PRESET_OPTS = [(n, n) for n in adjust.preset_names()]


class ColorTab(TabBase):
    PLACEHOLDER = ("打开一张图片\n\n"
                   "先挑一个滤镜预设，再微调下面的参数")

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.preset = tk.StringVar(value="原图")
        self.preset_amt = tk.IntVar(value=100)
        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.IntVar(value=0)
        self.saturation = tk.IntVar(value=0)
        self.temperature = tk.IntVar(value=0)
        self.shadow = tk.IntVar(value=0)
        self.highlight = tk.IntVar(value=0)
        self.sharpen = tk.IntVar(value=0)
        self.gamma = tk.DoubleVar(value=1.0)
        self._build()

    # -- 界面 ---------------------------------------------------------
    def _build(self):
        c = self.card("操作")
        c.pack(fill="x")
        self.open_btn = Button(c.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        row = tk.Frame(c.body, bg=PANEL)
        row.pack(fill="x", pady=(10, 0))
        self.auto_btn = Button(row, "一键智能增强", stretch=True, bg=PANEL,
                               command=self.auto_enhance)
        self.auto_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.reset_btn = Button(row, "重置参数", stretch=True, bg=PANEL,
                                command=self.reset)
        self.reset_btn.pack(side="left", fill="x", expand=True)

        c1 = self.card("滤镜预设")
        c1.pack(fill="x", pady=(10, 0))
        OptionChips(c1.body, PRESET_OPTS, self.preset, command=self.refresh,
                    bg=PANEL, per_row=3).pack(fill="x", pady=(2, 6))
        ParamRow(c1.body, "滤镜浓度", self.preset_amt, 0, 100,
                 command=self.refresh, bg=PANEL, width=self.W).pack(fill="x")

        c2 = self.card("基础调节")
        c2.pack(fill="x", pady=(10, 0))
        ParamRow(c2.body, "亮度", self.brightness, -100, 100,
                 command=self.refresh, bg=PANEL, width=self.W).pack(fill="x")
        ParamRow(c2.body, "对比度", self.contrast, -100, 100,
                 command=self.refresh, bg=PANEL, width=self.W).pack(
            fill="x", pady=(8, 0))
        ParamRow(c2.body, "饱和度", self.saturation, -100, 100,
                 command=self.refresh, bg=PANEL, width=self.W).pack(
            fill="x", pady=(8, 0))
        ParamRow(c2.body, "色温", self.temperature, -100, 100,
                 command=self.refresh, bg=PANEL, width=self.W,
                 hint="负值偏冷（蓝），正值偏暖（黄）").pack(
            fill="x", pady=(8, 0))

        c3 = self.card("进阶")
        c3.pack(fill="x", pady=(10, 0))
        ParamRow(c3.body, "阴影提亮", self.shadow, 0, 100,
                 command=self.refresh, bg=PANEL, width=self.W).pack(fill="x")
        ParamRow(c3.body, "高光压暗", self.highlight, 0, 100,
                 command=self.refresh, bg=PANEL, width=self.W).pack(
            fill="x", pady=(8, 0))
        ParamRow(c3.body, "Gamma", self.gamma, 0.4, 2.2,
                 command=self.refresh, bg=PANEL, width=self.W,
                 fmt="{:.2f}").pack(fill="x", pady=(8, 0))
        ParamRow(c3.body, "锐化", self.sharpen, 0, 100,
                 command=self.refresh, bg=PANEL, width=self.W).pack(
            fill="x", pady=(8, 0))

    # -- 行为 ---------------------------------------------------------
    def reset(self):
        for v in (self.brightness, self.contrast, self.saturation,
                  self.temperature, self.shadow, self.highlight,
                  self.sharpen):
            v.set(0)
        self.gamma.set(1.0)
        self.preset.set("原图")
        self.preset_amt.set(100)
        self.refresh(force=True)

    def refresh(self, force=False):
        self.schedule_preview(self._preview, 120 if force else 180)

    def _snapshot(self):
        """后台线程不能读 Tk 变量，统一在主线程拍快照。"""
        return self.snapshot(
            ("preset", self.preset), ("preset_amt", self.preset_amt),
            ("brightness", self.brightness), ("contrast", self.contrast),
            ("saturation", self.saturation),
            ("temperature", self.temperature),
            ("shadow", self.shadow), ("highlight", self.highlight),
            ("sharpen", self.sharpen), ("gamma", self.gamma))

    def _make(self, img, snap):
        out = adjust.apply_preset(img, snap["preset"], snap["preset_amt"])
        return adjust.adjust(
            out, brightness=snap["brightness"], contrast=snap["contrast"],
            saturation=snap["saturation"], temperature=snap["temperature"],
            gamma=snap["gamma"], sharpen=snap["sharpen"],
            shadow=snap["shadow"], highlight=snap["highlight"])

    def _preview(self):
        src = self.src
        if src is None:
            return
        snap = self._snapshot()
        self.run_preview(lambda: self._make(src, snap))

    def auto_enhance(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        self.push_history(self.result if self.result is not None else self.src)
        self.result = adjust.auto_enhance(self.src)
        self._show_result(self.result)
        self.set_status("已应用智能增强（自动色阶 + 局部对比 + 轻锐化）")

    def apply_current(self):
        """把当前预览固化为结果（供子类/按钮复用）。"""
        if self.src is None:
            return
        src, snap = self.src, self._snapshot()
        self.run_bg(lambda p, c: (self._make(src, snap), "调色已应用"),
                    busy=(self.auto_btn,), start_msg="应用调色…")
