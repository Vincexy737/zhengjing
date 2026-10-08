# -*- coding: utf-8 -*-
"""AI 抠图与背景替换页。

抠图（较慢，要跑模型）与换背景（纯本地合成，很快）分开：
抠一次拿到 alpha 后，换底色 / 换背景图 / 加投影都是实时预览。
"""

import os
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox

from app import PANEL, Button, ui_post
from core import imglib, matting, models

from ..base import TabBase
from ..widgets import OptionChips, ParamRow

BG_MODES = [("color", "纯色"), ("image", "图片"), ("blur", "虚化"),
            ("transparent", "透明")]
COLORS = [("白底", "#FFFFFF"), ("蓝底", "#438EDB"), ("红底", "#D9534F"),
          ("灰底", "#E8EAED"), ("自定义", "#CUSTOM")]


class MattingTab(TabBase):
    PLACEHOLDER = ("打开一张图片\n\n"
                   "一键抠出主体，换成纯色 / 图片 / 透明背景")

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.engine = tk.StringVar(value=matting.default_engine())
        self.bg_mode = tk.StringVar(value="color")
        self.color_name = tk.StringVar(value="白底")
        self.color_hex = "#FFFFFF"
        self.tol = tk.IntVar(value=32)
        self.edge = tk.IntVar(value=0)
        self.shadow = tk.IntVar(value=0)
        self.alpha = None
        self.bg_img = None
        self._chips = None
        self._build()

    # -- 界面 ---------------------------------------------------------
    def _build(self):
        c = self.card("操作")
        c.pack(fill="x")
        self.open_btn = Button(c.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        self.matte_btn = Button(c.body, "一键抠图", size="lg", stretch=True,
                                bg=PANEL, command=self.do_matte)
        self.matte_btn.pack(fill="x", pady=(10, 0))
        self.dl_btn = Button(c.body, "下载 AI 抠图模型（176MB）", stretch=True,
                             bg=PANEL, command=self.download_model)
        self.dl_btn.pack(fill="x", pady=(8, 0))
        self._sync_model_state()

        c1 = self.card("抠图引擎")
        c1.pack(fill="x", pady=(10, 0))
        self._eng_box = tk.Frame(c1.body, bg=PANEL)
        self._eng_box.pack(fill="x")
        # eng_tip 必须先建好：_build_engine_chips 内部会回调 _on_engine
        self.eng_tip = tk.Label(c1.body, text="", bg=PANEL, fg="#99A1AF",
                                font=("Microsoft YaHei UI", 8), anchor="w",
                                wraplength=self.W, justify="left")
        self.eng_tip.pack(fill="x", pady=(4, 0))
        self._build_engine_chips()
        ParamRow(c1.body, "纯色容差", self.tol, 5, 60, bg=PANEL,
                 width=self.W,
                 hint="仅「纯色背景抠图」需要，背景不干净时调大").pack(
            fill="x", pady=(6, 0))

        c2 = self.card("背景")
        c2.pack(fill="x", pady=(10, 0))
        OptionChips(c2.body, BG_MODES, self.bg_mode,
                    command=self.on_bg_change, bg=PANEL,
                    per_row=4).pack(fill="x", pady=(2, 6))
        self._color_box = tk.Frame(c2.body, bg=PANEL)
        self._color_box.pack(fill="x")
        OptionChips(self._color_box,
                    [(n, n) for n, _h in COLORS], self.color_name,
                    command=self.on_color, bg=PANEL,
                    per_row=3).pack(fill="x")
        self._img_box = tk.Frame(c2.body, bg=PANEL)
        self.bg_btn = Button(self._img_box, "选择背景图片…", stretch=True,
                             bg=PANEL, command=self.pick_bg)
        self.bg_btn.pack(fill="x")
        self.bg_label = tk.Label(self._img_box, text="未选择", bg=PANEL,
                                 fg="#99A1AF", font=("Microsoft YaHei UI", 8),
                                 anchor="w")
        self.bg_label.pack(fill="x", pady=(4, 0))

        c3 = self.card("边缘处理")
        c3.pack(fill="x", pady=(10, 0))
        ParamRow(c3.body, "去除白边", self.edge, 0, 100,
                 command=self.refresh, bg=PANEL, width=self.W,
                 hint="抠图后主体边缘有残留背景色时调大").pack(fill="x")
        ParamRow(c3.body, "投影强度", self.shadow, 0, 100,
                 command=self.refresh, bg=PANEL, width=self.W).pack(
            fill="x", pady=(8, 0))
        self.on_bg_change()

    def _build_engine_chips(self):
        for w in self._eng_box.winfo_children():
            w.destroy()
        opts = matting.engine_names()
        self._chips = OptionChips(self._eng_box, opts, self.engine,
                                  command=self._on_engine, bg=PANEL,
                                  per_row=2)
        self._chips.pack(fill="x")
        if not matting.ai_ready():
            self.engine.set("color")
            self._chips._sync()
        self._on_engine()

    def _sync_model_state(self):
        if matting.ai_ready():
            self.dl_btn.pack_forget()
        try:
            self.dl_btn.set_enabled(not matting.ai_ready())
        except Exception:
            pass

    def _on_engine(self):
        e = self.engine.get()
        tips = {
            "rmbg": "AI 通用抠图：任意背景、发丝级边缘，适合人像与商品图",
            "u2netp": "AI 轻量抠图：模型小、速度快，适合批量与纯 CPU",
            "color": "纯色背景抠图：无需模型，对证件照蓝/白/红底效果很好",
        }
        self.eng_tip.configure(text=tips.get(e, ""))

    def on_bg_change(self):
        m = self.bg_mode.get()
        if m == "color":
            self._color_box.pack(fill="x")
            self._img_box.pack_forget()
        elif m == "image":
            self._color_box.pack_forget()
            self._img_box.pack(fill="x")
        else:
            self._color_box.pack_forget()
            self._img_box.pack_forget()
        self.refresh()

    def on_color(self):
        name = self.color_name.get()
        if name == "自定义":
            rgb, hexstr = colorchooser.askcolor(title="选择背景颜色")
            if hexstr:
                self.color_hex = hexstr
        else:
            for n, h in COLORS:
                if n == name:
                    self.color_hex = h
                    break
        self.refresh()

    def _hex_to_bgr(self):
        h = self.color_hex.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def pick_bg(self):
        path = filedialog.askopenfilename(
            title="选择背景图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.webp *.bmp"),
                       ("所有文件", "*.*")])
        if not path:
            return
        try:
            self.bg_img = imglib.imread(path)
            self.bg_label.configure(text=os.path.basename(path))
            self.refresh()
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    # -- 抠图 ---------------------------------------------------------
    def _snapshot(self):
        """后台线程不能读 Tk 变量，统一在主线程拍快照。"""
        return self.snapshot(("engine", self.engine), ("tol", self.tol),
                             ("bg_mode", self.bg_mode),
                             ("shadow", self.shadow), ("edge", self.edge))

    def do_matte(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        snap = self._snapshot()
        color, bg = self._hex_to_bgr(), self.bg_img
        self.run_bg(lambda p, c: self._matte_worker(snap, color, bg, p, c),
                    busy=(self.matte_btn, self.open_btn),
                    start_msg="抠图中…")

    def _matte_worker(self, snap, color, bg, progress_cb, cancel):
        a = matting.matte(self.src, engine=snap["engine"], tol=snap["tol"],
                          progress_cb=progress_cb, cancel_cb=cancel)
        if a is None:
            return None
        self.alpha = a
        out = self._compose(snap, color, bg)
        cov = (a > 127).mean() * 100
        return out, f"抠图完成 · 主体占比 {cov:.0f}%"

    def _compose(self, snap, color=None, bg=None):
        if self.alpha is None or self.src is None:
            return None
        return matting.compose(
            self.src, self.alpha, mode=snap["bg_mode"],
            color=color if color is not None else self._hex_to_bgr(),
            bg_img=bg, shadow=snap["shadow"], edge=snap["edge"])

    def refresh(self):
        if self.alpha is None:
            return
        snap = self._snapshot()
        color, bg = self._hex_to_bgr(), self.bg_img
        self.schedule_preview(
            lambda: self.run_preview(lambda: self._compose(snap, color, bg)),
            120)

    # -- 模型下载 -----------------------------------------------------
    def download_model(self):
        self.dl_btn.set_enabled(False)
        self.set_status("正在下载抠图模型…")

        def work():
            try:
                models.download(
                    "rmbg",
                    progress_cb=lambda d, t: ui_post(
                        lambda: self._dl_progress(d, t)))
            except Exception as exc:                      # noqa: BLE001
                ui_post(lambda err=exc: self._dl_done(False, str(err)))
            else:
                ui_post(lambda: self._dl_done(True, ""))

        threading.Thread(target=work, daemon=True).start()

    def _dl_progress(self, done, total):
        if total:
            pct = int(min(100, done * 100.0 / total))
            self.progress.config(value=pct)
            self.pct_label.configure(text=f"{pct}%")
            self.set_status(
                f"下载中 {done / 1048576:.1f} / {total / 1048576:.1f} MB")

    def _dl_done(self, ok, err):
        self.progress.config(value=0)
        self.pct_label.configure(text="")
        if ok:
            self._build_engine_chips()
            self._sync_model_state()
            self.set_status("抠图模型已就绪，可以开始使用了")
        else:
            self.dl_btn.set_enabled(True)
            self.set_status(f"下载失败：{err}", "#D9534F")

    # -- 保存 ---------------------------------------------------------
    def save_result(self, default_ext=".png", quality=95,
                    filetypes=None, initial_name=""):
        if self.bg_mode.get() == "transparent":
            super().save_result(
                default_ext=".png",
                filetypes=[("PNG 图片（透明背景）", "*.png")],
                initial_name=initial_name or "抠图_透明背景")
        else:
            super().save_result(default_ext=default_ext, quality=quality,
                                filetypes=filetypes,
                                initial_name=initial_name)
