# -*- coding: utf-8 -*-
"""图片降噪页：弱光噪点 / 高感颗粒 / 老照片颗粒。"""

import tkinter as tk

from app import PANEL, Button
from core import denoise as dn

from ..base import TabBase
from ..widgets import OptionChips, ParamRow

MODES = [("auto", "自动"), ("fast", "快速"), ("fine", "精细"),
         ("grain", "老照片")]
MODE_TIP = {
    "auto": "按图片大小自动选择：小图走精细，大图走混合，速度与质量兼顾",
    "fast": "保边滤波，秒级出图，适合大图或先试效果",
    "fine": "非局部均值，质量最好，大图较慢",
    "grain": "针对老照片颗粒 / 扫描网点，额外做褪色还原",
}


class DenoiseTab(TabBase):
    PLACEHOLDER = ("打开一张噪点较多的照片\n\n"
                   "弱光手机照、高 ISO 夜景、老照片颗粒都适用")

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.mode = tk.StringVar(value="auto")
        self.strength = tk.IntVar(value=50)
        self.detail = tk.IntVar(value=35)
        self._tip = None
        self._build()

    # -- 界面 ---------------------------------------------------------
    def _build(self):
        c = self.card("操作")
        c.pack(fill="x")
        self.open_btn = Button(c.body, "打开图片…", size="lg", stretch=True,
                               bg=PANEL, command=self.open_image)
        self.open_btn.pack(fill="x")
        self.apply_btn = Button(c.body, "开始降噪", size="lg", stretch=True,
                                bg=PANEL, command=self.apply)
        self.apply_btn.pack(fill="x", pady=(10, 0))
        self.detect_btn = Button(c.body, "自动检测噪点并推荐强度",
                                 stretch=True, bg=PANEL,
                                 command=self.auto_detect)
        self.detect_btn.pack(fill="x", pady=(8, 0))

        c2 = self.card("降噪参数")
        c2.pack(fill="x", pady=(10, 0))
        tk.Label(c2.body, text="降噪方式", bg=PANEL,
                 fg="#5B6472", font=("Microsoft YaHei UI", 9),
                 anchor="w").pack(fill="x")
        OptionChips(c2.body, MODES, self.mode, command=self._on_mode,
                    bg=PANEL, per_row=4).pack(fill="x", pady=(4, 2))
        self._tip = tk.Label(c2.body, text=MODE_TIP["auto"], bg=PANEL,
                             fg="#99A1AF", font=("Microsoft YaHei UI", 8),
                             anchor="w", wraplength=self.W, justify="left")
        self._tip.pack(fill="x", pady=(0, 6))
        ParamRow(c2.body, "降噪强度", self.strength, 0, 100, bg=PANEL,
                 width=self.W).pack(fill="x", pady=(6, 0))
        ParamRow(c2.body, "细节回补", self.detail, 0, 100, bg=PANEL,
                 width=self.W,
                 hint="降噪会让画面变软，回补可以找回边缘锐度").pack(
            fill="x", pady=(8, 0))

        c3 = self.card("说明")
        c3.pack(fill="x", pady=(10, 0))
        tk.Label(c3.body,
                 text="建议先用「自动检测」拿到推荐强度，再微调。\n"
                      "强度过高会丢失皮肤质感与细小纹理，"
                      "此时把细节回补调高即可缓解。",
                 bg=PANEL, fg="#5B6472", font=("Microsoft YaHei UI", 8),
                 anchor="w", justify="left", wraplength=self.W).pack(fill="x")

    # -- 行为 ---------------------------------------------------------
    def _on_mode(self):
        self._tip.configure(text=MODE_TIP.get(self.mode.get(), ""))

    def auto_detect(self):
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        level = dn.estimate_noise(self.src)
        self.strength.set(level)
        self.set_status(f"检测到噪点水平 {level}，已设为推荐强度"
                        f"（0 = 干净，100 = 严重）")

    def apply(self):
        # 先把 Tk 变量快照下来，worker 线程里不能再读 Tk 变量
        snap = self.snapshot(("strength", self.strength),
                             ("mode", self.mode),
                             ("detail", self.detail))
        self.run_bg(lambda p, c: self._worker(snap, p, c),
                    busy=(self.apply_btn, self.open_btn, self.detect_btn),
                    start_msg="降噪中…")

    def _worker(self, snap, progress_cb, cancel):
        out = dn.denoise(self.src, strength=snap["strength"],
                         mode=snap["mode"], detail=snap["detail"],
                         progress_cb=progress_cb, cancel_cb=cancel)
        if out is None:
            return None
        return out, (f"降噪完成 · 强度 {snap['strength']} "
                     f"· 细节回补 {snap['detail']}")
