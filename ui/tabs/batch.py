# -*- coding: utf-8 -*-
"""批量处理页：多张图一次性跑完，统一输出格式与命名。

这是相对在线工具的核心优势：本地批量、不上传、不限制张数。
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app import (BG, PANEL, TEXT, TEXT_2, Button, Card, ProgressBar,
                 ScrollFrame, ui_post)
from core import adjust, basic, batch, denoise as dn, imglib, processor, sr

from ..widgets import FileList, OptionChips, ParamRow

TASKS = [("denoise", "批量降噪"), ("upscale", "批量高清放大"),
         ("watermark", "批量去水印"), ("preset", "批量调色"),
         ("crop", "批量裁剪比例"), ("convert", "批量格式压缩")]
OUT_MODES = [("folder", "存到文件夹"), ("suffix", "同目录加后缀"),
             ("overwrite", "覆盖原图")]
FMTS = [("保持原格式", "保持")] + [(f, f) for f in basic.FORMATS]


class BatchTab(ttk.Frame):
    """批量处理页（交互模式与单图页不同，独立实现布局）。"""

    RIGHT_W = 326

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._running = False
        self._cancel = batch.CancelToken()
        self.task = tk.StringVar(value="denoise")
        self.strength = tk.IntVar(value=50)
        self.sens = tk.IntVar(value=55)
        self.sr_model = tk.StringVar(value="general")
        self.preset = tk.StringVar(value="鲜艳")
        self.ratio = tk.StringVar(value="1:1")
        self.out_mode = tk.StringVar(value="folder")
        self.out_dir = tk.StringVar(value="")
        self.fmt = tk.StringVar(value="保持原格式")
        self.quality = tk.IntVar(value=85)
        self.name_rule = tk.StringVar(value="{name}")
        self._boxes = {}
        self._build()

    # -- 界面 ---------------------------------------------------------
    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        left = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, sticky="nsew")
        bar = tk.Frame(left, bg=BG)
        bar.pack(fill="x", padx=(0, 4), pady=(14, 6))
        self.add_btn = Button(bar, "添加图片", stretch=True, bg=BG,
                              command=self.add_files)
        self.add_btn.pack(side="left", padx=(0, 8))
        self.add_dir_btn = Button(bar, "添加文件夹", stretch=True, bg=BG,
                                  command=self.add_folder)
        self.add_dir_btn.pack(side="left", padx=(0, 8))
        self.clear_btn = Button(bar, "清空", stretch=True, bg=BG,
                                command=self.clear)
        self.clear_btn.pack(side="left")
        self.count_label = tk.Label(bar, text="共 0 张", bg=BG, fg=TEXT_2,
                                    font=("Microsoft YaHei UI", 9))
        self.count_label.pack(side="right")

        self.listbox = FileList(left, bg=BG)
        self.listbox.pack(fill="both", expand=True, padx=(0, 14), pady=(0, 14))

        right = tk.Frame(self, width=self.RIGHT_W, bg=BG)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)

        self.status = tk.Label(right, bg=BG, fg=TEXT_2,
                               font=("Microsoft YaHei UI", 9),
                               text="先添加图片，再选择任务", anchor="w",
                               wraplength=self.RIGHT_W - 28, justify="left")
        self.status.pack(fill="x", side="bottom", padx=14, pady=(4, 12))

        out = Card(right, "执行")
        out.pack(fill="x", side="bottom", padx=14, pady=(0, 10))
        self.start_btn = Button(out.body, "开始处理", size="lg", stretch=True,
                                bg=PANEL, command=self.start)
        self.start_btn.pack(fill="x")
        self.cancel_btn = Button(out.body, "取消", variant="danger",
                                 stretch=True, bg=PANEL, command=self.cancel)
        self.cancel_btn.pack(fill="x", pady=(8, 0))
        self.cancel_btn.pack_forget()
        pb = tk.Frame(out.body, bg=PANEL)
        pb.pack(fill="x", pady=(10, 0))
        self.progress = ProgressBar(pb, width=self.RIGHT_W - 106, bg=PANEL)
        self.progress.pack(side="left")
        self.pct_label = tk.Label(pb, text="", bg=PANEL, fg=TEXT_2,
                                  font=("Microsoft YaHei UI", 9), width=5,
                                  anchor="e")
        self.pct_label.pack(side="left", padx=(8, 0))

        wrap = ScrollFrame(right)
        wrap.pack(fill="both", expand=True, padx=(0, 14), pady=(14, 6))
        p = wrap.inner
        W = self.RIGHT_W - 14 - 28

        c0 = Card(p, "任务")
        c0.pack(fill="x")
        OptionChips(c0.body, TASKS, self.task, command=self._on_task,
                    bg=PANEL, per_row=2).pack(fill="x")

        c1 = Card(p, "任务参数")
        c1.pack(fill="x", pady=(10, 0))
        self._boxes["denoise"] = self._box_denoise(c1.body, W)
        self._boxes["upscale"] = self._box_upscale(c1.body, W)
        self._boxes["watermark"] = self._box_watermark(c1.body, W)
        self._boxes["preset"] = self._box_preset(c1.body, W)
        self._boxes["crop"] = self._box_crop(c1.body, W)
        self._boxes["convert"] = self._box_convert(c1.body, W)

        c2 = Card(p, "输出设置")
        c2.pack(fill="x", pady=(10, 0))
        tk.Label(c2.body, text="输出方式", bg=PANEL, fg=TEXT_2,
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")
        OptionChips(c2.body, OUT_MODES, self.out_mode, command=self._on_out,
                    bg=PANEL, per_row=2).pack(fill="x", pady=(4, 8))
        self._dir_box = tk.Frame(c2.body, bg=PANEL)
        self._dir_box.pack(fill="x", pady=(0, 8))
        Button(self._dir_box, "选择输出文件夹…", stretch=True, bg=PANEL,
               command=self.pick_dir).pack(fill="x")
        self.dir_label = tk.Label(self._dir_box, text="未选择（默认为源目录下的"
                                                     "「帧净批量输出」）",
                                  bg=PANEL, fg="#99A1AF",
                                  font=("Microsoft YaHei UI", 8), anchor="w",
                                  wraplength=W, justify="left")
        self.dir_label.pack(fill="x", pady=(4, 0))
        tk.Label(c2.body, text="输出格式", bg=PANEL, fg=TEXT_2,
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")
        OptionChips(c2.body, FMTS, self.fmt, bg=PANEL,
                    per_row=3).pack(fill="x", pady=(4, 8))
        ParamRow(c2.body, "质量", self.quality, 1, 100, bg=PANEL,
                 width=W).pack(fill="x")
        tk.Label(c2.body, text="文件命名规则", bg=PANEL, fg=TEXT_2,
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(
            fill="x", pady=(8, 0))
        tk.Entry(c2.body, textvariable=self.name_rule, bg="#F2F4F8",
                 relief="flat", font=("Microsoft YaHei UI", 9)).pack(
            fill="x", pady=(4, 0))
        tk.Label(c2.body,
                 text="可用：{name} 原名  {index} 序号  {w} 宽  "
                      "{h} 高  {date} 日期",
                 bg=PANEL, fg="#99A1AF", font=("Microsoft YaHei UI", 8),
                 anchor="w", justify="left", wraplength=W).pack(fill="x",
                                                                pady=(4, 0))
        self._on_task()
        self._on_out()

    # -- 参数分区 -----------------------------------------------------
    def _box_denoise(self, parent, W):
        box = tk.Frame(parent, bg=PANEL)
        ParamRow(box, "降噪强度", self.strength, 0, 100, bg=PANEL,
                 width=W).pack(fill="x")
        return box

    def _box_upscale(self, parent, W):
        box = tk.Frame(parent, bg=PANEL)
        tk.Label(box, text="超分模型", bg=PANEL, fg=TEXT_2,
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")
        avail = sr.available_models()
        opts = []
        if "general" in avail:
            opts.append(("general", "通用照片"))
        if "anime" in avail:
            opts.append(("anime", "动漫插画"))
        if not opts:
            opts = [("general", "通用照片（模型缺失）")]
        self.sr_model.set(opts[0][0])
        OptionChips(box, opts, self.sr_model, bg=PANEL, per_row=2).pack(
            fill="x", pady=(4, 0))
        tk.Label(box, text="每张放大 2 倍，大图耗时较长，请耐心等待",
                 bg=PANEL, fg="#99A1AF", font=("Microsoft YaHei UI", 8),
                 anchor="w", wraplength=W, justify="left").pack(
            fill="x", pady=(6, 0))
        return box

    def _box_watermark(self, parent, W):
        box = tk.Frame(parent, bg=PANEL)
        ParamRow(box, "检测灵敏度", self.sens, 0, 100, bg=PANEL,
                 width=W, hint="自动检测水印位置后做 AI 修复").pack(fill="x")
        return box

    def _box_preset(self, parent, W):
        box = tk.Frame(parent, bg=PANEL)
        tk.Label(box, text="滤镜预设", bg=PANEL, fg=TEXT_2,
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")
        om = tk.OptionMenu(box, self.preset, *adjust.preset_names())
        om.configure(bg="#F2F4F8", fg=TEXT, relief="flat",
                     font=("Microsoft YaHei UI", 9), highlightthickness=0)
        om["menu"].configure(bg=PANEL, fg=TEXT,
                             font=("Microsoft YaHei UI", 9))
        om.pack(fill="x", pady=(4, 0))
        return box

    def _box_crop(self, parent, W):
        box = tk.Frame(parent, bg=PANEL)
        tk.Label(box, text="目标比例", bg=PANEL, fg=TEXT_2,
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(fill="x")
        om = tk.OptionMenu(box, self.ratio, *[k for k in basic.RATIOS
                                              if k != "原始"])
        om.configure(bg="#F2F4F8", fg=TEXT, relief="flat",
                     font=("Microsoft YaHei UI", 9), highlightthickness=0)
        om["menu"].configure(bg=PANEL, fg=TEXT,
                             font=("Microsoft YaHei UI", 9))
        om.pack(fill="x", pady=(4, 0))
        return box

    def _box_convert(self, parent, W):
        box = tk.Frame(parent, bg=PANEL)
        tk.Label(box, text="仅做格式转换与压缩，不改动画面内容。\n"
                           "在下方「输出设置」里选择目标格式与质量。",
                 bg=PANEL, fg="#99A1AF", font=("Microsoft YaHei UI", 8),
                 anchor="w", justify="left", wraplength=W).pack(fill="x")
        return box

    def _on_task(self):
        for k, box in self._boxes.items():
            box.pack_forget()
        cur = self._boxes.get(self.task.get())
        if cur is not None:
            cur.pack(fill="x")

    def _on_out(self):
        if self.out_mode.get() == "folder":
            self._dir_box.pack(fill="x", pady=(0, 8))
        else:
            self._dir_box.pack_forget()

    # -- 文件 ---------------------------------------------------------
    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif"),
                       ("所有文件", "*.*")])
        for p in paths:
            self.listbox.add(p)
        self._sync_count()

    def add_folder(self):
        d = filedialog.askdirectory(title="选择包含图片的文件夹")
        if not d:
            return
        for p in imglib.list_images(d, recursive=True):
            self.listbox.add(p)
        self._sync_count()

    def clear(self):
        self.listbox.clear()
        self._sync_count()

    def _sync_count(self):
        self.count_label.configure(text=f"共 {self.listbox.count()} 张")

    def pick_dir(self):
        d = filedialog.askdirectory(title="选择输出文件夹")
        if d:
            self.out_dir.set(d)
            self.dir_label.configure(text=d)

    # -- 执行 ---------------------------------------------------------
    def _make_worker(self):
        """在主线程取好参数快照：worker 跑在后台线程，不能读 Tk 变量。"""
        task = self.task.get()
        snap = self.snapshot(("strength", self.strength), ("sens", self.sens),
                             ("sr_model", self.sr_model),
                             ("preset", self.preset), ("ratio", self.ratio))
        if task == "denoise":
            def w(img, path, cancel):
                return dn.denoise(img, strength=snap["strength"],
                                  mode="auto", detail=35, cancel_cb=cancel)
            return w
        if task == "upscale":
            def w(img, path, cancel):
                return sr.upscale(img, cancel_cb=cancel,
                                  model=snap["sr_model"])
            return w
        if task == "watermark":
            def w(img, path, cancel):
                s = snap["sens"] / 100.0
                mask = processor.detect_watermark_photo(img, s)
                if mask is None or not mask.any():
                    return None                     # 没检测到就跳过
                return processor.inpaint_image(
                    img, mask, radius=5, engine="ai", cancel_check=cancel)
            return w
        if task == "preset":
            def w(img, path, cancel):
                return adjust.apply_preset(img, snap["preset"], 100)
            return w
        if task == "crop":
            def w(img, path, cancel):
                r = basic.RATIOS.get(snap["ratio"])
                if not r:
                    return None
                out, _box = basic.crop_ratio(img, r, "auto")
                return out
            return w
        return lambda img, path, cancel: img

    def start(self):
        paths = self.listbox.paths()
        if not paths:
            self.status.configure(text="请先添加图片")
            return
        if self._running:
            return
        if self.out_mode.get() == "overwrite":
            if not messagebox.askyesno(
                    "确认覆盖",
                    f"将覆盖原文件，共 {len(paths)} 张。\n"
                    "此操作不可撤销，建议先备份。确定继续吗？"):
                return

        out_dir = self.out_dir.get()
        if self.out_mode.get() == "folder" and not out_dir:
            out_dir = os.path.join(os.path.dirname(paths[0]), "帧净批量输出")
        root = os.path.commonpath([os.path.dirname(p) for p in paths]) \
            if len(paths) > 1 else os.path.dirname(paths[0])

        self._cancel.reset()
        self._running = True
        self.start_btn.set_enabled(False)
        self.cancel_btn.pack(fill="x")
        self.progress.config(value=0)
        self.pct_label.configure(text="0%")
        self.status.configure(text="处理中…")

        runner = batch.Runner(
            worker=self._make_worker(), out_mode=self.out_mode.get(),
            out_dir=out_dir, fmt=self.fmt.get(), quality=self.quality.get(),
            name_rule=self.name_rule.get(), keep_tree=True, root=root)

        def progress_cb(done, total):
            pct = int(done * 100.0 / max(total, 1))
            ui_post(lambda: self._set_progress(pct))

        def item_cb(idx, state, msg):
            ui_post(lambda: self.listbox.set_status(idx, state, msg))

        def work():
            try:
                stats = runner.run(paths, progress_cb=progress_cb,
                                   item_cb=item_cb, cancel_cb=self._cancel)
            except Exception as exc:                      # noqa: BLE001
                ui_post(lambda err=exc: self._fail(str(err)))
            else:
                ui_post(lambda: self._done(stats, out_dir))

        threading.Thread(target=work, daemon=True).start()

    def _set_progress(self, pct):
        self.progress.config(value=pct)
        self.pct_label.configure(text=f"{pct}%")

    def _done(self, stats, out_dir):
        self._running = False
        self.cancel_btn.pack_forget()
        self.start_btn.set_enabled(True)
        self._set_progress(100)
        saved = ""
        if stats["before"] and stats["after"]:
            delta = stats["before"] - stats["after"]
            saved = f"，体积 {imglib.human_size(stats['before'])} → " \
                    f"{imglib.human_size(stats['after'])}"
        self.status.configure(
            text=f"完成 {stats['ok']} / {stats['total']} 张"
                 f"（失败 {stats['failed']}）{saved}")
        if self.out_mode.get() == "folder" and out_dir:
            try:
                os.startfile(out_dir)
            except Exception:
                pass

    def _fail(self, err):
        self._running = False
        self.cancel_btn.pack_forget()
        self.start_btn.set_enabled(True)
        self.status.configure(text=f"处理失败：{err}", fg="#D9534F")

    def cancel(self):
        if self._running:
            self._cancel.cancel()
            self.status.configure(text="正在取消…")
