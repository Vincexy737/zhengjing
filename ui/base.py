# -*- coding: utf-8 -*-
"""
扩展功能页基类 TabBase。

每个功能页都要重复实现「打开图片 / 后台跑算法 / 进度 / 取消 /
结果预览 / 保存 / 撤销」，这里统一收拢，子类只关心三件事：
  1. 往 self.props 里放参数控件
  2. 实现 build_worker() 返回真正的处理函数
  3. 需要时重写 on_source_loaded()

布局与 app.py 的 PhotoTab 保持一致（左预览 + 右参数 + 底部输出），
保证新页面看起来像是原生的一部分。
"""

import os
import threading

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app import (BG, BG_WELL, PANEL, TEXT, TEXT_2, TEXT_3, FONT_SM,
                 Button, Card, ProgressBar, ScrollFrame, ui_post)
from core import imglib

from .widgets import CompareView

IMAGE_FILETYPES = [
    ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("所有文件", "*.*"),
]


class _NoImage:
    """处理完成，但不产生新的结果图（例如只更新了选区蒙版）。

    用于区分「算法被取消返回 None」和「正常完成但无图像」两种情况，
    否则只更新蒙版的检测类任务会被界面误报成「已取消」。
    """

    def __repr__(self):
        return "<NO_IMAGE>"


NO_IMAGE = _NoImage()


class TabBase(ttk.Frame):
    """功能页骨架。PREVIEW = "compare" | "region" 决定左侧预览组件。"""

    PREVIEW = "compare"
    RIGHT_W = 326
    MAX_HISTORY = 5
    PLACEHOLDER = "打开一张图片开始处理"

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.src_path = None
        self.src = None              # 原图 BGR
        self.result = None           # 处理结果 BGR
        self._working = False
        self._cancel = threading.Event()
        self._history = []           # 撤销栈
        self._last_pct = -1
        self._busy_btns = []

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # ---- 左：预览 ----
        left = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, sticky="nsew")
        if self.PREVIEW == "region":
            from app import RegionEditor
            self.view = RegionEditor(left)
            self.view.pack(fill="both", expand=True, padx=(0, 4))
            self.view.set_placeholder(self.PLACEHOLDER)
        else:
            self.view = CompareView(left, bg=BG_WELL)
            self.view.pack(fill="both", expand=True, padx=(0, 4))
            self.view.set_placeholder(self.PLACEHOLDER)

        # ---- 右：参数 + 输出 ----
        right = tk.Frame(self, width=self.RIGHT_W, bg=BG)
        right.grid(row=0, column=1, sticky="ns")
        right.grid_propagate(False)

        self.status = tk.Label(
            right, bg=BG, fg=TEXT_2, font=FONT_SM, text="请先打开一张图片",
            anchor="w", wraplength=self.RIGHT_W - 28, justify="left")
        self.status.pack(fill="x", side="bottom", padx=14, pady=(4, 12))

        out = Card(right, "输出")
        out.pack(fill="x", side="bottom", padx=14, pady=(0, 10))
        self.save_btn = Button(out.body, "保存结果", size="lg", stretch=True,
                               bg=PANEL, command=self.save_result)
        self.save_btn.pack(fill="x")
        row = tk.Frame(out.body, bg=PANEL)
        row.pack(fill="x", pady=(8, 0))
        self.undo_btn = Button(row, "撤销", stretch=True, bg=PANEL,
                               command=self.undo)
        self.undo_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.cancel_btn = Button(row, "取消", variant="danger", stretch=True,
                                 bg=PANEL, command=self.cancel)
        self.cancel_btn.pack(side="left", fill="x", expand=True)
        self.cancel_btn.pack_forget()

        pb_row = tk.Frame(out.body, bg=PANEL)
        pb_row.pack(fill="x", pady=(10, 0))
        self.progress = ProgressBar(pb_row, width=self.RIGHT_W - 106, bg=PANEL)
        self.progress.pack(side="left")
        self.pct_label = tk.Label(pb_row, text="", bg=PANEL, fg=TEXT_2,
                                  font=FONT_SM, width=5, anchor="e")
        self.pct_label.pack(side="left", padx=(8, 0))

        self.props_wrap = ScrollFrame(right)
        self.props_wrap.pack(fill="both", expand=True, padx=(0, 14),
                             pady=(14, 6))
        self.props = self.props_wrap.inner
        self.W = self.RIGHT_W - 14 - 28        # 卡内可用宽度

    # ------------------------------------------------------------------
    # 供子类调用的工具
    # ------------------------------------------------------------------
    def card(self, title):
        return Card(self.props, title)

    def set_status(self, text, color=TEXT_2):
        self.status.configure(text=text, fg=color)

    def set_placeholder(self, text):
        if hasattr(self.view, "set_placeholder"):
            self.view.set_placeholder(text)

    # -- 打开 ---------------------------------------------------------
    def open_image(self):
        path = filedialog.askopenfilename(title="选择图片",
                                          filetypes=IMAGE_FILETYPES)
        if path:
            self.load_path(path)

    def load_path(self, path):
        try:
            img = imglib.imread(path)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))
            return False
        self.src_path = path
        self.src = img
        self.result = None
        self._history = []
        self._show_before(img)
        self.set_status(f"已载入：{os.path.basename(path)}  "
                        f"{img.shape[1]}×{img.shape[0]}")
        self.on_source_loaded(img)
        return True

    def on_source_loaded(self, img):
        """子类钩子：原图载入后刷新控件可用性 / 重置参数。"""

    def _show_before(self, img):
        if self.PREVIEW == "region":
            self.view.set_image(img)
        else:
            self.view.set_before(img)
            self.view.set_after(None)

    def _show_result(self, img):
        if self.PREVIEW == "region":
            self.view.set_image(img)
        else:
            self.view.set_after(img)

    # -- 历史 ---------------------------------------------------------
    def push_history(self, img):
        if img is None:
            return
        self._history.append(img.copy())
        if len(self._history) > self.MAX_HISTORY:
            self._history.pop(0)

    def undo(self):
        if not self._history:
            self.set_status("没有可撤销的操作")
            return
        img = self._history.pop()
        self.result = img
        self._show_result(img)
        self.set_status("已撤销一步")

    # -- 后台任务 -----------------------------------------------------
    def run_bg(self, target, on_done=None, on_error=None, busy=(),
               start_msg="处理中…"):
        """在后台线程执行 target(progress_cb, cancel_cb) 并把结果送回主线程。

        target 返回 None 视为被取消；抛异常时走 on_error。
        """
        if self._working:
            return
        if self.src is None:
            self.set_status("请先打开一张图片")
            return
        self._cancel.clear()
        self._working = True
        self._last_pct = -1
        self._busy_btns = list(busy)
        for b in self._busy_btns:
            try:
                b.set_enabled(False)
            except Exception:
                pass
        self.save_btn.set_enabled(False)
        self.undo_btn.set_enabled(False)
        self.cancel_btn.pack(fill="x", expand=True)
        self.progress.config(value=0)
        self.pct_label.configure(text="0%")
        self.set_status(start_msg, TEXT_2)

        def worker():
            def progress_cb(done, total):
                if not total:
                    return
                pct = int(max(0, min(100, round(done * 100.0 / total))))
                if pct != self._last_pct:
                    self._last_pct = pct
                    ui_post(lambda p=pct: self._set_progress(p))

            try:
                # 各算法的取消回调都是「调用」它（cancel_cb()），
                # 所以要传 bound method 而不是 Event 实例本身
                out = target(progress_cb, self._cancel.is_set)
            except Exception as exc:                     # noqa: BLE001
                # 必须用默认参数把异常绑进闭包：except 块结束后 e 会被
                # 隐式 del，而 ui_post 的 lambda 是稍后才执行的。
                ui_post(lambda err=exc: self._fail(err, on_error))
                return
            if self._cancel.is_set() or out is None:
                ui_post(lambda: self._cancelled())
                return
            if out is NO_IMAGE:
                ui_post(lambda: self._done_without_image("处理完成"))
                return
            ui_post(lambda: self._succeed(out, on_done))

        threading.Thread(target=worker, daemon=True).start()

    def _set_progress(self, pct):
        self.progress.config(value=pct)
        self.pct_label.configure(text=f"{pct}%")

    def _succeed(self, out, on_done):
        self._working = False
        self.cancel_btn.pack_forget()
        self._restore_buttons()
        self._set_progress(100)
        if isinstance(out, tuple):
            img, note = out
        else:
            img, note = out, ""
        if img is not None:
            self.push_history(self.result if self.result is not None
                              else self.src)
            self.result = img
            self._show_result(img)
        self.set_status(note or "处理完成")
        if on_done:
            on_done(img)

    def _done_without_image(self, note="处理完成"):
        """任务成功结束，但没有新的结果图需要显示。"""
        self._working = False
        self.cancel_btn.pack_forget()
        self._restore_buttons()
        self._set_progress(100)
        self.set_status(note)

    def _fail(self, exc, on_error):
        self._working = False
        self.cancel_btn.pack_forget()
        self._restore_buttons()
        self._set_progress(0)
        if on_error:
            on_error(exc)
        else:
            self.set_status(f"处理失败：{exc}", "#D9534F")
            messagebox.showerror("处理失败", str(exc))

    def _cancelled(self):
        self._working = False
        self.cancel_btn.pack_forget()
        self._restore_buttons()
        self._set_progress(0)
        self.set_status("已取消")

    def _restore_buttons(self):
        for b in self._busy_btns:
            try:
                b.set_enabled(True)
            except Exception:
                pass
        self.save_btn.set_enabled(True)
        self.undo_btn.set_enabled(True)

    def cancel(self):
        if self._working:
            self._cancel.set()
            self.set_status("正在取消…")

    # -- 参数快照 -----------------------------------------------------
    def snapshot(self, *pairs):
        """把 Tk 变量读成普通 dict，供后台线程安全使用。

        必须在主线程调用。原因：IntVar / StringVar 的 get() 会走 Tcl
        解释器，在后台线程里调用会直接抛
        "main thread is not in main loop"（Tk 不是线程安全的）。
        因此凡是要跨到 worker 线程的参数，都得先在这里拍一份快照。

        用法：snap = self.snapshot(("strength", self.strength), ...)
        """
        return {name: var.get() for name, var in pairs}

    # -- 轻量预览 -----------------------------------------------------
    def schedule_preview(self, fn, delay=180):
        """参数微调后的防抖预览：连续拖动滑块只在停手后跑一次。"""
        job = getattr(self, "_preview_job", None)
        if job is not None:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        self._preview_job = self.after(delay, fn)

    def run_preview(self, target):
        """后台跑轻量处理并刷新预览（不动进度条、不禁用按钮）。

        用递增的版本号丢弃过期结果，避免快速调参时旧结果覆盖新结果。
        """
        if self.src is None or self._working:
            return
        self._pv_ver = getattr(self, "_pv_ver", 0) + 1
        ver = self._pv_ver

        def worker():
            try:
                out = target()
            except Exception:
                return
            if ver != getattr(self, "_pv_ver", 0):
                return                       # 已有更新版本的任务，丢弃
            ui_post(lambda: self._apply_preview(out))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_preview(self, out):
        if out is None:
            return
        if isinstance(out, tuple):
            img, _note = out
        else:
            img = out
        if img is None:
            return
        self.result = img
        self._show_result(img)

    # -- 保存 ---------------------------------------------------------
    def save_result(self, default_ext=".png", quality=95,
                    filetypes=None, initial_name=""):
        img = self.result
        if img is None:
            self.set_status("还没有可保存的结果")
            return
        ft = filetypes or [
            ("PNG 图片", "*.png"), ("JPG 图片", "*.jpg"),
            ("WEBP 图片", "*.webp"), ("所有文件", "*.*")]
        if not initial_name and self.src_path:
            stem = os.path.splitext(os.path.basename(self.src_path))[0]
            initial_name = stem + "_已处理"
        path = filedialog.asksaveasfilename(
            title="保存结果", defaultextension=default_ext,
            filetypes=ft, initialfile=initial_name)
        if not path:
            return
        try:
            imglib.imwrite(path, img, quality)
            self.set_status(f"已保存：{path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
