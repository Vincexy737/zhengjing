# -*- coding: utf-8 -*-
"""
批量处理引擎。

任何「单张图 → 单张图」的处理函数都能直接挂上来批量跑：
    runner = Runner(worker=denoise_worker, out_mode="folder", out_dir=...)
    stats = runner.run(paths, progress_cb=..., item_cb=..., cancel_cb=...)

设计要点：
  - 串行执行：AI 模型吃显存，并发反而会 OOM 或拖慢单张速度
  - 单张失败不影响整批：错误记进 Result.message 继续跑
  - 输出路径规则统一在这里算，各功能页不必各自实现一套
"""

import os
import threading

import cv2

from . import basic, imglib


class Result:
    __slots__ = ("path", "ok", "out_path", "message", "before", "after")

    def __init__(self, path, ok=False, out_path="", message="", before=0,
                 after=0):
        self.path = path
        self.ok = ok
        self.out_path = out_path
        self.message = message
        self.before = before
        self.after = after


class Runner:
    """批量任务执行器。

    Args:
        worker: (img, path, cancel_cb) -> ndarray | None；返回 None 表示跳过
        out_mode: folder（存到 out_dir）/ suffix（同目录加后缀）/ overwrite（覆盖原图）
        out_dir: out_mode=folder 时的目标目录
        fmt: "保持原格式" 或 JPG / PNG / WEBP / BMP / TIFF
        quality: 0-100
        target_kb: >0 时压到目标体积
        name_rule: 重命名模板，见 basic.rename
        suffix: out_mode=suffix 时的后缀
        keep_tree: 保留源目录的子目录结构
    """

    def __init__(self, worker, out_mode="folder", out_dir="", fmt="保持原格式",
                 quality=92, target_kb=0, name_rule="{name}", suffix="_已处理",
                 keep_tree=True, root=""):
        self.worker = worker
        self.out_mode = out_mode
        self.out_dir = out_dir
        self.fmt = fmt
        self.quality = quality
        self.target_kb = target_kb
        self.name_rule = name_rule
        self.suffix = suffix
        self.keep_tree = keep_tree
        self.root = root
        self.results = []

    # -- 输出路径 ---------------------------------------------------
    def target_path(self, src, index):
        src_dir, base = os.path.split(src)
        name, ext = os.path.splitext(base)
        if self.fmt and self.fmt != "保持原格式":
            ext = basic.ext_of(self.fmt)
        new_name = basic.rename(src, index=index, pattern=self.name_rule,
                                start=1, digits=3) if \
            self.name_rule and self.name_rule != "{name}" else None
        if new_name:
            base = new_name
            if ext and not base.lower().endswith(ext.lower()):
                base = os.path.splitext(base)[0] + ext
        else:
            base = name + ext

        if self.out_mode == "overwrite":
            return os.path.join(src_dir, base)
        if self.out_mode == "suffix":
            stem, e = os.path.splitext(base)
            return imglib.unique_path(os.path.join(src_dir,
                                                   f"{stem}{self.suffix}{e}"))
        # folder
        if self.keep_tree and self.root and src_dir.startswith(self.root):
            rel = os.path.relpath(src_dir, self.root)
            dst_dir = os.path.join(self.out_dir, "" if rel == "." else rel)
        else:
            dst_dir = self.out_dir
        return os.path.join(dst_dir, base)

    # -- 执行 -------------------------------------------------------
    def run(self, paths, progress_cb=None, item_cb=None, cancel_cb=None):
        """串行处理。返回统计字典。"""
        self.results = []
        total = len(paths)
        if self.out_mode == "folder" and self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        for i, path in enumerate(paths):
            if cancel_cb and cancel_cb():
                self.results.append(Result(path, False, message="已取消"))
                if item_cb:
                    item_cb(i, "cancel", "已取消")
                break
            if item_cb:
                item_cb(i, "running", "处理中…")
            res = Result(path, False)
            try:
                res.before = _size(path)
                img = imglib.imread(path)
                out = self.worker(img, path, cancel_cb)
                if out is None:
                    if cancel_cb and cancel_cb():
                        res.message = "已取消"
                        if item_cb:
                            item_cb(i, "cancel", "已取消")
                    else:
                        res.message = "已跳过"
                        if item_cb:
                            item_cb(i, "skip", "已跳过")
                    self.results.append(res)
                    if progress_cb:
                        progress_cb(i + 1, total)
                    continue
                dst = self.target_path(path, i)
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                fmt = None if (not self.fmt or self.fmt == "保持原格式") \
                    else self.fmt
                _dst, nbytes = basic.convert(
                    out, dst, fmt=fmt, quality=self.quality,
                    target_kb=self.target_kb)
                res.ok = True
                res.out_path = _dst
                res.after = nbytes
                res.message = os.path.basename(_dst)
                if item_cb:
                    item_cb(i, "done", res.message)
            except Exception as e:                       # 单张失败不中断整批
                res.ok = False
                res.message = str(e)
                if item_cb:
                    item_cb(i, "error", res.message)
            self.results.append(res)
            if progress_cb:
                progress_cb(i + 1, total)
        return self.stats()

    def stats(self):
        ok = sum(1 for r in self.results if r.ok)
        total = len(self.results)
        return {
            "total": total,
            "ok": ok,
            "failed": total - ok,
            "before": sum(r.before for r in self.results),
            "after": sum(r.after for r in self.results),
        }


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ----------------------------------------------------------------------
# 批量重命名（纯文件操作，不解码图像）
# ----------------------------------------------------------------------
def rename_batch(paths, pattern, start=1, digits=3, apply=False):
    """按模板批量重命名。

    apply=False 时只返回 (src, dst) 预览列表，不改动磁盘。
    返回 (计划列表, 冲突数)。
    """
    plan = []
    used = set()
    conflict = 0
    for i, src in enumerate(paths):
        folder = os.path.dirname(src)
        new = basic.rename(src, index=i + 1, pattern=pattern,
                           start=start, digits=digits)
        dst = os.path.join(folder, new)
        if dst in used or (os.path.exists(dst) and dst != src):
            conflict += 1
            stem, ext = os.path.splitext(new)
            k = 1
            while os.path.join(folder, f"{stem}_{k}{ext}") in used or \
                    os.path.exists(os.path.join(folder, f"{stem}_{k}{ext}")):
                k += 1
            dst = os.path.join(folder, f"{stem}_{k}{ext}")
        used.add(dst)
        plan.append((src, dst))
    if apply:
        for src, dst in plan:
            try:
                os.rename(src, dst)
            except OSError:
                pass
    return plan, conflict


class CancelToken:
    """线程安全的取消标志，供后台任务使用。"""

    def __init__(self):
        self._e = threading.Event()

    def cancel(self):
        self._e.set()

    @property
    def cancelled(self):
        return self._e.is_set()

    def __call__(self):
        return self._e.is_set()

    def reset(self):
        self._e.clear()
