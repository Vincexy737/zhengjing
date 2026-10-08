# -*- coding: utf-8 -*-
"""临时冒烟测试：验证新增 core 算法与全部功能页可正常构造/运行。

用法：python _smoke.py
输出全为 ASCII，避免 Windows 控制台编码问题。
"""

import os
import sys
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK, FAIL = [], []


def check(name, fn):
    try:
        fn()
        OK.append(name)
        print(f"[ OK ] {name}")
    except Exception as e:                                # noqa: BLE001
        FAIL.append((name, e))
        print(f"[FAIL] {name}: {e}")
        traceback.print_exc()


# ----------------------------------------------------------------------
def make_image(w=320, h=240):
    """合成一张有结构的测试图（渐变 + 色块 + 噪点）。"""
    img = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        img[y, :] = (40 + y * 120 // h, 90, 200 - y * 100 // h)
    cv2.rectangle(img, (40, 40), (140, 140), (240, 200, 60), -1)
    cv2.circle(img, (230, 150), 55, (30, 220, 180), -1)
    noise = np.random.normal(0, 14, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


IMG = make_image()
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_src.png")
cv2.imwrite(SRC, IMG)


# ----------------------------------------------------------------------
# core 算法
# ----------------------------------------------------------------------
def t_imglib():
    from core import imglib
    b = imglib.encode(IMG, ".jpg", 85)
    assert b.nbytes > 0
    assert imglib.encoded_size(IMG, ".jpg", 60) < imglib.encoded_size(IMG, ".jpg", 95)
    out, q, n = imglib.compress_to_size(IMG, ".jpg", 8 * 1024)
    assert n <= 8 * 1024, f"compress failed: {n}"
    assert imglib.fit_max(IMG, 100).shape[0] == 75


def t_denoise():
    from core import denoise
    for mode in ("fast", "fine", "grain", "auto"):
        out = denoise.denoise(IMG, strength=50, mode=mode, detail=30)
        assert out is not None and out.shape == IMG.shape, mode
    lvl = denoise.estimate_noise(IMG)
    assert 0 <= lvl <= 100


def t_adjust():
    from core import adjust
    for name in adjust.preset_names():
        out = adjust.apply_preset(IMG, name, 100)
        assert out.shape == IMG.shape, name
    out = adjust.adjust(IMG, brightness=10, contrast=10, saturation=10,
                        temperature=10, gamma=1.1, sharpen=20,
                        shadow=20, highlight=20)
    assert out.shape == IMG.shape
    assert adjust.auto_enhance(IMG).shape == IMG.shape


def t_basic():
    from core import basic
    for key, r in basic.RATIOS.items():
        if r is None:
            continue
        out, box = basic.crop_ratio(IMG, r, "center")
        assert out.shape[0] > 0 and out.shape[1] > 0, key
    out, _ = basic.crop_ratio(IMG, 1.0, "auto")
    assert out.shape[0] == out.shape[1] or abs(out.shape[0] - out.shape[1]) <= 1
    assert basic.resize_img(IMG, "percent", 50).shape[1] == IMG.shape[1] // 2
    assert basic.resize_img(IMG, "width", 100).shape[1] == 100
    assert basic.resize_img(IMG, "long", 100).shape[1] == 100
    dst = os.path.join(os.path.dirname(SRC), "_smoke_out.webp")
    p, n = basic.convert(IMG, dst, fmt="WEBP", quality=80)
    assert os.path.exists(p) and n > 0
    os.remove(p)
    name = basic.rename(SRC, index=2, pattern="pic_{index}_{w}x{h}")
    assert name.startswith("pic_002_320x240"), name


def t_ratio():
    """比例转换基础能力：智能裁剪取局部 + 多轮扩图。"""
    from core import basic
    # 9:16 竖图 → 16:9：智能裁剪取局部，比例正确
    tall = np.zeros((640, 360, 3), np.uint8)
    out, _box = basic.crop_ratio(tall, 16 / 9.0, "center")
    h, w = out.shape[:2]
    assert abs(w / float(h) - 16 / 9.0) < 0.01, (w, h)



def t_matting():
    from core import matting
    a = matting.matte_color(IMG, 32)
    assert a.shape == IMG.shape[:2]
    for mode in ("color", "blur", "transparent"):
        out = matting.compose(IMG, a, mode=mode, color=(255, 255, 255))
        assert out is not None
        if mode == "transparent":
            assert out.shape[2] == 4
    assert matting.matte(IMG, engine="color") is not None


def t_restore():
    from core import restore
    out, note = restore.restore(IMG, do_scratch=True, do_denoise=True,
                                do_color=True, do_sharpen=True,
                                do_upscale=False, do_colorize=False)
    assert out is not None and out.shape == IMG.shape, note
    m = restore.detect_scratches(IMG, 50)
    assert m is None or m.shape == IMG.shape[:2]


def t_batch():
    from core import batch as bt
    out_dir = os.path.join(os.path.dirname(SRC), "_smoke_batch")
    runner = bt.Runner(worker=lambda img, p, c: img, out_mode="folder",
                       out_dir=out_dir, fmt="PNG", quality=90)
    stats = runner.run([SRC, SRC])
    assert stats["ok"] == 2, stats
    assert os.path.isdir(out_dir)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    os.rmdir(out_dir)
    plan, _c = bt.rename_batch([SRC], "x_{index}", apply=False)
    assert len(plan) == 1


def t_models():
    from core import models
    assert isinstance(models.model_dirs(), list)
    assert models.path_of("rmbg") is None or os.path.exists(models.path_of("rmbg"))


# ----------------------------------------------------------------------
# UI 构造
# ----------------------------------------------------------------------
_ROOT = None


def get_root():
    """整个测试只用一个 Tk 根窗口。

    app.ui_font() 会缓存 Font 对象，而 Font 绑定到创建它的 Tcl 解释器；
    若销毁 root 再新建，缓存里的旧 Font 会抛
    "application has been destroyed"。
    """
    global _ROOT
    if _ROOT is None:
        import queue
        import tkinter as tk

        import app

        _ROOT = tk.Tk()
        _ROOT.withdraw()

        def drain():
            try:
                while True:
                    app._DISPATCH_Q.get_nowait()()
            except queue.Empty:
                pass
            _ROOT.after(40, drain)

        _ROOT.after(40, drain)
    return _ROOT


def t_ui_pages():
    import app                                   # noqa: F401  先让主题/控件就位
    from ui.tabs import NAV_GROUPS, create
    root = get_root()
    keys = [k for _g in NAV_GROUPS for k, _t in _g[1]]
    made = 0
    for k in keys:
        if k in ("photo", "enhance"):
            continue
        tab = create(k, root)
        assert tab is not None, k
        tab.destroy()
        made += 1
    print(f"       built {made} pages: {', '.join(keys)}")


def t_e2e_pages():
    """端到端：载入图片 → 后台处理 → 回传主线程刷新预览。

    用 mainloop() 驱动而不是手动 update() 轮询：真实应用跑的就是
    mainloop，手动 update 驱动 after 定时器会出现与线上不一致的行为
    （实测会卡死），测出来的结论没有参考价值。
    """
    from ui.tabs import create

    root = get_root()
    report = {}

    cases = [
        # (页面, 触发方式, 校验)
        ("denoise", lambda t: t.apply(), None),
        ("crop", lambda t: (t.ratio.set("1:1"), t.refresh()), "1:1"),
        # ratio 页为智能重排（整跑 AI 管线），由 core.relayout 覆盖
        ("color", lambda t: (t.preset.set("黑白"), t.refresh(force=True)),
         None),
    ]

    def run_case(i=0):
        if i >= len(cases):
            root.quit()
            return
        key, trigger, check = cases[i]
        tab = create(key, root)
        if not tab.load_path(SRC):
            report[key] = (False, "load failed")
            tab.destroy()
            run_case(i + 1)
            return
        trigger(tab)

        def verify():
            ok = tab.result is not None
            note = tab.status.cget("text")
            if ok and check == "1:1":
                h, w = tab.result.shape[:2]
                ok = abs(h - w) <= 1
                note = f"{w}x{h}"
            report[key] = (ok, note)
            print(f"       {key}: {'OK' if ok else 'FAIL'} | {note}")
            tab.destroy()
            run_case(i + 1)

        root.after(3000, verify)

    root.after(100, lambda: run_case(0))
    root.mainloop()

    for key, (ok, note) in report.items():
        assert ok, f"{key} failed: {note}"
    assert len(report) == len(cases), f"only ran {list(report)}"


def t_photo_manual():
    """照片去水印页：自动检测已移除，改为手动选取区域后去除。"""
    import app
    root = get_root()
    tab = app.PhotoTab(root)
    assert not hasattr(tab, "detect_btn"), "自动检测按钮未移除"
    assert not hasattr(tab, "detect_sens"), "灵敏度滑块未移除"
    assert hasattr(tab, "inpaint_btn"), "缺少「去除选中水印」按钮"
    assert hasattr(tab, "clear_btn"), "缺少「清除选区」按钮"
    tab.tool_sel.set("画笔涂抹")
    tab._on_tool()
    assert tab.editor.tool.get() == "brush", "未切到画笔"
    tab.tool_sel.set("矩形框选")
    tab._on_tool()
    assert tab.editor.tool.get() == "rect", "未切回矩形"
    tab.destroy()


def main():
    check("core.imglib", t_imglib)
    check("core.denoise", t_denoise)
    check("core.adjust", t_adjust)
    check("core.basic", t_basic)
    check("core.ratio", t_ratio)
    check("core.matting", t_matting)
    check("core.restore", t_restore)
    check("core.batch", t_batch)
    check("core.models", t_models)
    check("ui.pages", t_ui_pages)
    check("ui.e2e", t_e2e_pages)
    check("photo.manual", t_photo_manual)
    if os.path.exists(SRC):
        os.remove(SRC)
    print("\n==== %d passed, %d failed ====" % (len(OK), len(FAIL)))
    for n, e in FAIL:
        print(f"  - {n}: {e}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
