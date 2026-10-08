# -*- coding: utf-8 -*-
"""
扩展功能页注册表。

app.py 的 main() 通过 NAV_GROUPS + create() 拿到全部页面，
新增功能只需在这里加两行，不必再改 app.py 的导航代码。

页面采用「首次切换时才创建」的惰性策略：10 个页面若在启动时
全部构建，会让窗口出现明显卡顿。
"""

# (分组标题, [(key, 显示名), ...])
# photo / enhance 是 app.py 原有页面，只在此登记导航项，不走 create()
NAV_GROUPS = [
    ("修复增强", [("photo", "去水印"), ("enhance", "清晰度增强"),
                 ("denoise", "图片降噪"), ("restore", "老照片修复")]),
    ("编辑工具", [("crop", "裁剪与尺寸"), ("convert", "格式转换"),
                 ("color", "调色滤镜")]),
    ("AI 实验室", [("matting", "AI 抠图"), ("erase", "物体消除")]),
    ("批量处理", [("batch", "批量任务")]),
]


def create(key, parent):
    """按 key 创建功能页实例。

    这里用显式 import 而非 importlib 动态导入，好让 PyInstaller 的
    静态分析能收集到全部子模块，避免打包后页面缺失。
    """
    from . import batch as _batch
    from . import color as _color
    from . import convert as _convert
    from . import crop as _crop
    from . import denoise as _denoise
    from . import erase as _erase
    from . import matting as _matting
    from . import restore as _restore

    table = {
        "denoise": _denoise.DenoiseTab,
        "restore": _restore.RestoreTab,
        "crop": _crop.CropTab,
        "convert": _convert.ConvertTab,
        "color": _color.ColorTab,
        "matting": _matting.MattingTab,
        "erase": _erase.EraseTab,
        "batch": _batch.BatchTab,
    }
    cls = table.get(key)
    if cls is None:
        raise KeyError(f"未注册的功能页：{key}")
    return cls(parent)
