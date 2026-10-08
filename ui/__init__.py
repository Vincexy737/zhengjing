# -*- coding: utf-8 -*-
"""
扩展功能 UI 包。

模块在 app.py 的 main() 里以「函数内延迟导入」的方式加载：
此时 app 模块已完全初始化，因此本包可以安全地从 app 复用主题常量
与自绘控件（Button / Card / Slider / ProgressBar …），
同时不必改动 app.py 现有的一行代码。
"""
