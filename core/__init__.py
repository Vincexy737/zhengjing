# -*- coding: utf-8 -*-
"""去水印核心算法模块"""

from .processor import (
    INPAINT_NS,
    INPAINT_TELEA,
    ProcessingCancelled,
    clean_mask,
    detect_watermark_ocr,
    detect_watermark_photo,
    fsr_available,
    inpaint_image,
    load_image,
    mask_bbox,
    save_image,
)

__all__ = [
    "INPAINT_NS",
    "INPAINT_TELEA",
    "ProcessingCancelled",
    "clean_mask",
    "detect_watermark_ocr",
    "detect_watermark_photo",
    "fsr_available",
    "inpaint_image",
    "load_image",
    "mask_bbox",
    "save_image",
]
