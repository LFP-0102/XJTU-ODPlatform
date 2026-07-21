#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : visualization
# @Function  : core 子包公共 API
"""core 子包 — 数据类型 / 文本缓存 / 绘制工具 / 渲染器。"""
from __future__ import annotations

from .data_types import Detection, DrawStyle, LabelLayout, LabelPosition
from .draw_utils import LayoutCalculator, RoundedRect
from .renderers import PillowTextRenderer
from .text_cache import TextSizeCache

__all__ = [
    "Detection",
    "DrawStyle",
    "LabelPosition",
    "LabelLayout",
    "TextSizeCache",
    "RoundedRect",
    "LayoutCalculator",
    "PillowTextRenderer",
]