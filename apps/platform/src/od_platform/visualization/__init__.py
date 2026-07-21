#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : __init__.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/visualization/__init__.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 14:34:08
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:34:08 | Matri | v1.0.0 | 初始化创建
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : visualization
# @Function  : visualization 模块公共 API
"""visualization — YOLO 检测结果美化绘制模块。

提供:
  - 圆角检测框
  - 中英文标签支持
  - 标签映射(person -> 人员)
  - 文本尺寸预计算缓存

使用示例:
    from visualization import BeautifyVisualizer, Detection, DrawStyle

    visualizer = BeautifyVisualizer(
        labels=["person", "car"],
        label_mapping={"person": "人员", "car": "汽车"},
        color_mapping={"person": (0, 255, 0), "car": (255, 0, 0)},
    )

    detections = BeautifyVisualizer.from_yolo_results(
        boxes=boxes.xyxy.cpu().numpy(),
        confidences=boxes.conf.cpu().numpy(),
        labels=labels,
    )

    annotated = visualizer.draw(frame, detections, use_label_mapping=True)

可拷贝性:
    本模块不依赖宿主项目的任何内部基础设施,整个 visualization/ 目录可整包
    拷贝到任何 Python 项目下使用。字体放在 visualization/assets/ 内,跟随模块走。
"""
from __future__ import annotations

from .core.data_types import Detection, DrawStyle, LabelLayout, LabelPosition
from .core.draw_utils import LayoutCalculator, RoundedRect
from .core.renderers import PillowTextRenderer
from .core.text_cache import TextSizeCache
from .visualizer import BeautifyVisualizer

__all__ = [
    # 数据类型
    "Detection",
    "DrawStyle",
    "LabelPosition",
    "LabelLayout",
    # 工具类
    "TextSizeCache",
    "RoundedRect",
    "LayoutCalculator",
    "PillowTextRenderer",
    # 主类
    "BeautifyVisualizer",
]