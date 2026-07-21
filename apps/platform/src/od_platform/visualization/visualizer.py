#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : visualizer.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/visualization/visualizer.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 14:33:29
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:33:29 | Matri | v1.0.0 | 初始化创建
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : visualizer.py
# @Author    : 雨霓同学
# @Project   : visualization
# @Function  : 美化可视化器 — cv2 画框 + Pillow 画文本,支持中英文 / 圆角 / 标签映射
"""美化可视化器。

职责: 美化绘制 YOLO 检测结果(支持中英文)。
适用: 需要圆角框 / 自定义字体 / 中文标签 / 标签映射的场景。
不适用: 朴素 YOLO 绘制(直接 results[0].plot() 更简单)。

特点:
  - cv2 绘制智能圆角框(角落自适应:标签贴上方/下方/内嵌时圆角动态切换)
  - Pillow 绘制文本(无 BGR<->RGB 转换开销)
  - 文本尺寸启动期预计算,运行期 O(1) 查表
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .core.data_types import Detection, DrawStyle
from .core.draw_utils import LayoutCalculator, RoundedRect
from .core.renderers import PillowTextRenderer
from .core.text_cache import TextSizeCache


class BeautifyVisualizer:
    """YOLO 检测结果美化可视化器。

    使用场景:
      - 需要美化效果(圆角框、自定义字体)
      - 需要中文标签显示
      - 需要标签映射(如 person -> 人员)

    若不需要美化,请直接用 YOLO 原生 ``results[0].plot()``。
    """

    def __init__(
            self,
            labels: List[str],
            label_mapping: Optional[Dict[str, str]] = None,
            color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
            default_color: Tuple[int, int, int] = (0, 255, 0),
            font_path: Optional[str] = None,
            font_sizes: Optional[Tuple[int, ...]] = None,
    ):
        """初始化美化可视化器。

        Args:
            labels: 标签列表(英文原始标签,如 YOLO 模型的 names)
            label_mapping: 标签映射字典(例如:{"person": "人员", "car": "汽车"})
            color_mapping: 颜色映射字典,键为原始标签,值为 BGR 颜色
            default_color: 默认颜色 (BGR)
            font_path: 字体绝对路径;None 时使用模块内置字体
                       (visualization/assets/LXGWWenKai-Bold.ttf)
            font_sizes: 预计算的字号范围
        """
        self.label_mapping = label_mapping or {}
        self.color_mapping = color_mapping or {}
        self.default_color = default_color

        # 文本尺寸缓存(font_path=None 时由 TextSizeCache 内部解析模块内置字体)
        self._size_cache = TextSizeCache(
            labels=labels,
            label_mapping=label_mapping,
            font_path=font_path,
            font_sizes=font_sizes,
        )

        # Pillow 文本渲染器
        self._renderer = PillowTextRenderer(size_cache=self._size_cache)

    def draw(
            self,
            image: np.ndarray,
            detections: List[Detection],
            style: Optional[DrawStyle] = None,
            use_label_mapping: bool = False,
    ) -> np.ndarray:
        """美化绘制检测结果。

        Args:
            image: 输入图像 (BGR)
            detections: 检测结果列表
            style: 绘制样式(None 则根据图像尺寸自动生成)
            use_label_mapping: 是否使用标签映射

        Returns:
            绘制后的图像 (BGR)
        """
        if not detections:
            return image.copy()

        h, w = image.shape[:2]
        style = style or DrawStyle.from_image_size(h, w)

        result = image.copy()

        # 收集文本(用于批量渲染)
        texts: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []

        for det in detections:
            x1, y1, x2, y2 = det.box
            color = self.color_mapping.get(det.label, det.color or self.default_color)

            # 获取显示标签
            display_label = (
                self.label_mapping.get(det.label, det.label)
                if use_label_mapping
                else det.label
            )
            label_text = f"{display_label} {det.confidence * 100:.1f}%"

            # 获取文本尺寸
            text_size = self._size_cache.get_size(display_label, style.font_size)

            # 计算标签布局
            layout = LayoutCalculator.compute(det.box, text_size, (h, w), style)

            # 计算圆角配置
            det_corners = LayoutCalculator.get_corners(layout, for_detection=True)
            label_corners = LayoutCalculator.get_corners(layout, for_detection=False)

            # 1. cv2 绘制检测框(圆角边框)
            RoundedRect.bordered(
                result, (x1, y1), (x2, y2),
                color, style.line_width, style.radius, det_corners,
            )

            # 2. cv2 绘制标签背景(圆角填充)
            lx1, ly1, lx2, ly2 = layout.box
            RoundedRect.filled(
                result, (lx1, ly1), (lx2, ly2),
                color, style.radius, label_corners,
            )

            # 3. 收集文本
            texts.append((label_text, layout.text_pos, style.text_color))

        # 4. Pillow 批量渲染文本(无颜色转换)
        if texts:
            result = self._renderer.render_batch(result, texts, style)

        return result

    @staticmethod
    def from_yolo_results(
            boxes: np.ndarray,
            confidences: np.ndarray,
            labels: List[str],
            color_mapping: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> List[Detection]:
        """从 YOLO 推理结果创建 Detection 列表。"""
        color_mapping = color_mapping or {}
        return [
            Detection(
                box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                confidence=float(conf),
                label=label,
                color=color_mapping.get(label, (0, 255, 0)),
            )
            for box, conf, label in zip(boxes, confidences, labels)
        ]