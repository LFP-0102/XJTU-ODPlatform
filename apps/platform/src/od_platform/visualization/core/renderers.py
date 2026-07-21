#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : renderers.py
# @Author    : 雨霓同学
# @Project   : visualization
# @Function  : 文本渲染器 — Pillow 绘制文本,不做 BGR<->RGB 转换
"""文本渲染器。

特点:
  - 不做 BGR<->RGB 颜色转换(框由 cv2 画,文本由 Pillow 画,颜色全程 BGR)
  - 优先复用 TextSizeCache 里的字体缓存;无缓存时本地加载
  - 字体加载失败显式 logger.warning(撞墙②修复,与 text_cache 一致)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data_types import DrawStyle
from .text_cache import TextSizeCache, _resolve_font_path


logger = logging.getLogger(__name__)


class PillowTextRenderer:
    """Pillow 文本渲染器(BGR 进 BGR 出,支持中英文,支持批量)。"""

    def __init__(self, size_cache: Optional[TextSizeCache] = None):
        self._size_cache = size_cache
        self._fallback_warned: bool = False

    def set_cache(self, cache: TextSizeCache) -> None:
        self._size_cache = cache

    def render_batch(
            self,
            img: np.ndarray,
            texts: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]],
            style: DrawStyle,
    ) -> np.ndarray:
        """批量渲染文本(不做颜色转换)。

        Args:
            img: BGR 图像
            texts: [(text, position, color_bgr), ...]
            style: 绘制样式

        Returns:
            绘制后的 BGR 图像
        """
        if not texts:
            return img

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font = self._get_font(style)

        for text, pos, color in texts:
            draw.text(pos, text, font=font, fill=color)

        return np.array(pil_img)

    def get_text_size(self, text: str, style: DrawStyle) -> Tuple[int, int]:
        """获取文本尺寸(优先走缓存)。"""
        if self._size_cache is not None:
            parts = text.rsplit(" ", 1)
            if len(parts) == 2:
                label = parts[0]
                return self._size_cache.get_size(label, style.font_size)

        font = self._get_font(style)
        bbox = font.getbbox(text)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])

    # ── 内部 ────────────────────────────────────────────────
    def _get_font(self, style: DrawStyle) -> ImageFont.FreeTypeFont:
        """获取字体;优先复用 TextSizeCache 缓存,缺失时本地加载并显式 fallback。"""
        if self._size_cache is not None:
            return self._size_cache.get_font(style.font_size)

        font_path = _resolve_font_path(style.font_path)
        try:
            return ImageFont.truetype(font_path, style.font_size)
        except OSError as e:
            if not self._fallback_warned:
                logger.warning(
                    f"字体 '{font_path}' 加载失败({e}),已回退到 PIL 默认 bitmap 字体。"
                    f"后果: 中文字符无法正常显示,且默认字体不支持自定义字号 —— "
                    f"渲染出的文本与预计算的标签框尺寸可能错位。"
                    f"修复: 把支持中文的 ttf 放到 visualization/assets/LXGWWenKai-Bold.ttf,"
                    f"或在 DrawStyle 中显式传入 font_path。"
                )
                self._fallback_warned = True
            return ImageFont.load_default()