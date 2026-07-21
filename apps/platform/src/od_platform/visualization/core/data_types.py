#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : data_types.py
# @Author    : 雨霓同学
# @Project   : visualization
# @Function  : 核心数据类型 — Detection / DrawStyle / LabelPosition / LabelLayout
"""核心数据类型。

设计纪律(对齐 frame_source):
  - 配置类(DrawStyle)用 Pydantic v2 + 字段约束,拼错/越界立刻 ValidationError
  - 运行时数据(Detection / LabelLayout)用 dataclass,创建成本低、不做验证
  - 配置层不绑 logger,验证失败直接 raise(由调用方决定如何处理)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── 运行时数据:dataclass ──────────────────────────────────────
@dataclass
class Detection:
    """单个检测结果(运行时数据,每帧高频创建)。"""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    label: str
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR


class LabelPosition(Enum):
    """标签相对检测框的位置。"""
    ABOVE = auto()
    INSIDE_TOP = auto()
    BELOW = auto()


@dataclass
class LabelLayout:
    """标签布局信息(LayoutCalculator 的内部计算结果)。"""
    box: Tuple[int, int, int, int]
    text_pos: Tuple[int, int]
    position: LabelPosition
    align_right: bool = False
    label_wider: bool = False


# ── 配置类:Pydantic v2 ───────────────────────────────────────
class DrawStyle(BaseModel):
    """绘制样式配置。

    字段约束:
      - font_path=None 表示用模块内置字体(visualization/assets/LXGWWenKai-Bold.ttf)
      - 数值字段全部带上下界,防止用户写出负 padding / 巨型字号
      - text_color 是 BGR 三元组,每个分量 0-255
      - extra="forbid":字段名拼错(如 padding_X)当场 raise,不静默丢弃
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
        arbitrary_types_allowed=False,
    )

    font_path: Optional[str] = Field(
        default=None,
        description="字体绝对路径;None 时由 TextSizeCache 解析为模块内置字体",
    )
    font_size: int = Field(default=26, gt=0, le=500, description="字号(像素)")
    line_width: int = Field(default=1, gt=0, le=50, description="检测框边框宽度")
    padding_x: int = Field(default=6, ge=0, le=500, description="标签内左右内边距")
    padding_y: int = Field(default=10, ge=0, le=500, description="标签内上下内边距")
    radius: int = Field(default=3, ge=0, le=500, description="圆角半径")
    text_color: Tuple[int, int, int] = Field(
        default=(0, 0, 0),
        description="文本 BGR 颜色",
    )

    @field_validator("text_color")
    @classmethod
    def _validate_color(cls, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        for c in v:
            if not isinstance(c, int) or not (0 <= c <= 255):
                raise ValueError(
                    f"text_color 每个分量必须是 0-255 之间的整数,得到 {v}"
                )
        return v

    @classmethod
    def from_image_size(
            cls,
            height: int,
            width: int,
            ref_dim: int = 720,
            base_font_size: int = 26,
            base_line_width: int = 2,
            base_padding_x: int = 10,
            base_padding_y: int = 10,
            base_radius: int = 8,
            **kwargs,
    ) -> "DrawStyle":
        """根据图像尺寸自适应计算样式参数。

        透传 kwargs 给 cls(),允许调用方覆盖任意字段(如 font_path / text_color)。
        """
        scale = min(height, width) / max(ref_dim, 1)
        return cls(
            font_size=max(10, int(base_font_size * scale)),
            line_width=max(1, int(base_line_width * scale)),
            padding_x=max(5, int(base_padding_x * scale)),
            padding_y=max(5, int(base_padding_y * scale)),
            radius=max(3, int(base_radius * scale)),
            **kwargs,
        )