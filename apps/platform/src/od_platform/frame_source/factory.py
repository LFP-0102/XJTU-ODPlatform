#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : factory.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/factory.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 11:05:50
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:05:50 | Matri | v1.0.0 | 初始化创建
from __future__ import  annotations

from .core.config import SourceConfig
from .core.registry import build_from_config, build_from_string
from .core.base import FrameSource

def _resolve(source:str | SourceConfig) -> FrameSource:
    if isinstance(source, SourceConfig):
        return build_from_config(source)
    if isinstance(source, str):
        return build_from_string(source)

    raise TypeError(
        f"source 必须是 SourceConfig 或 str 类型，但 收到 {type(source)}"
    )


def create_frame_source(source: str | SourceConfig, * , stride: int = 1) -> FrameSource:
    if stride < 1:
        raise ValueError(f"stride 必须大于等于1,收到stride =  {stride}")
    src = _resolve(source)
    if stride > 1:
        src.set_stride(stride)
    return src