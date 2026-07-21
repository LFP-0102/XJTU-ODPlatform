#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : __init__.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/core/__init__.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 09:40:35
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:40:35 | Matri | v1.0.0 | 初始化创建
from __future__ import annotations

from .types import (SourceType, FrameInfo, Frame, CameraIntrinsics,
                IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, STREAM_SCHEMES,is_stream_url)

from .config import (SourceConfig, CameraConfig, VideoConfig, ImageConfig, ImageFolderConfig,CameraCodec, CameraBackend)

from .base import FrameSource

from .registry import (register_source, build_from_config, build_from_string, list_sources)
__all__ = [
    "SourceType", "FrameInfo", "Frame", "CameraIntrinsics",
    "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "STREAM_SCHEMES", "is_stream_url",
    "SourceConfig", "CameraConfig", "VideoConfig", "ImageConfig", "ImageFolderConfig","CameraCodec", "CameraBackend",
    "FrameSource",
    "register_source", "build_from_config", "build_from_string", "list_sources",
]



