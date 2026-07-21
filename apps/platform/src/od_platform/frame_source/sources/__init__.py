#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : __init__.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/core/sources/__init__.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 10:36:08
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:36:08 | Matri | v1.0.0 | 初始化创建
from __future__ import annotations

from .camera import CameraSource
from .image import  ImageFolderSource, ImageSource
from .video import VideoSource

__all__ = ["CameraSource", "ImageSource", "ImageFolderSource", 'VideoSource']