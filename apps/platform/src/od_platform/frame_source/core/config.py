#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : config.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/core/config.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 10:03:41
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:03:41 | Matri | v1.0.0 | 初始化创建
"""
源配置(基于 Pydantic v2 BaseModel, 不继承宿主项目内部配置基类)。

本模块把"配置类型"当派发轴: build_from_config(cfg) 靠 type(cfg) 查该造哪个源。
所以每种源都有一个可区分的 config 类, 它们各扛两件实事:
    1. 派发的钥匙 —— type(cfg) 唯一确定造哪个源;
    2. fail-fast 校验 —— 路径存在性 / 字段取值当场校验, 别拖到 open()。

设计原则:
    - 所有 config 继承 SourceConfig, 统一 extra=forbid + validate_assignment;
    - 路径类 config 在构造时校验存在性(网络流豁免, URL 图片直接拒);
    - 配置层不绑 logger: 校验失败直接 raise ValidationError, 由调用方决定记录方式。
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .types import is_stream_url


CameraBackend = Literal["auto", "msmf", "dshow", "v4l2"]
CameraCodec   = Literal["MJPG", "YUYV", "H264", "MP4V"]


class SourceConfig(BaseModel):
    """
    所有源配置的基类。

    一处封装 fail-fast: extra=forbid(拼错字段名当场拦下)+ validate_assignment
    (后续赋值也走校验)。子类继承即享用, 不用各写各的。

    identity: 人类可读的源标识(日志/调试用), 子类按需覆盖。
    """
    model_config = ConfigDict(
        extra="forbid",              # 拼错字段名第一时间拦下
        validate_assignment=True,    # 后续赋值也走验证
    )

    @property
    def identity(self) -> str:
        return self.__class__.__name__


class CameraConfig(SourceConfig):
    """
    cv2 摄像头配置。

    示例:
        CameraConfig()                                             # 默认
        CameraConfig(width=1280, height=720, fps=90, backend="msmf")

    字段说明:
        camera_id : OpenCV 设备 ID(≥ 0)
        width/height/fps : 请求值(实际生效以 set+get 协商结果为准)
        backend   : auto(系统自选) / msmf(Win 高帧率) / dshow(Win 兼容) / v4l2(Linux)
        codec     : FOURCC 编码, 高帧率必须 MJPG
    """
    camera_id: int = Field(default=0,    ge=0,            description="OpenCV 设备 ID")
    width:     int = Field(default=1280, gt=0, le=7680,   description="请求分辨率宽")
    height:    int = Field(default=720,  gt=0, le=4320,   description="请求分辨率高")
    fps:       int = Field(default=30,   gt=0, le=1000,   description="请求帧率")

    backend: CameraBackend = Field(default="auto", description="摄像头后端")
    codec:   CameraCodec   = Field(default="MJPG", description="FOURCC 编码")

    @property
    def identity(self) -> str:
        return f"camera:{self.camera_id}"

    def get_resolution(self) -> tuple[int, int, int]:
        """返回 (width, height, fps) 三元组。"""
        return (self.width, self.height, self.fps)


class VideoConfig(SourceConfig):
    """
    视频文件 / 网络视频流配置。

    path 可以是本地视频文件路径, 也可以是 rtsp/rtmp/http(s) 网络流 URL。
    本地文件在构造时校验存在性; 网络流豁免(不是本地文件)。
    """
    path: str = Field(description="视频文件路径或网络流 URL")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if is_stream_url(v):
            return v                       # 网络流不做本地存在性校验
        if not Path(v).is_file():
            raise ValueError(f"视频文件不存在: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path


class ImageConfig(SourceConfig):
    """
    单张图片配置。仅支持本地文件(cv2.imread 拉不了远程 URL, 故拒绝 URL)。
    """
    path: str = Field(description="本地图片文件路径")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if is_stream_url(v):
            raise ValueError(f"图片源不支持 URL(cv2.imread 无法拉取远程图片): {v}")
        if not Path(v).is_file():
            raise ValueError(f"图片文件不存在: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path


class ImageFolderConfig(SourceConfig):
    """
    图片文件夹配置。path 必须是已存在的目录。
    """
    path: str = Field(description="图片文件夹路径")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, v: str) -> str:
        if not Path(v).is_dir():
            raise ValueError(f"不是有效文件夹: {v}")
        return v

    @property
    def identity(self) -> str:
        return self.path