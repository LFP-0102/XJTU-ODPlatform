#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : types.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/core/types.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 09:40:42
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:40:42 | Matri | v1.0.0 | 初始化创建
"""
frame_source 核心数据类型定义(多模态版)。

一帧 = 一个"主视觉平面"(image)+ 一份元数据(info), 外加可选的副平面:
    - depth      : 深度图(RGBD 相机), 高频重要模态 → 一等字段
    - aux        : 开放式副平面字典(双目 right / 多目第 N 路)
    - intrinsics : 相机内参(深度反投影点云要用)

彩色源(摄像头/视频/图片/文件夹)每帧只产一张 BGR 图, 只填 image + info,
副平面走默认值——所以老的 `model(frame.image)` 一个字不用改。

模块独立性原则(规矩 D):
    IMAGE_EXTENSIONS / VIDEO_EXTENSIONS / STREAM_SCHEMES 是 frame_source 的
    自有 SSOT, 与宿主项目同名常量内容重叠是巧合, 不是约束。本模块不引用任何
    外部基础设施常量, 保证整包可拷贝至其他项目, 也保证第三方源能独立扩展。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class SourceType(str, Enum):
    """内置输入源类型枚举(继承 str 便于序列化与日志)。

    注意: 这里列的是**内置**四种。第三方源不必、也无法往这个枚举里加成员——
    它们在自己的 FrameInfo 里用自己的类型字符串即可(见 FrameInfo.source_type
    的类型是 str, 不是 SourceType)。这是为了不把第三方源的类型堵死(规矩 E)。
    """
    CAMERA       = "camera"
    VIDEO        = "video"
    IMAGE        = "image"
    IMAGE_FOLDER = "image_folder"


# ── 模块自有 SSOT(frozenset 兼顾不可变 + set 语义)─────────────
IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff",
})
VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm",
})

# 网络流 scheme: cv2.VideoCapture 能直接吃这些 URL, 一律按视频流走 VideoSource
STREAM_SCHEMES: tuple[str, ...] = ("rtsp://", "rtmp://", "http://", "https://")


def is_stream_url(source: str) -> bool:
    """是否是 cv2 能直接打开的网络流 URL(单一事实来源, 全模块共用)。"""
    return source.lower().startswith(STREAM_SCHEMES)


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    相机内参(不可变值对象)。深度相机反投影点云、双目算视差时需要。

    彩色/普通源没有标定信息, intrinsics 恒为 None; 深度/双目源填充它。

    字段:
        fx, fy      : 焦距(像素)
        cx, cy      : 主点(像素)
        depth_scale : 深度原始单位 → 米的比例(如 uint16 毫米图, scale=0.001)
        baseline    : 双目基线(米), 单目/深度可为 None
    """
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float = 1.0
    baseline: Optional[float] = None


@dataclass(frozen=True)
class FrameInfo:
    """
    帧元数据(不可变值对象)。

    所有输入源都返回 width/height(主平面尺寸), 供下游计算字体大小/缩放/输出尺寸等。

    ⚠️ source_type 的类型是 str(不是 SourceType):
        内置源传 SourceType.XXX(枚举成员本身是 str, 天然兼容);
        第三方源传自己的字符串(如 "depth")。故意放宽以支持包外扩展(规矩 E)。

    ⚠️ 关于 frame_index 与 stride 的关系:
        frame_index 是**源帧号**, 不是"调用方收到的第几帧"。
        - 普通迭代(stride=1):  0, 1, 2, 3, ...
        - stride=N 跳采下:     0, N, 2N, 3N, ...(保留原始帧号)
        理由: timestamp = frame_index / fps 才正确; 下游帧标注/回放对照需要原始帧号;
            需要连续序号的调用方自行 enumerate。

    关于 timestamp 字段的语义差异(已知不一致, 故意保留):
        - CameraSource:                    墙钟时间(自 open 起经过的秒数)
        - VideoSource:                     媒体时间(frame_index / fps, 即在视频里的位置)
        - ImageSource / ImageFolderSource: 0.0(无时间概念)
    """
    # 主平面尺寸(所有源都有)
    width: int
    height: int

    # 源信息(source_type 收 str, 兼容内置枚举与第三方字符串)
    source_type: str
    source_path: str

    # 序列信息
    frame_index: int = 0
    total_frames: Optional[int] = None

    # 时间信息(摄像头 / 视频)
    timestamp: float = 0.0
    fps: Optional[float] = None

    # 文件名(所有源都填充; 摄像头填 'camera:<id>' 占位)
    filename: Optional[str] = None

    @property
    def resolution(self) -> tuple[int, int]:
        """主平面分辨率 (width, height)。"""
        return (self.width, self.height)


@dataclass
class Frame:
    """
    帧数据(统一返回类型, 多模态)。

    Attributes:
        image      : 主视觉平面 ndarray。彩色源=BGR(OpenCV 标准), 红外源=热图,
                    双目源=左目。默认零拷贝传递, 调用方需持久保存时自行 .copy()。
        info       : 帧元数据。
        depth      : 深度图 ndarray(RGBD 源), 一般 uint16 毫米; 无深度时 None。
        aux        : 开放式副平面字典(如 {'right': 右目图}); 默认空。
        intrinsics : 相机内参(深度反投影/双目视差用); 无标定时 None。

    设计说明:
        彩色四源每帧只产一张主图, 只填 image + info, 副平面走默认值。所以下游
        `model(frame.image)` 换任何一种彩色源都不用改。多模态源(深度/双目/红外)
        额外填 depth / aux / intrinsics, 需要它们的新代码按需去取。

        深度是一等字段(不塞进 aux): 它是最常见最重要的第二模态, 值得类型提示
        和 has_depth 便捷判断。aux 只接"开放式、不确定有几路"的副平面, 不做杂物袋。
    """
    image: np.ndarray
    info: FrameInfo
    depth: Optional[np.ndarray] = None
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    intrinsics: Optional[CameraIntrinsics] = None

    # ── 能力便捷判断 ──
    @property
    def has_depth(self) -> bool:
        """本帧是否带深度图。彩色源恒 False, 深度源为 True。"""
        return self.depth is not None

    # ── 便捷属性(少写一层 .info)──
    @property
    def resolution(self) -> tuple[int, int]:
        return self.info.resolution

    @property
    def width(self) -> int:
        return self.info.width

    @property
    def height(self) -> int:
        return self.info.height