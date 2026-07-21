#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : video.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/sources/video.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 11:01:39
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:01:39 | Matri | v1.0.0 | 初始化创建
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : video.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 视频源 —— 本地视频文件 + rtsp/rtmp/http(s) 网络流
"""
视频输入源(VideoSource)。自注册, 登记字符串规则(视频后缀 + 网络流 URL 能自证身份)。

关键处理:
    - stride 用 cap.grab() 跳采(跳过 YUV→BGR 转换, 快 3~5 倍)
    - 网络流诚实标注: is_stream_url 检测 → seekable=False, total_frames=None
    - fps 缺失/异常 → 明确 warning + 置 None, 按秒 seek 直接拒绝
    - seek 与 stride 交互: _just_sought 标志, seek 后下一 read 不做 stride 预跳
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import cv2

from ..core.base     import FrameSource
from ..core.config   import VideoConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType, VIDEO_EXTENSIONS, is_stream_url


logger = logging.getLogger(__name__)


@register_source(
    VideoConfig,
    # 能自证身份: 网络流 URL, 或后缀属于视频扩展名
    str_matcher=lambda s: is_stream_url(s) or Path(s).suffix.lower() in VIDEO_EXTENSIONS,
    str_to_config=lambda s: VideoConfig(path=s),
)
class VideoSource(FrameSource):
    """视频文件 / 网络流源。"""

    def __init__(self, config: VideoConfig):
        super().__init__(config)
        self._path: str = config.path
        self._is_stream: bool = is_stream_url(config.path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: Optional[float] = None
        self._total: Optional[int] = None
        self._just_sought: bool = False

    def open(self) -> bool:
        # 复位位置状态
        self._frame_index = 0
        self._just_sought = False

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            logger.error(f"无法打开视频源: {self._path}")
            self._cap = None
            return False

        # fps: 缺失/0/NaN → 明确置 None
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or math.isnan(fps):
            logger.warning(f"视频 fps 缺失或异常({fps}), 时间相关功能(时间戳/按秒 seek)不可用: {self._path}")
            self._fps = None
        else:
            self._fps = float(fps)

        # total_frames: 网络流没有总帧数(无限流); 文件读 FRAME_COUNT, ≤0 也置 None(诚实)
        if self._is_stream:
            self._total = None
            logger.info(f"网络流已打开(不可 seek, 无总帧数): {self._path}")
        else:
            n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._total = n if n > 0 else None
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None

        # stride: 非首帧且未刚 seek 过时, grab 跳过 stride-1 帧(只解复用不解码, 快)
        if self._stride > 1 and self._frame_index > 0 and not self._just_sought:
            for _ in range(self._stride - 1):
                if not self._cap.grab():
                    return None
        self._just_sought = False   # 消费掉 seek 标志

        ret, img = self._cap.read()
        if not ret:
            return None

        # 当前帧号: 文件用解码器位置(POS_FRAMES 反映 seek); 流不可靠 → 退回自增计数
        if self._is_stream:
            idx = self._frame_index
        else:
            idx = max(0, int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self._frame_index = idx + 1     # 下一帧位置

        h, w = img.shape[:2]
        timestamp = (idx / self._fps) if self._fps else 0.0
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.VIDEO,
            source_path=self._path,
            frame_index=idx, total_frames=self._total,
            timestamp=timestamp, fps=self._fps,
            filename=(self._path if self._is_stream else Path(self._path).name),
        )
        return Frame(image=img, info=info)   # 零拷贝

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.VIDEO

    # ── seek(仅文件, 流不可 seek)──
    @property
    def seekable(self) -> bool:
        return self._cap is not None and not self._is_stream

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        if not self.seekable:
            reason = "网络流" if self._is_stream else "未打开的视频"
            logger.warning(f"{reason}不支持 seek")
            return False
        if frame is None and time_sec is None:
            logger.warning("seek 需要 frame 或 time_sec 之一")
            return False

        if time_sec is not None:
            if self._fps is None:
                logger.warning("视频无 fps, 无法按秒 seek")
                return False
            frame = int(time_sec * self._fps)

        frame = max(0, frame)
        ok = self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame))
        if ok:
            self._frame_index = frame
            self._just_sought = True    # 下一次 read 不做 stride 预跳, 直接返回本帧
        return ok