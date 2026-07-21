#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : image.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/sources/image.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 11:02:27
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:02:27 | Matri | v1.0.0 | 初始化创建
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : image.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 图片源 —— 单张图片 (ImageSource) + 图片文件夹 (ImageFolderSource)
"""
图片输入源。两个类都自注册, 都登记字符串规则(路径能自证身份)。

ImageSource       : 单张图片, read() 只产一帧(第二次返回 None)。
ImageFolderSource : 文件夹, 按排序后的文件列表逐张产出; stride 走索引跳转(零 IO)。

统一零拷贝: read() 返回的 Frame.image 直接引用底层数组, 调用方需持久保存自行 .copy()。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

from ..core.base     import FrameSource
from ..core.config   import ImageConfig, ImageFolderConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType, IMAGE_EXTENSIONS, is_stream_url


logger = logging.getLogger(__name__)


@register_source(
    ImageConfig,
    # 能自证身份: 非 URL 且后缀属于图片扩展名(URL 一律排除, 交给视频源按流处理)
    str_matcher=lambda s: (not is_stream_url(s)) and Path(s).suffix.lower() in IMAGE_EXTENSIONS,
    str_to_config=lambda s: ImageConfig(path=s),
)
class ImageSource(FrameSource):
    """单张图片源。read() 只产一帧。"""

    def __init__(self, config: ImageConfig):
        super().__init__(config)
        self._path: str = config.path
        self._image: Optional["cv2.typing.MatLike"] = None
        self._read_count: int = 0

    def open(self) -> bool:
        # 复位位置状态(兑现"重新 open 即重头"契约)
        self._read_count = 0
        self._frame_index = 0

        self._image = cv2.imread(self._path)
        if self._image is None:
            logger.error(f"无法读取图片: {self._path}")
            return False
        return True

    def read(self) -> Optional[Frame]:
        if self._image is None or self._read_count > 0:
            return None     # 未 open 或已读过 → 耗尽
        self._read_count += 1

        h, w = self._image.shape[:2]
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.IMAGE,
            source_path=self._path,
            frame_index=0, total_frames=1,
            timestamp=0.0,                       # 图片无时间概念
            filename=Path(self._path).name,
        )
        # 零拷贝: 与其它源一致, 调用方需持久保存自行 .copy()
        return Frame(image=self._image, info=info)

    def close(self) -> None:
        self._image = None

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE

    # 单图 stride 无意义(read 一帧即止), 沿用基类默认(set_stride 只登记不生效)


@register_source(
    ImageFolderConfig,
    str_matcher=lambda s: Path(s).is_dir(),      # 是已存在的目录 → 归文件夹源
    str_to_config=lambda s: ImageFolderConfig(path=s),
)
class ImageFolderSource(FrameSource):
    """图片文件夹源。按排序后的文件列表逐张产出。"""

    def __init__(self, config: ImageFolderConfig):
        super().__init__(config)
        self._folder: str = config.path
        self._files: list[Path] = []
        self._current_index: int = 0

    def open(self) -> bool:
        # 复位位置状态
        self._current_index = 0
        self._frame_index = 0

        folder = Path(self._folder)
        # 过滤(后缀小写)+ 排序(帧序稳定可复现)
        self._files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self._files:
            logger.warning(f"文件夹内没有支持的图片: {self._folder}")
            return False
        logger.info(f"文件夹共 {len(self._files)} 张图片: {self._folder}")
        return True

    def read(self) -> Optional[Frame]:
        # stride: 跳过 stride-1 张(纯索引跳转, 零 IO)
        if self._stride > 1 and self._current_index > 0:
            self._current_index += self._stride - 1

        # 跳过读不出的坏图, 直到读到一张或耗尽
        while self._current_index < len(self._files):
            path = self._files[self._current_index]
            img = cv2.imread(str(path))
            if img is None:
                logger.warning(f"跳过无法读取的图片: {path}")
                self._current_index += 1
                continue

            idx = self._current_index
            self._current_index += 1
            self._frame_index = idx

            h, w = img.shape[:2]
            info = FrameInfo(
                width=w, height=h,
                source_type=SourceType.IMAGE_FOLDER,
                source_path=self._folder,
                frame_index=idx, total_frames=len(self._files),
                timestamp=0.0,
                filename=path.name,
            )
            return Frame(image=img, info=info)   # 零拷贝

        return None     # 耗尽

    def close(self) -> None:
        self._files = []

    def get_source_type(self) -> SourceType:
        return SourceType.IMAGE_FOLDER

    # ── 文件夹支持 seek(按文件索引跳转, 廉价)──
    @property
    def seekable(self) -> bool:
        return True

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        if frame is None:
            logger.warning("文件夹源只支持按帧索引 seek(frame=N)")
            return False
        if not (0 <= frame < len(self._files)):
            logger.warning(f"seek 越界: {frame} 不在 [0, {len(self._files)})")
            return False
        self._current_index = frame
        return True