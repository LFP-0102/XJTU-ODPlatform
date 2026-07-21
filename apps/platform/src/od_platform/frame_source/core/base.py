#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : base.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/core/base.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 10:11:21
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:11:21 | Matri | v1.0.0 | 初始化创建
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .config import  SourceConfig
from .types import Frame, SourceType


logger = logging.getLogger(__name__)

class FrameSource(ABC):
    def __init__(self, config: SourceConfig):
        self.config = config
        self.source_path = config.identity
        self._frame_index = 0
        self._stride = 1
        self._start_time = time.time()

    @abstractmethod
    def open(self) -> bool:
        """打开输入源，返回是否成功，实现必须复位位置状态"""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """读取一帧数据，源耗尽时返回None"""

    @abstractmethod
    def close(self) -> None:
        """关闭输入源"""

    @abstractmethod
    def get_source_type(self) -> SourceType:
        """获取输入源类型"""

    @property
    def modalities(self) -> frozenset[str]:
        return  frozenset({'image'})

    def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        """跳转到指定帧，返回是否成功"""
        logger.warning(f"{self.__class__.__name__}不支持seek操作")
        return False

    @property
    def seekable(self) -> bool:
        """是否支持seek操作,子类覆盖返回True"""
        return False

    def set_stride(self, stride: int) -> None:
        """设置步长"""
        if stride < 1:
            raise ValueError(f"步长必须大于等于1. 收到的值为: {stride}")
        self._stride = stride
        if stride > 1:
            logger.debug(f"{self.__class__.__name__}: stride设置为:{stride}")

    @property
    def stride(self) -> int:
        return self._stride

    # 实现上下文管理器协议
    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    # 迭代器协议
    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
