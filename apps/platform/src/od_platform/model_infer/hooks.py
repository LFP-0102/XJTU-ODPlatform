#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : hooks.py
# @Project   : ODPlatform
# @Function  : 推理生命周期钩子 (InferHooks + FrameEvent + ProgressEvent)
"""推理生命周期回调.

InferService / ThreadedPipeline 在 4 个时机触发回调:
  on_frame    : 每帧推理 + 绘制完成 (高频, 适合实时画面推送)
  on_progress : 每 N 帧 (低频, 适合进度条 / 状态更新)
  on_complete : 推理收尾 (1 次, 拿到完整 InferResult)
  on_error    : 异常发生 (0 或 1 次, 拿到 Exception)

设计纪律:
  - 所有 fire_xxx 用 try/except 包住, 业务回调抛异常【不】让 pipeline 死
  - 回调里【不要】做阻塞操作 (网络/db), 应塞到队列异步处理
  - on_frame 在 renderer 线程触发, 其它回调在主线程或 service 线程触发

CLI 默认不传 hooks → 全空回调 → 行为跟改造前 100% 一致.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameEvent:
    """每帧推理完后的事件数据.

    image / annotated 都是 ndarray 视图, 业务方如果要异步处理需要先 .copy().
    detections 是从 ultralytics result.boxes 提取的字段字典列表 (可序列化).
    """
    frame_idx: int                      # 帧序号 (从 0 开始)
    image: np.ndarray                   # 原始 BGR 图像
    annotated: np.ndarray               # 画完框的 BGR 图像
    n_detections: int                   # 当前帧检测数
    detections: list = None             # list[dict] {label, conf, xyxy}, 可选


@dataclass
class ProgressEvent:
    """周期性进度事件 (每 progress_interval_frames 帧触发)."""
    frame_idx: int                      # 当前帧序号 (累计)
    total_frames: Optional[int]         # 总帧数 (视频源知道, 摄像头/RTSP 为 None)
    elapsed_sec: float                  # 已运行秒数
    fps_loop: float                     # 端到端 FPS
    fps_infer: float                    # 推理 FPS
    detections_total: int               # 累计检测数


@dataclass
class InferHooks:
    """所有回调钩子. None 表示空回调.

    CLI 默认行为: InferHooks() 全部 None → 触发时全 short-circuit, 零开销.
    """
    on_frame:    Optional[Callable[[FrameEvent],    None]] = None
    on_progress: Optional[Callable[[ProgressEvent], None]] = None
    on_complete: Optional[Callable[[Any],           None]] = None   # InferResult → None
    on_error:    Optional[Callable[[Exception],     None]] = None

    progress_interval_frames: int = 30      # 每 N 帧触发一次 on_progress

    # ------------------------------------------------------------------
    # 触发 helper —— 全部 try/except 包住, 业务回调抛异常不能炸 pipeline
    # ------------------------------------------------------------------
    def fire_frame(self, evt: FrameEvent) -> None:
        if self.on_frame is not None:
            try:
                self.on_frame(evt)
            except Exception as e:
                logger.warning(f"on_frame 回调异常 (已吞): {e}")

    def fire_progress(self, evt: ProgressEvent) -> None:
        if self.on_progress is not None:
            try:
                self.on_progress(evt)
            except Exception as e:
                logger.warning(f"on_progress 回调异常 (已吞): {e}")

    def fire_complete(self, result: Any) -> None:
        if self.on_complete is not None:
            try:
                self.on_complete(result)
            except Exception as e:
                logger.warning(f"on_complete 回调异常 (已吞): {e}")

    def fire_error(self, exc: Exception) -> None:
        if self.on_error is not None:
            try:
                self.on_error(exc)
            except Exception as e:
                logger.warning(f"on_error 回调异常 (已吞): {e}")