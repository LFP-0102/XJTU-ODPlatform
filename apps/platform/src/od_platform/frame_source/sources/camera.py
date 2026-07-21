#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : camera.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 摄像头源 —— 跨平台后端协商 + 高帧率参数验证
"""
摄像头输入源(CameraSource)。自注册, 登记字符串规则("0"/"1" 这类裸数字 = 默认 webcam)。

三条撞墙(高帧率协商, 见讲义 §4.3):
    F①  高帧率必须先设编码 MJPG(裸 YUYV 带宽不够)
    F②  set() 只是请求, 必须 read() 一帧才触发驱动协商; 故 open() 里读一帧丢掉是必要动作
    F③  参数顺序: 编码 → 分辨率 → 帧率(MSMF 下顺序错高帧率失效)

跨平台: backend 选 auto/msmf/dshow/v4l2。Windows 高帧率走 msmf, Linux 走 v4l2。

注意: 摄像头是实时源, 不支持 stride(见 set_stride 覆盖)。字符串糖只认裸数字并默认
     映射到 CameraConfig(彩色 webcam); 深度/红外/双目这类同样是"0"的硬件不走字符串,
     必须显式传各自的 config —— 这正是"配置类型做派发轴"的意义(讲义 §1.2、§3.1)。
"""
from __future__ import annotations

import logging
import time
from typing import Optional
import os
import cv2

from ..core.base     import FrameSource
from ..core.config   import CameraConfig
from ..core.registry import register_source
from ..core.types    import Frame, FrameInfo, SourceType


logger = logging.getLogger(__name__)

# 后端名 → OpenCV flag
_BACKEND_FLAGS = {
    "auto":  cv2.CAP_ANY,
    "msmf":  cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "v4l2":  cv2.CAP_V4L2,
}


@register_source(
    CameraConfig,
    str_matcher=lambda s: s.isdigit(),                       # 裸数字 = 默认彩色 webcam
    str_to_config=lambda s: CameraConfig(camera_id=int(s)),
)
class CameraSource(FrameSource):
    """cv2 摄像头源, 带后端协商与参数验证。"""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_fps: Optional[float] = None

    def open(self) -> bool:
        # 复位位置状态
        self._frame_index = 0
        cfg: CameraConfig = self.config

        if cfg.backend == "msmf":
            os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
        backend_flag = _BACKEND_FLAGS.get(cfg.backend, cv2.CAP_ANY)
        self._cap = cv2.VideoCapture(cfg.camera_id, backend_flag)
        if not self._cap.isOpened():
            logger.error(f"无法打开摄像头 {cfg.camera_id}(backend={cfg.backend})")
            self._cap = None
            return False

        # 撞墙 F①: 先设编码为 MJPG, 否则高帧率上不去
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.codec))
        # 撞墙 F③: 顺序 —— 编码 → 分辨率 → 帧率
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS,          cfg.fps)

        # 撞墙 F②: set() 只是请求, read 一帧才真正触发驱动协商
        ret, _ = self._cap.read()
        if not ret:
            logger.error(f"摄像头 {cfg.camera_id} 已打开但读不到帧")
            self._cap.release()
            self._cap = None
            return False

        # 参数验证: 读回实际生效值, 与请求不符则告警
        aw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        af = self._cap.get(cv2.CAP_PROP_FPS)
        if (aw, ah) != (cfg.width, cfg.height):
            logger.warning(f"分辨率: 请求 {cfg.width}x{cfg.height}, 实际 {aw}x{ah}")
        if af and abs(af - cfg.fps) > 1:
            logger.warning(f"帧率: 请求 {cfg.fps}, 实际 {af:.1f}")
        self._actual_fps = float(af) if af and af > 0 else None
        logger.info(f"摄像头 {cfg.camera_id} 就绪: {aw}x{ah} @ {af:.1f}fps(backend={cfg.backend})")
        return True

    def read(self) -> Optional[Frame]:
        if self._cap is None:
            return None
        ret, img = self._cap.read()
        if not ret:
            return None

        idx = self._frame_index
        self._frame_index += 1
        h, w = img.shape[:2]
        info = FrameInfo(
            width=w, height=h,
            source_type=SourceType.CAMERA,
            source_path=self.source_path,
            frame_index=idx, total_frames=None,     # 实时源无总帧数
            timestamp=time.time() - self._start_time,   # 墙钟时间
            fps=self._actual_fps,
            filename=self.source_path,               # 'camera:<id>'
        )
        return Frame(image=img, info=info)           # 零拷贝

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_source_type(self) -> SourceType:
        return SourceType.CAMERA

    def set_stride(self, stride: int) -> None:
        # 摄像头是实时源, 跳采无意义; 覆盖基类, 不改变 self._stride(始终 1)
        if stride != 1:
            logger.warning(
                "摄像头是实时源, 不支持 stride 跳采。"
                "要降频请在消费端丢帧, 或用 ThreadedSource 始终取最新帧。"
            )