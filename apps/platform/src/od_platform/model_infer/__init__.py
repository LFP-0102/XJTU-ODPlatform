# apps/platform/src/odp_platform/inference/__init__.py
"""ODPlatform 推理子系统对外面板.

业务方按需引入:
    from odp_platform.inference import infer_yolo                       # 一行启动
    from odp_platform.inference import InferService, InferResult, InferStats
    from odp_platform.inference import (
        OutputSink, LocalFileSink, NullSink,                            # 接缝 1
        InferHooks, FrameEvent, ProgressEvent,                          # 接缝 2
        CancelToken, InferenceCancelled,                                # 接缝 3
    )

★ 不导出 ThreadedPipeline / _Reader / _Renderer / _Display / _Controller / _FrameProcessor —
  那些是引擎内部细节, 外部不应该直接持有 (持有了一升级就坏).
"""
from .cancel import CancelToken, InferenceCancelled
from .hooks import FrameEvent, InferHooks, ProgressEvent
from .pipeline_config import PipelineConfig, load_pipeline_config
from .service import InferResult, InferService, InferStats, infer_yolo, log_infer_stats
from .sinks import LocalFileSink, NullSink, OutputSink

__all__ = [
    # 主入口 (跟 D6 train_yolo 平行)
    "infer_yolo",
    # Service / Result / Stats
    "InferService", "InferResult", "InferStats", "log_infer_stats",
    # 配置
    "PipelineConfig", "load_pipeline_config",
    # ★ 接缝 1: 输出
    "OutputSink", "LocalFileSink", "NullSink",
    # ★ 接缝 2: 事件
    "InferHooks", "FrameEvent", "ProgressEvent",
    # ★ 接缝 3: 取消
    "CancelToken", "InferenceCancelled",
]