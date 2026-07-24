#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : __init__.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_train/__init__.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-19 09:19:55
# @Version    : v1.0.0
# @Description:model_train 子系统公共 API
# @ChangeLog:
#   2026-07-19 | Matri | v1.0.0 | 初始化创建
#   2026-07-23 | Matri | v1.0.0 | 新增 TrainTrackingHooks / MlflowTracker 导出
"""model_train — 模型训练子系统.

公共 API:
  编排:
      train_yolo, TrainResult
  结果:
      TrainMetrics, log_train_metrics
  追踪:
      TrainTrackingHooks, TrainStartEvent, FitEpochEndEvent,
      TrainEndEvent, TrainErrorEvent, MlflowTracker,
      build_tracking_hooks_from_config
  归档:
      archive_best_weight
  报告:
      TrainingReport, EpochRow, save_report,
      build_training_report, generate_report
"""
from __future__ import annotations

from od_platform.model_train.service import train_yolo, TrainResult
from od_platform.model_train.result import TrainMetrics, log_train_metrics
from od_platform.model_train.archive import archive_best_weight
from od_platform.model_train.tracking import (
    TrainTrackingHooks,
    TrainStartEvent,
    FitEpochEndEvent,
    TrainEndEvent,
    TrainErrorEvent,
    MlflowTracker,
    build_tracking_hooks_from_config,
)
from od_platform.model_train.report import TrainingReport, EpochRow, save_report
from od_platform.model_train.report_builder import build_training_report, generate_report

__all__ = [
    "train_yolo",
    "TrainResult",
    "TrainMetrics",
    "log_train_metrics",
    "archive_best_weight",
    # 追踪
    "TrainTrackingHooks",
    "TrainStartEvent",
    "FitEpochEndEvent",
    "TrainEndEvent",
    "TrainErrorEvent",
    "MlflowTracker",
    "build_tracking_hooks_from_config",
    # 报告
    "TrainingReport",
    "EpochRow",
    "save_report",
    "build_training_report",
    "generate_report",
]
