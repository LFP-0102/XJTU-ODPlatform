#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_report — 训练分析报告自动生成.

公共 API:
  构建:  build_training_report, generate_report
  数据:  TrainingReport, EpochRow
  输出:  save_report
"""
from __future__ import annotations

from od_platform.train_report.report import TrainingReport, save_report
from od_platform.train_report.readers import EpochRow
from od_platform.train_report.builder import build_training_report, generate_report

__all__ = [
    "TrainingReport",
    "EpochRow",
    "build_training_report",
    "generate_report",
    "save_report",
]
