#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :registry.py
# @Time      :2026/7/18 14:07:16
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :apps/platform/src/od_platform/runtime_config/registry.py
from __future__ import annotations


from od_platform.runtime_config.infer_config import YOLOInferConfig
from od_platform.runtime_config.val_config import YOLOValConfig
from od_platform.runtime_config.train_config import YOLOTrainConfig
from od_platform.runtime_config.pipeline_config import InferPipelineConfig

CONFIG_REGISTRY: dict[str, tuple[type,str]] = {
    "train":          (YOLOTrainConfig,      "YOLO 训练配置"),
    "val":            (YOLOValConfig,        "YOLO 验证配置"),
    "infer":          (YOLOInferConfig,      "YOLO 推理配置"),
    "infer_pipeline": (InferPipelineConfig,  "推理管线配置(帧源 + 美化)"),
}

__all__ = ["CONFIG_REGISTRY"]


