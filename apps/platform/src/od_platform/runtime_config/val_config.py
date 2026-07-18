#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : val.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——YOLO 验证配置类
"""YOLO 验证配置: YOLOValConfig.

继承 BaseConfig + 添加 11 个验证特有字段, 分 4 组:
    核心参数 / 验证控制 / 评估设置 / 任务特定

★ 跟 train 一致的设计:
   - task + experiment_name 拆分
   - extra='forbid' 拒绝拼错的字段名
   - 验证器走 field_validator / model_validator

★ 跟 train 不一致的地方:
   - val 模式下 task 不传给 ultralytics(从权重推断), FRAMEWORK_ONLY_FIELDS 多一个 task
   - conf 默认 0.001 (跟推理的 0.25 差两个数量级, 见 3.6 设计点 B)
"""
from __future__ import annotations

import warnings
from typing import ClassVar, Optional

from pydantic import Field, field_validator, model_validator

from od_platform.common.constants import Task
from od_platform.runtime_config.base_config import BaseConfig


class YOLOValConfig(BaseConfig):
    """YOLO 验证配置: 继承 BaseConfig + 11 个验证特有字段."""

    # ============================================================
    # 框架级字段扩展:
    #   - experiment_name 不传给 ultralytics
    #   - task 不传给 ultralytics(model.val() 从权重推断)
    # ============================================================
    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = (
        BaseConfig.FRAMEWORK_ONLY_FIELDS | {"experiment_name", "task"}
    )

    # ============================================================
    # 核心参数 (★ 跟 train 共享 task + experiment_name 设计)
    # ============================================================

    task: str = Field(
        default=Task.DETECT,
        description="任务类型(SSoT)",
        json_schema_extra={
            "group":    "核心参数",
            "examples": list(Task.all()),
            "tips": [
                "限定取值: " + " / ".join(Task.all()),
                "决定输出目录的一级子目录",
                "val 模式下不传给 ultralytics(从权重自动推断)",
            ],
            "yaml_comment": "任务类型(只能是 detect / segment)",
        },
    )

    experiment_name: Optional[str] = Field(
        default=None,
        description="实验名(只用作目录组织)",
        json_schema_extra={
            "group":    "核心参数",
            "examples": ["val_baseline", "val_helmet_v1", None],
            "tips": [
                "None: 由 ultralytics 自动生成 val / val2 / val3...",
                "自定义: 用作 runs/<task>/<experiment_name>/ 子目录",
            ],
            "yaml_comment": "实验名(目录组织, None=自动生成)",
        },
    )

    # ============================================================
    # 验证控制
    # ============================================================

    split: str = Field(
        default="val",
        description="数据集划分",
        json_schema_extra={
            "group":    "验证控制",
            "examples": ["val", "test", "train"],
            "tips": [
                "val: 使用验证集(默认)",
                "test: 使用测试集",
                "train: 使用训练集(调试用)",
                "对应 data.yaml 里的字段名",
            ],
            "yaml_comment": "数据集划分(val / test / train)",
        },
    )

    conf: Optional[float] = Field(
        default=0.001,
        ge=0.0, le=1.0,
        description="置信度阈值",
        json_schema_extra={
            "group":    "验证控制",
            "examples": [0.001, 0.25, 0.5],
            "tips": [
                "★ 验证时通常设 0.001 (捕获所有预测, 用于计算 mAP)",
                "推理时建议 0.25-0.5 (过滤低置信度)",
                "不影响最终模型, 仅影响评估指标",
            ],
            "yaml_comment": "置信度阈值(验证 0.001; 推理 0.25+)",
        },
    )

    iou: float = Field(
        default=0.6,
        ge=0.0, le=1.0,
        description="NMS IoU 阈值",
        json_schema_extra={
            "group":    "验证控制",
            "examples": [0.5, 0.6, 0.7],
            "tips": [
                "NMS(非极大值抑制) IoU 阈值",
                "COCO 标准: 0.6",
                "密集目标场景可提高到 0.7",
            ],
            "yaml_comment": "NMS IoU 阈值",
        },
    )

    max_det: int = Field(
        default=300, ge=1,
        description="每张图最大检测数",
        json_schema_extra={
            "group":    "验证控制",
            "examples": [100, 300, 1000],
            "tips": [
                "每张图像最多保留的检测框数量",
                "COCO 默认: 300",
                "密集目标场景可增加",
            ],
            "yaml_comment": "每张图最大检测数",
        },
    )

    # ============================================================
    # 评估设置
    # ============================================================

    half: bool = Field(
        default=True,
        description="半精度推理",
        json_schema_extra={
            "group":    "评估设置",
            "examples": [True, False],
            "tips": [
                "True: 使用 FP16 半精度(快 + 省显存)",
                "False: 使用 FP32 全精度",
                "现代 GPU 推荐开启",
                "CPU 验证时自动禁用",
            ],
            "yaml_comment": "是否使用半精度推理",
        },
    )

    plots: bool = Field(
        default=True,
        description="生成评估图表",
        json_schema_extra={
            "group":    "评估设置",
            "examples": [True, False],
            "tips": [
                "True: 生成混淆矩阵、PR 曲线等",
                "False: 不生成图表",
                "批量验证时可关闭以加速",
            ],
            "yaml_comment": "是否生成评估图表",
        },
    )

    save_json: bool = Field(
        default=True,
        description="保存 COCO JSON 结果",
        json_schema_extra={
            "group":    "评估设置",
            "examples": [False, True],
            "tips": [
                "True: 保存 COCO 格式的预测结果",
                "False: 不保存 JSON",
                "用于 COCO 官方评估工具",
                "提交比赛时需要开启",
            ],
            "yaml_comment": "是否保存 COCO JSON 格式结果",
        },
    )

    save_hybrid: bool = Field(
        default=False,
        description="保存混合标签",
        json_schema_extra={
            "group":    "评估设置",
            "examples": [False, True],
            "tips": ["True: 保存混合标签(预测 + 真实)", "False: 不保存", "调试和分析用"],
            "yaml_comment": "是否保存混合标签",
        },
    )

    # ============================================================
    # 任务特定
    # ============================================================

    mask_ratio: int = Field(
        default=4, ge=1,
        description="掩码下采样比例(分割)",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [1, 2, 4],
            "tips": ["仅实例分割任务", "掩码相对原图的下采样倍数"],
            "yaml_comment": "掩码下采样比例(仅分割)",
        },
    )

    overlap_mask: bool = Field(
        default=True,
        description="重叠掩码(分割)",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [True, False],
            "tips": ["仅实例分割任务", "True: 允许掩码重叠"],
            "yaml_comment": "是否允许掩码重叠(仅分割)",
        },
    )

    dnn: bool = Field(
        default=False,
        description="使用 OpenCV DNN",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [False, True],
            "tips": [
                "True: 使用 OpenCV DNN 后端",
                "False: 使用 PyTorch(默认)",
                "DNN 适合边缘设备部署",
            ],
            "yaml_comment": "是否使用 OpenCV DNN 后端",
        },
    )

    # ============================================================
    # 验证器
    # ============================================================

    @field_validator("task")
    @classmethod
    def _validate_task(cls, v: str) -> str:
        """task 必须在 Task SSoT 范围内"""
        if v not in Task.all():
            raise ValueError(
                f"task 必须是 {sorted(Task.all())} 之一, 当前值: {v!r}. "
                f"若想给本次实验取名, 请使用 experiment_name 字段."
            )
        return v

    @field_validator("split")
    @classmethod
    def _validate_split(cls, v: str) -> str:
        """split 必须是 train / val / test"""
        valid = {"train", "val", "test"}
        if v not in valid:
            raise ValueError(f"split 必须是 {sorted(valid)} 之一, 当前: {v!r}")
        return v

    @model_validator(mode="after")
    def _cross_field_validation(self) -> "YOLOValConfig":
        """跨字段验证: segment 专属参数在 detect 任务下警告."""
        if self.task != Task.SEGMENT:
            # 用户在非 segment 任务下改了 segment 专属字段, 提醒一下
            offenders = []
            if self.mask_ratio != 4:
                offenders.append(f"mask_ratio={self.mask_ratio}")
            if self.overlap_mask is False:
                offenders.append(f"overlap_mask={self.overlap_mask}")

            if offenders:
                warnings.warn(
                    f"{' / '.join(offenders)} 仅对 segment 任务有效, "
                    f"当前 task={self.task!r}, 这些值将被忽略.",
                    UserWarning,
                )
        return self