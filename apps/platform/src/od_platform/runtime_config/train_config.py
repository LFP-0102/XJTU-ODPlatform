#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : train.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——YOLO 训练配置类
"""YOLO 训练配置: YOLOTrainConfig.

继承 BaseConfig + 添加 52 个训练特有字段, 全部分 11 个组:
    核心参数 / 训练控制 / 优化器配置 / 学习率预热 / 损失权重 /
    数据增强-颜色 / 数据增强-几何 / 数据增强-拼接 / 验证和输出 /
    任务特定 / 高级设置

★ 两个设计要点:
   1. task / experiment_name 是两个字段, 各扛一个职责(见 3.2)
   2. FRAMEWORK_ONLY_FIELDS 用 ClassVar 集合扩展, 不 override 方法
"""
from __future__ import annotations

import warnings
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from od_platform.common.constants import Task
from od_platform.runtime_config.base_config import BaseConfig


class YOLOTrainConfig(BaseConfig):
    """YOLO 训练配置: 继承 BaseConfig + 52 个训练特有字段."""

    # ============================================================
    # 框架级字段扩展: experiment_name 不传给 ultralytics
    # (BaseConfig 已经把 verbose 放进去; 这里继续扩)
    # ============================================================
    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = (
        BaseConfig.FRAMEWORK_ONLY_FIELDS | {"experiment_name"}
    )

    # ============================================================
    # 核心参数 (★ D5 重构点: task + experiment_name 拆分)
    # ============================================================

    task: str = Field(
        default=Task.DETECT,
        description="任务类型(SSoT, 封闭取值)",
        json_schema_extra={
            "group":    "核心参数",
            "examples": list(Task.all()),
            "tips": [
                "限定取值: " + " / ".join(Task.all()),
                "决定模型加载方式 + 输出目录的一级子目录",
                "想给实验起名? → 用 experiment_name 字段",
            ],
            "yaml_comment": "任务类型(只能是 detect / segment)",
        },
    )

    experiment_name: Optional[str] = Field(
        default=None,
        description="实验名(只用作目录组织, 不影响任务语义)",
        json_schema_extra={
            "group":    "核心参数",
            "examples": ["baseline", "helmet_v1", "lr001_aug_strong", None],
            "tips": [
                "None: 由 ultralytics 自动生成 exp / exp2 / exp3...",
                "自定义: 用作 runs/<task>/<experiment_name>/ 子目录",
                "推荐命名: <模型>_<数据集>_<关键调参>",
            ],
            "yaml_comment": "实验名(目录组织, 不影响语义; None=自动生成)",
        },
    )

    # ============================================================
    # 训练控制
    # ============================================================

    epochs: int = Field(
        default=100,
        ge=1,
        description="训练总轮数",
        json_schema_extra={
            "group":    "训练控制",
            "examples": [50, 100, 200, 300],
            "tips": [
                "小数据集: 50-100 轮",
                "中等数据集: 100-200 轮",
                "大数据集 / 工业检测: 200-300 轮",
                "过拟合风险时可减少轮数",
            ],
            "yaml_comment": "训练总轮数(epoch)",
        },
    )

    time: Optional[float] = Field(
        default=None,
        gt=0,
        description="最长训练时间(小时)",
        json_schema_extra={
            "group":    "训练控制",
            "examples": [1.0, 2.5, 12.0, 24.0],
            "tips": [
                "限制最长训练时间(小时), 设置后可提前终止",
                "适合有时间限制的场景",
                "None 表示不限制",
            ],
            "yaml_comment": "最长训练时间(小时, None=不限制)",
        },
    )

    patience: int = Field(
        default=50,
        ge=0,
        description="Early Stopping 耐心值",
        json_schema_extra={
            "group":    "训练控制",
            "examples": [50, 100, 150],
            "tips": [
                "多少轮无提升后停止训练",
                "0 表示禁用 Early Stopping",
                "建议设为 epochs 的 30-50%",
            ],
            "yaml_comment": "Early Stopping 耐心值(0=禁用)",
        },
    )

    resume: bool = Field(
        default=False,
        description="恢复训练",
        json_schema_extra={
            "group":    "训练控制",
            "examples": [False, True],
            "tips": [
                "True: 从最近的检查点恢复",
                "False: 从头开始训练",
                "需要已有 last.pt 检查点",
                "恢复时会保留优化器状态",
            ],
            "yaml_comment": "是否从检查点恢复训练",
        },
    )

    close_mosaic: int = Field(
        default=10,
        ge=0,
        description="关闭 Mosaic 的轮数",
        json_schema_extra={
            "group":    "训练控制",
            "examples": [0, 10, 15],
            "tips": [
                "最后 N 轮关闭 mosaic 增强",
                "有助于模型适应真实分布",
                "0 表示不关闭",
                "建议 10-15 轮",
            ],
            "yaml_comment": "最后 N 轮关闭 mosaic(0=不关闭)",
        },
    )

    multi_scale: bool = Field(
        default=False,
        description="多尺度训练",
        json_schema_extra={
            "group":    "训练控制",
            "examples": [False, True],
            "tips": [
                "True: 随机改变输入尺寸(±50%)",
                "提升模型对不同尺度的鲁棒性",
                "会略微增加训练时间",
                "小目标检测时推荐开启",
            ],
            "yaml_comment": "是否启用多尺度训练",
        },
    )

    fraction: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="训练数据使用比例",
        json_schema_extra={
            "group":    "训练控制",
            "examples": [0.1, 0.5, 1.0],
            "tips": [
                "1.0 表示使用全部数据",
                "0.1 表示使用 10% 数据",
                "调试时可设为小值快速验证",
            ],
            "yaml_comment": "训练数据使用比例(0-1)",
        },
    )

    freeze: Optional[Union[int, List[int]]] = Field(
        default=None,
        description="冻结层",
        json_schema_extra={
            "group":    "训练控制",
            "examples": [None, 10, [0, 1, 2, 3]],
            "tips": [
                "None: 不冻结任何层",
                "int N: 冻结前 N 层(常用 10 表示冻结 backbone)",
                "list: 冻结指定层索引",
                "迁移学习场景使用",
            ],
            "yaml_comment": "冻结层(None / int / list)",
        },
    )

    # ============================================================
    # 优化器配置
    # ============================================================

    optimizer: str = Field(
        default="auto",
        description="优化器类型",
        json_schema_extra={
            "group":    "优化器配置",
            "examples": ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam"],
            "tips": [
                "auto: 自动选择(推荐)",
                "SGD: 传统随机梯度下降",
                "Adam / AdamW: 自适应学习率",
                "小数据集推荐 AdamW",
            ],
            "yaml_comment": "优化器(auto / SGD / Adam / AdamW)",
        },
    )

    lr0: float = Field(
        default=0.01, gt=0,
        description="初始学习率",
        json_schema_extra={
            "group":    "优化器配置",
            "examples": [0.001, 0.01, 0.1],
            "tips": [
                "SGD 推荐: 0.01",
                "Adam / AdamW 推荐: 0.001",
                "大 batch 可适当增大",
                "小 batch 需适当减小",
            ],
            "yaml_comment": "初始学习率",
        },
    )

    lrf: float = Field(
        default=0.01, gt=0,
        description="最终学习率比例",
        json_schema_extra={
            "group":    "优化器配置",
            "examples": [0.01, 0.1, 0.2],
            "tips": [
                "★ 最终学习率 = lr0 × lrf",
                "0.01 表示衰减到初始的 1%",
                "配合余弦退火 cos_lr=true 使用",
                "一般保持默认值即可",
            ],
            "yaml_comment": "最终学习率比例(final_lr = lr0 × lrf)",
        },
    )

    momentum: float = Field(
        default=0.937, gt=0, lt=1.0,
        description="SGD 动量 / Adam beta1",
        json_schema_extra={
            "group":    "优化器配置",
            "examples": [0.9, 0.937, 0.95],
            "tips": [
                "SGD 动量参数 / Adam 的 beta1",
                "一般保持默认值",
                "增大可提高稳定性",
            ],
            "yaml_comment": "动量参数",
        },
    )

    weight_decay: float = Field(
        default=0.0005, ge=0,
        description="权重衰减(L2 正则化)",
        json_schema_extra={
            "group":    "优化器配置",
            "examples": [0.0, 0.0001, 0.0005, 0.001],
            "tips": [
                "防止过拟合",
                "小数据集可增大(0.001)",
                "大数据集可减小(0.0001)",
                "0 表示不使用正则化",
            ],
            "yaml_comment": "权重衰减(L2 正则)",
        },
    )

    cos_lr: bool = Field(
        default=False,
        description="余弦学习率调度",
        json_schema_extra={
            "group":    "优化器配置",
            "examples": [False, True],
            "tips": [
                "True: 余弦退火",
                "False: 线性退火",
                "长训练周期推荐开启",
            ],
            "yaml_comment": "是否使用余弦学习率调度",
        },
    )

    # ============================================================
    # 学习率预热
    # ============================================================

    warmup_epochs: float = Field(
        default=3.0, ge=0,
        description="预热轮数",
        json_schema_extra={
            "group":    "学习率预热",
            "examples": [0.0, 1.0, 3.0, 5.0],
            "tips": [
                "前 N 轮逐渐增加学习率, 避免初期震荡",
                "3-5 轮通常足够",
                "0 表示禁用预热",
            ],
            "yaml_comment": "学习率预热轮数",
        },
    )

    warmup_momentum: float = Field(
        default=0.8, ge=0, le=1.0,
        description="预热初始动量",
        json_schema_extra={
            "group":    "学习率预热",
            "examples": [0.5, 0.8, 0.9],
            "tips": [
                "预热阶段的初始动量, 逐渐增加到 momentum",
                "一般保持默认值",
            ],
            "yaml_comment": "预热阶段初始动量",
        },
    )

    warmup_bias_lr: float = Field(
        default=0.1, ge=0,
        description="预热 bias 学习率",
        json_schema_extra={
            "group":    "学习率预热",
            "examples": [0.0, 0.1, 0.2],
            "tips": [
                "预热阶段 bias 层的学习率",
                "通常设为 lr0 的 10 倍",
                "一般保持默认值",
            ],
            "yaml_comment": "预热阶段 bias 学习率",
        },
    )

    # ============================================================
    # 损失权重
    # ============================================================

    box: float = Field(
        default=7.5, ge=0,
        description="Box Loss 权重",
        json_schema_extra={
            "group":    "损失权重",
            "examples": [5.0, 7.5, 10.0],
            "tips": ["边界框回归损失权重", "定位不准时可增大"],
            "yaml_comment": "Box Loss 权重",
        },
    )

    cls: float = Field(
        default=0.5, ge=0,
        description="Cls Loss 权重",
        json_schema_extra={
            "group":    "损失权重",
            "examples": [0.3, 0.5, 1.0],
            "tips": ["分类损失权重", "类别混淆严重时可增大"],
            "yaml_comment": "Cls Loss 权重",
        },
    )

    dfl: float = Field(
        default=1.5, ge=0,
        description="DFL Loss 权重",
        json_schema_extra={
            "group":    "损失权重",
            "examples": [1.0, 1.5, 2.0],
            "tips": ["Distribution Focal Loss 权重", "改善边界框质量"],
            "yaml_comment": "DFL Loss 权重",
        },
    )

    pose: float = Field(
        default=12.0, ge=0,
        description="Pose Loss 权重(姿态任务)",
        json_schema_extra={
            "group":    "损失权重",
            "examples": [10.0, 12.0, 15.0],
            "tips": ["仅姿态估计任务", "检测任务时不生效"],
            "yaml_comment": "Pose Loss 权重(仅姿态任务)",
        },
    )

    kobj: float = Field(
        default=2.0, ge=0,
        description="Keypoint Obj Loss 权重(姿态任务)",
        json_schema_extra={
            "group":    "损失权重",
            "examples": [1.0, 2.0, 3.0],
            "tips": ["仅姿态估计任务", "关键点置信度损失"],
            "yaml_comment": "Keypoint Obj Loss 权重",
        },
    )

    nbs: int = Field(
        default=64, ge=1,
        description="标称批次大小",
        json_schema_extra={
            "group":    "损失权重",
            "examples": [16, 32, 64],
            "tips": ["用于损失归一化", "与实际 batch 可以不同", "一般保持默认值 64"],
            "yaml_comment": "标称批次大小(损失归一化)",
        },
    )

    # ============================================================
    # 数据增强 - 颜色
    # ============================================================

    hsv_h: float = Field(
        default=0.015, ge=0.0, le=1.0,
        description="色调扰动",
        json_schema_extra={
            "group":    "数据增强-颜色",
            "examples": [0.0, 0.015, 0.05],
            "tips": ["HSV 色调扰动幅度", "0 表示禁用", "光照变化大时可增大"],
            "yaml_comment": "色调扰动(0-1)",
        },
    )

    hsv_s: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="饱和度扰动",
        json_schema_extra={
            "group":    "数据增强-颜色",
            "examples": [0.0, 0.5, 0.7],
            "tips": ["HSV 饱和度扰动幅度", "0.7 为中等增强"],
            "yaml_comment": "饱和度扰动(0-1)",
        },
    )

    hsv_v: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="亮度扰动",
        json_schema_extra={
            "group":    "数据增强-颜色",
            "examples": [0.0, 0.3, 0.4],
            "tips": ["HSV 亮度扰动幅度", "明暗变化大时可增大"],
            "yaml_comment": "亮度扰动(0-1)",
        },
    )

    bgr: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="BGR 通道翻转概率",
        json_schema_extra={
            "group":    "数据增强-颜色",
            "examples": [0.0, 0.1, 0.5],
            "tips": ["RGB ↔ BGR 随机翻转", "增强颜色鲁棒性", "一般不需要开启"],
            "yaml_comment": "BGR 通道翻转概率(0-1)",
        },
    )

    # ============================================================
    # 数据增强 - 几何
    # ============================================================

    degrees: float = Field(
        default=0.0, ge=0.0, le=180.0,
        description="随机旋转角度",
        json_schema_extra={
            "group":    "数据增强-几何",
            "examples": [0.0, 10.0, 30.0],
            "tips": ["随机旋转最大角度", "物体方向多样时可增大", "建议 10-30 度"],
            "yaml_comment": "随机旋转角度(0-180)",
        },
    )

    translate: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="随机平移比例",
        json_schema_extra={
            "group":    "数据增强-几何",
            "examples": [0.0, 0.1, 0.2],
            "tips": ["图像平移的最大比例", "0.1 表示最多平移 10%"],
            "yaml_comment": "随机平移比例(0-1)",
        },
    )

    scale: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="随机缩放增益",
        json_schema_extra={
            "group":    "数据增强-几何",
            "examples": [0.0, 0.3, 0.5],
            "tips": ["图像缩放增益范围", "0.5 表示 0.5-1.5 倍缩放"],
            "yaml_comment": "随机缩放增益(0-1)",
        },
    )

    shear: float = Field(
        default=0.0, ge=-180.0, le=180.0,
        description="随机错切角度",
        json_schema_extra={
            "group":    "数据增强-几何",
            "examples": [0.0, 2.0, 5.0],
            "tips": ["图像错切变换角度", "一般不需要太大", "建议 0-5 度"],
            "yaml_comment": "随机错切角度(-180到180)",
        },
    )

    perspective: float = Field(
        default=0.0, ge=0.0, le=0.001,
        description="随机透视变换",
        json_schema_extra={
            "group":    "数据增强-几何",
            "examples": [0.0, 0.0001, 0.0005],
            "tips": ["透视变换幅度", "0 表示禁用", "一般保持较小值"],
            "yaml_comment": "透视变换幅度(0-0.001)",
        },
    )

    flipud: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="上下翻转概率",
        json_schema_extra={
            "group":    "数据增强-几何",
            "examples": [0.0, 0.5],
            "tips": ["物体上下对称时可开启", "大多数情况保持 0"],
            "yaml_comment": "上下翻转概率(0-1)",
        },
    )

    fliplr: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="左右翻转概率",
        json_schema_extra={
            "group":    "数据增强-几何",
            "examples": [0.0, 0.5, 1.0],
            "tips": ["物体左右对称时推荐开启", "常用的数据增强手段"],
            "yaml_comment": "左右翻转概率(0-1)",
        },
    )

    # ============================================================
    # 数据增强 - 拼接混合
    # ============================================================

    mosaic: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Mosaic 增强概率",
        json_schema_extra={
            "group":    "数据增强-拼接",
            "examples": [0.0, 0.5, 1.0],
            "tips": [
                "4 张图拼成一张",
                "1.0 表示总是使用",
                "增强小目标检测",
                "训练后期自动关闭(close_mosaic)",
            ],
            "yaml_comment": "Mosaic 增强概率(0-1)",
        },
    )

    mixup: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="MixUp 增强概率",
        json_schema_extra={
            "group":    "数据增强-拼接",
            "examples": [0.0, 0.1, 0.15],
            "tips": ["两张图混合", "0.1-0.15 为合理范围", "可提升泛化能力"],
            "yaml_comment": "MixUp 增强概率(0-1)",
        },
    )

    copy_paste: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Copy-Paste 增强概率(分割)",
        json_schema_extra={
            "group":    "数据增强-拼接",
            "examples": [0.0, 0.1, 0.5],
            "tips": ["仅实例分割任务使用", "复制粘贴实例到其他图像"],
            "yaml_comment": "Copy-Paste 概率(仅分割, 0-1)",
        },
    )

    # ============================================================
    # 验证和输出
    # ============================================================

    val: bool = Field(
        default=True,
        description="训练期间验证",
        json_schema_extra={
            "group":    "验证和输出",
            "examples": [True, False],
            "tips": [
                "True: 每轮后在验证集上评估",
                "False: 不验证(快速调试用)",
                "正式训练必须开启",
            ],
            "yaml_comment": "是否在训练期间验证",
        },
    )

    plots: bool = Field(
        default=True,
        description="生成训练曲线图",
        json_schema_extra={
            "group":    "验证和输出",
            "examples": [False, True],
            "tips": ["生成损失曲线、mAP 曲线等", "保存在输出目录"],
            "yaml_comment": "是否生成训练可视化图",
        },
    )

    save_period: int = Field(
        default=-1,
        description="检查点保存周期",
        json_schema_extra={
            "group":    "验证和输出",
            "examples": [-1, 10, 20, 50],
            "tips": [
                "-1: 只保存 last.pt 和 best.pt",
                "> 0: 每 N 轮保存一次检查点",
                "长时间训练推荐开启",
            ],
            "yaml_comment": "检查点保存周期(-1=禁用)",
        },
    )

    # ============================================================
    # 任务特定
    # ============================================================

    overlap_mask: bool = Field(
        default=True,
        description="合并重叠 Mask(分割)",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [True, False],
            "tips": ["仅实例分割任务使用", "True: 合并重叠 mask"],
            "yaml_comment": "是否合并重叠 mask(仅分割)",
        },
    )

    mask_ratio: int = Field(
        default=4, ge=1,
        description="Mask 下采样率(分割)",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [2, 4, 8],
            "tips": ["仅实例分割任务使用", "mask 下采样倍数"],
            "yaml_comment": "Mask 下采样率(仅分割)",
        },
    )

    dropout: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Dropout 比例(分类)",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [0.0, 0.2, 0.5],
            "tips": ["仅分类任务使用", "防止过拟合"],
            "yaml_comment": "Dropout 比例(仅分类, 0-1)",
        },
    )

    copy_paste_mode: str = Field(
        default="flip",
        description="Copy-Paste 模式(分割)",
        json_schema_extra={
            "group":    "任务特定",
            "examples": ["flip", "mixup"],
            "tips": ["仅分割任务", "flip: 翻转模式", "mixup: 混合模式"],
            "yaml_comment": "Copy-Paste 模式(flip / mixup, 仅分割)",
        },
    )

    auto_augment: str = Field(
        default="randaugment",
        description="自动增强策略(分类)",
        json_schema_extra={
            "group":    "任务特定",
            "examples": ["randaugment", "autoaugment", "augmix"],
            "tips": ["仅分类任务"],
            "yaml_comment": "自动增强策略(仅分类)",
        },
    )

    erasing: float = Field(
        default=0.4, ge=0.0, le=0.9,
        description="随机擦除概率(分类)",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [0.0, 0.2, 0.4],
            "tips": ["仅分类任务", "随机擦除图像区域"],
            "yaml_comment": "随机擦除概率(仅分类, 0-0.9)",
        },
    )

    # ============================================================
    # 高级设置
    # ============================================================

    pretrained: Union[bool, str] = Field(
        default=True,
        description="预训练权重",
        json_schema_extra={
            "group":    "高级设置",
            "examples": [True, False, "path/to/weights.pt"],
            "tips": [
                "True: 使用官方预训练权重",
                "False: 从头训练",
                "str: 自定义权重路径",
                "小数据集强烈推荐使用预训练",
            ],
            "yaml_comment": "预训练权重(True / False / path)",
        },
    )

    single_cls: bool = Field(
        default=False,
        description="单类训练模式",
        json_schema_extra={
            "group":    "高级设置",
            "examples": [False, True],
            "tips": ["True: 将多类数据集视为单类", "特殊场景使用"],
            "yaml_comment": "是否单类训练",
        },
    )

    classes: Optional[List[int]] = Field(
        default=None,
        description="训练指定类别",
        json_schema_extra={
            "group":    "高级设置",
            "examples": [None, [0], [0, 1, 2]],
            "tips": ["None: 训练所有类别", "list: 只训练指定类别 ID"],
            "yaml_comment": "训练指定类别(None / [0,1,2])",
        },
    )

    compile: Union[bool, str] = Field(
        default=False,
        description="模型编译(PyTorch 2.0+)",
        json_schema_extra={
            "group":    "高级设置",
            "examples": [False, True, "default"],
            "tips": ["PyTorch 2.0+ 支持", "首次编译会较慢"],
            "yaml_comment": "是否编译模型(PyTorch 2.0+)",
        },
    )

    profile: bool = Field(
        default=False,
        description="速度分析",
        json_schema_extra={
            "group":    "高级设置",
            "examples": [False, True],
            "tips": ["ONNX / TensorRT 速度分析", "训练时一般不需要"],
            "yaml_comment": "是否进行速度分析",
        },
    )

    augmentations: Optional[List[Any]] = Field(
        default=None,
        description="自定义增强(Albumentations)",
        json_schema_extra={
            "group":    "高级设置",
            "examples": [None],
            "tips": ["None: 使用默认增强", "list: 自定义 Albumentations 变换"],
            "yaml_comment": "自定义增强(高级用户)",
        },
    )

    # ============================================================
    # 验证器
    # ============================================================

    @field_validator("task")
    @classmethod
    def _validate_task(cls, v: str) -> str:
        """task 必须在 Task SSoT 范围内(拒绝 'detect_v1' / 'pose' 等)"""
        if v not in Task.all():
            raise ValueError(
                f"task 必须是 {sorted(Task.all())} 之一, 当前值: {v!r}. "
                f"若想给本次实验取名, 请使用 experiment_name 字段."
            )
        return v

    @field_validator("copy_paste_mode")
    @classmethod
    def _validate_copy_paste_mode(cls, v: str) -> str:
        if v not in {"flip", "mixup"}:
            raise ValueError(f"copy_paste_mode 必须是 'flip' 或 'mixup', 当前: {v!r}")
        return v

    @field_validator("compile")
    @classmethod
    def _validate_compile(cls, v: Union[bool, str]) -> Union[bool, str]:
        if isinstance(v, str):
            return v.strip()
        return v

    @model_validator(mode="after")
    def _cross_field_validation(self) -> "YOLOTrainConfig":
        """跨字段验证: save / save_period / mosaic / close_mosaic 的合理性"""

        # save=False 时不应再设置 save_period > 0 — 矛盾意图
        if not self.save and self.save_period > 0:
            raise ValueError(
                "save=False 时不应设置 save_period > 0. "
                "如果想关闭检查点保存, save_period 也应设为 -1; "
                "如果想周期性保存, save 必须为 True."
            )

        # close_mosaic 在 mosaic=0 时无效 — 冗余但不阻塞
        if self.close_mosaic > 0 and self.mosaic == 0.0:
            warnings.warn(
                "mosaic=0.0 但设置了 close_mosaic > 0, "
                "close_mosaic 将不会生效.",
                UserWarning,
            )

        return self