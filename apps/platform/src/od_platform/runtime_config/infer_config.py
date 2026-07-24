#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : infer.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——YOLO 推理配置类
"""YOLO 推理配置: YOLOInferConfig.

继承 BaseConfig + 添加 17 个推理特有字段, 分 5 组:
    核心参数 / 推理控制 / 视频流 / 输出控制 / 任务特定

★ 跟 val 的关键区别:
   1. source 字段 (推理输入: 图/视频/目录/URL/摄像头) —— val 没有
   2. conf 默认 0.25 —— val 默认 0.001 (语义不同)
   3. 视频流字段 (vid_stride, stream, stream_buffer)
   4. 输出可视化字段 (show_labels, line_width 等)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing  import Any, ClassVar, List, Optional, Union

from pydantic import Field, field_validator, model_validator

from od_platform.common.constants     import Task
from od_platform.runtime_config.base_config import BaseConfig


class YOLOInferConfig(BaseConfig):
    """YOLO 推理配置: 继承 BaseConfig + 17 个推理特有字段."""

    # ============================================================
    # 框架级字段扩展: 跟 val 同款 — experiment_name + task 不传给 ultralytics
    # ============================================================
    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = (
        BaseConfig.FRAMEWORK_ONLY_FIELDS | {"experiment_name", "task"}
    )

    # ============================================================
    # 核心参数 (task / experiment_name / source)
    # ============================================================

    task: str = Field(
        default=Task.DETECT,
        description="任务类型(SSoT, 封闭取值)",
        json_schema_extra={
            "group":    "核心参数",
            "examples": list(Task.all()),
            "tips": [
                "限定取值: " + " / ".join(Task.all()),
                "推理时从模型权重自动推断, 此处仅用于日志和目录组织",
                "想给实验起名? → 用 experiment_name 字段",
            ],
            "yaml_comment": "任务类型(只能是 detect / segment)",
        },
    )

    experiment_name: Optional[str] = Field(
        default=None,
        description="实验名(目录组织, 不影响推理结果)",
        json_schema_extra={
            "group":    "核心参数",
            "examples": ["helmet_demo", "video_test", "rtsp_stream_v1", None],
            "tips": [
                "None: 由 ultralytics 自动生成 predict, predict2, ...",
                "自定义: 用作 runs/<task>/<experiment_name>/ 子目录",
            ],
            "yaml_comment": "实验名(None=自动生成)",
        },
    )

    source: Optional[Union[str, Path]] = Field(
        default=None,
        description="推理输入源",
        json_schema_extra={
            "group":    "核心参数",
            "examples": [
                "image.jpg",
                "video.mp4",
                "path/to/images/",
                "rtsp://example.com/live.sdp",
                "0",
                "https://example.com/img.jpg",
            ],
            "tips": [
                "支持: 图像 / 视频 / 目录 / URL / 摄像头索引 / RTSP 流",
                "目录: 自动遍历所有图像文件",
                "0 / '0' / 1: 摄像头编号",
                "None: 必须通过 CLI 传 --source (运行时)",
            ],
            "yaml_comment": "推理输入源(图/视频/目录/URL/摄像头/流)",
        },
    )

    # ============================================================
    # 推理控制
    # ============================================================

    conf: float = Field(
        default=0.25,                   # ★ vs val 的 0.001
        ge=0.0, le=1.0,
        description="置信度阈值",
        json_schema_extra={
            "group":    "推理控制",
            "examples": [0.1, 0.25, 0.5, 0.7],
            "tips": [
                "★ 推理时 0.25-0.5 (过滤低置信度, 给用户看的)",
                "验证时是 0.001 (捕获所有预测, 算 mAP 用)",
                "工业部署常用 0.5+",
                "调高 = 漏检多但误检少",
            ],
            "yaml_comment": "置信度阈值(推理 0.25+, 验证 0.001)",
        },
    )

    iou: float = Field(
        default=0.7,
        ge=0.0, le=1.0,
        description="NMS IoU 阈值",
        json_schema_extra={
            "group":    "推理控制",
            "examples": [0.5, 0.7, 0.45],
            "tips": [
                "NMS 时重叠框去重的 IoU 阈值",
                "调高 = 保留更多重叠框",
                "调低 = 抑制更激进",
                "工业 0.5-0.7 较常用",
            ],
            "yaml_comment": "NMS IoU 阈值",
        },
    )

    max_det: int = Field(
        default=300,
        ge=1,
        description="单图最大检测数",
        json_schema_extra={
            "group":    "推理控制",
            "examples": [100, 300, 1000],
            "tips": [
                "单张图最多保留多少个检测框",
                "密集场景 (人群计数) 调大到 1000+",
                "稀疏场景 50-100 即可",
            ],
            "yaml_comment": "单图最大检测数",
        },
    )

    classes: Optional[List[int]] = Field(
        default=None,
        description="只检测指定类别",
        json_schema_extra={
            "group":    "推理控制",
            "examples": [None, [0], [0, 1, 2]],
            "tips": [
                "None: 检测所有类别",
                "list[int]: 只保留指定类别 ID 的检测结果",
                "类别 ID 由 data.yaml 的 names 决定",
            ],
            "yaml_comment": "只检测的类别 ID 列表(None=全部)",
        },
    )

    agnostic_nms: bool = Field(
        default=False,
        description="类别无关 NMS",
        json_schema_extra={
            "group":    "推理控制",
            "examples": [False, True],
            "tips": [
                "True: 不同类别之间也做 NMS",
                "False: 同类内 NMS, 不同类保留",
                "类别相互排斥时 (一个目标只属于一类) 开 True",
            ],
            "yaml_comment": "是否类别无关 NMS",
        },
    )

    augment: bool = Field(
        default=False,
        description="测试时增强 (TTA)",
        json_schema_extra={
            "group":    "推理控制",
            "examples": [False, True],
            "tips": [
                "True: 多尺度+翻转推理后融合, 精度↑速度↓",
                "False: 单次推理",
                "对比赛/打榜场景适用, 工业部署一般 False",
            ],
            "yaml_comment": "是否启用 TTA 测试时增强",
        },
    )

    # ============================================================
    # 视频/流
    # ============================================================

    vid_stride: int = Field(
        default=1,
        ge=1,
        description="视频抽帧间隔",
        json_schema_extra={
            "group":    "视频流",
            "examples": [1, 2, 5],
            "tips": [
                "每 N 帧推理一次",
                "1: 处理每一帧(默认)",
                "5: 每 5 帧推理一次(加速)",
                "仅对视频/流有效, 图像无视此参数",
            ],
            "yaml_comment": "视频抽帧间隔(每 N 帧推理一次)",
        },
    )

    stream: bool = Field(
        default=False,
        description="流式推理",
        json_schema_extra={
            "group":    "视频流",
            "examples": [False, True],
            "tips": [
                "True: 返回 generator (适合长视频/流, 内存友好)",
                "False: 返回 list (一次性出结果, 适合短视频/单图)",
                "实时摄像头 / RTSP 必须 True",
            ],
            "yaml_comment": "是否流式推理(长视频/实时流必须 True)",
        },
    )

    stream_buffer: bool = Field(
        default=False,
        description="流缓冲(防止丢帧)",
        json_schema_extra={
            "group":    "视频流",
            "examples": [False, True],
            "tips": [
                "True: 缓存所有帧, 慢但不丢",
                "False: 实时丢弃旧帧, 快但可能漏",
                "实时性要求高 → False; 离线分析 → True",
            ],
            "yaml_comment": "是否缓存所有帧(防丢帧但延迟↑)",
        },
    )

    # ============================================================
    # 输出控制
    # ============================================================

    save_txt: bool = Field(
        default=False,
        description="保存预测为 txt",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [False, True],
            "tips": [
                "True: 输出 YOLO 格式 txt (每张图一份)",
                "用于后续半自动标注 / 二次审查",
            ],
            "yaml_comment": "是否保存预测结果为 txt",
        },
    )

    save_conf: bool = Field(
        default=False,
        description="txt 里包含置信度",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [False, True],
            "tips": [
                "True: txt 行追加置信度 (cls x y w h conf)",
                "False: 标准 YOLO 格式 (cls x y w h)",
                "需要 save_txt=True 配套",
            ],
            "yaml_comment": "txt 中是否包含 conf",
        },
    )

    save_crop: bool = Field(
        default=False,
        description="裁剪检测框保存",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [False, True],
            "tips": [
                "True: 把每个检测框抠出来单独存图",
                "用于人脸库 / 物体库构建",
            ],
            "yaml_comment": "是否裁剪检测框另存",
        },
    )

    save_frames: bool = Field(
        default=False,
        description="保存视频每一帧",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [False, True],
            "tips": [
                "True: 视频推理时, 每一帧另存为图像",
                "False: 只输出标注视频",
                "调试 / 抽样审查用",
            ],
            "yaml_comment": "是否保存视频每一帧(仅 source 为视频时)",
        },
    )

    show: bool = Field(
        default=True,
        description="实时显示推理结果",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [False, True],
            "tips": [
                "True: 弹窗实时显示 (需要 GUI)",
                "服务器/Docker 环境必须 False",
                "调试时用, 生产环境关",
            ],
            "yaml_comment": "是否弹窗显示(服务器环境必须 False)",
        },
    )

    show_labels: bool = Field(
        default=True,
        description="可视化是否带类别标签",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [True, False],
            "tips": [
                "True: 检测框上方写类别名",
                "False: 只画框, 适合给非技术人员看",
            ],
            "yaml_comment": "可视化是否带类别标签",
        },
    )

    show_conf: bool = Field(
        default=True,
        description="可视化是否带置信度",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [True, False],
            "tips": [
                "True: 类别名后追加 (0.87)",
                "False: 只显示类别名",
            ],
            "yaml_comment": "可视化是否带置信度",
        },
    )

    show_boxes: bool = Field(
        default=True,
        description="可视化是否画框",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [True, False],
            "tips": [
                "True: 画矩形框 + 标签",
                "False: 只画标签, 不画框 (segment 任务有时用)",
            ],
            "yaml_comment": "可视化是否画框",
        },
    )

    line_width: Optional[int] = Field(
        default=None,
        ge=1,
        description="可视化线宽",
        json_schema_extra={
            "group":    "输出控制",
            "examples": [None, 2, 3, 5],
            "tips": [
                "None: 自动 (根据图像大小)",
                "int: 固定像素宽度",
                "大图建议 3-5, 小图 1-2",
            ],
            "yaml_comment": "可视化线宽(像素, None=自动)",
        },
    )

    # ============================================================
    # 任务特定
    # ============================================================

    retina_masks: bool = Field(
        default=False,
        description="高分辨率分割 mask",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [False, True],
            "tips": [
                "仅 segment 任务: True 输出原图分辨率 mask",
                "False: 输出推理分辨率 mask (节省内存)",
                "detect 任务忽略此字段",
            ],
            "yaml_comment": "分割任务: 是否输出原图分辨率 mask",
        },
    )

    visualize: bool = Field(
        default=False,
        description="保存特征图可视化",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [False, True],
            "tips": [
                "True: 保存模型各层的特征图",
                "调试 / 论文用",
                "会占大量磁盘, 生产环境关",
            ],
            "yaml_comment": "是否保存特征图(调试用)",
        },
    )

    embed: Optional[List[int]] = Field(
        default=None,
        description="提取指定层 embedding",
        json_schema_extra={
            "group":    "任务特定",
            "examples": [None, [10], [10, 20]],
            "tips": [
                "None: 不提取",
                "list[int]: 提取指定层索引的特征向量",
                "用于相似度检索 / 特征聚类",
            ],
            "yaml_comment": "提取指定层 embedding(None=不提取)",
        },
    )

    # ============================================================
    # 验证器
    # ============================================================

    @field_validator("task")
    @classmethod
    def _validate_task(cls, v: str) -> str:
        """task SSoT — 严格限定到 Task.all()"""
        if v not in Task.all():
            raise ValueError(
                f"task 必须是 {Task.all()} 之一, 当前值: {v!r}. "
                f"若想给本次实验取名, 请使用 experiment_name 字段."
            )
        return v

    @field_validator("source")
    @classmethod
    def _validate_source(cls, v: Any) -> Any:
        """source 类型标准化."""
        if v is None:
            return v
        if isinstance(v, (str, Path)):
            return str(v)
        if isinstance(v, int):
            # 摄像头索引整数: 转字符串保留 (跟 ultralytics 习惯一致)
            return str(v)
        raise TypeError(
            f"source 必须是 str / Path / int (摄像头索引) 或 None, "
            f"当前类型: {type(v).__name__}"
        )

    @model_validator(mode="after")
    def _cross_field_validation(self) -> "YOLOInferConfig":
        """跨字段验证 (全部 warn, 推理场景的'奇怪'都能跑)."""
        # save_conf 必须配套 save_txt
        if self.save_conf and not self.save_txt:
            warnings.warn(
                "save_conf=True 但 save_txt=False, save_conf 不会生效. "
                "若想保存置信度, save_txt 必须为 True.",
                UserWarning,
            )

        # stream_buffer 隐含 stream=True
        if self.stream_buffer and not self.stream:
            warnings.warn(
                "stream_buffer=True 但 stream=False, stream_buffer 不会生效. "
                "stream_buffer 仅在流式推理 (stream=True) 时有意义.",
                UserWarning,
            )

        # retina_masks 仅 segment 任务有效
        if self.retina_masks and self.task != Task.SEGMENT:
            warnings.warn(
                f"retina_masks=True 仅对 segment 任务有效, "
                f"当前 task={self.task!r}, 该字段将被忽略.",
                UserWarning,
            )

        return self