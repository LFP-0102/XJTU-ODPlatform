#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :pipeline_config.py
# @Time      :2026/7/23 14:00:00
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :推理管线配置(帧源 + 美化) —— Pydantic 模型, 供 odp-gen-config 生成 YAML 模板
"""推理管线配置模型: InferPipelineConfig.

管理 infer.yaml 管不到的另一半:
  - frame_source : 摄像头 / RTSP 帧源参数
  - visualization: 检测框美化、类别名翻译、颜色与字体

跟 D5 的 YOLOInferConfig 互不重叠, 各管各的.
"""
from __future__ import annotations

from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 嵌套子模型
# ============================================================

class CameraSettings(BaseModel):
    """摄像头帧源参数."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(
        default=1280,
        description="请求分辨率宽",
        json_schema_extra={
            "group": "摄像头",
            "examples": [640, 1280, 1920],
            "tips": [
                "实际分辨率以摄像头协商结果为准",
                "USB 摄像头常见: 640x480 / 1280x720",
                "工业相机可能支持更高分辨率",
            ],
            "yaml_comment": "请求分辨率宽(实际以摄像头协商结果为准)",
        },
    )

    height: int = Field(
        default=720,
        description="请求分辨率高",
        json_schema_extra={
            "group": "摄像头",
            "examples": [480, 720, 1080],
            "tips": [
                "配合 width 组成目标分辨率",
                "width:height 比例建议与摄像头原生比例一致",
            ],
            "yaml_comment": "请求分辨率高",
        },
    )

    fps: int = Field(
        default=30,
        ge=1,
        description="请求帧率",
        json_schema_extra={
            "group": "摄像头",
            "examples": [15, 30, 60],
            "tips": [
                "高帧率需要摄像头硬件支持 + 高带宽 USB 接口",
                "MJPG 编码通常能达到更高帧率",
                "实际帧率受光照 / 曝光时间影响",
            ],
            "yaml_comment": "请求帧率",
        },
    )

    backend: str = Field(
        default="auto",
        description="摄像头后端",
        json_schema_extra={
            "group": "摄像头",
            "examples": ["auto", "msmf", "dshow", "v4l2"],
            "tips": [
                "auto: 系统自选 (推荐)",
                "msmf: Windows 高帧率后端",
                "dshow: Windows 兼容后端 (DirectShow)",
                "v4l2: Linux 后端",
            ],
            "yaml_comment": "后端: auto(系统自选) | msmf(Win高帧率) | dshow(Win兼容) | v4l2(Linux)",
        },
    )

    codec: str = Field(
        default="MJPG",
        description="FOURCC 编码格式",
        json_schema_extra={
            "group": "摄像头",
            "examples": ["MJPG", "YUYV", "H264", "MP4V"],
            "tips": [
                "MJPG: 高帧率首选, 几乎所有摄像头支持",
                "YUYV: 未压缩, 画质好但帧率低",
                "H264: 硬件编码, 低带宽",
                "高帧率务必 MJPG",
            ],
            "yaml_comment": "FOURCC: MJPG | YUYV | H264 | MP4V (高帧率务必 MJPG)",
        },
    )

    camera_id: Optional[int] = Field(
        default=None,
        description="设备号",
        json_schema_extra={
            "group": "摄像头",
            "examples": [None, 0, 1],
            "tips": [
                "None: 由 CLI --source 决定 (推荐, 设备号通常当场指定)",
                "写死设备号: 仅当确定设备不会变时使用",
                "CLI --source 传的摄像头号会覆盖此处",
            ],
            "yaml_comment": "设备号通常由 --source 决定(会覆盖此处); 只想在配置里写死时才取消注释",
        },
    )


class StyleSettings(BaseModel):
    """美化绘制细节参数——数值按画面尺寸自适应缩放, 以下为 720p 基准值."""

    model_config = ConfigDict(extra="forbid")

    ref_dim: int = Field(
        default=720,
        ge=1,
        description="自适应参考短边",
        json_schema_extra={
            "group": "绘制细节",
            "examples": [480, 720, 1080],
            "tips": [
                "画面短边 = ref_dim 时使用下列基准值",
                "画面更大则等比放大, 更小则等比缩小",
                "一般设为常见输入分辨率短边即可",
            ],
            "yaml_comment": "自适应参考短边: 画面短边=ref_dim 时用基准值, 更大则等比放大",
        },
    )

    base_font_size: int = Field(
        default=26,
        ge=1,
        description="标签字号基准",
        json_schema_extra={
            "group": "绘制细节",
            "examples": [18, 26, 32],
            "tips": [
                "720p 画面下的字号 (像素)",
                "实际字号 = base * (实际短边 / ref_dim)",
            ],
            "yaml_comment": "标签字号基准(像素)",
        },
    )

    base_line_width: int = Field(
        default=2,
        ge=1,
        description="检测框线宽基准",
        json_schema_extra={
            "group": "绘制细节",
            "examples": [1, 2, 3],
            "tips": [
                "720p 画面下的线宽 (像素)",
                "粗线更醒目但可能遮挡目标",
            ],
            "yaml_comment": "检测框线宽基准",
        },
    )

    base_padding_x: int = Field(
        default=10,
        ge=0,
        description="标签左右内边距基准",
        json_schema_extra={
            "group": "绘制细节",
            "examples": [6, 10, 14],
            "tips": [
                "标签背景框左右留白",
                "文字越长需要越大",
            ],
            "yaml_comment": "标签左右内边距基准",
        },
    )

    base_padding_y: int = Field(
        default=10,
        ge=0,
        description="标签上下内边距基准",
        json_schema_extra={
            "group": "绘制细节",
            "examples": [6, 10, 14],
            "tips": [
                "标签背景框上下留白",
            ],
            "yaml_comment": "标签上下内边距基准",
        },
    )

    base_radius: int = Field(
        default=8,
        ge=0,
        description="标签圆角半径基准",
        json_schema_extra={
            "group": "绘制细节",
            "examples": [0, 4, 8, 12],
            "tips": [
                "0: 直角",
                "8-12: 柔和圆角",
            ],
            "yaml_comment": "标签圆角半径基准",
        },
    )

    text_color: list = Field(
        default=[0, 0, 0],
        description="标签文字颜色",
        json_schema_extra={
            "group": "绘制细节",
            "examples": [[0, 0, 0], [255, 255, 255]],
            "tips": [
                "格式: [B, G, R] (注意是 BGR 不是 RGB)",
                "[0, 0, 0] = 黑色, [255, 255, 255] = 白色",
            ],
            "yaml_comment": "标签文字颜色 [B, G, R]",
        },
    )


class VisualizationSettings(BaseModel):
    """检测框美化 + 类别名翻译 + 颜色映射."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="是否启用美化绘制",
        json_schema_extra={
            "group": "美化",
            "examples": [True, False],
            "tips": [
                "True: 使用美化模块绘制 (中文标签 + 自定义颜色 + 圆角框)",
                "False: 退回 YOLO 原生 result.plot() (更快但样式朴素)",
            ],
            "yaml_comment": "false → 用 YOLO 原生绘制",
        },
    )

    use_label_mapping: bool = Field(
        default=True,
        description="是否启用标签映射",
        json_schema_extra={
            "group": "美化",
            "examples": [True, False],
            "tips": [
                "True: 用 label_mapping 翻译类名为中文",
                "False: 即使 label_mapping 有内容也显示英文原名",
            ],
            "yaml_comment": "false → 即使下面写了 label_mapping 也一律显示英文原名",
        },
    )

    label_mapping: dict = Field(
        default={
            "hat": "安全帽",
            "person": "人员",
            "head": "未佩戴安全帽",
            "ordinary_clothes": "普通衣物",
            "reflective_vest": "反光衣",
        },
        description="英文类名 → 中文映射",
        json_schema_extra={
            "group": "美化",
            "tips": [
                "没列出的类别保持模型原始英文名",
                "这几项是随数据集变化的, 按你的类别改",
            ],
            "yaml_comment": "英文类名 → 中文(没列出的类别保持模型原始英文名)",
        },
    )

    color_mapping: dict = Field(
        default={
            "hat": [0, 255, 255],
            "person": [255, 0, 0],
            "head": [0, 255, 0],
            "ordinary_clothes": [0, 0, 255],
            "reflective_vest": [255, 255, 0],
        },
        description="类别 → 颜色映射",
        json_schema_extra={
            "group": "美化",
            "tips": [
                "格式: 类名: [B, G, R] (注意是 BGR 不是 RGB)",
                "没列出的类别用 default_color",
                "示例: [0, 255, 0] = 绿色, [0, 0, 255] = 红色",
            ],
            "yaml_comment": "类别 → 颜色 [B, G, R](注意是 BGR 不是 RGB; 没列出的用 default_color)",
        },
    )

    default_color: list = Field(
        default=[0, 255, 0],
        description="默认检测框颜色",
        json_schema_extra={
            "group": "美化",
            "examples": [[0, 255, 0], [255, 0, 0], [0, 0, 255]],
            "tips": [
                "不在 color_mapping 里的类别统一用这个颜色",
                "[0, 255, 0] = 绿色 (最常用)",
            ],
            "yaml_comment": "不在 color_mapping 里的类别统一用这个颜色",
        },
    )

    font_path: Optional[str] = Field(
        default=None,
        description="中文字体文件路径",
        json_schema_extra={
            "group": "美化",
            "examples": [
                None,
                "C:/Windows/Fonts/msyh.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/System/Library/Fonts/PingFang.ttc",
            ],
            "tips": [
                "null: 使用模块内置字体 (可正常显示中文)",
                "Windows: C:/Windows/Fonts/msyh.ttc",
                "Linux: /usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "macOS: /System/Library/Fonts/PingFang.ttc",
            ],
            "yaml_comment": "中文字体路径(null=内置字体, 可正常显示中文)",
        },
    )

    style: StyleSettings = Field(
        default_factory=StyleSettings,
        description="绘制细节参数",
        json_schema_extra={
            "group": "美化",
            "tips": [
                "这里只填 base_* / ref_dim / text_color 这些",
                "不要直接写 font_size / line_width / padding_x —— 它们由 base_* 自动算出",
            ],
            "yaml_comment": "绘制细节 style(数值都会按画面尺寸自适应缩放, 下面是 720p 基准值)",
        },
    )


class FrameSourceSettings(BaseModel):
    """帧源配置——仅摄像头块, 视频/图片/目录源用默认即可."""

    model_config = ConfigDict(extra="forbid")

    camera: CameraSettings = Field(
        default_factory=CameraSettings,
        description="摄像头参数",
        json_schema_extra={
            "group": "帧源",
            "tips": [
                "camera 块【只有】当输入源是摄像头号(--source 0/1/2 ...)时才生效",
                "视频文件 / 图片 / 目录 / RTSP-URL 源会整块忽略它",
            ],
            "yaml_comment": "camera 块【只有】当输入源是摄像头号(--source 0/1/2 ...)时才生效; 视频文件/图片/目录/RTSP-URL 源会整块忽略它",
        },
    )


# ============================================================
# 顶层配置模型
# ============================================================

class InferPipelineConfig(BaseModel):
    """推理管线配置: 帧源 + 美化.

    管的是 infer.yaml 管不到的另一半:
      - frame_source  : 摄像头 / RTSP 怎么抓帧
      - visualization : 检测框怎么画、类别名怎么翻译、用什么颜色和字体

    两份 yaml 互不重叠, 各管各的.
    """

    model_config = ConfigDict(extra="forbid")

    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = set()

    frame_source: FrameSourceSettings = Field(
        default_factory=FrameSourceSettings,
        description="帧源配置",
        json_schema_extra={
            "group": "帧源",
            "tips": [
                "camera 块只在摄像头源时生效",
                "视频 / 图片 / 目录源直接忽略整块",
            ],
            "yaml_comment": "帧源 frame_source — 摄像头 / RTSP 怎么抓帧(视频文件、图片、目录源用默认即可)",
        },
    )

    visualization: VisualizationSettings = Field(
        default_factory=VisualizationSettings,
        description="美化配置",
        json_schema_extra={
            "group": "美化",
            "tips": [
                "enabled: false → 直接退回 YOLO 原生 result.plot()",
                "想换一份就用 odp-infer --pipeline-yaml <名字或路径>",
            ],
            "yaml_comment": "美化 visualization — 检测框怎么画、类别名怎么翻译、用什么颜色和字体",
        },
    )

    # ============================================================
    # 工具方法 (对齐 BaseConfig 接口, 供 ConfigGenerator 调用)
    # ============================================================

    def get_field_groups(self) -> dict[str, list[str]]:
        """获取字段分组信息. {group_name: [field_names]}"""
        groups: dict[str, list[str]] = {}
        for field_name, field_info in self.model_fields.items():
            extra = field_info.json_schema_extra or {}
            group = extra.get("group", "其他") if isinstance(extra, dict) else "其他"
            groups.setdefault(group, []).append(field_name)
        return groups

    def get_field_metadata(self, field_name: str) -> dict[str, Any]:
        """获取字段的完整元数据(供 generator 写注释用)"""
        if field_name not in self.model_fields:
            raise ValueError(f"字段 '{field_name}' 不存在")

        field_info = self.model_fields[field_name]
        metadata: dict[str, Any] = {
            "description":  field_info.description,
            "default":      field_info.default,
            "examples":     [],
            "tips":         [],
            "yaml_comment": field_info.description,
            "group":        "其他",
            "sensitive":    False,
        }
        if isinstance(field_info.json_schema_extra, dict):
            metadata.update(field_info.json_schema_extra)
        return metadata


__all__ = [
    "InferPipelineConfig",
    "FrameSourceSettings",
    "VisualizationSettings",
    "StyleSettings",
    "CameraSettings",
]
