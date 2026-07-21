#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : pipline_config.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_infer/pipline_config.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-21 09:39:58
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-21:39:58 | Matri | v1.0.0 | 初始化创建
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : pipeline_config.py
# @Project   : ODPlatform
# @Function  : 读取 infer_pipeline.yaml(帧源+美化) — D5 管不到的那半边配置
"""帧源捕获 + 美化 的配置读取 helper.

★ 核心纪律: 不重新发明校验. 这个 helper 只做"yaml → 子字典 → 喂给现成 pydantic 模型":
  - camera 块  → frame_source.CameraConfig(**...)   (extra=forbid, 自带校验)
  - style 块   → 透传给 visualization.DrawStyle.from_image_size(**...) (拿到帧尺寸后才构造)
  - 颜色 list  → tuple (BGR), 因为美化模块吃 tuple

跟 D5 的关系: D5 的 infer.yaml 管 YOLO predict 参数(build_infer_config 读),
这份 infer_pipeline.yaml 管帧源+美化, 两份互不干涉. service 阶段 1 各读各的再捏一起.

文件缺失不算错误 —— 用默认值(美化开启、无中文映射、摄像头走 CameraConfig 默认),
打一条 warning 即可. 基本版只要有模型 + 源就能跑, pipeline yaml 是锦上添花.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from od_platform.common.refs import resolve_config_yaml
import yaml

logger = logging.getLogger(__name__)


def _to_bgr_tuple(value: Any) -> tuple[int, int, int]:
    """yaml 里颜色是 list [B,G,R], 美化模块吃 tuple, 转一下."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"颜色必须是 3 元素的 [B, G, R], 收到: {value!r}")


@dataclass
class PipelineConfig:
    """帧源 + 美化 配置的解析结果(纯数据, 不含行为)."""

    # ---- 帧源 ----
    camera_raw: dict[str, Any] = field(default_factory=dict)   # 原始 camera 子字典, 延迟构造 CameraConfig

    # ---- 美化 ----
    viz_enabled: bool = True
    use_label_mapping: bool = True
    label_mapping: dict[str, str] = field(default_factory=dict)
    color_mapping: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    default_color: tuple[int, int, int] = (0, 255, 0)
    font_path: str | None = None
    style_overrides: dict[str, Any] = field(default_factory=dict)   # 透传给 DrawStyle.from_image_size

    def build_camera_config(self):
        """延迟构造 CameraConfig —— 不是摄像头源时 create_frame_source 会忽略."""
        from od_platform.frame_source import CameraConfig
        if not self.camera_raw:
            return None
        try:
            return CameraConfig(**self.camera_raw)
        except Exception as e:
            logger.warning(f"camera 配置无效, 走默认: {e}")
            return None

    def to_audit(self) -> dict[str, Any]:
        """给 odp_audit.json 用的纯字典快照."""
        return {
            "viz_enabled": self.viz_enabled,
            "use_label_mapping": self.use_label_mapping,
            "label_mapping_n": len(self.label_mapping),
            "color_mapping_n": len(self.color_mapping),
            "default_color": list(self.default_color),
            "font_path": self.font_path,
            "camera": dict(self.camera_raw),
            "style_overrides_n": len(self.style_overrides),
        }


def load_pipeline_config(yaml_path: str | Path | None) -> PipelineConfig:
    """读 infer_pipeline.yaml. None / 文件不存在 → 空配置 + warning, 不抛."""
    if yaml_path is None:
        # 走默认路径: configs/runtime/infer_pipeline.yaml
        yaml_path = Path("configs/runtime/infer_pipeline.yaml")
    else:
        yaml_path = Path(yaml_path)

    if not yaml_path.exists():
        logger.warning(f"pipeline yaml 不存在, 用默认配置: {yaml_path}")
        return PipelineConfig()

    yaml_path = resolve_config_yaml(yaml_path)
    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"pipeline yaml 解析失败, 用默认配置: {e}")
        return PipelineConfig()

    pc = PipelineConfig()

    # ---- 帧源 ----
    fs = raw.get("frame_source", {}) or {}
    pc.camera_raw = fs.get("camera", {}) or {}

    # ---- 美化 ----
    vz = raw.get("visualization", {}) or {}
    pc.viz_enabled = bool(vz.get("enabled", True))
    pc.use_label_mapping = bool(vz.get("use_label_mapping", True))
    pc.label_mapping = dict(vz.get("label_mapping", {}) or {})

    raw_colors = vz.get("color_mapping", {}) or {}
    pc.color_mapping = {k: _to_bgr_tuple(v) for k, v in raw_colors.items()}

    if "default_color" in vz:
        pc.default_color = _to_bgr_tuple(vz["default_color"])

    pc.font_path = vz.get("font_path")
    pc.style_overrides = dict(vz.get("style", {}) or {})

    logger.info(f"pipeline 配置加载: {yaml_path}, "
                f"美化={pc.viz_enabled}, 标签映射={len(pc.label_mapping)}, "
                f"颜色映射={len(pc.color_mapping)}")
    return pc