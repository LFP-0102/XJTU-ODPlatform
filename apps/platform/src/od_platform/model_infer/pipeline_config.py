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
from datetime import datetime
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

    def build_camera_config(self, camera_id: int | None = None):
        """构造 CameraConfig(帧源的相机块).

        camera_id 来自推理输入源(如 --source 0), 是用户当场选的设备号, 优先级最高——
        它会覆盖 infer_pipeline.yaml camera 块里可能写死的 camera_id; 其余字段
        (width/height/fps/backend/codec)取自 yaml.

        返回值:
          - CameraConfig: 有设备号, 或 yaml 写了 camera 块;
          - None:         既没给设备号, yaml 也没 camera 块 → 调用方按普通字符串源处理.
        """
        from od_platform.frame_source import CameraConfig
        raw = dict(self.camera_raw)
        if camera_id is not None:
            raw["camera_id"] = camera_id          # 源里的设备号优先于 yaml
        if not raw:
            return None
        try:
            return CameraConfig(**raw)
        except Exception as e:
            logger.warning(f"camera 配置无效, 回退默认相机参数: {e}")
            # yaml 里某字段非法时, 至少保住设备号, 其余走 CameraConfig 默认
            return CameraConfig(camera_id=camera_id) if camera_id is not None else None

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


# 默认 pipeline 配置的文件名(裸名, 落在 paths.RUNTIME_CONFIGS_DIR 下, 与 train.yaml/infer.yaml 同目录)
DEFAULT_PIPELINE_YAML_STEM = "infer_pipeline"


def resolve_pipeline_yaml_path(yaml_path: str | Path | None) -> Path:
    """把 --pipeline-yaml 的取值解析成实际路径 —— 与 train.yaml/infer.yaml 同一套规则.

    - None            → paths.RUNTIME_CONFIGS_DIR/infer_pipeline.yaml (app 私有默认位置);
    - 裸名 "xxx"      → paths.RUNTIME_CONFIGS_DIR/xxx.yaml (没写 .yaml 自动补);
    - 带目录 / 绝对路径 → 原样使用.

    ★ 老实现的坑: 默认写死成相对路径 Path("configs/runtime/infer_pipeline.yaml"),
      是相对"当前工作目录"而非 app 私有配置目录, 于是几乎必然找不到 → 每次都告警走内存默认.
    """
    ref = DEFAULT_PIPELINE_YAML_STEM if yaml_path is None else str(yaml_path)
    return resolve_config_yaml(ref)


# 自动生成的默认配置模板. 每个键都对应 PipelineConfig / CameraConfig / DrawStyle 里真实读取的字段,
# 不塞摆设字段(CameraConfig / DrawStyle 都是 extra=forbid, 写错字段会当场报错).
_PIPELINE_YAML_TEMPLATE = """\
#==============================================================================
# infer_pipeline.yaml — ODPlatform 推理子系统"帧源 + 美化"配置
# 自动生成于: __GENERATED_AT__
#
# 管的是 infer.yaml 管不到的另一半:
#   · frame_source  : 摄像头 / RTSP 怎么抓帧(视频文件、图片、目录源用默认即可)
#   · visualization : 检测框怎么画、类别名怎么翻译、用什么颜色和字体
# 两份 yaml 互不重叠, 各管各的. 想换一份就用 `odp-infer --pipeline-yaml <名字或路径>`.
#==============================================================================

#==============================================================================
# 帧源 frame_source
#   camera 块【只有】当输入源是摄像头号(--source 0/1/2 ...)时才生效;
#   视频文件 / 图片 / 目录 / RTSP-URL 源会整块忽略它.
#==============================================================================
frame_source:
  camera:
    width:   1280        # 请求分辨率宽(实际以摄像头协商结果为准)
    height:  720         # 请求分辨率高
    fps:     30          # 请求帧率
    backend: auto        # 后端: auto(系统自选) | msmf(Win高帧率) | dshow(Win兼容) | v4l2(Linux)
    codec:   MJPG        # FOURCC: MJPG | YUYV | H264 | MP4V  (高帧率务必 MJPG)
    # camera_id: 0       # 设备号通常由 --source 决定(会覆盖此处); 只想在配置里写死时才取消注释

#==============================================================================
# 美化 visualization
#   enabled: false → 直接退回 YOLO 原生 result.plot()(更快但样式朴素)
#==============================================================================
visualization:
  enabled:           true    # false → 用 YOLO 原生绘制
  use_label_mapping: true    # false → 即使下面写了 label_mapping 也一律显示英文原名

  # ---- 英文类名 → 中文(没列出的类别保持模型原始英文名)----
  # 这几项是随数据集变化的, 下面只是示例, 按你的类别改
  label_mapping:
    person:  人
    car:     汽车
    bicycle: 自行车

  # ---- 类别 → 颜色 [B, G, R](注意是 BGR 不是 RGB; 没列出的用 default_color)----
  color_mapping:
    person: [0, 255, 0]      # 绿
    car:    [0, 0, 255]      # 红

  default_color: [0, 255, 0] # 不在 color_mapping 里的类别统一用这个颜色

  # ---- 中文字体 ----
  # null → 用模块内置字体(可正常显示中文); 想换就填字体文件绝对路径:
  #   Windows: C:/Windows/Fonts/msyh.ttc
  #   Linux:   /usr/share/fonts/truetype/wqy/wqy-microhei.ttc
  #   macOS:   /System/Library/Fonts/PingFang.ttc
  font_path: null

  # ---- 绘制细节 style(可选; 数值都会按画面尺寸自适应缩放, 下面是"720p 基准值")----
  # 提示: 这里只填 base_* / ref_dim / text_color / font_path 这些;
  #       不要直接写 font_size / line_width / padding_x —— 它们由 base_* 自动算出.
  style:
    ref_dim:         720     # 自适应参考短边: 画面短边=ref_dim 时用下列基准值, 更大则等比放大
    base_font_size:  26      # 标签字号基准(像素)
    base_line_width: 2       # 检测框线宽基准
    base_padding_x:  10      # 标签左右内边距基准
    base_padding_y:  10      # 标签上下内边距基准
    base_radius:     8       # 标签圆角半径基准
    text_color:      [0, 0, 0]   # 标签文字颜色 [B, G, R]
"""


def write_default_pipeline_yaml(path: str | Path, *, overwrite: bool = False) -> bool:
    """把带注释的默认 pipeline 配置写到 path.

    Returns:
        True  = 实际写了新文件;
        False = 文件已存在且 overwrite=False, 跳过(不覆盖用户已有配置).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _PIPELINE_YAML_TEMPLATE.replace(
        "__GENERATED_AT__", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    path.write_text(content, encoding="utf-8")
    return True


def load_pipeline_config(
    yaml_path: str | Path | None = None,
    *,
    auto_generate: bool = True,
) -> PipelineConfig:
    """读 infer_pipeline.yaml(帧源 + 美化). 永不抛 —— 出问题就退回内存默认.

    Args:
        yaml_path:     None / 裸名 / 路径, 解析规则见 resolve_pipeline_yaml_path.
        auto_generate: 文件不存在时是否自动生成一份带注释的默认配置(默认 True).
                       生成后立即读它 —— 用户下次就有真实文件可编辑, 不必手写.

    与老实现的区别:
      1. 路径解析对齐 train/infer(默认落在 app 私有 RUNTIME_CONFIGS_DIR, 不再相对 CWD);
      2. 缺失时"先解析再判断", 并且自动生成默认文件, 而不是每次只告警走内存默认.
    """
    resolved = resolve_pipeline_yaml_path(yaml_path)

    if not resolved.exists():
        if auto_generate:
            try:
                write_default_pipeline_yaml(resolved)
                logger.info(f"未找到 pipeline 配置, 已自动生成默认: {resolved}")
            except OSError as e:
                logger.warning(f"pipeline 配置不存在且自动生成失败, 用内存默认: {e}")
                return PipelineConfig()
        else:
            logger.warning(f"pipeline 配置不存在(未开启自动生成), 用内存默认: {resolved}")
            return PipelineConfig()

    try:
        with resolved.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"pipeline 配置解析失败, 用内存默认: {e}")
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

    logger.info(f"pipeline 配置加载: {resolved}, "
                f"美化={pc.viz_enabled}, 标签映射={len(pc.label_mapping)}, "
                f"颜色映射={len(pc.color_mapping)}")
    return pc