#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : base.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统基础模型——BaseConfig
"""运行配置子系统基础模块.

只装 BaseConfig: 所有模式(训练 / 验证 / 推理)共享的字段.
专属字段去 train.py / val.py.

★ 注意: 这里【不】设任何项目路径字段——
    paths.py 才是项目路径的【唯一真相来源】(D2 立的 SSoT 纪律).
    service 层直接 from od_platform.common.paths import LOGGING_DIR, ... 即可.
"""
from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ============================================================
# BaseConfig: 所有模式共享的基础配置
# ============================================================

class BaseConfig(BaseModel):
    """所有模式(训练 / 验证 / 推理)共享的基础配置.

    字段分组:
        - 核心参数: model, data
        - 输入配置: batch, imgsz, workers, cache, rect
        - 设备配置: device, amp
        - 输出配置: project, name, exist_ok, save
        - 基础设置: verbose, seed, deterministic
    """

    # extra="forbid": 拒绝 YAML / CLI 里拼错的字段名(详见 2.4 例 4)
    model_config = ConfigDict(extra="forbid")

    # 框架级字段: 只在 D5 框架内部使用, to_ultralytics_kwargs() 会把它们剔除,
    # 不传给 ultralytics. 子类(如 YOLOValConfig)可在此基础上并入更多字段.
    FRAMEWORK_ONLY_FIELDS: ClassVar[set[str]] = {"verbose"}

    # 敏感字段 mask 时的替代字符串. 子类可重新赋值定制.
    SENSITIVE_MASK: ClassVar[str] = "***"

    model: Optional[str] = Field(
        default="yolov8n.pt",
        description="模型文件路径",
        json_schema_extra={
            "group": "核心参数",
            "examples": ["yolo11n.pt", "yolo11s.pt", "path/to/custom.pt"],
            "tips": [
                "支持 .pt 权重文件或 .yaml 配置文件",
                "官方模型: yolo11n/s/m/l/x.pt (n 最快, x 最准)",
                "训练时如未指定, 默认从头训练",
            ],
            "yaml_comment": "模型文件路径(.pt 或 .yaml)",
        },
    )

    data: Optional[str] = Field(
        default=None,
        description="数据集配置文件路径",
        json_schema_extra={
            "group": "核心参数",
            "examples": ["data.yaml", "rsod.yaml", "configs/datasets/safety.yaml"],
            "tips": [
                "YAML 文件定义数据集路径和类别信息",
                "必须包含: path, train, val, names",
                "如果只给数据集名(如 'rsod'), 框架会通过 paths.dataset_yaml_path() 解析",
            ],
            "yaml_comment": "数据集配置文件(D3 odp-transform 产出的那份 yaml)",
        },
    )

    # -------- 输入配置 --------

    batch: Union[int, float] = Field(
        default=16,
        description="批次大小",
        json_schema_extra={
            "group": "输入配置",
            "examples": [8, 16, 32, -1, 0.7],
            "tips": [
                "int: 固定批次大小(如 16)",
                "-1: 自动批次(使用约 60% 显存)",
                "0.0-1.0: 显存利用率(如 0.7 = 70% 显存)",
                "显存不足时建议: 8GB → 8-16, 16GB → 32-64",
            ],
            "yaml_comment": "批次大小(int=固定值, -1=自动, 0-1=显存利用率)",
        },
    )

    imgsz: int = Field(
        default=640,
        ge=32,
        description="输入图像尺寸",
        json_schema_extra={
            "group": "输入配置",
            "examples": [320, 480, 640, 800, 1280],
            "tips": [
                "推荐使用 32 的倍数(640, 672, 704...)",
                "更大尺寸 = 更高精度 + 更多显存 + 更慢速度",
                "工业检测推荐: 640-800",
                "小目标检测推荐: 1280+",
            ],
            "yaml_comment": "输入图像尺寸(像素)",
        },
    )

    workers: int = Field(
        default=8,
        ge=0,
        description="数据加载工作线程数",
        json_schema_extra={
            "group": "输入配置",
            "examples": [0, 4, 8, 16],
            "tips": [
                "多线程加速数据加载",
                "推荐: CPU 核心数的 1-2 倍",
                "Windows 可能需要设为 0",
                "过高会导致内存占用过多",
            ],
            "yaml_comment": "数据加载线程数",
        },
    )

    cache: Union[bool, str] = Field(
        default=False,
        description="数据缓存策略",
        json_schema_extra={
            "group": "输入配置",
            "examples": [False, True, "ram", "disk"],
            "tips": [
                "False: 不缓存(默认)",
                "True/'ram': 缓存到内存(快但占内存)",
                "'disk': 缓存到磁盘(节省内存但慢)",
                "小数据集推荐 ram, 大数据集推荐 disk",
            ],
            "yaml_comment": "数据缓存(False/True/ram/disk)",
        },
    )

    rect: bool = Field(
        default=False,
        description="矩形训练(最小填充)",
        json_schema_extra={
            "group": "输入配置",
            "examples": [False, True],
            "tips": [
                "True: 减少填充, 提升训练速度",
                "推理时自动启用",
                "可能轻微影响精度",
            ],
            "yaml_comment": "是否启用矩形训练",
        },
    )

    # -------- 设备配置 --------

    device: Optional[Union[int, str, List[Union[int, str]]]] = Field(
        default=None,
        description="训练设备",
        json_schema_extra={
            "group": "设备配置",
            "examples": [0, "0", "cpu", "mps", [0, 1], "0,1"],
            "tips": [
                "None: 自动选择(优先 GPU)",
                "int: GPU 编号(0, 1, 2...)",
                "'cpu': 使用 CPU",
                "'mps': Mac M1/M2 GPU",
                "[0,1] 或 '0,1': 多 GPU 训练",
                "-1: 选择最空闲的 GPU",
            ],
            "yaml_comment": "设备选择(0/cpu/mps/[0,1]/0,1)",
        },
    )

    amp: bool = Field(
        default=True,
        description="自动混合精度",
        json_schema_extra={
            "group": "设备配置",
            "examples": [True, False],
            "tips": [
                "True: 使用 FP16 混合精度(快 + 省显存)",
                "False: 使用 FP32 全精度",
                "现代 GPU (20 系+) 强烈推荐开启",
                "CPU 训练时自动禁用",
            ],
            "yaml_comment": "是否启用 AMP(推荐开启)",
        },
    )

    # -------- 输出配置 --------

    project: Optional[str] = Field(
        default=None,
        description="输出项目目录",
        json_schema_extra={
            "group": "输出配置",
            "examples": ["runs/detect", "experiments/safety"],
            "tips": [
                "训练输出的根目录",
                "默认: 由 service 层根据 task 自动拼装(runs/<task>/)",
            ],
            "yaml_comment": "项目输出目录(None=框架自动)",
        },
    )

    name: Optional[str] = Field(
        default=None,
        description="实验名称",
        json_schema_extra={
            "group": "输出配置",
            "examples": ["exp", "safety_v1", "baseline"],
            "tips": [
                "每次运行的子目录名",
                "默认: ultralytics 自动生成 exp, exp2, exp3...",
            ],
            "yaml_comment": "实验名称",
        },
    )

    exist_ok: bool = Field(
        default=False,
        description="允许覆盖已有实验",
        json_schema_extra={
            "group": "输出配置",
            "examples": [False, True],
            "tips": [
                "False: 自动创建新目录(exp2, exp3...)",
                "True: 覆盖已有目录",
                "生产环境建议 False",
            ],
            "yaml_comment": "是否覆盖已有实验目录",
        },
    )

    save: bool = Field(
        default=True,
        description="保存训练检查点",
        json_schema_extra={
            "group": "输出配置",
            "examples": [True, False],
            "tips": [
                "True: 保存 last.pt 和 best.pt",
                "False: 不保存检查点(节省磁盘)",
            ],
            "yaml_comment": "是否保存检查点",
        },
    )

    # -------- 基础设置 --------

    verbose: bool = Field(
        default=False,
        description="详细输出模式",
        json_schema_extra={
            "group": "基础设置",
            "examples": [True, False],
            "tips": [
                "True: 显示详细日志",
                "False: 只显示关键信息",
            ],
            "yaml_comment": "是否显示详细日志",
        },
    )

    seed: int = Field(
        default=1210,
        ge=0,
        description="随机种子",
        json_schema_extra={
            "group": "基础设置",
            "examples": [0, 42, 2026],
            "tips": [
                "控制随机性, 确保可复现",
                "相同种子 = 相同结果",
                "建议每个实验使用不同种子",
            ],
            "yaml_comment": "随机种子(可复现性)",
        },
    )

    deterministic: bool = Field(
        default=True,
        description="确定性算法",
        json_schema_extra={
            "group": "基础设置",
            "examples": [True, False],
            "tips": [
                "True: 强制使用确定性算法(完全可复现)",
                "False: 允许非确定性算法(可能更快)",
                "实验对比时必须设为 True",
            ],
            "yaml_comment": "是否强制确定性算法",
        },
    )

    # ============================================================
    # 验证器
    # ============================================================

    @field_validator("imgsz")
    @classmethod
    def _validate_imgsz(cls, v: int) -> int:
        """验证图像尺寸(建议 32 的倍数, 但不强制)"""
        if v % 32 != 0:
            warnings.warn(
                f"imgsz={v} 不是 32 的倍数. "
                f"虽然可以运行, 但建议使用 32 的倍数(如 640, 672, 704)以获得最佳性能.",
                UserWarning,
            )
        return v

    @field_validator("batch", mode="before")
    @classmethod
    def _validate_batch(cls, v: Any) -> Union[int, float]:
        """验证批次大小格式(mode="before": 在类型转换前拿到原始值, 才能拦住 bool)"""
        if isinstance(v, bool):
            raise TypeError("batch 不能是 bool 类型")
        if isinstance(v, int):
            if v == -1:
                return v
            if v <= 0:
                raise ValueError("batch 为 int 时必须为 -1(自动) 或 >= 1(固定值)")
            return v
        v_float = float(v)
        if not (0.0 < v_float <= 1.0):
            raise ValueError(
                f"batch 为 float 时必须在 (0, 1] 范围内, 表示显存利用率. "
                f"当前值: {v_float}"
            )
        return v_float

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: Any) -> Any:
        """标准化设备参数"""
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, (int, list)):
            return v
        raise TypeError(
            f"device 必须是 None, str, int 或 list 类型, "
            f"当前类型: {type(v).__name__}"
        )

    @field_validator("cache")
    @classmethod
    def _validate_cache(cls, v: Union[bool, str]) -> Union[bool, str]:
        """验证缓存参数"""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in ("ram", "disk"):
                raise ValueError(
                    f"cache 为 str 时必须是 'ram' 或 'disk', 当前值: {v}"
                )
            return v_lower
        raise TypeError(
            f"cache 必须是 bool 或 str 类型, 当前类型: {type(v).__name__}"
        )

    @model_validator(mode="after")
    def _cross_field_validation(self) -> "BaseConfig":
        """跨字段验证"""
        if isinstance(self.batch, int) and self.batch > 0:
            if self.workers > self.batch * 2:
                warnings.warn(
                    f"workers={self.workers} 远大于 batch={self.batch}, "
                    f"可能造成资源浪费. 建议 workers <= batch * 2",
                    UserWarning,
                )
        return self

    # ============================================================
    # 工具方法
    # ============================================================

    def to_ultralytics_kwargs(self) -> Dict[str, Any]:
        """转换为 Ultralytics 训练/验证参数字典.

        - 自动过滤 None 值
        - 剔除 FRAMEWORK_ONLY_FIELDS 中的框架级参数(不传递给 Ultralytics)
        子类(YOLOValConfig)只需扩展 FRAMEWORK_ONLY_FIELDS, 无需 override 本方法.
        """
        d = self.model_dump(exclude_none=True)
        for field_name in self.FRAMEWORK_ONLY_FIELDS:
            d.pop(field_name, None)
        return d

    def get_field_groups(self) -> Dict[str, List[str]]:
        """获取字段分组信息. {group_name: [field_names]}"""
        groups: Dict[str, List[str]] = {}
        for field_name, field_info in self.__class__.model_fields.items():
            extra = field_info.json_schema_extra or {}
            group = extra.get("group", "其他") if isinstance(extra, dict) else "其他"
            groups.setdefault(group, []).append(field_name)
        return groups

    def get_field_metadata(self, field_name: str) -> Dict[str, Any]:
        """获取字段的完整元数据(供 generator 写注释用)"""
        if field_name not in self.__class__.model_fields:
            raise ValueError(f"字段 '{field_name}' 不存在")

        field_info = self.__class__.model_fields[field_name]
        metadata: Dict[str, Any] = {
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

    # ============================================================
    # 实验复现接口 (to_audit_snapshot / from_audit_snapshot)
    # ============================================================

    def to_audit_snapshot(self) -> Dict[str, Any]:
        """返回可序列化的完整快照, 用于实验复现.

        Returns:
            {
                "config_class":  类名 (如 "YOLOTrainConfig"),
                "config_module": 模块路径,
                "frozen_at":     ISO 时间戳,
                "values":        所有字段值,
            }
        """
        return {
            "config_class":  self.__class__.__name__,
            "config_module": self.__class__.__module__,
            "frozen_at":     datetime.now().isoformat(timespec="seconds"),
            "values":        self.model_dump(),
        }

    @classmethod
    def from_audit_snapshot(cls, snapshot: Dict[str, Any]) -> "BaseConfig":
        """从快照恢复配置实例 (走完整 Pydantic 验证).

        Raises:
            ValueError:        快照类名与 cls 不匹配
            ValidationError:   快照值非法 (字段定义可能已演进)
        """
        expected = cls.__name__
        got = snapshot.get("config_class")
        if got != expected:
            raise ValueError(
                f"快照来自配置类 {got!r}, 不能恢复到 {expected!r}. "
                f"请用 {got}.from_audit_snapshot(snapshot)."
            )
        return cls(**snapshot.get("values", {}))

    # ============================================================
    # 敏感字段 mask
    # ============================================================

    def mask_sensitive_dump(self) -> Dict[str, Any]:
        """model_dump() 但 sensitive 字段值替换成 SENSITIVE_MASK.

        进日志 / 报告前调用, 避免敏感数据外泄.
        进数据库存原值时用 model_dump() (不 mask).
        """
        dump = self.model_dump()
        for field_name, field_info in self.__class__.model_fields.items():
            extra = field_info.json_schema_extra or {}
            if not isinstance(extra, dict):
                continue
            if not extra.get("sensitive"):
                continue
            if dump.get(field_name) is None:
                continue                              # None 不 mask, 避免误导
            dump[field_name] = self.SENSITIVE_MASK
        return dump

    @classmethod
    def sensitive_field_names(cls) -> set[str]:
        """返回标了 sensitive=True 的字段名集合 (供 merger 自动 mask 用)."""
        result: set[str] = set()
        for name, field_info in cls.model_fields.items():
            extra = field_info.json_schema_extra or {}
            if isinstance(extra, dict) and extra.get("sensitive"):
                result.add(name)
        return result