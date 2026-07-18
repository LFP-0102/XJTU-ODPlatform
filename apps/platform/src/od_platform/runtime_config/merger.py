#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : merger.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——配置合并器 (sources list + 链表式溯源)
"""配置合并器: 按优先级合并多个配置源 + 配置溯源追踪.

★ 四个核心:
   1. sources list 接口:  任意源数量, source_id 接受枚举或字符串(扩展点)
   2. 链表式溯源:         ConfigMetadata.overridden_from 指向被覆盖的上一个
   3. 错误信息增强:       Pydantic ValidationError 加 "[来源: ...]" 后缀
   4. 三个产物:           get_source_report (人) / to_audit_log (机器) /
                          ValidationError 增强 (错误)

★ 跟 阶段 4 的 Loader 衔接:
   Loader 产出独立的 dict, Merger 把它们组装进 sources list +
   Pydantic 模型默认值 合并 + 验证 → Pydantic 实例
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError


# ============================================================
# 配置来源枚举 (内置 3 档, 自定义源走字符串)
# ============================================================

class ConfigSource(str, Enum):
    """配置来源 (内置 3 档). 自定义源直接用字符串, 不必扩 enum."""
    DEFAULT = "DEFAULT"
    YAML    = "YAML"
    CLI     = "CLI"


# ============================================================
# 配置元数据: 链表节点
# ============================================================

@dataclass
class ConfigMetadata:
    """单个字段的溯源信息.

    Attributes:
        key:             字段名
        value:           当前值
        source:          配置来源 (ConfigSource 枚举 或 自定义字符串)
        timestamp:       记录时间 (每次覆盖更新)
        overridden_from: ★ 指向被覆盖的上一个 ConfigMetadata, 形成链表
    """
    key:             str
    value:           Any
    source:          Union[ConfigSource, str]
    timestamp:       datetime
    overridden_from: Optional["ConfigMetadata"] = None

    @property
    def source_label(self) -> str:
        """统一拿 source 的字符串展示 (枚举走 .value, 字符串原样)."""
        if isinstance(self.source, ConfigSource):
            return self.source.value
        return self.source

    def chain(self) -> List["ConfigMetadata"]:
        """顺着 overridden_from 遍历, 返回从当前到根的历史链.

        例: 链表为 CLI ← YAML ← DEFAULT, chain() 返回 [CLI, YAML, DEFAULT]
        """
        result = [self]
        current = self.overridden_from
        while current:
            result.append(current)
            current = current.overridden_from
        return result

    def chain_str(self) -> str:
        """格式化覆盖链: '50(CLI) ← 200(YAML) ← 100(DEFAULT)'.

        '←' 从左到右读: 当前 ← 上一个 ← 更早, 符合"从结果倒推原因"的阅读习惯.
        """
        parts = [f"{m.value}({m.source_label})" for m in self.chain()]
        return " ← ".join(parts)


# ============================================================
# 配置合并器
# ============================================================

T = TypeVar("T", bound=BaseModel)


class ConfigMerger:
    """sources list 接口的三源 (及更多) 合并 + 链表式溯源.

    优先级由 sources list 顺序决定 (后面的覆盖前面的). DEFAULT 自动注入到最前.

    使用示例 (内置 yaml/cli):
        merger = ConfigMerger()
        config = merger.merge(
            YOLOTrainConfig,
            sources=[
                (ConfigSource.YAML, yaml_dict),
                (ConfigSource.CLI,  cli_dict),
            ],
        )

    使用示例 (自定义 ENV 源, 不扩 enum):
        config = merger.merge(
            YOLOTrainConfig,
            sources=[
                (ConfigSource.YAML, yaml_dict),
                ("ENV",             env_dict),       # ★ 字符串作为 source 标识
                (ConfigSource.CLI,  cli_dict),
            ],
        )

    使用示例 (dry-run 预览):
        merged_dict = merger.preview(YOLOTrainConfig, sources=[...])
        print(merger.get_source_report())     # 用户先看再决定

    三个产物:
        get_source_report() / get_conflict_report() — 给人看的字符串
        to_audit_log() — 给 ELK / 数据库的结构化 dict
        (BaseConfig.to_audit_snapshot() — 字段值快照, 在 base.py)
    """

    def __init__(self, track_sources: bool = True):
        """
        Args:
            track_sources: 是否追踪配置来源 (关掉只合并不溯源, 稍快)
        """
        self.track_sources = track_sources
        self._metadata: Dict[str, ConfigMetadata] = {}
        self._overridden_keys: List[str] = []
        self._last_config_class: Optional[Type[BaseModel]] = None    # 给 mask / audit 用

    # ============================================================
    # 公开接口
    # ============================================================

    def merge(
        self,
        config_class: Type[T],
        *,
        sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    ) -> T:
        """合并多个配置源 + Pydantic 验证, 返回实例.

        Args:
            config_class: Pydantic 配置类 (YOLOTrainConfig 等)
            sources:      [(source_id, dict), ...] 顺序 = 优先级 (低 → 高).
                          DEFAULT 自动注入到最前, 不需要手动传.

        Returns:
            验证后的 Pydantic 配置实例

        Raises:
            ValidationError: 字段值非法 (错误信息已增强, 含来源链)
        """
        merged = self._do_merge(config_class, sources)

        try:
            return config_class(**merged)
        except ValidationError as e:
            if self.track_sources:
                self._enhance_validation_error(e)
            raise

    def preview(
        self,
        config_class: Type[BaseModel],
        *,
        sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """Dry-run: 跑合并 + 维护溯源, 但不实例化配置类 (跳验证).

        用法:
            merged = merger.preview(YOLOTrainConfig, sources=[...])
            print(merger.get_source_report())     # 用户先看再决定

        Returns:
            合并后的原始 dict (未经验证, 可能含非法值).
        """
        return self._do_merge(config_class, sources)

    def get_metadata(self, key: str) -> Optional[ConfigMetadata]:
        """获取某字段的最新 ConfigMetadata (可顺 .overridden_from 拉链)."""
        return self._metadata.get(key)

    # ============================================================
    # 报告接口 (给人看)
    # ============================================================

    def get_source_report(self) -> str:
        """配置来源报告——按 source 分组列出最终生效的字段值.

        sensitive 字段自动 mask 成 SENSITIVE_MASK (跟 BaseConfig 联动).
        """
        if not self.track_sources:
            return "配置溯源未启用"

        lines = ["=" * 70, "配置来源报告".center(70), "=" * 70]

        # 按 source_label 分桶 (兼容枚举 + 字符串)
        by_source: Dict[str, List[str]] = {}
        for key, meta in self._metadata.items():
            by_source.setdefault(meta.source_label, []).append(key)

        # 内置 source 在前 (CLI/YAML/DEFAULT 优先级序), 自定义 source 字母序在后
        builtin_order = [s.value for s in
                         [ConfigSource.CLI, ConfigSource.YAML, ConfigSource.DEFAULT]]
        ordered = [s for s in builtin_order if s in by_source]
        custom  = sorted(s for s in by_source if s not in builtin_order)
        all_labels = ordered + custom

        for label in all_labels:
            keys = sorted(by_source[label])
            lines.append(f"\n{label} ({len(keys)} 项)")
            lines.append("-" * 70)
            if not keys:
                lines.append("  (无)")
            else:
                for key in keys:
                    value = self._display_value(key, self._metadata[key].value)
                    lines.append(f"  {key} = {value}")
        return "\n".join(lines)

    def get_conflict_report(self) -> str:
        """配置覆盖报告——列出被高优先级覆盖的字段.

        sensitive 字段值 (新旧两处) 都自动 mask.
        """
        if not self.track_sources:
            return "配置溯源未启用"

        overridden = list(dict.fromkeys(self._overridden_keys))   # 去重保序

        lines = ["=" * 70, "配置覆盖报告".center(70), "=" * 70]
        lines.append(f"\n共 {len(overridden)} 项配置被覆盖\n")

        if not overridden:
            lines.append("  (无)")
            return "\n".join(lines)

        for key in sorted(overridden):
            meta = self._metadata.get(key)
            if not meta:
                continue
            chain = meta.chain()
            if len(chain) <= 1:
                continue
            newest, previous = chain[0], chain[1]
            new_val = self._display_value(key, newest.value)
            old_val = self._display_value(key, previous.value)
            lines.append(
                f"  {key}: {old_val} ({previous.source_label}) "
                f"→ {new_val} ({newest.source_label})"
            )
        return "\n".join(lines)

    def to_audit_log(self) -> Dict[str, Any]:
        """结构化审计日志 (一行 JSON, 给 ELK / 数据库用).

        只装"哪些字段从哪儿来"的元信息, 不装字段值 (避免敏感数据进 ELK).
        字段值快照走 BaseConfig.to_audit_snapshot().

        Returns:
            {
                "merger_completed_at":  "2026-05-20T14:32:15",
                "track_sources":        true,
                "fields_count_total":   75,
                "fields_by_source":     {"CLI": [...], "YAML": [...], "DEFAULT": [...]},
                "overridden_count":     3,
                "overridden_fields":    ["batch", "epochs", "lr0"],
            }
        """
        if not self.track_sources:
            return {
                "merger_completed_at": datetime.now().isoformat(timespec="seconds"),
                "track_sources":       False,
            }

        # 按 source_label 分桶, 字段名排序保证幂等
        by_source: Dict[str, List[str]] = {}
        for key, meta in self._metadata.items():
            by_source.setdefault(meta.source_label, []).append(key)
        by_source = {k: sorted(v) for k, v in by_source.items()}

        overridden = sorted(set(self._overridden_keys))

        return {
            "merger_completed_at": datetime.now().isoformat(timespec="seconds"),
            "track_sources":       True,
            "fields_count_total":  len(self._metadata),
            "fields_by_source":    by_source,
            "overridden_count":    len(overridden),
            "overridden_fields":   overridden,
        }

    # ============================================================
    # 内部方法
    # ============================================================

    def _do_merge(
        self,
        config_class: Type[BaseModel],
        sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]],
    ) -> Dict[str, Any]:
        """合并 + 维护溯源, 返回 dict, 不实例化."""
        self._metadata.clear()
        self._overridden_keys.clear()
        self._last_config_class = config_class

        # DEFAULT 源永远自动注入到最前
        defaults = self._extract_defaults(config_class)
        all_sources: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = [
            (ConfigSource.DEFAULT, defaults),
        ]
        all_sources.extend(sources or [])

        merged: Dict[str, Any] = {}
        for source, cfg in all_sources:
            self._apply_source(merged, dict(cfg or {}), source)

        return merged

    def _apply_source(
        self,
        merged: Dict[str, Any],
        config: Mapping[str, Any],
        source: Union[ConfigSource, str],
    ) -> None:
        """应用一个配置源到合并结果. 同时维护溯源链表."""
        for key, value in config.items():
            # None 等同于 "没给", 不覆盖 (跟 Loader 的 _drop_none 同源)
            if value is None:
                continue

            # 记录覆盖事件
            if key in merged and self.track_sources:
                self._overridden_keys.append(key)

            # 更新值
            merged[key] = value

            # 维护溯源链表: 新节点的 overridden_from 指向上一个节点
            if self.track_sources:
                prev_meta = self._metadata.get(key)
                self._metadata[key] = ConfigMetadata(
                    key=key,
                    value=value,
                    source=source,
                    timestamp=datetime.now(),
                    overridden_from=prev_meta,
                )

    @staticmethod
    def _extract_defaults(config_class: Type[BaseModel]) -> Dict[str, Any]:
        """从 Pydantic 模型反射默认值 (SSoT: Field(default=...) 是唯一真相)."""
        defaults = {}
        for name, field in config_class.model_fields.items():
            if field.default is not None:        # None 默认值不进合并 (让 Pydantic 自己处理)
                defaults[name] = field.default
        return defaults

    def _enhance_validation_error(self, error: ValidationError) -> None:
        """把溯源链塞进 ValidationError 的 msg 后缀."""
        for err in error.errors():
            loc = err.get("loc", ())
            if not loc:
                continue
            key = loc[0]
            if not isinstance(key, str):
                continue       # 嵌套模型 loc 可能是 int, 跳过
            meta = self._metadata.get(key)
            if not meta:
                continue
            msg = err.get("msg", "")
            err["msg"] = f"{msg} [来源: {meta.chain_str()}]"

    # ============================================================
    # 敏感字段 mask (跟 BaseConfig 的 sensitive 元数据联动)
    # ============================================================

    def _is_sensitive(self, key: str) -> bool:
        """判断字段是否标了 sensitive=True."""
        if self._last_config_class is None:
            return False
        if not hasattr(self._last_config_class, "sensitive_field_names"):
            return False
        return key in self._last_config_class.sensitive_field_names()

    def _display_value(self, key: str, value: Any) -> Any:
        """获取展示值 (sensitive 字段自动 mask 成 SENSITIVE_MASK)."""
        if value is None:
            return value
        if self._is_sensitive(key):
            return getattr(self._last_config_class, "SENSITIVE_MASK", "***")
        return value