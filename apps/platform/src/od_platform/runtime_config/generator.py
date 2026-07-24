#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :generator.py
# @Time      :2026/7/18 14:11:04
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :配置文件生成器——支持嵌套 Pydantic 模型递归生成 YAML 模板
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConfigGenerator:
    def __init__(self, indent: int = 4):
        self.indent = indent

    def generate(self, config_class: Type[BaseModel],
                output_path: Union[str, Path], *,
                overwrite: bool = False,
                backup: bool = True,
                title: Optional[str] = None,
                overrides: Optional[dict] = None,
                ) -> bool:
        output_path = Path(output_path)

        if output_path.exists() and not overwrite:
            logger.info(f"配置文件已经存在, 跳过生成 {output_path}")
            return False

        if output_path.exists() and backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_name(f"{output_path.name}.bak.{stamp}")
            shutil.copy2(output_path, backup_path)
            logger.info(f"配置文件已存在，已备份为 {backup_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._generate_yaml(config_class, title, overrides)
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"配置文件已经生成 {output_path}")
        return True

    def _generate_yaml(self, config_class: Type[BaseModel], title: Optional[str] = None,
                       overrides: Optional[dict] = None) -> str:
        lines: List[str] = []
        lines.append("#" + "=" * 78)
        lines.append(f"# {title or config_class.__name__}")
        lines.append(f"# 自动生成的时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Author：Xiao Liang")
        lines.append("#" + "=" * 78)
        lines.append("")

        config = config_class(**(overrides or {}))

        groups = self._get_field_groups(config)
        for group_name, field_names in groups.items():
            lines.append("")
            lines.append("#" + "=" * 78)
            lines.append(f"# {group_name}")
            lines.append("#" + "=" * 78)
            lines.append("")
            for field_name in field_names:
                lines.extend(self._generate_field(config, field_name))
                lines.append("")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # 字段生成
    # ----------------------------------------------------------------

    def _generate_field(self, config: BaseModel, field_name: str, depth: int = 0) -> List[str]:
        """生成单个字段的 YAML 行.

        根据字段值类型分发:
          - BaseModel → 递归展开嵌套模型
          - dict      → YAML 映射 (缩进子项)
          - list      → YAML 列表 (缩进子项)
          - 其他      → 单行键值对
        """
        lines: List[str] = []
        metadata = self._get_field_metadata(config, field_name)
        indent = " " * (self.indent * depth)

        value = getattr(config, field_name)

        # ---- 嵌套 BaseModel ----
        if isinstance(value, BaseModel):
            yaml_comment = metadata.get("yaml_comment") or metadata.get("description")
            if yaml_comment:
                lines.append(f"{indent}# {yaml_comment}")
            tips = metadata.get("tips", [])
            if tips:
                for tip in tips:
                    lines.append(f"{indent}#   - {tip}")
            lines.append(f"{indent}{field_name}:")
            lines.extend(self._generate_nested_fields(value, depth + 1))
            return lines

        # ---- 注释 (顶层字段) ----
        if depth == 0:
            yaml_comment = metadata.get("yaml_comment") or metadata.get("description")
            if yaml_comment:
                lines.append(f"# {yaml_comment}")

            examples = metadata.get("examples", [])
            if examples:
                examples_str = ", ".join(str(e) for e in examples[:5])
                lines.append(f"# 示例: {examples_str}")

            tips = metadata.get("tips", [])
            if tips:
                lines.append("# 提示")
                for tip in tips:
                    lines.append(f"    # - {tip}")

        # ---- dict → YAML 映射 ----
        if isinstance(value, dict):
            if not value:
                lines.append(f"{indent}{field_name}: {{}}")
            else:
                lines.append(f"{indent}{field_name}:")
                for k, v in value.items():
                    formatted = self._format_value(v)
                    lines.append(f"{indent}{' ' * self.indent}{k}: {formatted}")
            return lines

        # ---- 标量值 (list/tuple 统一走 _format_value 行内渲染) ----
        yaml_value = self._format_value(value)
        lines.append(f"{indent}{field_name}: {yaml_value}")
        return lines

    def _generate_nested_fields(self, model: BaseModel, depth: int) -> List[str]:
        """递归展开嵌套模型的全部字段, 按组输出."""
        lines: List[str] = []
        indent = " " * (self.indent * depth)
        groups = self._get_field_groups(model)

        first_group = True
        for group_name, field_names in groups.items():
            if not first_group:
                lines.append("")
            first_group = False

            # 子组注释
            lines.append(f"{indent}# --- {group_name} ---")
            for field_name in field_names:
                lines.extend(self._generate_field(model, field_name, depth))
        return lines

    # ----------------------------------------------------------------
    # 元数据提取 (兼容 BaseConfig.get_field_groups/get_field_metadata)
    # ----------------------------------------------------------------

    @staticmethod
    def _get_field_groups(model: BaseModel) -> dict[str, list[str]]:
        """从 Pydantic 模型的 json_schema_extra 提取字段分组."""
        if hasattr(model, "get_field_groups"):
            return model.get_field_groups()
        groups: dict[str, list[str]] = {}
        for field_name, field_info in model.model_fields.items():
            extra = field_info.json_schema_extra or {}
            group = extra.get("group", "其他") if isinstance(extra, dict) else "其他"
            groups.setdefault(group, []).append(field_name)
        return groups

    @staticmethod
    def _get_field_metadata(model: BaseModel, field_name: str) -> dict[str, Any]:
        """从 Pydantic 模型的 json_schema_extra 提取字段元数据."""
        if hasattr(model, "get_field_metadata"):
            return model.get_field_metadata(field_name)

        field_info = model.model_fields[field_name]
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

    # ----------------------------------------------------------------
    # 值格式化
    # ----------------------------------------------------------------

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, str):
            if any(c in value for c in [":", "#", "[", "]", "{", "}"]):
                return f'"{value}"'
            return value

        if isinstance(value, (list, tuple)):
            if not value:
                return "[]"
            items = ", ".join(str(v) for v in value)
            return f"[{items}]"
        if isinstance(value, dict):
            return "{}"

        return str(value)
