#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :generator.py
# @Time      :2026/7/18 14:11:04
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :配置文件生成器
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
                output_path: Union[str, Path], * ,
                overwrite: bool = False,
                backup: bool = True,
                title: Optional[str] = None,
                ) -> bool:
        output_path = Path(output_path)

        # 第一个检查：不要覆盖
        if output_path.exists() and not overwrite:
            logger.info(f"配置文件已经存在, 跳过生成 {output_path}")
            return False

        # 覆盖前做好备份
        if output_path.exists() and backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_name(f"{output_path.name}.bak.{stamp}")
            shutil.copy2(output_path, backup_path)
            logger.info(f"配置文件已存在，已备份为 {backup_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._generate_yaml(config_class, title)
        output_path.write_text(content, encoding='utf-8')
        logger.info(f"配置文件已经生成 {output_path}")
        return True

    def _generate_yaml(self, config_class: Type[BaseModel], title: Optional[str] = None) -> str:
        lines: List[str] = []
        lines.append("#" + "=" * 78)
        lines.append(f"# {title or config_class.__name__}")
        lines.append(f"# 自动生成的时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"# Author：Xiao Liang")
        lines.append("#" + "=" * 78)
        lines.append("")

        config = config_class()

        groups = config.get_field_groups()
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

    def _generate_field(self, config:BaseModel, field_name: str) -> List[str]:
        lines: List[str] = []
        metadata = config.get_field_metadata(field_name)
        yaml_comment = metadata.get("yaml_comment") or metadata.get("description")
        if yaml_comment:
            lines.append(f"# {yaml_comment}")

        examples = metadata.get("examples", [])
        if examples:
            examples_str = ", ".join(str(e) for e in examples[:5])
            lines.append(f"# 示例: {examples_str}")

        tips = metadata.get('tips', [])
        if tips:
            lines.append("# 提示")
            for tip in tips:
                lines.append(f"    # - {tip}")

        # 字段值
        value = getattr(config, field_name)
        yaml_value = self._format_value(value)
        lines.append(f"{field_name}: {yaml_value}")
        return lines

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return 'true' if value else "false"

        if isinstance(value, str):
            if any(c in value for c in [":", "#", "[", "]", "{", "}"]):
                return  f'"{value}"'
        if isinstance(value, (list, tuple)):
            if not value:
                return '[]'
            items = ", ".join(str(v) for v in value)
            return f"[{items}]"
        if isinstance(value, dict):
            return "{}"

        return str(value)



