from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel

from od_platform.common.paths import runtime_config_path
from od_platform.runtime_config.registry import CONFIG_REGISTRY

logger = logging.getLogger(__name__)

class ConfigGenerator:
    def __init__(self, indent: int = 4):
        self.indent = indent
    def generate(
        self,
        config_class: Type[BaseModel],
        output_path: Union[str, Path],
        *,
        overwrite: bool = False,
        backup:    bool = True,
        title:     Optional[str] = None,
    ) -> bool:
        output_path = Path(output_path)

        # 第一道防线: 不覆盖
        if output_path.exists() and not overwrite:
            logger.info(f"配置文件已存在, 跳过生成: {output_path}")
            return False

        # 第二道防线: 覆盖前备份
        if output_path.exists() and backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_name(
                f"{output_path.name}.bak.{stamp}"
            )
            shutil.copy2(output_path, backup_path)        # copy2 保留 metadata
            logger.info(f"覆盖前已备份原配置: {backup_path}")

        # 创建父目录 (不存在的话)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成 YAML 内容
        content = self._generate_yaml(config_class, title)

        # 写入文件
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"配置文件已生成: {output_path}")
        return True

    def _generate_yaml(
        self,
        config_class: Type[BaseModel],
        title: Optional[str] = None,
    ) -> str:
        """生成完整 YAML 内容(头部 + 字段分组 + 尾部 FAQ)"""
        lines: List[str] = []

        # 文件头部
        lines.append("#" + "=" * 78)
        lines.append(f"# {title or config_class.__name__}")
        lines.append(f"# 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("#" + "=" * 78)
        lines.append("")

        # 创建默认实例(用于反射 group 信息和默认值)
        config = config_class()

        # 按分组生成字段
        groups = config.get_field_groups()
        for group_name, field_names in groups.items():
            lines.append("")
            lines.append("#" + "-" * 78)
            lines.append(f"# {group_name}")
            lines.append("#" + "-" * 78)
            lines.append("")
            for field_name in field_names:
                lines.extend(self._generate_field(config, field_name))
                lines.append("")        # 字段间空行

        return "\n".join(lines)
    def _generate_field(
        self,
        config: BaseModel,
        field_name: str,
    ) -> List[str]:
        """生成单个字段的 YAML 内容(含注释 + 值)"""
        lines: List[str] = []

        # 取字段元数据
        metadata = config.get_field_metadata(field_name)

        # 主注释 (yaml_comment, fallback 到 description)
        yaml_comment = metadata.get("yaml_comment") or metadata.get("description")
        if yaml_comment:
            lines.append(f"# {yaml_comment}")

        # 示例 (最多 5 个, 太多反而花眼)
        examples = metadata.get("examples", [])
        if examples:
            examples_str = ", ".join(str(e) for e in examples[:5])
            lines.append(f"# 示例: {examples_str}")

        # 提示 (一行一条)
        tips = metadata.get("tips", [])
        if tips:
            lines.append("# 提示:")
            for tip in tips:
                lines.append(f"#   - {tip}")

        # 字段值(默认值)
        value = getattr(config, field_name)
        yaml_value = self._format_value(value)
        lines.append(f"{field_name}: {yaml_value}")

        return lines

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "null"

        if isinstance(value, bool):       # ★ 必须在 int / 数字判断之前
            return "true" if value else "false"

        if isinstance(value, str):
            # 含 YAML 特殊字符时加引号
            if any(c in value for c in [":", "#", "[", "]", "{", "}"]):
                return f'"{value}"'
            return value

        if isinstance(value, (list, tuple)):
            if not value:
                return "[]"
            items = ", ".join(str(v) for v in value)
            return f"[{items}]"

        if isinstance(value, dict):
            # 当前没有顶层 dict 字段, 简化处理
            return "{}"

        # 数字 / 其他类型
        return str(value)
