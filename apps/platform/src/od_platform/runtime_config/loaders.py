"""配置加载器: 从不同来源加载配置(不负责验证和合并)."""
from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

from od_platform.common.refs import resolve_ref

logger = logging.getLogger(__name__)

def _drop_none(d: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}

# YAMLLoader
class YAMLLoader:
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        self.config_dir = Path(config_dir) if config_dir else None

    def load(self, filename: Union[str, Path]) -> Dict[str, Any]:
        # 1. 解析路径
        filepath = resolve_ref(
            str(filename),
            base_dir=self.config_dir or Path.cwd(),
            default_suffix=".yaml",
        )

        # 2. 文件不存在 → fail-fast + 修复指引(★ 撞墙③)
        if not filepath.exists():
            raise FileNotFoundError(
                f"YAML 配置文件不存在: {filepath}请先生成默认配置模板"
            )

        # 3. 读文件(默认 UTF-8, 失败 fallback)
        try:
            content = filepath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 解码失败, 尝试系统默认编码: {filepath}")
            content = filepath.read_text()

        # 4. 空文件 → 返回 {}, 等同 "全用默认值"
        if not content.strip():
            logger.debug(f"YAML 文件为空: {filepath}")
            return {}

        # 5. 解析 YAML — 失败 fail-fast, 保留 exception chain
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 格式错误:{filepath}，原始错误为:{e}")

        # 6. 顶层结构检查
        if data is None:
            return {}     # YAML 显式 null, 等同空文件

        if not isinstance(data, dict):
            raise ValueError(
                f"YAML 顶层必须是字典, 当前是 {type(data).__name__}: {filepath}"
            )

        # 7. 过滤 None 值(保留 False / 0 / '')
        return _drop_none(data)

class CLILoader:
    # 默认排除的控制字段(不该进配置 dict)
    DEFAULT_EXCLUDE: set[str] = {
        "help",
        "config", "cfg", "yaml_path",       # yaml 路径是 loader 的输入
        "debug",
        "version",
    }

    def __init__(
        self,
        exclude: Optional[List[str]] = None,
        mapping: Optional[Dict[str, str]] = None,
    ):
        self.exclude = self.DEFAULT_EXCLUDE | set(exclude or [])
        self.mapping = mapping or {}

    def load(
        self,
        args: Optional[Union[Namespace, Dict[str, Any]]] = None,
        filter_none: bool = True,
    ) -> Dict[str, Any]:
        
        if args is None:
            return {}

        # 转字典
        if isinstance(args, Namespace):
            raw = vars(args)
        elif isinstance(args, dict):
            raw = args
        else:
            raise TypeError(
                f"args 必须是 argparse.Namespace 或 dict, "
                f"当前是 {type(args).__name__}"
            )

        # 过滤 + 映射
        result: Dict[str, Any] = {}
        for key, value in raw.items():
            # 排除控制字段 + 私有字段
            if key in self.exclude or key.startswith("_"):
                continue

            # 过滤 None
            if filter_none and value is None:
                continue

            # 参数名映射
            mapped_key = self.mapping.get(key, key)
            result[mapped_key] = value

        return result


# ============================================================
# 便捷函数: 一次性加载所有源
# ============================================================

def load_all_sources(
    yaml_path: Optional[Union[str, Path]] = None,
    yaml_dir: Optional[Union[str, Path]] = None,
    cli_args: Optional[Union[Namespace, Dict[str, Any]]] = None,
    cli_exclude: Optional[List[str]] = None,
    cli_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    yaml_config: Dict[str, Any] = {}
    if yaml_path:
        loader = YAMLLoader(config_dir=yaml_dir)
        yaml_config = loader.load(yaml_path)

    cli_loader = CLILoader(exclude=cli_exclude, mapping=cli_mapping)
    cli_config = cli_loader.load(cli_args)

    return {"yaml": yaml_config, "cli": cli_config}