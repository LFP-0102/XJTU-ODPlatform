#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :loaders.py
# @Time      :2026/7/18 13:03:39
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import  annotations
import logging
from argparse import Namespace
from pathlib import Path
from typing import  Any, Dict, List, Mapping, Optional, Union
from od_platform.common.refs import resolve_ref
import yaml
logger = logging.getLogger(__name__)

def _drop_none(d: Mapping[str,Any]) -> Dict[str, Any]:
    return  {k:v for k, v in d.items() if v is not None}

# YAML LOADER
class YAMLLoader:
    """加载 YAML 配置文件 → dict.

    ★ 路径解析复用 common/refs.resolve_ref(D3)——"名字还是路径"的判断
    全平台只有那一份。本类只负责【从路径到 dict】。
    """

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        # 默认 None → 裸名字相对 cwd;测试传 tmp_path 重定向约定目录
        self.config_dir = Path(config_dir) if config_dir else None

    def load(self, filename: Union[str, Path]) -> Dict[str, Any]:
        # 拿到路径解析路径
        filepath = resolve_ref(
            str(filename),
            base_dir=self.config_dir or Path.cwd(),
            default_suffix=".yaml",
        )

        # 2.文件是否存在
        if not filepath.exists():
            raise FileNotFoundError(f"YAML 配置文件不存在: {filepath}"
                                    f"清先生成默认的配置模板")

        # 3. 读文件
        try:
            content = filepath.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8解码失败，尝试使用系统默认的编码：{filepath}")
            content = filepath.read_text()

        # 4. 是不是空文件
        if not content.strip():
            logger.debug(f"YAML 配置文件为空: {filepath}")
            return {}

        # 5. 解析YAML
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 格式错误：{filepath}, 原始错误为：{e}")

        # 6. 顶层的结构检查
        if data is None:
            return  {}

        if not isinstance(data, dict):
            raise ValueError(f"YAML 配置文件结构错误: 当前是：{type(data).__name__}: {filepath}")

        # 过滤None值
        return _drop_none(data)

# CLI LOADER
class CLILoader:
    DEFAULT_EXCLUDE: set[str] = {
        "help", "config", "cfg", "yaml_path", "debug", "version"
    }
    def __init__(self,exclude: Optional[List[str]] = None,mapping: Optional[Dict[str, str]] = None):
        self.exclude = self.DEFAULT_EXCLUDE | set(exclude or [])
        self.mapping = mapping or {}

    def load(self, args: Optional[Union[Namespace, Dict[str, Any]]] = None,
            filter_none: bool = True
            ):
        if args is None:
            return {}

        # 传字典
        if isinstance(args, Namespace):
            raw = vars(args)
        elif isinstance(args,dict):
            raw = args
        else:
            raise TypeError(f"args must be Namespace or dict, got: {type(args).__name__}")

        result: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in self.exclude or key.startswith("_"):
                continue
            if filter_none and value is None:
                continue
            mapped_key = self.mapping.get(key, key)
            result[mapped_key] = value

        return result

def load_all_sources(
        yaml_path: Optional[Union[str, Path]] = None,
        yaml_dir: Optional[Union[str, Path]] = None,
        cli_args: Optional[Union[Namespace,Dict[str, Any]]] = None,
        cli_exclude: Optional[Union[List[str]]] = None,
        cli_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    yaml_config: Dict[str,Any] = {}
    if yaml_path:
        loader = YAMLLoader(config_dir=yaml_dir)
        yaml_config = loader.load(yaml_path)

    cli_loader = CLILoader(exclude=cli_exclude, mapping=cli_mapping)
    cli_config = cli_loader.load(cli_args)

    return {"yaml": yaml_config, "cli": cli_config}