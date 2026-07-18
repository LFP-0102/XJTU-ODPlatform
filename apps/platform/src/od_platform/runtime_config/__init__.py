"""runtime_config — YOLO 训练 / 验证配置子系统.

公共 API:

  配置类:
      BaseConfig, YOLOTrainConfig, YOLOValConfig

  加载器(外部数据 → dict):
      YAMLLoader, CLILoader, load_all_sources

  合并器(三源合并 + 溯源):
      ConfigMerger, ConfigSource, ConfigMetadata

  生成器(Pydantic → YAML 模板):
      ConfigGenerator

  注册表(名字 → 配置类, generator 与 build_* 共读):
      CONFIG_REGISTRY

  一键 build:
      build_train_config, build_val_config

不公开:
  - _drop_none 等内部工具函数
  - 不在 __all__ 里的所有符号都是内部细节, 可能在小版本变化

典型用法 (在 D6 service 层):

    from od_platform.runtime_config import build_train_config

    config, merger = build_train_config(
        yaml_path = "train.yaml",
        cli_args  = args,         # argparse.Namespace
    )

    # 主线: 用 config 跑训练
    model.train(**config.to_ultralytics_kwargs())

    # 副线: 打来源报告 / 查单字段溯源
    print(merger.get_source_report())
    print(merger.get_metadata("lr0").chain_str())
"""
from __future__ import annotations

from argparse import Namespace
from pathlib  import Path
from typing   import Any, Dict, List, Mapping, Optional, Tuple, Union

# 配置类
from od_platform.runtime_config.base_config   import BaseConfig
from od_platform.runtime_config.train_config  import YOLOTrainConfig
from od_platform.runtime_config.val_config    import YOLOValConfig
from od_platform.runtime_config.infer_config  import YOLOInferConfig   # ★ infer 闭环

# 加载器
from od_platform.runtime_config.loaders import (
    YAMLLoader,
    CLILoader,
    load_all_sources,
)

# 合并器
from od_platform.runtime_config.merger import (
    ConfigMerger,
    ConfigSource,
    ConfigMetadata,
)

# 生成器
from od_platform.runtime_config.generator import ConfigGenerator
from od_platform.runtime_config.registry  import CONFIG_REGISTRY


# ============================================================
# 一键 build 便捷函数
# ============================================================

def build_train_config(
    yaml_path: Optional[Union[str, Path]] = "train.yaml",
    cli_args:  Optional[Union[Namespace, Dict[str, Any]]] = None,
    *,
    yaml_dir:      Optional[Union[str, Path]] = None,
    cli_exclude:   Optional[List[str]]        = None,
    cli_mapping:   Optional[Dict[str, str]]   = None,
    extra_sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    track_sources: bool = True,
    dry_run:       bool = False,
) -> Tuple[Optional[YOLOTrainConfig], ConfigMerger]:
    """一键完成 train 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证.

    Args:
        yaml_path:     YAML 文件名 / 路径. None 跳过 yaml 加载.
                      默认 "train.yaml" (从 yaml_dir 找).
        cli_args:      argparse.Namespace 或 dict. None 跳过 CLI 加载.
        yaml_dir:      YAML 目录, 默认 paths.RUNTIME_CONFIGS_DIR.
        cli_exclude:   CLI 额外排除字段 (并入 CLILoader.DEFAULT_EXCLUDE).
        cli_mapping:   CLI 参数名映射 (CLI 名 → Pydantic 字段名).
        extra_sources: 额外配置源 [(source_id, dict), ...],
                       插入到 YAML 和 CLI 之间. 例: 环境变量 / etcd.
                       想换顺序? 直接调 ConfigMerger.merge().
        track_sources: 是否追踪溯源 (默认 True).
        dry_run:       True = 只合并不验证, 返回 (None, merger).
                       False = 正常合并 + 验证, 返回 (config, merger).

    Returns:
        正常模式: (validated_config, merger)
        dry_run:  (None, merger)         # config=None, merger 内有完整溯源

    Raises:
        FileNotFoundError: YAML 文件不存在 (带修复指引)
        ValidationError:   Pydantic 字段验证失败 (已注入溯源链), dry_run=True 时不 raise
    """
    # 延迟 import 避免循环依赖 + 让 paths.py 改动不击穿这里
    from od_platform.common.paths import RUNTIME_CONFIGS_DIR

    # 1. 加载各源
    sources = load_all_sources(
        yaml_path   = yaml_path,
        yaml_dir    = yaml_dir or RUNTIME_CONFIGS_DIR,
        cli_args    = cli_args,
        cli_exclude = cli_exclude,
        cli_mapping = cli_mapping,
    )

    # 2. 拼 sources list, 顺序: YAML < extra_sources < CLI
    sources_list: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = []
    if sources["yaml"]:
        sources_list.append((ConfigSource.YAML, sources["yaml"]))
    if extra_sources:
        sources_list.extend(extra_sources)
    if sources["cli"]:
        sources_list.append((ConfigSource.CLI, sources["cli"]))

    # 3. 合并 (dry_run 走 preview, 否则走 merge)
    merger = ConfigMerger(track_sources=track_sources)
    if dry_run:
        merger.preview(YOLOTrainConfig, sources=sources_list)
        return None, merger
    config = merger.merge(YOLOTrainConfig, sources=sources_list)
    return config, merger


def build_val_config(
    yaml_path: Optional[Union[str, Path]] = "val.yaml",
    cli_args:  Optional[Union[Namespace, Dict[str, Any]]] = None,
    *,
    yaml_dir:      Optional[Union[str, Path]] = None,
    cli_exclude:   Optional[List[str]]        = None,
    cli_mapping:   Optional[Dict[str, str]]   = None,
    extra_sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    track_sources: bool = True,
    dry_run:       bool = False,
) -> Tuple[Optional[YOLOValConfig], ConfigMerger]:
    """一键完成 val 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证.

    参数语义和返回值结构与 build_train_config 完全一致, 仅类型不同.
    """
    from od_platform.common.paths import RUNTIME_CONFIGS_DIR

    sources = load_all_sources(
        yaml_path   = yaml_path,
        yaml_dir    = yaml_dir or RUNTIME_CONFIGS_DIR,
        cli_args    = cli_args,
        cli_exclude = cli_exclude,
        cli_mapping = cli_mapping,
    )

    sources_list: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = []
    if sources["yaml"]:
        sources_list.append((ConfigSource.YAML, sources["yaml"]))
    if extra_sources:
        sources_list.extend(extra_sources)
    if sources["cli"]:
        sources_list.append((ConfigSource.CLI, sources["cli"]))

    merger = ConfigMerger(track_sources=track_sources)
    if dry_run:
        merger.preview(YOLOValConfig, sources=sources_list)
        return None, merger
    config = merger.merge(YOLOValConfig, sources=sources_list)
    return config, merger


def build_infer_config(
    yaml_path: Optional[Union[str, Path]] = "infer.yaml",
    cli_args:  Optional[Union[Namespace, Dict[str, Any]]] = None,
    *,
    yaml_dir:      Optional[Union[str, Path]] = None,
    cli_exclude:   Optional[List[str]]        = None,
    cli_mapping:   Optional[Dict[str, str]]   = None,
    extra_sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    track_sources: bool = True,
    dry_run:       bool = False,
) -> Tuple[Optional[YOLOInferConfig], ConfigMerger]:
    """一键完成 infer 配置流水线: YAML + CLI (+ 自定义源) → 合并 → 验证.

    参数语义和返回值结构与 build_train_config 完全一致, 仅类型不同.

    典型用法 (D8 InferService):
        # source 通过 CLI 当场传, 不写进 yaml
        config, merger = build_infer_config(
            yaml_path="infer.yaml",
            cli_args={"source": "video.mp4"},
        )
        model.predict(**config.to_ultralytics_kwargs())
    """
    from od_platform.common.paths import RUNTIME_CONFIGS_DIR

    sources = load_all_sources(
        yaml_path   = yaml_path,
        yaml_dir    = yaml_dir or RUNTIME_CONFIGS_DIR,
        cli_args    = cli_args,
        cli_exclude = cli_exclude,
        cli_mapping = cli_mapping,
    )

    sources_list: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = []
    if sources["yaml"]:
        sources_list.append((ConfigSource.YAML, sources["yaml"]))
    if extra_sources:
        sources_list.extend(extra_sources)
    if sources["cli"]:
        sources_list.append((ConfigSource.CLI, sources["cli"]))

    merger = ConfigMerger(track_sources=track_sources)
    if dry_run:
        merger.preview(YOLOInferConfig, sources=sources_list)
        return None, merger
    config = merger.merge(YOLOInferConfig, sources=sources_list)
    return config, merger


# ============================================================
# 公开 API 清单 (★ 唯一真相)
# ============================================================

__all__ = [
    # 配置类
    "BaseConfig",
    "YOLOTrainConfig",
    "YOLOValConfig",
    "YOLOInferConfig",          # ★ infer 闭环

    # 加载器
    "YAMLLoader",
    "CLILoader",
    "load_all_sources",

    # 合并器
    "ConfigMerger",
    "ConfigSource",
    "ConfigMetadata",

    # 生成器
    "ConfigGenerator",

    # 注册表(名字 → 配置类的唯一真相)
    "CONFIG_REGISTRY",

    # 一键 build
    "build_train_config",
    "build_val_config",
    "build_infer_config",        # ★ infer 闭环
]