#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :gen_config.py
# @Time      :2026/7/18 14:45:47
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
import argparse
import logging

from od_platform.runtime_config.generator import ConfigGenerator
from od_platform.common.paths import runtime_config_path
from od_platform.runtime_config.registry import CONFIG_REGISTRY
from pathlib import Path
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="gen_config",
    temp_log=False
)
def main():
    parser = argparse.ArgumentParser(
        prog="odp-gen-config",
        description="生成运行时配置文件",
        epilog="例如：odp-gen-config train"
    )
    parser.add_argument("name", choices=list(CONFIG_REGISTRY), help="配置文件名称 train / val / infer")
    parser.add_argument("-o","--output",type=Path, default=None, help="输出路径")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖输出路径")
    parser.add_argument('--no-backup', action="store_true", help="是否不备份输出路径")
    parser.add_argument("--mlflow", action="store_true", help="预设 mlflow_enabled: true (训练配置专用)")
    args = parser.parse_args()

    config_class, title = CONFIG_REGISTRY[args.name]
    output_path = args.output or runtime_config_path(args.name)

    overrides = None
    if getattr(args, "mlflow", False):
        overrides = {"mlflow_enabled": True}

    gen = ConfigGenerator()
    success = gen.generate(
        config_class,
        output_path,
        overwrite=args.overwrite,
        backup=not args.no_backup,
        title=title,
        overrides=overrides,
    )

    if success:
        logger.info(f"配置文件已经生成")
    else:
        logger.info(f"你可以使用--overwrite参数覆盖输出路径, 默认不覆盖")


