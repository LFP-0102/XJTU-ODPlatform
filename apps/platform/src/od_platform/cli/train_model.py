#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : train_model.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/cli/train_model.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-19 13:35:40
# @Version    : v1.0.0
# @Description:CLI端入口
# @ChangeLog:
#   2026-07-19:35:40 | Matri | v1.0.0 | 初始化创建
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

from od_platform.common import paths, refs
from od_platform.common.logging_utils import get_logger
from od_platform.common.naming import run_stem
from od_platform.common.paths import RUNTIME_CONFIGS_DIR
from od_platform.common.run_context import RunContext
from od_platform.runtime_config import build_train_config

from od_platform.model_train.service import train_yolo


_CLI_SWITCHES = {'config', "no_archive"}

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odp-train",description="模型训练")
    parser.add_argument("--config",default="train", help="运行时配置文件")
    parser.add_argument("--no-archive", action="store_true", help="不归档运行时配置文件")

    parser.add_argument("--model",default="yolov8n.pt", help="模型文件")
    parser.add_argument("--data",default=None, help="数据集文件")
    parser.add_argument("--epochs",type=int, default=200, help="训练轮数")
    parser.add_argument("--batch",default=16, type=int, help="批量大小")
    parser.add_argument("-imgsz", type=int, default=640, help="图片大小")
    parser.add_argument("--device",default="0", help="运行设备")

    return parser


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str,Any]:
    return {k:v for k,v in vars(args).items() if v is not None and k not in _CLI_SWITCHES}


def main() ->int:
    args = _build_parser().parse_args()
    cli_overrides = _collect_cli_overrides(args)

    config_ref = args.config
    yaml_path = config_ref if ("/" in config_ref or "\\" in config_ref) else RUNTIME_CONFIGS_DIR / (config_ref if config_ref.endswith(".yaml") else f"{config_ref}.yaml")
    config, merger = build_train_config(yaml_path=str(yaml_path), cli_args=cli_overrides or None)

    with RunContext("training") as run:
        data_yaml = refs.resolve_dataset_yaml(config.data)

        model_ref = getattr(config, "model", None) or "yolov8n.pt"
        stem = run_stem(stage="train", run_id=run.run_id, dataset=data_yaml.stem, model=model_ref)

        logger = get_logger(paths.LOGGING_DIR, log_type="train", run_id=run.run_id,
                            dataset_name=data_yaml.stem,model_name=model_ref
                            )
        logger.info("启动训练")
        result = train_yolo(config=config, data_yaml=data_yaml, run=run, stem=stem,merger=merger,
                        archive=not args.no_archive
                        )
        if result.success:
            logger.info("训练成功")
        else:
            logger.error("训练失败")
    return 0 if result.success else 1

if __name__ == "__main__":
    sys.exit(main())

