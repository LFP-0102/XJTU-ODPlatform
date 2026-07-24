#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : evaluate_model.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/cli/evaluate_model.py
# @Project    : XJTU-ODPlatfrom
# @Author     : ODPlatform Team
# @Function   : 模型评估 CLI 入口(单模型 / 多模型对比)
"""odp-eval: 模型评估命令行入口.

职责(与 odp-train / odp-infer 同构):
  1. argparse -> cli_args dict, build_val_config 建好 val 配置
  2. RunContext("evaluation", sub_dir="single"/"compare") 定 run_id / run_dir
    runs/evaluation/single/<run_id>/ 或 runs/evaluation/compare/<run_id>/
  3. get_logger 装 handler(console + file), 用 run_id + 模型名预命名日志
  4. --model 单模型 -> evaluate_model; --models 多模型 -> compare_models
  5. 退出码: 0 成功, 1 失败(CI 友好)

用法示例:
  # 单模型评估
  odp-eval --model best.pt --data helmet_detection_v1 --split val

  # 多模型对比
  odp-eval --models yolo11n.pt yolo11s.pt best.pt --data helmet_detection_v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

from od_platform.common import paths, refs
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext
from od_platform.runtime_config import build_val_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odp-eval",
        description="模型评估(单模型评估 / 多模型对比)",
    )
    # 模型: --model(单个) 与 --models(多个) 二选一
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", type=str, default=None,
                     help="单个模型引用(名 / 路径), 如 best.pt")
    grp.add_argument("--models", type=str, nargs="+", default=None,
                     help="多个模型引用(对比评估), 如 yolo11n.pt yolo11s.pt")

    # 数据集
    p.add_argument("--data", default=None,
                   help="数据集名(如 helmet_detection_v1)或 yaml 路径")

    # 配置
    p.add_argument("--config", default="val", help="运行时 val 配置文件(默认 val.yaml)")
    p.add_argument("--no-archive", action="store_true", help="不归档运行时配置")

    # 评估控制(覆盖 val.yaml)
    p.add_argument("--split", default=None, choices=["train", "val", "test"],
                   help="数据集划分(默认走 val.yaml)")
    p.add_argument("--conf", type=float, default=None, help="置信度阈值")
    p.add_argument("--iou", type=float, default=None, help="NMS IoU 阈值")
    p.add_argument("--imgsz", type=int, default=None, help="推理输入尺寸")
    p.add_argument("--device", default=None, help="运行设备(cpu / 0 / 0,1)")
    p.add_argument("--batch", type=int, default=None, help="批大小")
    p.add_argument("--no-plots", dest="plots", action="store_false", default=None,
                   help="不生成评估图表(会关闭 accuracy 的混淆矩阵来源)")

    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    return p


_CLI_SWITCHES = {"config", "no_archive", "model", "models", "log_level", "no_plots"}


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """收集进入 val 配置的字段(排除 CLI 行为开关)."""
    return {k: v for k, v in vars(args).items()
            if v is not None and k not in _CLI_SWITCHES}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cli_overrides = _collect_cli_overrides(args)
    if args.plots is False:
        cli_overrides["plots"] = False

    # 1) 建 val 配置(跟 odp-train 同构)
    config_ref = args.config
    if "/" in config_ref or "\\" in config_ref:
        yaml_path = config_ref
    else:
        yaml_path = paths.RUNTIME_CONFIGS_DIR / (
            config_ref if config_ref.endswith(".yaml") else f"{config_ref}.yaml")
    try:
        config, merger = build_val_config(yaml_path=str(yaml_path),
                                          cli_args=cli_overrides or None)
    except Exception as e:
        sys.stderr.write(f"\n评估配置错误: {e}\n")
        return 1

    # 2) 解析数据集 yaml
    data_yaml = refs.resolve_dataset_yaml(args.data) if args.data else refs.resolve_dataset_yaml(
        getattr(config, "data", None) or "")

    sub_dir = "single" if args.model else "compare"
    with RunContext("evaluation", sub_dir=sub_dir) as run:
        # 模型名用于日志预命名(单模型用 --model, 多模型用第一个)
        first_model = args.model or (args.models[0] if args.models else "eval")
        get_logger(
            base_path=paths.LOGGING_DIR,
            log_type="evaluate",
            run_id=run.run_id,
            model_name=first_model,
            log_level=getattr(logging, args.log_level),
        )
        logger = logging.getLogger("od_platform")
        logger.info("=" * 60)
        logger.info("odp-eval 启动 | run_id=%s", run.run_id)
        logger.info("数据集: %s | split=%s", data_yaml, getattr(config, "split", "val"))

        if args.model:
            # 单模型评估
            from od_platform.model_eval import evaluate_model
            result = evaluate_model(
                config=config, data_yaml=data_yaml, run=run,
                model_ref=args.model, merger=merger,
            )
        else:
            # 多模型对比
            from od_platform.model_eval import compare_models
            result = compare_models(
                config=config, data_yaml=data_yaml, run=run,
                model_refs=args.models, merger=merger,
            )

    if result.success:
        return 0
    sys.stderr.write(f"\n评估失败: {result.message}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
