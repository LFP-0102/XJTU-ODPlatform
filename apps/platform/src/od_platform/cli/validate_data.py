#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :validate_data.py
# @Time      :2026/7/17 15:56:10
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
# apps/platform/src/od_platform/cli/validate_data.py
"""odp-validate:训练前质量闸门的命令行入口。退出码是给机器的第一公民。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from od_platform.common import refs
from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.data_validation.registry import CheckSeverity
from od_platform.data_validation.service import validate_dataset

# 总评 severity → 进程退出码。CI 直接读这个数。
_EXIT_CODE = {
    CheckSeverity.ERROR:   2,   # 阻断:必须停
    CheckSeverity.WARNING: 1,   # 有条件放行:人工确认
    CheckSeverity.INFO:    0,
    CheckSeverity.PASS:    0,
}


def main() -> int:
    parser = argparse.ArgumentParser(prog="odp-validate",
                                     description="数据集验证(训练前质量闸门)")
    parser.add_argument("--dataset", help="数据集名(如 MRI_PASCAL)或 yaml 路径")
    parser.add_argument("--run-id", default=None,
                        help="指定 run_id(默认时间戳;CI 可传流水线号)")
    parser.add_argument("--no-report", action="store_true",
                        help="只判定不落盘 report.json / results.csv")
    args = parser.parse_args()

    get_logger(paths.LOGGING_DIR, log_type="validate")
    report = validate_dataset(refs.resolve_dataset_yaml(args.dataset),
                          run_id=args.run_id, write_report=not args.no_report)
    return _EXIT_CODE.get(report.overall_severity, 2)


if __name__ == "__main__":
    sys.exit(main())