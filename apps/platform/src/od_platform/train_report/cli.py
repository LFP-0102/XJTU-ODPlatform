#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""odp-report CLI — 训练分析报告生成器."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.train_report.builder import generate_report

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="odp-report",
        description="训练分析报告生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  odp-report                                     # 对最近一次训练生成报告
  odp-report --run 20260724-105403              # 指定 run_id
  odp-report --run 20260724-105403 -o ./my.md   # 指定输出路径
        """,
    )
    parser.add_argument(
        "--run", default=None,
        help="训练 run ID(如 20260724-105403), 默认=最近一次",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="输出路径(默认保存到 run 目录下的 training_report.md)",
    )

    args = parser.parse_args()

    get_logger(
        base_path=LOGGING_DIR,
        log_type="train_report",
        model_name="report",
        dataset_name="",
    )

    output = Path(args.output) if args.output else None

    report_path = generate_report(run_id=args.run, output_path=output)

    if report_path:
        print(f"报告已生成: {report_path}")
        return 0
    else:
        print("报告生成失败, 请检查 run_id 是否正确或日志获取详情。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
