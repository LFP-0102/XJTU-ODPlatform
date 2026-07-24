#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : tensorboard.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/cli/tensorboard.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-23 15:00:00
# @Version    : v1.0.0
# @Description:TensorBoard 启动器 CLI
# @ChangeLog:
#   2026-07-23:00:00 | Matri | v1.0.0 | 初始化创建
"""TensorBoard 启动器 —— 一键可视化 ultralytics 训练日志."""
from __future__ import annotations

import argparse
import logging
import sys

from od_platform.common.paths import RUNS_DIR

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="odp-tensorboard",
        description="启动 TensorBoard 查看训练日志",
        epilog="示例: odp-tensorboard --logdir runs/training/ --port 6006",
    )
    parser.add_argument(
        "--logdir", default=None,
        help=f"TensorBoard 日志目录 (默认: {RUNS_DIR / 'training'})",
    )
    parser.add_argument("--port", type=int, default=6006, help="端口号 (默认: 6006)")
    parser.add_argument("--host", default="localhost", help="绑定地址 (默认: localhost)")
    parser.add_argument("--bind-all", action="store_true", help="绑定所有网络接口")

    args = parser.parse_args()
    logdir = args.logdir or str(RUNS_DIR / "training")

    try:
        from tensorboard.program import TensorBoard
    except ImportError:
        print("TensorBoard 未安装. 请执行: pip install tensorboard")
        return 1

    argv = [None, "--logdir", logdir, "--port", str(args.port)]
    if args.bind_all:
        argv.append("--bind_all")
    else:
        argv.extend(["--host", args.host])

    tb = TensorBoard()
    tb.configure(argv=argv)
    url = f"http://{args.host}:{args.port}/" if not args.bind_all else f"http://localhost:{args.port}/"
    print(f"TensorBoard 已启动: {url}")
    tb.launch()
    return 0


if __name__ == "__main__":
    sys.exit(main())
