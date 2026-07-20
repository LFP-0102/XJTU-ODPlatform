#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : train_helmet.py
# @Project   : ODPlatform
# @Function  : 安全帽检测模型训练入口脚本
"""安全帽检测模型训练——一行启动训练。

用法:
    # 在项目根目录下运行
    python scripts/train_helmet.py                     # 默认参数
    python scripts/train_helmet.py --epochs 200        # 覆盖训练轮数
    python scripts/train_helmet.py --batch 32 --lr0 0.001  # 覆盖多个参数

训练前请确认:
    1. 数据集已转换: data/processed/helmet_detection_v1/ 存在
    2. 预训练权重: yolo12n.pt 在 model_train/ 目录下
    3. ultralytics 已安装: pip install ultralytics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目 src 加入 sys.path (确保 od_platform 可导入)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "apps" / "platform" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
from od_platform.model_train.train_model import TrainService


def main():
    parser = argparse.ArgumentParser(
        description="安全帽检测模型训练 (YOLO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/train_helmet.py
  python scripts/train_helmet.py --epochs 200 --batch 32
  python scripts/train_helmet.py --model yolo12n.pt --imgsz 800
  python scripts/train_helmet.py --device cpu           # CPU 训练
        """,
    )
    parser.add_argument(
        "--dataset", default="helmet_detection_v1",
        help="数据集名 (默认: helmet_detection_v1)",
    )
    parser.add_argument(
        "--model", default="yolo12n.pt",
        help="模型文件 (默认: yolo12n.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练轮数 (默认: 使用 YAML 中的 100)",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="批次大小 (默认: 使用 YAML 中的 16)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="图像尺寸 (默认: 使用 YAML 中的 640)",
    )
    parser.add_argument(
        "--device", default=None,
        help="设备 (默认: 自动选择, 可用 cpu / 0 / 1)",
    )
    parser.add_argument(
        "--lr0", type=float, default=None,
        help="初始学习率 (默认: 使用 YAML 中的 0.01)",
    )
    parser.add_argument(
        "--experiment-name", default=None,
        help="实验名 (默认: 自动生成)",
    )

    args = parser.parse_args()

    # ---- 装配日志系统 ----
    get_logger(
        base_path=LOGGING_DIR,
        log_type="model_train",
        model_name=args.model,
        dataset_name=args.dataset,
    )

    # ---- 组装 CLI 覆盖参数 ----
    cli_overrides = {}
    if args.epochs is not None:
        cli_overrides["epochs"] = args.epochs
    if args.batch is not None:
        cli_overrides["batch"] = args.batch
    if args.imgsz is not None:
        cli_overrides["imgsz"] = args.imgsz
    if args.device is not None:
        cli_overrides["device"] = args.device
    if args.lr0 is not None:
        cli_overrides["lr0"] = args.lr0

    # ---- 启动训练 ----
    service = TrainService(
        dataset=args.dataset,
        model=args.model,
        cli_args=cli_overrides if cli_overrides else None,
        experiment_name=args.experiment_name,
    )
    metrics = service.run()

    # ---- 最终输出 ----
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print(f"   run_id:   {metrics.run_id}")
    print(f"   mAP50:    {metrics.overall.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP50-95: {metrics.overall.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"   权重位置:  models/trained/")
    print("=" * 60)


if __name__ == "__main__":
    main()
