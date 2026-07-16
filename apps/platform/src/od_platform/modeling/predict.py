"""推理模块 —— 对图片 / 视频跑检测，画框并保存结果。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERR = 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="odp-predict",
        description="用训练好的模型对图片 / 视频做推理并保存结果。",
    )
    p.add_argument("--model", required=True, help="模型权重路径 (.pt)")
    p.add_argument("--source", required=True, help="输入图片 / 文件夹 / 视频路径")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--device", default="0")
    a = p.parse_args(argv)

    get_logger(base_path=paths.LOGGING_DIR, log_type="predict")

    source = Path(a.source)
    if not source.exists():
        logger.error("输入不存在: %s", source)
        return EXIT_ERR

    with RunContext("predict") as run:
        logger.info("开始推理 run_id=%s source=%s", run.run_id, a.source)

        model = YOLO(a.model)
        model.predict(
            source=str(source),
            imgsz=a.imgsz,
            conf=a.conf,
            device=a.device,
            project=str(run.run_dir),
            name="results",
            exist_ok=True,
            save=True,
        )

        out_dir = run.run_dir / "results"
        print(f"✅ 推理完成，结果: {out_dir}")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
