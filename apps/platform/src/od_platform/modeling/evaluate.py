"""评估模块 —— 在测试集上验证模型，输出 mAP / Precision / Recall，结果落盘供审计。"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from ultralytics import YOLO

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERR = 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="odp-evaluate",
        description="在测试集上评估 YOLO 模型，输出 mAP / Precision / Recall 等指标。",
    )
    p.add_argument("--model", required=True, help="模型权重路径 (.pt)")
    p.add_argument("--dataset", required=True, help="数据集名")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    a = p.parse_args(argv)

    get_logger(base_path=paths.LOGGING_DIR, log_type="evaluate")

    yaml_path = paths.dataset_yaml_path(a.dataset)
    if not yaml_path.exists():
        logger.error("找不到数据集配置: %s", yaml_path)
        return EXIT_ERR

    with RunContext("evaluate") as run:
        logger.info("开始评估 run_id=%s model=%s", run.run_id, a.model)

        model = YOLO(a.model)
        metrics = model.val(
            data=str(yaml_path),
            imgsz=a.imgsz,
            device=a.device,
            project=str(run.run_dir),
            name="eval",
            exist_ok=True,
            split="test",
        )

        report = {
            "run_id": run.run_id,
            "model": a.model,
            "dataset": a.dataset,
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
        }
        report_path = run.artifact_path("metrics.json")
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("评估报告已写入 %s", report_path)

        print(f"✅ mAP@50={report['mAP50']:.4f}  mAP@50-95={report['mAP50_95']:.4f}")
        print(f"   Precision={report['precision']:.4f}  Recall={report['recall']:.4f}")
        print(f"   报告: {report_path}")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
