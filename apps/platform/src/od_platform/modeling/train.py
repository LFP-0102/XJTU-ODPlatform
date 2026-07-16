"""训练模块 —— 对接 ultralytics YOLO，复用 RunContext / paths / logging 体系。"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

from od_platform.common import paths
from od_platform.common.constants import DEFAULT_RANDOM_STATE
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERR = 1

# 默认模型 —— yolo12n.pt
_DEFAULT_MODEL = Path(__file__).resolve().parents[5] / "yolo12n.pt"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="odp-train",
        description="用 ultralytics YOLO 训练目标检测模型，自动对接平台数据集配置。",
    )
    p.add_argument("--dataset", required=True, help="数据集名（对应 configs/datasets/<name>.yaml）")
    p.add_argument("--model", default=str(_DEFAULT_MODEL), help=f"YOLO 模型文件路径（默认 {_DEFAULT_MODEL}）")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", default="0", help="GPU 编号；cpu 填 cpu")
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--resume", action="store_true", help="从上次 checkpoint 继续训练")
    a = p.parse_args(argv)

    # 如果传入的不是绝对路径，尝试在项目根下查找
    model_path = Path(a.model)
    if not model_path.is_absolute():
        candidate = Path(__file__).resolve().parents[5] / model_path.name
        if candidate.exists():
            model_path = candidate
    if not model_path.exists():
        logger.error("找不到模型文件: %s", model_path)
        return EXIT_ERR
    # 如果用户没指定 --model 且默认模型不存在，回退到 yolo12n.pt 同目录下的 yolov8n.pt
    if not model_path.exists():
        logger.error("找不到模型文件: %s", model_path)
        return EXIT_ERR

    get_logger(base_path=paths.LOGGING_DIR, log_type="train")

    yaml_path = paths.dataset_yaml_path(a.dataset)
    if not yaml_path.exists():
        logger.error("找不到数据集配置文件: %s", yaml_path)
        return EXIT_ERR

    with RunContext("train") as run:
        run_meta = {
            "run_id": run.run_id,
            "dataset": a.dataset,
            "model": str(model_path),
            "epochs": a.epochs,
            "imgsz": a.imgsz,
            "batch": a.batch,
            "lr0": a.lr,
            "device": a.device,
            "seed": a.seed,
            "resume": a.resume,
        }
        # yaml 里的 contract_fingerprint 与 manifest 路径一并记进元数据，供 D4 审计。
        try:
            import yaml as _yaml
            ds_cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            odp = ds_cfg.get("odp_meta", {}) if isinstance(ds_cfg, dict) else {}
            run_meta["contract_fingerprint"] = odp.get("contract_fingerprint")
            run_meta["manifest_ref"] = odp.get("manifest_path")
        except Exception:
            logger.warning("读取 dataset yaml 元信息失败，跳过 contract_fingerprint。")

        meta_path = run.artifact_path("run_meta.json")
        meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("训练元数据已写入 %s", meta_path)

        logger.info(
            "开始训练 dataset=%s model=%s epochs=%d imgsz=%d batch=%d run_id=%s",
            a.dataset, str(model_path), a.epochs, a.imgsz, a.batch, run.run_id,
        )

        # 加载模型，amp=False 避免触发 GitHub 下载
        logger.info("加载模型: %s", model_path)
        model = YOLO(str(model_path))
        results = model.train(
            data=str(yaml_path),
            epochs=a.epochs,
            imgsz=a.imgsz,
            batch=a.batch,
            lr0=a.lr,
            device=a.device,
            seed=a.seed,
            resume=a.resume,
            amp=False,
            workers=2,
            cache="disk",
            project=str(run.run_dir),
            name="weights",
            exist_ok=True,
        )

        # 确定 best.pt 路径并拷贝到 runs 根下（方便直接引用）。
        src_best = run.run_dir / "weights" / "weights" / "best.pt"
        if src_best.exists():
            dst = run.artifact_path("best.pt")
            shutil.copy2(src_best, dst)
            logger.info("最佳权重已拷贝至 %s", dst)
            print(f"✅ 训练完成: {dst}")
        else:
            logger.warning("未找到 best.pt，请检查训练结果。")
            print(f"⚠ 训练结束但未找到 best.pt，产物在 {run.run_dir / 'weights'}")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
