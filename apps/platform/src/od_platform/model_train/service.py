#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : service.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_train/service.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-19 11:07:53
# @Version    : v1.0.0
# @Description:模型训练整体编排服务
# @ChangeLog:
#   2026-07-19:07:53 | Matri | v1.0.0 | 初始化创建
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from od_platform.common import refs
from od_platform.common.paths import PRETRAINED_MODELS_DIR
from od_platform.common.run_context import RunContext
from od_platform.common.report_config import log_config_report
from od_platform.model_train.archive import archive_best_weight
from od_platform.model_train.result import TrainMetrics, log_train_metrics

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class TrainResult:
    success: bool
    run_id : str
    save_dir: Optional[Path] = None
    best_weight: Optional[Path] = None
    metrics: Optional[TrainMetrics] = None
    message: str = ""

def _resolve_model_arg(model: str) -> str:
    p = refs.resolve_ref(model, base_dir=PRETRAINED_MODELS_DIR, default_suffix=".pt")
    if p.exists():
        logger.info(f"模型解析： %s", p)
        return str(p)
    logger.info(f"本地未发现预训练模型{p.name}, Ultralytics会自动下载该模型 {model}")
    return model

def _run_training(model_arg: str, config: Any, data_yaml: Path, run: RunContext) -> Any:
    from ultralytics import YOLO
    model = YOLO(model_arg)
    kwargs = config.to_ultralytics_kwargs()
    kwargs.pop("data", None)
    kwargs.pop("model", None)
    kwargs.update(project=str(run.run_dir.parent), name=run.run_id, exist_ok = True)

    return model.train(data=str(data_yaml), **kwargs)

def _write_audit(run: RunContext, config: Any, merger: Optional[Any],
                data_yaml: Path,stem: str, metrics: TrainMetrics) -> Path:

    audit = {
        "run_id": run.run_id,
        "created_at": run.created_at,
        "artifacts": {"run_dir": str(run.run_dir), "stem": stem},
        "experiment_name": getattr(config, 'experiment_name', None),
        "data_yaml": str(data_yaml),
        "config": config.to_audit_snapshot(),
        "metrics": metrics.to_dict()
    }
    if merger is not None:
        audit["config_sources"] = merger.to_audit_log()
    path =run.artifact_path("odp_audit.json")
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2),encoding='utf-8')
    logger.info(f"审计信息已写入 %s", path)
    return path

def train_yolo(*, config: Any, data_yaml: Path, run: RunContext, stem: str,
            merger: Optional[Any] = None, archive: bool = True) -> TrainResult:
    logger.info("=" * 60)
    logger.info(f"训练开始 | run_id={run.run_id}")
    logger.info(f"输出目录：{run.run_dir}")
    logger.info(f"产物名称：{stem}")
    logger.info(f"实验名称：{getattr(config, "experiment_name", None) or ("未命名")}")
    logger.info(f"数据集名称：{data_yaml}")
    logger.info("=" * 60)

    if merger is not None:
        log_config_report(config, merger, logger=logger)

    logger.info("模型声明：%s",config.model if hasattr(config, "model") else "配置没有model字段")
    model_ref = config.model if hasattr(config, "model") else "yolov8n.pt"
    model_arg = _resolve_model_arg(model_ref)

    # 训练
    try:
        results = _run_training(model_arg, config, data_yaml, run)
    except KeyboardInterrupt:
        logger.warn(f"训练被手动中断 | run_id={run.run_id}")
        logger.warn(f"训练结果保留在 {run.run_dir}")
        return TrainResult(success=False, run_id=run.run_id, save_dir=run.run_dir,message="训练被手动中断")
    except Exception as e:
        logger.exception(f"训练失败 | run_id={run.run_id} 错误如下：{e}")
        return TrainResult(success=False, run_id=run.run_id, save_dir=run.run_dir,
                        message=f"训练失败 | run_id={run.run_id} 错误如下：{e}")
    # 成功记录指标
    metrics = TrainMetrics.from_yolo_results(results, run_id=run.run_id)
    log_train_metrics(metrics, logger=logger)

    _write_audit(run, config, merger, data_yaml, stem, metrics)

    best = archive_best_weight(run.run_dir, stem) if archive else None

    logger.info(f"训练完成 | run_id={run.run_id} | 最佳模型保存在 {best}")

    return TrainResult(success=True, run_id=run.run_id,
                save_dir=run.run_dir, best_weight=best, metrics=metrics)

