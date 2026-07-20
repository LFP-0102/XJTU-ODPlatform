#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : train_model.py
# @Project   : ODPlatform
# @Function  : 训练服务——串联「配置→模型→训练→归档」的完整训练流水线
"""训练服务: TrainService 把配置子系统、ultralytics YOLO、结果/归档串成一条流水线。

典型用法 (脚本):
    from od_platform.model_train.train_model import TrainService
    service = TrainService(dataset="helmet_detection_v1", model="yolo12n.pt")
    metrics = service.run()

也可传入 CLI 参数覆盖 YAML 配置:
    service = TrainService(
        dataset="helmet_detection_v1",
        model="yolo12n.pt",
        cli_args={"epochs": 200, "batch": 32},
    )
    metrics = service.run()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ultralytics import YOLO

from od_platform.common import paths
from od_platform.common.refs import resolve_dataset_yaml
from od_platform.common.run_context import RunContext
from od_platform.model_train.archive import archive_best_weight
from od_platform.model_train.result import TrainMetrics, log_train_metrics
from od_platform.runtime_config import build_train_config

logger = logging.getLogger(__name__)


class TrainService:
    """训练服务: 加载配置 → 训练 → 收集指标 → 归档权重。

    设计要点:
      - run_id 在训练开始前由 RunContext 生成 (唯一 now()), 保证命名与审计一致
      - 数据集名作为 yaml 名字传入, 通过 refs.resolve_dataset_yaml() 解析为绝对路径
      - 模型名支持路径或 ultralytics 官方模型名 (如 yolo12n.pt)
    """

    def __init__(
        self,
        dataset: str,
        model: str = "yolo12n.pt",
        *,
        cli_args: Optional[Union[Dict[str, Any], "argparse.Namespace"]] = None,
        yaml_path: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None,
    ):
        """
        Args:
            dataset:         数据集名 (如 "helmet_detection_v1") 或 yaml 路径
            model:           模型文件路径或 ultralytics 模型名 (如 "yolo12n.pt")
            cli_args:        CLI 参数 (argparse.Namespace 或 dict), 覆盖 YAML 默认值
            yaml_path:       train YAML 路径, 默认 configs/runtime/train.yaml
            experiment_name: 实验名 (None=自动生成 exp/exp2/...)
        """
        self._dataset_name = dataset
        self._model_ref = model
        self._cli_args = cli_args
        self._yaml_path = yaml_path
        self._experiment_name = experiment_name

        # 解析数据集 yaml 为绝对路径
        self._dataset_yaml = resolve_dataset_yaml(self._dataset_name)

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(self) -> TrainMetrics:
        """执行完整训练流水线。

        Returns:
            TrainMetrics: 训练结果的结构化快照
        """
        with RunContext("model_train") as run:
            return self._execute(run)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _execute(self, run: RunContext) -> TrainMetrics:
        """在已有 RunContext 内执行训练 (供 orchestrator 注入现场)。"""
        logger.info("=" * 60)
        logger.info("训练流水线启动")
        logger.info("  数据集: %s", self._dataset_name)
        logger.info("  数据集 YAML: %s", self._dataset_yaml)
        logger.info("  模型: %s", self._model_ref)
        logger.info("  run_id: %s", run.run_id)
        logger.info("=" * 60)

        # ---- Step A: 加载 + 合并配置 ----
        logger.info("[Step 1/5] 加载训练配置...")
        config, merger = build_train_config(
            yaml_path=self._yaml_path,
            cli_args=self._cli_args,
        )
        if config is None:
            raise RuntimeError("训练配置构建失败 (config is None)")

        # 把数据集 yaml 路径注入配置 (data 字段)
        config.data = str(self._dataset_yaml)

        # 如果调用方传了 experiment_name, 覆盖 YAML 里的值
        if self._experiment_name:
            config.experiment_name = self._experiment_name

        # 如果模型路径是本地文件, 解析为绝对路径
        model_path = self._resolve_model(self._model_ref)
        config.model = str(model_path)

        # 打印配置摘要
        logger.info("配置溯源报告:\n%s", merger.get_source_report())
        logger.info(
            "有效配置: model=%s, data=%s, epochs=%s, batch=%s, imgsz=%s, device=%s",
            config.model, config.data, config.epochs,
            config.batch, config.imgsz, config.device,
        )

        # ---- Step B: 初始化模型 ----
        logger.info("[Step 2/5] 初始化 YOLO 模型...")
        model = YOLO(str(model_path))

        # ---- Step C: 训练 ----
        logger.info("[Step 3/5] 开始训练...")
        train_kwargs = config.to_ultralytics_kwargs()
        logger.info(
            "传给 ultralytics 的参数: %s",
            {k: v for k, v in train_kwargs.items()
             if k not in ("data", "model")},
        )

        results = model.train(**train_kwargs)

        # ---- Step D: 收集指标 ----
        logger.info("[Step 4/5] 收集训练结果...")
        metrics = TrainMetrics.from_yolo_results(
            results,
            run_id=run.run_id,
            model_trainer=model.trainer if hasattr(model, "trainer") else None,
        )
        log_train_metrics(metrics)

        # ---- Step E: 归档最佳权重 ----
        logger.info("[Step 5/5] 归档最佳权重...")
        stem = f"{Path(self._model_ref).stem}_{self._dataset_name}_{run.run_id}"
        archive_best_weight(metrics.save_dir, stem)

        # 保存审计快照
        snapshot = config.to_audit_snapshot()
        snapshot["train_metrics"] = metrics.to_dict()
        audit_path = run.artifact_path("audit.json")
        audit_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("审计快照已保存: %s", audit_path)

        logger.info("=" * 60)
        logger.info("训练流水线完成 ✓")
        logger.info("=" * 60)

        return metrics

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model(model_ref: str) -> Path:
        """解析模型引用: 本地文件 → 绝对路径, 否则原样返回。"""
        p = Path(model_ref)
        if p.exists():
            return p.resolve()
        # 尝试在 model_train 目录下找 (yolo12n.pt 放在这里)
        candidate = (
            paths.APP_DIR / "src" / "od_platform" / "model_train" / model_ref
        )
        if candidate.exists():
            return candidate.resolve()
        # 尝试在 models/pretrained 下找
        candidate2 = paths.PRETRAINED_MODELS_DIR / model_ref
        if candidate2.exists():
            return candidate2.resolve()
        # 都不存在, 原样返回 (ultralytics 会自动下载官方模型)
        return Path(model_ref)


# ============================================================
# 便捷函数: 一行启动训练
# ============================================================

def train(
    dataset: str,
    model: str = "yolo12n.pt",
    *,
    epochs: Optional[int] = None,
    batch: Optional[int] = None,
    imgsz: Optional[int] = None,
    device: Optional[Union[int, str]] = None,
    experiment_name: Optional[str] = None,
    **extra_kwargs,
) -> TrainMetrics:
    """一行启动训练的便捷函数。

    Args:
        dataset:         数据集名 (如 "helmet_detection_v1")
        model:           模型文件 (如 "yolo12n.pt")
        epochs:          覆盖训练轮数 (None=使用 YAML 默认 100)
        batch:           覆盖批次大小 (None=使用 YAML 默认 16)
        imgsz:           覆盖图像尺寸 (None=使用 YAML 默认 640)
        device:          覆盖设备 (None=自动选择)
        experiment_name: 实验名
        **extra_kwargs:  其他覆盖参数 (如 lr0=0.001, optimizer="AdamW")

    Returns:
        TrainMetrics: 训练结果的结构化快照
    """
    cli_overrides: Dict[str, Any] = {}
    if epochs is not None:
        cli_overrides["epochs"] = epochs
    if batch is not None:
        cli_overrides["batch"] = batch
    if imgsz is not None:
        cli_overrides["imgsz"] = imgsz
    if device is not None:
        cli_overrides["device"] = device
    cli_overrides.update(extra_kwargs)

    service = TrainService(
        dataset=dataset,
        model=model,
        cli_args=cli_overrides if cli_overrides else None,
        experiment_name=experiment_name,
    )
    return service.run()
