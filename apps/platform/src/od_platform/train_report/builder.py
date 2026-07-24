#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练报告构建器——从 run 目录读取所有数据源, 组装 TrainingReport."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from od_platform.common.paths import DATASET_CONFIG_DIR, RUNS_DIR
from od_platform.train_report.analysis import (
    analyze_bbox_distribution,
    analyze_class_imbalance,
    analyze_convergence,
    generate_suggestions,
)
from od_platform.train_report.readers import (
    find_latest_run,
    read_audit_json,
    read_dataset_yaml,
    read_results_csv,
    read_split_report,
)
from od_platform.train_report.report import TrainingReport, save_report

logger = logging.getLogger(__name__)


# ============================================================
# 主入口
# ============================================================

def build_training_report(run_dir: Path) -> TrainingReport:
    """从训练 run 目录构建完整分析报告.

    自动查找:
      · odp_audit.json → 训练配置 + 最终指标
      · dataset YAML   → 数据集元信息
      · split_report.json → 类别/bbox/JS 分布详情
      · results.csv    → 逐轮训练过程

    Args:
        run_dir: runs/training/<run_id>/ 目录路径

    Returns:
        TrainingReport (部分字段可能为空, 取决于数据可用性)
    """
    report = TrainingReport(
        run_id=run_dir.name,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # ---- 1. 审计 JSON (config + metrics) ----
    audit = read_audit_json(run_dir)
    _fill_from_audit(report, audit)

    # ---- 2. 数据集 YAML ----
    data_yaml_path = _resolve_data_yaml(audit)
    if data_yaml_path:
        dataset = read_dataset_yaml(data_yaml_path)
        _fill_from_dataset(report, dataset, data_yaml_path)

    # ---- 3. 划分报告 ----
    manifest_path = _extract_manifest_path(audit, dataset if 'dataset' in dir() else None)  # type: ignore[possibly-undefined-name]
    if not manifest_path:
        # fallback: try from dataset yaml
        if data_yaml_path:
            dataset = read_dataset_yaml(data_yaml_path)
            meta = dataset.get("odp_meta", {}) if dataset else {}
            manifest_path = meta.get("manifest_path")
    if manifest_path:
        split_data = read_split_report(manifest_path)
        if split_data:
            _fill_from_split_report(report, split_data)

    # ---- 4. 逐轮训练过程 ----
    report.epoch_history = read_results_csv(run_dir)
    report.epochs_actual = len(report.epoch_history)

    # ---- 5. 分析 ----
    cls_warn = analyze_class_imbalance(report.class_distribution)
    if cls_warn:
        report.analysis_warnings.append(cls_warn)

    bbox_warn = analyze_bbox_distribution(report.bbox_size_distribution)
    if bbox_warn:
        report.analysis_warnings.append(bbox_warn)

    conv_notes = analyze_convergence(report.epoch_history)
    report.convergence_note = "; ".join(conv_notes)
    for note in conv_notes:
        report.analysis_warnings.append(note)

    report.improvement_suggestions = generate_suggestions(
        class_distribution=report.class_distribution,
        bbox_size_distribution=report.bbox_size_distribution,
        mAP50=report.mAP50,
        mAP50_95=report.mAP50_95,
        epochs_actual=report.epochs_actual,
        epoch_history=report.epoch_history,
    )

    return report


# ============================================================
# 内部填充函数
# ============================================================

def _fill_from_audit(report: TrainingReport, audit: Dict[str, Any]) -> None:
    """从 odp_audit.json 提取训练配置和最终指标."""
    if not audit:
        return

    config = audit.get("config", {})
    values = config.get("values", {}) if isinstance(config, dict) else {}

    if isinstance(values, dict):
        report.task = str(values.get("task", ""))
        report.model_name = str(values.get("model", ""))
        report.epochs_planned = int(values.get("epochs", 0))
        report.imgsz = int(values.get("imgsz", 640))
        report.batch = values.get("batch", 16)
        report.optimizer = str(values.get("optimizer", "auto"))
        report.lr0 = float(values.get("lr0", 0.01))
        report.lrf = float(values.get("lrf", 0.01))
        report.pretrained = values.get("pretrained", True)
        report.close_mosaic = int(values.get("close_mosaic", 10))
        report.seed = int(values.get("seed", 0))
        report.config_snapshot = config

    metrics = audit.get("metrics", {})
    if isinstance(metrics, dict):
        overall = metrics.get("overall", {})
        if isinstance(overall, dict):
            report.mAP50 = float(overall.get("metrics/mAP50(B)", float("nan")))
            report.mAP50_95 = float(overall.get("metrics/mAP50-95(B)", float("nan")))
            report.precision = float(overall.get("metrics/precision(B)", float("nan")))
            report.recall = float(overall.get("metrics/recall(B)", float("nan")))
            report.fitness = float(overall.get("fitness", float("nan")))
        report.speed_ms = dict(metrics.get("speed_ms", {}))
        report.class_map_50_95 = dict(metrics.get("class_map_50_95", {}))


def _fill_from_dataset(
    report: TrainingReport, dataset: Dict[str, Any], yaml_path: Path,
) -> None:
    """从数据集 YAML 提取元信息."""
    if not dataset:
        return

    odp_meta = dataset.get("odp_meta", {}) or {}
    report.dataset_name = str(odp_meta.get("dataset_name", yaml_path.stem))
    report.dataset_yaml = str(yaml_path)

    names = dataset.get("names", {})
    if isinstance(names, dict):
        report.class_names = [str(v) for v in names.values()]
        report.num_classes = len(report.class_names)

    split_meta = odp_meta.get("split", {})
    if isinstance(split_meta, dict):
        report.split_strategy = split_meta.get("strategy")
        report.split_seed = split_meta.get("seed")
        rates = split_meta.get("rate")
        if isinstance(rates, dict):
            report.split_rates = {str(k): float(v) for k, v in rates.items()}
        counts = split_meta.get("counts", {})
        if isinstance(counts, dict):
            report.split_counts = {str(k): int(v) for k, v in counts.items()}


def _fill_from_split_report(
    report: TrainingReport, split_data: Dict[str, Any],
) -> None:
    """从划分报告 JSON 提取详细分布数据."""
    if not split_data:
        return
    report.box_counts = split_data.get("box_counts", {})
    report.class_distribution = split_data.get("class_distribution", {})
    report.bbox_size_distribution = split_data.get("bbox_size_distribution", {})
    report.js_divergence = split_data.get("inter_split_consistency")
    report.image_size_stats = split_data.get("image_size_stats", {})


def _resolve_data_yaml(audit: Dict[str, Any]) -> Optional[Path]:
    """从审计数据中解析数据集 YAML 路径."""
    data_yaml_str = audit.get("data_yaml", "")
    if data_yaml_str:
        p = Path(data_yaml_str)
        if p.exists():
            return p
    # fallback: 从 config.values.data 推断
    data_name = ""
    config = audit.get("config", {})
    values = config.get("values", {}) if isinstance(config, dict) else {}
    if isinstance(values, dict):
        data_name = str(values.get("data", ""))
    if data_name and data_name.endswith(".yaml"):
        data_name = data_name[:-5]
    if data_name:
        candidate = DATASET_CONFIG_DIR / f"{data_name}.yaml"
        if candidate.exists():
            return candidate
    return None


def _extract_manifest_path(
    audit: Dict[str, Any], dataset: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """提取 manifest_path: 优先从 dataset YAML 的 odp_meta, fallback 到 audit."""
    if dataset:
        meta = dataset.get("odp_meta", {}) or {}
        mp = meta.get("manifest_path")
        if mp:
            return str(mp)
    # fallback: 从 config_sources 或其他字段
    return None


# ============================================================
# 快捷入口
# ============================================================

def generate_report(
    run_id: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Optional[Path]:
    """训练报告生成主入口.

    Args:
        run_id: 训练 run ID(如 '20260724-105403'), None=自动找最近一次
        output_path: 输出路径, None=保存到 run 目录下的 training_report.md

    Returns:
        报告文件路径, 失败返回 None
    """
    try:
        training_dir = RUNS_DIR / "training"

        if run_id:
            run_dir = training_dir / run_id
            if not run_dir.exists():
                logger.error("训练 run 目录不存在: %s", run_dir)
                return None
        else:
            run_dir = find_latest_run(training_dir)
            if run_dir is None:
                logger.error("未找到任何训练 run 目录")
                return None

        report = build_training_report(run_dir)

        if output_path is None:
            output_path = run_dir / "training_report.md"

        save_report(report, output_path)
        logger.info("训练分析报告已保存: %s", output_path)
        return output_path
    except Exception:
        logger.exception("训练报告生成失败")
        return None
