#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : report_builder.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_train/report_builder.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-24
# @Version    : v1.0.0
# @Description:训练报告构建器——数据读取 + 分析 + 组装
# @ChangeLog:
#   2026-07-24 | Matri | v1.0.0 | 从 train_report 迁移至 model_train
"""训练报告构建器——从 run 目录读取所有数据源, 组装 TrainingReport."""
from __future__ import annotations

import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from od_platform.common.paths import DATASET_CONFIG_DIR, RUNS_DIR
from od_platform.model_train.report import EpochRow, TrainingReport, save_report

logger = logging.getLogger(__name__)


# ============================================================
# 数据读取
# ============================================================

def _csv_float(row: Dict[str, str], key: str) -> Optional[float]:
    val = row.get(key, "").strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


def read_dataset_yaml(path: Path) -> Dict[str, Any]:
    """读取数据集 YAML, 返回 parsed dict (失败返回 {})."""
    try:
        if not path.exists():
            logger.debug("数据集 YAML 不存在: %s", path)
            return {}
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception:
        logger.debug("数据集 YAML 解析失败: %s", path, exc_info=True)
        return {}


def read_split_report(manifest_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """从 manifest_path 推导 split_report.json 并读取."""
    if not manifest_path:
        return None
    try:
        manifest = Path(manifest_path)
        if not manifest.exists():
            return None
        report_path = manifest.parent / "split_report.json"
        if not report_path.exists():
            return None
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("划分报告读取失败", exc_info=True)
        return None


def read_audit_json(run_dir: Path) -> Dict[str, Any]:
    """读取训练审计 JSON(包含 config + metrics 快照). 失败返回 {}."""
    audit_path = run_dir / "odp_audit.json"
    try:
        if not audit_path.exists():
            return {}
        return json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("审计 JSON 读取失败: %s", audit_path, exc_info=True)
        return {}


def read_results_csv(run_dir: Path) -> List[EpochRow]:
    """解析训练产出的 results.csv, 返回逐轮指标列表."""
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        logger.debug("results.csv 不存在: %s", csv_path)
        return []

    rows: List[EpochRow] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for line in reader:
                try:
                    rows.append(EpochRow(
                        epoch=int(line.get("epoch", 0)),
                        time_seconds=_csv_float(line, "time"),
                        train_box_loss=_csv_float(line, "train/box_loss"),
                        train_cls_loss=_csv_float(line, "train/cls_loss"),
                        train_dfl_loss=_csv_float(line, "train/dfl_loss"),
                        val_box_loss=_csv_float(line, "val/box_loss"),
                        val_cls_loss=_csv_float(line, "val/cls_loss"),
                        val_dfl_loss=_csv_float(line, "val/dfl_loss"),
                        precision=_csv_float(line, "metrics/precision(B)"),
                        recall=_csv_float(line, "metrics/recall(B)"),
                        mAP50=_csv_float(line, "metrics/mAP50(B)"),
                        mAP50_95=_csv_float(line, "metrics/mAP50-95(B)"),
                    ))
                except (ValueError, KeyError):
                    pass
    except Exception:
        logger.debug("results.csv 解析失败", exc_info=True)
    return rows


def find_latest_run(training_dir: Path) -> Optional[Path]:
    """查找最近一次训练 run 目录(按目录名时间戳排序)."""
    try:
        if not training_dir.exists():
            return None
        dirs = sorted(
            [d for d in training_dir.iterdir() if d.is_dir()],
            reverse=True,
        )
        return dirs[0] if dirs else None
    except Exception:
        return None


# ============================================================
# 分析
# ============================================================

def _fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def analyze_class_imbalance(
    class_distribution: Dict[str, Dict[str, int]],
) -> Optional[str]:
    """检测类别不均衡, 返回警告文字或 None."""
    if not class_distribution:
        return None
    totals: Dict[str, int] = {}
    for sp_dist in class_distribution.values():
        for cls, cnt in sp_dist.items():
            totals[cls] = totals.get(cls, 0) + cnt
    if len(totals) < 2:
        return None
    max_cls = max(totals, key=totals.get)
    min_cls = min(totals, key=totals.get)
    max_cnt = totals[max_cls]
    min_cnt = totals[min_cls]
    if min_cnt == 0:
        return None
    ratio = max_cnt / min_cnt
    if ratio > 10:
        return (
            f"严重类别不均衡: `{max_cls}`({max_cnt}框) 是 `{min_cls}`({min_cnt}框) "
            f"的 {ratio:.1f} 倍。少数类样本严重不足, 模型可能难以学习其特征。"
        )
    elif ratio > 3:
        return (
            f"类别不均衡: `{max_cls}`({max_cnt}框) 是 `{min_cls}`({min_cnt}框) "
            f"的 {ratio:.1f} 倍。建议关注少数类的 Precision/Recall。"
        )
    return None


def analyze_bbox_distribution(
    bbox_size_distribution: Dict[str, Dict[str, int]],
) -> Optional[str]:
    """检测小目标占比, 返回分析文本或 None."""
    if not bbox_size_distribution:
        return None
    total = 0
    small = 0
    for d in bbox_size_distribution.values():
        small += d.get("small", 0)
        total += d.get("small", 0) + d.get("medium", 0) + d.get("large", 0)
    if total == 0:
        return None
    ratio = small / total
    if ratio > 0.7:
        return (
            f"小目标占比较高({ratio*100:.1f}%), "
            f"当前 imgsz 配置下小目标特征提取困难。建议: 增大 imgsz、使用更小 stride 的模型。"
        )
    elif ratio > 0.5:
        return (
            f"小目标占比偏高({ratio*100:.1f}%), "
            f"小目标检测可能精度不足, 可考虑增大 imgsz 或使用 multi_scale 训练。"
        )
    return None


def analyze_convergence(epoch_history: List[EpochRow]) -> List[str]:
    """分析训练收敛情况: loss 下降、过拟合、mAP 平台期."""
    notes: List[str] = []
    if len(epoch_history) < 3:
        notes.append("训练轮数过少(<3 轮), 无法判断收敛性。")
        return notes

    first = epoch_history[0]
    last = epoch_history[-1]

    # train/box_loss 下降检查
    if first.train_box_loss is not None and last.train_box_loss is not None:
        change = last.train_box_loss - first.train_box_loss
        if change > 0.1:
            notes.append(
                f"train/box_loss 未有效下降(变化={_fmt(change)}), "
                f"学习率可能偏高或训练发散。"
            )

    # 过拟合检查: 后半段 val_loss 均值是否高于前半段
    mid = len(epoch_history) // 2
    early = epoch_history[:mid]
    late = epoch_history[mid:]
    early_vals = [r.val_box_loss for r in early if r.val_box_loss is not None]
    late_vals = [r.val_box_loss for r in late if r.val_box_loss is not None]
    if early_vals and late_vals:
        early_avg = sum(early_vals) / len(early_vals)
        late_avg = sum(late_vals) / len(late_vals)
        if late_avg - early_avg > 0.05:
            notes.append(
                f"可能过拟合: 验证 val/box_loss 后半段均值({_fmt(late_avg)}) "
                f"高于前半段({_fmt(early_avg)})。"
                f"建议: 加强数据增强、增大 weight_decay、或启用 dropout。"
            )

    # mAP 平台期检查
    mAPs = [(i, r.mAP50_95) for i, r in enumerate(epoch_history) if r.mAP50_95 is not None]
    if len(mAPs) >= 5:
        best_idx, best_val = max(mAPs, key=lambda x: x[1])
        if best_idx < len(epoch_history) - 10:
            notes.append(
                f"mAP50-95 在第 {best_idx + 1} 轮达到最佳({_fmt(best_val)})后 "
                f"不再提升, 训练已充分收敛。可考虑减少 epochs。"
            )

    if not notes:
        notes.append("训练过程正常收敛, 未发现异常。")
    return notes


def generate_suggestions(
    *,
    class_distribution: Dict[str, Dict[str, int]],
    bbox_size_distribution: Dict[str, Dict[str, int]],
    mAP50: float,
    mAP50_95: float,
    epochs_actual: int,
    epoch_history: List[EpochRow],
) -> List[str]:
    """基于报告数据自动生成改进建议."""
    suggestions: List[str] = []

    # 类别不均衡建议
    totals: Dict[str, int] = {}
    for sp_dist in class_distribution.values():
        for cls, cnt in sp_dist.items():
            totals[cls] = totals.get(cls, 0) + cnt
    if len(totals) >= 2:
        max_cnt = max(totals.values())
        min_cnt = min(totals.values())
        if max_cnt > 0 and min_cnt > 0 and max_cnt / min_cnt > 5:
            suggestions.append(
                "类别不均衡: 使用类别加权损失(增大少数类权重)、"
                "对少数类做 Oversampling 数据增强。"
            )

    # 小目标建议
    small_total = sum(
        d.get("small", 0) for d in bbox_size_distribution.values()
    )
    all_total = sum(
        d.get(k, 0) for d in bbox_size_distribution.values()
        for k in ("small", "medium", "large")
    )
    if all_total > 0 and small_total / all_total > 0.5:
        suggestions.append(
            "小目标占比高: 增大 imgsz(如 960/1280)、开启 multi_scale 训练。"
        )

    # mAP 偏低建议
    if not math.isnan(mAP50) and mAP50 < 0.5:
        suggestions.append(
            f"mAP50 较低({_fmt(mAP50)}): 检查标注质量、增加 epochs、"
            f"或尝试更大的模型。"
        )
    if not math.isnan(mAP50_95) and mAP50_95 < 0.3:
        suggestions.append(
            f"mAP50-95 较低({_fmt(mAP50_95)}): 模型定位精度不足, "
            f"考虑增大 imgsz 或使用更精确的模型架构。"
        )

    # 训练轮数
    if epochs_actual < 20:
        suggestions.append(
            f"训练轮数较少({epochs_actual}), 如需更充分收敛建议增加 epochs。"
        )

    # 过拟合检查
    if len(epoch_history) > 10:
        mid = len(epoch_history) // 2
        early = [r for r in epoch_history[:mid] if r.val_box_loss is not None]
        late = [r for r in epoch_history[mid:] if r.val_box_loss is not None]
        if early and late:
            early_avg = sum(r.val_box_loss for r in early) / len(early)  # type: ignore[arg-type]
            late_avg = sum(r.val_box_loss for r in late) / len(late)  # type: ignore[arg-type]
            if late_avg - early_avg > 0.05:
                suggestions.append(
                    "过拟合风险: 加强数据增强(mosaic/mixup)、增大 weight_decay、"
                    "或减少 epochs。"
                )

    if not suggestions:
        suggestions.append(
            "当前训练配置和结果处于合理范围。"
            "常用优化方向: 数据增强调参、模型缩放(n/s/m/l/x)、超参数搜索。"
        )
    return suggestions


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
    return None


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
    dataset: Dict[str, Any] = {}
    if data_yaml_path:
        dataset = read_dataset_yaml(data_yaml_path)
        _fill_from_dataset(report, dataset, data_yaml_path)

    # ---- 3. 划分报告 ----
    manifest_path = _extract_manifest_path(audit, dataset if dataset else None)
    if not manifest_path:
        # fallback: try from dataset yaml
        if data_yaml_path:
            ds = read_dataset_yaml(data_yaml_path)
            meta = ds.get("odp_meta", {}) if ds else {}
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
