#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练报告数据读取——YAML / CSV / JSON.

设计纪律:
  · 所有 read_* 函数失败返回 None 或空结构, 不抛异常
  · 调用方直接使用返回值, 无需 try/except
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ============================================================
# EpochRow: results.csv 单行
# ============================================================

@dataclass(frozen=True)
class EpochRow:
    """单轮训练/验证指标(从 results.csv 解析)."""
    epoch: int
    time_seconds: float = 0.0
    train_box_loss: Optional[float] = None
    train_cls_loss: Optional[float] = None
    train_dfl_loss: Optional[float] = None
    val_box_loss: Optional[float] = None
    val_cls_loss: Optional[float] = None
    val_dfl_loss: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    mAP50: Optional[float] = None
    mAP50_95: Optional[float] = None


# ============================================================
# 数据集 YAML
# ============================================================

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


# ============================================================
# 划分报告 (split_report.json)
# ============================================================

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


# ============================================================
# 训练审计 (odp_audit.json)
# ============================================================

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


# ============================================================
# results.csv
# ============================================================

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


def _csv_float(row: Dict[str, str], key: str) -> Optional[float]:
    val = row.get(key, "").strip()
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ============================================================
# 训练 run 目录查找
# ============================================================

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
