#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : analyzer.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_eval/analyzer.py
# @Project    : XJTU-ODPlatfrom
# @Author     : ODPlatform Team
# @Function   : 评估深度分析 —— 混淆矩阵错误分析、类别性能排序、误检/漏检诊断
"""模型评估深度分析工具.

在 EvalMetrics / EvalReport 基础上提供:
  · 混淆矩阵错误分析: 每类误检(FP) / 漏检(FN) 统计
  · 类别性能排序: 按 mAP / F1 / Precision / Recall 排名
  · 问题类别诊断: 标记性能最差 / 最容易混淆的类别
  · 类别级对比: 多模型在同一类别上的表现差异

设计纪律(与 report.py 同款):
  · 纯数据产出, 不持有展示逻辑
  · 消费 EvalMetrics / ComparisonReport, 产出 dict / str
  · 所有 NaN 均被安全处理
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from od_platform.model_eval.metrics import EvalMetrics
from od_platform.model_eval.report import _fmt

logger = logging.getLogger(__name__)


# ============================================================
# 混淆矩阵错误分析
# ============================================================

@dataclass
class ConfusionAnalysis:
    """从 ultralytics 混淆矩阵结果中提取的错误分析.

    confusion_matrix.matrix: (nc+1) x (nc+1) 矩阵
      - 前 nc 行/列对应各类别
      - 最后一行: background FN (真实框未被任何预测匹配)
      - 最后一列: background FP (预测框未匹配任何真实框)
    """
    class_name: str
    class_id: int
    tp: int            # 对角线: 正确检测
    fp: int            # 列和-对角线: 该类被误检
    fn: int            # 行和-对角线: 该类被漏检
    total_gt: int      # 行和: 真实框总数
    total_pred: int    # 列和: 预测框总数

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else math.nan

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else math.nan

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r):
            return math.nan
        denom = p + r
        return 2 * p * r / denom if denom > 0 else math.nan

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "total_gt": self.total_gt,
            "total_pred": self.total_pred,
            "precision": None if math.isnan(self.precision) else round(self.precision, 4),
            "recall": None if math.isnan(self.recall) else round(self.recall, 4),
            "f1": None if math.isnan(self.f1) else round(self.f1, 4),
        }


def extract_confusion_analysis(results: Any, names: Dict[Any, str]) -> List[ConfusionAnalysis]:
    """从 ultralytics val results 提取每类混淆矩阵分析.

    Args:
        results: ultralytics model.val() 的返回值 (需要 plots=True)
        names:   类别名映射 {0: "helmet", 1: "person", ...}

    Returns:
        每类的 ConfusionAnalysis 列表, 按 class_id 排序.
        若混淆矩阵不可用, 返回空列表.
    """
    cm = getattr(results, "confusion_matrix", None)
    matrix = getattr(cm, "matrix", None) if cm is not None else None
    if matrix is None:
        logger.warning("混淆矩阵不可用(可能 plots=False), 跳过错误分析")
        return []

    try:
        import numpy as np
        mat = np.array(matrix, dtype=np.float64)
        nc = mat.shape[0] - 1  # 最后一行/列是 background
        if nc <= 0:
            return []

        out: List[ConfusionAnalysis] = []
        for cls_id in range(nc):
            tp = int(mat[cls_id, cls_id])
            # 第 cls_id 列的其余行之和 = 该类被误判为其他类 (FP: 预测为此类但不是)
            fp = int(mat[:nc, cls_id].sum()) - tp
            # 第 cls_id 行的其余列之和 = 该类被漏检 (FN: 真实为此类但预测为其他)
            fn = int(mat[cls_id, :nc].sum()) - tp
            total_gt = int(mat[cls_id, :].sum())    # 包含 background
            total_pred = int(mat[:, cls_id].sum())  # 包含 background

            out.append(ConfusionAnalysis(
                class_name=str(names.get(cls_id, f"class_{cls_id}")),
                class_id=cls_id,
                tp=tp, fp=fp, fn=fn,
                total_gt=total_gt, total_pred=total_pred,
            ))
        return out
    except Exception as e:
        logger.warning("混淆矩阵分析失败: %s", e)
        return []


# ============================================================
# 类别性能排序
# ============================================================

@dataclass
class ClassRanking:
    """类别级性能排名."""
    class_name: str
    rank: int
    mAP50: float
    mAP50_95: float
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "rank": self.rank,
            "mAP50": None if math.isnan(self.mAP50) else round(self.mAP50, 4),
            "mAP50_95": None if math.isnan(self.mAP50_95) else round(self.mAP50_95, 4),
            "precision": None if math.isnan(self.precision) else round(self.precision, 4),
            "recall": None if math.isnan(self.recall) else round(self.recall, 4),
            "f1": None if math.isnan(self.f1) else round(self.f1, 4),
        }


def rank_classes(metrics: EvalMetrics, *,
                 sort_by: str = "mAP50_95",
                 top_n: Optional[int] = None,
                 worst: bool = False) -> List[ClassRanking]:
    """按指定指标对类别做性能排序.

    Args:
        metrics:  评估指标
        sort_by:  排序指标 ("mAP50_95", "mAP50", "f1", "precision", "recall")
        top_n:    返回前 N 个类别, None 返回全部
        worst:    True 升序(最差在前), False 降序(最好在前)

    Returns:
        ClassRanking 列表, 已排序.
    """
    if not metrics.per_class:
        return []

    items = []
    for cname, vals in metrics.per_class.items():
        v = vals.get(sort_by, math.nan)
        items.append(ClassRanking(
            class_name=cname,
            rank=0,
            mAP50=vals.get("mAP50", math.nan),
            mAP50_95=vals.get("mAP50_95", math.nan),
            precision=vals.get("precision", math.nan),
            recall=vals.get("recall", math.nan),
            f1=vals.get("f1", math.nan),
        ))

    # 排序: NaN 排到最后
    def _key(r: ClassRanking) -> Tuple[int, float]:
        v = getattr(r, sort_by, math.nan)
        return (1 if math.isnan(v) else 0, v if not math.isnan(v) else 0.0)

    items.sort(key=_key, reverse=not worst)

    for i, item in enumerate(items, 1):
        item.rank = i

    return items[:top_n] if top_n else items


def diagnose_problem_classes(metrics: EvalMetrics, *,
                              threshold: float = 0.3) -> Dict[str, List[str]]:
    """诊断问题类别: F1 低于阈值的算"需要关注".

    Returns:
        {"critical": [...], "warning": [...], "good": [...]}
        分别对应 F1 < threshold/2, F1 < threshold, F1 >= threshold
    """
    result: Dict[str, List[str]] = {"critical": [], "warning": [], "good": []}
    if not metrics.per_class:
        return result

    for cname, vals in metrics.per_class.items():
        f1 = vals.get("f1", math.nan)
        if math.isnan(f1):
            continue
        if f1 < threshold / 2:
            result["critical"].append(cname)
        elif f1 < threshold:
            result["warning"].append(cname)
        else:
            result["good"].append(cname)
    return result


# ============================================================
# 多模型类别级对比
# ============================================================

@dataclass
class PerClassDiff:
    """两个模型在同一类别上的指标差异."""
    class_name: str
    metric_name: str
    value_a: float
    value_b: float
    diff: float       # value_a - value_b
    winner: str       # "A", "B", 或 "tie"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "metric": self.metric_name,
            "model_a_value": None if math.isnan(self.value_a) else round(self.value_a, 4),
            "model_b_value": None if math.isnan(self.value_b) else round(self.value_b, 4),
            "diff": None if math.isnan(self.diff) else round(self.diff, 4),
            "winner": self.winner,
        }


def compare_per_class(metrics_a: EvalMetrics, metrics_b: EvalMetrics, *,
                       metric: str = "mAP50_95",
                       min_diff: float = 0.01) -> List[PerClassDiff]:
    """对比两个模型在每个类别上的表现差异.

    Args:
        metrics_a: 模型 A 的指标
        metrics_b: 模型 B 的指标
        metric:    对比指标
        min_diff:  差异阈值, 低于此值标记为 "tie"

    Returns:
        PerClassDiff 列表, 按 |diff| 降序排列(差异大的在前).
    """
    all_classes = set(metrics_a.per_class.keys()) | set(metrics_b.per_class.keys())
    diffs: List[PerClassDiff] = []

    for cname in sorted(all_classes):
        va = metrics_a.per_class.get(cname, {}).get(metric, math.nan)
        vb = metrics_b.per_class.get(cname, {}).get(metric, math.nan)

        if math.isnan(va) and math.isnan(vb):
            continue
        na = 0.0 if math.isnan(va) else va
        nb = 0.0 if math.isnan(vb) else vb
        d = va - vb if not (math.isnan(va) or math.isnan(vb)) else math.nan

        if math.isnan(va):
            winner = "B"
        elif math.isnan(vb):
            winner = "A"
        elif abs(d) < min_diff:
            winner = "tie"
        else:
            winner = "A" if d > 0 else "B"

        diffs.append(PerClassDiff(
            class_name=cname, metric_name=metric,
            value_a=va, value_b=vb, diff=d, winner=winner,
        ))

    diffs.sort(key=lambda x: abs(x.diff) if not math.isnan(x.diff) else 0.0, reverse=True)
    return diffs


# ============================================================
# 报告增强: 渲染分析结果为 Markdown
# ============================================================

def render_confusion_markdown(analysis: List[ConfusionAnalysis]) -> str:
    """将混淆矩阵分析渲染为 Markdown 表格."""
    if not analysis:
        return "_混淆矩阵数据不可用。_"

    lines: List[str] = [
        "## 混淆矩阵错误分析",
        "",
        "| 类别 | TP | FP | FN | Precision | Recall | F1 |",
        "|------|----|----|----|-----------|--------|----|",
    ]
    total_tp = total_fp = total_fn = 0
    for a in analysis:
        lines.append(
            f"| {a.class_name} | {a.tp} | {a.fp} | {a.fn} | "
            f"{_fmt(a.precision)} | {_fmt(a.recall)} | {_fmt(a.f1)} |"
        )
        total_tp += a.tp
        total_fp += a.fp
        total_fn += a.fn

    total_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else math.nan
    total_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else math.nan
    total_f1 = 2 * total_p * total_r / (total_p + total_r) if not (
        math.isnan(total_p) or math.isnan(total_r) or (total_p + total_r) <= 0
    ) else math.nan

    lines.append(
        f"| **总计** | {total_tp} | {total_fp} | {total_fn} | "
        f"{_fmt(total_p)} | {_fmt(total_r)} | {_fmt(total_f1)} |"
    )
    lines.append("")
    lines.append(f"- 误检 (FP) 最多类别: {_find_most(analysis, 'fp')}")
    lines.append(f"- 漏检 (FN) 最多类别: {_find_most(analysis, 'fn')}")
    lines.append("")
    return "\n".join(lines)


def render_ranking_markdown(rankings: List[ClassRanking],
                             title: str = "类别性能排序") -> str:
    """将类别排名渲染为 Markdown 表格."""
    if not rankings:
        return "_无可用的类别级指标。_"

    lines: List[str] = [
        f"## {title}",
        "",
        "| 排名 | 类别 | mAP50 | mAP50-95 | Precision | Recall | F1 |",
        "|------|------|-------|----------|-----------|--------|----|",
    ]
    for r in rankings:
        lines.append(
            f"| {r.rank} | {r.class_name} | {_fmt(r.mAP50)} | {_fmt(r.mAP50_95)} | "
            f"{_fmt(r.precision)} | {_fmt(r.recall)} | {_fmt(r.f1)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _find_most(analysis: List[ConfusionAnalysis], field: str) -> str:
    if not analysis:
        return "N/A"
    return max(analysis, key=lambda a: getattr(a, field, 0)).class_name


__all__ = [
    "ConfusionAnalysis",
    "ClassRanking",
    "PerClassDiff",
    "extract_confusion_analysis",
    "rank_classes",
    "diagnose_problem_classes",
    "compare_per_class",
    "render_confusion_markdown",
    "render_ranking_markdown",
]
