#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : metrics.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_eval/metrics.py
# @Project    : XJTU-ODPlatfrom
# @Author     : ODPlatform Team
# @Function   : 模型评估指标 —— 从 YOLO val 结果抽取 + 派生 F1 / accuracy
"""模型评估指标数据结构.

设计点(与 model_train/result.py 同款纪律):
  · EvalMetrics 是纯数据(frozen), 给报告 / 审计 / 日志共用, 不持有展示逻辑
  · 复用 TrainMetrics.from_yolo_results 抽取 ultralytics 原生指标
    (作者本人在 result.py 注释里就写明 "ValService 可直接共用 TrainMetrics")
  · 在原生指标上派生两个评估报告必需要、但 ultralytics 不直接给的量:
        F1        = 2*P*R/(P+R)
        accuracy  = 混淆矩阵对角线 / 总数 (含 background 行列, 标准 CM 准确率)
  · 每类指标(per-class): precision / recall / f1 / mAP50 / mAP50-95
  · 所有数值经 _safe_float 兜底: ultralytics 返回结构跨版本会变, 缺键是常态
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from od_platform.common.constants import Task
from od_platform.model_train.result import TrainMetrics, _safe_float

logger = logging.getLogger(__name__)

# 评估报告核心指标(数据驱动: 加新任务只加一行), 跟 result.py 的 _METRIC_FIELDS_BY_TASK 对齐
_EVAL_OVERALL_KEYS: Dict[str, str] = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)":    "recall",
    "metrics/mAP50(B)":     "mAP50",
    "metrics/mAP50-95(B)":  "mAP50_95",
}


def _f1(precision: float, recall: float) -> float:
    """由 P / R 派生 F1, 任一为 NaN 或两者同 0 返回 NaN."""
    if math.isnan(precision) or math.isnan(recall):
        return math.nan
    denom = precision + recall
    if denom <= 0:
        return math.nan
    return 2.0 * precision * recall / denom


def _accuracy_from_confusion_matrix(results: Any) -> float:
    """从 ultralytics val 结果的混淆矩阵算 accuracy = trace / total.

    val(plots=True) 时 results.confusion_matrix.matrix 是 (nc+1)x(nc+1),
    最后一行/列是 background. 取全部元素的总和与对角线之和算标准 CM 准确率.
    任一环节缺位都兜底 NaN, 不让报告崩.
    """
    cm = getattr(results, "confusion_matrix", None)
    matrix = getattr(cm, "matrix", None) if cm is not None else None
    if matrix is None:
        return math.nan
    try:
        total = float(matrix.sum())
        if total <= 0:
            return math.nan
        diag = float(matrix.diagonal().sum())
        return diag / total
    except Exception as e:  # numpy / 鸭子类型都可能抛, 兜底
        logger.warning("混淆矩阵 accuracy 计算失败, 已兜底为 NaN: %s", e)
        return math.nan


def _per_class_from_box(results: Any, names: Dict[Any, str]) -> Dict[str, Dict[str, float]]:
    """从 results.box 抽每类 P/R/AP50/AP50-95(val 模式), 并派生每类 F1.

    ultralytics val 结果的 box 是 DetMetrics, 持有 .p / .r / .ap50 / .ap 数组.
    跨版本缺位则该类对应字段为 NaN, 不影响整体报告.
    """
    box = getattr(results, "box", None)
    if box is None or not names:
        return {}

    def _arr(name: str) -> Any:
        a = getattr(box, name, None)
        return a if a is not None and hasattr(a, "__len__") else None

    p_arr, r_arr = _arr("p"), _arr("r")
    ap50_arr, ap_arr = _arr("ap50"), _arr("ap")

    out: Dict[str, Dict[str, float]] = {}
    for idx, class_name in names.items():
        i = int(idx)
        p = _safe_float(p_arr[i]) if p_arr is not None and i < len(p_arr) else math.nan
        r = _safe_float(r_arr[i]) if r_arr is not None and i < len(r_arr) else math.nan
        ap50 = _safe_float(ap50_arr[i]) if ap50_arr is not None and i < len(ap50_arr) else math.nan
        ap = _safe_float(ap_arr[i]) if ap_arr is not None and i < len(ap_arr) else math.nan
        out[str(class_name)] = {
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
            "mAP50": ap50,
            "mAP50_95": ap,
        }
    return out


@dataclass(frozen=True)
class EvalMetrics:
    """一次模型评估的结构化快照(纯数据, frozen).

    既成事实不该被改 —— 跟 TrainMetrics 同纪律.
    run_id / model_name 由调用方(service / CLI)注入, 不在本类里 now().
    """
    run_id:        str
    model_name:    str
    model_path:    str
    task:          str
    split:         str
    # 整体核心指标
    precision:     float
    recall:        float
    mAP50:         float
    mAP50_95:      float
    f1:            float
    accuracy:      float
    fitness:       float
    speed_ms:      Dict[str, float] = field(default_factory=dict)
    raw_overall:   Dict[str, float] = field(default_factory=dict)  # 完整 results_dict, 审计用
    per_class:     Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_yolo_results(cls, results: Any, *, run_id: str, model_name: str,
                          model_path: str, split: str) -> "EvalMetrics":
        """从 ultralytics val 结果构造评估指标.

        ★ run_id / model_name 必填 keyword-only —— 同 result.py 纪律:
          这个方法在【评估结束时】被调用, 而 run_id 是【评估开始前】定的,
          在这里 now() 会出现两个时间, "结果属于哪次"只能靠猜.
        """
        # 复用 TrainMetrics 抽取原生指标(speed / overall results_dict / 每类 mAP)
        base = TrainMetrics.from_yolo_results(results, run_id=run_id)
        overall = dict(base.overall)

        precision = overall.get("metrics/precision(B)", math.nan)
        recall = overall.get("metrics/recall(B)", math.nan)
        mAP50 = overall.get("metrics/mAP50(B)", math.nan)
        mAP50_95 = overall.get("metrics/mAP50-95(B)", math.nan)

        f1 = _f1(precision, recall)
        accuracy = _accuracy_from_confusion_matrix(results)

        # 每类指标: val 模式优先 results.box, 兜底用 TrainMetrics 已抽的每类 mAP
        names = getattr(results, "names", {}) or {}
        per_class = _per_class_from_box(results, names)
        if not per_class and base.class_map_50_95:
            # val.box 不可用时, 至少把每类 mAP@50:95 摆出来(来自 maps)
            per_class = {
                cname: {"mAP50_95": v} for cname, v in base.class_map_50_95.items()
            }

        return cls(
            run_id=run_id,
            model_name=model_name,
            model_path=model_path,
            task=getattr(results, "task", Task.DETECT),
            split=split,
            precision=precision,
            recall=recall,
            mAP50=mAP50,
            mAP50_95=mAP50_95,
            f1=f1,
            accuracy=accuracy,
            fitness=overall.get("fitness", math.nan),
            speed_ms=dict(base.speed_ms),
            raw_overall=overall,
            per_class=per_class,
        )

    def to_dict(self) -> Dict[str, Any]:
        """转 dict: 路径转字符串, NaN 转 None. 可直接 json.dumps."""
        def _clean(d: Dict[str, float]) -> Dict[str, Any]:
            return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in d.items()}

        def _clean_pc(d: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
            return {k: _clean(v) for k, v in d.items()}

        def _scalar(v: float) -> Optional[float]:
            return None if (isinstance(v, float) and math.isnan(v)) else v

        return {
            "run_id":      self.run_id,
            "model_name":  self.model_name,
            "model_path":  self.model_path,
            "task":        self.task,
            "split":       self.split,
            "precision":   _scalar(self.precision),
            "recall":      _scalar(self.recall),
            "mAP50":       _scalar(self.mAP50),
            "mAP50_95":    _scalar(self.mAP50_95),
            "f1":          _scalar(self.f1),
            "accuracy":    _scalar(self.accuracy),
            "fitness":     _scalar(self.fitness),
            "speed_ms":    _clean(self.speed_ms),
            "raw_overall": _clean(self.raw_overall),
            "per_class":   _clean_pc(self.per_class),
        }


__all__ = ["EvalMetrics"]
