#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : history.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_eval/history.py
# @Project    : XJTU-ODPlatfrom
# @Author     : ODPlatform Team
# @Function   : 评估历史追踪 —— 保存、加载、对比历史评估记录
"""模型评估历史追踪.

核心价值:
  · 追踪同一个模型在迭代训练中的评估指标变化
  · 对比不同模型在同一数据集上的历史最佳记录
  · 生成"指标趋势"报告, 直观展示模型改进方向

数据结构:
  EvalRecord   — 单条评估记录(可序列化为 JSON)
  EvalHistory  — 某数据集的评估历史集合
  TrendReport  — 指标趋势报告(同一模型多次评估的变化曲线数据)

使用方式:
  # 保存本次评估
  from od_platform.model_eval.history import EvalHistory
  history = EvalHistory.load_or_create(data_yaml, data_name)
  history.add_record(report)
  history.save()

  # 查看趋势
  trend = history.trend("my_model")
  print(trend.render_markdown())
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from od_platform.common.paths import RUNS_DIR
from od_platform.model_eval.metrics import EvalMetrics
from od_platform.model_eval.report import EvalReport, _fmt

logger = logging.getLogger(__name__)

_HISTORY_DIR: Path = RUNS_DIR / "evaluation" / "_history"


# ============================================================
# 评估记录
# ============================================================

@dataclass
class EvalRecord:
    """一条评估记录(可序列化的轻量快照)."""
    run_id: str
    model_name: str
    model_path: str
    data_yaml: str
    split: str
    task: str
    created_at: str
    # 核心指标
    precision: float
    recall: float
    mAP50: float
    mAP50_95: float
    f1: float
    accuracy: float
    fitness: float
    speed_total_ms: float

    @classmethod
    def from_report(cls, report: EvalReport) -> "EvalRecord":
        m = report.metrics
        return cls(
            run_id=report.run_id,
            model_name=report.model_name,
            model_path=report.model_path,
            data_yaml=report.data_yaml,
            split=report.split,
            task=m.task,
            created_at=report.created_at,
            precision=m.precision,
            recall=m.recall,
            mAP50=m.mAP50,
            mAP50_95=m.mAP50_95,
            f1=m.f1,
            accuracy=m.accuracy,
            fitness=m.fitness,
            speed_total_ms=m.speed_ms.get("total", math.nan),
        )

    def to_dict(self) -> Dict[str, Any]:
        def _s(v: float) -> Optional[float]:
            return None if (isinstance(v, float) and math.isnan(v)) else v
        return {
            "run_id": self.run_id, "model_name": self.model_name,
            "model_path": self.model_path, "data_yaml": self.data_yaml,
            "split": self.split, "task": self.task, "created_at": self.created_at,
            "precision": _s(self.precision), "recall": _s(self.recall),
            "mAP50": _s(self.mAP50), "mAP50_95": _s(self.mAP50_95),
            "f1": _s(self.f1), "accuracy": _s(self.accuracy),
            "fitness": _s(self.fitness), "speed_total_ms": _s(self.speed_total_ms),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalRecord":
        def _f(k: str) -> float:
            v = d.get(k)
            return float(v) if v is not None else math.nan
        return cls(
            run_id=d["run_id"], model_name=d["model_name"],
            model_path=d.get("model_path", ""), data_yaml=d.get("data_yaml", ""),
            split=d.get("split", "val"), task=d.get("task", "detect"),
            created_at=d.get("created_at", ""),
            precision=_f("precision"), recall=_f("recall"),
            mAP50=_f("mAP50"), mAP50_95=_f("mAP50_95"),
            f1=_f("f1"), accuracy=_f("accuracy"), fitness=_f("fitness"),
            speed_total_ms=_f("speed_total_ms"),
        )


# ============================================================
# 评估历史
# ============================================================

@dataclass
class EvalHistory:
    """某个数据集下的评估历史(纯数据, 可序列化)."""
    data_yaml: str
    records: List[EvalRecord] = field(default_factory=list)
    _file_path: Optional[Path] = None

    # ---- 记录管理 ----

    def add_record(self, report: EvalReport) -> None:
        """添加一条评估记录(重复 run_id 会跳过)."""
        record = EvalRecord.from_report(report)
        if any(r.run_id == record.run_id for r in self.records):
            logger.warning("run_id=%s 已存在于历史中, 跳过", record.run_id)
            return
        self.records.append(record)
        logger.info("评估历史已记录: %s | run_id=%s", record.model_name, record.run_id)

    def get_records(self, model_name: Optional[str] = None) -> List[EvalRecord]:
        """按模型名过滤历史记录."""
        if model_name is None:
            return list(self.records)
        return [r for r in self.records if r.model_name == model_name]

    def best_record(self, model_name: Optional[str] = None,
                     metric: str = "mAP50_95") -> Optional[EvalRecord]:
        """返回指定模型在某指标上的最佳记录(越高越好)."""
        records = self.get_records(model_name)
        if not records:
            return None
        valid = [(r, getattr(r, metric, math.nan)) for r in records]
        valid = [(r, v) for r, v in valid if not math.isnan(v)]
        if not valid:
            return None
        return max(valid, key=lambda x: x[1])[0]

    # ---- 趋势分析 ----

    def trend(self, model_name: str) -> TrendReport:
        """生成某个模型的指标趋势报告(按时间排序)."""
        records = sorted(
            [r for r in self.records if r.model_name == model_name],
            key=lambda r: r.created_at,
        )
        return TrendReport(model_name=model_name, records=records)

    def compare_best(self, model_names: List[str],
                      metric: str = "mAP50_95") -> Dict[str, Optional[EvalRecord]]:
        """对比多个模型的历史最佳记录."""
        return {name: self.best_record(name, metric) for name in model_names}

    # ---- 序列化 ----

    @staticmethod
    def _history_path(data_yaml: str) -> Path:
        """历史文件路径: runs/evaluation/_history/<dataset_name>.json"""
        stem = Path(data_yaml).stem
        new_path = _HISTORY_DIR / f"{stem}_eval_history.json"
        # 向后兼容: 如果旧位置有历史文件而新位置还没有, 自动迁移
        old_path = RUNS_DIR / "model_evaluation" / "_history" / f"{stem}_eval_history.json"
        if old_path.exists() and not new_path.exists():
            return old_path
        return new_path

    def save(self, file_path: Optional[Path] = None) -> Path:
        """保存为 JSON."""
        fp = file_path or self._file_path or self._history_path(self.data_yaml)
        fp.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "data_yaml": self.data_yaml,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "records": [r.to_dict() for r in self.records],
        }
        fp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("评估历史已保存到 %s (%d 条记录)", fp, len(self.records))
        return fp

    @classmethod
    def load(cls, file_path: Path) -> "EvalHistory":
        """从 JSON 加载历史."""
        data = json.loads(file_path.read_text(encoding="utf-8"))
        history = cls(data_yaml=data["data_yaml"], _file_path=file_path)
        history.records = [EvalRecord.from_dict(r) for r in data.get("records", [])]
        return history

    @classmethod
    def load_or_create(cls, data_yaml: str) -> "EvalHistory":
        """加载已有历史或创建新历史."""
        fp = cls._history_path(data_yaml)
        if fp.exists():
            return cls.load(fp)
        return cls(data_yaml=data_yaml, _file_path=fp)


# ============================================================
# 趋势报告
# ============================================================

_METRIC_LABELS: Dict[str, Tuple[str, bool]] = {
    "precision":      ("Precision",      True),
    "recall":         ("Recall",         True),
    "mAP50":          ("mAP50",          True),
    "mAP50_95":       ("mAP50-95",       True),
    "f1":             ("F1",             True),
    "accuracy":       ("Accuracy",       True),
    "fitness":        ("Fitness",        True),
    "speed_total_ms": ("推理速度(ms)",   False),
}


@dataclass
class TrendReport:
    """某个模型多次评估的趋势(纯数据)."""
    model_name: str
    records: List[EvalRecord] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return len(self.records) == 0

    def _trend_values(self, key: str) -> List[Tuple[str, float]]:
        return [(
            r.run_id[:15] if len(r.run_id) > 15 else r.run_id,
            getattr(r, key, math.nan),
        ) for r in self.records]

    def improvement(self, metric: str = "mAP50_95") -> Optional[float]:
        """首个到最后一个记录的指标变化."""
        if len(self.records) < 2:
            return None
        first = getattr(self.records[0], metric, math.nan)
        last = getattr(self.records[-1], metric, math.nan)
        if math.isnan(first) or math.isnan(last):
            return None
        return last - first

    def render_markdown(self) -> str:
        """生成趋势 Markdown 报告."""
        if self.empty:
            return f"_模型 `{self.model_name}` 暂无评估历史。_"

        lines: List[str] = [
            f"## 评估趋势 — `{self.model_name}`",
            "",
            f"共 {len(self.records)} 次评估记录",
            "",
            "| # | 时间 | mAP50 | mAP50-95 | F1 | Precision | Recall | 推理(ms) |",
            "|---|------|-------|----------|----|-----------|--------|----------|",
        ]
        for i, r in enumerate(self.records, 1):
            lines.append(
                f"| {i} | {r.created_at[:16]} | {_fmt(r.mAP50)} | {_fmt(r.mAP50_95)} | "
                f"{_fmt(r.f1)} | {_fmt(r.precision)} | {_fmt(r.recall)} | "
                f"{_fmt(r.speed_total_ms)} |"
            )
        lines.append("")

        # 改进/退步
        imp = self.improvement("mAP50_95")
        if imp is not None:
            direction = "↑ 提升" if imp > 0 else ("↓ 下降" if imp < 0 else "→ 持平")
            lines.append(f"- **mAP50-95 变化**: {_fmt(imp)}  {direction}")
        imp_f1 = self.improvement("f1")
        if imp_f1 is not None:
            direction = "↑ 提升" if imp_f1 > 0 else ("↓ 下降" if imp_f1 < 0 else "→ 持平")
            lines.append(f"- **F1 变化**: {_fmt(imp_f1)}  {direction}")
        lines.append("")
        return "\n".join(lines)


# ============================================================
# 公开 API
# ============================================================

__all__ = [
    "EvalRecord",
    "EvalHistory",
    "TrendReport",
]
