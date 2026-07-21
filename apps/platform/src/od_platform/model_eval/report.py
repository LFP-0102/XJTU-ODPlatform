#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : report.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_eval/report.py
# @Project    : XJTU-ODPlatfrom
# @Author     : ODPlatform Team
# @Function   : 评估报告 —— 单模型报告 + 多模型对比报告
"""模型评估报告.

纪律(与 data_validation/report.py 同款):
  · EvalReport / ComparisonReport 是纯数据(可落 report.json / result.csv)
  · 渲染(render_to_logger / render_markdown)是另一回事, 只消费数据不持有数据
  · 字段 additive: 只增不改不删, 下游才敢长期依赖

双轨输出(沿用项目惯例):
  · 机器轨道: report.json  —— 结构化完整结果
  · 人工轨道: result.csv   —— 每行一个指标, 拖进表格即可排序
  · 可读报告: report.md     —— 数据摘要 + 指标对比表 + 差异分析(给人看的)
"""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from od_platform.common.string_utils import (
    get_display_width, pad_to_width, format_table_row, format_table_separator,
)
from od_platform.model_eval.metrics import EvalMetrics

logger = logging.getLogger(__name__)

_WIDTH = 64

# 评估核心指标定义: (内部字段, 展示名, 格式, 是否越高越好)
# 格式化时统一保留 4 位小数, speed 单独走毫秒格式
_OVERALL_METRIC_DEFS: List[Tuple[str, str, bool]] = [
    ("precision",  "Precision",  True),
    ("recall",     "Recall",     True),
    ("f1",         "F1",         True),
    ("accuracy",   "Accuracy",   True),
    ("mAP50",      "mAP50",      True),
    ("mAP50_95",   "mAP50-95",   True),
    ("fitness",    "Fitness",    True),
]

_SPEED_KEY = "total"  # speed_ms["total"], 越低越好


def _fmt(v: float) -> str:
    """NaN -> 'N/A', 否则 4 位小数."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def _fmt_ms(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.2f} ms"


# ============================================================
# 单模型评估报告
# ============================================================

@dataclass
class EvalReport:
    """一次单模型评估的完整结果(纯数据)."""
    run_id:     str
    model_name: str
    model_path: str
    data_yaml:  str
    split:      str
    created_at: str
    metrics:    EvalMetrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":     self.run_id,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "data_yaml":  self.data_yaml,
            "split":      self.split,
            "created_at":  self.created_at,
            "metrics":    self.metrics.to_dict(),
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    def write_csv(self, path: Path) -> None:
        """人工轨道: 每行一个指标, utf-8-sig 让 Excel 直接认中文."""
        m = self.metrics
        rows = [(disp, _fmt(getattr(m, key))) for key, disp, _ in _OVERALL_METRIC_DEFS]
        rows.append(("Inference(ms)", _fmt_ms(m.speed_ms.get(_SPEED_KEY, math.nan))))
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["run_id", "model", "metric", "value"])
            for disp, val in rows:
                w.writerow([self.run_id, self.model_name, disp, val])

    def render_to_logger(self, log: Optional[logging.Logger] = None) -> None:
        lg = log or logger
        m = self.metrics
        lg.info("=" * _WIDTH)
        lg.info("模型评估报告   run_id=%s   模型=%s", self.run_id, self.model_name)
        lg.info("数据集: %s   split=%s   task=%s", self.data_yaml, self.split, m.task)
        lg.info("-" * _WIDTH)
        for key, disp, _ in _OVERALL_METRIC_DEFS:
            lg.info("%s: %s", pad_to_width(disp, 20), _fmt(getattr(m, key)))
        lg.info("%s: %s", pad_to_width("推理速度", 20),
                _fmt_ms(m.speed_ms.get(_SPEED_KEY, math.nan)))
        if m.per_class:
            lg.info("-" * _WIDTH)
            lg.info("类别级指标".center(_WIDTH))
            for cname, vals in m.per_class.items():
                lg.info("  %s  P=%s R=%s F1=%s mAP50=%s",
                        pad_to_width(cname, 18),
                        _fmt(vals.get("precision", math.nan)),
                        _fmt(vals.get("recall", math.nan)),
                        _fmt(vals.get("f1", math.nan)),
                        _fmt(vals.get("mAP50", math.nan)))
        lg.info("=" * _WIDTH)

    def render_markdown(self) -> str:
        """生成单模型 Markdown 报告."""
        m = self.metrics
        lines: List[str] = []
        lines.append(f"# 模型评估报告 — {self.model_name}")
        lines.append("")
        lines.append("## 数据摘要")
        lines.append("")
        lines.append(f"- **运行 ID**: `{self.run_id}`")
        lines.append(f"- **模型名称**: {self.model_name}")
        lines.append(f"- **模型路径**: `{self.model_path}`")
        lines.append(f"- **数据集**: `{self.data_yaml}`")
        lines.append(f"- **数据划分**: {self.split}")
        lines.append(f"- **任务类型**: {m.task}")
        lines.append(f"- **生成时间**: {self.created_at}")
        lines.append("")

        lines.append("## 核心评估指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        for key, disp, _ in _OVERALL_METRIC_DEFS:
            lines.append(f"| {disp} | {_fmt(getattr(m, key))} |")
        lines.append(f"| 推理速度(ms) | {_fmt_ms(m.speed_ms.get(_SPEED_KEY, math.nan))} |")
        lines.append("")

        if m.per_class:
            lines.append("## 类别级指标")
            lines.append("")
            lines.append("| 类别 | Precision | Recall | F1 | mAP50 | mAP50-95 |")
            lines.append("|------|-----------|--------|----|-------|----------|")
            for cname, vals in m.per_class.items():
                lines.append(
                    f"| {cname} | {_fmt(vals.get('precision', math.nan))} | "
                    f"{_fmt(vals.get('recall', math.nan))} | "
                    f"{_fmt(vals.get('f1', math.nan))} | "
                    f"{_fmt(vals.get('mAP50', math.nan))} | "
                    f"{_fmt(vals.get('mAP50_95', math.nan))} |"
                )
            lines.append("")
        lines.append("")
        return "\n".join(lines)


# ============================================================
# 多模型对比报告
# ============================================================

@dataclass
class ComparisonReport:
    """多个模型在同一数据集上的对比结果(纯数据)."""
    run_id:      str
    data_yaml:   str
    split:       str
    created_at:  str
    models:      List[EvalMetrics] = field(default_factory=list)

    # ---- 派生分析 ----

    def _metric_values(self, key: str) -> List[Tuple[str, float]]:
        """[(model_name, value)] for a given metric key, NaN 用 -inf 占位便于比较."""
        out = []
        for m in self.models:
            v = getattr(m, key, math.nan)
            if isinstance(v, float) and math.isnan(v):
                v = float("-inf")
            out.append((m.model_name, v))
        return out

    def _speed_values(self) -> List[Tuple[str, float]]:
        out = []
        for m in self.models:
            v = m.speed_ms.get(_SPEED_KEY, math.nan)
            if isinstance(v, float) and math.isnan(v):
                v = float("inf")  # 速度越高越好 -> 缺失当作最差
            out.append((m.model_name, v))
        return out

    def best_model(self, key: str) -> Optional[str]:
        """单指标最优模型名(越高越好). 缺失值不参与."""
        vals = self._metric_values(key)
        valid = [(n, v) for n, v in vals if v != float("-inf")]
        if not valid:
            return None
        return max(valid, key=lambda kv: kv[1])[0]

    def fastest_model(self) -> Optional[str]:
        """推理速度最优(最低 ms)的模型."""
        vals = self._speed_values()
        valid = [(n, v) for n, v in vals if v != float("inf")]
        if not valid:
            return None
        return min(valid, key=lambda kv: kv[1])[0]

    def overall_winner(self) -> Optional[str]:
        """按"指标胜出数"加权(不含 speed)决定总赢家, 平票取 fitness 高者."""
        if not self.models:
            return None
        scores: Dict[str, int] = {m.model_name: 0 for m in self.models}
        for key, _, higher_better in _OVERALL_METRIC_DEFS:
            if not higher_better:
                continue
            best = self.best_model(key)
            if best:
                scores[best] = scores.get(best, 0) + 1
        top = max(scores.values()) if scores else 0
        winners = [n for n, s in scores.items() if s == top]
        if len(winners) == 1:
            return winners[0]
        # 平票 -> fitness 高者胜
        fit = {m.model_name: m.fitness for m in self.models if m.model_name in winners}
        fit = {n: (v if not math.isnan(v) else float("-inf")) for n, v in fit.items()}
        return max(fit, key=fit.get) if fit else winners[0]

    # ---- 序列化 ----

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id":     self.run_id,
            "data_yaml":  self.data_yaml,
            "split":      self.split,
            "created_at": self.created_at,
            "models":     [m.to_dict() for m in self.models],
            "analysis": {
                "overall_winner":  self.overall_winner(),
                "fastest_model":   self.fastest_model(),
                "best_per_metric": {
                    key: self.best_model(key) for key, _, _ in _OVERALL_METRIC_DEFS
                },
            },
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")

    def write_csv(self, path: Path) -> None:
        """对比表: 每行一个模型, 每列一个指标. utf-8-sig."""
        headers = ["model"] + [disp for _, disp, _ in _OVERALL_METRIC_DEFS] + ["speed_ms"]
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(headers)
            for m in self.models:
                row = [m.model_name]
                for key, _, _ in _OVERALL_METRIC_DEFS:
                    row.append(_fmt(getattr(m, key)))
                row.append(_fmt_ms(m.speed_ms.get(_SPEED_KEY, math.nan)))
                w.writerow(row)

    def render_to_logger(self, log: Optional[logging.Logger] = None) -> None:
        lg = log or logger
        lg.info("=" * _WIDTH)
        lg.info("多模型对比评估   run_id=%s   模型数=%d", self.run_id, len(self.models))
        lg.info("数据集: %s   split=%s", self.data_yaml, self.split)
        lg.info("-" * _WIDTH)

        # 对齐表头
        name_w = max([get_display_width(m.model_name) for m in self.models] + [6])
        col_w = 10
        header = format_table_row(
            ["模型"] + [disp for _, disp, _ in _OVERALL_METRIC_DEFS] + ["speed(ms)"],
            [name_w] + [col_w] * (len(_OVERALL_METRIC_DEFS) + 1),
        )
        lg.info(header)
        lg.info(format_table_separator([name_w] + [col_w] * (len(_OVERALL_METRIC_DEFS) + 1)))

        winner = self.overall_winner()
        for m in self.models:
            cells = [_fmt(getattr(m, key)) for key, _, _ in _OVERALL_METRIC_DEFS]
            cells.append(_fmt_ms(m.speed_ms.get(_SPEED_KEY, math.nan)))
            mark = " ★" if m.model_name == winner else ""
            row = format_table_row(
                [m.model_name + mark] + cells,
                [name_w] + [col_w] * (len(_OVERALL_METRIC_DEFS) + 1),
            )
            lg.info(row)

        lg.info("-" * _WIDTH)
        lg.info("★ 总体最优: %s", winner or "无(指标缺失)")
        lg.info("最快模型: %s", self.fastest_model() or "无")
        for key, disp, _ in _OVERALL_METRIC_DEFS:
            lg.info("  %-10s 最佳: %s", disp, self.best_model(key) or "无")
        lg.info("=" * _WIDTH)

    def render_markdown(self) -> str:
        """生成对比 Markdown 报告: 摘要 + 对比表(高亮最优) + 差异分析."""
        lines: List[str] = []
        lines.append(f"# 多模型对比评估报告")
        lines.append("")
        lines.append("## 数据摘要")
        lines.append("")
        lines.append(f"- **运行 ID**: `{self.run_id}`")
        lines.append(f"- **数据集**: `{self.data_yaml}`")
        lines.append(f"- **数据划分**: {self.split}")
        lines.append(f"- **参与模型数**: {len(self.models)}")
        lines.append(f"- **生成时间**: {self.created_at}")
        lines.append(f"- **对比模型**: {', '.join(m.model_name for m in self.models)}")
        lines.append("")

        winner = self.overall_winner()
        fastest = self.fastest_model()

        lines.append("## 指标对比表")
        lines.append("")
        lines.append("> ★ 标记该指标最优模型; 总体最优以粗体标出。")
        lines.append("")
        header = "| 模型 |"
        sep = "|------|"
        for _, disp, _ in _OVERALL_METRIC_DEFS:
            header += f" {disp} |"
            sep += "------|"
        header += " 推理速度(ms) |"
        sep += "------|"
        lines.append(header)
        lines.append(sep)

        best_map = {key: self.best_model(key) for key, _, _ in _OVERALL_METRIC_DEFS}
        fastest_name = fastest
        for m in self.models:
            is_winner = (m.model_name == winner)
            name = f"**{m.model_name}**" if is_winner else m.model_name
            row = f"| {name} |"
            for key, _, _ in _OVERALL_METRIC_DEFS:
                cell = _fmt(getattr(m, key))
                if best_map.get(key) == m.model_name:
                    cell = f"★ {cell}"
                row += f" {cell} |"
            speed_cell = _fmt_ms(m.speed_ms.get(_SPEED_KEY, math.nan))
            if fastest_name == m.model_name:
                speed_cell = f"★ {speed_cell}"
            row += f" {speed_cell} |"
            lines.append(row)
        lines.append("")

        # 差异分析
        lines.append("## 差异分析")
        lines.append("")
        if winner:
            lines.append(f"- **总体最优模型**: `{winner}` (按各指标胜出数加权, 平票取 Fitness 高者)")
        if fastest:
            lines.append(f"- **推理最快模型**: `{fastest}` ({_fmt_ms(min((m.speed_ms.get(_SPEED_KEY, math.inf) for m in self.models), default=math.inf))})")
        lines.append("")
        lines.append("### 各指标最优模型")
        lines.append("")
        lines.append("| 指标 | 最优模型 |")
        lines.append("|------|----------|")
        for key, disp, _ in _OVERALL_METRIC_DEFS:
            lines.append(f"| {disp} | {self.best_model(key) or 'N/A'} |")
        lines.append("")

        # 极差分析: 每个指标 max-min
        lines.append("### 指标极差(max − min)")
        lines.append("")
        lines.append("| 指标 | 最高 | 最低 | 极差 |")
        lines.append("|------|------|------|------|")
        for key, disp, _ in _OVERALL_METRIC_DEFS:
            vals = self._metric_values(key)
            valid = [(n, v) for n, v in vals if v != float("-inf")]
            if len(valid) < 2:
                lines.append(f"| {disp} | — | — | — |")
                continue
            hi = max(valid, key=lambda kv: kv[1])
            lo = min(valid, key=lambda kv: kv[1])
            diff = hi[1] - lo[1]
            lines.append(f"| {disp} | {hi[0]} ({_fmt(hi[1])}) | {lo[0]} ({_fmt(lo[1])}) | {_fmt(diff)} |")
        lines.append("")

        # 结论
        lines.append("## 结论")
        lines.append("")
        if not self.models:
            lines.append("无可用模型结果。")
        elif winner:
            w = next((mm for mm in self.models if mm.model_name == winner), None)
            if w is not None:
                lines.append(f"`{winner}` 综合表现最优:")
                lines.append(f"- Precision = {_fmt(w.precision)}, Recall = {_fmt(w.recall)}, "
                             f"F1 = {_fmt(w.f1)}, mAP50 = {_fmt(w.mAP50)}, mAP50-95 = {_fmt(w.mAP50_95)}")
                if fastest and fastest != winner:
                    lines.append(f"- 若更看重推理速度, `{fastest}` 更合适 (推理 {_fmt_ms(next(m.speed_ms.get(_SPEED_KEY, math.inf) for m in self.models if m.model_name == fastest))})。")
                lines.append("")
                lines.append("> 注: mAP50-95 是最严格的综合指标, 优先以它衡量检测质量; "
                             "F1 平衡 Precision 与 Recall; Accuracy 来自混淆矩阵对角线占比。")
        lines.append("")
        return "\n".join(lines)


__all__ = ["EvalReport", "ComparisonReport"]
