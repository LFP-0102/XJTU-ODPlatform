#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练分析报告——数据模型 + Markdown 渲染.

设计纪律(与 model_eval/report.py 同款):
  · TrainingReport 是纯数据
  · render_markdown() 只消费数据不持有数据
  · to_dict() 用于 JSON 序列化(NaN -> null)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from od_platform.train_report.readers import EpochRow


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.4f}"


def _fmt_ms(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.2f} ms"


# ============================================================
# TrainingReport dataclass
# ============================================================

@dataclass
class TrainingReport:
    """训练综合分析报告(纯数据).

    章节:
      一、数据集概况
      二、训练配置
      三、检测结果
      四、问题分析
      五、改进建议
    """
    # 元数据
    run_id:      str = ""
    created_at:  str = ""

    # ---- 一、数据集 ----
    dataset_name:        str = ""
    dataset_yaml:        str = ""
    num_classes:         int = 0
    class_names:         List[str] = field(default_factory=list)
    split_strategy:      Optional[str] = None
    split_seed:          Optional[int] = None
    split_rates:         Optional[Dict[str, float]] = None
    split_counts:        Dict[str, int] = field(default_factory=dict)
    box_counts:          Dict[str, int] = field(default_factory=dict)
    class_distribution:  Dict[str, Dict[str, int]] = field(default_factory=dict)
    bbox_size_distribution: Dict[str, Dict[str, int]] = field(default_factory=dict)
    js_divergence:       Optional[Dict[str, Any]] = None
    image_size_stats:    Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ---- 二、训练配置 ----
    task:            str = ""
    model_name:      str = ""
    epochs_planned:  int = 0
    imgsz:           int = 640
    batch:           Union[int, float] = 16
    optimizer:       str = "auto"
    lr0:             float = 0.01
    lrf:             float = 0.01
    pretrained:      Union[bool, str] = True
    close_mosaic:    int = 10
    seed:            int = 0
    config_snapshot: Dict[str, Any] = field(default_factory=dict)

    # ---- 三、检测指标 ----
    mAP50:          float = math.nan
    mAP50_95:       float = math.nan
    precision:      float = math.nan
    recall:         float = math.nan
    fitness:        float = math.nan
    speed_ms:       Dict[str, float] = field(default_factory=dict)
    class_map_50_95: Dict[str, float] = field(default_factory=dict)

    # ---- 三(b)、逐轮过程 ----
    epoch_history:  List[EpochRow] = field(default_factory=list)
    epochs_actual:  int = 0

    # ---- 四、问题分析 ----
    analysis_warnings: List[str] = field(default_factory=list)
    convergence_note:  str = ""

    # ---- 五、改进建议 ----
    improvement_suggestions: List[str] = field(default_factory=list)

    # ================================================================
    # 序列化
    # ================================================================

    def to_dict(self) -> Dict[str, Any]:
        def _safe(v: Optional[float]) -> Any:
            return None if (isinstance(v, float) and math.isnan(v)) else v

        return {
            "run_id":              self.run_id,
            "created_at":          self.created_at,
            "dataset_name":        self.dataset_name,
            "num_classes":         self.num_classes,
            "class_names":         self.class_names,
            "split_strategy":      self.split_strategy,
            "split_seed":          self.split_seed,
            "split_counts":        self.split_counts,
            "class_distribution":  self.class_distribution,
            "bbox_size_distribution": self.bbox_size_distribution,
            "task":                self.task,
            "model_name":          self.model_name,
            "epochs_planned":      self.epochs_planned,
            "imgsz":               self.imgsz,
            "batch":               self.batch,
            "optimizer":           self.optimizer,
            "lr0":                 self.lr0,
            "lrf":                 self.lrf,
            "mAP50":               _safe(self.mAP50),
            "mAP50_95":            _safe(self.mAP50_95),
            "precision":           _safe(self.precision),
            "recall":              _safe(self.recall),
            "fitness":             _safe(self.fitness),
            "speed_ms":            self.speed_ms,
            "class_map_50_95":     self.class_map_50_95,
            "epochs_actual":       self.epochs_actual,
            "analysis_warnings":   self.analysis_warnings,
            "convergence_note":    self.convergence_note,
            "improvement_suggestions": self.improvement_suggestions,
        }

    # ================================================================
    # Markdown 渲染
    # ================================================================

    def render_markdown(self) -> str:
        lines: List[str] = []
        self._header(lines)
        self._dataset_section(lines)
        self._config_section(lines)
        self._results_section(lines)
        self._epoch_section(lines)
        self._analysis_section(lines)
        self._suggestions_section(lines)
        return "\n".join(lines)

    def _header(self, lines: List[str]) -> None:
        name = self.dataset_name or "(未命名)"
        lines.append(f"# 训练分析报告 · {name}")
        lines.append("")
        lines.append(f"- **运行 ID**: `{self.run_id}`")
        lines.append(f"- **生成时间**: {self.created_at}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ----------------------------------------------------------------
    # 一、数据集概况
    # ----------------------------------------------------------------

    def _dataset_section(self, lines: List[str]) -> None:
        lines.append("## 一、数据集概况")
        lines.append("")

        lines.append("### 1.1 基本信息")
        lines.append("")
        lines.append(f"| 项目 | 内容 |")
        lines.append(f"|------|------|")
        lines.append(f"| 数据集名称 | {self.dataset_name or '(未知)'} |")
        lines.append(f"| 配置文件 | `{self.dataset_yaml}` |")
        lines.append(f"| 类别数 | {self.num_classes} |")
        names = ", ".join(self.class_names) if self.class_names else "(未知)"
        lines.append(f"| 类别列表 | {names} |")

        if self.split_counts:
            total = sum(self.split_counts.values())
            parts = ", ".join(f"{s}={n}" for s, n in self.split_counts.items())
            lines.append(f"| 样本量 | {total} ({parts}) |")
        if self.box_counts:
            parts = ", ".join(f"{s}={n}" for s, n in self.box_counts.items())
            lines.append(f"| 标注框总数 | {sum(self.box_counts.values())} ({parts}) |")
        if self.split_strategy:
            rates = self.split_rates or {}
            rates_str = " / ".join(
                f"{k}: {v}" for k, v in rates.items()
            ) if rates else ""
            lines.append(f"| 划分方式 | {self.split_strategy}, seed={self.split_seed} ({rates_str}) |")
        lines.append("")

        # 1.2 类别分布
        if self.class_distribution and self.class_names:
            lines.append("### 1.2 类别分布")
            lines.append("")
            splits = list(self.class_distribution.keys())
            header = "| 类别 | " + " | ".join(splits) + " | 总计 |"
            sep = "|------|" + "|".join(["------|"] * (len(splits) + 1))
            lines.append(header)
            lines.append(sep)
            for c in self.class_names:
                counts = [str(self.class_distribution.get(sp, {}).get(c, 0)) for sp in splits]
                total_c = sum(self.class_distribution.get(sp, {}).get(c, 0) for sp in splits)
                lines.append("| " + c + " | " + " | ".join(counts) + f" | {total_c} |")
            lines.append("")

        # 1.3 bbox 尺寸
        if self.bbox_size_distribution:
            lines.append("### 1.3 标注框尺寸分布 (COCO 面积分桶)")
            lines.append("")
            lines.append("| split | 小 (<32^2) | 中 (32-96^2) | 大 (>96^2) |")
            lines.append("|-------|-----------|-------------|-----------|")
            for sp in splits:
                d = self.bbox_size_distribution.get(sp, {})
                lines.append(
                    f"| {sp} | {d.get('small', 0)} | {d.get('medium', 0)} "
                    f"| {d.get('large', 0)} |"
                )
            lines.append("")
            total_boxes = sum(
                sum(d.get(k, 0) for k in ("small", "medium", "large"))
                for d in self.bbox_size_distribution.values()
            )
            total_small = sum(d.get("small", 0) for d in self.bbox_size_distribution.values())
            if total_boxes > 0:
                small_pct = total_small / total_boxes * 100
                lines.append(f"> 小目标占比: **{small_pct:.1f}%** ({total_small}/{total_boxes})")
                lines.append("")

        # 1.4 JS 散度
        if self.js_divergence:
            lines.append("### 1.4 集间分布一致性 (JS 散度)")
            lines.append("")
            lines.append("| 对比 | JS |")
            lines.append("|---|---|")
            for key in ("train_vs_val_js", "train_vs_test_js", "val_vs_test_js"):
                if key in self.js_divergence:
                    lines.append(f"| {key.replace('_', ' ')} | {self.js_divergence[key]:.6f} |")
            if "assessment" in self.js_divergence:
                lines.append(f"\n**评估**: {self.js_divergence['assessment']}")
            lines.append("")

    # ----------------------------------------------------------------
    # 二、训练配置
    # ----------------------------------------------------------------

    def _config_section(self, lines: List[str]) -> None:
        lines.append("## 二、训练配置")
        lines.append("")

        lines.append("### 2.1 任务与模型")
        lines.append("")
        lines.append(f"| 项目 | 内容 |")
        lines.append(f"|------|------|")
        lines.append(f"| 任务类型 | {self.task} |")
        lines.append(f"| 模型 | {self.model_name} |")
        lines.append(f"| 预训练 | {self.pretrained} |")
        lines.append("")

        lines.append("### 2.2 关键超参数")
        lines.append("")
        lines.append("| 参数 | 值 |")
        lines.append("|------|------|")
        for name, val in [
            ("epochs (计划)", self.epochs_planned),
            ("epochs (实际)", self.epochs_actual),
            ("imgsz", self.imgsz),
            ("batch", self.batch),
            ("optimizer", self.optimizer),
            ("lr0", self.lr0),
            ("lrf", self.lrf),
            ("close_mosaic", self.close_mosaic),
            ("seed", self.seed),
        ]:
            lines.append(f"| {name} | {val} |")
        lines.append("")

    # ----------------------------------------------------------------
    # 三、检测结果
    # ----------------------------------------------------------------

    def _results_section(self, lines: List[str]) -> None:
        lines.append("## 三、检测结果")
        lines.append("")

        lines.append("### 3.1 核心指标")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| mAP50 | {_fmt(self.mAP50)} |")
        lines.append(f"| mAP50-95 | {_fmt(self.mAP50_95)} |")
        lines.append(f"| Precision(B) | {_fmt(self.precision)} |")
        lines.append(f"| Recall(B) | {_fmt(self.recall)} |")
        lines.append(f"| Fitness | {_fmt(self.fitness)} |")
        if self.speed_ms:
            lines.append(f"| 总耗时/帧 | {_fmt_ms(self.speed_ms.get('total'))} |")
            lines.append(f"| 推理耗时 | {_fmt_ms(self.speed_ms.get('inference'))} |")
            lines.append(f"| 预处理 | {_fmt_ms(self.speed_ms.get('preprocess'))} |")
            lines.append(f"| 后处理 | {_fmt_ms(self.speed_ms.get('postprocess'))} |")
        lines.append("")

        # 3.2 分类别 mAP
        if self.class_map_50_95:
            lines.append("### 3.2 分类别 mAP50-95")
            lines.append("")
            lines.append("| 类别 | mAP50-95 |")
            lines.append("|------|----------|")
            sorted_items = sorted(
                [(k, v) for k, v in self.class_map_50_95.items()
                 if not (isinstance(v, float) and math.isnan(v))],
                key=lambda kv: kv[1], reverse=True,
            )
            for cname, v in sorted_items:
                lines.append(f"| {cname} | {_fmt(v)} |")
            lines.append("")

            # 分类别差距分析
            if len(sorted_items) >= 2:
                best_name, best_val = sorted_items[0]
                worst_name, worst_val = sorted_items[-1]
                gap = best_val - worst_val
                if gap > 0.2:
                    lines.append(
                        f"> 最佳类别 `{best_name}`({_fmt(best_val)}) 与 "
                        f"最差类别 `{worst_name}`({_fmt(worst_val)}) 差距 "
                        f"{_fmt(gap)}, 类别间性能差异较大。"
                    )
                    lines.append("")

    # ----------------------------------------------------------------
    # 三(b)、训练过程
    # ----------------------------------------------------------------

    def _epoch_section(self, lines: List[str]) -> None:
        if not self.epoch_history:
            return

        lines.append("### 3.3 训练过程")
        lines.append("")

        first = self.epoch_history[0]
        last = self.epoch_history[-1]

        lines.append("**Loss 收敛**:")
        lines.append("")
        items = []
        if first.train_box_loss is not None and last.train_box_loss is not None:
            items.append(
                f"train/box_loss: {_fmt(first.train_box_loss)} → {_fmt(last.train_box_loss)}"
                f" (↓{_fmt(first.train_box_loss - last.train_box_loss)})"
            )
        if first.train_cls_loss is not None and last.train_cls_loss is not None:
            items.append(
                f"train/cls_loss: {_fmt(first.train_cls_loss)} → {_fmt(last.train_cls_loss)}"
                f" (↓{_fmt(first.train_cls_loss - last.train_cls_loss)})"
            )
        for item in items:
            lines.append(f"- {item}")
        lines.append("")

        if self.convergence_note:
            lines.append(f"> {self.convergence_note}")
            lines.append("")

        # mAP 增长
        mAPs = [(r.epoch, r.mAP50_95) for r in self.epoch_history if r.mAP50_95 is not None]
        if mAPs:
            best_epoch, best_mAP = max(mAPs, key=lambda x: x[1])
            first_epoch, first_mAP = mAPs[0]
            last_epoch, last_mAP = mAPs[-1]
            lines.append("**mAP 增长**:")
            lines.append("")
            lines.append(f"- 初始 (epoch {first_epoch}): {_fmt(first_mAP)}")
            lines.append(f"- 最佳 (epoch {best_epoch}): {_fmt(best_mAP)}")
            lines.append(f"- 最终 (epoch {last_epoch}): {_fmt(last_mAP)}")
            if best_mAP is not None and last_mAP is not None and not math.isnan(best_mAP) and not math.isnan(last_mAP):
                if best_mAP - last_mAP > 0.005:
                    lines.append(f"- 最终较最佳下降 {_fmt(best_mAP - last_mAP)}, 可能出现过拟合")
            lines.append("")

        # 逐轮表格(折叠超长)
        lines.append("**逐轮指标**:")
        lines.append("")
        cols = ["Epoch", "box_loss", "cls_loss", "val_box", "val_cls", "Precision", "Recall", "mAP50", "mAP50-95"]
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["------"] * len(cols)) + "|"
        lines.append(header)
        lines.append(sep)

        rows_to_show: list = []
        if len(self.epoch_history) <= 20:
            rows_to_show = list(self.epoch_history)
        else:
            rows_to_show = list(self.epoch_history[:5])
            rows_to_show.append("...")
            rows_to_show.extend(list(self.epoch_history[-10:]))

        for r in rows_to_show:
            if r == "...":
                lines.append("| ... | ... | ... | ... | ... | ... | ... | ... | ... |")
            else:
                line = (
                    f"| {r.epoch} | {_fmt(r.train_box_loss)} | {_fmt(r.train_cls_loss)} "
                    f"| {_fmt(r.val_box_loss)} | {_fmt(r.val_cls_loss)} "
                    f"| {_fmt(r.precision)} | {_fmt(r.recall)} "
                    f"| {_fmt(r.mAP50)} | {_fmt(r.mAP50_95)} |"
                )
                lines.append(line)
        lines.append("")

    # ----------------------------------------------------------------
    # 四、问题分析
    # ----------------------------------------------------------------

    def _analysis_section(self, lines: List[str]) -> None:
        lines.append("## 四、问题分析")
        lines.append("")
        if self.analysis_warnings:
            for w in self.analysis_warnings:
                lines.append(f"- {w}")
        else:
            lines.append("未发现明显问题。")
        lines.append("")

    # ----------------------------------------------------------------
    # 五、改进建议
    # ----------------------------------------------------------------

    def _suggestions_section(self, lines: List[str]) -> None:
        lines.append("## 五、改进建议")
        lines.append("")
        if self.improvement_suggestions:
            for i, s in enumerate(self.improvement_suggestions, 1):
                lines.append(f"{i}. {s}")
        else:
            lines.append("暂无特定改进建议。")
        lines.append("")
        lines.append("---")
        lines.append(f"*报告由 ODPlatform `odp-report` 自动生成 · run_id=`{self.run_id}`*")


# ============================================================
# 便捷函数
# ============================================================

def save_report(report: TrainingReport, path: Any) -> None:
    """将 TrainingReport 保存为 Markdown 文件."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.render_markdown(), encoding="utf-8")
