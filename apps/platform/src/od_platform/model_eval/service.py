#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : service.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_eval/service.py
# @Project    : XJTU-ODPlatfrom
# @Author     : ODPlatform Team
# @Function   : 模型评估编排服务 —— 单模型评估 + 多模型对比
"""模型评估整体编排.

职责(与 model_train/service.py 同构):
  1. 解析模型引用(trained / pretrained / 绝对路径 / ultralytics 自动下载)
  2. 调 ultralytics model.val() 跑验证, 拿到 results
  3. EvalMetrics.from_yolo_results 抽取 + 派生指标
  4. 落盘报告(report.json + result.csv + report.md)

复用:
  · RunContext("evaluation")        —— runs/model_evaluation/<run_id>/
  · YOLOValConfig / build_val_config —— val 配置(split / conf / iou / plots ...)
  · TrainMetrics.from_yolo_results   —— 原生指标抽取(EvalMetrics 内部复用)
  · refs.resolve_model               —— 模型引用解析
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from od_platform.common import refs
from od_platform.common.paths import PRETRAINED_MODELS_DIR, TRAINED_MODELS_DIR
from od_platform.common.run_context import RunContext
from od_platform.common.report_config import log_config_report
from od_platform.model_eval.analyzer import (
    extract_confusion_analysis,
    rank_classes,
    diagnose_problem_classes,
    render_confusion_markdown,
    render_ranking_markdown,
)
from od_platform.model_eval.history import EvalHistory
from od_platform.model_eval.metrics import EvalMetrics
from od_platform.model_eval.report import EvalReport, ComparisonReport, _fmt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    """单模型评估结果."""
    success: bool
    run_id: str
    report: Optional[EvalReport] = None
    message: str = ""


@dataclass(frozen=True)
class ComparisonResult:
    """多模型对比结果."""
    success: bool
    run_id: str
    report: Optional[ComparisonReport] = None
    message: str = ""


def _resolve_model_arg(model: str) -> str:
    """解析模型引用 -> 可加载路径.

    顺序: trained/ -> pretrained/ -> 原样交给 ultralytics(自动下载官方权重).
    与 train service 的 _resolve_model_arg 对齐, 但评估优先 trained.
    """
    # 1. trained 目录优先(评估通常针对训练好的权重)
    p_trained = refs.resolve_model(model)
    if p_trained.exists():
        logger.info("模型解析(trained): %s", p_trained)
        return str(p_trained)
    # 2. pretrained 目录
    p_pre = refs.resolve_ref(model, base_dir=PRETRAINED_MODELS_DIR, default_suffix=".pt")
    if p_pre.exists():
        logger.info("模型解析(pretrained): %s", p_pre)
        return str(p_pre)
    # 3. 绝对路径 / 带分隔符的相对路径
    p = Path(model)
    if p.is_absolute() and p.exists():
        logger.info("模型解析(绝对路径): %s", p)
        return str(p)
    logger.info("本地未发现模型 %s, Ultralytics 将自动下载 %s", model, model)
    return model


def _display_name(model_ref: str) -> str:
    """给报告用的模型展示名(去路径去后缀)."""
    return Path(model_ref).stem


def _run_val(model_arg: str, config: Any, data_yaml: Path, run: RunContext) -> Any:
    """调 ultralytics model.val(), 返回 results.

    ★ 目录对齐: 让 ultralytics 的图表(PR/混淆矩阵/val_batch)直接落进我们的
       runs/model_evaluation/<run_id>/ 下, 和 report.json/csv/md 聚在一处.
       走 "project=run.run_dir, name='ultra'" 的技巧: ultralytics 会建
       <project>/<name>/ 放图表, 避免和我们的 report.* 同名冲突.
    """
    from ultralytics import YOLO
    model = YOLO(model_arg)
    kwargs = config.to_ultralytics_kwargs()
    # data / model 由调用方显式传, 不走 kwargs
    kwargs.pop("data", None)
    kwargs.pop("model", None)
    # project=run.run_dir 让图表落到我们的评估目录下; name='ultra' 建子目录隔开图表与报告
    run.run_dir.mkdir(parents=True, exist_ok=True)
    kwargs.update(project=str(run.run_dir), name="ultra", exist_ok=True)
    return model.val(data=str(data_yaml), **kwargs)


def _eval_one(model_ref: str, config: Any, data_yaml: Path, run: RunContext,
              run_id: str, created_at: str) -> EvalMetrics:
    """评估单个模型, 返回 EvalMetrics. 不落盘(落盘由上层统一做)."""
    model_name = _display_name(model_ref)
    model_arg = _resolve_model_arg(model_ref)

    logger.info("-" * 60)
    logger.info("评估模型: %s  (split=%s)", model_name, getattr(config, "split", "val"))
    results = _run_val(model_arg, config, data_yaml, run)
    split = getattr(config, "split", "val")
    metrics = EvalMetrics.from_yolo_results(
        results, run_id=run_id, model_name=model_name,
        model_path=model_arg, split=split,
    )
    logger.info("完成: %s  P=%s R=%s F1=%s mAP50=%s mAP50-95=%s acc=%s",
                model_name,
                _s(metrics.precision), _s(metrics.recall), _s(metrics.f1),
                _s(metrics.mAP50), _s(metrics.mAP50_95), _s(metrics.accuracy))
    return metrics


def _s(v: float) -> str:
    import math as _m
    return "N/A" if (isinstance(v, float) and _m.isnan(v)) else f"{v:.4f}"


# ============================================================
# 单模型评估
# ============================================================

def evaluate_model(*, config: Any, data_yaml: Path, run: RunContext,
                   model_ref: str, merger: Optional[Any] = None,
                   write_report: bool = True,
                   save_history: bool = True,
                   enhanced_analysis: bool = True) -> EvalResult:
    """评估单个模型, 生成评估报告.

    Args:
        config:     YOLOValConfig 实例
        data_yaml:  数据集 yaml 路径
        run:        RunContext("evaluation")
        model_ref:  模型引用(名 / 路径)
        merger:     配置溯源(可选, 打来源报告)
        write_report: 是否落盘 report.json / result.csv / report.md
        save_history: 是否将本次评估追加到历史记录
        enhanced_analysis: 是否生成增强分析(混淆矩阵分析 + 类别排名 + 问题诊断)
    """
    logger.info("=" * 60)
    logger.info("模型评估开始 | run_id=%s", run.run_id)
    logger.info("输出目录: %s", run.run_dir)
    logger.info("数据集: %s", data_yaml)
    if merger is not None:
        log_config_report(config, merger, logger=logger)

    model_name = _display_name(model_ref)
    try:
        metrics = _eval_one(model_ref, config, data_yaml, run,
                            run_id=run.run_id, created_at=run.created_at)
    except KeyboardInterrupt:
        logger.warning("评估被手动中断 | run_id=%s", run.run_id)
        return EvalResult(success=False, run_id=run.run_id, message="评估被手动中断")
    except Exception as e:
        logger.exception("评估失败 | run_id=%s 错误: %s", run.run_id, e)
        return EvalResult(success=False, run_id=run.run_id, message=f"评估失败: {e}")

    report = EvalReport(
        run_id=run.run_id, model_name=model_name,
        model_path=metrics.model_path, data_yaml=str(data_yaml),
        split=metrics.split, created_at=run.created_at, metrics=metrics,
    )
    report.render_to_logger(logger)

    if write_report:
        run.run_dir.mkdir(parents=True, exist_ok=True)

        # 基础报告
        report.write_json(run.run_dir / "report.json")
        report.write_csv(run.run_dir / "result.csv")

        # 构建增强 Markdown 报告(基础报告 + 增强分析)
        md_parts = [report.render_markdown()]

        if enhanced_analysis:
            # 类别排名
            rankings = rank_classes(metrics, sort_by="mAP50_95")
            if rankings:
                md_parts.append(render_ranking_markdown(rankings, "类别性能排序 (按 mAP50-95)"))

            # 问题诊断
            diagnosis = diagnose_problem_classes(metrics)
            if diagnosis["critical"] or diagnosis["warning"]:
                lines: list[str] = ["## 问题类别诊断", ""]
                if diagnosis["critical"]:
                    lines.append(f"- 🔴 **严重** (F1 极低): {', '.join(diagnosis['critical'])}")
                if diagnosis["warning"]:
                    lines.append(f"- 🟡 **关注** (F1 偏低): {', '.join(diagnosis['warning'])}")
                lines.append(f"- 🟢 **正常**: {len(diagnosis['good'])} 个类别")
                lines.append("")
                md_parts.append("\n".join(lines))

        # 写入增强版 Markdown 报告
        full_md = "\n".join(md_parts)
        (run.run_dir / "report.md").write_text(full_md, encoding="utf-8")
        logger.info("报告已写入 %s (report.json + result.csv + report.md)", run.run_dir)

    # 追加到评估历史
    if save_history:
        try:
            history = EvalHistory.load_or_create(str(data_yaml))
            history.add_record(report)
            history.save()
        except Exception as e:
            logger.warning("评估历史保存失败(不阻塞主流程): %s", e)

    return EvalResult(success=True, run_id=run.run_id, report=report)


# ============================================================
# 多模型对比评估
# ============================================================

def compare_models(*, config: Any, data_yaml: Path, run: RunContext,
                   model_refs: List[str], merger: Optional[Any] = None,
                   write_report: bool = True,
                   save_history: bool = True,
                   enhanced_analysis: bool = True) -> ComparisonResult:
    """对多个模型在同一数据集上做对比评估, 生成对比报告.

    所有模型共享同一个 run_id / run_dir, 保证对比产物聚在一处.
    单个模型失败不影响其余模型, 失败的模型在报告中标注缺失.

    Args:
        config:     YOLOValConfig 实例
        data_yaml:  数据集 yaml 路径
        run:        RunContext("evaluation")
        model_refs: 模型引用列表
        merger:     配置溯源(可选)
        write_report: 是否落盘 comparison.json / comparison.csv / comparison.md
        save_history: 是否将每个模型的评估追加到历史记录
        enhanced_analysis: 是否生成增强分析(类别级对比 + 差异分析)
    """
    logger.info("=" * 60)
    logger.info("多模型对比评估开始 | run_id=%s | 模型数=%d",
                run.run_id, len(model_refs))
    logger.info("输出目录: %s", run.run_dir)
    logger.info("数据集: %s", data_yaml)
    logger.info("对比模型: %s", ", ".join(model_refs))
    if merger is not None:
        log_config_report(config, merger, logger=logger)

    all_metrics: List[EvalMetrics] = []
    all_reports: List[EvalReport] = []
    failures: List[str] = []
    for i, ref in enumerate(model_refs, 1):
        logger.info("\n[%d/%d] 评估模型: %s", i, len(model_refs), ref)
        try:
            metrics = _eval_one(ref, config, data_yaml, run,
                                run_id=run.run_id, created_at=run.created_at)
            all_metrics.append(metrics)
            all_reports.append(EvalReport(
                run_id=run.run_id, model_name=_display_name(ref),
                model_path=metrics.model_path, data_yaml=str(data_yaml),
                split=metrics.split, created_at=run.created_at, metrics=metrics,
            ))
        except KeyboardInterrupt:
            logger.warning("对比评估被手动中断 | run_id=%s", run.run_id)
            return ComparisonResult(success=False, run_id=run.run_id,
                                    message="对比评估被手动中断")
        except Exception as e:
            logger.error("模型 %s 评估失败, 跳过: %s", ref, e)
            failures.append(f"{_display_name(ref)}: {e}")

    if not all_metrics:
        return ComparisonResult(success=False, run_id=run.run_id,
                                message="所有模型评估均失败: " + " | ".join(failures))

    report = ComparisonReport(
        run_id=run.run_id, data_yaml=str(data_yaml),
        split=getattr(config, "split", "val"),
        created_at=run.created_at, models=all_metrics,
    )
    report.render_to_logger(logger)

    if write_report:
        run.run_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(run.run_dir / "comparison.json")
        report.write_csv(run.run_dir / "comparison.csv")

        # 构建增强 Markdown 报告(基础对比 + 增强分析)
        md_parts = [report.render_markdown()]

        if enhanced_analysis and len(all_metrics) >= 2:
            from od_platform.model_eval.analyzer import compare_per_class
            # 只对比前两个模型的类别级差异(太多模型表格太长)
            diffs = compare_per_class(all_metrics[0], all_metrics[1], metric="mAP50_95")
            if diffs:
                lines: list[str] = [
                    "## 类别级差异分析",
                    "",
                    f"对比 `{all_metrics[0].model_name}` vs `{all_metrics[1].model_name}` (按 mAP50-95)",
                    "",
                    "| 类别 | {:<16} | {:<16} | 差异 | 胜出 |".format(
                        all_metrics[0].model_name[:16], all_metrics[1].model_name[:16]),
                    "|------|------------------|------------------|------|------|",
                ]
                for d in diffs:
                    winner_icon = {"A": "←", "B": "→", "tie": "—"}.get(d.winner, "?")
                    lines.append(
                        f"| {d.class_name} | {_fmt(d.value_a)} | {_fmt(d.value_b)} | "
                        f"{_fmt(d.diff)} | {winner_icon} |"
                    )
                lines.append("")
                lines.append("> ← 表示左侧模型更优, → 表示右侧模型更优, — 表示差异不显著")
                lines.append("")
                md_parts.append("\n".join(lines))

        # 写入增强版 Markdown 报告
        full_md = "\n".join(md_parts)
        (run.run_dir / "comparison.md").write_text(full_md, encoding="utf-8")
        logger.info("对比报告已写入 %s (comparison.json + comparison.csv + comparison.md)",
                    run.run_dir)

    # 各模型追加到评估历史
    if save_history:
        try:
            history = EvalHistory.load_or_create(str(data_yaml))
            for r in all_reports:
                history.add_record(r)
            history.save()
        except Exception as e:
            logger.warning("评估历史保存失败(不阻塞主流程): %s", e)

    msg = "对比完成"
    if failures:
        msg += f"; 失败模型: {', '.join(failures)}"
    return ComparisonResult(success=True, run_id=run.run_id, report=report, message=msg)


__all__ = ["EvalResult", "ComparisonResult", "evaluate_model", "compare_models"]
