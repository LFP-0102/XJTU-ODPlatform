#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : __init__.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_eval/__init__.py
# @Project    : XJTU-ODPlatfrom
# @Author     : ODPlatform Team
# @Function   : model_eval 子系统对外公共 API
"""model_eval — 模型评估子系统(单模型评估 + 多模型对比).

公共 API:
  指标:    EvalMetrics
  报告:    EvalReport, ComparisonReport
  服务:    evaluate_model, compare_models
  结果:    EvalResult, ComparisonResult

典型用法 (CLI / 脚本):

    from od_platform.model_eval import evaluate_model
    result = evaluate_model(config=cfg, data_yaml=yaml, run=run, model_ref="best.pt")
"""
from __future__ import annotations

from od_platform.model_eval.analyzer import (
    ConfusionAnalysis,
    ClassRanking,
    PerClassDiff,
    extract_confusion_analysis,
    rank_classes,
    diagnose_problem_classes,
    compare_per_class,
    render_confusion_markdown,
    render_ranking_markdown,
)
from od_platform.model_eval.history import (
    EvalRecord,
    EvalHistory,
    TrendReport,
)
from od_platform.model_eval.metrics import EvalMetrics
from od_platform.model_eval.report import EvalReport, ComparisonReport
from od_platform.model_eval.service import (
    EvalResult, ComparisonResult, evaluate_model, compare_models,
    evaluate_from_infer,
)

__all__ = [
    # 指标
    "EvalMetrics",
    # 报告
    "EvalReport",
    "ComparisonReport",
    # 服务
    "EvalResult",
    "ComparisonResult",
    "evaluate_model",
    "compare_models",
    "evaluate_from_infer",
    # 分析
    "ConfusionAnalysis",
    "ClassRanking",
    "PerClassDiff",
    "extract_confusion_analysis",
    "rank_classes",
    "diagnose_problem_classes",
    "compare_per_class",
    "render_confusion_markdown",
    "render_ranking_markdown",
    # 历史
    "EvalRecord",
    "EvalHistory",
    "TrendReport",
]
