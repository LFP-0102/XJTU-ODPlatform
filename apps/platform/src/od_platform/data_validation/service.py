#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName  :service.py
# @Time      :2026/7/17 13:47:13
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import annotations
import json
import logging
import shutil
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from od_platform.data_validation import snapshot
from od_platform.data_validation.distribution import analyze_distribution

from od_platform.data_validation.registry import (CheckContext, CheckEntry, CheckResult, CheckSeverity, get_all_checks)
from od_platform.data_validation.report import (
    ValidationReport, render_to_logger, write_results_csv, write_json, write_quality_report_md
)
from od_platform.data_validation.snapshot import build_snapshot
from od_platform.common.performance_utils import time_it
from od_platform.common import paths

logger = logging.getLogger(__name__)

@time_it(name=lambda entry, ctx: f"检查:【{entry.name}】", logger_instance=logger, iterations=1)
def _safe_run_one(entry: CheckEntry, ctx: CheckContext) -> CheckResult:
    try:
        return entry.func(ctx)
    except Exception as e:
        logger.error(f"Check {entry.name} 执行异常，兜底为 ERROR")
        return CheckResult(
            entry.name,
            CheckSeverity.ERROR,
            f"Check failed with exception: {e}",
            {"exception": str(e)},
        )

def run_all_checks(ctx: CheckContext) -> List[CheckResult]:
    results = [_safe_run_one(e, ctx) for e in get_all_checks()]
    logger.info("All checks completed.共完成 %d 个检查", len(results))
    return results

def _collect_problem_images(report: ValidationReport, dest_dir: Path) -> None:
    """把问题图片(损坏 + 完全重复)复制到报告目录的 problem_images/,方便人工审查。

    按 split 分子目录,附 problem_images_manifest.json 清单。最多收集 200 张。
    复制(非移动),原图不动,审查后可用 --purge 删除。
    """
    targets: List[tuple] = []
    img_chk = next((r for r in report.results if r.name == "ImageIntegrityCheck"), None)
    if img_chk:
        for p in img_chk.details.get("problem_images", []):
            targets.append((p["path"], p["split"], p["image"], p["issue"]))
    dup_chk = next((r for r in report.results if r.name == "DuplicateImageCheck"), None)
    if dup_chk:
        for d in dup_chk.details.get("exact_duplicates", []):
            targets.append((d["path"], d["split"], d["image"], f"完全重复(同 {d['duplicate_of']})"))
    if not targets:
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []
    for path_str, split, image, issue in targets[:200]:
        src = Path(path_str)
        if not src.exists():
            continue
        sub = dest_dir / split
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / image
        try:
            shutil.copy2(src, dst)
            manifest.append({"split": split, "image": image, "issue": issue,
                             "path": str(dst.relative_to(dest_dir))})
        except OSError as e:
            logger.warning("复制问题图片失败 %s: %s", src, e)
    (dest_dir / "problem_images_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        rel = dest_dir.relative_to(paths.ROOT_DIR)
    except ValueError:
        rel = dest_dir
    logger.info("问题图片已收集到 %s (共 %d 张,详见 problem_images_manifest.json)", rel, len(manifest))

def validate_dataset(yaml_path: Path, task_type: Optional[str] = None,
            run_id: Optional[str] = None, write_report: bool = True
                    ) -> ValidationReport:
    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    snap = build_snapshot(yaml_path=Path(yaml_path), task_type=task_type)

    # 分布分析(非 check,纯统计;失败不阻断校验)
    try:
        distribution = analyze_distribution(snap)
    except Exception as e:
        logger.warning("分布分析失败(不影响校验): %s", e)
        distribution = {}

    results = run_all_checks(CheckContext(yaml_path=yaml_path, snapshot=snap))
    report = ValidationReport(
        run_id=resolved_run_id, snapshot=snap, results=results,
        distribution=distribution,
    )

    render_to_logger(report)

    if write_report:
        run_dir = paths.validation_run_dir(resolved_run_id)
        run_dir.mkdir(parents=True,exist_ok=True)
        write_json(report, run_dir / "report.json")
        write_results_csv(report, run_dir / "result.csv")
        write_quality_report_md(report, run_dir / "quality_report.md")
        _collect_problem_images(report, run_dir / "problem_images")
        logger.info(f"报告已经写入 %s (report.json + result.csv + quality_report.md + problem_images/)", run_dir)

    # 问题图片提醒(损坏图 + 完全重复图)
    img_chk = next((r for r in results if r.name == "ImageIntegrityCheck"), None)
    dup_chk = next((r for r in results if r.name == "DuplicateImageCheck"), None)
    n_bad = 0
    if img_chk and img_chk.details.get("problem_images"):
        n_bad += img_chk.details["n_problems"]
    if dup_chk and dup_chk.details.get("exact_duplicates"):
        n_bad += len(dup_chk.details["exact_duplicates"])
    if n_bad:
        logger.warning("⚠️ 发现 %d 张问题图片(损坏/完全重复),已收集到 problem_images/,详见 quality_report.md", n_bad)
        logger.warning("⚠️ 请亲自审查后,运行 'odp-validate --dataset <name> --purge' 并输入 YES 确认删除")
    return report
