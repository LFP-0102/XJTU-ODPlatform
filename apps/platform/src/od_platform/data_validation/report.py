#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :report.py
# @Time      :2026/7/17 15:35:59
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
# apps/platform/src/od_platform/data_validation/report.py
"""ValidationReport:纯数据(可落 report.json / results.csv);渲染是另一回事(logger / 文件)。"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from od_platform.data_validation.registry import CheckResult, CheckSeverity
from od_platform.data_validation.snapshot import DatasetSnapshot

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """一次验证的完整结果(纯数据)。run_id 可注入,便于测试断言与 CI 对账。"""
    run_id:           str
    snapshot:         DatasetSnapshot
    results:          List[CheckResult]

    @property
    def overall_severity(self) -> str:
        """总评 = 最坏一项(用 rank 取最大)。空结果按 PASS。这是退出码的来源。"""
        if not self.results:
            return CheckSeverity.PASS
        return max((r.severity for r in self.results), key=CheckSeverity.rank)

    def to_dict(self) -> Dict[str, Any]:
        snap = self.snapshot
        return {
            "run_id":           self.run_id,
            "overall_severity": self.overall_severity,
            "dataset": {
                "yaml_path":    str(snap.yaml_path),
                "nc":           snap.nc,
                "class_names":  list(snap.class_names),
                "splits":       {s: len(snap.images_per_split[s]) for s in snap.splits},
                "total_images": snap.total_images,
            },
            "scan_warnings": list(snap.scan_warning),
            "results": [
                {"name": r.name, "severity": r.severity, "summary": r.summary, "details": r.details}
                for r in self.results
            ],
        }

def write_json(report: ValidationReport, path: Path) -> None:
    """机器轨道:结构化完整结果。字段只增不改不删(additive schema),下游才敢长期依赖。"""
    import json
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def write_results_csv(report: ValidationReport, path: Path) -> None:
    """人工轨道:每个 check 一行的扁平清单,拖进表格即可按 severity 排序、分派整改。

    只放 4 列稳定字段——嵌套细节归 report.json,CSV 只负责"一眼扫 + 好排序"。
    utf-8-sig:带 BOM,Excel 双击直接认中文,不乱码(踩过这个坑的都懂)。
    """
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "check", "severity", "summary"])
        for r in report.results:
            w.writerow([report.run_id, r.name, r.severity, r.summary])

def render_to_logger(report: ValidationReport) -> None:
    """给工程师的三段式:头部总评 → 逐项结果 → 指纹段(审计要的第一行)。"""
    snap = report.snapshot
    logger.info("=" * 64)
    logger.info("数据验证报告   run_id=%s   总评=%s", report.run_id, report.overall_severity)
    logger.info("数据集: %s   nc=%s   图像=%d   splits=%s",
                snap.yaml_path.name, snap.nc, snap.total_images,
                {s: len(snap.images_per_split[s]) for s in snap.splits})
    logger.info("-" * 64)
    for r in report.results:
        logger.info("[%-7s] %-16s %s", r.severity, r.name, r.summary)

    # 指纹段:把 contract_fingerprint 原样摆出来——审计问的第一句就靠它答
    fp = next((r for r in report.results if r.name == "fingerprint_match"), None)
    if fp is not None:
        cfp = fp.details.get("contract_fingerprint") or fp.details.get("manifest_fingerprint")
        logger.info("-" * 64)
        logger.info("划分契约指纹 (contract_fingerprint): %s", cfp or "（不可用）")
    if snap.scan_warning:
        logger.info("扫描告警 %d 条（详见 report.json）", len(snap.scan_warning))
    logger.info("=" * 64)