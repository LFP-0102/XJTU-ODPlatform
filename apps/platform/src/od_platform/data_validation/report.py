#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :report.py
# @Time      :2026/7/17 15:35:59
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
# apps/platform/src/od_platform/data_validation/report.py
"""ValidationReport:纯数据(可落 report.json / results.csv / quality_report.md);渲染是另一回事(logger / 文件)。"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
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
    # 分布分析(非 check,纯统计):类别/尺寸/集间漂移。空 dict 表示未分析。
    distribution:     Dict[str, Any] = field(default_factory=dict)

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
            "distribution": self.distribution,
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
    if report.distribution:
        drift = report.distribution.get("inter_split_drift", {})
        if drift:
            logger.info("集间分布漂移: %s", drift.get("assessment", "（无）"))
    if snap.scan_warning:
        logger.info("扫描告警 %d 条（详见 report.json）", len(snap.scan_warning))
    logger.info("=" * 64)


def write_quality_report_md(report: ValidationReport, path: Path) -> None:
    """人类可读的 Markdown 数据质量分析报告:校验结果 + 分布分析 + 指纹。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = report.snapshot
    dist = report.distribution or {}
    lines: List[str] = []

    lines.append(f"# 数据质量分析报告 · {snap.yaml_path.name}\n")
    lines.append(f"- **运行 ID**: {report.run_id}  |  **总评**: {report.overall_severity}")
    lines.append(f"- **数据集**: {snap.yaml_path}  |  **类别数**: {snap.nc}  "
                 f"|  **图像**: {snap.total_images}")
    lines.append(f"- **splits**: "
                 + ", ".join(f"{s}={len(snap.images_per_split[s])}" for s in snap.splits))
    lines.append("")

    lines.append("## 校验结果\n")
    lines.append("| 严重度 | check | 摘要 |")
    lines.append("|---|---|---|")
    for r in report.results:
        lines.append(f"| {r.severity} | {r.name} | {r.summary} |")
    lines.append("")

    # 问题图片清单(来自 ImageIntegrityCheck)
    img_chk = next((r for r in report.results if r.name == "ImageIntegrityCheck"), None)
    if img_chk and img_chk.details.get("problem_images"):
        probs = img_chk.details["problem_images"]
        lines.append("## ⚠️ 问题图片清单(需人工审查)\n")
        lines.append("| split | 图片 | 问题 |")
        lines.append("|---|---|---|")
        for p in probs[:50]:
            lines.append(f"| {p['split']} | {p['image']} | {p['issue']} |")
        if len(probs) > 50:
            lines.append(f"\n*... 共 {img_chk.details['n_problems']} 张,仅列前 50,完整清单见 report.json*\n")
        lines.append("\n> **请亲自审查上述图片。确认删除请运行 `odp-validate --dataset <name> --purge` 并输入大写 YES。**\n")

    if dist:
        cls = dist.get("class_names", [])
        cd = dist.get("class_distribution", {})
        cp = dist.get("class_proportion", {})
        sd = dist.get("bbox_size_distribution", {})
        drift = dist.get("inter_split_drift", {})
        isz = dist.get("image_size_stats", {})

        lines.append("## 类别分布\n")
        lines.append("| 类别 | train | val | test | train占比 | val占比 | test占比 |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in cls:
            lines.append(
                f"| {c} | {cd.get('train', {}).get(c, 0)} | {cd.get('val', {}).get(c, 0)} "
                f"| {cd.get('test', {}).get(c, 0)} | {cp.get('train', {}).get(c, 0)} "
                f"| {cp.get('val', {}).get(c, 0)} | {cp.get('test', {}).get(c, 0)} |")
        lines.append("")

        lines.append("## bbox 尺寸分布 (COCO 面积分桶)\n")
        lines.append("| split | 小 (<32²) | 中 (32-96²) | 大 (>96²) |")
        lines.append("|---|---|---|---|")
        for sp in ("train", "val", "test"):
            d = sd.get(sp, {})
            lines.append(f"| {sp} | {d.get('small', 0)} | {d.get('medium', 0)} | {d.get('large', 0)} |")
        lines.append("")

        if drift:
            lines.append("## 集间分布漂移 (JS 散度, 越小越一致)\n")
            lines.append("| 对比 | JS |")
            lines.append("|---|---|")
            lines.append(f"| train vs val | {drift.get('train_vs_val_js', '-')} |")
            lines.append(f"| train vs test | {drift.get('train_vs_test_js', '-')} |")
            lines.append(f"| val vs test | {drift.get('val_vs_test_js', '-')} |")
            lines.append(f"\n**评估**: {drift.get('assessment', '-')}\n")

        if isz:
            lines.append("## 图片尺寸统计\n")
            lines.append("| split | 数量 | 平均宽 | 平均高 |")
            lines.append("|---|---|---|---|")
            for sp in ("train", "val", "test"):
                s = isz.get(sp, {})
                lines.append(f"| {sp} | {s.get('count', 0)} | {s.get('width_mean', 0)} | {s.get('height_mean', 0)} |")
            lines.append("")

    if snap.scan_warning:
        lines.append("## 扫描告警\n")
        for w in snap.scan_warning:
            lines.append(f"- {w}")
        lines.append("")

    fp = next((r for r in report.results if r.name == "fingerprint_match"), None)
    if fp is not None:
        cfp = fp.details.get("contract_fingerprint") or fp.details.get("manifest_fingerprint")
        lines.append("## 划分契约指纹\n")
        lines.append(f"`{cfp}`\n")

    path.write_text("\n".join(lines), encoding="utf-8")
