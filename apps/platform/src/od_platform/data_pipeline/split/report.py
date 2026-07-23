#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : report.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_pipeline/split/report.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:划分报告:类别分布 / bbox 尺寸 / 集间一致性 / 指纹
"""划分报告:把一次划分冻结成人类可读、机器可解析的质量快照。

四块内容:
  1. 各 split 规模与类别分布 —— 每类在 train/val/test 的计数。
  2. bbox 尺寸分布 —— 按 COCO 面积分桶(小<32²/中 32-96²/大>96²),看小目标占比。
  3. 集间分布一致性 —— train/val/test 类别比例的两两 JS 散度(归一化到 [0,1]),
     越小越一致;<0.05 良好、<0.15 略有差异、否则建议检查。
  4. 划分契约指纹 —— contract_fingerprint,审计回溯用。

数据来源:manifest(逐样本 split)+ staging 目录的 YOLO 标签(归一化 bbox)+
原始图片目录(读像素尺寸算面积)。报告生成失败不影响主流程(orchestrator 已兜底)。
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from od_platform.common.constants import IMAGE_EXTENSIONS
from od_platform.data_pipeline.convert.converters._geometry import read_image_size

logger = logging.getLogger(__name__)

# COCO 面积分桶阈值(像素²)
_SMALL = 32 * 32      # 1024
_MEDIUM = 96 * 96     # 9216

_SPLITS = ("train", "val", "test")


def _read_image_wh(image_dir: Path, stem: str) -> tuple[int, int] | None:
    for ext in IMAGE_EXTENSIONS:
        cand = image_dir / f"{stem}{ext}"
        if cand.exists():
            sz = read_image_size(cand)
            if sz:
                return sz
    return None


def _js_divergence(p: List[float], q: List[float]) -> float:
    """归一化 JS 散度 ∈ [0,1]。p/q 等长概率分布(和为1)。"""
    n = len(p)
    m = [(p[i] + q[i]) / 2 for i in range(n)]

    def kl(a: List[float], b: List[float]) -> float:
        return sum(a[i] * math.log(a[i] / b[i]) for i in range(n)
                   if a[i] > 0 and b[i] > 0)

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return js / math.log(2)  # 归一化到 [0,1]


def build_split_report(manifest, staging_dir: Path, image_dir: Path,
                       classes: List[str]) -> Dict[str, Any]:
    """构建划分报告 dict。manifest 来自 common.lineage.SplitManifest。"""
    stems_per_split: Dict[str, List[str]] = defaultdict(list)
    for s in manifest.samples:
        stems_per_split[s.split].append(s.stem)

    class_dist = {sp: defaultdict(int) for sp in _SPLITS}
    size_dist = {sp: {"small": 0, "medium": 0, "large": 0} for sp in _SPLITS}
    box_total = {sp: 0 for sp in _SPLITS}
    img_w: Dict[str, List[int]] = {sp: [] for sp in _SPLITS}
    img_h: Dict[str, List[int]] = {sp: [] for sp in _SPLITS}

    for sp in _SPLITS:
        for stem in stems_per_split.get(sp, []):
            wh = _read_image_wh(image_dir, stem)
            if wh:
                img_w[sp].append(wh[0])
                img_h[sp].append(wh[1])
            txt = staging_dir / f"{stem}.txt"
            if not txt.exists():
                continue
            for line in txt.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    w_norm, h_norm = float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                cname = classes[cid] if 0 <= cid < len(classes) else f"cls_{cid}"
                class_dist[sp][cname] += 1
                box_total[sp] += 1
                if wh:
                    area = w_norm * wh[0] * h_norm * wh[1]
                    if area < _SMALL:
                        size_dist[sp]["small"] += 1
                    elif area < _MEDIUM:
                        size_dist[sp]["medium"] += 1
                    else:
                        size_dist[sp]["large"] += 1

    # 集间类别分布一致性
    all_classes = sorted({c for sp in _SPLITS for c in class_dist[sp]})

    def prob(sp: str) -> List[float]:
        total = sum(class_dist[sp].values()) or 1
        return [class_dist[sp].get(c, 0) / total for c in all_classes]

    p_tr, p_va, p_te = prob("train"), prob("val"), prob("test")
    consistency = {
        "train_vs_val_js": round(_js_divergence(p_tr, p_va), 4),
        "train_vs_test_js": round(_js_divergence(p_tr, p_te), 4),
        "val_vs_test_js": round(_js_divergence(p_va, p_te), 4),
    }
    max_js = max(consistency.values())
    consistency["max_js"] = max_js
    if max_js < 0.05:
        consistency["assessment"] = f"分布一致性良好 (max JS={max_js})"
    elif max_js < 0.15:
        consistency["assessment"] = f"分布略有差异 (max JS={max_js})"
    else:
        consistency["assessment"] = f"分布差异较大,建议检查划分 (max JS={max_js})"

    img_stats = {}
    for sp in _SPLITS:
        ws, hs = img_w[sp], img_h[sp]
        img_stats[sp] = {
            "count": len(ws),
            "width_mean": round(sum(ws) / len(ws), 1) if ws else 0,
            "height_mean": round(sum(hs) / len(hs), 1) if hs else 0,
        }

    return {
        "contract_fingerprint": manifest.contract_fingerprint,
        "dataset": manifest.dataset,
        "strategy": manifest.strategy,
        "seed": manifest.seed,
        "rations": list(manifest.rations),
        "classes": list(classes),
        "split_counts": {sp: len(stems_per_split.get(sp, [])) for sp in _SPLITS},
        "box_counts": box_total,
        "class_distribution": {sp: dict(class_dist[sp]) for sp in _SPLITS},
        "bbox_size_distribution": size_dist,
        "inter_split_consistency": consistency,
        "image_size_stats": img_stats,
    }


def write_split_report_json(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def write_split_report_md(report: Dict[str, Any], path: Path) -> None:
    """人类可读的 Markdown 划分报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    cls = report["classes"]
    cd = report["class_distribution"]
    sd = report["bbox_size_distribution"]
    con = report["inter_split_consistency"]
    isz = report["image_size_stats"]
    sc = report["split_counts"]
    bc = report["box_counts"]

    lines: List[str] = []
    lines.append(f"# 数据划分报告 · {report['dataset']}\n")
    lines.append(f"- **契约指纹**: `{report['contract_fingerprint']}`")
    lines.append(f"- **策略**: {report['strategy']}  |  **种子**: {report['seed']}  "
                 f"|  **比例** train/val/test = {report['rations'][0]}/{report['rations'][1]}/{report['rations'][2]}")
    lines.append(f"- **类别** ({len(cls)}): {', '.join(cls)}\n")

    lines.append("## 各 split 规模\n")
    lines.append("| split | 图片数 | 框总数 |")
    lines.append("|---|---|---|")
    for sp in _SPLITS:
        lines.append(f"| {sp} | {sc.get(sp, 0)} | {bc.get(sp, 0)} |")
    lines.append("")

    lines.append("## 类别分布\n")
    lines.append("| 类别 | train | val | test |")
    lines.append("|---|---|---|---|")
    for c in cls:
        lines.append(f"| {c} | {cd['train'].get(c, 0)} | {cd['val'].get(c, 0)} | {cd['test'].get(c, 0)} |")
    lines.append("")

    lines.append("## bbox 尺寸分布 (COCO 面积分桶)\n")
    lines.append("| split | 小 (<32²) | 中 (32-96²) | 大 (>96²) |")
    lines.append("|---|---|---|---|")
    for sp in _SPLITS:
        d = sd[sp]
        lines.append(f"| {sp} | {d['small']} | {d['medium']} | {d['large']} |")
    lines.append("")

    lines.append("## 集间分布一致性 (JS 散度, 越小越一致)\n")
    lines.append("| 对比 | JS |")
    lines.append("|---|---|")
    lines.append(f"| train vs val | {con['train_vs_val_js']} |")
    lines.append(f"| train vs test | {con['train_vs_test_js']} |")
    lines.append(f"| val vs test | {con['val_vs_test_js']} |")
    lines.append(f"\n**评估**: {con['assessment']}\n")

    lines.append("## 图片尺寸统计\n")
    lines.append("| split | 数量 | 平均宽 | 平均高 |")
    lines.append("|---|---|---|---|")
    for sp in _SPLITS:
        s = isz[sp]
        lines.append(f"| {sp} | {s['count']} | {s['width_mean']} | {s['height_mean']} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
