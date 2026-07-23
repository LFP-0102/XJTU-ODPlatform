#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : distribution.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_validation/distribution.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:数据分布分析器(非 check,纯统计,供质量报告消费)
"""数据分布分析器:对校验快照做一次纯统计扫描,产出分布数据供质量报告消费。

它不是 check(不判定严重级别、不影响退出码)——不均衡不该算"错误",很多数据集
天然不均衡。它只回答"数据长什么样":类别分布、bbox 尺寸分布、集间分布漂移、
图片尺寸统计。check 负责找问题,分布分析负责描摹画像,两者互补。

数据源:DatasetSnapshot(已扫好的 images/labels 路径)+ YOLO 标签内容 + 图片像素
尺寸(纯 Python 读头,无第三方依赖)。
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

from od_platform.data_pipeline.convert.converters._geometry import read_image_size
from od_platform.data_validation.snapshot import DatasetSnapshot

_SMALL = 32 * 32      # COCO 小目标面积阈值(像素²)
_MEDIUM = 96 * 96     # COCO 中目标面积阈值
_SPLITS = ("train", "val", "test")


def _js_divergence(p: List[float], q: List[float]) -> float:
    """归一化 JS 散度 ∈ [0,1],衡量两个概率分布的差异。"""
    n = len(p)
    m = [(p[i] + q[i]) / 2 for i in range(n)]

    def kl(a: List[float], b: List[float]) -> float:
        return sum(a[i] * math.log(a[i] / b[i]) for i in range(n)
                   if a[i] > 0 and b[i] > 0)

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)


def analyze_distribution(snapshot: DatasetSnapshot) -> Dict[str, Any]:
    """对快照做分布分析,返回结构化 dict(供 report.json / quality_report.md 消费)。"""
    class_names = list(snapshot.class_names)
    class_dist = {sp: defaultdict(int) for sp in _SPLITS}
    size_dist = {sp: {"small": 0, "medium": 0, "large": 0} for sp in _SPLITS}
    box_counts = {sp: 0 for sp in _SPLITS}
    n_images = {sp: 0 for sp in _SPLITS}
    img_w: Dict[str, List[int]] = {sp: [] for sp in _SPLITS}
    img_h: Dict[str, List[int]] = {sp: [] for sp in _SPLITS}

    for sp in snapshot.splits:
        imgs = snapshot.images_per_split[sp]
        labels = snapshot.labels_per_split.get(sp, ())
        for i, img in enumerate(imgs):
            n_images[sp] += 1
            wh = read_image_size(img)
            if wh:
                img_w[sp].append(wh[0])
                img_h[sp].append(wh[1])
            label_path = labels[i] if i < len(labels) else None
            if label_path is None or not label_path.exists():
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    w_norm, h_norm = float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                cname = class_names[cid] if 0 <= cid < len(class_names) else f"cls_{cid}"
                class_dist[sp][cname] += 1
                box_counts[sp] += 1
                if wh:
                    area = w_norm * wh[0] * h_norm * wh[1]
                    if area < _SMALL:
                        size_dist[sp]["small"] += 1
                    elif area < _MEDIUM:
                        size_dist[sp]["medium"] += 1
                    else:
                        size_dist[sp]["large"] += 1

    # 类别占比
    class_prop: Dict[str, Dict[str, float]] = {}
    for sp in _SPLITS:
        total = sum(class_dist[sp].values()) or 1
        class_prop[sp] = {c: round(class_dist[sp].get(c, 0) / total, 4) for c in class_names}

    # 集间分布漂移(JS 散度)
    all_cls = sorted({c for sp in _SPLITS for c in class_dist[sp]})

    def prob(sp: str) -> List[float]:
        total = sum(class_dist[sp].values()) or 1
        return [class_dist[sp].get(c, 0) / total for c in all_cls]

    p_tr, p_va, p_te = prob("train"), prob("val"), prob("test")
    drift = {
        "train_vs_val_js": round(_js_divergence(p_tr, p_va), 4),
        "train_vs_test_js": round(_js_divergence(p_tr, p_te), 4),
        "val_vs_test_js": round(_js_divergence(p_va, p_te), 4),
    }
    max_js = max(drift.values())
    drift["max_js"] = max_js
    if max_js < 0.05:
        drift["assessment"] = f"分布一致性良好 (max JS={max_js})"
    elif max_js < 0.15:
        drift["assessment"] = f"分布略有差异 (max JS={max_js})"
    else:
        drift["assessment"] = f"分布差异较大 (max JS={max_js})"

    img_stats: Dict[str, Dict[str, Any]] = {}
    for sp in _SPLITS:
        ws, hs = img_w[sp], img_h[sp]
        img_stats[sp] = {
            "count": len(ws),
            "width_mean": round(sum(ws) / len(ws), 1) if ws else 0,
            "height_mean": round(sum(hs) / len(hs), 1) if hs else 0,
        }

    return {
        "class_names": class_names,
        "n_classes": len(class_names),
        "total_images": sum(n_images.values()),
        "total_boxes": sum(box_counts.values()),
        "images_per_split": n_images,
        "boxes_per_split": box_counts,
        "class_distribution": {sp: dict(class_dist[sp]) for sp in _SPLITS},
        "class_proportion": class_prop,
        "bbox_size_distribution": size_dist,
        "inter_split_drift": drift,
        "image_size_stats": img_stats,
    }
