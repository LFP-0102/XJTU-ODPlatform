#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : annotation_integrity.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_validation/checks/annotation_integrity.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:标注完整性检测(越界/零面积/倒置/解析错误)
"""标注完整性检测。

逐框核对 YOLO 归一化标注(cx cy w h)还原成像素角点后:
  - zero_or_negative_size: w<=0 或 h<=0(空框/负值,源数据错误)。
  - inverted_box: x2<x1 或 y2<y1(角点倒置,转换器算错或源数据乱)。
  - out_of_bounds: 框超出图像边界超过 2px 容差(标注溢出,采集或对齐错误)。
  - parse_error: 行无法按 class cx cy w h 解析。

判定:问题占比 ≥5% → ERROR(数据质量不可用);有问题但少 → WARNING;无问题 → PASS。
图片尺寸用纯 Python 读 JPEG/PNG/BMP 头(_geometry.read_image_size),无第三方依赖。
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from od_platform.data_pipeline.convert.converters._geometry import read_image_size
from od_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check

_NAME = "AnnotationIntegrityCheck"
_TOL = 2          # 像素容差:框轻微贴边不算越界
_ERROR_RATIO = 0.05
_MAX_DETAIL = 20


@check(_NAME)
def validate_annotation_integrity(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    issues: List[dict] = []
    n_boxes = 0
    n_images_checked = 0

    for split in snap.splits:
        imgs = snap.images_per_split[split]
        labels = snap.labels_per_split.get(split, ())
        for i, img in enumerate(imgs):
            label_path = labels[i] if i < len(labels) else None
            if label_path is None or not label_path.exists():
                continue
            wh = read_image_size(img)
            if not wh:
                continue
            W, H = wh
            n_images_checked += 1
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                    float(parts[3]), float(parts[4]))
                except (ValueError, IndexError):
                    issues.append({"split": split, "image": img.name,
                                   "issue": "parse_error", "line": line[:50]})
                    continue
                n_boxes += 1
                x1 = (cx - w / 2) * W
                y1 = (cy - h / 2) * H
                x2 = (cx + w / 2) * W
                y2 = (cy + h / 2) * H
                if w <= 0 or h <= 0:
                    issues.append({"split": split, "image": img.name,
                                   "issue": "zero_or_negative_size", "w": w, "h": h})
                elif x2 < x1 or y2 < y1:
                    issues.append({"split": split, "image": img.name, "issue": "inverted_box"})
                elif x1 < -_TOL or y1 < -_TOL or x2 > W + _TOL or y2 > H + _TOL:
                    issues.append({"split": split, "image": img.name, "issue": "out_of_bounds",
                                   "x1": round(x1, 1), "y1": round(y1, 1),
                                   "x2": round(x2, 1), "y2": round(y2, 1), "W": W, "H": H})

    n_issues = len(issues)
    details = {
        "n_boxes": n_boxes,
        "n_images_checked": n_images_checked,
        "n_issues": n_issues,
        "issues_by_type": dict(Counter(i["issue"] for i in issues)),
        "examples": issues[:_MAX_DETAIL],
    }
    if n_boxes == 0:
        return CheckResult(_NAME, CheckSeverity.WARNING, "没有可检查的标注框", details)
    ratio = n_issues / n_boxes
    if ratio >= _ERROR_RATIO:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           f"标注问题比例高: {n_issues}/{n_boxes} ({ratio:.1%})", details)
    if n_issues > 0:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           f"{n_issues} 个标注问题 (共 {n_boxes} 框, {ratio:.1%})", details)
    return CheckResult(_NAME, CheckSeverity.PASS,
                       f"{n_boxes} 个标注框全部合法", details)
