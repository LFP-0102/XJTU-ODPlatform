#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : labelme.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_pipeline/convert/converters/labelme.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:LabelMe JSON -> YOLO 转换器(detect + segment)
"""LabelMe JSON -> YOLO 转换器(detect + segment)。

LabelMe 每张图一个同名 JSON,结构:
  {
    "imagePath": "xxx.jpg", "imageWidth": W, "imageHeight": H,
    "shapes": [
      {"label": "cat", "shape_type": "rectangle", "points": [[x1,y1],[x2,y2]]},
      {"label": "dog", "shape_type": "polygon", "points": [[x1,y1],...]}
    ]
  }

- detect:rectangle 取两角点;polygon 取外接矩形。
- segment:polygon 直接归一化;rectangle 不参与(无多边形信息)。
- 类名从 shapes[].label 探测(三态 classes 语义与 VOC/COCO 一致)。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter
from od_platform.data_pipeline.convert.converters._geometry import normalize_box, normalize_polygon


@register_converter(AnnotationFormat.LABELME, supported_tasks=(Task.DETECT, Task.SEGMENT))
def convert_labelme(input_dir: Path, out: Path, options: ConvertOptions) -> List[str]:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"LabelMe: 目录下没有 json 文件: {input_dir}")

    names: List[str] = list(options.classes) if options.classes else []
    discovering = options.classes is None
    seg = options.task == Task.SEGMENT
    out.mkdir(parents=True, exist_ok=True)

    for jp in json_files:
        data = json.loads(jp.read_text(encoding="utf-8"))
        W = float(data.get("imageWidth", 0) or 0)
        H = float(data.get("imageHeight", 0) or 0)
        if W <= 0 or H <= 0:
            continue
        lines: List[str] = []
        for sh in data.get("shapes", []):
            nm = sh.get("label")
            pts = sh.get("points", []) or []
            stype = sh.get("shape_type", "")
            if nm not in names:
                if discovering:
                    names.append(nm)
                else:
                    continue
            cid = names.index(nm)

            if seg and stype == "polygon" and len(pts) >= 3:
                flat = [float(v) for pt in pts for v in pt]
                p = normalize_polygon(flat, W, H)
                lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in p))
            else:
                if stype == "rectangle" and len(pts) >= 2:
                    (x1, y1), (x2, y2) = pts[0], pts[1]
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                elif len(pts) >= 2:
                    xs = [float(p[0]) for p in pts]
                    ys = [float(p[1]) for p in pts]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                else:
                    continue
                cx, cy, bw, bh = normalize_box(x1, y1, x2, y2, W, H)
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out / (jp.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return names
