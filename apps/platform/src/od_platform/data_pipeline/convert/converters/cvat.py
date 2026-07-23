#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : cvat.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_pipeline/convert/converters/cvat.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:CVAT XML -> YOLO 转换器(detect + segment)
"""CVAT XML -> YOLO 转换器(detect + segment)。

支持 CVAT 标注工具导出的 XML,处理其中的 <image> 元素(图像标注模式):
  <annotations>
    <image id="0" name="img1.jpg" width="W" height="H">
      <box label="cat" xtl="10" ytl="20" xbr="100" ybr="200"/>
      <polygon label="dog" points="1.0,2.0;3.0,4.0;5.0,6.0"/>
    </image>
    ...
  </annotations>

- detect:box 元素 -> YOLO bbox(xtl/ytl/xbr/ybr 像素角点)。
- segment:polygon 元素 -> YOLO 多边形(points 用分号分隔的 x,y 对)。
- <track>(视频轨迹模式)暂不处理:静态检测用不到,遇到静默跳过。
- 每张图的尺寸来自 <image> 的 width/height 属性,无需读图片。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter
from od_platform.data_pipeline.convert.converters._geometry import normalize_box, normalize_polygon


@register_converter(AnnotationFormat.CVAT, supported_tasks=(Task.DETECT, Task.SEGMENT))
def convert_cvat(input_dir: Path, out: Path, options: ConvertOptions) -> List[str]:
    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"CVAT: 目录下没有 xml 文件: {input_dir}")

    names: List[str] = list(options.classes) if options.classes else []
    discovering = options.classes is None
    seg = options.task == Task.SEGMENT
    out.mkdir(parents=True, exist_ok=True)

    for xp in xml_files:
        root = ET.parse(xp).getroot()
        for img_el in root.iter("image"):
            W = float(img_el.get("width", 0) or 0)
            H = float(img_el.get("height", 0) or 0)
            name = img_el.get("name") or img_el.get("id")
            if not name or W <= 0 or H <= 0:
                continue
            stem = Path(name).stem
            lines: List[str] = []

            for box in img_el.findall("box"):
                nm = box.get("label")
                if nm not in names:
                    if discovering:
                        names.append(nm)
                    else:
                        continue
                cid = names.index(nm)
                x1, y1 = float(box.get("xtl")), float(box.get("ytl"))
                x2, y2 = float(box.get("xbr")), float(box.get("ybr"))
                cx, cy, bw, bh = normalize_box(x1, y1, x2, y2, W, H)
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if seg:
                for poly in img_el.findall("polygon"):
                    nm = poly.get("label")
                    if nm not in names:
                        if discovering:
                            names.append(nm)
                        else:
                            continue
                    cid = names.index(nm)
                    pts: List[float] = []
                    for pt in (poly.get("points", "") or "").split(";"):
                        xy = pt.split(",")
                        if len(xy) == 2:
                            pts.extend([float(xy[0]), float(xy[1])])
                    if len(pts) >= 6:
                        p = normalize_polygon(pts, W, H)
                        lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in p))

            (out / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return names
