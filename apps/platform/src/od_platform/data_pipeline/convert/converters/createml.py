#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : createml.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_pipeline/convert/converters/createml.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:CreateML JSON -> YOLO 转换器(detect)
"""CreateML JSON -> YOLO 转换器(detect only)。

苹果 CreateML 格式:一个 JSON 文件,顶层是数组,每项:
  {
    "image": "img1.jpg",
    "annotations": [
      {"label": "cat", "coordinates": {"x": cx, "y": cy, "width": w, "height": h}}
    ]
  }

注意:CreateML 的 coordinates 是【中心点 + 宽高】(像素),不是角点;且 JSON 本身
不含图像尺寸。归一化需要的 W/H 从相邻图片读取(候选路径:父目录/images/、父目录、
input_dir 下同名文件),用纯 Python 读 JPEG/PNG/BMP 头,无第三方依赖。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter
from od_platform.data_pipeline.convert.converters._geometry import normalize_box, read_image_size


@register_converter(AnnotationFormat.CREATEML, supported_tasks=(Task.DETECT,))
def convert_createml(input_dir: Path, out: Path, options: ConvertOptions) -> List[str]:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"CreateML: 目录下没有 json 文件: {input_dir}")

    names: List[str] = list(options.classes) if options.classes else []
    discovering = options.classes is None
    images_dir = input_dir.parent / "images"
    out.mkdir(parents=True, exist_ok=True)

    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("annotations") or data.get("images") or [data]
        if not isinstance(data, list):
            continue

        for item in data:
            img_name = item.get("image") or item.get("file_name") or item.get("filename")
            if not img_name:
                continue
            stem = Path(img_name).stem

            W = H = 0
            for cand in (input_dir.parent / img_name, images_dir / img_name, input_dir / img_name):
                if cand.exists():
                    sz = read_image_size(cand)
                    if sz:
                        W, H = sz
                        break

            lines: List[str] = []
            for a in item.get("annotations", []) or []:
                nm = a.get("label")
                if nm not in names:
                    if discovering:
                        names.append(nm)
                    else:
                        continue
                cid = names.index(nm)
                co = a.get("coordinates", a)
                cx, cy = float(co.get("x", 0)), float(co.get("y", 0))
                bw, bh = float(co.get("width", 0)), float(co.get("height", 0))
                if W <= 0 or H <= 0 or bw <= 0 or bh <= 0:
                    continue
                x1, y1 = cx - bw / 2, cy - bh / 2
                x2, y2 = cx + bw / 2, cy + bh / 2
                nx, ny, nbw, nbh = normalize_box(x1, y1, x2, y2, float(W), float(H))
                lines.append(f"{cid} {nx:.6f} {ny:.6f} {nbw:.6f} {nbh:.6f}")

            (out / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return names
