#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : dota.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_pipeline/convert/converters/dota.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:DOTA 旋转框 -> YOLO 转换器(detect)
"""DOTA 旋转框 -> YOLO 转换器(detect only)。

DOTA 每张图一个同名 txt,每行:
  x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult

四个顶点是旋转框的像素坐标(顺时针/逆时针均可)。本转换器取四点的外接水平
矩形(min/max)作为 YOLO bbox 输出——适配 detect 任务。OBB(旋转框训练)不在
本引擎 Task 范围内。

尺寸获取(归一化必须):DOTA txt 本身不含图像尺寸,按优先级:
  1. txt 内的 "image size: W H" / "imagesize: W H" 行(DOTA v2 常见);
  2. 相邻 images 目录下的同名图片(纯 Python 读 JPEG/PNG/BMP 头)。
两者都拿不到时该样本跳过(不输出)。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, IMAGE_EXTENSIONS, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter
from od_platform.data_pipeline.convert.converters._geometry import normalize_box, read_image_size

_SIZE_RE = re.compile(r"\s*image\s*size\s*:\s*(\d+)\s+(\d+)", re.IGNORECASE)


@register_converter(AnnotationFormat.DOTA, supported_tasks=(Task.DETECT,))
def convert_dota(input_dir: Path, out: Path, options: ConvertOptions) -> List[str]:
    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"DOTA: 目录下没有 txt 文件: {input_dir}")

    names: List[str] = list(options.classes) if options.classes else []
    discovering = options.classes is None
    images_dir = input_dir.parent / "images"
    out.mkdir(parents=True, exist_ok=True)

    for tp in txt_files:
        raw_lines = tp.read_text(encoding="utf-8").splitlines()

        W = H = 0
        for ln in raw_lines:
            m = _SIZE_RE.match(ln)
            if m:
                W, H = int(m.group(1)), int(m.group(2))
                break
        if W <= 0 or H <= 0:
            for ext in IMAGE_EXTENSIONS:
                cand = images_dir / (tp.stem + ext)
                if cand.exists():
                    sz = read_image_size(cand)
                    if sz:
                        W, H = sz
                        break

        out_lines: List[str] = []
        for ln in raw_lines:
            ln = ln.strip()
            if not ln or _SIZE_RE.match(ln):
                continue
            parts = ln.split()
            if len(parts) < 9:
                continue
            try:
                coords = [float(x) for x in parts[:8]]
            except ValueError:
                continue
            nm = parts[8]
            if nm not in names:
                if discovering:
                    names.append(nm)
                else:
                    continue
            if W <= 0 or H <= 0:
                continue
            cid = names.index(nm)
            xs = coords[0::2]
            ys = coords[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            cx, cy, bw, bh = normalize_box(x1, y1, x2, y2, float(W), float(H))
            out_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out / (tp.stem + ".txt")).write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    return names
