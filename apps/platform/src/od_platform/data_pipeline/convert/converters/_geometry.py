#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :_geometry.py
# @Time      :2026/7/15 13:27:35
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :共用几何：像素-> 归一化，文件名务必以下划线开头，这个和后面的扫描机制有关
from __future__ import annotations
from typing import List, Tuple

def normalize_box(x1: float, y1: float,
                x2: float, y2: float,
                W: float, H: float
                ) -> Tuple[float, float, float, float]:
    return (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H


def normalize_polygon(pts: List[float], W: float, H: float) -> List[float]:
    return [pts[i] / W if i%2 ==0 else pts[i] / H for i in range(len(pts))]


def read_image_size(path) -> "Tuple[int, int] | None":
    """纯 Python 读取 JPEG/PNG/BMP 的 (width, height),不依赖 PIL/cv2。

    DOTA / CreateML 等标注格式不含图像尺寸,归一化时需要从相邻图片读取。
    读不到或格式不支持返回 None。
    """
    from pathlib import Path
    p = Path(path)
    try:
        with p.open("rb") as f:
            head = f.read(32)
        if head[:2] == b"\xff\xd8":  # JPEG: 扫描 SOF 标记
            with p.open("rb") as f:
                f.read(2)
                while True:
                    marker = f.read(2)
                    if len(marker) < 2:
                        return None
                    if marker[0] != 0xff:
                        continue
                    if marker[1] in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
                                     0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
                        f.read(3)  # length(2) + precision(1)
                        h = int.from_bytes(f.read(2), "big")
                        w = int.from_bytes(f.read(2), "big")
                        return w, h
                    length = int.from_bytes(f.read(2), "big")
                    f.read(length - 2)
        elif head[:8] == b"\x89PNG\r\n\x1a\n":  # PNG: IHDR 在固定偏移
            w = int.from_bytes(head[16:20], "big")
            h = int.from_bytes(head[20:24], "big")
            return w, h
        elif head[:2] == b"BM":  # BMP: DIB header
            w = int.from_bytes(head[18:22], "little")
            h = abs(int.from_bytes(head[22:26], "little"))
            return w, h
    except Exception:
        return None
    return None


