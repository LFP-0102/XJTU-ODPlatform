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


