#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :registry_autoloads.py
# @Time      :2026/7/15 14:19:45
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from pathlib import Path
from od_platform.data_pipeline.convert.service import convert_data_to_yolo

names = convert_data_to_yolo(
    input_dir=Path(r"C:\Users\Matri\Desktop\XJTU-ODPlatfrom\data\raw\MRI_PASCAL\annotations"),
    output_labels_dir=Path(r"C:\Users\Matri\Desktop\XJTU-ODPlatfrom\data\processed"),
    annotation_format = "pascal_voc"
)

print(names)
