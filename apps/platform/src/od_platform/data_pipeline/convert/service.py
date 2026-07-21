#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :service.py
# @Time      :2026/7/15 14:16:00
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import  annotations

from pathlib import Path
from typing import  List

from od_platform.data_pipeline.convert.registry import get_converters, ConvertOptions

def convert_data_to_yolo(
        input_dir: Path,
        output_labels_dir:Path,
        annotation_format: str,
        options: ConvertOptions,
    ) -> List[str]:
    entry = get_converters(annotation_format)
    if not entry.supports(options.task):
        raise ValueError(
            f"格式{annotation_format}不支持任务{options.task}，支持的格式有{entry.supported_tasks}"
        )
    return entry.func(input_dir, output_labels_dir, options)