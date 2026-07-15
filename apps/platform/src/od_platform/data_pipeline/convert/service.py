"""convert 调度层(门)—— 几行代码接住所有格式。永不增长:只查表分发。"""
from __future__ import annotations

from pathlib import Path
from typing import List

from od_platform.data_pipeline.convert.registry import get_converter,ConvertOptions


def convert_data_to_yolo(
    input_dir: Path,
    output_labels_dir: Path,
    annotation_format: str,
    options: ConvertOptions,
) -> List[str]:
    entry = get_converter(annotation_format)
    if not entry.supports(options.task):
        raise ValueError(
            f"格式 {annotation_format!r} 不支持 task={options.task!r}。支持: {entry.supported_tasks}")
    return entry.func(input_dir, output_labels_dir, options)