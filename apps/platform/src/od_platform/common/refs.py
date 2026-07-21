#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :refs.py
# @Time      :2026/7/16 15:01:45
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom\apps\platform\src\od_platform\common\refs.py
# @Function  :引用解析，把命令行给的参数解析成实际路径，
from __future__ import annotations
from pathlib import Path
from typing import Optional

from od_platform.common.paths import (
    PROCESS_DATA_DIR, RAW_DATA_DIR,CONFIG_DIR, RUNTIME_CONFIGS_DIR, TRAINED_MODELS_DIR
)

def resolve_ref(ref: str, * ,base_dir: Path, default_suffix: Optional[str] = None) -> Path:
    p = Path(ref)
    if p.is_absolute() or  len(p.parts) > 1:
        return p.resolve()
    name = ref if (not default_suffix or ref.endswith(default_suffix)) else ref + default_suffix
    return (base_dir / name).resolve()

def resolve_dataset(ref: str) -> Path:
    return resolve_ref(ref, base_dir = RAW_DATA_DIR)

def resolve_dataset_yaml(ref: str) -> Path:
    dataset_yaml = CONFIG_DIR / "datasets"
    return resolve_ref(ref, base_dir = dataset_yaml, default_suffix = ".yaml")

def resolve_config_yaml(ref: str) -> Path:
    return resolve_ref(ref, base_dir = RUNTIME_CONFIGS_DIR, default_suffix=".yaml")


# ultralytics 能吃的推理权重/导出格式后缀. 模型名不是只有 .pt 一种,
# 还可能是 onnx / tensorrt engine / torchscript 等, 不能把后缀写死.
_MODEL_SUFFIXES: tuple[str, ...] = (
    ".pt", ".onnx", ".engine", ".torchscript",
    ".mlmodel", ".mlpackage", ".xml", ".pb", ".tflite", ".param",
)

def resolve_model(ref: str) -> Path:
    """把模型引用解析成实际路径(不做存在性检查, 由调用方决定找不到时怎么办).

    规则:
      - 绝对路径 / 带目录分隔的相对路径 → 原样解析(用户显式给了位置就尊重它,
        推理时用户也可能直接传一个绝对路径);
      - 裸文件名 → 落到 models/trained/ 下(推理跑的是训练好的权重);
      - 后缀: 已经是已知模型格式(.pt/.onnx/.engine/...)就保持不动;
              完全没写后缀时才补默认 .pt —— 不再无脑给 foo.onnx 拼成 foo.onnx.pt.
    """
    p = Path(ref)
    if p.is_absolute() or len(p.parts) > 1:
        return p.resolve()
    name = ref if p.suffix.lower() in _MODEL_SUFFIXES else ref + ".pt"
    return (TRAINED_MODELS_DIR / name).resolve()


