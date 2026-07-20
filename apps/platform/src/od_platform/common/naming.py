#!/usr/bin/env python
# @File       : naming.py
# @Path       : apps/platform/src/od_platform/common/naming.py
# @Author     : 刘赋平
# @Date       : 2026-07-19 10:13:42
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/7/19 | 刘赋平 | v1.0.0 | 初始化创建

from __future__ import annotations

from pathlib import PurePath
from typing import Optional


def sanitize(text: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def model_stem(model: str) -> str:
    return sanitize(PurePath(model).stem)


def run_stem(*, stage: str, run_id: str, dataset: Optional[str] = None, model: Optional[str] = None) -> str:
    parts = [sanitize(stage.replace("_", "-")), run_id]
    if dataset:
        parts.append(sanitize(dataset))
    if model:
        parts.append(sanitize(model))
    return "-".join(parts)
