#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : naming.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/common/naming.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-19 09:45:20
# @Version    : v1.0.0
# @Description:
#   [模块功能简述]
#   运行时产物命名规则
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026-07-19 | Matri | v1.0.0 | 初始化创建
from __future__ import annotations

from pathlib import PurePath
from typing import Optional

def sanitize(text: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)

def model_stem(model: str) -> str:
    return sanitize(PurePath(model).stem)


def run_stem(*,stage: str, run_id: str, dataset: Optional[str] = None, model: Optional[str] = None) -> str:
    parts = [sanitize(stage.replace("_", "-")), run_id]
    if dataset:
        parts.append(sanitize(dataset))
    if model:
        parts.append(sanitize(model))
    return "-".join(parts)

