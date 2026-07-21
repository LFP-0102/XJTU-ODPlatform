#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :registry_utils.py
# @Time      :2026/7/16 13:40:10
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType

def import_submodules(package: ModuleType) -> None:
    for m in pkgutil.iter_modules(package.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{m.name}")
