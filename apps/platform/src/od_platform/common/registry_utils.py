"""注册表通用工具:自动发现并 import 一个包下的所有实现模块,触发其中的 @register 副作用。"""
from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def import_submodules(package: ModuleType) -> None:
    """import package 下所有【非下划线】子模块,触发其中的 @register 注册。"""
    for m in pkgutil.iter_modules(package.__path__):
        if not m.name.startswith("_"):
            importlib.import_module(f"{package.__name__}.{m.name}")