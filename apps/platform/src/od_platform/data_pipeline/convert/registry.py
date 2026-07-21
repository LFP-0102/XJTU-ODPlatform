#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :registry.py
# @Time      :2026/7/15 13:06:14
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  : 一张表：{格式明：函数} + 登记装饰器
from __future__ import  annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ConvertOptions:
    task: str = "detect"
    classes: Optional[List[str]] = field(default=None)

# 1. 统一接口定义
ConverterFunc = Callable[[Path, Path, ConvertOptions], List[str]]


@dataclass(frozen=True)
class ConverterEntry:
    """表里面一条记录： 实现的函数 + 它能干什么task声明"""
    func: ConverterFunc
    supported_tasks: Tuple[str, ...]

    def supports(self, task: str) -> bool:
        return task in self.supported_tasks

# 2. 注册表本体
_CONVERTERS: Dict[str, ConverterEntry] = {}

# 3. 登记装饰器
def register_converter(format_name: str, *, supported_tasks: Tuple[str, ...]):

    def decorator(func: ConverterFunc) -> ConverterFunc:
        if format_name in _CONVERTERS:
            raise ValueError(f"格式{format_name!r}被重复注册，已经注册的有：{sorted(_CONVERTERS)}")
        _CONVERTERS[format_name] = ConverterEntry(func, tuple(supported_tasks))
        return func
    return decorator

# 4. 查询接口
def get_converters(format_name: str) -> ConverterEntry:
    """根据格式名获取对应的转换函数"""
    _lazy_init()
    if format_name not in _CONVERTERS:
        raise ValueError(
            f"未注册的格式：{format_name}, 已经注册的格式有：{sorted(_CONVERTERS)}"
        )
    return _CONVERTERS[format_name]


# 5. 查询接口2： 返回当前已注册的格式名列表
def list_converters() -> Tuple[str,...]:
    _lazy_init()
    """返回当前已注册的格式名列表"""
    return tuple(sorted(_CONVERTERS))

_LAZY_INITIALIZED = False
def _lazy_init() -> None:
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    from od_platform.common.registry_utils import import_submodules
    from od_platform.data_pipeline.convert import converters
    import_submodules(converters)
    _LAZY_INITIALIZED = True


