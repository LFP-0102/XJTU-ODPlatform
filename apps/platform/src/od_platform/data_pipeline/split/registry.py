#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :registry.py
# @Time      :2026/7/16 13:21:38
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from od_platform.common.constants import (
    DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE, DEFAULT_RANDOM_STATE
)

logger = logging.getLogger(__name__)

@dataclass
class SplitOptions:
    train_rate: float=  DEFAULT_TRAIN_RATE
    val_rate: float=  DEFAULT_VAL_RATE
    random_state: int= DEFAULT_RANDOM_STATE
    labels_per_image: Optional[Dict[str, List[str]]] = field(default=None)
    group_per_image: Optional[Dict[str, str]] = field(default=None)

StrategyFunc = Callable[[List[str], SplitOptions], Dict[str, List[str]]]

@dataclass(frozen=True)
class StrategyEntry:
    func: StrategyFunc
    requires_labels: bool = False

_STRATEGY_REGISTRY: Dict[str, StrategyEntry] = {}

def register_strategy(name: str, * , requires_labels: bool = False):
    def decorator(func: StrategyFunc) -> StrategyFunc:
        if name in _STRATEGY_REGISTRY:
            logger.warning(f"Strategy {name} already registered, will be overridden.")
            raise ValueError(f"格式{name!r} 被重复注册，已经注册有：{sorted(_STRATEGY_REGISTRY)}")
        _STRATEGY_REGISTRY[name] = StrategyEntry(func=func, requires_labels=requires_labels)
        return func
    return decorator

def get_strategy(name: str) -> StrategyEntry:
    _lazy_init()
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"格式{name!r} 被重复注册，已经注册有：{sorted(_STRATEGY_REGISTRY)}")
    return _STRATEGY_REGISTRY[name]

def list_strategies() -> Tuple[str, ...]:
    _lazy_init()
    return tuple(sorted(_STRATEGY_REGISTRY))


_LAZY_INITIALIZED = False
def _lazy_init() -> None:
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    from od_platform.common.registry_utils import import_submodules
    from od_platform.data_pipeline.split import strategies
    import_submodules(strategies)
    _LAZY_INITIALIZED = True

