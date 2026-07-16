"""split 子系统的划分策略注册表 + 统一参数包。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
)

logger = logging.getLogger(__name__)

@dataclass
class SplitOptions:
    train_rate: float = DEFAULT_TRAIN_RATE
    val_rate: float = DEFAULT_VAL_RATE
    random_state: int = DEFAULT_RANDOM_STATE
    labels_per_image: Optional[Dict[str, List[str]]] = field(default=None)
    group_per_image: Optional[Dict[str, str]] = field(default=None)

StrategyFunc = Callable[[List[str], SplitOptions], Dict[str, List[str]]]

@dataclass(frozen=True)
class StrategyEntry:
    """注册表里一条记录:策略函数 + 它的能力声明。"""
    func: StrategyFunc
    requires_labels: bool = False

_STRATEGY_REGISTRY: Dict[str, StrategyEntry] = {}

def register_strategy(name: str, *, requires_labels: bool = False):
    """装饰器:把划分策略登记进表。等价于手写 _STRATEGY_REGISTRY[name] = StrategyEntry(...)。"""
    def decorator(func: StrategyFunc) -> StrategyFunc:
        if name in _STRATEGY_REGISTRY:
            logger.warning("划分策略 %s 被重复注册,后者覆盖前者", name)
            raise ValueError(f"策略{name!r} 被重复注册，已经注册的有: {sorted(_STRATEGY_REGISTRY.keys())}")
        _STRATEGY_REGISTRY[name] = StrategyEntry(func=func, requires_labels=requires_labels)
        return func
    return decorator

def get_strategy(name: str) -> StrategyEntry:
    """按名取策略条目(先触发自动发现)。Raises ValueError:未注册的策略。"""
    _lazy_init()
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"未注册的划分策略: {name!r}。已注册: {sorted(_STRATEGY_REGISTRY)}")
    return _STRATEGY_REGISTRY[name]

def list_strategies() -> Tuple[str, ...]:
    """当前已注册的策略名(先触发自动发现)。"""
    _lazy_init()
    return tuple(sorted(_STRATEGY_REGISTRY))


_LAZY_INITIALIZED = False

def _lazy_init() -> None:
    """扫描 strategies/*.py 触发 @register_strategy。"""
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    from od_platform.common.registry_utils import import_submodules
    from od_platform.data_pipeline.split import strategies
    import_submodules(strategies)
    _LAZY_INITIALIZED = True

