from __future__ import annotations
import logging
from dataclasses import dataclass,field
from typing import Any,Callable,Dict,List
from pathlib import Path
from od_platform.data_validation.snapshot import DatasetSnapshot

logger = logging.getLogger(__name__)


class CheckSeverity:
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    PASS = "PASS"

    _ORDER = {"PASS": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}

    @classmethod
    def rank(cls, level: str) -> int:
        return cls._ORDER.get(level,0)


@dataclass
class CheckResult:
    name: str
    severity: str
    summary: str
    details:Dict[str,Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.severity in (CheckSeverity.PASS,CheckSeverity.INFO)

@dataclass(frozen=True)
class CheckEntry:
    name: str
    func:Callable[[CheckContext],CheckResult]

_REGISTRY: Dict[str, CheckEntry] = {}

def check(name: str) -> Callable:
    def decorator(func: Callable[[CheckContext],CheckResult]) -> Callable:
        if name in _REGISTRY:
            logger.warning(f"Strategy{name} already registered,will be overwritten")
            raise ValueError(f"格式{name!r} 被重复注册，已经注册的有: {sorted(_REGISTRY)}")
        _REGISTRY[name] = CheckEntry(name=name,func=func)
        return func
    return decorator

_LAZY_INITIALIZED: bool = False

def _lazy_init() -> None:
    global _LAZY_INITIALIZED
    if _LAZY_INITIALIZED:
        return
    from od_platform.common.registry_utils import import_submodules
    from od_platform.data_validation import checks
    import_submodules(checks)
    _LAZY_INITIALIZED = True

def get_all_checks() -> List[CheckEntry]:
    _lazy_init()
    return list(_REGISTRY.values())

def get_check(name: str) -> CheckEntry:
    _lazy_init()
    if name not in _REGISTRY:
        raise  KeyError(f"check {name} 没有注册，已经注册的有:{list(_REGISTRY)}")
    return _REGISTRY[name]

def list_check_names() -> List[str]:
    _lazy_init()
    return list(_REGISTRY.keys())

@dataclass
class CheckContext:
    """check 函数的入参合同——签名 (ctx: CheckContext) -> CheckResult 永不变。

    yaml_path: 数据集 yaml 路径
    snapshot : 一次扫描的产物(阶段 4 加入)——所有要碰数据的 check 从这里取,不再自己扫盘
    """
    yaml_path: Path
    snapshot: "DatasetSnapshot"






