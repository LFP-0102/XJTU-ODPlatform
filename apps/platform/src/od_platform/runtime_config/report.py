from __future__ import annotations

import logging
from typing import Any, List, Tuple, Optional

from torch.masked import mean
from torchvision.tv_tensors import _keypoints

from od_platform.common.string_utils import pad_to_width

_KEY_WIDTH = 26
_WIDTH = 60


def _chain_oldest_first(meta: Any) -> List[Any]:
    return list(reversed(meta.chain()))


def _was_overridden(meta: Any) -> bool:
    chain = _chain_oldest_first(meta)
    return any(chain[i].value != chain[i - 1].value for i in range(1, len(chain)))


def render_effective(config: Any, merger: Any, *, width: int = _WIDTH) -> List[str]:
    lines = ['=' * width, '配置参数信息'.center(width), "-" * width]
    for name in type(config).model_fields:
        value = getattr(config, name)
        meta = merger.get_metadata(name)
        source = meta.source_label if meta is not None else "DEFAULT"
        lines.append(f"{pad_to_width(name, _KEY_WIDTH)}: {value} (来源: {source})")
    return lines


# 配置溯源报告-覆盖链
def render_overrides(config: Any, merger: Any, *, width: int = _WIDTH) -> List[str]:
    changed: List[Tuple[str, str]] = []
    for name in type(config).model_fields:
        meta = merger.get_metadata(name)
        if meta is not None and _was_overridden(meta):
            chain = _chain_oldest_first(meta)
            changed.append((name, " → ".join(f"{m.value}({m.source_label})" for m in chain)))

    total = len(type(config).model_fields)
    lines = ["-" * width, "配置覆盖情况".center(width), "-" * width]
    if not changed:
        lines.append(f"没有字段被改写:  {total}各个字段取值一致")
        return lines

    lines.append(f"{len(changed)}个字段在合并中被改写:  其余{total - len(changed)}个字段取值一致, 未列出")
    lines += [f"{pad_to_width(n, _KEY_WIDTH)}: {c}" for n, c in changed]
    return lines


def log_config_report(config: Any, merger: Any, logger: Optional[logging.Logger] = None) -> None:
    log = logger or logging.getLogger(__name__)
    for line in render_effective(config, merger):
        log.info(line)
    for line in render_overrides(config, merger):
        log.info(line)