from __future__ import annotations

import logging
from typing import Dict, List, Optional

from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
)
from od_platform.data_pipeline.split.registry import SplitOptions, get_strategy

logger = logging.getLogger(__name__)

def split_dataset(
    stems: List[str],
    train_rate: float = DEFAULT_TRAIN_RATE,
    val_rate: float = DEFAULT_VAL_RATE,
    random_state: int = DEFAULT_RANDOM_STATE,
    *,
    strategy: str = DEFAULT_SPLIT_STRATEGY,
    labels_per_image: Optional[Dict[str, List[str]]] = None,
    group_per_image: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:

    entry = get_strategy(strategy)
    # 能力声明驱动的 fail-fast:需要标签的策略(stratified*)没拿到标签,当场报清楚。
    if entry.requires_labels and labels_per_image is None:
        raise ValueError(f"划分策略 {strategy!r} 需要 labels_per_image,但未提供。")
    options = SplitOptions(
        train_rate=train_rate, val_rate=val_rate, random_state=random_state,
        labels_per_image=labels_per_image, group_per_image=group_per_image,
    )
    result = entry.func(stems, options)
    logger.info("划分完成 strategy=%s 规模=%s", strategy,
                {k: len(v) for k, v in result.items()})
    return result