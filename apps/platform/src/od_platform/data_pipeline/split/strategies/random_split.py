from __future__ import annotations

import random
from typing import Dict, List

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.strategies._common import (
    seeded_shuffled, three_way_counts, validate_rates,
)
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy

@register_strategy(SplitStrategy.RANDOM, requires_labels=False)   # 随机不需要类别信息
def random_split(stems: List[str], options: SplitOptions) -> Dict[str, List[str]]:
    """把 stems 随机切成 train/val/test。返回 {"train"/"val"/"test": [stem, ...]}。"""
    validate_rates(options.train_rate, options.val_rate)   # 只为尽早校验比例;返回值这里用不到
    n = len(stems)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(options.random_state)
    # 先 sorted 把输入顺序固定下来,再用固定种子洗牌 —— 两者缺一不可,否则不可复现
    # (光固定种子,若输入顺序本身随机,结果照样会变)。
    shuffled = seeded_shuffled(sorted(stems), rng)
    n_train, n_val, _ = three_way_counts(n, options.train_rate, options.val_rate)
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }
