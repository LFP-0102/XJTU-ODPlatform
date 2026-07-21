#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :random_split.py
# @Time      :2026/7/16 14:13:39
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import annotations

import random
from typing import Dict, List

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.strategies._common import (seeded_shuffled, three_way_counts, validate_rates)
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy

@register_strategy(SplitStrategy.RANDOM, requires_labels=False)
def random_split(stems: List[str], options: SplitOptions) -> Dict[str, List[str]]:
    validate_rates(options.train_rate, options.val_rate)
    n = len(stems)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    rng = random.Random(options.random_state)
    shuffled = seeded_shuffled(sorted(stems), rng)
    n_train, n_val, _ = three_way_counts(n, options.train_rate, options.val_rate)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:]
    }