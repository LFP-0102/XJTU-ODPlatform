#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : stratified_weighted.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_pipeline/split/strategies/stratified_weighted.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:按主类逆频率加权的分层划分
"""按主类逆频率加权的分层划分(单标签语义)。

每张图取"主类"(labels_per_image[stem] 的第一个类;无标签归为 "_none"),按主类
分层:每个类内部独立洗牌后按 train/val/test 比例切,再合并。这样各类在三个 split
里的比例都与整体比例一致,稀缺类不会被随机稀释——"加权"体现在稀缺类也严格按
自身比例切分,而不是被多数类淹没。

与 stratified_multilabel 的区别:multilabel 处理多标签(一图多类,迭代分配);
weighted 处理单主类(一图一类,按类分层 random),适合单类目标检测数据集。
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.strategies._common import (
    seeded_shuffled, three_way_counts, validate_rates)
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy

_SPLITS = ("train", "val", "test")
_NONE = "_none"


@register_strategy(SplitStrategy.STRATIFIED_WEIGHTED, requires_labels=True)
def stratified_weighted_split(stems: List[str], options: SplitOptions) -> Dict[str, List[str]]:
    validate_rates(options.train_rate, options.val_rate)
    n = len(stems)
    if n == 0:
        return {s: [] for s in _SPLITS}
    _ = three_way_counts(n, options.train_rate, options.val_rate)  # 触发比例校验

    labels_per_image = options.labels_per_image or {}
    by_class: Dict[str, List[str]] = defaultdict(list)
    for s in stems:
        labs = labels_per_image.get(s) or []
        primary = labs[0] if labs else _NONE
        by_class[primary].append(s)

    rng = random.Random(options.random_state)
    result: Dict[str, List[str]] = {sp: [] for sp in _SPLITS}

    # 每类内部按比例切,保证各类在三集的比例一致
    for cls in sorted(by_class):
        members = seeded_shuffled(by_class[cls], rng)
        m = len(members)
        c_train = int(round(m * options.train_rate))
        c_val = int(round(m * options.val_rate))
        c_train = max(0, min(c_train, m))
        c_val = max(0, min(c_val, m - c_train))
        result["train"].extend(members[:c_train])
        result["val"].extend(members[c_train:c_train + c_val])
        result["test"].extend(members[c_train + c_val:])

    # 各 split 内部再洗一次,打乱类间顺序
    for sp in _SPLITS:
        result[sp] = seeded_shuffled(result[sp], rng)
    return result
