#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : group_split.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_pipeline/split/strategies/group_split.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.1.0
# @Description:按组划分(同 group 整体进同一 split,防泄漏)
"""按组划分:同一 group 的样本必须整体进同一个 split。

典型场景:视频帧序列(同视频的帧不能跨 train/val,否则评估会因相邻帧近似而虚高)、
同场景多角度拍摄、同一病人的多张影像。group_per_image: {stem: group_id}。

算法:把样本按 group 聚合,对 group 列表洗牌,按 group 累计样本数依次填进 train/val/test
的容量槽。无 group 信息的样本每图自成一组(退化为 random,但仍是组级 random)——
此时会发 warning 提醒,避免用户误以为 group 生效。
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.strategies._common import (
    seeded_shuffled, three_way_counts, validate_rates)
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy

logger = logging.getLogger(__name__)
_SPLITS = ("train", "val", "test")


@register_strategy(SplitStrategy.GROUP, requires_labels=False)
def group_split(stems: List[str], options: SplitOptions) -> Dict[str, List[str]]:
    validate_rates(options.train_rate, options.val_rate)
    n = len(stems)
    if n == 0:
        return {s: [] for s in _SPLITS}
    n_train, n_val, _ = three_way_counts(n, options.train_rate, options.val_rate)

    group_per_image = options.group_per_image or {}
    if not group_per_image:
        logger.warning(
            "⚠️ group 策略未收到 group 信息(--group-by-prefix / --groups-file 未提供),"
            "每图自成一组,实际等价于 random 划分。若需防泄漏分组,请指定 group 来源。")

    groups: Dict[str, List[str]] = defaultdict(list)
    for s in stems:
        g = group_per_image.get(s, s)  # 无 group 信息:每图自成一组
        groups[g].append(s)

    rng = random.Random(options.random_state)
    group_keys = seeded_shuffled(sorted(groups.keys()), rng)

    result: Dict[str, List[str]] = {sp: [] for sp in _SPLITS}
    counts = {"train": 0, "val": 0, "test": 0}
    for g in group_keys:
        members = groups[g]
        if counts["train"] < n_train:
            sp = "train"
        elif counts["val"] < n_val:
            sp = "val"
        else:
            sp = "test"
        result[sp].extend(members)
        counts[sp] += len(members)
    return result
