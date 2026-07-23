#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : stratified_multilabel.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_pipeline/split/strategies/stratified_multilabel.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.0.0
# @Description:多标签迭代分层划分(Sechidis 2011 简化版)
"""多标签迭代分层划分(Sechidis 2011 简化版)。

每张图可能有多个标签(labels_per_image: {stem: [类名]})。目标:让 train/val/test
每个 split 里各类的比例尽量一致,而不是纯 random 把稀缺类全塞某一组。

算法(稀缺类优先的迭代分配):
  1. 按类的文档频率升序处理(稀缺类先分,避免最后没得选)。
  2. 对每个类 c,把"含 c 且未分配"的样本洗牌,逐个分配到"该类当前最缺"的 split
     (期望数 = 该类总数 × split 比例 - 该 split 已有该类数),同时尊重 split 容量。
  3. 剩余无标签或未分配的样本按剩余容量随机分。

requires_labels=True:必须传 labels_per_image,否则 split_dataset 会拒绝。
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.strategies._common import (
    seeded_shuffled, three_way_counts, validate_rates)
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy

_SPLITS = ("train", "val", "test")


@register_strategy(SplitStrategy.STRATIFIED_MULTILABEL, requires_labels=True)
def stratified_multilabel_split(stems: List[str], options: SplitOptions) -> Dict[str, List[str]]:
    validate_rates(options.train_rate, options.val_rate)
    n = len(stems)
    if n == 0:
        return {s: [] for s in _SPLITS}
    n_train, n_val, _ = three_way_counts(n, options.train_rate, options.val_rate)
    capacity = {"train": n_train, "val": n_val, "test": n - n_train - n_val}

    labels_per_image = options.labels_per_image or {}
    label_sets = {s: set(labels_per_image.get(s, [])) for s in stems}

    doc_freq = Counter()
    for s in stems:
        for c in label_sets[s]:
            doc_freq[c] += 1

    rng = random.Random(options.random_state)
    assignment: Dict[str, str] = {}
    class_counts = {sp: defaultdict(int) for sp in _SPLITS}
    remaining = dict(capacity)

    # 稀缺类优先,逐类把含该类的样本按 deficit 分配
    for cls in sorted(doc_freq, key=lambda c: doc_freq[c]):
        candidates = [s for s in stems if cls in label_sets[s] and s not in assignment]
        rng.shuffle(candidates)
        desired = {sp: doc_freq[cls] * (capacity[sp] / n) for sp in _SPLITS}

        for s in candidates:
            best_sp, best_deficit = None, None
            for sp in _SPLITS:
                if remaining[sp] <= 0:
                    continue
                deficit = desired[sp] - class_counts[sp][cls]
                if best_deficit is None or deficit > best_deficit:
                    best_deficit, best_sp = deficit, sp
            if best_sp is None:
                best_sp = max(_SPLITS, key=lambda sp: remaining[sp])
            assignment[s] = best_sp
            remaining[best_sp] -= 1
            class_counts[best_sp][cls] += 1

    # 剩余(无标签 / 未分配)按剩余容量随机分
    leftover = seeded_shuffled([s for s in stems if s not in assignment], rng)
    for s in leftover:
        sp = max(_SPLITS, key=lambda x: remaining[x])
        if remaining[sp] <= 0:
            sp = "test"
        assignment[s] = sp
        remaining[sp] -= 1

    result: Dict[str, List[str]] = {sp: [] for sp in _SPLITS}
    for s in stems:
        result[assignment[s]].append(s)
    return result
