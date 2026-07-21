#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :_common.py
# @Time      :2026/7/16 14:13:46
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import  annotations
import random
from typing import List, Sequence, Tuple, TypeVar


from od_platform.common.constants import RATE_EPSILON
T = TypeVar("T")

def validate_rates(train_rate: float, val_rate: float) -> float:
    test_rate = 1.0 - train_rate - val_rate
    if not (0 <= train_rate <= 1 and 0 <= val_rate <= 1 and -RATE_EPSILON <= test_rate <= 1):
        raise ValueError(f"比例越界：train_rate={train_rate:.4f}, val_rate={val_rate:.4f}, test_rate={test_rate:.4f}")
    return max(0.0, test_rate)

def three_way_counts(n: int, train_rate: float, val_rate: float) -> Tuple[int, int, int]:
    n_train = int(round(n * train_rate))
    n_val = int(round(n * val_rate))
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    return n_train, n_val, n - n_train - n_val

def seeded_shuffled(seq: Sequence[T], rng = random.Random) -> List[T]:
    out = list(seq)
    rng.shuffle(out)
    return out
