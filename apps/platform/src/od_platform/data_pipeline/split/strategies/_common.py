"""划分策略公用的纯函数(下划线开头 → 自动发现会跳过它,因为它不是一个"策略")。

设计原则:无 IO、无随机副作用。随机由【调用方传入的 Random 实例】控制,不碰全局 random
—— 这样同一个种子在哪里调用都得到同一序列,是可复现性的基石。
"""
from __future__ import annotations

import random
from typing import List, Sequence, Tuple, TypeVar

from od_platform.common.constants import RATE_EPSILON

T = TypeVar("T")


def validate_rates(train_rate: float, val_rate: float) -> float:
    """校验比例合法,并返回推算出的 test_rate(= 1 - train - val)。

    Raises:
        ValueError: 任一比例越界。用 RATE_EPSILON 吸收浮点误差,避免正常的 70/30 被误杀。
    """
    test_rate = 1.0 - train_rate - val_rate
    if not (0 <= train_rate <= 1 and 0 <= val_rate <= 1 and -RATE_EPSILON <= test_rate <= 1):
        raise ValueError(f"比例越界: train={train_rate}, val={val_rate}, test={test_rate:.4f}")
    return max(0.0, test_rate)               # 钳掉 -5e-17 这种浮点噪声,保证非负


def three_way_counts(n: int, train_rate: float, val_rate: float) -> Tuple[int, int, int]:
    """把 n 个样本按比例切成 (n_train, n_val, n_test)。

    四舍五入 + 钳位,保证三者非负且求和恰好等于 n(不会因取整丢样本)。
    极小 n 自然退化:n=1 → 全进 train;n=2 → train=2(val/test 摊不到,符合预期)。
    """
    n_train = int(round(n * train_rate))
    n_val = int(round(n * val_rate))
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    return n_train, n_val, n - n_train - n_val   # test 拿剩下的,保证总和 = n


def seeded_shuffled(seq: Sequence[T], rng: random.Random) -> List[T]:
    """返回 seq 的洗牌副本(不改原序列;随机性完全由传入的 rng 决定)。"""
    out = list(seq)
    rng.shuffle(out)
    return out