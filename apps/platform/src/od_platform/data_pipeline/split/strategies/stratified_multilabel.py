"""L1 策略:多标签迭代分层(自实现,不依赖第三方库)。"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Set

from od_platform.common.constants import SplitStrategy
from od_platform.data_pipeline.split.strategies._common import validate_rates
from od_platform.data_pipeline.split.registry import SplitOptions, register_strategy

_EPS = 1e-9   # 比较浮点期望值时的容差


def _argmax_tiebreak(primary: Sequence[float], secondary: Sequence[float], rng: random.Random) -> int:
    """返回使 primary 最大的下标;并列时看 secondary 最大;再并列则(seeded)随机。"""
    pmax = max(primary)
    cand = [j for j, v in enumerate(primary) if v >= pmax - _EPS]
    if len(cand) > 1:
        smax = max(secondary[j] for j in cand)
        cand = [j for j in cand if secondary[j] >= smax - _EPS]
    return rng.choice(sorted(cand)) if len(cand) > 1 else cand[0]


def iterative_stratify(sample_labels: List[Set[str]], proportions: Sequence[float],
                       random_state: int) -> List[int]:
    """把 N 个多标签样本分到 len(proportions) 个子集,尽量保持每个标签的占比。

    Returns:
        fold[i] = 第 i 个样本被分到的子集下标(0/1/2 对应 train/val/test)。
    """
    n = len(sample_labels)
    k = len(proportions)
    fold = [-1] * n
    if n == 0:
        return fold

    rng = random.Random(random_state)
    # 每个标签 → 仍未分配且含该标签的样本下标集合(随分配进行不断缩小)
    label_to_samples: Dict[str, Set[int]] = defaultdict(set)
    for i, labs in enumerate(sample_labels):
        for l in labs:
            label_to_samples[l].add(i)
    # 期望:每个子集还"想要"多少样本 / 每个标签在每个子集还想要多少(浮点,随分配递减)
    desired = [p * n for p in proportions]
    desired_l: Dict[str, List[float]] = {
        l: [p * len(s) for p in proportions] for l, s in label_to_samples.items()
    }

    remaining = set(range(n))
    while remaining:
        # 在"还有样本的标签"里挑剩余最少的(最稀有);并列按名字定序后随机。
        candidates = [l for l, s in label_to_samples.items() if s]
        if not candidates:
            break                            # 只剩"无标签"样本,留到下面统一处理
        min_count = min(len(label_to_samples[l]) for l in candidates)
        tied = sorted(l for l in candidates if len(label_to_samples[l]) == min_count)
        l_star = rng.choice(tied)
        # 把这个最稀有类当前所有剩余样本逐个安置进"最缺它"的子集
        for i in sorted(label_to_samples[l_star]):
            if i not in remaining:
                continue
            j = _argmax_tiebreak(desired_l[l_star], desired, rng)
            fold[i] = j
            remaining.discard(i)
            for m in sample_labels[i]:        # 该样本所有标签的期望都要相应扣减
                desired_l[m][j] -= 1
                label_to_samples[m].discard(i)
            desired[j] -= 1

    # 无标签样本:按各子集剩余期望从大到小填
    for i in sorted(remaining):
        j = _argmax_tiebreak(desired, [0.0] * k, rng)
        fold[i] = j
        desired[j] -= 1
    return fold


@register_strategy(SplitStrategy.STRATIFIED_MULTILABEL, requires_labels=True)
def stratified_multilabel_split(stems: List[str], options: SplitOptions) -> Dict[str, List[str]]:
    """按每张图的多标签集合做迭代分层,返回 {"train"/"val"/"test": [stem, ...]}。"""
    test_rate = validate_rates(options.train_rate, options.val_rate)
    if options.labels_per_image is None:
        raise ValueError("stratified_multilabel 策略需要 labels_per_image,但收到 None。")
    if not stems:
        return {"train": [], "val": [], "test": []}

    ordered = sorted(stems)                                   # 复现前提:先把输入顺序固定
    labels = options.labels_per_image
    sample_labels: List[Set[str]] = [set(labels.get(s, [])) for s in ordered]
    fold = iterative_stratify(
        sample_labels, [options.train_rate, options.val_rate, test_rate], options.random_state
    )
    keys = ("train", "val", "test")
    out: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for stem, j in zip(ordered, fold):
        out[keys[j]].append(stem)
    return out