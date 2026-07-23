#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :split_not_reproducible.py
# @Time      :2026/7/16 09:29:10
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
# scripts/checkpoints/split_not_reproducible.py
"""最朴素的随机划分:打乱 + 切三刀。连跑两遍,看它每次都变。"""
import random
from od_platform.common.paths import RAW_DATA_DIR

IMAGES = RAW_DATA_DIR / "demo" / "images"
stems = sorted(p.stem for p in IMAGES.glob("*.jpg"))   # 63 个样本;sorted 让起点确定


def naive_split(items, ratios=(0.7, 0.15, 0.15), seed=None):
    xs = list(items)
    random.Random(seed).shuffle(xs)          # seed=None → 每次进程都不一样(最朴素的默认)
    n = len(xs); n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    return xs[:n_tr], xs[n_tr:n_tr + n_va], xs[n_tr + n_va:]


def show(tag, s):
    tr, va, te = s
    print(f"[{tag}] 规模 train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    print(f"        train 前 6: {tr[:6]}")
    print(f"        val  全部: {sorted(va)}")


print("=" * 68); print("A. 不设种子,连跑两次(模拟你今天跑、同事明天跑)"); print("=" * 68)
r1 = naive_split(stems); r2 = naive_split(stems)
show("第 1 次", r1); show("第 2 次", r2)
v1, v2 = set(r1[1]), set(r2[1])
print(f"\n  两次 val 是否相同?           {v1 == v2}")
print(f"  两次 val 的交集大小:         {len(v1 & v2)} / {len(v1)}")
print(f"  只在某一次 val 里的样本数:   {len(v1 ^ v2)}")

print("\n" + "=" * 68); print("B. 补一个 seed=42,再连跑两次"); print("=" * 68)
r3 = naive_split(stems, seed=42); r4 = naive_split(stems, seed=42)
show("seed=42 第 1 次", r3); show("seed=42 第 2 次", r4)
print(f"\n  两次 val 是否相同?           {set(r3[1]) == set(r4[1])}")
