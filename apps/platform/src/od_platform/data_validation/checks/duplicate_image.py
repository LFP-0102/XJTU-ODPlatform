#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : duplicate_image.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_validation/checks/duplicate_image.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.1.0
# @Description:重复图像检测(精确 sha256 + 近似 dHash)
"""重复图像检测。

两层:
  - 精确重复(sha256 字节相同):ERROR —— 这是数据泄漏,训练集和验证集出现字节级
    相同的图会让评估虚高,必须清理。
  - 近似重复(dHash 汉明距离 < 阈值):INFO —— 仅提示。很多数据集同场景多图视觉
    相似属正常(如视频帧、连拍),不应判为错误,故只报 INFO 不影响总评。

为什么 dHash 而非 aHash:aHash 对整体亮度敏感,安全帽/监控等背景相似的数据集
会把大量图判为近似(曾出现 7581 图报 37960 近似的误报);dHash 比较相邻像素差值,
对结构敏感、对整体亮度不敏感,误报率低很多。分桶(高 12 位)把 O(n²) 降到桶内
O(k²),7500 图几秒完成。
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List

from od_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check

_NAME = "DuplicateImageCheck"
_HAMMING_THRESHOLD = 4   # dHash 汉明距离 < 4 才算近似(收紧,减少误报)
_BUCKET_BITS = 12        # 用 dHash 高 12 位分桶,同桶内两两比较
_MAX_DETAIL = 20


def _dhash(path, size: int = 8) -> int:
    """差值哈希:9x8 灰度,每行比较相邻像素,输出 64bit。对结构敏感,对亮度不敏感。"""
    from PIL import Image
    img = Image.open(path).convert("L").resize((size + 1, size))
    pixels = list(img.getdata())
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if pixels[base + col] > pixels[base + col + 1] else 0)
    return bits


@check(_NAME)
def validate_duplicate_image(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    all_images: List = []
    for split in snap.splits:
        for img in snap.images_per_split[split]:
            all_images.append((split, img))

    # 1) 精确重复:sha256 字节相同 -> ERROR(数据泄漏)
    hash_map: Dict[str, tuple] = {}
    exact_dups: List[dict] = []
    for split, img in all_images:
        try:
            h = hashlib.sha256(img.read_bytes()).hexdigest()
        except OSError:
            continue
        if h in hash_map:
            e_split, e_img = hash_map[h]
            exact_dups.append({"image": img.name, "split": split, "path": str(img),
                               "duplicate_of": e_img.name, "duplicate_of_split": e_split})
        else:
            hash_map[h] = (split, img)

    # 2) 近似重复:dHash + 分桶,同桶内两两比较 -> INFO(提示性,场景相似属正常)
    approx_dups: List[dict] = []
    approx_method = "skipped (PIL unavailable)"
    try:
        from PIL import Image  # noqa: F401  探测 PIL 是否可用
        buckets: Dict[int, list] = defaultdict(list)
        for split, img in all_images:
            try:
                dh = _dhash(img)
            except Exception:
                continue
            buckets[dh >> (64 - _BUCKET_BITS)].append((dh, split, img))
        for items in buckets.values():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    dist = bin(items[i][0] ^ items[j][0]).count("1")
                    if dist < _HAMMING_THRESHOLD:
                        approx_dups.append({"image": items[j][2].name, "split": items[j][1],
                                            "similar_to": items[i][2].name,
                                            "similar_to_split": items[i][1],
                                            "hamming_distance": dist})
        approx_method = f"dHash (hamming<{_HAMMING_THRESHOLD}, bucketed by top {_BUCKET_BITS} bits)"
    except ImportError:
        pass

    n_exact, n_approx = len(exact_dups), len(approx_dups)
    details = {
        "n_images": len(all_images),
        "n_exact_duplicates": n_exact,
        "n_approx_duplicates": n_approx,
        "approx_method": approx_method,
        "exact_examples": exact_dups[:_MAX_DETAIL],
        "exact_duplicates": exact_dups,   # 全量,供 --purge 清理用
        "approx_examples": approx_dups[:_MAX_DETAIL],
    }

    # 完全重复 = ERROR(真数据泄漏);近似重复 = INFO(提示,不影响总评)
    if n_exact > 0:
        return CheckResult(
            _NAME, CheckSeverity.ERROR,
            f"{n_exact} 张完全重复(数据泄漏,需清理) + {n_approx} 对近似重复(提示)", details)
    if n_approx > 0:
        return CheckResult(
            _NAME, CheckSeverity.INFO,
            f"无完全重复,{n_approx} 对近似重复(dHash 提示,场景相似属正常)", details)
    return CheckResult(_NAME, CheckSeverity.PASS,
                       f"{len(all_images)} 张图无重复", {"n_images": len(all_images)})
