"""check: cross_split_leakage —— 检测同一个 stem 是否出现在多个 split 中。

数据泄露的典型场景:同一张图被同时放入了 train 和 val,验证分数会虚高。
这个 check 对 stem 做去重——stem 是文件名去掉扩展名,两个 split 下同名 stem
即判定为泄露(即使图片像素不同也不应该同名)。
"""
from __future__ import annotations

from typing import Any, Dict, List

from od_platform.common.constants import LEAK_MAX_DETAIL
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "cross_split_leakage"


@check(_NAME)
def validate_cross_split_leakage(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot

    # stem → 它出现的 split 列表
    stem_splits: Dict[str, List[str]] = {}
    total_images = 0

    for split in snap.splits:
        for img in snap.images_per_split.get(split, ()):
            total_images += 1
            stem = img.stem
            stem_splits.setdefault(stem, []).append(split)

    # 出现在多个 split 中的 stem
    leaked: List[Dict[str, Any]] = []
    for stem, splits in stem_splits.items():
        if len(splits) > 1:
            leaked.append({"stem": stem, "splits": splits})

    details: Dict[str, Any] = {
        "total_images": total_images,
        "unique_stems": len(stem_splits),
        "n_leaked": len(leaked),
    }

    if total_images == 0:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           "没有图像可检查", details)

    if leaked:
        details["leaked"] = leaked[:LEAK_MAX_DETAIL]
        # 按泄露的 split 组合分类
        leak_pairs: Dict[str, int] = {}
        for item in leaked:
            key = "+".join(sorted(item["splits"]))
            leak_pairs[key] = leak_pairs.get(key, 0) + 1
        details["leak_pairs"] = leak_pairs

        return CheckResult(
            _NAME, CheckSeverity.ERROR,
            f"检测到数据泄露:{len(leaked)} 个 stem 出现在多个 split 中"
            f" —— {leak_pairs}",
            details,
        )

    return CheckResult(
        _NAME, CheckSeverity.PASS,
        f"无跨集合泄露:全部 {total_images} 张图 stem 唯一归属一个 split",
        details,
    )
