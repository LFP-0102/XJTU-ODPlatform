"""check: pair_existence —— 每张图是否都有对应标签,按缺失比例分级。"""
from __future__ import annotations

from typing import Dict, List

from od_platform.common.constants import (
    PAIR_MISSING_ERROR_RATIO, PAIR_MISSING_WARN_RATIO, PAIR_MAX_DETAIL,
)
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "PairExistenceCheck"

@check(_NAME)
def validate_pair_existence(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    total = 0
    missing: List[Dict] = []
    for split in snap.splits:
        on_disk = {p.stem for p in snap.labels_files_per_split.get(split, ())}
        for img in snap.images_per_split[split]:
            total += 1
            if img.stem not in on_disk:
                missing.append({"split": split, "image": img.name})

    if total == 0:   # 一张图都没有:这本身是个关注项,交给报告去解释
        return CheckResult(_NAME, CheckSeverity.WARNING, "没有可配对的图像", {"n_images": 0})

    ratio = len(missing) / total
    details = {
        "n_images": total, "n_missing": len(missing),
        "missing_ratio": round(ratio, 4), "missing": missing[:PAIR_MAX_DETAIL],
    }
    if ratio >= PAIR_MISSING_ERROR_RATIO:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           f"缺失的标签比例很高{len(missing)}/{total} 张图缺标签 | 比例（{ratio:.1%}）", details)
    if ratio >= PAIR_MISSING_WARN_RATIO:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           f"缺失的标签比例较高{len(missing)}/{total} 张图缺标签 | 比例（{ratio:.1%}）", details)
    if missing:
        return CheckResult(_NAME, CheckSeverity.INFO,
                           f"存在{len(missing)} 张图缺标签（{ratio:.1%}，低于告警线）", details)
    return CheckResult(_NAME, CheckSeverity.PASS, f"{total} 张图全部配对", {"n_images": total})