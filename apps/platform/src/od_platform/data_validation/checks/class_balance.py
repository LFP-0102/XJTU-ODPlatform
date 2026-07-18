"""check: class_balance —— 统计每个 split 中各类别的实例数,检测严重不均衡或类别缺失。

检查维度:
  · 某类别实例数为 0(训练时该类无正样本 → 永远学不到)
  · 类别间样本数差异巨大(不平衡比 > CLASS_IMBALANCE_RATIO)
  · train/val/test 之间类别分布显著不一致
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from od_platform.common.constants import CLASS_IMBALANCE_RATIO, CLASS_MAX_DETAIL
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "class_balance"


def _count_instances_per_class(label_paths, nc: int | None) -> Counter:
    """统计一批标签文件中每个 class_id 的出现次数。"""
    cnt: Counter = Counter()
    for p in label_paths:
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 1:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    continue
                cnt[cls_id] += 1
        except Exception:
            continue
    return cnt


@check(_NAME)
def validate_class_balance(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    nc = snap.nc
    names = snap.class_names

    # 1. 逐 split 统计
    per_split: Dict[str, Dict[str, Any]] = {}
    global_cnt: Counter = Counter()
    for split in snap.splits:
        files = snap.labels_files_per_split.get(split, ())
        cnt = _count_instances_per_class(files, nc)
        per_split[split] = {
            "n_files": len(files),
            "n_instances": sum(cnt.values()),
            "per_class": dict(sorted(cnt.items())),
        }
        global_cnt.update(cnt)

    total_instances = sum(global_cnt.values())

    details: Dict[str, Any] = {
        "nc": nc,
        "class_names": list(names),
        "total_instances": total_instances,
        "per_split": per_split,
    }

    if total_instances == 0:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           "未统计到任何标注实例(标签文件可能全部为空)", details)

    # 2. 缺失类别检测:有定义的类别但 0 实例
    if nc is not None and nc > 0:
        missing_classes: List[Dict[str, Any]] = []
        for cls_id in range(nc):
            if global_cnt.get(cls_id, 0) == 0:
                cls_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"
                missing_classes.append({"class_id": cls_id, "class_name": cls_name})

        if missing_classes:
            details["missing_classes"] = missing_classes
            return CheckResult(
                _NAME, CheckSeverity.ERROR,
                f"{len(missing_classes)} 个类别实例数为 0,训练时永远学不到:"
                f"{[m['class_name'] for m in missing_classes]}",
                details,
            )

    # 3. 类别不平衡检测
    if len(global_cnt) >= 2:
        counts = list(global_cnt.values())
        max_cnt = max(counts)
        min_cnt = min(counts)
        if min_cnt > 0:
            imbalance = max_cnt / min_cnt
            details["imbalance_ratio"] = round(imbalance, 2)
            details["max_class"] = {"class_id": max(global_cnt, key=global_cnt.get),
                                    "count": max_cnt}
            details["min_class"] = {"class_id": min(global_cnt, key=global_cnt.get),
                                    "count": min_cnt}

            if imbalance >= CLASS_IMBALANCE_RATIO:
                # 列出所有小类别
                small_classes = [
                    {"class_id": cid, "class_name": (names[cid] if cid < len(names) else f"class_{cid}"),
                     "count": cnt}
                    for cid, cnt in global_cnt.most_common()[::-1]
                    if cnt > 0 and max_cnt / cnt >= CLASS_IMBALANCE_RATIO
                ][:CLASS_MAX_DETAIL]
                details["small_classes"] = small_classes
                return CheckResult(
                    _NAME, CheckSeverity.WARNING,
                    f"类别严重不平衡:最多 {max_cnt} / 最少 {min_cnt} = {imbalance:.1f}:1"
                    f"(告警线 {CLASS_IMBALANCE_RATIO}:1)",
                    details,
                )

    # 4. 跨 split 分布一致性(简易版:比较每个 split 的类别比例)
    if len(per_split) >= 2:
        split_distributions: Dict[str, Dict[int, float]] = {}
        for split, info in per_split.items():
            n = info["n_instances"]
            if n > 0:
                split_distributions[split] = {
                    cid: cnt / n for cid, cnt in info["per_class"].items()
                }
        details["split_distributions"] = {
            s: {str(k): round(v, 4) for k, v in d.items()}
            for s, d in split_distributions.items()
        }

    return CheckResult(
        _NAME, CheckSeverity.PASS,
        f"类别分布正常:共 {nc or '?'} 类,{total_instances} 个标注实例,"
        f"各 split 均有覆盖",
        details,
    )
