"""check: bbox_quality —— 检查每个标注框的几何合理性。

检查维度:
  · 面积极小(box_area < BBOX_MIN_AREA)——可能是误标点
  · 面积过大(box_area > BBOX_MAX_AREA)——几乎覆盖全图
  · 极端宽高比(aspect > BBOX_MAX_ASPECT_RATIO)——可能是标注错误
  · 宽/高 ≤ 0——退化框
"""
from __future__ import annotations

from typing import Any, Dict, List

from od_platform.common.constants import (
    BBOX_MAX_AREA, BBOX_MAX_ASPECT_RATIO, BBOX_MIN_AREA, BBOX_MAX_DETAIL,
)
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "bbox_quality"


def _check_one_box(cls_id: int, xc: float, yc: float, w: float, h: float,
                   nc: int | None) -> str | None:
    """检查单个框,合法返回 None,非法返回错误描述。"""
    area = w * h
    if area <= 0:
        return f"框面积=0 (class={cls_id}, w={w}, h={h})"
    if area < BBOX_MIN_AREA:
        return f"框面积极小 area={area:.6f} (class={cls_id})——可能是误标点"
    if area > BBOX_MAX_AREA:
        return f"框面积过大 area={area:.2f} (class={cls_id})——几乎覆盖全图"

    # 宽高比:max(w,h)/min(w,h) 控制在合理范围
    if w > 0 and h > 0:
        aspect = max(w, h) / min(w, h)
        if aspect > BBOX_MAX_ASPECT_RATIO:
            return f"框宽高比极端 aspect={aspect:.1f}:1 (class={cls_id}, w={w:.4f}, h={h:.4f})"

    return None


@check(_NAME)
def validate_bbox_quality(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    nc = snap.nc

    total_boxes = 0
    total_files = 0
    bad_boxes: List[Dict[str, Any]] = []  # 有问题的框

    for split in snap.splits:
        for label_path in snap.labels_files_per_split.get(split, ()):
            total_files += 1
            try:
                text = label_path.read_text(encoding="utf-8").strip()
            except Exception:
                continue  # 读不出的文件交给 label_integrity 去报

            if not text:
                continue

            for line_no, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) != 5:
                    continue  # 格式错交给 label_integrity

                try:
                    vals = [float(p) for p in parts]
                except ValueError:
                    continue

                cls_id = int(vals[0])
                xc, yc, w, h = vals[1], vals[2], vals[3], vals[4]
                total_boxes += 1

                err = _check_one_box(cls_id, xc, yc, w, h, nc)
                if err:
                    bad_boxes.append({
                        "split": split, "file": label_path.name, "line": line_no,
                        "class_id": cls_id, "xc": xc, "yc": yc, "w": w, "h": h,
                        "reason": err,
                    })

    n_bad = len(bad_boxes)

    details: Dict[str, Any] = {
        "n_label_files": total_files,
        "n_total_boxes": total_boxes,
        "n_bad_boxes": n_bad,
    }

    if total_boxes == 0:
        return CheckResult(_NAME, CheckSeverity.INFO,
                           "没有标注框可检查(标签文件可能全部为空)", details)

    if n_bad > 0:
        bad_ratio = n_bad / total_boxes
        details["bad_boxes"] = bad_boxes[:BBOX_MAX_DETAIL]
        details["bad_ratio"] = round(bad_ratio, 4)
        # 按问题类型分类
        by_reason: Dict[str, int] = {}
        for b in bad_boxes:
            reason_type = b["reason"].split("(")[0].strip()
            by_reason[reason_type] = by_reason.get(reason_type, 0) + 1
        details["by_reason"] = by_reason

        if bad_ratio >= 0.1:
            return CheckResult(
                _NAME, CheckSeverity.ERROR,
                f"标注框质量问题严重:{n_bad}/{total_boxes} 个框异常({bad_ratio:.1%})"
                f" —— {by_reason}",
                details,
            )
        return CheckResult(
            _NAME, CheckSeverity.WARNING,
            f"标注框质量问题:{n_bad}/{total_boxes} 个框异常({bad_ratio:.1%})"
            f" —— {by_reason}",
            details,
        )

    return CheckResult(
        _NAME, CheckSeverity.PASS,
        f"全部 {total_boxes} 个标注框几何合法({total_files} 个标签文件)",
        details,
    )
