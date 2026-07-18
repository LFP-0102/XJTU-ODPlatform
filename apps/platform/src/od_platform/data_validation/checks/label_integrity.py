"""check: label_integrity —— 逐行解析标签文件,校验 YOLO 格式是否合法。

检查维度:
  · 空文件(0 行标注——可能是漏标也可能是真负样本,单独统计)
  · 列数不对(YOLO 检测应为 5 列: class xc yc w h)
  · class_id 越界(>= nc 或 < 0)
  · 归一化坐标超出 [0,1]
  · 框宽/高 ≤ 0
  · 重复行
"""
from __future__ import annotations

from typing import Any, Dict, List

from od_platform.common.constants import LABEL_MAX_DETAIL
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "label_integrity"

# ── 单行解析 ──────────────────────────────────────────────

def _parse_yolo_line(line: str, line_no: int, nc: int | None) -> str | None:
    """解析一行 YOLO 标注,合法返回 None,非法返回错误描述字符串。"""
    stripped = line.strip()
    if not stripped:
        return None  # 空白行跳过,不计为错误

    parts = stripped.split()
    if len(parts) != 5:
        return f"第{line_no}行:期望5列,实际{len(parts)}列 —— {stripped[:80]}"

    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return f"第{line_no}行:包含非数字内容 —— {stripped[:80]}"

    cls_id = int(vals[0])
    xc, yc, w, h = vals[1], vals[2], vals[3], vals[4]

    # class_id 范围:0 ≤ cls_id < nc(若 nc 已知)
    if nc is not None and (cls_id < 0 or cls_id >= nc):
        return f"第{line_no}行:class_id={cls_id} 越界(nc={nc})"

    # 坐标范围:[0,1]
    if not (0.0 <= xc <= 1.0):
        return f"第{line_no}行:xc={xc} 超出 [0,1]"
    if not (0.0 <= yc <= 1.0):
        return f"第{line_no}行:yc={yc} 超出 [0,1]"
    if not (0.0 <= w <= 1.0):
        return f"第{line_no}行:w={w} 超出 [0,1]"
    if not (0.0 <= h <= 1.0):
        return f"第{line_no}行:h={h} 超出 [0,1]"

    # 退化框:w 或 h ≤ 0
    if w <= 0:
        return f"第{line_no}行:框宽 w={w} ≤ 0"
    if h <= 0:
        return f"第{line_no}行:框高 h={h} ≤ 0"

    return None


# ── check 入口 ────────────────────────────────────────────

@check(_NAME)
def validate_label_integrity(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    nc = snap.nc

    total_files = 0
    total_annotations = 0
    empty_files: List[str] = []             # 0 行标注(可能是真负样本)
    bad_files: List[Dict[str, Any]] = []    # 有格式错误的文件
    dup_files: List[Dict[str, Any]] = []    # 有重复行的文件

    for split in snap.splits:
        for label_path in snap.labels_files_per_split.get(split, ()):
            total_files += 1
            try:
                text = label_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                bad_files.append({"split": split, "file": label_path.name,
                                  "error": f"无法读取: {e}"})
                continue

            if not text:
                empty_files.append(f"{split}/{label_path.name}")
                continue  # 空文件单独归类,不算 bad

            lines = text.splitlines()
            errors: List[str] = []
            seen: set = set()
            dups: List[str] = []

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                err = _parse_yolo_line(line, i, nc)
                if err:
                    errors.append(err)
                # 去重:把多余空白归一化后检测重复行
                normalized = " ".join(stripped.split())
                if normalized in seen:
                    dups.append(f"第{i}行重复: {normalized[:60]}")
                seen.add(normalized)

            total_annotations += len(seen)

            if errors:
                bad_files.append({
                    "split": split, "file": label_path.name,
                    "n_errors": len(errors),
                    "errors": errors[:LABEL_MAX_DETAIL],
                })
            if dups:
                dup_files.append({
                    "split": split, "file": label_path.name,
                    "n_dups": len(dups),
                    "duplicates": dups[:LABEL_MAX_DETAIL],
                })

    n_bad = len(bad_files)
    n_empty = len(empty_files)
    n_dup = len(dup_files)

    details: Dict[str, Any] = {
        "n_label_files": total_files,
        "n_total_annotations": total_annotations,
        "n_empty_files": n_empty,
        "n_bad_files": n_bad,
        "n_files_with_duplicates": n_dup,
    }

    if total_files == 0:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           "没有标签文件可检查", details)

    if n_bad > 0:
        bad_ratio = n_bad / total_files
        details["bad_files"] = bad_files[:LABEL_MAX_DETAIL]
        details["empty_files"] = empty_files[:LABEL_MAX_DETAIL]
        details["duplicate_files"] = dup_files[:LABEL_MAX_DETAIL]
        if bad_ratio >= 0.1:
            return CheckResult(
                _NAME, CheckSeverity.ERROR,
                f"标签格式错误:{n_bad}/{total_files} 个文件有格式问题({bad_ratio:.1%}),"
                f"另有 {n_empty} 个空文件、{n_dup} 个文件含重复行",
                details,
            )
        return CheckResult(
            _NAME, CheckSeverity.WARNING,
            f"标签格式问题:{n_bad}/{total_files} 个文件有格式问题({bad_ratio:.1%}),"
            f"另有 {n_empty} 个空文件、{n_dup} 个文件含重复行",
            details,
        )

    if n_dup > 0:
        details["duplicate_files"] = dup_files[:LABEL_MAX_DETAIL]
        return CheckResult(
            _NAME, CheckSeverity.INFO,
            f"标签格式合法({total_files} 文件,{total_annotations} 个标注),"
            f"{n_dup} 个文件含重复行",
            details,
        )

    msg = f"全部 {total_files} 个标签文件格式合法,共 {total_annotations} 个标注"
    if n_empty > 0:
        msg += f",{n_empty} 个空文件(负样本)"
    return CheckResult(_NAME, CheckSeverity.PASS, msg, details)
