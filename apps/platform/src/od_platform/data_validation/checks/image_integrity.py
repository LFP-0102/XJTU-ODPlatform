"""check: image_integrity —— 检查图像文件是否可正常读取,检测损坏/截断/异常尺寸。

检查维度:
  · 文件大小为 0
  · PIL 无法打开(损坏/截断/格式不支持)
  · 尺寸异常(单边 ≤ 1 像素——几乎不可能是有意义的训练样本)

注意:本 check 会真正打开图像文件读取像素头,大样本量时耗时略高于纯文本类 check。
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    Image = None  # type: ignore[assignment]
    UnidentifiedImageError = Exception  # type: ignore[assignment,misc]

from od_platform.common.constants import IMAGE_MAX_DETAIL
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)

_NAME = "image_integrity"

# 最小合法尺寸:单边 ≤ 此值视为异常
_MIN_SIDE = 1


@check(_NAME)
def validate_image_integrity(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot

    if Image is None:
        return CheckResult(
            _NAME, CheckSeverity.WARNING,
            "PIL 未安装,跳过图像完整性检查。pip install Pillow",
            {},
        )

    total_images = 0
    zero_byte: List[str] = []               # 文件大小为 0
    unreadable: List[Dict[str, Any]] = []   # PIL 打不开
    abnormal_size: List[Dict[str, Any]] = []  # 尺寸异常

    for split in snap.splits:
        for img_path in snap.images_per_split.get(split, ()):
            total_images += 1

            # 1. 文件大小
            try:
                fsize = img_path.stat().st_size
            except OSError:
                unreadable.append({"split": split, "file": img_path.name,
                                   "reason": "无法获取文件信息"})
                continue

            if fsize == 0:
                zero_byte.append(f"{split}/{img_path.name}")
                continue

            # 2. PIL 打开
            try:
                img = Image.open(img_path)
                w, h = img.size
                img.verify()  # 不加载全部像素,只校验文件结构
            except Exception as e:
                unreadable.append({"split": split, "file": img_path.name,
                                   "reason": f"{type(e).__name__}: {e}"})
                continue

            # 3. 尺寸异常
            if w <= _MIN_SIDE or h <= _MIN_SIDE:
                abnormal_size.append({"split": split, "file": img_path.name,
                                      "width": w, "height": h})

    n_bad = len(zero_byte) + len(unreadable) + len(abnormal_size)

    details: Dict[str, Any] = {
        "total_images": total_images,
        "n_zero_byte": len(zero_byte),
        "n_unreadable": len(unreadable),
        "n_abnormal_size": len(abnormal_size),
        "n_bad": n_bad,
    }

    if total_images == 0:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           "没有图像文件可检查", details)

    if n_bad > 0:
        details["zero_byte"] = zero_byte[:IMAGE_MAX_DETAIL]
        details["unreadable"] = unreadable[:IMAGE_MAX_DETAIL]
        details["abnormal_size"] = abnormal_size[:IMAGE_MAX_DETAIL]
        bad_ratio = n_bad / total_images
        if bad_ratio >= 0.05:
            return CheckResult(
                _NAME, CheckSeverity.ERROR,
                f"图像完整性异常:{n_bad}/{total_images} 张图有问题({bad_ratio:.1%})"
                f" —— 零字节 {len(zero_byte)} / 不可读 {len(unreadable)} "
                f"/ 尺寸异常 {len(abnormal_size)}",
                details,
            )
        return CheckResult(
            _NAME, CheckSeverity.WARNING,
            f"图像完整性问题:{n_bad}/{total_images} 张图有问题({bad_ratio:.1%})"
            f" —— 零字节 {len(zero_byte)} / 不可读 {len(unreadable)} "
            f"/ 尺寸异常 {len(abnormal_size)}",
            details,
        )

    return CheckResult(
        _NAME, CheckSeverity.PASS,
        f"全部 {total_images} 张图像可正常读取,尺寸合法",
        details,
    )
