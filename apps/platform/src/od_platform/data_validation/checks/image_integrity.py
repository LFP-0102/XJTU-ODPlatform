#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : image_integrity.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/data_validation/checks/image_integrity.py
# @Project    : XJTU-ODPlatform
# @Author     : 数据师
# @Date       : 2026-07-21
# @Version    : v1.1.0
# @Description:图片完整性检测(损坏/截断/缺EOI/无法解码)
"""图片完整性检测。

三关检测(与 ultralytics 训练时的 corrupt 判定标准对齐):
  1. jpeg_structure: JPEG 必须以 SOI(FFD8)开头、以 EOI(FFD9)结尾。
     尾部缺 EOI 的图 PIL 能解码,但 ultralytics 判 corrupt 并覆盖——这是之前
     "质检 PASS 但训练报 corrupt"的根因。本关与 ultralytics 对齐,补上这个缺口。
  2. verify_fail: PIL verify() 检查文件结构。
  3. decode_fail: PIL load() 实际解码像素。

检测到即 ERROR,可用 'odp-validate --purge' 清理。PIL 不可用降级 WARNING。
"""
from __future__ import annotations

from typing import Dict, List

from od_platform.data_validation.registry import CheckContext, CheckResult, CheckSeverity, check

_NAME = "ImageIntegrityCheck"
_MAX_LIST = 200
_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


def _check_jpeg_structure(path) -> str | None:
    """检查 JPEG 结构完整性:必须 SOI 开头 + EOI 结尾。返回 issue 字符串或 None。

    PIL 的 verify/load 对"尾部缺 EOI 但能解码"的图容忍,但 ultralytics 训练时
    会判 corrupt 并重新保存。本检查与 ultralytics 对齐,补上 PIL 的盲区。
    """
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            if head != _JPEG_SOI:
                return None  # 非 JPEG,交给 PIL 检查
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64))
            tail = f.read(64)
        if _JPEG_EOI not in tail:
            return "jpeg_missing_eoi"
    except OSError:
        return "read_fail"
    return None


@check(_NAME)
def validate_image_integrity(ctx: CheckContext) -> CheckResult:
    snap = ctx.snapshot
    try:
        from PIL import Image
    except ImportError:
        return CheckResult(_NAME, CheckSeverity.WARNING,
                           "PIL 不可用,跳过图片完整性检测", {"method": "skipped"})

    problems: List[dict] = []
    n_images = 0
    for split in snap.splits:
        for img in snap.images_per_split[split]:
            n_images += 1
            issue = None
            # 第一关:JPEG 结构检查(SOI/EOI,与 ultralytics 对齐)
            issue = _check_jpeg_structure(img)
            # 第二关:PIL verify 检查文件结构
            if issue is None:
                try:
                    with Image.open(img) as im:
                        im.verify()
                except Exception:
                    issue = "verify_fail"
            # 第三关:reopen + load 实际解码
            if issue is None:
                try:
                    with Image.open(img) as im:
                        im.load()
                except Exception:
                    issue = "decode_fail"
            if issue:
                problems.append({"split": split, "image": img.name,
                                 "path": str(img), "issue": issue})

    details: Dict = {
        "n_images": n_images,
        "n_problems": len(problems),
        "problem_images": problems[:_MAX_LIST],
    }
    if problems:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                           f"{len(problems)} 张图片有问题(损坏/截断/缺EOI),需审查", details)
    return CheckResult(_NAME, CheckSeverity.PASS,
                       f"{n_images} 张图片完整性正常",
                       {"n_images": n_images, "n_problems": 0, "problem_images": []})
