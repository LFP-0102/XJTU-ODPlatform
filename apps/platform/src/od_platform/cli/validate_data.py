#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName  :validate_data.py
# @Time      :2026/7/17 15:56:10
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
# apps/platform/src/od_platform/cli/validate_data.py
"""odp-validate:训练前质量闸门的命令行入口。退出码是给机器的第一公民。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from od_platform.common import refs
from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.data_validation.registry import CheckSeverity
from od_platform.data_validation.service import validate_dataset

# 总评 severity → 进程退出码。CI 直接读这个数。
_EXIT_CODE = {
    CheckSeverity.ERROR:   2,   # 阻断:必须停
    CheckSeverity.WARNING: 1,   # 有条件放行:人工确认
    CheckSeverity.INFO:    0,
    CheckSeverity.PASS:    0,
}

_PURGE_KEYWORD = "YES"


def _collect_purge_targets(report) -> dict:
    """从校验结果收集待清理图片:损坏图 + 完全重复图(保留原始,删重复)。返回 {path: info}。"""
    to_delete: dict = {}

    # 1) 损坏/截断/无法解码(ImageIntegrityCheck)
    img_chk = next((r for r in report.results if r.name == "ImageIntegrityCheck"), None)
    if img_chk:
        for p in img_chk.details.get("problem_images", []):
            to_delete[p["path"]] = {"split": p["split"], "image": p["image"],
                                    "reason": f"损坏({p['issue']})"}

    # 2) 完全重复(DuplicateImageCheck):删后续重复,保留第一张(original)
    dup_chk = next((r for r in report.results if r.name == "DuplicateImageCheck"), None)
    if dup_chk:
        for d in dup_chk.details.get("exact_duplicates", []):
            to_delete[d["path"]] = {"split": d["split"], "image": d["image"],
                                    "reason": f"完全重复(与 {d['duplicate_of']} 字节相同)"}
    return to_delete


def _purge_bad_images(report, yes: bool = False) -> int:
    """交互式确认后删除问题图片(损坏 + 完全重复)+ 同名标签。删除后需重跑 transform。"""
    to_delete = _collect_purge_targets(report)
    if not to_delete:
        print("✅ 没有问题图片需要清理")
        return 0

    print(f"\n⚠️ 发现 {len(to_delete)} 张问题图片需清理:")
    for i, (path_str, info) in enumerate(to_delete.items()):
        if i >= 20:
            print(f"  ... 还有 {len(to_delete) - 20} 张(详见 quality_report.md)")
            break
        print(f"  [{info['split']}] {info['image']} - {info['reason']}")

    print("\n⚠️ 此操作将删除图片 + 同名标签文件,删除后需重跑 odp-transform 重建 manifest。")
    if not yes:
        try:
            ans = input(f'确认删除请输入大写 "{_PURGE_KEYWORD}"(其他任意输入取消): ').strip()
        except (KeyboardInterrupt, EOFError):
            print("\n已取消")
            return 0
        if ans != _PURGE_KEYWORD:
            print("已取消,未删除任何文件")
            return 0

    deleted = 0
    for path_str in to_delete:
        img_path = Path(path_str)
        # 标签路径:把路径里的 images 段替换为 labels,后缀改 .txt
        parts = list(img_path.parts)
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                break
        label_path = Path(*parts).with_suffix(".txt")
        try:
            img_path.unlink(missing_ok=True)
            deleted += 1
        except OSError:
            pass
        try:
            label_path.unlink(missing_ok=True)
        except OSError:
            pass

    print(f"\n✅ 已删除 {deleted} 张问题图片及其标签")
    print("⚠️ manifest 已失效,请重跑 'odp-transform --dataset <name> --format <fmt> ...' 重建数据集与指纹")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="odp-validate",
                                     description="数据集验证(训练前质量闸门)")
    parser.add_argument("--dataset", help="数据集名(如 MRI_PASCAL)或 yaml 路径")
    parser.add_argument("--run-id", default=None,
                        help="指定 run_id(默认时间戳;CI 可传流水线号)")
    parser.add_argument("--no-report", action="store_true",
                        help="只判定不落盘 report.json / results.csv")
    parser.add_argument("--purge", action="store_true",
                        help="检测到问题图片(损坏/完全重复)后,交互式确认删除(需输入大写 YES)")
    parser.add_argument("--yes", action="store_true",
                        help="配合 --purge:跳过交互确认直接删除(慎用)")
    args = parser.parse_args()

    get_logger(paths.LOGGING_DIR, log_type="validate")
    report = validate_dataset(refs.resolve_dataset_yaml(args.dataset),
                          run_id=args.run_id, write_report=not args.no_report)

    if args.purge:
        return _purge_bad_images(report, yes=args.yes)
    return _EXIT_CODE.get(report.overall_severity, 2)


if __name__ == "__main__":
    sys.exit(main())
