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
import datetime
import json
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


def _resolve_processed_root(path_str: str) -> Path | None:
    """从 processed 图片路径推断数据集根目录。
    路径形如 .../data/processed/<dataset>/images/<split>/<file>,返回 .../data/processed/<dataset>。"""
    p = Path(path_str)
    parts = p.parts
    try:
        idx = parts.index("images")
        return Path(*parts[:idx])
    except ValueError:
        return None


def _purge_bad_images(report, yes: bool = False) -> int:
    """交互式确认后删除问题图片(损坏 + 完全重复)+ 同名标签 + 写 purge_list.json 黑名单。

    只删 processed,不动 raw。purge_list.json 记录被删图的 stem,
    重跑 odp-transform 时 orchestrator 会读取并跳过这些 stem——
    这样 raw 完全不动,重跑 transform 数量也会真正减少。
    """
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

    print("\n⚠️ 此操作将:")
    print(f"   1. 删除 processed 的图片 + 标签({len(to_delete)} 张,raw 不动)")
    print("   2. 写 purge_list.json 黑名单(重跑 transform 时自动跳过这些图)")
    print("   3. 删除后需重跑 odp-transform 重建数据集与指纹(数量会真正减少)")
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
    purge_items = []
    for path_str, info in to_delete.items():
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
        purge_items.append({"stem": img_path.stem, "image": info["image"],
                            "reason": info["reason"]})

    # 写 purge_list.json 黑名单(放到 processed 数据集根目录)
    processed_root = _resolve_processed_root(next(iter(to_delete)))
    if processed_root:
        purge_list_path = processed_root / "purge_list.json"
        purge_data = {
            "dataset": processed_root.name,
            "purged_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "count": len(purge_items),
            "stems": [it["stem"] for it in purge_items],
            "items": purge_items,
        }
        try:
            purge_list_path.write_text(
                json.dumps(purge_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"   - 黑名单已写入: {purge_list_path}")
        except OSError as e:
            print(f"   ⚠️ 黑名单写入失败: {e}")

    print(f"\n✅ 已删除 {deleted} 张问题图片及其标签(raw 未改动)")
    print("⚠️ manifest 已失效,请重跑(会自动跳过黑名单中的图):")
    print(f"   odp-transform --dataset <name> --format <fmt> --split-strategy <strategy>")
    print("   重跑后数据集数量会真正减少。")
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
                        help="检测到问题图片(损坏/完全重复)后,交互式确认删除 processed(不动 raw)+ 写黑名单,需输入大写 YES")
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
