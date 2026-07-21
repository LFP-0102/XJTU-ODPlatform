#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :manifest_lineage.py
# @Time      :2026/7/17 15:27:22
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
# apps/platform/src/od_platform/data_validation/checks/manifest_lineage.py
"""check: manifest_lineage —— 逐样本核对"冻结之后有没有人动过数据"。

对 manifest 的每个 SampleLineage{stem, split, sha256},去盘上找到标签、重算哈希、核对位置,
把 改内容 / 挪组 / 丢失 三种形态各自抓出来。任何一种都属阻塞级:训练即将吃到的,
已不是被冻结、被审计签过字的那一份。

诚实边界:只核对【标签内容】,不核对图片像素——因为当初冻结的就是标签字节
(SampleLineage.sha256 哈的是标签)。核对范围与冻结范围严格对齐,不多报也不少报。
图片存在性交给 pair_existence。
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Set, Tuple

from od_platform.common.constants import LINEAGE_MAX_DETAIL
from od_platform.data_validation.registry import (
    CheckContext, CheckResult, CheckSeverity, check,
)
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml

_NAME = "manifest_lineage"


@check(_NAME)
def validate_manifest_lineage(ctx: CheckContext) -> CheckResult:
    manifest, err, _ = load_manifest_from_yaml(ctx.snapshot.yaml_data)
    if err:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                        "拿不到 manifest,无法逐样本核对内容/位置", {"reason": err})

    # 盘上真实存在的标签(复用 snapshot 一次扫描的结果):(split, stem) -> path;并记 stem 实际所在 split
    disk: Dict[Tuple[str, str], object] = {}
    disk_splits_of_stem: Dict[str, Set[str]] = {}
    for split, files in ctx.snapshot.labels_files_per_split.items():
        for p in files:
            disk[(split, p.stem)] = p
            disk_splits_of_stem.setdefault(p.stem, set()).add(split)

    changed: List[Dict] = []
    moved:   List[Dict] = []
    missing: List[Dict] = []
    for s in manifest.samples:
        here = disk.get((s.split, s.stem))
        if here is None:
            others = disk_splits_of_stem.get(s.stem)
            if others:
                moved.append({"stem": s.stem, "recorded": s.split, "found_in": sorted(others)})
            else:
                missing.append({"stem": s.stem, "split": s.split})
            continue
        actual = hashlib.sha256(here.read_bytes()).hexdigest()
        if actual != s.sha256:
            changed.append({"stem": s.stem, "split": s.split,
                            "recorded_sha256": s.sha256[:16], "actual_sha256": actual[:16]})

    n_bad = len(changed) + len(moved) + len(missing)
    if n_bad == 0:
        return CheckResult(_NAME, CheckSeverity.PASS,
                        f"逐样本核对通过：{len(manifest.samples)} 个样本内容与位置均与冻结时一致",
                        {"n_samples": len(manifest.samples)})
    return CheckResult(
        _NAME, CheckSeverity.ERROR,
        (f"逐样本核对失败：改内容 {len(changed)} / 挪组 {len(moved)} / 丢失 {len(missing)}"
        f"（共 {len(manifest.samples)} 个样本）"),
        {"n_samples": len(manifest.samples),
        "n_changed": len(changed), "n_moved": len(moved), "n_missing": len(missing),
        "changed": changed[:LINEAGE_MAX_DETAIL], "moved": moved[:LINEAGE_MAX_DETAIL],
        "missing": missing[:LINEAGE_MAX_DETAIL]},
    )

