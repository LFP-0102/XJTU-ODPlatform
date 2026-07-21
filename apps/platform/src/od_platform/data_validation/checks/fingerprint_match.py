#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :figureprint_match.py
# @Time      :2026/7/17 14:30:53
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import annotations
import logging
from od_platform.common.lineage import compute_contract_fingerprint
from od_platform.data_validation.registry import (CheckContext, CheckResult, CheckSeverity, check)

from od_platform.data_validation.checks._lineage import  load_manifest_from_yaml

_NAME = "fingerprint_match"
logger = logging.getLogger(__name__)
@check(_NAME)
def validate_fingerprint_match(ctx: CheckContext) -> CheckResult:
    manifest, error, odp = load_manifest_from_yaml(ctx.snapshot.yaml_data)
    if error:
        return CheckResult(
            _NAME,
            severity=CheckSeverity.ERROR,
            summary=f"拿不到划分契约，无法确定验证的是否为同一个数据: {error}",
            details={"reason": error},
        )
    # 第一问：自洽：从samples重新计算，应该等于账本自己记的指纹
    recomputed = compute_contract_fingerprint(
        manifest.dataset, manifest.strategy, manifest.seed, manifest.rations,
        manifest.names, manifest.samples, manifest.tool_version
    )
    logger.info(f"         质检模块计算的指纹信息如下：{recomputed}")
    if recomputed != manifest.contract_fingerprint:
        return CheckResult(
            _NAME,
            severity=CheckSeverity.ERROR,
            summary="从samples重新计算的指纹与账本记录的指纹不一致",
            details={"recomputed": recomputed, "expected": manifest.contract_fingerprint},
        )
    # 第二问： 同源：yaml改的指纹，应该等于账本的指纹
    yaml_fp = odp.get("contract_fingerprint")

    logger.info(f"         yaml 文件中记录的指纹信息如下：{yaml_fp}")
    logger.info(f"manifest.json 记录的指纹信息如下：{manifest.contract_fingerprint}")
    if yaml_fp != manifest.contract_fingerprint:
        return CheckResult(_NAME, CheckSeverity.ERROR,
                    "yaml 指纹与manifest 不一致--这份yaml 指向的不是 这次的划分",
                    {"yaml_fingerprint": yaml_fp, "manifest_fingerprint": manifest.contract_fingerprint}
                    )
    return CheckResult(_NAME, CheckSeverity.PASS,
                    f"指纹一致，确定为同一份划分 ({manifest.contract_fingerprint})  ",
                    {"contract_fingerprint": manifest.contract_fingerprint,
                    "n_samples": len(manifest.samples),
                    "strategy": manifest.strategy,
                    "seed": manifest.seed,
                    })


