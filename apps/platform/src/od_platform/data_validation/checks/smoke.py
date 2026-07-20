#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :smoke.py
# @Time      :2026/7/17 09:44:05
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :

from od_platform.data_validation.registry import (
    check, CheckContext, CheckResult, CheckSeverity
)
@check("_smoke")
def smoke_check(ctx: CheckContext) -> CheckResult:
    return  CheckResult(
        name="smoke",
        severity=CheckSeverity.PASS,
        summary="占位测试，检测注册表机制是否工作正常",
        details={f"yaml_path": str(ctx.yaml_path)}
    )

