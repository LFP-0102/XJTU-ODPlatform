#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yaml_schema.py
# @Time      :2026/7/17 10:11:46
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import  annotations
from typing import  List

from od_platform.data_validation.registry import (CheckContext, CheckResult, CheckSeverity, check)
_NAME = "yaml_schema"


@check(_NAME)
def validate_yaml_schema(ctx: CheckContext) -> CheckResult:
    if ctx.snapshot.yaml_load_error:
        return CheckResult(_NAME, CheckSeverity.ERROR, "yaml 无法加载", {"error": ctx.snapshot.yaml_load_error})
    data = ctx.snapshot.yaml_data
    problems: List[str] = []
    nc, names = data.get("nc"), data.get("names")
    if not isinstance(nc, int):
        problems.append("nc字段缺失或者nc不是整数")
    if not isinstance(names, (list, dict)):
        problems.append("names字段缺失或为不合法类型: 应该是list或者dict")
    elif isinstance(nc, int) and (len(names)) != nc:
        problems.append(f"nc={nc} 与 names数量{len(names)}不一致")
    if not data.get("path"):
        problems.append("path字段缺失或者找不到数据集的根")
    for split in ("train", "val", "test"):
        if split in data and not isinstance(data[split], str):
            problems.append(f"{split} 字段应该为字符串路径")
    if problems:
        return CheckResult(_NAME,CheckSeverity.ERROR,
            f"yaml 有{len(problems)}个问题: 问题如下：{problems}", {"problems": problems}
        )
    return CheckResult(_NAME, CheckSeverity.PASS, "yaml结构合法", {"nc":nc})