#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :test_data_validate.py
# @Time      :2026/7/17 09:47:41
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from od_platform.data_validation.registry import get_check, CheckContext, get_all_checks, list_check_names

from pathlib import Path

entry = get_check("_smoke")
entry2 = get_check("yaml_schema")
yaml_path = Path(r"C:\Users\fym\Desktop\XJTU-ODPlatform\apps\platform\configs\datasets\MRI_PASCAL.yaml")
result = entry.func(CheckContext(yaml_path=yaml_path))
result2 = entry2.func(CheckContext(yaml_path=yaml_path))
print(result.severity, result.summary)
print(result2.severity, result2.summary, result2.details)

print(f"已经注册的检测有：{get_all_checks()}")
print(f"已经注册的质检项目有： {list_check_names()}")