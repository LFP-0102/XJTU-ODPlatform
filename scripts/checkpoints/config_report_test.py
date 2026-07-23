#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : config_report_test.py
# @Path       : XJTU-ODPlatfrom/scripts/checkpoints/config_report_test.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-19 10:36:58
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-19:36:58 | Matri | v1.0.0 | 初始化创建
from od_platform.common import refs
from od_platform.runtime_config import build_train_config
from od_platform.runtime_config.report import render_overrides, render_effective,log_config_report
from od_platform.common.logging_utils import get_logger
from pathlib import Path
logger = get_logger(
    base_path=Path("."),
    log_type="init_project",
    temp_log=False
)
config, merger = build_train_config(
    yaml_path=r"C:\Users\fym\Desktop\XJTU-ODPlatform\apps\platform\configs\runtime\train.yaml",
    cli_args={"epochs": 300, "batch": 32, "device": 0}
)

eff = render_effective(config, merger)
ovr = render_overrides(config, merger)

log_config_report(config, merger, logger=logger)