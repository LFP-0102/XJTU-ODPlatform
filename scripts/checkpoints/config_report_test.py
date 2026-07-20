#!/usr/bin/env python
# @File       : config_report_test.py.py
# @Path       : scripts/checkpoints/config_report_test.py
# @Author     : 刘赋平
# @Date       : 2026-07-19 10:51:37
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/7/19 | 刘赋平 | v1.0.0 | 初始化创建
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
    yaml_path=r"C:\Users\刘赋平\Desktop\XJTU-ODPlatform\apps\platform\configs\runtime\train.yaml",
    cli_args={"epochs": 300, "batch": 32, "device": 0}
)

eff = render_effective(config, merger)
ovr = render_overrides(config, merger)

log_config_report(config, merger, logger=logger)
