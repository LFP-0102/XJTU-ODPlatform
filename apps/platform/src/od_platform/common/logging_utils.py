#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :logging_utils.py
# @Time      :2026/7/13 15:23:45
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :

import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import Optional

from colorlog import ColoredFormatter
from od_platform.common.naming import run_stem


import platform

ROOT_LOGGER_NAME: str = "od_platform"


def get_logger(
        base_path: Path,
        log_type: str = 'general',
        model_name: Optional[str] = None,
        temp_log: bool = False,
        encoding: str = "utf-8",
        log_level: int = logging.INFO,
        logger_name: str = ROOT_LOGGER_NAME,
        run_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        ) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    # 准确日志的目录
    log_dir: Path = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    # 构造日志的名字
    if run_id:
        stem = run_stem(
            stage=("temp" if temp_log else log_type),
            run_id=run_id,
            dataset=dataset_name,
            model=model_name
        )
        filename = f"{stem}.log"
        log_file = log_dir / filename
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:21]
        prefix = "temp" if temp_log else log_type.replace("_", '-')
        filename_parts = [prefix, timestamp]
        if model_name:
            safe_model = "".join(
                c if c.isalnum() or c in "_-" else "_" for c in model_name
            )
            filename_parts.append(safe_model)
        file_name = f"{'_'.join(filename_parts)}.log"
        log_file = log_dir / f"{file_name}.log"

    # ============================================================
    # 4. 文件 Handler(完整格式, 含 logger 名)
    # ============================================================
    # %(name)s 字段会输出 logger 名, 比如 "odp_platform.cli.init_project"
    # 这是 getLogger(__name__) 设计带来的福利——一眼看出日志来源模块
    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)-8s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


    # ============================================================
    # 5. 控制台 Handler(彩色 + 紧凑格式)
    # ============================================================
    console_formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s%(reset)s "
        "%(log_color)s[%(levelname)-8s]%(reset)s "
        "%(cyan)s%(filename)-25s%(reset)s:"
        "%(blue)s%(lineno)-4d%(reset)s "
        "%(log_color)s│ %(message)s%(reset)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG":    "white",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red,bg_white",
        },
        style='%'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ============================================================
    # 6. 初始化信息
    # ============================================================
    logger.info('=' * 60)
    logger.info(f"日志系统初始化完成")
    logger.info(f"运行环境: {platform.system()} {platform.release()}")
    logger.info(f"阶段类型: {log_type}")
    logger.info(f"日志存储位置: {log_dir}")
    logger.info(f"日志级别: {logging.getLevelName(log_level)}")
    logger.info(f"模型名称: {model_name or '无'}")
    logger.info('=' * 60)

    return logger

if __name__ == "__main__":
    # 模块自测——演示"基础设施装 handler + 业务模块发声"的标准用法
    from od_platform.common.paths import LOGGING_DIR

    # 第一步: 装配根 logger 的 handler(这是 CLI 入口该做的事, 这里只是自测演示)
    logger = get_logger(
        base_path=LOGGING_DIR,
        log_type="log_test",
        temp_log=False,
    )

    # 第二步: 业务代码就这样写——拿一个 __name__ logger, 直接发声
    # 注意: 这里 __name__ 是 "__main__"(因为直接跑这个文件),
    # 实际项目里它会是 "odp_platform.common.logging_utils"
    logger.debug("这是 DEBUG (默认 INFO 级别看不到)")
    logger.info("这是 INFO")
    logger.warning("这是 WARNING")
    logger.error("这是 ERROR")