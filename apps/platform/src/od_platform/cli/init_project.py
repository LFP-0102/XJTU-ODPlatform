from pathlib import Path
from typing import List

import logging

from torch.utils.data.datapipes.dataframe.dataframe_wrapper import create_dataframe
from ultralytics.models.yolo import detect

from od_platform.common.paths import ROOT_DIR,get_dirs_to_initialize,LOGGING_DIR
from od_platform.common.logging_utills import get_logger

LINE_WIDTH = 60
logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="init_project",
    temp_log=False,
)

def initialize_project()->None:
    logger.info("="*LINE_WIDTH)
    logger.info(f"开始初始化项目核心目录".center(LINE_WIDTH,'='))
    logger.info(f"项目的根目录为:{ROOT_DIR}")

    created: List[Path] = []
    existed: List[Path] = []

    for d in get_dirs_to_initialize():
        rel = d.relative_to(ROOT_DIR)
        if d.exists():
            logger.info(f"文件已经存在：{rel}")
            existed.append(d)
        else:
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"文件已经创建成功：{rel}")
            created.append(d)

    logger.info(f"汇总:此次操作新建了 {len(created)}，已经存在的有:{len(existed)}")

if __name__ == "__main__":
    initialize_project()