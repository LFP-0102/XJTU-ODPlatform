#!/usr/bin/env python
# @File       : archive.py
# @Path       : apps/platform/src/od_platform/model_train/archive.py
# @Author     : 刘赋平
# @Date       : 2026-07-19 11:06:29
# @Version    : v1.0.0
# @Description: 
#   请在此处填写该模块的功能概述。
#   例如：封装数据库连接工具类，提供增删改查接口。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026/7/19 | 刘赋平 | v1.0.0 | 初始化创建
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from od_platform.common.paths import TRAINED_MODELS_DIR

logger = logging.getLogger(__name__)


def archive_best_weight(save_dir: Path, stem: str) -> Optional[Path]:
    src = save_dir / "weights" / "best.pt"
    if not src.exists():
        logger.warning(f"此次训练可能失败了，并没有发现最佳权重文件，跳过归档: %s", src)
        return None

    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = TRAINED_MODELS_DIR / f"{stem}-best.pt"
    shutil.copy2(src, dst)
    logger.info("权重已归档: %s", dst)
    return dst


