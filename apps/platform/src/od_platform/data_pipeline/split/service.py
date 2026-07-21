#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :service.py
# @Time      :2026/7/16 14:01:19
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import annotations
from typing import List, Dict, Optional
import logging
from od_platform.common.constants import (DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY,
                                    DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE
                                        )

from od_platform.data_pipeline.split.registry import SplitOptions, get_strategy

logger = logging.getLogger(__name__)

def split_dataset(
    stems: List[str],
    train_rate: float = DEFAULT_TRAIN_RATE,
    val_rate: float = DEFAULT_VAL_RATE,
    random_state: int = DEFAULT_RANDOM_STATE,
    *,
    strategy: str = DEFAULT_SPLIT_STRATEGY,
    labels_per_image: Optional[Dict[str, List[str]]] = None,
    group_per_image: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[str]]:
    entry = get_strategy(strategy)
    if entry.requires_labels and labels_per_image is None:
        raise ValueError(f"划分策略 {strategy} 需要标签信息 labels_per_iamge,但是没有提供")
    options = SplitOptions(
        train_rate = train_rate, val_rate=val_rate, random_state=random_state,
        labels_per_image=labels_per_image, group_per_image=group_per_image
    )
    result = entry.func(stems, options)
    logger.info("划分完成 strategy= %s 规模= %s",strategy,{k: len(v) for k, v in result.items()})
    return  result
