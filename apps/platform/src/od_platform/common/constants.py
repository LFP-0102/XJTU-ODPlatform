#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :constants.py
# @Time      :2026/7/15 10:36:59
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from typing import Tuple


class AnnotationFormat:
    PASCAL_VOC = 'pascal_voc'
    COCO = 'coco'
    YOLO = "yolo"
    LABELME = 'labelme'
    DOTA = 'dota'
    CVAT = 'cvat'
    CREATEML = 'createml'
    @classmethod
    def all(cls) ->Tuple[str, ...]:
        return cls.PASCAL_VOC, cls.COCO, cls.YOLO, cls.LABELME, cls.DOTA, cls.CVAT, cls.CREATEML

class Task:
    """支持的任务类型"""
    DETECT = "detect"
    SEGMENT = "segment"

    @classmethod
    def all(cls) ->Tuple[str, ...]:
        return cls.DETECT, cls.SEGMENT

class SplitStrategy:
    RANDOM = "random"
    STRATIFIED_MULTILABEL = 'stratified_multilabel' # 基于多标签的分层划分
    GROUP = 'group'                       # 按组划分(同 group 整体进同一 split)
    STRATIFIED_WEIGHTED = 'stratified_weighted'  # 按主类逆频率加权的分层划分

    @classmethod
    def all(cls) ->Tuple[str, ...]:
        return cls.RANDOM, cls.STRATIFIED_MULTILABEL, cls.GROUP, cls.STRATIFIED_WEIGHTED


DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_RATE = 0.8
DEFAULT_VAL_RATE = 0.1

DEFAULT_SPLIT_STRATEGY = SplitStrategy.RANDOM

RATE_EPSILON = 1e-6  # 比例校验吸收的浮点数误差

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PAIR_MISSING_ERROR_RATIO = 0.5  # 缺失标签的比例>= 50% -> ERROR
PAIR_MISSING_WARN_RATIO = 0.05  # 5% - 50% -> WARN
PAIR_MAX_DETAIL = 20    # details 我们最多列前N个文件，避免报告爆炸

LINEAGE_MAX_DETAIL: int = 20
