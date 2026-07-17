from typing import Tuple

class AnnotationFormat:
    PASCAL_VOC = 'pascal_voc'
    COCO = 'coco'
    YOLO = "yolo"
    @classmethod
    def all(cls) ->Tuple[str, ...]:
        return cls.PASCAL_VOC, cls.COCO, cls.YOLO

class Task:
    """支持的任务类型"""
    DETECT = "detect"
    SEGMENT = "segment"

    @classmethod
    def all(cls):
        return cls.DETECT, cls.SEGMENT

class SplitStrategy:
    """划分策略名集合(拼错当场报错,all() 供遍历;与 AnnotationFormat 同套路)。"""
    RANDOM = "random"
    STRATIFIED_MULTILABEL = "stratified_multilabel"
    @classmethod
    def all(cls) -> Tuple[str, ...]:
        return cls.RANDOM, cls.STRATIFIED_MULTILABEL

DEFAULT_RANDOM_STATE = 42   # 划分默认种子。四处共享:SplitOptions / split_service / CLI --seed / 回归测试
DEFAULT_TRAIN_RATE = 0.8    # 四处共享:SplitOptions / split_service / DatasetPipeline / CLI --train-rate
DEFAULT_VAL_RATE = 0.1      # 同上;test_rate 由 1 - train - val 推出,不单独设常量
RATE_EPSILON = 1e-6
DEFAULT_SPLIT_STRATEGY = SplitStrategy.RANDOM
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

PAIR_MISSING_ERROR_RATIO: float = 0.5    # 缺标签比例 >= 50% → ERROR(流程级问题)
PAIR_MISSING_WARN_RATIO:  float = 0.05   # 5% ~ 50% → WARNING
PAIR_MAX_DETAIL:          int   = 20     # details 里最多列前 N 个缺失文件,避免报告爆炸
LINEAGE_MAX_DETAIL: int = 20