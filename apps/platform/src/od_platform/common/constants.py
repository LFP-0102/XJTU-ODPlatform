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
        return (cls.DETECT, cls.SEGMENT)