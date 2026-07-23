"""推理器:把一张 BGR 图片 → 检测结果 + 美化标注图。

两种后端(settings.INFER_BACKEND):
  - yolo:真实 ultralytics 推理(生产默认)。复用引擎 YOLOInferConfig 校验参数、
          BeautifyVisualizer 出图(与 CLI 一致)。
  - demo:免 torch,生成合成检测框,仍用真实 BeautifyVisualizer 绘制,便于无模型演示。

契约:detections 为 [{label, confidence, bbox:[x1,y1,x2,y2]}],与引擎 model_infer 对齐。
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass, field

import numpy as np
from django.conf import settings

from . import registry

# 传给 ultralytics predict 的干净子集(避免 to_ultralytics_kwargs 里训练参数泄漏)
_PREDICT_KEYS = ('conf', 'iou', 'imgsz', 'max_det', 'classes', 'agnostic_nms', 'augment', 'half')


@dataclass
class PredictOutcome:
    detections: list[dict] = field(default_factory=list)
    annotated: np.ndarray | None = None
    width: int = 0
    height: int = 0
    infer_ms: float = 0.0


def _make_visualizer(labels: list[str]):
    from od_platform.visualization import BeautifyVisualizer
    return BeautifyVisualizer(
        labels=labels,
        label_mapping=settings.VIZ_LABEL_MAPPING,
        color_mapping=settings.VIZ_COLOR_MAPPING,
        font_path=settings.VIZ_FONT_PATH,
    )


def _draw(image_bgr, boxes_xyxy, confs, labels):
    from od_platform.visualization import BeautifyVisualizer
    dets = BeautifyVisualizer.from_yolo_results(
        boxes=np.asarray(boxes_xyxy, dtype=float),
        confidences=np.asarray(confs, dtype=float),
        labels=labels,
        color_mapping=settings.VIZ_COLOR_MAPPING,
    )
    viz = _make_visualizer(labels=sorted(set(labels)) or list(settings.VIZ_LABEL_MAPPING.keys()))
    return viz.draw(image_bgr, dets, use_label_mapping=settings.VIZ_USE_LABEL_MAPPING)


def _predict_kwargs(model_name: str, params: dict) -> dict:
    """用引擎 YOLOInferConfig 校验参数,再抽取干净的 predict 子集。"""
    from od_platform.runtime_config.infer_config import YOLOInferConfig
    cfg = YOLOInferConfig(
        task='detect',
        conf=float(params.get('conf', 0.25)),
        iou=float(params.get('iou', 0.45)),
        imgsz=int(params.get('imgsz', 640)),
        max_det=int(params.get('max_det', 300)),
    )
    full = cfg.to_ultralytics_kwargs()
    return {k: full[k] for k in _PREDICT_KEYS if k in full}


# --------------------------------------------------------------------------- #
# 真实 YOLO 后端
# --------------------------------------------------------------------------- #
def _predict_yolo(model_name: str, image_bgr: np.ndarray, params: dict) -> PredictOutcome:
    model = registry.get_model(model_name)
    kwargs = _predict_kwargs(model_name, params)
    results = model.predict(
        source=image_bgr, verbose=False, save=False, device=settings.INFER_DEVICE, **kwargs
    )
    r = results[0]
    names = model.names
    h, w = image_bgr.shape[:2]

    detections: list[dict] = []
    boxes_xyxy, confs, labels = [], [], []
    boxes = getattr(r, 'boxes', None)
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.int().cpu().tolist()
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            label = names[k] if isinstance(names, dict) else names[int(k)]
            detections.append({
                'label': label,
                'confidence': float(c),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
            })
            boxes_xyxy.append([x1, y1, x2, y2])
            confs.append(c)
            labels.append(label)

    annotated = _draw(image_bgr, boxes_xyxy, confs, labels) if detections else image_bgr.copy()

    speed = getattr(r, 'speed', None) or {}
    infer_ms = float(sum(v for v in speed.values() if isinstance(v, (int, float)))) or 0.0
    return PredictOutcome(detections, annotated, w, h, round(infer_ms, 1))


# --------------------------------------------------------------------------- #
# 演示后端(免 torch)
# --------------------------------------------------------------------------- #
def _predict_demo(model_name: str, image_bgr: np.ndarray, params: dict) -> PredictOutcome:
    h, w = image_bgr.shape[:2]
    classes = registry.model_classes(model_name)
    conf_th = float(params.get('conf', 0.25))

    t0 = time.perf_counter()
    n = random.choices([0, 1, 2, 3], weights=[12, 58, 22, 8])[0]
    detections, boxes_xyxy, confs, labels = [], [], [], []
    for _ in range(n):
        bw = w * random.uniform(0.12, 0.28)
        bh = h * random.uniform(0.12, 0.28)
        cx = w * random.uniform(0.32, 0.68)
        cy = h * random.uniform(0.32, 0.68)
        x1 = max(2, cx - bw / 2)
        y1 = max(2, cy - bh / 2)
        x2 = min(w - 2, cx + bw / 2)
        y2 = min(h - 2, cy + bh / 2)
        conf = round(random.uniform(max(conf_th, 0.5), 0.98), 3)
        label = random.choice(classes)
        detections.append({'label': label, 'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]})
        boxes_xyxy.append([x1, y1, x2, y2]); confs.append(conf); labels.append(label)

    detections.sort(key=lambda d: d['confidence'], reverse=True)
    annotated = _draw(image_bgr, boxes_xyxy, confs, labels) if detections else image_bgr.copy()
    infer_ms = round((time.perf_counter() - t0) * 1000 + random.uniform(20, 80), 1)
    return PredictOutcome(detections, annotated, w, h, infer_ms)


def predict(model_name: str, image_bgr: np.ndarray, params: dict) -> PredictOutcome:
    if settings.INFER_BACKEND == 'demo':
        return _predict_demo(model_name, image_bgr, params)
    return _predict_yolo(model_name, image_bgr, params)
