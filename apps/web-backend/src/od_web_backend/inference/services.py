"""检测任务编排(纯业务逻辑,不碰 request/response;将来可整段搬进 Celery 任务)。

流程:保存上传原图 → 调 predictor → 保存标注图 → 落库(Job/Image/Detection)→ 汇总。
"""
from __future__ import annotations
import logging
import re
import time
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from . import predictor
from od_web_backend.history.models import DetectionJob, DetectionImage, Detection

logger = logging.getLogger(__name__)

_SAFE = re.compile(r'[^A-Za-z0-9._\u4e00-\u9fff-]+')


def _safe_name(name: str) -> str:
    base = Path(name).name
    return _SAFE.sub('_', base) or 'image.jpg'


def _media_path(*parts: str) -> Path:
    p = Path(settings.MEDIA_ROOT).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _rel(path: Path) -> str:
    return str(path.relative_to(settings.MEDIA_ROOT)).replace('\\', '/')


def _read_bgr(path: Path) -> np.ndarray | None:
    # 用 imdecode 读,兼容中文路径
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _write_jpg(path: Path, image_bgr: np.ndarray) -> None:
    ok, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(path))


def _merge_summary(images: list[DetectionImage]) -> dict:
    count = 0
    per_class: dict[str, int] = {}
    infer_ms = 0.0
    for im in images:
        s = im.summary or {}
        count += s.get('count', 0)
        infer_ms += im.infer_ms
        for k, v in s.get('per_class', {}).items():
            per_class[k] = per_class.get(k, 0) + v
    return {'count': count, 'per_class': per_class, 'infer_ms': round(infer_ms, 1)}


def _process_one(job: DetectionJob, uploaded, order: int, params: dict,
                 used_names: set[str]) -> DetectionImage:
    # 文件名去重
    fname = _safe_name(getattr(uploaded, 'name', f'image_{order}.jpg'))
    stem, suffix = Path(fname).stem, Path(fname).suffix or '.jpg'
    candidate = fname
    i = 1
    while candidate in used_names:
        candidate = f'{stem}_{i}{suffix}'
        i += 1
    used_names.add(candidate)

    job_id = str(job.id)
    original_path = _media_path('uploads', job_id, candidate)
    with open(original_path, 'wb') as f:
        for chunk in uploaded.chunks():
            f.write(chunk)

    image_bgr = _read_bgr(original_path)
    if image_bgr is None:
        return DetectionImage.objects.create(
            job=job, order=order, filename=candidate, original=_rel(original_path),
            result='', width=0, height=0, status='failed', infer_ms=0,
            summary={'count': 0, 'per_class': {}, 'infer_ms': 0},
            error='无法解码图片(格式不支持或文件损坏)',
        )

    outcome = predictor.predict(job.model, image_bgr, params)

    result_path = _media_path('results', job_id, f'{Path(candidate).stem}_det.jpg')
    _write_jpg(result_path, outcome.annotated if outcome.annotated is not None else image_bgr)

    per_class: dict[str, int] = {}
    for d in outcome.detections:
        per_class[d['label']] = per_class.get(d['label'], 0) + 1

    img = DetectionImage.objects.create(
        job=job, order=order, filename=candidate,
        original=_rel(original_path), result=_rel(result_path),
        width=outcome.width, height=outcome.height, status='done',
        infer_ms=outcome.infer_ms,
        summary={'count': len(outcome.detections), 'per_class': per_class, 'infer_ms': outcome.infer_ms},
    )
    if outcome.detections:
        Detection.objects.bulk_create([
            Detection(image=img, label=d['label'], confidence=d['confidence'],
                      x1=d['bbox'][0], y1=d['bbox'][1], x2=d['bbox'][2], y2=d['bbox'][3])
            for d in outcome.detections
        ])
    return img


def run_single(uploaded, params: dict, user: str | None = None) -> DetectionJob:
    job = DetectionJob.objects.create(
        type='single', status='running', model=params['model'], params=params,
        image_count=1, created_by=user,
    )
    try:
        img = _process_one(job, uploaded, 0, params, set())
        with transaction.atomic():
            job.summary = _merge_summary([img])
            job.done_count = 1
            job.status = 'done' if img.status == 'done' else 'failed'
            job.finished_at = timezone.now()
            job.save()
    except Exception as exc:
        logger.exception('单图检测失败')
        job.status = 'failed'
        job.error = str(exc)
        job.finished_at = timezone.now()
        job.save()
        raise
    return job


def run_batch(files: list, params: dict, user: str | None = None) -> DetectionJob:
    job = DetectionJob.objects.create(
        type='batch', status='running', model=params['model'], params=params,
        image_count=len(files), created_by=user,
    )
    images: list[DetectionImage] = []
    used_names: set[str] = set()
    try:
        for i, f in enumerate(files):
            images.append(_process_one(job, f, i, params, used_names))
            job.done_count = i + 1
            job.save(update_fields=['done_count'])
        job.summary = _merge_summary(images)
        job.status = 'done'
        job.finished_at = timezone.now()
        job.save()
    except Exception as exc:
        logger.exception('批量检测失败')
        job.summary = _merge_summary(images)
        job.status = 'failed'
        job.error = str(exc)
        job.finished_at = timezone.now()
        job.save()
        raise
    return job
