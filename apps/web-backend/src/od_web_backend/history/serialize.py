"""把 ORM 对象序列化成与前端 src/types/index.ts 完全一致的结构。"""
from __future__ import annotations

from django.conf import settings

from .models import DetectionJob, DetectionImage, Detection


def media_url(rel: str) -> str:
    if not rel:
        return ''
    return settings.MEDIA_URL + str(rel).lstrip('/')


def serialize_detection(d: Detection) -> dict:
    return {
        'label': d.label,
        'confidence': round(d.confidence, 4),
        'bbox': d.bbox,
    }


def serialize_image(img: DetectionImage) -> dict:
    return {
        'id': str(img.id),
        'filename': img.filename,
        'original_url': media_url(img.original),
        'result_url': media_url(img.result),
        'width': img.width,
        'height': img.height,
        'detections': [serialize_detection(d) for d in img.detections.all()],
        'summary': img.summary or {'count': 0, 'per_class': {}, 'infer_ms': img.infer_ms},
        'status': img.status,
        'infer_ms': round(img.infer_ms, 1),
        'error': img.error,
    }


def serialize_job(job: DetectionJob) -> dict:
    images = list(job.images.all().prefetch_related('detections'))
    return {
        'id': str(job.id),
        'type': job.type,
        'status': job.status,
        'model': job.model,
        'params': job.params,
        'image_count': job.image_count,
        'done_count': job.done_count,
        'summary': job.summary or {'count': 0, 'per_class': {}, 'infer_ms': 0},
        'images': [serialize_image(im) for im in images],
        'created_by': job.created_by,
        'created_at': job.created_at.isoformat(),
        'finished_at': job.finished_at.isoformat() if job.finished_at else None,
        'error': job.error,
        'has_report': job.has_report,
    }


def _dominant_class(per_class: dict) -> str | None:
    if not per_class:
        return None
    return max(per_class.items(), key=lambda kv: kv[1])[0]


def serialize_brief(job: DetectionJob) -> dict:
    summary = job.summary or {}
    per_class = summary.get('per_class', {})
    return {
        'id': str(job.id),
        'type': job.type,
        'status': job.status,
        'model': job.model,
        'image_count': job.image_count,
        'total_detections': summary.get('count', 0),
        'dominant_class': _dominant_class(per_class),
        'created_at': job.created_at.isoformat(),
        'has_report': job.has_report,
    }
