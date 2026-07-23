from __future__ import annotations

from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError

from . import registry, services
from od_web_backend.history.serialize import serialize_job


def _parse_params(data) -> dict:
    model = (data.get('model') or '').strip()
    if not model:
        raise ValidationError({'model': '缺少模型参数'})

    def _f(key, default, lo, hi):
        try:
            v = float(data.get(key, default))
        except (TypeError, ValueError):
            raise ValidationError({key: '必须是数字'})
        if not (lo <= v <= hi):
            raise ValidationError({key: f'取值需在 {lo}~{hi} 之间'})
        return v

    def _i(key, default):
        try:
            return int(data.get(key, default))
        except (TypeError, ValueError):
            raise ValidationError({key: '必须是整数'})

    return {
        'model': model,
        'conf': _f('conf', 0.25, 0.0, 1.0),
        'iou': _f('iou', 0.45, 0.0, 1.0),
        'imgsz': _i('imgsz', 640),
        'max_det': _i('max_det', 300),
    }


class ModelListView(APIView):
    def get(self, request):
        return Response(registry.list_trained_models())


class ModelSyncView(APIView):
    def post(self, request):
        return Response(registry.sync_models())


class DetectSingleView(APIView):
    def post(self, request):
        image = request.FILES.get('image')
        if image is None:
            raise ValidationError({'image': '请上传一张图片'})
        params = _parse_params(request.data)
        job = services.run_single(image, params, user=_user(request))
        return Response(serialize_job(job))


class DetectBatchView(APIView):
    def post(self, request):
        files = request.FILES.getlist('images')
        if not files:
            raise ValidationError({'images': '请上传至少一张图片'})
        if len(files) > settings.MAX_BATCH_IMAGES:
            raise ValidationError({'images': f'单次最多 {settings.MAX_BATCH_IMAGES} 张'})
        params = _parse_params(request.data)
        job = services.run_batch(files, params, user=_user(request))
        return Response(serialize_job(job))


def _user(request):
    # 预留:接鉴权后返回用户标识
    return None
