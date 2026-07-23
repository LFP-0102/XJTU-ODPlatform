from __future__ import annotations

from django.db.models import Q
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import NotFound

from .models import DetectionJob
from .serialize import serialize_job, serialize_brief
from .services import dashboard_stats
from od_web_backend.core.pagination import FrontendPagination


class HistoryListView(APIView):
    def get(self, request):
        qs = DetectionJob.objects.all()
        p = request.query_params

        if p.get('type'):
            qs = qs.filter(type=p['type'])
        if p.get('status'):
            qs = qs.filter(status=p['status'])
        if p.get('model'):
            qs = qs.filter(model=p['model'])
        if p.get('keyword'):
            k = p['keyword']
            qs = qs.filter(Q(model__icontains=k) | Q(images__filename__icontains=k)).distinct()
        if p.get('date_from'):
            qs = qs.filter(created_at__date__gte=p['date_from'])
        if p.get('date_to'):
            qs = qs.filter(created_at__date__lte=p['date_to'])

        paginator = FrontendPagination()
        page = paginator.paginate_queryset(qs, request, view=self)
        data = [serialize_brief(j) for j in page]
        return paginator.get_paginated_response(data)


class JobDetailView(APIView):
    def _get(self, pk) -> DetectionJob:
        try:
            return DetectionJob.objects.prefetch_related('images__detections').get(pk=pk)
        except DetectionJob.DoesNotExist:
            raise NotFound('未找到该任务记录')

    def get(self, request, pk):
        return Response(serialize_job(self._get(pk)))

    def delete(self, request, pk):
        job = self._get(pk)
        job.delete()
        return Response({'code': 0, 'message': '已删除', 'data': None})


class DashboardStatsView(APIView):
    def get(self, request):
        return Response(dashboard_stats())
