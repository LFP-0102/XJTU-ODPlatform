from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings


class HealthView(APIView):
    """健康检查 + 后端能力自述(供前端/运维确认当前模式)。"""

    def get(self, request):
        return Response({
            'status': 'ok',
            'infer_backend': settings.INFER_BACKEND,
            'llm_provider': settings.LLM_PROVIDER,
            'device': settings.INFER_DEVICE,
        })
