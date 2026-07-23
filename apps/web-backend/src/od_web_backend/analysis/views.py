from __future__ import annotations

from django.http import HttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import NotFound, ValidationError

from . import llm, report
from od_web_backend.history.models import DetectionJob, AnalysisReport


def _get_job(pk) -> DetectionJob:
    try:
        return DetectionJob.objects.prefetch_related('images__detections').get(pk=pk)
    except DetectionJob.DoesNotExist:
        raise NotFound('未找到该任务记录')


def _analysis_dict(rep: AnalysisReport) -> dict:
    return {
        'provider': rep.provider,
        'llm_model': rep.llm_model,
        'sections': rep.sections,
        'created_at': rep.created_at.isoformat(),
    }


class AnalyzeView(APIView):
    """POST /api/jobs/<id>/analyze/ —— 触发大模型分析并落库。"""

    def post(self, request, pk):
        job = _get_job(pk)
        result = llm.analyze(job)
        AnalysisReport.objects.update_or_create(
            job=job,
            defaults={
                'provider': result['provider'],
                'llm_model': result['llm_model'],
                'sections': result['sections'],
            },
        )
        if not job.has_report:
            job.has_report = True
            job.save(update_fields=['has_report'])
        return Response(result)


class ReportDownloadView(APIView):
    """GET /api/jobs/<id>/report/?format=pdf|docx —— 返回二进制文件(绕过 JSON 信封)。"""

    def get(self, request, pk):
        job = _get_job(pk)
        fmt = (request.query_params.get('format') or 'pdf').lower()
        if fmt not in ('pdf', 'docx'):
            raise ValidationError({'format': '仅支持 pdf 或 docx'})

        rep = getattr(job, 'report', None)
        if rep is None:
            result = llm.analyze(job)
            rep = AnalysisReport.objects.create(
                job=job, provider=result['provider'],
                llm_model=result['llm_model'], sections=result['sections'],
            )
            job.has_report = True
            job.save(update_fields=['has_report'])
        analysis = _analysis_dict(rep)

        short = str(job.id)[:8]
        if fmt == 'docx':
            content = report.render_docx(job, analysis)
            ctype = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            filename = f'detection_report_{short}.docx'
        else:
            html = report.build_report_html(job, analysis)
            content = report.render_pdf(html)
            ctype = 'application/pdf'
            filename = f'detection_report_{short}.pdf'

        resp = HttpResponse(content, content_type=ctype)
        resp['Content-Disposition'] = f'attachment; filename="{filename}"'
        resp['Content-Length'] = str(len(content))
        return resp
