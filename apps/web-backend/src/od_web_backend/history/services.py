from __future__ import annotations
from collections import defaultdict

from django.db.models import Count, Avg, Sum

from .models import DetectionJob, Detection


def dashboard_stats() -> dict:
    jobs = DetectionJob.objects.all()
    dets = Detection.objects.all()

    total_jobs = jobs.count()
    total_images = jobs.aggregate(s=Sum('image_count'))['s'] or 0
    total_detections = dets.count()
    avg_conf = dets.aggregate(a=Avg('confidence'))['a'] or 0.0

    class_distribution = {
        row['label']: row['c']
        for row in dets.values('label').annotate(c=Count('id')).order_by('-c')
    }

    daily = defaultdict(lambda: {'jobs': 0, 'detections': 0})
    for row in jobs.values('created_at__date').annotate(c=Count('id')):
        daily[str(row['created_at__date'])]['jobs'] = row['c']
    for row in dets.values('image__job__created_at__date').annotate(c=Count('id')):
        key = str(row['image__job__created_at__date'])
        daily[key]['detections'] = row['c']

    daily_counts = [
        {'date': d, 'jobs': v['jobs'], 'detections': v['detections']}
        for d, v in sorted(daily.items())
        if d != 'None'
    ][-14:]

    return {
        'total_jobs': total_jobs,
        'total_images': total_images,
        'total_detections': total_detections,
        'avg_confidence': round(avg_conf, 4),
        'class_distribution': class_distribution,
        'daily_counts': daily_counts,
    }
