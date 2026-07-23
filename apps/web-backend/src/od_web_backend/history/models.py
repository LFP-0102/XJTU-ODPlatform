from __future__ import annotations
import uuid

from django.db import models


class DetectionJob(models.Model):
    """一次检测任务(单图或批量)。"""

    TYPE_CHOICES = [('single', '单图'), ('batch', '批量')]
    STATUS_CHOICES = [
        ('pending', '等待中'),
        ('running', '检测中'),
        ('done', '完成'),
        ('failed', '失败'),
        ('canceled', '已取消'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    type = models.CharField(max_length=16, choices=TYPE_CHOICES, default='single')
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default='done')
    model = models.CharField(max_length=255)
    params = models.JSONField(default=dict)   # {model, conf, iou, imgsz, max_det}
    summary = models.JSONField(default=dict)  # {count, per_class, infer_ms}
    image_count = models.IntegerField(default=0)
    done_count = models.IntegerField(default=0)
    created_by = models.CharField(max_length=64, null=True, blank=True)
    has_report = models.BooleanField(default=False)
    error = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['type']),
            models.Index(fields=['status']),
            models.Index(fields=['model']),
        ]

    def __str__(self):
        return f'{self.type} job {self.id}'


class DetectionImage(models.Model):
    """任务下的一张图片及其检测结果。"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(DetectionJob, related_name='images', on_delete=models.CASCADE)
    order = models.IntegerField(default=0)
    filename = models.CharField(max_length=255)
    original = models.CharField(max_length=512)   # 相对 MEDIA_ROOT 的路径
    result = models.CharField(max_length=512)
    width = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    status = models.CharField(max_length=16, default='done')
    infer_ms = models.FloatField(default=0.0)
    summary = models.JSONField(default=dict)      # {count, per_class, infer_ms}
    error = models.TextField(null=True, blank=True)

    class Meta:
        ordering = ['order']

    def __str__(self):
        return self.filename


class Detection(models.Model):
    """单个检测框(每框一行)。契约: {label, confidence, bbox[x1,y1,x2,y2]}。"""

    image = models.ForeignKey(DetectionImage, related_name='detections', on_delete=models.CASCADE)
    label = models.CharField(max_length=64)
    confidence = models.FloatField()
    x1 = models.FloatField()
    y1 = models.FloatField()
    x2 = models.FloatField()
    y2 = models.FloatField()

    class Meta:
        ordering = ['-confidence']

    @property
    def bbox(self):
        return [round(self.x1), round(self.y1), round(self.x2), round(self.y2)]


class AnalysisReport(models.Model):
    """一次任务的大模型分析结果。"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.OneToOneField(DetectionJob, related_name='report', on_delete=models.CASCADE)
    provider = models.CharField(max_length=32)
    llm_model = models.CharField(max_length=64)
    sections = models.JSONField(default=list)  # [{title, content}]
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'report for {self.job_id}'
