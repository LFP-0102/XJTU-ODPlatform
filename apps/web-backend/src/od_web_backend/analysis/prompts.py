"""构造发给大模型的提示词。只发检测结果摘要(类别/计数/置信度),绝不发 MRI 原图。"""
from __future__ import annotations

CLASS_CN = {'glioma': '胶质瘤', 'meningioma': '脑膜瘤', 'pituitary': '垂体瘤'}


def cn(label: str) -> str:
    return CLASS_CN.get(label.lower(), label)


def build_summary(job) -> dict:
    """从任务聚合出纯文本可读的检测摘要(结构化)。"""
    images = list(job.images.all().prefetch_related('detections'))
    per_class: dict[str, int] = {}
    confs: list[float] = []
    with_finding = 0
    for im in images:
        dets = list(im.detections.all())
        if dets:
            with_finding += 1
        for d in dets:
            per_class[d.label] = per_class.get(d.label, 0) + 1
            confs.append(d.confidence)
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return {
        'model': job.model,
        'image_count': job.image_count,
        'with_finding': with_finding,
        'total': sum(per_class.values()),
        'per_class': per_class,
        'avg_conf': avg_conf,
    }


def build_prompt(summary: dict) -> str:
    lines = [
        '你是一名资深医学影像分析助理。下面是一次脑部 MRI 肿瘤目标检测的结果摘要,',
        '请基于这些统计信息(不涉及任何患者隐私、也没有原始影像)生成一份结构化的中文分析。',
        '',
        f'检测模型:{summary["model"]}',
        f'影像总数:{summary["image_count"]},其中检出疑似病变的影像 {summary["with_finding"]} 例',
        f'检出目标总数:{summary["total"]}',
        '各类别检出数:'
        + (', '.join(f'{cn(k)}({k}) {v}' for k, v in summary['per_class'].items()) or '无'),
        f'检出置信度均值:{summary["avg_conf"]:.3f}',
        '',
        '请输出 JSON 数组,每个元素形如 {"title": "小节标题", "content": "正文"},',
        '依次包含这几节:总体概述、分类统计解读、置信度与不确定性说明、建议与后续、免责声明。',
        '正文用中文,专业但克制,强调结果仅供临床参考、不能替代医师诊断。只输出 JSON,不要额外解释。',
    ]
    return '\n'.join(lines)
