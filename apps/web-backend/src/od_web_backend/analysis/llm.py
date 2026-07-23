"""大模型分析客户端。

Provider 由 settings.LLM_PROVIDER 决定:
  - template:无需密钥的确定性结构化分析(默认,开箱即用,与前端演示一致)
  - dashscope:通义千问(需 DASHSCOPE_API_KEY + dashscope SDK)
  - openai:OpenAI / 兼容端点(需 OPENAI_API_KEY + openai SDK)

缺密钥 / 缺 SDK / 调用异常 → 一律回退 template,保证接口可用。
只发送检测摘要,绝不发送原始影像。
"""
from __future__ import annotations
import json
import logging

from django.conf import settings

from .prompts import build_summary, build_prompt, cn

logger = logging.getLogger(__name__)


def _template_sections(summary: dict) -> list[dict]:
    per = summary['per_class']
    entries = sorted(per.items(), key=lambda kv: kv[1], reverse=True)
    total = summary['total']
    avg = summary['avg_conf']

    if entries:
        dominant = f'检出类别以{cn(entries[0][0])}为主({entries[0][1]} 处)。'
        dist = '\n'.join(
            f'· {cn(k)}({k}):{v} 处,占检出总数的 {(v / total * 100):.0f}%。'
            for k, v in entries
        ) + '\n上述分布仅反映本批次检出构成,不代表确诊比例;多序列/多层面可能对同一病灶重复计数,需结合原始影像核对。'
    else:
        dominant = '本批次未检出明显疑似病变区域。'
        dist = '本批次未产生分类检出,建议结合临床表现与其他序列综合判断,必要时复查。'

    return [
        {
            'title': '总体概述',
            'content': (
                f'本次共分析 MRI 影像 {summary["image_count"]} 例,其中 {summary["with_finding"]} 例检出疑似占位性病变,'
                f'累计检出目标 {total} 处。所用模型为 {summary["model"]}。{dominant}'
                f'整体检出置信度均值约 {avg * 100:.1f}%,结果仅供临床医师参考。'
            ),
        },
        {'title': '分类统计解读', 'content': dist},
        {
            'title': '置信度与不确定性说明',
            'content': (
                f'检出置信度均值约 {avg * 100:.1f}%。置信度反映模型对"该区域存在目标"的判断强度,'
                '并非病变恶性程度或诊断确定性。对置信度偏低(如低于 0.5)的检出,假阳性风险较高,应重点复核;'
                '对高置信度检出,也需人工确认边界与定位,避免漏诊邻近微小病灶。'
            ),
        },
        {
            'title': '建议与后续',
            'content': (
                '1. 由具备资质的放射科/神经外科医师对上述检出逐一复核,结合 T1/T2/FLAIR/增强等多序列判断;\n'
                '2. 对疑似病灶建议补充增强扫描或随访复查,明确性质与范围;\n'
                '3. 结合患者临床症状、病史与实验室检查综合评估,制定进一步诊疗方案;\n'
                '4. 本系统输出为辅助筛查结果,不作为独立诊断依据。'
            ),
        },
        {
            'title': '免责声明',
            'content': (
                '本报告由人工智能辅助检测系统自动生成,所有检测与分析结果仅供临床参考,不能替代专业医师的诊断与临床判断。'
                '任何诊疗决策应由具备资质的医师结合完整临床资料作出。系统开发方不对基于本报告作出的临床决策承担责任。'
            ),
        },
    ]


def _parse_sections(text: str) -> list[dict] | None:
    """尽量把 LLM 文本解析成 [{title, content}]。"""
    text = text.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.lower().startswith('json'):
            text = text[4:]
    try:
        data = json.loads(text)
        if isinstance(data, list):
            out = []
            for it in data:
                if isinstance(it, dict) and 'title' in it and 'content' in it:
                    out.append({'title': str(it['title']), 'content': str(it['content'])})
            if out:
                return out
    except Exception:
        pass
    return None


def _call_dashscope(prompt: str) -> list[dict] | None:
    try:
        import dashscope  # type: ignore
    except ImportError:
        logger.warning('未安装 dashscope SDK,回退模板分析')
        return None
    if not settings.DASHSCOPE_API_KEY:
        return None
    try:
        resp = dashscope.Generation.call(
            api_key=settings.DASHSCOPE_API_KEY,
            model=settings.LLM_MODEL,
            prompt=prompt,
            result_format='message',
        )
        content = resp.output.choices[0].message.content  # type: ignore
        return _parse_sections(content) or [{'title': '智能分析', 'content': content}]
    except Exception as exc:
        logger.warning('DashScope 调用失败,回退模板: %s', exc)
        return None


def _call_openai(prompt: str) -> list[dict] | None:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        logger.warning('未安装 openai SDK,回退模板分析')
        return None
    if not settings.OPENAI_API_KEY:
        return None
    try:
        kwargs = {'api_key': settings.OPENAI_API_KEY}
        if settings.OPENAI_BASE_URL:
            kwargs['base_url'] = settings.OPENAI_BASE_URL
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
        )
        content = resp.choices[0].message.content
        return _parse_sections(content) or [{'title': '智能分析', 'content': content}]
    except Exception as exc:
        logger.warning('OpenAI 调用失败,回退模板: %s', exc)
        return None


def analyze(job) -> dict:
    """对一次任务生成分析结果(结构对齐前端 AnalysisResult)。"""
    from django.utils import timezone

    summary = build_summary(job)
    provider = settings.LLM_PROVIDER
    sections = None
    used_provider = provider
    used_model = settings.LLM_MODEL

    if provider == 'dashscope':
        sections = _call_dashscope(build_prompt(summary))
    elif provider == 'openai':
        sections = _call_openai(build_prompt(summary))

    if sections is None:
        sections = _template_sections(summary)
        used_provider = 'template'
        used_model = 'rule-based'

    return {
        'provider': used_provider,
        'llm_model': used_model,
        'sections': sections,
        'created_at': timezone.now().isoformat(),
    }
