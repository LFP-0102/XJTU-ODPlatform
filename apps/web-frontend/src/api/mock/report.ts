// Mock 大模型分析 + 报告 HTML 生成(后端未就绪时的演示)。
// 真后端:分析由 LLMClient 产出,报告由 WeasyPrint 渲染 PDF / python-docx 渲染 DOCX。
import type { AnalysisResult, AnalysisSection, DetectionJob } from '@/types'
import { formatDateTime, percent } from '@/utils/format'

const CLASS_CN: Record<string, string> = {
  glioma: '胶质瘤',
  meningioma: '脑膜瘤',
  pituitary: '垂体瘤',
}
const cn = (l: string) => CLASS_CN[l.toLowerCase()] || l

export function buildMockAnalysis(job: DetectionJob): AnalysisResult {
  const per = job.summary.per_class
  const entries = Object.entries(per).sort((a, b) => b[1] - a[1])
  const total = job.summary.count
  const withFinding = job.images.filter((i) => i.detections.length > 0).length
  const allConf = job.images.flatMap((i) => i.detections.map((d) => d.confidence))
  const avgConf = allConf.length
    ? allConf.reduce((a, b) => a + b, 0) / allConf.length
    : 0

  const sections: AnalysisSection[] = [
    {
      title: '总体概述',
      content:
        `本次共分析 MRI 影像 ${job.image_count} 例,其中 ${withFinding} 例检出疑似占位性病变,` +
        `累计检出目标 ${total} 处。所用模型为 ${job.model}。` +
        (entries.length
          ? `检出类别以${cn(entries[0][0])}为主(${entries[0][1]} 处)。`
          : '本批次未检出明显疑似病变区域。') +
        `整体检出置信度均值约 ${percent(avgConf)},结果仅供临床医师参考。`,
    },
    {
      title: '分类统计解读',
      content: entries.length
        ? entries
            .map(
              ([k, v]) =>
                `· ${cn(k)}(${k}):${v} 处,占检出总数的 ${percent(v / total, 0)}。`
            )
            .join('\n') +
          '\n上述分布反映本批次的检出构成,不代表确诊比例;同一患者多序列/多层面可能对同一病灶重复计数,需结合原始影像核对。'
        : '本批次未产生分类检出,建议结合临床表现与其他序列综合判断,必要时复查。',
    },
    {
      title: '置信度与不确定性说明',
      content:
        `检出置信度均值约 ${percent(avgConf)}。置信度反映模型对"该区域存在目标"的判断强度,` +
        '并非病变恶性程度或诊断确定性。对置信度偏低(如低于 0.5)的检出,假阳性风险较高,应重点复核;' +
        '对高置信度检出,也需人工确认边界与定位,避免漏诊邻近微小病灶。',
    },
    {
      title: '建议与后续',
      content:
        '1. 由具备资质的放射科/神经外科医师对上述检出逐一复核,结合 T1/T2/FLAIR/增强等多序列判断;\n' +
        '2. 对疑似病灶建议补充增强扫描或随访复查,明确性质与范围;\n' +
        '3. 结合患者临床症状、病史与实验室检查综合评估,制定进一步诊疗方案;\n' +
        '4. 本系统输出为辅助筛查结果,不作为独立诊断依据。',
    },
    {
      title: '免责声明',
      content:
        '本报告由人工智能辅助检测系统自动生成,所有检测与分析结果仅供临床参考,不能替代专业医师的诊断与临床判断。' +
        '任何诊疗决策应由具备资质的医师结合完整临床资料作出。系统开发方不对基于本报告作出的临床决策承担责任。',
    },
  ]

  return {
    provider: 'dashscope',
    llm_model: 'qwen-plus(mock)',
    sections,
    created_at: new Date().toISOString(),
  }
}

/** 生成企业风格的自包含 HTML 报告(可直接打印为 PDF) */
export function buildReportHtml(job: DetectionJob, analysis: AnalysisResult): string {
  const per = job.summary.per_class
  const entries = Object.entries(per).sort((a, b) => b[1] - a[1])
  const imgCards = job.images
    .map((im, i) => {
      const dets = im.detections
        .map(
          (d) =>
            `<tr><td>${cn(d.label)}</td><td>${d.label}</td><td class="mono">${(
              d.confidence * 100
            ).toFixed(1)}%</td><td class="mono">[${d.bbox.join(', ')}]</td></tr>`
        )
        .join('')
      return `
      <div class="img-block">
        <div class="img-title">影像 ${i + 1} · ${escapeHtml(im.filename)}
          <span class="muted">检出 ${im.detections.length} 处 · ${im.width}×${im.height}</span></div>
        <div class="img-pair">
          <figure><img src="${im.original_url}" /><figcaption>原始影像</figcaption></figure>
          <figure><img src="${im.result_url}" /><figcaption>检测标注</figcaption></figure>
        </div>
        ${
          im.detections.length
            ? `<table class="det"><thead><tr><th>类别</th><th>标签</th><th>置信度</th><th>坐标 [x1,y1,x2,y2]</th></tr></thead><tbody>${dets}</tbody></table>`
            : '<div class="no-det">未检出疑似病变区域</div>'
        }
      </div>`
    })
    .join('')

  const sectionsHtml = analysis.sections
    .map(
      (s) =>
        `<section class="analysis"><h3>${escapeHtml(s.title)}</h3><p>${escapeHtml(
          s.content
        ).replace(/\n/g, '<br/>')}</p></section>`
    )
    .join('')

  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/>
<title>脑部 MRI 肿瘤检测报告 · ${job.id}</title>
<style>
  @page { size: A4; margin: 18mm 16mm; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; color:#1f2733; font-size:12.5px; line-height:1.6; }
  .mono { font-family: "SFMono-Regular",Consolas,Menlo,monospace; }
  .muted { color:#909aa6; font-weight:400; font-size:11px; }
  header.rpt { display:flex; justify-content:space-between; align-items:flex-start; border-bottom:2px solid #0d8b8a; padding-bottom:12px; margin-bottom:16px; }
  .brand { display:flex; align-items:center; gap:10px; }
  .brand .logo { width:34px; height:34px; border-radius:8px; background:#0d8b8a; color:#fff; display:flex; align-items:center; justify-content:center; font-weight:700; }
  .brand h1 { font-size:16px; margin:0; }
  .brand p { margin:2px 0 0; color:#56606d; font-size:11px; }
  .meta { text-align:right; font-size:11px; color:#56606d; }
  .summary { display:flex; gap:10px; margin:14px 0 18px; }
  .kpi { flex:1; border:1px solid #e6e9ef; border-radius:8px; padding:10px 12px; background:#fafbfc; }
  .kpi .n { font-size:20px; font-weight:700; color:#0d8b8a; }
  .kpi .l { font-size:11px; color:#56606d; }
  h2.sec { font-size:14px; margin:20px 0 10px; padding-left:9px; border-left:3px solid #0d8b8a; }
  table { width:100%; border-collapse:collapse; margin-top:6px; }
  th,td { border:1px solid #e6e9ef; padding:5px 8px; text-align:left; font-size:11.5px; }
  th { background:#f4f6f9; }
  .img-block { margin-bottom:16px; page-break-inside:avoid; }
  .img-title { font-weight:600; font-size:12.5px; margin-bottom:6px; }
  .img-pair { display:flex; gap:10px; }
  .img-pair figure { flex:1; margin:0; text-align:center; }
  .img-pair img { width:100%; border:1px solid #24304a; border-radius:6px; background:#0b1220; }
  .img-pair figcaption { font-size:10.5px; color:#56606d; margin-top:3px; }
  .no-det { color:#909aa6; font-size:11.5px; padding:6px 0; }
  section.analysis { margin-bottom:10px; page-break-inside:avoid; }
  section.analysis h3 { font-size:12.5px; margin:0 0 4px; color:#0a6e6d; }
  section.analysis p { margin:0; }
  .disclaimer { margin-top:16px; padding:10px 12px; background:#fff6ee; border:1px solid #f2d5b8; border-radius:8px; font-size:11px; color:#8a5a2b; }
  footer.rpt { margin-top:18px; border-top:1px solid #e6e9ef; padding-top:8px; font-size:10px; color:#909aa6; text-align:center; }
</style></head><body>
<header class="rpt">
  <div class="brand">
    <div class="logo">OD</div>
    <div><h1>脑部 MRI 肿瘤检测报告</h1><p>AI 辅助影像检测 · 仅供临床参考</p></div>
  </div>
  <div class="meta">
    报告编号:${job.id}<br/>
    生成时间:${formatDateTime(analysis.created_at)}<br/>
    检测模型:${escapeHtml(job.model)}<br/>
    分析引擎:${escapeHtml(analysis.llm_model)}
  </div>
</header>

<div class="summary">
  <div class="kpi"><div class="n">${job.image_count}</div><div class="l">影像数量</div></div>
  <div class="kpi"><div class="n">${job.summary.count}</div><div class="l">检出目标总数</div></div>
  <div class="kpi"><div class="n">${entries.length}</div><div class="l">涉及类别</div></div>
  <div class="kpi"><div class="n">${entries[0] ? cn(entries[0][0]) : '—'}</div><div class="l">主要类别</div></div>
</div>

<h2 class="sec">一、智能分析</h2>
${sectionsHtml}

<h2 class="sec">二、逐例影像检测</h2>
${imgCards}

<div class="disclaimer">⚠️ 免责声明:本报告由 AI 辅助检测系统自动生成,检测与分析结果仅供临床参考,<b>不能替代专业医师的诊断</b>。任何诊疗决策应由具备资质的医师结合完整临床资料作出。</div>

<footer class="rpt">ODPlatform 脑部 MRI 肿瘤检测系统 · 本页由系统自动生成</footer>
</body></html>`
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}
