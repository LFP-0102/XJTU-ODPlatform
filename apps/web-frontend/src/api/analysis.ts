import type { AnalysisResult, ReportFormat } from '@/types'
import { http, request, USE_MOCK } from './client'
import { getJob } from './history'
import { jobRepo } from './mock/data'
import { buildMockAnalysis, buildReportHtml } from './mock/report'
import { sleep } from './mock/detect'

/** 触发大模型分析 */
export async function analyzeJob(jobId: string): Promise<AnalysisResult> {
  if (USE_MOCK) {
    await sleep(900 + Math.random() * 800) // 模拟 LLM 调用延时
    const job = await getJob(jobId)
    const result = buildMockAnalysis(job)
    const saved = jobRepo.get(jobId)
    if (saved) {
      saved.has_report = true
      jobRepo.save(saved)
    }
    return result
  }
  return request<AnalysisResult>({
    url: `/jobs/${jobId}/analyze/`,
    method: 'post',
  })
}

/** 下载报告。mock 下用前端生成的自包含 HTML(可打印为 PDF)。 */
export async function downloadReport(
  jobId: string,
  format: ReportFormat,
  analysis?: AnalysisResult
): Promise<Blob> {
  if (USE_MOCK) {
    await sleep(400)
    const job = await getJob(jobId)
    const a = analysis ?? buildMockAnalysis(job)
    const html = buildReportHtml(job, a)
    // mock 只能出 HTML(PDF/DOCX 由后端渲染);打印此 HTML 即得 PDF
    return new Blob([html], { type: 'text/html;charset=utf-8' })
  }
  const resp = await http.request({
    url: `/jobs/${jobId}/report/`,
    method: 'get',
    params: { format },
    responseType: 'blob',
  })
  return resp.data as Blob
}

export const mockReportIsHtml = USE_MOCK
