import type { DetectParams, DetectionJob, ImageResult, JobStatus } from '@/types'
import { request, USE_MOCK } from './client'
import { mockDetectFile, sleep } from './mock/detect'
import { jobRepo, uuid } from './mock/data'

function emptySummary() {
  return { count: 0, per_class: {} as Record<string, number>, infer_ms: 0 }
}

function buildImageResult(
  file: File,
  outcome: Awaited<ReturnType<typeof mockDetectFile>>,
  status: JobStatus = 'done'
): ImageResult {
  return {
    id: uuid(),
    filename: file.name,
    original_url: outcome.original_url,
    result_url: outcome.result_url,
    width: outcome.width,
    height: outcome.height,
    detections: outcome.detections,
    summary: outcome.summary,
    status,
    infer_ms: outcome.infer_ms,
  }
}

/** 单图检测 —— 同步返回整任务(含一张图) */
export async function detectSingle(
  file: File,
  params: DetectParams
): Promise<DetectionJob> {
  if (USE_MOCK) {
    const outcome = await mockDetectFile(file)
    const img = buildImageResult(file, outcome)
    const job: DetectionJob = {
      id: uuid(),
      type: 'single',
      status: 'done',
      model: params.model,
      params,
      image_count: 1,
      done_count: 1,
      summary: outcome.summary,
      images: [img],
      created_at: new Date().toISOString(),
      finished_at: new Date().toISOString(),
    }
    jobRepo.save(job)
    return job
  }
  const form = new FormData()
  form.append('image', file)
  form.append('model', params.model)
  form.append('conf', String(params.conf))
  form.append('iou', String(params.iou))
  form.append('imgsz', String(params.imgsz))
  return request<DetectionJob>({
    url: '/detect/single/',
    method: 'post',
    data: form,
  })
}

export interface BatchProgress {
  done: number
  total: number
  latest?: ImageResult
}

/**
 * 批量检测 —— 基线同步。mock 下逐张回调 onProgress 模拟 SSE 的"逐张点亮"。
 * 真后端:一次 multipart 提交,同步返回整任务(如需逐张进度,后端走 SSE,这里可扩展)。
 */
export async function detectBatch(
  files: File[],
  params: DetectParams,
  onProgress?: (p: BatchProgress) => void,
  shouldCancel?: () => boolean
): Promise<DetectionJob> {
  if (USE_MOCK) {
    const images: ImageResult[] = []
    const per_class: Record<string, number> = {}
    let count = 0
    let totalMs = 0
    let canceled = false
    for (let i = 0; i < files.length; i++) {
      if (shouldCancel?.()) {
        canceled = true
        break
      }
      const outcome = await mockDetectFile(files[i])
      const img = buildImageResult(files[i], outcome)
      images.push(img)
      count += outcome.summary.count
      totalMs += outcome.infer_ms
      for (const [k, v] of Object.entries(outcome.summary.per_class))
        per_class[k] = (per_class[k] || 0) + v
      onProgress?.({ done: i + 1, total: files.length, latest: img })
    }
    const job: DetectionJob = {
      id: uuid(),
      type: 'batch',
      status: canceled ? 'canceled' : 'done',
      model: params.model,
      params,
      image_count: files.length,
      done_count: images.length,
      summary: { count, per_class, infer_ms: +totalMs.toFixed(1) },
      images,
      created_at: new Date().toISOString(),
      finished_at: new Date().toISOString(),
    }
    jobRepo.save(job)
    return job
  }

  const form = new FormData()
  files.forEach((f) => form.append('images', f))
  form.append('model', params.model)
  form.append('conf', String(params.conf))
  form.append('iou', String(params.iou))
  form.append('imgsz', String(params.imgsz))
  const job = await request<DetectionJob>({
    url: '/detect/batch/',
    method: 'post',
    data: form,
  })
  onProgress?.({ done: job.image_count, total: job.image_count })
  return job
}

export { emptySummary }
export const _sleep = sleep
