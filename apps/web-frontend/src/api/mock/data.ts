// Mock 数据仓 —— 种子模型 + localStorage 持久化的任务历史。
// 真接后端后不参与。
import type {
  DetectionJob,
  DetectionModel,
  JobBrief,
  DashboardStats,
} from '@/types'

export const MOCK_MODELS: DetectionModel[] = [
  {
    name: 'brain-tumor-yolo11s-best.pt',
    task: 'detect',
    classes: ['glioma', 'meningioma', 'pituitary'],
    num_classes: 3,
    size_bytes: 22_413_000,
    updated_at: '2026-07-18T09:20:00',
    metrics: { 'mAP50': 0.912, 'mAP50-95': 0.678, precision: 0.9, recall: 0.87 },
  },
  {
    name: 'brain-tumor-yolo11m-best.pt',
    task: 'detect',
    classes: ['glioma', 'meningioma', 'pituitary'],
    num_classes: 3,
    size_bytes: 49_720_000,
    updated_at: '2026-07-20T15:42:00',
    metrics: { 'mAP50': 0.934, 'mAP50-95': 0.712, precision: 0.921, recall: 0.895 },
  },
  {
    name: 'brain-tumor-yolo11n-best.pt',
    task: 'detect',
    classes: ['glioma', 'meningioma', 'pituitary'],
    num_classes: 3,
    size_bytes: 6_240_000,
    updated_at: '2026-07-12T11:05:00',
    metrics: { 'mAP50': 0.883, 'mAP50-95': 0.641, precision: 0.872, recall: 0.84 },
  },
]

const JOBS_KEY = 'od_mock_jobs'

function readJobs(): DetectionJob[] {
  try {
    return JSON.parse(localStorage.getItem(JOBS_KEY) || '[]')
  } catch {
    return []
  }
}
function writeJobs(jobs: DetectionJob[]) {
  // 历史图里含 data URL,只保留最近 40 条,避免撑爆 localStorage
  localStorage.setItem(JOBS_KEY, JSON.stringify(jobs.slice(0, 40)))
}

export const jobRepo = {
  all(): DetectionJob[] {
    return readJobs()
  },
  get(id: string): DetectionJob | undefined {
    return readJobs().find((j) => j.id === id)
  },
  save(job: DetectionJob): void {
    const jobs = readJobs().filter((j) => j.id !== job.id)
    jobs.unshift(job)
    writeJobs(jobs)
  },
  remove(id: string): void {
    writeJobs(readJobs().filter((j) => j.id !== id))
  },
}

export function toBrief(j: DetectionJob): JobBrief {
  const total = j.summary?.count ?? 0
  const per = j.summary?.per_class ?? {}
  const dominant =
    Object.entries(per).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null
  return {
    id: j.id,
    type: j.type,
    status: j.status,
    model: j.model,
    image_count: j.image_count,
    total_detections: total,
    dominant_class: dominant,
    created_at: j.created_at,
    has_report: !!j.has_report,
  }
}

export function computeStats(): DashboardStats {
  const jobs = readJobs()
  let images = 0
  let detections = 0
  let confSum = 0
  let confN = 0
  const classDist: Record<string, number> = {}
  const dailyMap: Record<string, { jobs: number; detections: number }> = {}

  for (const j of jobs) {
    images += j.image_count
    const day = (j.created_at || '').slice(0, 10)
    dailyMap[day] = dailyMap[day] || { jobs: 0, detections: 0 }
    dailyMap[day].jobs += 1
    for (const img of j.images || []) {
      for (const d of img.detections || []) {
        detections += 1
        confSum += d.confidence
        confN += 1
        classDist[d.label] = (classDist[d.label] || 0) + 1
        dailyMap[day].detections += 1
      }
    }
  }

  const daily = Object.entries(dailyMap)
    .map(([date, v]) => ({ date, jobs: v.jobs, detections: v.detections }))
    .sort((a, b) => a.date.localeCompare(b.date))
    .slice(-14)

  return {
    total_jobs: jobs.length,
    total_images: images,
    total_detections: detections,
    avg_confidence: confN ? confSum / confN : 0,
    class_distribution: classDist,
    daily_counts: daily,
  }
}

export function uuid(): string {
  return 'xxxxxxxxxxxx4xxx'.replace(/x/g, () =>
    ((Math.random() * 16) | 0).toString(16)
  )
}
