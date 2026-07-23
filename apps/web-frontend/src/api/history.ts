import type {
  DashboardStats,
  DetectionJob,
  HistoryQuery,
  JobBrief,
  Paginated,
} from '@/types'
import { request, USE_MOCK } from './client'
import { jobRepo, toBrief, computeStats } from './mock/data'
import { sleep } from './mock/detect'

export async function listHistory(
  query: HistoryQuery = {}
): Promise<Paginated<JobBrief>> {
  if (USE_MOCK) {
    await sleep(180)
    let jobs = jobRepo.all()
    if (query.type) jobs = jobs.filter((j) => j.type === query.type)
    if (query.status) jobs = jobs.filter((j) => j.status === query.status)
    if (query.model) jobs = jobs.filter((j) => j.model === query.model)
    if (query.keyword) {
      const k = query.keyword.toLowerCase()
      jobs = jobs.filter(
        (j) =>
          j.id.includes(k) ||
          j.model.toLowerCase().includes(k) ||
          j.images.some((im) => im.filename.toLowerCase().includes(k))
      )
    }
    if (query.date_from)
      jobs = jobs.filter((j) => j.created_at >= query.date_from!)
    if (query.date_to)
      jobs = jobs.filter((j) => j.created_at <= query.date_to! + 'T23:59:59')

    const page = query.page ?? 1
    const size = query.page_size ?? 10
    const total = jobs.length
    const items = jobs.slice((page - 1) * size, page * size).map(toBrief)
    return { items, total, page, page_size: size }
  }
  return request<Paginated<JobBrief>>({
    url: '/history/',
    method: 'get',
    params: query,
  })
}

export async function getJob(id: string): Promise<DetectionJob> {
  if (USE_MOCK) {
    await sleep(120)
    const job = jobRepo.get(id)
    if (!job) throw new Error('未找到该任务记录')
    return job
  }
  return request<DetectionJob>({ url: `/history/${id}/`, method: 'get' })
}

export async function deleteJob(id: string): Promise<void> {
  if (USE_MOCK) {
    await sleep(120)
    jobRepo.remove(id)
    return
  }
  await request<void>({ url: `/history/${id}/`, method: 'delete' })
}

export async function getDashboardStats(): Promise<DashboardStats> {
  if (USE_MOCK) {
    await sleep(160)
    return computeStats()
  }
  return request<DashboardStats>({ url: '/dashboard/stats/', method: 'get' })
}
