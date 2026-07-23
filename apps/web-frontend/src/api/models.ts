import type { DetectionModel } from '@/types'
import { request, USE_MOCK } from './client'
import { MOCK_MODELS } from './mock/data'
import { sleep } from './mock/detect'

export async function listModels(): Promise<DetectionModel[]> {
  if (USE_MOCK) {
    await sleep(200)
    return structuredClone(MOCK_MODELS)
  }
  return request<DetectionModel[]>({ url: '/models/', method: 'get' })
}

export async function syncModels(): Promise<DetectionModel[]> {
  if (USE_MOCK) {
    await sleep(500)
    return structuredClone(MOCK_MODELS)
  }
  return request<DetectionModel[]>({ url: '/models/sync/', method: 'post' })
}
