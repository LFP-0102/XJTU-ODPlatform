// ============================================================================
// 前后端共享数据契约
// 与《Web 端架构设计方案》§5 的 REST 契约一一对应。
// detections 的 {label, confidence, bbox} 结构对齐引擎 model_infer.hooks.FrameEvent。
// ============================================================================

/** 统一响应封装:后端所有接口都返回 { code, message, data } */
export interface ApiEnvelope<T> {
  code: number
  message: string
  data: T
}

/** 训练模型(来自 models/trained) */
export interface DetectionModel {
  name: string
  task: string // detect / segment
  classes: string[]
  num_classes: number
  size_bytes: number
  updated_at: string
  metrics?: Record<string, number> | null
}

/** 单个检测框 —— 前后端契约核心 */
export interface Detection {
  label: string
  confidence: number
  /** 像素坐标 [x1, y1, x2, y2] */
  bbox: [number, number, number, number]
}

/** 推理参数 */
export interface DetectParams {
  model: string
  conf: number
  iou: number
  imgsz: number
  max_det?: number
}

/** 一张图片的检测结果 */
export interface ImageResult {
  id: string
  filename: string
  original_url: string
  result_url: string
  width: number
  height: number
  detections: Detection[]
  summary: DetectionSummary
  status: JobStatus
  infer_ms: number
  error?: string | null
}

/** 检测汇总 */
export interface DetectionSummary {
  count: number
  per_class: Record<string, number>
  infer_ms: number
}

export type JobType = 'single' | 'batch'
export type JobStatus = 'pending' | 'running' | 'done' | 'failed' | 'canceled'

/** 一次检测任务(单图或批量) */
export interface DetectionJob {
  id: string
  type: JobType
  status: JobStatus
  model: string
  params: DetectParams
  image_count: number
  done_count: number
  summary: DetectionSummary
  images: ImageResult[]
  created_by?: string
  created_at: string
  finished_at?: string | null
  error?: string | null
  has_report?: boolean
}

/** 历史列表里的精简任务(不含逐图明细) */
export interface JobBrief {
  id: string
  type: JobType
  status: JobStatus
  model: string
  image_count: number
  total_detections: number
  dominant_class: string | null
  created_at: string
  has_report: boolean
}

export interface Paginated<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

export interface HistoryQuery {
  page?: number
  page_size?: number
  type?: JobType
  status?: JobStatus
  model?: string
  date_from?: string
  date_to?: string
  keyword?: string
}

// ---- 大模型分析 / 报告 ----
export type LLMProvider = 'dashscope' | 'openai' | 'anthropic' | 'local'

export interface AnalysisSection {
  title: string
  content: string
}

export interface AnalysisResult {
  provider: LLMProvider
  llm_model: string
  sections: AnalysisSection[]
  created_at: string
}

export type ReportFormat = 'pdf' | 'docx'

// ---- 看板统计 ----
export interface DashboardStats {
  total_jobs: number
  total_images: number
  total_detections: number
  avg_confidence: number
  class_distribution: Record<string, number>
  daily_counts: { date: string; jobs: number; detections: number }[]
}
