// ============================================================================
// Mock 推理引擎 —— 后端未就绪时,在浏览器里模拟一次检测。
// 用 <canvas> 在原图上画圆角检测框 + 标签,产出一张 "标注图" data URL,
// 视觉上贴近引擎 BeautifyVisualizer 的输出,让并列对比组件可以直接演示。
// 真接后端后(VITE_USE_MOCK=false)这个文件完全不参与,删掉也不影响。
// ============================================================================
import type { Detection, DetectionSummary } from '@/types'
import { classColor, withAlpha } from '@/utils/colors'

const MOCK_CLASSES = ['glioma', 'meningioma', 'pituitary']

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

/** 生成 0~3 个"像那么回事"的检测框(脑肿瘤多在图像中部偏内) */
function genDetections(w: number, h: number): Detection[] {
  const rng = Math.random()
  const n = rng < 0.12 ? 0 : rng < 0.75 ? 1 : rng < 0.94 ? 2 : 3
  const out: Detection[] = []
  for (let i = 0; i < n; i++) {
    const bw = w * (0.12 + Math.random() * 0.16)
    const bh = h * (0.12 + Math.random() * 0.16)
    const cx = w * (0.32 + Math.random() * 0.36)
    const cy = h * (0.32 + Math.random() * 0.36)
    const x1 = Math.max(2, cx - bw / 2)
    const y1 = Math.max(2, cy - bh / 2)
    const x2 = Math.min(w - 2, cx + bw / 2)
    const y2 = Math.min(h - 2, cy + bh / 2)
    out.push({
      label: MOCK_CLASSES[Math.floor(Math.random() * MOCK_CLASSES.length)],
      confidence: +(0.62 + Math.random() * 0.36).toFixed(3),
      bbox: [Math.round(x1), Math.round(y1), Math.round(x2), Math.round(y2)],
    })
  }
  return out.sort((a, b) => b.confidence - a.confidence)
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number
) {
  const rr = Math.min(r, w / 2, h / 2)
  ctx.beginPath()
  ctx.moveTo(x + rr, y)
  ctx.arcTo(x + w, y, x + w, y + h, rr)
  ctx.arcTo(x + w, y + h, x, y + h, rr)
  ctx.arcTo(x, y + h, x, y, rr)
  ctx.arcTo(x, y, x + w, y, rr)
  ctx.closePath()
}

/** 在原图上绘制标注框,返回标注图 data URL */
export function drawBoxes(img: HTMLImageElement, dets: Detection[]): string {
  const w = img.naturalWidth || img.width
  const h = img.naturalHeight || img.height
  const canvas = document.createElement('canvas')
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(img, 0, 0, w, h)

  const scale = Math.max(w, h) / 640
  const lw = Math.max(2, Math.round(2.4 * scale))
  const fontSize = Math.max(13, Math.round(15 * scale))
  ctx.font = `600 ${fontSize}px ${getComputedStyle(document.body).getPropertyValue('--font-sans') || 'sans-serif'}`
  ctx.textBaseline = 'middle'

  for (const d of dets) {
    const [x1, y1, x2, y2] = d.bbox
    const color = classColor(d.label)
    // 框
    ctx.lineWidth = lw
    ctx.strokeStyle = color
    ctx.fillStyle = withAlpha(color, 0.12)
    roundRect(ctx, x1, y1, x2 - x1, y2 - y1, 8 * scale)
    ctx.fill()
    ctx.stroke()

    // 标签底
    const text = `${d.label} ${(d.confidence * 100).toFixed(0)}%`
    const padX = 8 * scale
    const tw = ctx.measureText(text).width + padX * 2
    const th = fontSize + 8 * scale
    let ly = y1 - th
    if (ly < 0) ly = y1 // 顶到边界就贴内
    ctx.fillStyle = color
    roundRect(ctx, x1, ly, tw, th, 5 * scale)
    ctx.fill()
    // 标签字
    ctx.fillStyle = '#ffffff'
    ctx.fillText(text, x1 + padX, ly + th / 2 + 0.5)
  }
  return canvas.toDataURL('image/jpeg', 0.92)
}

export interface MockDetectOutcome {
  original_url: string
  result_url: string
  width: number
  height: number
  detections: Detection[]
  summary: DetectionSummary
  infer_ms: number
}

/** 对一个上传文件跑一次 mock 检测 */
export async function mockDetectFile(file: File): Promise<MockDetectOutcome> {
  const originalUrl = await fileToDataUrl(file)
  const img = await loadImage(originalUrl)
  const w = img.naturalWidth
  const h = img.naturalHeight
  // 模拟推理耗时
  const infer_ms = +(28 + Math.random() * 90).toFixed(1)
  await sleep(120 + Math.random() * 260)
  const detections = genDetections(w, h)
  const result_url = drawBoxes(img, detections)
  const per_class: Record<string, number> = {}
  for (const d of detections) per_class[d.label] = (per_class[d.label] || 0) + 1
  return {
    original_url: originalUrl,
    result_url,
    width: w,
    height: h,
    detections,
    summary: { count: detections.length, per_class, infer_ms },
    infer_ms,
  }
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onload = () => resolve(r.result as string)
    r.onerror = reject
    r.readAsDataURL(file)
  })
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms))
}
