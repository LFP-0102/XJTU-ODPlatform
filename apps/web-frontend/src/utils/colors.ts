// 肿瘤类别 → 颜色。与 styles/index.scss 的 --cls-* 保持一致。
// 未知类别按名称哈希稳定分配一个调色板色,保证同名同色。

const KNOWN: Record<string, string> = {
  glioma: '#ef4444',
  meningioma: '#f59e0b',
  pituitary: '#8b5cf6',
  notumor: '#2e9e6b',
  no_tumor: '#2e9e6b',
}

const PALETTE = [
  '#0d8b8a',
  '#ef4444',
  '#f59e0b',
  '#8b5cf6',
  '#2563eb',
  '#db2777',
  '#0891b2',
  '#65a30d',
]

function hash(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = (h << 5) - h + s.charCodeAt(i)
    h |= 0
  }
  return Math.abs(h)
}

export function classColor(label: string): string {
  const key = label.toLowerCase().trim()
  if (KNOWN[key]) return KNOWN[key]
  return PALETTE[hash(key) % PALETTE.length]
}

/** 把 hex 转成带透明度的 rgba(给检测框填充用) */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}
