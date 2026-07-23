export function formatBytes(bytes: number): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}

export function formatDateTime(iso?: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (isNaN(d.getTime())) return '—'
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(
    d.getMinutes()
  )}`
}

export function timeAgo(iso?: string | null): string {
  if (!iso) return '—'
  const diff = Date.now() - new Date(iso).getTime()
  const min = Math.floor(diff / 60000)
  if (min < 1) return '刚刚'
  if (min < 60) return `${min} 分钟前`
  const hr = Math.floor(min / 60)
  if (hr < 24) return `${hr} 小时前`
  const day = Math.floor(hr / 24)
  if (day < 30) return `${day} 天前`
  return formatDateTime(iso)
}

export function percent(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

const STATUS_MAP: Record<string, { text: string; type: 'success' | 'info' | 'warning' | 'danger' | 'primary' }> = {
  done: { text: '完成', type: 'success' },
  running: { text: '检测中', type: 'primary' },
  pending: { text: '等待中', type: 'info' },
  failed: { text: '失败', type: 'danger' },
  canceled: { text: '已取消', type: 'warning' },
}
export function statusMeta(status: string) {
  return STATUS_MAP[status] ?? { text: status, type: 'info' as const }
}

const TYPE_MAP: Record<string, string> = { single: '单图', batch: '批量' }
export function jobTypeText(t: string): string {
  return TYPE_MAP[t] ?? t
}
