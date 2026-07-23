<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { Picture, Files, Aim, TrendCharts, Right } from '@element-plus/icons-vue'
import type { DashboardStats, JobBrief } from '@/types'
import { getDashboardStats, listHistory } from '@/api/history'
import StatCard from '@/components/StatCard.vue'
import ClassTag from '@/components/ClassTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import { classColor } from '@/utils/colors'
import { percent, formatDateTime, jobTypeText, statusMeta } from '@/utils/format'

const router = useRouter()
const stats = ref<DashboardStats | null>(null)
const recent = ref<JobBrief[]>([])
const loading = ref(true)

onMounted(async () => {
  try {
    const [s, h] = await Promise.all([
      getDashboardStats(),
      listHistory({ page: 1, page_size: 6 }),
    ])
    stats.value = s
    recent.value = h.items
  } finally {
    loading.value = false
  }
})

// 类别分布(横向条)
const classBars = computed(() => {
  if (!stats.value) return []
  const entries = Object.entries(stats.value.class_distribution).sort((a, b) => b[1] - a[1])
  const max = Math.max(1, ...entries.map((e) => e[1]))
  return entries.map(([label, count]) => ({
    label,
    count,
    pct: (count / max) * 100,
    share: stats.value!.total_detections ? count / stats.value!.total_detections : 0,
    color: classColor(label),
  }))
})

// 每日趋势折线(检出数)
const trend = computed(() => {
  const data = stats.value?.daily_counts ?? []
  const W = 640
  const H = 150
  const pad = { l: 8, r: 8, t: 12, b: 22 }
  if (!data.length) return { data, path: '', area: '', points: [], W, H, pad, max: 0 }
  const max = Math.max(1, ...data.map((d) => d.detections))
  const iw = W - pad.l - pad.r
  const ih = H - pad.t - pad.b
  const step = data.length > 1 ? iw / (data.length - 1) : 0
  const points = data.map((d, i) => ({
    x: pad.l + i * step,
    y: pad.t + ih - (d.detections / max) * ih,
    ...d,
  }))
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const area = `${path} L${points[points.length - 1].x.toFixed(1)},${pad.t + ih} L${points[0].x.toFixed(1)},${pad.t + ih} Z`
  return { data, path, area, points, W, H, pad, max }
})

const hasData = computed(() => (stats.value?.total_jobs ?? 0) > 0)
</script>

<template>
  <div class="page" v-loading="loading">
    <div class="page-head">
      <h1>概览</h1>
      <p>脑部 MRI 肿瘤检测 · 平台使用情况一览</p>
    </div>

    <EmptyState
      v-if="!loading && !hasData"
      icon="Odometer"
      title="还没有检测数据"
      desc="完成第一次检测后,这里会汇总任务量、检出分布与趋势"
    >
      <div class="empty-actions">
        <el-button type="primary" @click="router.push('/detect/single')">单图检测</el-button>
        <el-button @click="router.push('/detect/batch')">批量检测</el-button>
      </div>
    </EmptyState>

    <template v-else-if="stats">
      <!-- KPI -->
      <div class="kpis">
        <StatCard label="检测任务" :value="stats.total_jobs" icon="Histogram" accent="#0d8b8a" />
        <StatCard label="累计影像" :value="stats.total_images" icon="Picture" accent="#2563eb" />
        <StatCard label="检出目标" :value="stats.total_detections" icon="Aim" accent="#8b5cf6" />
        <StatCard
          label="平均置信度"
          :value="percent(stats.avg_confidence, 1)"
          icon="TrendCharts"
          accent="#e0982c"
        />
      </div>

      <div class="charts-row">
        <!-- 类别分布 -->
        <div class="card card-pad chart-card">
          <div class="card-title"><span class="bar" />检出类别分布</div>
          <div v-if="classBars.length" class="dist">
            <div v-for="b in classBars" :key="b.label" class="dist-row">
              <div class="dist-head">
                <ClassTag :label="b.label" size="small" />
                <span class="dist-val mono">{{ b.count }} · {{ percent(b.share, 0) }}</span>
              </div>
              <div class="dist-track">
                <div class="dist-fill" :style="{ width: b.pct + '%', background: b.color }" />
              </div>
            </div>
          </div>
          <EmptyState v-else icon="PieChart" title="暂无检出" />
        </div>

        <!-- 每日趋势 -->
        <div class="card card-pad chart-card">
          <div class="card-title"><span class="bar" />近期检出趋势</div>
          <div v-if="trend.data.length" class="trend">
            <svg :viewBox="`0 0 ${trend.W} ${trend.H}`" class="trend-svg" preserveAspectRatio="none">
              <defs>
                <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#0d8b8a" stop-opacity="0.28" />
                  <stop offset="100%" stop-color="#0d8b8a" stop-opacity="0" />
                </linearGradient>
              </defs>
              <path :d="trend.area" fill="url(#tg)" />
              <path :d="trend.path" fill="none" stroke="#0d8b8a" stroke-width="2" stroke-linejoin="round" />
              <g v-for="(p, i) in trend.points" :key="i">
                <circle :cx="p.x" :cy="p.y" r="3" fill="#fff" stroke="#0d8b8a" stroke-width="1.6" />
              </g>
            </svg>
            <div class="trend-x">
              <span v-for="(p, i) in trend.points" :key="i" class="tx-lbl">
                {{ p.date.slice(5) }}
              </span>
            </div>
          </div>
          <EmptyState v-else icon="TrendCharts" title="暂无趋势数据" />
        </div>
      </div>

      <!-- 最近任务 -->
      <div class="card recent-card">
        <div class="recent-head">
          <div class="card-title" style="margin-bottom: 0"><span class="bar" />最近任务</div>
          <el-button text type="primary" :icon="Right" @click="router.push('/history')">
            查看全部
          </el-button>
        </div>
        <div v-if="recent.length" class="recent-list">
          <div
            v-for="r in recent"
            :key="r.id"
            class="recent-item"
            @click="router.push(`/history/${r.id}`)"
          >
            <span class="ri-id mono">{{ r.id.slice(0, 8) }}</span>
            <el-tag size="small" :type="r.type === 'batch' ? 'warning' : 'primary'" effect="light">
              {{ jobTypeText(r.type) }}
            </el-tag>
            <span class="ri-imgs text-secondary">{{ r.image_count }} 张 · 检出 {{ r.total_detections }}</span>
            <ClassTag v-if="r.dominant_class" :label="r.dominant_class" size="small" />
            <span class="ri-time text-muted">{{ formatDateTime(r.created_at) }}</span>
            <el-icon class="ri-arrow"><Right /></el-icon>
          </div>
        </div>
        <EmptyState v-else icon="Clock" title="暂无任务记录" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.kpis {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 18px;
}
.charts-row {
  display: grid;
  grid-template-columns: 1fr 1.2fr;
  gap: 16px;
  margin-bottom: 18px;
}
.chart-card {
  min-height: 240px;
}
.dist {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding-top: 6px;
}
.dist-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 7px;
}
.dist-val {
  font-size: 12px;
  color: var(--text-secondary);
}
.dist-track {
  height: 9px;
  background: #eef1f5;
  border-radius: 5px;
  overflow: hidden;
}
.dist-fill {
  height: 100%;
  border-radius: 5px;
  transition: width 0.5s ease;
}
.trend {
  padding-top: 8px;
}
.trend-svg {
  width: 100%;
  height: 150px;
  display: block;
}
.trend-x {
  display: flex;
  justify-content: space-between;
  margin-top: 2px;
}
.tx-lbl {
  font-size: 10px;
  color: var(--text-muted);
  font-family: var(--font-mono);
  flex: 1;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
}
.recent-card {
  padding: 18px;
}
.recent-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.recent-list {
  display: flex;
  flex-direction: column;
}
.recent-item {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 11px 8px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  border-radius: 6px;
  transition: background 0.12s;
}
.recent-item:last-child {
  border-bottom: none;
}
.recent-item:hover {
  background: var(--surface-2);
}
.ri-id {
  font-size: 12.5px;
  color: var(--brand);
  font-weight: 600;
  min-width: 72px;
}
.ri-imgs {
  font-size: 12.5px;
}
.ri-time {
  font-size: 12px;
  margin-left: auto;
}
.ri-arrow {
  color: var(--text-muted);
}
.empty-actions {
  display: flex;
  gap: 10px;
  margin-top: 14px;
}

@media (max-width: 1100px) {
  .kpis {
    grid-template-columns: repeat(2, 1fr);
  }
  .charts-row {
    grid-template-columns: 1fr;
  }
}
</style>
