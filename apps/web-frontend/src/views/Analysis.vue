<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  MagicStick,
  Download,
  Document,
  Refresh,
  WarningFilled,
} from '@element-plus/icons-vue'
import type { AnalysisResult, DetectionJob, JobBrief, ReportFormat } from '@/types'
import { listHistory, getJob } from '@/api/history'
import { analyzeJob, downloadReport, mockReportIsHtml } from '@/api/analysis'
import ClassTag from '@/components/ClassTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import { downloadBlob } from '@/utils/download'
import { formatDateTime, jobTypeText } from '@/utils/format'

const route = useRoute()
const router = useRouter()

const jobOptions = ref<JobBrief[]>([])
const selectedId = ref<string>('')
const job = ref<DetectionJob | null>(null)
const analysis = ref<AnalysisResult | null>(null)
const analyzing = ref(false)
const downloading = ref<ReportFormat | ''>('')

onMounted(async () => {
  const res = await listHistory({ page: 1, page_size: 50 })
  jobOptions.value = res.items
  const routeId = route.params.id as string | undefined
  if (routeId) {
    selectedId.value = routeId
  } else if (jobOptions.value.length) {
    selectedId.value = jobOptions.value[0].id
  }
})

watch(selectedId, async (id) => {
  analysis.value = null
  job.value = null
  if (!id) return
  try {
    job.value = await getJob(id)
    // 自动生成分析
    runAnalysis()
  } catch {
    /* handled */
  }
})

async function runAnalysis() {
  if (!selectedId.value) return
  analyzing.value = true
  try {
    analysis.value = await analyzeJob(selectedId.value)
  } catch {
    /* handled */
  } finally {
    analyzing.value = false
  }
}

async function download(format: ReportFormat) {
  if (!job.value) return
  downloading.value = format
  try {
    const blob = await downloadReport(job.value.id, format, analysis.value ?? undefined)
    // mock 下后端未渲染,拿到的是 HTML;真后端拿到 pdf/docx
    const ext = mockReportIsHtml ? 'html' : format
    downloadBlob(blob, `检测报告_${job.value.id.slice(0, 8)}.${ext}`)
    if (mockReportIsHtml) {
      ElMessage.success('已下载 HTML 报告 · 用浏览器打开后「打印 → 另存为 PDF」即可')
    } else {
      ElMessage.success('报告已下载')
    }
  } catch {
    /* handled */
  } finally {
    downloading.value = ''
  }
}

const dominant = computed(() => {
  if (!job.value) return null
  const e = Object.entries(job.value.summary.per_class).sort((a, b) => b[1] - a[1])
  return e[0]?.[0] ?? null
})
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>分析报告</h1>
      <p>调用大模型对检测结果生成结构化分析,并导出企业规范报告</p>
    </div>

    <!-- 任务选择 -->
    <div class="card card-pad picker">
      <span class="picker-label">选择检测任务</span>
      <el-select
        v-model="selectedId"
        placeholder="选择要分析的历史任务"
        filterable
        style="width: 360px"
      >
        <el-option
          v-for="o in jobOptions"
          :key="o.id"
          :label="`${o.id.slice(0, 8)}… · ${jobTypeText(o.type)} · ${o.image_count}张 · 检出${o.total_detections}`"
          :value="o.id"
        />
      </el-select>
      <el-button v-if="job" :icon="Refresh" :loading="analyzing" @click="runAnalysis">
        重新分析
      </el-button>
    </div>

    <EmptyState
      v-if="!selectedId"
      icon="DataAnalysis"
      title="尚无可分析的任务"
      desc="先在「单图检测」或「批量检测」中完成一次检测,再回到这里生成分析报告"
    >
      <el-button type="primary" style="margin-top: 12px" @click="router.push('/detect/single')">
        去检测
      </el-button>
    </EmptyState>

    <div v-else-if="job" class="analysis-grid">
      <!-- 左:任务概要 -->
      <div class="card card-pad summary-col">
        <div class="card-title"><span class="bar" />任务概要</div>
        <div class="sc-item"><span class="k">任务号</span><span class="v mono">{{ job.id.slice(0, 12) }}</span></div>
        <div class="sc-item"><span class="k">类型</span><span class="v">{{ jobTypeText(job.type) }}检测</span></div>
        <div class="sc-item"><span class="k">模型</span><span class="v sc-model">{{ job.model }}</span></div>
        <div class="sc-item"><span class="k">影像数量</span><span class="v mono">{{ job.image_count }}</span></div>
        <div class="sc-item"><span class="k">检出总数</span><span class="v mono">{{ job.summary.count }}</span></div>
        <div class="sc-item"><span class="k">主要类别</span>
          <span class="v"><ClassTag v-if="dominant" :label="dominant" size="small" /><template v-else>—</template></span>
        </div>
        <div class="sc-item"><span class="k">检测时间</span><span class="v">{{ formatDateTime(job.created_at) }}</span></div>

        <div class="sc-cls">
          <ClassTag
            v-for="(c, k) in job.summary.per_class"
            :key="k"
            :label="String(k)"
            :count="c"
            size="small"
          />
        </div>

        <el-divider />
        <div class="dl-title">导出报告</div>
        <div class="dl-btns">
          <el-button
            type="primary"
            :icon="Document"
            :loading="downloading === 'pdf'"
            @click="download('pdf')"
          >
            PDF 报告
          </el-button>
          <el-button
            :icon="Download"
            :loading="downloading === 'docx'"
            @click="download('docx')"
          >
            Word 报告
          </el-button>
        </div>
        <div v-if="mockReportIsHtml" class="dl-note">
          <el-icon><WarningFilled /></el-icon>
          演示模式导出为 HTML,浏览器打印即可转 PDF;接入后端后由 WeasyPrint / python-docx 直接生成 PDF / DOCX。
        </div>
      </div>

      <!-- 右:分析内容 -->
      <div class="card card-pad analysis-col">
        <div class="ac-head">
          <div class="card-title" style="margin-bottom: 0"><span class="bar" />智能分析</div>
          <el-tag v-if="analysis" size="small" effect="plain" type="info">
            {{ analysis.llm_model }}
          </el-tag>
        </div>

        <div v-if="analyzing" class="analyzing">
          <el-icon class="spin"><MagicStick /></el-icon>
          <div class="an-t">大模型分析中…</div>
          <div class="an-s text-muted">正在综合检测结果生成结构化分析</div>
        </div>

        <div v-else-if="analysis" class="sections">
          <section
            v-for="(s, i) in analysis.sections"
            :key="i"
            class="an-section"
            :class="{ disclaimer: s.title.includes('免责') }"
          >
            <h3>{{ s.title }}</h3>
            <p>{{ s.content }}</p>
          </section>
        </div>

        <EmptyState v-else icon="MagicStick" title="点击「重新分析」生成内容" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.picker {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 16px;
}
.picker-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
}
.analysis-grid {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 16px;
  align-items: start;
}
.summary-col {
  position: sticky;
  top: 20px;
}
.sc-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 7px 0;
  border-bottom: 1px dashed var(--border);
}
.sc-item .k {
  font-size: 12.5px;
  color: var(--text-muted);
  flex-shrink: 0;
}
.sc-item .v {
  font-size: 13px;
  color: var(--text);
  text-align: right;
}
.sc-model {
  font-size: 12px;
  word-break: break-all;
}
.sc-cls {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 12px;
}
.dl-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 10px;
}
.dl-btns {
  display: flex;
  gap: 8px;
}
.dl-note {
  display: flex;
  gap: 6px;
  align-items: flex-start;
  margin-top: 12px;
  font-size: 11.5px;
  color: var(--warning);
  background: #fff6ee;
  border: 1px solid #f4dcc0;
  border-radius: 8px;
  padding: 8px 10px;
  line-height: 1.5;
}

.analysis-col {
  min-height: 400px;
}
.ac-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.analyzing {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
}
.spin {
  font-size: 40px;
  color: var(--brand);
  animation: spin 1.4s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
.an-t {
  font-size: 15px;
  font-weight: 600;
  margin-top: 14px;
}
.an-s {
  font-size: 12.5px;
  margin-top: 4px;
}
.sections {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.an-section {
  padding: 14px 16px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--brand);
  border-radius: var(--radius-sm);
}
.an-section h3 {
  font-size: 14px;
  font-weight: 600;
  margin: 0 0 8px;
  color: var(--brand-dark);
}
.an-section p {
  margin: 0;
  font-size: 13px;
  line-height: 1.75;
  color: var(--text-secondary);
  white-space: pre-line;
}
.an-section.disclaimer {
  background: #fff6ee;
  border-color: #f2d5b8;
  border-left-color: var(--warning);
}
.an-section.disclaimer h3 {
  color: #b3701f;
}

@media (max-width: 900px) {
  .analysis-grid {
    grid-template-columns: 1fr;
  }
  .summary-col {
    position: static;
  }
}
</style>
