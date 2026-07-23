<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ArrowLeft, DataAnalysis, Download } from '@element-plus/icons-vue'
import type { DetectionJob, ImageResult } from '@/types'
import { getJob } from '@/api/history'
import ImageCompare from '@/components/ImageCompare.vue'
import DetectionTable from '@/components/DetectionTable.vue'
import ClassTag from '@/components/ClassTag.vue'
import { formatDateTime, formatMs, jobTypeText, statusMeta } from '@/utils/format'
import { exportCsv } from '@/utils/download'

const route = useRoute()
const router = useRouter()

const loading = ref(true)
const job = ref<DetectionJob | null>(null)
const selectedId = ref<string>('')
const activeIndex = ref<number | null>(null)

const selected = computed<ImageResult | null>(
  () => job.value?.images.find((i) => i.id === selectedId.value) ?? job.value?.images[0] ?? null
)

onMounted(async () => {
  try {
    job.value = await getJob(route.params.id as string)
    if (job.value.images.length) selectedId.value = job.value.images[0].id
  } catch {
    /* handled */
  } finally {
    loading.value = false
  }
})

function pick(img: ImageResult) {
  selectedId.value = img.id
  activeIndex.value = null
}

function exportCsvAll() {
  if (!job.value) return
  const rows: (string | number)[][] = [['图片', '类别', '置信度', 'x1', 'y1', 'x2', 'y2']]
  for (const im of job.value.images)
    for (const d of im.detections)
      rows.push([im.filename, d.label, (d.confidence * 100).toFixed(1) + '%', ...d.bbox])
  exportCsv(rows, `任务_${job.value.id}.csv`)
}
</script>

<template>
  <div class="page" v-loading="loading">
    <div class="detail-top">
      <el-button text :icon="ArrowLeft" @click="router.back()">返回</el-button>
    </div>

    <template v-if="job">
      <!-- 头部信息 -->
      <div class="card card-pad head-card">
        <div class="head-main">
          <div class="head-title">
            <span class="jid mono">{{ job.id }}</span>
            <el-tag size="small" :type="job.type === 'batch' ? 'warning' : 'primary'" effect="light">
              {{ jobTypeText(job.type) }}检测
            </el-tag>
            <el-tag size="small" :type="statusMeta(job.status).type" effect="light">
              {{ statusMeta(job.status).text }}
            </el-tag>
          </div>
          <div class="head-actions">
            <el-button size="small" :icon="Download" @click="exportCsvAll">导出 CSV</el-button>
            <el-button size="small" type="primary" :icon="DataAnalysis" @click="router.push(`/analysis/${job.id}`)">
              生成分析报告
            </el-button>
          </div>
        </div>
        <div class="head-meta">
          <div class="hm"><span class="k">模型</span><span class="v">{{ job.model }}</span></div>
          <div class="hm"><span class="k">置信度</span><span class="v mono">{{ job.params.conf }}</span></div>
          <div class="hm"><span class="k">IoU</span><span class="v mono">{{ job.params.iou }}</span></div>
          <div class="hm"><span class="k">尺寸</span><span class="v mono">{{ job.params.imgsz }}</span></div>
          <div class="hm"><span class="k">影像</span><span class="v mono">{{ job.image_count }}</span></div>
          <div class="hm"><span class="k">检出</span><span class="v mono">{{ job.summary.count }}</span></div>
          <div class="hm"><span class="k">耗时</span><span class="v mono">{{ formatMs(job.summary.infer_ms) }}</span></div>
          <div class="hm"><span class="k">时间</span><span class="v">{{ formatDateTime(job.created_at) }}</span></div>
        </div>
        <div class="head-cls">
          <ClassTag
            v-for="(c, k) in job.summary.per_class"
            :key="k"
            :label="String(k)"
            :count="c"
            size="small"
          />
        </div>
      </div>

      <!-- 影像查看 -->
      <div class="detail-grid" :class="{ single: job.images.length <= 1 }">
        <!-- 缩略图侧栏(多图时) -->
        <div v-if="job.images.length > 1" class="gallery card card-pad">
          <div class="card-title"><span class="bar" />影像列表 ({{ job.images.length }})</div>
          <div class="gal-list">
            <div
              v-for="im in job.images"
              :key="im.id"
              class="gal-item"
              :class="{ active: selected?.id === im.id }"
              @click="pick(im)"
            >
              <img :src="im.result_url" />
              <div class="gal-meta">
                <span class="gal-name">{{ im.filename }}</span>
                <span class="gal-cnt mono" :class="{ zero: !im.detections.length }">
                  {{ im.detections.length }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- 对比 + 明细 -->
        <div class="viewer-col">
          <div class="card card-pad">
            <ImageCompare
              v-if="selected"
              v-model:active="activeIndex"
              :original="selected.original_url"
              :result="selected.result_url"
              :detections="selected.detections"
              :width="selected.width"
              :height="selected.height"
            />
          </div>
          <div class="card card-pad">
            <div class="card-title">
              <span class="bar" />检测明细
              <span v-if="selected" class="text-muted" style="margin-left: 6px; font-weight: 400">
                {{ selected.filename }} · {{ selected.detections.length }} 处
              </span>
            </div>
            <DetectionTable v-if="selected" v-model:active="activeIndex" :detections="selected.detections" />
          </div>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.detail-top {
  margin-bottom: 12px;
}
.head-card {
  margin-bottom: 16px;
}
.head-main {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 14px;
}
.head-title {
  display: flex;
  align-items: center;
  gap: 10px;
}
.jid {
  font-size: 14px;
  font-weight: 600;
  color: var(--brand);
}
.head-actions {
  display: flex;
  gap: 8px;
}
.head-meta {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 10px 20px;
  padding: 12px 0;
  border-top: 1px solid var(--border);
}
.hm {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.hm .k {
  font-size: 11px;
  color: var(--text-muted);
}
.hm .v {
  font-size: 13px;
  color: var(--text);
}
.head-cls {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-top: 6px;
}

.detail-grid {
  display: grid;
  grid-template-columns: 240px 1fr;
  gap: 16px;
  align-items: start;
}
.detail-grid.single {
  grid-template-columns: 1fr;
}
.viewer-col {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.gallery {
  position: sticky;
  top: 20px;
}
.gal-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 620px;
  overflow-y: auto;
}
.gal-item {
  display: flex;
  gap: 10px;
  padding: 6px;
  border-radius: 8px;
  border: 1px solid transparent;
  cursor: pointer;
  align-items: center;
}
.gal-item:hover {
  background: var(--surface-2);
}
.gal-item.active {
  border-color: var(--brand);
  background: var(--brand-light);
}
.gal-item img {
  width: 48px;
  height: 48px;
  object-fit: cover;
  border-radius: 6px;
  background: var(--viewer-bg);
  flex-shrink: 0;
}
.gal-meta {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.gal-name {
  font-size: 12px;
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.gal-cnt {
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  background: var(--brand);
  border-radius: 10px;
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.gal-cnt.zero {
  background: #9aa4b1;
}

@media (max-width: 900px) {
  .detail-grid {
    grid-template-columns: 1fr;
  }
  .gallery {
    position: static;
  }
  .gal-list {
    flex-direction: row;
    overflow-x: auto;
  }
  .gal-item {
    flex-direction: column;
    min-width: 90px;
  }
}
</style>
