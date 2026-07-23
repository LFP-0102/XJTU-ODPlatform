<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import {
  VideoPlay,
  Delete,
  Download,
  CircleClose,
  DataAnalysis,
  Search,
} from '@element-plus/icons-vue'
import type { DetectParams, DetectionJob, ImageResult } from '@/types'
import UploadDropzone from '@/components/UploadDropzone.vue'
import ParamPanel from '@/components/ParamPanel.vue'
import ImageCompare from '@/components/ImageCompare.vue'
import DetectionTable from '@/components/DetectionTable.vue'
import ClassTag from '@/components/ClassTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import { detectBatch, type BatchProgress } from '@/api/detection'
import { exportCsv } from '@/utils/download'
import { formatMs } from '@/utils/format'

const router = useRouter()
const params = ref<DetectParams>({ model: '', conf: 0.25, iou: 0.45, imgsz: 640 })

const files = ref<File[]>([])
const thumbs = ref<string[]>([])
const detecting = ref(false)
const progress = ref<BatchProgress>({ done: 0, total: 0 })
const job = ref<DetectionJob | null>(null)
let cancelFlag = false

// 结果筛选/排序
const filterMode = ref<'all' | 'hit' | 'miss'>('all')
const sortBy = ref<'default' | 'count-desc' | 'count-asc'>('default')

function onFiles(fs: File[]) {
  files.value = fs
  thumbs.value = fs.map((f) => URL.createObjectURL(f))
  job.value = null
}
function removeAt(i: number) {
  files.value.splice(i, 1)
  thumbs.value.splice(i, 1)
}
function clearAll() {
  files.value = []
  thumbs.value = []
  job.value = null
}

async function run() {
  if (!files.value.length) return ElMessage.warning('请先选择图片')
  if (!params.value.model) return ElMessage.warning('请选择检测模型')
  detecting.value = true
  cancelFlag = false
  progress.value = { done: 0, total: files.value.length }
  job.value = null
  try {
    const result = await detectBatch(
      files.value,
      params.value,
      (p) => (progress.value = p),
      () => cancelFlag
    )
    job.value = result
    if (result.status === 'canceled') ElMessage.warning('检测已取消')
    else ElMessage.success(`批量检测完成:${result.image_count} 张,累计检出 ${result.summary.count} 处`)
  } catch {
    /* handled */
  } finally {
    detecting.value = false
  }
}

function cancel() {
  cancelFlag = true
}

const progressPct = computed(() =>
  progress.value.total ? Math.round((progress.value.done / progress.value.total) * 100) : 0
)

// 结果视图
const shownImages = computed<ImageResult[]>(() => {
  if (!job.value) return []
  let list = [...job.value.images]
  if (filterMode.value === 'hit') list = list.filter((i) => i.detections.length > 0)
  else if (filterMode.value === 'miss') list = list.filter((i) => i.detections.length === 0)
  if (sortBy.value === 'count-desc') list.sort((a, b) => b.detections.length - a.detections.length)
  else if (sortBy.value === 'count-asc') list.sort((a, b) => a.detections.length - b.detections.length)
  return list
})

const hitCount = computed(() => job.value?.images.filter((i) => i.detections.length).length ?? 0)

// 详情弹窗
const detailVisible = ref(false)
const detailImg = ref<ImageResult | null>(null)
const detailActive = ref<number | null>(null)
function openDetail(img: ImageResult) {
  detailImg.value = img
  detailActive.value = null
  detailVisible.value = true
}

function exportResults() {
  if (!job.value) return
  const rows: (string | number)[][] = [
    ['图片', '检出数', '类别', '置信度', 'x1', 'y1', 'x2', 'y2', '耗时(ms)'],
  ]
  for (const im of job.value.images) {
    if (!im.detections.length) {
      rows.push([im.filename, 0, '—', '—', '', '', '', '', im.infer_ms])
    } else {
      for (const d of im.detections) {
        rows.push([
          im.filename,
          im.detections.length,
          d.label,
          (d.confidence * 100).toFixed(1) + '%',
          d.bbox[0],
          d.bbox[1],
          d.bbox[2],
          d.bbox[3],
          im.infer_ms,
        ])
      }
    }
  }
  exportCsv(rows, `批量检测_${job.value.id}.csv`)
}

function goAnalysis() {
  if (job.value) router.push(`/analysis/${job.value.id}`)
}
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>批量检测</h1>
      <p>一次上传多张 MRI 影像,逐张检测并汇总记录结果</p>
    </div>

    <div class="grid">
      <!-- 左:输入 -->
      <div class="col-left">
        <div class="card card-pad">
          <div class="card-title">
            <span class="bar" />影像输入
            <span v-if="files.length" class="cnt-badge">{{ files.length }} 张</span>
          </div>
          <UploadDropzone multiple :max-files="50" @change="onFiles" />
          <div v-if="files.length" class="thumbs">
            <div v-for="(t, i) in thumbs" :key="i" class="thumb">
              <img :src="t" />
              <span class="rm" @click="removeAt(i)"><el-icon><CircleClose /></el-icon></span>
            </div>
          </div>
          <el-button v-if="files.length" text size="small" :icon="Delete" class="clear-all" @click="clearAll">
            清空全部
          </el-button>
        </div>

        <div class="card card-pad">
          <div class="card-title"><span class="bar" />检测参数</div>
          <ParamPanel v-model="params" />
        </div>

        <el-button
          v-if="!detecting"
          type="primary"
          size="large"
          :icon="VideoPlay"
          class="run-btn"
          @click="run"
        >
          开始批量检测
        </el-button>
        <el-button v-else type="danger" size="large" :icon="CircleClose" class="run-btn" @click="cancel">
          取消检测
        </el-button>
      </div>

      <!-- 右:结果 -->
      <div class="col-right">
        <!-- 进度 -->
        <div v-if="detecting" class="card card-pad prog-card">
          <div class="prog-head">
            <span>正在检测… <b class="mono">{{ progress.done }}/{{ progress.total }}</b></span>
            <span class="mono">{{ progressPct }}%</span>
          </div>
          <el-progress :percentage="progressPct" :stroke-width="10" :show-text="false" />
          <div v-if="progress.latest" class="prog-latest text-muted">
            最近完成:{{ progress.latest.filename }} · 检出 {{ progress.latest.detections.length }} 处
          </div>
        </div>

        <div class="card card-pad">
          <div class="result-head">
            <div class="card-title" style="margin-bottom: 0">
              <span class="bar" />检测结果
            </div>
            <div v-if="job" class="head-actions">
              <el-button size="small" :icon="Download" @click="exportResults">导出 CSV</el-button>
              <el-button size="small" type="primary" plain :icon="DataAnalysis" @click="goAnalysis">
                生成报告
              </el-button>
            </div>
          </div>

          <EmptyState
            v-if="!job && !detecting"
            icon="Files"
            title="等待批量检测"
            desc="选择多张影像与模型后开始,每张的检测结果会实时汇总到这里"
          />

          <template v-if="job">
            <!-- 汇总 -->
            <div class="batch-summary">
              <div class="bs-item"><span class="n mono">{{ job.image_count }}</span><span class="l">影像总数</span></div>
              <div class="bs-item"><span class="n mono">{{ hitCount }}</span><span class="l">检出病变</span></div>
              <div class="bs-item"><span class="n mono">{{ job.summary.count }}</span><span class="l">检出目标</span></div>
              <div class="bs-item"><span class="n mono">{{ formatMs(job.summary.infer_ms) }}</span><span class="l">总耗时</span></div>
              <div class="bs-cls">
                <ClassTag
                  v-for="(c, k) in job.summary.per_class"
                  :key="k"
                  :label="String(k)"
                  :count="c"
                  size="small"
                />
              </div>
            </div>

            <!-- 筛选栏 -->
            <div class="filter-bar">
              <el-radio-group v-model="filterMode" size="small">
                <el-radio-button value="all">全部 {{ job.image_count }}</el-radio-button>
                <el-radio-button value="hit">有检出 {{ hitCount }}</el-radio-button>
                <el-radio-button value="miss">无检出 {{ job.image_count - hitCount }}</el-radio-button>
              </el-radio-group>
              <el-select v-model="sortBy" size="small" style="width: 150px">
                <el-option label="默认顺序" value="default" />
                <el-option label="检出数 多→少" value="count-desc" />
                <el-option label="检出数 少→多" value="count-asc" />
              </el-select>
            </div>

            <!-- 结果网格 -->
            <div class="result-grid">
              <div
                v-for="im in shownImages"
                :key="im.id"
                class="rcard"
                :class="{ hit: im.detections.length }"
                @click="openDetail(im)"
              >
                <div class="rimg">
                  <img :src="im.result_url" />
                  <span class="rimg-badge" :class="{ zero: !im.detections.length }">
                    {{ im.detections.length }}
                  </span>
                  <div class="rimg-hover"><el-icon><Search /></el-icon> 查看对比</div>
                </div>
                <div class="rmeta">
                  <div class="rname" :title="im.filename">{{ im.filename }}</div>
                  <div class="rtags">
                    <ClassTag
                      v-for="(c, k) in im.summary.per_class"
                      :key="k"
                      :label="String(k)"
                      size="small"
                    />
                    <span v-if="!im.detections.length" class="text-muted rmiss">无检出</span>
                  </div>
                </div>
              </div>
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- 详情弹窗 -->
    <el-dialog v-model="detailVisible" width="82%" top="5vh" :title="detailImg?.filename" class="detail-dialog">
      <div v-if="detailImg" class="detail-body">
        <ImageCompare
          v-model:active="detailActive"
          :original="detailImg.original_url"
          :result="detailImg.result_url"
          :detections="detailImg.detections"
          :width="detailImg.width"
          :height="detailImg.height"
        />
        <div class="detail-det">
          <div class="dd-title">检测明细 · {{ detailImg.detections.length }} 处</div>
          <DetectionTable v-model:active="detailActive" :detections="detailImg.detections" />
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: 340px 1fr;
  gap: 18px;
  align-items: start;
}
.col-left {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.col-right {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.run-btn {
  width: 100%;
}
.cnt-badge {
  margin-left: auto;
  font-size: 12px;
  color: var(--brand);
  background: var(--brand-light);
  padding: 1px 9px;
  border-radius: 12px;
  font-family: var(--font-mono);
}
.thumbs {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin-top: 12px;
  max-height: 200px;
  overflow-y: auto;
}
.thumb {
  position: relative;
  aspect-ratio: 1;
  border-radius: 6px;
  overflow: hidden;
  background: var(--viewer-bg);
}
.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.rm {
  position: absolute;
  top: 2px;
  right: 2px;
  color: #fff;
  background: rgba(0, 0, 0, 0.5);
  border-radius: 50%;
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 12px;
}
.rm:hover {
  background: var(--danger);
}
.clear-all {
  margin-top: 8px;
}

.prog-card {
  padding: 16px 18px;
}
.prog-head {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  margin-bottom: 10px;
}
.prog-latest {
  font-size: 12px;
  margin-top: 8px;
}

.result-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.head-actions {
  display: flex;
  gap: 8px;
}
.batch-summary {
  display: flex;
  gap: 24px;
  align-items: center;
  padding: 14px 16px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.bs-item {
  display: flex;
  flex-direction: column;
}
.bs-item .n {
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
  line-height: 1.1;
}
.bs-item .l {
  font-size: 11px;
  color: var(--text-muted);
}
.bs-cls {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-left: auto;
}
.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
  gap: 10px;
  flex-wrap: wrap;
}
.result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 12px;
}
.rcard {
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  overflow: hidden;
  cursor: pointer;
  transition: all 0.15s;
  background: var(--surface);
}
.rcard:hover {
  border-color: var(--brand);
  box-shadow: var(--shadow);
  transform: translateY(-2px);
}
.rcard.hit {
  border-left: 3px solid var(--brand);
}
.rimg {
  position: relative;
  aspect-ratio: 1;
  background: var(--viewer-bg);
}
.rimg img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.rimg-badge {
  position: absolute;
  top: 6px;
  right: 6px;
  min-width: 22px;
  height: 22px;
  padding: 0 6px;
  border-radius: 11px;
  background: var(--brand);
  color: #fff;
  font-size: 12px;
  font-weight: 600;
  font-family: var(--font-mono);
  display: flex;
  align-items: center;
  justify-content: center;
}
.rimg-badge.zero {
  background: #8a95a3;
}
.rimg-hover {
  position: absolute;
  inset: 0;
  background: rgba(11, 18, 32, 0.55);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 5px;
  font-size: 12.5px;
  opacity: 0;
  transition: opacity 0.15s;
}
.rcard:hover .rimg-hover {
  opacity: 1;
}
.rmeta {
  padding: 8px 10px;
}
.rname {
  font-size: 12px;
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 5px;
}
.rtags {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}
.rmiss {
  font-size: 11.5px;
}
.detail-body {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.dd-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 10px;
}

@media (max-width: 980px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
