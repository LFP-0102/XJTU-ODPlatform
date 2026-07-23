<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { VideoPlay, Delete, DataAnalysis, Timer, Aim } from '@element-plus/icons-vue'
import type { DetectParams, DetectionJob } from '@/types'
import UploadDropzone from '@/components/UploadDropzone.vue'
import ParamPanel from '@/components/ParamPanel.vue'
import ImageCompare from '@/components/ImageCompare.vue'
import DetectionTable from '@/components/DetectionTable.vue'
import ClassTag from '@/components/ClassTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import { detectSingle } from '@/api/detection'
import { formatMs } from '@/utils/format'

const router = useRouter()

const params = ref<DetectParams>({ model: '', conf: 0.25, iou: 0.45, imgsz: 640 })
const file = ref<File | null>(null)
const previewUrl = ref('')
const detecting = ref(false)
const job = ref<DetectionJob | null>(null)
const activeIndex = ref<number | null>(null)

function onFile(files: File[]) {
  file.value = files[0]
  previewUrl.value = URL.createObjectURL(files[0])
  job.value = null
}

function clear() {
  file.value = null
  previewUrl.value = ''
  job.value = null
}

async function run() {
  if (!file.value) return ElMessage.warning('请先选择一张 MRI 影像')
  if (!params.value.model) return ElMessage.warning('请选择检测模型')
  detecting.value = true
  try {
    job.value = await detectSingle(file.value, params.value)
    const n = job.value.summary.count
    ElMessage.success(n ? `检测完成,检出 ${n} 处疑似目标` : '检测完成,未检出疑似目标')
  } catch {
    /* 拦截器已提示 */
  } finally {
    detecting.value = false
  }
}

function goAnalysis() {
  if (job.value) router.push(`/analysis/${job.value.id}`)
}

const img = () => job.value?.images[0]
</script>

<template>
  <div class="page single">
    <div class="page-head">
      <h1>单图检测</h1>
      <p>上传单张脑部 MRI 影像,检测疑似肿瘤区域并对照查看</p>
    </div>

    <div class="grid">
      <!-- 左:输入 -->
      <div class="col-left">
        <div class="card card-pad">
          <div class="card-title"><span class="bar" />影像输入</div>
          <UploadDropzone v-if="!file" @change="onFile" />
          <div v-else class="preview">
            <img :src="previewUrl" />
            <div class="preview-info">
              <div class="fname" :title="file.name">{{ file.name }}</div>
              <el-button text size="small" :icon="Delete" @click="clear">移除</el-button>
            </div>
          </div>
        </div>

        <div class="card card-pad">
          <div class="card-title"><span class="bar" />检测参数</div>
          <ParamPanel v-model="params" />
        </div>

        <el-button
          type="primary"
          size="large"
          :icon="VideoPlay"
          :loading="detecting"
          class="run-btn"
          @click="run"
        >
          {{ detecting ? '检测中…' : '开始检测' }}
        </el-button>
      </div>

      <!-- 右:结果 -->
      <div class="col-right">
        <div class="card card-pad result-card">
          <div class="card-title"><span class="bar" />检测结果</div>

          <EmptyState
            v-if="!job && !previewUrl"
            icon="Picture"
            title="等待检测"
            desc="在左侧选择影像与模型,点击「开始检测」后,原图与检测图将并列显示在这里"
          />

          <div v-else-if="!job" class="pending-preview">
            <img :src="previewUrl" />
            <div class="ph text-muted">已就绪,点击「开始检测」查看结果</div>
          </div>

          <template v-else>
            <!-- 汇总条 -->
            <div class="summary-bar">
              <div class="metric">
                <el-icon :size="16"><Aim /></el-icon>
                <div>
                  <div class="m-val mono">{{ job.summary.count }}</div>
                  <div class="m-lbl">检出目标</div>
                </div>
              </div>
              <div class="metric">
                <el-icon :size="16"><Timer /></el-icon>
                <div>
                  <div class="m-val mono">{{ formatMs(job.summary.infer_ms) }}</div>
                  <div class="m-lbl">推理耗时</div>
                </div>
              </div>
              <div class="cls-list">
                <ClassTag
                  v-for="(c, k) in job.summary.per_class"
                  :key="k"
                  :label="String(k)"
                  :count="c"
                  size="small"
                />
                <span v-if="!job.summary.count" class="text-muted no-cls">未检出疑似病变</span>
              </div>
            </div>

            <!-- 并列对比 -->
            <ImageCompare
              v-if="img()"
              v-model:active="activeIndex"
              :original="img()!.original_url"
              :result="img()!.result_url"
              :detections="img()!.detections"
              :width="img()!.width"
              :height="img()!.height"
            />

            <!-- 检测明细 -->
            <div class="det-section">
              <div class="det-head">
                <span class="det-title">检测明细</span>
                <el-button
                  type="primary"
                  plain
                  size="small"
                  :icon="DataAnalysis"
                  @click="goAnalysis"
                >
                  生成分析报告
                </el-button>
              </div>
              <DetectionTable v-model:active="activeIndex" :detections="img()!.detections" />
            </div>
          </template>
        </div>
      </div>
    </div>
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
.run-btn {
  width: 100%;
}
.preview {
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--viewer-bg);
}
.preview img {
  width: 100%;
  max-height: 240px;
  object-fit: contain;
  display: block;
}
.preview-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 10px;
  background: var(--surface-2);
}
.fname {
  font-size: 12.5px;
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 220px;
}
.result-card {
  min-height: 500px;
}
.pending-preview {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 20px;
}
.pending-preview img {
  max-width: 100%;
  max-height: 380px;
  object-fit: contain;
  border-radius: var(--radius-sm);
  background: var(--viewer-bg);
}
.summary-bar {
  display: flex;
  align-items: center;
  gap: 22px;
  padding: 12px 16px;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.metric {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--brand);
}
.m-val {
  font-size: 18px;
  font-weight: 700;
  color: var(--text);
  line-height: 1;
}
.m-lbl {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 2px;
}
.cls-list {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-left: auto;
}
.no-cls {
  font-size: 12.5px;
}
.det-section {
  margin-top: 16px;
}
.det-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}
.det-title {
  font-size: 13px;
  font-weight: 600;
}

@media (max-width: 980px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
</style>
