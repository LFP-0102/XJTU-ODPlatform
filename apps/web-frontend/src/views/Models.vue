<script setup lang="ts">
import { onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { ElMessage } from 'element-plus'
import { Refresh, Cpu } from '@element-plus/icons-vue'
import { useModelsStore } from '@/stores/models'
import ClassTag from '@/components/ClassTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import { formatBytes, formatDateTime } from '@/utils/format'

const store = useModelsStore()
const { models, loading } = storeToRefs(store)

onMounted(() => store.load())

async function sync() {
  await store.sync()
  ElMessage.success(`已同步,共 ${models.value.length} 个模型`)
}

function metricPairs(m: Record<string, number> | null | undefined) {
  if (!m) return []
  return Object.entries(m)
}
</script>

<template>
  <div class="page">
    <div class="page-head head-row">
      <div>
        <h1>模型管理</h1>
        <p>来自 <code>models/trained</code> 的已训练检测模型</p>
      </div>
      <el-button type="primary" :icon="Refresh" :loading="loading" @click="sync">
        同步模型目录
      </el-button>
    </div>

    <div v-loading="loading" class="model-list">
      <div v-for="m in models" :key="m.name" class="card model-card">
        <div class="mc-head">
          <div class="mc-icon"><el-icon :size="20"><Cpu /></el-icon></div>
          <div class="mc-title">
            <div class="mc-name" :title="m.name">{{ m.name }}</div>
            <div class="mc-sub">
              <el-tag size="small" effect="light" type="info">{{ m.task }}</el-tag>
              <span class="text-muted mono">{{ formatBytes(m.size_bytes) }}</span>
            </div>
          </div>
        </div>

        <div class="mc-section">
          <div class="mc-label">检测类别 ({{ m.num_classes }})</div>
          <div class="mc-classes">
            <ClassTag v-for="c in m.classes" :key="c" :label="c" size="small" />
          </div>
        </div>

        <div v-if="metricPairs(m.metrics).length" class="mc-section">
          <div class="mc-label">评估指标</div>
          <div class="mc-metrics">
            <div v-for="[k, v] in metricPairs(m.metrics)" :key="k" class="metric-chip">
              <span class="mk">{{ k }}</span>
              <span class="mv mono">{{ (v * 100).toFixed(1) }}</span>
            </div>
          </div>
        </div>

        <div class="mc-foot text-muted">
          更新于 {{ formatDateTime(m.updated_at) }}
        </div>
      </div>

      <EmptyState
        v-if="!loading && !models.length"
        icon="Cpu"
        title="未发现模型"
        desc="请确认 models/trained 目录下存在训练好的权重文件"
      />
    </div>
  </div>
</template>

<style scoped>
.head-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}
.head-row code {
  background: var(--surface-2);
  padding: 1px 6px;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--brand);
}
.model-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 16px;
}
.model-card {
  padding: 18px;
  transition: box-shadow 0.15s, transform 0.15s;
}
.model-card:hover {
  box-shadow: var(--shadow);
  transform: translateY(-2px);
}
.mc-head {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
}
.mc-icon {
  width: 42px;
  height: 42px;
  border-radius: 10px;
  background: var(--brand-light);
  color: var(--brand);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.mc-title {
  min-width: 0;
  flex: 1;
}
.mc-name {
  font-size: 14px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.mc-sub {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
  font-size: 12px;
}
.mc-section {
  margin-bottom: 14px;
}
.mc-label {
  font-size: 11.5px;
  color: var(--text-muted);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.mc-classes {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.mc-metrics {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.metric-chip {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 6px 12px;
  min-width: 76px;
}
.mk {
  font-size: 10.5px;
  color: var(--text-muted);
}
.mv {
  font-size: 16px;
  font-weight: 700;
  color: var(--brand);
}
.mc-foot {
  font-size: 11.5px;
  border-top: 1px solid var(--border);
  padding-top: 10px;
  margin-top: 2px;
}
</style>
