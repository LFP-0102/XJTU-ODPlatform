<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search, Refresh, View, DataAnalysis, Delete } from '@element-plus/icons-vue'
import type { HistoryQuery, JobBrief } from '@/types'
import { listHistory, deleteJob } from '@/api/history'
import { useModelsStore } from '@/stores/models'
import ClassTag from '@/components/ClassTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import { formatDateTime, jobTypeText, statusMeta } from '@/utils/format'

const router = useRouter()
const modelsStore = useModelsStore()

const loading = ref(false)
const rows = ref<JobBrief[]>([])
const total = ref(0)
const query = reactive<HistoryQuery>({ page: 1, page_size: 10 })
const dateRange = ref<[string, string] | null>(null)

async function fetch() {
  loading.value = true
  try {
    if (dateRange.value) {
      query.date_from = dateRange.value[0]
      query.date_to = dateRange.value[1]
    } else {
      query.date_from = undefined
      query.date_to = undefined
    }
    const res = await listHistory(query)
    rows.value = res.items
    total.value = res.total
  } finally {
    loading.value = false
  }
}

function search() {
  query.page = 1
  fetch()
}
function resetFilters() {
  query.type = undefined
  query.status = undefined
  query.model = undefined
  query.keyword = undefined
  dateRange.value = null
  query.page = 1
  fetch()
}

async function onDelete(row: JobBrief) {
  await ElMessageBox.confirm(`确认删除任务 ${row.id.slice(0, 8)}… 的记录?`, '删除确认', {
    type: 'warning',
    confirmButtonText: '删除',
    cancelButtonText: '取消',
  })
  await deleteJob(row.id)
  ElMessage.success('已删除')
  fetch()
}

onMounted(async () => {
  await modelsStore.load()
  fetch()
})
</script>

<template>
  <div class="page">
    <div class="page-head">
      <h1>历史记录</h1>
      <p>查看、筛选与管理历次检测任务</p>
    </div>

    <!-- 筛选栏 -->
    <div class="card filter card-pad">
      <el-input
        v-model="query.keyword"
        placeholder="搜索任务号 / 文件名"
        clearable
        class="f-item"
        style="width: 220px"
        @keyup.enter="search"
      >
        <template #prefix><el-icon><Search /></el-icon></template>
      </el-input>
      <el-select v-model="query.type" placeholder="类型" clearable class="f-item" style="width: 110px">
        <el-option label="单图" value="single" />
        <el-option label="批量" value="batch" />
      </el-select>
      <el-select v-model="query.status" placeholder="状态" clearable class="f-item" style="width: 120px">
        <el-option label="完成" value="done" />
        <el-option label="已取消" value="canceled" />
        <el-option label="失败" value="failed" />
      </el-select>
      <el-select v-model="query.model" placeholder="模型" clearable filterable class="f-item" style="width: 220px">
        <el-option v-for="m in modelsStore.models" :key="m.name" :label="m.name" :value="m.name" />
      </el-select>
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        value-format="YYYY-MM-DD"
        range-separator="至"
        start-placeholder="开始"
        end-placeholder="结束"
        class="f-item"
        style="width: 240px"
      />
      <el-button type="primary" :icon="Search" @click="search">查询</el-button>
      <el-button :icon="Refresh" @click="resetFilters">重置</el-button>
    </div>

    <!-- 列表 -->
    <div class="card table-card">
      <el-table v-loading="loading" :data="rows" style="width: 100%" row-key="id">
        <el-table-column label="任务号" min-width="120">
          <template #default="{ row }">
            <span class="mono jid">{{ row.id.slice(0, 10) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="类型" width="90">
          <template #default="{ row }">
            <el-tag size="small" :type="row.type === 'batch' ? 'warning' : 'primary'" effect="light">
              {{ jobTypeText(row.type) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="状态" width="100">
          <template #default="{ row }">
            <el-tag size="small" :type="statusMeta(row.status).type" effect="light">
              {{ statusMeta(row.status).text }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="模型" min-width="200" show-overflow-tooltip>
          <template #default="{ row }">
            <span class="model-cell">{{ row.model }}</span>
          </template>
        </el-table-column>
        <el-table-column label="影像" width="80" align="center">
          <template #default="{ row }"><span class="mono">{{ row.image_count }}</span></template>
        </el-table-column>
        <el-table-column label="检出" width="80" align="center">
          <template #default="{ row }"><span class="mono">{{ row.total_detections }}</span></template>
        </el-table-column>
        <el-table-column label="主要类别" width="130">
          <template #default="{ row }">
            <ClassTag v-if="row.dominant_class" :label="row.dominant_class" size="small" />
            <span v-else class="text-muted">—</span>
          </template>
        </el-table-column>
        <el-table-column label="时间" width="150">
          <template #default="{ row }">
            <span class="text-secondary time">{{ formatDateTime(row.created_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="190" fixed="right">
          <template #default="{ row }">
            <el-button text size="small" :icon="View" @click="router.push(`/history/${row.id}`)">
              详情
            </el-button>
            <el-button text size="small" type="primary" :icon="DataAnalysis" @click="router.push(`/analysis/${row.id}`)">
              分析
            </el-button>
            <el-button text size="small" type="danger" :icon="Delete" @click="onDelete(row)" />
          </template>
        </el-table-column>
        <template #empty>
          <EmptyState icon="Clock" title="暂无历史记录" desc="完成一次检测后,记录会显示在这里" />
        </template>
      </el-table>

      <div v-if="total > 0" class="pager">
        <el-pagination
          v-model:current-page="query.page"
          v-model:page-size="query.page_size"
          :total="total"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          @current-change="fetch"
          @size-change="search"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
.filter {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 16px;
}
.table-card {
  padding: 6px 6px 0;
}
.jid {
  font-size: 12.5px;
  color: var(--brand);
}
.model-cell {
  font-size: 12.5px;
}
.time {
  font-size: 12.5px;
}
.pager {
  display: flex;
  justify-content: flex-end;
  padding: 14px 12px;
}
</style>
