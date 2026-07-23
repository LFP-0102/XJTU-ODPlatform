<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { storeToRefs } from 'pinia'
import { useModelsStore } from '@/stores/models'
import { formatBytes } from '@/utils/format'

const model = defineModel<string>({ default: '' })
const store = useModelsStore()
const { models, loading } = storeToRefs(store)

onMounted(async () => {
  await store.load()
  if (!model.value && models.value.length) model.value = models.value[0].name
})

const current = computed(() => store.byName(model.value))
</script>

<template>
  <div class="model-select">
    <el-select
      v-model="model"
      :loading="loading"
      placeholder="选择检测模型"
      class="sel"
      popper-class="model-popper"
    >
      <el-option
        v-for="m in models"
        :key="m.name"
        :label="m.name"
        :value="m.name"
      >
        <div class="opt">
          <span class="opt-name">{{ m.name }}</span>
          <span v-if="m.metrics?.['mAP50']" class="opt-map mono">
            mAP50 {{ (m.metrics['mAP50'] * 100).toFixed(1) }}
          </span>
        </div>
      </el-option>
    </el-select>

    <div v-if="current" class="model-meta">
      <span class="mm"><el-icon><PriceTag /></el-icon>{{ current.num_classes }} 类</span>
      <span class="mm mono">{{ formatBytes(current.size_bytes) }}</span>
      <span v-if="current.metrics?.['mAP50-95']" class="mm mono">
        mAP50-95 {{ (current.metrics['mAP50-95'] * 100).toFixed(1) }}
      </span>
    </div>
  </div>
</template>

<style scoped>
.sel {
  width: 100%;
}
.opt {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.opt-map {
  color: var(--brand);
  font-size: 11.5px;
}
.model-meta {
  display: flex;
  gap: 12px;
  margin-top: 8px;
  flex-wrap: wrap;
}
.mm {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: var(--text-secondary);
}
</style>
