<script setup lang="ts">
import type { DetectParams } from '@/types'
import ModelSelect from './ModelSelect.vue'

const params = defineModel<DetectParams>({ required: true })

const IMGSZ_OPTS = [416, 512, 640, 768, 1024]
</script>

<template>
  <div class="param-panel">
    <div class="field">
      <label>检测模型</label>
      <ModelSelect v-model="params.model" />
    </div>

    <div class="field">
      <label>
        置信度阈值
        <span class="val mono">{{ params.conf.toFixed(2) }}</span>
      </label>
      <el-slider
        v-model="params.conf"
        :min="0.05"
        :max="0.95"
        :step="0.05"
        :show-tooltip="false"
      />
      <div class="hint">仅保留置信度高于该值的检测框</div>
    </div>

    <div class="field">
      <label>
        NMS IoU 阈值
        <span class="val mono">{{ params.iou.toFixed(2) }}</span>
      </label>
      <el-slider
        v-model="params.iou"
        :min="0.1"
        :max="0.9"
        :step="0.05"
        :show-tooltip="false"
      />
      <div class="hint">重叠框合并的严格程度</div>
    </div>

    <div class="field">
      <label>推理尺寸</label>
      <el-radio-group v-model="params.imgsz" size="default">
        <el-radio-button v-for="s in IMGSZ_OPTS" :key="s" :value="s">
          {{ s }}
        </el-radio-button>
      </el-radio-group>
      <div class="hint">越大越准、越慢,常用 640</div>
    </div>
  </div>
</template>

<style scoped>
.param-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.field label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  margin-bottom: 8px;
}
.val {
  color: var(--brand);
  font-weight: 600;
  background: var(--brand-light);
  padding: 1px 8px;
  border-radius: 5px;
  font-size: 12.5px;
}
.hint {
  font-size: 11.5px;
  color: var(--text-muted);
  margin-top: 5px;
}
:deep(.el-radio-button__inner) {
  padding: 8px 12px;
}
</style>
