<script setup lang="ts">
import type { Detection } from '@/types'
import ClassTag from './ClassTag.vue'
import { percent } from '@/utils/format'
import { classColor } from '@/utils/colors'

defineProps<{ detections: Detection[] }>()
// 与影像上的框联动:hover 行 → 高亮框(通过 v-model 双向)
const activeIndex = defineModel<number | null>('active', { default: null })

function confBarColor(c: number): string {
  if (c >= 0.8) return 'var(--success)'
  if (c >= 0.5) return 'var(--warning)'
  return 'var(--danger)'
}
</script>

<template>
  <div class="det-table">
    <div v-if="!detections.length" class="none">未检出疑似病变区域</div>
    <div v-else class="rows">
      <div
        v-for="(d, i) in detections"
        :key="i"
        class="row"
        :class="{ active: activeIndex === i }"
        :style="{ '--c': classColor(d.label) }"
        @mouseenter="activeIndex = i"
        @mouseleave="activeIndex = null"
      >
        <div class="idx mono">{{ i + 1 }}</div>
        <div class="cls"><ClassTag :label="d.label" size="small" /></div>
        <div class="conf">
          <div class="bar-wrap">
            <div
              class="bar"
              :style="{ width: percent(d.confidence, 0), background: confBarColor(d.confidence) }"
            />
          </div>
          <span class="conf-val mono">{{ percent(d.confidence) }}</span>
        </div>
        <div class="bbox mono">[{{ d.bbox.join(', ') }}]</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.none {
  color: var(--text-muted);
  font-size: 13px;
  padding: 16px 4px;
  text-align: center;
}
.rows {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.row {
  display: grid;
  grid-template-columns: 28px 1fr 130px 1fr;
  align-items: center;
  gap: 10px;
  padding: 7px 10px;
  border-radius: 8px;
  border: 1px solid transparent;
  cursor: default;
  transition: background 0.12s;
}
.row:hover,
.row.active {
  background: var(--surface-2);
  border-color: var(--c);
}
.idx {
  color: var(--text-muted);
  font-size: 12px;
  text-align: center;
}
.conf {
  display: flex;
  align-items: center;
  gap: 8px;
}
.bar-wrap {
  flex: 1;
  height: 6px;
  background: #eef1f5;
  border-radius: 4px;
  overflow: hidden;
}
.bar {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
}
.conf-val {
  font-size: 12px;
  color: var(--text-secondary);
  min-width: 46px;
  text-align: right;
}
.bbox {
  font-size: 11.5px;
  color: var(--text-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
@media (max-width: 640px) {
  .row {
    grid-template-columns: 24px 1fr 100px;
  }
  .bbox {
    display: none;
  }
}
</style>
