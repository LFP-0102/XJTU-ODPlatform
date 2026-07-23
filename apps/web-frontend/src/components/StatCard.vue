<script setup lang="ts">
withDefaults(
  defineProps<{
    label: string
    value: string | number
    icon?: string
    accent?: string
    suffix?: string
  }>(),
  { icon: 'DataLine', accent: 'var(--brand)', suffix: '' }
)
</script>

<template>
  <div class="stat-card card">
    <div class="ic" :style="{ background: accent }">
      <el-icon><component :is="icon" /></el-icon>
    </div>
    <div class="body">
      <div class="val mono">{{ value }}<em v-if="suffix">{{ suffix }}</em></div>
      <div class="lbl">{{ label }}</div>
    </div>
  </div>
</template>

<style scoped>
.stat-card {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 18px;
}
.ic {
  width: 44px;
  height: 44px;
  border-radius: 11px;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 21px;
  flex-shrink: 0;
  opacity: 0.92;
}
.val {
  font-size: 24px;
  font-weight: 700;
  line-height: 1.1;
  color: var(--text);
}
.val em {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-muted);
  margin-left: 3px;
  font-style: normal;
}
.lbl {
  font-size: 12.5px;
  color: var(--text-secondary);
  margin-top: 3px;
}
</style>
