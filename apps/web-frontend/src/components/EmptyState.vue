<script setup lang="ts">
withDefaults(
  defineProps<{ icon?: string; title?: string; desc?: string }>(),
  { icon: 'DocumentDelete', title: '暂无数据', desc: '' }
)
</script>

<template>
  <div class="empty">
    <el-icon class="ei"><component :is="icon" /></el-icon>
    <div class="t">{{ title }}</div>
    <div v-if="desc" class="d">{{ desc }}</div>
    <slot />
  </div>
</template>

<style scoped>
.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 44px 20px;
  color: var(--text-muted);
  text-align: center;
}
.ei {
  font-size: 44px;
  color: var(--border-strong);
  margin-bottom: 12px;
}
.t {
  font-size: 14px;
  color: var(--text-secondary);
  font-weight: 500;
}
.d {
  font-size: 12.5px;
  margin-top: 4px;
  max-width: 320px;
}
</style>
