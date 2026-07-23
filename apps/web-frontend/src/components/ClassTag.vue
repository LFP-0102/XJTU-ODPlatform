<script setup lang="ts">
import { computed } from 'vue'
import { classColor, withAlpha } from '@/utils/colors'

const props = withDefaults(
  defineProps<{
    label: string
    count?: number | null
    active?: boolean
    size?: 'small' | 'default'
  }>(),
  { count: null, active: false, size: 'default' }
)

const CN: Record<string, string> = {
  glioma: '胶质瘤',
  meningioma: '脑膜瘤',
  pituitary: '垂体瘤',
  notumor: '无肿瘤',
  no_tumor: '无肿瘤',
}
const color = computed(() => classColor(props.label))
const cnName = computed(() => CN[props.label.toLowerCase()] || '')
</script>

<template>
  <span
    class="cls-tag"
    :class="[size, { active }]"
    :style="{
      '--c': color,
      background: active ? color : withAlpha(color, 0.1),
      color: active ? '#fff' : color,
      borderColor: active ? color : withAlpha(color, 0.25),
    }"
  >
    <span class="dot" :style="{ background: active ? '#fff' : color }" />
    <span class="name">{{ label }}<em v-if="cnName">·{{ cnName }}</em></span>
    <span v-if="count !== null" class="cnt">{{ count }}</span>
  </span>
</template>

<style scoped>
.cls-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid;
  border-radius: 20px;
  padding: 2px 10px;
  font-size: 12.5px;
  font-weight: 500;
  line-height: 1.7;
  white-space: nowrap;
}
.cls-tag.small {
  padding: 1px 8px;
  font-size: 11.5px;
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.name em {
  font-style: normal;
  opacity: 0.7;
  margin-left: 1px;
}
.cnt {
  font-family: var(--font-mono);
  font-weight: 600;
  padding-left: 4px;
  border-left: 1px solid currentColor;
  opacity: 0.85;
}
</style>
