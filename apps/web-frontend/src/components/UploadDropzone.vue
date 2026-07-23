<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'

const props = withDefaults(
  defineProps<{ multiple?: boolean; maxFiles?: number }>(),
  { multiple: false, maxFiles: 50 }
)
const emit = defineEmits<{ (e: 'change', files: File[]): void }>()

const dragOver = ref(false)
const inputRef = ref<HTMLInputElement>()

const ACCEPT = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp']

function pickFiles() {
  inputRef.value?.click()
}

function validate(files: File[]): File[] {
  const ok = files.filter((f) => ACCEPT.includes(f.type) || /\.(jpe?g|png|bmp|webp)$/i.test(f.name))
  if (ok.length < files.length) {
    ElMessage.warning('已忽略非图片文件(仅支持 JPG / PNG / BMP / WEBP)')
  }
  if (!props.multiple) return ok.slice(0, 1)
  if (ok.length > props.maxFiles) {
    ElMessage.warning(`单次最多 ${props.maxFiles} 张,已截取前 ${props.maxFiles} 张`)
    return ok.slice(0, props.maxFiles)
  }
  return ok
}

function onInput(e: Event) {
  const files = Array.from((e.target as HTMLInputElement).files || [])
  const ok = validate(files)
  if (ok.length) emit('change', ok)
  ;(e.target as HTMLInputElement).value = ''
}

function onDrop(e: DragEvent) {
  dragOver.value = false
  const files = Array.from(e.dataTransfer?.files || [])
  const ok = validate(files)
  if (ok.length) emit('change', ok)
}
</script>

<template>
  <div
    class="dropzone"
    :class="{ over: dragOver }"
    @click="pickFiles"
    @dragover.prevent="dragOver = true"
    @dragleave.prevent="dragOver = false"
    @drop.prevent="onDrop"
  >
    <input
      ref="inputRef"
      type="file"
      accept="image/*"
      :multiple="multiple"
      hidden
      @change="onInput"
    />
    <el-icon class="ic"><UploadFilled /></el-icon>
    <div class="t">
      拖拽图片到此,或<span class="lk">点击选择</span>
    </div>
    <div class="s">
      {{ multiple ? `支持多选,单次最多 ${maxFiles} 张` : '选择单张 MRI 影像' }} ·
      JPG / PNG / BMP / WEBP
    </div>
  </div>
</template>

<style scoped>
.dropzone {
  border: 1.5px dashed var(--border-strong);
  border-radius: var(--radius);
  background: var(--surface-2);
  padding: 30px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.15s ease;
}
.dropzone:hover,
.dropzone.over {
  border-color: var(--brand);
  background: var(--brand-light);
}
.ic {
  font-size: 40px;
  color: var(--brand);
  margin-bottom: 8px;
}
.t {
  font-size: 14px;
  color: var(--text);
}
.lk {
  color: var(--brand);
  font-weight: 600;
  margin: 0 2px;
}
.s {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 5px;
}
</style>
