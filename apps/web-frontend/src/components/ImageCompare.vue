<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue'
import type { Detection } from '@/types'
import { classColor, withAlpha } from '@/utils/colors'
import {
  ZoomIn,
  ZoomOut,
  RefreshLeft,
  Rank,
  Switch,
  PictureRounded,
} from '@element-plus/icons-vue'

const props = withDefaults(
  defineProps<{
    original: string
    result: string
    detections?: Detection[]
    width?: number
    height?: number
  }>(),
  { detections: () => [], width: 0, height: 0 }
)

const activeIndex = defineModel<number | null>('active', { default: null })

type Mode = 'side' | 'slider' | 'boxes'
const mode = ref<Mode>('side')

// ---- 共享缩放/平移(三种模式通用) ----
const t = reactive({ scale: 1, x: 0, y: 0 })
const dragging = ref(false)
let startX = 0
let startY = 0
let startTx = 0
let startTy = 0

const stageStyle = computed(() => ({
  transform: `translate(${t.x}px, ${t.y}px) scale(${t.scale})`,
  cursor: t.scale > 1 ? (dragging.value ? 'grabbing' : 'grab') : 'default',
}))

function clampScale(s: number) {
  return Math.min(8, Math.max(1, s))
}
function zoom(delta: number, cx?: number, cy?: number, rect?: DOMRect) {
  const old = t.scale
  const next = clampScale(+(old + delta).toFixed(2))
  if (next === old) return
  if (cx !== undefined && cy !== undefined && rect) {
    // 以光标为锚点缩放
    const px = cx - rect.left - rect.width / 2
    const py = cy - rect.top - rect.height / 2
    t.x = px - ((px - t.x) * next) / old
    t.y = py - ((py - t.y) * next) / old
  }
  t.scale = next
  if (next === 1) {
    t.x = 0
    t.y = 0
  }
}
function onWheel(e: WheelEvent) {
  e.preventDefault()
  const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
  zoom(e.deltaY < 0 ? 0.3 : -0.3, e.clientX, e.clientY, rect)
}
function onPointerDown(e: PointerEvent) {
  if (t.scale <= 1) return
  dragging.value = true
  startX = e.clientX
  startY = e.clientY
  startTx = t.x
  startTy = t.y
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
}
function onPointerMove(e: PointerEvent) {
  if (!dragging.value) return
  t.x = startTx + (e.clientX - startX)
  t.y = startTy + (e.clientY - startY)
}
function onPointerUp() {
  dragging.value = false
}
function reset() {
  t.scale = 1
  t.x = 0
  t.y = 0
}
watch(mode, reset)

// ---- 滑块模式 ----
const sliderPos = ref(50)
const sliderPaneRef = ref<HTMLElement>()
function startSlider(e: PointerEvent) {
  e.stopPropagation()
  window.addEventListener('pointermove', onSliderMove)
  window.addEventListener('pointerup', endSlider)
}
function onSliderMove(e: PointerEvent) {
  const el = sliderPaneRef.value
  if (!el) return
  const rect = el.getBoundingClientRect()
  const pct = ((e.clientX - rect.left) / rect.width) * 100
  sliderPos.value = Math.min(100, Math.max(0, pct))
}
function endSlider() {
  window.removeEventListener('pointermove', onSliderMove)
  window.removeEventListener('pointerup', endSlider)
}

// ---- 标注框(交互 overlay) ----
const vb = computed(() => `0 0 ${props.width || 1} ${props.height || 1}`)
const strokeW = computed(() => Math.max(props.width, props.height) / 320)
</script>

<template>
  <div class="cmp">
    <!-- 工具条 -->
    <div class="toolbar">
      <el-radio-group v-model="mode" size="small">
        <el-radio-button value="side"><el-icon><Switch /></el-icon> 并列</el-radio-button>
        <el-radio-button value="slider"><el-icon><Rank /></el-icon> 滑块</el-radio-button>
        <el-radio-button value="boxes"><el-icon><PictureRounded /></el-icon> 标注框</el-radio-button>
      </el-radio-group>
      <div class="zoom-ctrl">
        <span class="scale-txt mono">{{ Math.round(t.scale * 100) }}%</span>
        <el-button-group>
          <el-button size="small" :icon="ZoomOut" @click="zoom(-0.3)" />
          <el-button size="small" :icon="ZoomIn" @click="zoom(0.3)" />
          <el-button size="small" :icon="RefreshLeft" @click="reset" />
        </el-button-group>
      </div>
    </div>

    <!-- 并列 -->
    <div v-show="mode === 'side'" class="viewer side">
      <div
        class="pane"
        @wheel="onWheel"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
      >
        <div class="stage" :style="stageStyle"><img :src="original" draggable="false" /></div>
        <span class="badge">原始影像</span>
      </div>
      <div
        class="pane"
        @wheel="onWheel"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
      >
        <div class="stage" :style="stageStyle"><img :src="result" draggable="false" /></div>
        <span class="badge brand">检测标注</span>
      </div>
    </div>

    <!-- 滑块 -->
    <div v-show="mode === 'slider'" class="viewer">
      <div ref="sliderPaneRef" class="pane slider-pane" @wheel="onWheel">
        <div class="stage" :style="stageStyle">
          <div class="slider-imgs">
            <img :src="original" draggable="false" class="base" />
            <img
              :src="result"
              draggable="false"
              class="top"
              :style="{ clipPath: `inset(0 ${100 - sliderPos}% 0 0)` }"
            />
          </div>
        </div>
        <div class="divider" :style="{ left: sliderPos + '%' }" @pointerdown="startSlider">
          <div class="handle"><el-icon><Rank /></el-icon></div>
        </div>
        <span class="badge">原始</span>
        <span class="badge right brand">标注</span>
      </div>
    </div>

    <!-- 标注框(交互) -->
    <div v-show="mode === 'boxes'" class="viewer">
      <div
        class="pane"
        @wheel="onWheel"
        @pointerdown="onPointerDown"
        @pointermove="onPointerMove"
        @pointerup="onPointerUp"
      >
        <div class="stage" :style="stageStyle">
          <div class="box-wrap">
            <img :src="original" draggable="false" />
            <svg class="overlay" :viewBox="vb" preserveAspectRatio="xMidYMid meet">
              <g v-for="(d, i) in detections" :key="i">
                <rect
                  :x="d.bbox[0]"
                  :y="d.bbox[1]"
                  :width="d.bbox[2] - d.bbox[0]"
                  :height="d.bbox[3] - d.bbox[1]"
                  :stroke="classColor(d.label)"
                  :stroke-width="activeIndex === i ? strokeW * 1.8 : strokeW"
                  :fill="activeIndex === i ? withAlpha(classColor(d.label), 0.22) : withAlpha(classColor(d.label), 0.08)"
                  rx="4"
                  class="dbox"
                  :class="{ dim: activeIndex !== null && activeIndex !== i }"
                  @mouseenter="activeIndex = i"
                  @mouseleave="activeIndex = null"
                />
                <text
                  :x="d.bbox[0] + strokeW"
                  :y="d.bbox[1] - strokeW * 2"
                  :fill="classColor(d.label)"
                  :font-size="strokeW * 7"
                  font-weight="700"
                >{{ d.label }} {{ (d.confidence * 100).toFixed(0) }}%</text>
              </g>
            </svg>
          </div>
        </div>
        <span class="badge">交互标注 · 悬停查看</span>
      </div>
    </div>

    <div class="cmp-foot text-muted">
      滚轮缩放 · 放大后拖动平移 · 三种视图切换对照
    </div>
  </div>
</template>

<style scoped>
.cmp {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}
.toolbar :deep(.el-radio-button__inner) {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.zoom-ctrl {
  display: flex;
  align-items: center;
  gap: 10px;
}
.scale-txt {
  font-size: 12px;
  color: var(--text-secondary);
  min-width: 42px;
  text-align: right;
}

.viewer {
  background: var(--viewer-bg);
  border: 1px solid var(--viewer-border);
  border-radius: var(--radius);
  overflow: hidden;
  height: 460px;
}
.viewer.side {
  display: flex;
  gap: 2px;
}
.viewer.side .pane {
  flex: 1;
  border-right: 1px solid var(--viewer-border);
}
.viewer.side .pane:last-child {
  border-right: none;
}
.pane {
  position: relative;
  height: 100%;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  touch-action: none;
}
.stage {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.06s linear;
}
.stage img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  user-select: none;
  display: block;
}
.badge {
  position: absolute;
  top: 10px;
  left: 10px;
  background: rgba(11, 18, 32, 0.82);
  color: var(--viewer-text);
  font-size: 11.5px;
  padding: 3px 9px;
  border-radius: 6px;
  border: 1px solid var(--viewer-border);
  backdrop-filter: blur(4px);
  pointer-events: none;
}
.badge.right {
  left: auto;
  right: 10px;
}
.badge.brand {
  color: #7fe0df;
  border-color: rgba(13, 139, 138, 0.5);
}

/* 滑块 */
.slider-imgs {
  position: relative;
  width: 100%;
  height: 100%;
}
.slider-imgs img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.divider {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: #fff;
  transform: translateX(-1px);
  z-index: 3;
}
.handle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: #fff;
  color: var(--brand);
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: ew-resize;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  touch-action: none;
}

/* 标注框 */
.box-wrap {
  position: relative;
  display: inline-flex;
  max-width: 100%;
  max-height: 100%;
}
.box-wrap img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}
.overlay {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}
.dbox {
  transition: opacity 0.15s, stroke-width 0.15s;
  cursor: pointer;
}
.dbox.dim {
  opacity: 0.35;
}
.cmp-foot {
  font-size: 11.5px;
  text-align: center;
}
</style>
