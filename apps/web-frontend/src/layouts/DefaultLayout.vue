<script setup lang="ts">
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAppStore } from '@/stores/app'
import { USE_MOCK } from '@/api/client'
import { Fold, Expand } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()
const app = useAppStore()

const menu = [
  { path: '/dashboard', title: '概览', icon: 'Odometer' },
  { path: '/detect/single', title: '单图检测', icon: 'Picture' },
  { path: '/detect/batch', title: '批量检测', icon: 'Files' },
  { path: '/history', title: '历史记录', icon: 'Clock' },
  { path: '/analysis', title: '分析报告', icon: 'DataAnalysis' },
  { path: '/models', title: '模型管理', icon: 'Cpu' },
]

const activePath = computed(() => {
  const p = route.path
  if (p.startsWith('/history')) return '/history'
  if (p.startsWith('/analysis')) return '/analysis'
  return p
})

function go(path: string) {
  if (route.path !== path) router.push(path)
}
</script>

<template>
  <div class="layout" :class="{ collapsed: app.sidebarCollapsed }">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="brand" @click="go('/dashboard')">
        <div class="logo">
          <svg viewBox="0 0 32 32" width="26" height="26">
            <path
              d="M16 6c-4.4 0-6.6 2.9-6.6 5.9 0 1.3.5 2.2.5 2.2s-1.6 1.7-1.6 4.1C8.3 21.5 11.6 24.5 16 24.5s7.7-3 7.7-6.3c0-2.4-1.6-4.1-1.6-4.1s.5-.9.5-2.2C22.6 8.9 20.4 6 16 6z"
              fill="none"
              stroke="#fff"
              stroke-width="1.7"
            />
            <circle cx="16" cy="16" r="2.4" fill="#fff" />
          </svg>
        </div>
        <div v-show="!app.sidebarCollapsed" class="brand-text">
          <span class="t">MRI 检测</span>
          <span class="s">脑肿瘤智能识别</span>
        </div>
      </div>

      <el-menu
        :default-active="activePath"
        :collapse="app.sidebarCollapsed"
        :collapse-transition="false"
        class="nav"
        @select="go"
      >
        <el-menu-item v-for="m in menu" :key="m.path" :index="m.path">
          <el-icon><component :is="m.icon" /></el-icon>
          <template #title>{{ m.title }}</template>
        </el-menu-item>
      </el-menu>

      <div v-show="!app.sidebarCollapsed" class="side-foot">
        <div class="ver">ODPlatform · Web v0.1</div>
      </div>
    </aside>

    <!-- 主区 -->
    <div class="main">
      <header class="topbar">
        <div class="left">
          <el-icon class="collapse-btn" @click="app.toggleSidebar()">
            <Fold v-if="!app.sidebarCollapsed" />
            <Expand v-else />
          </el-icon>
          <span class="crumb">{{ route.meta.title || '概览' }}</span>
          <el-tag v-if="USE_MOCK" size="small" type="warning" effect="light" round>
            演示模式(Mock 数据)
          </el-tag>
        </div>
        <div class="right">
          <span class="disclaimer-mini">
            结果仅供临床参考,不能替代医师诊断
          </span>
          <el-avatar :size="30" class="avatar">医</el-avatar>
        </div>
      </header>

      <main class="content">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </main>
    </div>
  </div>
</template>

<style scoped>
.layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* 侧栏 */
.sidebar {
  width: var(--sidebar-w);
  flex-shrink: 0;
  background: #0f2733;
  background: linear-gradient(180deg, #103038 0%, #0c2029 100%);
  display: flex;
  flex-direction: column;
  transition: width 0.2s ease;
}
.collapsed .sidebar {
  width: 64px;
}
.brand {
  height: var(--header-h);
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 16px;
  cursor: pointer;
  border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}
.logo {
  width: 34px;
  height: 34px;
  border-radius: 9px;
  background: var(--brand);
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
.brand-text {
  display: flex;
  flex-direction: column;
  line-height: 1.25;
  overflow: hidden;
  white-space: nowrap;
}
.brand-text .t {
  color: #fff;
  font-weight: 650;
  font-size: 15px;
}
.brand-text .s {
  color: #7fa7ad;
  font-size: 11px;
}
.nav {
  flex: 1;
  border-right: none;
  background: transparent;
  padding: 10px 8px;
}
.nav :deep(.el-menu-item) {
  color: #a9c2c7;
  border-radius: 8px;
  height: 44px;
  margin-bottom: 3px;
}
.nav :deep(.el-menu-item:hover) {
  background: rgba(255, 255, 255, 0.06);
  color: #fff;
}
.nav :deep(.el-menu-item.is-active) {
  background: var(--brand);
  color: #fff;
}
.nav :deep(.el-menu-item.is-active .el-icon) {
  color: #fff;
}
.side-foot {
  padding: 12px 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.07);
}
.ver {
  color: #5c7a80;
  font-size: 11px;
}

/* 主区 */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.topbar {
  height: var(--header-h);
  flex-shrink: 0;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
}
.topbar .left {
  display: flex;
  align-items: center;
  gap: 14px;
}
.collapse-btn {
  font-size: 19px;
  color: var(--text-secondary);
  cursor: pointer;
}
.collapse-btn:hover {
  color: var(--brand);
}
.crumb {
  font-weight: 600;
  font-size: 15px;
}
.topbar .right {
  display: flex;
  align-items: center;
  gap: 14px;
}
.disclaimer-mini {
  font-size: 12px;
  color: var(--warning);
  background: #fff6ee;
  border: 1px solid #f4dcc0;
  padding: 3px 10px;
  border-radius: 20px;
}
.avatar {
  background: var(--brand);
  font-size: 13px;
}
.content {
  flex: 1;
  overflow-y: auto;
  background: var(--bg);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 720px) {
  .disclaimer-mini {
    display: none;
  }
}
</style>
