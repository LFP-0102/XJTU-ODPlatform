import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  { path: '/', redirect: '/dashboard' },
  {
    path: '/dashboard',
    name: 'dashboard',
    component: () => import('@/views/Dashboard.vue'),
    meta: { title: '概览', icon: 'Odometer' },
  },
  {
    path: '/detect/single',
    name: 'single-detect',
    component: () => import('@/views/SingleDetect.vue'),
    meta: { title: '单图检测', icon: 'Picture' },
  },
  {
    path: '/detect/batch',
    name: 'batch-detect',
    component: () => import('@/views/BatchDetect.vue'),
    meta: { title: '批量检测', icon: 'Files' },
  },
  {
    path: '/history',
    name: 'history',
    component: () => import('@/views/History.vue'),
    meta: { title: '历史记录', icon: 'Clock' },
  },
  {
    path: '/history/:id',
    name: 'history-detail',
    component: () => import('@/views/HistoryDetail.vue'),
    meta: { title: '任务详情', hidden: true },
  },
  {
    path: '/analysis/:id?',
    name: 'analysis',
    component: () => import('@/views/Analysis.vue'),
    meta: { title: '分析报告', icon: 'DataAnalysis' },
  },
  {
    path: '/models',
    name: 'models',
    component: () => import('@/views/Models.vue'),
    meta: { title: '模型管理', icon: 'Cpu' },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior: () => ({ top: 0 }),
})

router.afterEach((to) => {
  const base = '脑部 MRI 肿瘤检测系统'
  document.title = to.meta.title ? `${to.meta.title} · ${base}` : base
})

export default router
