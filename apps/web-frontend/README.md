# 脑部 MRI 肿瘤检测系统 · Web 前端

基于 **Vue 3 + Vite + TypeScript + Element Plus + Pinia** 的前端,对接 ODPlatform 检测引擎的 Django 后端。

覆盖四大能力:单图检测(原图 / 检测图并列对照)、批量检测(逐张记录 + 汇总)、历史记录、大模型分析报告(可导出)。

## 快速开始

```bash
npm install
npm run dev
```

打开 http://localhost:5173 即可。

> **默认是「演示模式」(Mock 数据)**:后端还没就绪时,前端在浏览器内模拟检测——上传图片后用 canvas 画出检测框,历史记录存在浏览器 localStorage。所有页面无需后端即可完整点通、演示。

## 目录结构

```
src/
├── api/            # 接口层(axios 封装 + 各资源 API)
│   └── mock/       # 演示模式的模拟实现(接后端后不参与,可整个删除)
├── components/     # 复用组件(ImageCompare 并列对比、DetectionTable 等)
├── layouts/        # 应用外壳(侧栏 + 顶栏)
├── router/         # 路由
├── stores/         # Pinia 状态(模型列表、UI)
├── styles/         # 设计令牌 + 全局样式
├── types/          # 前后端共享的 TS 类型契约(对应架构文档 §5)
├── utils/          # 工具(格式化、颜色、下载)
└── views/          # 页面(概览 / 单图 / 批量 / 历史 / 分析 / 模型)
```

## 接入真实后端

1. 把 `.env.development` 里的 `VITE_USE_MOCK` 改为 `false`(生产环境 `.env.production` 已是 `false`)。
2. 后端按《Web 端架构设计方案》§5 的 REST 契约实现,dev 环境由 Vite 代理转发到 `VITE_PROXY_TARGET`(默认 `http://127.0.0.1:8000`),无需改跨域。
3. 接口返回统一信封 `{ code, message, data }`,`code !== 0` 会被拦截器统一报错;`code === 0` 时 `data` 直接交给页面。

对应的后端接口(详见架构文档):

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/models/` | 模型列表 |
| POST | `/api/models/sync/` | 同步模型目录 |
| POST | `/api/detect/single/` | 单图检测 |
| POST | `/api/detect/batch/` | 批量检测(同步) |
| GET | `/api/history/` | 历史列表(分页/筛选) |
| GET | `/api/history/{id}/` | 任务详情 |
| DELETE | `/api/history/{id}/` | 删除任务 |
| POST | `/api/jobs/{id}/analyze/` | 大模型分析 |
| GET | `/api/jobs/{id}/report/?format=pdf\|docx` | 下载报告 |
| GET | `/api/dashboard/stats/` | 概览统计 |

检测框数据契约(与引擎 `model_infer` 对齐):

```ts
interface Detection {
  label: string
  confidence: number
  bbox: [number, number, number, number] // 像素坐标 [x1, y1, x2, y2]
}
```

## 构建

```bash
npm run build      # 产物在 dist/
npm run preview    # 本地预览构建产物
npm run type-check # 可选:全量类型检查(vue-tsc)
```

> `npm run build` 仅用 Vite(esbuild)转译,不阻塞类型检查,构建更稳。需要严格类型校验时单独跑 `npm run type-check`。

## 说明

- 影像输入格式:JPG / PNG / BMP / WEBP(按项目决策,不含 DICOM)。
- 报告:演示模式导出自包含 HTML(浏览器「打印 → 另存为 PDF」即得 PDF);接后端后由 WeasyPrint 出 PDF、python-docx 出 DOCX。
- 医疗免责声明贯穿界面与报告:检测与分析结果仅供临床参考,不能替代专业医师诊断。
