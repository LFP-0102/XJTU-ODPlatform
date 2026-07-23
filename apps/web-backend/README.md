# 脑部 MRI 肿瘤检测系统 · Web 后端

基于 **Django 5 + DRF** 的后端,复用同仓 `od_platform` 引擎(推理 / 可视化 / 模型解析),
为前端提供检测、历史、大模型分析与报告导出等接口。

## 放置位置(重要)

后端依赖同仓的引擎,必须放进 monorepo 里、与引擎并列。解压后目录应为:

```
<仓库根>/
├── .odp-workspace                 ← 引擎用来定位仓库根的标记(必须存在)
├── models/
│   └── trained/                   ← 训练好的权重放这里(*.pt / *.onnx ...)
└── apps/
    ├── platform/
    │   └── src/od_platform/        ← 现有引擎
    └── web-backend/                ← 本后端(解压到这里)
        ├── manage.py
        ├── pyproject.toml
        └── src/od_web_backend/
```

`settings.py` 会自动向上找 `.odp-workspace`,并把 `apps/platform/src` 挂到 `sys.path`,
因此**无需单独安装引擎**即可 `import od_platform`。

> 若引擎的 `visualization/assets/` 下缺少中文字体(`LXGWWenKai-Bold.ttf`),标注图上的中文标签
> 可能无法正常显示。补上该字体,或在 `.env` 里用 `VIZ_FONT_PATH` 指向一个系统 CJK 字体即可。

## 快速开始

```bash
cd apps/web-backend

# 1) 装依赖(建议用虚拟环境)
pip install -e .
#   真实推理另装(生产):pip install -e ".[yolo]"      # ultralytics + torch
#   更高保真 PDF(可选):pip install -e ".[weasyprint]" # 需系统库 cairo/pango
#   大模型(可选)   :pip install -e ".[dashscope]" 或 ".[openai]"

# 2) 配置
cp .env.example .env        # 按需修改

# 3) 建表 + 启动
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

服务起在 `http://127.0.0.1:8000`,前端 dev 代理已指向这里。

## 两种推理后端

用 `.env` 里的 `INFER_BACKEND` 切换:

| 值 | 说明 | 依赖 |
|---|---|---|
| `yolo` | **真实推理(生产默认)**。用 ultralytics 加载 `models/trained/` 下的权重跑检测,BeautifyVisualizer 出图。 | 需 `.[yolo]` + 权重文件 |
| `demo` | **演示后端**。免 torch,生成合成检测框,仍用真实可视化器出图。无模型 / 无 GPU 时用于打通全链路、演示界面。 | 仅核心依赖 |

**上真实模型的步骤**:把训练好的权重(如 `brain-tumor-best.pt`)放进 `<仓库根>/models/trained/`,
`INFER_BACKEND=yolo`,重启即可。首次会通过 `sync` 加载一次以回填类别名(生成 `<权重>.meta.json` sidecar)。

## 大模型分析

`.env` 里的 `LLM_PROVIDER`:

- `template`(默认):无需密钥,按检测结果确定性生成结构化分析(总述 / 分类解读 / 置信度说明 / 建议 / 免责声明),开箱即用。
- `dashscope`:通义千问,设 `DASHSCOPE_API_KEY` + `pip install ".[dashscope]"`。
- `openai`:OpenAI / 兼容端点,设 `OPENAI_API_KEY`(可选 `OPENAI_BASE_URL`)+ `pip install ".[openai]"`。

任一在线 provider 缺密钥 / 缺 SDK / 调用失败,都会**自动回退 template**,保证接口始终可用。
**只发送检测摘要(类别 / 计数 / 置信度),绝不上传 MRI 原图。**

## 报告导出

- **PDF**:`REPORT_PDF_ENGINE=auto` 时优先 WeasyPrint(CSS / 中文保真最好),缺系统库自动回退
  纯 Python 的 xhtml2pdf。生产建议装 WeasyPrint 系统库。
- **DOCX**:python-docx,含 KPI 表、逐例原图 / 标注图对照、检测明细表、分析小节与免责声明。

## 接口一览

统一响应信封 `{ code, message, data }`,`code === 0` 为成功;报告下载返回二进制(绕过信封)。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/models/` | 模型列表 |
| POST | `/api/models/sync/` | 同步模型目录 |
| POST | `/api/detect/single/` | 单图检测(multipart: `image`, `model`, `conf`, `iou`, `imgsz`) |
| POST | `/api/detect/batch/` | 批量检测(multipart: `images[]`, ...) |
| GET | `/api/history/` | 历史列表(`page`/`page_size`/`type`/`status`/`model`/`keyword`/`date_from`/`date_to`) |
| GET | `/api/history/{id}/` | 任务详情 |
| DELETE | `/api/history/{id}/` | 删除任务 |
| POST | `/api/jobs/{id}/analyze/` | 大模型分析 |
| GET | `/api/jobs/{id}/report/?format=pdf\|docx` | 下载报告 |
| GET | `/api/dashboard/stats/` | 概览统计 |
| GET | `/api/health/` | 健康检查 + 当前后端自述 |

## 与前端联立

前端已按同一契约写好。联立只需一步:把前端 `.env.development` 里的
`VITE_USE_MOCK` 改成 `false`,前端即改走 `/api` 真实接口(dev 由 Vite 代理转发到本服务,免跨域)。

## 目录结构

```
src/od_web_backend/
├── settings.py            # 引擎挂载 / 信封 / media / 业务开关
├── urls.py                # /api 路由 + DEBUG 下 media 服务
├── core/                  # 信封渲染器、异常处理、分页、健康检查
├── inference/             # 模型注册表(LRU)+ 推理器(yolo/demo)+ 检测编排 + 端点
├── history/              # 数据模型 + 序列化 + 历史/详情/删除/统计
└── analysis/             # 大模型客户端 + 报告渲染(HTML/PDF/DOCX)+ 端点
```

## 数据模型

- `DetectionJob`:一次任务(单图 / 批量),含参数、汇总、状态、时间。
- `DetectionImage`:任务下一张图及其原图 / 标注图路径、尺寸、耗时。
- `Detection`:单个检测框(每框一行),`{label, confidence, x1, y1, x2, y2}`。
- `AnalysisReport`:一次任务的大模型分析结果(sections)。

## 生产部署要点

- WSGI:`pip install ".[prod]"` 后用 `gunicorn od_web_backend.wsgi:application`(记得先把 `src/` 上 `PYTHONPATH`,或 `pip install -e .`)。
- 数据库:设 `POSTGRES_*` 环境变量即切 PostgreSQL。
- 媒体:`DEBUG=false` 时由 Nginx 直接服务 `media/`(`/media/` → `MEDIA_ROOT`),Django 不再兜底。
- 规模化:批量当前为同步基线;需要时把 `inference/services.py` 的 `run_batch` 整段搬进 Celery 任务即可(业务逻辑已与 request/response 解耦)。
```
