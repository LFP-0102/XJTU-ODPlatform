# XJTU-ODPlatform 项目长期备忘

> 脑部 MRI 肿瘤检测系统(对外主题)/ 通用目标检测平台(引擎本体)。XJTU 学生实训项目。

## 顶层定位
- 引擎是**通用 YOLO 目标检测平台**,Web 应用层主题为「脑部 MRI 肿瘤检测」(类别 glioma/meningioma/pituitary)。
- 当前 data/raw 实际放的是 VOC2028 安全帽数据集(hat/person,6065+758+758),用于引擎流程验证;models/trained 目录尚未建立,Web 端默认走 `INFER_BACKEND=demo` 合成检测。

## 三层 Monorepo 架构
仓库根有 `.odp-workspace` 标记文件,引擎与后端都靠它向上查找 ROOT_DIR。
- `apps/platform` — Python 引擎 + CLI(`od_platform` 包,hatchling 打包)。提供 7 个 CLI 入口:odp-init / odp-reset / odp-transform / odp-validate / odp-gen-config / odp-train / odp-infer。
- `apps/web-backend` — Django 5 + DRF(`od_web_backend` 包)。settings.py 自动把 `apps/platform/src` 挂到 sys.path,**无需单独安装引擎**即可 import。
- `apps/web-frontend` — Vue 3 + Vite + TS + Element Plus + Pinia。`.env.development` 的 `VITE_USE_MOCK` 切换 mock 演示 / 真实接口。

## 共享资产(仓库根级,三端共用)
- `data/` raw(原始)→ processed(转换后,按 train/val/test 划分)
- `models/` pretrained(预训练权重)→ trained(训练产出 best.pt)
- `runs/` data_pipeline(数据管线产物)/ training(训练产物)
- `.odp-meta/` 运行元数据;`apps/platform/configs/` datasets(runtime yaml,端私有,gitignore)

## 引擎核心模块(apps/platform/src/od_platform/)
- `common/paths.py` — 全部路径常量与 `_find_workspace_root()`;`PROTECTED_DIRS` 防误删;`get_dirs_to_initialize/reset()`。
- `data_pipeline/` — orchestrator(DatasetPipeline 编排)+ convert(COCO/Pascal VOC/YOLO 三转换器,registry 模式)+ split(random/stratified_multilabel,产出 manifest.json + dataset.yaml,带 contract_fingerprint 指纹)。
- `data_validation/` — 5 项 checks(pair_existence/fingerprint_match/manifest_lineage/yaml_schema/smoke),registry 模式。
- `model_train/service.py` — train_yolo() 编排:解析模型 → YOLO.train → TrainMetrics → 写 odp_audit.json 审计 → archive_best_weight 归档。
- `model_infer/` — pipeline + hooks + sinks + cancel(支持 image/video/camera 帧源)。
- `visualization/` — BeautifyVisualizer(圆角框、中英标签映射、文本缓存),**可整包拷贝复用**,字体放 assets/。
- `runtime_config/` — train/val/infer 三套配置,合并优先级 **CLI > YAML > DEFAULT**,generator 可生成带注释模板。
- `frame_source/` — image/video/camera 三种帧源,registry 模式。
- `common/RunContext` — 贯穿训练/管线/校验,统一 run_id + 产物归档目录。

## Web 后端契约
- 统一信封 `{code, message, data}`,code===0 成功;core/EnvelopeJSONRenderer + envelope_exception_handler。
- 路由:`/api/models/` `/api/detect/single|batch/` `/api/history/` `/api/jobs/{id}/analyze/` `/api/jobs/{id}/report/?format=pdf|docx` `/api/dashboard/stats/` `/api/health/`。
- 数据模型:DetectionJob(任务)→ DetectionImage(图,原图+标注图)→ Detection(框,label/confidence/x1y1x2y2);AnalysisReport(LLM 分析)。
- 推理后端 `INFER_BACKEND`:yolo(真实 ultralytics)/ demo(免 torch 合成框,仍用真实 BeautifyVisualizer 出图)。
- LLM_PROVIDER:template(默认免密钥)/ dashscope / openai,缺密钥自动回退 template,**只发检测摘要不上传原图**。
- 报告 PDF(WeasyPrint 优先,回退 xhtml2pdf)/ DOCX(python-docx)。

## 前后端契约(前端 types/index.ts)
- Detection: `{label, confidence, bbox:[x1,y1,x2,y2]}` 像素坐标,与引擎 model_infer 对齐。
- 批量检测当前同步基线;services.py 已与 request/response 解耦,迁 Celery 时整段搬即可。

## 关键配置文件
- `apps/platform/configs/datasets/VOC2028.yaml` — 数据集配置(path/train/val/test/names/nc + odp_meta 含 split 与 contract_fingerprint)。
- `apps/platform/configs/runtime/train.yaml` — 训练运行时配置(超长带注释模板,odp-gen-config 生成)。
- `apps/web-backend/.env` — INFER_BACKEND / LLM_PROVIDER / VIZ_* / REPORT_PDF_ENGINE 等业务开关。

## 运行入口
- 引擎 CLI:`odp-init / odp-transform / odp-validate / odp-gen-config / odp-train / odp-infer`(装包后)或 `python scripts/init_project.py`。
- 后端:`cd apps/web-backend && python manage.py migrate && python manage.py runserver 127.0.0.1:8000`。
- 前端:`cd apps/web-frontend && npm install && npm run dev`(默认 5173,Vite 代理 /api → 8000)。
