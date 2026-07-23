# XJTU-ODPlatform 项目长期记忆

## 项目概览
YOLO 目标检测平台, 模块化 Python 包 `od_platform`(可编辑安装在 D:/miniconda3).
- 根目录标记: `.odp-workspace`
- 核心引擎: `apps/platform/`
- 数据: `data/`, 模型: `models/`, 运行产物: `runs/`

## 架构约定(必须遵守)
- **paths.py 是路径 SSoT**: 路径常量 + `*_run_dir()` 函数集中在此, 不在配置类里写路径
- **RunContext**: 管理 runs/<subsystem>/<run_id>/ 目录, 自动去重(同名 _2)
- **数据/渲染分离**: result.py / report.py 是纯数据(frozen dataclass), render_* 只消费数据不持有
- **复用 TrainMetrics**: `model_train/result.py` 的 `TrainMetrics.from_yolo_results` 可被 val/eval 共用(作者注释明确)
- **registry 模式**: data_validation 用 `@check` 装饰器 + lazy_init 注册, `import_submodules` 自动发现
- **双轨输出**: 机器轨 report.json + 人工轨 result.csv(utf-8-sig BOM, Excel 认中文) + 可读 report.md
- **三源配置合并**: YAML + CLI + extra → ConfigMerger → pydantic 验证, `build_*_config()` 一键构建

## CLI 入口(pyproject.toml [project.scripts])
odp-init / odp-reset / odp-transform / odp-validate / odp-gen-config / odp-train / odp-infer / **odp-eval**(新增)

## 运行环境
- Python: D:/miniconda3/python.exe (有 ultralytics 8.4.96 / pydantic 2.13.4 / colorlog)
- managed python (3.13.12) 无 ultralytics, 测试用 miniconda 的
