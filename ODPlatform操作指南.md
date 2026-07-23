# ODPlatform 操作指南

> 面向数据师与算法/训练同学的 CLI 操作手册。覆盖数据准备、转换划分、质检清理、训练推理的全流程。

---

## 一、环境准备

```bash
# 1. 创建 conda 环境(Python 3.11+)
conda create -n odplat python=3.12 -y
conda activate odplat

# 2. 安装引擎(editable 模式,改代码立即生效)
cd C:\Users\fym\Desktop\XJTU-ODPlatform
pip install -e apps/platform

# 3. 训练另装(按需)
pip install ultralytics torch    # 真实训练
# pip install -e ".[yolo]"       # 或用 optional deps
```

装完后有以下命令可用（`odp-` 前缀）：

| 命令 | 作用 | 谁用 |
|---|---|---|
| `odp-init` | 初始化项目目录 | 首次准备 |
| `odp-transform` | 数据转换 + 划分 + 报告 | 数据师 |
| `odp-validate` | 数据质检 + 清理 | 数据师 |
| `odp-gen-config` | 生成训练/推理配置 | 训练 |
| `odp-train` | 模型训练 | 训练 |
| `odp-infer` | 推理 | 训练 |
| `odp-reset` | 重置项目(可选备份) | 维护 |

---

## 二、目录结构

```
XJTU-ODPlatform/
├── .odp-workspace              ← 仓库根标记(勿删)
├── data/
│   ├── raw/<数据集名>/         ← 原始数据(数据师放这里)
│   │   ├── images/ 或 JPEGImages/
│   │   └── annotations/ 或 Annotations/
│   └── processed/<数据集名>/   ← transform 产出(训练读这里)
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── models/
│   ├── pretrained/             ← 预训练权重(.pt)
│   └── trained/                ← 训练产出的 best.pt
├── runs/
│   ├── data_pipeline/<run_id>/ ← transform 产物(manifest + split_report)
│   ├── data_validation/<run_id>/ ← validate 产物(report + quality_report + problem_images)
│   └── training/<run_id>/      ← 训练产物(best.pt + 曲线图)
└── apps/platform/configs/
    ├── datasets/<数据集>.yaml   ← transform 产出的数据集配置
    └── runtime/{train,val,infer}.yaml ← 训练/推理运行时配置
```

---

## 三、数据师工作流

### 3.1 准备原始数据

把数据集放到 `data/raw/<数据集名>/` 下，引擎会自动识别两种布局：

```
# 布局 A(标准)
data/raw/我的数据集/
├── images/        ← *.jpg/png
└── annotations/   ← *.xml(pascal_voc) / *.json(coco/labelme) / *.txt(yolo/dota)

# 布局 B(Pascal VOC 标准)
data/raw/我的数据集/
├── JPEGImages/    ← *.jpg
└── Annotations/   ← *.xml
```

### 3.2 初始化项目（首次）

```bash
odp-init
```

创建 data/models/runs/configs 等空目录。只在首次或目录缺失时跑。

### 3.3 数据转换 + 划分（核心）

```bash
odp-transform \
  --dataset VOC2028 \
  --format pascal_voc \
  --task detect \
  --split-strategy random \
  --train-rate 0.8 \
  --val-rate 0.1 \
  --seed 42
```

**参数说明：**

| 参数 | 说明 | 可选值 |
|---|---|---|
| `--dataset` | 数据集名(data/raw/ 下的文件夹名)或路径 | 自定义 |
| `--format` | 标注格式 | `pascal_voc` / `coco` / `yolo` / `labelme` / `dota` / `cvat` / `createml` |
| `--task` | 任务类型 | `detect` / `segment` |
| `--split-strategy` | 划分策略 | `random` / `stratified_multilabel` / `group` / `stratified_weighted` |
| `--train-rate` | 训练集比例 | 0~1，默认 0.8 |
| `--val-rate` | 验证集比例 | 0~1，默认 0.1（test = 1 - train - val） |
| `--seed` | 随机种子 | 默认 42 |
| `--classes` | 类别白名单(可选) | 如 `--classes hat person` |
| `--group-by-prefix N` | group 策略:按文件名前 N 字符分组 | 仅 `--split-strategy group` 时用 |
| `--groups-file PATH` | group 策略:CSV 文件(stem,group) | 优先于 prefix |

**划分策略怎么选：**

| 策略 | 适用场景 | 说明 |
|---|---|---|
| `random` | 通用 | 随机划分，最常用 |
| `stratified_weighted` | 类别不均衡 | 按主类分层，保证各类比例一致 |
| `stratified_multilabel` | 一图多标签 | 多标签迭代分层(Sechidis 算法) |
| `group` | 视频/同场景 | 同组不跨 split(需 `--group-by-prefix` 或 `--groups-file`) |

**产出：**
- `data/processed/<数据集>/images/{train,val,test}/` + `labels/{train,val,test}/`（图片自动 PIL 重新保存，修复 corrupt）
- `apps/platform/configs/datasets/<数据集>.yaml`（数据集配置，含指纹）
- `runs/data_pipeline/<run_id>/manifest.json`（逐样本血缘 + 契约指纹）
- `runs/data_pipeline/<run_id>/split_report.json` + `split_report.md`（划分报告）

### 3.4 数据质检

```bash
odp-validate --dataset VOC2028
```

跑 8 项检查，产出报告：

| check | 检测内容 |
|---|---|
| AnnotationIntegrityCheck | 标注框越界/零面积/倒置/解析错误 |
| DuplicateImageCheck | 完全重复(sha256) + 近似重复(dHash) |
| ImageIntegrityCheck | 图片损坏/截断/缺 EOI(与 ultralytics 对齐) |
| fingerprint_match | yaml/manifest/重算 三方指纹一致 |
| manifest_lineage | 逐样本内容+位置与冻结时一致 |
| PairExistenceCheck | 每张图有同名标签 |
| yaml_schema | yaml 结构合法 |
| smoke | 注册表机制自检 |

**产出：**
- `runs/data_validation/<run_id>/report.json`（机器读，全量结构化）
- `runs/data_validation/<run_id>/result.csv`（Excel 看，每 check 一行）
- `runs/data_validation/<run_id>/quality_report.md`（人读，含 check 结果 + 类别分布 + bbox 尺寸 + 集间漂移 JS + 问题图片清单）
- `runs/data_validation/<run_id>/problem_images/<split>/`（问题图片副本，供人工审查）
- `runs/data_validation/<run_id>/problem_images/problem_images_manifest.json`（问题图片清单）

**退出码：** 0=PASS/INFO，1=WARNING，2=ERROR。CI 可直接判断。

### 3.5 清理问题图片

```bash
# 交互式(推荐):列出问题图片,输入大写 YES 才删
odp-validate --dataset VOC2028 --purge

# 跳过确认(脚本/CI 用,慎用)
odp-validate --dataset VOC2028 --purge --yes
```

清理范围：
- **损坏图**（ImageIntegrityCheck 检出的 corrupt/截断/缺 EOI）→ 删图 + 标签
- **完全重复图**（DuplicateImageCheck 检出的 sha256 相同）→ 删重复的，保留原始

删除后 manifest 失效，**必须重跑 transform 重建**：
```bash
odp-transform --dataset VOC2028 --format pascal_voc --split-strategy random
```

### 3.6 划分报告解读（split_report.md）

```
# 数据划分报告 · VOC2028
- 契约指纹: b1e5a25b...（审计回溯用）
- 策略: random | 种子: 42 | 比例: 0.8/0.1/0.1

## 各 split 规模          ← train/val/test 各多少图+框
## 类别分布               ← 每类在三个 split 的计数
## bbox 尺寸分布 (COCO)   ← 小(<32²)/中(32-96²)/大(>96²) 占比
## 集间分布一致性 (JS)     ← train vs val vs test 的类别分布差异
## 图片尺寸统计            ← 平均宽高
```

**JS 散度怎么看：** <0.05 一致性良好，<0.15 略有差异，≥0.15 建议检查划分。

### 3.7 质量报告解读（quality_report.md）

```
# 数据质量分析报告
- 总评: ERROR/PASS（最坏一项决定）

## 校验结果              ← 8 项 check 的严重级 + 摘要
## ⚠️ 问题图片清单        ← 损坏/重复图(若有)
## 类别分布              ← 每类计数 + 占比
## bbox 尺寸分布         ← COCO 分桶
## 集间分布漂移 (JS)      ← 一致性评估
## 图片尺寸统计
## 划分契约指纹           ← 审计用
```

---

## 四、训练工作流

### 4.1 生成训练配置（首次）

```bash
odp-gen-config train          # 生成 train.yaml(带注释模板)
odp-gen-config val            # 生成 val.yaml
odp-gen-config infer          # 生成 infer.yaml
```

配置在 `apps/platform/configs/runtime/train.yaml`。已存在则跳过，`--overwrite` 覆盖（自动备份）。

> 如果 `odp-gen-config` 报 `cannot import name 'main'`，说明包没重装，跑 `pip install -e apps/platform`。

### 4.2 训练

```bash
odp-train \
  --config train \
  --model yolo12n.pt \
  --data VOC2028 \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

**参数说明：**

| 参数 | 说明 | 优先级 |
|---|---|---|
| `--config` | 运行时配置文件名(默认 train) | — |
| `--model` | 预训练权重(放 models/pretrained/) | CLI > YAML > DEFAULT |
| `--data` | 数据集名(对应 configs/datasets/<name>.yaml) | CLI |
| `--epochs` | 训练轮数 | CLI |
| `--batch` | 批次大小 | CLI |
| `--imgsz` | 输入尺寸 | CLI |
| `--device` | 设备(0/cpu) | CLI |
| `--no-archive` | 不归档 best.pt | — |

其余参数从 `train.yaml` 读取。CLI 参数覆盖 YAML，YAML 覆盖默认值。

**产出：**
- `runs/training/<run_id>/best.pt`（最佳权重）
- `runs/training/<run_id>/last.pt`（最后一轮）
- `runs/training/<run_id>/results.csv` + `results.png`（loss/mAP 曲线）
- `runs/training/<run_id>/confusion_matrix.png`（混淆矩阵）
- `runs/training/<run_id>/odp_audit.json`（训练审计快照）
- best.pt 自动归档到 `models/trained/`

### 4.3 推理

```bash
odp-infer \
  --model models/trained/<best>.pt \
  --source <图片或视频路径> \
  --conf 0.25 \
  --iou 0.45
```

支持图片/视频/摄像头。用 `BeautifyVisualizer` 出标注图（圆角框 + 中英标签）。

---

## 五、项目重置（维护）

```bash
# 干跑(默认,只看计划不删)
odp-reset

# 真删(需确认)
odp-reset --yes

# 重置前备份(推荐)
odp-reset --yes --backup
odp-reset --yes --backup --backup-dir D:/my_backups
```

重置清理：`data/processed`、`runs/`、`models/trained`、`logging/`、`configs/`。
**不删**：`data/raw`、`models/pretrained`、代码、文档。
备份产物在 `backups/reset_<时间戳>/`，附 `backup_manifest.json`。

---

## 六、完整工作流（数据师 → 训练）

```bash
# ===== 数据师 =====
# 1. 放数据到 data/raw/我的数据集/
# 2. 初始化
odp-init

# 3. 转换 + 划分(PIL 自动修复 corrupt)
odp-transform --dataset 我的数据集 --format pascal_voc --split-strategy random

# 4. 质检
odp-validate --dataset 我的数据集

# 5. 如有问题图,审查 problem_images/ 后清理
odp-validate --dataset 我的数据集 --purge
# 输入 YES → 删问题图 → 重跑 transform
odp-transform --dataset 我的数据集 --format pascal_voc --split-strategy random

# 6. 再质检确认 PASS
odp-validate --dataset 我的数据集

# ===== 训练同学 =====
# 7. 生成训练配置(首次)
odp-gen-config train

# 8. 训练
odp-train --config train --model yolo12n.pt --data 我的数据集 --epochs 100 --batch 16 --imgsz 640 --device 0

# 9. 推理
odp-infer --model models/trained/<best>.pt --source 测试图.jpg
```

---

## 七、常见问题

### Q: 训练报 `corrupt JPEG restored and saved`
**A:** 这是旧问题。现在 transform 会用 PIL 重新保存图片（自动修复 corrupt）。如果你还看到这个提示，说明 transform 用的是旧代码——重跑 `odp-transform` 即可（确保 `pip install -e apps/platform` 是 editable 模式）。

### Q: 质检报 `DuplicateImageCheck ERROR: N 张完全重复`
**A:** 数据集有字节级完全相同的图（数据泄漏）。跑 `odp-validate --dataset <name> --purge`，输入 YES 删重复图，然后重跑 transform。

### Q: 质检报 `ImageIntegrityCheck ERROR: N 张图片有问题`
**A:** 有损坏/截断/缺 EOI 的图。问题图片已收集到 `problem_images/`，审查后用 `--purge` 清理。transform 也会自动修复这类图（PIL 重新保存）。

### Q: `odp-gen-config` 报 `cannot import name 'main'`
**A:** 包没重装。跑 `pip install -e apps/platform` 更新入口。临时可用 `python -m od_platform.cli.gen_config train`。

### Q: group 划分策略报 warning "等价于 random"
**A:** group 策略需要指定 group 来源。加 `--group-by-prefix N`（按文件名前 N 字符分组）或 `--groups-file groups.csv`（CSV: stem,group）。否则每图自成一组，等价 random。

### Q: 划分后想换策略/比例怎么办
**A:** 直接重跑 `odp-transform`，会自动清理旧产物（materializer 的 `_safe_clear`）。指纹会变（策略/比例/seed 进指纹），重跑 `odp-validate` 校验新指纹。

### Q: 训练配置怎么调
**A:** 编辑 `apps/platform/configs/runtime/train.yaml`（带注释模板）。CLI 参数会覆盖 YAML。如 `--epochs 2` 覆盖 yaml 的 `epochs: 100`。用 `odp-gen-config train --overwrite` 重新生成模板（自动备份旧的）。

---

## 八、关键文件速查

| 文件 | 作用 |
|---|---|
| `apps/platform/configs/datasets/<name>.yaml` | 数据集配置(path/train/val/test/names/nc + odp_meta 指纹) |
| `apps/platform/configs/runtime/train.yaml` | 训练运行时配置(超参全量带注释) |
| `runs/data_pipeline/<run_id>/manifest.json` | 划分契约(逐样本血缘 + 契约指纹) |
| `runs/data_pipeline/<run_id>/split_report.md` | 划分报告(类别/尺寸/一致性) |
| `runs/data_validation/<run_id>/quality_report.md` | 质量报告(8 check + 分布 + 问题图片) |
| `runs/data_validation/<run_id>/problem_images/` | 问题图片副本(供审查) |
| `runs/training/<run_id>/odp_audit.json` | 训练审计(配置快照 + 指标) |
| `runs/training/<run_id>/best.pt` | 最佳权重(自动归档到 models/trained/) |
