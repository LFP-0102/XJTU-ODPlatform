# 数据集组织指南

本文档说明如何在 ODPlatform 项目中组织和管理数据集，确保数据转换脚本能够正确读取和处理你的数据。

---

## 目录结构总览

```
XJTU-ODPlatfrom/              # 项目根目录（包含 .odp-workspace 标记文件）
├── data/                     # 所有数据的根目录
│   ├── raw/                  # 原始数据集（手动放入）
│   │   └── <数据集名称>/      # 每个数据集一个独立文件夹
│   │       ├── images/       # 图像文件
│   │       └── annotations/  # 标注文件
│   └── processed/            # 处理后的数据（由脚本自动生成，不要手动修改）
├── models/                   # 模型存放
│   ├── pretrained/           # 预训练模型
│   └── trained/              # 训练完成的模型
├── runs/                     # 训练运行输出
├── apps/platform/            # 平台核心代码
├── scripts/                  # 工具脚本
└── docs/                     # 文档
```

> **关键路径**：`data/raw/` 是你唯一需要手动操作的数据目录，其他数据目录由平台脚本自动管理。

---

## 数据集命名规范

每个数据集在 `data/raw/` 下拥有独立文件夹，命名遵循以下规则：

### 基本规则

| 规则 | 说明 |
|---|---|
| **使用英文小写** | 文件夹名统一使用小写英文字母 |
| **单词间用下划线** | 多个单词以 `_` 分隔，不使用空格或连字符 |
| **简洁有含义** | 名称应能直观表达数据集的内容和用途 |
| **可含版本号** | 同一数据集的多个版本可使用 `_v1`、`_v2` 后缀区分 |

### 推荐命名模式

```
<任务类型>_<目标对象>[_<场景/来源>][_v<版本号>]
```

### 命名示例

| ✅ 推荐 | ❌ 不推荐 | 说明 |
|---|---|---|
| `ppe_detection_v1` | `PPE检测数据` | 避免中文、空格 |
| `traffic_signals_urban` | `traffic-signals-urban` | 统一用下划线而非连字符 |
| `helmet_dataset_campus` | `helmetDatasetCampus` | 统一小写，不用驼峰 |
| `reflective_vest_factory_v2` | `反光衣_factory_v2` | 避免中英混合 |
| `head_detection_v1` | `head_v1` | 应表明任务类型（检测/分类/分割） |

---

## 数据集内部结构

每个数据集文件夹必须包含以下两个子目录：

```
data/raw/<数据集名称>/
├── images/          # 所有图像文件
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
└── annotations/     # 所有标注文件
    ├── 000001.txt   # 与 images/ 中的文件名一一对应（扩展名可能不同）
    ├── 000002.txt
    └── ...
```

### images/ — 图像文件

- **支持格式**：`.jpg`、`.jpeg`、`.png`、`.bmp`
- **命名建议**：使用固定位数的数字编号（如 `000001.jpg`），便于排序和匹配
- **图像质量**：建议原始分辨率，不要提前压缩或缩放

### annotations/ — 标注文件

- **文件名**：必须与 `images/` 中对应的图像**主文件名一致**，仅扩展名不同
  - 例：`images/000123.jpg` ↔ `annotations/000123.txt`
- **支持的标注格式**（当前阶段）：
  - YOLO 格式（`.txt`）：每行 `class_id cx cy w h`（归一化坐标）
  - COCO 格式（`.json`）：单个 `annotations.json` 文件
  - VOC XML 格式（`.xml`）：每张图片一个 XML 文件

> **注意**：同一个数据集内只能使用**一种**标注格式，不要混用。

---

## 完整示例

假设你有一个安全帽检测数据集（来自校园场景，第一个版本），应按如下方式组织：

```
data/raw/
└── helmet_detection_campus_v1/
    ├── images/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── 000003.jpg
    │   └── ...              # 共 5000 张图像
    └── annotations/
        ├── 000001.txt       # YOLO 格式：0 0.512 0.438 0.124 0.156
        ├── 000002.txt
        ├── 000003.txt
        └── ...              # 与 images/ 一一对应
```

再放置几个数据集后，`data/raw/` 看起来是这样的：

```
data/raw/
├── helmet_detection_campus_v1/
│   ├── images/
│   └── annotations/
├── reflective_vest_factory_v2/
│   ├── images/
│   └── annotations/
├── traffic_signals_urban/
│   ├── images/
│   └── annotations/
└── ppe_multiclass_v1/
    ├── images/
    └── annotations/
```

---

## 验证数据集是否正确

运行项目初始化命令，平台会自动检查 `data/raw/` 的状态：

```bash
python -m od_platform.cli.init_project
```

输出示例：

```
========================== 原始数据集目录状态 ==========================
 - data/raw 就绪（包含 4 个数据集）
 - 数据集 helmet_detection_campus_v1 就绪
 - 数据集 ppe_multiclass_v1 就绪
 - 数据集 reflective_vest_factory_v2 就绪
 - 数据集 traffic_signals_urban 就绪
```

如果看到警告信息，说明数据集结构有问题，请根据日志提示调整。

---

## 常见问题

### Q: 我可以直接在 `data/raw/` 下放图片，不建子文件夹吗？

**不可以。** 平台要求每个数据集必须有独立的子文件夹，且内部必须有 `images/` 和 `annotations/` 两个子目录。

### Q: 标注文件可以为空吗？

可以但不推荐。如果某个数据集暂时没有标注，请在 `annotations/` 目录下放置一个 `README.txt` 说明情况，避免平台报告目录为空。

### Q: `data/processed/` 目录是什么？

该目录由数据预处理/转换脚本自动生成，存放的是从 `raw/` 转换而来的标准化数据。**不要手动修改此目录的内容。**

### Q: 数据集名称用中文可以吗？

技术上可能可行，但**强烈不推荐**。英文小写 + 下划线的命名方式能避免跨平台路径问题，也与代码中的变量命名一致。
