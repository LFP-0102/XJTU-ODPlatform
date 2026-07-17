#数据快照:对数据集做一次完整的扫描，供所有check消费
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
from od_platform.common.constants import IMAGE_EXTENSIONS, Task

@dataclass(frozen=True)
class DatasetSnapshot:
    yaml_path: Path
    yaml_data: Dict[str, Any]
    yaml_load_error: Optional[str]
    data_root: Path
    nc : Optional[int]
    class_names: Tuple[str, ...]
    images_per_split: Dict[str, Tuple[Path, ...]]
    labels_per_split: Dict[str, Tuple[Path, ...]]
    labels_files_per_split: Dict[str, Tuple[Path, ...]]
    scan_warning: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def splits(self) -> Tuple[str, ...]:
        return tuple(s for s in ("train", "val", "test") if s in self.images_per_split)

    @property
    def total_images(self) -> int:
        return sum(len(v) for v in self.images_per_split.values())

# 内部辅助
def _load_yaml(path: Path) -> Tuple[Optional[Dict[str, Any]],Optional[str]]:
    if not path.exists():
        return {}, f"YAML文件不存在: {path}"
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as e:
        return {}, f"YAML文件 {path} 解析失败: {e}"
    if not isinstance(data, dict):
        return {}, f"YAML 顶层不是dict: {type(data).__name__}"
    return data, None

def _resolve_data_root(yaml_path: Path, yaml_data: Dict[str, Any]) -> Path:
    p = yaml_data.get("path")
    if not p:
        return yaml_path.parent.resolve()
    p = Path(p)
    return p.resolve() if p.is_absolute() else (yaml_path.parent / p).resolve()

def _split_dir(data_root: Path, field_val: Any) -> Optional[Path]:
    """yaml 的 train/val/test 字段(只认字符串,ultralytics 主流写法)→ 绝对目录。"""
    if not isinstance(field_val, str) or not field_val.strip():
        return None
    p = Path(field_val)
    return p.resolve() if p.is_absolute() else (data_root / p).resolve()

def _list_images(split_dir: Path) -> List[Path]:
    """split 目录下所有支持扩展名的图(绝对、去重、排序)。"""
    if not split_dir.is_dir():
        return []
    out: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        out.extend(split_dir.glob(f"*{ext}"))
        out.extend(split_dir.glob(f"*{ext.upper()}"))   # Windows 大小写
    return sorted(set(out))

def _label_path_for_image(image_path: Path) -> Path:
    """YOLO 布局:images/<split>/foo.jpg → labels/<split>/foo.txt。

    倒着找最后一个 'images' 替换为 'labels'(路径里可能多次出现 images,取最后一个最稳)。
    路径规则只此一处实现,所有消费者引用——SSoT。
    """
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")

def _list_label_files(split_dir_labels: Path) -> Tuple[Path, ...]:
    """labels/<split>/ 下真实存在的 .txt(绝对、排序)。"""
    if not split_dir_labels.is_dir():
        return tuple()
    return tuple(sorted(split_dir_labels.glob("*.txt")))


def _normalize_names(names: Any) -> Tuple[str, ...]:
    """names 支持 list 或 {id: name};统一成按 id 升序的 tuple。非法→空。"""
    if isinstance(names, list):
        return tuple(str(x) for x in names)
    if isinstance(names, dict):
        try:
            return tuple(str(names[k]) for k in sorted(names, key=lambda x: int(x)))
        except (ValueError, TypeError):
            return tuple(str(v) for v in names.values())
    return tuple()

def build_snapshot(yaml_path: Path, task_type: Optional[str] = None) -> DatasetSnapshot:
    """扫一次,产出后续 check 全部所需素材。永不抛异常(问题都进 warnings/error)。"""
    yaml_path = yaml_path.resolve()
    warnings: List[str] = []

    yaml_data, yaml_err = _load_yaml(yaml_path)
    if yaml_err:
        warnings.append(yaml_err)

    data_root = _resolve_data_root(yaml_path, yaml_data)
    nc = yaml_data.get("nc") if isinstance(yaml_data.get("nc"), int) else None
    class_names = _normalize_names(yaml_data.get("names"))

    # task_type 目前无 check 消费,但保留解析以备扩展;非法值记 warning 不抛。
    resolved_task = task_type or yaml_data.get("task") or Task.DETECT
    if resolved_task not in (Task.DETECT, Task.SEGMENT):
        warnings.append(f"未知 任务类型 {resolved_task},默认使用 {Task.DETECT}")
        resolved_task = Task.DETECT

    images_per_split: Dict[str, Tuple[Path, ...]] = {}
    labels_per_split: Dict[str, Tuple[Path, ...]] = {}
    labels_files_per_split: Dict[str, Tuple[Path, ...]] = {}

    for split in ("train", "val", "test"):
        sdir = _split_dir(data_root, yaml_data.get(split))
        if sdir is None or not sdir.exists():
            if split in yaml_data:
                warnings.append(f"split '{split}' 目录不可用: {sdir}")
            continue
        images = _list_images(sdir)
        if not images:
            warnings.append(f"split '{split}' 目录下无图像: {sdir}")
            continue
        labels = [_label_path_for_image(img) for img in images]
        images_per_split[split] = tuple(images)
        labels_per_split[split] = tuple(labels)
        # 由第一张图的期望标签路径,反推 labels/<split>/ 目录,列真实存在的 .txt
        labels_files_per_split[split] = _list_label_files(labels[0].parent)

    return DatasetSnapshot(
        yaml_path=yaml_path, yaml_data=yaml_data, yaml_load_error=yaml_err,
        data_root=data_root, nc=nc, class_names=class_names,
        images_per_split=images_per_split, labels_per_split=labels_per_split,
        labels_files_per_split=labels_files_per_split, scan_warning=tuple(warnings),
    )














