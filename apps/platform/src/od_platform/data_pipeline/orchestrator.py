"""端到端编排(指挥家):raw -> 转换(tempfile) -> 配对 -> 平衡报告 -> 划分 -> 冻结 manifest -> 落盘 -> 写 yaml。"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from od_platform.common import paths
from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
    Task,IMAGE_EXTENSIONS
)
from od_platform.common.refs import resolve_dataset
from od_platform.common.run_context import RunContext
from od_platform.data_pipeline.convert.registry import ConvertOptions
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
#from od_platform.data_pipeline.report import analyze_balance, format_report, warn_if_rare
from od_platform.data_pipeline.split.manifest import SplitManifest
from od_platform.data_pipeline.split.materializer import SplitOutputDirs, SplitSourceDirs, materialize
from od_platform.data_pipeline.split.service import split_dataset
from od_platform.data_pipeline.split.yaml_writer import write_dataset_yaml

logger = logging.getLogger(__name__)

class DatasetPipeline:
    """一次"把某数据集转换 + 划分成可训练数据集"的完整流程。"""

    def __init__(
        self, dataset: str, annotation_format: str, *,
        task: str = Task.DETECT,
        train_rate: float = DEFAULT_TRAIN_RATE, val_rate: float = DEFAULT_VAL_RATE,
        classes: Optional[List[str]] = None, random_state: int = DEFAULT_RANDOM_STATE,
        split_strategy: str = DEFAULT_SPLIT_STRATEGY,
        run: Optional[RunContext] = None,   # 可注入的现场;None → run() 里自开一个
    ):
        self.annotation_format = annotation_format
        self.task = task
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.random_state = random_state
        self.split_strategy = split_strategy
        self._run_ctx = run
        self._options = ConvertOptions(task=task, classes=classes)

        # 名字或路径 → 具体目录;并据数据集名,提前算好"按数据集分桶"的落盘根与 yaml 路径。
        self.raw_root = resolve_dataset(dataset)
        self.dataset_name = self.raw_root.name
        self.raw_images = self.raw_root / "images"
        self.raw_annotations = self.raw_root / "annotations"
        self.processed_root = paths.dataset_processed_dir(self.dataset_name)
        self.output_dirs = SplitOutputDirs.under(self.processed_root)
        self.yaml_out = paths.dataset_yaml_path(self.dataset_name)

    def run(self) -> Dict:
        # 现场归谁管:CLI 已开好并注入 → 直接用;独立跑(checkpoint/测试)→ 自开一个,退出即收。
        if self._run_ctx is not None:
            return self._execute(self._run_ctx)
        with RunContext("data_pipeline") as run:
            return self._execute(run)

    def _execute(self, run: RunContext) -> Dict:
        logger.info("处理数据集 %r (format=%s, task=%s, split=%s)",
                    self.dataset_name, self.annotation_format, self.task, self.split_strategy)
        self._check_raw()                              # 预检不合格 → 直接抛,fail-fast

        # ★ 转换的中间产物写进系统临时目录,出了 with 自动清理 —— 全程不碰 data/raw/(只读圣地)。
        with tempfile.TemporaryDirectory(prefix="odp_pipe_") as tmp:
            staging = Path(tmp) / "labels"
            classes = convert_data_to_yolo(self.raw_annotations, staging, self.annotation_format, self._options)
            stems, labels_per_image, label_bytes = self._scan(staging, classes)


            # 划分前先打一份只读平衡报告(检查 ≠ 过滤:只提醒,不影响后续流程)。
            #min_split_rate = min(self.val_rate, 1 - self.train_rate - self.val_rate)
            #report = analyze_balance(labels_per_image, classes, min_split_rate)
            #for line in format_report(report).splitlines():
            #    logger.info(line)
            #warn_if_rare(report)

            # 指挥家对所有策略统一传 labels_per_image;随机策略会忽略它,分层策略会读它。
            assignment = split_dataset(
                stems, self.train_rate, self.val_rate, self.random_state,
                strategy=self.split_strategy, labels_per_image=labels_per_image,
            )
            test_rate = round(1 - self.train_rate - self.val_rate, 6)
            manifest = SplitManifest.build(
                self.dataset_name, self.split_strategy, self.random_state,
                (self.train_rate, self.val_rate, test_rate), classes, assignment, label_bytes,
                created_at=run.created_at,             # ★ 时间戳与 run_id / 现场目录名同源(而非各自 now())
            )
            manifest_path = run.artifact_path("manifest.json")
            manifest.write(manifest_path)              # ★ 审计留痕 → runs/data_pipeline/<ts>/manifest.json
            manifest_ref = manifest_path.relative_to(paths.ROOT_DIR).as_posix()  # 相对仓库根,写进 yaml 给 D4
            # ★ 落盘必须在 with 之内:此刻临时标签还活着,SplitSourceDirs 才取得到。
            source = SplitSourceDirs(images=self.raw_images, labels=staging)
            counts = materialize(manifest, source, self.output_dirs)
            write_dataset_yaml(
                self.yaml_out, dataset_root=self.processed_root, classes=classes,
                manifest=manifest, dataset_name=self.dataset_name,
                source_format=self.annotation_format, task=self.task,
                manifest_ref=manifest_ref,             # ★ yaml 的 odp_meta 记下 manifest 路径(给 D4)
            )
        return {"counts": counts, "yaml": str(self.yaml_out),
                "manifest_path": str(manifest_path), "run_dir": str(run.run_dir),
                "contract_fingerprint": manifest.contract_fingerprint}

    # ---- 预检(本阶段:仅结构;覆盖率闸门见阶段 8)-------------------------
    def _check_raw(self) -> None:
        if not self.raw_root.is_dir():
            raise FileNotFoundError(f"数据集目录不存在: {self.raw_root}")
        if not self.raw_images.is_dir():
            raise FileNotFoundError(f"缺少 images 子目录: {self.raw_images}")
        if not self.raw_annotations.is_dir():
            raise FileNotFoundError(f"缺少 annotations 子目录: {self.raw_annotations}")

    # ---- 配对 / 标签提取(合成 stems + labels_per_image + label_bytes)------
    def _scan(self, labels_dir: Path, classes: List[str]):
        image_index: Dict[str, Path] = {}
        for ext in IMAGE_EXTENSIONS:
            for img in self.raw_images.glob(f"*{ext}"):
                image_index.setdefault(img.stem, img)
        stems: List[str] = []
        labels_per_image: Dict[str, List[str]] = {}
        label_bytes: Dict[str, bytes] = {}
        for lbl in sorted(labels_dir.glob("*.txt")):
            if lbl.stem not in image_index:
                continue
            stems.append(lbl.stem)
            label_bytes[lbl.stem] = lbl.read_bytes()
            names: List[str] = []
            for line in lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    cid = int(line.split()[0])
                    if 0 <= cid < len(classes):
                        names.append(classes[cid])
            labels_per_image[lbl.stem] = names
        return stems, labels_per_image, label_bytes

