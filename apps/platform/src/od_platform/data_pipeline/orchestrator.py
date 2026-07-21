#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :orchestrator.py
# @Time      :2026/7/16 15:25:31
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom/apps/platform/src/od_platform/data_pipeline/orchestrator.py
# @Function  :编排器，组织所有底层脚本的逻辑
"""raw -> 转换 ->配对 -> 划分 -> 冻结指纹 -> 落盘 -> 写yaml"""

from __future__ import annotations
import logging
import tempfile  # 临时输出
from pathlib import Path
from typing import Dict, List, Optional
from od_platform.common import paths
from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
    Task, IMAGE_EXTENSIONS
)
from od_platform.common.refs import resolve_dataset
from od_platform.common.run_context import RunContext
from od_platform.data_pipeline.convert.registry import ConvertOptions
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
from od_platform.data_pipeline.split import manifest
from od_platform.data_pipeline.split.manifest import SplitManifest
from od_platform.data_pipeline.split.materializer import SplitOutputDirs,SplitSourceDirs,materialize

from od_platform.data_pipeline.split.service import split_dataset
from od_platform.data_pipeline.split.yaml_writer import write_dateset_yaml


logger = logging.getLogger(__name__)

class DatasetPipeline:
    def __init__(self,
        dataset: str, annotation_format: str, * , task: str = Task.DETECT,
        train_rate: float = DEFAULT_TRAIN_RATE, val_rate: float = DEFAULT_VAL_RATE,
        classes: Optional[List[str]] = None, random_state: int = DEFAULT_RANDOM_STATE,
        split_strategy: str = DEFAULT_SPLIT_STRATEGY,
        run: Optional[RunContext] = None,
            ):
        self.annotation_format = annotation_format
        self.task = task
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.random_state = random_state
        self.split_strategy = split_strategy
        self._run_ctx = run
        self._options = ConvertOptions(task=task, classes=classes)

        self.raw_root = resolve_dataset(dataset)
        self.dataset_name = self.raw_root.name
        self.raw_images = self.raw_root / "images"
        self.raw_annotations = self.raw_root / "annotations"
        self.processed_root = paths.dataset_processed_dir(self.dataset_name)
        self.output_dir = SplitOutputDirs(self.processed_root)
        self.yaml_out = paths.dataset_yaml_path(self.dataset_name)

    def run(self) ->Dict:
        if self._run_ctx is not None:
            return self._execute(self._run_ctx)
        with RunContext("data_pipeline") as run:
            return self._execute(run)

    def _execute(self, run: RunContext) -> Dict:
        logger.info("处理数据集 %r (format=%s, task=%s, split=%s)",
                self.dataset_name, self.annotation_format, self.task, self.split_strategy
                    )
        with tempfile.TemporaryDirectory(prefix="odp_pipe_") as tmp:
            staging = Path(tmp) / "labels"
            classes = convert_data_to_yolo(self.raw_annotations, staging, self.annotation_format, self._options)

            stems, label_per_image, label_bytes = self._scan(staging, classes)

            assignment = split_dataset(
                stems, self.train_rate, self.val_rate, self.random_state,
                strategy=self.split_strategy, labels_per_image=label_per_image,
            )
            test_rate = round(1 - self.train_rate - self.val_rate, 6)
            manifest = SplitManifest.build(
                self.dataset_name, self.split_strategy, self.random_state,
                (self.train_rate, self.val_rate, test_rate), classes,assignment,label_bytes,
                created_at=run.created_at
            )
            manifest_path = run.artifact_path("manifest.json")
            manifest.write(manifest_path)

            manifest_ref = manifest_path.relative_to(paths.ROOT_DIR).as_posix()
            source = SplitSourceDirs(images=self.raw_images, labels=staging)
            counts = materialize(manifest, source, self.output_dir)
            write_dateset_yaml(
                self.yaml_out, dataset_root=self.processed_root, classes=classes,
                manifest=manifest,dataset_name=self.dataset_name,
                source_format=self.annotation_format,task=self.task,
                manifest_ref=manifest_ref
            )
        return {"counts": counts, "yaml": str(self.yaml_out),
                "manifest_path": str(manifest_path),
                "run_dir": str(run.run_dir),
                "contract_figure": manifest.contract_fingerprint
                }


    def _scan(self, labels_dir:Path, classes:List[str]):
        image_index: Dict[str, Path]  = {}
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
            for line in lbl.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line:
                    cid = int(line.split()[0])
                    if 0 <= cid <= len(classes):
                        names.append(classes[cid])
            labels_per_image[lbl.stem] = names
        return stems, labels_per_image, label_bytes

