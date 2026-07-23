#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :orchestrator.py
# @Time      :2026/7/16 15:25:31
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom/apps/platform/src/od_platform/data_pipeline/orchestrator.py
# @Function  :编排器，组织所有底层脚本的逻辑
"""raw -> 转换 ->配对 -> 划分 -> 冻结指纹 -> 落盘 -> 写yaml"""

from __future__ import annotations
import csv
import logging
import tempfile  # 临时输出
from pathlib import Path
from typing import Dict, List, Optional
from od_platform.common import paths
from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
    Task, IMAGE_EXTENSIONS
)
from od_platform.common.lineage import SplitManifest
from od_platform.common.refs import resolve_dataset
from od_platform.common.run_context import RunContext
from od_platform.data_pipeline.convert.registry import ConvertOptions
from od_platform.data_pipeline.convert.service import convert_data_to_yolo
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
        group_by_prefix: Optional[int] = None,
        groups_file: Optional[Path] = None,
        run: Optional[RunContext] = None,
            ):
        self.annotation_format = annotation_format
        self.task = task
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.random_state = random_state
        self.split_strategy = split_strategy
        self.group_by_prefix = group_by_prefix
        self.groups_file = groups_file
        self._run_ctx = run
        self._options = ConvertOptions(task=task, classes=classes)

        self.raw_root = resolve_dataset(dataset)
        self.dataset_name = self.raw_root.name
        self.raw_images, self.raw_annotations = self._resolve_raw_dirs()
        self.processed_root = paths.dataset_processed_dir(self.dataset_name)
        self.output_dir = SplitOutputDirs(self.processed_root)
        self.yaml_out = paths.dataset_yaml_path(self.dataset_name)

    def run(self) ->Dict:
        if self._run_ctx is not None:
            return self._execute(self._run_ctx)
        with RunContext("data_pipeline") as run:
            return self._execute(run)

    def _resolve_raw_dirs(self) -> tuple[Path, Path]:
        """智能识别 raw 目录布局。

        数据师的 raw 数据来源多样,常见布局都能吃:
          images/ + annotations/    (引擎默认)
          JPEGImages/ + Annotations/ (Pascal VOC 标准)
          imgs/ + labels/            (部分标注工具)
        任一组合命中即可;都找不到则抛 FileNotFoundError 给出明确提示。
        """
        img_names = ["images", "JPEGImages", "imgs", "data"]
        ann_names = ["annotations", "Annotations", "labels", "LabelXml", "xmls"]
        raw_images = next((self.raw_root / n for n in img_names
                           if (self.raw_root / n).is_dir()), None)
        raw_annots = next((self.raw_root / n for n in ann_names
                           if (self.raw_root / n).is_dir()), None)
        if raw_images is None:
            raise FileNotFoundError(
                f"raw 数据集 {self.raw_root} 下找不到图片目录(尝试过 {img_names})")
        if raw_annots is None:
            raise FileNotFoundError(
                f"raw 数据集 {self.raw_root} 下找不到标注目录(尝试过 {ann_names})")
        return raw_images, raw_annots

    def _build_groups(self, stems: List[str]) -> Optional[Dict[str, str]]:
        """构造 group_per_image 映射。groups_file 优先;其次 group_by_prefix;都没有返回 None。"""
        if self.groups_file is not None:
            mapping: Dict[str, str] = {}
            p = Path(self.groups_file)
            with p.open("r", encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    if len(row) >= 2 and row[0].strip():
                        mapping[row[0].strip()] = row[1].strip()
            return {s: mapping.get(s, s) for s in stems}
        if self.group_by_prefix:
            n = int(self.group_by_prefix)
            return {s: s[:n] for s in stems}
        return None

    def _execute(self, run: RunContext) -> Dict:
        logger.info("处理数据集 %r (format=%s, task=%s, split=%s)",
                self.dataset_name, self.annotation_format, self.task, self.split_strategy
                    )
        with tempfile.TemporaryDirectory(prefix="odp_pipe_") as tmp:
            staging = Path(tmp) / "labels"
            classes = convert_data_to_yolo(self.raw_annotations, staging, self.annotation_format, self._options)

            stems, label_per_image, label_bytes = self._scan(staging, classes)

            group_per_image = self._build_groups(stems)
            assignment = split_dataset(
                stems, self.train_rate, self.val_rate, self.random_state,
                strategy=self.split_strategy, labels_per_image=label_per_image,
                group_per_image=group_per_image,
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

            # 划分报告:类别分布 / bbox 尺寸 / 集间一致性 / 指纹(失败不阻断主流程)
            try:
                from od_platform.data_pipeline.split.report import (
                    build_split_report, write_split_report_json, write_split_report_md)
                sr = build_split_report(manifest, staging, self.raw_images, classes)
                write_split_report_json(sr, run.artifact_path("split_report.json"))
                write_split_report_md(sr, run.artifact_path("split_report.md"))
                logger.info("划分报告已生成: %s", run.artifact_path("split_report.json").name)
            except Exception as e:
                logger.warning("划分报告生成失败(不影响主流程): %s", e)
        return {"counts": counts, "yaml": str(self.yaml_out),
                "manifest_path": str(manifest_path),
                "run_dir": str(run.run_dir),
                "contract_fingerprint": manifest.contract_fingerprint
                }


    def _load_purge_list(self) -> set:
        """读取 processed_root/purge_list.json 黑名单(由 odp-validate --purge 生成)。
        返回待跳过的 stem 集合。文件不存在或读取失败返回空集(不阻断主流程)。"""
        purge_path = self.processed_root / "purge_list.json"
        if not purge_path.exists():
            return set()
        try:
            import json
            data = json.loads(purge_path.read_text(encoding="utf-8"))
            stems = set(data.get("stems", []))
            if stems:
                logger.info("读取黑名单 purge_list.json: %d 张图将被跳过(raw 不动)", len(stems))
            return stems
        except Exception as e:
            logger.warning("读取 purge_list.json 失败(忽略,继续全量处理): %s", e)
            return set()

    def _scan(self, labels_dir:Path, classes:List[str]):
        image_index: Dict[str, Path]  = {}
        for ext in IMAGE_EXTENSIONS:
            for img in self.raw_images.glob(f"*{ext}"):
                image_index.setdefault(img.stem, img)
        purged_stems = self._load_purge_list()
        stems: List[str] = []
        labels_per_image: Dict[str, List[str]] = {}
        label_bytes: Dict[str, bytes] = {}
        for lbl in sorted(labels_dir.glob("*.txt")):
            if lbl.stem not in image_index:
                continue
            if lbl.stem in purged_stems:
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
