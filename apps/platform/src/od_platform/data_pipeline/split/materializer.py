#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :materializer.py
# @Time      :2026/7/16 11:06:27
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :按照SplitManifest 把三组样本落到ultralytics 需要的目录里面
from __future__ import annotations
import logging, shutil
from dataclasses import  dataclass
from pathlib import Path
from  typing import Dict, Tuple

from od_platform.common.paths import  PROCESS_DATA_DIR
from od_platform.data_pipeline.split.manifest import SplitManifest

logger = logging.getLogger(__name__)

def _assert_within(sandbox: Path, target: Path) -> Path:
    sandbox_r, target_r = sandbox.resolve(), target.resolve()
    try:
        target_r.relative_to(sandbox_r)
    except ValueError:
        raise RuntimeError(f"拒绝删除：{target_r}, 不在 {sandbox_r} 之内")
    if target_r == sandbox_r:
        raise RuntimeError(f"拒绝删除围栏本身：{sandbox_r}")
    return target_r

def _safe_clear(target:Path, sandbox:Path) -> None:
    _assert_within(sandbox, target)
    if target.is_symlink():
        raise RuntimeError(f"拒绝删除符号链接：{target}")
    if target.exists():
        logger.info("清理旧产物: %s", target)
        shutil.rmtree(target)  # 真正地删除操作

@dataclass(frozen=True)
class SplitOutputDirs:
    root: Path

    @property
    def leaves(self) -> Dict[str, Tuple[Path, Path]]:
        return {
            "train": (self.root / "images" /"train", self.root / "labels" / "train"),
            "val": (self.root / "images" / "val", self.root / "labels" / "val"),
            "test": (self.root / "images" / "test", self.root / "labels" / "test"),
        }

    @classmethod
    def for_dataset(cls, dataset: str) -> "SplitOutputDirs":
        if not dataset or "/" in dataset or dataset in (".", ".."):
            raise ValueError("非法的数据集名称")
        root = PROCESS_DATA_DIR / dataset
        _assert_within(PROCESS_DATA_DIR, root)
        return cls(root = root)

    @classmethod
    def under(cls, root: Path) -> "SplitOutputDirs":
        return cls(root = Path(root))

def _copy_pair(img: Path, lbl: Path, images_dir: Path, labels_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, images_dir / img.name)
    shutil.copy2(lbl, labels_dir / lbl.name)

@dataclass(frozen=True)
class SplitSourceDirs:
    images: Path
    labels: Path

    def resolve(self, stem: str) -> Tuple[Path, Path]:
        hits = sorted(self.images.glob(f"{stem}.*"))
        if len(hits) > 1:
            logger.warning("stem %s 找到多个图像文件 %s, 将使用第一个", stem, hits)
        if not hits:
            raise FileNotFoundError(f"stem {stem} 找不到图片：{self.images}/{stem}.*")
        lbl = self.labels / f"{stem}.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"stem {stem} 找不到标签：{lbl}")
        return hits[0], lbl


def materialize(manifest: SplitManifest,
                source: SplitSourceDirs,
                out: SplitOutputDirs) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for split, (img_dir, lbl_dir) in out.leaves.items():
        _safe_clear(img_dir, sandbox=out.root)
        _safe_clear(lbl_dir, sandbox=out.root)
        stems = manifest.stems_of(split)
        for stem in stems:
            img, lbl =source.resolve(stem)
            _copy_pair(img, lbl, img_dir, lbl_dir)
            counts[split] = len(stems)
    logger.info("materialize 完成 %s", counts)
    return counts



