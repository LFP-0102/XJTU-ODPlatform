"""materializer —— 照 SplitManifest 把三组样本落盘成 ultralytics 要的物理目录。"""
from __future__ import annotations
import logging, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from od_platform.common.paths import PROCESSED_DATA_DIR
from od_platform.data_pipeline.split.manifest import SplitManifest

logger = logging.getLogger(__name__)

def _assert_within(sandbox: Path, target: Path) -> Path:
    """target 必须落在 sandbox 之内,否则拒绝——rmtree 的围栏。"""
    sandbox_r, target_r = sandbox.resolve(), target.resolve()
    try:
        target_r.relative_to(sandbox_r)
    except ValueError:
        raise RuntimeError(f"拒绝删除:{target_r} 不在围栏 {sandbox_r} 之内")
    if target_r == sandbox_r:
        raise RuntimeError(f"拒绝删除围栏根本身:{sandbox_r}")
    return target_r

def _safe_clear(target: Path, sandbox: Path) -> None:
    """带围栏地清空一个目录:越界拒绝、符号链接拒绝、不存在则跳过。"""
    _assert_within(sandbox, target)
    if target.is_symlink():
        raise RuntimeError(f"拒绝删除符号链接:{target}")
    if target.exists():
        logger.info("清理旧产物:%s", target)
        shutil.rmtree(target)


@dataclass(frozen=True)
class SplitSourceDirs:
    """源:原图目录(来自 raw/,只读)+ 转换后的 YOLO 标签目录。"""
    images: Path
    labels: Path
    def resolve(self, stem: str) -> Tuple[Path, Path]:
        hits = sorted(self.images.glob(f"{stem}.*"))
        if len(hits) > 1:
            logger.warning("stem %s 找到多个图片文件 %s,将使用第一个", stem, hits)
        if not hits:
            raise FileNotFoundError(f"stem {stem} 找不到图片:{self.images}/{stem}.*")
        lbl = self.labels / f"{stem}.txt"
        if not lbl.exists():
            raise FileNotFoundError(f"stem {stem} 找不到标签:{lbl}")
        return hits[0], lbl


@dataclass(frozen=True)
class SplitOutputDirs:
    """本数据集的输出根(root 同时是删除围栏)。images-first,严格对齐 yaml。"""
    root: Path
    @property
    def leaves(self) -> Dict[str, Tuple[Path, Path]]:
        return {
            "train": (self.root/"images"/"train", self.root/"labels"/"train"),
            "val":   (self.root/"images"/"val",   self.root/"labels"/"val"),
            "test":  (self.root/"images"/"test",  self.root/"labels"/"test"),
        }
    @classmethod
    def for_dataset(cls, dataset: str) -> "SplitOutputDirs":
        if not dataset or "/" in dataset or dataset in (".", ".."):
            raise ValueError(f"非法数据集名:{dataset!r}")
        root = PROCESSED_DATA_DIR / dataset
        _assert_within(PROCESSED_DATA_DIR, root)
        return cls(root=root)
    @classmethod
    def under(cls, root: Path) -> "SplitOutputDirs":
        return cls(root=Path(root))

def _copy_pair(img: Path, lbl: Path, images_dir: Path, labels_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, images_dir / img.name)     # 复制,不硬链接:物理隔离
    shutil.copy2(lbl, labels_dir / lbl.name)


def materialize(manifest: SplitManifest,
                source: SplitSourceDirs,
                out: SplitOutputDirs) -> Dict[str, int]:
    """按 manifest.stems_of(split) 把三组样本复制到 out,返回各组计数。幂等。"""
    counts: Dict[str, int] = {}
    for split, (img_dir, lbl_dir) in out.leaves.items():
        _safe_clear(img_dir, sandbox=out.root)
        _safe_clear(lbl_dir, sandbox=out.root)
        stems = manifest.stems_of(split)              # 从逐样本血缘派生出"某组有哪些 stem"
        for stem in stems:
            img, lbl = source.resolve(stem)
            _copy_pair(img, lbl, img_dir, lbl_dir)
        counts[split] = len(stems)
    logger.info("materialize 完成:%s", counts)
    return counts


