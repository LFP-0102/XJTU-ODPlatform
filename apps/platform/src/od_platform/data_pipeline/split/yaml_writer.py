# apps/platform/src/od_platform/data_pipeline/split/yaml_writer.py
"""yaml_writer —— 写 ultralytics 的 dataset.yaml,并附一个 odp_meta 审计块。"""
from __future__ import annotations
import logging, yaml
from pathlib import Path
from typing import List
from od_platform.data_pipeline.split.manifest import SplitManifest

logger = logging.getLogger(__name__)
_SCHEMA_VERSION = "v1"

def write_dataset_yaml(
        yaml_path: Path, *, dataset_root: Path, classes: List[str],
        manifest: SplitManifest, dataset_name: str,
        source_format: str, task: str, manifest_ref: str) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    tr, va, te = manifest.rations
    counts = {s: len(manifest.stems_of(s)) for s in ("train", "val", "test")}

    doc = {
        "path": dataset_root.as_posix(),  # ★ 强制 POSIX 路径,杜绝 Windows 反斜杠玄学
        "train": "images/train",  # images-first,严格对齐 materializer 落盘布局
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
        "odp_meta": {  # ★ 审计块:盖上划分契约指纹
            "schema_version": _SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "source_format": source_format,
            "task": task,
            "tool_version": manifest.tool_version,
            "created_at": manifest.created_at,
            "contract_fingerprint": manifest.contract_fingerprint,  # 划分版本印章
            "manifest_path": manifest_ref,  # ★ manifest.json 相对仓库根路径(给 D4 直接定位,免于反查)
            "split": {
                "strategy": manifest.strategy,
                "seed": manifest.seed,
                #"rates": {"train": round(tr, 6), "val": round(va, 6), "test": round(te, 6)},
                "rates": {"train": tr, "val": va, "test": te},
                "counts": counts,
            },
        },
    }
    yaml_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info("已写入 dataset yaml: %s", yaml_path)