from __future__ import annotations
import logging, yaml
from pathlib import Path
from typing import List
from od_platform.data_pipeline.split.manifest import SplitManifest

logger = logging.getLogger(__name__)
_SCHEMA_VERSION = "v1"

def write_dataset_yaml(
        yaml_path: Path, * , dataset_root: Path, classes: List[str],
        manifest: SplitManifest, dataset_name: str,
        source_format: str, task: str, manifest_ref: str) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    tr, va, te = manifest.rations
    counts = {s: len(manifest.stems_of(s)) for s in ("train", "val", "test")}
    doc = {
        "path": dataset_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
        "nc": len(classes),
        "odp_meta": {
            "schema_version": _SCHEMA_VERSION,
            "dataset_name": dataset_name,
            "source_format": source_format,
            "task": task,
            "total_version":manifest.tool_version,
            "created_at": manifest.created_at,
            "contract_fingerprint": manifest.contract_fingerprint,
            "manifest_path": manifest_ref,
            "split":{
                "strategy": manifest.strategy,
                "seed": manifest.seed,
                "rate": {"train": tr, "val": va, "test": te},
                "counts": counts
            },
        },
    }
    yaml_path.write_text(yaml.dump(doc,allow_unicode=True, sort_keys=False), encoding="utf-8")
    logger.info("已经写入 dataset yaml: %s", yaml_path)