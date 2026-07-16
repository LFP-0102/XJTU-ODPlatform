"""引用解析:把命令行给的 dataset / yaml / model「名字或路径」,统一解析成确定的 Path。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from od_platform.common.paths import (
    PRETRAINED_MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, TRAINED_MODELS_DIR, CONFIGS_DIR
)

def resolve_ref(ref: str, *, base_dir: Path, default_suffix: Optional[str] = None) -> Path:
    p = Path(ref)
    if p.is_absolute() or len(p.parts) > 1:          # 绝对 / 带分隔符 → 当路径
        return p.resolve()
    name = ref if (not default_suffix or ref.endswith(default_suffix)) else ref + default_suffix
    return (base_dir / name).resolve()               # 裸名字 → 约定目录下

def resolve_dataset(ref: str) -> Path:
    return resolve_ref(ref, base_dir=RAW_DATA_DIR)

def resolve_yaml(ref: str) -> Path:
    return resolve_ref(ref, base_dir=CONFIGS_DIR, default_suffix=".yaml")
"""
def resolve_pretrained_model(ref: str) -> Path:
    return resolve_ref(ref, base_dir=PRETRAINED_MODELS_DIR, default_suffix=".pt")

def resolve_trained_model(ref: str) -> Path:
    return resolve_ref(ref, base_dir=TRAINED_MODELS_DIR, default_suffix=".pt")
"""