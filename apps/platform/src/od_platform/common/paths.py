#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :paths.py
# @Time      :2026/7/13 14:22:23
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from pathlib import Path
from typing import Tuple, List



WORKSPACE_MARKER: str = '.odp-workspace'

def _find_workspace_root(
        start: Path,
        markers: Tuple[str,...] = (WORKSPACE_MARKER,)
    ) -> Path:
    """从start位置开始，沿着目录树向上查找，寻找第一个包含任意一个marker文件的目录"""
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for parent in [current, *current.parents]:
        for marker in markers:
            if (parent / marker).exists():
                return parent
    raise FileNotFoundError(f"找不到 workspace marker 文件{markers},"
                            f"请确认仓库的根存在这个{WORKSPACE_MARKER}文件")

ROOT_DIR: Path = _find_workspace_root(Path(__file__))

# 端的根目录(核心引擎的目录)
APP_DIR: Path = ROOT_DIR / 'apps' / 'platform'

# 共享资产
DATA_DIR: Path = ROOT_DIR / 'data'
MODELS_DIR: Path = ROOT_DIR / 'models'
RUNS_DIR: Path = ROOT_DIR / 'runs'

# 模型的子目录
PRETRAINED_MODELS_DIR: Path = MODELS_DIR / 'pretrained'
TRAINED_MODELS_DIR: Path = MODELS_DIR / 'trained'

# 数据集的目录
RAW_DATA_DIR: Path = DATA_DIR / 'raw'   # 原始数据
PROCESS_DATA_DIR: Path = DATA_DIR / 'processed'  # 处理后的数据

# 端私有资产
CONFIG_DIR: Path = APP_DIR / 'configs'
LOGGING_DIR: Path = APP_DIR / 'logging'
UNIT_TEST_DIR: Path = APP_DIR / 'tests'

DOCS_DIR: Path = ROOT_DIR / 'docs'
SCRIPTS_DIR: Path = ROOT_DIR / "scripts"

META_DIR: Path = ROOT_DIR / ".odp-meta"
META_LOGGING_DIR = META_DIR / "logging"

DATASET_CONFIG_DIR: Path = CONFIG_DIR / "datasets"

def raw_dataset_root(name: str) -> Path:
    return RAW_DATA_DIR / name

def dataset_processed_dir(name: str) -> Path:
    return PROCESS_DATA_DIR / name

def dataset_yaml_path(name: str) -> Path:
    return DATASET_CONFIG_DIR / f"{name}.yaml"

VALIDATION_RUNS_DIR: Path = RUNS_DIR / "data_validation"

def validation_run_dir(run_id: str) -> Path:
    return VALIDATION_RUNS_DIR / run_id

# 模型评估产物根目录(runs/model_evaluation/<run_id>/), 与 data_validation 同构
EVALUATION_RUNS_DIR: Path = RUNS_DIR / "model_evaluation"

def evaluation_run_dir(run_id: str) -> Path:
    return EVALUATION_RUNS_DIR / run_id

RUNTIME_CONFIGS_DIR = CONFIG_DIR / "runtime"

def runtime_config_path(name: str) -> Path:
    return RUNTIME_CONFIGS_DIR / f"{name}.yaml"

# 对外暴露要初始化的目录列表
def get_dirs_to_initialize() -> List[Path]:
    return [
        DATA_DIR,
        MODELS_DIR,
        RUNS_DIR,
        PRETRAINED_MODELS_DIR,
        TRAINED_MODELS_DIR,
        RAW_DATA_DIR,
        PROCESS_DATA_DIR,
        CONFIG_DIR,
        LOGGING_DIR,
        UNIT_TEST_DIR,
        DOCS_DIR,
        SCRIPTS_DIR,
        META_LOGGING_DIR,
        DATASET_CONFIG_DIR,
        RUNTIME_CONFIGS_DIR
    ]
def get_dirs_to_reset() -> List[Path]:
    return [
        PROCESS_DATA_DIR,
        RUNS_DIR,
        TRAINED_MODELS_DIR,
        LOGGING_DIR,
        CONFIG_DIR,

    ]

PROTECTED_DIRS: Tuple[Path, ...] = (
    ROOT_DIR,
    ROOT_DIR / "apps",
    APP_DIR,
    APP_DIR / "src",
    SCRIPTS_DIR,
    DOCS_DIR,
    UNIT_TEST_DIR,
    ROOT_DIR / ".git",
    ROOT_DIR / ".odp-workspace",
    META_DIR,
    META_LOGGING_DIR
)

def is_protected(path: Path) -> bool:
    path = path.resolve(strict=False)
    for protected in PROTECTED_DIRS:
        protected_resolved = protected.resolve(strict=False)
        if path == protected_resolved:
            return True
        if protected_resolved.is_relative_to(path):
            return True
    return False



if __name__ == "__main__":
    print(f"ROOT DIR: {ROOT_DIR}")
    print(f"APP DIR: {APP_DIR}")
    for d in get_dirs_to_initialize():
        print(f"DIR: {d.relative_to(ROOT_DIR)} 将要被创建")
