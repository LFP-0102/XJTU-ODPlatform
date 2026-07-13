from pathlib import Path
from typing import List,Tuple

from fontTools.misc.plistlib import Data
from tests import MODELS
from ultralytics.utils import RUNS_DIR

WORKSPACE_MARKER: str = '.odp-workspace'

def _find_workspace_root(
        start: Path,
        makers: Tuple[str,...] = (WORKSPACE_MARKER,)
) -> Path:
    """""从start位置开始，沿着目录树向上查找，寻找第一个包含任意一个marker文件的目录"""
    current = start.resolve()
    if current.is_file():
        current = current.parent
    for parent in [current,*current.parents]:
        for marker in makers:
            if (parent / marker).exists():
                return parent
    raise FileNotFoundError(f"找不到 workspace marker 文件{markers}),"
                             f"请确认仓库的根存在这个{WORKSPACE_MARKER}文件")

ROOT_DIR: Path = _find_workspace_root(Path(__file__))

#端的根目录(核心引擎的目录)
APP_DIR: Path = ROOT_DIR / 'apps' / 'platform'

#共享资产
DATA_DIR: Path = ROOT_DIR / 'data'
MODELS_DIR: Path = ROOT_DIR / 'models'
RUNS_DIR: Path = ROOT_DIR / 'runs'

# 模型的子目录
PRETRAINED_MODELS_DIR: Path = MODELS_DIR / 'pretrained'
TRAINED_MODELS_DIR: Path = MODELS_DIR / 'trained'

# 数据集的目录
RAW_DATA_DIR: Path = DATA_DIR / 'raw'
PROCESSED_DATA_DIR: Path = DATA_DIR / 'processed'


# 端私有资产
CONFIG_DIR: Path = APP_DIR / 'configs'
LOGGING_DIR: Path = ROOT_DIR / 'logging'
UNIT_TEST_DIR: Path = APP_DIR / 'tests'

DOC_DIR: Path = ROOT_DIR / 'docs'
SCRIPTS_DIR: Path = ROOT_DIR / 'scripts'

# 对外暴露要初始化的目录列表
def get_dirs_to_initialize() -> List[Path]:
    return [
        DATA_DIR,
        MODELS_DIR,
        RUNS_DIR,
        PRETRAINED_MODELS_DIR,
        TRAINED_MODELS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CONFIG_DIR,
        LOGGING_DIR,
        UNIT_TEST_DIR,
        DOC_DIR,
        SCRIPTS_DIR,
    ]

if __name__ == '__main__':
    print(f"RO0T DIR: {ROOT_DIR}")
    print(f"APP DIR: {APP_DIR}")
    for d in get_dirs_to_initialize():
        print(f"DIR:{d.relative_to(ROOT_DIR)}将要被创建")


