from pathlib import Path
from typing import List
import logging
from od_platform.common.paths import ROOT_DIR, get_dirs_to_initialize, LOGGING_DIR, RAW_DATA_DIR
from od_platform.common.logging_utils import get_logger
from od_platform.common.performance_utils import time_it
from od_platform.common.string_utils import format_table_row, format_table_separator

LINE_WIDTH: int = 60
logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="init_project",
    temp_log=False
)

def _check_raw_data_status() -> List[str]:
    """检查原始数据集的状态"""
    raw_status: List[str] = []
    rel_raw = RAW_DATA_DIR.relative_to(ROOT_DIR)

    if not RAW_DATA_DIR.exists():
        logger.warning(f"原始数据集根目录不存在：{RAW_DATA_DIR}\n"
                    f"请在该目录下创建以 【数据集名称】明明都文件夹")
        raw_status.append(f"{rel_raw}不存在 -> 请创建并放入数据集")
    elif not any(RAW_DATA_DIR.iterdir()):
        logger.warning(f"原始数据集目录为空： {RAW_DATA_DIR}\n"
                    f"预期的结构是：\n"
                    f" {rel_raw}/<数据集名称>/\n"
                    f" ├── images/ \n"
                    f" └── annotations/ "
        )
        raw_status.append(f"{rel_raw} 为空，请至少放入一个数据集")
    else:
        sub_dirs = [p for p in RAW_DATA_DIR.iterdir() if p.is_dir()]
        logger.info(f"原始数据集根目录就绪，检测到{len(sub_dirs)}个数据集文件")
        raw_status.append(f"{rel_raw} 就绪（包含 {len(sub_dirs)}个数据集）")
        for sub in sorted(sub_dirs):
            raw_status.append(f" - 数据集 {sub.name} 就绪")
    return raw_status


@time_it(iterations=1, name="项目初始化", logger_instance=logger)
def initialize_project() -> None:
    logger.info("=" * LINE_WIDTH)
    logger.info(f"开始初始化项目核心目录".center(LINE_WIDTH, '='))
    logger.info(f"项目的核心目录为：{ROOT_DIR}")


    created: List[Path] = []
    existed: List[Path] = []

    for d in get_dirs_to_initialize():
        rel = d.relative_to(ROOT_DIR)
        if d.exists():
            logger.info(f"文件已经存在：{rel}")
            existed.append(d)
        else:
            d.mkdir(parents=True, exist_ok=True)
            logger.info(f"文件已成功创建：{rel}")
            created.append(d)

    logger.info("=" * LINE_WIDTH)
    logger.info(f"初始化完成".center(LINE_WIDTH, '='))
    logger.info(f"开始检查原始数据就目录".center(LINE_WIDTH, "="))
    raw_status = _check_raw_data_status()

    logger.info(f"项目初始化汇总信息".center(LINE_WIDTH, '-'))
    width = [32,12]
    align = ['left','right']
    logger.info(format_table_row(['目录', '状态'], width, align))
    logger.info(format_table_separator(width))
    for d in created:
        logger.info(format_table_row([str(d.relative_to(ROOT_DIR)), '新建'], width, align))
    for d in existed:
        logger.info(format_table_row([str(d.relative_to(ROOT_DIR)), '已存在'], width, align))
    if not created and not existed:
        logger.info(f"本次初始化目录没有任何变化")

    logger.info("原始数据集目录状态")
    for status in raw_status:
        logger.info(f" - {status}")

    logger.info("项目初始化完成".center(LINE_WIDTH, "="))
    logger.info("下一步：把数据放到 指定位置，即可开始执行数据转换脚本")
    logger.info("请参见 docs/DATASET_GUIDE.md 文件，获取更多详细信息")


if __name__ == "__main__":
    initialize_project()