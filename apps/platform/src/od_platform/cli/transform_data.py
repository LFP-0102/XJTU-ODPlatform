from __future__ import annotations

import argparse
import logging
import sys

from od_platform.common import paths
from od_platform.common.constants import (
    DEFAULT_RANDOM_STATE, DEFAULT_SPLIT_STRATEGY, DEFAULT_TRAIN_RATE, DEFAULT_VAL_RATE,
    AnnotationFormat, SplitStrategy, Task,
)
from od_platform.common.logging_utils import get_logger      # D1,原样用,不改
from od_platform.common.run_context import RunContext
from od_platform.data_pipeline.orchestrator import DatasetPipeline

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_DATA_ERR = 1
EXIT_USAGE = 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="odp-transform",
        description="把 data/raw/<dataset> 转换 + 划分成可训练的 YOLO 数据集。",
    )
    p.add_argument("--dataset", required=True, help="数据集名 = data/raw/ 下的文件夹名(或一个路径)")
    # choices 直接用 constants 的 all():加了新格式/新策略,CLI 选项自动跟着多,无需改这里。
    p.add_argument("--format", required=True, choices=AnnotationFormat.all(), dest="fmt")
    p.add_argument("--task", default=Task.DETECT, choices=Task.all())
    p.add_argument("--split-strategy", default=DEFAULT_SPLIT_STRATEGY, choices=SplitStrategy.all(), dest="strategy")
    p.add_argument("--classes", nargs="+", default=None,
                   help="类别白名单(voc/coco: 过滤+重排; yolo: 仅命名,不能删类)")
    # 默认值全部取自 constants:CLI 和 split_service / SplitOptions 说的是同一句话,不会漂。
    p.add_argument("--train-rate", type=float, default=DEFAULT_TRAIN_RATE)
    p.add_argument("--val-rate", type=float, default=DEFAULT_VAL_RATE)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    a = p.parse_args(argv)

    # D1 纪律:控制台 + 端级(全局)日志,入口装一次(get_logger 一行不改)。日志统一落 logging/。
    get_logger(base_path=paths.LOGGING_DIR, log_type="transform")

    # 现场:唯一时间戳 + 目录(runs/ 下只放产物,不写日志);注入给 pipeline 共用。
    with RunContext("data_pipeline") as run:
        logger.info("odp-transform run_id=%s 现场=%s", run.run_id, run.run_dir)  # ← 靠 run_id 关联日志与现场
        pipe = DatasetPipeline(
            a.dataset, a.fmt, task=a.task, train_rate=a.train_rate, val_rate=a.val_rate,
            classes=a.classes, random_state=a.seed, split_strategy=a.strategy,
            run=run,
        )
        try:
            res = pipe.run()
        except (FileNotFoundError, ValueError) as e:   # 预期内的数据/用法错:友好提示 + 退出码 1
            logger.error("处理失败 (run_id=%s): %s", run.run_id, e)
            return EXIT_DATA_ERR

    print("✅ 完成:", res["counts"], "->", res["yaml"])
    print("   现场:", res["run_dir"], "(日志见 logging/transform/,run_id 相同)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())