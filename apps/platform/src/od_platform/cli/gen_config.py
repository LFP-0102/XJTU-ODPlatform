import argparse

from od_platform.runtime_config.generator import ConfigGenerator
from od_platform.common.paths import runtime_config_path
from od_platform.runtime_config.registry import CONFIG_REGISTRY
from pathlib import Path
from od_platform.common.logging_utils import get_logger
from od_platform.common.paths import LOGGING_DIR
logger = get_logger(
    base_path=LOGGING_DIR,
    log_type="gen_config",
    temp_log=False,
)


def main():
    parser = argparse.ArgumentParser(
        prog="odp-gen-config",
        description="生成 YOLO 运行配置 YAML 模板",
        epilog="例如：odp-gen-config train",
    )
    parser.add_argument(
        "name",
        choices=list(CONFIG_REGISTRY),                # ★ 合法取值也由那张共享表说了算
        help="要生成的配置名 (train / val / infer)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出路径 (默认: configs/runtime/<name>.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有文件 (默认不覆盖)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="覆盖时不备份原文件 (默认会备份成 <name>.yaml.bak.<时间戳>)",
    )
    args = parser.parse_args()

    # name → 配置类: 读那张共享表 (registry.py), 不在这里另立 if-elif
    config_class, title = CONFIG_REGISTRY[args.name]

    # 输出路径: 走 paths.py SSoT
    output_path = args.output or runtime_config_path(args.name)

    # 生成
    gen = ConfigGenerator()
    success = gen.generate(
        config_class,
        output_path,
        overwrite=args.overwrite,
        backup=not args.no_backup,
        title=title,
    )

    if success:
        logger.info(f"配置文件已生成")
    else:
        logger.info(
            f"你可以使用——overwrite参数覆盖输出路径，默认不覆盖"
        )