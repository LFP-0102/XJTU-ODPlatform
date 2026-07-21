# apps/platform/src/odp_platform/cli/infer.py
#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""ODPlatform odp-infer CLI 入口.

职责 (跟 odp-train 完全同构):
  1. argparse → cli_args dict, 先 build_infer_config 建好配置 (好拿到模型名)
  2. RunContext("inference") 定 run_id / run_dir(runs/inference/<run_id>)
  3. 装 logging handler (get_logger, console+file), 用 run_id + 模型名把日志名预先定死
  4. 调 InferService.predict(config, merger, run, ...) 跑
  5. 退出码: 0 成功, 1 失败 (CI 友好)

★ 不传任何接缝参数(output_sink/hooks/cancel_token) → service 走默认 → 等价老式 CLI 行为.
"""
from __future__ import annotations

import argparse
import logging
import sys

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext
from od_platform.runtime_config import build_infer_config
from od_platform.model_infer import InferService


def _str2bool(v: str) -> bool:
    """把 --save/--show 后面跟的字符串解析成布尔值(容忍 true/false/1/0/yes/no/on/off)."""
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"需要布尔值 (true/false), 收到: {v!r}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="odp-infer",
        description="ODPlatform YOLO 推理 CLI",
    )
    # ---- 输入输出 ----
    p.add_argument("--source", type=str, default=None,
                help="输入源: 摄像头号(0/1) / 视频文件 / 图片或目录 / RTSP. 不传则用 infer.yaml.")
    p.add_argument("--model", type=str, default=None,
                help="模型文件名 (best.pt / yolo11n.pt). 不传则用 infer.yaml.")
    p.add_argument("--yaml", type=str, default=None,
                help="D5 infer.yaml 路径. 不传走默认.")
    p.add_argument("--pipeline-yaml", type=str, default=None,
                help="帧源+美化 配置: 裸名(在 runtime 配置目录下找)或完整路径. 不传用默认 infer_pipeline.yaml.")
    p.add_argument("--init-pipeline-config", action="store_true",
                help="在 --pipeline-yaml 指定位置(或默认位置)生成一份带注释的默认 帧源+美化 配置后退出, 不跑推理.")
    p.add_argument("--name", type=str, default=None, dest="experiment_name",
                help="输出子目录名 (重复自增: predict / predict2 / ...).")

    # ---- 推理参数 ----
    p.add_argument("--conf", type=float, default=None, help="置信度阈值 (0~1).")
    p.add_argument("--iou",  type=float, default=None, help="NMS IoU 阈值 (0~1).")
    p.add_argument("--imgsz", type=int, default=None, help="推理输入尺寸.")
    p.add_argument("--max-det", type=int, default=None, dest="max_det", help="单图最大检测数.")
    p.add_argument("--classes", type=int, nargs="+", default=None, help="只保留这些类别 ID.")
    p.add_argument("--device", type=str, default=None, help="cpu / 0 / 0,1 / mps.")
    p.add_argument("--batch", type=int, default=None, help="批大小 (视频/图片夹).")

    # ---- 显示 / 存盘 ----
    # --show / --save 支持三种写法: 裸写(=开) / 显式 --save true / --save false;
    # --no-show / --no-save 保留为等价 false 的别名. 不放互斥组 —— 同时给两个不再报错,
    # 而是"命令行里靠后的那个生效"(argparse 天然行为), 对手滑更宽容.
    p.add_argument("--show", nargs="?", const=True, default=None, type=_str2bool,
                help="弹窗显示画面: --show 或 --show true 开, --show false 关. 不传则用 infer.yaml.")
    p.add_argument("--no-show", dest="show", action="store_const", const=False,
                help="等价 --show false(不弹窗).")
    p.add_argument("--save", nargs="?", const=True, default=None, type=_str2bool,
                help="保存渲染后的结果(美化图): --save 或 --save true 开, --save false 关. 不传则用 infer.yaml.")
    p.add_argument("--no-save", dest="save", action="store_const", const=False,
                help="等价 --save false(不存盘).")
    # 额外落盘(对齐 ultralytics): 存盘位置在 runs/inference/<run_id>/result/ 下.
    # 这几个不传就走 infer.yaml 的默认(默认 False); 传了即开启.
    p.add_argument("--save-txt", dest="save_txt", action="store_true", default=None,
                help="额外保存 YOLO 格式标签到 result/labels/<名>.txt.")
    p.add_argument("--save-conf", dest="save_conf", action="store_true", default=None,
                help="标签 txt 里追加置信度(需配合 --save-txt).")
    p.add_argument("--save-crop", dest="save_crop", action="store_true", default=None,
                help="额外保存每个检测框的切片到 result/crops/<类别>/.")
    p.add_argument("--no-viz", action="store_true", help="关掉美化, 用 YOLO 原生 plot().")
    p.add_argument("--no-hud", action="store_true", help="画面不叠 HUD 信息面板.")

    # ---- 其他 ----
    p.add_argument("--warmup", type=int, default=0,
                   help="启动丢弃前 N 帧 (摄像头帧率不稳).")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="日志级别.")
    return p


def _ns_to_cli_args(ns: argparse.Namespace) -> dict:
    """argparse Namespace → cli_args dict (D5 merger 吃 dict).

    只放非 None 的字段, 让 D5 / pydantic 的 default 兜底.
    `--no-viz` / `--no-hud` / `--warmup` / `--yaml` / `--pipeline-yaml` / `--log-level`
    不进 cli_args (这些是 CLI 自己的行为开关, 不是 D5 配置字段).
    """
    keys = ("source", "model", "experiment_name",
            "conf", "iou", "imgsz", "max_det", "classes", "device", "batch",
            "show", "save", "save_txt", "save_conf", "save_crop")
    return {k: v for k in keys if (v := getattr(ns, k, None)) is not None}


def main(argv: list[str] | None = None) -> int:
    ns = _build_parser().parse_args(argv)

    # 显式生成默认 帧源+美化 配置后退出(方便"先生成再编辑, 然后再跑推理")
    if ns.init_pipeline_config:
        from od_platform.model_infer.pipeline_config import (
            resolve_pipeline_yaml_path, write_default_pipeline_yaml,
        )
        target = resolve_pipeline_yaml_path(ns.pipeline_yaml)
        created = write_default_pipeline_yaml(target, overwrite=False)
        if created:
            print(f"已生成默认 pipeline 配置: {target}")
        else:
            print(f"配置已存在, 未覆盖: {target}\n如需重建请先删除或改名该文件后再执行本命令.")
        return 0

    cli_args = _ns_to_cli_args(ns)

    # 1) 先把 D5 推理配置建好 —— 跟 odp-train 同构: CLI 建 config, 好拿到模型名给日志/产物预命名.
    #    配置阶段的错误在这里直接落地成退出码 1 (此时日志还没装, 直接写 stderr).
    try:
        config, merger = build_infer_config(
            yaml_path=ns.yaml or "infer.yaml",
            cli_args=cli_args,
        )
    except Exception as e:
        sys.stderr.write(f"\n推理配置错误: {e}\n")
        return 1

    model_ref = getattr(config, "model", None) or "yolo11n.pt"

    # 2) RunContext 定 run_id / run_dir(runs/inference/<run_id>), 与训练 runs/training/<run_id> 对齐
    with RunContext("inference") as run:
        # ★ 纪律 B: 整个进程里【唯一】装 handler 的地方.
        #   用 run_id + 模型名把日志名预先定死(get_logger 内部走 run_stem 生成),
        #   不再"先临时名跑、事后重命名" —— 这正是要跟训练模板保持一致的点.
        get_logger(
            base_path=paths.LOGGING_DIR,
            log_type="infer",
            run_id=run.run_id,
            model_name=model_ref,
            log_level=getattr(logging, ns.log_level),
        )

        result = InferService().predict(
            config=config,
            merger=merger,
            run=run,
            pipeline_yaml=ns.pipeline_yaml,
            beautify=(not ns.no_viz),
            show_info=(not ns.no_hud),
            warmup_frames=ns.warmup,
            # ★ 不传 output_sink / hooks / cancel_token → service 走默认 → 等价老式 CLI
        )

    if result.success:
        return 0
    sys.stderr.write(f"\n推理失败: {result.error}\n")
    if result.log_path:
        sys.stderr.write(f"详细日志见: {result.log_path}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())