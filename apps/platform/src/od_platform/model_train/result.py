#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : result.py
# @Project   : ODPlatform
# @Function  : 训练结果 dataclass + 日志输出(数据/日志分离)
"""训练结果指标。

  TrainMetrics      纯数据(frozen), 给 audit / 日志共用
  log_train_metrics 纯日志, 消费 TrainMetrics, 不持有数据

分离的理由(与 D4 report 同款): 数据结构要能被单测直接断言, 不该只能靠 caplog 抓日志。
复用: D7 的 ValService 可直接共用 TrainMetrics —— train/val 指标结构基本一致。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from od_platform.common.constants import Task
from od_platform.common.string_utils import pad_to_width


# task -> 该任务要展示的指标(数据驱动: 加新任务只加一行)
_METRIC_FIELDS_BY_TASK: Dict[str, List[Tuple[str, str]]] = {
    Task.DETECT: [
        ("metrics/precision(B)", "Precision(B)"),
        ("metrics/recall(B)",    "Recall(B)"),
        ("metrics/mAP50(B)",     "mAP50(B)"),
        ("metrics/mAP50-95(B)",  "mAP50-95(B)"),
    ],
    Task.SEGMENT: [
        ("metrics/precision(B)", "Precision(B)"),
        ("metrics/recall(B)",    "Recall(B)"),
        ("metrics/mAP50(B)",     "mAP50(B)"),
        ("metrics/mAP50-95(B)",  "mAP50-95(B)"),
        ("metrics/precision(M)", "Precision(M)"),
        ("metrics/recall(M)",    "Recall(M)"),
        ("metrics/mAP50(M)",     "mAP50(M)"),
        ("metrics/mAP50-95(M)",  "mAP50-95(M)"),
    ],
}


def _safe_float(value: Any, default: float = math.nan) -> float:
    """numpy scalar / None / 字符串 -> float, 失败给 NaN。

    ★ 这里【该】兜底: ultralytics 返回结构跨版本会变, 某键缺失是可预期的常态。
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class TrainMetrics:
    """一次训练结果的结构化快照(纯数据)。

    设计点:
      · frozen: 既成事实不该被改
      · speed_ms / overall / class_map dict 而非固定字段 —— ultralytics 内容随版本变
      · ★ 没有 timestamp 字段, run_id 由调用方注入 —— 见 from_yolo_results
    """
    run_id:          str
    task:            str
    save_dir:        Path
    speed_ms:        Dict[str, float]
    overall:         Dict[str, float]
    class_map_50_95: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yolo_results(cls, results: Any, *, run_id: str,
                          model_trainer: Any = None) -> "TrainMetrics":
        """从 ultralytics 训练/验证结果对象构造。

        ★ run_id 必填 keyword-only, 不是内部 datetime.now():
          这方法在【训练结束时】被调用, 而 run_id 是【训练开始前】定的。
          若在这里 now(), 审计里就会出现两个时间, "结果属于哪次"只能靠猜(裂缝 A 复发)。
          同 naming.py: 这个类【不该有能力知道现在几点】。
        """
        save_dir_raw = getattr(results, "save_dir", None)
        if save_dir_raw is None and model_trainer is not None:
            save_dir_raw = getattr(model_trainer, "save_dir", None)
        save_dir = Path(save_dir_raw) if save_dir_raw is not None else Path("unknown")

        speed_raw = getattr(results, "speed", {}) or {}
        speed_ms: Dict[str, float] = {
            k: _safe_float(speed_raw.get(k))
            for k in ("preprocess", "inference", "loss", "postprocess")
        }
        valid = [v for v in speed_ms.values() if not math.isnan(v)]
        speed_ms["total"] = sum(valid) if valid else math.nan

        overall: Dict[str, float] = {"fitness": _safe_float(getattr(results, "fitness", None))}
        for k, v in (getattr(results, "results_dict", {}) or {}).items():
            overall[str(k)] = _safe_float(v)

        # 类别级 mAP: maps 是 numpy array, 这里只用鸭子类型, 不 import numpy
        # (免得纯数据模块背上重依赖, 同 service 里延迟 import ultralytics)
        class_map: Dict[str, float] = {}
        names = getattr(results, "names", {}) or {}
        maps = getattr(results, "maps", None)
        size = int(getattr(maps, "size", 0) or 0)
        if names and size:
            for idx, class_name in names.items():
                if int(idx) < size:
                    class_map[str(class_name)] = _safe_float(maps[idx])

        return cls(run_id=run_id, task=getattr(results, "task", "unknown"),
                   save_dir=save_dir, speed_ms=speed_ms, overall=overall,
                   class_map_50_95=class_map)

    def to_dict(self) -> Dict[str, Any]:
        """转 dict: 路径转字符串, NaN 转 None。可直接 json.dumps 进审计。"""
        def _clean(d: Dict[str, float]) -> Dict[str, Any]:
            return {k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in d.items()}
        return {
            "run_id":          self.run_id,
            "task":            self.task,
            "save_dir":        str(self.save_dir),
            "speed_ms":        _clean(self.speed_ms),
            "overall":         _clean(self.overall),
            "class_map_50_95": _clean(self.class_map_50_95),
        }


def log_train_metrics(metrics: TrainMetrics, *,
                      logger: Optional[logging.Logger] = None,
                      key_width: int = 20, width: int = 60) -> None:
    """把 TrainMetrics 打进 logger。纯展示, 不持有数据。"""
    log = logger or logging.getLogger(__name__)
    log.info("=" * width)
    log.info(f"训练结果 ({metrics.task})".center(width))
    log.info("=" * width)
    log.info(f"{pad_to_width('run_id', key_width)}: {metrics.run_id}")
    log.info(f"{pad_to_width('现场目录', key_width)}: {metrics.save_dir}")

    log.info("处理速度 (ms/image)".center(width))
    log.info("-" * width)
    for disp, key in [("预处理", "preprocess"), ("推理", "inference"),
                      ("损失计算", "loss"), ("后处理", "postprocess"), ("总计", "total")]:
        log.info(f"{pad_to_width(disp, key_width)}: {metrics.speed_ms.get(key, math.nan):.3f} ms")

    log.info("整体评估指标".center(width))
    log.info("-" * width)
    log.info(f"{pad_to_width('Fitness', key_width)}: {metrics.overall.get('fitness', math.nan):.4f}")
    fields = _METRIC_FIELDS_BY_TASK.get(metrics.task, [])
    if fields:
        for raw_key, disp in fields:
            log.info(f"{pad_to_width(disp, key_width)}: {metrics.overall.get(raw_key, math.nan):.4f}")
    else:
        log.info(f"(task='{metrics.task}' 不在已知表, 打印 results_dict 全量)")
        for k, v in metrics.overall.items():
            if k != "fitness":
                log.info(f"{pad_to_width(k, key_width)}: {v:.4f}")

    if metrics.class_map_50_95:
        # ★ (Box) 跟着 task 走 —— segment 任务下 maps 是 mask 的, 写死会撒谎
        kind = "Mask" if metrics.task == Task.SEGMENT else "Box"
        log.info(f"类别级 mAP@0.5:0.95 ({kind})".center(width))
        log.info("-" * width)
        valid = {k: v for k, v in metrics.class_map_50_95.items() if not math.isnan(v)}
        if valid:
            for name, v in sorted(valid.items(), key=lambda kv: kv[1], reverse=True):
                log.info(f"{pad_to_width(name, key_width)}: {v:.4f}")
        else:
            log.warning("类别 mAP 全为 NaN, 跳过打印")
    log.info("=" * width)