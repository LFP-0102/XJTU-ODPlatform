#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : infer_eval.py
# @Path       : XJTU-ODPlatform/apps/platform/src/od_platform/model_eval/infer_eval.py
# @Project    : XJTU-ODPlatform
# @Function   : 对 odp-infer 推理结果做评估 —— 预测标签 vs 数据集 ground truth
"""从推理输出评估模型质量.

核心流程:
  1. 读取 odp-infer 产出的 YOLO 格式预测标签 (--save-txt)
  2. 从数据集 YAML 读取 ground truth 标注
  3. 按文件名 stem 匹配预测 ↔ 真值
  4. IoU 匹配 (同 ultralytics validator 逻辑), 累积 tp / conf / pred_cls / target_cls
  5. 用 ultralytics 内置 ap_per_class 计算 mAP / Precision / Recall / F1
  6. 构建 EvalMetrics → 生成与常规 odp-eval 完全一致的报告

用法:
  from od_platform.model_eval.infer_eval import evaluate_infer_results
  result = evaluate_infer_results(
      infer_dir="runs/inference/20240724-120000",
      data_yaml="configs/datasets/helmet.yaml",
      split="val",
      config=val_config,
      run=run_ctx,
  )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from od_platform.common import refs
from od_platform.common.run_context import RunContext
from od_platform.model_eval.metrics import EvalMetrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 轻量数据结构
# ---------------------------------------------------------------------------


@dataclass
class PredBox:
    """一条模型预测框 (YOLO 归一化坐标)."""
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    conf: float


@dataclass
class GTBox:
    """一条 ground truth 框 (YOLO 归一化坐标, 无置信度)."""
    class_id: int
    cx: float
    cy: float
    w: float
    h: float


# ---------------------------------------------------------------------------
# 坐标工具
# ---------------------------------------------------------------------------


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    """cx,cy,w,h (归一化) → x1,y1,x2,y2."""
    hw, hh = w / 2.0, h / 2.0
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def _box_iou(box1: Tuple[float, float, float, float],
             box2: Tuple[float, float, float, float]) -> float:
    """两个 xyxy 框的 IoU."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 读取推理预测
# ---------------------------------------------------------------------------


def load_predictions_from_infer(infer_dir: str | Path) -> Dict[str, List[PredBox]]:
    """从 odp-infer 输出目录读取预测标签.

    扫描 result/labels/*.txt, 每行格式: class_id cx cy w h conf.
    返回 {image_stem: [PredBox, ...]}, 未保存标签时返回空 dict.
    """
    infer_dir = Path(infer_dir)
    labels_dir = infer_dir / "result" / "labels"
    if not labels_dir.is_dir():
        logger.warning("推理输出中未找到 result/labels/ 目录 (%s), 确认推理时用了 --save-txt", labels_dir)
        return {}

    predictions: Dict[str, List[PredBox]] = {}
    for txt_path in sorted(labels_dir.glob("*.txt")):
        stem = txt_path.stem
        boxes: List[PredBox] = []
        try:
            for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                conf = float(parts[5]) if len(parts) >= 6 else 1.0
                boxes.append(PredBox(class_id=cls_id, cx=cx, cy=cy, w=w, h=h, conf=conf))
        except Exception as e:
            logger.warning("解析预测文件失败 %s: %s", txt_path, e)
            continue
        predictions[stem] = boxes

    logger.info("从 %s 加载了 %d 张图的预测结果", labels_dir, len(predictions))
    return predictions


# ---------------------------------------------------------------------------
# 读取数据集 ground truth
# ---------------------------------------------------------------------------


def load_ground_truth(data_yaml: str | Path,
                      split: str = "val") -> Tuple[Dict[str, List[GTBox]], Dict[int, str]]:
    """从数据集 YAML 读取指定 split 的 ground truth 标签.

    Returns:
        (gt_dict, class_names)
        gt_dict:  {image_stem: [GTBox, ...]}
        class_names: {0: "helmet", 1: "person", ...}
    """
    data_path = Path(data_yaml)
    if not data_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {data_path}")

    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))

    # 类别名
    names: Dict[int, str]
    raw_names = data.get("names", {})
    if isinstance(raw_names, list):
        names = {i: str(n) for i, n in enumerate(raw_names)}
    else:
        names = {int(k): str(v) for k, v in raw_names.items()}

    # 解析 labels 目录
    # 数据集 YAML 中:
    #   path: <数据根目录>      ← 全局路径前缀 (可选)
    #   val:  images/val       ← 相对于 path 或 YAML 所在目录
    if split not in data:
        available = [k for k in data if k not in ("names", "nc", "path", "download")]
        raise ValueError(
            f"数据集 {data_path} 中未找到 split '{split}', "
            f"可用: {available}"
        )

    # 数据根目录: 优先 YAML 里的 path, 兜底 YAML 文件所在目录
    dataset_root = Path(data.get("path", data_path.parent))
    if not dataset_root.is_absolute():
        dataset_root = (data_path.parent / dataset_root).resolve()

    # 图片目录: data[split] 可能是绝对路径或相对路径
    images_dir_raw = data[split]
    if isinstance(images_dir_raw, list):
        images_dir_raw = images_dir_raw[0]
    images_dir = Path(images_dir_raw)
    if not images_dir.is_absolute():
        images_dir = (dataset_root / images_dir).resolve()

    # 推导 labels 目录
    # 常见结构: images/val → labels/val
    labels_dir = _derive_labels_dir(images_dir)
    if labels_dir is None or not labels_dir.is_dir():
        # 兜底: 从 dataset_root 直接找 labels/<split>/
        for candidate in [
            dataset_root / "labels" / split,
            data_path.parent / "labels" / split,
        ]:
            if candidate.is_dir():
                labels_dir = candidate
                break

    if labels_dir is None or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"无法定位 ground truth 标签目录.\n"
            f"  图片目录: {images_dir}\n"
            f"  期望标签在: {labels_dir}\n"
            f"  请确认数据集目录结构包含 labels/{split}/ 子目录."
        )

    gt: Dict[str, List[GTBox]] = {}
    for txt_path in sorted(labels_dir.glob("*.txt")):
        stem = txt_path.stem
        boxes: List[GTBox] = []
        try:
            for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append(GTBox(class_id=cls_id, cx=cx, cy=cy, w=w, h=h))
        except Exception as e:
            logger.warning("解析 GT 文件失败 %s: %s", txt_path, e)
            continue
        if boxes:
            gt[stem] = boxes

    logger.info("从 %s 加载了 %d 张图的 ground truth (%d 个类别)",
                labels_dir, len(gt), len(names))
    return gt, names


def _derive_labels_dir(images_dir: Path) -> Path | None:
    """从图片目录推导 labels 目录.

    替换路径中首次出现的 'images' → 'labels'.
    例: .../images/val → .../labels/val
    """
    parts = list(images_dir.parts)
    try:
        idx = parts.index("images")
    except ValueError:
        return None
    parts[idx] = "labels"
    return Path(*parts) if not parts[0] else Path(parts[0]).joinpath(*parts[1:])


# ---------------------------------------------------------------------------
# 匹配 & 指标计算
# ---------------------------------------------------------------------------


_IOU_THRESHOLDS: np.ndarray = np.linspace(0.5, 0.95, 10)  # COCO 标准 10 个 IoU 阈值


def _match_predictions_multi_iou(
    preds: List[PredBox],
    gts: List[GTBox],
) -> Tuple[List[List[int]], List[float], List[int], List[int], int]:
    """单张图: 预测框与真值框在 10 个 IoU 阈值下同时匹配.

    算法 (与 ultralytics DetectionValidator 一致):
      对每个 IoU 阈值独立匹配 —— 预测按置信度降序, 贪心匹配同类 GT.

    Returns:
        tp_per_iou:    List of 10 lists, 每个元素是该 IoU 阈值下每个预测的 TP(1)/FP(0)
        conf_list:     每个预测的置信度
        pred_cls_list: 每个预测的类别 ID
        target_cls_list: 匹配到的 GT 类别 ID (IoU=0.5 时的结果, FP 填 -1)
        matched_count: IoU=0.5 下匹配到的 GT 数量
    """
    n_thresh = len(_IOU_THRESHOLDS)

    if not preds:
        return [[] for _ in range(n_thresh)], [], [], [], 0

    # 按置信度降序排列
    sorted_preds = sorted(enumerate(preds), key=lambda x: -x[1].conf)

    # 按类别分组 GT
    gt_by_class: Dict[int, List[Tuple[int, GTBox]]] = {}
    for gt_idx, gt in enumerate(gts):
        gt_by_class.setdefault(gt.class_id, []).append((gt_idx, gt))

    conf_list: List[float] = []
    pred_cls_list: List[int] = []
    target_cls_list: List[int] = []

    # 对每个 IoU 阈值独立维护 matched_gt
    matched_per_iou: List[set] = [set() for _ in range(n_thresh)]
    tp_per_iou: List[List[int]] = [[] for _ in range(n_thresh)]

    for _, pred in sorted_preds:
        pred_xyxy = _xywh_to_xyxy(pred.cx, pred.cy, pred.w, pred.h)
        candidates = gt_by_class.get(pred.class_id, [])

        # 计算与所有候选 GT 的 IoU
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in candidates:
            gt_xyxy = _xywh_to_xyxy(gt.cx, gt.cy, gt.w, gt.h)
            iou = _box_iou(pred_xyxy, gt_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # 对每个 IoU 阈值判定 TP/FP
        for t_idx in range(n_thresh):
            thr = float(_IOU_THRESHOLDS[t_idx])
            if best_iou >= thr and best_gt_idx >= 0 and best_gt_idx not in matched_per_iou[t_idx]:
                tp_per_iou[t_idx].append(1)
                matched_per_iou[t_idx].add(best_gt_idx)
            else:
                tp_per_iou[t_idx].append(0)

        conf_list.append(pred.conf)
        pred_cls_list.append(pred.class_id)
        # target_cls: IoU=0.5 时的匹配结果
        if best_iou >= float(_IOU_THRESHOLDS[0]) and best_gt_idx >= 0:
            target_cls_list.append(pred.class_id)
        else:
            target_cls_list.append(-1)

    return tp_per_iou, conf_list, pred_cls_list, target_cls_list, len(matched_per_iou[0])


def _compute_metrics(
    all_preds: Dict[str, List[PredBox]],
    all_gts: Dict[str, List[GTBox]],
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    """主计算: mAP50 + mAP50-95 + 每类指标.

    一次性在 10 个 IoU 阈值上匹配, 构造 2D tp 数组 (N, 10),
    调 ultralytics ap_per_class 一次得到所有指标.

    Returns dict, 字段与 EvalMetrics 构造所需对齐.
    """
    from ultralytics.utils.metrics import ap_per_class

    n_thresh = len(_IOU_THRESHOLDS)
    all_tp_cols: List[List[int]] = [[] for _ in range(n_thresh)]
    all_conf: List[float] = []
    all_pred_cls: List[int] = []
    all_target_cls: List[int] = []
    gt_count_per_class: Dict[int, int] = {}

    common = sorted(set(all_preds.keys()) & set(all_gts.keys()))
    only_gt = sorted(set(all_gts.keys()) - set(all_preds.keys()))

    skipped = 0

    # --- 仅 GT 无预测的图: 全部 FN ---
    for stem in only_gt:
        for gt in all_gts[stem]:
            gt_count_per_class[gt.class_id] = gt_count_per_class.get(gt.class_id, 0) + 1
            for t_idx in range(n_thresh):
                all_tp_cols[t_idx].append(0)
            all_conf.append(0.0)
            all_pred_cls.append(gt.class_id)
            all_target_cls.append(gt.class_id)

    # --- 共同图片: 匹配+补 FN ---
    for stem in common:
        preds = all_preds[stem]
        gts = all_gts[stem]

        for gt in gts:
            gt_count_per_class[gt.class_id] = gt_count_per_class.get(gt.class_id, 0) + 1

        if not preds:
            skipped += 1
            # 所有 GT 都没匹配 → 每条 GT 补一条 FN dummy
            for gt in gts:
                for t_idx in range(n_thresh):
                    all_tp_cols[t_idx].append(0)
                all_conf.append(0.0)
                all_pred_cls.append(gt.class_id)
                all_target_cls.append(gt.class_id)
            continue

        tp_per_iou, conf, pred_cls, target_cls, matched_count = _match_predictions_multi_iou(preds, gts)
        for t_idx in range(n_thresh):
            all_tp_cols[t_idx].extend(tp_per_iou[t_idx])
        all_conf.extend(conf)
        all_pred_cls.extend(pred_cls)
        all_target_cls.extend(target_cls)

        # 补全未匹配 GT (FN): 每个未匹配 GT 加一条 tp=0 的 dummy 条目
        #   ap_per_class 用 target_cls 的 unique + counts 统计每类 GT 总数,
        #   不加这些 dummy 会导致 recall 分母偏小
        gt_count = len(gts)
        fn_count = gt_count - matched_count
        for _ in range(fn_count):
            # FN 条目不参与排序 (conf=0, 排最后), target_cls 随便取一个 GT 类别
            dummy_cls = gts[0].class_id if gts else 0
            for t_idx in range(n_thresh):
                all_tp_cols[t_idx].append(0)
            all_conf.append(0.0)
            all_pred_cls.append(dummy_cls)
            all_target_cls.append(dummy_cls)

    if skipped:
        logger.debug("%d 张图有 GT 但无预测 (全部漏检)", skipped)

    if not all_conf:
        return {
            "precision": math.nan, "recall": math.nan,
            "mAP50": math.nan, "mAP50_95": math.nan,
            "f1": math.nan, "accuracy": math.nan, "fitness": math.nan,
            "speed_ms": {}, "per_class": {},
            "tp": 0, "fp": 0, "fn": sum(len(v) for v in all_gts.values()),
        }

    # 构造 2D tp 数组: (N, 10)
    tp_2d = np.column_stack([np.array(col, dtype=bool) for col in all_tp_cols])
    conf_arr = np.array(all_conf, dtype=np.float64)
    pred_cls_arr = np.array(all_pred_cls, dtype=int)
    target_cls_arr = np.array(all_target_cls, dtype=int)

    # ap_per_class 一次计算所有 IoU 阈值
    # 返回值: (tp, fp, p, r, f1, ap, ap_class, unique_classes)
    result = ap_per_class(
        tp_2d, conf_arr, pred_cls_arr, target_cls_arr,
        plot=False, names=class_names, prefix="InferEval",
    )

    # ap_arr: (nc, 10) — 每类在每个 IoU 阈值下的 AP
    ap_arr = result[5]      # shape: (nc, 10) or (nc,)
    ap_class_indices = result[6]  # shape: (nc,)
    p_arr = result[2]
    r_arr = result[3]
    f1_arr = result[4]

    # 处理 ap_arr 维度: 可能是 1D (单 IoU) 或 2D (多 IoU)
    if ap_arr.ndim == 1:
        ap_arr = ap_arr.reshape(-1, 1)

    per_class_ap50: Dict[int, float] = {}       # IoU=0.5 列
    per_class_mAP50_95: Dict[int, float] = {}    # 10 列平均
    per_class_p: Dict[int, float] = {}
    per_class_r: Dict[int, float] = {}
    per_class_f1: Dict[int, float] = {}

    for i, cls_idx in enumerate(ap_class_indices):
        cls_id = int(cls_idx)
        if cls_id < 0:
            continue  # 跳过 background 类 (target_cls=-1 的 FP 预测)
        per_class_ap50[cls_id] = float(ap_arr[i, 0])   # 第一列 = IoU=0.5
        per_class_mAP50_95[cls_id] = float(np.mean(ap_arr[i, :]))  # 所有列平均
        per_class_p[cls_id] = float(p_arr[i])
        per_class_r[cls_id] = float(r_arr[i])
        per_class_f1[cls_id] = float(f1_arr[i])

    # 整体指标
    valid_ap50 = [v for v in per_class_ap50.values() if not math.isnan(v)]
    valid_ap50_95 = [v for v in per_class_mAP50_95.values() if not math.isnan(v)]
    valid_p = [v for v in per_class_p.values() if not math.isnan(v)]
    valid_r = [v for v in per_class_r.values() if not math.isnan(v)]
    valid_f1 = [v for v in per_class_f1.values() if not math.isnan(v)]

    mAP50 = float(np.mean(valid_ap50)) if valid_ap50 else math.nan
    mAP50_95 = float(np.mean(valid_ap50_95)) if valid_ap50_95 else math.nan
    precision = float(np.mean(valid_p)) if valid_p else math.nan
    recall = float(np.mean(valid_r)) if valid_r else math.nan
    f1 = float(np.mean(valid_f1)) if valid_f1 else math.nan

    # 混淆矩阵统计 (基于 IoU=0.5)
    tp_total = int(tp_2d[:, 0].sum())
    all_pred_total = sum(len(v) for v in all_preds.values())
    all_gt_total = sum(len(v) for v in all_gts.values())
    fp_total = max(0, all_pred_total - tp_total)
    fn_total = max(0, all_gt_total - tp_total)
    accuracy = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else math.nan

    # 每类汇总
    per_class: Dict[str, Dict[str, float]] = {}
    for cls_id, name in class_names.items():
        per_class[str(name)] = {
            "precision": per_class_p.get(cls_id, math.nan),
            "recall": per_class_r.get(cls_id, math.nan),
            "f1": per_class_f1.get(cls_id, math.nan),
            "mAP50": per_class_ap50.get(cls_id, math.nan),
            "mAP50_95": per_class_mAP50_95.get(cls_id, math.nan),
        }

    return {
        "precision": precision,
        "recall": recall,
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "f1": f1,
        "accuracy": accuracy,
        "fitness": f1,  # fitness ≈ F1 (无 ultralytics 原生 fitness)
        "speed_ms": {},
        "per_class": per_class,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


# ---------------------------------------------------------------------------
# 主编排函数
# ---------------------------------------------------------------------------


def evaluate_infer_results(
    *,
    infer_dir: str | Path,
    data_yaml: str | Path,
    split: str = "val",
    config: Any = None,
    run: RunContext,
    merger: Any = None,
    write_report: bool = True,
    save_history: bool = True,
    enhanced_analysis: bool = True,
) -> Any:
    """评估一次 odp-infer 推理结果, 生成报告.

    Args:
        infer_dir:  odp-infer 输出目录 (runs/inference/<run_id>/)
        data_yaml:  数据集配置文件
        split:      数据集划分 (val / test)
        config:     YOLOValConfig (可选, 仅用于审计快照)
        run:        RunContext("evaluation", sub_dir="single")
        merger:     配置溯源器 (可选)
        write_report:  是否落盘报告文件
        save_history:  是否追加评估历史
        enhanced_analysis: 是否生成增强 Markdown 分析

    Returns:
        EvalResult
    """
    from od_platform.common.report_config import log_config_report
    from od_platform.model_eval.analyzer import (
        rank_classes,
        diagnose_problem_classes,
        render_ranking_markdown,
    )
    from od_platform.model_eval.history import EvalHistory
    from od_platform.model_eval.report import EvalReport, _fmt
    from od_platform.model_eval.service import EvalResult

    infer_dir = Path(infer_dir)
    if not infer_dir.is_dir():
        return EvalResult(
            success=False, run_id=run.run_id,
            message=f"推理输出目录不存在: {infer_dir}",
        )

    logger.info("=" * 60)
    logger.info("从推理结果评估 | run_id=%s", run.run_id)
    logger.info("推理目录: %s", infer_dir)
    logger.info("数据集: %s | split=%s", data_yaml, split)

    # 1) 加载预测
    try:
        predictions = load_predictions_from_infer(infer_dir)
    except Exception as e:
        logger.exception("加载预测失败")
        return EvalResult(success=False, run_id=run.run_id, message=f"加载预测失败: {e}")

    if not predictions:
        return EvalResult(
            success=False, run_id=run.run_id,
            message="推理输出中无预测标签文件, 请确认推理时用了 --save-txt",
        )

    # 2) 加载 ground truth
    try:
        gt, class_names = load_ground_truth(data_yaml, split)
    except Exception as e:
        logger.exception("加载 ground truth 失败")
        return EvalResult(success=False, run_id=run.run_id, message=f"加载 GT 失败: {e}")

    # 3) 匹配度检查
    common = set(predictions.keys()) & set(gt.keys())
    if not common:
        logger.warning("预测 (%d 张) 与 GT (%d 张) 无交集, 无法评估",
                      len(predictions), len(gt))
        return EvalResult(
            success=False, run_id=run.run_id,
            message=f"预测与 GT 无共同图片 (预测 {len(predictions)} 张, GT {len(gt)} 张)",
        )
    only_pred = set(predictions.keys()) - set(gt.keys())
    only_gt = set(gt.keys()) - set(predictions.keys())
    if only_pred:
        logger.info("仅预测无 GT: %d 张 (%s ...)", len(only_pred),
                    ", ".join(sorted(only_pred)[:3]))
    if only_gt:
        logger.info("仅 GT 无预测: %d 张 (%s ...)", len(only_gt),
                    ", ".join(sorted(only_gt)[:3]))
    logger.info("可评估图片: %d 张", len(common))

    # 4) 计算指标
    try:
        metrics_dict = _compute_metrics(predictions, gt, class_names)
    except Exception as e:
        logger.exception("指标计算失败")
        return EvalResult(success=False, run_id=run.run_id, message=f"指标计算失败: {e}")

    # 推断模型名 (从 infer 目录的 audit 或目录名猜测)
    model_name = _infer_model_name(infer_dir)

    metrics = EvalMetrics(
        run_id=run.run_id,
        model_name=model_name,
        model_path=str(infer_dir),
        task="detect",
        split=split,
        precision=metrics_dict["precision"],
        recall=metrics_dict["recall"],
        mAP50=metrics_dict["mAP50"],
        mAP50_95=metrics_dict["mAP50_95"],
        f1=metrics_dict["f1"],
        accuracy=metrics_dict["accuracy"],
        fitness=metrics_dict["fitness"],
        speed_ms=metrics_dict["speed_ms"],
        per_class=metrics_dict["per_class"],
    )

    # 5) 构建报告
    report = EvalReport(
        run_id=run.run_id,
        model_name=model_name,
        model_path=str(infer_dir),
        data_yaml=str(data_yaml),
        split=split,
        created_at=run.created_at,
        metrics=metrics,
    )
    report.render_to_logger(logger)

    # 6) 落盘
    if write_report:
        run.run_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(run.run_dir / "report.json")
        report.write_csv(run.run_dir / "result.csv")

        md_parts = [report.render_markdown()]

        if enhanced_analysis:
            rankings = rank_classes(metrics, sort_by="mAP50_95")
            if rankings:
                md_parts.append(render_ranking_markdown(rankings, "类别性能排序 (按 mAP50-95)"))

            diagnosis = diagnose_problem_classes(metrics)
            if diagnosis["critical"] or diagnosis["warning"]:
                lines: List[str] = ["## 问题类别诊断", ""]
                if diagnosis["critical"]:
                    lines.append(f"- 🔴 **严重** (F1 极低): {', '.join(diagnosis['critical'])}")
                if diagnosis["warning"]:
                    lines.append(f"- 🟡 **关注** (F1 偏低): {', '.join(diagnosis['warning'])}")
                lines.append(f"- 🟢 **正常**: {len(diagnosis.get('good', []))} 个类别")
                lines.append("")
                md_parts.append("\n".join(lines))

        # 附加上下文信息
        md_parts.append("## 推理评估上下文")
        md_parts.append("")
        md_parts.append(f"- **推理目录**: `{infer_dir}`")
        md_parts.append(f"- **匹配图片**: {len(common)} 张")
        md_parts.append(f"- **TP / FP / FN**: {metrics_dict.get('tp', 0)} / {metrics_dict.get('fp', 0)} / {metrics_dict.get('fn', 0)}")
        md_parts.append("")
        md_parts.append("> 注: 速度指标为空 (推理已提前完成); mAP50-95 通过 10 个 IoU 阈值 (0.5~0.95) 取平均计算.")
        md_parts.append("")

        full_md = "\n".join(md_parts)
        (run.run_dir / "report.md").write_text(full_md, encoding="utf-8")
        logger.info("报告已写入 %s (report.json + result.csv + report.md)", run.run_dir)

    # 7) 评估历史
    if save_history:
        try:
            history = EvalHistory.load_or_create(str(data_yaml))
            history.add_record(report)
            history.save()
        except Exception as e:
            logger.warning("评估历史保存失败 (不阻塞主流程): %s", e)

    return EvalResult(success=True, run_id=run.run_id, report=report)


def _infer_model_name(infer_dir: Path) -> str:
    """从推理输出目录猜测模型名."""
    # 尝试读 audit.json
    audit_path = infer_dir / "odp_audit.json"
    if audit_path.exists():
        try:
            import json
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            config = audit.get("config", {})
            model = config.get("model", "")
            if model:
                return Path(model).stem
        except Exception:
            pass
    # 用目录名兜底
    return infer_dir.name


__all__ = [
    "PredBox",
    "GTBox",
    "load_predictions_from_infer",
    "load_ground_truth",
    "evaluate_infer_results",
]
