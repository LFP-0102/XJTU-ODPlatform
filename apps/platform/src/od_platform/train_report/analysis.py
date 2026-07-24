#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练报告分析——类别不均衡 / 小目标 / 收敛性 / 改进建议."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from od_platform.train_report.readers import EpochRow


def _fmt(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


# ============================================================
# 类别不均衡分析
# ============================================================

def analyze_class_imbalance(
    class_distribution: Dict[str, Dict[str, int]],
) -> Optional[str]:
    """检测类别不均衡, 返回警告文字或 None."""
    if not class_distribution:
        return None
    totals: Dict[str, int] = {}
    for sp_dist in class_distribution.values():
        for cls, cnt in sp_dist.items():
            totals[cls] = totals.get(cls, 0) + cnt
    if len(totals) < 2:
        return None
    max_cls = max(totals, key=totals.get)
    min_cls = min(totals, key=totals.get)
    max_cnt = totals[max_cls]
    min_cnt = totals[min_cls]
    if min_cnt == 0:
        return None
    ratio = max_cnt / min_cnt
    if ratio > 10:
        return (
            f"严重类别不均衡: `{max_cls}`({max_cnt}框) 是 `{min_cls}`({min_cnt}框) "
            f"的 {ratio:.1f} 倍。少数类样本严重不足, 模型可能难以学习其特征。"
        )
    elif ratio > 3:
        return (
            f"类别不均衡: `{max_cls}`({max_cnt}框) 是 `{min_cls}`({min_cnt}框) "
            f"的 {ratio:.1f} 倍。建议关注少数类的 Precision/Recall。"
        )
    return None


# ============================================================
# bbox 尺寸分布分析
# ============================================================

def analyze_bbox_distribution(
    bbox_size_distribution: Dict[str, Dict[str, int]],
) -> Optional[str]:
    """检测小目标占比, 返回分析文本或 None."""
    if not bbox_size_distribution:
        return None
    total = 0
    small = 0
    for d in bbox_size_distribution.values():
        small += d.get("small", 0)
        total += d.get("small", 0) + d.get("medium", 0) + d.get("large", 0)
    if total == 0:
        return None
    ratio = small / total
    if ratio > 0.7:
        return (
            f"小目标占比较高({ratio*100:.1f}%), "
            f"当前 imgsz 配置下小目标特征提取困难。建议: 增大 imgsz、使用更小 stride 的模型。"
        )
    elif ratio > 0.5:
        return (
            f"小目标占比偏高({ratio*100:.1f}%), "
            f"小目标检测可能精度不足, 可考虑增大 imgsz 或使用 multi_scale 训练。"
        )
    return None


# ============================================================
# 收敛性分析
# ============================================================

def analyze_convergence(epoch_history: List[EpochRow]) -> List[str]:
    """分析训练收敛情况: loss 下降、过拟合、mAP 平台期."""
    notes: List[str] = []
    if len(epoch_history) < 3:
        notes.append("训练轮数过少(<3 轮), 无法判断收敛性。")
        return notes

    first = epoch_history[0]
    last = epoch_history[-1]

    # train/box_loss 下降检查
    if first.train_box_loss is not None and last.train_box_loss is not None:
        change = last.train_box_loss - first.train_box_loss
        if change > 0.1:
            notes.append(
                f"train/box_loss 未有效下降(变化={_fmt(change)}), "
                f"学习率可能偏高或训练发散。"
            )

    # 过拟合检查: 后半段 val_loss 均值是否高于前半段
    mid = len(epoch_history) // 2
    early = epoch_history[:mid]
    late = epoch_history[mid:]
    early_vals = [r.val_box_loss for r in early if r.val_box_loss is not None]
    late_vals = [r.val_box_loss for r in late if r.val_box_loss is not None]
    if early_vals and late_vals:
        early_avg = sum(early_vals) / len(early_vals)
        late_avg = sum(late_vals) / len(late_vals)
        if late_avg - early_avg > 0.05:
            notes.append(
                f"可能过拟合: 验证 val/box_loss 后半段均值({_fmt(late_avg)}) "
                f"高于前半段({_fmt(early_avg)})。"
                f"建议: 加强数据增强、增大 weight_decay、或启用 dropout。"
            )

    # mAP 平台期检查
    mAPs = [(i, r.mAP50_95) for i, r in enumerate(epoch_history) if r.mAP50_95 is not None]
    if len(mAPs) >= 5:
        best_idx, best_val = max(mAPs, key=lambda x: x[1])
        if best_idx < len(epoch_history) - 10:
            notes.append(
                f"mAP50-95 在第 {best_idx + 1} 轮达到最佳({_fmt(best_val)})后 "
                f"不再提升, 训练已充分收敛。可考虑减少 epochs。"
            )

    if not notes:
        notes.append("训练过程正常收敛, 未发现异常。")
    return notes


# ============================================================
# 自动改进建议
# ============================================================

def generate_suggestions(
    *,
    class_distribution: Dict[str, Dict[str, int]],
    bbox_size_distribution: Dict[str, Dict[str, int]],
    mAP50: float,
    mAP50_95: float,
    epochs_actual: int,
    epoch_history: List[EpochRow],
) -> List[str]:
    """基于报告数据自动生成改进建议."""
    suggestions: List[str] = []

    # 类别不均衡建议
    totals: Dict[str, int] = {}
    for sp_dist in class_distribution.values():
        for cls, cnt in sp_dist.items():
            totals[cls] = totals.get(cls, 0) + cnt
    if len(totals) >= 2:
        max_cnt = max(totals.values())
        min_cnt = min(totals.values())
        if max_cnt > 0 and min_cnt > 0 and max_cnt / min_cnt > 5:
            suggestions.append(
                "类别不均衡: 使用类别加权损失(增大少数类权重)、"
                "对少数类做 Oversampling 数据增强。"
            )

    # 小目标建议
    small_total = sum(
        d.get("small", 0) for d in bbox_size_distribution.values()
    )
    all_total = sum(
        d.get(k, 0) for d in bbox_size_distribution.values()
        for k in ("small", "medium", "large")
    )
    if all_total > 0 and small_total / all_total > 0.5:
        suggestions.append(
            "小目标占比高: 增大 imgsz(如 960/1280)、开启 multi_scale 训练。"
        )

    # mAP 偏低建议
    if not math.isnan(mAP50) and mAP50 < 0.5:
        suggestions.append(
            f"mAP50 较低({_fmt(mAP50)}): 检查标注质量、增加 epochs、"
            f"或尝试更大的模型。"
        )
    if not math.isnan(mAP50_95) and mAP50_95 < 0.3:
        suggestions.append(
            f"mAP50-95 较低({_fmt(mAP50_95)}): 模型定位精度不足, "
            f"考虑增大 imgsz 或使用更精确的模型架构。"
        )

    # 训练轮数
    if epochs_actual < 20:
        suggestions.append(
            f"训练轮数较少({epochs_actual}), 如需更充分收敛建议增加 epochs。"
        )

    # 过拟合检查
    if len(epoch_history) > 10:
        mid = len(epoch_history) // 2
        early = [r for r in epoch_history[:mid] if r.val_box_loss is not None]
        late = [r for r in epoch_history[mid:] if r.val_box_loss is not None]
        if early and late:
            early_avg = sum(r.val_box_loss for r in early) / len(early)  # type: ignore[arg-type]
            late_avg = sum(r.val_box_loss for r in late) / len(late)  # type: ignore[arg-type]
            if late_avg - early_avg > 0.05:
                suggestions.append(
                    "过拟合风险: 加强数据增强(mosaic/mixup)、增大 weight_decay、"
                    "或减少 epochs。"
                )

    if not suggestions:
        suggestions.append(
            "当前训练配置和结果处于合理范围。"
            "常用优化方向: 数据增强调参、模型缩放(n/s/m/l/x)、超参数搜索。"
        )
    return suggestions
