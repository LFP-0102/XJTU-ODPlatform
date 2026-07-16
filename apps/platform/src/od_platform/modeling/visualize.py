"""可视化模块 —— 训练曲线 + 预测结果对比（预测 vs 真值）。"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext
from od_platform.data_pipeline.split.manifest import SplitManifest

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERR = 1


# ── 子命令分发 ──────────────────────────────────────────────
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="odp-visualize", description="训练曲线 / 预测对比 可视化。")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- curves ----
    p_curves = sub.add_parser("curves", help="绘制训练 loss + 指标曲线")
    p_curves.add_argument("--results-csv", required=True, help="results.csv 路径")
    p_curves.add_argument("--output", default=None, help="输出图片路径（默认同目录）")

    # ---- samples ----
    p_samples = sub.add_parser("samples", help="预测 vs 真值 对比图")
    p_samples.add_argument("--model", required=True, help="模型权重路径 (.pt)")
    p_samples.add_argument("--dataset", required=True, help="数据集名")
    p_samples.add_argument("--split", default="val", choices=("train", "val", "test"))
    p_samples.add_argument("--n", type=int, default=6, help="展示图片数量")
    p_samples.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p_samples.add_argument("--device", default="0")
    p_samples.add_argument("--output", default=None, help="输出图片路径（默认 runs/ 下）")

    # ---- compare ----
    p_comp = sub.add_parser("compare", help="两个模型在相同图片上对比（基准 vs 训练后）")
    p_comp.add_argument("--base-model", required=True, help="基准模型路径 (.pt)")
    p_comp.add_argument("--trained-model", required=True, help="训练后模型路径 (.pt)")
    p_comp.add_argument("--dataset", required=True, help="数据集名")
    p_comp.add_argument("--split", default="test", choices=("train", "val", "test"))
    p_comp.add_argument("--n", type=int, default=6, help="展示图片数量")
    p_comp.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p_comp.add_argument("--device", default="0")
    p_comp.add_argument("--output", default=None, help="输出图片路径")

    a = p.parse_args(argv)

    get_logger(base_path=paths.LOGGING_DIR, log_type="visualize")

    try:
        if a.cmd == "curves":
            _plot_curves(a)
        elif a.cmd == "samples":
            _plot_samples(a)
        elif a.cmd == "compare":
            _plot_compare(a)
    except Exception as e:
        logger.exception("可视化失败: %s", e)
        return EXIT_ERR
    return EXIT_OK


# ── 训练曲线 ─────────────────────────────────────────────────
def _plot_curves(a) -> None:
    csv_path = Path(a.results_csv)
    if not csv_path.exists():
        logger.error("找不到 results.csv: %s", csv_path)
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    for col, label, c in [
        ("train/box_loss", "train box", "#1f77b4"),
        ("train/cls_loss", "train cls", "#ff7f0e"),
        ("train/dfl_loss", "train dfl", "#2ca02c"),
        ("val/box_loss", "val box", "#d62728"),
        ("val/cls_loss", "val cls", "#9467bd"),
        ("val/dfl_loss", "val dfl", "#8c564b"),
    ]:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=label, color=c, alpha=0.85)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize="small")

    # mAP
    ax = axes[1]
    for col, label, c in [
        ("metrics/mAP50(B)", "mAP@50", "#1f77b4"),
        ("metrics/mAP50-95(B)", "mAP@50-95", "#ff7f0e"),
    ]:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=label, color=c, linewidth=2)
    ax.set_title("mAP")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    # Precision / Recall
    ax = axes[2]
    for col, label, c in [
        ("metrics/precision(B)", "Precision", "#1f77b4"),
        ("metrics/recall(B)", "Recall", "#d62728"),
    ]:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=label, color=c, linewidth=2)
    ax.set_title("Precision & Recall")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves — {csv_path.parent.name}", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out = Path(a.output) if a.output else csv_path.with_suffix(".png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 训练曲线已保存: {out}")


# ── 预测 vs 真值 对比 ──────────────────────────────────────────
def _plot_samples(a) -> None:
    from ultralytics import YOLO

    yaml_path = paths.dataset_yaml_path(a.dataset)
    if not yaml_path.exists():
        logger.error("找不到数据集配置: %s", yaml_path)
        raise FileNotFoundError(yaml_path)

    import yaml as _yaml
    ds_cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    class_names: dict = ds_cfg.get("names", {})
    processed_root = Path(ds_cfg["path"])

    # 读 manifest 取 split → stem 映射
    odp_meta = ds_cfg.get("odp_meta", {})
    manifest_ref = odp_meta.get("manifest_path")
    if manifest_ref:
        manifest = SplitManifest.read(paths.ROOT_DIR / manifest_ref)
        stems = manifest.stems_of(a.split)
    else:
        # 回退：直接读 processed 目录
        img_dir = processed_root / "images" / a.split
        stems = sorted(p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    if len(stems) == 0:
        logger.error("split=%s 无样本", a.split)
        raise ValueError(f"无样本: split={a.split}")

    # 随机挑 n 个
    rng = torch.Generator().manual_seed(42)
    chosen = torch.randperm(len(stems), generator=rng)[: a.n].tolist()
    chosen_stems = [stems[i] for i in chosen]

    # 推理
    model = YOLO(a.model)
    img_dir = processed_root / "images" / a.split
    lbl_dir = processed_root / "labels" / a.split

    n = len(chosen_stems)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, "__iter__") else [row])]

    for idx, stem in enumerate(chosen_stems):
        ax = axes_flat[idx]
        # 找图片
        hits = sorted(img_dir.glob(f"{stem}.*"))
        if not hits:
            ax.set_title(f"{stem} — 无图片")
            ax.axis("off")
            continue
        img = Image.open(hits[0])
        ax.imshow(img)

        # 真值（绿色虚线）
        lbl_file = lbl_dir / f"{stem}.txt"
        if lbl_file.exists():
            h, w = img.height, img.width
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid, cx, cy, bw, bh = map(float, parts[:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                name = class_names.get(int(cid), f"cls{cid}")
                rect = plt.Rectangle((x1, y1), bw * w, bh * h, fill=False,
                                     edgecolor="lime", linewidth=2, linestyle="--")
                ax.add_patch(rect)
                ax.text(x1, y1 - 4, f"GT:{name}", fontsize=7, color="lime",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1))

        # 预测（红色实线）
        results = model(hits[0], conf=a.conf, device=a.device, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = class_names.get(cls_id, f"cls{cls_id}")
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                 edgecolor="red", linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 4, f"{name} {conf:.2f}", fontsize=7, color="white",
                    bbox=dict(facecolor="red", alpha=0.7, pad=1))

        ax.set_title(stem, fontsize=9)
        ax.axis("off")

    # 隐藏多余的子图
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Predictions (red) vs Ground Truth (green) — {a.dataset}/{a.split}  n={n}",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()

    out = Path(a.output) if a.output else (
        paths.ROOT_DIR / "runs" / "visualize" / f"comparison_{a.dataset}_{a.split}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 对比图已保存: {out}")


# ── 基准模型 vs 训练后模型 对比 ──────────────────────────────────
def _plot_compare(a) -> None:
    from ultralytics import YOLO

    yaml_path = paths.dataset_yaml_path(a.dataset)
    if not yaml_path.exists():
        logger.error("找不到数据集配置: %s", yaml_path)
        raise FileNotFoundError(yaml_path)

    import yaml as _yaml
    ds_cfg = _yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    class_names: dict = ds_cfg.get("names", {})
    processed_root = Path(ds_cfg["path"])

    # 读 manifest
    odp_meta = ds_cfg.get("odp_meta", {})
    manifest_ref = odp_meta.get("manifest_path")
    if manifest_ref:
        manifest = SplitManifest.read(paths.ROOT_DIR / manifest_ref)
        stems = manifest.stems_of(a.split)
    else:
        img_dir = processed_root / "images" / a.split
        stems = sorted(p.stem for p in img_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})

    if len(stems) == 0:
        logger.error("split=%s 无样本", a.split)
        raise ValueError(f"无样本: split={a.split}")

    rng = torch.Generator().manual_seed(42)
    chosen = torch.randperm(len(stems), generator=rng)[: a.n].tolist()
    chosen_stems = [stems[i] for i in chosen]

    base_model = YOLO(a.base_model)
    trained_model = YOLO(a.trained_model)
    img_dir = processed_root / "images" / a.split
    lbl_dir = processed_root / "labels" / a.split

    n = len(chosen_stems)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = [axes]

    col_labels = ["Baseline (yolo12n)", "Trained (helmet)"]

    for row_idx, stem in enumerate(chosen_stems):
        hits = sorted(img_dir.glob(f"{stem}.*"))
        if not hits:
            for col in range(2):
                axes[row_idx][col].set_title(f"{stem} — 无图片")
                axes[row_idx][col].axis("off")
            continue

        img = Image.open(hits[0])
        h, w = img.height, img.width

        # 真值框（绿色虚线）—— 左右都画
        gt_boxes = []
        lbl_file = lbl_dir / f"{stem}.txt"
        if lbl_file.exists():
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid, cx, cy, bw, bh = map(float, parts[:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                gt_boxes.append((x1, y1, bw * w, bh * h, int(cid)))

        # 两个模型分别推理
        for col, (model, label) in enumerate(zip([base_model, trained_model], col_labels)):
            ax = axes[row_idx][col]
            ax.imshow(img)

            # 真值
            for x1, y1, bw, bh, cid in gt_boxes:
                name = class_names.get(cid, f"cls{cid}")
                rect = plt.Rectangle((x1, y1), bw, bh, fill=False,
                                     edgecolor="lime", linewidth=2, linestyle="--")
                ax.add_patch(rect)
                ax.text(x1, y1 - 4, f"GT:{name}", fontsize=7, color="lime",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1))

            # 预测
            results = model(hits[0], conf=a.conf, device=a.device, verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = class_names.get(cls_id, f"cls{cls_id}")
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                     edgecolor="red", linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1 - 4, f"{name} {conf:.2f}", fontsize=7, color="white",
                        bbox=dict(facecolor="red", alpha=0.7, pad=1))

            ax.set_title(f"{stem}  [{label}]", fontsize=10, fontweight="bold")
            ax.axis("off")

    fig.suptitle(
        f"Baseline vs Fine-tuned — {a.dataset}/{a.split}  (green=GT, red=prediction)",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()

    out = Path(a.output) if a.output else (
        paths.ROOT_DIR / "runs" / "visualize" / f"compare_{a.dataset}_{a.split}.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 模型对比图已保存: {out}")


if __name__ == "__main__":
    sys.exit(main())
