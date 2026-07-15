from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task  # ← 改:多导入 Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter  # ← 改:多导入 ConvertOptions
from od_platform.data_pipeline.convert.converters._geometry import normalize_box, normalize_polygon  # ← 改:多导入 normalize_polygon(分割用)


@register_converter(AnnotationFormat.COCO, supported_tasks=(Task.DETECT, Task.SEGMENT))  # ← 改:声明支持 detect + segment
def convert_coco(input_dir: Path, out: Path, options: ConvertOptions) -> List[str]:  # ← 改:classes → options
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(sorted(input_dir.glob("*.json"))[0].read_text(encoding="utf-8"))
    cat = {c["id"]: c["name"] for c in data["categories"]}
    imgs = {im["id"]: im for im in data["images"]}
    per = defaultdict(list)
    for a in data["annotations"]:
        per[a["image_id"]].append(a)

    names: List[str] = list(options.classes) if options.classes else [cat[cid] for cid in sorted(cat)]
    seg = options.task == Task.SEGMENT  # ← 新增:按 task 分流

    for i, im in imgs.items():
        W, H = float(im["width"]), float(im["height"])
        stem = Path(im["file_name"]).stem
        lines: List[str] = []
        for a in per.get(i, []):
            nm = cat[a["category_id"]]
            if nm not in names:              # 白名单外的类:跳过
                continue
            cid = names.index(nm)
            if seg:  # ← 新增:分割分支(吐多边形,用上 normalize_polygon)
                for poly in a.get("segmentation", []):            # 逐条多边形
                    p = normalize_polygon(list(poly), W, H)
                    lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in p))
            else:
                x, y, w, h = a["bbox"]
                cx, cy, bw, bh = normalize_box(x, y, x + w, y + h, W, H)
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / (stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return names