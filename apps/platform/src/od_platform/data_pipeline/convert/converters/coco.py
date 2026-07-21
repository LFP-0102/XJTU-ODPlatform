# apps/platform/src/od_platform/data_pipeline/convert/converters/coco.py  (改进版)
"""COCO -> YOLO 转换器(detect + segment,纯 Python,不依赖 ultralytics)。

支持 input_dir 下存在多个 COCO json(例如源站已切好的 train/val/test 各一份):
逐个读入并合并,输出按 image 的 file_name 落 txt。类别取所有 json 的并集,
按 category id 升序给 class_id。划分(train/val/test)交给 split 子系统,convert 不管。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter
from od_platform.data_pipeline.convert.converters._geometry import normalize_box, normalize_polygon


@register_converter(AnnotationFormat.COCO, supported_tasks=(Task.DETECT, Task.SEGMENT))
def convert_coco(input_dir: Path, out: Path, options: ConvertOptions) -> List[str]:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"COCO: 目录下没有 json 文件: {input_dir}")

    # 1) 先扫一遍所有 json:合并类别词汇表 + 统计【实际被标注用到】的类别 id
    cat: Dict[int, str] = {}
    used_ids: set = set()
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        for c in data.get("categories", []):
            cat.setdefault(c["id"], c["name"])
        for a in data.get("annotations", []):
            used_ids.add(a["category_id"])

    # 三态语义(与 VOC 一致,用 is None 区分"探测"和"空白名单"):
    #   None  -> 探测:只取【真的被标注用到】的类别,按 id 升序
    #   [...] -> 白名单:名单顺序即 class_id,名单外的类跳过
    #   []    -> 明确一个都不要
    # 探测为什么只认 used_ids:躲开 Roboflow 等导出在 categories 里塞的零标注占位类
    # (如 id=0 "objects"/supercategory)——声明了却没人用的类,不该占一个 class_id。
    if options.classes is None:
        names: List[str] = [cat[cid] for cid in sorted(cat) if cid in used_ids]
    else:
        names = list(options.classes)
    seg = options.task == Task.SEGMENT

    out.mkdir(parents=True, exist_ok=True)

    # 2) 再逐个 json 处理,输出按 image 的 file_name 落盘(合并到同一个 out)
    #    用 .get() 兜底:test.json 常常只有 images、没有 annotations,不该让它把整条流水线炸掉。
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        imgs = {im["id"]: im for im in data.get("images", [])}
        per = defaultdict(list)
        for a in data.get("annotations", []):
            per[a["image_id"]].append(a)

        for i, im in imgs.items():
            W, H = float(im["width"]), float(im["height"])
            stem = Path(im["file_name"]).stem
            lines: List[str] = []
            for a in per.get(i, []):
                nm = cat.get(a["category_id"])
                if nm is None or nm not in names:   # 未声明 / 白名单外 / 被过滤的占位类:跳过
                    continue
                cid = names.index(nm)
                if seg:
                    for poly in a.get("segmentation", []):
                        p = normalize_polygon(list(poly), W, H)
                        lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in p))
                else:
                    x, y, w, h = a["bbox"]
                    cx, cy, bw, bh = normalize_box(x, y, x + w, y + h, W, H)
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            # 无标注的图也落一个空 txt(YOLO 视作纯背景),行为可预期
            (out / (stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return names