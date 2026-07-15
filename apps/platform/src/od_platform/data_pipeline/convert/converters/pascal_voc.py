from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat,Task
from od_platform.data_pipeline.convert.registry import register_converter,ConvertOptions
from od_platform.data_pipeline.convert.converters._geometry import normalize_box


@register_converter(AnnotationFormat.PASCAL_VOC, supported_tasks=(Task.DETECT,))
def convert_voc(input_dir: Path, out: Path, options: ConvertOptions)-> List[str]:
    out.mkdir(parents=True, exist_ok=True)
    names: List[str] = list(options.classes) if options.classes else []
    discovering = options.classes is None
    for xp in sorted(input_dir.glob("*.xml")):
        root = ET.parse(xp).getroot()
        size = root.find("size")
        W, H = float(size.findtext("width")), float(size.findtext("height"))
        lines: List[str] = []
        for obj in root.findall("object"):
            nm = obj.findtext("name")
            if nm not in names:
                if discovering:
                    names.append(nm)
                else:
                    continue
            b = obj.find("bndbox")
            cx, cy, bw, bh = normalize_box(
                float(b.findtext("xmin")), float(b.findtext("ymin")),
                float(b.findtext("xmax")), float(b.findtext("ymax")), W, H)
            lines.append(f"{names.index(nm)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / (xp.stem + ".txt")).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return names
