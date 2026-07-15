from __future__ import annotations

from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task  # ← 改:多导入 Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter  # ← 改:多导入 ConvertOptions


@register_converter(AnnotationFormat.YOLO, supported_tasks=(Task.DETECT, Task.SEGMENT))  # ← 改:声明支持 detect + segment
def convert_yolo(input_dir: Path, out: Path, options: ConvertOptions) -> List[str]:  # ← 改:classes → options
    if not options.classes:                     # 轴A:命名信息必填,第一行 fail-fast(无副作用)
        raise ValueError("yolo 格式不含类名,必须通过 options.classes 显式提供类别顺序。")
    names = list(options.classes)
    seg = options.task == Task.SEGMENT  # ← 新增:按 task 定列数规则

    out.mkdir(parents=True, exist_ok=True)
    for txt in sorted(input_dir.glob("*.txt")):
        kept: List[str] = []
        for ln in txt.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            n = len(ln.split())
            ok = (n >= 7 and n % 2 == 1) if seg else (n == 5)  # ← 新增:列数即路由自洽校验(直通版没有)
            if not ok:                          # detect/segment 路由自洽(非数据质检)
                raise ValueError(
                    f"{txt.name}: task={options.task} 期望"
                    f"{'奇数>=7' if seg else '5'} 列,得到 {n}: {ln!r}")
            kept.append(ln)
        (out / txt.name).write_text(
            "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return names