"""YOLO -> YOLO 直通(detect + segment,task 无所谓 —— 两者皆可)。

源已是 YOLO 格式,不做坐标换算。"直通"不等于"来者不拒":直通无法自证源是不是请求的那种
task,故按 task 校验列数,堵住"把检测数据当分割贴出去"的静默错配(这是 detect/segment 的
【路由自洽】,不是数据质检)。detect=5 列;segment=奇数且>=7 列。
classes 在这里是【必填的命名信息】(yolo txt 不含类名)。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from od_platform.common.constants import AnnotationFormat, Task
from od_platform.data_pipeline.convert.registry import ConvertOptions, register_converter


@register_converter(AnnotationFormat.YOLO, supported_tasks=(Task.DETECT, Task.SEGMENT))
def convert_yolo(input_dir: Path, out: Path, options: ConvertOptions) -> List[str]:
    if not options.classes:
        raise ValueError("yolo 格式不含类名,必须通过 options.classes 显式提供类别顺序。")
    names = list(options.classes)
    seg = options.task == Task.SEGMENT

    out.mkdir(parents=True, exist_ok=True)
    for txt in sorted(input_dir.glob("*.txt")):
        kept: List[str] = []
        for ln in txt.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            n = len(ln.split())
            ok = (n >= 7 and n % 2 == 1) if seg else (n == 5)
            if not ok:
                raise ValueError(
                    f"{txt.name}: task={options.task} 期望"
                    f"{'奇数>=7' if seg else '5'} 列,得到 {n}: {ln!r}")
            kept.append(ln)
        (out / txt.name).write_text(
            "\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return names
