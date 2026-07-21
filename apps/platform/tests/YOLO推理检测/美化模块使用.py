#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : 美化模块使用.py
# @Path       : XJTU-ODPlatfrom/apps/platform/tests/YOLO推理检测/美化模块使用.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 14:36:36
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:36:36 | Matri | v1.0.0 | 初始化创建
import sys
import cv2
from ultralytics import YOLO
from od_platform.frame_source import create_frame_source        # 帧源模块
from od_platform.visualization import BeautifyVisualizer         # 美化模块

model = YOLO("train3-20251223-112728-yolov8n-best.pt")
viz = BeautifyVisualizer(
    labels=list(model.names.values()),
    label_mapping={"person": "人员", "head": "未佩戴安全帽","ordinary_clothes": "普通衣物",
                "reflective_vest":"反光衣","safety_helmet":"安全帽"},
    color_mapping={
        "person": (255, 218, 114),  # 颜色顺序是BGR
        "head": (0, 0, 255),
        "ordinary_clothes": (0, 255, 0),
        "reflective_vest": (255, 255, 0),
        "safety_helmet": (0, 255, 255)
    }
)

def run(source: str) -> None:
    with create_frame_source(source) as src:          # ← 帧源:换字符串即换源
        for frame in src:
            result = model(frame.image, verbose=False, classes = [0])[0]
            dets = BeautifyVisualizer.from_yolo_results(   # ← 衔接点:打包
                boxes=result.boxes.xyxy.cpu().numpy(),
                confidences=result.boxes.conf.cpu().numpy(),
                labels=[model.names[i] for i in result.boxes.cls.int().cpu().tolist()],
            )
            annotated = viz.draw(frame.image, dets, use_label_mapping=True)  # ← 美化
            cv2.imshow("frame_source + visualization", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):   # q / Esc 退出
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "0")    # 默认摄像头