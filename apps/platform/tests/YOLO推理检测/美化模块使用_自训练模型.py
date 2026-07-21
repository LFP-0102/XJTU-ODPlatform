#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : 美化模块使用_自训练模型.py
# @Project    : XJTU-ODPlatform
# @Author     : Matri
# @Date       : 2026-07-20
# @Description: 使用自训练的安全帽检测模型 + 帧源模块 + 美化可视化模块
import sys
import cv2
from ultralytics import YOLO
from od_platform.frame_source import create_frame_source        # 帧源模块
from od_platform.visualization import BeautifyVisualizer         # 美化模块

# 使用你自己训练的模型
model = YOLO(r"C:\Users\刘赋平\Desktop\XJTU-ODPlatform\models\trained\train-20260720-172106-helmet_detection_v1-yolo26n_pt-best.pt")
viz = BeautifyVisualizer(
    labels=list(model.names.values()),          # ['hat', 'person']
    label_mapping={
        "hat": "安全帽",
        "person": "未佩戴安全帽",
    },
    color_mapping={
        "hat": (0, 255, 255),                  # BGR 黄色
        "person": (0, 0, 255),             # BGR 浅蓝
    }
)


def run(source: str) -> None:
    with create_frame_source(source) as src:
        for frame in src:
            # 检测全部 2 个类别: hat, person
            result = model(frame.image, verbose=False)[0]
            dets = BeautifyVisualizer.from_yolo_results(
                boxes=result.boxes.xyxy.cpu().numpy(),
                confidences=result.boxes.conf.cpu().numpy(),
                labels=[model.names[i] for i in result.boxes.cls.int().cpu().tolist()],
            )
            annotated = viz.draw(frame.image, dets, use_label_mapping=True)
            cv2.imshow("frame_source + visualization", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):   # q / Esc 退出
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 用法: python 美化模块使用_自训练模型.py 0         → 摄像头
    #       python 美化模块使用_自训练模型.py video.mp4 → 视频文件
    #       python 美化模块使用_自训练模型.py image.jpg → 图片
    run(sys.argv[1] if len(sys.argv) > 1 else "0")
