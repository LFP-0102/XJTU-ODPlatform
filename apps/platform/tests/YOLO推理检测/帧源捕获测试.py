#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : 帧源捕获测试.py
# @Path       : XJTU-ODPlatfrom/apps/platform/tests/YOLO推理检测/帧源捕获测试.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 11:12:19
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:12:19 | Matri | v1.0.0 | 初始化创建
from od_platform.frame_source import create_frame_source,CameraConfig
import cv2


config = CameraConfig(fps=90, backend="msmf", camera_id=0)

font = cv2.FONT_HERSHEY_SIMPLEX
with create_frame_source(source=config) as src:
    for frame in src:
        image = frame.image
        cv2.putText(image, f"frame_index: {frame.info.frame_index}", (10, 30), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, f"frame_FPS: {frame.info.fps}", (10, 60), font, 0.6, (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.putText(image, f"frame_time: {frame.info.timestamp}", (10, 90), font, 0.6, (200, 200, 200), 1,
                    cv2.LINE_AA)
        cv2.imshow("YOLOv8 Inference", frame.image)  # 显示结果
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
    cv2.destroyAllWindows()
