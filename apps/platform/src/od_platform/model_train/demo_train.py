#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : demo_train.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_train/demo_train.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-19 09:20:39
# @Version    : v1.0.0
# @Description:
#   [模块功能简述]
#   1. 核心功能1说明。
# -----------------------------------------------------------------------------
# @ChangeLog:
#   2026-07-19 | Matri | v1.0.0 | 初始化创建
if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("../../../../../models/pretrained/yolo12n.pt")
    results = model.train(data=r"C:\Users\Matri\Desktop\XJTU-ODPlatfrom\apps\platform\configs\datasets\MRI_PASCAL.yaml",
                    epochs=2, imgsz=640)







