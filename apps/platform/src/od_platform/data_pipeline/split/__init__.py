#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2026/7/16 10:08:50
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
if __name__ == "__main__":
    # 运行时获取实际路径信息
    import sys, os, platform

    print("\n===== 环境信息 =====")
    print(f"解释器路径: {sys.executable}")
    print(f"脚本路径: {os.path.abspath(__file__)}")
    print(f"操作系统: {platform.system()} {platform.release()}")

    run_code = 0
