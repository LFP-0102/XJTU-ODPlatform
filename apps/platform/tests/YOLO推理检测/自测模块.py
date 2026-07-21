#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : 帧源捕获测速.py
# @Path       : XJTU-ODPlatfrom/apps/platform/tests/YOLO推理检测/帧源捕获测速.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20
# @Version    : v1.0.0
# @Description: 测量 frame_source 模块的实际采集/显示速度

import time
import threading
from typing import Optional

import cv2
import numpy as np

from od_platform.frame_source import create_frame_source, CameraConfig

# ==================== 配置 ====================
CAMERA_ID = 0
TARGET_FPS = 90
WIDTH, HEIGHT = 1280, 720
BACKEND = "msmf"

WINDOW_NAME = "FrameSource Benchmark"
MAX_DISPLAY_FPS = 0          # 0=满速, >0=限速(丢帧实现)
DURATION_SEC = 0             # 0=手动按q停止, >0=自动运行N秒后停止

# ==================== 共享状态 ====================
cond = threading.Condition()
latest_frame: Optional[np.ndarray] = None
frame_seq = 0

stop_event = threading.Event()
ready_event = threading.Event()

capture_count = 0
capture_drop = 0
display_count = 0
display_skip = 0

stats = {"cap_start": 0.0, "cap_end": 0.0, "disp_start": 0.0, "disp_end": 0.0}
imshow_cost = 0.0

# 额外：记录帧源模块自身上报的 fps / timestamp
last_src_fps = 0.0
last_src_ts = 0.0
src_fps_samples: list = []   # 收集模块自报fps用于统计


def capture_thread():
    """生产者：从 frame_source 迭代器拉帧"""
    global capture_count, capture_drop, latest_frame, frame_seq
    global last_src_fps, last_src_ts

    config = CameraConfig(
        fps=TARGET_FPS,
        backend=BACKEND,
        camera_id=CAMERA_ID,
        # 如果你的 CameraConfig 支持 width/height，取消下面注释：
        width=WIDTH,
        height=HEIGHT,
    )

    try:
        with create_frame_source(source=config) as src:
            ready_event.set()
            stats["cap_start"] = time.perf_counter()

            for frame in src:
                if stop_event.is_set():
                    break

                capture_count += 1
                image = frame.image

                # 记录模块自报信息
                if frame.info is not None:
                    last_src_fps = frame.info.fps
                    last_src_ts = frame.info.timestamp
                    if frame.info.fps > 0:
                        src_fps_samples.append(frame.info.fps)

                with cond:
                    latest_frame = image
                    frame_seq += 1
                    cond.notify()

    except Exception as e:
        print(f"[Capture] 异常: {e}")
    finally:
        stats["cap_end"] = time.perf_counter()
        ready_event.set()          # 防止主线程卡死
        with cond:
            cond.notify_all()


def main():
    global display_count, display_skip, imshow_cost

    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()
    ready_event.wait(timeout=10)

    if stop_event.is_set():
        print("[Error] 采集线程启动失败")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    min_interval = (1.0 / MAX_DISPLAY_FPS) if MAX_DISPLAY_FPS > 0 else 0.0
    last_show = 0.0
    run_start = time.perf_counter()

    with cond:
        last_seq = frame_seq

    stats["disp_start"] = time.perf_counter()
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        while not stop_event.is_set():
            # 自动限时
            if DURATION_SEC > 0 and (time.perf_counter() - run_start) > DURATION_SEC:
                stop_event.set()
                break

            # ---- 1) 等新帧 ----
            with cond:
                while frame_seq == last_seq and not stop_event.is_set():
                    cond.wait(0.05)
                if latest_frame is None:
                    continue
                frame = latest_frame
                display_skip += frame_seq - last_seq - 1
                last_seq = frame_seq

            # ---- 2) 可选限速 ----
            now = time.perf_counter()
            if min_interval and now - last_show < min_interval:
                continue
            last_show = now

            # ---- 3) 绘制信息 & 显示 ----
            display_count += 1
            elapsed = now - stats["disp_start"]
            realtime_cap_fps = capture_count / max(elapsed, 0.001)
            realtime_disp_fps = display_count / max(elapsed, 0.001)

            cv2.putText(frame, f"Cap: {capture_count}  FPS: {realtime_cap_fps:.1f}",
                        (10, 30), font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Disp: {display_count}  FPS: {realtime_disp_fps:.1f}  Skip: {display_skip}",
                        (10, 60), font, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"SrcFPS: {last_src_fps:.1f}  Ts: {last_src_ts:.3f}",
                        (10, 90), font, 0.7, (200, 200, 0), 2)

            t0 = time.perf_counter()
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.pollKey()
            imshow_cost += time.perf_counter() - t0

            if key != -1 and (key & 0xFF) == ord('q'):
                stop_event.set()

    except KeyboardInterrupt:
        print("\n[Info] Ctrl+C 中断")
        stop_event.set()
    finally:
        stop_event.set()
        with cond:
            cond.notify_all()
        t.join(timeout=3)
        cv2.destroyAllWindows()
        stats["disp_end"] = time.perf_counter()
        print_report()


def print_report():
    cap_elapsed = max(stats["cap_end"] - stats["cap_start"], 0.001)
    disp_elapsed = max(stats["disp_end"] - stats["disp_start"], 0.001)
    cap_fps = capture_count / cap_elapsed
    disp_fps = display_count / disp_elapsed
    avg_imshow = (imshow_cost / display_count * 1000.0) if display_count else 0.0
    ceil_fps = (1000.0 / avg_imshow) if avg_imshow > 0 else float('inf')

    # 模块自报 fps 统计
    if src_fps_samples:
        src_avg = sum(src_fps_samples) / len(src_fps_samples)
        src_min = min(src_fps_samples)
        src_max = max(src_fps_samples)
    else:
        src_avg = src_min = src_max = 0.0

    print("\n" + "=" * 62)
    print("     FrameSource Module Benchmark Report")
    print("=" * 62)
    print(f" 配置: camera={CAMERA_ID}  target={TARGET_FPS}fps  "
          f"res={WIDTH}x{HEIGHT}  backend={BACKEND}")
    print("-" * 62)
    print(" [1] 采集 (frame_source 迭代器)")
    print(f"     运行时长        : {cap_elapsed:.2f} s")
    print(f"     总帧数          : {capture_count}")
    print(f"     采集失败/丢帧   : {capture_drop}")
    print(f"     ★ 实测采集 FPS  : {cap_fps:.2f}")
    print(f"     模块自报 FPS    : avg={src_avg:.1f}  min={src_min:.1f}  max={src_max:.1f}")
    print("-" * 62)
    print(" [2] 显示 (事件驱动)")
    print(f"     运行时长        : {disp_elapsed:.2f} s")
    print(f"     显示帧数        : {display_count}")
    print(f"     跳过帧数        : {display_skip}")
    print(f"     ★ 实测显示 FPS  : {disp_fps:.2f}")
    print("-" * 62)
    print(" [3] 渲染瓶颈")
    print(f"     imshow+pollKey  : {avg_imshow:.2f} ms/帧")
    print(f"     渲染理论上限    : {ceil_fps:.1f} FPS")
    print("-" * 62)
    print(" [4] 结论")
    if cap_fps >= TARGET_FPS * 0.9:
        print(f"     ✅ 采集达标 ({cap_fps:.1f} >= {TARGET_FPS*0.9:.0f})")
    else:
        print(f"     ⚠️  采集未达标 ({cap_fps:.1f} < {TARGET_FPS*0.9:.0f})")
    if disp_fps >= cap_fps * 0.95:
        print(f"     ✅ 显示跟得上采集")
    else:
        print(f"     ⚠️  显示是瓶颈 (disp={disp_fps:.1f} < cap={cap_fps:.1f})")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()