#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : 03_CameraFPS_Test.py
# @Description: 摄像头纯净捕获测试 (包含所有维度的 FPS 统计算法对比)

import time
import cv2


def draw_simple_ui(frame, rated_fps, display_fps):
    """绘制简单的实时UI (使用平滑后的FPS防止数字狂跳)"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Rated: {rated_fps:.0f} FPS", (10, 30), font, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    color = (0, 255, 0) if display_fps >= 25 else (0, 0, 255)
    cv2.putText(frame, f"Smooth: {display_fps:.1f} FPS", (10, 60), font, 0.8, color, 2, cv2.LINE_AA)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    rated_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Info] 摄像头已打开，驱动标称 FPS: {rated_fps}")

    # ================= 核心统计变量 =================
    frame_count = 0
    sum_read_ms = 0.0
    sum_show_ms = 0.0
    sum_instant_fps = 0.0  # 用于计算瞬时 FPS 的算术平均值

    # 滑动平均参数 (用于 UI 实时显示，让数字平滑)
    alpha = 0.1
    avg_ema_fps = 0.0
    first_frame = True

    # 记录程序开始的绝对物理时间
    global_start_time = time.perf_counter()
    prev_time = global_start_time

    try:
        print("[提示] 测试运行中... 按 'q' 或 'Esc' 退出并查看全维度统计报告。\n")
        while True:
            # 1. 统计读取耗时
            t0 = time.perf_counter()
            ret, frame = cap.read()
            read_ms = (time.perf_counter() - t0) * 1000
            if not ret:
                break

            frame_count += 1
            sum_read_ms += read_ms

            # 2. 统计显示与UI等待耗时
            t1 = time.perf_counter()

            # 计算当前帧的瞬时物理帧率
            curr_time = time.perf_counter()
            loop_dt = curr_time - prev_time
            real_time_fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
            prev_time = curr_time
            sum_instant_fps += real_time_fps

            # 计算滑动平均 FPS (EMA)
            if first_frame:
                avg_ema_fps = real_time_fps
                first_frame = False
            else:
                avg_ema_fps = alpha * real_time_fps + (1 - alpha) * avg_ema_fps

            # 绘制 UI 并显示
            draw_simple_ui(frame, rated_fps, avg_ema_fps)
            cv2.imshow("Camera FPS Test", frame)
            key = cv2.waitKey(1) & 0xFF

            show_ms = (time.perf_counter() - t1) * 1000
            sum_show_ms += show_ms

            if key in [ord("q"), 27]:
                break

    except KeyboardInterrupt:
        print("\n[Info] 检测到 Ctrl+C 中断...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ================= 最终全维度统计报告 =================
        total_elapsed_time = time.perf_counter() - global_start_time

        if frame_count > 0 and total_elapsed_time > 0:
            # 1. 基础平均耗时
            avg_read_ms = sum_read_ms / frame_count
            avg_show_ms = sum_show_ms / frame_count

            # 2. 各种维度的 FPS 计算
            overall_fps = frame_count / total_elapsed_time  # 最真实的物理全局 FPS
            avg_instant_fps = sum_instant_fps / frame_count  # 单帧瞬时 FPS 的算术平均值
            theo_read_fps = 1000.0 / avg_read_ms if avg_read_ms > 0 else 0  # 纯读取的理论极限 FPS
            theo_show_fps = 1000.0 / avg_show_ms if avg_show_ms > 0 else 0  # 纯渲染的理论极限 FPS

            print("\n" + "=" * 60)
            print("           Camera Capture Final Statistics           ")
            print("=" * 60)
            print(f"  Total Frames Shown   : {frame_count} 帧")
            print(f"  Total Time Elapsed   : {total_elapsed_time:.2f} 秒 (物理挂钟时间)")
            print("-" * 60)
            print("  [1] 耗时与理论极限 (Theoretical Max FPS):")
            print(f"      Read Time  : {avg_read_ms:5.2f} ms -> 理论极限 {theo_read_fps:5.1f} FPS")
            print(f"      Show Time  : {avg_show_ms:5.2f} ms -> 理论极限 {theo_show_fps:5.1f} FPS")
            print("-" * 60)
            print("  [2] 帧率统计算法对比 (FPS Metrics):")
            # 最终 UI 上定格的数字
            print(f"      UI Smooth FPS    : {avg_ema_fps:5.2f} FPS (画面最后显示的平滑值)")
            # 数学上的算术平均
            print(f"      Instant Avg FPS  : {avg_instant_fps:5.2f} FPS (瞬时FPS的算术平均值)")
            # 工业界唯一认可的真实指标
            print(f"      Global Avg FPS   : {overall_fps:5.2f} FPS (总帧数 / 物理总耗时) ★最真实★")
            print("=" * 60 + "\n")

            # 诊断提示
            if avg_instant_fps > overall_fps + 2.0:
                print("💡 [诊断]: 算术平均FPS 大于 真实全局FPS，说明测试过程中存在偶发的掉帧/卡顿现象。")
            else:
                print("💡 [诊断]: 帧率输出非常稳定，系统运行平滑。")
            print()
        else:
            print("\n[Warning] 未捕获到有效帧或耗时为0。\n")