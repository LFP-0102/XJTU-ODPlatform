#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : 02_Opencv捕获输入.py
# @Path       : XJTU-ODPlatfrom/apps/platform/tests/YOLO推理检测/02_Opencv捕获输入.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-19 15:29:23
# @Version    : v1.3.0
# @Description: YOLOv8 实时推理 + 实时面板 + 退出时打印最终平均性能统计
# @ChangeLog:
#   2026-07-19:29:23 | Matri | v1.0.0 | 初始化创建
#   2026-07-19:30:00 | Matri | v1.1.0 | 添加各步骤耗时与FPS显示
#   2026-07-19:45:00 | Matri | v1.2.0 | 修复单步FPS逻辑漏洞，改用工业界标准的 ms + 真实 FPS 监控
#   2026-07-19:50:00 | Matri | v1.3.0 | 增加退出时的最终算术平均统计面板，支持 Ctrl+C 优雅退出

import time
import cv2
from ultralytics import YOLO


def put_performance_panel(frame, step_times, overall_fps):
    """绘制实时的性能监控面板"""
    x, y = 10, 30
    line_h = 26
    font = cv2.FONT_HERSHEY_SIMPLEX

    panel_h = line_h * (len(step_times) + 2) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - 25), (x + 260, y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, "Profiling Monitor", (x, y), font, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cy = y + line_h + 5

    for label, ms in step_times.items():
        text = f"{label:<12s}: {ms:6.2f} ms"
        color = (0, 200, 255) if ms < 10.0 else (0, 165, 255) if ms < 30.0 else (0, 0, 255)
        cv2.putText(frame, text, (x, cy), font, 0.55, color, 1, cv2.LINE_AA)
        cy += line_h

    cy += 5
    fps_text = f"Overall FPS  : {overall_fps:5.1f}"
    if overall_fps >= 30:
        fps_color = (0, 255, 0)
    elif overall_fps >= 15:
        fps_color = (0, 255, 255)
    else:
        fps_color = (0, 0, 255)

    cv2.putText(frame, fps_text, (x, cy), font, 0.7, fps_color, 2, cv2.LINE_AA)


if __name__ == "__main__":
    model = YOLO("train3-20251223-112728-yolov8n-best.pt")
    cap = cv2.VideoCapture(0)

    # ---------- 实时显示用的滑动平均参数 (EMA) ----------
    alpha = 0.15
    avg_read_ms, avg_infer_ms, avg_plot_ms, avg_total_ms, avg_overall_fps = 0.0, 0.0, 0.0, 0.0, 0.0

    # ---------- 最终打印用的算术平均累加器 ----------
    frame_count = 0
    sum_read_ms = 0.0
    sum_infer_ms = 0.0
    sum_plot_ms = 0.0
    sum_loop_time = 0.0  # 累加真实的循环总耗时(秒)

    first_frame = True
    prev_loop_time = time.perf_counter()

    try:
        while True:
            curr_loop_time = time.perf_counter()

            # 1. 读取帧 (Read)
            t0 = time.perf_counter()
            ret, frame = cap.read()
            read_ms = (time.perf_counter() - t0) * 1000
            if not ret:
                break

            # 2. 模型推理 (Infer)
            t0 = time.perf_counter()
            results = model(source=frame, verbose=False)
            infer_ms = (time.perf_counter() - t0) * 1000

            # 3. 绘制结果 (Plot)
            t0 = time.perf_counter()
            annotated_frame = results[0].plot()
            plot_ms = (time.perf_counter() - t0) * 1000

            algo_total_ms = read_ms + infer_ms + plot_ms

            # 4. 计算真实的整体 FPS
            loop_dt = curr_loop_time - prev_loop_time
            true_fps = 1.0 / loop_dt if loop_dt > 0 else 0.0
            prev_loop_time = curr_loop_time

            # ---------- 累加统计信息 (用于最终打印) ----------
            frame_count += 1
            sum_read_ms += read_ms
            sum_infer_ms += infer_ms
            sum_plot_ms += plot_ms
            sum_loop_time += loop_dt

            # ---------- 滑动平均处理 (用于实时UI) ----------
            if first_frame:
                avg_read_ms, avg_infer_ms, avg_plot_ms, avg_total_ms = read_ms, infer_ms, plot_ms, algo_total_ms
                avg_overall_fps = true_fps
                first_frame = False
            else:
                avg_read_ms = alpha * read_ms + (1 - alpha) * avg_read_ms
                avg_infer_ms = alpha * infer_ms + (1 - alpha) * avg_infer_ms
                avg_plot_ms = alpha * plot_ms + (1 - alpha) * avg_plot_ms
                avg_total_ms = alpha * algo_total_ms + (1 - alpha) * avg_total_ms
                avg_overall_fps = alpha * true_fps + (1 - alpha) * avg_overall_fps

            # ---------- 实时 UI 显示 ----------
            step_times = {
                "Read": avg_read_ms,
                "Infer": avg_infer_ms,
                "Plot": avg_plot_ms,
                "Algo Total": avg_total_ms
            }
            put_performance_panel(annotated_frame, step_times, avg_overall_fps)
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        print("\n[Info] 检测到 Ctrl+C，正在生成统计报告...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # ================= 最终统计打印 =================
        if frame_count > 0:
            # 计算算术平均值
            final_avg_read = sum_read_ms / frame_count
            final_avg_infer = sum_infer_ms / frame_count
            final_avg_plot = sum_plot_ms / frame_count
            final_avg_algo = final_avg_read + final_avg_infer + final_avg_plot

            # 计算最真实的总平均 FPS (总帧数 / 总耗时秒数)
            final_overall_fps = frame_count / sum_loop_time if sum_loop_time > 0 else 0.0

            print("\n" + "=" * 55)
            print("             Performance Summary (Final)             ")
            print("=" * 55)
            print(f"Total Frames Processed : {frame_count}")
            print(f"Overall Average FPS    : {final_overall_fps:.2f} FPS (真实系统帧率)")
            print("-" * 55)
            print("Step-wise Average Time & Theoretical Max FPS:")
            print(
                f"  Read       : {final_avg_read:6.2f} ms  (理论极限: {1000 / final_avg_read if final_avg_read > 0 else 0:>8.1f} FPS)")
            print(
                f"  Infer      : {final_avg_infer:6.2f} ms  (理论极限: {1000 / final_avg_infer if final_avg_infer > 0 else 0:>8.1f} FPS)")
            print(
                f"  Plot       : {final_avg_plot:6.2f} ms  (理论极限: {1000 / final_avg_plot if final_avg_plot > 0 else 0:>8.1f} FPS)")
            print(
                f"  Algo Total : {final_avg_algo:6.2f} ms  (理论极限: {1000 / final_avg_algo if final_avg_algo > 0 else 0:>8.1f} FPS)")
            print("=" * 55 + "\n")
        else:
            print("\n[Warning] 未处理任何帧，无法生成统计报告。\n")