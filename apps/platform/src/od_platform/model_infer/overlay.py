#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : overlay.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/model_infer/overlay.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-21 09:48:52
# @Version    : v1.0.0
# @Description: 主要负责画面上帧率信息显示
# @ChangeLog:
#   2026-07-21:48:52 | Matri | v1.0.0 | 初始化创建
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : overlay.py
# @Project   : ODPlatform
# @Function  : 画面信息面板(HUD) + 多维帧率采集 — 实时/顺序引擎共用
"""推理画面信息叠加 + 帧率指标.

提供:
  - FPSCounter : 滑动窗口 FPS, 带瞬时值 (当前画面帧率)
  - Metrics    : 一次推理的多维帧率 (捕获/推理/渲染/loop) + 模型 speed 分解, 可快照
  - draw_hud   : 在画面左上角画一个半透明信息面板 (开关 show_info 控制)
  - draw_pause : 暂停时的居中提示层

帧率口径 (写论文/快照都用得上):
  - capture(捕获) : 帧源读帧的速率
  - infer(推理)   : 模型推理速率, 取自 ultralytics result.speed (纯模型, 不含读帧/绘制)
  - render(渲染)  : 美化绘制 (BeautifyVisualizer.draw) 的速率
  - loop(端到端)  : 整个主循环一帧的速率 (批级测量, 均摊到帧 — 反映真实吞吐)
  - current(当前) : loop 的瞬时值 (最近一个样本)

★ loop 测量纪律 (在 pipeline.py 的主循环里兑现, 详见阶段 7.9):
  - 测量点: 每批派发完时算一次 (batch_end - last_batch_end) / batch_size
  - 这个值 update batch_size 次填窗口
  - 不在"批内 for 每一帧"测 —— 会得到几微秒的污染值
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


class FPSCounter:
    """滑动窗口 FPS (按每帧耗时 ms 估算), 同时保留最近一帧的瞬时值."""

    def __init__(self, window_size: int = 30) -> None:
        self._samples: deque[float] = deque(maxlen=window_size)
        self._last_ms: float = 0.0

    def update(self, ms: float) -> None:
        if ms > 0:
            self._samples.append(ms)
            self._last_ms = ms

    @property
    def fps(self) -> float:
        """滑动平均 FPS."""
        if not self._samples:
            return 0.0
        avg = sum(self._samples) / len(self._samples)
        return 1000.0 / avg if avg > 0 else 0.0

    @property
    def inst(self) -> float:
        """瞬时 FPS (最近一帧)."""
        return 1000.0 / self._last_ms if self._last_ms > 0 else 0.0


@dataclass
class Metrics:
    """一次推理的多维帧率 + 模型 speed 分解."""
    capture: FPSCounter = field(default_factory=FPSCounter)
    infer:   FPSCounter = field(default_factory=FPSCounter)
    render:  FPSCounter = field(default_factory=FPSCounter)
    loop:    FPSCounter = field(default_factory=FPSCounter)

    # ultralytics result.speed 累计 (ms), 用于平均出 preprocess/inference/postprocess
    _spp: float = 0.0
    _sinf: float = 0.0
    _spost: float = 0.0
    _sn: int = 0

    def add_speed(self, speed: dict | None) -> None:
        """喂入 ultralytics result.speed (单位 ms). 同时更新 infer 帧率."""
        if not speed:
            return
        self._spp += float(speed.get("preprocess", 0.0))
        self._sinf += float(speed.get("inference", 0.0))
        self._spost += float(speed.get("postprocess", 0.0))
        self._sn += 1
        if speed.get("inference"):
            self.infer.update(float(speed["inference"]))

    def speed_avg_ms(self) -> dict[str, float]:
        if not self._sn:
            return {}
        return {
            "preprocess":  round(self._spp / self._sn, 2),
            "inference":   round(self._sinf / self._sn, 2),
            "postprocess": round(self._spost / self._sn, 2),
        }

    def snapshot(self) -> dict[str, Any]:
        """给日志 / audit 快照用的纯数据 (论文里可直接引用)."""
        return {
            "capture_fps": round(self.capture.fps, 1),
            "infer_fps":   round(self.infer.fps, 1),
            "render_fps":  round(self.render.fps, 1),
            "loop_fps":    round(self.loop.fps, 1),
            "current_fps": round(self.loop.inst, 1),
            "speed_ms":    self.speed_avg_ms(),
        }


# ============================================================================
# 画面叠加
# ============================================================================
# 颜色统一 BGR
_C_LABEL = (200, 200, 200)
_C_CAPTURE = (0, 200, 255)     # 橙黄
_C_INFER = (0, 230, 0)         # 绿
_C_RENDER = (255, 120, 220)    # 粉紫
_C_LOOP = (255, 200, 0)        # 青蓝
_C_CURRENT = (255, 255, 255)   # 白
_C_OBJ = (120, 220, 255)


def draw_hud(
    frame,
    metrics: Metrics,
    *,
    n_dets: int = 0,
    recording: bool = False,
    show_info: bool = True,
) -> None:
    """左上角半透明信息面板. show_info=False 则不画.

    直接在传入的 frame (已是绘制后的副本) 上原地叠加.
    """
    import cv2

    if not show_info:
        return

    rows = [
        ("Capture", f"{metrics.capture.fps:5.1f} FPS", _C_CAPTURE),
        ("Infer",   f"{metrics.infer.fps:5.1f} FPS", _C_INFER),
        ("Render",  f"{metrics.render.fps:5.1f} FPS", _C_RENDER),
        ("Loop",    f"{metrics.loop.fps:5.1f} FPS", _C_LOOP),
        ("Current", f"{metrics.loop.inst:5.1f} FPS", _C_CURRENT),
        ("Objects", f"{n_dets}", _C_OBJ),
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th, lh = 0.5, 1, 22
    pad = 10
    label_w = 84
    panel_w = 230
    panel_h = pad * 2 + lh * len(rows)
    x0, y0 = 12, 12

    # 半透明深色面板
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    # 左侧彩色装饰条
    cv2.rectangle(frame, (x0, y0), (x0 + 4, y0 + panel_h), (0, 200, 255), -1)

    y = y0 + pad + 14
    for label, value, color in rows:
        cv2.putText(frame, label, (x0 + pad + 6, y), font, fs, _C_LABEL, th, cv2.LINE_AA)
        cv2.putText(frame, value, (x0 + pad + label_w, y), font, fs, color, th, cv2.LINE_AA)
        y += lh

    # 录制指示 (右上角红点 + REC)
    if recording:
        h, w = frame.shape[:2]
        cv2.circle(frame, (w - 70, 24), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 56, 30), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)


def draw_pause(frame) -> None:
    """暂停: 暗化全屏 + 居中 PAUSED + 提示."""
    import cv2
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text, fscale, fth = "PAUSED", 1.8, 3
    (tw, tht), _ = cv2.getTextSize(text, font, fscale, fth)
    cx, cy = (w - tw) // 2, (h + tht) // 2
    cv2.putText(frame, text, (cx, cy), font, fscale, (255, 255, 255), fth, cv2.LINE_AA)

    hint = "Press SPACE to resume  |  Q / Esc to quit"
    (hw, _), _ = cv2.getTextSize(hint, font, 0.6, 1)
    cv2.putText(frame, hint, ((w - hw) // 2, cy + 44), font, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
