#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : cancel.py
# @Project   : ODPlatform
# @Function  : 线程安全的取消信号 (CancelToken)
"""推理取消信号.

GUI 端 / web 端 没有键盘按键, 需要程序化告诉 pipeline "该停了".
ThreadedPipeline 在主循环 4 个检查点查询 is_cancelled(), 收到信号优雅退出.

使用模式:
    token = CancelToken()

    # 在后台启动推理
    threading.Thread(target=lambda: service.predict(..., cancel_token=token)).start()

    # 用户点取消按钮 / WebSocket 断开 / Celery task revoke
    token.cancel()

    # pipeline 在下次主循环查询点检测到 (< 2s), 退出主循环
"""
from __future__ import annotations

import threading


class CancelToken:
    """线程安全的取消信号.

    调用方持有, pipeline 周期性查询. 永远只查不阻塞 (除非显式 wait).
    """

    def __init__(self) -> None:
        self._evt = threading.Event()

    def cancel(self) -> None:
        """请求取消. 幂等 (多次调无副作用)."""
        self._evt.set()

    def is_cancelled(self) -> bool:
        """非阻塞查询. pipeline 主循环每个检查点调一次."""
        return self._evt.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """阻塞等待 cancel 信号. 给"等推理结束 + 超时强终止"的场景."""
        return self._evt.wait(timeout)


class InferenceCancelled(Exception):
    """推理被主动取消. 不一定真用 (pipeline 走优雅退出更常见),
    留个类型给业务方在需要明确区分"用户取消"和"真错误"时用."""
    pass