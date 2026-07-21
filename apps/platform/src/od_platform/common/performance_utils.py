#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :performance_utils.py
# @Time      :2026/7/14 09:38:25
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :性能测量工具
import logging
import time
from functools import wraps
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)

def time_it(
        iterations: int = 1,
        name: Optional[Union[str, Callable[..., str]]] = None,
        logger_instance: logging.Logger = None,
    ):
    log = logger_instance if logger_instance is not None else logger

    # 可以先让产出的单位统一
    def _format_time_auto_unit(seconds: float) -> str:
        """自动选择合适的单位"""
        if seconds < 0.001:
            return f"{seconds * 1_000_000:.3f} 微妙"
        elif seconds < 1.0:
            return f"{seconds * 1000:.3f} 毫秒"
        elif seconds < 60:
            return f"{seconds:.2f} 秒"
        elif seconds < 3600:
            mins = seconds // 60
            secs = seconds % 60
            return f"{mins:.0f}分钟 {secs:.2f}秒"
        else:
            hours = seconds // 3600
            mins = seconds // 60
            secs = seconds % 60
            return  f"{hours}小时 {mins:.0f}分钟 {secs:.2f}秒"

    def _resolve_name(func, args, kwargs) -> str:
        if callable(name):
            try:
                return name(*args, **kwargs)
            except Exception:
                log.warning(f"time_it: name() 计算失败，会退到{func.__name__}",exc_info=True)
                return func.__name__
        if name is not None:
            return name
        return func.__name__

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            display_name = _resolve_name(func, args, kwargs)
            total = 0.0
            result = None
            for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                total += (end - start)
            avg = total / iterations
            avg_str = _format_time_auto_unit(avg)

            if iterations == 1:
                log.info(f"性能报告： '{display_name}' 执行了{iterations}次，耗时 {avg_str}")
            else:
                total_str = _format_time_auto_unit(total)
                log.info(f"性能报告： '{display_name}' 执行了{iterations}次 | "
                        f"平均耗时 {avg_str}，总耗时 {total_str}")

            return result
        return wrapper
    return decorator
