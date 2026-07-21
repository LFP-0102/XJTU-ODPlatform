#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : registry.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/frame_source/core/registry.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 10:32:48
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:32:48 | Matri | v1.0.0 | 初始化创建
# @Function  : 源注册表 —— 配置类型是派发轴; @register_source 让源自注册
"""
源注册表: frame_source 可扩展性的核心。

派发轴 = 配置类型:
    build_from_config(cfg) 靠 type(cfg) 在 _BY_CONFIG 里一步查到源类。
    确定、不歧义, 且天然支持硬件源(深度/红外/双目都传各自的 config)。

字符串糖(可选, 二线):
    能自证身份的输入(路径/URL/裸数字)可由源自愿登记一条字符串规则,
    build_from_string 先把字符串翻译成 config, 再走 build_from_config。
    硬件源不登记字符串规则 —— 所以 "0" 不会在深度/红外/彩色之间产生歧义。

扩展方式(规矩 E, 对修改关闭):
    @register_source(XxxConfig, str_matcher=..., str_to_config=...)
    class XxxSource(FrameSource): ...
    内置源在自己模块里这样登记; 第三方源在自己的包里这样登记, import 一下即接入,
    factory.py / registry.py 的既有代码一个字不用改。

设计取舍(不做什么):
    不做 matcher 打分 + 优先级 + spec 那套重机制 —— 派发轴是配置类型, 一个
    dict 查表即可, 没有歧义要仲裁。字符串命中多个时当场 raise 报歧义, 不猜。
"""
from __future__ import annotations

import logging
from typing import Callable, NamedTuple

from .base   import FrameSource
from .config import SourceConfig


logger = logging.getLogger(__name__)


class StrRule(NamedTuple):
    """一条字符串识别规则(某个源自愿登记)。"""
    name:      str                              # 源类名(报错/内省用)
    matcher:   Callable[[str], bool]            # 纯判断: 这个字符串归我吗(契约: 不得抛异常)
    to_config: Callable[[str], SourceConfig]    # 翻译: 字符串 → 该源的 config


# ── 注册表(模块级单例)────────────────────────────────────
_BY_CONFIG: dict[type[SourceConfig], type[FrameSource]] = {}   # 配置类型 → 源类(主派发表)
_STR_RULES: list[StrRule] = []                                 # 字符串规则(可选糖)


def register_source(
    config_cls: type[SourceConfig],
    *,
    str_matcher:   Callable[[str], bool] | None = None,
    str_to_config: Callable[[str], SourceConfig] | None = None,
):
    """
    源自注册装饰器。

    Args:
        config_cls    : 该源对应的配置类(派发的钥匙, 必填)。
        str_matcher   : 可选。str -> bool, 判断某字符串是否归本源。
                        契约: 必须是纯判断, 不得抛异常。
        str_to_config : 可选。str -> config, 把字符串翻译成本源的 config。
                        提供 str_matcher 时必须一并提供。

    用法:
        @register_source(VideoConfig,
                        str_matcher=lambda s: Path(s).suffix.lower() in VIDEO_EXTENSIONS,
                        str_to_config=lambda s: VideoConfig(path=s))
        class VideoSource(FrameSource): ...

        # 硬件源: 只给 config_cls, 不给字符串规则(字符串分不开硬件)
        @register_source(DepthConfig)
        class DepthSource(FrameSource): ...

    Raises:
        TypeError : config_cls 不是 SourceConfig 子类, 或被装饰类不是 FrameSource 子类。
        ValueError: 给了 str_matcher 却没给 str_to_config; 或该 config 已被别的源注册。
    """
    if not (isinstance(config_cls, type) and issubclass(config_cls, SourceConfig)):
        raise TypeError(f"config_cls 必须是 SourceConfig 的子类, 收到: {config_cls!r}")
    if str_matcher is not None and str_to_config is None:
        raise ValueError("提供 str_matcher 时必须同时提供 str_to_config")

    def decorator(source_cls: type[FrameSource]) -> type[FrameSource]:
        if not (isinstance(source_cls, type) and issubclass(source_cls, FrameSource)):
            raise TypeError(f"被注册的类必须是 FrameSource 的子类, 收到: {source_cls!r}")

        existing = _BY_CONFIG.get(config_cls)
        if existing is not None and existing is not source_cls:
            raise ValueError(
                f"配置 {config_cls.__name__} 已被 {existing.__name__} 注册, "
                f"拒绝用 {source_cls.__name__} 覆盖(一种配置只能对应一个源)"
            )

        _BY_CONFIG[config_cls] = source_cls
        if str_matcher is not None:
            _STR_RULES.append(StrRule(source_cls.__name__, str_matcher, str_to_config))

        logger.debug(
            f"已注册源: {config_cls.__name__} → {source_cls.__name__}"
            + (" (+字符串规则)" if str_matcher is not None else "")
        )
        return source_cls

    return decorator


def build_from_config(config: SourceConfig) -> FrameSource:
    """
    按配置类型派发, 造出对应的源。这是唯一的"造源"出口。

    Args:
        config: 任意 SourceConfig 子类实例。

    Returns:
        对应的 FrameSource 实例(尚未 open)。

    Raises:
        TypeError : config 不是 SourceConfig 实例。
        ValueError: config 的类型没有被任何源注册。
    """
    if not isinstance(config, SourceConfig):
        raise TypeError(f"build_from_config 需要 SourceConfig 实例, 收到 {type(config).__name__}")

    source_cls = _BY_CONFIG.get(type(config))
    if source_cls is None:
        raise ValueError(
            f"未注册的配置类型: {type(config).__name__}。"
            f"已注册: {sorted(c.__name__ for c in _BY_CONFIG)}"
        )
    return source_cls(config)


def build_from_string(source: str) -> FrameSource:
    """
    把字符串按注册的规则翻译成 config, 再走 build_from_config。这是字符串糖的入口。

    Args:
        source: 输入字符串(如 "0" / "x.mp4" / "rtsp://..." / "./imgs")。

    Returns:
        对应的 FrameSource 实例(尚未 open)。

    Raises:
        TypeError : source 不是 str。
        ValueError: 没有任何规则命中(无法识别); 或命中多个规则(歧义);
                    或命中规则但构造 config 失败(如路径不存在)。
    """
    if not isinstance(source, str):
        raise TypeError(f"build_from_string 需要 str, 收到 {type(source).__name__}")

    # 注: matcher 契约要求纯判断不抛异常, 故此处不做 try 容错 —— matcher 抛了就该暴露
    hits = [rule for rule in _STR_RULES if rule.matcher(source)]

    if not hits:
        raise ValueError(
            f"无法识别的源字符串: {source!r}。"
            f"已注册的字符串规则: {[r.name for r in _STR_RULES]}"
        )
    if len(hits) > 1:
        raise ValueError(
            f"源字符串 {source!r} 命中多个规则(歧义): {[h.name for h in hits]}。"
            f"这通常意味着某个源的 matcher 写得过宽, 或存在冲突的第三方源。"
        )

    rule = hits[0]
    try:
        config = rule.to_config(source)     # 把字符串翻译成 config(可能因路径不存在等 raise)
    except Exception as e:
        # 统一字符串边界的异常类型为 ValueError, 方便调用方一处 except;
        # from e 保留原始异常链(含 pydantic ValidationError 的详细字段报错), 不丢调试信息
        raise ValueError(f"由 {source!r} 构造 {rule.name} 的配置失败: {e}") from e

    return build_from_config(config)


def list_sources() -> dict[str, str]:
    """
    内省当前注册表, 返回 {配置类名: 源类名}。

    用于回答"到底支持哪些源"—— 这是注册表相对 if/elif 工厂损失的可发现性的补偿:
    if/elif 读一个函数就全知道, 注册表得靠这个内省(或全仓 grep config 类)。
    """
    return {cfg.__name__: src.__name__ for cfg, src in _BY_CONFIG.items()}