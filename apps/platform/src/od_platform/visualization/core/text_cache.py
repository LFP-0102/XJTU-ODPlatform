#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File       : text_cache.py
# @Path       : XJTU-ODPlatfrom/apps/platform/src/od_platform/visualization/core/text_cache.py
# @Project    : XJTU-ODPlatfrom
# @Author     : Matri
# @Date       : 2026-07-20 14:11:53
# @Version    : v1.0.0
# @Description:
# @ChangeLog:
#   2026-07-20:11:53 | Matri | v1.0.0 | 初始化创建
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : text_cache.py
# @Author    : 雨霓同学
# @Project   : visualization
# @Function  : 文本尺寸预计算缓存 — 启动期一次性算完,运行期 O(1) 查表
"""文本尺寸预计算缓存。

撞墙修复:
  ① 字体路径裸文件名破坏可拷贝性 → font_path=None 时解析到模块内置 assets/
  ② 字体加载失败静默 fallback → 显式 logger.warning(只发一次)+ 用回退值

  其中 ② 的 fallback 值 ImageFont.load_default() 是 bitmap 字体,不接受 size 参数,
  所有字号都拿到同一个尺寸 —— 预计算的"按字号查表"将失效。warning 文案里点出来,
  让用户在 log 里看到,而不是看着没事实际跑出错位的标签。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)

# 模块内置字体目录(整包拷走后依然有效)
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
# 默认字体名(不带扩展名;会在 assets / 系统字体目录中查找)
_DEFAULT_FONT_NAME = "LXGWWenKai-Bold"
# 认得的字体文件扩展名
_FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")


def _iter_system_font_dirs() -> List[Path]:
    """按操作系统返回存在的系统字体目录。"""
    dirs: List[Path] = []
    if sys.platform.startswith("win"):
        win = os.environ.get("WINDIR", r"C:\Windows")
        dirs.append(Path(win) / "Fonts")
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    elif sys.platform == "darwin":
        dirs += [
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path.home() / "Library" / "Fonts",
        ]
    else:  # linux / 其它 unix
        dirs += [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".fonts",
            Path.home() / ".local" / "share" / "fonts",
        ]
    return [d for d in dirs if d.is_dir()]


def _match_font_in_dir(directory: Path, name: str, recursive: bool) -> Optional[str]:
    """在 directory 中按"文件名 / 去扩展名文件名"查字体(大小写不敏感)。"""
    has_ext = Path(name).suffix.lower() in _FONT_EXTENSIONS

    # 非递归(assets):直接拼扩展名精确命中,快
    if not recursive:
        if has_ext:
            cand = directory / name
            return str(cand) if cand.is_file() else None
        for ext in _FONT_EXTENSIONS:
            cand = directory / f"{name}{ext}"
            if cand.is_file():
                return str(cand)
        return None

    # 递归(系统字体目录):遍历匹配。字体可能很多,但只在初始化时扫一次
    name_lower = name.lower()
    try:
        for f in directory.rglob("*"):
            if f.suffix.lower() not in _FONT_EXTENSIONS:
                continue
            if f.name.lower() == name_lower or f.stem.lower() == name_lower:
                return str(f)
    except (OSError, PermissionError):
        pass
    return None


def _resolve_font_path(font: Optional[str]) -> str:
    """把"字体名 / 文件名 / 路径 / None"解析成一个可加载的字体路径。

    解析顺序:
      0. None         -> 用默认字体名 'LXGWWenKai-Bold' 继续往下找
      1. 已存在的文件  -> 直接用(兼容老的"传绝对/相对路径"写法)
      2. 模块 assets/  -> 找 <name> 或 <name>.ttf/.otf/.ttc
      3. 系统字体目录   -> 递归找文件名(可不带扩展名)匹配的字体
      4. 都没有        -> 原样返回 name,交给 PIL 最后尝试;失败则在加载处给出 warning

    所谓字体"名字"= 字体文件名,可不带扩展名,例如 'LXGWWenKai-Bold' / 'msyh' / 'simhei'。
    (注:这里按文件名匹配,不是字体内部的 family 名;Windows 的 'msyh.ttc' 写 'msyh' 即可)
    """
    name = font if font else _DEFAULT_FONT_NAME

    # 1) 本身就是存在的文件(绝对或相对路径)
    if Path(name).is_file():
        return str(name)

    # 2) 模块内置 assets 目录
    hit = _match_font_in_dir(_ASSETS_DIR, name, recursive=False)
    if hit:
        return hit

    # 3) 系统字体目录
    for d in _iter_system_font_dirs():
        hit = _match_font_in_dir(d, name, recursive=True)
        if hit:
            return hit

    # 4) 兜底:原样返回,让 PIL 自己试(部分系统能按名找到);失败则走 fallback warning
    return name


class TextSizeCache:
    """文本尺寸预计算缓存。

    初始化时一次性计算所有 (display_label, font_size) 组合,运行时 O(1) 查表。
    """

    def __init__(
            self,
            labels: List[str],
            label_mapping: Optional[Dict[str, str]] = None,
            font_path: Optional[str] = None,
            font_sizes: Optional[Tuple[int, ...]] = None,
            confidence_template: str = "99.0%",
    ):
        self.font_path = _resolve_font_path(font_path)
        self.label_mapping = label_mapping or {}
        self.font_sizes = font_sizes or tuple(range(10, 42, 2))
        self.confidence_template = confidence_template

        # fallback 只 warning 一次,避免预计算 16 个字号刷 16 条相同 log
        self._fallback_warned: bool = False

        self._size_cache: Dict[Tuple[str, int], Tuple[int, int]] = {}
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

        self._precompute(labels)

    # ── 内部:字体加载(显式 fallback)──────────────────────────
    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """加载一个字号的字体;失败时显式 warning + 回退到 PIL 默认字体。"""
        try:
            return ImageFont.truetype(self.font_path, size)
        except OSError as e:
            if not self._fallback_warned:
                logger.warning(
                    f"字体 '{self.font_path}' 加载失败({e}),已回退到 PIL 默认 bitmap 字体。"
                    f"后果: (1)中文字符将无法正常显示 (2)默认字体不支持自定义字号,"
                    f"TextSizeCache 的按字号预计算会失效 —— 标签框尺寸可能与文本错位。"
                    f"修复: 把支持中文的字体放到 visualization/assets/ 下,"
                    f"或把字体名/文件名(如 'LXGWWenKai-Bold' / 'msyh')传给 font_path"
                    f"(会自动在 assets 和系统字体目录中查找)。"
                )
                self._fallback_warned = True
            return ImageFont.load_default()

    def _load_fonts(self) -> None:
        for size in self.font_sizes:
            self._font_cache[size] = self._load_font(size)

    def _precompute(self, labels: List[str]) -> None:
        self._load_fonts()

        measure_img = Image.new("RGB", (1, 1))
        measure_draw = ImageDraw.Draw(measure_img)

        display_labels = set(labels)
        for label in labels:
            if label in self.label_mapping:
                display_labels.add(self.label_mapping[label])

        for display_label in display_labels:
            full_text = f"{display_label} {self.confidence_template}"

            for size in self.font_sizes:
                font = self._font_cache[size]
                bbox = measure_draw.textbbox((0, 0), full_text, font=font)
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                self._size_cache[(display_label, size)] = (width, height)

    # ── 公共 API ────────────────────────────────────────────
    def get_size(self, display_label: str, font_size: int) -> Tuple[int, int]:
        """O(1) 获取文本尺寸;字号不在预计算集合时按比例缩放最近邻。"""
        key = (display_label, font_size)
        if key in self._size_cache:
            return self._size_cache[key]

        nearest_size = min(self.font_sizes, key=lambda s: abs(s - font_size))
        fallback_key = (display_label, nearest_size)

        if fallback_key in self._size_cache:
            w, h = self._size_cache[fallback_key]
            scale = font_size / nearest_size
            return int(w * scale), int(h * scale)

        return (100, 30)

    def get_font(self, font_size: int) -> ImageFont.FreeTypeFont:
        """获取缓存的字体对象;未缓存的字号现场加载并入缓存。"""
        if font_size in self._font_cache:
            return self._font_cache[font_size]

        font = self._load_font(font_size)
        self._font_cache[font_size] = font
        return font