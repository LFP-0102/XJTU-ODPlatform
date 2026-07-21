#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :_lineage.py
# @Time      :2026/7/17 14:20:46
# @Author    :雨霓同学
# @Project   :XJTU-ODPlatfrom
# @Function  :
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from od_platform.common import paths
from od_platform.common.lineage import SplitManifest

def load_manifest_from_yaml(
        yaml_data: Dict[str, Any],
) -> Tuple[Optional[SplitManifest], Optional[str], Dict[str, Any]]:
    odp: Dict[str, Any] = (yaml_data or {}).get("odp_meta") or {}
    ref = odp.get("manifest_path")
    if not ref:
        return None, "Yaml 的odp_meta没有manifest_path -- 这个配置不是odp-transform产出的",  odp
    manifest_path = paths.ROOT_DIR / ref
    if not manifest_path.exists():
        return None, f"manifest_path 不存在: {manifest_path}(odp_meta.manifest_path={ref})", odp
    try:
        return SplitManifest.read(manifest_path), None, odp
    except Exception as e:
        return None, f"manifest 读取失败：{type(e).__name__}:{e}", odp


