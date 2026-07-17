from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from od_platform.common import paths
# 契约类型来自 common/lineage(生产/核验两方共用的公共设施),不 import data_pipeline。
from od_platform.common.lineage import SplitManifest


def load_manifest_from_yaml(
    yaml_data: Dict[str, Any],
) -> Tuple[Optional[SplitManifest], Optional[str], Dict[str, Any]]:
    odp: Dict[str, Any] = (yaml_data or {}).get("odp_meta") or {}
    ref = odp.get("manifest_path")
    if not ref:
        return None, ("yaml 的 odp_meta 里没有 manifest_path —— "
                      "这份 yaml 可能不是 odp-transform 产出的"), odp
    manifest_path = paths.ROOT_DIR / ref
    if not manifest_path.exists():
        return None, f"manifest 不存在: {manifest_path}(odp_meta.manifest_path={ref})", odp
    try:
        return SplitManifest.read(manifest_path), None, odp
    except Exception as e:
        return None, f"manifest 读取失败: {type(e).__name__}: {e}", odp