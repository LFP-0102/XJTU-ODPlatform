from __future__ import  annotations
from pathlib import Path
from typing import Any,Dict, List, Optional, Tuple
import yaml
from od_platform.data_validation.registry import (CheckContext, CheckResult, CheckSeverity, check)
_NAME = "yaml_schema"

def _load_yaml(path: Path) -> Tuple[Optional[Dict[str, Any]],Optional[str]]:
    if not path.exists():
        return None, f"yaml file {path} does not exist"
    try:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except yaml.YAMLError as e:
        return None,  f"yaml file {path} 解析失败: {e}"
    if not isinstance(data, dict):
        return None, f"yaml 顶层不是dict: {type(data).__name__}"
    return data, None

@check(_NAME)
def validate_yaml_schema(ctx: CheckContext) -> CheckResult:
    if ctx.snapshot.yaml_load_error:  # 前置错:早返回
        return CheckResult(_NAME, CheckSeverity.ERROR, "yaml 无法加载",
                           {"error": ctx.snapshot.yaml_load_error})
    data = ctx.snapshot.yaml_data
    problems: List[str] = []
    nc, names = data.get("nc"), data.get("names")
    if not isinstance(nc, int):
        problems.append("nc字段缺失或者nc不是整数")
    if not isinstance(names, (list, dict)):
        problems.append("names字段缺失或为不合法类型: 应该是list或者dict")
    elif isinstance(nc, int) and (len(names)) != nc:
        problems.append(f"nc={nc} 与 names数量{len(names)}不一致")
    if not data.get("path"):
        problems.append("path字段缺失或者找不到数据集的根")
    for split in ("train", "val", "test"):
        if split in data and not isinstance(data[split], str):
            problems.append(f"{split} 字段应该为字符串路径")
    if problems:
        return CheckResult(_NAME,CheckSeverity.ERROR,
            f"yaml 有{len(problems)}个问题: 问题如下：{problems}", {"problems": problems}
        )
    return CheckResult(_NAME, CheckSeverity.PASS, "yaml结构合法", {"nc":nc})