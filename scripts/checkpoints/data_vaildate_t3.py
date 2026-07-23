from od_platform.data_validation.registry import get_check, CheckContext, get_all_checks, list_check_names
from od_platform.data_validation.snapshot import build_snapshot
from od_platform.data_validation.service import run_all_checks
from pathlib import Path

yaml_path = Path(r"C:\Users\fym\Desktop\XJTU-ODPlatform\apps\platform\configs\datasets\MRI.yaml")
snap = build_snapshot(yaml_path)

for r in run_all_checks(CheckContext(yaml_path=yaml_path, snapshot=snap)):
    print(f"{r.severity:<7} |  {r.name:<24}  | {r.summary}")