from od_platform.data_validation.registry import get_check, CheckContext, get_all_checks, list_check_names
from od_platform.data_validation.snapshot import build_snapshot
from pathlib import Path

yaml_path = Path(r"C:\Users\fym\Desktop\XJTU-ODPlatform\apps\platform\configs\datasets\MRI.yaml")
snap = build_snapshot(yaml_path)

r = get_check("PairExistenceCheck").func(CheckContext(yaml_path=yaml_path, snapshot=snap))

print(r.severity, r.summary)