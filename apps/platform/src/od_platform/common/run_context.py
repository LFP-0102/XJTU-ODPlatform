from __future__ import annotations

from datetime import datetime
from pathlib import Path

from od_platform.common import paths

class RunContext:
    def __init__(self, subsystem: str) -> None:
        if not subsystem or "/" in subsystem or subsystem in (".", ".."):
            raise ValueError(f"非法 subsystem: {subsystem!r}")
        self.subsystem = subsystem
        self.started_at: datetime = datetime.now()                   # ★ 这次运行唯一的一次 now()
        self.run_id: str = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.run_dir: Path = paths.RUNS_DIR / subsystem / self.run_id

    @property
    def created_at(self) -> str:
        return self.started_at.isoformat(timespec="seconds")

    def artifact_path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __enter__(self) -> "RunContext":
        base = self.run_dir
        candidate, i = base, 2
        while candidate.exists():
            candidate = base.parent / f"{base.name}-{i}"
            i += 1
        self.run_dir = candidate
        self.run_id = candidate.name
        self.run_dir.mkdir(parents=True)  # exist_ok=False:此刻它一定不存在
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False  # 不吞异常;现场目录留在盘上(产物 + 供事后审计)