#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :run_context.py
# @Time      :2026/7/16 15:14:48
# @Author    :雨霓同学
# @Project   :apps/platform/src/od_platform/common/run_context.py
# @Function  :
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from od_platform.common import paths

class RunContext:
    def __init__(self, subsystem: str) -> None:
        if not subsystem or "/" in subsystem in (".", ".."):
            raise ValueError(f"非法 subsystem: {subsystem}")
        self.subsystem = subsystem  # runs目录下的子目录名,每个子系统对应一个
        self.started_at = datetime.now()
        self.run_id: str = self.started_at.strftime("%Y%m%d-%H%M%S")
        self.run_dir = paths.RUNS_DIR / self.subsystem / self.run_id

    @property
    def created_at(self) -> str:
        return self.started_at.isoformat(timespec='seconds')

    def artifact_path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __enter__(self) -> "RunContext":
        base = self.run_dir
        candidate, i = base, 2
        while candidate.exists():
            candidate = base.parent / f"{base.name}_{i}"
            i += 1
        self.run_dir = candidate
        self.run_id = candidate.name
        self.run_dir.mkdir(parents=True)
        return self

    def __exit__(self, exc_type, val, tb) -> bool:
        return False



