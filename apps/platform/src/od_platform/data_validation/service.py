from __future__ import annotations
import logging
from typing import List, Optional
from pathlib import Path
from datetime import datetime

from od_platform.data_validation import snapshot

from od_platform.data_validation.registry import (CheckContext, CheckEntry, CheckResult, CheckSeverity, get_all_checks)
from od_platform.data_validation.report import (
    ValidationReport, render_to_logger, write_results_csv, write_json
)
from od_platform.data_validation.snapshot import build_snapshot
from od_platform.common.performance_utils import time_it
from od_platform.common import paths

logger = logging.getLogger(__name__)

@time_it(name=lambda entry, ctx: f"检查:【{entry.name}】", logger_instance=logger, iterations=1)
def _safe_run_one(entry: CheckEntry, ctx: CheckContext) -> CheckResult:
    try:
        return entry.func(ctx)
    except Exception as e:
        logger.error(f"Check {entry.name} 执行异常，兜底为 ERROR")
        return CheckResult(
            entry.name,
            CheckSeverity.ERROR,
            f"Check failed with exception: {e}",
            {"exception": str(e)},
        )

def run_all_checks(ctx: CheckContext) -> List[CheckResult]:
    results = [_safe_run_one(e, ctx) for e in get_all_checks()]
    logger.info("All checks completed.共完成 %d 个检查", len(results))
    return results

def validate_dataset(yaml_path: Path, task_type: Optional[str] = None,
            run_id: Optional[str] = None, write_report: bool = True
                    ) -> ValidationReport:
    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot = build_snapshot(yaml_path=Path(yaml_path), task_type=task_type)
    results = run_all_checks(CheckContext(yaml_path=yaml_path, snapshot=snapshot))
    report = ValidationReport(run_id=resolved_run_id, snapshot=snapshot, results=results)

    render_to_logger(report)

    if write_report:
        run_dir = paths.validation_run_dir(resolved_run_id)
        run_dir.mkdir(parents=True,exist_ok=True)
        write_json(report, run_dir / "report.json")
        write_results_csv(report, run_dir / "result.csv")
        logger.info(f"报告已经写入 %s (report.json + result.csv)", run_dir)
    return report

