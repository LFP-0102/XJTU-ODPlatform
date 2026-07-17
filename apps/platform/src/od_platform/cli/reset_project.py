from __future__ import  annotations
import os
import argparse
import getpass
import shutil
import stat
import sys
from pathlib import Path
import subprocess
import platform
import datetime
from od_platform.common.paths import (ROOT_DIR, RAW_DATA_DIR, PRETRAINED_MODELS_DIR,META_LOGGING_DIR,
                                get_dirs_to_reset, is_protected)
from od_platform.common.performance_utils import time_it

from od_platform.common.logging_utils import get_logger
from od_platform.common.string_utils import format_table_row, format_table_separator


logger = get_logger(base_path=META_LOGGING_DIR, log_type='项目重置')

CONFIRM_KEYWORD = "RESET"
LINE_WIDTH = 70


def _audit_context() -> dict:
    """收集审计上下文：出事时确定谁、哪个版本、在什么环境、被谁触发"""
    ctx = {}

    # ----- 基础执行者与进程 -----
    ctx["user"] = getpass.getuser()
    ctx["pid"] = os.getpid()
    ctx["argv"] = sys.argv
    ctx["cwd"] = os.getcwd()

    # ----- 时间与主机 -----
    ctx["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    try:
        import socket
        ctx["hostname"] = socket.gethostname()
    except Exception:
        ctx["hostname"] = "unknown"

    # ----- Python 运行时 -----
    ctx["python_version"] = sys.version.split()[0]  # 如 "3.9.18"
    ctx["platform"] = platform.platform()           # 如 "Linux-5.15.0-...-x86_64"

    # ----- Git 版本与状态（增强） -----
    try:
        # 获取 commit hash
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip()
        # 获取分支名
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip()
        # 检查是否有未提交修改（dirty）
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT_DIR, text=True, stderr=subprocess.DEVNULL, timeout=2
        ).strip()
        git_dirty = bool(git_status)
    except Exception:
        git_rev = "(not a git repository)"
        git_branch = "unknown"
        git_dirty = False

    ctx["git_rev"] = git_rev
    ctx["git_branch"] = git_branch
    ctx["git_dirty"] = git_dirty   # True 表示本地有未提交修改（慎用此版本）

    # ----- 父进程信息（可选，依赖 psutil） -----
    try:
        import psutil
        proc = psutil.Process()
        parent = proc.parent()
        if parent:
            ctx["ppid"] = parent.pid
            ctx["parent_cmdline"] = " ".join(parent.cmdline())
        else:
            ctx["ppid"] = None
            ctx["parent_cmdline"] = ""
    except (ImportError, Exception):
        # 如果没有 psutil 或获取失败，则忽略这两个字段
        pass

    # ----- 环境标识（常用于区分 dev/test/prod） -----
    ctx["env_stage"] = os.environ.get("ENV") or os.environ.get("STAGE") or "unknown"

    return ctx


def _format_size(bytes_size: int) -> str:
    if bytes_size >= 1024 ** 3:
        return f"{bytes_size / (1024 ** 3):.2f} GiB"
    elif bytes_size >= 1024 ** 2:
        return f"{bytes_size / (1024 ** 2):.2f} MiB"
    elif bytes_size >= 1024:
        return f"{bytes_size / 1024:.2f} KiB"
    else:
        return f"{bytes_size} B"

def _on_rm_error(func, path, exc):
    """解决windows上某些只读文件删除权限不足的异常场景"""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise

def _scan_targets() -> tuple[list[tuple[Path, int ,int]], list[Path]]:
    """扫描所有目标，返回可删除的，可跳过的目录
    可删除项目：path file_count, total_size_bytes
    """
    deletable: list[tuple[Path, int, int]] = []
    skipped: list[Path] = []


    for d in get_dirs_to_reset():
        if is_protected(d):
            logger.warning(f"🙅‍♂️ 拒绝删除受保护的目录： {d}")
            skipped.append(d)
            continue
        if not d.exists():
            skipped.append(d)
            continue
        file_count = 0
        total_size = 0
        try:
            for f in d.rglob("*"):
                if f.is_file():
                    file_count += 1
                    try:
                        total_size += f.stat().st_size
                    except OSError:
                        pass
        except OSError as e:
            logger.warning(f"扫描目录 {d} 时出错：{e}")
        deletable.append((d, file_count, total_size))
    return  deletable, skipped

def _print_plan(
        deletable: list[tuple[Path, int, int]],
        skipped: list[Path],
        will_actually_delete: bool
    ) -> None:
    if will_actually_delete:
        logger.warning(f"  🚨  即将删除以下目录".center(LINE_WIDTH, '='))
    else:
        logger.info(f"  📦  [DRY-RUN] 计划如下(未实际删除)".center(LINE_WIDTH, '='))

    if not deletable:
        logger.info(f"没有可删除的目录-当前项目已经是干净的状态")
        return

    widths = [42, 12, 14]
    align = ['left', 'right', 'right']
    logger.info(format_table_row(['目录', '文件数量', '总大小'], widths, align))
    logger.info(format_table_separator(widths))

    total_files = 0
    total_bytes = 0
    for path, count , size in deletable:
        rel = path.relative_to(ROOT_DIR)
        logger.info(format_table_row([str(rel), str(count), _format_size(size)], widths,align))
        total_files += count
        total_bytes += size
    logger.info(format_table_separator(widths))
    logger.info(format_table_row([f"【 总计 】", f"{str(total_files)}", _format_size(total_bytes)], widths, align))

    logger.info("")
    logger.info("以下重要目录不会被删除")
    logger.info(f" -原始数据： {RAW_DATA_DIR.relative_to(ROOT_DIR)}/")
    logger.info(f" -预训练模型： {PRETRAINED_MODELS_DIR.relative_to(ROOT_DIR)}/")
    logger.info(f" -所有的代码，文档不会被删除")


def _confirm(deletable_count: int) -> bool:
    """交互式确认"""
    print()
    print('=' * LINE_WIDTH)
    print(f"  🚨  你正要删除 {deletable_count} 个目录内容，这个操作不可撤销")
    print(f"  ⚠️  如果确认删除，请输入大写的 '{CONFIRM_KEYWORD}' (其他任何输入都会被取消)")
    print('=' * LINE_WIDTH)
    try:
        user_input = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    return user_input == CONFIRM_KEYWORD


def _delete_one(path: Path, idx: int, total:int, file_count: int, size: int) -> str | None:
    """删除单个目录，带进度提示，返回None=成功，返回字符串=失败原因"""
    if is_protected(path):
        logger.error(f"[{idx}/{total}] 删除前检查失败，跳过{path}")
        return "受保护的目录"
    rel = path.relative_to(ROOT_DIR)
    size_str = _format_size(size)

    if size > 1024 ** 3:
        logger.warning(f"[{idx}/{total}] 正在删除{rel} ({size_str}, {file_count}个文件)"
                    f"这可能需要等一会，请等待...")
    else:
        logger.info(f"[{idx} / {total}] 删除{rel} ({size_str}, {file_count}个文件)")

    try:
        shutil.rmtree(path, onexc=_on_rm_error)
        logger.info(f"[{idx}/{total}] ✅ {rel} 删除成功：")
    except OSError as e:
        logger.error(f"[{idx}/{total}] ❌ {rel} 删除失败：，错误信息：{e}")
        return f"删除失败：{e}"

def _execute_delete(deletable: list[tuple[Path, int, int]]) -> None:
    total = len(deletable)
    success: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for idx, (path, file_count, size ) in enumerate(deletable, 1):
        reason = _delete_one(path, idx, total, file_count, size)
        if reason is None:
            success.append(path)
        else:
            failed.append((path, reason))
    logger.info('=' * LINE_WIDTH)
    if failed:
        logger.warning(f"完成：成功{len(success)}，失败{len(failed)}，共{total}个目录")
        for p, reason in failed:
            logger.warning(f" - {p.relative_to(ROOT_DIR)}: {reason}")
    else:
        logger.info(f"完成： 成功{len(success)}，失败0个")

@time_it(iterations=1, logger_instance=logger, name='项目初始化')
def reset_project(yes: bool = False, force: bool = False, dry_run: bool = False) -> int:
    logger.info("项目重置工具启动".center(LINE_WIDTH, "="))
    logger.info(f"项目的根路径是：{ROOT_DIR}")

    # ----- 审计上下文（增强） -----
    ctx = _audit_context()
    logger.info(f"审计： user={ctx['user']}, pid={ctx['pid']}, host={ctx.get('hostname', 'unknown')}")
    logger.info(f"审计： git_rev={ctx['git_rev']}, branch={ctx.get('git_branch', 'unknown')}, dirty={ctx.get('git_dirty', False)}")
    logger.info(f"审计： cwd={ctx['cwd']}, env={ctx.get('env_stage', 'unknown')}")
    logger.info(f"审计： argv={' '.join(ctx['argv'])}")
    # 如果父进程信息存在（需安装 psutil），再额外打印
    if 'ppid' in ctx:
        logger.info(f"审计： parent_pid={ctx['ppid']}, parent_cmd={ctx.get('parent_cmdline', '')}")
    # 也可选择将时间戳记录下来（但不必须，因为日志系统本身会有时间）
    logger.info(f"审计： timestamp={ctx.get('timestamp', '')}")

    if dry_run and yes:
        logger.warning(f"‼️  同时给了 --dry-run 和 --yes，以 --dry-run模式运行" )
        yes = False
    deletable, skipped = _scan_targets()
    _print_plan(deletable, skipped, will_actually_delete=yes)

    if not deletable:
        return 0

    if not yes:
        if dry_run:
            logger.info(" 这是显式的 --dry-run模式，要执行真正的删除，请加 --yes：")
        else:
            logger.info(" 这是 --dry-run(默认行为)，要执行真正的删除，请加 --yes：")
        return 0

    if not force:
        if not _confirm(len(deletable)):
            logger.info("用户取消, 未执行任何删除操作")
            return 1
    logger.info("")
    logger.info("开始执行删除操作".center(LINE_WIDTH, '='))
    _execute_delete(deletable)
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(
        description="项目重置工具 主要用于将项目恢复到初始化的状态",
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--yes",action="store_true", help="真正执行删除 (默认是dry-run)")
    parser.add_argument("--force",action="store_true", help="强制删除，不进行确认")
    parser.add_argument("--dry-run",action="store_true", help=" dry-run模式，不实际删除")
    args = parser.parse_args()
    return reset_project(yes=args.yes, force=args.force, dry_run=args.dry_run)

if __name__ == "__main__":
    sys.exit(main())
