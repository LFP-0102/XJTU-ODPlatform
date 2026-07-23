#!/usr/bin/env python
"""Django 管理入口。"""
import os
import sys
from pathlib import Path


def main():
    # 把 src/ 挂上 sys.path,让 od_web_backend 可被导入
    src = Path(__file__).resolve().parent / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'od_web_backend.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "无法导入 Django,请确认已安装依赖(见 pyproject.toml)并激活虚拟环境。"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
