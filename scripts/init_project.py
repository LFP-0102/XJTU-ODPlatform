import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_SRC = REPO_ROOT / "apps" / "platform" / "src"

sys.path.insert(0, str(PLATFORM_SRC))

from od_platform.cli.init_project import initialize_project

if __name__ == '__main__':
    initialize_project()