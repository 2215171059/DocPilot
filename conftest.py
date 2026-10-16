"""
pytest 的根配置：把 src 目录加进模块搜索路径。

不做这一步的话，pytest 找不到 docpilot 包 ——
因为我们的项目用的是 src 布局（源码放在 src/ 下），
而不是把包直接放在项目根目录。
src 布局的好处是能避免"本地目录被误当成已安装的包"这类问题，
代价就是要多写这几行。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
