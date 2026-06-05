"""conftest.py: 确保项目根目录在 sys.path 中，方便 pytest 发现模块。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
