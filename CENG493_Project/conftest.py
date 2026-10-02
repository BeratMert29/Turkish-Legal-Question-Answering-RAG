"""Root conftest.py — adds CENG493_Project/ to sys.path for top-level imports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
