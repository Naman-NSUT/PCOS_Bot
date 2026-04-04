"""
conftest.py — pytest configuration
Adds the project root to sys.path so imports like `from config.settings import ...` work.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
