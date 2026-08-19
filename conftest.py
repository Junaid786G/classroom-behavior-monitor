"""Ensure the project root is importable as `backend...` during tests."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
