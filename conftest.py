# conftest.py - pytest configuration for PyMVPA
import sys
import os

# Ensure mvpa2 is importable from the source tree
sys.path.insert(0, os.path.dirname(__file__))

# Configure pytest collection to avoid picking up classes
# that happen to start with "Test" but aren't test classes
collect_ignore_glob = [
    "doc/*",
    "build/*",
    "3rd/*",
]
