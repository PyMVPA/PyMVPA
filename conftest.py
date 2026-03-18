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


def pytest_configure(config):
    """Suppress upstream warnings from third-party libraries."""
    # Use pytest's native filterwarnings so they survive per-test resets
    config.addinivalue_line(
        "filterwarnings",
        "ignore:the matrix subclass:PendingDeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:.*__array__.*keyword argument:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*invalid value encountered:RuntimeWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*The integral is probably divergent:UserWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*distutils.*:DeprecationWarning",
    )
    # scipy.stats ConstantInputWarning — suppress via warnings module
    # since pytest filterwarnings can't resolve scipy's warning class
    import warnings
    try:
        from scipy.stats import ConstantInputWarning
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
    except ImportError:
        pass
