#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""PyMVPA build is now driven by pyproject.toml + meson-python.

This file remains as a thin shim that points users at the modern
invocation if they invoke ``python setup.py ...`` out of habit.
"""

import sys

print(
    "PyMVPA no longer uses setup.py / numpy.distutils.\n"
    "Build with: pip install .         (or: pip install -e .)\n"
    "See pyproject.toml + meson.build for build configuration.",
    file=sys.stderr,
)
sys.exit(2)
