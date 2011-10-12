# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support for python's copy module.

"""

__docformat__ = 'restructuredtext'

import sys

# We have to use deepcopy from python 2.5, since otherwise it fails to
# copy sensitivity analyzers with assigned combiners which are just
# functions not functors
if sys.version_info[:2] >= (2, 6):
    # enforce absolute import
    _copy = __import__('copy', globals(), locals(), [], 0)
    copy = _copy.copy
    deepcopy = _copy.deepcopy
else:
    from mvpa2.support._copy import copy, deepcopy
