# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Re-export of the stdlib `copy` module.

Historically pymvpa shipped a vendored PY2-era copy module
(`mvpa2.support._copy`) as a fallback for Python <2.6.  That fallback
is gone; this module simply re-exports `copy.copy` and `copy.deepcopy`
under the legacy import path so existing call sites
(`from mvpa2.support.copy import deepcopy`) keep working.
"""

__docformat__ = 'restructuredtext'

from copy import copy, deepcopy  # noqa: F401  (re-exported)
