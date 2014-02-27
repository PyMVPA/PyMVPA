# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test some base functionality which did not make it into a separate unittests"""

import os
import unittest

from mvpa2.base.info import wtf
from mvpa2.testing.tools import *

@with_tempfile()
def test_wtf(filename):
    """Very basic testing of wtf()"""

    sinfo = str(wtf())
    sinfo_excludes = str(wtf(exclude=['runtime']))
    ok_(len(sinfo) > len(sinfo_excludes),
        msg="Got not less info when excluded runtime."
        " Original one was:\n%s and without process:\n%s"
        % (sinfo, sinfo_excludes))
    ok_(not 'RUNTIME' in sinfo_excludes)

    # check if we could store and load it back
    wtf(filename)
    try:
        sinfo_from_file = '\n'.join(open(filename, 'r').readlines())
    except Exception, e:
        raise AssertionError(
            'Testing of loading from a stored a file has failed: %r'
            % (e,))
