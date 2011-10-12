# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA cmdline helpers"""

import unittest
from mvpa2.misc.cmdline import *

if __debug__:
    from mvpa2.base import debug

class CmdlineHelpersTest(unittest.TestCase):

    def test_basic(self):
        """Test if we are not missing basic parts"""
        globals_ = globals()
        for member in  [#'_verbose_callback',
                        'parser', 'opt', 'opts']:
            self.failUnless(globals_.has_key(member),
                msg="We must have imported %s from mvpa2.misc.cmdline!" % member)

def suite():
    return unittest.makeSuite(CmdlineHelpersTest)


if __name__ == '__main__':
    import runner

