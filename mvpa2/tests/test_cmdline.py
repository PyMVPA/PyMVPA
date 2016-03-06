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
from mvpa2.testing import *

from mvpa2.misc.cmdline import *

if __debug__:
    from mvpa2.base import debug

class CmdlineHelpersTest(unittest.TestCase):

    def test_basic(self):
        """Test if we are not missing basic parts"""
        globals_ = globals()
        for member in  [#'_verbose_callback',
                        'parser', 'opt', 'opts']:
            self.assertTrue(member in globals_,
                msg="We must have imported %s from mvpa2.misc.cmdline!" % member)

    @sweepargs(example=[
        ('targets:rest', None, [('targets', ['rest'])]),
        ('targets:rest;trial:bad,crap,shit', None,
         [('targets', ['rest']), ('trial', ['bad', 'crap', 'shit'])]),
        ])
    def test_split_comma_semicolon_lists(self, example):
        s, dtype, t = example
        v = split_comma_semicolon_lists(s, dtype=dtype)
        assert_equal(v, t)

def suite():  # pragma: no cover
    return unittest.makeSuite(CmdlineHelpersTest)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

