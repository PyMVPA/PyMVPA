# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit test for PyMVPA mvpa.suite() of being loading ok"""

import unittest

class SuiteTest(unittest.TestCase):

    def testBasic(self):
        """Test if we are loading fine"""
        try:
            exec "from mvpa.suite import *"
        except Exception, e:
            self.fail(msg="Cannot import everything from mvpa.suite."
                      "Getting %s" % e)

def suite():
    return unittest.makeSuite(SuiteTest)


if __name__ == '__main__':
    import runner

