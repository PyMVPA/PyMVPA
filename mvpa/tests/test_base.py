# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Test some base functionality which did not make it into a separate unittests"""

import unittest
from tempfile import mktemp

from mvpa.base.sysinfo import sysinfo

class TestBases(unittest.TestCase):

    def testSysInfo(self):
        """Very basic testing -- just to see if it doesn't crash"""

        try:
            sysinfo()
        except Exception, e:
            self.fail('Testing of systemInfo failed with "%s"' % str(e))

        filename = mktemp('mvpa', 'test')
        sysinfo(filename)
        try:
            syslines = open(filename, 'r').readlines()
        except Exception, e:
            self.fail('Testing of dumping systemInfo into a file failed: %s' % str(e))

def suite():
    return unittest.makeSuite(TestBases)


if __name__ == '__main__':
    import runner

