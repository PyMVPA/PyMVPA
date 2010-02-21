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
from tempfile import mktemp

from mvpa.base.info import wtf

class TestBases(unittest.TestCase):

    def test_wtf(self):
        """Very basic testing -- just to see if it doesn't crash"""

        sinfo = str(wtf())
        sinfo_excludes = str(wtf(exclude=['process']))
        self.failUnless(len(sinfo) > len(sinfo_excludes))
        self.failUnless(not 'Process Info' in sinfo_excludes)

        # check if we could store and load it back
        filename = mktemp('mvpa', 'test')
        wtf(filename)
        try:
            sinfo_from_file = '\n'.join(open(filename, 'r').readlines())
        except Exception, e:
            self.fail('Testing of loading from a stored a file has failed: %r'
                      % (e,))
        os.remove(filename)


def suite():
    return unittest.makeSuite(TestBases)


if __name__ == '__main__':
    import runner

