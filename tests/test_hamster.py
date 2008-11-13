#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Hamster"""

import os
import unittest
from tempfile import mktemp
import numpy as N

from mvpa.misc.io.hamster import *

class HamsterHelperTests(unittest.TestCase):

    def testSimpleStorage(self):
        ex1 = """eins zwei drei
        0 1 2
        3 4 5
        """
        ex2 = {'d1': N.random.normal(size=(4,4))}

        hamster = Hamster(ex1=ex1)
        hamster.d = ex2
        hamster.boo = HamsterHelperTests

        filename = mktemp('mvpa', 'test')
        # dump
        hamster.dump(filename)
        # load
        hamster2 = Hamster(filename)

        # check if we re-stored all the keys
        k =  hamster.keys(); k.sort()
        k2 = hamster2.keys(); k2.sort()
        self.failUnless(k == k2)

        # identity should be lost
        self.failUnless(hamster.ex1 is hamster.ex1)
        self.failUnless(not (hamster.ex1 is hamster2.ex1))

        # lets compare
        self.failUnless(hamster.ex1 == hamster2.ex1)

        self.failUnless(hamster.d.keys() == hamster2.d.keys())
        self.failUnless((hamster.d['d1'] == hamster2.d['d1']).all())


        self.failUnless(hamster.boo == hamster2.boo)
        # not sure if that is a feature or a bug
        self.failUnless(hamster.boo is hamster2.boo)

        # cleanup
        os.remove(filename)




def suite():
    return unittest.makeSuite(HamsterHelperTests)


if __name__ == '__main__':
    import runner

