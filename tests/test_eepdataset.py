#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA EEP dataset"""

import unittest
import os.path
import numpy as N

#from mvpa.datasets.eepdataset import *
from mvpa.misc.eepbin import EEPBin


class EEPDatasetTests(unittest.TestCase):

    def testLoad(self):
        pass


    def testEEPBin(self):
        eb = EEPBin(os.path.join('..', 'data', 'eep.bin'))

        self.failUnless(eb.nchannels == 32)
        self.failUnless(eb.ntrials == 2)
        self.failUnless(eb.nsamples == 4)
        self.failUnless(eb.t0 - eb.dt < 0.00000001)
        self.failUnless(len(eb.channels) == 32)
        self.failUnless(eb.data.shape == (2, 32, 4))


def suite():
    return unittest.makeSuite(EEPDatasetTests)


if __name__ == '__main__':
    import runner

