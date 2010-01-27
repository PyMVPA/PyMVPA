# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA MEG stuff"""

import unittest
import os.path

from mvpa import pymvpa_dataroot
from mvpa.base import externals
from mvpa.misc.io.meg import TuebingenMEG

class MEGTests(unittest.TestCase):

    def test_tuebingen_meg(self):
        if not externals.exists('gzip'):
            return
        meg = TuebingenMEG(os.path.join(pymvpa_dataroot, 'tueb_meg.dat.gz'))

        # check basics
        self.failUnless(meg.channelids == ['BG1', 'MLC11', 'EEG02'])
        self.failUnless(meg.ntimepoints == 814)
        self.failUnless(meg.nsamples == 4)
        # check correct axis order (samples x channels x timepoints)
        self.failUnless(meg.data.shape == (4, 3, 814))

        # check few values
        self.failUnless(meg.data[0, 1, 4] == -2.318207982e-14)
        self.failUnless(meg.data[3, 0, 808] == -4.30692876e-12)


def suite():
    return unittest.makeSuite(MEGTests)


if __name__ == '__main__':
    import runner

