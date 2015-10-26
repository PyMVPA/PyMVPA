# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA MEG stuff"""

from os.path import join as pathjoin

from mvpa2.testing import *
from mvpa2 import pymvpa_dataroot
from mvpa2.misc.io.meg import TuebingenMEG

class MEGTests(unittest.TestCase):

    def test_tuebingen_meg(self):
        # Use this whenever we fully switch to nose to run tests
        #skip_if_no_external('gzip')
        if not externals.exists('gzip'):
            return

        meg = TuebingenMEG(pathjoin(pymvpa_dataroot, 'tueb_meg.dat.gz'))

        # check basics
        self.assertTrue(meg.channelids == ['BG1', 'MLC11', 'EEG02'])
        self.assertTrue(meg.ntimepoints == 814)
        self.assertTrue(meg.nsamples == 4)
        # check correct axis order (samples x channels x timepoints)
        self.assertTrue(meg.data.shape == (4, 3, 814))

        # check few values
        self.assertTrue(meg.data[0, 1, 4] == -2.318207982e-14)
        self.assertTrue(meg.data[3, 0, 808] == -4.30692876e-12)


def suite():  # pragma: no cover
    return unittest.makeSuite(MEGTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

