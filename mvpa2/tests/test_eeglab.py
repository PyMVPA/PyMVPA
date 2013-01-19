# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA EEGLAB stuff"""

import os.path

from mvpa2.testing import *
from mvpa2 import pymvpa_dataroot
from mvpa2.datasets.eeglab import eeglab_dataset

import tempfile

class MEGTests(unittest.TestCase):

    def test_eeglab_dataset(self):
        data = '''    Fpz Cz Pz
0 30.2 20.3 20.2
2 1.5 1.6 1.72
0 1.1 1.2 1.3
2 2.5 2.6 -0.2
0 -2 -3 1
2 1 2 2.234'''

        # Use this whenever we fully switch to nose to run tests
        #skip_if_no_external('gzip')
        _, fn = tempfile.mkstemp('eeglab.txt', 'eeglab')
        with open(fn, 'w') as f:
            f.write(data)

        eeg = eeglab_dataset(fn)
        os.remove(fn)

        assert_array_equal(eeg.a['channels'].value,
                           np.asarray(['Fpz', 'Cz', 'Pz']))
        assert_array_equal(eeg.a['timepoints'].value,
                           np.asarray([0., 2.]))

        assert_true(eeg.nsamples == 3)
        assert_true(eeg.nfeatures == 6)

        assert_true(eeg.a['dt'].value == 2)
        assert_true(eeg.a['t0'].value == 0)

def suite():
    return unittest.makeSuite(MEGTests)


if __name__ == '__main__':
    import runner

