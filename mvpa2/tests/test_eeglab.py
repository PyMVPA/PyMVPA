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
        fd, fn = tempfile.mkstemp('eeglab.txt', 'eeglab'); os.close(fd)
        with open(fn, 'w') as f:
            f.write(data)

        eeg = eeglab_dataset(fn)
        os.remove(fn)

        assert_array_equal(set(eeg.channelids), set(['Fpz', 'Cz', 'Pz']))
        assert_array_equal(eeg.timepoints, np.asarray([0., 2.]))

        assert_equal(eeg.nchannels, 3)
        assert_equal(eeg.ntimepoints, 2)

        assert_equal(eeg.nsamples, 3)
        assert_equal(eeg.nfeatures, 6)

        assert_equal(eeg.dt, 2)
        assert_equal(eeg.t0, 0)

        assert_array_equal(eeg.samples[0, 3], 1.5)

        sel_time = eeg[:, eeg.get_features_by_timepoints(lambda x:x > 0)]
        assert_equal(sel_time.ntimepoints, 1)
        assert_equal(sel_time.t0, 2)

        sel_chan = eeg[:, eeg.get_features_by_channelids(['Fpz', 'Pz'])]
        assert_equal(sel_chan.nchannels, 2)
        assert_array_equal(sel_chan.channelids, ['Fpz', 'Pz'])


def suite():  # pragma: no cover
    return unittest.makeSuite(MEGTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
