# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA ..."""

import unittest
import numpy as N

# See other tests and test_procrust.py for some example on what to do ;)
from mvpa.algorithms.hyperalignment import Hyperalignment

# Somewhat slow but provides all needed ;)
from tests_warehouse import datasets, get_random_rotation

# if you need some classifiers
#from tests_warehouse_clfs import *

class HyperAlignmentTests(unittest.TestCase):


    def testBasicFunctioning(self):
        # get a dataset with some prominent trends in it
        ds4l = datasets['uni4large']
        # lets select for now only meaningful features
        ds_orig = ds4l[:, ds4l.a.nonbogus_features]
        n = 5 # # of datasets to generate
        Rs, dss_rotated, dss_rotated_clean, random_shifts, random_scales \
            = [], [], [], [], []
        # now lets compose derived datasets by using some random
        # rotation(s)
        for i in xrange(n):
            R = get_random_rotation(ds_orig.nfeatures)
            Rs.append(R)
            ds_ = ds_orig.copy()
            # reusing random data from dataset itself
            random_scales += [ds_orig.samples[i, 3] * 100]
            random_shifts += [ds_orig.samples[i+10] * 10]
            random_noise = ds4l.samples[:, ds4l.a.bogus_features[:4]]
            ds_.samples = N.dot(ds_orig.samples, R) * random_scales[-1] \
                          + random_shifts[-1]
            dss_rotated_clean.append(ds_)

            ds_ = ds_.copy()
            ds_.samples = ds_.samples + 0.1 * random_noise
            dss_rotated.append(ds_)

        ref_ds = 0                      # by default should be this one
        ha = Hyperalignment()
        # Lets test two scenarios -- in one with no noise -- we should get
        # close to perfect reconstruction.  If noise was added -- not so good
        for noisy, dss in ((False, dss_rotated_clean),
                           (True, dss_rotated)):
            mappers = ha(dss)
            # Map data back

            dss_clean_back = [m.forward(ds_)
                              for m, ds_ in zip(mappers, dss_rotated_clean)]

            ds_norm = N.linalg.norm(dss[ref_ds].samples)
            nddss = []
            ds_orig_Rref = N.dot(ds_orig.samples, Rs[ref_ds]) \
                           * random_scales[ref_ds] \
                           + random_shifts[ref_ds]
            for ds_back in dss_clean_back:
                dds = ds_back.samples - ds_orig_Rref
                ndds = N.linalg.norm(dds) / ds_norm
                nddss += [ndds]
            self.failUnless(N.all(ndds <= (1e-10, 1e-2)[int(noisy)]),
                msg="Should have reconstructed original dataset more or less."
                    " Got normed differences %s in %s case."
                    % (nddss, ('clean', 'noisy')[int(noisy)]))
        pass


    def _testOnSwaroopData(self):
        #
        print "Running swaroops test on data we don't have"
        pass



def suite():
    return unittest.makeSuite(HyperAlignmentTests)


if __name__ == '__main__':
    import runner

