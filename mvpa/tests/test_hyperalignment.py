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
        ds_orig = datasets['uni4large']
        # lets select for now only meaningful features
        ds_orig = ds_orig[:, ds_orig.a.nonbogus_features]
        n = 5 # # of datasets to generate
        Rs, dss_rotated = [], []
        # now lets compose derived datasets by using some random
        # rotation(s)
        for _ in xrange(n):
            R = get_random_rotation(ds_orig.nfeatures)
            Rs.append(R)
            ds_ = ds_orig.copy()
            ds_.samples = N.dot(ds_orig.samples, R)
            dss_rotated.append(ds_)

        ha = Hyperalignment()
        mappers = ha(dss_rotated)
        # Map data back
        dss_back = [m.forward(ds_) for m, ds_ in zip(mappers, dss_rotated)]
        ds_orig_norm = N.linalg.norm(ds_orig.samples)
        nddss = []
        for ds_back in dss_back:
            dds = ds_back.samples - ds_orig.samples
            ndds = N.linalg.norm(dds) / ds_orig_norm
            nddss += [ndds]
        self.failUnless(N.all(ndds <= 1e-5),
            msg="Should have reconstructed original dataset more or less."
                " Got normed differences %s" % nddss)
        pass


    def _testOnSwaroopData(self):
        #
        print "Running swaroops test on data we don't have"
        pass



def suite():
    return unittest.makeSuite(HyperAlignmentTests)


if __name__ == '__main__':
    import runner

