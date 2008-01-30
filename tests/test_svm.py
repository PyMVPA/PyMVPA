#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for SVM classifier"""

import unittest

import numpy as N
from sets import Set

from mvpa.clfs.svm import RbfNuSVMC, LinearNuSVMC
from mvpa.clfs.libsvm import svmc

from tests_warehouse import dumbFeatureDataset, pureMultivariateSignal, sweepargs
from tests_warehouse_clfs import clfs

class SVMTests(unittest.TestCase):

    @sweepargs(l_clf=clfs['LinearSVMC'])
    def testMultivariate(self, l_clf):
        mv_perf = []
        mv_lin_perf = []
        uv_perf = []

        nl_clf = RbfNuSVMC()
        orig_keys = nl_clf.param._params.keys()
        nl_param_orig = nl_clf.param._params.copy()

        # l_clf = LinearNuSVMC()

        # for some reason order is not preserved thus dictionaries are not
        # the same any longer -- lets compare values
        self.failUnlessEqual([nl_clf.param._params[k] for k in orig_keys],
                             [nl_param_orig[k] for k in orig_keys],
           msg="New instance mustn't override values in previously created")
        # and keys separately
        self.failUnlessEqual(Set(nl_clf.param._params.keys()),
                             Set(orig_keys),
           msg="New instance doesn't change set of parameters in original")

        # We must be able to deepcopy not yet trained SVMs now
        import copy
        nl_clf_copy = copy.deepcopy(nl_clf)

        try:
            nl_clf_copy = copy.deepcopy(nl_clf)
        except:
            self.fail(msg="Failed to deepcopy not-yet trained SVM")

        for i in xrange(20):
            train = pureMultivariateSignal( 20, 3 )
            test = pureMultivariateSignal( 20, 3 )

            # use non-linear CLF on 2d data
            nl_clf.train(train)
            p_mv = nl_clf.predict(test.samples)
            mv_perf.append(N.mean(p_mv==test.labels))

            # use linear CLF on 2d data
            l_clf.train(train)
            p_lin_mv = l_clf.predict(test.samples)
            mv_lin_perf.append(N.mean(p_lin_mv==test.labels))

            # use non-linear CLF on 1d data
            nl_clf.train(train.selectFeatures([0]))
            p_uv = nl_clf.predict(test.selectFeatures([0]).samples)
            uv_perf.append(N.mean(p_uv==test.labels))

        mean_mv_perf = N.mean(mv_perf)
        mean_mv_lin_perf = N.mean(mv_lin_perf)
        mean_uv_perf = N.mean(uv_perf)

        # non-linear CLF has to be close to perfect
        self.failUnless( mean_mv_perf > 0.9 )
        # linear CLF cannot learn this problem!
        self.failUnless( mean_mv_perf > mean_mv_lin_perf )
        # univariate has insufficient information
        self.failUnless( mean_uv_perf < mean_mv_perf )


#    def testFeatureBenchmark(self):
#        pat = dumbFeatureDataset()
#        clf = SVM()
#        clf.train(pat)
#        rank = clf.getFeatureBenchmark()
#
#        # has to be 1d array
#        self.failUnless(len(rank.shape) == 1)
#
#        # has to be one value per feature
#        self.failUnless(len(rank) == pat.nfeatures)
#
#        # first feature is discriminative, second is not
#        self.failUnless(rank[0] > rank[1])
#

def suite():
    return unittest.makeSuite(SVMTests)


if __name__ == '__main__':
    import test_runner

