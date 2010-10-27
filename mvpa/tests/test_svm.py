# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for SVM classifier"""

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.clfs.meta import ProxyClassifier
from mvpa.clfs.transerror import TransferError
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError

from tests_warehouse import pureMultivariateSignal
from tests_warehouse import *
from tests_warehouse_clfs import *

class SVMTests(unittest.TestCase):

#    @sweepargs(nl_clf=clfswh['non-linear', 'svm'] )
#    @sweepargs(nl_clf=clfswh['non-linear', 'svm'] )
    def testMultivariate(self):
        mv_perf = []
        mv_lin_perf = []
        uv_perf = []

        l_clf = clfswh['linear', 'svm'][0]
        nl_clf = clfswh['non-linear', 'svm'][0]

        #orig_keys = nl_clf.param._params.keys()
        #nl_param_orig = nl_clf.param._params.copy()

        # l_clf = LinearNuSVMC()

        # XXX ??? not sure what below meant and it is obsolete if
        # using SG... commenting out for now
        # for some reason order is not preserved thus dictionaries are not
        # the same any longer -- lets compare values
        #self.failUnlessEqual([nl_clf.param._params[k] for k in orig_keys],
        #                     [nl_param_orig[k] for k in orig_keys],
        #   msg="New instance mustn't override values in previously created")
        ## and keys separately
        #self.failUnlessEqual(set(nl_clf.param._params.keys()),
        #                     set(orig_keys),
        #   msg="New instance doesn't change set of parameters in original")

        # We must be able to deepcopy not yet trained SVMs now
        import mvpa.support.copy as copy
        try:
            nl_clf.untrain()
            nl_clf_copy = copy.deepcopy(nl_clf)
        except:
            self.fail(msg="Failed to deepcopy not-yet trained SVM %s" % nl_clf)

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


    # XXX for now works only with linear... think it through -- should
    # work non-linear, shouldn't it?
    # now all non-linear have C>0 thus skipped anyways

    # TODO: For some reason libsvm's weight assignment has no effect
    # as well -- need to be fixed :-/
    @sweepargs(clf=clfswh['svm', 'sg', '!regression', '!gnpp', '!meta'])
    def testCperClass(self, clf):
        try:
            if clf.C > 0:
                # skip those with fixed C
                return
        except:
            # classifier has no C
            return

        if clf.C < -5:
            # too soft margin helps to fight disbalance, thus skip
            # it in testing
            return
        #print clf
        ds = datasets['uni2small'].copy()
        ds__ = datasets['uni2small'].copy()
        #
        # ballanced set
        # Lets add a bit of noise to drive classifier nuts. same
        # should be done for disballanced set
        ds__.samples = ds__.samples + 0.5 * N.random.normal(size=(ds__.samples.shape))
        #
        # disballanced set
        # lets overpopulate label 0
        times = 10
        ds_ = ds.selectSamples(range(ds.nsamples) + range(ds.nsamples/2) * times)
        ds_.samples = ds_.samples + 0.7 * N.random.normal(size=(ds_.samples.shape))
        spl = ds_.samplesperlabel
        #print ds_.labels, ds_.chunks

        cve = CrossValidatedTransferError(TransferError(clf), NFoldSplitter(),
                                          enable_states='confusion')
        e = cve(ds__)
        if cfg.getboolean('tests', 'labile', default='yes'):
            # without disballance we should already have some hits
            self.failUnless(cve.confusion.stats["P'"][1] > 0)

        e = cve(ds_)
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(cve.confusion.stats["P'"][1] < 5,
                            msg="With disballance we should have almost no "
                            "hits. Got %f" % cve.confusion.stats["P'"][1])
            #print "D:", cve.confusion.stats["P'"][1], cve.confusion.stats['MCC'][1]

        # Set '1 C per label'
        # recreate cvte since previous might have operated on copies
        cve = CrossValidatedTransferError(TransferError(clf), NFoldSplitter(),
                                          enable_states='confusion')
        oldC = clf.C
        ratio = N.sqrt(float(spl[0])/spl[1])
        clf.C = (-1/ratio, -1*ratio)
        try:
            e_ = cve(ds_)
            # reassign C
            clf.C = oldC
        except:
            clf.C = oldC
            raise
        #print "B:", cve.confusion.stats["P'"][1], cve.confusion.stats['MCC'][1]
        if cfg.getboolean('tests', 'labile', default='yes'):
            # Finally test if we get any 'hit' for minor category. In the
            # classifier, which has way to 'ballance' should be non-0
            self.failUnless(cve.confusion.stats["P'"][1] > 0)


    def testSillyness(self):
        """Test if we raise exceptions on incorrect specifications
        """

        if externals.exists('libsvm') or externals.exists('shogun'):
            self.failUnlessRaises(TypeError, SVM,  C=1.0, nu=2.3)

        if externals.exists('libsvm'):
            self.failUnlessRaises(TypeError, libsvm.SVM,  C=1.0, nu=2.3)
            self.failUnlessRaises(TypeError, LinearNuSVMC, C=2.3)
            self.failUnlessRaises(TypeError, LinearCSVMC, nu=2.3)

        if externals.exists('shogun'):
            self.failUnlessRaises(TypeError, sg.SVM, C=10, kernel_type='RBF',
                                  coef0=3)

def suite():
    return unittest.makeSuite(SVMTests)


if __name__ == '__main__':
    import runner

