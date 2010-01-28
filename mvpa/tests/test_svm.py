# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for SVM classifier"""

from sets import Set

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.datasets.miscfx import get_nsamples_per_attr
from mvpa.clfs.meta import ProxyClassifier
from mvpa.clfs.transerror import TransferError
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError

from tests_warehouse import pureMultivariateSignal
from tests_warehouse import *
from tests_warehouse_clfs import *

class SVMTests(unittest.TestCase):

#    @sweepargs(nl_clf=clfswh['non-linear', 'svm'] )
#    @sweepargs(nl_clf=clfswh['non-linear', 'svm'] )
    def test_multivariate(self):
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
        #self.failUnlessEqual(Set(nl_clf.param._params.keys()),
        #                     Set(orig_keys),
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
            nl_clf.train(train[:, 0])
            p_uv = nl_clf.predict(test[:, 0].samples)
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
    #     work non-linear, shouldn't it?
    @sweepargs(clf=clfswh['svm', 'linear', '!regression', '!gnpp', '!meta'])
    def test_cper_class(self, clf):
        if not (clf.params.has_key('C')):
            # skip those without C
            return

        ds = datasets['uni2medium'].copy()
        ds__ = datasets['uni2medium'].copy()
        #
        # ballanced set
        # Lets add a bit of noise to drive classifier nuts. same
        # should be done for disballanced set
        ds__.samples = ds__.samples + \
                       0.5 * N.random.normal(size=(ds__.samples.shape))
        #
        # disballanced set
        # lets overpopulate label 0
        times = 20
        ds_ = ds[(range(ds.nsamples) + range(ds.nsamples/2) * times)]
        ds_.samples = ds_.samples + \
                      0.5 * N.random.normal(size=(ds_.samples.shape))
        spl = get_nsamples_per_attr(ds_, 'labels') #_.samplesperlabel
        #print ds_.labels, ds_.chunks

        cve = CrossValidatedTransferError(TransferError(clf), NFoldSplitter(),
                                          enable_states='confusion')
        # on balanced
        e = cve(ds__)
        tpr_1 = cve.states.confusion.stats["TPR"][1]

        # on disbalanced
        e = cve(ds_)
        tpr_2 =  cve.states.confusion.stats["TPR"][1]

        # Set '1 C per label'
        # recreate cvte since previous might have operated on copies
        cve = CrossValidatedTransferError(TransferError(clf), NFoldSplitter(),
                                          enable_states='confusion')
        oldC = clf.params.C
        # TODO: provide clf.params.C not with a tuple but dictionary
        #       with C per label (now order is deduced in a cruel way)
        ratio = N.sqrt(float(spl[ds_.UL[0]])/spl[ds_.UL[1]])
        clf.params.C = (-1/ratio, -1*ratio)
        try:
            # on disbalanced but with balanced C
            e_ = cve(ds_)
            # reassign C
            clf.params.C = oldC
        except:
            clf.params.C = oldC
            raise
        tpr_3 = cve.states.confusion.stats["TPR"][1]

        # Actual tests
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(tpr_1 > 0.25,
                            msg="Without disballance we should have some "
                            "hits, but got TPR=%.3f" % tpr_1)

            self.failUnless(tpr_2 < 0.25,
                            msg="With disballance we should have almost no "
                            "hits for minor, but got TPR=%.3f" % tpr_2)

            self.failUnless(tpr_3 > 0.25,
                            msg="With disballanced data but ratio-based Cs "
                            "we should have some hits for minor, but got "
                            "TPR=%.3f" % tpr_3)



    def test_sillyness(self):
        """Test if we raise exceptions on incorrect specifications
        """

        if externals.exists('libsvm') or externals.exists('shogun'):
            self.failUnlessRaises(TypeError, libsvm.SVM, C=1.0, nu=2.3)

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

