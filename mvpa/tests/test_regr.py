# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA Regressions"""

from mvpa.testing import *
from mvpa.testing.clfs import *
from mvpa.testing.datasets import dataset_wizard, datasets

from mvpa.datasets.splitters import NFoldSplitter, OddEvenSplitter

from mvpa.misc.errorfx import corr_error

from mvpa.clfs.meta import SplitClassifier
from mvpa.clfs.transerror import TransferError
from mvpa.misc.exceptions import UnknownStateError
from mvpa.misc.attrmap import AttributeMap
from mvpa.mappers.fx import mean_sample
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError


class RegressionsTests(unittest.TestCase):

    @sweepargs(ml=clfswh['regression_based'] + regrswh[:])
    def test_non_regressions(self, ml):
        """Test If binary regression-based  classifiers have proper tag
        """
        self.failUnless(('binary' in ml.__tags__) != ml.__is_regression__,
            msg="Inconsistent tagging with binary and regression features"
                " detected in %s having %r" % (ml, ml.__tags__))

    @sweepargs(regr=regrswh['regression'])
    def test_regressions(self, regr):
        """Simple tests on regressions
        """
        ds = datasets['chirp_linear']
        # we want numeric labels to maintain the previous behavior, especially
        # since we deal with regressions here
        ds.sa.targets = AttributeMap().to_numeric(ds.targets)

        cve = CrossValidatedTransferError(
            TransferError(regr),
            splitter=NFoldSplitter(),
            postproc=mean_sample(),
            enable_ca=['training_confusion', 'confusion'])
        # check the default
        self.failUnless(cve.transerror.errorfx is corr_error)
        corr = np.asscalar(cve(ds).samples)

        # Our CorrErrorFx should never return NaN
        self.failUnless(not np.isnan(corr))
        self.failUnless(corr == cve.ca.confusion.stats['CCe'])

        splitregr = SplitClassifier(
            regr, splitter=OddEvenSplitter(),
            enable_ca=['training_confusion', 'confusion'])
        splitregr.train(ds)
        split_corr = splitregr.ca.confusion.stats['CCe']
        split_corr_tr = splitregr.ca.training_confusion.stats['CCe']

        for confusion, error in (
            (cve.ca.confusion, corr),
            (splitregr.ca.confusion, split_corr),
            (splitregr.ca.training_confusion, split_corr_tr),
            ):
            #TODO: test confusion statistics
            # Part of it for now -- CCe
            for conf in confusion.summaries:
                stats = conf.stats
                if cfg.getboolean('tests', 'labile', default='yes'):
                    self.failUnless(stats['CCe'] < 0.5)
                self.failUnlessEqual(stats['CCe'], stats['Summary CCe'])

            s0 = confusion.as_string(short=True)
            s1 = confusion.as_string(short=False)

            for s in [s0, s1]:
                self.failUnless(len(s) > 10,
                                msg="We should get some string representation "
                                "of regression summary. Got %s" % s)
            if cfg.getboolean('tests', 'labile', default='yes'):
                self.failUnless(error < 0.2,
                            msg="Regressions should perform well on a simple "
                            "dataset. Got correlation error of %s " % error)

            # Test access to summary statistics
            # YOH: lets start making testing more reliable.
            #      p-value for such accident to have is verrrry tiny,
            #      so if regression works -- it better has at least 0.5 ;)
            #      otherwise fix it! ;)
            # YOH: not now -- issues with libsvr in SG and linear kernel
            if cfg.getboolean('tests', 'labile', default='yes'):
                self.failUnless(confusion.stats['CCe'] < 0.5)

        # just to check if it works fine
        split_predictions = splitregr.predict(ds.samples)

        # To test basic plotting
        #import pylab as pl
        #cve.confusion.plot()
        #pl.show()

    @sweepargs(clf=clfswh['regression'])
    def test_regressions_classifiers(self, clf):
        """Simple tests on regressions being used as classifiers
        """
        # check if we get values set correctly
        clf.ca.change_temporarily(enable_ca=['estimates'])
        self.failUnlessRaises(UnknownStateError, clf.ca['estimates']._get)
        cv = CrossValidatedTransferError(
            TransferError(clf),
            NFoldSplitter(),
            enable_ca=['confusion', 'training_confusion'])
        ds = datasets['uni2small'].copy()
        # we want numeric labels to maintain the previous behavior, especially
        # since we deal with regressions here
        ds.sa.targets = AttributeMap().to_numeric(ds.targets)
        cverror = cv(ds)

        self.failUnless(len(clf.ca.estimates) == ds[ds.chunks == 1].nsamples)
        clf.ca.reset_changed_temporarily()


    @sweepargs(regr=regrswh['regression', 'has_sensitivity', '!gpr'])
    def test_sensitivities(self, regr):
        """Test "sensitivities" provided by regressions

        Inspired by a snippet leading to segfault from Daniel Kimberg

        lead to segfaults due to inappropriate access of SVs thinking
        that it is a classification problem (libsvm keeps SVs at None
        for those, although reports nr_class to be 2.
        """
        myds = dataset_wizard(samples=np.random.normal(size=(10,5)),
                              targets=np.random.normal(size=10))
        sa = regr.get_sensitivity_analyzer()
        #try:
        if True:
            res = sa(myds)
        #except Exception, e:
        #    self.fail('Failed to obtain a sensitivity due to %r' % (e,))
        self.failUnless(res.shape == (1, myds.nfeatures))
        # TODO: extend the test -- checking for validity of sensitivities etc


def suite():
    return unittest.makeSuite(RegressionsTests)


if __name__ == '__main__':
    import runner
