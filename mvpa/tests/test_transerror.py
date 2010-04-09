# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA classifier cross-validation"""

import unittest
import numpy as np

from mvpa.support.copy import copy

from mvpa.base import externals, warning
from mvpa.datasets.splitters import OddEvenSplitter

from mvpa.clfs.meta import MulticlassClassifier
from mvpa.clfs.transerror import \
     TransferError, ConfusionMatrix, ConfusionBasedError
from mvpa.algorithms.cvtranserror import CrossValidatedTransferError

from mvpa.clfs.stats import MCNullDist

from mvpa.misc.exceptions import UnknownStateError
from mvpa.mappers.fx import mean_sample

from mvpa.testing import *
from mvpa.testing.datasets import datasets
from mvpa.testing.clfs import *

class ErrorsTests(unittest.TestCase):

    def test_confusion_matrix(self):
        data = np.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = [1,1,1,2,2,2,3,3,3]
        regl = [1,2,1,2,2,2,3,2,1]
        correct_cm = [[2,0,1],[1,3,1],[0,0,1]]
        # Check if we are ok with any input type - either list, or np.array, or tuple
        for t in [reg, tuple(reg), list(reg), np.array(reg)]:
            for p in [regl, tuple(regl), list(regl), np.array(regl)]:
                cm = ConfusionMatrix(targets=t, predictions=p)
                # check table content
                self.failUnless((cm.matrix == correct_cm).all())


        # Do a bit more thorough checking
        cm = ConfusionMatrix()
        self.failUnlessRaises(ZeroDivisionError, lambda x:x.percent_correct, cm)
        """No samples -- raise exception"""

        cm.add(reg, regl)

        self.failUnlessEqual(len(cm.sets), 1,
            msg="Should have a single set so far")
        self.failUnlessEqual(cm.matrix.shape, (3,3),
            msg="should be square matrix (len(reglabels) x len(reglabels)")

        self.failUnlessRaises(ValueError, cm.add, reg, np.array([1]))
        """ConfusionMatrix must complaint if number of samples different"""

        # check table content
        self.failUnless((cm.matrix == correct_cm).all())

        # lets add with new labels (not yet known)
        cm.add(reg, np.array([1,4,1,2,2,2,4,2,1]))

        self.failUnlessEqual(cm.labels, [1,2,3,4],
                             msg="We should have gotten 4th label")

        matrices = cm.matrices          # separate CM per each given set
        self.failUnlessEqual(len(matrices), 2,
                             msg="Have gotten two splits")

        self.failUnless((matrices[0].matrix + matrices[1].matrix == cm.matrix).all(),
                        msg="Total votes should match the sum across split CMs")

        # check pretty print
        # just a silly test to make sure that printing works
        self.failUnless(len(cm.as_string(
            header=True, summary=True,
            description=True))>100)
        self.failUnless(len(str(cm))>100)
        # and that it knows some parameters for printing
        self.failUnless(len(cm.as_string(summary=True,
                                       header=False))>100)

        # lets check iadd -- just itself to itself
        cm += cm
        self.failUnlessEqual(len(cm.matrices), 4, msg="Must be 4 sets now")

        # lets check add -- just itself to itself
        cm2 = cm + cm
        self.failUnlessEqual(len(cm2.matrices), 8, msg="Must be 8 sets now")
        self.failUnlessEqual(cm2.percent_correct, cm.percent_correct,
                             msg="Percent of corrrect should remain the same ;-)")

        self.failUnlessEqual(cm2.error, 1.0-cm.percent_correct/100.0,
                             msg="Test if we get proper error value")


    def test_degenerate_confusion(self):
        # We must not just puke -- some testing splits might
        # have just a single target label

        for orig in ([1], [1, 1], [0], [0, 0]):
            cm = ConfusionMatrix(targets=orig, predictions=orig, estimates=orig)

            scm = str(cm)
            self.failUnless(cm.stats['ACC%'] == 100)


    def test_confusion_matrix_acc(self):
        reg  = [0,0,1,1]
        regl = [1,0,1,0]
        cm = ConfusionMatrix(targets=reg, predictions=regl)
        self.failUnless('ACC%         50' in str(cm))
        skip_if_no_external('scipy')
        self.failUnless(cm.stats['CHI^2'] == (0., 1.))


    def test_confusion_matrix_with_mappings(self):
        data = np.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = [1,1,1,2,2,2,3,3,3]
        regl = [1,2,1,2,2,2,3,2,1]
        correct_cm = [[2,0,1], [1,3,1], [0,0,1]]
        lm = {'apple':1, 'orange':2, 'shitty apple':1, 'candy':3}
        cm = ConfusionMatrix(targets=reg, predictions=regl,
                             labels_map=lm)
        # check table content
        self.failUnless((cm.matrix == correct_cm).all())
        # assure that all labels are somewhere listed ;-)
        s = str(cm)
        for l in lm.keys():
            self.failUnless(l in s)


    @sweepargs(l_clf=clfswh['linear', 'svm'])
    def test_confusion_based_error(self, l_clf):
        train = datasets['uni2medium_train']
        # to check if we fail to classify for 3 labels
        test3 = datasets['uni3medium_train']
        err = ConfusionBasedError(clf=l_clf)
        terr = TransferError(clf=l_clf)

        self.failUnlessRaises(UnknownStateError, err, None)
        """Shouldn't be able to access the state yet"""

        l_clf.train(train)
        e, te = err(None), terr(train)
        self.failUnless(abs(e-te) < 1e-10,
            msg="ConfusionBasedError (%.2g) should be equal to TransferError "
                "(%.2g) on traindataset" % (e, te))

        # this will print nasty WARNING but it is ok -- it is just checking code
        # NB warnings are not printed while doing whole testing
        warning("Don't worry about the following warning.")
        self.failIf(terr(test3) is None)

        # try copying the beast
        terr_copy = copy(terr)


    @sweepargs(l_clf=clfswh['linear', 'svm'])
    def test_null_dist_prob(self, l_clf):
        train = datasets['uni2medium']

        num_perm = 10
        # define class to estimate NULL distribution of errors
        # use left tail of the distribution since we use MeanMatchFx as error
        # function and lower is better
        terr = TransferError(
            clf=l_clf,
            null_dist=MCNullDist(permutations=num_perm,
                                 tail='left'))

        # check reasonable error range
        err = terr(train, train)
        self.failUnless(err < 0.4)

        # Lets do the same for CVTE
        cvte = CrossValidatedTransferError(
            TransferError(clf=l_clf),
            OddEvenSplitter(),
            null_dist=MCNullDist(permutations=num_perm,
                                 tail='left',
                                 enable_ca=['dist_samples']),
            postproc=mean_sample())
        cv_err = cvte(train)

        # check that the result is highly significant since we know that the
        # data has signal
        null_prob = terr.ca.null_prob
        if cfg.getboolean('tests', 'labile', default='yes'):
            self.failUnless(null_prob <= 0.1,
                msg="Failed to check that the result is highly significant "
                    "(got %f) since we know that the data has signal"
                    % null_prob)

            self.failUnless(cvte.ca.null_prob <= 0.1,
                msg="Failed to check that the result is highly significant "
                    "(got p(cvte)=%f) since we know that the data has signal"
                    % cvte.ca.null_prob)

            # and we should be able to access the actual samples of the distribution
            self.failUnlessEqual(len(cvte.null_dist.ca.dist_samples),
                                 num_perm)


    @sweepargs(l_clf=clfswh['linear', 'svm'])
    def test_per_sample_error(self, l_clf):
        train = datasets['uni2medium']
        train.init_origids('samples')
        terr = TransferError(clf=l_clf, enable_ca=['samples_error'])
        err = terr(train, train)
        se = terr.ca.samples_error

        # one error per sample
        self.failUnless(len(se) == train.nsamples)
        # for this simple test it can only be correct or misclassified
        # (boolean)
        self.failUnless(
            np.sum(np.array(se.values(), dtype='float') \
                  - np.array(se.values(), dtype='b')) == 0)


    @sweepargs(clf=clfswh['multiclass'])
    def test_auc(self, clf):
        """Test AUC computation
        """
        if isinstance(clf, MulticlassClassifier):
            # TODO: handle those values correctly
            return
        clf.ca.change_temporarily(enable_ca = ['estimates'])
        # uni2 dataset with reordered labels
        ds2 = datasets['uni2small'].copy()
        # revert labels
        ds2.sa['targets'].value = ds2.targets[::-1].copy()
        # same with uni3
        ds3 = datasets['uni3small'].copy()
        ul = ds3.sa['targets'].unique
        nl = ds3.targets.copy()
        for l in xrange(3):
            nl[ds3.targets == ul[l]] = ul[(l+1)%3]
        ds3.sa.targets = nl
        for ds in [datasets['uni2small'], ds2,
                   datasets['uni3small'], ds3]:
            cv = CrossValidatedTransferError(
                TransferError(clf),
                OddEvenSplitter(),
                enable_ca=['confusion', 'training_confusion'])
            cverror = cv(ds)
            stats = cv.ca.confusion.stats
            Nlabels = len(ds.uniquetargets)
            # so we at least do slightly above chance
            self.failUnless(stats['ACC'] > 1.2 / Nlabels)
            auc = stats['AUC']
            if (Nlabels == 2) or (Nlabels > 2 and auc[0] is not np.nan):
                mauc = np.min(stats['AUC'])
                if cfg.getboolean('tests', 'labile', default='yes'):
                    self.failUnless(mauc > 0.55,
                         msg='All AUCs must be above chance. Got minimal '
                             'AUC=%.2g among %s' % (mauc, stats['AUC']))
        clf.ca.reset_changed_temporarily()




    def test_confusion_plot(self):
        """Basic test of confusion plot

        Based on existing cell dataset results.

        Let in for possible future testing, but is not a part of the
        unittests suite
        """
        #from matplotlib import rc as rcmpl
        #rcmpl('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans']})
        ##rcmpl('text', usetex=True)
        ##rcmpl('font',  family='sans', style='normal', variant='normal',
        ##   weight='bold',  stretch='normal', size='large')
        #import numpy as np
        #from mvpa.clfs.transerror import \
        #     TransferError, ConfusionMatrix, ConfusionBasedError

        array = np.array
        uint8 = np.uint8
        sets = [
           (array([47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44], dtype=uint8),
            array([40, 39, 47, 43, 45, 41, 44, 41, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 38, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 46,
                 45, 38, 44, 39, 46, 38, 39, 39, 38, 43, 45, 41, 44, 40, 46, 42, 38,
                 40, 47, 43, 45, 41, 44, 40, 46, 42, 38, 39, 40, 43, 45, 41, 44, 39,
                 46, 42, 47, 38, 38, 43, 45, 41, 44, 38, 46, 42, 47, 38, 39, 43, 45,
                 41, 44, 40, 46, 42, 47, 38, 38, 43, 45, 41, 44, 40, 46, 42, 47, 38,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 47, 43, 45, 41, 44, 40, 46,
                 42, 47, 38, 38, 43, 45, 41, 44, 40, 46, 42, 39, 39, 38, 43, 45, 41,
                 44, 47, 46, 42, 47, 38, 39, 43, 45, 40, 44, 40, 46, 42, 47, 39, 40,
                 43, 45, 41, 44, 38, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 41,
                 47, 39, 38, 46, 45, 41, 44, 40, 46, 42, 40, 38, 38, 43, 45, 41, 44,
                 40, 45, 42, 47, 39, 39, 43, 45, 41, 44, 38, 46, 42, 47, 38, 42, 43,
                 45, 41, 44, 39, 46, 42, 39, 39, 39, 47, 45, 41, 44], dtype=uint8)),
           (array([40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43], dtype=uint8),
            array([40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 47, 46, 42, 47, 39, 40, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 39, 46, 42, 47, 47, 47, 43, 45, 41, 44, 40,
                 46, 42, 43, 39, 38, 43, 45, 41, 44, 38, 38, 42, 38, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 40, 38, 43, 45, 41, 44, 40, 40, 42, 47, 40,
                 40, 43, 45, 41, 44, 38, 38, 42, 47, 38, 38, 47, 45, 41, 44, 40, 46,
                 42, 47, 39, 40, 43, 45, 41, 44, 40, 46, 42, 47, 47, 39, 43, 45, 41,
                 44, 40, 46, 42, 39, 39, 42, 43, 45, 41, 44, 40, 46, 42, 47, 39, 39,
                 43, 45, 41, 44, 47, 46, 42, 40, 39, 39, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 40, 44, 40, 46, 42, 47, 39, 39, 43, 45, 41, 44,
                 38, 46, 42, 47, 39, 39, 43, 45, 41, 44, 40, 46, 46, 47, 38, 39, 43,
                 45, 41, 44, 40, 46, 42, 47, 38, 39, 43, 45, 41, 44, 40, 46, 42, 39,
                 39, 38, 47, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43], dtype=uint8)),
           (array([45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47], dtype=uint8),
            array([45, 41, 44, 40, 46, 42, 47, 39, 46, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 39, 43, 45, 41, 44, 38, 46, 42, 47, 38, 39, 43, 45, 41, 44, 40,
                 46, 42, 47, 38, 39, 43, 45, 41, 44, 40, 46, 42, 47, 39, 43, 43, 45,
                 40, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 47,
                 40, 43, 45, 41, 44, 40, 47, 42, 38, 47, 38, 43, 45, 41, 44, 40, 40,
                 42, 47, 39, 39, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 38, 46, 42, 47, 39, 39, 43, 45, 41, 44, 40, 46, 42, 47, 40, 38,
                 43, 45, 41, 44, 40, 46, 38, 38, 39, 38, 43, 45, 41, 44, 39, 46, 42,
                 47, 40, 39, 43, 45, 38, 44, 38, 46, 42, 47, 47, 40, 43, 45, 41, 44,
                 40, 40, 42, 47, 40, 38, 43, 39, 41, 44, 41, 46, 42, 39, 39, 38, 38,
                 45, 41, 44, 38, 46, 40, 46, 46, 46, 43, 45, 38, 44, 40, 46, 42, 39,
                 39, 45, 43, 45, 41, 44, 38, 46, 42, 38, 39, 39, 43, 45, 41, 38, 40,
                 46, 42, 47, 38, 39, 43, 45, 41, 44, 40, 46, 42, 40], dtype=uint8)),
           (array([39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40], dtype=uint8),
            array([39, 38, 43, 45, 41, 44, 40, 46, 38, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 41, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 38, 43, 47, 38, 38, 43, 45, 41, 44, 39, 46, 42, 39, 39,
                 38, 43, 45, 41, 44, 43, 46, 42, 47, 39, 39, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 40, 43, 45, 41, 44, 40, 46, 42, 39, 38, 38, 43, 45, 40,
                 44, 47, 46, 38, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 38, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 38, 39, 38, 43, 45, 47, 44, 45, 46, 42,
                 38, 39, 41, 43, 45, 41, 44, 38, 38, 42, 39, 40, 40, 43, 45, 41, 39,
                 40, 46, 42, 47, 39, 40, 43, 45, 41, 44, 40, 47, 42, 47, 38, 38, 43,
                 45, 41, 44, 47, 46, 42, 47, 40, 47, 43, 45, 41, 44, 40, 46, 42, 47,
                 38, 39, 43, 45, 41, 44, 40, 46, 42, 39, 38, 43, 45, 46, 44, 38, 46,
                 42, 47, 38, 44, 43, 45, 42, 44, 41, 46, 42, 47, 47, 38, 43, 45, 41,
                 44, 38, 46, 42, 39, 39, 38, 43, 45, 41, 44, 40], dtype=uint8)),
           (array([46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45], dtype=uint8),
            array([46, 42, 39, 38, 38, 43, 45, 41, 44, 40, 46, 42, 47, 47, 42, 43, 45,
                 42, 44, 40, 46, 42, 38, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 47,
                 40, 43, 45, 41, 44, 41, 46, 42, 38, 39, 38, 43, 45, 41, 44, 38, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 46, 38, 38, 43, 45, 41,
                 44, 39, 46, 42, 47, 39, 40, 43, 45, 41, 44, 40, 46, 42, 47, 39, 39,
                 43, 45, 41, 44, 40, 47, 42, 47, 38, 39, 43, 45, 41, 44, 39, 46, 42,
                 47, 39, 46, 43, 45, 41, 44, 39, 46, 42, 39, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 38, 38, 43, 45, 41, 44, 40, 46, 42, 39, 39, 38, 43,
                 45, 41, 44, 40, 38, 42, 46, 39, 38, 43, 45, 41, 44, 38, 46, 42, 46,
                 46, 38, 43, 45, 41, 44, 40, 46, 42, 47, 47, 38, 38, 45, 41, 44, 38,
                 38, 42, 43, 39, 40, 43, 45, 41, 44, 38, 46, 42, 47, 38, 39, 47, 45,
                 46, 44, 40, 46, 42, 47, 40, 38, 43, 45, 41, 44, 40, 46, 42, 47, 40,
                 38, 43, 45, 41, 44, 38, 46, 42, 38, 39, 38, 47, 45], dtype=uint8)),
           (array([41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39], dtype=uint8),
            array([41, 44, 38, 46, 42, 47, 39, 47, 40, 45, 41, 44, 40, 46, 42, 38, 40,
                 38, 43, 45, 41, 44, 40, 46, 42, 38, 38, 38, 43, 45, 41, 44, 46, 38,
                 42, 40, 38, 39, 43, 45, 41, 44, 41, 46, 42, 47, 47, 38, 43, 45, 41,
                 44, 40, 46, 42, 38, 39, 39, 43, 45, 41, 44, 38, 46, 42, 47, 43, 39,
                 43, 45, 41, 44, 40, 46, 42, 38, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 40, 39, 38, 43, 45, 41, 44, 38, 46, 42, 39, 39, 39, 43, 45, 41, 44,
                 40, 46, 42, 39, 38, 47, 43, 45, 38, 44, 40, 38, 42, 47, 38, 38, 43,
                 45, 41, 44, 40, 38, 46, 47, 38, 38, 43, 45, 41, 44, 41, 46, 42, 40,
                 38, 38, 40, 45, 41, 44, 40, 40, 42, 43, 38, 40, 43, 39, 41, 44, 40,
                 40, 42, 47, 38, 46, 43, 45, 41, 44, 47, 41, 42, 43, 40, 47, 43, 45,
                 41, 44, 41, 38, 42, 40, 39, 40, 43, 45, 41, 44, 39, 43, 42, 47, 39,
                 40, 43, 45, 41, 44, 42, 46, 42, 47, 40, 46, 43, 45, 41, 44, 38, 46,
                 42, 47, 47, 38, 43, 45, 41, 44, 40, 38, 39, 47, 38], dtype=uint8)),
           (array([38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46], dtype=uint8),
            array([39, 43, 45, 41, 44, 40, 46, 42, 47, 38, 38, 43, 45, 41, 44, 41, 46,
                 42, 47, 47, 39, 43, 45, 41, 44, 40, 46, 42, 47, 38, 39, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 40, 43, 45, 41, 44, 40, 46, 42, 47, 45, 38,
                 43, 45, 41, 44, 38, 46, 42, 47, 38, 39, 43, 45, 41, 44, 40, 46, 42,
                 39, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 40, 39, 43, 45, 41, 44, 40, 39, 42, 40, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 38, 46, 42, 39,
                 39, 47, 43, 45, 41, 44, 40, 46, 42, 47, 39, 39, 43, 45, 41, 44, 40,
                 46, 42, 46, 47, 39, 47, 45, 41, 44, 40, 46, 42, 47, 39, 39, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 38, 46, 42, 47, 39,
                 38, 43, 45, 42, 44, 39, 47, 42, 39, 39, 47, 43, 47, 40, 44, 40, 46,
                 42, 39, 39, 38, 39, 45, 41, 44, 40, 46, 42, 47, 38, 38, 43, 45, 41,
                 44, 46, 38, 42, 47, 39, 43, 43, 45, 41, 44, 40, 46], dtype=uint8)),
           (array([42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42,
                 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44,
                 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43,
                 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47,
                 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40,
                 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45,
                 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39,
                 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46,
                 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 40, 46, 42, 47, 39, 38, 43, 45, 41, 44, 40, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 40, 46, 42, 47, 39, 38, 43, 45], dtype=uint8),
            array([42, 38, 38, 40, 43, 45, 41, 44, 39, 46, 42, 47, 39, 38, 43, 45, 41,
                 44, 39, 38, 42, 47, 41, 40, 43, 45, 41, 44, 40, 41, 42, 47, 38, 46,
                 43, 45, 41, 44, 41, 41, 42, 40, 39, 39, 43, 45, 41, 44, 46, 45, 42,
                 39, 39, 40, 43, 45, 41, 44, 40, 46, 42, 40, 44, 38, 43, 41, 41, 44,
                 39, 46, 42, 39, 39, 39, 43, 45, 41, 44, 40, 43, 42, 47, 39, 39, 43,
                 45, 41, 44, 40, 47, 42, 38, 46, 39, 47, 45, 41, 44, 39, 46, 42, 47,
                 41, 38, 43, 45, 41, 44, 42, 46, 42, 46, 39, 38, 43, 45, 41, 44, 41,
                 46, 42, 46, 39, 38, 43, 45, 41, 44, 40, 46, 42, 38, 38, 38, 43, 45,
                 41, 44, 38, 46, 42, 39, 40, 43, 43, 45, 41, 44, 39, 38, 40, 40, 38,
                 38, 43, 45, 41, 44, 41, 40, 42, 39, 39, 39, 43, 45, 41, 44, 40, 46,
                 42, 47, 40, 40, 43, 45, 41, 44, 40, 46, 42, 41, 39, 39, 43, 45, 41,
                 44, 40, 38, 42, 40, 39, 46, 43, 45, 41, 44, 47, 46, 42, 47, 39, 38,
                 43, 45, 41, 44, 41, 46, 42, 43, 39, 39, 43, 45], dtype=uint8))]
        labels_map = {'12kHz': 40,
                      '20kHz': 41,
                      '30kHz': 42,
                      '3kHz': 38,
                      '7kHz': 39,
                      'song1': 43,
                      'song2': 44,
                      'song3': 45,
                      'song4': 46,
                      'song5': 47}
        try:
            cm = ConfusionMatrix(sets=sets, labels_map=labels_map)
        except:
            self.fail()
        self.failUnless('3kHz / 38' in cm.as_string())

        if externals.exists("pylab plottable"):
            import pylab as pl
            pl.figure()
            labels_order = ("3kHz", "7kHz", "12kHz", "20kHz","30kHz", None,
                            "song1","song2","song3","song4","song5")
            #print cm
            #fig, im, cb = cm.plot(origin='lower', labels=labels_order)
            fig, im, cb = cm.plot(labels=labels_order[1:2] + labels_order[:1]
                                         + labels_order[2:], numbers=True)
            self.failUnless(cm._plotted_confusionmatrix[0,0] == cm.matrix[1,1])
            self.failUnless(cm._plotted_confusionmatrix[0,1] == cm.matrix[1,0])
            self.failUnless(cm._plotted_confusionmatrix[1,1] == cm.matrix[0,0])
            self.failUnless(cm._plotted_confusionmatrix[1,0] == cm.matrix[0,1])
            pl.close(fig)
            fig, im, cb = cm.plot(labels=labels_order, numbers=True)
            pl.close(fig)
            # pl.show()

    def test_confusion_plot2(self):

        array = np.array
        uint8 = np.uint8
        sets = [(array([1, 2]), array([1, 1]),
                 array([[ 0.54343765,  0.45656235],
                        [ 0.92395853,  0.07604147]])),
                (array([1, 2]), array([1, 1]),
                 array([[ 0.98030832,  0.01969168],
                        [ 0.78998763,  0.21001237]])),
                (array([1, 2]), array([1, 1]),
                 array([[ 0.86125263,  0.13874737],
                        [ 0.83674113,  0.16325887]])),
                (array([1, 2]), array([1, 1]),
                 array([[ 0.57870383,  0.42129617],
                        [ 0.59702509,  0.40297491]])),
                (array([1, 2]), array([1, 1]),
                 array([[ 0.89530255,  0.10469745],
                        [ 0.69373919,  0.30626081]])),
                (array([1, 2]), array([1, 1]),
                 array([[ 0.75015218,  0.24984782],
                        [ 0.9339767 ,  0.0660233 ]])),
                (array([1, 2]), array([1, 2]),
                 array([[ 0.97826616,  0.02173384],
                        [ 0.38620638,  0.61379362]])),
                (array([2]), array([2]),
                 array([[ 0.46893776,  0.53106224]]))]
        try:
            cm = ConfusionMatrix(sets=sets)
        except:
            self.fail()
        if externals.exists("pylab plottable"):
            import pylab as pl
            #pl.figure()
            #print cm
            fig, im, cb = cm.plot(origin='lower', numbers=True)
            #pl.plot()
            self.failUnless((cm._plotted_confusionmatrix == cm.matrix).all())
            pl.close(fig)
            #fig, im, cb = cm.plot(labels=labels_order, numbers=True)
            #pl.close(fig)
            #pl.show()


def suite():
    return unittest.makeSuite(ErrorsTests)


if __name__ == '__main__':
    import runner

