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

from mvpa2.support.copy import copy

from mvpa2.base.dataset import vstack
from mvpa2.datasets import Dataset
from mvpa2.base import externals, warning
from mvpa2.generators.partition import OddEvenPartitioner, NFoldPartitioner
from mvpa2.generators.base import Repeater
from mvpa2.generators.permutation import AttributePermutator
from mvpa2.generators.splitters import Splitter

from mvpa2.clfs.meta import MulticlassClassifier
from mvpa2.clfs.transerror import ConfusionMatrix, ConfusionBasedError
from mvpa2.measures.base import CrossValidation, TransferMeasure

from mvpa2.clfs.stats import MCNullDist

from mvpa2.misc.exceptions import UnknownStateError
from mvpa2.misc.errorfx import mean_mismatch_error
from mvpa2.mappers.fx import mean_sample, BinaryFxNode

from mvpa2.testing import *
from mvpa2.testing.datasets import datasets
from mvpa2.testing.clfs import *

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
                self.assertTrue((cm.matrix == correct_cm).all())


        # Do a bit more thorough checking
        cm = ConfusionMatrix()
        self.assertRaises(ZeroDivisionError, lambda x:x.percent_correct, cm)
        """No samples -- raise exception"""

        cm.add(reg, regl)

        self.assertEqual(len(cm.sets), 1,
            msg="Should have a single set so far")
        self.assertEqual(cm.matrix.shape, (3,3),
            msg="should be square matrix (len(reglabels) x len(reglabels)")

        self.assertRaises(ValueError, cm.add, reg, np.array([1]))
        """ConfusionMatrix must complaint if number of samples different"""

        # check table content
        self.assertTrue((cm.matrix == correct_cm).all())

        # lets add with new labels (not yet known)
        cm.add(reg, np.array([1,4,1,2,2,2,4,2,1]))

        self.assertEqual(cm.labels, [1,2,3,4],
                             msg="We should have gotten 4th label")

        matrices = cm.matrices          # separate CM per each given set
        self.assertEqual(len(matrices), 2,
                             msg="Have gotten two splits")

        self.assertTrue((matrices[0].matrix + matrices[1].matrix == cm.matrix).all(),
                        msg="Total votes should match the sum across split CMs")

        # check pretty print
        # just a silly test to make sure that printing works
        self.assertTrue(len(cm.as_string(
            header=True, summary=True,
            description=True))>100)
        self.assertTrue(len(str(cm))>100)
        # and that it knows some parameters for printing
        self.assertTrue(len(cm.as_string(summary=True,
                                       header=False))>100)

        # lets check iadd -- just itself to itself
        cm += cm
        self.assertEqual(len(cm.matrices), 4, msg="Must be 4 sets now")

        # lets check add -- just itself to itself
        cm2 = cm + cm
        self.assertEqual(len(cm2.matrices), 8, msg="Must be 8 sets now")
        self.assertEqual(cm2.percent_correct, cm.percent_correct,
                             msg="Percent of corrrect should remain the same ;-)")

        self.assertEqual(cm2.error, 1.0-cm.percent_correct/100.0,
                             msg="Test if we get proper error value")


    def test_confusion_matrix_addition(self):
        """Test confusions addition inconsistent results (GH #51)

        Was fixed by deepcopying instead of copying in __add__
        """
        cm1 = ConfusionMatrix(sets=[[np.array((1,2)), np.array((1,2))]])
        cm2 = ConfusionMatrix(sets=[[np.array((3,2)), np.array((3,2))]])
        assert_array_equal(cm1.stats['P'], [1, 1])
        assert_array_equal(cm2.stats['P'], [1, 1])

        # actual bug scenario -- results would be different
        r1 = (cm1 + cm2).stats['P']
        r2 = (cm1 + cm2).stats['P']
        assert_array_equal(r1, r2)
        assert_array_equal(r1, [1, 2, 1])


    def test_degenerate_confusion(self):
        # We must not just puke -- some testing splits might
        # have just a single target label

        for orig in ([1], [1, 1], [0], [0, 0]):
            cm = ConfusionMatrix(targets=orig, predictions=orig, estimates=orig)

            scm = str(cm)
            self.assertTrue(cm.stats['ACC%'] == 100)


    def test_confusion_matrix_acc(self):
        reg  = [0,0,1,1]
        regl = [1,0,1,0]
        cm = ConfusionMatrix(targets=reg, predictions=regl)
        self.assertTrue('ACC%         50' in str(cm))
        skip_if_no_external('scipy')
        self.assertTrue(cm.stats['CHI^2'] == (0., 1.))


    def test_confusion_matrix_with_mappings(self):
        data = np.array([1,2,1,2,2,2,3,2,1], ndmin=2).T
        reg = [1,1,1,2,2,2,3,3,3]
        regl = [1,2,1,2,2,2,3,2,1]
        correct_cm = [[2,0,1], [1,3,1], [0,0,1]]
        lm = {'apple':1, 'orange':2, 'shitty apple':1, 'candy':3}
        cm = ConfusionMatrix(targets=reg, predictions=regl,
                             labels_map=lm)
        # check table content
        self.assertTrue((cm.matrix == correct_cm).all())
        # assure that all labels are somewhere listed ;-)
        s = str(cm)
        for l in lm.keys():
            self.assertTrue(l in s)

    def test_confusion_call(self):
        # Also tests for the consistency of the labels as
        # either provided or collected by ConfusionMatrix through its lifetime
        self.assertRaises(RuntimeError, ConfusionMatrix(), [1], [1])
        self.assertRaises(ValueError, ConfusionMatrix(labels=[2]), [1], [1])
        # Now lets test proper matrix and either we obtain the same
        t = ['ho', 'ho', 'ho', 'fa', 'fa', 'ho', 'ho']
        p = ['ho','ho', 'ho', 'ho', 'fa', 'fa', 'fa']
        cm1 = ConfusionMatrix(labels=['ho', 'fa'])
        cm2 = ConfusionMatrix(labels=['fa', 'ho'])
        assert_array_equal(cm1(p, t), [[3, 1], [2, 1]])
        assert_array_equal(cm2(p, t), [[1, 2], [1, 3]]) # reverse order of labels

        cm1_ = ConfusionMatrix(labels=['ho', 'fa'], sets=[(t,p)])
        assert_array_equal(cm1(p, t), cm1_.matrix) # both should be identical
        # Lets provoke "mother" CM to get to know more labels which could get ahead
        # of the known ones
        cm1.add(['ho', 'aa'], ['ho', 'aa'])
        # compare and cause recomputation so .__labels get reassigned
        assert_equal(cm1.labels, ['ho', 'fa', 'aa'])
        assert_array_equal(cm1(p, t), [[3, 1, 0], [2, 1, 0], [0, 0, 0]])
        assert_equal(len(cm1.sets), 1)  # just 1 must be known atm from above add
        assert_array_equal(cm1(p, t, store=True), [[3, 1, 0], [2, 1, 0], [0, 0, 0]])
        assert_equal(len(cm1.sets), 2)  # and now 2
        assert_array_equal(cm1(p + ['ho', 'aa'], t + ['ho', 'aa']), cm1.matrix)

    @sweepargs(l_clf=clfswh['linear', 'svm'])
    def test_confusion_based_error(self, l_clf):
        train = datasets['uni2medium']
        train = train[train.sa.train == 1]
        # to check if we fail to classify for 3 labels
        test3 = datasets['uni3medium']
        test3 = test3[test3.sa.train == 1]
        err = ConfusionBasedError(clf=l_clf)
        terr = TransferMeasure(l_clf, Splitter('train', attr_values=[1,1]),
                               postproc=BinaryFxNode(mean_mismatch_error,
                                                     'targets'))

        self.assertRaises(UnknownStateError, err, None)
        """Shouldn't be able to access the state yet"""

        l_clf.train(train)
        e, te = err(None), terr(train)
        te = np.asscalar(te)
        self.assertTrue(abs(e-te) < 1e-10,
            msg="ConfusionBasedError (%.2g) should be equal to TransferError "
                "(%.2g) on traindataset" % (e, te))

        # this will print nasty WARNING but it is ok -- it is just checking code
        # NB warnings are not printed while doing whole testing
        warning("Don't worry about the following warning.")
        if 'multiclass' in l_clf.__tags__:
            self.assertFalse(terr(test3) is None)

        # try copying the beast
        terr_copy = copy(terr)


    @sweepargs(l_clf=clfswh['linear', 'svm'])
    def test_null_dist_prob(self, l_clf):
        train = datasets['uni2medium']

        num_perm = 10
        permutator = AttributePermutator('targets', count=num_perm,
                                         limit='chunks')
        # define class to estimate NULL distribution of errors
        # use left tail of the distribution since we use MeanMatchFx as error
        # function and lower is better
        terr = TransferMeasure(
            l_clf,
            Repeater(count=2),
            postproc=BinaryFxNode(mean_mismatch_error, 'targets'),
            null_dist=MCNullDist(permutator,
                                 tail='left'))

        # check reasonable error range
        err = terr(train)
        self.assertTrue(np.mean(err) < 0.4)

        # Lets do the same for CVTE
        cvte = CrossValidation(l_clf, OddEvenPartitioner(),
            null_dist=MCNullDist(permutator,
                                 tail='left',
                                 enable_ca=['dist_samples']),
            postproc=mean_sample())
        cv_err = cvte(train)

        # check that the result is highly significant since we know that the
        # data has signal
        null_prob = np.asscalar(terr.ca.null_prob)

        if cfg.getboolean('tests', 'labile', default='yes'):
            self.assertTrue(null_prob <= 0.1,
                msg="Failed to check that the result is highly significant "
                    "(got %f) since we know that the data has signal"
                    % null_prob)

            self.assertTrue(np.asscalar(cvte.ca.null_prob) <= 0.1,
                msg="Failed to check that the result is highly significant "
                    "(got p(cvte)=%f) since we know that the data has signal"
                    % np.asscalar(cvte.ca.null_prob))

        # we should be able to access the actual samples of the distribution
        # yoh: why it is 3D really?
        # mih: because these are the distribution samples for the ONE error
        #      collapsed into ONE value across all folds. It will also be
        #      3d if the return value of the measure isn't a scalar and it is
        #      not collapsed across folds. it simply corresponds to the shape
        #      of the output dataset of the respective measure (+1 axis)
        # Some permutations could have been skipped since classifier failed
        # to train due to degenerate situation etc, thus accounting for them
        self.assertEqual(cvte.null_dist.ca.dist_samples.shape[2],
                             num_perm - cvte.null_dist.ca.skipped)



    @sweepargs(clf=clfswh['multiclass'])
    def test_auc(self, clf):
        """Test AUC computation
        """
        if isinstance(clf, MulticlassClassifier):
            raise SkipTest, \
                  "TODO: handle values correctly in MulticlassClassifier"
        clf.ca.change_temporarily(enable_ca = ['estimates'])
        if 'qda' in clf.__tags__:
            # for reliable estimation of covariances, need sufficient
            # sample size
            ds_size = 'large'
        else:
            ds_size = 'small'
        # uni2 dataset with reordered labels
        ds2 = datasets['uni2' + ds_size].copy()
        # revert labels
        ds2.sa['targets'].value = ds2.targets[::-1].copy()
        # same with uni3
        ds3 = datasets['uni3' + ds_size].copy()
        ul = ds3.sa['targets'].unique
        nl = ds3.targets.copy()
        for l in xrange(3):
            nl[ds3.targets == ul[l]] = ul[(l+1)%3]
        ds3.sa.targets = nl
        for ds in [datasets['uni2' + ds_size], ds2,
                   datasets['uni3' + ds_size], ds3]:
            cv = CrossValidation(clf, OddEvenPartitioner(),
                enable_ca=['stats', 'training_stats'])
            cverror = cv(ds)
            stats = cv.ca.stats.stats
            Nlabels = len(ds.uniquetargets)
            # so we at least do slightly above chance
            # But LARS manages to screw up there as well ATM from time to time, so making
            # its testing labile
            if (('lars' in clf.__tags__) and cfg.getboolean('tests', 'labile', default='yes')) \
                or (not 'lars' in clf.__tags__):
                self.assertTrue(stats['ACC'] > 1.2 / Nlabels)
            auc = stats['AUC']
            if (Nlabels == 2) or (Nlabels > 2 and auc[0] is not np.nan):
                mauc = np.min(stats['AUC'])
                if cfg.getboolean('tests', 'labile', default='yes'):
                    self.assertTrue(mauc > 0.55,
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
        #from mvpa2.clfs.transerror import \
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

        cms = str(cm)
        self.assertTrue('3kHz / 38' in cms)
        if externals.exists("scipy"):
            self.assertTrue('ACC(i) = 0.82-0.012*i p=0.12 r=-0.59 r^2=0.35' in cms)

        if externals.exists("pylab plottable"):
            import pylab as pl
            pl.figure()
            labels_order = ("3kHz", "7kHz", "12kHz", "20kHz","30kHz", None,
                            "song1","song2","song3","song4","song5")
            #print cm
            #fig, im, cb = cm.plot(origin='lower', labels=labels_order)
            fig, im, cb = cm.plot(labels=labels_order[1:2] + labels_order[:1]
                                         + labels_order[2:], numbers=True)
            self.assertTrue(cm._plotted_confusionmatrix[0,0] == cm.matrix[1,1])
            self.assertTrue(cm._plotted_confusionmatrix[0,1] == cm.matrix[1,0])
            self.assertTrue(cm._plotted_confusionmatrix[1,1] == cm.matrix[0,0])
            self.assertTrue(cm._plotted_confusionmatrix[1,0] == cm.matrix[0,1])
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
            self.assertTrue((cm._plotted_confusionmatrix == cm.matrix).all())
            pl.close(fig)
            #fig, im, cb = cm.plot(labels=labels_order, numbers=True)
            #pl.close(fig)
            #pl.show()

    @reseed_rng()
    @labile(3, 1)
    def test_confusionmatrix_nulldist(self):
        from mvpa2.clfs.gnb import GNB
        from mvpa2.clfs.transerror import ConfusionMatrixError
        from mvpa2.misc.data_generators import normal_feature_dataset
        for snr in [0., 2.,]:
            ds = normal_feature_dataset(snr=snr, perlabel=42, nchunks=3,
                                        nonbogus_features=[0,1], nfeatures=2)

            clf = GNB()
            num_perm = 50
            permutator = AttributePermutator('targets',
                                             limit='chunks',
                                             count=num_perm)
            cv = CrossValidation(
                clf, NFoldPartitioner(),
                errorfx=ConfusionMatrixError(labels=ds.sa['targets'].unique),
                postproc=mean_sample(),
                null_dist=MCNullDist(permutator,
                                     tail='right', # because we now look at accuracy not error
                                     enable_ca=['dist_samples']),
                enable_ca=['stats'])
            cmatrix = cv(ds)
            #print "Result:\n", cmatrix.samples
            cvnp = cv.ca.null_prob.samples
            #print cvnp
            self.assertTrue(cvnp.shape, (2, 2))
            if cfg.getboolean('tests', 'labile', default='yes'):
                if snr == 0.:
                    # all p should be high since no signal
                    assert_array_less(0.05, cvnp)
                else:
                    # diagonal p is low -- we have signal after all
                    assert_array_less(np.diag(cvnp), 0.05)
                    # off diagonals are high p since for them we would
                    # need to look at the other tail
                    assert_array_less(0.9,
                                      cvnp[(np.array([0,1]), np.array([1,0]))])


def test_confusion_as_node():
    from mvpa2.misc.data_generators import normal_feature_dataset
    from mvpa2.clfs.gnb import GNB
    from mvpa2.clfs.transerror import Confusion
    ds = normal_feature_dataset(snr=2.0, perlabel=42, nchunks=3,
                                nonbogus_features=[0,1], nfeatures=2)
    clf = GNB()
    cv = CrossValidation(
        clf, NFoldPartitioner(),
        errorfx=None,
        postproc=Confusion(labels=ds.UT),
        enable_ca=['stats'])
    res = cv(ds)
    # needs to be identical to CA
    assert_array_equal(res.samples, cv.ca.stats.matrix)
    assert_array_equal(res.sa.predictions, ds.UT)
    assert_array_equal(res.fa.targets, ds.UT)

    skip_if_no_external('scipy')

    from mvpa2.clfs.transerror import BayesConfusionHypothesis
    from mvpa2.base.node import ChainNode
    # same again, but this time with Bayesian hypothesis testing at the end
    cv = CrossValidation(
        clf, NFoldPartitioner(),
        errorfx=None,
        postproc=ChainNode([Confusion(labels=ds.UT),
                            BayesConfusionHypothesis()]))
    res = cv(ds)
    # only two possible hypothesis with two classes
    assert_equals(len(res), 2)
    # the first hypothesis is the can't discriminate anything
    assert_equal(len(res.sa.hypothesis[0]), 1)
    assert_equal(len(res.sa.hypothesis[0][0]), 2)
    # and the hypothesis is actually less likely than the other one
    # (both classes can be distinguished)
    assert(np.e**res.samples[0,0] < np.e**res.samples[1,0])

    # Let's see how well it would work within the searchlight when we also
    # would like to store the hypotheses per each voxel
    # Somewhat an ad-hoc solution for the answer posted on the ML
    #
    # run 1d searchlight of radii 0, for that just provide a .fa with coordinates
    ds.fa['voxel_indices'] = [[0], [1]]
    # and a custom Node which would collect .sa.hypothesis to place together along
    # with the posterior probabilities
    from mvpa2.base.node import Node
    from mvpa2.measures.searchlight import sphere_searchlight

    class KeepBothPosteriorAndHypothesis(Node):
        def _call(self, ds):
            out = np.zeros(1, dtype=object)
            out[0] = (ds.samples, ds.sa.hypothesis)
            return out
    cv.postproc.append(KeepBothPosteriorAndHypothesis())
    sl = sphere_searchlight(cv, radius=0, nproc=1)
    res = sl(ds)

    assert_equal(res.shape, (1, 2))
    assert_equal(len(res.samples[0,0]), 2)
    assert_equal(res.samples[0,0][0].shape, (2, 2))   # posteriors per 1st SL
    assert_equal(len(res.samples[0,0][1]), 2)   # 2 of hypotheses


def test_bayes_confusion_hyp():
    from mvpa2.clfs.transerror import BayesConfusionHypothesis
    conf = np.array([
            [ 10,  0,  5,  5],
            [  0, 10,  5,  5],
            [  5,  5, 10,  0],
            [  5,  5,  0, 10]
            ])
    conf = Dataset(conf, sa={'labels': ['A', 'B', 'C', 'D']})
    bayes = BayesConfusionHypothesis(labels_attr='labels')
    skip_if_no_external('scipy')        # uses factorial from scipy.misc
    hyptest = bayes(conf)
    # by default comes with all hypothesis and posterior probs
    assert_equal(hyptest.shape, (15,2))
    assert_array_equal(hyptest.fa.stat, ['log(p(C|H))', 'log(p(H|C))'])
    # check order of hypothesis (coarse)
    assert_array_equal(hyptest.sa.hypothesis[0], [['A', 'B', 'C', 'D']])
    assert_array_equal(hyptest.sa.hypothesis[-1], [['A'], ['B'], ['C'], ['D']])
    # now with limited hypothesis (given with literal labels), set and in
    # non-log scale
    bayes = BayesConfusionHypothesis(labels_attr='labels', log=False,
                hypotheses=[[['A', 'B', 'C', 'D']],
                            [['A', 'C',], ['B', 'D']],
                            [['A', 'D',], ['B', 'C']],
                            [['A'], ['B'], ['C'], ['D']]])
    hyptest = bayes(conf)
    # also with custom hyp the post-probs must add up to 1
    post_prob = hyptest.samples[:,1]
    assert_almost_equal(np.sum(post_prob), 1)
    # in this particular case ...
    assert(post_prob[3] - np.sum(post_prob[1:3]) < 0.02)


def suite():  # pragma: no cover
    return unittest.makeSuite(ErrorsTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()

