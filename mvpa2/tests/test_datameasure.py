# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Unit tests for PyMVPA SplittingSensitivityAnalyzer"""

import numpy as np

from mvpa2.testing import *
from mvpa2.testing.clfs import *
from mvpa2.testing.datasets import *

from mvpa2.base import externals, warning
from mvpa2.base.node import ChainNode, CombinedNode
from mvpa2.datasets.base import Dataset, AttrDataset
from mvpa2.featsel.base import SensitivityBasedFeatureSelection, \
        CombinedFeatureSelection
from mvpa2.featsel.helpers import FixedNElementTailSelector, \
                                 FractionTailSelector, RangeElementSelector

from mvpa2.featsel.rfe import RFE

from mvpa2.clfs.meta import SplitClassifier, MulticlassClassifier, \
     FeatureSelectionClassifier
from mvpa2.clfs.smlr import SMLR, SMLRWeights
from mvpa2.mappers.zscore import zscore
from mvpa2.mappers.fx import sumofabs_sample, absolute_features, FxMapper, \
     maxofabs_sample, BinaryFxNode, \
     mean_sample, mean_feature
from mvpa2.generators.splitters import Splitter
from mvpa2.generators.partition import NFoldPartitioner
from mvpa2.generators.resampling import Balancer

from mvpa2.misc.errorfx import mean_mismatch_error
from mvpa2.misc.transformers import Absolute, \
     DistPValue

from mvpa2.measures.base import Measure, \
        TransferMeasure, RepeatedMeasure, CrossValidation
from mvpa2.measures.anova import OneWayAnova, CompoundOneWayAnova
from mvpa2.measures.irelief import IterativeRelief, IterativeReliefOnline, \
     IterativeRelief_Devel, IterativeReliefOnline_Devel


_MEASURES_2_SWEEP = [ OneWayAnova(),
                      CompoundOneWayAnova(postproc=sumofabs_sample()),
                      IterativeRelief(), IterativeReliefOnline(),
                      IterativeRelief_Devel(), IterativeReliefOnline_Devel()
                      ]
if externals.exists('scipy'):
    from mvpa2.measures.corrcoef import CorrCoef
    _MEASURES_2_SWEEP += [ CorrCoef(),
                           # that one is good when small... handle later
                           #CorrCoef(pvalue=True)
                           ]
    from mvpa2.featsel.base import SplitSamplesProbabilityMapper

class SensitivityAnalysersTests(unittest.TestCase):

    def setUp(self):
        self.dataset = datasets['uni2large']


    @sweepargs(dsm=_MEASURES_2_SWEEP)
    def test_basic(self, dsm):
        data = datasets['dumbinv']
        datass = data.samples.copy()

        # compute scores
        f = dsm(data)
        # check if nothing evil is done to dataset
        self.assertTrue(np.all(data.samples == datass))
        self.assertTrue(f.shape == (1, data.nfeatures))
        self.assertTrue(abs(f.samples[0, 1]) <= 1e-12, # some small value
            msg="Failed test with value %g instead of != 0.0" % f.samples[0, 1])
        self.assertTrue(f.samples[0, 0] > 0.1)     # some reasonably large value

        # we should not have NaNs
        self.assertTrue(not np.any(np.isnan(f)))



    # NOTE: lars with stepwise used to segfault if all ca are enabled
    @sweepargs(clfds=
               [(c, datasets['uni2large'])
                for c in clfswh['has_sensitivity', 'binary']] +
               [(c, datasets['uni4large'])
                for c in clfswh['has_sensitivity', 'multiclass']]
               )
    def test_analyzer_with_split_classifier(self, clfds):
        """Test analyzers in split classifier
        """
        clf, ds = clfds             # unroll the tuple
        # We need to skip some LARSes here
        _sclf = str(clf)
        if 'LARS(' in _sclf and "type='stepwise'" in _sclf:
            # ADD KnownToFail thingie from NiPy
            return

        # To don't waste too much time testing lets limit to 3 splits
        nsplits = 3
        partitioner = NFoldPartitioner(count=nsplits)
        mclf = SplitClassifier(clf=clf,
                               partitioner=partitioner,
                               enable_ca=['training_stats',
                                              'stats'])
        sana = mclf.get_sensitivity_analyzer(# postproc=absolute_features(),
                                           pass_attr=['fa.nonbogus_targets'],
                                           enable_ca=["sensitivities"])

        ulabels = ds.uniquetargets
        nlabels = len(ulabels)
        # Can't rely on splitcfg since count-limit is done in __call__
        assert(nsplits == len(list(partitioner.generate(ds))))
        sens = sana(ds)
        assert('nonbogus_targets' in sens.fa) # were they passsed?
        # TODO: those few do not expose biases
        if not len(set(clf.__tags__).intersection(('lars', 'glmnet', 'gpr'))):
            assert('biases' in sens.sa)
            # print sens.sa.biases
        # It should return either ...
        #  nlabels * nsplits
        req_nsamples = [ nlabels * nsplits ]
        if nlabels == 2:
            # A single sensitivity in case of binary
            req_nsamples += [ nsplits ]
        else:
            # and for pairs in case of multiclass
            req_nsamples += [ (nlabels * (nlabels - 1) / 2) * nsplits ]
            # and for 1-vs-1 embedded within Multiclass operating on
            # pairs (e.g. SMLR)
            req_nsamples += [req_nsamples[-1] * 2]

            # Also for regression_based -- they can do multiclass
            # but only 1 sensitivity is provided
            if 'regression_based' in clf.__tags__:
                req_nsamples += [ nsplits ]

        # # of features should correspond
        self.assertEqual(sens.shape[1], ds.nfeatures)
        # # of samples/sensitivities should also be reasonable
        self.assertTrue(sens.shape[0] in req_nsamples)

        # Check if labels are present
        self.assertTrue('splits' in sens.sa)
        self.assertTrue('targets' in sens.sa)
        # should be 1D -- otherwise dtype object
        self.assertTrue(sens.sa.targets.ndim == 1)

        sens_ulabels = sens.sa['targets'].unique
        # Some labels might be pairs(tuples) so ndarray would be of
        # dtype object and we would need to get them all
        if sens_ulabels.dtype is np.dtype('object'):
            sens_ulabels = np.unique(
                reduce(lambda x, y: x + y, [list(x) for x in sens_ulabels]))

        assert_array_equal(sens_ulabels, ds.sa['targets'].unique)

        errors = [x.percent_correct
                    for x in sana.clf.ca.stats.matrices]

        # lets go through all sensitivities and see if we selected the right
        # features
        #if 'meta' in clf.__tags__ and len(sens.samples[0].nonzero()[0])<2:
        if '5%' in clf.descr \
               or (nlabels > 2 and 'regression_based' in clf.__tags__):
            # Some meta classifiers (5% of ANOVA) are too harsh ;-)
            # if we get less than 2 features with on-zero sensitivities we
            # cannot really test
            # Also -- regression based classifiers performance for multiclass
            # is expected to suck in general
            return

        if cfg.getboolean('tests', 'labile', default='yes'):
            for conf_matrix in [sana.clf.ca.training_stats] \
                              + sana.clf.ca.stats.matrices:
                self.assertTrue(
                    conf_matrix.percent_correct >= 70,
                    msg="We must have trained on each one more or " \
                    "less correctly. Got %f%% correct on %d labels" %
                    (conf_matrix.percent_correct,
                     nlabels))


        # Since  now we have per split and possibly per label -- lets just find
        # mean per each feature per label across splits
        sensm = FxMapper('samples', lambda x: np.sum(x),
                         uattrs=['targets']).forward(sens)
        sensgm = maxofabs_sample().forward(sensm)    # global max of abs of means

        assert_equal(sensgm.shape[0], 1)
        assert_equal(sensgm.shape[1], ds.nfeatures)

        selected = FixedNElementTailSelector(
            len(ds.a.bogus_features))(sensgm.samples[0])

        if cfg.getboolean('tests', 'labile', default='yes'):

            self.assertEqual(
                set(selected), set(ds.a.nonbogus_features),
                msg="At the end we should have selected the right features. "
                "Chose %s whenever nonbogus are %s"
                % (selected, ds.a.nonbogus_features))

            # Now test each one per label
            # TODO: collect all failures and spit them out at once --
            #       that would make it easy to see if the sensitivity
            #       just has incorrect order of labels assigned
            for sens1 in sensm:
                labels1 = sens1.targets  # labels (1) for this sensitivity
                lndim = labels1.ndim
                label = labels1[0]      # current label

                # XXX whole lndim comparison should be gone after
                #     things get fixed and we arrive here with a tuple!
                if lndim == 1: # just a single label
                    self.assertTrue(label in ulabels)

                    ilabel_all = np.where(ds.fa.nonbogus_targets == label)[0]
                    # should have just 1 feature for the label
                    self.assertEqual(len(ilabel_all), 1)
                    ilabel = ilabel_all[0]

                    maxsensi = np.argmax(sens1) # index of max sensitivity
                    self.assertEqual(maxsensi, ilabel,
                        "Maximal sensitivity for %s was found in %i whenever"
                        " original feature was %i for nonbogus features %s"
                        % (labels1, maxsensi, ilabel, ds.a.nonbogus_features))
                elif lndim == 2 and labels1.shape[1] == 2: # pair of labels
                    # we should have highest (in abs) coefficients in
                    # those two labels
                    maxsensi2 = np.argsort(np.abs(sens1))[0][-2:]
                    ilabel2 = [np.where(ds.fa.nonbogus_targets == l)[0][0]
                                    for l in label]
                    self.assertEqual(
                        set(maxsensi2), set(ilabel2),
                        "Maximal sensitivity for %s was found in %s whenever"
                        " original features were %s for nonbogus features %s"
                        % (labels1, maxsensi2, ilabel2, ds.a.nonbogus_features))
                    """
                    # Now test for the sign of each one in pair ;) in
                    # all binary problems L1 (-1) -> L2(+1), then
                    # weights for L2 should be positive.  to test for
                    # L1 -- invert the sign
                    # We already know (if we haven't failed in previous test),
                    # that those 2 were the strongest -- so check only signs
                    """
                    self.assertTrue(
                        sens1.samples[0, ilabel2[0]] < 0,
                        "With %i classes in pair %s got feature %i for %r >= 0"
                        % (nlabels, label, ilabel2[0], label[0]))
                    self.assertTrue(sens1.samples[0, ilabel2[1]] > 0,
                        "With %i classes in pair %s got feature %i for %r <= 0"
                        % (nlabels, label, ilabel2[1], label[1]))
                else:
                    # yoh could be wrong at this assumption... time will show
                    self.fail("Got unknown number labels per sensitivity: %s."
                              " Should be either a single label or a pair"
                              % labels1)


    @sweepargs(clf=clfswh['has_sensitivity'])
    def test_mapped_classifier_sensitivity_analyzer(self, clf):
        """Test sensitivity of the mapped classifier
        """
        # Assuming many defaults it is as simple as
        mclf = FeatureSelectionClassifier(
            clf,
            SensitivityBasedFeatureSelection(
                OneWayAnova(),
                FractionTailSelector(0.5, mode='select', tail='upper')),
            enable_ca=['training_stats'])

        sana = mclf.get_sensitivity_analyzer(postproc=sumofabs_sample(),
                                             enable_ca=["sensitivities"])
        # and lets look at all sensitivities
        dataset = datasets['uni2small']
        # and we get sensitivity analyzer which works on splits
        sens = sana(dataset)
        self.assertEqual(sens.shape, (1, dataset.nfeatures))



    @sweepargs(svm=clfswh['linear', 'svm'])
    def test_linear_svm_weights(self, svm):
        # assumming many defaults it is as simple as
        sana = svm.get_sensitivity_analyzer(enable_ca=["sensitivities"])
        # and lets look at all sensitivities
        sens = sana(self.dataset)
        # for now we can do only linear SVM, so lets check if we raise
        # a concern
        svmnl = clfswh['non-linear', 'svm'][0]
        self.assertRaises(NotImplementedError,
                              svmnl.get_sensitivity_analyzer)


    # XXX doesn't work easily with meta since it would need
    #     to be explicitely passed to the slave classifier's
    #     getSengetSensitivityAnalyzer
    # Note: only libsvm interface supports split_weights
    @sweepargs(svm=clfswh['linear', 'svm', 'libsvm', '!sg', '!meta'])
    def test_linear_svm_weights_per_class(self, svm):
        # assumming many defaults it is as simple as
        kwargs = dict(enable_ca=["sensitivities"])
        sana_split = svm.get_sensitivity_analyzer(
            split_weights=True, **kwargs)
        sana_full = svm.get_sensitivity_analyzer(
            force_train=False, **kwargs)

        # and lets look at all sensitivities
        ds2 = datasets['uni4large'].copy()
        zscore(ds2, param_est=('targets', ['L2', 'L3']))
        ds2 = ds2[np.logical_or(ds2.sa.targets == 'L0', ds2.sa.targets == 'L1')]

        senssplit = sana_split(ds2)
        sensfull = sana_full(ds2)

        self.assertEqual(senssplit.shape, (2, ds2.nfeatures))
        self.assertEqual(sensfull.shape, (1, ds2.nfeatures))

        # just to verify that we split properly and if we reconstruct
        # manually we obtain the same
        dmap = (-1 * senssplit.samples[1] + senssplit.samples[0]) \
               - sensfull.samples
        self.assertTrue((np.abs(dmap) <= 1e-10).all())
        #print "____"
        #print senssplit
        #print SMLR().get_sensitivity_analyzer(combiner=None)(ds2)

        # for now we can do split weights for binary tasks only, so
        # lets check if we raise a concern
        # we temporarily shutdown warning, since it is going to complain
        # otherwise, but we do it on purpose here
        handlers = warning.handlers
        warning.handlers = []
        self.assertRaises(NotImplementedError,
                              sana_split, datasets['uni3medium'])
        # reenable the warnings
        warning.handlers = handlers


    def test_split_featurewise_dataset_measure(self):
        ds = datasets['uni3small']
        sana = RepeatedMeasure(
            SMLR(fit_all_weights=True).get_sensitivity_analyzer(),
            ChainNode([NFoldPartitioner(),
                       Splitter('partitions', attr_values=[1])]))

        sens = sana(ds)
        # a sensitivity for each chunk and each label combination
        assert_equal(sens.shape,
                     (len(ds.sa['chunks'].unique) * len(ds.sa['targets'].unique),
                      ds.nfeatures))

        # Lets try more complex example with 'boosting'
        ds = datasets['uni3medium']
        ds.init_origids('samples')
        sana = RepeatedMeasure(
            SMLR(fit_all_weights=True).get_sensitivity_analyzer(),
            Balancer(amount=0.25, count=2, apply_selection=True),
            enable_ca=['datasets', 'repetition_results'])
        sens = sana(ds)

        assert_equal(sens.shape, (2 * len(ds.sa['targets'].unique),
                                  ds.nfeatures))
        splits = sana.ca.datasets
        self.assertEqual(len(splits), 2)
        self.assertTrue(np.all([s.nsamples == ds.nsamples // 4 for s in splits]))
        # should have used different samples
        self.assertTrue(np.any([splits[0].sa.origids != splits[1].sa.origids]))
        # and should have got different sensitivities
        self.assertTrue(np.any(sens[0] != sens[3]))


        #skip_if_no_external('scipy')
        # Let's disable this one for now until we are sure about the destiny of
        # DistPValue -- read the docstring of it!
        # Most evil example
        #ds = datasets['uni2medium']
        #plain_sana = SVM().get_sensitivity_analyzer(
        #       transformer=DistPValue())
        #boosted_sana = SplitFeaturewiseMeasure(
        #    analyzer=SVM().get_sensitivity_analyzer(
        #       transformer=DistPValue(fpp=0.05)),
        #    splitter=NoneSplitter(npertarget=0.8, mode='first', nrunspersplit=2),
        #    enable_ca=['splits', 'sensitivities'])
        ## lets create feature selector
        #fsel = RangeElementSelector(upper=0.1, lower=0.9, inclusive=True)

        #sanas = dict(plain=plain_sana, boosted=boosted_sana)
        #for k,sana in sanas.iteritems():
        #    clf = FeatureSelectionClassifier(SVM(),
        #                SensitivityBasedFeatureSelection(sana, fsel),
        #                descr='SVM on p=0.2(both tails) using %s' % k)
        #    ce = CrossValidatedTransferError(TransferError(clf),
        #                                     NFoldSplitter())
        #    error = ce(ds)

        #sens = boosted_sana(ds)
        #sens_plain = plain_sana(ds)

        ## TODO: make a really unittest out of it -- not just runtime
        ##       bugs catcher

    # TODO -- unittests for sensitivity analyzers which use combiners
    # (linsvmweights for multi-class SVMs and smlrweights for SMLR)


    @sweepargs(basic_clf=clfswh['has_sensitivity'])
    ##REF: Name was automagically refactored
    def __test_fspipeline_with_split_classifier(self, basic_clf):
        #basic_clf = LinearNuSVMC()
        multi_clf = MulticlassClassifier(clf=basic_clf)
        #svm_weigths = LinearSVMWeights(svm)

        # Proper RFE: aggregate sensitivities across multiple splits,
        # but also due to multi class those need to be aggregated
        # somehow. Transfer error here should be 'leave-1-out' error
        # of split classifier itself
        sclf = SplitClassifier(clf=basic_clf)
        rfe = RFE(sensitivity_analyzer=
                    sclf.get_sensitivity_analyzer(
                        enable_ca=["sensitivities"]),
                  transfer_error=trans_error,
                  feature_selector=FeatureSelectionPipeline(
                      [FractionTailSelector(0.5),
                       FixedNElementTailSelector(1)]),
                  train_pmeasure=True)

        # and we get sensitivity analyzer which works on splits and uses
        # sensitivity
        selected_features = rfe(self.dataset)

    def test_union_feature_selection(self):
        # two methods: 5% highes F-scores, non-zero SMLR weights
        fss = [SensitivityBasedFeatureSelection(
                    OneWayAnova(),
                    FractionTailSelector(0.05, mode='select', tail='upper')),
               SensitivityBasedFeatureSelection(
                    SMLRWeights(SMLR(lm=1, implementation="C"),
                                postproc=sumofabs_sample()),
                    RangeElementSelector(mode='select'))]

        fs = CombinedFeatureSelection(fss, method='union')

        od_union = fs(self.dataset)

        self.assertTrue(fs.method == 'union')
        # check output dataset
        self.assertTrue(od_union.nfeatures <= self.dataset.nfeatures)
        # again for intersection
        fs = CombinedFeatureSelection(fss, method='intersection')
        od_intersect = fs(self.dataset)
        assert_true(od_intersect.nfeatures < od_union.nfeatures)

    @sweepargs(do_int=(False, True))
    def test_anova(self, do_int):
        """Additional aspects of OnewayAnova
        """
        oa = OneWayAnova()
        oa_custom = OneWayAnova(space='custom')

        ds = datasets['uni4large'].copy()
        if do_int:
            ds.samples = (ds.samples * 1000).astype(np.int)
        ds_samples_orig = ds.samples.copy()  # to verify that nothing was modified
        ds_custom = Dataset(ds.samples, sa={'custom': ds.targets})

        r = oa(ds)
        assert_array_equal(ds.samples, ds_samples_orig)  # no inplace changes!
        self.assertRaises(KeyError, oa_custom, ds)
        r_custom = oa_custom(ds_custom)

        self.assertTrue(np.allclose(r.samples, r_custom.samples))

        # we should get the same results on subsequent runs
        r2 = oa(ds)
        r_custom2 = oa_custom(ds_custom)
        self.assertTrue(np.allclose(r.samples, r2.samples))
        self.assertTrue(np.allclose(r_custom.samples, r_custom2.samples))

        skip_if_no_external('scipy')
        from scipy.stats.stats import f_oneway
        # compare against scipy implementation
        # we need to create groups of those target samples
        groups = [
            ds[ds.targets == ut]
            for ut in ds.sa['targets'].unique
        ]
        spf, spp = f_oneway(*groups)
        assert_array_almost_equal(r.samples[0], spf)


    def test_transfer_measure(self):
        # come up with my own measure that only checks if training data
        # and test data are the same
        class MyMeasure(Measure):
            def _train(self, ds):
                self._tds = ds
            def _call(self, ds):
                return Dataset(ds.samples == self._tds.samples)

        tm = TransferMeasure(MyMeasure(), Splitter('chunks', count=2))
        # result should not be all True (== identical)
        assert_true((tm(self.dataset).samples == False).any())


    def test_clf_transfer_measure(self):
        # and now on a classifier
        clf = SMLR()
        enode = BinaryFxNode(mean_mismatch_error, 'targets')
        tm = TransferMeasure(clf, Splitter('chunks', count=2),
                             enable_ca=['stats'])
        res = tm(self.dataset)
        manual_error = np.mean(res.samples.squeeze() != res.sa.targets)
        postproc_error = enode(res)
        tm_err = TransferMeasure(clf, Splitter('chunks', count=2),
                                 postproc=enode)
        auto_error = tm_err(self.dataset)
        ok_(manual_error == postproc_error.samples[0, 0])


    def test_pseudo_cv_measure(self):
        clf = SMLR()
        enode = BinaryFxNode(mean_mismatch_error, 'targets')
        tm = TransferMeasure(clf, Splitter('partitions'), postproc=enode)
        cvgen = NFoldPartitioner()
        rm = RepeatedMeasure(tm, cvgen)
        res = rm(self.dataset)
        # one error per fold
        assert_equal(res.shape, (len(self.dataset.sa['chunks'].unique), 1))

        # we can do the same with Crossvalidation
        cv = CrossValidation(clf, cvgen, enable_ca=['stats', 'training_stats',
                                                    'datasets'])
        res = cv(self.dataset)
        assert_equal(res.shape, (len(self.dataset.sa['chunks'].unique), 1))


    def test_repeated_features(self):
        class CountFeatures(Measure):
            is_trained = True
            def _call(self, ds):
                return Dataset([ds.nfeatures],
                                fa={'nonbogus_targets': list(ds.fa['nonbogus_targets'].unique)})

        cf = CountFeatures()
        spl = Splitter('fa.nonbogus_targets')
        nsplits = len(list(spl.generate(self.dataset)))
        assert_equal(nsplits, 3)
        rm = RepeatedMeasure(cf, spl, concat_as='features')
        res = rm(self.dataset)
        assert_equal(res.shape, (1, nsplits))
        # due to https://github.com/numpy/numpy/issues/641 we are
        # using list(set(...)) construct and there order of
        # nonbogus_targets.unique can vary from run to run, thus there
        # is no guarantee that we would get 18 first, which is a
        # questionable assumption anyways, thus performing checks
        # which do not require any specific order.
        # And yet due to another issue
        # https://github.com/numpy/numpy/issues/3759
        # we can't just == None for the bool mask
        None_fa = np.array([x == None for x in  res.fa.nonbogus_targets])
        assert_array_equal(res.samples[0, None_fa], [18])
        assert_array_equal(res.samples[0, ~None_fa], [1, 1])

        if sys.version_info[0] < 3:
            # with python2 order seems to be consistent
            assert_array_equal(res.samples[0], [18, 1, 1])

    def test_custom_combined_selectors(self):
        """Test combination of the selectors in a single function
        """

        def custom_tail_selector(seq):
            seq1 = FractionTailSelector(0.01, mode='discard', tail='upper')(seq)
            seq2 = FractionTailSelector(0.05, mode='select', tail='upper')(seq)
            return list(set(seq1).intersection(seq2))

        seq = np.arange(100)
        seq_ = custom_tail_selector(seq)

        assert_array_equal(sorted(seq_), [95, 96, 97, 98])
        # verify that this function could be used in place of the selector
        fs = SensitivityBasedFeatureSelection(
                    OneWayAnova(),
                    custom_tail_selector)
        ds = datasets['3dsmall']
        fs.train(ds)          # XXX: why needs to be trained here explicitly?
        ds_ = fs(ds)
        assert_equal(ds_.nfeatures, int(ds.nfeatures * 0.04))

    def test_combined_node(self):
        ds = datasets['3dsmall']
        axis2nodes = dict(h=(mean_feature, mean_feature),
                          v=(mean_sample, mean_sample))

        for i, axis in enumerate('vh'):
            nodes = axis2nodes[axis]
            combined = CombinedNode([n() for n in nodes], axis, False)
            assert_true(combined(ds).shape[i] == 2)
            assert_true(combined(ds).shape[1 - i] == ds.shape[1 - i])

    def test_split_samples_probability_mapper(self):
        skip_if_no_external('scipy')
        nf = 10
        ns = 100
        nsubj = 5
        nchunks = 5
        data = np.random.normal(size=(ns, nf))
        ds = AttrDataset(data, sa=dict(sidx=np.arange(ns),
                                    targets=np.arange(ns) % nchunks,
                                    chunks=np.floor(np.arange(ns) * nchunks / ns),
                                    subjects=np.arange(ns) / (ns / nsubj / nchunks) % nsubj),
                            fa=dict(fidx=np.arange(nf)))
        analyzer = OneWayAnova()
        element_selector = FractionTailSelector(.4, mode='select', tail='upper')
        common = True
        m = SplitSamplesProbabilityMapper(analyzer, 'subjects', probability_label='fprob',
                            select_common_features=common,
                            selector=element_selector)

        m.train(ds)
        y = m(ds)
        z = m(ds.samples)

        assert_array_equal(z, y.samples)
        assert_equal(y.shape, (100, 4))


    def test_pass_attr(self):
        from mvpa2.base.node import Node
        from mvpa2.base.state import ConditionalAttribute

        ds = datasets['dumbinv']

        class MyNode(Node):
            some_sa = ConditionalAttribute(enabled=True)
            some_fa = ConditionalAttribute(enabled=True)
            some_complex = ConditionalAttribute(enabled=True)
            def _call(self, ds):
                return Dataset(np.zeros(ds.shape))
        node = MyNode(pass_attr=['ca.some_sa',
                                 ('ca.some_fa', 'fa'),
                                 ('ca.some_complex', 'fa', 1, 'transposed'),
                                 'sa.targets'])
        node.ca.some_sa = np.arange(len(ds))
        node.ca.some_fa = np.arange(ds.nfeatures)
        node.ca.some_complex = ds.samples
        res = node(ds)
        assert_true('some_sa' in res.sa)
        assert_true('some_fa' in res.fa)
        assert_true('transposed' in res.fa)
        assert_true('targets' in res.sa)
        # view on original array
        assert_true(res.fa.transposed.base is ds.samples)
        assert_array_equal(res.fa.transposed.T, ds.samples)


def suite():  # pragma: no cover
    return unittest.makeSuite(SensitivityAnalysersTests)


if __name__ == '__main__':  # pragma: no cover
    import runner
    runner.run()
