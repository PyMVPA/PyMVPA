# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Collection of classifiers to ease the exploration.
"""

__docformat__ = 'restructuredtext'

from mvpa2.base.types import is_sequence_type

# Define sets of classifiers
from mvpa2.clfs.meta import FeatureSelectionClassifier, SplitClassifier, \
     MulticlassClassifier, RegressionAsClassifier
from mvpa2.clfs.smlr import SMLR
from mvpa2.clfs.knn import kNN
from mvpa2.clfs.gda import LDA, QDA
from mvpa2.clfs.gnb import GNB
from mvpa2.kernels.np import LinearKernel, SquaredExponentialKernel, \
     GeneralizedLinearKernel
from mvpa2.featsel.rfe import SplitRFE


# Helpers
from mvpa2.base import externals, cfg
from mvpa2.measures.anova import OneWayAnova
from mvpa2.mappers.fx import absolute_features, maxofabs_sample
from mvpa2.clfs.smlr import SMLRWeights
from mvpa2.featsel.helpers import FractionTailSelector, \
    FixedNElementTailSelector, RangeElementSelector
from mvpa2.generators.partition import OddEvenPartitioner
from mvpa2.featsel.base import SensitivityBasedFeatureSelection

# Kernels
from mvpa2.kernels.libsvm import LinearLSKernel, RbfLSKernel, \
     PolyLSKernel, SigmoidLSKernel

_KNOWN_INTERNALS = [ 'knn', 'binary', 'svm', 'linear',
        'smlr', 'does_feature_selection', 'has_sensitivity',
        'multiclass', 'non-linear', 'kernel-based', 'lars',
        'regression', 'regression_based', 'random_tie_breaking',
        'non-deterministic', 'needs_population',
        'libsvm', 'sg', 'meta', 'retrainable', 'gpr',
        'notrain2predict', 'ridge', 'blr', 'gnpp', 'enet', 'glmnet',
        'gnb', 'plr', 'rpy2', 'swig', 'skl', 'lda', 'qda',
        'random-forest', 'extra-trees']

class Warehouse(object):
    """Class to keep known instantiated classifiers

    Should provide easy ways to select classifiers of needed kind:
    clfswh['linear', 'svm'] should return all linear SVMs
    clfswh['linear', 'multiclass'] should return all linear classifiers
    capable of doing multiclass classification
    """

    def __init__(self, known_tags=None, matches=None):
        """Initialize warehouse

        Parameters
        ----------
        known_tags : list of str
          List of known tags
        matches : dict
          Optional dictionary of additional matches. E.g. since any
          regression can be used as a binary classifier,
          matches={'binary':['regression']}, would allow to provide
          regressions also if 'binary' was requested
          """
        self._known_tags = set(known_tags)
        self.__items = []
        self.__keys = set()
        if matches is None:
            matches = {}
        self.__matches = matches
        self.__descriptions = {}

    def __getitem__(self, *args):
        if isinstance(args[0], tuple):
            args = args[0]

        # so we explicitely handle [:]
        if args == (slice(None),):
            args = []

        # lets remove optional modifier '!'
        dargs = set([str(x).lstrip('!') for x in args]).difference(
            self._known_tags)

        if len(dargs)>0:
            raise ValueError, "Unknown internals %s requested. Known are %s" % \
                  (list(dargs), list(self._known_tags))

        # dummy implementation for now
        result = []
        # check every known item
        for item in self.__items:
            good = True
            # by default each one counts
            for arg in args:
                # check for rejection first
                if arg.startswith('!'):
                    if (arg[1:] in item.__tags__):
                        good = False
                        break
                    else:
                        continue
                # check for inclusion
                found = False
                for arg in [arg] + self.__matches.get(arg, []):
                    if (arg in item.__tags__):
                        found = True
                        break
                good = found
                if not good:
                    break
            if good:
                result.append(item)
        return result

    def __iadd__(self, item):
        if is_sequence_type(item):
            for item_ in item:
                self.__iadd__(item_)
        else:
            if not hasattr(item, '__tags__'):
                raise ValueError, "Cannot register %s " % item + \
                      "which has no __tags__ defined"
            if item.descr in self.__descriptions:
                raise ValueError("Cannot register %s, " % item + \
                      "an item with descriptions '%s' already exists" \
                      % item.descr)
            if len(item.__tags__) == 0:
                raise ValueError, "Cannot register %s " % item + \
                      "which has empty __tags__"
            clf_internals = set(item.__tags__)
            if clf_internals.issubset(self._known_tags):
                self.__items.append(item)
                self.__keys |= clf_internals
            else:
                raise ValueError, 'Unknown clf internal(s) %s' % \
                      clf_internals.difference(self._known_tags)
            # access by descr
            self.__descriptions[item.descr] = item
        return self

    def get_by_descr(self, descr):
        return self.__descriptions[descr]

    @property
    def internals(self):
        """Known internal tags of the classifiers
        """
        return self.__keys

    def listing(self):
        """Listing (description + internals) of registered items
        """
        return [(x.descr, x.__tags__) for x in self.__items]

    def print_registered(self, *args):
        if not len(args):
            args = (slice(None))
        import numpy as np
        import textwrap
        # sort by description
        for lrn in sorted(self.__getitem__(args), key=lambda x: x.descr.lower()):
            print '%s\n%s' % (
                    lrn.descr,
                    textwrap.fill(', '.join(np.unique(lrn.__tags__)), 70,
                                  initial_indent=' ' * 4,
                                  subsequent_indent=' ' * 4)
                )

    @property
    def items(self):
        """Registered items
        """
        return self.__items

    @property
    def descriptions(self):
        """Descriptions of registered items"""
        return self.__descriptions.keys()


clfswh = Warehouse(known_tags=_KNOWN_INTERNALS) # classifiers
regrswh = Warehouse(known_tags=_KNOWN_INTERNALS) # regressions

# NB:
#  - Nu-classifiers are turned off since for haxby DS default nu
#    is an 'infisible' one
#  - Python's SMLR is turned off for the duration of development
#    since it is slow and results should be the same as of C version
#
clfswh += [ SMLR(lm=0.1, implementation="C", descr="SMLR(lm=0.1)"),
          SMLR(lm=1.0, implementation="C", descr="SMLR(lm=1.0)"),
          #SMLR(lm=10.0, implementation="C", descr="SMLR(lm=10.0)"),
          #SMLR(lm=100.0, implementation="C", descr="SMLR(lm=100.0)"),
          #SMLR(implementation="Python", descr="SMLR(Python)")
          ]

clfswh += \
     [ MulticlassClassifier(SMLR(lm=0.1),
                            descr='Pairs+maxvote multiclass on SMLR(lm=0.1)') ]

if externals.exists('libsvm'):
    from mvpa2.clfs.libsvmc import svm as libsvm
    clfswh._known_tags.update(libsvm.SVM._KNOWN_IMPLEMENTATIONS.keys())
    clfswh += [libsvm.SVM(descr="libsvm.LinSVM(C=def)", probability=1),
             libsvm.SVM(
                 C=-10.0, descr="libsvm.LinSVM(C=10*def)", probability=1),
             libsvm.SVM(
                 C=1.0, descr="libsvm.LinSVM(C=1)", probability=1),
             libsvm.SVM(svm_impl='NU_SVC',
                        descr="libsvm.LinNuSVM(nu=def)", probability=1)
             ]
    clfswh += [libsvm.SVM(kernel=RbfLSKernel(), descr="libsvm.RbfSVM()"),
             libsvm.SVM(kernel=RbfLSKernel(), svm_impl='NU_SVC',
                        descr="libsvm.RbfNuSVM(nu=def)"),
             libsvm.SVM(kernel=PolyLSKernel(),
                        descr='libsvm.PolySVM()', probability=1),
             #libsvm.svm.SVM(kernel=SigmoidLSKernel(),
             #               svm_impl='C_SVC',
             #               descr='libsvm.SigmoidSVM()'),
             ]

    # regressions
    regrswh._known_tags.update(['EPSILON_SVR', 'NU_SVR'])
    regrswh += [libsvm.SVM(svm_impl='EPSILON_SVR', descr='libsvm epsilon-SVR'),
                libsvm.SVM(svm_impl='NU_SVR', descr='libsvm nu-SVR')]

if externals.exists('shogun'):
    from mvpa2.clfs import sg
    
    from mvpa2.kernels.sg import LinearSGKernel, PolySGKernel, RbfSGKernel
    clfswh._known_tags.update(sg.SVM._KNOWN_IMPLEMENTATIONS)

    # TODO: some classifiers are not yet ready to be used out-of-the-box in
    # PyMVPA, thus we don't populate warehouse with their instances
    bad_classifiers = [
        'mpd',  # was segfault, now non-training on testcases, and XOR.
                # and was described as "for educational purposes", thus
                # shouldn't be used for real data ;-)
        # Should be a drop-in replacement for lightsvm
        'gpbt', # fails to train for testAnalyzerWithSplitClassifier
                # also 'retraining' doesn't work -- fails to generalize
        'gmnp', # would fail with 'assertion Cache_Size > 2'
                # if shogun < 0.6.3, also refuses to train
        'svrlight', # fails to 'generalize' as a binary classifier
                    # after 'binning'
        'krr', # fails to generalize
        'svmocas', # fails to generalize
        'libsvr'                        # XXXregr removing regressions as classifiers
        ]
    if not externals.exists('sg_fixedcachesize'):
        # would fail with 'assertion Cache_Size > 2' if shogun < 0.6.3
        bad_classifiers.append('gnpp')

    for impl in sg.SVM._KNOWN_IMPLEMENTATIONS:
        # Uncomment the ones to disable
        if impl in bad_classifiers:
            continue
        clfswh += [
            sg.SVM(
                descr="sg.LinSVM(C=def)/%s" % impl, svm_impl=impl),
            sg.SVM(
                C=-10.0, descr="sg.LinSVM(C=10*def)/%s" % impl, svm_impl=impl),
            sg.SVM(
                C=1.0, descr="sg.LinSVM(C=1)/%s" % impl, svm_impl=impl),
            ]
        if not impl in ['svmocas']:     # inherently linear only
            clfswh += [
                sg.SVM(kernel=RbfSGKernel(),
                       descr="sg.RbfSVM()/%s" % impl, svm_impl=impl),
    #            sg.SVM(kernel=RbfSGKernel(),
    #                   descr="sg.RbfSVM(gamma=0.1)/%s"
    #                    % impl, svm_impl=impl, gamma=0.1),
    #           sg.SVM(descr="sg.SigmoidSVM()/%s"
    #                   % impl, svm_impl=impl, kernel=SigmoidSGKernel(),),
                ]

    _optional_regressions = []
    if externals.exists('shogun.krr') and externals.versions['shogun'] >= '0.9':
        _optional_regressions += ['krr']
    for impl in ['libsvr'] + _optional_regressions:# \
        # XXX svrlight sucks in SG -- dont' have time to figure it out
        #+ ([], ['svrlight'])['svrlight' in sg.SVM._KNOWN_IMPLEMENTATIONS]:
        regrswh._known_tags.update([impl])
        regrswh += [ sg.SVM(svm_impl=impl, descr='sg.LinSVMR()/%s' % impl),
                   #sg.SVM(svm_impl=impl, kernel_type='RBF',
                   #       descr='sg.RBFSVMR()/%s' % impl),
                   ]

if len(clfswh['svm', 'linear']) > 0:
    # if any SVM implementation is known, import default ones
    from mvpa2.clfs.svm import *

# lars from R via RPy
if externals.exists('lars'):
    import mvpa2.clfs.lars as lars
    from mvpa2.clfs.lars import LARS
    for model in lars.known_models:
        # XXX create proper repository of classifiers!
        lars_clf = RegressionAsClassifier(
            LARS(descr="LARS(%s)" % model,
                 model_type=model),
            descr='LARS(model_type=%r) classifier' % model)
        clfswh += lars_clf

        # is a regression, too
        lars_regr = LARS(descr="_LARS(%s)" % model,
                         model_type=model)
        regrswh += lars_regr
        # clfswh += MulticlassClassifier(lars,
        #             descr='Multiclass %s' % lars.descr)

## Still fails unittests battery although overhauled otherwise.
## # enet from R via RPy2
## if externals.exists('elasticnet'):
##     from mvpa2.clfs.enet import ENET
##     clfswh += RegressionAsClassifier(ENET(),
##                                      descr="RegressionAsClassifier(ENET())")
##     regrswh += ENET(descr="ENET()")

# glmnet from R via RPy
if externals.exists('glmnet'):
    from mvpa2.clfs.glmnet import GLMNET_C, GLMNET_R
    clfswh += GLMNET_C(descr="GLMNET_C()")
    regrswh += GLMNET_R(descr="GLMNET_R()")

# LDA/QDA
clfswh += LDA(descr='LDA()')
clfswh += QDA(descr='QDA()')

if externals.exists('skl'):
    _skl_version = externals.versions['skl']
    _skl_api09 = _skl_version >= '0.9'
    def _skl_import(submod, class_):
        if _skl_api09:
            submod_ = __import__('sklearn.%s' % submod, fromlist=[submod])
        else:
            submod_ = __import__('scikits.learn.%s' % submod, fromlist=[submod])
        return getattr(submod_, class_)

    sklLDA = _skl_import('lda', 'LDA')
    from mvpa2.clfs.skl.base import SKLLearnerAdapter
    clfswh += SKLLearnerAdapter(sklLDA(),
                                tags=['lda', 'linear', 'multiclass', 'binary'],
                                descr='skl.LDA()')

    if _skl_version >= '0.10':
        # Out of Bag Estimates
        sklRandomForestClassifier = _skl_import('ensemble', 'RandomForestClassifier')
        clfswh += SKLLearnerAdapter(sklRandomForestClassifier(),
                                     tags=['random-forest', 'linear', 'non-linear',
                                           'binary', 'multiclass',
                                           'non-deterministic', 'needs_population',],
                                     descr='skl.RandomForestClassifier()')

        sklRandomForestRegression = _skl_import('ensemble', 'RandomForestRegressor')
        regrswh += SKLLearnerAdapter(sklRandomForestRegression(),
                                     tags=['random-forest', 'linear', 'non-linear',
                                           'regression',
                                           'non-deterministic', 'needs_population',],
                                     descr='skl.RandomForestRegression()')


        sklExtraTreesClassifier = _skl_import('ensemble', 'ExtraTreesClassifier')
        clfswh += SKLLearnerAdapter(sklExtraTreesClassifier(),
                                     tags=['extra-trees', 'linear', 'non-linear',
                                           'binary', 'multiclass',
                                           'non-deterministic', 'needs_population',],
                                     descr='skl.ExtraTreesClassifier()')

        sklExtraTreesRegression = _skl_import('ensemble', 'ExtraTreesRegressor')
        regrswh += SKLLearnerAdapter(sklExtraTreesRegression(),
                                     tags=['extra-trees', 'linear', 'non-linear',
                                           'regression',
                                           'non-deterministic', 'needs_population',],
                                     descr='skl.ExtraTreesRegression()')


    if _skl_version >= '0.8':
        if _skl_version >= '0.14':
            sklPLSRegression = _skl_import('cross_decomposition', 'PLSRegression')
        else:
            sklPLSRegression = _skl_import('pls', 'PLSRegression')
        # somewhat silly use of PLS, but oh well
        regrswh += SKLLearnerAdapter(sklPLSRegression(n_components=1),
                                     tags=['linear', 'regression'],
                                     enforce_dim=1,
                                     descr='skl.PLSRegression_1d()')

    if externals.versions['skl'] >= '0.6.0':
        sklLars = _skl_import('linear_model',
                              _skl_api09 and 'Lars' or 'LARS')
        sklLassoLars = _skl_import('linear_model',
                                   _skl_api09 and 'LassoLars' or 'LassoLARS')
        sklElasticNet = _skl_import('linear_model', 'ElasticNet')
        _lars_tags = ['lars', 'linear', 'regression', 'does_feature_selection']

        _lars = SKLLearnerAdapter(sklLars(),
                                  tags=_lars_tags,
                                  descr='skl.Lars()')

        _lasso_lars = SKLLearnerAdapter(sklLassoLars(alpha=0.01),
                                        tags=_lars_tags,
                                        descr='skl.LassoLars()')

        _elastic_net = SKLLearnerAdapter(
            sklElasticNet(alpha=.01,
                          **{'l1_ratio' if externals.versions['skl'] >= '0.13'
                                        else 'rho': .3}),
            tags=['enet', 'regression', 'linear', # 'has_sensitivity',
                 'does_feature_selection'],
            descr='skl.ElasticNet()')

        regrswh += [_lars, _lasso_lars, _elastic_net]
        clfswh += [RegressionAsClassifier(_lars, descr="skl.Lars_C()"),
                   RegressionAsClassifier(_lasso_lars, descr="skl.LassoLars_C()"),
                   RegressionAsClassifier(_elastic_net, descr="skl.ElasticNet_C()"),
                   ]

    if _skl_version >= '0.10':
        sklLassoLarsIC = _skl_import('linear_model', 'LassoLarsIC')
        _lasso_lars_ic = SKLLearnerAdapter(sklLassoLarsIC(),
                                           tags=_lars_tags,
                                           descr='skl.LassoLarsIC()')
        regrswh += [_lasso_lars_ic]
        clfswh += [RegressionAsClassifier(_lasso_lars_ic,
                                          descr='skl.LassoLarsIC_C()')]

# kNN
clfswh += kNN(k=5, descr="kNN(k=5)")
clfswh += kNN(k=5, voting='majority', descr="kNN(k=5, voting='majority')")

clfswh += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           SMLRWeights(SMLR(lm=1.0, implementation="C"),
                       postproc=maxofabs_sample()),
           RangeElementSelector(mode='select')),
        descr="kNN on SMLR(lm=1) non-0")

clfswh += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="kNN on 5%(ANOVA)")

clfswh += \
    FeatureSelectionClassifier(
        kNN(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FixedNElementTailSelector(50, mode='select', tail='upper')),
        descr="kNN on 50(ANOVA)")


# GNB
clfswh += GNB(descr="GNB()")
clfswh += GNB(common_variance=True, descr="GNB(common_variance=True)")
clfswh += GNB(prior='uniform', descr="GNB(prior='uniform')")
clfswh += \
    FeatureSelectionClassifier(
        GNB(),
        SensitivityBasedFeatureSelection(
           OneWayAnova(),
           FractionTailSelector(0.05, mode='select', tail='upper')),
        descr="GNB on 5%(ANOVA)")

# GPR
if externals.exists('scipy'):
    from mvpa2.clfs.gpr import GPR

    regrswh += GPR(kernel=LinearKernel(), descr="GPR(kernel='linear')")
    regrswh += GPR(kernel=SquaredExponentialKernel(),
                   descr="GPR(kernel='sqexp')")

    # Add wrapped GPR as a classifier
    gprcb = RegressionAsClassifier(
        GPR(kernel=GeneralizedLinearKernel()), descr="GPRC(kernel='linear')")
    # lets remove multiclass label from it
    gprcb.__tags__.pop(gprcb.__tags__.index('multiclass'))
    clfswh += gprcb

    # and create a proper multiclass one
    clfswh += MulticlassClassifier(
        RegressionAsClassifier(
            GPR(kernel=GeneralizedLinearKernel())),
        descr="GPRCM(kernel='linear')")

# BLR
from mvpa2.clfs.blr import BLR
clfswh += RegressionAsClassifier(BLR(descr="BLR()"),
                                 descr="BLR Classifier")

#PLR
from mvpa2.clfs.plr import PLR
clfswh += PLR(descr="PLR()")
if externals.exists('scipy'):
    clfswh += PLR(reduced=0.05, descr="PLR(reduced=0.01)")

# SVM stuff

if len(clfswh['linear', 'svm']) > 0:

    linearSVMC = clfswh['linear', 'svm',
                             cfg.get('svm', 'backend', default='libsvm').lower()
                             ][0]

    # "Interesting" classifiers
    clfswh += \
         FeatureSelectionClassifier(
             linearSVMC.clone(),
             SensitivityBasedFeatureSelection(
                SMLRWeights(SMLR(lm=0.1, implementation="C"),
                            postproc=maxofabs_sample()),
                RangeElementSelector(mode='select')),
             descr="LinSVM on SMLR(lm=0.1) non-0")

    _rfeclf = linearSVMC.clone()
    clfswh += \
         FeatureSelectionClassifier(
             _rfeclf,
             SplitRFE(
                 _rfeclf,
                 OddEvenPartitioner(),
                 fselector=FractionTailSelector(
                     0.2, mode='discard', tail='lower')),
             descr="LinSVM with nested-CV RFE")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
                SMLRWeights(SMLR(lm=1.0, implementation="C"),
                            postproc=maxofabs_sample()),
                RangeElementSelector(mode='select')),
            descr="LinSVM on SMLR(lm=1) non-0")


    # "Interesting" classifiers
    clfswh += \
        FeatureSelectionClassifier(
            RbfCSVMC(),
            SensitivityBasedFeatureSelection(
               SMLRWeights(SMLR(lm=1.0, implementation="C"),
                           postproc=maxofabs_sample()),
               RangeElementSelector(mode='select')),
            descr="RbfSVM on SMLR(lm=1) non-0")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
               OneWayAnova(),
               FractionTailSelector(0.05, mode='select', tail='upper')),
            descr="LinSVM on 5%(ANOVA)")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
               OneWayAnova(),
               FixedNElementTailSelector(50, mode='select', tail='upper')),
            descr="LinSVM on 50(ANOVA)")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
               linearSVMC.get_sensitivity_analyzer(postproc=maxofabs_sample()),
               FractionTailSelector(0.05, mode='select', tail='upper')),
            descr="LinSVM on 5%(SVM)")

    clfswh += \
        FeatureSelectionClassifier(
            linearSVMC.clone(),
            SensitivityBasedFeatureSelection(
               linearSVMC.get_sensitivity_analyzer(postproc=maxofabs_sample()),
               FixedNElementTailSelector(50, mode='select', tail='upper')),
            descr="LinSVM on 50(SVM)")


    ### Imports which are specific to RFEs
    # from mvpa2.datasets.splitters import OddEvenSplitter
    # from mvpa2.clfs.transerror import TransferError
    # from mvpa2.featsel.rfe import RFE
    # from mvpa2.featsel.helpers import FixedErrorThresholdStopCrit
    # from mvpa2.clfs.transerror import ConfusionBasedError

    # SVM with unbiased RFE -- transfer-error to another splits, or in
    # other terms leave-1-out error on the same dataset
    # Has to be bound outside of the RFE definition since both analyzer and
    # error should use the same instance.
    rfesvm_split = SplitClassifier(linearSVMC)#clfswh['LinearSVMC'][0])

    # "Almost" classical RFE. If this works it would differ only that
    # our transfer_error is based on internal splitting and classifier used
    # within RFE is a split classifier and its sensitivities per split will get
    # averaged
    #

    #clfswh += \
    #  FeatureSelectionClassifier(
    #    clf = LinearCSVMC(), #clfswh['LinearSVMC'][0],         # we train LinearSVM
    #    feature_selection = RFE(             # on features selected via RFE
    #        # based on sensitivity of a clf which does splitting internally
    #        sensitivity_analyzer=rfesvm_split.get_sensitivity_analyzer(),
    #        transfer_error=ConfusionBasedError(
    #           rfesvm_split,
    #           confusion_state="confusion"),
    #           # and whose internal error we use
    #        feature_selector=FractionTailSelector(
    #                           0.2, mode='discard', tail='lower'),
    #                           # remove 20% of features at each step
    #        update_sensitivity=True),
    #        # update sensitivity at each step
    #    descr='LinSVM+RFE(splits_avg)' )
    #
    #clfswh += \
    #  FeatureSelectionClassifier(
    #    clf = LinearCSVMC(),                 # we train LinearSVM
    #    feature_selection = RFE(             # on features selected via RFE
    #        # based on sensitivity of a clf which does splitting internally
    #        sensitivity_analyzer=rfesvm_split.get_sensitivity_analyzer(),
    #        transfer_error=ConfusionBasedError(
    #           rfesvm_split,
    #           confusion_state="confusion"),
    #           # and whose internal error we use
    #        feature_selector=FractionTailSelector(
    #                           0.2, mode='discard', tail='lower'),
    #                           # remove 20% of features at each step
    #        update_sensitivity=False),
    #        # update sensitivity at each step
    #    descr='LinSVM+RFE(splits_avg,static)' )

    rfesvm = LinearCSVMC()

    # This classifier will do RFE while taking transfer error to testing
    # set of that split. Resultant classifier is voted classifier on top
    # of all splits, let see what that would do ;-)
    #clfswh += \
    #  SplitClassifier(                      # which does splitting internally
    #   FeatureSelectionClassifier(
    #    clf = LinearCSVMC(),
    #    feature_selection = RFE(             # on features selected via RFE
    #        sensitivity_analyzer=\
    #            rfesvm.get_sensitivity_analyzer(postproc=absolute_features()),
    #        transfer_error=TransferError(rfesvm),
    #        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
    #        feature_selector=FractionTailSelector(
    #                           0.2, mode='discard', tail='lower'),
    #                           # remove 20% of features at each step
    #        update_sensitivity=True)),
    #        # update sensitivity at each step
    #    descr='LinSVM+RFE(N-Fold)')
    #
    #
    #clfswh += \
    #  SplitClassifier(                      # which does splitting internally
    #   FeatureSelectionClassifier(
    #    clf = LinearCSVMC(),
    #    feature_selection = RFE(             # on features selected via RFE
    #        sensitivity_analyzer=\
    #            rfesvm.get_sensitivity_analyzer(postproc=absolute_features()),
    #        transfer_error=TransferError(rfesvm),
    #        stopping_criterion=FixedErrorThresholdStopCrit(0.05),
    #        feature_selector=FractionTailSelector(
    #                           0.2, mode='discard', tail='lower'),
    #                           # remove 20% of features at each step
    #        update_sensitivity=True)),
    #        # update sensitivity at each step
    #   splitter = OddEvenSplitter(),
    #   descr='LinSVM+RFE(OddEven)')

