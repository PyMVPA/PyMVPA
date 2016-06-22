# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Recursive feature elimination."""

__docformat__ = 'restructuredtext'

from mvpa2.base import externals
from mvpa2.base.dochelpers import _repr_attrs
from mvpa2.support.copy import copy
from mvpa2.clfs.transerror import ClassifierError
from mvpa2.measures.base import Sensitivity
from mvpa2.featsel.base import IterativeFeatureSelection
from mvpa2.featsel.helpers import BestDetector, \
                                 NBackHistoryStopCrit, \
                                 FractionTailSelector

# For RFELearner
from mvpa2.clfs.meta import ProxyClassifier, FeatureSelectionClassifier
from mvpa2.misc.errorfx import mean_mismatch_error
from mvpa2.measures.base import ProxyMeasure
from mvpa2.generators.splitters import Splitter
from mvpa2.mappers.fx import maxofabs_sample, BinaryFxNode
from mvpa2.base.dochelpers import _str
from mvpa2.generators.base import Repeater

from mvpa2.support.due import due, Doi, BibTeX

import numpy as np
from mvpa2.base.state import ConditionalAttribute

if externals.exists('joblib'):
    import joblib as jl

if __debug__:
    from mvpa2.base import debug

# TODO: Abs value of sensitivity should be able to rule RFE
# Often it is what abs value of the sensitivity is what matters.
# So we should either provide a simple decorator around arbitrary
# FeatureSelector to convert sensitivities to abs values before calling
# actual selector, or a decorator around SensitivityEstimators


class RFE(IterativeFeatureSelection):
    """Recursive feature elimination.

    A `FeaturewiseMeasure` is used to compute sensitivity maps given a
    certain dataset. These sensitivity maps are in turn used to discard
    unimportant features. For each feature selection the transfer error on some
    testdatset is computed. This procedure is repeated until a given
    `StoppingCriterion` is reached.

    References
    ----------
    Such strategy after
      Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene
      selection for cancer classification using support vector
      machines. Mach. Learn., 46(1-3), 389--422.
    was applied to SVM-based analysis of fMRI data in
      Hanson, S. J. & Halchenko, Y. O. (2008). Brain reading using
      full brain support vector machines for object recognition:
      there is no "face identification area". Neural Computation, 20,
      486--503.

    Examples
    --------

    There are multiple possible ways to design an RFE.  Here is one
    example which would rely on a SplitClassifier to extract
    sensitivities and provide estimate of performance (error)

    >>> # Lazy import
    >>> from mvpa2.suite import *
    >>> rfesvm_split = SplitClassifier(LinearCSVMC(), OddEvenPartitioner())
    >>> # design an RFE feature selection to be used with a classifier
    >>> rfe = RFE(rfesvm_split.get_sensitivity_analyzer(
    ...              # take sensitivities per each split, L2 norm, mean, abs them
    ...              postproc=ChainMapper([ FxMapper('features', l2_normed),
    ...                                     FxMapper('samples', np.mean),
    ...                                     FxMapper('samples', np.abs)])),
    ...           # use the error stored in the confusion matrix of split classifier
    ...           ConfusionBasedError(rfesvm_split, confusion_state='stats'),
    ...           # we just extract error from confusion, so no need to split dataset
    ...           Repeater(2),
    ...           # select 50% of the best on each step
    ...           fselector=FractionTailSelector(
    ...               0.50,
    ...               mode='select', tail='upper'),
    ...           # and stop whenever error didn't improve for up to 10 steps
    ...           stopping_criterion=NBackHistoryStopCrit(BestDetector(), 10),
    ...           # we just extract it from existing confusion
    ...           train_pmeasure=False,
    ...           # but we do want to update sensitivities on each step
    ...           update_sensitivity=True)
    >>> clf = FeatureSelectionClassifier(
    ...           LinearCSVMC(),
    ...           # on features selected via RFE
    ...           rfe,
    ...           # custom description
    ...           descr='LinSVM+RFE(splits_avg)' )
    
    Note: If you rely on cross-validation for the StoppingCriterion, make sure
    that you have at least 3 chunks so that SplitClassifier could have at least
    2 chunks to split. Otherwise it can not split more (one chunk could not be
    splitted).

    """

    history = ConditionalAttribute(
        doc="Last step # when each feature was still present")
    sensitivities = ConditionalAttribute(enabled=False,
        doc="History of sensitivities (might consume too much memory")

    def __init__(self,
                 fmeasure,
                 pmeasure,
                 splitter,
                 fselector=FractionTailSelector(0.05),
                 update_sensitivity=True,
                 nfeatures_min=0,
                 **kwargs):
        # XXX Allow for multiple stopping criterions, e.g. error not decreasing
        # anymore OR number of features less than threshold
        """Initialize recursive feature elimination

        Parameters
        ----------
        fmeasure : FeaturewiseMeasure
        pmeasure : Measure
          used to compute the transfer error of a classifier based on a
          certain feature set on the test dataset.
          NOTE: If sensitivity analyzer is based on the same
          classifier as transfer_error is using, make sure you
          initialize transfer_error with train=False, otherwise
          it would train classifier twice without any necessity.
        splitter: Splitter
          This splitter instance has to generate at least two dataset splits
          when called with the input dataset. The first split serves as the
          training dataset and the second as the evaluation dataset.
        fselector : Functor
          Given a sensitivity map it has to return the ids of those
          features that should be kept.
        update_sensitivity : bool
          If False the sensitivity map is only computed once and reused
          for each iteration. Otherwise the sensitivities are
          recomputed at each selection step.
        nfeatures_min : int
          Number of features for RFE to stop if reached.
        """
        # bases init first
        IterativeFeatureSelection.__init__(self, fmeasure, pmeasure, splitter,
                                           fselector, **kwargs)

        self.__update_sensitivity = update_sensitivity
        """Flag whether sensitivity map is recomputed for each step."""

        self._nfeatures_min = nfeatures_min


    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(RFE, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['update_sensitivity'], default=True))

    @due.dcite(
        BibTeX("""
@Article{ GWB+02,
    author = "I. Guyon and J. Weston and S. Barnhill and V. Vapnik",
    title = "Gene Selection for Cancer Classification using Support Vector Machines",
    volume = "46",
    year = "2002",
    pages = "389--422",
    publisher = "Kluwer",
    address = "Hingham, MA, USA",
    journal = "Machine Learning"
}"""),
        description="Recursive feature elimination procedure",
        tags=["implementation"])
    @due.dcite(
        Doi("10.1162/neco.2007.09-06-340"),
        description="Full-brain fMRI decoding using SVM RFE",
        tags=["use"])
    def _train(self, ds):
        """Proceed and select the features recursively eliminating less
        important ones.

        Parameters
        ----------
        ds : Dataset
          used to compute sensitivity maps and train a classifier
          to determine the transfer error
        """
        # get the initial split into train and test
        dataset, testdataset = self._get_traintest_ds(ds)

        if __debug__:
            debug('RFEC',
                  "Initiating RFE with training on %s and testing using %s",
                  (dataset, testdataset))
        errors = []
        """Computed error for each tested features set."""

        ca = self.ca
        ca.nfeatures = []
        """Number of features at each step. Since it is not used by the
        algorithm it is stored directly in the conditional attribute"""

        ca.history = np.arange(dataset.nfeatures)
        """Store the last step # when the feature was still present
        """

        ca.sensitivities = []

        stop = False
        """Flag when RFE should be stopped."""

        results = None
        """Will hold the best feature set ever."""

        wdataset = dataset
        """Operate on working dataset initially identical."""

        wtestdataset = testdataset
        """Same feature selection has to be performs on test dataset as well.
        This will hold the current testdataset."""

        step = 0
        """Counter how many selection step where done."""

        orig_feature_ids = np.arange(dataset.nfeatures)
        """List of feature Ids as per original dataset remaining at any given
        step"""

        sensitivity = None
        """Contains the latest sensitivity map."""

        result_selected_ids = orig_feature_ids
        """Resultant ids of selected features. Since the best is not
        necessarily is the last - we better keep this one around. By
        default -- all features are there"""
        selected_ids = result_selected_ids

        isthebest = True
        """By default (e.g. no errors even estimated) every step is the best one
        """

        while wdataset.nfeatures > 0:

            if __debug__:
                debug('RFEC',
                      "Step %d: nfeatures=%d" % (step, wdataset.nfeatures))

            # mark the features which are present at this step
            # if it brings anyb mentionable computational burden in the future,
            # only mark on removed features at each step
            ca.history[orig_feature_ids] = step

            # Compute sensitivity map
            if self.__update_sensitivity or sensitivity == None:
                sensitivity = self._fmeasure(wdataset)
                if len(sensitivity) > 1:
                    raise ValueError(
                            "RFE cannot handle multiple sensitivities at once. "
                            "'%s' returned %i sensitivities."
                            % (self._fmeasure.__class__.__name__,
                               len(sensitivity)))

            if ca.is_enabled("sensitivities"):
                ca.sensitivities.append(sensitivity)

            if self._pmeasure:
                # get error for current feature set (handles optional retraining)
                error = np.asscalar(self._evaluate_pmeasure(wdataset, wtestdataset))
                # Record the error
                errors.append(error)

                # Check if it is time to stop and if we got
                # the best result
                if self._stopping_criterion is not None:
                    stop = self._stopping_criterion(errors)
                if self._bestdetector is not None:
                    isthebest = self._bestdetector(errors)
            else:
                error = None

            nfeatures = wdataset.nfeatures

            if ca.is_enabled("nfeatures"):
                ca.nfeatures.append(wdataset.nfeatures)

            # store result
            if isthebest:
                result_selected_ids = orig_feature_ids

            if __debug__:
                debug('RFEC',
                      "Step %d: nfeatures=%d error=%s best/stop=%d/%d " %
                      (step, nfeatures, error, isthebest, stop))

            # stop if it is time to finish
            if nfeatures == 1 or nfeatures <= self.nfeatures_min or stop:
                break

            # Select features to preserve
            selected_ids = self._fselector(sensitivity)

            if __debug__:
                debug('RFEC_',
                      "Sensitivity: %s, nfeatures_selected=%d, selected_ids: %s" %
                      (sensitivity, len(selected_ids), selected_ids))


            # Create a dataset only with selected features
            wdataset = wdataset[:, selected_ids]

            # select corresponding sensitivity values if they are not
            # recomputed
            if not self.__update_sensitivity:
                if len(sensitivity.shape) >= 2:
                    assert(sensitivity.shape[0] == 1) # there must be only 1 sample
                    sensitivity = sensitivity[:, selected_ids]
                else:
                    sensitivity = sensitivity[selected_ids]

            # need to update the test dataset as well
            # XXX why should it ever become None?
            # yoh: because we can have __transfer_error computed
            #      using wdataset. See xia-generalization estimate
            #      in lightsvm. Or for god's sake leave-one-out
            #      on a wdataset
            # TODO: document these cases in this class
            if testdataset is not None:
                wtestdataset = wtestdataset[:, selected_ids]

            step += 1

            # WARNING: THIS MUST BE THE LAST THING TO DO ON selected_ids
            selected_ids.sort()
            if self.ca.is_enabled("history") \
                   or self.ca.is_enabled('selected_ids'):
                orig_feature_ids = orig_feature_ids[selected_ids]

            # we already have the initial sensitivities, so even for a shared
            # classifier we can cleanup here
            if self._pmeasure:
                self._pmeasure.untrain()

        # charge conditional attributes
        self.ca.errors = errors
        self.ca.selected_ids = result_selected_ids
        if __debug__:
            debug('RFEC',
                  "Selected %d features: %s",
                  (len(result_selected_ids), result_selected_ids))

        # announce desired features to the underlying slice mapper
        # do copy to survive later selections
        self._safe_assign_slicearg(copy(result_selected_ids))
        # call super to set _Xshape etc
        super(RFE, self)._train(dataset)

    def _untrain(self):
        super(RFE, self)._untrain()
        if self._pmeasure:
            self._pmeasure.untrain()
        if self._fmeasure:
            self._fmeasure.untrain()

    def _get_nfeatures_min(self):
        return self._nfeatures_min

    def _set_nfeatures_min(self, v):
        if self.is_trained:
            self.untrain()
        if v < 0:
            raise ValueError("nfeatures_min must not be negative. Got %s" % v)
        self._nfeatures_min = v

    nfeatures_min = property(fget=_get_nfeatures_min, fset=_set_nfeatures_min)
    update_sensitivity = property(fget=lambda self: self.__update_sensitivity)

def _process_partition(rfe, partition):
    """Helper function to be used to parallelize SplitRFE
    """
    rfe.train(partition)
    return rfe.ca.errors, rfe.ca.nfeatures

class SplitRFE(RFE):
    """RFE with the nested cross-validation to estimate optimal number of features.

    Given a learner (classifier) with a sensitivity analyzer and a
    partitioner, during training SplitRFE first performs a
    cross-validation with RFE to later estimate optimal number of
    features which should survive in RFE.  Optimal number is chosen as
    the mid-point among all minimums of the average errors across
    splits.  After deducing optimal number of features, SplitRFE
    applies regular RFE again on the full training dataset stopping at
    the estimated optimal number of features.

    Examples
    --------

    Resting on an example giving for the :class:`~mvpa2.featself.rfe.RFE` here
    is an implementation using SplitRFE helper:

    >>> # Lazy import
    >>> from mvpa2.suite import *
    >>> # design an RFE feature selection to be used with a classifier
    >>> rfe = SplitRFE(
    ...           LinearCSVMC(),
    ...           OddEvenPartitioner(),
    ...           # take sensitivities per each split, L2 norm, abs, mean them
    ...           fmeasure_postproc=ChainMapper([
    ...               FxMapper('features', l2_normed),
    ...               FxMapper('samples', np.abs),
    ...               FxMapper('samples', np.mean)]),
    ...           # select 50% of the best on each step
    ...           fselector=FractionTailSelector(
    ...               0.50,
    ...               mode='select', tail='upper'),
    ...           # but we do want to update sensitivities on each step
    ...           update_sensitivity=True)
    >>> clf = FeatureSelectionClassifier(
    ...           LinearCSVMC(),
    ...           # on features selected via RFE
    ...           rfe,
    ...           # custom description
    ...           descr='LinSVM+RFE(splits_avg)' )


    But not only classifiers and their sensitivites could be used for RFE. It
    could be used even with univariate measures (e.g. OnewayAnova).
    """

    # exclude those since we are really an adapter here
    __init__doc__exclude__ = RFE.__init__doc__exclude__ + \
      ['fmeasure', 'pmeasure', 'splitter',
       'train_pmeasure', 'stopping_criterion',
       'bestdetector',   # now it is a diff strategy
       'nfeatures_min'   # will get 'trained'
       ]

    nested_errors = ConditionalAttribute(
        doc="History of errors per each nested split")
    nested_nfeatures = ConditionalAttribute(
        doc="History of # of features left per each nested split")

    def __init__(self, lrn, partitioner,
                 fselector,
                 errorfx=mean_mismatch_error,
                 fmeasure_postproc=None,
                 fmeasure=None,
                 nproc=1,
                 # callback?
                 **kwargs):
        """
        Parameters
        ----------
        lrn : Learner
          Learner with a sensitivity analyzer which will be used both
          for the sensitivity analysis and transfer error estimation
        partitioner : Partitioner
          Used to generate cross-validation partitions for cross-validation
          to deduce optimal number of features to maintain
        fselector : Functor
          Given a sensitivity map it has to return the ids of those
          features that should be kept.
        errorfx : func, optional
          Functor to use for estimation of cross-validation error
        fmeasure_postproc : func, optional
          Function to provide to the sensitivity analyzer as postproc.  If no
          fmeasure is provided and classifier sensitivity is used, then
          maxofabs_sample() would be used for this postproc, unless other
          value is provided
        fmeasure : Function, optional
          Featurewise measure.  If None was provided, lrn's sensitivity
          analyzer will be used.
        """
        # Initialize itself preparing for the 2nd invocation
        # with determined number of nfeatures_min
        # TODO:  move this into _train since better not to assign anything here
        # to avoid possible problems with copies needing to deal with the same
        # lrn... but then we might like again to reconsider delegation instead
        # of subclassing here....
        if fmeasure is None:
            if __debug__:
                debug('RFE', 'fmeasure was not provided, will be using the '
                             'sensitivity analyzer for %s' % lrn)
            fmeasure = lrn.get_sensitivity_analyzer(
                postproc=fmeasure_postproc if fmeasure_postproc is not None
                                           else maxofabs_sample())
            train_pmeasure = False
        else:
            assert fmeasure_postproc is None, "There should be no explicit " \
                    "fmeasure_postproc when fmeasure is specified"
            # if user provided explicit value -- use it! otherwise, we do want
            # to train an arbitrary fmeasure
            train_pmeasure = kwargs.pop('train_pmeasure', True)

        RFE.__init__(self,
                     fmeasure,
                     None,
                     Repeater(2),
                     fselector=fselector,
                     bestdetector=None,
                     train_pmeasure=train_pmeasure,
                     stopping_criterion=None,
                     **kwargs)
        self._lrn = lrn                   # should not be modified, thus _
        self.partitioner = partitioner
        self.errorfx = errorfx
        self.fmeasure_postproc = fmeasure_postproc
        self.nproc = nproc

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        return super(SplitRFE, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['lrn', 'partitioner'])
            + _repr_attrs(self, ['errorfx'], default=mean_mismatch_error)
            + _repr_attrs(self, ['fmeasure_postproc'], default=None)
            + _repr_attrs(self, ['nproc'], default=1)
            )


    @property
    def lrn(self):
        return self._lrn

    def _train(self, dataset):
        pmeasure = ProxyMeasure(self.lrn,
                                postproc=BinaryFxNode(self.errorfx,
                                                      self.lrn.space),
                                skip_train=not self.train_pmeasure   # do not train since fmeasure will
                                )

        # First we need to replicate our RFE construct but this time
        # with pmeasure for the classifier
        rfe = RFE(self.fmeasure,
                  pmeasure,
                  Splitter('partitions'),
                  fselector=self.fselector,
                  bestdetector=None,
                  train_pmeasure=self.train_pmeasure,
                  stopping_criterion=None,   # full "track"
                  update_sensitivity=self.update_sensitivity,
                  enable_ca=['errors', 'nfeatures'])

        errors, nfeatures = [], []

        if __debug__:
            debug("RFEC", "Stage 1: initial nested CV/RFE for %s", (dataset,))

        if self.nproc != 1 and externals.exists('joblib'):
            nested_results = jl.Parallel(self.nproc)(
                jl.delayed(_process_partition)(rfe, partition)
                for partition in self.partitioner.generate(dataset))
        else:
            nested_results = [
                _process_partition(rfe, partition)
                for partition in self.partitioner.generate(dataset)]

        # unzip
        errors = [x[0] for x in nested_results]
        nfeatures = [x[1] for x in nested_results]

        self.ca.nested_nfeatures = nfeatures
        self.ca.nested_errors = errors

        # mean errors across splits and find optimal number
        errors_mean = np.mean(errors, axis=0)
        nfeatures_mean = np.mean(nfeatures, axis=0)
        # we will take the "mean location" of the min to stay
        # within the most 'stable' choice

        mins_idx = np.where(errors_mean==np.min(errors_mean))[0]
        min_idx = mins_idx[int(len(mins_idx)/2)]
        min_error = errors_mean[min_idx]
        assert(min_error == np.min(errors_mean))
        nfeatures_min = nfeatures_mean[min_idx]

        if __debug__:
            debug("RFEC",
                  "Choosing among %d choices to have %d features with "
                  "mean error=%.2g (initial mean error %.2g)",
                  (len(mins_idx), nfeatures_min, min_error, errors_mean[0]))

        self.nfeatures_min = nfeatures_min

        if __debug__:
            debug("RFEC", "Stage 2: running RFE on full training dataset to "
                  "obtain the best %d features" % nfeatures_min)

        super(SplitRFE, self)._train(dataset)


    def _untrain(self):
        super(SplitRFE, self)._untrain()
        self.lrn.untrain()
        self.nfeatures_min = 0            # reset the knowledge

