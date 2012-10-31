# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Feature selection base class and related stuff base classes and helpers."""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.featsel.helpers import FractionTailSelector, \
                                 NBackHistoryStopCrit, \
                                 BestDetector
from mvpa2.mappers.slicing import SliceMapper
from mvpa2.mappers.base import accepts_dataset_as_samples
from mvpa2.base.dochelpers import _repr_attrs
from mvpa2.base.state import ConditionalAttribute
from mvpa2.generators.splitters import mask2slice

if __debug__:
    from mvpa2.base import debug


class FeatureSelection(SliceMapper):
    """Mapper to select a subset of features.

    Depending on the actual slicing two FeatureSelections can be merged in a
    number of ways: incremental selection (+=), union (&=) and intersection
    (|=).  Were the former assumes that two feature selections are applied
    subsequently, and the latter two assume that both slicings operate on the
    set of input features.

    Examples
    --------
    >>> from mvpa2.datasets import *
    >>> ds = Dataset([[1,2,3,4,5]])
    >>> fs0 = StaticFeatureSelection([0,1,2,3])
    >>> fs0(ds).samples
    array([[1, 2, 3, 4]])

    Merge two incremental selections: the resulting mapper performs a selection
    that is equivalent to first applying one slicing and subsequently the next
    slicing. In this scenario the slicing argument of the second mapper is
    relative to the output feature space of the first mapper.

    >>> fs1 = StaticFeatureSelection([0,2])
    >>> fs0 += fs1
    >>> fs0(ds).samples
    array([[1, 3]])
    """
    def __init__(self, filler=0, **kwargs):
        """
        Parameters
        ----------
        filler : optional
          Value to fill empty entries upon reverse operation
        """
        # init slicearg with None
        SliceMapper.__init__(self, None, **kwargs)
        self._dshape = None
        self._oshape = None
        self.filler = filler


    def __repr__(self, prefixes=[]):
        return super(FeatureSelection, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['filler'], default=0))


    def _forward_data(self, data):
        """Map data from the original dataspace into featurespace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        mdata = data[:, self._slicearg]
        # store the output shape if not set yet
        if self._oshape is None:
            self._oshape = mdata.shape[1:]
        return mdata


    def _forward_dataset(self, dataset):
        # XXX this should probably not affect the source dataset, but right now
        # init_origid is not flexible enough
        if not self.get_space() is None:
            # TODO need to do a copy first!!!
            dataset.init_origids('features', attr=self.get_space())
        # invoke super class _forward_dataset, this calls, _forward_dataset
        # and this calles _forward_data in this class
        mds = super(FeatureSelection, self)._forward_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now slice all feature attributes
        for k in mds.fa:
            mds.fa[k] = self.forward1(mds.fa[k].value)
        return mds


    def reverse1(self, data):
        # we need to reject inappropriate "single" samples to allow
        # chainmapper to properly switch to reverse() for multiple samples
        # use the fact that a single sample needs to conform to the known
        # data shape -- but may have additional appended dimensions
        if not data.shape[:len(self._oshape)] == self._oshape:
            raise ValueError("Data shape does not match training "
                             "(trained: %s; got: %s)"
                             % (self._dshape, data.shape))
        return super(FeatureSelection, self).reverse1(data)


    def _reverse_data(self, data):
        """Reverse map data from featurespace into the original dataspace.

        Parameters
        ----------
        data : array-like
          Either one-dimensional sample or two-dimensional samples matrix.
        """
        if self._dshape is None:
            raise RuntimeError(
                "Cannot reverse-map data since the original data shape is "
                "unknown. Either set `dshape` in the constructor, or call "
                "train().")
        # this wouldn't preserve ndarray subclasses
        #mapped = np.zeros(data.shape[:1] + self._dshape,
        #                 dtype=data.dtype)
        # let's do it a little awkward but pass subclasses through
        # suggestions for improvements welcome
        mapped = data.copy() # make sure we own the array data
        # "guess" the shape of the final array, the following only supports
        # changes in the second axis -- the feature axis
        # this madness is necessary to support mapping of multi-dimensional
        # features
        mapped.resize(data.shape[:1] + self._dshape + data.shape[2:],
                      refcheck=False)
        mapped.fill(self.filler)
        mapped[:, self._slicearg] = data
        return mapped


    def _reverse_dataset(self, dataset):
        # invoke super class _reverse_dataset, this calls, _reverse_dataset
        # and this calles _reverse_data in this class
        mds = super(FeatureSelection, self)._reverse_dataset(dataset)
        # attribute collection needs to have a new length check
        mds.fa.set_length_check(mds.nfeatures)
        # now reverse all feature attributes
        for k in mds.fa:
            mds.fa[k] = self.reverse1(mds.fa[k].value)
        return mds


    @accepts_dataset_as_samples
    def _train(self, data):
        if self._dshape is None:
            # XXX what about arrays of generic objects???
            # MH: in this case the shape will be (), which is just
            # fine since feature slicing is meaningless without features
            # the only thing we can do is kill the whole samples matrix
            self._dshape = data.shape[1:]
            # we also need to know what the output shape looks like
            # otherwise we cannot reliably say what is appropriate input
            # for reverse*()
            self._oshape = data[:, self._slicearg].shape[1:]


    def _untrain(self):
        if __debug__:
            debug("FS_", "Untraining FS: %s" % self)
        self._dshape = None
        self._oshape = None
        super(SliceMapper, self)._untrain()



class StaticFeatureSelection(FeatureSelection):
    """Feature selection by static slicing argument.
    """
    def __init__(self, slicearg, dshape=None, oshape=None, **kwargs):
        """
        Parameters
        ----------
        slicearg : int, list(int), array(int), array(bool)
          Any slicing argument that is compatible with numpy arrays. Depending
          on the argument the mapper will perform basic slicing or
          advanced indexing (with all consequences on speed and memory
          consumption).
        dshape : tuple
          Preseed the mappers input data shape (single sample shape).
        oshape: tuple
          Preseed the mappers output data shape (single sample shape).
        """
        FeatureSelection.__init__(self, **kwargs)
        # store it here, might be modified later
        self._dshape = self.__orig_dshape = dshape
        self._oshape = self.__orig_oshape = oshape
        # we also want to store the original slicearg to be able to reset to it
        # during training. Derived classes will override this default
        # implementation of _train()
        self.__orig_slicearg = slicearg
        self._safe_assign_slicearg(slicearg)

    def __repr__(self, prefixes=[]):
        return super(FeatureSelection, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['dshape', 'oshape']))

    @accepts_dataset_as_samples
    def _train(self, ds):
        # first thing is to reset the slicearg to the original value passed to
        # the constructor
        self._safe_assign_slicearg(self.__orig_slicearg)
        # not resetting {d,o}shape here as they will be handled upstream
        # and perform base training
        super(StaticFeatureSelection, self)._train(ds)


    def _untrain(self):
        # make trained again immediately
        self._safe_assign_slicearg(self.__orig_slicearg)
        self._dshape = self.__orig_dshape
        self._oshape = self.__orig_oshape
        super(FeatureSelection, self)._untrain()


    dshape = property(fget=lambda self: self.__orig_dshape)
    oshape = property(fget=lambda self: self.__orig_oshape)

class SensitivityBasedFeatureSelection(FeatureSelection):
    """Feature elimination.

    A `FeaturewiseMeasure` is used to compute sensitivity maps given a certain
    dataset. These sensitivity maps are in turn used to discard unimportant
    features.
    """

    sensitivity = ConditionalAttribute(enabled=False)

    def __init__(self,
                 sensitivity_analyzer,
                 feature_selector=FractionTailSelector(0.05),
                 train_analyzer=True,
                 **kwargs
                 ):
        """Initialize feature selection

        Parameters
        ----------
        sensitivity_analyzer : FeaturewiseMeasure
          sensitivity analyzer to come up with sensitivity
        feature_selector : Functor
          Given a sensitivity map it has to return the ids of those
          features that should be kept.
        train_analyzer : bool
          Flag whether to train the sensitivity analyzer on the input dataset
          during train(). If False, the employed sensitivity measure has to be
          already trained before.
        """

        # base init first
        FeatureSelection.__init__(self, **kwargs)

        self.__sensitivity_analyzer = sensitivity_analyzer
        """Sensitivity analyzer to use once"""

        self.__feature_selector = feature_selector
        """Functor which takes care about removing some features."""

        self.__train_analyzer = train_analyzer


    def _train(self, dataset):
        """Select the most important features

        Parameters
        ----------
        dataset : Dataset
          used to compute sensitivity maps
        """
        # optionally train the analyzer first
        if self.__train_analyzer:
            self.__sensitivity_analyzer.train(dataset)

        sensitivity = self.__sensitivity_analyzer(dataset)
        """Compute the sensitivity map."""

        self.ca.sensitivity = sensitivity

        # Select features to preserve
        selected_ids = self.__feature_selector(sensitivity)

        if __debug__:
            debug("FS_", "Sensitivity: %s Selected ids: %s" %
                  (sensitivity, selected_ids))

        # XXX not sure if it really has to be sorted
        selected_ids.sort()
        # announce desired features to the underlying slice mapper
        self._safe_assign_slicearg(selected_ids)
        # and perform its own training
        super(SensitivityBasedFeatureSelection, self)._train(dataset)


    def _untrain(self):
        if __debug__:
            debug("FS_", "Untraining sensitivity-based FS: %s" % self)
        self.__sensitivity_analyzer.untrain()
        # ask base class to do its untrain
        super(SensitivityBasedFeatureSelection, self)._untrain()

    # make it accessible from outside
    sensitivity_analyzer = property(fget=lambda self:self.__sensitivity_analyzer,
                                    doc="Measure which was used to do selection")



class IterativeFeatureSelection(FeatureSelection):
    """
    """
    errors = ConditionalAttribute(
        doc="History of errors")
    nfeatures = ConditionalAttribute(
        doc="History of # of features left")

    def __init__(self,
                 fmeasure,
                 pmeasure,
                 splitter,
                 fselector,
                 stopping_criterion=NBackHistoryStopCrit(BestDetector()),
                 bestdetector=BestDetector(),
                 train_pmeasure=True,
                 # XXX should we may be guard splitter so we do not end up
                 # with inappropriate one for the use, i.e. which
                 # generates more than 2 splits
                 # guard_splitter=True,
                 **kwargs
                 ):
        """
        Parameters
        ----------
        fmeasure : Measure
          Computed for each candidate feature selection. The measure has
          to compute a scalar value.
        pmeasure : Measure
          Compute against a test dataset for each incremental feature
          set.
        splitter: Splitter
          This splitter instance has to generate at least one dataset split
          when called with the input dataset that is used to compute the
          per-feature criterion for feature selection.
        bestdetector : Functor
          Given a list of error values it has to return a boolean that
          signals whether the latest error value is the total minimum.
        stopping_criterion : Functor
          Given a list of error values it has to return whether the
          criterion is fulfilled.
        fselector : Functor
        train_clf : bool
          Flag whether the classifier in `transfer_error` should be
          trained before computing the error. In general this is
          required, but if the `sensitivity_analyzer` and
          `transfer_error` share and make use of the same classifier it
          can be switched off to save CPU cycles. Default `None` checks
          if sensitivity_analyzer is based on a classifier and doesn't train
          if so.
        """
        # bases init first
        FeatureSelection.__init__(self, **kwargs)

        self._fmeasure = fmeasure
        self._pmeasure = pmeasure
        self._splitter = splitter
        self._fselector = fselector
        self._stopping_criterion = stopping_criterion
        self._bestdetector = bestdetector
        self._train_pmeasure = train_pmeasure


    def _untrain(self):
        if __debug__:
            debug("FS_", "Untraining Iterative FS: %s" % self)
        self._fmeasure.untrain()
        if self._pmeasure is not None:
            self._pmeasure.untrain()
        # ask base class to do its untrain
        super(IterativeFeatureSelection, self)._untrain()


    def _evaluate_pmeasure(self, train, test):
        # local binding
        pmeasure = self._pmeasure
        # might safe some cycles to prevent training the measure, but only
        # the user can know whether this is sensible or possible
        if self._train_pmeasure:
            pmeasure.train(train)
        # actually run the performance measure to estimate "quality" of
        # selection
        return pmeasure(test)


    def _get_traintest_ds(self, ds):
        # activate the dataset splitter
        dsgen = self._splitter.generate(ds)
        # and derived the dataset part that is used for computing the selection
        # criterion
        trainds = dsgen.next()
        testds = dsgen.next()
        return trainds, testds

    # access properties
    fmeasure = property(fget=lambda self: self._fmeasure)
    pmeasure = property(fget=lambda self: self._pmeasure)
    splitter = property(fget=lambda self: self._splitter)
    fselector = property(fget=lambda self: self._fselector)
    stopping_criterion = property(fget=lambda self: self._stopping_criterion)
    bestdetector = property(fget=lambda self: self._bestdetector)
    train_pmeasure = property(fget=lambda self: self._train_pmeasure)


class CombinedFeatureSelection(FeatureSelection):
    """Meta feature selection utilizing several embedded selection methods.

    During training each embedded feature selection method is computed
    individually. Afterwards all feature sets are combined by either taking the
    union or intersection of all sets.
    """
    def __init__(self, selectors, method, **kwargs):
        """
        Parameters
        ----------
        selectors : list
          FeatureSelection instances to run. Order is not important.
        method : {'union', 'intersection'}
          which method to be used to combine the feature selection set of
          all computed methods.
        """
        # by default -- auto_train
        kwargs['auto_train'] = kwargs.get('auto_train', True)
        FeatureSelection.__init__(self, **kwargs)

        self.__selectors = selectors
        self.__method = method


    def _untrain(self):
        if __debug__:
            debug("FS_", "Untraining combined FS: %s" % self)
        for fs in self.__selectors:
            fs.untrain()
        # ask base class to do its untrain
        super(CombinedFeatureSelection, self)._untrain()


    def _train(self, ds):
        # local binding
        method = self.__method

        # two major modes
        if method == 'union':
            # slice mask default: take none
            mask = np.zeros(ds.shape[1], dtype=np.bool)
            # method: OR
            cfunc = np.logical_or
        elif method == 'intersection':
            # slice mask default: take all
            mask = np.ones(ds.shape[1], dtype=np.bool)
            # method: AND
            cfunc = np.logical_and
        else:
            raise ValueError("Unknown combining method '%s'" % method)

        for fs in self.__selectors:
            # first: train all embedded selections
            fs.train(ds)
            # now get boolean mask of selections
            fsmask = np.zeros(mask.shape, dtype=np.bool)
            # use slicearg to select features
            fsmask[fs._slicearg] = True
            # merge with current global mask
            mask = cfunc(mask, fsmask)

        # turn the derived boolean mask into a slice if possible
        slicearg = mask2slice(mask)
        # and assign to baseclass, done
        self._safe_assign_slicearg(slicearg)

    method = property(fget=lambda self: self.__method)
    selectors = property(fget=lambda self: self.__selectors)
