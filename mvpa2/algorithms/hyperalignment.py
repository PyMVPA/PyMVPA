# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Transformation of individual feature spaces into a common space

The :class:`Hyperalignment` class in this module implements an algorithm
published in :ref:`Haxby et al., Neuron (2011) <HGC+11>` *A common,
high-dimensional model of the representational space in human ventral temporal
cortex.*

"""

__docformat__ = 'restructuredtext'

# don't leak the world
__all__ = ['Hyperalignment']

from mvpa2.support.copy import deepcopy

import numpy as np

from mvpa2.base.state import ConditionalAttribute, ClassWithCollections
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import *
from mvpa2.mappers.procrustean import ProcrusteanMapper
from mvpa2.datasets import Dataset
from mvpa2.mappers.base import ChainMapper
from mvpa2.mappers.zscore import zscore, ZScoreMapper
from mvpa2.mappers.staticprojection import StaticProjectionMapper

if __debug__:
    from mvpa2.base import debug

__all__ = [ "Hyperalignment" ]

class Hyperalignment(ClassWithCollections):
    """Align the features across multiple datasets into a common feature space.

    This is a three-level algorithm. In the first level, a series of input
    datasets is projected into a common feature space using a configurable
    mapper. The common space is initially defined by a chosen exemplar from the
    list of input datasets, but is subsequently refined by iteratively combining
    the common space with the projected input datasets.

    In the second (optional) level, the original input datasets are again
    aligned with (or projected into) the intermediate first-level common
    space. Through a configurable number of iterations the common space is
    further refined by repeated projections of the input datasets and
    combination/aggregation of these projections into an updated common space.

    In the third level, the input datasets are again aligned with the, now
    final, common feature space. The output of this algorithm are trained
    mappers (one for each input dataset) that transform the individual features
    spaces into the common space.

    Level 1 and 2 are performed by the ``train()`` method, and level 3 is
    performed when the trained Hyperalignment instance is called with a list of
    datasets. This dataset list may or may not be identical to the training
    datasets.

    The default values for the parameters of the algorithm (e.g. projection via
    Procrustean transformation, common space aggregation by averaging) resemble
    the setup reported in :ref:`Haxby et al., Neuron (2011) <HGC+11>` *A common,
    high-dimensional model of the representational space in human ventral
    temporal cortex.*

    Examples
    --------
    >>> # get some example data
    >>> from mvpa2.testing.datasets import datasets
    >>> from mvpa2.misc.data_generators import random_affine_transformation
    >>> ds4l = datasets['uni4large']
    >>> # generate a number of distorted variants of this data
    >>> dss = [random_affine_transformation(ds4l) for i in xrange(4)]
    >>> ha = Hyperalignment()
    >>> ha.train(dss)
    >>> mappers = ha(dss)
    >>> len(mappers)
    4
    """

    training_residual_errors = ConditionalAttribute(enabled=False,
            doc="""Residual error (norm of the difference between common space
                and projected data) per each training dataset at each level. The
                residuals are stored in a dataset with one row per level, and
                one column per input dataset. The first row corresponds to the
                error 1st-level of hyperalignment the remaining rows store the
                residual errors for each 2nd-level iteration.""")

    residual_errors = ConditionalAttribute(enabled=False,
            doc="""Residual error (norm of the difference between common space
                and projected data) per each dataset. The residuals are stored
                in a single-row dataset with one column per input dataset.""")

    # XXX Who cares whether it was chosen, or specified? This should be just
    # 'ref_ds'
    chosen_ref_ds = ConditionalAttribute(enabled=True,
            doc="""Index of the input dataset used as 1st-level reference
                dataset.""")

    # Lets use built-in facilities to specify parameters which
    # constructor should accept
    # the ``space`` of the mapper determines where the algorithm places the
    # common space definition in the datasets
    alignment = Parameter(ProcrusteanMapper(space='commonspace'), # might provide allowedtype
	    # XXX Currently, there's no way to handle this with connstraints  
            doc="""The multidimensional transformation mapper. If
            `None` (default) an instance of
            :class:`~mvpa2.mappers.procrustean.ProcrusteanMapper` is
            used.""")

    alpha = Parameter(1, constraints=EnsureFloat() & EnsureRange(min=0, max=1),
            doc="""Regularization parameter to traverse between (Shrinkage)-CCA
                (canonical correlation analysis) and regular hyperalignment.
                Setting alpha to 1 makes the algorithm identical to
                hyperalignment and alpha of 0 makes it CCA. By default,
                it is 1, therefore hyperalignment. """)

    level2_niter = Parameter(1, constraints=EnsureInt() & EnsureRange(min=0),
            doc="Number of 2nd-level iterations.")

    ref_ds = Parameter(None, constraints=(EnsureInt() & EnsureRange(min=0) 
                                          | EnsureNone()),
            doc="""Index of a dataset to use as 1st-level common space
                reference.  If `None`, then the dataset with the maximum
                number of features is used.""")

    zscore_all = Parameter(False, constraints='bool',
            doc="""Flag to Z-score all datasets prior hyperalignment.
            Turn it off if Z-scoring is not desired or was already performed.
            If True, returned mappers are ChainMappers with the Z-scoring
            prepended to the actual projection.""")

    zscore_common = Parameter(True, constraints='bool',
            doc="""Flag to Z-score the common space after each adjustment.
                This should be left enabled in most cases.""")

    combiner1 = Parameter(lambda x,y: 0.5*(x+y), #
            doc="""How to update common space in the 1st-level loop. This must
                be a callable that takes two arguments. The first argument is
                one of the input datasets after projection onto the 1st-level
                common space. The second argument is the current 1st-level
                common space. The 1st-level combiner is called iteratively for
                each projected input dataset, except for the reference dataset.
                By default the new common space is the average of the current
                common space and the recently projected dataset.""")

    combiner2 = Parameter(lambda l: np.mean(l, axis=0),
            doc="""How to combine all individual spaces to common space. This
            must be a callable that take a sequence of datasets as an argument.
            The callable must return a single array. This combiner is called
            once with all datasets after 1st-level projection to create an
            updated common space, and is subsequently called again after each
            2nd-level iteration.""")


    def __init__(self, **kwargs):
        ClassWithCollections.__init__(self, **kwargs)
        self.commonspace = None


    def train(self, datasets):
        """Derive a common feature space from a series of datasets.

        Parameters
        ----------
        datasets : sequence of datasets

        Returns
        -------
        A list of trained Mappers matching the number of input datasets.
        """
        params = self.params            # for quicker access ;)
        ca = self.ca
        ndatasets = len(datasets)
        nfeatures = [ds.nfeatures for ds in datasets]
        alpha = params.alpha
        
        residuals = None
        if ca['training_residual_errors'].enabled:
            residuals = np.zeros((1 + params.level2_niter, ndatasets))
            ca.training_residual_errors = Dataset(
                samples = residuals,
                sa = {'levels' :
                       ['1'] +
                       ['2:%i' % i for i in xrange(params.level2_niter)]})

        if __debug__:
            debug('HPAL', "Hyperalignment %s for %i datasets"
                  % (self, ndatasets))

        if params.ref_ds is None:
            ref_ds = np.argmax(nfeatures)
        else:
            ref_ds = params.ref_ds
            if ref_ds < 0 and ref_ds >= ndatasets:
                raise ValueError, "Requested reference dataset %i is out of " \
                      "bounds. We have only %i datasets provided" \
                      % (ref_ds, ndatasets)
        ca.chosen_ref_ds = ref_ds
        # zscore all data sets
        # ds = [ zscore(ds, chunks_attr=None) for ds in datasets]

        # TODO since we are doing in-place zscoring create deep copies
        # of the datasets with pruned targets and shallow copies of
        # the collections (if they would come needed in the transformation)
        # TODO: handle floats and non-floats differently to prevent
        #       waste of memory if there is no need (e.g. no z-scoring)
        #otargets = [ds.sa.targets for ds in datasets]
        datasets = [ds.copy(deep=False) for ds in datasets]
        #datasets = [Dataset(ds.samples.astype(float), sa={'targets': [None] * len(ds)})
        #datasets = [Dataset(ds.samples, sa={'targets': [None] * len(ds)})
        #            for ds in datasets]

        if params.zscore_all:
            if __debug__:
                debug('HPAL', "Z-scoring all datasets")
            for ids in xrange(len(datasets)):
                zmapper = ZScoreMapper(chunks_attr=None)
                zmapper.train(datasets[ids])
                datasets[ids] = zmapper.forward(datasets[ids])

        if alpha < 1:
            datasets, wmappers = self._regularize(datasets, alpha)

        # initial common space is the reference dataset
        commonspace = datasets[ref_ds].samples
        # the reference dataset might have been zscored already, don't do it
        # twice
        if params.zscore_common and not params.zscore_all:
            if __debug__:
                debug('HPAL_',
                      "Creating copy of a commonspace and assuring "
                      "it is of a floating type")
            commonspace = commonspace.astype(float)
            zscore(commonspace, chunks_attr=None)

        # create a mapper per dataset
        # might prefer some other way to initialize... later
        mappers = [deepcopy(params.alignment) for ds in datasets]

        #
        # Level 1 -- initial projection
        #
        lvl1_projdata = self._level1(datasets, commonspace, ref_ds, mappers,
                                     residuals)
        #
        # Level 2 -- might iterate multiple times
        #
        # this is the final common space
        self.commonspace = self._level2(datasets, lvl1_projdata, mappers,
                                        residuals)


    def __call__(self, datasets):
        """Derive a common feature space from a series of datasets.

        Parameters
        ----------
        datasets : sequence of datasets

        Returns
        -------
        A list of trained Mappers matching the number of input datasets.
        """
        if self.commonspace is None:
            self.train(datasets)

        # place datasets into a copy of the list since items
        # will be reassigned
        datasets = list(datasets)

        params = self.params            # for quicker access ;)
        alpha = params.alpha             # for letting me be lazy ;)
        if params.zscore_all:
            if __debug__:
                debug('HPAL', "Z-scoring all datasets")
            # zscore them once while storing corresponding ZScoreMapper's
            # so we can assemble a comprehensive mapper at the end
            # (together with procrustes)
            zmappers = []
            for ids in xrange(len(datasets)):
                zmapper = ZScoreMapper(chunks_attr=None)
                zmappers.append(zmapper)
                zmapper.train(datasets[ids])
                datasets[ids] = zmapper.forward(datasets[ids])

        if alpha < 1:
            datasets, wmappers = self._regularize(datasets, alpha)

        #
        # Level 3 -- final, from-scratch, alignment to final common space
        #
        mappers = self._level3(datasets)
        # return trained mappers for projection from all datasets into the
        # common space
        if params.zscore_all:
            # We need to construct new mappers which would chain
            # zscore and then final transformation
            if params.alpha < 1:
                return [ChainMapper([zm, wm, m]) for zm, wm, m in zip(zmappers, wmappers, mappers)]
            else:
                return [ChainMapper([zm, m]) for zm, m in zip(zmappers, mappers)]
        else:
            if params.alpha < 1:
                return [ChainMapper([wm, m]) for wm, m in zip(wmappers, mappers)]
            else:
                return mappers


    def _regularize(self, datasets, alpha):
        if __debug__:
            debug('HPAL', "Using regularized hyperalignment with alpha of %d"
                    % alpha)
        wmappers = []
        for ids in xrange(len(datasets)):
            U, S, Vh = np.linalg.svd(datasets[ids])
            S = 1/np.sqrt( (1-alpha)*np.square(S) + alpha )
            S.resize(len(Vh))
            S = np.matrix(np.diag(S))
            W = np.matrix(Vh.T)*S*np.matrix(Vh)
            wmapper = StaticProjectionMapper(proj=W, auto_train=False)
            wmapper.train(datasets[ids])
            wmappers.append(wmapper)
            datasets[ids] = wmapper.forward(datasets[ids])
        return datasets, wmappers


    def _level1(self, datasets, commonspace, ref_ds, mappers, residuals):
        params = self.params            # for quicker access ;)
        data_mapped = [ds.samples for ds in datasets]
        for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
            if __debug__:
                debug('HPAL_', "Level 1: ds #%i" % i)
            if i == ref_ds:
                continue
            # assign common space to ``space`` of the mapper, because this is
            # where it will be looking for it
            ds_new.sa[m.get_space()] = commonspace
            # find transformation of this dataset into the current common space
            m.train(ds_new)
            # remove common space attribute again to save on memory when the
            # common space is updated for the next iteration
            del ds_new.sa[m.get_space()]
            # project this dataset into the current common space
            ds_ = m.forward(ds_new.samples)
            if params.zscore_common:
                zscore(ds_, chunks_attr=None)
            # replace original dataset with mapped one -- only the reference
            # dataset will remain unchanged
            data_mapped[i] = ds_

            # compute first-level residuals wrt to the initial common space
            if residuals is not None:
                residuals[0, i] = np.linalg.norm(ds_ - commonspace)

            # Update the common space. This is an incremental update after
            # processing each 1st-level dataset. Maybe there should be a flag
            # to make a batch update after processing all 1st-level datasets
            # to an identical 1st-level common space
            # TODO: make just a function so we dont' waste space
            commonspace = params.combiner1(ds_, commonspace)
            if params.zscore_common:
                zscore(commonspace, chunks_attr=None)
        return data_mapped


    def _level2(self, datasets, lvl1_data, mappers, residuals):
        params = self.params            # for quicker access ;)
        data_mapped = lvl1_data
        # aggregate all processed 1st-level datasets into a new 2nd-level
        # common space
        commonspace = params.combiner2(data_mapped)

        # XXX Why is this commented out? Who knows what combiner2 is doing and
        # whether it changes the distribution of the data
        #if params.zscore_common:
        #zscore(commonspace, chunks_attr=None)

        ndatasets = len(datasets)
        for loop in xrange(params.level2_niter):
            # 2nd-level alignment starts from the original/unprojected datasets
            # again
            for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
                if __debug__:
                    debug('HPAL_', "Level 2 (%i-th iteration): ds #%i" % (loop, i))

                # Optimization speed up heuristic
                # Slightly modify the common space towards other feature
                # spaces and reduce influence of this feature space for the
                # to-be-computed projection
                temp_commonspace = (commonspace * ndatasets - data_mapped[i]) \
                                    / (ndatasets - 1)

                if params.zscore_common:
                    zscore(temp_commonspace, chunks_attr=None)
                # assign current common space
                ds_new.sa[m.get_space()] = temp_commonspace
                # retrain the mapper for this dataset
                m.train(ds_new)
                # remove common space attribute again to save on memory when the
                # common space is updated for the next iteration
                del ds_new.sa[m.get_space()]
                # obtain the 2nd-level projection
                ds_ =  m.forward(ds_new.samples)
                if params.zscore_common:
                    zscore(ds_, chunks_attr=None)
                # store for 2nd-level combiner
                data_mapped[i] = ds_
                # compute residuals
                if residuals is not None:
                    residuals[1+loop, i] = np.linalg.norm(ds_ - commonspace)

            commonspace = params.combiner2(data_mapped)

        # and again
        if params.zscore_common:
            zscore(commonspace, chunks_attr=None)

        # return the final common space
        return commonspace


    def _level3(self, datasets):
        params = self.params            # for quicker access ;)
        # create a mapper per dataset
        mappers = [deepcopy(params.alignment) for ds in datasets]

        # key different from level-2; the common space is uniform
        #temp_commonspace = commonspace

        residuals = None
        if self.ca['residual_errors'].enabled:
            residuals = np.zeros((1, len(datasets)))
            self.ca.residual_errors = Dataset(samples=residuals)

        # start from original input datasets again
        for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
            if __debug__:
                debug('HPAL_', "Level 3: ds #%i" % i)

            # retrain mapper on final common space
            ds_new.sa[m.get_space()] = self.commonspace
            m.train(ds_new)
            # remove common space attribute again to save on memory
            del ds_new.sa[m.get_space()]

            if residuals is not None:
                # obtain final projection
                data_mapped = m.forward(ds_new.samples)
                residuals[0, i] = np.linalg.norm(data_mapped - self.commonspace)

        return mappers

