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
from mvpa2.mappers.svd import SVDMapper

from mvpa2.support.due import due, Doi

if __debug__:
    from mvpa2.base import debug

__all__ = ["Hyperalignment"]

#
# Helper functions which will be used as defaults for Hyperalignment parameters
# to avoid lambdas (we fail to serialize them ATM in h5save) and to
# provide easier to comprehend repr of their values
#

def mean_xy(x, y, weights=(.5, .5)):
    return (weights[0] * x + weights[1] * y) / (weights[0] + weights[1])


def mean_axis0(a):
    return np.mean(a, axis=0)


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
    alignment = Parameter(ProcrusteanMapper(space='commonspace'),
            # might provide allowedtype
            # XXX Currently, there's no way to handle this with constraints
            doc="""The multidimensional transformation mapper. If
            `None` (default) an instance of
            :class:`~mvpa2.mappers.procrustean.ProcrusteanMapper` is
            used.""")
    output_dim = Parameter(None, constraints=(EnsureInt() & EnsureRange(min=1)| EnsureNone()),
            doc="""Output common space dimensionality. If None, datasets are aligned
             to the features of the `ref_ds`. Otherwise, dimensionality reduction is
             performed using SVD and only the top SVs are kept. To get all features in
             SVD-aligned space, give output_dim>=nfeatures.
            """)

    alpha = Parameter(1, constraints=EnsureFloat() & EnsureRange(min=0, max=1),
            doc="""Regularization parameter to traverse between (Shrinkage)-CCA
                (canonical correlation analysis) and regular hyperalignment.
                Setting alpha to 1 makes the algorithm identical to
                hyperalignment and alpha of 0 makes it CCA. By default,
                it is 1, therefore hyperalignment. """)

    level2_niter = Parameter(1, constraints=EnsureInt() & EnsureRange(min=0),
            doc="Number of 2nd-level iterations.")

    ref_ds = Parameter(None, constraints=(EnsureRange(min=0) & EnsureInt()
                                          | EnsureNone()),
            doc="""Index of a dataset to use as 1st-level common space
                reference.  If `None`, then the dataset with the maximum
                number of features is used.""")

    nproc = Parameter(1, constraints=EnsureInt(),
            doc="""Number of processes to use to parallelize the last step of
                alignment. If different from 1, it passes it as n_jobs to
                `joblib.Parallel`. Requires joblib package.""")

    zscore_all = Parameter(False, constraints='bool',
            doc="""Flag to Z-score all datasets prior hyperalignment.
            Turn it off if Z-scoring is not desired or was already performed.
            If True, returned mappers are ChainMappers with the Z-scoring
            prepended to the actual projection.""")

    zscore_common = Parameter(True, constraints='bool',
            doc="""Flag to Z-score the common space after each adjustment.
                This should be left enabled in most cases.""")

    combiner1 = Parameter(mean_xy,  #
            doc="""How to update common space in the 1st-level loop. This must
                be a callable that takes two arguments. The first argument is
                one of the input datasets after projection onto the 1st-level
                common space. The second argument is the current 1st-level
                common space. The 1st-level combiner is called iteratively for
                each projected input dataset, except for the reference dataset.
                By default the new common space is the average of the current
                common space and the recently projected dataset.""")

    level1_equal_weight = Parameter(False, constraints='bool',
            doc="""Flag to force all datasets to have the same weight in the
            level 1 iteration. False (default) means each time the new common
            space is the average of the current common space and the newly
            aligned dataset, and therefore earlier datasets have less weight.""")

    combiner2 = Parameter(mean_axis0,
            doc="""How to combine all individual spaces to common space. This
            must be a callable that take a sequence of datasets as an argument.
            The callable must return a single array. This combiner is called
            once with all datasets after 1st-level projection to create an
            updated common space, and is subsequently called again after each
            2nd-level iteration.""")

    joblib_backend = Parameter(None, constraints=EnsureChoice('multiprocessing',
                                                    'threading') | EnsureNone(),
            doc="""Backend to use for joblib when using nproc>1.
            Options are 'multiprocessing' and 'threading'. Default is to use
            'multiprocessing' unless run on OSX which have known issues with
            joblib v0.10.3. If it is set to specific value here, then that will
            be used at the risk of failure.""")

    def __init__(self, **kwargs):
        ClassWithCollections.__init__(self, **kwargs)
        self.commonspace = None
        # mapper to a low-dimensional subspace derived using SVD on training data
        # Initializing here so that call can access it without passing after train.
        # Moreover, it is similar to commonspace, in that, it is required for mapping
        # new subjects
        self._svd_mapper = None


    @due.dcite(
        Doi('10.1016/j.neuron.2011.08.026'),
        description="Hyperalignment of data to a common space",
        tags=["implementation"])
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
        # Check to make sure we get a list of datasets as input.
        if not isinstance(datasets, (list, tuple, np.ndarray)):
            raise TypeError("Input datasets should be a sequence "
                            "(of type list, tuple, or ndarray) of datasets.")

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
            # Making sure that ref_ds is within range.
            #Parameter() already checks for it being a non-negative integer
            if ref_ds >= ndatasets:
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
        # If there is only one dataset in training phase, there is nothing to be done
        # just use that data as the common space
        if len(datasets) < 2:
            self.commonspace = commonspace
        else:
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
        if params.output_dim is not None:
            mappers = self._level3(datasets)
            self._svd_mapper = SVDMapper()
            self._svd_mapper.train(self._map_and_mean(datasets, mappers))
            self._svd_mapper = StaticProjectionMapper(
                proj=self._svd_mapper.proj[:, :params.output_dim])

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
        else:
            # Check to make sure we get a list of datasets as input.
            if not isinstance(datasets, (list, tuple, np.ndarray)):
                raise TypeError("Input datasets should be a sequence "
                                "(of type list, tuple, or ndarray) of datasets.")

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
                mappers = [ChainMapper([zm, wm, m]) for zm, wm, m in zip(zmappers, wmappers, mappers)]
            else:
                mappers = [ChainMapper([zm, m]) for zm, m in zip(zmappers, mappers)]
        elif params.alpha < 1:
            mappers = [ChainMapper([wm, m]) for wm, m in zip(wmappers, mappers)]
        if params.output_dim is not None:
            mappers = [ChainMapper([m, self._svd_mapper]) for m in mappers]
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
        counts = 1  # number of datasets used so far for generating commonspace
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
            if params.level1_equal_weight:
                commonspace = params.combiner1(ds_, commonspace,
                                               weights=(float(counts), 1.0))
            else:
                commonspace = params.combiner1(ds_, commonspace)
            counts += 1
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
                ds_ = m.forward(ds_new.samples)
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
        # Fixing nproc=0
        if params.nproc == 0:
            from mvpa2.base import warning
            warning("nproc of 0 doesn't make sense. Setting nproc to 1.")
            params.nproc = 1
        # Checking for joblib, if not, set nproc to 1
        if params.nproc != 1:
            from mvpa2.base import externals, warning
            if not externals.exists('joblib'):
                warning("Setting nproc different from 1 requires joblib package, which "
                        "does not seem to exist. Setting nproc to 1.")
                params.nproc = 1

        # start from original input datasets again
        if params.nproc == 1:
            residuals = []
            for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
                if __debug__:
                    debug('HPAL_', "Level 3: ds #%i" % i)
                m, residual = get_trained_mapper(ds_new, self.commonspace, m,
                                                 self.ca['residual_errors'].enabled)
                if self.ca['residual_errors'].enabled:
                    residuals.append(residual)
        else:
            if __debug__:
                debug('HPAL_', "Level 3: Using joblib with nproc = %d " % params.nproc)
            verbose_level_parallel = 20 \
                if (__debug__ and 'HPAL' in debug.active) else 0
            from joblib import Parallel, delayed
            import sys
            # joblib's 'multiprocessing' backend has known issues of failure on OSX
            # Tested with MacOS 10.12.13, python 2.7.13, joblib v0.10.3
            if params.joblib_backend is None:
                params.joblib_backend = 'threading' if sys.platform == 'darwin' \
                                        else 'multiprocessing'
            res = Parallel(
                    n_jobs=params.nproc, pre_dispatch=params.nproc,
                    backend=params.joblib_backend,
                    verbose=verbose_level_parallel
                    )(
                        delayed(get_trained_mapper)
                        (ds, self.commonspace, mapper, self.ca['residual_errors'].enabled)
                        for ds, mapper in zip(datasets, mappers)
                    )
            mappers = [m for m, r in res]
            if self.ca['residual_errors'].enabled:
                residuals = [r for m, r in res]

        if self.ca['residual_errors'].enabled:
            self.ca.residual_errors = Dataset(samples=np.array(residuals)[None, :])

        return mappers

    def _map_and_mean(self, datasets, mappers):
        params = self.params
        data_mapped = [[] for ds in datasets]
        for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
            if __debug__:
                debug('HPAL_', "Mapping training data for SVD: ds #%i" % i)
            ds_ = m.forward(ds_new.samples)
            # XXX should we zscore data before averaging and running SVD?
            # zscore(ds_, chunks_attr=None)
            data_mapped[i] = ds_
        dss_mean = params.combiner2(data_mapped)
        return dss_mean


def get_trained_mapper(ds, commonspace, mapper, compute_residual=False):
    """
    Trains a given mapper using dataset and commonspace and computes residuals if
    necessary.

    Parameters
    ----------
    ds: dataset
        A dataset
    commonspace: ndarray
        Commonspace data.
    mapper: Mapper
        Typically ProcrusteanMapper.
    compute_residual: bool
        Whether to compute residuals or not. Default is False and returns None.
    """
    # retrain mapper on final common space
    ds.sa[mapper.get_space()] = commonspace
    mapper.train(ds)
    # remove common space attribute again to save on memory
    del ds.sa[mapper.get_space()]
    residual = None
    if compute_residual:
        # obtain final projection
        data_mapped = mapper.forward(ds.samples)
        residual = np.linalg.norm(data_mapped - commonspace)
    return mapper, residual
