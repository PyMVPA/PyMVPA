# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Hyperalignment of functional data to the common space

References: TODO...

see SMLR code for example on how to embed the reference so in future
it gets properly referenced...
"""

__docformat__ = 'restructuredtext'

from mvpa2.support.copy import deepcopy

import numpy as np

from mvpa2.base.state import ConditionalAttribute, ClassWithCollections
from mvpa2.base.param import Parameter
from mvpa2.mappers.procrustean import ProcrusteanMapper
from mvpa2.datasets import Dataset
from mvpa2.mappers.base import ChainMapper
from mvpa2.mappers.zscore import zscore, ZScoreMapper

if __debug__:
    from mvpa2.base import debug


class Hyperalignment(ClassWithCollections):
    """ ...

    Given a set of datasets (may be just data) provide mapping of
    features into a common space
    """

    residual_errors = ConditionalAttribute(enabled=False,
            doc="""Residual error per each dataset at each level.""")

    choosen_ref_ds = ConditionalAttribute(enabled=True,
            doc="""If ref_ds wasn't provided, it gets choosen.""")

    # Lets use built-in facilities to specify parameters which
    # constructor should accept
    # the ``space`` of the mapper determines where the algorithm places the
    # common space definition in the datasets
    alignment = Parameter(ProcrusteanMapper(space='commonspace'), # might provide allowedtype
            allowedtype='basestring',
            doc="""The multidimensional transformation mapper. If
            `None` (default) an instance of
            :class:`~mvpa2.mappers.procrustean.ProcrusteanMapper` is
            used.""")

    level2_niter = Parameter(1, allowedtype='int', min=0,
            doc="Number of 2nd level iterations.")

    ref_ds = Parameter(None, allowedtype='int', min=0,
            doc="""Index of a dataset to use as a reference.  If `None`, then
            dataset with maximal number of features is used.""")

    zscore_all = Parameter(False, allowedtype='bool',
            doc="""Z-score all datasets prior hyperalignment.  Turn it off
            if zscoring is not desired or was already performed. If on,
            resultant mapping becomes a chain with ZScoreMapper""")

    zscore_common = Parameter(True, allowedtype='bool',
            doc="""Z-score common space after each adjustment.""")

    combiner1 = Parameter(lambda x,y: 0.5*(x+y), #
            doc="""How to update common space in the 1st-level loop. This must
                be a callable that takes two arguments. The first argument is
                one of the input dataset after projection onto the 1st-level
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


    def __call__(self, datasets):
        """Estimate mappers for each dataset

        Parameters
        ----------
          datasets : list or tuple of datasets

        Returns
        -------
        A list of trained Mappers of the same length as datasets
        """
        params = self.params            # for quicker access ;)
        ca = self.ca
        ndatasets = len(datasets)
        nfeatures = [ds.nfeatures for ds in datasets]

        residuals = None
        if ca['residual_errors'].enabled:
            residuals = np.zeros((2 + params.level2_niter, ndatasets))
            ca.residual_errors = Dataset(
                samples = residuals,
                sa = {'levels' :
                       ['1'] +
                       ['2:%i' % i for i in xrange(params.level2_niter)] +
                       ['3']})

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
        ca.choosen_ref_ds = ref_ds
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
            # zscore them once while storing corresponding ZScoreMapper's
            # so we can assemble a comprehensive mapper at the end
            # (together with procrustes)
            zmappers = []
            for ids in xrange(len(datasets)):
                zmapper = ZScoreMapper(chunks_attr=None)
                zmappers.append(zmapper)
                zmapper.train(datasets[ids])
                datasets[ids] = zmapper.forward(datasets[ids])

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
        commonspace = self._level2(datasets, lvl1_projdata, mappers, residuals)
        #
        # Level 3 -- final, from-scratch, alignment to final common space
        #
        mappers = self._level3(datasets, commonspace, mappers, residuals)
        # return trained mappers for projection from all datasets into the
        # common space
        return mappers


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

                # XXX this step is not mentioned in the paper
                # why is the common space modified before alignment? This is
                # what is different in level-2 and level-3, so it should be
                # explained somewhere
                temp_commonspace = (commonspace * ndatasets - data_mapped[i]) \
                                    / (ndatasets - 1)

                if params.zscore_common:
                    zscore(temp_commonspace, chunks_attr=None)
                # assign current common space
                ds_new.sa[m.get_space()] = temp_commonspace
                # retrain the mapper for this dataset
                m.train(ds_new)
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
        # return the final common space
        return commonspace


    def _level3(self, datasets, commonspace, mappers, residuals):
        params = self.params            # for quicker access ;)
        # start from original input datasets again
        for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
            if __debug__:
                debug('HPAL_', "Level 3: ds #%i" % i)

            # key different to level-2; the common space is uniform
            temp_commonspace = commonspace
            # and again
            # XXX Why is an unmodified common space zscore over and over again?
            if params.zscore_common:
                zscore(temp_commonspace, chunks_attr=None)

            # retrain mapper on final common space
            ds_new.sa[m.get_space()] = temp_commonspace
            m.train(ds_new)
            if residuals is not None:
                # obtain final projection
                data_mapped = m.forward(ds_new.samples)
                residuals[-1, i] = np.linalg.norm(data_mapped - commonspace)

        if params.zscore_all:
            # We need to construct new mappers which would chain
            # zscore and then final transformation
            return [ChainMapper([zm, m]) for zm, m in zip(zmappers, mappers)]
        else:
            return mappers

