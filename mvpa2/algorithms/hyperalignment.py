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
    alignment = Parameter(ProcrusteanMapper(), # might provide allowedtype
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
            doc="How to update common space in the 1st loop")

    combiner2 = Parameter(lambda l: np.mean(l, axis=0),
            doc="How to combine all individual spaces to common space.")

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
        # might prefer some other way to initialize... later
        mappers = [deepcopy(params.alignment) for ds in datasets]
        # zscore all data sets
        # ds = [ zscore(ds, chunks_attr=None) for ds in datasets]

        # Level 1 (first)

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
            zmappers = []
            for ids in xrange(len(datasets)):
                zmapper = ZScoreMapper(chunks_attr=None)
                zmappers.append(zmapper)
                zmapper.train(datasets[ids])
                datasets[ids] = zmapper.forward(datasets[ids])

        commonspace = np.asanyarray(datasets[ref_ds])
        if params.zscore_common and not params.zscore_all:
            if __debug__:
                debug('HPAL_',
                      "Creating copy of a commonspace and assuring "
                      "it is of a floating type")
            commonspace = commonspace.astype(float)
            zscore(commonspace, chunks_attr=None)

        data_mapped = [np.asanyarray(ds) for ds in datasets]
        #zscore(data_mapped[ref_ds],chunks_attr=None)
        for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
            if __debug__:
                debug('HPAL_', "Level 1: ds #%i" % i)
            if i == ref_ds:
                continue
            #ds_new = ds.copy()
            #zscore(ds_new, chunks_attr=None);
            ds_new.targets = commonspace
            m.train(ds_new)
            ds_ = m.forward(np.asanyarray(ds_new))
            if params.zscore_common:
                zscore(ds_, chunks_attr=None)
            data_mapped[i] = ds_

            if residuals is not None:
                residuals[0, i] = np.linalg.norm(ds_ - commonspace)

            ## if ds_mapped == []:
            ##     ds_mapped = [zscore(m.forward(d), chunks_attr=None)]
            ## else:
            ##     ds_mapped += [zscore(m.forward(d), chunks_attr=None)]

            # zscore before adding
            # TODO: make just a function so we dont' waste space
            commonspace = params.combiner1(data_mapped[i], commonspace)
            if params.zscore_common:
                zscore(commonspace, chunks_attr=None)

        # update commonspace to mean of ds_mapped
        commonspace = params.combiner2(data_mapped)
        #if params.zscore_common:
        #zscore(commonspace, chunks_attr=None)
        # Level 2 -- might iterate multiple times
        for loop in xrange(params.level2_niter):
            for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
                if __debug__:
                    debug('HPAL_', "Level 2 (%i-th iteration): ds #%i" % (loop, i))

                ds_temp = (commonspace*ndatasets - data_mapped[i])/(ndatasets-1)
                if params.zscore_common:
                    zscore(ds_temp, chunks_attr=None)
                #ds_new = ds.copy()
                #zscore(ds_new, chunks_attr=None)
                ds_new.targets = ds_temp #commonspace #PRJ ds_temp
                m.train(ds_new) # ds_temp)
                ds_ =  m.forward(np.asanyarray(ds_new))
                if params.zscore_common:
                    zscore(ds_, chunks_attr=None)
                data_mapped[i] = ds_
                if residuals is not None:
                    residuals[1+loop, i] = np.linalg.norm(ds_ - commonspace)

                #ds_mapped[i] = zscore( m.forward(ds_temp), chunks_attr=None)

            commonspace = params.combiner2(data_mapped)
            #if params.zscore_common:
                #zscore(commonspace, chunks_attr=None)

        # Level 3 (last) to params.levels
        for i, (m, ds_new) in enumerate(zip(mappers, datasets)):
            if __debug__:
                debug('HPAL_', "Level 3: ds #%i" % i)

            #ds_new = ds.copy()     # shallow copy so we could assign new labels
            #zscore(ds_new, chunks_attr=None)
            ds_temp = commonspace
            #ds_temp = (commonspace*ndatasets - data_mapped[i])/(ndatasets-1)
            if params.zscore_common:
                zscore(ds_temp, chunks_attr=None)
            ds_new.targets = ds_temp #commonspace #PRJ ds_temp#
            m.train(ds_new) #ds_temp)
            data_mapped[i] = m.forward(np.asanyarray(ds_new))
            if residuals is not None:
                residuals[-1, i] = np.linalg.norm(data_mapped[i] - commonspace)

        if params.zscore_all:
            # We need to construct new mappers which would chain
            # zscore and then final transformation
            return [ChainMapper([zm, m]) for zm, m in zip(zmappers, mappers)]
        else:
            return mappers

