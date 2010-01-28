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

from mvpa.support.copy import deepcopy

import numpy as N

from mvpa.misc.state import StateVariable, ClassWithCollections
from mvpa.misc.param import Parameter
from mvpa.misc.transformers import GrandMean
from mvpa.mappers.procrustean import ProcrusteanMapper
from mvpa.datasets import dataset_wizard, Dataset
from mvpa.mappers.zscore import zscore

if __debug__:
    from mvpa.base import debug


class Hyperalignment(ClassWithCollections):
    """ ...

    Given a set of datasets (may be just data) provide mapping of
    features into a common space
    """

    residual_errors = StateVariable(enabled=False,
            doc="""Residual error per each dataset at each level.""")

    choosen_ref_ds = StateVariable(enabled=True,
            doc="""If ref_ds wasn't provided, it gets choosen.""")

    # Lets use built-in facilities to specify parameters which
    # constructor should accept
    alignment = Parameter(ProcrusteanMapper(), # might provide allowedtype
            doc="""... XXX If `None` (default) an instance of
            :class:`~mvpa.mappers.procrustean.ProcrusteanMapper` is
            used.""")

    level2_niter = Parameter(1, allowedtype='int', min=0,
            doc="Number of 2nd level iterations.")

    ref_ds = Parameter(None, allowedtype='int', min=0,
            doc="""Index of a dataset to use as a reference.  If `None`, then
            dataset with maximal number of features is used.""")

    zscore_common = Parameter(False, allowedtype='bool',
            doc="""Z-score common space after each adjustment.  Might prove
            to be useful.  !!WiP!!""")

    combiner1 = Parameter(lambda x,y: 0.5*(x+y), #
            doc="How to update common space in the 1st loop")

    combiner2 = Parameter(lambda l: N.mean(l, axis=0),
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
        states = self.states
        ndatasets = len(datasets)
        nfeatures = [ds.nfeatures for ds in datasets]

        residuals = None
        if states['residual_errors'].enabled:
            residuals = N.zeros((2 + params.level2_niter, ndatasets))
            states.residual_errors = Dataset(
                samples = residuals,
                sa = {'levels' :
                       ['1'] +
                       ['2:%i' % i for i in xrange(params.level2_niter)] +
                       ['3']})

        if __debug__:
            debug('HPAL', "Hyperalignment %s for %i datasets"
                  % (self, ndatasets))

        if params.ref_ds is None:
            ref_ds = N.argmax(nfeatures)
        else:
            ref_ds = params.ref_ds
            if ref_ds < 0 and ref_ds >= ndatasets:
                raise ValueError, "Requested reference dataset %i is out of " \
                      "bounds. We have only %i datasets provided" \
                      % (ref_ds, ndatasets)
        states.choosen_ref_ds = ref_ds
        # might prefer some other way to initialize... later
        mappers = [deepcopy(params.alignment) for ds in datasets]
        # zscore all data sets
        # ds = [ zscore(ds, perchunk=False) for ds in datasets]

        # Level 1 (first)
        commonspace = N.asanyarray(datasets[ref_ds])
        if params.zscore_common:
            zscore(commonspace, chunks=None)
        data_mapped = [N.asanyarray(ds) for ds in datasets]
        for i, (m, data) in enumerate(zip(mappers, data_mapped)):
            if __debug__:
                debug('HPAL_', "Level 1: ds #%i" % i)
            if i == ref_ds:
                continue
            #ZSC zscore(data, perchunk=False)
            ds = dataset_wizard(samples=data, labels=commonspace)
            #ZSC zscore(ds, perchunk=False)
            m.train(ds)
            data_temp = m.forward(data)
            #ZSC zscore(data_temp, perchunk=False)
            data_mapped[i] = data_temp

            if residuals is not None:
                residuals[0, i] = N.linalg.norm(data_temp - commonspace)

            ## if ds_mapped == []:
            ##     ds_mapped = [zscore(m.forward(d), perchunk=False)]
            ## else:
            ##     ds_mapped += [zscore(m.forward(d), perchunk=False)]

            # zscore before adding
            # TODO: make just a function so we dont' waste space
            commonspace = params.combiner1(data_mapped[i], commonspace)
            if params.zscore_common:
                zscore(commonspace, chunks=None)

        # update commonspace to mean of ds_mapped
        commonspace = params.combiner2(data_mapped)
        if params.zscore_common:
            zscore(commonspace, chunks=None)

        # Level 2 -- might iterate multiple times
        for loop in xrange(params.level2_niter):
            for i, (m, ds) in enumerate(zip(mappers, datasets)):
                if __debug__:
                    debug('HPAL_', "Level 2 (%i-th iteration): ds #%i" % (loop, i))

                ## ds_temp = zscore( (commonspace*ndatasets - ds_mapped[i])
                ##                   /(ndatasets-1), perchunk=False )
                ds_new = ds.copy()
                #ZSC zscore(ds_new, perchunk=False)
                #PRJ ds_temp = (commonspace*ndatasets - ds_mapped[i])/(ndatasets-1)
                #ZSC zscore(ds_temp, perchunk=False)
                ds_new.labels = commonspace #PRJ ds_temp
                m.train(ds_new) # ds_temp)
                data_mapped[i] = m.forward(N.asanyarray(ds))
                if residuals is not None:
                    residuals[1+loop, i] = N.linalg.norm(data_mapped - commonspace)

                #ds_mapped[i] = zscore( m.forward(ds_temp), perchunk=False)

            commonspace = params.combiner2(data_mapped)
            if params.zscore_common:
                zscore(commonspace, chunks=None)

        # Level 3 (last) to params.levels
        for i, (m, ds) in enumerate(zip(mappers, datasets)):
            if __debug__:
                debug('HPAL_', "Level 3: ds #%i" % i)

            ## ds_temp = zscore( (commonspace*ndatasets - ds_mapped[i])
            ##                   /(ndatasets-1), perchunk=False )
            ds_new = ds.copy()     # shallow copy so we could assign new labels
            #ZSC zscore(ds_new, perchunk=False)
            #PRJ ds_temp = (commonspace*ndatasets - ds_mapped[i])/(ndatasets-1)
            #ZSC zscore(ds_temp, perchunk=False)
            ds_new.labels = commonspace #PRJ ds_temp#
            m.train(ds_new) #ds_temp)

            if residuals is not None:
                data_mapped = m.forward(ds_new)
                residuals[-1, i] = N.linalg.norm(data_mapped - commonspace)

        return mappers

