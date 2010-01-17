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

from mvpa.base import warning
from mvpa.misc.state import StateVariable, ClassWithCollections
from mvpa.misc.param import Parameter
from mvpa.misc.transformers import GrandMean
from mvpa.mappers.procrustean import ProcrusteanMapper

if __debug__:
    from mvpa.base import debug


class Hyperalignment(ClassWithCollections):
    """ ...

    Given a set of datasets (may be just data) provide mapping of
    features into a common space
    """

    # May be something we might store optionally upon user request
    who_knows_maybe_something_to_store_optionally = \
       StateVariable(enabled=False, doc= """....""")

    # Lets use built-in facilities to specify parameters which
    # constructor should accept
    alignment = Parameter(ProcrusteanMapper(), # might provide allowedtype
            doc="""... XXX If `None` (default) an instance of
            :class:`~mvpa.mappers.procrustean.ProcrusteanMapper` is
            used.""")

    levels = Parameter(3, allowedtype='int', min=2,
            doc="Number of levels ....XXX ")

    ref_ds = Parameter(None, allowedtype='int', min=0,
            doc="""Index of a dataset to use as a reference.  If `None`, then
            dataset with maximal number of features is used.""")

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
        nelements = len(datasets)
        nfeatures = [datasets[i].shape[1] for i in xrange(nelements)]

        if params.ref_ds is None:
            ref_ds = N.argmax(nfeatures)
        else:
            ref_ds = params.ref_ds
            if ref_ds < 0 and ref_ds >= nelements:
                raise ValueError, "Requested reference dataset %i is out of " \
                      "bounds. We have only %i datasets provided" \
                      % (ref_ds, nelements)

        # might prefer some other way to initialize... later
        mappers = [deepcopy(params.alignment) for ds in datasets]
        # zscore all data sets
        # ds = [ zscore(ds, perchunk=False) for ds in datasets]

        # Level 1 (first)
        commonspace = N.asanyarray(datasets[ref_ds])
        data_mapped = [N.asanyarray(ds) for ds in datasets]
        for i, (m, data) in enumerate(zip(mappers, data_mapped)):
            if i == ref_ds:
                continue

            # XXX For now lets just call this way:
            m.train(data, commonspace)
            data_mapped[i] = m.forward(data)

            ## if ds_mapped == []:
            ##     ds_mapped = [zscore(m.forward(d), perchunk=False)]
            ## else:
            ##     ds_mapped += [zscore(m.forward(d), perchunk=False)]

            # zscore before adding
            # TODO: make just a function so we dont' waste space
            commonspace = params.combiner1(data_mapped[i], commonspace)

        # update commonspace to mean of ds_mapped
        commonspace = params.combiner2(data_mapped)

        # Level 2 -- might iterate multiple times
        for loop in xrange(params.levels - 2):
            for i, (m, ds) in enumerate(zip(mappers, datasets)):
                ## ds_temp = zscore( (commonspace*nelements - ds_mapped[i])/(nelements-1), perchunk=False )
                m.train(ds, commonspace) # ds_temp)
                data_mapped[i] = m.forward(N.asanyarray(ds))
                #ds_mapped[i] = zscore( m.forward(ds_temp), perchunk=False)

            commonspace = params.combiner2(data_mapped)

        # Level 3 (last) to params.levels
        for i, (m, ds) in enumerate(zip(mappers, datasets)):
            ## ds_temp = zscore( (commonspace*nelements - ds_mapped[i])/(nelements-1), perchunk=False )
            m.train(ds, commonspace) #ds_temp)

        return mappers

