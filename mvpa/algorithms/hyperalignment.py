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
    alignment = Parameter(ProcrusteanMapper(), # might provide allowedtype later on
            doc="""... XXX If `None` (default) an instance of
            :class:`~mvpa.mappers.procrustean.ProcrusteanMapper` is
            used.""")

    levels = Parameter(3, allowedtype='int', min=1,
            doc="Number of levels ....XXX ")

    combiner1 = Parameter('mean', #
            doc="XXX ")

    combiner2 = Parameter('mean', #
            doc="XXX ")


    def __init__(self,
                 alignment=ProcrusteanMapper(),
                 levels=3,
                 combiner1='mean',
                 combiner2='mean',
				 ref_subj=0,
                 **kwargs):

        ClassWithCollections.__init__(self, **kwargs)

    def __call__(self, data):
        """Estimate mappers for each data(set)

        Parameters
        ----------
          data : list or tuple of dataset of data
            XXX

        Returns
        -------
        A list of trained Mappers ... of length equal to len(data)
        """
        params = self.params            # for quicker access ;)
        nelements = len(data)
        nfeatures = [data[i].shape[1] for i in xrange(nelements)]
		
        if min(vox_size) == max(vox_size):
            if ref_subj < 0 or ref_subj >= nelements:
                ref_subj = 0
        else:
            nf = max(nfeatures)
            ref_subj = nfeatures.index(nf) 
		
        # might prefer some other way to initialize... later
        result = [deepcopy(params.alignment) for i in xrange(nelements)]
        # zscore all data sets
        ds = [ zscore(ds[i], perchunk=False) for i in xrange(nelements)]
        # Level 1
        commonspace = data[ref_subj]
        ds_mapped = []
        for m, d, i in zip(mappers[0:], data[0:], xrange(nelements)):
            if i!=ref_subj:
				# XXX For now lets just call this way:
            	m.train(d, commonspace)
            	if ds_mapped == []:
            	    ds_mapped = zscore( m.forward(d), perchunk=False)
            	else:
            	    ds_mapped += zscore( m.forward(d), perchunk=False)
            	commonspace = mean( ds_mapped[i], commonspace)   # zscore before adding
            	# here yarik stopped ;)

        # update commonspace to mean of ds_mapped
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        commonspace = mean(ds_mapped)
        # Level 2 to params.levels to derive common space
        for m, d, i in zip(mappers[0:], data[0:], xrange(nelements)):
            ds_temp = zscore( (commonspace*nelements - ds_mapped[i])/(nelements-1), perchunk=False )
            m.train(d, ds_temp)
            ds_mapped[i] = zscore( m.forward(ds_temp), perchunk=False)

        # update commonspace to mean of ds_mapped
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXX

        commonspace = mean(ds_mapped)
        # Level 3 to params.levels
        for m, d, i in zip(mappers[0:], data[0:], xrange(nelements)):
            ds_temp = zscore( (commonspace*nelements - ds_mapped[i])/(nelements-1), perchunk=False )
            m.train(d, ds_temp )

        # might prefer some other way to initialize... later
        result = [deepcopy(mappers[i]) for i in xrange(nelements)]
        
        return result

