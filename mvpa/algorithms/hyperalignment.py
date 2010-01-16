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
    alignment = Parameter(None, # might provide allowedtype later on
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
                 alignment=None,
                 levels=3,
                 combiner1='mean',
                 combiner2='mean',
                 **kwargs):

        ClassWithCollections.__init__(self, **kwargs)

        if self.params.alignment == None:
            self.params.alignment = ProcrusteanMapper()

        raise NotImlementedError, "WiP! Come back later"

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

        # might prefer some other way to initialize... later
        result = [deepcopy(params.alignment) for i in xrange(nelements)]

        # Level 1
        commonspace = data[0]
        for m, d in zip(mappers[1:], data[1:]):
            # XXX For now lets just call this way:
            m.train(d, commonspace)
            commonspace = mean(m.forward(d), commonspace)# here yarik stopped ;)


        # Level 2 to params.levels

        
        return result

