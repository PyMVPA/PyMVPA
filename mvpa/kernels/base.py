# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Base Kernel classes

"""

_DEV_DOC_ = """
Concerns:

- Assure proper type of _k assigned
- The same issue "Dataset vs data" in input arguments

"""

__docformat__ = 'restructuredtext'

import numpy as N

from mvpa.base.types import is_datasetlike
from mvpa.misc.state import ClassWithCollections
from mvpa.misc.param import Parameter
from mvpa.misc.sampleslookup import SamplesLookup # required for CachedKernel

# Imports enough to convert to shogun kernels if shogun is installed
#try:
    #from mvpa.kernels.sg import SGKernel
    #_has_shogun=True
#except RuntimeError:
    #_has_shogun=False

class Kernel(ClassWithCollections):
    """Abstract class which calculates a kernel function between datasets

    Each instance has an internal representation self._k which might be of
    a different form depending on the intended use.  Some kernel types should
    be translatable to other representations where possible, e.g., between
    Numpy and Shogun-based kernels.

    This class should not be used directly, but rather use a subclass which
    enforces a consistent internal representation.
    """

    _ATTRIBUTE_COLLECTIONS = ['params'] # enforce presence of params collections

    def __init__(self, *args, **kwargs):
        ClassWithCollections.__init__(self, *args, **kwargs)
        self._k = None
        """Implementation specific version of the kernel"""

    def compute(self, ds1, ds2=None):
        """Generic computation of any kernel

        Assumptions:

         - ds1, ds2 are either datasets or arrays,
         - presumably 2D (not checked neither enforced here
         - _compute takes ndarrays. If your kernel needs datasets,
           override compute
        """
        if is_datasetlike(ds1):
            ds1 = ds1.samples
        if ds2 is None:
            ds2 = ds1
        elif is_datasetlike(ds2):
            ds2 = ds2.samples
        # TODO: assure 2D shape
        self._compute(ds1, ds2)

    def _compute(self, d1, d2):
        raise NotImplemented, "Abstract method"

    def __array__(self):
        return self.as_np()._k

    def as_np(self):
        """Converts this kernel to a Numpy-based representation"""
        p = PrecomputedKernel(matrix=N.array(self))
        p.compute()
        return p

    def cleanup(self):
        """Wipe out internal representation

        XXX unify: we have reset in other places to accomplish similar
        thing
        """
        self._k = None


class NumpyKernel(Kernel):
    """A Kernel object with internal representation as a 2d numpy array"""

    _ATTRIBUTE_COLLECTIONS = Kernel._ATTRIBUTE_COLLECTIONS + ['states']
    # enforce presence of params AND states collections for gradients etc

    def __array__(self):
        # By definintion, a NumpyKernel's internal representation is an array
        return self._k

    def as_np(self):
        # Already numpy!!
        return self

    # wasn't that easy?


class CustomKernel(NumpyKernel):

    kernelfunc = Parameter(None, doc="""Function to generate the matrix""")

    def _compute(self, d1, d2):
        self._k = self.params.kernelfunc(d1, d2)



class PrecomputedKernel(NumpyKernel):
    """Precomputed matrix
    """

    matrix = Parameter(None, allowedtype="ndarray",
                       doc="""ndarray to use as a matrix for the kernel""")

    # NB: to avoid storing matrix twice, after compute 
    # self.params.matrix = self._k
    def __init__(self, *args, **kwargs):
        NumpyKernel.__init__(self, *args, **kwargs)
        self.compute() # Makes sure _k is always available
        
    def compute(self, *args, **kwargs):
        if self._k is None:
            self._k = N.asanyarray(self.params.matrix)
            self.params.matrix = self._k
        #pass


class CachedKernel(NumpyKernel):
    """Kernel decorator to cache all data to avoid duplicate computation
    """

    kernel = Parameter(None, allowedtype="Kernel",
                       doc="""Base kernel to cache""")

    def __init__(self, *args, **kwargs):
        """Initialize CachedKernel

        Parameters
        ----------
          kernel : Kernel
            Base kernel to cache
        """
        super(CachedKernel, self).__init__(*args, **kwargs)
        self.params.update(self.params.kernel.params)
        self._rhids = self._lhids = None

    def _cache(self, ds1, ds2=None):
        """Initializes internal lookups + _kfull
        """
        self._lhsids = SamplesLookup(ds1)
        if ds2 is None:
            self._rhsids = self._lhsids
        else:
            self._rhsids = SamplesLookup(ds2)

        ckernel = self.params.kernel
        ckernel.compute(ds1, ds2)
        self._kfull = ckernel.as_np()._k
        ckernel.cleanup()
        self._k = self._kfull
        
        self._recomputed=True
        # TODO: store params representation for later comparison

    def compute(self, ds1, ds2=None):
        """Computes full or extracts relevant part of kernel as _k
        """
        #if self._ds_cached_info is not None:
        # Check either those ds1, ds2 are coming from the same
        # dataset as before

        # TODO: figure out if data were modified...
        # params_modified = True
        changedData = False
        self._recomputed=False # Flag lets us know whether cache was recomputed
        if len(self.params.whichSet()) or changedData:
            self._cache(ds1, ds2)
            self.params.reset() # hopefully this will never reset values, just
            # changed status
        else:
            # figure d1, d2
            # TODO: find saner numpy way to select both rows and columns
            try:
                lhsids = self._lhsids(ds1)
                if ds2 is None:
                    rhsids = lhsids
                else:
                    rhsids = self._rhsids(ds2)
                self._k = self._kfull.take(
                    lhsids, axis=0).take(
                    rhsids, axis=1)
            except KeyError:
                self._cache(ds1, ds2)

"""
if ds1 is the "derived" dataset as it was computed on:
    * ds2 is None
      ds2 bound to ds1
      -
    * ds1 and ds2 present
      - ds1 and ds2 come from the same dataset
        - whatever CachedKernel was computed on is a superset
        - not a superset -- puke?
      - ds2 comes from different than ds1
        - puke?
else:
    compute (ds1, ds2)
      - different data ids


ckernel = PrecomputedKernel(matrix=N.array([1,2,3]))
ck = CachedKernel(kernel=ckernel)

"""

