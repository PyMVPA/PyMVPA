
import numpy as N

# Imports enough to convert to shogun kernels if shogun is installed
#try:
    #from mvpa.kernels.sg import SGKernel
    #_has_shogun=True
#except RuntimeError:
    #_has_shogun=False

class Kernel(object):
    """Abstract class which calculates a kernel function between datasets
    
    Each instance has an internal representation self._k which might be of
    a different form depending on the intended use.  Some kernel types should
    be translatable to other representations where possible, e.g., between 
    Numpy and Shogun-based kernels.
    
    This class should not be used directly, but rather use a subclass which
    enforces a consistent internal representation.
    """

    def __init__(self):
        self._k = None

    def compute(self, ds1, ds2=None):
        raise NotImplemented, "Abstract method"

    # NB: __array__ is circular with as_np.  Any subclass must override either
    # one of these and the other will work
    def __array__(self):
        return self.as_np()._k
    
    def as_np(self):
        """Converts this kernel to a Numpy-based representation"""
        return StaticKernel(N.array(self))

    def cleanup(self):
        self._k = None


class NumpyKernel(Kernel):
    """A Kernel object with internal representation as a 2d numpy array"""
    # Conversions
    def __init__(self):
        Kernel.__init__(self)
        
    def __array__(self):
        # By definintion, a NumpyKernel's internal representation is an array
        return self._k
    
    def as_np(self):
        # Already numpy!!
        return self

    # wasn't that easy?

class CustomKernel(NumpyKernel):
    def __init__(self, kernelfunc):
        NumpyKernel.__init__(self)
        self._kf = kernelfunc
        
    def compute(self, d1,d2=None):
        if d2 is None:
            d2=d1
        self._k = self._kf(d1, d2)
        
class LinearKernel(CustomKernel):
    def __init__(self):
        CustomKernel.__init__(self, self._compute)
    @staticmethod
    def _compute(d1, d2):
        if d2 is None:
            d2=d1
        return N.dot(d1.samples, d2.samples.T)
            
class StaticKernel(NumpyKernel):

    def __init__(self, matrix):
        super(StaticKernel, self).__init__()
        self._k = N.array(matrix)

    def compute(self, *args, **kwargs):
        pass


class CachedKernel(NumpyKernel):

    def __init__(self, kernel):
        super(CachedKernel, self).__init__()
        self._ckernel = kernel
        self._ds_cached_info = None
        self._rhids = self._lhids = None

    def _init(self, ds1, ds2=None):
        """Initializes internal lookups + _kfull
        """
        self._lhsids = SampleLookup(ds1)
        if ds2 is None:
            self._rhsids = self._lhsids
        else:
            self._rhsids = SampleLookup(ds2)

        self._ckernel.compute(ds1, ds2)
        self._kfull = self._ckernel.as_np()._k
        self._ckernel.cleanup()
        self._k = self._kfull
        # TODO: store params representation for later comparison

    def compute(self, ds1, ds2=None):
        """Computes full or extracts relevant part of kernel as _k
        """
        #if self._ds_cached_info is not None:
        # Check either those ds1, ds2 are coming from the same
        # dataset as before

        # TODO: figure out if params were modified...
        # params_modified = True
        if params_modified:
            self._init(ds1, ds2)
        else:
            # figure d1, d2
            # TODO: find saner numpy way to select both rows and columns
            try:
                lhsids = self._lhsids(ds1)
                if ds2 is None:
                    rhids = lhsids
                else:
                    rhsids = self._rhsids(ds2)
                self._k = self._kfull.take(
                    lhsids, axis=0).take(
                    rhsids, axis=1)
            except:
                self._init(ds1, ds2)

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
"""

