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
    
    Conversion mechanisms:
    Each kernel type should implement methods as necessary for the following two
    methods to work:

    kernel.as_np() # Return a new NumpyKernel object with internal Numpy kernel
    kernel.as_raw_np() # Return a raw Numpy array from this kernel
    
    Other kernel types should implement similar mechanisms to convert numpy
    arrays to their own internal representations.  Assuming such methods exist,
    all kernel types should be seamlessly convertable amongst each other.
    """

    _ATTRIBUTE_COLLECTIONS = ['params'] # enforce presence of params collections

    # Define this per class: standard string describing kernel type, ie 
    # 'linear', or 'rbf', to help coordinate kernel types across backends
    __kernel_name__ = None
    
    # Class holder which can create feature sensitivities
    __svm_sensitivity__ = None
    
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
        """Specific implementation to be overridden
        """
        raise NotImplemented, "Abstract method"

    def computed(self, *args, **kwargs):
        """Compute kernel and return self
        """
        self.compute(*args, **kwargs)
        return self

    ############################################################################
    # The following methods are circularly defined.  Child kernel types can
    # override either one of them to allow conversion to Numpy
    def __array__(self):
        return self.as_raw_np()

    def as_raw_np(self):
        """Directly return this kernel as a numpy array"""
        return N.array(self)
    
    ############################################################################
    
    def as_np(self):
        """Converts this kernel to a Numpy-based representation"""
        p = PrecomputedKernel(matrix=self.as_raw_np())
        p.compute()
        return p

    def cleanup(self):
        """Wipe out internal representation

        XXX unify: we have reset in other places to accomplish similar
        thing
        """
        self._k = None

    @classmethod
    def add_conversion(cls, typename, methodfull, methodraw):
        """Adds methods to the Kernel class for new conversions
        
        :Parameters:
        typename: string naming kernel type
        methodfull: method which converts to the new kernel object class
        methodraw: method which returns a raw kernel
        
        :Examples:
        Kernel.add_conversion('np', fullmethod, rawmethod)
        binds kernel.as_np() to fullmethod()
        binds kernel.as_raw_np() to rawmethod()
        
        Can also be used on subclasses to override the default conversions
        """
        setattr(cls, 'as_%s'%typename, methodfull)
        setattr(cls, 'as_raw_%s'%typename, methodraw)

class NumpyKernel(Kernel):
    """A Kernel object with internal representation as a 2d numpy array"""

    _ATTRIBUTE_COLLECTIONS = Kernel._ATTRIBUTE_COLLECTIONS + ['states']
    # enforce presence of params AND states collections for gradients etc

    def __array__(self):
        # By definintion, a NumpyKernel's internal representation is an array
        return self._k

    def as_np(self):
        """Converts this kernel to a Numpy-based representation"""
        # Already numpy!!
        return self

    def as_raw_np(self):
        return self._k
    # wasn't that easy?


class CustomKernel(NumpyKernel):
    """Custom Kernel defined by an arbitrary function
    """

    kernelfunc = Parameter(None, doc="""Function to generate the matrix""")

    def _compute(self, d1, d2):
        self._k = self.params.kernelfunc(d1, d2)



class PrecomputedKernel(NumpyKernel):
    """Precomputed matrix
    """

    matrix = Parameter(None, allowedtype="ndarray",
                       doc="ndarray to use as a matrix for the kernel",
                       ro=True)

    # NB: to avoid storing matrix twice, after compute
    # self.params.matrix = self._k
    def __init__(self, *args, **kwargs):
        NumpyKernel.__init__(self, *args, **kwargs)

        m = self.params.matrix
        if m is None:
            # Otherwise SG would segfault ;-)
            raise TypeError, "You must specify matrix parameter"

        # Make sure _k is always available
        self.__set_matrix(m)

    def __set_matrix(self, m):
        """Set matrix -- may be some time we would allow params.matrix to be R/W
        """
        self._k = N.asanyarray(m)
        self.params['matrix']._set(self._k, init=True)

    def compute(self, *args, **kwargs):
        pass


class CachedKernel(NumpyKernel):
    """Kernel which caches all data to avoid duplicate computation
    
    This kernel is very useful for any analysis which will retrain or
    repredict the same data multiple times, as this kernel will avoid
    recalculating the kernel function.  Examples of such analyses include cross
    validation, bootstrapping, and model selection (assuming the kernel function
    itself does not change, e.g. when selecting for C in an SVM).
    
    The kernel will automatically cache any new data sent through compute, and
    will be able to use this cache whenever a subset of this data is sent
    through compute again.  If new (uncached) data is sent through compute, then
    the cache is recreated from scratch.  Therefore, you should compute the
    kernel on the entire superset of your data before using this kernel
    normally.
    
    The cache is asymmetric for lhs and rhs:
    """

    # TODO: Figure out how to design objects like CrossValidation etc to
    # precompute this kernel automatically, making it transparent to the user
    
    kernel = Parameter(None, allowedtype=Kernel,
                       doc="Base kernel to cache.  Any kernel which can be " +\
                       "converted to a NumpyKernel is allowed")

    @property
    def __kernel_name__(self):
        """Allows checking name of subkernel"""
        return self.params.kernel.__kernel_name__
    
    def __init__(self, kernel=None, **kwargs):
        """
        :Parameters:
        kernel: Base kernel to cache.  Any kernel which can be converted to a 
          NumpyKernel is allowed
        """
        super(CachedKernel, self).__init__(**kwargs)
        self._kernel = kernel
        self.params.update(self._kernel.params)
        self._rhsids = self._lhsids = self._kfull = None

    def _cache(self, ds1, ds2=None):
        """Initializes internal lookups + _kfull via caching the kernel matrix
        """
        self._lhsids = SamplesLookup(ds1)
        if (ds2 is None) or (ds2 is ds1):
            self._rhsids = self._lhsids
        else:
            self._rhsids = SamplesLookup(ds2)

        ckernel = self._kernel
        ckernel.compute(ds1, ds2)
        self._kfull = ckernel.as_raw_np()
        ckernel.cleanup()
        self._k = self._kfull
        
        self._recomputed = True
        self.params.reset()
        # TODO: store params representation for later comparison

    def compute(self, ds1, ds2=None):
        """Automatically computes and caches the kernel or extracts the
        relevant part of a precached kernel into self._k
        """
        # Flag lets us know whether cache was recomputed
        self._recomputed = False
        
        #if self._ds_cached_info is not None:
        # Check either those ds1, ds2 are coming from the same
        # dataset as before

        # TODO: figure out if data were modified...
        # params_modified = True
        changedData = False
        if len(self.params.whichSet()) or changedData \
           or self._lhsids is None:
            self._cache(ds1, ds2)# hopefully this will never reset values, just
            # changed status
        else:
            # figure d1, d2
            # TODO: find saner numpy way to select both rows and columns
            try:
                lhsids = self._lhsids(ds1) #
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

