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

import numpy as np

from mvpa2.base.types import is_datasetlike
from mvpa2.base.state import ClassWithCollections
from mvpa2.base.param import Parameter
from mvpa2.misc.sampleslookup import SamplesLookup # required for CachedKernel

if __debug__:
    from mvpa2.base import debug

__all__ = ['Kernel', 'NumpyKernel', 'CustomKernel', 'PrecomputedKernel',
           'CachedKernel']

class Kernel(ClassWithCollections):
    """Abstract class which calculates a kernel function between datasets

    Each instance has an internal representation self._k which might be of
    a different form depending on the intended use.  Some kernel types should
    be translatable to other representations where possible, e.g., between
    Numpy and Shogun-based kernels.

    This class should not be used directly, but rather use a subclass which
    enforces a consistent internal representation, such as a NumpyKernel.

    Notes
    -----
    Conversion mechanisms: Each kernel type should implement methods
    as necessary for the following two methods to work:

    :meth:`~mvpa2.kernels.Kernel.as_np`
      *Return a new NumpyKernel object with internal Numpy kernel*.
      This method can be generally inherited from the base Kernel class by
      creating a PrecomputedKernel from the raw numpy matrix, as implemented
      here.

    :meth:`~mvpa2.kernels.Kernel.as_raw_np`
      *Return a raw Numpy array from this kernel*.
      This method should behave identically to numpy.array(kernel), and in fact,
      defining either method (via defining Kernel.__array__) will be sufficient
      for both method calls to work.  See this source code for more details.

    Other kernel types should implement similar mechanisms to convert numpy
    arrays to their own internal representations.  See `add_conversion` for a
    helper method, and examples in mvpa2.kernels.sg

    Assuming such `Kernel.as_*` methods exist, all kernel types should be
    seamlessly convertable amongst each other.

    Note that kernels are not meant to be 'functionally translateable' in the
    sense that one kernel can be created, translated, then used to compute
    results in a new framework.  Rather, the results are meant to be
    exchangeable, hence the standard practice of using a precomputed kernel
    object to store the results in the new kernel type.

    For example:

    ::

      k = SomeShogunKernel()
      k.compute(data1, data2)

      # Incorrect and unsupported use
      k2 = k.as_cuda()
      k2.compute(data3, data4) # Would require 'functional translation' to the new
                               # backend, which is impossible

      # Correct use
      someOtherAlgorithm(k.as_raw_cuda()) # Simply uses kernel results in CUDA
    """

    _ATTRIBUTE_COLLECTIONS = ['params'] # enforce presence of params collections

    # Define this per class: standard string describing kernel type, ie
    # 'linear', or 'rbf', to help coordinate kernel types across backends
    __kernel_name__ = None

    def __init__(self, *args, **kwargs):
        """Base Kernel class has no parameters
        """
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
        raise NotImplementedError("Abstract method")

    def computed(self, *args, **kwargs):
        """Compute kernel and return self
        """
        self.compute(*args, **kwargs)
        return self

    ############################################################################
    # The following methods are circularly defined.  Child kernel types can
    # override either one or both to allow conversion to Numpy
    def __array__(self):
        return self.as_raw_np()

    def as_raw_np(self):
        """Directly return this kernel as a numpy array"""
        return np.array(self)

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

        Parameters
        ----------
        typename : string
          Describes kernel type
        methodfull : function
          Method which converts to the new kernel object class
        methodraw : function
          Method which returns a raw kernel

        Examples
        --------
        Kernel.add_conversion('np', fullmethod, rawmethod)
        binds kernel.as_np() to fullmethod()
        binds kernel.as_raw_np() to rawmethod()

        Can also be used on subclasses to override the default conversions
        """
        setattr(cls, 'as_%s'%typename, methodfull)
        setattr(cls, 'as_raw_%s'%typename, methodraw)

class NumpyKernel(Kernel):
    """A Kernel object with internal representation as a 2d numpy array"""

    _ATTRIBUTE_COLLECTIONS = Kernel._ATTRIBUTE_COLLECTIONS + ['ca']
    # enforce presence of params AND ca collections for gradients etc

    def __array__(self):
        # By definintion, a NumpyKernel's internal representation is an array
        return self._k

    def as_np(self):
        """Converts this kernel to a Numpy-based representation"""
        # Already numpy!!
        return self

    def as_raw_np(self):
        """Directly return this kernel as a numpy array.

        For Numpy-based kernels - simply returns stored matrix."""

        return self._k
    # wasn't that easy?


class CustomKernel(NumpyKernel):
    """Custom Kernel defined by an arbitrary function

    Examples
    --------

    Basic linear kernel
    >>> k = CustomKernel(kernelfunc=lambda a,b: numpy.dot(a,b.T))
    """

    __TODO__ = """
    - repr/doc sicne now kernelfunc is not a Parameter
    """

    def __init__(self, kernelfunc=None, *args, **kwargs):
        """Initialize CustomKernel with an arbitrary function.

        Parameters
        ----------
        kernelfunc : function
          Any callable function which takes two numpy arrays and
          calculates a kernel function, treating the rows as samples and the
          columns as features. It is called from compute(d1, d2) -> func(d1,d2)
          and should return a numpy matrix K(i,j) which holds the kernel
          evaluated from d1 sample i and d2 sample j
        """
        NumpyKernel.__init__(self, *args, **kwargs)
        self._kernelfunc = kernelfunc

    def _compute(self, d1, d2):
        self._k = self._kernelfunc(d1, d2)



class PrecomputedKernel(NumpyKernel):
    """Precomputed matrix
    """

    __TODO__ = """
    - repr/doc sicne now matrix is not a Parameter
    """

    # NB: to avoid storing matrix twice, after compute
    # self.params.matrix = self._k
    def __init__(self, matrix=None, *args, **kwargs):
        """
        Parameters
        ----------
        matrix : Numpy array or convertable kernel, or other object type
        """
        NumpyKernel.__init__(self, *args, **kwargs)

        self._k = np.array(matrix)

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
    normally (computing a new cache invalidates any previous cached data).

    The cache is asymmetric for lhs and rhs, so compute(d1, d2) does not create
    a cache usable for compute(d2, d1).
    """

    # TODO: Figure out how to design objects like CrossValidation etc to
    # precompute this kernel automatically, making it transparent to the user

    @property
    def __kernel_name__(self):
        """Allows checking name of subkernel"""
        return self._kernel.__kernel_name__

    def __init__(self, kernel=None, *args, **kwargs):
        """Initialize `CachedKernel`

        Parameters
        ----------
        kernel : Kernel
          Base kernel to cache.  Any kernel which can be converted to a
          `NumpyKernel` is allowed
        """
        super(CachedKernel, self).__init__(*args, **kwargs)
        self._kernel = kernel
        self.params.update(self._kernel.params)
        self._rhsids = self._lhsids = self._kfull = None
        self._recomputed = None

    def _cache(self, ds1, ds2=None):
        """Initializes internal lookups + _kfull via caching the kernel matrix
        """
        if __debug__ and 'KRN' in debug.active:
            debug('KRN', "Caching %(inst)s for ds1=%(ds1)s, ds2=%(ds1)s"
                  % dict(inst=self, ds1=ds1, ds2=ds2))

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

    def compute(self, ds1, ds2=None, force=False):
        """Automatically computes and caches the kernel or extracts the
        relevant part of a precached kernel into self._k

        Parameters
        ----------
        force : bool
          If True it forces re-caching of the kernel.  It is advised
          to be used whenever explicitly pre-caching the kernel and
          it is known that data was changed.
        """
        if __debug__ and 'KRN' in debug.active:
            debug('KRN', "Computing kernel %(inst)s on ds1=%(ds1)s, ds2=%(ds1)s"
                  % dict(inst=self, ds1=ds1, ds2=ds2))

        # Flag lets us know whether cache was recomputed
        self._recomputed = False

        #if self._ds_cached_info is not None:
        # Check either those ds1, ds2 are coming from the same
        # dataset as before

        # TODO: figure out if data were modified...
        # params_modified = True
        changedData = False or force
        if len(self.params.which_set()) or changedData \
           or self._lhsids is None:
            self._cache(ds1, ds2)# hopefully this will never reset values, just
            # changed status
        else:
            # figure d1, d2
            try:
                lhsids = self._lhsids(ds1) #
                if ds2 is None:
                    rhsids = lhsids
                else:
                    rhsids = self._rhsids(ds2)
                self._k = self._kfull[np.ix_(lhsids, rhsids)]
            except KeyError:
                self._cache(ds1, ds2)

        if __debug__ and self._recomputed:
            debug('KRN',
                  "Kernel %(inst)s was recomputed on ds1=%(ds1)s, ds2=%(ds1)s"
                  % dict(inst=self, ds1=ds1, ds2=ds2))


__BOGUS_NOTES__ = """
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


ckernel = PrecomputedKernel(matrix=np.array([1,2,3]))
ck = CachedKernel(kernel=ckernel)

"""

