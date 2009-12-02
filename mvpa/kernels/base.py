
import numpy as N
import shogun

class Kernel(object):

    def __init__(self):
        self._k = None

    def compute(self, ds1, ds2=None):
        raise NotImplemented, "Abstract method"

    def __array__(self):
        raise NotImplemented, "Abstract method"

    if externals.exists('sg'):
        def as_sg(self):
            return SGKernel(sg.CustomKernel, N.array(self))

    def as_np(self):
        return StaticKernel(N.array(self))

    def cleanup(self):
        self._k = None


class NumpyKernel(Kernel):

    # Conversions
    def __array__(self):
        return self._k

    def as_np(self):
        return self

    def _subkernel(self, i1, i2=None):
        return


class StaticKernel(NumpyKernel):

    def __init__(self, a=None):
        super(StaticKernel, self).__init__()
        self._k = a

    def compute(self, *args, **kwargs):
        pass


class SGKernel(Kernel):

    def __init__(self, sg_cls, *args, **kwargs):
        """
        Parameters
        ----------
          cls : basestring or shogun kernel class
            The shogun kernel class to instantiate upon compute
        """
        super(SGKernel, self).__init__()
        # TODO: store args/kwargs to initiate sg_cls
        pass # XXX

    def compute(self, ds1, ds2=None):
        # XXX
        raise NotImplemented, "TODO"

    def as_sg(self):
        return self

    def __array__(self):
        return self._k.get_full_matrix()


#class LIBSVMKernel(Kernel):
#
#    def as_libsvm(self)


class CachedKernel(NumpyKernel):

    def __init__(self, kernel):
        super(CachedKernel, self).__init__()
        self._ckernel = kernel
        self._ds_cached_info = None
        self._rhids = self._lhids = None

    def _init(self, ds1, ds2=None):
        """Initializes internal lookups + full _k
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

