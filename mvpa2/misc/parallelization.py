# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" General parallelization support

The idea here is to provide a wrapper around pprocess so that 
in the feature we could switch easily to another package

XXX currently only supports calling with arguments that are tuples 
or single values - not named arguments. Ideas on how to improve on
this are welcome

XXX not sure if we should keep fold_ option - this is based on the
original map-fold a.k.a. map-reduce

WiP"""

import numpy as np
from mvpa2.base import debug
import os
from mvpa2.base import externals

from mvpa2.base.dochelpers import borrowdoc

if __debug__ and not "PAR" in debug.registered:
    debug.register("PAR", "Parallalization")

class Parallelizer(object):
    """Abstract parallelization interface"""
    def __init__(self, proc_func, merge_func=None, fold_func=None,
                        nproc=None, results_backend=None, **kwargs):
        """Initializes the parallelizer
        
        Parameters
        ----------
        proc_func: callable
            this function is applied to each input element x
        merge_func: callable or None
            If not none, then fold_func must be None. When called 
            with inputs xs the value of m(p(x) for x in xs),
            where m=merge_func and p=proc_func, is returned
        fold_func: callable or None
            If not none, then merge_func must be None. When called
            with inputs xs the value of 
            f(p(x[0]), f(p(x[1]), ..., f(p(x[-2]),p(x[-1]))...)), 
            where f=fold_func and p=proc_func, is returned
        nproc: int or None
            Number of processes to use. None for maximum number available
        results_backend: 'native' or 'hdf5' or None
            If none then the result of get_best_results_backend() is taken.
            If 'hdf5' then it is assumed that proc_func returns a string
            that denotes a filename with results that can be loaded
            using h5load; upon collecting the results this file
            is removed.
        
        Notes
        -----
        Sub-classes are callable with an iterable that consist of inputs 
        that are either tuples or single values. If the input is a tuple
        x=(x0, x1, ... xN) then this is expanded to *x when called by
        {merge,fold}_func.  
        
        XXX get rid of fold? 
        """

        if sum(map(lambda x:x is None, [merge_func, fold_func])) != 1:
            raise ValueError("Need exactly one of {merge,fold}_func")

        self.proc_func = proc_func
        self.merge_func = merge_func
        self.fold_func = fold_func
        self.nproc = nproc

        if results_backend is None:
            results_backend = self.get_best_results_backend()
            debug('PAR', 'Using auto results backend %s' % results_backend)

        self.results_backend = results_backend


    def __call__(self, xs):
        '''Applies this class to the input
        
        Parameters
        ----------
        xs: iterable:
            input
        
        Returns
        -------
        y: any type
            Either merge_func((f(x) for x in xs),
            or fold_func(x[0], fold_func(x[1], ..., fold_func(x[-2],x[-1])...))
        '''
        raise NotImplementedError()

    @property
    def number_of_cores_available(self):
        '''
        Returns
        -------
        nc: int
            Number of cores available in this parallelizer
        '''
        raise NotImplementedError()

    @staticmethod
    def is_available():
        '''
        Returns
        -------
        is_available: bool
            Whether this parallelizer is available on this platform
        '''
        raise NotImplementedError()

    def _handle_result(self, x):
        debug('PAR', 'handling result %s using %s', (type(x), self.results_backend))
        if self.results_backend == 'hdf5':
            if not isinstance(x, basestring):
                raise ValueError("Expected a filename")
            from mvpa2.base.hdf5 import h5load
            debug('PAR_', "Loading hdf5 results from %s" % x)
            try:
                result = h5load(x)
            finally:
                os.unlink(x)
            return result
        else:
            return x

    def _join_results(self, xs):
        f = self._handle_result

        debug("PAR_", "Joining results using %s (%s)" % (f, type(f)))

        if self.fold_func:
            debug("PAR", "Folding results of type %s" % type(xs))
            fold_func = self.fold_func
            for i, x in enumerate(xs):
                if i == 0:
                    y = f(x)
                else:
                    y = fold_func(y, f(x))
        elif self.merge_func:
            debug("PAR", "Merging results: %s" % type(xs))
            y = self.merge_func((f(x) for x in xs))
        else:
            raise ValueError("this should not happen")

        debug("PAR", "Results joined: %s" % type(y))

        return y

    @property
    def number_of_cores_needed(self):
        '''Number of cores needed to perform the computation
        
        Returns
        -------
        nc: int
            number of cores available, or nproc, whichever is smaller
        '''
        n = self.number_of_cores_available
        if self.nproc and self.nproc < n:
            n = self.nproc
        return n

    @classmethod
    def get_best_results_backend(self):
        '''Hints which result backend to use
        
        Returns
        -------
        best: str
            'native' or 'hdf5'
        '''
        raise NotImplementedError()
"""
    @property
    def results_backend_to_use(self):
        results_backend = self.results_backend
        if results_backend is None:
            if externals.exists('h5py') and self.number_of_cores_available > 1:
                results_backend = 'hdf5'
            else:
                results_backend = 'native'
        if not results_backend in ('hdf5', 'native'):
            raise ValueError("Illegal backend %s" % backend)
        if results_backend == 'hdf5':
            externals.exists('h5py', raise_=True)
        debug('PAR_', "Backend to use: %s (asked for %s)" %
                            (results_backend, self.results_backend))
        return results_backend
"""


class SingleThreadParallelizer(Parallelizer):
    '''Simple single thread parallelizer. Used as fallback option'''
    @borrowdoc(Parallelizer)
    def __init__(self, proc_func, merge_func=None,
                 fold_func=None, nproc=None, results_backend=None, **kwargs):
        super(SingleThreadParallelizer, self).__init__(proc_func,
                                        merge_func=merge_func,
                                        fold_func=fold_func,
                                        nproc=nproc,
                                        results_backend=results_backend,
                                        **kwargs)
    @borrowdoc(Parallelizer)
    def __call__(self, xs):
        f = self.proc_func
        return self._join_results(f(*x) if type(x) is tuple else f(x) for x in xs)

    @property
    @borrowdoc(Parallelizer)
    def number_of_cores_available(self):
        return 1

    @staticmethod
    @borrowdoc(Parallelizer)
    def is_available():
        return True

    @classmethod
    @borrowdoc(Parallelizer)
    def get_best_results_backend(self):
        return 'native'


class PProcessParallelizer(Parallelizer):
    '''Uses pprocess.'''
    @borrowdoc(Parallelizer)
    def __init__(self, proc_func, merge_func=None, fold_func=None, nproc=None,
                        results_backend=None, **kwargs):
        super(PProcessParallelizer, self).__init__(proc_func,
                                       merge_func=merge_func,
                                       fold_func=fold_func,
                                       nproc=nproc,
                                       results_backend=results_backend,
                                       **kwargs)
        import pprocess
        self.pp = pprocess

    @property
    @borrowdoc(Parallelizer)
    def number_of_cores_available(self):
        return self.pp.get_number_of_cores()

    @borrowdoc(Parallelizer)
    def __call__(self, xs):
        nproc = self.nproc
        if __debug__:
            import time
        debug_ = lambda x:debug('PAR', x)
        if nproc is None:
            nproc = self.number_of_cores_needed

        debug_("pprocess starting with %d processes" % nproc)
        ys = self.pp.Map(limit=nproc)

        debug_("Using %d parallel threads" % nproc)
        compute_func = ys.manage(self.pp.MakeParallel(self.proc_func))

        try:
            n = len(xs)
        except TypeError: # when xs is a generator
            n = "<unknown>"

        for i, x in enumerate(xs):
            debug_("Starting child process %d/%s" % (i + 1, n))
            if type(x) is tuple:
                compute_func(*x)
            else:
                compute_func(x)

        debug_("All %s child processes were started" % n)
        return self._join_results(ys)


    @staticmethod
    @borrowdoc(Parallelizer)
    def is_available():
        try:
            import pprocess
            return True
        except:
            return False

    @classmethod
    @borrowdoc(Parallelizer)
    def get_best_results_backend(self):
        return 'hdf5' if externals.exists('h5py') else 'native'


def get_best_parallelizer(nproc=None):
    '''Returns an instance with the 'best' parallelizer, that is pprocess
    if available and nproc>1, else single thread.
    '''

    if PProcessParallelizer.is_available() and (not nproc or nproc > 1):
        return PProcessParallelizer

    return SingleThreadParallelizer
