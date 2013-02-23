# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
""" General paralleziation support """

import numpy as np
from mvpa2.base import debug

if __debug__ and not "PAR" in debug.registered:
    debug.register("PAR", "Parallalization")

class Parallelizer(object):

    def __init__(self, proc_func, merge_func=None, fold_func=None, nproc=None):
        if sum(map(lambda x:x is None, [merge_func, fold_func])) != 1:
            raise ValueError("Need exactly one of {merge,fold}_func")
        self.proc_func = proc_func
        self.merge_func = merge_func
        self.fold_func = fold_func
        self.nproc = nproc


    def __call__(self, xs):
        raise NotImplementedError()

    @property
    def number_of_cores_available(self):
        raise NotImplementedError()

    @staticmethod
    def is_available():
        raise NotImplementedError()

    def join_results(self, xs):
        if self.fold_func:
            fold_func = self.fold_func
            for i, x in enumerate(xs):
                if i == 0:
                    y = x
                else:
                    y = fold_func(y, x)
        elif self.merge_func:
            debug("PAR", "Merging results")
            y = self.merge_func(xs)

        return y

    @property
    def number_of_cores_needed(self):
        n = self.number_of_cores_available
        if self.nproc and self.nproc < n:
            n = self.nproc
        return self.nproc



class SingleThreadParallelizer(Parallelizer):
    def __init__(self, proc_func, merge_func=None, fold_func=None, nproc=None):
        super(SingleThreadParallelizer, self).__init__(proc_func, merge_func=merge_func, fold_func=fold_func, nproc=nproc)

    def __call__(self, xs):
        f = self.proc_func
        return self.join_results(f(*x) if type(x) is tuple else f(x) for x in xs)

    @property
    def number_of_cores_available(self):
        return 1

    @staticmethod
    def is_available():
        return True


class PProcessParallelizer(Parallelizer):
    def __init__(self, proc_func, merge_func=None, fold_func=None, nproc=None):
        super(PProcessParallelizer, self).__init__(proc_func, merge_func=merge_func, fold_func=fold_func, nproc=nproc)
        import pprocess
        self.pp = pprocess

    @property
    def number_of_cores_available(self):
        return self.pp.get_number_of_cores()

    def __call__(self, xs):
        nproc = self.number_of_cores_needed
        if __debug__:
            import time
        t0 = time.time()
        debug_ = lambda x:debug('PAR', '%s: %s' % (str(time.time() - t0), x))

        debug_("pprocess starting with %d processes" % nproc)
        ys = self.pp.Map(limit=nproc)

        debug_("Using %d parallel threads" % nproc)
        compute_func = ys.manage(self.pp.MakeParallel(self.proc_func))

        n = len(xs)
        for i, x in enumerate(xs):
            debug_("Starting child process %d/%d" % (i + 1, n))
            if type(x) is tuple:
                compute_func(*x)
            else:
                compute_func(x)

        debug_("All %d child processes were started" % n)
        return self.join_results(ys)


    @staticmethod
    def is_available():
        try:
            import pprocess
            return True
        except:
            return False

def get_parallelizer(proc_func, merge_func=None, fold_func=None, nproc=None):
    classes = [PProcessParallelizer, SingleThreadParallelizer]

    for cls in classes:
        is_available = cls.is_available()
        debug("PAR", "%r is available: %s", (cls, is_available))
        if is_available:
            return cls(proc_func=proc_func, merge_func=merge_func, fold_func=fold_func, nproc=nproc)

    raise ValueError("No parallelization available")
