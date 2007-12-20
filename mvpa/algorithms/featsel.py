#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Feature selection base class and related stuff base classes and helpers."""

__docformat__ = 'restructuredtext'

import numpy as N

from math import floor

from mvpa.misc.vproperty import VProperty
from mvpa.misc.state import State
from mvpa.misc.exceptions import UnknownStateError

class FeatureSelection(State):
    """Base class for any feature selection

    TODO...
    """

    def __init__(self, **kargs):
        # base init first
        State.__init__(self, **kargs)
        self.__mask = None
        """Binary mask defining the voxels which were selected"""


    def __call__(self, dataset, testdataset=None, callables=[]):
        """Invocation of the feature selection

        dataset: Dataset
            dataset used to select features
        testdataset: Dataset
            dataset the might be used to compute a stopping criterion
        callables: sequence
            a list of functors to be called with locals()

        Returns a tuple with the dataset containing the selected features.
        If present the tuple also contains the selected features of the
        test dataset. Derived classes must provide interface to access other
        relevant to the feature selection process information (e.g. mask,
        elimination step (in RFE), etc)
        """
        raise NotImplementedError


    def getMask(self):
        """Returns a mask computed during previous call()
        """
        if self.__mask is None:
            raise UnknownStateError
        return self.__mask

    mask = VProperty(fget=getMask)



class BestDetector(object):
    """Determine whether the last value in a sequence is the best one given
    some criterion.
    """
    def __init__(self, func=min, lastminimum=False):
        """Initialize with number of steps

        :Parameters:
            fun : functor
                Functor to select the best results. Defaults to min
            lastminimum : bool
                Toggle whether the latest or the earliest minimum is used as
                optimal value to determine the stopping criterion.
        """
        self.__func = func
        self.__lastminimum = lastminimum
        self.__bestindex = None
        """Stores the index of the last detected best value."""


    def __call__(self, errors):
        """Returns True if the last value in `errors` is the best or False
        otherwise.
        """
        isbest = False

        # just to prevent ValueError
        if len(errors)==0:
            return isbest

        minerror = self.__func(errors)

        if self.__lastminimum:
            # make sure it is an array
            errors = N.array(errors)
            # to find out the location of the minimum but starting from the
            # end!
            minindex = N.array((errors == minerror).nonzero()).max()
        else:
            minindex = errors.index(minerror)

        # if minimal is the last one reported -- it is the best
        if minindex == len(errors)-1:
            isbest = True
            self.__bestindex = minindex

        return isbest

    bestindex = property(fget=lambda self:self.__bestindex)



class StoppingCriterion(object):
    """Base class for all functors to decide when to stop RFE (or may
    be general optimization... so it probably will be moved out into
    some other module
    """

    def __call__(self, errors):
        # XXX Michael thinks that this method should also get the history
        # of the number of features, e.g. I'd like to have: If error is equal
        # but nfeatures is smaller -> isthebest == True
        """Instruct when to stop

        Returns tuple (`stop`, `isthebest`)
        """
        raise NotImplementedError



class StopNBackHistoryCriterion(StoppingCriterion):
    """Stop computation if for a number of steps error was increasing
    """

    def __init__(self, steps=10, func=min, lateminimum=False):
        """Initialize with number of steps

        :Parameters:
            steps : int
                How many steps to check after optimal value.
            fun : functor
                Functor to select the best results. Defaults to min
            lateminimum : bool
                Toggle whether the latest or the earliest minimum is used as
                optimal value to determine the stopping criterion.
        """
        StoppingCriterion.__init__(self)
        if steps < 0:
            raise ValueError, \
                  "Number of steps (got %d) should be non-negative" % steps
        self.__steps = steps
        self.__func = func
        self.__lateminimum = lateminimum


    def __call__(self, errors):
        isbest = False
        stop = False

        # just to prevent ValueError
        if len(errors)==0:
            return (isbest, stop)

        minerror = self.__func(errors)

        if self.__lateminimum:
            # make sure it is an array
            errors = N.array(errors)
            # to find out the location of the minimum but starting from the
            # end!
            minindex = N.array((errors == minerror).nonzero()).max()
        else:
            minindex = errors.index(minerror)

        # if minimal is the last one reported -- it is the best
        if minindex == len(errors)-1:
            isbest = True

        # if number of elements after the min >= len -- stop
        if len(errors) - minindex > self.__steps:
            stop = True

        return (stop, isbest)

    steps = property(fget=lambda x:x.__steps)



class ElementSelector(State):
    """Base class to implement functors to select some elements based on a
    sequence of values.
    """
    def __init__(self):
        """Cheap initialization.
        """
        State.__init__(self)


    def __call__(self, seq):
        """Implementations in derived classed have to return a list of selected
        element IDs based on the given sequence.
        """
        raise NotImplementedError



class TailSelector(ElementSelector):
    """Select elements from a tail of a distribution.

    The default behaviour is to discard the lower tail of a given distribution.
    """

    # TODO: 'both' to select from both tails
    def __init__(self, tail='lower', mode='discard', sort=True):
        """Initialize TailSelector

        :Parameters:
            `tail` : ['lower', 'upper']
                Choose the tail to be processed.
                otherwise.
            `mode` : ['discard', 'select']
                decides whether to `select` or to `discard` features.
            `sort` : Bool
                Flag whether selected IDs will be sorted. Disable if not
                necessary to save some CPU cycles.
        """
        ElementSelector.__init__(self)  # init State before registering anything

        self._registerState('ndiscarded', True)    # state variable
        """Store number of discarded elements.
        """

        self._setTail(tail)
        """Know which tail to select."""

        self._setMode(mode)
        """Flag whether to select or to discard elements."""

        self.__sort = sort


    def _setTail(self, tail):
        """Set the tail to be processed."""
        if not tail in ['lower', 'upper']:
            raise ValueError, "Unkown tail argument [%s]. Can only be one " \
                              "of 'lower' or 'upper'." % tail

        self.__tail = tail


    def _setMode(self, mode):
        """Choose `select` or `discard` mode."""

        if not mode in ['discard', 'select']:
            raise ValueError, "Unkown selection mode [%s]. Can only be one " \
                              "of 'select' or 'discard'." % mode

        self.__mode = mode


    def _getNElements(self, seq):
        """In derived classes has to return the number of elements to be
        processed given a sequence values forming the distribution.
        """
        raise NotImplementedError


    def __call__(self, seq):
        """Returns selected IDs.
        """
        # TODO: Think about selecting features which have equal values but
        #       some are selected and some are not
        len_seq = len(seq)
        # how many to select (cannot select more than available)
        nelements = min(self._getNElements(seq), len_seq)

        # make sure that data is ndarray and compute a sequence rank matrix
        # lowest value is first
        seqrank = N.array(seq).argsort()

        if self.__mode == 'discard' and self.__tail == 'upper':
            good_ids = seqrank[:-1*nelements]
            self['ndiscarded'] = nelements
        elif self.__mode == 'discard' and self.__tail == 'lower':
            good_ids = seqrank[nelements:]
            self['ndiscarded'] = nelements
        elif self.__mode == 'select' and self.__tail == 'upper':
            good_ids = seqrank[-1*nelements:]
            self['ndiscarded'] = len_seq - nelements
        else: # select lower tail
            good_ids = seqrank[:nelements]
            self['ndiscarded'] = len_seq - nelements

        # sort ids to keep order
        # XXX should we do here are leave to other place
        if self.__sort:
            good_ids.sort()

        return good_ids



class FixedNElementTailSelector(TailSelector):
    """Given a sequence, provide set of IDs for a fixed number of to be selected
    elements.
    """

    def __init__(self, nelements, *args, **kwargs):
        """Cheap initialization.

        :Parameter:
            `nselect`: Int
                Number of elements to select/discard.
        """
        TailSelector.__init__(self, *args, **kwargs)
        self._setNElements(nelements)


    def __repr__(self):
        return "%s number=%f" % (
            TailSelector.__repr__(self), self.__nselect)


    def _getNElements(self, seq):
        return self.__nelements


    def _setNElements(self, nelements):
        if __debug__:
            if nelements <= 0:
                raise ValueError, "Number of elements less or equal to zero " \
                                  "does not make sense."

        self.__nelements = nelements


    nelements = property(fget=lambda x:x.__nelements,
                         fset=_setNElements)



class FractionTailSelector(TailSelector):
    """Given a sequence, provide Ids for a fraction of elements
    """

    def __init__(self, felements, **kargs):
        """Cheap initialization.

        :Parameter:
            `felements`: Float (0,1.0]
                Fraction of elements to select/discard.
        """
        TailSelector.__init__(self, **kargs)
        self._setFElements(felements)


    def __repr__(self):
        return "%s fraction=%f" % (
            TailSelector.__repr__(self), self.__felements)


    def _getNElements(self, seq):
        num = int(floor(self.__felements * len(seq)))
        num = max(1, num)               # remove at least 1
        # no need for checks as base class will do anyway
        #return min(num, nselect)
        return num


    def _setFElements(self, felements):
        """What fraction to discard"""
        if felements > 1.0 or felements < 0.0:
            raise ValueError, \
                  "Fraction (%f) cannot be outside of [0.0,1.0]" \
                  % felements

        self.__felements = felements


    felements = property(fget=lambda x:x.__felements,
                         fset=_setFElements)

