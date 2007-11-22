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


class FeatureSelection(object):
    """ Base class for any feature selection

    TODO...
    """

    def __init__(self):
        self.__mask = None
        """Binary mask defining the voxels which were selected"""


    def __call__(self, dataset, callables=[]):
        """Invocation of the feature selection

        - `dataset`: actually dataset.
        - `callables`: a list of functors to be called with locals()

        Returns a dataset with selected features.  Derived classes
        must provide interface to access other relevant to the feature
        selection process information (e.g. mask, elimination step
        (in RFE), etc)
        """
        raise NotImplementedError


    def getMask(self):
        """ Returns a mask computed during previous call()
        """
        if self.__mask is None:
            raise UnknownStateError
        return self.__mask

    mask = VProperty(fget=getMask)


class StoppingCriterion(object):
    """Base class for all functors to decide when to stop RFE (or may
    be general optimization... so it probably will be moved out into
    some other module
    """

    def __call__(self, errors):
        """Instruct when to stop

        Returns tuple (`stop`, `isthebest`)
        """
        raise NotImplementedError


class StopNBackHistoryCriterion(StoppingCriterion):
    """ Stop computation if for a number of steps error was increasing
    """

    def __init__(self, steps=10, func=min):
        """Initialize with number of steps
        `steps`: int, for how many steps to check after optimal value
        `fun`: functor, to select the best results. Defaults to min
        """
        StoppingCriterion.__init__(self)
        if steps < 0:
            raise ValueError, \
                  "Number of steps (got %d) should be non-negative" % steps
        self.__steps = steps
        self.__func = func


    def __call__(self, errors):
        isbest = False
        stop = False

        # just to prevent ValueError
        if len(errors)==0:
            return (isbest, stop)

        minerror = self.__func(errors)
        minindex = errors.index(minerror)

        # if minimal is the last one reported -- it is the best
        if minindex == len(errors)-1:
            isbest = True

        # if number of elements after the min >= len -- stop
        if len(errors) - minindex > self.__steps:
            stop = True

        return (stop, isbest)

    steps = property(fget=lambda x:x.__steps)




class FeatureSelector(State):
    """Base class to implement functors to select the set of properties
    """
    def __init__(self):
        State.__init__(self)


class TailFeatureSelector(FeatureSelector):
    """Remove features in the tail of the distribution.
    """

    def __init__(self, removeminimal=True, exactnumber=False):
        """
        Initialize TailFeatureSelector
        `removeminimal`: Bool, False signals to remove maximal elements
        `exactnumber`: Bool, TODO!!: given a number of features to remove,
                      when `exactnumber`==`False` (default now) selector might
                      remove slightly more features if there are multiple
                      features with the same minimal/maximal value.
                      Implementation of `exactnumber`==True needs sorting
                      to return indecies of the sorted array, so either zipping
                      of values with indicies has to be done first or may be
                      there is a better way
        """
        FeatureSelector.__init__(self)  # init State before registering anything

        self._registerState('ndiscarded')    # state variable
        """Store number of discarded since we might remove less than requested
        if not enough features left
        """
        self.__removeminimal = removeminimal
        """Know which tail to remove
        """


    def __repr__(self):
        s = "%s: remove-minimal=%d" % (self.__name__, self.__removeminimal)
        if not self['ndiscarded'] is None:
            s += " {discarded: %d}" % self['ndiscarded']
        return s


    def __call__(self, sensitivity):
        """Call function returns Ids to be kept
        """
        nfeatures = len(sensitivity)
        # how many to discard
        nremove = min(self._getNumberToDiscard(nfeatures), nfeatures)
        sensmap = N.array(sensitivity)  # assure that it is ndarray
        sensmap2 = sensmap.copy()       # make a copy to sort
        sensmap2.sort()                 # sort inplace
        if self.__removeminimal:
            good_ids = sensmap[sensmap>sensmap2[nremove-1]]
        else:
            # remove maximal elements
            good_ids = sensmap[sensmap<sensmap2[-nremove]]
        # compute actual number of discarded elements
        self['ndiscarded'] = nfeatures - len(good_ids)
        return good_ids


class FixedNumberFeatureSelector(TailFeatureSelector):
    """Given a sensitivity map, provide Ids given a number features to remove

    TODO: Should be DiscardSelector on top of it... __repr__, ndiscarded should
          belong to it.

    TODO: Should API would be unified with XPercentFeatureSelector so they both
          simple have .discard property and corresponding constructor parameter
    """

    def __init__(self, number_discard=1, *args, **kwargs):
        """
        """
        TailFeatureSelector.__init__(self, *args, **kwargs)
        self.__number_discard = number_discard      # pylint should smile


    def __repr__(self):
        return "%s number=%f" % (
            TailFeatureSelector.__repr__(self), self.__number_discard)


    def _getNumberToDiscard(self, nfeatures):
        return min(nfeatures, self.__number_discard)


    def _setNumberDiscard(self, number_discard):
        self.__number_discard = number_discard


    number_discard = property(fget=lambda x:x.__number_discard,
                              fset=_setNumberDiscard)



class XPercentFeatureSelector(TailFeatureSelector):
    """Given a sensitivity map, provide Ids given a percentage of features

    Since silly Yarik could not recall alternative to "proportion" to select
    in units like 0.05 for 5%, now this selector does take values in percents
    TODO: Should be DiscardSelector on top of it... __repr__, ndiscarded should
          belong to it.
    """

    def __init__(self, perc_discard=5.0, **kargs):
        """XXX???
        """
        self.__perc_discard = None      # pylint should smile
        self.perc_discard = perc_discard
        TailFeatureSelector.__init__(self, **kargs)


    def __repr__(self):
        return "%s perc=%f" % (
            TailFeatureSelector.__repr__(self), self.__perc_discard)


    def _getNumberToDiscard(self, nfeatures):
        num = int(floor(self.__perc_discard * nfeatures * 1.0 / 100.0))
        num = max(1, num)               # remove at least 1
        return min(num, nfeatures)


    def _setPercDiscard(self, perc_discard):
        """How many percent to discard"""
        if perc_discard>100 or perc_discard<0:
            raise ValueError, \
                  "Percentage (%f) cannot be outside of [0,100]" \
                  % perc_discard
        self.__perc_discard = perc_discard


    perc_discard = property(fget=lambda x:x.__perc_discard,
                            fset=_setPercDiscard)

