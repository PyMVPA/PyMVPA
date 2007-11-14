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

        Returns tuple (`go`, `isthebest`)
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


class FeatureSelector(object):
    """Base class to implement functors to select the set of properties
    """
    pass


class XPercentFeatureSelector(FeatureSelector):
    """Given a sensitivity map, provide Ids given a percentage of features

    Since silly Yarik could not recall alternative to "proportion" to select
    in units like 0.05 for 5%, now this selector does take values in percents
    TODO: Should be DiscardSelector on top of it... __repr__, ndiscarded should
          belong to it.
    """

    def __init__(self, perc_discard=5.0, removeminimal=True):
        """XXX???
        """
        self.__perc_discard = None      # pylint should smile
        self.perc_discard = perc_discard
        self.__removeminimal = removeminimal
        self.__ndiscarded = None
        """Store number of discarded since for a given value we might
        remove more than 1 if they are all equal -- it would be unfair
        to leave some features in while another with the same value
        got discarded
        """


    def __repr__(self):
        s = "%s: perc=%f minimal=%s" % (
            self.__name__, self.__perc_discard, self.__removeminimal)
        if not self.__ndiscarded is None:
            s += " discarded: %d" % self.ndiscarded


    def __call__(self, sensitivity):
        nfeatures = len(sensitivity)
        # how many to discard
        nremove = int(floor(self.__perc_discard * nfeatures * 1.0 / 100.0))
        sensmap = N.array(sensitivity)  # assure that it is ndarray
        sensmap2 = sensmap.copy()       # make a copy to sort
        sensmap2.sort()                 # sort inplace
        if self.__removeminimal:
            good_ids = sensmap[sensmap>sensmap2[nremove-1]]
        else:
            # remove maximal elements
            good_ids = sensmap[sensmap<sensmap2[-nremove]]
        # compute actual number of discarded elements
        self.__ndiscarded = nfeatures - len(good_ids)
        return good_ids


    def _getNDiscarded(self):
        """Return number of discarded elements

        Raises an UnknownStateError exception if the instance wasn't
        called yet
        """
        if self.__ndiscarded == None:
            raise UnknownStateError
        return self.__ndiscarded


    def _setPercDiscard(self, perc_discard):
        """How many percent to discard"""
        if perc_discard>100 or perc_discard<0:
            raise ValueError, \
                  "Percentage (%f) cannot be outside of [0,100]" \
                  % perc_discard
        self.__perc_discard = perc_discard


    ndiscarded = property(fget=_getNDiscarded)
    perc_discard = property(fget=lambda x:x.__perc_discard,
                            fset=_setPercDiscard)

