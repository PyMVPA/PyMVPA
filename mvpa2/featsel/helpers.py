# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Helpers for feature selection (scoring, selection strategies)"""

__docformat__ = 'restructuredtext'

from math import floor
import numpy as np

from mvpa2.base.dataset import AttrDataset
from mvpa2.base.state import ClassWithCollections, ConditionalAttribute

if __debug__:
    from mvpa2.base import debug

#
# Functors to be used for FeatureSelection
#

class BestDetector(object):
    """Determine whether the last value in a sequence is the best one given
    some criterion.
    """
    def __init__(self, func=min, lastminimum=False):
        """Initialize with number of steps

        Parameters
        ----------
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
            errors = np.array(errors)
            # to find out the location of the minimum but starting from the
            # end!
            minindex = np.array((errors == minerror).nonzero()).max()
        else:
            minindex = errors.index(minerror)

        self.__bestindex = minindex

        # if minimal is the last one reported -- it is the best
        if minindex == len(errors)-1:
            isbest = True

        return isbest

    bestindex = property(fget=lambda self:self.__bestindex)



class StoppingCriterion(object):
    """Base class for all functors to decide when to stop RFE (or may
    be general optimization... so it probably will be moved out into
    some other module
    """

    def __call__(self, errors):
        """Instruct when to stop.

        Every implementation should return `False` when an empty list is
        passed as argument.

        Returns tuple `stop`.
        """
        raise NotImplementedError



class MultiStopCrit(StoppingCriterion):
    """Stop computation if the latest error drops below a certain threshold.
    """
    def __init__(self, crits, mode='or'):
        """
        Parameters
        ----------
        crits : list of StoppingCriterion instances
          For each call to MultiStopCrit all of these criterions will
          be evaluated.
        mode : {'and', 'or'}
          Logical function to determine the multi criterion from the set
          of base criteria.
        """
        if not mode in ('and', 'or'):
            raise ValueError, \
                  "A mode %r is not supported." % (mode, )

        self.__mode = mode
        self.__crits = crits


    def __call__(self, errors):
        """Evaluate all criteria to determine the value of the multi criterion.
        """
        # evaluate all crits
        crits = [ c(errors) for c in self.__crits ]

        if self.__mode == 'and':
            return np.all(crits)
        else:
            return np.any(crits)



class FixedErrorThresholdStopCrit(StoppingCriterion):
    """Stop computation if the latest error drops below a certain threshold.
    """
    def __init__(self, threshold):
        """Initialize with threshold.

        Parameters
        ----------
          threshold : float [0,1]
              Error threshold.
        """
        StoppingCriterion.__init__(self)
        if threshold > 1.0 or threshold < 0.0:
            raise ValueError, \
                  "Threshold %f is out of a reasonable range [0,1]." \
                    % threshold
        self.__threshold = threshold


    def __call__(self, errors):
        """Nothing special."""
        if len(errors)==0:
            return False
        if errors[-1] < self.__threshold:
            return True
        else:
            return False


    threshold = property(fget=lambda x:x.__threshold)



class NStepsStopCrit(StoppingCriterion):
    """Stop computation after a certain number of steps.
    """
    def __init__(self, steps):
        """Initialize with number of steps.

        Parameters
        ----------
        steps : int
          Number of steps after which to stop.
        """
        StoppingCriterion.__init__(self)
        if steps < 0:
            raise ValueError, \
                  "Number of steps %i is out of a reasonable range." \
                    % steps
        self.__steps = steps


    def __call__(self, errors):
        """Nothing special."""
        if len(errors) >= self.__steps:
            return True
        else:
            return False


    steps = property(fget=lambda x:x.__steps)



class NBackHistoryStopCrit(StoppingCriterion):
    """Stop computation if for a number of steps error was increasing
    """

    def __init__(self, bestdetector=BestDetector(), steps=10):
        """Initialize with number of steps

        Parameters
        ----------
        bestdetector : BestDetector
          used to determine where the best error is located.
        steps : int
          How many steps to check after optimal value.
        """
        StoppingCriterion.__init__(self)
        if steps < 0:
            raise ValueError, \
                  "Number of steps (got %d) should be non-negative" % steps
        self.__bestdetector = bestdetector
        self.__steps = steps


    def __call__(self, errors):
        stop = False

        # just to prevent ValueError
        if len(errors)==0:
            return stop

        # charge best detector
        self.__bestdetector(errors)

        # if number of elements after the min >= len -- stop
        if len(errors) - self.__bestdetector.bestindex > self.__steps:
            stop = True

        return stop

    steps = property(fget=lambda x:x.__steps)



class ElementSelector(ClassWithCollections):
    """Base class to implement functors to select some elements based on a
    sequence of values.
    """

    ndiscarded = ConditionalAttribute(enabled=True,
        doc="Store number of discarded elements.")

    def __init__(self, mode='discard', **kwargs):
        """
        Parameters
        ----------
         mode : {'discard', 'select'}
            Decides whether to `select` or to `discard` features.
        """
        ClassWithCollections.__init__(self, **kwargs)

        self._set_mode(mode)
        """Flag whether to select or to discard elements."""


    ##REF: Name was automagically refactored
    def _set_mode(self, mode):
        """Choose `select` or `discard` mode."""

        if not mode in ['discard', 'select']:
            raise ValueError, "Unkown selection mode [%s]. Can only be one " \
                              "of 'select' or 'discard'." % mode

        self.__mode = mode


    def __call__(self, seq):
        """
        Parameters
        ----------
        seq
           Sequence based on values of which to perform the selection.
           If `Dataset`, then only 1st sample is taken.
        """
        if isinstance(seq, AttrDataset):
            if len(seq)>1:
                raise ValueError(
                    "Feature selectors cannot handle multiple "
                    "sequences in a Dataset at once.  We got dataset %s "
                    "as input."
                    % (seq,))
            seq = seq.samples[0]
        elif hasattr(seq, 'shape'):
            shape = seq.shape
            if len(shape) > 1:
                raise ValueError(
                    "Feature selectors cannot handle multidimensional "
                    "inputs (such as ndarrays with more than a single "
                    "dimension.  We got %s with shape %s "
                    "as input." % (seq.__class__, shape))
        return self._call(seq)

    def _call(self, seq):
        """Implementations in derived classed have to return a list of selected
        element IDs based on the given sequence.
        """
        raise NotImplementedError

    mode = property(fget=lambda self:self.__mode, fset=_set_mode)


class RangeElementSelector(ElementSelector):
    """Select elements based on specified range of values"""

    def __init__(self, lower=None, upper=None, inclusive=False,
                 mode='select', **kwargs):
        """Initialization `RangeElementSelector`

        Parameters
        ----------
         lower
           If not None -- select elements which are above of
           specified value
         upper
           If not None -- select elements which are lower of
           specified value
         inclusive
           Either to include end points
         mode
           overrides parent's default to be 'select' since it is more
           native for RangeElementSelector
           XXX TODO -- unify??

        `upper` could be lower than `lower` -- then selection is done
        on values <= lower or >=upper (ie tails). This would produce
        the same result if called with flipped values for mode and
        inclusive.

        If no upper no lower is set, assuming upper,lower=0, thus
        outputing non-0 elements
        """

        if lower is None and upper is None:
            lower, upper = 0, 0
            """Lets better return non-0 values if none of bounds is set"""

        # init State before registering anything
        ElementSelector.__init__(self, mode=mode, **kwargs)

        self.__range = (lower, upper)
        """Values on which to base selection"""

        self.__inclusive = inclusive

    def _call(self, seq):
        """Returns selected IDs.
        """
        lower, upper = self.__range
        len_seq = len(seq)
        if not lower is None:
            if self.__inclusive:
                selected = seq >= lower
            else:
                selected = seq > lower
        else:
            selected = np.ones( (len_seq), dtype=np.bool )

        if not upper is None:
            if self.__inclusive:
                selected_upper = seq <= upper
            else:
                selected_upper = seq < upper
            if not lower is None:
                if lower < upper:
                    # regular range
                    selected = np.logical_and(selected, selected_upper)
                else:
                    # outside, though that would be similar to exclude
                    selected = np.logical_or(selected, selected_upper)
            else:
                selected = selected_upper

        if self.mode == 'discard':
            selected = np.logical_not(selected)

        result = np.where(selected)[0]

        if __debug__:
            debug("ES", "Selected %d out of %d elements" %
                  (len(result), len_seq))
        return result

    lower = property(lambda self: self.__range[0])
    upper = property(lambda self: self.__range[1])

class TailSelector(ElementSelector):
    """Select elements from a tail of a distribution.

    The default behaviour is to discard the lower tail of a given distribution.
    """

    # TODO: 'both' to select from both tails
    def __init__(self, tail='lower', sort=True, **kwargs):
        """Initialize TailSelector

        Parameters
        ----------
         tail : ['lower', 'upper']
            Choose the tail to be processed.
         sort : bool
            Flag whether selected IDs will be sorted. Disable if not
            necessary to save some CPU cycles.

        """
        # init State before registering anything
        ElementSelector.__init__(self, **kwargs)

        self._set_tail(tail)
        """Know which tail to select."""

        self.__sort = sort


    ##REF: Name was automagically refactored
    def _set_tail(self, tail):
        """Set the tail to be processed."""
        if not tail in ['lower', 'upper']:
            raise ValueError, "Unkown tail argument [%s]. Can only be one " \
                              "of 'lower' or 'upper'." % tail

        self.__tail = tail


    ##REF: Name was automagically refactored
    def _get_n_elements(self, seq):
        """In derived classes has to return the number of elements to be
        processed given a sequence values forming the distribution.
        """
        raise NotImplementedError


    def _call(self, seq):
        """Returns selected IDs.
        """
        # TODO: Think about selecting features which have equal values but
        #       some are selected and some are not
        len_seq = len(seq)
        # how many to select (cannot select more than available)
        nelements = min(self._get_n_elements(seq), len_seq)

        # make sure that data is ndarray and compute a sequence rank matrix
        # lowest value is first
        seqrank = np.array(seq).argsort()

        if self.mode == 'discard' and self.__tail == 'upper':
            good_ids = seqrank[:-1*nelements]
            self.ca.ndiscarded = nelements
        elif self.mode == 'discard' and self.__tail == 'lower':
            good_ids = seqrank[nelements:]
            self.ca.ndiscarded = nelements
        elif self.mode == 'select' and self.__tail == 'upper':
            good_ids = seqrank[-1*nelements:]
            self.ca.ndiscarded = len_seq - nelements
        else: # select lower tail
            good_ids = seqrank[:nelements]
            self.ca.ndiscarded = len_seq - nelements

        # sort ids to keep order
        # XXX should we do here are leave to other place
        if self.__sort:
            good_ids.sort()

        # only return proper slice args: this is a list of int ids, hence return
        # a list not an array
        return list(good_ids)



class FixedNElementTailSelector(TailSelector):
    """Given a sequence, provide set of IDs for a fixed number of to be selected
    elements.
    """

    def __init__(self, nelements, **kwargs):
        """
        Parameters
        ----------
        nelements : int
          Number of elements to select/discard.
        """
        TailSelector.__init__(self, **kwargs)
        self.__nelements = None
        self._set_n_elements(nelements)


    def __repr__(self):
        return "%s number=%f" % (
            TailSelector.__repr__(self), self.nelements)


    ##REF: Name was automagically refactored
    def _get_n_elements(self, seq):
        return self.__nelements


    ##REF: Name was automagically refactored
    def _set_n_elements(self, nelements):
        if __debug__:
            if nelements <= 0:
                raise ValueError, "Number of elements less or equal to zero " \
                                  "does not make sense."

        self.__nelements = nelements


    nelements = property(fget=lambda x:x.__nelements,
                         fset=_set_n_elements)



class FractionTailSelector(TailSelector):
    """Given a sequence, provide Ids for a fraction of elements
    """

    def __init__(self, felements, **kwargs):
        """
        Parameters
        ----------
         felements : float (0,1.0]
            Fraction of elements to select/discard. Note: Even when 0.0 is
            specified at least one element will be selected.
        """
        TailSelector.__init__(self, **kwargs)
        self._set_f_elements(felements)


    def __repr__(self):
        return "%s fraction=%f" % (
            TailSelector.__repr__(self), self.__felements)


    ##REF: Name was automagically refactored
    def _get_n_elements(self, seq):
        num = int(floor(self.__felements * len(seq)))
        num = max(1, num)               # remove at least 1
        # no need for checks as base class will do anyway
        #return min(num, nselect)
        return num


    ##REF: Name was automagically refactored
    def _set_f_elements(self, felements):
        """What fraction to discard"""
        if felements > 1.0 or felements < 0.0:
            raise ValueError, \
                  "Fraction (%f) cannot be outside of [0.0,1.0]" \
                  % felements

        self.__felements = felements


    felements = property(fget=lambda x:x.__felements,
                         fset=_set_f_elements)



