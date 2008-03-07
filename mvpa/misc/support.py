#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support function -- little helpers in everyday life"""

__docformat__ = 'restructuredtext'

import numpy as N
import re

from copy import copy, deepcopy
from operator import isSequenceType

if __debug__:
    from mvpa.misc import debug

def transformWithBoxcar( data, startpoints, boxlength, offset=0, fx = N.mean ):
    """This function transforms a dataset by calculating the mean of a set of
    patterns. Such a pattern set is defined by a starting point and the size
    of the window along the first axis of the data ('boxlength').

    Parameters:
        data:           An array with an arbitrary number of dimensions.
        startpoints:    A sequence of index value along the first axis of
                        'data'.
        boxlength:      The number of elements after 'startpoint' along the
                        first axis of 'data' to be considered for averaging.
        offset:         The offset between the starting point and the
                        averaging window (boxcar).

    The functions returns an array with the length of the first axis being
    equal to the length of the 'startpoints' sequence.
    """
    if boxlength < 1:
        raise ValueError, "Boxlength lower than 1 makes no sense."

    # check for illegal boxes
    for sp in startpoints:
        if ( sp + offset + boxlength - 1 > len(data)-1 ) \
           or ( sp + offset < 0 ):
            raise ValueError, \
                  'Illegal box: start: %i, offset: %i, length: %i' \
                  % (sp, offset, boxlength)

    # build a list of list where each sublist contains the indexes of to be
    # averaged data elements
    selector = [ range( i + offset, i + offset + boxlength ) \
                 for i in startpoints ]

    # average each box
    selected = [ fx( data[ N.array(box) ], axis=0 ) for box in selector ]

    return N.array( selected )



def getUniqueLengthNCombinations(data, n):
    """Generates a list of lists containing all combinations of
    elements of data of length 'n' without repetitions.

        data: list
        n:    integer

    This function is adapted from a Java version posted in some forum on
    the web as an answer to the question 'How can I generate all possible
    combinations of length n?'. Unfortunately I cannot remember which
    forum it was.
    """

    # to be returned
    combos = []

    # local function that will be called recursively to collect the
    # combination elements
    def take(data, occupied, depth, taken):
        for i, d in enumerate(data):
            # only do something if this element hasn't been touch yet
            if occupied[i] == False:
                # see whether will reached the desired length
                if depth < n-1:
                    # flag the current element as touched
                    occupied[i] = True
                    # next level
                    take(data, occupied, depth+1, taken + [d])
                    # if the current element would be set 'free', it would
                    # results in ALL combinations of elements (obeying order
                    # of elements) and not just in the unique sets of
                    # combinations (without order)
                    #occupied[i] = False
                else:
                    # store the final combination
                    combos.append(taken + [d])
    # some kind of bitset that stores the status of each element
    # (contained in combination or not)
    occupied = [False] * len(data)
    # get the combinations
    take(data, occupied, 0, [])

    # return the result
    return combos


def indentDoc(v):
    """Given a `value` returns a string where each line is indented

    Needed for a cleaner __repr__ output
    `v` - arbitrary
    """
    return re.sub('\n', '\n  ', str(v))


def isSorted(items):
    """Check if listed items are in sorted order.

    :Parameters:
        `items`: iterable container

    :return: `True` if were sorted. Otherwise `False` + Warning
    """
    itemsOld = deepcopy(items)
    items.sort()
    equality = itemsOld == items
    # XXX yarik forgotten analog to isiterable
    if hasattr(equality, '__iter__'):
        equality = N.all(equality)
    return equality


def getBreakPoints(items, contiguous=True):
    """Return a list of break points.

    :Parameters:
      items : iterable
        list of items, such as chunks
      contiguous : bool
        if `True` (default) then raise Value Error if items are not
        contiguous, i.e. a label occur in multiple contiguous sets

    :raises: ValueError

    :return: list of indexes for every new set of items
    """
    prev = None # pylint happiness event!
    known = []
    """List of items which was already seen"""
    result = []
    """Resultant list"""
    for index in xrange(len(items)):
        item = items[index]
        if item in known:
            if index > 0:
                if prev != item:            # breakpoint
                    if contiguous:
                        raise ValueError, \
                        "Item %s was already seen before" % str(item)
                    else:
                        result.append(index)
        else:
            known.append(item)
            result.append(index)
        prev = item
    return result


class MapOverlap(object):
    """Compute some overlap stats from a sequence of binary maps.

    When called with a sequence of binary maps (e.g. lists or arrays) the
    fraction of mask elements that are non-zero in a customizable proportion
    of the maps is returned. By default this threshold is set to 1.0, i.e.
    such an element has to be non-zero in *all* maps.

    Three additional maps (same size as original) are computed:

      * overlap_map: binary map which is non-zero for each overlapping element.
      * spread_map:  binary map which is non-zero for each element that is
                     non-zero in any map, but does not exceed the overlap
                     threshold.
      * ovstats_map: map of float with the raw elementwise fraction of overlap.

    All maps are available via class members.
    """
    def __init__(self, overlap_threshold=1.0):
        """Nothing to be seen here.
        """
        self.__overlap_threshold = overlap_threshold

        # pylint happiness block
        self.overlap_map = None
        self.spread_map = None
        self.ovstats_map = None


    def __call__(self, maps):
        """Returns fraction of overlapping elements.
        """
        ovstats = N.mean(maps, axis=0)

        self.overlap_map = (ovstats >= self.__overlap_threshold )
        self.spread_map = N.logical_and(ovstats > 0.0,
                                        ovstats < self.__overlap_threshold)
        self.ovstats_map = ovstats

        return N.mean(ovstats >= self.__overlap_threshold)


class Loop(object):
    """World domination helper: do whatever it is asked and accumulate results

    XXX Thinks about:
      - Can we have multiple 'call's?
      - Might we need to deepcopy attributes values?
      - Might we need to specify what attribs to copy and which just to bind?
    """

    def __init__(self, looper, call,
                 unroll=True, attribs=None, copy_attribs=True):
        """Initialize

        :Parameters:
          looper
            Generator which feeds the loop with new entries
          call : Functor
            Functor which is called in the loop
          unroll : bool
            Either to unroll output of looper into a list of arguments for
            call
          attribs : list of basestr
            What attributes of call to store and return later on?
          copy_attribs : bool
            Force copying values of attributes
        """

        self.__looper = looper
        """Generator which feeds the loop"""

        self.__call = call
        """Call which gets called in the loop"""

        self.__unroll = unroll

        if attribs is None:
            attribs = []

        self.__attribs = attribs
        self.__copy_attribs = copy_attribs


    def __call__(self, *args, **kwargs):
        """
        """
        # assign to local unroll since we might change it locally
        # later on
        unroll = self.__unroll

        # Initialize returned value -- dictionary of desired things
        results = dict([ ('result', [])] +
                       [(a, []) for a in self.__attribs])

        # Lets do it!
        for (i, X) in enumerate(self.__looper(*args, **kwargs)):

            if i == 0 and unroll and not isSequenceType(X):

                # XXX or should it be warning?
                # MH: should be an exception as it is definitely not going
                #     like it was intended.
                raise RuntimeError, \
                      "Cannot unroll non-sequence result from looper %s" % \
                      `self.__looper`
                #if __debug__:
                #    debug("LOOP",
                #          "Cannot unroll non-sequence result from looper %s" %
                #          `self.__looper` + " disabling unrolling")
                #unroll = False

            if unroll:
                result = self.__call(*X)
            else:
                result = self.__call(X)

            # XXX pylint doesn't like `` for some reason
            if __debug__:
                debug("LOOP", "Iteration %i on call %s. Got result %s" %
                      (i, `self.__call`, `result`))


            results['result'].append(result)

            for attrib in self.__attribs:
                attrv = self.__call.__getattribute__(attrib)

                if self.__copy_attribs:
                    attrv = copy(attrv)

                results[attrib].append(attrv)

        if len(self.__attribs) > 0:
            return results
        else:
            return results['result']


def loop(looper, call,
         unroll=True, attribs=None, copy_attribs=True, *args, **kwargs):
    """XXX Loop twin brother

    Helper for those who just wants to do smth like
       loop(blah, bleh, grgr)
     instead of
       Loop(blah, bleh)(grgr)
    """

    return Loop(looper=looper, call=call, unroll=unroll,
                attribs=attribs, copy_attribs=copy_attribs)(*args, **kwargs)
