# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Support function -- little helpers in everyday life"""

__docformat__ = 'restructuredtext'

import numpy as N
import re, os

from mvpa.support.copy import copy, deepcopy
from operator import isSequenceType

if __debug__:
    from mvpa.base import debug


def reuseAbsolutePath(file1, file2, force=False):
    """Use path to file1 as the path to file2 is no absolute
    path is given for file2

    :Parameters:
      force : bool
        if True, force it even if the file2 starts with /
    """
    if not file2.startswith('/') or force:
        # lets reuse path to file1
        return os.path.join(os.path.dirname(file1), file2.lstrip('/'))
    else:
        return file2


def transformWithBoxcar(data, startpoints, boxlength, offset=0, fx=N.mean):
    """This function extracts boxcar windows from an array. Such a boxcar is
    defined by a starting point and the size of the window along the first axis
    of the array (`boxlength`). Afterwards a customizable function is applied
    to each boxcar individually (Default: averaging).

    :param data: An array with an arbitrary number of dimensions.
    :type data: array
    :param startpoints: Boxcar startpoints as index along the first array axis
    :type startpoints: sequence
    :param boxlength: Length of the boxcar window in #array elements
    :type boxlength: int
    :param offset: Optional offset between the configured starting point and the
      actual begining of the boxcar window.
    :type offset: int
    :rtype: array (len(startpoints) x data.shape[1:])
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


def idhash(val):
    """Craft unique id+hash for an object
    """
    res = "%s" % id(val)
    if isinstance(val, list):
        val = tuple(val)
    try:
        res += ":%s" % hash(buffer(val))
    except:
        try:
            res += ":%s" % hash(val)
        except:
            pass
        pass
    return res

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


def isInVolume(coord, shape):
    """For given coord check if it is within a specified volume size.

    Returns True/False. Assumes that volume coordinates start at 0.
    No more generalization (arbitrary minimal coord) is done to save
    on performance
    """
    for i in xrange(len(coord)):
        if coord[i] < 0 or coord[i] >= shape[i]:
            return False
    return True


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


def RFEHistory2maps(history):
    """Convert history generated by RFE into the array of binary maps

    Example:
      history2maps(N.array( [ 3,2,1,0 ] ))
    results in
      array([[ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  0.],
             [ 1.,  1.,  0.,  0.],
             [ 1.,  0.,  0.,  0.]])
    """

    # assure that it is an array
    history = N.array(history)
    nfeatures, steps = len(history), max(history) - min(history) + 1
    history_maps = N.zeros((steps, nfeatures))

    for step in xrange(steps):
        history_maps[step, history >= step] = 1

    return history_maps


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


class Event(dict):
    """Simple class to define properties of an event.

    The class is basically a dictionary. Any properties can
    be passed as keyword arguments to the constructor, e.g.:

      >>> ev = Event(onset=12, duration=2.45)

    Conventions for keys:

    `onset`
      The onset of the event in some unit.
    `duration`
      The duration of the event in the same unit as `onset`.
    `label`
      E.g. the condition this event is part of.
    `chunk`
      Group this event is part of (if any), e.g. experimental run.
    `features`
      Any amount of additional features of the event. This might include
      things like physiological measures, stimulus intensity. Must be a mutable
      sequence (e.g. list), if present.
    """
    _MUSTHAVE = ['onset']

    def __init__(self, **kwargs):
        # store everything
        dict.__init__(self, **kwargs)

        # basic checks
        for k in Event._MUSTHAVE:
            if not self.has_key(k):
                raise ValueError, "Event must have '%s' defined." % k


    def asDescreteTime(self, dt, storeoffset=False):
        """Convert `onset` and `duration` information into descrete timepoints.

        :Parameters:
          dt: float
            Temporal distance between two timepoints in the same unit as `onset`
            and `duration`.
          storeoffset: bool
            If True, the temporal offset between original `onset` and
            descretized `onset` is stored as an additional item in `features`.

        :Return:
          A copy of the original `Event` with `onset` and optionally `duration`
          replaced by their corresponding descrete timepoint. The new onset will
          correspond to the timepoint just before or exactly at the original
          onset. The new duration will be the number of timepoints covering the
          event from the computed onset timepoint till the timepoint exactly at
          the end, or just after the event.

          Note again, that the new values are expressed as #timepoint and not
          in their original unit!
        """
        dt = float(dt)
        onset = self['onset']
        out = deepcopy(self)

        # get the timepoint just prior the onset
        out['onset'] = int(N.floor(onset / dt))

        if storeoffset:
            # compute offset
            offset = onset - (out['onset'] * dt)

            if out.has_key('features'):
                out['features'].append(offset)
            else:
                out['features'] = [offset]

        if out.has_key('duration'):
            # how many timepoint cover the event (from computed onset
            # to the one timepoint just after the end of the event
            out['duration'] = int(N.ceil((onset + out['duration']) / dt) \
                                  - out['onset'])

        return out



class HarvesterCall(object):
    def __init__(self, call, attribs=None, argfilter=None, expand_args=True,
                 copy_attribs=True):
        """Initialize

        :Parameters:
          expand_args : bool
            Either to expand the output of looper into a list of arguments for
            call
          attribs : list of basestr
            What attributes of call to store and return later on?
          copy_attribs : bool
            Force copying values of attributes
        """

        self.call = call
        """Call which gets called in the harvester."""

        if attribs is None:
            attribs = []
        if not isSequenceType(attribs):
            raise ValueError, "'attribs' have to specified as a sequence."

        if not (argfilter is None or isSequenceType(argfilter)):
            raise ValueError, "'argfilter' have to be a sequence or None."

        # now give it to me...
        self.argfilter = argfilter
        self.expand_args = expand_args
        self.copy_attribs = copy_attribs
        self.attribs = attribs



class Harvester(object):
    """World domination helper: do whatever it is asked and accumulate results

    XXX Thinks about:
      - Might we need to deepcopy attributes values?
      - Might we need to specify what attribs to copy and which just to bind?
    """

    def __init__(self, source, calls, simplify_results=True):
        """Initialize

        :Parameters:
          source
            Generator which produce food for the calls.
          calls : sequence of HarvesterCall instances
            Calls which are processed in the loop. All calls are processed in
            order of apperance in the sequence.
          simplify_results: bool
            Remove unecessary overhead in results if possible (nested lists
            and dictionaries).
       """
        if not isSequenceType(calls):
            raise ValueError, "'calls' have to specified as a sequence."

        self.__source = source
        """Generator which feeds the harvester"""

        self.__calls = calls
        """Calls which gets called with each generated source"""

        self.__simplify_results = simplify_results


    def __call__(self, *args, **kwargs):
        """
        """
        # prepare complex result structure for all calls and their respective
        # attributes: calls x dict(attributes x loop iterations)
        results = [dict([('result', [])] + [(a, []) for a in c.attribs]) \
                        for c in self.__calls]

        # Lets do it!
        for (i, X) in enumerate(self.__source(*args, **kwargs)):
            for (c, call) in enumerate(self.__calls):
                # sanity check
                if i == 0 and call.expand_args and not isSequenceType(X):
                    raise RuntimeError, \
                          "Cannot expand non-sequence result from %s" % \
                          `self.__source`

                # apply argument filter (and reorder) if requested
                if call.argfilter:
                    filtered_args = [X[f] for f in call.argfilter]
                else:
                    filtered_args = X

                if call.expand_args:
                    result = call.call(*filtered_args)
                else:
                    result = call.call(filtered_args)

#                # XXX pylint doesn't like `` for some reason
#                if __debug__:
#                    debug("LOOP", "Iteration %i on call %s. Got result %s" %
#                          (i, `self.__call`, `result`))


                results[c]['result'].append(result)

                for attrib in call.attribs:
                    attrv = call.call.__getattribute__(attrib)

                    if call.copy_attribs:
                        attrv = copy(attrv)

                    results[c][attrib].append(attrv)

        # reduce results structure
        if self.__simplify_results:
            # get rid of dictionary if just the results are requested
            for (c, call) in enumerate(self.__calls):
                if not len(call.attribs):
                    results[c] = results[c]['result']

            if len(self.__calls) == 1:
                results = results[0]

        return results


# XXX MH: this doesn't work in all cases, as you cannot have *args after a
#         kwarg.
#def loop(looper, call,
#         unroll=True, attribs=None, copy_attribs=True, *args, **kwargs):
#    """XXX Loop twin brother
#
#    Helper for those who just wants to do smth like
#       loop(blah, bleh, grgr)
#     instead of
#       Loop(blah, bleh)(grgr)
#    """
#    print looper, call, unroll, attribs, copy_attribs
#    print args, kwargs
#    return Loop(looper=looper, call=call, unroll=unroll,
#                attribs=attribs, copy_attribs=copy_attribs)(*args, **kwargs)
