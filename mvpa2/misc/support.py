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

import itertools
import math
import random
import re
import os
from os.path import join as pathjoin
import sys

from itertools import product
# for SmartVersion
from distutils.version import Version

import numpy as np
#import numpy.random as npr

from mvpa2.base import warning
from mvpa2.support.copy import copy, deepcopy
from mvpa2.base.types import is_sequence_type

if __debug__:
    from mvpa2.base import debug


##REF: Name was automagically refactored
def reuse_absolute_path(file1, file2, force=False):
    """Use path to file1 as the path to file2 is no absolute
    path is given for file2

    Parameters
    ----------
    force : bool
      if True, force it even if the file2 starts with /
    """
    if not file2.startswith(os.path.sep) or force:
        # lets reuse path to file1
        return pathjoin(os.path.dirname(file1), file2.lstrip(os.path.sep))
    else:
        return file2


##REF: Name was automagically refactored
def transform_with_boxcar(data, startpoints, boxlength, offset=0, fx=np.mean):
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
    selected = [ fx( data[ np.array(box) ], axis=0 ) for box in selector ]

    return np.array( selected )


def xunique_combinations(L, n):
    """Generator of unique combinations form a list L of objects in
    groups of size n.

    Parameters
    ----------
    L : list
      list of unique ids
    n : int
      grouping size

    Adopted from Li Daobing
    http://code.activestate.com/recipes/190465/
    (MIT license, according to activestate.com's policy)

    Also good discussions on combinations/variations/permutations
    with various implementations are available at
    http://mail.python.org/pipermail/python-list/2004-October/286054.html
    """
    if n == 0:
        yield []
    else:
        for i in xrange(len(L)-n+1):
            for cc in xunique_combinations(L[i+1:], n-1):
                yield [L[i]]+cc

def __xrandom_unique_combinations(L, n, k=None):
    """Generator of unique combinations form a list L of objects in
    groups of size n produced in random order

    "Elegant" but incorrect since pretty much samples the "tail"

    Parameters
    ----------
    L : list
      list of unique ids
    n : int
      grouping size
    k : int or None, optional
      limit number of combinations.  All of combinations are produced
      if k is None (default)

    Based on xunique_combinations adopted from Li Daobing
    http://code.activestate.com/recipes/190465/
    (MIT license, according to activestate.com's policy)
    """
    if k is not None:
        # Just a helper for convenient limiting
        g = xrandom_unique_combinations(L, n)
        for i in xrange(k):
            yield next(g)
    elif n == 0:
        yield []
    else:
        for i in npr.permutation(len(L)-n+1):
            for cc in xrandom_unique_combinations(
                npr.permutation(L[i+1:]), n-1):
                yield [L[i]]+cc


def ncombinations(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke

    Source: http://stackoverflow.com/questions/3025162/statistics-combinations-in-python/3025194

    Alternative implementations:
       scipy.misc.comb() -- approximation
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def xrandom_unique_combinations(L, n, k=None):
    """Generator of unique combinations form a list L of objects in
    groups of size n produced in random order

    Parameters
    ----------
    L : list
      list of unique ids
    n : int
      grouping size
    k : int or None
      limit number of combinations.  All of combinations are produced
      if k is None (default)

    """
    ncomb = ncombinations(len(L), n)
    if k is None:
        k = ncomb

    if (ncomb < 1e6 or k > math.sqrt(ncomb)) \
           and sys.version_info[:2] >= (2, 6):
        # so there is no sense really to mess with controlling for
        # non-repeats -- we can pre-generate all of them and just
        # choose needed number of random samples
        # Python2.5 doesn't have itertools.combinations
        for s in random.sample(list(itertools.combinations(L, n)), k):
            yield list(s)
    else:
        # Let's cycle through permutations while tracking
        # repeats
        seen = set()
        indexes = range(len(L)) # switch to indices so we could
                                # reliably hash them
        while len(seen) < min(k, ncomb):
            np.random.shuffle(indexes)
            sample = tuple(sorted(indexes[:n]))
            if not (sample in seen):
                yield [L[x] for x in sample]
                seen.add(sample)


def unique_combinations(L, n, sort=False):
    """Return unique combinations form a list L of objects in groups of size n.

    Parameters
    ----------
    L : list
      list of unique ids
    n : int
      length of the subsets to return
    sort : bool, optional
      if True -- result is sorted before returning

    If you are intended to use only a small subset of possible
    combinations, it is advised to use a generator
    `xunique_combinations`.
    """
    res = list(xunique_combinations(L, n))
    if sort:
        res = sorted(res)
    return res


##REF: Name was automagically refactored
def indent_doc(v):
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
    elif isinstance(val, dict):
        val = tuple(val.items())
    try:
        if sys.version_info[0] >= 3:
            # TODO: bytes is just a workaround and is slower
            # Anyway -- research joblib for hashing
            res += ":%s" % hash(bytes(val))
        else:
            res += ":%s" % hash(buffer(val))
    except:
        try:
            res += ":%s" % hash(val)
        except:
            pass
        pass
    return res

##REF: Name was automagically refactored
def is_sorted(items):
    """Check if listed items are in sorted order.

    Parameters
    ----------
      `items`: iterable container

    :return: `True` if were sorted. Otherwise `False` + Warning
    """
    items_sorted = deepcopy(items)
    items_sorted.sort()
    equality = items_sorted == items
    # XXX yarik forgotten analog to isiterable
    if hasattr(equality, '__iter__'):
        equality = np.all(equality)
    return equality


##REF: Name was automagically refactored
def is_in_volume(coord, shape):
    """For given coord check if it is within a specified volume size.

    Returns True/False. Assumes that volume coordinates start at 0.
    No more generalization (arbitrary minimal coord) is done to save
    on performance
    """
    for i in xrange(len(coord)):
        if coord[i] < 0 or coord[i] >= shape[i]:
            return False
    return True


def array_whereequal(a, x):
    """Reliable comparison for `numpy.ndarray`

    `numpy.ndarray` (as of 1.5.0.dev) fails to compare tuples in array of
    dtype object, e.g.

    >>> import numpy as np; a=np.array([1, (0,1)], dtype=object); print a == (0,1),  a[1] == (0,1)
    [False False] True

    This function checks if dtype is object and just does list
    comprehension in that case
    """
    if a.dtype is np.dtype('object'):
        return np.array([i==x for i in a], dtype=bool)
    else:
        return a == x


def version_to_tuple(v):
    """Convert literal string into a tuple, if possible of ints

    Tuple of integers constructed by splitting at '.' or interleaves
    of numerics and alpha numbers
    """
    if isinstance(v, basestring):
        v = map(str, v.split('.'))
    elif isinstance(v, (tuple, list)):
        # assure tuple
        pass
    else:
        raise ValueError, "Do not know how to treat version '%s'" % str(v)

    # Try to convert items into ints
    vres = []

    regex = re.compile('(?P<numeric>[0-9]*)'
                       '(?P<alpha>[~+-]*[A-Za-z]*)(?P<suffix>.*)')
    for x in v:
        try:
            vres += [int(x)]
        except ValueError:
            # try to split into sequences of literals and numerics
            suffix = x
            resd_prev = {}
            while suffix != '':
                res = regex.search(suffix)
                if res:
                    resd = res.groupdict()
                    if resd == resd_prev:
                        # we are in a loop, nothing meaningful would come out
                        vres += [suffix]
                        break
                    resd_prev = resd
                    if resd['numeric'] != '':
                        vres += [int(resd['numeric'])]
                    if resd['alpha'] != '':
                        vres += [resd['alpha']]
                    suffix = resd['suffix']
                else:
                    # We can't detect anything meaningful -- let it go as is
                    vres += [suffix]
                    break
    v = tuple(vres)

    return v

class SmartVersion(Version):
    """A bit evolved comparison of versions

    The reason for not using python's distutil.version is that it
    seems to have no clue about somewhat common conventions of using
    '-dev' or 'dev' or 'rc' suffixes for upcoming releases (so major
    version does contain upcoming release already).

    So here is an ad-hoc and not as nice implementation
    """

    def __reduce__(self):
        """Rudimentary __reduce__ because Version is not derived from object"""
        # parent class Version might not even assign any vstring when empty
        return self.__class__, (getattr(self, 'vstring', ''),)

    def parse(self, vstring):
        # Unicode gives grief on older releases and anyway arguably comparable
        if isinstance(vstring, unicode):
            vstring = str(vstring)
        self.vstring = vstring
        self.version = version_to_tuple(vstring)

    def __str__(self):
        try:
            return self.vstring
        except AttributeError:
            # Version.__init__ doesn't take care about assigning
            # .vstring if None is given, so let's just treat as it is
            # an empty string
            return ""

    def __cmp__(self, other):
        if isinstance(other, (str, unicode, tuple, list)):
            other = SmartVersion(other)
        elif isinstance(other, SmartVersion):
            pass
        elif isinstance(other, Version):
            other = SmartVersion(other.vstring)
        else:
            raise ValueError("Do not know how to treat version %s"
                             % str(other))

        if sys.version >= '3':
            def cmp(a, b):
                """Compatibility with Python3 -- regular (deprecated
                in 3) cmp operation should be sufficient for our needs"""
                return (a > b) - (a < b)
        else:
            # having above cmp overloads builtin cmp for this function so we
            # need manually rebind it or just resort to above cmp in general
            # (why not?)
            from __builtin__ import cmp

        # Do ad-hoc comparison of strings
        i = 0

        # if any of the versions was not parsed (e.g. if None was provided),
        # comparison can't be performed really unless both have no version
        # assigned
        if (not hasattr(self, 'version')) and (not hasattr(other, 'version')):
            return 0

        for v in (self, other):
            if not (hasattr(v, 'version')):
                raise ValueError('%s has no version information' % v)

        s, o = self.version, other.version
        regex_prerelease = re.compile('~|-?dev|-?rc|-?svn|-?pre|-?beta|-?alpha', re.I)
        for i in xrange(max(len(s), len(o))):
            if i < len(s): si = s[i]
            else: si = None
            if i < len(o): oi = o[i]
            else: oi = None

            if si == oi:
                continue

            for x,y,mult in ((si, oi, 1), (oi, si, -1)):
                if x is None:
                    if isinstance(y, int):
                        return -mult #  we got '.1' suffix
                    if isinstance(y, str):
                        if (regex_prerelease.match(y)):
                            return mult        # so we got something to signal
                                               # pre-release, so first one won
                        else:
                            # otherwise the other one wins
                            return -mult
                    else:
                        raise RuntimeError, "Should not have got here with %s" \
                              % y
                elif isinstance(x, int):
                    if not isinstance(y, int):
                        return mult
                    return mult*cmp(x, y) # both are ints
                elif isinstance(x, str):
                    if isinstance(y, str):
                        return mult*cmp(x,y)
        return 0

    if sys.version >= '3':
        # version.py logic in python3 does not rely on deprecated
        # __cmp__ but renames it into _cmp  and wraps in those various
        # comparators...  thus our good old __cmp__ should be ok for our
        # purposes here
        _cmp = __cmp__

##REF: Name was automagically refactored
def get_break_points(items, contiguous=True):
    """Return a list of break points.

    Parameters
    ----------
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


##REF: Name was automagically refactored
def rfe_history_to_maps(history):
    """Convert history generated by RFE into the array of binary maps

    Example:
      history2maps(np.array( [ 3,2,1,0 ] ))
    results in
      array([[ 1.,  1.,  1.,  1.],
             [ 1.,  1.,  1.,  0.],
             [ 1.,  1.,  0.,  0.],
             [ 1.,  0.,  0.,  0.]])
    """

    # assure that it is an array
    history = np.array(history)
    nfeatures, steps = len(history), max(history) - min(history) + 1
    history_maps = np.zeros((steps, nfeatures))

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
        ovstats = np.mean(maps, axis=0)

        self.overlap_map = (ovstats >= self.__overlap_threshold )
        self.spread_map = np.logical_and(ovstats > 0.0,
                                        ovstats < self.__overlap_threshold)
        self.ovstats_map = ovstats

        return np.mean(ovstats >= self.__overlap_threshold)


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
        """
        Parameters
        ----------
        **kwargs : dict
          All keys to describe the Event to initialize its dict.
        """
        # store everything
        dict.__init__(self, **kwargs)

        # basic checks
        for k in Event._MUSTHAVE:
            if not k in self:
                raise ValueError, "Event must have '%s' defined." % k


    ##REF: Name was automagically refactored
    def as_descrete_time(self, dt, storeoffset=False, offsetattr='offset'):
        """Convert `onset` and `duration` information into descrete timepoints.

        Parameters
        ----------
        dt : float
          Temporal distance between two timepoints in the same unit as `onset`
          and `duration`.
        storeoffset : bool
          If True, the temporal offset between original `onset` and
          descretized onset is stored as an additional item.
        offsetattr : str
          The name of the attribute that is used to store the computed offset
          in case the `storeoffset` is enabled.

        Returns
        -------
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
        out = copy(self)

        # get the timepoint just prior the onset
        out['onset'] = int(np.floor(onset / dt))

        if storeoffset:
            # compute offset
            offset = onset - (out['onset'] * dt)
            out[offsetattr] = offset

        if 'duration' in out:
            # how many timepoint cover the event (from computed onset
            # to the one timepoint just after the end of the event
            out['duration'] = int(np.ceil((onset + out['duration']) / dt) \
                                  - out['onset'])

        return out


def value2idx(val, x, solv='round'):
    """Convert a value into an index of the closes matching array element.

    Parameters
    ----------
    val : scalar
      Value that is to be converted.
    x : array or sequence
      One-dimensional array whose elements are used for comparision.
    solv : {'round', 'floor', 'ceil'}
      Resolver strategie: absolute closest element (round), closest smaller
      element (floor), or closest larger element (ceil).

    Returns
    -------
    int
    """
    # distance to val
    x = np.asanyarray(x) - val
    if solv == 'round':
        pass
    elif solv == 'ceil':
        x[x<0] = np.inf
    elif solv == 'floor':
        x[x>0] = np.inf
    else:
        ValueError("Unkown resolving method '%s'." % solv)
    x = np.abs(x)
    idx = np.argmin(x)
    return idx


def mask2slice(mask):
    """Convert a boolean mask vector into an equivalent slice (if possible).

    Parameters
    ----------
    mask: boolean array
      The mask.

    Returns
    -------
    slice or boolean array
      If possible the boolean mask is converted into a `slice`. If this is not
      possible the unmodified boolean mask is returned.
    """
    # the filter should be a boolean array
    # TODO Could be easily extended to also accept index arrays
    if not len(mask):
        raise ValueError("Got an empty mask.")
    # get indices of non-zero filter elements
    idx = mask.nonzero()[0]
    if not len(idx):
        return slice(0)
    idx_start = idx[0]
    idx_end = idx[-1] + 1
    idx_step = None
    if len(idx) > 1:
        # we need to figure out if there is a regular step-size
        # between elements
        stepsizes = np.unique(idx[1:] - idx[:-1])
        if len(stepsizes) > 1:
            # multiple step-sizes -> slicing is not possible -> return
            # orginal filter
            return mask
        else:
            idx_step = stepsizes[0]

    sl = slice(idx_start, idx_end, idx_step)
    if __debug__:
        debug("SPL", "Boolean mask conversion to slice is possible (%s)." % sl)
    return sl


def get_limit_filter(limit, collection):
    """Create a filter array from a limit definition.

    Parameters
    -----------
    limit : None or str or list or dict
      If ``None`` all elements will be included in the filter. If a single
      attribute name is given, its unique values will be used to define
      chunks of data that are marked in the filter as unique integers.
      If a list given, then combination of those attributes is used as a pair.
      Finally, if a dictionary is provided, its keys define attribute names and
      its values (single value or sequence thereof) attribute value, where all
      key-value combinations across all given items define a "selection" of
      elements to be included in the filter (OR combination).
    collection : Collection
      Dataset attribute collection instance that contains all attributes
      referenced in the limit specification, as well as defines the shape of
      the filter.

    Returns
    -------
    array
      This array is either boolean, where a `True` elements represent including
      in the filter, or the array is numerical, where its unique integer values
      define individual chunks of a filter.
    """
    attr_length = collection.attr_length

    if limit is None:
        # no limits
        limit_filter = np.ones(attr_length, dtype='bool')
    elif isinstance(limit, str):
        # use the unique values of this attribute to permute each chunk
        # individually
        lattr = collection[limit]
        lattr_data = lattr.value
        limit_filter = np.zeros(attr_length, dtype='int')
        for i, uv in enumerate(lattr.unique):
            limit_filter[lattr_data == uv] = i
    elif isinstance(limit, list):
        limit = list(set(limit))  # so if someone insane specified the same attr twice
        limit_filter = np.zeros(attr_length, dtype='int')
        for i, uvs in enumerate(product(*(collection[x].unique for x in limit))):
            uv_filter = np.ones(attr_length, dtype=bool)
            for l, uv in zip(limit, uvs):
                np.logical_and(uv_filter, collection[l].value==uv,
                               out=uv_filter)
            limit_filter[uv_filter] = i
    elif isinstance(limit, dict):
        limit_filter = np.zeros(attr_length, dtype='bool')
        for a in limit:
            if is_sequence_type(limit[a]):
                for v in limit[a]:
                    # enable the samples matching the value 'v' of the
                    # current limit attribute 'a'
                    limit_filter[collection[a].value == v] = True
            else:
                limit_filter[collection[a].value == limit[a]] = True
    else:
        raise RuntimeError("Unhandle condition")

    return limit_filter


def get_nelements_per_value(data):
    """Returns the number of elements per unique value of some sequence.

    Parameters
    ----------
    data : sequence
      This can be any sequence. In addition also ArrayCollectables are supported
      and this function will make use of any available pre-cached list of unique
      values.

    Returns
    -------
    dict with the number of elements (value) per unique value (key) in the
    sequence.
    """
    if hasattr(data, 'unique'):
        # if this is an ArrayAttribute save some time by using pre-cached unique
        # values
        uniquevalues = data.unique
        values = data.value
    else:
        uniquevalues = np.unique(data)
        values = data

    # use dictionary to cope with arbitrary values
    result = dict(zip(uniquevalues, [ 0 ] * len(uniquevalues)))
    for l in values:
        result[l] += 1

    return result

