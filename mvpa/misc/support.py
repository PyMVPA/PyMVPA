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

from StringIO import StringIO
from copy import deepcopy
from math import log10, ceil

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


def buildConfusionMatrix( labels, targets, predictions ):
    """Create a (N x N) confusion matrix.

    'N' is the number of labels in the matrix. The labels itself have to be
    given in the 'labels' argument. 'targets' and 'predictions' are two
    length-n vectors, one containing the classification targets and the other
    the corresponding predictions. The confusion matrix has to following
    layout:

                  predictions
                  1  2  .  .  N
                1
                2
      targets   .
                .     (i,j)
                N

    where cell (i,j) contains the absolute number of predictions j where
    the target would have been i.
    """
    # needs to be an array
    pred = N.array(predictions)

    # create the contingency table template
    mat = N.zeros( (len(labels), len(labels)), dtype = 'uint' )

    for t, tl in enumerate( labels ):
        for p, pl in enumerate( labels ):
            mat[t, p] = N.sum( pred[targets==tl] == pl )

    return mat


# XXX we have to refactor those two functions -- probably into some class
def pprintConfusionMatrix(labels, matrix,
                          header=True, percents=True, summary=True):
    """Returns a string with nicely formatted matrix"""
    out = StringIO()
    Nlabels = len(labels)
    Nsamples = N.sum(matrix, axis=1)
    Ndigitsmax = int(ceil(log10(max(Nsamples))))
    Nlabelsmax = max( [len(x) for x in labels] )
    L = max(Ndigitsmax, Nlabelsmax)     # length of a single label/value
    res = ""

    prefixlen = Nlabelsmax+2+Ndigitsmax+2
    pref = ' '*(prefixlen) # empty prefix
    if header:
        # print out the header
        out.write(pref)
        for label in labels:
            # center damn label
            Nspaces = int(ceil((L-len(label))/2.0))
            out.write(" %%%ds%%s%%%ds"
                      % (Nspaces, L-Nspaces-len(label))
                      % ('', label, ''))
        out.write("\n")

        # underscores
        out.write("%s%s\n" % (pref, (" %s" % ("-" * L)) * Nlabels))

    if matrix.shape != (Nlabels, Nlabels):
        raise ValueError, "Number of labels %d doesn't correspond the size" + \
              " of a confusion matrix %s" % (Nlabels, matrix.shape)

    correct = 0
    for i in xrange(Nlabels):
        # print the label
        out.write("%%%ds {%%%dd}" % (Nlabelsmax, Ndigitsmax) % (labels[i], Nsamples[i])),
        for j in xrange(Nlabels):
            out.write(" %%%dd" % L % matrix[i, j])
        if percents:
            out.write(' [%5.2f%%]' % (matrix[i, i] * 100.0 / Nsamples[i]))
        correct += matrix[i, i]
        out.write("\n")

    if summary:
        out.write("%%-%ds%%s\n"
                  % prefixlen
                  % ("", "-"*((L+1)*Nlabels)))

        out.write("%%-%ds[%%5.2f%%%%]\n"
                  % (prefixlen + (L+1)*Nlabels)
                  % ("Total Correct {%d}" % correct, 100.0*correct/sum(Nsamples) ))


    result = out.getvalue()
    out.close()
    return result
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
                    take(data, occupied, depth+1, taken + [data[i]])
                    # 'free' the current element
                    occupied[i] == False
                else:
                    # store the final combination
                    combos.append(taken + [data[i]])
    # some kind of bitset that stores the status of each element
    # (contained in combination or not)
    occupied = [ False for i in data ]
    # get the combinations
    take(data, occupied, 0, [])

    # return the result
    return combos


def indentDoc(v):
    """Given a `value` returns a string where each line is indented

    Needed for a cleaner __repr__ output
    `v` - arbitrary
    """
    return re.sub('\n', '\n  ', `v`)


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

    known = []
    """List of items which was already seen"""
    result = []
    """Resultant list"""
    for index in xrange(len(items)):
        item = items[index]
        if item in known:
            if index>0:
                if prev != item:            # breakpoint
                    if contiguous:
                        raise ValueError, \
                        "Item %s was already seen before" % `item`
                    else:
                        result.append(index)
        else:
            known.append(item)
            result.append(index)
        prev = item
    return result
