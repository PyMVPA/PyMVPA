### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#    PyMVPA: Support function -- little helpers in everyday life
#
#    Copyright (C) 2007 by
#    Michael Hanke <michael.hanke@gmail.com>
#
#    This package is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2 of the License, or (at your option) any later version.
#
#    This package is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np

def transformWithBoxcar( data, startpoints, boxlength, offset=0):
    """ This function transforms a dataset by calculating the mean of a set of
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
    selected = [ np.mean( data[ np.array(box) ], axis=0 ) for box in selector ]

    return np.array( selected )


def buildConfusionMatrix( labels, targets, predictions ):
    """ Create a (N x N) confusion matrix.

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
    pred = np.array(predictions)

    # create the contingency table template
    mat = np.zeros( (len(labels), len(labels)), dtype = 'uint' )

    for t, tl in enumerate( labels ):
        for p, pl in enumerate( labels ):
            mat[t, p] = np.sum( pred[targets==tl] == pl )

    return mat


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
