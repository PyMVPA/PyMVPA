# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data mapper"""

__docformat__ = 'restructuredtext'

import numpy as N
import operator

from mvpa.mappers.base import Mapper
from mvpa.misc.transformers import FirstAxisMean

if __debug__:
    from mvpa.base import debug


def uniquemerge2literal(attrs):
    """Compress a sequence into its unique elements (with string merge).

    Whenever there is more then one unique element in `attrs`, these
    are converted to a string and join with a '+' character inbetween.

    Parameters
    ----------
    attrs: sequence, arbitrary

    Returns
    -------
    Non-sequence arguments are passed as is. Sequences are converted into
    a single item representation (see above) and returned.
    """
    # only do something if multiple items are given
    if not operator.isSequenceType(attrs):
        return attrs
    unq = N.unique(attrs)
    if len(unq) > 1:
        return '+'.join([str(l) for l in unq])
    else:
        return unq[0]


class SampleGroupMapper(Mapper):
    """Mapper to apply a mapping function to samples of the same type.

    A customizable function is applied individually to all samples with the
    same unique combination of items of all desired attributes. This mapper is
    somewhat unconventional since it doesn't preserve number of samples (ie the
    size of 0-th dimension...)
    """
    def __init__(self, attrs, sfx=FirstAxisMean, safx=uniquemerge2literal):
        """
        Parameters
        ----------
        attrs: list
          List of atrribute names to consider. All possible combinations
          of unqiue elements of these attributes are used to determine the
          sample groups to operate on.
        sfx: functor
          Functor that is called with the samples matching each group. By
          default the mean along the first axis is computed.
        safx: functor
          Functor that is called with each sample attribute elements matching
          the respective samples group. By default the unique value is
          determined. If the content of the attribute is not uniform for a
          samples group a unique string representation is created.
        """
        # TODO need to have resolver for samples attrs.
        Mapper.__init__(self)

        self.__attrs = attrs
        self.__datashape = None
        self.__fx = sfx
        self.__safx = safx


    def _train(self, dataset):
        # store the datashape to be able to have working get_(in,out)size()
        self.__datashape = (dataset.nfeatures, )


    def _forward_dataset(self, ds):
        # create a dictionary for all unique elements in all attribute this
        # mapper should operate on
        combs = dict(zip(self.__attrs,
                         [ds.sa[attr].unique for attr in self.__attrs]))

        mdata = [] # list of samples array pieces
        sattrs = dict(zip(ds.sa.keys(), [[] for i in ds.sa]))

        # let it generate all combinations of unique elements in any attr
        for comb in _orthogonal_permutations(combs):
            selector = reduce(N.multiply,
                                [ds.sa[attr].value == value
                                    for attr, value in comb.iteritems()])
            # process the samples
            fxed_samples = self.__fx(ds.samples[selector])
            mdata.append(fxed_samples)
            # and now all samples attributes
            # TODO use different resolver/fx
            fxed_sattrs = [self.__safx(ds.sa[attr].value[selector])
                                for attr in ds.sa]
            for i, attr in enumerate(ds.sa):
                sattrs[attr].append(fxed_sattrs[i])

        out = ds.copy(deep=False)
        # assign samples to do COW
        out.samples = N.array(mdata)
        out.sa.set_length_check(len(out.samples))
        for attr in sattrs:
            out.sa[attr].value = N.array(sattrs[attr])

        return out


    def get_insize(self):
        """Returns the number of original samples which were combined.
        """
        return self.__datashape[0]


    def get_outsize(self):
        """Returns the number of output samples.
        """
        return self.__datashape[0]


def _orthogonal_permutations(a_dict):
    """
    Takes a dictionary with lists as keys and returns all permutations
    of these list elements in new dicts.

    This function is useful, when a method with several arguments
    shall be tested and all of the arguments can take several values.

    The order is not defined, therefore the elements should be
    orthogonal to each other.

    >>> for i in orthogonal_permutations({'a': [1,2,3], 'b': [4,5]}):
            print i
    {'a': 1, 'b': 4}
    {'a': 1, 'b': 5}
    {'a': 2, 'b': 4}
    {'a': 2, 'b': 5}
    {'a': 3, 'b': 4}
    {'a': 3, 'b': 5}
    """
    # Taken from MDP (LGPL)
    pool = dict(a_dict)
    args = []
    for func, all_args in pool.items():
        # check the size of the list in the second item of the tuple
        args_with_fun = [(func, arg) for arg in all_args]
        args.append(args_with_fun)
    for i in _product(args):
        yield dict(i)

def _product(iterable):
    # MDP took it and adapted it from itertools 2.6 (Python license)
    # PyMVPA took it from MDP (LGPL)
    pools = tuple(iterable)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)
