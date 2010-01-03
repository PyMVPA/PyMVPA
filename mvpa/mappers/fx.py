# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Transform data by applying a function along samples or feature axis."""

__docformat__ = 'restructuredtext'

import numpy as N
import operator

from mvpa.mappers.base import Mapper


class FxMapper(Mapper):
    """Apply a custom transformation to (groups of) samples or features.

    """
    def __init__(self, axis, fx, fxargs=None, uattrs=None,
                 attrfx='merge'):
        """
        Parameters
        ----------
        axis : {'samples', 'features'}
        fx : callable
        fxargs : tuple
        uattrs: list
          List of atrribute names to consider. All possible combinations
          of unqiue elements of these attributes are used to determine the
          sample groups to operate on.
        attrfx: callable
          Functor that is called with each sample attribute elements matching
          the respective samples group. By default the unique value is
          determined. If the content of the attribute is not uniform for a
          samples group a unique string representation is created.
          If `None`, attributes are not altered.
        """
        Mapper.__init__(self)

        if not axis in ['samples', 'features']:
            raise ValueError("%s `axis` arguments can only be 'samples' or "
                             "'features' (got: '%s')." % repr(axis))
        self.__axis = axis
        self.__uattrs = uattrs
        self.__fx = fx
        if not fxargs is None:
            self.__fxargs = fxargs
        else:
            self.__fxargs = ()
        if attrfx == 'merge':
            self.__attrfx = _uniquemerge2literal
        else:
            self.__attrfx = attrfx


    def _train(self, ds):
        # right now it needs no training
        pass


    def _forward_data(self, data):
        if not self.__uattrs is None:
            raise RuntimeError("%s does not support forward-mapping of plain "
                               "data when data grouping based on attributes "
                               "is requested" 
                               % self.__class__.__name__)
        # apply fx along samples axis for each feature
        if self.__axis == 'samples':
            mdata = N.apply_along_axis(self.__fx, 0, data, *self.__fxargs)
        # apply fx along features axis for each sample
        elif self.__axis == 'features':
            mdata = N.apply_along_axis(self.__fx, 1, data, *self.__fxargs)
        return N.atleast_2d(mdata)


    def _forward_dataset(self, ds):
        if self.__uattrs is None:
            mdata, sattrs = self._forward_dataset_full(ds)
        else:
            mdata, sattrs = self._forward_dataset_grouped(ds)

        samples = N.atleast_2d(mdata)

        # return early if there is no attribute treatment desired
        if self.__attrfx is None:
            out = ds.copy(deep=False)
            out.samples = samples
            return out

        # not copying the samples attributes, since they have to be modified
        # anyway
        if self.__axis == 'samples':
            out = ds.copy(deep=False, sa=[])
            col = out.sa
            col.set_length_check(samples.shape[0])
        else:
            out = ds.copy(deep=False, fa=[])
            col = out.fa
            col.set_length_check(samples.shape[1])
        # assign samples to do COW
        out.samples = samples

        for attr in sattrs:
            a = sattrs[attr]
            # need to handle single literal attributes
            if isinstance(a, str):
                col[attr] = [a]
            else:
                col[attr] = N.atleast_1d(a)

        return out


    def _forward_dataset_grouped(self, ds):
        mdata = [] # list of samples array pieces
        if self.__axis == 'samples':
            col = ds.sa
            axis = 0
        elif self.__axis == 'features':
            col = ds.fa
            axis = 1
        else:
            raise RuntimeError("This should not have happened!")

        attrs = dict(zip(col.keys(), [[] for i in col]))

        # create a dictionary for all unique elements in all attribute this
        # mapper should operate on
        self.__attrcombs = dict(zip(self.__uattrs,
                                [col[attr].unique for attr in self.__uattrs]))
        # let it generate all combinations of unique elements in any attr
        for comb in _orthogonal_permutations(self.__attrcombs):
            selector = reduce(N.multiply,
                                [col[attr].value == value
                                    for attr, value in comb.iteritems()])
            # process the samples
            if axis == 0:
                samples = ds.samples[selector]
            else:
                samples = ds.samples[:,selector]

            fxed_samples = N.apply_along_axis(self.__fx, axis, samples,
                                              *self.__fxargs)
            mdata.append(fxed_samples)
            if not self.__attrfx is None:
                # and now all samples attributes
                fxed_attrs = [self.__attrfx(col[attr].value[selector])
                                    for attr in col]
                for i, attr in enumerate(col):
                    attrs[attr].append(fxed_attrs[i])

        if axis == 0:
            mdata = N.vstack(mdata)
        else:
            mdata = N.vstack(N.transpose(mdata))
        return mdata, attrs


    def _forward_dataset_full(self, ds):
        # simply map the all of the data
        mdata = self._forward_data(ds.samples)

        # if the attributes should not be handled, don't handle them
        if self.__attrfx is None:
            return mdata, None

        # and now all attributes
        if self.__axis == 'samples':
            attrs = dict(zip(ds.sa.keys(),
                              [self.__attrfx(ds.sa[attr].value)
                                    for attr in ds.sa]))
        if self.__axis == 'features':
            attrs = dict(zip(ds.fa.keys(),
                              [self.__attrfx(ds.fa[attr].value)
                                    for attr in ds.fa]))
        return mdata, attrs


#
# Convenience functions to create some useful mapper with less complexity
#

def mean_sample(attrfx='merge'):
    """Returns a mapper that computes the mean sample of a dataset.

    Parameters
    ----------
    attrfx : 'merge' or callable
      Callable that is used to determine the sample attributes of the computed
      mean samples. By default this will be a string representation of all
      unique value of a particular attribute in any sample group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('samples', N.mean, attrfx=attrfx)


def mean_group_sample(attrs, attrfx='merge'):
    """Returns a mapper that computes the mean samples of unique sample groups.

    The sample groups are identified by the unique combination of all values of
    a set of provided sample attributes.

    Parameters
    ----------
    attrs : list
      List of sample attributes whos unique values will be used to identify the
      samples groups.
    attrfx : 'merge' or callable
      Callable that is used to determine the sample attributes of the computed
      mean samples. By default this will be a string representation of all
      unique value of a particular attribute in any sample group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('samples', N.mean, uattrs=attrs, attrfx=attrfx)


def mean_group_feature(attrs, attrfx='merge'):
    """Returns a mapper that computes the mean features of unique feature groups.

    The feature groups are identified by the unique combination of all values of
    a set of provided feature attributes.

    Parameters
    ----------
    attrs : list
      List of feature attributes whos unique values will be used to identify the
      feature groups.
    attrfx : 'merge' or callable
      Callable that is used to determine the feature attributes of the computed
      mean features. By default this will be a string representation of all
      unique value of a particular attribute in any feature group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('features', N.mean, uattrs=attrs, attrfx=attrfx)


def absolute_features():
    """Returns a mapper that converts features into absolute values.

    This mapper does not alter any attributes.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('features', N.absolute, attrfx=None)


#
# Utility functions
#

def _uniquemerge2literal(attrs):
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


def _orthogonal_permutations(a_dict):
    """
    Takes a dictionary with lists as values and returns all permutations
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
