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

import numpy as np
import inspect

from mvpa2.base import warning
from mvpa2.base.node import Node
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import *
from mvpa2.datasets import Dataset
from mvpa2.base.dochelpers import _str, _repr_attrs
from mvpa2.mappers.base import Mapper
from mvpa2.misc.support import array_whereequal
from mvpa2.base.dochelpers import borrowdoc

from mvpa2.misc.transformers import sum_of_abs, max_of_abs, subtract_mean

if __debug__:
    from mvpa2.base import debug

class FxMapper(Mapper):
    """Apply a custom transformation to (groups of) samples or features.
    """

    is_trained = True
    """Indicate that this mapper is always trained."""

    def __init__(self, axis, fx, fxargs=None, uattrs=None,
                 attrfx='merge', order='uattrs'):
        """
        Parameters
        ----------
        axis : {'samples', 'features'}
        fx : callable
        fxargs : tuple
          Passed as *args to ``fx``
        uattrs : list
          List of attribute names to consider. All possible combinations
          of unique elements of these attributes are used to determine the
          sample groups to operate on.
        attrfx : callable
          Functor that is called with each sample attribute elements matching
          the respective samples group. By default the unique value is
          determined. If the content of the attribute is not uniform for a
          samples group a unique string representation is created.
          If `None`, attributes are not altered.
        order : {'uattrs', 'occurrence', None}
          If which order groups should be merged together.  If `None` (default
          before 2.3.1), the order is imposed only by the order of
          `uattrs` as keys in the dictionary, thus can vary from run to run.
          If `'occurrence'`, groups will be ordered by the first occurrence
          of group samples in original dataset. If `'uattrs'`, groups will be
          sorted by the values of uattrs with follow-up attr having higher
          importance for ordering (e .g. `uattrs=['targets', 'chunks']` would
          order groups first by `chunks` and then by `targets` within each
          chunk).
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
        assert(order in (None, 'uattrs', 'occurrence'))
        self.__order = order


    @borrowdoc(Mapper)
    def __repr__(self, prefixes=[]):
        return super(FxMapper, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['axis', 'fx', 'uattrs'])
            + _repr_attrs(self, ['fxargs'], default=())
            + _repr_attrs(self, ['attrfx'], default='merge')
            + _repr_attrs(self, ['order'], default='uattrs')
            )


    def __str__(self):
        return _str(self, fx=self.__fx.__name__)


    def _train(self, ds):
        # right now it needs no training, if anything is added here make sure to
        # remove is_trained class attribute
        pass

    def __smart_apply_along_axis(self, data):
        # because apply_along_axis could be very much slower than a
        # direct invocation of native functions capable of operating
        # along specific axis, let's make it smarter for those we know
        # could do that.
        fx = None
        naxis = {'samples': 0, 'features': 1}[self.__axis]
        try:
            # if first argument is 'axis' -- just proceed with a native call
            if inspect.getargs(self.__fx.__code__).args[1] == 'axis':
                fx = self.__fx
            elif __debug__:
                debug('FX', "Will apply %s via apply_along_axis",
                          (self.__fx))
        except Exception, e:
            if __debug__:
                debug('FX',
                      "Failed to deduce either %s has 'axis' argument: %s",
                      (self.__fx, repr(e)))
            pass

        if fx is not None:
            if __debug__:
                debug('FX', "Applying %s directly to data giving axis=%d",
                      (self.__fx, naxis))
            mdata = fx(data, naxis, *self.__fxargs)
        else:
            # either failed to deduce signature or just didn't
            # have 'axis' second
            # apply fx along naxis for each sample/feature
            mdata = np.apply_along_axis(self.__fx, naxis, data, *self.__fxargs)
        assert(mdata.ndim in (data.ndim, data.ndim-1))
        return mdata

    @borrowdoc(Mapper)
    def _forward_data(self, data):
        if not self.__uattrs is None:
            raise RuntimeError("%s does not support forward-mapping of plain "
                               "data when data grouping based on attributes "
                               "is requested"
                               % self.__class__.__name__)

        mdata = self.__smart_apply_along_axis(data)

        if self.__axis == 'features':
            if len(mdata.shape) == 1:
                # in case we only have a scalar per sample we need to transpose
                # it properly, to keep the length of the samples axis intact
                mdata = np.atleast_2d(mdata).T
        return np.atleast_2d(mdata)

    @borrowdoc(Mapper)
    def _forward_dataset(self, ds):
        if self.__uattrs is None:
            mdata, sattrs = self._forward_dataset_full(ds)
        else:
            mdata, sattrs = self._forward_dataset_grouped(ds)

        samples = np.atleast_2d(mdata)

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
            incol = ds.sa
            col.set_length_check(samples.shape[0])
        else:
            out = ds.copy(deep=False, fa=[])
            col = out.fa
            incol = ds.fa
            col.set_length_check(samples.shape[1])
        # assign samples to do COW
        out.samples = samples

        for attr in sattrs:
            a = sattrs[attr]
            # TODO -- here might puke if e.g it is a list where some items
            # are empty lists... I guess just wrap in try/except and
            # do dtype=object if catch
            a = np.atleast_1d(a)
            # make sure we do not inflate the number of dimensions for no reason
            # this could happen if there was only one unique value for an
            # attribute and the default 'uniquemerge2literal' attrfx was given
            if len(a.shape) > 1 and a.shape[-1] == 1 and attr in incol \
                    and len(a.shape) > len(incol[attr].value.shape):
                a.shape = a.shape[:-1]
            col[attr] = a

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
        order = self.order
        order_keys = []
        for comb in _orthogonal_permutations(self.__attrcombs):
            selector = reduce(np.multiply,
                                [array_whereequal(col[attr].value, value)
                                 for attr, value in comb.iteritems()])

            # process the samples
            if axis == 0:
                samples = ds.samples[selector]
            else:
                samples = ds.samples[:, selector]

            # check if there were any samples for such a combination,
            # if not -- warning and skip the rest of the loop body
            if not len(samples):
                warning('There were no samples for combination %s. It might be '
                        'a sign of a disbalanced dataset %s.' % (comb, ds))
                continue

            fxed_samples = self.__smart_apply_along_axis(samples)
            mdata.append(fxed_samples)
            if not self.__attrfx is None:
                # and now all samples attributes
                for i, attr in enumerate(col):
                    fxed_attr = self.__attrfx(col[attr].value[selector])
                    attrs[attr].append(fxed_attr)
            # possibly take care about collecting information to have groups ordered
            if order == 'uattrs':
                # reverse order as per docstring -- most of the time we have
                # used uattrs=['targets', 'chunks'] and did expect chunks being
                # groupped together.
                order_keys.append([comb[a] for a in self.__uattrs[::-1]])
            elif order == 'occurrence':
                # First index should be sufficient since we are dealing
                # with unique non-overlapping groups here (AFAIK ;) )
                order_keys.append(np.where(selector)[0][0])

        if order:
            # reorder our groups using collected "order_keys"
            # data
            order_idxs = argsort(order_keys)
            mdata = [mdata[i] for i in order_idxs]
            # and attributes
            attrs = dict((k, [v[i] for i in order_idxs])
                         for k,v in attrs.iteritems())

        if axis == 0:
            mdata = np.vstack(mdata)
        else:
            mdata = np.vstack(np.transpose(mdata))
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

    axis = property(fget=lambda self:self.__axis)
    fx = property(fget=lambda self:self.__fx)
    fxargs = property(fget=lambda self:self.__fxargs)
    uattrs = property(fget=lambda self:self.__uattrs)
    attrfx = property(fget=lambda self:self.__attrfx)
    order = property(fget=lambda self:self.__order)

#
# Convenience functions to create some useful mapper with less complexity
#

def mean_sample(attrfx='merge'):
    """Returns a mapper that computes the mean sample of a dataset.

    Parameters
    ----------
    attrfx : 'merge' or callable, optional
      Callable that is used to determine the sample attributes of the computed
      mean samples. By default this will be a string representation of all
      unique value of a particular attribute in any sample group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('samples', np.mean, attrfx=attrfx)


def mean_group_sample(attrs, attrfx='merge', **kwargs):
    """Returns a mapper that computes the mean samples of unique sample groups.

    The sample groups are identified by the unique combination of all
    values of a set of provided sample attributes.  Order of output
    samples might differ from original and correspond to sorted order
    of corresponding `attrs`  by default.  Use `order='occurrence'` if you would
    like to maintain the order.

    Parameters
    ----------
    attrs : list
      List of sample attributes whose unique values will be used to identify the
      samples groups.
    attrfx : 'merge' or callable, optional
      Callable that is used to determine the sample attributes of the computed
      mean samples. By default this will be a string representation of all
      unique value of a particular attribute in any sample group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('samples', np.mean, uattrs=attrs, attrfx=attrfx, **kwargs)


def sum_sample(attrfx='merge'):
    """Returns a mapper that computes the sum sample of a dataset.

    Parameters
    ----------
    attrfx : 'merge' or callable, optional
      Callable that is used to determine the sample attributes of the computed
      sum samples. By default this will be a string representation of all
      unique value of a particular attribute in any sample group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('samples', np.sum, attrfx=attrfx)


def mean_feature(attrfx='merge'):
    """Returns a mapper that computes the mean feature of a dataset.

    Parameters
    ----------
    attrfx : 'merge' or callable, optional
      Callable that is used to determine the feature attributes of the computed
      mean features. By default this will be a string representation of all
      unique value of a particular attribute in any feature group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('features', np.mean, attrfx=attrfx)


def subtract_mean_feature(attrfx='merge'):
    """Subtract mean of features across samples.
    Functionaly equivalent to MeanRemoval, but much slower.
    """
    return FxMapper('features', subtract_mean, attrfx=attrfx)


def mean_group_feature(attrs, attrfx='merge', **kwargs):
    """Returns a mapper that computes the mean features of unique feature groups.

    The feature groups are identified by the unique combination of all values of
    a set of provided feature attributes.  Order of output
    features might differ from original and correspond to sorted order
    of corresponding `attrs` by default.  Use `order='occurrence'` if you would
    like to maintain the order.

    Parameters
    ----------
    attrs : list
      List of feature attributes whos unique values will be used to identify the
      feature groups.
    attrfx : 'merge' or callable, optional
      Callable that is used to determine the feature attributes of the computed
      mean features. By default this will be a string representation of all
      unique value of a particular attribute in any feature group. If there is
      only a single value in a group it will be used as the new attribute value.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('features', np.mean, uattrs=attrs, attrfx=attrfx, **kwargs)


def absolute_features():
    """Returns a mapper that converts features into absolute values.

    This mapper does not alter any attributes.

    Returns
    -------
    FxMapper instance.
    """
    return FxMapper('features', np.absolute, attrfx=None)


def sumofabs_sample():
    """Returns a mapper that returns the sum of absolute values of all samples.
    """
    return FxMapper('samples', sum_of_abs)

def maxofabs_sample():
    """Returns a mapper that finds max of absolute values of all samples.
    """
    return FxMapper('samples', max_of_abs)


class BinomialProportionCI(Mapper):
    """Compute binomial proportion confidence intervals

    This is a convenience frontend for binomial_proportion_ci_from_bool()
    and supports all methods implemented in this function.

    The confidence interval is computed independently for each feature column.
    The returned dataset contains two samples.  The first one contains the
    lower CI boundary and the second sample the upper boundary.

    Returns
    -------
    dataset
    """

    is_trained = True

    width = Parameter(.95,
                      constraints=EnsureFloat() & EnsureRange(min=0, max=1),
                      doc="Confidence interval width")
    meth = Parameter('jeffreys',
                     constraints=EnsureChoice('wald', 'wilson', 'agresti-coull',
                                              'jeffreys', 'clopper-pearson',
                                              'arc-sine', 'logit', 'anscombe'),
                     doc="Interval estimation method")

    def __init__(self, **kwargs):
        Mapper.__init__(self, **kwargs)

    def _train(self, ds):
        pass

    def _forward_data(self, data):
        from mvpa2.misc.stats import binomial_proportion_ci_from_bool
        return binomial_proportion_ci_from_bool(data, axis=0,
                                                alpha=1-self.params.width,
                                                meth=self.params.meth)

    def _forward_dataset(self, ds):
        msamp = self._forward_data(ds.samples)
        mds = Dataset(msamp, sa=dict(ci_boundary=['lower', 'upper']))
        return mds

#
# Utility functions
#

def _uniquemerge2literal(attrs):
    """Compress a sequence into its unique elements (with string merge).

    Whenever there is more then one unique element in `attrs`, these
    are converted to a string and join with a '+' character inbetween.

    Parameters
    ----------
    attrs : sequence, arbitrary

    Returns
    -------
    Non-sequence arguments are passed as is, otherwise a sequences of unique
    items is. None is returned in case of an empty sequence.
    """
    try:
        if isinstance(attrs[0], basestring):
            # do not try to disassemble sequences of strings
            raise TypeError
        unq = [np.array(u) for u in set([tuple(p) for p in attrs])]
    except TypeError:
        # either no 2d-iterable...
        try:
            unq = np.unique(attrs)
        except TypeError:
            # or no iterable at all -- return the original
            return attrs

    lunq = len(unq)
    if lunq > 1:
        return ['+'.join([str(l) for l in unq])]
    elif lunq:
        return unq
    else:
        return None

def merge2first(attrs):
    """Compress a sequence by discard all but the first element

    This function can be useful as 'attrfx' argument for an FxMapper.

    Parameters
    ----------
    attrs : sequence, arbitrary

    Returns
    -------
    First element of the input sequence.
    """
    return attrs[0]

def argsort(seq, reverse=False):
    """Return indices to get sequence sorted
    """
    # Based on construct from
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    # Thanks!

    # cmp was not passed through since seems to be absent in python3
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)

def _orthogonal_permutations(a_dict):
    """
    Takes a dictionary with lists as values and returns all permutations
    of these list elements in new dicts.

    This function is useful, when a method with several arguments
    shall be tested and all of the arguments can take several values.

    The order is not defined, therefore the elements should be
    orthogonal to each other.

    >>> for i in _orthogonal_permutations({'a': [1,2,3], 'b': [4,5]}):
    ...     print i
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



class BinaryFxNode(Node):
    """Extract a dataset attribute and call a function with it and the samples.

    This node takes a dataset's samples and a configurable attribute and passes
    them to a custom callable. This node can be used to implement comparisons,
    or error quantifications.

    When called with a dataset the node returns a new dataset with the return
    value of the callable as samples.
    """
    # TODO: Allow using feature attributes too
    def __init__(self, fx, space, **kwargs):
        """
        Parameters
        ----------
        fx : callable
          Callable that is passed with the dataset samples as first and
          attribute values as second argument.
        space : str
          name of the sample attribute that contains the target values.
        """
        Node.__init__(self, space=space, **kwargs)
        self.fx = fx


    def _call(self, ds):
        # extract samples and targets and pass them to the errorfx
        targets = ds.sa[self.get_space()].value
        # squeeze to remove bogus dimensions and prevent problems during
        # comparision later on
        values = np.atleast_1d(ds.samples.squeeze())
        if not values.shape == targets.shape:
            # if they have different shape numpy's broadcasting might introduce
            # pointless stuff (compare individual features or yield a single
            # boolean
            raise ValueError("Trying to compute an error between data of "
                             "different shape (%s vs. %s)."
                             % (values.shape, targets.shape))
        err = self.fx(values, targets)
        if np.isscalar(err):
            err = np.array(err, ndmin=2)
        return Dataset(err)


class MeanRemoval(Mapper):
    """Subtract sample mean from features."""

    is_trained = True

    in_place = Parameter(
        False,
        doc="""If False: a copy of the dataset will be made before demeaning.
        If True: demeaning will be performed in-place, i.e. input data is
        modified. This is faster, but can have side-effects when the original
        dataset is used elsewhere again, and implies that floating point data
        types are required to prevent rounding errors in this case.""",
        constraints=EnsureBool())

    def __init__(self, in_place=False, **kwargs):
        Mapper.__init__(self, **kwargs)
        self.in_place = in_place

    def _forward_data(self, data):
        mdata = data
        mean = np.mean(mdata, axis=1)

        if self.in_place:
            if not np.issubdtype(mdata.dtype, float):
                warning("Integer dtype. Mean removal won't work correctly for "
                        "this implementation. Rounding errors will occur. "
                        "Use in_place=False instead")
            mdata -= mean[:, np.newaxis]

        else:
            mdata = mdata - mean[:, np.newaxis]
        return mdata
