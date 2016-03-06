# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Data aggregation procedures"""

__docformat__ = 'restructuredtext'

import numpy as np
from functools import partial

from mvpa2.base import externals
from mvpa2.base.learner import ChainLearner
from mvpa2.measures.base import Measure
from mvpa2.base.dataset import vstack
from mvpa2.datasets.base import Dataset
from mvpa2.mappers.fx import mean_group_sample
from mvpa2.base.node import ChainNode
import copy
from mvpa2.base.dochelpers import _repr_attrs

class WinnerMeasure(Measure):
    '''Select a "winning" element along samples or features.

    Given the specification would return a Dataset with a single sample
    (or feature).
    '''
    is_trained = True
    def __init__(self, axis, fx, other_axis_prefix=None, **kwargs):
        '''
        Parameters
        ----------
        axis: str or int
            'samples' (or 0) or 'features' (or 1).
        fx: callable
            function to determine the winner. When called with a dataset ds,
            it should return a vector with ds.nsamples values 
            (if axis=='features') or ds.nfeatures values (if axis=='samples').  
        other_axis_prefix: str
            prefix used for feature or sample attributes set on the other axis.
        '''
        Measure.__init__(self, **kwargs)
        if type(axis) is str:
            str2num = dict(samples=0, features=1)
            if not axis in str2num:
                raise ValueError("Illegal axis: should be %s" %
                                        ' or '.join(str2num))
            axis = str2num[axis]

        elif not axis in (0, 1):
            raise ValueError("Illegal axis: should be 0 or 1")

        self.__axis = axis
        self.__fx = fx
        self.__other_axis_prefix = other_axis_prefix

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        prefixes_ = ['axis=%r,fx=%r,other_axis_prefix=%r' % (
                        self.__axis, self.__fx, self.__other_axis_prefix)]
        return "%s(%s)" % (self.__class__.__name__, ','.join(prefixes_))

    def _call(self, ds):
        '''
        Parameters
        ----------
        ds: Dataset
            input dataset 
        
        Returns
        -------
        wds: Dataset
            Result with one sample (if axis=='feature') or one feature (if 
            axis=='samples') and an equal number of features (or samples,
            respectively) as the input dataset.
        '''

        axis = self.__axis
        fx = self.__fx

        # ensure it's a dataset
        if not isinstance(ds, Dataset):
            ds = Dataset(ds)

        samples = ds.samples

        # apply the function
        winners = fx(ds)

        # set the new shape
        new_shape = list(ds.shape)
        new_shape[axis] = 1

        # the output dataset
        wta = Dataset(np.reshape(winners, new_shape))

        # copy dataset attributes
        wta.a = ds.a.copy()

        # copy feature attribute and set sample attributes, or vice versa
        fas = [ds.fa, wta.fa]
        sas = [ds.sa, wta.sa]
        fas_sas = [fas, sas]
        to_copy, to_leave = [fas_sas[(i + axis) % 2] for i in xrange(2)]

        # copy each attribute
        for k, v in to_copy[0].iteritems():
            to_copy[1][k] = copy.copy(v)

        # set source and target. feature attributes become
        # sample attributes; or vice versa
        src, _ = to_leave
        trg = to_copy[1]
        prefix = self.__other_axis_prefix
        for k, v in src.iteritems():
            # set the prefix
            prek = ('' if prefix is None else prefix) + k

            if prek in trg:
                raise KeyError("Key clash: %s already in %s"
                                    % (prek, to_copy[1]))
            trg[prek] = v.value[winners]

        return wta

def feature_winner_measure():
    '''takes winner over features'''
    return WinnerMeasure('features', partial(np.argmax, axis=1), 'wta_')

def feature_loser_measure():
    '''takes loser over features'''
    return WinnerMeasure('features', partial(np.argmin, axis=1), 'lta_')

def sample_winner_measure():
    '''takes winner over samples'''
    return WinnerMeasure('samples', partial(np.argmax, axis=0), 'wta_')

def sample_loser_measure():
    '''takes loser over samples'''
    return WinnerMeasure('samples', partial(np.argmin, axis=0), 'lta_')

def group_sample_winner_measure(attrs=('targets',)):
    '''takes winner after meaning over attrs'''
    return ChainNode((mean_group_sample(attrs), sample_winner_measure()))

def group_sample_loser_measure(attrs=('targets',)):
    '''takes loser after meaning over attrs'''
    return ChainNode((mean_group_sample(attrs), sample_loser_measure()))



if __name__ == '__main__':
    ns = 4
    nf = 3
    n = ns * nf
    ds = Dataset(np.reshape(np.mod(np.arange(0, n * 5, 5) + .5 * n, n), (ns, nf)),
                 sa=dict(targets=[0, 0, 1, 1], x=[3, 2, 1, 0]),
                 fa=dict(v=[3, 2, 1], w=['a', 'b', 'c']))

    measures2out = {feature_winner_measure : [1, 0, 2, 1],
                    feature_loser_measure: [2, 1, 0, 2],
                    sample_winner_measure: [1, 0, 2],
                    sample_loser_measure:[2, 1, 3],
                    group_sample_winner_measure:[0, 0, 0],
                    group_sample_loser_measure: [1, 0, 0]}

    for m, out in measures2out.iteritems():
        print np.all(m()(ds).samples.ravel() == np.asarray(out))

