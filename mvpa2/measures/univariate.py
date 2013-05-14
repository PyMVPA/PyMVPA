# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Measures whose results are univariate."""

__docformat__ = 'restructuredtext'

import numpy as np
import functools

from mvpa2.base import externals
from mvpa2.base.learner import ChainLearner
from mvpa2.measures.base import FeaturewiseMeasure
from mvpa2.base.dataset import vstack
from mvpa2.datasets.base import Dataset

class CompoundFeaturewiseMeasure(FeaturewiseMeasure):
    '''Compute a summary value for each unique sample attribute value'''
    is_trained = True

    def __init__(self, space='targets', sa_labels=None,
                        summary_func=None, **kwargs):
        '''
        Parameters
        ----------
        space: string
            feature attribute label over which values are summarized
        summary_func: callable
            function defining how a summary is made over multiple values
        sa_labels: list of str
            the values corresponding to space on which the measure is
            to be computed. If None then all values are taken. 
        '''
        if summary_func is None:
            raise ValueError("summary_func has to be callable")

        FeaturewiseMeasure.__init__(self, **kwargs)
        self.space = space
        self._sa_labels = sa_labels
        self._summary_func = summary_func

    def _call(self, dataset):
        targets = dataset.sa[self.space].value
        sa_labels = self._sa_labels
        if sa_labels is None:
            sa_labels = np.unique(targets)

        ns = len(sa_labels)

        # compute for each unique value
        summary_func = self._summary_func
        ys = [summary_func(dataset[sa_label == targets].samples)
                    for sa_label in sa_labels]

        ds = vstack(ys, True)

        # set the labels
        ds.sa[self.space] = sa_labels

        # copy dataset attribtues
        ds.a = dataset.a.copy()

        return ds


def compound_mean_measure(space='targets', sa_labels=None, **kwargs):
    '''Returns a measure that computes the mean for each unique
    value in sample attributes over all features
    '''
    summary_func = np.mean
    return CompoundFeaturewiseMeasure(space=space,
                                      sa_labels=sa_labels,
                                      summary_func=summary_func,
                                      **kwargs)

def compound_univariate_mean_measure(space='targets', sa_labels=None, **kwargs):
    '''Returns a measure that computes the mean for each unique
    value in sample attributes, for each feature seperately.
    '''
    summary_func = lambda x:np.mean(x, axis=0)
    return CompoundFeaturewiseMeasure(space=space,
                                      sa_labels=sa_labels,
                                      summary_func=summary_func,
                                      **kwargs)

class WinnerTakeAllMeasure(FeaturewiseMeasure):
    '''Finds for each feature the sample with the highest value'''
    is_trained = True
    def __init__(self, sign=1, **kwargs):
        '''
        Parameters
        ----------
        sign: -1 or 1
            whether to look for maxima (sign=1) or minima (sign=-1)
        '''
        FeaturewiseMeasure.__init__(self, **kwargs)
        self._sign = sign

    def _call(self, ds):
        '''
        Parameters
        ----------
        ds: Dataset
            input dataset 
        
        Returns
        -------
        wta: Dataset
            Result with one sample and an equal number of features
            as the input dataset, where wta.samples[k]==v means that
            max(ds.samples[:,k])==ds.samples[v,k]. (or min if sign==-1).
            Every sample attribute is renamed and stored as a feature 
            attribute in wta with prefix 'wta_' (sign==1) or 'lta_' (sign==-1).
        '''

        sign = self._sign

        sign2prefix = {-1:'lta_', 1:'wta_'}
        if not sign in sign2prefix:
            raise KeyError('Sign %s should be in %s' % (sign, sign2prefix.keys()))

        nf = ds.nfeatures
        samples = np.argmax(sign * ds.samples, axis=0)

        wta = Dataset(np.reshape(samples, (1, nf)))
        wta.fa = ds.fa.copy()
        wta.a = ds.a.copy()

        prefix = sign2prefix[sign]
        for k, v in ds.sa.iteritems():
            prek = prefix + k

            if prek in wta.fa:
                raise KeyError("Key clash: %s already in feature attributes"
                                    % (prek))

            wta.fa[prek] = v.value[samples]

        return wta


def compound_winner_take_all_measure(sign=1, space='targets', sa_labels=None,
                                        **kwargs):
    '''Computes the maximum mean value for each unique value in a sample 
    attribute'''
    meaner = compound_univariate_mean_measure(space=space, sa_labels=sa_labels, **kwargs)
    winner = WinnerTakeAllMeasure(sign=sign, **kwargs)
    return ChainLearner((meaner, winner))
