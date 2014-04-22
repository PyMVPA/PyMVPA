# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Stability of labels across chunks based on correlation."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.measures.base import FeaturewiseMeasure

class CorrStability(FeaturewiseMeasure):
    """Correlation of a feature values per each target across chunks.

    It will assesses feature stability across runs for each unique label by
    correlating feature values across all labels for pairwise combinations of
    the chunks.

    If there are multiple samples with the same label in a single
    chunk (as is typically the case) this algorithm will take the
    featurewise average of the sample activations to get a single
    value per label, per chunk.

    """

    is_trained = True

    def __init__(self, space='targets', **kwargs):
        """Initialize

        Parameters
        ----------
        space : str
          What samples attribute to use as targets (labels).
        """
        # init base classes first
        FeaturewiseMeasure.__init__(self, space=space, **kwargs)

    def _call(self, dataset):
        """Computes feature-wise scores."""

        # get the attributes (usually the labels) and the samples
        attrdata = dataset.sa[self.space].value
        samples = dataset.samples

        # take mean within chunks
        dat = []
        labels = []
        chunks = []
        for c in dataset.uniquechunks:
            for l in np.unique(attrdata):
                ind = (dataset.chunks==c)&(attrdata==l)
                if ind.sum() == 0:
                    # no instances, so skip
                    continue
                # append the mean, and the label/chunk info
                dat.append(np.mean(samples[ind, :], axis=0))
                labels.append(l)
                chunks.append(c)

        # convert to arrays
        dat = np.asarray(dat)
        labels = np.asarray(labels)
        chunks = np.asarray(chunks)

        # get indices for correlation (all pairwise values across
        # chunks)
        ind1 = []
        ind2 = []
        for i,c1 in enumerate(np.unique(chunks)[:-1]):
            for c2 in np.unique(chunks)[i+1:]:
                for l in np.unique(labels):
                    v1 = np.where((chunks==c1)&(labels==l))[0]
                    v2 = np.where((chunks==c2)&(labels==l))[0]
                    if labels[v1] == labels[v2]:
                        # the labels match, so add them
                        ind1.extend(v1)
                        ind2.extend(v2)
        
        # convert the indices to arrays
        ind1 = np.asarray(ind1)
        ind2 = np.asarray(ind2)

        # remove the mean from the datasets
        dat1 = dat[ind1] - dat[ind1].mean(0)[np.newaxis]
        dat2 = dat[ind2] - dat[ind2].mean(0)[np.newaxis]

        # calculate the correlation from the covariance and std
        covar = (dat1*dat2).mean(0) / (dat1.std(0) * dat2.std(0))
        covar[np.isnan(covar)] = 0.0 # reset nan's to 0s
        return covar
