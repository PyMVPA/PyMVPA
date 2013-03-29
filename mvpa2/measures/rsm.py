# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Similarity measure correlation computed over all
subjects
"""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.measures.base import Measure
from mvpa2.misc.stats import DSMatrix
from mvpa2.datasets.base import Dataset
import copy

class RSMMeasure(Measure):
    """RSMMeasure creates a DatasetMeasure object
       where metric can be one of 'euclidean', 'spearman', 'pearson'
       or 'confusion' and nsubjs has to be number of subjects
       and compare_ave flag determines whether to correlate with
       average of all other subjects or just one-to-one
       k should be 0 to n 0-to use DSM including diagonal 
       """

    def __init__(self, dset_metric, nsubjs, compare_ave, k, **kwargs):
        Measure.__init__(self, **kwargs)

        self.dset_metric = dset_metric
        self.dset_dsm = []
        self.nsubjs = nsubjs
        self.compare_ave = compare_ave
        self.k = k


    def __call__(self, dataset):
        dsm_all = []
        rsm_all = []
        nsubjs = self.nsubjs
        # create the dissimilarity matrix for each subject's data in the input dataset
        ''' TODO: How to handle Nan? should we uncomment the workarounds?
        '''
        for i in xrange(nsubjs):
            if self.dset_metric == 'pearson':
                self.dset_dsm = np.corrcoef(dataset.samples[i * dataset.nsamples / nsubjs:((i + 1) * dataset.nsamples / nsubjs), :])
            else:
                self.dset_dsm = DSMatrix(dataset.samples[i * dataset.nsamples / nsubjs:((i + 1) * dataset.nsamples / nsubjs), :], self.dset_metric)
                self.dset_dsm = self.dset_dsm.full_matrix
            orig_dsmatrix = copy.deepcopy(np.matrix(self.dset_dsm))
            #orig_dsmatrix[np.isnan(orig_dsmatrix)] = 0
            #orig_dsmatrix[orig_dsmatrix == 0] = -2
            #orig_tri = np.triu(orig_dsmatrix, k=self.k)
            #vector_form = orig_tri[abs(orig_tri) > 0]
            #vector_form[vector_form == -2] = 0
            vector_form = orig_dsmatrix[np.tri(len(orig_dsmatrix), k= -1 * self.k, dtype=bool)]
            vector_form = np.asarray(vector_form)
            dset_vec = vector_form[0]
            dsm_all.append(dset_vec)
        dsm_all = np.vstack(dsm_all)
        #print dsm_all.shape
        if self.compare_ave:
            for i in xrange(nsubjs):
                dsm_temp = nsubjs * np.mean(dsm_all, axis=0) - dsm_all[i, :]
                rsm = np.corrcoef(dsm_temp, dsm_all[i, :])
                rsm_all.append(rsm[0, 1])
        else:
            rsm = np.corrcoef(dsm_all)
            rsm = np.matrix(rsm)
            #rsm[np.isnan(rsm)] = 0
            #rsm[rsm == 0] = -2
            #rsm = np.triu(rsm, k=1)
            #rsm = rsm[abs(rsm) > 0]
            #rsm[rsm == -2] = 0
            rsm = rsm[np.tri(len(rsm), k= -1, dtype=bool)]
            rsm = np.asarray(rsm)
            rsm_all = rsm[0]

        return Dataset(rsm_all)

class RSM_Correlation_Measure(Measure):
    is_trained = True
    def __init__(self, metric='pearson', space='targets', **kwargs):
        Measure.__init__(self, **kwargs)
        self.metric = metric
        self.space = space

    def _call(self, dataset):
        # compute rsm 
        dsm = DSMatrix(dataset.samples, metric=self.metric)
        vec_form = samples = dsm.get_vector_form()

        n = dsm.full_matrix.shape[0]

        # mask for upper diagonal
        msk = np.asarray([[i > j for i in xrange(n)] for j in xrange(n)])

        # compute z scores        
        arr = np.reshape(np.asarray(dsm.full_matrix[msk]), (-1,))
        arr_z = (arr - np.mean(arr)) / np.std(arr)

        # set the space
        ds = Dataset(samples=arr_z)
        if not self.space is None:
            space_vals = dataset.sa[self.space].value
            labels = ['%s-%s' % (space_vals[i], space_vals[j])
                        for i in xrange(n) for j in xrange(n) if msk[i, j]]
            ds.sa[self.space] = labels

        return ds
