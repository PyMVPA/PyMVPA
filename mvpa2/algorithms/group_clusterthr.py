# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Cluster thresholding algorithm for accuracy maps after Stelzer et al.

Johannes Stelzer, Yi Chen and Turner (2013). Statistical inference and multiple
testing correction in classification-based multi-voxel pattern analysis (MVPA):
Random permutations and cluster size control. NeuroImage, 65, 69--82.
"""

__docformat__ = 'restructuredtext'

import os
import random
import bisect
from collections import Counter

import numpy as np

from scipy.ndimage import measurements
from scipy.sparse import dok_matrix

from mvpa2.mappers.base import IdentityMapper
from mvpa2.datasets import Dataset
from mvpa2.base.learner import Learner
from mvpa2.base.param import Parameter
from mvpa2.base.constraints import \
        EnsureInt, EnsureFloat, EnsureRange, EnsureChoice

class ACCClusterThreshold(Learner):
    """ WAIT FOR ME """

    n_bootstrap = Parameter(100000, constraints=EnsureInt() & EnsureRange(min=1),
            doc="")

    feprob = Parameter(0.001,
            constraints=EnsureFloat() & EnsureRange(min=0.0, max=1.0),
            doc="")

    chunks_attr = Parameter('chunks',
            doc="")

    fwe_rate = Parameter(0.05,
            constraints=EnsureFloat() & EnsureRange(min=0.0, max=1.0),
            doc="")

    multicomp_correction = Parameter('fdr_bh',
             constraints=EnsureChoice('bonferroni', 'sidak', 'holm-sidak',
                                      'holm', 'simes-hochberg', 'hommel',
                                      'fdr_bh', 'fdr_by'),
             doc="")

    n_blocks = Parameter(1, constraints=EnsureInt() & EnsureRange(min=1),
             doc="")

    def __init__(self, **kwargs):
        # force disable auto-train: would make no sense
        Learner.__init__(self, auto_train=False, **kwargs)
        self.untrain()

    def _untrain(self):
        self._thrmap = None
        self._null_cluster_sizes = None

    def _train(self, ds):
        # shortcuts
        chunks_attr = self.params.chunks_attr
        #
        # Step 0: bootstrap maps by drawing one for each chunk and average them
        # (do N iterations)
        # this could take a lot of memory, hence instead of computing the maps
        # we compute the source maps they can be computed from and then (re)build
        # the matrix of bootstrapped maps either row-wise or column-wise (as
        # needed) to save memory by a factor of (close to) `n_bootstrap`
        # which samples belong to which chunk
        chunk_samples = dict([(c, np.where(ds.sa[chunks_attr].value == c)[0])
                                    for c in ds.sa[chunks_attr].unique])
        # pre-built the bootstrap combinations
        bcombos = [[random.sample(v, 1)[0] for v in chunk_samples.values()]
                        for i in xrange(self.params.n_bootstrap)]
        bcombos = np.array(bcombos, dtype=int)
        # TODO implement
        # TODO implement parallel procedure as the estimation is independent
        # across features
        #
        # Step 1: find the per-voxel threshold that corresponds to some p
        # in the NULL
        thrsegs = []
        segwidth = ds.nfeatures/self.params.n_blocks
        ds_samples = ds.samples
        #for segstart in xrange(0, ds.nfeatures, segwidth):
            # get a view to a subset of the features -- should be efficient
            #seg_samples = ds_samples[:, segstart:segstart + segwidth]
            # compute average bootstrapped maps using the stored bcombos
        thrmap = np.hstack( # merge across compute blocks
                [get_thresholding_map(
                    # one average map for every stored bcombo
                    # this also slices the input data into feature subsets
                    # for the compute blocks
                    [np.mean(
                        # get a view to a subset of the features
                        # -- should be somewhat efficient as feature axis is
                        # sliced
                        ds_samples[sidx, segstart:segstart + segwidth],
                        axis=0)
                            for sidx in bcombos],
                    self.params.feprob)
                        # compute a partial threshold map for as mabny features
                        # as fit into a compute block
                        for segstart in xrange(0, ds.nfeatures, segwidth)])
        # store for later thresholding of input data
        self._thrmap = thrmap
        #
        # Step 2: threshold all NULL maps and build distribution of NULL cluster
        #         sizes
        #
        cluster_sizes = Counter()
        # recompute the bootstrap average maps to threshold them and determine
        # cluster sizes
        if 'mapper' in ds.a:
            dsa = dict(mapper=ds.a.mapper)
        else:
            dsa = {}
        # this step can be computed in parallel chunks to speeds things up
        for sidx in bcombos:
            avgmap = np.mean(ds_samples[sidx], axis=0)[None]
            # apply threshold
            clustermap = avgmap > thrmap
            # wrap into a throw-away dataset to get the reverse mapping right
            bds = Dataset(clustermap, a=dsa)
            # this function reverse-maps every sample one-by-one, hence no need
            # to collect chunks of bootstrapped maps
            cluster_sizes = get_cluster_sizes(bds, cluster_sizes)
        # store cluster size histogram for later p-value evaluation
        # use a sparse matrix for easy consumption (max dim is the number of
        # features, i.e. biggest possible cluster)
        scl = dok_matrix((ds.nfeatures, 1), dtype=int)
        for s in cluster_sizes:
            scl[s,0] = cluster_sizes[s]
        self._null_cluster_sizes = scl

    def _call(self, ds):
        if len(ds) > 1:
            raise ValueError("cannot handle more than one sample per dataset")
        # threshold input
        thrd = ds.samples > self._thrmap
        # mapper default
        mapper = IdentityMapper()
        # overwrite if possible
        if hasattr(ds, 'a') and 'mapper' in ds.a:
            mapper = ds.a.mapper
        # reverse-map input
        osamp = mapper.reverse1(thrd[0])
        # prep output dataset
        outds = ds.copy(deep=False)
        # determine clusters
        labels, num = measurements.label(osamp)
        outds.fa['clusters_voxelwise_thresh'] = labels
        area = measurements.sum(thrd,
                                labels,
                                index=np.arange(1, num + 1)).astype(int)
        # update cluster size histogram with the actual result to get a
        # proper lower bound for p-values
        # this will make a copy, because the original matrix is int
        histogrm = self._null_cluster_sizes.astype('float')
        for a in area:
            histogrm[a,0] += 1
        # normalize histogram
        histogrm /= histogrm.sum()
        # compute p-values for each cluster
        cache = {}
        cluster_prob_raw = {}
        for cidx, csize in enumerate(area):
            # try the cache
            prob = cache.get(csize, None)
            if prob is None:
                # no cache
                # probability is the sum of a relative frequencies for clusters
                # larger OR EQUAL than the current one
                prob = histogrm[csize:].sum()
                cache[csize] = prob
            # store for output
            cluster_prob_raw[cidx + 1] = prob
        outds.a['cluster_probs_uncorrected'] = cluster_prob_raw

        # convert pvals into a simple sequence to ensure order
        probs = [cluster_prob_raw[i] for i in xrange(1, num + 1)]
        if self.params.multicomp_correction is None:
            probs_corr = np.array(pvals)
            rej = probs_corr <= self.params.fwe_rate
        else:
            # do a local import as only this tiny portion needs statsmodels
            import statsmodels.stats.multitest as smm
            rej, probs_corr = smm.multipletests(
                                probs,
                                alpha=self.params.fwe_rate,
                                method=self.params.multicomp_correction)[:2]
        # store corrected per-cluster probabilities
        outds.a['cluster_probs_fwe_corrected'] = \
                dict([(i + 1, probs_corr[i]) for i in xrange(num)])
        # remove cluster labels that did not pass the FWE threshold
        for i, r in enumerate(rej):
            if not r:
                labels[labels == i + 1] = 0
        outds.fa['clusters_fwe_thresh'] = labels
        return outds


def get_thresholding_map(data, p=0.001):
    """Return array of thresholds corresponding to a probability of such value in the input

    Thresholds are returned as an array with one value per column in the input
    data.

    Parameters
    ----------
    data : 2D-array
      Array with data on which the cumulative distribution is based.
      Values in each column are sorted and the value corresponding to the
      desired probability is returned.
    p : float [0,1]
      Value greater or equal than the returned threshold have a probability `p` or less.
    """
    # we need NumPy indexing logic, even if a dataset comes in
    data = np.asanyarray(data)
    p_index = int(len(data) * p)
    if p_index < 1:
        raise ValueError("requested probability is too low for the given number of samples")
    # threshold indices are all in one row of the argsorted inputs
    thridx = np.argsort(data, axis=0, kind='quicksort')[-p_index]
    return data[thridx, np.arange(data.shape[1])]


def _get_map_cluster_sizes(map_):
    labels, num = measurements.label(map_)
    area = measurements.sum(map_, labels, index=np.arange(1, num + 1))
    if not len(area):
        return [0]
    else:
        return area.astype(int)


def get_cluster_sizes(ds, cluster_counter=None):
    """Computer cluster sizes from all samples in a boolean dataset.

    Individually for each sample, in the input dataset, clusters of non-zero
    values will be determined after reverse-applying any transformation of the
    dataset's mapper (if any).

    Parameters
    ----------
    ds : dataset or array
      A dataset with boolean samples.
    cluster_counter : list or None
      If not None, the given list is extended with the cluster sizes computed
      from the present input dataset. Otherwise, a new list is generated.

    Returns
    -------
    list
      Unsorted list of cluster sizes from all samples in the input dataset
      (optionally appended to any values passed via ``cluster_counter``).
    """
    # XXX input needs to be boolean for the cluster size calculation to work
    if cluster_counter is None:
        cluster_counter = Counter()

    mapper = IdentityMapper()
    data = np.asanyarray(ds)
    if hasattr(ds, 'a') and 'mapper' in ds.a:
        mapper = ds.a.mapper

    for i in xrange(len(ds)):
        osamp = mapper.reverse1(data[i])
        m_clusters = _get_map_cluster_sizes(osamp)
        cluster_counter.update(m_clusters)
    return cluster_counter


def get_pval(x, null_dist, sort=True):
    if sort is True:
        null_dist = sorted(null_dist)
    return 1 - (bisect.bisect_left(null_dist, x) / float(len(null_dist)))


def transform_to_pvals(arr, null_dist):
    """"
    will get p value for every element in the array based on null distribution
    """
    null_dist = sorted(null_dist)
    p_vals = [get_pval(e, null_dist, False) for e in arr]
    return p_vals


def label_clusters(null_dist_clusters, thresholded_orig_map,
                   alpha=0.05, method='fdr_i', return_type='binary_map'):

    """
    will label clusters in 3d thresholded map based on their corrected pvalue
    computed against distribution of the clusters in maps created by
    permutation. It will add clusters from original map to the null
    distribution

    alpha: rejection level
    method: method of multiple comparison correction of
    statsmodels.stats.multitest or "None" for no correction

    return_type: how will be clusters labeled in the returned 3d map
    return_type = 'binary_map': clusters that are over the threshold are
    labeled as 1, everything else is 0
    return_type = 'p_vals': clusters are labeled 1-cluster p value
    return_type = 'thresholded_p_vals': clusters that are over the threshold
    are labeled 1-cluster p value, everything else is 0
    return_type = 'unique_clusters': every cluster will be labeled by unique
    value
    return_type = 'cluster_sizes': clusters are labeled by their size
    """
    orig_clusters = _get_map_cluster_sizes(thresholded_orig_map)
    # add clusters from the original map to null dist, important for p val
    null_dist_clusters = np.hstack([null_dist_clusters, orig_clusters])
    null_dist_clusters = np.sort(null_dist_clusters)
    pval_clusters = transform_to_pvals(orig_clusters, null_dist_clusters)

    if method == "None":
        pval_corr = np.array(pval_clusters)
        rej = pval_corr < alpha
    else:
        rej, pval_corr = smm.multipletests(pval_clusters, alpha=alpha,
                                           method=method)[:2]
    pval_corr = 1 - pval_corr  # 1 - p value for visualization purposes

    pval_corr = np.hstack([[0], pval_corr])  # will add cluster of size zero,
    rej = np.hstack([[0], rej])  # that was deleted in get_map_cluster_sizes
    rej = rej.astype(int)
    labels, num = measurements.label(thresholded_orig_map)

    if return_type == 'binary_map':
        areaImg = rej[labels]
    elif return_type == 'thresholded_p_vals':
        pval_corr = np.array([0 if rejc == 0 else pval
                             for rejc, pval in zip(rej, pval_corr)])
        areaImg = pval_corr[labels]
    elif return_type == 'p_vals':
        areaImg = pval_corr[labels]

    elif return_type == 'unique_clusters':
        areaImg = labels

    elif return_type == 'cluster_sizes':
        areaImg = np.hstack([[0], orig_clusters])[labels]
    else:
        raise AssertionError("Unrecognized return_type")
    return areaImg



####Create bootstrapped maps
## to start we need to have our 'permuted maps' ready, Maps that were created
## by permuting the targets. We should have 100 for every subject.
## From those maps we will create 100000 'bootstrapped maps' one bootstrapped map
## is mean of one randomly chosen permuted map from every subject
## here I am temporarily saving permuted maps, but that step can be skipped
## and just thresholded binary maps can be saved, therefore it will eat less
## space on the disk. However, creating permuted maps is done for every map
## with random element involved and thresholding is done voxel-wise and it will
## not be possible to load whole dataset to threshold, but it can be done if
## the permuted maps are already split, so it's not necessary to load all
## the data, but only data from some voxels. Since bootstrapped maps are created
## randomly it will be necessary to use random seed, so all splits belong to
## same maps

#perm = sys.argv[1]
##perm = 1
#path = './bootstrapped/group_space/'
## we will create matrix of 500 maps here. In the end we need 100000 maps.
#M = create_bootstr_M(500, path, ['001', '002', '003', '004', '005', '006', '007',
#                               '008', '009', '010', '011', '012', '013', '014',
#                               '015', '016', '017', '018', '019', '020'])
#
## splitting for memory reasons, next steps will be done voxel-wise, this one
## is done 'map vise.' We will later hstack and vstack those splits
#M = np.split(M, (range(20000,600000,20000)), axis=1)


#for i in range(len(M)):
#    np.save("./bootstrapped/bootstrapped/M_perm"+str(perm)+"_split"+str(i), M[i])
#print "Done: M_perm%s.npy" % i



####Create thresholding_map
## now we will find a threshold value, specified by p value for every voxel
## this is done voxel-wise, so we will vertically stack matrices that we created
## in previous step.

#split = sys.argv[1]
##split = 1
#M = np.vstack([np.load("./bootstrapped/bootstrapped/M_perm%s_split%s.npy" % (i,
#               split)) for i in range(200)])
#thresholding_map = get_thresholding_map(M, 0.0005)
#np.save('thresholding_map_p0005split%s.npy' % split, thresholding_map)
#print "done"



#### create thresholded orig map
#thresholding_map = np.hstack([np.load('thresholding_map_p001split%s.npy' %i)
#                              for i in range(0,30)])
#subjects = range(1,20)
#subjects = [('00' + str(x))[-3:] for x in subjects]
#orig_map = np.mean([np.load('./bootstrapped/group_space/sub%s_perm000_gs.npy' % subj)
#                    for subj in subjects], axis=0)
#thresholded_orig_map = threshold(orig_map, thresholding_map)
#thresholded_orig_map = unmask(thresholded_orig_map,
#                              np.load('./bootstrapped/nonzero_mask.npy'),
#                              nib.load('./bootstrapped/temp001.nii.gz'))
###save as nifty
##affine = nib.load('./bootstrapped/temp001.nii.gz').get_affine()
##image_3d = nib.Nifti1Image(thresholded_orig_map, affine)
##nib.save(image_3d, 'thresholded_orig_map.nii.gz')



#### get null distribution clusters
## now we will count clusters in every one of the 100000 maps,
#x = int(sys.argv[1])  # 0-10
#y = (x+1)*20 # because there are 200 matrices of 500 maps (200*500=100000)
#
#null_dist_clusters = []
##for perm_n in range(y-20,y):
#for perm_n in range(1):
#    M = np.hstack([np.load('./bootstrapped/bootstrapped/M_perm%s_split%s.npy'
#                   % (perm_n, i)) for i in range(30)])
#    null_dist_clusters.append(get_null_dist_clusters(M,
#                                    np.load('./bootstrapped/nonzero_mask.npy'),
#                                    (132, 175, 48),
#                                    thresholding_map))
#null_dist_clusters = np.hstack(null_dist_clusters)
#np.save('clusters%s.npy' % x, null_dist_clusters)



#### label clusters
## in our thresholded original map, we will compute p value for clusters and
## label them
#null_dist_clusters = np.hstack([np.load('clusters%s.npy' % i)
#                                for i in range(10)])
#p_val_map = label_clusters(null_dist_clusters, thresholded_orig_map,
#                           return_type='p_vals'
##                           return_type='thresholded_p_vals'
##                           return_type='binary_map'
##                           return_type='unique_clusters'
##                            return_type='cluster_sizes'
#                            )
#
#affine = nib.load('./bootstrapped/temp001.nii.gz').get_affine()
#image_3d = nib.Nifti1Image(p_val_map, affine)
#nib.save(image_3d, 'pval_map.nii.gz')


#### print clusters p values to console, just for curiosity

#pval_clusters = transform_to_pvals(orig_clusters, null_clusters)
#rej, pval_corr = smm.multipletests(pval_clusters, alpha=0.05,
#                                    method='fdr_i')[:2]
#x = zip(orig_clusters, pval_clusters, pval_corr, rej)
#x = list(set(x))
#x = sorted(x)
#
#for e in x[-25:]:
#    print e[0], 1-e[1], 1-e[2], e[3]
#print mean(rej)
