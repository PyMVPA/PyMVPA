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
import numpy as np
from scipy.ndimage import measurements
import statsmodels.stats.multitest as smm
from mvpa2.mappers.base import IdentityMapper

# TODO: remove
def make_file_list(path, subj_names):
    """
    randomly choose one file for every subject from subj_names and return it's
    addresses as a list

    all the files need to be in same folder in name format sub+subj_name*.npy

    """

#        directories = [f for f in os.listdir('.') if os.path.isdir(f)
#                     and f.startswith('sub')]
#        files = []
#        for directory in directories:
#    #        print directory
#    #        print random.choice(os.listdir(directory))
#            files.append(os.path.join(directory,
#                                      random.choice(os.listdir(directory))))
#        return files

    files = []
#    path = './bootstrapped/group_space/'
    for subj in subj_names:
        if not isinstance(subj, basestring):
            raise AssertionError()

        files.append(random.choice([path + f for f in os.listdir(path)
                if f.startswith('sub%s' % subj) and f.endswith('.npy')]))
    return files


# TODO: refactor to use non-file data source
def get_bootstrapped_map(path, subj_names):
    """
    create one bootstrapped map by randomly taking one map for every subject
    and average them voxel-wise

    all the files need to be in same folder in name format sub+subj_name+*.npy

    path: path to the directory of the files
    subj_names: names of the subjects to create bootstrapped map from
    """
    def nonzero_mean(M):   # quick fix for the 'hole in the middle of the head'
        def my_func(arr):  # problem we have, should it stay here in the future?
            if sum(arr != 0) == 0:
                return 0
            else:
                return np.mean(arr[arr != 0])
        return np.apply_along_axis(my_func, 0, M)

    file_list = make_file_list(path, subj_names)
    # this takes forever, make smarter algorithm, maybe?
    return nonzero_mean([np.load(file_).flatten()
                    for file_ in file_list])


# TODO: refactor to return a dataset
def create_bootstr_M(n, path, subj_names):
    """
    create matrix of n bootstrapped maps
    """
    bootstr_M = [get_bootstrapped_map(path, subj_names) for _ in range(n)]
    bootstr_M = np.vstack(bootstr_M)
    return bootstr_M


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


def threshold(M, thresholding_map):
    """Threshold map with a feature-wise map of thresholds.

    Parameters
    ----------
    M : array
      To be thresholded input.
    thresholding_map : array
      Array with threshold map. Needs to be of the same shape as ``M``.

    Returns
    -------
    array
      Binary map indicating the location of super-threshold value.
    """
    thresholded_M = M > thresholding_map
    # XXX: MH disabled this, because he did not understand why it was necessary
    # the tests still pass. Please check and remove comment.
    #thresholded_M = thresholded_M.astype(int)
    return(thresholded_M)


def _get_map_cluster_sizes(map_):
    labels, num = measurements.label(map_)
    area = measurements.sum(map_, labels, index=np.arange(labels.max() + 1))
    return area.astype(int)[1:]  # delete cluster of size 0


def get_cluster_sizes(ds, cluster_sizes=None):
    """Computer cluster sizes from all samples in a boolean dataset.

    Individually for each sample, in the input dataset, clusters of non-zero
    values will be determined after reverse-applying any transformation of the
    dataset's mapper (if any).

    Parameters
    ----------
    ds : dataset or array
      Typically a dataset with boolean samples.
    cluster_sizes : list or None
      If not None, the given list is extended with the cluster sizes computed
      from the present input dataset. Otherwise, a new list is generated.

    Returns
    -------
    list
      Unsorted list of cluster sizes from all samples in the input dataset
      (optionally appended to any values passed via ``cluster_sizes``).
    """
    if cluster_sizes is None:
        cluster_sizes = []

    mapper = IdentityMapper()
    data = np.asanyarray(ds)
    if hasattr(ds, 'a') and 'mapper' in ds.a:
        mapper = ds.a.mapper

    for i in xrange(len(ds)):
        osamp = mapper.reverse1(data[i])
        m_clusters = _get_map_cluster_sizes(osamp)
        cluster_sizes.extend(m_clusters)
    return cluster_sizes


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
