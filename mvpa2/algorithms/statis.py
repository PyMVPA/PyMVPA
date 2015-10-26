# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Transformation of individual feature spaces into a compromise common space

See :class:`Statis` for more information.

"""

import numpy as np
from mvpa2.base.constraints import EnsureInt, EnsureRange
from mvpa2.base.types import is_datasetlike
from mvpa2.base.state import ClassWithCollections
from mvpa2.mappers.zscore import ZScoreMapper
from mvpa2.mappers.staticprojection import StaticProjectionMapper
from mvpa2.base.dataset import hstack
from mvpa2.base.param import Parameter
from mvpa2.datasets.base import Dataset
from mvpa2.base import warning
if __debug__:
    from mvpa2.base import debug

# make only main utility visible outside

__all__ = ['Statis']


def get_normalized_ds(ds):
    """Returns normalized version of dataset for Statis
    1) zscore each column
    2) make sum of squares to one (Frobenius norm = 1)

    Parameters
    ----------
    ds : dataset

    Returns
    -------
    Normalized input dataset
    """
    ds = ds.copy()
    if isinstance(ds, Dataset):
        ds.samples = ds.samples - ds.samples.mean(axis=0)
        ds.samples = ds.samples / ds.samples.std(axis=0)
        ds.samples /= np.sqrt((ds.samples**2).sum())
    else:
        ds = ds - ds.mean(axis=0)
        ds = ds / ds.std(axis=0)
        ds /= np.sqrt((ds**2).sum())
    return ds


def get_Rv(dss):
    """Compute Rv matrix: Cosine correlation of RSMs across datasets

    Parameters
    ----------
    dss: list of datasets,
     typically one per subject, with matching samples
    """
    z = np.vstack([np.dot(sd.samples, sd.samples.T).flatten() for sd in dss])
    return z, np.dot(z, z.T)


def get_eig(h):
    """Return sorted and sign flipped eigen decomposition of a matrix

    Parameters
    ----------
    C: np.ndarray
       Symmetric matrix

    Returns
    -------
    eigen values, eigen vectors
    """
    evals, evecs = np.linalg.eigh(h)
    # To deal with numerical errors
    evals[evals < 0] = 0
    eval_desc_order = np.argsort(evals)[-1::-1]
    evecs = evecs[:, eval_desc_order]
    evals = evals[eval_desc_order]
    # Fix signs
    evecs = evecs * np.sign(evecs[0, 0])
    return evals, evecs


class Statis(ClassWithCollections):
    """Alignment of subjects or tables using STATIS

    This algorithm takes a list of datasets with matching samples
    or a single dataset with `fa.chunks_attr` separating subjects/tables.
    It produces a list of `StaticProjectionMapper`s that project these feature spaces
    of input datasets into STATIS compromise space.

    Note that when provided a single dataset as input, returned mappers are in
    the order fa.chunks_attr.

    Statis also stores subject/table factor scores as `G_t`,
    each subject's contribution to compromise as `alpha`,
    and sample factor scores as `G_s`.

    References
    ----------
    .. [1] Abdi, H., Williams, L.J., Valentin, D., & Bennani-Dosse, M. (2012).
       STATIS and DISTATIS: Optimum multi-table principal component analysis and
       three way metric multidimensional scaling. Wiley Interdisciplinary
       Reviews: Computational Statistics, 4, 124-167.
    """

    chunks_attr = Parameter(
        'chunks',
        doc="""Name of the attribute indicating the individual chunks from
            which a single subject/table is drawn.""")

    nperms = Parameter(
        1000, constraints=EnsureInt() & EnsureRange(min=1),
        doc="""Number of permutations to carry out for statistical assessment.""")

    def __init__(self, **kwargs):
        ClassWithCollections.__init__(self, **kwargs)
        # Subject factor scores
        self.G_t = None
        # Sample factor scores
        self.G_s = None
        # output space dimensionality
        self.outdim = None
        # Subject specific weights for compromise matrix
        self.alpha = None

    def __call__(self, dss):
        """Derive compromise matrix and aligned space.
        Parameters
        ----------
        dss: a list of datasets

        Returns
        -------
        A list of StaticProjectionMappers matching the number of input datasets.
        """
        chunks_attr = self.params.chunks_attr
        if type(dss) == list:
            if __debug__:
                debug('STATIS', "Processing each of %d list items as tables" % len(dss))
            if len(dss) < 2:
                raise ValueError("You need more than one dataset for alignment.")
            nss = [sd.nsamples for sd in dss]
            if not nss.count(nss[0]) == len(nss):
                raise ValueError(
                    "All the datasets in the list should have matching "
                    "sample size. Input samples sizes are: %s" % nss)
        elif is_datasetlike(dss):
            if __debug__:
                debug('STATIS', "Processing each unique %s as table" % (chunks_attr))
            chunks = dss.fa[chunks_attr].value
            unique_chunks = dss.fa[chunks_attr].unique
            if len(unique_chunks) < 2:
                raise ValueError("You need more than one dataset for alignment."
                                 " Got a single one %s" % unique_chunks)
            dss = [dss[:, chunks == chunk] for chunk in unique_chunks]
        else:
            raise ValueError("Input should either be a list of datasets/matrices"
                             " or a dataset")

        # Normalizing data tables
        dss = [get_normalized_ds(sd) for sd in dss]
        nsamples = dss[0].nsamples
        nfs = [sd.nfeatures for sd in dss]
        self.outdim = min(nsamples - 1, min(nfs))

        # Deriving compromise matrix
        cpms, C = get_Rv(dss)
        e_t, ev_t = get_eig(C)
        # XXX We can repeat the above two steps with permuted samples per column in each dataset
        # to get a null distr of eigen values to evaluate how many are significant.
        # First one should be, if second one is also, then there are two clusters.
        e_t_perms = []
        for iperm in xrange(self.params.nperms):
            # Permute each column independently
            dss_perm = [hstack([sd[np.random.permutation(nsamples), icol]
                                for icol in xrange(sd.nfeatures)]) for sd in dss]
            e_t_perms.append(get_eig(get_Rv(dss_perm)[1])[0])
        e_t_perms = np.vstack(e_t_perms).T
        e_t_perms /= np.sum(e_t_perms, axis=0)

        # Check the significance of first and second eigenvalues
        # after normalizing by sum() This is a bit different from the paper
        if e_t[0] / np.sum(e_t) < np.percentile(e_t_perms[0, :], 90):
            raise ValueError(
                "First eigenvalue of subject COV is not significantly different"
                " from permutation distribution.")
        if e_t[1] / np.sum(e_t) >= np.percentile(e_t_perms[1, :], 90):
            warning("Second eigenvalue of subject factors is significant."
                    " There might be multiple clusters.")

        # Factor scores for subjects/tables
        self.G_t = np.dot(ev_t, np.diag(np.sqrt(e_t)))
        # Subject weights for compromise cross-product matrix
        self.alpha = ev_t[:, 0] / np.sum(ev_t[:, 0])
        # Checking if all weights are positive, otherwise thrown a warning or error
        # This is true in almost all cases, even random datasets due to correlation whole
        # RSMs and not just lower triangle without diagonal
        if not np.all(self.alpha > 0.0):
            if __debug__:
                debug('STATIS', "Subject weights for compromise matrix: %s"
                      % str(self.alpha))
            raise ValueError(
                "Input data similarity structures are not similar "
                " enough for alignment (i.e. not positively-correlated)."
                " Remove inconsistent subjects/tables and try again.")

        # Compromise cross-product matrix
        compromise = np.dot(self.alpha, cpms)
        compromise = compromise.reshape((int(np.sqrt(len(compromise))), -1))
        # Eigen decomposition of compromise cross-product matrix
        e_s, ev_s = get_eig(compromise)

        # Compromise factor scores for samples
        self.G_s = np.dot(ev_s, np.diag(np.sqrt(e_s)))
        # Projection matrix for subjects to compromise space
        # proj_k = X_k.T*P*D^-0.5
        e_s_invsqrt = np.zeros(e_s.shape)
        e_s_invsqrt[e_s > 0] = np.sqrt(1.0 / e_s[e_s > 0])
        Q_s = np.dot(ev_s, np.diag(e_s_invsqrt))
        projs = [np.dot(sd.samples.T, Q_s) for sd in dss]

        return [StaticProjectionMapper(proj=proj[:, :self.outdim])
                for proj in projs]
