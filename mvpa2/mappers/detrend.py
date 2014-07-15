# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Polynomial de-trending and regression."""

__docformat__ = 'restructuredtext'

import numpy as np
from mvpa2.base.types import is_sequence_type

from mvpa2.base import externals
if externals.exists('scipy', raise_=True):
    # if we construct the polynomials ourselves, we wouldn't need scipy here
    from scipy.special import legendre

    def legendre_(n, x):
        """Helper to avoid problems with scipy 0.8.0 returning inf for -1

        Scipy 0.8.0 (and possibly later) has regression of reporting
        'inf's for negative boundary. Lets guard against it for now
        """
        leg = legendre(n)
        r = leg(x)
        infs = np.isinf(r)
        if np.any(infs):
            r[infs] = leg(x[infs] + 1e-10) # offset to try to overcome problems
        return r

from mvpa2.base.dochelpers import _str, borrowkwargs
from mvpa2.mappers.base import Mapper
from ..base.param import Parameter
from ..base import constraints as cts

class PolyDetrendMapper(Mapper):
    """Mapper for regression-based removal of polynomial trends.

    Noteworthy features are the possibility for chunk-wise detrending, optional
    regressors, and the ability to use positional information about the samples
    from the dataset.

    Any sample attribute from the to be mapped dataset can be used to define
    `chunks` that shall be detrended separately. The number of chunks is
    determined from the number of unique values of the attribute and all samples
    with the same value are considered to be in the same chunk.

    It is possible to provide a list of additional sample attribute names that
    will be used as confound regressors during detrending. This, for example,
    allows to use fMRI motion correction parameters to be considered.

    Finally, it is possible to use positional information about the dataset
    samples for the detrending. This is useful, for example, if the samples in
    the dataset are not equally spaced out over the acquisition time-window.
    In that case an actually linear trend in the data would be distorted and
    not properly removed. By setting the `inspace` argument to the name of a
    samples attribute that carries this information, the mapper will take this
    into account and shift the polynomials accordingly. If `inspace` is given,
    but the dataset doesn't contain such an attribute evenly spaced coordinates
    are generated and this information is stored in the mapped dataset.

    Notes
    -----
    The mapper only support mapping of datasets, not plain data. Moreover,
    reverse mapping, or subsequent forward-mapping of partial datasets are
    currently not implemented.

    Examples
    --------
    >>> from mvpa2.datasets import dataset_wizard
    >>> from mvpa2.mappers.detrend import PolyDetrendMapper
    >>> samples = np.array([[1.0, 2, 3, 3, 2, 1],
    ...                    [-2.0, -4, -6, -6, -4, -2]]).T
    >>> chunks = [0, 0, 0, 1, 1, 1]
    >>> ds = dataset_wizard(samples, chunks=chunks)
    >>> dm = PolyDetrendMapper(chunks_attr='chunks', polyord=1)

    >>> # the mapper will be auto-trained upon first use
    >>> mds = dm.forward(ds)

    >>> # total removal all all (chunk-wise) linear trends
    >>> np.sum(np.abs(mds)) < 0.00001
    True
    """
    polyord = Parameter(1, doc=\
          """Order of the Legendre polynomial to remove from the data.  This
          will remove every polynomial up to and including the provided
          value.  For example, 3 will remove 0th, 1st, 2nd, and 3rd order
          polynomials from the data.  np.B.: The 0th polynomial is the
          baseline shift, the 1st is the linear trend.
          If you specify a single int and the `chunks_attr` parameter is not None, then this value
          is used for each chunk.  You can also specify a different polyord
          value for each chunk by providing a list or ndarray of polyord
          values with the length equal to the number of chunks.""",
          constraints=cts.AltConstraints(cts.EnsureInt()))

    chunks_attr = Parameter(None, doc=\
          """If None, the whole dataset is detrended at once. Otherwise, the given
          samples attribute (given by its name) is used to define chunks of the
          dataset that are processed individually. In that case, all the samples
          within a chunk should be in contiguous order and the chunks should be
          sorted in order from low to high -- unless the dataset provides
          information about the coordinate of each sample in the space that
          should be spanned be the polynomials (see `space` argument).""",
          constraints=cts.AltConstraints(None, cts.EnsureStr()))

    opt_regs = Parameter(None, doc=\
          """List of sample attribute names that should be used as
          additional regressors.  An example use would be to regress out motion
          parameters.""",
          constraints=cts.AltConstraints(None, cts.EnsureListOf(str)))

    def __init__(self, polyord=1, chunks_attr=None, opt_regs=None, **kwargs):
        """
        Parameters
        ----------
        space : str or None
          If not None, a samples attribute of the same name is added to the
          mapped dataset that stores the coordinates of each sample in the
          space that is spanned by the polynomials. If an attribute of that
          name is already present in the input dataset its values are interpreted
          as sample coordinates in the space that should be spanned by the
          polynomials.
        """
        # keep param init for historical reasons
        self.params.chunks_attr = chunks_attr
        self.params.polyord = polyord
        self.params.opt_regs = opt_regs

        # things that come from train()
        self._polycoords = None
        self._regs = None

        # secret switch to perform in-place detrending
        self._secret_inplace_detrend = False

        # need to init last to prevent base class puking
        Mapper.__init__(self, **kwargs)


    def __repr__(self):
        s = super(PolyDetrendMapper, self).__repr__()
        return s.replace("(",
                         "(polyord=%i, chunks_attr=%s, opt_regs=%s, "
                          % (self.params.polyord,
                             repr(self.params.chunks_attr),
                             repr(self.params.opt_regs)),
                         1)

    def __str__(self):
        return _str(self, ord=self.params.polyord)


    def _scale_array(self, a):
        # scale an array into the interval [-1,1] using its min and max values
        # as input range
        return a / ((a.max() - a.min()) / 2) - 1


    def _get_polycoords(self, ds, chunk_slicer):
        """Compute human-readable and scaled polycoords.

        Parameters
        ----------
        ds : dataset
        chunk_slicer : boolean array
          True elements indicate samples selected for detrending.

        Returns
        -------
        tuple
          (real-world polycoords, scaled polycoords into [-1,1])
        """
        inspace = self.get_space()
        if chunk_slicer is None:
            nsamples = len(ds)
        else:
            nsamples = chunk_slicer.sum()

        # if we don't have to take care of an inspace thing are easy
        if inspace is None:
            polycoords_scaled = np.linspace(-1, 1, nsamples)
            return None, polycoords_scaled
        else:
            # there is interest in the inspace, but do we have information, or
            # just want to store it later on
            if inspace in ds.sa:
                # we have info
                if chunk_slicer is None:
                    chunk_slicer = slice(None)
                polycoords = ds.sa[inspace].value[chunk_slicer]
            else:
                # no info in the dataset, just be nice and generate some
                # meaningful polycoords
                polycoords = np.arange(nsamples)
            return polycoords, self._scale_array(polycoords.astype('float'))


    def _train(self, ds):
        # local binding
        chunks_attr = self.params.chunks_attr
        polyord = self.params.polyord
        opt_reg = self.params.opt_regs
        inspace = self.get_space()
        self._polycoords = None

        # global detrending is desired
        if chunks_attr is None:
            # consider the entire dataset
            reg = []
            # create the timespan
            self._polycoords, polycoords_scaled = self._get_polycoords(ds, None)
            for n in range(polyord + 1):
                reg.append(legendre_(n, polycoords_scaled)[:, np.newaxis])
        # chunk-wise detrending is desired
        else:
             # get the unique chunks
            uchunks = ds.sa[chunks_attr].unique

            # Process the polyord to be a list with length of the number of
            # chunks
            if not is_sequence_type(polyord):
                # repeat to be proper length
                polyord = [polyord] * len(uchunks)
            elif not chunks_attr is None and len(polyord) != len(uchunks):
                raise ValueError("If you specify a sequence of polyord values "
                                 "they sequence length must match the "
                                 "number of unique chunks in the dataset.")

            # loop over each chunk
            reg = []
            update_polycoords = True
            # if the dataset know about the inspace we can store the
            # polycoords right away
            if not inspace is None and inspace in ds.sa:
                self._polycoords = ds.sa[inspace].value
                update_polycoords = False
            else:
                # otherwise we prepare and empty array that is going to be
                # filled below -- we know that those polycoords are going to
                # be ints
                self._polycoords = np.empty(len(ds), dtype='int')
            for n, chunk in enumerate(uchunks):
                # get the indices for that chunk
                cinds = ds.sa[chunks_attr].value == chunk

                # create the timespan
                polycoords, polycoords_scaled = self._get_polycoords(ds, cinds)
                if update_polycoords and not polycoords is None:
                    self._polycoords[cinds] = polycoords
                # create each polyord with the value for that chunk
                for n in range(polyord[n] + 1):
                    newreg = np.zeros((len(ds), 1))
                    newreg[cinds, 0] = legendre_(n, polycoords_scaled)
                    reg.append(newreg)

        # if we don't handle in inspace, there is no need to store polycoords
        if inspace is None:
            self._polycoords = None

        # see if add in optional regs
        if not opt_reg is None:
            # add in the optional regressors, too
            for oreg in opt_reg:
                reg.append(ds.sa[oreg].value[np.newaxis].T)

        # combine the regs (time x reg)
        self._regs = np.hstack(reg)


    def _forward_dataset(self, ds):
        # auto-train the mapper if not yet done
        if self._regs is None:
            self.train(ds)

        if self._secret_inplace_detrend:
            mds = ds
        else:
            # shallow copy to put the new stuff in
            mds = ds.copy(deep=False)

        # local binding
        regs = self._regs
        inspace = self.get_space()
        polycoords = self._polycoords

        # is it possible to map that dataset?
        if inspace is None and len(regs) != len(ds):
            raise ValueError("Cannot detrend the dataset, since it neither "
                             "provides location information of its samples "
                             "in the space spanned by the polynomials, "
                             "nor does it match the number of samples this "
                             "this mapper has been trained on. (got: %i "
                             " and was trained on %i)."
                             % (len(ds), len(regs)))
        # do we have to handle the polynomial space somehow?
        if not inspace is None:
            if inspace in ds.sa:
                space_coords = ds.sa[inspace].value
                # this dataset has some notion about our polyspace
                # we need to put the right regressor values for its samples
                # let's first see whether the coords are identical to the
                # trained ones (that should be the common case and nothing needs
                # to be done
                # otherwise look whether we can find the right regressors
                if not np.all(space_coords == polycoords):
                    # to make the stuff below work, we'd need to store chunk
                    # info too, otherwise we cannot determine the correct
                    # regressor rows
                    raise NotImplementedError
                    # determine the regressor rows that match the samples
                    reg_idx = [np.argwhere(polycoords == c).flatten()[0]
                                    for c in space_coords]
                    # slice the regressors accordingly
                    regs = regs[reg_idx]
            else:
                # the input dataset knows nothing about the polyspace
                # let's put that information into the output dataset
                mds.sa[inspace] = self._polycoords

        # regression for each feature
        fit = np.linalg.lstsq(regs, ds.samples)
        # actually we are only interested in the solution
        # res[0] is (nregr x nfeatures)
        y = fit[0]
        # remove all and keep only the residuals
        if self._secret_inplace_detrend:
            # if we are in evil mode do evil

            # cast the data to float, since in-place operations below do not
            # upcast!
            if np.issubdtype(mds.samples.dtype, np.integer):
                mds.samples = mds.samples.astype('float')

            mds.samples -= np.dot(regs, y)
        else:
            # important to assign to ensure COW behavior
            mds.samples = ds.samples - np.dot(regs, y)

        return mds



    def _forward_data(self, data):
        raise RuntimeError("%s cannot map plain data."
                           % self.__class__.__name__)



@borrowkwargs(PolyDetrendMapper, '__init__')
def poly_detrend(ds, **kwargs):
    """In-place polynomial detrending.

    This function behaves identical to the `PolyDetrendMapper`. The only
    difference is that the actual detrending is done in-place -- potentially
    causing a significant reduction of the memory demands.

    Parameters
    ----------
    ds : Dataset
      The dataset that will be detrended in-place.
    **kwargs
      For all other arguments, please see the documentation of
      PolyDetrendMapper.
    """
    dm = PolyDetrendMapper(**kwargs)
    dm._secret_inplace_detrend = True
    # map
    mapped = dm.forward(ds)
    # and append the mapper to the dataset
    mapped._append_mapper(dm)
