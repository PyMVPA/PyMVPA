#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc function performing operations on datasets."""

__docformat__ = 'restructuredtext'


def zscore(dataset, mean = None, std = None, perchunk=True):
    """Z-Score the samples of a `Dataset` (in-place).

    `mean` and `std` can be used to pass custom values to the z-scoring.
    Both may be scalars or arrays.

    All computations are done in place. Data upcasting is done
    automatically if necessary.

    If `perchunk` is True samples within the same chunk are z-scored independent
    of samples from other chunks, e.i. mean and standard deviation are
    calculated individually.
    """
    # cast to floating point datatype if necessary
    if str(dataset.samples.dtype).startswith('uint') \
       or str(dataset.samples.dtype).startswith('int'):
        dataset.setSamplesDType('float64')

    def doit(samples, mean, std):
        # calculate mean if necessary
        if not mean:
            mean = samples.mean(axis=0)

        # calculate std-deviation if necessary
        if not std:
            std = samples.std(axis=0)

        # do the z-scoring
        samples -= mean
        samples /= std

        return samples

    if perchunk:
        for c in dataset.uniquechunks:
            slicer = dataset.chunks == c
            dataset.samples[slicer] = doit(dataset.samples[slicer], mean, std)
    else:
        doit(dataset.samples, mean, std)
