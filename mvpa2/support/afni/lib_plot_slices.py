# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
'''
A simple slice plotter, mostly aimed at AFNI files
'''

import numpy as np
import nibabel as nb
import argparse

from mvpa2.base import externals
externals.exists('matplotlib', raise_=True)

import matplotlib
matplotlib.use('Agg') # this seems to work on non-(X windows) systems
import matplotlib.pyplot as plt


import os
import subprocess
from os.path import join, split, abspath

_VERSION = ".1"
_NAME = 'crisp - co-registered image slice plotter'
_AUTHOR = "Nov 2013, Nikolaas N. Oosterhof"
_NP_FLOAT_DTYPE = np.float16

def reduce_size(a):
    '''
    Reduces storage space for a numpy array

    Parameters
    ----------
    a: np.ndarray

    Returns
    -------
    b: np.ndarray stored as _NP_FLOAT_DTYPE
    '''
    return np.asarray(a, dtype=_NP_FLOAT_DTYPE)

def load_afni(fn, master_fn=None):
    '''
    Loads an AFNI volume.

    Parameters
    ----------
    fn: string
        Filename of AFNI file to load.
    master_fn: string or None
        If a string, then 'fn' is resampled to the grid of 'master_fn'.

    Returns
    -------
    img: np.ndarray
        Volumetric data in 'fn'.
    '''

    [p, nm] = split(abspath(fn))

    # find a unique filename for temporary storage
    tmp_pat = '__tmp%d.nii'
    d = 0
    while True:
        tmp_fn = join(p, tmp_pat % d)
        if not os.path.exists(tmp_fn):
            break
        d += 1

    # resample if 'master_fn' is supplied, otherwise just copy.
    # in both cases convert to nifti

    if master_fn is None:
        cmd = '3dbucket -overwrite -prefix %s %s[0]' % \
                (tmp_fn, fn)
    else:
        cmd = '3dresample -overwrite -prefix %s -master %s -inset %s[0]' % \
                (tmp_fn, master_fn, fn)

    # convert file
    subprocess.check_call(cmd.split())

    # load nifti file
    data = load_nii(tmp_fn)

    # remove temporary file
    os.unlink(tmp_fn)

    return data


def load_nii(fn):
    '''
    Loads a NIFTI volume.

    Parameters
    ----------
    fn: string
        Filename of NIFTI file to load.

    Returns
    -------
    img: np.ndarray
        Volumetric data in 'fn'.
    '''
    img = nb.load(fn)
    data = reduce_size(img.get_data())
    if len(data.shape) != 3:
        raise ValueError('Expected 3D array, but shape is %s' %
                                                (data.shape,))
    return data

def load_vol(fn, master_fn=None):
    '''
    Loads an AFNI or NIFTI volume.

    Parameters
    ----------
    fn: string
        Filename of AFNI file to load.
    master_fn: string or None
        If a string, then 'fn' is resampled to the grid of 'master_fn'
        if 'fn' is an AFNI volume. If 'fn' is a NIFTI volume then
        'master_fn' is ignored.

    Returns
    -------
    img: np.ndarray
        Volumetric data in 'fn'.
    '''

    is_nifti = any(fn.endswith(e) for e in ['.nii', '.nii.gz'])

    if is_nifti:
        if master_fn is not None:
            print "Warning: Ignoring master %s for %s" % (master_fn, fn)
        return load_nii(fn)
    else:
        return load_afni(fn, master_fn)

def slice_data(data, dim, where):
    '''
    Slices data along a dimension

    Parameters
    ----------
    data: np.ndarray
        Image data
    dim: int or list of int
        Dimension along which to slice
    where: float or list of float
        Position where to slice (in range 0..1)

    Returns
    -------
    sl: (list of (list of)) np.ndarray
        Depending on whether 'dim' and/or 'where' are lists or not,
        sl is either a single np.ndarray, a list of such arrays,
        or a list of a list of such arrays
    '''
    if type(dim) in (list, tuple):
        # use recursion
        return [slice_data(data, d, where) for d in dim]

    # make first dimension of interest
    if dim > 0 and not data is None:
        data = data.swapaxes(0, dim)

    if type(where) in (list, tuple):
        # use recursion
        return [slice_data(data, 0, w) for w in where]

    if data is None:
        return None

    if type(where) is float:
        # convert to integer index
        n = data.shape[0]
        where = int(float(n) * where)

    return np.squeeze(data[where, :, :]).T[::-1, ::-1]

def color_slice(data, cmap, min_max=None):
    '''
    Applies a colormap to data

    Parameters
    ----------
    data: (list of) np.ndarray
        Data to be colored
    cmap: colormap instance, bool or string
        If a string then a colormap with the name 'cmap' is used.
        True is equivalent to 'hot'. False is equivalent to 'gray'.
    min_max: pair of float or None
        Range to which values in 'data' or scaled. If None then
        the minimum and maximum values in 'data' are used.

    Returns
    -------
    c: np.ndarray
        Color value array with size PxQx4.
    '''

    # get colormap
    if type(cmap) is bool:
        cmap = 'hot' if cmap else 'gray'
    if isinstance(cmap, basestring):
        cmap = plt.get_cmap(cmap)

    if type(data) is list:
        return [color_slice(d, cmap, min_max) for d in data]

    if data is None:
        return None

    if min_max is None:
        max_ = np.max(data)
        min_ = np.min(data)

    # make sure it floats & scale to 0..1
    scale_inplace(data)
    clip_inplace(data)

    sh = data.shape

    # colormapping requires an array or matrix, so
    # ravel array here, then reshape after mapping
    cs_lin = cmap(data.ravel())

    cs = np.reshape(cs_lin, sh + cs_lin.shape[-1:])
    cs = reduce_size(cs)

    return cs

def blend(datas, weights=None):
    '''
    Blends different color value arrays

    Parameters
    ----------
    datas: tuple of np.ndarray
        Arrays to be blended
    weights: list of float or None
        Weight for each array. If None then weights is an array of ones.

    Returns
    -------
    b: np.ndarray
        Weighted sum of 'datas'. Data is clipped to range 0..1
    '''
    if type(datas) is list:
        return [blend(d, weights) for d in datas]

    n = len(datas)
    if weights is None:
        weights = np.ones((n, 1))

    #  space for output
    b = None

    for (data, weight) in zip(datas, weights):
        if data is None:
            continue

        if b is None:
            b = np.zeros(data.shape, dtype=_NP_FLOAT_DTYPE)

        # ensure shape is the same for all inputs
        if data.shape != b.shape:
            raise ValueError('Shape mismatch: %s %s %s' %
                        slice_.shape, b.sh)
        b += weight * data

    # clip range
    clip_inplace(b)

    return b

def scale_inplace(x, range_=None):
    '''
    Scales array in-place to 0..1

    Parameters
    ----------
    x: np.ndarray
        Data to be scaled. Data is modified in-place.
    range_: None or [min_, max_]
        Range to scale the data to. None is equivalent to
        [np.min(x), np.max(x)]
    '''
    if x is None:
        return

    if range_ is None:
        range_ = [np.min(x), np.max(x)]

    min_, max_ = range_
    x -= min_
    x /= (max_ - min_)

def clip_inplace(x, range_=(0., 1.)):
    '''
    Clips data in-place

    Parameters
    ----------
    x: np.ndarray
        Data to be clipped (in-place)
    range_: None or [min_, max_]
        Range to clip to. None is equivalent to [0,1]
    '''
    if x is None:
        return
    min_, max_ = range_
    x[x < min_] = min_
    x[x > max_] = max_

def make_plot(ulay, olay, dims, pos, title=None,
                ulay_range=None, olay_range=None, output_fn=None):
    '''
    Generates a plot of slices with different overlays and underlays

    Parameters
    ----------
    ulay: np.ndarray or str or None
        (filename of) underlay
    olay: np.ndarray or str or None
        (filename of) overlay
    dims: list of int
        dimensions to plot (0=x, 1=y, 2=z)
    pos: list of float
        relative positions to slice (in range 0..1)
    title: str or None
        title of plot
    ulay_range: None or [min_, max_]
        range to scale underlay to
    olay_range: None or [min_, max_]
        range to scale overlay to
    output_fn: None or str
        If not None the output is saved to this file

    Returns
    -------
    plt: plt
    '''

    # set some defaults
    figsize = (15, 12)
    imglength = 200
    fontsize = 30
    fontcolor = 'white'
    bgcolor = 'black'

    # load underlay and overlay
    if isinstance(ulay, basestring):
        ulay_name = split(ulay)[1]
        u = load_vol(ulay)
    else:
        ulay_name = ''
        u = ulay

    if isinstance(olay, basestring):
        olay_name = split(olay)[1]
        o = load_vol(olay)
    else:
        olay_name = ''
        o = olay

    title = '%s / %s' % (ulay_name, olay_name)

    # scale
    scale_inplace(u, ulay_range)
    scale_inplace(o, olay_range)

    # slice data, so that u and o become lists of lists of 2D-arrays
    u = slice_data(u, dims, pos)
    o = slice_data(o, dims, pos)

    # color slices, so that u and o become lists of lists of 3D-arrays (X*Y*4)
    u = color_slice(u, False, (0, 1))
    o = color_slice(o, True, (0, 1))

    # blend underlay and overlay
    uo = map(zip, *zip(*zip(u, o)))
    sl = blend(uo)

    # free up some space
    del(o)
    del(u)

    # generate slices
    ncol = len(sl[0])
    nrow = len(sl)

    # define subplots: one row per dimension, one column per position
    [f, axs] = plt.subplots(nrow, ncol, sharex=True, figsize=figsize)
    f.set_facecolor(bgcolor)
    counter = 0
    for j in xrange(ncol):
        for i in xrange(nrow):
            counter += 1
            ax = axs[i, j]

            ax.imshow(sl[i][j], extent=(0, imglength) * 2)
            ax.axis('off')
            ax.set_axis_bgcolor(bgcolor)

    # attempt to set a tight layout (doesn't work greatly)
    f.tight_layout()
    if not title is None:
        # set title
        axs[0][0].text(0, imglength, title, color=fontcolor, fontsize=fontsize)

    plt.draw()

    if output_fn is not None:
        plt.savefig(output_fn, facecolor='black')


def make_scatter(ulay, olay, output_fn=None):
    '''
    Generates a scatter plot between intensity values of underlay and overlay

    Generates a plot of slices with different overlays and underlays

    Parameters
    ----------
    ulay: np.ndarray or str or None
        (filename of) underlay
    olay: np.ndarray or str or None
        (filename of) overlay
    output_fn: None or str
        If not None the output is saved to this file

    Returns
    -------
    plt: plt
    '''

    cutoff_rel = .1  # ignore lowest 10% of values
    figsize = (15, 12) # size of figure
    histbincount = 25 # number of bins in histogram
    internal_rel = .25 # only show voxels within 50% of center of mass

    # load data
    u = load_vol(ulay)
    o = load_vol(olay, ulay)

    # define cutoff function
    def cutoff(x, cutoff_rel=cutoff_rel):
        xr = x.ravel()
        x_sorted = np.sort(xr)
        cutoff_abs = xr[np.round(cutoff_rel * x.size)]
        return cutoff_abs

    # only keep voxels that survive cutoff in both underlay and overlay
    msk = np.logical_and(o > cutoff(o), u > cutoff(u))
    apply_msk = lambda x:np.asarray(x[msk], dtype=np.float_)

    um, om = map(apply_msk, (u, o))

    ## compute distance of each voxel from center of mass
    sh = np.asarray(u.shape)
    ndim = len(sh)
    nf = np.prod(sh)

    #
    xyz = np.zeros((nf, ndim), dtype=u.dtype)

    for dim in xrange(ndim):
        r = np.arange(sh[dim])
        vec_dim = np.ones(ndim)
        vec_dim[dim] = sh[dim]
        r_shaped = np.reshape(r, vec_dim)

        tile_dim = sh.copy()
        tile_dim[dim] = 1
        dim_coord = np.tile(r_shaped, tile_dim)
        xyz[:, dim] = dim_coord.ravel()

    xyzm = np.asarray(xyz[msk.ravel(), :], dtype=np.float_) # mask & get to native float dtype
    com = np.mean(xyzm, 0) # center of mass

    delta = xyzm - com
    c = np.sqrt(np.sum(delta ** 2, 1)) # distance from center of mass

    c[c > np.max(c) * internal_rel] = np.Inf

    # indices to sort by distance
    ii = np.argsort(c)
    ii = ii[np.isfinite(c[ii])] # remove infinite values
    ii = ii[::-1] # reverse order - small ones on top

    # apply indices
    c = c[ii]
    um = um[ii]
    om = om[ii]


    # build the scatter plot
    # inspired by http://matplotlib.org/examples/pylab_examples/scatter_hist.html
    # by the matplotlib development team

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(figsize=figsize)

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    nullfmt = plt.NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axScatter.scatter(um, om, c=c)
    axScatter.set_xlabel('Underlay intensity')
    axScatter.set_ylabel('Overlay intensity')

    umax, omax = map(np.max, (um, om))

    axScatter.set_xlim((0, umax))
    axScatter.set_ylim((0, omax))

    ubins = np.arange(histbincount + 1) * (umax / histbincount)
    obins = np.arange(histbincount + 1) * (omax / histbincount)

    axHistx.hist(um, bins=ubins)
    axHisty.hist(om, bins=obins, orientation='horizontal')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    title = '%s / %s (r=%.3f)' % (split(ulay)[1], split(olay)[1], np.corrcoef(um, om)[0, 1])
    axHistx.set_title(title)

    plt.draw()

    if output_fn is not None:
        plt.savefig(output_fn)

def get_parser():
    description = '%s version %s\n%s' % \
                        (_NAME, _VERSION, _AUTHOR)
    epilog = '''
    Example:

        %s  ulay+tlrc.HEAD olay+tlrc.HEAD combined_image_out.png

    If NIFTI files are supplied to this function then they have to be in
    correspondence. If AFNI files are supplied the overlay is resampled to
    the grid of the underlay.
        ''' % __file__

    formatter_class = argparse.ArgumentDefaultsHelpFormatter

    p = argparse.ArgumentParser(description=description,
                                     epilog=epilog,
                                     formatter_class=formatter_class)
    p.add_argument("ulay", help="Underlay file name")
    p.add_argument("olay", help="Overlay file name")
    p.add_argument("output_fn", help="Output file name")
    p.add_argument('-u', '--ulay_range', nargs=2, default=None, type=float, help='Underlay range (min max)')
    p.add_argument('-o', '--olay_range', nargs=2, default=None, type=float, help='Overlay range  (min max)')
    p.add_argument('-d', '--dims', nargs='?', default=[0, 1, 2], type=int, help='Dimensions to plot; 0=x, 1=y, 2=z')
    p.add_argument('-p', '--pos', nargs='?', default=[.35, .45, .55, .65], type=float, help='Relative slice positions in range 0..1')
    p.add_argument('-t', '--title', default=None, help='Plot title')

    return p
'''
if __name__ == '__main__':
    p = get_parser()
    args = p.parse_args()
    v = vars(args)
    make_plot(v)
'''
