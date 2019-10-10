# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Routines to scatterplot data"""

__docformat__ = 'restructuredtext'


import sys, os
import pylab as pl
import nibabel as nb
import numpy as np

from mvpa2.base import verbose, warning
from mvpa2.base import externals

__all__ = ['plot_scatter']


def fill_nonfinites(a, fill=0, inplace=True):
    """Replace all non-finites (NaN, inf) etc with a fill value
    """
    nonfinites = ~np.isfinite(a)
    if np.any(nonfinites):
        if not inplace:
            a = a.copy()
        a[nonfinites] = fill
    return a


if externals.versions['matplotlib'] >= '2':
    pl_axes = pl.axes
else:
    # older versions, e.g. 1.3, do not understand facecolor
    def pl_axes(*args, **kwargs):
        if 'facecolor' in kwargs:
            kwargs['axisbg'] = kwargs.pop('facecolor')
        return pl.axes(*args, **kwargs)
    pl_axes.__doc__ = pl.axes.__doc__


def plot_scatter(dataXd, mask=None, masked_opacity=0.,
                 labels=None, colors=True,
                 dimcolor=1, title=None, limits='auto',
                 thresholds=None, hint_opacity=0.9,
                 x_jitter=None, y_jitter=None,
                 fig=None,
                 ax_scatter=None, ax_hist_x=None, ax_hist_y=None,
                 bp_location='scatter',
                 xlim=None, ylim=None,
                 rasterized=None,
                 uniq=False,
                 include_stats=False,
                 ):
    """
    Parameters
    ----------
    dataXd: array
      The volumetric (or not) data to plot where first dimension
      should only have 2 items
    mask: array, optional
      Additional mask to specify which values do not consider to plot.
      By default values with 0s in both dimensions are not plotted.
    masked_opacity: float, optional
      By default masked out values are not plotted at all.  Value in
      (0,1] will make them visible with this specified opacity
    labels: list of str, optional
      Labels to place for x and y axes
    colors: bool or string or colormap, optional
      Either to use colors to associate with physical location and
      what colormap to use (jet by default if colors=True)
    dimcolor: int
      If `colors`, then which dimension (within given 3D volume) to
      "track"
    limits: 'auto', 'same', 'per-axis' or (min, max)
      Limits for axes: when 'auto' if data ranges overlap is more than
      50% of the union range, 'same' is considered.  When 'same' --
      the same limits on both axes as determined by data.  If
      two-element tuple or list is provided, then that range is
      applied to both axes.
    hint_opacity: float, optional
      If `colors` is True, to then a "slice" of the volumetric data
      is plotted in the specified opacity to hint about the location
      of points in the original Xd data in `dimcolor` dimension
    x_jitter: float, optional
      Half-width of uniform noise added to x values.  Might be useful if data
      is quantized so it is valuable to jitter points a bit.
    y_jitter: float, optional
      Half-width of uniform noise added to y values.  Might be useful if data
      is quantized so it is valuable to jitter points a bit
    fig : Figure, optional
      Figure to plot on, otherwise new one created
    ax_*: axes, optional
      Axes for the scatter plot and histograms. If none of them is specified
      (which is the default) then 'classical' plot is rendered with histograms
      above and to the right
    bp_location: ('scatter', 'hist', None), optional
      Where to place boxplots depicting data range
    xlim: tuple, optional
    ylim: tuple, optional
      To fix plotted range
    rasterized: bool, optional
      Passed to scatter call, to allow rasterization of heavy scatter plots
    uniq: bool, optional
      Plot uniq values (those present in one but not in the other) along
      each axis with crosses
    include_stats: bool, optional
      Whether to report additional statistics on the data. Stats are also
      reported via verbose at level 2
    """
    if len(dataXd) != 2:
        raise ValueError("First axis of dataXd can only have two dimensions, "
                         "got {0}".format(len(dataXd)))
    dataXd = np.asanyarray(dataXd)      # TODO: allow to operate on list of arrays to not waste RAM/cycles
    data = dataXd.reshape((2, -1))
    if dataXd.ndim < 5:
        ntimepoints = 1
    elif dataXd.ndim == 5:
        ntimepoints = dataXd.shape[-1]
    else:
        raise ValueError("Do not know how to handle data with %d dimensions" % (dataXd.ndim - 1))
    if x_jitter or y_jitter:
        data = data.copy()              # lazy and wasteful
        def jitter_me(x, w):
            x += np.random.uniform(-w, w, size=data.shape[-1])
        if x_jitter:
            jitter_me(data[0, :], x_jitter)
        if y_jitter:
            jitter_me(data[1, :], y_jitter)

    finites = np.isfinite(data)
    nz = np.logical_and(data != 0, finites)
    # TODO : avoid doing data !=0 and just use provided utter mask
    #nz[:, 80000:] = False # for quick testing

    nzsum = np.sum(nz, axis=0)

    intersection = nzsum == 2
    # for coloring we would need to know all the indices
    union = nzsum > 0
    x, y = datainter = data[:, intersection]

    if mask is not None:
        if mask.size * ntimepoints == intersection.size:
            # we have got a single mask applicable to both x and y
            pass
        elif mask.size * ntimepoints == 2 * intersection.size:
            # we have got a mask per each, let's get an intersection
            assert mask.shape[0] == 2, "had to get 1 for x, 1 for y"
            mask = np.logical_and(mask[0], mask[1])
        else:
            raise ValueError(
                "mask of shape %s. data of shape %s. ntimepoints=%d.  "
                "Teach me how to apply it" % (mask.shape, data.shape, ntimepoints)
            )
        # replicate mask ntimepoints times
        mask = np.repeat(mask.ravel(), ntimepoints)[intersection] != 0
        x_masked = x[mask]
        y_masked = y[mask]


    xnoty = (nz[0].astype(int) - nz[1].astype(int))>0
    ynotx = (nz[1].astype(int) - nz[0].astype(int))>0

    msg = ''
    if not np.all(finites):
        msg = " non-finite x: %d, y: %d" % (np.sum(~finites[0]), np.sum(~finites[1]))

    verbose(1, "total: %d union: %d%s intersection: %d x_only: %d y_only: %d%s"
            % (len(nzsum),
               np.sum(union),
               mask is not None and ' masked: %d' % np.sum(mask) or '',
               np.sum(intersection),
               np.sum(xnoty), np.sum(ynotx),
               msg))

    if include_stats:
        # report some statistics as well
        import scipy.stats as ss
        r, p = ss.pearsonr(x, y)
        d = np.linalg.norm(x-y)
        statsline = "r=%.2f  p=%.4g  ||x-y||=%.4g" % (r, p, d)
        try:
            from mvpa2.misc.dcov import dcorcoef
            nmax = min(1000, len(x))
            idx = np.random.permutation(np.arange(len(x)))[:nmax]
            dcor = dcorcoef(x[idx], y[idx])
            dcor_s = '' if len(x) == nmax else '[%d random]' % nmax
            statsline += '  dcorr%s=%.4g' % (dcor_s, dcor)
        except ImportError:
            pass
        verbose(2, statsline)
    else:
        statsline = ''


    #fig=pl.figure()
    #pl.plot(datainter[0], datainter[1], '.')
    #fig.show()

    nullfmt  = pl.NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    if not (bool(ax_scatter) or bool(ax_hist_x) or bool(ax_hist_y)): # no custom axes specified
        # our default setup
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        # start with a rectangular Figure
        if fig is None:
            fig = pl.figure(figsize=(10,10))

        ax_scatter = pl.axes(rect_scatter)
        ax_hist_x = pl.axes(rect_histx)
        ax_hist_y = pl.axes(rect_histy)

    else:
        # check if all not None?
        # assert(len(axes) == 3)

        ax_bp_x, ax_bp_y = None, None
        if ax_scatter is None:
            raise ValueError("Makes no sense to do not have scatter plot")

    ax_bp_x = ax_bp_y = None
    if bp_location is not None:
        ax_bp_x_parent = ax_bp_y_parent = None
        if bp_location == 'scatter':
            # place boxplots into histogram plots
            ax_bp_x_parent = ax_scatter
            ax_bp_y_parent = ax_scatter
        elif bp_location == 'hist':
            ax_bp_x_parent = ax_hist_x
            ax_bp_y_parent = ax_hist_y
        else:
            raise ValueError("bp_location needs to be from (None, 'scatter', 'hist')")

        if ax_bp_x_parent:
            hist_x_pos = ax_bp_x_parent.get_position()
            ax_bp_x = pl_axes( [hist_x_pos.x0,    hist_x_pos.y0 + hist_x_pos.height * 0.9,
                                hist_x_pos.width, hist_x_pos.height * 0.1],  facecolor='y' )

        if ax_bp_y_parent:
            hist_y_pos = ax_bp_y_parent.get_position()
            ax_bp_y = pl_axes( [hist_y_pos.x0 + hist_y_pos.width*0.9,  hist_y_pos.y0,
                                hist_y_pos.width * 0.1, hist_y_pos.height],  facecolor='y' )

        # ax_bp_y = pl_axes( [left + width * 0.9, bottom, width/10, height], facecolor='y' ) if ax_hist_y else None


    sc_kwargs = dict(facecolors='none', s=1, rasterized=rasterized) # common kwargs

    # let's use colormap to get non-boring colors
    cm = colors                     # e.g. if it is None
    if colors is True:
        cm = pl.matplotlib.cm.get_cmap('jet')
    elif isinstance(colors, str):
        cm = pl.matplotlib.cm.get_cmap(colors)
    if cm and len(dataXd.shape) > dimcolor+1:
        cm.set_under((1, 1, 1, 0.1))             # transparent what is not in range
        # we need to get our indices back for those we are going to plot.  probably this is the least efficient way:
        ndindices_all = np.array(list(np.ndindex(dataXd.shape[1:])))
        ndindices_nz = ndindices_all[intersection]
        # choose color based on dimcolor
        dimcolor_len = float(dataXd.shape[1+dimcolor])
        edgecolors = cm(((cm.N-1) * ndindices_nz[:, dimcolor] / dimcolor_len).astype(int))
        if mask is not None:
            # Plot first those which might be masked out
            if masked_opacity:
                mask_inv = np.logical_not(mask)
                mask_edgecolors = edgecolors[mask_inv].copy()
                # Adjust alpha value
                mask_edgecolors[:, -1] *= masked_opacity
                ax_scatter.scatter(x[mask_inv], y[mask_inv],
                                  edgecolors=mask_edgecolors,
                                  alpha=masked_opacity,
                                  **sc_kwargs)

            # Plot (on top) those which are not masked-out
            if mask.size:
                x_plot, y_plot, edgecolors_plot = x[mask], y[mask], edgecolors[mask]
            else:
                # older numpys blow here
                x_plot, y_plot, edgecolors_plot = (np.array([]),) * 3
        else:
            # Just plot all of them at once
            x_plot, y_plot, edgecolors_plot = x, y, edgecolors

        if len(x_plot):
            ax_scatter.scatter(x_plot, y_plot, edgecolors=edgecolors_plot,
                             **sc_kwargs)

        # for orientation we need to plot 1 slice... assume that the last dimension is z -- figure out a slice with max # of non-zeros
        zdim_entries = ndindices_nz[:, -1]
        if np.size(zdim_entries):
            zdim_counts, _ = np.histogram(zdim_entries, bins=np.arange(0, np.max(zdim_entries)+1))
            zdim_max = np.argmax(zdim_counts)

            if hint_opacity:
                # now we need to plot that zdim_max slice taking into account our colormap
                # create new axes
                axslice = pl_axes([left, bottom+height * 0.72, width/4., height/5.],
                                  facecolor='y')
                axslice.axis('off')
                sslice = np.zeros(dataXd.shape[1:3]) # XXX hardcoded assumption on dimcolor =1
                sslice[:, : ] = np.arange(dimcolor_len)[None, :]
                # if there is time dimension -- choose minimal value across all values
                dataXd_mint = np.min(dataXd, axis=-1) if dataXd.ndim == 5 else dataXd
                sslice[dataXd_mint[0, ..., zdim_max] == 0] = -1 # reset those not in the picture to be "under" range
                axslice.imshow(sslice, alpha=hint_opacity, cmap=cm)
    else:
        # the scatter plot without colors to distinguish location
        ax_scatter.scatter(x, y, **sc_kwargs)

    if labels:
        ax_scatter.set_xlabel(labels[0])
        ax_scatter.set_ylabel(labels[1])

    # "unique" points on each of the axes
    if uniq:
        if np.sum(xnoty):
            ax_scatter.scatter(fill_nonfinites(data[0, np.where(xnoty)[0]]),
                              fill_nonfinites(data[1, np.where(xnoty)[0]]),
                              edgecolor='b', **sc_kwargs)
        if np.sum(ynotx):
            ax_scatter.scatter(fill_nonfinites(data[0, np.where(ynotx)[0]]),
                              fill_nonfinites(data[1, np.where(ynotx)[0]]),
                              edgecolor='g', **sc_kwargs)


    # Axes
    if np.size(x):
        ax_scatter.plot((np.min(x), np.max(x)), (0, 0), 'r', alpha=0.5)
    else:
        warning("There is nothing to plot, returning early")
        return pl.gcf()

    ax_scatter.plot((0, 0), (np.min(y), np.max(y)), 'r', alpha=0.5)

    if (mask is not None and not masked_opacity and np.sum(mask)):
        # if there is a non-degenerate mask which was not intended to be plotted,
        # take those values away while estimating min/max range
        _ = x[mask]; minx, maxx = np.min(_), np.max(_)
        _ = y[mask]; miny, maxy = np.min(_), np.max(_)
        del _                           # no need to consume RAM
        # print "Here y range", miny, maxy
    else:
        minx, maxx = np.min(x), np.max(x)
        miny, maxy = np.min(y), np.max(y)

    # Process 'limits' option
    if isinstance(limits, str):
        limits = limits.lower()
        if limits == 'auto':
            overlap = min(maxx, maxy) - max(minx, miny)
            range_ = max(maxx, maxy) - min(minx, miny)
            limits = {True: 'same', False: 'per-axis'}[not range_ or overlap/float(range_) > 0.5]

        if limits == 'per-axis':
            same_range = False
            if xlim is None:
                # add some white border
                dx = (maxx - minx)/20.
                xlim = (minx-dx, maxx+dx)
            if ylim is None:
                dy = (maxy - miny)/20.
                ylim = (miny-dy, maxy+dy)

        elif limits == 'same':
            same_range = True
            # assign limits the numerical range
            limits = (np.min( [minx, miny] ),  np.max( [maxx, maxy] ))
        else:
            raise ValueError("Do not know how to handle same_range=%r" % (limits,))
    else:
        same_range = True

    # Let's now plot threshold lines if provided
    if thresholds is not None:
        stylekwargs = dict(colors='k', linestyles='dotted')
        if len(thresholds):
            ax_scatter.vlines(thresholds[0], ax_scatter.get_xlim()[0]*0.9,
                             ax_scatter.get_xlim()[1]*0.9, **stylekwargs)
        if len(thresholds)>1:
            ax_scatter.hlines(thresholds[1], ax_scatter.get_ylim()[0]*0.9,
                             ax_scatter.get_ylim()[1]*0.9, **stylekwargs)

    if same_range:
        # now determine nice limits by hand:
        binwidthx = binwidthy = binwidth = np.max(datainter)/51. # 0.25

        minxy, maxxy = limits
        sgn = np.sign(minxy)
        xyrange = maxxy - minxy
        xyamax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
        limn = sgn*( int(sgn*minxy/binwidth) - sgn) * binwidth
        limp = ( int(maxxy/binwidth) + 1) * binwidth

        ax_scatter.plot((limn*0.9, limp*0.9), (limn*0.9, limp*0.9), 'y--')
        if xlim is None:
            xlim = (limn, limp)
        if ylim is None:
            ylim = (limn, limp)

        binsx = binsy = bins = np.arange(limn, limp + binwidth, binwidth)
    else:
        binwidthx = (maxx - minx)/51.
        binwidthy = (maxy - miny)/51.

        try:
            binsx = np.arange(minx, maxx + binwidthx, binwidthx)
            binsy = np.arange(miny, maxy + binwidthy, binwidthy)
        except Exception as exc:
            warning(
                "Received following exception while trying to get bins for "
                "minx=%(minx)f maxx=%(maxx)f binwidthx=%(binwidthx)s "
                "miny=%(miny)f maxy=%(maxy)f binwidthy=%(binwidthy)s: %(exc)s. "
                "Returning early"
                % locals()
            )
            return pl.gcf()

    if xlim is not None:
        ax_scatter.set_xlim( xlim )
    if ylim is not None:
        ax_scatter.set_ylim( ylim )

    # get values to plot for histogram and boxplot
    x_hist, y_hist = (x, y) if (mask is None or not np.sum(mask)) else (x_masked, y_masked)

    if np.any(binsx) and ax_hist_x is not None:
        ax_hist_x.xaxis.set_major_formatter(nullfmt)
        histx = ax_hist_x.hist(x_hist, bins=binsx, facecolor='b')
        ax_hist_x.set_xlim( ax_scatter.get_xlim() )
        ax_hist_x.vlines(0, 0, 0.9*np.max(histx[0]), 'r')

    if np.any(binsy) and ax_hist_y is not None:
        ax_hist_y.yaxis.set_major_formatter(nullfmt)
        histy = ax_hist_y.hist(y_hist, bins=binsy,
                               orientation='horizontal', facecolor='g')
        ax_hist_y.set_ylim( ax_scatter.get_ylim() )
        ax_hist_y.hlines(0, 0, 0.9*np.max(histy[0]), 'r')

    rect_scatter = [left, bottom, width, height]

    # Box plots
    if ax_bp_x is not None:
        ax_bp_x.axis('off')
        bpx = ax_bp_x.boxplot(x_hist, vert=0) #'r', 0)
        ax_bp_x.set_xlim(ax_scatter.get_xlim())

    if ax_bp_y is not None:
        ax_bp_y.axis('off')
        bpy = ax_bp_y.boxplot(y_hist, sym='g+')
        ax_bp_y.set_ylim(ax_scatter.get_ylim())

    if statsline:
        # draw the text based on gca
        y1, y2 = ax_scatter.get_ylim(); x1, x2 = ax_scatter.get_xlim();
        ax_scatter.text(0.5*(x1+x2),            # center
                       y2 - 0.02*(y2-y1),
                       statsline,
                       verticalalignment = "top", horizontalalignment="center")

    if title:
        pl.title(title)

    return pl.gcf()


def plot_scatter_matrix(d, style='full', labels=None, fig=None, width_=6, **kwargs):
    """
    Parameters
    ----------
    width_ : float, optional
      Width for each subplot if no fig was provided
    """
    n = len(d)
    if fig is None:
        fig = pl.figure(figsize=(width_*n, width_*n))

    # predefine axes for the plots
    # 1st row -- histograms
    # next ones -- scatter plots
    axes = np.zeros(shape=(n, n), dtype=object)

    if style == 'upper_triang':
        # style with upper row -- hists
        # next -- upper triang only
        for r in xrange(n):
            for c in xrange(r, n):
                sp = pl.subplot(n, n, r*n+c+1)
                axes[r,c] = pl.gca()

        for d1 in xrange(0, n-1):
            for d2 in xrange(d1+1, n):
                # only upper triangle
                plot_scatter([d[i] for i in [d2, d1]], ax_scatter=axes[d1+1, d2],
                             ax_hist_x=axes[0, d2] if d1==0 else None,
                             ax_hist_y=None,
                             bp_location='hist')
    elif style == 'full':

        nullfmt   = pl.NullFormatter()         # no labels

        # diagonal -- histograms
        for r in xrange(n):
            for c in xrange(n):
                sp = pl.subplot(n, n, r*n+c+1)
                axes[r, c] = pl.gca()

        for d1 in xrange(0, n):
            # we should unify the ranges of values displayed
            ylim = np.min(d[d1]), np.max(d[d1])
            for d2 in xrange(0, n):
                if d1 == d2:
                    continue
                xlim = np.min(d[d2]), np.max(d[d2])
                # only upper triangle
                hint_opacity = kwargs.pop('hint_opacity', 0.9) if (d1==0 and d2==1) else 0
                plot_scatter([d[d2], d[d1]],
                             ax_scatter=axes[d1, d2],
                             ax_hist_x=axes[d2, d2]
                                       if (d2==d1+1 or (d1==1 and d2==0)) else None,
                             ax_hist_y=None,
                             bp_location='hist',
                             hint_opacity=hint_opacity,
                             xlim=xlim,
                             ylim=ylim,
                             **kwargs
                             )
                # adjust slightly
                if not (d2==0 or (d1==0 and d2==1)): # not first column or in first row the first one
                    # forget about y axis labels
                    axes[d1, d2].yaxis.set_major_formatter(nullfmt)
                if not (d1==n-1 or (d2==n-1 and d1==n-2)): # not first row or in first column in the first one
                    # forget about y axis labels
                    axes[d1, d2].xaxis.set_major_formatter(nullfmt)

        if not (labels  in (None, [])):
            assert len(labels) == n, "We should be provided all needed labels"
            for d1,l in enumerate(labels):
                axes[d1, 0].set_ylabel(l)
            for d1,l in enumerate(labels):
                axes[n-1,d1].set_xlabel(l)

    else:
        raise ValueError("Unknown style %s" % style)

    return fig, axes


def unique_path_parts(*paths):
    paths_splits = [p.split(os.path.sep) for p in paths]
    minlen = min([len(x) for x in paths_splits])
    paths_split = np.array([p.split(os.path.sep, minlen-1) for p in paths]).T
    paths_short = []
    for p in paths_split:
        if len(np.unique(p)) > 1:
            # so do not match
            paths_short.append(list(p))
        else:
            if not len(paths_short) or paths_short[-1][0] != '...':
                paths_short.append(['...']*len(p))
    ret = [os.path.sep.join(p) for p in np.array(paths_short).T]
    return ret


def _get_data(f):
    """Adapter to load data from various formats
    """
    if f.endswith('.hdf5'):
        from mvpa2.base.hdf5 import h5load
        data = h5load(f).samples
    else: #if f.endswith('.nii.gz') or f.endswith('.img') or f.endswith('.hdr'):
        n = nb.load(f)
        data = n.get_data()
        # strip rudimentary 4th dimension
        if len(data.shape) == 4 and data.shape[-1] == 1:
            data = data[:, :, :, 0]
    return data


def plot_scatter_files(files,
                       mask_file=None,
                       masked_opacity=0.,
                       mask_thresholds=None,
                       volume=None,
                       scales=None,
                       return_data=False,
                       thresholds=None,
                       style='auto',
                       **kwargs):
    """
    Plot scatter plots based on data from files.

    Should work with volumetric file formats supported by nibabel
    and also .hdf5 files from PyMVPA

    Parameters
    ----------
    files: iterable of strings
      Files to load
    mask_file: string, optional
      Mask file.  0-ed out elements considered to be masked out
    masked_opacity: float, optional
      By default masked out values are not plotted at all.  Value in
      (0,1] will make them visible with this specified opacity
    mask_thresholds: int or list or tuple, optional
      A single (min) or two (lower, upper) values to decide which values to mask
      out (exclude).  If lower < upper, excludes range [lower, upper].
      if upper < lower, plots only values within [lower, upper] range.
    volume: int, None
      If multi-volume files provided, which volume to consider.
      Otherwise will plot for points from all volumes
    scales: iterable of floats, optional
      Scaling factors for the volumes
    return_data: bool, optional
      Flag to return loaded data as the output of this function call
    style: str, ('auto', 'full', 'pair1', 'upper_triang')
      Style of plotting -- full and upper_triang are the scatter
      matrix plots. With 'auto' chooses pair1 for 2 files, and full if more
      than 2 provided
    """
    datas = []
    labels_add = []
    for ix, f in enumerate(files):
        data = _get_data(f)
        #else:
        #    raise ValueError("Do not know how to handle %s" % f)
        if scales is not None and scales[ix]:
            data *= scales[ix]
        label = ''
        if volume is not None:
            if len(data.shape) > 4:
                assert(data.shape[3] == 1)
                data = data[:, :, :, 0, ...]
            if len(data.shape) > 3:
                data = data[:, :, :, volume]
                label += ' t=%i' % volume
        datas.append(data)
        labels_add.append(label)

    # Load mask if filename was specified
    if mask_file:
        mask = _get_data(mask_file)
        if mask_thresholds:
            mask_thresholds = np.atleast_1d(mask_thresholds)
            if len(mask_thresholds) == 1:
                mask[mask < mask_thresholds[0]] = 0
            else:
                min_, max_ = mask_thresholds
                if min_ < max_: # e.g. (-3, 3) to exclude
                    mask[(mask > min_) & (mask < max_)] = 0
                else:                # e.g. (3, -3) to exclude all > 3
                    mask[(mask > min_) | (mask < max_)] = 0
    else:
        mask = None

    if style == 'auto':
        style = 'pair1' if len(datas) <= 2 else 'full'
    kwargs_orig = kwargs
    figs = []
    if style == 'pair1':
        for i in xrange(1, len(datas)):
            data4d = np.asarray([datas[0], datas[i]])
            #del datas               # free up memory because above made a copy
            kwargs = kwargs_orig.copy()
            if not 'labels' in kwargs:
                kwargs['labels'] = unique_path_parts(files[0] + labels_add[0],
                                                     files[i] + labels_add[i])
            figs.append(plot_scatter(data4d, mask=mask,
                                     masked_opacity=masked_opacity, # TODO: **kwargs
                                     thresholds=thresholds,
                                     **kwargs))
    elif style in ('full', 'upper_triang'):
        if not 'labels' in kwargs:
            kwargs['labels'] = unique_path_parts(*[f+l for f,l in zip(files, labels_add)])

        figs = [plot_scatter_matrix(datas,
                                    style=style,
                                    mask=mask,
                                    masked_opacity=masked_opacity,
                                    thresholds=thresholds,
                                    **kwargs)]
    else:
        raise ValueError("Unknown style %s" % style)

    figs = figs[0] if len(figs) == 1 else figs
    if return_data:
        return figs, datas
    else:
        return figs
