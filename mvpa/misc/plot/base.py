# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Misc. plotting helpers."""

__docformat__ = 'restructuredtext'

import pylab as P
import numpy as N

from mvpa.datasets.splitters import NFoldSplitter
from mvpa.clfs.distance import squared_euclidean_distance
from mvpa.datasets.miscfx import get_samples_by_attr


##REF: Name was automagically refactored
def plot_err_line(data, x=None, errtype='ste', curves=None, linestyle='--',
                fmt='o', perc_sigchg=False, baseline=None):
    """Make a line plot with errorbars on the data points.

    Parameters
    ----------
    data : sequence of sequences
      First axis separates samples and second axis will appear as
      x-axis in the plot.
    x : sequence
      Value to be used as 'x-values' corresponding to the elements of
      the 2nd axis id `data`. If `None`, a sequence of ascending integers
      will be generated.
    errtype : 'ste' or 'std'
      Type of error value to be computed per datapoint: 'ste' --
      standard error of the mean, 'std' -- standard deviation.
    curves : None or list of tuple(x, y)
      Each tuple represents an additional curve, with x and y coordinates of
      each point on the curve.
    linestyle : str
      matplotlib linestyle argument. Applied to either the additional
      curve or a the line connecting the datapoints. Set to 'None' to
      disable the line completely.
    fmt : str
      matplotlib plot style argument to be applied to the data points
      and errorbars.
    perc_sigchg : bool
      If `True` the plot will show percent signal changes relative to a
      baseline.
    baseline : float or None
      Baseline used for converting values into percent signal changes.
      If `None` and `perc_sigchg` is `True`, the absolute of the mean of the
      first feature (i.e. [:,0]) will be used as a baseline.


    Examples
    --------

    Make dataset with 20 samples from a full sinus wave period,
    computed 100 times with individual noise pattern.

        >>> x = N.linspace(0, N.pi * 2, 20)
        >>> data = N.vstack([N.sin(x)] * 30)
        >>> data += N.random.normal(size=data.shape)

      Now, plot mean data points with error bars, plus a high-res
      version of the original sinus wave.

        >>> x = N.linspace(0, N.pi * 2, 200)
        >>> plot_err_line(data, curves=[(x, N.sin(x))])
        >>> #P.show()
    """
    data = N.asanyarray(data)

    if len(data.shape) < 2:
        data = N.atleast_2d(data)

    # compute mean signal course
    md = data.mean(axis=0)

    if baseline is None:
        baseline = N.abs(md[0])

    if perc_sigchg:
        md /= baseline
        md -= 1.0
        md *= 100.0
        # not in-place to keep original data intact
        data = data / baseline
        data *= 100.0

    # compute matching datapoint locations on x-axis
    if x is None:
        x = N.arange(len(md))
    else:
        if not len(md) == len(x):
            raise ValueError, "The length of `x` (%i) has to match the 2nd " \
                              "axis of the data array (%i)" % (len(x), len(md))

    # plot highres line if present
    if curves is not None:
        for c in curves:
            xc, yc = c
            # scales line array to same range as datapoints
            P.plot(xc, yc, linestyle)

        # no line between data points
        linestyle = 'None'

    # compute error per datapoint
    if errtype == 'ste':
        err = data.std(axis=0) / N.sqrt(len(data))
    elif errtype == 'std':
        err = data.std(axis=0)
    else:
        raise ValueError, "Unknown error type '%s'" % errtype

    # plot datapoints with error bars
    P.errorbar(x, md, err, fmt=fmt, linestyle=linestyle)


##REF: Name was automagically refactored
def plot_feature_hist(dataset, xlim=None, noticks=True, perchunk=False,
                    **kwargs):
    """Plot histograms of feature values for each labels.

    Parameters
    ----------
    dataset : Dataset
    xlim : None or 2-tuple
      Common x-axis limits for all histograms.
    noticks : bool
      If True, no axis ticks will be plotted. This is useful to save
      space in large plots.
    perchunk : bool
      If True, one histogramm will be plotted per each label and each
      chunk, resulting is a histogram grid (labels x chunks).
    **kwargs
      Any additional arguments are passed to matplotlib's hist().
    """
    lsplit = NFoldSplitter(1, attr='targets')
    csplit = NFoldSplitter(1, attr='chunks')

    nrows = len(dataset.sa['targets'].unique)
    ncols = len(dataset.sa['chunks'].unique)

    def doplot(data):
        P.hist(data, **kwargs)

        if xlim is not None:
            P.xlim(xlim)

        if noticks:
            P.yticks([])
            P.xticks([])

    fig = 1

    # for all labels
    for row, (ignore, ds) in enumerate(lsplit(dataset)):
        if perchunk:
            for col, (alsoignore, d) in enumerate(csplit(ds)):

                P.subplot(nrows, ncols, fig)
                doplot(d.samples.ravel())

                if row == 0:
                    P.title('C:' + str(d.sa['chunks'].unique[0]))
                if col == 0:
                    P.ylabel('L:' + str(d.sa['targets'].unique[0]))

                fig += 1
        else:
            P.subplot(1, nrows, fig)
            doplot(ds.samples)

            P.title('L:' + str(ds.sa['targets'].unique[0]))

            fig += 1


##REF: Name was automagically refactored
def plot_samples_distance(dataset, sortbyattr=None):
    """Plot the euclidean distances between all samples of a dataset.

    Parameters
    ----------
    dataset : Dataset
      Providing the samples.
    sortbyattr : None or str
      If None, the samples distances will be in the same order as their
      appearance in the dataset. Alternatively, the name of a samples
      attribute can be given, which wil then be used to sort/group the
      samples, e.g. to investigate the similarity samples by label or by
      chunks.
    """
    if sortbyattr is not None:
        slicer = []
        for attr in dataset.sa[sortbyattr].unique:
            slicer += \
                get_samples_by_attr(dataset, sortbyattr, attr).tolist()
        samples = dataset.samples[slicer]
    else:
        samples = dataset.samples

    ed = N.sqrt(squared_euclidean_distance(samples))

    P.imshow(ed)
    P.colorbar()


def plot_decision_boundary_2d(dataset, clf=None,
                              targets=None, regions=None, maps=None,
                              maps_res=50, vals=[-1, 0, 1],
                              data_callback=None):
    """Plot a scatter of a classifier's decision boundary and data points

    Assumes data is 2d (no way to visualize otherwise!!)

    Parameters
    ----------
    dataset : `Dataset`
      Data points to visualize (might be the data `clf` was train on, or
      any novel data).
    clf : `Classifier`, optional
      Trained classifier
    targets : string, optional
      What samples attributes to use for targets.  If None and clf is
      provided, then `clf.params.targets` is used.
    regions : string, optional
      Plot regions (polygons) around groups of samples with the same
      attribute (and target attribute) values. E.g. chunks.
    maps : string in {'targets', 'estimates'}, optional
      Either plot underlying colored maps, such as clf predictions
      within the spanned regions, or estimates from the classifier
      (might not work for some).
    maps_res : int, optional
      Number of points in each direction to evaluate.
      Points are between axis limits, which are set automatically by
      matplotlib.  Higher number will yield smoother decision lines but come
      at the cost of O^2 classifying time/memory.
    vals : array of floats, optional
      Where to draw the contour lines if maps='estimates'
    data_callback : callable, optional
      Callable object to preprocess the new data points.
      Classified points of the form samples = data_callback(xysamples).
      I.e. this can be a function to normalize them, or cache them
      before they are classified.
    """
    #from mvpa.misc.attrmap import AttributeMap
    #from mvpa.misc.plot.tools import *

    if False:
        from mvpa.misc.data_generators import *
        from mvpa.clfs.svm import *
        from mvpa.clfs.knn import *
        #ds = dumb_feature_binary_dataset()
        dataset = normal_feature_dataset(nfeatures=2, nchunks=5,
                                         snr=10, nlabels=4, means=[ [0,1], [1,0], [1,1], [0,0] ])
        dataset.samples += dataset.sa.chunks[:, None]*0.1 # slight shifts for chunks ;)
        #dataset = normal_feature_dataset(nfeatures=2, nlabels=3, means=[ [0,1], [1,0], [1,1] ])
        #dataset = normal_feature_dataset(nfeatures=2, nlabels=2, means=[ [0,1], [1,0] ])
        #clf = LinearCSVMC(C=-1)
        clf = kNN(4)#LinearCSVMC(C=-1)
        clf.train(dataset)
        #clf = None
        #plot_decision_boundary_2d(ds, clf)
        targets = 'targets'
        regions = 'chunks'
        #maps = 'estimates'
        maps = 'targets'
        #maps = None #'targets'
        res = 50
        vals = [-1, 0, 1]
        data_callback=None
        P.clf()

    if dataset.nfeatures != 2:
        raise ValueError('Can only plot a decision boundary in 2D')

    Pioff()
    a = P.gca() # f.add_subplot(1,1,1)

    attrmap = None
    if clf:
        estimates_were_enabled = clf.ca.is_enabled('estimates')
        clf.ca.enable('estimates')

        if targets is None:
            targets = clf.params.targets
        # Lets reuse classifiers attrmap if it is good enough
        attrmap = clf._attrmap
        predictions = clf.predict(dataset)

    targets_sa_name = targets           # bad Yarik -- will rebind targets to actual values
    targets_lit = dataset.sa[targets_sa_name].value
    utargets_lit = dataset.sa[targets_sa_name].unique

    if not (attrmap is not None
            and len(attrmap)
            and set(clf._attrmap.keys()).issuperset(utargets_lit)):
        # create our own
        attrmap = AttributeMap(mapnumeric=True)

    targets = attrmap.to_numeric(targets_lit)
    utargets = attrmap.to_numeric(utargets_lit)

    vmin = min(utargets)
    vmax = max(utargets)
    cmap = P.cm.RdYlGn                  # argument

    # Scatter points
    if clf:
        all_hits = predictions == targets_lit
    else:
        all_hits = N.ones((len(targets),), dtype=bool)

    targets_colors = {}
    for l in utargets:
        targets_mask = targets==l
        s = dataset[targets_mask]
        targets_colors[l] = c \
            = cmap((l-vmin)/float(vmax-vmin))

        # We want to plot hits and misses with different symbols
        hits = all_hits[targets_mask]
        misses = N.logical_not(hits)
        scatter_kwargs = dict(
            c=[c], zorder=10+(l-vmin))

        if sum(hits):
            a.scatter(s.samples[hits, 0], s.samples[hits, 1], marker='o',
                      label='%s [%d]' % (attrmap.to_literal(l), sum(hits)),
                      **scatter_kwargs)
        if sum(misses):
            a.scatter(s.samples[misses, 0], s.samples[misses, 1], marker='x',
                      label='%s [%d] (miss)' % (attrmap.to_literal(l), sum(misses)),
                      edgecolor=[c], **scatter_kwargs)

    (xmin, xmax) = a.get_xlim()
    (ymin, ymax) = a.get_ylim()
    extent = (xmin, xmax, ymin, ymax)

    # Create grid to evaluate, predict it
    (x,y) = N.mgrid[xmin:xmax:N.complex(0, res),
                    ymin:ymax:N.complex(0, res)]
    news = N.vstack((x.ravel(), y.ravel())).T
    try:
        news = data_callback(news)
    except TypeError: # Not a callable object
        pass

    imshow_kwargs = dict(origin='lower',
            zorder=1,
            aspect='auto',
            interpolation='bilinear', alpha=0.9, cmap=cmap,
            vmin=vmin, vmax=vmax,
            extent=extent)

    if maps is not None:
        if clf is None:
            raise ValueError, \
                  "Please provide classifier for plotting maps of %s" % maps
        predictions_new = clf.predict(news)

    if maps == 'estimates':
        # Contour and show predictions
        trained_targets = attrmap.to_numeric(clf.ca.trained_targets)

        if len(trained_targets)==2:
            linestyles = []
            for v in vals:
                if v == 0:
                    linestyles.append('solid')
                else:
                    linestyles.append('dashed')
            vmin, vmax = -3, 3 # Gives a nice tonal range ;)
            map_ = 'estimates' # should actually depend on estimates
        else:
            vals = (trained_targets[:-1] + trained_targets[1:])/2.
            linestyles = ['solid'] * len(vals)
            map_ = 'targets'

        try:
            clf.ca.estimates.reshape(x.shape)
            a.imshow(map_values.T, **imshow_kwargs)
            CS = a.contour(x, y, map_values, vals, zorder=6,
                           linestyles=linestyles, extent=extent, colors='k')
        except ValueError, e:
            print "Sorry - plotting of estimates isn't full supported for %s. " \
                  "Got exception %s" % (clf, e)
    elif maps == 'targets':
        map_values = attrmap.to_numeric(predictions_new).reshape(x.shape)
        a.imshow(map_values.T, **imshow_kwargs)
        #CS = a.contour(x, y, map_values, vals, zorder=6,
        #               linestyles=linestyles, extent=extent, colors='k')

    # Plot regions belonging to the same pair of attribute given
    # (e.g. chunks) and targets attribute
    if regions:
        chunks_sa = dataset.sa[regions]
        chunks_lit = chunks_sa.value
        uchunks_lit = chunks_sa.value
        chunks_attrmap = AttributeMap(mapnumeric=True)
        chunks = chunks_attrmap.to_numeric(chunks_lit)
        uchunks = chunks_attrmap.to_numeric(uchunks_lit)

        from matplotlib.delaunay.triangulate import Triangulation
        from matplotlib.patches import Polygon
        # Lets figure out convex halls for each chunk/label pair
        for target in utargets:
            t_mask = targets == target
            for chunk in uchunks:
                tc_mask = N.logical_and(t_mask,
                                        chunk == chunks)
                tc_samples = dataset.samples[tc_mask]
                tr = Triangulation(tc_samples[:, 0],
                                   tc_samples[:, 1])
                poly = P.fill(tc_samples[tr.hull, 0],
                              tc_samples[tr.hull, 1],
                              closed=True,
                              facecolor=targets_colors[target],
                              #fill=False,
                              alpha=0.01,
                              edgecolor='gray',
                              linestyle='dotted',
                              linewidth=0.5,
                              )

    P.legend(scatterpoints=1)
    if clf and not estimates_were_enabled:
        clf.ca.disable('estimates')
    Pion()
    P.axis('tight')
    #P.show()

##REF: Name was automagically refactored
def plot_bars(data, labels=None, title=None, ylim=None, ylabel=None,
               width=0.2, offset=0.2, color='0.6', distance=1.0,
               yerr='ste', xloc=None, **kwargs):
    """Make bar plots with automatically computed error bars.

    Candlestick plot (multiple interleaved barplots) can be done,
    by calling this function multiple time with appropriatly modified
    `offset` argument.

    Parameters
    ----------
    data : array (nbars x nobservations) or other sequence type
      Source data for the barplot. Error measure is computed along the
      second axis.
    labels : list or None
      If not None, a label from this list is placed on each bar.
    title : str
      An optional title of the barplot.
    ylim : 2-tuple
      Y-axis range.
    ylabel : str
      An optional label for the y-axis.
    width : float
      Width of a bar. The value should be in a reasonable relation to
      `distance`.
    offset : float
      Constant offset of all bar along the x-axis. Can be used to create
      candlestick plots.
    color : matplotlib color spec
      Color of the bars.
    distance : float
      Distance of two adjacent bars.
    yerr : {'ste', 'std', None}
      Type of error for the errorbars. If `None` no errorbars are plotted.
    xloc : sequence
      Locations of the bars on the x axis.
    **kwargs
      Any additional arguments are passed to matplotlib's `bar()` function.
    """
    # determine location of bars
    if xloc is None:
        xloc = (N.arange(len(data)) * distance) + offset

    if yerr == 'ste':
        yerr = [N.std(d) / N.sqrt(len(d)) for d in data]
    elif yerr == 'std':
        yerr = [N.std(d) for d in data]
    else:
        # if something that we do not know just pass on
        pass

    # plot bars
    plot = P.bar(xloc,
                 [N.mean(d) for d in data],
                 yerr=yerr,
                 width=width,
                 color=color,
                 ecolor='black',
                 **kwargs)

    if ylim:
        P.ylim(*(ylim))
    if title:
        P.title(title)

    if labels:
        P.xticks(xloc + width / 2, labels)

    if ylabel:
        P.ylabel(ylabel)

    # leave some space after last bar
    P.xlim(0, xloc[-1] + width + offset)

    return plot


##REF: Name was automagically refactored
def inverse_cmap(cmap_name):
    """Create a new colormap from the named colormap, where it got reversed

    """
    import matplotlib._cm as _cm
    import matplotlib as mpl
    try:
        cmap_data = eval('_cm._%s_data' % cmap_name)
    except:
        raise ValueError, "Cannot obtain data for the colormap %s" % cmap_name
    new_data = dict( [(k, [(v[i][0], v[-(i+1)][1], v[-(i+1)][2])
                           for i in xrange(len(v))])
                      for k,v in cmap_data.iteritems()] )
    return mpl.colors.LinearSegmentedColormap('%s_rev' % cmap_name,
                                              new_data, _cm.LUTSIZE)


##REF: Name was automagically refactored
def plot_dataset_chunks(ds, clf_labels=None):
    """Quick plot to see chunk sctructure in dataset with 2 features

    if clf_labels is provided for the predicted labels, then
    incorrectly labeled samples will have 'x' in them
    """
    if ds.nfeatures != 2:
        raise ValueError, "Can plot only in 2D, ie for datasets with 2 features"
    if P.matplotlib.get_backend() == 'TkAgg':
        P.ioff()
    if clf_labels is not None and len(clf_labels) != ds.nsamples:
        clf_labels = None
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
    labels = ds.uniquetargets
    labels_map = dict(zip(labels, colors[:len(labels)]))
    for chunk in ds.uniquechunks:
        chunk_text = str(chunk)
        ids = ds.where(chunks=chunk)
        ds_chunk = ds[ids]
        for i in xrange(ds_chunk.nsamples):
            s = ds_chunk.samples[i]
            l = ds_chunk.targets[i]
            format = ''
            if clf_labels != None:
                if clf_labels[i] != ds_chunk.targets[i]:
                    P.plot([s[0]], [s[1]], 'x' + labels_map[l])
            P.text(s[0], s[1], chunk_text, color=labels_map[l],
                   horizontalalignment='center',
                   verticalalignment='center',
                   )
    dss = ds.samples
    P.axis((1.1 * N.min(dss[:, 0]),
            1.1 * N.max(dss[:, 1]),
            1.1 * N.max(dss[:, 0]),
            1.1 * N.min(dss[:, 1])))
    P.draw()
    P.ion()

