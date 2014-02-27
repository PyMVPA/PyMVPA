# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Basic ERP (here ERP = Event Related Plot ;-)) plotting

Can be used for plotting not only ERP but any event-locked data
"""

import pylab as pl
import numpy as np
import matplotlib as mpl

from mvpa2.base import warning
from mvpa2.mappers.boxcar import BoxcarMapper

#
# Few helper functions
#
import matplotlib.transforms as mlt
def _offset(ax, x, y):
    """Provide offset in pixels

    Parameters
    ----------
    x : int
      Offset in pixels for x
    y : int
      Offset in pixels for y

    Idea borrowed from
     http://www.scipy.org/Cookbook/Matplotlib/Transformations
    but then heavily extended to be compatible with many
    reincarnations of matplotlib
    """
    d = dir(mlt)
    if 'offset_copy' in d:
        # tested with python-matplotlib 0.98.3-5
        # ??? if pukes, might need to replace 2nd parameter from
        #     ax to ax.get_figure()
        return mlt.offset_copy(ax.transData, ax, x=x, y=y, units='dots')
    elif 'BlendedAffine2D' in d:
        # some newer versions of matplotlib
        return ax.transData + \
               mlt.Affine2D().translate(x, y)
    elif 'blend_xy_sep_transform' in d:
        trans = mlt.blend_xy_sep_transform(ax.transData, ax.transData)
        # Now we set the offset in pixels
        trans.set_offset((x, y), mlt.identity_transform())
        return trans
    else:
        raise RuntimeError, \
              "Lacking needed functions in matplotlib.transform " \
              "for _offset. Please upgrade"


def _make_centeredaxis(ax, loc, offset=5, ai=0, mult=1.0,
                       format='%4g', label=None, **props):
    """Plot an axis which is centered at loc (e.g. 0)

    Parameters
    ----------
    ax
     Axes from the figure
    loc
     Value to center at
    offset
     Relative offset (in pixels) for the labels
    ai : int
     Axis index: 0 for x, 1 for y
    mult
     Multiplier for the axis labels. ERPs for instance need to be
     inverted, thus labels too manually here since there is no easy
     way in matplotlib to invert an axis
    label : str or None
     If not -- put a label outside of the axis
    **props
     Given to underlying plotting functions

    Idea borrowed from
      http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net \
      /msg05669.html
    It sustained heavy refactoring/extension
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xlocs = [l for l in ax.xaxis.get_ticklocs()
            if l >= xmin and l <= xmax]
    ylocs = [l for l in ax.yaxis.get_ticklocs()
            if l >= ymin and l <= ymax]

    if ai == 0:
        hlocs = ylocs
        locs = xlocs
        vrange = [xmin, xmax]
        tdir = mpl.lines.TICKDOWN
        halignment = 'center'
        valignment = 'top'
        lhalignment = 'left'
        lvalignment = 'center'
        lx, ly = xmax, 0
        ticklength = ax.xaxis.get_ticklines()[0]._markersize
    elif ai == 1:
        hlocs = xlocs
        locs = ylocs
        vrange = [ymin, ymax]
        tdir = mpl.lines.TICKLEFT
        halignment = 'right'
        valignment = 'center'
        lhalignment = 'center'
        lvalignment = 'bottom'
        lx, ly = 0, ymax
        ticklength = ax.yaxis.get_ticklines()[0]._markersize
    else:
        raise ValueError, "Illegal ai=%s" % ai

    args = [ (locs, [loc] * len(locs)),
             (vrange, [loc, loc]),
             [locs, (loc,) * len(locs)]
             ]

    offset_abs = offset + ticklength
    if ai == 1:
        # invert
        args = [ [x[1], x[0]] for x in args ]
        # shift the tick labels labels
        trans = _offset(ax, -offset_abs, 0)
        transl = _offset(ax, 0, offset)
    else:
        trans = _offset(ax, 0, -offset_abs)
        transl = _offset(ax, offset, 0)

    tickline, = ax.plot(linestyle='', marker=tdir, *args[0], **props)
    axline, = ax.plot(*args[1], **props)

    tickline.set_clip_on(False)
    axline.set_clip_on(False)


    for i, l in enumerate(locs):
        if l == 0:                    # no origin label
            continue
        coor = [args[2][0][i], args[2][1][i], format % (mult * l)]
        ax.text(horizontalalignment=halignment,
                verticalalignment=valignment, transform=trans, *coor)


    if label is not None:
        ax.text(
            #max(args[2][0]), max(args[2][1]),
            lx, ly,
            label,
            horizontalalignment=lhalignment,
            verticalalignment=lvalignment, fontsize=14,
            # fontweight='bold',
            transform=transl)


##REF: Name was automagically refactored
def plot_erp(data, SR=500, onsets=None,
            pre=0.2, pre_onset=None, post=None, pre_mean=None,
            color='r', errcolor=None, errtype=None, ax=pl,
            ymult=1.0, *args, **kwargs):
    """Plot single ERP on existing canvas

    Parameters
    ----------
    data : 1D or 2D ndarray
      The data array can either be 1D (samples over time) or 2D
      (trials x samples). In the first case a boxcar mapper is used to
      extract the respective trial timecourses given a list of trial onsets.
      In the latter case, each row of the data array is taken as the EEG
      signal timecourse of a particular trial.
    onsets : list(int)
      List of onsets (in samples not in seconds).
    SR : int, optional
      Sampling rate (1/s) of the signal.
    pre : float, optional
      Duration (in seconds) to be plotted prior to onset.
    pre_onset : float or None
      If data is already in epochs (2D) then pre_onset provides information
      on how many seconds pre-stimulus were used to generate them. If None,
      then pre_onset = pre
    post : float
      Duration (in seconds) to be plotted after the onset.
    pre_mean : float
      Duration (in seconds) at the beginning of the window which is used
      for deriving the mean of the signal. If None, pre_mean = pre. If 0,
      then the mean is not subtracted from the signal.
    errtype : None or 'ste' or 'std' or 'ci95' or list of previous three
      Type of error value to be computed per datapoint.  'ste' --
      standard error of the mean, 'std' -- standard deviation 'ci95'
      -- 95% confidence interval (1.96 * ste), None -- no error margin
      is plotted (default)
      Optionally, multiple error types can be specified in a list. In that
      case all of them will be plotted.
    color : matplotlib color code, optional
      Color to be used for plotting the mean signal timecourse.
    errcolor : matplotlib color code
      Color to be used for plotting the error margin. If None, use main color
      but with weak alpha level
    ax :
      Target where to draw.
    ymult : float, optional
      Multiplier for the values. E.g. if negative-up ERP plot is needed:
      provide ymult=-1.0
    *args, **kwargs
      Additional arguments to `pylab.plot`.

    Returns
    -------
    array
      Mean ERP timeseries.
    """
    if pre_mean is None:
        pre_mean = pre

    # set default
    pre_discard = 0

    if onsets is not None: # if we need to extract ERPs
        if post is None:
            raise ValueError, \
                  "Duration post onsets must be provided if onsets are given"
        # trial timecourse duration
        duration = pre + post

        # We are working with a full timeline
        bcm = BoxcarMapper(onsets,
                           boxlength=int(SR * duration),
                           offset= -int(SR * pre))
        erp_data = bcm(data)

        # override values since we are using Boxcar
        pre_onset = pre
    else:
        if pre_onset is None:
            pre_onset = pre

        if pre_onset < pre:
            warning("Pre-stimulus interval to plot %g is smaller than provided "
                    "pre-stimulus captured interval %g, thus plot interval was "
                    "adjusted" % (pre, pre_onset))
            pre = pre_onset

        if post is None:
            # figure out post
            duration = float(data.shape[1]) / SR - pre_discard
            post = duration - pre
        else:
            duration = pre + post

        erp_data = data
        pre_discard = pre_onset - pre

    # Scale the data appropriately
    erp_data *= ymult

    # validity check -- we should have 2D matrix (trials x samples)
    if len(erp_data.shape) != 2:
        raise RuntimeError, \
              "plot_erp() supports either 1D data with onsets, or 2D data " \
              "(trials x sample_points). Shape of the data at the point " \
              "is %s" % erp_data.shape

    if not (pre_mean == 0 or pre_mean is None):
        # mean of pre-onset signal accross trials
        erp_baseline = np.mean(
            erp_data[:, int((pre_onset - pre_mean) * SR):int(pre_onset * SR)])
        # center data on pre-onset mean
        # NOTE: make sure that we make a copy of the data to don't
        #       alter the original. Better be safe than sorry
        erp_data = erp_data - erp_baseline

    # generate timepoints and error ranges to plot filled error area
    # top ->
    # bottom <-
    time_points = np.arange(erp_data.shape[1]) * 1.0 / SR - pre_onset

    # if pre != pre_onset
    if pre_discard > 0:
        npoints = int(pre_discard * SR)
        time_points = time_points[npoints:]
        erp_data = erp_data[:, npoints:]

    # select only time points of interest (if post is provided)
    if post is not None:
        npoints = int(duration * SR)
        time_points = time_points[:npoints]
        erp_data = erp_data[:, :npoints]

    # compute mean signal timecourse accross trials
    erp_mean = np.mean(erp_data, axis=0)

    # give sane default
    if errtype is None:
        errtype = []
    if not isinstance(errtype, list):
        errtype = [errtype]

    for et in errtype:
        # compute error per datapoint
        if et in ['ste', 'ci95']:
            erp_stderr = erp_data.std(axis=0) / np.sqrt(len(erp_data))
            if et == 'ci95':
                erp_stderr *= 1.96
        elif et == 'std':
            erp_stderr = erp_data.std(axis=0)
        else:
            raise ValueError, "Unknown error type '%s'" % errtype

        time_points2w = np.hstack((time_points, time_points[::-1]))

        error_top = erp_mean + erp_stderr
        error_bottom = erp_mean - erp_stderr
        error2w = np.hstack((error_top, error_bottom[::-1]))

        if errcolor is None:
            errcolor = color

        # plot error margin
        pfill = ax.fill(time_points2w, error2w,
                        edgecolor=errcolor, facecolor=errcolor, alpha=0.2,
                        zorder=3)

    # plot mean signal timecourse
    ax.plot(time_points, erp_mean, lw=2, color=color, zorder=4,
            *args, **kwargs)
#    ax.xaxis.set_major_locator(pl.MaxNLocator(4))
    return erp_mean


##REF: Name was automagically refactored
def plot_erps(erps, data=None, ax=None, pre=0.2, post=None,
             pre_onset=None,
             xlabel='time (s)', ylabel='$\mu V$',
             ylim=None, ymult=1.0, legend=None,
             xlformat='%4g', ylformat='%4g',
             loffset=10, alinewidth=2,
             **kwargs):
    """Plot multiple ERPs on a new figure.

    Parameters
    ----------
    erps : list of tuples
      List of definitions of ERPs. Each tuple should consist of
      (label, color, onsets) or a dictionary which defines,
      label, color, onsets, data. Data provided in dictionary overrides
      'common' data provided in the next argument `data`
    data
      Data for ERPs to be derived from 1D (samples)
    ax
      Where to draw (e.g. subplot instance). If None, new figure is
      created
    pre : float, optional
      Duration (seconds) to be plotted prior to onset
    pre_onset : None or float
      If data is already in epochs (2D) then pre_onset provides information
      on how many seconds pre-stimulus were used to generate them. If None,
      then pre_onset = pre
    post : None or float
      Duration (seconds) to be plotted after the onset. If any data is
      provided with onsets, it can't be None. If None -- plots all time
      points after onsets
    ymult : float, optional
      Multiplier for the values. E.g. if negative-up ERP plot is needed:
      provide ymult=-1.0
    xlformat : str, optional
      Format of the x ticks
    ylformat : str, optional
      Format of the y ticks
    legend : None or string
      If not None, legend will be plotted with position argument
      provided in this argument
    loffset : int, optional
      Offset in voxels for axes and tick labels. Different
      matplotlib frontends might have different opinions, thus
      offset value might need to be tuned specifically per frontend
    alinewidth : int, optional
      Axis and ticks line width
    **kwargs
      Additional arguments provided to plot_erp()


    Examples
    --------

    ::

      kwargs  = {'SR' : eeg.SR, 'pre_mean' : 0.2}
      fig = plot_erps((('60db', 'b', eeg.erp_onsets['60db']),
                       ('80db', 'r', eeg.erp_onsets['80db'])),
                      data[:, eeg.sensor_mapping['Cz']],
                      ax=fig.add_subplot(1,1,1,frame_on=False), pre=0.2,
                      post=0.6, **kwargs)

    or

    ::
    
        fig = plot_erps((('60db', 'b', eeg.erp_onsets['60db']),
                          {'color': 'r',
                           'onsets': eeg.erp_onsets['80db'],
                           'data' : data[:, eeg.sensor_mapping['Cz']]}
                         ),
                        data[:, eeg.sensor_mapping['Cz']],
                        ax=fig.add_subplot(1,1,1,frame_on=False), pre=0.2,
                        post=0.6, **kwargs)

    Returns
    -------
    h
      current fig handler
    """

    if ax is None:
        fig = pl.figure(facecolor='white')
        fig.clf()
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig = pl.gcf()

    # We don't want original axis being on
    ax.axison = True

    labels = []
    for erp_def in erps:
        plot_data = data
        params = {'ymult' : ymult}

        # provide custom parameters per ERP
        if isinstance(erp_def, tuple) and len(erp_def) == 3:
            params.update(
                {'label': erp_def[0],
                 'color': erp_def[1],
                 'onsets': erp_def[2]})
        elif isinstance(erp_def, dict):
            plot_data = erp_def.pop('data', None)
            params.update(erp_def)

        labels.append(params.get('label', ''))

        # absorb common parameters
        params.update(kwargs)

        if plot_data is None:
            raise ValueError, "Channel %s got no data provided" \
                  % params.get('label', 'UNKNOWN')


        plot_erp(plot_data, pre=pre, pre_onset=pre_onset, post=post, ax=ax,
                **params)
        #             plot_kwargs={'label':label})

        if isinstance(erp_def, dict):
            erp_def['data'] = plot_data # return it back

    props = dict(color='black',
                 linewidth=alinewidth, markeredgewidth=alinewidth,
                 zorder=1, offset=loffset)

    def set_limits():
        """Helper to set x and y limits"""
        ax.set_xlim((-pre, post))
        if ylim != None:
            ax.set_ylim(*ylim)

    set_limits()
    _make_centeredaxis(ax, 0, ai=0, label=xlabel, **props)
    set_limits()
    _make_centeredaxis(ax, 0, ai=1, mult=np.sign(ymult), label=ylabel, **props)

    ax.yaxis.set_major_locator(pl.NullLocator())
    ax.xaxis.set_major_locator(pl.NullLocator())

    # legend obscures plotting a bit... seems to be plotting
    # everything twice. Thus disabled by default
    if legend is not None and np.any(np.array(labels) != ''):
        pl.legend(labels, loc=legend)

    fig.canvas.draw()
    return fig

