#emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
#ex: set sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Basic ERP (here ERP = Event Related Plot ;-)) plotting"""

import pylab as P
import numpy as N
import matplotlib as mpl

from mvpa.mappers.boxcar import BoxcarMapper

#
# Original code was borrowed from
# http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net \
#   /msg05669.html
# It sustained heavy refactoring/extension
#

def _make_centeredaxis(ax, loc, offset=0.5, ai=0, mult=1.0, **props):
    """Plot an axis which is centered at loc (e.g. 0)

    :Parameters:
     ax
       Axes from the figure
     loc
       Value to center at
     offset
       ralative offset for the labels
     mult
       multiplier for the axis labels. ERPs for instance need to be
       inverted, thus labels too manually here since there is no easy
       way in matplotlib to invert an axis
     **props
       given to underlying plotting functions
    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    xlocs = [l for l in ax.xaxis.get_ticklocs()
            if l>=xmin and l<=xmax]
    ylocs = [l for l in ax.yaxis.get_ticklocs()
            if l>=ymin and l<=ymax]

    if ai == 0:
        hlocs = ylocs
        locs = xlocs
        vrange = [xmin, xmax]
        tdir = mpl.lines.TICKDOWN
        horizontalalignment = 'center'
        verticalalignment = 'top'
    elif ai == 1:
        hlocs = xlocs
        locs = ylocs
        vrange = [ymin, ymax]
        tdir = mpl.lines.TICKLEFT
        horizontalalignment = 'right'
        verticalalignment = 'center'
    else:
        raise ValueError, "Illegal ai=%s" % ai

    # absolute offset
    offset_abs = offset * float(hlocs[1]-hlocs[0])

    args = [ (locs, [loc]*len(locs)),
             (vrange, [loc, loc]),
             [locs, (loc-offset_abs,)*len(locs)]]

    if ai == 1:
        # invert
        args = [ [x[1], x[0]] for x in args ]

    tickline, = ax.plot(linestyle='', marker=tdir, *args[0], **props)
    axline, = ax.plot(*args[1], **props)

    tickline.set_clip_on(False)
    axline.set_clip_on(False)

    for i,l in enumerate(locs):
        if l == 0:                    # no origin label
            continue
        coor = [args[2][0][i], args[2][1][i], '%1.1f'%(mult*l)]
        ax.text(horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment, *coor)


def plotERP(data, SR=500, onsets=None, pre=0.2, post=0.6, pre_mean=0.2,
            color='r', errcolor=None, errtype='ste', ax=P,
            *args, **kwargs):
    """Plot single ERP on existing canvas

    :Parameters:
      data: 1D or 2D ndarray
        The data array can either be 1D (samples over time) or 2D
        (trials x samples). In the first case a boxcar mapper is used to
        extract the respective trial timecourses given a list of trial onsets.
        In the latter case, each row of the data array is taken as the EEG
        signal timecourse of a particular trial.
      onsets: list(int)
        List of onsets (in samples not in seconds).
      SR: int
        Sampling rate (1/s) of the signal.
      pre: float
        Duration (in seconds) to be plotted prior to onset.
      post: float
        Duration (in seconds) to be plotted after the onset.
      pre_mean: float
        Duration (in seconds) at the beginning of the window which is used
        for deriving the mean of the signal.
      errtype: 'ste' | 'std'
        Type of error value to be computed per datapoint.
          'ste': standard error of the mean
          'std': standard deviation
      color: matplotlib color code
        Color to be used for plotting the mean signal timecourse.
      errcolor: matplotlib color code
        Color to be used for plotting the error margin. If None, use main color
        but with weak alpha level
      ax:
        Target where to draw.
      *args, **kwargs
        Additional arguments to plot().
    """
    # trial timecourse duration
    duration = pre + post

    if onsets is not None: # if we need to extract ERPs
        # We are working with a full timeline
        bcm = BoxcarMapper(onsets,
                           boxlength = int(SR * duration),
                           offset = -int(SR * pre))
        erp_data = bcm(data)
    else:
        erp_data = data

    # validity check -- we should have 2D matrix (trials x samples)
    if len(erp_data.shape) != 2:
        raise RuntimeError, \
              "plotERP() supports either 1D data with onsets, or 2D data " \
              "(trials x sample_points). Shape of the data at the point " \
              "is %s" % erp_data.shape

    # mean of pre-onset signal accross trials
    erp_baseline = N.mean(erp_data[:,:int(pre_mean*SR)])
    # center data on pre-onset mean
    erp_data -= erp_baseline
    # compute mean signal timecourse accross trials
    erp_mean = N.mean(erp_data, axis=0)

    # compute error per datapoint
    if errtype == 'ste':
        erp_stderr = erp_data.std(axis=0) / N.sqrt(len(erp_data))
    elif errtype == 'std':
        erp_stderr = erp_data.std(axis=0)
    else:
        raise ValueError, "Unknown error type '%s'" % errtype

    # generate timepoints and error ranges to plot filled error area
    # top ->
    # bottom <-
    time_points = N.arange(len(erp_mean)) * 1.0 / SR - pre
    time_points2w = N.hstack((time_points, time_points[::-1]))

    error_top = -erp_mean-erp_stderr
    error_bottom = -erp_mean+erp_stderr
    error2w = N.hstack((error_top, error_bottom[::-1]))

    if errcolor is None:
        errcolor = color

    # plot error margin
    pfill = ax.fill(time_points2w, error2w,
                    facecolor=errcolor, alpha=0.2,
                    zorder=3)

    # plot mean signal timecourse
    ax.plot(time_points, -erp_mean, lw=2, color=color, zorder=4,
            *args, **kwargs)


def plotERPs(erps, data=None, ax=None, pre=0.2, post=0.6,
             xlabel = 'time (s)', ylabel='$\mu V$',
             ylim=None, **kwargs):
    """Plot multiple ERPs on a new figure.

    :Parameters:
      erps : list of tuples
        List of definitions of ERPs. Each tuple should consist of
        (label, color, onsets) or a dictionary which defines,
        label, color, onsets, data. Data provided in dictionary overrides
        'common' data provided in the next argument ``data``
      data
        Data for ERPs to be derived from 1D (samples)
      ax
        Where to draw (e.g. subplot instance). If None, new figure is
        created
      pre
        duration (seconds) to be plotted prior to onset
      post
        duration (seconds) to be plotted after the onset
      **kwargs
        Additional arguments provided to plot_erp


    :Examples:
        kwargs  = {'SR' : eeg.SR, 'pre_mean' : 0.2}
        fig = plot_erps((('60db', 'b', eeg.erp_onsets['60db']),
                         ('80db', 'r', eeg.erp_onsets['80db'])),
                        data[:, eeg.sensor_mapping['Cz']],
                        ax=fig.add_subplot(1,1,1,frame_on=False), pre=0.2,
                        post=0.6, **kwargs)

        or
        fig = plot_erps((('60db', 'b', eeg.erp_onsets['60db']),
                          {'color': 'r',
                           'onsets': eeg.erp_onsets['80db'],
                           'data' : data[:, eeg.sensor_mapping['Cz']]}
                         ),
                        data[:, eeg.sensor_mapping['Cz']],
                        ax=fig.add_subplot(1,1,1,frame_on=False), pre=0.2,
                        post=0.6, **kwargs)

    :Returns: current fig handler
    """

    props = dict(color='black', linewidth=2, markeredgewidth=2, zorder=1)

    if ax is None:
        fig = P.figure(facecolor='white')
        fig.clf()
        ax = fig.add_subplot(111, frame_on=False)
    else:
        fig = P.gcf()

    ax.axison = True

    for erp_def in erps:
        plot_data = data
        color = None
        onsets = None

        if isinstance(erp_def, tuple) and len(erp_def) == 3:
            label, color, onsets = erp_def
        elif isinstance(erp_def, dict):
            label = erp_def.get('label', None)
            plot_data = erp_def.get('data', None)
            onsets = erp_def.get('onsets', None)
            color = erp_def.get('color', None)

        if plot_data is None:
            raise ValueError, "Channel %s got not data provided" % label

        plotERP(plot_data,
                onsets=onsets, pre=pre, post=post,
                color=color, ax=ax,
                **kwargs)
        #             plot_kwargs={'label':label})

    # legend obscures plotting a bit... disabled for now
    #P.legend([x[0] for x in erps], loc='best')

    if xlabel is not None:
        P.xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        P.ylabel(ylabel,  fontsize=16)

    _make_centeredaxis(ax, 0, offset=0.3, ai=0, **props)
    _make_centeredaxis(ax, 0, offset=0.3, ai=1, mult=-1.0, **props)

    ax.yaxis.set_major_locator(P.NullLocator())
    ax.xaxis.set_major_locator(P.NullLocator())
    ax.set_xlim( (-pre, post) )
    if ylim != None:
        ax.set_ylim(*ylim)
    fig.canvas.draw()
    return fig

