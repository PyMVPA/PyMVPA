# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Basic (f)MRI plotting with ability to interactively perform thresholding

"""

import pylab as pl
import numpy as np
import matplotlib as mpl

from mvpa2.base import warning, externals
from mvpa2.misc.plot.tools import Pion, Pioff, mpl_backend_isinteractive

if __debug__:
    from mvpa2.base import debug

if externals.exists('nibabel'):
    import nibabel as nb
    from nibabel.spatialimages import SpatialImage
else:
    class SpatialImage(object):
        """Just a helper to allow plot_lightbox be used even if no
        nibabel module available for plotting regular 2D/3D images
        (ndarrays)"""
        def __init__(self, filename):
            raise ValueError, "plot_lightbox was provided a filename %s.  " \
                  "By now we only support loading data from Nifti/Analyze " \
                  "files, but nibabel module is not available" % filename


def plot_lightbox(background=None, background_mask=None, cmap_bg='gray',
            overlay=None, overlay_mask=None, cmap_overlay='autumn',
            vlim=(0.0, None), vlim_type=None,
            do_stretch_colors=False,
            add_info=True, add_hist=True, add_colorbar=True,
            fig=None, interactive=None,
            nrows=None, ncolumns=None,
            slices=None, slice_title="k=%(islice)s"
            ):
    """Very basic plotting of 3D data with interactive thresholding.

    `background`/`overlay` and corresponding masks could be nifti
    files names or `SpatialImage` objects, or 3D `ndarrays`. If no mask
    is provided, only non-0 elements are plotted.

    Notes
    -----
    No care is taken to deduce the orientation (e.g. Left-to-Right,
    Posterior-to-Anterior) of fMRI volumes.  Therefore all input
    volumes should be in the same orientation.

    Parameters
    ----------
    do_stretch_colors : bool, optional
      Stratch color range to the data (not just to visible data)
    vlim : tuple, optional
      2 element tuple of low/upper bounds of values to plot
    vlim_type : None or 'symneg_z'
      If not None, then vlim would be treated accordingly:
       symneg_z
         z-score values of symmetric normal around 0, estimated
         by symmetrizing negative part of the distribution, which
         often could be assumed when total distribution is a mixture of
         by-chance performance normal around 0, and some other in the
         positive tail
    ncolumns : int or None
      Explicit starting number of columns into which position the
      slice renderings.
      If None, square arrangement would be used
    nrows : int or None
      Explicit starting number of rows into which position the
      slice renderings.
      If None, square arrangement would be used
    add_hist : bool or tuple (int, int)
      If True, add histogram and position automagically.
      If a tuple -- use as (row, column)
    add_info : bool or tuple (int, int)
      If True, add information and position automagically.
      If a tuple -- use as (row, column).
    slices : None or list of int
      If not to plot whole volume, what slices to plot.
    slice_title : None or str
      Desired title of slices.  Use string comprehension and assume
      `islice` variable present with current slice index.

    Available colormaps are presented nicely on
      http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps

    TODO:
      * Make interface more attractive/usable
      * allow multiple overlays... or just unify for them all to be just a list of entries
      * handle cases properly when there is only one - background/overlay
    """

    def handle_arg(arg):
        """Helper which would read in SpatialImage if necessary
        """
        if arg is None:
            return arg
        if isinstance(arg, basestring):
            arg = nb.load(arg)
            argshape = arg.shape
            # Assure that we have 3D (at least)
            if len(argshape)<3:
                arg = nb.Nifti1Image(
                        arg.get_data().reshape(argshape + (1,)*(3-len(argshape))),
                        arg.affine, arg.header)
        else:
            argshape = arg.shape

        if len(argshape) == 4:
            if argshape[-1] > 1:
                warning("For now plot_lightbox can handle only 3d, 4d data was provided."
                        " Plotting only the first volume")
            if isinstance(arg, SpatialImage):
                arg = nb.Nifti1Image(arg.get_data()[..., 0], arg.affine, arg.header)
            else:
                arg = arg[..., 0]
        elif len(argshape) != 3:
            raise ValueError, "For now just handling 3D volumes"
        return arg

    bg = handle_arg(background)
    if isinstance(bg, SpatialImage):
        # figure out aspect
        # fov = (np.array(bg.header['pixdim']) * bg.header['dim'])[3:0:-1]
        # aspect = fov[1]/fov[2]
        # just scale by voxel-size ratio (extent is disabled)
        bg_hdr = bg.header
        aspect = bg_hdr.get_zooms()[2] / bg_hdr.get_zooms()[1]

        bg = bg.get_data()
    else:
        aspect = 1.0

    bg_mask = None
    if bg is not None:
        bg_mask = handle_arg(background_mask)
        if isinstance(bg_mask, SpatialImage):
            bg_mask = bg_mask.get_data()
        if bg_mask is not None:
            bg_mask = bg_mask != 0
        else:
            bg_mask = bg != 0

    func = handle_arg(overlay)

    if func is not None:
        if isinstance(func, SpatialImage):
            func = func.get_data()

        func_mask = handle_arg(overlay_mask)
        if isinstance(func_mask, SpatialImage):
            func_mask = func_mask.get_data() #[..., ::-1, :] # XXX
        if func_mask is not None:
            func_mask = func_mask != 0
        else:
            func_mask = func != 0

    # Lets assure that that we are dealing with <= 3D
    for v in (bg, bg_mask, func, func_mask):
        if v is not None:
            if v.ndim > 3:
                # we could squeeze out first bogus dimensions
                if np.all(np.array(v.shape[3:]) == 1):
                    v.shape = v.shape[:3]
                else:
                    raise ValueError, \
                          "Original shape of some data is %s whenever we " \
                          "can accept only 3D images (or ND with degenerate " \
                          "first dimensions)" % (v.shape,)

    # process vlim
    vlim = list(vlim)
    vlim_orig = vlim[:]
    add_dist2hist = []
    if isinstance(vlim_type, basestring):
        if vlim_type == 'symneg_z':
            func_masked = func[func_mask]
            fnonpos = func_masked[func_masked<=0]
            fneg = func_masked[func_masked<0]
            # take together with sign-reverted negative values
            fsym = np.hstack((-fneg, fnonpos))
            nfsym = len(fsym)
            # Estimate normal std under assumption of mean=0
            std = np.sqrt(np.mean(abs(fsym)**2))
            # convert vlim assuming it is z-scores
            for i,v in enumerate(vlim):
                if v is not None:
                    vlim[i] = std * v
            # add a plot to histogram
            add_dist2hist = [(lambda x: nfsym/(np.sqrt(2*np.pi)*std) \
                                        *np.exp(-(x**2)/(2*std**2)),
                              {})]
        else:
            raise ValueError, 'Unknown specification of vlim=%s' % vlim + \
                  ' Known is: symneg'


    class Plotter(object):
        """
        TODO
        """

        #_store_attribs = ('vlim', 'fig', 'bg', 'bg_mask')

        def __init__(self, _locals):
            """TODO"""
            self._locals = _locals
            self.fig = _locals['fig']

        def do_plot(self):
            """TODO"""
            # silly yarik didn't find proper way
            vlim = self._locals['vlim']
            bg = self._locals['bg']
            bg_mask = self._locals['bg_mask']
            ncolumns = self._locals['ncolumns']
            nrows = self._locals['nrows']
            add_info = self._locals['add_info']
            add_hist = self._locals['add_hist']
            slices = self._locals['slices']
            slice_title = self._locals['slice_title']
            if np.isscalar(vlim): vlim = (vlim, None)
            if vlim[0] is None: vlim = (np.min(func), vlim[1])
            if vlim[1] is None: vlim = (vlim[0], np.max(func))
            if __debug__ and 'PLLB' in debug.active:
                debug('PLLB', "Maximum %g at %s, vlim is %s" %
                      (np.max(func), np.where(func==np.max(func)), str(vlim)))
            invert = vlim[1] < vlim[0]
            if invert:
                vlim = (vlim[1], vlim[0])
                print "Not yet fully supported"

            # adjust lower bound if it is too low
            # and there are still multiple values ;)
            func_masked = func[func_mask]
            if vlim[0] < np.min(func_masked) and \
                   np.min(func_masked) != np.max(func_masked):
                vlim = list(vlim)
                vlim[0] = np.min(func[func_mask])
                vlim = tuple(vlim)

            bound_above = (max(vlim) < np.max(func))
            bound_below = (min(vlim) > np.min(func))

            #
            # reverse the map if needed
            cmap_ = cmap_overlay
            if not bound_below and bound_above:
                if cmap_.endswith('_r'):
                    cmap_ = cmap_[:-2]
                else:
                    cmap_ += '_r'

            func_cmap = eval("pl.cm.%s" % cmap_)
            bg_cmap = eval("pl.cm.%s" % cmap_bg)

            if do_stretch_colors:
                clim = (np.min(func), np.max(func))#vlim
            else:
                clim = vlim

            #
            # figure out 'extend' for colorbar and threshold string
            extend, thresh_str = {
                (True, True) : ('both', 'x in [%.3g, %.3g]' % tuple(vlim)),
                (True, False): ('min', 'x in [%.3g, +inf]' % vlim[0]),
                (False, True): ('max', 'x in (-inf, %.3g]' % vlim[1]),
                (False, False): ('neither', 'none') }[(bound_below,
                                                       bound_above)]

            #
            # Figure out subplots
            dshape = func.shape
            if slices is None:
                slices = range(func.shape[-1])
            nslices = len(slices)

            # more or less square alignment ;-)
            if ncolumns is None:
                ncolumns = int(np.sqrt(nslices))
            ndcolumns = ncolumns
            # or 0 for the case if nrows is None and max needs both numerics
            nrows = max(nrows or 0, int(np.ceil(nslices * 1.0 / ncolumns)))

            # Check if additional column/row information was provided
            # and extend nrows/ncolumns
            for v in (add_hist, add_info):
                if v and not isinstance(v, bool):
                    ncolumns = max(ncolumns, v[1]+1)
                    nrows = max(nrows, v[0]+1)



            # Decide either we need more cells where to add hist and/or info
            nadd = bool(add_info) + bool(add_hist)
            while ncolumns*nrows - (nslices + nadd) < 0:
                ncolumns += 1

            locs = ['' for i in xrange(ncolumns*nrows)]

            # Fill in predefined locations
            for v,vl in ((add_hist, 'hist'),
                         (add_info, 'info')):
                if v and not isinstance(v, bool):
                    locs[ncolumns*v[0] + v[1]] = vl

            # Fill in slices
            for islice in slices:
                locs[locs.index('')] = islice

            # Fill the last available if necessary
            if add_hist and isinstance(add_hist, bool):
                locs[locs.index('')] = 'hist'
            if add_info and isinstance(add_info, bool):
                locs[locs.index('')] = 'info'

            Pioff()

            if self.fig is None:
                self.fig = pl.figure(facecolor='white',
                                    figsize=(4*ncolumns, 4*nrows))
            fig = self.fig
            fig.clf()
            #
            # how to threshold images
            thresholder = lambda x: np.logical_and(x>=vlim[0],
                                                  x<=vlim[1]) ^ invert

            #
            # Draw all slices
            self.slices_ax = []
            im0 = None
            for islice in slices[::-1]: #range(nslices)[::-1]:
                ax = fig.add_subplot(nrows, ncolumns,
                                     locs.index(islice) + 1,
                                     frame_on=False)
                self.slices_ax.append(ax)
                ax.axison = False
                slice_bg_nvoxels = None
                if bg is not None:
                    slice_bg = bg[:, :, islice]

                    slice_bg_ = np.ma.masked_array(slice_bg,
                                                  mask=np.logical_not(bg_mask[:, :, islice]))
                                                  #slice_bg<=0)
                    slice_bg_nvoxels = len(slice_bg_.nonzero()[0])
                    if __debug__:
                        debug('PLLB', "Plotting %i background elements in slice %i"
                              % (slice_bg_nvoxels, islice))

                slice_sl  = func[:, :, islice]

                in_thresh = thresholder(slice_sl)
                out_thresh = np.logical_not(in_thresh)
                slice_sl_ = np.ma.masked_array(slice_sl,
                                mask=np.logical_or(out_thresh,
                                                  np.logical_not(func_mask[:, :, islice])))

                slice_func_nvoxels = len(slice_sl_.nonzero()[0])
                if __debug__:
                    debug('PLLB', "Plotting %i foreground elements in slice %i"
                          % (slice_func_nvoxels, islice))

                kwargs = dict(aspect=aspect, origin='upper')
                              #extent=(0, slice_bg.shape[0],
                              #        0, slice_bg.shape[1]))

                # paste a blank white background first, since otherwise
                # recent matplotlib screws up those masked imshows
                im = ax.imshow(np.ones(slice_sl_.shape).T,
                               cmap=bg_cmap,
                               **kwargs)
                im.set_clim((0,1))

                # ax.clim((0,1))
                if slice_bg_nvoxels:
                    ax.imshow(slice_bg_.T,
                             # let's stay close to the ugly truth ;-)
                             #interpolation='bilinear',
                             interpolation='nearest',
                             cmap=bg_cmap,
                             **kwargs)

                if slice_func_nvoxels:
                    alpha = slice_bg_nvoxels and 0.9 or 1.0
                    im = ax.imshow(slice_sl_.T,
                                   interpolation='nearest',
                                   cmap=func_cmap,
                                   alpha=alpha,
                                   **kwargs)
                    im.set_clim(*clim)
                    im0 = im

                if slice_title:
                    pl.title(slice_title % locals())

            # func_masked = func[func_mask]

            #
            # Add summary information
            func_thr = func[np.logical_and(func_mask, thresholder(func))]
            if add_info and len(func_thr):
                self.info_ax = ax = fig.add_subplot(nrows, ncolumns,
                                                    locs.index('info')+1,
                                                    frame_on=False)
                #    cb = pl.colorbar(shrink=0.8)
                #    #cb.set_clim(clim[0], clim[1])
                ax.axison = False
                #if add_colorbar:
                #    cb = pl.colorbar(im, shrink=0.8, pad=0.0, drawedges=False,
                #                    extend=extend, cmap=func_cmap)

                stats = {'v':len(func_masked),
                         'vt': len(func_thr),
                         'm': np.mean(func_masked),
                         'mt': np.mean(func_thr),
                         'min': np.min(func_masked),
                         'mint': np.min(func_thr),
                         'max': np.max(func_masked),
                         'maxt': np.max(func_thr),
                         'mm': np.median(func_masked),
                         'mmt': np.median(func_thr),
                         'std': np.std(func_masked),
                         'stdt': np.std(func_thr),
                         'sthr': thresh_str}
                pl.text(0, 0.5, """
 Original:
  voxels = %(v)d
  range = [%(min).3g, %(max).3g]
  mean = %(m).3g
  median = %(mm).3g
  std = %(std).3g

 Thresholded: %(sthr)s:
  voxels = %(vt)d
  range = [%(mint).3g, %(maxt).3g]
  median = %(mt).3g
  mean = %(mmt).3g
  std = %(stdt).3g
  """ % stats,
                    horizontalalignment='left',
                    verticalalignment='center',
                    transform = ax.transAxes,
                    fontsize=14)

            cb = None
            if add_colorbar and im0 is not None:
                kwargs_cb = {}
                #if add_hist:
                #    kwargs_cb['cax'] = self.hist_ax
                self.cb_ax = cb = pl.colorbar(
                    im0, #self.hist_ax,
                    shrink=0.8, pad=0.0, drawedges=False,
                    extend=extend, cmap=func_cmap, **kwargs_cb)
                cb.set_clim(*clim)

            # Add histogram
            if add_hist:
                self.hist_ax = fig.add_subplot(nrows, ncolumns,
                                               locs.index('hist') + 1,
                                               frame_on=True)

                minv, maxv = np.min(func_masked), np.max(func_masked)
                if minv<0 and maxv>0:               # then make it centered on 0
                    maxx = max(-minv, maxv)
                    range_ = (-maxx, maxx)
                else:
                    range_ = (minv, maxv)
                H = np.histogram(func_masked, range=range_, bins=31)
                # API changed since v0.99.0-641-ga7c2231
                halign = externals.versions['matplotlib'] >= '1.0.0' \
                         and 'mid' or 'center'
                H2 = pl.hist(func_masked, bins=H[1], align=halign,
                             facecolor='#FFFFFF', hold=True)
                for a, kwparams in add_dist2hist:
                    dbin = (H[1][1] - H[1][0])
                    pl.plot(H2[1], [a(x) * dbin for x in H2[1]], **kwparams)
                if add_colorbar and cb:
                    cbrgba = cb.to_rgba(H2[1])
                    for face, facecolor, value in zip(H2[2], cbrgba, H2[1]):
                        if not thresholder(value):
                            color = '#FFFFFF'
                        else:
                            color = facecolor
                        face.set_facecolor(color)


            fig.subplots_adjust(left=0.01, right=0.95, hspace=0.25)
            # , bottom=0.01
            if ncolumns - int(bool(add_info) or bool(add_hist)) < 2:
                fig.subplots_adjust(wspace=0.4)
            else:
                fig.subplots_adjust(wspace=0.1)

            Pion()

        def on_click(self, event):
            """Actions to perform on click
            """
            if id(event.inaxes) != id(plotter.hist_ax):
                return
            xdata, ydata, button = event.xdata, event.ydata, event.button
            vlim = self._locals['vlim']
            if button == 1:
                vlim[0] = xdata
            elif button == 3:
                vlim[1] = xdata
            elif button == 2:
                vlim[0], vlim[1] = vlim[1], vlim[0]
            self.do_plot()

    plotter = Plotter(locals())
    plotter.do_plot()

    if interactive is None:
        interactive = mpl_backend_isinteractive

    # Global adjustments
    if interactive:
        # if pl.matplotlib.is_interactive():
        pl.connect('button_press_event', plotter.on_click)
        pl.show()

    plotter.fig.plotter = plotter
    return plotter.fig
