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

import pylab as P
import numpy as N
import matplotlib as mpl

from mvpa.base import warning, externals

if externals.exists('nifti'):
    from nifti import NiftiImage
else:
    class NiftiImage(object):
        """Just a helper to allow plot_lightbox be used even if no
        nifti module available for plotting regular 2D/3D images
        (ndarrays)"""
        def __init__(self, filename):
            raise ValueError, "plot_lightbox was provided a filename %s.  " \
                  "By now we only support loading data from Nifti/Analyze " \
                  "files, but nifti module is not available" % filename

_interactive_backends = ['GTKAgg', 'TkAgg']

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
    files names or `NiftiImage` objects, or 3D `ndarrays`. If no mask
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
        """Helper which would read in NiftiImage if necessary
        """
        if isinstance(arg, basestring):
            arg = NiftiImage(arg)
            argshape = arg.data.shape
            # Assure that we have 3D (at least)
            if len(argshape)<3:
                arg.data = arg.data.reshape((1,)*(3-len(argshape)) + argshape)
        if isinstance(arg, N.ndarray):
            if len(arg.shape) != 3:
                raise ValueError, "For now just handling 3D volumes"
        return arg

    bg = handle_arg(background)
    if isinstance(bg, NiftiImage):
        # figure out aspect
        fov = (N.array(bg.header['pixdim']) * bg.header['dim'])[3:0:-1]
        aspect = fov[1]/fov[2]

        bg = bg.data #[..., ::-1, ::-1] # XXX custom for now
    else:
        aspect = 1.0

    if bg is not None:
        bg_mask = handle_arg(background_mask)
        if isinstance(bg_mask, NiftiImage):
            bg_mask = bg_mask.data #[..., ::-1, ::-1] # XXX
        if bg_mask is not None:
            bg_mask = bg_mask != 0
        else:
            bg_mask = bg != 0

    func = handle_arg(overlay)

    if func is not None:
        if isinstance(func, NiftiImage):
            func = func.data #[..., ::-1, :] # XXX

        func_mask = handle_arg(overlay_mask)
        if isinstance(func_mask, NiftiImage):
            func_mask = func_mask.data #[..., ::-1, :] # XXX
        if func_mask is not None:
            func_mask = func_mask != 0
        else:
            func_mask = func != 0

    # Lets assure that that we are dealing with <= 3D
    for v in (bg, bg_mask, func, func_mask):
        if v is not None:
            if v.ndim > 3:
                # we could squeeze out first bogus dimensions
                if N.all(N.array(v.shape[:v.ndim-3]) == 1):
                    v.shape = v.shape[v.ndim-3:]
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
            fsym = N.hstack((-fneg, fnonpos))
            nfsym = len(fsym)
            # Estimate normal std under assumption of mean=0
            std = N.sqrt(N.mean(abs(fsym)**2))
            # convert vlim assuming it is z-scores
            for i,v in enumerate(vlim):
                if v is not None:
                    vlim[i] = std * v
            # add a plot to histogram
            add_dist2hist = [(lambda x: nfsym/(N.sqrt(2*N.pi)*std) \
                                        *N.exp(-(x**2)/(2*std**2)),
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
            if N.isscalar(vlim): vlim = (vlim, None)
            if vlim[0] is None: vlim = (N.min(func), vlim[1])
            if vlim[1] is None: vlim = (vlim[0], N.max(func))
            invert = vlim[1] < vlim[0]
            if invert:
                vlim = (vlim[1], vlim[0])
                print "Not yet fully supported"

            # adjust lower bound if it is too low
            # and there are still multiple values ;)
            func_masked = func[func_mask]
            if vlim[0] < N.min(func_masked) and \
                   N.min(func_masked) != N.max(func_masked):
                vlim = list(vlim)
                vlim[0] = N.min(func[func_mask])
                vlim = tuple(vlim)

            bound_above = (max(vlim) < N.max(func))
            bound_below = (min(vlim) > N.min(func))

            #
            # reverse the map if needed
            cmap_ = cmap_overlay
            if not bound_below and bound_above:
                if cmap_.endswith('_r'):
                    cmap_ = cmap_[:-2]
                else:
                    cmap_ += '_r'

            func_cmap = eval("P.cm.%s" % cmap_)
            bg_cmap = eval("P.cm.%s" % cmap_bg)

            if do_stretch_colors:
                clim = (N.min(func), N.max(func))#vlim
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
                slices = range(func.shape[0])
            nslices = len(slices)

            # more or less square alignment ;-)
            if ncolumns is None:
                ncolumns = int(N.sqrt(nslices))
            ndcolumns = ncolumns
            nrows = max(nrows, int(N.ceil(nslices*1.0/ncolumns)))

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

            # should compare by backend?
            if P.matplotlib.get_backend() in _interactive_backends:
                P.ioff()

            if self.fig is None:
                self.fig = P.figure(facecolor='white',
                                    figsize=(4*ncolumns, 4*nrows))
            else:
                self.fig.clf()
            fig = self.fig
            # fig.clf()

            #
            # how to threshold images
            thresholder = lambda x: N.logical_and(x>=vlim[0],
                                                  x<=vlim[1]) ^ invert

            #
            # Draw all slices
            self.slices_ax = []
            for islice in slices[::-1]: #range(nslices)[::-1]:
                ax = fig.add_subplot(nrows, ncolumns,
                                     locs.index(islice) + 1,
                                     frame_on=False)
                self.slices_ax.append(ax)
                ax.axison = False
                slice_bg = bg[islice]
                slice_bg_ = N.ma.masked_array(slice_bg,
                                              mask=N.logical_not(bg_mask[islice]))
                                              #slice_bg<=0)

                slice_sl  = func[islice]

                in_thresh = thresholder(slice_sl)
                out_thresh = N.logical_not(in_thresh)
                slice_sl_ = N.ma.masked_array(slice_sl,
                                mask=N.logical_or(out_thresh,
                                                  N.logical_not(func_mask[islice])))

                kwargs = dict(aspect=aspect, origin='upper',
                              extent=(0, slice_bg.shape[0],
                                      0, slice_bg.shape[1]))

                # paste a blank white background first, since otherwise
                # recent matplotlib screws up those masked imshows
                im = ax.imshow(N.ones(slice_sl_.shape),
                               cmap=bg_cmap,
                               **kwargs)
                im.set_clim((0,1))

                # ax.clim((0,1))
                ax.imshow(slice_bg_,
                         interpolation='bilinear',
                         cmap=bg_cmap,
                         **kwargs)

                im = ax.imshow(slice_sl_,
                               interpolation='nearest',
                               cmap=func_cmap,
                               alpha=0.8,
                               **kwargs)
                im.set_clim(*clim)
                if slice_title:
                    P.title(slice_title % locals())
                if islice == slices[0]:
                    im0 = im

            # func_masked = func[func_mask]

            #
            # Add summary information
            func_thr = func[N.logical_and(func_mask, thresholder(func))]
            if add_info and len(func_thr):
                self.info_ax = ax = fig.add_subplot(nrows, ncolumns,
                                                    locs.index('info')+1,
                                                    frame_on=False)
                #    cb = P.colorbar(shrink=0.8)
                #    #cb.set_clim(clim[0], clim[1])
                ax.axison = False
                #if add_colorbar:
                #    cb = P.colorbar(im, shrink=0.8, pad=0.0, drawedges=False,
                #                    extend=extend, cmap=func_cmap)

                stats = {'v':len(func_masked),
                         'vt': len(func_thr),
                         'm': N.mean(func_masked),
                         'mt': N.mean(func_thr),
                         'min': N.min(func_masked),
                         'mint': N.min(func_thr),
                         'max': N.max(func_masked),
                         'maxt': N.max(func_thr),
                         'mm': N.median(func_masked),
                         'mmt': N.median(func_thr),
                         'std': N.std(func_masked),
                         'stdt': N.std(func_thr),
                         'sthr': thresh_str}
                P.text(0, 0.5, """
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

            if add_colorbar:
                kwargs_cb = {}
                #if add_hist:
                #    kwargs_cb['cax'] = self.hist_ax
                self.cb_ax = cb = P.colorbar(
                    im0, #self.hist_ax,
                    shrink=0.8, pad=0.0, drawedges=False,
                    extend=extend, cmap=func_cmap, **kwargs_cb)
                cb.set_clim(*clim)

            # Add histogram
            if add_hist:
                self.hist_ax = fig.add_subplot(nrows, ncolumns,
                                               locs.index('hist') + 1,
                                               frame_on=True)

                minv, maxv = N.min(func_masked), N.max(func_masked)
                if minv<0 and maxv>0:               # then make it centered on 0
                    maxx = max(-minv, maxv)
                    range_ = (-maxx, maxx)
                else:
                    range_ = (minv, maxv)
                H = N.histogram(func_masked, range=range_, bins=31)
                H2 = P.hist(func_masked, bins=H[1], align='center',
                            facecolor='#FFFFFF', hold=True)
                for a, kwparams in add_dist2hist:
                    dbin = (H[1][1] - H[1][0])
                    P.plot(H2[1], [a(x) * dbin for x in H2[1]], **kwparams)
                if add_colorbar:
                    cbrgba = cb.to_rgba(H2[1])
                    for face, facecolor, value in zip(H2[2], cbrgba, H2[1]):
                        if not thresholder(value):
                            color = '#FFFFFF'
                        else:
                            color = facecolor
                        face.set_facecolor(color)


            fig.subplots_adjust(left=0.01, right=0.95, hspace=0.01)
            # , bottom=0.01
            if ncolumns - int(bool(add_info) or bool(add_hist)) < 2:
                fig.subplots_adjust(wspace=0.4)
            else:
                fig.subplots_adjust(wspace=0.1)

            if P.matplotlib.get_backend() in _interactive_backends:
                P.draw()
                P.ion()

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
        interactive = P.matplotlib.get_backend() in _interactive_backends

    # Global adjustments
    if interactive:
        # if P.matplotlib.is_interactive():
        P.connect('button_press_event', plotter.on_click)
        P.show()

    plotter.fig.plotter = plotter
    return plotter.fig


if __name__ == "__main__":
    # for easy debugging
    import os
    from mvpa.base import cfg
    impath = os.path.join('datadb', 'haxby2001', 'subj1')
    plot_lightbox(
        #background = NiftiImage('%s/anat.nii.gz' % impath),
        background = NiftiImage('%s/bold.nii.gz' % impath).data.squeeze()[0],
        background_mask = None,
        overlay = NiftiImage('%s/mask4_vt.nii.gz' % impath),
        #overlay_mask = NiftiImage('%s/masks/example_func_brain_mask.nii.gz' % impath),
        #
        do_stretch_colors = False,
        add_colorbar = True,
        cmap_bg = 'gray',
        cmap_overlay = 'Set3', #'hot', # YlOrRd_r # P.cm.autumn
        #
        fig = None,
        # vlim describes value limits
        # clim color limits (same by default)
        vlim = [0, 1],
        #vlim_type = 'symneg_z',
        interactive = True,
        #
        #nrows = 2,
        #ncolumns = 3,
        add_info = (1, 2),
        add_hist = (0, 2),
        #
        slices = [20, 23, 26, 29, 32, 35]
        )

    P.show()
