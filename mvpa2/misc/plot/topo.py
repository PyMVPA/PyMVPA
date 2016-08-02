# emacs: -*- coding: utf-8; mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
#   The initial version of the code was contributed by Ingo Fründ and is
#   Coypright (c) 2008 by Ingo Fründ ingo.fruend@googlemail.com
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Plot parameter distributions on a head surface (topography plots)."""

__docformat__ = 'restructuredtext'

import numpy as np

from mvpa2.base import externals

if externals.exists("pylab", raise_=True):
    import pylab as pl

if externals.exists("griddata", raise_=True):
    from mvpa2.support.griddata import griddata

if externals.exists("scipy", raise_=True):
    from scipy.optimize import leastsq

if externals.versions['numpy'] > '1.1.0':
    from numpy import ma
else:
    from matplotlib.numerix import ma

# TODO : add optional plotting labels for the sensors
##REF: Name was automagically refactored
def plot_head_topography(topography, sensorlocations, plotsensors=False,
                       resolution=51, masked=True, plothead=True,
                       plothead_kwargs=None, **kwargs):
    """Plot distribution to a head surface, derived from some sensor locations.

    The sensor locations are first projected onto the best fitting sphere and
    finally projected onto a circle (by simply ignoring the z-axis).

    Parameters
    ----------
    topography : array
      A vector of some values corresponding to each sensor.
    sensorlocations : (nsensors x 3) array
      3D coordinates of each sensor. The order of the sensors has to match
      with the `topography` vector.
    plotsensors : bool
      If True, sensor will be plotted on their projected coordinates.
      No sensor are shown otherwise.
    plothead : bool
      If True, a head outline is plotted.
    plothead_kwargs : dict
      Additional keyword arguments passed to `plot_head_outline()`.
    resolution : int
      Number of surface samples along both x and y-axis.
    masked : bool
      If True, all surface sample extending to head outline will be
      masked.
    **kwargs
      All additional arguments will be passed to `pylab.imshow()`.

    Returns
    -------
    (map, head, sensors)
      The corresponding matplotlib objects are returned if plotted, ie.
      if plothead is set to `False`, `head` will be `None`.

          map
            The colormap that makes the actual plot, a
            matplotlib.image.AxesImage instance.
          head
            What is returned by `plot_head_outline()`.
          sensors
            The dots marking the electrodes, a matplotlib.lines.Line2d
            instance.
    """
    # give sane defaults
    if plothead_kwargs is None:
        plothead_kwargs = {}

    # error function to fit the sensor locations to a sphere
    def err(params):
        r, cx, cy, cz = params
        return (sensorlocations[:, 0] - cx) ** 2 \
               + (sensorlocations[:, 1] - cy) ** 2 \
               + (sensorlocations[:, 2] - cz) ** 2 \
               - r ** 2

    # initial guess of sphere parameters (radius and center)
    params = (1, 0, 0, 0)

    # do fit
    (r, cx, cy, cz), stuff = leastsq(err, params)

    # size of each square
    ssh = float(r) / resolution         # half-size
    ss = ssh * 2.0                      # full-size

    # Generate a grid and interpolate using the griddata module
    x = np.arange(cx - r, cx + r, ss) + ssh
    y = np.arange(cy - r, cy + r, ss) + ssh
    x, y = pl.meshgrid(x, y)

    # project the sensor locations onto the sphere
    sphere_center = np.array((cx, cy, cz))
    sproj = sensorlocations - sphere_center
    sproj = r * sproj / np.c_[np.sqrt(np.sum(sproj ** 2, axis=1))]
    sproj += sphere_center

    # fit topology onto xy projection of sphere
    topo = griddata(sproj[:, 0], sproj[:, 1],
                    np.ravel(np.array(topography)), x, y,
                    interp='nn' if externals.versions['matplotlib'] < '1.4.0'
                           else 'linear')

    # mask values outside the head
    if masked:
        notinhead = np.greater_equal((x - cx) ** 2 + (y - cy) ** 2,
                                    (1.0 * r) ** 2)
        topo = ma.masked_where(notinhead, topo)

    # show surface
    map = pl.imshow(topo, origin="lower", extent=(-r, r, -r, r), **kwargs)
    pl.axis('off')

    if plothead:
        # plot scaled head outline
        head = plot_head_outline(scale=r, shift=(cx/2.0, cy/2.0), **plothead_kwargs)
    else:
        head = None

    if plotsensors:
        # plot projected sensor locations

        # reorder sensors so the ones below plotted first
        # TODO: please fix with more elegant solution
        zenum = [x[::-1] for x in enumerate(sproj[:, 2].tolist())]
        zenum.sort()
        indx = [ x[1] for x in zenum ]
        sensors = pl.plot(sproj[indx, 0] - cx/2.0, sproj[indx, 1] - cy/2.0, 'wo')
    else:
        sensors = None

    return map, head, sensors


##REF: Name was automagically refactored
def plot_head_outline(scale=1, shift=(0, 0), color='k', linewidth='5', **kwargs):
    """Plots a simple outline of a head viewed from the top.

    The plot contains schematic representations of the nose and ears. The
    size of the head is basically a unit circle for nose and ears attached
    to it.

    Parameters
    ----------
    scale : float
      Factor to scale the size of the head.
    shift : 2-tuple of floats
      Shift the center of the head circle by these values.
    color : matplotlib color spec
      The color the outline should be plotted in.
    linewidth : int
      Linewidth of the head outline.
    **kwargs
      All additional arguments are passed to `pylab.plot()`.

    Returns
    -------
    Matplotlib lines2D object
      can be used to tweak the look of the head outline.
    """

    rmax = 0.5
    # factor used all the time
    fac = 2 * np.pi * 0.01

    # Koordinates for the ears
    EarX1 =  -1 * np.array(
            [.497, .510, .518, .5299,
            .5419, .54, .547, .532, .510,
            rmax * np.cos(fac * (54 + 42))])
    EarY1 = np.array(
            [.0655, .0775, .0783, .0746, .0555,
            -.0055, -.0932, -.1313, -.1384,
            rmax * np.sin(fac * (54 + 42))])
    EarX2 = np.array(
            [rmax * np.cos(fac * (54 + 42)),
            .510, .532, .547, .54, .5419,
            .5299, .518, .510, .497] )
    EarY2 = np.array(
            [rmax * np.sin(fac * (54 + 42)),
            -.1384, -.1313, -.0932, -.0055,
            .0555, .0746, .0783, .0775, .0655] )

    # Coordinates for the Head
    HeadX1 = np.fromfunction(
            lambda x: rmax * np.cos(fac * (x + 2)), (21,))
    HeadY1 = np.fromfunction(
            lambda y: rmax * np.sin(fac * (y + 2)), (21,))
    HeadX2 = np.fromfunction(
            lambda x: rmax * np.cos(fac * (x + 28)), (21,))
    HeadY2 = np.fromfunction(
            lambda y: rmax * np.sin(fac * (y + 28)), (21,))
    HeadX3 = np.fromfunction(
            lambda x: rmax * np.cos(fac * (x + 54)), (43,))
    HeadY3 = np.fromfunction(
            lambda y: rmax * np.sin(fac * (y + 54)), (43,))

    # Coordinates for the Nose
    NoseX = np.array([.18 * rmax, 0, -.18 * rmax])
    NoseY = np.array([rmax - 0.004, rmax * 1.15, rmax - 0.004])

    # Combine to one
    X = np.concatenate((EarX2, HeadX1, NoseX, HeadX2, EarX1, HeadX3))
    Y = np.concatenate((EarY2, HeadY1, NoseY, HeadY2, EarY1, HeadY3))

    X *= 2 * scale
    Y *= 2 * scale
    X += shift[0]
    Y += shift[1]

    return pl.plot(X, Y, color=color, linewidth=linewidth)
