#!/usr/bin/env python
# encoding: utf-8

__doc__ = """
This module contains a funcitons the plot the distribution of a parameter
on the head

(c) 2008 by Ingo Fründ ingo.fruend@googlemail.com
"""

__all__ = ["plotHead"]

from numpy import *
import pylab as p
import matplotlib.numerix.ma as M
from griddata import griddata


def plotHead(topography,sensorlocations,channels=None,plotelectrodes=True):
    """
    Plot a topology

    Input
    =====
    topography  a vector of channel activations that is meant to be
                visualized
    sensorlocations could either be a nchannels x 3 array of sensorlocations
                or a string indication the name of a file the contains
                the locations of the channels.
    channels    should be a list of channels in the order they appear in
                the topography vector
    plotelectrodes  boolean indicating whether or not the positions of the
                electrodes should be marked by small dots

    Output
    ======
    map, head (,electrodes)
    map         the colormap that makes the actual plot, a
                matplotlib.image.AxesImage instance
    head        the line defining the head, a matplotlib.lines.Line2d
                instance
    electrodes  the dots marking the electrodes, a matplotlib.lines.Line2d
                instance

    These outputs can be used to modify the appearance of the plot using
    matplotlib/pylab functions.
    """

    # remove EOG if needed
    if "EOG" in channels:
        EOG = channels.index("EOG")
        i = arange(len(channels),dtype='i')
        i = where(logical_not(i==EOG))
        topography = topography[i]
        channels.pop(EOG)
        if not isinstance(sensorlocations,str):
            sensorlocations = sensorlocations[i]

    # We need this if the channellocations are a file
    if isinstance(sensorlocations,str):
        f = {}
        for l in open(sensorlocations,"r"):
            if len(l.split())==0:
                continue
            key = l.split()[0]
            f[key] = [float(x) for x in l.split()[1:]]
        nsensors = max(topography.shape)
        sensorlocations = zeros((nsensors,3),'d')
        for k,ch in enumerate(channels):
            try:
                sensorlocations[k,:] = array(f[ch],'d')
            except IndexError:
                print sensorlocations.shape,k,ch
                raise IndexError

    # Generate a grid and interpolate using the griddata module
    x = arange(65,dtype='d')-32
    x /= 32
    x,y = p.meshgrid(x,x)
    topo = griddata(sensorlocations[:,0],sensorlocations[:,1],\
            ravel(array(topography)),x,y)

    # Mask values outside the head
    notinhead = fromfunction (
            lambda k,l: greater_equal ((k-32)**2+(l-32)**2,1024 ),
            (65,65) )
    topo_masked = M.masked_where(notinhead,topo)

    # Show everything
    map = p.imshow(topo_masked,origin="lower")
    head = genHead(scale=32,shift=(32,32))
    p.axis('off')
    if plotelectrodes:
        electrodes = p.plot(32*sensorlocations[:,0]+32,
                32*sensorlocations[:,1]+32,'wo')
        return map,head,electrodes
    else:
        return map,head


def genHead(scale=1, shift=(0,0)):
    """
    Generate a list that contains the coordinates to plot a head

    use scale to scale the head to the desired size and shift to shift
    it to the desired place
    """

    rmax = 0.5

    # Koordinates for the ears
    EarX1 =  -1*array (
            [.497, .510, .518, .5299,
            .5419, .54, .547, .532, .510,
            rmax*cos (2*pi*0.01*(54+42))])
    EarY1 = array (
            [.0655, .0775, .0783, .0746, .0555,
            -.0055, -.0932, -.1313, -.1384,
            rmax*sin(2*pi*0.01*(54+42))])
    EarX2 = array (
            [rmax*cos ( 2*pi*0.01*(54+42)),
            .510, .532, .547, .54, .5419,
            .5299, .518, .510, .497] )
    EarY2 = array (
            [rmax*sin ( 2*pi*0.01*(54+42)),
            -.1384, -.1313, -.0932, -.0055,
            .0555, .0746, .0783, .0775, .0655] )

    # Coordinates for the Head
    HeadX1 = fromfunction (
            lambda x: rmax*cos (2*pi*0.01*(x+2)), (21,))
    HeadY1 = fromfunction (
            lambda y: rmax*sin (2*pi*0.01*(y+2)), (21,))
    HeadX2 = fromfunction (
            lambda x: rmax*cos (2*pi*0.01*(x+28)), (21,))
    HeadY2 = fromfunction (
            lambda y: rmax*sin (2*pi*0.01*(y+28)), (21,))
    HeadX3 = fromfunction (
            lambda x: rmax*cos ( 2*pi*0.01*(x+54)),(43,))
    HeadY3 = fromfunction (
            lambda y: rmax*sin ( 2*pi*0.01*(y+54)),(43,))

    # Coordinates for the Nose
    NoseX = array ( [.18*rmax, 0, -.18*rmax] )
    NoseY = array ( [rmax-0.004,rmax*1.15,rmax-0.004] )

    # Combine to one
    X = concatenate ( (EarX2, HeadX1, NoseX, HeadX2, EarX1, HeadX3) )
    Y = concatenate ( (EarY2, HeadY1, NoseY, HeadY2, EarY1, HeadY3) )

    X *= 2*scale
    Y *= 2*scale
    X += shift[0]
    Y += shift[1]

    return p.plot(X,Y,'k',linewidth=3)
