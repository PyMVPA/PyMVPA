'''
Simple Freesurfer ASCII surface file I/O functions

Reads and writes surface anatomy files as understood by AFNI SUMA (and maybe other programs)
The format for a surface with NV vertices and NF faces is:

NV NF
x_0 y_0 z_0 0
x_1 y_1 z_1 0
...
x_[NV-1] y_[NV-1] z_[NV-1] 0
f_00 f01 f02 0
f_10 f11 f12 0
...
f_[NF-1]0 f_[NF-1]1 f_[NF-1]2 0

where the (x,y,z) triples are coordinates and fi(p,q,r) are faces so that vertices with
indices p, q and r form a single triangle

Created on Feb 12, 2012

@author: nick
'''

import numpy as np, os, datetime, utils, afni_suma_1d, afni_niml_dset
from surf import Surface


def read(fn):
    '''
    Reads a AFNI SUMA ASCII surface
    
    Parameters
    ----------
    fn : str
        Filename of ASCII surface file
    
    Returns
    -------
    s : Surface
        a surf.Surface as defined in 'fn'
    '''

    if not os.path.exists(fn):
        raise Exception("File not found: %s" % fn)

    with open(fn) as f:
        r = f.read().split("\n")

    row = 0
    nv = nf = None # number of vertices and faces
    while True:
        line = r[row]
        row += 1

        if line.startswith("#"):
            continue

        try:
            nvnf = line.split(" ")
            nv = int(nvnf[0])
            nf = int(nvnf[1])
            break

        except:
            continue

    if not nf:
        raise Exception("Not found in %s: number of nodes and faces" % fn)

    # helper function to get a numpy Cx3 ndarray
    def getrows(c, s): # c: number of rows, s is string with data
        vs = np.fromstring(s, count=4 * c, sep=" ")
        vx = np.reshape(vs, (c, 4))
        return vx[:, :3]

    # coordinates should start at pos...
    v = getrows(nv, "\n".join(r[row:(row + nv)]))

    # and the faces just after those
    ffloat = getrows(nf, "\n".join(r[(row + nv):(row + nv + nf)]))
    f = ffloat.astype(int)

    return Surface(v=v, f=f)

def write(surf, fn, overwrite=False, comment=None):
    '''
    Writes a AFNI SUMA ASCII surface
    
    Parameters
    ----------
    surf: surf.Surface
        surface to be written
    fn : str
        Output filename of ASCII surface file
    overwrite : bool
        Whether to overwrite 'fn' if it exists
    comment : str
        Comments to add to 'fn'
    '''

    if isinstance(surf, str) and isinstance(fn, Surface):
        surf, fn = fn, surf
        print "Swap"

    if not overwrite and os.path.exists(fn):
        raise Exception("File already exists: %s" % fn)

    s = []
    if comment == None:
        comment = '# Created %s' % str(datetime.datetime.now())
    s.append(comment)

    nv, nf, v, f = surf.nvertices, surf.nfaces, surf.vertices(), surf.faces()

    # number of vertices and faces
    s.append('%d %d' % (nv, nf))

    # add vertices and faces
    s.extend('%f %f %f 0' % (v[i, 0], v[i, 1], v[i, 2]) for i in xrange(nv))
    s.extend('%d %d %d 0' % (f[i, 0], f[i, 1], f[i, 2]) for i in xrange(nf))

    # write to file
    f = open(fn, 'w')
    f.write("\n".join(s))
    f.close()

def hemi_pairs_reposition(surf_left, surf_right, touching_side, min_distance=10.):
    '''moves and rotates pairs of hemispheres so that they are almost touching on
    one side (with distance min_distance)
    touching side should be 'm','i','s','a', or 'p'
    returns a tuple of left and right hemispheres that can be viewed together
    and have a space of min_distance in between them'''

    touching_side = touching_side[0].lower()

    mn, mx = np.min, np.max
    #min=-1, max=1
    side2dimsigns = dict(m=(0, -1), i=(1, 1), s=(1, -1), a=(2, 1), p=(2, -1))

    dim, rotatesign = side2dimsigns[touching_side]
    if dim == 0:
        rotate_axis = None
    else:
        rotate_axis = dim #1+((dim+1) % 2)
        rotate_angle = 90

    surfs = [surf_left, surf_right]
    nsurfs = len(surfs)
    hemisigns = [1, -1]
    if not rotate_axis is None:
        theta = [0] * 3

        for i in xrange(nsurfs):
            theta[rotate_axis] = rotate_angle * hemisigns[i] * rotatesign
            surfs[i] = surfs[i].rotate(theta, unit='deg')


    for i in xrange(nsurfs):
        hemisign = hemisigns[i]
        sign = rotatesign * hemisign
        coords = surfs[i].v()

        xtreme = np.min(coords[:, 0] * -hemisign)

        #delta = np.zeros((1, 3))
        # sometimes the surfaces are not properly aligned along x and y
        # so fix it by moving by center of mass values along x and y

        delta = -np.reshape(surfs[i].center_of_mass(), (1, 3))
        delta[0, 0] = hemisign * (xtreme - min_distance * .5)
        surfs[i] = surfs[i] + delta # make an implicit copy

    return tuple(surfs)

def sphere_reg_leftrightmapping(surf_sphere_reg_left, surf_sphere_reg_right):
    '''finds the mapping from left to right hemispehre and vice versa
    (the mapping is symmetric)
    
    this only works on sphere.reg.asc files made with AFNI/SUMA's mapicosehedron'''

    vL, vR = surf_sphere_reg_left.v(), surf_sphere_reg_right.v()

    def _check_is_sphere(v, eps=.0001):
        com = np.mean(vL, axis=0)

        if max(abs(com)) > eps:
            raise ValueError('Center of Mass not origin: %r' % com)

        dst = np.sum(v ** 2, axis=1) ** .5

        if min(dst) < 99 or max(dst) > 101:
            raise ValueError('Not a sphere with approximately radius of 100')

    map(_check_is_sphere, (vL, vR))

    # swap left and right, to make righth 'like' left
    swapX = np.array([[-1, 1, 1]])
    vS = vR * swapX
    n = vS.shape[0]

    # make a mapping so that node i on the left surface is corresponding to L2R[i] 
    # on the right surface
    left2right = np.zeros((n,), dtype=np.int)

    # find corresponding nodes
    for i in xrange(n):
        d2 = np.sum((vL[i, :] - vS) ** 2, axis=1) # squared distance
        minidx = np.argmin(d2)
        left2right[i] = minidx

        if d2[minidx] > .001:
            raise ValueError('no mapping found for node %r: min distance %r' % (i, d2[minidx] ** .5))

    # just one final check     
    if not all(left2right[left2right[i]] == i for i in xrange(n)):
        raise ValueError('unsymmetric mapping - this should not happen')

    return left2right
