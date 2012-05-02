"""
Associate volume geometry with surface meshes

@author: nick
"""

import surf, volgeom, collections, surf_fs_asc, numpy as np, utils, time, nibabel as ni, operator, random
import sparse_attributes, afni_niml_dset, cPickle as pickle
import collections

class VolSurf():
    '''
    Associates a volume geometry with two surfaces (pial and white), so that 
    voxel selection in the volume can be performed using the surface information
    
    Parameters
    ----------
    volgeom: volgeom.VolGeom
        Volume geometry
    pial: surf.Surface
        Surface representing pial-grey matter boundary
    white: surf.Surface
        Surface representing white-grey matter boundary
    intermediate: surf.Surface
        Surface representing intermediate surface. This surface can have fewer 
        nodes than the pial and white surface. If None it uses the node-wise
         average of pial and white
    eps: float
        Maximum allowed distance between nodes on intermediate and the
        node-wise average of pial and white. This value is only used
        if intermediate is not None.
        
    Note
    ----
    'pial' and 'white' should have the same topology. 
    '''

    def __init__(self, volgeom, pial, white):
        if not pial.same_topology(white):
            raise Exception("Not same topology for white and pial")

        self._volgeom = volgeom
        self._pial = pial
        self._white = white

    def __repr__(self):
        r = ("volgeom: %r\npial: %rwhite:%r" %
                (self._volgeom, self._pial, self._white))
        return r

    def node2voxels(self, steps=10, start=0.0, stop=1.0):
        '''
        Generates a mapping from node indices to voxels that are at or near
        the nodes at the pial and white surface.
        
        Parameters
        ----------
        steps: int (default: 10)
            Number of steps from pial to white matter. For each node pair
            across the 'pial' and 'white' surface, a line is constructing connecting
            the pairs. Subsequently 'steps' steps are taken from 'pial' to 'white' 
            and the voxel at that position at the line is associated with that node pair.
        start: float (default: 0)
            Relative start position of line in gray matter, 0.=white surface, 1.=pial surface
            CheckMe: it might be the other way around 
        stop: float (default: 1)
            Relative stop position of line, in gray matter, 0.=white surface, 1.=pial surface
            
              
        Returns
        -------
        n2v: dict
            A mapping from node indices to voxels. In this mapping, the 'i'-th node 
            is associated with 'n2v[i]=v2p' which contains the mapping from 
            linear voxel indices to grey matter positions. In other words, 
            'n2v[i][idx]=v2p[idx]=pos' means that the voxel with linear index 'idx' is 
            associated with node 'i' and has has relative position 'pos' in the gray matter
            
            If node 'i' is outside the volume, then 'n2v[i]=None'
            
        Note
        ----
        The rationale is that a sufficient dense cortical surface mesh, combined
        with a sufficient number of steps, the grey matter is sampled dense enough
        so that  'no voxels are left out'. 
         
        '''
        if start > stop or steps < 1:
            raise ValueError("Illegal start/stop combination, or not enough steps")

        # make a list of the different relative gray matter positions
        if steps > 1:
            step = (stop - start) / float(steps - 1)
        else:
            step = 0
            start = stop = .5

        # node to voxels mapping
        # if n2v[i]=vs, then node i is associated with the voxels vs
        #
        # vs is a mapping from indices to relative position in grey matter
        # wheere 0 means white surface and 1 means pial surface
        # vs[k]=pos means that voxel with linear index k is 
        # associated with relative positions pos0
        #
        # CHECKME that I did not confuse (the order of) pial and white surface



        center_ids = range(self._pial.nv())
        nv = len(center_ids) # number of nodes on the surface
        n2vs = dict() # node to voxel indices mapping
        for j in xrange(nv):
            n2vs[j] = None # by default, no voxels associated with each node

        volgeom = self._volgeom

        # different 'layers' (depths) in the grey matter
        for i in xrange(steps):
            pialweight = start + step * i
            whiteweight = 1 - pialweight

            # compute weighted intermediate surface in between pial and white
            surf = self._pial * pialweight + self._white * whiteweight

            surf_xyz = surf.v() # coordinates
            
            lin_vox = volgeom.xyz2lin(surf_xyz) # linear indices of voxels containing nodes

            is_vox_in_vol = volgeom.lininvol(lin_vox) # which of these voxels are actually in the volume

            vol_xyz = volgeom.lin2xyz(lin_vox) # coordinates of voxels

            # compute relative position of each voxel in grey matter
            grey_matter_pos = self.surf_project_weights_nodewise(vol_xyz)

            for center_id in center_ids: # for each node on the surface
                # associate voxels with the present center node
                # - center node must be in volume
                # - if onlnyinbetween is required, then also make sure that the 
                #   relative position is in between 0 and 1 (or whatever the values of
                #   start and stop are)

                if is_vox_in_vol[center_id]:
                    if n2vs[center_id] is None: # no voxels yet assocaited with this node
                        n2vs[center_id] = dict()

                    n2vs[center_id][lin_vox[center_id]] = grey_matter_pos[center_id]

        return n2vs

    def surf_project_nodewise(self, xyz):
        '''
        Project coordinates on lines connecting pial and white matter
        
        Parameters
        ----------
        xyz: numpy.ndarray (float)
            Px3 array with coordinates, assuming 'white' and 'pial' surfaces have P nodes each
        
        Returns
        -------
        xyz_proj: numpy.ndarray (float)
            Px3 array with coordinates the constraints that xyz_proj[i,:] lies on the 
            line connecting node 'i' on the white and pial surface, and that xyz_proj[i,:]
            is closest to xyz[i,:] of all points on this line.   
        '''

        pxyz = self._pial.v()
        weights = self.surf_project_weights(xyz)
        return pxyz + np.reshape(weights, (-1, 1)) * pxyz

    def surf_project_weights_nodewise(self, xyz):
        '''
        Computes relative position of xyz on lines from pial to white matter
        
        Parameters
        ----------
        xyz: numpy.ndarray (float)
            Px3 array with coordinates, assuming 'white' and 'pial' surfaces have P nodes each
        
        Returns
        -------
        weights: numpy.ndarray (float)
            P values of relative grey matter positions, where 0=white surface and 1=pial surface
        '''

        # compute relative to pial_xyz
        pxyz = self._pial.v()
        qxyz = self._white.v()

        dxyz = qxyz - pxyz # difference vector

        scale = np.sum(dxyz * dxyz, axis=1)

        ps = xyz - pxyz
        proj = np.sum(ps * dxyz, axis=1)

        return proj / scale



    def _img_count_voxels(self, c2v):
        '''just for testing that centernode to voxel mapping actually works'''
        v = self._volgeom
        nv, nt = v.nv(), v.nt()

        voldata = np.zeros((nv,), dtype=float)

        for i, vx2d in c2v.iteritems():
            #vxf=[x for x in vx if x==x]
            if vx2d:
                for vx in vx2d:
                    voldata[vx] += 1

        rs = np.reshape(voldata, v.shape())
        img = ni.Nifti1Image(rs, v.affine())
        return img

def _test_voxsel():
    d = '%s/qref/' % utils._get_fingerdata_dir()
    epifn = d + "../glm/rall_vol00.nii"

    ld = 36 * 4 # mapicosahedron linear divisions
    smallld = 9
    hemi = 'l'
    nodecount = 10 * ld ** 2 + 2

    pialfn = d + "ico%d_%sh.pial_al.asc" % (ld, hemi)
    whitefn = d + "ico%d_%sh.smoothwm_al.asc" % (ld, hemi)

    intermediatefn = d + "ico%d_%sh.intermediate_al.asc" % (smallld, hemi)

    fnoutprefix = d + "_voxsel1_ico%d_%sh_" % (ld, hemi)
    radius = 100 # 100 voxels per searchlight
    srcs = range(nodecount) # use all nodes as a center

    vg = volgeom.from_nifti_file(epifn)

    p, i, w = map(surf_fs_asc.read, (pialfn, intermediatefn, whitefn))

    vs = VolSurf(vg, p, w)
    print "Made vs"
    n2v = vs.node2voxels()
    print "Done n2v"
    #print n2v

    img = vs._img_count_voxels(n2v)
    print "Made image"

    fnout = d + "__test_n2v2.nii"
    img.to_filename(fnout)

    '''
    sel=run_voxelselection(epifn, whitefn, pialfn, radius, srcs,require_center_in_gm=require_center_in_gm)
    print "Completed voxel selection"
    
    # save voxel selection results
    f=open(fnoutprefix + ".pickle",'w')
    pickle.dump(sel, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
    _voxelselection_write_vol_and_surf_files(sel,fnoutprefix)
    print "Data saved to %s.{pickle,nii,niml.dset}" % fnoutprefix
    '''

def _demo_small_voxelsection():
    d = '%s/qref/' % utils._get_fingerdata_dir()
    epifn = d + "../glm/rall_vol00.nii"

    ld = 36 # mapicosahedron linear divisions
    hemi = 'l'
    nodecount = 10 * ld ** 2 + 2


    pialfn = d + "ico%d_%sh.pial_al.asc" % (ld, hemi)
    whitefn = d + "ico%d_%sh.smoothwm_al.asc" % (ld, hemi)

    fnoutprefix = d + "_voxsel1_ico%d_%sh_" % (ld, hemi)
    radius = 100 # 100 voxels per searchlight
    srcs = range(nodecount) # use all nodes as a center

    require_center_in_gm = False # setting this to True is not a good idea at the moment - not well tested

    print "Starting voxel selection"
    sel = run_voxelselection(epifn, whitefn, pialfn, radius, srcs, require_center_in_gm=require_center_in_gm)
    print "Completed voxel selection"

    # save voxel selection results
    f = open(fnoutprefix + ".pickle", 'w')
    pickle.dump(sel, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    _voxelselection_write_vol_and_surf_files(sel, fnoutprefix)
    print "Data saved to %s.{pickle,nii,niml.dset}" % fnoutprefix


def _demo_voxelsection():
    d = '%s/glm/' % utils._get_fingerdata_dir()
    epifn = d + "rall_vol00.nii"

    pialfn = d + "ico100_lh.pial_al.asc"
    whitefn = d + "ico100_lh.smoothwm_al.asc"

    fnoutprefix = d + "_voxsel1_"
    radius = 100 # 100 voxels per searchlight
    srcs = range(100002) # use all nodes as a center

    require_center_in_gm = False # setting this to True is not a good idea at the moment - not well tested

    print "Starting voxel selection"
    sel = run_voxelselection(epifn, whitefn, pialfn, radius, srcs, require_center_in_gm=require_center_in_gm)
    print "Completed voxel selection"

    # save voxel selection results
    f = open(fnoutprefix + ".pickle", 'w')
    pickle.dump(sel, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()

    _voxelselection_write_vol_and_surf_files(sel, fnoutprefix)
    print "Data saved to %s.{pickle,nii,niml.dset}" % fnoutprefix


def _voxelselection_write_vol_and_surf_files(sel, fnoutprefix):
    # write a couple of files to store results:
    # - volume file with number of times each voxel was selected
    # - surface file with searchlight radius

    vg = sel.getvolgeom() # volume geometry
    nvox = vg.nv() # number of voxels
    voldata = np.zeros((nvox, 2), dtype=np.float) # allocate space for volume file, in linear shape

    roilabels = sel.getroilabels() # these are actually the node indices (0..100001)
    nlabels = len(roilabels) # number of ROIs (nodes), 100002

    surfdata = np.zeros((nlabels, 1)) # allocate space for surface file

    # go over the surface nodes
    for i, roilabel in enumerate(roilabels):
        # get a binary mask for this ROI (indexed by roilabel)
        msk = sel.getbinaryroimask(roilabel)

        # add 1 to all voxels that are around the present nodes 
        voldata[msk, 0] += 1.

        # alternative way to get the linear voxel indices associated with this
        # center node (ROI)
        idxs = sel.getroimeta(roilabel, sparse_volmasks._VOXIDXSLABEL)

        # get the distances for each voxel from this node
        ds = sel.getroimeta(roilabel, sparse_volmasks._VOXDISTLABEL)

        # set distances for these voxels (overwrite if the distances are 
        # already set for these voxels 
        voldata[idxs, 1] = ds

        # for the node on the surfae, set the maximum distance 
        # (i.e. searchlight radius) 
        surfdata[i, 0] = ds[-1]

    print "Prepared voldata and surfdata"

    sh = list(vg.shape()) # get the shape of the original volume (3 dimensions)
    sh.append(voldata.shape[-1]) # last dimension 
    datars = np.reshape(voldata, tuple(sh)) # reshape it to 4D

    # save volume data
    img = ni.Nifti1Image(datars, vg.affine())
    img.to_filename(fnoutprefix + ".nii")
    print "Written volume voldata"

    # save surface data
    s = dict(data=surfdata, node_indices=roilabels)
    afni_niml_dset.write(fnoutprefix + ".niml.dset", s, 'text')
    print "Written surface voldata"

if __name__ == '__main__':
    #_demo_voxelsection()
    _test_voxsel()

