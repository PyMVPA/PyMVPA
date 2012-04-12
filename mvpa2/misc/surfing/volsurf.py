'''
Functionality for surface-based voxel selection

Created on Feb 13, 2012

@author: nick

References
----------
NN Oosterhof, T Wiestler, PE Downing (2011). A comparison of volume-based 
and surface-based multi-voxel pattern analysis. Neuroimage, 56(2), pp. 593-600

'Surfing' toolbox: http://surfing.sourceforge.net 
(and the associated documentation)
'''

import surf, volgeom, collections, surf_fs_asc, numpy as np, utils, time, nibabel as ni, operator, random
import sparse_volmasks, afni_niml_dset, cPickle as pickle
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
        
    Note
    ----
    'pial' and 'white' should have the same topology
    '''
    
    def __init__(self,volgeom,pial,white):
        if not pial.same_topology(white):
            raise Exception("Not same topology for white and pial")
        
        self._volgeom=volgeom
        self._pial=pial
        self._white=white
        
    def node2voxels(self,steps=10,start=0.0,stop=1.0,require_center_in_gm=False):
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
        require_center_in_gm: bool (default: False)
            Only select voxels that are 'truly' in between the white and pial matter.
            Specifically, each voxel's position is projected on the line connecting pial-
            white matter pairs, and only voxels in between 'start' and 'stop' are selected.
            At the moment using True is not well tested and use is discouraged
            
              
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
        if start>stop or steps<1:
            raise ValueError("Illegal start/stop combination, or not enough steps")
        
        # make a list of the different relative gray matter positions
        if steps>1:
            step=(stop-start)/float(steps-1)
        else:
            step=0
            start=stop=.5
        
        # node to voxels mapping
        # if n2v[i]=vs, then node i is associated with the voxels vs
        #
        # vs is a mapping from indices to relative position in grey matter
        # wheere 0 means white surface and 1 means pial surface
        # vs[k]=pos means that voxel with linear index k is 
        # associated with relative positions pos0
        #
        # CHECKME that I did not confuse pial and white surface

        nv=self._pial.nv() # number of nodes on the surface
        n2vs=dict() # node to voxel indices mapping
        for j in xrange(nv):
            n2vs[j]=None # by default, no voxels associated with each node
        
        volgeom=self._volgeom
        
        # different 'layers' (depths) in the grey matter
        for i in xrange(steps):
            pialweight=start+step*i
            whiteweight=1-pialweight
            
            # compute weighted intermediate surface in between pial and white
            surf=self._pial*pialweight+self._white*whiteweight
            
            surfxyz=surf.v() # coordinates
            linvox=volgeom.xyz2lin(surfxyz) # linear indices of voxels containing nodes
            
            voxinvol=volgeom.lininvol(linvox) # which of these voxels are actually in the volume
            
            volxyz=volgeom.lin2xyz(linvox) # coordinates of voxels
            
            # compute relative position of each voxel in grey matter
            relpos=self.surf_project_weights_nodewise(volxyz) 
                        
            for j in xrange(nv): # for each node on the surface
                # associate voxels with the present center node
                # - center node must be in volume
                # - if onlnyinbetween is required, then also make sure that the 
                #   relative position is in between 0 and 1 (or whatever the values of
                #   start and stop are)
                
                if (voxinvol[j] and ((not require_center_in_gm) or start<=relpos[j]<=stop)):
                    if n2vs[j] is None: # no voxels yet assocaited with this node
                        n2vs[j]=dict()  
                    
                    n2vs[j][linvox[j]]=relpos[j]
        
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
        
        pxyz=self._pial.v()
        weights=self.surf_project_weights(xyz)
        return pxyz+np.reshape(weights,(-1,1))*pxyz
    
    def surf_project_weights_nodewise(self,xyz):
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
        pxyz=self._pial.v()
        qxyz=self._white.v()
        
        dxyz=qxyz-pxyz # difference vector
         
        scale=np.sum(dxyz*dxyz,axis=1)
        
        ps=xyz-pxyz
        proj=np.sum(ps*dxyz,axis=1)
        
        return proj / scale
  
    
    
    def _setmask(self,c2v):
        '''just for testing that centernode to voxel mapping actually works'''
        v=self._volgeom
        nv,nt=v.nv(),v.nt()
        
        voldata=np.zeros((nv,),dtype=float)
                
        for i,vxd in c2v.iteritems():
            #vxf=[x for x in vx if x==x]
            vxf=[vx[0] for vx in list(vxd)]
            #voldata[vxf]+=1
            voldata[vxf]=1
            
        rs=np.reshape(voldata,v.shape())
        img=ni.Nifti1Image(rs,v.affine())
        return img


class _RadiusOptimizer():
    '''
    Internal class to optimize the initial radius used for voxel selection.
    
    In the case of selecting a fixed number of voxels in each searchlight, the 
    required radius will vary across searchlights. The general strategy is to take
    some initial radius, find the nodes that are within that radius, select the 
    corresponding voxels, and see if enough voxels are selected. If not, the radius is 
    increased and these steps repeated.
    
    A larger initial radius means a decrease in the probability that not enough voxels are 
    selected, but an increase in time to compute distances and select voxels.
    
    The challenge therefore to find the optimal initial radius so that overall computational
    time is minimized.
    
    The present implementation is very stupid and just increases the radius every time 
    by a factor of 1.5.
    
    
    '''
    def __init__(self,initradius):
        '''new instance, with certain initial radius'''
        self._initradius=initradius
        self._initmult=1.5
        
    def getnew(self):
        '''get an (initial) radius for a new searchlight.'''
        self._curradius=self._initradius
        self._count=0
        return self._curradius
        
    def getbetter(self):
        '''get a new (better=larger) radius for the current searchlight'''
        self._count+=1
        self._curradius*=self._initmult
        return self._curradius
        
    def setfinal(self,finalradius):
        '''to tell what the final radius was that satisfied the number of required voxels'''
        pass
    
    def __repr__(self):
        return 'radius is %f, %d steps' % (self._curradius, self._count) 

class VoxelSelector():
    '''
    Voxel selection using the surfaces
    
    Parameters
    ----------
    radius: int or float
        Searchlight radius. If the type is int, then this set the number of voxels
        in each searchlight (with variable radii (in metric distance) across searchlights). 
        If the type is float, then this sets the radius in metric distance (with variable number of
        voxels across searchlights). In the latter case, the distance unit is usually in milimeters 
        (which is the unit used for Freesurfer surfaces)
    surf: surf.Surface
        A surface to be used for distance measurement. Usually this is the intermediate distance
        constructed by taking the node-wise average of the pial and white surface
    n2v: dict
        Mapping from center nodes to surrounding voxels (and their distances). Usually this
        is the output from volsurf.node2voxels(...)
    distancemetric: str
        Distance measure used to define distances between nodes on the surface. 
        Currently supports 'dijkstra' and 'euclidian'     
    '''
    
    def __init__(self,radius,surf,n2v,distancemetric='dijkstra'):
        tp=type(radius)
        if tp is int: # fixed number of voxels
            self._fixedradius=False
            # use a (very arbitary) way to estimate how big the radius should
            # be initally to select the required number of voxels
            initradius_mm=.001+1.5*float(radius)**.5  
        elif tp is float: # fixed metric radius
            self._fixedradius=True
            initradius_mm=radius
        else:
            raise TypeError("Illegal type for radius: expected int or float")
        self._targetradius=radius # radius to achieve (float or int)
        self._initradius_mm=initradius_mm # initial radius in mm
        self._optimizer=_RadiusOptimizer(initradius_mm) 
        self._distancemetric=distancemetric # }
        self._surf=surf                     # } save input
        self._n2v=n2v                       # }
    
    def _select_approx(self,voxprops,count=None):
        '''
        Select approximately a certain number of voxels.
        
        Parameters
        ----------
        voxprops: dict
            Various voxel properties, where voxprops[key] is a list with
            voxprops[key][i] property key for voxel i. 
            Should at least have a key 'distances'.
            Each of voxprops[key] should be list-like of equal length.
        count: int (default: None)
            How many voxels should be selected, approximately
            
        Returns
        -------
        v2d_sel: dict
            Similar mapping as the input, with approximately 'count' voxels
            with the smallest distance in 'voxprops'
            If 'count is None' then 'v2d_sel is None'
            If voxprops has fewer than 'count' elemens then 'v2d_sel' is None
            
        Note
        ----
        Distances are only computed along the surface; the relative position of 
        a voxel within the gray matter is ignored. Therefore, multiple voxels
        can have the same distance from a center node. See node2voxels
        ''' 
        
        if count is None:
            return voxprops
        
        distkey=sparse_volmasks._VOXDISTLABEL
        if not distkey in voxprops:
            raise KeyError("voxprops has no distance key %s in it - cannot select voxels" % distkey)
        allds=voxprops[distkey]
        
        #assert sorted(allds)==allds #that's what voxprops should give us 
        
        n=len(allds)
        
        if n<count or n==0:
            return None
                
        # here, a 'chunk' is a set of voxels at the same distance. voxels are selected in chunks
        # with increasing distance. either all voxels in a chunk are selected or none.
        curchunk=[]
        prevd=allds[0]
        chunkcount=1
        for i in xrange(n):
            d=allds[i] # distance  
            if i>0 and prevd!=d:
                if i>=count: # we're done, use the chunk we have now
                    break
                curchunk=[i] # start a new chunk
                chunkcount+=1
            else:
                curchunk.append(i)
            
            prevd=d
        
        # see if the last chunk should be added or not to be as close as 
        # possible to count
        firstpos=curchunk[0]
        lastpos=curchunk[-1]
                
        # difference in distance between desired count and positions
        delta=(count-firstpos) - (lastpos-count)
        if delta>0:
            # lastpos is closer to count
            cutpos=lastpos+1
        elif delta<0:
            # firstpos is closer to count
            cutpos=firstpos
        else:
            # it's a tie, choose quasi-randomly based on chunkcount
            cutpos=firstpos if chunkcount%2==0 else (lastpos+1) 
        
        for k in voxprops.keys():
            voxprops[k]=voxprops[k][:cutpos]
        
        return voxprops
        
   
    def select_multiple(self,srcs=None,etastep=None):
        '''
        Voxel selection for multiple center nodes
        
        Parameters
        ----------
        srcs: array-like
            Indices of center nodes to be used as searchlight center.
            If None, then all center nodes are used as a center.
        etastep: int or None
            After how many searchlights the the estimated remaining time 
            are printed.
            If None, then no messages are printed.
        
        Returns
        -------
        n2vs: sparse_volmasks.SparseVolMask
            node to voxel properties mapping, as represented in a set of 
            sparse volume masks.
        '''
        
        if srcs is None:
            srcs=np.arange(len(self._n2v))
            
        n=len(srcs)
        visitorder=list(np.random.permutation(n)) # for better ETA estimate
        
        tstart=time.time()
        
        n2vs=sparse_volmasks.SparseVolMask()
        
        for i,order in enumerate(visitorder):
            src=srcs[order]
            n2v=self.select_one(src)
            
            n2vs.addroi_fromdict(src, n2v)
        
            if etastep and (i%etastep==0 or i==n-1) and n2v:
                utils.eta(tstart, float(i+1)/n, '%d/%d: %d' % (i,n,src))
            
        return n2vs
        
    def select_one(self,src): 
        '''
        Voxel selection for single center node
        
        Parameters
        ----------
        src: int
            Index of center node to be used as searchlight center
            
        Returns
        -------
        voxprops: dict
            Various voxel properties, where voxprops[key] is a list with
            voxprops[key][i] property key for voxel i. 
            Has at least a key 'distances'.
            Each of voxprops[key] should be list-like of equal length.
        '''
        optimizer=self._optimizer
        surf=self._surf
        n2v=self._n2v
        
        if not src in n2v or n2v[src] is None:
            # no voxels associated with this node, skip
            print "Skipping %d" % src
            voxaround=[]
        else:            
            radius_mm=optimizer.getnew()
            radius=self._targetradius
            
            
            while True:
                around_n2d=surf.circlearound_n2d(src,radius_mm,self._distancemetric)
                
                allvxdist=self.nodes2voxeldist(around_n2d,n2v)
                
                if not allvxdist:
                    voxaround=[]
                            
                if self._fixedradius:
                    # select all voxels
                    voxaround=self._select_approx(allvxdist, count=None)
                else:
                    # select only certain number
                    voxaround=self._select_approx(allvxdist, count=radius)
                
                if voxaround is None:
                    # coult not find enough voxels, stay in loop and try again 
                    # with bigger radius
                    radius_mm=optimizer.getbetter()
                else:
                    break
                
        
        if voxaround:
            # found at least one voxel; update our ioptimizer
            maxradius=voxaround['distances'][-1]
            optimizer.setfinal(maxradius)
                
        return voxaround
            
    def nodes2voxeldist(self,n2d,n2v,distancesummary=min):
        '''
        Computes voxel distances 
        
        Parameters
        ----------
        n2d: dict
            A mapping from node indices to distances (to a center node)
            Usually this is the output from surf.circlearound_n2d and thus
            only contains voldata for voxels surrounding a single center node
        n2v: dict
            A mapping from nodes to surrounding voxel indices and distances.
            n2v[i]=v2d is a dict mapping node i to a dict v2d, which in turn
            maps voxel indices to distances to the center node (i.e. v2d[j]=d
            means that the distance from voxel with linear index j to the
            node with index i is d
        distancesummary: function
            This is by default the min function. It is used to summarize
            cases where a single voxels has multiple distances (and nodes)
            associated with it. By default we take the minimum distance, and
            the node that gives rise to this distance, as a representative 
            for the distance.
            
        Returns
        -------
        voxelprops: dict
            Mapping from keys to lists that contain voxel properties.
            Each list should have the same length
            It has at least a key sparse_volmasks._VOXIDXSLABEL which maps to
            the linear voxel indices. It may also have 'distances' (distance from
            center node along the cortex)  and 'gmpositions' (relative position in
            the gray matter) 
        
        '''
        
        
        '''takes the node to distance mapping n2d, and the node to voxel mapping n2v,
        and returns a pair of lists with voxel indices and distances
        If a voxel is associated with multiple nodes (i.e. n2d[i] and n2d[j] are not disjunct
        for i!=j, then the voxel with minimum value for distance is taken'''
        # mapping from voxel indices to all distances
        v2dps=collections.defaultdict(set)

        # get node indices and associated (distance, grey matter positions) 
        for nd, d in n2d.iteritems():
            if nd in n2v:
                vps=n2v[nd] # all voxels associated with this node
                if not vps is None:
                    for vx,pos in vps.items():
                        v2dps[vx].add((d,pos)) # associate voxel with tuple of distance and relative position
            

        # converts a tuple (vx, set([(d0,p0),(d1,p1),...]) to a triple (vx,pM,dM)
        # where dM is the minimum across all d*
        def unpack_dp(vx,dp,distancesummary=distancesummary):
            d,p=distancesummary(dp) # implicit sort by first elemnts first, i.e. distance
            return vx,d,p            
        

        # make triples of (voxel index, distance to center node, relative position in grey matter)            
        vdp=[unpack_dp(vx,dp) for vx,dp in v2dps.iteritems()] 

        # sort triples by distance to center node
        vdp.sort(key=operator.itemgetter(1))
        
        if not vdp:
            v=d=p=[] # empty
        else:
            v,d,p=zip(*vdp) # unzip triples into three lists
        
        def toarray(x):
            return np.asarray(x)
        
        # make this a dict
        asdict=dict(distances=toarray(d),gmpositions=toarray(p))
        
        # ensure that voxel indices have a proper label
        asdict[sparse_volmasks._VOXIDXSLABEL]=toarray(v)
        return asdict


def run_voxelselection(epifn,whitefn,pialfn,radius,srcs=None,start=0,stop=1,steps=10,require_center_in_gm=False,distancemetric='dijkstra',intermediateat=.5,etastep=1):
    '''Wrapper function that is supposed to make voxel selection 
    on the surface easy.
    
    Parameters
    ----------
    epifn: str
        Filename of functional volume in which voxel selection is performed.
        At the moment only nifti (.nii) files are supported
    whitefn: str
        Filename of white matter surface. Only .asc files at the moment
    whitefn: str
        Filename of pial surface. Only .asc files at the moment
    radius: int or float
        Searchlight radius with number of voxels (if int) or maximum distance
        from searchlight center in metric units (if float)
    srcs: list-like or None
        Node indices of searchlight centers. If None, then all nodes are used 
        as a center
    start: float (default: 0)
            Relative start position of line in gray matter, 0.=white surface, 1.=pial surface
            CheckMe: it might be the other way around 
    stop: float (default: 1)
        Relative stop position of line, in gray matter, 0.=white surface, 1.=pial surface
    require_center_in_gm: bool (default: False)
        Only select voxels that are 'truly' in between the white and pial matter.
        Specifically, each voxel's position is projected on the line connecting pial-
        white matter pairs, and only voxels in between 'start' and 'stop' are selected    
    require_center_in_gm: bool (default: False)
        Accept only voxels that fall in the grey matter. Not well tested at the moment,
        use is currently discouraged 
    distancemetric: str
        Distance metric between nodes. 'euclidian' or 'dijksta'
    intermediateat: float (default: .5)
        Relative positiion of intermediate surface that is used to measure distances.
        By default this is the average of the pial and white surface
    etastep: int (default: 1)
        After how many searchlights an estimate should be printed of the remaining
        time until completion of all searchlights
    
    Returns
    -------
    sel: sparse_volmasks.SparseVolMask
        Voxel selection results, that associates, which each node, the indices
        of the surrounding voxels.
    '''
    
    
    # read volume geometry
    vg=volgeom.from_nifti_filename(epifn)
    
    # read surfaces
    whitesurf=surf_fs_asc.read(whitefn)
    pialsurf=surf_fs_asc.read(pialfn)
    
    # compute itnermediate surface
    intermediatesurf=whitesurf*(1-intermediateat)+pialsurf*intermediateat
    
    # make a volume surface instance
    vs=VolSurf(vg,whitesurf,pialsurf)
    
    # find mapping from nodes to enclosing voxels
    n2v=vs.node2voxels(steps, start, stop, require_center_in_gm)
    
    # make a voxel selection instance
    voxsel=VoxelSelector(radius, intermediatesurf, n2v, distancemetric)
    
    # run voxel selection
    sel=voxsel.select_multiple(srcs, etastep)
    
    # store the volgemoetry results in the selection results
    sel.setvolgeom(vg)
    
    return sel

def _demo_small_voxelsection():
    d='%s/qref/' % utils._get_fingerdata_dir()
    epifn=d+"../glm/rall_vol00.nii"
    
    ld=36 # mapicosahedron linear divisions
    hemi='l'
    nodecount=10*ld**2+2
    
    
    pialfn=d+"ico%d_%sh.pial_al.asc" % (ld,hemi)
    whitefn=d+"ico%d_%sh.smoothwm_al.asc" % (ld,hemi)
    
    fnoutprefix=d+"_voxsel1_ico%d_%sh_" % (ld,hemi)
    radius=100 # 100 voxels per searchlight
    srcs=range(nodecount) # use all nodes as a center
    
    require_center_in_gm=False # setting this to True is not a good idea at the moment - not well tested
    
    print "Starting voxel selection"
    sel=run_voxelselection(epifn, whitefn, pialfn, radius, srcs,require_center_in_gm=require_center_in_gm)
    print "Completed voxel selection"
    
    # save voxel selection results
    f=open(fnoutprefix + ".pickle",'w')
    pickle.dump(sel, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
    _voxelselection_write_vol_and_surf_files(sel,fnoutprefix)
    print "Data saved to %s.{pickle,nii,niml.dset}" % fnoutprefix

    
def _demo_voxelsection():
    d='%s/glm/' % utils._get_fingerdata_dir()
    epifn=d+"rall_vol00.nii"
    
    pialfn=d+"ico100_lh.pial_al.asc"
    whitefn=d+"ico100_lh.smoothwm_al.asc"
    
    fnoutprefix=d+"_voxsel1_"
    radius=100 # 100 voxels per searchlight
    srcs=range(100002) # use all nodes as a center
    
    require_center_in_gm=False # setting this to True is not a good idea at the moment - not well tested
    
    print "Starting voxel selection"
    sel=run_voxelselection(epifn, whitefn, pialfn, radius, srcs,require_center_in_gm=require_center_in_gm)
    print "Completed voxel selection"
    
    # save voxel selection results
    f=open(fnoutprefix + ".pickle",'w')
    pickle.dump(sel, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
    _voxelselection_write_vol_and_surf_files(sel,fnoutprefix)
    print "Data saved to %s.{pickle,nii,niml.dset}" % fnoutprefix


def _voxelselection_write_vol_and_surf_files(sel,fnoutprefix):
    # write a couple of files to store results:
    # - volume file with number of times each voxel was selected
    # - surface file with searchlight radius
    
    vg=sel.getvolgeom() # volume geometry
    nvox=vg.nv() # number of voxels
    voldata=np.zeros((nvox,2),dtype=np.float) # allocate space for volume file, in linear shape
    
    roilabels=sel.getroilabels() # these are actually the node indices (0..100001)
    nlabels=len(roilabels) # number of ROIs (nodes), 100002
    
    surfdata=np.zeros((nlabels,1)) # allocate space for surface file
    
    # go over the surface nodes
    for i,roilabel in enumerate(roilabels):
        # get a binary mask for this ROI (indexed by roilabel)
        msk=sel.getbinaryroimask(roilabel)
        
        # add 1 to all voxels that are around the present nodes 
        voldata[msk,0]+=1.
        
        # alternative way to get the linear voxel indices associated with this
        # center node (ROI)
        idxs=sel.getroimeta(roilabel,sparse_volmasks._VOXIDXSLABEL)
        
        # get the distances for each voxel from this node
        ds=sel.getroimeta(roilabel,sparse_volmasks._VOXDISTLABEL)
        
        # set distances for these voxels (overwrite if the distances are 
        # already set for these voxels 
        voldata[idxs,1]=ds
        
        # for the node on the surfae, set the maximum distance 
        # (i.e. searchlight radius) 
        surfdata[i,0]=ds[-1]
        
    print "Prepared voldata and surfdata"    
    
    sh=list(vg.shape()) # get the shape of the original volume (3 dimensions)
    sh.append(voldata.shape[-1]) # last dimension 
    datars=np.reshape(voldata, tuple(sh)) # reshape it to 4D
    
    # save volume data
    img=ni.Nifti1Image(datars,vg.affine())
    img.to_filename(fnoutprefix + ".nii")
    print "Written volume voldata"
    
    # save surface data
    s=dict(data=surfdata,node_indices=roilabels)
    afni_niml_dset.write(fnoutprefix +".niml.dset", s, 'text')
    print "Written surface voldata"

if __name__ == '__main__':
    #_demo_voxelsection()
    _demo_small_voxelsection()

    