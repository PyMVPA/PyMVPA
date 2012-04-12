
_CENTER_IDS="center_ids"
_CENTER_DISTANCES="center_distances"
_GREY_MATTER_POSITION="grey_matter_position"

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
            radius_mm=optimizer.get_start()
            radius=self._targetradius
            
            while True:
                around_n2d=surf.circlearound_n2d(src,radius_mm,self._distancemetric)
                
                allvxdist=self.nodes2voxel_attributes(around_n2d,n2v)
                
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
                    radius_mm=optimizer.get_next()
                else:
                    break
                
        
        if voxaround:
            # found at least one voxel; update our ioptimizer
            maxradius=voxaround['distances'][-1]
            optimizer.set_final(maxradius)
                
        return voxaround
            
    def nodes2voxel_attributes(self,n2d,n2v,distancesummary=min):
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
            vdp_tup=([],[],[]) # empty
        else:
            vdp_tup=zip(*vdp) # unzip triples into three lists
        
        vdp_tps=(np.int32,np.float32,np.float32)
        vdp_labels=(_CENTER_IDS,_CENTER_DISTANCES,_GREY_MATTER_POSITION)
        
        voxel_attributes=dict()
        for i in xrange(3):
            voxel_attributes[vdp_labels[i]]=np.asarray(vdp_tup[i],dtype=vtp_tps[i])

        return voxel_attributes


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
        
    def get_start(self):
        '''get an (initial) radius for a new searchlight.'''
        self._curradius=self._initradius
        self._count=0
        return self._curradius
        
    def get_next(self):
        '''get a new (better=larger) radius for the current searchlight'''
        self._count+=1
        self._curradius*=self._initmult
        return self._curradius
        
    def set_final(self,finalradius):
        '''to tell what the final radius was that satisfied the number of required voxels'''
        pass
    
    def __repr__(self):
        return 'radius is %f, %d steps' % (self._curradius, self._count) 
