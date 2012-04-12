'''
General support for cortical surface meshes

Created on Feb 11, 2012

@author: nick
'''

import numpy as np, os, collections, networkx as nx, datetime, time, utils, heapq, afni_suma_1d, math

class Surface(object):
    '''Cortical surface mesh
    
    A surface consists of a set of vertices (each with a x, y, and z coordinate)
    and a set of faces (triangles; each has three indices referring to the vertices
    that make up the triangles)
    
    In the present implementation new surfaces should be made using the __init__ 
    constructor; internal fields should not be changed manually
    
    Parameters
    ----------
    v : numpy.ndarray (float)
        Px3 array with coordinates for P vertices
    f : numpy.ndarray (int)
        Qx3 array with vertex indices for Q faces (triangles)
    check: boolean (default=True)
        Do some sanity checks to ensure that v and f have proper size and values
        
    Returns
    -------
    s : Surface
        a surface specified by v and f
    '''
    def __init__(self,v=None,f=None,check=True):
        if not (v is None or f is None):
            self._v=np.asarray(v)
            self._f=np.asarray(f)
            self._nv=v.shape[0]
            self._nf=f.shape[0]
            
        else:
            raise Exception("Cannot make new surface from nothing")
                
        if check:
            self._check()

    def _check(self):
        '''ensures that different fields are sort of consistent'''
        fields=['_v','_f','_nv','_nf']
        if not all(hasattr(self,field) for field in fields):
            raise Exception("Incomplete surface!")
        if self._v.shape!=(self._nv,3):
            raise Exception("Wrong shape for vertices")
        if self._f.shape!=(self._nf,3):
            raise Exception("Wrong shape for faces")
        if (np.unique(self._f)!=np.arange(self._nv)).any():
            raise Exception("Missing values in faces")
    
    def node2faces(self):
        '''
        A mapping from node indices to the faces that contain those nodes.
        
        Returns
        -------
        n2v : dict
            A dict "n2v" so that "n2v[i]=faceidxs" contains a list of the faces 
            (indexed by faceidxs) that contain node "i".
        
        '''
        
        if not hasattr(self,'_n2f'):
            # run the first time this function is called            
            n2f=collections.defaultdict(set)
            for i in xrange(self._nf):
                fi=self._f[i]
                for j in xrange(3):
                    p=fi[j]
                    n2f[p].append(i)
            self._n2f=n2f
            
        return self._n2f
    
    def nbrs(self):
        '''Finds the neighbours for each node and their (Euclidian) distance. 
        
        Returns
        -------
        nbrs : dict
            A dict "nbrs" so that "nbrs[i]=n2d" contains the distances from node i
            to the neighbours of node "i" in "n2d". "n2d" is, in turn, a dict 
            so that "n2d[k]=d" is the distance "d" from node "i" to node "j".
        
        Note
        ----
        This function computes nbrs if called for the first time, otherwise
        it caches the results and returns these immediately on the next call'''
        
       
        if not hasattr(self,'_nbrs'):     
            nbrs=collections.defaultdict(dict)
            for i in xrange(self._nf):
                fi=self._f[i]
                
                for j in xrange(3):
                    p=fi[j]
                    q=fi[(j+1) % 3]
                    
                    if p in nbrs and q in nbrs[p]:
                        continue
                
                    pv=self._v[p]
                    qv=self._v[q]
                    
                    # writing this out seems a bit quicker - but have to test
                    sqdist=((pv[0]-qv[0])*(pv[0]-qv[0])
                           +(pv[1]-qv[1])*(pv[1]-qv[1])
                           +(pv[2]-qv[2])*(pv[2]-qv[2]))
                    
                    dist=math.sqrt(sqdist)
                    nbrs[q][p]=dist
                    nbrs[p][q]=dist
            
            self._nbrs=nbrs
            
        return self._nbrs
    
    def circlearound_n2d(self,src,radius,metric='euclidian'):
        '''Finds the distances from a center node to surrounding nodes
        
        Parameters
        ----------
        src : int
            Index of center node
        radius : float
            Maximum distance for other nodes to quallify as a 'surrounding' node.
        metric : string (default: euclidian)
            'euclidian' or 'dijkstra': distance metric
             
        
        Returns
        -------
        n2d : dict
            A dict "n2d" so that n2d[j]=d" is the distance "d" from node "src" to node "j"
        '''
        
        if radius==0:
            return {src:0}
                
        shortmetric=metric.lower()[0] # only take first letter - for now
        
        if shortmetric=='e':
            ds=self.euclidian_distance(src)
            c=dict((nd,d) for (nd,d) in zip(xrange(self._nv),ds) if d<=radius)

        elif shortmetric=='d':
            c=self.dijkstra_distance(src, maxdistance=radius)

        else:
            raise Exception("Unknown metric %s" % metric)
        
        return c
    
    
    def dijkstra_distance(self, src, maxdistance=None):
        '''Computes Dijkstra distance from one node to surrounding nodes
        
        Parameters
        ----------
        src : int
            Index of center (source) node
        maxdistance: float (default: None)
            Maximum distance for a node to qualify as a 'surrounding' node.
            If 'maxdistance is None' then the distances to all nodes is returned
        
        Returns:
        --------
        n2d : dict
            A dict "n2d" so that n2d[j]=d" is the distance "d" from node "src" to node "j"
            
        Note
        ----
        Preliminary analyses show that the Dijkstra distance gives very similar 
        results to geodesic distances (unpublished results, NNO)
        '''    
        
        
        tdist={src:0} # tentative distances
        fdist=dict()  # final distances
        candidates=[]
        heapq.heappush(candidates,(0,src)) # queue of candidates, sorted by tentative distance

        nbrs=self.nbrs()
        
        # algoritm from wikipedia
        while candidates:
            d,i=heapq.heappop(candidates) # distance and index of current candidate
            
            if i in fdist:
                continue # we already have a final distance for this node
            
            nbr=nbrs[i] # neighbours of current candidate
            
            for nbr_i, nbr_d in nbr.items():
                dnew=d+nbr_d
                
                if not maxdistance is None and dnew>maxdistance:
                    continue # skip if too far away
                
                if nbr_i not in tdist or dnew<tdist[nbr_i]:
                    # set distance and append to queue
                    tdist[nbr_i]=dnew
                    heapq.heappush(candidates, (tdist[nbr_i],nbr_i))
            
            fdist[i]=tdist[i] # set final distance
        
        return fdist
    
    def euclidian_distance(self,src,trg=None):
        '''Computes Euclidian distance from one node to other nodes
        
        Parameters
        ----------
        src : int
            Index of center (source) node
        trg : int
            Target node(s) to which the distance is computed.
            If 'trg is None' then distances to all nodes are computed  
        
        Returns:
        --------
        n2d : dict
            A dict "n2d" so that n2d[j]=d" is the distance "d" from node "src" to node "j"
        '''    
        
        if trg is None:
            delta=self._v-self._v[src]
        else:
            delta=self._v[trg]-self._v[src]
        
        delta2=delta*delta
        ss=np.sum(delta2,axis=delta.ndim-1)
        d=np.power(ss,.5)
        return d
        
    def sub_surface(self,src,radius):
        '''Makes a smaller surface consisting of nodes around a center node
        
        Parameters
        ----------
        src : int
            Index of center (source) node
        radius : float
            Lower bound of (Euclidian) distance to 'src' in order to be part
            of the smaller surface. In other words, if a node 'j' is within 'radius'
            from 'src', then 'j' is also part of the resulting surface. 
        
        Returns
        -------
        small_surf: Surface
            a smaller surface containing nodes surrounding 'src'
        nsel: np.array (int)
            indices of nodes selected from the original surface
        fsel: np.array (int)
            indices of faces selected from the original surface
        orig_src: int
            index of 'src' in the original surface
            
        Note
        ----
        This function is a port from the Matlab surfing toolbox function 'surfing_subsurface'.
        
        With the 'dijkstra_distance' function, this function is more or less obsolete.
        
         
        ''' 
        n2f=self.node2faces() 
        
        msk=self.euclidian_distance(src)<=radius
        
        vidxs=[i for i,m in enumerate(msk) if m] # node indices of those within distance r
        
        funq=list(set.union(*[n2f[vidx] for vidx in vidxs])) # unique face indices that contain nodes within that distance
        fsel=self._f[funq,:] # these are the node indices contained in one of the faces
        nsel,q=np.unique(fsel,return_inverse=True) # selected nodes
       
        nsel=np.array(nsel,dtype=int)
        fsel=np.array(fsel,dtype=int)
       
        sv=self._v[nsel,:] # sub_surface from selected nodes
        sf=np.reshape(q, (-1,3)) # corresponding faces
        
        ss=Surface(v=sv,f=sf,check=False) # make a new sub_surface
        
        # find the src node corresponding to the sub_surface
        for ss_src,sel in enumerate(nsel):
            if sel==src:
                break
        else:
            # do not expect this, but for now it's ok
            ss_src=None
        
        return ss, nsel, fsel, ss_src 

    def __repr__(self):
        s=[]
        s.append("%d nodes, %d faces" % (self._nv, self._nf))
        
        nfirst=3 # how many of first and last nodes and faces to show
        
        def getrange(n,nfirst=nfirst):
            # gets the indices of first and last nodes (or all, if there
            # are only a few)
            if n<2*nfirst:
                return xrange(n)
            else:
                r=range(nfirst)
                r.extend(range(n-nfirst,n))
                return r
        
        def getlist(vs, prefix):
            s=[]
            n=vs.shape[0]
            for i in getrange(n):
                s.append('%s %8d: %r' % (prefix,i,vs[i]))
            return s
        
        s.extend(getlist(self._v, "vertex"))
        s.extend(getlist(self._f, "face"))
        
        return "\n".join(s)
     
    
    
    def same_topology(self,other):
        '''
        Returns whether another surface has the same topology
        
        Parameters
        ----------
        other: surf.Surface
            another surface
        
        Returns
        -------
        bool
            True iff the current surface has the same number of coordinates and the
            same faces as 'other'. '''
        
        return self._v.shape==other._v.shape and (other._f==self._f).all() 
    
    def __add__(self,other):
        '''coordinate-wise addition of two surfaces with the same topology'''
        if not self.same_topology(other):
            raise Exception("Different topologies")
       
        return Surface(v=self._v+other._v,f=self._f,check=False)
    
    def __mul__(self,other):
        '''coordinate-wise scaling'''
        return Surface(v=self._v*other,f=self._f)
    
    # return copies of internal values
    def v(self): 
        '''
        Returns
        -------
        v: numpy.ndarray (int)
            Px3 coordinates for P vertices
        ''' 
        return np.array(self._v)
    
    def f(self): 
        '''
        Returns
        -------
        f: numpy.ndarray (float)
            Qx3 coordinates for Q vertices
        ''' 
        return np.array(self._f)
    
    def nv(self): 
        '''
        Returns
        -------
        nv: int
            Number of vertices
        ''' 
        return self._nv
    
    def nf(self): 
        '''
        Returns
        -------
        nf: int
            Number of faces
        ''' 
        return self._nf
    
    def mapicosahedron_to_high_resolution(self,highres,epsilon=.001):
        nx=self.nv()
        ny=highres.nv()
        
        def getld(n):
            # a mapicosahedron surface with LD linear divisions has
            # N=10*LD^2+2 nodes 
            ld=((nx-2)/10)**2
            if ld!=int(ld):
                raise ValueError("Not from mapicosahedron with %d nodes" % n)
            return int(ld)
             
        
        ldx, ldy=getld(nx), getld(ny)
        
        if ldx>ldy:
            raise ValueError("Other surface has fewer nodes (%d) than this one (%d)" %
                             nx, ny)
        
        mapping=dict()
        x=self.v()
        y=highres.v()
        for i in xrange(nx):
            ds=np.sum((x[i,:]-y)**2,axis=1)
            minpos=np.argmin(ds)
            
            mind=ds[minpos]**.5
            
            if not epsilon is None and mind>epsilon:
                raise ValueError("Not found near node for node %i (min distance %f > %f)" %
                                 i, mind, epsilon)
            mapping[i]=minpos
            
        return mapping
            
        
'''    

def _test_distance():
    d='/Users/nick/Downloads/fingerdata-0.2/glm/'
    fn=d+"rall_vol00.nii"
    fnout=d+"__test8.1D"
    surffn1=d+"../myref/ico100_lh.pial_al.asc"
    import surf_fs_asc
    Surface=surf_fs_asc.read(surffn1)     
    centernode=43523
    
    ds=Surface.dijkstra_distance(centernode)
    nv=Surface.nv()
    data=np.zeros((nv,1))
    for i,d in ds.iteritems():
        data[i]=d
    
    afni_suma_1d.write(fnout, data)
    
def _test_l2h():
    d=utils._get_fingerdata_dir() + "qref/" 
    
    pat=d+'ico%d_lh.intermediate_al.asc'
    lds=[9,72]
    
    fns=map(lambda x : pat % x, lds)
    s_lo, s_hi=map(surf_fs_asc.read,fns)
    
    mapping=s_lo.mapicosahedron_to_high_resolution(s_hi)
    print mapping
       
    

if __name__ == '__main__':
    #_test_distance()
    _test_l2h()
    pass#_test_project()
    
    
    
    
    
 
    rs=[10,20,30,40]
    
    for r in rs:
        ss=s.sub_surface(c,r)
        t=ss.pairdistances(cutoff=cutoff)
        print '%d (%d nodes): %r' % (r, ss._nv, t)
    
    fnout=d+"__test.asc"
    ss.write(fnout,True)
    
    print s
    print s+s
    print s*.5+s*.5
    '''