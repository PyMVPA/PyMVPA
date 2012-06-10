# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
'''
General support for cortical surface meshes

Created on Feb 11, 2012

@author: nick
'''

# yoh: nick, do you have any preference to have trailing whitespace
#      lines through out the code or would be it be ok to remove them?

import numpy as np, os, collections, datetime, time,  \
       heapq, afni_suma_1d, math

class Surface(object):
    '''Cortical surface mesh
    
    A surface consists of a set of vertices (each with an x, y, and z coordinate)
    and a set of faces (triangles; each has three indices referring to the vertices
    that make up a triangle)
    
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
            In other words, nbrs[i][j]=d means that the distance from node i
            to node j is d. It holds that nbrs[i][j]=nbrs[j][i].
        
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
        
        # algoritm from wikipedia (http://en.wikipedia.org/wiki/Dijkstra's_algorithm)
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
        if isinstance(other, Surface):
            if not self.same_topology(other):
                raise Exception("Different topologies - cannot add")
            vother=other._v
        else:
            vother=other
           
        return Surface(v=self._v+vother,f=self._f,check=False)
    
    def __mul__(self,other):
        '''coordinate-wise scaling'''
        
        return Surface(v=self._v*other,f=self._f)
    
    def rotate(self, theta, center=None, unit='rad'):
        if unit.startswith('rad'):
            fac=1.
        elif unit.startswith('deg'):
            fac=math.pi/180.
        else:
            raise ValueError('Illegal unit for rotation: %r' % unit)
        
        theta=map(lambda x:x*fac, theta)
        
        cx,cy,cz = np.cos(theta)
        sx,sy,sz = np.sin(theta)
        
        # rotation matrix *in row-first order*
        # in other words, we compute v*R' 
        m=np.asarray([[cy*cz,-cy*sz,sy],
                      [cx*sz+sx*sy*cz,cx*cz-sx*sy*sz,-sx*cy],
                      [sx*sz-cx*sy*cz,sx*cz+cx*sy*sz,cx*cy]])
        '''
        m=np.asarray([[cx*cz - sx*cy*sz, cx*sz + sx*cy*cz, sx*sy],
                     [-sx*cz - cx*cy*sz, -sx*sz + cx*cy*cz, cx*sy],
                     [sy*sz, -sy*cz, cy]])
        '''
        
        if center is None:
            center=0
        center=np.reshape(np.asarray(center),(1,-1))
        
        
        v_rotate=center+np.dot(self._v-center,m)
        
        return Surface(v=v_rotate,f=self._f)
    
    def center_of_mass(self):
        return np.mean(self.v(),axis=0)
    
    def merge(self, *others):
        all=[self]
        all.extend(list(others))
        n=len(all)
        
        def border_positions(xs,f):
            # positions of transitiions between surface
            # f should return number of relevant values (nodes or vertices) 
            n=len(xs)
            
            fxs=map(f, all)
            
            positions=[0]
            for i in xrange(n):
                positions.append(positions[i]+fxs[i])
                
            zeros_arr=np.zeros((positions[-1],xs[0].v().shape[1]))    
            return positions, zeros_arr
            
        
        pos_v, all_v=border_positions(all, lambda x:x.nv())
        pos_f, all_f=border_positions(all, lambda x:x.nf())
        
        print pos_f
        
        for i in xrange(n):
            all_v[pos_v[i]:pos_v[i+1],:]=all[i].v()
            all_f[pos_f[i]:pos_f[i+1],:]=all[i].f()+pos_v[i]
            
        return Surface(v=all_v, f=all_f)
        
        
        
        
        
    # return copies of internal values
    # yoh: is it intentional to return copies instead of just
    #      original arrays?  why?
    #
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

    # yoh: would you mind if they become @property
    #      and spelled out fully, i.e. nvertices, nfaces
    #      to stay coherent with Dataset.nfeatures, .nsamples?
    #      and I guess the same for above v, f?
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
    
    def map_to_high_resolution_surf(self,highres,epsilon=.001,accept_only_icosahedron=False):
        nx=self.nv()
        ny=highres.nv()
        
        if accept_only_icosahedron:
            def getld(n):
                # a mapicosahedron surface with LD linear divisions has
                # N=10*LD^2+2 nodes 
                ld=((nx-2)/10)**2
                if ld!=int(ld):
                    raise ValueError("Not from mapicosahedron with %d nodes" % n)
                return int(ld)
                 
            ldx,ldy=map(getld,(nx,ny))
            r=ldy/ldx # ratio
            
            if int(r)!=r:
                raise ValueError("ico linear divisions for high res surface (%d)"
                                 "should be multiple of that for low res surface (%d)",
                                 (ldy,ldx))
        
        mapping=dict()
        x=self.v()
        y=highres.v()
        
        # shortcut in case the surfaces are the same
        # if this fails, then we just continue normally
        if self.same_topology(highres):
            d=np.sum((x-y)**2,axis=1)**.5
            
            if all(d<epsilon):
                for i in xrange(nx):
                    mapping[i]=i
                return mapping
        
        if nx>ny:
            raise ValueError("Other surface has fewer nodes (%d) than this one (%d)" %
                             nx, ny)
        
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
#  yoh: tests would be nearly mandatory and should go under mvpa2/tests
#       so why not start by refactoring these with some rudimentary example
#       files if necessary under mvpa2/data?
'''
    
