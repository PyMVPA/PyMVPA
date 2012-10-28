# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##'''
'''
General support for cortical surface meshes

Created on Feb 11, 2012

@author: nick
'''

import os, collections, datetime, time, heapq, math

import numpy as np

class Surface(object):
    '''Cortical surface mesh
    
    A surface consists of a set of vertices (each with an x, y, and z 
    coordinate) and a set of faces (triangles; each has three indices
    referring to the vertices that make up a triangle).
    
    In the present implementation new surfaces should be made using the 
    __init__ constructor; internal fields should not be changed manually
    
    Parameters
    ----------
    vertices : numpy.ndarray (float)
        Px3 array with coordinates for P vertices.
    faces : numpy.ndarray (int)
        Qx3 array with vertex indices for Q faces (triangles).
    check: boolean (default=True)
        Do some sanity checks to ensure that vertices and faces have proper 
        size and values.
        
    Returns
    -------
    s : Surface
        a surface specified by vertices and faces
    '''
    def __init__(self, v=None, f=None, check=True):
        if not (v is None or f is None):
            self._v = np.asarray(v)
            self._f = np.asarray(f)
            self._nv = v.shape[0]
            self._nf = f.shape[0]

        else:
            raise Exception("Cannot make new surface from nothing")

        if check:
            self._check()

    def _check(self):
        '''ensures that different fields are sort of consistent'''
        fields = ['_v', '_f', '_nv', '_nf']
        if not all(hasattr(self, field) for field in fields):
            raise Exception("Incomplete surface!")

        if self._v.shape != (self._nv, 3):
            raise Exception("Wrong shape for vertices")

        if self._f.shape != (self._nf, 3):
            raise Exception("Wrong shape for faces")

        # see if all faces have a corresponding node.
        # actually this would not invalidate the surface, so
        # we only give a warning
        unqf = np.unique(self._f)
        if unqf.size != self._nv:
            from mvpa2.base import warning
            warning("Count mismatch for face range (%d!=%d), "
                            "faces without node: %r" % (unqf.size, self._nv,
                                            set(range(self._nv)) - set(unqf)))


        if np.any(unqf != np.arange(self._nv)):
            from mvpa2.base import warning
            warning("Missing values in faces")

    @property
    def node2faces(self):
        '''
        A mapping from node indices to the faces that contain those nodes.
        
        Returns
        -------
        n2v : dict
            A dict "n2v" so that "n2v[i]=faceidxs" contains a list of the faces
            (indexed by faceidxs) that contain node "i".
        
        '''

        if not hasattr(self, '_n2f'):
            # run the first time this function is called
            n2f = collections.defaultdict(list)
            for i in xrange(self._nf):
                fi = self._f[i]
                for j in xrange(3):
                    p = fi[j]
                    n2f[p].append(i)
            self._n2f = n2f

        return self._n2f

    @property
    def face_edge_length(self):
        '''
        Length of edges associated with each face
        
        Returns
        -------
        f2el: np.ndarray
            Px3 array where P==self.nfaces. f2el[i,:] contains the
            length of the (three) edges that make up face i. 
        '''

        if not hasattr(self, '_f2el'):
            n, f, v = self.nfaces, self.faces, self.vertices

            f2el = np.zeros((n, 3))
            p = v[f[:, 0]] # first coordinate
            for i in xrange(3):
                q = v[f[:, (i + 1) % 3]] # second coordinate
                d = p - q # difference vector

                f2el[:, i] = np.sum(d * d, 1) ** .5 # length
                p = q

            v = f2el.view()
            v.flags.writeable = False
            self._f2el = v

        return self._f2el

    @property
    def average_node_edge_length(self):
        '''
        Average length of edges associated with each face
        
        Returns
        -------
        n2el: np.ndarray
            P-valued vector where P==self.nvertices, where n2el[i] is the 
            average length of the edges that contain node i.
        '''
        if not hasattr(self, '_n2ael'):
            n, v, f = self.nvertices, self.vertices, self.faces

            sum_dist = np.zeros((n,))
            count_dist = np.zeros((n,))
            a = f[:, 0]
            p = v[a]
            for j in xrange(3):
                b = f[:, (j + 1) % 3]
                q = v[b]

                d = np.sum((p - q) ** 2, 1) ** .5

                count_dist[a] += 1
                count_dist[b] += 1

                sum_dist[a] += d
                sum_dist[b] += d

                a = b

            sum_dist[count_dist == 0] = 0
            count_dist[count_dist == 0] = 1

            v = (sum_dist / count_dist).view()
            v.flags.writeable = False
            self._v2ael = v

        return self._v2ael


    @property
    def edge2face(self):
        '''A mapping from edges to the face that contains that edge
        
        Returns
        -------
        e2f: dict
            a mapping from edges to faces. e2f[(i,j)]==f means that
            the edge connecting nodes i and j contains node f.
            It is assumed that faces are consistent with respect to 
            the direction of their normals: if self.faces[j,:]==[p,q,r]
            then the normal of vectors pq and pr should all either point
            'inwards' or 'outwards'.
        '''

        if not hasattr(self, '_e2f'):
            faces = self.faces


            e2f = dict()
            for i in xrange(self.nfaces):
                for j in xrange(3):
                    e = (faces[i, j], faces[i, (j + 1) % 3])
                    if e in e2f:
                        raise ValueError('duplicate key (%d,%d). Do all normals'
                                         ' point in the same "direction"?' % e)
                    e2f[e] = i
            self._e2f = e2f

        return dict(self._e2f) # make a copy



    @property
    def neighbors(self):
        '''Finds the neighbours for each node and their (Euclidean) distance.
        
        Returns
        -------
        nbrs : dict
            A dict "nbrs" so that "nbrs[i]=n2d" contains the distances from 
            node i to the neighbours of node "i" in "n2d". "n2d" is, in turn, 
            a dict so that "n2d[k]=d" is the distance "d" from node "i" to 
            node "j". In other words, nbrs[i][j]=d means that the distance from 
            node i to node j is d. It holds that nbrs[i][j]=nbrs[j][i].
        
        Note
        ----
        This function computes nbrs if called for the first time, otherwise
        it caches the results and returns these immediately on the next call'''


        if not hasattr(self, '_nbrs'):
            nbrs = collections.defaultdict(dict)
            for i in xrange(self._nf):
                fi = self._f[i]

                for j in xrange(3):
                    p = fi[j]
                    q = fi[(j + 1) % 3]

                    if p in nbrs and q in nbrs[p]:
                        continue

                    pv = self._v[p]
                    qv = self._v[q]

                    # writing this out seems a bit quicker - but have to test
                    sqdist = ((pv[0] - qv[0]) * (pv[0] - qv[0])
                           + (pv[1] - qv[1]) * (pv[1] - qv[1])
                           + (pv[2] - qv[2]) * (pv[2] - qv[2]))

                    dist = math.sqrt(sqdist)
                    nbrs[q][p] = dist
                    nbrs[p][q] = dist

            self._nbrs = nbrs

        return self._nbrs

    def circlearound_n2d(self, src, radius, metric='euclidean'):
        '''Finds the distances from a center node to surrounding nodes.
        
        Parameters
        ----------
        src : int
            Index of center node
        radius : float
            Maximum distance for other nodes to qualify as a 'surrounding' 
            node.
        metric : string (default: euclidean)
            'euclidean' or 'dijkstra': distance metric
             
        
        Returns
        -------
        n2d : dict
            A dict "n2d" so that n2d[j]=d" is the distance "d" from node 
            "src" to node "j".
        '''

        if radius == 0:
            return {src:0}

        shortmetric = metric.lower()[0] # only take first letter - for now

        if shortmetric == 'e':
            ds = self.euclidean_distance(src)
            c = dict((nd, d) for (nd, d) in zip(xrange(self._nv), ds) if d <= radius)

        elif shortmetric == 'd':
            c = self.dijkstra_distance(src, maxdistance=radius)

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
            If 'maxdistance is None' then the distances to all nodes is 
            returned/
        
        Returns:
        --------
        n2d : dict
            A dict "n2d" so that n2d[j]=d" is the distance "d" from node 
            "src" to node "j".
            
        Note
        ----
        Preliminary analyses show that the Dijkstra distance gives very similar
        results to geodesic distances (unpublished results, NNO)
        '''


        tdist = {src:0} # tentative distances
        fdist = dict()  # final distances
        candidates = []

        # queue of candidates, sorted by tentative distance
        heapq.heappush(candidates, (0, src))

        nbrs = self.neighbors

        # algorithm from wikipedia 
        # (http://en.wikipedia.org/wiki/Dijkstra's_algorithm)
        while candidates:
            # distance and index of current candidate
            d, i = heapq.heappop(candidates)

            if i in fdist:
                continue # we already have a final distance for this node

            nbr = nbrs[i] # neighbours of current candidate

            for nbr_i, nbr_d in nbr.items():
                dnew = d + nbr_d

                if not maxdistance is None and dnew > maxdistance:
                    continue # skip if too far away

                if nbr_i not in tdist or dnew < tdist[nbr_i]:
                    # set distance and append to queue
                    tdist[nbr_i] = dnew
                    heapq.heappush(candidates, (tdist[nbr_i], nbr_i))

            fdist[i] = tdist[i] # set final distance

        return fdist

    def dijkstra_shortest_path(self, src, maxdistance=None):
        '''Computes Dijkstra shortest path from one node to surrounding nodes.
        
        Parameters
        ----------
        src : int
            Index of center (source) node
        maxdistance: float (default: None)
            Maximum distance for a node to qualify as a 'surrounding' node.
            If 'maxdistance is None' then the shortest path to all nodes is 
            returned.
        
        Returns:
        --------
        n2dp : dict
            A dict "n2d" so that n2d[j]=(d,p)" contains the distance "d" from 
            node  "src" to node "j", and p is a list of the nodes of the path
            with p[0]==src and p[-1]==j.
            
        Note
        ----
        Preliminary analyses show that the Dijkstra distance gives very similar
        results to geodesic distances (unpublished results, NNO)
        '''


        tdist = {src:(0, [src])} # tentative distances and path
        fdist = dict()  # final distances
        candidates = []

        # queue of candidates, sorted by tentative distance
        heapq.heappush(candidates, (0, src))

        nbrs = self.neighbors

        # algorithm from wikipedia 
        #(http://en.wikipedia.org/wiki/Dijkstra's_algorithm)
        while candidates:
            # distance and index of current candidate
            d, i = heapq.heappop(candidates)

            if i in fdist:
                continue # we already have a final distance for this node

            nbr = nbrs[i] # neighbours of current candidate

            for nbr_i, nbr_d in nbr.items():
                dnew = d + nbr_d

                if not maxdistance is None and dnew > maxdistance:
                    continue # skip if too far away

                if nbr_i not in tdist or dnew < tdist[nbr_i][0]:
                    # set distance and append to queue
                    pnew = tdist[i][1] + [nbr_i] # append current node to path
                    tdist[nbr_i] = (dnew, pnew)
                    heapq.heappush(candidates, (tdist[nbr_i][0], nbr_i))

            fdist[i] = tdist[i] # set final distance
        return fdist

    def dijkstra_shortest_path_visiting(self, to_visit):
        '''Computes a list of paths that visit specific nodes
        
        Parameters
        ----------
        to_visit: list of int
            P indices of nodes to visit
        
        Returns
        -------
        path_distances: list of tuple (int, list of int)
            List with (P-1) elements, where the i-th element is a tuple
            (d_i, q_i) with distance d_i between nodes i and (i+1), and 
            q_i a list of node indices on the path between nodes i and (i+1)
            so that q_i[0]==i and q_i[-1]==(i+1)
        '''
        if not to_visit:
            raise ValueError("Cannot operate on empty list")

        src = to_visit[0]
        if len(to_visit) == 1:
            return []

        trg = to_visit[1]

        tdist = {src:(0, [src])} # tentative distances and path
        fdist = dict()  # final distances
        candidates = []

        # queue of candidates, sorted by tentative distance
        heapq.heappush(candidates, (0, src))

        nbrs = self.neighbors

        # algorithm from wikipedia 
        #(http://en.wikipedia.org/wiki/Dijkstra's_algorithm)
        while candidates:
            # distance and index of current candidate
            d, i = heapq.heappop(candidates)

            if i in fdist:
                if i == trg:
                    break
                else:
                    continue # we already have a final distance for this node

            nbr = nbrs[i] # neighbours of current candidate

            for nbr_i, nbr_d in nbr.items():
                dnew = d + nbr_d

                if nbr_i not in tdist or dnew < tdist[nbr_i][0]:
                    # set distance and append to queue
                    pnew = tdist[i][1] + [nbr_i] # append current node to path
                    tdist[nbr_i] = (dnew, pnew)
                    heapq.heappush(candidates, (tdist[nbr_i][0], nbr_i))

            fdist[i] = tdist[i] # set final distance
            if i == trg:
                break

        pth = [fdist[i]]

        # recursion to find remaining paths (if any)
        pth.extend(self.dijkstra_shortest_path_visiting(to_visit[1:]))
        return pth



    def euclidean_distance(self, src, trg=None):
        '''Computes Euclidean distance from one node to other nodes
        
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
            A dict "n2d" so that n2d[j]=d" is the distance "d" from node 
            "src" to node "j".
        '''

        if trg is None:
            delta = self._v - self._v[src]
        else:
            delta = self._v[trg] - self._v[src]

        delta2 = delta * delta
        ss = np.sum(delta2, axis=delta.ndim - 1)
        d = np.power(ss, .5)
        return d

    def nearest_node_index(self, src_coords, node_mask_indices=None):
        '''Computes index of nearest node to src
        
        Parameters
        ----------
        src_coords: numpy.ndarray (Px3 array)
            Coordinates of center
        node_mask_idxs numpy.ndarray (default: None):
            Indices of nodes to consider. By default all nodes are considered
        
        Returns
        -------
        idxs: numpy.ndarray (P-valued vector)
            Indices of nearest nodes
        '''

        if not isinstance(src_coords, np.ndarray):
            src_coords = np.asarray(src_coords)
        if len(src_coords.shape) == 1:
            if src_coords.shape[0] != 3:
                raise ValueError("no three values for src_coords")
            else:
                src_coords = np.reshape(src_coords, (1, -1))
        elif len(src_coords.shape) != 2 or src_coords.shape[1] != 3:
            raise ValueError("Expected Px3 array for src_coords")

        use_mask = not node_mask_indices is None
        # vertices to consider
        v = self.vertices[node_mask_indices] if use_mask else self.vertices

        # indices of these vertices
        all_idxs = np.arange(self.nvertices)
        masked_idxs = all_idxs[node_mask_indices] if use_mask else all_idxs

        n = src_coords.shape[0]
        idxs = np.zeros((n,), dtype=np.int)
        for i in xrange(n):
            delta = v - src_coords[i]
            minidx = np.argmin(np.sum(delta ** 2, 1))
            idxs[i] = masked_idxs[minidx]

        return idxs




    def sub_surface(self, src, radius):
        '''Makes a smaller surface consisting of nodes around a center node
        
        Parameters
        ----------
        src : int
            Index of center (source) node
        radius : float
            Lower bound of (Euclidean) distance to 'src' in order to be part
            of the smaller surface. In other words, if a node 'j' is within 
            'radius' from 'src', then 'j' is also part of the resulting surface.
        
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
        This function is a port from the Matlab surfing toolbox function 
        'surfing_subsurface' (see http://surfing.sourceforge.net)
        
        With the 'dijkstra_distance' function, this function is more or 
        less obsolete.
        
         
        '''
        n2f = self.node2faces

        msk = self.euclidean_distance(src) <= radius

        # node indices of those within distance r
        vidxs = [i for i, m in enumerate(msk) if m]

        # unique face indices that contain nodes within that distance
        funq = list(set.union(*[n2f[vidx] for vidx in vidxs]))

        # these are the node indices contained in one of the faces
        fsel = self._f[funq, :]

        # selected nodes
        nsel, q = np.unique(fsel, return_inverse=True)

        nsel = np.array(nsel, dtype=int)
        fsel = np.array(fsel, dtype=int)

        sv = self._v[nsel, :] # sub_surface from selected nodes
        sf = np.reshape(q, (-1, 3)) # corresponding faces

        ss = Surface(v=sv, f=sf, check=False) # make a new sub_surface

        # find the src node corresponding to the sub_surface
        for ss_src, sel in enumerate(nsel):
            if sel == src:
                break
        else:
            # do not expect this, but for now it's ok
            ss_src = None

        return ss, nsel, fsel, ss_src

    def __repr__(self):
        s = []
        s.append("%d nodes, %d faces" % (self._nv, self._nf))

        nfirst = 3 # how many of first and last nodes and faces to show

        def getrange(n, nfirst=nfirst):
            # gets the indices of first and last nodes (or all, if there
            # are only a few)
            if n < 2 * nfirst:
                return xrange(n)
            else:
                r = range(nfirst)
                r.extend(range(n - nfirst, n))
                return r

        def getlist(vs, prefix):
            s = []
            n = vs.shape[0]
            for i in getrange(n):
                s.append('%s %8d: %r' % (prefix, i, vs[i]))
            return s

        s.extend(getlist(self._v, "vertex"))
        s.extend(getlist(self._f, "face"))

        return "\n".join(s)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        return (np.all(self.vertices == other.vertices) and
                np.all(self.faces == other.faces))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce__(self):
        return (self.__class__, (self._v, self._f))

    def same_topology(self, other):
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

        return self._v.shape == other._v.shape and (other._f == self._f).all()

    def __add__(self, other):
        '''coordinate-wise addition of two surfaces with the same topology'''
        if isinstance(other, Surface):
            if not self.same_topology(other):
                raise Exception("Different topologies - cannot add")
            vother = other._v
        else:
            vother = other

        return Surface(v=self._v + vother, f=self._f, check=False)

    def __mul__(self, other):
        '''coordinate-wise scaling'''

        return Surface(v=self._v * other, f=self._f)

    def rotate(self, theta, center=None, unit='rad'):
        '''Rotates the surface
        
        Parameters
        ----------
        theta:
            np.array with 3 values for rotation along x, y, z axes
        center:
            np.array with center around which surface is rotated. If None,
            then rotation is around the origin (0,0,0).
        unit:
            'rad' or 'deg' for angles in theta in either radians or degrees.
        
        Returns
        -------
        surf.Surface
            the result after rotating with angles theta around center.
        '''

        if unit.startswith('rad'):
            fac = 1.
        elif unit.startswith('deg'):
            fac = math.pi / 180.
        else:
            raise ValueError('Illegal unit for rotation: %r' % unit)

        theta = map(lambda x:x * fac, theta)


        cx, cy, cz = np.cos(theta)
        sx, sy, sz = np.sin(theta)

        # rotation matrix *in row-first order*
        # in other words, we compute vertices*R' 
        m = np.asarray(
                [[cy * cz, -cy * sz, sy],
                 [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
                 [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy]])
        '''
        m=np.asarray([[cx*cz - sx*cy*sz, cx*sz + sx*cy*cz, sx*sy],
                     [-sx*cz - cx*cy*sz, -sx*sz + cx*cy*cz, cx*sy],
                     [sy*sz, -sy*cz, cy]])
        '''

        if center is None:
            center = 0
        center = np.reshape(np.asarray(center), (1, -1))

        v_rotate = center + np.dot(self._v - center, m)

        return Surface(v=v_rotate, f=self._f)

    @property
    def center_of_mass(self):
        '''Computes the center of mass
        
        Returns
        -------
        np.array
            3-value vector with x,y,z coordinates of center of mass
        '''
        return np.mean(self.vertices, axis=0)

    def merge(self, *others):
        '''Merges the present surface with other surfaces
        
        Parameters
        ----------
        others: list of surf.Surface
            List of other surfaces to be merged with present one
        
        Returns
        -------
        surf.Surface
            A surface that has all the nodes of the current surface
            and the surfaces in others, and has the topologies combined
            from these surfaces as well. 
            If the current surface has v_0 vertices and f_0 faces, and the 
            i-th surface has v_i and f_i faces, then the output has
            sum_j (v_j) vertices and sum_j (f_j) faces.
        '''

        all = [self]
        all.extend(list(others))
        n = len(all)

        def border_positions(xs, f, dt):
            # positions of transitions between surface
            # faces should return number of relevant values (nodes or vertices)
            n = len(xs)

            fxs = map(f, all)

            positions = [0]
            for i in xrange(n):
                positions.append(positions[i] + fxs[i])

            zeros_arr = np.zeros((positions[-1], xs[0].vertices.shape[1]),
                                        dtype=dt)
            return positions, zeros_arr


        pos_v, all_v = border_positions(all, lambda x:x.nvertices,
                                                self.vertices.dtype)
        pos_f, all_f = border_positions(all, lambda x:x.nfaces,
                                                self.faces.dtype)

        for i in xrange(n):
            all_v[pos_v[i]:pos_v[i + 1], :] = all[i].vertices
            all_f[pos_f[i]:pos_f[i + 1], :] = all[i].faces + pos_v[i]

        return Surface(v=all_v, f=all_f)



    @property
    def vertices(self):
        '''
        Returns
        -------
        vertices: numpy.ndarray (int)
            Px3 coordinates for P vertices
        '''

        v = self._v.view()
        v.flags.writeable = False

        return v

    @property
    def faces(self):
        '''
        Returns
        -------
        faces: numpy.ndarray (float)
            Qx3 coordinates for Q vertices
        '''
        f = self._f.view()
        f.flags.writeable = False

        return f

    @property
    def nvertices(self):
        '''
        Returns
        -------
        nvertices: int
            Number of vertices
        '''
        return self._nv

    @property
    def nfaces(self):
        '''
        Returns
        -------
        nfaces: int
            Number of faces
        '''
        return self._nf

    def map_to_high_resolution_surf_slow(self, highres, epsilon=.001,
                                         accept_only_icosahedron=False):
        '''
        Finds a mapping to a higher resolution (denser) surface.
        A typical use case is mappings between surfaces generated by
        MapIcosahedron, where the lower resolution surface defines centers
        in a searchlight whereas the higher resolution surfaces is used to
        delineate the grey matter for voxel selection. Unlike the function
        named "map_to_high_resolution_surf", this function is both slow
        and exact---and is actually used in case the former function does
        not find a solution.
        
        Parameters
        ----------
        highres: surf.Surface
            high resolution surface
        epsilon: float
            maximum margin (distance) between nodes mapped from low to
            high resolution surface
        accept_only_icosahedron: bool
            if True, then this function raises an error if the number of
            nodes does not match those which would be expected from
            MapIcosahedorn.
        
        Returns
        -------
        low2high: dict
            mapping so that low2high[i]==j means that node i in the current
            (low-resolution) surface is mapped to node j in the highres
            surface.
            
        '''
        nx = self.nvertices
        ny = highres.nvertices

        if accept_only_icosahedron:
            def getld(n):
                # a mapicosahedron surface with LD linear divisions has
                # N=10*LD^2+2 nodes
                ld = ((nx - 2) / 10) ** 2
                if ld != int(ld):
                    raise ValueError("Not from mapicosahedron with %d nodes" % n)
                return int(ld)

            ldx, ldy = map(getld, (nx, ny))
            r = ldy / ldx # ratio

            if int(r) != r:
                raise ValueError("ico linear divisions for high res surface (%d)"
                                 "should be multiple of that for low res surface (%d)",
                                 (ldy, ldx))

        mapping = dict()
        x = self.vertices
        y = highres.vertices

        # shortcut in case the surfaces are the same
        # if this fails, then we just continue normally
        if self.same_topology(highres):
            d = np.sum((x - y) ** 2, axis=1) ** .5

            if all(d < epsilon):
                for i in xrange(nx):
                    mapping[i] = i
                return mapping

        if nx > ny:
            raise ValueError("Other surface has fewer nodes (%d) than this one (%d)" %
                             (nx, ny))

        for i in xrange(nx):
            ds = np.sum((x[i, :] - y) ** 2, axis=1)
            minpos = np.argmin(ds)

            mind = ds[minpos] ** .5

            if not epsilon is None and mind > epsilon:
                print minpos
                raise ValueError("Not found near node for node %i (min distance %f > %f)" %
                                 (i, mind, epsilon))
            mapping[i] = minpos

        return mapping

    def coordinates_to_box_indices(self, box_size, min_coord=None,
                                                   master=None):
        ''''Boxes' coordinates into triples
        
        Parameters
        ----------
        box_sizes: 
            
        min_coord: triple or ndarray
            Minimum coordinates; maps to (0,0,0). 
            If omitted, it defaults to the mininum coordinates in this surface.
        max_coord: triple or ndarray
            Minimum coordinates; maps to (nboxes[0]-1,nboxes[1]-1,nboxes[2]-1)). 
            If omitted, it defaults to the maximum coordinates in this surface.
        master: Surface.surf (default: None)
            If provided, then min_coord and max_coord are taken from master.
        
        Returns
        -------
        boxes_indices: np.ndarray of float
            Array of size Px3, where P is the number of vertices
        '''

        box_sizes = np.asarray([box_size, box_size, box_size]).ravel()
        box_sizes = np.reshape(box_sizes, (1, 3))

        if not master is None:
            if min_coord:
                raise ValueError('Cannot have both {min,max}_coord and master')
            c = master.vertices
        else:
            c = self.vertices

        if min_coord is None:
            min_coord = np.min(c, 0)
        else:
            min_coord = np.asarray(min_coord).ravel()

        return (self.vertices - min_coord) / box_sizes


    def map_to_high_resolution_surf(self, highres, epsilon=.001,
                                    accept_only_icosahedron=False):
        '''
        Finds a mapping to a higher resolution (denser) surface.
        A typical use case is mappings between surfaces generated by
        MapIcosahedron, where the lower resolution surface defines centers
        in a searchlight whereas the higher resolution surfaces is used to
        delineate the grey matter for voxel selection.
        This function implements an optimization which in most cases
        yields solutions much faster than map_to_high_resolution_surf_exact,
        but may fail to find the correct solution for larger values 
        of epsilon.
        
        Parameters
        ----------
        highres: surf.Surface
            high resolution surface
        epsilon: float
            maximum margin (distance) between nodes mapped from low to
            high resolution surface. Default None, which implies .001.
        accept_only_icosahedron: bool
            if True, then this function raises an error if the number of
            nodes does not match those which would be expected from
            MapIcosahedorn.
        
        Returns
        -------
        low2high: dict
            mapping so that low2high[i]==j means that node i in the current
            (low-resolution) surface is mapped to node j in the highres
            surface.
            
        '''

        #return self.map_to_high_resolution_surf_slow(highres, epsilon, accept_only_icosahedron)

        nx = self.nvertices
        ny = highres.nvertices

        if accept_only_icosahedron:
            def getld(n):
                # a mapicosahedron surface with LD linear divisions has
                # N=10*LD^2+2 nodes
                ld = ((nx - 2) / 10) ** 2
                if ld != int(ld):
                    raise ValueError("Not from mapicosahedron with %d nodes" % n)
                return int(ld)

            ldx, ldy = map(getld, (nx, ny))
            r = ldy / ldx # ratio

            if int(r) != r:
                raise ValueError("ico linear divisions for high res surface (%d)"
                                 "should be multiple of that for low res surface (%d)",
                                 (ldy, ldx))

        mapping = dict()
        x = self.vertices
        y = highres.vertices

        # shortcut in case the surfaces are the same
        # if this fails, then we just continue normally
        if self.same_topology(highres):
            d = np.sum((x - y) ** 2, axis=1) ** .5

            if all(d < epsilon):
                for i in xrange(nx):
                    mapping[i] = i
                return mapping

        if nx > ny:
            raise ValueError("Other surface has fewer nodes (%d) than "
                             "this one (%d)" % (nx, ny))


        # use a fast approach
        # slice up the high and low res in smaller boxes
        # and index them, so that when finding the nearest coordinates
        # it only requires to consider a limited number of nodes
        n_boxes = 20
        box_size = max(np.max(x, 0) - np.min(x, 0)) / n_boxes

        x_boxed = self.coordinates_to_box_indices(box_size, master=highres) + .5
        y_boxed = highres.coordinates_to_box_indices(box_size) + .5

        # get indices of nodes that are very near a box boundary
        delta = epsilon / box_size
        on_borders = np.nonzero(np.logical_or(\
                        np.floor(y_boxed + delta) - np.floor(y_boxed) > 0,
                        np.floor(y_boxed) - np.floor(y_boxed - delta) > 0))[0]

        # on_borders may have duplicates - so get rid of those.
        msk = np.zeros((ny,), dtype=np.int)
        msk[on_borders] = 1
        on_borders = np.nonzero(msk)[0]

        # convert to tuples with integers for indexing
        # (tuples are hashable so can be used as keys in dictionary)
        x_tuples = map(tuple, np.asarray(x_boxed, dtype=np.int))
        y_tuples = map(tuple, np.asarray(y_boxed, dtype=np.int))

        # maps box indices in low-resolution surface to indices
        # of potentially nearby nodes in highres surface 
        x_tuple2near_indices = dict()

        # add border nodes to all low-res surface
        # this is a bit inefficient 
        # TODO optimize to consider neighboorhood
        for x_tuple in x_tuples:
            x_tuple2near_indices[x_tuple] = list(on_borders)

        # for non-border nodes in high-res surface, add them to
        # a single box 
        for i, y_tuple in enumerate(y_tuples):
            if i in on_borders:
                continue # because it was added above

            if not y_tuple in x_tuple2near_indices:
                x_tuple2near_indices[y_tuple] = list()
            x_tuple2near_indices[y_tuple].append(i)

        # it now holds that for every node i in low-res surface (which is
        # identified by t=x_tuples[i]), there is no node j in the high-res surface
        # within distance epsilon for which j in x_tuple2near_indices[t]  

        for i, x_tuple in enumerate(x_tuples):
            idxs = np.asarray(x_tuple2near_indices[x_tuple])

            ds = np.sum((x[i, :] - y[idxs, :]) ** 2, axis=1)
            minpos = np.argmin(ds)

            mind = ds[minpos] ** .5

            if not epsilon is None and mind > epsilon:
                raise ValueError("Not found for node %i: %s > %s" %
                                        (i, mind, epsilon))

            mapping[i] = idxs[minpos]

        return mapping

    @property
    def face_areas(self):
        if not hasattr(self, '_face_areas'):
            f = self.faces
            v = self.vertices

            # consider three sides of each triangle
            a = v[f[:, 0]]
            b = v[f[:, 1]]
            c = v[f[:, 2]]

            # vectors of two sides
            ab = a - b
            ac = a - c

            # area (from wikipedia)
            f2a = .5 * np.sqrt(np.sum(ab * ab, 1) * np.sum(ac * ac, 1) -
                               np.sum(ab * ac, 1) ** 2)

            vw = f2a.view()
            vw.flags.writeable = False
            self._face_areas = vw

        return self._face_areas

    @property
    def node_areas(self):
        if not hasattr(self, '_node_areas'):
            f2a = self.face_areas

            # area is one third of sum of faces that contain the node
            n2a = np.zeros((self.nvertices,))
            for v, fs in self.node2faces.iteritems():
                n2a[v] = sum(f2a[fs]) / 3.

            vw = n2a.view()
            vw.flags.writeable = False
            self._node_areas = vw

        return self._node_areas

    def connected_components(self):
        nv = self.nvertices

        components = []
        visited = set()

        nbrs = self.neighbors
        for i in xrange(nv):
            if i in visited:
                continue

            component = set([i])
            components.append(component)

            nbr = nbrs[i]
            print nbr
            candidates = set(nbr)

            visited.add(i)
            while candidates:
                candidate = candidates.pop()
                component.add(candidate)
                visited.add(candidate)
                nbr = nbrs[candidate]

                for n in nbr:
                    if not n in visited:
                        candidates.add(n)

        return components

    def connected_components_slow(self):
        f, nv, nf = self.faces, self.nvertices, self.nfaces

        node2component = dict()

        def index_component(x):
            if not x in node2component:
                return x, None

            k, v = x, node2component[x]
            while not type(v) is set:
                k, v = v, node2component[v]
                #print k, v, node2component

            return k, v

        for i in xrange(nf):
            p, q, r = f[i]

            #print p, q, r
            #print node2component

            pk, pv = index_component(p)
            qk, qv = index_component(q)
            rk, rv = index_component(r)

            #print pk, pv, " - ", qk, qv, ' - ', rk, rv

            if pv is None:
                if qv is None:
                    if rv is None:
                        node2component[p] = set([p, q, r])
                        node2component[q] = node2component[r] = p
                    else:
                        rv.add(p)
                        rv.add(q)
                        node2component[p] = node2component[q] = rk
                else:
                    if rv is None:
                        qv.add(p)
                        qv.add(r)
                        node2component[p] = node2component[r] = qk
                    else:
                        qv.add(p)
                        node2component[p] = qk
                        if qk != rk:
                            qv.update(rv)
                            node2component[rk] = qk
            else:
                if qv is None:
                    if rv is None:
                        pv.add(q)
                        pv.add(r)
                        node2component[q] = node2component[r] = pk
                    else:
                        if pk != rk:
                            pv.update(rv)
                            node2component[rk] = pk
                        pv.add(q)
                        node2component[q] = pk
                else:
                    if rv is None:
                        if pk != qk:
                            pv.update(qv)
                            node2component[qk] = pk
                        pv.add(r)
                        node2component[r] = pk
                    else:
                        if pk != qk:
                            pv.update(qv)
                            node2component[qk] = pk
                        if pk != rk:
                            if rk != qk:
                                pv.update(rv)
                            node2component[rk] = pk

        components = list()
        for node in xrange(nv):
            v = node2component[node]
            if type(v) is set:
                components.append(v)

        return components


def merge(*surfs):
    if not surfs:
        return None
    s0 = surfs[0]
    return s0.merge(*surfs[1:])

def generate_cube():
    '''
    Generates a cube with sides 2 centered at the origin.
    
    Returns:
    cube: surf.Surface
        A cube with 8 vertices at coordinates (+/-1.,+/-1.,+/-1.),
        and with 12 faces (two for each square side).
    '''

    # map (0,1) to (-1.,1.)
    f = lambda x:float(x) * 2 - 1

    # compute coordinates
    cs = [[f(i / (2 ** j) % 2) for j in xrange(3)] for i in xrange(8)]
    vs = np.asarray(cs)

    # manually set topology
    trias = [(2, 3, 0), (0, 1, 2), (0, 4, 1), (1, 4, 5),
             (1, 5, 2), (2, 5, 6), (2, 6, 3), (3, 6, 7),
             (3, 7, 0), (0, 7, 4), (4, 7, 5), (5, 7, 6)]
    fs = np.asarray(trias, dtype=np.int)

    return Surface(vs, fs)


def generate_sphere(density=10):
    '''
    Generates a sphere-like surface with unit radius centered at the origin.
    
    Parameters
    ----------
    d: int (default: 10)
        Level of detail
    
    Returns
    -------
    sphere: surf.Surface
        A sphere with d**2+2 vertices and 2*d**2 faces. Seen as the planet
        Earth, node 0 and 1 correspond to the north and south poles. 
        The remaining d**2 vertices are in d circles of latitute, each with
        d vertices in them. 
    '''

    hsteps = density # 'horizontal' steps (in each circle of latitude)
    vsteps = density # 'vertical' steps (number of circles of latitude, 
                     #                   excluding north and south poles)

    vs = [(0., 0., 1.), (0., 0., -1)] # top and bottom nodes
    fs = []

    # z values for each ring (excluding top and bottom), equally spaced
    zs = [-1. + 2 * (1. / (vsteps + 1)) * (i + 1) for i in xrange(vsteps)]

    # angles for x and y
    alphastep = 2. * math.pi / hsteps
    alphas = [float(i) * alphastep for i in xrange(hsteps)]

    # generate coordinates, one ring at a time
    for vi in xrange(vsteps):
        z = math.sin(zs[vi] * math.pi * .5) # z coordinate
        scz = (1 - z * z) ** .5 # scaling for z

        alphaplus = vi * alphastep * .5 # each next ring is offset by half
                                        # of the length of a triangle

        # x and y coordinates
        xs = map(lambda x:scz * math.cos(x + alphaplus), alphas)
        ys = map(lambda x:scz * math.sin(x + alphaplus), alphas)

        vs.extend((xs[i], ys[i], z) for i in xrange(hsteps))

    # set topology, one ring at a time
    top = [1] * hsteps
    cur = [2 + i for i in xrange(hsteps)]

    for vi in xrange(vsteps):
        bot = ([0] * hsteps if vi == vsteps - 1
                            else map(lambda x:x + hsteps, cur))

        for hi in xrange(hsteps):
            left = cur[hi]
            right = cur[(hi + 1) % hsteps]

            fs.append((left, right, top[hi]))
            fs.append((right, left, bot[hi]))

        top, cur = cur, [bot[-1]] + bot[:-1]

    return Surface(np.array(vs), np.array(fs))

def generate_plane(x00, x01, x10, n01, n10):
    '''
    Generates a plane.
    
    Parameters
    ----------
    x00: np.array with 3 values
        origin
    x01: np.array with 3 values
        vector indicating first direction of plane
    x10: np.array with 3 values
        vector indicating second direction of plane
    n01: int
        number of points in first direction
    n10: int
        number of points in second direction
    
    Returns
    -------
    surf.Surface
        surface with n01*n10 nodes and (n01-1)*(n10-1)*2 faces. 
        The (i,j)-th point is at coordinate x01+i*x01+j*x10 and
        is stored as the (i*n10+j)-th vertex.
    '''
    def as_three_vec(v):
        a = np.reshape(np.asarray(v, dtype=np.float), (-1,))
        if len(a) != 3:
            raise ValueError('expected three values for %r' % v)
        return a

    # ensure they are proper vectors
    x00, x01, x10 = map(as_three_vec, (x00, x01, x10))

    vs = np.zeros((n01 * n10, 3))
    fs = np.zeros((2 * (n01 - 1) * (n10 - 1), 3), dtype=np.int)
    for i in xrange(n01):
        for j in xrange(n10):
            vpos = i * n10 + j
            vs[vpos, :] = x00 + i * x01 + j * x10
            if i < n01 - 1 and j < n10 - 1: # not at upper borders
                # make a square pqrs from two triangles
                p = vpos
                q = vpos + 1
                r = vpos + n10
                s = r + 1

                fpos = (i * (n10 - 1) + j) * 2
                fs[fpos, :] = [p, q, r]
                fs[fpos + 1, :] = [s, r, q]

    return Surface(vs, fs)


def read(fn):
    '''General read function for surfaces
    
    For now only supports ascii (as used in AFNI's SUMA) and freesurfer formats
    '''
    if fn.endswith('.asc'):
        from mvpa2.support.nibabel import surf_fs_asc
        return surf_fs_asc.read(fn)
    else:
        import nibabel.freesurfer.io as fsio
        coords, faces = fsio.read_geometry(fn)
        return Surface(coords, faces)

def write(fn, s, overwrite=False):
    '''General write function for surfaces
    
    For now only supports ascii (as used in AFNI's SUMA)
    '''
    if fn.endswith('.asc'):
        from mvpa2.support.nibabel import surf_fs_asc
        surf_fs_asc.write(fn, s, overwrite=overwrite)
    else:
        raise ValueError("Not implemented (based on extension): %r" % fn)

def from_any(s):
    if s is None or isinstance(s, Surface):
        return s
    elif isinstance(s, basestring):
        return read(s)
    elif type(s) is tuple and len(ts) == 2:
        return Surface(ts[0], ts[1])
    else:
        raise ValueError("Not understood: %r" % s)
