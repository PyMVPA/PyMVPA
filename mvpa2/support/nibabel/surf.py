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

_COORD_EPS = 1e-14 # maximum allowed difference between coordinates
                   # in order to be considered equal

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
    def __init__(self, v, f=None, check=True):
        # set vertices
        v = np.asarray(v)
        if len(v.shape) != 2 or v.shape[1] != 3:
            raise ValueError("Expected Px3 array for coordinates")
        self._v = v

        # set faces
        if f is None:
            f = np.zeros((0, 3), dtype=np.int)
        else:
            f = np.asarray(f)
            if len(f.shape) != 2 or f.shape[1] != 3:
                raise ValueError("Expected Qx3 array for faces")
        self._f = f

        self._nv = v.shape[0]
        self._nf = f.shape[0]

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
                                    len(set(range(self._nv)) - set(unqf))))


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
            n2f = dict()
            for i in xrange(self._nf):
                fi = self._f[i]
                for j in xrange(3):
                    p = fi[j]
                    if not p in n2f:
                        n2f[p] = []
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
            nbrs = dict()
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
                    if not p in nbrs:
                        nbrs[p] = dict()
                    if not q in nbrs:
                        nbrs[q] = dict()

                    nbrs[q][p] = dist
                    nbrs[p][q] = dist

            self._nbrs = nbrs

        return dict(self._nbrs) # make a copy

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

        shortmetric = metric.lower()[0] # only take first letter - for now

        if shortmetric == 'e':
            ds = self.euclidean_distance(src)
            c = dict((nd, d) for (nd, d) in zip(xrange(self._nv), ds)
                                            if d <= radius)

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

                if maxdistance is not None and dnew > maxdistance:
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

                if maxdistance is not None and dnew > maxdistance:
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
        if to_visit is None or len(to_visit) == 0:
            raise ValueError("Cannot operate on empty list")

        src = to_visit[0]
        if not src in np.arange(self.nvertices):
            raise ValueError("%d is not a valid node index" % src)
        if len(to_visit) == 1:
            return []

        trg = to_visit[1]
        if not trg in np.arange(self.nvertices):
            raise ValueError("%d is not a valid node index" % trg)

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

        if i != trg:
            raise ValueError('Node %d could not be reached from %d' %
                                                        (trg, src))

        pth = [fdist[i]]

        # recursion to find remaining paths (if any)
        pth.extend(self.dijkstra_shortest_path_visiting(to_visit[1:]))
        return pth



    def euclidean_distance(self, src, trg=None):
        '''Computes Euclidean distance from one node to other nodes

        Parameters
        ----------
        src : int or numpy.ndarray
            Index of center (source) node, or a 1x3 array with coordinates
            of the center (source) node.
        trg : int
            Target node(s) to which the distance is computed.
            If 'trg is None' then distances to all nodes are computed

        Returns:
        --------
        n2d : dict
            A dict "n2d" so that n2d[j]=d" is the distance "d" from node
            "src" to node "j".
        '''

        if type(src) is tuple and len(src) == 3:
            src = np.asarray(src)

        if isinstance(src, np.ndarray):
            if src.shape not in ((1, 3), (3,), (3, 1)):
                raise ValueError("Illegal shape: should have 3 elements")

            src_coord = src if src.shape == (1, 3) else np.reshape(src, (1, 3))
        else:
            src_coord = self._v[src]


        if trg is None:
            delta = self._v - src_coord
        else:
            delta = self._v[trg] - src_coord

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

        use_mask = node_mask_indices is not None
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

    def nodes_on_border(self, node_indices=None):
        '''Determines which nodes are on the border of the surface

        Parameters
        ----------
        node_indices: np.ndarray or None
            Vector with node indices for which their bordership status is to
            be deteremined. None means all node indices

        Returns
        -------
        on_border: np.ndarray
            Boolean array of shape (len(node_indices),). A node i is
            considered on the border if there is a face that contains node i
            and another node j so that no other face contains both i and j.
            In other words a node i is *not* on the border if there is a path
            of nodes p1,...pN so that N>1, p1==pN, pj!=pk if j!=k<N, and
            each node pk (and no other node) is a neighbor of node i.
        '''

        if node_indices is None:
            node_indices = np.arange(self.nvertices)

        if not isinstance(node_indices, np.ndarray):
            node_indices = np.asarray(node_indices)[np.newaxis]

        if len(node_indices.shape) != 1:
            raise ValueError("Only supported for vectors")

        n = len(node_indices)
        on_border = np.zeros((n,), dtype=np.bool_) # allocate space for output

        n2f = self.node2faces
        f = self.faces

        def except_(vs, x):
            return filter(lambda y:y != x, vs)

        for i, node_index in enumerate(node_indices):
            face_indices = n2f[node_index]
            nf = len(face_indices)

            # node indices of neighbouring nodes (one for each face containing
            # node with index node_index)
            fs = [except_(f[fi], node_index) for fi in face_indices]

            a = np.asarray(fs)
            if a.size == 0:
                continue

            # initial position and value
            ipos, jpos = 0, 0
            a_init = a[ipos, jpos]

            for j in xrange(nf):
                # go over the faces that contain node_index
                # for each row take the other value, and try to match
                # it to another face
                jpos_ = (jpos + 1) % 2
                target = a[ipos, jpos_]
                a[ipos, jpos_] = -1 # is visited

                ijpos = np.nonzero(a == target)
                if len(ijpos[0]) != 1:
                    #
                    on_border[i] = True
                    break
                ipos, jpos = ijpos[0], ijpos[1]

            on_border[i] = on_border[i] or target != a_init

        return on_border


    def nodes_on_border_paths(self):
        '''Find paths of nodes on the border

        Returns
        -------
        paths: list of lists
            paths[i]=[k_0,...k_N] means that there is path of N+1 edges
            [(k_0,k_1),(k_1,...,k_N),(k_N,k_0)] where each k_i is on the
            border of the surface

        '''
        border_mask = self.nodes_on_border()
        faces = self.faces
        nbrs = self.neighbors
        border_nodes = set(np.nonzero(border_mask)[0])
        if not len(border_nodes):
            return []

        # for each edge, see which is the next edge
        # in the same triangle (clock-wise)
        edge2next = dict()
        for i in xrange(self.nfaces):
            for j in xrange(3):
                p, q, r = faces[i]

                # make edges
                pp, qq, rr = (p, q), (q, r), (r, p)

                edge2next[pp] = qq
                edge2next[qq] = rr
                edge2next[rr] = pp

        # mapping from edge to face
        e2f = self.edge2face

        pths = [] # space for output
        while border_nodes:
            b0 = border_nodes.pop() # select a random node on the border
            ns = [b for b in nbrs[b0] if b in border_nodes]
            if not ns:
                # not a proper node - no neighbors - so skip
                continue

            # find an edge on the border
            for n in ns:
                edge = (b0, n)
                if edge in edge2next:
                    break

            if not edge in edge2next:
                # this should not happen really
                raise ValueError("no edge on border found")

            # start a path
            pth = []
            pths.append(pth)
            while True:
                p, q = edge2next[edge]

                if (q, p) in e2f:
                    # node q is 'inside' - not on the border
                    # continue looking
                    edge = (q, p)
                else:
                    # on the border, so swap
                    edge = (p, q)
                    pth.append(p) # p is on the border
                    if p in border_nodes:
                        border_nodes.remove(p)
                    else:
                        # we made a tour and back to the starting point
                        break

        return pths




    def pairwise_near_nodes(self, max_distance=None, src=None, trg=None):
        '''Finds the distances between pairs of nodes

        Parameters
        ----------
        max_distance: None or float
            maximum distance (None: no maximum distance)
        src: array of int or None
            source indices
        trg: array of int or None
            target indices

        Returns
        -------
        source_target2distance: dict
            A dictionary so that source_target2distance[i,j]=d means that the
            Euclidean distance between nodes i and j is d, where i in src
            and j in trg.

        Notes
        -----
        If src and trg are both None, then this function checks if the surface
        has two components; if so they are taken as source and target. A use
        case for this behaviour is a surface consisting of two hemispheres
        '''

        if src is None and trg is None:
            components = self.connected_components()
            if len(components) != 2:
                raise ValueError("Empty src and trg: requires two components")
            src, trg = (np.asarray([i for i in c]) for c in components)

        v = self.vertices
        if max_distance is not None:
            # hopefully we can reduce the number of vertices significantly
            # if src and trg can be seperated easily (as in the case of
            # two hemispheres).

            # vector connecting centers of mass of src and trg
            n = np.mean(v[src], 0) - np.mean(v[trg], 0)

            # normalize
            n /= np.sum(n ** 2) ** .5

            # compute projection on normal
            ps = self.project_vertices(n, v[src])
            pt = self.project_vertices(n, v[trg])

            def remove_far(s, t, ps, pt, max_distance=max_distance):
                keep_idxs = np.arange(len(s))
                for sign in (-1, 1):
                    far_idxs = np.nonzero(sign * ps[keep_idxs] + \
                                            max_distance < min(sign * pt))[0]

                    keep_idxs = np.setdiff1d(keep_idxs, far_idxs)

                return s[keep_idxs]

            src, trg = remove_far(src, trg, ps, pt), \
                                remove_far(trg, src, pt, ps)

        st2d = dict() # source-target pair to distance
        for s in src:
            ds = self.euclidean_distance(s, trg)
            for t, d in zip(trg, ds):
                if max_distance is None or d <= max_distance:
                    st2d[(s, t)] = d


        return st2d

    def project_vertices(self, n, v=None):
        '''Projects vertex coordinates onto a vector

        Parameters
        ----------
        n: np.ndarray
            Vector with 3 elements
        v: np.ndarray or None
            coordinates to be projected. If None then the vertices of the
            current instance are used.

        Returns
        -------
        p: np.ndarray
            Vector with coordinates projected onto n
        '''

        if not isinstance(n, np.ndarray):
            n = np.asarray(n)
        if n.shape != (3,):
            raise ValueError("Expected vector with 3 elements, found %s" % ((n.shape,)))

        if v is None:
            v = self.vertices

        return np.dot(v, n)

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
        funq = list(set.union(*[set(n2f[vidx]) for vidx in vidxs]))

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

    def __repr__(self, prefixes=None):
        if prefixes is None:
            prefixes = []
        prefixes_ = ['v=%r' % self._v, 'f=%r' % self._f] + prefixes
        return "%s(%s)" % (self.__class__.__name__, ', '.join(prefixes_))

    def __str__(self):
        # helper function to print coordinates. f should be np.min or np.max
        func_coord2str = lambda f: '%.3f %.3f %.3f' % tuple(
                                                        f(self.vertices, 0))

        return '%s(%d vertices (range %s ... %s), %d faces)' % (
                        self.__class__.__name__,
                        self.nvertices,
                        func_coord2str(np.min),
                        func_coord2str(np.max),
                        self.nfaces)


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        sv = self.vertices
        ov = other.vertices

        # must be identical where NaNs occur
        if np.any(np.logical_xor(np.isnan(sv), np.isnan(ov))):
            return False

        # difference in vertices
        v = np.abs(self.vertices - other.vertices)

        return (np.all(np.logical_or(v < _COORD_EPS, np.isnan(v)))
                and np.all(self.faces == other.faces))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce__(self):
        # these are lazily computed on the first call to e.g. node2faces
        lazy_keys = ('_n2f', '_f2el', '_v2ael', '_e2f', '_nbrs')
        lazy_dict = dict()
        # TODO: add in efficient way to translate these dictionaries
        #       to something like a numpy array, and implement the 
        #       translation back. Types for these dicts:
        #       _n2f: int -> [int]
        #       _f2el: array
        #       _v2ael: array
        #       _e2f: (int,int) -> int
        #       _nbrs: int -> (int -> float)
        #       
        # For now this this functionaltiy is switched off,
        # because pickling it (also with hdf5) takes a long time
        #for lazy_key in lazy_keys:
        #    if lazy_key in self.__dict__:
        #        lazy_dict[lazy_key] = self.__dict__[lazy_key]


        return (self.__class__, (self._v, self._f), lazy_dict)

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

        return self._v.shape == other._v.shape and np.array_equal(self._f, other._f)

    def __add__(self, other):
        '''coordinate-wise addition of two surfaces with the same topology'''
        if isinstance(other, Surface):
            if not self.same_topology(other):
                raise Exception("Different topologies - cannot add")
            vother = other.vertices
        else:
            vother = other

        return Surface(v=self.vertices + vother, f=self.faces, check=False)

    def __mul__(self, other):
        '''coordinate-wise scaling'''
        return Surface(v=self._v * other, f=self.faces, check=False)

    def __neg__(self, other):
        '''coordinate-wise inversion with respect to addition'''
        return Surface(v=-self.vertices, f=self.faces, check=False)

    def __sub__(self, other):
        '''coordiante-wise subtraction'''
        return self +(-other)

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


    def split_by_connected_components(self):
        '''Splits a surface by its connected components

        Returns
        -------
        splits: list of surf.Surface
            A list of all surfaces that make up the original surface,
            split when they are not connected to each other.
            (If all nodes in the original surface are connected
            then a list is returned with a single surface that is
            identical to the input).
            The output is sorted by the number of vertices.

        '''
        components = self.connected_components()
        n2f = self.node2faces

        n = len(components)
        splits = []

        face_mask = np.zeros((self.nfaces,), dtype=np.False_)
        for i, component in enumerate(components):
            face_mask[:] = False

            node_idxs = np.asarray(list(component))
            for node_idx in node_idxs:
                face_mask[n2f[node_idx]] = True

            nodes = self.vertices[node_idxs, :]

            face_idxs = np.nonzero(face_mask)[0]
            unq, unq_inv = np.unique(self.faces[face_idxs], False, True)
            faces = np.reshape(unq_inv, (-1, 3))

            s = Surface(nodes, faces)
            splits.append(s)

        splits.sort(key=lambda x:x.nvertices)
        return splits



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
            if all(d[np.logical_not(np.isnan(d))] < epsilon):
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

            if epsilon is not None and mind > epsilon:
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

        if master is not None:
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

            if all(d[np.logical_not(np.isnan(d))] < epsilon):
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
            i_xyz = x[i, :]
            if np.any(np.isnan(i_xyz)):
                continue

            idxs = np.asarray(x_tuple2near_indices[x_tuple])

            ds = np.sum((x[i, :] - y[idxs, :]) ** 2, axis=1)

            not_nan_idxs = np.nonzero(np.logical_not(np.isnan(ds)))[0]
            if len(not_nan_idxs) == 0:
                raise ValueError("Empty sequence: is center %d (%r)"
                                 " illegal?" % (i, (x[i],)))

            minpos = not_nan_idxs[np.argmin(ds[not_nan_idxs])]

            mind = ds[minpos] ** .5

            if epsilon is not None and not (mind < epsilon):
                raise ValueError("Not found for node %i: %s > %s" %
                                        (i, mind, epsilon))

            mapping[i] = idxs[minpos]

        return mapping

    def vonoroi_map_to_high_resolution_surf(self, highres_surf,
                                highres_indices=None, epsilon=.001,
                                    accept_only_icosahedron=False):
        '''
        Computes a Vonoroi mapping for the current (low-res) surface

        Parameters
        ----------
        highres_surf: Surface
            High-resolution surface.
        highres_indices: np.ndarray
            List of indices in high-res surface that have to be mapped.
        epsilon: float
            maximum margin (distance) between nodes mapped from low to
            high resolution surface. Default None, which implies .001.
        accept_only_icosahedron: bool
            if True, then this function raises an error if the number of
            nodes does not match those which would be expected from
            MapIcosahedorn.

        Returns
        -------
        high2high_in_low: dict
            A mapping so that high2high_in_low[high_idx]=(high_in_low_idx,d)
            means that the node on the high-res surface indexed by high_idx is
            nearest (in a Dijsktra distance sense) distance d to the node on the
            high-res surface high_in_low_idx that has a corresponding
            node on the low-res surface
        '''

        # the set of indidces that will serve as keys in high2high_in_low
        if highres_indices is None:
            highres_indices = np.arange(highres_surf.nvertices)
        highres_indices = set(highres_indices)


        low2high = self.map_to_high_resolution_surf(highres_surf, epsilon,
                                                  accept_only_icosahedron)



        # reverse mapping, only containing nodes that are both in
        # highres_indices and have a partner in self (lowres)
        high2low = dict((v, k) for k, v in low2high.iteritems()
                                if v in highres_indices)

        # node indices in high-res surface that have a mapping
        # and thus are acceptable
        highres_center_set = set(high2low)


        # starting value for radius
        radius = np.mean(self.average_node_edge_length)
        max_radius = radius * 10000.

        # set of node indices of low-res surface
        lowres_node_set = set(xrange(self.nvertices))

        # space for output
        high2high_in_low = dict()

        # continue increasing radius until all high-res nodes
        # have been mapped to a low-res node
        while set(high2high_in_low) != highres_indices:
            for highres_index in highres_indices:
                if highres_index in high2high_in_low:
                    # already has a low-res node mapped to it
                    continue

                # compute distances in high-res surface
                ds = highres_surf.dijkstra_distance(highres_index, radius)

                common = set.intersection(set(ds), highres_center_set)

                if len(common):
                    # keep only distances to allowed nodes
                    small_ds = dict((k, v) for k, v in ds.iteritems() if k in common)

                    # find nearest node
                    nearest_node_highres = min(small_ds, key=small_ds.get)
                    d = small_ds[nearest_node_highres]

                    # store the result
                    high2high_in_low[highres_index] = (nearest_node_highres, d)

            radius *= 2

            if radius > max_radius:
                # safety mechanism to avoid endless loop
                raise RuntimeError("Radius increased to %d - too big" % radius)


        return high2high_in_low


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

    @property
    def face_normals(self):
        if not hasattr(self, '_face_normals'):
            f = self.faces
            v = self.vertices

             # consider three sides of each triangle
            a = v[f[:, 0]]
            b = v[f[:, 1]]
            c = v[f[:, 2]]

            # vectors of two sides
            ab = a - b
            ac = a - c

            abXac = np.cross(ab, ac)
            n = normalized(abXac)

            vw = n.view()
            vw.flags.writeable = False

            self._face_normals = vw

        return self._face_normals

    @property
    def node_normals(self):
        if not hasattr(self, '_node_normals'):
            f = self.faces
            v = self.vertices
            n = self.nfaces

            f_nrm = self.face_normals

            v_sum = np.zeros(v.shape, dtype=v.dtype)
            for i in xrange(3):
                for j in xrange(n):
                    v_sum[f[j, i]] += f_nrm[j]

            v_nrm = normalized(v_sum)

            vw = v_nrm.view()
            vw.flags.writeable = False

            self._node_normals = vw

        return self._node_normals

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

            return k, v

        for i in xrange(nf):
            p, q, r = f[i]

            pk, pv = index_component(p)
            qk, qv = index_component(q)
            rk, rv = index_component(r)

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


    def write(self, fn):
        write(fn, self)

def reposition_hemisphere_pairs(surf_left, surf_right, facing_side,
                          min_distance=10.):
    '''moves and rotates pairs of hemispheres so that they are facing each
    other on one side, good for visualization. It is assumed that the input
    surfaces were generated by FreeSurfer's recon-all.

    Parameters
    ----------
    surf_left: surf.Surface
        surface of left hemisphere
    surf_right: surf.Surface
        surface of right hemisphere
    facing_side: str
        determines on which sides the surfaces should be facing each other.
        'm'=medial,'i'=inferior, 's'=superior, 'a'=anterior,'p'=posterior


    '''
    facing_side = facing_side[0].lower()

    mn, mx = np.min, np.max
    #min=-1, max=1
    side2dimsigns = dict(m=(0, -1), i=(1, 1), s=(1, -1), a=(2, 1), p=(2, -1))

    dim, rotatesign = side2dimsigns[facing_side]
    if dim == 0:
        rotate_axis = None
    else:
        rotate_axis = dim #1+((dim+1) % 2)
        rotate_angle = 90

    surfs = [surf_left, surf_right]
    nsurfs = len(surfs)
    hemisigns = [1, -1]
    if rotate_axis is not None:
        theta = [0] * 3

        for i in xrange(nsurfs):
            theta[rotate_axis] = rotate_angle * hemisigns[i] * rotatesign
            surfs[i] = surfs[i].rotate(theta, unit='deg')


    for i in xrange(nsurfs):
        hemisign = hemisigns[i]
        sign = rotatesign * hemisign
        coords = surfs[i].vertices

        xtreme = np.min(coords[:, 0] * -hemisign)

        # sometimes the surfaces are not properly aligned along x and y
        # so fix it by moving by center of mass values along x and y

        delta = -np.reshape(surfs[i].center_of_mass, (1, 3))
        delta[0, 0] = hemisign * (xtreme - min_distance * .5)
        surfs[i] = surfs[i] + delta # make an implicit copy

    return tuple(surfs)




def get_sphere_left_right_mapping(surf_left, surf_right, eps=.001):
    '''finds the mapping from left to right hemisphere and vice versa
    (the mapping is symmetric)

    this only works on sphere.reg.asc files made with AFNI/SUMA's mapicosehedron'''

    if not surf_left.same_topology(surf_right):
        raise ValueError('topology mismatch')

    nv = surf_left.nvertices

    # to swap right surface along x-axis (i.e. mirror along x=0 plane)
    swapLR = np.array([[-1, 1, 1]])

    vL, vR = surf_left.vertices, surf_right.vertices * swapLR
    nL, nR = surf_left.neighbors, surf_right.neighbors


    # flip along x-axis
    #vR = vR * np.asarray([[-1., 1., 1.]])

    def find_nearest(src_coords, trgs_coords, eps=eps):
        # finds the index of the nearest node in trgs_coords to src_coords.
        # if the distance is more than eps, an exception is thrown
        d2 = np.sum((src_coords - trgs_coords) ** 2, 1)
        nearest = np.argmin(d2)
        if d2[nearest] > eps ** 2:
            raise ValueError('eps too big: %r > %r' % (d2[nearest] ** .5, eps))
        return nearest

    # get a (random) starting point
    pivotL = 0
    pivotR = find_nearest(vL[pivotL, :], vR)

    # mapping from left to right
    l2r = {pivotL:pivotR}
    to_visit = nL[pivotL].keys()

    # for each node (in the left hemispehre) still to visit, keep track of its
    # 'parent'
    to_visit2source = dict(zip(to_visit, [pivotL] * len(to_visit)))

    # invariants:
    # 1) if to_visit2source[v]==s, then s in l2r.keys()
    # 2) if to_visit2source[v]==s, then v in nL[s].keys()

    while to_visit2source:
        # find the corresponding node in right hemi for pivotL,
        # using sourceL as a neighbor which a corresponding node
        # on the other hemisphere is already known
        pivotL, sourceL = to_visit2source.popitem()

        # get the corresponding node of sourceL on the other hemisphere
        sourceR = l2r[sourceL]

        # of all the neighbors of sourceR, one of them should be
        # corresponding to pivotL
        nbr_surf_right = nR[sourceR].keys()
        nearestR = nbr_surf_right[find_nearest(vL[pivotL, :],
                                               vR[nbr_surf_right, :])]

        # store result
        l2r[pivotL] = nearestR

        # add new neighbors to to_visit2source; but not those that
        # have already a corresponding node on the other hemisphere
        for nbrL in nL[pivotL].keys():
            if not nbrL in l2r:
                to_visit2source[nbrL] = pivotL

    # store values in an array - this is easier for indexing
    l2r_arr = np.zeros((nv,), dtype=np.int32)
    for p, q in l2r.iteritems():
        l2r_arr[p] = q

    v_range = np.arange(nv)

    # final check: make sure it's a bijection
    if not np.all(l2r_arr[l2r_arr] == v_range):
        raise ValueError('Not found a bijection - this should not happen')

    return l2r_arr




def normalized(v):
    '''Normalizes vectors

    Parameters
    ==========
    v: np.ndarray
        PxQ matrix for P vectors that are all Q-dimensional

    Returns
    =======
    n: np.ndarray
        P-valued vector with the norm for each vector
    '''

    v_norm = np.sqrt(np.sum(v ** 2, 1))
    return v / np.reshape(v_norm, (-1, 1))



def merge(*surfs):
    if not surfs:
        return None
    s0 = surfs[0]
    return s0.merge(*surfs[1:])

def generate_cube():
    '''
    Generates a cube with sides 2 centered at the origin.

    Returns
    -------
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
    trias = [(0, 1, 3), (0, 3, 2), (1, 0, 5), (5, 0, 4),
           (3, 1, 5), (3, 5, 7), (3, 7, 6), (3, 6, 2),
           (2, 6, 0), (0, 6, 4), (5, 4, 6), (5, 6, 7)]


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

def generate_bar(start, stop, radius, poly=10):
    '''Generates a bar-like surface

    Parameters
    ----------
    start: np.ndarray
        3-elemtent vector indicating top part of the bar
    stop: np.ndarray
        3-elemtent vector indicating bottom side of the bar
    radius: float
        radius of the bar
    poly: int
        the top and bottom part will be a regular polygon.

    Returns
    -------
    bar: surf.Surface
        A surface with poly*2+2 vertices and poly*4 faces

    Example
    -------
    generate_bar((0,0,0),(0,0,177.6),14.1,4)

    This generates a surface resembling the new One World Trade center, New York
    '''

    start = np.asarray(start)
    stop = np.asarray(stop)

    nv = poly * 2 + 2
    delta = start - stop
    delta_n = delta / np.sqrt(np.sum(delta ** 2))

    # get a normal vector
    # make sure that we don't use zero values
    i = np.argsort(np.abs(delta_n))
    vec_x = np.zeros(3)
    vec_x[i] = delta_n[i[[0, 2, 1]]] * np.asarray((0, -1, 1))
    vec_y = np.cross(delta_n, vec_x)

    coords = np.zeros((nv, 3))
    sc = 2 * np.pi / poly # angle scaling
    alpha = np.arange(poly) * sc # for top art
    beta = alpha + sc / 2

    # first and last node are top and bottom.
    # nodes in between are the edges at top and bottom
    coords[0, :] = start
    dtop = np.cos(alpha)[np.newaxis].T * vec_x[np.newaxis] + \
                        np.sin(alpha)[np.newaxis].T * vec_y[np.newaxis]
    dbot = np.cos(beta)[np.newaxis].T * vec_x[np.newaxis] + \
                        np.sin(beta)[np.newaxis].T * vec_y[np.newaxis]

    coords[1:-1:2, :] = dtop * radius + start
    coords[2::2, :] = dbot * radius + stop
    coords[-1, :] = stop

    # set up faces
    nf = poly * 4
    faces = np.zeros((nf, 3), dtype=np.int_)
    for i in xrange(poly):
        j = i * 2
        faces[j + 0, :] = (j + 1, j + 2, j + 3) # top part
        faces[j + 1, :] = (j + 2, j + 4, j + 3) # side with top
        faces[j + 2 * poly, :] = (j + 3, 0, j + 1) # side with bottom
        faces[j + 2 * poly + 1, :] = (j + 2, nv - 1, j + 4) # bottom part

    nrm = lambda x: (x - 1) % (2 * poly) + 1
    faces[:2 * poly, :] = nrm(faces[:2 * poly, :])
    faces[2 * poly:, 0] = nrm(faces[2 * poly:, 0])
    faces[2 * poly:, 2] = nrm(faces[2 * poly:, 2])

    s = Surface(coords, faces)

    return s


def read(fn):
    '''General read function for surfaces

    Parameters
    ----------
    fn: str
        Surface filename. The extension determines how the file is read as
        follows. '.asc', FreeSurfer ASCII format; '.coord'; Caret, '.gii',
        GIFTI; anything else: FreeSurfer geometry.

    Returns
    -------
    surf_: surf.Surface
        Surface object

    '''
    if fn.endswith('.asc'):
        from mvpa2.support.nibabel import surf_fs_asc
        return surf_fs_asc.read(fn)
    elif fn.endswith('.coord'):
        from mvpa2.support.nibabel import surf_caret
        return surf_caret.read(fn)
    elif fn.endswith('.gii'):
        # XXX require .surf.gii? Not for now - but may want to change
        from mvpa2.support.nibabel import surf_gifti
        return surf_gifti.read(fn)
    else:
        import nibabel.freesurfer.io as fsio
        coords, faces = fsio.read_geometry(fn)
        return Surface(coords, faces)

def write(fn, s, overwrite=True):
    '''General write function for surfaces

    Parameters
    ----------
    fn: str
        Surface filename. The extension determines how the file is written as
        follows. '.asc', FreeSurfer ASCII format; '.gii', GIFTI.
        Other formats are not supported.
    '''
    if fn.endswith('.asc'):
        from mvpa2.support.nibabel import surf_fs_asc
        surf_fs_asc.write(fn, s, overwrite=overwrite)
    elif fn.endswith('.gii'):
        if not fn.endswith('.surf.gii'):
            raise ValueError("GIFTI output requires extension .surf.gii")
        from mvpa2.support.nibabel import surf_gifti
        surf_gifti.write(fn, s, overwrite=overwrite)
    else:
        raise ValueError("Not implemented (based on extension): %r" % fn)

def from_any(s):
    if s is None or isinstance(s, Surface):
        return s
    elif isinstance(s, basestring):
        return read(s)
    elif type(s) is tuple and len(s) == 2:
        return Surface(s[0], s[1])
    else:
        raise ValueError("Not understood: %r" % s)

