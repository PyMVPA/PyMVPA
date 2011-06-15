# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the PyMVPA package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Implementation of a common processing object (node)."""

__docformat__ = 'restructuredtext'

import time
from mvpa.support import copy

from mvpa.base.dochelpers import _str, _repr, _repr_attrs
from mvpa.base.state import ClassWithCollections, ConditionalAttribute

if __debug__:
    from mvpa.base import debug

class Node(ClassWithCollections):
    """Common processing object.

    A `Node` is an object the processes datasets. It can be called with a
    `Dataset` and returns another dataset with the results. In addition, a node
    can also be used as a generator. Upon calling ``generate()`` with a datasets
    it yields (potentially) multiple result datasets.

    Node have a notion of ``space``. The meaning of this space may vary heavily
    across sub-classes. In general, this is a trigger that tells the node to
    compute and store information about the input data that is "interesting" in
    the context of the corresponding processing in the output dataset.
    """

    calling_time = ConditionalAttribute(enabled=True,
        doc="Time (in seconds) it took to call the node")

    def __init__(self, space=None, postproc=None, **kwargs):
        """
        Parameters
        ----------
        space: str, optional
          Name of the 'processing space'. The actual meaning of this argument
          heavily depends on the sub-class implementation. In general, this is
          a trigger that tells the node to compute and store information about
          the input data that is "interesting" in the context of the
          corresponding processing in the output dataset.
        postproc : Node instance, optional
          Node to perform post-processing of results. This node is applied
          in `__call__()` to perform a final processing step on the to be
          result dataset. If None, nothing is done.
        """
        ClassWithCollections.__init__(self, **kwargs)
        self.set_space(space)
        self.set_postproc(postproc)


    def __call__(self, ds):
        """
        The default implementation calls ``_precall()``, ``_call()``, and
        finally returns the output of ``_postcall()``.

        Parameters
        ----------
        ds: Dataset
          Input dataset.

        Returns
        -------
        Dataset
        """
        t0 = time.time()                # record the time when call initiated

        self._precall(ds)
        result = self._call(ds)
        result = self._postcall(ds, result)

        self.ca.calling_time = time.time() - t0 # set the calling_time
        return result


    def _precall(self, ds):
        """Preprocessing of data

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original input dataset.

        Returns
        -------
        Dataset
        """
        return ds


    def _call(self, ds):
        raise NotImplementedError


    def _postcall(self, ds, result):
        """Postprocessing of results.

        By default, does nothing.

        Parameters
        ----------
        ds: Dataset
          Original input dataset.
        result: Dataset
          Preliminary result dataset (as produced by ``_call()``).

        Returns
        -------
        Dataset
        """
        if not self.__postproc is None:
            if __debug__:
                debug("NO",
                      "Applying post-processing node %s", (self.__postproc,))
            result = self.__postproc(result)

        return result


    def generate(self, ds):
        """Yield processing results.

        This methods causes the node to behave like a generator. By default it
        simply yields a single result of its processing -- identical to the
        output of calling the node with a dataset. Subclasses might implement
        generators that yield multiple results.

        Parameters
        ----------
        ds: Dataset
          Input dataset

        Returns
        -------
        generator
          the generator yields the result of the processing.
        """
        yield self(ds)


    def get_space(self):
        """Query the processing space name of this node."""
        return self.__space


    def set_space(self, name):
        """Set the processing space name of this node."""
        self.__space = name


    def get_postproc(self):
        """Returns the post-processing node or None."""
        return self.__postproc


    def set_postproc(self, node):
        """Assigns a post-processing node

        Set to `None` to disable postprocessing.
        """
        self.__postproc = node


    def __str__(self):
        return _str(self)


    def __repr__(self, prefixes=[]):
        return super(Node, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['space', 'postproc']))

    space = property(get_space, set_space,
                     doc="Processing space name of this node")

    postproc = property(get_postproc, set_postproc,
                        doc="Node to perform post-processing of results")


class ChainNode(Node):
    """Chain of nodes.

    This class allows to concatenate a list of nodes into a processing chain.
    When called with a dataset, it is sequentially fed through a nodes in the
    chain. A ChainNode may also be used as a generator. In this case, all
    nodes in the chain are treated as generators too, and the ChainNode
    behaves as a single big generator that recursively calls all embedded
    generators and yield the results.

    A ChainNode behaves similar to a list container: Nodes can be appended,
    and the chain can be sliced like a list, etc ...
    """
    def __init__(self, nodes, **kwargs):
        """
        Parameters
        ----------
        nodes: list
          Node instances.
        """
        if not len(nodes):
            raise ValueError("%s needs at least one embedded node."
                             % self.__class__.__name__)
        Node.__init__(self, **kwargs)
        self._nodes = nodes


    def __copy__(self):
        # XXX how do we safely and exhaustively copy a node?
        return self.__class__([copy.copy(n) for n in self])


    def _call(self, ds):
        mp = ds
        for i, n in enumerate(self):
            if __debug__:
                debug('MAP', "%s: input (%s) -> node (%i/%i): '%s'",
                      (self.__class__.__name__, mp.shape,
                       i + 1, len(self),
                       n))
            mp = n(mp)
        if __debug__:
            debug('MAP', "%s: output (%s)", (self.__class__.__name__, mp.shape))
        return mp


    def generate(self, ds, startnode=0):
        """
        Parameters
        ----------
        ds: Dataset
          To be processed dataset
        startnode: int
          First node in the chain that shall be considered. This argument is
          mostly useful for internal optimization.
        """
        first_node = self[startnode]
        if __debug__:
            debug('MAP', "%s: input (%s) -> generator (%i/%i): '%s'",
                  (self.__class__.__name__, ds.shape,
                   startnode + 1, len(self), first_node))
        # let the first node generator as many datasets as it wants
        for gds in first_node.generate(ds):
            if startnode == len(self) - 1:
                # if this is already the last node yield the result
                yield gds
            else:
                # otherwise feed them through the rest of the chain
                for rgds in self.generate(gds, startnode=startnode + 1):
                    yield rgds


    #
    # Behave as a container
    #
    def append(self, node):
        """Append a node to the chain."""
        # XXX and if a node is a ChainMapper itself -- should we just
        # may be loop and add all the entries?
        self._nodes.append(node)


    def __len__(self):
        return len(self._nodes)


    def __iter__(self):
        for n in self._nodes:
            yield n


    def __reversed__(self):
        return reversed(self._nodes)


    def __getitem__(self, key):
        # if just one is requested return just one, otherwise return a
        # NodeChain again
        if isinstance(key, int):
            return self._nodes[key]
        else:
            # operate on shallow copy of self
            sliced = copy.copy(self)
            sliced._nodes = self._nodes[key]
            return sliced


    def __repr__(self, prefixes=[]):
        return super(ChainNode, self).__repr__(
            prefixes=prefixes
            + _repr_attrs(self, ['nodes']))


    def __str__(self):
        return _str(self, '-'.join([str(n) for n in self]))

    nodes = property(fget=lambda self:self._nodes)
